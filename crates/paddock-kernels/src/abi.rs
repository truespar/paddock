//! The C ABI every kernel pack exports.
//!
//! Contract v0 is intentionally tiny: one well-known symbol (`paddock_pack_info`)
//! returning a static descriptor. The kernel entry-point tables get added when P1
//! defines the first real kernels - the descriptor's `abi_version` is what lets
//! us evolve that without silent breakage: loader rejects any mismatch, loudly.

/// Bump on any breaking change to the exported structs or symbol contracts.
/// v1: adds the kernel entry-point table (paddock_pack_kernels_v1).
pub const PACK_ABI_VERSION: u32 = 2;

/// Magic value so a random DLL can't masquerade as a pack ("PDKP").
pub const PACK_MAGIC: u32 = 0x504b_4450;

/// Static descriptor a pack exposes via `paddock_pack_info()`.
///
/// repr(C) and primitive-only deliberately: this crosses the FFI boundary and must
/// have an identical layout from any compiler that built the pack.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PackInfo {
    pub magic: u32,
    pub abi_version: u32,
    /// Target architecture tag, e.g. "cuda-sm86\0" - fixed buffer, NUL-terminated ASCII.
    pub arch: [u8; 16],
    /// Pack semantic version (its own release cadence, independent of the engine).
    pub pack_version: [u32; 3],
}

impl PackInfo {
    /// arch tag as &str, trimmed at the first NUL. Returns None on non-UTF8 garbage.
    pub fn arch_str(&self) -> Option<&str> {
        let end = self
            .arch
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.arch.len());
        std::str::from_utf8(&self.arch[..end]).ok()
    }
}

/// Symbol name packs must export: `extern "C" fn() -> *const PackInfo`.
pub const PACK_INFO_SYMBOL: &[u8] = b"paddock_pack_info\0";

/// Symbol for the v1 kernel table: `extern "C" fn() -> *const KernelTableV1`.
pub const PACK_KERNELS_V1_SYMBOL: &[u8] = b"paddock_pack_kernels_v1\0";

/// Return convention for every kernel launcher: 0 = success, otherwise the
/// raw cudaError_t - the engine formats it into a real error message.
pub type KernelStatus = i32;

/// Dequant launcher: `input` = device pointer to packed blocks, `output` =
/// device pointer to f32, `n_blocks` = block count, `stream` = CUstream the
/// engine owns (packs never create streams - scheduling belongs to the engine).
pub type DequantF32Fn = unsafe extern "C" fn(
    input: *const core::ffi::c_void,
    output: *mut core::ffi::c_void,
    n_blocks: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// RMSNorm: out = x * w / sqrt(mean(x²) + eps), one row of n elements.
pub type RmsNormF32Fn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// YaRN NEOX rope in-place over [n_heads, head_dim]; parameter order matches
/// reference::ops::YarnRope::kernel_params so both sides consume identical
/// numbers.
#[allow(clippy::too_many_arguments)]
pub type RopeYarnF32Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    pos: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// In-place softmax over n scores with a sink logit (joins max + denominator,
/// holds no slot).
pub type SoftmaxSinkF32Fn = unsafe extern "C" fn(
    scores: *mut core::ffi::c_void,
    n: u32,
    sink: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// gate = swiglu_oai(gate, up) in-place, n elements.
pub type SwigluOaiF32Fn = unsafe extern "C" fn(
    gate: *mut core::ffi::c_void,
    up: *const core::ffi::c_void,
    n: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// x += y, n elements (residual/bias adds).
pub type AddInplaceF32Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    y: *const core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// x += w * y, n elements (weighted expert accumulation).
pub type ScaleAddF32Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    y: *const core::ffi::c_void,
    w: f32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched multi-head attention for one decode query token: GQA + per-head
/// attention sinks + sliding window, online (flash-style) softmax. Replaces the
/// per-head score-gemv / softmax / output-gemv loop with a single launch.
///
/// q/out: [n_heads*head_dim] (query already RoPE'd). k_cache/v_cache:
/// [_ * kv_dim] rows, head kvh at `p*kv_dim + kvh*head_dim`. sinks: [n_heads].
/// Attends positions `first_pos .. first_pos+n_pos`. head_dim must be a power of 2.
#[allow(clippy::too_many_arguments)]
pub type AttnDecodeF32Fn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k_cache: *const core::ffi::c_void,
    v_cache: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    first_pos: u32,
    n_pos: u32,
    kv_dim: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// MoE top-k router: biased `logits` [n_expert] -> top-k expert ids (u32) and
/// softmax-over-selected weights (f32), on device (no host round-trip).
pub type MoeTopkFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    n_expert: u32,
    k: u32,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused MXFP4-dequant + GEMV for one expert selected by a device index:
/// y[o] = bias[e][o] + Σ_i dequant(W[e][o][i])·x[i], e = idx[slot]. `bias` may be
/// null. Rows are block-aligned (in_dim % 32 == 0).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4GemvIndexedFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    slot: u32,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// x[..n] += w[slot] * y[..n], with the weight read from a device buffer.
pub type ScaleAddDevFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    y: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    slot: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused Q8_0-dequant + GEMV for a dense weight (attention q/k/v/o, router,
/// lm_head): y[o] = bias[o] + Σ_i dequant(W[o][i])·x[i]. `w` is Q8_0
/// [in_dim, out_dim] in GGUF layout (out_dim rows of in_dim contiguous elems);
/// `bias` may be null. Rows are block-aligned (in_dim % 32 == 0). Keeps the
/// weight Q8_0-resident - ~3.8× less bandwidth than dequant-to-f32 then cuBLAS.
pub type Q8_0GemvFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// FlashDecoding partial: one block per (head, split), online softmax over the
/// split's KV slice -> UNnormalized partial output `out_o` [n_heads*n_splits*head_dim]
/// plus per-partial (max, sum) in `out_ml` [n_heads*n_splits*2]. Sink is applied
/// in the combine, not here. head_dim must be a multiple of 32.
#[allow(clippy::too_many_arguments)]
pub type AttnDecodePartialFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k_cache: *const core::ffi::c_void,
    v_cache: *const core::ffi::c_void,
    out_o: *mut core::ffi::c_void,
    out_ml: *mut core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    first_pos: u32,
    n_pos: u32,
    n_splits: u32,
    kv_dim: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// FlashDecoding combine: merge the n_splits partials per head (flash log-sum-exp
/// rule) into `out` [n_heads*head_dim], folding the per-head sink into the
/// denominator. One block per head; head_dim threads.
pub type AttnDecodeCombineFn = unsafe extern "C" fn(
    in_o: *const core::ffi::c_void,
    in_ml: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    n_splits: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused MoE gate+up+swiglu, batched over active experts. grid (ff, n_active):
/// each block drives both projections for expert `idx[slot]` from a single read of
/// `x`, then applies swiglu_oai and writes `out` [n_active*ff]. Replaces the
/// per-expert gate/up GEMV + swiglu chain.
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused MoE down projection + weighted expert mix + residual add. grid (embd):
/// each block sums Σ_slot w[slot]·(down_e·fused[slot] + bias) over active experts
/// and adds it straight into `residual` [embd]. Replaces the per-expert down GEMV
/// + scale_add + the zero-init and final residual add.
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fused: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched Q8_0 GEMM: y[b][o] = bias[o] + Σ_i dequant(W[o][i])·x[b][i] for
/// b in 0..batch. `x` is row-major [batch, in_dim], `y` is [batch, out_dim].
/// The weight row is dequanted once and applied across the batch - the
/// weight-read amortization that makes concurrent decode scale with the batch.
pub type Q8_0GemmFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Quantize an f32 activation `x` [n] to symmetric int8 `q` [n] + per-32-block
/// f32 `scale` [n/32] - the Q8_1-style activation quantization that enables the
/// integer `__dp4a` dot-product path (llama.cpp/mistral.rs method).
pub type QuantizeQ8Fn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dp4a Q8_0 GEMV: y[o] = bias[o] + Σ_b wscale[b]·ascale[b]·Σ dp4a(w_int8, a_int8).
/// `xq`/`xs` are the pre-quantized activation (int8 + scale). Integer dot on the
/// hardware dp4a unit - ~10× fewer instructions than f32 dequant, at the cost of
/// activation-quantization error (perplexity-close, not bit-exact to f32).
pub type Q8_0GemvDp4aFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dp4a MXFP4 GEMV for one device-indexed expert (MoE), against a pre-quantized
/// activation. Nibbles unpack to int8 in-register (`__byte_perm`, no LUT) then
/// integer `__dp4a` - the compute-bound MoE's ~10× lever. Perplexity-close.
#[allow(clippy::too_many_arguments)]
pub type Mxfp4GemvIndexedDp4aFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    slot: u32,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dp4a fused MoE gate+up+swiglu over active experts, against a pre-quantized
/// activation (`xq`/`xs`). Integer __dp4a versions of both projections.
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpDp4aFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dp4a fused MoE down + weighted mix + residual add. The per-expert swiglu
/// output arrives pre-quantized per slot (`fused_q`/`fused_s`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownDp4aFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fused_q: *const core::ffi::c_void,
    fused_s: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched decode attention: grid (n_heads, batch). Block (h, b) attends
/// sequence b's own KV cache (contiguous [batch][max_ctx][kv_dim]) up to its own
/// `positions[b]`. The per-sequence attention continuous batching needs; at batch
/// B it launches n_heads*B blocks (better GPU fill than the batch-1 path).
#[allow(clippy::too_many_arguments)]
pub type AttnDecodeBatchFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    kc: *const core::ffi::c_void,
    vc: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    // per-row KV slot (null -> row index; prefill maps many rows to one slot)
    slots: *const core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_dim: u32,
    swa_window: u32,
    batch: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched rmsnorm: one block per row of x [batch, n], shared norm weight.
pub type RmsNormBatchFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched YaRN rope: one thread per (sequence, head), each at its sequence's
/// own `positions[b]`. x is [batch, n_heads*head_dim].
#[allow(clippy::too_many_arguments)]
pub type RopeYarnBatchFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched KV append: scatter each sequence's kv row into its own cache
/// [batch, max_ctx, kv_dim] at its own `positions[b]`.
pub type KvAppendBatchFn = unsafe extern "C" fn(
    kv: *const core::ffi::c_void,
    cache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    // per-row KV slot (null -> row index; prefill maps many rows to one slot)
    slots: *const core::ffi::c_void,
    kv_dim: u32,
    max_ctx: u32,
    batch: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Paged batched decode attention: same online-softmax math as
/// `AttnDecodeBatchFn`, but K/V live in a shared block pool `[n_blocks, 16,
/// kv_dim]` and are addressed through each slot's block table
/// (`block_tables + slot*blocks_per_slot`), one internally-contiguous 16-token
/// run per block. No `max_ctx` - capacity is the pool, not a per-slot
/// reservation. Bit-exact vs the dense kernel (only the per-token base differs).
#[allow(clippy::too_many_arguments)]
pub type AttnDecodeBatchPagedFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    pool_k: *const core::ffi::c_void,
    pool_v: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    // per-row KV slot (null -> row index; prefill maps many rows to one slot)
    slots: *const core::ffi::c_void,
    // flattened per-slot block tables, stride `blocks_per_slot` (u32 block ids)
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    swa_window: u32,
    batch: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Paged batched KV append: scatter each row's kv into the block pool at
/// `block_tables[slot*blocks_per_slot + pos/16]`, intra-block offset `pos%16`.
#[allow(clippy::too_many_arguments)]
pub type KvAppendBatchPagedFn = unsafe extern "C" fn(
    kv: *const core::ffi::c_void,
    pool: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    // per-row KV slot (null -> row index; prefill maps many rows to one slot)
    slots: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    kv_dim: u32,
    batch: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Paged FlashDecoding partial (P3b): the split analog of `AttnDecodeBatchPagedFn`
/// - each block writes an unnormalized `(O, m, l)` partial over its KV chunk,
///   merged by the (unchanged, position-agnostic) `attn_decode_batch_combine`. K/V
///   come from the block pool via block tables. Bit-exact vs the dense partial.
#[allow(clippy::too_many_arguments)]
pub type AttnDecodeBatchPartialPagedFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    pool_k: *const core::ffi::c_void,
    pool_v: *const core::ffi::c_void,
    out_o: *mut core::ffi::c_void,
    out_ml: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    swa_window: u32,
    n_splits: u32,
    batch: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Paged tiled prefill (P4b): the tiled flash-prefill (single slot, all rows
/// share it) reading K/V from the block pool via the slot's block table.
/// Bit-exact vs the dense tiled `attn_prefill`; gives paged prefill the tiled
/// perf class instead of the P4 decode-class fallback. WMMA (f16) paging is P4b-2.
#[allow(clippy::too_many_arguments)]
pub type AttnPrefillPagedFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    pool_k: *const core::ffi::c_void,
    pool_v: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    swa_window: u32,
    batch: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched fused MoE gate+up+swiglu: grid (ff, n_active, batch); block (o,slot,b)
/// drives expert idx[b][slot] for token b's activation x[b] -> out [batch,n_active,ff].
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpBatchFn = unsafe extern "C" fn(
    gate_w: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_w: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    batch: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched fused MoE down + weighted mix + residual add: grid (embd, batch);
/// block (o, b) sums the active experts into `residual` [batch, embd] (pre-zeroed).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownBatchFn = unsafe extern "C" fn(
    down_w: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fused: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched MoE top-k router: grid `batch`, token b's biased logits [n_expert]
/// -> top-k `out_idx`/`out_w` [batch, k]. Bias folded in.
pub type MoeTopkBatchFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    n_expert: u32,
    k: u32,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Reverse routing map: slot_of[b][e] = slot at which token b selected expert e,
/// else 255. `idx` [batch, n_active] -> `slot_of` [batch, n_expert] (u8).
pub type MoeSlotMapFn = unsafe extern "C" fn(
    idx: *const core::ffi::c_void,
    slot_of: *mut core::ffi::c_void,
    n_active: u32,
    n_expert: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Grouped MoE gate+up+swiglu: grid (ff, n_expert). Each expert row is dequanted
/// once and reused across every token that selected it (via `slot_of`) - the
/// weight-read+dequant amortization behind concurrent MoE throughput.
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpGroupedFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    slot_of: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_expert: u32,
    n_active: u32,
    batch: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Grouped MoE down + weighted mix + residual add: grid (embd, n_expert). Each
/// expert down row dequanted once, reused across its tokens, atomic-added into
/// `residual` [batch, embd] - which must already hold the post-attention hidden
/// state (the expert mix accumulates on top; do not zero it).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownGroupedFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    slot_of: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fused: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_expert: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Tiled grouped MoE gate+up+swiglu (SGEMM shape): a BN-wide output tile per
/// expert, K-tiled with staged activations/weights reused across the register
/// micro-tile. Same signature/output layout as the grouped kernel - a drop-in
/// higher-arithmetic-intensity replacement.
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpGemmFn = unsafe extern "C" fn(
    gate_w: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_w: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    slot_of: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_expert: u32,
    n_active: u32,
    batch: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// moe_align: group token-expert pairs by expert into contiguous BM-padded blocks
/// (sorted_row / sorted_slot / block_expert) so the sorted GEMM reads each expert's
/// tokens directly. Single block.
pub type MoeAlignFn = unsafe extern "C" fn(
    idx: *const core::ffi::c_void,
    sorted_row: *mut core::ffi::c_void,
    sorted_slot: *mut core::ffi::c_void,
    block_expert: *mut core::ffi::c_void,
    rows: u32,
    n_active: u32,
    n_expert: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted tiled MoE gate+up+swiglu: reads the moe_align layout, writes swiglu output
/// contiguously to fused_sorted. grid (ceil(ff/BN), max_blocks).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpGemmSortedFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    fused_sorted: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    alpha: f32,
    limit: f32,
    use_tc: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted tiled MoE down + weighted mix + residual add: reads fused_sorted, scatters
/// weighted results into `residual` (which must hold the post-attention hidden
/// state). grid (ceil(embd/BN), max_blocks).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownGemmSortedFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fused_sorted: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    use_tc: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Repack MXFP4 from the 17-byte on-disk block into aligned data (16 bytes/block)
/// + a separate contiguous e8m0 scale stream, so the sorted GEMM's weight load is
///   coalesced. Run once per expert-weight tensor at load. `n_blocks` = bytes/17.
pub type Mxfp4RepackFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    dst_data: *mut core::ffi::c_void,
    dst_scale: *mut core::ffi::c_void,
    n_blocks: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Convert an f32 buffer to f16 element-wise (`dst[i] = f16(src[i])`), n elements.
/// Used to store a post-rope K/V row into the fp16 KV cache on the single-stream path.
pub type ConvertF32F16Fn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched FlashDecoding partial: grid (n_heads, batch, n_splits). Each block runs
/// a partial online softmax over sequence b's KV slice s (its own position/slot/
/// window) and writes an unnormalized partial. Pair with `attn_decode_batch_combine`.
#[allow(clippy::too_many_arguments)]
pub type AttnDecodeBatchPartialFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    kc: *const core::ffi::c_void,
    vc: *const core::ffi::c_void,
    out_o: *mut core::ffi::c_void,
    out_ml: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_dim: u32,
    swa_window: u32,
    n_splits: u32,
    batch: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched FlashDecoding combine: grid (n_heads, batch). Merges the `n_splits`
/// partials per (head, sequence) into `out` [batch, n_heads, head_dim], folding the
/// per-head sink into the denominator.
pub type AttnDecodeBatchCombineFn = unsafe extern "C" fn(
    in_o: *const core::ffi::c_void,
    in_ml: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    n_splits: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Qwen3.5 Gated DeltaNet recurrence (linear attention). Grid = n_heads,
/// block = head_dim; thread j owns column j of the [head_dim x head_dim] per-head
/// state. Processes the whole `n_tokens` sequence, L2-norming q,k and scaling q
/// by 1/sqrt(head_dim) internally; `state` is [n_heads, head_dim, head_dim] read/
/// written (pass zeros to start). Matches reference::delta_net::gated_delta_recurrent.
#[allow(clippy::too_many_arguments)]
pub type GatedDeltaRecurrentFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    state: *mut core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Depthwise causal conv1d (kernel `k`) + SiLU - the Qwen3.5 DeltaNet input conv.
/// `x` [n_tokens, conv_dim], `w` [conv_dim, k] (`w[c*k+kk]`), `out` [n_tokens,
/// conv_dim]. Matches reference::delta_net::causal_conv1d_silu (zero left-padding).
pub type CausalConv1dSiluFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_tokens: u32,
    conv_dim: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// DeltaNet gate math: `beta = sigmoid(b)`; `g = ssm_a · softplus(a + dt_bias)`.
/// `a`,`b` [n_tokens, n_heads]; `ssm_a`,`dt_bias` [n_heads]; `g`,`beta` out.
/// Matches reference::delta_net::delta_gate.
#[allow(clippy::too_many_arguments)]
pub type DeltaGateFn = unsafe extern "C" fn(
    a: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    ssm_a: *const core::ffi::c_void,
    dt_bias: *const core::ffi::c_void,
    g: *mut core::ffi::c_void,
    beta: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Gated RMSNorm over `d` per row: `out = (x·rsqrt(mean(x²)+eps))·weight·silu(z)`.
/// `x`,`z`,`out` [n_rows, d]; `weight` [d]. Matches reference::delta_net::gated_rmsnorm.
#[allow(clippy::too_many_arguments)]
pub type GatedRmsNormFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    z: *const core::ffi::c_void,
    weight: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_rows: u32,
    d: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Split the DeltaNet conv output [n_tokens, 2·key_dim+value_dim] into q,k (key
/// heads, GQA-repeated to n_v_heads) and v (value heads); each output is
/// [n_tokens, n_v_heads, s]. `key_dim = s·n_k_heads`, `value_dim = s·n_v_heads`.
pub type DeltanetSplitGqaFn = unsafe extern "C" fn(
    conv: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    k_out: *mut core::ffi::c_void,
    v_out: *mut core::ffi::c_void,
    n_tokens: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    s: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Partial sectioned M-RoPE (Qwen3.5 multimodal rotary), in-place over
/// `x` [n_tokens, n_heads*head_dim]. Rotates NEOX pairs (p, p + n_rot/2) for
/// p in [0, n_rot/2); channels [n_rot, head_dim) pass through untouched. Each
/// pair p picks its position axis from the cumulative `sections` [t,h,w,e]:
/// `positions` is [4, n_tokens] axis-major (positions[axis*n_tokens + t]). For
/// text all four axes carry the token index and this collapses to partial NEOX
/// rope; for vision the axes differ. theta_scale = freq_base^(-2/n_rot). Matches
/// reference::qwen35_attn::mrope (ggml GGML_ROPE_TYPE_MROPE, non-interleaved).
#[allow(clippy::too_many_arguments)]
pub type MropeF32Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    n_rot: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sigmoid gate, in-place: `x[i] *= sigmoid(gate[i])`, n elements. The Qwen3.5
/// full-attention output gate (`attn * sigmoid(gate)`).
pub type MulSigmoidF32Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    gate: *const core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Plain SwiGLU in-place: `gate[i] = silu(gate[i]) * up[i]`, n elements. The
/// standard Llama/Qwen FFN activation (no OAI clamps). Matches
/// reference::ops::swiglu.
pub type SwigluF32Fn = unsafe extern "C" fn(
    gate: *mut core::ffi::c_void,
    up: *const core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Gather one embedding row selected by a DEVICE-resident token id:
/// `out[i] = table[token[0]*embd + i]`. The token id lives on device so the whole
/// decode step is CUDA-graph-capturable (only device-buffer contents change per
/// token, never a captured host address). `table` [vocab*embd] f32, `out` [embd].
pub type EmbedGatherFn = unsafe extern "C" fn(
    table: *const core::ffi::c_void,
    token: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    embd: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Greedy decode epilogue for a graph-resident loop: argmax(`logits`[vocab]) ->
/// next token id (ties broken by lowest index, matching a host first-max scan),
/// then advance all per-token state on-device so the captured graph can replay
/// with no host round-trip: write the id into `token` (next replay's input),
/// append it to `out_ids[step]`, bump `step`, and set `pos`/`mrope` to pos+1.
/// One block; grid-independent (capturable).
pub type ArgmaxAdvanceFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    vocab: u32,
    pmax: *mut core::ffi::c_void,
    pidx: *mut core::ffi::c_void,
    n_parts: u32,
    token: *mut core::ffi::c_void,
    pos: *mut core::ffi::c_void,
    mrope: *mut core::ffi::c_void,
    out_ids: *mut core::ffi::c_void,
    step: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused FFN gate+up+SwiGLU over the repacked Q8_0 layout: `out[o] =
/// silu(sum_i gate[o][i]*gscale[o][i/32]*x[i]) * sum_i up[o][i]*uscale[o][i/32]*x[i]`.
/// Collapses the two projection GEMVs + the elementwise SwiGLU into one launch
/// (reads x once, no intermediate buffers), f32-exact. `ff` = output width.
pub type Q8_0FfnGateUpSwigluFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused DeltaNet alpha+beta projection + gate (decode): `a=alpha·x`, `b=beta·x`,
/// then `beta_out[o]=sigmoid(b)`, `g[o]=ssm_a[o]*softplus(a+dt_bias[o])`. Collapses
/// two skinny (out=n_heads) latency-bound GEMVs + delta_gate into one launch.
/// alpha/beta in the repacked Q8_0 layout; f32-exact.
pub type DeltanetAlphaBetaGateFn = unsafe extern "C" fn(
    a_data: *const core::ffi::c_void,
    a_scale: *const core::ffi::c_void,
    b_data: *const core::ffi::c_void,
    b_scale: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    ssm_a: *const core::ffi::c_void,
    dt_bias: *const core::ffi::c_void,
    g: *mut core::ffi::c_void,
    beta: *mut core::ffi::c_void,
    in_dim: u32,
    n_heads: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched GEMM over the repacked Q8_0 layout - the prefill matmul. `y[b][o] =
/// bias[o] + Σ_i data[o][i]·scale[o][i/32]·x[b][i]` for b in [0, batch). The batch
/// is tiled inside the kernel so the weight is read once per 16 tokens (not per
/// token); at batch=1 the math is bit-identical to `q8_0_gemv_repacked`.
pub type Q8_0GemmRepackedFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched embedding gather: `out[t] = table[tokens[t]]` for t in [0, n_tokens) -
/// the prefill analog of `embed_gather` (token ids read from device).
pub type EmbedGatherBatchFn = unsafe extern "C" fn(
    table: *const core::ffi::c_void,
    tokens: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    embd: u32,
    n_tokens: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Dequant a repacked Q8_0 weight into dense f16 (`out[i] = f16(data[i] *
/// scale[i/32])`) - staging for the cuBLAS tensor-core prefill GEMM.
pub type Q8_0RepackedToF16Fn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// `gated_delta_recurrent` + per-token state snapshots (`snap` [n_tokens, n_heads,
/// D, D] gets each token's post-update state) - the speculative-decode verify
/// pass, so partial acceptance rolls the recurrent state back with one memcpy.
/// Math/order identical to the plain kernel (f32-exact).
pub type GatedDeltaRecurrentSnapFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    state: *mut core::ffi::c_void,
    out: *mut core::ffi::c_void,
    snap: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Repack a Q8_0 weight from the interleaved 34-byte block (f16 scale + 32 int8)
/// into two aligned streams: `dst_data` [n_blocks*32] contiguous int8 and
/// `dst_scale` [n_blocks] f16 - so the GEMV can vectorize the weight load. Run
/// once per weight at load. `n_blocks` = total elements / 32. Reorganization only
/// (same byte count as the source), no precision change.
pub type Q8_0RepackFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    dst_data: *mut core::ffi::c_void,
    dst_scale: *mut core::ffi::c_void,
    n_blocks: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Vectorized fused Q8_0 GEMV over the repacked layout: `y[o] = bias[o] +
/// sum_i (int8 data[o][i]) * scale[o][i/32] * x[i]`, f32 accumulate. Loads the
/// int8 weight stream 16 bytes at a time (int4) instead of byte-wise - the decode
/// bandwidth lever. `data` [out_dim*in_dim] int8, `scale` [out_dim*(in_dim/32)]
/// f16. `in_dim % 32 == 0`.
pub type Q8_0GemvRepackedFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Single-token depthwise causal conv1d + SiLU with a persistent window - the
/// Qwen3.5 DeltaNet conv on the O(1) decode path. `win` [(k-1), conv_dim] holds the
/// previous k-1 tokens' pre-conv input (win[0]=oldest); `x_new` [conv_dim] is this
/// token. `out[c] = silu(sum_{j<k-1} w[c,j]*win[j,c] + w[c,k-1]*x_new[c])`, then the
/// window advances (drop oldest, append x_new). `w` [conv_dim, k] (`w[c*k+kk]`).
pub type ConvStepF32Fn = unsafe extern "C" fn(
    win: *mut core::ffi::c_void,
    x_new: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    conv_dim: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Split the Qwen3.5 full-attn joint QG projection `qg` [n_tokens, n_heads*2*head_dim]
/// (per head: query[head_dim] then gate[head_dim]) into contiguous `q_out` and
/// `gate_out`, each [n_tokens, n_heads*head_dim]. `q_out[t,h,:]=qg[t,h,0:head_dim]`,
/// `gate_out[t,h,:]=qg[t,h,head_dim:2*head_dim]`.
pub type SplitQgFn = unsafe extern "C" fn(
    qg: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    gate_out: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// The v1 entry-point table. Growth rule: new entries append at the end and
/// `size` tells the engine how much of the table the pack actually filled -
/// old packs stay loadable, missing entries read as None (the loader
/// zero-fills past the pack's declared size).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelTableV1 {
    /// size_of the table as the PACK compiled it
    pub size: u32,
    pub _reserved: u32,
    pub mxfp4_dequant_f32: Option<DequantF32Fn>,
    pub q8_0_dequant_f32: Option<DequantF32Fn>,
    pub rmsnorm_f32: Option<RmsNormF32Fn>,
    pub rope_yarn_f32: Option<RopeYarnF32Fn>,
    pub softmax_sink_f32: Option<SoftmaxSinkF32Fn>,
    pub swiglu_oai_f32: Option<SwigluOaiF32Fn>,
    pub add_inplace_f32: Option<AddInplaceF32Fn>,
    pub scale_add_f32: Option<ScaleAddF32Fn>,
    pub attn_decode_f32: Option<AttnDecodeF32Fn>,
    pub moe_topk: Option<MoeTopkFn>,
    pub mxfp4_gemv_indexed: Option<Mxfp4GemvIndexedFn>,
    pub scale_add_dev: Option<ScaleAddDevFn>,
    pub q8_0_gemv: Option<Q8_0GemvFn>,
    pub attn_decode_partial: Option<AttnDecodePartialFn>,
    pub attn_decode_combine: Option<AttnDecodeCombineFn>,
    pub mxfp4_moe_gate_up: Option<Mxfp4MoeGateUpFn>,
    pub mxfp4_moe_down: Option<Mxfp4MoeDownFn>,
    pub q8_0_gemm: Option<Q8_0GemmFn>,
    pub quantize_q8: Option<QuantizeQ8Fn>,
    pub q8_0_gemv_dp4a: Option<Q8_0GemvDp4aFn>,
    pub mxfp4_gemv_indexed_dp4a: Option<Mxfp4GemvIndexedDp4aFn>,
    pub mxfp4_moe_gate_up_dp4a: Option<Mxfp4MoeGateUpDp4aFn>,
    pub mxfp4_moe_down_dp4a: Option<Mxfp4MoeDownDp4aFn>,
    pub attn_decode_batch: Option<AttnDecodeBatchFn>,
    pub rmsnorm_batch: Option<RmsNormBatchFn>,
    pub rope_yarn_batch: Option<RopeYarnBatchFn>,
    pub kv_append_batch: Option<KvAppendBatchFn>,
    pub mxfp4_moe_gate_up_batch: Option<Mxfp4MoeGateUpBatchFn>,
    pub mxfp4_moe_down_batch: Option<Mxfp4MoeDownBatchFn>,
    pub moe_topk_batch: Option<MoeTopkBatchFn>,
    pub moe_slot_map: Option<MoeSlotMapFn>,
    pub mxfp4_moe_gate_up_grouped: Option<Mxfp4MoeGateUpGroupedFn>,
    pub mxfp4_moe_down_grouped: Option<Mxfp4MoeDownGroupedFn>,
    pub mxfp4_moe_gate_up_gemm: Option<Mxfp4MoeGateUpGemmFn>,
    pub moe_align: Option<MoeAlignFn>,
    pub mxfp4_moe_gate_up_gemm_sorted: Option<Mxfp4MoeGateUpGemmSortedFn>,
    pub mxfp4_moe_down_gemm_sorted: Option<Mxfp4MoeDownGemmSortedFn>,
    pub mxfp4_repack: Option<Mxfp4RepackFn>,
    pub convert_f32_f16: Option<ConvertF32F16Fn>,
    pub attn_decode_batch_partial: Option<AttnDecodeBatchPartialFn>,
    pub attn_decode_batch_combine: Option<AttnDecodeBatchCombineFn>,
    pub gated_delta_recurrent: Option<GatedDeltaRecurrentFn>,
    pub causal_conv1d_silu: Option<CausalConv1dSiluFn>,
    pub delta_gate: Option<DeltaGateFn>,
    pub gated_rmsnorm: Option<GatedRmsNormFn>,
    pub deltanet_split_gqa: Option<DeltanetSplitGqaFn>,
    pub mrope: Option<MropeF32Fn>,
    pub mul_sigmoid: Option<MulSigmoidF32Fn>,
    pub swiglu: Option<SwigluF32Fn>,
    pub split_qg: Option<SplitQgFn>,
    pub conv_step: Option<ConvStepF32Fn>,
    pub q8_0_repack: Option<Q8_0RepackFn>,
    pub q8_0_gemv_repacked: Option<Q8_0GemvRepackedFn>,
    pub embed_gather: Option<EmbedGatherFn>,
    pub argmax_advance: Option<ArgmaxAdvanceFn>,
    pub q8_0_ffn_gate_up_swiglu: Option<Q8_0FfnGateUpSwigluFn>,
    pub deltanet_alpha_beta_gate: Option<DeltanetAlphaBetaGateFn>,
    pub q8_0_gemm_repacked: Option<Q8_0GemmRepackedFn>,
    pub embed_gather_batch: Option<EmbedGatherBatchFn>,
    pub q8_0_repacked_to_f16: Option<Q8_0RepackedToF16Fn>,
    pub gated_delta_recurrent_snap: Option<GatedDeltaRecurrentSnapFn>,
    /// Small-batch (≤12 rows) tiled GEMM over the repacked layout - the spec-decode
    /// verify matmul: x staged in shared across 16 output rows/block, so activation
    /// L2 traffic doesn't scale with out_dim. Same signature as Q8_0GemmRepackedFn.
    pub q8_0_gemm_repacked_mt: Option<Q8_0GemmRepackedFn>,
    /// int8 MMQ small-batch GEMM: pre-quantized activations (xq int8 + xs per-32
    /// f32 scales, from `quantize_q8`) × repacked Q8_0 weight via __dp4a, f32
    /// accumulate. `(data, scale, xq, xs, y, in_dim, out_dim, batch, stream)`.
    /// Not f32-bit-exact (activation quantization) - llama.cpp's own numeric class.
    pub q8_0_gemm_mt_dp4a: Option<Q8_0GemmRepackedFn>,
    pub layernorm: Option<LayernormFn>,
    pub gelu: Option<GeluFn>,
    pub bias_add: Option<BiasAddFn>,
    pub mrope_vision: Option<MropeVisionFn>,
    pub vision_attn: Option<VisionAttnFn>,
    pub gated_delta_recurrent_slots: Option<GatedDeltaRecurrentSlotsFn>,
    pub conv_step_slots: Option<ConvStepSlotsFn>,
    /// Multi-column (2..=8) dp4a GEMV - llama's mmvq shape: one block per output
    /// row, weight read once, int8 activation columns from L2. Same signature as
    /// the MT dp4a GEMM (`batch` = ncols).
    pub q8_0_gemv_dp4a_nc: Option<Q8_0GemmRepackedFn>,
    /// Wide-batch dp4a GEMM (32 rows per weight pass, 1 output row/warp) - the
    /// B>=17 serving matmul. Same signature as the MT variant.
    pub q8_0_gemm_mt_dp4a_wide: Option<Q8_0GemmRepackedFn>,
    /// DeltaNet split+GQA with fused q/k L2-normalization (q pre-scaled by
    /// 1/sqrt(s)) - feeds the v2 recurrence, which takes q,k pre-normalized.
    /// Same signature/layouts as `deltanet_split_gqa`; `s` must be a multiple
    /// of 32, at most 128.
    pub deltanet_split_gqa_norm: Option<DeltanetSplitGqaFn>,
    /// Gated delta recurrence v2 - warp-per-state-column (llama shape), one body
    /// for all variants: slots NULL => seq b uses state slot b; snap non-NULL =>
    /// per-token t-major snapshots; n_tokens > 1 => in-register chunk loop.
    /// State/snapshot [D, D] tiles are TRANSPOSED vs v1 (column-contiguous);
    /// q/k/v/out [B, T, n_heads, D] with q,k pre-normalized; g/beta [B, T,
    /// n_heads]. head_dim must be 128.
    pub gated_delta_recurrent_v2: Option<GatedDeltaRecurrentV2Fn>,
    /// Row-wise argmax: out[row] = index of max logit (lowest index on ties,
    /// matching the host argmax). `(logits, out_u32, rows, n, stream)`.
    pub argmax_rows: Option<ArgmaxRowsFn>,
    /// Batched-spec conv-ext staging: ext[b] = window(slots[b]) ++ mixed[b].
    /// `(wins, slots, mixed, ext, batch, km1, r, conv_dim, stream)`.
    pub conv_ext_build_slots: Option<ConvExtBuildSlotsFn>,
    /// Causal conv+SiLU over per-slot extended segments, emitting only the r
    /// real rows. `(ext, w, out, batch, km1, r, conv_dim, k, stream)`; k must
    /// equal km1+1.
    pub conv_chunk_ext: Option<ConvChunkExtFn>,
    /// Ragged spec commit, state half: roll each short slot's state back to
    /// snapshot committed[b]-1. `(states, snap, slots, committed, batch, r,
    /// n_heads, head_dim, stream)`; head_dim must be a multiple of 4.
    pub state_restore_slots: Option<StateRestoreSlotsFn>,
    /// Ragged spec commit, conv half: window(slots[b]) = ext[b] rows
    /// [committed[b], committed[b]+km1). `(ext, wins, slots, committed, batch,
    /// km1, r, conv_dim, stream)`.
    pub conv_commit_slots: Option<ConvCommitSlotsFn>,
    /// Advance staged per-row positions on device: pos[0..r] += 1 and
    /// mrope[0..4r] += 1. `(pos, mrope, r, stream)` - captured between the
    /// unrolled MTP draft steps so the whole draft loop replays as one graph.
    pub bump_rows_u32: Option<BumpRowsU32Fn>,
    /// int8 tensor-core Q8_0 GEMM (mma.sync m16n8k32); same per-block-scale
    /// numeric class as the dp4a MT kernel. `(data, scale, xq, xs, y, in_dim,
    /// out_dim, batch, stream)`; out_dim % 16 == 0, in_dim % 32 == 0.
    pub q8_0_gemm_mma: Option<Q8GemmMmaFn>,
    /// Activation quantize into the flat mmq layout: `[chunk][col][4 f32
    /// scales + 128 int8]` (144 B per block), columns zero-padded to a
    /// multiple of 128. Scale math identical to `quantize_q8` - the values
    /// are bit-identical, only the placement differs. Caller allocates
    /// `ceil(in_dim/128) * pad128(batch) * 144` bytes.
    /// `(x, yq, in_dim, batch, stream)`.
    pub quantize_q8_mmq: Option<QuantizeQ8MmqFn>,
    /// mmq-class int8 tensor-core GEMM (K staged 256 deep, ntx=2 warp shape,
    /// one block/SM, stream-k on low-tile-count grids): activations in
    /// `quantize_q8_mmq` layout. Same numeric class as `q8_0_gemm_mma` (and
    /// bit-exact with it when the launch tiles plainly, i.e. `fixup` NULL or
    /// tile efficiency >= 90%). `fixup` is stream-k scratch of >= 256 * 128 *
    /// 128 f32 (the 256-SM sizing contract), or NULL to force plain tiling.
    /// `(data, scale, yq, fixup, y, in_dim, out_dim, batch, stream)`;
    /// out_dim % 16 == 0, in_dim % 32 == 0.
    pub q8_0_gemm_mmq: Option<Q8GemmMmqFn>,
    /// Tiled prefill attention: one block per (q-head, 16-query tile), K/V
    /// streamed through shared, online softmax. Same signature and value
    /// math as `attn_decode_batch` (sinks/positions/swa/GQA/kv_dtype), same
    /// numeric class (per-32-key-tile update order, not bit-identical).
    /// Requires head_dim == 128 and `slots` non-null with one slot shared by
    /// all rows (true for every prefill pass) - use `attn_decode_batch`
    /// otherwise.
    pub attn_prefill: Option<AttnDecodeBatchFn>,
    /// Tensor-core (f16 WMMA) prefill attention - llama's own prefill
    /// attention class on this hardware: f16 Q/K/V inputs, f32 score
    /// accumulate + softmax, f16 O accumulate, exp flushed to zero below
    /// -20. Same signature/sink/swa semantics as `attn_prefill`. Requires
    /// head_dim == 256, fp16 KV, max_ctx % 64 == 0, uniform `slots`.
    pub attn_prefill_f16: Option<AttnDecodeBatchFn>,
    /// SwiGLU fused into the mmq activation quantize: yq = quantize(silu(gate)
    /// * up) in the `quantize_q8_mmq` layout - the ffn_down input without
    ///   materializing the f32 activation (bit-identical values to pd_swiglu +
    ///   quantize_q8_mmq run separately). `(gate, up, yq, in_dim, batch,
    /// stream)`.
    pub quantize_q8_mmq_swiglu: Option<QuantizeQ8MmqSwigluFn>,
    /// Residual-add + rmsnorm + mmq quantize in one pass: `x += proj` (if
    /// proj non-null; the residual write lands), `v = rmsnorm(x) * w`, `yq =
    /// quantize(v)` in the mmq layout, and `xn = v` if xn non-null. Bit-exact
    /// with the separate kernels. n % 4 == 0 and n <= 24576.
    /// `(x, proj, w, xn, yq, n, batch, eps, stream)`.
    pub add_rmsnorm_quant_mmq: Option<AddRmsnormQuantMmqFn>,
    /// Chunked gated delta rule for prefill (the `gated_delta_recurrent_v2`
    /// recurrence with only n_tokens/64 sequential state hops; matches the CPU
    /// oracle `reference::delta_net::gated_delta_chunked` to f32 tolerance, not
    /// bit-identical to the sequential kernels - prefill spans only). Same
    /// tensor layouts as v2 with batch fixed at 1: q/k/v/out `[T, H, D]`
    /// (q,k pre-normalized), g/beta `[T, H]`, state transposed `[H, D, D]`.
    /// Scratch, all sized for `nc = ceil(T/64)` chunks: dw/du `[nc, H, 64, D]`
    /// f32, aqk `[nc, H, 64, 64]` f32 (holds the pre-folded output
    /// coefficients), cg `[nc, H, 64]` f64. head_dim must be 128.
    pub gated_delta_chunked: Option<GatedDeltaChunkedFn>,
    /// int8 tensor-core MoE gate+up+swiglu over the sorted (moe_align) layout
    /// - the mmq structure on the MXFP4 expert weights (fp4 -> int8 unpack,
    ///   halved e8m0 scales folded per k32 block). Activations pre-quantized in
    ///   the STRIDED `quantize_q8` layout, gathered per sorted_row. The swiglu
    ///   output is quantized in REGISTERS and emitted as strided int8 + scales
    ///   (`fq`/`fs`, the down GEMM's direct input; bit-identical to swiglu +
    ///   `quantize_q8` run separately); PAD rows come out as exact zeros. Same
    ///   numeric class as `q8_0_gemm_mmq`, not the f32/f16 `_gemm_sorted` pair.
    ///   `(gate_data, gate_scale, gate_bias, up_data, up_scale, up_bias,
    /// sorted_row, block_expert, xq, xs, fq, fs, in_dim, ff, max_blocks,
    /// alpha, limit, stream)`; in_dim % 32 == 0, ff % 32 == 0.
    pub mxfp4_moe_gate_up_mmq: Option<Mxfp4MoeGateUpMmqFn>,
    /// int8 tensor-core sorted MoE down + weighted mix + residual add.
    /// Activation rows = fused_sorted quantized in place (strided layout,
    /// indexed blk*32+row directly). Writes topk-weighted per-(token, slot)
    /// PARTIALS (each written by exactly one block - no atomics); pair with
    /// `moe_slot_combine` for the deterministic residual add.
    /// `(down_data, down_scale, down_bias, sorted_row, sorted_slot,
    /// block_expert, topk_w, fq, fs, part, ff, embd, n_active, max_blocks,
    /// stream)`; ff % 32 == 0; part is [rows, n_active, embd] f32.
    pub mxfp4_moe_down_mmq: Option<Mxfp4MoeDownMmqFn>,
    /// One launch for a batch of device-to-device copies: `descs` is n
    /// consecutive `{src_ptr, dst_ptr, bytes}` u64 triples (device-resident);
    /// bytes % 16 == 0. Replaces per-page memcpy storms (the radix
    /// prefix-cache insert paid ~10 ms of host submit per pp512 prompt).
    /// `(descs, n, stream)`.
    pub batched_copy: Option<BatchedCopyFn>,
    /// Fold `mxfp4_moe_down_mmq`'s per-(token, slot) partials into the
    /// residual in FIXED slot order - deterministic (an atomicAdd scatter
    /// flipped near-tie greedy tokens run-to-run).
    /// `(part, residual, embd, n_active, rows, stream)`.
    pub moe_slot_combine: Option<MoeSlotCombineFn>,
    /// Batched (grid.z = token) fused dp4a MoE gate+up+swiglu: the llama
    /// mmvq-with-ids shape for tiny serving batches, where the sorted mmq
    /// tiles run latency-bound on a handful of blocks. Same per-row math as
    /// `mxfp4_moe_gate_up_dp4a` (token 0 is bit-identical to it). Layouts
    /// gain a leading token dim: `idx` [batch, n_active], `xq`/`xs` [batch,
    /// in_dim], `out` [batch, n_active, ff].
    /// `(gate_data, gate_scale, gate_bias, up_data, up_scale, up_bias, idx,
    /// xq, xs, out, in_dim, ff, n_active, batch, alpha, limit, stream)`.
    pub mxfp4_moe_gate_up_dp4a_b: Option<Mxfp4MoeGateUpDp4aBatchFn>,
    /// Batched fused dp4a MoE down + weighted mix + residual add (companion
    /// of `mxfp4_moe_gate_up_dp4a_b`; one writer per (token, element) - plain
    /// stores, deterministic). `residual` is [batch, embd], `fused_q`/
    /// `fused_s` [batch, n_active, ff].
    /// `(down_data, down_scale, down_bias, idx, topk_w, fused_q, fused_s,
    /// residual, ff, embd, n_active, batch, stream)`.
    pub mxfp4_moe_down_dp4a_b: Option<Mxfp4MoeDownDp4aBatchFn>,
    /// f32 batched matvec for tiny GEMMs (the MoE router): out[t][o] =
    /// dot(w[o], x[t]), one launch, grid (out_dim, batch). Replaces the
    /// cuBLAS gemmSN + splitKreduce pair at serving batch (two launches,
    /// ~19 us/layer of latency on a 368 KB weight).
    /// `(w, x, out, in_dim, out_dim, batch, stream)`; w is [out_dim, in_dim]
    /// row-major.
    pub matvec_f32_batch: Option<MatvecF32BatchFn>,
    /// Bias-carrying multi-column repacked dp4a GEMV - same kernel body and
    /// numeric class as `q8_0_gemv_dp4a_nc`, plus a nullable per-output-row
    /// f32 bias. The B=1 decode projections (wq/wk/wv/wo carry biases on
    /// gpt-oss) ride the block-per-row repacked shape this way; the old
    /// warp-per-row 34-byte-block GEMV underfills many-SM parts.
    /// `(data, scale, bias, xq, xs, y, in_dim, out_dim, ncols, stream)`.
    pub q8_0_gemv_dp4a_nc_b: Option<Q8_0GemvDp4aNcBFn>,
    /// f32 -> e4m3 + ue8m0 per-32 activation quantize (the block-scale mma's
    /// B side; power-of-2 scales). `(x, q, scale, n, stream)`; n % 32 == 0.
    pub quantize_e4m3: Option<QuantizeE4m3Fn>,
    /// sm_120a block-scale (kind::mxf8f6f4) sorted-MoE gate+up+swiglu: mxfp4
    /// weights feed the tensor core packed, e4m3 activations, HARDWARE ue8m0
    /// scaling - the s8 mmq pair spent ~4 issue slots of unpack/rescale per
    /// mma. Emits e4m3+ue8m0 swiglu output (sorted-row indexed), the down
    /// kernel's direct input. NUMERIC CLASS: fp8 activations (perplexity
    /// class, not greedy-exact vs mmq). Returns cudaErrorNotSupported unless
    /// the pack was built for sm_120a. `(gate_data, gate_scale, gate_bias,
    /// up_data, up_scale, up_bias, sorted_row, block_expert, yq, ys, fq, fs,
    /// in_dim, ff, max_blocks, alpha, limit, stream)`.
    pub mxfp4_moe_gate_up_bs: Option<Mxfp4MoeGateUpBsFn>,
    /// Block-scale companion of `mxfp4_moe_gate_up_bs`: down over the sorted
    /// layout, emitting the same deterministic per-(token, slot) partials as
    /// `mxfp4_moe_down_mmq` (fold with `moe_slot_combine`). `(down_data,
    /// down_scale, down_bias, sorted_row, sorted_slot, block_expert, topk_w,
    /// fq, fs, part, ff, embd, n_active, max_blocks, stream)`.
    pub mxfp4_moe_down_bs: Option<Mxfp4MoeDownBsFn>,
    /// K-split int8 tensor-core GEMM for the B <= 64 serving rung: the plain
    /// 64x64-tile mma grid is out_dim/64 blocks (8 on wk/wv) and idles
    /// many-SM dies; this launcher z-splits K into partial planes (`part` >=
    /// nz * out_dim * batch f32 - the stream-k fixup scratch fits) and sums
    /// them in fixed z order (deterministic; same f32-regroup numeric class
    /// as `q8_0_gemm_mma`). `(data, scale, xq, xs, part, y, in_dim, out_dim,
    /// batch, stream)`; out_dim % 16 == 0, in_dim % 32 == 0, batch <= 64.
    pub q8_0_gemm_mma_ks: Option<Q8GemmMmaKsFn>,
    /// Q8_0 routed-expert fused gate+up+SwiGLU, token-batched (qwen3.6-A3B
    /// class: 256 small experts, top-8, Q8_0 weights - the mxfp4 MoE family
    /// does not apply). Weight row (e, o) at (e*ff + o) in the repacked
    /// stream; dp4a int8 class. `(gate_data, gate_scale, up_data, up_scale,
    /// idx, xq, xs, out, in_dim, ff, n_active, batch, stream)`;
    /// out [batch, n_active, ff], in_dim % 32 == 0.
    pub q8_0_moe_gate_up_dp4a: Option<Q8MoeGateUpDp4aFn>,
    /// Companion down + weighted combine: out[b][o] = sum_slot topk_w *
    /// dot(down[e][o], fused_q[b][slot]). Plain write; the caller folds the
    /// shared expert and residual. `(down_data, down_scale, idx, topk_w, fq,
    /// fs, out, ff, embd, n_active, batch, stream)`; ff % 32 == 0,
    /// n_active <= 8.
    pub q8_0_moe_down_dp4a: Option<Q8MoeDownDp4aFn>,
    /// Shared-expert scalar sigmoid gate fold: dst[b][i] +=
    /// sigmoid(dot(x[b], w)) * src[b][i] (w = ffn_gate_inp_shexp [n_in]).
    /// `(dst, src, x, w, n_out, n_in, batch, stream)`.
    pub shexp_gate_add: Option<ShexpGateAddFn>,
    /// Sorted Q8_0 MoE gate+up+SwiGLU over the moe_align layout: each
    /// expert's weights stream once per pass regardless of token count (the
    /// token-batched dp4a pair re-reads routed rows per token). Output is
    /// sorted-contiguous, zeros on PAD rows. `(gate_data, gate_scale,
    /// up_data, up_scale, sorted_row, block_expert, xq, xs, fused, in_dim,
    /// ff, max_blocks, stream)`; in_dim % 256 == 0.
    pub q8_0_moe_gate_up_sorted: Option<Q8MoeGateUpSortedFn>,
    /// Sorted companion down: reads the sorted-contiguous quantized SwiGLU
    /// output, writes per-(token, slot) weighted partials for
    /// `moe_slot_combine`. `(down_data, down_scale, sorted_row, sorted_slot,
    /// block_expert, topk_w, fq, fs, part, ff, embd, n_active, max_blocks,
    /// stream)`; ff % 32 == 0 (K tail staged as zeros since  -
    /// previously ff % 256).
    pub q8_0_moe_down_sorted: Option<Q8MoeDownSortedFn>,
    /// int8-MMA (tensor-core) sorted MoE gate+up+SwiGLU with fused
    /// in-register per-32 output quantize - fq/fs are emitted directly, the
    /// f32 activation never lands in memory. Same signature semantics as the
    /// dp4a sorted kernel plus the fused quantize outputs. `(gate_data,
    /// gate_scale, up_data, up_scale, sorted_row, block_expert, xq, xs, fq,
    /// fs, in_dim, ff, max_blocks, stream)`; in_dim % 256 == 0, ff % 32 == 0.
    /// Empty (returns ok, computes nothing) on packs built below sm_80.
    pub q8_0_moe_gate_up_mma: Option<Q8MoeGateUpMmaFn>,
    /// int8-MMA sorted MoE down: deterministic per-(token, slot) weighted
    /// partials for `moe_slot_combine`. `(down_data, down_scale, sorted_row,
    /// sorted_slot, block_expert, topk_w, fq, fs, part, ff, embd, n_active,
    /// max_blocks, stream)`; ff % 256 == 0, embd % 32 == 0.
    pub q8_0_moe_down_mma: Option<Q8MoeDownMmaFn>,
    /// Fused residual-add + RMSNorm: x += proj (written back), out =
    /// rmsnorm(x, w). Bit-identical to add-then-norm; removes one graph-node
    /// drain per layer on the decode paths. `(x, proj, w, out, n, eps,
    /// batch, stream)`.
    pub add_rmsnorm_batch: Option<AddRmsnormBatchFn>,
    /// High-occupancy variant of `q8_0_gemm_mmq` for the very-large-M encoder
    /// prefill: same 128x128 output tile (weight L2 reuse preserved) but K is
    /// staged 128-deep, halving the weight tile's shared so two blocks/SM fit
    /// (33% vs 16.6% occupancy) - a second resident block fills the
    /// `__syncthreads` gaps that are the #1 warp stall at large M. TILED only
    /// (no stream-k fixup). Same Q8_0 int8 numeric class as `q8_0_gemm_mmq`.
    /// `(data, scale, yq, y, in_dim, out_dim, batch, stream)`; out_dim % 16 ==
    /// 0, in_dim % 32 == 0.
    pub q8_0_gemm_mmq_hi: Option<Q8GemmMmqHiFn>,
    /// Software-pipelined (2-stage cp.async, double-buffered) variant of
    /// `q8_0_gemm_mmq` for the very-large-M encoder prefill - the llama
    /// mul_mat_q approach: same 1-block/SM tile but the next K-chunk is
    /// prefetched via cp.async so `__syncthreads` never waits (barrier stall
    /// 0.2 vs 1.2). TILED only; requires in_dim % 128 == 0. Same Q8_0 int8
    /// numeric class. `(data, scale, yq, y, in_dim, out_dim, batch, stream)`;
    /// out_dim % 16 == 0.
    pub q8_0_gemm_mmq_pipe: Option<Q8GemmMmqPipeBFn>,
    /// Multi-slot batched TILED (flash) prefill attention - the encoder runs
    /// many short sequences at once, each query row in a slot. Same value math
    /// / numeric class as `attn_prefill`/`attn_decode_batch` but tiled with
    /// per-32-key online softmax instead of decode-class per-row. Per-text
    /// tiling: `tile_row0`/`tile_slot` (len `n_qtiles`) put each 16-query tile
    /// inside one slot. head_dim must be 128. `(q, kc, vc, sinks, out,
    /// positions, slots, tile_row0, tile_slot, n_qtiles, n_heads, n_kv_heads,
    /// head_dim, max_ctx, kv_dim, swa_window, n_rows, scale, kv_dtype, stream)`.
    pub attn_prefill_batch: Option<AttnPrefillBatchFn>,
    /// Device re-quant of a repacked Q8_0 weight (int8 rows + f16 per-32
    /// scales) into mxfp4 planes (packed e2m1 nibbles in the GGUF split order
    /// + ue8m0 per-32) - the block-scale GEMM's weight format. LOSSY (4-bit
    ///   weights); run once at load, sm_120a packs only (NULL elsewhere).
    ///   `(q8_data, q8_scale, mx_data, mx_scale, n_blocks, stream)`.
    pub q8_0_to_mxfp4: Option<Q8ToMxfp4Fn>,
    /// sm_120a dense block-scale GEMM (kind::mxf8f6f4): mxfp4 weight x e4m3
    /// activation with HARDWARE ue8m0 scaling - the Blackwell encoder-prefill
    /// path (the Q8_0 mmq GEMMs fold scales on CUDA cores and leave the
    /// tensor pipe idle). Activations from `quantize_e4m3`. NUMERIC CLASS:
    /// fp4 weights + fp8 activations (retrieval-quality gated, never
    /// greedy-exact vs mmq). NULL unless built for sm_120a.
    /// `(data, scale, xq, xs, y, in_dim, out_dim, batch, stream)`;
    /// in_dim % 32 == 0.
    pub mxfp4_gemm_bs: Option<Mxfp4GemmBsFn>,
    /// SwiGLU fused into the e4m3 + ue8m0 quantize (the P6j fusion in the
    /// block-scale numeric class): silu(gate)*up quantized in registers, the
    /// f32 product never lands in memory. Same silu math as `swiglu`.
    /// `(gate, up, q, scale, n, stream)`; n % 32 == 0.
    pub quantize_e4m3_swiglu: Option<QuantizeE4m3SwigluFn>,
    /// Q8_0 -> packed-ADJACENT e2m1 planes (low nibble of byte j = element
    /// 2j - the mxf4 m16n8k64 A format, not the GGUF split order the
    /// `q8_0_to_mxfp4` planes keep). Same scale pick and RN-even encode.
    /// NULL off sm_120a packs. `(q8_data, q8_scale, mx_data, mx_scale,
    /// n_blocks, stream)`.
    pub q8_0_to_fp4p: Option<Q8ToMxfp4Fn>,
    /// f32 -> packed-adjacent e2m1 + ue8m0 per-32 (the mxf4 B side; half the
    /// e4m3 activation bytes). `(x, q, scale, n, stream)`; n % 32 == 0.
    pub quantize_e2m1: Option<QuantizeE4m3Fn>,
    /// SwiGLU fused into the e2m1 quantize. `(gate, up, q, scale, n,
    /// stream)`; n % 32 == 0.
    pub quantize_e2m1_swiglu: Option<QuantizeE4m3SwigluFn>,
    /// sm_120a mxf4 dense GEMM (m16n8k64, fp4 x fp4, scale_vec::2X hardware
    /// ue8m0): the full Blackwell fp4 rate - 2x the mxf8f6f4/int8 MMA issue
    /// rate, half the activation bytes. NUMERIC CLASS: fp4 weights AND fp4
    /// activations (the lossiest rung; retrieval-quality gated, `mxfp4_gemm_bs`
    /// is the e4m3-activation fallback). Weights from `q8_0_to_fp4p`,
    /// activations from `quantize_e2m1`. NULL unless built for sm_120a.
    /// `(data, scale, xq, xs, y, in_dim, out_dim, batch, stream)`;
    /// in_dim % 32 == 0.
    pub mxfp4_gemm_f4: Option<Mxfp4GemmBsFn>,
    /// Q8_0 -> nvf4 planes: packed-adjacent e2m1 + E4M3 scales per SIXTEEN
    /// elements (the NVFP4 recipe - finer, non-power-of-2 scales; the scale
    /// plane is numel/16 bytes, twice the mxfp4 plane). NULL off sm_120a.
    /// `(q8_data, q8_scale, mx_data, mx_scale, n_blocks, stream)` where
    /// n_blocks counts 32-element Q8_0 blocks.
    pub q8_0_to_nvf4: Option<Q8ToMxfp4Fn>,
    /// f32 -> nvf4 (packed e2m1 + e4m3 per-16). `(x, q, scale, n, stream)`;
    /// n % 32 == 0, scale plane n/16 bytes.
    pub quantize_nvf4: Option<QuantizeE4m3Fn>,
    /// SwiGLU fused into the nvf4 quantize. `(gate, up, q, scale, n, stream)`.
    pub quantize_nvf4_swiglu: Option<QuantizeE4m3SwigluFn>,
    /// sm_120a nvf4 dense GEMM (m16n8k64, kind::mxf4nvf4, scale_vec::4X):
    /// the mxf4 issue rate with E4M3-per-16 scaling - the outlier-tolerant
    /// fp4 x fp4 rung. Weights from `q8_0_to_nvf4`, activations from
    /// `quantize_nvf4`. `(data, scale, xq, xs, y, in_dim, out_dim, batch,
    /// stream)`; in_dim % 32 == 0.
    pub mxfp4_gemm_nv4: Option<Mxfp4GemmBsFn>,
    /// Q8_0 -> H128-ROTATED nvf4 weight planes (QuaRot class): each row's
    /// 128-chunks multiply through an orthonormal blockwise Hadamard before
    /// the per-16 e4m3 quantize. Pair only with `quantize_nvf4_rot`
    /// activations - the two rotations cancel inside the GEMM; mixing with
    /// unrotated planes computes a rotated (wrong) product. NULL off
    /// sm_120a. `(q8_data, q8_scale, mx_data, mx_scale, n_blocks, stream)`;
    /// n_blocks % 4 == 0 (in_dim % 128 == 0).
    pub q8_0_to_nvf4_rot: Option<Q8ToMxfp4Fn>,
    /// f32 -> H128-rotated nvf4 activations (the fused QuaRot rotation -
    /// the runtime cost of taming the hidden's outlier channels is ~zero).
    /// `(x, q, scale, n, stream)`; n % 128 == 0.
    pub quantize_nvf4_rot: Option<QuantizeE4m3Fn>,
    /// Fused dense FFN front half: gate+up block-scale GEMMs (mxf8f6f4,
    /// e4m3 activations staged once for both matrices) + in-register
    /// silu(g)*u + nvf4 quantize - emits the down GEMM's fq/fs planes
    /// directly; the f32 gate/up planes never exist. BIT-IDENTICAL to the
    /// unfused chain (same per-acc MMA order, same swiglu/quantize math).
    /// `(gate_data, gate_scale, up_data, up_scale, xq, xs, fq, fs, in_dim,
    /// ff, batch, stream)`; in_dim % 32 == 0, ff % 16 == 0. NULL off
    /// sm_120a.
    pub mxfp4_gemm_bs_gu: Option<Mxfp4GemmBsGuFn>,
    /// Per-column running abs-max over a row-major [rows, n] f32 plane,
    /// accumulated across calls (caller zeroes `out` once) - SmoothQuant
    /// activation statistics. `(x, out, rows, n, stream)`; n % 32 == 0.
    pub col_absmax: Option<ColAbsmaxFn>,
    /// Per-column abs-max of a repacked Q8_0 weight - the weight half of
    /// the SmoothQuant balance. `(data, scale, out, in_dim, out_dim,
    /// stream)`; in_dim % 32 == 0.
    pub q8_0_col_absmax: Option<Q8ColAbsmaxFn>,
    /// nvf4 activation quantize with the SmoothQuant fold: v[c] * sinv[c]
    /// before the per-16 quantize. `(x, sinv, q, scale, n, in_dim,
    /// stream)`; n % 32 == 0, in_dim % 8 == 0.
    pub quantize_nvf4_smooth: Option<QuantizeNvf4SmoothFn>,
    /// Q8_0 -> nvf4 weight requant with the SmoothQuant fold: w[r][c] *
    /// svec[c] before the per-16 quantize. Pair only with
    /// `quantize_nvf4_smooth` activations using the inverse vector. NULL
    /// off sm_120a. `(q8_data, q8_scale, svec, mx_data, mx_scale, n_blocks,
    /// in_dim, stream)`.
    pub q8_0_to_nvf4_smooth: Option<Q8ToNvf4SmoothFn>,
    /// SwiGLU + SmoothQuant fold + nvf4 quantize in one pass (the smoothed
    /// down site's input). `(gate, up, sinv, q, scale, n, in_dim, stream)`;
    /// n % 32 == 0, in_dim % 8 == 0.
    pub quantize_nvf4_swiglu_smooth: Option<QuantizeNvf4SwigluSmoothFn>,
    /// e4m3 + ue8m0 quantize with the SmoothQuant fold - the mxf8f6f4
    /// class's activation side (migration usually runs weight-ward here:
    /// low alpha normalizes the coarse fp4 weight columns and the finer
    /// e4m3 activations absorb the range). `(x, sinv, q, scale, n, in_dim,
    /// stream)`; n % 32 == 0, in_dim % 4 == 0.
    pub quantize_e4m3_smooth: Option<QuantizeNvf4SmoothFn>,
    /// SwiGLU + fold + e4m3 quantize in one pass (the smoothed-F8 down
    /// input). `(gate, up, sinv, q, scale, n, in_dim, stream)`.
    pub quantize_e4m3_swiglu_smooth: Option<QuantizeNvf4SwigluSmoothFn>,
    /// Q8_0 -> split-order mxfp4 with the SmoothQuant weight fold. Pair
    /// only with `quantize_e4m3_smooth` activations using the inverse
    /// vector. NULL off sm_120a. `(q8_data, q8_scale, svec, mx_data,
    /// mx_scale, n_blocks, in_dim, stream)`.
    pub q8_0_to_mxfp4_smooth: Option<Q8ToNvf4SmoothFn>,
    /// Multi-slot tensor-core prefill attention (WMMA f16, head_dim 128,
    /// fp16 KV, max_ctx % 64 == 0). Same contract as `attn_prefill_batch`
    /// but the host tiles queries every 32 rows per text instead of 16.
    /// Numeric class: f16 QKV in, f32 softmax, f16 O accumulate - Not the
    /// scalar batch kernel's class; encoder calibration re-gates quality.
    pub attn_prefill_batch_f16: Option<AttnPrefillBatchFn>,
    /// Fused per-head RMSNorm + YaRN rope for the q projection (bit-exact
    /// with the rmsnorm_batch -> rope_yarn_batch sequence; head_dim 128
    /// only). `(x, w, out, positions, n_heads, head_dim, eps, theta_scale,
    /// freq_scale, corr_low, corr_high, ext_factor, mscale, batch, stream)`.
    pub q_norm_rope: Option<QNormRopeFn>,
    /// Fused per-head RMSNorm + YaRN rope + KV-cache scatter for the k
    /// projection (replaces norm + rope + kv_append_batch, bit-exact modulo
    /// the cache dtype store; head_dim 128 only). `(x, w, cache, positions,
    /// slots, n_kv_heads, head_dim, max_ctx, eps, theta_scale, freq_scale,
    /// corr_low, corr_high, ext_factor, mscale, batch, kv_dtype, stream)`.
    pub k_norm_rope_append: Option<KNormRopeAppendFn>,
    /// Q8_0 -> e4m3 weight planes (8-bit data + ue8m0/32 scales) for the
    /// W8A8-FP8 GEMM. NULL off sm_120a. `(q8_data, q8_scale, f8_data,
    /// f8_scale, n_blocks, stream)`.
    pub q8_0_to_f8w: Option<Q8ToMxfp4Fn>,
    /// Dense W8A8-FP8 GEMM: e4m3 weights x e4m3 activations on the
    /// block-scale MMA. Same call shape as `mxfp4_gemm_bs` but the weight
    /// data plane is full bytes (out_dim x in_dim). NULL off sm_120a.
    pub f8_gemm_w8: Option<Mxfp4GemmBsFn>,
    /// Fused-QKV consumer: from one combined [batch, q_dim+2*kv_dim] GEMM
    /// output, norm+rope q into the packed attention input, norm+rope+
    /// scatter k into the K cache, convert+scatter v into the V cache -
    /// bit-exact with the three separate kernels it replaces. head_dim 128.
    /// `(x, wq_norm, wk_norm, qn_out, kcache, vcache, positions, slots,
    /// n_heads, n_kv_heads, head_dim, max_ctx, eps, theta_scale, freq_scale,
    /// corr_low, corr_high, ext_factor, mscale, batch, kv_dtype, stream)`.
    pub qkv_norm_rope_append: Option<QkvNormRopeAppendFn>,
    /// mmq_pipe with 64-deep K stages at 2 blocks/SM - the small-grid
    /// wave-quantization variant (same Q8_0 numeric class, same tile; the
    /// engine picks it when the grid is under ~2 waves). Same call shape
    /// as `q8_0_gemm_mmq_pipe`.
    pub q8_0_gemm_mmq_pipe64: Option<Q8GemmMmqHiFn>,
    /// mmq_pipe with tail split-K (Stream-K lite): full waves run the
    /// unchanged pipe kernel; the last-wave tail tiles split over K with a
    /// deterministic reduce. Same per-k32 fold; the tail tiles' outer f32
    /// sum regroups (mmq class, not bit-identical). `partials` holds
    /// tail x splits x 128x128 f32; NULL falls back to the plain kernel.
    /// `(data, scale, yq, y, partials, in_dim, out_dim, batch, sm_count,
    /// stream)`.
    pub q8_0_gemm_mmq_pipe_sk: Option<Q8GemmMmqPipeSkFn>,
    /// Fused per-row token sampling on decode logits: `params[row]` = 4 u32
    /// words `{inv_t f32-bits, u f32-bits, mode, pad}`; mode 0 = skip, 1 =
    /// greedy argmax (lowest index on ties, matching the host scan), 2 =
    /// temperature-only categorical draw at the u-quantile (host `sample_all`
    /// semantics incl. degenerate-mass argmax fallbacks). `out[row]` = token
    /// id for modes 1/2, untouched otherwise. `(logits, params, out_u32,
    /// rows, n, stream)`.
    pub sample_rows: Option<SampleRowsFn>,
    /// Bias-folding variant of `q8_0_gemm_mt_dp4a` (the serving 5..=8 dense
    /// rung; gpt-oss projections are biased). Bit-exact vs GEMM + `bias_add`:
    /// the single f32 bias add lands on the completed per-element sum.
    /// `(data, scale, xq, xs, bias, y, in_dim, out_dim, batch, stream)`.
    pub q8_0_gemm_mt_dp4a_b: Option<Q8GemmMtDp4aBFn>,
    /// Bias-folding variant of `q8_0_gemm_mma_ks` (the 9..=64 rung): bias adds
    /// in the fixed-order K-split combine (or the GEMM epilogue when the split
    /// collapses to nz == 1). Bit-exact vs GEMM + `bias_add`.
    /// `(data, scale, xq, xs, bias, part, y, in_dim, out_dim, batch, stream)`.
    pub q8_0_gemm_mma_ks_b: Option<Q8GemmMmaKsBFn>,
    /// Bias-folding variant of `q8_0_gemm_mmq` (the b>64 rung): unsplit tiles
    /// fold bias in the GEMM store, stream-k split tiles in the fixup pass -
    /// bit-exact vs GEMM -> fixup -> `bias_add` either way.
    /// `(data, scale, yq, bias, fixup, y, in_dim, out_dim, batch, stream)`.
    pub q8_0_gemm_mmq_b: Option<Q8GemmMmqBFn>,
    /// `mxfp4_moe_down_bs` with the slot_combine fold fused into the epilogue
    /// (last-arrival winner per (token, 128-col y-tile) via `cnt`, one u32
    /// per key, zeroed once at alloc and never reset - each launch adds
    /// exactly n_active per key, so power-of-two n_active only). Bit-exact vs
    /// down_bs + `moe_slot_combine` (fixed slot order, residual added once).
    /// `(data, scale, bias, sorted_row, sorted_slot, block_expert, topk_w,
    /// fq, fs, part, residual, cnt, ff, embd, n_active, max_blocks, stream)`.
    pub mxfp4_moe_down_bs_res: Option<Mxfp4MoeDownBsResFn>,
    /// `moe_align` with a caller-chosen power-of-two block tile `bm` (the
    /// bs64 prefill path sorts into 64-row blocks). `(idx, sorted_row,
    /// sorted_slot, block_expert, rows, n_active, n_expert, bm, max_blocks,
    /// stream)`.
    pub moe_align_bm: Option<MoeAlignBmFn>,
    /// Prefill-config gate_up_bs: 64-token sorted blocks on 64-row weight
    /// tiles - half the block count at fat experts for the same per-launch
    /// weight traffic; same KC/read granularity as the decode config. Call
    /// shape identical to `mxfp4_moe_gate_up_bs`; sorted arrays must come
    /// from `moe_align_bm` with bm=64 and the down half must be
    /// `mxfp4_moe_down_bs64`.
    pub mxfp4_moe_gate_up_bs64: Option<Mxfp4MoeGateUpBsFn>,
    /// Prefill-config down_bs (64-row sorted blocks, 64-row weight tiles).
    /// Call shape identical to `mxfp4_moe_down_bs`.
    pub mxfp4_moe_down_bs64: Option<Mxfp4MoeDownBsFn>,
    /// Fused-QKV consumer for the yarn/no-norm family (gpt-oss): one launch
    /// replaces rope(q) + rope(k) + kv_append(k) + kv_append(v) on the fused
    /// GEMM output [batch, qdim + 2*kvdim]. Rope math is bit-identical to
    /// `rope_yarn_batch`. `(qkv, q_out, k_cache, v_cache, positions, slots,
    /// n_heads, n_kv_heads, head_dim, max_ctx, theta_scale, freq_scale,
    /// corr_low, corr_high, ext_factor, mscale, batch, kv_dtype, stream)`.
    pub qkv_rope_append_batch: Option<QkvRopeAppendBatchFn>,
    /// Pipelined-decode tick advance: feed the previous tick's sampled tokens
    /// straight into the next tick's fixed input buffers on device -
    /// `tokens[i] = out[i]; positions[i] += 1` for i < rows. Lets the serving
    /// loop enqueue tick N+1 (graph replay) before tick N's ids ever reach the
    /// host. `(out_u32, tokens_u32, positions_u32, rows, stream)`.
    pub pipe_advance: Option<PipeAdvanceFn>,
    /// Fused rmsnorm + Q8_0 quantize: `out = rmsnorm(x, w)` (f32 kept - the
    /// fallback GEMM paths read it) plus the int8/scale planes in one pass.
    /// Values identical to rmsnorm_batch(1024-wide) -> quantize_q8 (the warp
    /// amax is commutative). n % 32 == 0. `(x, w, out, q, qs, n, eps, batch,
    /// stream)`.
    pub rmsnorm_quant_q8_batch: Option<RmsnormQuantQ8BatchFn>,
    /// Fused residual-add + rmsnorm + e4m3/ue8m0 quantize (the block-scale
    /// MoE input): x += proj (written back), out = rmsnorm(x, w), q/s8 =
    /// e4m3 planes (pd_e4m3_quant4 math). n % 32 == 0. `(x, proj, w, out,
    /// q, s8, n, eps, batch, stream)`.
    pub add_rmsnorm_quant_e4m3_batch: Option<AddRmsnormQuantE4m3BatchFn>,
    /// wqkv all-in-one: ks GEMM into partial planes + fused combine/rope/
    /// append (no y materialization, no combine or rope launches). Bit-
    /// identical to `q8_0_gemm_mma_ks_b` -> `qkv_rope_append_batch`.
    /// `(data, scale, xq, xs, bias, part, q_out, k_cache, v_cache,
    /// positions, slots, in_dim, n_heads, n_kv_heads, head_dim, max_ctx,
    /// theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
    /// batch, kv_dtype, stream)`.
    pub q8_0_gemm_mma_ks_qkv_rope: Option<Q8GemmMmaKsQkvRopeFn>,
    /// Cross-layer glue fold: layer N's MoE slot-combine (fixed ascending
    /// slot order - pd_moe_slot_combine's exact math) rides layer N+1's
    /// norm+quantize pass. `(x, part, w, out, q, qs, n, n_active, eps,
    /// batch, stream)`.
    pub moe_combine_rmsnorm_quant_q8: Option<MoeCombineRmsnormQuantQ8Fn>,
    /// Load-time g||u interleave: fuse repacked gate/up planes into one
    /// plane of 128 B pairs (row pitch ceil(n_kb/4)*128, zero-padded tail).
    /// `(gate, up, dst, n_kb, rows, stream)`.
    pub mxfp4_gu_interleave: Option<Mxfp4GuInterleaveFn>,
    /// Paged decode attention: K/V from a shared block pool via per-slot block
    /// tables (page = 16). Bit-exact vs `attn_decode_batch`. See
    /// `AttnDecodeBatchPagedFn`.
    pub attn_decode_batch_paged: Option<AttnDecodeBatchPagedFn>,
    /// Paged KV append: scatter into the block pool via block tables. See
    /// `KvAppendBatchPagedFn`.
    pub kv_append_batch_paged: Option<KvAppendBatchPagedFn>,
    /// Paged FlashDecoding partial (split decode over the pool, ≥128-SM dies).
    /// Pairs with `attn_decode_batch_combine`. See `AttnDecodeBatchPartialPagedFn`.
    pub attn_decode_batch_partial_paged: Option<AttnDecodeBatchPartialPagedFn>,
    /// Paged tiled prefill (perf class for paged prefill). Bit-exact vs the dense
    /// tiled `attn_prefill`. See `AttnPrefillPagedFn`.
    pub attn_prefill_paged: Option<AttnPrefillPagedFn>,
    /// Paged f16 WMMA prefill (P4b-2): full prefill perf parity (qwen35's dense
    /// default class). Same ABI as `AttnPrefillPagedFn`. Bit-exact vs the dense
    /// `attn_prefill_f16`; page=16 aligns with the 16-key WMMA tile.
    pub attn_prefill_f16_paged: Option<AttnPrefillPagedFn>,
    /// gpt-oss paged fused-append (G1): block-table twin of
    /// `qkv_rope_append_batch` (b>64 mixed/prefill K/V append). Bit-exact vs
    /// dense under an identity block table. See `QkvRopeAppendBatchPagedFn`.
    pub qkv_rope_append_batch_paged: Option<QkvRopeAppendBatchPagedFn>,
    /// gpt-oss paged fused-append (G1): block-table twin of
    /// `q8_0_gemm_mma_ks_qkv_rope` (b<=64 fused GEMM-combine+rope+append). Same
    /// GEMM, paged K/V store. See `Q8GemmMmaKsQkvRopePagedFn`.
    pub q8_0_gemm_mma_ks_qkv_rope_paged: Option<Q8GemmMmaKsQkvRopePagedFn>,
    /// Two-weight fused repacked GEMM for the alpha/beta pair (exact-f32
    /// activations, P6b): stages the x batch-tile once and computes both
    /// weights' outputs over it - bit-exact per output vs two
    /// `q8_0_gemm_repacked` calls, ~13x less activation L2 traffic.
    /// (dataA, scaleA, dataB, scaleB, x, yA, yB, in_dim, outA, outB, batch, stream)
    pub q8_0_gemm_repacked_x2: Option<Q8GemmRepackedX2Fn>,
    /// DeltaNet decay gate over the FUSED alpha||beta activation layout
    /// (x2-v3): `ab` is [n_tokens] rows of 2*n_heads floats (alpha cols
    /// 0..h, beta h..2h) from the one-call f32-plane decay GEMM. Same
    /// per-element math as `delta_gate`.
    /// (ab, ssm_a, dt_bias, g, beta, n_tokens, n_heads, stream)
    pub delta_gate_ab: Option<DeltaGateAbFn>,
    /// slot_combine over bf16 partials (the PADDOCK_MOE_PART_BF16 prefill
    /// trade): same fixed-slot-order f32 fold, partials read as bf16.
    /// (part, residual, embd, n_active, rows, stream)
    pub moe_slot_combine_bf16: Option<MoeSlotCombineFn>,
    /// GEGLU in place on gate: gate = gelu_tanh(gate) * up, GELU constant
    /// exactly ggml_gelu_f32 (gemma4 FFN). (gate, up, n, stream)
    pub geglu: Option<GegluFn>,
    /// `rope_yarn_batch` with ggml `freq_factors`: pair k's theta is divided
    /// by factors[k] (null = all 1.0). gemma4 global layers pass rope_freqs
    /// whose 1e30 entries freeze those pairs (partial rotary).
    /// (x, positions, factors, n_heads, head_dim, theta_scale, freq_scale,
    ///  corr_low, corr_high, ext_factor, mscale, batch, stream)
    pub rope_factors_batch: Option<RopeFactorsBatchFn>,
    /// gemma4 vision 2D rope: two independent NEOX blocks per head - dims
    /// [0,hd/2) by pos_x, [hd/2,hd) by pos_y, pairs (i, i+hd/4).
    /// (x, pos_x, pos_y, n_tokens, n_heads, head_dim, theta_scale, stream)
    pub rope2d_neox: Option<Rope2dNeoxFn>,
    /// in-place final-logit softcap: x = cap·tanh(x/cap) (gemma4 head).
    /// (x, n, cap, stream)
    pub softcap: Option<SoftcapFn>,
    /// Q8_0 embedding gather with fused scale - decode's graph-capturable
    /// token input. (q8_table, tokens_dev, out, embd, n_tokens, scale, stream)
    pub embed_gather_q8: Option<EmbedGatherQ8Fn>,
    /// x = (x + y)·s - gemma4 layer tail. (x, y, s, n, stream)
    pub add_scale: Option<AddScaleFn>,
    /// GEGLU over concatenated gate|up rows. (x, ff, rows, stream)
    pub geglu_pair: Option<GegluPairFn>,
    /// gemma4 fused decode QKV epilogue (see pack comment).
    /// (qkv, wq_norm, wk_norm, q_out, kc, vc, positions, slots, factors,
    ///  block_tables, bps, n_head, n_kv, head_dim, max_ctx, batch, eps,
    ///  theta_scale, stream)
    pub gemma_qkv_nra: Option<GemmaQkvNraFn>,
    /// x = (x + rmsnorm(proj)·w)·s per row - gemma4 layer-half tail.
    /// (x, proj, w, n, eps, s, rows, stream)
    pub rmsnorm_add_scale: Option<RmsnormAddScaleFn>,
    /// GEGLU fused into the mmq quantize (gemma4 FFN-down feed): yq =
    /// mmq-quantize(gelu_tanh(gate)·up) - gate/up read once, no f32 landing.
    /// (gate, up, yq, in_dim, batch, stream)
    pub quantize_q8_mmq_geglu: Option<QuantizeQ8MmqGegluFn>,
    /// Wide-batch spec-verify attention: k1-deep GQA-fused FlashDecoding
    /// partial over PADDED slot-major verify chunks - one KV walk covers a
    /// chunk's k1 consecutive rows, per-row causal/window masks. Partials
    /// feed the unchanged `attn_decode_batch_combine`.
    /// (q, pool_k, pool_v, out_o, out_ml, positions, slots, block_tables,
    ///  bps, n_heads, n_kv_heads, head_dim, kv_dim, swa_window, n_splits,
    ///  rows, k1, scale, kv_dtype, stream)
    pub attn_spec_batch_paged: Option<AttnSpecBatchPagedFn>,
    /// kv_dtype-aware gemma qkv epilogue: fp8 KV appends (per-element e4m3
    /// rn-sat casts); f16 delegates to the original `gemma_qkv_nra`.
    /// (q, k, v, wq_norm, wk_norm, q_out, kc, vc, positions, slots, factors,
    ///  block_tables, bps, n_head, n_kv, head_dim, max_ctx, batch, eps,
    ///  theta_scale, kv_dtype, stream)
    pub gemma_qkv_nra2: Option<GemmaQkvNra2Fn>,
    /// e4m3 decode-lane GEMV over f8w planes.
    /// (data, scale, bias, x, y, in_dim, out_dim, stream)
    pub f8_gemv: Option<F8GemvFn>,
    /// batched twin (2..16 rows, weights read once).
    /// (data, scale, x, y, in_dim, out_dim, batch, stream)
    pub f8_gemv_batch: Option<F8GemvBatchFn>,
    /// f8 mma_ks twin: K-split block-scale MMA GEMM over the f8w planes for
    /// the 4..64-row serving band (the ue8m0 scales apply in the tensor core).
    /// (data, scale, xq, xs, part, y, in_dim, out_dim, batch, stream)
    pub f8_gemm_mma_ks: Option<F8GemmMmaKsFn>,
    /// fused GEGLU (gelu_tanh) + e4m3 quantize - bit-identical to
    /// geglu -> quantize_e4m3. Same signature as the swiglu twin.
    /// (gate, up, q, scale, n, stream)
    pub quantize_e4m3_geglu: Option<QuantizeE4m3SwigluFn>,
    /// Q8_0 -> per-ROW e4m3 requant (one power-of-2 scale per output row,
    /// f32 plane) - the sm_100 prefill class where the per-32 block-scale
    /// GEMM has no hardware fold. (q8_data, q8_scale, f8_data, row_scale,
    /// in_dim, out_dim, stream)
    pub q8_0_to_f8row: Option<Q8ToF8RowFn>,
    /// f32 -> per-ROW e4m3 activation quant. (x, q, row_scale, n_dim, batch,
    /// stream)
    pub quantize_e4m3_row: Option<QuantizeE4m3RowFn>,
    /// fold-free e4m3 GEMM over per-row planes: scales touch the epilogue
    /// only. (data, w_rowscale, xq, x_rowscale, y, in_dim, out_dim, batch,
    /// stream)
    pub f8row_gemm: Option<F8RowGemmFn>,
    /// v4 decode class: bake the SW128 smem image of a rowwise e4m3
    /// plane into contiguous 16 KB tiles laid (row_tile, k_slab)-major, so
    /// the decode GEMM streams W as one linear 1D-bulk sequence.
    /// (rowmajor, tiles, in_dim, out_dim, stream); dims must be 128-multiples.
    pub f8_repack_tiles: Option<F8RepackTilesFn>,
    /// Rowwise tcgen05 GEMM over the tile-image plane for the r<=64 decode
    /// band (cc 10 only). (wtiles, wrs, xq, xrs, part, y, in, out, batch,
    /// stream) - xq must be a >=64-row buffer (64-row TMA boxes); part is the
    /// K-split partial-plane scratch (out*batch*8 floats worst case).
    pub f8t_gemm: Option<F8TGemmFn>,
    /// fused-plane GEGLU quantize: (gu [rows][2*n_ff], q, scale, n_ff, rows)
    pub quantize_e4m3_geglu2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// qkv-concat nra epilogue: nra2 args + a shared row stride (f16 caches)
    #[allow(clippy::type_complexity)]
    pub gemma_qkv_nra2s: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused-plane GEGLU + per-ROW e4m3 quant with COMPACT [rows][n_ff]
    /// output (the f8t gu decode epilogue): (gu, q, rscale, n_ff, rows).
    pub quantize_e4m3_geglu2_row: Option<QuantizeE4m3Geglu2RowFn>,
    /// fused batched rmsnorm -> e4m3 quantize: (x, w, q, scale, n, batch, eps)
    pub rmsnorm_e4m3_batch: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// fp4 GEMV: (data, scale, bias, x, y, in_dim, out_dim) - e2m1 weights
    pub fp4_gemv: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// fp4 mma_ks twin: (data, scale, xq, xs, part, y, in, out, batch)
    pub fp4_gemm_mma_ks: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused rmsnorm -> per-32 e4m3 quantize (the prefill norm band):
    /// (x, norm_w, q, scale, n, eps, rows). Bit-identical to
    /// rmsnorm_batch (256-wide) + quantize_e4m3.
    #[allow(clippy::type_complexity)]
    pub rmsnorm_e4m3: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused rmsnorm -> ROW-scale e4m3 (the f8t decode band's format):
    /// (x, norm_w, q, row_scale_f32, n, eps, rows). Bit-identical to
    /// rmsnorm_batch + quantize_e4m3_row at the same width.
    #[allow(clippy::type_complexity)]
    pub rmsnorm_e4m3_row: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Band-boundary fusion: residual-add + post-norm + next pre-norm +
    /// row-scale e4m3 in one kernel: (x_inout, proj, post_w, pre_w, q,
    /// row_scale, n, eps, stream_scale, rows). Bit-identical to the chain
    /// rmsnorm_add_scale -> rmsnorm_batch -> quantize_e4m3_row.
    #[allow(clippy::type_complexity)]
    pub addnorm_e4m3_row: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused FlashDecoding combine + per-ROW e4m3 quant (the wo input never
    /// lands in f32): (in_o, in_ml, sinks, q, row_scale, n_heads, head_dim,
    /// n_splits, batch). Bit-identical to combine + quantize_e4m3_row.
    #[allow(clippy::type_complexity)]
    pub attn_combine_e4m3_row: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused prefill QKV epilogue norms + rope (five launches -> one; the
    /// SWA ring keeps its separate sub-span appends): (q, k, v, q_norm,
    /// k_norm, qn, kn, vn, positions, factors, n_head, n_kv, head_dim, eps,
    /// theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
    /// rows). Bit-identical to the rmsnorm_batch x3 + rope_factors x2 chain.
    #[allow(clippy::type_complexity)]
    pub qkv_norm_rope_batch: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// element-wise += k on a device u32 buffer: (buf, n, k, stream). Used
    /// to advance MTP chain rope positions inside the captured draft graph
    /// (replaces a per-step host memcpy).
    pub u32_addk: Option<
        unsafe extern "C" fn(*mut core::ffi::c_void, u32, u32, *mut core::ffi::c_void) -> i32,
    >,
    /// f8t GEMM leaving nz partial planes (no_combine, out_nz).
    pub f8t_gemm2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// nz-aware addnorm (proj = nz partial planes).
    pub addnorm_e4m3_nz: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// nz-aware fused geglu2 quant (gu = nz partial planes).
    pub quantize_e4m3_geglu2_nz: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// GGUF k-quant family (Q4_K/Q5_K/Q6_K; dtype = GGUF raw id 12/13/14) -
    /// append-only tail, see 18_kquant.cuh. Full-tensor dequant to f32:
    /// (src, dst, n_super, dtype, stream).
    pub kquant_dequant: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u64,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// load-time repack -> aligned data stream + 24B/super-block scale records:
    /// (src, dst_data, dst_scales, n_super, dtype, stream).
    pub kquant_repack: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u64,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// exact fused decode GEMV over the repacked streams:
    /// (data, scales, x, y, in_dim, out_dim, dtype, stream).
    pub kquant_gemv: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// embedding row-gather from the repacked streams:
    /// (data, scales, tokens, out, embd, n_tokens, dtype, stream).
    pub kquant_gather: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// dequant a whole repacked k-quant weight to f32 (batch-GEMM interim):
    /// (data, scales, dst, n_super, dtype, stream).
    pub kquant_dequant_rp: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u64,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// unconditionally-tiled f32 GEMM (k-quant interim compute stage):
    /// (w [out,in], x [batch,in], out [batch,out], in_dim, out_dim, batch, stream).
    pub gemm_f32: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// per-32-block activation sums off the mmq layout (W4A8 min-term operand):
    /// (yq, sums [chunk][col_pad][4] f32, in_dim, batch, stream).
    pub mmq_sums: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// stage-2 W4A8 GEMM off the k-quant repacked streams (int8 tensor cores):
    /// (data, scales, yq mmq-layout, xsums (Q4K/Q5K, else null), y [batch,out],
    /// in_dim, out_dim, batch, dtype, stream).
    pub kquant_gemm_w4a8: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// per-16 activation sums off the STRIDED int8 layout (dp4a decode ladder):
    /// (xq, sums [batch][in/16] f32, in_dim, batch, stream).
    pub q8_sums_strided: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// W4A8 dp4a batch GEMM (decode-batch shape, strided int8 activations):
    /// (data, scales, xq, xs, xsums (Q4K/Q5K, else null), y [batch,out],
    /// in_dim, out_dim, batch, dtype, stream).
    pub kquant_gemm_dp4a: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// K-split W4A8 mma GEMM (17..64 decode-batch rung; strided int8 activations,
    /// partial planes + fixed-order combine): (data, scales, xq, xs, xsums
    /// (Q4K/Q5K, else null), part (>= 8*out*batch f32), y [batch,out], in_dim,
    /// out_dim, batch, dtype, stream).
    pub kquant_gemm_mma_ks: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// k-quant routed-expert MoE gate+up+SwiGLU, token-batched (decode class):
    /// (gate_data, gate_scales, up_data, up_scales, idx, xq, xs, xsums (mu
    /// types, else null), out [B,n_active,ff], in_dim, ff, n_active, batch,
    /// gate_dtype, up_dtype, stream).
    pub kquant_moe_gate_up: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// k-quant routed-expert down + weighted combine: (down_data, down_scales,
    /// idx, topk_w, fq, fs, fsums (mu, else null), out [B,embd], ff, embd,
    /// n_active, batch, down_dtype, stream).
    pub kquant_moe_down: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// sorted k-quant MoE mma gate+up+SwiGLU (prefill/serving class; BM=32
    /// moe_align layout, one dtype for the pair): (gate_data, gate_scales,
    /// up_data, up_scales, sorted_row, block_expert, xq, xs, xsums (mu types,
    /// else null), fq, fs [sorted-contiguous rows], in_dim, ff, max_blocks,
    /// dtype, stream); in_dim % 256 == 0, ff % 32 == 0.
    pub kquant_moe_gate_up_mma: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// sorted k-quant MoE mma down: deterministic (token, slot) weighted
    /// partials for `moe_slot_combine`: (down_data, down_scales, sorted_row,
    /// sorted_slot, block_expert, topk_w, fq, fs, fsums (mu, else null),
    /// part, ff, embd, n_active, max_blocks, dtype, stream); ff % 256 == 0,
    /// embd % 32 == 0.
    pub kquant_moe_down_mma: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// W4A8 b=1 decode GEMV - the mmvq-class serving default for k-quant
    /// weights (int8 activations, dp4a dots; exact int8 dots + f32 scale
    /// application, the batch ladder's numeric class). The exact-f32
    /// `kquant_gemv` stays the oracle path. (data, scales, xq, xs, xsums (mu
    /// formats, else null), y, in_dim, out_dim, dtype, stream); in_dim % 256
    /// == 0.
    pub kquant_gemv_w4a8: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused activation quantize + per-16 int8 sums - one graph node where the
    /// b=1 kq tick launched `quantize_q8` then `q8_sums_strided` (~143 extra
    /// ~1.3 us launches per token). Outputs bit-identical to
    /// that pair: (x, q, scale, sums, n, stream).
    pub quantize_q8_sums: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// GEGLU twin of `q8_0_moe_gate_up_dp4a` (gemma4-A4B hybrid MoE): same
    /// signature/layout, epilogue is gelu_tanh(gate)*up with pd_geglu's
    /// constants - the routed branch stays in the dense gemma4 FFN's
    /// numeric class.
    pub q8_0_moe_gate_up_geglu: Option<Q8MoeGateUpDp4aFn>,
    /// Fold per-expert scalars into routed top-k weights: w[i] *= scale[idx[i]]
    /// (gemma4-A4B `ffn_down_exps.scale`, folded before the down combine).
    pub moe_scale_w: Option<MoeScaleWFn>,
    /// GEGLU twin of `q8_0_moe_gate_up_mma` (same signature/sorted layout,
    /// gelu_tanh in the in-register quantize epilogue) - the gemma4-A4B
    /// sorted expert class.
    pub q8_0_moe_gate_up_mma_geglu: Option<Q8MoeGateUpMmaFn>,
    /// Q8_0 -> per-32 e4m3 planes with K-tail padding (zero blocks) - the
    /// A4B down-expert conversion (704 -> 768 K).
    pub q8_0_to_f8w_pad: Option<Q8ToF8wPadFn>,
    /// Sorted gather of e4m3 activations + ue8m0 scales (PAD rows -> zeros)
    /// feeding the grouped tc5 GEMMs' dense Y tensor map.
    pub moe_gather_e4m3: Option<MoeGatherE4m3Fn>,
    /// Fused-plane GEGLU quantize with a PADDED output row stride (the
    /// caller owns the standing zero K-tail).
    pub quantize_e4m3_geglu2_pad: Option<QuantizeE4m3Geglu2PadFn>,
    /// tcgen05 e4m3 grouped MoE gate_up (fused per-expert [gate|up] planes,
    /// sorted-dense f32 out). NotSupported off cc-10 - callers fall back to
    /// the s8-mma sorted pair.
    pub f8bs_moe_gemm_gu: Option<F8bsMoeGemmGuFn>,
    /// tcgen05 e4m3 grouped MoE down (scattered topk_w epilogue into the
    /// slot-partials layout).
    pub f8bs_moe_gemm_dn: Option<F8bsMoeGemmDnFn>,
    /// Decode-band expert pair, intensity rebuild (4 rows/block; REORDER
    /// class vs the dp4a originals, shipped as separate entries so qwen's
    /// launchers keep exact numerics). Same signatures as the originals.
    pub q8_0_moe_gu_dec2_geglu: Option<Q8MoeGateUpDp4aFn>,
    pub q8_0_moe_dn_dec2: Option<Q8MoeDownDp4aFn>,
    /// MoE tail fusions (gemma4-A4B serial-chain depth cuts): dual-weight
    /// head norm+quant (3 nodes -> 1), topk+per-expert-scale fold (2 -> 1),
    /// combine trailer (4 -> 1).
    pub moe_head: Option<MoeHeadFn>,
    pub moe_topk_scaled: Option<MoeTopkScaledFn>,
    pub moe_tail: Option<MoeTailFn>,
    /// W4A8 multi-column decode GEMV (spec-verify r-class, ncols 2..5): the
    /// b=1 GEMV's weight walk with each window unpacked once and dotted
    /// against ncols STRIDED activation rows (`quantize_q8` layout - the
    /// same buffers the dp4a/mma_ks ladder eats). Per column the math is the
    /// b=1 GEMV's exact expressions in its exact chunk order. (data, scales,
    /// xq, xs, xsums (mu formats, else null), y [ncols × out_dim], in_dim,
    /// out_dim, ncols, dtype, stream); in_dim % 256 == 0 (windowed staging,
    /// smem bounded by the window).
    pub kquant_gemv_w4a8_nc: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused alpha/beta matvec + delta gate: one launch replacing the
    /// `matvec_f32_batch` + `delta_gate_ab` pair, bit-identical outputs
    /// (per-element summation schedule preserved; epilogue expressions
    /// verbatim). (ab_w [2·n_heads, in_dim] f32, x, ssm_a, dt_bias, g, beta,
    /// in_dim, n_heads, batch, stream).
    pub matvec_ab_gate: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// dec3 bulk-streamed decode-band expert pair: each touched
    /// expert's slab rows stream once through a cp.async.bulk ring and apply
    /// to the moe_align BM=8 block's routed rows (bexp/srow/sslot CSR).
    /// gate_up is bitwise dec2; the down leg is a reorder class (per-pair
    /// partials + `moe_combine_dec3` in dec2's slot-half order). sm_90+ only
    /// - the pack NULLs these per-device below cc 9.
    ///   (gate_data, gate_scale, up_data, up_scale, bexp, srow, sslot, xq, xs,
    ///   out, in_dim, ff, n_active, max_blocks, stream)
    pub q8_0_moe_gu_dec3_geglu: Option<Q8MoeGuDec3Fn>,
    /// (down_data, down_scale, bexp, srow, sslot, topk_w, fq, fs, part, ff,
    /// embd, n_active, max_blocks, stream)
    pub q8_0_moe_dn_dec3: Option<Q8MoeDnDec3Fn>,
    /// Fixed-order slot combine for the dec3 partials: dec2's slot-half sum
    /// tree, plain write. (part, out, n, n_active, batch, stream)
    pub moe_combine_dec3: Option<MoeCombineDec3Fn>,
    ///  decode-band f8 expert shapes. The BM=32 grouped tc5 gate_up
    /// (BN=32 mma tiles over 32-row sorted blocks - same args as the BM=128
    /// `f8bs_moe_gemm_gu`, but bexp/srow/srp come from a `moe_align_bm` at
    /// bm=32). sm_100a only - the pack NULLs these per-device off cc 10.
    pub f8bs_moe_gemm_gu_d32: Option<F8bsMoeGemmGuFn>,
    /// Y-resident BM=32 down: the block's whole fq tile stays in smem while
    /// one CTA walks OTL out-tiles streaming only W (prelude amortized;
    /// PADDOCK_MOE_F8D_OTL retunes). Live outputs bitwise-match the BM=128
    /// dn. Same args as `f8bs_moe_gemm_dn`.
    pub f8bs_moe_gemm_dn_d32: Option<F8bsMoeGemmDnFn>,
    /// PAD-block-aware fused GEGLU quantize: like `quantize_e4m3_geglu2_pad`
    /// plus (bexp, bm) - rows in PAD blocks retire after one load, so the
    /// worst-case-srp grid stops paying garbage traffic at decode.
    pub quantize_e4m3_geglu2_pad_b: Option<QuantizeE4m3Geglu2PadBFn>,
    ///  uniq-routing diagnostic: one tiny block ORs the routed pair
    /// ids into a 128-bit presence bitmap and bumps a persistent device
    /// accumulator (hist[uniq]++, pairs_sum[uniq] += pairs, plus totals;
    /// four 260-u32 regions banded device-side by the pair count:
    /// <=64 / <=256 / <=1024 / >1024). Launch-only work, so captured
    /// decode graphs bake it in and accumulation stays live on replays.
    /// The engine arms it only under PADDOCK_MOE_UNIQ, into a NON-POOL
    /// allocation (see the gemma4 Scratch notes /).
    /// (idx, pairs, n_expert (<= 128), out_accum, stream)
    pub moe_uniq_hist: Option<MoeUniqHistFn>,
    /// SwiGLU over a FUSED gate|up GEMM output ([tok][gate(ff)|up(ff)] rows,
    /// the merged gate_up plane's layout): out[t*ff+j] = silu(f[t*2ff+j]) *
    /// f[t*2ff+ff+j], packed for the down GEMM. Same silu expression as
    /// `swiglu` - bit-identical values. (fused, out, ff, n_rows, stream)
    pub swiglu_fused: Option<SwigluFusedFn>,
    /// Packed row-slice from a fused GEMM landing: dst[r*width + c] =
    /// src[r*src_stride + col_off + c] - the split epilogue for merged
    /// projection planes (DN in_qkv|gate_w etc.).
    /// (src, dst, src_stride, col_off, width, rows, stream)
    pub row_slice: Option<RowSliceFn>,
    /// e4m3 decode-band K-split GEMM (the fp8 twin of `q8_0_gemm_mma_ks`):
    /// f8w weights (e4m3 + per-32 e8m0 scale bytes) x e4m3 activations,
    /// b <= 64. PRECISION CLASS: e4m3 operands, gate before defaulting.
    /// (data, scale, xq, xs, part, y, in_dim, out_dim, batch, stream)
    pub f8d_gemm_mma_ks: Option<F8dGemmMmaKsFn>,
    /// bf16-out f8 prefill GEMM (tma route only - NotSupported elsewhere; the
    /// engine probes once). Same contract as `f8_gemm_w8`, y written bf16.
    pub f8_gemm_w8_o16: Option<Mxfp4GemmBsFn>,
    /// bf16-input swiglu + e4m3 quant - the o16 epilogue's consumer.
    /// (gate_bf16, up_bf16, q, scale, n, stream)
    pub quantize_e4m3_swiglu_b16: Option<QuantE4m3SwigluB16Fn>,
    /// bf16-residual add+rmsnorm+quant - consumes the o16 down-GEMM output.
    pub add_rmsnorm_quant_mmq_b16: Option<AddRmsnormQuantMmqFn>,
    /// x (f32) += y (bf16) - the loop-tail residual consumer for the o16
    /// down-GEMM. (x, y_bf16, n, stream)
    pub add_inplace_b16: Option<AddInplaceF32Fn>,
    /// Native-bf16 -> f8w conversion (the fp8 ingestion lane; same per-32
    /// e8m0 + e4m3 encode as q8_0_to_f8w, no Q8 double quantization).
    /// (bf16_src, f8_data, f8_scale, n_blocks, stream)
    pub bf16_to_f8w: Option<Bf16ToF8wFn>,
    /// bf16 -> f8r (per-ROW e8m0 scale - the scale-free 1.0 B/param stream).
    /// (bf16_src, f8_data, f8_scale, in_dim, out_dim, stream)
    pub bf16_to_f8r: Option<Bf16ToF8rFn>,
    /// Per-row-scale e4m3 decode ks GEMM (f8r planes; same contract as f8d).
    pub f8r_gemm_mma_ks: Option<F8dGemmMmaKsFn>,
    /// Fused-landing swiglu + e4m3 quant - one kernel replacing swiglu_fused
    /// + quantize_e4m3 at decode. (fused, q, scale, ff, n_rows, stream)
    pub swiglu_fused_e4m3: Option<SwigluFusedE4m3Fn>,
    /// add+rmsnorm writing both xn (f32) and the e4m3+scale staging - the
    /// decode norm+quant fuse. (x, proj|null, w, xn, q, scale, n, batch, eps, stream)
    pub add_rmsnorm_e4m3_xn: Option<AddRmsnormE4m3XnFn>,
    /// f8w row-major -> tile-linear box repack (load-time, one pass).
    /// (data, scale, dst, in_dim, out_dim, stream)
    pub f8w_repack_lin: Option<F8wRepackLinFn>,
    /// Tile-linear e4m3 decode GEMM (b <= 64; per-CTA contiguous stream).
    /// (wlin, xq, xs, part, y, in_dim, out_dim, batch, stream)
    pub f8_gemm_lin: Option<F8GemmLinFn>,
    /// Tile-linear prefill GEMM (tma_kt twin; o16 flag -> bf16 out).
    /// (wlin, xq, xs, y, in_dim, out_dim, batch, o16, stream)
    pub f8_gemm_lin_kt: Option<F8GemmLinKtFn>,
    /// add+rmsnorm+e4m3 with a BF16 residual (o16 prefill post-norm sites).
    /// Same contract as `add_rmsnorm_e4m3_xn`, proj read as bf16.
    pub add_rmsnorm_e4m3_xn_b16: Option<AddRmsnormE4m3XnFn>,
    /// gated rmsnorm + e4m3 quant (DN out_proj prefill glue).
    /// (x, z, w, out, q, scale, n_rows, d, eps, stream)
    pub gated_rmsnorm_e4m3: Option<GatedRmsnormE4m3Fn>,
    /// raw e4m3 -> data-only lin box repack (official-FP8 byte passthrough).
    /// (data, dst, in_dim, out_dim, stream)
    pub f8w_repack_lin_bs: Option<F8wRepackLinBsFn>,
    /// block-scale tile-linear decode GEMM: data-only boxes + f32
    /// [out/128][in/128] scale plane. (wlin, wsc, xq, xs, part, y, in_dim,
    /// out_dim, batch, stream)
    pub f8_gemm_lin_bs: Option<F8GemmLinBsFn>,
    /// fused gate|up-layout bf16 swiglu + e4m3 quant (single-GEMM prefill
    /// FFN epilogue): one [rows][2*ff] bf16 buffer, gate cols [0,ff), up
    /// [ff,2ff). (gu, q, scale, n, ff, stream)
    pub quantize_e4m3_swiglu_b16_gu: Option<QuantSwigluB16GuFn>,
    /// fused conv1d+SiLU+split+GQA+q/k-norm (DN prefill glue; the caller
    /// offsets pointers for the _at span convention). (x, w, q, k, v,
    /// n_rows, n_k_heads, n_v_heads, s, conv_k, stream)
    pub causal_conv1d_silu_qkv: Option<ConvSiluQkvFn>,
    /// bf16-out twin of `causal_conv1d_silu_qkv` (DN bf16-operand chain).
    pub causal_conv1d_silu_qkv_b16: Option<ConvSiluQkvFn>,
    /// chunked DN with bf16 v operand (per-call route; q/k/dw/du stay f32).
    /// Same signature as `gated_delta_chunked`.
    pub gated_delta_chunked_vb16: Option<GatedDeltaChunkedFn>,
    /// Per-32 twin of `addnorm_e4m3_row` (the f8a/f8r wide-decode band):
    /// (x_inout, proj, post_w, pre_w, q, scale32, n, eps, stream_scale,
    /// rows). Bit-identical to rmsnorm_add_scale -> rmsnorm_e4m3_batch.
    #[allow(clippy::type_complexity)]
    pub addnorm_e4m3_b32: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Spec-verify FIN: the FA route at n_splits==1 with in-kernel finalize
    /// (bit-identical to the walk + -inf-sink combine; batch-major rows land
    /// in `out`, ml is dead scratch). Same params as `attn_spec_batch_paged`
    /// minus n_splits. Returns -2 when the FA geometry can't engage - the
    /// caller keeps the partial+combine chain for that layer.
    #[allow(clippy::type_complexity)]
    pub attn_spec_batch_fin: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// gu-interleave lin repack (fused-epilogue layout: gate/up pair p at
    /// tile rows (p>>3)*16+(p&7) / +8). Same contract as `f8w_repack_lin`,
    /// additionally requires out_dim % 16 == 0.
    pub f8w_repack_lin_gui: Option<F8wRepackLinFn>,
    /// Interleaved-plane geglu2 twin (same formula/scale/cvt bytes, pair
    /// addressing) - the interleaved gu plane's non-fused consumers.
    pub quantize_e4m3_geglu2i: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused gu GEMM + geglu + per-32 e4m3 quant on the interleaved lin
    /// plane: q gets [batch][out_dim/2] e4m3 bytes, qs the ue8m0 scales -
    /// bit-identical to lin_kt -> geglu2i. Returns -2 when the route can't
    /// engage (caller keeps the 2-launch chain).
    /// (wlin, xq, xs, q, qs, in_dim, out_dim, batch, stream)
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Spec-verify LCO: krs spec-FA with in-kernel last-CTA-out
    /// combine - `out_f` receives the combined batch-major rows (pf_attn
    /// layout), bit-identical to partial + -inf-sink combine; out_o/out_ml
    /// stay partial scratch, `tickets` is the per-(kvh, chunk) arrival
    /// counter (wraps in-kernel; zero once at alloc). Returns -2 when the
    /// geometry isn't covered - caller keeps the partial+combine chain.
    /// (q, pool_k, pool_v, out_o, out_ml, sinks, out_f, tickets, positions,
    /// slots, block_tables, bps, n_heads, n_kv, head_dim, kv_dim,
    /// swa_window, n_splits, rows, k1, scale, kv_dtype, stream)
    #[allow(clippy::type_complexity)]
    pub attn_spec_lco_paged: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Per-channel gu GEMM (kt4a scale-free mainloop): as_row =
    /// f32 per-token scales (row quantizer), ws = f32 per-channel scales
    /// [out_dim] (gate half at 0, up half at out_dim/2). Serves the pc plane
    /// whose per-row pow2 exponents also fill the per-32 strip (all bands
    /// dequantize identically). -2 = route not covered.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_pc: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// pc lin GEMM for the qkv/wo classes (kt4 scale-free twin):
    /// (wlin, row_off, xq, as_row, ws segment-sliced, y, in_dim, out_dim,
    /// batch, o16, stream); -2 = route not covered.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_w8_pc: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// down twin (kt4d): weights per-channel in the epilogue,
    /// activations per-32 in-loop (xs = the fused gu epilogue's scales).
    /// (wlin, row_off, xq, xs, ws, y, in, out, batch, o16, stream); -2 = not
    /// covered.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_w8_pcd: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Async spec round token assembly: build the verify tick's
    /// slot-major token rows on device from the drafter chain's step-major
    /// output plane. meta = [pend | srcrow | ndr | clen | base], 5n u32.
    /// (meta, drafts, dst, n, cmax, rr, stream)
    #[allow(clippy::type_complexity)]
    pub spec_toks: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Device-side spec accept (rung B1): the accept-while-match
    /// walk on device, one compact per-slot strip out.
    /// (sampled, drafts, meta, pos, strip, n, rr, stride, stream)
    #[allow(clippy::type_complexity)]
    pub spec_accept: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Accept + next-round device prep (rung B2): the strip walk
    /// plus the chain tok/rope/bound writes, the meta pend lane, and the
    /// next verify's position rows.
    /// (sampled, drafts, meta, pos, strip, m_tok, m_pos, m_attn, n, rr,
    ///  stride, hold2, stream)
    #[allow(clippy::type_complexity)]
    pub spec_prep: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Accepted-final hidden gather into the chain's h input (rung B2).
    /// (normed, strip, meta, h, n, n_main, stride, stream)
    #[allow(clippy::type_complexity)]
    pub spec_hgather: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused K/V norm+rope+append (kv-epilogue fold): reads the
    /// RAW k/v GEMM planes, norms (k learned / v weightless) + ropes k, and
    /// appends straight into the paged cache - the kn/vn intermediates the
    /// chunk band round-tripped never land. V-less layers pass vp == kp.
    /// (kp, vp, kw, k_pool, v_pool, positions, slots, factors, block_tables,
    ///  bps, n_kv, head_dim, eps, theta_scale, freq_scale, corr_low,
    ///  corr_high, ext_factor, mscale, rows, kv_dtype, stream)
    #[allow(clippy::type_complexity)]
    pub kv_nra_rows: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Canonical spec rejection sampling, chain half: sampled
    /// draft draw from the drafter softmax at the request temperature.
    /// Writes the fp16 q row (unnormalized exp mass) + its exact f32 sum to
    /// the q-store at [step*rmax + row] (qsum 0 marks a greedy row), and the
    /// drawn token to `tok[row]`. `invt[row] <= 0` = greedy argmax (classic
    /// chain). `step` is a device u32 counter so captured chain graphs
    /// replay k times unpatched.
    /// `(logits, invt, uplane, step, qstore, qsum, tok, rows, n, rmax,
    /// stream)`.
    pub draft_rs: Option<DraftRsFn>,
    /// Canonical spec rejection sampling, verify half: per
    /// drafted verify row, accept the draft with probability min(1, p/q)
    /// against the tick's softcapped logits, else emit a residual
    /// max(p-q, 0) draw - `out[vrow]` then feeds the unchanged
    /// accept-while-match walk (accepted rows match, rejects provably
    /// differ). par = 8 u32 words per row
    /// `{vrow, jstep, srow, invt, u1, u2, pad, pad}` (f32 as bits).
    /// `(logits, drafts, qstore, qsum, par, out, nrs, rr, n, rmax, stream)`.
    pub spec_rs_resolve: Option<SpecRsResolveFn>,
    /// Drafter xh stitch - `xh[i] = [emb[i] | h[i]]` for
    /// r contiguous f32 rows, one launch replacing the 2-copies-per-row
    /// DtoD loop (bit-identical movement).
    /// `(emb, h, xh, r, n_main, stream)`.
    pub spec_xh_stitch: Option<SpecXhStitchFn>,
    /// Host-indexed f32 row gather: `dst[i] = src[idx[i]]` for i in 0..n
    /// (idx a device u32 plane). `(src, idx, dst, n, n_main, stream)`.
    pub hrow_gather: Option<HrowGatherFn>,
    /// Rowwise (strip-free) pc plane lane: pc planes quantize
    /// per-row pow2, so the in-box per-32 strip repeated one exponent -
    /// 3.03% dead weight bytes. This lane serves the same logical plane
    /// from data-only 16,384 B boxes plus a per-row ue8m0 byte vector
    /// (`wse`, padded to the 128-row tail). Bit-exact vs the strip lane.
    /// cc12-only (nulled by the pack's per-device resolution).
    /// gu-interleaved data-only repack: `(data, dst, in_dim, out_dim,
    /// stream)`.
    pub f8w_repack_lin_bs_gui: Option<F8wRepackLinBsFn>,
    /// Decode-band rowwise GEMM (b <= 64): `(wlin, wse, xq, xs, part, y,
    /// in_dim, out_dim, batch, stream)`.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// kt3-band rowwise GEMM: `(wlin, wse, xq, xs, y, in_dim, out_dim,
    /// batch, o16, stream)`.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_kt_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused gu rowwise (`wse` in BOX ROW / interleaved order): `(wlin,
    /// wse, xq, xs, q, qs, in_dim, out_dim, batch, stream)`; -2 = route
    /// not covered.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// pc chunk twins on rowwise planes - same signatures as the strip pc
    /// entries (their mainloops never read the strip, no `wse` needed).
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_pc_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    #[allow(clippy::type_complexity)]
    pub f8_gemm_w8_pc_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    #[allow(clippy::type_complexity)]
    pub f8_gemm_w8_pcd_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused qkv single-launch on the rowwise plane: `(wlin, xq,
    /// as_row, ws, yq, yk, yv, in_dim, q_dim, kv_dim, batch, stream)`.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_w8_pc_qkv_r: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Chunk-band 16-bit streams (291..295). o16 fused-qkv twin:
    /// `f8_gemm_w8_pc_qkv_r`'s signature + `o16: u32` before stream (o16=1
    /// writes bf16 into yq/yk/yv - same mainloop, final store converts).
    #[allow(clippy::type_complexity)]
    pub f8_gemm_w8_pc_qkv_r2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `qkv_norm_rope_batch` + `i16: u32` before stream (i16=1 reads bf16
    /// q/k/v; outputs stay f32).
    #[allow(clippy::type_complexity)]
    pub qkv_norm_rope_batch2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `kv_nra_rows` + `i16: u32` before stream (i16=1 reads bf16 raw k/v).
    #[allow(clippy::type_complexity)]
    pub kv_nra_rows2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `addnorm_e4m3_row` + `p16: u32` before stream (p16=1 reads bf16 proj).
    #[allow(clippy::type_complexity)]
    pub addnorm_e4m3_row2: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `rmsnorm_add_scale` + `p16: u32` before stream (p16=1 reads bf16 proj).
    #[allow(clippy::type_complexity)]
    pub rmsnorm_add_scale2: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Attention streams (296..303): f16 pf_qn/pf_attn planes on
    /// the mixed-tick route. `qkv_norm_rope_batch` + `i16, o16: u32` before
    /// stream (o16=1 writes the f16 q plane; v3 register form only).
    #[allow(clippy::type_complexity)]
    pub qkv_norm_rope_batch3: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `attn_prefill_f16_paged` + `a16: u32` before stream (a16=1: f16 q/out
    /// planes; v3c/v3s/v3w arms only, other geometries error).
    #[allow(clippy::type_complexity)]
    pub attn_prefill_f16_paged2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `attn_spec_batch_paged` + `a16: u32` before stream (a16=1: f16 q
    /// plane on the krs serve elections; partials stay f32).
    #[allow(clippy::type_complexity)]
    pub attn_spec_batch_paged2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `attn_decode_batch_paged` + `a16: u32` before stream (a16=1: f16
    /// q/out planes on the splits==1 direct-write walk).
    #[allow(clippy::type_complexity)]
    pub attn_decode_batch_paged2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `attn_decode_batch_partial_paged` + `a16: u32` before stream (a16=1:
    /// f16 q plane on the plain partial walk; partials stay f32).
    #[allow(clippy::type_complexity)]
    pub attn_decode_batch_partial_paged2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `attn_decode_batch_combine` + `o16: u32` before stream (o16=1 writes
    /// the f16 final plane; (o, m, l) partials stay f32).
    #[allow(clippy::type_complexity)]
    pub attn_decode_batch_combine2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `quantize_e4m3` + `i16: u32` before stream (i16=1 reads the f16 plane).
    #[allow(clippy::type_complexity)]
    pub quantize_e4m3_i16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `quantize_e4m3_row` + `i16: u32` before stream (i16=1 reads the f16
    /// plane).
    #[allow(clippy::type_complexity)]
    pub quantize_e4m3_row_i16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Laguna per-head softplus attention output gate:
    /// `x[r,h,d] *= softplus(gate[r,h])` in f32 (overflow-safe form),
    /// broadcast over head_dim. `(x, gate, n_heads, head_dim, rows, stream)`.
    pub mul_softplus_head: Option<MulSoftplusHeadFn>,
    /// Laguna sigmoid MoE router: selection = top-k over sigmoid(logits) +
    /// bias (selection-only correction); out weights = the UNBIASED sigmoid
    /// scores of the selected experts, sum-normalized, × routed_scale.
    /// `(logits, bias, routed_scale, n_expert, k, out_idx, out_w, batch,
    /// stream)`; logits/out per-token rows like `moe_topk_batch`.
    pub moe_topk_sigmoid_batch: Option<MoeTopkSigmoidBatchFn>,
    /// Laguna decode-tick epilogue fold: q/k per-head RMS norm + rope (plain
    /// yarn, or sectioned partial mrope when `mpos` is non-null) + paged k/v
    /// append in one launch - bit-identical to the six-kernel chain it
    /// replaces. q/k may share a fused GEMV plane via q_off/k_off.
    /// `(q_src, q_off, q_stride, k_src, k_off, k_stride, v_src, v_stride,
    /// qw, kw, q_out, k_pool, v_pool, positions, slots, mpos, block_tables,
    /// bps, n_head, n_kv, head_dim, n_rot, eps, theta_scale, freq_scale,
    /// corr_low, corr_high, ext_factor, mscale, s0..s3, rows, kv_dtype,
    /// stream)`.
    pub lag_qk_nra_rows: Option<LagQkNraRowsFn>,
    /// Standalone scalar multiply `x[..n] *= s` - ggml_scale's shape.
    /// Granite's embedding_multiplier and logits_scaling are exactly this
    /// (its residual_multiplier uses `scale_add_f32`, x += w·y, instead);
    /// minicpm and grok carry the same multiplier family. `(x, s, n, stream)`.
    pub scale_f32: Option<ScaleF32Fn>,
    /// NORM-convention twin of `rope_yarn_batch` (llama.cpp ROPE_TYPE_NORM):
    /// rotates interleaved `(2k, 2k+1)` pairs instead of half-split
    /// `(k, k+half)`. Identical signature and theta chain. Granite and the
    /// llama-arch lineage rope this way; everything else here is NEOX. Split
    /// into its own entry point (not a mode flag) so the NEOX instantiation
    /// keeps its exact SASS - the same shape llama.cpp ships as rope_norm /
    /// rope_neox and vLLM as an IS_NEOX template arg.
    pub rope_yarn_batch_norm: Option<RopeYarnBatchFn>,
    /// Qwen3.5-family fused-plane prefill consumer: reads the one-GEMM
    /// `[q|gate (per-head interleaved) | k | v]` plane; q per-head RMS norm +
    /// sectioned partial mrope -> `q_out`, raw gate halves -> `gate_out`,
    /// k norm+mrope and raw v paged-append through the kv store - the
    /// split_qg + 2×rmsnorm + 2×mrope + 2×append chain in one launch,
    /// bit-identical. hd 256 / n_rot 64 only (the qwen3.6-27b shape).
    /// `(qkg, q_off, row_stride, k_off, v_off, qw, kw, q_out, gate_out,
    /// k_pool, v_pool, positions, slots, mpos, block_tables, bps, n_head,
    /// n_kv, head_dim, n_rot, eps, theta_scale, freq_scale, corr_low,
    /// corr_high, ext_factor, mscale, s0..s3, rows, kv_dtype, stream)`.
    pub q36_qkg_nra_rows: Option<Q36QkgNraRowsFn>,
    /// q36 DN: one kt3 GEMM over a fused lin plane with a two-buffer
    /// epilogue - output rows `[0, ncut)` land in `y` and `[ncut, out_dim)`
    /// in `y2`, each at its own row stride, so the two consumers keep their
    /// layouts while the grid pays one wave tail instead of two. Returns -2
    /// when the route can't engage (no TMA / kt3 not the incumbent / odd
    /// dims) - the caller keeps its two-launch pair.
    /// `(wlin, xq, xs, y, y2, ncut, in_dim, out_dim, batch, stream)`.
    pub f8_gemm_lin_kt_split: Option<F8GemmLinKtSplitFn>,
    pub vision_attn_x: Option<VisionAttnXFn>,
    pub gather_rows_avg: Option<GatherRowsAvgFn>,
    pub gelu_erf: Option<GeluErfFn>,
    pub add_rows_bcast: Option<AddRowsBcastFn>,
    /// pipelined sibling of kquant_gemm_w4a8 - same signature/numerics, the
    /// >64-batch rung's raw weight+scale bytes ride cp.async into a shared
    /// > buffer instead of a synchronous global load, with the next
    /// > super-block's fetch overlapping this one's MMA compute.
    pub kquant_gemm_w4a8_pipe: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// kquant_gemm_w4a8_pipe's genuinely-double-buffered sibling - same
    /// signature/numerics, a real 2-deep raw byte ring (half-width tile_x)
    /// so the next super-block's load overlaps this one's entire
    /// build+compute phase, not just compute. Stays __launch_bounds__(256,1)
    /// - a 2-blocks/SM attempt hit its register target but sm_120's SM
    ///   shared-memory budget blocked occupancy from actually rising.
    pub kquant_gemm_w4a8_pipe2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Multi-segment `q8_0_gemv_repacked`: one launch over up to three
    /// same-in_dim planes sharing one activation vector (decode QKV merge,
    /// FFN gate|up merge). Bit-identical per row to the split launches -
    /// small grids waste launch ramp/drain (a 1024-row launch streams at
    /// 47% of the die's practical read ceiling, the merged 6144-row grid at
    /// 85%). `(d0,s0,b0,y0,rows0, d1,s1,b1,y1,rows1, d2,s2,b2,y2,rows2,
    /// x, in_dim, n_segs, stream)`; unused trailing segments pass nulls/0.
    pub q8_0_gemv_repacked_multi: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Fused NORM-rope(q in place) + NORM-rope(k)->paged append + v paged
    /// append - granite's 4-launch rope/append decode band as one kernel,
    /// cache and q bytes bit-identical (same theta chain, same pd_kv_store).
    /// `(q, k, v, pool_k, pool_v, positions, slots, block_tables, bps,
    /// n_heads, n_kv, head_dim, theta_scale, freq_scale, corr_low,
    /// corr_high, ext_factor, mscale, batch, kv_dtype, stream)`.
    pub rope_norm_qk_append_paged: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Multi-segment W4A8 k-quant decode GEMV - up to 3 same-in_dim planes
    /// (mixed k-quant dtypes) sharing one staged int8 activation, one launch
    /// (granite-30b QKV / gate|up merge; `q8_0_gemv_repacked_multi`'s launch
    /// economics on the k-quant family). `xsums` may be null when no segment
    /// is Q4_K/Q5_K. `(d0,s0,y0,out0,dt0, d1,s1,y1,out1,dt1,
    /// d2,s2,y2,out2,dt2, xq,xs,xsums, in_dim, n_segs, stream)`; unused
    /// trailing segments pass nulls/0.
    pub kquant_gemv_w4a8_multi: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Multi-segment nc GEMV - up to four same-in_dim Q8_0 planes sharing one
    /// staged int8 activation at ncols = 1..8 columns each, one launch (the
    /// r=2..4 batched-decode q|k|v|g and shexp gate|up merges;
    /// `q8_0_gemv_repacked_multi`'s launch economics on the multi-column
    /// class). Per-segment nullable bias. `(d0,s0,b0,y0,out0, d1,s1,b1,y1,out1,
    /// d2,s2,b2,y2,out2, d3,s3,b3,y3,out3, xq, xs, in_dim, n_segs, ncols,
    /// stream)`; unused trailing segments pass nulls/0.
    // The C signature, spelled out where the table entry is rather than
    // behind an alias: 26 parameters is what the kernel takes.
    #[allow(clippy::type_complexity)]
    pub q8_0_gemv_dp4a_nc_multi: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Packed multi-span gated delta recurrence: decode rows (len-1 items),
    /// independent short span walks, and same-slot fused-ckpt tail CHAINS
    /// (one item per chain - the shares' rows are contiguous) in one launch,
    /// driven by u32 descriptors of STRIDE 8 `(row0, len, slot, snapA_t,
    /// snapA_sel, snapB_t, snapB_sel, pad)`. Internal chain seams write
    /// in-kernel state snapshots to `snap0`/`snap1` (the per-layer
    /// pre-offset stage-blob state regions; `sel` picks the blob, `t == 0`
    /// means none) - bit-exact replacements for the between-share
    /// `copy_region` staging. Rows are addressed absolutely in q/k/v/out;
    /// items must touch distinct slots; launch after the chunked span loop
    /// so chain leaders have advanced the state. `(q, k, v, g, beta, items,
    /// states, out, snap0, snap1, n_items, n_heads, head_dim, stream)`.
    pub gated_delta_recurrent_v2_packed: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// pf7 varlen packed prefill attention (AF3): one launch per
    /// layer covering every eligible prefill span of the tick. `vl_items`
    /// is stride-4 u32 per 64-head-row tile `(q_row0, span_rows,
    /// tile_flat_row0, slot)`; tiles never cross spans so each packed CTA
    /// is bit-identical to its per-span pf7 twin. fp8 pools, hd256,
    /// G 4/6/8 only - the engine pre-checks and keeps per-span launches as
    /// the fallback. `(q, pool_k, pool_v, sinks, out, positions, vl_items,
    /// n_tiles, block_tables, blocks_per_slot, n_heads, n_kv_heads,
    /// head_dim, kv_dim, swa_window, scale, kv_dtype, stream)`.
    pub attn_prefill_f16_paged_vl: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 323: varlen chunked-GDN (GDN formulation band) - One
    /// stage1 + register-state-walk launch pair covers every eligible span
    /// of the tick. `chunk_items`: stride-2 u32 `(global row0, chunk len)`
    /// per launch chunk; `span_items`: stride-4 u32 `(first launch chunk,
    /// span rows, state f32 offset, out row0)` per span. Per-span math is
    /// identical to the per-span RS calls; the RS-route env gates are
    /// mirrored inside and any other elected arm returns
    /// cudaErrorNotSupported (the engine keeps per-span dispatch as the
    /// fallback). `(q, k, v, g, beta, states, out, dw, du, aqk, cg,
    /// chunk_items, n_chunks, span_items, n_spans, n_tokens, n_heads,
    /// head_dim, stream)`.
    pub gated_delta_chunked_rs_vl: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 324: fused-GLU W4A8 decode GEMV  - gate+up+SwiGLU as one
    /// launch: each block walks the gate row and the matching up row over
    /// one staged activation and writes `silu(g)*u` directly. Bit-exact vs
    /// the `multi<4,128>` + `swiglu` split path (identical row walks,
    /// identical epilogue expression). `(gate_data, gate_scales, up_data,
    /// up_scales, xq, xs, xsums, y, in_dim, out_dim, gate_dtype, up_dtype,
    /// stream)`.
    pub kquant_gemv_w4a8_glu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// qwen twin of `addnorm_e4m3_row`: PLAIN residual add (no post-norm, no
    /// stream scale) + pre-norm + row-scale e4m3, one launch.
    /// `(x_inout, proj, pre_w, q, row_scale, n, eps, rows, stream)`.
    /// Bit-identical to add_rmsnorm_batch + quantize_e4m3_row.
    #[allow(clippy::type_complexity)]
    pub add_rmsnorm_e4m3_row: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Up-to-4 slices of one fused landing in a single launch. Unused slots
    /// take a null dst and width 0. Bit-identical to N x `row_slice`.
    /// `(src, src_stride, rows, d0,o0,w0, d1,o1,w1, d2,o2,w2, d3,o3,w3, stream)`
    #[allow(clippy::type_complexity)]
    pub row_slice4: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// swiglu of a fused `[rows, 2*ff]` gate|up landing -> per-ROW e4m3, one
    /// launch. `(fused, q, row_scale, ff, rows, stream)`. Bit-identical to
    /// `swiglu_fused` + `quantize_e4m3_row`.
    #[allow(clippy::type_complexity)]
    pub swiglu_e4m3_row: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Batched single-query flash-decoding attention over f16 K/V slot
    /// planes - whisper's cross- and self-attention in one kernel. q/out are
    /// compact `[batch, n_heads, hd]` in ACTIVE order; K/V are
    /// `[cap, kv_stride, n_heads*hd]` indexed through `slots[b]`; the live
    /// key count for slot b is `lens[b] + len_bias` (self-attention passes
    /// the position cursor with bias 1), or `kv_len_def` when `lens` is null.
    /// `part` is `[batch, n_heads, splits, hd+2]` scratch - pass null to
    /// force the single-chunk form. `hd` must be 64 or 128.
    /// `(q, qbias, k, v, slots, lens, out, part, kv_stride, kv_len_def,
    ///   len_bias, n_heads, hd, batch, scale, kv_dtype, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_dec_attn: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `x[b] = tok[tokens[b]] + postab[pos[b]]` - whisper's decoder embedding
    /// (row copy + LEARNED position row) for every active slot in one launch.
    /// `(tok, postab, tokens, pos, x, d, batch, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_embed_pos: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Split a merged `[batch, 3*d]` q|k|v landing: q gets its bias, K and V
    /// land in the slot caches at `pos[b]` at `kv_dtype`. `bq`/`bv` may be
    /// null; whisper's k_proj genuinely has no bias.
    /// `(qkv, bq, bv, q, kc, vc, slots, pos, d, ctx, batch, kv_dtype, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_qkv_split: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Store a window's cross-attention K or V into its slot plane at
    /// `kv_dtype`, bias folded in.
    /// `(src, bias, dst, slots, rows, d, stride, batch, kv_dtype, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_kv_store: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// LayerNorm with an f16 landing - same reduction structure as
    /// `layernorm`, cast folded in. `(x, w, b, out, rows, n, eps, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_ln_f16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `x += proj + bias`, then the next block's pre-norm out of the updated
    /// residual, at f16 - whisper's whole residual seam in one launch.
    /// `(x, proj, bias, w, b, out, rows, n, eps, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_res_ln_f16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// bias + exact-erf GELU + f16 cast on the fc1 landing, one pass.
    /// `(x, bias, out, rows, n, stream)`
    #[allow(clippy::type_complexity)]
    pub whisper_bias_gelu_f16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// bias + SiLU + f16 cast on a granite-speech macaron FFN landing.
    /// `(x, bias, out, rows, n, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_bias_silu_f16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// bias + sigmoid-GLU over a `[rows, 2*d]` channel split, landing
    /// `[rows, d]` f32. `(x, bias, out, rows, d, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_bias_glu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Centered depthwise conv over time + folded BatchNorm + SiLU, f16 out.
    /// The weight plane is tap-major (`w[j*d + c]`) - transposed at load from
    /// the file's channel-major order so every tap read is coalesced.
    /// `(x, w, bnw, bnb, out, rows, d, k, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_dwconv_bn_silu_f16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Conformer blockwise attention with Shaw relative position embeddings,
    /// over a merged `[rows, 3*n_heads*hd]` q|k|v landing, f16 out.
    /// `(qkv, out, rel, rows, ctx, n_heads, hd, max_pos, scale, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_conf_attn: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// bias + row softmax + f16 cast - the CTC branch head.
    /// `(x, bias, out, rows, n, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_bias_softmax_f16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `x += s*(proj + bias)`, then the next block's pre-norm out of the
    /// updated residual, at f16. `w`/`b`/`out` may be null (residual only).
    /// `(x, proj, bias, w, b, out, rows, n, s, eps, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_res_ln_f16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `x = LN(x + s*(proj + bias))` in place (the post-LN contract: the
    /// residual stream is the normalized value), plus an optional f16
    /// landing. `(x, proj, bias, w, b, out, rows, n, s, eps, stream)`
    #[allow(clippy::type_complexity)]
    pub gs_post_ln_f16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// ROW-scale twin of `gated_rmsnorm_e4m3` (decode band): gated
    /// rmsnorm + per-ROW e4m3 for the f8t out_proj arm, one launch. d must be
    /// 128 and n_heads a multiple of 16. Bit-identical to `gated_rmsnorm` +
    /// `quantize_e4m3_row`. f32 out nullable.
    /// `(x, z, w, out, q, row_scale, batch, n_heads, d, eps, stream)`
    #[allow(clippy::type_complexity)]
    pub gated_rmsnorm_e4m3_row: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// `argmax_rows` plus everything that falls out of the same log-sum-exp:
    /// the RUNNER-UP's id in `alt`, and
    /// `stats[row] = {log p(top1), p(probe), log p(top2), H2}` where H2 is the
    /// Renyi-2 (collision) entropy in nats. `alt[row] = n` means the row had
    /// no runner-up.
    ///
    /// The pick is bit-identical to `argmax_rows` (same tie rule, applied at
    /// both ranks), so asking for confidence can never move a transcript.
    /// `(logits, out_u32, alt_u32 or null, stats_f32 or null, probe, rows, n, stream)`.
    pub argmax_top2_rows: Option<ArgmaxTop2RowsFn>,
    /// Whisper's `ApplyTimestampRules` over a `[rows, vocab]` logits block, in
    /// place. Without it a fine-tune greedily picks `<|notimestamps|>` and
    /// emits no times at all.
    /// `(logits, state_u32, rows, vocab, eot, no_ts, ts_begin, max_init, stream)`.
    pub whisper_ts_rules: Option<WhisperTsRulesFn>,
    /// row_slice4's DN split with the delta gate folded into the ab parts
    /// slots 0/1 copy (mixed, z); the `2*n_heads` ab columns at
    /// `ab_off` become g/beta directly. Bit-identical to `row_slice4` +
    /// `delta_gate`, minus one launch and the d_a/d_b intermediates.
    /// `(src, stride, rows, d0, o0, w0, d1, o1, w1, ab_off, n_heads,
    ///  ssm_a, dt_bias, g, beta, stream)`
    #[allow(clippy::type_complexity)]
    pub row_slice2_gate: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// f32 -> e4m3 with one f32 scale per 32 elements - the same plane shape
    /// `quantize_q8` writes, so the sorted MoE activation stager consumes it
    /// unchanged. Feeds `f8row_moe_gate_up_mma_geglu`.
    /// `(x, q, scale, n, stream)`
    pub quantize_e4m3_b32f: Option<QuantizeE4m3B32fFn>,
    /// Flat-scale twin of `q8_0_moe_gate_up_mma_geglu`: e4m3
    /// expert weights carrying one scale per output row (`q8_0_to_f8row`
    /// planes) against per-32-scaled e4m3 activations. The weight scale is
    /// loop-invariant, so the k walk carries no scale traffic at all - that
    /// is the whole point. Same sorted layout, same grid, and the same int8
    /// fq/fs handshake into `q8_0_moe_down_mma`. PRECISION CLASS: lossy vs
    /// Q8_0, gated like every other one.
    /// `(gate_data, gate_rs, up_data, up_rs, sorted_row, block_expert, xq,
    ///   xs, fq, fs, in_dim, ff, max_blocks, bm, stream)`
    pub f8row_moe_gate_up_mma_geglu: Option<F8RowMoeGateUpFn>,
    /// Same GEMM as `f8row_moe_gate_up_mma_geglu`, but the epilogue quantizes
    /// the geglu output to e4m3 per-32 instead of int8 per-32 (same `fs` f32
    /// scale plane, same buffer sizes) - the B operand `f8row_moe_down_mma`
    /// needs. Pair them: int8-out with the Q8_0 down, e4m3-out with this one.
    pub f8row_moe_gate_up_mma_geglu_f8: Option<F8RowMoeGateUpFn>,
    /// Flat-scale twin of `q8_0_moe_down_mma`: e4m3 expert weights with one
    /// scale per output row against the e4m3 per-32 `fq`/`fs` the f8-out
    /// gate_up wrote. Same deterministic partials epilogue.
    /// `(down_data, down_rs, sorted_row, sorted_slot, block_expert, topk_w,
    ///   fq, fs, part, ff, embd, n_active, max_blocks, bm, stream)`
    pub f8row_moe_down_mma: Option<F8RowMoeDownFn>,
    /// 350: conv-window VL store - each fresh span's last (k-1) pre-conv rows
    /// into its slot's conv window, span `(row0, take, slot, _)` quads read
    /// from device contents (chunk-tick graph capture).
    /// `(src, spans, win, n_spans, km1, conv_dim, stream)`
    pub conv_win_store_vl: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 351: BF16 dense weight-plane GEMV - per-TENSOR quant dispatch for
    /// mixed UD files. Weights stay bf16 in DRAM, activations
    /// f32, accumulation f32: the same arithmetic class the Q8_0 lane runs
    /// and strictly more precise than it. Layout matches the repacked Q8_0
    /// planes (out rows of in_dim contiguous), so this is a drop-in twin of
    /// `q8_0_gemv_repacked` at the call site.
    /// `(w, bias, x, y, in_dim, out_dim, stream)`
    pub bf16_gemv_f32: Option<Bf16GemvF32Fn>,
    /// 352: r>1 twin of `bf16_gemv_f32`.
    /// `(w, bias, x, y, in_dim, out_dim, batch, stream)`
    pub bf16_gemm_f32: Option<Bf16GemmF32Fn>,
    /// 353: bf16 -> f32 widen in the `DequantF32Fn` shape (32 elems per
    /// "block") so a bf16 tensor slots into `dequant_for` next to the real
    /// quant types - that is what makes single-row embedding gathers off a
    /// bf16 `token_embd` work unchanged.
    pub bf16_dequant_f32: Option<DequantF32Fn>,
    /// 354: bf16 twin of `embed_gather_q8` (fused output scale, device token
    /// ids, graph-capturable).
    /// `(table, tokens, out, embd, n_tokens, scale, stream)`
    pub embed_gather_bf16: Option<EmbedGatherBf16Fn>,
    /// 355-363: SiLU twins of the whole gated-FFN carrier set -
    /// muse-glimmer's FFN is SwiGLU where gemma4's is GeGLU. Each is the
    /// same kernel instantiated on the other branch of the pack's
    /// `pd_glu_act` template, so the two activations share every tile,
    /// election and quant rule and can only diverge in the nonlinearity.
    /// They are separate slots rather than an `act` argument because the
    /// table grows append-only - an existing signature may never move.
    ///
    /// 355: in-place fold over a `[rows, 2*ff]` concat row.
    pub swiglu_pair: Option<GegluPairFn>,
    /// 356: per-32 e4m3 quant of `silu(gate)*up` off the concat plane.
    #[allow(clippy::type_complexity)]
    pub quantize_e4m3_swiglu2: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 357: same, on the gu-INTERLEAVED plane (`f8w_repack_lin_gui` layout).
    #[allow(clippy::type_complexity)]
    pub quantize_e4m3_swiglu2i: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 358: per-ROW e4m3 into a compact `[rows][n_ff]` plane (the down GEMM's
    /// 64-row TMA boxes need contiguous rows).
    pub quantize_e4m3_swiglu2_row: Option<QuantizeE4m3Geglu2RowFn>,
    /// 359: nz-aware twin - `gu` is the fused GEMM's nz partial planes.
    #[allow(clippy::type_complexity)]
    pub quantize_e4m3_swiglu2_nz: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 360-363: the four fused gu-GEMM + glu + quant epilogues (strip and
    /// rowwise planes x per-32 and per-channel scales). Same -2 = route not
    /// covered convention as the GELU entries.
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_silu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_r_silu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_pc_silu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    #[allow(clippy::type_complexity)]
    pub f8_gemm_lin_gu_pc_r_silu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 364: `ROPE_TYPE_NORM` twin of `rope_factors_batch` - interleaved
    /// `(2k, 2k+1)` pairs instead of NEOX's half-split `(k, k+half)`. Same
    /// signature, same theta chain; only the pairing differs. muse-glimmer
    /// ropes NORM where gemma4, whose graph it shares in this engine, ropes
    /// NEOX, and the wrong pairing scrambles position on every
    /// roped layer while still producing fluent-looking text.
    pub rope_factors_batch_norm: Option<RopeFactorsBatchFn>,
    /// 365: fused QK-norm + rope, superset of `qkv_norm_rope_batch3` - `neox`
    /// joins `i16`/`o16` as a shape bit (1 = the half-split layout every
    /// earlier caller of this family assumes).
    /// `(q, k, v, qw, kw, qn, kn, vn, positions, factors, n_head, n_kv,
    ///  head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
    ///  ext_factor, mscale, rows, i16, o16, neox, stream)`
    #[allow(clippy::type_complexity)]
    pub qkv_norm_rope_batch4: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 366: fused QK-norm + rope, superset of `qkv_norm_rope_batch4` -
    /// `vnorm` says whether the V slots get the weightless per-head RMS norm.
    /// gemma4 does (`gemma4.cpp`: `Vcur = ggml_rms_norm(ctx0, Vcur,
    /// f_norm_rms_eps)`); muse-glimmer hands the RAW `Vcur` to `build_attn`
    /// and must not. Like `neox`, an architecture constant that
    /// appears in no metadata key - normalizing V anyway reads fluent for a
    /// few tokens and then degenerates, because it flattens the per-head
    /// magnitude differences attention is supposed to weight.
    /// `(q, k, v, qw, kw, qn, kn, vn, positions, factors, n_head, n_kv,
    ///  head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
    ///  ext_factor, mscale, rows, i16, o16, neox, vnorm, stream)`
    #[allow(clippy::type_complexity)]
    pub qkv_norm_rope_batch5: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 367: `kv_nra_rows2` plus the same two architecture constants. This
    /// kernel is `qkv_norm_rope_batch5`'s K/V half folded into the paged
    /// append, so it carries both: without `neox` the fold would rope K on
    /// the half-split layout while Q rode the interleaved one (a defect the
    /// q-side twin alone cannot expose, because the two sides only disagree
    /// inside the score), and without `vnorm` it would normalize V.
    /// `(kp, vp, kw, k_pool, v_pool, positions, slots, factors, block_tables,
    ///  blocks_per_slot, n_kv, head_dim, eps, theta_scale, freq_scale,
    ///  corr_low, corr_high, ext_factor, mscale, rows, kv_dtype, i16, neox,
    ///  vnorm, stream)`
    #[allow(clippy::type_complexity)]
    pub kv_nra_rows3: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 368: `gemma_qkv_nra2s` plus the three architecture constants its
    /// epilogue baked in - `freq_scale`, `neox`, `vnorm`.
    ///
    /// This is the BATCHED DECODE epilogue; prefill rides
    /// `qkv_norm_rope_batch5` + `kv_nra_rows3`. A model whose graph disagrees
    /// with the old constants therefore came out right on the prompt and wrong
    /// on every generated token - muse-glimmer's full-attention layers are
    /// NoPE (`freq_scale` 0), and with no `freq_scale` argument at all this
    /// kernel re-roped them once per decode step.
    /// `(qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions, slots,
    ///   factors, block_tables, bps, n_head, n_kv, head_dim, max_ctx, batch,
    ///   eps, theta_scale, kv_dtype, qkv_stride, freq_scale, neox, vnorm,
    ///   stream)`
    #[allow(clippy::type_complexity)]
    pub gemma_qkv_nra3: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            f32,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 369: `softmax(QK^T)` over the encoder frames for NOMINATED
    /// cross-attention heads - the read-out word-level timing is derived from
    /// `out` is `[batch, n_sel, n_enc]` and each row sums to 1.
    ///
    /// Separate from `whisper_dec_attn` deliberately: that one is flash-style and
    /// consumes the probabilities inside its online loop without ever
    /// materialising them, which is right for the hot path. Word timing is
    /// opt-in and runs off it.
    /// `(q, qbias, k, slots, heads, out, kv_stride, n_enc, n_heads, hd, n_sel,
    ///   batch, scale, kv_dtype, stream)`
    pub whisper_xattn_probs: Option<WhisperXattnProbsFn>,
    /// 370: `rope2d_neox` plus the pair layout it hardcoded. The vision
    /// towers disagree on it and no GGUF key states it - gemma4v's ropes NEOX
    /// (partner `i + hd/4`), muse-glimmer's ropes NORM (partner `2i+1`), which
    /// is the `mode` argument in each one's reference clip graph.
    /// `(x, pos_x, pos_y, n_tokens, n_heads, head_dim, theta_scale, neox,
    ///   stream)`
    pub rope2d: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 371: pixel-shuffle merge, `out[o][c*k + s] = src[idx[o*k + s]][c]`.
    /// Neither `gather_rows_avg` (which pools) nor a plain k-row concat (which
    /// is spatial-outer) - muse-glimmer's downsampler is channel-outer.
    /// `(src, idx, out, rows, k, width, stream)`
    pub pixel_shuffle_rows: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    ///  (slot 374) dim-major twin V pool sync:
    /// `(pool, vdim, positions, slots, block_tables, blocks_per_slot,
    ///   kv_dim, rows, stream)`
    pub vdim_sync: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 375: registers the vdim twin pool base for the VD launcher
    pub vdim_register: Option<unsafe extern "C" fn(*mut core::ffi::c_void) -> i32>,
    /// slot 376: arms the batched-runs prefill attention for one pass
    /// (device u32 prefix array [n_runs+1], n_runs, max run rows); null
    /// disarms
    pub pf_runs_register: Option<unsafe extern "C" fn(*const core::ffi::c_void, u32, u32) -> i32>,
    /// slot 378: bf16-in whole-row glu2 quantize (gu_bf16, q, rscale, n_ff,
    /// rows, act 0=gelu 1=silu, stream)
    pub quantize_e4m3_glu2_row_b16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 379: bf16 -> e4m3 + F32 per-row scale (bf16, f8_data, row_scale,
    /// in_dim, out_dim, stream). Same absmax/exponent pick and the same
    /// `rscale[row] = 2^e` convention as `q8_0_to_f8row`, so it emits the
    /// identical `F8RowPlane` shape from a bf16 source instead of a Q8 one -
    /// which is what lets a bf16 lm_head reach the f8t tile route.
    pub bf16_to_f8row: Option<Bf16ToF8rFn>,
    /// slot 380: SAM ViTDet attention with the decomposed relative-position
    /// bias - `softmax(q·kᵀ·scale + rel_h + rel_w)·v`, where the bias is the
    /// query contracted against per-axis [side, side, hd] tables the host
    /// prepared per geometry (DeepSeek-OCR's first tower).
    /// q/k/v/out are `[n_batch, side², heads, hd]` f32; the tables are shared
    /// by every batch element. -3 = shape not covered (hd or side over 64).
    pub sam_attn: Option<SamAttnFn>,
    /// slot 381: DeepSeek-greedy MoE router epilogue  - the same
    /// top-k selection as `moe_topk_batch`, but the weights are the full
    /// softmax probabilities (denominator over all `n_expert` logits, no
    /// renormalization among the selected k). The two classes pick the same
    /// experts and differ by the top-k's captured probability mass - silently
    /// wrong if conflated. `(logits [batch, n_expert], n_expert, k, out_idx,
    /// out_w, batch, stream)`; -3 when n_expert > 256 or k > 16.
    pub moe_topk_softmax_all: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> KernelStatus,
    >,
    /// slot 382: fused single-pass GQA-16 decode attention (the
    /// trtllm-gen chase): one CTA per (kv-head, row), the whole windowed K/V
    /// run staged to shared once, 16 warps = 16 q-heads, online softmax in
    /// 32-token score tiles, FINAL output written in-kernel with the sink
    /// folded - no partial planes, no combine launch, 1/G the pool traffic.
    /// fp8-e4m3 paged KV, head_dim 128, group 16 only (rc -2 otherwise; rc
    /// -3 when the windowed context exceeds the shared-memory opt-in).
    /// Params = `AttnDecodeBatchPagedFn` + `pos_max: u32` after `batch` (max
    /// position over the rows - a HOST-side hint that sizes shared memory;
    /// callers pass the kv_split_band ceiling so captured graphs stay valid
    /// across the band).
    pub attn_decode_fused_gqa16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 383: in-house f16xf16->f32 tensor-core dense GEMM (
    /// PADDOCK_INHOUSE_F16 cuBLAS removal). (w, x, y, beta, in, out, batch, stream)
    pub f16_gemm: Option<F16GemmFn>,
    /// slot 384: ring twin of `rope_norm_qk_append_paged` (
    /// deepseek-ocr) - rope turns by the true position stream while the K/V
    /// appends land at the R-SWA ring's WRITE stream, and `neox` picks the
    /// rope pair layout. (q, k, v, pool_k, pool_v, positions, wpos, slots,
    /// block_tables, bps, n_heads, n_kv, head_dim, rope×6, batch, neox,
    /// kv_dtype, stream)
    pub rope_qk_append_paged_ring: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 385: residual-add + rmsnorm + Q8_0 quantize  - the
    /// dp4a-class sibling of `add_rmsnorm_quant_e4m3_batch`. x += proj is
    /// written back; out keeps the f32 plane (router), q/qs get the int8 +
    /// per-32 scales. (x, proj, w, out, q, qs, n, eps, batch, stream)
    pub add_rmsnorm_quant_q8_batch: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 386: SwiGLU + Q8_0 quantize in one pass  - the
    /// activation is computed in registers (gate stays unmodified) and
    /// quantized with `quantize_q8`'s exact warp-per-32-block math.
    /// (gate, up, q, scale, n, stream)
    pub swiglu_quant_q8: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 387: OCR tower patch stem  - u8 interleaved-RGB views
    /// to normalized f16 patch rows in the SAM conv stem's im2row order, one
    /// gather. Bit-identical to host normalize_rgb8 + im2row + the f32->f16
    /// convert (IEEE f32 divisions, RNE f16), at a quarter of the PCIe bytes.
    /// (pixels, out, mean0, mean1, mean2, std0, std1, std2, views, px, patch,
    /// stream)
    pub ocr_patches_u8: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 388: split the whisper encoder's FUSED q|k|v GEMM landing into
    /// the three planes attention consumes, q/v biases folded (k has none -
    /// architecture). Exists so the encoder's three per-layer projections
    /// run as one M=3d tc5p GEMM (3x12.60us -> 19.09 at 1500x1280)
    /// and the two full-width bias_add launches disappear into this one.
    /// (qkv, bq|null, bv|null, q, k, v, d, rows, stream)
    pub whisper_enc_qkv_split: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 389: cross-K/V store off a LAYER-BATCHED [rows, n_layer*d]
    /// landing: every decoder layer's cross K (or V) projection
    /// reads the same encoder states, so the runner concatenates the weight
    /// planes into one M=n_layer*d GEMM and this stores every layer's slot
    /// plane in one launch. `dsts` is a device array of n_layer plane base
    /// pointers (uploaded once at pool alloc - capture-safe); `bias` the
    /// concatenated [n_layer*d] plane (V) or null (K).
    /// (src, bias|null, dsts, slots, rows, d, n_layer, stride, kv_dtype, stream)
    pub whisper_kv_store_batch: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 390: decode-band multi-row bf16 GEMV (2 <= batch <= 8).
    /// The bf16 tile GEMM collapses to out_dim/64 blocks at decode widths -
    /// this is its GEMV-shaped twin, same params as `bf16_gemm_f32`.
    pub bf16_gemv_mr_f32: Option<Bf16GemmF32Fn>,
    /// slot 391: bf16 tensor-core prefill GEMM (batch > 8),
    /// f32 activations cast to bf16 in the smem stage - the parity
    /// reference's own batched-BF16 class. Same params as `bf16_gemm_f32`;
    /// rc -2 outside the band (caller keeps the f32-FMA tile).
    pub bf16_gemm_mma: Option<Bf16GemmF32Fn>,
    /// slot 392: LayerNorm writing f16 directly (the GEMM staging
    /// dtype) - kills the LN->convert round-trip. Params = `layernorm`, out
    /// is a f16 plane. Bit-identical to LN followed by convert_f32_f16.
    pub layernorm_f16: Option<LayernormFn>,
    /// slot 393: bias + tanh-GELU + f16 store in one pass -
    /// replaces bias_add -> gelu -> convert_f32_f16 on the tower FFN plane.
    pub gelu_bias_f16: Option<GeluBiasF16Fn>,
    /// slot 394: erf twin of 393 (the projector FFN - the two
    /// GELUs must never be swapped, see `gelu_erf`).
    pub gelu_erf_bias_f16: Option<GeluBiasF16Fn>,
    /// slot 395: residual + projection bias,
    /// `x[r][i] += src[r][i] + bias[i]` - replaces bias_add -> add. src stays
    /// unbiased in memory.
    pub add_bias_res: Option<AddBiasResFn>,
    /// slot 396: `mrope_vision` with the q/k projection bias
    /// folded into the load - the pair walk touches every head element once,
    /// so `x = rope(x + b)` lands in one pass.
    pub mrope_vision_bias: Option<MropeVisionBiasFn>,
    /// slot 397: modelopt NVFP4 checkpoint dequant - the
    /// tensor-level oracle over the shipped triple (adjacent e2m1 nibbles,
    /// e4m3 per-16 scales, per-tensor f32 scale2). Debug/oracle only;
    /// serving consumers read the packed plane directly.
    pub nvf4_dequant: Option<Nvf4DequantFn>,
    /// slot 398: W4A16-class GEMV over a checkpoint NVFP4 plane
    /// (f32 activations, scale2 folded once after the reduction).
    pub nvf4_gemv: Option<Nvf4GemvFn>,
    /// slot 399: GEMV over a checkpoint FP8 plane held as e4m3
    /// bytes + one f32 scale per output row (nemotron mamba in/out_proj -
    /// the per-tensor weight_scale broadcast into the row array, byte-exact).
    pub f8r_gemv: Option<F8rGemvFn>,
    /// slot 400: mamba-2 decode conv step - conv_step + bias
    /// before the SiLU (nemotron's conv1d carries bias).
    pub mamba_conv_step: Option<MambaConvStepFn>,
    /// slot 401: sequential mamba-2 SSD scan over a token span,
    /// state `[H, head_dim, d_state]` f32 register-resident, repeat_interleave
    /// group broadcast, softplus dt, D-skip fused.
    pub mamba2_scan_seq: Option<Mamba2ScanSeqFn>,
    /// slot 402: grouped gated RMSNorm (Mixer2RMSNormGated) -
    /// gate first in f32, per-group variance, per-channel weight.
    pub mamba_rmsnorm_gated_g: Option<MambaRmsnormGatedGFn>,
    /// slot 403: token-batched NVFP4 MoE expert up GEMV + fused
    /// squared-relu; per-expert scale2 array. cc-gated with 397-398.
    pub nvf4_moe_up_relu2: Option<Nvf4MoeUpRelu2Fn>,
    /// slot 404: token-batched NVFP4 MoE expert down GEMV with
    /// deterministic k-slot combine and accumulate flag. cc-gated with 397-398.
    pub nvf4_moe_down_acc: Option<Nvf4MoeDownAccFn>,
    /// slot 405: slot-389's store off an AUDIO-MAJOR batched
    /// landing - row r lands in `slots[r / rows_per_slot]` at row
    /// `r % rows_per_slot`. (src, bias|null, dsts, slots, rows, d, n_layer,
    /// stride, kv_dtype, rows_per_slot, stream)
    pub whisper_kv_store_slots: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 406: bulk mamba-2 conv over a token span - the step
    /// kernel's math with the window carried across T tokens in one launch;
    /// bit-exact vs T serial steps.
    pub mamba_conv_seq: Option<MambaConvSeqFn>,
    /// slot 407: sorted-tile NVFP4 expert up +
    /// squared-relu over the moe_align BM=32 layout, re-quantized to nvf4
    /// in registers (fq/fs sorted-position indexed - slot 408's direct B
    /// input). The mxf4nvf4 MMA class: W4A4 at prefill; decode stays on
    /// slot 403's W4A16 GEMV.
    pub nvf4_moe_up_relu2_bs: Option<Nvf4MoeUpRelu2BsFn>,
    /// slot 408: sorted-tile NVFP4 expert down ->
    /// weighted per-(token, slot) f32 partials at
    /// `part[(tok*np + slt + slot_off) * embd]`; fold with
    /// `moe_slot_combine` (fixed slot order). `topk_w` NULL means 1.0 (the
    /// shared-expert pass); `kw` is topk_w's row stride.
    pub nvf4_moe_down_bs: Option<Nvf4MoeDownBsFn>,
    /// slot 409 (decode rung): decode multi-task NVFP4 expert up
    /// + squared-relu - one wave-dense launch over all k routed slots AND
    ///   the shared expert (`act` layout `[k*ff_r | ff_s]`); per-row math and
    ///   layout verbatim from slot 403 (rows bit-identical, fused grid).
    pub nvf4_moe_up_relu2_mt: Option<Nvf4MoeUpRelu2MtFn>,
    /// slot 410 (decode rung): decode slot-split NVFP4 expert
    /// down -> pre-weighted f32 partials at `part[slot*embd + r]` (shared
    /// expert = slot k), each slot plane bit-identical to a k=1 slot-404
    /// launch; fold with `moe_slot_combine` (fixed ascending slot order).
    pub nvf4_moe_down_part: Option<Nvf4MoeDownPartFn>,
    /// slot 411: capture-time f16 mmaf election gate - `0`
    /// declines the mmaf fine-tile arm at `pd_f16_gemm` dispatch (election
    /// falls through to GEMV/tc5g), `1` restores it. Read at launch/capture
    /// time, so it bakes into any graph captured while set - whisper's
    /// overlap routing captures its mmaf-off decode variant behind this
    /// (mmaf × tc5p is the one overlap-poison pairing). Returns 0.
    pub f16_mmaf_set: Option<F16MmafSetFn>,
    /// slot 412 (stage A): batched single-token conv step
    /// over a slot arena of windows (`[n_slots, k-1, conv_dim]`) - row r of
    /// x (stride/offset like conv_seq) advances slot `slots[r]`. Bit-exact
    /// per row vs `mamba_conv_step`.
    pub mamba_conv_step_batch: Option<MambaConvStepBatchFn>,
    /// slot 413 (stage A): batched single-token SSD scan
    /// step over a slot arena of states (`[n_slots, H, hd, S]`). Bit-exact
    /// per row vs `mamba2_scan_seq` at `n_tokens = 1`.
    pub mamba2_scan_step_batch: Option<Mamba2ScanStepBatchFn>,
    /// slot 414 (stage A): row-batched nvf4 GEMV twin
    /// (x `[B, in]`, y `[B, out]`, grid.y = B). Bit-exact per row vs
    /// `nvf4_gemv`.
    pub nvf4_gemv_batch: Option<Nvf4GemvBatchFn>,
    /// slot 415: token-batched Q8_0 single-plane expert up +
    /// squared-relu (nemotron_h_moe - no gate matrix). Same dp4a class and
    /// grid as `q8_0_moe_gate_up` with half the weight streams; serves the
    /// shared expert as a 1-expert plane with a zero idx. `(up_data,
    /// up_scale, idx, xq, xs, out, in_dim, ff, n_active, batch, stream)`.
    pub q8_0_moe_up_relu2: Option<Q8MoeUpRelu2Fn>,
    /// slot 416: sorted twin over the moe_align layout,
    /// K-tail-guarded - in_dim % 32 (nemotron's 2688/1856/3712 dims are not
    /// 256-aligned; out-of-range words/scales stage as zeros). `(up_data,
    /// up_scale, sorted_row, block_expert, xq, xs, fused, in_dim, ff,
    /// max_blocks, stream)`.
    pub q8_0_moe_up_relu2_sorted: Option<Q8MoeUpRelu2SortedFn>,
    /// slot 417 (spec core): `mamba2_scan_seq` twin writing a
    /// per-row state snapshot (`snap[t] = state after row t`) - the spec
    /// verify's rollback source; the walk itself is bit-identical to the
    /// plain kernel. `(state, xbc, dt_raw, dt_stride, A, D, dt_bias, y,
    /// snap, n_tokens, n_heads, head_dim, d_state, n_groups, stream)`.
    pub mamba2_scan_seq_snap: Option<Mamba2ScanSeqSnapFn>,
    /// slot 418: strided-rows copy `dst[r] = src[src_off + r*src_stride..]`
    /// (f32 elements) - conv-input snapshots for the verify commit.
    pub copy_rows_strided: Option<CopyRowsStridedFn>,
    /// slot 419: multi-row W4A16 nvf4 GEMM - `nvf4_gemv_batch`'s
    /// signature and per-row math verbatim, but each weight fragment is
    /// decoded once per 16-row group (the plane streams ceil(batch/16)
    /// times, not `batch` times). Bit-exact per row vs `nvf4_gemv_batch`.
    pub nvf4_gemm_mr: Option<Nvf4GemvBatchFn>,
    /// slot 420: `moe_topk_sigmoid_batch` writing k+ns-wide
    /// rows - lanes k.. append the shared pseudo-expert ids `sh0..sh0+ns`
    /// with weight 1.0, so one moe_align covers routed + shared (the
    /// shared-expert fold-in).
    pub moe_topk_sigmoid_batch_sh: Option<MoeTopkSigmoidBatchShFn>,
    /// slot 421: `gemma_qkv_nra3` twin whose q/k/v GEMM planes are
    /// PACKED bf16 (the b16-D election's p16 convention: same element
    /// indexing, half the bytes). `q_out` stays f32 and the KV appends are
    /// unchanged - only the read side differs. Same argument list as
    /// `gemma_qkv_nra3`; byte offsets into the planes are the CALLER's to
    /// halve.
    #[allow(clippy::type_complexity)]
    pub gemma_qkv_nra3_b16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            f32,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 422: tensor-core NVFP4 GEMM -
    /// `nvf4_gemv_batch`'s signature, exact-dequant bf16 weights fed to
    /// m16n8k16 mma. The batched lm_head class: the scalar mr kernel is
    /// issue-bound at ~1.15 ms at the vocab shape, this one is
    /// stream/tensor-bound. Not bit-comparable to 414/419 (bf16 activation
    /// cast + mma reassociation) - callers elect it explicitly.
    pub nvf4_gemm_tc: Option<Nvf4GemvBatchFn>,
    /// 423: `attn_spec_batch_fin` twin that quantizes the
    /// finalized rows in-kernel - arg 4 receives the e4m3 plane
    /// `[rows x n_heads*head_dim]`, arg 5 the f32 per-row scales (fin's
    /// dead ml slot). Bit-identical to fin + `quantize_e4m3_row` on the
    /// same rows. Returns -2 when the geometry isn't the whole-row-CTA
    /// SWA verify shape - the caller keeps the f32 fin + row quantize.
    #[allow(clippy::type_complexity)]
    pub attn_spec_batch_fin_e4: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 424 (thin-k/v rung): fused q|k|v decode-band bf16
    /// GEMM - one launch over the load-time-concatenated `[q;k;v]` plane
    /// against the shared x, segmented store into the three separate y
    /// planes. The thin k/v planes (256 rows) are latency-starved on their
    /// own grids (~40 us floor at 1-3% of roof, any kernel shape); riding
    /// the fused grid is the fix. Per out-row bit-identical to the plain
    /// mma on that segment (the k-walk is config-independent). Args:
    /// (w, x, yq, yk, yv, in_dim, oq, okv, batch, stream); no bias (the
    /// nemotron attn planes carry none).
    pub bf16_qkv_gemm_mma: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Slot 425: fin twin storing the finalized rows as
    /// e4m3 at static scale 1.0 into the wo-in quantized plane (arg 4 =
    /// pf_e4q); the GEMM's xrs is a ones vector. Same accept envelope as
    /// `attn_spec_batch_fin`; -2 = caller keeps the fin + quantize chain.
    pub attn_spec_batch_fin_e4s: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,

    /// Slot 426: checkpoint-plane W4A4 GEMM - `mxfp4_gemm_nv4`'s
    /// fp4 x fp4 block-scale mma with the `Nvf4Plane` epilogue (acc*scale2,
    /// +bias when present). `(data, scale, bias, xq, xs, y, scale2, in_dim,
    /// out_dim, batch, stream)`; xq/xs from `quantize_nvf4[_swiglu]`.
    /// cc-gated with the block-scale family.
    pub nvf4_gemm_f4: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,

    /// Slot 427: v2 of 426 - both scale planes on
    /// cp.async, one barrier per K-step, ring depth `st` (probe elected 3).
    /// Same signature plus `st` before stream. Requires in_dim % 128 == 0.
    pub nvf4_gemm_f4b: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,

    /// Slot 428: split-K twin for machine-starved tile
    /// grids. `(data, scale, bias, xq, xs, part, y, scale2, in_dim,
    /// out_dim, batch, sk, stream)`; `part` holds sk*batch*out_dim f32 raw
    /// partials; a deterministic two-pass reduce owns the epilogue.
    pub nvf4_gemm_f4s: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,

    /// Slot 429: the KC=256 arm - half the barriers, 2x
    /// per-stage flight, one 16 B scale cp.async per row. Same signature as
    /// 428; sk=1 runs unsplit (part unused). Requires in_dim % 256 == 0;
    /// bit-exact vs 426/427.
    pub nvf4_gemm_f4c: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,

    /// Slot 430: TMA + mbarrier ring (8 consumer + 4
    /// producer warps, 4 tensor maps over the plain layouts) - the prefill
    /// band. (data, scale, bias, xq, xs, y, scale2, in, out, batch, stream);
    /// in_dim % 256 == 0; bit-exact vs 426/427/429.
    pub nvf4_gemm_f4t: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,

    /// Slot 431: tcgen05/TMEM decode attention - FINAL
    /// output (batch-major rows in `out`, no partials/out_ml; the caller
    /// skips the combine when rc == 0). Params = `AttnDecodeBatchPagedFn`
    /// (`sinks` accepted and ignored - gemma's -inf sinks are a no-op).
    /// fp8-e4m3 paged KV, head_dim 256, group 2, swa_window > 0 only:
    /// rc -2 = shape/arch not covered, rc -3 = smem over the opt-in - the
    /// caller keeps the partial+combine route on any nonzero rc.
    pub attn_decode_tc5_paged: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 434: device top-K prefilter for HOST-HEAD sampling rows.
    /// `(logits, params [rows × PdSampleRow - mode 4 rows are selected],
    /// out [rows × k × 2 u32 = (token id, raw-logit f32 bits)], rows, n,
    /// k, stream)`. The host runs its existing nucleus pipeline over the
    /// K-head instead of reading the full vocab row back (qwen3.x's
    /// top-k/top-p defaults made every c32 round pay 21.3 ms of host
    /// sampling on a B200). k ≤ 64; rc -2 = k range.
    pub topk_rows: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 435: full-device truncation sampling -
    /// `(logits, params [PdSampleRow - mode 5 rows are sampled], trunc
    /// params [rows × {k u32, top_p f32, min_p f32, pad}], out [rows token
    /// ids], rows, n, stream)`. The host `sample_trunc_head` pipeline runs
    /// on device (same distribution class as mode 2: expf ulps may flip a
    /// cum-boundary pick; the seed->token mapping is not a contract). Rows
    /// carrying mode 5 are zero-host - pipe/overlap admissible.
    pub sample_rows_t: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 436: GENERAL truncation sampling - mode-6 rows,
    /// same signature and planes as slot 435 but no top-k bound: top-p
    /// only (nemotron's published profile is temperature 1.0 / top_p 0.95
    /// with no top_k), min-p only, and combinations sample exactly on
    /// device via a histogram quantile walk (`build_nucleus` top_k==0
    /// semantics; expf-ulp cum boundaries are the mode-2 class).
    pub sample_rows_p: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 437: DECODE gated-delta recurrence with the split +
    /// qk-L2-norm fused in - `(conv [B, conv_dim], g [B,HV], beta [B,HV],
    /// slots, states, out [B,HV,D], batch, n_k_heads, n_v_heads,
    /// head_dim(=128), stream)`. Byte-identical STATE evolution vs
    /// split_gqa_norm + recurrent_v2 at n_tokens=1/no-snap (readout is the
    /// documented 1-ulp reassociation class); kills one kernel + the
    /// dq/dk/dv plane round trip per GDN layer per decode round.
    pub gated_delta_recurrent_v2f: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 438: `pd_conv_step_slots` with x_new read STRIDED out
    /// of the DN in-proj fused plane - `(wins, x_new, w, out, slots, batch,
    /// conv_dim, k, x_stride, stream)`; bit-identical to slice-then-conv.
    pub conv_step_slots_s: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 439: `pd_gated_rmsnorm` with z read STRIDED out of the
    /// fused plane - `(x, z, weight, out, n_rows, d, eps, z_stride,
    /// z_rows_per_b, stream)`; z element (r, j) lives at
    /// `(r / rpb) * z_stride + (r % rpb) * d + j`. Bit-identical.
    pub gated_rmsnorm_s: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            f32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 440: the v2f recurrence with g/beta computed IN-KERNEL
    /// from the fused plane's alpha/beta columns (row_slice2_gate's
    /// expressions verbatim) - `(conv, fused, ab_off, fused_stride, ssm_a,
    /// dt_bias, slots, states, out, batch, n_k_heads, n_v_heads,
    /// head_dim(=128), stream)`. With 438/439 the slice launch and the
    /// g/beta/mixed/z planes all disappear at decode.
    pub gated_delta_recurrent_v2f_g: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 441: VL conv+silu+qkv over the wave pass's packed fresh
    /// spans - `(x, w, row0s, q, k, v, n_rows, n_k_heads, n_v_heads, s, k,
    /// stream)`; `row0s` is a per-row u32 span-start plane. Each row's adds
    /// are identical in value and order to its per-span offset launch
    /// (bit-exact by construction); resumed spans keep the copy path.
    pub causal_conv1d_silu_qkv_vl: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 442 (glue rung): residual-add + rmsnorm + nvf4
    /// quantize in one row-per-CTA launch - `(x, proj, w, out, q, scale, n,
    /// eps, batch, stream)`. Replaces `add_inplace + rmsnorm_batch +
    /// quantize_nvf4`, which nemotron runs 23x per decode tick (every MoE
    /// layer's prologue). Still writes the f32 normed row: the router reads
    /// it. Bit-exact to that chain - same f64 reduction at the same nth,
    /// same `v * inv * w`, same `pd_nvf4_quant8`.
    pub add_rmsnorm_quant_nvf4_batch: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 443 (scan rung): seq SSD walk over an f16 state arena
    /// - `(state, xbc, dt_raw, dt_stride, A, D, dt_bias, y, n_tokens,
    /// n_heads, head_dim, d_state, n_groups, stream)`. State is STORED f16
    ///   and computed f32 (the numeric class of vLLM's
    ///   `--mamba-ssm-cache-dtype float16`). The walk keeps state register-resident for the
    ///   whole span, so its per-token arithmetic is bit-identical to the f32
    ///   twin; only the hand-off between launches rounds.
    pub mamba2_scan_seq_f16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 444 (scan rung): seq walk + per-row snapshots, f16
    /// state AND f16 snap blob - a partial spec accept rolls back by flat-
    /// copying a snap row over the live state, so both must share a
    /// representation or the rollback re-rounds.
    pub mamba2_scan_seq_snap_f16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 445 (scan rung): batched decode step over an f16 state
    /// arena, `__half2`-paired along `i`. The pairing is load-bearing, not an
    /// optimization: a warp's lanes cover consecutive `i`, so a naive
    /// `__half` port halves the transaction to 64 B and measures below the
    /// f32 kernel's bandwidth (1337 vs 1473 GB/s); paired it reads 1457 and
    /// the f16 class comes out ahead. head_dim is pinned to 64.
    pub mamba2_scan_step_batch_f16: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 446: widen an f16 SSM state region to f32 - `(src, dst, n, stream)`.
    /// The prefix-cache checkpoint pool, its staging blobs and the radix
    /// accounting all key off an f32 `[state | win]` layout, which is the
    /// correctness-critical restore path; the f16 class converts at the arena
    /// boundary rather than re-laying it out. Exact.
    pub ssm_state_widen: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 447: narrow an f32 checkpoint region back to f16. Lossless in
    /// practice because it only ever sees values that originated as f16, so
    /// widen->narrow round-trips bit-for-bit.
    pub ssm_state_narrow: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 448 (QKC pair): `causal_conv1d_silu_qkv_vl` twin
    /// whose q/k outputs are COMPACT bf16 `[rows, n_k_heads, s]` planes
    /// (one bf16 round of the same f32 - the consumer used to round them
    /// itself, so the pipeline stays bit-identical). v stays f32 expanded.
    /// Must be paired with `gated_delta_chunked_rs_vl_qkc` by the caller.
    pub causal_conv1d_silu_qkv_vl_qkc: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 449 (QKC pair): `gated_delta_chunked_rs_vl` reading the compact
    /// bf16 q/k planes; extra `n_k_heads` u32 before `head_dim`. Fails
    /// NotSupported when mispaired (the stage1-rs route must be live).
    pub gated_delta_chunked_rs_vl_qkc: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 450: single-plane relu^2 expert up in the dec2 class (warp per
    /// output row, `rows_pb` rows per CTA). The decode-band shape for
    /// nemotron_h_moe - the sorted tile below it pads 32-row blocks that hold
    /// one real row at batch widths, and the token-batched dp4a original
    /// spends a whole CTA per output element. Down reuses `q8_0_moe_dn_dec2`.
    /// REORDER class vs `q8_0_moe_up_relu2` (per-lane partials, not
    /// per-thread), which is why it is its own slot.
    pub q8_0_moe_up_relu2_dec2: Option<Q8MoeUpRelu2Dec2Fn>,
    /// slot 451: `quantize_q8` with relu(x)^2 folded in front of the per-32
    /// amax. Lets a squared-relu dense FFN (nemotron's shared expert) put its
    /// up plane on the plain q8 GEMM ladder instead of an expert kernel, and
    /// still hand the down plane a correctly quantized activation.
    /// Bit-identical to relu^2-into-f32 followed by `quantize_q8`.
    pub quantize_q8_relu2: Option<QuantizeQ8Fn>,
    /// slot 452 (lm_head repack rung): `nvf4_gemv_batch` reading
    /// the TILE-MAJOR plane layout - `[row_tile 128][k_stage 128][row]`,
    /// weights and scale records each contiguous per (tile, stage) block,
    /// out padded to 128 rows and zero-filled, `in_dim % 128 == 0`.
    /// Bit-exact vs slot 414 on the same logical plane.
    pub nvf4_gemv_batch_tm: Option<Nvf4GemvBatchFn>,
    /// slot 453: `nvf4_gemm_mr`'s tile-major twin - bit-exact vs slot 419.
    pub nvf4_gemm_mr_tm: Option<Nvf4GemvBatchFn>,
    /// slot 454: the tensor-core head class over a tile-major plane (the
    /// persistent tcp arm's REPK instantiation; each stage cp.asyncs one
    /// sequential 10.25 KB block instead of 128 rows at 1344 B stride -
    /// measured 225 -> 205 us b32 / 180 us b8). Bit-exact vs slot 422's tcp
    /// arm on the same logical plane; same numeric class.
    pub nvf4_gemm_tc_tm: Option<Nvf4GemvBatchFn>,
    /// slot 455 (fragment rung): `nvf4_gemv_batch` over the
    /// FRAGMENT layout - the tile-major blocks additionally permuted to
    /// `[w][k16][g][u32 of a0..a3 mma-fragment bytes per lane]` (scales
    /// stay tile-major). One CTA per 16-row group, each warp a (g, g+8)
    /// row pair sharing every weight u64 and x float4. Bit-exact per row
    /// vs slot 414.
    pub nvf4_gemv_batch_tf: Option<Nvf4GemvBatchFn>,
    /// slot 456: `nvf4_gemm_mr`'s fragment-layout twin - bit-exact vs 419.
    pub nvf4_gemm_mr_tf: Option<Nvf4GemvBatchFn>,
    /// slot 457: the tensor-core head over a fragment plane - the tcv arm
    /// (bf16-table decode + B-once ldmatrix tile, 2 CTA/SM) reading one
    /// conflict-free LDS.32 per fragment group with flat 8 KB stages.
    /// Probe: 167.0 -> 159.2 us b32 / 144.2 b8 (marlin parity on the
    /// kernel). Bit-exact vs slot 422's class on the same logical plane.
    pub nvf4_gemm_tc_tf: Option<Nvf4GemvBatchFn>,
    /// slot 458: Q16xKv128 tensor-core decode attention for the muse
    /// hd128/G16 geometry. At G=16 the sixteen q-heads sharing a kv-head are
    /// an M=16 MMA dimension, so decode attention is two m16n8k16 GEMMs
    /// instead of a gemv with 16-fold redundant KV traffic plus a separate
    /// full-die combine. FINAL output with the sink folded in - the caller
    /// must not run a combine. 5.35x the shipped vec8 splits=2 + combine at
    /// B=32/ctx256 and ahead at every rung including B=1, so no row gate.
    /// Params = `attn_decode_fused_gqa16` minus `pos_max`: this arm chunks
    /// the KV walk (KVT=128), so shared memory is constant in context and it
    /// carries no pos_max ceiling. rc -2 on a non-muse shape or cc < 8.
    pub attn_decode_fmha16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 459: DFlash2's grouped dynamic convolution - the depthwise
    /// token-axis conv that wraps each drafter sublayer. Coefficients are a
    /// per-channel static `base` plus a per-token, per-GROUP `delta` that a
    /// projection GEMM predicts from the sublayer input; `side` selects the
    /// before/after half of both, since one projection feeds both wraps.
    ///
    /// `(h, out, base, delta, side, embd, taps, num_groups, group_size,
    /// rows, r, stream)`. `rows` is the RUNTIME block length (k+1), not the
    /// trained `block_size`: the tap mask is `row % rows`, which is what
    /// stops a tap reaching from one slot's block into another's. `out` must
    /// not alias `h` (it reads row-1 while writing row). rc -2 when embd or
    /// group_size is not a multiple of 4, or the groups do not tile embd.
    pub dflash_conv: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 460: unpack `topk_rows`' interleaved (id, logit-bits) pairs into
    /// the flat u32 id plane `kquant_gather` consumes, appending each block's
    /// anchor token at the tail so one gather serves candidates and anchors.
    /// `(topk, toks, ids, k, rows, r, vocab, stream)`.
    pub dflash_cand_ids: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 461: DFlash2's candidate-selector walk. Greedy forward pass (not
    /// Viterbi) over `edge[p][c] = unary[c] + sum_r pred[p][r]*hidden[r]*succ[c][r]`,
    /// one CTA per block, carrying the chosen index forward - so only one row
    /// of the KxK matrix is ever materialised.
    ///
    /// `(topk, pred, succ, hs, out, scale, cap, rank, k, rows, r, stream)`.
    /// `scale`/`cap` are the DRAFTER's own logit epilogue and are applied to
    /// the unary term here: it is added to a bilinear score, so the monotone
    /// argument that lets greedy per-row drafting skip the epilogue does not
    /// hold. rc -2 when k > 64 or rank is not a multiple of 32.
    pub dflash_select: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            f32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 462: spec-verify twin of `gated_delta_recurrent_v2` - same math
    /// and identical `out[]` values, but no per-token snapshots and no final
    /// state writeback: the live state stays at ROUND-START so the commit
    /// can recompute forward over just the accepted prefix (slot 463).
    ///
    /// `(q, k, v, g, beta, slots, states, out, batch, n_tokens, n_heads,
    /// head_dim, stream)` - `states` is read-only here.
    pub gated_delta_verify_hold: Option<GatedDeltaVerifyHoldFn>,
    /// slot 463: commit-time recompute - re-runs the recurrence from the
    /// round-start state over each row's accepted prefix (`committed[b]`,
    /// device-staged, capture-safe) on the stashed split/gate planes, then
    /// writes the state back once. Bit-exact vs the per-token snapshot the
    /// old `state_restore_slots` path picked (same fixed f32 op order on
    /// the same inputs). `committed[b] == 0` leaves the state untouched.
    ///
    /// `(k, v, g, beta, slots, committed, states, batch, n_tokens, n_heads,
    /// head_dim, stream)`.
    pub gated_delta_commit_walk: Option<GatedDeltaCommitWalkFn>,
    /// slot 464: dflash async round - copy the block-draft picks from the
    /// draft graph's row-major `d_out` (`[n, rows]`, pick j of block b at
    /// `b*rows + 1 + j`) into the MTP chain's i-major `d_draft` layout
    /// (`d_draft[i*n + b]`), device-side. Kills the per-round dtoh sync on
    /// the dflash draft->verify boundary (the armed-chain verify assembles
    /// tokens from `d_draft`; the host readback happens post-verify).
    ///
    /// `(out, draft, n, rows, k_use, stream)`.
    pub dflash_chain_picks: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 469: dflash conditioning fold (rung C) - the append
    /// that norms for the drafter ring. Per written row (`rows_w`, the
    /// flattened cut windows): k-norm (rmsnorm math verbatim, thread width
    /// elected from `norm_batch` exactly like the rmsnorm launcher, so the
    /// reduction grouping matches) + NEOX yarn rope + paged f16 K/V store.
    /// Pool bytes bit-identical to the norm -> rope -> 2×cuts kv_append
    /// chain it replaces (one launch per drafter layer instead of
    /// `2 + 2·cuts`).
    ///
    /// `(fk, fv, kw, pool_k, pool_v, rows_w, positions, slots,
    /// block_tables, blocks_per_slot, n_kv, head_dim, eps, theta_scale,
    /// freq_scale, corr_low, corr_high, ext_factor, mscale, nw,
    /// norm_batch, stream)`.
    pub dflash_cond_append: Option<DflashCondAppendFn>,
    /// slot 470: DFlash2 SAMPLED selector walk (rung G) - the
    /// greedy walk's twin with per-block `invt` (1/T, 0 = argmax) and a u32
    /// seed; writes the chosen token per row AND the row's K-way draft
    /// distribution `q16[row*k + c]` (one-hot on greedy blocks). The K
    /// candidates are the `dflash_cand_ids` plane, so q lives on K floats
    /// per row - the rejection-sampling resolve below reads both.
    ///
    /// `(topk, pred, succ, hs, invt, seeds, out, q16, scale, cap, rank, k,
    /// rows, r, stream)`.
    pub dflash_select_rs: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            f32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 471: canonical rejection-sampling verify resolve over the K
    /// draft candidates, TRUNCATION-aware (rung G). Rows planned mode 7
    /// (`PdSampleRow {inv_t, u1, 7, u2 bits}` + the mode-5 trunc plane):
    /// head + nucleus exactly as mode 5, accept the draft at row j+1 with
    /// probability min(1, p/q), residual max(p-q, 0) on reject; writes the
    /// resolved token into the sampled-ids plane so the accept-while-match
    /// walk consumes the round unchanged. Other modes untouched.
    ///
    /// `(logits, params, trunc, meta, toks, cand, q16, out, rows, n_blocks,
    /// k1, drows, k, n, stream)`.
    pub dflash_rs_resolve: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Slot 472: BM=8 skinny up over the TILED expert-plane layout.
    /// Same argument contract as `nvf4_moe_up_relu2_bs` with
    /// BM=8 strides (`sorted_row` is `[nb*8]`, fq/fs rows `nb*8`); blocks
    /// come from `moe_align_bm(bm=8)`. Bit-exact vs the bs pair on identical
    /// routing (same kt/k64 accumulate order).
    pub nvf4_moe_up_relu2_st: Option<Nvf4MoeUpRelu2BsFn>,
    /// Slot 473: BM=8 skinny down over the tiled layout (bs down contract at
    /// BM=8 strides).
    pub nvf4_moe_down_st: Option<Nvf4MoeDownBsFn>,
    /// Slot 474: BM=32 wide up over the tiled layout - the prefill twin of
    /// `nvf4_moe_up_relu2_bs`, same contract verbatim.
    pub nvf4_moe_up_relu2_stw: Option<Nvf4MoeUpRelu2BsFn>,
    /// Slot 475: BM=32 wide down over the tiled layout.
    pub nvf4_moe_down_stw: Option<Nvf4MoeDownBsFn>,
    /// Slot 476: r=1 mt-class up over the tiled layout (16-row-group CTAs;
    /// same numeric class and argument contract as `nvf4_moe_up_relu2_mt`,
    /// different grouping - gates are rel-to-rms + determinism, the mt rule).
    pub nvf4_moe_up_relu2_mtt: Option<Nvf4MoeUpRelu2MtFn>,
    /// Slot 477: r=1 mt-class down over the tiled layout.
    pub nvf4_moe_down_part_tt: Option<Nvf4MoeDownPartFn>,
    /// Capability marker: present iff the kquant family serves Q4_0 (GGUF raw
    /// id 2) - the QAT lineage's native format. The dtype rides the EXISTING
    /// kquant entry points, so their slot presence cannot answer; this one
    /// can. Pure append at the tail (slot 478).
    pub kquant_q40: Option<unsafe extern "C" fn() -> i32>,
    /// Slot 479: KV tier extent GATHER (kv-offload) - rearranges
    /// scattered paged-pool blocks into one page-first contiguous extent in
    /// device staging, so the PCIe leg is a single >=2 MiB DMA instead of
    /// per-page fragments (R1: 8 KiB fragments run at 5% of the bus). Slot
    /// presence is the engine's tier-transfer capability probe.
    pub kv_gather_blocks: Option<KvXferBlocksFn>,
    /// Slot 480: the restore-direction SCATTER twin (staging extent back
    /// into pool blocks).
    pub kv_scatter_blocks: Option<KvXferBlocksFn>,
    /// Slot 481: b=1 GEMV over the tile-linear (lin) boxes - the kernel that
    /// lets a plane class serve every width from one resident format, so the
    /// Q8_0 originals beside it can be reclaimed (non-KV-overhead R2.2).
    /// `part` needs nz*out_dim floats and may alias `y` when `ticket` is
    /// null (which pins nz=1); `ticket` is 2*ceil(out_dim/128) u32 zeroed
    /// once at allocation and owned by launches of one shape (its wrap value
    /// is that shape's elected nz).
    /// (wlin, x, part, y, ticket, in_dim, out_dim, stream)
    pub f8lin_gemv: Option<F8LinGemvFn>,
    /// Slot 482: granite's f32/Q8 residual fusion - `scale_add` +
    /// `rmsnorm_batch` + `quantize_q8_sums` in one launch, carrying the
    /// `residual_multiplier` that kept granite off both existing fused norms
    /// (neither has room for a scale). `proj` may be null for an entry norm
    /// that has no residual to fold. `n % 32 == 0`. Bit-exact against the
    /// three it replaces: the double-float sumsq is width-invariant, so the
    /// norm matches at any thread count, and the quantize half is a max
    /// reduce plus integer sums. Slot presence is the capability probe.
    pub add_rmsnorm_q8_xn: Option<AddRmsnormQ8XnFn>,
    /// Slot 492: v2 ring twin of `q8_0_moe_gate_up_mma_geglu` (S-stage
    /// cp.async ring + live-quarter skip).
    /// Bitwise on live outputs; `bm` must be 32 (NotSupported otherwise).
    pub q8_0_moe_gate_up_mma2_geglu: Option<Q8MoeGateUpMmaFn>,
    /// Slot 493: v2 ring twin of `q8_0_moe_down_mma`. `bm` must be 32 and
    /// ff % 64 == 0 (even n_blocks for the 4B async scale copies).
    pub q8_0_moe_down_mma2: Option<Q8MoeDownMmaFn>,
    /// Slot 485: write-out slot combine (`residual = sum(part)`, no read) -
    /// bitwise the memset_zeros + `moe_slot_combine` chain it replaces.
    pub moe_slot_combine_init: Option<MoeSlotCombineFn>,
    /// Slot 486: K-split decode router matvec. `scratch` holds 8 * batch *
    /// out_dim f32 partials (caller-owned; never allocated in the call -
    /// the decode graph bakes the launch). Deterministic ascending-split
    /// fold; new summation order vs `matvec_f32_batch` (token gates
    /// arbitrate). `(w, x, scratch, out, in_dim, out_dim, batch, stream)`
    pub matvec_f32_ks: Option<MatvecF32KsFn>,
    /// Slot 487: head+router+topk single-launch fusion - the head,
    /// the router GEMV (tile-matvec walk, bit-identical logits) and the
    /// scaled top-k in one kernel; rn never touches gmem.
    pub moe_head_router: Option<MoeHeadRouterFn>,
    /// Slot 488: v5 gate_up - the small-CTA geometry port (BM=16 token view
    /// over the bm32 CSR, 64-row both-mat slices, 128 threads). Bitwise
    /// the v2 pair on live outputs; bm must be 32.
    pub q8_0_moe_gate_up_mma3_geglu: Option<Q8MoeGateUpMmaFn>,
    /// Slot 489: bm128->bm32 pair map for the prefill dn hybrid.
    pub moe_pair_map: Option<MoePairMapFn>,
    /// Slot 490: q8 GEGLU remap quantize - f8s-gu f32 output into the bm32
    /// fq/fs rows the v2 down reads.
    pub quantize_q8_geglu_remap: Option<QuantQ8GegluRemapFn>,
    /// Slot 491: tail+combine fold - the slot-combine's ascending-k sum at
    /// the tail's dn reads; bitwise the combine_init -> moe_tail chain.
    pub moe_tail_combine: Option<MoeTailCombineFn>,
    /// slot 492: merged q|k|v NVFP4 GEMV. One grid over up to three checkpoint
    /// planes that share the same activation vector - the fix for small
    /// `out_dim` planes (granite k/v at 1024) that cannot fill the die on
    /// their own. `(segs, x, in_dim, n_segs, stream)`.
    pub nvf4_gemv_multi: Option<Nvf4GemvMultiFn>,
    /// Fused residual-add + RMSNorm with a residual MULTIPLIER folded into
    /// the add - granite's `residual_multiplier`. Bit-identical to
    /// `scale_add` + `rmsnorm_batch`; saves one launch per norm.
    pub add_rmsnorm_scaled_batch: Option<AddRmsnormScaledFn>,
    /// hibatch lane M1: hb head+router+topk -
    /// 8-token blocks, bf16 smem normed rows, router plane read once per 8
    /// tokens. Precision-class vs the chain (lane gates arbitrate). Same
    /// signature as `moe_head_router`.
    pub moe_head_router_hb: Option<MoeHeadRouterFn>,
    /// P1-2 (hibatch path 1): head twin emitting PER-128 activation scale
    /// groups (qs at n/128 stride). Same signature as `moe_head`.
    pub moe_head_xg: Option<MoeHeadFn>,
    /// P1-2: mma2 ILV gate_up consumer for per-128 activation scales
    /// (reassociated group fold). Same signature as the mma2 GEGLU.
    pub q8_0_moe_gate_up_mma2g_geglu: Option<Q8MoeGateUpMmaFn>,
    /// P1-1: down twin storing bf16 partials (same signature as down_mma2).
    pub q8_0_moe_down_mma2_pbf16: Option<Q8MoeDownMmaFn>,
    /// P1-1: tail+combine over bf16 partials (f32 sums, same fold order).
    pub moe_tail_combine_bf16: Option<MoeTailCombineFn>,
    /// B3-1: cooperative router stage - tile-matvec dots + topk in one
    /// die-filling kernel (grid.sync); per-logit math verbatim.
    pub moe_router_stage: Option<MoeRouterStageFn>,
    /// dn64: gu GEGLU quantize with per-64 Y-scale groups (fs at ff/64
    /// stride). Same signature as the mma2g GEGLU.
    pub q8_0_moe_gate_up_mma2g_y64_geglu: Option<Q8MoeGateUpMmaFn>,
    /// dn64: down consuming per-64 fs (pair-grouped fold) + pbf16 flag.
    pub q8_0_moe_down_mma2_fs64: Option<Q8MoeDownMmaFs64Fn>,
    /// v3t: TMA-staged v2 ring gate_up - bitwise to v2,
    /// sm_90+ only (NULL below). `Q8MoeGateUpMmaFn` + n_expert before
    /// max_blocks.
    pub q8_0_moe_gate_up_mma2t_geglu: Option<Q8MoeGateUpMma2tFn>,
    /// v3t down twin: `Q8MoeDownMmaFn` + n_expert before max_blocks.
    pub q8_0_moe_down_mma2t: Option<Q8MoeDownMma2tFn>,
    /// g2 (slot 504): token-major gate_up at bm=16, fq/fs at bm32 rows via
    /// the pair map - bitwise to v2, decode widths only.
    pub q8_0_moe_gate_up_g2_geglu: Option<Q8MoeGateUpG2Fn>,
    /// dual-output align (slot 505): bm32 CSR + bm16 CSR + pair map, one
    /// launch (g2 lane).
    pub moe_align_dual: Option<MoeAlignDualFn>,
    /// qwen4_exp (Qwen3.8-Flash-Next) grouped RMSNorm, Gemma (1+w) FMA affine
    /// (slot 506) - 4 hyper-connection streams normalized independently under
    /// one full-width weight; no existing rmsnorm shape fits it.
    pub q4x_group_norm_1p: Option<Q4xGroupNorm1pFn>,
    /// hyper-connection mix reduce (slot 507).
    pub q4x_hc_mix: Option<Q4xHcMixFn>,
    /// hyper-connection combine (slot 508).
    pub q4x_hc_combine: Option<Q4xHcCombineFn>,
    /// in-place silu after a scalar scale (slot 509).
    pub q4x_scale_silu: Option<Q4xScaleSiluFn>,
    /// PLE n-gram per-stream gate (slot 510).
    pub q4x_ple_gate: Option<Q4xPleGateFn>,
    /// dilated causal conv1d + silu over a sequence (slot 511).
    pub q4x_conv_dil: Option<Q4xConvDilFn>,
    /// one-token dilated conv off a carried window (slot 512).
    pub q4x_conv_dil_step: Option<Q4xConvDilStepFn>,
    /// GDN gated norm, plain w + SIGMOID gate (slot 513) - the pack's
    /// `gated_rmsnorm` is the qwen3.5 shape and gates with silu.
    pub q4x_gdn_gated_norm: Option<Q4xGdnGatedNormFn>,
    /// GDN conv split with repeat-interleave key-head widening (slot 514).
    pub q4x_gdn_split_widen: Option<Q4xGdnSplitWidenFn>,
    /// shared-expert scalar-gate fold (slot 515).
    pub q4x_add_gated_row: Option<Q4xAddGatedRowFn>,
    /// NVFP4 MoE gate+up GEMV with a fused swiglu (slot 516) - this family's
    /// experts carry both planes; the nemotron consumers are relu2-only.
    pub q4x_moe_gu_swiglu: Option<Q4xMoeGuSwigluFn>,
    /// hyper-connection combine fused with the following grouped (1+w) norm
    /// (slot 517) - one launch and one pass over the 4-stream state.
    pub q4x_combine_norm: Option<Q4xCombineNormFn>,
    /// granite fused wqkv (f8row class): one mma over the q|k|v-concat plane
    /// into K-split partials, then combine + NORM-rope + paged K/V append in
    /// one kernel. (data, w_rowscale, xq, x_rowscale, part, q_out, k_pool,
    /// v_pool, positions, slots, block_tables, blocks_per_slot, in_dim,
    /// n_heads, n_kv, head_dim, 6x yarn, batch, kv_dtype, stream)
    pub f8row_gemm_mma_qkv_norm_paged: Option<F8RowQkvRopeNormPagedFn>,
    /// pf-side rope-only twin: combine+NORM-rope+paged-append over an
    /// already-computed fused-qkv plane (nz=1 partials layout, batch uncapped).
    pub f8row_qkv_rope_norm_from_y_paged: Option<F8RowQkvRopeFromYFn>,
    /// two-segment decode GEMM (gate|up as one grid over two f8row planes of
    /// the same shape): (d0, w0, d1, w1, xq, xrs, y0, y1, in_dim,
    /// out_dim, batch, stream). Returns 100 when it declines the shape.
    pub f8row_gemm2: Option<F8RowGemm2Fn>,
    /// prefill-width swiglu + e4m3-row quant (gate, up, q, rscale, n_ff,
    /// batch, stream) -- gate is left unmodified; bit-identical to
    /// `swiglu` then `quantize_e4m3_row`.
    pub swiglu_quant_e4m3_row: Option<SwigluQuantE4m3RowFn>,
    /// norm -> e4m3-row quant fusion: (x, w, xn, q, rscale, n, eps, batch, stream); 100 = declined
    pub rmsnorm_quant_e4m3_row: Option<RmsnormQuantE4m3RowFn>,
    /// (x, proj, w, xn, q, rscale, n, eps, pscale, batch, stream); 100 = declined
    pub add_rmsnorm_scaled_quant_e4m3_row: Option<AddRmsnormScaledQuantE4m3RowFn>,
    /// Slot 523 (granite NVFP4 fused qkv): the f4c split GEMM
    /// leaving RAW K-split partials in `part` ([sk][batch][out] f32) -- no
    /// fold, no epilogue; the consumer folds. (data, scale, xq, xs, part,
    /// in_dim, out_dim, batch, sk >= 2, stream). in_dim % 256 == 0.
    pub nvf4_gemm_f4c_raw: Option<Nvf4GemmF4cRawFn>,
    /// Slot 524: fold `nz` raw partial planes (fixed ascending order, then
    /// `part_scale`) + NORM-rope + paged K/V append -- the partials twin of
    /// `f8row_qkv_rope_norm_from_y_paged` (nz=1, scale 1.0 is that kernel).
    /// (part, nz, part_scale, q_out, k_pool, v_pool, positions, slots,
    /// block_tables, bps, n_heads, n_kv, head_dim, 6x yarn, batch, kv_dtype,
    /// stream)
    pub qkv_rope_norm_from_parts_paged: Option<QkvRopeFromPartsFn>,
    /// Slot 525: swiglu over a merged [rows, 2*ff] gate|up plane
    /// straight into the nvf4 down-input staging. (fused, q, scale, ff,
    /// n_rows, stream). ff % 32 == 0.
    pub swiglu_fused_nvf4: Option<SwigluFusedNvf4Fn>,
    /// Slot 526: decode narrow-tile W4A4 GEMM (BN=32, 2 CTA/SM),
    /// batch<=32. Bit-exact to `nvf4_gemm_f4c` at the same shape. (data, scale,
    /// bias, xq, xs, part, y, scale2, in, out, batch, sk, stream).
    pub nvf4_gemm_f4cn: Option<Nvf4GemmF4cnFn>,
    /// Slot 527: f4cn writing RAW split-K partials, no reduce --
    /// the reduce-fold twin of `nvf4_gemm_f4cn`. (data, scale, bias, xq, xs,
    /// part, scale2, in, out, batch, sk >= 2, stream).
    pub nvf4_gemm_f4cn_raw: Option<Nvf4GemmF4cnRawFn>,
    /// Slot 528: add_rmsnorm_scaled consuming `nz` raw partial
    /// planes (folded with scale2) as the residual, bit-identical to
    /// reduce-then-scaled. (x, part, w, out, bias, n, eps, batch, pscale,
    /// scale2, nz, stream).
    pub add_rmsnorm_scaled_from_parts: Option<AddRmsnormScaledFromPartsFn>,
    /// Slot 529 (nvf4 decode fold-2): residual fold of `nz` raw
    /// split-K partials + rmsnorm + nvf4 quant in one launch. `acc_sel` 0 =
    /// the add_rmsnorm family (f32 accumulate, f32 divide), 1 = the
    /// rmsnorm_batch family (pd_norm_acc_mode(), double divide); `out` may be
    /// null. (x, part, w, out, bias, q, scale, n, eps, batch, pscale, scale2,
    /// nz, acc_sel, stream).
    pub add_rmsnorm_quant_nvf4_from_parts: Option<AddRmsnormQuantNvf4FromPartsFn>,
    /// Slot 530: gate|up raw partials fold + swiglu + nvf4 quant
    /// of the down input. (part, bias, q, scale, ff, n_rows, scale2, nz, stream).
    pub swiglu_quant_nvf4_from_parts: Option<SwigluQuantNvf4FromPartsFn>,
    /// Slot 531: the f4cn decode tile with a DEEP cp.async ring
    /// (`st` = 3 or 4 chunks in flight, 1 CTA/SM) for the small-out, short-K
    /// decode shapes where the 2-stage ring is latency-serialized (qkv 48
    /// tiles: 17.9 -> 12.3 us). (data, scale, bias, xq, xs, part, y, scale2,
    /// in, out, batch, sk, st, rt, stream); sk == 1 writes y unsplit; rt 64|128.
    pub nvf4_gemm_f4cd: Option<Nvf4GemmF4cdFn>,
    /// Slot 532: raw-partials twin (sk >= 1 slices, no reduce, no scale2).
    pub nvf4_gemm_f4cd_raw: Option<Nvf4GemmF4cdRawFn>,
    /// Slot 533: f4t with the swiglu + nvf4-quant epilogue over an
    /// INTERLEAVED gate|up plane (`Nvf4Plane::gu_pairs`): (data, scale, xq, xs,
    /// q, qs, scale2, in, out = 2*ff, batch, stream). No f32 landing.
    pub nvf4_gemm_f4t_swq: Option<Nvf4GemmF4tSwqFn>,
    /// Slot 534: pd_swiglu_fused over an interleaved [rows, 2ff] landing.
    pub swiglu_fused_il: Option<SwigluFusedFn>,
    /// Slot 535: pd_swiglu_fused_nvf4 over an interleaved landing.
    pub swiglu_fused_nvf4_il: Option<SwigluFusedNvf4Fn>,
    /// Slot 536: pd_swiglu_quant_nvf4_from_parts over interleaved partials.
    pub swiglu_quant_nvf4_from_parts_il: Option<SwigluQuantNvf4FromPartsFn>,
    /// 518: narrow-K bf16 GEMV - one warp per output row, 8 rows per block.
    /// Elected by SHAPE for planes far below the stock kernel's 2048-wide
    /// per-block walk; its own slot so no other lane's reduction order moves.
    pub bf16_gemv_nk_f32: Option<Bf16GemvNkF32Fn>,
    /// 519: batch-1 split-K f32 matvec with a deterministic combine and
    /// caller-owned scratch. For skinny-out decode planes.
    pub matvec_f32_sk: Option<MatvecF32SkFn>,
    /// 520: bf16 GEMV with a fused `silu(v * inv)` epilogue over the first
    /// `silu_rows` output rows.
    pub bf16_gemv_silu_f32: Option<Bf16GemvSiluF32Fn>,
    /// 522: multi-row narrow-K GEMV - the batch > 1 arm of slot 518.
    pub bf16_gemv_nk_mr_f32: Option<Bf16GemvNkMrF32Fn>,
    /// 521: per-slot dilated conv1d+silu window step (PLE), batched over rows.
    /// The GDN causal twin already existed as `conv_step_slots` above.
    pub q4x_conv_dil_step_slots: Option<Q4xConvDilStepSlotsFn>,
    /// 529: one launch over a plane folding two projections, segmented store.
    pub bf16_seg2_gemm_mma: Option<Bf16Seg2GemmMmaFn>,
    /// 530: load-time row permute of the hc up plane so the hc mean lands in
    /// one thread's registers in the fused epilogue.
    pub bf16_hcmix_permute: Option<Bf16HcmixPermuteFn>,
    /// 531: up-GEMM with the hyper-connection mix tail in its epilogue.
    pub bf16_hcmix_gemm: Option<Bf16HcmixGemmFn>,
    /// 532: PLE n-gram row gather off the device-resident 51.2 GB fp8 table.
    pub q4x_ple_gather: Option<Q4xPleGatherFn>,
    /// 533: per-slot dilated conv step over a position-indexed ring window,
    /// window advance fused in.
    pub q4x_conv_dil_step_ring: Option<Q4xConvDilStepRingFn>,
    /// 534: `n_runs` independent sequences through the gated-delta recurrence
    /// in one launch, each against its own slot's carried state.
    pub gated_delta_recurrent_runs: Option<GatedDeltaRecurrentRunsFn>,
    /// 535: co-residency gate for the f16 tensor-core lane - 0 forbids the
    /// cross-CTA K-split so the lane is safe beside a forked side stream.
    pub f16_ksplit_set: Option<F16KsplitSetFn>,
    /// 536: `attn_decode_batch` with the parallel-score walk (every thread
    /// works the dot product instead of `n_t` of them).
    pub attn_decode_batch_ps: Option<AttnDecodeBatchFn>,
    /// 537: FMHA-style decode attention. Every warp walks its own key stream
    /// with (m, l, acc) in registers, so there is no per-tile barrier and
    /// shared holds only the final cross-warp merge. head_dim 128/256 only.
    /// Its own numeric class - not bit-exact against the tile walk.
    pub attn_decode_fmha: Option<AttnDecodeBatchFn>,
    /// slot 539: strided-logits twin of moe_topk_batch - same kernels, same
    /// selection, the row stride is a parameter so a [n, n_expert+1] fused
    /// router plane can feed it without a repack.
    pub moe_topk_batch_s: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 540: strided-gate twin of q4x_add_gated_row (gate at s[r*rs]).
    pub q4x_add_gated_row_s: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 541: pre-normed runs walk - the per-token q/k L2 trees hoisted to
    /// a parallel companion kernel (bit-identical scalars), `rn` caller-owned.
    pub gated_delta_recurrent_runs_pn: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 542: EXPERIMENT cuBLASLt gemm (datapath-ceiling probe; stub =
    /// NotSupported in shipped packs).
    pub exp_lt_gemm: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 543: low-M cluster GEMM (pr4266-class decode kernel; f16 W twin
    /// x f16 X -> f32 Y, batch <= 8, cluster split-K, deterministic reduce).
    pub lowm_gemm: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 544: its load-time cluster warmup (the first cluster launch must
    /// run on a quiet context - bench/cluster_fork_probe law).
    pub lowm_warmup: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 545: SPLIT-KV fmha decode attention (grid.z KV slices + a
    /// fixed-order sink-seeded merge pass; `part` is caller-owned scratch of
    /// batch*n_heads*split*(head_dim+2) f32). Own numeric class, env-gated.
    pub attn_decode_fmha_sp: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            f32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 546: dual-plane swiglu GEMV - the n=1 shared-expert gate|up pair
    /// + swiglu in one launch (x reads amortised across both planes).
    pub bf16_gemv2_swiglu: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 548: f32 -> bf16 convert (TGV activation staging).
    pub convert_f32_bf16: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u64,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 562: HC up plane with the hyper-connection mix fused into the
    /// epilogue (w, x, xn, out, out16, in_dim, hidden, hc, stream).
    pub bf16_gemv_up_hcmix: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 563: GDN conv step with the q/k/v split+widen fused in.
    pub conv_step_slots_split: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 564: GDN recurrence with the gated norm fused into its epilogue.
    pub gated_delta_recurrent_slots_gn: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 565: batched block-per-row bf16 gemv for narrow-out decode planes.
    pub bf16_gemv_mrow_f32: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            f32,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 568: bf16 -> f32 cast.
    pub convert_bf16_f32: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u64,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 569: strided bf16 -> f32 (unpads a padded-N plane).
    pub convert_bf16_f32_rows: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 570: swiglu with a bf16 mirror of the result.
    pub swiglu_mir: Option<
        unsafe extern "C" fn(
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 573: pad a bf16 plane's row count with zero rows.
    pub bf16_pad_rows: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// slot 574: up plane in the gate epilogue's row order (d*hc+s), padded.
    pub bf16_hc_perm_pad: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            u32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// Slot 575: MoE expert-offload cache resolve - routed expert ids to
    /// VRAM slot ids with device-side LRU bookkeeping; writes the miss jobs.
    /// (idx, rows, n_slots, slot_of[n_expert], expert_in[S], last_use[S],
    /// tick, idx_slot[rows], jobs[2*rows], n_jobs, stats[2] (rows, misses
    /// accumulators), stream); rows <= n_slots.
    pub moe_cache_resolve: Option<MoeCacheResolveFn>,
    /// Slot 576: MoE expert-offload cache fill - copies the resolve's miss
    /// jobs from the host-mapped mirror into their slots over six streams
    /// (gate/up/down x data/scales). (jobs, n_jobs (device), max_jobs,
    /// src[6], dst[6], bytes[6] (HOST u64 arrays), stream).
    pub moe_cache_fill: Option<MoeCacheFillFn>,
    /// Slot 577: capability marker - present iff the k-quant repack, dequant
    /// and token-batched MoE pair serve the ggml i-quant family (IQ1_S/M,
    /// IQ2_XXS/XS/S, IQ3_XXS/S) and IQ4_NL (quant/iquant.cuh). The dtypes
    /// ride the existing entry points, so only this slot can say.
    pub kquant_iq: Option<unsafe extern "C" fn() -> i32>,
    /// 578: capability marker - the dense k-quant entry points (gemv, gather,
    /// the W4A8 gemv / nc / multi / glu, the dp4a and mma_ks GEMMs) serve
    /// the i-quant family + IQ4_NL (quant/iquant_dense.cuh). Without it a
    /// dense i-quant tensor is refused at load.
    pub kquant_iq_dense: Option<unsafe extern "C" fn() -> i32>,
    /// 579: the GDN conv split with the GGUF lane's TILED value-head order
    /// (key head `vh % hk` serves value head `vh` - what llama.cpp's
    /// converter writes); `q4x_gdn_split_widen` keeps the raw-safetensors
    /// interleave map. Same signature.
    pub q4x_gdn_split_widen_tiled: Option<Q4xGdnSplitWidenFn>,
    /// 580: expert-major prefill plan over the offload cache - marks the
    /// experts a launch routes and enumerates them into waves of `n_slots`.
    /// (idx, rows, n_expert, n_slots, n_waves, wave_of[n_expert],
    /// wave_ids[n_waves*n_slots], wave_cnt[n_waves], stream).
    pub moe_wave_plan: Option<MoeWavePlanFn>,
    /// 581: cache resolve over a DEVICE id list with a device count (one
    /// wave): (ids, n_ids, n_slots, slot_of, expert_in, last_use, tick, jobs,
    /// n_jobs, stats, stream). Same LRU as slot 575, writes no idx_slot.
    pub moe_cache_resolve_dev: Option<MoeCacheResolveDevFn>,
    /// 582: the wave's remapped routing + compaction: (idx, rows, n_active,
    /// wave_of, slot_of, wave, absent, idx_slot[rows], pairs[rows], n_pairs,
    /// rows_list, n_rows, stream) - out-of-wave pairs take `absent`; the two
    /// lists are the in-wave pair indices and the tokens holding one,
    /// device-counted, for the LIST pair kernels.
    pub moe_wave_mask: Option<MoeWaveMaskFn>,
    /// 583: `kquant_moe_gate_up` striding a device-counted pair list:
    /// the plain arguments + (pairs, n_pairs) before the stream.
    pub kquant_moe_gate_up_list: Option<KquantMoeGateUpListFn>,
    /// 584: `kquant_moe_down` striding a device-counted token list and
    /// ACCUMULATING into out: the plain arguments + (rows, n_rows).
    pub kquant_moe_down_list: Option<KquantMoeDownListFn>,
    /// 585: capability marker - `kquant_gemm_w4a8_pipe2` (and the v1 / pipe
    /// launchers, which forward to it) serve the i-quant family + Q2_K /
    /// Q3_K / IQ4_NL: the >64-row prefill tile for dense i-quant planes.
    /// Without it such a plane stays on the per-token dp4a walk.
    pub kquant_iq_tile: Option<unsafe extern "C" fn() -> i32>,
    /// 586: expert-GROUPED `kquant_moe_gate_up` over a
    /// `moe_align_bm(bm = group)` layout: (gate_data, gate_scales, up_data,
    /// up_scales, sorted_row, sorted_slot, block_expert, xq, xs, xsums, out,
    /// in_dim, ff, n_active, max_blocks, group, gdt, udt). One block per
    /// (expert group, out row), so the i-quant window unpack is paid once per
    /// group instead of once per routed row - the prefill class. Same output
    /// layout and the same numerics as slot 494.
    pub kquant_moe_gate_up_grp: Option<KquantMoeGateUpGrpFn>,
    /// 587: COLUMN-TILED routed down - slot 495's arguments plus `cols`
    /// (4, 8, 16) before the dtype. One block owns `cols` output columns and
    /// reuses each activation window across them; at prefill widths the grid
    /// is still full with `cols` times fewer blocks, and the dependent weight
    /// loads finally have something to overlap. Numerics are slot 495's.
    pub kquant_moe_down_cols: Option<KquantMoeDownColsFn>,
    /// 588: K-SPLIT Q8_0 GEMV/GEMM for NARROW-OUT planes - (data, scale,
    /// bias, x, y, partials, counters, in_dim, out_dim, batch, split), grid
    /// (out_dim, split, batch). A row-per-block GEMV stops filling the die
    /// long before the plane stops being big (the hyper-connection down plane
    /// is [10240, 320]: 320 blocks on 148 SMs, and the batched mt tile makes
    /// twenty). Deterministic ascending fold; counters self-reset so a
    /// captured graph replays identically.
    pub q8_0_gemv_sk: Option<Q80GemvSkFn>,
    /// Slot 589: `pd_nvf4_gemm_f4t4` - the f4t TMA ring at four 128-K stages in
    /// the same smem (GB10 2026-09-07). Same contract as slot 430; NULL wherever
    /// 430 is NULL.
    pub nvf4_gemm_f4t4: Option<
        unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *mut core::ffi::c_void,
            f32,
            u32,
            u32,
            u32,
            *mut core::ffi::c_void,
        ) -> i32,
    >,
    /// 590: expert-GROUPED routed down over one COLUMN CHUNK - (down_data,
    /// down_scales, sorted_row, sorted_slot, block_expert, topk_w, fq, fs,
    /// fsums, part, ff, embd, o0, ocols, n_active, max_blocks, group). The
    /// group's activation rows are staged once and each column's weight row
    /// unpacked once, so the unpack falls by the rows per group; a grouped
    /// block holds different tokens, so it writes one partial per (pair,
    /// column) and slot 591 does the fold.
    pub kquant_moe_down_grp: Option<KquantMoeDownGrpFn>,
    /// 591: fold a chunk of slot 590's partials into `out` in ASCENDING slot
    /// order - (part, out, embd, o0, ocols, n_active, rows).
    pub moe_part_fold_at: Option<MoePartFoldAtFn>,
    /// 592: REGISTER-TILED grouped gate+up - a BM x BN tile off a
    /// `moe_align_bm(16)` CSR with both operands staged per BK slice, so a
    /// thread owns whole dots and the activations are read once per column
    /// TILE instead of once per column. Pair-major output like slot 586's;
    /// `in_dim` must be a multiple of the kernel's BK (128). Its f32
    /// association is the tile's, not the pair kernel's.
    pub kquant_moe_gate_up_tile: Option<KquantMoeGateUpTileFn>,
    /// 593: the register-tiled DOWN twin of 592, over one column chunk -
    /// (down_data, down_scales, sorted_row, sorted_slot, block_expert, topk_w,
    /// fq, fs, fsums, part, ff, embd, o0, ocols, n_active, max_blocks). Writes
    /// the partials slot 591 folds; `ff` must be a multiple of the tile's BK.
    pub kquant_moe_down_tile: Option<KquantMoeDownTileFn>,
    /// Slots 594/595: `pd_add_rmsnorm_quant_{e4m3,nvf4}_pf` - the prefill
    /// prenorm (the mmq kernel's exact norm phase) with the e4m3 / nvf4 quant
    /// epilogue in place of the mmq tiles (GB10 2026-09-08). xn nullable.
    pub add_rmsnorm_quant_e4m3_pf: Option<AddRmsnormQuantPfFn>,
    pub add_rmsnorm_quant_nvf4_pf: Option<AddRmsnormQuantPfFn>,
    /// 596: the single-sequence GDN walk, P-SPLIT and PRE-NORMED - slot 533's
    /// arguments plus caller-owned `rn` (n_tokens * n_heads * 2 floats, the
    /// buffer the runs_pn entry already sizes). P threads share a state
    /// column, so the walk stops being register-bound; the norms come from the
    /// companion pass, so its token loop carries no barrier.
    pub gated_delta_recurrent_pn: Option<GatedDeltaRecurrentPnFn>,
}

/// Pre-normed single-sequence GDN walk (see `KernelTableV1::gated_delta_recurrent_pn`).
pub type GatedDeltaRecurrentPnFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// Register-tiled routed down (see `KernelTableV1::kquant_moe_down_tile`).
pub type KquantMoeDownTileFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Register-tiled grouped gate+up (see `KernelTableV1::kquant_moe_gate_up_tile`).
pub type KquantMoeGateUpTileFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Expert-grouped routed down (see `KernelTableV1::kquant_moe_down_grp`).
pub type KquantMoeDownGrpFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot fold for the grouped down (see `KernelTableV1::moe_part_fold_at`).
pub type MoePartFoldAtFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// K-split Q8_0 GEMV (see `KernelTableV1::q8_0_gemv_sk`).
pub type Q80GemvSkFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Column-tiled routed down (see `KernelTableV1::kquant_moe_down_cols`).
pub type KquantMoeDownColsFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Expert-grouped gate+up (see `KernelTableV1::kquant_moe_gate_up_grp`).
pub type KquantMoeGateUpGrpFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Expert-major prefill plan (see `KernelTableV1::moe_wave_plan`).
pub type MoeWavePlanFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// Device-count cache resolve (see `KernelTableV1::moe_cache_resolve_dev`).
pub type MoeCacheResolveDevFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// Wave routing mask + compaction (see `KernelTableV1::moe_wave_mask`).
pub type MoeWaveMaskFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    u32,
    u32,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// LIST gate_up (see `KernelTableV1::kquant_moe_gate_up_list`).
pub type KquantMoeGateUpListFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// LIST down (see `KernelTableV1::kquant_moe_down_list`).
pub type KquantMoeDownListFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// MoE expert-offload cache resolve (see `KernelTableV1::moe_cache_resolve`).
pub type MoeCacheResolveFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    u32,
    u32,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// MoE expert-offload cache fill (see `KernelTableV1::moe_cache_fill`).
pub type MoeCacheFillFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 533: see the KernelTableV1 field doc.
pub type Nvf4GemmF4tSwqFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    f32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 531: see the KernelTableV1 field doc.
pub type Nvf4GemmF4cdFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    f32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 532: see the KernelTableV1 field doc.
pub type Nvf4GemmF4cdRawFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    f32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 529: see the KernelTableV1 field doc.
pub type AddRmsnormQuantNvf4FromPartsFn = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    f32,
    u32,
    f32,
    f32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 530: see the KernelTableV1 field doc.
pub type SwigluQuantNvf4FromPartsFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    f32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 527: see the KernelTableV1 field doc.
pub type Nvf4GemmF4cnRawFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    f32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 528: see the KernelTableV1 field doc.
pub type AddRmsnormScaledFromPartsFn = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    f32,
    u32,
    f32,
    f32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 526: see the KernelTableV1 field doc.
pub type Nvf4GemmF4cnFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    f32,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 525: see the KernelTableV1 field doc.
pub type SwigluFusedNvf4Fn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 523: see the KernelTableV1 field doc.
pub type Nvf4GemmF4cRawFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Slot 524: see the KernelTableV1 field doc.
pub type QkvRopeFromPartsFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    u32,
    f32,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// pf-side rope-only twin: see the KernelTableV1 field doc.
pub type F8RowQkvRopeFromYFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// two-segment decode GEMM: see the KernelTableV1 field doc.
pub type F8RowGemm2Fn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// prefill swiglu + e4m3-row quant: see the KernelTableV1 field doc.
pub type SwigluQuantE4m3RowFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;
pub type RmsnormQuantE4m3RowFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    f32,
    u32,
    *mut core::ffi::c_void,
) -> i32;
pub type AddRmsnormScaledQuantE4m3RowFn = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    f32,
    f32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// granite fused wqkv (f8row): see the KernelTableV1 field doc.
pub type F8RowQkvRopeNormPagedFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    f32,
    f32,
    f32,
    f32,
    f32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Merged NVFP4 GEMV over up to 3 planes sharing one `x` (slot 492).
/// `segs` is `[PdNv4GemvSeg; n_segs]`: `(data, scale, bias, y, scale2, out_dim)`.
pub type Nvf4GemvMultiFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;
pub type AddRmsnormScaledFn = unsafe extern "C" fn(
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    f32,
    u32,
    *mut core::ffi::c_void,
    f32,
) -> i32;

/// granite's fused residual-add + rmsnorm + Q8 quantize (slot 482).
/// `(x, proj, w, xn, q, scale, sums, n, batch, eps, res_scale, stream)`
#[allow(clippy::too_many_arguments)]
pub type AddRmsnormQ8XnFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    xn: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    sums: *mut core::ffi::c_void,
    n: u32,
    batch: u32,
    eps: f32,
    res_scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// How many function-pointer slots `KernelTableV1` carries after the 8-byte
/// `size`/`_reserved` header. **Bump this in the same edit as the append.**
///
/// It exists because the count went stale twice. The first report was the
/// table growing 3056 -> 3064 with the guard untouched; the guard
/// was then rewritten from a bare byte count into a derived `8 + N * 8`, which
/// was the right shape - and the count went stale again when slot 481
/// (`f8lin_gemv`) landed, because the shape was never what failed. A number a
/// human must remember to change is a number that is eventually not changed,
/// and the only witness was a test somebody had to run.
///
/// So the witness is the COMPILER now. `size_of` is const, so the assertion
/// below is evaluated at compile time: append a field without touching this
/// const and the crate does not build. No stale count can reach a binary and
/// wait to be found by a gate run.
///
/// The pack side needs no twin - `exports.cuh` reports
/// `(uint32_t)sizeof(KernelTableV1)` and is self-describing by construction.
/// The two are reconciled at load by `loader.rs`'s `table_fit`, which clamps
/// the copy to the smaller of declared and expected, so an old pack against a
/// new engine (or the reverse) reads missing entries as None rather than a
/// shifted slot.
pub const KERNEL_TABLE_SLOTS: usize = 582;

const _: () = assert!(
    core::mem::size_of::<KernelTableV1>() == 8 + KERNEL_TABLE_SLOTS * 8,
    "KernelTableV1 changed size without KERNEL_TABLE_SLOTS being updated - \
     bump the const in the same edit as the slot append",
);

/// DFlash conditioning fold (see `KernelTableV1::dflash_cond_append`).
#[allow(clippy::too_many_arguments)]
pub type DflashCondAppendFn = unsafe extern "C" fn(
    fk: *const core::ffi::c_void,
    fv: *const core::ffi::c_void,
    kw: *const core::ffi::c_void,
    pool_k: *mut core::ffi::c_void,
    pool_v: *mut core::ffi::c_void,
    rows_w: *const core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_kv: u32,
    head_dim: u32,
    eps: f32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    nw: u32,
    norm_batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Spec-verify hold recurrence (see `KernelTableV1::gated_delta_verify_hold`).
#[allow(clippy::too_many_arguments)]
pub type GatedDeltaVerifyHoldFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    states: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    batch: u32,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Commit-time accepted-prefix recompute (see
/// `KernelTableV1::gated_delta_commit_walk`).
#[allow(clippy::too_many_arguments)]
pub type GatedDeltaCommitWalkFn = unsafe extern "C" fn(
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    committed: *const core::ffi::c_void,
    states: *mut core::ffi::c_void,
    batch: u32,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Per-row-snapshot scan (see `KernelTableV1::mamba2_scan_seq_snap`).
pub type Mamba2ScanSeqSnapFn = unsafe extern "C" fn(
    state: *mut core::ffi::c_void,
    xbc: *const core::ffi::c_void,
    dt_raw: *const core::ffi::c_void,
    dt_stride: u32,
    a: *const core::ffi::c_void,
    d: *const core::ffi::c_void,
    dt_bias: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    snap: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    d_state: u32,
    n_groups: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// KV tier extent gather/scatter (see `KernelTableV1::kv_gather_blocks` /
/// `kv_scatter_blocks`). `planes` is a device u64 array of 4-tuples
/// {plane_base, src_stride, plane_bytes, dst_off}; `block_ids` a device u32
/// array; record layout `extent[b*record_stride + dst_off[p] ..+ bytes[p]]`.
/// Every base/stride/bytes/offset/record_stride must be a multiple of 16
/// (uint4-vectorized copies) - the engine wrapper validates before launch.
/// `max_plane_bytes` shapes the grid host-side (the descriptor lives on
/// device and cannot).
pub type KvXferBlocksFn = unsafe extern "C" fn(
    planes: *const core::ffi::c_void,
    block_ids: *const core::ffi::c_void,
    extent: *mut core::ffi::c_void,
    record_stride: u64,
    max_plane_bytes: u64,
    n_planes: u32,
    n_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Strided-rows copy (see `KernelTableV1::copy_rows_strided`).
pub type CopyRowsStridedFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    src_off: u32,
    src_stride: u32,
    dst: *mut core::ffi::c_void,
    len: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Token-batched Q8_0 expert up + relu^2 (see
/// `KernelTableV1::q8_0_moe_up_relu2`).
pub type Q8MoeUpRelu2Fn = unsafe extern "C" fn(
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Decode-band single-plane relu^2 expert up (see
/// `KernelTableV1::q8_0_moe_up_relu2_dec2`). `rows_pb` 0 takes the pack's
/// elected rows-per-CTA; nonzero values are the lab sweep's instrument.
pub type Q8MoeUpRelu2Dec2Fn = unsafe extern "C" fn(
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    batch: u32,
    rows_pb: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Sorted Q8_0 expert up + relu^2 (see
/// `KernelTableV1::q8_0_moe_up_relu2_sorted`).
pub type Q8MoeUpRelu2SortedFn = unsafe extern "C" fn(
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fused: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Batched slot-arena conv step (see `KernelTableV1::mamba_conv_step_batch`).
pub type MambaConvStepBatchFn = unsafe extern "C" fn(
    win: *mut core::ffi::c_void,
    x: *const core::ffi::c_void,
    x_off: u32,
    x_stride: u32,
    slots: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    conv_dim: u32,
    k: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Batched slot-arena single-token SSD scan step (see
/// `KernelTableV1::mamba2_scan_step_batch`).
pub type Mamba2ScanStepBatchFn = unsafe extern "C" fn(
    state: *mut core::ffi::c_void,
    xbc: *const core::ffi::c_void,
    dt_raw: *const core::ffi::c_void,
    dt_stride: u32,
    slots: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    d: *const core::ffi::c_void,
    dt_bias: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    batch: u32,
    n_heads: u32,
    head_dim: u32,
    d_state: u32,
    n_groups: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Row-batched NVFP4 GEMV (see `KernelTableV1::nvf4_gemv_batch`).
pub type Nvf4GemvBatchFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    scale2: f32,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Decode multi-task NVFP4 expert up + relu^2 (see
/// `KernelTableV1::nvf4_moe_up_relu2_mt`).
pub type Nvf4MoeUpRelu2MtFn = unsafe extern "C" fn(
    rdata: *const core::ffi::c_void,
    rscale: *const core::ffi::c_void,
    rscale2: *const core::ffi::c_void,
    sdata: *const core::ffi::c_void,
    sscale: *const core::ffi::c_void,
    sscale2: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    act: *mut core::ffi::c_void,
    in_dim: u32,
    ff_r: u32,
    ff_s: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Decode slot-split NVFP4 expert down -> pre-weighted slot partials (see
/// `KernelTableV1::nvf4_moe_down_part`).
pub type Nvf4MoeDownPartFn = unsafe extern "C" fn(
    rdata: *const core::ffi::c_void,
    rscale: *const core::ffi::c_void,
    rscale2: *const core::ffi::c_void,
    sdata: *const core::ffi::c_void,
    sscale: *const core::ffi::c_void,
    sscale2: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    act: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff_r: u32,
    ff_s: u32,
    embd: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted-tile NVFP4 expert up + relu^2 (see
/// `KernelTableV1::nvf4_moe_up_relu2_bs`).
pub type Nvf4MoeUpRelu2BsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    scale2: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    nb: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted-tile NVFP4 expert down -> per-(token, slot) partials (see
/// `KernelTableV1::nvf4_moe_down_bs`).
pub type Nvf4MoeDownBsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    scale2: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    kw: u32,
    np: u32,
    slot_off: u32,
    nb: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Capture-time f16 mmaf election gate (see `KernelTableV1::f16_mmaf_set`).
pub type F16MmafSetFn = unsafe extern "C" fn(on: i32) -> KernelStatus;

/// modelopt NVFP4 checkpoint dequant (see `KernelTableV1::nvf4_dequant`).
pub type Nvf4DequantFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    scale2: f32,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// W4A16-class NVFP4 GEMV (see `KernelTableV1::nvf4_gemv`).
pub type Nvf4GemvFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    scale2: f32,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Per-row-f32-scale e4m3 GEMV (see `KernelTableV1::f8r_gemv`).
pub type F8rGemvFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    rscale: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Bulk mamba-2 conv over a token span (see `KernelTableV1::mamba_conv_seq`).
pub type MambaConvSeqFn = unsafe extern "C" fn(
    win: *mut core::ffi::c_void,
    xbc: *const core::ffi::c_void,
    x_off: u32,
    x_stride: u32,
    w: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    conv_dim: u32,
    k: u32,
    n_tokens: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Mamba-2 conv step with bias (see `KernelTableV1::mamba_conv_step`).
pub type MambaConvStepFn = unsafe extern "C" fn(
    win: *mut core::ffi::c_void,
    x_new: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    conv_dim: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sequential mamba-2 SSD scan (see `KernelTableV1::mamba2_scan_seq`).
pub type Mamba2ScanSeqFn = unsafe extern "C" fn(
    state: *mut core::ffi::c_void,
    xbc: *const core::ffi::c_void,
    dt_raw: *const core::ffi::c_void,
    dt_stride: u32,
    a: *const core::ffi::c_void,
    d: *const core::ffi::c_void,
    dt_bias: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    d_state: u32,
    n_groups: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Grouped gated RMSNorm (see `KernelTableV1::mamba_rmsnorm_gated_g`).
pub type MambaRmsnormGatedGFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    z: *const core::ffi::c_void,
    z_stride: u32,
    weight: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_tokens: u32,
    d: u32,
    n_groups: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// NVFP4 MoE expert up GEMV + squared-relu (see
/// `KernelTableV1::nvf4_moe_up_relu2`).
pub type Nvf4MoeUpRelu2Fn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    scale2: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    k: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// NVFP4 MoE expert down GEMV + weighted combine (see
/// `KernelTableV1::nvf4_moe_down_acc`).
pub type Nvf4MoeDownAccFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    scale2: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    xr: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    k: u32,
    batch: u32,
    accumulate: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// SAM ViTDet attention with the decomposed rel-pos bias (see
/// `KernelTableV1::sam_attn`).
pub type SamAttnFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    rh: *const core::ffi::c_void,
    rw: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_batch: u32,
    side: u32,
    n_heads: u32,
    hd: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Alignment-head cross-attention read-out (see
/// `KernelTableV1::whisper_xattn_probs`).
pub type WhisperXattnProbsFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    qbias: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    heads: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    kv_stride: u32,
    n_enc: u32,
    n_heads: u32,
    hd: u32,
    n_sel: u32,
    batch: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// bf16 dense GEMV (see `KernelTableV1::bf16_gemv_f32`).
pub type Bf16GemvF32Fn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// bf16 dense GEMM (see `KernelTableV1::bf16_gemm_f32`).
pub type Bf16GemmF32Fn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// In-house f16xf16->f32 tensor-core dense GEMM (see `KernelTableV1::f16_gemm`).
pub type F16GemmFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    beta: f32,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// bf16 embedding gather (see `KernelTableV1::embed_gather_bf16`).
pub type EmbedGatherBf16Fn = unsafe extern "C" fn(
    table: *const core::ffi::c_void,
    tokens: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    embd: u32,
    n_tokens: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// flat-scale e4m3 sorted MoE down (see `KernelTableV1::f8row_moe_down_mma`).
#[allow(clippy::type_complexity)]
pub type F8RowMoeDownFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_rs: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// f32 -> per-32-scaled e4m3 (see `KernelTableV1::quantize_e4m3_b32f`).
pub type QuantizeE4m3B32fFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// flat-scale e4m3 sorted MoE gate+up
/// (see `KernelTableV1::f8row_moe_gate_up_mma_geglu`).
#[allow(clippy::type_complexity)]
pub type F8RowMoeGateUpFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_rs: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_rs: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Two-buffer fused-plane prefill GEMM (see
/// `KernelTableV1::f8_gemm_lin_kt_split`).
pub type F8GemmLinKtSplitFn = unsafe extern "C" fn(
    wlin: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    y2: *mut core::ffi::c_void,
    ncut: u32,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Standalone scalar multiply (see `KernelTableV1::scale_f32`).
pub type ScaleF32Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    s: f32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Qwen3.5 fused-plane prefill consumer (see
/// `KernelTableV1::q36_qkg_nra_rows`).
#[allow(clippy::type_complexity)]
pub type Q36QkgNraRowsFn = unsafe extern "C" fn(
    qkg: *const core::ffi::c_void,
    q_off: u32,
    row_stride: u32,
    k_off: u32,
    v_off: u32,
    qw: *const core::ffi::c_void,
    kw: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    gate_out: *mut core::ffi::c_void,
    k_pool: *mut core::ffi::c_void,
    v_pool: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    mpos: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    n_rot: u32,
    eps: f32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    rows: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Laguna decode-tick epilogue fold (see `KernelTableV1::lag_qk_nra_rows`).
#[allow(clippy::type_complexity)]
pub type LagQkNraRowsFn = unsafe extern "C" fn(
    q_src: *const core::ffi::c_void,
    q_off: u32,
    q_stride: u32,
    k_src: *const core::ffi::c_void,
    k_off: u32,
    k_stride: u32,
    v_src: *const core::ffi::c_void,
    v_stride: u32,
    qw: *const core::ffi::c_void,
    kw: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    k_pool: *mut core::ffi::c_void,
    v_pool: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    mpos: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    n_rot: u32,
    eps: f32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    rows: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Laguna per-head softplus gate (see `KernelTableV1::mul_softplus_head`).
pub type MulSoftplusHeadFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    gate: *const core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Laguna sigmoid MoE router (see `KernelTableV1::moe_topk_sigmoid_batch`).
pub type MoeTopkSigmoidBatchFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    routed_scale: f32,
    n_expert: u32,
    k: u32,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Shared fold-in topk (see `KernelTableV1::moe_topk_sigmoid_batch_sh`).
pub type MoeTopkSigmoidBatchShFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    routed_scale: f32,
    n_expert: u32,
    k: u32,
    ns: u32,
    sh0: u32,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Drafter xh stitch (see `KernelTableV1::spec_xh_stitch`).
pub type SpecXhStitchFn = unsafe extern "C" fn(
    emb: *const core::ffi::c_void,
    h: *const core::ffi::c_void,
    xh: *mut core::ffi::c_void,
    r: u32,
    n_main: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Host-indexed row gather (see `KernelTableV1::hrow_gather`).
pub type HrowGatherFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n: u32,
    n_main: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sampled draft draw + q-store write (see `KernelTableV1::draft_rs`).
pub type DraftRsFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    invt: *const core::ffi::c_void,
    uplane: *const core::ffi::c_void,
    step: *const core::ffi::c_void,
    qstore: *mut core::ffi::c_void,
    qsum: *mut core::ffi::c_void,
    tok: *mut core::ffi::c_void,
    rows: u32,
    n: u32,
    rmax: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Canonical rejection-sampling verify resolve (see
/// `KernelTableV1::spec_rs_resolve`).
pub type SpecRsResolveFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    drafts: *const core::ffi::c_void,
    qstore: *const core::ffi::c_void,
    qsum: *const core::ffi::c_void,
    par: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    nrs: u32,
    rr: u32,
    n: u32,
    rmax: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused DN input conv + split + norm (see `KernelTableV1::causal_conv1d_silu_qkv`).
pub type ConvSiluQkvFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    k: *mut core::ffi::c_void,
    v: *mut core::ffi::c_void,
    n_rows: u32,
    n_k_heads: u32,
    n_v_heads: u32,
    s: u32,
    conv_k: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Fused-layout bf16 swiglu quant (see `KernelTableV1::quantize_e4m3_swiglu_b16_gu`).
pub type QuantSwigluB16GuFn = unsafe extern "C" fn(
    gu: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    ff: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Byte-passthrough lin repack (see `KernelTableV1::f8w_repack_lin_bs`).
pub type F8wRepackLinBsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Block-scale tile-linear decode GEMM (see `KernelTableV1::f8_gemm_lin_bs`).
pub type F8GemmLinBsFn = unsafe extern "C" fn(
    wlin: *const core::ffi::c_void,
    wsc: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Fused gated-rmsnorm + e4m3 quant (see `KernelTableV1::gated_rmsnorm_e4m3`).
pub type GatedRmsnormE4m3Fn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    z: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n_rows: u32,
    d: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Load-time f8w tile-linear repack (see `KernelTableV1::f8w_repack_lin`).
pub type F8LinGemvFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

pub type F8wRepackLinFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Tile-linear decode GEMM (see `KernelTableV1::f8_gemm_lin`).
pub type F8GemmLinFn = unsafe extern "C" fn(
    wlin: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Tile-linear prefill GEMM (see `KernelTableV1::f8_gemm_lin_kt`).
pub type F8GemmLinKtFn = unsafe extern "C" fn(
    wlin: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    o16: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Decode norm+e4m3 fuse (see `KernelTableV1::add_rmsnorm_e4m3_xn`).
pub type AddRmsnormE4m3XnFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    xn: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    batch: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused swiglu+e4m3 over a merged landing (see `KernelTableV1::swiglu_fused_e4m3`).
pub type SwigluFusedE4m3Fn = unsafe extern "C" fn(
    fused: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    ff: u32,
    n_rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// bf16 -> f8r conversion (see `KernelTableV1::bf16_to_f8r`).
pub type Bf16ToF8rFn = unsafe extern "C" fn(
    bf16: *const core::ffi::c_void,
    f8_data: *mut core::ffi::c_void,
    f8_scale: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// bf16 -> f8w conversion (see `KernelTableV1::bf16_to_f8w`).
pub type Bf16ToF8wFn = unsafe extern "C" fn(
    bf16: *const core::ffi::c_void,
    f8_data: *mut core::ffi::c_void,
    f8_scale: *mut core::ffi::c_void,
    n_blocks: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// bf16-input swiglu+e4m3 quant (see `KernelTableV1::quantize_e4m3_swiglu_b16`).
pub type QuantE4m3SwigluB16Fn = unsafe extern "C" fn(
    gate: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// e4m3 decode-band ks GEMM (see `KernelTableV1::f8d_gemm_mma_ks`).
pub type F8dGemmMmaKsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused-landing row slice (see `KernelTableV1::row_slice`).
pub type RowSliceFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    src_stride: u32,
    col_off: u32,
    width: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused-plane SwiGLU (see `KernelTableV1::swiglu_fused`).
pub type SwigluFusedFn = unsafe extern "C" fn(
    fused: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    ff: u32,
    n_rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dec3 gate_up (see `KernelTableV1::q8_0_moe_gu_dec3_geglu`).
pub type Q8MoeGuDec3Fn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    bexp: *const core::ffi::c_void,
    srow: *const core::ffi::c_void,
    sslot: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    max_blocks: u32,
    pairs: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// dec3 down (see `KernelTableV1::q8_0_moe_dn_dec3`).
pub type Q8MoeDnDec3Fn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    bexp: *const core::ffi::c_void,
    srow: *const core::ffi::c_void,
    sslot: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    pairs: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// dec3 combine (see `KernelTableV1::moe_combine_dec3`).
pub type MoeCombineDec3Fn = unsafe extern "C" fn(
    part: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Q8_0 -> per-row e4m3 requant (see `KernelTableV1::q8_0_to_f8row`).
pub type Q8ToF8RowFn = unsafe extern "C" fn(
    q8_data: *const core::ffi::c_void,
    q8_scale: *const core::ffi::c_void,
    f8_data: *mut core::ffi::c_void,
    row_scale: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// per-row e4m3 activation quant (see `KernelTableV1::quantize_e4m3_row`).
pub type QuantizeE4m3RowFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    row_scale: *mut core::ffi::c_void,
    n_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// fold-free per-row e4m3 GEMM (see `KernelTableV1::f8row_gemm`).
pub type F8RowGemmFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    w_rowscale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    x_rowscale: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// e4m3 GEMV (see `KernelTableV1::f8_gemv`).
pub type F8GemvFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// e4m3 batched GEMV (see `KernelTableV1::f8_gemv_batch`).
pub type F8GemvBatchFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// f8 K-split block-scale MMA GEMM (see `KernelTableV1::f8_gemm_mma_ks`).
pub type F8GemmMmaKsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// kv_dtype-aware gemma qkv epilogue (see `KernelTableV1::gemma_qkv_nra2`).
pub type GemmaQkvNra2Fn = unsafe extern "C" fn(
    q: *mut core::ffi::c_void,
    k: *mut core::ffi::c_void,
    v: *mut core::ffi::c_void,
    wq_norm: *const core::ffi::c_void,
    wk_norm: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    kc: *mut core::ffi::c_void,
    vc: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    factors: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    bps: u32,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    max_ctx: u32,
    batch: u32,
    eps: f32,
    theta_scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Wide-batch spec-verify attention (see `KernelTableV1::attn_spec_batch_paged`).
pub type AttnSpecBatchPagedFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    pool_k: *const core::ffi::c_void,
    pool_v: *const core::ffi::c_void,
    out_o: *mut core::ffi::c_void,
    out_ml: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    kv_dim: u32,
    swa_window: u32,
    n_splits: u32,
    rows: u32,
    k1: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// GEGLU-fused mmq quantize (see `KernelTableV1::quantize_q8_mmq_geglu`).
pub type QuantizeQ8MmqGegluFn = unsafe extern "C" fn(
    gate: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    yq: *mut core::ffi::c_void,
    in_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Fused post-norm + residual + scale (see `KernelTableV1::rmsnorm_add_scale`).
pub type RmsnormAddScaleFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    n: u32,
    eps: f32,
    s: f32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Fused residual+scale (see `KernelTableV1::add_scale`).
pub type AddScaleFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    y: *const core::ffi::c_void,
    s: f32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Paired-layout GEGLU (see `KernelTableV1::geglu_pair`).
pub type GegluPairFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    ff: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Gemma4 fused QKV epilogue (see `KernelTableV1::gemma_qkv_nra`).
#[allow(clippy::too_many_arguments)]
pub type GemmaQkvNraFn = unsafe extern "C" fn(
    qp: *mut core::ffi::c_void,
    kp: *mut core::ffi::c_void,
    vp: *mut core::ffi::c_void,
    wq_norm: *const core::ffi::c_void,
    wk_norm: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    kc: *mut core::ffi::c_void,
    vc: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    factors: *const core::ffi::c_void,
    block_tables: *const core::ffi::c_void,
    bps: u32,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    max_ctx: u32,
    batch: u32,
    eps: f32,
    theta_scale: f32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Q8_0 embedding gather (see `KernelTableV1::embed_gather_q8`).
pub type EmbedGatherQ8Fn = unsafe extern "C" fn(
    table: *const core::ffi::c_void,
    tokens: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    embd: u32,
    n_tokens: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Elementwise softcap (see `KernelTableV1::softcap`).
pub type SoftcapFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    n: u32,
    cap: f32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Vision 2D NEOX rope (see `KernelTableV1::rope2d_neox`).
pub type Rope2dNeoxFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    pos_x: *const core::ffi::c_void,
    pos_y: *const core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// GEGLU elementwise (see `KernelTableV1::geglu`).
pub type GegluFn = unsafe extern "C" fn(
    gate: *mut core::ffi::c_void,
    up: *const core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Factored batched rope (see `KernelTableV1::rope_factors_batch`).
#[allow(clippy::too_many_arguments)]
pub type RopeFactorsBatchFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    factors: *const core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Tail split-K mmq (see `KernelTableV1::q8_0_gemm_mmq_pipe_sk`).
#[allow(clippy::too_many_arguments)]
pub type Q8GemmMmqPipeSkFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    yq: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    partials: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    sm_count: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused-QKV norm/rope/scatter (see `KernelTableV1::qkv_norm_rope_append`).
#[allow(clippy::too_many_arguments)]
pub type QkvNormRopeAppendFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    wq_norm: *const core::ffi::c_void,
    wk_norm: *const core::ffi::c_void,
    qn_out: *mut core::ffi::c_void,
    kcache: *mut core::ffi::c_void,
    vcache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    eps: f32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused q-side RMSNorm + rope (see `KernelTableV1::q_norm_rope`).
#[allow(clippy::too_many_arguments)]
pub type QNormRopeFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused k-side RMSNorm + rope + cache append (see
/// `KernelTableV1::k_norm_rope_append`).
#[allow(clippy::too_many_arguments)]
pub type KNormRopeAppendFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    cache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    eps: f32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused SwiGLU + smooth + nvf4 quantize (see
/// `KernelTableV1::quantize_nvf4_swiglu_smooth`).
pub type QuantizeNvf4SwigluSmoothFn = unsafe extern "C" fn(
    gate: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    sinv: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    in_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Per-column f32 abs-max (see `KernelTableV1::col_absmax`).
pub type ColAbsmaxFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Per-column Q8_0 weight abs-max (see `KernelTableV1::q8_0_col_absmax`).
pub type Q8ColAbsmaxFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// SmoothQuant-folded nvf4 activation quantize (see
/// `KernelTableV1::quantize_nvf4_smooth`).
pub type QuantizeNvf4SmoothFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    sinv: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    in_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// SmoothQuant-folded nvf4 weight requant (see
/// `KernelTableV1::q8_0_to_nvf4_smooth`).
pub type Q8ToNvf4SmoothFn = unsafe extern "C" fn(
    q8_data: *const core::ffi::c_void,
    q8_scale: *const core::ffi::c_void,
    svec: *const core::ffi::c_void,
    mx_data: *mut core::ffi::c_void,
    mx_scale: *mut core::ffi::c_void,
    n_blocks: u64,
    in_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused gate+up+swiglu->nvf4 (see `KernelTableV1::mxfp4_gemm_bs_gu`).
pub type Mxfp4GemmBsGuFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused SwiGLU + e4m3 quantize (see `KernelTableV1::quantize_e4m3_swiglu`).
pub type QuantizeE4m3SwigluFn = unsafe extern "C" fn(
    gate: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Q8_0 -> mxfp4 device weight re-quant (see `KernelTableV1::q8_0_to_mxfp4`).
pub type Q8ToMxfp4Fn = unsafe extern "C" fn(
    q8_data: *const core::ffi::c_void,
    q8_scale: *const core::ffi::c_void,
    mx_data: *mut core::ffi::c_void,
    mx_scale: *mut core::ffi::c_void,
    n_blocks: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Dense block-scale GEMM (see `KernelTableV1::mxfp4_gemm_bs`).
pub type Mxfp4GemmBsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Multi-slot batched tiled prefill attention (see `KernelTableV1::attn_prefill_batch`).
#[allow(clippy::too_many_arguments)]
pub type AttnPrefillBatchFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    kc: *const core::ffi::c_void,
    vc: *const core::ffi::c_void,
    sinks: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    tile_row0: *const core::ffi::c_void,
    tile_slot: *const core::ffi::c_void,
    n_qtiles: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    kv_dim: u32,
    swa_window: u32,
    n_rows: u32,
    scale: f32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused add + RMSNorm (see `KernelTableV1::add_rmsnorm_batch`).
pub type AddRmsnormBatchFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// int8-MMA sorted MoE gate+up (see `KernelTableV1::q8_0_moe_gate_up_mma`).
#[allow(clippy::too_many_arguments)]
pub type Q8MoeGateUpMmaFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// int8-MMA sorted MoE down (see `KernelTableV1::q8_0_moe_down_mma`).
#[allow(clippy::too_many_arguments)]
pub type Q8MoeDownMmaFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// P1 dn64 down twin: `Q8MoeDownMmaFn` + trailing `pbf16` flag.
#[allow(clippy::too_many_arguments)]
pub type Q8MoeDownMmaFs64Fn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    bm: u32,
    pbf16: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// v3t TMA gate_up (slot 502): `Q8MoeGateUpMmaFn` shape + n_expert.
#[allow(clippy::too_many_arguments)]
pub type Q8MoeGateUpMma2tFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_expert: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// v3t TMA down (slot 503): `Q8MoeDownMmaFn` shape + n_expert.
#[allow(clippy::too_many_arguments)]
pub type Q8MoeDownMma2tFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_expert: u32,
    n_active: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dual-output align (slot 505).
#[allow(clippy::too_many_arguments)]
pub type MoeAlignDualFn = unsafe extern "C" fn(
    idx: *const core::ffi::c_void,
    sr32: *mut core::ffi::c_void,
    ss32: *mut core::ffi::c_void,
    be32: *mut core::ffi::c_void,
    sr16: *mut core::ffi::c_void,
    ss16: *mut core::ffi::c_void,
    be16: *mut core::ffi::c_void,
    pmap: *mut core::ffi::c_void,
    rows: u32,
    n_active: u32,
    n_expert: u32,
    mb32: u32,
    mb16: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// qwen4_exp grouped RMSNorm with the Gemma (1+w) affine (slot 506).
/// `(x, w, out, rows, groups, gd, eps, stream)` - w spans `groups*gd`.
#[allow(clippy::too_many_arguments)]
pub type Q4xGroupNorm1pFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    out16: *mut core::ffi::c_void,
    rows: u32,
    groups: u32,
    gd: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// hyper-connection mix reduce (slot 507): `(xn, gate, out, rows, hc, hidden, stream)`.
pub type Q4xHcMixFn = unsafe extern "C" fn(
    xn: *const core::ffi::c_void,
    gate: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    out16: *mut core::ffi::c_void,
    rows: u32,
    hc: u32,
    hidden: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// hyper-connection combine (slot 508): `(h, block_out, inj, rows, hc, hidden, stream)`.
pub type Q4xHcCombineFn = unsafe extern "C" fn(
    h: *mut core::ffi::c_void,
    block_out: *const core::ffi::c_void,
    inj: *const core::ffi::c_void,
    rows: u32,
    hc: u32,
    hidden: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// in-place `silu(m * inv)` (slot 509).
pub type Q4xScaleSiluFn = unsafe extern "C" fn(
    m: *mut core::ffi::c_void,
    n: u32,
    inv: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// PLE per-stream gate (slot 510): `(kn, qn, value, gv, rows, hc, hidden, stream)`,
/// kn/qn already group-normalized.
#[allow(clippy::too_many_arguments)]
pub type Q4xPleGateFn = unsafe extern "C" fn(
    kn: *const core::ffi::c_void,
    qn: *const core::ffi::c_void,
    value: *const core::ffi::c_void,
    gv: *mut core::ffi::c_void,
    rows: u32,
    hc: u32,
    hidden: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// dilated causal conv1d + silu (slot 511): `(src, w, out, n_tokens, dim, k, dil, stream)`.
/// `src` and `out` must be distinct buffers.
#[allow(clippy::too_many_arguments)]
pub type Q4xConvDilFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n_tokens: u32,
    dim: u32,
    k: u32,
    dil: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// one-token dilated conv off a carried window (slot 512):
/// `(x, win, w, out, dim, k, dil, stream)`; `win` is `[(k-1)*dil, dim]` oldest-first.
#[allow(clippy::too_many_arguments)]
pub type Q4xConvDilStepFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    win: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    dim: u32,
    k: u32,
    dil: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// GDN gated norm, plain w + sigmoid gate (slot 513):
/// `(x, z, w, out, n_rows, d, eps, stream)`.
#[allow(clippy::too_many_arguments)]
pub type Q4xGdnGatedNormFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    z: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    out16: *mut core::ffi::c_void,
    n_rows: u32,
    d: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// GDN conv split + repeat-interleave widening (slot 514):
/// `(conv, q, k, v, rows, k_heads, v_heads, k_dim, v_dim, stream)`.
#[allow(clippy::too_many_arguments)]
pub type Q4xGdnSplitWidenFn = unsafe extern "C" fn(
    conv: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    k: *mut core::ffi::c_void,
    v: *mut core::ffi::c_void,
    rows: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// fused hyper-connection combine + grouped (1+w) norm (slot 517):
/// `(h, block_out, inj, norm_w, xn, rows, hc, hidden, eps, stream)`.
#[allow(clippy::too_many_arguments)]
pub type Q4xCombineNormFn = unsafe extern "C" fn(
    h: *mut core::ffi::c_void,
    block_out: *const core::ffi::c_void,
    inj: *const core::ffi::c_void,
    norm_w: *const core::ffi::c_void,
    xn: *mut core::ffi::c_void,
    rows: u32,
    hc: u32,
    hidden: u32,
    eps: f32,
    xn16: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// narrow-K bf16 GEMV (slot 518): `y[o] = dot(w[o,:], x) + bias[o]`, f32
/// accumulation, one warp per output row. Identical contract to
/// `Bf16GemvF32Fn`; differs only in the block decomposition and therefore in
/// reduction ORDER, which is why it is a separate slot.
pub type Bf16GemvNkF32Fn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// batch-1 split-K f32 matvec (slot 519):
/// `(w, x, out, partials, counters, in_dim, out_dim, split, stream)`.
/// `partials` is `out_dim * split` f32 and `counters` is `out_dim` u32, both
/// CALLER-OWNED and address-stable; the kernel leaves `counters` zeroed so a
/// captured graph replays cleanly.
pub type MatvecF32SkFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    partials: *mut core::ffi::c_void,
    counters: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    split: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// bf16 GEMV + fused silu epilogue (slot 520):
/// `(w, bias, x, y, in_dim, out_dim, silu_rows, inv, stream)` -
/// `y[o] = dot(w[o,:], x) + bias[o]`, then `silu(y[o] * inv)` for
/// `o < silu_rows`. Bit-identical to the GEMV followed by a separate
/// scale+silu pass.
pub type Bf16GemvSiluF32Fn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    y16: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    silu_rows: u32,
    inv: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// multi-row narrow-K GEMV (slot 522):
/// `(w, bias, x, y, in_dim, out_dim, batch, stream)` - `x` is `[batch, in_dim]`
/// and `y` is `[batch, out_dim]`, row-major, like every other batched entry.
pub type Bf16GemvNkMrF32Fn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// per-slot dilated conv window step (slot 521):
/// `(x, win, w, out, slots, dim, k, dil, rows, stream)`.
pub type Q4xConvDilStepSlotsFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    win: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    slots: *const core::ffi::c_void,
    dim: u32,
    k: u32,
    dil: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// shared-expert scalar-gate fold (slot 515):
/// `(y, x, s, rows, n, stream)` - `y[r,:] += x[r,:] * sigmoid(s[r])`.
pub type Q4xAddGatedRowFn = unsafe extern "C" fn(
    y: *mut core::ffi::c_void,
    x: *const core::ffi::c_void,
    s: *const core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// NVFP4 MoE gate+up GEMV + fused swiglu (slot 516). Operands are the two
/// planes' `(data, scale, scale2)` triples, then
/// `(idx, x, y, in_dim, ff, k, batch, stream)`.
#[allow(clippy::too_many_arguments)]
pub type Q4xMoeGuSwigluFn = unsafe extern "C" fn(
    gdata: *const core::ffi::c_void,
    gscale: *const core::ffi::c_void,
    gscale2: *const core::ffi::c_void,
    udata: *const core::ffi::c_void,
    uscale: *const core::ffi::c_void,
    uscale2: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    k: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// g2 token-major gate_up (slot 504): bm16 CSR + sorted_slot + pair map.
#[allow(clippy::too_many_arguments)]
pub type Q8MoeGateUpG2Fn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    pmap: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_expert: u32,
    n_active: u32,
    max_blocks: u32,
    bm: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted Q8_0 MoE gate+up (see `KernelTableV1::q8_0_moe_gate_up_sorted`).
#[allow(clippy::too_many_arguments)]
pub type Q8MoeGateUpSortedFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fused: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted Q8_0 MoE down (see `KernelTableV1::q8_0_moe_down_sorted`).
#[allow(clippy::too_many_arguments)]
pub type Q8MoeDownSortedFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Q8_0 routed-expert gate+up (see `KernelTableV1::q8_0_moe_gate_up_dp4a`).
#[allow(clippy::too_many_arguments)]
pub type Q8MoeGateUpDp4aFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Q8_0 routed-expert down (see `KernelTableV1::q8_0_moe_down_dp4a`).
#[allow(clippy::too_many_arguments)]
pub type Q8MoeDownDp4aFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Q8_0 -> e4m3 K-padded planes (see `KernelTableV1::q8_0_to_f8w_pad`).
pub type Q8ToF8wPadFn = unsafe extern "C" fn(
    q8_data: *const core::ffi::c_void,
    q8_scale: *const core::ffi::c_void,
    f8_data: *mut core::ffi::c_void,
    f8_scale: *mut core::ffi::c_void,
    rows: u64,
    bpr: u32,
    bpr_pad: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Sorted e4m3 activation gather (see `KernelTableV1::moe_gather_e4m3`).
pub type MoeGatherE4m3Fn = unsafe extern "C" fn(
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    srow: *const core::ffi::c_void,
    xg: *mut core::ffi::c_void,
    sg: *mut core::ffi::c_void,
    in_dim: u32,
    srows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Padded-stride fused GEGLU quantize (see `KernelTableV1::quantize_e4m3_geglu2_pad`).
pub type QuantizeE4m3Geglu2PadFn = unsafe extern "C" fn(
    gu: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n_ff: u32,
    n_ff_pad: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// PAD-block-aware GEGLU quantize (see `KernelTableV1::quantize_e4m3_geglu2_pad_b`).
#[allow(clippy::too_many_arguments)]
pub type QuantizeE4m3Geglu2PadBFn = unsafe extern "C" fn(
    gu: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    bexp: *const core::ffi::c_void,
    n_ff: u32,
    n_ff_pad: u32,
    bm: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Uniq-experts histogram accumulate (see `KernelTableV1::moe_uniq_hist`).
pub type MoeUniqHistFn = unsafe extern "C" fn(
    idx: *const core::ffi::c_void,
    pairs: u32,
    n_expert: u32,
    out_accum: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Grouped tc5 e4m3 MoE gate_up (see `KernelTableV1::f8bs_moe_gemm_gu`).
#[allow(clippy::too_many_arguments)]
pub type F8bsMoeGemmGuFn = unsafe extern "C" fn(
    wdata: *const core::ffi::c_void,
    wsc: *const core::ffi::c_void,
    xg: *const core::ffi::c_void,
    sg: *const core::ffi::c_void,
    bexp: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    rows_per_e: u32,
    n_expert: u32,
    srows_pad: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Grouped tc5 e4m3 MoE down (see `KernelTableV1::f8bs_moe_gemm_dn`).
#[allow(clippy::too_many_arguments)]
pub type F8bsMoeGemmDnFn = unsafe extern "C" fn(
    wdata: *const core::ffi::c_void,
    wsc: *const core::ffi::c_void,
    xg: *const core::ffi::c_void,
    sg: *const core::ffi::c_void,
    bexp: *const core::ffi::c_void,
    srow: *const core::ffi::c_void,
    sslot: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    in_dim: u32,
    rows_per_e: u32,
    n_expert: u32,
    srows_pad: u32,
    max_blocks: u32,
    n_active: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Dual-weight MoE head norm + q8 quant (see `KernelTableV1::moe_head`).
pub type MoeHeadFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    gamma: *const core::ffi::c_void,
    pre2: *const core::ffi::c_void,
    rn: *mut core::ffi::c_void,
    pn: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    qs: *mut core::ffi::c_void,
    n: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// topk + per-expert scale fold (see `KernelTableV1::moe_topk_scaled`).
pub type MoeTopkScaledFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    n_expert: u32,
    k: u32,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// MoE combine trailer fusion (see `KernelTableV1::moe_tail`).
pub type MoeTailFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    dn: *const core::ffi::c_void,
    pn1: *const core::ffi::c_void,
    pn2: *const core::ffi::c_void,
    postw: *const core::ffi::c_void,
    n: u32,
    eps: f32,
    os: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Per-expert scalar fold into top-k weights (see `KernelTableV1::moe_scale_w`).
pub type MoeScaleWFn = unsafe extern "C" fn(
    w: *mut core::ffi::c_void,
    idx: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Shared-expert sigmoid gate fold (see `KernelTableV1::shexp_gate_add`).
pub type ShexpGateAddFn = unsafe extern "C" fn(
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    n_out: u32,
    n_in: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// K-split mma GEMM (see `KernelTableV1::q8_0_gemm_mma_ks`).
#[allow(clippy::too_many_arguments)]
pub type Q8GemmMmaKsFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Bias-folding K-split mma GEMM (see `KernelTableV1::q8_0_gemm_mma_ks_b`).
pub type Q8GemmMmaKsBFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Bias-folding small-batch dp4a GEMM (see `KernelTableV1::q8_0_gemm_mt_dp4a_b`).
pub type Q8GemmMtDp4aBFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Bias-folding stream-k mmq GEMM (see `KernelTableV1::q8_0_gemm_mmq_b`).
pub type Q8GemmMmqBFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    yq: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    fixup: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// f32 -> e4m3 + ue8m0 quantize (see `KernelTableV1::quantize_e4m3`).
pub type QuantizeE4m3Fn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Block-scale sorted-MoE gate+up (see `KernelTableV1::mxfp4_moe_gate_up_bs`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpBsFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    yq: *const core::ffi::c_void,
    ys: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    rows: u32,
    alpha: f32,
    limit: f32,
    // SwiGLU up-term: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u.
    up_add: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Block-scale sorted-MoE down (see `KernelTableV1::mxfp4_moe_down_bs`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownBsFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused qkv rope/append (see `KernelTableV1::qkv_rope_append_batch`).
pub type QkvRopeAppendBatchFn = unsafe extern "C" fn(
    qkv: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    k_cache: *mut core::ffi::c_void,
    v_cache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Paged twin of `QkvRopeAppendBatchFn` (gpt-oss G1): block-table K/V append.
/// `max_ctx` dropped; `block_tables` + `blocks_per_slot` appended before
/// `kv_dtype`. See `KernelTableV1::qkv_rope_append_batch_paged`.
#[allow(clippy::too_many_arguments)]
pub type QkvRopeAppendBatchPagedFn = unsafe extern "C" fn(
    qkv: *const core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    k_cache: *mut core::ffi::c_void,
    v_cache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// bm-parameterized moe_align (see `KernelTableV1::moe_align_bm`).
pub type MoeAlignBmFn = unsafe extern "C" fn(
    idx: *const core::ffi::c_void,
    sorted_row: *mut core::ffi::c_void,
    sorted_slot: *mut core::ffi::c_void,
    block_expert: *mut core::ffi::c_void,
    rows: u32,
    n_active: u32,
    n_expert: u32,
    bm: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// down_bs with fused residual fold (see `KernelTableV1::mxfp4_moe_down_bs_res`).
pub type Mxfp4MoeDownBsResFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    cnt: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Bias-carrying repacked dp4a GEMV (see `KernelTableV1::q8_0_gemv_dp4a_nc_b`).
#[allow(clippy::too_many_arguments)]
pub type Q8_0GemvDp4aNcBFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    ncols: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// f32 batched matvec (see `KernelTableV1::matvec_f32_batch`).
pub type MatvecF32BatchFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// K-split decode router matvec (slot 486).
pub type MatvecF32KsFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    scratch: *mut core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Two-projection fused GEMM with a segmented store (slot 529).
pub type Bf16Seg2GemmMmaFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Load-time permute of the hc up plane (slot 530).
pub type Bf16HcmixPermuteFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Co-residency gate for the f16 tensor-core lane (535). `0` declares that
/// another kernel may be resident, which clamps the tc5g/tc5gp K-split to 1
/// and makes its cross-CTA spin unreachable; `1` restores the free election.
pub type F16KsplitSetFn = unsafe extern "C" fn(i32) -> i32;

/// Batched-runs gated-delta recurrence (534): `(q, k, v, g, beta, state, out,
/// run_off, run_len, run_slot, n_runs, n_heads, head_dim, stream)`. grid is
/// (n_heads, n_runs); run `r` walks rows `run_off[r] .. +run_len[r]` against
/// slot `run_slot[r]`'s state.
pub type GatedDeltaRecurrentRunsFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Per-slot dilated conv step over a POSITION-indexed ring window (533):
/// `(x, win, w, out, slots, pos, dim, k, dil, rows, stream)`. The window
/// advance is fused in - the row for position `q` lives at ring row
/// `q % ((k-1)*dil)`, so nothing host-side enters the launch and one capture
/// serves every slot set.
pub type Q4xConvDilStepRingFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// PLE n-gram row gather off the device-resident fp8 table (532):
/// `(table, ids[rows*heads], out[rows*heads*width], scale, rows, heads,
/// width, stream)`. `ids` carries GLOBAL row ids - the per-head table offset
/// is folded in by the host hash, which touches no table memory.
pub type Q4xPleGatherFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    f32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// Up-GEMM with the hyper-connection mix tail fused into its epilogue (531).
pub type Bf16HcmixGemmFn = unsafe extern "C" fn(
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *const core::ffi::c_void,
    *mut core::ffi::c_void,
    u32,
    u32,
    u32,
    u32,
    *mut core::ffi::c_void,
) -> i32;

/// head+router+topk fusion (slot 487).
#[allow(clippy::too_many_arguments)]
pub type MoeRouterStageFn = unsafe extern "C" fn(
    w: *const core::ffi::c_void,
    x: *const core::ffi::c_void,
    logits: *mut core::ffi::c_void,
    dscale: *const core::ffi::c_void,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

pub type MoeHeadRouterFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    gamma: *const core::ffi::c_void,
    pre2: *const core::ffi::c_void,
    rw: *const core::ffi::c_void,
    dscale: *const core::ffi::c_void,
    pn: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    qs: *mut core::ffi::c_void,
    out_idx: *mut core::ffi::c_void,
    out_w: *mut core::ffi::c_void,
    n: u32,
    n_expert: u32,
    k: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Pair map for the prefill dn hybrid (slot 489).
pub type MoePairMapFn = unsafe extern "C" fn(
    srow32: *const core::ffi::c_void,
    sslot32: *const core::ffi::c_void,
    map: *mut core::ffi::c_void,
    n_active: u32,
    srp32: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// q8 GEGLU remap quantize (slot 490).
#[allow(clippy::too_many_arguments)]
pub type QuantQ8GegluRemapFn = unsafe extern "C" fn(
    gu: *const core::ffi::c_void,
    srow128: *const core::ffi::c_void,
    sslot128: *const core::ffi::c_void,
    map: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    n_ff: u32,
    n_active: u32,
    srp128: u32,
    act: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// tail+combine fold (slot 491).
#[allow(clippy::too_many_arguments)]
pub type MoeTailCombineFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    part: *const core::ffi::c_void,
    pn1: *const core::ffi::c_void,
    pn2: *const core::ffi::c_void,
    postw: *const core::ffi::c_void,
    n: u32,
    n_active: u32,
    eps: f32,
    os: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Batched fused dp4a MoE gate+up+swiglu (see
/// `KernelTableV1::mxfp4_moe_gate_up_dp4a_b`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpDp4aBatchFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    n_active: u32,
    batch: u32,
    alpha: f32,
    limit: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched fused dp4a MoE down (see `KernelTableV1::mxfp4_moe_down_dp4a_b`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownDp4aBatchFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fused_q: *const core::ffi::c_void,
    fused_s: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Deterministic MoE partials fold (see `KernelTableV1::moe_slot_combine`).
pub type MoeSlotCombineFn = unsafe extern "C" fn(
    part: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    embd: u32,
    n_active: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched device-to-device copy (see `KernelTableV1::batched_copy`).
pub type BatchedCopyFn = unsafe extern "C" fn(
    descs: *const core::ffi::c_void,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// int8 MoE gate+up mmq GEMM (see `KernelTableV1::mxfp4_moe_gate_up_mmq`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeGateUpMmqFn = unsafe extern "C" fn(
    gate_data: *const core::ffi::c_void,
    gate_scale: *const core::ffi::c_void,
    gate_bias: *const core::ffi::c_void,
    up_data: *const core::ffi::c_void,
    up_scale: *const core::ffi::c_void,
    up_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    fq: *mut core::ffi::c_void,
    fs: *mut core::ffi::c_void,
    in_dim: u32,
    ff: u32,
    max_blocks: u32,
    alpha: f32,
    limit: f32,
    // SwiGLU up-term offset: 1.0 = gpt-oss `silu(alpha*g)*(clamp(u)+up_add)`;
    // 0.0 = qwen plain `silu(g)*u`. See b2-fp4-grouped-moe-scope.
    up_add: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// int8 MoE down mmq GEMM (see `KernelTableV1::mxfp4_moe_down_mmq`).
#[allow(clippy::too_many_arguments)]
pub type Mxfp4MoeDownMmqFn = unsafe extern "C" fn(
    down_data: *const core::ffi::c_void,
    down_scale: *const core::ffi::c_void,
    down_bias: *const core::ffi::c_void,
    sorted_row: *const core::ffi::c_void,
    sorted_slot: *const core::ffi::c_void,
    block_expert: *const core::ffi::c_void,
    topk_w: *const core::ffi::c_void,
    fq: *const core::ffi::c_void,
    fs: *const core::ffi::c_void,
    residual: *mut core::ffi::c_void,
    ff: u32,
    embd: u32,
    n_active: u32,
    max_blocks: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Chunked gated delta rule (see `KernelTableV1::gated_delta_chunked`).
#[allow(clippy::too_many_arguments)]
pub type GatedDeltaChunkedFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    state: *mut core::ffi::c_void,
    out: *mut core::ffi::c_void,
    dw: *mut core::ffi::c_void,
    du: *mut core::ffi::c_void,
    aqk: *mut core::ffi::c_void,
    cg: *mut core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Device-side position advance (see `KernelTableV1::bump_rows_u32`).
pub type BumpRowsU32Fn = unsafe extern "C" fn(
    pos: *mut core::ffi::c_void,
    mrope: *mut core::ffi::c_void,
    r: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// int8 tensor-core Q8_0 GEMM (see `KernelTableV1::q8_0_gemm_mma`).
#[allow(clippy::too_many_arguments)]
pub type Q8GemmMmaFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// mmq-layout activation quantize (see `KernelTableV1::quantize_q8_mmq`).
pub type QuantizeQ8MmqFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    yq: *mut core::ffi::c_void,
    in_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused add+rmsnorm+quantize (see `KernelTableV1::add_rmsnorm_quant_mmq`).
#[allow(clippy::too_many_arguments)]
pub type AddRmsnormQuantMmqFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    xn: *mut core::ffi::c_void,
    yq: *mut core::ffi::c_void,
    n: u32,
    batch: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Prefill add+rmsnorm+quantize with an e4m3 / nvf4 epilogue (see
/// `KernelTableV1::add_rmsnorm_quant_e4m3_pf`). `proj_b16` != 0 reads the
/// residual as bf16.
#[allow(clippy::too_many_arguments)]
pub type AddRmsnormQuantPfFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    xn: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    scale: *mut core::ffi::c_void,
    n: u32,
    batch: u32,
    eps: f32,
    proj_b16: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused SwiGLU + mmq quantize (see `KernelTableV1::quantize_q8_mmq_swiglu`).
pub type QuantizeQ8MmqSwigluFn = unsafe extern "C" fn(
    gate: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    yq: *mut core::ffi::c_void,
    in_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// mmq-class int8 tensor-core GEMM (see `KernelTableV1::q8_0_gemm_mmq`).
#[allow(clippy::too_many_arguments)]
pub type Q8GemmMmqFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    yq: *const core::ffi::c_void,
    fixup: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// High-occupancy tiled mmq GEMM (see `KernelTableV1::q8_0_gemm_mmq_hi`) - as
/// `Q8GemmMmqFn` without the stream-k `fixup` pointer (tiled only).
pub type Q8GemmMmqHiFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    yq: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Like `Q8GemmMmqHiFn` but with a folded per-output-row bias - the K-padded
/// cp.async pipe GEMM used for the dense prefill rung (mm_pre b>64).
pub type Q8GemmMmqPipeBFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    yq: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Row-wise argmax (see `KernelTableV1::argmax_rows`).
pub type ArgmaxRowsFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Row-wise argmax with the runner-up and the confidence readouts (see
/// `KernelTableV1::argmax_top2_rows`).
pub type ArgmaxTop2RowsFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    alt: *mut core::ffi::c_void,
    stats: *mut core::ffi::c_void,
    probe: u32,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Whisper timestamp grammar (see `KernelTableV1::whisper_ts_rules`).
pub type WhisperTsRulesFn = unsafe extern "C" fn(
    logits: *mut core::ffi::c_void,
    state: *const core::ffi::c_void,
    rows: u32,
    vocab: u32,
    eot: u32,
    no_ts: u32,
    ts_begin: u32,
    max_init: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused per-row token sampling (see `KernelTableV1::sample_rows`).
pub type SampleRowsFn = unsafe extern "C" fn(
    logits: *const core::ffi::c_void,
    params: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused rmsnorm+q8 quantize (see `KernelTableV1::rmsnorm_quant_q8_batch`).
pub type RmsnormQuantQ8BatchFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    qs: *mut core::ffi::c_void,
    n: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused add+rmsnorm+e4m3 quantize (see
/// `KernelTableV1::add_rmsnorm_quant_e4m3_batch`).
#[allow(clippy::too_many_arguments)]
pub type AddRmsnormQuantE4m3BatchFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    proj: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    s8: *mut core::ffi::c_void,
    n: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// wqkv GEMM + fused combine/rope/append (see
/// `KernelTableV1::q8_0_gemm_mma_ks_qkv_rope`).
#[allow(clippy::too_many_arguments)]
pub type Q8GemmMmaKsQkvRopeFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    k_cache: *mut core::ffi::c_void,
    v_cache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    in_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_ctx: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Paged twin of `Q8GemmMmaKsQkvRopeFn` (gpt-oss G1): same GEMM, block-table
/// K/V append. `max_ctx` dropped; `block_tables` + `blocks_per_slot` appended
/// before `kv_dtype`. See `KernelTableV1::q8_0_gemm_mma_ks_qkv_rope_paged`.
#[allow(clippy::too_many_arguments)]
pub type Q8GemmMmaKsQkvRopePagedFn = unsafe extern "C" fn(
    data: *const core::ffi::c_void,
    scale: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xs: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    q_out: *mut core::ffi::c_void,
    k_cache: *mut core::ffi::c_void,
    v_cache: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    in_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    freq_scale: f32,
    corr_low: f32,
    corr_high: f32,
    ext_factor: f32,
    mscale: f32,
    batch: u32,
    block_tables: *const core::ffi::c_void,
    blocks_per_slot: u32,
    kv_dtype: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Cross-layer combine+norm+quantize fold (see
/// `KernelTableV1::moe_combine_rmsnorm_quant_q8`).
#[allow(clippy::too_many_arguments)]
pub type MoeCombineRmsnormQuantQ8Fn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    part: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    q: *mut core::ffi::c_void,
    qs: *mut core::ffi::c_void,
    n: u32,
    n_active: u32,
    eps: f32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Load-time g||u weight interleave (see `KernelTableV1::mxfp4_gu_interleave`).
pub type Mxfp4GuInterleaveFn = unsafe extern "C" fn(
    gate: *const core::ffi::c_void,
    up: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n_kb: u32,
    rows: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Pipelined-decode tick advance (see `KernelTableV1::pipe_advance`).
pub type PipeAdvanceFn = unsafe extern "C" fn(
    out: *const core::ffi::c_void,
    tokens: *mut core::ffi::c_void,
    positions: *mut core::ffi::c_void,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Conv-ext staging for batched spec (see `KernelTableV1::conv_ext_build_slots`).
#[allow(clippy::too_many_arguments)]
pub type ConvExtBuildSlotsFn = unsafe extern "C" fn(
    wins: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    mixed: *const core::ffi::c_void,
    ext: *mut core::ffi::c_void,
    batch: u32,
    km1: u32,
    r: u32,
    conv_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Segmented causal conv+SiLU (see `KernelTableV1::conv_chunk_ext`).
#[allow(clippy::too_many_arguments)]
pub type ConvChunkExtFn = unsafe extern "C" fn(
    ext: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    batch: u32,
    km1: u32,
    r: u32,
    conv_dim: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Ragged spec state restore (see `KernelTableV1::state_restore_slots`).
#[allow(clippy::too_many_arguments)]
pub type StateRestoreSlotsFn = unsafe extern "C" fn(
    states: *mut core::ffi::c_void,
    snap: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    committed: *const core::ffi::c_void,
    batch: u32,
    r: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Ragged spec conv-window commit (see `KernelTableV1::conv_commit_slots`).
#[allow(clippy::too_many_arguments)]
pub type ConvCommitSlotsFn = unsafe extern "C" fn(
    ext: *const core::ffi::c_void,
    wins: *mut core::ffi::c_void,
    slots: *const core::ffi::c_void,
    committed: *const core::ffi::c_void,
    batch: u32,
    km1: u32,
    r: u32,
    conv_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Gated delta recurrence v2 (see `KernelTableV1::gated_delta_recurrent_v2`).
#[allow(clippy::too_many_arguments)]
pub type GatedDeltaRecurrentV2Fn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    states: *mut core::ffi::c_void,
    snap: *mut core::ffi::c_void,
    out: *mut core::ffi::c_void,
    batch: u32,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Slot-indexed gated delta recurrence - B sequences each advance their own
/// state by one token (continuous-batching decode). q/k/v [B, n_heads, D],
/// g/beta [B, n_heads], states [n_slots, n_heads, D, D], slots [B].
pub type GatedDeltaRecurrentSlotsFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    g: *const core::ffi::c_void,
    beta: *const core::ffi::c_void,
    slots: *const core::ffi::c_void,
    states: *mut core::ffi::c_void,
    out: *mut core::ffi::c_void,
    batch: u32,
    n_heads: u32,
    head_dim: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Slot-indexed single-token conv+silu: B sequences advance their own persistent
/// window. wins [n_slots, k-1, conv_dim], x_new/out [B, conv_dim], slots [B].
pub type ConvStepSlotsFn = unsafe extern "C" fn(
    wins: *mut core::ffi::c_void,
    x_new: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    slots: *const core::ffi::c_void,
    batch: u32,
    conv_dim: u32,
    k: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Row-wise LayerNorm (mean/var + weight + bias) - the ViT norm.
/// `(x, w, b, out, rows, n, eps, stream)`.
pub type LayernormFn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    w: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    rows: u32,
    n: u32,
    eps: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// In-place GELU (tanh approximation, exactly ggml_gelu_f32) over n elements.
pub type GeluFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    n: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Broadcast bias add: `x[row][i] += bias[i]` over rows × n.
pub type BiasAddFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    bias: *const core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Vision M-RoPE (ggml ROPE_TYPE_VISION, indep_sects): full-head pairs
/// (p, p+head_dim/2); pair p < head_dim/4 rotates by pos_y·ts^p, else
/// pos_x·ts^(p-head_dim/4). positions [4, n_tokens] axis-major = [y, x, y, x].
pub type MropeVisionFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    positions: *const core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused bias + GELU + f16 store over rows × n:
/// `out[r][i] = f16(gelu(x[r][i] + bias[i]))` (tanh or erf form per slot).
pub type GeluBiasF16Fn = unsafe extern "C" fn(
    x: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused residual + projection bias: `x[r][i] += src[r][i] + bias[i]`.
pub type AddBiasResFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    bias: *const core::ffi::c_void,
    rows: u32,
    n: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// [`MropeVisionFn`] with the q/k projection bias folded into the load:
/// `x = rope(x + bias)`, bias broadcast per head-feature (n_heads·head_dim).
pub type MropeVisionBiasFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    bias: *const core::ffi::c_void,
    positions: *const core::ffi::c_void,
    n_tokens: u32,
    n_heads: u32,
    head_dim: u32,
    theta_scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Non-causal (bidirectional) ViT attention: q/k/v [n, heads, hd] f32 ->
/// out [n, heads*hd], online softmax over all n keys.
pub type VisionAttnFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    n: u32,
    n_heads: u32,
    head_dim: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Batched cross/self attention (see `KernelTableV1::vision_attn_x`). q/out are
/// `[n_batch, nq, heads, hd]`, k/v are `[n_batch, nkv, heads, hd]`; each batch
/// entry attends independently. Self-attention passes nq == nkv with the same
/// pointer three times; granite-vision's Q-Former cross-attention runs 16
/// queries against 64 encoder rows, batched over the windows.
pub type VisionAttnXFn = unsafe extern "C" fn(
    q: *const core::ffi::c_void,
    k: *const core::ffi::c_void,
    v: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    nq: u32,
    nkv: u32,
    n_heads: u32,
    head_dim: u32,
    n_batch: u32,
    scale: f32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Row gather with averaging fan-in (see `KernelTableV1::gather_rows_avg`):
/// `out[r][d] = mean over j<k of src[idx[r*k + j]][d]`. k == 1 is a plain
/// `get_rows`; k == 4 is a 2×2 average pool expressed through its index table.
pub type GatherRowsAvgFn = unsafe extern "C" fn(
    src: *const core::ffi::c_void,
    idx: *const core::ffi::c_void,
    out: *mut core::ffi::c_void,
    rows: u32,
    k: u32,
    width: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// In-place exact-erf GELU (see `KernelTableV1::gelu_erf`) - `0.5x(1+erf(x/√2))`,
/// ggml's `gelu_erf`. Distinct from [`GeluFn`], which is the tanh approximation:
/// granite-vision uses tanh in the tower and erf in the Q-Former FFN.
pub type GeluErfFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    n: u64,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Broadcast row add (see `KernelTableV1::add_rows_bcast`):
/// `x[r][d] += src[r % src_rows][d]`. Generalizes [`BiasAddFn`] (src_rows == 1)
/// to a cycling table - the Q-Former's learned query and encoder-position rows
/// repeat across windows.
pub type AddRowsBcastFn = unsafe extern "C" fn(
    x: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    rows: u32,
    src_rows: u32,
    width: u32,
    stream: *mut core::ffi::c_void,
) -> KernelStatus;

/// Fused-layout DeltaNet decay gate (see `KernelTableV1::delta_gate_ab`).
pub type DeltaGateAbFn = unsafe extern "C" fn(
    *const core::ffi::c_void, // ab [n_tokens, 2*n_heads] f32 (alpha||beta)
    *const core::ffi::c_void, // ssm_a [n_heads]
    *const core::ffi::c_void, // dt_bias [n_heads]
    *mut core::ffi::c_void,   // g [n_tokens, n_heads]
    *mut core::ffi::c_void,   // beta [n_tokens, n_heads]
    u32,                      // n_tokens
    u32,                      // n_heads
    *mut core::ffi::c_void,   // stream
) -> i32;

/// Two-weight fused repacked GEMM (see `KernelTableV1::q8_0_gemm_repacked_x2`).
pub type Q8GemmRepackedX2Fn = unsafe extern "C" fn(
    *const core::ffi::c_void, // weight A data (repacked Q8 int8)
    *const core::ffi::c_void, // weight A scales (f16 per-32)
    *const core::ffi::c_void, // weight B data
    *const core::ffi::c_void, // weight B scales
    *const core::ffi::c_void, // x [batch, in_dim] f32
    *mut core::ffi::c_void,   // yA [batch, outA]
    *mut core::ffi::c_void,   // yB [batch, outB]
    u32,                      // in_dim
    u32,                      // outA
    u32,                      // outB
    u32,                      // batch
    *mut core::ffi::c_void,   // stream
) -> i32;

/// Tile-image repack for the v4 decode plane (see `KernelTableV1::f8_repack_tiles`).
pub type F8RepackTilesFn = unsafe extern "C" fn(
    rowmajor: *const core::ffi::c_void,
    tiles: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Rowwise tcgen05 decode GEMM over the tile plane (see `KernelTableV1::f8t_gemm`).
pub type F8TGemmFn = unsafe extern "C" fn(
    wtiles: *const core::ffi::c_void,
    wrs: *const core::ffi::c_void,
    xq: *const core::ffi::c_void,
    xrs: *const core::ffi::c_void,
    part: *mut core::ffi::c_void,
    y: *mut core::ffi::c_void,
    in_dim: u32,
    out_dim: u32,
    batch: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

/// Fused GEGLU + row-quant, compact output (see `KernelTableV1`).
pub type QuantizeE4m3Geglu2RowFn = unsafe extern "C" fn(
    gu: *const core::ffi::c_void,
    q: *mut core::ffi::c_void,
    rscale: *mut core::ffi::c_void,
    n_ff: u32,
    rows: u32,
    stream: *mut core::ffi::c_void,
) -> i32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_info_layout_is_stable() {
        // if this changes, PACK_ABI_VERSION must bump - the test is the tripwire
        assert_eq!(std::mem::size_of::<PackInfo>(), 4 + 4 + 16 + 12);
    }

    #[test]
    fn kernel_table_layout_is_stable() {
        // 4 + 4 header, then 8 bytes per entry; niche optimization guarantees
        // Option<fn> == nullable pointer. Appends are ABI-compatible by the
        // growth rule (bump this count with the append); anything other than
        // a pure append must bump PACK_ABI_VERSION.
        // MERGE ORDER (append-only): when two branches both append, the
        // entries that were PUSHED first keep their positions and the later
        // appends move to the end. Renumbering a pushed entry instead would
        // shift a slot under any binary already built against it - the
        // phantom "CUDA error 1" failure, which is invisible to profilers and
        // to compute-sanitizer alike. The canonical order below is what every
        // pack rebuilds against, and every entry in it is a PURE APPEND unless
        // marked otherwise.
        //
        // The other recurring hazard is a slot landing without this count
        // being bumped: the assertion then sits red, and a stale number
        // disarms it for every later append too - a non-append edit would
        // slide through without the PACK_ABI_VERSION bump it owes. Re-derive
        // the count from the struct, never assume it.
        //
        // 223..229 f8w_pad..dec2, 230..232 moe_head / topk_scaled / tail,
        // 233 kquant_gemv_w4a8_nc, 234 matvec_ab_gate.
        // 235..237: the dec3 decode-band trio (gu/dn/combine).
        // 238..240: the decode-band f8 trio (gu_d32/dn_d32/geglu2_pad_b).
        // 241: the uniq-routing diagnostic (moe_uniq_hist).
        // 242: the fusion-program swiglu_fused (merged gate_up planes).
        // 243: row_slice (DN in_proj fusion split epilogue).
        // 244: f8d_gemm_mma_ks (native-fp8 decode lane).
        // 245/246: f8_gemm_w8_o16 + quantize_e4m3_swiglu_b16 (bf16 prefill
        // epilogue pair). 247: add_rmsnorm_quant_mmq_b16 (bf16 residual
        // consumer). 248: add_inplace_b16 (tail add). 249: bf16_to_f8w
        // (fp8-native ingestion).
        // 250/251: bf16_to_f8r + f8r_gemm_mma_ks (per-row scale-free stream).
        // 252: swiglu_fused_e4m3 (decode step-glue fusion).
        // 253: add_rmsnorm_e4m3_xn (decode norm+quant fuse).
        // 575/576: moe_cache_resolve + moe_cache_fill (MoE expert offload,
        // device-managed LRU slot cache over host-mapped expert planes).
        // 577: kquant_iq - i-quant family capability marker.
        // 254..256: the tile-linear f8 lane (f8w_repack_lin / f8_gemm_lin /
        // f8_gemm_lin_kt - access-pattern fix).
        // 257: add_rmsnorm_e4m3_xn_b16 (prefill glue fusion).
        // 258: gated_rmsnorm_e4m3 (DN out_proj prefill glue).
        // 259/260: f8w_repack_lin_bs + f8_gemm_lin_bs (official-FP8 byte
        // passthrough: data-only boxes + f32 block-scale plane).
        // 261: quantize_e4m3_swiglu_b16_gu (fused gate|up single-GEMM
        // prefill FFN epilogue).
        // 262: causal_conv1d_silu_qkv (DN prefill conv+split+norm fusion).
        // 263: causal_conv1d_silu_qkv_b16 (bf16-operand chain).
        // 264: gated_delta_chunked_vb16 (per-call v-bf16 route).
        // 265: addnorm_e4m3_b32 (per-32 wide-decode band-boundary fusion).
        // 266: attn_spec_batch_fin (spec-verify in-kernel finalize at one
        // split).
        // 267..269: gu epilogue-fusion trio - f8w_repack_lin_gui /
        // quantize_e4m3_geglu2i / f8_gemm_lin_gu (interleaved gu plane +
        // fused geglu+quant kt3 epilogue).
        // 270: attn_spec_lco_paged (in-kernel last-CTA-out combine on the
        // krs spec-FA arms).
        // 271: f8_gemm_lin_gu_pc (kt4a scale-free per-channel gu).
        // 272: f8_gemm_w8_pc (kt4 scale-free qkv/wo lin twin).
        // 277: spec_prep + spec_hgather (rung B2 pipeline).
        // 278: kv_nra_rows (kv-epilogue fold).
        // 279..280: draft_rs + spec_rs_resolve (canonical spec rejection
        // sampling - sampled drafts + full-q verify).
        // 281..282: spec_xh_stitch + hrow_gather (the drafter round's 224
        // single-row DtoD memcpys -> 4 launches).
        // 283..289: rowwise (strip-free) pc plane lane -
        // f8w_repack_lin_bs_gui + f8_gemm_lin_r / lin_kt_r / lin_gu_r /
        // lin_gu_pc_r / w8_pc_r / w8_pcd_r (data-only boxes + per-row wse).
        // 290: f8_gemm_w8_pc_qkv_r (fused qkv single-launch on the rowwise
        // plane - admission grid-width rung).
        // 291..295: chunk-band 16-bit streams - f8_gemm_w8_pc_qkv_r2 (o16
        // flag) + the bf16-in consumer twins qkv_norm_rope_batch2 /
        // kv_nra_rows2 / addnorm_e4m3_row2 / rmsnorm_add_scale2.
        // 296..303: attention streams - f16 pf_qn/pf_attn planes on the
        // mixed-tick route: qkv_norm_rope_batch3 (o16 q out),
        // attn_prefill_f16_paged2 / attn_spec_batch_paged2 /
        // attn_decode_batch_paged2 / attn_decode_batch_partial_paged2 /
        // attn_decode_batch_combine2 (a16/o16 flags), quantize_e4m3_i16 /
        // quantize_e4m3_row_i16 (f16-in twins).
        // 304..305: mul_softplus_head + moe_topk_sigmoid_batch (laguna
        // bring-up: per-head softplus attention gate + sigmoid router with
        // selection-only bias; slotted after the gemma4 16-bit and attention
        // lanes - trunk offsets never move).
        // 306: lag_qk_nra_rows (laguna decode-tick epilogue fold - q/k norm
        // + rope + paged k/v append, 6 launches -> 1).
        // 307..308: granite bring-up - scale_f32 (standalone x *= s for the
        // embedding/logit multipliers) and rope_yarn_batch_norm
        // (ROPE_TYPE_NORM interleaved pairs; granite is the first non-NEOX
        // family here).
        // 309: q36_qkg_nra_rows (qwen3.5 fused-plane prefill consumer -
        // split_qg + norms + mropes + appends off the one-GEMM plane,
        // 7 launches -> 1).
        // 310: f8_gemm_lin_kt_split (q36 DN - fused in_qkv|gate prefill
        // GEMM, two-buffer kt3 epilogue).
        // 311..314: granite-vision - vision_attn_x (the windowed Q-Former's
        // batched cross/self attention), gather_rows_avg (window
        // permutations + the 2x2 area downsampler), gelu_erf (the tower's
        // exact-erf GELU, not the tanh one) and add_rows_bcast (cycling
        // learned-query / encoder-position rows).
        // 315: kquant_gemm_w4a8_pipe (pipelined k-quant W4A8 GEMM, cp.async
        // off the >64-batch tile).
        // 316: kquant_gemm_w4a8_pipe2 (its genuinely-double-buffered sibling,
        // 2-deep raw ring + half-width tile_x, __launch_bounds__(256,1)).
        // 317: q8_0_gemv_repacked_multi (decode QKV / gate|up one-launch
        // merge).
        // 318: rope_norm_qk_append_paged (granite's rope+append band folded
        // to one launch).
        // 319: kquant_gemv_w4a8_multi (granite-30b decode QKV / gate|up
        // one-launch merge on the k-quant family).
        // 320: q8_0_gemv_dp4a_nc_multi (laguna batched-decode q|k|v|g /
        // gate|up one-launch merge on the nc class).
        // 321: gated_delta_recurrent_v2_packed (qwen GDN decode rows + short
        // prefill span walks in one launch).
        // 322: attn_prefill_f16_paged_vl (pf7 varlen packed prefill
        // attention, one launch/layer over all spans).
        // 323: gated_delta_chunked_rs_vl (varlen chunked-GDN, one stage1+walk
        // pair over all eligible spans).
        // 324: kquant_gemv_w4a8_glu (fused gate+up+SwiGLU decode GEMV).
        // 325-327: add_rmsnorm_e4m3_row, row_slice4, swiglu_e4m3_row - the
        // qwen decode-epilogue trio.
        // 328-334: the whisper decode lane - whisper_dec_attn,
        // whisper_embed_pos, whisper_qkv_split, whisper_kv_store,
        // whisper_ln_f16, whisper_res_ln_f16, whisper_bias_gelu_f16.
        // 335-341: the granite-speech conformer tower - gs_bias_silu_f16,
        // gs_bias_glu, gs_dwconv_bn_silu_f16, gs_conf_attn,
        // gs_bias_softmax_f16, gs_res_ln_f16, gs_post_ln_f16.
        // 342: gated_rmsnorm_e4m3_row (DN out_proj decode row fuse).
        // 343: argmax_top2_rows - the greedy pick plus log p(argmax) and one
        // probe token's probability, out of one log-sum-exp pass; whisper's
        // per-token confidence and no_speech_prob ride it.
        //   WIDENED in PLACE: the same slot now also returns the runner-up's
        //   id and log-probability plus the row's Renyi-2 entropy - margin
        //   (p1-p2) is what makes a confidence mark mean something, and all
        //   of it is free inside the pass that was already running.
        //   Deliberately not a new slot: the callers are all in this repo,
        //   and the rename (argmax_logprob_rows -> argmax_top2_rows) turns a
        //   stale call site into a compile error instead of a shifted
        //   argument list. Rebuild every consumer, examples included.
        // 344: whisper_ts_rules - whisper's own ApplyTimestampRules as a
        // device-side logit filter; without it greedy decoding picks
        // `<|notimestamps|>` and the timestamp mode silently does nothing.
        // 345: row_slice2_gate (delta-gate fold).
        // 346-347: quantize_e4m3_b32f + f8row_moe_gate_up_mma_geglu - the
        // flat per-output-row e4m3 expert lane, whose k loop carries no
        // weight scale at all. The weight converter reuses the dense lane's
        // existing q8_0_to_f8row.
        // 348-349: f8row_moe_gate_up_mma_geglu_f8 + f8row_moe_down_mma (the
        // down half of the same lane, plus the gate_up epilogue variant that
        // feeds it e4m3 instead of int8).
        // 350: conv_win_store_vl - the batched pass's conv-window store with
        // span geometry in device contents; graph-capture enabler and a
        // 576->48 launch fold.
        // 351-354: bf16_gemv_f32 / bf16_gemm_f32 / bf16_dequant_f32 /
        // embed_gather_bf16 - per-tensor quant dispatch, so a mixed UD file's
        // bf16 planes serve in their own class instead of being
        // down-quantized into the Q8_0 lane.
        // 355-363: the SiLU twins of the gated-FFN carrier set - muse-glimmer
        // is SwiGLU where gemma4 is GeGLU. Same kernels on the other branch of
        // the pack's pd_glu_act template; separate slots rather than an act
        // argument precisely so the append rule holds.
        // 364-365: rope_factors_batch_norm / qkv_norm_rope_batch4 - the
        // ROPE_TYPE_NORM twins of this engine's two rope carriers
        // (muse-glimmer ropes NORM where gemma4 ropes NEOX).
        // 366-367: qkv_norm_rope_batch5 / kv_nra_rows3 - the same two rope+V
        // arch constants on the fused QK epilogue and on the K/V-fold twin
        // that owns the paged append (gemma4 norms V weightlessly and
        // muse-glimmer does not touch V at all).
        // 368: gemma_qkv_nra3 - the same constants plus freq_scale on the
        // BATCHED DECODE epilogue.
        // 369: whisper_xattn_probs - softmax(QK^T) over the encoder frames
        // for the alignment heads only, which is what word-level timing is
        // derived from; the decode kernel is flash-style and never
        // materialises it.
        // 370-371: rope2d / pixel_shuffle_rows - muse-glimmer's VISION tower:
        // the NORM-paired 2D rope and the channel-outer pixel-shuffle merge.
        // 372-378: the vendored-CUTLASS fp8 decode set (f8cut_gemm,
        // f8t_detile, vdim_sync/register, pf_runs_register, f8cut_gemm_b16,
        // quantize_e4m3_glu2_row_b16), authored on a B200.
        // 379: bf16_to_f8row.
        // 380: sam_attn - SAM ViTDet attention with the decomposed rel-pos
        // bias, DeepSeek-OCR's first tower.
        // 381: moe_topk_softmax_all - DeepSeek-greedy router epilogue,
        // full-softmax topk weights.
        // 382: attn_decode_fused_gqa16.
        // 383: f16_gemm - in-house f16 GEMM for cuBLAS removal.
        // 384-386: rope_qk_append_paged_ring / add_rmsnorm_quant_q8_batch /
        // swiglu_quant_q8 - the deepseek-ocr decode launch folds.
        // 387: ocr_patches_u8 - the u8 tower patch stem.
        // 388-389: whisper_enc_qkv_split / whisper_kv_store_batch (encoder
        // fusion).
        // 390: bf16_gemv_mr_f32 - decode-band multi-row bf16 GEMV (the tile
        // GEMM collapses at batch <= 8).
        // 391: bf16_gemm_mma - bf16 tensor-core prefill GEMM, bf16-cast
        // activations (the reference's batched class).
        // 392-396: layernorm_f16 / gelu_bias_f16 / gelu_erf_bias_f16 /
        // add_bias_res / mrope_vision_bias - the PaddleOCR-VL tower
        // elementwise fusions, each bit-identical to the unfused ops it
        // replaces.
        // 397-398: nvf4_dequant / nvf4_gemv - modelopt NVFP4 checkpoint
        // consumers (the tensor-level dequant oracle and the W4A16-class GEMV
        // over the shipped triple).
        // 399-402: f8r_gemv / mamba_conv_step / mamba2_scan_seq /
        // mamba_rmsnorm_gated_g - the nemotron_h_moe mamba-2 lane,
        // arch-generic SIMT.
        // 403-404: nvf4_moe_up_relu2 / nvf4_moe_down_acc - NVFP4 MoE expert
        // consumers, cc-gated with 397-398.
        // 405: whisper_kv_store_slots - slot 389's store with a rows_per_slot
        // axis for batched admission; 389's signature stays frozen.
        // 406: mamba_conv_seq (bulk prefill) - the conv step over a token
        // span, arch-generic SIMT.
        // 407-408: nvf4_moe_up_relu2_bs / nvf4_moe_down_bs - sorted-tile
        // NVFP4 expert GEMMs over the moe_align layout, cc-gated with
        // 397-398.
        // 409-410: nvf4_moe_up_relu2_mt / nvf4_moe_down_part - wave-dense
        // fused decode expert GEMVs, cc-gated with 397-398.
        // 411: f16_mmaf_set - capture-time mmaf election gate for whisper's
        // dual-graph overlap routing.
        // 412-414: mamba_conv_step_batch / mamba2_scan_step_batch /
        // nvf4_gemv_batch - batched decode steps over slot arenas for the
        // continuous-batching tick (412-413 arch-generic SIMT, 414 cc-gated
        // with 397-398).
        // 415-416: q8_0_moe_up_relu2 / q8_0_moe_up_relu2_sorted - Q8_0
        // up-only relu^2 expert kernels for the nemotron GGUF lane,
        // arch-generic dp4a.
        // 417-418: mamba2_scan_seq_snap / copy_rows_strided - per-row-snapshot
        // scan for the verify rollback + the conv-input row snapshots.
        // Arch-generic SIMT.
        // 419: nvf4_gemm_mr - multi-row W4A16 nvf4 GEMM, the gemv_batch twin
        // with once-per-16-rows weight streaming.
        // 420: moe_topk_sigmoid_batch_sh - topk rows widened to k+ns with
        // constant shared pseudo-expert picks. Arch-generic SIMT.
        // 421: gemma_qkv_nra3_b16 - packed-bf16 q/k/v read twin for the
        // spec-verify b16-D election.
        // 422: nvf4_gemm_tc - tensor-core NVFP4 GEMM (exact-dequant bf16
        // mma), the batched lm_head class.
        // 423: attn_spec_batch_fin_e4 - fin twin with the in-kernel wo-in row
        // quantize (e4m3 plane + row scales).
        // 424: bf16_qkv_gemm_mma - fused q|k|v decode-band bf16 GEMM with
        // segmented store.
        // 425: attn_spec_batch_fin_e4s - fin twin storing e4m3 at static
        // scale 1.0 into pf_e4q (ones xrs).
        // 426: nvf4_gemm_f4 - checkpoint-plane W4A4 GEMM (fp4 x fp4
        // block-scale mma + Nvf4Plane epilogue).
        // 427: nvf4_gemm_f4b - v2, async scale planes + one barrier per
        // K-step, st ring.
        // 428: nvf4_gemm_f4s - split-K twin + reduce for starved tile grids.
        // 429: nvf4_gemm_f4c - KC=256 arm.
        // 430: nvf4_gemm_f4t - TMA + mbarrier ring.
        // 431: attn_decode_tc5_paged - tcgen05 decode attention,
        // final-output contract.
        // 432-433: f8cut_gemm_gluq / f8t_detile_gui - fused geglu+quantize gu
        // GEMM and its interleaved detiler.
        // 442: add_rmsnorm_quant_nvf4_batch - the nemotron MoE prologue's
        // three launches as one.
        // 443-445: the f16 SSM-state class - seq, seq+snap and
        // batched-decode walks over a half-width state arena.
        // 446-447: f16 SSM state <-> f32 checkpoint blob.
        // 448-449: the QKC compact-bf16 q/k pair (conv emitter + vl
        // chunked-GDN reader; one caller-side latch drives both).
        // 450: q8_0_moe_up_relu2_dec2 - the decode-band relu^2 expert UP.
        // 451: quantize_q8_relu2 - activation-fused quantize, the seam that
        // puts the shared expert on the dense ladder.
        // 452-454: tile-major NVFP4 plane twins (lm_head repack rung) -
        // gemv_batch / mr / tc over the [tile][stage][row] layout.
        // 455-457: fragment-layout NVFP4 plane twins - the same trio over the
        // mma-fragment-ordered blocks.
        // 458: attn_decode_fmha16 - Q16xKv128 tensor-core decode attention
        // for the muse hd128/G16 geometry.
        // 459: dflash_conv - DFlash2's grouped dynamic convolution.
        // 460-461: dflash_cand_ids / dflash_select - DFlash2's candidate
        // selector.
        // 462-463: gated_delta_verify_hold / gated_delta_commit_walk - the
        // snapshot-free spec-verify pair.
        // 464: dflash_chain_picks - the async block round's device-side pick
        // copy into the chain layout.
        // 465-468: nv4cut_sf_bytes / sf_repack / quant_a / gemm - the
        // checkpoint-native NVFP4 decode GEMM (CUTLASS sm100 block-scaled).
        // 469: dflash_cond_append - the conditioning fold (rung C).
        // 470-471: dflash_select_rs / dflash_rs_resolve - the sampled
        // selector walk + K-candidate rejection-sampling resolve (rung G).
        // 472-477: the tiled-layout NVFP4 MoE family: up/down _st (BM=8
        // skinny decode), _stw (BM=32 prefill twins), _mtt/_part_tt (r=1
        // mt-class group-GEMV twins).
        // 478: kquant_q40 - Q4_0-serves capability marker.
        // 479-480: kv_gather_blocks / kv_scatter_blocks - tier extent
        // pack/unpack, arch-generic SIMT; slot presence is the tier-transfer
        // capability probe.
        // 481: f8lin_gemv - b=1 GEMV over the tile-linear boxes.
        // 482: add_rmsnorm_q8_xn - granite's residual+norm+quantize fusion.
        // 492: nvf4_gemv_multi - merged q|k|v NVFP4 GEMV; the small-out_dim
        // occupancy fix.
        //
        // The size guard itself is now a COMPILE-TIME assertion at the struct,
        // against KERNEL_TABLE_SLOTS - this runtime one went stale twice, at
        // 3056 -> 3064 and again at slot 481, each time silently until someone
        // ran it. What is left here is the reading a const assert cannot make:
        // that the layout the loader assumes still holds.
        assert_eq!(
            std::mem::size_of::<KernelTableV1>(),
            8 + KERNEL_TABLE_SLOTS * 8
        );
        assert_eq!(
            std::mem::size_of::<Option<DequantF32Fn>>(),
            std::mem::size_of::<usize>()
        );
        // A nullable fn pointer must be exactly one slot wide, or "8 + N*8" is
        // arithmetic about nothing and every index past the first mis-sized
        // field resolves to the wrong function.
        assert_eq!(std::mem::size_of::<Option<F8LinGemvFn>>(), 8);
    }

    #[test]
    fn arch_str_stops_at_nul() {
        let mut info = PackInfo {
            magic: PACK_MAGIC,
            abi_version: PACK_ABI_VERSION,
            arch: [0; 16],
            pack_version: [0, 1, 0],
        };
        info.arch[..10].copy_from_slice(b"cuda-sm86\0");
        assert_eq!(info.arch_str(), Some("cuda-sm86"));
    }
}
