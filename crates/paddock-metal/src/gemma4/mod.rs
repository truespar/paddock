//! Native Gemma 4 dense/MoE execution. Global KV is paged and copy-on-write;
//! sliding layers use bounded rings, including one append quantum of slack.
//! Both geometries share the exact-GGUF projection ladder, not a CPU model.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::{Weight, projections};
use paddock_engine::generator::{GenError, Generator};
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use std::collections::VecDeque;
use std::path::Path;

mod dflash;
mod dflash_budget;
mod diffusion;
mod diffusion_mlx;
#[cfg(test)]
mod diffusion_packed_tests;
mod diffusion_residency;
mod forward;
mod load;
mod mlx;
#[cfg(test)]
mod mlx_attention_tests;
#[cfg(test)]
mod mlx_operation_tests;
#[cfg(test)]
mod mlx_tests;
mod moe;
mod mtp;
mod multimodal;
#[cfg(test)]
mod multimodal_tests;
mod muse;
mod muse_vision;
#[cfg(test)]
mod perf_tests;
mod prefix;
mod serving;
mod sliced_prefill;
mod spec;
#[cfg(test)]
mod spec_tests;
#[cfg(test)]
mod tests;
mod tower;
mod vision;
#[cfg(test)]
mod vision_boundary_tests;

const CHUNK: usize = 512;
const HEADS: usize = 32;
const SPLITS: usize = 32;

struct Layer {
    heads: usize,
    moe: Option<moe::Experts>,
    norm: Weight,
    post_attn: Weight,
    ffn_norm: Weight,
    post_ffn: Weight,
    q: Weight,
    k: Weight,
    v: Option<Weight>,
    o: Weight,
    attn_gate: Option<Weight>,
    q_norm: Weight,
    k_norm: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    scale: f32,
    sliding: bool,
    keys: Buffer,
    values: Buffer,
}
impl Layer {
    fn hd(&self) -> usize {
        self.q.n / self.heads
    }
    fn kh(&self) -> usize {
        self.k.n / self.hd()
    }
    fn kv_width(&self) -> usize {
        self.hd() * self.kh()
    }
}
struct Scratch {
    // Allocation capacity is distinct from the text scheduler's elected chunk.
    rows: usize,
    ids: Buffer,
    meta: Buffer,
    limits: Buffer,
    pages: Buffer,
    outputs: Buffer,
    tiles: Buffer,
    decode_rows: Buffer,
    x: Buffer,
    norm: Buffer,
    delta: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    attn_gate: Buffer,
    parts: Buffer,
    gate: Buffer,
    up: Buffer,
    gemm: Buffer,
    logits: Buffer,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
    mm: Option<multimodal::Layout>,
}
#[derive(Default)]
struct Checkpoint {
    table: BlockTable,
    history: Vec<u32>,
    touched: u64,
    images: Vec<multimodal::ImageKey>,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}

pub struct Gemma4 {
    diffusion: Option<diffusion::Lane>,
    // Only explicitly validated graphs share the scheduler. Muse is not a
    // Gemma alias: the factory and every arithmetic fork inspect this tag.
    muse: bool,
    // Storage and arithmetic are independent of the scheduler/cache owners.
    // GGUF retains its qualified F32 graph; MLX uses BF16 operation cuts.
    mlx: bool,
    device: MetalDevice,
    embedding: Weight,
    output: Option<Weight>,
    output_norm: Weight,
    factors: Weight,
    layers: Vec<Layer>,
    scratch: Scratch,
    moe_scratch: Option<moe::Workspace>,
    mtp: Option<mtp::Mtp>,
    dflash: Option<dflash::Dflash>,
    vision: Option<tower::Tower>,
    image_markers: Option<(u32, u32)>,
    image_cache: Vec<multimodal::CachedImage>,
    image_cache_reused: u64,
    encoding: VecDeque<multimodal::Encoding>,
    spec: Option<spec::Verify>,
    verifying: bool,
    greedy_verify: bool,
    slots: Vec<Slot>,
    cache: Vec<Checkpoint>,
    clock: u64,
    pending: VecDeque<Pending>,
    prefill_phase: Option<sliced_prefill::Phase>,
    pool: KvPool,
    width: usize,
    ff: usize,
    vocab: usize,
    context: usize,
    window: usize,
    ring: usize,
    page_stride: usize,
    eps: f32,
    rope: [f32; 2],
    softcap: f32,
    logit_scale: f32,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}

fn copy_words(
    cmd: &Commands<'_>,
    src: &Buffer,
    dst: &Buffer,
    from: usize,
    to: usize,
    words: usize,
) {
    cmd.dispatch(
        "spec_copy_words",
        &[src, dst],
        &[from as u32, to as u32, words as u32],
        [words.div_ceil(256), 1, 1],
        256,
    );
}
