//! Native Kolibri-1 MLX. Packed affine weights, GPU-only top-6 routing,
//! paged BF16 KV, and continuous batching; no Python serving dependency.
//! Full paged sliding KV deliberately retains exact prefix restores. A
//! compact sliding-window snapshot cache is a separate qualification target.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_engine::paged_radix::PagedRadix;
use paddock_models::kolibri::*;
use std::collections::VecDeque;
#[cfg(test)]
mod bench;
#[cfg(test)]
mod diagnostic;
mod forward;
mod load;
mod projection;
mod serving;
#[cfg(test)]
mod tests;

const CHUNK: usize = 256;
const SPLITS: usize = 16;
const KVWIDTH: usize = KV_HEADS * HEAD_DIM;

struct Matrix {
    buffer: Buffer,
    k: usize,
    n: usize,
    bits: usize,
}
struct Layer {
    norm: Buffer,
    post_attn: Buffer,
    pre_ffn: Buffer,
    post_ffn: Buffer,
    qnorm: Buffer,
    knorm: Buffer,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    router: Buffer,
    bias: Buffer,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    shared_gate: Matrix,
    shared_up: Matrix,
    shared_down: Matrix,
    keys: Buffer,
    values: Buffer,
}
struct Scratch {
    ids: Buffer,
    meta: Buffer,
    pages: Buffer,
    output_rows: Buffer,
    decode_rows: Buffer,
    attention_tiles: Buffer,
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    parts: Buffer,
    delta: Buffer,
    normalized: Buffer,
    fg: Buffer,
    fu: Buffer,
    router: Buffer,
    picks: Buffer,
    probabilities: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    gate: Buffer,
    up: Buffer,
    expert_out: Buffer,
    logits: Buffer,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}
pub struct Kolibri {
    device: MetalDevice,
    embedding: Matrix,
    output_norm: Buffer,
    head: Matrix,
    layers: Vec<Layer>,
    scratch: Scratch,
    slots: Vec<Slot>,
    pending: VecDeque<Pending>,
    pool: KvPool,
    radix: PagedRadix,
    context: usize,
    page_stride: usize,
    weight_bytes: u64,
    kv_bytes: u64,
    pub last_gpu_seconds: f64,
}

impl Kolibri {
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= VOCAB)
        {
            return Err(MetalError::Model(
                "invalid Kolibri prefill slot/tokens/context".into(),
            ));
        }
        let s = &mut self.slots[slot];
        s.table.clear(&mut self.pool);
        s.history.clear();
        let blocks = self.radix.match_prefix(tokens);
        let reused = blocks.len() * BLOCK_TOKENS;
        s.table.share_prefix(&blocks, &mut self.pool);
        s.history.extend_from_slice(&tokens[..reused]);
        s.reused = reused;
        Ok(reused)
    }
    fn publish(&mut self, slot: usize) {
        let s = &self.slots[slot];
        self.radix
            .insert(&s.history, s.table.blocks(), &mut self.pool);
    }
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let reused = self.prepare(slot, tokens)?;
        let mut last = Vec::new();
        for chunk in tokens[reused..].chunks(CHUNK) {
            let pos = self.slots[slot].history.len();
            let rows = chunk
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (pos + i) as u32))
                .collect::<Vec<_>>();
            let output = if pos + chunk.len() == tokens.len() {
                vec![chunk.len() - 1]
            } else {
                Vec::new()
            };
            last = self.execute(&rows, &output)?;
        }
        self.publish(slot);
        Ok(last)
    }
}
