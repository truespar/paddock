//! Native Qwen3.5/3.6/3.8 Metal graph: paged head-256 attention plus Gated
//! DeltaNet with exact-boundary state checkpoints. No CUDA or CPU model path.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::{Weight, projections};
use paddock_engine::generator::{GenError, Generator};
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use std::collections::VecDeque;
use std::path::Path;

mod attention;
mod bonsai;
#[cfg(test)]
mod bonsai_tests;
mod checkpoint;
mod dflash;
mod forward;
mod geometry;
mod offload;
mod ternary;
#[cfg(test)]
mod ternary_add_tests;
#[cfg(test)]
mod ternary_tests;
use geometry::Geometry;
#[cfg(test)]
mod few_bench;
mod load;
mod lookup;
#[cfg(test)]
mod mlx_parity_tests;
#[cfg(test)]
mod mlx_spec_tests;
#[cfg(test)]
mod mlx_tests;
mod moe;
mod mtp;
mod multimodal;
#[cfg(test)]
mod multimodal_tests;
mod projection;
#[cfg(test)]
mod projection_tests;
#[cfg(test)]
mod quant_boundary_tests;
mod retention;
mod serving;
mod spec;
#[cfg(test)]
mod spec_tests;
#[cfg(test)]
mod tests;
mod vision;
mod workspace;

const CHUNK: usize = 512;
const KEY_HEADS: usize = 16;
const MAX_SPLITS: usize = 32;
const PREFILL_QUANTUM: std::time::Duration = std::time::Duration::from_millis(128);

struct FullAttention {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: Weight,
    k_norm: Weight,
    keys: Buffer,
    values: Buffer,
}
struct DeltaNet {
    qkv: Weight,
    z: Weight,
    alpha: Weight,
    beta: Weight,
    out: Weight,
    conv: Weight,
    a: Weight,
    dt: Weight,
    norm: Weight,
    index: usize,
}
enum Mixer {
    Full(FullAttention),
    Linear(DeltaNet),
}
struct Layer {
    norm: Weight,
    post_norm: Weight,
    mixer: Mixer,
    gate: Weight,
    up: Weight,
    down: Weight,
    // Dense FFN, or the shared FFN of a routed-expert block. The latter adds
    // a sigmoid gate and the independent routed branch before the residual.
    moe: Option<moe::Experts>,
}
struct Scratch {
    gemm: Buffer,
    ids: Buffer,
    outputs: Buffer,
    meta: Buffer,
    mrope: Buffer,
    limits: Buffer,
    pages: Buffer,
    spans: Buffer,
    chunks: Buffer,
    bounds: Buffer,
    attn_tiles: Buffer,
    decode_rows: Buffer,
    long_decode_tiles: Buffer,
    checkpoint_rows: Buffer,
    checkpoint_spans: Buffer,
    x: Buffer,
    norm: Buffer,
    delta: Buffer,
    gate: Buffer,
    up: Buffer,
    logits: Buffer,
    qraw: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    attn_parts: Buffer,
    qkv: Buffer,
    convolved: Buffer,
    z: Buffer,
    alpha: Buffer,
    beta: Buffer,
    gates: Buffer,
    prepared: Buffer,
}
#[derive(Default)]
struct Slot {
    cold_consulted: bool,
    // Distinguish a tiny cached prompt suffix from a decode candidate. The
    // former must keep the prompt's attention arithmetic, regardless of size.
    prefill_end: usize,
    table: BlockTable,
    history: Vec<u32>,
    reused: usize,
    cuts: Vec<usize>,
    mm: Option<multimodal::Layout>,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
    work: usize,
}
#[derive(Default)]
struct Checkpoint {
    reserved: bool,
    table: BlockTable,
    history: Vec<u32>,
    touched: u64,
    images: Vec<multimodal::ImageKey>,
}

pub struct Qwen35 {
    cold: Option<offload::Tier>,
    source_versions: Vec<crate::offload::FileVersion>,
    // Native MLX affine checkpoint: BF16 boundaries and HF DeltaNet order.
    mlx: bool,
    splash: bool,
    bonsai: Option<bonsai::Bonsai>,
    ternary: Option<ternary::Ternary>,
    geometry: Geometry,
    device: MetalDevice,
    embedding: Weight,
    output_norm: Weight,
    head: Weight,
    layers: Vec<Layer>,
    scratch: Scratch,
    moe_scratch: Option<moe::Workspace>,
    state: Buffer,
    conv: Buffer,
    slots: Vec<Slot>,
    pending: VecDeque<Pending>,
    cache: Vec<Checkpoint>,
    clock: u64,
    pool: KvPool,
    width: usize,
    ff: usize,
    vocab: usize,
    context: usize,
    page_stride: usize,
    state_slots: usize,
    eps: f32,
    rope: f32,
    rotary: usize,
    weight_bytes: u64,
    kv_bytes: u64,
    spec: Option<spec::Verify>,
    verifying: bool,
    greedy_output: bool,
    mtp: Option<mtp::Mtp>,
    dflash: Option<dflash::Dflash>,
    lookup: lookup::Lookup,
    vision: Option<vision::Vision>,
    encoding: VecDeque<multimodal::Encoding>,
    image_cache: Vec<multimodal::CachedImage>,
    image_cache_reused: u64,
    row_capacity: usize,
    pub last_gpu_seconds: f64,
    admission_cost: serving::AdmissionCost,
    #[cfg(test)]
    diagnostic_serial_prefill: bool,
}

impl Qwen35 {
    fn checkpoint_copy(&self, cmd: &Commands<'_>, from: usize, to: usize) {
        let g = self.geometry;
        self.dflash_checkpoint(cmd, from, to);
        if let Some(d) = &self.mtp {
            cmd.dispatch(
                "spec_copy",
                &[&d.pending, &d.pending],
                &[
                    (from * self.width) as u32,
                    (to * self.width) as u32,
                    self.width as u32,
                ],
                [self.width.div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.dispatch(
            "dn_checkpoint",
            &[&self.state, &self.conv],
            &[
                from as u32,
                to as u32,
                self.state_slots as u32,
                g.state() as u32,
                (g.conv() * 3) as u32,
                g.linear_layers() as u32,
            ],
            [
                (g.linear_layers() * (g.state() + g.conv() * 3)).div_ceil(256),
                1,
                1,
            ],
            256,
        );
    }

    fn evict_checkpoint(&mut self) -> bool {
        let Some(index) = self
            .cache
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.reserved && !c.history.is_empty())
            .min_by_key(|(_, c)| c.touched)
            .map(|(i, _)| i)
        else {
            return false;
        };
        self.spill_checkpoint(index);
        if retention::trace() {
            tracing::info!(
                index,
                tokens = self.cache[index].history.len(),
                "metal-cache-pressure-evict"
            );
        }
        self.cache[index].table.clear(&mut self.pool);
        self.cache[index].history.clear();
        self.cache[index].images.clear();
        true
    }

    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        self.require_committed()?;
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= self.vocab)
        {
            return Err(MetalError::Model(
                "invalid Qwen prefill slot, tokens or context".into(),
            ));
        }
        self.prepare_mm(slot, tokens, None)
    }

    fn prepare_mm(
        &mut self,
        slot: usize,
        tokens: &[u32],
        mm: Option<multimodal::Layout>,
    ) -> Result<usize> {
        self.require_committed()?;
        if self.pending.iter().any(|p| p.slot == slot) {
            return Err(MetalError::Model("slot has an unfinished prefill".into()));
        }
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= self.vocab)
        {
            return Err(MetalError::Model("invalid multimodal prefill plan".into()));
        }
        self.slots[slot].table.clear(&mut self.pool);
        self.slots[slot].history.clear();
        self.slots[slot].mm = mm;
        self.slots[slot].prefill_end = tokens.len();
        // The disk scheduler may already have landed this complete checkpoint.
        // Image admission consults durable pixel/layout identities before the
        // tower. Only direct text calls use this synchronous hot fallback.
        let consulted = std::mem::take(&mut self.slots[slot].cold_consulted);
        if !consulted && self.slots[slot].mm.is_none() {
            self.restore_cold(tokens)?;
        }
        // Two trailing page boundaries survive chat-template header edits
        // across either side of a page. Never restore arbitrary partial state.
        let last = (tokens.len() - 1) / BLOCK_TOKENS * BLOCK_TOKENS;
        self.slots[slot].cuts = [last.saturating_sub(BLOCK_TOKENS), last]
            .into_iter()
            .filter(|&n| n >= 3 * BLOCK_TOKENS)
            .filter(|&n| {
                self.slots[slot]
                    .mm
                    .as_ref()
                    .is_none_or(|m| !m.inside_image(n))
            })
            .collect();
        let matched = self
            .cache
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                !c.history.is_empty()
                    && !c.reserved
                    && c.history.len() < tokens.len()
                    && tokens.starts_with(&c.history)
                    && multimodal::prefix_images_match(
                        &c.images,
                        self.slots[slot].mm.as_ref().map_or(&[][..], |m| &m.keys),
                        c.history.len(),
                    )
            })
            .max_by_key(|(_, c)| c.history.len())
            .map(|(i, _)| i);
        let reused = if let Some(i) = matched {
            let cmd = self.device.begin()?;
            self.checkpoint_copy(&cmd, self.slots.len() + i, slot);
            cmd.finish()?;
            self.slots[slot]
                .table
                .share_prefix(self.cache[i].table.blocks(), &mut self.pool);
            self.slots[slot].history.clone_from(&self.cache[i].history);
            self.clock += 1;
            self.cache[i].touched = self.clock;
            self.cache[i].history.len()
        } else {
            0
        };
        // On a miss, pos=0 lazily zeros state and the convolution window in
        // their GPU producers. No host writes to multi-megabyte state slabs.
        self.slots[slot].reused = reused;
        if retention::trace() {
            tracing::info!(slot, tokens = tokens.len(), reused, checkpoint = ?matched,
                cache = ?self.cache.iter().enumerate().map(|(i, c)|
                    (i, c.history.len(), c.touched, c.reserved)).collect::<Vec<_>>(),
                "metal-cache-prepare");
        }
        Ok(reused)
    }

    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut position = self.prepare(slot, tokens)?;
        let mut logits = Vec::new();
        while position < tokens.len() {
            let n = CHUNK.min(tokens.len() - position);
            let rows: Vec<_> = tokens[position..position + n]
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (position + i) as u32))
                .collect();
            let selected = if position + n == tokens.len() {
                vec![n - 1]
            } else {
                Vec::new()
            };
            logits = self.execute(&rows, &selected)?;
            position += n;
        }
        Ok(logits)
    }
}
