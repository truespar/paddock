//! Whole-model ownership. Lengths and scheduler progress are committed only
//! after one complete GPU walk and its accumulated validation status succeed.
use super::{FlashNextPlan, deltanet, moe, ple, qsa, residual};
use crate::{
    device::{Buffer, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
use paddock_models::mapped::MappedGguf;
use residual::{HyperConnection, WIDE, WIDTH};
use std::{collections::VecDeque, path::Path};

mod forward;
mod mlx_load;
mod prefix;
mod prompt;
mod serving;
#[cfg(test)]
mod tests;

const VOCAB: usize = 248320;
const CHUNK: usize = 128;
const MLX_CHUNK: usize = 1024;
enum Mixer {
    Delta(deltanet::Weights, deltanet::Cache),
    Qsa(qsa::Weights, qsa::Cache),
}
struct Layer {
    hc: HyperConnection,
    mixer: Mixer,
    ffn: moe::Weights,
}
struct Scratch {
    ids: Buffer,
    output_rows: Buffer,
    bad: Buffer,
    x: Buffer,
    h: Buffer,
    delta: Buffer,
    selected_h: Buffer,
    logits: Buffer,
    hc: residual::Workspace,
    moe: moe::Workspace,
    dn: deltanet::Workspace,
    qsa: qsa::Workspace,
    ple: ple::State,
}
#[derive(Default)]
struct Slot {
    table: BlockTable,
    length: usize,
    reused: usize,
    plan: prompt::Plan,
}
struct Pending {
    slot: usize,
    tokens: Vec<u32>,
    offset: usize,
}

/// Native IQ3-labelled GGUF and affine MLX text graphs. Explicit-only experimental
/// implementation, not an automatic-device or performance qualification.
/// F32 recurrent/indexer state; GGUF F16 / MLX BF16 paged KV. No vision/MTP.
pub struct FlashNext {
    device: MetalDevice,
    embedding: Weight,
    head: Weight,
    output_hc: HyperConnection,
    ple_weights: ple::Weights,
    ple_table: ple::Table,
    layers: Vec<Layer>,
    scratch: Scratch,
    affine_scratch: Option<Buffer>,
    slots: Vec<Slot>,
    pool: KvPool,
    prefix: prefix::PrefixCache,
    markers: Option<prompt::Markers>,
    pending: VecDeque<Pending>,
    context: usize,
    chunk: usize,
    // Maximum physical rows shared across requests; `chunk` still defines
    // each request's arithmetic and cache-compatible prompt boundaries.
    capacity: usize,
    pages: usize,
    weight_bytes: u64,
    cache_bytes: u64,
    poisoned: bool,
    pub last_gpu_seconds: f64,
}

impl FlashNext {
    pub fn load(path: &Path, context: usize, batch: usize, budget: Option<u64>) -> Result<Self> {
        objc2::rc::autoreleasepool(|_| Self::load_inner(path, context, batch, budget))
    }
    fn load_inner(path: &Path, context: usize, batch: usize, budget: Option<u64>) -> Result<Self> {
        if path.is_dir() {
            return Self::load_mlx(path, context, batch, budget);
        }
        let (cache_bytes, scratch_bytes) = Self::memory(context, batch)?;
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        let plan = FlashNextPlan::validate_map(&map)?;
        let weight_bytes = plan.backbone_bytes + plan.ple_bytes;
        let required = weight_bytes + cache_bytes + scratch_bytes;
        let device = MetalDevice::new(budget)?;
        if required > device.budget_bytes() {
            return Err(MetalError::Memory(format!(
                "Flash Next compressed weights + PLE + all layer caches + shared scratch need {required} bytes; grant {}",
                device.budget_bytes()
            )));
        }
        let pages = context.div_ceil(BLOCK_TOKENS);
        let embedding =
            residual::load_weight(&device, &map, "token_embd.weight", &[WIDTH, VOCAB], 14)?;
        let head = residual::load_weight(&device, &map, "output.weight", &[WIDTH, VOCAB], 14)?;
        let output_hc = HyperConnection::load(&device, &map, "output_hc", false)?;
        let ple_weights = ple::Weights::load(&device, &map)?;
        let ple_table = ple::Table::Resident(ple::load_table(&device, &map)?);
        let mut layers = Vec::with_capacity(48);
        for li in 0..48 {
            let hc = HyperConnection::load(&device, &map, &format!("blk.{li}.hc_attn"), true)?;
            let mixer = if li % 4 == 3 {
                Mixer::Qsa(
                    qsa::Weights::load(&device, &map, li)?,
                    qsa::Cache::new(&device, batch, pages)?,
                )
            } else {
                Mixer::Delta(
                    deltanet::Weights::load(&device, &map, li)?,
                    deltanet::Cache::new(&device, batch)?,
                )
            };
            layers.push(Layer {
                hc,
                mixer,
                ffn: moe::Weights::load(&device, &map, li)?,
            });
            if li % 8 == 7 {
                tracing::info!(layers = li + 1, "loading native Metal Flash Next");
            }
        }
        let scratch = Scratch::new(&device, context, batch, CHUNK)?;
        if device.allocated_bytes() != required {
            return Err(MetalError::Memory(format!(
                "Flash Next allocation ledger differs: {} vs planned {required}",
                device.allocated_bytes()
            )));
        }
        tracing::warn!(
            weight_bytes,
            cache_bytes,
            scratch_bytes,
            context,
            prefill_rows = CHUNK,
            batch,
            "EXPERIMENTAL Flash Next Metal full text graph; no vision/speculation or automatic-device qualification; exact-boundary prefix reuse not implemented"
        );
        Ok(Self {
            device,
            embedding,
            head,
            output_hc,
            ple_weights,
            ple_table,
            layers,
            scratch,
            affine_scratch: None,
            slots: (0..batch).map(|_| Slot::default()).collect(),
            pool: KvPool::with_blocks((pages * batch) as u32),
            prefix: prefix::PrefixCache::default(),
            markers: None,
            pending: VecDeque::new(),
            context,
            chunk: CHUNK,
            capacity: CHUNK,
            pages,
            weight_bytes,
            cache_bytes,
            poisoned: false,
            last_gpu_seconds: 0.,
        })
    }
    /// Exact native-buffer reservation. All products are bounded before
    /// allocation; OS/compiler/mmap residency is not called free GPU capacity.
    fn memory(context: usize, batch: usize) -> Result<(u64, u64)> {
        Self::memory_rows(context, batch, CHUNK)
    }
    fn memory_rows(context: usize, batch: usize, rows: usize) -> Result<(u64, u64)> {
        if !(1..=262144).contains(&context)
            || !(1..=64).contains(&batch)
            || !(1..=super::affine::MAX_ROWS).contains(&rows)
        {
            return Err(MetalError::Model("Flash Next bounds: context 1..=262144, batch 1..=64, physical prefill rows 1..=2048 (implementation limits, not qualification)".into()));
        }
        let pages = context.div_ceil(BLOCK_TOKENS);
        let pc = ple::State::cache_bytes(batch);
        let cache = 36 * deltanet::Cache::bytes(batch) + 12 * qsa::Cache::bytes(batch, pages) + pc;
        let scratch = deltanet::Workspace::bytes(rows, batch)
            + qsa::Workspace::bytes(rows, batch, pages)
            + ple::State::bytes(rows, batch, context)?
            - pc
            + residual::Workspace::bytes(rows)?
            + moe::Workspace::bytes(rows)?
            + Scratch::own_bytes(batch, rows);
        Ok((cache as u64, scratch as u64))
    }
    fn mlx_memory_rows(context: usize, batch: usize, rows: usize) -> Result<(u64, u64)> {
        let (cache, scratch) = Self::memory_rows(context, batch, rows)?;
        let pages = context.div_ceil(BLOCK_TOKENS);
        let saved = deltanet::Workspace::bytes(rows, batch)
            - deltanet::Workspace::mlx_bytes(rows, batch)
            + qsa::Workspace::bytes(rows, batch, pages)
            - qsa::Workspace::mlx_bytes(rows, batch, pages);
        Ok((cache, scratch - saved as u64))
    }
    fn healthy(&self) -> Result<()> {
        if self.poisoned {
            Err(MetalError::Device(
                "Flash Next state poisoned; reload the model".into(),
            ))
        } else {
            Ok(())
        }
    }
    // Host release never claims to repair a failed GPU walk. On the next use
    // of a healthy empty slot, all recurrent/logical histories are reset on
    // the same command buffer before that slot's first token is processed.
    fn release(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.table.clear(&mut self.pool);
            s.length = 0;
            s.reused = 0;
            s.plan = prompt::Plan::default();
        }
    }
    fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        self.healthy()?;
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= VOCAB)
            || self.pending.iter().any(|p| p.slot == slot)
        {
            return Err(MetalError::Model(
                "invalid Flash Next prefill slot/tokens/context or duplicate admission".into(),
            ));
        }
        self.release(slot);
        self.slots[slot].plan = self.prompt_plan(tokens);
        let reused = self.restore_prefix(slot, tokens)?;
        self.slots[slot].reused = reused;
        Ok(reused)
    }
    fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut pos = self.prepare(slot, tokens)?;
        let mut logits = Vec::new();
        // The checkpoint's GPU reference uses decode arithmetic for the
        // final prompt token. Preserve that boundary even for short prompts.
        while pos < tokens.len() {
            let (logical, mut n) = if self.is_mlx() {
                self.slots[slot].plan.at(pos)
            } else {
                let n = self.chunk.min(tokens.len() - pos);
                (n, n)
            };
            if self.prefix.enabled() {
                n = n.min(self.slots[slot].plan.until_cut(pos));
            }
            let rows = tokens[pos..pos + n]
                .iter()
                .enumerate()
                .map(|(i, &t)| (slot, t, (pos + i) as u32))
                .collect::<Vec<_>>();
            let outputs = if pos + n == tokens.len() {
                vec![n - 1]
            } else {
                Vec::new()
            };
            logits =
                self.execute_contracts(&rows, &outputs, self.is_mlx().then_some(&[logical]))?;
            self.capture_prefix(slot, tokens)?;
            pos += n;
        }
        Ok(logits)
    }
}
impl Scratch {
    fn own_bytes(batch: usize, rows: usize) -> usize {
        4 * (rows + batch + 1 + rows * (WIDTH * 2 + WIDE) + batch * (WIDE + VOCAB))
    }
    fn new(d: &MetalDevice, context: usize, batch: usize, rows: usize) -> Result<Self> {
        Self::new_inner(d, context, batch, rows, false)
    }
    fn new_mlx(d: &MetalDevice, context: usize, batch: usize, rows: usize) -> Result<Self> {
        Self::new_inner(d, context, batch, rows, true)
    }
    fn new_inner(
        d: &MetalDevice,
        context: usize,
        batch: usize,
        rows: usize,
        mlx: bool,
    ) -> Result<Self> {
        let b = |n| d.alloc(n * 4);
        Ok(Self {
            ids: b(rows)?,
            output_rows: b(batch)?,
            bad: b(1)?,
            x: b(rows * WIDTH)?,
            h: b(rows * WIDE)?,
            delta: b(rows * WIDTH)?,
            selected_h: b(batch * WIDE)?,
            logits: b(batch * VOCAB)?,
            hc: residual::Workspace::new(d, rows)?,
            moe: moe::Workspace::new(d, rows)?,
            dn: if mlx {
                deltanet::Workspace::new_mlx(d, rows, batch)?
            } else {
                deltanet::Workspace::new(d, rows, batch)?
            },
            qsa: if mlx {
                qsa::Workspace::new_mlx(d, rows, batch, context.div_ceil(BLOCK_TOKENS))?
            } else {
                qsa::Workspace::new(d, rows, batch, context.div_ceil(BLOCK_TOKENS))?
            },
            ple: ple::State::new(d, rows, batch, context)?,
        })
    }
}
