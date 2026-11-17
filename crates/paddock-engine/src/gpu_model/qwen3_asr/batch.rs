//! Qwen3-ASR continuous batching: paged KV, batched decode,
//! slot-resident multimodal prefill - the whisper.cpp-mutex differentiator.
//! Before this lane the family served SERIALLY (`enable_batch` default
//! `Ok(1)`), so concurrent transcription requests queued one at a time.
//!
//! Ported from granite's batch lane (the simplest in the repo and the shape
//! this family's serial bring-up already followed) with the qwen3 deltas:
//! no granite scalars (embedding/residual/logit scales, attention_scale),
//! per-head q/k RMS-norm between the projections and rope, NEOX rope
//! (granite is NORM), and audio-tower splices instead of DeepStack taps.
//! Deliberately not here in v1, recorded rather than forgotten:
//!
//! - chunked prefill / mixed ticks: ASR prompts are short (a 30 s clip is
//!   331 audio tokens + ~20 text tokens), so a blocking whole-prompt pass
//!   holds the tick for single-digit milliseconds, not the seconds that
//!   forced Sarathi chunking on the text families. Falls out of
//!   `supports_chunked_prefill` = false; the scheduler takes the classic
//!   batched-prefill path.
//! - device sampling / decode pipe / spec: transcripts are short and the
//!   [rows, vocab] readback at r <= max_batch is not the wall; add them if
//!   the cells say otherwise. No drafter ships for this model and
//!   prompt-lookup has nothing to match in an audio-token prompt.
//! - radix prefix cache: the only shareable prefix is the ~5-token
//!   `<|im_start|>user\n<|audio_start|>` scaffold - worthless.
//!
//! Decode ticks are captured into per-r CUDA graphs (granite's recipe):
//! scratch allocated once at enable so addresses never move, every replay-
//! read bound comes from a device buffer (d_pos, the block tables), and all
//! host work - table growth, the d_bt upload, row uploads - happens outside
//! the captured region.
//!
//! The audio splice needs no cross-tick registry: a multimodal prompt is
//! prefilled whole inside one `forward_prefill_multimodal` call, so the
//! tower outputs live on this call's stack and each `pf_rows` chunk
//! overwrites the placeholder rows that fall inside it. Decode ticks never
//! carry a splice (every audio row is consumed at prefill), which also
//! keeps the captured graph byte-identical to the text-only case.

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{GpuError, KvDtype, QuantW};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{
    gemv_any, mmq_pre_any, prefill_add_norm_quant, prefill_ffn_down_any, prefill_mm_any,
    prefill_mm_pre_any,
};
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};
use crate::service::MmChunk;

use super::{AudioOutput, GpuQwen3Asr, SendGraph, TokEmbd};

/// Rows per prefill chunk - the granite/laguna convention (PADDOCK_PF_ROWS,
/// 256-aligned). The scratch planes scale linearly with it; the default
/// covers every single-window clip (<= ~350 rows) in one pass and a 5-minute
/// long-form prompt in a handful.
pub(crate) fn pf_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PF_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (256..=8192).contains(&n) && n % 256 == 0)
            .unwrap_or(1024)
    })
}

/// VRAM slack the pool sizing leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;

/// FlashDecoding split ceiling (partial-scratch sizing).
const MAX_ATTN_SPLITS: usize = 16;

/// PADDOCK_NO_WMMA_PREFILL=1 pins the scalar tiled prefill attention - the
/// A/B reference for the tensor-core paged prefill tile.
fn no_wmma_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_WMMA_PREFILL").is_some())
}

/// PADDOCK_NO_GEMV_MULTI=1 restores the split q/k/v and gate/up decode GEMV
/// launches - the A/B reference for the one-launch merge.
fn no_gemv_multi() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_GEMV_MULTI").is_some())
}

/// True when the pack's batched partial fuses the q-group per KV head for
/// this shape. Must mirror the paged launcher's own predicate - split
/// budgeting and the partial-vs-plain dispatch both key off it. qwen3-asr is
/// 16q/8kv = group 2, inside the fused range, so this engages at every
/// batched decode width.
fn attn_gqa_fused(n_heads: usize, n_kv_heads: usize, batch: usize) -> bool {
    let group = n_heads.checked_div(n_kv_heads).unwrap_or(1);
    batch > 1
        && (2..=8).contains(&group)
        && n_kv_heads >= 2
        && n_heads == n_kv_heads * group
        && std::env::var_os("PD_NO_GQA_FUSE").is_none()
}

/// KV splits for the batched decode attention - position-INDEPENDENT so the
/// captured per-r graph can bake it (granite's formula, minus the fp8 vec8
/// arm this family cannot take: its GQA ratio is 2 and the vec8 walk is a
/// G=4 fp8 kernel).
fn attn_splits_for(n_heads: usize, n_kv_heads: usize, batch: usize, sm_count: usize) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    if attn_gqa_fused(n_heads, n_kv_heads, batch) {
        // the fused walk launches n_kv*batch blocks per split, so budget the
        // die on those, not on the q-head count
        let want = (2 * 3 * sm_count).div_ceil(n_kv_heads * batch).max(1);
        return want.min(MAX_ATTN_SPLITS);
    }
    if n_heads * batch >= 2 * 3 * sm_count {
        return 1; // die already saturated (un-fused shapes only)
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n.min(MAX_ATTN_SPLITS);
    }
    MAX_ATTN_SPLITS
}

pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// Prefill-mode dispatch cuts for one rows pass (granite's shape). `runs` =
/// contiguous same-slot row runs: an attention launch must never mix two
/// slots' query rows. `dec` = leading rows that are DECODE rows (q_len 1,
/// one per slot) in a fused mixed tick - one band, dispatched to the decode
/// attention kernel.
struct PfCuts {
    runs: Vec<(usize, usize)>,
    dec: usize,
}

/// A prompt queued for stall-free chunked prefill: rows
/// advance inside mixed ticks alongside every decode row, one
/// weight-amortized pass, instead of a blocking whole-prompt admission that
/// froze the streams - the c32 cell's serial floor. Audio rows carry
/// placeholder id 0; their tower embeddings live in the slot's span registry
/// and splice at embed time, so a chunk cut inside a clip is just another
/// cut. No radix keys: this lane has no prefix cache.
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    /// the ROW stream's ids, one per KV row (audio rows = placeholder 0)
    pub tokens: Vec<u32>,
    /// next row to compute
    pub cursor: usize,
}

/// Batched-lane scratch, sized once at enable for `cap`-row passes (decode
/// reuses the same planes at rows = live slots « cap).
pub(crate) struct BatchScratch {
    pub x: CudaSlice<f32>,  // residual [cap, embd]
    pub xn: CudaSlice<f32>, // normed rows [cap, embd]
    pub q: CudaSlice<f32>,
    /// post-q-norm plane - rope runs on this, and attention reads it. The
    /// raw projection stays in `q` (rmsnorm_batch is out-of-place), exactly
    /// like the serial lane's d_q/d_qn split.
    pub qn: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub kn: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    /// no-op sink logits [n_heads] - qwen3 has none, so the -inf identity.
    pub sinks: CudaSlice<f32>,
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    /// group-quantized activations for the r>1 decode mmq rungs AND the
    /// prefill helpers - widest consumer is the FFN down input (n_ff wide)
    pub xq: CudaSlice<i8>,
    pub xs: CudaSlice<f32>,
    /// per-16 int8 sums (k-quant mu term; unconditional so a Q4_K file takes
    /// the same code path as the elected Q8_0)
    pub ssums: CudaSlice<f32>,
    /// mma_ks K-split partials for the decode mmq rungs
    pub part: CudaSlice<f32>,
    /// flat-mmq activations for the prefill tensor-core GEMMs
    pub yq: CudaSlice<u8>,
    pub xsums: CudaSlice<f32>,
    /// Q8_0 mmq stream-k fixup plane
    pub skfix: CudaSlice<f32>,
    pub d_toks: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    /// [n_slots, vocab] logits - decode graphs bake this address.
    pub head_logits: CudaSlice<f32>,
    /// device sampler params [n_slots, 4] (inv_t, u, mode, pad) - :
    /// at c32 the host-sampled tick read [32, 151936] logits (~19 MB) back
    /// every step; device sampling returns bare ids instead
    pub d_par: CudaSlice<u32>,
    /// sampled token ids [n_slots]
    pub d_out: CudaSlice<u32>,
    /// FlashDecoding partial scratch [n_heads, n_slots, MAX_SPLITS, hd]
    pub attn_o: CudaSlice<f32>,
    /// per-partial (m, l) [n_heads, n_slots, MAX_SPLITS, 2]
    pub attn_ml: CudaSlice<f32>,
}

/// The whole batching state: pool + tables + scratch, one struct so
/// enable/teardown is atomic.
pub(crate) struct BatchState {
    pub n_slots: usize,
    /// Row capacity of every scratch plane = `pf_rows()` + one row per slot.
    pub cap: usize,
    /// logical blocks per slot (max_ctx/16) - the block table's slot stride
    pub bps: usize,
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    /// per-layer K/V stores over the shared pool
    pub kv: Vec<LayerKv>,
    pub sc: BatchScratch,
    pub kv_bytes: u64,
    /// captured decode ticks, keyed by row count r.
    pub graphs: HashMap<usize, SendGraph>,
    /// Per-slot audio splices: (prompt row where the clip starts, tower
    /// output). Registered at admission, read by `embed_rows` for whichever
    /// chunk rows fall inside a span (chunk cuts land mid-clip under mixed
    /// ticks), cleared on the slot's next admission or release. Decode rows
    /// sit past every span, so decode ticks never resolve one.
    pub spans: Vec<Vec<(usize, AudioOutput)>>,
    /// Slot-0 decode cursor for the serial `Generator` surface (`forward` /
    /// `reset`) once the engine is batched - the warmup generation and any
    /// serial-path caller decode through the batch lane at slot 0 instead of
    /// resurrecting the dense serial KV next to the pool.
    pub slot0_pos: usize,
}

/// One audio splice restricted to the current rows chunk: overwrite chunk
/// rows [dst_row, dst_row+len) with tower embedding rows starting at
/// src_row of the registry entry `spans[slot][span]` (indices, not
/// references - the registry lives inside the same BatchState the scratch
/// planes do, so `embed_rows` resolves them under a split field borrow).
struct Splice {
    dst_row: usize,
    src_row: usize,
    len: usize,
    slot: usize,
    span: usize,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

impl GpuQwen3Asr {
    /// Allocate the paged-KV + scratch state for up to `max_batch` slots.
    /// Returns the capacity actually enabled; 1 = stay on the serial loop
    /// (no paged kernels in the pack).
    pub(crate) fn enable_batch_impl(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        if !self.exec.has_paged_kv() {
            return Ok(1);
        }
        // Regime election: ASR prompts are ~160-380 rows, so the
        // 1024-row pipe/hi gate parked every prefill GEMM on the synchronous
        // mmq kernel - 26.3% of c32 GPU time. 128 measured better on
        // sm_120a at every concurrency, c32 by a wide margin.
        // An explicit PADDOCK_MMQ_HI_MIN pin still wins for A/B.
        crate::gpu_model::qwen35::elect_mmq_hi_min(128);
        // the serial dense KV (n_layer × max_ctx) makes way for the paged
        // stores; the serial lane stays lazily self-healing for batch-off use
        self.decode = None;
        self.scratch = None;
        self.prefill = None;
        self.batch = None;
        self.exec.trim_mem_pool();

        let hp = &self.hp;
        let (embd, nh, n_kv, hd) = (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv * hd;
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let n_layer = self.layers.len();

        // One block id addresses every layer (combined block table), so a
        // block costs all layers' K+V at once.
        let block_bytes = n_layer * 16 * kv_dim * 2 * kvb;
        let q_dim = nh * hd;
        let wide = hp.n_ff.max(q_dim).max(embd);
        let cap = pf_rows() + max_batch;
        let scratch_est = (cap * (2 * hp.n_ff + 3 * embd + 2 * q_dim + 3 * kv_dim) * 4)
            + (cap * wide * 4)
            + (8 * 64 * wide * 4)
            + (max_batch * hp.n_vocab * 4)
            + (256 << 20);
        // One arbiter sizes the KV store: crate::kv_plan.
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        let demand = kv_plan::Demand {
            family: "qwen3-asr",
            max_ctx: self.max_ctx,
            slots: max_batch,
            // Cap the pool at what (slots × max_ctx) can actually address - a
            // slot's block table holds exactly `bps` entries. No radix retention
            // slack: this lane has no prefix cache to hold blocks after death.
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            // one block id addresses every layer, so no KV is per-slot here
            per_slot_bytes: 0,
            // every slot must at least hold a full chunk's worth of prompt, or
            // admission deadlocks on its own first chunk
            floor_blocks_per_slot: pf_rows().div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                kv_plan::Reserve::new("prefill scratch", scratch_est as u64),
            ],
            ..Default::default()
        };
        // A real Err, not a lying Ok(1): service.rs treats Ok(c) as proof
        // self.batch is genuinely populated at capacity c.
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        let slots = plan.slots;

        let e = &self.exec;
        let mut kv = Vec::with_capacity(n_layer);
        let mut kv_bytes = 0u64;
        for _ in 0..n_layer {
            let bytes = pool_blocks * 16 * kv_dim * kvb;
            kv_bytes += 2 * bytes as u64;
            kv.push(LayerKv {
                k: e.alloc_u8(bytes)?,
                v: e.alloc_u8(bytes)?,
            });
        }

        let sc = BatchScratch {
            x: e.alloc(cap * embd)?,
            xn: e.alloc(cap * embd)?,
            q: e.alloc(cap * q_dim)?,
            qn: e.alloc(cap * q_dim)?,
            k: e.alloc(cap * kv_dim)?,
            kn: e.alloc(cap * kv_dim)?,
            v: e.alloc(cap * kv_dim)?,
            attn: e.alloc(cap * q_dim)?,
            proj: e.alloc(cap * embd)?,
            sinks: e.alloc_no_sinks(nh)?,
            ffn_gate: e.alloc(cap * hp.n_ff)?,
            ffn_up: e.alloc(cap * hp.n_ff)?,
            xq: e.alloc_i8(cap * wide)?,
            xs: e.alloc(cap * wide / 32)?,
            ssums: e.alloc(cap * wide / 16)?,
            part: e.alloc(8 * 64 * wide)?,
            yq: e.alloc_u8(wide.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            xsums: e.alloc(wide.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
            skfix: e.alloc(256 * 128 * 128 + 256)?,
            d_toks: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_slots: e.alloc_u32(cap)?,
            head_logits: e.alloc(slots * hp.n_vocab)?,
            d_par: e.alloc_u32(slots * 4)?,
            d_out: e.alloc_u32(slots)?,
            attn_o: e.alloc(nh * slots * MAX_ATTN_SPLITS * hd)?,
            attn_ml: e.alloc(nh * slots * MAX_ATTN_SPLITS * 2)?,
        };

        self.batch = Some(BatchState {
            n_slots: slots,
            cap,
            bps,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            sc,
            kv_bytes,
            graphs: HashMap::new(),
            spans: (0..slots).map(|_| Vec::new()).collect(),
            slot0_pos: 0,
        });
        self.chunked.clear();
        tracing::info!(
            "qwen3-asr batch: {slots} slots, {n_layer}-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} tokens), {} rows/chunk; left {:.2} GiB of the {:.2} GiB granted",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            pf_rows(),
            grant
                .saturating_sub((pool_blocks * block_bytes) as u64)
                .saturating_sub(scratch_est as u64) as f64
                / (1u64 << 30) as f64,
            grant as f64 / (1u64 << 30) as f64,
        );
        Ok(slots)
    }

    /// Back every `(slot, position)` this pass will touch with a physical
    /// pool block, re-uploading the device table once on growth.
    fn ensure_rows(&mut self, slots: &[u32], positions: &[u32]) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            bs.tables[s]
                .ensure(positions[i] as usize, &mut bs.pool)
                .map_err(|_| GpuModelError::PoolExhausted)?;
            let now = bs.tables[s].blocks().len();
            if now > before {
                grew = true;
                let base = s * bs.bps;
                for j in before..now {
                    bs.bt_host[base + j] = bs.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(drv)?;
        }
        Ok(())
    }

    /// Free-on-completion: an idle slot's blocks return to the shared pool
    /// immediately instead of being pinned until the slot is next reused.
    /// A dead slot's audio spans go too (device buffers), and its queued
    /// chunk entry - a client that hung up mid-prompt must not keep feeding
    /// rows into a slot the scheduler is about to hand to somebody else.
    pub(crate) fn release_inactive_slots_impl(&mut self, occupied: &[bool]) {
        if let Some(bs) = self.batch.as_mut() {
            for (s, occ) in occupied.iter().enumerate().take(bs.tables.len()) {
                if !occ {
                    if !bs.tables[s].blocks().is_empty() {
                        bs.tables[s].clear(&mut bs.pool);
                    }
                    bs.spans[s].clear();
                }
            }
        }
        self.chunked
            .retain(|c| occupied.get(c.slot).copied().unwrap_or(false));
    }

    pub(crate) fn pool_free_blocks_impl(&self) -> Option<usize> {
        self.batch.as_ref().map(|b| b.pool.free_blocks())
    }

    pub(crate) fn batch_kv_mem_bytes(&self) -> Option<u64> {
        self.batch.as_ref().map(|b| b.kv_bytes)
    }

    /// Admission prologue shared by both prefill entries: bounds-check the
    /// ROW count, drop the slot's previous sequence, and back the blocks the
    /// whole prompt will need up front (so a mid-prompt chunk can never
    /// discover the pool is dry with rows already written). Row-counted, not
    /// token-sliced, because a multimodal prompt's rows are not its tokens.
    fn admit_rows(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        if n_rows == 0 {
            return Err(GpuModelError::Unsupported("empty prompt".into()));
        }
        if n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {n_rows} tokens > max_ctx {}",
                self.max_ctx
            )));
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
            // a fresh sequence in this slot: the last request's audio spans
            // are gone, or the new prompt's rows would fall inside the old
            // clip's position span and get its embeddings spliced in -
            // fluent, wrong, silent
            bs.spans[slot].clear();
        }
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    // ── prefill ─────────────────────────────────────────────────────────────

    /// Build a prompt's ROW stream from its chunks: tower-encode each clip
    /// (mel is runner-precomputed on its own thread pool - ; the
    /// fallback keeps frontend-less callers working) and REGISTER the spans
    /// in the slot's registry for `embed_rows` to splice from. Caller must
    /// `admit_rows` first (it clears the previous registration).
    fn register_mm_rows(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<Vec<u32>, GpuModelError> {
        let mut ids: Vec<u32> = Vec::new();
        let mut spans: Vec<(usize, AudioOutput)> = Vec::new();
        for ch in chunks {
            match ch {
                MmChunk::Text(t) => ids.extend_from_slice(t),
                MmChunk::Audio { samples, mel } => {
                    let tower = self.tower.as_mut().ok_or_else(|| {
                        GpuModelError::Unsupported(
                            "audio request but no audio tower attached (--mmproj)".into(),
                        )
                    })?;
                    let owned;
                    let mel = match mel {
                        Some(m) => m,
                        None => {
                            owned = crate::audio::qwen3_asr_features(samples)
                                .map_err(GpuModelError::Unsupported)?;
                            &owned
                        }
                    };
                    let out = tower.encode(mel)?;
                    let off = ids.len();
                    // placeholder rows (overwritten by the splice)
                    ids.extend(std::iter::repeat_n(0u32, out.n_tokens));
                    spans.push((off, out));
                }
                MmChunk::Image { .. } => {
                    return Err(GpuModelError::Unsupported(
                        "qwen3-asr serves audio, not images".into(),
                    ));
                }
                MmChunk::OcrCrop(_) => {
                    return Err(GpuModelError::Unsupported(
                        "OCR crop directive on qwen3-asr - routing bug".into(),
                    ));
                }
                MmChunk::CtcTimestamps { .. } => {
                    return Err(GpuModelError::Unsupported(
                        "CTC timestamp directive on qwen3-asr - routing bug".into(),
                    ));
                }
                MmChunk::VisionPixels { .. } => {
                    return Err(GpuModelError::Unsupported(
                        "pixel-budget directive on qwen3-asr - routing bug".into(),
                    ));
                }
            }
        }
        if ids.is_empty() {
            return Err(GpuModelError::Unsupported("empty multimodal prompt".into()));
        }
        self.batch.as_mut().expect("batch enabled").spans[slot] = spans;
        Ok(ids)
    }

    /// Prefill a whole text prompt into `slot` (chunked at `pf_rows`) and
    /// return the last token's logits.
    pub(crate) fn forward_prefill_impl(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.admit_rows(slot, tokens.len())?;
        self.prefill_rows_whole(slot, tokens)
    }

    /// Multimodal prefill into a batch slot (the classic blocking path - the
    /// scheduler prefers the chunked queue via `prefill_begin_multimodal`).
    /// Returns (last-row logits, total rows) - rows is what the request
    /// bills as prompt_tokens.
    pub(crate) fn forward_prefill_multimodal_impl(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let n_rows: usize = chunks
            .iter()
            .map(|c| match c {
                MmChunk::Text(t) => t.len(),
                MmChunk::Audio { samples, mel } => mel.as_ref().map_or_else(
                    || crate::audio::audio_tokens_for_samples(samples.len()),
                    |m| crate::audio::audio_token_count(m.n_frames),
                ),
                MmChunk::Image { .. }
                | MmChunk::OcrCrop(_)
                | MmChunk::VisionPixels { .. }
                | MmChunk::CtcTimestamps { .. } => 0,
            })
            .sum();
        self.admit_rows(slot, n_rows.max(1))?;
        let ids = self.register_mm_rows(slot, chunks)?;
        debug_assert_eq!(ids.len(), n_rows, "row-count estimate vs built stream");
        let rows = ids.len();
        Ok((self.prefill_rows_whole(slot, &ids)?, rows))
    }

    /// Whole-prompt prefill body shared by the text and mm classic paths:
    /// pf_rows chunks through the shared rows pass, head on the last row.
    fn prefill_rows_whole(&mut self, slot: usize, ids: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let mut base = 0usize;
        let mut last_len = 0usize;
        for chunk in ids.chunks(pf_rows()) {
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            self.rows_pass_body(&rows, 0)?;
            base += chunk.len();
            last_len = chunk.len();
        }
        if slot == 0 {
            self.batch.as_mut().expect("batch enabled").slot0_pos = ids.len();
        }
        self.head_row(last_len - 1)
    }

    /// One weight-amortized pass over a ready-made row stream - the shared
    /// body of every prefill lane (whole-prompt and the stall-free mixed
    /// tick). `chunk` is (slot, pos, token) with same-slot rows contiguous;
    /// the leading `dec` rows are a fused tick's decode band. Audio splices
    /// resolve from the span registry by (slot, position), so a chunk cut
    /// inside a clip is just another cut.
    fn rows_pass_body(
        &mut self,
        chunk: &[(u32, u32, u32)],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        assert!(r <= self.batch.as_ref().expect("batch enabled").cap);
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        // contiguous same-slot runs over the PREFILL rows; the decode band is
        // one un-split leading run
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                _ => runs.push((i, 1)),
            }
        }
        // resolve audio splices per run before the walk (chunk-row coords)
        let mut splices: Vec<Splice> = Vec::new();
        {
            let bs = self.batch.as_ref().expect("batch enabled");
            for &(off, len) in &runs {
                let s = chunk[off].0 as usize;
                let p0 = chunk[off].1 as usize;
                for (si, (start, out)) in bs.spans[s].iter().enumerate() {
                    let lo = (*start).max(p0);
                    let hi = (start + out.n_tokens).min(p0 + len);
                    if lo < hi {
                        splices.push(Splice {
                            dst_row: off + (lo - p0),
                            src_row: lo - start,
                            len: hi - lo,
                            slot: s,
                            span: si,
                        });
                    }
                }
            }
        }
        let mut cuts = PfCuts { runs, dec };
        if dec > 0 {
            cuts.runs.insert(0, (0, dec));
        }
        self.upload_rows(&toks, &positions, &slots_v)?;
        self.embed_rows(r, &splices)?;
        self.layer_walk(r, Some(&cuts))
    }

    // ── decode ──────────────────────────────────────────────────────────────

    /// One batched decode step over rows 0..r (row i drives slot i - the
    /// engine's identity contract). Leaves [r, vocab] logits in head_logits.
    pub(crate) fn batch_step(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        let ident: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &ident)
    }

    /// `batch_step` with explicit slot ids (the slot-0 serial cursor uses it).
    pub(crate) fn batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        assert_eq!(r, positions.len());
        assert_eq!(r, slots.len());
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(r <= n_slots, "rows {r} > enabled {n_slots}");
        self.ensure_rows(slots, positions)?;
        self.upload_rows(tokens, positions, slots)?;
        self.step_replay(r)
    }

    /// The pure-device decode tick body - everything the per-r graph
    /// captures. All inputs are device buffers written before replay.
    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.embed_rows(r, &[])?;
        self.layer_walk(r, None)?;
        self.head_rows(r)
    }

    /// Record `body`'s launches into a CUDA graph (recording only).
    fn capture_body(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<(), GpuModelError>,
        what: &str,
    ) -> Result<SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("{what} pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("{what} begin_capture: {e}")))?;
        let rec = body(self);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("{what} end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver(format!("{what} capture produced no graph")))?;
        Ok(SendGraph(graph))
    }

    /// Replay the fixed-r decode tick, capturing it first if unseen.
    fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .graphs
            .contains_key(&r)
        {
            let g = self.capture_body(|s| s.step_body(r), "decode")?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .graphs
                .insert(r, g);
        }
        self.batch.as_ref().expect("batch enabled").graphs[&r]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("decode graph launch: {e}")))?;
        Ok(())
    }

    /// Host->device row streams: tokens, positions, slots.
    fn upload_rows(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        let st = &self.exec.stream;
        let mut t = sc
            .d_toks
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_toks".into()))?;
        st.memcpy_htod(tokens, &mut t).map_err(drv)?;
        let mut p = sc
            .d_pos
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_pos".into()))?;
        st.memcpy_htod(positions, &mut p).map_err(drv)?;
        let mut s = sc
            .d_slots
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_slots".into()))?;
        st.memcpy_htod(slots, &mut s).map_err(drv)?;
        Ok(())
    }

    /// Gather the rows' embeddings; audio splices overwrite their placeholder
    /// rows (prefill only - decode ticks pass no splices).
    fn embed_rows(&mut self, r: usize, splices: &[Splice]) -> Result<(), GpuModelError> {
        let embd = self.hp.n_embd;
        let bs = self.batch.as_mut().expect("batch enabled");
        let (sc, spans) = (&mut bs.sc, &bs.spans);
        match &self.tok_embd {
            TokEmbd::Q8(t) => self
                .exec
                .embed_gather_batch_q8(t, &sc.d_toks, &mut sc.x, embd, r)?,
            TokEmbd::Kq(t) => self.exec.kquant_gather(t, &sc.d_toks, &mut sc.x, embd, r)?,
        }
        for s in splices {
            let out = &spans[s.slot][s.span].1;
            self.exec.copy_region(
                &out.embd,
                s.src_row * embd,
                &mut sc.x,
                s.dst_row * embd,
                s.len * embd,
            )?;
        }
        Ok(())
    }

    /// The whole-stack walk over r rows. `cuts`: Some = prefill mode (the
    /// serial lane's parity-proven prefill helper sequence, attention per
    /// same-slot run + a fused tick's decode band on the decode kernel);
    /// None = decode mode (granite's rung split: r==1 rides the tuned
    /// single-row GEMVs, r>1 the batched mmq class). Rungs key on MODE,
    /// never on r, so a warm tail chunk reproduces a cold chunk's bytes
    /// exactly - and a fused tick's decode rows ride the prefill GEMM class
    /// (one weight stream for both, the whole point of the fuse).
    fn layer_walk(&mut self, r: usize, cuts: Option<&PfCuts>) -> Result<(), GpuModelError> {
        let pf = cuts.is_some();
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, nh, n_kv, hd, n_ff) =
            (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.n_ff);
        let kv_dim = n_kv * hd;
        let q_dim = nh * hd;
        let eps = hp.eps;
        let scale = 1.0 / (hd as f32).sqrt();
        let rope = hp.rope;
        let kv_dtype = self.kv_dtype;
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;
        let r1 = r == 1 && !pf;
        // Tensor-core paged prefill attention: the pack's f16 tile takes any
        // GQA group at hd128; the fp8 arm does not include group 2 (16q/8kv),
        // so a kv8 serve falls to the scalar tiled walk - the recorded
        // ratio-2 fp8-tile gap, not a silent downgrade.
        let wmma_pf = pf
            && matches!(kv_dtype, KvDtype::Fp16)
            && exec.has_attn_prefill_f16_paged()
            && !no_wmma_prefill();

        for (li, layer) in self.layers.iter().enumerate() {
            let sc = &mut bs.sc;
            if pf {
                // attn residual is folded into the FFN-norm call below; the
                // first norm of the layer therefore takes no residual - the
                // exact serial prefill sequence
                prefill_add_norm_quant(
                    &exec,
                    &mut sc.x,
                    None,
                    false,
                    &layer.attn_norm.buf,
                    &mut sc.xn,
                    false,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.yq,
                    embd,
                    r,
                    eps,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wq,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.q,
                    r,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wk,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.k,
                    r,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wv,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.v,
                    r,
                )?;
            } else {
                exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, eps, r)?;
                if r1 {
                    // q|k|v as one launch where the pack has the merge - the
                    // split k/v grids are launch-latency-floored
                    match (&layer.wq, &layer.wk, &layer.wv) {
                        (QuantW::Q8(wq), QuantW::Q8(wk), QuantW::Q8(wv))
                            if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
                        {
                            exec.q8_0_gemv_repacked_multi(
                                &mut [(wq, &mut sc.q), (wk, &mut sc.k), (wv, &mut sc.v)],
                                &sc.xn,
                            )?;
                        }
                        _ => {
                            gemv_any(&exec, &layer.wq, &sc.xn, &mut sc.q)?;
                            gemv_any(&exec, &layer.wk, &sc.xn, &mut sc.k)?;
                            gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
                        }
                    }
                } else {
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                    mmq_pre_any(
                        &exec,
                        &layer.wq,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.q,
                        r,
                    )?;
                    mmq_pre_any(
                        &exec,
                        &layer.wk,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.k,
                        r,
                    )?;
                    mmq_pre_any(
                        &exec,
                        &layer.wv,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.v,
                        r,
                    )?;
                }
            }
            // per-head q/k RMS norm, then plain NEOX rope - the qwen3 deltas
            // granite doesn't have; identical op order to the serial lane
            exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, eps, r * nh)?;
            exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, eps, r * n_kv)?;
            exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, nh, hd, rope, r)?;
            exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope, r)?;
            let kvs = &mut bs.kv[li];
            exec.kv_append_batch_paged(
                &sc.kn,
                &mut kvs.k,
                &sc.d_pos,
                Some(&sc.d_slots),
                &bs.d_bt,
                bps,
                kv_dim,
                r,
                kv_dtype,
            )?;
            exec.kv_append_batch_paged(
                &sc.v,
                &mut kvs.v,
                &sc.d_pos,
                Some(&sc.d_slots),
                &bs.d_bt,
                bps,
                kv_dim,
                r,
                kv_dtype,
            )?;
            if let Some(c) = cuts {
                // per same-slot run: causal within each slot, every K/V row
                // of this chunk already appended above. A fused tick's decode
                // band (runs[0] when dec > 0) takes the decode kernel.
                for &(off, len) in &c.runs {
                    if off < c.dec {
                        exec.attn_decode_batch_rows_paged(
                            &sc.qn,
                            &kvs.k,
                            &kvs.v,
                            &sc.sinks,
                            &mut sc.attn,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            off,
                            len,
                            scale,
                            kv_dtype,
                        )?;
                    } else if wmma_pf {
                        exec.attn_prefill_f16_paged_at(
                            &sc.qn,
                            &kvs.k,
                            &kvs.v,
                            &sc.sinks,
                            &mut sc.attn,
                            &sc.d_pos,
                            &sc.d_slots,
                            off,
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            len,
                            scale,
                            kv_dtype,
                        )?;
                    } else if len > 24 && exec.has_attn_prefill_paged() {
                        exec.attn_prefill_rows_paged(
                            &sc.qn,
                            &kvs.k,
                            &kvs.v,
                            &sc.sinks,
                            &mut sc.attn,
                            &sc.d_pos,
                            &sc.d_slots,
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            off,
                            len,
                            scale,
                            kv_dtype,
                        )?;
                    } else {
                        exec.attn_decode_batch_rows_paged(
                            &sc.qn,
                            &kvs.k,
                            &kvs.v,
                            &sc.sinks,
                            &mut sc.attn,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            0,
                            off,
                            len,
                            scale,
                            kv_dtype,
                        )?;
                    }
                }
            } else {
                // FlashDecoding partial+combine when the unsplit grid would
                // starve the die; fused GQA shapes (group 2 here) take
                // partial+combine even at ns == 1 - the plain per-q-head
                // kernel re-reads each K/V tile `group` times
                let ns = attn_splits_for(nh, n_kv, r, exec.sm_count());
                let fused1 = ns == 1 && attn_gqa_fused(nh, n_kv, r);
                if (ns > 1 || fused1) && exec.has_attn_partial_batch_paged() {
                    exec.attn_partial_batch_paged(
                        &sc.qn,
                        &kvs.k,
                        &kvs.v,
                        &mut sc.attn_o,
                        &mut sc.attn_ml,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        kv_dim,
                        0,
                        ns,
                        r,
                        scale,
                        kv_dtype,
                    )?;
                    exec.attn_combine_batch(
                        &sc.attn_o,
                        &sc.attn_ml,
                        &sc.sinks,
                        &mut sc.attn,
                        nh,
                        hd,
                        ns,
                        r,
                    )?;
                } else {
                    exec.attn_decode_batch_paged(
                        &sc.qn,
                        &kvs.k,
                        &kvs.v,
                        &sc.sinks,
                        &mut sc.attn,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        kv_dim,
                        0,
                        r,
                        scale,
                        kv_dtype,
                    )?;
                }
            }
            if pf {
                prefill_mm_any(
                    &exec,
                    &layer.wo,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &sc.attn,
                    &mut sc.proj,
                    r,
                )?;
                prefill_add_norm_quant(
                    &exec,
                    &mut sc.x,
                    Some(&sc.proj),
                    false,
                    &layer.ffn_norm.buf,
                    &mut sc.xn,
                    false,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.yq,
                    embd,
                    r,
                    eps,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.gate,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.ffn_gate,
                    r,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.up,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.ffn_up,
                    r,
                )?;
                prefill_ffn_down_any(
                    &exec,
                    &layer.down,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.ffn_gate,
                    &sc.ffn_up,
                    &mut sc.proj,
                    n_ff,
                    r,
                )?;
                exec.add(&mut sc.x, &sc.proj, r * embd)?;
            } else {
                if r1 {
                    gemv_any(&exec, &layer.wo, &sc.attn, &mut sc.proj)?;
                } else {
                    exec.quantize_q8(&sc.attn, &mut sc.xq, &mut sc.xs, r * q_dim)?;
                    mmq_pre_any(
                        &exec,
                        &layer.wo,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.proj,
                        r,
                    )?;
                }
                exec.add(&mut sc.x, &sc.proj, r * embd)?;
                exec.rmsnorm_batch(&sc.x, &layer.ffn_norm.buf, &mut sc.xn, embd, eps, r)?;
                if r1 {
                    match (&layer.gate, &layer.up) {
                        (QuantW::Q8(wg), QuantW::Q8(wu))
                            if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
                        {
                            exec.q8_0_gemv_repacked_multi(
                                &mut [(wg, &mut sc.ffn_gate), (wu, &mut sc.ffn_up)],
                                &sc.xn,
                            )?;
                        }
                        _ => {
                            gemv_any(&exec, &layer.gate, &sc.xn, &mut sc.ffn_gate)?;
                            gemv_any(&exec, &layer.up, &sc.xn, &mut sc.ffn_up)?;
                        }
                    }
                    exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                    gemv_any(&exec, &layer.down, &sc.ffn_gate, &mut sc.proj)?;
                } else {
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                    mmq_pre_any(
                        &exec,
                        &layer.gate,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.ffn_gate,
                        r,
                    )?;
                    mmq_pre_any(
                        &exec,
                        &layer.up,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.ffn_up,
                        r,
                    )?;
                    exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * n_ff)?;
                    exec.quantize_q8(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, r * n_ff)?;
                    mmq_pre_any(
                        &exec,
                        &layer.down,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.proj,
                        r,
                    )?;
                }
                exec.add(&mut sc.x, &sc.proj, r * embd)?;
            }
        }
        Ok(())
    }

    /// Final norm + tied lm_head over rows 0..rows, leaving [rows, vocab] in
    /// head_logits. No logit scale - qwen3 has none.
    fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, eps) = (hp.n_embd, hp.eps);
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(&sc.x, &self.output_norm.buf, &mut sc.xn, embd, eps, rows)?;
        if rows == 1 {
            gemv_any(&exec, &self.lm_head, &sc.xn, &mut sc.head_logits)?;
        } else {
            exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, rows * embd)?;
            mmq_pre_any(
                &exec,
                &self.lm_head,
                &sc.xq,
                &sc.xs,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?;
        }
        Ok(())
    }

    /// Stage residual row `row` at row 0 so a single-row head pass reads it.
    /// Bounced through `proj` because src and dst live in the same buffer.
    fn head_row_at(&mut self, row: usize) -> Result<(), GpuModelError> {
        let n_embd = self.hp.n_embd;
        if row > 0 {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src =
                sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .proj
                .try_slice_mut(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let ps = sc
                .proj
                .try_slice(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            let mut xd =
                sc.x.try_slice_mut(0..n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)
    }

    /// Prefill tail: head over residual row `row`, returning that one vocab
    /// row on the host.
    fn head_row(&mut self, row: usize) -> Result<Vec<f32>, GpuModelError> {
        let n_vocab = self.hp.n_vocab;
        self.head_row_at(row)?;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..n_vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    // ── stall-free chunked prefill + mixed ticks  ─────────

    /// Queue a TEXT prompt for chunked prefill (whole admission prologue now,
    /// so a mixed tick only has to move rows). Rare on this family - chat
    /// without audio - but the scheduler may route it here once
    /// `supports_chunked_prefill` holds.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        // a queued entry for this slot is STALE (the old request died and the
        // slot was reused): evict rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        self.admit_rows(slot, tokens.len())?;
        self.chunked.push(ChunkedPrefill {
            slot,
            tokens,
            cursor: 0,
        });
        Ok(())
    }

    /// Queue a whole MULTIMODAL admission wave (the scheduler's mm admission
    /// path once `supports_chunked_multimodal` holds): tower-encode each
    /// item's clips now - a few ms each, no encoder budget needed at this
    /// scale, unlike the 300 ms pictures granite budgets for - register the
    /// spans, and push the row plan on the queue. Rows then advance inside
    /// mixed ticks and the streams never freeze for a whole prompt: the c32
    /// serial-admission floor this rung exists to kill.
    pub(crate) fn prefill_begin_multimodal_impl(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        use crate::generator::MmAdmit;
        items
            .into_iter()
            .map(|(slot, chunks)| {
                self.chunked.retain(|c| c.slot != slot);
                let admit: Result<(), GpuModelError> = (|| {
                    let n_rows: usize = chunks
                        .iter()
                        .map(|c| match c {
                            MmChunk::Text(t) => t.len(),
                            MmChunk::Audio { samples, mel } => mel.as_ref().map_or_else(
                                || crate::audio::audio_tokens_for_samples(samples.len()),
                                |m| crate::audio::audio_token_count(m.n_frames),
                            ),
                            MmChunk::Image { .. }
                            | MmChunk::OcrCrop(_)
                            | MmChunk::VisionPixels { .. }
                            | MmChunk::CtcTimestamps { .. } => 0,
                        })
                        .sum();
                    self.admit_rows(slot, n_rows.max(1))?;
                    let ids = self.register_mm_rows(slot, &chunks)?;
                    debug_assert_eq!(ids.len(), n_rows, "row-count estimate vs built stream");
                    self.chunked.push(ChunkedPrefill {
                        slot,
                        tokens: ids,
                        cursor: 0,
                    });
                    Ok(())
                })();
                match admit {
                    Ok(()) => (slot, MmAdmit::Queued),
                    Err(e) => (
                        slot,
                        MmAdmit::Failed(crate::generator::GenError::Backend(format!("{e}"))),
                    ),
                }
            })
            .collect()
    }

    /// Drop slot's in-flight prefill (client hung up mid-prompt).
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// Pick this tick's chunk rows: FIFO over the queue, up to `budget` rows,
    /// splitting the last prompt if it does not fit. Returns the row stream
    /// and (queue index, rows taken, finishes?) per touched entry.
    fn plan_chunk(&self, budget: usize) -> (Vec<(u32, u32, u32)>, Vec<(usize, usize, bool)>) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        if self.chunked.is_empty() {
            return (rows, take);
        }
        let cap = budget.clamp(1, pf_rows());
        for (qi, c) in self.chunked.iter().enumerate() {
            if rows.len() >= cap {
                break;
            }
            let remaining = c.tokens.len() - c.cursor;
            let n = remaining.min(cap - rows.len()).max(1);
            for j in 0..n {
                let p = c.cursor + j;
                rows.push((c.slot as u32, p as u32, c.tokens[p]));
            }
            take.push((qi, n, n == remaining));
        }
        (rows, take)
    }

    /// Advance cursors and drop finished prompts from the queue.
    fn commit_chunk(
        &mut self,
        take: &[(usize, usize, bool)],
        finished_raw: Vec<(usize, crate::generator::FinishSample)>,
    ) -> Vec<(usize, crate::generator::FinishSample, usize)> {
        for &(qi, n, _) in take {
            self.chunked[qi].cursor += n;
        }
        let mut out = Vec::new();
        for (qi, fs) in finished_raw {
            let slot = self.chunked[qi].slot;
            let toks = std::mem::take(&mut self.chunked[qi].tokens);
            if slot == 0 {
                self.batch.as_mut().expect("batch enabled").slot0_pos = toks.len();
            }
            out.push((slot, fs, toks.len()));
        }
        self.chunked.retain(|c| !c.tokens.is_empty());
        out
    }

    /// Build the fused tick's row stream: decode rows first (one band), then
    /// as much of the prefill queue as the scratch capacity allows.
    fn fuse_rows(
        &self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> (
        Vec<(u32, u32, u32)>,
        usize,
        Vec<(usize, usize)>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: Vec<(u32, u32, u32)> =
            decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        // decode rows and the chunk share one scratch plane, bounded by
        // BatchState::cap (= pf_rows + one row per slot) - the band never
        // eats chunk rows
        let room = self
            .batch
            .as_ref()
            .expect("batch enabled")
            .cap
            .saturating_sub(dec_n);
        let (chunk_rows, take) = if room == 0 {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room))
        };
        rows.extend_from_slice(&chunk_rows);
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        (rows, dec_n, fin, take)
    }

    /// One FUSED mixed tick (granite's shape): decode rows and the prefill
    /// chunk in a single weight-amortized pass, decode rows device-sampled.
    /// Running them as two passes streams every weight twice - this fuse is
    /// where the admission cost stops being serial with decode.
    pub(crate) fn forward_mixed_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::{FinishSample, SampledStep};
        // nothing queued -> a plain decode tick on its captured graph
        if self.chunked.is_empty() {
            let step = if decodes.is_empty() {
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                }
            } else {
                let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
                let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
                // the scheduler's decode set is dense rows 0..n in slot order
                // only when every slot below the high-water mark is live; the
                // mixed tick's decode band carries explicit slots instead
                self.batch_step_slots_sampled(
                    &toks,
                    &pos,
                    &decodes.iter().map(|d| d.0 as u32).collect::<Vec<_>>(),
                    plans,
                )?
            };
            return Ok((step, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        self.rows_pass_body(&rows, dec_n)?;
        // decode rows first: bulk norm+head over rows 0..dec_n, sampled on
        // device. Must precede the finisher heads - head_row bounces its row
        // through x[0] and rewrites head_logits[0..vocab] (decode row 0's).
        let step = if dec_n > 0 {
            self.head_rows(dec_n)?;
            self.sample_head_rows(dec_n, plans)?
        } else {
            SampledStep {
                ids: Vec::new(),
                host_rows: Vec::new(),
            }
        };
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            let slot = self.chunked[qi].slot;
            let plan = fin_plans.iter().find(|(s, _)| *s == slot).map(|(_, p)| *p);
            // a finisher whose plan is Device samples on the GPU like a
            // decode row: its [1, vocab] logits never leave the card
            let fs = match plan {
                Some(p @ crate::generator::RowSample::Device(_)) => {
                    self.head_row_at(row)?;
                    let s = self.sample_head_rows(1, std::slice::from_ref(&p))?;
                    FinishSample::Sampled(s.ids[0])
                }
                _ => FinishSample::Logits(self.head_row(row)?),
            };
            finished_raw.push((qi, fs));
        }
        let finished = self.commit_chunk(&take, finished_raw);
        Ok((step, finished))
    }

    /// The unsampled mixed tick (full logits readback) - the scheduler's
    /// non-device-sampling fallback path.
    pub(crate) fn forward_mixed_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.read_batch_logits(decodes.len())?, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        self.rows_pass_body(&rows, dec_n)?;
        let mut dec_logits = Vec::new();
        if dec_n > 0 {
            self.head_rows(dec_n)?;
            dec_logits = self.read_batch_logits(dec_n)?;
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((
                qi,
                crate::generator::FinishSample::Logits(self.head_row(row)?),
            ));
        }
        let finished = self
            .commit_chunk(&take, finished_raw)
            .into_iter()
            .map(|(slot, fs, n)| match fs {
                crate::generator::FinishSample::Logits(l) => (slot, l, n),
                crate::generator::FinishSample::Sampled(_) => unreachable!("unsampled mixed tick"),
            })
            .collect();
        Ok((dec_logits, finished))
    }

    // ── device sampling (granite's recipe) ───────────────────────

    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows stay
    /// mode 0 = untouched. RsVerify is a spec-only plan; this family has no
    /// drafter, so it can never arrive here.
    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> Vec<u32> {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                // P65 TruncCat is qwen35-only (supports_host_head); skip-safe
                RowSample::Device(DevicePlan::TruncCat { .. }) => {}
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        par
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head_logits rows 0..r on device with `plans`; only Host-plan
    /// rows pay a vocab-row readback. Assumes the head has already run.
    fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let par = Self::pack_samp_par(plans);
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
        }
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ids_view = sc
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
        let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let v = sc
                    .head_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Device-sampled decode tick: graph replay + sample_rows, ids come back
    /// as r u32s instead of the [r, vocab] logits plane.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        let r = tokens.len();
        let ident: Vec<u32> = (0..r as u32).collect();
        self.batch_step_slots_sampled(tokens, positions, &ident, plans)
    }

    /// The sampled tick with explicit slot ids - the mixed tick's
    /// nothing-queued branch uses it (its decode set is compacted, not the
    /// dense row-i = slot-i prefix).
    fn batch_step_slots_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        let r = tokens.len();
        assert_eq!(r, slots.len(), "one slot per row");
        self.ensure_rows(slots, positions)?;
        self.upload_rows(tokens, positions, slots)?;
        self.step_replay(r)?;
        self.sample_head_rows(r, plans)
    }

    /// Read the [rows, vocab] logits back to the host.
    pub(crate) fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.n_vocab;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }
}
