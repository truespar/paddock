//! The decoder's continuous-batching lane: paged KV, batched decode ticks,
//! chunked prefill of `[picture prefix | prompt]` - the PaddleOCR-VL lane's
//! shape with this graph's deltas:
//!
//! - **no positions anywhere** - not rope, not learned rows. KV lands at the
//!   sequence position (it bounds the causal attention), nothing else reads it;
//! - **LayerNorm with bias** (eps 1e-5) and biased projections; the FFN is
//!   fc1 + bias + exact GELU in the GEMM's epilogue (624), fc2 + bias +
//!   residual in its landing (835); out_proj lands the same way;
//! - **`layernorm_embedding` over every row** of a pass, picture rows included,
//!   ahead of layer 0 - the gather (x 32) or the splice first lands in `xe`;
//! - **the checkpoint's repetition penalty is applied on the device to every
//!   head row** (`np_pen_argmax` with `write_back`), so the logits a host reads
//!   and the ids the device picks are the same penalized numbers. The seen
//!   bitmap gets the prompt's text ids at admission and every decode row's
//!   input token as the row runs (a decode input IS the last committed token).
//!   The service therefore sees a greedy request (see the kernel note in
//!   `nemotron_parse.cuh` for why the family owns this).
//!
//! Decode ticks are captured per row count (granite's recipe); the bitmap
//! update rides inside the capture (it reads the device row streams).

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;
use half::f16;

use crate::gpu::{GpuError, GpuExecutor, HalfTensor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};
use paddock_models::nemotron_parse::DECODER_LN_EPS;

use super::load::GpuNemotronParse;

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;
/// FlashDecoding split ceiling (partial-scratch sizing).
const MAX_ATTN_SPLITS: usize = 16;
/// Head rows a slot can own in one tick: its decode row plus the MTP draft's
/// verify row.
pub(super) const ROWS_PER_SLOT: usize = 2;
/// Per-row extra penalized ids the penalty kernel reads (a verify row's own
/// uncommitted chunk tokens).
pub(super) const N_EXTRA: usize = 1;

/// KV splits for batched decode attention - position-independent so a
/// captured per-r graph can bake it. 16 q-heads x few rows underfills the
/// die; a ~3K-row picture prefix per sequence is exactly where splitting pays.
fn attn_splits_for(n_heads: usize, rows: usize, sm_count: usize) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    if n_heads * rows >= 2 * 3 * sm_count {
        return 1;
    }
    MAX_ATTN_SPLITS
}

/// `y = W x` for decode-class rows: `dense_w16` (one class at every row
/// count) where `w16`, else the GEMM the row count elects.
pub(super) fn lin(
    exec: &GpuExecutor,
    w16: bool,
    w: &HalfTensor,
    x16: &CudaSlice<f16>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    if w16 {
        exec.dense_w16_f16(w, x16, y, w.dims[1], rows)?;
    } else {
        exec.matvec_batch_f16(w, x16, y, rows)?;
    }
    Ok(())
}

pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

pub(crate) struct BatchScratch {
    pub x: CudaSlice<f32>,
    /// embedding staging (gather x 32 / picture splice) ahead of layernorm_embedding
    pub xe: CudaSlice<f32>,
    pub xn16: CudaSlice<f16>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    pub attn16: CudaSlice<f16>,
    pub ff16: CudaSlice<f16>,
    /// fused-landing fallback scratch, [cap][ffn] f32
    pub scr: CudaSlice<f32>,
    pub sinks: CudaSlice<f32>,
    pub attn_o: CudaSlice<f32>,
    pub attn_ml: CudaSlice<f32>,
    pub d_toks: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    /// final-normed hidden rows of the last head pass, [hcap][dm] (MTP input)
    pub hid: CudaSlice<f32>,
    pub head_logits: CudaSlice<f32>,
    /// penalty row -> slot map for head rows that are not decode-band rows
    pub d_rslot: CudaSlice<u32>,
    pub d_extra: CudaSlice<u32>,
    pub d_par: CudaSlice<u32>,
    pub d_out: CudaSlice<u32>,
    /// per-slot seen-id bitmaps, `words` u32 a slot
    pub seen: CudaSlice<u32>,
    /// per-slot final-normed hidden of the row behind the slot's pending
    /// token - the MTP head's `h`, [slots][dm]
    pub mtp_h: CudaSlice<f32>,
    /// (slot, id) lists for bitmap updates outside the decode band
    pub d_or_s: CudaSlice<u32>,
    pub d_or_t: CudaSlice<u32>,
    /// the decode band's rows-partial groups, `[first row, rows]` pairs
    pub d_groups: CudaSlice<u32>,
}

pub(crate) struct BatchState {
    pub n_slots: usize,
    /// scratch row capacity (pf_rows + decode rows) - mixed ticks size by it
    pub cap: usize,
    /// head-row capacity
    pub hcap: usize,
    pub bps: usize,
    pub words: usize,
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    pub kv: Vec<LayerKv>,
    pub sc: BatchScratch,
    pub kv_bytes: u64,
    pub graphs: HashMap<usize, super::SendGraph>,
    /// captured MTP draft / verify shapes (spec.rs)
    pub spec_graphs: HashMap<super::spec::GraphKey, super::SendGraph>,
    /// slot 0's decode cursor for the serial `Generator::forward` surface
    pub slot0_pos: usize,
    /// per slot: the KV position of the row whose hidden `mtp_h` holds
    /// (u32::MAX = none) - the MTP warmth test
    pub h_pos: Vec<u32>,
    /// groups in `d_groups` for the next decode band (`walk.rs`)
    pub n_groups: usize,
}

/// Contiguous same-slot runs over a pass's prefill rows (an attention launch
/// never mixes two slots' query rows); `dec` leading rows are the decode band.
pub(super) struct PfCuts {
    pub(super) dec: usize,
    pub(super) runs: Vec<(usize, usize)>,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

impl GpuNemotronParse {
    pub fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        if !self.exec.has_paged_kv() {
            return Err(GpuModelError::Unsupported(
                "nemotron-parse serving needs the paged-KV pack".into(),
            ));
        }
        self.batch = None;
        self.exec.trim_mem_pool();
        let cfg = self.cfg.clone();
        let (dm, nh, hd, ffn, vocab) =
            (cfg.d_model, cfg.n_heads, cfg.head_dim(), cfg.ffn, cfg.vocab);
        let kv_dim = cfg.n_kv_heads * hd;
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let n_layer = self.layers.len();
        let block_bytes = n_layer * 16 * kv_dim * 2 * kvb;
        let words = vocab.div_ceil(32);
        let hcap_for = |s: usize| s.max(1) * ROWS_PER_SLOT;
        let scratch_est = |s: usize| {
            let cap = pf_rows() + hcap_for(s);
            cap * (5 * dm + 2 * kv_dim + 2 * ffn) * 4
                + hcap_for(s) * (vocab + dm) * 4
                + s * words * 4
                + (128 << 20)
        };
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        let demand = kv_plan::Demand {
            family: "nemotron-parse",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            per_slot_bytes: 0,
            retention_blocks: 0,
            floor_blocks_per_slot: pf_rows().div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                kv_plan::Reserve::new("pass scratch", scratch_est(max_batch) as u64),
            ],
            ..Default::default()
        };
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        let slots = plan.slots;
        let hcap = hcap_for(slots);
        let cap = pf_rows() + hcap;

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
            x: e.alloc(cap * dm)?,
            xe: e.alloc(cap * dm)?,
            xn16: e.alloc_f16(cap * dm)?,
            q: e.alloc(cap * dm)?,
            k: e.alloc(cap * kv_dim)?,
            v: e.alloc(cap * kv_dim)?,
            attn: e.alloc(cap * dm)?,
            attn16: e.alloc_f16(cap * dm)?,
            ff16: e.alloc_f16(cap * ffn)?,
            scr: e.alloc(cap * ffn)?,
            sinks: e.alloc_no_sinks(nh)?,
            attn_o: e.alloc(nh * hcap * MAX_ATTN_SPLITS * hd)?,
            attn_ml: e.alloc(nh * hcap * MAX_ATTN_SPLITS * 2)?,
            d_toks: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_slots: e.alloc_u32(cap)?,
            hid: e.alloc(hcap * dm)?,
            head_logits: e.alloc(hcap * vocab)?,
            d_rslot: e.alloc_u32(hcap)?,
            d_extra: e.to_device_u32(&vec![u32::MAX; hcap * N_EXTRA])?,
            d_par: e.alloc_u32(hcap * 4)?,
            d_out: e.alloc_u32(hcap)?,
            seen: e.to_device_u32(&vec![0u32; slots.max(1) * words])?,
            mtp_h: e.alloc(slots.max(1) * dm)?,
            d_or_s: e.alloc_u32(hcap)?,
            d_or_t: e.alloc_u32(hcap)?,
            d_groups: e.alloc_u32(2 * hcap)?,
        };
        self.batch = Some(BatchState {
            n_slots: slots,
            cap,
            hcap,
            bps,
            words,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            sc,
            kv_bytes,
            graphs: HashMap::new(),
            spec_graphs: HashMap::new(),
            slot0_pos: 0,
            h_pos: vec![u32::MAX; slots],
            n_groups: 0,
        });
        tracing::info!(
            "nemotron-parse batch: {slots} slots, {n_layer}-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} rows), {} rows/chunk, kv {:?}",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            pf_rows(),
            self.kv_dtype,
        );
        Ok(slots)
    }

    /// Back every `(slot, position)` with a physical pool block.
    pub(super) fn ensure_rows(&mut self, slots: &[u32], pos: &[u32]) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            if bs.tables[s].ensure(pos[i] as usize, &mut bs.pool).is_err() {
                return Err(GpuModelError::PoolExhausted);
            }
            let now = bs.tables[s].blocks().len();
            // a table past its stripe would write the NEIGHBOUR slot's
            // entries; the service clamps generation to the window, so this
            // firing means that contract broke - refuse loudly
            if now > bs.bps {
                return Err(GpuModelError::ContextExceeded {
                    got: now * 16,
                    max: bs.bps * 16,
                });
            }
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

    pub fn release_inactive_slots(&mut self, occupied: &[bool]) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < bs.tables.len() && !bs.tables[s].blocks().is_empty() {
                bs.tables[s].clear(&mut bs.pool);
            }
        }
    }

    pub fn pool_free_blocks(&self) -> Option<usize> {
        self.batch.as_ref().map(|b| b.pool.free_blocks())
    }

    /// Admission checks + slot reset for an `n_rows`-row prompt, and the
    /// slot's seen set reset to the prompt's TEXT ids - the reference's
    /// penalty reads its `input_ids`, which never hold the picture rows.
    pub(super) fn admit(
        &mut self,
        slot: usize,
        n_rows: usize,
        text: &[u32],
    ) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        if slot >= bs.n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {}",
                bs.n_slots
            )));
        }
        if n_rows == 0 || n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {n_rows} rows out of range (max_ctx {})",
                self.max_ctx
            )));
        }
        bs.tables[slot].clear(&mut bs.pool);
        bs.h_pos[slot] = u32::MAX;
        let words = bs.words;
        let mut bits = vec![0u32; words];
        for &t in text {
            let t = t as usize;
            if t < words * 32 {
                bits[t >> 5] |= 1 << (t & 31);
            }
        }
        let mut dst = bs
            .sc
            .seen
            .try_slice_mut(slot * words..(slot + 1) * words)
            .ok_or_else(|| GpuError::Driver("seen slice".into()))?;
        self.exec.stream.memcpy_htod(&bits, &mut dst).map_err(drv)?;
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    pub(super) fn upload_rows(
        &mut self,
        tokens: &[u32],
        pos: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec.upload_u32(tokens, &mut bs.sc.d_toks)?;
        self.exec.upload_u32(pos, &mut bs.sc.d_pos)?;
        self.exec.upload_u32(slots, &mut bs.sc.d_slots)?;
        Ok(())
    }

    /// Gather `r` rows' token embeddings (x 32) into the staging plane - picture
    /// rows are overwritten by the caller's splice before [`Self::embed_norm`].
    pub(super) fn embed_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let dm = self.cfg.d_model;
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec
            .embed_gather_batch(&self.embed32, &bs.sc.d_toks, &mut bs.sc.xe, dm, r)?;
        Ok(())
    }

    /// `layernorm_embedding` over the staged rows into the residual stream.
    pub(super) fn embed_norm(&mut self, r: usize) -> Result<(), GpuModelError> {
        let dm = self.cfg.d_model;
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec.layernorm(
            &bs.sc.xe,
            &self.ln_emb.w,
            &self.ln_emb.b,
            &mut bs.sc.x,
            r,
            dm,
            DECODER_LN_EPS,
        )?;
        Ok(())
    }

    /// Mark the decode band's input tokens seen (rows 0..dec: one committed
    /// token per slot). Graph-safe: reads the device row streams.
    pub(super) fn seen_band(&mut self, dec: usize) -> Result<(), GpuModelError> {
        if dec == 0 {
            return Ok(());
        }
        let bs = self.batch.as_mut().expect("batch enabled");
        let words = bs.words;
        let sc = &mut bs.sc;
        self.exec
            .np_seen_or(&mut sc.seen, &sc.d_slots, &sc.d_toks, dec, words)?;
        Ok(())
    }

    /// `k1 > 1` marks a verify pass of slot-major chunks of exactly `k1`
    /// rows (pending + drafts): the chunk's rows share ONE walk over the
    /// slot's KV (`attn_spec_batch_paged`, per-row causal masks inside)
    /// instead of one walk a row - a page's ~3K-row picture prefix is ~60 MB
    /// of KV a slot, so the per-row walk read it twice per verify round.
    pub(super) fn layer_walk(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
    ) -> Result<(), GpuModelError> {
        self.layer_walk_k(r, cuts, 1)
    }

    pub(super) fn layer_walk_k(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
        k1: usize,
    ) -> Result<(), GpuModelError> {
        if self.inv {
            let (dec, runs) = match cuts {
                Some(c) => (c.dec, c.runs.clone()),
                None => (r, Vec::new()),
            };
            return self.layer_walk_inv(r, dec, &runs);
        }
        let exec = self.exec.clone();
        let cfg = &self.cfg;
        let (dm, nh, n_kv, hd) = (cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim());
        let kv_dim = n_kv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_dtype = self.kv_dtype;
        let pf = cuts.is_some();
        // the hd 64 WMMA prefill tile wants f16 caches (the deepseek-ocr gate)
        let wmma_pf = pf
            && matches!(kv_dtype, KvDtype::Fp16)
            && exec.has_attn_prefill_f16_paged()
            && paddock_models::dev_var_os!("PADDOCK_OCR_NO_WMMA").is_none();
        // decode-class rows only: a prefill chunk / mixed tick keeps the tiles
        let w16 = self.w16 && !pf;
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;
        for (li, layer) in self.layers.iter().enumerate() {
            let sc = &mut bs.sc;
            exec.layernorm_f16(
                &sc.x,
                &layer.ln1.w,
                &layer.ln1.b,
                &mut sc.xn16,
                r,
                dm,
                DECODER_LN_EPS,
            )?;
            lin(&exec, w16, &layer.wq, &sc.xn16, &mut sc.q, r)?;
            exec.bias_add(&mut sc.q, &layer.bq, r, dm)?;
            lin(&exec, w16, &layer.wk, &sc.xn16, &mut sc.k, r)?;
            exec.bias_add(&mut sc.k, &layer.bk, r, kv_dim)?;
            lin(&exec, w16, &layer.wv, &sc.xn16, &mut sc.v, r)?;
            exec.bias_add(&mut sc.v, &layer.bv, r, kv_dim)?;
            let kvs = &mut bs.kv[li];
            exec.kv_append_batch_paged(
                &sc.k,
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
            match cuts {
                Some(c) => {
                    if c.dec > 0 {
                        exec.attn_decode_batch_rows_paged(
                            &sc.q,
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
                            0,
                            c.dec,
                            scale,
                            kv_dtype,
                        )?;
                    }
                    for &(off, len) in &c.runs {
                        if wmma_pf {
                            exec.attn_prefill_f16_paged_at(
                                &sc.q,
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
                        } else {
                            exec.attn_decode_batch_rows_paged(
                                &sc.q,
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
                }
                None if k1 > 1 && r.is_multiple_of(k1) && exec.has_attn_spec_batch_paged() => {
                    // splits fill the die: (kv heads x chunks x splits) CTAs
                    let chunks = r / k1;
                    let ns =
                        ((4 * exec.sm_count()) / (n_kv * chunks).max(1)).clamp(1, MAX_ATTN_SPLITS);
                    let ok = exec
                        .attn_spec_batch_paged(
                            &sc.q,
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
                            k1,
                            scale,
                            kv_dtype,
                        )
                        .is_ok();
                    if ok {
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
                        // the kernel declined this geometry: one walk a row
                        let ns = attn_splits_for(nh, r, exec.sm_count());
                        exec.attn_partial_batch_paged(
                            &sc.q,
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
                    }
                }
                None => {
                    let ns = attn_splits_for(nh, r, exec.sm_count());
                    if ns > 1 && exec.has_attn_partial_batch_paged() {
                        exec.attn_partial_batch_paged(
                            &sc.q,
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
                            &sc.q,
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
            }
            exec.convert_f32_f16(&sc.attn, &mut sc.attn16, r * dm)?;
            if w16 {
                exec.dense_w16_f16(&layer.wo, &sc.attn16, &mut sc.scr, dm, r)?;
                exec.add_bias_res(&mut sc.x, &sc.scr, &layer.bo, r, dm)?;
            } else {
                exec.matvec_batch_f16_bias_res(
                    &layer.wo,
                    &sc.attn16,
                    &mut sc.x,
                    &layer.bo,
                    &mut sc.scr,
                    r,
                )?;
            }
            exec.layernorm_f16(
                &sc.x,
                &layer.ln2.w,
                &layer.ln2.b,
                &mut sc.xn16,
                r,
                dm,
                DECODER_LN_EPS,
            )?;
            if w16 {
                let ffn = layer.fc1.dims[1];
                exec.dense_w16_f16(&layer.fc1, &sc.xn16, &mut sc.scr, ffn, r)?;
                exec.gelu_erf_bias_f16(&sc.scr, &layer.fc1_b, &mut sc.ff16, r, ffn)?;
                exec.dense_w16_f16(&layer.fc2, &sc.ff16, &mut sc.scr, dm, r)?;
                exec.add_bias_res(&mut sc.x, &sc.scr, &layer.fc2_b, r, dm)?;
            } else {
                exec.matvec_batch_f16_h_gelu(&layer.fc1, &sc.xn16, &mut sc.ff16, &layer.fc1_b, r)?;
                exec.matvec_batch_f16_bias_res(
                    &layer.fc2,
                    &sc.ff16,
                    &mut sc.x,
                    &layer.fc2_b,
                    &mut sc.scr,
                    r,
                )?;
            }
        }
        Ok(())
    }

    /// Final norm + the tied head over residual rows `0..rows` into
    /// `head_logits`, then the penalty in place and the greedy picks in
    /// `d_out`. `rslot`: the slot of each head row (None = the decode band's
    /// own `d_slots`). The normed rows stay in `hid` (the MTP head's input).
    pub(super) fn head_rows(
        &mut self,
        rows: usize,
        rslot: Option<&[u32]>,
        n_extra: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let dm = self.cfg.d_model;
        let vocab = self.cfg.vocab;
        let pen = self.cfg.repetition_penalty;
        let bs = self.batch.as_mut().expect("batch enabled");
        let words = bs.words;
        let sc = &mut bs.sc;
        exec.layernorm(
            &sc.x,
            &self.ln_f.w,
            &self.ln_f.b,
            &mut sc.hid,
            rows,
            dm,
            DECODER_LN_EPS,
        )?;
        exec.convert_f32_f16(&sc.hid, &mut sc.xn16, rows * dm)?;
        lin(
            &exec,
            self.w16,
            &self.head,
            &sc.xn16,
            &mut sc.head_logits,
            rows,
        )?;
        let map = match rslot {
            Some(s) => {
                exec.upload_u32(s, &mut sc.d_rslot)?;
                &sc.d_rslot
            }
            None => &sc.d_slots,
        };
        exec.np_pen_argmax(
            &mut sc.head_logits,
            &sc.seen,
            map,
            &sc.d_extra,
            n_extra,
            pen,
            vocab,
            words,
            rows,
            true,
            &mut sc.d_out,
        )?;
        Ok(())
    }

    /// One prefill finisher: bounce residual row `row` to the front, head it
    /// (penalized against `slot`'s seen set) and read its logits back.
    pub(super) fn head_row(
        &mut self,
        row: usize,
        slot: usize,
        pos: u32,
    ) -> Result<Vec<f32>, GpuModelError> {
        let dm = self.cfg.d_model;
        let vocab = self.cfg.vocab;
        if row > 0 {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            // through attn as a bounce row (x cannot be both ends of one copy)
            exec.copy_region(&sc.x, row * dm, &mut sc.attn, 0, dm)?;
            exec.copy_region(&sc.attn, 0, &mut sc.x, 0, dm)?;
        }
        self.head_rows(1, Some(&[slot as u32]), 0)?;
        if self.mtp.is_some() {
            self.keep_h(0, slot, pos)?;
        }
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    pub fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.cfg.vocab;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    pub(super) fn read_picks(&mut self, rows: usize) -> Result<Vec<u32>, GpuModelError> {
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .d_out
            .try_slice(0..rows)
            .ok_or_else(|| GpuError::Driver("picks slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    // ── decode ──────────────────────────────────────────────────────────────

    /// One batched decode step; row i drives `slots[i]` at `positions[i]`.
    /// Leaves penalized [r, vocab] logits in head_logits and picks in d_out.
    pub fn batch_step_slots(
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
        if self.inv {
            // one row a slot: r one-row groups, the same for every tick at r
            self.upload_band_groups(slots, positions)?;
        }
        self.step_replay(r)?;
        self.keep_h_rows(slots, positions)
    }

    /// After a decode pass whose head rows 0..n are `slots`' rows: keep each
    /// row's hidden as its slot's MTP input. Outside any captured graph (the
    /// slot offsets change tick to tick).
    pub(super) fn keep_h_rows(&mut self, slots: &[u32], pos: &[u32]) -> Result<(), GpuModelError> {
        if self.mtp.is_none() {
            return Ok(());
        }
        for (i, (&s, &p)) in slots.iter().zip(pos).enumerate() {
            self.keep_h(i, s as usize, p)?;
        }
        Ok(())
    }

    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.seen_band(r)?;
        self.embed_rows(r)?;
        self.embed_norm(r)?;
        self.layer_walk(r, None)?;
        self.head_rows(r, None, 0)
    }

    fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        if paddock_models::dev_var_os!("PADDOCK_NP_NO_GRAPH").is_some() {
            return self.step_body(r);
        }
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .graphs
            .contains_key(&r)
        {
            // warm run first: see spec.rs `replay_or_capture`
            self.step_body(r)?;
            let exec = self.exec.clone();
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("decode pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| GpuError::Driver(format!("decode begin_capture: {e}")))?;
            let rec = self.step_body(r);
            let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("decode end_capture: {e}")));
            rec?;
            let g = graph?
                .ok_or_else(|| GpuError::Driver("decode capture produced no graph".into()))?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .graphs
                .insert(r, super::SendGraph(g));
        }
        self.batch.as_ref().expect("batch enabled").graphs[&r]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("decode graph launch: {e}")))?;
        Ok(())
    }

    // ── device sampling ─────────────────────────────────────────────────────

    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> Vec<u32> {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                _ => {}
            }
        }
        par
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head rows 0..r. Greedy rows take the penalty kernel's picks;
    /// any other device plan runs `sample_rows` over the already-penalized
    /// logits; Host rows read the penalized row back.
    pub(super) fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        use crate::sampler::DevicePlan;
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.cfg.vocab;
        let all_greedy = plans.iter().all(|p| {
            matches!(
                p,
                RowSample::Device(DevicePlan::Greedy) | RowSample::Host | RowSample::Hole
            )
        });
        if !all_greedy {
            let par = Self::pack_samp_par(plans);
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
        }
        let ids = self.read_picks(r)?;
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
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
}
