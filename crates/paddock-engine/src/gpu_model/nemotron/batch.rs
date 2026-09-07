//! Nemotron continuous batching - stage B:
//! the allocation/admission substrate. The batched ticks land in stage C;
//! until they do, `Generator::enable_batch` keeps returning Err and serving
//! stays on the serial lane, so nothing here is reachable from a serve yet.
//!
//! Hybrid-state shape - the piece granite's batch lane doesn't have. Only
//! the 6 attention layers hold PAGED KV: one budget pool of 16-token blocks,
//! one combined block table addressing every attention layer (granite's
//! shape - a block id costs all 6 layers' K+V at once, which on this model
//! is 96 KiB/block at f16, so the pool is cheap next to granite's 4 MiB).
//! The 23 mamba layers hold per-slot FIXED state instead: an f32 SSM state
//! + conv window per slot per layer, allocated as slot ARENAS the stage-A
//!   batched step kernels index through d_slots. Recurrent state is O(1) in
//!   sequence length - it doesn't page, it's a flat cost paid at enable
//!   (~50 MB/slot on this geometry), and it makes admission a two-part act:
//!   back the prompt's blocks AND zero the slot's arenas.
//!
//! Scratch is sized once at enable for `cap = PREFILL_CHUNK + n_slots` rows
//! (granite's law: a fused mixed tick carries the decode band on TOP of a
//! full chunk, so sizing at the chunk alone would make the band steal chunk
//! rows). Decode graphs will bake these addresses in stage C - allocated
//! once, never grown.

use super::ssm_arena::SsmArena;
use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};

use super::forward::PREFILL_CHUNK;
use super::*;
use crate::gpu_model::qwen35::{gemv_any, mmq_pre, mmq_pre_any, prefill_mm_pre_any, prefill_quant};
use paddock_models::nemotron::NemotronBlock;

/// Prefill-mode dispatch cuts for one pass (granite's PfCuts with the slot
/// carried per run): `runs` = contiguous same-slot CHUNK row runs as
/// `(row offset, len, slot)` - an attention launch never mixes two slots'
/// query rows, and nemotron additionally needs the host slot id per run
/// because the recurrent conv/scan advance that slot's arena sequentially.
/// `dec` = leading decode-band rows of a fused mixed tick (q_len 1, one per
/// slot); they take the batched STEP kernels + the decode attention walk.
/// `breaks` = ascending (pass-row end, stage index) checkpoint breaks
/// (stage D): the mamba run walk pauses its advance at each break row,
/// copies that layer's slot state into the staging blob, and continues -
/// the GEMM passes never split (splitting a pass at a cut would re-stream
/// the whole weight set).
pub(super) struct PfCuts {
    pub(super) runs: Vec<(usize, usize, u32)>,
    pub(super) dec: usize,
    pub(super) breaks: Vec<(usize, usize)>,
}

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;

/// FlashDecoding split ceiling for the batched decode attention (partial
/// plane sizing). Nemotron is 32q/2kv hd128 - the q-head grid is already
/// wide, so granite's fused-walk cap is plenty.
pub(crate) const MAX_ATTN_SPLITS: usize = 16;

// dead_code allows below: the stage-C batched ticks are these fields'
// consumers - stage B only allocates and accounts them. Drop the allows
// when the tick lands.
#[allow(dead_code)]
pub(crate) struct LayerKvPaged {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// Batched-lane scratch, sized once at enable for `cap`-row passes (decode
/// reuses the same planes at rows = live slots « cap). Field-for-field the
/// serial `PrefillScratch` twin plus the decode-tick extras (sampling,
/// pipe rings, attention partials, the r=1 fused-MoE lane).
#[allow(dead_code)]
pub(crate) struct NemoBatchScratch {
    pub d_tok: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    pub d_zxbcdt: CudaSlice<f32>,
    pub d_conv: CudaSlice<f32>,
    pub d_y: CudaSlice<f32>,
    pub d_yn: CudaSlice<f32>,
    /// e4m3 activation image for the W8A8 f8row GEMM, [cap, max(hidden, d_inner)]
    pub d_xq: CudaSlice<i8>,
    pub d_xrs: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_sinks: CudaSlice<f32>,
    pub d_logits_r: CudaSlice<f32>,
    pub d_idx: CudaSlice<u32>,
    pub d_w: CudaSlice<f32>,
    ///  uniq-routing diagnostic (PADDOCK_MOE_UNIQ=path): raw non-pool
    /// accumulator + detached dumper, armed at enable_batch - 0 when off.
    /// Same instrument as gemma4/deepseek_ocr (g4_moe_uniq_arm); the hist
    /// launch sits after the topk so captured decode graphs bake it in.
    pub moe_uniq_dev: u64,
    /// zeroed - the shared expert is plane index 0 for every row
    pub d_sh_idx: CudaSlice<u32>,
    /// all-ones combine weights for the shared expert
    pub d_sh_w: CudaSlice<f32>,
    // sorted-tile MoE MMA lane (the serial prefill's rung-2 class, reused at
    // every batch width > 1). nb_r/nb_s are the moe_align block capacities
    // the buffers were sized for.
    pub nb_r: usize,
    pub nb_s: usize,
    pub d_xq4: CudaSlice<i8>,
    pub d_xs4: CudaSlice<u8>,
    pub d_srow: CudaSlice<u32>,
    pub d_sslot: CudaSlice<u32>,
    pub d_bexp: CudaSlice<u32>,
    pub d_srow_s: CudaSlice<u32>,
    pub d_sslot_s: CudaSlice<u32>,
    pub d_bexp_s: CudaSlice<u32>,
    pub d_fq: CudaSlice<u8>,
    pub d_fs: CudaSlice<u8>,
    pub d_fq_s: CudaSlice<u8>,
    pub d_fs_s: CudaSlice<u8>,
    pub d_part: CudaSlice<f32>,
    /// r=1 decode keeps the serial lane's fused wave-dense MoE pair (the bs
    /// tiles pad 1 row to 32); these are its activation + partial planes
    pub d_act: CudaSlice<f32>,
    pub d_part7: CudaSlice<f32>,
    /// [n_slots, vocab] logits - decode graphs bake this address
    pub head_logits: CudaSlice<f32>,
    /// device sampler params [n_slots, 4] (inv_t, u, mode, pad)
    pub d_par: CudaSlice<u32>,
    /// sampled token ids [n_slots]
    pub d_out: CudaSlice<u32>,
    /// mode-5/6 truncation side plane [n_slots, 4] {k, top_p bits, min_p bits,
    /// pad} - nemotron's election is 1.0/top_p 0.95 with no top_k, so every
    /// un-dialled request is a mode-6 (general truncation) row
    pub d_tpar: CudaSlice<u32>,
    /// decode-pipe sampler-param ring [2, n_slots, 4] (stage E)
    pub d_pipe_par: CudaSlice<u32>,
    /// pipe ring twin of `d_tpar` ([2, n_slots, 4])
    pub d_pipe_tpar: CudaSlice<u32>,
    /// decode-pipe sampled-id ring [2, n_slots]
    pub d_pipe_out: CudaSlice<u32>,
    /// FlashDecoding partial scratch [n_heads, n_slots, MAX_ATTN_SPLITS, hd]
    pub attn_o: CudaSlice<f32>,
    /// per-partial (m, l) [n_heads, n_slots, MAX_ATTN_SPLITS, 2]
    pub attn_ml: CudaSlice<f32>,
    /// GGUF-lane extras (None on the NVFP4 lane) - the serial lane's
    /// PrefillQ8/ScratchQ8 union at batch capacity
    pub q8: Option<BatchQ8>,
}

/// Q8_0-lane batch scratch: int8 activation images + kquant sums/fixups for
/// the mmq GEMM ladder, the sorted-MoE fused planes and quantized twins
/// (r>1), and the token-batched dec1 lane's activation/quantized/shared-row
/// buffers (r==1 stays in the serial decode's numeric class).
pub(crate) struct BatchQ8 {
    pub xq: CudaSlice<i8>,
    pub xs: CudaSlice<f32>,
    pub yq: CudaSlice<u8>,
    pub skfix: CudaSlice<f32>,
    pub xsums: CudaSlice<f32>,
    pub ssums: CudaSlice<f32>,
    pub fu_r: CudaSlice<f32>,
    pub fq_r: CudaSlice<i8>,
    pub fs_r: CudaSlice<f32>,
    pub fu_s: CudaSlice<f32>,
    pub fq_s: CudaSlice<i8>,
    pub fs_s: CudaSlice<f32>,
    pub act_r: CudaSlice<f32>,
    pub act_s: CudaSlice<f32>,
    pub fq_r1: CudaSlice<i8>,
    pub fs_r1: CudaSlice<f32>,
    pub fq_s1: CudaSlice<i8>,
    pub fs_s1: CudaSlice<f32>,
    pub shproj: CudaSlice<f32>,
}

/// The whole batching state: pool + tables + arenas + scratch. One struct so
/// enable/teardown is atomic.
#[allow(dead_code)]
pub(crate) struct NemoBatch {
    pub n_slots: usize,
    /// Row capacity of every scratch plane = PREFILL_CHUNK + one row per slot.
    pub cap: usize,
    /// logical blocks per slot (max_ctx/16) - the block table's slot stride
    pub bps: usize,
    /// the attention-layer budget pool + per-slot tables (combined table:
    /// one block id addresses all 6 attention layers)
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    /// per-layer paged K/V stores, Some on the 6 attention layers
    pub kv: Vec<Option<LayerKvPaged>>,
    /// per-layer SSM slot arenas [n_slots, heads, hd, d_state], Some on the
    /// 23 mamba layers. Class is elected (f32 default = the checkpoint's own
    /// mamba_ssm_cache_dtype); arithmetic is f32 either way.
    pub ssm: Vec<Option<SsmArena>>,
    /// per-layer conv-window slot arenas [n_slots, k-1, conv_dim] f32
    pub conv_win: Vec<Option<CudaSlice<f32>>>,
    pub sc: NemoBatchScratch,
    /// device bytes the sequence state holds: paged KV stores + mamba arenas
    pub kv_bytes: u64,
    /// captured decode ticks keyed by row count r (stage C)
    pub graphs: HashMap<usize, SendGraph>,
    /// Radix prefix cache over `pool` (stage D, prefix.rs). A hit adopts
    /// attention blocks by refcount AND restores a mamba state checkpoint.
    pub prefix: Option<crate::paged_radix::PagedRadix>,
    /// KV tier over the attention-layer pool planes; mamba state checkpoint
    /// blobs ride as aux components (qwen35's hybrid recipe).
    pub tier: Option<crate::kv_tier::PoolTier<crate::kv_tier::RamTransport>>,
    /// mamba state checkpoint pool [n_ckpt, state_ckpt_f32] f32 - the blobs
    /// `PagedRadix::attach_state` indices point into
    pub d_state_pool: Option<CudaSlice<f32>>,
    /// f32 elements per checkpoint (all mamba layers' state+window)
    pub state_ckpt_f32: usize,
    /// per-pass staging blobs the layer walk fills at break rows
    pub d_ckpt_stage: Vec<CudaSlice<f32>>,
    /// spec verify planes  - lazily allocated at first spec use
    pub verify: Option<super::spec::VerifyPlanes>,
}

/// A prompt queued for stall-free chunked prefill. `keys` mirrors `tokens`
/// today; it exists so the stage-D radix insert keys the same way the match
/// will (granite's contract).
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    pub tokens: Vec<u32>,
    /// next row to compute (starts at the prefix-resume point)
    pub cursor: usize,
    pub keys: Vec<u32>,
}

/// Batched depth-2 decode-pipe state (stage E - granite's PipeStateG shape):
/// tick N+1's inputs advance on device from tick N's sampled ids, so the
/// host's per-token turnaround overlaps the GPU instead of gapping it.
pub(crate) struct PipeB {
    pub b: usize,
    pub tick: usize,
    pub ev: [Option<cudarc::driver::CudaEvent>; 2],
    /// row start positions (advanced by tick on device; mirrored here for
    /// the per-tick ensure_rows)
    pub pos0: Vec<u32>,
    /// explicit row->slot mapping for a pipe over an arbitrary slot set
    pub slots: Option<Vec<u32>>,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// Blocks `moe_align` can actually fill at this row count - the launch extent
/// for the sorted-tile MoE pair and, crucially, for the intermediate
/// `quantize_q8` between them.
///
/// `bs.nb_r` / `bs.nb_s` are ARENA capacities, sized once for the widest row
/// stream a tick can carry (`cap = PREFILL_CHUNK + max_batch` - 520 rows on a
/// max-batch-8 server, so nb_r = 260). Handing that to the kernels launched
/// 260 blocks for a 4-row decode that fills about 24 of them. The GEMM tiles
/// early-out on PD_MOE_PAD so the pad blocks only cost a CTA slot, but the
/// epilogue quantize is sized in ELEMENTS (`blocks * 32 * moe_ff`) and has no
/// pad to early-out on: it walked all 260 blocks' 32-row tiles. At c4 that
/// one launch family measured 16.3% of the whole decode tick, third behind
/// the two MoE GEMMs themselves.
///
/// `align` emits `sum_e ceil(count_e / 32)` blocks and every `count_e <= rows`,
/// so `distinct * ceil(rows/32)` bounds it - and is exact for rows <= 32, where
/// each touched expert is one block. Deliberately an UPPER bound: a short one
/// would silently drop real expert blocks, which is wrong output, not slow
/// output. Clamped to the arena regardless.
/// `PADDOCK_NO_MOE_NBLIVE=1` pins the old capacity-sized launches for the A/B.
/// The BM=8 analog of [`moe_live_blocks`] for the skinny decode pair: an
/// expert with p picks takes ceil(p/8) blocks, and
/// sum(ceil(p_e/8)) <= min(pairs, experts) + pairs/8 for any distribution.
fn moe_live_blocks_bm8(rows: usize, picks: usize, experts: usize, cap: usize) -> usize {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_MOE_NBLIVE").is_some()) {
        return cap;
    }
    let pairs = rows.saturating_mul(picks);
    (experts.min(pairs) + pairs / 8).min(cap)
}

fn moe_live_blocks(rows: usize, picks: usize, experts: usize, cap: usize) -> usize {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_MOE_NBLIVE").is_some()) {
        return cap;
    }
    (experts.min(rows.saturating_mul(picks)) * rows.div_ceil(32)).min(cap)
}

/// Dense-projection dispatch for the GGUF lane above r = 1.
///
/// `prefill_mm_pre_any` is the PREFILL ladder, and below 64 rows it lands on
/// `q8_0_gemm_mma` - the shared-staging MT tile built for prefill row counts,
/// which is latency-bound across the whole serving band. qwen35's decode
/// ladder (`mmq_pre_any`: the multi-column dp4a GEMV to r = 4, the K-split
/// int8 MMA to 64) is the right family here, and the crossover is measured on
/// this checkpoint's own planes, not assumed - examples/nemo_decgemm_bench,
/// A6000 sm_86, min-of-5, us at r = 4:
///
///   plane                  mma (was)     nc   mma_ks
///   ssm_in  2688->10304         69.6   57.3     47.7
///   ssm_out 4096->2688          71.3   23.5     24.4
///   attn_q  2688->4096          49.5   25.4     25.4
///   attn_k  2688->256           36.3    7.8     13.0
///   attn_o  4096->2688          71.2   23.8     24.3
///
/// Over one r = 4 tick's 70 dense projections that table is 4.40 -> 2.11 ms.
/// The MT tile's problem is not the arithmetic: on ssm_out it holds 164 GB/s
/// where the same weight streams at 490 through the K-split.
///
/// PREFILL rows keep the prefill ladder - granite's law, every prefill row
/// takes the same rungs at any r so a warm-resume tail reproduces the cold
/// chunk's bytes. Past 64 rows the staging layout itself differs
/// (`prefill_quant` flips to the flat mmq plane above 64), so that band stays
/// where it was. `part` is the MoE partials plane, dead outside the MoE arm
/// and sized well past the K-split's 64-row envelope; when a weight does
/// exceed it, `mmq_pre` drops back to the MT tile on its own.
/// `PADDOCK_NO_NEMO_DECMM=1` pins the old route for the A/B.
/// Widest decode tick that takes the dec2 expert pair instead of the sorted
/// tile. dec2 streams a routed expert's planes once per (row, slot) with no
/// dedup, so it has to lose eventually - but not inside any width this engine
/// decodes at. MEASURED on sm_86 at nemotron's shape
/// (examples/nemo_moe_kbench.rs;
/// the sorted routed pair sits FLAT at 137-144 GB/s of deduped expert bytes
/// (18-19% of this card's 768 GB/s) all the way from r=1 to r=64, while dec2
/// runs 594-666 (87% of peak) through r=8 and decays only as the picks start
/// colliding. dec2 is ahead at every width measured - 4.6x at r=4, 2.5x at
/// r=32, still 1.6x at r=64 - so the band is capped by the measurement, not
/// by a crossover: 64 is the widest r the lab priced.
///
/// Prefill is not in the band at any width: chunks are hundreds of rows, and
/// granite's law (every prefill row takes the same rungs at any r, so a warm
/// resume reproduces the cold chunk's bytes) forbids splitting a chunk's
/// route by width.
const MOE_DEC2_MAX_ROWS: usize = 64;

/// Whether this pack carries the decode-band MoE route. Not part of the
/// family's capability gate: a pack without it serves fine on the sorted
/// tile, just slower, and folding it into the gate is the over-broad-bundle
/// shape is auditing for.
fn moe_dec2_ok(exec: &GpuExecutor) -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_NEMO_MOEDEC2").is_none()
            && exec.has_q8_0_moe_relu2_dec2()
            && exec.has_quantize_q8_relu2()
    })
}

#[allow(clippy::too_many_arguments)]
fn dense_mm_pre(
    exec: &GpuExecutor,
    w: &QuantW,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
    pf: bool,
) -> Result<(), GpuModelError> {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NEMO_DECMM").is_some());
    if !off && !pf && r <= 64 {
        return mmq_pre_any(exec, w, xq, xs, ssums, part, y, r);
    }
    prefill_mm_pre_any(exec, w, xq, xs, yq, xsums, ssums, skfix, y, r)
}

impl GpuNemotron {
    /// Allocate the paged-KV + arena + scratch state for up to `max_batch`
    /// slots - granite's budget/floor/Err contract. Returns the capacity
    /// actually enabled; Ok(1) = stay on the serial loop (pack lacks the
    /// batched kernel set); Err(WontFit) = VRAM can't seat the floor, and the
    /// caller's serial fallback is safe because the serial state re-builds
    /// lazily (`ensure_decode`).
    pub(crate) fn enable_batch_impl(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        self.pipe_abort();
        self.pipe_b_abort();
        // The batch tick needs: the paged attention set, the stage-A batched
        // mamba steps, the bulk-prefill consumers (the chunk rows ride
        // them), and the weight class's own GEMM/MoE lanes - NVFP4 (bs tiles
        // for r>1 + the fused mt pair for r=1 + the batched head GEMV) or
        // Q8_0 (the relu2 pair; head/attn/mamba ride the always-present mmq
        // ladder). A real Err, never Ok(1): service.rs's single-user branch
        // routes any Ok through run_batched, so an Ok(1) with
        // self.batch=None would hand the batched loop a batch-less
        // generator. Err lands on the honest serial fallback in both service
        // branches.
        let class_ok = if self.is_gguf() {
            self.exec.has_q8_0_moe_relu2()
        } else {
            self.exec.has_nvf4_gemv_batch()
                && self.exec.has_nvf4_moe_bs()
                && self.exec.has_nvf4_moe_mt()
        };
        // The prefill set is asked per LANE: the fp8 pair (f8row_gemm +
        // quantize_e4m3_row) is only ever called from the LinW::F8 arms, so
        // demanding it from a GGUF checkpoint stranded every Q8_0 nemotron on
        // pre-sm_89 silicon for no reason. See has_nemotron_prefill_gguf.
        let gguf = self.is_gguf();
        let prefill_ok = if gguf {
            self.exec.has_nemotron_prefill_gguf()
        } else {
            self.exec.has_nemotron_prefill_f8()
        };
        if !self.exec.has_paged_kv() || !self.exec.has_mamba2_batch() || !prefill_ok || !class_ok {
            // Name what is actually absent. The old text said "lower --max-ctx
            // or PADDOCK_MAX_BATCH so the batched KV fits", which is a lie when
            // the failure is a missing kernel - no context or width value can
            // help, and the width-by-VRAM backstop then retried 8/4/2 against a
            // condition width does not affect.
            let mut missing = self.exec.nemotron_prefill_missing(!gguf);
            if !self.exec.has_paged_kv() {
                missing.push("paged_kv");
            }
            if !self.exec.has_mamba2_batch() {
                missing.push("mamba2_batch");
            }
            if !class_ok {
                missing.push(if gguf {
                    "q8_0_moe_relu2"
                } else {
                    "nvf4_moe/gemv"
                });
            }
            return Err(GpuModelError::Unsupported(format!(
                "nemotron enable_batch: this GPU's kernel pack is missing {} - staying serial.                  Not a memory or context limit; no --max-ctx or --max-batch value changes it.",
                missing.join(", ")
            )));
        }
        // Stage E: fp8-e4m3 KV serves through the batch lane - the pool
        // allocates at the dtype's byte width, appends/decode take kv_dtype,
        // and the prefill rides the v4 tile's raw-e4m3 hd128 G=16 arm (the
        // pack's granite/laguna/muse/paddleocr arm covers G in {4,6,8,9,16}).
        // the serial dense KV (6 × max_ctx rings) and chunk scratch make way;
        // the serial lane re-builds lazily if the caller falls back
        self.decode = None;
        self.scratch = None;
        self.prefill = None;
        self.batch = None;
        self.exec.trim_mem_pool();

        let hp = self.hp.clone();
        let (embd, nh, n_kv, hd) = (hp.hidden, hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv * hd;
        let q_dim = nh * hd;
        let d_inner = hp.d_inner();
        let conv_dim = hp.conv_dim();
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let n_attn = hp
            .blocks
            .iter()
            .filter(|b| matches!(b, NemotronBlock::Attention))
            .count();
        let n_mamba = hp
            .blocks
            .iter()
            .filter(|b| matches!(b, NemotronBlock::Mamba))
            .count();

        // One block id addresses every attention layer (combined table), so a
        // block costs all n_attn layers' K+V at once.
        let block_bytes = n_attn * 16 * kv_dim * 2 * kvb;
        // per-slot recurrent state (flat, not paged)
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * conv_dim;
        let ssm_dt = self.ssm_dtype;
        let arena_bytes = max_batch * n_mamba * (state_elems * ssm_dt.bytes() + win_elems * 4);

        let cap = PREFILL_CHUNK + max_batch;
        let qmax = embd.max(d_inner);
        // shared fold-: the r>1 path serves the shared expert
        // as ns_sh pseudo-experts inside the routed launch, so the align
        // capacity and the idx/w/part planes carry n_active + ns_sh picks
        let ns_sh = if hp.shared_ff.is_multiple_of(hp.moe_ff) && hp.moe_ff.is_multiple_of(32) {
            hp.shared_ff / hp.moe_ff
        } else {
            0
        };
        let kw_r = hp.n_active + ns_sh;
        let nb_r = cap * kw_r / 32 + hp.n_expert + ns_sh;
        let nb_s = cap / 32 + 1;
        // estimate the scratch before committing to a pool size, so the pool
        // can never starve it: the f32 row planes dominate, then the bs fq/fs
        // tiles, the vocab head, and the attention partials; 128 MiB covers
        // the u32 metadata + graph churn
        // 4 * embd: d_x, d_xn, d_proj shproj, which grew from
        // one row to a whole tick when the shared expert moved onto the dense
        // ladder - an estimate that misses a cap-scaled plane hands the pool
        // memory the scratch still needs
        let scratch_est = cap
            * (4 * embd
                + hp.in_proj_rows()
                + conv_dim
                + 2 * d_inner
                + 2 * q_dim
                + 2 * kv_dim
                + hp.n_expert
                + kw_r.max(hp.n_active + 1) * embd)
            * 4
            + cap * qmax
            + cap * (embd / 2 + embd / 16)
            + nb_r * 32 * (hp.moe_ff / 2 + hp.moe_ff / 16)
            + nb_s * 32 * (hp.shared_ff / 2 + hp.shared_ff / 16)
            + max_batch * hp.vocab * 4
            + nh * max_batch * MAX_ATTN_SPLITS * (hd + 2) * 4
            + (128 << 20);
        let px_on = !super::prefix::prefix_disabled();
        let retain = if px_on {
            super::prefix::retention_blocks()
        } else {
            0
        };
        // One arbiter sizes the KV store: crate::kv_plan. Nemotron's
        // own arithmetic was already budget-correct - this is the same solve,
        // moved somewhere a new family cannot forget to do it, and it reports the
        // pool's TOKEN CAPACITY rather than leaving max_ctx to imply it.
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        // What the process holds as the plan is made: the audit at the end
        // adds the plan's charges to it.
        let ledger_at_plan = self.exec.process_mem_used().unwrap_or(0);
        // Mamba state checkpoints for the prefix cache, sized by DEMAND and
        // charged as a reserve (the qwen35 rule, 2026-09-06): two per
        // requested slot, clamped 16..256, then capped by what the grant has
        // left once every other term - including a full-context pool for
        // every slot - is charged. It used to be computed AFTER the plan as a
        // fifth of whatever was still free, clamped to 256, with no reserve:
        // 256 f32 snapshots (~6 GiB) for a one-slot server on a 96 GB card.
        let state_ckpt_f32 = n_mamba * (state_elems + win_elems);
        let per_ckpt = (state_ckpt_f32 * 4) as u64;
        let tier_staging: u64 = if crate::kv_tier::pool_tier::tier_ram_bytes().is_some() {
            crate::kv_tier::ram_transport::device_staging_bytes()
        } else {
            0
        };
        // the Mamba arenas: one recurrent state + conv window per slot per
        // mamba layer (`arena_bytes` was max_batch x this)
        let per_slot_bytes = (n_mamba * (state_elems + win_elems) * 4) as u64;
        let charged_without_pool = VRAM_HEADROOM as u64
            + scratch_est as u64
            + tier_staging
            + max_batch as u64 * per_slot_bytes
            + (max_batch * bps * block_bytes) as u64;
        let n_ckpt: u32 = if !px_on || per_ckpt == 0 {
            0
        } else if let Some(n) = paddock_models::dev_var!("PADDOCK_KV_STATE_CKPTS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|&n| n > 0)
        {
            n
        } else {
            let want = (max_batch as u64 * 2).clamp(16, 256);
            let leftover = grant.saturating_sub(charged_without_pool);
            want.min(leftover / per_ckpt).max(16) as u32
        };
        let demand = kv_plan::Demand {
            family: "nemotron",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            per_slot_bytes,
            // Cap the pool at what (slots × max_ctx) can actually ADDRESS plus
            // explicit radix retention (blocks the tree may hold after their
            // sequence ends - cheap here at 96 KiB/block-set, ~48 MB default).
            retention_blocks: retain,
            // every slot must at least hold a full chunk's worth of prompt, or
            // admission deadlocks on its own first chunk
            floor_blocks_per_slot: PREFILL_CHUNK.div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                kv_plan::Reserve::new("prefill scratch", scratch_est as u64),
                kv_plan::Reserve::new("prefix state pool", n_ckpt as u64 * per_ckpt),
                kv_plan::Reserve::new("kv-tier staging", tier_staging),
            ],
            ..Default::default()
        };
        // A real Err, not a lying Ok(1): the caller treats Ok(c) as proof
        // self.batch is genuinely populated at capacity c. The serial state
        // re-builds lazily, so the caller's fallback on Err is safe.
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        let slots = plan.slots;
        let plan_reserved: u64 = demand.reserves.iter().map(|r| r.bytes).sum::<u64>()
            + plan.pool_bytes
            + plan.slot_bytes;

        let e = &self.exec;
        let mut kv: Vec<Option<LayerKvPaged>> = Vec::with_capacity(hp.n_layer);
        let mut ssm: Vec<Option<SsmArena>> = Vec::with_capacity(hp.n_layer);
        let mut conv_win: Vec<Option<CudaSlice<f32>>> = Vec::with_capacity(hp.n_layer);
        let mut kv_bytes = arena_bytes as u64;
        for li in 0..hp.n_layer {
            match hp.blocks[li] {
                NemotronBlock::Attention => {
                    let bytes = pool_blocks * 16 * kv_dim * kvb;
                    kv_bytes += 2 * bytes as u64;
                    kv.push(Some(LayerKvPaged {
                        k: e.alloc_u8(bytes)?,
                        v: e.alloc_u8(bytes)?,
                    }));
                    ssm.push(None);
                    conv_win.push(None);
                }
                NemotronBlock::Mamba => {
                    kv.push(None);
                    // alloc() zeroes; admission re-zeroes per slot - a fresh
                    // sequence must start from S = 0 / an all-zero window
                    ssm.push(Some(SsmArena::alloc(e, slots * state_elems, ssm_dt)?));
                    conv_win.push(Some(e.alloc(slots * win_elems)?));
                }
                NemotronBlock::Moe => {
                    kv.push(None);
                    ssm.push(None);
                    conv_win.push(None);
                }
            }
        }

        let d_sh_w = e.to_device(&vec![1.0f32; cap])?;
        let sc = NemoBatchScratch {
            d_tok: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_slots: e.alloc_u32(cap)?,
            d_x: e.alloc(cap * embd)?,
            d_xn: e.alloc(cap * embd)?,
            d_proj: e.alloc(cap * embd)?,
            d_zxbcdt: e.alloc(cap * hp.in_proj_rows())?,
            d_conv: e.alloc(cap * conv_dim)?,
            d_y: e.alloc(cap * d_inner)?,
            d_yn: e.alloc(cap * d_inner)?,
            d_xq: e.alloc_i8(cap * qmax)?,
            d_xrs: e.alloc(cap)?,
            d_q: e.alloc(cap * q_dim)?,
            d_k: e.alloc(cap * kv_dim)?,
            d_v: e.alloc(cap * kv_dim)?,
            d_attn: e.alloc(cap * q_dim)?,
            d_sinks: e.alloc_no_sinks(nh)?,
            d_logits_r: e.alloc(cap * hp.n_expert)?,
            d_idx: e.alloc_u32(cap * kw_r.max(hp.n_active))?,
            d_w: e.alloc(cap * kw_r.max(hp.n_active))?,
            //  diagnostic: armed only under PADDOCK_MOE_UNIQ (raw
            // non-pool buffer + dumper thread - the gemma4 instrument)
            moe_uniq_dev: if hp.n_expert != 0
                && paddock_models::dev_var_os!("PADDOCK_MOE_UNIQ").is_some()
            {
                crate::gpu_model::gemma4::g4_moe_uniq_arm(e)
                    .map_err(|err| GpuModelError::Unsupported(format!("moe_uniq arm: {err}")))?
            } else {
                0
            },
            d_sh_idx: e.alloc_u32(cap)?, // zeroed -> plane index 0
            d_sh_w,
            nb_r,
            nb_s,
            d_xq4: e.alloc_i8(cap * embd / 2)?,
            d_xs4: e.alloc_u8(cap * embd / 16)?,
            d_srow: e.alloc_u32(nb_r * 32)?,
            d_sslot: e.alloc_u32(nb_r * 32)?,
            d_bexp: e.alloc_u32(nb_r)?,
            d_srow_s: e.alloc_u32(nb_s * 32)?,
            d_sslot_s: e.alloc_u32(nb_s * 32)?,
            d_bexp_s: e.alloc_u32(nb_s)?,
            d_fq: e.alloc_u8(nb_r * 32 * hp.moe_ff / 2)?,
            d_fs: e.alloc_u8(nb_r * 32 * hp.moe_ff / 16)?,
            d_fq_s: e.alloc_u8(nb_s * 32 * hp.shared_ff / 2)?,
            d_fs_s: e.alloc_u8(nb_s * 32 * hp.shared_ff / 16)?,
            d_part: e.alloc(cap * kw_r.max(hp.n_active + 1) * embd)?,
            d_act: e.alloc(hp.n_active * hp.moe_ff + hp.shared_ff)?,
            d_part7: e.alloc((hp.n_active + 1) * embd)?,
            head_logits: e.alloc(slots * hp.vocab)?,
            d_par: e.alloc_u32(slots * 4)?,
            d_out: e.alloc_u32(slots)?,
            d_tpar: e.alloc_u32(slots * 4)?,
            d_pipe_par: e.alloc_u32(2 * slots * 4)?,
            d_pipe_tpar: e.alloc_u32(2 * slots * 4)?,
            d_pipe_out: e.alloc_u32(2 * slots)?,
            attn_o: e.alloc(nh * slots * MAX_ATTN_SPLITS * hd)?,
            attn_ml: e.alloc(nh * slots * MAX_ATTN_SPLITS * 2)?,
            q8: if self.is_gguf() {
                Some(BatchQ8 {
                    xq: e.alloc_i8(cap * qmax)?,
                    xs: e.alloc(cap * qmax / 32)?,
                    yq: e.alloc_u8(qmax.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
                    skfix: e.alloc(256 * 128 * 128 + 256)?,
                    xsums: e.alloc(qmax.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
                    ssums: e.alloc(cap * qmax / 16)?,
                    fu_r: e.alloc(nb_r * 32 * hp.moe_ff)?,
                    fq_r: e.alloc_i8(nb_r * 32 * hp.moe_ff)?,
                    fs_r: e.alloc(nb_r * 32 * hp.moe_ff / 32)?,
                    fu_s: e.alloc(nb_s * 32 * hp.shared_ff)?,
                    fq_s: e.alloc_i8(nb_s * 32 * hp.shared_ff)?,
                    fs_s: e.alloc(nb_s * 32 * hp.shared_ff / 32)?,
                    act_r: e.alloc(hp.n_active * hp.moe_ff)?,
                    act_s: e.alloc(hp.shared_ff)?,
                    fq_r1: e.alloc_i8(hp.n_active * hp.moe_ff)?,
                    fs_r1: e.alloc(hp.n_active * hp.moe_ff / 32)?,
                    fq_s1: e.alloc_i8(hp.shared_ff)?,
                    fs_s1: e.alloc(hp.shared_ff / 32)?,
                    shproj: e.alloc(cap * embd)?,
                })
            } else {
                None
            },
        };

        // Stage D: the radix + mamba state-checkpoint pool. Each checkpoint
        // is a full 23-layer state snapshot (~48 MB on this geometry), so
        // The checkpoint pool at the count the plan charged (see the reserve
        // above) - the reservation and the allocation cannot drift apart.
        let (prefix, d_state_pool, d_ckpt_stage, n_ckpt) = if px_on && n_ckpt > 0 {
            let mut pr = crate::paged_radix::PagedRadix::new();
            pr.set_state_capacity(n_ckpt);
            let pool_f32 = self.exec.alloc(n_ckpt as usize * state_ckpt_f32)?;
            let stages = (0..super::prefix::CKPT_STAGES)
                .map(|_| self.exec.alloc(state_ckpt_f32))
                .collect::<Result<Vec<_>, _>>()?;
            (Some(pr), Some(pool_f32), stages, n_ckpt)
        } else {
            (None, None, Vec::new(), 0)
        };

        // KV tier (kv-offload): attention-layer pool planes; mamba state
        // checkpoint blobs ride as aux components. Loud decline on any
        // failure - serving continues untiered.
        let tier = match (prefix.as_ref(), crate::kv_tier::pool_tier::tier_ram_bytes()) {
            (Some(_), Some(ram)) => {
                use crate::kv_tier::digest::{IdentityDigest, IdentityFields, PrivacyScope};
                use crate::kv_tier::{CacheNamespace, PlaneDesc, PoolTier, RamTransport};
                use cudarc::driver::DevicePtr;
                let e = &self.exec;
                let stride = (16 * kv_dim * kvb) as u64;
                let mut planes = Vec::new();
                for l in kv.iter().flatten() {
                    for plane in [&l.k, &l.v] {
                        let (pp, _g) = plane.device_ptr(&e.stream);
                        planes.push(PlaneDesc {
                            base: pp,
                            stride,
                            bytes: stride,
                        });
                    }
                }
                let content_id = self.content_id;
                let architecture = format!(
                    "nemotron v1 attn_layers={} kv_dim={kv_dim} kvb={kvb} state_ckpt_f32={state_ckpt_f32}",
                    planes.len() / 2,
                );
                let ns = CacheNamespace {
                    identity: IdentityDigest::compute(&IdentityFields {
                        model_tensors: &content_id.0,
                        adapter: b"",
                        architecture: architecture.as_bytes(),
                        cache_schema: b"pool-planes k/v interleaved + mamba-ckpt aux v1",
                        layout_abi: 1,
                        tokenizer: &content_id.1,
                    }),
                    scope: PrivacyScope::Shared,
                };
                let transport = match crate::kv_tier::pool_tier::nvme_dir_for(&ns) {
                    Some((dir, quota)) => RamTransport::with_t2(e, ram, &dir, quota),
                    None => RamTransport::new(e, ram),
                };
                match transport
                    .map_err(|x| x.to_string())
                    .and_then(|t| PoolTier::new(&ns, planes, ram, t).map_err(|x| x.to_string()))
                {
                    Ok(mut t) => {
                        t.preload_from_t2();
                        Some(t)
                    }
                    Err(err) => {
                        tracing::warn!(err = %err, "nemotron KV tier declined");
                        None
                    }
                }
            }
            _ => None,
        };
        let mut prefix = prefix;
        if let (Some(pr), Some(t)) = (prefix.as_mut(), tier.as_ref()) {
            pr.set_tier_root(t.tier_root());
        }

        let e = &self.exec;
        self.batch = Some(NemoBatch {
            n_slots: slots,
            cap,
            bps,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            ssm,
            conv_win,
            sc,
            kv_bytes,
            graphs: HashMap::new(),
            prefix,
            tier,
            d_state_pool,
            state_ckpt_f32,
            d_ckpt_stage,
            verify: None,
        });
        self.last_reused = vec![0; slots];
        self.dflash_ensure_state()?;
        self.mtp_ensure_state()?;
        tracing::info!(
            "nemotron batch: {slots} slots, {n_attn}-attn-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} tokens), mamba arenas {:.2} GiB, {n_ckpt} state ckpts, \
             {} rows/chunk; left {:.2} GiB of the {:.2} GiB granted",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            arena_bytes as f64 / (1u64 << 30) as f64,
            PREFILL_CHUNK,
            grant.saturating_sub(plan_reserved) as f64 / (1u64 << 30) as f64,
            grant as f64 / (1u64 << 30) as f64,
        );
        // Plan audit (the qwen35 line): the plan's charges on top of what the
        // process held when it was made, against the ledger now.
        let expected = ledger_at_plan + plan_reserved;
        let actual = self.exec.process_mem_used().unwrap_or(0);
        tracing::info!(
            expected_gib = expected as f64 / (1u64 << 30) as f64,
            ledger_gib = actual as f64 / (1u64 << 30) as f64,
            unplanned_mib = actual.saturating_sub(expected) as f64 / (1u64 << 20) as f64,
            "nemotron VRAM plan audit: ledger vs plan after enable_batch"
        );
        Ok(slots)
    }

    /// Back every `(slot, position)` this pass will touch with a physical
    /// pool block, re-uploading the device table once on growth.
    /// PoolExhausted surfaces to the scheduler, which preempts. (Stage D adds
    /// the radix-LRU shed before that surfaces.)
    pub(super) fn ensure_rows(
        &mut self,
        slots: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        let max_ctx = self.max_ctx;
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            // a position past the window would grow the table past its bps
            // stride and corrupt the next slot's rows in bt_host - refuse
            // loudly instead
            if positions[i] as usize >= max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: positions[i] as usize + 1,
                    max: max_ctx,
                });
            }
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            loop {
                match bs.tables[s].ensure(positions[i] as usize, &mut bs.pool) {
                    Ok(()) => break,
                    // Dry pool: shed radix retention (LRU leaves) before
                    // asking the scheduler to preempt a live sequence - the
                    // cache is reclaimable capacity. Tier-aware (qwen35
                    // recipe): closing runs and their mamba checkpoint
                    // blobs demote to T1 before eviction; pins defer the
                    // frees - drain briefly.
                    Err(_) => {
                        let shed = match (bs.tier.as_mut(), bs.prefix.as_mut()) {
                            (Some(tier), Some(pr)) => {
                                let exec = self.exec.clone();
                                let state_bytes = (bs.state_ckpt_f32 * 4) as u64;
                                let state = bs.d_state_pool.as_ref().map(|sp| {
                                    use cudarc::driver::DevicePtr;
                                    let (pp, _g) = sp.device_ptr(&exec.stream);
                                    (pp, state_bytes)
                                });
                                let want = bs.pool.free_blocks() + 1;
                                tier.make_room_blocking(pr, &mut bs.pool, want, state, &mut || {
                                    exec.record_event().ok()
                                })
                            }
                            (None, Some(pr)) => pr.evict_lru(&mut bs.pool).is_some(),
                            _ => false,
                        };
                        if !shed {
                            return Err(GpuModelError::PoolExhausted);
                        }
                    }
                }
            }
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

    /// Admission prologue: bounds-check, drop the slot's previous sequence
    /// (blocks AND recurrent state), back the whole prompt's blocks up front
    /// (a mid-prompt chunk must never find the pool dry with rows written),
    /// and zero the slot's mamba arenas - the recurrent twin of "fresh
    /// sequence: old pool blocks return first". Stale state here is not a
    /// crash, it's silent cross-request contamination.
    pub(super) fn admit_rows(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        self.dflash_clear_slot(slot);
        self.mtp_clear_slot(slot)?;
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
            return Err(GpuModelError::ContextExceeded {
                got: n_rows,
                max: self.max_ctx,
            });
        }
        let exec = self.exec.clone();
        let state_elems = self.hp.mamba_heads * self.hp.mamba_head_dim * self.hp.d_state;
        let win_elems = (self.hp.d_conv - 1) * self.hp.conv_dim();
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
            for s in bs.ssm.iter_mut().flatten() {
                s.zero_region(&exec, slot * state_elems, state_elems)?;
            }
            for w in bs.conv_win.iter_mut().flatten() {
                exec.zero_region(w, slot * win_elems, win_elems)?;
            }
        }
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    /// Free-on-completion: an idle slot's blocks return to the shared pool
    /// immediately. The mamba arenas need no action here - admission zeroes
    /// them, and they hold no pool capacity.
    pub(crate) fn release_inactive_slots_impl(&mut self, occupied: &[bool]) {
        for (s, &occ) in occupied.iter().enumerate() {
            if !occ {
                self.dflash_clear_slot(s);
                // release is not a data path - a clear-copy failure here
                // can't corrupt anything the next admit won't re-clear
                let _ = self.mtp_clear_slot(s);
            }
        }
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < bs.tables.len() && !bs.tables[s].blocks().is_empty() {
                bs.tables[s].clear(&mut bs.pool);
            }
        }
    }

    /// Free blocks for the admission watermark, INCLUDING what the prefix
    /// cache could give back (the gemma4 lesson: counting only free_blocks
    /// lets retention drive admission to ~0 and serialize the server).
    pub(crate) fn pool_free_blocks_impl(&self) -> Option<usize> {
        self.batch
            .as_ref()
            .map(|b| b.pool.free_blocks() + self.prefix_evictable())
    }

    pub(crate) fn kv_mem_bytes_impl(&self) -> Option<u64> {
        self.batch.as_ref().map(|b| b.kv_bytes)
    }

    // ── the batched pass (stage C) ──────────────────────────────────

    /// One weight-amortized pass over a ready-made row stream - the shared
    /// body of every prefill lane and the fused mixed tick. `chunk` is
    /// (slot, pos, token) with items contiguous; the leading `dec` rows are
    /// a fused tick's decode band. Rows may start at any position in their
    /// slot - a mid-prompt chunk resume is the same thing to this pass as a
    /// fresh prompt (granite's stall-free law), with the nemotron addition
    /// that the mamba conv/scan state carries per SLOT in the arenas, so a
    /// resumed chunk continues exactly where the previous chunk's state
    /// advance stopped.
    pub(super) fn rows_pass_body(
        &mut self,
        chunk: &[(u32, u32, u32)],
        dec: usize,
        breaks: Vec<(usize, usize)>,
    ) -> Result<(), GpuModelError> {
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        // contiguous same-slot runs over the PREFILL rows, slot carried for
        // the per-run recurrent advance
        let mut runs: Vec<(usize, usize, u32)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n, s)) if *s == x.0 && *off + *n == i => *n += 1,
                _ => runs.push((i, 1, x.0)),
            }
        }
        self.upload_rows(&toks, &positions, &slots_v)?;
        self.embed_rows(chunk.len())?;
        let note: Vec<(usize, usize, usize)> = {
            // per-slot covered spans for the drafter's coverage bookkeeping
            let mut spans: Vec<(usize, usize, usize)> = Vec::new();
            for x in chunk.iter() {
                let (s, p) = (x.0 as usize, x.1 as usize);
                match spans.last_mut() {
                    Some((ls, _, le)) if *ls == s && *le == p => *le += 1,
                    _ => spans.push((s, p, p + 1)),
                }
            }
            spans
        };
        self.layer_walk(chunk.len(), Some(&PfCuts { runs, dec, breaks }), false)?;
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) {
            self.dflash_append_features(chunk.len())?;
            for &(s, a, b) in &note {
                self.dflash_note_rows(s, a, b);
            }
        }
        if self.mtp.as_ref().is_some_and(|m| m.state.is_some()) {
            // the MTP block advances over the same spans; every span is a
            // plain walk commit, so coverage and the h chain advance to the
            // span end (verify rounds instead advance only accepted rows -
            // spec.rs owns that call site)
            let mut mruns = Vec::with_capacity(note.len());
            let mut off = 0usize;
            for &(s, a, b) in &note {
                mruns.push((s, off, b - a));
                off += b - a;
            }
            self.mtp_append_rows(&mruns)?;
            let mut off = 0usize;
            for &(s, a, b) in &note {
                self.mtp_advance(s, a, b, off + (b - a) - 1)?;
                off += b - a;
            }
        }
        Ok(())
    }

    /// Host->device row streams: tokens, positions, slots.
    pub(super) fn upload_rows(
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
            .d_tok
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_tok".into()))?;
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

    /// Gather the rows' embeddings (nemotron has no embedding scale).
    pub(super) fn embed_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let embd = self.hp.hidden;
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        match &self.tok_embd {
            TokEmbd::F32(tab) => {
                self.exec
                    .embed_gather_batch(tab, &sc.d_tok, &mut sc.d_x, embd, r)?
            }
            TokEmbd::Bf16(tab) => {
                self.exec
                    .embed_gather_bf16(tab, &sc.d_tok, &mut sc.d_x, embd, r, 1.0)?
            }
            TokEmbd::Q8(tab) => {
                self.exec
                    .embed_gather_batch_q8(tab, &sc.d_tok, &mut sc.d_x, embd, r)?
            }
        }
        Ok(())
    }

    /// The whole-stack walk over r rows. `cuts`: Some = prefill mode (append
    /// the whole chunk, attend + advance recurrent state per same-slot run);
    /// None = decode mode (every row is one new token of its slot, the
    /// stage-A batched step kernels advance the arenas by d_slots).
    ///
    /// Compute classes, keyed on MODE (granite's law - every prefill row
    /// takes the same rungs at any r, so a warm-resume tail reproduces the
    /// cold chunk's bytes): prefill rows ride the serial bulk-prefill lane's
    /// W8A8 f8row GEMM / gemm_f32 / sorted-tile bs MoE, all arbiter-gated by
    /// the bulk-prefill parity gate. A PURE r==1 decode tick rides the
    /// serial decode twins (f8r GEMV, bf16 GEMV, fused mt MoE) so the c1
    /// battery stays in the serial lane's numeric class; r>1 decode takes
    /// the batched GEMM class - attention projections in the twins' bf16
    /// class when the pack carries them (the serial GEMV twins' own class),
    /// f32 otherwise; prefill rows always the exact-f32 planes.
    /// `verify` (spec core): the mamba advance runs the spec
    /// verify's non-committing discipline - conv on the slot's SCRATCH
    /// window (the live window stays pre-round for the commit rebuild),
    /// scan on the live state with per-row snapshots, and the xBC rows
    /// snapshot so a partial accept can rebuild the window. Everything else
    /// (KV, attention, MoE, residuals) is byte-identical to the plain walk -
    /// KV needs no rollback (stale cells past the accept are overwritten
    /// before any later read). Requires `bs.verify` populated.
    pub(super) fn layer_walk(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
        verify: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let (embd, eps) = (hp.hidden, hp.eps);
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let (nh, n_kv, hd) = (hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let d_inner = hp.d_inner();
        let conv_dim = hp.conv_dim();
        let in_rows = hp.in_proj_rows();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * conv_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_dtype = self.kv_dtype;
        let pf = cuts.is_some();
        // pure single-row decode: the serial lane's kernel twins
        let dec1 = r == 1 && !pf;
        // KV splits for the batched decode attention (position-independent,
        // so the per-r graph can bake it): nh×r blocks starve the die at
        // small r - budget ~2-3 CTAs/SM like granite.
        //
        // Head-packed arm: at this exact shape (fp8
        // KV, hd128, G=16, no window, r>=2) the pack elects the lagd WIDE
        // partial whose grid is (n_kv, r, ns) - 16x fewer blocks than
        // vec8's (nh, r, ns) - so the budget divides by n_kv and caps at 8
        // (ctx/8 chunks keep the 32-token tiles full at serve contexts).
        // The kill env is the same one the pack's election reads, so an
        // off-arm A/B leg gets vec8's budget too.
        let hp16 = matches!(kv_dtype, KvDtype::Fp8E4m3)
            && hd == 128
            && n_kv > 0
            && nh == n_kv * 16
            && r >= 2
            && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_HP16").is_none();
        let ns = if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
            1
        } else if hp16 {
            (3 * exec.sm_count()).div_ceil(n_kv * r).clamp(1, 8)
        } else {
            (2 * 3 * exec.sm_count())
                .div_ceil(nh * r)
                .clamp(1, MAX_ATTN_SPLITS)
        };
        // f16 always rides the wmma tile; fp8 rides its raw-e4m3 hd128 arm
        // (pack group set {4,6,8,9,16} - nemotron is G=16). PADDOCK_NO_NPF8
        // pins fp8 back onto the scalar paged walk (granite's A/B precedent).
        let pf8_ok = match kv_dtype {
            KvDtype::Fp16 => true,
            KvDtype::Fp8E4m3 => {
                n_kv > 0
                    && nh == n_kv * 16
                    && paddock_models::dev_var_os!("PADDOCK_NO_NPF8").is_none()
            }
        };
        let wmma_pf = pf
            && hd == 128
            && pf8_ok
            && exec.has_attn_prefill_f16_paged()
            && paddock_models::dev_var_os!("PADDOCK_NO_WMMA_PREFILL").is_none();
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;
        // running mamba-layer index: the checkpoint blob offset for a layer
        // is mi * (state + win), matching the pool/restore layout
        let mut mi = 0usize;

        // Glue rung: every MoE layer's prologue is add(x += proj)
        // + rmsnorm + quantize_nvf4, three latency-bound launches, and the
        // checkpoint's pattern makes that 23 of them per decode tick. One
        // fused row-per-CTA kernel does all three, so the previous layer's
        // trailing add is hoisted into it. Only the bs arm consumes the nvf4
        // planes, so dec1 (which takes the mt path) keeps the plain chain.
        // Bit-exact - PADDOCK_NO_GLUE_FUSE restores the three launches for the
        // A/B on one binary.
        let glue_fuse = !dec1
            && paddock_models::dev_var_os!("PADDOCK_NO_GLUE_FUSE").is_none()
            && exec.has_add_rmsnorm_quant_nvf4();
        let mut fused_pro = false;
        for (li, layer) in self.layers.iter().enumerate() {
            let pro_done = std::mem::take(&mut fused_pro);
            let sc = &mut bs.sc;
            // DFlash aux tap: d_x here is the post-block residual of layer
            // li-1 - copy it into the drafter's aux band when li-1 is a
            // target layer (the last target layer taps after the loop)
            if li > 0
                && let Some(df) = self.dflash.as_mut()
                && let Some(st) = df.state.as_mut()
                && let Some(ai) = df.target_layers.iter().position(|&t| t == li - 1)
            {
                exec.copy_region(&sc.d_x, 0, &mut st.aux[ai], 0, r * embd)?;
            }
            if !pro_done {
                exec.rmsnorm_batch(&sc.d_x, &layer.norm.buf, &mut sc.d_xn, embd, eps, r)?;
            }
            match &layer.mixer {
                Mixer::Mamba(w) => {
                    match &w.in_proj {
                        LinW::F8(p) => {
                            if dec1 {
                                exec.f8r_gemv(p, &sc.d_xn, &mut sc.d_zxbcdt, embd, in_rows)?;
                            } else {
                                exec.quantize_e4m3_row(
                                    &sc.d_xn,
                                    &mut sc.d_xq,
                                    &mut sc.d_xrs,
                                    embd,
                                    r,
                                )?;
                                exec.f8row_gemm(
                                    p,
                                    &sc.d_xq,
                                    &sc.d_xrs,
                                    &mut sc.d_zxbcdt,
                                    embd,
                                    in_rows,
                                    r,
                                )?;
                            }
                        }
                        // GGUF lane: repacked GEMV at r=1 (the serial decode
                        // class), the int8 mmq ladder above it
                        LinW::Qw(q) => {
                            if dec1 {
                                gemv_any(&exec, q, &sc.d_xn, &mut sc.d_zxbcdt)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_xn, embd, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    q,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_zxbcdt,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                    let win = bs.conv_win[li].as_mut().expect("conv arena");
                    let ssm = bs.ssm[li].as_mut().expect("ssm arena");
                    match cuts {
                        None => {
                            // every row advances its own slot's arena
                            exec.mamba_conv_step_batch(
                                win,
                                &sc.d_zxbcdt,
                                d_inner,
                                in_rows,
                                &sc.d_slots,
                                &w.conv_w,
                                &w.conv_b,
                                &mut sc.d_conv,
                                conv_dim,
                                hp.d_conv,
                                r,
                            )?;
                            ssm.scan_step_batch(
                                &exec,
                                &sc.d_conv,
                                &sc.d_zxbcdt,
                                d_inner + conv_dim,
                                in_rows,
                                &sc.d_slots,
                                &w.a,
                                &w.d,
                                &w.dt_bias,
                                &mut sc.d_y,
                                r,
                                hp.mamba_heads,
                                hp.mamba_head_dim,
                                hp.d_state,
                                hp.n_groups,
                            )?;
                        }
                        Some(c) => {
                            if c.dec > 0 {
                                exec.mamba_conv_step_batch(
                                    win,
                                    &sc.d_zxbcdt,
                                    d_inner,
                                    in_rows,
                                    &sc.d_slots,
                                    &w.conv_w,
                                    &w.conv_b,
                                    &mut sc.d_conv,
                                    conv_dim,
                                    hp.d_conv,
                                    c.dec,
                                )?;
                                ssm.scan_step_batch(
                                    &exec,
                                    &sc.d_conv,
                                    &sc.d_zxbcdt,
                                    d_inner + conv_dim,
                                    in_rows,
                                    &sc.d_slots,
                                    &w.a,
                                    &w.d,
                                    &w.dt_bias,
                                    &mut sc.d_y,
                                    c.dec,
                                    hp.mamba_heads,
                                    hp.mamba_head_dim,
                                    hp.d_state,
                                    hp.n_groups,
                                )?;
                            }
                            if verify {
                                // spec verify: conv on the slot's SCRATCH
                                // window (live stays pre-round), scan on the
                                // live state with per-row snapshots, xBC rows
                                // snapshotted for the window rebuild
                                let vp = bs.verify.as_mut().expect("verify planes");
                                let vw = vp.vwin[li].as_mut().expect("vwin");
                                let snap = vp.snap[li].as_mut().expect("snap");
                                for &(off, len, slot) in &c.runs {
                                    let s = slot as usize;
                                    exec.copy_region(
                                        win,
                                        s * win_elems,
                                        vw,
                                        s * win_elems,
                                        win_elems,
                                    )?;
                                    exec.mamba_conv_seq_at(
                                        vw,
                                        s * win_elems,
                                        &sc.d_zxbcdt,
                                        off * in_rows + d_inner,
                                        in_rows,
                                        &w.conv_w,
                                        &w.conv_b,
                                        &mut sc.d_conv,
                                        off * conv_dim,
                                        conv_dim,
                                        hp.d_conv,
                                        len,
                                    )?;
                                    ssm.scan_seq_snap_at(
                                        &exec,
                                        s * state_elems,
                                        &sc.d_conv,
                                        off * conv_dim,
                                        &sc.d_zxbcdt,
                                        off * in_rows + d_inner + conv_dim,
                                        in_rows,
                                        &w.a,
                                        &w.d,
                                        &w.dt_bias,
                                        &mut sc.d_y,
                                        off * d_inner,
                                        snap,
                                        off * state_elems,
                                        len,
                                        hp.mamba_heads,
                                        hp.mamba_head_dim,
                                        hp.d_state,
                                        hp.n_groups,
                                    )?;
                                }
                                exec.copy_rows_strided(
                                    &sc.d_zxbcdt,
                                    d_inner,
                                    in_rows,
                                    vp.xbc[li].as_mut().expect("xbc"),
                                    0,
                                    conv_dim,
                                    r,
                                )?;
                                // falls through to the arm's shared tail
                                // (gated norm + out_proj + residual)
                            } else {
                                // each chunk run advances its slot's arena
                                // sequentially - the run walk is the whole reason
                                // runs carry their slot. Checkpoint break rows
                                // split the ADVANCE only (never the pass): the
                                // state at the break is copied into the staging
                                // blob before the tail of the run continues.
                                for &(off, len, slot) in &c.runs {
                                    let s = slot as usize;
                                    let mut seg = off;
                                    for &(brow, stg) in
                                        c.breaks.iter().filter(|&&(b, _)| b > off && b <= off + len)
                                    {
                                        if brow > seg {
                                            exec.mamba_conv_seq_at(
                                                win,
                                                s * win_elems,
                                                &sc.d_zxbcdt,
                                                seg * in_rows + d_inner,
                                                in_rows,
                                                &w.conv_w,
                                                &w.conv_b,
                                                &mut sc.d_conv,
                                                seg * conv_dim,
                                                conv_dim,
                                                hp.d_conv,
                                                brow - seg,
                                            )?;
                                            ssm.scan_seq_at(
                                                &exec,
                                                s * state_elems,
                                                &sc.d_conv,
                                                seg * conv_dim,
                                                &sc.d_zxbcdt,
                                                seg * in_rows + d_inner + conv_dim,
                                                in_rows,
                                                &w.a,
                                                &w.d,
                                                &w.dt_bias,
                                                &mut sc.d_y,
                                                seg * d_inner,
                                                brow - seg,
                                                hp.mamba_heads,
                                                hp.mamba_head_dim,
                                                hp.d_state,
                                                hp.n_groups,
                                            )?;
                                            seg = brow;
                                        }
                                        let blob = mi * (state_elems + win_elems);
                                        let stage_buf = &mut bs.d_ckpt_stage[stg];
                                        ssm.save_to_blob(
                                            &exec,
                                            s * state_elems,
                                            stage_buf,
                                            blob,
                                            state_elems,
                                        )?;
                                        exec.copy_region(
                                            win,
                                            s * win_elems,
                                            stage_buf,
                                            blob + state_elems,
                                            win_elems,
                                        )?;
                                    }
                                    if off + len > seg {
                                        exec.mamba_conv_seq_at(
                                            win,
                                            s * win_elems,
                                            &sc.d_zxbcdt,
                                            seg * in_rows + d_inner,
                                            in_rows,
                                            &w.conv_w,
                                            &w.conv_b,
                                            &mut sc.d_conv,
                                            seg * conv_dim,
                                            conv_dim,
                                            hp.d_conv,
                                            off + len - seg,
                                        )?;
                                        ssm.scan_seq_at(
                                            &exec,
                                            s * state_elems,
                                            &sc.d_conv,
                                            seg * conv_dim,
                                            &sc.d_zxbcdt,
                                            seg * in_rows + d_inner + conv_dim,
                                            in_rows,
                                            &w.a,
                                            &w.d,
                                            &w.dt_bias,
                                            &mut sc.d_y,
                                            seg * d_inner,
                                            off + len - seg,
                                            hp.mamba_heads,
                                            hp.mamba_head_dim,
                                            hp.d_state,
                                            hp.n_groups,
                                        )?;
                                    }
                                }
                            }
                        }
                    }
                    exec.mamba_rmsnorm_gated_g(
                        &sc.d_y,
                        &sc.d_zxbcdt,
                        0,
                        in_rows,
                        &w.norm_w,
                        &mut sc.d_yn,
                        r,
                        d_inner,
                        hp.n_groups,
                        eps,
                    )?;
                    match &w.out_proj {
                        LinW::F8(p) => {
                            if dec1 {
                                exec.f8r_gemv(p, &sc.d_yn, &mut sc.d_proj, d_inner, embd)?;
                            } else {
                                exec.quantize_e4m3_row(
                                    &sc.d_yn,
                                    &mut sc.d_xq,
                                    &mut sc.d_xrs,
                                    d_inner,
                                    r,
                                )?;
                                exec.f8row_gemm(
                                    p,
                                    &sc.d_xq,
                                    &sc.d_xrs,
                                    &mut sc.d_proj,
                                    d_inner,
                                    embd,
                                    r,
                                )?;
                            }
                        }
                        LinW::Qw(q) => {
                            if dec1 {
                                gemv_any(&exec, q, &sc.d_yn, &mut sc.d_proj)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_yn, d_inner, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    q,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_proj,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                    mi += 1;
                }
                Mixer::Attn(w) => {
                    // NoPE - no rotary anywhere, projections go to the pool
                    // as computed
                    match w {
                        AttnWeights::F32 {
                            wq, wk, wv, bf16, ..
                        } => {
                            if dec1 {
                                if let Some(b) = bf16 {
                                    exec.bf16_gemv_rows(
                                        &b.wqkv,
                                        0,
                                        b.q_dim,
                                        &sc.d_xn,
                                        &mut sc.d_q,
                                    )?;
                                    exec.bf16_gemv_rows(
                                        &b.wqkv,
                                        b.q_dim,
                                        b.kv_dim,
                                        &sc.d_xn,
                                        &mut sc.d_k,
                                    )?;
                                    exec.bf16_gemv_rows(
                                        &b.wqkv,
                                        b.q_dim + b.kv_dim,
                                        b.kv_dim,
                                        &sc.d_xn,
                                        &mut sc.d_v,
                                    )?;
                                } else {
                                    exec.matvec_f32_batch(wq, &sc.d_xn, &mut sc.d_q, 1)?;
                                    exec.matvec_f32_batch(wk, &sc.d_xn, &mut sc.d_k, 1)?;
                                    exec.matvec_f32_batch(wv, &sc.d_xn, &mut sc.d_v, 1)?;
                                }
                            } else {
                                match (pf, bf16) {
                                    // batched decode/verify rows: the twins'
                                    // bf16 class (half the plane bytes - the
                                    // The c32 ledger had these f32 GEMMs at
                                    // 8.6% of GPU time). One fused launch past
                                    // the mr band - the thin k/v rows ride the
                                    // q grid instead of starving on their own
                                    // (thin-k/v rung). Prefill stays
                                    // on the exact-f32 planes, the
                                    // arbiter-gated class.
                                    (false, Some(b)) => {
                                        super::attn_qkv_batch(
                                            &exec,
                                            b,
                                            &sc.d_xn,
                                            &mut sc.d_q,
                                            &mut sc.d_k,
                                            &mut sc.d_v,
                                            r,
                                        )?;
                                    }
                                    _ => {
                                        exec.gemm_f32(
                                            &wq.buf,
                                            embd,
                                            q_dim,
                                            &sc.d_xn,
                                            &mut sc.d_q,
                                            r,
                                        )?;
                                        exec.gemm_f32(
                                            &wk.buf,
                                            embd,
                                            kv_dim,
                                            &sc.d_xn,
                                            &mut sc.d_k,
                                            r,
                                        )?;
                                        exec.gemm_f32(
                                            &wv.buf,
                                            embd,
                                            kv_dim,
                                            &sc.d_xn,
                                            &mut sc.d_v,
                                            r,
                                        )?;
                                    }
                                }
                            }
                        }
                        AttnWeights::Qw { wq, wk, wv, .. } => {
                            if dec1 {
                                gemv_any(&exec, wq, &sc.d_xn, &mut sc.d_q)?;
                                gemv_any(&exec, wk, &sc.d_xn, &mut sc.d_k)?;
                                gemv_any(&exec, wv, &sc.d_xn, &mut sc.d_v)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_xn, embd, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wq,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_q,
                                    r,
                                    pf,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wk,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_k,
                                    r,
                                    pf,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wv,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_v,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                    let kvs = bs.kv[li].as_mut().expect("paged kv");
                    exec.kv_append_batch_paged(
                        &sc.d_k,
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
                        &sc.d_v,
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
                        None => {
                            if ns > 1 && exec.has_attn_partial_batch_paged() {
                                exec.attn_partial_batch_paged(
                                    &sc.d_q,
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
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
                                    nh,
                                    hd,
                                    ns,
                                    r,
                                )?;
                            } else {
                                exec.attn_decode_batch_paged(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
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
                        Some(c) => {
                            if c.dec > 0 {
                                exec.attn_decode_batch_rows_paged(
                                    &sc.d_q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.d_sinks,
                                    &mut sc.d_attn,
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
                            for &(off, len, _slot) in &c.runs {
                                if wmma_pf {
                                    exec.attn_prefill_f16_paged_at(
                                        &sc.d_q,
                                        &kvs.k,
                                        &kvs.v,
                                        &sc.d_sinks,
                                        &mut sc.d_attn,
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
                                        &sc.d_q,
                                        &kvs.k,
                                        &kvs.v,
                                        &sc.d_sinks,
                                        &mut sc.d_attn,
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
                                        &sc.d_q,
                                        &kvs.k,
                                        &kvs.v,
                                        &sc.d_sinks,
                                        &mut sc.d_attn,
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
                    }
                    match w {
                        AttnWeights::F32 { wo, bf16, .. } => {
                            if dec1 {
                                if let Some(b) = bf16 {
                                    exec.bf16_gemv(&b.wo, None, &sc.d_attn, &mut sc.d_proj)?;
                                } else {
                                    exec.matvec_f32_batch(wo, &sc.d_attn, &mut sc.d_proj, 1)?;
                                }
                            } else {
                                match (pf, bf16) {
                                    // batched decode: bf16 twin class (see
                                    // the QKV arm above)
                                    (false, Some(b)) => {
                                        exec.bf16_gemm(&b.wo, None, &sc.d_attn, &mut sc.d_proj, r)?;
                                    }
                                    _ => {
                                        exec.gemm_f32(
                                            &wo.buf,
                                            q_dim,
                                            embd,
                                            &sc.d_attn,
                                            &mut sc.d_proj,
                                            r,
                                        )?;
                                    }
                                }
                            }
                        }
                        AttnWeights::Qw { wo, .. } => {
                            if dec1 {
                                gemv_any(&exec, wo, &sc.d_attn, &mut sc.d_proj)?;
                            } else {
                                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                                prefill_quant(
                                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_attn, q_dim, r,
                                )?;
                                dense_mm_pre(
                                    &exec,
                                    wo,
                                    &s8.xq,
                                    &s8.xs,
                                    &s8.yq,
                                    &mut s8.xsums,
                                    &mut s8.ssums,
                                    &mut s8.skfix,
                                    &mut sc.d_part,
                                    &mut sc.d_proj,
                                    r,
                                    pf,
                                )?;
                            }
                        }
                    }
                }
                Mixer::Moe(w) => {
                    exec.matvec_f32_batch(&w.router, &sc.d_xn, &mut sc.d_logits_r, r)?;
                    // shared fold-: the loader registers the
                    // shared expert as ns_sh pseudo-experts appended to the
                    // NVFP4 routed planes; the r>1 path then widens the topk
                    // rows by ns_sh constant picks and serves everything in
                    // one sorted-tile pair (the separate 1-block shared pass
                    // ran at 10-12% of the stream roof). dec1 and the Q8
                    // lane keep the plain k-wide rows.
                    let (ns_sh, moe_tiled) = match &w.planes {
                        MoePlanes::Nvf4 { up, .. } => (
                            up.n_expert - hp.n_expert,
                            up.layout == crate::gpu::Nvf4MoeLayout::Tiled64,
                        ),
                        _ => (0, false),
                    };
                    // skinny-tile decode election: tiled
                    // planes + a pure-decode tick route the ROUTED experts
                    // through the BM=8 pair (fill is ~2.4 at c32; 32-wide
                    // blocks are ~7.5% live) and the shared expert through
                    // the WIDE tiled pair on its resident planes - a BM=8
                    // fold-in would split the always-full shared pseudo-
                    // experts into ceil(r/8) blocks and re-read their strips.
                    // The tiny shared grid rides the routed grid's PDL tail
                    // (the rung-17 law working for us, not against).
                    let skinny = moe_tiled && !pf && !dec1;
                    if ns_sh > 0 && !dec1 && !skinny {
                        exec.moe_topk_sigmoid_batch_sh(
                            &sc.d_logits_r,
                            &w.bias.buf,
                            hp.routed_scale,
                            hp.n_expert,
                            hp.n_active,
                            ns_sh,
                            hp.n_expert,
                            &mut sc.d_idx,
                            &mut sc.d_w,
                            r,
                        )?;
                    } else {
                        exec.moe_topk_sigmoid_batch(
                            &sc.d_logits_r,
                            &w.bias.buf,
                            hp.routed_scale,
                            hp.n_expert,
                            hp.n_active,
                            &mut sc.d_idx,
                            &mut sc.d_w,
                            r,
                        )?;
                    }
                    //  diagnostic (PADDOCK_MOE_UNIQ=path), the task
                    // The MoE rung's attribution instrument: the real
                    // uniq-routed-experts-per-(tick,layer) histogram -
                    // pairs walks the full idx rows (k routed picks + the
                    // ns_sh shared folds in the _sh lane; the kernel's
                    // 128-bit bitmap skips ids >= 128, so uniq = ROUTED
                    // uniq and the shared folds only ride the pairs
                    // totals). Sits before the arm branches so every route
                    // is measured; launch-only, so captured decode graphs
                    // bake it in and it keeps counting on replays.
                    if sc.moe_uniq_dev != 0 {
                        let kw = if ns_sh > 0 && !dec1 && !skinny {
                            hp.n_active + ns_sh
                        } else {
                            hp.n_active
                        };
                        exec.moe_uniq_hist(&sc.d_idx, r * kw, hp.n_expert, sc.moe_uniq_dev)?;
                    }
                    let MoePlanes::Nvf4 {
                        up,
                        down,
                        sh_up,
                        sh_down,
                    } = &w.planes
                    else {
                        // GGUF lane: same class split as the serial spine -
                        // r=1 decode on the token-batched dp4a relu2 pair
                        // (write + one add; falls through to the residual
                        // add), r>1 on the sorted tiles folding straight
                        // into the residual
                        let MoePlanes::Q8 {
                            up,
                            down,
                            sh_up,
                            sh_down,
                        } = &w.planes
                        else {
                            unreachable!("nemotron MoePlanes is Nvf4 or Q8");
                        };
                        let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                        if dec1 {
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, embd)?;
                            exec.q8_0_moe_up_relu2(
                                up,
                                &sc.d_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.act_r,
                                hp.n_active,
                                1,
                            )?;
                            exec.quantize_q8(
                                &s8.act_r,
                                &mut s8.fq_r1,
                                &mut s8.fs_r1,
                                hp.n_active * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_down(
                                down,
                                &sc.d_idx,
                                &sc.d_w,
                                &s8.fq_r1,
                                &s8.fs_r1,
                                &mut sc.d_proj,
                                hp.n_active,
                                1,
                            )?;
                            exec.q8_0_moe_up_relu2(
                                sh_up,
                                &sc.d_sh_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.act_s,
                                1,
                                1,
                            )?;
                            exec.quantize_q8(
                                &s8.act_s,
                                &mut s8.fq_s1,
                                &mut s8.fs_s1,
                                hp.shared_ff,
                            )?;
                            exec.q8_0_moe_down(
                                sh_down,
                                &sc.d_sh_idx,
                                &sc.d_sh_w,
                                &s8.fq_s1,
                                &s8.fs_s1,
                                &mut s8.shproj,
                                1,
                                1,
                            )?;
                            exec.add(&mut sc.d_proj, &s8.shproj, embd)?;
                        } else if !pf && r <= MOE_DEC2_MAX_ROWS && moe_dec2_ok(&exec) {
                            // DECODE BAND. Two separate
                            // shape mistakes shared one arm before this:
                            //
                            //  - the ROUTED experts rode the sorted BM=32
                            //    tile, which at r=4/top-6 over 128 experts
                            //    puts one real row in a 32-row block. It is
                            //    not just wasted flops: the tile measured
                            //    FLAT at ~144 GB/s where the same bytes on
                            //    the dec2 pair (warp per output row, no pad,
                            //    no align, no combine) stream at ~660.
                            //  - the shared expert is not an expert at all.
                            //    Every row uses it, so it is a plain dense
                            //    FFN, and it belongs on the same q8 ladder
                            //    the dense projections took at the time: one
                            //    weight pass for the whole tick instead of
                            //    the 1-block align+tile pair, which was
                            //    another flat ~117 GB/s. The only thing that
                            //    was missing is the activation - hence
                            //    quantize_q8_relu2, which folds relu(x)^2
                            //    into the quantize between up and down and
                            //    is bit-identical to doing it in f32.
                            //
                            // The epilogue quantize shrinks with the routed
                            // plane too: nb*32*moe_ff -> r*n_active*moe_ff,
                            // 24x fewer elements at r=4 (it was 16% of the
                            // tick in the profile).
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, r * embd)?;
                            exec.q8_0_moe_up_relu2_dec2(
                                up,
                                &sc.d_idx,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.fu_r,
                                hp.n_active,
                                r,
                                0,
                            )?;
                            exec.quantize_q8(
                                &s8.fu_r,
                                &mut s8.fq_r,
                                &mut s8.fs_r,
                                r * hp.n_active * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_dn_dec2(
                                down,
                                &sc.d_idx,
                                &sc.d_w,
                                &s8.fq_r,
                                &s8.fs_r,
                                &mut sc.d_proj,
                                hp.n_active,
                                r,
                            )?;
                            mmq_pre(
                                &exec,
                                sh_up,
                                &s8.xq,
                                &s8.xs,
                                &mut sc.d_part,
                                &mut s8.fu_s,
                                r,
                            )?;
                            exec.quantize_q8_relu2(
                                &s8.fu_s,
                                &mut s8.fq_s,
                                &mut s8.fs_s,
                                r * hp.shared_ff,
                            )?;
                            mmq_pre(
                                &exec,
                                sh_down,
                                &s8.fq_s,
                                &s8.fs_s,
                                &mut sc.d_part,
                                &mut s8.shproj,
                                r,
                            )?;
                            exec.add(&mut sc.d_proj, &s8.shproj, r * embd)?;
                        } else {
                            exec.quantize_q8(&sc.d_xn, &mut s8.xq, &mut s8.xs, r * embd)?;
                            let nbr = moe_live_blocks(r, hp.n_active, hp.n_expert, sc.nb_r);
                            exec.moe_align(
                                &sc.d_idx,
                                &mut sc.d_srow,
                                &mut sc.d_sslot,
                                &mut sc.d_bexp,
                                r,
                                hp.n_active,
                                hp.n_expert,
                                nbr,
                            )?;
                            exec.q8_0_moe_up_relu2_sorted(
                                up,
                                &sc.d_srow,
                                &sc.d_bexp,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.fu_r,
                                nbr,
                            )?;
                            exec.quantize_q8(
                                &s8.fu_r,
                                &mut s8.fq_r,
                                &mut s8.fs_r,
                                nbr * 32 * hp.moe_ff,
                            )?;
                            exec.q8_0_moe_down_sorted(
                                down,
                                &sc.d_srow,
                                &sc.d_sslot,
                                &sc.d_bexp,
                                &sc.d_w,
                                &s8.fq_r,
                                &s8.fs_r,
                                &mut sc.d_part,
                                hp.n_active,
                                nbr,
                            )?;
                            exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, hp.n_active, r)?;
                            let nbs = moe_live_blocks(r, 1, 1, sc.nb_s);
                            exec.moe_align(
                                &sc.d_sh_idx,
                                &mut sc.d_srow_s,
                                &mut sc.d_sslot_s,
                                &mut sc.d_bexp_s,
                                r,
                                1,
                                1,
                                nbs,
                            )?;
                            exec.q8_0_moe_up_relu2_sorted(
                                sh_up,
                                &sc.d_srow_s,
                                &sc.d_bexp_s,
                                &s8.xq,
                                &s8.xs,
                                &mut s8.fu_s,
                                nbs,
                            )?;
                            exec.quantize_q8(
                                &s8.fu_s,
                                &mut s8.fq_s,
                                &mut s8.fs_s,
                                nbs * 32 * hp.shared_ff,
                            )?;
                            exec.q8_0_moe_down_sorted(
                                sh_down,
                                &sc.d_srow_s,
                                &sc.d_sslot_s,
                                &sc.d_bexp_s,
                                &sc.d_sh_w,
                                &s8.fq_s,
                                &s8.fs_s,
                                &mut sc.d_proj,
                                1,
                                nbs,
                            )?;
                            exec.moe_slot_combine(&sc.d_proj, &mut sc.d_x, embd, 1, r)?;
                            continue;
                        }
                        // residual add for the two arms that leave their
                        // whole MoE output in d_proj (dec1 and the decode
                        // band); the sorted arm folds through slot_combine
                        // and skips this with its own continue
                        exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
                        continue;
                    };
                    if dec1 {
                        // the serial decode's fused wave-dense pair + the
                        // fixed ascending-slot fold into the residual
                        // (tiled planes ride the regrouped _mtt twins)
                        if moe_tiled {
                            exec.nvf4_moe_up_relu2_mtt(
                                up,
                                sh_up,
                                &sc.d_idx,
                                &sc.d_xn,
                                &mut sc.d_act,
                                hp.n_active,
                            )?;
                            exec.nvf4_moe_down_part_tt(
                                down,
                                sh_down,
                                &sc.d_idx,
                                &sc.d_w,
                                &sc.d_act,
                                &mut sc.d_part7,
                                hp.n_active,
                            )?;
                        } else {
                            exec.nvf4_moe_up_relu2_mt(
                                up,
                                sh_up,
                                &sc.d_idx,
                                &sc.d_xn,
                                &mut sc.d_act,
                                hp.n_active,
                            )?;
                            exec.nvf4_moe_down_part(
                                down,
                                sh_down,
                                &sc.d_idx,
                                &sc.d_w,
                                &sc.d_act,
                                &mut sc.d_part7,
                                hp.n_active,
                            )?;
                        }
                        exec.moe_slot_combine(&sc.d_part7, &mut sc.d_x, embd, hp.n_active + 1, 1)?;
                        continue;
                    }
                    // sorted-tile mxf4nvf4 MMA class (the serial bulk
                    // prefill's rung-2 lane at r rows; BM=32 pad waste at
                    // small r is the accepted first cut). With the shared
                    // fold-in (ns_sh > 0) the shared expert's pseudo-expert
                    // blocks ride the same align + pair launch - its picks
                    // occupy slots n_active.., and the fixed-order combine
                    // sums the down K halves (the sanctioned split-K
                    // regroup, same class as the slot fold itself).
                    if skinny {
                        // routed experts: plain topk rows, BM=8 blocks
                        let kw = hp.n_active;
                        let np = hp.n_active + 1;
                        if !pro_done {
                            exec.quantize_nvf4(&sc.d_xn, &mut sc.d_xq4, &mut sc.d_xs4, r * embd)?;
                        }
                        let nbr = moe_live_blocks_bm8(r, kw, hp.n_expert, sc.nb_r);
                        exec.moe_align_bm(
                            &sc.d_idx,
                            &mut sc.d_srow,
                            &mut sc.d_sslot,
                            &mut sc.d_bexp,
                            r,
                            kw,
                            hp.n_expert,
                            8,
                            nbr,
                        )?;
                        exec.nvf4_moe_up_relu2_st(
                            up,
                            &sc.d_srow,
                            &sc.d_bexp,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq,
                            &mut sc.d_fs,
                            nbr,
                            8,
                        )?;
                        exec.nvf4_moe_down_st(
                            down,
                            &sc.d_srow,
                            &sc.d_sslot,
                            &sc.d_bexp,
                            Some(&sc.d_w),
                            &sc.d_fq,
                            &sc.d_fs,
                            &mut sc.d_part,
                            kw,
                            np,
                            0,
                            nbr,
                            8,
                        )?;
                        // shared expert: resident sh planes, WIDE tiled pair
                        // (full 32-blocks; the 1-block grid overlaps the
                        // routed grid's drain under PDL)
                        let nbs = moe_live_blocks(r, 1, 1, sc.nb_s);
                        exec.moe_align(
                            &sc.d_sh_idx,
                            &mut sc.d_srow_s,
                            &mut sc.d_sslot_s,
                            &mut sc.d_bexp_s,
                            r,
                            1,
                            1,
                            nbs,
                        )?;
                        exec.nvf4_moe_up_relu2_st(
                            sh_up,
                            &sc.d_srow_s,
                            &sc.d_bexp_s,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq_s,
                            &mut sc.d_fs_s,
                            nbs,
                            32,
                        )?;
                        exec.nvf4_moe_down_st(
                            sh_down,
                            &sc.d_srow_s,
                            &sc.d_sslot_s,
                            &sc.d_bexp_s,
                            None,
                            &sc.d_fq_s,
                            &sc.d_fs_s,
                            &mut sc.d_part,
                            1,
                            np,
                            hp.n_active,
                            nbs,
                            32,
                        )?;
                        exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, np, r)?;
                        continue;
                    }
                    let kw = hp.n_active + ns_sh;
                    let np = if ns_sh > 0 { kw } else { hp.n_active + 1 };
                    if !pro_done {
                        exec.quantize_nvf4(&sc.d_xn, &mut sc.d_xq4, &mut sc.d_xs4, r * embd)?;
                    }
                    // same live-block extent as the Q8 arm - this one has no
                    // capacity-sized epilogue quantize (up_bs writes fq/fs per
                    // block), so all it drops is pad CTAs. UNMEASURED on
                    // sm_120: nvf4 needs a Blackwell die.
                    let nbr = moe_live_blocks(r, kw, hp.n_expert + ns_sh, sc.nb_r);
                    exec.moe_align(
                        &sc.d_idx,
                        &mut sc.d_srow,
                        &mut sc.d_sslot,
                        &mut sc.d_bexp,
                        r,
                        kw,
                        hp.n_expert + ns_sh,
                        nbr,
                    )?;
                    if moe_tiled {
                        exec.nvf4_moe_up_relu2_st(
                            up,
                            &sc.d_srow,
                            &sc.d_bexp,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq,
                            &mut sc.d_fs,
                            nbr,
                            32,
                        )?;
                        exec.nvf4_moe_down_st(
                            down,
                            &sc.d_srow,
                            &sc.d_sslot,
                            &sc.d_bexp,
                            Some(&sc.d_w),
                            &sc.d_fq,
                            &sc.d_fs,
                            &mut sc.d_part,
                            kw,
                            np,
                            0,
                            nbr,
                            32,
                        )?;
                    } else {
                        exec.nvf4_moe_up_relu2_bs(
                            up,
                            &sc.d_srow,
                            &sc.d_bexp,
                            &sc.d_xq4,
                            &sc.d_xs4,
                            &mut sc.d_fq,
                            &mut sc.d_fs,
                            nbr,
                        )?;
                        exec.nvf4_moe_down_bs(
                            down,
                            &sc.d_srow,
                            &sc.d_sslot,
                            &sc.d_bexp,
                            Some(&sc.d_w),
                            &sc.d_fq,
                            &sc.d_fs,
                            &mut sc.d_part,
                            kw,
                            np,
                            0,
                            nbr,
                        )?;
                    }
                    if ns_sh == 0 {
                        // No fold-in (shared_ff not a clean multiple of
                        // moe_ff): the separate 1-block shared pass. Its grid
                        // is (1, rt) - 29 and 21 CTAs on a 188-SM die at ~20%
                        // of peak DRAM - which looks like an obvious
                        // underfill rung and is not one: a delete-the-work
                        // probe measured its WALL cost at ~zero. It rides
                        // entirely in the routed pair's PDL shadow. Do not
                        // build a K-split for it without new evidence.
                        let nbs = moe_live_blocks(r, 1, 1, sc.nb_s);
                        exec.moe_align(
                            &sc.d_sh_idx,
                            &mut sc.d_srow_s,
                            &mut sc.d_sslot_s,
                            &mut sc.d_bexp_s,
                            r,
                            1,
                            1,
                            nbs,
                        )?;
                        if moe_tiled {
                            exec.nvf4_moe_up_relu2_st(
                                sh_up,
                                &sc.d_srow_s,
                                &sc.d_bexp_s,
                                &sc.d_xq4,
                                &sc.d_xs4,
                                &mut sc.d_fq_s,
                                &mut sc.d_fs_s,
                                nbs,
                                32,
                            )?;
                            exec.nvf4_moe_down_st(
                                sh_down,
                                &sc.d_srow_s,
                                &sc.d_sslot_s,
                                &sc.d_bexp_s,
                                None,
                                &sc.d_fq_s,
                                &sc.d_fs_s,
                                &mut sc.d_part,
                                1,
                                np,
                                hp.n_active,
                                nbs,
                                32,
                            )?;
                        } else {
                            exec.nvf4_moe_up_relu2_bs(
                                sh_up,
                                &sc.d_srow_s,
                                &sc.d_bexp_s,
                                &sc.d_xq4,
                                &sc.d_xs4,
                                &mut sc.d_fq_s,
                                &mut sc.d_fs_s,
                                nbs,
                            )?;
                            exec.nvf4_moe_down_bs(
                                sh_down,
                                &sc.d_srow_s,
                                &sc.d_sslot_s,
                                &sc.d_bexp_s,
                                None,
                                &sc.d_fq_s,
                                &sc.d_fs_s,
                                &mut sc.d_part,
                                1,
                                np,
                                hp.n_active,
                                nbs,
                            )?;
                        }
                    }
                    exec.moe_slot_combine(&sc.d_part, &mut sc.d_x, embd, np, r)?;
                    continue;
                }
            }
            // Hoist this add into the next layer's prologue when that layer is
            // an nvf4 MoE on the bs arm - the shape the fused kernel serves,
            // and the one every non-MoE layer in this checkpoint precedes.
            let next_bs_moe = glue_fuse
                && matches!(
                    self.layers.get(li + 1).map(|l| &l.mixer),
                    Some(Mixer::Moe(w)) if matches!(w.planes, MoePlanes::Nvf4 { .. })
                );
            if next_bs_moe {
                let next_w = &self.layers[li + 1].norm.buf;
                exec.add_rmsnorm_quant_nvf4_batch(
                    &mut bs.sc.d_x,
                    Some(&bs.sc.d_proj),
                    next_w,
                    &mut bs.sc.d_xn,
                    &mut bs.sc.d_xq4,
                    &mut bs.sc.d_xs4,
                    embd,
                    eps,
                    r,
                )?;
                fused_pro = true;
            } else {
                exec.add(&mut bs.sc.d_x, &bs.sc.d_proj, r * embd)?;
            }
        }
        // last aux tap: the final layer's post-block residual
        if let Some(df) = self.dflash.as_mut()
            && let Some(st) = df.state.as_mut()
            && let Some(ai) = df
                .target_layers
                .iter()
                .position(|&t| t == self.hp.n_layer - 1)
        {
            let sc = &bs.sc;
            exec.copy_region(&sc.d_x, 0, &mut st.aux[ai], 0, r * embd)?;
        }
        Ok(())
    }

    /// Final norm + lm_head over rows 0..rows, leaving [rows, vocab] in
    /// head_logits (row-batched GEMV - bit-exact per row vs the serial head).
    fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.hidden, self.hp.eps);
        let final_norm = self.final_norm.buf.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(&sc.d_x, &final_norm, &mut sc.d_xn, embd, eps, rows)?;
        match &self.lm_head {
            HeadW::Nvf4(h) => {
                super::head_nvf4_batch(&exec, h, &sc.d_xn, &mut sc.head_logits, rows)?
            }
            // GGUF lane: the mmq ladder at batch=rows (strided mma at the
            // decode widths) - row-batched, same class as the serial head
            HeadW::Qw(q) => {
                let s8 = sc.q8.as_mut().expect("q8 batch scratch");
                prefill_quant(
                    &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &sc.d_xn, embd, rows,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    q,
                    &s8.xq,
                    &s8.xs,
                    &s8.yq,
                    &mut s8.xsums,
                    &mut s8.ssums,
                    &mut s8.skfix,
                    &mut sc.head_logits,
                    rows,
                )?;
            }
        }
        Ok(())
    }

    /// Stage residual row `row` at row 0 so a single-row head pass reads it.
    /// Bounced through `d_proj` because src and dst share a buffer.
    fn head_row_at(&mut self, row: usize) -> Result<(), GpuModelError> {
        let embd = self.hp.hidden;
        if row > 0 {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src = sc
                .d_x
                .try_slice(row * embd..(row + 1) * embd)
                .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .d_proj
                .try_slice_mut(0..embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let ps = sc
                .d_proj
                .try_slice(0..embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            let mut xd = sc
                .d_x
                .try_slice_mut(0..embd)
                .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)
    }

    /// Prefill tail: head over residual row `row`, one vocab row to host.
    pub(super) fn head_row(&mut self, row: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.vocab;
        self.head_row_at(row)?;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    /// Read the [rows, vocab] logits back to the host.
    pub(crate) fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.vocab;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    // ── decode ticks + per-r graphs ────────────────────────────────────────

    /// The pure-device decode tick body - everything the per-r graph
    /// captures. All inputs are device buffers written before replay
    /// (d_tok/d_pos/d_slots + the block tables); the mamba step kernels read
    /// their slot indirection from d_slots, so one capture serves any slot
    /// composition at this r.
    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.embed_rows(r)?;
        self.layer_walk(r, None, false)?;
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) {
            // decode rows' features (positions/slots still live in the
            // scratch streams the tick uploaded); coverage notes happen at
            // the host call sites - inside a captured graph only the device
            // ops replay
            self.dflash_append_features(r)?;
        }
        self.head_rows(r)
    }

    /// Record `body`'s launches into a CUDA graph (recording only). An alloc
    /// during capture is a hard driver error - every plane exists at enable.
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

    /// One batched decode step with explicit slot ids (the identity mapping
    /// only holds when the live set is a dense prefix). Leaves [r, vocab]
    /// logits in head_logits.
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
        self.step_replay(r)?;
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) {
            for i in 0..r {
                let p = positions[i] as usize;
                self.dflash_note_rows(slots[i] as usize, p, p + 1);
            }
        }
        self.mtp_append_ticks(slots, positions)?;
        Ok(())
    }

    pub(crate) fn batch_step(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        let ident: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &ident)
    }

    // ── prefill lanes ──────────────────────────────────────────────────────

    /// The checkpoint plan for rows covering positions `[base, base+len)` of
    /// a prompt of `t_len` rows resumed at `start`: which pass rows (offset
    /// by `row0`, the rows' base index within the whole pass) end at a
    /// `ckpt_cuts` boundary. Returns (pass-row breaks for PfCuts, and the
    /// (stage, cut) commits to run after the pass). `stage0` threads the
    /// stage counter across multiple prompts sharing one pass.
    fn stage_plan(
        t_len: usize,
        start: usize,
        base: usize,
        len: usize,
        row0: usize,
        stage0: &mut usize,
        step: usize,
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut breaks = Vec::new();
        let mut after = Vec::new();
        for cut in super::prefix::ckpt_cuts(t_len, step) {
            if cut > start.max(base) && cut <= base + len && *stage0 < super::prefix::CKPT_STAGES {
                breaks.push((row0 + (cut - base), *stage0));
                after.push((*stage0, cut));
                *stage0 += 1;
            }
        }
        (breaks, after)
    }

    /// Prefill a whole prompt into `slot` (chunked at PREFILL_CHUNK) and
    /// return the last token's logits. Trailing-boundary checkpoints stage
    /// during the passes and commit between chunks (stage D).
    pub(crate) fn forward_prefill_impl(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.admit_rows(slot, tokens.len())?;
        let start = self.prefix_resume_rows(slot, tokens, tokens.len())?;
        let mut base = start;
        let mut last_len = 0usize;
        for chunk in tokens[base..].chunks(PREFILL_CHUNK) {
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            let mut stage = 0usize;
            let (breaks, after) = Self::stage_plan(
                tokens.len(),
                start,
                base,
                chunk.len(),
                0,
                &mut stage,
                self.tier_ckpt_step(),
            );
            self.rows_pass_body(&rows, 0, breaks)?;
            for (st, cut) in after {
                self.commit_stage(st, slot, tokens, cut);
            }
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens);
        self.head_row(last_len - 1)
    }

    /// COALESCED multi-prompt prefill: every pending prompt's rows
    /// concatenate into shared chunks - one weight-amortized pass over the
    /// wave (granite's shape; run isolation keeps attention AND the
    /// recurrent advance per slot).
    pub(crate) fn forward_prefill_batch_impl(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        if items.len() == 1 || paddock_models::dev_var_os!("PADDOCK_NO_COALESCED_PREFILL").is_some()
        {
            return items
                .iter()
                .map(|(slot, toks)| self.forward_prefill_impl(*slot, toks))
                .collect();
        }
        let mut starts = vec![0usize; items.len()];
        for (it, (slot, tokens)) in items.iter().enumerate() {
            self.admit_rows(*slot, tokens.len())?;
            starts[it] = self.prefix_resume_rows(*slot, tokens, tokens.len())?;
        }
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut last_row = vec![0usize; items.len()];
        for (it, (slot, toks)) in items.iter().enumerate() {
            for (j, &t) in toks.iter().enumerate().skip(starts[it]) {
                rows.push((*slot as u32, j as u32, t));
            }
            last_row[it] = rows.len() - 1;
        }
        // per-item global row base within the wave stream, for the cut plan
        let mut item_base = vec![0usize; items.len()];
        {
            let mut acc = 0usize;
            for (it, (_, toks)) in items.iter().enumerate() {
                item_base[it] = acc;
                acc += toks.len() - starts[it];
            }
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let step = self.tier_ckpt_step();
        let mut base = 0usize;
        for chunk in rows.chunks(PREFILL_CHUNK) {
            let r = chunk.len();
            // finishers whose last row landed in this chunk read inside the
            // pass - the next chunk's embed overwrites d_x. Ascending by row
            // because head_row bounces its row through x[0].
            let mut fin: Vec<(usize, usize)> = last_row
                .iter()
                .enumerate()
                .filter(|&(_, &lr)| lr >= base && lr < base + r)
                .map(|(it, &lr)| (lr - base, it))
                .collect();
            fin.sort_unstable();
            // checkpoint cuts of every item whose boundary rows land in this
            // chunk; global row of item position p = item_base + (p - start)
            let mut breaks: Vec<(usize, usize)> = Vec::new();
            let mut after: Vec<(usize, usize, usize)> = Vec::new();
            let mut stage = 0usize;
            for (it, (_, toks)) in items.iter().enumerate() {
                for cut in super::prefix::ckpt_cuts(toks.len(), step) {
                    if cut <= starts[it] || stage >= super::prefix::CKPT_STAGES {
                        continue;
                    }
                    let grow = item_base[it] + (cut - starts[it]);
                    if grow > base && grow <= base + r {
                        breaks.push((grow - base, stage));
                        after.push((stage, it, cut));
                        stage += 1;
                    }
                }
            }
            breaks.sort_unstable();
            self.rows_pass_body(chunk, 0, breaks)?;
            for (st, it, cut) in after {
                let (slot, toks) = &items[it];
                let keys = toks.clone();
                self.commit_stage(st, *slot, &keys, cut);
            }
            for (row, it) in fin {
                out[it] = self.head_row(row)?;
            }
            base += r;
        }
        for (slot, toks) in items {
            self.prefix_insert(*slot, toks);
        }
        Ok(out)
    }

    /// Queue a prompt for STALL-FREE chunked prefill (Sarathi shape). Does
    /// the whole admission prologue now, so a mixed tick only moves rows.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        // a queued entry for this slot is stale (the old request died and the
        // slot was reused): evict rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        self.admit_rows(slot, tokens.len())?;
        let cursor = self.prefix_resume_rows(slot, &tokens, tokens.len())?;
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG_IDS").is_some() {
            tracing::info!("[spec2a-ids] admit slot={slot} cursor={cursor} prompt={tokens:?}");
        }
        self.chunked.push(ChunkedPrefill {
            slot,
            keys: tokens.clone(),
            tokens,
            cursor,
        });
        Ok(())
    }

    /// Drop slot's in-flight prefill (client hung up mid-prompt).
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// Pick this tick's chunk rows: FIFO over the queue, up to `budget`
    /// rows, splitting the last prompt if it does not fit.
    fn plan_chunk(&self, budget: usize) -> (Vec<(u32, u32, u32)>, Vec<(usize, usize, bool)>) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        if self.chunked.is_empty() {
            return (rows, take);
        }
        let cap = budget.clamp(1, PREFILL_CHUNK);
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
            let keys = std::mem::take(&mut self.chunked[qi].keys);
            self.prefix_insert(slot, &keys);
            out.push((slot, fs, toks.len()));
        }
        self.chunked.retain(|c| !c.tokens.is_empty());
        out
    }

    /// Build the fused tick's row stream: decode rows first (one band), then
    /// as much of the prefill queue as scratch capacity allows.
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

    /// The mixed tick's checkpoint plan: cuts of any queued prompt whose
    /// boundary rows land inside this tick's take. Returns (PfCuts breaks,
    /// (stage, queue index, cut) commits for after the pass).
    fn mixed_stage_plan(
        &self,
        take: &[(usize, usize, bool)],
        dec_n: usize,
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize, usize)>) {
        let mut breaks = Vec::new();
        let mut after = Vec::new();
        let mut stage = 0usize;
        let mut row_base = dec_n;
        let step = self.tier_ckpt_step();
        for &(qi, n, _) in take {
            let c = &self.chunked[qi];
            for cut in super::prefix::ckpt_cuts(c.tokens.len(), step) {
                if cut > c.cursor && cut <= c.cursor + n && stage < super::prefix::CKPT_STAGES {
                    breaks.push((row_base + (cut - c.cursor), stage));
                    after.push((stage, qi, cut));
                    stage += 1;
                }
            }
            row_base += n;
        }
        (breaks, after)
    }

    /// Run the staged checkpoint commits after a mixed tick's pass.
    fn mixed_stage_commit(&mut self, after: Vec<(usize, usize, usize)>) {
        for (st, qi, cut) in after {
            let slot = self.chunked[qi].slot;
            let keys = self.chunked[qi].keys.clone();
            self.commit_stage(st, slot, &keys, cut);
        }
    }

    /// One FUSED mixed tick: decode rows and the prefill chunk in a single
    /// weight-amortized pass, decode rows device-sampled (granite's shape).
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
        // Nothing queued -> a plain decode tick on the captured graph.
        if self.chunked.is_empty() {
            let step = if decodes.is_empty() {
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                }
            } else {
                let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
                let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
                let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
                self.forward_batch_sampled_slots(&toks, &pos, Some(&slots), plans)?
            };
            return Ok((step, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        let (breaks, after) = self.mixed_stage_plan(&take, dec_n);
        self.rows_pass_body(&rows, dec_n, breaks)?;
        self.mixed_stage_commit(after);
        // Decode rows first: one bulk head over rows 0..dec_n, then device
        // sampling - it must precede the finisher heads because head_row
        // bounces through x[0] and rewrites head_logits[0..vocab].
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

    /// The unsampled mixed tick (full logits readback).
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
        let (breaks, after) = self.mixed_stage_plan(&take, dec_n);
        self.rows_pass_body(&rows, dec_n, breaks)?;
        self.mixed_stage_commit(after);
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

    // ── device sampling ────────────────────────────────────────────────────

    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows
    /// stay mode 0 = untouched.
    /// TruncCat rows pack mode 5 (top_k 1..=64) or mode 6 (k-less -
    /// nemotron's own election) + the tpar side plane (Some iff any).
    fn pack_samp_par(
        plans: &[crate::generator::RowSample],
    ) -> (Vec<u32>, Option<Vec<u32>>, bool, bool) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        // which trunc chains a tick actually needs: nemotron's own election
        // is pure mode 6, so the mode-5 launches would all early-return -
        // skip whole chains, not rows (a launch is the unit of waste here)
        let (mut any5, mut any6) = (false, false);
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    let mode5 = *k >= 1 && *k <= 64;
                    par[i * 4 + 2] = if mode5 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    if mode5 { any5 = true } else { any6 = true }
                }
                // RS plans are spec-only; nemotron's batch lane has no
                // drafter yet
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, (any5 || any6).then_some(tpar), any5, any6)
    }

    /// device-truncation engagement witness (bisect-trap law): once per process.
    fn trunc_dev_witness(rows: usize) {
        static DEV: std::sync::Once = std::sync::Once::new();
        DEV.call_once(|| {
            eprintln!("[trunc-dev6] engaged: r={rows} (nemotron device truncation sampling)");
        });
    }

    /// TruncCat rows execute fully on device (slots 435+436).
    pub(crate) fn device_trunc_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head_logits rows 0..r on device with `plans`; only Host-plan
    /// rows pay a vocab-row readback. Assumes the head already ran.
    fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.hp.vocab;
        let (par, tpar, any5, any6) = Self::pack_samp_par(plans);
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..r * 4)
                    .ok_or_else(|| GpuError::Driver("d_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
            if tpar.is_some() {
                Self::trunc_dev_witness(r);
                if any5 {
                    exec.sample_rows_t_at(
                        &sc.head_logits,
                        &sc.d_par,
                        0,
                        &sc.d_tpar,
                        0,
                        &mut sc.d_out,
                        0,
                        r,
                        vocab,
                    )?;
                }
                if any6 {
                    exec.sample_rows_p_at(
                        &sc.head_logits,
                        &sc.d_par,
                        0,
                        &sc.d_tpar,
                        0,
                        &mut sc.d_out,
                        0,
                        r,
                        vocab,
                    )?;
                }
            }
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

    /// Device-sampled decode tick: graph replay + sample_rows.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.forward_batch_sampled_slots(tokens, positions, None, plans)
    }

    pub(crate) fn forward_batch_sampled_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        let r = tokens.len();
        let owned: Vec<u32> = (0..r as u32).collect();
        let ident: &[u32] = slots.unwrap_or(&owned);
        assert_eq!(ident.len(), r, "one slot per row");
        self.ensure_rows(ident, positions)?;
        self.upload_rows(tokens, positions, ident)?;
        self.step_replay(r)?;
        let step = self.sample_head_rows(r, plans)?;
        self.mtp_append_ticks(ident, positions)?;
        Ok(step)
    }

    // ── batched depth-2 decode pipe (stage E, granite's pipe-under-pool) ───

    pub(crate) fn supports_decode_pipe_batch(&self) -> bool {
        self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
            // the in-file MTP's h chain needs host staging around every
            // tick - incompatible with the pipe's fire-and-forget replays.
            // Spec-on serving decodes through spec rounds instead; --no-spec
            // serves never load the block, so the pipe survives there.
            && !self.mtp_active()
    }

    fn pipe_launch_tick_b(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.hp.vocab;
        let (b, tick) = {
            let p = self.pipe_b.as_ref().expect("pipe active");
            (p.b, p.tick)
        };
        // back every row's THIS-tick write position before anything mutates -
        // a growth error leaves the rings/inputs untouched
        {
            let (pos0, slot_map) = {
                let p = self.pipe_b.as_ref().expect("pipe active");
                (p.pos0.clone(), p.slots.clone())
            };
            let slots_v: Vec<u32> = (0..b as u32)
                .map(|i| slot_map.as_ref().map_or(i, |s| s[i as usize]))
                .collect();
            let pos_v: Vec<u32> = pos0.iter().map(|&p0| p0 + tick as u32).collect();
            self.ensure_rows(&slots_v, &pos_v)?;
        }
        let ring = tick % 2;
        let (par, tpar, any5, any6) = Self::pack_samp_par(plans);
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let off = ring * n_slots * 4;
            let mut v = sc
                .d_pipe_par
                .try_slice_mut(off..off + b * 4)
                .ok_or_else(|| GpuError::Driver("d_pipe_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_pipe_tpar
                    .try_slice_mut(off..off + b * 4)
                    .ok_or_else(|| GpuError::Driver("d_pipe_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
        }
        if advance {
            // tokens <- previous ring's sampled ids, positions += 1, on device
            let prev = (tick + 1) % 2;
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let (out, tok, pos) = (&sc.d_pipe_out, &mut sc.d_tok, &mut sc.d_pos);
            exec.pipe_advance(out, prev * n_slots, tok, pos, b)?;
        }
        self.step_replay(b)?;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            exec.sample_rows_at(
                &sc.head_logits,
                &sc.d_pipe_par,
                ring * n_slots * 4,
                &mut sc.d_pipe_out,
                ring * n_slots,
                b,
                vocab,
            )?;
            // trunc rows draw into the same out ring - pipe_advance
            // feeds their ids forward exactly like mode-1/2 rows
            if tpar.is_some() {
                Self::trunc_dev_witness(b);
                if any5 {
                    exec.sample_rows_t_at(
                        &sc.head_logits,
                        &sc.d_pipe_par,
                        ring * n_slots * 4,
                        &sc.d_pipe_tpar,
                        ring * n_slots * 4,
                        &mut sc.d_pipe_out,
                        ring * n_slots,
                        b,
                        vocab,
                    )?;
                }
                if any6 {
                    exec.sample_rows_p_at(
                        &sc.head_logits,
                        &sc.d_pipe_par,
                        ring * n_slots * 4,
                        &sc.d_pipe_tpar,
                        ring * n_slots * 4,
                        &mut sc.d_pipe_out,
                        ring * n_slots,
                        b,
                        vocab,
                    )?;
                }
            }
        }
        let ev = exec.record_event()?;
        self.pipe_b.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    pub(crate) fn decode_pipe_begin_b(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        assert_eq!(positions.len(), b, "one position per row");
        if !self.supports_decode_pipe_batch() {
            return Err(GpuModelError::Config("decode pipe unsupported".into()));
        }
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.n_slots => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.n_slots,
                });
            }
            _ => {}
        }
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.pipe_b.is_none(), "decode pipe already active");
        // tick-0 inputs land in the fixed graph buffers (advance=false keeps
        // them); ensure_rows runs inside pipe_launch_tick_b at tick 0
        let ident: Vec<u32> = (0..b as u32).collect();
        self.upload_rows(tokens, positions, slots.unwrap_or(&ident))?;
        self.pipe_b = Some(PipeB {
            b,
            tick: 0,
            ev: [None, None],
            pos0: positions.to_vec(),
            slots: slots.map(<[u32]>::to_vec),
        });
        if let Err(e) = self.pipe_launch_tick_b(plans, false) {
            self.pipe_b_abort();
            return Err(e);
        }
        Ok(())
    }

    /// Enqueue the next tick and return the OLDEST in-flight tick's ids, read
    /// via the copy stream while the new tick executes.
    pub(crate) fn decode_pipe_next_b(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self
                .pipe_b
                .as_ref()
                .ok_or_else(|| GpuModelError::Config("decode_pipe_next without begin".into()))?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe_b.as_mut().expect("pipe active").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick_b(plans, true) {
            self.pipe_b_abort();
            return Err(e);
        }
        let ring = j % 2;
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        let r = {
            let sc = &self.batch.as_ref().expect("batch enabled").sc;
            let ev = self.pipe_b.as_ref().expect("pipe active").ev[ring]
                .as_ref()
                .expect("in-flight event");
            exec.to_host_u32_after(ev, &sc.d_pipe_out, ring * n_slots, b)
        };
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_b_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids. The fixed input
    /// buffers are stale after this - every other path re-uploads them.
    pub(crate) fn decode_pipe_drain_b(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe_b
            .take()
            .ok_or_else(|| GpuModelError::Config("decode_pipe_drain without begin".into()))?;
        let ring = st.tick % 2;
        let n_slots = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .n_slots;
        let ev = st.ev[ring].as_ref().expect("in-flight event");
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        match exec.to_host_u32_after(ev, &sc.d_pipe_out, ring * n_slots, st.b) {
            Ok(ids) => Ok(ids),
            Err(e) => {
                let _ = exec.synchronize(); // state gone - quiesce ring readers
                Err(e.into())
            }
        }
    }

    /// Kill an in-flight batch pipe (error/reset/re-enable): quiesce so
    /// nothing still reads the rings, then drop the state.
    pub(crate) fn pipe_b_abort(&mut self) {
        if self.pipe_b.take().is_some() {
            let _ = self.exec.synchronize();
        }
    }

    // ── stage-B gate probes (tests/gpu_nemotron_batch.rs) ──────────────────

    #[doc(hidden)]
    pub fn batch_enable_probe(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        self.enable_batch_impl(max_batch)
    }

    #[doc(hidden)]
    pub fn batch_admit_probe(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        self.admit_rows(slot, n_rows)
    }

    /// Diagnostic: host copies of one slot's per-mamba-layer SSM state and
    /// conv window (layer index, state, window). Test-only introspection.
    #[doc(hidden)]
    pub fn state_dump_probe(&mut self, slot: usize) -> Vec<(usize, Vec<f32>, Vec<f32>)> {
        let hp = self.hp.clone();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();
        let bs = self.batch.as_ref().expect("batch enabled");
        let mut out = Vec::new();
        for li in 0..hp.n_layer {
            let Some(s) = bs.ssm[li].as_ref() else {
                continue;
            };
            let w = bs.conv_win[li].as_ref().expect("mamba layer has window");
            let sh = s
                .dump_slot(&self.exec, slot * state_elems, state_elems)
                .expect("state dump");
            let wv = w
                .try_slice(slot * win_elems..(slot + 1) * win_elems)
                .expect("win view");

            let wh = self.exec.stream.clone_dtoh(&wv).expect("win dtoh");
            out.push((li, sh, wh));
        }
        out
    }

    /// (free blocks, pool capacity) - None until enable succeeded.
    #[doc(hidden)]
    pub fn batch_pool_stats(&self) -> Option<(usize, usize)> {
        self.batch
            .as_ref()
            .map(|b| (b.pool.free_blocks(), b.pool.capacity() as usize))
    }
}
