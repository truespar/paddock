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
//! Scratch is sized once at enable for `cap = prefill_chunk + n_slots` rows
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

use super::*;
use crate::gpu_model::qwen35::{mmq_pre_any, prefill_mm_pre_any, prefill_quant};
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
    /// each run's first position (a run's rows are consecutive positions)
    pub(super) run_pos: Vec<u32>,
    pub(super) dec: usize,
    pub(super) breaks: Vec<(usize, usize)>,
}

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
pub(super) const VRAM_HEADROOM: usize = 1 << 30;

/// Mamba state checkpoints the prefix cache wants per seated slot (see the
/// plan in `enable_batch_sized` for how 8 was measured; the per-turn floor
/// and caps every hybrid shares are `ckpt_pages::page_demand`'s).
const CKPTS_PER_SLOT: u64 = 8;

/// FlashDecoding split ceiling for the per-q-head decode walk (the arm a
/// geometry off the head-packed election rides). Nemotron is 32q/2kv hd128 -
/// that q-head grid is already wide, so granite's fused-walk cap is plenty.
pub(crate) const MAX_ATTN_SPLITS: usize = 16;

/// Split election for the batched decode attention at `r` rows: (head-packed
/// arm?, splits per row). Position-independent, so the per-r decode graph
/// bakes it, and the partial scratch is sized from the same function
/// ([`attn_partial_rows`]) - the budget and the plane can't drift apart.
///
/// Head-packed arm: hd128 at G=16, either KV class (no window) - the pack
/// elects the lagd WIDE partial, grid (n_kv, r, ns), one CTA per KV head
/// streaming the group's KV once. f16 joined on GB10 2026-09-25: it rode the
/// per-(q-head, split) walk, 16 heads each re-reading the group's KV, at ~24
/// GB/s - 11.2 ms a layer at 256K, the whole of the decode-vs-depth cliff
/// (67.9 -> 12.5 tok/s from 10K to 259K). The same kernel on f16 streams at
/// ~193 GB/s (1.38 ms).
///
/// Budget: ONE live CTA per SM (T = n_kv * r * ns ~= sm). The lagd partial
/// is tile-rate bound per CTA (~5 GB/s fp8, ~10 f16 at 47 KB smem), so the
/// stream saturates at ~185-195 GB/s once the die is covered: measured
/// (32q/2kv, r 1..8, ctx 2K..256K) fp8 is best at T 48-64 and f16 flat from
/// T 32, with T 96+ a few % worse (combine + DRAM contention). The old
/// `3*sm/(n_kv*r)` capped at 8 gave T = 16 at one row - a third of the die,
/// fp8 at 81 GB/s (256K: 1647 -> 700 us a layer at T 48). ns may be 1: the
/// head-packed arm then still runs partial + combine, because the unsplit
/// decode kernel is the per-q-head walk again.
/// `PADDOCK_NO_ATTN_HP16` (the pack's own kill) returns the walk's budget.
pub(crate) fn attn_split_election(
    nh: usize,
    n_kv: usize,
    hd: usize,
    sm: usize,
    r: usize,
) -> (bool, usize) {
    let hp = hd == 128
        && n_kv > 0
        && nh == n_kv * 16
        && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_HP16").is_none();
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return (hp, 1);
    }
    let r = r.max(1);
    if hp {
        (true, sm.div_ceil(n_kv * r).max(1))
    } else {
        (
            false,
            (2 * 3 * sm).div_ceil(nh * r).clamp(1, MAX_ATTN_SPLITS),
        )
    }
}

/// Rows per group of the multi-row split partial (`attn_rows_partial`): one
/// warp a row, eight warps a CTA.
pub(crate) const ROWS_GROUP: usize = 8;

/// Rows per group under slot 745 (two warps a row, twelve a CTA): the W16
/// class caps its verify groups here whenever it attends through 745.
pub(crate) const ROWS_GROUP_KH: usize = 6;

/// Split the rows of each same-slot run `(first row, rows)` into the
/// multi-row partial's groups - flat `[first row, rows <= ROWS_GROUP]` pairs.
pub(crate) fn rows_groups(runs: impl IntoIterator<Item = (usize, usize)>) -> Vec<u32> {
    rows_groups_cap(runs, ROWS_GROUP)
}

/// [`rows_groups`] at `cap` rows a group.
pub(crate) fn rows_groups_cap(
    runs: impl IntoIterator<Item = (usize, usize)>,
    cap: usize,
) -> Vec<u32> {
    let mut g = Vec::new();
    for (off, len) in runs {
        let mut o = off;
        while o < off + len {
            let n = (off + len - o).min(cap);
            g.extend([o as u32, n as u32]);
            o += n;
        }
    }
    g
}

/// The split size a row with `n` keys attends by under a W16 law word: 0 or
/// `0x8000_0000 | budget` = slot 744's pow2 law (0 takes `ns` as the
/// budget), `0x4000_0000 | budget` = slot 745's TILE law, anything else a
/// fixed size. Mirrors the kernels' own arithmetic - the host groups rows by
/// it, so it must stay exact.
pub(crate) fn w16_split_size(law: usize, ns: usize, n: usize) -> usize {
    if law == 0 || law & 0x8000_0000 != 0 {
        let budget = if law == 0 { ns } else { law & 0x7fff_ffff };
        n.div_ceil(budget.max(1)).next_power_of_two().max(256)
    } else if law & 0x4000_0000 != 0 {
        let budget = (law & 0x3fff_ffff).max(1);
        (64 * n.div_ceil(64 * budget)).max(256)
    } else {
        law
    }
}

/// [`rows_groups`] under a key-dependent split law (744's pow2, 745's TILE):
/// a group's rows must share one split size, and the size follows the key
/// count (n = pos + 1), so a run that crosses a size-bucket edge splits
/// there too. `runs` carry each run's first position; `cap` rows a group.
pub(crate) fn rows_groups_law(
    runs: impl IntoIterator<Item = (usize, usize, u32)>,
    z: impl Fn(usize) -> usize,
    cap: usize,
) -> Vec<u32> {
    let mut pieces = Vec::new();
    for (off, len, p0) in runs {
        let p0 = p0 as usize;
        let mut a = 0;
        for j in 1..=len {
            if j == len || z(p0 + j + 1) != z(p0 + a + 1) {
                pieces.push((off + a, j - a));
                a = j;
            }
        }
    }
    rows_groups_cap(pieces, cap)
}

/// Slot 745's TILE-law budget: whole waves of one CTA an SM over the kv
/// heads - the largest multiple of sm / n_kv splits the `ns` planes hold
/// (GB10, 2 kv heads: 120 splits = 240 CTAs = five waves).
pub(crate) fn kh_tile_budget(sm: usize, n_kv: usize, ns: usize) -> usize {
    let per = (sm / n_kv.max(1)).max(1);
    if per >= ns { ns } else { ns / per * per }
}

/// Splits for the multi-row partial: one live CTA per SM over the (kv head,
/// group) pairs - the head-packed decode law (`attn_split_election`), which
/// the same measurement motivates: the stream saturates at one live CTA per
/// SM, and every group reads its context once whatever its row count.
pub(crate) fn rows_split(n_kv: usize, n_groups: usize, sm: usize) -> usize {
    sm.div_ceil(n_kv.max(1) * n_groups.max(1)).max(1)
}

/// Partial-plane rows (rows x splits) a multi-row round of up to `max_rows`
/// rows can address, worst over how its rows fall into groups.
pub(crate) fn rows_partial_cap(n_kv: usize, sm: usize, max_rows: usize) -> usize {
    (1..=max_rows.max(1))
        .map(|g| (ROWS_GROUP * g).min(max_rows) * rows_split(n_kv, g, sm))
        .max()
        .unwrap_or(1)
}

/// Partial-plane rows (q-head-major rows x splits) the decode attention can
/// address over every tick width 1..=`max_rows`: max of r * ns(r).
pub(crate) fn attn_partial_rows(
    nh: usize,
    n_kv: usize,
    hd: usize,
    sm: usize,
    max_rows: usize,
) -> usize {
    (1..=max_rows.max(1))
        .map(|r| r * attn_split_election(nh, n_kv, hd, sm, r).1)
        .max()
        .unwrap_or(1)
}

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
    /// the W16 decode class's 16-bit planes (W16_ROWS rows): activations cast
    /// once for the experts and wide attention calls (bf16), for the wide FP8
    /// projections (f16), and the experts' up output (bf16)
    pub d_x16b: CudaSlice<half::bf16>,
    pub d_x16h: CudaSlice<half::f16>,
    pub d_act16: CudaSlice<half::bf16>,
    /// the routing front's hand-off tickets (zeroed once; the kernel leaves
    /// them zero)
    pub d_route_tickets: CudaSlice<u32>,
    /// a mixed tick's decode band: its residual rows, held while the
    /// chunk's MoE folds over every row (W16_ROWS rows)
    pub d_band_x: CudaSlice<f32>,
    /// the W16 class's attention: fixed-split partial planes for up to
    /// `w16_rows` rows x `w16_ns` splits, the law word (`w16_split_size`:
    /// a fixed size, 0 = 744's pow2 law, 0x4000_0000 | budget = 745's TILE
    /// law), and the one-row-a-group list a decode tick attends through
    pub w16_split: usize,
    pub w16_ns: usize,
    pub w16_rows: usize,
    /// the class attends through slot 745 (two warps a row): decode ticks
    /// and verify rounds alike, verify groups capped at ROWS_GROUP_KH
    pub w16_kh: bool,
    pub d_w16o: CudaSlice<f32>,
    pub d_w16ml: CudaSlice<f32>,
    pub d_w16_groups: CudaSlice<u32>,
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
    /// FlashDecoding partial scratch [n_heads, attn_partial_rows(n_slots), hd]
    /// (rows x splits of the widest tick the split election hands out)
    pub attn_o: CudaSlice<f32>,
    /// per-partial (m, l) [n_heads, attn_partial_rows(n_slots), 2]
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
    /// Row capacity of every scratch plane = prefill_chunk + one row per slot.
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
    /// Where mamba state checkpoints live: in the attention pool's own pages
    /// (issue #33). `PagedRadix` owns each checkpoint's page list; this is
    /// the plane geometry that turns a checkpoint byte into a device address.
    /// `None` when the prefix cache is off.
    pub ckpt_layout: Option<crate::ckpt_pages::PageLayout>,
    /// One checkpoint's flat f32 blob, which every live-state snapshot and
    /// restore passes through: the SSM arena may be f16 and widens/narrows
    /// into a contiguous f32 span, which pages are not.
    pub d_ckpt_bounce: Option<CudaSlice<f32>>,
    /// Descriptor scratch for the page-split copies (see
    /// `GpuExecutor::batched_copy_upload`).
    pub d_ckpt_desc: Option<CudaSlice<u64>>,
    /// f32 elements per checkpoint (all mamba layers' state+window)
    pub state_ckpt_f32: usize,
    /// per-pass staging blobs the layer walk fills at break rows
    pub d_ckpt_stage: Vec<CudaSlice<f32>>,
    /// spec verify planes  - lazily allocated at first spec use
    pub verify: Option<super::spec::VerifyPlanes>,
    /// Stage F: per-slot tracked sequence (prompt keys, then every fed
    /// decode token); empty = not tracking.
    pub seq: Vec<Vec<u32>>,
    /// Stage F: the slot's live reply checkpoint (cut, pool index).
    pub reply_ckpt: Vec<Option<(usize, u32)>>,
    /// Stage F: the reply checkpoint held at the reply's first tool call
    /// (`reply_pin`); the live one moves on past the call.
    pub reply_pinned: Vec<Option<(usize, u32)>>,
    /// Stage F: snapshots copied on the device whose ids the host has not
    /// seen yet (slot, cut, pool index).
    pub reply_pending: Vec<(usize, usize, u32)>,
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

pub(super) fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// Blocks `moe_align` can actually fill at this row count - the launch extent
/// for the sorted-tile MoE pair and, crucially, for the intermediate
/// `quantize_q8` between them.
///
/// `bs.nb_r` / `bs.nb_s` are ARENA capacities, sized once for the widest row
/// stream a tick can carry (`cap = prefill_chunk + max_batch` - 520 rows on a
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
pub(super) fn moe_live_blocks_bm8(rows: usize, picks: usize, experts: usize, cap: usize) -> usize {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_MOE_NBLIVE").is_some()) {
        return cap;
    }
    let pairs = rows.saturating_mul(picks);
    (experts.min(pairs) + pairs / 8).min(cap)
}

pub(super) fn moe_live_blocks(rows: usize, picks: usize, experts: usize, cap: usize) -> usize {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_MOE_NBLIVE").is_some()) {
        return cap;
    }
    // sum_e ceil(c_e / 32) <= sum_e (c_e / 32 + 1) = pairs / 32 + distinct,
    // and distinct <= min(experts, pairs): the same distribution-free bound
    // the BM=8 twin uses. The old `distinct * ceil(rows/32)` is exact only
    // at rows <= 32 and grows with rows x experts (4224 at 1036 rows), so a
    // wide scratch cap turned it into launches of mostly PAD CTAs (GB10
    // 2026-09-11: +9 ms on the 1k request at an 8192-row cap).
    let pairs = rows.saturating_mul(picks);
    let tight = experts.min(pairs) + pairs / 32;
    let old = experts.min(pairs) * rows.div_ceil(32);
    tight.min(old).min(cap)
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
pub(super) const MOE_DEC2_MAX_ROWS: usize = 64;

/// Whether this pack carries the decode-band MoE route. Not part of the
/// family's capability gate: a pack without it serves fine on the sorted
/// tile, just slower, and folding it into the gate is the over-broad-bundle
/// shape is auditing for.
/// Lever 14: the BM=8 shared-expert fold on the skinny decode path
/// (`PADDOCK_NO_NEMO_SH_FOLD8=1` keeps the separate wide pair).
/// Rows that stand in for decode steps - one-row decode, a decode tick's
/// rows, a spec verify round's - run the W16 decode class: the checkpoint's
/// W4A16 experts with bf16 activations (moe/nvf4_w16), the FP8 mamba
/// projections widened to f16 against f16 activations and the bf16
/// attention planes against bf16 ones (gemm/dense_w16), the head's
/// tensor-core tile - all on tensor cores, all batch-invariant: a row's bits
/// do not depend on what shares its tick. The lanes they replace changed
/// class with the row count (W4A16 at f32 activations for one row, W4A4 /
/// dynamic-e4m3 W8A8 / bf16-cast past it), so a verify row sat up to 1.5
/// logits from the one-row tick it stands for and the DFlash spec loop parted
/// from greedy (GB10 2026-09-26). Prefill keeps the block-scaled lanes; past
/// this many rows a tick does too (the class's scratch is sized for it).
pub(crate) const W16_ROWS: usize = 64;

/// The W16 class's attention splits: fixed key ranges of `split` keys (a
/// multiple of the rows kernel's 32-key tile), `n` of them covering the
/// context, at most 128 - the partial planes scale with it, so a long
/// context limit coarsens the split rather than growing the planes. A row's
/// splits depend on its own keys alone (see attn_rows_partial_fixed).
pub(crate) fn w16_attn_splits(max_ctx: usize) -> (usize, usize) {
    let split = max_ctx.div_ceil(128).next_multiple_of(32).max(256);
    (split, max_ctx.div_ceil(split))
}

/// The W16 decode class serves `rows` rows: the pack carries its kernels and
/// `PADDOCK_NO_NEMO_W16` does not pin the old lanes back (dev: the A/B).
pub(crate) fn w16_class(exec: &GpuExecutor, rows: usize) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (1..=W16_ROWS).contains(&rows)
        && exec.has_nvf4_moe_w16()
        && exec.has_dense_w16()
        && exec.has_nvf4_gemm_tc()
        && exec.has_attn_rows_partial_fixed()
        && !*OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NEMO_W16").is_some())
}

pub(super) fn sh_fold8_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NEMO_SH_FOLD8").is_none())
}

pub(super) fn moe_dec2_ok(exec: &GpuExecutor) -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_NEMO_MOEDEC2").is_none()
            && exec.has_q8_0_moe_relu2_dec2()
            && exec.has_quantize_q8_relu2()
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dense_mm_pre(
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
        // block costs all n_attn layers' K+V at once - and the drafter's
        // stripes, which ride the same ids (dflash.rs, mtp.rs).
        let block_bytes = n_attn * 16 * kv_dim * 2 * kvb
            + self.dflash.as_ref().map_or(0, |d| d.stripe_bytes(kv_dim))
            + self.mtp_stripe_bytes(kv_dim);
        // per-slot recurrent state (flat, not paged)
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * conv_dim;
        let ssm_dt = self.ssm_dtype;
        let arena_bytes = max_batch * n_mamba * (state_elems * ssm_dt.bytes() + win_elems * 4);

        let cap = self.prefill_chunk + max_batch;
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
            + nh * attn_partial_rows(nh, n_kv, hd, self.exec.sm_count(), max_batch) * (hd + 2) * 4
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
        // Mamba state checkpoints for the prefix cache, sized by DEMAND (the
        // qwen35 rule, 2026-09-06): eight per requested slot, clamped 16..256.
        // It used to be computed after the plan as a fifth of whatever was
        // still free, clamped to 256, with no reserve: 256 f32 snapshots
        // (~6 GiB) for a one-slot server on a 96 GB card. Two per slot (16 at
        // 8 slots) was the working set of one wave: an agentic session keeps
        // two live cuts and its previous turn's two until the next commit, so
        // eight sessions cycled the pool every turn and resumed from the
        // shared system prompt instead of their own boundary (GB10
        // 2026-09-11; 64 held every session's cut).
        //
        // Since issue #33 they live in the attention pool's own pages: the
        // live turns' checkpoints are backed beside full context, the rest of
        // the want rides retention (bought only while the grant affords it),
        // and pages a context has not reached yet hold more for free. The
        // staging blobs (one pass's cuts, plus the bounce blob every live
        // snapshot and restore goes through) stay a reserve.
        let state_ckpt_f32 = n_mamba * (state_elems + win_elems);
        let per_ckpt = (state_ckpt_f32 * 4) as u64;
        let n_stages = if px_on {
            super::prefix::ckpt_stages(max_batch)
        } else {
            0
        };
        let bounce = usize::from(px_on && per_ckpt > 0);
        let staging_bytes = (n_stages + bounce) as u64 * per_ckpt;
        let tier_staging: u64 = if crate::kv_tier::pool_tier::tier_ram_bytes().is_some() {
            crate::kv_tier::ram_transport::device_staging_bytes()
        } else {
            0
        };
        // the Mamba arenas: one recurrent state + conv window per slot per
        // mamba layer (`arena_bytes` was max_batch x this)
        let per_slot_bytes = (n_mamba * (state_elems + win_elems) * 4) as u64;
        // a page's payload: its slot in every attention layer's K and V plane
        // (the drafter's stripes ride the same ids but are not part of it)
        let page_payload = (n_attn * 16 * kv_dim * 2 * kvb) as u64;
        let pages_per_ckpt = if px_on && per_ckpt > 0 && page_payload > 0 {
            per_ckpt.div_ceil(page_payload) as usize
        } else {
            0
        };
        let (ckpt_blocks, ckpt_retention) =
            crate::ckpt_pages::page_demand(max_batch, pages_per_ckpt, CKPTS_PER_SLOT);
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
            retention_blocks: retain + ckpt_retention,
            ckpt_blocks,
            // every slot must at least hold a base tick's worth of prompt, or
            // admission deadlocks on its own first chunk. The BASE tick, not
            // the scratch cap: with the cap at 8192 rows on small dies the
            // cap-derived floor (512 blocks x 8 slots) was the whole pool
            // and the prefix cache stopped retaining (GB10 2026-09-11 -
            // serving logprobs moved, PPL did not; the resumes had vanished)
            floor_blocks_per_slot: self
                .prefill_chunk
                .min(super::forward::PREFILL_TICK_BASE)
                .div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                kv_plan::Reserve::new("prefill scratch", scratch_est as u64),
                kv_plan::Reserve::new("prefix state staging", staging_bytes),
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
        let mut kv_bytes = arena_bytes as u64
            + (pool_blocks
                * (self.dflash.as_ref().map_or(0, |d| d.stripe_bytes(kv_dim))
                    + self.mtp_stripe_bytes(kv_dim))) as u64;
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

        // the W16 class's attention: a decode tick's rows or a verify round's
        let (mut w16_split, w16_ns) = w16_attn_splits(self.max_ctx);
        // slot 744: the split size follows each row's keys instead of
        // max_ctx (0 = that law; w16_ns stays the budget and the planes)
        if e.has_attn_rows_partial_pow2()
            && paddock_models::dev_var_os!("PADDOCK_NO_W16_POW2").is_none()
        {
            w16_split = 0;
        }
        // slot 745 when the pack carries it and the pool is e4m3 (the class
        // must not mix kernels: decode == verify rides on one fold)
        let w16_kh = e.has_attn_rows_partial_kh()
            && self.kv_dtype == KvDtype::Fp8E4m3
            && paddock_models::dev_var_os!("PADDOCK_NO_ROWS_KH").is_none();
        if w16_kh {
            // 745 attends by its TILE law: whole waves at every depth
            w16_split = 0x4000_0000 | kh_tile_budget(e.sm_count(), hp.n_kv_heads, w16_ns);
        }
        let w16_rows = slots.clamp(super::spec::SPEC_ROWS_NEMO, W16_ROWS).min(cap);
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
            d_x16b: e.stream_alloc_bf16(W16_ROWS.min(cap) * embd.max(q_dim))?,
            d_x16h: e.alloc_f16(W16_ROWS.min(cap) * embd.max(d_inner))?,
            d_act16: e
                .stream_alloc_bf16(W16_ROWS.min(cap) * (hp.n_active * hp.moe_ff + hp.shared_ff))?,
            d_route_tickets: e.alloc_u32(GpuExecutor::moe_route_w16_tickets(W16_ROWS.min(cap)))?,
            d_band_x: e.alloc(W16_ROWS.min(cap) * embd)?,
            w16_split,
            w16_ns,
            w16_rows,
            w16_kh,
            d_w16o: e.alloc(nh * w16_rows * w16_ns * hd)?,
            d_w16ml: e.alloc(nh * w16_rows * w16_ns * 2)?,
            d_w16_groups: e.to_device_u32(
                &(0..w16_rows as u32)
                    .flat_map(|i| [i, 1u32])
                    .collect::<Vec<_>>(),
            )?,
            d_part7: e.alloc((hp.n_active + 1) * embd)?,
            head_logits: e.alloc(slots * hp.vocab)?,
            d_par: e.alloc_u32(slots * 4)?,
            d_out: e.alloc_u32(slots)?,
            d_tpar: e.alloc_u32(slots * 4)?,
            d_pipe_par: e.alloc_u32(2 * slots * 4)?,
            d_pipe_tpar: e.alloc_u32(2 * slots * 4)?,
            d_pipe_out: e.alloc_u32(2 * slots)?,
            attn_o: e.alloc(nh * attn_partial_rows(nh, n_kv, hd, e.sm_count(), slots) * hd)?,
            attn_ml: e.alloc(nh * attn_partial_rows(nh, n_kv, hd, e.sm_count(), slots) * 2)?,
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

        // Stage D: the radix + mamba state checkpoints. Each checkpoint is a
        // full 23-layer state snapshot (~48 MB on this geometry), held in the
        // attention pool's own pages (issue #33) - the plan above backed the
        // live turns' pages beside full context.
        let (prefix, ckpt_layout, d_ckpt_stage, d_ckpt_bounce, d_ckpt_desc) =
            if px_on && pages_per_ckpt > 0 {
                use cudarc::driver::DevicePtr;
                let mut pr = crate::paged_radix::PagedRadix::new();
                // the K and V plane of each attention layer, in layer order - the
                // KV tier's plane order too, so a checkpoint's pages ship to RAM
                // as they are
                let slot = (16 * kv_dim * kvb) as u64;
                let mut planes = Vec::with_capacity(2 * n_attn);
                for l in kv.iter().flatten() {
                    for plane in [&l.k, &l.v] {
                        let (pp, _g) = plane.device_ptr(&self.exec.stream);
                        planes.push((pp, slot));
                    }
                }
                let layout = crate::ckpt_pages::PageLayout::new(planes);
                debug_assert_eq!(layout.payload(), page_payload);
                // index bookkeeping only: as many as the pool could ever hold
                let max_ckpts = pool_blocks.checked_div(pages_per_ckpt).unwrap_or(0).max(1) as u32;
                pr.set_state_paged(max_ckpts, pages_per_ckpt);
                let stages = (0..n_stages)
                    .map(|_| self.exec.alloc(state_ckpt_f32))
                    .collect::<Result<Vec<_>, _>>()?;
                let desc = self
                    .exec
                    .alloc_u64(crate::ckpt_pages::desc_cap(per_ckpt, slot, 1))?;
                tracing::info!(
                    pages_per_checkpoint = pages_per_ckpt,
                    guaranteed = ckpt_blocks / pages_per_ckpt,
                    wanted = (ckpt_blocks + ckpt_retention) / pages_per_ckpt,
                    "nemotron prefix cache active - checkpoints in pool pages"
                );
                (
                    Some(pr),
                    Some(layout),
                    stages,
                    Some(self.exec.alloc(state_ckpt_f32)?),
                    Some(desc),
                )
            } else {
                (None, None, Vec::new(), None, None)
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
                        // v2: checkpoint blobs are pool pages (issue #33) - a
                        // v1 cache's flat blobs must never load into them
                        cache_schema: b"pool-planes k/v interleaved + mamba-ckpt paged aux v2",
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
            ckpt_layout,
            d_ckpt_bounce,
            d_ckpt_desc,
            state_ckpt_f32,
            d_ckpt_stage,
            verify: None,
            seq: vec![Vec::new(); slots],
            reply_ckpt: vec![None; slots],
            reply_pinned: vec![None; slots],
            reply_pending: Vec::new(),
        });
        self.last_reused = vec![0; slots];
        self.dflash_ensure_state()?;
        self.mtp_ensure_state()?;
        tracing::info!(
            "nemotron batch: {slots} slots, {n_attn}-attn-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} tokens), mamba arenas {:.2} GiB, checkpoints in pool pages, \
             {} rows/chunk; left {:.2} GiB of the {:.2} GiB granted",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            arena_bytes as f64 / (1u64 << 30) as f64,
            self.prefill_chunk,
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
                                // the checkpoints are pool pages, which the
                                // tier reads off the radix
                                let exec = self.exec.clone();
                                let want = bs.pool.free_blocks() + 1;
                                tier.make_room_blocking(pr, &mut bs.pool, want, None, &mut || {
                                    exec.record_event().ok()
                                })
                            }
                            // dead KV, then the stalest checkpoint's pages,
                            // then LRU KV - promised context outranks cache
                            (None, Some(pr)) => {
                                let want = bs.pool.free_blocks() + 1;
                                pr.make_room(&mut bs.pool, want, 0)
                            }
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
    /// (blocks AND recurrent state) and zero the slot's mamba arenas - the
    /// recurrent twin of "fresh sequence: old pool blocks return first".
    /// Stale state here is not a crash, it's silent cross-request
    /// contamination.
    ///
    /// It backs NO rows: `prefix_resume_rows` runs next, adopts whatever the
    /// radix holds of the prompt, and only then backs the rows the prompt
    /// still writes - all of them, up front, so a mid-prompt chunk never finds
    /// the pool dry with rows written. Backing the whole prompt here first
    /// made the allocation evict the very pages the match was about to adopt
    /// whenever the conversation outgrew the free part of the pool: a 262K
    /// single-slot serve (270K-token pool) holding a ~210K Claude Code
    /// conversation re-prefilled all of it every turn, resuming at the 1856-
    /// token system-prompt checkpoint (GB10, 2026-09-25).
    pub(super) fn admit_rows(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        // the drafter span restarts with the pages; prefix_resume_rows
        // re-derives it from whatever pages the radix hands back
        self.dflash_reset_slot(slot);
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
        Ok(())
    }

    /// Free-on-completion: an idle slot's blocks return to the shared pool
    /// immediately. The mamba arenas need no action here - admission zeroes
    /// them, and they hold no pool capacity.
    pub(crate) fn release_inactive_slots_impl(&mut self, occupied: &[bool]) {
        for (s, &occ) in occupied.iter().enumerate() {
            if !occ {
                self.reply_release(s);
                // the drafter's rows ride the pages: whatever of them the
                // radix keeps is re-adopted with the pages (dflash_adopt_slot)
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
        let note: Vec<(usize, usize, usize, usize)> = {
            // per-slot covered spans (slot, first pos, end pos, first chunk
            // row) for the drafters' coverage bookkeeping
            let mut spans: Vec<(usize, usize, usize, usize)> = Vec::new();
            for (i, x) in chunk.iter().enumerate() {
                let (s, p) = (x.0 as usize, x.1 as usize);
                match spans.last_mut() {
                    Some((ls, _, le, _)) if *ls == s && *le == p => *le += 1,
                    _ => spans.push((s, p, p + 1, i)),
                }
            }
            spans
        };
        let run_pos = runs.iter().map(|&(off, _, _)| positions[off]).collect();
        self.layer_walk(
            chunk.len(),
            Some(&PfCuts {
                runs,
                run_pos,
                dec,
                breaks,
            }),
            false,
        )?;
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) {
            self.dflash_append_features(chunk.len())?;
            for &(s, a, b, _) in &note {
                self.dflash_note_rows(s, a, b - a);
            }
        }
        if self.mtp.as_ref().is_some_and(|m| m.state.is_some()) {
            // the MTP block advances over the same spans; every span is a
            // plain walk commit, so coverage and the h chain advance to the
            // span end (verify rounds instead advance only accepted rows -
            // spec.rs owns that call site)
            let mut mruns = Vec::with_capacity(note.len());
            let mut off = 0usize;
            for &(s, a, b, _) in &note {
                mruns.push((s, off, b - a));
                off += b - a;
            }
            self.mtp_append_rows(&mruns)?;
            let mut off = 0usize;
            for &(s, a, b, _) in &note {
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

    /// Final norm + lm_head over rows 0..rows, leaving [rows, vocab] in
    /// head_logits (row-batched GEMV - bit-exact per row vs the serial head).
    pub(super) fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.hidden, self.hp.eps);
        let final_norm = self.final_norm.buf.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(&sc.d_x, &final_norm, &mut sc.d_xn, embd, eps, rows)?;
        match &self.lm_head {
            HeadW::Nvf4(h) => super::head_nvf4_batch(
                &exec,
                h,
                &sc.d_xn,
                &mut sc.head_logits,
                rows,
                w16_class(&exec, rows),
            )?,
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
    pub(super) fn head_row_at(&mut self, row: usize) -> Result<(), GpuModelError> {
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
    pub(super) fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
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
    pub(super) fn capture_body(
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
    pub(super) fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        // the serial path's eager pin covers the batch ticks too
        if paddock_models::dev_var_os!("PADDOCK_NO_NEMO_GRAPH").is_some() {
            return self.step_body(r);
        }
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
        for i in 0..r {
            self.reply_feed(slots[i] as usize, positions[i], tokens[i]);
        }
        self.step_replay(r)?;
        self.reply_after_rows(slots, positions)?;
        self.dflash_note_ticks(slots, positions);
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

    // ── device sampling ────────────────────────────────────────────────────

    // ── batched depth-2 decode pipe (stage E, granite's pipe-under-pool) ───

    // ── stage-B gate probes (tests/gpu_nemotron_batch.rs) ──────────────────

    #[doc(hidden)]
    pub fn batch_enable_probe(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        self.enable_batch_impl(max_batch)
    }

    #[doc(hidden)]
    pub fn batch_admit_probe(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        self.admit_rows(slot, n_rows)?;
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    /// Whether `rows` decode-class rows ride the W16 class on this pack -
    /// the gates assert bit-exactness only where the class serves.
    #[doc(hidden)]
    pub fn w16_class_probe(&self, rows: usize) -> bool {
        self.w16_lane() && w16_class(&self.exec, rows)
    }

    /// The model is on the lane whose kernels define the W16 class: the
    /// NVFP4 checkpoint (NVFP4 experts and head, the FP8 / bf16 dense
    /// planes). The GGUF lane's planes are Q8_0, which the class's kernels do
    /// not read - elected there anyway (the class was keyed on the pack
    /// alone), its decode ticks and verify rounds took the class's attention
    /// law while every projection stayed Q8_0, and the in-file MTP drafts
    /// stopped landing: gpu_nemotron_gguf's mtp_drafts_accept accepted 0 of
    /// 7 drafts and spec_serve_cadence left the greedy stream (GB10,
    /// 2026-09-27, bisected to the class's first commit). The GGUF lane keeps
    /// its own decode class.
    pub(super) fn w16_lane(&self) -> bool {
        matches!(self.lm_head, HeadW::Nvf4(_))
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

    /// Diagnostic: `slot`'s live reply checkpoint as (cut, its pool blob -
    /// per mamba layer the f32 state, then the conv window). Test-only.
    #[doc(hidden)]
    pub fn reply_ckpt_probe(&mut self, slot: usize) -> Option<(usize, Vec<f32>)> {
        let (cut, idx) = (*self.batch.as_ref()?.reply_ckpt.get(slot)?)?;
        // gather the checkpoint's pages into the bounce blob, read that
        self.ckpt_pages_copy(idx, crate::ckpt_pages::Dir::FromPages)
            .ok()?;
        let bs = self.batch.as_ref()?;
        let v = bs.d_ckpt_bounce.as_ref()?;
        Some((cut, self.exec.stream.clone_dtoh(v).ok()?))
    }

    /// (pages one checkpoint draws, checkpoint pages the plan bought beside
    /// full context and retention) - the pool's size over what its slots
    /// address, by design since checkpoints live in its pages (issue #33).
    /// None before `enable_batch` or with the prefix cache off.
    #[doc(hidden)]
    pub fn batch_ckpt_plan_probe(&self) -> Option<(usize, usize)> {
        let bs = self.batch.as_ref()?;
        let ppc = bs.prefix.as_ref()?.pages_per_ckpt();
        let (must, want) = crate::ckpt_pages::page_demand(bs.n_slots, ppc, CKPTS_PER_SLOT);
        Some((ppc, must + want))
    }

    /// (free blocks, pool capacity) - None until enable succeeded.
    #[doc(hidden)]
    pub fn batch_pool_stats(&self) -> Option<(usize, usize)> {
        self.batch
            .as_ref()
            .map(|b| (b.pool.free_blocks(), b.pool.capacity() as usize))
    }
}

#[cfg(test)]
mod law_tests {
    use super::*;

    #[test]
    fn split_size_mirrors_the_kernel_laws() {
        // 744's pow2 law at the 128-split budget (0 and the flagged word agree)
        assert_eq!(w16_split_size(0, 128, 10_431), 256);
        assert_eq!(w16_split_size(0, 128, 83_027), 1024);
        assert_eq!(w16_split_size(0x8000_0000 | 128, 0, 233_332), 2048);
        // 745's TILE law at B = 120: whole 64-key tiles, ~B splits
        let tile = 0x4000_0000 | 120;
        assert_eq!(w16_split_size(tile, 128, 10_431), 256);
        assert_eq!(w16_split_size(tile, 128, 83_027), 704);
        assert_eq!(w16_split_size(tile, 128, 233_332), 1984);
        for n in [1usize, 255, 7_680, 7_681, 83_027, 262_144] {
            let z = w16_split_size(tile, 128, n);
            assert!(
                z.is_multiple_of(64) && z >= 256 && n.div_ceil(z) <= 120,
                "n {n}: z {z}"
            );
        }
        // a fixed size is itself
        assert_eq!(w16_split_size(2048, 128, 5), 2048);
    }

    #[test]
    fn tile_budget_fills_whole_waves() {
        assert_eq!(kh_tile_budget(48, 2, 128), 120);
        assert_eq!(kh_tile_budget(170, 2, 128), 85);
        assert_eq!(kh_tile_budget(300, 2, 128), 128);
    }

    #[test]
    fn law_groups_split_at_a_size_edge_and_at_the_cap() {
        let tile = 0x4000_0000 | 120;
        let z = |n| w16_split_size(tile, 128, n);
        // 6 rows from position 30716: n = 30717..30722 crosses the TILE
        // law's first step past the 256 floor (z 256 -> 320 at n 30721)
        assert_eq!((z(30_720), z(30_721)), (256, 320));
        let g = rows_groups_law([(0usize, 6usize, 30_716u32)], z, 6);
        assert_eq!(g, vec![0, 4, 4, 2]);
        assert!(g.chunks(2).all(|c| {
            let (o, l) = (c[0] as usize, c[1] as usize);
            (o..o + l).all(|r| z(30_716 + r + 1) == z(30_716 + o + 1))
        }));
        // inside one bucket: one group up to the cap
        assert_eq!(
            rows_groups_law([(0usize, 8usize, 3000u32)], z, 6),
            vec![0, 6, 6, 2]
        );
    }
}
