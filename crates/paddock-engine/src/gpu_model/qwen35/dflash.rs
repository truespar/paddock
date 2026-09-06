//! Qwen3.8-27B's DFlash2 block-diffusion drafter (incoai/z-lab release,
//! 2026-08). Ported from `gemma4/dflash.rs` (muse's DFlash1/2 - the template
//! for every line here); the checkpoint schema is identical to muse's v2
//! (81 tensors = 5 x 15 + 6: per-sublayer two-tap grouped dynamic convs +
//! the rank-256 candidate selector), only the geometry differs: block_size 8
//! (muse 16), taps [6,20,34,48,62] over the 64-layer target, eps 1e-6, rope
//! theta 1e7, no logit epilogue (Qwen head: no scale, no softcap).
//!
//! One drafter forward drafts a whole block: rows = [last committed token,
//! 7 x mask id 248070] embedded through the TARGET's `token_embd`, five
//! non-causal SWA drafter layers over a per-slot feature-KV ring, the
//! drafter's final norm, then the TARGET's `output.weight` and the selector
//! (or per-row argmax). Rows 1.. are the drafts.
//!
//! What the drafter conditions on is the target's own middle: per layer it
//! reads a KV ring built from `z = enc_norm(fc(concat h_i))` where `h_i` is
//! the residual ENTERING target layers `dflash.target_layers` (the GGUF
//! spelling; the HF config card says [5,19,33,47,61] = the residual LEAVING
//! those layers - same rows, +1 convention, exactly the muse decode).
//!
//! qwen35-specific deviations from the gemma4 template, all structural:
//! - the target is a GDN hybrid, but the drafter itself is attention-only
//!   and the taps read the raw residual, so the mixer kinds never matter;
//! - the head election is qwen35's own (f8d lm head at rows >= 8, mmq
//!   below), not gemma4's f8t/f8row trio;
//! - `dflash_append_features` takes the row-position/slot device buffers as
//!   parameters - qwen35's three walks (batched decode, unified span, spec
//!   verify) stage rows in different buffers, where gemma4 had one pf pair;
//! - under the v2 overlapped mixed tick the round and the span would race
//!   the one fusion accumulator (`zacc`), so the mixed composition falls
//!   back to the sequential round -> span order while a DFlash drafter is
//!   armed (see forward_mixed_spec_plans_mtp).

use std::collections::HashMap;
use std::path::Path;

use cudarc::driver::CudaSlice;

use super::ops::{embed_any, kq_mm_pre, mmq, mmq_pre_any, prefill_mm_pre_any};
use super::{GpuQwen35, SendGraph};
use crate::gpu::{GpuError, KvDtype, QuantW};
use crate::gpu_model::gemma4::load::{
    LoadError, key_bool_array, key_f32, key_u64, key_u64_array, plain_rope,
};
use crate::gpu_model::gpt_oss::GpuModelError;
use paddock_models::mapped::MappedGguf;

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

fn le(e: LoadError) -> GpuModelError {
    GpuError::Driver(format!("qwen35 dflash attach: {e}")).into()
}

/// Draft blocks per round. One DFlash forward covers every warm slot, but
/// the rows are not free: the drafter's head runs the TARGET's 248k vocab
/// over all of them. Same cap rationale as muse/laguna.
fn max_blocks() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_DFLASH_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            // 32, raised from 8: the round buffers key off min(this,
            // slots), so narrow serves are untouched (8 slots still sizes
            // 64 rows) while a 32-slot serve gets the full 256-row round the
            // drafter forward already batches (graphs keyed (n, rows)). At
            // the old 8, a 32-live round declined the chain arm and the sync
            // hook drafted for only the first 8 slots.
            .unwrap_or(32)
    })
}

/// PADDOCK_DFLASH_NOCONV=1: v2 weights with the conv skipped. Not a shipping
/// mode - separates "the conv is wrong" from "something else is missing".
fn conv_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NOCONV").is_some())
}

/// PADDOCK_DFLASH_NOSELECT=1: draft a v2 checkpoint by per-row argmax with
/// the selector loaded but unused. Separates the selector from the conv.
fn sel_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NOSELECT").is_some())
}

/// PADDOCK_DFLASH_NORS=1: the selector walk stays GREEDY and drafted verify
/// rows resolve under the classic sample-and-match rule (the rung-G A/B;
/// both arms are lossless, the sampled walk + rejection sampling just
/// accepts more at temperature > 0).
fn rs_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NORS").is_some())
}

/// PADDOCK_DFLASH_NORUNS=1: the ring attention goes back to one launch per
/// block (the rung-E2 A/B; bit-identical either way).
fn runs_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NORUNS").is_some())
}

/// PADDOCK_DFLASH_NOFUSE=1: taps skipped entirely (prices the per-tick
/// fusion; slots never warm, so speculation stays off).
pub(crate) fn fuse_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NOFUSE").is_some())
}

/// DFlash2's grouped dynamic convolution, one per sublayer - `prepare`
/// before with base side 0, `finish` after with side 1; both sides' dynamic
/// deltas come out of one projection GEMM.
pub(crate) struct DflashConv {
    /// `[embd, taps, 2]` F32 in GGUF order (side-major last).
    pub base: CudaSlice<f32>,
    /// `[embd, 2*taps*num_groups]`
    pub proj: QuantW,
    /// e4m3 twin of `proj` - see DflashLayerF8 for why.
    pub proj_f8: Option<crate::gpu::F8RowPlane>,
}

/// DFlash2's candidate selector: rank-r bilinear transition scores so a
/// block of drafts is chosen as a coherent PATH (v1's conditional-
/// independence fix).
pub(crate) struct DflashSelector {
    pub hidden: QuantW,
    /// e4m3 twin of `hidden`. `pred`/`succ` stay k-quant: they are CODEBOOKS
    /// read by kquant_gather, not GEMM operands.
    pub hidden_f8: Option<crate::gpu::F8RowPlane>,
    pub pred: QuantW,
    pub succ: QuantW,
}

/// e4m3 row-scaled twins of one drafter layer's seven projections.
///
/// The drafter ships Q4_K, so every one of its GEMMs ran the W4A8 dp4a class
/// - `pd_kquant_w4a8_pipe2` profiles at **42.6% of the whole GPU** at c16
///   spec. That class is the one B200 de-rates hardest: ~1148 TOPS against
///   ~7.5 PF for e4m3. It is not a bandwidth problem - the drafter moves
///   ~1.19 GB per forward in ~2.94 ms, about 405 GB/s or 5% of the die's roof -
///   so paying 2x the bytes (0.5 -> 1.0 B/param) to buy the e4m3 arithmetic
///   class is the right trade here, and only here.
///
/// Built once at attach: dequantise the k-quant plane on device, then hand the
/// bf16 stream to `bf16_to_f8row`. No new pack kernel and no ABI slot - the
/// drafter's head has ridden an f8 plane all along, this extends it to the
/// layer walk. Kill: PADDOCK_DFLASH_F8=0.
pub(crate) struct DflashLayerF8 {
    pub wq: crate::gpu::F8RowPlane,
    pub wk: crate::gpu::F8RowPlane,
    pub wv: crate::gpu::F8RowPlane,
    pub wo: crate::gpu::F8RowPlane,
    pub w_gate: crate::gpu::F8RowPlane,
    pub w_up: crate::gpu::F8RowPlane,
    pub w_down: crate::gpu::F8RowPlane,
}

/// PLD (Paddock Lattice Drafter) parts - the pilot successor drafter
/// Rides the
/// dflash lane wholesale (same rings, tables, controller surface); what
/// changes is the feature transform (whitened rank-`rf` tap codes + a
/// per-layer bus instead of the full-width fc fold), the narrow block
/// width (`d` < target embd, entered through `in_proj`), and the head
/// (rank-`rank` lattice scan `W_h` then codebook instead of the target
/// lm_head). No selector: greedy chain via argmax_rows (the v1 path).
pub(crate) struct PldParts {
    /// [target_embd -> d] block-row entry after the target-table embed
    pub in_proj: QuantW,
    /// per tap: [target_embd -> rf] whitened PCA projector (mean folded
    /// into the bus bias by the exporter)
    pub pca: Vec<QuantW>,
    /// per layer x per tap: [rf -> bus] bands of the bus MLP's first GEMM
    pub bus_w: Vec<Vec<QuantW>>,
    /// per layer: host copy of the bus bias (tiled to cap rows at state
    /// build; the mean fold lives in here)
    pub bus_b: Vec<Vec<f32>>,
    /// per layer: [bus -> kv_dim] context K / V projections
    pub kt: Vec<QuantW>,
    pub vt: Vec<QuantW>,
    /// [d -> rank] head query
    pub head: QuantW,
    /// [rank -> vocab] lattice codebook (the head scan's replacement)
    pub codebook: QuantW,
    /// drafter block width (embedding_length of the GGUF)
    pub d: usize,
    /// target embedding width (tap/embed input width)
    pub d_t: usize,
    pub rf: usize,
    pub bus: usize,
    pub rank: usize,
}

pub(crate) struct DflashLayer {
    /// e4m3 twins of the seven projections below; None = the k-quant ladder.
    pub f8: Option<DflashLayerF8>,
    pub attn_norm: CudaSlice<f32>,
    pub wq: QuantW,
    pub wk: QuantW,
    pub wv: QuantW,
    pub wo: QuantW,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub w_gate: QuantW,
    pub w_up: QuantW,
    pub w_down: QuantW,
    pub attn_conv: Option<DflashConv>,
    pub ffn_conv: Option<DflashConv>,
}

/// Serving state: feature KV rings + row bands (fusion at tick width, the
/// draft round at block width). Built by `dflash_ensure_state` once the
/// batch lane knows its slot count.
pub(crate) struct DflashState {
    /// per-layer feature K/V stores, f16. Dense mode: slot-strided rings,
    /// [slots * ring * 16, kv_dim]. Paged mode: POOL STRIPES,
    /// [pool_blocks * 16, kv_dim] - one more full-attn-shaped store beside
    /// the backbone's, addressed by the combined block tables so
    /// prefix-cache adoption restores drafter features with the pages
    /// (the mtp_kv_* precedent).
    pub kv: Vec<(CudaSlice<u8>, CudaSlice<u8>)>,
    /// Block table the append/attention kernels read. Dense mode: the
    /// static ring map s*ring + j%ring, written once. Paged mode: a MIRROR
    /// of `block_table_host` (the combined tables' truth), re-staged by
    /// `dflash_stage_tables` at every append/draft entry - the draft graphs
    /// bake this buffer's address, so content updates are all it takes.
    pub d_bt: CudaSlice<u32>,
    pub bps: usize,
    /// Paged (pool-stripe) mode - see `kv`/`d_bt`.
    pub paged: bool,
    /// Fusion row capacity - must cover a whole walk's rows or the touched
    /// slots go cold (`stale`) rather than keeping a silent hole.
    pub cap: usize,
    /// Per slot, the tokens whose features sit in the ring, coverage order.
    pub cov: Vec<Vec<u32>>,
    zacc: CudaSlice<f32>,
    ztmp: CudaSlice<f32>,
    tq: CudaSlice<i8>,
    ts: CudaSlice<f32>,
    tyq: CudaSlice<u8>,
    txs: CudaSlice<f32>,
    tss: CudaSlice<f32>,
    fk: CudaSlice<f32>,
    fkn: CudaSlice<f32>,
    fv: CudaSlice<f32>,
    /// append row staging (positions/slots for the fused rows) - the walks
    /// each stage rows differently, so the append uploads its own mirrors
    ap_pos: CudaSlice<u32>,
    ap_slots: CudaSlice<u32>,
    /// written-row indices for the conditioning fold (rung C): the cut
    /// windows flattened once per round, layer-invariant - the fused
    /// norm+rope+store kernel reads fk/fv/ap_pos/ap_slots through these
    ap_rows: CudaSlice<u32>,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    q: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    k: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    cvx: CudaSlice<f32>,
    cvc: CudaSlice<f32>,
    sel_params: CudaSlice<u32>,
    sel_topk: CudaSlice<u32>,
    pub(super) sel_ids: CudaSlice<u32>,
    sel_pred: CudaSlice<f32>,
    sel_succ: CudaSlice<f32>,
    sel_hs: CudaSlice<f32>,
    /// Rung G: the sampled selector walk's K-way draft distribution per
    /// drafter row (`[rows][top_k]`, one-hot on greedy blocks) - the q the
    /// rejection-sampling verify resolves against. 1 float when there is no
    /// selector (the arm needs one).
    pub(super) q16: CudaSlice<f32>,
    /// Rung G: per-block 1/T (0 = greedy walk) and u32 seed for the sampled
    /// walk, staged per round from the service's RS chain draws.
    blk_invt: CudaSlice<f32>,
    blk_seed: CudaSlice<u32>,
    ffn_gate: CudaSlice<f32>,
    ffn_up: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xs: CudaSlice<f32>,
    ssums: CudaSlice<f32>,
    part: CudaSlice<f32>,
    yq: CudaSlice<u8>,
    xsums: CudaSlice<f32>,
    skfix: CudaSlice<f32>,
    /// e4m3 staging for the f8d head route (scales are e8m0 bytes,
    /// one per 32-element group - same shape as the shared d_exs).
    /// e4m3 row-scaled activation staging for the drafter's e4m3 twins
    /// (DflashLayerF8). Widest consumer is the FFN down input, `rows * ff`.
    f8q: CudaSlice<i8>,
    f8rs: CudaSlice<f32>,
    e4q: CudaSlice<i8>,
    e4rs: CudaSlice<u8>,
    sinks: CudaSlice<f32>,
    d_toks: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_apos: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    /// pub(super): the async chain arm (spec.rs dflash_draft_begin) copies
    /// these picks device-side into the chain's d_draft.
    pub(super) d_out: CudaSlice<u32>,
    /// per-slot contiguous feature coverage [start, end)
    pub feat: Vec<(u32, u32)>,
    pub stale: bool,
    /// PLD scratch (1-element when the attachment is DFlash): tap-code
    /// staging, bus band dst + accumulator, per-layer tiled bias, the
    /// silu-via-swiglu ones plane, the head query, and the embed staging.
    pld_stage: CudaSlice<f32>,
    pld_band: CudaSlice<f32>,
    pld_acc: CudaSlice<f32>,
    pld_bias: CudaSlice<f32>,
    pld_ones: CudaSlice<f32>,
    pld_u: CudaSlice<f32>,
    pld_emb: CudaSlice<f32>,
    /// zeros for the hybrid selector's j=0 gate (anchor pred codes)
    pld_zero: CudaSlice<f32>,
    /// captured draft rounds keyed by (block count, rows per block)
    pub graphs: HashMap<(usize, usize), SendGraph>,
    /// batched-runs row tables for the ring attention, keyed by rows per
    /// block: `[0, rows, 2*rows, ..., cap*rows]` (cap = dflash_round_cap).
    /// One launch with grid.z = blocks replaces the per-block loop (rung
    /// E2); the captured graph bakes the table's pointer, so a table lives
    /// as long as the state does (built in dflash_draft_launch, outside
    /// any capture).
    run_offs: HashMap<usize, CudaSlice<u32>>,
}

pub(crate) struct DflashDrafter {
    pub layers: Vec<DflashLayer>,
    /// `fc` split into per-tap [embd, embd] bands, target_layers order.
    pub fc_bands: Vec<QuantW>,
    /// e4m3 twins of `fc_bands`. The tap fold runs once per band per tick and
    /// was the last k-quant GEMM left in the drafter: at imax,
    /// pd_kquant_gemm_dp4a is 18.2% of the die across 20520 launches, and the
    /// same census on other engines shows no int8 class at all.
    pub fc_f8: Vec<Option<crate::gpu::F8RowPlane>>,
    pub enc_norm: CudaSlice<f32>,
    pub final_norm: CudaSlice<f32>,
    /// target-layer INPUT indices the walk taps
    pub target_layers: Vec<usize>,
    pub block: usize,
    pub mask_token: u32,
    pub n_heads: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub window: usize,
    pub eps: f32,
    pub rope: (f32, f32, f32, f32, f32, f32),
    /// (taps, group_size, num_groups); `None` = v1.
    pub conv: Option<(usize, usize, usize)>,
    /// selector + (rank, top_k); `None` = v1.
    pub selector: Option<(DflashSelector, usize, usize)>,
    /// drafter logit epilogue (scale then cap); this checkpoint declares
    /// neither so both are inert, but the plumbing stays - see the muse
    /// the acceptance postmortem.
    pub logit_scale: f32,
    pub softcap: f32,
    /// `Some` = this attachment is a PLD pilot pack, not a DFlash release.
    pub pld: Option<PldParts>,
    pub state: Option<DflashState>,
}

pub struct DflashSelftest {
    pub drafts: Vec<u32>,
    pub repeat_identical: bool,
    pub ms_per_round: f64,
}

impl GpuQwen35 {
    /// Sideload the DFlash2 drafter GGUF (arch "dflash"). Validates its
    /// geometry against the target, splits `fc` into per-tap bands, leaves
    /// serving untouched until `dflash_ensure_state`.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<(), GpuModelError> {
        self.attach_dflash_inner(path).map_err(le)?;
        // Width elections for the ATTACHED block drafter.
        // serve_spec_k_budget's ladder with a 256-row serving budget and a
        // 128-row 9..16 boundary:
        //   live <= 8   k = 32/live - 1  (7 at c4, 3 at c8 - unchanged)
        //   live 9..16  k = 128/live - 1 -> 7 (MAX_K)
        //   live 17..32 k = 256/live - 1 -> 7
        // A 64-row budget was tried first, back when the f8 lin GEMMs refused
        // batch > 64: every deeper round failed its head launch and the
        // controller pinned k=1 at width. Once lin planes above 64 rows route
        // through their prefill-class kt arm, the deep rungs run clean and
        // win: acceptance falls with depth (92.6%/tok at k=1 -> ~76% at k=7)
        // and throughput still climbs, because DEPTH at WIDTH is where the
        // wide-batch spec win actually comes from - the same fixed gamma~7
        // the other engines run. c8 is the exception and keeps k=3.
        // The plans gate matches gemma4's election (32). gemma4 precedent:
        // attach-time set_env, explicit env always wins; load runs before the
        // serve loop's read-once.
        if std::env::var_os("PADDOCK_SPEC_MAX_ROWS").is_none() {
            crate::envset::set_env("PADDOCK_SPEC_MAX_ROWS", "256");
        }
        if std::env::var_os("PADDOCK_SPEC_NARROW_ROWS").is_none() {
            crate::envset::set_env("PADDOCK_SPEC_NARROW_ROWS", "128");
        }
        if std::env::var_os("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX").is_none() {
            crate::envset::set_env("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX", "32");
        }
        Ok(())
    }

    fn attach_dflash_inner(&mut self, path: &Path) -> Result<(), LoadError> {
        let file = if path.is_dir() {
            // catalog artifact name first, then the upstream basename
            let a = path.join("dflash2-Q4_K_M.gguf");
            if a.exists() {
                a
            } else {
                path.join("Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
            }
        } else {
            path.to_path_buf()
        };
        let map = MappedGguf::open(&file).map_err(|e| {
            LoadError::Tensor(file.display().to_string(), format!("open dflash gguf: {e}"))
        })?;
        let arch = map
            .gguf()
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str().map(|s| s.to_owned()))
            .unwrap_or_default();
        if arch != "dflash" {
            return Err(LoadError::BadKey(format!(
                "general.architecture = {arch:?}, want \"dflash\""
            )));
        }
        let k = |s: &str| format!("dflash.{s}");
        let n_layer = key_u64(&map, &k("block_count"))? as usize;
        let embd = key_u64(&map, &k("embedding_length"))? as usize;
        let ff = key_u64(&map, &k("feed_forward_length"))? as usize;
        let n_heads = key_u64(&map, &k("attention.head_count"))? as usize;
        let n_kv = key_u64(&map, &k("attention.head_count_kv"))? as usize;
        let hd = key_u64(&map, &k("attention.key_length"))? as usize;
        let hd_v = key_u64(&map, &k("attention.value_length"))? as usize;
        let window = key_u64(&map, &k("attention.sliding_window"))? as usize;
        let eps = key_f32(&map, &k("attention.layer_norm_rms_epsilon"), Some(1e-6))?;
        let theta = key_f32(&map, &k("rope.freq_base"), Some(10_000_000.0))?;
        let block = key_u64(&map, &k("block_size"))? as usize;
        let taps: Vec<usize> = key_u64_array(&map, &k("target_layers"))?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let mask_token = key_u64(&map, "tokenizer.ggml.mask_token_id")? as u32;

        let conv_cfg = match map.gguf().metadata.get(&k("conv_kernel_size")) {
            None => None,
            Some(_) => {
                let ctaps = key_u64(&map, &k("conv_kernel_size"))? as usize;
                let gsz = key_u64(&map, &k("conv_group_size"))? as usize;
                if !(1..=8).contains(&ctaps) {
                    return Err(LoadError::BadKey(format!(
                        "dflash conv_kernel_size {ctaps} outside 1..=8"
                    )));
                }
                if gsz == 0 || !embd.is_multiple_of(gsz) {
                    return Err(LoadError::BadKey(format!(
                        "dflash conv_group_size {gsz} does not divide embd {embd}"
                    )));
                }
                // REFUSE rather than fall back to the v1 forward: v2 weights
                // are trained expecting the conv (see the gemma4 note).
                if !self.exec.has_dflash_conv() {
                    return Err(LoadError::BadKey(
                        "dflash2 checkpoint needs the grouped-conv kernel (pack slot 459); \
                         this pack predates it - rebuild packs/cuda"
                            .into(),
                    ));
                }
                if !embd.is_multiple_of(4) || !gsz.is_multiple_of(4) {
                    return Err(LoadError::BadKey(format!(
                        "dflash2 conv geometry embd {embd} / group {gsz} must both be \
                         multiples of 4 for the packed channel walk"
                    )));
                }
                Some((ctaps, gsz, embd / gsz))
            }
        };
        let sel_cfg = match conv_cfg {
            None => None,
            // PLD packs without selector tensors: greedy chain (v1 path).
            // With selector_rank present, the hybrid selector loads normally.
            Some(_)
                if map.gguf().metadata.contains_key(&k("pld"))
                    && !map.gguf().metadata.contains_key(&k("selector_rank")) =>
            {
                None
            }
            Some(_) => Some((
                key_u64(&map, &k("selector_rank"))? as usize,
                key_u64(&map, &k("selector_top_k"))? as usize,
            )),
        };
        let logit_scale = key_f32(&map, &k("logit_scale"), Some(1.0))?;
        let softcap = key_f32(&map, &k("final_logit_softcapping"), Some(0.0))?;
        // PLD pilot pack: narrow block width + tap-code bus + lattice head,
        // no selector. Presence of the flag key is the discriminator.
        let is_pld = map.gguf().metadata.contains_key(&k("pld"));

        if !is_pld && embd != self.embd {
            return Err(LoadError::BadKey(format!(
                "dflash embedding_length {embd} != target n_embd {}",
                self.embd
            )));
        }
        if hd != hd_v {
            return Err(LoadError::BadKey(format!(
                "dflash key_length {hd} != value_length {hd_v} - the ring is one dim"
            )));
        }
        if taps.is_empty() || taps.iter().any(|&t| t >= self.n_layers) || !taps.is_sorted() {
            return Err(LoadError::BadKey(format!(
                "dflash target_layers {taps:?} don't index the {}-layer target",
                self.n_layers
            )));
        }
        if !(2..=64).contains(&block) {
            return Err(LoadError::BadKey(format!(
                "dflash block_size {block} outside the 2..=64 a draft round sizes"
            )));
        }
        if window == 0 {
            return Err(LoadError::BadKey(
                "dflash sliding_window 0 - the ring needs a bound".into(),
            ));
        }
        if let Ok(pat) = key_bool_array(&map, &k("attention.sliding_window_pattern"))
            && (pat.len() != n_layer || pat.iter().any(|&b| !b))
        {
            return Err(LoadError::BadKey(format!(
                "dflash sliding_window_pattern {pat:?}: this lane implements the all-SWA drafter"
            )));
        }

        let vocab = self.vocab;

        self.exec
            .vram_load_gate(
                file.metadata().map(|m| m.len()).unwrap_or(0),
                "qwen35-dflash",
            )
            .map_err(|e| LoadError::WontFit(e.to_string()))?;

        let exec = self.exec.clone();
        let mut bytes = 0u64;
        let mut plane = |name: &str, want: [usize; 2]| -> Result<QuantW, LoadError> {
            let w = exec.load_quantw(&map, name)?;
            let dims = match &w {
                QuantW::Q8(q) => q.dims.clone(),
                QuantW::Kq(q) => q.dims.clone(),
            };
            if dims.len() != 2 || dims[1] != want[1] || dims[0] < want[0] {
                return Err(LoadError::Tensor(
                    name.to_owned(),
                    format!("dims {dims:?}, want {want:?}"),
                ));
            }
            bytes += w.bytes();
            Ok(w)
        };
        let mut conv_bytes = 0u64;
        let mut conv_base = |name: &str, ctaps: usize| -> Result<CudaSlice<f32>, LoadError> {
            let tt = exec.upload(&map, name)?;
            let want = vec![embd, ctaps, 2];
            if tt.dims != want {
                return Err(LoadError::Tensor(
                    name.to_owned(),
                    format!("{:?} != {want:?}", tt.dims),
                ));
            }
            conv_bytes += 4 * (embd * ctaps * 2) as u64;
            Ok(tt.buf)
        };
        let norm = |name: &str, n: usize| -> Result<CudaSlice<f32>, LoadError> {
            let t = exec.upload(&map, name)?;
            if t.dims != vec![n] {
                return Err(LoadError::Tensor(
                    name.to_owned(),
                    format!("{:?} != [{n}]", t.dims),
                ));
            }
            Ok(t.buf)
        };

        let (q_dim, kv_dim) = (n_heads * hd, n_kv * hd);
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            layers.push(DflashLayer {
                f8: None, // filled by the e4m3 pass below, once all layers exist
                attn_norm: norm(&p("attn_norm.weight"), embd)?,
                wq: plane(&p("attn_q.weight"), [embd, q_dim])?,
                wk: plane(&p("attn_k.weight"), [embd, kv_dim])?,
                wv: plane(&p("attn_v.weight"), [embd, kv_dim])?,
                wo: plane(&p("attn_output.weight"), [q_dim, embd])?,
                q_norm: norm(&p("attn_q_norm.weight"), hd)?,
                k_norm: norm(&p("attn_k_norm.weight"), hd)?,
                ffn_norm: norm(&p("ffn_norm.weight"), embd)?,
                w_gate: plane(&p("ffn_gate.weight"), [embd, ff])?,
                w_up: plane(&p("ffn_up.weight"), [embd, ff])?,
                w_down: plane(&p("ffn_down.weight"), [ff, embd])?,
                attn_conv: match conv_cfg {
                    None => None,
                    Some((ctaps, _, ng)) => Some(DflashConv {
                        proj_f8: None, // filled by the e4m3 pass below
                        base: conv_base(&p("attn_conv_base"), ctaps)?,
                        proj: plane(&p("attn_conv_proj.weight"), [embd, 2 * ctaps * ng])?,
                    }),
                },
                ffn_conv: match conv_cfg {
                    None => None,
                    Some((ctaps, _, ng)) => Some(DflashConv {
                        proj_f8: None, // filled by the e4m3 pass below
                        base: conv_base(&p("ffn_conv_base"), ctaps)?,
                        proj: plane(&p("ffn_conv_proj.weight"), [embd, 2 * ctaps * ng])?,
                    }),
                },
            });
        }
        let mut selector = match sel_cfg {
            None => None,
            Some((rank, top_k)) => {
                if top_k == 0 {
                    return Err(LoadError::BadKey(format!(
                        "dflash selector_top_k {top_k} must be >= 1"
                    )));
                }
                if !self.exec.has_dflash_select() || !self.exec.has_topk_rows() {
                    return Err(LoadError::BadKey(
                        "dflash2 selector needs pack slots 460/461 + topk_rows; \
                         this pack predates them - rebuild packs/cuda"
                            .into(),
                    ));
                }
                if rank % 256 != 0 {
                    return Err(LoadError::BadKey(format!(
                        "dflash2 selector_rank {rank} must be a multiple of 256 - the \
                         codebook row-gather dequants whole k-quant super-blocks"
                    )));
                }
                let (hidden, pred, succ) = (
                    plane("selector_hidden.weight", [embd, rank])?,
                    plane("selector_predecessor.weight", [rank, vocab])?,
                    plane("selector_successor.weight", [rank, vocab])?,
                );
                if !matches!((&pred, &succ), (QuantW::Kq(_), QuantW::Kq(_))) {
                    return Err(LoadError::Tensor(
                        "selector_predecessor/successor".into(),
                        "codebook row-gather implements the k-quant classes only".into(),
                    ));
                }
                Some((
                    DflashSelector {
                        hidden,
                        hidden_f8: None,
                        pred,
                        succ,
                    },
                    rank,
                    top_k,
                ))
            }
        };
        // PLD: the fc fold is replaced by per-tap PCA projectors + a
        // per-layer bus; load those instead of the full-width bands.
        let pld_parts = if is_pld {
            let d_t = key_u64(&map, &k("pld_target_embedding_length"))? as usize;
            let rf = key_u64(&map, &k("pld_feature_rank"))? as usize;
            let bus = key_u64(&map, &k("pld_bus_width"))? as usize;
            let rank = key_u64(&map, &k("pld_head_rank"))? as usize;
            if d_t != self.embd {
                return Err(LoadError::BadKey(format!(
                    "pld target width {d_t} != target n_embd {}",
                    self.embd
                )));
            }
            let mut pca = Vec::with_capacity(taps.len());
            for t in 0..taps.len() {
                pca.push(plane(&format!("pld_pca.{t}.weight"), [d_t, rf])?);
            }
            let mut bus_w = Vec::with_capacity(n_layer);
            let mut bus_b = Vec::with_capacity(n_layer);
            let mut kt = Vec::with_capacity(n_layer);
            let mut vt = Vec::with_capacity(n_layer);
            for i in 0..n_layer {
                let mut bands = Vec::with_capacity(taps.len());
                for t in 0..taps.len() {
                    bands.push(plane(&format!("pld_bus.{i}.{t}.weight"), [rf, bus])?);
                }
                bus_w.push(bands);
                let bt = exec.upload(&map, &format!("pld_bus.{i}.bias"))?;
                if bt.dims != vec![bus] {
                    return Err(LoadError::Tensor(
                        format!("pld_bus.{i}.bias"),
                        format!("{:?} != [{bus}]", bt.dims),
                    ));
                }
                let mut host = vec![0f32; bus];
                exec.stream
                    .memcpy_dtoh(&bt.buf, &mut host)
                    .map_err(|e| LoadError::Tensor(format!("pld_bus.{i}.bias"), e.to_string()))?;
                bus_b.push(host);
                kt.push(plane(&format!("pld_kt.{i}.weight"), [bus, kv_dim])?);
                vt.push(plane(&format!("pld_vt.{i}.weight"), [bus, kv_dim])?);
            }
            Some(PldParts {
                in_proj: plane("pld_in_proj.weight", [d_t, embd])?,
                pca,
                bus_w,
                bus_b,
                kt,
                vt,
                head: plane("pld_head.weight", [embd, rank])?,
                codebook: plane("pld_codebook.weight", [rank, vocab])?,
                d: embd,
                d_t,
                rf,
                bus,
                rank,
            })
        } else {
            None
        };
        let fc_bands = if is_pld {
            Vec::new()
        } else {
            self.dflash_fc_bands(&map, taps.len(), embd, &mut bytes)?
        };
        bytes += conv_bytes;
        let mut fc_f8: Vec<Option<crate::gpu::F8RowPlane>> = Vec::new();
        let enc_norm = norm("enc.output_norm.weight", embd)?;
        let final_norm = norm("output_norm.weight", embd)?;

        // Header audit: a tensor we silently ignore is modeling drift. v1 is
        // 11/layer + 3; v2 adds two conv modules per layer + 3 selector planes.
        // PLD: 15/layer + 8 bus/kt/vt per layer + 12 globals (incl. the
        // confidence head, exported but unread by this v0 lane).
        let expected = if is_pld {
            n_layer * 23 + 12 + if sel_cfg.is_some() { 3 } else { 0 }
        } else if conv_cfg.is_some() {
            n_layer * 15 + 6
        } else {
            n_layer * 11 + 3
        };
        let seen = map.tensor_infos().count();
        if seen != expected {
            return Err(LoadError::BadKey(format!(
                "dflash: {seen} tensors in file, {expected} consumed"
            )));
        }

        // No attach-time spec-env defaults. The first cut set
        // PADDOCK_SPEC_K_MISS_FLOOR = block-1 here, following gemma4's attach
        // precedent, to fix the dflash k death-spiral - but the env is a
        // GLOBAL OnceLock in the service, so the floor leaked into the MTP
        // regime too: chain rounds at live 4..8 re-drafted 7 deep after every
        // miss, and an attached serve came out slower than pure MTP. The
        // floor is now per-ROUND via
        // spec_k_miss_floor_mtp (spec.rs): dflash rounds keep block-1, MTP
        // rounds keep the classic service default. PADDOCK_SPEC_MAX_K needs
        // no default either - the service ceiling is already 7 = block-1.

        tracing::info!(
            "qwen35 dflash{} drafter attached: {n_layer} layers, block {block}, mask \
             {mask_token}, taps {taps:?} (= residual leaving {:?}), swa {window}, {:.2} GB{}",
            if conv_cfg.is_some() { "2" } else { "" },
            taps.iter().map(|t| t.saturating_sub(1)).collect::<Vec<_>>(),
            bytes as f64 / 1e9,
            match (conv_cfg, sel_cfg) {
                (Some((tp, gs, ng)), Some((rank, tk))) => format!(
                    " | conv {tp}-tap x {ng} groups of {gs}, selector rank {rank} top-{tk}, \
                     epilogue scale {logit_scale} cap {softcap}"
                ),
                _ => String::new(),
            },
        );
        self.weights_bytes = Some(self.weights_bytes.unwrap_or(0) + bytes);
        // e4m3 twins for the layer walk (see DflashLayerF8). Built here, after
        // every layer exists and before the drafter is published, so a partial
        // conversion can never be observed. One f32 scratch sized to the
        // widest projection is reused for all of them.
        // Device GATE. dflash_f8_on() reads an env var and nothing
        // else, so on a card whose pack nulls the e4m3 kernels the twins were
        // built anyway and every request died at the first drafter walk with
        // `kernel f8row_gemm missing from the loaded pack`. exports.cuh nulls
        // that whole family below sm_89 (`cma < 9 && !(cma == 8 && cmi >= 9)`
        // - mma.sync e4m3 is sm_89+), so an A6000 could not run this lane at
        // all. Measured on sm_86: 500 on every request, spec on.
        //
        // Ask for the kernels the walk actually calls, not just the one that
        // happened to fail first - gating on f8row_gemm alone would be the
        // same too-narrow mistake in a new place. quantize_e4m3_row is not in
        // the nulled set (it is a quantize, no mma), so the three mma-class
        // ones are the gate. Same shape gemma4 already uses at its two load
        // sites.
        //
        // Failing the gate is not a downgrade to something broken: `f8: None`
        // is the k-quant ladder (see DflashLayerF8), which is the tuned W4A8
        // dp4a path Ampere wants and what the drafter ran before the e4m3
        // twins existed. sm_89 and every Blackwell part keep the twins.
        let f8_kernels =
            exec.has_f8row_gemm() && exec.has_f8d_gemm_mma_ks() && exec.has_bf16_to_f8row();
        if !f8_kernels && dflash_f8_on() {
            tracing::info!(
                "qwen35 dflash: e4m3 twins unavailable on this GPU (needs sm_89+                  for the mma.sync e4m3 kernels) - the drafter keeps the W4A8                  k-quant ladder"
            );
        }
        if dflash_f8_on() && f8_kernels && !is_pld {
            let widest = layers
                .iter()
                .flat_map(|l| [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down])
                .map(|w| match w {
                    QuantW::Kq(k) => k.dims[0] * k.dims[1],
                    QuantW::Q8(q) => q.dims[0] * q.dims[1],
                })
                .max()
                .unwrap_or(0);
            match self.exec.alloc(widest.max(1)) {
                Ok(mut scratch) => {
                    let mut host: Vec<u8> = Vec::new();
                    let mut built = 0usize;
                    for l in layers.iter_mut() {
                        l.f8 = build_layer_f8(&self.exec, l, &mut scratch, &mut host);
                        built += usize::from(l.f8.is_some());
                        // the conv side-projections are GEMM operands too - a
                        // c16 re-census after the seven main projections landed
                        // showed the remaining 14.7% of k-quant was exactly
                        // these plus the selector hidden.
                        for cv in [l.attn_conv.as_mut(), l.ffn_conv.as_mut()]
                            .into_iter()
                            .flatten()
                        {
                            cv.proj_f8 = kq_to_f8row(&self.exec, &cv.proj, &mut scratch, &mut host);
                        }
                    }
                    if let Some((sc, _, _)) = selector.as_mut() {
                        sc.hidden_f8 = kq_to_f8row(&self.exec, &sc.hidden, &mut scratch, &mut host);
                    }
                    fc_f8 = fc_bands
                        .iter()
                        .map(|w| kq_to_f8row(&self.exec, w, &mut scratch, &mut host))
                        .collect();
                    let gb = layers
                        .iter()
                        .filter_map(|l| l.f8.as_ref())
                        .map(|f| {
                            [&f.wq, &f.wk, &f.wv, &f.wo, &f.w_gate, &f.w_up, &f.w_down]
                                .iter()
                                .map(|p| p.data.len() as u64 + p.scale.len() as u64 * 4)
                                .sum::<u64>()
                        })
                        .sum::<u64>() as f64
                        / 1e9;
                    tracing::info!(
                        "qwen35 dflash: e4m3 twins built for {built}/{} layers (+{gb:.2} GB) -                         the drafter's layer walk leaves the W4A8 dp4a class                          (PADDOCK_DFLASH_F8=0 restores it)",
                        layers.len()
                    );
                }
                Err(e) => tracing::warn!("qwen35 dflash: no e4m3 twins ({e}) - keeping k-quant"),
            }
        }
        self.dflash = Some(DflashDrafter {
            layers,
            fc_bands,
            fc_f8,
            enc_norm,
            final_norm,
            target_layers: taps,
            block,
            mask_token,
            n_heads,
            n_kv,
            hd,
            window,
            eps,
            rope: plain_rope(theta, hd),
            conv: conv_cfg,
            selector,
            logit_scale,
            softcap,
            pld: pld_parts,
            state: None,
        });
        Ok(())
    }

    /// Split the fusion `fc` into one weight per tap (k-quant superblock
    /// strips; refuse other classes loudly).
    fn dflash_fc_bands(
        &self,
        map: &MappedGguf,
        n_taps: usize,
        embd: usize,
        bytes: &mut u64,
    ) -> Result<Vec<QuantW>, LoadError> {
        let (info, _) = map.tensor_bytes("fc.weight").map_err(GpuError::from)?;
        let (fin, fout) = (info.dims[0] as usize, info.dims[1] as usize);
        if fin != n_taps * embd || fout != embd {
            return Err(LoadError::Tensor(
                "fc.weight".into(),
                format!("dims [{fin}, {fout}], want [{}, {embd}]", n_taps * embd),
            ));
        }
        if crate::gpu::kq_params(info.ggml_type).is_none() {
            return Err(LoadError::Tensor(
                "fc.weight".into(),
                format!(
                    "{:?}: the band split implements the k-quant classes",
                    info.ggml_type
                ),
            ));
        }
        let bands = self.exec.repack_kquant_bands(map, "fc.weight", n_taps)?;
        Ok(bands
            .into_iter()
            .map(|b| {
                let w = QuantW::Kq(b);
                *bytes += w.bytes();
                w
            })
            .collect())
    }

    /// Attached AND servable by this pack (attach-time fact - read before
    /// the serving state exists).
    /// The attached block drafter's block size (the service's low-live row
    /// tier sizes the verify width from it; see `Generator::spec_block_width`).
    pub(crate) fn dflash_block_width(&self) -> Option<usize> {
        self.dflash.as_ref().map(|d| d.block)
    }

    pub(crate) fn dflash_attached(&self) -> bool {
        self.dflash.is_some() && self.exec.has_argmax_rows()
    }

    /// The block drafter owns every round at or below its live max - the
    /// same threshold `spec_draft_batch_mtp` routes on (a decline there is
    /// the only way the MTP chain sees a round at this width).
    pub(crate) fn dflash_owns_round(&self, live: usize) -> bool {
        self.dflash_attached() && live <= Self::dflash_live_max()
    }

    /// Serving state live? Every tap/append/draft site keys on this.
    pub(crate) fn dflash_armed(&self) -> bool {
        self.dflash.as_ref().is_some_and(|d| d.state.is_some())
    }

    /// Build the rings + row bands once the batch lane knows its slot count.
    pub(crate) fn dflash_ensure_state(&mut self) -> Result<(), GpuError> {
        if self.dflash.as_ref().is_none_or(|d| d.state.is_some()) {
            return Ok(());
        }
        let slots = self.batch.as_ref().map_or(1, |b| b.max_batch).max(1);
        // Maintenance-width bound. Above it the state is never built -
        // armed() stays false, every tap filter and append declines through
        // the existing state checks, and a wide serve is byte-identical to
        // pure MTP. This was 8 at first, because ring upkeep is priced per
        // ROW and the wide cells never actually drafted - pure maintenance
        // tax on top of our own MTP rows. With the async chain, the
        // snapshot-free verify, the 32-block round and the 256-row serving
        // budget, wide rounds do draft and the upkeep is repaid, so the bound
        // is 32. The k=0-at-width shape that made upkeep pure tax can no
        // longer occur at <= 32 live (the budget ladder floors at k=1
        // there).
        let maint_max = paddock_models::dev_var!("PADDOCK_QWEN35_DFLASH_MAINT_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32usize);
        if slots > maint_max {
            tracing::info!(
                "qwen35 dflash: serve width {slots} > maintenance bound {maint_max} - \
                 ring not built, serving pure MTP (block drafting pays at low live only)"
            );
            return Ok(());
        }
        let embd = self.embd;
        let vocab = self.vocab;
        let bps = self.max_ctx.div_ceil(16);
        let e = self.exec.clone();
        let out_f8 = self.out_f8.is_some();
        // computed before the dflash borrow: pool-stripe mode + its size
        let paged_blocks = self.batch.as_ref().and_then(|b| {
            (b.paged && b.d_block_tables.is_some())
                .then(|| b.pool.as_ref().map(|p| p.capacity() as usize))
                .flatten()
        });
        let table_bps = self.batch.as_ref().map_or(0, |b| b.blocks_per_slot);
        let df = self.dflash.as_mut().expect("attached");
        let (kv_dim, q_dim) = (df.n_kv * df.hd, df.n_heads * df.hd);
        let ff = match &df.layers[0].w_gate {
            QuantW::Q8(w) => w.dims[1],
            QuantW::Kq(w) => w.dims[1],
        };
        // any layer carrying e4m3 twins needs the rowwise activation staging
        let lane_f8 = df.layers.iter().any(|l| l.f8.is_some());
        let any_q8 = df.layers.iter().any(|l| {
            [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                .iter()
                .any(|w| matches!(w, QuantW::Q8(_)))
        });
        let ring = ((df.block + df.window).div_ceil(16) + 1).min(bps);
        // Paged serves get POOL STRIPES addressed by the combined block
        // tables (the mtp_kv_* precedent): adoption then restores drafter
        // features with the pages, so cross-slot/cross-request prefix hits
        // keep the block drafter warm. A slot-local static ring cannot do
        // this: every prefix-restored slot starts its ring at the resume
        // position and can never satisfy the exact warm rule, because the
        // features for the hit span were never computed - so repeated
        // prompts serve the MTP fallback for the whole session at roughly
        // half the throughput. Dense mode keeps the static rings.
        let paged = paged_blocks.is_some();
        if paged {
            assert_eq!(
                table_bps, bps,
                "dflash mirror table shape must match the combined tables"
            );
        }
        let mut host = vec![0u32; slots * bps];
        if !paged {
            for s in 0..slots {
                for j in 0..bps {
                    host[s * bps + j] = (s * ring + (j % ring)) as u32;
                }
            }
        }
        let d_bt = e.to_device_u32(&host)?;
        let mut kv = Vec::with_capacity(df.layers.len());
        for _ in 0..df.layers.len() {
            let b = match paged_blocks {
                Some(n) => n * 16 * kv_dim * 2,         // pool stripe, f16
                None => slots * ring * 16 * kv_dim * 2, // static ring, f16
            };
            kv.push((e.alloc_u8(b)?, e.alloc_u8(b)?));
        }
        // Fusion width: the widest tapped walk. qwen35's span tick admits
        // prefill_chunk_rows prefill rows (the elected chunk); the verify adds
        // live * (k+1); the window+block floor is what makes a truncating tap
        // harmless.
        let cap =
            (self.prefill_chunk_rows + slots * (df.block + 1) + 64).max(df.window + df.block + 64);
        let rows = max_blocks().min(slots) * df.block;
        let wide = embd.max(ff).max(q_dim);
        let conv_dim = df.conv.map_or(0, |(taps, _, ng)| 2 * taps * ng);
        let n_blk = max_blocks().min(slots);
        let (sel_r, sel_k) = df.selector.as_ref().map_or((0, 0), |(_, r, k)| (*r, *k));
        tracing::info!(
            "qwen35 dflash state: {slots} slots x {ring} ring blocks x {} layers \
             ({:.1} MB/slot), fusion cap {cap} rows, {} draft rows/round",
            df.layers.len(),
            (df.layers.len() * ring * 16 * kv_dim * 2 * 2) as f64 / 1e6,
            rows,
        );
        let (p_rf, p_bus, p_rank, p_dt, p_l) = df.pld.as_ref().map_or((0, 0, 0, 0, 0), |p| {
            (p.rf, p.bus, p.rank, p.d_t, df.layers.len())
        });
        df.state = Some(DflashState {
            kv,
            d_bt,
            bps,
            cap,
            pld_stage: e.alloc((cap * p_rf).max(1))?,
            pld_band: e.alloc((cap * p_bus).max(1))?,
            pld_acc: e.alloc((cap * p_bus).max(1))?,
            pld_bias: e.alloc((p_l * cap * p_bus).max(1))?,
            pld_ones: e.alloc((cap * p_bus).max(1))?,
            pld_u: e.alloc((rows * p_rank).max(1))?,
            pld_emb: e.alloc((rows * p_dt).max(1))?,
            pld_zero: e.alloc((n_blk * p_rank.max(sel_r)).max(1))?,
            zacc: e.alloc(cap * embd)?,
            ztmp: e.alloc(cap * embd)?,
            tq: e.alloc_i8(cap * embd)?,
            ts: e.alloc(cap * embd / 32)?,
            tyq: e.alloc_u8(embd.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            txs: e.alloc(embd.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
            tss: e.alloc(cap * embd / 16)?,
            fk: e.alloc(cap * kv_dim)?,
            fkn: e.alloc(cap * kv_dim)?,
            fv: e.alloc(cap * kv_dim)?,
            ap_pos: e.alloc_u32(cap)?,
            ap_slots: e.alloc_u32(cap)?,
            ap_rows: e.alloc_u32(cap)?,
            x: e.alloc(rows * embd)?,
            xn: e.alloc(rows * embd)?,
            q: e.alloc(rows * q_dim)?,
            qn: e.alloc(rows * q_dim)?,
            k: e.alloc(rows * kv_dim)?,
            kn: e.alloc(rows * kv_dim)?,
            v: e.alloc(rows * kv_dim)?,
            attn: e.alloc(rows * q_dim)?,
            proj: e.alloc(rows * embd)?,
            cvx: e.alloc(if conv_dim > 0 { rows * embd } else { 1 })?,
            cvc: e.alloc(if conv_dim > 0 { rows * conv_dim } else { 1 })?,
            sel_params: e.alloc_u32(if sel_k > 0 { rows * 4 } else { 1 })?,
            sel_topk: e.alloc_u32(if sel_k > 0 { rows * sel_k * 2 } else { 1 })?,
            sel_ids: e.alloc_u32(if sel_k > 0 { rows * sel_k + n_blk } else { 1 })?,
            sel_pred: e.alloc(if sel_k > 0 {
                (rows * sel_k + n_blk) * sel_r
            } else {
                1
            })?,
            sel_succ: e.alloc(if sel_k > 0 { rows * sel_k * sel_r } else { 1 })?,
            sel_hs: e.alloc(if sel_k > 0 { rows * sel_r } else { 1 })?,
            q16: e.alloc(if sel_k > 0 { rows * sel_k } else { 1 })?,
            blk_invt: e.alloc(n_blk.max(1))?,
            blk_seed: e.alloc_u32(n_blk.max(1))?,
            ffn_gate: e.alloc(rows * ff)?,
            ffn_up: e.alloc(rows * ff)?,
            logits: e.alloc(rows * vocab)?,
            xq: e.alloc_i8(rows * wide)?,
            xs: e.alloc(rows * wide / 32)?,
            ssums: e.alloc(rows * wide / 16)?,
            // split-K partials for the round's GEMMs: nz (<= 8) x rows x
            // out_dim. Was a fixed 64 rows - the wall that made every block
            // round deeper than k=1 at 32 live fail its head launch (rung D).
            part: e.alloc(8 * rows.max(64) * wide.max(vocab / 8))?,
            yq: e.alloc_u8(wide.div_ceil(128) * rows.next_multiple_of(128) * 144)?,
            xsums: e.alloc(wide.div_ceil(128) * rows.next_multiple_of(128) * 4)?,
            skfix: e.alloc(if any_q8 || out_f8 {
                256 * 128 * 128 + 256
            } else {
                1
            })?,
            f8q: e.alloc_i8(if lane_f8 { rows * embd.max(ff) } else { 1 })?,
            f8rs: e.alloc(if lane_f8 { rows } else { 1 })?,
            e4q: e.alloc_i8(if out_f8 { rows * embd } else { 1 })?,
            e4rs: e.alloc_u8(if out_f8 { rows * embd / 32 } else { 1 })?,
            sinks: e.alloc_no_sinks(df.n_heads)?,
            d_toks: e.alloc_u32(rows)?,
            d_pos: e.alloc_u32(rows)?,
            d_apos: e.alloc_u32(rows)?,
            d_slots: e.alloc_u32(rows)?,
            d_out: e.alloc_u32(rows)?,
            feat: vec![(0, 0); slots],
            cov: vec![Vec::new(); slots],
            stale: false,
            paged,
            graphs: HashMap::new(),
            run_offs: HashMap::new(),
        });
        if let Some(p) = df.pld.as_ref() {
            // tile the bus biases to cap rows and fill the silu ones plane
            let st = df.state.as_mut().expect("just built");
            let z = vec![0f32; st.pld_zero.len()];
            let mut v = st
                .pld_zero
                .try_slice_mut(0..z.len())
                .ok_or_else(|| GpuError::Driver("pld zero fill".into()))?;
            e.stream.memcpy_htod(&z, &mut v).map_err(drv)?;
            let mut host = Vec::with_capacity(p_l * cap * p_bus);
            for b in &p.bus_b {
                for _ in 0..cap {
                    host.extend_from_slice(b);
                }
            }
            let mut v = st
                .pld_bias
                .try_slice_mut(0..host.len())
                .ok_or_else(|| GpuError::Driver("pld bias tile slice".into()))?;
            e.stream.memcpy_htod(&host, &mut v).map_err(drv)?;
            let ones = vec![1f32; cap * p_bus];
            let mut v = st
                .pld_ones
                .try_slice_mut(0..ones.len())
                .ok_or_else(|| GpuError::Driver("pld ones slice".into()))?;
            e.stream.memcpy_htod(&ones, &mut v).map_err(drv)?;
        }
        if sel_k > 0 {
            // static mode-4 row table for topk_rows (see the gemma4 note)
            let st = df.state.as_mut().expect("just built");
            let host: Vec<u32> = (0..rows).flat_map(|_| [0u32, 0, 4, 0]).collect();
            let mut v = st
                .sel_params
                .try_slice_mut(0..host.len())
                .ok_or_else(|| GpuError::Driver("dflash selector params slice".into()))?;
            e.stream.memcpy_htod(&host, &mut v).map_err(drv)?;
        }
        Ok(())
    }

    /// Can the drafter legally draft for `slot` at position `p`?
    pub(crate) fn dflash_warm(&self, slot: usize, p: usize) -> bool {
        let Some(df) = self.dflash.as_ref() else {
            return false;
        };
        let Some(st) = df.state.as_ref() else {
            return false;
        };
        let Some(&(s, e)) = st.feat.get(slot) else {
            return false;
        };
        if e as usize != p {
            return false;
        }
        // Exact rule: the ring covers the drafter's whole SWA window (or the
        // sequence from 0). A prefix-restored slot starts its ring at the
        // resume position - the target KV is fully restored but features for
        // the hit span were never computed - so the exact rule keeps it cold
        // until p >= s + window (~never at 4k ctx). The relaxed arm accepts a
        // TRUNCATED window once the covered recent span reaches
        // PADDOCK_QWEN35_DFLASH_WARM_MIN tokens: draft quality degrades
        // gracefully (verify still gates every token), and whether the
        // acceptance holds is the A/B this dev knob exists for. Unset = exact.
        if (s as usize) <= p.saturating_sub(df.window) {
            return true;
        }
        static RELAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let relax = *RELAX.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_QWEN35_DFLASH_WARM_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(usize::MAX)
        });
        p.saturating_sub(s as usize) >= relax
    }

    /// Drop a slot's coverage (fresh sequence / release).
    pub(crate) fn dflash_clear_slot(&mut self, slot: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) {
            if let Some(f) = st.feat.get_mut(slot) {
                *f = (0, 0);
            }
            if let Some(c) = st.cov.get_mut(slot) {
                c.clear();
            }
        }
    }

    /// Cut a slot's coverage to the longest prefix of `tokens` the ring
    /// provably describes - what a prefix restore leaves valid.
    pub(crate) fn dflash_trim_slot(&mut self, slot: usize, tokens: &[u32]) {
        let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) else {
            return;
        };
        let (Some(&(s, e)), Some(cov)) = (st.feat.get(slot), st.cov.get(slot)) else {
            return;
        };
        let span = (e - s) as usize;
        let mut agree = 0usize;
        while agree < span
            && agree < cov.len()
            && (s as usize) + agree < tokens.len()
            && cov[agree] == tokens[(s as usize) + agree]
        {
            agree += 1;
        }
        if agree == 0 {
            self.dflash_clear_slot(slot);
            return;
        }
        st.feat[slot] = (s, s + agree as u32);
        st.cov[slot].truncate(agree);
    }

    /// Paged-stripe resume: the adopted pages carry the drafter's feature
    /// rows for [0..tokens.len()) (checkpoint in `dflash_cover`) - set the
    /// slot's host coverage to match so the block drafter resumes warm.
    pub(crate) fn dflash_restore_slot(&mut self, slot: usize, tokens: &[u32]) {
        let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) else {
            return;
        };
        if !st.paged {
            return;
        }
        if let (Some(f), Some(cov)) = (st.feat.get_mut(slot), st.cov.get_mut(slot)) {
            *f = (0, tokens.len() as u32);
            cov.clear();
            cov.extend_from_slice(tokens);
        }
    }

    /// Paged mode: refresh the drafter's block-table mirror from the host
    /// truth. Called at every append/draft entry after the caller's
    /// ensure_slot_blocks (so the host table is final for the rows about to
    /// be touched). The draft graphs bake this buffer's ADDRESS; content is
    /// all that moves. Dense mode: no-op (the static ring map is immutable).
    pub(crate) fn dflash_stage_tables(&mut self) -> Result<(), GpuError> {
        let this = &mut *self;
        let Some(bs) = this.batch.as_ref() else {
            return Ok(());
        };
        let Some(st) = this.dflash.as_mut().and_then(|d| d.state.as_mut()) else {
            return Ok(());
        };
        if !st.paged {
            return Ok(());
        }
        let n = bs.block_table_host.len().min(st.d_bt.len());
        let mut v = st
            .d_bt
            .try_slice_mut(0..n)
            .ok_or_else(|| GpuError::Driver("dflash d_bt slice".into()))?;
        this.exec
            .stream
            .memcpy_htod(&bs.block_table_host[..n], &mut v)
            .map_err(drv)
    }

    /// Fuse + ring-append feature K/V for rows the walk just tapped.
    ///
    /// Contract: `tap_band` ran for every band over exactly these `r` rows;
    /// `positions`/`slots` are the rows' host mirrors and `d_pos`/`d_slots`
    /// their device twins (the caller's own row staging - qwen35's walks
    /// each stage their own, unlike gemma4's single pf pair). `spans = None`
    /// appends same-slot runs; `Some` appends only those ranges (the verify
    /// commit).
    pub(crate) fn dflash_append_features(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
        spans: Option<&[(usize, usize)]>,
    ) -> Result<(), GpuError> {
        let r = positions.len();
        assert_eq!(r, slots.len());
        assert_eq!(r, tokens.len(), "coverage mirror needs one token per row");
        if r == 0 || !self.dflash_armed() {
            return Ok(());
        }
        // paged: the append kernels read the mirror; make it current for the
        // rows' slots (their blocks are ensured by the caller's protocol)
        self.dflash_stage_tables()?;
        let exec = self.exec.clone();
        let embd = self.embd;
        let df = self.dflash.as_mut().expect("armed");
        let (n_kv, hd, eps, rope, window, block) =
            (df.n_kv, df.hd, df.eps, df.rope, df.window, df.block);
        let kv_dim = n_kv * hd;
        let DflashDrafter {
            layers,
            enc_norm,
            state,
            pld,
            ..
        } = df;
        let st = state.as_mut().expect("armed");
        if st.stale || r > st.cap {
            // Loud, not debug-gated: a wipe restarts every touched ring at
            // its CURRENT position, and the exact warm rule then keeps those
            // slots cold until p >= start + window (~2k tokens) - i.e. for
            // the rest of a normal request. This used to fire silently
            // whenever the mixed tick's soft row cap admitted one prompt tail
            // too many (8 x 1075 rows > the 8544-row fusion cap), which cost
            // a whole run several times its throughput; advance_chunks now
            // holds the tick under the cap, so any wipe left is worth a
            // line.
            tracing::warn!(
                "[dflash-append] WIPE r={r} stale={} cap={} - {} slot rings restart cold (slots {:?})",
                st.stale,
                st.cap,
                slots.len(),
                slots
            );
            st.stale = false;
            for &s in slots {
                if let Some(f) = st.feat.get_mut(s as usize) {
                    *f = (0, 0);
                }
                if let Some(c) = st.cov.get_mut(s as usize) {
                    c.clear();
                }
            }
            return Ok(());
        }

        // row mirrors for the ring rope/append (see ap_pos above)
        {
            let mut v = st
                .ap_pos
                .try_slice_mut(0..r)
                .ok_or_else(|| GpuError::Driver("dflash ap_pos slice".into()))?;
            exec.stream.memcpy_htod(positions, &mut v).map_err(drv)?;
            let mut v = st
                .ap_slots
                .try_slice_mut(0..r)
                .ok_or_else(|| GpuError::Driver("dflash ap_slots slice".into()))?;
            exec.stream.memcpy_htod(slots, &mut v).map_err(drv)?;
        }
        // z = enc_norm(fc(concat taps)); the fc sum landed in zacc mid-walk.
        // (PLD: zacc holds tap-major codes instead - the bus runs per layer.)
        if pld.is_none() {
            exec.rmsnorm_batch(&st.zacc, enc_norm, &mut st.ztmp, embd, eps, r)?;
            exec.quantize_q8(&st.ztmp, &mut st.tq, &mut st.ts, r * embd)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.ztmp, &mut st.tyq, embd, r)?;
            }
        }

        let keep = window + block;
        let runs: Vec<(usize, usize)> = match spans {
            Some(s) => s.to_vec(),
            None => {
                let mut runs = Vec::new();
                let mut i = 0;
                while i < r {
                    let mut j = i + 1;
                    while j < r && slots[j] == slots[i] && positions[j] == positions[j - 1] + 1 {
                        j += 1;
                    }
                    runs.push((i, j - i));
                    i = j;
                }
                runs
            }
        };
        let mut cuts: Vec<(usize, usize)> = Vec::new();
        for &(row, n) in &runs {
            let (off, len) = if n > keep {
                (row + n - keep, keep)
            } else {
                (row, n)
            };
            let mut o = off;
            while o < off + len {
                let l = (off + len - o).min(keep);
                cuts.push((o, l));
                o += l;
            }
        }

        // rung C: the conditioning fold - One fused
        // norm+rope+store launch per layer over the written rows replaces
        // rmsnorm + rope + 2 x cuts kv_append (~340 eager launches/round at
        // 32 live fold to ~12; both rivals condition in ~14 constant
        // launches). Pool bytes bit-identical (parity test in
        // gpu_spec_batch_kernels), and the fold norms only the written rows
        // where the chain normed all r. The row list is layer-invariant:
        // upload once. Old packs (or an hd > 256 drafter, the fused
        // kernel's shared-staging bound) keep the chain.
        let fused = exec.has_dflash_cond_append() && hd <= 256;
        let mut nw = 0usize;
        if fused {
            let mut rows_w: Vec<u32> = Vec::new();
            for &(off, len) in &cuts {
                rows_w.extend(off as u32..(off + len) as u32);
            }
            nw = rows_w.len();
            let mut v = st
                .ap_rows
                .try_slice_mut(0..nw)
                .ok_or_else(|| GpuError::Driver("dflash ap_rows slice".into()))?;
            exec.stream.memcpy_htod(&rows_w, &mut v).map_err(drv)?;
        }
        for (li, layer) in layers.iter().enumerate() {
            if let Some(p) = pld.as_ref() {
                // PLD bus: acc = silu(sum_t W_{li,t} codes_t + b_li), then
                // K/V from the dedicated context projections.
                let nb = r * p.bus;
                {
                    let boff = li * st.cap * p.bus;
                    let src = st
                        .pld_bias
                        .try_slice(boff..boff + nb)
                        .ok_or_else(|| GpuError::Driver("pld bias slice".into()))?;
                    let mut dst = st
                        .pld_acc
                        .try_slice_mut(0..nb)
                        .ok_or_else(|| GpuError::Driver("pld acc slice".into()))?;
                    exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
                }
                for t in 0..p.pca.len() {
                    let n = r * p.rf;
                    let off = t * st.cap * p.rf;
                    {
                        let src = st
                            .zacc
                            .try_slice(off..off + n)
                            .ok_or_else(|| GpuError::Driver("pld code slice".into()))?;
                        let mut dst = st
                            .pld_stage
                            .try_slice_mut(0..n)
                            .ok_or_else(|| GpuError::Driver("pld stage slice".into()))?;
                        exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
                    }
                    exec.quantize_q8(&st.pld_stage, &mut st.tq, &mut st.ts, n)?;
                    if r > 64 {
                        exec.quantize_q8_mmq(&st.pld_stage, &mut st.tyq, p.rf, r)?;
                    }
                    prefill_mm_pre_any(
                        &exec,
                        &p.bus_w[li][t],
                        &st.tq,
                        &st.ts,
                        &st.tyq,
                        &mut st.txs,
                        &mut st.tss,
                        &mut st.skfix,
                        &mut st.pld_band,
                        r,
                    )
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                    exec.add(&mut st.pld_acc, &st.pld_band, nb)?;
                }
                exec.swiglu(&mut st.pld_acc, &st.pld_ones, nb)?;
                exec.quantize_q8(&st.pld_acc, &mut st.tq, &mut st.ts, nb)?;
                if r > 64 {
                    exec.quantize_q8_mmq(&st.pld_acc, &mut st.tyq, p.bus, r)?;
                }
                prefill_mm_pre_any(
                    &exec,
                    &p.kt[li],
                    &st.tq,
                    &st.ts,
                    &st.tyq,
                    &mut st.txs,
                    &mut st.tss,
                    &mut st.skfix,
                    &mut st.fk,
                    r,
                )
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            } else {
                prefill_mm_pre_any(
                    &exec,
                    &layer.wk,
                    &st.tq,
                    &st.ts,
                    &st.tyq,
                    &mut st.txs,
                    &mut st.tss,
                    &mut st.skfix,
                    &mut st.fk,
                    r,
                )
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            if !fused {
                exec.rmsnorm_batch(&st.fk, &layer.k_norm, &mut st.fkn, hd, eps, r * n_kv)?;
                exec.rope_yarn_batch(&mut st.fkn, &st.ap_pos, n_kv, hd, rope, r)?;
            }
            prefill_mm_pre_any(
                &exec,
                pld.as_ref().map_or(&layer.wv, |p| &p.vt[li]),
                &st.tq,
                &st.ts,
                &st.tyq,
                &mut st.txs,
                &mut st.tss,
                &mut st.skfix,
                &mut st.fv,
                r,
            )
            .map_err(|e| GpuError::Driver(e.to_string()))?;
            let (pool_k, pool_v) = &mut st.kv[li];
            if fused {
                exec.dflash_cond_append(
                    &st.fk,
                    &st.fv,
                    &layer.k_norm,
                    pool_k,
                    pool_v,
                    &st.ap_rows,
                    &st.ap_pos,
                    &st.ap_slots,
                    &st.d_bt,
                    st.bps,
                    n_kv,
                    hd,
                    eps,
                    rope,
                    nw,
                    r * n_kv,
                )?;
                continue;
            }
            for &(off, len) in &cuts {
                exec.kv_append_batch_paged_rows(
                    &st.fkn,
                    pool_k,
                    &st.ap_pos,
                    Some(&st.ap_slots),
                    &st.d_bt,
                    st.bps,
                    kv_dim,
                    off,
                    len,
                    KvDtype::Fp16,
                )?;
                exec.kv_append_batch_paged_rows(
                    &st.fv,
                    pool_v,
                    &st.ap_pos,
                    Some(&st.ap_slots),
                    &st.d_bt,
                    st.bps,
                    kv_dim,
                    off,
                    len,
                    KvDtype::Fp16,
                )?;
            }
        }

        for &(row, len) in &runs {
            let slot = slots[row] as usize;
            let (p0, p1) = (positions[row], positions[row] + len as u32);
            let new_start = if len > keep { p1 - keep as u32 } else { p0 };
            let Some(&(s, e)) = st.feat.get(slot) else {
                continue;
            };
            let extend = p0 <= e && new_start <= e && new_start >= s;
            let cov = &mut st.cov[slot];
            let rows = &tokens[row + (new_start - p0) as usize..row + len];
            if extend {
                st.feat[slot] = (s, p1.max(e));
                let base = (new_start - s) as usize;
                cov.resize(base.max(cov.len()), 0);
                for (i, &t) in rows.iter().enumerate() {
                    match base + i < cov.len() {
                        true => cov[base + i] = t,
                        false => cov.push(t),
                    }
                }
                cov.truncate((p1.max(e) - s) as usize);
            } else {
                st.feat[slot] = (new_start, p1);
                cov.clear();
                cov.extend_from_slice(rows);
            }
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[dflash-append] r={r} spans={} runs={} pos0={} posN={} feat={:?}",
                spans.is_some(),
                runs.len(),
                positions[0],
                positions[r - 1],
                st.feat
            );
        }
        Ok(())
    }

    /// Fuse+append the unified tick's stashed row mirrors (see
    /// `dflash_pending_append` - the walk holds field borrows, so the caller
    /// flushes right after the core returns; the append kernels enqueue
    /// behind the walk on the same stream).
    pub(crate) fn dflash_flush_pending_append(&mut self) -> Result<(), GpuError> {
        if let Some((toks, pos, slots)) = self.dflash_pending_append.take() {
            self.dflash_append_features(&toks, &pos, &slots, None)?;
        }
        Ok(())
    }

    /// Post-verify commit: replay the accept rule on this round's picks and
    /// ring-append only the accepted rows' features (`padded` = the verify's
    /// actual block-major rows, k1 wide per block; `committed[i]` = accepted
    /// count per block, 1..=k1 - the same values the round committed).
    pub(crate) fn dflash_spec_commit(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        padded: &[u32],
        committed: &[u32],
        k1: usize,
    ) -> Result<(), GpuError> {
        if !self.dflash_armed() {
            return Ok(());
        }
        // PADDOCK_SPEC_DEBUG: live acceptance histogram - committed[i] is
        // accepted drafts + 1 for slot i this round. Printed every 500
        // slot-rounds; the whole diagnosis of a drafter lives in this shape.
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static HIST: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
            static N: AtomicU64 = AtomicU64::new(0);
            for &c in committed {
                HIST[(c as usize).min(8)].fetch_add(1, Ordering::Relaxed);
            }
            let n = N.fetch_add(committed.len() as u64, Ordering::Relaxed) + committed.len() as u64;
            if n % 500 < committed.len() as u64 {
                let h: Vec<u64> = HIST.iter().map(|a| a.load(Ordering::Relaxed)).collect();
                let tot: u64 = h.iter().sum::<u64>().max(1);
                let mean: f64 = h
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| i as f64 * v as f64)
                    .sum::<f64>()
                    / tot as f64;
                eprintln!("[dflash-acc] rounds={tot} committed-hist={h:?} mean={mean:.2}");
            }
        }
        debug_assert_eq!(padded.len(), reqs.len() * k1);
        let mut positions = Vec::with_capacity(padded.len());
        let mut slots = Vec::with_capacity(padded.len());
        let mut spans = Vec::with_capacity(reqs.len());
        for (i, (slot, start, _)) in reqs.iter().enumerate() {
            spans.push((i * k1, committed[i] as usize));
            for j in 0..k1 {
                positions.push((*start + j) as u32);
                slots.push(*slot as u32);
            }
        }
        self.dflash_append_features(padded, &positions, &slots, Some(&spans))
    }

    /// Warmth for the service's spec gate. `want_pos` is the last COMMITTED
    /// position (service passes s.pos - 1), so the ring must cover through
    /// it: end == want_pos + 1. The one structural ramp: a token emitted by
    /// a span finisher has no feature until it walks as an input row, so the
    /// slot stays cold for exactly one classic decode tick after prefill
    /// (the tick's append supplies it) - which is why the decode PIPE is
    /// disabled while a DFlash drafter is attached (pipe ticks are device-
    /// driven and never append; measured as a permanent cold-stall, c1 44.1).
    pub(crate) fn dflash_ensure_warm(&self, slot: usize, want_pos: u32) -> bool {
        let ok = self.dflash_armed() && self.dflash_warm(slot, want_pos as usize + 1);
        if !ok && paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let feat = self
                .dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .and_then(|st| st.feat.get(slot).copied());
            tracing::info!(
                "[dflash-warm] slot={slot} want={want_pos} armed={} feat={feat:?}",
                self.dflash_armed()
            );
        }
        ok
    }

    /// Model-side drafts for one serving spec round: one DFlash forward
    /// covers every warm slot with k drafts each. `None` = nothing draftable.
    pub(crate) fn dflash_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuError> {
        if !self.dflash_armed() || k == 0 {
            return Ok(None);
        }
        let (block, feat) = {
            let df = self.dflash.as_ref().expect("armed");
            (df.block, df.state.as_ref().expect("armed").feat.clone())
        };
        let cap = self.dflash_round_cap();
        // RUNTIME block = k drafts + the committed anchor, not the trained
        // block_size (the muse/llama.cpp precedent - building all 8 rows to
        // use k+1 of them pays the whole head for nothing).
        let rows = k.min(block - 1) + 1;
        let mut reqs: Vec<(usize, usize, u32)> = Vec::new();
        let mut which: Vec<usize> = Vec::new();
        for (i, &(slot, tok)) in pendings.iter().enumerate() {
            let Some(&(_, e)) = feat.get(slot) else {
                continue;
            };
            let p = e as usize;
            if reqs.len() < cap && p + rows <= self.max_ctx && self.dflash_warm(slot, p) {
                reqs.push((slot, p, tok));
                which.push(i);
            }
        }
        if reqs.is_empty() {
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[dflash-draft] pendings={} eligible=0 k={k}",
                    pendings.len()
                );
            }
            return Ok(None);
        }
        // Paged stripe: the draft body appends its own rows at p..p+rows
        // through the mirror table - back those positions with real blocks
        // and refresh the mirror before the graph replays (the verify's own
        // ensure runs too late for the draft's writes; same rule as the MTP
        // chain's pre-draft ensure).
        if self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.paged)
        {
            for &(slot, p, _) in &reqs {
                self.ensure_slot_blocks(slot, p + rows)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            self.dflash_stage_tables()?;
        }
        let Some(blocks) = self.dflash_draft_blocks(&reqs, rows)? else {
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!("[dflash-draft] round DECLINED n={} rows={rows}", reqs.len());
            }
            return Ok(None);
        };
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[dflash-draft] pendings={} eligible={} k={k} rows={rows} cap={cap} first={:?}",
                pendings.len(),
                reqs.len(),
                blocks.first().map(|b: &Vec<u32>| b.as_slice())
            );
        }
        let mut out = vec![Vec::new(); pendings.len()];
        for (w, b) in which.into_iter().zip(blocks) {
            debug_assert_eq!(b.len(), rows - 1, "one draft per noise row");
            out[w] = b;
        }
        Ok(Some(out))
    }

    /// One draft round: per (slot, p, committed) request, one forward over
    /// [committed, (rows-1) x mask] rows at positions p..p+rows.
    pub(crate) fn dflash_draft_blocks(
        &mut self,
        reqs: &[(usize, usize, u32)],
        rows: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuError> {
        if !self.dflash_draft_launch(reqs, rows)? {
            return Ok(None);
        }
        let n = reqs.len();
        let r = n * rows;
        let exec = self.exec.clone();
        let st = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .expect("armed");
        let v = st
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("dflash d_out slice".into()))?;
        let picks = exec.stream.clone_dtoh(&v).map_err(drv)?;
        Ok(Some(
            (0..n)
                .map(|b| picks[b * rows + 1..(b + 1) * rows].to_vec())
                .collect(),
        ))
    }

    /// Stage + launch the (n, rows) draft graph without the picks readback -
    /// picks land in `st.d_out` ([n, rows] row-major, pick j of block b at
    /// [b*rows + 1 + j]). The async round (spec_draft_begin's dflash arm)
    /// copies them device-side into the chain's d_draft and reads back only
    /// post-verify; the sync round (dflash_draft_blocks) dtoh's right here.
    /// `Ok(false)` = ineligible (caller declines or falls back).
    /// How many requests one draft round can take - the round buffers are
    /// sized to this at arm time (`rows = max_blocks().min(slots) * block`).
    /// Deliberately cheap: callers must screen `n` against it before paying
    /// any staging work. Rung A: `dflash_draft_begin` used to reach
    /// `dflash_draft_launch`'s own reject only after ensure_serve_spec +
    /// per-slot ensure_slot_blocks + stage_tables + spec_set_live had run -
    /// a declining width round paid that every TICK (the w32k7 probe
    /// measured c32 at 37 t/s from exactly this arm-and-decline loop).
    /// Rung G: the sampled selector walk + rejection-sampling verify arm is
    /// live - a DFlash2 drafter with its selector, the pack's slots 470/471,
    /// and neither kill switch. The service reads this through
    /// `supports_spec_rs`/`supports_spec_rs_trunc`.
    pub(crate) fn dflash_rs_available(&self) -> bool {
        self.dflash.as_ref().is_some_and(|d| d.selector.is_some())
            && !sel_off()
            && !rs_off()
            && self.exec.has_dflash_rs()
    }

    /// Rung G: keep the service's per-slot chain draws for the round about
    /// to be drafted (`Generator::spec_rs_stash`).
    pub(crate) fn spec_rs_stash_draws(&mut self, draws: Vec<crate::generator::SpecRsDraw>) {
        self.spec_rs_draws = Some(draws);
    }

    pub(crate) fn dflash_round_cap(&self) -> usize {
        let slots_max = self.batch.as_ref().map_or(1, |b| b.max_batch).max(1);
        max_blocks().min(slots_max)
    }

    pub(crate) fn dflash_draft_launch(
        &mut self,
        reqs: &[(usize, usize, u32)],
        rows: usize,
    ) -> Result<bool, GpuError> {
        if !self.dflash_armed() || !self.exec.has_argmax_rows() {
            return Ok(false);
        }
        let (block, mask) = {
            let df = self.dflash.as_ref().expect("armed");
            (df.block, df.mask_token)
        };
        let n = reqs.len();
        if n == 0 || rows < 2 || rows > block || n > self.dflash_round_cap() {
            return Ok(false);
        }
        for &(slot, p, _) in reqs {
            if p + rows > self.max_ctx || !self.dflash_warm(slot, p) {
                return Ok(false);
            }
        }

        let r = n * rows;
        let mut toks = Vec::with_capacity(r);
        let mut pos = Vec::with_capacity(r);
        let mut apos = Vec::with_capacity(r);
        let mut slots_v = Vec::with_capacity(r);
        for &(slot, p, committed) in reqs {
            toks.push(committed);
            toks.extend(std::iter::repeat_n(mask, rows - 1));
            pos.extend(p as u32..(p + rows) as u32);
            // non-causal block: every row's attention bound is the block end
            apos.extend(std::iter::repeat_n((p + rows - 1) as u32, rows));
            slots_v.extend(std::iter::repeat_n(slot as u32, rows));
        }
        {
            let exec = self.exec.clone();
            let st = self
                .dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("armed");
            let up = |host: &[u32], dst: &mut CudaSlice<u32>| -> Result<(), GpuError> {
                let mut v = dst
                    .try_slice_mut(0..host.len())
                    .ok_or_else(|| GpuError::Driver("dflash row stage".into()))?;
                exec.stream.memcpy_htod(host, &mut v).map_err(drv)
            };
            up(&toks, &mut st.d_toks)?;
            up(&pos, &mut st.d_pos)?;
            up(&apos, &mut st.d_apos)?;
            up(&slots_v, &mut st.d_slots)?;
        }
        // Rung G: per-block 1/T + seed for the sampled selector walk, from
        // the service's chain draws (stashed right before this round; absent
        // on the synchronous path and for greedy slots -> 0 = greedy walk,
        // q one-hot, which the resolve treats as the classic rule). The seed
        // is the slot's first draw's bits mixed with the slot id, so two
        // blocks of one round never share a Gumbel stream.
        let rs_live = self.dflash_rs_available();
        {
            let draws = self.spec_rs_draws.take().unwrap_or_default();
            let exec = self.exec.clone();
            let st = self
                .dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("armed");
            let mut invt = vec![0f32; n];
            let mut seed = vec![0u32; n];
            if rs_live {
                for (i, &(slot, _, _)) in reqs.iter().enumerate() {
                    if let Some(d) = draws.iter().find(|d| d.slot == slot) {
                        invt[i] = d.inv_t.max(0.0);
                        seed[i] = d.u.first().map_or(0, |u| u.to_bits())
                            ^ (slot as u32).wrapping_mul(0x9e37_79b9);
                    }
                }
            }
            let up = |host: &[u32], dst: &mut CudaSlice<u32>| -> Result<(), GpuError> {
                let mut v = dst
                    .try_slice_mut(0..host.len())
                    .ok_or_else(|| GpuError::Driver("dflash rs stage".into()))?;
                exec.stream.memcpy_htod(host, &mut v).map_err(drv)
            };
            {
                let mut v = st
                    .blk_invt
                    .try_slice_mut(0..n)
                    .ok_or_else(|| GpuError::Driver("dflash rs stage".into()))?;
                exec.stream.memcpy_htod(&invt, &mut v).map_err(drv)?;
            }
            up(&seed, &mut st.blk_seed)?;
        }
        self.spec_round_rs = rs_live;
        // the batched-runs row table for this block width (rung E2) - built
        // here, before the body runs or is captured: an allocation inside a
        // capture is illegal, and the graph bakes this pointer
        if !runs_off() && self.exec.kernels_pf_runs_available() {
            let cap = self.dflash_round_cap();
            let exec = self.exec.clone();
            let st = self
                .dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("armed");
            if let std::collections::hash_map::Entry::Vacant(e) = st.run_offs.entry(rows) {
                let host: Vec<u32> = (0..=cap).map(|i| (i * rows) as u32).collect();
                e.insert(exec.to_device_u32(&host)?);
            }
        }

        // One captured replay per (block count, rows) - see the gemma4 note.
        // Replays follow the CURRENT exec lane (launch_on) like the qwen35
        // spec graphs do, so a future overlapped composition stays legal.
        let have = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.graphs.contains_key(&(n, rows)));
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.dflash_draft_body(n, rows)?;
        } else if !have {
            let exec = self.exec.clone();
            self.dflash_draft_body(n, rows)?;
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("dflash pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
                .map_err(|e| GpuError::Driver(format!("dflash begin_capture: {e}")))?;
            let rec = self.dflash_draft_body(n, rows);
            let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("dflash end_capture: {e}")));
            rec?;
            let g = SendGraph(
                graph?
                    .ok_or_else(|| GpuError::Driver("dflash capture produced no graph".into()))?,
            );
            g.0.launch_on(&self.exec.stream)
                .map_err(|e| GpuError::Driver(format!("dflash first launch: {e}")))?;
            self.dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("armed")
                .graphs
                .insert((n, rows), g);
        } else {
            self.dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("armed")
                .graphs[&(n, rows)]
                .0
                .launch_on(&self.exec.stream)
                .map_err(|e| GpuError::Driver(format!("dflash draft graph launch: {e}")))?;
        }
        let _ = r; // row count consumed by the caller's readback / pick copy
        Ok(true)
    }

    /// The draft round's device body (capture-safe): embed the staged rows,
    /// five drafter layers with transient block appends + ring attention,
    /// drafter final norm, TARGET head, selector or device argmax.
    fn dflash_draft_body(&mut self, n: usize, rows: usize) -> Result<(), GpuError> {
        let exec = self.exec.clone();
        let embd = self.embd;
        let vocab = self.vocab;
        let r = n * rows;
        let tok_embd = &self.tok_embd;
        let output = &self.output;
        let out_f8 = self.out_f8.as_ref();
        let df = self.dflash.as_mut().expect("armed");
        // PLD: layers run at the drafter's narrow width; only the embed
        // gather (and its in_proj entry) sees the target width.
        let embd_t = embd;
        let embd = df.pld.as_ref().map_or(embd, |p| p.d);
        let (n_heads, n_kv, hd, eps, rope, window) =
            (df.n_heads, df.n_kv, df.hd, df.eps, df.rope, df.window);
        let (q_dim, kv_dim) = (n_heads * hd, n_kv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let conv = if conv_off() { None } else { df.conv };
        let (logit_scale, softcap) = (df.logit_scale, df.softcap);
        let use_sel = !sel_off();
        let DflashDrafter {
            layers,
            final_norm,
            state,
            selector,
            pld,
            ..
        } = df;
        let pld = pld.as_ref();
        let sel = use_sel
            .then_some(())
            .and(selector.as_ref().map(|(s, rk, k)| (s, *rk, *k)));
        let st = state.as_mut().expect("armed");
        let ff = match &layers[0].w_gate {
            QuantW::Q8(w) => w.dims[1],
            QuantW::Kq(w) => w.dims[1],
        };

        // RAW embedding rows from the target's table (plain gather - the
        // drafter trained against ggml_get_rows with no scale, and qwen35's
        // own embeddings are unscaled).
        if let Some(p) = pld {
            // PLD: gather at target width, then enter the narrow block
            embed_any(&exec, tok_embd, &st.d_toks, &mut st.pld_emb, embd_t, r)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.quantize_q8(&st.pld_emb, &mut st.xq, &mut st.xs, r * embd_t)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.pld_emb, &mut st.yq, embd_t, r)?;
            }
            draft_mm(
                &exec,
                &p.in_proj,
                None,
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.x,
                r,
            )?;
        } else {
            embed_any(&exec, tok_embd, &st.d_toks, &mut st.x, embd, r)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }

        for (li, layer) in layers.iter().enumerate() {
            let lf8 = layer.f8.as_ref();
            exec.rmsnorm_batch(&st.x, &layer.attn_norm, &mut st.xn, embd, eps, r)?;
            exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
            if lf8.is_some()
                || layer
                    .attn_conv
                    .as_ref()
                    .is_some_and(|c| c.proj_f8.is_some())
                || layer.ffn_conv.as_ref().is_some_and(|c| c.proj_f8.is_some())
            {
                exec.quantize_e4m3_row(&st.xn, &mut st.f8q, &mut st.f8rs, embd, r)?;
            }
            if r > 64 {
                exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
            }
            // DFlash2 `prepare`: projection off the sublayer INPUT; side 0
            // convolves the same input into what attention sees; side 1
            // stays in `cvc` for `finish`.
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.attn_conv) {
                draft_mm(
                    &exec,
                    &cv.proj,
                    cv.proj_f8.as_ref(),
                    &st.f8q,
                    &st.f8rs,
                    &st.xq,
                    &st.xs,
                    &st.yq,
                    &mut st.xsums,
                    &mut st.ssums,
                    &mut st.part,
                    &mut st.skfix,
                    &mut st.cvc,
                    r,
                )?;
                exec.dflash_conv(
                    &st.xn,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    0,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.quantize_q8(&st.cvx, &mut st.xq, &mut st.xs, r * embd)?;
                if lf8.is_some() {
                    exec.quantize_e4m3_row(&st.cvx, &mut st.f8q, &mut st.f8rs, embd, r)?;
                }
                if r > 64 {
                    exec.quantize_q8_mmq(&st.cvx, &mut st.yq, embd, r)?;
                }
            }
            for (w, wf8, dst) in [
                (
                    &layer.wq,
                    lf8.map(|f| &f.wq),
                    &mut st.q as *mut CudaSlice<f32>,
                ),
                (
                    &layer.wk,
                    lf8.map(|f| &f.wk),
                    &mut st.k as *mut CudaSlice<f32>,
                ),
                (
                    &layer.wv,
                    lf8.map(|f| &f.wv),
                    &mut st.v as *mut CudaSlice<f32>,
                ),
            ] {
                // SAFETY: three distinct destination fields; raw pointers only
                // keep one loop over three planes with `st` borrowed for the
                // shared staging.
                let dst = unsafe { &mut *dst };
                draft_mm(
                    &exec,
                    w,
                    wf8,
                    &st.f8q,
                    &st.f8rs,
                    &st.xq,
                    &st.xs,
                    &st.yq,
                    &mut st.xsums,
                    &mut st.ssums,
                    &mut st.part,
                    &mut st.skfix,
                    dst,
                    r,
                )?;
            }
            exec.rmsnorm_batch(&st.q, &layer.q_norm, &mut st.qn, hd, eps, r * n_heads)?;
            exec.rmsnorm_batch(&st.k, &layer.k_norm, &mut st.kn, hd, eps, r * n_kv)?;
            exec.rope_yarn_batch(&mut st.qn, &st.d_pos, n_heads, hd, rope, r)?;
            exec.rope_yarn_batch(&mut st.kn, &st.d_pos, n_kv, hd, rope, r)?;
            let (pool_k, pool_v) = &mut st.kv[li];
            exec.kv_append_batch_paged_rows(
                &st.kn,
                pool_k,
                &st.d_pos,
                Some(&st.d_slots),
                &st.d_bt,
                st.bps,
                kv_dim,
                0,
                r,
                KvDtype::Fp16,
            )?;
            exec.kv_append_batch_paged_rows(
                &st.v,
                pool_v,
                &st.d_pos,
                Some(&st.d_slots),
                &st.d_bt,
                st.bps,
                kv_dim,
                0,
                r,
                KvDtype::Fp16,
            )?;
            // prefill (WMMA) class over the 2048-key window; `d_apos` (block
            // end) is what makes it non-causal.
            // One batched-runs launch - grid.z indexes the per-block row
            // table, each z re-aims q/out/positions/slots at its block and
            // reads that block's slot (the v4 hd128 arm's run prologue, same
            // shape muse uses). Same kernel, same tile per block =
            // bit-identical to the per-block loop it replaces; what it buys
            // is 5 launches per round instead of 5 x blocks - 160 at 32 live,
            // each an 8-CTA grid that fills nothing, and together ~16% of the
            // GPU. The loop stays as the fallback for packs without the runs
            // slot and for the A/B.
            if let Some(offs) = st.run_offs.get(&rows) {
                exec.pf_runs_register(Some((offs, n as u32, rows as u32)))?;
                let res = exec.attn_prefill_f16_rows_paged(
                    &st.qn,
                    pool_k,
                    pool_v,
                    &st.sinks,
                    &mut st.attn,
                    &st.d_apos,
                    &st.d_slots,
                    &st.d_bt,
                    st.bps,
                    n_heads,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    0,
                    r,
                    scale,
                    KvDtype::Fp16,
                );
                exec.pf_runs_register(None)?;
                res?;
            } else {
                for b in 0..n {
                    exec.attn_prefill_f16_rows_paged(
                        &st.qn,
                        pool_k,
                        pool_v,
                        &st.sinks,
                        &mut st.attn,
                        &st.d_apos,
                        &st.d_slots,
                        &st.d_bt,
                        st.bps,
                        n_heads,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        b * rows,
                        rows,
                        scale,
                        KvDtype::Fp16,
                    )?;
                }
            }
            exec.quantize_q8(&st.attn, &mut st.xq, &mut st.xs, r * q_dim)?;
            if lf8.is_some() {
                exec.quantize_e4m3_row(&st.attn, &mut st.f8q, &mut st.f8rs, q_dim, r)?;
            }
            if r > 64 {
                exec.quantize_q8_mmq(&st.attn, &mut st.yq, q_dim, r)?;
            }
            draft_mm(
                &exec,
                &layer.wo,
                lf8.map(|f| &f.wo),
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.proj,
                r,
            )?;
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.attn_conv) {
                exec.dflash_conv(
                    &st.proj,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    1,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.add(&mut st.x, &st.cvx, r * embd)?;
            } else {
                exec.add(&mut st.x, &st.proj, r * embd)?;
            }

            exec.rmsnorm_batch(&st.x, &layer.ffn_norm, &mut st.xn, embd, eps, r)?;
            exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
            if lf8.is_some()
                || layer
                    .attn_conv
                    .as_ref()
                    .is_some_and(|c| c.proj_f8.is_some())
                || layer.ffn_conv.as_ref().is_some_and(|c| c.proj_f8.is_some())
            {
                exec.quantize_e4m3_row(&st.xn, &mut st.f8q, &mut st.f8rs, embd, r)?;
            }
            if r > 64 {
                exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
            }
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.ffn_conv) {
                draft_mm(
                    &exec,
                    &cv.proj,
                    cv.proj_f8.as_ref(),
                    &st.f8q,
                    &st.f8rs,
                    &st.xq,
                    &st.xs,
                    &st.yq,
                    &mut st.xsums,
                    &mut st.ssums,
                    &mut st.part,
                    &mut st.skfix,
                    &mut st.cvc,
                    r,
                )?;
                exec.dflash_conv(
                    &st.xn,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    0,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.quantize_q8(&st.cvx, &mut st.xq, &mut st.xs, r * embd)?;
                if lf8.is_some() {
                    exec.quantize_e4m3_row(&st.cvx, &mut st.f8q, &mut st.f8rs, embd, r)?;
                }
                if r > 64 {
                    exec.quantize_q8_mmq(&st.cvx, &mut st.yq, embd, r)?;
                }
            }
            draft_mm(
                &exec,
                &layer.w_gate,
                lf8.map(|f| &f.w_gate),
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.ffn_gate,
                r,
            )?;
            draft_mm(
                &exec,
                &layer.w_up,
                lf8.map(|f| &f.w_up),
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.ffn_up,
                r,
            )?;
            exec.swiglu(&mut st.ffn_gate, &st.ffn_up, r * ff)?;
            exec.quantize_q8(&st.ffn_gate, &mut st.xq, &mut st.xs, r * ff)?;
            if lf8.is_some() {
                exec.quantize_e4m3_row(&st.ffn_gate, &mut st.f8q, &mut st.f8rs, ff, r)?;
            }
            if r > 64 {
                exec.quantize_q8_mmq(&st.ffn_gate, &mut st.yq, ff, r)?;
            }
            draft_mm(
                &exec,
                &layer.w_down,
                lf8.map(|f| &f.w_down),
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.proj,
                r,
            )?;
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.ffn_conv) {
                exec.dflash_conv(
                    &st.proj,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    1,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.add(&mut st.x, &st.cvx, r * embd)?;
            } else {
                exec.add(&mut st.x, &st.proj, r * embd)?;
            }
        }

        // drafter final norm -> TARGET head. The head is the round's bulk
        // (full lm_head read regardless of rows); ride the f8d plane at the
        // same b>=8 boundary every qwen35 head site uses. Drafts are greedy
        // picks, so the (absent) epilogue stays argmax-invariant; the
        // selector applies scale-then-cap itself because path scores ADD.
        exec.rmsnorm_batch(&st.x, final_norm, &mut st.xn, embd, eps, r)?;
        if let Some(p) = pld {
            // PLD lattice head: W_h then the rank-r codebook scan - same
            // [rows, vocab] logits plane, so the greedy pick is unchanged.
            exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
            }
            draft_mm(
                &exec,
                &p.head,
                None,
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.pld_u,
                r,
            )?;
            exec.quantize_q8(&st.pld_u, &mut st.xq, &mut st.xs, r * p.rank)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.pld_u, &mut st.yq, p.rank, r)?;
            }
            draft_mm(
                &exec,
                &p.codebook,
                None,
                &st.f8q,
                &st.f8rs,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.logits,
                r,
            )?;
            return exec.argmax_rows(&st.logits, &mut st.d_out, r, vocab);
        }
        // was a bare `r >= 8` -- the drafter's TARGET head, so under the
        // REPLACE lane r < 8 would read the dropped Q8_0 plane.
        if let Some((p8, pi, po)) = super::head_f8(out_f8, r) {
            exec.quantize_e4m3(&st.xn, &mut st.e4q, &mut st.e4rs, r * pi)?;
            exec.f8d_gemm_mma_ks(
                p8,
                *pi,
                *po,
                &st.e4q,
                &st.e4rs,
                &mut st.part,
                &mut st.logits,
                r,
            )?;
        } else {
            // this fn returns GpuError, not GpuModelError
            super::stub_guard(output, "dflash.rs drafter target head")
                .map_err(|e| GpuError::Unsupported(e.to_string()))?;
            mmq(
                &exec,
                output,
                &st.xn,
                &mut st.xq,
                &mut st.xs,
                &mut st.ssums,
                &mut st.part,
                &mut st.logits,
                r,
            )
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        let Some((sc, rank, top_k)) = sel else {
            return exec.argmax_rows(&st.logits, &mut st.d_out, r, vocab);
        };
        let n = r / rows;
        exec.topk_rows(
            &st.logits,
            &st.sel_params,
            &mut st.sel_topk,
            r,
            vocab,
            top_k,
        )?;
        exec.dflash_cand_ids(
            &st.sel_topk,
            &st.d_toks,
            &mut st.sel_ids,
            top_k,
            rows,
            r,
            vocab,
        )?;
        let (QuantW::Kq(pw), QuantW::Kq(sw)) = (&sc.pred, &sc.succ) else {
            unreachable!("selector codebooks are k-quant - attach refuses anything else")
        };
        exec.kquant_gather(pw, &st.sel_ids, &mut st.sel_pred, rank, r * top_k + n)?;
        if pld.is_some() {
            // hybrid selector (sel_from_pos = 1): the anchor carries no path
            // information at j=0 - zero its pred codes (the n tail entries)
            // so row 0 walks pure unary = top-1, matching training.
            let nz = n * rank;
            let src = st
                .pld_zero
                .try_slice(0..nz)
                .ok_or_else(|| GpuError::Driver("pld zero src".into()))?;
            let mut dst = st
                .sel_pred
                .try_slice_mut(r * top_k * rank..r * top_k * rank + nz)
                .ok_or_else(|| GpuError::Driver("pld selpred tail".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
        }
        exec.kquant_gather(sw, &st.sel_ids, &mut st.sel_succ, rank, r * top_k)?;
        exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
        if r > 64 {
            exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
        }
        if sc.hidden_f8.is_some() {
            exec.quantize_e4m3_row(&st.xn, &mut st.f8q, &mut st.f8rs, embd, r)?;
        }
        draft_mm(
            &exec,
            &sc.hidden,
            sc.hidden_f8.as_ref(),
            &st.f8q,
            &st.f8rs,
            &st.xq,
            &st.xs,
            &st.yq,
            &mut st.xsums,
            &mut st.ssums,
            &mut st.part,
            &mut st.skfix,
            &mut st.sel_hs,
            r,
        )?;
        if !rs_off() && exec.has_dflash_rs() {
            // rung G: the walk samples at the block's 1/T (0 = greedy) and
            // leaves the row's K-way q behind for the verify's rejection
            // sampler; process-constant election, baked into the graphs
            return exec.dflash_select_rs(
                &st.sel_topk,
                &st.sel_pred,
                &st.sel_succ,
                &st.sel_hs,
                &st.blk_invt,
                &st.blk_seed,
                &mut st.d_out,
                &mut st.q16,
                logit_scale,
                softcap,
                rank,
                top_k,
                rows,
                r,
            );
        }
        exec.dflash_select(
            &st.sel_topk,
            &st.sel_pred,
            &st.sel_succ,
            &st.sel_hs,
            &mut st.d_out,
            logit_scale,
            softcap,
            rank,
            top_k,
            rows,
            r,
        )
    }

    /// Synthetic end-to-end smoke (bring-up gate): seed slot 0's ring with
    /// deterministic pseudo-features past the wrap point, draft twice
    /// (equality catches append races), time the eager round.
    pub fn dflash_selftest(&mut self) -> Result<DflashSelftest, GpuError> {
        if self.dflash.is_none() {
            return Err(GpuError::Driver("dflash: not attached".into()));
        }
        self.dflash_ensure_state()?;
        let embd = self.embd;
        let (window, block) = {
            let d = self.dflash.as_ref().expect("attached");
            (d.window, d.block)
        };
        let n = (window + block + 64).min(
            self.dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("armed")
                .cap,
        );

        // deterministic xorshift features
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
        };
        let host: Vec<f32> = (0..n * embd).map(|_| next()).collect();
        let positions: Vec<u32> = (0..n as u32).collect();
        let slots = vec![0u32; n];
        // Stage the pseudo-residual + row mirrors in the DRAFTER's own row
        // buffers: n can exceed the target's scratch rows, and the selftest
        // has no live walk to borrow staging from. d_pos/d_slots are sized
        // `rows` (small), so the tap/append below uses two throwaway device
        // uploads instead.
        let exec = self.exec.clone();
        self.ensure_scratch(1)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        {
            let sc = self.scratch.as_mut().expect("scratch");
            if sc.d_x.len() < n * embd {
                self.ensure_scratch(n.div_ceil(64) * 64)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
        }
        {
            let sc = self.scratch.as_mut().expect("scratch");
            let mut dst = sc
                .d_x
                .try_slice_mut(0..n * embd)
                .ok_or_else(|| GpuError::Driver("selftest d_x slice".into()))?;
            exec.stream.memcpy_htod(&host, &mut dst).map_err(drv)?;
        }
        // target_layers, not fc_bands: PLD has no fc bands (its taps are PCA
        // projectors) and a zero-band selftest never exercises the tap path.
        let n_taps = self.dflash.as_ref().expect("attached").target_layers.len();
        for band in 0..n_taps {
            let exec = self.exec.clone();
            let embd = self.embd;
            let sc_x = {
                let sc = self.scratch.as_ref().expect("scratch");
                sc.d_x.clone()
            };
            let df = self.dflash.as_mut().expect("attached");
            tap_band(&exec, df, &sc_x, band, embd, n)?;
        }
        let stoks: Vec<u32> = (0..n as u32).collect();
        self.dflash_append_features(&stoks, &positions, &slots, None)?;

        let reqs = [(0usize, n, 1u32)];
        let rows = std::env::var("DFLASH_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|r: &usize| (2..=block).contains(r))
            .unwrap_or(block);
        let d1 = self
            .dflash_draft_blocks(&reqs, rows)?
            .ok_or_else(|| GpuError::Driver("dflash selftest: draft declined".into()))?;
        let d2 = self
            .dflash_draft_blocks(&reqs, rows)?
            .ok_or_else(|| GpuError::Driver("dflash selftest: repeat declined".into()))?;
        let vocab = self.vocab as u32;
        if d1[0].iter().any(|&t| t >= vocab) {
            return Err(GpuError::Driver(format!(
                "dflash selftest: draft out of vocab: {:?}",
                d1[0]
            )));
        }
        self.exec.synchronize()?;
        let t0 = std::time::Instant::now();
        let rounds = 50;
        for _ in 0..rounds {
            let _ = self.dflash_draft_blocks(&reqs, rows)?;
        }
        self.exec.synchronize()?;
        Ok(DflashSelftest {
            repeat_identical: d1 == d2,
            drafts: d1.into_iter().next().expect("one req"),
            ms_per_round: t0.elapsed().as_secs_f64() * 1e3 / rounds as f64,
        })
    }
}

/// One drafter GEMM over whichever quant ladder covers this round's width
/// (see the gemma4 original's ladder note - the split is drafter-safe
/// because drafts are re-derived by the verify).
#[allow(clippy::too_many_arguments)]
fn draft_mm(
    exec: &crate::gpu::GpuExecutor,
    w: &QuantW,
    f8: Option<&crate::gpu::F8RowPlane>,
    f8q: &CudaSlice<i8>,
    f8rs: &CudaSlice<f32>,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    // e4m3 twin first when it was built (see DflashLayerF8): same rows, same
    // destination, one arithmetic class up from W4A8 dp4a.
    if let Some(p) = f8 {
        let (in_dim, out_dim) = match w {
            QuantW::Q8(q) => (q.dims[0], q.dims[1]),
            QuantW::Kq(k) => (k.dims[0], k.dims[1]),
        };
        return exec.f8row_gemm(p, f8q, f8rs, y, in_dim, out_dim, r);
    }
    if r > 64 {
        prefill_mm_pre_any(exec, w, xq, xs, yq, xsums, ssums, skfix, y, r)
    } else {
        mmq_pre_any(exec, w, xq, xs, ssums, part, y, r)
    }
    .map_err(|e| GpuError::Driver(e.to_string()))
}

/// Whether the drafter's layer walk rides its e4m3 twins. Default on once
/// measured; PADDOCK_DFLASH_F8=0 restores the k-quant ladder.
pub(crate) fn dflash_f8_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        !matches!(
            paddock_models::dev_var!("PADDOCK_DFLASH_F8")
                .ok()
                .as_deref(),
            Some("0")
        )
    })
}

/// Build the seven e4m3 twins for one drafter layer. Returns None if any
/// projection cannot convert - the layer then keeps the k-quant ladder whole,
/// because a half-converted layer would run two numeric classes in one walk.
fn kq_to_f8row(
    exec: &crate::gpu::GpuExecutor,
    w: &QuantW,
    scratch: &mut CudaSlice<f32>,
    host: &mut Vec<u8>,
) -> Option<crate::gpu::F8RowPlane> {
    {
        let kq = match w {
            QuantW::Kq(k) => k,
            QuantW::Q8(_) => return None, // Q8 already has its own f8 route
        };
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let n = in_dim.checked_mul(out_dim)?;
        if n == 0 || n > scratch.len() {
            return None;
        }
        exec.kquant_dequant_rp(kq, scratch).ok()?;
        let f32s = exec.to_host_len(scratch, n).ok()?;
        host.clear();
        host.reserve(n * 2);
        for v in &f32s {
            // f32 -> bf16, round-to-nearest-even: bf16 keeps the exponent and
            // 8 mantissa bits, and e4m3 keeps 3, so nothing this drops can
            // reach the output. Little-endian, matching bf16_to_f8row's reader.
            let b = v.to_bits();
            let rounded = ((b >> 16)
                + (((b >> 15) & 1) & ((b & 0x7fff != 0) as u32 | ((b >> 16) & 1))))
                as u16;
            host.extend_from_slice(&rounded.to_le_bytes());
        }
        exec.bf16_to_f8row(host, in_dim, out_dim).ok()
    }
}

fn build_layer_f8(
    exec: &crate::gpu::GpuExecutor,
    l: &DflashLayer,
    scratch: &mut CudaSlice<f32>,
    host: &mut Vec<u8>,
) -> Option<DflashLayerF8> {
    Some(DflashLayerF8 {
        wq: kq_to_f8row(exec, &l.wq, scratch, host)?,
        wk: kq_to_f8row(exec, &l.wk, scratch, host)?,
        wv: kq_to_f8row(exec, &l.wv, scratch, host)?,
        wo: kq_to_f8row(exec, &l.wo, scratch, host)?,
        w_gate: kq_to_f8row(exec, &l.w_gate, scratch, host)?,
        w_up: kq_to_f8row(exec, &l.w_up, scratch, host)?,
        w_down: kq_to_f8row(exec, &l.w_down, scratch, host)?,
    })
}

/// Fold one tap band into the fusion accumulator, straight off the residual
/// rows (`x`) the layer walk is holding. Called inside captured bodies -
/// everything it touches is device-resident and shapes key on `r` alone.
/// `r > cap` marks the state stale; the append then clears touched slots.
pub(crate) fn tap_band(
    exec: &crate::gpu::GpuExecutor,
    df: &mut DflashDrafter,
    x: &CudaSlice<f32>,
    band: usize,
    embd: usize,
    r: usize,
) -> Result<(), GpuError> {
    let DflashDrafter {
        fc_bands,
        fc_f8,
        state,
        pld,
        ..
    } = df;
    let st = state.as_mut().expect("armed");
    if r > st.cap {
        st.stale = true;
        return Ok(());
    }
    exec.quantize_q8(x, &mut st.tq, &mut st.ts, r * embd)?;
    if r > 64 {
        exec.quantize_q8_mmq(x, &mut st.tyq, embd, r)?;
    }
    if let Some(p) = pld.as_ref() {
        // PLD: whitened rank-rf tap code (uncentered - the mean fold lives
        // in the bus bias), parked TAP-MAJOR in zacc [ntap, cap, rf] so the
        // append's per-tap bus bands read contiguous rows.
        prefill_mm_pre_any(
            exec,
            &p.pca[band],
            &st.tq,
            &st.ts,
            &st.tyq,
            &mut st.txs,
            &mut st.tss,
            &mut st.skfix,
            &mut st.ztmp,
            r,
        )
        .map_err(|e| GpuError::Driver(e.to_string()))?;
        let n = r * p.rf;
        let off = band * st.cap * p.rf;
        let src = st
            .ztmp
            .try_slice(0..n)
            .ok_or_else(|| GpuError::Driver("pld tap src slice".into()))?;
        let mut dst = st
            .zacc
            .try_slice_mut(off..off + n)
            .ok_or_else(|| GpuError::Driver("pld tap dst slice".into()))?;
        exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
        return Ok(());
    }
    let QuantW::Kq(kw) = &fc_bands[band] else {
        unreachable!("fc bands are k-quant - attach refuses anything else")
    };
    // e4m3 twin when it built AND the rowwise staging can hold this width
    // (tap_band runs up to st.cap rows; f8q is sized for the state's row cap).
    let fb8 = fc_f8
        .get(band)
        .and_then(|o| o.as_ref())
        .filter(|_| r * embd <= st.f8q.len() && r <= st.f8rs.len());
    let dst = if band == 0 {
        &mut st.zacc
    } else {
        &mut st.ztmp
    };
    if let Some(p) = fb8 {
        exec.quantize_e4m3_row(x, &mut st.f8q, &mut st.f8rs, embd, r)?;
        exec.f8row_gemm(p, &st.f8q, &st.f8rs, dst, embd, embd, r)?;
    } else {
        kq_mm_pre(
            exec,
            kw,
            &st.tq,
            &st.ts,
            &st.tyq,
            &mut st.txs,
            &mut st.tss,
            dst,
            r,
        )
        .map_err(|e| GpuError::Driver(e.to_string()))?;
    }
    if band > 0 {
        let (zacc, ztmp) = (&mut st.zacc, &st.ztmp);
        exec.add(zacc, ztmp, r * embd)?;
    }
    Ok(())
}
