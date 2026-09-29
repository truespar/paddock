//! Qwen3.8-Flash-Next forward graph - stage 3.
//!
//! Prompt-at-once (every token of the prompt in one pass per layer), which is
//! the shape the parity gate needs and the shape a chunked-prefill serving
//! lane grows from. What it is not yet: incremental decode off carried state,
//! CUDA-graph capture, continuous batching, or the QSA sparse walk - those are
//! the perf/serving rungs, each with its own gate.
//!
//! The gate this file answers to is `examples/q38fn_host_forward.rs`, the
//! host-exact forward that holds ARBITER parity (docs/qwen38-flash-next-
//! bringup.md stage 2). Every op here is either an existing pack lane
//! whose semantics were source-verified for this family, or one of the new
//! `q4x_*` slots - no formula is re-derived in this file.
//!
//! PLE residency: the 51.2 GB n-gram table is DEVICE-RESIDENT and the rows
//! are gathered by slot 532 (`load_ple_table` + `q4x_ple_gather`), with the
//! host mmap gather kept as the fallback for a card that cannot hold it
//! (`PADDOCK_Q4X_PLE_HOST=1` forces it for A/Bs).
//!
//! The host lane was the original design and it was wrong on measurement: a
//! token needs 16 rows of 160 B drawn uniformly from 320M rows, so every row
//! is a 4 KB page fault carrying 160 useful bytes, and the page cache only
//! helps once the whole 51.2 GB is resident. On the first serve ladder it
//! showed as prefill ticks of 891-48697 ms and a c8 TTFT p50 of 7858 ms. The
//! rival never made that trade: vLLM's `NgramEmbedding` holds the table in a
//! `VocabParallelEmbedding` (a device Parameter) and gathers it with an
//! index_select - `vllm/model_executor/models/longcat_flash_ngram.py`,
//! `embed_batched`. On device the same access is 2560 B/token of coalesced
//! HBM.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{DeviceTensor, ExpertCache, GpuExecutor, KvDtype, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::gpu_model::st_load::bf16_bytes;
use paddock_kernels::reference::qwen4exp as rq;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::{MapAccess, MappedGguf};
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::load::{dense_head, hc_weights, load_layer, load_ple_projections, load_ple_table};
use super::{DensePlane, DenseStage, Embed, ExpertSeats, HcW, KqSeat, MixerW, PleW, Qwen4ExpLayer};

/// Attention KV element type. f16 is the narrowest class the pack's attention
/// lanes take (there is no f32 KV kernel); the rival stores BF16, which carries
/// three FEWER mantissa bits, so this is not a fairness concession - but it is
/// the dominant deviation from the f32 host reference, and the full-forward
/// gate is stated in those terms.
///
/// `PADDOCK_Q38FN_KV8=1` stores e4m3 instead: halves KV bytes for every
/// attention lane AND is the pool class the tcgen05 decode arm (slot 431)
/// requires - its TMA maps and in-kernel e4m3->bf16 converts assume 1-byte
/// elements, so f16 pools can never elect it. A numerics CLASS change
/// (quality-gated, not bit-gated), which is why it is opt-in.
#[allow(non_snake_case)]
fn KV() -> KvDtype {
    use std::sync::OnceLock;
    static V: OnceLock<KvDtype> = OnceLock::new();
    *V.get_or_init(|| {
        if matches!(std::env::var("PADDOCK_Q38FN_KV8").as_deref(), Ok("1")) {
            KvDtype::Fp8E4m3
        } else {
            KvDtype::Fp16
        }
    })
}

/// The most tokens a row can see and still select everything under QSA:
/// 512 complete blocks of 4 plus a tail of at most 3 - the dense walk is
/// exact up to here and only up to here (bring-up §3.5).
const QSA_DENSE_EXACT: usize = 512 * 4 + 3;

/// Depth of the per-slot ring of raw QSA indexer keys: a block's first keys
/// may come from earlier walks, and a verify chunk's rejected drafts must
/// never alias a committed position the next pool reads - so the ring holds
/// QSA_BLOCK + the deepest verify chunk, rounded up to whole blocks (the
/// shape vLLM sizes its ring by). 16 covers chunks up to 12 rows.
const QSA_RING: usize = 16;
use crate::gpu::qsa::{QSA_BLOCK, QsaRoute};

/// Which attention a walk takes. `Auto` is the serving rule: QSA sparse
/// attention once any row of the walk sees more than `QSA_DENSE_EXACT` tokens.
/// Below that the selection is every block, dense is the same answer, and the
/// dense kernels are the faster way to it (DeepSeek's production rule for its
/// own indexer). `Dense` and `Sparse` pin one path: gate instruments (the
/// sparse path against the dense one where both must agree) and the A/B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsaMode {
    Auto,
    Dense,
    Sparse,
}

/// The serving mode, or a pinned one for A/B legs: `PADDOCK_Q38FN_QSA=dense`
/// or `=sparse` (a development switch; `QsaMode::Auto` otherwise).
fn qsa_mode_env() -> QsaMode {
    match paddock_models::dev_var!("PADDOCK_Q38FN_QSA")
        .ok()
        .as_deref()
    {
        Some("dense") => QsaMode::Dense,
        Some("sparse") => QsaMode::Sparse,
        _ => QsaMode::Auto,
    }
}

/// The sparse path's kernels: tensor cores wherever the pack has them and
/// the shape fits, the f32 SIMT kernels otherwise - (scores, attention).
/// Dev switch `PADDOCK_Q38FN_QSA_SIMT=1` pins both to SIMT (the parity
/// anchors) for A/B legs.
fn qsa_routes(e: &GpuExecutor, c: &Qwen4ExpConfig) -> (QsaRoute, QsaRoute) {
    static SIMT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let pinned = *SIMT.get_or_init(|| paddock_models::dev_var!("PADDOCK_Q38FN_QSA_SIMT").is_ok());
    let pick = |ok: bool| {
        if ok && !pinned {
            QsaRoute::Mma
        } else {
            QsaRoute::Simt
        }
    };
    (
        pick(
            e.has_qsa_logits_mma() && QsaRoute::logits_fits(c.idx_heads, c.idx_head_dim, QSA_BLOCK),
        ),
        pick(
            e.has_qsa_attn_mma()
                && QsaRoute::attn_fits(c.n_heads, c.n_kv_heads, c.head_dim, QSA_BLOCK),
        ),
    )
}

/// Bytes of the per-walk QSA score scratch: rows are scored and selected in
/// batches of as many as fit, so the plane stays this size at any context
/// (65536 blocks x 256 rows at 262K).
const QSA_SCORE_BYTES: usize = 64 << 20;

/// PLE conv dilation - a k=4 kernel over a 9-token receptive ring.
pub(super) const PLE_DILATION: usize = 3;

/// Query-tile height of the prefill attention family (`PD_APF_TQ` in the
/// pack): the batched entry takes one (row0, slot) per tile.
const PD_APF_TQ: usize = 16;

/// Tiles a `PrefillRuns` wave needs: each run is tiled from its own first row,
/// so no tile has to serve two slots.
fn n_qtiles(runs: &[Run]) -> usize {
    runs.iter().map(|r| r.len.div_ceil(PD_APF_TQ)).sum()
}

/// Per-op dump sink, armed by `PADDOCK_Q38FN_DUMP=<dir>`. Writes the same tag
/// names `examples/q38fn_host_forward.rs --dump` writes, so the two trees diff
/// directly and a deviation localizes to one op of one layer instead of to
/// "the logits". Readback is capture-illegal, so this is a triage path only -
/// nothing reads the env var on the serving walk.
/// Per-pass wall clock for a PREFILL walk (`PADDOCK_Q4X_PHASE_MS`). Syncs
/// between passes, so what it reports is a SERIALIZED attribution - it exists
/// to say which pass owns a prefill's wall, not to price a pipelined one. It
/// never arms on a decode walk: a sync inside graph capture is illegal.
struct PhaseMs {
    on: bool,
    t: std::time::Instant,
    acc: Vec<(&'static str, f64, u32)>,
}

thread_local! {
    /// The armed walk's timer, so a pass nested below `device_walk` can lap
    /// into the same map without threading a &mut through every signature.
    static PHASE_MS: std::cell::RefCell<Option<PhaseMs>> = const { std::cell::RefCell::new(None) };
}

/// Lap the armed walk timer, if any (inert when `PADDOCK_Q4X_PHASE_MS` is not
/// set or the walk is a decode - see `PhaseMs::arm`).
pub(crate) fn pm_lap(e: &GpuExecutor, tag: &'static str) {
    PHASE_MS.with(|p| {
        if let Some(pm) = p.borrow_mut().as_mut() {
            pm.lap(e, tag);
        }
    });
}

impl PhaseMs {
    fn arm(phase: Phase) -> Self {
        Self {
            // decode walks only with graph capture off: a sync inside stream
            // capture is illegal, so `PADDOCK_Q38FN_NO_GRAPH=1` is what makes
            // a decode tick measurable here (its absolute numbers carry the
            // eager-launch cost the graph exists to remove - read the SHARES).
            on: (matches!(phase, Phase::Prefill | Phase::PrefillRuns) || !capture_wanted())
                && std::env::var_os("PADDOCK_Q4X_PHASE_MS").is_some(),
            t: std::time::Instant::now(),
            acc: Vec::new(),
        }
    }

    fn lap(&mut self, e: &GpuExecutor, tag: &'static str) {
        if !self.on {
            return;
        }
        let _ = e.synchronize();
        let ms = self.t.elapsed().as_secs_f64() * 1e3;
        match self.acc.iter_mut().find(|(k, _, _)| *k == tag) {
            Some(slot) => {
                slot.1 += ms;
                slot.2 += 1;
            }
            None => self.acc.push((tag, ms, 1)),
        }
        self.t = std::time::Instant::now();
    }

    fn report(&self, n: usize) {
        self.report_as(n, "walk");
    }

    fn report_as(&self, n: usize, what: &str) {
        if !self.on || self.acc.is_empty() {
            return;
        }
        let tot: f64 = self.acc.iter().map(|(_, ms, _)| ms).sum();
        let mut rows = self.acc.clone();
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("[q4x-phase] {what} n={n} total {tot:.2} ms");
        for (tag, ms, cnt) in rows {
            eprintln!(
                "[q4x-phase]   {tag:10} {ms:8.2} ms  ({:5.1}%)  {cnt:4} calls  {:6.3} ms/call",
                100.0 * ms / tot,
                ms / cnt as f64
            );
        }
    }
}

/// `PADDOCK_Q38FN_DUMP_N=<rows>` arms the sink only on walks of that width
/// (every later walk and decode tick writes the same file names, so a
/// prefill's own tree is gone by the end of a run otherwise), and
/// `PADDOCK_Q38FN_DUMP_TAGS=a,b,..` keeps just those tags - a whole 992-row
/// Flash-Next walk is ~12 GB - and `PADDOCK_Q38FN_DUMP_LAYERS=3,7,..` just
/// those layers (the QSA gate reads one attention layer's output, not 48).
struct Dump(
    Option<std::path::PathBuf>,
    Option<Vec<String>>,
    Option<Vec<usize>>,
);

impl Dump {
    fn arm(n: usize) -> Self {
        let width_ok = std::env::var("PADDOCK_Q38FN_DUMP_N")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .is_none_or(|w| w == n);
        let tags = std::env::var("PADDOCK_Q38FN_DUMP_TAGS")
            .ok()
            .map(|s| s.split(',').map(str::to_owned).collect());
        let layers = std::env::var("PADDOCK_Q38FN_DUMP_LAYERS")
            .ok()
            .map(|s| s.split(',').filter_map(|l| l.trim().parse().ok()).collect());
        Self(
            std::env::var_os("PADDOCK_Q38FN_DUMP")
                .filter(|_| width_ok)
                .map(|d| {
                    let p = std::path::PathBuf::from(d);
                    let _ = std::fs::create_dir_all(&p);
                    p
                }),
            tags,
            layers,
        )
    }
    fn on(&self) -> bool {
        self.0.is_some()
    }
    /// The dump dir, when `tag` of layer `li` is one the sink keeps
    /// (`li == usize::MAX`: not a layer's tag, kept by the tag filter alone).
    fn dir_for(&self, li: usize, tag: &str) -> Option<&std::path::Path> {
        let dir = self.0.as_deref()?;
        let layer_ok = li == usize::MAX || self.2.as_ref().is_none_or(|l| l.contains(&li));
        (layer_ok && self.1.as_ref().is_none_or(|t| t.iter().any(|x| x == tag))).then_some(dir)
    }
    /// Write host-side values already read back.
    fn put_host(&self, li: usize, tag: &str, v: &[f32]) -> Result<(), GpuModelError> {
        let Some(dir) = self.dir_for(li, tag) else {
            return Ok(());
        };
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(dir.join(format!("L{li}.{tag}.bin")), b)
            .map_err(|err| GpuModelError::Unsupported(format!("dump write: {err}")))?;
        Ok(())
    }

    /// Read `len` elements back and write them as raw little-endian f32.
    fn put(
        &self,
        e: &GpuExecutor,
        li: usize,
        tag: &str,
        buf: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), GpuModelError> {
        let Some(dir) = self.dir_for(li, tag) else {
            return Ok(());
        };
        let v = e.to_host_len(buf, len)?;
        let mut b = Vec::with_capacity(len * 4);
        for x in &v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let name = if li == usize::MAX {
            format!("{tag}.bin")
        } else {
            format!("L{li}.{tag}.bin")
        };
        std::fs::write(dir.join(name), b)
            .map_err(|err| GpuModelError::Unsupported(format!("dump write: {err}")))?;
        Ok(())
    }
}

/// M-RoPE section split for this family (text uses all four axes equal, so the
/// split only matters for the vision rung; it is the checkpoint's own
/// `[11,11,10]` plus the zero extra axis).
const MROPE_SECTIONS: [u32; 4] = [11, 11, 10, 0];

/// A captured decode-step graph. The raw CUDA handles are only ever used from
/// the engine thread that owns the context and stream - the same
/// single-owner-thread contract every other family's `SendGraph` carries, and
/// what lets `Qwen4ExpGpu` satisfy `Generator: Send` for the serving seam.
pub struct Q4xSendGraph(pub crate::gpu::CapturedGraph);
// SAFETY: see above - single-owner-thread usage, never shared or moved across
// threads while a replay is in flight.
unsafe impl Send for Q4xSendGraph {}

impl std::ops::Deref for Q4xSendGraph {
    type Target = crate::gpu::CapturedGraph;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Where the host lane gathers PLE n-gram rows from: the safetensors shards
/// (e4m3 rows, `weight_scale`), or the GGUF's own table tensor (quantized
/// 32-wide blocks, decoded per row on the CPU).
pub enum PleSource {
    St(ShardedSafetensors),
    Gguf {
        map: Arc<MappedGguf>,
        name: String,
        ty: GgmlType,
        /// bytes per table row (`width` elements)
        row_bytes: usize,
    },
}

/// Bytes one `width`-wide row occupies in the host row decoder's types.
/// `None` for a type it does not decode.
pub(super) fn ple_row_bytes(ty: GgmlType, width: usize) -> Option<usize> {
    let blocks32 = |bytes: usize| width.is_multiple_of(32).then_some(width / 32 * bytes);
    match ty {
        GgmlType::F32 => Some(width * 4),
        GgmlType::F16 | GgmlType::Bf16 => Some(width * 2),
        GgmlType::Q8_0 => blocks32(34),
        GgmlType::Q4_0 | GgmlType::Iq4Nl => blocks32(18),
        _ => None,
    }
}

/// The IQ4_NL codebook (ggml-common.h, MIT).
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Decode one table row (`row.len() == ple_row_bytes(ty, out.len())`).
fn ple_row_dequant(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = half::f16::from_le_bytes(*c).to_f32();
            }
        }
        GgmlType::Bf16 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
            }
        }
        GgmlType::Q8_0 => {
            for (blk, o) in row
                .as_chunks::<34>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
            {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..32 {
                    o[j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_0 => {
            for (blk, o) in row
                .as_chunks::<18>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
            {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..16 {
                    let q = blk[2 + j];
                    o[j] = ((q & 0xf) as i32 - 8) as f32 * d;
                    o[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Iq4Nl => {
            for (blk, o) in row
                .as_chunks::<18>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
            {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..16 {
                    let q = blk[2 + j];
                    o[j] = KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * d;
                    o[j + 16] = KVALUES_IQ4NL[(q >> 4) as usize] as f32 * d;
                }
            }
        }
        _ => unreachable!("ple_row_bytes gated the type"),
    }
}

pub struct Qwen4ExpGpu {
    exec: Arc<GpuExecutor>,
    cfg: Qwen4ExpConfig,
    /// where the host-side PLE n-gram gather reads its rows
    /// Resident WEIGHT bytes, stamped once at load by `settled_mem_used()`
    /// before anything pool-sized allocates - the number `will-it-fit` prices
    /// this model with. It excludes what this family deliberately keeps off
    /// the device: the n-gram table (51B of the parameter count, read from the
    /// GGUF mmap) and any host-mapped experts, which is why the honest figure
    /// is far under the download size.
    weights_bytes: Option<u64>,
    st: PleSource,
    layers: Vec<Qwen4ExpLayer>,
    embed: Embed,
    lm_head: DensePlane,
    final_mix: HcW,
    /// the context: KV, state and index caches hold this many positions a slot
    max_tokens: usize,
    /// the most rows one device walk carries - every per-row scratch plane's
    /// size (`super::walk_rows()`, capped at the context); longer walks split
    walk_rows: usize,
    sc: Scratch,
    /// per-layer GDN recurrent state `[v_heads][k_dim][v_dim]`, None on attn layers
    recur: Vec<Option<CudaSlice<f32>>>,
    /// per attention layer, the K and V pool planes `[pool blocks, 16,
    /// kv_dim]` (`pages` addresses them), None on GDN layers
    kv_k: Vec<Option<CudaSlice<u8>>>,
    kv_v: Vec<Option<CudaSlice<u8>>>,
    /// per attention layer, the QSA indexer's compressed key plane
    /// `[pool blocks, 4, 128]` bf16 on the same block ids - one normed,
    /// rotated key per 4-token block, written as each block completes
    /// (attn/qsa.cuh). None on GDN layers and on a pack without the indexer.
    idx_cache: Vec<Option<CudaSlice<half::bf16>>>,
    /// per attention layer, the raw indexer keys the next pool may still need
    /// `[slots, QSA_RING, 128]` f32 (a ring by position)
    idx_ring: Vec<Option<CudaSlice<f32>>>,
    /// the KV pool's bookkeeping, the slots' block tables and the device
    /// table every trunk attention launch reads (pages.rs)
    pages: super::pages::KvPages,
    /// per-GDN-layer conv window: the last `k-1` PRE-conv rows, oldest first
    /// (`conv_step`'s contract - it shifts the window itself)
    gdn_win: Vec<Option<CudaSlice<f32>>>,
    /// the PLE conv's window: the last `(k-1)*dilation` pre-conv rows of
    /// `norm_conv(gv)`, oldest first. `q4x_conv_dil_step` is stateless by
    /// design (graph-safe), so this side advances it.
    ple_win: Option<CudaSlice<f32>>,
    /// slot id of each ROW of the walk currently staged (len == rows). The
    /// single-sequence lane leaves this `[0]`.
    cur_slots: Vec<usize>,
    /// the run table the CURRENT walk carries; empty outside `PrefillRuns`
    cur_runs: Vec<Run>,
    /// A MIXED walk (forward/chunked.rs): the first `walk_lead` rows of a
    /// `PrefillRuns` walk are decode rows - one token each for distinct
    /// slots at their own positions, the first `walk_lead` entries of
    /// `cur_runs` as length-1 runs. Their three carried-state ops and their
    /// attention take the batched DECODE entries (one launch for all of
    /// them, the decode tick's class) and the prompt runs behind them keep
    /// the per-run arms; everything row-parallel runs once over all rows.
    /// 0 outside a mixed tick.
    walk_lead: usize,
    /// QSA sparse attention serves the next walk - set by each walk's caller
    /// from its rows' positions (`qsa_for`), read by `device_walk`
    walk_qsa: bool,
    /// the pack carries the QSA selection and attention, and the lane keeps
    /// the index caches they read
    qsa_ready: bool,
    /// which attention the walks take (`QsaMode::Auto` serves)
    qsa_mode: QsaMode,
    /// how many independent sequences this instance carries. 1 is the
    /// single-sequence lane every gate is stamped against; > 1 sizes every
    /// carried-state buffer per slot and unlocks `decode_step_batch`.
    slots: usize,
    /// next position to write, per SLOT - the cursor prefill leaves behind and
    /// decode advances. 0 means "no sequence started" in that slot.
    pos: Vec<usize>,
    /// each slot's token stream with the 2-token EOS priming already on the
    /// front, carried so a decode step can hash its n-gram window.
    stream: Vec<Vec<i64>>,
    /// The captured decode tick. Every per-token INPUT (token id, positions,
    /// the PLE n-gram rows) is staged into address-stable buffers before the
    /// replay, and every kernel in the tick reads its position from the device
    /// - so one capture is valid at every position, exactly as in the qwen3.5
    ///   lane. `None` until the first decode step builds it, or forever under
    ///   `PADDOCK_Q38FN_NO_GRAPH`.
    ///
    /// Indexed by attention mode (0 dense, 1 QSA): a walk's kernels differ by
    /// mode, so each captures its own graph.
    decode_graph: [Option<Q4xSendGraph>; 2],
    /// one captured batched tick per WIDTH. Valid only for the dense slot set
    /// `0..n`: the PLE window advance bakes per-slot copy offsets into the
    /// graph, so a different slot set must not replay it.
    batch_graphs: Vec<[Option<Q4xSendGraph>; 2]>,
    /// staging the 8-bit classes need on their batch > 1 arm
    stage: DenseStage,
    /// Whether decode ticks may be captured at all. Defaults to the env gate;
    /// `set_graph_capture` lets one process A/B the two paths, which is how
    /// the capture gate proves the graph and the eager walk agree.
    graph_capture: bool,
    /// The prefix cache (prefix.rs): `None` when the model has nothing to
    /// cache or the engine-wide switch is off.
    prefix: Option<super::prefix::PrefixCache>,
    /// The walk in flight CONTINUES a sequence: rows at positions
    /// `walk_row0..` of a slot whose carried state was restored. The two
    /// causal convs re-stage their windows in front of the span's first rows
    /// (`resume_gdn_conv` / `resume_ple_conv`); 0 = a fresh sequence.
    walk_row0: usize,
    /// in-walk prefix-cache checkpoints for the span `device_walk` runs next:
    /// (row within the span, staging blob), ascending - the GDN and PLE
    /// passes write each one's carried state as it stands at that row into
    /// the prefix cache's staging blob
    walk_cuts: Vec<(usize, u32)>,
    /// the reserved checkpoint index each staging blob of the walk just run
    /// commits into (`commit_walk_cuts`), blob order
    walk_ckpts: Vec<u32>,
    /// Stage F, the reply checkpoint (`reply_snapshot`): per slot the cut
    /// and pool index of its live in-reply checkpoint, and whether the
    /// slot's reply is tracked at all (a prompt admitted through the cache,
    /// long enough to be worth a checkpoint).
    reply_ckpt: Vec<Option<(usize, u32)>>,
    /// The reply checkpoint held at the reply's first tool call
    /// (`reply_pin`); the rolling one moves on past the call.
    reply_pinned: Vec<Option<(usize, u32)>>,
    reply_track: Vec<bool>,
    /// The MTP drafter (forward/mtp.rs), once a head GGUF is attached.
    mtp: Option<Box<mtp::Mtp>>,
    /// The verify round's planes (forward/spec.rs), built at its first round.
    verify: Option<Box<spec::Verify>>,
    /// the host-sampled verify round between its walk and its commit
    /// (`verify_open` / `verify_close`)
    spec_open: Option<Vec<Run>>,
    /// Prompts queued for chunked prefill (forward/chunked.rs), FIFO.
    chunked: Vec<chunked::ChunkedPrefill>,
    /// Canonical RS (PADDOCK_SPEC_RS): this round's per-slot chain draws, put
    /// here by the service immediately before it arms the chain. The backend
    /// has no sampler access, so the inverse temperature and the per-step draft
    /// uniforms have to arrive from the slot's own seed stream or the draw is
    /// not the one the request asked for. Cleared as the chain consumes it.
    spec_rs_draws: Option<Vec<crate::generator::SpecRsDraw>>,
}

/// Every device buffer the walk touches, allocated once at `max_tokens`.
/// Address-stable by construction - the graph-capture rung depends on it.
/// Split factor for the batch-1 split-K matvec arm (slot 519).
///
/// DEFAULT off, and that is a measured verdict, not caution. The two planes it
/// targets - the GDN alpha||beta plane (96 blocks) and the MoE router (513) -
/// look starved per-kernel, but both sit on a FORKED branch or opposite one,
/// so the stream forks already hide them: swept split 4/8/16 against off and
/// the wall does not move (121.4 tok/s at off and at 8, 118.8 at 4 and 16).
/// Once concurrency is matched, per-kernel starvation stops predicting wall
/// time for anything that is not on the critical path. The arm is kept because
/// it is correct and deterministic and a future batched lane may want it.
const SK_SPLIT: u32 = 0;

fn sk_split() -> u32 {
    use std::sync::OnceLock;
    static N: OnceLock<u32> = OnceLock::new();
    *N.get_or_init(|| match std::env::var("PADDOCK_Q38FN_SK").ok().as_deref() {
        None => SK_SPLIT,
        Some("off") | Some("0") => 0,
        Some(v) => v.parse().unwrap_or(SK_SPLIT),
    })
}

struct Scratch {
    d_tok: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_mrope: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    /// QSA indexer planes (`qsa_index`): the fused q|k projection
    /// `[t, (idx_heads+idx_kv_heads)*idx_head_dim]`, the normed+rotated query
    /// `[t, idx_heads*idx_head_dim]`, the pooled keys staged for their rotary
    /// `[t, idx_head_dim]` and those keys' block positions `[4, t]`
    d_idx_qk: CudaSlice<f32>,
    d_idx_q: CudaSlice<f32>,
    d_idx_stage: CudaSlice<f32>,
    d_idx_spos: CudaSlice<u32>,
    /// QSA selection + attention planes (`qsa_attend`): block scores for a
    /// batch of `qsa_rb` rows `[qsa_rb, cap]`, each row's selected blocks
    /// `[t, k]` and count `[t]`, and the split partials `[t, heads, hd]` /
    /// `[t, heads, 2]` (rows x splits never exceed t)
    d_qsa_scores: CudaSlice<f32>,
    d_qsa_sel: CudaSlice<u32>,
    d_qsa_cnt: CudaSlice<u32>,
    d_qsa_po: CudaSlice<f32>,
    d_qsa_pml: CudaSlice<f32>,
    qsa_rb: usize,
    d_x: CudaSlice<f32>,
    d_h: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_m: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_bi: CudaSlice<f32>,
    d_inj: CudaSlice<f32>,
    d_mix: CudaSlice<f32>,
    // GDN
    d_qkv: CudaSlice<f32>,
    d_zg: CudaSlice<f32>,
    d_ab: CudaSlice<f32>,
    d_g: CudaSlice<f32>,
    d_beta: CudaSlice<f32>,
    d_conv: CudaSlice<f32>,
    d_dq: CudaSlice<f32>,
    d_dk: CudaSlice<f32>,
    d_dv: CudaSlice<f32>,
    d_dattn: CudaSlice<f32>,
    d_core: CudaSlice<f32>,
    // attention
    d_qg: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_agate: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_qn: CudaSlice<f32>,
    d_kn: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    d_sinks: CudaSlice<f32>,
    // MoE
    d_logits: CudaSlice<f32>,
    /// pre-norm scalars for the slot-541 walk: [t, hv, 2] f32
    d_dnrn: CudaSlice<f32>,
    d_moe_part: CudaSlice<f32>,
    /// slot-544 warmup's dummy output (kept alive; also marks warmup done)
    #[allow(dead_code)]
    d_lowm_warm: CudaSlice<f32>,
    /// The low-M cluster warm-up was refused on this card (sm_120 consumer
    /// Blackwell answers cudaErrorNotSupported): the lane stays off, whatever
    /// the opt-in says - a refused warm-up used to only print and leave the
    /// decode gate to elect the lane anyway.
    lowm_refused: bool,
    /// split-KV fmha partials (slot 545): [rows<=64][n_heads][S<=16][hd+2],
    /// caller-owned and address-stable (graph capture)
    d_fmha_part: CudaSlice<f32>,
    d_zero_bias: CudaSlice<f32>,
    /// split-K matvec scratch (slot 519), caller-owned and address-stable so
    /// nothing allocates inside the captured decode tick
    d_skp: CudaSlice<f32>,
    d_skc: CudaSlice<u32>,
    d_idx: CudaSlice<u32>,
    d_topw: CudaSlice<f32>,
    d_act: CudaSlice<f32>,
    /// GGUF expert seats: the int8 block input + per-32 scales, the per-16
    /// sums (Q4/Q5 min term, sized for the wider of the two stages), and the
    /// quantized swiglu output the down stage reads
    d_xq: CudaSlice<i8>,
    d_xs: CudaSlice<f32>,
    d_ssums: CudaSlice<f32>,
    d_fq: CudaSlice<i8>,
    d_fs: CudaSlice<f32>,
    /// expert-grouped prefill MoE (slot 586): the moe_align CSR over the
    /// walk's routed pairs. Sized for the SMALLEST group (8) because that is
    /// the one that pads the most blocks; a wider group needs fewer.
    d_msrow: CudaSlice<u32>,
    d_msslot: CudaSlice<u32>,
    d_mbexp: CudaSlice<u32>,
    /// per-(pair, column) partials for the grouped down (slot 589), one COLUMN
    /// CHUNK wide - a full [pairs, hidden] plane would be 419 MB at a 4096-row
    /// wave, and the fold consumes a chunk at a time anyway
    d_dpart: CudaSlice<f32>,
    // the tensor-core gate/up's moe_align (bm = 32) CSR and its fused-quantize
    // output in that SORTED layout: [blocks][32][moe_ff] int8 + per-32 scales
    /// NVFP4 activation pair for the W4A4 routed arm (slot 631): packed e2m1
    /// over an i8 plane plus its per-16 e4m3 scales. Only the PREFILL arm
    /// stages these - the decode GEMV eats f32 straight.
    d_xq4: CudaSlice<i8>,
    d_xs4: CudaSlice<u8>,
    /// the W4A4 pair's intermediate: SwiGLU output requantized to nvf4,
    /// sorted-position indexed, which is nvf4_moe_down_bs's direct B input
    d_nfq: CudaSlice<u8>,
    d_nfs: CudaSlice<u8>,
    /// Per-(token, slot) f32 partials for the W4A4 down half. Its OWN plane,
    /// not `d_moe_part`: that one is a decode-band buffer (64 rows x the
    /// z-split's halved slots, ~3 MB) and `nvf4_moe_down_bs` lands
    /// n * (k+1) * hidden floats, so a 128-token prefill overran it 4x and
    /// took the serve down with CUDA_ERROR_ILLEGAL_ADDRESS.
    ///
    /// BOUNDED, not full-width: at max_tokens 4096 a full-width plane is 461
    /// MB out of the same headroom the KV pool sizes from, for an arm that
    /// only engages on prefill. Sized to `NVF4_BS_MAX_ROWS` tokens instead,
    /// and the dispatch checks this length before electing - so a wave wider
    /// than the plane keeps the GEMV, correctly and quietly, instead of
    /// scribbling. Proper token-chunking (nemotron's shape) would lift that
    /// ceiling; it needs token offsets on four shared wrappers.
    d_nvf4_part: CudaSlice<f32>,
    d_srow32: CudaSlice<u32>,
    d_sslot32: CudaSlice<u32>,
    d_bexp32: CudaSlice<u32>,
    d_sfq: CudaSlice<i8>,
    d_sfs: CudaSlice<f32>,
    /// Per-expert (first, end) sorted positions the expert-major tensor-core
    /// down (slot 603) derives from the bm = 32 layout: 2 x n_expert.
    d_emap: CudaSlice<u32>,
    /// One [norm_w (hc * hidden) | 1/rms (max_tokens * hc)] plane per
    /// hyper-connection mix (attention mix of layer l at 2l, MLP mix at 2l +
    /// 1): the norm prefix is copied once at load, and a combine that stores
    /// no normalized state (slot 606) writes the 1/rms tail its next mix's
    /// rebuild consumers read (slots 607 / 608). Empty without those slots.
    d_hcaux: Vec<CudaSlice<f32>>,
    /// Slot 609's (row, stream, output) inject partials, [max_tokens][hc][hc].
    d_injp: CudaSlice<f32>,
    /// the next hyper-connection down's mmq rows, emitted by the quantizing
    /// combine (slot 602) as one group padded to 128 rows, for walks up to
    /// `HC_PREQ_MAX_ROWS` (a stub without the slot)
    d_hcq: CudaSlice<u8>,
    /// device-sampling scratch (slots 4-wide each): the packed per-row plan,
    /// the truncation params, and the sampled ids. Caller-owned and
    /// address-stable like everything else the decode graph can see.
    d_par: CudaSlice<u32>,
    d_tpar: CudaSlice<u32>,
    d_ids: CudaSlice<u32>,
    d_shg: CudaSlice<f32>,
    d_shu: CudaSlice<f32>,
    d_shd: CudaSlice<f32>,
    d_shgate: CudaSlice<f32>,
    // PLE
    d_emb: CudaSlice<f32>,
    /// run table for a `PrefillRuns` walk: [runs] row offset / row count /
    /// slot, device-side because the batched recurrence reads them per block
    d_run_off: CudaSlice<u32>,
    d_run_len: CudaSlice<u32>,
    d_run_slot: CudaSlice<u32>,
    /// the same runs as a prefix table, [runs + 1] row offsets: the run table
    /// the tensor-core prefill's batched-runs arm reads (`pf_runs_register`)
    d_run_offs: CudaSlice<u32>,
    /// per-q-tile (row0, slot) for the batched prefill attention - the
    /// single-slot twin reads `slots[0]` for every row, so a wave needs the
    /// per-tile entry
    d_tile_row0: CudaSlice<u32>,
    d_tile_slot: CudaSlice<u32>,
    /// [max_tokens, ple_heads] global n-gram row ids for the device gather
    /// (slot 532). 64 B per row against the 10 KB/row f32 plane the host
    /// gather used to push across PCIe every tick.
    d_ple_ids: CudaSlice<u32>,
    d_pkey: CudaSlice<f32>,
    d_pval: CudaSlice<f32>,
    /// resumed-conv staging: `[window ; first rows]` in, the conv of it out
    /// (GDN: 2(k-1) x qkv rows; PLE: 2 x ring rows x hc width)
    d_gdn_ext_in: CudaSlice<f32>,
    d_gdn_ext_out: CudaSlice<f32>,
    d_ple_ext_in: CudaSlice<f32>,
    d_ple_ext_out: CudaSlice<f32>,
    d_pkn: CudaSlice<f32>,
    d_pqn: CudaSlice<f32>,
    d_pgv: CudaSlice<f32>,
    d_pconv: CudaSlice<f32>,
    // head
    d_fin: CudaSlice<f32>,
    d_out: CudaSlice<f32>,
}

impl Qwen4ExpGpu {
    /// Load the whole text model. `max_tokens` sizes every scratch plane and
    /// the KV caches - a longer prompt is a loud refusal, never a silent
    /// truncation.
    pub fn load(
        exec: &Arc<GpuExecutor>,
        dir: &std::path::Path,
        max_tokens: usize,
    ) -> Result<Self, GpuModelError> {
        Self::load_with_slots(exec, dir, max_tokens, 1)
    }

    /// Load sized for `slots` concurrent sequences. Every carried-state buffer
    /// (GDN recurrence, both conv windows, the KV cache) is allocated per slot;
    /// the GDN recurrence dominates at
    /// `v_heads * k_dim * v_dim * 4 B` per layer. `slots == 1` is byte-for-byte
    /// the single-sequence lane.
    pub fn load_with_slots(
        exec: &Arc<GpuExecutor>,
        dir: &std::path::Path,
        max_tokens: usize,
        slots: usize,
    ) -> Result<Self, GpuModelError> {
        if slots == 0 {
            return Err(GpuModelError::Unsupported("slots must be >= 1".into()));
        }
        if !exec.has_delta_gate_ab() {
            return Err(GpuModelError::Unsupported(
                "pack has no delta_gate_ab - the folded GDN a||b plane needs it".into(),
            ));
        }
        if !exec.has_qwen4exp_ops() {
            return Err(GpuModelError::Unsupported(
                "kernel pack has no qwen4exp family (slots 506-516) - rebuild packs/cuda".into(),
            ));
        }
        if !exec.has_q4x_conv_dil_step_ring() {
            // the PLE window is a position-indexed ring end to end (prefill
            // seed and decode step agree on `q % wrows`); there is no second
            // convention to fall back to
            return Err(GpuModelError::Unsupported(
                "kernel pack has no q4x_conv_dil_step_ring (slot 533) - rebuild packs/cuda".into(),
            ));
        }
        let cfg = Qwen4ExpConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp shards: {e}")))?;

        let h = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            let mut layer = load_layer(exec, &st, &cfg, li)?;
            if cfg.ple_layers.contains(&li) {
                layer.ple = Some(load_ple_projections(exec, &st, &cfg, li)?);
            }
            layers.push(layer);
        }

        // The 51.2 GB n-gram table goes to the device unless it will not fit
        // (or the host lane is forced). The host mmap gather it replaces is a
        // uniform random read at 160 useful bytes per 4 KB page over a 51.2 GB
        // file: it put prefill ticks of 891-48697 ms and a c8 TTFT p50 of
        // 7858 ms on the serve ladder. vLLM has always kept this table
        // device-resident (`NgramEmbedding.oe_embedder`, a
        // `VocabParallelEmbedding`).
        //
        // After the layer loop, deliberately. `ple_device_table` prices the
        // table against memory free at the moment it runs, which is only a
        // sound test if the table is the last big claim - and inside the loop
        // it is not: `ple_layer_ids` puts it at decoder layer 1 of 48, so the
        // check saw a nearly empty device, took the table, and the remaining
        // 46 layers then ran the box out of memory. On a unified-memory board
        // that is not a CUDA OOM the loader can catch and fall back from - it
        // is the kernel's OOM killer taking the process, with nothing in the
        // log but the "resident on device" line it had just printed
        // (measured 2026-09-19: NVIDIA's official NVFP4 checkpoint, 132.7 GB
        // over 10 shards, killed at layer ~1 of 48 on a 121 GiB GB10).
        // Loading it last makes the comment above true on every board and
        // costs nothing: the projections are already seated, and the table is
        // a single contiguous claim either way.
        for li in cfg.ple_layers.clone() {
            let Some(ple) = layers.get_mut(li).and_then(|l| l.ple.as_mut()) else {
                continue;
            };
            if ple_device_table(exec, &cfg) {
                match load_ple_table(exec, &st, &cfg, li, ple) {
                    Ok(()) => {
                        tracing::info!(
                            "qwen4exp: PLE n-gram table resident on device ({} rows)",
                            ple.table_rows
                        );
                        eprintln!("[q4x-ple] device table: {} rows", ple.table_rows);
                    }
                    // Never silent: the host lane is the same answer at
                    // ~100x the prefill cost, and the bench that measures
                    // it looks identical from the outside
                    Err(e) => {
                        tracing::warn!("qwen4exp: PLE table stays on the host: {e}");
                        eprintln!("[q4x-ple] HOST lane (device table refused): {e}");
                        warm_ple_table(&st, &cfg, li);
                    }
                }
            } else {
                // Host lane by election, not by failure - fault the table in
                // now rather than one disk seek at a time on the critical path
                warm_ple_table(&st, &cfg, li);
            }
        }

        let embed = Embed::Bf16(bf16_plane(
            exec,
            &st,
            "model.language_model.embed_tokens.weight",
            cfg.vocab,
            h,
        )?);
        let lm_head = dense_head(exec, &st, "lm_head.weight", cfg.vocab, h)?;
        let final_mix = hc_weights(
            exec,
            &st,
            &cfg,
            "model.language_model.hyper_connection_mixer",
            false,
        )?;
        Self::from_parts(
            exec,
            cfg,
            PleSource::St(st),
            layers,
            embed,
            lm_head,
            final_mix,
            max_tokens,
            slots,
        )
    }

    /// The same model off a llama.cpp `qwen4exp` GGUF (see `load_gguf.rs`):
    /// dense planes on the k-quant streams, experts as k-quant / i-quant
    /// seats (host-mapped under `[moe_offload]` - call `enable_moe_cache`
    /// after the KV plan to seat the slot cache), the PLE table gathered
    /// from the mmap. `path` is the first shard of a split family or the
    /// single file.
    pub fn load_gguf_with_slots(
        exec: &Arc<GpuExecutor>,
        path: &std::path::Path,
        max_tokens: usize,
        slots: usize,
    ) -> Result<Self, GpuModelError> {
        if slots == 0 {
            return Err(GpuModelError::Unsupported("slots must be >= 1".into()));
        }
        if !exec.has_delta_gate_ab() {
            return Err(GpuModelError::Unsupported(
                "pack has no delta_gate_ab - the folded GDN a||b plane needs it".into(),
            ));
        }
        if !exec.has_qwen4exp_ops() {
            return Err(GpuModelError::Unsupported(
                "kernel pack has no qwen4exp family (slots 506-516) - rebuild packs/cuda".into(),
            ));
        }
        if !exec.has_q4x_conv_dil_step_ring() {
            return Err(GpuModelError::Unsupported(
                "kernel pack has no q4x_conv_dil_step_ring (slot 533) - rebuild packs/cuda".into(),
            ));
        }
        let map = Arc::new(
            MappedGguf::open(path)
                .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp gguf: {e}")))?,
        );
        let cfg = Qwen4ExpConfig::from_gguf(map.gguf(), |name| {
            map.tensor_info(name)
                .map(|t| t.dims.iter().map(|&d| d as usize).collect())
        })
        .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp gguf config: {e}")))?;
        let hash = Qwen4ExpConfig::ple_hash_from_gguf(map.gguf())
            .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp gguf ple: {e}")))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            tracing::debug!("qwen4exp gguf: layer {li} ({:?})", cfg.blocks[li]);
            let mut layer = super::load_gguf::load_layer(exec, &map, &cfg, li)?;
            if cfg.ple_layers.contains(&li) {
                layer.ple = Some(super::load_gguf::load_ple(exec, &map, &cfg, li, &hash)?);
                tracing::info!("qwen4exp: PLE n-gram table stays in the GGUF mmap (host lane)");
            }
            layers.push(layer);
        }
        let embed = super::load_gguf::load_embed(exec, &map, &cfg)?;
        let lm_head = super::load_gguf::load_head(exec, &map, &cfg)?;
        let final_mix = super::load_gguf::hc_weights(exec, &map, &cfg, "output_hc", false)?;
        let src = {
            let name = super::load_gguf::PLE_TABLE.to_owned();
            let (info, _) = map
                .tensor_bytes(&name)
                .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}")))?;
            let width = cfg.ple_embed / cfg.ple_heads();
            let row_bytes = ple_row_bytes(info.ggml_type, width).ok_or_else(|| {
                GpuModelError::Unsupported(format!("{name}: type {:?}", info.ggml_type))
            })?;
            // the host gather reads single rows at hashed offsets: a fault
            // should bring in its page, not the readahead window around it
            // (the gather hints its own batch - see gather_ple_rows_gguf)
            let len = map.tensor_bytes(&name).map(|(_, b)| b.len()).unwrap_or(0);
            let _ = map.advise_tensor(&name, MapAccess::Random, &[(0, len)]);
            PleSource::Gguf {
                ty: info.ggml_type,
                row_bytes,
                name,
                map: map.clone(),
            }
        };
        Self::from_parts(
            exec, cfg, src, layers, embed, lm_head, final_mix, max_tokens, slots,
        )
    }

    /// `[moe_offload]`: seat the VRAM slot cache over the host-mapped expert
    /// planes (`gpu/moe_cache.rs`) inside `budget` bytes. Returns the number
    /// of layers seated; 0 when no layer is host-mapped, the pack lacks the
    /// cache kernels, or the budget does not fit 8 slots per layer (the
    /// experts then serve zero-copy over PCIe). Mirrors qwen35's.
    pub fn enable_moe_cache(&mut self, budget: u64) -> Result<usize, GpuModelError> {
        fn host_seats(
            l: &Qwen4ExpLayer,
        ) -> Option<(
            &crate::gpu::HostMappedKq,
            &crate::gpu::HostMappedKq,
            &crate::gpu::HostMappedKq,
        )> {
            match &l.moe.seats {
                ExpertSeats::Kq {
                    gate,
                    up,
                    down,
                    cache: None,
                } => Some((gate.host()?, up.host()?, down.host()?)),
                _ => None,
            }
        }
        let price: u64 = self
            .layers
            .iter()
            .filter_map(host_seats)
            .map(|(g, u, d)| ExpertCache::slot_bytes(g, u, d))
            .sum();
        if price == 0 || !self.exec.has_moe_cache() {
            return Ok(0);
        }
        let cfg = crate::gpu::moe_offload();
        let budget = cfg.vram_bytes.map_or(budget, |cap| budget.min(cap));
        let auto = (budget / price) as usize;
        let slots = crate::gpu::moe_cache_slots_pin().unwrap_or(auto);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        if slots < 8 {
            tracing::warn!(
                slots,
                budget_gib = gib(budget),
                "qwen4exp MoE expert offload: no room for a slot cache - experts serve \
                 zero-copy over PCIe (slow); lower max_ctx or max_batch"
            );
            return Ok(0);
        }
        let n_expert = self.cfg.n_expert;
        let slots = slots.min(n_expert);
        let max_rows = self.walk_rows * self.cfg.n_active;
        let mut seated = 0usize;
        let mut vram = 0u64;
        for l in self.layers.iter_mut() {
            let cache = match &l.moe.seats {
                ExpertSeats::Kq {
                    gate: KqSeat::Host(g),
                    up: KqSeat::Host(u),
                    down: KqSeat::Host(d),
                    cache: None,
                } => self.exec.new_expert_cache(g, u, d, slots, max_rows)?,
                _ => continue,
            };
            vram += cache.vram_bytes();
            if let ExpertSeats::Kq { cache: c, .. } = &mut l.moe.seats {
                *c = Some(Box::new(cache));
            }
            seated += 1;
        }
        tracing::info!(
            layers = seated,
            slots,
            vram_gib = gib(vram),
            budget_gib = gib(budget),
            "qwen4exp MoE expert offload: slot cache seated"
        );
        eprintln!(
            "[q4x-moe] slot cache: {seated} layers x {slots} slots, {:.2} GiB of {:.2} GiB budget",
            gib(vram),
            gib(budget)
        );
        Ok(seated)
    }

    /// Slot-cache counters summed over the layers: (rows resolved, misses).
    pub fn moe_cache_stats(&self) -> Result<(u64, u64), GpuModelError> {
        let mut acc = (0u64, 0u64);
        for l in &self.layers {
            if let ExpertSeats::Kq { cache: Some(c), .. } = &l.moe.seats {
                let (r, m) = c.stats(&self.exec)?;
                acc.0 += r;
                acc.1 += m;
            }
        }
        Ok(acc)
    }

    /// Host bytes the expert planes hold (device-mapped), for the load log.
    pub fn expert_host_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| match &l.moe.seats {
                ExpertSeats::Kq { gate, up, down, .. } => {
                    gate.bytes().1 + up.bytes().1 + down.bytes().1
                }
                ExpertSeats::Nvf4 { .. } | ExpertSeats::Q8 { .. } => 0,
            })
            .sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        exec: &Arc<GpuExecutor>,
        cfg: Qwen4ExpConfig,
        st: PleSource,
        layers: Vec<Qwen4ExpLayer>,
        embed: Embed,
        lm_head: DensePlane,
        final_mix: HcW,
        max_tokens: usize,
        slots: usize,
    ) -> Result<Self, GpuModelError> {
        // the walks read every KV and index plane through the pool's table
        if !exec.has_attn_paged_modes()
            || !exec.has_attn_prefill_paged()
            || (exec.has_qsa_indexer() && !exec.has_qsa_paged())
        {
            return Err(GpuModelError::Unsupported(
                "qwen4exp needs the paged attention modes (pack slots 688-696) - rebuild packs/cuda"
                    .into(),
            ));
        }
        let recur = alloc_recur(exec, &cfg, slots)?;
        // QSA indexer state beside every attention layer's KV (tiny: 64 B a
        // token a layer, plus a 16-deep raw ring per slot). Written on every
        // walk, dense or not, so a sequence crossing the dense-exact window
        // already has every block key the sparse path will score.
        let qsa = exec.has_qsa_indexer();
        let qsa_ready = qsa && exec.has_qsa_select() && exec.has_qsa_attn();
        if max_tokens > QSA_DENSE_EXACT && !qsa_ready {
            tracing::warn!(
                max_ctx = max_tokens,
                "qwen4exp: this pack has no QSA sparse attention, so attention is served DENSE - \
                 rows past {QSA_DENSE_EXACT} visible tokens will not be the reference model's (they \
                 attend to every token, the model to its indexer's selection)"
            );
        }
        let mut idx_ring = Vec::new();
        for b in &cfg.blocks {
            let attn = qsa && matches!(b, Qwen4ExpBlock::Attention);
            idx_ring.push(if attn {
                Some(exec.alloc(slots * QSA_RING * cfg.idx_head_dim)?)
            } else {
                None
            });
        }
        let mut gdn_win = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            gdn_win.push(match cfg.blocks[li] {
                Qwen4ExpBlock::Gdn => {
                    Some(exec.alloc(slots * (cfg.gdn_conv - 1) * cfg.gdn_qkv_rows())?)
                }
                Qwen4ExpBlock::Attention => None,
            });
        }
        let ple_win = if cfg.ple_layers.is_empty() {
            None
        } else {
            Some(exec.alloc(slots * (cfg.ple_conv - 1) * PLE_DILATION * cfg.hc_width())?)
        };
        // only the GGUF lane seats k-quant planes (safetensors is bf16 / NVFP4)
        let kq_lanes = matches!(st, PleSource::Gguf { .. });
        // The PUBLISHED resident-weight line, stamped exactly here: every
        // plane is uploaded, and the scratch below is the first pool-sized
        // claim. The moe slot cache the runner seats afterwards is a CACHE
        // sized from leftover headroom, not weights, and is correctly outside
        // this number.
        let weights_bytes = exec.settled_mem_used();
        let walk_rows = max_tokens.min(super::walk_rows());
        let mut sc = Scratch::new(exec, &cfg, walk_rows, max_tokens, slots, kq_lanes)?;
        // the rebuild planes' norm prefixes: each mix's (1+w) weight, once
        if !sc.d_hcaux.is_empty() {
            let hw = cfg.hc_width();
            for (li, l) in layers.iter().enumerate() {
                exec.copy_region(&l.attn_hc.norm.buf, 0, &mut sc.d_hcaux[2 * li], 0, hw)?;
                exec.copy_region(&l.mlp_hc.norm.buf, 0, &mut sc.d_hcaux[2 * li + 1], 0, hw)?;
            }
        }
        // the widest activation any dense plane reads is the 4-stream state
        let stage = DenseStage {
            row_exact: false,
            prefill: false,
            xrow: if kq_lanes {
                Some(exec.alloc(cfg.hc_width())?)
            } else {
                None
            },
            yrow: if kq_lanes {
                Some(exec.alloc(cfg.vocab.max(cfg.hc_width()))?)
            } else {
                None
            },
            q: exec.alloc_i8(walk_rows * cfg.hc_width())?,
            // widest activation row any dense plane takes is hc_width
            xq8: exec.alloc_i8(walk_rows * cfg.hc_width())?,
            xs8: exec.alloc_u8(walk_rows * cfg.hc_width() / 32)?,
            rs: exec.alloc(walk_rows)?,
            xs: exec.alloc(if kq_lanes {
                walk_rows * cfg.hc_width() / 32
            } else {
                1
            })?,
            ssums: exec.alloc(if kq_lanes {
                walk_rows * cfg.hc_width() / 16
            } else {
                1
            })?,
            // the > 64-row tile's mmq tiles for one KQ_TILE_ROWS-row chunk:
            // 144 B per (128-wide k chunk, row), and 4 per-32 sums per pair
            // (two KQ_TILE_ROWS of the widest input: a 992-row walk then
            // launches every other plane once - `q8_launch_rows`)
            yq: exec.alloc_u8(if kq_lanes {
                cfg.hc_width().div_ceil(128) * 2 * super::KQ_TILE_ROWS * 144
            } else {
                1
            })?,
            xsums: exec.alloc(if kq_lanes {
                cfg.hc_width().div_ceil(128) * super::KQ_TILE_ROWS * 4
            } else {
                1
            })?,
            // K-split partials (slot 588) for the narrow-out Q8 planes, sized
            // by the rung's own caps: batch <= SK_MAX_BATCH rows of
            // out <= SK_MAX_OUT, SK_MAX_SPLIT chunks each. 4 MB at the caps,
            // and the counters are the same plane one word per (row, out).
            sk_part: exec.alloc(if kq_lanes {
                super::SK_MAX_BATCH * super::SK_MAX_OUT * super::SK_MAX_SPLIT
            } else {
                1
            })?,
            sk_cnt: exec.alloc_u32(if kq_lanes {
                super::SK_MAX_BATCH * super::SK_MAX_OUT
            } else {
                1
            })?,
            // the widest activation any dense plane reads is the 4-stream state
            x16: exec.alloc_f16(walk_rows * cfg.hc_width())?,
            xb16: exec.stream_alloc_bf16(walk_rows * cfg.hc_width())?,
            // the low-M arm runs at batch <= 8 only, so this is sized by the
            // widest plane (q at 12288) and not by the prefill width
            f16_ok: false,
            lowm_ok: false,
            f16_max: usize::MAX,
        };
        // The KV pool, planned last against what is left: weights, per-slot
        // state and the walk scratch are in the ledger by now, so the grant
        // is exactly what the pool may take (pages.rs) - its pages hold the
        // live KV, the prefix cache's retained pages and its checkpoints.
        let kv_row_bytes = cfg.n_kv_heads * cfg.head_dim * KV().bytes();
        let idx_row_bytes = if qsa { cfg.idx_head_dim * 2 } else { 0 };
        let n_attn = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b, Qwen4ExpBlock::Attention))
            .count();
        let geo = (slots >= 1 && !super::prefix::prefix_disabled())
            .then(|| super::prefix::CkptGeometry::of(&cfg));
        let plan = super::pages::plan_pool(
            exec,
            max_tokens,
            slots,
            n_attn,
            kv_row_bytes,
            idx_row_bytes,
            geo.map(|g| g.bytes()),
        )?;
        let (kv_k, kv_v, idx_cache) =
            alloc_pool_planes(exec, &cfg, plan.blocks, kv_row_bytes, qsa)?;
        let pages = super::pages::KvPages::new(exec, slots, max_tokens, &plan)?;
        let prefix = match (geo, plan.ckpt) {
            (Some(geo), Some((n_ckpt, _))) => {
                let layout =
                    checkpoint_layout(exec, &kv_k, &kv_v, &idx_cache, kv_row_bytes, idx_row_bytes);
                Some(super::prefix::PrefixCache::new(
                    exec, geo, layout, slots, n_ckpt,
                )?)
            }
            _ => None,
        };
        let me = Self {
            exec: exec.clone(),
            cfg,
            weights_bytes,
            st,
            layers,
            embed,
            lm_head,
            final_mix,
            max_tokens,
            walk_rows,
            sc,
            recur,
            kv_k,
            kv_v,
            idx_cache,
            idx_ring,
            pages,
            gdn_win,
            ple_win,
            slots,
            cur_slots: vec![0usize],
            cur_runs: Vec::new(),
            walk_lead: 0,
            walk_qsa: false,
            qsa_ready,
            qsa_mode: qsa_mode_env(),
            pos: vec![0usize; slots],
            stream: vec![Vec::new(); slots],
            batch_graphs: (0..=slots).map(|_| [None, None]).collect(),
            stage,
            decode_graph: [None, None],
            graph_capture: capture_wanted(),
            prefix,
            walk_row0: 0,
            walk_cuts: Vec::new(),
            walk_ckpts: Vec::new(),
            reply_ckpt: vec![None; slots],
            reply_pinned: vec![None; slots],
            reply_track: vec![false; slots],
            mtp: None,
            verify: None,
            spec_open: None,
            chunked: Vec::new(),
            spec_rs_draws: None,
        };
        Ok(me)
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    /// Prefill a fresh prompt: resets every carried state, runs all prompt
    /// tokens, and leaves the cursor, conv windows, GDN recurrence and KV
    /// ready for [`Self::decode_step`]. Returns the FINAL position's logits.
    pub fn forward_prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let n = ids.len();
        if n == 0 || n > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "prompt of {n} tokens; this lane is sized for 1..={}",
                self.max_tokens
            )));
        }
        self.reset()?;
        // the PLE hash reads a 2-token EOS-primed stream (vLLM `ngram_context`)
        self.stream[0] = vec![self.cfg.bos_id as i64; 2];
        self.stream[0].extend(ids.iter().map(|&i| i as i64));
        let logits = self.walk(ids, Phase::Prefill)?;
        self.pos[0] = n;
        Ok(logits)
    }

    /// Continue the live sequence by one token off the carried state - the
    /// GDN recurrence, both conv windows, the KV cache and the position
    /// cursor all pick up where the prefill (or the previous step) left them.
    pub fn decode_step(&mut self, id: u32) -> Result<Vec<f32>, GpuModelError> {
        if self.pos[0] == 0 {
            return Err(GpuModelError::Unsupported(
                "decode_step before any prompt - call forward_prompt first".into(),
            ));
        }
        if self.pos[0] >= self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "sequence reached {} tokens, the size this lane was built for",
                self.max_tokens
            )));
        }
        let timing = std::env::var_os("PADDOCK_Q38FN_TIMING").is_some();
        let t_stage = std::time::Instant::now();
        self.stream[0].push(id as i64);
        self.stage_inputs(&[id])?;
        let d_stage = t_stage.elapsed();
        self.walk_qsa = self.qsa_for(self.pos[0]);
        let mode = self.walk_qsa as usize;
        if self.decode_graph[mode].is_none() && self.graph_capture {
            self.capture_decode_tick()?;
        }
        let t_launch = std::time::Instant::now();
        match self.decode_graph[mode].as_ref() {
            Some(g) => g
                .launch()
                .map_err(|e| crate::gpu::GpuError::Driver(format!("decode graph replay: {e}")))?,
            None => self.device_walk(1, Phase::Decode)?,
        }
        let d_launch = t_launch.elapsed();
        let t_copy = std::time::Instant::now();
        let logits = self.exec.to_host_len(&self.sc.d_out, self.cfg.vocab)?;
        let d_copy = t_copy.elapsed();
        if timing {
            eprintln!(
                "[tick] stage_inputs {:7.3} ms | graph_launch {:7.3} ms | logits_d2h+wait {:7.3} ms",
                d_stage.as_secs_f64() * 1e3,
                d_launch.as_secs_f64() * 1e3,
                d_copy.as_secs_f64() * 1e3,
            );
        }
        self.pos[0] += 1;
        Ok(logits)
    }

    /// Zero just one slot's carried state, leaving every other slot alone.
    fn reset_slot(&mut self, slot: usize) -> Result<(), GpuModelError> {
        self.reply_track[slot] = false;
        self.reply_ckpt[slot] = None;
        self.reply_pinned[slot] = None;
        // a fresh sequence writes its own pages: the old ones may be the
        // radix's too (a publish retains a slot's pages), and a page it
        // shares must never be written
        self.pages.release(slot);
        let st = self.cfg.gdn_v_heads * self.cfg.gdn_k_dim * self.cfg.gdn_v_dim;
        for r in self.recur.iter_mut().flatten() {
            self.exec.zero_region(r, slot * st, st)?;
        }
        let wl = (self.cfg.gdn_conv - 1) * self.cfg.gdn_qkv_rows();
        for w in self.gdn_win.iter_mut().flatten() {
            self.exec.zero_region(w, slot * wl, wl)?;
        }
        if let Some(w) = self.ple_win.as_mut() {
            let pl = (self.cfg.ple_conv - 1) * PLE_DILATION * self.cfg.hc_width();
            self.exec.zero_region(w, slot * pl, pl)?;
        }
        self.pos[slot] = 0;
        self.stream[slot].clear();
        self.mtp_clear(slot);
        Ok(())
    }

    /// Prefill walk for one slot: `ids[from..to]` of one sequence at positions
    /// `from..to`, all rows carrying that slot id. `from > 0` continues a
    /// sequence whose carried state (KV rows, recurrence, conv windows) is
    /// already the state after `from` tokens - a prefix-cache resume or the
    /// chunk after a checkpoint cut; the stream must hold the whole prompt.
    fn walk_span(
        &mut self,
        slot: usize,
        ids: &[u32],
        from: usize,
        to: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        self.walk_span_dev(slot, ids, from, to)?;
        Ok(self.exec.to_host_len(&self.sc.d_out, self.cfg.vocab)?)
    }

    /// [`Self::walk_span`] without the logits readback: the span's last row
    /// is left in `d_out` row 0 (a mid-prompt span of the mixed tick has no
    /// use for it).
    fn walk_span_dev(
        &mut self,
        slot: usize,
        ids: &[u32],
        from: usize,
        to: usize,
    ) -> Result<(), GpuModelError> {
        let n = to - from;
        self.back_rows([(slot, to)])?;
        let pos: Vec<u32> = (from as u32..to as u32).collect();
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        let slots: Vec<u32> = vec![slot as u32; n];
        self.exec.upload_u32(&ids[from..to], &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&slots, &mut self.sc.d_slots)?;
        for li in 0..self.cfg.n_layer {
            if let Some(ple) = self.layers[li].ple.as_ref() {
                match ple.table.as_ref() {
                    Some(tab) => {
                        let ids = ple_row_ids(&self.cfg, ple, &self.stream[slot], 2 + from, n)?;
                        stage_ple_device(&self.exec, &self.cfg, ple, tab, &ids, &mut self.sc)?;
                    }
                    None => {
                        let emb = gather_ple_rows(
                            &self.st,
                            &self.cfg,
                            ple,
                            li,
                            &self.stream[slot],
                            2 + from,
                            n,
                        )?;
                        self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
                    }
                }
            }
        }
        self.cur_slots = vec![slot; n];
        self.walk_row0 = from;
        self.walk_qsa = self.qsa_for(to - 1);
        let walked = self.device_walk(n, Phase::Prefill);
        self.walk_row0 = 0;
        walked
    }

    /// The prefix-cache consult for `slot`: the resume point - the slot's
    /// table pointed at the cached pages under it, its carried state restored
    /// from the checkpoint's pages - or 0 (nothing touched). A continued
    /// conversation resumes the same way wherever its last turn ran: the
    /// radix holds that turn's pages and its reply checkpoint.
    fn prefix_resume(&mut self, slot: usize, ids: &[u32]) -> Result<usize, GpuModelError> {
        let Some(pc) = self.prefix.as_mut() else {
            return Ok(0);
        };
        pc.resume(
            &self.exec,
            slot,
            ids,
            &mut self.pages,
            super::prefix::SlotState {
                recur: &mut self.recur,
                gdn_win: &mut self.gdn_win,
                ple_win: self.ple_win.as_mut(),
            },
        )
    }

    /// File checkpoint cut `c` of `ids` in `slot`, whose carried state is the
    /// state after `c` right now, under the radix.
    fn prefix_cut(&mut self, slot: usize, ids: &[u32], c: usize) -> Result<(), GpuModelError> {
        self.prefix_publish(slot, ids, c, true)?;
        Ok(())
    }

    /// Attach the in-walk checkpoints a walk of `ids` wrote (`reserved`: cut,
    /// checkpoint index - their staging blobs already committed into the
    /// checkpoints' pages); a miss gives the index and its pages back.
    fn attach_cuts(&mut self, ids: &[u32], reserved: &[(usize, u32)]) {
        if let Some(pc) = self.prefix.as_mut() {
            for &(c, idx) in reserved {
                pc.attach_reserved(ids, c, idx, &mut self.pages);
            }
        }
    }

    /// Commit the in-walk checkpoints the walk just run wrote (`walk_ckpts`,
    /// staging blob order) into their checkpoints' pages - before the next
    /// walk reuses the staging blobs.
    fn commit_walk_cuts(&mut self) -> Result<(), GpuModelError> {
        let ck = std::mem::take(&mut self.walk_ckpts);
        if let Some(pc) = self.prefix.as_mut() {
            for (blob, &idx) in ck.iter().enumerate() {
                pc.commit_staged(&self.exec, &self.pages, blob, idx)?;
            }
        }
        Ok(())
    }

    /// Give back reserved checkpoints a failed walk never filed.
    fn recycle_reserved(&mut self, reserved: &[(usize, u32)]) {
        if let Some(pc) = self.prefix.as_mut() {
            for &(_, idx) in reserved {
                pc.recycle_ckpt(idx, &mut self.pages);
            }
        }
    }

    fn prefix_publish(
        &mut self,
        slot: usize,
        ids: &[u32],
        upto: usize,
        snapshot: bool,
    ) -> Result<Option<u32>, GpuModelError> {
        let Some(pc) = self.prefix.as_mut() else {
            return Ok(None);
        };
        pc.publish(
            &self.exec,
            slot,
            ids,
            upto,
            snapshot,
            &mut self.pages,
            super::prefix::SlotState {
                recur: &mut self.recur,
                gdn_win: &mut self.gdn_win,
                ple_win: self.ple_win.as_mut(),
            },
        )
    }

    // ---- stage F: the reply checkpoint (the qwen35 / nemotron design) ----
    //
    // A prompt's two cuts are filed during its prefill; without more, the
    // next turn - the same history plus this reply plus a new message -
    // resumes at the prompt's last cut and re-walks the whole reply. So a
    // tracked slot checkpoints its reply too: every time a decode tick
    // closes a 16-token page, the page is filed under the radix (the slot's page
    // already holds its rows) and the carried state is snapshotted there,
    // replacing the slot's previous reply checkpoint. One rolling
    // checkpoint per reply; the next turn prefills <= 15 reply tokens plus
    // its new message. Kill: PADDOCK_NO_REPLY_CKPT=1.

    /// The dense attention walk is the model only while a row sees at most
    /// `QSA_DENSE_EXACT` tokens: past that, QSA attends to its indexer's 512
    /// selected blocks of 4 plus the tail, and dense attends to everything - a
    /// different model (the Flash-Next QSA design note). When dense serves
    /// past it anyway - a pack without the QSA kernels, or `QsaMode::Dense`
    /// pinned - say so the first time a sequence crosses it, rather than
    /// serve the difference silently. Once per process: the condition is the
    /// lane's, not the request's.
    fn qsa_dense_guard(&self, slot: usize) {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let dense = !self.qsa_ready || self.qsa_mode == QsaMode::Dense;
        if dense
            && self.pos[slot] > QSA_DENSE_EXACT
            && !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                slot,
                visible = self.pos[slot],
                "qwen4exp: a sequence passed {QSA_DENSE_EXACT} visible tokens on the DENSE attention \
                 path - the model attends through QSA (512 selected blocks of 4) there, which this \
                 lane does not implement yet, so outputs past this point are not the reference \
                 model's"
            );
        }
    }

    /// Whether a walk whose deepest row sits at position `max_pos` takes QSA
    /// (see `QsaMode`): a row there sees `max_pos + 1` tokens.
    fn qsa_for(&self, max_pos: usize) -> bool {
        self.qsa_ready
            && match self.qsa_mode {
                QsaMode::Auto => max_pos + 1 > QSA_DENSE_EXACT,
                QsaMode::Sparse => true,
                QsaMode::Dense => false,
            }
    }

    /// `qsa_for` over a runs walk: its deepest row ends one of the runs.
    fn qsa_for_runs(&self, runs: &[Run]) -> bool {
        let deepest = runs.iter().map(|r| r.row0 + r.len.max(1) - 1).max();
        deepest.is_some_and(|p| self.qsa_for(p))
    }

    /// Pin the attention mode (a gate instrument; `QsaMode::Auto` serves).
    /// Returns false when the lane cannot run QSA at all (an older pack).
    pub fn set_qsa_mode(&mut self, mode: QsaMode) -> bool {
        self.qsa_mode = mode;
        self.qsa_ready
    }

    /// Admission: the prompt just landed in `slot`. Track its reply when the
    /// cache is on and the sequence is long enough to be worth a checkpoint.
    fn reply_track_admit(&mut self, slot: usize) {
        self.qsa_dense_guard(slot);
        self.reply_ckpt[slot] = None;
        self.reply_pinned[slot] = None;
        self.reply_track[slot] = self.prefix.is_some()
            && !crate::gpu_model::prefix_cache::reply_ckpt_disabled()
            && self.pos[slot] >= super::prefix::MIN_SNAPSHOT_LEN;
    }

    /// A decode tick advanced `rows` (its copies queue behind the walk on
    /// the stream): every tracked slot whose new position closes a page gets
    /// its reply checkpoint there.
    fn reply_after_rows(&mut self, rows: &[(usize, u32)]) -> Result<(), GpuModelError> {
        for &(sl, _) in rows {
            self.qsa_dense_guard(sl);
            if self.reply_track[sl] && self.pos[sl].is_multiple_of(BLOCK_TOKENS) {
                self.reply_snapshot(sl)?;
            }
        }
        Ok(())
    }

    /// File the reply so far (the pages the prompt's publish did not already
    /// hold) and snapshot the slot's carried state at its position; the
    /// slot's previous reply checkpoint is dropped. The prompt's own cuts
    /// stay - the prefill filed those, not this.
    fn reply_snapshot(&mut self, slot: usize) -> Result<(), GpuModelError> {
        let cut = self.pos[slot];
        if self.stream[slot].len() != cut + 2 {
            // the stream and the position parted (a reset the tracking did
            // not see): stop rather than file a mismatched sequence
            self.reply_track[slot] = false;
            return Ok(());
        }
        let tokens: Vec<u32> = self.stream[slot][2..].iter().map(|&t| t as u32).collect();
        let Some(idx) = self.prefix_publish(slot, &tokens, cut, true)? else {
            return Ok(());
        };
        self.reply_filed(slot, &tokens, cut, idx);
        Ok(())
    }

    /// Reply checkpoint `idx` is filed at `cut` of `tokens` (the slot's
    /// sequence up to there): it becomes the slot's rolling one and the
    /// previous rolling one is dropped.
    fn reply_filed(&mut self, slot: usize, tokens: &[u32], cut: usize, idx: u32) {
        if let Some((old_cut, old_idx)) = self.reply_ckpt[slot].replace((cut, idx))
            && let Some(pc) = self.prefix.as_mut()
        {
            pc.drop_ckpt(tokens, old_cut, old_idx, &mut self.pages);
        }
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!("qwen4exp-reply-ckpt: slot {slot} cut {cut} idx {idx}");
        }
    }

    /// The reply just started its first tool call (see
    /// `Generator::reply_pin`): the rolling checkpoint becomes the held one
    /// and the next page's snapshot opens a new rolling one instead of
    /// dropping it. Once per reply.
    fn reply_pin(&mut self, slot: usize) {
        if slot >= self.reply_pinned.len() || self.reply_pinned[slot].is_some() {
            return;
        }
        self.reply_pinned[slot] = self.reply_ckpt[slot].take();
    }

    /// Idle slots give their pages back to the pool - the radix keeps what
    /// it filed (the prompt's pages, the reply's closed pages and their
    /// checkpoints), so the conversation resumes from there wherever it
    /// lands next. The slot's cursor and stream go too: nothing may address
    /// its empty table before the next admission resumes or resets it. A
    /// slot with a prompt still queued for chunked prefill is not idle.
    pub fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for slot in 0..self.slots {
            // every scheduler pass calls this: the cheap checks first
            if occupied.get(slot).copied() == Some(true)
                || self.pages.blocks(slot).is_empty()
                || self.prefill_queue_impl().iter().any(|q| q.0 == slot)
            {
                continue;
            }
            self.reply_track[slot] = false;
            self.reply_ckpt[slot] = None;
            self.reply_pinned[slot] = None;
            self.pages.release(slot);
            self.pos[slot] = 0;
            self.stream[slot].clear();
            self.mtp_clear(slot);
        }
    }

    /// Walk `ids[from..to]` of `slot` - its state already the state after
    /// `from` - in walks of at most `walk_rows`, split at ABSOLUTE multiples
    /// of it, so a resumed walk meets exactly the boundaries the cold walk
    /// did. Each walk seeds the MTP head with its own rows. `cuts` are the
    /// in-walk checkpoints (absolute position, reserved pool index): each
    /// rides the walk it falls strictly inside; one landing ON a boundary is
    /// the state that walk ends in - filed by a snapshot publish there, and
    /// its reservation given back (removed from `cuts`). The last walk's
    /// final-row logits are returned.
    fn walk_bounded(
        &mut self,
        slot: usize,
        ids: &[u32],
        from: usize,
        to: usize,
        cuts: &mut Vec<(usize, u32)>,
    ) -> Result<Vec<f32>, GpuModelError> {
        let w = self.walk_rows;
        let mut a = from;
        loop {
            let b = to.min((a / w + 1) * w);
            let inside: Vec<(usize, u32)> = cuts
                .iter()
                .filter(|&&(c, _)| c > a && c < b)
                .copied()
                .collect();
            self.walk_cuts = inside
                .iter()
                .enumerate()
                .map(|(blob, &(c, _))| (c - a, blob as u32))
                .collect();
            self.walk_ckpts = inside.iter().map(|&(_, idx)| idx).collect();
            let walked = if b == to {
                self.walk_span(slot, ids, a, b).map(Some)
            } else {
                self.walk_span_dev(slot, ids, a, b).map(|()| None)
            };
            self.walk_cuts.clear();
            let logits = match walked {
                Ok(l) => {
                    self.commit_walk_cuts()?;
                    l
                }
                Err(e) => {
                    self.walk_ckpts.clear();
                    return Err(e);
                }
            };
            self.mtp_seed(slot, 0, a, b, ids)?;
            self.pos[slot] = b;
            if b == to {
                return Ok(logits.expect("the last walk reads its logits back"));
            }
            if let Some(i) = cuts.iter().position(|&(c, _)| c == b) {
                let (_, idx) = cuts.remove(i);
                self.recycle_reserved(&[(b, idx)]);
                self.prefix_cut(slot, ids, b)?;
            }
            a = b;
        }
    }

    /// Prefill `ids[start..]` into `slot` whose state is already the state
    /// after `start` tokens (0 = fresh: reset first). Splits at the prompt's
    /// checkpoint cuts so the state can be snapshotted there - the cold and
    /// the resumed walks of one prompt then share the same chunk geometry,
    /// which is what makes a resume bit-identical to the cold run.
    fn prefill_from(
        &mut self,
        slot: usize,
        ids: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        let n = ids.len();
        if start == 0 {
            self.reset_slot(slot)?;
        }
        self.stream[slot] = vec![self.cfg.bos_id as i64; 2];
        self.stream[slot].extend(ids.iter().map(|&i| i as i64));
        let cuts = if self.prefix.is_some() {
            super::prefix::ckpt_cuts(n)
        } else {
            [0, 0]
        };
        let mut pos = start;
        self.mtp_begin(slot, start);
        // The checkpoints taken inside the one walk: a cut walk on this
        // routed-expert family costs rows, not a weight pass (~290 ms of a
        // 1024-token prompt as two 16-row walks). The pn recurrence is the
        // one that can stop at a cut row, so its absence keeps the cut walks.
        if self.prefix.is_some()
            && super::inwalk_ckpt_enabled()
            && self.exec.has_gated_delta_recurrent_pn()
            && super::gdn_pn_enabled()
        {
            let mut reserved: Vec<(usize, u32)> = Vec::new();
            if let Some(pc) = self.prefix.as_mut() {
                for c in cuts {
                    if c > pos
                        && c < n
                        && let Some(idx) = pc.reserve_ckpt(&mut self.pages)
                    {
                        reserved.push((c, idx));
                    }
                }
            }
            let walked = self.walk_bounded(slot, ids, pos, n, &mut reserved);
            let logits = match walked {
                Ok(l) => l,
                Err(err) => {
                    self.recycle_reserved(&reserved);
                    return Err(err);
                }
            };
            self.pos[slot] = n;
            self.prefix_publish(slot, ids, n, false)?;
            self.attach_cuts(ids, &reserved);
            self.reply_track_admit(slot);
            return Ok(logits);
        }
        for c in cuts {
            if c <= pos || c >= n {
                continue;
            }
            self.walk_bounded(slot, ids, pos, c, &mut Vec::new())?;
            self.pos[slot] = c;
            self.prefix_cut(slot, ids, c)?;
            pos = c;
        }
        let logits = self.walk_bounded(slot, ids, pos, n, &mut Vec::new())?;
        self.pos[slot] = n;
        self.prefix_publish(slot, ids, n, false)?;
        self.reply_track_admit(slot);
        Ok(logits)
    }

    /// Advance `rows` INDEPENDENT slots by one token each, in one walk.
    ///
    /// Each row carries its own GDN recurrence, both conv windows, its KV cache
    /// and its own position cursor - the three carried-state kernels route to
    /// their `_slots` entries and everything else in the walk is already
    /// row-parallel because prefill uses it that way. Returns one logit vector
    /// per row, in the order given.
    ///
    /// Runs eager: the captured tick is shaped for the single-sequence lane,
    /// and a batched capture wants one graph per WIDTH (a later rung).
    pub fn decode_step_batch(
        &mut self,
        rows: &[(usize, u32)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if rows.len() > self.walk_rows {
            return Err(GpuModelError::Unsupported(format!(
                "{} rows; scratch is sized for {}",
                rows.len(),
                self.walk_rows
            )));
        }
        let mut seen = vec![false; self.slots];
        for &(sl, _) in rows {
            if sl >= self.slots {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} but this instance carries {}",
                    self.slots
                )));
            }
            if seen[sl] {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} appears twice in one batched step"
                )));
            }
            seen[sl] = true;
            if self.pos[sl] == 0 {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} has no prompt - prefill it first"
                )));
            }
            if self.pos[sl] >= self.max_tokens {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} reached {} tokens",
                    self.max_tokens
                )));
            }
        }
        let timing = std::env::var_os("PADDOCK_Q38FN_TIMING").is_some();
        let t_stage = std::time::Instant::now();
        for &(sl, id) in rows {
            self.stream[sl].push(id as i64);
        }
        self.stage_inputs_rows(rows)?;
        let d_stage = t_stage.elapsed();
        let n = rows.len();
        // One capture per WIDTH serves every slot set: since the PLE window
        // became a position-indexed ring (slot 533) nothing in the batched
        // walk reads the slot set on the host - the slots and positions ride
        // `d_slots`/`d_pos`, which the graph names rather than bakes. Before
        // that the capture was pinned to a dense slot set, so any hole in the
        // scheduler's occupied prefix dropped the tick to an eager walk.
        self.walk_qsa = self.qsa_for(rows.iter().map(|&(sl, _)| self.pos[sl]).max().unwrap_or(0));
        let mode = self.walk_qsa as usize;
        if self.graph_capture && self.batch_graphs[n][mode].is_none() {
            self.capture_batch_tick(n)?;
        }
        let t_launch = std::time::Instant::now();
        match self.batch_graphs[n][mode].as_ref() {
            Some(g) => g
                .launch()
                .map_err(|e| crate::gpu::GpuError::Driver(format!("batched graph replay: {e}")))?,
            None => self.device_walk(n, Phase::DecodeBatch)?,
        }
        let d_launch = t_launch.elapsed();
        let t_copy = std::time::Instant::now();
        let all = self
            .exec
            .to_host_len(&self.sc.d_out, rows.len() * self.cfg.vocab)?;
        let d_copy = t_copy.elapsed();
        if timing {
            eprintln!(
                "[tick{n}] stage {:7.3} | launch {:7.3} | logits_d2h+wait {:7.3} ms",
                d_stage.as_secs_f64() * 1e3,
                d_launch.as_secs_f64() * 1e3,
                d_copy.as_secs_f64() * 1e3,
            );
        }
        for &(sl, _) in rows {
            self.pos[sl] += 1;
        }
        self.mtp_note_rows(rows)?;
        self.reply_after_rows(rows)?;
        Ok(all.chunks(self.cfg.vocab).map(|c| c.to_vec()).collect())
    }

    /// The batched decode tick without the logits readback - every validation,
    /// stage and replay `decode_step_batch` does, stopping at the point where
    /// `d_out` holds `[rows, vocab]` on device. Split out so the device-sampled
    /// path never pays the readback: at this model's 248,320-wide vocab that is
    /// 0.99 MB per token at c1 and 31.8 MB per step at c32, which dominated the
    /// first serving measurement (27.6 ms/tok through the server against 7.9
    /// in a bare loop).
    fn decode_batch_walk(&mut self, rows: &[(usize, u32)]) -> Result<(), GpuModelError> {
        if rows.len() > self.walk_rows {
            return Err(GpuModelError::Unsupported(format!(
                "{} rows; scratch is sized for {}",
                rows.len(),
                self.walk_rows
            )));
        }
        let mut seen = vec![false; self.slots];
        for &(sl, _) in rows {
            if sl >= self.slots {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} but this instance carries {}",
                    self.slots
                )));
            }
            if seen[sl] {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} appears twice in one batched step"
                )));
            }
            seen[sl] = true;
            if self.pos[sl] == 0 {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} has no prompt - prefill it first"
                )));
            }
            if self.pos[sl] >= self.max_tokens {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} reached {} tokens",
                    self.max_tokens
                )));
            }
        }
        for &(sl, id) in rows {
            self.stream[sl].push(id as i64);
        }
        self.stage_inputs_rows(rows)?;
        let n = rows.len();
        // One capture per WIDTH serves every slot set: since the PLE window
        // became a position-indexed ring (slot 533) nothing in the batched
        // walk reads the slot set on the host - the slots and positions ride
        // `d_slots`/`d_pos`, which the graph names rather than bakes. Before
        // that the capture was pinned to a dense slot set, so any hole in the
        // scheduler's occupied prefix dropped the tick to an eager walk.
        self.walk_qsa = self.qsa_for(rows.iter().map(|&(sl, _)| self.pos[sl]).max().unwrap_or(0));
        let mode = self.walk_qsa as usize;
        if self.graph_capture && self.batch_graphs[n][mode].is_none() {
            self.capture_batch_tick(n)?;
        }
        match self.batch_graphs[n][mode].as_ref() {
            Some(g) => g
                .launch()
                .map_err(|e| crate::gpu::GpuError::Driver(format!("batched graph replay: {e}")))?,
            None => self.device_walk(n, Phase::DecodeBatch)?,
        }
        for &(sl, _) in rows {
            self.pos[sl] += 1;
        }
        self.mtp_note_rows(rows)?;
        self.reply_after_rows(rows)?;
        Ok(())
    }

    /// Pack the per-row sampling plans into the device param words. Mode codes
    /// match the shared `pd_sample_rows` family every other lane uses:
    /// 1 = greedy, 2 = temperature-categorical, 5/6 = truncation (5 when the
    /// top-k head fits the 64-wide superset the device selection walks).
    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> (Vec<u32>, Option<Vec<u32>>) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        let mut any_trunc = false;
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
                    par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, any_trunc.then_some(tpar))
    }

    /// Sample `d_out` rows 0..r on device; only `Host`-plan rows pay a
    /// vocab-row readback. `d_out` already holds this tick's logits.
    /// The scheduler passes positions explicitly while this model tracks them
    /// per slot; they are CHECKED rather than trusted, so a desync fails
    /// loudly instead of silently decoding at the wrong position - the
    /// failure mode a wrong-position KV read gives is PLAUSIBLE TEXT, which
    /// no gate would catch. Only the rows that actually decode are checked:
    /// hole rows carry a placeholder (0, 0) that means nothing.
    fn check_positions(
        pos: &[usize],
        rows: &[(usize, u32)],
        positions: &[u32],
    ) -> Result<(), crate::generator::GenError> {
        for &(i, _) in rows {
            if pos[i] != positions[i] as usize {
                return Err(crate::generator::GenError::Backend(format!(
                    "slot {i}: scheduler says position {}, model is at {}",
                    positions[i], pos[i]
                )));
            }
        }
        Ok(())
    }

    fn sample_rows_from_logits(
        &mut self,
        rows: &[(usize, u32)],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        // `d_out` holds one row per DECODED row, in `rows` order - hole rows
        // never reached the walk. Plans are indexed by the scheduler's slot,
        // so they are compacted the same way before packing and the ids are
        // scattered back at the end.
        let live: Vec<RowSample> = rows.iter().map(|&(i, _)| plans[i]).collect();
        let (packed, host) = self.sample_out_rows(&live)?;
        let mut ids = vec![0u32; plans.len()];
        for (j, &(i, _)) in rows.iter().enumerate() {
            ids[i] = packed[j];
        }
        let host_rows = host.into_iter().map(|(j, row)| (rows[j].0, row)).collect();
        Ok(SampledStep { ids, host_rows })
    }

    /// Sample `d_out` rows `[0, plans.len())` on device, row j by `plans[j]`:
    /// the ids in row order, and the `Host`-plan rows' logits read back as
    /// (row, logits).
    fn sample_out_rows(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<(Vec<u32>, Vec<(usize, Vec<f32>)>), GpuModelError> {
        use crate::generator::RowSample;
        let vocab = self.cfg.vocab;
        let exec = self.exec.clone();
        let r = plans.len();
        if r == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let (par, tpar) = Self::pack_samp_par(plans);
        {
            let sc = &mut self.sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| crate::gpu::GpuError::Driver("d_par slice".into()))?;
            exec.stream
                .memcpy_htod(&par, &mut v)
                .map_err(crate::gpu::from_driver)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..r * 4)
                    .ok_or_else(|| crate::gpu::GpuError::Driver("d_tpar slice".into()))?;
                exec.stream
                    .memcpy_htod(t, &mut v)
                    .map_err(crate::gpu::from_driver)?;
            }
            exec.sample_rows_at(&sc.d_out, &sc.d_par, 0, &mut sc.d_ids, 0, r, vocab)?;
            if tpar.is_some() {
                exec.sample_rows_t_at(
                    &sc.d_out,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_ids,
                    0,
                    r,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    &sc.d_out,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_ids,
                    0,
                    r,
                    vocab,
                )?;
            }
        }
        let ids_view = self
            .sc
            .d_ids
            .try_slice(0..r)
            .ok_or_else(|| crate::gpu::GpuError::Driver("d_ids slice".into()))?;
        let packed = exec
            .stream
            .clone_dtoh(&ids_view)
            .map_err(crate::gpu::from_driver)?;
        let mut host_rows = Vec::new();
        for (j, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                host_rows.push((j, self.out_row(j)?));
            }
        }
        Ok((packed, host_rows))
    }

    /// `d_out` row `j` read back.
    fn out_row(&self, j: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.cfg.vocab;
        let v = self
            .sc
            .d_out
            .try_slice(j * vocab..(j + 1) * vocab)
            .ok_or_else(|| crate::gpu::GpuError::Driver("d_out row slice".into()))?;
        Ok(self
            .exec
            .stream
            .clone_dtoh(&v)
            .map_err(crate::gpu::from_driver)?)
    }

    /// Back the pool pages a walk's rows reach - `(slot, positions up to)` -
    /// and upload the table if it grew. Every walk stages through one of the
    /// `stage_inputs*` entries or `walk_span_dev`, which call this before
    /// the walk (or the captured replay) reads the table; a verify span is
    /// backed whole, so a draft row past a page boundary never lands on
    /// another slot's page.
    fn back_rows(
        &mut self,
        reach: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<(), GpuModelError> {
        // A page a table takes may hold anything - with checkpoints in the
        // pool, a dropped checkpoint's f32 record (Inf / NaN read as f16).
        // No attention kernel of this lane lets a row past its keys into a
        // product (P6i and tc5 zero the V rows past the last key as they
        // stage them, the rest never load them), so nothing is cleared here.
        for (slot, upto) in reach {
            self.pages.back(slot, upto)?;
        }
        self.pages.sync(&self.exec)
    }

    /// Stage one token per row, each against its own slot and position.
    fn stage_inputs_rows(&mut self, rows: &[(usize, u32)]) -> Result<(), GpuModelError> {
        let n = rows.len();
        let reach: Vec<(usize, usize)> = rows.iter().map(|&(s, _)| (s, self.pos[s] + 1)).collect();
        self.back_rows(reach)?;
        let ids: Vec<u32> = rows.iter().map(|&(_, t)| t).collect();
        let slots: Vec<u32> = rows.iter().map(|&(s, _)| s as u32).collect();
        let pos: Vec<u32> = rows.iter().map(|&(s, _)| self.pos[s] as u32).collect();
        // mrope carries the same position four times, section-major
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        self.cur_slots = rows.iter().map(|&(s, _)| s).collect();
        self.exec.upload_u32(&ids, &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&slots, &mut self.sc.d_slots)?;
        for li in 0..self.cfg.n_layer {
            if let Some(ple) = self.layers[li].ple.as_ref() {
                // each row hashes its own stream at its own position
                match ple.table.as_ref() {
                    Some(tab) => {
                        let heads = self.cfg.ple_heads();
                        let mut ids = Vec::with_capacity(n * heads);
                        for &(sl, _) in rows {
                            ids.extend_from_slice(&ple_row_ids(
                                &self.cfg,
                                ple,
                                &self.stream[sl],
                                self.pos[sl] + 2,
                                1,
                            )?);
                        }
                        stage_ple_device(&self.exec, &self.cfg, ple, tab, &ids, &mut self.sc)?;
                    }
                    None => {
                        let mut emb = Vec::with_capacity(n * self.cfg.ple_embed);
                        for &(sl, _) in rows {
                            let one = gather_ple_rows(
                                &self.st,
                                &self.cfg,
                                ple,
                                li,
                                &self.stream[sl],
                                self.pos[sl] + 2,
                                1,
                            )?;
                            emb.extend_from_slice(&one);
                        }
                        self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Prefill SEVERAL prompts in one walk - the scheduler's whole admitted
    /// wave. Row `i` of the fused planes belongs to run `r` at position
    /// `i - off_r`, and `d_slots`/`d_pos` carry that per row, which is all the
    /// row-parallel majority of the walk needs. Returns each prompt's last
    /// logits, in the order given.
    ///
    /// Why this exists: the scheduler already calls `forward_prefill_batch`
    /// with the whole wave, and the trait default prefills one at a time. At
    /// c32 that is a 1.66 s blocking prefill tick (32 x ~50 ms) against a
    /// 14.6 ms decode tick - TTFT p50 1706 ms and 40% of the cell's wall.
    ///
    /// v1 scope: FRESH prompts only (each slot is reset first). Every op is
    /// shared except the three sequence-shaped ones - and of those, the two
    /// convs are just their Prefill entry at a row offset, so only the
    /// recurrence needed a new kernel (slot 534).
    pub fn prefill_slots(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if items.len() == 1 {
            // one run is the single-sequence walk, and that one is captured,
            // fork-enabled and already the best shape for it
            return Ok(vec![self.prefill_slot(items[0].0, &items[0].1)?]);
        }
        let n: usize = items.iter().map(|(_, t)| t.len()).sum();
        if n > self.walk_rows {
            return Err(GpuModelError::Unsupported(format!(
                "prefill wave of {n} rows; the walk is sized for {}",
                self.walk_rows
            )));
        }
        let mut seen = vec![false; self.slots];
        for (slot, toks) in items {
            if *slot >= self.slots {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {slot} but this instance carries {}",
                    self.slots
                )));
            }
            if seen[*slot] {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {slot} appears twice in one prefill wave"
                )));
            }
            seen[*slot] = true;
            if toks.is_empty() {
                return Err(GpuModelError::Unsupported(
                    "a prefill wave carries an empty prompt".into(),
                ));
            }
        }

        // where every item starts: the prefix cache's resume point (its
        // carried state restored), or 0 with the slot reset - a fresh run's
        // conv arms rely on a zero window at their offset base, a resumed
        // run's on the re-staged one (see the `row0` arms)
        let mut starts = Vec::with_capacity(items.len());
        for (slot, toks) in items {
            let start = self.prefix_resume(*slot, toks)?;
            if start == 0 {
                self.reset_slot(*slot)?;
            }
            self.mtp_begin(*slot, start);
            self.stream[*slot] = vec![self.cfg.bos_id as i64; 2];
            self.stream[*slot].extend(toks.iter().map(|&i| i as i64));
            starts.push(start);
        }
        // every item's stops: its checkpoint cuts past the start, then its end
        // - the same chunk geometry the single-slot path walks, so a cold
        // cohort leaves the checkpoints its next turn resumes from
        let stops: Vec<Vec<usize>> = items
            .iter()
            .zip(&starts)
            .map(|((_, toks), &start)| {
                let n = toks.len();
                let mut v: Vec<usize> = if self.prefix.is_some() {
                    super::prefix::ckpt_cuts(n)
                        .into_iter()
                        .filter(|&c| c > start && c < n)
                        .collect()
                } else {
                    Vec::new()
                };
                v.push(n);
                v
            })
            .collect();
        let mut cur = starts;
        let mut done = vec![false; items.len()];
        let mut out: Vec<Option<Vec<f32>>> = vec![None; items.len()];
        let mut stage = 0usize;
        while done.iter().any(|d| !d) {
            let mut runs = Vec::with_capacity(items.len());
            let mut which = Vec::with_capacity(items.len());
            let mut off = 0usize;
            for (i, (slot, _)) in items.iter().enumerate() {
                if done[i] {
                    continue;
                }
                let to = stops[i][stage.min(stops[i].len() - 1)];
                if to <= cur[i] {
                    continue;
                }
                runs.push(Run {
                    slot: *slot,
                    off,
                    len: to - cur[i],
                    row0: cur[i],
                });
                which.push(i);
                off += to - cur[i];
            }
            if runs.is_empty() {
                break;
            }
            let rows = off;
            self.cur_runs = runs.clone();
            self.cur_slots = runs
                .iter()
                .flat_map(|r| std::iter::repeat_n(r.slot, r.len))
                .collect();
            let prompts: Vec<(usize, Vec<u32>)> = which.iter().map(|&i| items[i].clone()).collect();
            self.stage_inputs_runs(&prompts, &runs)?;
            self.walk_qsa = self.qsa_for_runs(&runs);
            let walked = self.device_walk(rows, Phase::PrefillRuns);
            self.cur_runs.clear();
            walked?;
            let all = self
                .exec
                .to_host_len(&self.sc.d_out, runs.len() * self.cfg.vocab)?;
            // the head seeds off the wave's streams, run by run in row order
            for (&i, r) in which.iter().zip(&runs) {
                self.mtp_seed(r.slot, r.off, r.row0, r.row0 + r.len, &items[i].1)?;
            }
            for (k, (&i, r)) in which.iter().zip(&runs).enumerate() {
                let (slot, toks) = &items[i];
                let to = r.row0 + r.len;
                self.pos[*slot] = to;
                cur[i] = to;
                if to == toks.len() {
                    self.prefix_publish(*slot, toks, to, false)?;
                    self.reply_track_admit(*slot);
                    out[i] = Some(all[k * self.cfg.vocab..(k + 1) * self.cfg.vocab].to_vec());
                    done[i] = true;
                } else {
                    self.prefix_cut(*slot, toks, to)?;
                }
            }
            stage += 1;
        }
        Ok(out
            .into_iter()
            .map(|l| l.expect("every item walked"))
            .collect())
    }

    /// Everything a `PrefillRuns` walk reads from the host: the fused token
    /// plane, each row's position within its own run, the slot map, the run
    /// table, and the PLE n-gram ids hashed against each run's own stream.
    fn stage_inputs_runs(
        &mut self,
        items: &[(usize, Vec<u32>)],
        runs: &[Run],
    ) -> Result<(), GpuModelError> {
        let n: usize = runs.iter().map(|r| r.len).sum();
        let mut ids = Vec::with_capacity(n);
        for ((_, toks), r) in items.iter().zip(runs) {
            ids.extend_from_slice(&toks[r.row0..r.row0 + r.len]);
        }
        self.stage_inputs_runs_ids(&ids, runs, 0)
    }

    /// [`Self::stage_inputs_runs`] over an already-assembled token plane (row
    /// order = run order): the verify round stages its chunks without the
    /// whole sequence in hand. The PLE hash still reads each slot's stream,
    /// which must already hold the rows being staged. A mixed walk's leading
    /// `lead` decode runs (see `walk_lead`) stage their rows like any other
    /// but stay out of the tile and run tables - their attention and
    /// carried-state ops take the decode entries, which read `d_slots` and
    /// `d_pos` per row.
    fn stage_inputs_runs_ids(
        &mut self,
        ids: &[u32],
        runs: &[Run],
        lead: usize,
    ) -> Result<(), GpuModelError> {
        let n: usize = runs.iter().map(|r| r.len).sum();
        self.back_rows(runs.iter().map(|r| (r.slot, r.row0 + r.len)))?;
        let mut pos = Vec::with_capacity(n);
        let mut slots = Vec::with_capacity(n);
        for r in runs {
            pos.extend(r.row0 as u32..(r.row0 + r.len) as u32);
            slots.extend(std::iter::repeat_n(r.slot as u32, r.len));
        }
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        self.exec.upload_u32(ids, &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&slots, &mut self.sc.d_slots)?;
        let seq = &runs[lead.min(runs.len())..];
        let mut t_row0: Vec<u32> = Vec::with_capacity(n_qtiles(seq));
        let mut t_slot: Vec<u32> = Vec::with_capacity(n_qtiles(seq));
        for r in seq {
            let mut row = r.off;
            while row < r.off + r.len {
                t_row0.push(row as u32);
                t_slot.push(r.slot as u32);
                row += PD_APF_TQ;
            }
        }
        self.exec.upload_u32(&t_row0, &mut self.sc.d_tile_row0)?;
        self.exec.upload_u32(&t_slot, &mut self.sc.d_tile_slot)?;
        let roff: Vec<u32> = seq.iter().map(|r| r.off as u32).collect();
        let rlen: Vec<u32> = seq.iter().map(|r| r.len as u32).collect();
        let rslot: Vec<u32> = seq.iter().map(|r| r.slot as u32).collect();
        self.exec.upload_u32(&roff, &mut self.sc.d_run_off)?;
        self.exec.upload_u32(&rlen, &mut self.sc.d_run_len)?;
        self.exec.upload_u32(&rslot, &mut self.sc.d_run_slot)?;
        // the runs are contiguous in walk rows, so the prefix table is each
        // run's first row and the last run's end
        if let Some(last) = seq.last() {
            let mut offs = roff.clone();
            offs.push((last.off + last.len) as u32);
            self.exec.upload_u32(&offs, &mut self.sc.d_run_offs)?;
        }
        for li in 0..self.cfg.n_layer {
            if let Some(ple) = self.layers[li].ple.as_ref() {
                match ple.table.as_ref() {
                    Some(tab) => {
                        let heads = self.cfg.ple_heads();
                        let mut pids = Vec::with_capacity(n * heads);
                        for r in runs {
                            pids.extend_from_slice(&ple_row_ids(
                                &self.cfg,
                                ple,
                                &self.stream[r.slot],
                                2 + r.row0,
                                r.len,
                            )?);
                        }
                        stage_ple_device(&self.exec, &self.cfg, ple, tab, &pids, &mut self.sc)?;
                    }
                    None => {
                        let mut emb = Vec::with_capacity(n * self.cfg.ple_embed);
                        for r in runs {
                            emb.extend_from_slice(&gather_ple_rows(
                                &self.st,
                                &self.cfg,
                                ple,
                                li,
                                &self.stream[r.slot],
                                2 + r.row0,
                                r.len,
                            )?);
                        }
                        self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Prefill slot `slot` with `ids`, leaving its carried state ready for
    /// `decode_step_batch`. Slot 0 is what `forward_prompt` drives.
    pub fn prefill_slot(&mut self, slot: usize, ids: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        if slot >= self.slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} but this instance carries {}",
                self.slots
            )));
        }
        let n = ids.len();
        if n == 0 || n > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "prompt of {n} tokens; this lane is sized for 1..={}",
                self.max_tokens
            )));
        }
        let start = self.prefix_resume(slot, ids)?;
        self.prefill_from(slot, ids, start)
    }

    /// Record the decode tick as a CUDA graph. Capture RECORDS without
    /// executing, so the caller still has to replay once for the step to
    /// happen. Valid at every position: each per-token input was staged into
    /// an address-stable buffer beforehand, and every kernel reads its
    /// position from the device.
    fn capture_decode_tick(&mut self) -> Result<(), GpuModelError> {
        self.exec
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| crate::gpu::GpuError::Driver(format!("begin_capture: {e}")))?;
        let walked = self.device_walk(1, Phase::Decode);
        // end the capture even if the walk failed, or the stream stays in
        // capture mode and every later launch fails with a confusing error
        let graph = crate::gpu::end_capture_no_flags(&self.exec.stream)
            .map_err(|e| crate::gpu::GpuError::Driver(format!("end_capture: {e}")));
        walked?;
        self.decode_graph[self.walk_qsa as usize] = graph?.map(Q4xSendGraph);
        Ok(())
    }

    /// Record one batched tick of width `n` as a CUDA graph. Only valid for the
    /// dense slot set `0..n` - see `batch_graphs`.
    fn capture_batch_tick(&mut self, n: usize) -> Result<(), GpuModelError> {
        self.exec
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| crate::gpu::GpuError::Driver(format!("begin_capture: {e}")))?;
        let walked = self.device_walk(n, Phase::DecodeBatch);
        let graph = crate::gpu::end_capture_no_flags(&self.exec.stream)
            .map_err(|e| crate::gpu::GpuError::Driver(format!("end_capture: {e}")));
        walked?;
        self.batch_graphs[n][self.walk_qsa as usize] = graph?.map(Q4xSendGraph);
        Ok(())
    }

    /// Greedy continuation: prefill `prompt`, then take up to `n_new` steps,
    /// stopping at any configured EOS. Returns the generated ids.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        n_new: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        let mut logits = self.forward_prompt(prompt)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let id = argmax(&logits) as u32;
            out.push(id);
            if self.cfg.eos_ids.contains(&id) {
                break;
            }
            logits = self.decode_step(id)?;
        }
        Ok(out)
    }

    /// Turn decode-tick graph capture on or off. Dropping an existing capture
    /// is safe at any time - it is a pure accelerator over the eager walk, and
    /// the two are gated against each other in
    /// `tests/gpu_qwen4exp_forward.rs`.
    pub fn set_graph_capture(&mut self, on: bool) {
        self.graph_capture = on;
        if !on {
            self.decode_graph = [None, None];
            // the batched ticks too: a width already captured would otherwise
            // keep replaying, and an "eager" A/B would silently time the graph
            for g in self.batch_graphs.iter_mut() {
                *g = [None, None];
            }
        }
    }

    /// Whether the decode tick is currently running as a captured graph.
    pub fn graph_active(&self) -> bool {
        self.decode_graph.iter().any(|g| g.is_some())
    }

    /// Resident weight bytes measured at load - see the `weights_bytes` field.
    /// `Generator::weights_mem_bytes` forwards to this, which is what
    /// `/api/stats` publishes and the catalog shape survey measures a
    /// shape from; without it the picker prices this family by FILE size and
    /// refuses a card that serves it (76 GB of download against ~60 resident).
    pub fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_bytes
    }

    /// The dense weight class this model actually loaded - for benchmarks and
    /// the PPL gate, so the class is never implicit in a number.
    pub fn dense_class(&self) -> &'static str {
        self.layers
            .first()
            .map(|l| l.attn_hc.down.class())
            .unwrap_or("none")
    }

    /// How many tokens the live sequence holds (0 = nothing started).
    pub fn position(&self) -> usize {
        self.pos[0]
    }

    /// Position cursor of one slot.
    pub fn slot_position(&self, slot: usize) -> usize {
        self.pos[slot]
    }

    /// Lower the walk budget - a GATE instrument, so one loaded model can
    /// hold the whole walk against the bounded walks a long context takes.
    /// Only down (the scratch was sized at load) and in whole KV pages.
    pub fn cap_walk_rows(&mut self, w: usize) -> Result<(), GpuModelError> {
        if w == 0 || !w.is_multiple_of(16) || w > self.walk_rows {
            return Err(GpuModelError::Unsupported(format!(
                "walk budget {w}: a multiple of 16 no larger than {}",
                self.walk_rows
            )));
        }
        self.walk_rows = w;
        Ok(())
    }

    /// Diagnostic read of the QSA compressed key cache: attention layer
    /// `layer`'s keys for `slot`, blocks `b0..b1`, as `[b1-b0, idx_head_dim]`
    /// f32. None when the lane keeps no indexer cache there (a GDN layer, a
    /// pack without the indexer). For gates; the serving path never reads
    /// the cache back.
    pub fn qsa_index_keys(
        &self,
        layer: usize,
        slot: usize,
        b0: usize,
        b1: usize,
    ) -> Result<Option<Vec<f32>>, GpuModelError> {
        let Some(cache) = self.idx_cache.get(layer).and_then(|c| c.as_ref()) else {
            return Ok(None);
        };
        let (hd, cap) = (self.cfg.idx_head_dim, self.max_tokens.div_ceil(QSA_BLOCK));
        if slot >= self.slots || b0 > b1 || b1 > cap {
            return Err(GpuModelError::Unsupported(format!(
                "qsa_index_keys: slot {slot} blocks {b0}..{b1} outside {} x {cap}",
                self.slots
            )));
        }
        // block b of the slot is row table[b / 4] * 4 + b % 4 of the plane
        let pages = self.pages.blocks(slot);
        if b1.div_ceil(super::pages::IDX_ROWS_PER_PAGE) > pages.len() {
            return Err(GpuModelError::Unsupported(format!(
                "qsa_index_keys: slot {slot} backs {} blocks, asked for {b1}",
                pages.len() * super::pages::IDX_ROWS_PER_PAGE
            )));
        }
        let all = self.exec.to_host_bf16(cache)?;
        let rpp = super::pages::IDX_ROWS_PER_PAGE;
        let mut out = Vec::with_capacity((b1 - b0) * hd);
        for b in b0..b1 {
            let row = pages[b / rpp] as usize * rpp + b % rpp;
            out.extend(all[row * hd..(row + 1) * hd].iter().map(|v| v.to_f32()));
        }
        Ok(Some(out))
    }

    /// How many sequences this instance is sized for.
    pub fn slot_count(&self) -> usize {
        self.slots
    }

    /// Drop every per-sequence state. The allocations stay - a later capture
    /// rung bakes these addresses.
    fn reset(&mut self) -> Result<(), GpuModelError> {
        // fresh sequences write their own pages (see `reset_slot`)
        for slot in 0..self.slots {
            self.pages.release(slot);
        }
        let state_len = self.slots * self.cfg.gdn_v_heads * self.cfg.gdn_k_dim * self.cfg.gdn_v_dim;
        for r in self.recur.iter_mut().flatten() {
            self.exec.zero_region(r, 0, state_len)?;
        }
        let win_len = self.slots * (self.cfg.gdn_conv - 1) * self.cfg.gdn_qkv_rows();
        for w in self.gdn_win.iter_mut().flatten() {
            self.exec.zero_region(w, 0, win_len)?;
        }
        if let Some(w) = self.ple_win.as_mut() {
            let n = self.slots * (self.cfg.ple_conv - 1) * PLE_DILATION * self.cfg.hc_width();
            self.exec.zero_region(w, 0, n)?;
        }
        for p in self.pos.iter_mut() {
            *p = 0;
        }
        for st in self.stream.iter_mut() {
            st.clear();
        }
        for s in 0..self.slots {
            self.mtp_clear(s);
        }
        // the captured tick survives: it names buffers, not values, and every
        // one of them is address-stable for the life of this model
        Ok(())
    }

    /// The layer walk, shared by both phases. `ids` are the tokens to run and
    /// they start at the current cursor; the KV cache and both conv windows
    /// are read AND advanced, so a prefill of n then k decode steps is the
    /// same arithmetic as a prefill of n+k (that equality is the gate in
    /// `tests/gpu_qwen4exp_forward.rs`).
    fn walk(&mut self, ids: &[u32], phase: Phase) -> Result<Vec<f32>, GpuModelError> {
        self.stage_inputs(ids)?;
        self.walk_qsa = self.qsa_for(self.pos[0] + ids.len().max(1) - 1);
        self.device_walk(ids.len(), phase)?;
        Ok(self.exec.to_host_len(&self.sc.d_out, self.cfg.vocab)?)
    }

    /// Everything a tick reads from the host: the token ids, the position and
    /// mrope planes, the zero slot map, and the PLE n-gram rows (a pure
    /// function of the token stream, so it is known before the tick runs).
    /// Kept out of the device walk because an H2D copy from pageable memory is
    /// capture-illegal - and because staging is what makes one captured tick
    /// valid at every position.
    fn stage_inputs(&mut self, ids: &[u32]) -> Result<(), GpuModelError> {
        let n = ids.len();
        let base = self.pos[0];
        self.back_rows([(0, base + n)])?;
        let pos: Vec<u32> = (0..n).map(|i| (base + i) as u32).collect();
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        self.exec.upload_u32(ids, &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&vec![0u32; n], &mut self.sc.d_slots)?;
        for li in 0..self.cfg.n_layer {
            if let Some(ple) = self.layers[li].ple.as_ref() {
                match ple.table.as_ref() {
                    Some(tab) => {
                        let ids = ple_row_ids(&self.cfg, ple, &self.stream[0], base + 2, n)?;
                        stage_ple_device(&self.exec, &self.cfg, ple, tab, &ids, &mut self.sc)?;
                    }
                    None => {
                        let emb = gather_ple_rows(
                            &self.st,
                            &self.cfg,
                            ple,
                            li,
                            &self.stream[0],
                            base + 2,
                            n,
                        )?;
                        self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The device-only layer walk. Contains no host reads, no allocation and
    /// no data-dependent host branching, so a decode-shaped call is capturable
    /// as-is (the `Dump` triage path is the one exception, and it refuses to
    /// coexist with capture).
    fn device_walk(&mut self, n: usize, phase: Phase) -> Result<(), GpuModelError> {
        let Self {
            exec: e,
            cfg: c,
            layers,
            embed,
            lm_head,
            final_mix,
            max_tokens,
            sc,
            recur,
            kv_k,
            kv_v,
            idx_cache,
            idx_ring,
            gdn_win,
            ple_win,
            stage,
            cur_slots,
            cur_runs,
            walk_lead,
            walk_qsa,
            walk_row0,
            walk_cuts,
            prefix,
            verify,
            pages,
            ..
        } = self;
        let row0 = *walk_row0;
        // a mixed walk's decode rows lead; the per-run arms see the prompts
        let lead = if matches!(phase, Phase::PrefillRuns) {
            (*walk_lead).min(cur_runs.len())
        } else {
            0
        };
        let seq_runs = &cur_runs[lead..];
        // in-walk checkpoints: the pool the passes write them into, and each
        // GDN layer's place among the GDN layers (the checkpoint layout)
        let mut sink = if walk_cuts.is_empty() {
            None
        } else {
            prefix.as_mut().map(|p| p.ckpt_sink())
        };
        let n_gdn = c
            .blocks
            .iter()
            .filter(|b| matches!(b, Qwen4ExpBlock::Gdn))
            .count();
        let mut gdn_ord = 0usize;
        let (h, hw, hc, lr, eps) = (c.hidden, c.hc_width(), c.hc_count, c.hc_lowrank, c.eps);
        // Fork eligibility. The two branches both call `DensePlane::matmul`,
        // and only the F8Row class stages activations through the shared
        // `DenseStage` - the bf16 class never touches it, so its branches
        // cannot contend and the forks are safe at any width.
        // `gdn_fork_enabled()` belongs in this conjunction: the two gates
        // below are hazard bounds that exist only because a side-stream
        // kernel may be resident. With forks off no side stream is ever
        // created, so clamping the f16 K-split then costs occupancy and buys
        // nothing -- hc down (320 rows out of K=10240) falls to a 5-CTA
        // launch on a 148-SM machine.
        let fork_ok = matches!(phase, Phase::Decode | Phase::DecodeBatch)
            && super::gdn_fork_enabled()
            && layers
                .first()
                .map(|l| matches!(l.attn_hc.down, super::DensePlane::Bf16(_)))
                .unwrap_or(false)
            // the k-quant dense class stages its batched activations in the
            // one DenseStage; a side-stream twin would race it at width > 1
            && !matches!(lm_head, DensePlane::Kq { .. });
        // Declare co-residency to the f16 lane for this walk. A forked walk
        // may have a side-stream kernel resident, and the lane's K-split is a
        // cross-CTA spin whose factor assumes it owns the machine - so the
        // gate clamps that split rather than the lane being refused outright.
        // Read at dispatch, so a graph captured here bakes the election.
        // Two gates, both hazard bounds on the shared f16 lane, both set per
        // walk and read at dispatch so a captured graph bakes the election.
        //
        //  * K-SPLIT (535): the tc5g split is a cross-CTA spin whose factor is
        //    elected from `2*nsm / U0`, i.e. assuming this launch owns the
        //    machine. A forked walk breaks that and the device hangs.
        //  * MMAF (411): the fine-tile arm is RACY. `bench/q4x_dense_probe.cu`
        //    runs it back to back with the shipped bf16 route, one
        //    cudaDeviceSynchronize apart, and it returns garbage on a set of
        //    (plane, batch) pairs that MOVES between runs - hc up rel 7.5e+02
        //    at b8 in one run and clean at b8 in the next; gdn z and ple val
        //    likewise. Always inside its own batch 5..32 window, never at
        //    b2/b4/b6 or b64/b128, and it disappears completely and stably
        //    over three runs with the arm declined. That is a race inside a
        //    SHIPPED kernel other families call, not something this lane
        //    introduced; declining it here is a bound, and the race wants its
        //    own fix.
        // lowm (543) is a DECODE lever: serial prefill at batch<=8 would
        // elect it while the wave (n>8) cannot, splitting the two prefill
        // paths into different f32 orders - prefill_wave_matches_serial
        // FAILED exactly there in the round-4b battery.
        stage.lowm_ok = matches!(phase, Phase::Decode | Phase::DecodeBatch) && !sc.lowm_refused;
        stage.prefill = matches!(phase, Phase::Prefill | Phase::PrefillRuns);
        // A verify walk's dense planes run row-exact against the decode tick
        // (see `DenseStage::row_exact`). `PADDOCK_Q38FN_VERIFY_EXACT=0` keeps
        // them on the batched GEMMs (A/B).
        stage.row_exact = verify.as_ref().is_some_and(|v| v.active) && spec::verify_exact_on();
        // MIRROR EXPIRY (walk-scoped). A bf16 mirror is valid only inside the
        // walk that wrote it. The buffers never move, so a pointer match alone
        // survives across walks: a mirror written on an n==1 walk was matched
        // by a later batch walk and read as fresh, which corrupted every
        // batched cell and faked two wins (+16.5%, +8.6%) on 2026-09-01
        // before the next measurement contradicted them. Clearing here makes
        // that class impossible instead of relying on every writer to
        // invalidate; `mir_*_n` still bounds the row count within a walk.
        stage.f16_ok = e.has_f16_ksplit_set() && e.has_f16_mmaf_gate();
        if stage.f16_ok {
            e.f16_ksplit_set(!fork_ok)?;
            // MMAF RE-ENABLED 2026-08-29. The arm was declined because it
            // returned garbage on a MOVING set of (plane, batch) pairs inside
            // its own 5..32 window. Root-caused with bench/mmaf_race.cu:
            // a warp PAIR is two warps (rg=0,1) sharing the pair's ring slot,
            // and the park that ends the kernel overwrites that slot -- but
            // the loop's only synchronisation is the per-slot mbarrier pair,
            // which orders each warp against the PRODUCER and not against its
            // sibling. rg=0 could start parking while rg=1 still had ldmatrix
            // in flight on the same slot (racecheck: read f16_dense.cuh:2468
            // vs writes 2499-2502). Fixed by a CTA-wide barrier before the
            // park, off the K loop. Bit gate over 600 (plane,batch) runs:
            // 155 bad -> 0 bad. `PADDOCK_Q38FN_MMAF=0` declines it again.
            e.f16_mmaf_set(super::mmaf_enabled());
        }
        stage.f16_max = if fork_ok {
            super::f16_fork_max_batch()
        } else {
            usize::MAX
        };
        let dump = Dump::arm(n);
        let mut pm = PhaseMs::arm(phase);
        PHASE_MS.with(|p| *p.borrow_mut() = pm.on.then(|| PhaseMs::arm(phase)));

        // ---- embed -> the 4-stream hyper-connection state -----------------
        match embed {
            Embed::Bf16(t) => e.embed_gather_bf16(t, &sc.d_tok, &mut sc.d_x, h, n, 1.0)?,
            Embed::Kq(t) => e.kquant_gather(t, &sc.d_tok, &mut sc.d_x, h, n)?,
            Embed::Q8(t) => e.embed_gather_q8(t, &sc.d_tok, &mut sc.d_x, h, n, 1.0)?,
        }
        for t in 0..n {
            for s in 0..hc {
                e.copy_region(&sc.d_x, t * h, &mut sc.d_h, t * hw + s * h, h)?;
            }
        }
        pm_lap(e, "embed");
        pm.lap(e, "embed");
        dump.put(e, usize::MAX, "h_embed", &sc.d_h, n * hw)?;

        // Carries whether the previous combine already left this mix's
        // normalized state in d_xn. False at entry: the first mix reads the
        // freshly broadcast embedding, which no combine produced.
        let mut pre_normed = false;
        // ...and whether it also left the next hc down its mmq rows (slot 602)
        let mut pre_preq = false;
        // ...and, when it stored no state at all, the aux plane its 1/rms went to
        let mut pre_rn: Option<(usize, bool)> = None;
        // A verify walk's rows are few and each is its slot's next position, so
        // the decode attention (per-row position and slot, KV appended before
        // it reads) is exact for them. It is also the class a decode tick
        // runs - which is what a verify row has to agree with - and at 2 rows
        // it cost half of the prefill tile (phase census 2026-09-14: 3.4 ms
        // decode, 7.4-8.3 ms tile at 2-8 rows). Prefill keeps its tile.
        let attn_phase = if verify.as_ref().is_some_and(|v| v.active) && n <= 64 {
            Phase::DecodeBatch
        } else {
            phase
        };
        for li in 0..c.n_layer {
            let layer = &layers[li];
            if let Some(ple) = layer.ple.as_ref() {
                // rows already staged into d_emb by `stage_inputs`
                ple_pass(
                    e,
                    c,
                    ple,
                    sc,
                    stage,
                    n,
                    phase,
                    ple_win.as_mut().expect("ple window"),
                    cur_slots,
                    seq_runs,
                    lead,
                    row0,
                    walk_cuts,
                    sink.as_mut(),
                    n_gdn,
                )?;
                pm_lap(e, "ple");
                pm.lap(e, "ple");
                debug_assert!(
                    !pre_normed,
                    "a PLE layer must never be handed a pre-normed state"
                );
                if dump.on() {
                    dump.put(e, li, "ple_gv", &sc.d_pgv, n * hw)?;
                    dump.put(e, li, "ple_conv", &sc.d_pconv, n * hw)?;
                    dump.put(e, li, "h_ple", &sc.d_h, n * hw)?;
                }
            }

            let attn_inj = hc_mix_pass(
                e,
                c,
                &layer.attn_hc,
                sc,
                stage,
                n,
                pre_normed,
                pre_preq,
                pre_rn,
            )?;
            pm.lap(e, "hc_attn");
            dump.put(e, li, "hc_bi_attn", &sc.d_bi, n * h)?;
            dump.put(e, li, "hc_m_attn", &sc.d_m, n * c.hc_lowrank)?;
            dump.put(e, li, "hc_inj_attn", &sc.d_inj, n * hc)?;
            if dump.on() {
                dump.put(e, li, "attn_bi", &sc.d_bi, n * h)?;
                dump_inj(e, &dump, li, "attn_inj", sc, attn_inj, n, hc)?;
            }
            match c.blocks[li] {
                Qwen4ExpBlock::Gdn => {
                    let MixerW::Gdn(w) = &layer.mixer else {
                        return Err(GpuModelError::Unsupported(
                            "gdn layer has an attn mixer".into(),
                        ));
                    };
                    gdn_pass(
                        e,
                        c,
                        w,
                        sc,
                        stage,
                        recur[li].as_mut().expect("gdn state"),
                        gdn_win[li].as_mut().expect("gdn conv window"),
                        n,
                        phase,
                        cur_slots.first().copied().unwrap_or(0),
                        seq_runs,
                        lead,
                        fork_ok,
                        row0,
                        verify
                            .as_mut()
                            .filter(|v| v.active && spec::verify_exact_on())
                            .map(|v| &mut v.rows),
                        walk_cuts,
                        sink.as_mut(),
                        gdn_ord,
                    )?;
                    gdn_ord += 1;
                    if let Some(vf) = verify.as_mut().filter(|v| v.active) {
                        // a verify round keeps what a rejecting commit replays
                        vf.capture_gdn(e, c, sc, li, n)?;
                    }
                    pm_lap(e, "gdn-tail");
                    pm.lap(e, "gdn");
                    if dump.on() {
                        let (hv, vdim) = (c.gdn_v_heads, c.gdn_v_heads * c.gdn_v_dim);
                        dump.put(e, li, "gdn_qkv", &sc.d_qkv, n * c.gdn_qkv_rows())?;
                        dump.put(e, li, "gdn_conv", &sc.d_conv, n * c.gdn_qkv_rows())?;
                        dump.put(e, li, "gdn_z", &sc.d_zg, n * c.gdn_z_rows())?;
                        dump.put(e, li, "gdn_ab", &sc.d_ab, n * 2 * hv)?;
                        dump.put(e, li, "gdn_g", &sc.d_g, n * hv)?;
                        dump.put(e, li, "gdn_beta", &sc.d_beta, n * hv)?;
                        dump.put(e, li, "gdn_core", &sc.d_core, n * vdim)?;
                        dump.put(e, li, "gdn_final", &sc.d_dattn, n * vdim)?;
                    }
                }
                Qwen4ExpBlock::Attention => {
                    let MixerW::Attn(w) = &layer.mixer else {
                        return Err(GpuModelError::Unsupported(
                            "attn layer has a gdn mixer".into(),
                        ));
                    };
                    attn_pass(
                        e,
                        c,
                        w,
                        sc,
                        stage,
                        kv_k[li].as_mut().expect("kv k"),
                        kv_v[li].as_mut().expect("kv v"),
                        idx_cache[li].as_mut().zip(idx_ring[li].as_mut()),
                        *walk_qsa,
                        *max_tokens,
                        &pages.d_tab,
                        pages.bps,
                        n,
                        attn_phase,
                        seq_runs,
                        lead,
                        fork_ok,
                    )?;
                    pm_lap(e, "attn");
                    pm.lap(e, "attn");
                }
            }
            dump.put(e, li, "mix_out", &sc.d_mix, n * h)?;
            // the mlp mix always reads this combine's own output
            let (mlp_pre, mlp_preq, mlp_rn) = combine(
                e,
                sc,
                attn_inj,
                Some(&layer.mlp_hc.norm),
                Some(&layer.mlp_hc.down),
                Some((&layer.mlp_hc, 2 * li + 1)),
                n,
                hc,
                h,
                eps,
                stage,
            )?;
            pm_lap(e, "combine");
            pm.lap(e, "combine");
            dump.put(e, li, "h_mid", &sc.d_h, n * hw)?;

            let mlp_inj =
                hc_mix_pass(e, c, &layer.mlp_hc, sc, stage, n, mlp_pre, mlp_preq, mlp_rn)?;
            pm.lap(e, "hc_mlp");
            dump.put(e, li, "hc_bi_mlp", &sc.d_bi, n * h)?;
            if dump.on() {
                dump.put(e, li, "mlp_bi", &sc.d_bi, n * h)?;
                dump_inj(e, &dump, li, "mlp_inj", sc, mlp_inj, n, hc)?;
            }
            // Whoever reads the state next: the following layer's attention
            // mix, or the final mixer. Not fusable when the next layer carries
            // a PLE - that layer ADDS to the state before its mix reads it, so
            // a norm taken here would be of the wrong thing. Hoisted above the
            // MoE pass: when the combine below will FUSE the DSL gather (slot
            // 561), moe_dslfork skips its own combine and leaves C_dn.
            let next_norm = if li + 1 < c.n_layer {
                if layers[li + 1].ple.is_some() {
                    None
                } else {
                    Some(&layers[li + 1].attn_hc.norm)
                }
            } else {
                Some(&final_mix.norm)
            };
            // the plane that reads that state's mmq rows when the combine may
            // emit them: the next layer's attention hc down, never the final
            // mixer (it projects d_xn on its own silu arm)
            let next_down = (li + 1 < c.n_layer && layers[li + 1].ple.is_none())
                .then(|| &layers[li + 1].attn_hc.down);
            // ...and that mix itself with its aux plane, exactly where the down is
            let next_rn = next_down
                .is_some()
                .then(|| (&layers[li + 1].attn_hc, 2 * (li + 1)));
            // ORDER: (.., fork_ok, decode). These two were swapped once, which
            // pinned `decode` false for the whole batch band - the folded
            // router, the fused shared gate|up and the low-M dense arms all
            // sat behind it.
            let _fused = moe_pass(
                e,
                c,
                &layer.moe,
                sc,
                stage,
                n,
                fork_ok,
                matches!(phase, Phase::Decode | Phase::DecodeBatch),
                None,
            )?;
            pm_lap(e, "moe-tail");
            pm.lap(e, "moe");
            dump.put(e, li, "moe_out", &sc.d_mix, n * h)?;
            (pre_normed, pre_preq, pre_rn) = combine(
                e, sc, mlp_inj, next_norm, next_down, next_rn, n, hc, h, eps, stage,
            )?;
            pm_lap(e, "combine");
            pm.lap(e, "combine");
            dump.put(e, li, "h_out", &sc.d_h, n * hw)?;
        }

        // ---- final mixer (no inject) -> lm_head on the last position -----
        if !pre_normed {
            // mirror at the store across the whole low-M band, not just n==1:
            // the HC island reads xn as bf16, and an UNMIRRORED write here
            // would leave mir_xn aimed at this same buffer with stale bytes.
            e.q4x_group_norm_1p(
                &sc.d_h,
                &final_mix.norm.buf,
                &mut sc.d_xn,
                None,
                n,
                hc,
                h,
                eps,
            )?;
        }
        // same scale+silu epilogue fold as hc_mix_pass; `lr` rows, no inject
        let fm_done = {
            let done = final_mix.down.matmul_silu(
                e,
                &sc.d_xn,
                &mut sc.d_m,
                None,
                n,
                lr,
                1.0 / hc as f32,
            )?;
            // record only when the silu path actually wrote the mirror
            done
        };
        if !fm_done {
            final_mix.down.matmul(e, &sc.d_xn, &mut sc.d_m, n, stage)?;
            e.q4x_scale_silu(&mut sc.d_m, n * lr, 1.0 / hc as f32)?;
        }
        let upmix = n == 1
            && super::fuse_upmix_on()
            && match super::plane_bytes(&final_mix.up) {
                Some(wp) => {
                    e.bf16_gemv_up_hcmix(wp, &sc.d_m, &sc.d_xn, &mut sc.d_bi, None, h, hc)?
                }
                None => false,
            };
        if !upmix {
            final_mix.up.matmul(e, &sc.d_m, &mut sc.d_gate, n, stage)?;
            e.q4x_hc_mix(&sc.d_xn, &sc.d_gate, &mut sc.d_bi, None, n, hc, h)?;
        }
        pm.lap(e, "final_mix");
        if matches!(phase, Phase::DecodeBatch) {
            // every row is a live sequence's next-token distribution
            lm_head.matmul(e, &sc.d_bi, &mut sc.d_out, n, stage)?;
        } else if let Some(vf) = verify.as_mut().filter(|v| v.active) {
            // a verify round: every row is a position the round compares
            lm_head.matmul(e, &sc.d_bi, &mut vf.logits, n, stage)?;
            dump.put(e, usize::MAX, "vlogits", &vf.logits, n * c.vocab)?;
        } else if matches!(phase, Phase::PrefillRuns) {
            // one row per RUN - its own last position - so the head runs once
            // over `runs.len()` rows instead of once per prompt
            for (i, r) in cur_runs.iter().enumerate() {
                e.copy_region(&sc.d_bi, (r.off + r.len - 1) * h, &mut sc.d_fin, i * h, h)?;
            }
            lm_head.matmul(e, &sc.d_fin, &mut sc.d_out, cur_runs.len(), stage)?;
        } else {
            // prefill and the single-sequence tick want only the last position
            e.copy_region(&sc.d_bi, (n - 1) * h, &mut sc.d_fin, 0, h)?;
            dump.put(e, usize::MAX, "fin", &sc.d_fin, h)?;
            lm_head.matmul(e, &sc.d_fin, &mut sc.d_out, 1, stage)?;
        }
        pm.lap(e, "lm_head");
        dump.put(e, usize::MAX, "logits", &sc.d_out, c.vocab)?;
        let nested = PHASE_MS.with(|p| p.borrow_mut().take());
        pm.report(n);
        if let Some(np) = nested {
            np.report_as(n, "moe-split");
        }
        Ok(())
    }
}

/// Which shape the walk runs in. Prefill sees the whole span at once and can
/// use the sequence-form convs and the tiled prefill attention; decode sees
/// one token and reads its history out of the carried windows and the KV.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Prefill,
    /// one token for one sequence, against slot 0's carried state
    Decode,
    /// one token each for `n` different slots, in one walk. Routes the three
    /// carried-state kernels (GDN recurrence, GDN conv window, PLE conv
    /// window) to their `_slots` entries; every other kernel in the walk is
    /// already row-parallel because prefill uses it that way.
    DecodeBatch,
    /// `n_runs` INDEPENDENT prompts in one walk - the scheduler's whole
    /// admitted wave. Every row-parallel op runs once over all rows; only the
    /// three sequence-shaped ones (GDN conv, GDN recurrence, PLE conv) are
    /// per-run, and two of those are just the Prefill entry at a row offset
    /// (their left-pad guard is relative to the offset base, which is a fresh
    /// sequence's zero pad). Prefill attention already takes a per-row
    /// position AND slot vector, so it needs nothing.
    PrefillRuns,
}

/// One prompt inside a `PrefillRuns` walk: rows `off .. off+len` of the staged
/// planes belong to `slot`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Run {
    pub slot: usize,
    pub off: usize,
    pub len: usize,
    /// the sequence position of the run's first row: 0 = a fresh prompt,
    /// otherwise the run continues a slot whose carried state is the state
    /// after `row0` tokens (a prefix-cache resume, or the span after a cut)
    pub row0: usize,
}

/// Graph capture is on unless killed, and never while the triage dump is
/// armed - the dump reads device memory back mid-walk, which is
/// capture-illegal and would poison the whole tick.
fn capture_wanted() -> bool {
    std::env::var_os("PADDOCK_Q38FN_NO_GRAPH").is_none()
        && std::env::var_os("PADDOCK_Q38FN_DUMP").is_none()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Apply the combine against whichever place the mix left its inject, FUSING
/// the grouped norm that consumes the result when there is one - which is
/// every combine except the one whose output a PLE layer modifies before the
/// next mix reads it. Returns whether `d_xn` now holds that mix's normalized
/// state.
#[allow(clippy::too_many_arguments)]
fn combine(
    e: &GpuExecutor,
    sc: &mut Scratch,
    inj: Inj,
    next_norm: Option<&DeviceTensor>,
    // the down plane that reads the normalized state next (None where nothing
    // may take mmq rows): when it takes the pre-quantized pipe rung the norm
    // pass emits them too (slot 602), and the second flag says it did
    next_down: Option<&DensePlane>,
    // the mix that reads the state next and its `d_hcaux` index: when its
    // inject and up + mix both rebuild the state from h (slots 607 / 608) and
    // the down takes the rows above, no state is stored at all (slot 606) and
    // the third return names the aux plane the 1/rms went to
    next_rn: Option<(&HcW, usize)>,
    n: usize,
    hc: usize,
    hidden: usize,
    eps: f32,
    stage: &mut DenseStage,
) -> Result<(bool, bool, Option<(usize, bool)>), GpuModelError> {
    let preq = next_norm.is_some()
        && next_down.is_some_and(|p| p.takes_preq_mmq(e, n, stage))
        && sc.d_hcq.len() >= super::hc_preq_bytes(hc * hidden, n);
    // REBUILD (slots 606 - 608): with the down on the pre-quantized rows, the
    // next mix's inject and up + mix are the state's only readers; when both
    // rebuild it from h the combine stores none - a [rows][hc*hidden] write a
    // call (bench/combine_rn_gb10_bench.cu, 1024 rows: the chain 1223 -> 1022
    // us, every output byte-identical)
    // ...and when the pack folds the next mix's inject into this pass (slot
    // 609), the second field says the logits are already in `d_inj`
    let rn = next_rn
        .filter(|(w, idx)| {
            preq && *idx < sc.d_hcaux.len() && hc_takes_rn(e, w, n, hc, hidden, stage)
        })
        .map(|(w, idx)| {
            let fold = hc_inj_fold_ok(e, w, n, hc, hidden, sc.d_injp.len());
            (idx, fold)
        });
    // xn's bf16 MIRROR: this kernel is the only writer of d_xn on the decode
    // path, so mirroring at the store retires the per-consumer cast (the hc
    // down plane's TGV arm cast a [n, 10240] plane every layer). Any path
    // below that does not mirror invalidates, since d_xn is one buffer.
    // d_h, d_mix, d_m/d_inj and d_xn are disjoint fields, so the borrows below
    // are all simultaneous without any shuffling.
    match (inj, next_norm) {
        (Inj::InM(off), Some(nw)) => {
            if let (Some((idx, true)), Some((w, _))) = (rn, next_rn) {
                let wi = w.inject.as_ref().expect("a folded inject has its weight");
                e.q4x_combine_norm_q8mmq_nsi(
                    &mut sc.d_h,
                    &sc.d_mix,
                    Some((&sc.d_m, off)),
                    &nw.buf,
                    &mut sc.d_hcaux[idx],
                    hc * hidden,
                    &mut sc.d_hcq,
                    &wi.buf,
                    &mut sc.d_injp,
                    &mut sc.d_inj,
                    n,
                    hc,
                    hidden,
                    eps,
                    n.next_multiple_of(128),
                )?;
            } else if let Some((idx, _)) = rn {
                e.q4x_combine_norm_q8mmq_ns(
                    &mut sc.d_h,
                    &sc.d_mix,
                    &sc.d_m,
                    off,
                    &nw.buf,
                    &mut sc.d_hcaux[idx],
                    hc * hidden,
                    &mut sc.d_hcq,
                    n,
                    hc,
                    hidden,
                    eps,
                    n.next_multiple_of(128),
                )?;
            } else if preq {
                e.q4x_combine_norm_q8mmq(
                    &mut sc.d_h,
                    &sc.d_mix,
                    &sc.d_m,
                    off,
                    &nw.buf,
                    &mut sc.d_xn,
                    &mut sc.d_hcq,
                    n,
                    hc,
                    hidden,
                    eps,
                    n.next_multiple_of(128),
                )?;
            } else {
                e.q4x_combine_norm(
                    &mut sc.d_h,
                    &sc.d_mix,
                    &sc.d_m,
                    off,
                    &nw.buf,
                    &mut sc.d_xn,
                    None,
                    n,
                    hc,
                    hidden,
                    eps,
                )?;
            }
            Ok((true, preq, rn))
        }
        (Inj::Separate, Some(nw)) => {
            if let (Some((idx, true)), Some((w, _))) = (rn, next_rn) {
                // this combine's logits are read from d_inj before the fold
                // writes the next mix's there (one buffer, stream order)
                let wi = w.inject.as_ref().expect("a folded inject has its weight");
                e.q4x_combine_norm_q8mmq_nsi(
                    &mut sc.d_h,
                    &sc.d_mix,
                    None,
                    &nw.buf,
                    &mut sc.d_hcaux[idx],
                    hc * hidden,
                    &mut sc.d_hcq,
                    &wi.buf,
                    &mut sc.d_injp,
                    &mut sc.d_inj,
                    n,
                    hc,
                    hidden,
                    eps,
                    n.next_multiple_of(128),
                )?;
            } else if let Some((idx, _)) = rn {
                e.q4x_combine_norm_q8mmq_ns(
                    &mut sc.d_h,
                    &sc.d_mix,
                    &sc.d_inj,
                    0,
                    &nw.buf,
                    &mut sc.d_hcaux[idx],
                    hc * hidden,
                    &mut sc.d_hcq,
                    n,
                    hc,
                    hidden,
                    eps,
                    n.next_multiple_of(128),
                )?;
            } else if preq {
                e.q4x_combine_norm_q8mmq(
                    &mut sc.d_h,
                    &sc.d_mix,
                    &sc.d_inj,
                    0,
                    &nw.buf,
                    &mut sc.d_xn,
                    &mut sc.d_hcq,
                    n,
                    hc,
                    hidden,
                    eps,
                    n.next_multiple_of(128),
                )?;
            } else {
                e.q4x_combine_norm(
                    &mut sc.d_h,
                    &sc.d_mix,
                    &sc.d_inj,
                    0,
                    &nw.buf,
                    &mut sc.d_xn,
                    None,
                    n,
                    hc,
                    hidden,
                    eps,
                )?;
            }
            Ok((true, preq, rn))
        }
        (Inj::InM(off), None) => {
            e.q4x_hc_combine_at(&mut sc.d_h, &sc.d_mix, &sc.d_m, off, n, hc, hidden)?;
            Ok((false, false, None))
        }
        (Inj::Separate, None) => {
            e.q4x_hc_combine(&mut sc.d_h, &sc.d_mix, &sc.d_inj, n, hc, hidden)?;
            Ok((false, false, None))
        }
    }
}

/// Triage dump of the inject logits, wherever the fold put them.
#[allow(clippy::too_many_arguments)]
fn dump_inj(
    e: &GpuExecutor,
    dump: &Dump,
    li: usize,
    tag: &str,
    sc: &Scratch,
    inj: Inj,
    n: usize,
    hc: usize,
) -> Result<(), GpuModelError> {
    match inj {
        Inj::InM(off) => {
            let v = e.to_host_len(&sc.d_m, off + n * hc)?;
            dump.put_host(li, tag, &v[off..off + n * hc])
        }
        Inj::Separate => dump.put(e, li, tag, &sc.d_inj, n * hc),
    }
}

/// One hyper-connection mix: grouped (1+w) norm -> low-rank down -> scale+silu
/// -> up -> gated reduce, plus the raw inject logits. Leaves `d_bi` = block
/// input and `d_inj` = inject logits.
#[allow(clippy::too_many_arguments)]
fn hc_mix_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &HcW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    n: usize,
    pre_normed: bool,
    // the preceding combine also left this pass's down plane its mmq rows in
    // `d_hcq` (slot 602) - consumed here, never kept
    preq: bool,
    // ...and stored no normalized state (slot 606): the inject and the up +
    // mix rebuild it from h off `d_hcaux[rn.0]` (slots 607 / 608) - or, with
    // rn.1, the combine already folded the inject into `d_inj` (slot 609)
    rn: Option<(usize, bool)>,
) -> Result<Inj, GpuModelError> {
    let (h, hc, lr) = (c.hidden, c.hc_count, c.hc_lowrank);
    // `pre_normed`: the preceding combine already produced this mix's
    // normalized state as part of its own single pass (slot 517).
    if !pre_normed {
        // mirror at the store across the whole low-M band, not just n==1:
        // the HC island reads xn as bf16, and an UNMIRRORED write here
        // would leave mir_xn aimed at this same buffer with stale bytes.
        e.q4x_group_norm_1p(&sc.d_h, &w.norm.buf, &mut sc.d_xn, None, n, hc, h, c.eps)?;
    }
    pm_lap(e, "hc-norm");
    let mut silu_done = false;
    let inj = if w.inject_rows > 0 && n == 1 {
        // One launch for both projections: the inject logits come out as the
        // tail of the low-rank output and are read there, so folding the
        // launch does not cost a copy back. The scale+silu that follows is
        // folded into the same launch's epilogue over the low-rank rows -
        // the inject tail must pass through untouched, hence `lr`.
        silu_done = {
            let done =
                w.down
                    .matmul_silu(e, &sc.d_xn, &mut sc.d_m, None, 1, lr, 1.0 / hc as f32)?;
            // record only when the silu path actually wrote the mirror
            done
        };
        if !silu_done {
            w.down.matmul(e, &sc.d_xn, &mut sc.d_m, 1, stage)?;
        }
        pm_lap(e, "hc-down");
        Inj::InM(lr)
    } else if w.inject_rows > 0 {
        // One launch for both segments (the batch-1 arm above already fuses
        // them; above batch 1 the inject tail is not contiguous, so this uses
        // the segmented store instead of two row-range calls). Measured at
        // c32: 2.02 + 2.00 launches/layer and 0.944 + 0.801 ms/step went to
        // one launch - the hc chain was half our whole dense launch count.
        if !w
            .down
            .matmul_2seg(e, lr, hc, &sc.d_xn, &mut sc.d_m, &mut sc.d_inj, n)?
        {
            w.down
                .matmul_rows(e, 0, lr, &sc.d_xn, &mut sc.d_m, n, stage)?;
            w.down
                .matmul_rows(e, lr, hc, &sc.d_xn, &mut sc.d_inj, n, stage)?;
        }
        pm_lap(e, "hc-down");
        Inj::Separate
    } else {
        let taken = preq
            && w.down
                .matmul_preq_mmq(e, &sc.d_hcq, &mut sc.d_m, n, stage)?;
        debug_assert!(
            taken || !preq,
            "a combine emitted mmq rows the hc down did not take"
        );
        if !taken {
            if rn.is_some() {
                return Err(GpuModelError::Unsupported(
                    "hc down declined the mmq rows of a combine that stored no normalized state"
                        .into(),
                ));
            }
            w.down.matmul(e, &sc.d_xn, &mut sc.d_m, n, stage)?;
        }
        pm_lap(e, "hc-down");
        let wi = w.inject.as_ref().expect("unfolded block hc carries inject");
        match rn {
            Some((_, true)) => {} // the combine folded it (slot 609)
            Some((idx, false)) => e.q4x_hc_inject_rn(
                &wi.buf,
                &sc.d_h,
                &sc.d_hcaux[idx],
                &mut sc.d_inj,
                h,
                hc,
                hc,
                n,
            )?,
            None => e.matvec_f32_raw(&wi.buf, hc * h, hc, &sc.d_xn, &mut sc.d_inj, n)?,
        }
        pm_lap(e, "hc-inj");
        Inj::Separate
    };
    if !silu_done {
        e.q4x_scale_silu(&mut sc.d_m, n * lr, 1.0 / hc as f32)?;
    }
    // FUSED mix tail: the up GEMM emits the mixed output straight from its
    // epilogue, so the [rows][hc*hidden] gate plane is never materialised and
    // the separate q4x_hc_mix launch disappears. Bit-exact vs the two-launch
    // path (verified at batch 16/32/64); 1.03 -> 0.79 ms/step at c32 and -96
    // launches. The permute that makes it possible is done once at load.
    // DECODE-BAND only (n <= 32): the fused-epilogue kernel is bf16-only, and
    // at wave widths it was CAPTURING the up plane away from its tc5 twin --
    // 22 ms of the c8 prefill burst ran a 144 us bf16 tile where the f16 lane
    // does the same rows in ~20 us. Above the band the plain matmul below
    // takes the Dual election and the separate q4x_hc_mix launch is noise.
    let fused = rn.is_none()
        && match (&w.up_hcmix, (2..=32).contains(&n)) {
            (Some(wp), true) => {
                e.bf16_hcmix_gemm(wp, &sc.d_m, &sc.d_xn, &mut sc.d_bi, lr, h, hc, n)?
            }
            _ => false,
        };
    if !fused {
        let upmix = n == 1
            && super::fuse_upmix_on()
            && match super::plane_bytes(&w.up) {
                Some(wp) => {
                    e.bf16_gemv_up_hcmix(wp, &sc.d_m, &sc.d_xn, &mut sc.d_bi, None, h, hc)?
                }
                None => false,
            };
        // prefill widths: the up GEMM with the gated mix in its epilogue (slot
        // 605) - the [rows][hc*hidden] gate plane never lands, byte-identical
        // to the two launches below (bench/hcup_mix_gb10_bench.cu, 1024 rows:
        // 613 -> 415 us a call)
        if let Some((idx, _)) = rn {
            // the rung the combine elected on (`hc_takes_rn`); declining now
            // would leave a state nobody stored
            if !w.up.matmul_hcmix_rn(
                e,
                &sc.d_m,
                &sc.d_h,
                &sc.d_hcaux[idx],
                &mut sc.d_bi,
                n,
                hc,
                h,
                stage,
            )? {
                return Err(GpuModelError::Unsupported(
                    "hc up + mix declined the rebuild rung its combine elected".into(),
                ));
            }
        } else if !upmix
            && !w
                .up
                .matmul_hcmix(e, &sc.d_m, &sc.d_xn, &mut sc.d_bi, n, hc, h, stage)?
        {
            w.up.matmul(e, &sc.d_m, &mut sc.d_gate, n, stage)?;
            e.q4x_hc_mix(&sc.d_xn, &sc.d_gate, &mut sc.d_bi, None, n, hc, h)?;
        }
    }
    pm_lap(e, "hc-up");
    Ok(inj)
}

/// Where a mix pass left its inject logits: folded into the tail of the
/// low-rank output at an element offset, or in its own plane.
#[derive(Clone, Copy)]
enum Inj {
    InM(usize),
    Separate,
}

/// PLE n-gram layer: device projections off the host-gathered rows, per-stream
/// gate, dilated conv, then `H += gv + conv`.
#[allow(clippy::too_many_arguments)]
fn ple_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    n: usize,
    phase: Phase,
    win: &mut CudaSlice<f32>,
    slot_ids: &[usize],
    runs: &[Run],
    // a mixed walk's leading decode rows (see `walk_lead`); `runs` is then
    // just the prompt runs behind them
    lead: usize,
    row0: usize,
    // in-walk prefix-cache checkpoints (Prefill): (row in the span, pool
    // index), where their blobs sit, and how many GDN layers precede the ring
    cuts: &[(usize, u32)],
    sink: Option<&mut super::prefix::CkptSink<'_>>,
    n_gdn: usize,
) -> Result<(), GpuModelError> {
    let (h, hw, hc, eps) = (c.hidden, c.hc_width(), c.hc_count, c.eps);
    let wrows = (c.ple_conv - 1) * PLE_DILATION;
    ple.key.matmul(e, &sc.d_emb, &mut sc.d_pkey, n, stage)?;
    ple.value.matmul(e, &sc.d_emb, &mut sc.d_pval, n, stage)?;
    e.q4x_group_norm_1p(
        &sc.d_pkey,
        &ple.norm_key.buf,
        &mut sc.d_pkn,
        None,
        n,
        hc,
        h,
        eps,
    )?;
    e.q4x_group_norm_1p(
        &sc.d_h,
        &ple.norm_query.buf,
        &mut sc.d_pqn,
        None,
        n,
        hc,
        h,
        eps,
    )?;
    e.q4x_ple_gate(&sc.d_pkn, &sc.d_pqn, &sc.d_pval, &mut sc.d_pgv, n, hc, h)?;
    // the conv rides norm_conv(gv); d_pkn is free again, reuse it as the source
    e.q4x_group_norm_1p(
        &sc.d_pgv,
        &ple.norm_conv.buf,
        &mut sc.d_pkn,
        None,
        n,
        hc,
        h,
        eps,
    )?;
    match phase {
        Phase::Prefill => {
            e.q4x_conv_dil(
                &sc.d_pkn,
                &ple.conv.buf,
                &mut sc.d_pconv,
                n,
                hw,
                c.ple_conv,
                PLE_DILATION,
            )?;
            // Seed the RING: the pre-conv row for token position q lives at
            // ring row `q % wrows`, which is the same rule the decode step
            // applies, so the two need no handshake. For n >= wrows the last
            // `wrows` rows land rotated (two contiguous copies); for a prompt
            // shorter than the window the rows land at 0..n and the rest keeps
            // the zeros `reset` left - which under the ring's own indexing is
            // the tail, i.e. the zero left-pad the sequence form applies.
            let pbase = slot_ids.first().copied().unwrap_or(0) * wrows * hw;
            if row0 > 0 {
                // a continued sequence: the ring holds the pre-conv rows of
                // positions row0-wrows..row0 at index (pos % wrows); stage them
                // in order in front of the span's first rows and recompute
                // those rows' conv (the fresh conv above zero-padded them)
                let m = n.min(wrows);
                let s0 = row0 % wrows; // ring index of position row0 - wrows
                let Scratch {
                    d_pkn,
                    d_pconv,
                    d_ple_ext_in,
                    d_ple_ext_out,
                    ..
                } = sc;
                e.copy_region(win, pbase + s0 * hw, d_ple_ext_in, 0, (wrows - s0) * hw)?;
                if s0 > 0 {
                    e.copy_region(win, pbase, d_ple_ext_in, (wrows - s0) * hw, s0 * hw)?;
                }
                e.copy_region(d_pkn, 0, d_ple_ext_in, wrows * hw, m * hw)?;
                e.q4x_conv_dil(
                    d_ple_ext_in,
                    &ple.conv.buf,
                    d_ple_ext_out,
                    wrows + m,
                    hw,
                    c.ple_conv,
                    PLE_DILATION,
                )?;
                e.copy_region(d_ple_ext_out, wrows * hw, d_pconv, 0, m * hw)?;
            }
            // commit the span's last min(n, wrows) rows at their ring index
            // (pos % wrows); for a fresh sequence this is the two-copy
            // wrap-around commit above exactly
            let m = n.min(wrows);
            let q0 = row0 + n - m; // position of the first committed row
            let r = q0 % wrows;
            let len1 = m.min(wrows - r);
            e.copy_region(&sc.d_pkn, (n - m) * hw, win, pbase + r * hw, len1 * hw)?;
            if len1 < m {
                e.copy_region(&sc.d_pkn, (n - m + len1) * hw, win, pbase, (m - len1) * hw)?;
            }
            // in-walk checkpoints: the ring a walk ending at each cut row would
            // have committed (a cut sits at least a page into the walk, so all
            // `wrows` rows come from this span)
            if let Some(sk) = sink {
                for &(rc, idx) in cuts {
                    debug_assert!(rc >= wrows && rc <= n, "cut row {rc} of a {n}-row walk");
                    let base = sk.ple_off(idx, n_gdn);
                    let r = (row0 + rc - wrows) % wrows;
                    let len1 = wrows - r;
                    e.copy_region(
                        &sc.d_pkn,
                        (rc - wrows) * hw,
                        &mut *sk.pool,
                        base + r * hw,
                        len1 * hw,
                    )?;
                    if len1 < wrows {
                        e.copy_region(
                            &sc.d_pkn,
                            (rc - wrows + len1) * hw,
                            &mut *sk.pool,
                            base,
                            (wrows - len1) * hw,
                        )?;
                    }
                }
            }
        }
        Phase::PrefillRuns => {
            // A mixed walk's decode rows: the decode tick's one-launch ring
            // step over rows [0, lead), each at its own slot and position.
            if lead > 0 {
                e.q4x_conv_dil_step_ring(
                    &sc.d_pkn,
                    win,
                    &ple.conv.buf,
                    &mut sc.d_pconv,
                    &sc.d_slots,
                    &sc.d_pos,
                    hw,
                    c.ple_conv,
                    PLE_DILATION,
                    lead,
                )?;
            }
            // Per run, the Prefill arm at a row offset: the dilated conv's own
            // left-pad guard is relative to the offset base, so a run never
            // reads the run before it - exactly a fresh sequence's zero pad.
            for r in runs {
                e.q4x_conv_dil_at(
                    &sc.d_pkn,
                    &ple.conv.buf,
                    &mut sc.d_pconv,
                    r.off,
                    r.len,
                    hw,
                    c.ple_conv,
                    PLE_DILATION,
                )?;
                let pbase = r.slot * wrows * hw;
                if r.row0 > 0 {
                    // a continuing run (see the Prefill arm): the ring's rows
                    // in order in front of the run's first rows, recomputed
                    let m = r.len.min(wrows);
                    let s0 = r.row0 % wrows;
                    let Scratch {
                        d_pkn,
                        d_pconv,
                        d_ple_ext_in,
                        d_ple_ext_out,
                        ..
                    } = sc;
                    e.copy_region(win, pbase + s0 * hw, d_ple_ext_in, 0, (wrows - s0) * hw)?;
                    if s0 > 0 {
                        e.copy_region(win, pbase, d_ple_ext_in, (wrows - s0) * hw, s0 * hw)?;
                    }
                    e.copy_region(d_pkn, r.off * hw, d_ple_ext_in, wrows * hw, m * hw)?;
                    e.q4x_conv_dil(
                        d_ple_ext_in,
                        &ple.conv.buf,
                        d_ple_ext_out,
                        wrows + m,
                        hw,
                        c.ple_conv,
                        PLE_DILATION,
                    )?;
                    e.copy_region(d_ple_ext_out, wrows * hw, d_pconv, r.off * hw, m * hw)?;
                }
                // the run's last min(len, wrows) rows at their ring index
                // (pos % wrows) - for a fresh run this is the old commit exactly
                let m = r.len.min(wrows);
                let q0 = r.row0 + r.len - m;
                let ri = q0 % wrows;
                let len1 = m.min(wrows - ri);
                e.copy_region(
                    &sc.d_pkn,
                    (r.off + r.len - m) * hw,
                    win,
                    pbase + ri * hw,
                    len1 * hw,
                )?;
                if len1 < m {
                    e.copy_region(
                        &sc.d_pkn,
                        (r.off + r.len - m + len1) * hw,
                        win,
                        pbase,
                        (m - len1) * hw,
                    )?;
                }
            }
        }
        Phase::Decode | Phase::DecodeBatch => {
            // One launch: the conv step reads the ring by position and stores
            // this token's pre-conv row over the one it just evicted. The
            // shifted form this replaces cost 1 + 3*rows launches per tick
            // (96 dependent copies at c32, 10.5 MB through a shared scratch
            // row) and computed its offsets on the host from the slot set,
            // which is what pinned a captured decode graph to the slot set it
            // was taken against.
            e.q4x_conv_dil_step_ring(
                &sc.d_pkn,
                win,
                &ple.conv.buf,
                &mut sc.d_pconv,
                &sc.d_slots,
                &sc.d_pos,
                hw,
                c.ple_conv,
                PLE_DILATION,
                n,
            )?;
        }
    }
    e.add(&mut sc.d_h, &sc.d_pgv, n * hw)?;
    e.add(&mut sc.d_h, &sc.d_pconv, n * hw)?;
    Ok(())
}

/// The n-gram ids are a pure function of the token stream, so they are computed
/// host-side and the 16 x 160 fp8 rows are gathered from the still-mapped
/// shards and widened. Returns `[n, ple_embed]` f32.
#[allow(clippy::too_many_arguments)]
/// The n-gram row ids for `n` consecutive positions of one request's token
/// stream, starting at stream index `first` - `[n, ple_heads]`, GLOBAL ids
/// (each head's table offset already folded in).
///
/// Pure integer arithmetic on the token ids: it touches no table memory, which
/// is exactly why the hash stays on the host while the GATHER moves to the
/// device. `rq::ple_window`'s previous-EOS scan is O(i) per position, so the
/// running cursor here is what keeps a long prefill linear.
fn ple_row_ids(
    c: &Qwen4ExpConfig,
    ple: &PleW,
    stream: &[i64],
    first: usize,
    n: usize,
) -> Result<Vec<u32>, GpuModelError> {
    let hpn = c.heads_per_ngram;
    let heads = c.ple_heads();
    let eos = c.bos_id as i64;
    if first + n > stream.len() {
        return Err(GpuModelError::Unsupported(format!(
            "ple ids: {n} rows at {first} but the stream holds {}",
            stream.len()
        )));
    }
    let mut prev_eos: i64 = -1;
    for (j, &t) in stream[..first].iter().enumerate() {
        if t == eos {
            prev_eos = j as i64;
        }
    }
    let mut out = vec![0u32; n * heads];
    for tk in 0..n {
        let i = first + tk;
        let pos_in_seg = i as i64 - prev_eos - 1;
        // rq::ple_window: a token within `shift` of its segment start reads
        // EOS instead of the real previous token
        let mut w = [stream[i], eos, eos];
        for (shift, slot) in [(1usize, 1usize), (2, 2)] {
            if i >= shift && pos_in_seg >= shift as i64 {
                w[slot] = stream[i - shift];
            }
        }
        for ngram in 2..=c.ngram_size {
            let mut mixed = w[0].wrapping_mul(ple.multipliers[0]);
            for (wk, m) in w.iter().zip(&ple.multipliers).take(ngram).skip(1) {
                mixed ^= wk.wrapping_mul(*m);
            }
            let start = (ngram - 2) * hpn;
            for hh in 0..hpn {
                let rid =
                    mixed.rem_euclid(ple.head_vocab[start + hh]) + ple.head_offset[start + hh];
                // a bad id would read anywhere in a 51.2 GB buffer, so it is
                // checked rather than trusted
                if rid < 0 || rid as usize >= ple.table_rows.max(1) {
                    return Err(GpuModelError::Unsupported(format!(
                        "ple row id {rid} outside the {}-row table",
                        ple.table_rows
                    )));
                }
                out[tk * heads + start + hh] = rid as u32;
            }
        }
        if stream[i] == eos {
            prev_eos = i as i64;
        }
    }
    Ok(out)
}

/// Stage `n` PLE rows into `sc.d_emb` off the device table (slot 532).
fn stage_ple_device(
    exec: &Arc<GpuExecutor>,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    table: &CudaSlice<u8>,
    ids: &[u32],
    sc: &mut Scratch,
) -> Result<(), GpuModelError> {
    let heads = c.ple_heads();
    let width = c.ple_embed / heads;
    exec.upload_u32(ids, &mut sc.d_ple_ids)?;
    exec.q4x_ple_gather(
        table,
        &sc.d_ple_ids,
        &mut sc.d_emb,
        ple.table_scale,
        ids.len() / heads,
        heads,
        width,
    )?;
    Ok(())
}

/// Whether to make the 51.2 GB n-gram table device-resident. Refuses when
/// the card cannot hold it on top of everything already loaded, so a smaller
/// board still runs (slowly) rather than failing to load; `PADDOCK_Q4X_PLE_HOST`
/// forces the host lane for A/Bs.
/// Fault the host-lane PLE table into the page cache at LOAD, not during the
/// first prompts.
///
/// The gather reads 16 rows a token out of a table that is 26.8 GiB on an
/// MX-quantized export, so a cold mapping pays those as disk seeks on the
/// critical path - a TTFT problem, not a throughput one, and the repo has met
/// it before (`ple-table-page-cache-trap`, which the GGUF lane's baselines
/// note handles by warming the shards by hand before every lane). Measured on
/// Mia's export before this, 3 reps of one serve: 1486.8 -> 1312.5 -> 1168.1
/// ms p50 latency, i.e. still warming on the third rep, with aiperf's
/// end-to-end throughput climbing 24.35 -> 27.26 -> 28.24 underneath it.
///
/// Costs a one-off sequential read at load, which is the cheap way to buy it:
/// the load is already disk-bound and the table is contiguous per shard. On
/// an integrated die this is the whole table's residency plan - the mapping
/// IS device memory there, so there is no second copy to make.
fn warm_ple_table(st: &ShardedSafetensors, c: &Qwen4ExpConfig, li: usize) {
    let emb = format!("model.language_model.layers.{li}.ple.ple_embedding");
    let t0 = std::time::Instant::now();
    let mut bytes = 0usize;
    for sh in 0..c.ngram_split {
        for suffix in ["weight", "weight_scale"] {
            let name = format!("{emb}.ngram_embedding.shard_{sh}.{suffix}");
            // a shard that has no scale plane is the FP8 table, not an error
            if let Ok(n) = st.warm_tensor(&name) {
                bytes += n;
                // what the cache cannot keep faults back one row at a time:
                // its page only, never the readahead window around it
                let _ = st.advise_tensor(&name, MapAccess::Random, &[(0, n)]);
            }
        }
    }
    if bytes > 0 {
        let s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[q4x-ple] warmed {:.1} GiB of n-gram table in {s:.1}s ({:.0} MB/s)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            bytes as f64 / 1e6 / s.max(1e-9),
        );
    }
}

fn ple_device_table(exec: &Arc<GpuExecutor>, c: &Qwen4ExpConfig) -> bool {
    if std::env::var("PADDOCK_Q4X_PLE_HOST").is_ok_and(|v| v != "0") {
        eprintln!("[q4x-ple] HOST lane (PADDOCK_Q4X_PLE_HOST)");
        return false;
    }
    if !exec.has_q4x_ple_gather() {
        eprintln!("[q4x-ple] HOST lane: pack has no q4x_ple_gather (slot 532)");
        return false;
    }
    // UNIFIED-MEMORY DIE: there is nothing to move. The table is already in
    // DRAM as a file mapping, and on GB10/Jetson a "device" allocation is the
    // same physical memory - so copying it in does not shorten a single read,
    // it just holds 51.2 GB twice. The device table is a DISCRETE-card
    // optimization: there the copy turns a PCIe round trip per gather into a
    // local read, which is what bought the 891-48697 ms prefill ticks back.
    //
    // Taking it here is how NVIDIA's official NVFP4 checkpoint killed the box
    // (2026-09-19): 79 GB of weights plus a 51.2 GB second copy of a table
    // that was already resident, on a 121 GiB board. The GGUF lane has always
    // host-mapped this table on this hardware and serves 29-34 tok/s doing it.
    if exec.is_integrated() {
        eprintln!(
            "[q4x-ple] HOST lane: unified-memory die - the mapping IS device memory, \
             a device copy would hold the table twice"
        );
        return false;
    }
    let want = (c.ngram_vocab_base as usize) * c.ple_heads() * (c.ple_embed / c.ple_heads());
    let Ok((free, _)) = cudarc::driver::result::mem_get_info() else {
        return true; // no honest number - let the allocation decide
    };
    // 4 GiB of slack: the table is the last big claim and the scratch planes
    // below it still have to fit
    const SLACK: usize = 4 << 30;
    if want + SLACK > free {
        eprintln!(
            "[q4x-ple] HOST lane: table needs {:.1} GiB, {:.1} GiB free",
            want as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
        );
        return false;
    }
    true
}

/// One row out of a `[rows, per_row]` plane. The n-gram table is read a row
/// at a time (16 of ~10M per token), so this never materializes a shard.
fn scb_slice(plane: &[u8], row: usize, per_row: usize) -> &[u8] {
    &plane[row * per_row..(row + 1) * per_row]
}

fn gather_ple_rows(
    src: &PleSource,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    li: usize,
    stream: &[i64],
    first: usize,
    n: usize,
) -> Result<Vec<f32>, GpuModelError> {
    let st = match src {
        PleSource::St(st) => st,
        PleSource::Gguf {
            map,
            name,
            ty,
            row_bytes,
        } => return gather_ple_rows_gguf(map, name, *ty, *row_bytes, c, ple, stream, first, n),
    };
    let width = c.ple_embed / c.ple_heads();
    let emb_p = format!("model.language_model.layers.{li}.ple.ple_embedding");
    // take the shard row split from shard 0's own shape, so a re-sharded
    // checkpoint cannot silently read the wrong row
    let rows_per_shard = {
        let name = format!("{emb_p}.ngram_embedding.shard_0.weight");
        let (t, _) = st
            .bytes(&name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        t.shape[0]
    };
    // the caller carries the 2-token EOS priming on the front of `stream`
    // (vLLM's `ngram_context`), so a decode step hashes the same window a
    // prefill of the whole sequence would have.
    let eos = c.bos_id as i64;
    let ids: Vec<Vec<i64>> = (0..n)
        .map(|t| {
            let w3 = rq::ple_window(stream, first + t, eos);
            rq::ple_ngram_ids(
                &w3,
                &ple.multipliers,
                &ple.head_vocab,
                &ple.head_offset,
                c.heads_per_ngram,
            )
        })
        .collect();
    // every row (and group-scale row) hinted before any is read - the page-in
    // batching `gather_ple_rows_gguf` explains; a shard plane per hint call
    {
        let mut by_plane: std::collections::BTreeMap<String, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for &rid in ids.iter().flatten() {
            let rid = rid as usize;
            let (sh, local) = (rid / rows_per_shard, rid % rows_per_shard);
            let name = format!("{emb_p}.ngram_embedding.shard_{sh}.weight");
            let packed = st.bytes(&name).is_some_and(|(t, _)| t.dtype == StDtype::U8);
            let rb = if packed { width / 2 } else { width };
            by_plane
                .entry(name.clone())
                .or_default()
                .push((local * rb, rb));
            if packed {
                by_plane
                    .entry(format!("{name}_scale"))
                    .or_default()
                    .push((local * (width / 16), width / 16));
            }
        }
        for (name, ranges) in &by_plane {
            let _ = st.advise_tensor(name, MapAccess::WillNeed, ranges);
        }
    }
    let mut out = vec![0f32; n * c.ple_embed];
    for (t, row_ids) in ids.iter().enumerate() {
        for (hh, &rid) in row_ids.iter().enumerate() {
            let rid = rid as usize;
            let (sh, local) = (rid / rows_per_shard, rid % rows_per_shard);
            let name = format!("{emb_p}.ngram_embedding.shard_{sh}.weight");
            let (tinfo, sb) = st
                .bytes(&name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
            let dst = t * c.ple_embed + hh * width;
            match tinfo.dtype {
                // FP8 table: e4m3 bytes, one scalar for the whole tensor
                StDtype::F8E4m3 => {
                    let row = &sb[local * width..(local + 1) * width];
                    for (i, &byte) in row.iter().enumerate() {
                        out[dst + i] = rq::e4m3_to_f32(byte) * ple.table_scale;
                    }
                }
                // NVFP4 table: e2m1 nibbles with per-16 e4m3 group scales in a
                // companion plane, times the tensor's global f32. Decoded
                // through `Nvfp4View`, which is the same reference decode the
                // expert seats validate against - low nibble is the even
                // element, and the group scale is a second level, not a
                // replacement for the global one.
                StDtype::U8 => {
                    let sname = format!("{name}_scale");
                    let (st_i, scb) = st.bytes(&sname).ok_or_else(|| {
                        GpuModelError::Unsupported(format!("{sname}: missing (NVFP4 table)"))
                    })?;
                    if st_i.dtype != StDtype::F8E4m3 {
                        return Err(GpuModelError::Unsupported(format!(
                            "{sname}: dtype {:?}, want F8E4m3 group scales",
                            st_i.dtype
                        )));
                    }
                    let view = paddock_models::modelopt::Nvfp4View {
                        packed: scb_slice(sb, local, width / 2),
                        scales: scb_slice(scb, local, width / 16),
                        scale2: ple.table_scale,
                        n: 1,
                        k: width,
                    };
                    let vals = view.dequant_row_f32(0);
                    out[dst..dst + width].copy_from_slice(&vals);
                }
                other => {
                    return Err(GpuModelError::Unsupported(format!(
                        "{name}: dtype {other:?}, want F8E4m3 (FP8 table) or U8 (NVFP4 table)"
                    )));
                }
            }
        }
    }
    Ok(out)
}

/// The GGUF twin of `gather_ple_rows`: the same hashed row ids, rows read
/// straight out of the mmapped table tensor and decoded per 32-wide block.
#[allow(clippy::too_many_arguments)]
fn gather_ple_rows_gguf(
    map: &MappedGguf,
    name: &str,
    ty: GgmlType,
    row_bytes: usize,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    stream: &[i64],
    first: usize,
    n: usize,
) -> Result<Vec<f32>, GpuModelError> {
    let (heads, width) = (c.ple_heads(), c.ple_embed / c.ple_heads());
    let (_, table) = map
        .tensor_bytes(name)
        .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}")))?;
    let rows = table.len() / row_bytes;
    // Every row the walk reads, hinted before any of them is read. The table
    // is tens of GB of hashed, effectively random rows (26.8 GiB IQ4_NL in
    // the Flash-Next GGUF), and on a unified-memory die the page cache holds
    // only part of it beside a resident model - GB10 kept 2-47% of it however
    // it was warmed. Read one at a time, each miss was a device round trip in
    // series with the next: a 4-row speculative verify gathers 64 rows, and
    // the GPU sat idle 4-30 ms (median 16) in every round while they came in
    // one by one. Hinted together (`MapAccess::WillNeed`), the misses go to
    // the device at once and the gather waits for the slowest instead of
    // their sum. Timing only - the bytes read are the same. (The ids are the
    // device lane's hash: one pass over the stream, not one per row.)
    let ids = ple_row_ids(c, ple, stream, first, n)?;
    if let Some(&rid) = ids.iter().find(|&&r| r as usize >= rows) {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: n-gram row {rid} past the table's {rows} rows"
        )));
    }
    let ranges: Vec<(usize, usize)> = ids
        .iter()
        .map(|&r| (r as usize * row_bytes, row_bytes))
        .collect();
    let _ = map.advise_tensor(name, MapAccess::WillNeed, &ranges);
    let mut out = vec![0f32; n * c.ple_embed];
    for (i, &rid) in ids.iter().enumerate() {
        let rid = rid as usize;
        let row = &table[rid * row_bytes..(rid + 1) * row_bytes];
        let dst = (i / heads) * c.ple_embed + (i % heads) * width;
        ple_row_dequant(ty, row, &mut out[dst..dst + width]);
    }
    Ok(out)
}

/// Gated DeltaNet mixer. Writes `d_mix` `[n, hidden]` and advances `state`.
#[allow(clippy::too_many_arguments)]
fn gdn_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::GdnW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    state: &mut CudaSlice<f32>,
    win: &mut CudaSlice<f32>,
    n: usize,
    phase: Phase,
    // slot this PREFILL belongs to; ignored in the decode arms, which carry a
    // slot per row in `sc.d_slots`
    slot: usize,
    // run table; non-empty only in `PrefillRuns`
    runs: &[Run],
    // a mixed walk's leading decode rows (see `walk_lead`); `runs` is then
    // just the prompt runs behind them
    lead: usize,
    fork_ok: bool,
    // the sequence position of the span's first row (`walk_row0`): a
    // continued sequence re-stages its conv window in front of the span
    row0: usize,
    // a decode-exact verify walk's one-row staging (see `verify_gdn_rows`)
    vrows: Option<&mut spec::VerifyRows>,
    // in-walk prefix-cache checkpoints (Prefill): (row in the span, pool
    // index), where their blobs sit, and this layer's place among the GDN
    // layers
    cuts: &[(usize, u32)],
    mut sink: Option<&mut super::prefix::CkptSink<'_>>,
    gdn_ord: usize,
) -> Result<(), GpuModelError> {
    let (h, hv, kd, vd) = (c.hidden, c.gdn_v_heads, c.gdn_k_dim, c.gdn_v_dim);
    let (qkv_rows, km1) = (c.gdn_qkv_rows(), c.gdn_conv - 1);
    // Two independent chains hang off `d_bi` and only meet at the recurrence:
    //   MAIN:  qkv -> conv -> split_widen -> (dq,dk,dv)
    //   SIDE:  z;  ab -> delta_gate_ab     -> (g,beta)
    // Both are under-occupied on their own (the a||b plane is out=96, i.e. 96
    // blocks on a 148-SM die, measured 65 GB/s), so running them concurrently
    // costs nothing and hides one behind the other. Under graph capture the
    // fork/join record+wait pair lowers to plain DAG edges, which is how the
    // rival's decode graph reaches 17 streams and 30% overlap against our 1
    // stream and 13%.
    // One-launch z|qkv (2-segment plane, the rival's in_proj_qkvz shape).
    // Runs on the MAIN stream before any fork so both branches see the
    // results; per-segment output is the fused export's documented
    // bit-identity contract with the separate launches. Only where the
    // separate calls would take the bf16 route (the f16/tc5 election above
    // this width must keep its own class).
    let bf16_route = !(stage.f16_ok && n >= super::f16_min_batch() && n <= stage.f16_max);
    let fused_zq = n >= 2
        && bf16_route
        && super::fuse_gdn_zq_on()
        && match &w.zqkv {
            Some(f) => e.bf16_gemm_2seg(
                f,
                &sc.d_bi,
                &mut sc.d_zg,
                &mut sc.d_qkv,
                c.gdn_z_rows(),
                c.gdn_qkv_rows(),
                n,
            )?,
            None => false,
        };
    let forked = fork_ok && super::gdn_fork_enabled() && e.side_fork().is_ok();
    if !forked {
        if !fused_zq {
            w.z.matmul(e, &sc.d_bi, &mut sc.d_zg, n, stage)?;
        }
        // This is the branch prefill actually takes (the fork is decode-side),
        // and the lap that follows covers the z matmul AND the ab matvec. Price
        // them apart: if z is skipped (fused_zq) this reads ~0 and the whole
        // lap is the ab pair.
        pm_lap(e, "gdn-z");
        // one plane, one launch: rows [0,h) are alpha and [h,2h) beta, which is
        // delta_gate_ab's own layout
        {
            // [in=2560, out=96] is 96 blocks on a 148-SM die under the
            // one-block-per-row matvec; split-K refills the wave.
            let sp = sk_split();
            let done = n == 1
                && sp >= 2
                && e.matvec_f32_sk(
                    &w.ab.buf,
                    h,
                    2 * hv,
                    &sc.d_bi,
                    &mut sc.d_ab,
                    &mut sc.d_skp,
                    &mut sc.d_skc,
                    sp,
                )?;
            if !done {
                e.matvec_f32_raw(&w.ab.buf, h, 2 * hv, &sc.d_bi, &mut sc.d_ab, n)?;
            }
        }
    } else {
        if !fused_zq {
            w.z.matmul(e, &sc.d_bi, &mut sc.d_zg, n, stage)?;
        }
        {
            // [in=2560, out=96] is 96 blocks on a 148-SM die under the
            // one-block-per-row matvec; split-K refills the wave.
            //
            // NOTE (2026-09-16): this is the FORKED branch, which prefill does
            // not take - `forked` is decode-side. A fused ab+gate election
            // (`matvec_ab_gate`, bit-identical) was tried here and measured
            // nothing because it never ran. Elect it on the branch above if it
            // is ever worth it; at 1024 rows the ab matvec is 0.094 ms a layer
            // (47% of the bus), so it is not.
            let sp = sk_split();
            let done = n == 1
                && sp >= 2
                && e.matvec_f32_sk(
                    &w.ab.buf,
                    h,
                    2 * hv,
                    &sc.d_bi,
                    &mut sc.d_ab,
                    &mut sc.d_skp,
                    &mut sc.d_skc,
                    sp,
                )?;
            if !done {
                e.matvec_f32_raw(&w.ab.buf, h, 2 * hv, &sc.d_bi, &mut sc.d_ab, n)?;
            }
        }
        // g = ssm_a * softplus(a + dt_bias), beta = sigmoid(b) - depends only
        // on d_ab, so it belongs on this branch, not after the join.
        e.delta_gate_ab(
            &sc.d_ab,
            &w.ssm_a.buf,
            &w.dt_bias.buf,
            &mut sc.d_g,
            &mut sc.d_beta,
            n,
            hv,
        )?;
        e.side_end()?;
    }
    // Without this the qkv matmul's lap swallowed the z + ab work too, and
    // "gdn-proj" read as one 1.109 ms kernel when it is three things. Measured
    // a layer at 1024 rows: z 0.408, ab 0.094, qkv 0.653.
    pm_lap(e, "gdn-ab");
    if !fused_zq {
        w.qkv.matmul(e, &sc.d_bi, &mut sc.d_qkv, n, stage)?;
    }
    let mut split_done = false;
    pm_lap(e, "gdn-proj");
    match phase {
        Phase::Prefill => {
            e.causal_conv1d_silu(
                &sc.d_qkv,
                &w.conv.buf,
                &mut sc.d_conv,
                n,
                qkv_rows,
                c.gdn_conv,
            )?;
            // window = the last k-1 PRE-conv rows, oldest first (conv_step's
            // contract); a short prompt lands at the tail over the reset zeros
            let wbase = slot * km1 * qkv_rows;
            if row0 > 0 {
                // a continued sequence: the fresh conv above zero-padded the
                // span's first k-1 rows; recompute them over [window ; rows]
                // (the whole-sequence conv bit for bit), then the window
                // SHIFTS - a span shorter than the window keeps its tail
                let m = n.min(km1);
                let Scratch {
                    d_qkv,
                    d_conv,
                    d_gdn_ext_in,
                    d_gdn_ext_out,
                    ..
                } = sc;
                e.copy_region(win, wbase, d_gdn_ext_in, 0, km1 * qkv_rows)?;
                e.copy_region(d_qkv, 0, d_gdn_ext_in, km1 * qkv_rows, m * qkv_rows)?;
                e.causal_conv1d_silu(
                    d_gdn_ext_in,
                    &w.conv.buf,
                    d_gdn_ext_out,
                    km1 + m,
                    qkv_rows,
                    c.gdn_conv,
                )?;
                e.copy_region(d_gdn_ext_out, km1 * qkv_rows, d_conv, 0, m * qkv_rows)?;
                if n >= km1 {
                    e.copy_region(d_qkv, (n - km1) * qkv_rows, win, wbase, km1 * qkv_rows)?;
                } else {
                    let keep = km1 - n;
                    e.copy_region(win, wbase + n * qkv_rows, d_gdn_ext_in, 0, keep * qkv_rows)?;
                    e.copy_region(d_qkv, 0, d_gdn_ext_in, keep * qkv_rows, n * qkv_rows)?;
                    e.copy_region(d_gdn_ext_in, 0, win, wbase, km1 * qkv_rows)?;
                }
            } else if n >= km1 {
                e.copy_region(&sc.d_qkv, (n - km1) * qkv_rows, win, wbase, km1 * qkv_rows)?;
            } else {
                e.copy_region(
                    &sc.d_qkv,
                    0,
                    win,
                    wbase + (km1 - n) * qkv_rows,
                    n * qkv_rows,
                )?;
            }
            // in-walk checkpoints: the window a walk ending at each cut row
            // would leave (the last k-1 pre-conv rows before it)
            if let Some(sk) = sink.as_mut() {
                for &(rc, idx) in cuts {
                    debug_assert!(rc >= km1 && rc <= n, "cut row {rc} of a {n}-row walk");
                    let off = sk.win_off(idx, gdn_ord);
                    e.copy_region(
                        &sc.d_qkv,
                        (rc - km1) * qkv_rows,
                        &mut *sk.pool,
                        off,
                        km1 * qkv_rows,
                    )?;
                }
            }
        }
        Phase::PrefillRuns => {
            // A mixed walk's decode rows: the decode tick's per-slot conv step
            // over rows [0, lead) - the UNFUSED entry, so the split+widen below
            // runs once over every row of the walk.
            if lead > 0 {
                let Scratch {
                    d_qkv,
                    d_conv,
                    d_slots,
                    ..
                } = sc;
                e.conv_step_slots(
                    win,
                    d_qkv,
                    &w.conv.buf,
                    d_conv,
                    d_slots,
                    lead,
                    qkv_rows,
                    c.gdn_conv,
                )?;
            }
            // Same as the Prefill arm at a row offset - `causal_conv1d_silu_at`
            // documents exactly this contract (rows before the offset base are
            // never read, which is the fresh prompt's zero left-pad).
            for r in runs {
                e.causal_conv1d_silu_at(
                    &sc.d_qkv,
                    &w.conv.buf,
                    &mut sc.d_conv,
                    r.off,
                    r.off,
                    r.len,
                    qkv_rows,
                    c.gdn_conv,
                )?;
                let wbase = r.slot * km1 * qkv_rows;
                if r.row0 > 0 {
                    // a continuing run (see the Prefill arm): recompute the
                    // run's first k-1 rows over [window ; rows], then shift
                    let m = r.len.min(km1);
                    let Scratch {
                        d_qkv,
                        d_conv,
                        d_gdn_ext_in,
                        d_gdn_ext_out,
                        ..
                    } = sc;
                    e.copy_region(win, wbase, d_gdn_ext_in, 0, km1 * qkv_rows)?;
                    e.copy_region(
                        d_qkv,
                        r.off * qkv_rows,
                        d_gdn_ext_in,
                        km1 * qkv_rows,
                        m * qkv_rows,
                    )?;
                    e.causal_conv1d_silu(
                        d_gdn_ext_in,
                        &w.conv.buf,
                        d_gdn_ext_out,
                        km1 + m,
                        qkv_rows,
                        c.gdn_conv,
                    )?;
                    e.copy_region(
                        d_gdn_ext_out,
                        km1 * qkv_rows,
                        d_conv,
                        r.off * qkv_rows,
                        m * qkv_rows,
                    )?;
                    if r.len >= km1 {
                        e.copy_region(
                            d_qkv,
                            (r.off + r.len - km1) * qkv_rows,
                            win,
                            wbase,
                            km1 * qkv_rows,
                        )?;
                    } else {
                        let keep = km1 - r.len;
                        e.copy_region(
                            win,
                            wbase + r.len * qkv_rows,
                            d_gdn_ext_in,
                            0,
                            keep * qkv_rows,
                        )?;
                        e.copy_region(
                            d_qkv,
                            r.off * qkv_rows,
                            d_gdn_ext_in,
                            keep * qkv_rows,
                            r.len * qkv_rows,
                        )?;
                        e.copy_region(d_gdn_ext_in, 0, win, wbase, km1 * qkv_rows)?;
                    }
                } else if r.len >= km1 {
                    e.copy_region(
                        &sc.d_qkv,
                        (r.off + r.len - km1) * qkv_rows,
                        win,
                        wbase,
                        km1 * qkv_rows,
                    )?;
                } else {
                    e.copy_region(
                        &sc.d_qkv,
                        r.off * qkv_rows,
                        win,
                        wbase + (km1 - r.len) * qkv_rows,
                        r.len * qkv_rows,
                    )?;
                }
            }
        }
        // conv_step shifts the window itself
        Phase::Decode => e.conv_step(
            win,
            &sc.d_qkv,
            &w.conv.buf,
            &mut sc.d_conv,
            qkv_rows,
            c.gdn_conv,
        )?,
        // same contract, one window per slot. slot 563 folds the q/k/v
        // split+widen below into this kernel's epilogue (the conv row's only
        // consumer), retiring a launch on every GDN layer's critical branch.
        Phase::DecodeBatch => {
            let Scratch {
                d_qkv,
                d_conv,
                d_dq,
                d_dk,
                d_dv,
                d_slots,
                ..
            } = sc;
            // the fused conv+split widens with the interleave map; the tiled
            // lane takes the unfused conv and its own split below
            if w.tiled_heads
                || !e.conv_step_slots_split(
                    win,
                    d_qkv,
                    &w.conv.buf,
                    d_dq,
                    d_dk,
                    d_dv,
                    d_slots,
                    n,
                    qkv_rows,
                    c.gdn_conv,
                    (c.gdn_k_heads, hv, kd, vd),
                )?
            {
                e.conv_step_slots(
                    win,
                    d_qkv,
                    &w.conv.buf,
                    d_conv,
                    d_slots,
                    n,
                    qkv_rows,
                    c.gdn_conv,
                )?;
            } else {
                split_done = true;
            }
        }
    }
    if !split_done {
        if w.tiled_heads {
            if !e.has_q4x_gdn_split_widen_tiled() {
                return Err(GpuModelError::Unsupported(
                    "kernel pack has no q4x_gdn_split_widen_tiled (slot 540) - rebuild packs/cuda"
                        .into(),
                ));
            }
            e.q4x_gdn_split_widen_tiled(
                &sc.d_conv,
                &mut sc.d_dq,
                &mut sc.d_dk,
                &mut sc.d_dv,
                n,
                c.gdn_k_heads,
                hv,
                kd,
                vd,
            )?;
        } else {
            e.q4x_gdn_split_widen(
                &sc.d_conv,
                &mut sc.d_dq,
                &mut sc.d_dk,
                &mut sc.d_dv,
                n,
                c.gdn_k_heads,
                hv,
                kd,
                vd,
            )?;
        }
    }
    pm_lap(e, "gdn-conv");
    // Join before the recurrence: it is the first consumer of both branches.
    if forked {
        e.side_join()?;
    } else {
        // g = ssm_a * softplus(a + dt_bias) with ssm_a = -exp(A_log) folded at
        // load; beta = sigmoid(b). The same expressions as reference::gdn_gates.
        e.delta_gate_ab(
            &sc.d_ab,
            &w.ssm_a.buf,
            &w.dt_bias.buf,
            &mut sc.d_g,
            &mut sc.d_beta,
            n,
            hv,
        )?;
    }
    let mut gn_done = false;
    if let (Phase::PrefillRuns, Some(vr)) = (phase, vrows) {
        verify_gdn_rows(e, c, w, sc, state, vr, runs)?;
        gn_done = true;
    } else if matches!(phase, Phase::PrefillRuns) {
        // A mixed walk's decode rows: one token per slot against its carried
        // state, the decode tick's recurrence entry over rows [0, lead). The
        // gated norm stays unfused (below) - it covers every row of the walk.
        if lead > 0 {
            let Scratch {
                d_dq,
                d_dk,
                d_dv,
                d_g,
                d_beta,
                d_slots,
                d_dattn,
                ..
            } = sc;
            e.gated_delta_recurrent_slots(
                d_dq, d_dk, d_dv, d_g, d_beta, d_slots, state, d_dattn, lead, hv, kd,
            )?;
        }
        // Every run's whole sequence in one launch, grid (n_heads, n_runs).
        // The single-run entry grids 48 blocks at 255 registers - 32% of a
        // 148-SM die - and a serially-prefilled wave pays 195.9 us per layer
        // per prompt for it (7.05 ms of a 33.9 ms 128-token prefill).
        let pn = !runs.is_empty()
            && super::dn_prenorm_on()
            && e.gated_delta_recurrent_runs_pn(
                &sc.d_dq,
                &sc.d_dk,
                &sc.d_dv,
                &sc.d_g,
                &sc.d_beta,
                state,
                &mut sc.d_dattn,
                &sc.d_run_off,
                &sc.d_run_len,
                &sc.d_run_slot,
                runs.len(),
                n,
                hv,
                kd,
                &mut sc.d_dnrn,
            )?;
        if !pn && !runs.is_empty() {
            e.gated_delta_recurrent_runs(
                &sc.d_dq,
                &sc.d_dk,
                &sc.d_dv,
                &sc.d_g,
                &sc.d_beta,
                state,
                &mut sc.d_dattn,
                &sc.d_run_off,
                &sc.d_run_len,
                &sc.d_run_slot,
                runs.len(),
                hv,
                kd,
            )?;
        }
    } else if matches!(phase, Phase::DecodeBatch) {
        // one token per SLOT, each against its own carried state. slot 564
        // folds the gated norm below into this kernel's epilogue: the norm's
        // row is a block's head output, so its reduction is block-local.
        let Scratch {
            d_dq,
            d_dk,
            d_dv,
            d_g,
            d_beta,
            d_slots,
            d_dattn,
            d_zg,
            d_core,
            ..
        } = sc;
        gn_done = e.gated_delta_recurrent_slots_gn(
            d_dq,
            d_dk,
            d_dv,
            d_g,
            d_beta,
            d_slots,
            state,
            d_core,
            d_zg,
            &w.norm.buf,
            None,
            c.eps,
            n,
            hv,
            kd,
        )?;
        if !gn_done {
            e.gated_delta_recurrent_slots(
                d_dq, d_dk, d_dv, d_g, d_beta, d_slots, state, d_dattn, n, hv, kd,
            )?;
        }
    } else {
        // prefill walks `n` tokens of one sequence sequentially; decode is the
        // n == 1 case of that. Same kernel either way, taken at this slot's
        // region of the [slots, heads, D, D] state - at slot 0 that offset is
        // zero, so the single-sequence lane's numerics do not move.
        // The PRE-NORMED, P-split walk (slot 596) where the pack has it: the
        // shipped walk holds a whole state column per thread (255 registers,
        // 2 blocks an SM, 12.57% occupancy on ncu) and runs two shared trees
        // a token. Prefill widths only - the decode tick is n == 1, where the
        // split has nothing to spread and the legacy order is kept.
        // SEGMENT-TILED walk (slot 604) for a 128-wide head: a thread carries
        // 32 state floats instead of slot 596's 8, a head is 4 blocks at 4 an
        // SM, and the gates and norms are one float4 a (token, head) from its
        // pre-pass (bench/gdn_prefill_gb10_bench.cu, 1024 tokens: 2.76 -> 1.11
        // ms a layer). Same class as slot 596 (the dots re-associate), and
        // still a sequential token loop, so the checkpoint spans stay exact.
        let seg = n > 1
            && kd == 128
            && vd == kd
            && super::gdn_seg_enabled()
            && e.has_gated_delta_recurrent_seg();
        if seg || (n > 1 && e.has_gated_delta_recurrent_pn() && super::gdn_pn_enabled()) {
            let st_off = slot * hv * kd * kd;
            let ck = sink
                .as_mut()
                .filter(|_| matches!(phase, Phase::Prefill) && !cuts.is_empty());
            match ck {
                None => gdn_walk_rows(e, sc, state, st_off, 0, n, (hv, kd, vd), seg)?,
                Some(sk) => {
                    // in-walk checkpoints: stop at each cut row, copy the slot's
                    // state into the checkpoint, go on - both walks are
                    // sequential token loops, so the rows land as one call's
                    let mut from = 0;
                    for &(rc, idx) in cuts {
                        gdn_walk_rows(e, sc, state, st_off, from, rc - from, (hv, kd, vd), seg)?;
                        let (dst, len) = (sk.state_off(idx, gdn_ord), sk.st_elems);
                        e.copy_region(&*state, st_off, &mut *sk.pool, dst, len)?;
                        from = rc;
                    }
                    if from < n {
                        gdn_walk_rows(e, sc, state, st_off, from, n - from, (hv, kd, vd), seg)?;
                    }
                }
            }
        } else {
            e.gated_delta_recurrent_at(
                &sc.d_dq,
                &sc.d_dk,
                &sc.d_dv,
                &sc.d_g,
                &sc.d_beta,
                state,
                slot * hv * kd * kd,
                &mut sc.d_dattn,
                n,
                hv,
                kd,
            )?;
        }
    }
    if !gn_done {
        pm_lap(e, "gdn-recur");
        e.q4x_gdn_gated_norm(
            &sc.d_dattn,
            &sc.d_zg,
            &w.norm.buf,
            &mut sc.d_core,
            None,
            n * hv,
            vd,
            c.eps,
        )?;
    }
    pm_lap(e, "gdn-norm");
    w.out.matmul(e, &sc.d_core, &mut sc.d_mix, n, stage)?;
    pm_lap(e, "gdn-out");
    Ok(())
}

/// A verify walk's GDN rows through the decode tick's own recurrence entry,
/// one row per call in run order, so each row's state update and gated norm
/// land exactly where the tick that decodes that token lands. The runs walk
/// plus the separate norm agree with it only to the last ulp (12 of 6144
/// elements at 1.5e-8 on layer 0, exactness probe 2026-09-14), and a near-tie
/// does not survive that. The conv, the split and the gates before it are
/// already row-invariant. Interim form: 8 launches a row a layer; the SOTA
/// shape is a runs entry carrying the slots kernel's per-token body verbatim,
/// one launch for every row.
#[allow(clippy::too_many_arguments)]
fn verify_gdn_rows(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::GdnW,
    sc: &mut Scratch,
    state: &mut CudaSlice<f32>,
    vr: &mut spec::VerifyRows,
    runs: &[Run],
) -> Result<(), GpuModelError> {
    let (hv, kd, vd) = (c.gdn_v_heads, c.gdn_k_dim, c.gdn_v_dim);
    // One launch for every run's rows where the pack carries slot 599 - the
    // DecodeBatch arm's slots_gn body token for token over the walk's own
    // staged planes and run tables. It declines exactly where slots_gn does,
    // and then the per-row calls below take the tick's fallback too.
    if e.gated_delta_recurrent_runs_slots(
        &sc.d_dq,
        &sc.d_dk,
        &sc.d_dv,
        &sc.d_g,
        &sc.d_beta,
        state,
        &mut sc.d_core,
        &sc.d_run_off,
        &sc.d_run_len,
        &sc.d_run_slot,
        Some((&sc.d_zg, &w.norm.buf, c.eps)),
        runs.len(),
        hv,
        kd,
    )? {
        return Ok(());
    }
    let (kdim, vdim, zr) = (hv * kd, hv * vd, c.gdn_z_rows());
    for r in runs {
        e.upload_u32(&[r.slot as u32], &mut vr.slot)?;
        for t in 0..r.len {
            let row = r.off + t;
            e.copy_region(&sc.d_dq, row * kdim, &mut vr.q, 0, kdim)?;
            e.copy_region(&sc.d_dk, row * kdim, &mut vr.k, 0, kdim)?;
            e.copy_region(&sc.d_dv, row * vdim, &mut vr.v, 0, vdim)?;
            e.copy_region(&sc.d_g, row * hv, &mut vr.g, 0, hv)?;
            e.copy_region(&sc.d_beta, row * hv, &mut vr.b, 0, hv)?;
            e.copy_region(&sc.d_zg, row * zr, &mut vr.z, 0, zr)?;
            // the DecodeBatch arm's calls at n = 1, fallback included
            let done = e.gated_delta_recurrent_slots_gn(
                &vr.q,
                &vr.k,
                &vr.v,
                &vr.g,
                &vr.b,
                &vr.slot,
                state,
                &mut vr.core,
                &vr.z,
                &w.norm.buf,
                None,
                c.eps,
                1,
                hv,
                kd,
            )?;
            if !done {
                e.gated_delta_recurrent_slots(
                    &vr.q,
                    &vr.k,
                    &vr.v,
                    &vr.g,
                    &vr.b,
                    &vr.slot,
                    state,
                    &mut vr.attn,
                    1,
                    hv,
                    kd,
                )?;
                e.q4x_gdn_gated_norm(
                    &vr.attn,
                    &vr.z,
                    &w.norm.buf,
                    &mut vr.core,
                    None,
                    hv,
                    vd,
                    c.eps,
                )?;
            }
            e.copy_region(&vr.core, 0, &mut sc.d_core, row * vdim, vdim)?;
        }
    }
    Ok(())
}

/// QSA attention for this walk's `n` rows, into `d_attn` (where the dense
/// kernels write): every row's blocks scored against the layer's compressed
/// keys and the top k selected, in batches of `qsa_rb` rows (the score plane
/// stays bounded at any context), then each row attends to its selected
/// blocks' tokens plus its tail. Rows that see no more than the window select
/// every block, so a walk mixing short and long rows is exact for both.
/// Decode-width walks split each row's tokens across CTAs to fill the die;
/// a partial's merge is the combine, which for one split just normalizes.
#[allow(clippy::too_many_arguments)]
fn qsa_attend(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    sc: &mut Scratch,
    cache: &CudaSlice<half::bf16>,
    kc: &CudaSlice<u8>,
    vc: &CudaSlice<u8>,
    max_ctx: usize,
    tab: &CudaSlice<u32>,
    bps: usize,
    n: usize,
    scale: f32,
) -> Result<(), GpuModelError> {
    let (ih, ihd) = (c.idx_heads, c.idx_head_dim);
    let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
    let cap = max_ctx.div_ceil(QSA_BLOCK);
    let k = c.idx_budget / c.idx_compress;
    let (score_route, attn_route) = qsa_routes(e, c);
    let mut row0 = 0;
    while row0 < n {
        let rows = sc.qsa_rb.min(n - row0);
        e.q4x_qsa_logits_paged(
            score_route,
            &sc.d_idx_q,
            cache,
            &sc.d_pos,
            &sc.d_slots,
            tab,
            bps,
            &mut sc.d_qsa_scores,
            row0,
            rows,
            ih,
            ihd,
            cap,
            QSA_BLOCK,
            k,
        )?;
        e.q4x_qsa_topk(
            &sc.d_qsa_scores,
            &sc.d_pos,
            &mut sc.d_qsa_sel,
            &mut sc.d_qsa_cnt,
            row0,
            rows,
            cap,
            QSA_BLOCK,
            k,
        )?;
        row0 += rows;
    }
    // enough CTAs to fill the die at decode width; one split once the rows
    // themselves do (the partials plane holds rows x splits <= its rows)
    let ctas = n * nkv;
    let fill = 2 * e.sm_count().max(1);
    let splits = if ctas >= fill {
        1
    } else {
        fill.div_ceil(ctas)
            .clamp(1, 16)
            .min(sc.d_qsa_cnt.len() / n.max(1))
    };
    e.q4x_qsa_attn_paged(
        attn_route,
        &sc.d_qn,
        kc,
        vc,
        &sc.d_pos,
        &sc.d_slots,
        tab,
        bps,
        &sc.d_qsa_sel,
        &sc.d_qsa_cnt,
        &mut sc.d_qsa_po,
        &mut sc.d_qsa_pml,
        n,
        nh,
        nkv,
        hd,
        k,
        QSA_BLOCK,
        splits,
        scale,
        KV(),
    )?;
    e.q4x_qsa_combine(
        &sc.d_qsa_po,
        &sc.d_qsa_pml,
        &mut sc.d_attn,
        n,
        nh,
        nkv,
        hd,
        splits,
    )?;
    Ok(())
}

/// The QSA indexer for this walk's `n` rows (the Flash-Next QSA design
/// note, attn/qsa.cuh): project q|k off the layer input the main
/// projections read, normalize and rotate the 4 query heads (the scores'
/// left side - rung R2 reads `d_idx_q`), and pool every block a row closes
/// into the compressed cache, the raw keys filed in the slot's ring for the
/// blocks later walks close.
#[allow(clippy::too_many_arguments)]
fn qsa_index(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::AttnW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    cache: &mut CudaSlice<half::bf16>,
    ring: &mut CudaSlice<f32>,
    tab: &CudaSlice<u32>,
    bps: usize,
    n: usize,
) -> Result<(), GpuModelError> {
    let (ih, ihd) = (c.idx_heads, c.idx_head_dim);
    let ld = (ih + c.idx_kv_heads) * ihd;
    let koff = ih * ihd;
    let yarn = yarn_params(c);
    w.idx_qk.matmul(e, &sc.d_bi, &mut sc.d_idx_qk, n, stage)?;
    e.q4x_idx_q(
        &sc.d_idx_qk,
        &w.idx_q_norm.buf,
        &mut sc.d_idx_q,
        n,
        ih,
        ihd,
        ld,
        c.eps,
    )?;
    e.mrope(
        &mut sc.d_idx_q,
        &sc.d_mrope,
        n,
        ih,
        ihd,
        c.rotary_dim,
        yarn,
        MROPE_SECTIONS,
    )?;
    e.q4x_idx_pool(
        &sc.d_idx_qk,
        ring,
        &sc.d_pos,
        &sc.d_slots,
        &w.idx_k_norm.buf,
        &mut sc.d_idx_stage,
        &mut sc.d_idx_spos,
        n,
        ihd,
        ld,
        koff,
        QSA_RING,
        QSA_BLOCK,
        c.eps,
    )?;
    e.mrope(
        &mut sc.d_idx_stage,
        &sc.d_idx_spos,
        n,
        1,
        ihd,
        c.rotary_dim,
        yarn,
        MROPE_SECTIONS,
    )?;
    e.q4x_idx_store_paged(
        &sc.d_idx_qk,
        &sc.d_idx_stage,
        &sc.d_pos,
        &sc.d_slots,
        cache,
        ring,
        tab,
        bps,
        n,
        ihd,
        ld,
        koff,
        QSA_RING,
        QSA_BLOCK,
    )?;
    Ok(())
}

/// Gated full attention, dense path. The QSA sparse walk is a later rung and
/// is exact anyway while the visible window stays inside the indexer budget.
#[allow(clippy::too_many_arguments)]
fn attn_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::AttnW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    kc: &mut CudaSlice<u8>,
    vc: &mut CudaSlice<u8>,
    // the layer's QSA indexer state (compressed keys, raw ring); None where
    // the lane keeps none (an old pack, the MTP head's own layer)
    idx: Option<(&mut CudaSlice<half::bf16>, &mut CudaSlice<f32>)>,
    // this walk attends through QSA (needs `idx`); else the dense kernels
    qsa: bool,
    max_ctx: usize,
    // the block table `kc` / `vc` / the index are read through, and its
    // stride: the pool's (the trunk) or an identity table (the MTP head)
    tab: &CudaSlice<u32>,
    bps: usize,
    n: usize,
    phase: Phase,
    runs: &[Run],
    // a mixed walk's leading decode rows (see `walk_lead`); `runs` is then
    // just the prompt runs behind them
    lead: usize,
    fork_ok: bool,
) -> Result<(), GpuModelError> {
    let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
    let (kv_dim, q_dim) = (nkv * hd, nh * hd);
    let yarn = yarn_params(c);
    // q and the k/v pair are independent from `d_bi` down to the attention
    // kernel: same fork shape as the GDN and MoE blocks above.
    // One-launch q|k|v (slot 424, the rival's qkv_proj shape): q rows first,
    // then k, then v with equal widths -- exactly the export's row-routing
    // contract (r < oq -> Y, < oq+okv -> Yk, else Yv). Main stream, before
    // the fork; per-segment bit-identity with the separate launches is the
    // export's documented contract. bf16 route only.
    let bf16_route = !(stage.f16_ok && n >= super::f16_min_batch() && n <= stage.f16_max);
    let fused_qkv = n >= 2
        && bf16_route
        && super::fuse_attn_qkv_on()
        && e.has_bf16_qkv_gemm()
        && match &w.qkv_f {
            Some(f) => {
                e.bf16_qkv_gemm(
                    f,
                    &sc.d_bi,
                    &mut sc.d_qg,
                    &mut sc.d_k,
                    &mut sc.d_v,
                    c.attn_q_rows(),
                    kv_dim,
                    n,
                )?;
                true
            }
            None => false,
        };
    let attn_forked = fork_ok && super::gdn_fork_enabled() && e.side_fork().is_ok();
    if attn_forked {
        if !fused_qkv {
            w.k.matmul(e, &sc.d_bi, &mut sc.d_k, n, stage)?;
            w.v.matmul(e, &sc.d_bi, &mut sc.d_v, n, stage)?;
        }
        // k_norm carries the +1 already (Gemma (1+w), folded at load)
        e.rmsnorm_batch(&sc.d_k, &w.k_norm.buf, &mut sc.d_kn, hd, c.eps, n * nkv)?;
        e.mrope(
            &mut sc.d_kn,
            &sc.d_mrope,
            n,
            nkv,
            hd,
            c.rotary_dim,
            yarn,
            MROPE_SECTIONS,
        )?;
        e.kv_append_batch_paged(
            &sc.d_kn,
            kc,
            &sc.d_pos,
            Some(&sc.d_slots),
            tab,
            bps,
            kv_dim,
            n,
            KV(),
        )?;
        e.kv_append_batch_paged(
            &sc.d_v,
            vc,
            &sc.d_pos,
            Some(&sc.d_slots),
            tab,
            bps,
            kv_dim,
            n,
            KV(),
        )?;
        e.side_end()?;
        if !fused_qkv {
            w.q.matmul(e, &sc.d_bi, &mut sc.d_qg, n, stage)?;
        }
        e.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_agate, n, nh, hd)?;
        e.rmsnorm_batch(&sc.d_q, &w.q_norm.buf, &mut sc.d_qn, hd, c.eps, n * nh)?;
        e.mrope(
            &mut sc.d_qn,
            &sc.d_mrope,
            n,
            nh,
            hd,
            c.rotary_dim,
            yarn,
            MROPE_SECTIONS,
        )?;
        e.side_join()?;
    } else {
        // Close the window before the q matmul: attn-qmm opened at whatever
        // lap last closed before this function, so it billed the layer's
        // pre-attention work too. With this, attn-qmm is the q GEMM and the
        // activation quantize `kq_matmul` runs inside it - and nothing else -
        // which is what the bench's 353 us for this plane compares against.
        pm_lap(e, "attn-pre");
        if !fused_qkv {
            w.q.matmul(e, &sc.d_bi, &mut sc.d_qg, n, stage)?;
        }
        // The GEMM alone, then the split: split_qg is a full round trip of the
        // 25 MB q+gate plane (~0.23 ms at this die's copy rate) that exists
        // only to separate q from its gate - matmul_2seg could have the GEMM
        // write both planes and skip it, but only if the split is the cost.
        pm_lap(e, "attn-qmm");
        e.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_agate, n, nh, hd)?;
        pm_lap(e, "attn-q");
        if !fused_qkv {
            w.k.matmul(e, &sc.d_bi, &mut sc.d_k, n, stage)?;
            w.v.matmul(e, &sc.d_bi, &mut sc.d_v, n, stage)?;
        }
        // Split the projections from the passes that follow them: attn-qkv is
        // 1.62 ms a layer and holds three narrow GEMMs plus six full-plane
        // passes (split, 2 norms, 2 mrope, 2 append), and which half carries
        // the cost decides whether the lever is a fused-QKV plane (concat_q8 +
        // one wide GEMM) or a fused post-pass (the slot-309 shape).
        pm_lap(e, "attn-proj");
        // q_norm/k_norm carry the +1 already (Gemma (1+w), folded at load)
        e.rmsnorm_batch(&sc.d_q, &w.q_norm.buf, &mut sc.d_qn, hd, c.eps, n * nh)?;
        e.rmsnorm_batch(&sc.d_k, &w.k_norm.buf, &mut sc.d_kn, hd, c.eps, n * nkv)?;
        e.mrope(
            &mut sc.d_qn,
            &sc.d_mrope,
            n,
            nh,
            hd,
            c.rotary_dim,
            yarn,
            MROPE_SECTIONS,
        )?;
        e.mrope(
            &mut sc.d_kn,
            &sc.d_mrope,
            n,
            nkv,
            hd,
            c.rotary_dim,
            yarn,
            MROPE_SECTIONS,
        )?;
        e.kv_append_batch_paged(
            &sc.d_kn,
            kc,
            &sc.d_pos,
            Some(&sc.d_slots),
            tab,
            bps,
            kv_dim,
            n,
            KV(),
        )?;
        e.kv_append_batch_paged(
            &sc.d_v,
            vc,
            &sc.d_pos,
            Some(&sc.d_slots),
            tab,
            bps,
            kv_dim,
            n,
            KV(),
        )?;
    }
    pm_lap(e, "attn-qkv");
    let scale = 1.0 / (hd as f32).sqrt();
    let sparse = match idx {
        Some((cache, ring)) => {
            qsa_index(e, c, w, sc, stage, cache, ring, tab, bps, n)?;
            pm_lap(e, "attn-qsa-index");
            if qsa {
                qsa_attend(e, c, sc, cache, kc, vc, max_ctx, tab, bps, n, scale)?;
                pm_lap(e, "attn-qsa");
            }
            qsa
        }
        None => false,
    };
    if sparse {
        // the QSA output is in d_attn, where the dense kernels would put it
    } else if matches!(phase, Phase::PrefillRuns) && stage.row_exact {
        // A decode-exact verify walk: every row is a decode tick's row - one
        // query at its own position against its slot's keys, its K/V
        // appended above - so the decode attention's own dispatch covers
        // them, and each row's output is the one the tick computes (the
        // dense planes are row-exact the same way). Rows of one slot repeat
        // its slot; the decode kernels take every row independently.
        attn_core(
            e,
            c,
            sc,
            kc,
            vc,
            max_ctx,
            tab,
            bps,
            n,
            Phase::DecodeBatch,
            &[],
            scale,
        )?;
    } else if matches!(phase, Phase::PrefillRuns) && lead > 0 {
        // A mixed walk: the decode rows lead, one per slot at its own
        // position, so the decode attention's dispatch (the class a decode
        // tick runs, split-KV where elected) covers rows [0, lead) exactly -
        // every row's K/V was appended above, before any read. The prompt
        // runs behind them take the tiled prefill kernel through a tile table
        // staged over those runs alone.
        attn_core(
            e,
            c,
            sc,
            kc,
            vc,
            max_ctx,
            tab,
            bps,
            lead,
            Phase::DecodeBatch,
            &[],
            scale,
        )?;
        if !runs.is_empty() {
            attn_core(e, c, sc, kc, vc, max_ctx, tab, bps, n, phase, runs, scale)?;
        }
    } else {
        attn_core(e, c, sc, kc, vc, max_ctx, tab, bps, n, phase, runs, scale)?;
    }
    pm_lap(e, "attn-core");
    e.mul_sigmoid(&mut sc.d_attn, &sc.d_agate, n * q_dim)?;
    // The gate multiply is a full round trip of d_attn (~38 MB with the gate
    // plane) and the o projection is a [q_dim -> hidden] Q8_0 GEMM (~31 MB).
    // attn-out bills 0.813 ms a layer against a ~0.32 ms floor for the pair,
    // so price them apart before deciding whether the lever is folding the
    // gate into the GEMM's prologue or the GEMM itself.
    pm_lap(e, "attn-gate");
    w.o.matmul(e, &sc.d_attn, &mut sc.d_mix, n, stage)?;
    pm_lap(e, "attn-out");
    Ok(())
}

/// The attention core of `attn_pass`: rows [0, n) against the carried KV,
/// dispatched by `phase` (decode kernels read one query row per slot, the
/// prefill tile walks the run table staged for `runs`). K/V for every row
/// is appended before this runs. Output rows land in `d_attn`. Every arm is
/// a paged kernel reading `kc` / `vc` through `tab` - the pool's table, or
/// the MTP head's identity table over its own dense planes - each the
/// bit-identical twin of the dense-strip kernel this lane took before its KV
/// was paged (gate: tests/gpu_paged_attn_modes.rs).
#[allow(clippy::too_many_arguments)]
fn attn_core(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    sc: &mut Scratch,
    kc: &CudaSlice<u8>,
    vc: &CudaSlice<u8>,
    max_ctx: usize,
    tab: &CudaSlice<u32>,
    bps: usize,
    n: usize,
    phase: Phase,
    runs: &[Run],
    scale: f32,
) -> Result<(), GpuModelError> {
    let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
    let kv_dim = nkv * hd;
    // tcgen05 decode attention (pack slot 431, the <256,6> instantiation built
    // for qwen3.8's 24q/4kv/hd256 - the same geometry). Needs e4m3 pools
    // (PADDOCK_Q38FN_KV8=1): its TMA maps assume 1-byte elements. It reads
    // the table only for the pages under each row's keys (the window below
    // bounds its tick table, not its reach). The effective window is the CONSTANT
    // max_ctx+16, never a live band - this walk is graph-captured, and a
    // window derived from live positions would bake into a replay (qwen35
    // precedent, including the +16 exact-multiple corner). Sinks here are
    // -1e30 (= the no-op fold the kernel's contract ignores), so dropping
    // them is exact. FINAL-output contract: rows land in d_attn, no combine.
    // A numerics CLASS change (tcgen05 MMA vs the SIMT f32 walk) - judged on
    // quality, kill switch PADDOCK_Q38FN_ATTN_TC5=0. Declines (rc -2/-3)
    // fall through to the arms below.
    // This model is 24q/2kv (config.json num_key_value_heads = 2, G=12 - Not
    // the qwen3.8-dense 4kv the campaign notes carry). G=12 cannot ride the
    // kernel's 8-row M tile, so each physical kv head presents as two virtual
    // G=6 heads: q rows, cells and the output already index by kvh*G+g, which
    // the virtual numbering makes exactly right, and the pack infers the
    // physical head for the KV pool offset from the kv_dim mismatch
    // (kvh_div = nkv_virt*hd/kv_dim). Doubles the cells too: batch*4 CTAs.
    let nkv_virt = if nkv > 0 && nh == nkv * 12 {
        nkv * 2
    } else {
        nkv
    };
    let tc5_done = matches!(phase, Phase::Decode | Phase::DecodeBatch)
        && super::attn_tc5_enabled()
        && KV() == KvDtype::Fp8E4m3
        && hd == 256
        && nkv_virt > 0
        && nh == nkv_virt * 6
        && max_ctx.is_multiple_of(16)
        && e.has_attn_decode_tc5_paged()
        && {
            let ok = e.attn_decode_tc5_paged(
                &sc.d_qn,
                kc,
                vc,
                &sc.d_sinks,
                &mut sc.d_attn,
                &sc.d_pos,
                Some(&sc.d_slots),
                tab,
                bps,
                nh,
                nkv_virt,
                hd,
                kv_dim,
                max_ctx + 16,
                n,
                scale,
                KV(),
            )?;
            if ok {
                super::witness_once("attn-tc5", n, nh, hd);
            }
            ok
        };
    if !tc5_done {
        match phase {
            // The single-slot entry reads `slots[0]` for every row (its own
            // documented contract: "slots uniform across rows, true for every
            // prefill path"), so a wave has to take the per-TILE entry or every
            // run silently attends to the first run's cache. Measured before the
            // fix: run 0 exact, runs 1 and 2 off by 0.66-0.79 logits with the
            // grouped MoE lane off, i.e. not a numeric-class artefact.
            // Every run in ONE tensor-core launch: the paged dispatcher's hd256
            // v4 arm at G=12 (pack >= 0.25; older bodies send G=12 to a tile
            // that ignores a run table) behind the walk's run table, grid.z
            // over runs, each CTA re-aimed at its run - K/V staged once per
            // (kv head, 4-token tile) for all 12 q heads, HMMA for S and PV, O
            // in f32. The single-slot walk takes the same arm, so a prompt
            // attends through the same arithmetic in a wave, a chunked span or
            // alone (bench/attn_prefill_runs_q4x_gb10_bench.cu: a 3 x 1300-row
            // wave 27.0 -> 2.5 ms a layer, 2 x 512 mixed spans 13.6 -> 1.2).
            // The registration is process state in the pack: armed for this
            // one call, disarmed whatever it returned.
            Phase::PrefillRuns
                if hd == 256
                    && super::attn_pf16_enabled()
                    && e.pack_version() >= [0, 25, 0]
                    && e.kernels_pf_runs_available()
                    && e.has_attn_prefill_f16_paged() =>
            {
                let maxn = runs.iter().map(|r| r.len).max().unwrap_or(0);
                e.pf_runs_register(Some((&sc.d_run_offs, runs.len() as u32, maxn as u32)))?;
                let walked = e.attn_prefill_f16_paged(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    &sc.d_slots,
                    tab,
                    bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                );
                e.pf_runs_register(None)?;
                walked?
            }
            Phase::PrefillRuns => {
                // the tile table is staged once per walk (`stage_inputs_runs`);
                // a tile that spills past its run's end is masked row by row
                // (`slots[b] == slot`), and the spilled rows are covered by their
                // own run's tiles - the kernel writes nothing for a foreign row
                e.attn_prefill_batch_paged(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    &sc.d_slots,
                    tab,
                    bps,
                    &sc.d_tile_row0,
                    &sc.d_tile_slot,
                    n_qtiles(runs),
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            // The tensor-core prefill for a 256-wide head: the paged
            // dispatcher's v4 arm at G=12 on pack >= 0.25 (f16 or e4m3 pool,
            // K/V staged once per kv head for all 12 q heads, O in f32 - the
            // multi-run walks take the same arm, see above), P6i's WMMA tile on
            // an older body (f16 only: S and O on f16 fragments loaded straight
            // from the cache). The tiled f32 walk below stages every K/V tile
            // per (q head, 16-query tile) block - 12 q heads re-reading each kv
            // head's bytes - and dots 16 keys over 32 lanes: 2048 fresh rows
            // 21.7 ms a layer against v4's 1.9 and the WMMA tile's 2.2
            // (bench/attn_prefill_runs_q4x_gb10_bench.cu). A numerics CLASS
            // change against it (f16 Q/K/V inputs), judged by the golden;
            // `PADDOCK_Q38FN_ATTN_PF16=0` is the A/B.
            Phase::Prefill
                if hd == 256
                    && (KV() == KvDtype::Fp16 || e.pack_version() >= [0, 25, 0])
                    && super::attn_pf16_enabled()
                    && e.has_attn_prefill_f16_paged() =>
            {
                e.attn_prefill_f16_paged(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    &sc.d_slots,
                    tab,
                    bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            Phase::Prefill => e.attn_prefill_paged(
                &sc.d_qn,
                kc,
                vc,
                &sc.d_sinks,
                &mut sc.d_attn,
                &sc.d_pos,
                &sc.d_slots,
                tab,
                bps,
                nh,
                nkv,
                hd,
                kv_dim,
                0,
                n,
                scale,
                KV(),
            )?,
            // one query row per slot against that slot's carried cache - the same
            // entry either way; it already takes a slot vector and `n` rows
            // The parallel-score walk (slot 536) where the pack carries it. Same
            // grid and the same result class; it just stops parking seven of eight
            // warps at a barrier while one does the dot product.
            // `PADDOCK_Q38FN_ATTN_PS=0` restores the shipped walk.
            // SPLIT-KV fmha (slot 545): grid.z KV slices + a sink-seeded merge
            // pass. At c1 the un-split form is 24 CTAs on 148 SMs (39 us/layer
            // vs the rival's 9.1). Own numeric class; `PADDOCK_Q38FN_FMHA_SP=S`
            // arms it, battery judges.
            Phase::Decode | Phase::DecodeBatch
                if e.has_attn_paged_modes()
                    && super::attn_fmha_sp() >= 2
                    && super::attn_fmha_enabled()
                    && n <= 64
                    && (hd == 128 || hd == 256) =>
            {
                e.attn_decode_fmha_sp_paged(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &mut sc.d_fmha_part,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    tab,
                    bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    0,
                    n,
                    super::attn_fmha_sp(),
                    scale,
                    KV(),
                )?
            }
            // FMHA-style decode attention (slot 537), preferred where the pack
            // carries it: per-warp key streams with (m, l, acc) in registers, so
            // the tile walk's ~3 barriers per 16 keys disappear and shared drops
            // from 32.9 KB to 8.25 KB. head_dim 128/256 only - the register
            // layout needs (head_dim/32) % 4 == 0.
            // `PADDOCK_Q38FN_ATTN_FMHA=0` falls back to the walk below.
            Phase::Decode | Phase::DecodeBatch
                if e.has_attn_paged_modes()
                    && super::attn_fmha_enabled()
                    && (hd == 128 || hd == 256) =>
            {
                e.attn_decode_fmha_paged(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    tab,
                    bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            Phase::Decode | Phase::DecodeBatch
                if e.has_attn_paged_modes() && super::attn_ps_enabled() =>
            {
                e.attn_decode_batch_ps_paged(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    tab,
                    bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            Phase::Decode | Phase::DecodeBatch => e.attn_decode_batch_paged(
                &sc.d_qn,
                kc,
                vc,
                &sc.d_sinks,
                &mut sc.d_attn,
                &sc.d_pos,
                Some(&sc.d_slots),
                tab,
                bps,
                nh,
                nkv,
                hd,
                kv_dim,
                0,
                n,
                scale,
                KV(),
            )?,
        }
    }
    Ok(())
}

/// The routed pair on GGUF expert seats (k-quant / i-quant streams): the
/// qwen35 token-batched MoE class - int8 activations per 32 with f32 block
/// scales, exact int dots - through the `[moe_offload]` slot cache when the
/// layer carries one and this launch's rows fit it (resolve ids -> slots on
/// device, fill the misses from the host mirror, run the unchanged pair
/// over the slot planes with the remapped ids). Otherwise the seats serve
/// directly - VRAM planes, or host-mapped zero-copy over PCIe. Leaves
/// `d_mix` = the top-k-weighted routed output `[n, hidden]`, exactly what
/// the NVFP4 arm leaves.
#[allow(clippy::too_many_arguments)]
/// Column chunk the grouped down (slot 589) walks the output in. The partials
/// plane is [pairs, chunk]: 42 MB at a 4096-row wave against the 419 MB a
/// full-width plane would take, and the chunking costs only a re-stage of
/// activations the block loads anyway.
const MOE_DOWN_CHUNK: usize = 256;

/// Rows from which an UNGROUPED routed down takes the column-tiled kernel
/// (`_cols`) instead of the per-(row, pair) kernel. It used to be 2, and that
/// was never priced at those widths: the GB10 kernel bench
/// (`bench/fnmoe_gb10_bench.cu`, 2026-09-12, UD-IQ3_XXS shape, DRAM-cold) read
/// the production pair at 62.1 us / 148 GB/s at c1 and 496 us / 149 GB/s at c8
/// against `_cols` at 103-162 us and 766-1131 us - slower at both widths - and
/// the verify-walk phase census (2026-09-14) showed the down growing 3.5 ->
/// 9.8 -> 19.3 -> 37.7 ms/walk at 1/2/4/8 rows, where the gate_up pair beside
/// it grew 4.3 -> 7.6 -> 13.6 -> 24.7. The two kernels are bit-exact, so this
/// is a speed election only. Above 16 rows `_cols` keeps its election (the
/// prefill chunks that stay below the grouped rung). `PADDOCK_Q38FN_DOWN_COLS_MIN`
/// pins it for the A/B (2 = the old election).
fn down_cols_min_rows() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_DOWN_COLS_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(17)
    })
}

/// Blocks a `pd_moe_align_bm(bm)` layout needs for `rows` routed pairs over
/// `n_expert` experts: every expert rounds its own count up to a whole block,
/// so the worst case is one partial block per expert on top of the rows.
///
/// Clamped to `rows`, which is the other bound and the binding one whenever a
/// launch touches fewer experts than the pool holds. An expert that takes no
/// row contributes no block (pd_moe_align_kernel scans ceil(count/bm), which
/// is 0 at count 0), and a present expert contributes at most its own count,
/// so the total can never exceed `rows`. Without the clamp a 35-row verify
/// batch over 512 experts asks for a 452-block grid to run 26 blocks of work -
/// the count is only a grid/allocation bound, but an unclamped one makes
/// the grouped arm look absurd at decode widths and is why it was only ever
/// elected for prefill. Prefill is unaffected (at 2050 rows the first term
/// already wins).
/// Widest token wave the W4A4 routed pair will serve, set by what its
/// partials plane costs: n * (k+1) * hidden f32, which at 512 tokens, top-10
/// and hidden 2560 is 57.7 MB. Full width (max_tokens 4096) would be 461 MB
/// of the KV pool's headroom for an arm that only runs at prefill, and the
/// GEMV above this is correct, just slower.
const NVF4_BS_MAX_ROWS: usize = 512;

fn grp_align_blocks(rows: usize, n_expert: usize, bm: usize) -> usize {
    (rows + n_expert * (bm - 1)).div_ceil(bm).min(rows)
}

/// Entries the moe_align_bm sorted arrays need for `pairs` routed pairs at
/// any group size the election can pick (`kq_moe_group_for`: 8 or 16) - the
/// largest of blocks x bm, which is the LARGEST bm's, not the smallest's.
fn moe_sorted_capacity(pairs: usize, n_expert: usize) -> usize {
    [8usize, 16]
        .iter()
        .map(|&bm| grp_align_blocks(pairs, n_expert, bm) * bm)
        .max()
        .unwrap_or(0)
}

/// Whether the mix `w` reads its normalized state through the rebuild
/// consumers at `n` rows (slots 607 / 608): its inject is a separate f32
/// matvec and its up + mix takes the single-launch rebuild rung. `combine`
/// (which then stores no state) and `hc_mix_pass` (which then has none to
/// read) decide on this one predicate.
fn hc_takes_rn(
    e: &GpuExecutor,
    w: &HcW,
    n: usize,
    hc: usize,
    hidden: usize,
    stage: &DenseStage,
) -> bool {
    w.inject_rows == 0
        && w.inject
            .as_ref()
            .is_some_and(|wi| wi.buf.len() >= hc * hc * hidden)
        && w.up.takes_hcmix_rn(e, n, hc, hidden, stage)
}

/// Whether the combine before the rebuilt mix `w` also folds that mix's inject
/// into its norm pass (slot 609): a [hc][hc * hidden] f32 inject at hc 4, and
/// scratch for its partials. A re-association of the matvec's dot, so it has
/// its own switch.
fn hc_inj_fold_ok(
    e: &GpuExecutor,
    w: &HcW,
    n: usize,
    hc: usize,
    hidden: usize,
    injp_len: usize,
) -> bool {
    hc == 4
        && super::hc_inj_fold_enabled()
        && e.has_q4x_combine_norm_q8mmq_nsi()
        && injp_len >= n * hc * hc
        && w.inject
            .as_ref()
            .is_some_and(|wi| wi.buf.len() == hc * hc * hidden)
}

/// One span of a single-sequence GDN prefill walk, rows `from..from + len`:
/// slot 604's segment-tiled walk when `seg`, slot 596's P-split walk
/// otherwise. Both are sequential token loops, so spans compose exactly.
#[allow(clippy::too_many_arguments)]
fn gdn_walk_rows(
    e: &GpuExecutor,
    sc: &mut Scratch,
    state: &mut CudaSlice<f32>,
    st_off: usize,
    from: usize,
    len: usize,
    (hv, kd, vd): (usize, usize, usize),
    seg: bool,
) -> Result<(), GpuModelError> {
    if seg {
        e.gated_delta_recurrent_seg_rows(
            &sc.d_dq,
            &sc.d_dk,
            &sc.d_dv,
            &sc.d_g,
            &sc.d_beta,
            state,
            st_off,
            &mut sc.d_dattn,
            &mut sc.d_dnrn,
            from,
            len,
            hv,
            kd,
        )?;
    } else {
        e.gated_delta_recurrent_pn_rows(
            &sc.d_dq,
            &sc.d_dk,
            &sc.d_dv,
            &sc.d_g,
            &sc.d_beta,
            state,
            st_off,
            &mut sc.d_dattn,
            &mut sc.d_dnrn,
            from,
            len,
            hv,
            kd,
            vd,
        )?;
    }
    Ok(())
}

/// Counts how many distinct experts a routed launch actually touches, against
/// the `rows` windows the pair kernel will unpack.
///
/// Asked of the speculative verify batch specifically. `kq_moe_routed`'s
/// grouped arm is gated on `rows > n_expert` because at decode widths the
/// rows are assumed to spread thin over 512 experts - true when the rows are
/// unrelated, which is every ordinary decode. A depth-K verify batch is the
/// one decode shape where they are not unrelated: K+1 consecutive positions
/// of one sentence, and MoE routing is known to correlate between adjacent
/// tokens. If it correlates here, every duplicate is a weight window the pair
/// kernel unpacks twice and a fetch we already paid for.
///
/// Bucketed by width, because an aggregate over a serving run is worthless
/// here: a prefill launch carries n~128 rows and collides ~63% by chance
/// (1280 draws over 512 experts), so it swamps the decode launches that the
/// question is about AND it already takes the grouped arm. Only the 2..=8
/// bucket is the speculative verify.
///
/// Dev instrument (`PADDOCK_Q38FN_ROUTE_CENSUS=1`): it syncs the stream and
/// copies the index plane every launch, so a census run is not a timing run.
fn route_census(e: &GpuExecutor, sc: &Scratch, n: usize, k: usize) {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("PADDOCK_Q38FN_ROUTE_CENSUS").is_ok_and(|v| v == "1")) {
        return;
    }
    // batch 1 has nothing to share
    if n < 2 {
        return;
    }
    let rows = n * k;
    let Ok(idx) = e.to_host_u32(&sc.d_idx) else {
        return;
    };
    let Some(ids) = idx.get(..rows) else { return };
    let mut seen_all = ids.to_vec();
    seen_all.sort_unstable();
    let mut seen = seen_all.clone();
    seen.dedup();
    // the widest single expert in this launch: separates "a few hot experts"
    // from "uniformly a bit denser", which decides whether a compacted CSR
    // gets stragglers
    let mut hottest = 1usize;
    {
        let mut run = 1usize;
        let v = &seen_all;
        for i in 1..v.len() {
            if v[i] == v[i - 1] {
                run += 1;
                if run > hottest {
                    hottest = run;
                }
            } else {
                run = 1;
            }
        }
    }
    // 0: n 2..=8 (the speculative verify band), 1: 9..=32, 2: >32 (prefill)
    let b = if n <= 8 {
        0
    } else if n <= 32 {
        1
    } else {
        2
    };
    static LAUNCH: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static ROWS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static UNIQ: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static WIDTH: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static HOT: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static TOTAL: AtomicU64 = AtomicU64::new(0);
    LAUNCH[b].fetch_add(1, Relaxed);
    ROWS[b].fetch_add(rows as u64, Relaxed);
    UNIQ[b].fetch_add(seen.len() as u64, Relaxed);
    WIDTH[b].fetch_add(n as u64, Relaxed);
    HOT[b].fetch_add(hottest as u64, Relaxed);
    let t = TOTAL.fetch_add(1, Relaxed) + 1;
    if t.is_multiple_of(4800) {
        for (i, tag) in ["spec n2-8", "dec n9-32", "prefill  "].iter().enumerate() {
            let l = LAUNCH[i].load(Relaxed);
            if l == 0 {
                continue;
            }
            let (r, u) = (ROWS[i].load(Relaxed) as f64, UNIQ[i].load(Relaxed) as f64);
            eprintln!(
                "[route-census] {tag} launches {l:>7} mean_n {:>6.2} mean_rows {:>7.1} \
                 mean_uniq {:>7.1} re-unpacks {:>5.1}% hottest_expert {:.2} rows",
                WIDTH[i].load(Relaxed) as f64 / l as f64,
                r / l as f64,
                u / l as f64,
                100.0 * (1.0 - u / r),
                HOT[i].load(Relaxed) as f64 / l as f64,
            );
        }
    }
}

fn kq_moe_routed(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    gate: &KqSeat,
    up: &KqSeat,
    down: &KqSeat,
    cache: Option<&ExpertCache>,
    sc: &mut Scratch,
    n: usize,
) -> Result<(), GpuModelError> {
    let (h, k, ff) = (c.hidden, c.n_active, c.moe_ff);
    let rows = n * k;
    route_census(e, sc, n, k);
    // Expert-major prefill: more routed rows than slots, and the pack has
    // the wave kernels. Bytes moved are bounded by one pass over the experts
    // this launch touches (<= n_expert x slot bytes per layer), never by the
    // rows - a 256-token prompt on the 5060 Ti went from 46 s of zero-copy
    // reads to bulk fills once per expert. See `ExpertCache::n_waves`.
    if let Some(cc) = cache
        && rows > cc.slots
        && rows <= cc.max_rows
        && e.has_moe_wave()
    {
        return kq_moe_routed_waves(e, c, cc, sc, n);
    }
    let cache = cache.filter(|cc| rows <= cc.slots && rows <= cc.max_rows);
    if let Some(cc) = cache {
        e.moe_cache_resolve(cc, &sc.d_idx, rows)?;
        e.moe_cache_fill(cc, rows)?;
    }
    let idx: &CudaSlice<u32> = match cache {
        Some(cc) => cc.idx_slot(),
        None => &sc.d_idx,
    };
    let (g, u, d) = match cache {
        Some(cc) => (&cc.gate, &cc.up, &cc.down),
        None => (gate.kq(), up.kq(), down.kq()),
    };
    // one place decides which formats carry a per-16 mu term (Q2_K included -
    // the pack refuses a Q2_K seat launched without its sums)
    let needs = crate::gpu::kq_needs_sums;
    // block input -> int8 per 32
    e.quantize_q8(&sc.d_bi, &mut sc.d_xq, &mut sc.d_xs, n * h)?;
    let ng = needs(g.ty) || needs(u.ty);
    if ng {
        e.q8_sums_strided(&sc.d_xq, &mut sc.d_ssums, h, n)?;
    }
    pm_lap(e, "x-quant");
    // GROUPED gate+up (slot 586) whenever a routing has rows to share. The
    // pair kernel unpacks a weight window per ROUTED ROW, and a prefill routes
    // rows/n_expert of them to each expert - 3.9 at a 200-token prompt, so it
    // unpacked every expert 3.9 times over and MoE owned 57-62% of the walk
    // (ncu: SM 52% of peak, DRAM 23% - unpack-bound, not bandwidth-bound).
    // The grouped form pays the unpack once per (expert group, out row) and
    // is bit-identical to the pair kernel, so this needs no class gate: at
    // rows <= n_expert (every decode width here) the group holds one row and
    // the two kernels do exactly the same work, so the pair kernel keeps the
    // narrow band where its grid is the simpler one.
    // ...and at decode widths too when the rows are correlated, which is what
    // a speculative verify batch is: K+1 consecutive positions of one
    // sentence. The `rows > n_expert` bound above reads as "only a prefill can
    // share", and that is true of unrelated rows - at top-10 of 512, four
    // independent rows collide 2.5% of the time. Measured in serving on this
    // family (route_census, 23040 launches, syn_128x128_c1, 2026-09-19): the
    // n 2..=8 band runs 35.5 rows to 26.3 distinct experts, i.e. 26.0% of the
    // windows the pair kernel unpacks are re-unpacks - 10x chance - and the
    // hottest expert in a launch takes 2.89 of the 3.55 rows. The dedup grows
    // with width (n 9..=32: 181.4 rows to 52.2 uniq, 71.2%), which is the
    // interesting part, because expert fetch is the only term that scales
    // with speculative depth and is the whole reason `spec_depth_cap` is
    // pinned at 3 while the row budget asks for 7.
    let grp = (cache.is_none()
        && (rows > c.n_expert || super::moe_grp_decode_on())
        && super::moe_grp_enabled()
        && e.has_kquant_moe_grp()
        && e.has_moe_align_bm())
    .then(|| GpuExecutor::kq_moe_group_for(rows, c.n_expert));
    // TENSOR-CORE gate/up at the same prefill widths (slot 601's fused tail):
    // the int8 mma pair reads each expert's weights over 32-row sorted tiles
    // and quantizes its SwiGLU output in its own epilogue, so the float
    // intermediate and the separate quantize pass disappear; the sorted rows
    // are moved to the pair-major layout the grouped down (below, unchanged)
    // reads. The shape the kernel serves: one dtype for gate and up, a
    // super-block-aligned in_dim, a 32-aligned ff, no i-quant (no mma lane).
    // Measured on Flash-Next UD-Q4_K_XL (2026-09-14): gate/up owned 757 ms of
    // a 1656 ms 1024-token prefill on the register-tiled pair.
    let mma = grp.is_some()
        && super::moe_mma_enabled()
        && g.ty == u.ty
        && !crate::gpu::kq_is_iq(g.ty)
        && h.is_multiple_of(256)
        && ff.is_multiple_of(32)
        && e.has_kquant_moe_mma()
        && e.has_moe_q8_rows_unsort()
        && e.kernels().is_ok_and(|kt| kt.moe_align.is_some());
    // EXPERT-MAJOR tensor-core down (slot 603) behind the tensor-core gate/up:
    // one block per (64-row output strip, expert) walks all of the expert's
    // pairs straight off the gate/up's sorted rows, so the unsort to
    // pair-major, the separate sums pass and the bm = 16 CSR the tiled down
    // walks all drop out. Needs one scale per 32 weights (the flat 32-weight
    // downs) and whole 128-weight stages. bench/fnmoe_prefill_gb10_bench.cu:
    // the tiled down's time a layer was ~60% dp4a dot structure, not weight
    // bytes.
    let down_e = mma
        && crate::gpu::kq_flat32(d.ty)
        && ff.is_multiple_of(128)
        && super::moe_down_mma_enabled()
        && e.has_kquant_moe_down_mma_e();
    match grp {
        Some(bm) => {
            let blocks = grp_align_blocks(rows, c.n_expert, bm);
            if blocks * bm > sc.d_msrow.len() || blocks > sc.d_mbexp.len() {
                return Err(GpuModelError::Unsupported(format!(
                    "moe_align_bm: {rows} pairs at group {bm} need {} sorted entries and {blocks} \
                     blocks; the scratch holds {} and {}",
                    blocks * bm,
                    sc.d_msrow.len(),
                    sc.d_mbexp.len()
                )));
            }
            if !down_e {
                e.moe_align_bm(
                    idx,
                    &mut sc.d_msrow,
                    &mut sc.d_msslot,
                    &mut sc.d_mbexp,
                    n,
                    k,
                    c.n_expert,
                    bm,
                    blocks,
                )?;
            }
            if mma {
                let mb = grp_align_blocks(rows, c.n_expert, 32);
                e.moe_align(
                    idx,
                    &mut sc.d_srow32,
                    &mut sc.d_sslot32,
                    &mut sc.d_bexp32,
                    n,
                    k,
                    c.n_expert,
                    mb,
                )?;
                e.kquant_moe_gate_up_mma(
                    g,
                    u,
                    &sc.d_srow32,
                    &sc.d_bexp32,
                    &sc.d_xq,
                    &sc.d_xs,
                    ng.then_some(&sc.d_ssums),
                    &mut sc.d_sfq,
                    &mut sc.d_sfs,
                    mb,
                )?;
                if !down_e {
                    e.moe_q8_rows_unsort(
                        &sc.d_sfq,
                        &sc.d_sfs,
                        &sc.d_srow32,
                        &sc.d_sslot32,
                        &sc.d_bexp32,
                        &mut sc.d_fq,
                        &mut sc.d_fs,
                        ff,
                        k,
                        mb,
                    )?;
                }
            }
            // REGISTER-TILED pair (slot 592) when the CSR group is its own
            // tile height and K divides its slice: it stages both operands per
            // BK and reads the group's activations once per column TILE, where
            // the grouped pair kernel reads them once per column - 47 GB a
            // layer at a 2114-row wave, which is what that kernel waits on.
            let tiled = !mma
                && bm == GpuExecutor::KQ_MOE_TILE_BM
                && h.is_multiple_of(128)
                && super::moe_tile_enabled()
                && e.has_kquant_moe_gate_up_tile();
            if mma {
                // gate/up already ran on the tensor cores above
            } else if tiled {
                e.kquant_moe_gate_up_tile(
                    g,
                    u,
                    &sc.d_msrow,
                    &sc.d_msslot,
                    &sc.d_mbexp,
                    &sc.d_xq,
                    &sc.d_xs,
                    ng.then_some(&sc.d_ssums),
                    &mut sc.d_act,
                    k,
                    n,
                    blocks,
                )?;
            } else {
                e.kquant_moe_gate_up_grp(
                    g,
                    u,
                    &sc.d_msrow,
                    &sc.d_msslot,
                    &sc.d_mbexp,
                    &sc.d_xq,
                    &sc.d_xs,
                    ng.then_some(&sc.d_ssums),
                    &mut sc.d_act,
                    k,
                    n,
                    blocks,
                    bm,
                )?;
            }
        }
        None => e.kquant_moe_gate_up(
            g,
            u,
            idx,
            &sc.d_xq,
            &sc.d_xs,
            ng.then_some(&sc.d_ssums),
            &mut sc.d_act,
            k,
            n,
        )?,
    }
    pm_lap(e, "gate_up");
    if !mma {
        e.quantize_q8(&sc.d_act, &mut sc.d_fq, &mut sc.d_fs, rows * ff)?;
    }
    let nd = needs(d.ty);
    if nd && !down_e {
        e.q8_sums_strided(&sc.d_fq, &mut sc.d_ssums, ff, rows)?;
    }
    pm_lap(e, "act-quant");
    // Column-tiled down (slot 587) on the same prefill shapes the grouped
    // gate+up takes: the plain kernel's block computes one output float and
    // spends its life on dependent weight loads (ncu: SM 32%, DRAM 21%, and
    // 1.25 windows per lane to hide them with). Bit-identical, so the choice
    // is only about keeping the die full - which is why it is the prefill
    // widths that take it.
    if down_e {
        // The partials plane is sized for the widest walk at MOE_DOWN_CHUNK
        // columns ([max_tokens x n_active x 256]); a narrower walk spends the
        // same bytes on wider chunks, in whole 64-row strips. Narrow launches
        // cost this kernel more than they cost the tile: a hot expert's CTA
        // walks all of its pairs, and every 256-column launch waits on it with
        // only 4 of its strips running (bench/fnmoe_prefill_gb10_bench.cu at a
        // skewed routing, 1024 tokens: 6.5 ms a layer at 256 columns, 5.3 at
        // 1280, 5.2 at 2560 - bit-identical). The per-expert span pass reruns
        // a chunk (one 256-thread block).
        let mb = grp_align_blocks(rows, c.n_expert, 32);
        let chunk = (sc.d_dpart.len() / rows / 64 * 64).clamp(64, h);
        let mut o0 = 0;
        while o0 < h {
            let ocols = (h - o0).min(chunk);
            e.kquant_moe_down_mma_e(
                d,
                &sc.d_srow32,
                &sc.d_sslot32,
                &sc.d_bexp32,
                &sc.d_topw,
                &sc.d_sfq,
                &sc.d_sfs,
                &mut sc.d_emap,
                &mut sc.d_dpart,
                o0,
                ocols,
                k,
                n,
                c.n_expert,
                mb,
            )?;
            e.moe_part_fold_at(&sc.d_dpart, &mut sc.d_mix, h, o0, ocols, k, n)?;
            o0 += ocols;
        }
    } else if let (Some(bm), true) = (grp, e.has_kquant_moe_down_grp()) {
        // Expert-grouped down over the CSR the gate_up half already built:
        // the ungrouped kernels unpack a weight row per (routed pair, column),
        // which is 42% of a wave walk (17.2 ms a layer at 2114 rows). One
        // column chunk at a time, each chunk's partials folded in slot order
        // before the next - a grouped block holds different tokens, so the
        // slot sum cannot happen inside it.
        let blocks = grp_align_blocks(rows, c.n_expert, bm);
        let mut o0 = 0;
        while o0 < h {
            let ocols = (h - o0).min(MOE_DOWN_CHUNK);
            if bm == GpuExecutor::KQ_MOE_TILE_BM
                && ff.is_multiple_of(128)
                && super::moe_tile_enabled()
                && e.has_kquant_moe_down_tile()
            {
                e.kquant_moe_down_tile(
                    d,
                    &sc.d_msrow,
                    &sc.d_msslot,
                    &sc.d_mbexp,
                    &sc.d_topw,
                    &sc.d_fq,
                    &sc.d_fs,
                    nd.then_some(&sc.d_ssums),
                    &mut sc.d_dpart,
                    o0,
                    ocols,
                    k,
                    n,
                    blocks,
                )?;
                e.moe_part_fold_at(&sc.d_dpart, &mut sc.d_mix, h, o0, ocols, k, n)?;
                o0 += ocols;
                continue;
            }
            e.kquant_moe_down_grp(
                d,
                &sc.d_msrow,
                &sc.d_msslot,
                &sc.d_mbexp,
                &sc.d_topw,
                &sc.d_fq,
                &sc.d_fs,
                nd.then_some(&sc.d_ssums),
                &mut sc.d_dpart,
                o0,
                ocols,
                k,
                n,
                blocks,
                bm,
            )?;
            e.moe_part_fold_at(&sc.d_dpart, &mut sc.d_mix, h, o0, ocols, k, n)?;
            o0 += ocols;
        }
    } else if (grp.is_some() || n >= down_cols_min_rows()) && e.has_kquant_moe_down_cols() {
        e.kquant_moe_down_cols(
            d,
            idx,
            &sc.d_topw,
            &sc.d_fq,
            &sc.d_fs,
            nd.then_some(&sc.d_ssums),
            &mut sc.d_mix,
            k,
            n,
            4,
        )?;
    } else {
        e.kquant_moe_down(
            d,
            idx,
            &sc.d_topw,
            &sc.d_fq,
            &sc.d_fs,
            nd.then_some(&sc.d_ssums),
            &mut sc.d_mix,
            k,
            n,
        )?;
    }
    pm_lap(e, "down");
    Ok(())
}

/// The routed pair served expert-major through the slot cache (the prefill
/// class of `kq_moe_routed`): plan the waves once, then per wave resolve +
/// fill its experts, mask + compact its pairs, and run the LIST pair kernels
/// over exactly those pairs - gate_up striding the pair list, down striding
/// the token list and accumulating into a zeroed `d_mix`. A row's k pairs
/// are summed in wave order instead of pair order, the same f32
/// reassociation class the sorted MoE path carries; every (token, column)
/// has one writer per wave and the waves run in sequence, so the result is
/// deterministic. Waves are a fixed count so the launch sequence is
/// capture-stable; an empty wave resolves nothing, its lists are empty and
/// its LIST launches return at the count.
fn kq_moe_routed_waves(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    cc: &ExpertCache,
    sc: &mut Scratch,
    n: usize,
) -> Result<(), GpuModelError> {
    let (h, k, ff) = (c.hidden, c.n_active, c.moe_ff);
    let rows = n * k;
    let needs = crate::gpu::kq_needs_sums;
    let (g, u, d) = (&cc.gate, &cc.up, &cc.down);
    e.moe_wave_plan(cc, &sc.d_idx, rows)?;
    // block input -> int8 per 32, once for every wave
    e.quantize_q8(&sc.d_bi, &mut sc.d_xq, &mut sc.d_xs, n * h)?;
    let ng = needs(g.ty) || needs(u.ty);
    if ng {
        e.q8_sums_strided(&sc.d_xq, &mut sc.d_ssums, h, n)?;
    }
    let nd = needs(d.ty);
    e.zero_region(&mut sc.d_mix, 0, n * h)?;
    for w in 0..cc.n_waves {
        e.moe_wave_resolve(cc, w)?;
        e.moe_cache_fill(cc, cc.slots)?;
        e.moe_wave_mask(cc, &sc.d_idx, rows, k, w)?;
        let idx = cc.idx_wave();
        let (pairs, n_pairs, rws, n_rws) = cc.wave_lists();
        e.kquant_moe_gate_up_list(
            g,
            u,
            idx,
            &sc.d_xq,
            &sc.d_xs,
            ng.then_some(&sc.d_ssums),
            &mut sc.d_act,
            k,
            n,
            pairs,
            n_pairs,
        )?;
        // the act rows of absent pairs hold stale values; down skips them
        e.quantize_q8(&sc.d_act, &mut sc.d_fq, &mut sc.d_fs, rows * ff)?;
        if nd {
            e.q8_sums_strided(&sc.d_fq, &mut sc.d_ssums, ff, rows)?;
        }
        e.kquant_moe_down_list(
            d,
            idx,
            &sc.d_topw,
            &sc.d_fq,
            &sc.d_fs,
            nd.then_some(&sc.d_ssums),
            &mut sc.d_mix,
            k,
            n,
            rws,
            n_rws,
        )?;
    }
    Ok(())
}

/// Sorted-expert scratch for a Q8_0 seat (`q8_moe_routed` above
/// `Q8_SORTED_MIN_ROWS` rows), sized for `max_rows` rows.
pub(crate) struct Q8Wide {
    pub srow: CudaSlice<u32>,
    pub sslot: CudaSlice<u32>,
    pub bexp: CudaSlice<u32>,
    pub fused: CudaSlice<f32>,
    pub fq: CudaSlice<i8>,
    pub fs: CudaSlice<f32>,
    pub part: CudaSlice<f32>,
    pub max_rows: usize,
    pub max_blocks: usize,
}

/// Rows above which a Q8_0 seat walks its experts sorted: token-batched, every
/// routed pair reads its expert's rows once per pair; sorted, once per block of
/// 32 pairs.
const Q8_SORTED_MIN_ROWS: usize = 32;

/// The routed pair on all-Q8_0 seats (the MTP head's block): the qwen3.6-A3B
/// Q8_0 SwiGLU class - int8 activations per 32, f32 block scales, exact int
/// dots. Leaves `d_mix` = the top-k-weighted routed output, as the other arms.
#[allow(clippy::too_many_arguments)]
fn q8_moe_routed(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    gate: &crate::gpu::RepackedQ8,
    up: &crate::gpu::RepackedQ8,
    down: &crate::gpu::RepackedQ8,
    sc: &mut Scratch,
    n: usize,
    wide: Option<&mut Q8Wide>,
) -> Result<(), GpuModelError> {
    let (h, k, ff) = (c.hidden, c.n_active, c.moe_ff);
    e.quantize_q8(&sc.d_bi, &mut sc.d_xq, &mut sc.d_xs, n * h)?;
    match wide {
        Some(wd) if n > Q8_SORTED_MIN_ROWS => {
            if n > wd.max_rows {
                return Err(GpuModelError::Unsupported(format!(
                    "Q8_0 expert seat: {n} rows, the sorted scratch holds {}",
                    wd.max_rows
                )));
            }
            let nb = wd.max_blocks;
            e.moe_align(
                &sc.d_idx,
                &mut wd.srow,
                &mut wd.sslot,
                &mut wd.bexp,
                n,
                k,
                c.n_expert,
                nb,
            )?;
            e.q8_0_moe_gate_up_sorted(
                gate,
                up,
                &wd.srow,
                &wd.bexp,
                &sc.d_xq,
                &sc.d_xs,
                &mut wd.fused,
                nb,
            )?;
            e.quantize_q8(&wd.fused, &mut wd.fq, &mut wd.fs, nb * 32 * ff)?;
            e.q8_0_moe_down_sorted(
                down,
                &wd.srow,
                &wd.sslot,
                &wd.bexp,
                &sc.d_topw,
                &wd.fq,
                &wd.fs,
                &mut wd.part,
                k,
                nb,
            )?;
            e.zero_region(&mut sc.d_mix, 0, n * h)?;
            e.moe_slot_combine(&wd.part, &mut sc.d_mix, h, k, n)?;
        }
        _ => {
            e.q8_0_moe_gate_up(gate, up, &sc.d_idx, &sc.d_xq, &sc.d_xs, &mut sc.d_act, k, n)?;
            e.quantize_q8(&sc.d_act, &mut sc.d_fq, &mut sc.d_fs, n * k * ff)?;
            e.q8_0_moe_down(
                down,
                &sc.d_idx,
                &sc.d_topw,
                &sc.d_fq,
                &sc.d_fs,
                &mut sc.d_mix,
                k,
                n,
            )?;
        }
    }
    Ok(())
}

/// 512-expert top-10 NVFP4 MoE + the bf16 sigmoid-gated shared expert.
#[allow(clippy::too_many_arguments)]
fn moe_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::MoeW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    n: usize,
    fork_ok: bool,
    // DECODE PHASE, not decode WIDTH. The c16 decomposition (2026-08-30)
    // convicted the n<=32 bound: serve PREFILL CHUNKS of <=32 rows leaked
    // into the fold (513-row alignment trap) and the sh 2seg - the lone
    // RFOLD+SH leg ran -6.8% vs base while both arms win at decode widths.
    decode: bool,
    // sorted-expert scratch for a Q8_0 seat above token-batch widths (the MTP
    // head seeding a prompt); the trunk passes None
    wide: Option<&mut Q8Wide>,
) -> Result<bool, GpuModelError> {
    let (h, k, sff) = (c.hidden, c.n_active, c.shared_ff);
    // Router: softmax over all experts, top-k, renormalized over the picks -
    // which is exactly moe_topk_batch's local softmax over the selected logits
    // (the global denominator cancels). Bias is zero for this family.
    // one launch covers the router AND the shared expert's scalar gate (row
    // n_expert). At batch 1 the topk reads logits[0..n_expert] in place; above
    // it the two are row-segment reads of the same residency, because a fused
    // output is only contiguous per projection at one row.
    let fused_router = n == 1;
    // row stride of the logits plane: the low-M GEMM arm below writes the
    // PADDED width, every other arm the folded ne+1. `router_folded` says the
    // plane carries the shared-expert gate as row n_expert (every batch arm
    // here does; the plain 512-row arm does not).
    let router_rs = c.n_expert + 1;
    let mut router_folded = false;
    if fused_router {
        {
            // TGV lane (slot 547) on the bf16 router twin, fed by the
            // d_bi mirror (already fresh at n==1 - zero extra launches).
            // block-per-row gemv first: TGV's 64-row tiles grid only 9 CTAs
            // for a 513-row router (0.49 TB/s), and the gemv reads d_bi in
            // f32 directly, so it needs no bf16 mirror either.
            let gemv_done = match &w.router16 {
                Some(r16) if super::router_gemv_on() => {
                    e.bf16_gemv_bytes(r16, &sc.d_bi, &mut sc.d_logits, h, c.n_expert + 1)?
                }
                _ => false,
            };
            let tgv_done = gemv_done;
            let sp = sk_split();
            let done = tgv_done
                || (sp >= 2
                    && e.matvec_f32_sk(
                        &w.router.buf,
                        h,
                        c.n_expert + 1,
                        &sc.d_bi,
                        &mut sc.d_logits,
                        &mut sc.d_skp,
                        &mut sc.d_skc,
                        sp,
                    )?);
            if !done {
                e.matvec_f32_raw(
                    &w.router.buf,
                    h,
                    c.n_expert + 1,
                    &sc.d_bi,
                    &mut sc.d_logits,
                    1,
                )?;
            }
        }
    } else if super::router_fold_on() && decode && n <= 32 {
        // one launch covers router AND the shared gate row, exactly like the
        // n==1 arm: logits land [n, ne+1]; the strided topk/gated-add below
        // read the same values in the same order (bit-identical picks).
        //
        // DECODE-BAND only (n <= 32). The first board with this unbounded
        // regressed every batched cell 3-24%, p50 +15% across the ladder:
        // a PREFILL WAVE also routes here (n up to 512), and the folded
        // 513-row plane fails the matvec launcher's `out_dim & 7` gate that
        // sends the 512-row router to the tile kernel at batch >= 16 - the
        // wave's router fell from the tile arm to the scalar BT walk. The
        // example bench never runs the wave path, which is why b8/b32 legs
        // measured clean ([[prefill-path-decides-prefix-reuse]] again).
        e.matvec_f32_rows(
            &w.router.buf,
            0,
            h,
            c.n_expert + 1,
            &sc.d_bi,
            &mut sc.d_logits,
            n,
        )?;
        router_folded = true;
    } else {
        e.matvec_f32_rows(
            &w.router.buf,
            0,
            h,
            c.n_expert,
            &sc.d_bi,
            &mut sc.d_logits,
            n,
        )?;
    }
    // The router projection on its own: [h -> ne] at prefill width. Split out
    // because "pre-route" bills whatever preceded it, and the fork block below
    // is DECODE-ONLY (fork_ok gates on the phase), so at prefill that window
    // is not the shared expert it looks like.
    pm_lap(e, "router");
    let folded_wide = !fused_router && router_folded;
    // The shared expert reads `d_bi` and never touches the routed chain - the
    // two only meet at the gated add below. The routed chain is ~64 us/layer
    // (topk + gu_swiglu + down_acc) and the shared chain ~30 us, so running
    // the shared expert on the side stream hides it entirely.
    let moe_forked = fork_ok && super::gdn_fork_enabled() && e.side_fork().is_ok();
    if moe_forked {
        // shared expert: swiglu, then a per-token sigmoid scalar gate.
        // gate|up ride one 2-segment launch when the fused plane is loaded
        // (per-segment bit-identity is the export's contract).
        // slot 546 (PADDOCK_Q38FN_SH2): at n=1 the gate|up pair + swiglu run
        // as one dual-plane GEMV writing silu(g)*u into d_shg directly.
        let sh2 = n == 1
            && super::sh2_on()
            && match (super::plane_bytes(&w.sh_gate), super::plane_bytes(&w.sh_up)) {
                (Some(wg), Some(wu)) => {
                    e.bf16_gemv2_swiglu(&wg.bytes, &wu.bytes, &sc.d_bi, &mut sc.d_shg, h, sff, n)?
                }
                _ => false,
            };
        let sh_fused = !sh2
            && n >= 2
            && decode
            && match &w.sh_gu {
                Some(f) => {
                    e.bf16_gemm_2seg(f, &sc.d_bi, &mut sc.d_shg, &mut sc.d_shu, sff, sff, n)?
                }
                None => false,
            };
        if !sh2 && !sh_fused {
            w.sh_gate.matmul(e, &sc.d_bi, &mut sc.d_shg, n, stage)?;
            w.sh_up.matmul(e, &sc.d_bi, &mut sc.d_shu, n, stage)?;
        }
        if !sh2 {
            e.swiglu(&mut sc.d_shg, &sc.d_shu, n * sff)?;
        }
        w.sh_down.matmul(e, &sc.d_shg, &mut sc.d_shd, n, stage)?;
        e.side_end()?;
    }
    // The routed pair's laps used to open at whatever lapped last, so the
    // router and the shared-expert side billed as "x-quant" - 1.07 ms a layer
    // for two streaming kernels that move 17 MB. Split here.
    pm_lap(e, "pre-route");
    if !(folded_wide
        && e.moe_topk_batch_s(
            &sc.d_logits,
            &sc.d_zero_bias,
            c.n_expert,
            router_rs,
            k,
            &mut sc.d_idx,
            &mut sc.d_topw,
            n,
        )?)
    {
        e.moe_topk_batch(
            &sc.d_logits,
            &sc.d_zero_bias,
            c.n_expert,
            k,
            &mut sc.d_idx,
            &mut sc.d_topw,
            n,
        )?;
    }
    pm_lap(e, "route");
    let fused = false;
    match &w.seats {
        ExpertSeats::Nvf4 { gate, up, down } => {
            // W4A4 PREFILL arm (slot 631 + 408): the checkpoint quantizes the
            // routed experts at `input_activations` 4-bit, and the GEMV below
            // serves them W4A16 off f32 - GEMV cost for precision the export
            // was never quantized at. The bs pair is BM=32 token columns, so
            // it is the PREFILL geometry by construction: at decode routing a
            // 32-wide block is ~7.5% live (measured for the _st arm), which is
            // why the width gate is here and the decode tick keeps the GEMV
            // until the BM=8 tiled twin exists.
            let rows = n * k;
            // `nvf4_moe_down_bs` lands per-(token, slot) partials at
            // part[(tok*np + slot)*embd], so it needs n * (k+1) * hidden
            // floats. `d_moe_part` is a DECODE-BAND buffer (64 rows x the
            // z-split's halved slots) and a 128-token prefill overruns it 4x -
            // measured as CUDA_ERROR_ILLEGAL_ADDRESS on the first serve where
            // this arm actually ran. The bench never saw it because it
            // allocates its own part. Refuse rather than scribble: the pair
            // wants a chunked partials plane of its own (nemotron sizes one
            // per chunk), which is the work this arm is waiting on.
            let part_need = n * (k + 1) * h;
            let bs = n >= super::nvf4_bs_min_rows()
                && e.has_nvf4_moe_gu_swiglu_bs()
                && e.has_nvf4_moe_bs()
                && c.hidden.is_multiple_of(32)
                && c.moe_ff.is_multiple_of(16)
                && sc.d_nvf4_part.len() >= part_need;
            if bs {
                let nb = grp_align_blocks(rows, c.n_expert, 32);
                e.moe_align(
                    &sc.d_idx,
                    &mut sc.d_srow32,
                    &mut sc.d_sslot32,
                    &mut sc.d_bexp32,
                    n,
                    k,
                    c.n_expert,
                    nb,
                )?;
                e.quantize_nvf4(&sc.d_bi, &mut sc.d_xq4, &mut sc.d_xs4, n * h)?;
                e.nvf4_moe_gu_swiglu_bs(
                    gate,
                    up,
                    &sc.d_srow32,
                    &sc.d_bexp32,
                    &sc.d_xq4,
                    &sc.d_xs4,
                    &mut sc.d_nfq,
                    &mut sc.d_nfs,
                    nb,
                )?;
                e.nvf4_moe_down_bs(
                    down,
                    &sc.d_srow32,
                    &sc.d_sslot32,
                    &sc.d_bexp32,
                    Some(&sc.d_topw),
                    &sc.d_nfq,
                    &sc.d_nfs,
                    &mut sc.d_nvf4_part,
                    k,
                    k + 1,
                    0,
                    nb,
                )?;
                // the INIT twin: residual = sum. The plain `moe_slot_combine`
                // ACCUMULATES, so using it here added every layer's MoE on top
                // of the previous layer's stale d_mix - coherent-looking text
                // that drifts from the GEMV arm on every prompt. The GEMV path
                // beside this one has always used _init.
                e.moe_slot_combine_init(&sc.d_nvf4_part, &mut sc.d_mix, h, k + 1, n)?;
            } else {
                e.q4x_moe_gu_swiglu(gate, up, &sc.d_idx, &sc.d_bi, &mut sc.d_act, k, n)?;
                // z-split + deterministic combine (ncu: warp-per-row is CTA-starved;
                // ascending-z init-fold == the serial walk's exact order)
                let zs = n <= 64;
                if zs {
                    e.nvf4_moe_down_acc(
                        down,
                        &sc.d_idx,
                        &sc.d_topw,
                        &sc.d_act,
                        &mut sc.d_mix,
                        Some(&mut sc.d_moe_part),
                        k,
                        n,
                        false,
                    )?;
                    e.moe_slot_combine_init(&sc.d_moe_part, &mut sc.d_mix, h, k.div_ceil(2), n)?;
                } else {
                    e.nvf4_moe_down_acc(
                        down,
                        &sc.d_idx,
                        &sc.d_topw,
                        &sc.d_act,
                        &mut sc.d_mix,
                        None,
                        k,
                        n,
                        false,
                    )?;
                }
            }
        }
        ExpertSeats::Kq {
            gate,
            up,
            down,
            cache,
        } => kq_moe_routed(e, c, gate, up, down, cache.as_deref(), sc, n)?,
        ExpertSeats::Q8 { gate, up, down } => q8_moe_routed(e, c, gate, up, down, sc, n, wide)?,
    }
    if moe_forked {
        e.side_join()?;
    } else {
        // shared expert (unforked twin): same fused arm, same fallback
        // slot 546 (PADDOCK_Q38FN_SH2): at n=1 the gate|up pair + swiglu run
        // as one dual-plane GEMV writing silu(g)*u into d_shg directly.
        let sh2 = n == 1
            && super::sh2_on()
            && match (super::plane_bytes(&w.sh_gate), super::plane_bytes(&w.sh_up)) {
                (Some(wg), Some(wu)) => {
                    e.bf16_gemv2_swiglu(&wg.bytes, &wu.bytes, &sc.d_bi, &mut sc.d_shg, h, sff, n)?
                }
                _ => false,
            };
        let sh_fused = !sh2
            && n >= 2
            && decode
            && match &w.sh_gu {
                Some(f) => {
                    e.bf16_gemm_2seg(f, &sc.d_bi, &mut sc.d_shg, &mut sc.d_shu, sff, sff, n)?
                }
                None => false,
            };
        if !sh2 && !sh_fused {
            w.sh_gate.matmul(e, &sc.d_bi, &mut sc.d_shg, n, stage)?;
            w.sh_up.matmul(e, &sc.d_bi, &mut sc.d_shu, n, stage)?;
        }
        if !sh2 {
            e.swiglu(&mut sc.d_shg, &sc.d_shu, n * sff)?;
        }
        w.sh_down.matmul(e, &sc.d_shg, &mut sc.d_shd, n, stage)?;
    }
    if fused {
        // the shared row rides the fused combine_norm's gather instead
    } else if fused_router {
        e.q4x_add_gated_row_at(&mut sc.d_mix, &sc.d_shd, &sc.d_logits, c.n_expert, n, h)?;
    } else if folded_wide
        && e.q4x_add_gated_row_s(
            &mut sc.d_mix,
            &sc.d_shd,
            &sc.d_logits,
            c.n_expert,
            router_rs,
            n,
            h,
        )?
    {
        // gate came out of the folded router plane; nothing else to launch
    } else {
        e.matvec_f32_rows(
            &w.router.buf,
            c.n_expert,
            h,
            1,
            &sc.d_bi,
            &mut sc.d_shgate,
            n,
        )?;
        e.q4x_add_gated_row(&mut sc.d_mix, &sc.d_shd, &sc.d_shgate, n, h)?;
    }
    Ok(fused)
}

/// YaRN parameters for this family: plain rope, theta 1e7, no scaling.
/// `ext_factor = 0` makes the correction band inert - the pack kernel's way of
/// saying "no YaRN", which is what a `rope_parameters` dict with no scaling
/// means.
fn yarn_params(c: &Qwen4ExpConfig) -> (f32, f32, f32, f32, f32, f32) {
    let theta_scale = c.rope_theta.powf(-2.0 / c.rotary_dim as f32);
    (theta_scale, 1.0, 0.0, 1.0, 0.0, 1.0)
}

fn bf16_plane(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Result<QuantTensor, GpuModelError> {
    let raw = bf16_bytes(st, name, n * k)?;
    Ok(QuantTensor {
        bytes: exec.to_device_u8(raw).map_err(GpuModelError::from)?,
        ty: GgmlType::Bf16,
        dims: vec![k, n],
    })
}

/// Per-GDN-layer recurrent state, `[slots][v_heads][k_dim][v_dim]` f32.
fn alloc_recur(
    e: &Arc<GpuExecutor>,
    c: &Qwen4ExpConfig,
    slots: usize,
) -> Result<Vec<Option<CudaSlice<f32>>>, GpuModelError> {
    let state_len = slots * c.gdn_v_heads * c.gdn_k_dim * c.gdn_v_dim;
    let mut recur = Vec::with_capacity(c.n_layer);
    for li in 0..c.n_layer {
        recur.push(match c.blocks[li] {
            Qwen4ExpBlock::Gdn => Some(e.alloc(state_len)?),
            Qwen4ExpBlock::Attention => None,
        });
    }
    Ok(recur)
}

/// Every attention layer's pool planes on `pool_blocks` block ids: K and V
/// `[blocks, 16, kv_dim]` (zeroed, so the spare block and every unwritten row
/// are finite) and, with the indexer, the compressed keys `[blocks, 4, 128]`.
#[allow(clippy::type_complexity)]
fn alloc_pool_planes(
    e: &Arc<GpuExecutor>,
    c: &Qwen4ExpConfig,
    pool_blocks: usize,
    kv_row_bytes: usize,
    qsa: bool,
) -> Result<
    (
        Vec<Option<CudaSlice<u8>>>,
        Vec<Option<CudaSlice<u8>>>,
        Vec<Option<CudaSlice<half::bf16>>>,
    ),
    GpuModelError,
> {
    let kv_bytes = pool_blocks * crate::kv_pool::BLOCK_TOKENS * kv_row_bytes;
    let idx_len = pool_blocks * super::pages::IDX_ROWS_PER_PAGE * c.idx_head_dim;
    let (mut kk, mut kv, mut ix) = (Vec::new(), Vec::new(), Vec::new());
    for li in 0..c.n_layer {
        match c.blocks[li] {
            Qwen4ExpBlock::Gdn => {
                kk.push(None);
                kv.push(None);
                ix.push(None);
            }
            Qwen4ExpBlock::Attention => {
                kk.push(Some(e.alloc_u8(kv_bytes)?));
                kv.push(Some(e.alloc_u8(kv_bytes)?));
                ix.push(if qsa {
                    Some(e.stream_alloc_bf16(idx_len)?)
                } else {
                    None
                });
            }
        }
    }
    Ok((kk, kv, ix))
}

/// A checkpoint page's record over the pool planes, in layer order: each
/// attention layer's K page, V page and index rows (the planes one block id
/// addresses - every byte of a page a checkpoint draws).
fn checkpoint_layout(
    e: &Arc<GpuExecutor>,
    kv_k: &[Option<CudaSlice<u8>>],
    kv_v: &[Option<CudaSlice<u8>>],
    idx_cache: &[Option<CudaSlice<half::bf16>>],
    kv_row_bytes: usize,
    idx_row_bytes: usize,
) -> crate::ckpt_pages::PageLayout {
    use cudarc::driver::DevicePtr;
    let page = (crate::kv_pool::BLOCK_TOKENS * kv_row_bytes) as u64;
    let idx_page = (super::pages::IDX_ROWS_PER_PAGE * idx_row_bytes) as u64;
    let mut planes = Vec::new();
    for li in 0..kv_k.len() {
        let (Some(k), Some(v)) = (kv_k[li].as_ref(), kv_v[li].as_ref()) else {
            continue;
        };
        let (kp, _g1) = k.device_ptr(&e.stream);
        let (vp, _g2) = v.device_ptr(&e.stream);
        planes.push((kp, page));
        planes.push((vp, page));
        if let Some(x) = idx_cache[li].as_ref() {
            let (xp, _g3) = x.device_ptr(&e.stream);
            planes.push((xp, idx_page));
        }
    }
    crate::ckpt_pages::PageLayout::new(planes)
}

impl Scratch {
    fn new(
        e: &Arc<GpuExecutor>,
        c: &Qwen4ExpConfig,
        // rows per walk: every per-row plane
        t: usize,
        // the context: only the identity block table over the KV spans it
        ctx: usize,
        slots: usize,
        // the GGUF lane's k-quant dense planes / expert seats read int8
        // activations; the safetensors lane never does, and the pair below
        // is ~15 KB per token of max_ctx (2 GB at 128k), so it is a stub there
        kq_lanes: bool,
    ) -> Result<Self, GpuModelError> {
        // the safetensors lane is the one that seats NVFP4 routed experts,
        // which is the only seat the W4A4 pair (slot 631) can read
        let nvf4_lane = !kq_lanes;
        // QSA planes: scores for as many rows as fit QSA_SCORE_BYTES at this
        // context's block count
        let qsa_cap = ctx.div_ceil(QSA_BLOCK).max(1);
        let qsa_k = (c.idx_budget / c.idx_compress.max(1)).max(1);
        let qsa_rb = (QSA_SCORE_BYTES / (qsa_cap * 4)).clamp(1, t.max(1));
        let (h, hw, hc) = (c.hidden, c.hc_width(), c.hc_count);
        let kv_dim = c.n_kv_heads * c.head_dim;
        let q_dim = c.n_heads * c.head_dim;
        let vdim = c.gdn_v_heads * c.gdn_v_dim;
        let kdim = c.gdn_v_heads * c.gdn_k_dim;
        let mut lowm_refused = false;
        let d_lowm_warm: CudaSlice<f32> = {
            // slot 544 stores a full 64-row TILE, not 64 scalars: memcheck
            // flags a 4-byte write 1 past a 64-float y here, which fails
            // the whole load under compute-sanitizer. Slack, not 64.
            let mut yd: CudaSlice<f32> = e.alloc(256)?;
            // Warm up only where the arm can exist at all. Two gates, both
            // structural rather than discovered by launching:
            //   - the pack nulls slots 543/544 on every die but sm_100 (the
            //     kernel is tcgen05/TMEM), so `has_lowm` is the per-device
            //     truth - a 5060 Ti used to see "warm-up refused (801)" on
            //     every Flash-Next load for an arm it could never take;
            //   - the arm consumes `DensePlane::Dual` (the safetensors
            //     lane's bf16+f16 twin); the GGUF lane's dense planes are
            //     k-quant, so there is nothing for it to serve there.
            // A pack that has the entry and still refuses is the case worth
            // a warning: that is a real launch failure on a die the pack
            // claims to serve, and the lane goes off rather than the model.
            if !kq_lanes && e.has_lowm() {
                let w = e.f16_to_device(&vec![half::f16::from_f32(0.0); 64 * 128])?;
                let xd: CudaSlice<f32> = e.alloc(128)?;
                match e.lowm_warmup(&w, &xd, &mut yd) {
                    Ok(_) => e.synchronize()?,
                    Err(err) => {
                        lowm_refused = true;
                        tracing::warn!(
                            "qwen4exp: low-M cluster warm-up refused ({err}) - lane off"
                        );
                        eprintln!("[q4x] low-M cluster warm-up refused ({err}) - lane off");
                    }
                }
            } else {
                lowm_refused = true;
                tracing::debug!(
                    "qwen4exp: low-M cluster arm absent (pack entry {}, k-quant dense planes {}) - arm off",
                    e.has_lowm(),
                    kq_lanes
                );
            }
            yd
        };
        Ok(Self {
            d_tok: e.alloc_u32(t)?,
            d_pos: e.alloc_u32(t)?,
            d_mrope: e.alloc_u32(4 * t)?,
            d_slots: e.alloc_u32(t)?,
            d_idx_qk: e.alloc(t * (c.idx_heads + c.idx_kv_heads) * c.idx_head_dim)?,
            d_idx_q: e.alloc(t * c.idx_heads * c.idx_head_dim)?,
            d_idx_stage: e.alloc(t * c.idx_head_dim)?,
            d_idx_spos: e.alloc_u32(4 * t)?,
            d_qsa_scores: e.alloc(qsa_rb * qsa_cap)?,
            d_qsa_sel: e.alloc_u32(t * qsa_k)?,
            d_qsa_cnt: e.alloc_u32(t)?,
            d_qsa_po: e.alloc(t * q_dim)?,
            d_qsa_pml: e.alloc(t * c.n_heads * 2)?,
            qsa_rb,
            d_x: e.alloc(t * h)?,
            d_h: e.alloc(t * hw)?,
            d_xn: e.alloc(t * hw)?,
            // + hc: the folded inject rows land in this plane's tail at batch 1
            d_m: e.alloc(t * c.hc_lowrank + c.hc_count)?,
            d_gate: e.alloc(t * hw)?,
            d_bi: e.alloc(t * h)?,
            d_inj: e.alloc(t * hc)?,
            d_mix: e.alloc(t * h)?,
            d_qkv: e.alloc(t * c.gdn_qkv_rows())?,
            d_zg: e.alloc(t * c.gdn_z_rows())?,
            d_ab: e.alloc(t * 2 * c.gdn_v_heads)?,
            d_g: e.alloc(t * c.gdn_v_heads)?,
            d_beta: e.alloc(t * c.gdn_v_heads)?,
            d_conv: e.alloc(t * c.gdn_qkv_rows())?,
            d_dq: e.alloc(t * kdim)?,
            d_dk: e.alloc(t * kdim)?,
            d_dv: e.alloc(t * vdim)?,
            d_dattn: e.alloc(t * vdim)?,
            d_core: e.alloc(t * vdim)?,
            d_qg: e.alloc(t * c.attn_q_rows())?,
            d_q: e.alloc(t * q_dim)?,
            d_agate: e.alloc(t * q_dim)?,
            d_k: e.alloc(t * kv_dim)?,
            d_v: e.alloc(t * kv_dim)?,
            d_qn: e.alloc(t * q_dim)?,
            d_kn: e.alloc(t * kv_dim)?,
            d_attn: e.alloc(t * q_dim)?,
            d_sinks: e.alloc_no_sinks(c.n_heads)?,
            // + 1 row: the folded shared-expert gate
            d_logits: e.alloc(t * (c.n_expert + 1))?,
            // slot 596's norms take 2 floats a (token, head), slot 604's
            // pre-pass 4 ({exp(g), beta, q norm, k norm})
            d_dnrn: e.alloc(t * c.gdn_v_heads * 4)?,
            // down z-split partials: [rows<=64][ceil(k/2)][embd]
            d_moe_part: e.alloc(64 * c.n_active.div_ceil(2) * c.hidden)?,
            // decode-only scratch: 64 rows x heads x S<=16 x (256+2) ~ 25 MB
            d_fmha_part: e.alloc(64 * c.n_heads * 16 * (256 + 2))?,
            // slot-544 contract: the low-M kernel's first cluster launch
            // must happen on a quiet context (cluster_fork_probe law) -
            // here, at model build, before any fork or capture exists.
            d_lowm_warm,
            lowm_refused,
            d_zero_bias: e.alloc(c.n_expert)?,
            // widest split-K out_dim this model routes is the router row set
            // sized by the RUNTIME split (env can raise it above the const;
            // the 0-const sized a zero-length scratch and split-8 panicked)
            d_skp: e.alloc((c.n_expert + 1) * (sk_split().max(2) as usize))?,
            d_skc: e.alloc_u32(c.n_expert + 1)?,
            d_idx: e.alloc_u32(t * c.n_active)?,
            d_xq: e.alloc_i8(if kq_lanes { t * h } else { 1 })?,
            d_xs: e.alloc(if kq_lanes { t * h / 32 } else { 1 })?,
            d_ssums: e.alloc(if kq_lanes {
                t * h.max(c.n_active * c.moe_ff) / 16
            } else {
                1
            })?,
            d_fq: e.alloc_i8(if kq_lanes {
                t * c.n_active * c.moe_ff
            } else {
                1
            })?,
            d_fs: e.alloc(if kq_lanes {
                t * c.n_active * c.moe_ff / 32
            } else {
                1
            })?,
            // moe_align CSR: max_blocks = ceil((rows + n_expert*(bm-1))/bm).
            // The block COUNT peaks at the smallest group (bm = 8), but the
            // sorted arrays carry bm entries per block and every expert's
            // padding grows with bm - so their LENGTH peaks at the largest
            // group the election picks (16, `kq_moe_group_for`). Sized at 8 x
            // 8 they were n_expert*8 entries short of a full-height 16-row
            // walk: a 4096-row walk overran them by 4096 entries, silently at
            // 128K context and as an illegal address at 262K (2026-09-24).
            d_msrow: e.alloc_u32(if kq_lanes {
                moe_sorted_capacity(t * c.n_active, c.n_expert)
            } else {
                1
            })?,
            d_msslot: e.alloc_u32(if kq_lanes {
                moe_sorted_capacity(t * c.n_active, c.n_expert)
            } else {
                1
            })?,
            d_mbexp: e.alloc_u32(if kq_lanes {
                grp_align_blocks(t * c.n_active, c.n_expert, 8)
            } else {
                1
            })?,
            d_dpart: e.alloc(if kq_lanes {
                t * c.n_active * MOE_DOWN_CHUNK
            } else {
                1
            })?,
            // W4A4 routed arm (slot 631), PREFILL widths only: the activation
            // pair and the pair's nvf4 intermediate. Sized on the nvf4 seat,
            // so a k-quant lane pays one byte each.
            d_xq4: e.alloc_i8(if nvf4_lane { t * c.hidden / 2 } else { 1 })?,
            d_xs4: e.alloc_u8(if nvf4_lane { t * c.hidden / 16 } else { 1 })?,
            d_nfq: e.alloc_u8(if nvf4_lane {
                grp_align_blocks(t * c.n_active, c.n_expert, 32) * 32 * (c.moe_ff / 2)
            } else {
                1
            })?,
            d_nfs: e.alloc_u8(if nvf4_lane {
                grp_align_blocks(t * c.n_active, c.n_expert, 32) * 32 * (c.moe_ff / 16)
            } else {
                1
            })?,
            d_nvf4_part: e.alloc(if nvf4_lane {
                t.min(NVF4_BS_MAX_ROWS) * (c.n_active + 1) * h
            } else {
                1
            })?,
            d_srow32: e.alloc_u32(if kq_lanes || nvf4_lane {
                grp_align_blocks(t * c.n_active, c.n_expert, 32) * 32
            } else {
                1
            })?,
            // the W4A4 pair reads the SAME bm=32 CSR, so all THREE planes
            // follow the same condition - changing only `d_srow32` left these
            // two at length 1 on the safetensors lane while moe_align wrote
            // nb*32 and nb entries into them (the second illegal address this
            // arm produced)
            d_sslot32: e.alloc_u32(if kq_lanes || nvf4_lane {
                grp_align_blocks(t * c.n_active, c.n_expert, 32) * 32
            } else {
                1
            })?,
            d_bexp32: e.alloc_u32(if kq_lanes || nvf4_lane {
                grp_align_blocks(t * c.n_active, c.n_expert, 32)
            } else {
                1
            })?,
            d_sfq: e.alloc_i8(if kq_lanes {
                grp_align_blocks(t * c.n_active, c.n_expert, 32) * 32 * c.moe_ff
            } else {
                1
            })?,
            d_sfs: e.alloc(if kq_lanes {
                grp_align_blocks(t * c.n_active, c.n_expert, 32) * 32 * (c.moe_ff / 32)
            } else {
                1
            })?,
            d_emap: e.alloc_u32(if kq_lanes { 2 * c.n_expert } else { 1 })?,
            d_hcaux: if kq_lanes && e.has_q4x_hc_rebuild() {
                (0..2 * c.n_layer)
                    .map(|_| e.alloc(c.hc_width() + t * c.hc_count))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            },
            d_injp: e.alloc(if kq_lanes && e.has_q4x_combine_norm_q8mmq_nsi() {
                t * c.hc_count * c.hc_count
            } else {
                1
            })?,
            d_hcq: e.alloc_u8(if kq_lanes && e.has_q4x_combine_norm_q8mmq() {
                super::hc_preq_bytes(c.hc_width(), t.min(super::HC_PREQ_MAX_ROWS))
            } else {
                1
            })?,
            d_topw: e.alloc(t * c.n_active)?,
            d_act: e.alloc(t * c.n_active * c.moe_ff)?,
            d_par: e.alloc_u32(t.max(1) * 4)?,
            d_tpar: e.alloc_u32(t.max(1) * 4)?,
            d_ids: e.alloc_u32(t.max(1))?,
            d_shg: e.alloc(t * c.shared_ff)?,
            d_shu: e.alloc(t * c.shared_ff)?,
            d_shd: e.alloc(t * h)?,
            d_shgate: e.alloc(t)?,
            d_emb: e.alloc(t * c.ple_embed)?,
            d_ple_ids: e.alloc_u32(t * c.ple_heads())?,
            d_run_off: e.alloc_u32(slots.max(1))?,
            d_run_len: e.alloc_u32(slots.max(1))?,
            d_run_slot: e.alloc_u32(slots.max(1))?,
            d_run_offs: e.alloc_u32(slots.max(1) + 1)?,
            d_tile_row0: e.alloc_u32(t / PD_APF_TQ + slots.max(1) + 1)?,
            d_tile_slot: e.alloc_u32(t / PD_APF_TQ + slots.max(1) + 1)?,
            d_pkey: e.alloc(t * hw)?,
            d_pval: e.alloc(t * h)?,
            d_gdn_ext_in: e.alloc(2 * (c.gdn_conv - 1) * c.gdn_qkv_rows())?,
            d_gdn_ext_out: e.alloc(2 * (c.gdn_conv - 1) * c.gdn_qkv_rows())?,
            d_ple_ext_in: e.alloc(2 * (c.ple_conv - 1) * PLE_DILATION * hw)?,
            d_ple_ext_out: e.alloc(2 * (c.ple_conv - 1) * PLE_DILATION * hw)?,
            d_pkn: e.alloc(t * hw)?,
            d_pqn: e.alloc(t * hw)?,
            d_pgv: e.alloc(t * hw)?,
            d_pconv: e.alloc(t * hw)?,
            // one row per SLOT: the batched tick emits a distribution per live
            // sequence. Sized by slots, not max_tokens - a 248320-wide vocab at
            // 4096 rows would be 4 GB.
            d_fin: e.alloc(slots * h)?,
            d_out: e.alloc(slots * c.vocab)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Generator: the serving seam.
//
// Until this existed, `qwen4_exp` appeared in `gpu_model/` and nowhere else -
// no reference from paddock-runner or paddock-api. That is why
// this lane had no board cell: every number it could report came from a bare
// forward loop, while every rival number is `aiperf` against an OpenAI-
// compatible server carrying HTTP, scheduling, sampling and detokenisation.
// The two are not the same measurement, so neither the c1 nor the c32 figure
// was ever comparable to the bar.
//
// The pack was already slot-aware (`pd_gated_delta_recurrent_slots` grids
// (n_heads, batch); `kv_append_batch` and the decode attention take slot
// vectors), and `decode_step_batch` is gated by
// `batched_slots_match_single_slot_runs`. So this is a seam, not a rewrite.
mod chunked;
mod mtp;
mod spec;

fn q4x_gen_err(e: GpuModelError) -> crate::generator::GenError {
    crate::generator::GenError::Backend(e.to_string())
}

use crate::generator::{RowSample, SampledStep};

impl crate::generator::Generator for Qwen4ExpGpu {
    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    /// The resident-weight line `/api/stats` publishes and the catalog's
    /// shape generator measures from (see `weights_mem_bytes` above).
    fn weights_mem_bytes(&self) -> Option<u64> {
        Self::weights_mem_bytes(self)
    }

    /// The live mempool counter every family publishes as its device ledger
    /// (`/api/stats` `model_mem`): without it the manager had nothing to
    /// reconcile this lane's NVML figure against.
    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn reply_pin(&mut self, slot: usize) {
        Qwen4ExpGpu::reply_pin(self, slot);
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        Qwen4ExpGpu::release_inactive_slots(self, occupied);
    }

    fn reset(&mut self) {
        // trait returns unit; a state-clear failure here would surface on the
        // next forward as a driver error rather than being swallowed silently
        if let Err(e) = Qwen4ExpGpu::reset(self) {
            tracing::warn!("qwen4exp reset: {e}");
        }
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, crate::generator::GenError> {
        self.decode_step(token).map_err(q4x_gen_err)
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    fn max_context(&self) -> usize {
        self.max_tokens
    }

    /// Slots are allocated at LOAD (the GDN recurrent state, both conv windows
    /// and every scratch plane are sized by them), so this reports what the
    /// instance already carries rather than allocating. `serving.rs` passes the
    /// serve width into `load_with_slots` for exactly this reason.
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, crate::generator::GenError> {
        Ok(self.slots.min(max_batch.max(1)).max(1))
    }

    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        self.prefill_slot(slot, tokens).map_err(q4x_gen_err)
    }

    /// The scheduler's whole admitted wave in one walk. The trait default
    /// prefills one prompt at a time, which at c32 is a 1.66 s blocking
    /// prefill tick against a 14.6 ms decode tick.
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, crate::generator::GenError> {
        if !super::prefill_wave_enabled() || !self.exec.has_gated_delta_recurrent_runs() {
            return items
                .iter()
                .map(|(s, t)| self.forward_prefill(*s, t))
                .collect();
        }
        // A wave wider than the walk's scratch is SPLIT, not abandoned: at
        // imax the prompts are 1024 tokens, so a 32-wide admission is 32768
        // rows against a 4096-row walk and the whole cell would otherwise fall
        // back to one prompt at a time. Four per sub-wave still fills the
        // recurrence grid four times over.
        let mut out = Vec::with_capacity(items.len());
        let mut lo = 0usize;
        while lo < items.len() {
            let mut hi = lo;
            let mut rows = 0usize;
            while hi < items.len() && (hi == lo || rows + items[hi].1.len() <= self.walk_rows) {
                rows += items[hi].1.len();
                hi += 1;
            }
            if rows > self.walk_rows {
                // a single prompt longer than the walk: the serial entry owns
                // that case (it is the one the chunked lane would take)
                let (s, t) = &items[lo];
                out.push(self.forward_prefill(*s, t)?);
            } else {
                out.extend(self.prefill_slots(&items[lo..hi]).map_err(q4x_gen_err)?);
            }
            lo = hi;
        }
        Ok(out)
    }

    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        self.forward_prompt(tokens).map_err(q4x_gen_err)
    }

    /// Row i drives slot i. The scheduler passes positions explicitly while
    /// this model tracks them per slot; they are CHECKED rather than trusted,
    /// so a desync fails loudly here instead of silently decoding at the wrong
    /// position (the failure mode a wrong-position KV read would give is
    /// plausible text, which no gate would catch).
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        if tokens.len() != positions.len() {
            return Err(crate::generator::GenError::Backend(format!(
                "forward_batch: {} tokens vs {} positions",
                tokens.len(),
                positions.len()
            )));
        }
        // The scheduler ticks its whole occupied PREFIX, not just the live
        // rows: a slot that finished (or is occupied but not yet prefilled)
        // rides along as a HOLE feeding (token 0, position 0), and the rows
        // are only meaningful where it is not. `forward_batch` has no plans
        // to read, so position 0 is the hole marker - a decode row is always
        // at position >= 1 because it follows a prompt.
        let rows: Vec<(usize, u32)> = tokens
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, _)| positions[i] != 0)
            .collect();
        Self::check_positions(&self.pos, &rows, positions)?;
        let per_row = self.decode_step_batch(&rows).map_err(q4x_gen_err)?;
        self.mtp_flush().map_err(q4x_gen_err)?;
        // hand back a full [rows, vocab] plane: hole rows keep their zeros,
        // which is what the caller's own Hole arm discards
        let mut out = vec![0f32; tokens.len() * self.cfg.vocab];
        for ((i, _), row) in rows.iter().copied().zip(per_row) {
            out[i * self.cfg.vocab..(i + 1) * self.cfg.vocab].copy_from_slice(&row);
        }
        Ok(out)
    }

    /// The service checks this before drawing per-row uniforms, so answering
    /// truthfully is what keeps a slot's seed stream from paying for a path
    /// that will not run.
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.prefix.as_mut().map_or(0, |p| p.take_reused(slot))
    }
    fn supports_device_sampling(&self) -> bool {
        self.exec.has_sample_rows()
    }

    fn supports_device_trunc(&self) -> bool {
        self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    /// Device-sampled decode tick: the walk lands `[rows, vocab]` in `d_out`,
    /// the sampler reduces it on device, and only `Host`-plan rows read a
    /// vocab row back. This is the method whose absence made the first serving
    /// measurement 27.6 ms/tok against the engine's 7.9 - without it the
    /// service reads 0.99 MB of logits per token at c1 (31.8 MB/step at c32)
    /// and samples on the host.
    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, crate::generator::GenError> {
        if tokens.len() != positions.len() || plans.len() != tokens.len() {
            return Err(crate::generator::GenError::Backend(format!(
                "forward_batch_sampled: {} tokens, {} positions, {} plans",
                tokens.len(),
                positions.len(),
                plans.len()
            )));
        }
        // Hole rows are the scheduler's own convention for a slot inside the
        // occupied prefix that must not decode this tick (finished, or
        // occupied with no KV behind it yet) and they feed (0, 0). Ticking
        // them anyway is what a first cut did, and the position check then
        // failed the whole tick: 269 requests died as "slot 0: scheduler says
        // position 0, model is at 309" across the first serve ladder.
        let rows: Vec<(usize, u32)> = tokens
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, _)| !matches!(plans[i], RowSample::Hole))
            .collect();
        Self::check_positions(&self.pos, &rows, positions)?;
        if rows.is_empty() {
            return Ok(SampledStep {
                ids: vec![0u32; tokens.len()],
                host_rows: Vec::new(),
            });
        }
        self.decode_batch_walk(&rows).map_err(q4x_gen_err)?;
        let step = self
            .sample_rows_from_logits(&rows, plans)
            .map_err(q4x_gen_err)?;
        // after the sample: feeding the head reuses the walk's scratch
        self.mtp_flush().map_err(q4x_gen_err)?;
        Ok(step)
    }

    // ---- chunked prefill: the mixed tick (forward/chunked.rs) ----------

    fn supports_chunked_prefill(&self) -> bool {
        self.chunked_supported()
    }

    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), crate::generator::GenError> {
        self.prefill_begin_impl(slot, tokens).map_err(q4x_gen_err)
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.prefill_queue_impl()
    }

    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.prefill_tick_cap_impl(decode_rows)
    }

    /// The mixed tick with host sampling: the decode rows' `[nd, vocab]`
    /// logits in `decodes` order, and each finished prompt's last logits.
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), crate::generator::GenError> {
        let w = self.mixed_walk(decodes, budget).map_err(q4x_gen_err)?;
        let logits = if w.nd > 0 {
            self.exec
                .to_host_len(&self.sc.d_out, w.nd * self.cfg.vocab)
                .map_err(|e| q4x_gen_err(e.into()))?
        } else {
            Vec::new()
        };
        let fins: Vec<(usize, usize, usize)> = w.finishers(self).collect();
        let mut finished = Vec::with_capacity(fins.len());
        for (slot, row, rows) in fins {
            finished.push((slot, self.out_row(row).map_err(q4x_gen_err)?, rows));
        }
        self.mixed_commit(w).map_err(q4x_gen_err)?;
        Ok((logits, finished))
    }

    /// The mixed tick, decode rows sampled on device (plans in `decodes`
    /// order); a finishing prompt hands back its last logits.
    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<
        (
            SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        crate::generator::GenError,
    > {
        if plans.len() != decodes.len() {
            return Err(crate::generator::GenError::Backend(format!(
                "forward_mixed_sampled: {} decode rows, {} plans",
                decodes.len(),
                plans.len()
            )));
        }
        let w = self.mixed_walk(decodes, budget).map_err(q4x_gen_err)?;
        let (ids, host_rows) = self.sample_out_rows(&plans[..w.nd]).map_err(q4x_gen_err)?;
        let fins: Vec<(usize, usize, usize)> = w.finishers(self).collect();
        let mut finished = Vec::with_capacity(fins.len());
        for (slot, row, rows) in fins {
            let logits = self.out_row(row).map_err(q4x_gen_err)?;
            finished.push((slot, crate::generator::FinishSample::Logits(logits), rows));
        }
        // after the reads: seeding the drafter reuses the walk's scratch
        self.mixed_commit(w).map_err(q4x_gen_err)?;
        Ok((SampledStep { ids, host_rows }, finished))
    }

    /// A drafter exists once a head GGUF is attached (`attach_mtp`); the
    /// service's policy resolution handles --spec / --no-spec on top.
    fn spec_capable(&self) -> bool {
        self.mtp.is_some() && self.exec.has_argmax_rows()
    }

    /// Per-tick: keep feeding the draft head only while speculation is the
    /// plan for this width. Stashing every decode row and draining it through
    /// a head pass is most of the cost of having a drafter attached at all -
    /// at 8 live slots it was 19 ms of a 45 ms tick that no round consumed.
    fn spec_fuse_hint(&mut self, on: bool) {
        if let Some(m) = self.mtp.as_mut() {
            m.feed = on;
        }
    }

    /// Same question for the eager prefill seed.
    fn spec_warm_hint(&mut self, on: bool) {
        if let Some(m) = self.mtp.as_mut() {
            m.warm_prefills = on;
        }
    }

    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, crate::generator::GenError> {
        self.verify_round(reqs).map_err(q4x_gen_err)
    }

    /// The sampled twin: every verify row drawn on device from its own plan.
    /// Without it a temperature > 0 serve never speculated at all - see
    /// `verify_round_plans` for the exactness argument and what it measured.
    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, crate::generator::GenError> {
        self.verify_round_plans(reqs, plans).map_err(q4x_gen_err)
    }

    /// The host-sampled round - the one constrained slots, i.e. every
    /// tool-carrying request, speculate through: raw row logits out, the
    /// service samples them through the slot's sampler and grammar, then
    /// `spec_commit` keeps what it accepted (see `verify_open`).
    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, crate::generator::GenError> {
        self.verify_open(reqs).map_err(q4x_gen_err)
    }

    fn spec_commit(&mut self, committed: &[u32]) -> Result<(), crate::generator::GenError> {
        self.verify_close(committed).map_err(q4x_gen_err)
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, crate::generator::GenError> {
        if self.mtp.is_none() {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(pendings.len());
        for &(slot, tok) in pendings {
            out.push(self.mtp_draft(slot, tok, k).map_err(q4x_gen_err)?);
        }
        Ok(Some(out))
    }

    /// No re-warm path: every walk feeds the head's stash, so a cold slot is
    /// one whose rows outran what it tracks - it drafts nothing until its next
    /// prompt, and the verify still serves it one token a round.
    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, crate::generator::GenError> {
        Ok(self.mtp_warm(slot, want_pos as usize + 1))
    }

    /// Depth 3, elected in serving - the row budget asks for 7 at one live
    /// slot and this family cannot repay it.
    ///
    /// Every extra verify row costs a full per-row pass over the routed
    /// experts it touches, while the tokens it returns grow only with
    /// acceptance. GB10, code prompt, greedy, MTP head attached, decode-only
    /// tok/s with TTFT excluded (2026-09-19):
    ///
    ///   depth 1   29.95 median of 5   (acceptance 90.2%)
    ///   depth 3   33.95 median of 3   (acceptance 81.2%)   ELECTED
    ///
    /// Depth 7 - what the row budget asks for at one live slot - was capped
    /// here without ever being measured until 2026-09-20, when the line above
    /// said only that this family "cannot repay it". It does not: on
    /// syn_128x128_c1, 3 reps each, depth 3 medians 30.45 (30.45/32.12/30.01)
    /// against depth 7's 26.70 (26.70/27.23/23.73), -12.3%, losing every rep.
    /// A/B it with `PADDOCK_Q38FN_SPEC_DEPTH`.
    ///
    /// Why it loses is worth keeping, because the original reasoning here was
    /// wrong in a way that invites re-litigation. It used to say each extra
    /// row "re-streams" its experts; it does not - the rows of a verify batch
    /// share experts ~10x more than chance (26.3 distinct of 35.5 rows,
    /// measured in serving) and L2 banks ~41% of that for free, so the bytes
    /// are largely not re-read. The cost is per-row unpack on the i-quant
    /// seat, which no routing trick removes (priced on the seat formats,
    /// 2026-09-20), and on top of it the dense side is not flat in n either:
    /// GDN's per-token recurrence, attention over the wider batch and the K
    /// sequential MTP draft passes all grow with depth. Two terms growing
    /// against an acceptance curve that decays is why deeper does not pay.
    ///
    /// Elect this in serving, not in the harness. `examples/q38fn_spec_bench`
    /// prices the same depths at 31.60 (d1) against 28.09 (d3) - it inverts
    /// the ordering, and an earlier cut of this election took its answer and
    /// shipped depth 1. The harness is a bare loop over draft + verify and
    /// says so itself ("its numbers are never a serving cell"); a serving
    /// round also carries the state save, the rollback bookkeeping, the accept
    /// walk and the stream, and those are per-ROUND costs that a deeper draft
    /// amortizes over more committed tokens. Acceptance moves the other way
    /// (a single draft is easier to hit) and is not the thing to maximize -
    /// tokens per round against the round's real cost is.
    ///
    /// Re-measure in serving when the verify row's cost moves (the
    /// routed-expert seat, the expert byte count, the walk's kernel election).
    fn spec_depth_cap(&self) -> Option<usize> {
        Some(super::spec_depth_override().unwrap_or(3))
    }

    /// The drafter decides warmth per slot (a cold slot gets an empty draft
    /// list and rides the verify as a one-row chunk).
    fn spec_draft_per_slot_warm(&self) -> bool {
        self.mtp.is_some()
    }

    /// Canonical rejection sampling: armed only when the head's RS buffers
    /// exist, which needs PADDOCK_SPEC_RS *and* a pack carrying both kernels
    /// (`mtp::RsBufs`). Answering truthfully is load-bearing - the service
    /// stashes chain draws exactly when this is true, and a backend that
    /// claimed the arm without the buffers would take sampled rounds it cannot
    /// resolve and emit the wrong distribution.
    fn supports_spec_rs(&self) -> bool {
        self.mtp.as_ref().is_some_and(|m| m.rs.is_some())
    }

    fn spec_rs_stash(&mut self, draws: Vec<crate::generator::SpecRsDraw>) {
        self.spec_rs_draws = Some(draws);
    }
}

#[cfg(test)]
mod tests {
    use super::{grp_align_blocks, moe_sorted_capacity};

    /// The sorted arrays hold every group size the election can pick, at
    /// every walk height - and the old 8 x 8 sizing did not (the 4096-row
    /// walk that overran it on the Flash-Next lane, 10 routed of 512).
    #[test]
    fn moe_sorted_capacity_covers_every_group() {
        let (k, e) = (10usize, 512usize);
        for rows in [1usize, 7, 64, 511, 1024, 2048, 4095, 4096, 8192] {
            let pairs = rows * k;
            for bm in [8usize, 16] {
                let need = grp_align_blocks(pairs, e, bm) * bm;
                assert!(
                    moe_sorted_capacity(pairs, e) >= need,
                    "{rows} rows at bm {bm}"
                );
            }
        }
        let pairs = 4096 * k;
        assert!(grp_align_blocks(pairs, e, 8) * 8 < grp_align_blocks(pairs, e, 16) * 16);
    }
}
