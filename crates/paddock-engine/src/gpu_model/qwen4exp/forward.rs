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
use crate::gpu_model::st_load::bf16_bytes;
use paddock_kernels::reference::qwen4exp as rq;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;
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

/// PLE conv dilation - a k=4 kernel over a 9-token receptive ring.
const PLE_DILATION: usize = 3;

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
struct Dump(Option<std::path::PathBuf>);

impl Dump {
    fn arm() -> Self {
        Self(std::env::var_os("PADDOCK_Q38FN_DUMP").map(|d| {
            let p = std::path::PathBuf::from(d);
            let _ = std::fs::create_dir_all(&p);
            p
        }))
    }
    fn on(&self) -> bool {
        self.0.is_some()
    }
    /// Write host-side values already read back.
    fn put_host(&self, li: usize, tag: &str, v: &[f32]) -> Result<(), GpuModelError> {
        let Some(dir) = &self.0 else { return Ok(()) };
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
        let Some(dir) = &self.0 else { return Ok(()) };
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
    st: PleSource,
    layers: Vec<Qwen4ExpLayer>,
    embed: Embed,
    lm_head: DensePlane,
    final_mix: HcW,
    max_tokens: usize,
    sc: Scratch,
    /// per-layer GDN recurrent state `[v_heads][k_dim][v_dim]`, None on attn layers
    recur: Vec<Option<CudaSlice<f32>>>,
    /// per-layer KV caches `[max_tokens, kv_dim]`, None on GDN layers
    kv_k: Vec<Option<CudaSlice<u8>>>,
    kv_v: Vec<Option<CudaSlice<u8>>>,
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
    decode_graph: Option<Q4xSendGraph>,
    /// one captured batched tick per WIDTH. Valid only for the dense slot set
    /// `0..n`: the PLE window advance bakes per-slot copy offsets into the
    /// graph, so a different slot set must not replay it.
    batch_graphs: Vec<Option<Q4xSendGraph>>,
    /// staging the 8-bit classes need on their batch > 1 arm
    stage: DenseStage,
    /// Whether decode ticks may be captured at all. Defaults to the env gate;
    /// `set_graph_capture` lets one process A/B the two paths, which is how
    /// the capture gate proves the graph and the eager walk agree.
    graph_capture: bool,
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
    /// Identity block table for the tcgen05 decode arm (slot 431). The dense
    /// slot-major KV cache is a degenerate paged pool: rows are contiguous, so
    /// block `i` of slot `s` sits at pool block `s*(max_ctx/16)+i`, and the
    /// table is literally `0..slots*(max_ctx/16)`. Built once at load; the
    /// kernel's TMA fetch does `y = table[..]*16` into the flat row pool.
    d_blk_tab: CudaSlice<u32>,
    d_tok: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_mrope: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    d_x: CudaSlice<f32>,
    d_h: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_m: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_bi: CudaSlice<f32>,
    d_inj: CudaSlice<f32>,
    d_mix: CudaSlice<f32>,
    /// a wave's partial routed output on the expert-major prefill path
    d_mix_wave: CudaSlice<f32>,
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
    /// Blackwell answers cudaErrorNotSupported): the lane stays OFF, whatever
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
                let mut ple = load_ple_projections(exec, &st, &cfg, li)?;
                // The 51.2 GB n-gram table goes to the device unless it will
                // not fit (or the host lane is forced). The host mmap gather
                // it replaces is a uniform random read at 160 useful bytes
                // per 4 KB page over a 51.2 GB file: it put prefill ticks of
                // 891-48697 ms and a c8 TTFT p50 of 7858 ms on the serve
                // ladder. vLLM has always kept this table device-resident
                // (`NgramEmbedding.oe_embedder`, a `VocabParallelEmbedding`).
                if ple_device_table(exec, &cfg) {
                    match load_ple_table(exec, &st, &cfg, li, &mut ple) {
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
                        }
                    }
                }
                layer.ple = Some(ple);
            }
            layers.push(layer);
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
        let max_rows = self.max_tokens * self.cfg.n_active;
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
                ExpertSeats::Nvf4 { .. } => 0,
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
        let (recur, kv_k, kv_v) = alloc_state(exec, &cfg, max_tokens, slots)?;
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
        let sc = Scratch::new(exec, &cfg, max_tokens, slots, kq_lanes)?;
        // the widest activation any dense plane reads is the 4-stream state
        let stage = DenseStage {
            q: exec.alloc_i8(max_tokens * cfg.hc_width())?,
            rs: exec.alloc(max_tokens)?,
            xs: exec.alloc(if kq_lanes {
                max_tokens * cfg.hc_width() / 32
            } else {
                1
            })?,
            ssums: exec.alloc(if kq_lanes {
                max_tokens * cfg.hc_width() / 16
            } else {
                1
            })?,
            // the widest activation any dense plane reads is the 4-stream state
            x16: exec.alloc_f16(max_tokens * cfg.hc_width())?,
            xb16: exec.stream_alloc_bf16(max_tokens * cfg.hc_width())?,
            // the low-M arm runs at batch <= 8 only, so this is sized by the
            // widest plane (q at 12288) and not by the prefill width
            f16_ok: false,
            lowm_ok: false,
            f16_max: usize::MAX,
        };
        Ok(Self {
            exec: exec.clone(),
            cfg,
            st,
            layers,
            embed,
            lm_head,
            final_mix,
            max_tokens,
            sc,
            recur,
            kv_k,
            kv_v,
            gdn_win,
            ple_win,
            slots,
            cur_slots: vec![0usize],
            cur_runs: Vec::new(),
            pos: vec![0usize; slots],
            stream: vec![Vec::new(); slots],
            batch_graphs: (0..=slots).map(|_| None).collect(),
            stage,
            decode_graph: None,
            graph_capture: capture_wanted(),
        })
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
        if self.decode_graph.is_none() && self.graph_capture {
            self.capture_decode_tick()?;
        }
        let t_launch = std::time::Instant::now();
        match self.decode_graph.as_ref() {
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
        Ok(())
    }

    /// Prefill walk for one slot: `n` tokens of one sequence at positions
    /// `0..n`, all rows carrying that slot id.
    fn walk_slot(&mut self, slot: usize, ids: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let n = ids.len();
        let pos: Vec<u32> = (0..n as u32).collect();
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        let slots: Vec<u32> = vec![slot as u32; n];
        self.exec.upload_u32(ids, &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&slots, &mut self.sc.d_slots)?;
        for li in 0..self.cfg.n_layer {
            if let Some(ple) = self.layers[li].ple.as_ref() {
                match ple.table.as_ref() {
                    Some(tab) => {
                        let ids = ple_row_ids(&self.cfg, ple, &self.stream[slot], 2, n)?;
                        stage_ple_device(&self.exec, &self.cfg, ple, tab, &ids, &mut self.sc)?;
                    }
                    None => {
                        let emb = gather_ple_rows(
                            &self.st,
                            &self.cfg,
                            ple,
                            li,
                            &self.stream[slot],
                            2,
                            n,
                        )?;
                        self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
                    }
                }
            }
        }
        self.device_walk(n, Phase::Prefill)?;
        Ok(self.exec.to_host_len(&self.sc.d_out, self.cfg.vocab)?)
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
        if rows.len() > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "{} rows; scratch is sized for {}",
                rows.len(),
                self.max_tokens
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
        if self.graph_capture && self.batch_graphs[n].is_none() {
            self.capture_batch_tick(n)?;
        }
        let t_launch = std::time::Instant::now();
        match self.batch_graphs[n].as_ref() {
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
        if rows.len() > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "{} rows; scratch is sized for {}",
                rows.len(),
                self.max_tokens
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
        if self.graph_capture && self.batch_graphs[n].is_none() {
            self.capture_batch_tick(n)?;
        }
        match self.batch_graphs[n].as_ref() {
            Some(g) => g
                .launch()
                .map_err(|e| crate::gpu::GpuError::Driver(format!("batched graph replay: {e}")))?,
            None => self.device_walk(n, Phase::DecodeBatch)?,
        }
        for &(sl, _) in rows {
            self.pos[sl] += 1;
        }
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
        let vocab = self.cfg.vocab;
        let exec = self.exec.clone();
        // `d_out` holds one row per DECODED row, in `rows` order - hole rows
        // never reached the walk. Plans are indexed by the scheduler's slot,
        // so they are compacted the same way before packing and the ids are
        // scattered back at the end.
        let r = rows.len();
        let live: Vec<RowSample> = rows.iter().map(|&(i, _)| plans[i]).collect();
        let (par, tpar) = Self::pack_samp_par(&live);
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
        let mut ids = vec![0u32; plans.len()];
        let mut host_rows = Vec::new();
        for (j, &(i, _)) in rows.iter().enumerate() {
            ids[i] = packed[j];
            if matches!(plans[i], RowSample::Host) {
                let v = self
                    .sc
                    .d_out
                    .try_slice(j * vocab..(j + 1) * vocab)
                    .ok_or_else(|| crate::gpu::GpuError::Driver("host row slice".into()))?;
                host_rows.push((
                    i,
                    exec.stream
                        .clone_dtoh(&v)
                        .map_err(crate::gpu::from_driver)?,
                ));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Stage one token per row, each against its own slot and position.
    fn stage_inputs_rows(&mut self, rows: &[(usize, u32)]) -> Result<(), GpuModelError> {
        let n = rows.len();
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
        if n > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "prefill wave of {n} rows; the walk is sized for {}",
                self.max_tokens
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

        // fresh state per run: the runs walk's conv arms rely on the window
        // being zero at their offset base, which is what makes the offset
        // entry equal to a zero left-pad
        let mut runs = Vec::with_capacity(items.len());
        let mut off = 0usize;
        for (slot, toks) in items {
            self.reset_slot(*slot)?;
            self.stream[*slot] = vec![self.cfg.bos_id as i64; 2];
            self.stream[*slot].extend(toks.iter().map(|&i| i as i64));
            runs.push(Run {
                slot: *slot,
                off,
                len: toks.len(),
            });
            off += toks.len();
        }
        self.cur_runs = runs.clone();
        self.cur_slots = runs
            .iter()
            .flat_map(|r| std::iter::repeat_n(r.slot, r.len))
            .collect();
        self.stage_inputs_runs(items, &runs)?;
        let walked = self.device_walk(n, Phase::PrefillRuns);
        self.cur_runs.clear();
        walked?;
        for (slot, toks) in items {
            self.pos[*slot] = toks.len();
        }
        let all = self
            .exec
            .to_host_len(&self.sc.d_out, items.len() * self.cfg.vocab)?;
        Ok(all.chunks(self.cfg.vocab).map(|c| c.to_vec()).collect())
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
        let mut pos = Vec::with_capacity(n);
        let mut slots = Vec::with_capacity(n);
        for ((_, toks), r) in items.iter().zip(runs) {
            ids.extend_from_slice(toks);
            pos.extend(0..r.len as u32);
            slots.extend(std::iter::repeat_n(r.slot as u32, r.len));
        }
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        self.exec.upload_u32(&ids, &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&slots, &mut self.sc.d_slots)?;
        let mut t_row0: Vec<u32> = Vec::with_capacity(n_qtiles(runs));
        let mut t_slot: Vec<u32> = Vec::with_capacity(n_qtiles(runs));
        for r in runs {
            let mut row = r.off;
            while row < r.off + r.len {
                t_row0.push(row as u32);
                t_slot.push(r.slot as u32);
                row += PD_APF_TQ;
            }
        }
        self.exec.upload_u32(&t_row0, &mut self.sc.d_tile_row0)?;
        self.exec.upload_u32(&t_slot, &mut self.sc.d_tile_slot)?;
        let roff: Vec<u32> = runs.iter().map(|r| r.off as u32).collect();
        let rlen: Vec<u32> = runs.iter().map(|r| r.len as u32).collect();
        let rslot: Vec<u32> = runs.iter().map(|r| r.slot as u32).collect();
        self.exec.upload_u32(&roff, &mut self.sc.d_run_off)?;
        self.exec.upload_u32(&rlen, &mut self.sc.d_run_len)?;
        self.exec.upload_u32(&rslot, &mut self.sc.d_run_slot)?;
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
                                2,
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
                                2,
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
        self.reset_slot(slot)?;
        self.stream[slot] = vec![self.cfg.bos_id as i64; 2];
        self.stream[slot].extend(ids.iter().map(|&i| i as i64));
        // prefill walks one sequence; stage it against this slot
        self.cur_slots = vec![slot; n];
        let logits = self.walk_slot(slot, ids)?;
        self.pos[slot] = n;
        Ok(logits)
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
        self.decode_graph = graph?.map(Q4xSendGraph);
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
        self.batch_graphs[n] = graph?.map(Q4xSendGraph);
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
            self.decode_graph = None;
        }
    }

    /// Whether the decode tick is currently running as a captured graph.
    pub fn graph_active(&self) -> bool {
        self.decode_graph.is_some()
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

    /// How many sequences this instance is sized for.
    pub fn slot_count(&self) -> usize {
        self.slots
    }

    /// Drop every per-sequence state. The allocations stay - a later capture
    /// rung bakes these addresses.
    fn reset(&mut self) -> Result<(), GpuModelError> {
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
            gdn_win,
            ple_win,
            stage,
            cur_slots,
            cur_runs,
            ..
        } = self;
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
        let dump = Dump::arm();

        // ---- embed -> the 4-stream hyper-connection state -----------------
        match embed {
            Embed::Bf16(t) => e.embed_gather_bf16(t, &sc.d_tok, &mut sc.d_x, h, n, 1.0)?,
            Embed::Kq(t) => e.kquant_gather(t, &sc.d_tok, &mut sc.d_x, h, n)?,
        }
        for t in 0..n {
            for s in 0..hc {
                e.copy_region(&sc.d_x, t * h, &mut sc.d_h, t * hw + s * h, h)?;
            }
        }
        dump.put(e, usize::MAX, "h_embed", &sc.d_h, n * hw)?;

        // Carries whether the previous combine already left this mix's
        // normalized state in d_xn. False at entry: the first mix reads the
        // freshly broadcast embedding, which no combine produced.
        let mut pre_normed = false;
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
                    cur_runs,
                )?;
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

            let attn_inj = hc_mix_pass(e, c, &layer.attn_hc, sc, stage, n, pre_normed)?;
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
                        cur_runs,
                        fork_ok,
                    )?;
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
                        *max_tokens,
                        n,
                        phase,
                        cur_runs,
                        fork_ok,
                    )?;
                }
            }
            dump.put(e, li, "mix_out", &sc.d_mix, n * h)?;
            // the mlp mix always reads this combine's own output
            let mlp_pre = combine(
                e,
                sc,
                attn_inj,
                Some(&layer.mlp_hc.norm),
                n,
                hc,
                h,
                eps,
                stage,
            )?;
            dump.put(e, li, "h_mid", &sc.d_h, n * hw)?;

            let mlp_inj = hc_mix_pass(e, c, &layer.mlp_hc, sc, stage, n, mlp_pre)?;
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
            )?;
            dump.put(e, li, "moe_out", &sc.d_mix, n * h)?;
            pre_normed = combine(e, sc, mlp_inj, next_norm, n, hc, h, eps, stage)?;
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
        if matches!(phase, Phase::DecodeBatch) {
            // every row is a live sequence's next-token distribution
            lm_head.matmul(e, &sc.d_bi, &mut sc.d_out, n, stage)?;
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
        dump.put(e, usize::MAX, "logits", &sc.d_out, c.vocab)?;
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
    n: usize,
    hc: usize,
    hidden: usize,
    eps: f32,
    _stage: &mut DenseStage,
) -> Result<bool, GpuModelError> {
    // xn's bf16 MIRROR: this kernel is the only writer of d_xn on the decode
    // path, so mirroring at the store retires the per-consumer cast (the hc
    // down plane's TGV arm cast a [n, 10240] plane every layer). Any path
    // below that does not mirror invalidates, since d_xn is one buffer.
    // d_h, d_mix, d_m/d_inj and d_xn are disjoint fields, so the borrows below
    // are all simultaneous without any shuffling.
    match (inj, next_norm) {
        (Inj::InM(off), Some(nw)) => {
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
            Ok(true)
        }
        (Inj::Separate, Some(nw)) => {
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
            Ok(true)
        }
        (Inj::InM(off), None) => {
            e.q4x_hc_combine_at(&mut sc.d_h, &sc.d_mix, &sc.d_m, off, n, hc, hidden)?;
            Ok(false)
        }
        (Inj::Separate, None) => {
            e.q4x_hc_combine(&mut sc.d_h, &sc.d_mix, &sc.d_inj, n, hc, hidden)?;
            Ok(false)
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
        Inj::Separate
    } else {
        w.down.matmul(e, &sc.d_xn, &mut sc.d_m, n, stage)?;
        let wi = w.inject.as_ref().expect("unfolded block hc carries inject");
        e.matvec_f32_raw(&wi.buf, hc * h, hc, &sc.d_xn, &mut sc.d_inj, n)?;
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
    let fused = match (&w.up_hcmix, (2..=32).contains(&n)) {
        (Some(wp), true) => e.bf16_hcmix_gemm(wp, &sc.d_m, &sc.d_xn, &mut sc.d_bi, lr, h, hc, n)?,
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
        if !upmix {
            w.up.matmul(e, &sc.d_m, &mut sc.d_gate, n, stage)?;
            e.q4x_hc_mix(&sc.d_xn, &sc.d_gate, &mut sc.d_bi, None, n, hc, h)?;
        }
    }
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
            if n >= wrows {
                let r0 = n % wrows; // ring row of source row (n - wrows)
                let head = wrows - r0; // rows before the wrap
                e.copy_region(&sc.d_pkn, (n - wrows) * hw, win, pbase + r0 * hw, head * hw)?;
                if r0 > 0 {
                    e.copy_region(&sc.d_pkn, (n - r0) * hw, win, pbase, r0 * hw)?;
                }
            } else {
                e.copy_region(&sc.d_pkn, 0, win, pbase, n * hw)?;
            }
        }
        Phase::PrefillRuns => {
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
                if r.len >= wrows {
                    let r0 = r.len % wrows;
                    let head = wrows - r0;
                    e.copy_region(
                        &sc.d_pkn,
                        (r.off + r.len - wrows) * hw,
                        win,
                        pbase + r0 * hw,
                        head * hw,
                    )?;
                    if r0 > 0 {
                        e.copy_region(&sc.d_pkn, (r.off + r.len - r0) * hw, win, pbase, r0 * hw)?;
                    }
                } else {
                    e.copy_region(&sc.d_pkn, r.off * hw, win, pbase, r.len * hw)?;
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
fn ple_device_table(exec: &Arc<GpuExecutor>, c: &Qwen4ExpConfig) -> bool {
    if std::env::var("PADDOCK_Q4X_PLE_HOST").is_ok_and(|v| v != "0") {
        eprintln!("[q4x-ple] HOST lane (PADDOCK_Q4X_PLE_HOST)");
        return false;
    }
    if !exec.has_q4x_ple_gather() {
        eprintln!("[q4x-ple] HOST lane: pack has no q4x_ple_gather (slot 532)");
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
    let mut out = vec![0f32; n * c.ple_embed];
    for t in 0..n {
        let w3 = rq::ple_window(stream, first + t, eos);
        let row_ids = rq::ple_ngram_ids(
            &w3,
            &ple.multipliers,
            &ple.head_vocab,
            &ple.head_offset,
            c.heads_per_ngram,
        );
        for (hh, &rid) in row_ids.iter().enumerate() {
            let rid = rid as usize;
            let (sh, local) = (rid / rows_per_shard, rid % rows_per_shard);
            let name = format!("{emb_p}.ngram_embedding.shard_{sh}.weight");
            let (tinfo, sb) = st
                .bytes(&name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
            if tinfo.dtype != StDtype::F8E4m3 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: dtype {:?}, want F8E4m3",
                    tinfo.dtype
                )));
            }
            let row = &sb[local * width..(local + 1) * width];
            let dst = t * c.ple_embed + hh * width;
            for (i, &byte) in row.iter().enumerate() {
                out[dst + i] = rq::e4m3_to_f32(byte) * ple.table_scale;
            }
        }
    }
    Ok(out)
}

/// Gated DeltaNet mixer. Writes `d_mix` `[n, hidden]` and advances `state`.
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
    let width = c.ple_embed / c.ple_heads();
    let (_, table) = map
        .tensor_bytes(name)
        .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}")))?;
    let rows = table.len() / row_bytes;
    let eos = c.bos_id as i64;
    let mut out = vec![0f32; n * c.ple_embed];
    for t in 0..n {
        let w3 = rq::ple_window(stream, first + t, eos);
        let row_ids = rq::ple_ngram_ids(
            &w3,
            &ple.multipliers,
            &ple.head_vocab,
            &ple.head_offset,
            c.heads_per_ngram,
        );
        for (hh, &rid) in row_ids.iter().enumerate() {
            let rid = rid as usize;
            if rid >= rows {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: n-gram row {rid} past the table's {rows} rows"
                )));
            }
            let row = &table[rid * row_bytes..(rid + 1) * row_bytes];
            let dst = t * c.ple_embed + hh * width;
            ple_row_dequant(ty, row, &mut out[dst..dst + width]);
        }
    }
    Ok(out)
}

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
    fork_ok: bool,
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
    if !fused_zq {
        w.qkv.matmul(e, &sc.d_bi, &mut sc.d_qkv, n, stage)?;
    }
    let mut split_done = false;
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
            if n >= km1 {
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
        }
        Phase::PrefillRuns => {
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
                if r.len >= km1 {
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
    if matches!(phase, Phase::PrefillRuns) {
        // Every run's whole sequence in one launch, grid (n_heads, n_runs).
        // The single-run entry grids 48 blocks at 255 registers - 32% of a
        // 148-SM die - and a serially-prefilled wave pays 195.9 us per layer
        // per prompt for it (7.05 ms of a 33.9 ms 128-token prefill).
        let pn = super::dn_prenorm_on()
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
        if !pn {
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
    if !gn_done {
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
    w.out.matmul(e, &sc.d_core, &mut sc.d_mix, n, stage)?;
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
    max_ctx: usize,
    n: usize,
    phase: Phase,
    runs: &[Run],
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
        e.kv_append_batch(
            &sc.d_kn,
            kc,
            &sc.d_pos,
            Some(&sc.d_slots),
            kv_dim,
            max_ctx,
            n,
            KV(),
        )?;
        e.kv_append_batch(
            &sc.d_v,
            vc,
            &sc.d_pos,
            Some(&sc.d_slots),
            kv_dim,
            max_ctx,
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
        if !fused_qkv {
            w.q.matmul(e, &sc.d_bi, &mut sc.d_qg, n, stage)?;
        }
        e.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_agate, n, nh, hd)?;
        if !fused_qkv {
            w.k.matmul(e, &sc.d_bi, &mut sc.d_k, n, stage)?;
            w.v.matmul(e, &sc.d_bi, &mut sc.d_v, n, stage)?;
        }
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
        e.kv_append_batch(
            &sc.d_kn,
            kc,
            &sc.d_pos,
            Some(&sc.d_slots),
            kv_dim,
            max_ctx,
            n,
            KV(),
        )?;
        e.kv_append_batch(
            &sc.d_v,
            vc,
            &sc.d_pos,
            Some(&sc.d_slots),
            kv_dim,
            max_ctx,
            n,
            KV(),
        )?;
    }
    let scale = 1.0 / (hd as f32).sqrt();
    // tcgen05 decode attention (pack slot 431, the <256,6> instantiation built
    // for qwen3.8's 24q/4kv/hd256 - the same geometry). Needs e4m3 pools
    // (PADDOCK_Q38FN_KV8=1): its TMA maps assume 1-byte elements. The dense
    // slot-major cache rides as a degenerate paged pool through the identity
    // block table (see `d_blk_tab`). The effective window is the CONSTANT
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
                &sc.d_blk_tab,
                max_ctx / 16,
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
            Phase::PrefillRuns => {
                // the tile table is staged once per walk (`stage_inputs_runs`);
                // a tile that spills past its run's end is masked row by row
                // (`slots[b] == slot`), and the spilled rows are covered by their
                // own run's tiles - the kernel writes nothing for a foreign row
                e.attn_prefill_batch(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    &sc.d_slots,
                    &sc.d_tile_row0,
                    &sc.d_tile_slot,
                    n_qtiles(runs),
                    nh,
                    nkv,
                    hd,
                    max_ctx,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            Phase::Prefill => e.attn_prefill(
                &sc.d_qn,
                kc,
                vc,
                &sc.d_sinks,
                &mut sc.d_attn,
                &sc.d_pos,
                &sc.d_slots,
                nh,
                nkv,
                hd,
                max_ctx,
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
                if e.has_attn_decode_fmha_sp()
                    && super::attn_fmha_sp() >= 2
                    && super::attn_fmha_enabled()
                    && n <= 64
                    && (hd == 128 || hd == 256) =>
            {
                e.attn_decode_fmha_sp(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &mut sc.d_fmha_part,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    nh,
                    nkv,
                    hd,
                    max_ctx,
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
                if e.has_attn_decode_fmha()
                    && super::attn_fmha_enabled()
                    && (hd == 128 || hd == 256) =>
            {
                e.attn_decode_fmha(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    nh,
                    nkv,
                    hd,
                    max_ctx,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            Phase::Decode | Phase::DecodeBatch
                if e.has_attn_decode_batch_ps() && super::attn_ps_enabled() =>
            {
                e.attn_decode_batch_ps(
                    &sc.d_qn,
                    kc,
                    vc,
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    nh,
                    nkv,
                    hd,
                    max_ctx,
                    kv_dim,
                    0,
                    n,
                    scale,
                    KV(),
                )?
            }
            Phase::Decode | Phase::DecodeBatch => e.attn_decode_batch(
                &sc.d_qn,
                kc,
                vc,
                &sc.d_sinks,
                &mut sc.d_attn,
                &sc.d_pos,
                Some(&sc.d_slots),
                nh,
                nkv,
                hd,
                max_ctx,
                kv_dim,
                0,
                n,
                scale,
                KV(),
            )?,
        }
    }
    e.mul_sigmoid(&mut sc.d_attn, &sc.d_agate, n * q_dim)?;
    w.o.matmul(e, &sc.d_attn, &mut sc.d_mix, n, stage)?;
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
    e.kquant_moe_gate_up(
        g,
        u,
        idx,
        &sc.d_xq,
        &sc.d_xs,
        ng.then_some(&sc.d_ssums),
        &mut sc.d_act,
        k,
        n,
    )?;
    e.quantize_q8(&sc.d_act, &mut sc.d_fq, &mut sc.d_fs, rows * ff)?;
    let nd = needs(d.ty);
    if nd {
        e.q8_sums_strided(&sc.d_fq, &mut sc.d_ssums, ff, rows)?;
    }
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
    Ok(())
}

/// The routed pair served expert-major through the slot cache (the prefill
/// class of `kq_moe_routed`): plan the waves once, then per wave resolve +
/// fill its experts and run the token-batched pair over all rows with every
/// out-of-wave pair marked absent (its blocks exit at once). Wave 0's down writes
/// `d_mix`; later waves write `d_mix_wave` and add - the row's k pairs are
/// summed in wave order instead of pair order, the same f32 reassociation
/// class the sorted MoE path carries. Waves are a fixed count so the launch
/// sequence is capture-stable; an empty wave's pair kernels exit block by block.
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
    for w in 0..cc.n_waves {
        e.moe_wave_resolve(cc, w)?;
        e.moe_cache_fill(cc, cc.slots)?;
        e.moe_wave_mask(cc, &sc.d_idx, rows, w)?;
        let idx = cc.idx_wave();
        e.kquant_moe_gate_up(
            g,
            u,
            idx,
            &sc.d_xq,
            &sc.d_xs,
            ng.then_some(&sc.d_ssums),
            &mut sc.d_act,
            k,
            n,
        )?;
        e.quantize_q8(&sc.d_act, &mut sc.d_fq, &mut sc.d_fs, rows * ff)?;
        if nd {
            e.q8_sums_strided(&sc.d_fq, &mut sc.d_ssums, ff, rows)?;
        }
        let out = if w == 0 {
            &mut sc.d_mix
        } else {
            &mut sc.d_mix_wave
        };
        e.kquant_moe_down(
            d,
            idx,
            &sc.d_topw,
            &sc.d_fq,
            &sc.d_fs,
            nd.then_some(&sc.d_ssums),
            out,
            k,
            n,
        )?;
        if w > 0 {
            e.add(&mut sc.d_mix, &sc.d_mix_wave, n * h)?;
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
    let fused = false;
    match &w.seats {
        ExpertSeats::Nvf4 { gate, up, down } => {
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
        ExpertSeats::Kq {
            gate,
            up,
            down,
            cache,
        } => kq_moe_routed(e, c, gate, up, down, cache.as_deref(), sc, n)?,
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

#[allow(clippy::type_complexity)]
fn alloc_state(
    e: &Arc<GpuExecutor>,
    c: &Qwen4ExpConfig,
    max_tokens: usize,
    slots: usize,
) -> Result<
    (
        Vec<Option<CudaSlice<f32>>>,
        Vec<Option<CudaSlice<u8>>>,
        Vec<Option<CudaSlice<u8>>>,
    ),
    GpuModelError,
> {
    let kv_bytes = slots * max_tokens * c.n_kv_heads * c.head_dim * KV().bytes();
    let state_len = slots * c.gdn_v_heads * c.gdn_k_dim * c.gdn_v_dim;
    let (mut recur, mut kk, mut kv) = (Vec::new(), Vec::new(), Vec::new());
    for li in 0..c.n_layer {
        match c.blocks[li] {
            Qwen4ExpBlock::Gdn => {
                recur.push(Some(e.alloc(state_len)?));
                kk.push(None);
                kv.push(None);
            }
            Qwen4ExpBlock::Attention => {
                recur.push(None);
                kk.push(Some(e.alloc_u8(kv_bytes)?));
                kv.push(Some(e.alloc_u8(kv_bytes)?));
            }
        }
    }
    Ok((recur, kk, kv))
}

impl Scratch {
    fn new(
        e: &Arc<GpuExecutor>,
        c: &Qwen4ExpConfig,
        t: usize,
        slots: usize,
        // the GGUF lane's k-quant dense planes / expert seats read int8
        // activations; the safetensors lane never does, and the pair below
        // is ~15 KB per token of max_ctx (2 GB at 128k), so it is a stub there
        kq_lanes: bool,
    ) -> Result<Self, GpuModelError> {
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
            // A pack that HAS the entry and still refuses is the case worth
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
            d_blk_tab: e
                .to_device_u32(&(0..(slots * (t / 16)).max(1) as u32).collect::<Vec<u32>>())?,
            d_tok: e.alloc_u32(t)?,
            d_pos: e.alloc_u32(t)?,
            d_mrope: e.alloc_u32(4 * t)?,
            d_slots: e.alloc_u32(t)?,
            d_x: e.alloc(t * h)?,
            d_h: e.alloc(t * hw)?,
            d_xn: e.alloc(t * hw)?,
            // + hc: the folded inject rows land in this plane's tail at batch 1
            d_m: e.alloc(t * c.hc_lowrank + c.hc_count)?,
            d_gate: e.alloc(t * hw)?,
            d_bi: e.alloc(t * h)?,
            d_inj: e.alloc(t * hc)?,
            d_mix: e.alloc(t * h)?,
            d_mix_wave: e.alloc(t * h)?,
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
            d_dnrn: e.alloc(t * c.gdn_v_heads * 2)?,
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
            d_tile_row0: e.alloc_u32(t / PD_APF_TQ + slots.max(1) + 1)?,
            d_tile_slot: e.alloc_u32(t / PD_APF_TQ + slots.max(1) + 1)?,
            d_pkey: e.alloc(t * hw)?,
            d_pval: e.alloc(t * h)?,
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
fn q4x_gen_err(e: GpuModelError) -> crate::generator::GenError {
    crate::generator::GenError::Backend(e.to_string())
}

use crate::generator::{RowSample, SampledStep};

impl crate::generator::Generator for Qwen4ExpGpu {
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
            while hi < items.len() && (hi == lo || rows + items[hi].1.len() <= self.max_tokens) {
                rows += items[hi].1.len();
                hi += 1;
            }
            if rows > self.max_tokens {
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
        self.sample_rows_from_logits(&rows, plans)
            .map_err(q4x_gen_err)
    }
}
