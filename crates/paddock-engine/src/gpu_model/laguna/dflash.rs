//! DFlash block-diffusion drafter - poolside Laguna-XS-2.1-DFlash. Study
//! refs: arXiv 2602.06036, the z-lab reference generate loop, and vLLM's
//! laguna_dflash.py load path.
//!
//! One 5-layer forward drafts block-1 = 15 tokens: rows = [committed token,
//! 15 × mask id 12] embedded via the TARGET's table, five causal SWA-512
//! laguna layers (fused-qkv f16 planes, per-head softplus gate, qk-RMSNorm,
//! plain rope θ=500k), drafter final norm, TARGET lm_head over the rows,
//! greedy argmax = drafts. The drafter conditions on the target through its
//! KV: per layer, k/v projections of fused features z = hidden_norm(fc(concat
//! aux_norm_i(h_i))) - h_i the target residuals after blocks 1,13,25,33,39 -
//! live in a per-slot ring; the block's own rows append transiently and are
//! overwritten by later rounds (the reference's cache crop, done as ring
//! overwrite).
//!
//! Weights are BF16 safetensors loaded two ways (stage D-2): the five big
//! projection planes per layer (wq/wo/gate/up/down - 91 % of the drafter's
//! bytes) quantize to Q8_0 on host at load and ride the int8 mmq ladder
//! (r = 16..64 draft rounds hit the one-weight-pass `mma_ks` rung - cuBLAS
//! f16 ran the same planes at ~1/3 bandwidth); wk/wv/wg stay f16 (shared
//! with the feature-append path, ~9 MB/layer) on cuBLAS `gemm_f16_f32`, as
//! does the fusion fc. PADDOCK_DFLASH_F16=1 pins the all-f16 stage-C arm
//! for A/B. bf16 -> f16 is exact in the mantissa direction; range and
//! finiteness are audited - a checkpoint that doesn't cast cleanly is a
//! loud load error, never a silent clip. Draft numerics only move
//! acceptance (every commit is target-verified), so weight+activation
//! quant noise here is llama's own drafter class, gated on measured E[A].
//!
//! Ring discipline: ring = (block + window)/16 + 1 blocks = 34 (544
//! positions on the shipped config). Block rows p..p+15 alias only ring
//! slots p-544..p-529 - outside every query row's window [pos-511, pos] -
//! so append-before-attend is safe by construction. Feature appends longer
//! than the ring keep only the trailing window+block positions per run and
//! cut same-slot spans at SWA_SPAN so no launch writes one physical slot
//! twice.
//!
//! Stage C (spec-round integration): the target's `layer_walk` copies its
//! post-layer residuals for layers `target_layer_ids` into the aux bands on
//! every batched forward while armed, and each site then calls
//! `dflash_append_features` - prefill chunks, decode ticks, and (accepted
//! rows only, via `dflash_spec_commit`) verify rounds - so the per-slot
//! watermark tracks the serving position continuously and the drafter stays
//! warm across regime changes. Armed = state built, which happens only under
//! PADDOCK_LAGUNA_SPEC: default serving stays byte-identical with the
//! drafter merely attached. The service consumes the drafter through the
//! standard spec hooks (`spec_capable`/`spec_ensure_warm`/`spec_draft_batch`
//! + the verify rounds' internal commit).
//!
//! Stage D (captured rounds): the draft body replays as one CUDA graph per
//! block count, and the spec-verify walk as one per chunk-length signature
//! (batch.rs) - first sight runs eagerly (serves the round + warms cuBLAS
//! workspaces), then records the identical launch stream. The commit
//! fusion+append stays eager: its shapes depend on per-round accept counts
//! and it is ~50 launches, not ~600.

use std::collections::HashMap;
use std::path::Path;

use cudarc::driver::CudaSlice;
use half::f16;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::safetensors::{DflashConfig, SafetensorsFile, StDtype};

use crate::gpu::{GpuError, KvDtype, RepackedQ8};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{mmq_kq_pre, mmq_pre};

use super::batch::{LayerKv, SPEC_ROWS, SWA_SPAN, pf_rows};
use super::*;

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// A drafter projection plane: Q8_0-quantized at load onto the int8 mmq
/// ladder (the stage D-2 default - halves the draft round's weight bytes and
/// swaps cuBLAS's ~1/3-bandwidth skinny GEMM for the one-weight-pass mma_ks
/// rung), or the all-f16 cuBLAS arm (PADDOCK_DFLASH_F16=1, stage-C numerics
/// for A/B). One arm per attach - the body branches per call site.
pub(crate) enum DraftW {
    F16(CudaSlice<f16>),
    Q8(RepackedQ8),
}

impl DraftW {
    fn out_dim(&self, in_dim: usize) -> usize {
        match self {
            DraftW::F16(w) => w.len() / in_dim,
            DraftW::Q8(w) => w.dims[1],
        }
    }
}

/// One drafter block: exactly a laguna SWA layer. The five big projections
/// are DraftW (Q8 by default); wk/wv/wg stay f16 - the feature-append path
/// shares them, and at ~9 MB/layer they're not worth a second numeric class.
pub(crate) struct DflashLayer {
    /// input_layernorm [embd]
    pub attn_norm: DeviceTensor,
    /// q rows of the fused qkv_proj [n_heads*hd, embd] - vLLM's load splits
    /// the checkpoint plane at q_size: rows [q | k | v].
    pub wq: DraftW,
    pub wk: CudaSlice<f16>,
    pub wv: CudaSlice<f16>,
    /// g_proj [n_heads, embd] - the per-head softplus output gate.
    pub wg: CudaSlice<f16>,
    /// per-head qk RMSNorm [hd]
    pub q_norm: DeviceTensor,
    pub k_norm: DeviceTensor,
    /// o_proj [embd, n_heads*hd]
    pub wo: DraftW,
    /// post_attention_layernorm [embd]
    pub ffn_norm: DeviceTensor,
    pub w_gate: DraftW,
    pub w_up: DraftW,
    pub w_down: DraftW,
}

/// Lazily-built serving state: feature-KV rings + staging (needs the batch
/// lane's slot count, so it can't build at attach).
pub(crate) struct DflashState {
    /// per-layer feature K/V rings [slots * ring blocks, 16, kv_dim] f16
    pub kv: Vec<LayerKv>,
    /// static ring table [slots, bps] - same s*ring + j%ring shape as the
    /// target's SWA table, drafter-sized ring
    pub d_bt: CudaSlice<u32>,
    // geometry echoes for the stage-C capture/integration asserts
    #[allow(dead_code)]
    pub ring: usize,
    pub bps: usize,
    #[allow(dead_code)]
    pub slots: usize,
    /// Row stride of one aux band = the batch lane's scratch row capacity
    /// (`BatchState::cap`). Must match, or a fused tick's rows walk off the
    /// band into the next one's.
    pub band: usize,
    /// f16 GEMM-activation staging: `a` = embd-wide inputs (feature chunks up
    /// to `band` rows), `b` = the widest attn-out / gated-FFN intermediate the
    /// drafter reports (draft rounds only, so SPEC_ROWS-scaled)
    x16a: CudaSlice<f16>,
    x16b: CudaSlice<f16>,
    /// fusion pipeline: fc band-accumulate -> hidden_norm -> f16
    zacc: CudaSlice<f32>,
    z: CudaSlice<f32>,
    z16: CudaSlice<f16>,
    /// aux capture bands, BLOCK-major: band i (= target_layer_ids[i]) rows at
    /// [i*band*embd ..]. The capture site copies residuals here; fusion
    /// norms each band in place-order - no per-row concat staging (fc runs as
    /// accumulating band GEMMs instead).
    pub aux: CudaSlice<f32>,
    /// per-slot contiguous feature coverage [start, end): drafts at p are
    /// legal iff end == p and start ≤ max(0, p - window) - the warmth gate
    /// (prefix-restored spans have no features; coverage rebuilds as fresh
    /// rows walk).
    pub feat: Vec<(u32, u32)>,
    /// captured draft rounds keyed by block count n (r = n·block): the whole
    /// 5-layer forward + head + argmax as one replay. ≤ SPEC_ROWS/block keys.
    /// Dies with the state (enable_batch rebuilds the scratch it bakes).
    pub graphs: HashMap<usize, super::SendGraph>,
}

/// The attached drafter: f16 planes + fusion head + geometry.
pub(crate) struct DflashDrafter {
    pub layers: Vec<DflashLayer>,
    /// aux_hidden_norms.{i} [embd] - normalizes target residual band i
    pub aux_norms: Vec<DeviceTensor>,
    /// fc [embd, n_aux*embd] split into n_aux column bands [embd, embd]
    /// (concat order = target_layer_ids order, per the reference's
    /// extract_context_feature)
    pub fc_bands: Vec<CudaSlice<f16>>,
    pub hidden_norm: DeviceTensor,
    /// the drafter's own final norm (`norm.weight`) - the TARGET's lm_head
    /// runs on top of it (no drafter head in the checkpoint)
    pub final_norm: DeviceTensor,
    /// the layer_walk capture map (the walk taps residuals after these)
    pub target_layer_ids: Vec<usize>,
    pub block: usize,
    pub mask_token: u32,
    pub n_heads: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub window: usize,
    pub eps: f32,
    /// plain full-rotary rope θ=500k (ext_factor 0 - same construction as the
    /// target's SWA rope, different base)
    pub rope: (f32, f32, f32, f32, f32, f32),
    /// device bytes the planes hold (will-it-fit accounting)
    #[allow(dead_code)]
    pub bytes: u64,
    pub state: Option<DflashState>,
}

/// Result of the synthetic end-to-end smoke (`dflash_selftest`).
pub struct DflashSelftest {
    pub drafts: Vec<u32>,
    /// same round twice -> identical picks (catches ring-append races)
    pub repeat_identical: bool,
    pub ms_per_round: f64,
}

/// bf16 bytes -> f16, widening the mantissa exactly. Counts range clips and
/// NaNs so a checkpoint that doesn't fit f16 fails loudly (drafter numerics
/// only move acceptance, but a silent ±inf plane is a debugging tarpit).
fn bf16_to_f16(bytes: &[u8]) -> (Vec<f16>, usize) {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut bad = 0usize;
    for c in bytes.as_chunks::<2>().0 {
        let bits = u16::from_le_bytes(*c);
        let f = f32::from_bits((bits as u32) << 16);
        if !f.is_finite() || f.abs() > f16::MAX.to_f32() {
            bad += 1;
        }
        out.push(f16::from_f32(f));
    }
    (out, bad)
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// Quantize an f32 plane to raw GGUF-layout Q8_0 blocks (34 B: f16 scale +
/// 32 int8) for `repack_q8_blocks` - the drafter's load-time weight quant.
/// Standard ggml recipe: d = amax/127, q = round(x/d), round half away from
/// zero (Rust `round` == C `roundf`). Non-finite inputs are tallied so a
/// poisoned checkpoint fails loudly instead of quantizing NaN to 0.
fn q8_0_blocks(vals: &[f32], bad: &mut usize) -> Vec<u8> {
    debug_assert_eq!(vals.len() % 32, 0);
    let mut out = Vec::with_capacity(vals.len() / 32 * 34);
    for blk in vals.as_chunks::<32>().0 {
        let mut amax = 0.0f32;
        for &v in blk {
            if !v.is_finite() {
                *bad += 1;
            }
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in blk {
            out.push((v * id).round() as i8 as u8);
        }
    }
    out
}

impl GpuLaguna {
    /// Sideload the DFlash drafter checkpoint (model.safetensors + config.json;
    /// `path` may be either the file or its directory). Validates geometry
    /// against both the config and the target, widens BF16 -> f16 on device,
    /// splits the fused qkv and the fusion fc at load. Serving behavior is
    /// unchanged until the spec-round integration (stage C) consumes it.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<(), GpuModelError> {
        // the drafter's layers are laguna's (gated attention, no sandwich
        // norms); no Kolibri drafter ships in that class
        if self.hp.flavor != super::Flavor::Laguna {
            return Err(GpuModelError::Unsupported(format!(
                "{}: no DFlash drafter class for this family",
                self.hp.flavor.name()
            )));
        }
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(path)
        };
        let file = if path.is_file() {
            path.to_path_buf()
        } else {
            dir.join("model.safetensors")
        };
        let cfg = DflashConfig::read(&dir.join("config.json"))
            .map_err(|e| GpuModelError::Unsupported(format!("dflash config: {e}")))?;
        let st = SafetensorsFile::open(&file)
            .map_err(|e| GpuModelError::Unsupported(format!("dflash safetensors: {e}")))?;

        // Only the causal poolside variant is implemented - the bidirectional
        // block (z-lab's qwen3 arm) needs a mask our kernels don't do.
        if !cfg.causal {
            return Err(GpuModelError::Unsupported(
                "dflash: checkpoint is not the causal variant (dflash_config.causal false)".into(),
            ));
        }
        // shared embed/lm_head + the scratch reuse both hang on these
        for (name, got, want) in [
            ("vocab_size", cfg.vocab, self.hp.n_vocab),
            ("hidden_size", cfg.hidden, self.hp.n_embd),
            ("head_dim", cfg.head_dim, self.hp.head_dim),
            ("num_key_value_heads", cfg.n_kv_heads, self.hp.n_kv_heads),
        ] {
            if got != want {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash: {name} {got} != target {want}"
                )));
            }
        }
        let n_aux = cfg.target_layer_ids.len();
        if n_aux == 0
            || cfg.target_layer_ids.iter().any(|&i| i >= self.hp.n_layer)
            || !cfg.target_layer_ids.is_sorted()
        {
            return Err(GpuModelError::Unsupported(format!(
                "dflash: target_layer_ids {:?} don't index the {}-layer target",
                cfg.target_layer_ids, self.hp.n_layer
            )));
        }
        // The q-width ceiling was a stale guard: the staging plane is sized
        // from the drafter's own dims at enable (see `wide` below), so any
        // width it reports is already covered. S-2.1's drafter is 9216 and
        // would have been refused here for no reason. `block` still binds -
        // the draft-round kernels index a fixed SPEC_ROWS band.
        if cfg.block < 2 || cfg.block > 64 {
            return Err(GpuModelError::Unsupported(format!(
                "dflash: block {} outside the 2..=64 the draft rounds size",
                cfg.block
            )));
        }

        self.exec
            .vram_load_gate(
                file.metadata().map(|m| m.len()).unwrap_or(0),
                "laguna-dflash",
            )
            .map_err(GpuModelError::WontFit)?;

        let exec = self.exec.clone();
        let mut bytes = 0u64;
        let mut bad_total = 0usize;
        // f16 plane off a row range of a header tensor, shape-checked;
        // mutable tallies ride as params so `plane` and `norm` coexist
        let plane = |name: &str,
                     rows: std::ops::Range<usize>,
                     cols: usize,
                     bytes: &mut u64,
                     bad: &mut usize|
         -> Result<CudaSlice<f16>, GpuModelError> {
            let (t, b) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::MissingMeta(format!("dflash tensor {name}")))?;
            if t.dtype != StDtype::Bf16 {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash {name}: dtype {:?} (want BF16)",
                    t.dtype
                )));
            }
            if t.shape.len() != 2 || t.shape[1] != cols || rows.end > t.shape[0] {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash {name}: shape {:?}, wanted rows {rows:?} × {cols}",
                    t.shape
                )));
            }
            let (v, nb) = bf16_to_f16(&b[rows.start * cols * 2..rows.end * cols * 2]);
            *bad += nb;
            *bytes += (v.len() * 2) as u64;
            exec.stream
                .clone_htod(&v)
                .map_err(drv)
                .map_err(GpuModelError::from)
        };
        // big-plane loader (stage D-2): host-quantize the bf16 rows to Q8_0
        // and repack onto the int8 ladder - or the f16 arm under
        // PADDOCK_DFLASH_F16 (stage-C numerics, the A/B reference)
        let f16_arm = paddock_models::dev_var_os!("PADDOCK_DFLASH_F16").is_some();
        let bplane = |name: &str,
                      rows: std::ops::Range<usize>,
                      cols: usize,
                      bytes: &mut u64,
                      bad: &mut usize|
         -> Result<DraftW, GpuModelError> {
            if f16_arm {
                return Ok(DraftW::F16(plane(name, rows, cols, bytes, bad)?));
            }
            let (t, b) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::MissingMeta(format!("dflash tensor {name}")))?;
            if t.dtype != StDtype::Bf16 {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash {name}: dtype {:?} (want BF16)",
                    t.dtype
                )));
            }
            if t.shape.len() != 2 || t.shape[1] != cols || rows.end > t.shape[0] {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash {name}: shape {:?}, wanted rows {rows:?} × {cols}",
                    t.shape
                )));
            }
            let v = bf16_to_f32(&b[rows.start * cols * 2..rows.end * cols * 2]);
            let blocks = q8_0_blocks(&v, bad);
            let w = exec
                .repack_q8_blocks(&blocks, vec![cols, rows.end - rows.start])
                .map_err(GpuModelError::from)?;
            *bytes += (w.data.len() + w.scale.len()) as u64;
            Ok(DraftW::Q8(w))
        };
        // norms as f32 DeviceTensors (the rmsnorm kernels' weight side)
        let norm = |name: &str, n: usize, bytes: &mut u64| -> Result<DeviceTensor, GpuModelError> {
            let (t, b) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::MissingMeta(format!("dflash tensor {name}")))?;
            if t.dtype != StDtype::Bf16 || t.shape != vec![n] {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash {name}: {:?} {:?} (want BF16 [{n}])",
                    t.dtype, t.shape
                )));
            }
            let v = bf16_to_f32(b);
            *bytes += (v.len() * 4) as u64;
            Ok(DeviceTensor {
                buf: exec
                    .stream
                    .clone_htod(&v)
                    .map_err(drv)
                    .map_err(GpuModelError::from)?,
                dims: vec![n],
            })
        };

        let (embd, hd, n_heads, n_kv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let (q_dim, kv_dim) = (n_heads * hd, n_kv * hd);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("layers.{l}.{s}");
            let qkv = p("self_attn.qkv_proj.weight");
            layers.push(DflashLayer {
                attn_norm: norm(&p("input_layernorm.weight"), embd, &mut bytes)?,
                // fused rows [q | k | v] - the vLLM load_weights split order
                wq: bplane(&qkv, 0..q_dim, embd, &mut bytes, &mut bad_total)?,
                wk: plane(
                    &qkv,
                    q_dim..q_dim + kv_dim,
                    embd,
                    &mut bytes,
                    &mut bad_total,
                )?,
                wv: plane(
                    &qkv,
                    q_dim + kv_dim..q_dim + 2 * kv_dim,
                    embd,
                    &mut bytes,
                    &mut bad_total,
                )?,
                wg: plane(
                    &p("self_attn.g_proj.weight"),
                    0..n_heads,
                    embd,
                    &mut bytes,
                    &mut bad_total,
                )?,
                q_norm: norm(&p("self_attn.q_norm.weight"), hd, &mut bytes)?,
                k_norm: norm(&p("self_attn.k_norm.weight"), hd, &mut bytes)?,
                wo: bplane(
                    &p("self_attn.o_proj.weight"),
                    0..embd,
                    q_dim,
                    &mut bytes,
                    &mut bad_total,
                )?,
                ffn_norm: norm(&p("post_attention_layernorm.weight"), embd, &mut bytes)?,
                w_gate: bplane(
                    &p("mlp.gate_proj.weight"),
                    0..cfg.intermediate,
                    embd,
                    &mut bytes,
                    &mut bad_total,
                )?,
                w_up: bplane(
                    &p("mlp.up_proj.weight"),
                    0..cfg.intermediate,
                    embd,
                    &mut bytes,
                    &mut bad_total,
                )?,
                w_down: bplane(
                    &p("mlp.down_proj.weight"),
                    0..embd,
                    cfg.intermediate,
                    &mut bytes,
                    &mut bad_total,
                )?,
            });
        }
        // fusion fc [embd, n_aux*embd] -> n_aux column bands [embd, embd].
        // Band i multiplies aux band i (block-major capture layout), summed
        // via accumulating GEMMs - mathematically the concat fc, no restage.
        let mut fc_bands = Vec::with_capacity(n_aux);
        {
            let (t, b) = st
                .bytes("fc.weight")
                .ok_or_else(|| GpuModelError::MissingMeta("dflash tensor fc.weight".into()))?;
            if t.dtype != StDtype::Bf16 || t.shape != vec![embd, n_aux * embd] {
                return Err(GpuModelError::Unsupported(format!(
                    "dflash fc.weight: {:?} {:?} (want BF16 [{embd}, {}])",
                    t.dtype,
                    t.shape,
                    n_aux * embd
                )));
            }
            for i in 0..n_aux {
                let mut band = Vec::with_capacity(embd * embd);
                for o in 0..embd {
                    let off = (o * n_aux * embd + i * embd) * 2;
                    let (v, nb) = bf16_to_f16(&b[off..off + embd * 2]);
                    bad_total += nb;
                    band.extend_from_slice(&v);
                }
                bytes += (band.len() * 2) as u64;
                fc_bands.push(
                    exec.stream
                        .clone_htod(&band)
                        .map_err(drv)
                        .map_err(GpuModelError::from)?,
                );
            }
        }
        let mut aux_norms = Vec::with_capacity(n_aux);
        for i in 0..n_aux {
            aux_norms.push(norm(
                &format!("aux_hidden_norms.{i}.weight"),
                embd,
                &mut bytes,
            )?);
        }
        let hidden_norm = norm("hidden_norm.weight", embd, &mut bytes)?;
        let final_norm = norm("norm.weight", embd, &mut bytes)?;

        if bad_total > 0 {
            return Err(GpuModelError::Unsupported(format!(
                "dflash: {bad_total} weights outside f16 range or non-finite - refusing the cast"
            )));
        }
        // audit against the header: a tensor we silently ignore is a modeling
        // drift we didn't notice (10 per layer + n_aux aux norms + fc +
        // hidden_norm + norm)
        let expected = cfg.n_layer * 10 + n_aux + 3;
        if st.tensors().len() != expected {
            let known = |n: &str| {
                n.starts_with("layers.")
                    || n.starts_with("aux_hidden_norms.")
                    || n == "fc.weight"
                    || n == "hidden_norm.weight"
                    || n == "norm.weight"
            };
            let extra: Vec<_> = st.tensors().keys().filter(|n| !known(n)).collect();
            return Err(GpuModelError::Unsupported(format!(
                "dflash: {} tensors in file, {expected} consumed - unknown: {extra:?}",
                st.tensors().len()
            )));
        }

        // Serving spec knobs (the gemma4 wide-spec precedent; explicit env
        // always wins). DFlash drafts the whole block in one forward, so
        // draft depth is free - but verify rows are not free on an MoE
        // target: every row re-bills its top-8 experts across all routed
        // layers (~0.9-1.3 ms/row measured), while the accept distribution
        // is front-loaded (on a code probe, E[min(A,8)] = 3.38 of
        // E[A] = 3.56). k = 8 keeps ~95 % of the acceptance at 56 % of the
        // verify rows, and measured faster end to end than k = block-1.
        // Floor = cap: a missed round already paid the full draft - never
        // re-climb. Dormant unless the spec lane is enabled
        // (PADDOCK_LAGUNA_SPEC). SAFETY: model load runs before the serving
        // threads spawn.
        let k_default = (cfg.block - 1).min(8);
        for (k, v) in [
            ("PADDOCK_SPEC_MAX_K", k_default.to_string()),
            ("PADDOCK_SPEC_MAX_ROWS", SPEC_ROWS.to_string()),
            (
                "PADDOCK_SPEC_DEEP_LIVE_MAX",
                (SPEC_ROWS / cfg.block).to_string(),
            ),
            ("PADDOCK_SPEC_K_MISS_FLOOR", k_default.to_string()),
        ] {
            if std::env::var_os(k).is_none() {
                crate::envset::set_env(k, &v);
            }
        }

        let rope = YarnRope::new(hd, cfg.rope_theta, 1.0, cfg.max_pos, 0.0, 1.0, 32.0, 1.0)
            .kernel_params();
        self.weights_bytes += bytes;
        tracing::info!(
            "laguna dflash drafter attached: {} layers, block {}, mask {}, aux {:?}, {:.2} GB ({})",
            cfg.n_layer,
            cfg.block,
            cfg.mask_token,
            cfg.target_layer_ids,
            bytes as f64 / 1e9,
            if f16_arm { "f16" } else { "q8+f16" }
        );
        self.dflash = Some(DflashDrafter {
            layers,
            aux_norms,
            fc_bands,
            hidden_norm,
            final_norm,
            target_layer_ids: cfg.target_layer_ids,
            block: cfg.block,
            mask_token: cfg.mask_token,
            n_heads,
            n_kv,
            hd,
            window: cfg.window,
            eps: cfg.eps,
            rope,
            bytes,
            state: None,
        });
        Ok(())
    }

    /// The service's spec routing gate: drafter attached + the spec-lane
    /// env opt-in + the device argmax the greedy verify needs. Attach-time
    /// facts only (no serving state), so the single-user routing decision
    /// reads it correctly before enable_batch runs.
    pub(crate) fn serve_spec_on(&self) -> bool {
        super::batch::laguna_spec_on() && self.dflash.is_some() && self.exec.has_argmax_rows()
    }

    /// True when the serving state is live - built at enable_batch only
    /// under PADDOCK_LAGUNA_SPEC. Every capture/append/commit site keys on
    /// this, so default serving stays byte-identical with the drafter
    /// merely attached.
    pub(crate) fn dflash_armed(&self) -> bool {
        self.dflash.as_ref().is_some_and(|d| d.state.is_some())
    }

    /// Build the rings + staging once the batch lane knows its slot count.
    pub(crate) fn dflash_ensure_state(&mut self) -> Result<(), GpuModelError> {
        let Some(bs) = self.batch.as_ref() else {
            return Err(GpuModelError::Unsupported(
                "dflash: batch lane not enabled".into(),
            ));
        };
        let slots = bs.n_slots;
        let cap = bs.cap;
        let embd = self.hp.n_embd;
        let bps = self.max_ctx.div_ceil(16);
        let df = self.dflash.as_mut().expect("dflash attached");
        if df.state.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let kv_dim = df.n_kv * df.hd;
        // block rows + window, +1 block of slack - same shape as the target's
        // ring formula; 34 blocks (544 positions) on the shipped config
        let ring = ((df.block + df.window).div_ceil(16) + 1).min(bps);
        let mut host = vec![0u32; slots * bps];
        for s in 0..slots {
            for j in 0..bps {
                host[s * bps + j] = (s * ring + (j % ring)) as u32;
            }
        }
        let d_bt = e.to_device_u32(&host)?;
        let mut kv = Vec::with_capacity(df.layers.len());
        for _ in 0..df.layers.len() {
            let b = slots * ring * 16 * kv_dim * 2; // f16 bytes
            kv.push(LayerKv {
                k: e.alloc_u8(b)?,
                v: e.alloc_u8(b)?,
            });
        }
        // x16b stages both the attn-out rows (q_dim = n_heads*hd) and the
        // gated FFN rows (the drafter's own ff, read off w_gate) - sizing it
        // on q_dim alone silently overflows whenever ff is the wider of the
        // two, which is the common shape (XS: 2048 vs 8192).
        let wide = (df.n_heads * df.hd).max(df.layers[0].w_gate.out_dim(embd));
        tracing::info!(
            "laguna dflash state: {slots} slots × {ring} ring blocks × {} layers ({:.1} MB/slot)",
            df.layers.len(),
            (df.layers.len() * ring * 16 * kv_dim * 2 * 2) as f64 / 1e6
        );
        df.state = Some(DflashState {
            kv,
            d_bt,
            ring,
            bps,
            slots,
            band: cap,
            x16a: e.alloc_f16(cap * embd)?,
            x16b: e.alloc_f16(SPEC_ROWS * wide)?,
            zacc: e.alloc(cap * embd)?,
            z: e.alloc(cap * embd)?,
            z16: e.alloc_f16(cap * embd)?,
            aux: e.alloc(df.aux_norms.len() * cap * embd)?,
            feat: vec![(0, 0); slots],
            graphs: HashMap::new(),
        });
        Ok(())
    }

    /// True when the drafter can legally draft for `slot` at position `p`:
    /// features contiguous up to exactly p, covering the whole window.
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
        e as usize == p && (s as usize) <= p.saturating_sub(df.window)
    }

    /// Reset a slot's drafter coverage (fresh sequence / prefix restore /
    /// slot release: the ring content no longer matches what will serve).
    pub(crate) fn dflash_clear_slot(&mut self, slot: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut())
            && let Some(f) = st.feat.get_mut(slot)
        {
            *f = (0, 0);
        }
    }

    /// Fuse + ring-append feature K/V for freshly-walked target rows.
    /// Preconditions (the capture site's contract): the aux bands for rows
    /// 0..positions.len() sit in the state's aux buffer (band i at
    /// i*band*embd), and sc.d_pos / sc.d_slots still hold exactly these
    /// rows on device; `positions`/`slots` are their host mirrors.
    /// `spans` = None appends every row, grouped into same-slot runs (the
    /// prefill/decode case); Some((row, len)) appends only those row ranges
    /// (the verify-commit case - rejected rows' features are computed by the
    /// fusion but never ring-appended: their input tokens weren't what the
    /// sequence committed). Rows within a run/span must be one slot at
    /// contiguous ascending positions.
    pub(crate) fn dflash_append_features(
        &mut self,
        positions: &[u32],
        slots: &[u32],
        spans: Option<&[(usize, usize)]>,
    ) -> Result<(), GpuModelError> {
        let r = positions.len();
        assert_eq!(r, slots.len());
        if r == 0 {
            return Ok(());
        }
        // a fused mixed tick appends for chunk rows AND the decode band, so
        // the bound is the scratch row capacity, not the chunk size
        assert!(
            r <= self.batch.as_ref().map_or(pf_rows(), |b| b.cap),
            "feature append exceeds the row scratch"
        );
        self.dflash_ensure_state()?;
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let df = self.dflash.as_mut().expect("dflash attached");
        let (n_kv, hd, eps, rope, window, block) =
            (df.n_kv, df.hd, df.eps, df.rope, df.window, df.block);
        let kv_dim = n_kv * hd;
        let DflashDrafter {
            layers,
            aux_norms,
            fc_bands,
            hidden_norm,
            state,
            ..
        } = df;
        let st = state.as_mut().expect("state built");
        let sc = &mut self.batch.as_mut().expect("batch enabled").sc;

        // fusion: per band, aux-norm -> f16 -> accumulating fc band GEMM
        for (i, (an, fw)) in aux_norms.iter().zip(fc_bands.iter()).enumerate() {
            exec.rmsnorm_batch_at(
                &st.aux,
                i * st.band * embd,
                &an.buf,
                &mut sc.xn,
                embd,
                eps,
                r,
            )?;
            exec.convert_f32_f16(&sc.xn, &mut st.x16a, r * embd)?;
            if i == 0 {
                exec.gemm_f16_f32(fw, &st.x16a, &mut st.zacc, embd, embd, r)?;
            } else {
                exec.gemm_f16_f32_acc(fw, &st.x16a, &mut st.zacc, embd, embd, r)?;
            }
        }
        exec.rmsnorm_batch(&st.zacc, &hidden_norm.buf, &mut st.z, embd, eps, r)?;
        exec.convert_f32_f16(&st.z, &mut st.z16, r * embd)?;

        // append units: explicit spans, or same-slot runs derived from the
        // row stream. Keep only each unit's trailing window+block positions
        // (older rows sit outside every future query's window and would alias
        // ring-mates), then cut at SWA_SPAN so one launch never writes a
        // physical slot twice.
        let keep = window + block;
        let runs: Vec<(usize, usize)> = match spans {
            Some(s) => {
                if cfg!(debug_assertions) {
                    for &(row, len) in s {
                        for j in row + 1..row + len {
                            debug_assert_eq!(slots[j], slots[row], "span crosses slots");
                            debug_assert_eq!(
                                positions[j],
                                positions[j - 1] + 1,
                                "span not contiguous"
                            );
                        }
                    }
                }
                s.to_vec()
            }
            None => {
                let mut runs = Vec::new();
                let mut i = 0;
                while i < r {
                    let mut j = i + 1;
                    while j < r && slots[j] == slots[i] {
                        debug_assert_eq!(
                            positions[j],
                            positions[j - 1] + 1,
                            "feature rows must be contiguous"
                        );
                        j += 1;
                    }
                    runs.push((i, j - i));
                    i = j;
                }
                runs
            }
        };
        let mut cuts: Vec<(usize, usize)> = Vec::new(); // (row, len) appended
        for &(row, n) in &runs {
            let (off, len) = if n > keep {
                (row + n - keep, keep)
            } else {
                (row, n)
            };
            let mut o = off;
            while o < off + len {
                let l = (off + len - o).min(SWA_SPAN);
                cuts.push((o, l));
                o += l;
            }
        }

        for (li, layer) in layers.iter().enumerate() {
            exec.gemm_f16_f32(&layer.wk, &st.z16, &mut sc.k, embd, kv_dim, r)?;
            exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, eps, r * n_kv)?;
            exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope, r)?;
            exec.gemm_f16_f32(&layer.wv, &st.z16, &mut sc.v, embd, kv_dim, r)?;
            let kvs = &mut st.kv[li];
            for &(off, len) in &cuts {
                exec.kv_append_batch_paged_rows(
                    &sc.kn,
                    &mut kvs.k,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &st.d_bt,
                    st.bps,
                    kv_dim,
                    off,
                    len,
                    KvDtype::Fp16,
                )?;
                exec.kv_append_batch_paged_rows(
                    &sc.v,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &st.d_bt,
                    st.bps,
                    kv_dim,
                    off,
                    len,
                    KvDtype::Fp16,
                )?;
            }
        }

        // coverage bookkeeping off the (possibly truncated) spans
        for &(row, len) in &runs {
            let slot = slots[row] as usize;
            let (p0, p1) = (positions[row], positions[row] + len as u32);
            let new_start = if len > keep { p1 - keep as u32 } else { p0 };
            let (s, e) = st.feat[slot];
            st.feat[slot] = if p0 <= e && new_start <= e {
                (s.min(new_start), p1.max(e))
            } else {
                (new_start, p1)
            };
        }
        Ok(())
    }

    /// Post-verify feature commit: replay the SERVICE's accept rule on the
    /// picks this round returned (accept drafts while chunk[a+1] ==
    /// picks[base+a]; rows 0..=a became context) and ring-append only those
    /// rows' features. The rule must stay bit-identical to service.rs's
    /// walk - the watermark tracks each slot's true committed point, and a
    /// divergence would leave the drafter cold (warm wants end == pos
    /// exactly). Precondition: the verify walk just ran (aux bands +
    /// sc.d_pos/d_slots still hold this round's rows).
    pub(crate) fn dflash_spec_commit(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        picks: &[u32],
    ) -> Result<(), GpuModelError> {
        let total: usize = reqs.iter().map(|q| q.2.len()).sum();
        debug_assert_eq!(total, picks.len());
        let mut positions = Vec::with_capacity(total);
        let mut slots = Vec::with_capacity(total);
        let mut spans = Vec::with_capacity(reqs.len());
        let mut base = 0usize;
        for (slot, start, chunk) in reqs {
            let mut a = 0usize;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                a += 1;
            }
            spans.push((base, a + 1));
            for j in 0..chunk.len() {
                positions.push((*start + j) as u32);
                slots.push(*slot as u32);
            }
            base += chunk.len();
        }
        self.dflash_append_features(&positions, &slots, Some(&spans))
    }

    /// Warmth for the service's spec gate (`spec_ensure_warm`): can the
    /// drafter draft for `slot` at the next round's start position? There is
    /// deliberately no re-warm path - features flow from every batched
    /// forward while armed, so a cold slot here means a genuinely un-walked
    /// span (a warm-resume tail shorter than the window). Verify rounds
    /// still run then (n-gram drafts) and decode appends extend coverage
    /// until the window fills.
    pub(crate) fn spec_ensure_warm_impl(&self, slot: usize, want_pos: u32) -> bool {
        self.dflash_armed() && self.dflash_warm(slot, want_pos as usize + 1)
    }

    /// Model-side drafts for the serving spec round: one DFlash forward per
    /// round covers every warm slot with block-1 drafts each. Slots that
    /// don't fit - cold watermark, ctx-full, or past the SPEC_ROWS draft cap
    /// (8 blocks per round; first-come) - get empty draft lists and the
    /// service verifies them pending-only. None = nothing draftable this
    /// round (service falls back to its n-gram drafter).
    pub(crate) fn spec_draft_batch_impl(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuModelError> {
        if !self.dflash_armed() || k == 0 {
            return Ok(None);
        }
        let (block, feat) = {
            let df = self.dflash.as_ref().expect("armed");
            (df.block, df.state.as_ref().expect("armed").feat.clone())
        };
        // draft position = the watermark end. Appends at every batched
        // forward keep it equal to the slot's serving position; the
        // service's spec_ensure_warm gate rejects any slot where they could
        // have diverged (its want_pos comes from the slot's true position).
        let mut reqs: Vec<(usize, usize, u32)> = Vec::new();
        let mut which: Vec<usize> = Vec::new(); // pendings index per req
        for (i, &(slot, tok)) in pendings.iter().enumerate() {
            let Some(&(_, e)) = feat.get(slot) else {
                continue;
            };
            let p = e as usize;
            if (reqs.len() + 1) * block <= SPEC_ROWS
                && p + block <= self.max_ctx
                && self.dflash_warm(slot, p)
            {
                reqs.push((slot, p, tok));
                which.push(i);
            }
        }
        if reqs.is_empty() {
            return Ok(None);
        }
        let Some(blocks) = self.dflash_draft_blocks(&reqs)? else {
            return Ok(None);
        };
        let mut out = vec![Vec::new(); pendings.len()];
        for (w, mut b) in which.into_iter().zip(blocks) {
            b.truncate(k);
            out[w] = b;
        }
        Ok(Some(out))
    }

    /// One draft round: for each (slot, p, committed) request, one forward
    /// over [committed, block-1 × mask] rows at positions p..p+block drafts
    /// the next block-1 tokens greedily. Returns None (decline) when the
    /// drafter/state/argmax isn't available, a slot is cold, or the rows
    /// don't fit the spec scratch.
    pub(crate) fn dflash_draft_blocks(
        &mut self,
        reqs: &[(usize, usize, u32)],
    ) -> Result<Option<Vec<Vec<u32>>>, GpuModelError> {
        if self.dflash.is_none() || self.batch.is_none() || !self.exec.has_argmax_rows() {
            return Ok(None);
        }
        self.dflash_ensure_state()?;
        let (block, mask) = {
            let df = self.dflash.as_ref().expect("attached");
            (df.block, df.mask_token)
        };
        let r = reqs.len() * block;
        if reqs.is_empty() || r > SPEC_ROWS {
            return Ok(None);
        }
        for &(slot, p, _) in reqs {
            if p + block > self.max_ctx || !self.dflash_warm(slot, p) {
                return Ok(None);
            }
        }

        let mut toks = Vec::with_capacity(r);
        let mut positions = Vec::with_capacity(r);
        let mut slots_v = Vec::with_capacity(r);
        for &(slot, p, committed) in reqs {
            toks.push(committed);
            toks.extend(std::iter::repeat_n(mask, block - 1));
            positions.extend(p as u32..(p + block) as u32);
            slots_v.extend(std::iter::repeat_n(slot as u32, block));
        }
        self.upload_rows(&toks, &positions, &slots_v)?;

        // One captured replay per block count n: every per-row input the
        // body reads (tokens/positions/slots) is staged device data, and all
        // shapes/launches key on n alone. First sight of an n runs the body
        // eagerly - serving the round AND warming the cuBLAS f16 workspaces
        // for exactly these shapes before any recording (an alloc inside a
        // capture is a hard driver error) - then records the identical
        // launch stream. PADDOCK_SPEC_NOGRAPH pins eager for A/B.
        let n = reqs.len();
        let have = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.graphs.contains_key(&n));
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.dflash_draft_body(n)?;
        } else if !have {
            self.dflash_draft_body(n)?;
            let g = self.capture_body(|s| s.dflash_draft_body(n), "dflash draft")?;
            self.dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("state built")
                .graphs
                .insert(n, g);
        } else {
            self.dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("state built")
                .graphs[&n]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("dflash draft graph launch: {e}")))?;
        }

        let exec = self.exec.clone();
        let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
        let v = sc
            .d_spec_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_spec_out slice".into()))?;
        let picks = exec.stream.clone_dtoh(&v).map_err(drv)?;
        Ok(Some(
            (0..reqs.len())
                .map(|b| picks[b * block + 1..(b + 1) * block].to_vec())
                .collect(),
        ))
    }

    /// The draft round's device body (capture-safe): embed the staged rows,
    /// five drafter layers with transient block appends + ring attention,
    /// drafter final norm, TARGET lm_head, device argmax into d_spec_out.
    /// Shapes depend only on `n` (blocks) - all row data rides
    /// d_toks/d_pos/d_slots.
    fn dflash_draft_body(&mut self, n: usize) -> Result<(), GpuModelError> {
        let block = self.dflash.as_ref().expect("attached").block;
        let r = n * block;
        self.embed_rows(r)?;

        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let vocab = self.hp.n_vocab;
        let df = self.dflash.as_mut().expect("attached");
        let (n_heads, n_kv, hd, eps, rope, window) =
            (df.n_heads, df.n_kv, df.hd, df.eps, df.rope, df.window);
        let (q_dim, kv_dim) = (n_heads * hd, n_kv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let DflashDrafter {
            layers,
            final_norm,
            state,
            ..
        } = df;
        let st = state.as_mut().expect("state built");
        let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
        let ff = layers[0].w_gate.out_dim(embd);

        for (li, layer) in layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, eps, r)?;
            // wk/wv/wg ride the f16 route always; wq (and the other four big
            // planes below) branch on the load arm - Q8 quantizes the f32
            // rows and takes mmq_pre (r ≤ 64 = the one-weight-pass mma_ks
            // rung; deep-live r > 64 falls to the mt tile, documented gap)
            exec.convert_f32_f16(&sc.xn, &mut st.x16a, r * embd)?;
            match &layer.wq {
                DraftW::F16(w) => exec.gemm_f16_f32(w, &st.x16a, &mut sc.q, embd, q_dim, r)?,
                DraftW::Q8(w) => {
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                    mmq_pre(&exec, w, &sc.xq, &sc.xs, &mut sc.part, &mut sc.q, r)?;
                }
            }
            exec.gemm_f16_f32(&layer.wk, &st.x16a, &mut sc.k, embd, kv_dim, r)?;
            exec.gemm_f16_f32(&layer.wv, &st.x16a, &mut sc.v, embd, kv_dim, r)?;
            exec.gemm_f16_f32(&layer.wg, &st.x16a, &mut sc.gate_h, embd, n_heads, r)?;
            exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, eps, r * n_heads)?;
            exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, eps, r * n_kv)?;
            exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, n_heads, hd, rope, r)?;
            exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope, r)?;
            // block rows append transiently at p..p+block (their ring slots
            // sit outside every window read - module header), then attend per
            // same-slot run: ring features ∪ own rows, causal + window
            let kvs = &mut st.kv[li];
            exec.kv_append_batch_paged_rows(
                &sc.kn,
                &mut kvs.k,
                &sc.d_pos,
                Some(&sc.d_slots),
                &st.d_bt,
                st.bps,
                kv_dim,
                0,
                r,
                KvDtype::Fp16,
            )?;
            exec.kv_append_batch_paged_rows(
                &sc.v,
                &mut kvs.v,
                &sc.d_pos,
                Some(&sc.d_slots),
                &st.d_bt,
                st.bps,
                kv_dim,
                0,
                r,
                KvDtype::Fp16,
            )?;
            // WMMA prefill class, not the decode kernel: decode's (head, row)
            // grid re-reads each kv-head's ~540 KB window stream 128× at
            // 16 rows × GQA 8 - 553 MB/layer, measured 872 µs at the DRAM
            // roof. The prefill tile walk reads the window
            // once per span (96 µs/layer). Numeric class differs from decode
            // (f16 TC vs f32 scalar) - free for the drafter: drafts only
            // move acceptance, and the code probe measured zero acceptance
            // delta (accept counts identical across f16/Q8/decode/wmma legs;
            // the k=8-capped mean 3.36 matches E[min(A,8)] = 3.38).
            // PADDOCK_DFLASH_DECODE_ATTN=1 pins the decode class for A/B.
            // The target's verify rounds keep their decode class regardless
            // (spec parity contract).
            for b in 0..n {
                if hd == 128
                    && exec.has_attn_prefill_f16_paged()
                    && paddock_models::dev_var_os!("PADDOCK_DFLASH_DECODE_ATTN").is_none()
                {
                    exec.attn_prefill_f16_paged_at(
                        &sc.qn,
                        &kvs.k,
                        &kvs.v,
                        &sc.sinks,
                        &mut sc.attn,
                        &sc.d_pos,
                        &sc.d_slots,
                        b * block,
                        &st.d_bt,
                        st.bps,
                        n_heads,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        block,
                        scale,
                        KvDtype::Fp16,
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
                        &st.d_bt,
                        st.bps,
                        n_heads,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        b * block,
                        block,
                        scale,
                        KvDtype::Fp16,
                    )?;
                }
            }
            exec.mul_softplus_head(&mut sc.attn, &sc.gate_h, n_heads, hd, r)?;
            match &layer.wo {
                DraftW::F16(w) => {
                    exec.convert_f32_f16(&sc.attn, &mut st.x16b, r * q_dim)?;
                    exec.gemm_f16_f32(w, &st.x16b, &mut sc.proj, q_dim, embd, r)?;
                }
                DraftW::Q8(w) => {
                    exec.quantize_q8(&sc.attn, &mut sc.xq, &mut sc.xs, r * q_dim)?;
                    mmq_pre(&exec, w, &sc.xq, &sc.xs, &mut sc.part, &mut sc.proj, r)?;
                }
            }
            exec.add_rmsnorm_batch(
                &mut sc.x,
                &sc.proj,
                &layer.ffn_norm.buf,
                &mut sc.xn,
                embd,
                eps,
                r,
            )?;
            match (&layer.w_gate, &layer.w_up) {
                (DraftW::F16(g), DraftW::F16(u)) => {
                    exec.convert_f32_f16(&sc.xn, &mut st.x16a, r * embd)?;
                    exec.gemm_f16_f32(g, &st.x16a, &mut sc.ffn_gate, embd, ff, r)?;
                    exec.gemm_f16_f32(u, &st.x16a, &mut sc.ffn_up, embd, ff, r)?;
                }
                (DraftW::Q8(g), DraftW::Q8(u)) => {
                    // one quantize serves both - same normed rows
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                    mmq_pre(&exec, g, &sc.xq, &sc.xs, &mut sc.part, &mut sc.ffn_gate, r)?;
                    mmq_pre(&exec, u, &sc.xq, &sc.xs, &mut sc.part, &mut sc.ffn_up, r)?;
                }
                _ => unreachable!("one load arm per attach"),
            }
            exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * ff)?;
            match &layer.w_down {
                DraftW::F16(w) => {
                    exec.convert_f32_f16(&sc.ffn_gate, &mut st.x16b, r * ff)?;
                    exec.gemm_f16_f32(w, &st.x16b, &mut sc.proj, ff, embd, r)?;
                }
                DraftW::Q8(w) => {
                    exec.quantize_q8(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, r * ff)?;
                    mmq_pre(&exec, w, &sc.xq, &sc.xs, &mut sc.part, &mut sc.proj, r)?;
                }
            }
            exec.add(&mut sc.x, &sc.proj, r * embd)?;
        }

        // drafter final norm -> the TARGET's lm_head over all rows (mask rows
        // hold the drafts; anchor rows ride along - r ≤ SPEC_ROWS keeps the
        // waste at 1/block) -> device argmax
        exec.rmsnorm_batch(&sc.x, &final_norm.buf, &mut sc.xn, embd, eps, r)?;
        exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
        // Draft head stays on mmq_kq_pre's own rungs (the dp4a z-tile at
        // vocab width - 903 µs/round). A vocab-wide mma_ks via a drafter-
        // owned 8·64·vocab partial plane was BUILT AND FALSIFIED: isolated
        // it wins (3.56 -> 3.02 ms selftest round), but in serving the
        // ~410 MB/round of partial write+combine traffic cycles L2/TLB under
        // the adjacent verify rounds and end-to-end throughput fell. Head
        // levers must not add working
        // set; the remaining door is a fused band-GEMV+argmax (drafts
        // need argmax only, never the logit plane).
        // the drafter attaches to GGUF laguna only (attach_dflash refuses
        // the other builds), so the head is a quantized plane here
        let Head::Quant(lm) = &self.lm_head else {
            return Err(GpuModelError::Unsupported(
                "dflash: a BF16 head has no drafter".into(),
            ));
        };
        match lm {
            QuantW::Kq(k) => mmq_kq_pre(
                &exec,
                k,
                &sc.xq,
                &sc.xs,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                r,
            )?,
            QuantW::Q8(q) => mmq_pre(
                &exec,
                q,
                &sc.xq,
                &sc.xs,
                &mut sc.part,
                &mut sc.head_logits,
                r,
            )?,
        }
        exec.argmax_rows(&sc.head_logits, &mut sc.d_spec_out, r, vocab)?;
        Ok(())
    }

    /// Synthetic end-to-end smoke (the stage-B gate, driven by
    /// examples/laguna_dflash.rs): seed slot 0's ring with deterministic
    /// pseudo-features past the wrap point, draft twice (equality catches
    /// append races), time the eager round. Real-feature acceptance is stage
    /// D's harness.
    pub fn dflash_selftest(&mut self) -> Result<DflashSelftest, GpuModelError> {
        if self.dflash.is_none() {
            return Err(GpuModelError::Unsupported("dflash: not attached".into()));
        }
        if self.batch.is_none() {
            return Err(GpuModelError::Unsupported(
                "dflash selftest: enable_batch first".into(),
            ));
        }
        self.dflash_ensure_state()?;
        let embd = self.hp.n_embd;
        let n_aux = self.dflash.as_ref().expect("attached").aux_norms.len();
        let n = 600usize; // > ring positions (544): exercises wrap + trailing-keep

        // deterministic xorshift features, zero-mean unit-ish scale
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
        };
        {
            let exec = self.exec.clone();
            let st = self
                .dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("state built");
            for band in 0..n_aux {
                let host: Vec<f32> = (0..n * embd).map(|_| next()).collect();
                let mut dst = st
                    .aux
                    .try_slice_mut(band * st.band * embd..band * st.band * embd + n * embd)
                    .ok_or_else(|| GpuError::Driver("aux band slice".into()))?;
                exec.stream.memcpy_htod(&host, &mut dst).map_err(drv)?;
            }
        }
        let positions: Vec<u32> = (0..n as u32).collect();
        let slots = vec![0u32; n];
        // d_pos/d_slots contract: stage the rows like a capture site would
        let toks = vec![0u32; n];
        self.upload_rows(&toks, &positions, &slots)?;
        self.dflash_append_features(&positions, &slots, None)?;

        let reqs = [(0usize, n, 1u32)];
        let d1 = self
            .dflash_draft_blocks(&reqs)?
            .ok_or_else(|| GpuModelError::Unsupported("dflash selftest: draft declined".into()))?;
        let d2 = self
            .dflash_draft_blocks(&reqs)?
            .ok_or_else(|| GpuModelError::Unsupported("dflash selftest: repeat declined".into()))?;
        let vocab = self.hp.n_vocab as u32;
        if d1[0].iter().any(|&t| t >= vocab) {
            return Err(GpuModelError::Unsupported(format!(
                "dflash selftest: draft out of vocab: {:?}",
                d1[0]
            )));
        }
        self.exec.synchronize()?;
        let t0 = std::time::Instant::now();
        let rounds = 50;
        for _ in 0..rounds {
            let _ = self.dflash_draft_blocks(&reqs)?;
        }
        self.exec.synchronize()?;
        Ok(DflashSelftest {
            repeat_identical: d1 == d2,
            drafts: d1.into_iter().next().expect("one req"),
            ms_per_round: t0.elapsed().as_secs_f64() * 1e3 / rounds as f64,
        })
    }
}
