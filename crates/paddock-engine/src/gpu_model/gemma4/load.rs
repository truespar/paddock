//! GGUF -> device: header parse + weight upload for Gemma 4.
//!
//! Ground truth for every key/tensor:  (decoded from
//! the real 31B GGUF) and llama.cpp b10058 `load_arch_hparams` /
//! `load_arch_tensors`. Projection weights are repacked Q8_0 (RepackedQ8 -
//! `q8_0_gemv_repacked` at decode, `q8_0_gemm_repacked` at prefill, bit-equal
//! pair at batch 1); the tied token_embd stays raw Q8_0 for `dequant_slice`
//! row gathers + the head GEMV. Norms/scales are f32.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuError, GpuExecutor};

use super::{Arch, ExpertPlanes, GpuGemma4, Hparams, LayerWeights, MoeWeights, Plane};
use crate::gpu::{RepackedKQ, RepackedQ8};
use paddock_models::ggml_type::GgmlType;

/// The Q8 plane behind a seat, for the arms that convert FROM repacked Q8
/// (the fp8 twin builders, the reclaim pass). Those arms are elected off
/// when the file carries a k-quant plane, so this is a contract check, not
/// a dispatch: reaching it with another class is a loader defect, reported
/// as one rather than served.
fn q8_of(w: &Plane) -> Result<&RepackedQ8, LoadError> {
    w.q8().ok_or_else(|| {
        LoadError::Tensor(
            format!("plane {:?}", w.dims()),
            "an fp8-twin or reclaim arm reached a non-Q8 plane".into(),
        )
    })
}

/// Which tensor types the family's dense planes may take, and which class
/// each lands in. Anything else is a loud `NoKernel` from the repack.
fn plane_class(ty: GgmlType) -> Option<PlaneClass> {
    match ty {
        GgmlType::Bf16 => Some(PlaneClass::Bf16),
        GgmlType::Q8_0 => Some(PlaneClass::Q8),
        t if crate::gpu::kq_params(t).is_some() => Some(PlaneClass::Kq),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaneClass {
    Q8,
    Bf16,
    Kq,
}

/// Zero-extend a k-quant plane's OUT dim (its rows) with zero super-blocks:
/// the raw block stream is row-major, so the pad is a zero tail. Exact -
/// an all-zero Q4_K/Q5_K/Q6_K super-block dequants to 0 (d = dmin = 0).
fn kq_pad_out_raw(bytes: &[u8], row_b: usize, out: usize, out_p: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(out_p * row_b);
    v.extend_from_slice(&bytes[..out * row_b]);
    v.resize(out_p * row_b, 0);
    v
}

/// Zero-extend a flat 32-weight plane's IN dim (its K axis) with zero blocks
/// at each row's tail. Exact: a zero Q5_0 / Q8_0 block has d = 0. Only the
/// flat types can be padded this way (a 256-block row is whole super-blocks
/// by construction, and never needs it).
fn kq_pad_in_raw_flat(
    bytes: &[u8],
    blk_b: usize,
    in_dim: usize,
    in_p: usize,
    out: usize,
) -> Vec<u8> {
    let (bpr, bpr_p) = (in_dim / 32, in_p / 32);
    let mut v = vec![0u8; out * bpr_p * blk_b];
    for r in 0..out {
        v[r * bpr_p * blk_b..r * bpr_p * blk_b + bpr * blk_b]
            .copy_from_slice(&bytes[r * bpr * blk_b..(r + 1) * bpr * blk_b]);
    }
    v
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// The pre-load VRAM admission gate refused (honest will-it-fit: an
    /// oversubscribed card pages to system RAM and freezes the machine).
    #[error("{0}")]
    WontFit(String),
    #[error("GGUF missing metadata key {0}")]
    MissingKey(String),
    #[error("GGUF metadata {0}: unexpected type/value")]
    BadKey(String),
    #[error("gpu: {0}")]
    Gpu(#[from] GpuError),
    #[error("tensor {0}: {1}")]
    Tensor(String, String),
}

pub(crate) fn key_u64(map: &MappedGguf, key: &str) -> Result<u64, LoadError> {
    let v = map
        .gguf()
        .metadata
        .get(key)
        .ok_or_else(|| LoadError::MissingKey(key.to_owned()))?;
    v.as_u64().ok_or_else(|| LoadError::BadKey(key.to_owned()))
}

pub(crate) fn key_f32(map: &MappedGguf, key: &str, default: Option<f32>) -> Result<f32, LoadError> {
    match map.gguf().metadata.get(key) {
        Some(Value::F32(f)) => Ok(*f),
        Some(Value::F64(f)) => Ok(*f as f32),
        Some(_) => Err(LoadError::BadKey(key.to_owned())),
        None => default.ok_or_else(|| LoadError::MissingKey(key.to_owned())),
    }
}

pub(crate) fn key_bool_array(map: &MappedGguf, key: &str) -> Result<Vec<bool>, LoadError> {
    match map.gguf().metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                Value::Bool(b) => Ok(*b),
                _ => Err(LoadError::BadKey(key.to_owned())),
            })
            .collect(),
        _ => Err(LoadError::MissingKey(key.to_owned())),
    }
}

pub(crate) fn key_u64_array(map: &MappedGguf, key: &str) -> Result<Vec<u64>, LoadError> {
    match map.gguf().metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| v.as_u64().ok_or_else(|| LoadError::BadKey(key.to_owned())))
            .collect(),
        _ => Err(LoadError::MissingKey(key.to_owned())),
    }
}

/// Same key, but tolerate the SCALAR spelling by broadcasting it to `n`.
///
/// gemma4 writes the per-layer attention arrays out in full; muse-glimmer
/// writes one number and means "every layer". Both spellings are legal GGUF
/// and the converters disagree, so read either rather than making the caller
/// know which file it has. A wrong-length array is still an error - silently
/// padding one would hand the KV allocator a geometry the weights don't have.
pub(super) fn key_u64_array_or_scalar(
    map: &MappedGguf,
    key: &str,
    n: usize,
) -> Result<Vec<u64>, LoadError> {
    match map.gguf().metadata.get(key) {
        Some(Value::Array(_)) => {
            let v = key_u64_array(map, key)?;
            if v.len() == n {
                Ok(v)
            } else {
                Err(LoadError::BadKey(format!(
                    "{key}: {} entries, want {n}",
                    v.len()
                )))
            }
        }
        Some(v) => {
            let s = v
                .as_u64()
                .ok_or_else(|| LoadError::BadKey(key.to_owned()))?;
            Ok(vec![s; n])
        }
        None => Err(LoadError::MissingKey(key.to_owned())),
    }
}

/// Which blocks are sliding-window, from either spelling of the pattern key.
///
/// gemma4 ships a per-layer bool array. muse-glimmer ships a scalar PERIOD
/// (4), and the phase matters: llama-hparams.cpp `set_swa_pattern(n, dense_first=false)`
/// is `is_swa[il] = il % n < (n - 1)`, i.e. the last layer of each group is
/// the full-attention one. That matches Muse Glimmer's config.json
/// layer_types exactly - [sliding, sliding, sliding, full] x13 over 52 layers.
/// Getting the phase inverted still loads and still decodes; it just answers
/// wrong, so this is pinned against the reference rather than inferred.
pub(super) fn swa_pattern(
    map: &MappedGguf,
    key: &str,
    n_layer: usize,
) -> Result<Vec<bool>, LoadError> {
    match map.gguf().metadata.get(key) {
        Some(Value::Array(_)) => {
            let v = key_bool_array(map, key)?;
            if v.len() == n_layer {
                Ok(v)
            } else {
                Err(LoadError::BadKey(format!(
                    "{key}: {} entries, want {n_layer}",
                    v.len()
                )))
            }
        }
        Some(v) => {
            let period =
                v.as_u64()
                    .ok_or_else(|| LoadError::BadKey(key.to_owned()))? as usize;
            if period == 0 {
                // llama.cpp treats period 0 as "every layer sliding"
                return Ok(vec![true; n_layer]);
            }
            Ok((0..n_layer).map(|il| il % period < period - 1).collect())
        }
        None => Err(LoadError::MissingKey(key.to_owned())),
    }
}

/// Plain-rope param tuple for the shared yarn-shaped kernels: ext_factor 0
/// disables the ramp, so only theta_scale (from the freq base) matters.
/// Pad a repacked Q8 weight's out dim (dims[1]) up to an `align` multiple
/// with zero rows. Out-major repack layout makes this one contiguous prefix
/// copy (data and scale). No-op when already aligned. Exact: zero rows
/// produce zero outputs, which the A4B's padded consumers cancel by
/// construction.
fn pad_ffn_out(exec: &GpuExecutor, w: RepackedQ8, align: usize) -> Result<RepackedQ8, GpuError> {
    let (in_dim, out) = (w.dims[0], w.dims[1]);
    let out_p = out.next_multiple_of(align);
    if out_p == out {
        return Ok(w);
    }
    let bpr = in_dim / 32; // blocks per out-row
    let mut data = exec.alloc_u8(out_p * bpr * 32)?;
    let mut scale = exec.alloc_u8(out_p * bpr * 2)?;
    let live = out * bpr;
    let sv = w.data.try_slice(0..live * 32).expect("pad src data");
    let mut dv = data.try_slice_mut(0..live * 32).expect("pad dst data");
    exec.stream
        .memcpy_dtod(&sv, &mut dv)
        .map_err(|e| GpuError::Driver(e.to_string()))?;
    let ss = w.scale.try_slice(0..live * 2).expect("pad src scale");
    let mut ds = scale.try_slice_mut(0..live * 2).expect("pad dst scale");
    exec.stream
        .memcpy_dtod(&ss, &mut ds)
        .map_err(|e| GpuError::Driver(e.to_string()))?;
    Ok(RepackedQ8 {
        data,
        scale,
        dims: vec![in_dim, out_p],
    })
}

/// Pad a repacked Q8 weight's in dim (dims[0], the K axis) up to an `align`
/// multiple with zero blocks at each row's tail. Row-strided copies (one per
/// out-row); zero K-blocks accumulate exactly 0.0.
fn pad_ffn_in(exec: &GpuExecutor, w: RepackedQ8, align: usize) -> Result<RepackedQ8, GpuError> {
    let (in_dim, out) = (w.dims[0], w.dims[1]);
    let in_p = in_dim.next_multiple_of(align);
    if in_p == in_dim {
        return Ok(w);
    }
    let (bpr, bpr_p) = (in_dim / 32, in_p / 32);
    let mut data = exec.alloc_u8(out * bpr_p * 32)?;
    let mut scale = exec.alloc_u8(out * bpr_p * 2)?;
    for r in 0..out {
        let sv = w
            .data
            .try_slice(r * bpr * 32..(r + 1) * bpr * 32)
            .expect("pad src");
        let mut dv = data
            .try_slice_mut(r * bpr_p * 32..r * bpr_p * 32 + bpr * 32)
            .expect("pad dst");
        exec.stream
            .memcpy_dtod(&sv, &mut dv)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        let ss = w
            .scale
            .try_slice(r * bpr * 2..(r + 1) * bpr * 2)
            .expect("pad src s");
        let mut ds = scale
            .try_slice_mut(r * bpr_p * 2..r * bpr_p * 2 + bpr * 2)
            .expect("pad dst s");
        exec.stream
            .memcpy_dtod(&ss, &mut ds)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
    }
    Ok(RepackedQ8 {
        data,
        scale,
        dims: vec![in_p, out],
    })
}

/// An FFN plane (the shared gate/up/down, the diffusion self-cond MLP) in
/// its file class, padded to the served width: rows (`pad_out`, gate/up) or
/// the K axis (down). The Q8 pads move repacked blocks on the device; a
/// k-quant plane is padded in its raw bytes and repacked once - its rows are
/// whole super-blocks (in = n_embd) for the row pad, and the K pad exists
/// only for the flat 32-weight types, which are the only ones the quantizer
/// puts on a ragged K.
pub(super) fn ffn_plane(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    pad_out: bool,
    align: usize,
) -> Result<Plane, LoadError> {
    let (info, bytes) = map.tensor_bytes(name).map_err(GpuError::from)?;
    let (in_dim, out) = (info.dims[0] as usize, info.dims[1] as usize);
    match plane_class(info.ggml_type) {
        Some(PlaneClass::Q8) => {
            let w = exec.repack_q8(map, name)?;
            Ok(Plane::Q8(if pad_out {
                pad_ffn_out(exec, w, align)?
            } else {
                pad_ffn_in(exec, w, align)?
            }))
        }
        Some(PlaneClass::Kq) => {
            let ty = info.ggml_type;
            let (_, raw_b, _) = crate::gpu::kq_params(ty).expect("k-quant class");
            if pad_out {
                let out_p = out.next_multiple_of(align);
                if out_p == out {
                    return Ok(Plane::Kq(exec.repack_kquant(map, name)?));
                }
                if !in_dim.is_multiple_of(256) {
                    return Err(LoadError::Tensor(
                        name.into(),
                        format!("{ty:?} row of {in_dim} is not whole super-blocks"),
                    ));
                }
                let row_b = in_dim / 256 * raw_b;
                let raw = kq_pad_out_raw(bytes, row_b, out, out_p);
                Ok(Plane::Kq(exec.repack_kquant_raw(
                    &raw,
                    vec![in_dim, out_p],
                    ty,
                    name,
                )?))
            } else {
                let in_p = in_dim.next_multiple_of(align);
                if in_p == in_dim {
                    return Ok(Plane::Kq(exec.repack_kquant(map, name)?));
                }
                if !crate::gpu::kq_flat32(ty) {
                    return Err(LoadError::Tensor(
                        name.into(),
                        format!("{ty:?} row of {in_dim} is not whole super-blocks"),
                    ));
                }
                let raw = kq_pad_in_raw_flat(bytes, raw_b / 8, in_dim, in_p, out);
                Ok(Plane::Kq(exec.repack_kquant_raw(
                    &raw,
                    vec![in_p, out],
                    ty,
                    name,
                )?))
            }
        }
        Some(PlaneClass::Bf16) => {
            if !exec.has_bf16_dense() {
                return Err(GpuError::MissingOp("bf16 dense plane lane").into());
            }
            if (if pad_out { out } else { in_dim }) % align != 0 {
                return Err(LoadError::Tensor(
                    name.into(),
                    format!("bf16 FFN plane at a ragged width ({in_dim} x {out})"),
                ));
            }
            Ok(Plane::Bf16(exec.upload_raw(map, name)?))
        }
        None => Err(GpuError::NoKernel {
            name: name.to_owned(),
            ty: info.ggml_type,
        }
        .into()),
    }
}

pub(crate) fn plain_rope(freq_base: f32, head_dim: usize) -> (f32, f32, f32, f32, f32, f32) {
    let theta_scale = freq_base.powf(-2.0 / head_dim as f32);
    (theta_scale, 1.0, 0.0, 1.0, 0.0, 1.0)
}

impl GpuGemma4 {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, LoadError> {
        Self::load_with(exec, map, max_ctx, None)
    }

    /// `load` plus explicit options the caller's config layer resolved -
    /// `fp8_native_dir` is an official-FP8/bf16 safetensors snapshot to source
    /// the e4m3 serving planes from (the runner's `fp8_native` config field /
    /// env / flag; the engine never reads the environment for product config).
    pub fn load_with(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
        fp8_native_dir: Option<&std::path::Path>,
    ) -> Result<Self, LoadError> {
        exec.vram_load_gate(map.total_len(), "gemma4")
            .map_err(LoadError::WontFit)?;
        // The unified prefill+decode tick is the DEFAULT: one forward beats
        // the mixed tick's two at every config measured, and never loses.
        // Filled only when unset; PADDOCK_NO_UNIFIED kills.
        if std::env::var_os("PADDOCK_UNIFIED").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_UNIFIED").is_none()
        {
            crate::envset::set_env("PADDOCK_UNIFIED", "1");
        }
        // FA-lite spec-verify attention (a B200 kernel, ported to sm_120
        // behind the launcher smem guard): SWA layers ride the f16-mma tile,
        // oversized global-layer geometries fall through to the tuned GQA
        // walk. Wins on GB202 at spec widths and harms no other width;
        // f16-score class, so it is quality-gated. Kill:
        // PADDOCK_NO_SPEC_FA.
        if std::env::var_os("PADDOCK_SPEC_FA").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_SPEC_FA").is_none()
        {
            crate::envset::set_env("PADDOCK_SPEC_FA", "1");
        }
        // F8CUT (the vendored cutlass GEMM) is the gemma4 DEFAULT at cc-10,
        // BATCH-GATED at m>=16 (f8cut_minb floor). It fixes the narrow-N
        // o/down UNDER-OCCUPANCY - 42 CTAs on a 148-SM die at 0.56 TB/s,
        // where the sm100 cutlass tiles sit in a 2.7-6.7 TB/s band.
        // The gate is why a BLANKET (m>=1) default is not shippable: it
        // regresses the low rungs, because cutlass then fires at DECODE
        // (m<16) where the fused tc5r route wins. Under the m>=16 floor
        // decode/small ticks stay tc5r and only wide ticks take cutlass.
        // Cost: the flat twins are persistent VRAM (o/down/gu ~doubles the
        // f8t linear planes to ~55 GiB) - which a 96 GB-class card absorbs at
        // max-batch 32 / max-ctx 4096. qkv-concat cutlass stays OPT-IN
        // (PADDOCK_F8CUT_QKV, env-dead). Kills: PADDOCK_NO_F8CUT (whole
        // route), PADDOCK_F8CUT_MINB (raise/lower the gate). cuBLASLt is a
        // dead end here (per-row/OUTER_VEC fp8 NOT_SUPPORTED on sm_100).
        //
        // SPEC LANE: attach_mtp WIDTH-GATES the intercept
        // (set_f8cut_spec_minb(32)) rather than killing it outright - a
        // narrow ~18-row verify loses on cutlass and stays on tc5r below the
        // floor, while a wide ~96-row verify wins on it.
        // PADDOCK_SPEC_F8CUT=0 restores the blanket kill; =1 removes the
        // floor. Gauge spec changes on verify_ms, not throughput: spec
        // throughput swings ~14% run-to-run on acceptance-rate noise alone.
        // FOLLOW-UP: gate the twin BUILD on max_batch AND on the drafter
        // (needs threading into load_with) so narrow / spec deployments
        // don't pay the twin VRAM.
        // (the spec live-cap / row-budget raises live in attach_mtp - they
        // only apply when the drafter is actually attached)
        // Which of the two architectures this folder serves. Picked from the
        // file, never from a flag or the filename - see Arch's doc comment for
        // why Muse Glimmer lives here rather than in a family of its own.
        let arch = match map.gguf().architecture() {
            Some("gemma4") => Arch::Gemma4,
            Some("muse-glimmer") => Arch::MuseGlimmer,
            Some("diffusion-gemma") => Arch::DiffusionGemma,
            other => {
                return Err(LoadError::BadKey(format!(
                    "general.architecture {other:?} is not served by this family"
                )));
            }
        };
        let ak = arch.key();
        // muse-glimmer on sm_100: the tuned configuration is the default - an
        // operator should not need a ten-env incantation to get the numbers
        // this pair can do. Every lever below was measured on this exact
        // die/model pair and quality-gated.
        // One block owns the whole set; each key fills only when absent, so
        // any explicit setting still wins and FOO=0 reverts a single lever
        // (every reader is truthy via envset::env_on / pd_env_on).
        // PADDOCK_MUSE_NO_TUNED_DEFAULTS=1 skips the block for bare-engine
        // A/B. cc-gated: the cutlass tile arms are sm_100a images and the
        // tick numbers are B200 pass-wall measurements. PADDOCK_PF_RUNS
        // additionally needs pack >= 0.17 - the batched-runs v4 arm reads a
        // run table older kernel bodies don't have, and because the exports
        // match, a stale pack silently makes every row attend slot 0's KV.
        // So the DEFAULT never arms it on a stale .so; setting it explicitly
        // stays the operator's own risk, unchanged.
        if matches!(arch, Arch::MuseGlimmer)
            && exec.compute_capability().0 == 10
            && paddock_models::dev_var_os!("PADDOCK_MUSE_NO_TUNED_DEFAULTS").is_none()
        {
            let pf_runs_pack = exec.pack_version() >= [0, 17, 0];
            let defaults: &[(&str, &str, bool)] = &[
                // one fat mixed pass per admission wave (4x82ms
                // serial passes -> 1x233ms; riders vanish, ITL to parity)
                ("PADDOCK_G4_TICK_ROWS", "8192", true),
                ("PADDOCK_MAX_CHUNKS", "32", true),
                // batched-runs prefill attends (attention 40 -> 9.1ms/pass)
                ("PADDOCK_PF_RUNS", "1", pf_runs_pack),
                // bf16-D stream class (glu2 19.9 -> 13.4ms)
                ("PADDOCK_PF_B16", "1", true),
                // cutlass tile arms tuned on the muse wave shapes (m=5984,
                // gu n=39936); C141 is a muse-only win - it FALSIFIED on
                // gemma, which is one reason this block is arch-gated
                ("PADDOCK_F8CUT_C141", "1", true),
                ("PADDOCK_F8CUT_N256", "1", true),
                ("PADDOCK_F8CUT_M256", "1", true),
                // the both-wide 256x256 tile: an isolated EVT bf16-D epilogue
                // beats M256 on every muse shape at wave-class m
                // (band -13.5%, down -20%, ~4.2 PF). =0 reverts to M256.
                ("PADDOCK_F8CUT_BIG", "1", true),
                // fused-qkv flat twin (qkv off the tc5r K-split at wave m)
                ("PADDOCK_F8CUT", "1", true),
                ("PADDOCK_F8CUT_QKV", "1", true),
                // o-gate e4m3 tile plane + f8row lm_head: the elected muse
                // numerics lanes, quality-gated
                ("PADDOCK_MUSE_OGATE_F8T", "1", true),
                ("PADDOCK_MUSE_HEAD_F8ROW", "1", true),
                // slot 458 Q16xKv128 decode attention. Wins on the nospec
                // lane at most widths, and on the spec lane it is what
                // carries the mid widths.
                //
                // Note the shape of the quality gate this needs: an
                // ACCEPTANCE battery cannot answer it, because
                // attn_fmha16_arm requires spec_k1.is_none() and so never runs
                // during a verify tick - a paired battery at pinned k=8 came
                // back bit-identical (pooled 0.4136060100166945 both arms)
                // because neither arm ran it. The real question its numerics
                // pose (K/V exact, Q and P on bf16 big+residual, relRMS
                // 2.5e-6) is a DENSE-path output one, and that gate passes:
                // every objectively checkable answer identical and correct in
                // both arms (primes, 1234x5678 = 7,006,652, Canberra, a sort,
                // 366), with divergence only on free-choice prompts where the
                // two answers were equally valid.
                ("PADDOCK_ATTN_FMHA16", "1", true),
                // Admission hold: let a synchronized burst finish ARRIVING
                // before the first prefill pass starts. Without it a
                // 32-request burst prefills as 4 + 28 ([pfchunk] r=536
                // runs=4, then r=3752 runs=28), so 28 of 32 requests wait
                // behind a 536-row pass that cannot amortize the ~27 ms
                // per-pass fixed cost - and that split is the bimodal TTFT.
                //
                // The knob SATURATES: 5 ms already captures the arrival
                // spread and anything longer is pure delay. Set per-model,
                // not globally - an unmeasured value on gemma/qwen is exactly
                // the mistake PADDOCK_MAX_CHUNKS and the spec engagement cap
                // both made. The hold only engages when nothing is decoding
                // and no chunk is in flight, so steady-state serving never
                // sees it.
                ("PADDOCK_ADM_WINDOW_MS", "5", true),
            ];
            for &(k, v, on) in defaults {
                if on && std::env::var_os(k).is_none() {
                    crate::envset::set_env(k, v);
                }
            }
            if !pf_runs_pack {
                tracing::warn!(
                    pack = ?exec.pack_version(),
                    "muse tuned defaults: kernel pack predates the batched-runs \
                     v4 arm (needs >= 0.17) - PF_RUNS left off"
                );
            }
            tracing::info!(
                "muse-glimmer/sm_100: tuned defaults applied \
                 (FOO=0 reverts one lever; PADDOCK_MUSE_NO_TUNED_DEFAULTS=1 \
                 restores the bare engine)"
            );
        }
        // gemma4 on sm_100: the wide nospec wave-stack levers are the
        // default. On top of the batch-gated F8CUT above, PF_RUNS + PF_B16
        // (batched prefill attends + bf16-D chunk streams) lift wide-batch
        // throughput and cut TTFT, and the qkv flat twin at a 1024-row
        // routing floor adds a little more. Worth re-testing levers here
        // after any prefill-band speedup: a lever that harvested nothing
        // against a slow prefill band can turn positive once the band moves.
        // Spec-safe by construction: pf_runs_batched is gated to
        // decode_rows == 0; the spec lane's F8CUT exposure is width-gated at
        // 64 rows (sub-64 verify rides tc5r).
        // Same contract as the muse block: fill-only-when-absent (env wins,
        // FOO=0 reverts one lever), PF_RUNS needs pack >= 0.17 for the
        // batched-runs v4 arm. PADDOCK_G4_NO_TUNED_DEFAULTS=1 skips the block.
        if matches!(arch, Arch::Gemma4)
            && exec.compute_capability().0 == 10
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_TUNED_DEFAULTS").is_none()
        {
            let pf_runs_pack = exec.pack_version() >= [0, 17, 0];
            let defaults: &[(&str, &str, bool)] = &[
                ("PADDOCK_PF_RUNS", "1", pf_runs_pack),
                ("PADDOCK_PF_B16", "1", true),
                // qkv-concat flat twin: the build condition reads F8CUT &&
                // F8CUT_QKV (load path below). ~+5.5 GiB twin VRAM.
                ("PADDOCK_F8CUT", "1", true),
                ("PADDOCK_F8CUT_QKV", "1", true),
                // A ~128-row spec verify tick is the regime the cutlass WIDE
                // arm (128x128 cluster-(2,1)) wins - isolated trunk band
                // 6.53ms vs narrow 8.16 / tc5r 10.13, crossover m>=96 on
                // every plane. WIDEB=96 lowers the pack's wide floor
                // (default 1024) and QKV_MINB=96 routes the qkv twin at
                // verify widths. Below 96 rows nothing changes (spec floor
                // 32 still gates the intercept; sub-96 stays narrow/tc5r).
                ("PADDOCK_QMOE_SORTED_MIN", "64", true),
                // hibatch lane: per-128 activation scales (head_xg+mma2g) +
                // bf16 partials (down PBF16 + tail bf16). Engages only at
                // r>=48 (MIN_ROWS) so narrow widths are untouched by
                // construction; quality-checked at 16/16 batched temp-0
                // outputs byte-identical.
                // FOO=0 reverts one lever (elections parse the value).
                ("PADDOCK_HIBATCH_XG", "1", true),
                ("PADDOCK_HIBATCH_PARTBF16", "1", true),
                // ^ the v2 ring pair + combine_init beats dec2 at the 64-pair
                // band too, so QMOE_SORTED_MIN drops to 64. Same exact-class
                // sorted math the >=128 band ships; the boundary is a tuning
                // param.
                ("PADDOCK_F8CUT_QKV_MINB", "96", true),
                ("PADDOCK_F8CUT_WIDEB", "96", true),
                // b16-D verify election - the wide-tick body's o/gu/down ride
                // pd_f8cut_gemm_b16 + the p16/b16 consumer twins (batch.rs
                // vb16_on, row floor 65: only the wide 128/160-row spec
                // verify widths engage; every nospec tick and the sub-64
                // spec verifies are out by construction). Acceptance rate is
                // unchanged, which is the gate that matters here.
                ("PADDOCK_G4_VB16", "1", true),
                // qkv joins the b16-D verify election - the last f32-D plane
                // in the verify tick rides f8cut_gemm_b16, read by the
                // packed-bf16 nra3 twin (pack slot 420, gated on has_ so a
                // stale .so reverts to f32). qkv is only ~15% of the elected
                // width, so the win is small and the point is uniformity.
                ("PADDOCK_G4_VB16Q", "1", true),
                // g2 token-major GU + dual-output align (slots 504/505),
                // decode widths only (r*k <= G2_MAX=2048). Exact Q8 math;
                // no regression at the wider guard widths.
                // PADDOCK_MOE_G2=0 reverts.
                ("PADDOCK_MOE_G2", "1", true),
            ];
            for &(k, v, on) in defaults {
                if on && std::env::var_os(k).is_none() {
                    crate::envset::set_env(k, v);
                }
            }
            if !pf_runs_pack {
                tracing::warn!(
                    pack = ?exec.pack_version(),
                    "gemma4 tuned defaults: kernel pack predates the \
                     batched-runs v4 arm (needs >= 0.17) - PF_RUNS left off"
                );
            }
            tracing::info!(
                "gemma4/sm_100: tuned nospec defaults applied (PF_RUNS + \
                 PF_B16; FOO=0 reverts one lever; \
                 PADDOCK_G4_NO_TUNED_DEFAULTS=1 restores the bare engine)"
            );
        }
        // muse-glimmer text lane, complete as of the SiLU pass:
        // header/Hparams, NoPE globals, untied LM head, logit scale, the 1e-8
        // post-norm epsilon at every unfused site; the attention output gate -
        // loaded and applied in all three lanes (serial step, prefill chunk,
        // batched walk), with the fused-norm and f16-attn arms that would
        // break it routed around (LayerWeights::fused_norm_ok /
        // f16_attn_ok) and the fused addnorm_e4m3_* two-norm/one-epsilon
        // lanes routed around (Hparams::fused_two_norm_ok); PER-TENSOR QUANT
        // DISPATCH, so the file's bf16 token_embd / output / attn_k / attn_v
        // serve as bf16 instead of being down-quantized on the way in
        // (`Plane`, and the arms that can only eat Q8 gated on
        // LayerWeights::kv_q8); the embedding preamble, the reference's
        // unweighted RMSNorm rather than gemma4's sqrt(n_embd) scale
        // (Hparams::embd_scale + GpuGemma4::embd_preamble); and the FFN
        // activation, SiLU here where gemma4 is GELU - every carrier in the
        // chain now ships both instantiations of one `pd_glu_act` template
        // and the arch picks one (Hparams::glu_act / glu_act_of, and the
        // has_* predicates below take the act so a pack missing a SiLU twin
        // cannot elect a lane that would serve the wrong nonlinearity); and
        // the ROPE PAIR LAYOUT - muse-glimmer is ROPE_TYPE_NORM (interleaved
        // 2k/2k+1) where gemma4 is NEOX (half-split), so both rope carriers
        // grew the twin the plain yarn rope already had (Hparams::rope_neox).
        //
        // Closed against llama.cpp on the identical GGUF (the same-weights
        // oracle). Four more constants were wrong, and they are worth naming
        // because none of them lives in a metadata key and none of them
        // errors:
        //
        //  1. The ATTENTION SCALE. gemma4 pins f_attention_scale = 1.0 (its
        //     query scale is folded into attn_q_norm at conversion), so every
        //     attention call site in this family passed a literal 1.0.
        //     muse-glimmer.cpp computes kq_scale = 1/sqrt(n_embd_head) and
        //     passes it ALONGSIDE its own synthesized q-norm weights, which
        //     absorb a different constant (qk_scale_factor). See
        //     Hparams::attn_scale.
        //  2. V is not NORMED. gemma4 runs a weightless per-head RMS norm on
        //     Vcur; muse-glimmer hands the raw Vcur to build_attn. See
        //     Hparams::v_norm - every carrier that touches the V slots now
        //     takes it.
        //  3. The FUSED KV APPEND ROPED NEOX. pd_kv_nra_rows had no pair-layout
        //     argument, so prefill roped K half-split while Q rode the
        //     interleaved layout the rope pass had just landed.
        //  4. The BATCHED-DECODE EPILOGUE had no freq_scale at all.
        //     pd_gemma_qkv_nra assumed 1.0, so this arch's NoPE full-attention
        //     layers were re-roped on every generated token while prefill
        //     correctly left them alone. That is why the symptom read as
        //     "right on the prompt, drifting after a few tokens", and why
        //     prefill-only greedy matched the reference exactly while normal
        //     decode did not.
        //
        // Where it stands: 8/12 of the greedy battery is TOKEN-EXACT with the
        // reference at 32 tokens on the default (fp8-KV) config, and all four
        // remaining divergences are the model's own near-ties - scored at the
        // fork, the reference's runner-up sits at 66-94% of its leader and
        // both engines produce the same top-3 within ~0.05 probability. The
        // two engines break those ties differently on the same Q8_K_XL file.
        //
        let n_layer = key_u64(map, &format!("{ak}.block_count"))? as usize;
        let n_embd = key_u64(map, &format!("{ak}.embedding_length"))? as usize;
        let n_head = key_u64(map, &format!("{ak}.attention.head_count"))? as usize;
        // Which class the file's dense planes are in, read off the tensor
        // types before anything is uploaded: the fp8 twin builders and the
        // Q8 reclaim convert FROM repacked Q8 and stay off on a k-quant file
        // (its planes serve as loaded, on the W4A8 lanes), and the shared
        // FFN's pad alignment depends on it.
        let class_of = |name: &str| -> Option<PlaneClass> {
            map.tensor_info(name).and_then(|t| plane_class(t.ggml_type))
        };
        let dense_planes: Vec<String> = (0..n_layer)
            .flat_map(|i| {
                [
                    "attn_q",
                    "attn_k",
                    "attn_v",
                    "attn_output",
                    "attn_gate",
                    "ffn_gate",
                    "ffn_up",
                    "ffn_down",
                ]
                .into_iter()
                .map(move |t| format!("blk.{i}.{t}.weight"))
            })
            .chain(
                [
                    "token_embd.weight",
                    "output.weight",
                    "self_cond_gate.weight",
                    "self_cond_up.weight",
                    "self_cond_down.weight",
                ]
                .map(str::to_string),
            )
            .collect();
        let file_kq = dense_planes
            .iter()
            .any(|n| class_of(n) == Some(PlaneClass::Kq));
        let ffn_kq = (0..n_layer).any(|i| {
            ["ffn_gate", "ffn_up", "ffn_down"]
                .iter()
                .any(|t| class_of(&format!("blk.{i}.{t}.weight")) == Some(PlaneClass::Kq))
        });
        if file_kq && !exec.has_kquant() {
            return Err(LoadError::Gpu(GpuError::MissingOp(
                "k-quant lanes (the file carries k-quant planes)",
            )));
        }
        // SERVED shared-FFN width: the repacks pad ragged widths up to the
        // alignment the lanes need - 128 (the fp8 ladder's tile) on a Q8
        // file, 256 (a whole k-quant super-block, so the tile GEMM applies
        // to every plane) when the shared FFN is k-quant (A4B: 2112 -> 2176
        // / 2304; dense models unchanged). Every scratch plane and arm sizes
        // from this - the raw metadata value would under-size them against
        // dims[1].
        let ffn_align = if ffn_kq { 256 } else { 128 };
        let n_ff = (key_u64(map, &format!("{ak}.feed_forward_length"))? as usize)
            .next_multiple_of(ffn_align);
        let hd_global = key_u64(map, &format!("{ak}.attention.key_length"))? as usize;
        // gemma4 sizes its SWA heads independently; muse-glimmer has one head
        // dim for both classes and omits the _swa key entirely.
        let hd_swa = match key_u64(map, &format!("{ak}.attention.key_length_swa")) {
            Ok(v) => v as usize,
            Err(_) => hd_global,
        };
        let swa_window = key_u64(map, &format!("{ak}.attention.sliding_window"))? as usize;
        let eps = key_f32(map, &format!("{ak}.attention.layer_norm_rms_epsilon"), None)?;
        let base_global = key_f32(map, &format!("{ak}.rope.freq_base"), Some(1_000_000.0))?;
        // muse-glimmer writes one rope base and applies it to the SWA layers
        // (its global layers are NoPE), so default the SWA base to the global
        // one there rather than to gemma4's 10k.
        let base_swa = key_f32(
            map,
            &format!("{ak}.rope.freq_base_swa"),
            Some(match arch {
                Arch::Gemma4 | Arch::DiffusionGemma => 10_000.0,
                Arch::MuseGlimmer => base_global,
            }),
        )?;
        let final_softcap = key_f32(map, &format!("{ak}.final_logit_softcapping"), Some(0.0))?;
        let logit_scale = key_f32(map, &format!("{ak}.logit_scale"), Some(1.0))?;
        // The canvas width a block-diffusion file denoises in. The key has
        // no arch prefix (`diffusion.canvas_length`, the converter's
        // spelling); required on the diffusion arch, meaningless elsewhere.
        let canvas_len = if arch.is_diffusion() {
            let w = key_u64(map, "diffusion.canvas_length")? as usize;
            if w == 0 || w > 1024 {
                return Err(LoadError::BadKey(format!(
                    "diffusion.canvas_length {w}: this engine denoises 1..=1024 positions"
                )));
            }
            w
        } else {
            0
        };

        // gemma4 spells these per-layer; muse-glimmer spells them as a scalar
        // window PERIOD and a scalar KV head count. Both readers below accept
        // either and hard-fail on a wrong-length array.
        let is_swa = swa_pattern(
            map,
            &format!("{ak}.attention.sliding_window_pattern"),
            n_layer,
        )?;
        let n_kv = key_u64_array_or_scalar(map, &format!("{ak}.attention.head_count_kv"), n_layer)?;
        // per-model check, not per-release: cross-layer KV reuse (the MTP
        // drafter's shared_kv_layers) isn't built yet - fail loud, not wrong
        if key_u64(map, &format!("{ak}.attention.shared_kv_layers")).unwrap_or(0) != 0 {
            return Err(LoadError::BadKey(format!(
                "{ak}.attention.shared_kv_layers != 0 (cross-layer KV reuse not implemented)"
            )));
        }
        // 26B-A4B hybrid MoE geometry (128 experts / 8 used, fused
        // gate_up_exps, router-on-attn_out) - all zero on dense variants.
        // Loader + forward.
        let n_expert = key_u64(map, &format!("{ak}.expert_count")).unwrap_or(0) as usize;
        let n_expert_used = key_u64(map, &format!("{ak}.expert_used_count")).unwrap_or(0) as usize;
        let ff_exp =
            key_u64(map, &format!("{ak}.expert_feed_forward_length")).unwrap_or(0) as usize;
        if n_expert != 0
            && (n_expert_used == 0
                || n_expert_used > 8 // down-combine kernel maps slot -> warp (8 warps)
                || ff_exp == 0
                || !ff_exp.is_multiple_of(32) // Q8 block / dp4a int4-load contract
                || !n_embd.is_multiple_of(32))
        {
            return Err(LoadError::BadKey(
                "gemma4 MoE keys incomplete/unsupported".into(),
            ));
        }

        // vision splice markers - a one-time vocab scan; absent on text-only
        // vocab variants. The marker PAIR is per-arch (see `image_markers`):
        // gemma4v brackets with <|image>/<image|>, muse-glimmer with
        // <|image_start|>/<|image_end|>.
        let (beg_tok, end_tok) = super::image_markers(arch);
        let (mut img_beg_id, mut img_end_id) = (None, None);
        if let Some(Value::Array(toks)) = map.gguf().metadata.get("tokenizer.ggml.tokens") {
            for (i, t) in toks.iter().enumerate() {
                match t.as_str() {
                    Some(s) if s == beg_tok => img_beg_id = Some(i as u32),
                    Some(s) if s == end_tok => img_end_id = Some(i as u32),
                    _ => {}
                }
            }
        }
        // VRAM ledger: phase-boundary deltas from cuMemGetInfo so the
        // footprint is attributable. The weights live in up to three serving
        // classes (Q8 originals + f8w prefill + f8t decode planes) and the
        // split is invisible in nvidia-smi/NVML - this is the honest
        // will-it-fit breakdown. eprintln to match the loader's other
        // progress lines.
        let mut vram_prev: i64 = exec.process_mem_used().unwrap_or(0) as i64;
        // `held` is the pool's RESERVED high-water: bytes taken from the
        // driver, live or not. Tracking it beside live is what localises
        // stranded memory to a phase -- live alone cannot, because a transient
        // that inflates the pool and then frees leaves live unchanged and
        // reserved permanently higher. The phase where held grows faster than
        // live is the one that stranded the bytes.
        let vram_mark = |tag: &str, prev: &mut i64| {
            if let Some(b) = exec.process_mem_used() {
                let d = b as i64 - *prev;
                let gib = |x: f64| x / (1u64 << 30) as f64;
                let held = exec.pool_reserved_bytes().unwrap_or(0);
                tracing::info!(
                    "gemma4 vram: {tag:<28} {:+8.2} GiB  (total {:7.2} GiB | held {:7.2} \
                     | not-live {:6.2})",
                    gib(d as f64),
                    gib(b as f64),
                    gib(held as f64),
                    gib(held.saturating_sub(b) as f64),
                );
                *prev = b as i64;
            }
        };
        // PER-TENSOR QUANT DISPATCH. UD files are MIXED: muse-glimmer's
        // UD-Q8_K_XL ships token_embd / output / attn_k / attn_v at bf16 next
        // to Q8_0 everything else; the A4B Q4_K_M holds Q4_K attention beside
        // Q6_K attn_v and Q5_0 / Q8_0 downs. The project's seam for that is
        // the TENSOR, not the model, and the correctness spine is same-weights
        // parity on the identical GGUF - so a tensor keeps its class instead
        // of being requantized into another lane on the way in. Anything
        // outside the three classes still lands in a loud NoKernel.
        let plane = |name: &str| -> Result<Plane, GpuError> {
            let (info, _) = map.tensor_bytes(name)?;
            match plane_class(info.ggml_type) {
                Some(PlaneClass::Q8) => Ok(Plane::Q8(exec.repack_q8(map, name)?)),
                Some(PlaneClass::Bf16) => {
                    if !exec.has_bf16_dense() {
                        return Err(GpuError::MissingOp("bf16 dense plane lane"));
                    }
                    Ok(Plane::Bf16(exec.upload_raw(map, name)?))
                }
                Some(PlaneClass::Kq) => Ok(Plane::Kq(exec.repack_kquant(map, name)?)),
                None => Err(GpuError::NoKernel {
                    name: name.to_owned(),
                    ty: info.ggml_type,
                }),
            }
        };
        let ffn_plane = |name: &str, pad_out: bool| -> Result<Plane, LoadError> {
            ffn_plane(&exec, map, name, pad_out, ffn_align)
        };
        // The raw embedding table for row gathers. A k-quant embedding keeps
        // no raw copy: its repacked head is the table (planes::EmbdTable).
        let embd_info = map
            .tensor_info("token_embd.weight")
            .ok_or_else(|| LoadError::Tensor("token_embd.weight".into(), "missing".into()))?;
        let n_vocab = embd_info.dims[1] as usize;
        let token_embd = match plane_class(embd_info.ggml_type) {
            Some(PlaneClass::Kq) => None,
            _ => Some(exec.upload_raw(map, "token_embd.weight")?),
        };
        // The LM head. gemma4 TIES it to the embedding (no `output.weight` in
        // the file, so the repacked embedding doubles as the head); muse-glimmer
        // does not (config tie_word_embeddings=false, and `output.weight` is a
        // real 6656x202048 bf16 plane). Tying them there would be a quiet
        // accuracy bug, so the tensor decides, never the arch - and the plane
        // carries its own class, so a bf16 head serves as bf16.
        let head = match map.tensor_bytes("output.weight") {
            Ok(_) => plane("output.weight")?,
            Err(_) => plane("token_embd.weight")?,
        };
        if token_embd.is_none() && head.kq().is_none() {
            // an UNTIED head over a k-quant embedding: the gathers would
            // need a second repacked table; no file ships the pairing today
            return Err(LoadError::Tensor(
                "token_embd.weight".into(),
                format!(
                    "{:?} embedding under a separate output.weight - not served",
                    embd_info.ggml_type
                ),
            ));
        }
        if head.dims()[1] != n_vocab {
            return Err(LoadError::BadKey(format!(
                "lm head out dim {} != vocab {n_vocab}",
                head.dims()[1]
            )));
        }
        let output_norm = exec.upload(map, "output_norm.weight")?.buf;
        // Weight vector for muse-glimmer's UNWEIGHTED embedding RMSNorm. The
        // rmsnorm kernels are `x * inv_rms * w`, so all-ones is the reference's
        // weightless norm exactly - cheaper to keep true than a second kernel
        // that would have to track this one. n_embd floats; None on gemma4.
        let embd_ones = if arch == Arch::MuseGlimmer {
            Some(
                exec.stream
                    .clone_htod(&vec![1.0f32; n_embd])
                    .map_err(|e| LoadError::Tensor("embd_ones".into(), e.to_string()))?,
            )
        } else {
            None
        };
        // gemma4 ships per-frequency rope divisors and applies them on the
        // GLOBAL layers only (forward.rs: `(!lw.is_swa).then_some(..)`).
        // muse-glimmer has no such tensor - and could not use one, since its
        // global layers are NoPE. Stand in a ones vector so the shared
        // `theta / factors[k]` path stays an identity if it is ever reached.
        let rope_factors = match map.tensor_bytes("rope_freqs.weight") {
            Ok(_) => exec.upload(map, "rope_freqs.weight")?.buf,
            Err(_) => {
                let mut b = exec.alloc(hd_global / 2)?;
                exec.upload_f32(&vec![1.0f32; hd_global / 2], &mut b)?;
                b
            }
        };

        let norm = |name: String| -> Result<CudaSlice<f32>, LoadError> {
            Ok(exec.upload(map, &name)?.buf)
        };
        // host-side f32 vector read (router scale planes are small F32 tensors)
        let host_f32 = |name: String| -> Result<Vec<f32>, LoadError> {
            let (_, bytes) = map
                .tensor_bytes(&name)
                .map_err(|_| LoadError::BadKey(format!("{name} missing")))?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        // device-to-device byte-range copy for the fused gate_up split
        let d2d = |src: &CudaSlice<u8>,
                   so: usize,
                   dst: &mut CudaSlice<u8>,
                   doff: usize,
                   len: usize|
         -> Result<(), LoadError> {
            let sv = src
                .try_slice(so..so + len)
                .ok_or_else(|| LoadError::BadKey("moe split src range".into()))?;
            let mut dv = dst
                .try_slice_mut(doff..doff + len)
                .ok_or_else(|| LoadError::BadKey("moe split dst range".into()))?;
            exec.stream
                .memcpy_dtod(&sv, &mut dv)
                .map_err(|e| LoadError::BadKey(e.to_string()))?;
            Ok(())
        };

        let mut layers = Vec::with_capacity(n_layer);
        // A4B: e4m3 MoE expert duplicates (gu_f8/dn_f8) build inside this
        // loop, so their bytes land in the "q8 originals" bracket below -
        // and without this tally ~23 GiB gets reported under the wrong label
        // on the 26B-A4B. Count them here so the ledger can name them.
        let mut moe_f8_dup_bytes: u64 = 0;
        // Hoisted above the layer loop so the attention seats can be SKIPPED
        // at construction time rather than uploaded and stubbed later. Both
        // depend only on env and device capability -- nothing the loop
        // produces -- so they are the same values the phase gates below use;
        // the bindings are moved, not copied, so there is nothing to drift.
        // Every fp8 twin lane converts FROM repacked Q8 - none of them
        // engages on a file with k-quant planes (`file_kq`), which serve as
        // loaded on the W4A8 lanes.
        let f8_pf_on = !file_kq
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8ROW").is_none()
            && (paddock_models::dev_var_os!("PADDOCK_G4_F8ROW").is_some()
                || exec.compute_capability().0 == 10);
        let qkvfuse = paddock_models::dev_var_os!("PADDOCK_G4_NO_QKVFUSE").is_none()
            && exec.has_gemma_qkv_nra2s();
        // NEVER-UPLOAD for the attention planes.
        //
        // f8a replaces wq/wk/wv/wo with e4m3 twins and stubs the Q8_0 sources.
        // Uploading them first and freeing them per tensor is what strands the
        // memory: each hole lands under the twin allocated above it and the
        // pool cannot return a block with a live allocation in it. Measured on
        // gemma-4-31B, `PADDOCK_G4_NO_F8A=1` takes retained-not-live 4.95 ->
        // 0.89 GB, so the whole remainder is this.
        //
        // The twins are built from `map`, so the sources never have to be
        // resident at all. What has to be true for skipping to be safe is that
        // no earlier phase reads them:
        //   f8w prefill planes  (f8w_pf)  \
        //   f8t decode planes   (f8t_dec)  > all three require f8_pf_on
        //   f8row planes        (f8row)   /
        // so `!f8_pf_on` rules out all three at once. f8a's own gate already
        // demands `!f8row && !f8w_pf`, and the fp4 / fp8-native routes source
        // their bytes elsewhere, so both are excluded here too.
        //
        // Per LAYER this still needs the fused-qkv branch, because the split
        // branch (a layer whose k/v ship bf16) builds its planes from the
        // resident seats. `kv_q8` is decided by the file's tensor types, so it
        // is knowable here. Checked against reality after f8a rather than
        // trusted - a stubbed plane whose twin never got built is exactly the
        // silent corruption this seam has shipped twice already.
        let attn_never_upload = !f8_pf_on
            && qkvfuse
            && fp8_native_dir.is_none()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8A").is_none()
            && paddock_models::dev_var_os!("PADDOCK_G4_FP4").is_none()
            && exec.has_f8_gemm_w8()
            && exec.has_f8_gemv()
            && exec.has_f8_gemm_mma_ks();
        // file-type test, the same one `kv_q8()` makes of the loaded planes
        let file_q8 = |name: &str| -> bool {
            map.tensor_info(name)
                .is_some_and(|t| t.ggml_type == GgmlType::Q8_0)
        };
        let dims_of = |name: &str| -> Result<Vec<usize>, LoadError> {
            Ok(map
                .tensor_info(name)
                .ok_or_else(|| LoadError::Tensor(name.into(), "missing".into()))?
                .dims
                .iter()
                .map(|&d| d as usize)
                .collect())
        };
        // A seat with no plane behind it. Dims are real, so every `.dims()`
        // consumer and the f8a/reclaim stub tests behave exactly as they do
        // for a plane that was uploaded and then stubbed - the difference is
        // only that the bytes never occupied the pool.
        let stub_q8 =
            |exec: &GpuExecutor, dims: Vec<usize>| -> Result<crate::gpu::RepackedQ8, LoadError> {
                Ok(crate::gpu::RepackedQ8 {
                    data: exec
                        .alloc_u8(32)
                        .map_err(|e| LoadError::Tensor("attn seat stub".into(), e.to_string()))?,
                    scale: exec
                        .alloc_u8(32)
                        .map_err(|e| LoadError::Tensor("attn seat stub".into(), e.to_string()))?,
                    dims,
                })
            };
        let mut attn_skipped: Vec<bool> = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let swa = is_swa[i];
            let head_dim = if swa { hd_swa } else { hd_global };
            // V-less global layers: attn_v simply isn't in the file
            // This layer skips the attn upload only if it will take f8a's
            // FUSED branch, which needs k (and v, when present) in the Q8
            // class - the split branch reads the resident seats.
            let has_v = map.tensor_info(&format!("blk.{i}.attn_v.weight")).is_some();
            let skip_attn = attn_never_upload
                && file_q8(&format!("blk.{i}.attn_q.weight"))
                && file_q8(&format!("blk.{i}.attn_output.weight"))
                && file_q8(&format!("blk.{i}.attn_k.weight"))
                && (!has_v || file_q8(&format!("blk.{i}.attn_v.weight")));
            attn_skipped.push(skip_attn);
            let wv = if skip_attn {
                if has_v {
                    Some(Plane::Q8(stub_q8(
                        &exec,
                        dims_of(&format!("blk.{i}.attn_v.weight"))?,
                    )?))
                } else {
                    None
                }
            } else {
                match plane(&format!("blk.{i}.attn_v.weight")) {
                    Ok(t) => Some(t),
                    Err(GpuError::Map(_)) => None,
                    Err(e) => return Err(e.into()),
                }
            };
            if swa && wv.is_none() {
                return Err(LoadError::Tensor(
                    format!("blk.{i}.attn_v.weight"),
                    "missing on a sliding-window layer".into(),
                ));
            }
            // layer_output_scale is [1] f32 - read host-side, applied as a scalar
            let out_scale = match map.tensor_bytes(&format!("blk.{i}.layer_output_scale.weight")) {
                Ok((_, bytes)) => f32::from_le_bytes(bytes[..4].try_into().expect("f32 scalar")),
                Err(_) => 1.0,
            };
            // diffusion-gemma files carry a second scalar, `enc_layer_output_
            // scale`, next to this one - the converter writes the encoder
            // (causal) pass's copy separately although HF ties the two to
            // ONE `layer_scalar` per layer. Read off the shipped Q8_0 file
            // they are identical on all 30 layers, so the one graph serves
            // both passes. A checkpoint where they diverge would need a
            // per-pass scale in the layer walk; refuse rather than serve one
            // of the two silently.
            if let Ok((_, bytes)) =
                map.tensor_bytes(&format!("blk.{i}.enc_layer_output_scale.weight"))
            {
                let enc = f32::from_le_bytes(bytes[..4].try_into().expect("f32 scalar"));
                if enc != out_scale {
                    return Err(LoadError::Tensor(
                        format!("blk.{i}.enc_layer_output_scale.weight"),
                        format!(
                            "{enc} differs from layer_output_scale {out_scale}: this engine \
                             runs the encoder and the canvas through one scale"
                        ),
                    ));
                }
            }
            // 26B-A4B routed-expert group.
            // All folds here are load-time exact: see MoeWeights.
            let moe = if n_expert != 0 {
                // f32 (dequant-if-needed) - feeds the matvec_f32_batch router
                let router_w = exec.upload(map, &format!("blk.{i}.ffn_gate_inp.weight"))?;
                // reference chain rms_norm_unweighted(x)/sqrt(d) ⊙ s == one
                // weighted rmsnorm with gamma = s/sqrt(d)
                let mut gamma = host_f32(format!("blk.{i}.ffn_gate_inp.scale"))?;
                let inv_sqrt_d = 1.0f32 / (n_embd as f32).sqrt();
                for g in gamma.iter_mut() {
                    *g *= inv_sqrt_d;
                }
                let router_gamma = exec
                    .stream
                    .clone_htod(&gamma)
                    .map_err(|e| LoadError::BadKey(e.to_string()))?;
                // The expert trio takes ONE class per layer off the fused
                // gate_up's type: a Q8_0 file's experts stay on the s8 lanes
                // below; a k-quant file's ride the W4A8 expert lanes, with a
                // Q8_0 `ffn_down_exps` (the Q4_K_M's 14 such layers) repacked
                // as the flat Q8_0 seat so the layer's gate+up -> down
                // handshake stays inside one kernel family.
                let fused_name = format!("blk.{i}.ffn_gate_up_exps.weight");
                let down_name = format!("blk.{i}.ffn_down_exps.weight");
                let fused_ty = map
                    .tensor_info(&fused_name)
                    .ok_or_else(|| LoadError::Tensor(fused_name.clone(), "missing".into()))?
                    .ggml_type;
                if fused_ty != GgmlType::Q8_0 && crate::gpu::kq_params(fused_ty).is_none() {
                    return Err(GpuError::NoKernel {
                        name: fused_name,
                        ty: fused_ty,
                    }
                    .into());
                }
                let kq_experts = fused_ty != GgmlType::Q8_0;
                if kq_experts && !exec.has_kquant_moe() {
                    return Err(LoadError::Gpu(GpuError::MissingOp(
                        "k-quant MoE expert lanes (the file's experts are k-quant)",
                    )));
                }
                // fused [n_embd, 2*ff_exp, n_expert]: repack whole, split per
                // expert into gate rows [0,ff) / up rows [ff,2ff) - pure row
                // copies (blocks run along the input dim, and a row is whole
                // blocks - whole super-blocks on the k-quant streams, which
                // are position-independent), exact; the forward then sees the
                // qwen-MoE separate-plane layout.
                let dims = vec![n_embd, ff_exp, n_expert];
                let (experts, fused_q8) = if kq_experts {
                    let fused = exec.repack_kquant(map, &fused_name)?;
                    let (_, _, data_b) = crate::gpu::kq_params(fused.ty).expect("k-quant class");
                    let scb = crate::gpu::kq_scb(fused.ty);
                    let spr = n_embd / 256; // super-blocks per row
                    let half = ff_exp * spr; // one half-plane, per expert
                    let fr = 2 * half; // fused rows, per expert
                    let mut gdata = exec.alloc_u8(n_expert * half * data_b)?;
                    let mut gscale = exec.alloc_u8(n_expert * half * scb)?;
                    let mut udata = exec.alloc_u8(n_expert * half * data_b)?;
                    let mut uscale = exec.alloc_u8(n_expert * half * scb)?;
                    for e in 0..n_expert {
                        let sg = e * fr;
                        let su = sg + half;
                        let dm = e * half;
                        d2d(
                            &fused.data,
                            sg * data_b,
                            &mut gdata,
                            dm * data_b,
                            half * data_b,
                        )?;
                        d2d(
                            &fused.data,
                            su * data_b,
                            &mut udata,
                            dm * data_b,
                            half * data_b,
                        )?;
                        d2d(&fused.scales, sg * scb, &mut gscale, dm * scb, half * scb)?;
                        d2d(&fused.scales, su * scb, &mut uscale, dm * scb, half * scb)?;
                    }
                    let ty = fused.ty;
                    drop(fused);
                    // the down at its own width: a flat 32-weight type when
                    // the row is not whole super-blocks (704 on the A4B)
                    let down = match map.tensor_info(&down_name).map(|t| t.ggml_type) {
                        Some(GgmlType::Q8_0) => {
                            let (info, bytes) =
                                map.tensor_bytes(&down_name).map_err(GpuError::from)?;
                            let d: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
                            exec.repack_kquant_raw(bytes, d, GgmlType::Q8_0, &down_name)?
                        }
                        _ => exec.repack_kquant(map, &down_name)?,
                    };
                    (
                        ExpertPlanes::Kq {
                            gate: RepackedKQ {
                                data: gdata,
                                scales: gscale,
                                dims: dims.clone(),
                                ty,
                                act128: false,
                            },
                            up: RepackedKQ {
                                data: udata,
                                scales: uscale,
                                dims,
                                ty,
                                act128: false,
                            },
                            down,
                        },
                        None,
                    )
                } else {
                    let fused = exec.repack_q8(map, &fused_name)?;
                    let bpr = n_embd / 32; // Q8 blocks per row
                    let half_blocks = ff_exp * bpr; // one half-plane, per expert
                    let fr_blocks = 2 * half_blocks; // fused rows, per expert
                    let mut gdata = exec.alloc_u8(n_expert * half_blocks * 32)?;
                    let mut gscale = exec.alloc_u8(n_expert * half_blocks * 2)?;
                    let mut udata = exec.alloc_u8(n_expert * half_blocks * 32)?;
                    let mut uscale = exec.alloc_u8(n_expert * half_blocks * 2)?;
                    for e in 0..n_expert {
                        let sg = e * fr_blocks; // expert's gate rows (block idx)
                        let su = sg + half_blocks; // expert's up rows
                        let dm = e * half_blocks;
                        d2d(&fused.data, sg * 32, &mut gdata, dm * 32, half_blocks * 32)?;
                        d2d(&fused.data, su * 32, &mut udata, dm * 32, half_blocks * 32)?;
                        d2d(&fused.scale, sg * 2, &mut gscale, dm * 2, half_blocks * 2)?;
                        d2d(&fused.scale, su * 2, &mut uscale, dm * 2, half_blocks * 2)?;
                    }
                    (
                        ExpertPlanes::Q8 {
                            gate: RepackedQ8 {
                                data: gdata,
                                scale: gscale,
                                dims: dims.clone(),
                            },
                            up: RepackedQ8 {
                                data: udata,
                                scale: uscale,
                                dims,
                            },
                            down: exec.repack_q8(map, &down_name)?,
                        },
                        Some(fused),
                    )
                };
                // tcgen05 e4m3 expert planes (a4b-expert-tcgen05.md): the
                // FUSED repack converts as-is (1408 rows/expert = 11 exact
                // tiles - this is why the f8 lane keeps the fused layout the
                // GGUF shipped); the down stream K-pads 704 -> 768 with zero
                // blocks in the converter. Q8 originals stay for the decode
                // band. Kill: PADDOCK_G4_NO_MOE_F8.
                // e4m3 twins convert from the Q8 planes only - k-quant
                // experts serve as loaded
                let moe_f8 = fused_q8.is_some()
                    && exec.has_f8bs_moe()
                    && paddock_models::dev_var_os!("PADDOCK_G4_NO_MOE_F8").is_none()
                    && n_embd.is_multiple_of(128)
                    && (2 * ff_exp).is_multiple_of(128);
                let (gu_f8, dn_f8) = match (&fused_q8, experts.q8()) {
                    (Some(fused), Some((_, _, down_exps))) if moe_f8 => {
                        let ffp = ff_exp.next_multiple_of(128);
                        let gu = exec.q8_0_to_f8w(fused)?;
                        let dn = exec.q8_0_to_f8w_pad(down_exps, ff_exp / 32, ffp / 32)?;
                        moe_f8_dup_bytes +=
                            (gu.data.len() + gu.scale.len() + dn.data.len() + dn.scale.len())
                                as u64;
                        (Some(gu), Some(dn))
                    }
                    _ => (None, None),
                };
                // Flat-scale e4m3 expert planes (change A). Built
                // from the SPLIT halves so the row stream is (e*ff + o) - the
                // same indexing the sorted mma kernel already walks. OPT-IN
                // (PADDOCK_MOE_F8ROW=1) while the class is on trial: it is a
                // lossy requant AND a ~508 MB duplicate, since the decode
                // band still needs the q8 originals.
                let f8row_on = paddock_models::dev_var_os!("PADDOCK_MOE_F8ROW").is_some()
                    && exec.has_f8row_moe();
                let (gate_f8r, up_f8r) = match experts.q8() {
                    Some((gate_exps, up_exps, _)) if f8row_on => {
                        let rows = n_expert * ff_exp;
                        (
                            Some(exec.q8_0_to_f8row_rows(gate_exps, rows)?),
                            Some(exec.q8_0_to_f8row_rows(up_exps, rows)?),
                        )
                    }
                    _ => (None, None),
                };
                // The down half is separately killable (PADDOCK_MOE_F8ROW_DN=0)
                // so the two halves stay independently measurable -- the whole
                // point of this task is one change at a time.
                let down_f8r = match experts.q8() {
                    Some((_, _, down_exps))
                        if f8row_on
                            && exec.has_f8row_moe_down()
                            && paddock_models::dev_var!("PADDOCK_MOE_F8ROW_DN").as_deref()
                                != Ok("0") =>
                    {
                        Some(exec.q8_0_to_f8row_rows(down_exps, n_expert * n_embd)?)
                    }
                    _ => None,
                };
                drop(fused_q8); // the fused repack was staging only
                let down_scale_h = host_f32(format!("blk.{i}.ffn_down_exps.scale"))?;
                let down_scale = exec
                    .stream
                    .clone_htod(&down_scale_h)
                    .map_err(|e| LoadError::BadKey(e.to_string()))?;
                Some(MoeWeights {
                    router_w,
                    router_gamma,
                    experts,
                    down_scale,
                    gu_f8,
                    dn_f8,
                    gate_f8r,
                    up_f8r,
                    down_f8r,
                    pre_norm2: norm(format!("blk.{i}.pre_ffw_norm_2.weight"))?,
                    post_norm1: norm(format!("blk.{i}.post_ffw_norm_1.weight"))?,
                    post_norm2: norm(format!("blk.{i}.post_ffw_norm_2.weight"))?,
                })
            } else {
                None
            };
            // fp8 FFN planes (PADDOCK_G4_F8): built per layer right after
            // the q8 repack so peak VRAM stays bounded; the TMA GEMM path
            // needs PADDOCK_F8W8_TMA which load() sets below.
            let (f8_gate, f8_up, f8_down, f8_gu) = (None, None, None, None);
            layers.push(LayerWeights {
                moe,
                f8_gate,
                f8_gu,
                gu_il: false,
                gu_ws: None,
                qkv_ws: None,
                wo_ws: None,
                down_ws: None,
                fp4_gu: None,
                f8_up,
                f8_down,
                f8a_wq: None,
                f8a_wqkv: None,
                f8a_wk: None,
                f8a_wv: None,
                f8a_wo: None,
                f8_wq: None,
                f8_wk: None,
                f8_wv: None,
                f8_wo: None,
                f8r_gate: None,
                f8r_up: None,
                f8r_down: None,
                f8t_gu: None,
                f8t_gate: None,
                f8t_up: None,
                f8t_down: None,
                f8t_qkv: None,
                f8t_wq: None,
                f8t_wk: None,
                f8t_wv: None,
                f8t_wo: None,
                f8t_attn_gate: None,
                f8w_wq: None,
                f8w_wk: None,
                f8w_wv: None,
                f8w_wo: None,
                attn_norm: norm(format!("blk.{i}.attn_norm.weight"))?,
                wq: if skip_attn {
                    Plane::Q8(stub_q8(&exec, dims_of(&format!("blk.{i}.attn_q.weight"))?)?)
                } else {
                    plane(&format!("blk.{i}.attn_q.weight"))?
                },
                wk: if skip_attn {
                    Plane::Q8(stub_q8(&exec, dims_of(&format!("blk.{i}.attn_k.weight"))?)?)
                } else {
                    plane(&format!("blk.{i}.attn_k.weight"))?
                },
                wv,
                wo: if skip_attn {
                    Plane::Q8(stub_q8(
                        &exec,
                        dims_of(&format!("blk.{i}.attn_output.weight"))?,
                    )?)
                } else {
                    plane(&format!("blk.{i}.attn_output.weight"))?
                },
                // muse-glimmer only; gemma4 files have no such tensor
                attn_gate: match map.tensor_bytes(&format!("blk.{i}.attn_gate.weight")) {
                    Ok(_) => Some(plane(&format!("blk.{i}.attn_gate.weight"))?),
                    Err(_) => None,
                },
                q_norm: norm(format!("blk.{i}.attn_q_norm.weight"))?,
                k_norm: norm(format!("blk.{i}.attn_k_norm.weight"))?,
                attn_post_norm: norm(format!("blk.{i}.post_attention_norm.weight"))?,
                ffn_norm: norm(format!("blk.{i}.ffn_norm.weight"))?,
                // A4B shared-FFN width (2112) is ragged - the fp8 ladder (f8t
                // tile images, f8w TMA GEMMs) needs out%128, the k-quant tile
                // GEMM whole 256-super-blocks. Pad the planes once at load
                // (2176 / 2304, `ffn_align`): zero out-rows on gate/up, zero
                // K-tail blocks on down. Exact by construction (gelu(0)*0 = 0
                // feeds zero down columns; f32 x+0.0 == x), and every
                // consumer just reads dims[1]. Dense models (n_ff already
                // aligned) pass through untouched.
                ffn_gate: ffn_plane(&format!("blk.{i}.ffn_gate.weight"), true)?,
                ffn_up: ffn_plane(&format!("blk.{i}.ffn_up.weight"), true)?,
                ffn_down: ffn_plane(&format!("blk.{i}.ffn_down.weight"), false)?,
                ffn_post_norm: norm(format!("blk.{i}.post_ffw_norm.weight"))?,
                out_scale,
                is_swa: swa,
                n_kv_heads: n_kv[i] as usize,
                head_dim,
            });
        }
        vram_mark("q8 originals (upload+repack)", &mut vram_prev);
        if moe_f8_dup_bytes > 0 {
            // inside the bracket above, not additional to it - the e4m3
            // expert duplicates the decode band's q8 originals shadow
            tracing::info!(
                "gemma4 vram:   of which moe e4m3 duplicate planes (gu_f8+dn_f8) \
                 {:.2} GiB - q8 experts stay resident for the decode band",
                moe_f8_dup_bytes as f64 / (1u64 << 30) as f64
            );
        }

        // fp8 FFN plane build. Two modes:
        //   PADDOCK_G4_F8  - v1 duplicate (prefill r>1024 only; +20.8GB,
        //                    breaks 32 slots - long-ctx experiments only)
        //   PADDOCK_G4_F8R - REPLACE: every FFN lane goes e4m3 (gemv b1,
        //                    mma_ks twin 2..=31, TMA GEMM >=32) and the q8
        //                    trio is dropped to stubs -> VRAM-FLAT. Lossy
        //                    e4m3 class, quality-gated (temp-0 incl >1024-tok
        //                    prompt, drafter acceptance, long-gen coherence,
        //                    memcheck).
        // F8R is DEFAULT-ON when the pack ships the full ladder, where it
        // strictly dominates q8 at every config. Twin-less packs stay opt-in
        // (a TMA-only 4..31 band regresses narrow decode). Kill:
        // PADDOCK_G4_NO_F8R.
        // cc gate: F8R default-on holds only where the block-scale mma is
        // hardware (cc 12). On sm_100 the software fold regresses every
        // config badly - the per-k32 fold tax on CUDA cores squanders the
        // fast e4m3 pipe. sm_100 prefill rides the separate per-ROW-scaled
        // e4m3 planes instead (f8_attn/f8p below); explicit PADDOCK_G4_F8R
        // still forces for A/B.
        // fp4 weight-class cutover: e2m1 planes
        // REPLACE the e4m3 ones (VRAM-flat, ~-16GB) wherever both fused
        // planes are live; activations stay e4m3. Lossy - the full battery
        // gates the default. Call sites read PADDOCK_G4_FP4_ACTIVE so an
        // env-killed fusion can never mismatch lanes. Both fusion
        // predicates are hoisted here so one decision covers the attn and
        // FFN conversion blocks below.
        let gufuse = paddock_models::dev_var_os!("PADDOCK_G4_NO_GUFUSE").is_none()
            && exec.has_f8_gemm_mma_ks()
            && exec.has_quantize_e4m3_glu2(super::glu_act_of(arch));
        // OPT-IN only (a deliberate ruling: "I do not want to serve
        // fp4 weights on an fp8 model") - the served class must match the
        // model's own quantization by default. The fp4 ladder stays as an
        // explicit deployment mode, never the default and never the
        // benchmark basis vs same-class rivals.
        let fp4 = paddock_models::dev_var_os!("PADDOCK_G4_FP4").is_some()
            && !file_kq
            && n_expert == 0
            && exec.has_fp4_ladder()
            && gufuse
            && qkvfuse;
        if fp4 {
            // FP4_TMA routes the pack's bs launcher onto the fp4-TMA kernel
            // (a C-getenv gate - set_env, not set_var, or Windows never
            // delivers it); ACTIVE is the call-site lane switch.
            if std::env::var_os("PADDOCK_FP4_TMA").is_none() {
                crate::envset::set_env("PADDOCK_FP4_TMA", "1");
            }
            crate::envset::set_env("PADDOCK_G4_FP4_ACTIVE", "1");
        }
        // A4B fp8 ladder: the shared-FFN repacks are 128-padded
        // (pad_ffn_out/in above), so the f8w/f8t builders run unchanged on
        // MoE files - the pf8 profile showed the dense-GEMM block (~18%)
        // sitting on the int8 pipe this die serves slowly. Experts stay on
        // the s8-mma sorted class until the tcgen05 grouped GEMM lands.
        let f8r = n_expert == 0
            && !file_kq
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8R").is_none()
            && (paddock_models::dev_var_os!("PADDOCK_G4_F8R").is_some()
                || (exec.has_f8_gemm_mma_ks() && exec.compute_capability().0 == 12));
        let f8_on = (f8r || paddock_models::dev_var_os!("PADDOCK_G4_F8").is_some())
            && exec.has_f8_gemm_w8()
            && (!f8r || exec.has_f8_gemv());
        // Rowwise-e4m3 PREFILL planes (see the LayerWeights field note):
        // r >= 65 rows ride the fold-free per-row e4m3 GEMM; every decode/
        // verify rung keeps the exact-class q8 originals. Default on exactly
        // where the int8 mma pipe is crippled (cc 10.x, B200);
        // PADDOCK_G4_F8ROW=1 forces elsewhere for A/B, PADDOCK_G4_NO_F8ROW
        // kills. e4m3-class change to prefill hidden states AND KV contents
        // - coherence + long-prompt gates arbitrate.
        // Per-32 (f8w) is the cc-10 prefill default since async-SF v2 made
        // the tcgen05 block-scale route beat rowwise at every shape while
        // being strictly finer-grained; PADDOCK_G4_F8ROW forces the rowwise
        // planes instead, PADDOCK_G4_NO_F8ROW kills both.
        let f8row = f8_pf_on
            && exec.has_f8row_gemm()
            && paddock_models::dev_var_os!("PADDOCK_G4_F8ROW").is_some();
        let f8w_pf = f8_pf_on && !f8row && exec.has_f8_gemm_w8();
        // Wide-batch spec is uncapped on cc 10, like every other die. A
        // 16-slot cap once looked right, but that predated the f8t verify
        // lanes, the wmma bmax8 election and the spec-pipeline work; under
        // the current stack uncapped wins wide-batch throughput and TTFT
        // outright at unchanged acceptance. The cap only ever binds above 16
        // live slots, so narrower cells are untouched by construction. The
        // env always wins; PADDOCK_G4_SPEC_LIVE_MAX=16 restores the old cap.
        // wmma batch gate 24 -> 8 on cc10: at the spec-verify widths
        // (r 9..24) the wmma route runs the gu GEMM at 49 us against a
        // ~29 us weight-stream floor, so tc5q is the better route at those
        // rows. r <= 8 (narrow decode, shallow verify) keeps wmma - that
        // election measured the other way. The env always wins.
        if exec.compute_capability().0 == 10 && std::env::var_os("PADDOCK_F8T_WMMA_BMAX").is_none()
        {
            crate::envset::set_env("PADDOCK_F8T_WMMA_BMAX", "8");
        }
        // muse-glimmer: default-on the LIN K-split. On B200 it cuts the
        // mid-M PREFILL - wide-batch TTFT and throughput both improve - with
        // zero regression at narrow widths, because the ktz dispatch gate
        // (nt <= 1.5*SMs && batch <= 1024)
        // only fires at the coalesced-prefill widths a c32 pass hits, not the
        // small-M decode ticks. Numerically 100%-exact-bf16 vs the single chain
        // (decode-ks class), so no PPL gate needed. PADDOCK_LIN_KTZ=0 reverts;
        // the env always wins. gemma4 keeps its opt-in default (unmeasured here).
        if exec.compute_capability().0 == 10
            && arch == Arch::MuseGlimmer
            && std::env::var_os("PADDOCK_LIN_KTZ").is_none()
        {
            crate::envset::set_env("PADDOCK_LIN_KTZ", "1");
        }
        // tc5r K-split budget: publish pf_skfix's capacity (12M f32, load.rs
        // alloc) so the pack can K-split the verify-width tc5r launches.
        // Without it the o/down planes run 42 CTAs on a 148-SM die at
        // 0.56 TB/s, where a cutlass-class tile on the same shapes and the
        // same die reaches ~1.5 TB/s. The pack refuses
        // nz > 1 whenever out_dim*batch*nz would exceed this budget, and
        // stays fully off when the env is absent - other models keep their
        // own (smaller) scratch contracts. The env always wins;
        // PADDOCK_TC5R_NZ=1 forces the split off for A/B.
        if exec.compute_capability().0 == 10 && std::env::var_os("PADDOCK_TC5R_NZ_BUDGET").is_none()
        {
            crate::envset::set_env("PADDOCK_TC5R_NZ_BUDGET", "12582912");
        }
        // f8t16: publish the tc5r O16 election - the wo plane
        // (out n_embd, in = sliding-layer n_head*hd, 40/48 layers) writes
        // bf16 at chunk widths and batch.rs's wo_o16 consumers read it via
        // the p16 twins, both sides keyed on these same envs. Probe gate:
        // PADDOCK_G4_F8T16 (default off).
        if exec.compute_capability().0 == 10
            && paddock_models::dev_var_os!("PADDOCK_G4_F8T16").is_some()
            && std::env::var_os("PADDOCK_TC5R_O16_DIM").is_none()
        {
            crate::envset::set_env("PADDOCK_TC5R_O16_DIM", &n_embd.to_string());
            crate::envset::set_env("PADDOCK_TC5R_O16_IN", &(n_head * hd_swa).to_string());
        }
        // tc5r narrow-N arm: at batch <= 128 the N=256 tc5r mma spends half
        // its tensor-pipe cycles on padding cols - the verify band profiles
        // pipe-bound (per-active-SM throughput ~62%), not issue-bound, and
        // deepening the ring (S=7) does not help. N=128 halves the pipe work
        // and wins the wide spec verify outright with acceptance intact;
        // depth stays at 6. Other models keep N=256 until they have their own
        // batteries. The env always wins; PADDOCK_TC5R_N128=0 forces the
        // classic arm for A/B.
        if exec.compute_capability().0 == 10 && std::env::var_os("PADDOCK_TC5R_N128").is_none() {
            crate::envset::set_env("PADDOCK_TC5R_N128", "1");
        }
        // Plane unification: the f8t tile planes serve every band
        // through one launcher (tc5p <=64 / tc5r 65+), so the per-32 f8w
        // duplicates (-26 GiB) are built only on explicit request
        // (PADDOCK_G4_KEEP_F8W=1 restores the v61 world: f8w prefill class +
        // f8w r==1 gemvs) or when the f8t set is unavailable.
        let f8t_will_cover = f8_pf_on
            && exec.has_f8t_gemm()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8DEC").is_none()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8ATT").is_none();
        let keep_f8w = paddock_models::dev_var_os!("PADDOCK_G4_KEEP_F8W").is_some();
        if f8w_pf && (keep_f8w || !f8t_will_cover) {
            tracing::info!(
                "gemma4: building per-32 f8w prefill planes (attn + FFN duplicates, {n_layer} layers)"
            );
            for lw in layers.iter_mut() {
                lw.f8w_wq = Some(exec.q8_0_to_f8w(q8_of(&lw.wq)?)?);
                // k/v only when they are Q8 - every fp8 converter here reads a
                // repacked Q8 plane. A bf16 k/v keeps its own class and its
                // consumers fall through to Plane::gemv/gemm; the direct
                // bf16 -> e4m3 converters that would put it back on this ladder
                // (bf16_to_f8w already exists for the per-32 class) are the
                // follow-up rung, and they reach e4m3 in one step where this
                // path pays a Q8_0 hop first.
                lw.f8w_wk = match lw.wk.q8() {
                    Some(wk) => Some(exec.q8_0_to_f8w(wk)?),
                    None => None,
                };
                lw.f8w_wv = match lw.wv.as_ref().and_then(|v| v.q8()) {
                    Some(wv) => Some(exec.q8_0_to_f8w(wv)?),
                    None => None,
                };
                lw.f8w_wo = Some(exec.q8_0_to_f8w(q8_of(&lw.wo)?)?);
                lw.f8_gate = Some(exec.q8_0_to_f8w(q8_of(&lw.ffn_gate)?)?);
                lw.f8_up = Some(exec.q8_0_to_f8w(q8_of(&lw.ffn_up)?)?);
                lw.f8_down = Some(exec.q8_0_to_f8w(q8_of(&lw.ffn_down)?)?);
            }
            vram_mark("f8w prefill planes", &mut vram_prev);
        }
        // v4 decode planes: the r<=64 FFN band leaves the L2-capped q8 ring
        // for the tile-image rowwise class. Rowwise numerics on
        // DECODE tokens - coherence + long-prompt + acceptance gates
        // arbitrate; PADDOCK_G4_NO_F8DEC kills.
        let f8t_dec = f8_pf_on
            && exec.has_f8t_gemm()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8DEC").is_none();
        if f8t_dec {
            tracing::info!(
                "gemma4: building v4 tile-image decode planes (FFN trio, {n_layer} layers)"
            );
            let gu_fuse = exec.has_quantize_e4m3_glu2_row(super::glu_act_of(arch));
            for lw in layers.iter_mut() {
                let (gi, go) = (lw.ffn_gate.dims()[0], lw.ffn_gate.dims()[1]);
                if gu_fuse {
                    // fused gate|up plane: the two tile streams concatenate
                    // exactly (tile index = (row/128)*nkt + kt is relative to
                    // each plane, and up's stream lands at gate's byte size).
                    // Built instead of the split planes - VRAM-flat.
                    let g = exec.q8_0_to_f8row(q8_of(&lw.ffn_gate)?)?;
                    let g = exec.f8_repack_tiles(g, gi, go)?;
                    let u = exec.q8_0_to_f8row(q8_of(&lw.ffn_up)?)?;
                    let u = exec.f8_repack_tiles(u, gi, go)?;
                    let te = |e: cudarc::driver::DriverError| {
                        LoadError::Tensor("f8t_gu".into(), e.to_string())
                    };
                    let mut tiles = exec
                        .alloc_u8(g.tiles.len() + u.tiles.len())
                        .map_err(|e| LoadError::Tensor("f8t_gu".into(), e.to_string()))?;
                    let mut scale: cudarc::driver::CudaSlice<f32> =
                        exec.stream.alloc_zeros(go * 2).map_err(te)?;
                    let n = g.tiles.len();
                    let mut v = tiles.try_slice_mut(0..n).expect("f8t_gu lo");
                    exec.stream.memcpy_dtod(&g.tiles, &mut v).map_err(te)?;
                    let mut v = tiles
                        .try_slice_mut(n..n + u.tiles.len())
                        .expect("f8t_gu hi");
                    exec.stream.memcpy_dtod(&u.tiles, &mut v).map_err(te)?;
                    let mut v = scale.try_slice_mut(0..go).expect("f8t_gu s lo");
                    exec.stream.memcpy_dtod(&g.scale, &mut v).map_err(te)?;
                    let mut v = scale.try_slice_mut(go..2 * go).expect("f8t_gu s hi");
                    exec.stream.memcpy_dtod(&u.scale, &mut v).map_err(te)?;
                    // DEFAULT-ON at cc-10 (batch-gated at m>=16 in
                    // f8t_gemm_off) - it wins the wide nospec tick; low rungs
                    // stay tc5 under the gate. Kill: PADDOCK_NO_F8CUT.
                    let _f8cut = exec.compute_capability().0 == 10
                        && paddock_models::dev_var_os!("PADDOCK_NO_F8CUT").is_none();
                    // The GLOBAL f8cut floor is 8, lowered for qwen3.8's
                    // thin-m tile ladder. gemma4/muse were measured at 16 and
                    // are not re-measured here, so this plane pins its own
                    // floor and stays where it was.
                    let pgu = crate::gpu::F8TilePlane {
                        tiles,
                        scale,
                        flat: None,
                        flat_minb: 16,
                        flat_gui: false,
                        scale_il: None,
                    };
                    lw.f8t_gu = Some(pgu);
                } else {
                    let g = exec.q8_0_to_f8row(q8_of(&lw.ffn_gate)?)?;
                    lw.f8t_gate = Some(exec.f8_repack_tiles(g, gi, go)?);
                    let u = exec.q8_0_to_f8row(q8_of(&lw.ffn_up)?)?;
                    lw.f8t_up = Some(exec.f8_repack_tiles(u, gi, go)?);
                }
                let (di, dn) = (lw.ffn_down.dims()[0], lw.ffn_down.dims()[1]);
                let d = exec.q8_0_to_f8row(q8_of(&lw.ffn_down)?)?;
                let pdown = exec.f8_repack_tiles(d, di, dn)?;
                lw.f8t_down = Some(pdown);
            }
            // attn twin: qkv + wo on the same decode class (rowwise e4m3
            // attn weights - the riskier quality move, so its own kill:
            // PADDOCK_G4_NO_F8ATT)
            if paddock_models::dev_var_os!("PADDOCK_G4_NO_F8ATT").is_none() {
                tracing::info!(
                    "gemma4: building v4 tile-image decode planes (attn qkv+wo, {n_layer} layers)"
                );
                // qkv-concat on the tile route (cc-10 twin of f8a_wqkv): the
                // tile streams concatenate byte-exactly (same argument as the
                // gu fusion) and the strided nra2s epilogue reads the fused
                // GEMM's concat rows in place. nra2s has no fp8-cache arm, so
                // KV8 keeps the split planes. VRAM-flat: split planes are not
                // built when the fused plane is live.
                // KV8 no longer excludes the fusion: nra2s grew its
                // fp8-cache arm
                let qkvfuse = paddock_models::dev_var_os!("PADDOCK_G4_NO_QKVFUSE").is_none()
                    && exec.has_gemma_qkv_nra2s();
                for lw in layers.iter_mut() {
                    // PADDOCK_F8CUT builds flat k-major twins on every
                    // f8t plane; fp8.rs routes them through the vendored
                    // cutlass GEMM. Probe gate, default off.
                    // DEFAULT-ON at cc-10 (batch-gated at m>=16 in
                    // f8t_gemm_off) - it wins the wide nospec tick; low rungs
                    // stay tc5 under the gate. Kill: PADDOCK_NO_F8CUT.
                    let _f8cut = exec.compute_capability().0 == 10
                        && paddock_models::dev_var_os!("PADDOCK_NO_F8CUT").is_none();
                    let mk = |w: &crate::gpu::RepackedQ8| -> Result<_, LoadError> {
                        let (i, o) = (w.dims[0], w.dims[1]);
                        let r = exec.q8_0_to_f8row(w)?;
                        let p = exec.f8_repack_tiles(r, i, o)?;
                        Ok(p)
                    };
                    // the concat plane needs one operand class across all three
                    // segments, so a bf16 k/v takes the split branch (and only
                    // its Q8 members get tile planes at all)
                    if qkvfuse && lw.kv_q8() {
                        let te = |e: cudarc::driver::DriverError| {
                            LoadError::Tensor("f8t_qkv".into(), e.to_string())
                        };
                        let q = mk(q8_of(&lw.wq)?)?;
                        let k = mk(lw.wk.q8().expect("kv_q8"))?;
                        let v = match lw.wv.as_ref().and_then(|v| v.q8()) {
                            Some(wv) => Some(mk(wv)?),
                            None => None,
                        };
                        let vd = v.as_ref().map_or(0, |v| v.tiles.len());
                        let vs = v.as_ref().map_or(0, |v| v.scale.len());
                        let mut tiles = exec
                            .alloc_u8(q.tiles.len() + k.tiles.len() + vd)
                            .map_err(|e| LoadError::Tensor("f8t_qkv".into(), e.to_string()))?;
                        let mut scale: cudarc::driver::CudaSlice<f32> = exec
                            .stream
                            .alloc_zeros(q.scale.len() + k.scale.len() + vs)
                            .map_err(te)?;
                        let mut dpos = 0usize;
                        let mut spos = 0usize;
                        for pl in [Some(&q), Some(&k), v.as_ref()].into_iter().flatten() {
                            let mut w = tiles
                                .try_slice_mut(dpos..dpos + pl.tiles.len())
                                .expect("f8t_qkv tiles");
                            exec.stream.memcpy_dtod(&pl.tiles, &mut w).map_err(te)?;
                            dpos += pl.tiles.len();
                            let mut w = scale
                                .try_slice_mut(spos..spos + pl.scale.len())
                                .expect("f8t_qkv scale");
                            exec.stream.memcpy_dtod(&pl.scale, &mut w).map_err(te)?;
                            spos += pl.scale.len();
                        }
                        // The old wv.is_some() discriminator gate is gone:
                        // the corruption was f8t_gemm_off's intercept
                        // ignoring row_tile_off (fixed), not the
                        // V-less alias class. All layers flatten now; the
                        // per-plane floor (PADDOCK_F8CUT_QKV_MINB) confines
                        // routing to the wide prefill band where the
                        // 128x128 cutlass arm wins (decode/chunk stay tc5).
                        let pqkv = crate::gpu::F8TilePlane {
                            tiles,
                            scale,
                            flat: None,
                            flat_minb: 0,
                            flat_gui: false,
                            scale_il: None,
                        };
                        lw.f8t_qkv = Some(pqkv);
                    } else {
                        lw.f8t_wq = Some(mk(q8_of(&lw.wq)?)?);
                        lw.f8t_wk = match lw.wk.q8() {
                            Some(wk) => Some(mk(wk)?),
                            None => None,
                        };
                        lw.f8t_wv = match lw.wv.as_ref().and_then(|v| v.q8()) {
                            Some(wv) => Some(mk(wv)?),
                            None => None,
                        };
                    }
                    lw.f8t_wo = Some(mk(q8_of(&lw.wo)?)?);
                    // muse-glimmer o-gate e4m3 tile plane (opt-).
                    // The Q8 attn_gate is kept (prefill rides it; the reclaim
                    // never stubs it), so this is pure add - VRAM +~1.4 GiB on
                    // muse-30B. Route: forward::attn_gate_apply r==1 arm.
                    if crate::envset::env_on("PADDOCK_MUSE_OGATE_F8T")
                        && let Some(ag) = &lw.attn_gate
                    {
                        lw.f8t_attn_gate = Some(mk(q8_of(ag)?)?);
                    }
                }
            }
            vram_mark("f8t decode planes", &mut vram_prev);
        }
        // f8t TILE plane for the tied head (qwen35's out_f8t, ported): the
        // r==1 head was the last projection on the Q8 dp4a path - 1.5 GB
        // read/token, grid 262144 = one CTA per vocab row - while every
        // neighbour runs f8t. vocab/128 = 2048 row-tiles sits far above the
        // wmma 256-tile gate, so the head lands on the same warp-level
        // mma.sync route that carries gate|up. e4m3-class change on LOGITS
        // only (qwen's PPL gate measured the class at +0.22%); coherence +
        // acceptance gates arbitrate here. ~1.4 GB duplicate plane, only
        // where the f8t lane runs. Kill: PADDOCK_NO_F8T_LMHEAD (qwen's
        // switch - one env kills the class engine-wide).
        // Built from whatever plane `head` actually is - tied or untied. It
        // used to read token_embd_rep unconditionally, which made an untied
        // file decode against the EMBEDDING silently (every head call site
        // prefers head_f8t when it exists); sourcing it from `head` is what
        // makes the tie question stop mattering here.
        //
        // A BF16 head reaches this route too. It could not before -
        // `q8_0_to_f8row` is the only other producer of an F8RowPlane and it
        // wants a Q8 source - so muse-glimmer's bf16 head fell to the plain
        // bf16 kernel at every call site: dense decode, prefill finishers,
        // batched verify AND the DFlash draft round. Measured there:
        // 2.690 GB read in 6.46 ms = 416 GB/s, about a quarter of a
        // ~1531 GB/s DRAM roof, and 71% of a whole draft round.
        // `bf16_to_f8row` is the missing edge - straight from the native
        // weights, no Q8 round trip (which would double-quantize).
        let head_f8t = if f8t_dec && paddock_models::dev_var_os!("PADDOCK_NO_F8T_LMHEAD").is_none()
        {
            let (i, o) = (head.dims()[0], head.dims()[1]);
            let row = if i % 128 != 0 || o % 128 != 0 {
                None
            } else if let Some(hq) = head.q8() {
                Some(exec.q8_0_to_f8row(hq)?)
            } else if exec.has_bf16_to_f8row() {
                // the bf16 source bytes come straight from the mapped file -
                // `head` holds the device copy, and re-reading the mapping
                // keeps a second 2.7 GB host staging out of the picture
                let name = match map.tensor_bytes("output.weight") {
                    Ok(_) => "output.weight",
                    Err(_) => "token_embd.weight",
                };
                let (_, bytes) = map.tensor_bytes(name).map_err(GpuError::from)?;
                Some(exec.bf16_to_f8row(bytes, i, o)?)
            } else {
                None
            };
            match row {
                Some(r) => {
                    let p = exec
                        .f8_repack_tiles(r, i, o)
                        .map_err(|e| LoadError::Tensor("head_f8t".into(), e.to_string()))?;
                    tracing::info!(
                        "gemma4: f8t lm_head tile plane built ({i}x{o}, {} source)",
                        if head.q8().is_some() { "q8_0" } else { "bf16" }
                    );
                    vram_mark("f8t lm_head plane", &mut vram_prev);
                    Some(p)
                }
                None => None,
            }
        } else {
            None
        };
        // Untiled per-row twin, for the dies the tile route cannot reach:
        // `f8t_gemm`/`f8_repack_tiles` are NULLed unless cc == 10.0 exactly
        // (tc5p SASS is sm_100a-only), while `f8row_gemm` runs from sm_89. A
        // BF16 head is the one case with no fallback worth having - the whole
        // int8 ladder needs a repacked Q8 source - so on sm_120 muse-glimmer's
        // head ran the plain bf16 kernel at 416 GB/s (about a quarter of the
        // DRAM roof, and 71% of a DFlash draft round). Built only when the
        // tile plane is absent and
        // only for bf16; the Q8 heads already have their ladder.
        let head_f8row = if head_f8t.is_none()
            && exec.has_f8row_gemm()
            && paddock_models::dev_var_os!("PADDOCK_NO_F8T_LMHEAD").is_none()
        {
            let (i, o) = (head.dims()[0], head.dims()[1]);
            // The per-ROW e4m3 head has no o%128 tile constraint (i%32 suffices),
            // so it reaches muse-glimmer's 202048 vocab where head_f8t cannot
            // A Q8 head repacks straight through q8_0_to_f8row (the
            // f8r-plane producer) - opt-in PADDOCK_MUSE_HEAD_F8ROW, since e4m3
            // on logits is a small class change (qwen PPL +0.22%), quality-gated.
            // A bf16 head keeps its default-on route.
            let plane = if i % 32 != 0 {
                None
            } else if let Some(hq) = head
                .q8()
                .filter(|_| crate::envset::env_on("PADDOCK_MUSE_HEAD_F8ROW"))
            {
                Some(exec.q8_0_to_f8row(hq)?)
            } else if head.q8().is_none() && exec.has_bf16_to_f8row() {
                let name = match map.tensor_bytes("output.weight") {
                    Ok(_) => "output.weight",
                    Err(_) => "token_embd.weight",
                };
                let (_, bytes) = map.tensor_bytes(name).map_err(GpuError::from)?;
                Some(exec.bf16_to_f8row(bytes, i, o)?)
            } else {
                None
            };
            if plane.is_some() {
                tracing::info!(
                    "gemma4: f8row lm_head plane built ({i}x{o}, {} source)",
                    if head.q8().is_some() { "q8_0" } else { "bf16" }
                );
                vram_mark("f8row lm_head plane", &mut vram_prev);
            }
            plane
        } else {
            None
        };
        if f8row {
            tracing::info!(
                "gemma4: building rowwise-e4m3 prefill planes (attn + FFN duplicates, {n_layer} layers)"
            );
            for lw in layers.iter_mut() {
                lw.f8_wq = Some(exec.q8_0_to_f8row(q8_of(&lw.wq)?)?);
                lw.f8_wk = match lw.wk.q8() {
                    Some(wk) => Some(exec.q8_0_to_f8row(wk)?),
                    None => None,
                };
                lw.f8_wv = match lw.wv.as_ref().and_then(|v| v.q8()) {
                    Some(wv) => Some(exec.q8_0_to_f8row(wv)?),
                    None => None,
                };
                lw.f8_wo = Some(exec.q8_0_to_f8row(q8_of(&lw.wo)?)?);
                lw.f8r_gate = Some(exec.q8_0_to_f8row(q8_of(&lw.ffn_gate)?)?);
                lw.f8r_up = Some(exec.q8_0_to_f8row(q8_of(&lw.ffn_up)?)?);
                lw.f8r_down = Some(exec.q8_0_to_f8row(q8_of(&lw.ffn_down)?)?);
            }
        }
        // Tile-linear conversion of the F8R/F8A replace planes (qwen35's
        // f8_lin lane, gemm/f8_lin.cuh): per-CTA contiguous weight streams -
        // qwen measured the decode GEMM at its access-pattern roof and the
        // kt prefill twin at ~83% of the die's e4m3 ceiling vs the older
        // w8-TMA route gemma prefill sits on (68% of the c32-prefill
        // window). Converted planes carry the 4-byte marker scale; the exec
        // wrappers dispatch on it, and the r==1 gemv arms (f32 activations,
        // no lin twin) branch to the quantize+lin band explicitly.
        // OPT-IN while gating: PADDOCK_G4_F8LIN=1; shares qwen's kill
        // switches. e2m1 (fp4) planes keep their own layout. VRAM-neutral.
        // Tile-linear lane election. On a pre-ktz stack only the FFN lanes
        // were worth converting: attn lin cost narrow-batch decode more than
        // it bought at width. Once the ktz K-split went default (and its
        // graph-unsafe scratch was fixed) attn lin turned positive at every
        // width it used to lose, leaving only a small single-stream
        // regression, so the DEFAULT is all four lanes. PADDOCK_G4_F8LIN:
        // "ffn" ffn-only (the pre-ktz default, revert lever), "attn"
        // attn-only (A/B). Kill: PADDOCK_NO_F8LIN (or the TMA kill - lin
        // rides the same route). Remaining door: retune pd_f8_gemm_lin at
        // the r<=9 qkv/wo shapes to erase that last single-stream cost.
        let f8lin_mode = paddock_models::dev_var!("PADDOCK_G4_F8LIN").unwrap_or_default();
        let f8lin_base = exec.has_f8_lin()
            && exec.has_f8_gemm_mma_ks() // the r==1 lin routing rides the twin band
            && !fp4
            && paddock_models::dev_var_os!("PADDOCK_NO_F8LIN").is_none()
            && std::env::var_os("PADDOCK_NO_F8W8_TMA").is_none();
        let f8lin_attn = f8lin_base && f8lin_mode != "ffn";
        let f8lin_ffn = f8lin_base && f8lin_mode != "attn";
        let lin = |on: bool,
                   w: crate::gpu::RepackedMxfp4,
                   in_dim: usize,
                   out_dim: usize|
         -> Result<crate::gpu::RepackedMxfp4, LoadError> {
            if on && in_dim.is_multiple_of(128) && out_dim.is_multiple_of(16) {
                Ok(exec.f8w_repack_lin(w, in_dim, out_dim)?)
            } else {
                Ok(w)
            }
        };
        if f8lin_base {
            tracing::info!(
                attn = f8lin_attn,
                ffn = f8lin_ffn,
                "gemma4: converting f8 replace planes to the tile-linear layout"
            );
        }
        // fp8-native ingestion (the qwen35 lane pattern): an HF
        // snapshot dir sources the e4m3 serving planes from the bf16
        // safetensors checkpoint - One quantization (bf16 -> per-32 e4m3)
        // instead of two (bf16 -> Q8_0 -> e4m3). Same f8w format, so the
        // whole landed f8 stack (lin repacks, gu fuse/interleave, TMA images)
        // serves it unchanged. Planes only: the GGUF stays the artifact for
        // tokenizer/hparams/norms/embeddings and the Q8 lane is untouched
        // when no dir is given. The dir arrives as an explicit load option
        // (config field `fp8_native`), never via env. Shape-guarded per
        // tensor - any mismatch (padded MoE dims, missing name, non-bf16
        // shard) falls back to the Q8-derived plane with a warning.
        let fp8_native = fp8_native_dir.and_then(|d| {
            match paddock_models::safetensors::ShardedSafetensors::open_dir(d) {
                Ok(st) => {
                    tracing::info!(
                        "gemma4 fp8-native ingestion: {} tensors from {}",
                        st.names().count(),
                        d.display()
                    );
                    Some(st)
                }
                Err(e) => {
                    tracing::warn!(
                        "gemma4 fp8-native ingestion unavailable ({e}) - Q8-derived planes"
                    );
                    None
                }
            }
        });
        // reused staging cells for the ingestion lanes (declared before the
        // closures that capture them): per-tensor transients through the
        // mempool leave un-trimmable holes that shrink the serving pool
        let pc_stage: std::cell::RefCell<Option<CudaSlice<f32>>> = std::cell::RefCell::new(None);
        let nat_stage: std::cell::RefCell<Option<CudaSlice<u8>>> = std::cell::RefCell::new(None);
        let f8w_native = |gguf: &str,
                          q8: &crate::gpu::RepackedQ8|
         -> Result<crate::gpu::RepackedMxfp4, crate::gpu::GpuError> {
            if let Some(st) = fp8_native.as_ref()
                && let Some(hf) = paddock_models::safetensors::gemma4_hf_name(gguf)
                && let Some((t, b)) = st.bytes(&hf)
            {
                if t.dtype == paddock_models::safetensors::StDtype::Bf16
                    && b.len() / 2 == q8.dims[0] * q8.dims[1]
                {
                    let mut stage = nat_stage.borrow_mut();
                    if stage.as_ref().is_none_or(|st| st.len() < b.len()) {
                        *stage = None;
                        *stage = Some(exec.alloc_u8(b.len())?);
                    }
                    let st = stage.as_mut().expect("allocated above");
                    exec.stream
                        .memcpy_htod(b, st)
                        .map_err(|e| GpuError::Driver(e.to_string()))?;
                    return exec.bf16_to_f8w_dev(st, b.len());
                }
                tracing::warn!("fp8-native: {hf} dtype/shape mismatch - Q8-derived plane");
            }
            exec.q8_0_to_f8w(q8)
        };
        // PC lane (opt-in PADDOCK_G4_PC, requires the fp8-native
        // source): gu plane bytes quantized on a per-ROW pow2 grid via the
        // pack's row quantizer (house e-pick -> exactly-representable ue8m0
        // exponents), the per-32 strip filled with each row's exponent so
        // every existing consumer (kt3g fused, mma_ks tiny-r) dequantizes
        // identically, and the f32 row scales kept for the scale-free kt4a
        // chunk route. One plane, VRAM-flat; numerics move from per-32 to
        // per-row pow2 granularity on gu weights only - parity + serve
        // gates arbitrate the class.
        let g4_pc = paddock_models::dev_var_os!("PADDOCK_G4_PC").is_some() && fp8_native.is_some();
        // One reused f32 staging buffer for the pc quantizer (all 120 gu
        // tensors share [n_ff x n_embd]): per-tensor alloc/drop cycles ~55GB
        // of 462MB transients through the mempool and breaks the POOLED KV
        // stack's allocation, silently dropping the serve to f16 non-pooled
        // KV. Freed explicitly after the plane loop, before the KV stack
        // initializes.

        let f8w_pc = |gguf: &str,
                      q8: &crate::gpu::RepackedQ8|
         -> Result<Option<(crate::gpu::RepackedMxfp4, Vec<f32>)>, GpuError> {
            if !g4_pc {
                return Ok(None);
            }
            let Some(st) = fp8_native.as_ref() else {
                return Ok(None);
            };
            let Some(hf) = paddock_models::safetensors::gemma4_hf_name(gguf) else {
                return Ok(None);
            };
            let Some((t, b)) = st.bytes(&hf) else {
                return Ok(None);
            };
            let (rows, cols) = (q8.dims[1], q8.dims[0]);
            if t.dtype != paddock_models::safetensors::StDtype::Bf16
                || b.len() / 2 != rows * cols
                || cols % 32 != 0
            {
                tracing::warn!("pc lane: {hf} dtype/shape mismatch - per-32 plane");
                return Ok(None);
            }
            // bf16 -> f32 host-side (parallel), quantize on device
            let mut xf = vec![0f32; rows * cols];
            let nthreads = std::thread::available_parallelism()
                .map(|n| n.get().min(16))
                .unwrap_or(8);
            let band = rows.div_ceil(nthreads);
            std::thread::scope(|sc| {
                for (ti, chunk) in xf.chunks_mut(band * cols).enumerate() {
                    let src = &b[ti * band * cols * 2..];
                    sc.spawn(move || {
                        for (i, o) in chunk.iter_mut().enumerate() {
                            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                            *o = f32::from_bits((bits as u32) << 16);
                        }
                    });
                }
            });
            let mut stage = pc_stage.borrow_mut();
            // GROW-ONLY. This used to realloc on any size MISMATCH (`!=`), so
            // a model whose tensor shapes alternate freed and reallocated on
            // nearly every tensor, each time leaving a differently-sized hole
            // between two live weight planes - and cuMemPoolTrimTo can only
            // return a block when nothing in it is live.
            //
            // Honest NOTE: this is the right shape on its own terms (it
            // cannot churn, and it matches the shared staging buffer in
            // gpu/upload.rs) but it is not what strands memory on
            // gemma-4-31B. Measured before and after: retained-not-live
            // 4.95 GB either way, ~15% of live, against 0.3-4.1% on the other
            // families. Whatever holds those 5 GB is a different phase of
            // this load and still needs a bisect - do not read this comment
            // as the fix for it.
            //
            // A bigger-than-needed buffer is safe for the consumer:
            // quantize_e4m3_row_u8 takes n_dim/batch explicitly and reads the
            // prefix, and the memcpy below fills exactly that prefix, so no
            // stale tail is ever read. Released at the end of load with the
            // other staging (`pc_stage.take()`).
            let need = rows * cols;
            if stage.as_ref().is_none_or(|st| st.len() < need) {
                *stage = None; // drop before realloc so the grow can reuse it
                *stage = Some(
                    exec.stream
                        .alloc_zeros::<f32>(need)
                        .map_err(|e| GpuError::Driver(e.to_string()))?,
                );
            }
            let xd = stage.as_mut().expect("allocated above");
            {
                let mut view = xd
                    .try_slice_mut(0..need)
                    .ok_or_else(|| GpuError::Unsupported("pc stage slice".into()))?;
                exec.stream
                    .memcpy_htod(&xf, &mut view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            let xd = stage.as_mut().expect("allocated above");
            drop(xf);
            let mut data = exec.alloc_u8(rows * cols)?;
            let mut rs = exec
                .stream
                .alloc_zeros::<f32>(rows)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.quantize_e4m3_row_u8(xd, &mut data, &mut rs, cols, rows)?;
            let mut ws = vec![0f32; rows];
            exec.stream
                .memcpy_dtoh(&rs, &mut ws)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            // per-32 strip = the row exponent repeated (pow2-exact by the
            // e-pick); dead rows (ws == 0) keep exponent 0 -> strip 127
            let n_kb = cols / 32;
            let mut strip = vec![0u8; rows * n_kb];
            for (o, srow) in strip.chunks_mut(n_kb).enumerate() {
                let e = if ws[o] > 0.0 {
                    (ws[o].to_bits() >> 23) as i32 - 127
                } else {
                    0
                };
                let sb = (e + 127).clamp(0, 255) as u8;
                srow.fill(sb);
            }
            let scale = exec
                .stream
                .clone_htod(&strip)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            Ok(Some((crate::gpu::RepackedMxfp4 { data, scale }, ws)))
        };
        // rowwise (strip-free) plane class: pc planes quantize
        // per-row pow2, so the per-32 strip in every lin box repeats the row
        // exponent - 3.03% dead weight bytes on the whole f8 GEMM band.
        // When every consumer has its rowwise twin (has_f8_rowvec), pc
        // planes convert to data-only boxes + a per-row exponent tail
        // instead; bit-exact vs the strip route.
        // Kill: PADDOCK_NO_F8_ROWVEC (same binary, load-time A/B).
        let rowvec = g4_pc
            && exec.has_f8_rowvec(super::glu_act_of(arch))
            && paddock_models::dev_var_os!("PADDOCK_NO_F8_ROWVEC").is_none();
        if rowvec {
            tracing::info!("gemma4: pc planes -> rowwise (strip-free) lin boxes");
        }
        // per-row ue8m0 bytes from the pc row scales - the exact strip-byte
        // construction (dead rows exponent 0 -> 127)
        let wse_of = |ws: &[f32]| -> Vec<u8> {
            ws.iter()
                .map(|&w| {
                    if w > 0.0 {
                        (w.to_bits() >> 23).min(255) as u8
                    } else {
                        127u8
                    }
                })
                .collect()
        };
        if f8_on {
            if std::env::var_os("PADDOCK_F8W8_TMA").is_none() {
                crate::envset::set_env("PADDOCK_F8W8_TMA", "1");
            }
            // gufuse hoisted above (shared with the fp4 decision)
            for (li, lw) in layers.iter_mut().enumerate() {
                // verify-GEMM dedup: with gufuse the separate gate/up planes
                // are never built - every f8r arm rides the fused plane, so
                // the dedup is VRAM-flat (v1 kept both = +13.2GB = c32 pool
                // starvation, 547->501)
                if !gufuse {
                    lw.f8_gate = Some(f8w_native(
                        &format!("blk.{li}.ffn_gate.weight"),
                        q8_of(&lw.ffn_gate)?,
                    )?);
                    lw.f8_up = Some(f8w_native(
                        &format!("blk.{li}.ffn_up.weight"),
                        q8_of(&lw.ffn_up)?,
                    )?);
                }
                // The FFN pair converts to lin ALL-OR-NOTHING: the r==1 arm
                // branches on gu's layout for both GEMMs, so a split verdict
                // (gu row-major + down lin, or vice versa) would read one
                // plane in the wrong layout. One predicate implies both
                // planes' dim guards: gu (in=n_embd%128, out=2n_ff, seam at
                // n_ff), down (in=n_ff%128, out=n_embd%16).
                let ffn_lin = f8lin_ffn
                    && gufuse
                    && lw.ffn_gate.dims()[0] % 128 == 0
                    && lw.ffn_gate.dims()[1] % 128 == 0
                    && lw.ffn_down.dims()[1] % 16 == 0;
                lw.f8_down = Some(match (fp4, ffn_lin) {
                    (true, _) => exec.q8_0_to_mxfp4(q8_of(&lw.ffn_down)?)?,
                    (false, true) => {
                        match f8w_pc(&format!("blk.{li}.ffn_down.weight"), q8_of(&lw.ffn_down)?)? {
                            Some((plane, ws)) => {
                                lw.down_ws = Some(exec.stream.clone_htod(&ws).map_err(|e| {
                                    LoadError::Tensor("down_ws".into(), e.to_string())
                                })?);
                                if rowvec {
                                    exec.f8w_build_lin_rw(
                                        plane.data,
                                        &wse_of(&ws),
                                        lw.ffn_down.dims()[0],
                                        lw.ffn_down.dims()[1],
                                        false,
                                    )?
                                } else {
                                    lin(true, plane, lw.ffn_down.dims()[0], lw.ffn_down.dims()[1])?
                                }
                            }
                            None => lin(
                                true,
                                f8w_native(
                                    &format!("blk.{li}.ffn_down.weight"),
                                    q8_of(&lw.ffn_down)?,
                                )?,
                                lw.ffn_down.dims()[0],
                                lw.ffn_down.dims()[1],
                            )?,
                        }
                    }
                    (false, false) => {
                        f8w_native(&format!("blk.{li}.ffn_down.weight"), q8_of(&lw.ffn_down)?)?
                    }
                });
                if gufuse {
                    // verify-GEMM dedup: fused gate|up plane (rows 0..n_ff =
                    // gate, n_ff.. = up; both planes are plain out-row-major
                    // so concat is two dtod copies). See LayerWeights::f8_gu.
                    let te = |e: cudarc::driver::DriverError| {
                        LoadError::Tensor("f8_gu".into(), e.to_string())
                    };
                    let mut pc_ws: Option<(Vec<f32>, Vec<f32>)> = None;
                    let g = if fp4 {
                        exec.q8_0_to_mxfp4(q8_of(&lw.ffn_gate)?)?
                    } else if let Some((plane, ws)) =
                        f8w_pc(&format!("blk.{li}.ffn_gate.weight"), q8_of(&lw.ffn_gate)?)?
                    {
                        pc_ws = Some((ws, Vec::new()));
                        plane
                    } else {
                        f8w_native(&format!("blk.{li}.ffn_gate.weight"), q8_of(&lw.ffn_gate)?)?
                    };
                    let u = if fp4 {
                        exec.q8_0_to_mxfp4(q8_of(&lw.ffn_up)?)?
                    } else if pc_ws.is_some() {
                        // gate went pc -> up must too (one plane, one class)
                        let (plane, ws) =
                            f8w_pc(&format!("blk.{li}.ffn_up.weight"), q8_of(&lw.ffn_up)?)?
                                .ok_or_else(|| {
                                    LoadError::Tensor(
                                        "gu pc".into(),
                                        "gate pc-quantized but up unavailable".into(),
                                    )
                                })?;
                        pc_ws.as_mut().expect("set above").1 = ws;
                        plane
                    } else {
                        f8w_native(&format!("blk.{li}.ffn_up.weight"), q8_of(&lw.ffn_up)?)?
                    };
                    let (g, u) = (&g, &u);
                    let mut data = exec
                        .alloc_u8(g.data.len() + u.data.len())
                        .map_err(|e| LoadError::Tensor("f8_gu".into(), e.to_string()))?;
                    let mut scale = exec
                        .alloc_u8(g.scale.len() + u.scale.len())
                        .map_err(|e| LoadError::Tensor("f8_gu".into(), e.to_string()))?;
                    let n = g.data.len();
                    let mut v = data.try_slice_mut(0..n).expect("f8_gu data lo");
                    exec.stream.memcpy_dtod(&g.data, &mut v).map_err(te)?;
                    let mut v = data
                        .try_slice_mut(n..n + u.data.len())
                        .expect("f8_gu data hi");
                    exec.stream.memcpy_dtod(&u.data, &mut v).map_err(te)?;
                    let n = g.scale.len();
                    let mut v = scale.try_slice_mut(0..n).expect("f8_gu scale lo");
                    exec.stream.memcpy_dtod(&g.scale, &mut v).map_err(te)?;
                    let mut v = scale
                        .try_slice_mut(n..n + u.scale.len())
                        .expect("f8_gu scale hi");
                    exec.stream.memcpy_dtod(&u.scale, &mut v).map_err(te)?;
                    // lin conversion rides the shared ffn_lin verdict (see
                    // the down plane above) so gu and down always agree
                    let (gin, gout) = (lw.ffn_gate.dims()[0], lw.ffn_gate.dims()[1]);
                    let gu = crate::gpu::RepackedMxfp4 { data, scale };
                    // gu-interleave election (the epilogue-fusion door):
                    // permuted rows are invisible to the lin
                    // GEMMs, geglu consumers switch to geglu2i, and the
                    // r>=32 band fuses geglu+quant into the GEMM. Needs the
                    // whole trio - a plane interleaved without geglu2i would
                    // scramble the FFN. Kill: PADDOCK_G4_NO_GUIL.
                    let gu_il = ffn_lin
                        && exec.has_f8w_repack_lin_gui()
                        && exec.has_quantize_e4m3_glu2i(super::glu_act_of(arch))
                        && exec.has_f8_gemm_lin_gu(super::glu_act_of(arch))
                        && paddock_models::dev_var_os!("PADDOCK_G4_NO_GUIL").is_none();
                    lw.gu_il = gu_il;
                    lw.f8_gu = Some(if gu_il {
                        // rowwise: pc-quantized gu only (native per-32 strips
                        // carry real information and keep the strip route)
                        if rowvec && let Some((wsg, wsu)) = &pc_ws {
                            let mut wsa = wsg.clone();
                            wsa.extend_from_slice(wsu);
                            exec.f8w_build_lin_rw(gu.data, &wse_of(&wsa), gin, 2 * gout, true)?
                        } else {
                            exec.f8w_repack_lin_gui(gu, gin, 2 * gout)?
                        }
                    } else if ffn_lin {
                        lin(true, gu, gin, 2 * gout)?
                    } else {
                        gu
                    });
                    if let Some((mut wsg, wsu)) = pc_ws.take() {
                        wsg.extend_from_slice(&wsu);
                        lw.gu_ws = Some(
                            exec.stream
                                .clone_htod(&wsg)
                                .map_err(|e| LoadError::Tensor("gu_ws".into(), e.to_string()))?,
                        );
                    }
                    if paddock_models::dev_var_os!("PADDOCK_G4_FP4_PROBE").is_some() {
                        let g4 = exec.q8_0_to_mxfp4(q8_of(&lw.ffn_gate)?)?;
                        let u4 = exec.q8_0_to_mxfp4(q8_of(&lw.ffn_up)?)?;
                        let mut d4 = exec
                            .alloc_u8(g4.data.len() + u4.data.len())
                            .map_err(|e| LoadError::Tensor("fp4_gu".into(), e.to_string()))?;
                        let mut s4 = exec
                            .alloc_u8(g4.scale.len() + u4.scale.len())
                            .map_err(|e| LoadError::Tensor("fp4_gu".into(), e.to_string()))?;
                        let n = g4.data.len();
                        let mut v = d4.try_slice_mut(0..n).expect("fp4 d lo");
                        exec.stream.memcpy_dtod(&g4.data, &mut v).map_err(te)?;
                        let mut v = d4.try_slice_mut(n..n + u4.data.len()).expect("fp4 d hi");
                        exec.stream.memcpy_dtod(&u4.data, &mut v).map_err(te)?;
                        let n = g4.scale.len();
                        let mut v = s4.try_slice_mut(0..n).expect("fp4 s lo");
                        exec.stream.memcpy_dtod(&g4.scale, &mut v).map_err(te)?;
                        let mut v = s4.try_slice_mut(n..n + u4.scale.len()).expect("fp4 s hi");
                        exec.stream.memcpy_dtod(&u4.scale, &mut v).map_err(te)?;
                        lw.fp4_gu = Some(crate::gpu::RepackedMxfp4 {
                            data: d4,
                            scale: s4,
                        });
                    }
                }
                if f8r {
                    // drop the q8 planes (stubs keep dims for n_ff lookups;
                    // no consumer touches their data in F8R mode)
                    for w in [&mut lw.ffn_gate, &mut lw.ffn_up, &mut lw.ffn_down] {
                        let dims = w.dims().to_vec();
                        *w = Plane::Q8(crate::gpu::RepackedQ8 {
                            data: exec
                                .alloc_u8(32)
                                .map_err(|e| LoadError::Tensor("f8r stub".into(), e.to_string()))?,
                            scale: exec
                                .alloc_u8(32)
                                .map_err(|e| LoadError::Tensor("f8r stub".into(), e.to_string()))?,
                            dims,
                        });
                    }
                }
            }
        }
        // F8A - the attn-side twin of F8R: attn projections
        // (wq/wk/wv/wo) on the per-32 block-scale e4m3 class, REPLACE
        // design like the FFN trio (q8 -> 32-byte stubs, dims kept;
        // VRAM-flat: e4m3 1.031 vs q8 1.0625 B/w). Ladder: fused _at gemv
        // r==1 / mma_ks twin 2..=31 (and 32..64 on underfilled outs) /
        // TMA GEMM above. Lossy e4m3 on Q/K/V/O weights - K/V CACHE
        // CONTENTS change class too; full gate suite before any default.
        // Opt-in: PADDOCK_G4_F8A=1. Excluded when the cc-10 rowwise planes
        // are active (their consumers read the q8 originals).
        // DEFAULT-ON: every width is flat-to-up, and the long-prompt temp-0,
        // long-gen and acceptance gates all pass. Kill: PADDOCK_G4_NO_F8A.
        // A4B (n_expert != 0) now INCLUDED: every attn projection
        // dim on the A4B is 128-aligned (2816/2048/4096/8192), so the F8A
        // F8R is a REPLACE lane (rowwise e4m3 in, Q8 layer planes stubbed), so
        // its net is what the ledger should show. This block and F8A below
        // used to be the only unmarked spans in the file, which made it
        // impossible to tell from a boot log whether the cc-12 lanes had
        // engaged at all.
        if f8r {
            vram_mark(
                "f8r REPLACE (rowwise e4m3 in, q8 layer planes out)",
                &mut vram_prev,
            );
        }

        // planes build unchanged - and the pf8 profile put the q8 attn GEMMs
        // inside the 18% dense-GEMM block on a die whose int8 pipe is the
        // slow path. The original bring-up NaN was the pf_e4q stub sizing
        // (fixed: the alloc predicate includes f8a), not F8A itself.
        let f8a = !file_kq
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8A").is_none()
            && exec.has_f8_gemm_w8()
            && exec.has_f8_gemv()
            && exec.has_f8_gemm_mma_ks()
            && !f8row
            && !f8w_pf;
        if f8a {
            if std::env::var_os("PADDOCK_F8W8_TMA").is_none() {
                crate::envset::set_env("PADDOCK_F8W8_TMA", "1");
            }
            // qkv-concat (VRAM-flat): fused plane replaces the separate
            // wq/wk/wv; f16 KV only. qkvfuse/fp4 hoisted above.
            let conv = |name: &str,
                        w: &crate::gpu::RepackedQ8|
             -> Result<crate::gpu::RepackedMxfp4, crate::gpu::GpuError> {
                if fp4 {
                    exec.q8_0_to_mxfp4(w)
                } else {
                    f8w_native(name, w)
                }
            };
            for (li, lw) in layers.iter_mut().enumerate() {
                // one plane means one operand class across q|k|v, so a layer
                // whose k/v ship bf16 takes the split branch below
                // Q8-derived fused qkv, in one call. The general path below
                // builds three separate e4m3 planes, allocates a fourth to
                // concat them into, repacks that into the linear layout and
                // drops all four - five allocations and four frees per layer,
                // each hole landing under the linear plane allocated above it,
                // which the pool can never return. Measured across the phase:
                // 4.06 GB stranded, ~1.2 of it here. This arm allocates the
                // returned plane and nothing else.
                //
                // Restricted to the route where `conv` would have gone through
                // q8_0_to_f8w anyway: fp4 sources its bytes from mxfp4 and an
                // fp8-native snapshot from bf16, and `g4_pc` requires a
                // snapshot, so this condition also rules the pc arms out.
                if qkvfuse && lw.kv_q8() && !fp4 && fp8_native.is_none() {
                    let (qin, qd) = (lw.wq.dims()[0], lw.wq.dims()[1]);
                    let kvd = lw.wk.dims()[1];
                    let qkv_out = qd + kvd * if lw.wv.is_some() { 2 } else { 1 };
                    // both segment boundaries must be box-aligned or the
                    // layout stays row-major - the serial lane row-slices this
                    // plane at q_dim and q_dim+kv_dim
                    let use_lin = f8lin_attn && qd % 128 == 0 && kvd % 128 == 0;
                    // Where the seats were skipped at construction there is no
                    // resident source to convert from - read the Q8 blocks
                    // straight out of the mapped file instead. One tensor's
                    // transient at a time, dropped before the next, so the
                    // pool never holds a second copy of anything.
                    let staged: Vec<crate::gpu::RepackedQ8> = if attn_skipped[li] {
                        let mut v = vec![
                            exec.repack_q8(map, &format!("blk.{li}.attn_q.weight"))?,
                            exec.repack_q8(map, &format!("blk.{li}.attn_k.weight"))?,
                        ];
                        if lw.wv.is_some() {
                            v.push(exec.repack_q8(map, &format!("blk.{li}.attn_v.weight"))?);
                        }
                        v
                    } else {
                        Vec::new()
                    };
                    let plane = if attn_skipped[li] {
                        let srcs: Vec<&crate::gpu::RepackedQ8> = staged.iter().collect();
                        exec.q8_0_to_f8w_lin(&srcs, qin, qkv_out, use_lin)?
                    } else {
                        let mut srcs: Vec<&crate::gpu::RepackedQ8> =
                            vec![q8_of(&lw.wq)?, lw.wk.q8().expect("kv_q8")];
                        if let Some(wv) = lw.wv.as_ref().and_then(|v| v.q8()) {
                            srcs.push(wv);
                        }
                        exec.q8_0_to_f8w_lin(&srcs, qin, qkv_out, use_lin)?
                    };
                    drop(staged);
                    lw.f8a_wqkv = Some(plane);
                } else if qkvfuse && lw.kv_q8() {
                    let wk_q8 = lw.wk.q8().expect("kv_q8");
                    let te = |e: cudarc::driver::DriverError| {
                        LoadError::Tensor("f8a_wqkv".into(), e.to_string())
                    };
                    // pc arms: all present segments must go pc
                    // together (one plane, one class); any miss -> all native
                    let mut qkv_pc: Option<Vec<f32>> = None;
                    let (q, k, v);
                    if !fp4 {
                        let qp = f8w_pc(&format!("blk.{li}.attn_q.weight"), q8_of(&lw.wq)?)?;
                        let kp = f8w_pc(&format!("blk.{li}.attn_k.weight"), wk_q8)?;
                        let vp = match lw.wv.as_ref().and_then(|v| v.q8()) {
                            Some(wv) => f8w_pc(&format!("blk.{li}.attn_v.weight"), wv)?.map(Some),
                            None => Some(None), // no v segment: pc still valid
                        };
                        if let (Some((qpl, qws)), Some((kpl, kws)), Some(vopt)) = (qp, kp, vp) {
                            let mut ws = qws;
                            ws.extend_from_slice(&kws);
                            let vpl = match vopt {
                                Some((vpl, vws)) => {
                                    ws.extend_from_slice(&vws);
                                    Some(vpl)
                                }
                                None => None,
                            };
                            qkv_pc = Some(ws);
                            q = qpl;
                            k = kpl;
                            v = vpl;
                        } else {
                            q = conv(&format!("blk.{li}.attn_q.weight"), q8_of(&lw.wq)?)?;
                            k = conv(&format!("blk.{li}.attn_k.weight"), wk_q8)?;
                            v = match lw.wv.as_ref().and_then(|v| v.q8()) {
                                Some(wv) => Some(conv(&format!("blk.{li}.attn_v.weight"), wv)?),
                                None => None,
                            };
                        }
                    } else {
                        q = conv(&format!("blk.{li}.attn_q.weight"), q8_of(&lw.wq)?)?;
                        k = conv(&format!("blk.{li}.attn_k.weight"), wk_q8)?;
                        v = match lw.wv.as_ref().and_then(|v| v.q8()) {
                            Some(wv) => Some(conv(&format!("blk.{li}.attn_v.weight"), wv)?),
                            None => None,
                        };
                    }
                    let vd = v.as_ref().map_or(0, |v| v.data.len());
                    let vs = v.as_ref().map_or(0, |v| v.scale.len());
                    let mut data = exec
                        .alloc_u8(q.data.len() + k.data.len() + vd)
                        .map_err(|e| LoadError::Tensor("f8a_wqkv".into(), e.to_string()))?;
                    let mut scale = exec
                        .alloc_u8(q.scale.len() + k.scale.len() + vs)
                        .map_err(|e| LoadError::Tensor("f8a_wqkv".into(), e.to_string()))?;
                    let mut dpos = 0usize;
                    let mut spos = 0usize;
                    for pl in [Some(&q), Some(&k), v.as_ref()].into_iter().flatten() {
                        let mut w = data
                            .try_slice_mut(dpos..dpos + pl.data.len())
                            .expect("f8a_wqkv data");
                        exec.stream.memcpy_dtod(&pl.data, &mut w).map_err(te)?;
                        dpos += pl.data.len();
                        let mut w = scale
                            .try_slice_mut(spos..spos + pl.scale.len())
                            .expect("f8a_wqkv scale");
                        exec.stream.memcpy_dtod(&pl.scale, &mut w).map_err(te)?;
                        spos += pl.scale.len();
                    }
                    // lin conversion: the serial lane row-slices this plane
                    // at q_dim and q_dim+kv_dim, so both segment boundaries
                    // must be box-aligned or the layout stays row-major
                    let (qin, qd) = (lw.wq.dims()[0], lw.wq.dims()[1]);
                    let kvd = lw.wk.dims()[1];
                    let qkv_out = qd + kvd * if lw.wv.is_some() { 2 } else { 1 };
                    let qkv = crate::gpu::RepackedMxfp4 { data, scale };
                    lw.f8a_wqkv = Some(if f8lin_attn && qd % 128 == 0 && kvd % 128 == 0 {
                        if rowvec && let Some(ws) = &qkv_pc {
                            exec.f8w_build_lin_rw(qkv.data, &wse_of(ws), qin, qkv_out, false)?
                        } else {
                            lin(true, qkv, qin, qkv_out)?
                        }
                    } else {
                        qkv
                    });
                    if let Some(ws) = qkv_pc.take() {
                        lw.qkv_ws = Some(
                            exec.stream
                                .clone_htod(&ws)
                                .map_err(|e| LoadError::Tensor("qkv_ws".into(), e.to_string()))?,
                        );
                    }
                } else {
                    lw.f8a_wq = Some(f8w_native(
                        &format!("blk.{li}.attn_q.weight"),
                        q8_of(&lw.wq)?,
                    )?);
                    lw.f8a_wk = match lw.wk.q8() {
                        Some(wk) => Some(f8w_native(&format!("blk.{li}.attn_k.weight"), wk)?),
                        None => None,
                    };
                    lw.f8a_wv = match lw.wv.as_ref().and_then(|v| v.q8()) {
                        Some(wv) => Some(f8w_native(&format!("blk.{li}.attn_v.weight"), wv)?),
                        None => None,
                    };
                }
                // lin() no-ops under fp4 (the gate requires !fp4), so the
                // e2m1 conv output passes through untouched
                // Plane twin: bf16 planes are never stubbed (see below)
                let stub_plane = |w: &mut Plane| -> Result<(), LoadError> {
                    let Plane::Q8(q) = w else { return Ok(()) };
                    let dims = q.dims.clone();
                    *w = Plane::Q8(crate::gpu::RepackedQ8 {
                        data: exec
                            .alloc_u8(32)
                            .map_err(|e| LoadError::Tensor("f8a stub".into(), e.to_string()))?,
                        scale: exec
                            .alloc_u8(32)
                            .map_err(|e| LoadError::Tensor("f8a stub".into(), e.to_string()))?,
                        dims,
                    });
                    Ok(())
                };
                // Same one-call treatment for wo on the Q8-derived route: one
                // allocation, no transient to free. See the qkv arm above.
                if !fp4 && fp8_native.is_none() {
                    let (win, wout) = (lw.wo.dims()[0], lw.wo.dims()[1]);
                    let use_lin = f8lin_attn && win % 128 == 0 && wout % 16 == 0;
                    lw.f8a_wo = Some(if attn_skipped[li] {
                        let src = exec.repack_q8(map, &format!("blk.{li}.attn_output.weight"))?;
                        exec.q8_0_to_f8w_lin(&[&src], win, wout, use_lin)?
                    } else {
                        exec.q8_0_to_f8w_lin(&[q8_of(&lw.wo)?], win, wout, use_lin)?
                    });
                    // Already seats, not planes, where the upload was skipped -
                    // stubbing again would allocate a second pair per tensor.
                    if attn_skipped[li] {
                        continue;
                    }
                    stub_plane(&mut lw.wq)?;
                    if lw.f8a_wqkv.is_some() || lw.f8a_wk.is_some() {
                        stub_plane(&mut lw.wk)?;
                        if let Some(wv) = &mut lw.wv {
                            stub_plane(wv)?;
                        }
                    }
                    stub_plane(&mut lw.wo)?;
                    continue;
                }
                let mut wo_pc_ws: Option<Vec<f32>> = None;
                let wo_plane = if !fp4 {
                    match f8w_pc(&format!("blk.{li}.attn_output.weight"), q8_of(&lw.wo)?)? {
                        Some((plane, ws)) => {
                            lw.wo_ws =
                                Some(exec.stream.clone_htod(&ws).map_err(|e| {
                                    LoadError::Tensor("wo_ws".into(), e.to_string())
                                })?);
                            wo_pc_ws = Some(ws);
                            plane
                        }
                        None => conv(&format!("blk.{li}.attn_output.weight"), q8_of(&lw.wo)?)?,
                    }
                } else {
                    conv(&format!("blk.{li}.attn_output.weight"), q8_of(&lw.wo)?)?
                };
                lw.f8a_wo = Some(match wo_pc_ws {
                    Some(ws)
                        if rowvec
                            && f8lin_attn
                            && lw.wo.dims()[0] % 128 == 0
                            && lw.wo.dims()[1] % 16 == 0 =>
                    {
                        exec.f8w_build_lin_rw(
                            wo_plane.data,
                            &wse_of(&ws),
                            lw.wo.dims()[0],
                            lw.wo.dims()[1],
                            false,
                        )?
                    }
                    _ => lin(f8lin_attn, wo_plane, lw.wo.dims()[0], lw.wo.dims()[1])?,
                });
                // muse-glimmer's attention output gate. Built exactly like the
                // split-branch q above (same `conv`, no lin pass) because it is
                // wq's twin - same [n_embd -> n_head*head_dim] dims, same
                // post-attn_norm input - so it must land on whatever route wq
                // lands on. Until this existed the gate was the one per-layer
                // GEMM with no fp8 plane in any family, which is how it stayed
                // on q8_0_gemm_repacked through the whole prefill.
                // a plane only gets stubbed once something else serves it:
                // wq/wo always have an f8a twin here, k/v only when they were
                // in the Q8 class to begin with (a bf16 k/v is the serving
                // plane - freeing it would leave nothing behind)
                stub_plane(&mut lw.wq)?;
                if lw.f8a_wqkv.is_some() || lw.f8a_wk.is_some() {
                    stub_plane(&mut lw.wk)?;
                    if let Some(wv) = &mut lw.wv {
                        stub_plane(wv)?;
                    }
                }
                stub_plane(&mut lw.wo)?;
            }
        }

        // Every seat whose upload was skipped must have ended up with a twin.
        // `attn_never_upload` was computed from the phase gates before the
        // loop, so if any of them has since disagreed we have planes with no
        // bytes behind them - refuse the load instead of serving from a
        // 32-byte stub. This is the check that makes the skip safe to make
        // early: two shipped corruptions on this pattern came from a coverage
        // argument nobody verified.
        for (li, lw) in layers.iter().enumerate() {
            if attn_skipped.get(li).copied() != Some(true) {
                continue;
            }
            if lw.f8a_wqkv.is_none() || lw.f8a_wo.is_none() {
                return Err(LoadError::Tensor(
                    format!("blk.{li}.attn_*"),
                    "gemma4 skipped the Q8_0 attention upload for this layer but f8a did not \
                     build its e4m3 twin -- the seats hold 32-byte stubs with nothing behind \
                     them. This is a LOADER BUG in the attn_never_upload precondition, not a \
                     config error. Re-run with PADDOCK_G4_NO_F8A=1 to keep the Q8_0 attention \
                     planes resident while it is fixed."
                        .into(),
                ));
            }
        }
        if f8a {
            vram_mark(
                "f8a REPLACE (attn e4m3 in, q8 attn planes out)",
                &mut vram_prev,
            );
        }

        // Q8-original reclaim: with the f8w prefill planes, the f8t
        // decode tile planes, and the serial-lane f8w gemv arms covering
        // every live path, the Q8_0 layer originals' only remaining readers
        // are compiled fallback rungs that cannot engage while the planes
        // exist. Stub them to 32 bytes (dims kept for shape lookups):
        // ~29 GB back to the KV pool on gemma4-31B. Per-layer all-planes
        // guard; kill: PADDOCK_NO_Q8_RECLAIM.
        if f8w_pf
            && f8t_dec
            && paddock_models::dev_var_os!("PADDOCK_NO_Q8_RECLAIM").is_none()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_F8ATT").is_none()
        {
            let mut freed: u64 = 0;
            for lw in layers.iter_mut() {
                // every lane the stubs orphan must have an f8t serving
                // plane (f8w is no longer built by default - the
                // f8t launcher covers r==1 through prefill)
                let ok = (lw.f8t_qkv.is_some()
                    || (lw.f8t_wq.is_some()
                        && lw.f8t_wk.is_some()
                        && (lw.wv.is_none() || lw.f8t_wv.is_some())))
                    && lw.f8t_wo.is_some()
                    && (lw.f8t_gu.is_some() || (lw.f8t_gate.is_some() && lw.f8t_up.is_some()))
                    && lw.f8t_down.is_some();
                if !ok {
                    continue;
                }
                // 48-BYTE sentinel, deliberately not the F8R 32-byte one:
                // len<=32 flips the F8R prefill ladders (different plane
                // family, never validated here - it ILLEGAL_ADDRESSed on the
                // first cut). 48 keeps every prefill/decode route exactly as
                // pre-reclaim; only the serial gemv arms test for it.
                // A bf16 or k-quant plane returns 0 and keeps its bytes - it
                // is the serving plane for its tensor, so there is nothing to
                // reclaim to. (The `ok` guard above already refuses a layer
                // whose f8t k/v twins are missing, which is exactly that case;
                // this keeps the property local to the stub rather than
                // load-bearing on a predicate 40 lines up.)
                let stub_plane = |w: &mut Plane| -> Result<u64, LoadError> {
                    let Plane::Q8(q) = w else { return Ok(0) };
                    let bytes = (q.data.len() + q.scale.len()) as u64;
                    let dims = q.dims.clone();
                    *w = Plane::Q8(crate::gpu::RepackedQ8 {
                        data: exec.alloc_u8(48).map_err(|e| {
                            LoadError::Tensor("q8 reclaim stub".into(), e.to_string())
                        })?,
                        scale: exec.alloc_u8(48).map_err(|e| {
                            LoadError::Tensor("q8 reclaim stub".into(), e.to_string())
                        })?,
                        dims,
                    });
                    Ok(bytes)
                };
                // DEFAULT = ffn (20.6 GiB, request-validated). attn-only
                // also validates (8.4 GiB) but the COMBINATION illegal-
                // addresses on the first batched prefill - a consumer or
                // layout interaction the per-set runs don't reproduce;
                // needs the memcheck session before "all" can default.
                // DEFAULT off (WIP): the consumer matrix is wider
                // than the plane audit found - the DECODE walk has per-r
                // arms and the r==1 band reads Q8 FFN gemv (c1/serial ticks
                // degenerated to <pad> spam under the ffn stub; attn+ffn
                // combined ILLEGAL_ADDRESSes on batched prefill). Ship
                // needs the full band x family x lane matrix closed with
                // TEXT-validated probes per cell. Opt-in for that session:
                // PADDOCK_Q8_RECLAIM_SET=ffn|attn|all.
                // DEFAULT "all" : the Act-40 blocker was the batched
                // r==1 band (batch.rs) still reading q8 gemvs - the serial
                // arms alone never covered c1, which the batched engine
                // serves through batch.rs. With the four batched r==1
                // reclaim arms in place the full bisect TEXT-validates
                // (ffn 20.6 / attn 8.4 / all 28.98 GiB freed, coherent,
                // 12-way burst clean) and a memcheck-served request reports
                // zero invalid accesses. e4m3 gemv at r==1 joins the class
                // decode r 2..64 already serves (f8t). Env still
                // bisects; PADDOCK_NO_Q8_RECLAIM kills wholesale (checked
                // in the outer gate).
                let set = paddock_models::dev_var!("PADDOCK_Q8_RECLAIM_SET")
                    .unwrap_or_else(|_| "all".into());
                let do_attn = set == "all" || set == "attn";
                let do_ffn = set == "all" || set == "ffn";
                if do_attn {
                    freed += stub_plane(&mut lw.wq)?;
                    freed += stub_plane(&mut lw.wk)?;
                    if let Some(wv) = &mut lw.wv {
                        freed += stub_plane(wv)?;
                    }
                    freed += stub_plane(&mut lw.wo)?;
                }
                if do_ffn {
                    freed += stub_plane(&mut lw.ffn_gate)?;
                    freed += stub_plane(&mut lw.ffn_up)?;
                    freed += stub_plane(&mut lw.ffn_down)?;
                }
            }
            tracing::info!(
                "gemma4: Q8-original reclaim freed {:.2} GiB (f8w/f8t planes serve all lanes)",
                freed as f64 / (1u64 << 30) as f64
            );
            vram_mark("q8-original reclaim", &mut vram_prev);
        }
        // staging buffers released before anything pool-sized allocates
        pc_stage.borrow_mut().take();
        nat_stage.borrow_mut().take();
        // KV: SWA layers ride the WindowRing block pools when the pack has
        // the paged kernels (default); global layers are always dense planes.
        // Single slot here - enable_batch rebuilds for n_slots.
        let paged = paddock_models::dev_var_os!("PADDOCK_NO_PAGED_KV").is_none()
            && exec.has_paged_kv()
            && exec.has_attn_prefill_f16_paged()
            // %64: the paged SWA prefill rides the WMMA f16 tile unconditionally
            && max_ctx.is_multiple_of(64)
            && swa_window > 0;
        // Load starts at the widest span (or the operator's pin);
        // `enable_batch` re-elects it down the ladder if the asked-for width
        // will not otherwise fit, and rebuilds this ring to match.
        let swa_span = super::forward::swa_span_initial();
        let paging = if paged {
            // no tower yet: a picture no longer than a sub-span needs no
            // allowance, and `enable_batch` re-sizes once the tower is known
            Some(super::batch::build_swa_paging(
                &exec, max_ctx, swa_window, 1, swa_span, 0,
            )?)
        } else {
            None
        };
        // weights are complete here (uploads + planes + reclaim done, KV not
        // yet allocated) - this snapshot is the weights line of the
        // memory-breakdown API. `settled_` and not the raw read: it owns the
        // sync-then-trim that keeps freed staging out of the number.
        let weights_bytes = exec.settled_mem_used();
        // load-time serial alloc: the constructor runs before `set_kv_dtype`
        // can (serving.rs applies it right after load), so no pref yet -
        // this slots=1 alloc is always non-pooled and therefore f16 anyway;
        // enable_batch re-allocs with the pref.
        let kv = super::batch::alloc_kv(&exec, &layers, max_ctx, paging.as_ref(), None, 1, None)?;
        vram_mark("serial KV (1 slot)", &mut vram_prev);

        // Prefill scratch rows: the chunk width this server STARTS at.
        // `enable_batch` may step it down - the planes are ~0.62 MiB/row on
        // gemma-4-31B, so at ctx 16384 the full 8192-row set is 4.99 GiB taken
        // off the budget before the KV planner is asked anything, and the plan
        // then refuses beside it. Every prefill lane splits at
        // `GpuGemma4::pf_rows` (the live value), never at this one.
        let pf_rows = super::forward::pf_rows(max_ctx);

        let scratch_dims = super::scratch::ScratchDims {
            n_embd,
            n_head,
            hd_global,
            n_vocab,
            n_ff,
            max_q: n_head * hd_global,
            max_kv: layers
                .iter()
                .map(|l| l.n_kv_heads * l.head_dim)
                .max()
                .expect("layers non-empty"),
            max_qkv: layers
                .iter()
                .map(|l| {
                    n_head * l.head_dim
                        + l.n_kv_heads * l.head_dim * if l.wv.is_some() { 2 } else { 1 }
                })
                .max()
                .unwrap_or(n_head * hd_global),
            gated: layers.iter().any(|l| l.attn_gate.is_some()),
            n_expert,
            n_expert_used,
            ff_exp,
            f8_on,
            f8row,
            f8w_pf,
            f8a,
            f8t_dec,
            kq_dense: file_kq,
            kq_moe: layers
                .iter()
                .any(|l| l.moe.as_ref().is_some_and(|m| m.experts.kq().is_some())),
            // uniq-routing diagnostic (PADDOCK_MOE_UNIQ, MoE models only): raw
            // non-pool accumulator + detached dumper thread - see the Scratch
            // field comment for the two measured constraints. Armed once here
            // rather than inside the builder: the buffer is leaked by design
            // (process lifetime), so arming it per build would leak one per
            // rung of enable_batch's chunk ladder.
            moe_uniq_dev: if n_expert != 0
                && paddock_models::dev_var_os!("PADDOCK_MOE_UNIQ").is_some()
            {
                g4_moe_uniq_arm(&exec)?
            } else {
                0
            },
        };
        let scratch = super::scratch::build(&exec, &scratch_dims, pf_rows)?;

        vram_mark(
            &format!("serial scratch ({pf_rows} pf rows)"),
            &mut vram_prev,
        );
        // DiffusionGemma: the self-conditioning MLP and the transposed
        // embedding, on top of the body every other arch stops at.
        let diffusion = if arch.is_diffusion() {
            let lane = super::diffusion::DiffusionLane::load(
                &exec,
                map,
                super::EmbdTable::of(&token_embd, &head),
                n_embd,
                n_vocab,
                canvas_len,
                ffn_align,
            )?;
            vram_mark("diffusion lane (self-cond MLP + E^T bf16)", &mut vram_prev);
            Some(lane)
        } else {
            None
        };
        Ok(Self {
            hp: Hparams {
                arch,
                n_layer,
                n_embd,
                n_head,
                n_vocab,
                eps,
                // gemma4 norms everything at one eps; muse-glimmer's reference
                // pins the two POST norms to 1e-8 and leaves the pre-norms at
                // the header's 1e-5. Not a metadata key on either file.
                post_norm_eps: match arch {
                    Arch::Gemma4 | Arch::DiffusionGemma => eps,
                    Arch::MuseGlimmer => 1e-8,
                },
                swa_window,
                rope_swa: plain_rope(base_swa, hd_swa),
                // NoPE on the full-attention layers: freq_scale 0 makes the
                // shared yarn kernels a bit-exact identity (see Hparams).
                rope_global: match arch {
                    Arch::Gemma4 | Arch::DiffusionGemma => plain_rope(base_global, hd_global),
                    Arch::MuseGlimmer => {
                        let (ts, _, cl, ch, ef, ms) = plain_rope(base_global, hd_global);
                        (ts, 0.0, cl, ch, ef, ms)
                    }
                },
                final_softcap,
                logit_scale,
                embd_rmsnorm: matches!(arch, Arch::MuseGlimmer),
                n_expert,
                n_expert_used,
                ff_exp,
                canvas_len,
            },
            exec,
            layers,
            token_embd,
            head,
            embd_ones,
            head_f8t,
            head_f8row,
            output_norm,
            rope_factors,
            kv,
            weights_bytes,
            content_id: (
                crate::kv_tier::fingerprint::weights(map),
                crate::kv_tier::fingerprint::tokenizer(map),
            ),
            scratch,
            scratch_dims,
            max_ctx,
            pf_rows,
            swa_span,
            swa_overshoot: 0,
            pos: 0,
            n_slots: 1,
            paging,
            kv_dtype_pref: None,
            prefix: None,
            vision: None,
            // budgeted for the load-time pass until enable_batch elects one
            pictures: crate::gpu_model::picture_store::PictureStore::new(
                (2 * pf_rows * n_embd * std::mem::size_of::<f32>()) as u64,
            ),
            img_beg_id,
            img_end_id,
            batch_logits: None,
            samp: None,
            fin_samp: None,
            samp_tpar: None,
            fin_tpar: None,
            d_tokens: None,
            d_slots: None,
            canvases: Vec::new(),
            gpool: None,
            decode_graphs: std::collections::HashMap::new(),
            graph_seen: std::collections::HashSet::new(),
            prefill_graphs: std::collections::HashMap::new(),
            prefill_graph_seen: std::collections::HashSet::new(),
            attn_scratch: None,
            lco_tickets: None,
            chunked: Vec::new(),
            mtp: None,
            dflash: None,
            dflash_defer: false,
            dflash_fuse_wanted: true,
            spec_rows: Vec::new(),
            mix_inflight: None,
            mix_fallback: None,
            spec_stats: Default::default(),
            spec_async: None,
            toks_dev: false,
            ids_skip: false,
            spec_strip: None,
            spec_rs_draws: None,
            spec_rs_chain: None,
            want_strip: false,
            spec_pipe_cfg: None,
            spec_pipe_slots: Default::default(),
            spec_pipe: None,
            pipe_copy: None,
            pipe: None,
            d_pipe_out: None,
            spec_k1: None,
            spec_long: false,
            attn_pos_max: 0,
            spec_shallow: false,
            mtp_graphs: std::collections::HashMap::new(),
            pf_side: None,
            diffusion,
        })
    }
}

/// arm the uniq-routing diagnostic - a RAW cuMemAlloc accumulator
/// (outside the mempool:  first-decode sweep overwrites pool
/// placements) plus a detached thread that copies the ~4KB down every 5s
/// and rewrites the JSON at $PADDOCK_MOE_UNIQ (pkill-safe; independent of
/// serving code paths, which replay captured graphs and run no host code
/// at steady decode). Returns the device pointer; leaked at process end.
pub(crate) fn g4_moe_uniq_arm(exec: &GpuExecutor) -> Result<u64, LoadError> {
    use cudarc::driver::result as cur;
    let words = 4 * 260usize;
    let dev = unsafe { cur::malloc_sync(words * 4) }
        .map_err(|e| LoadError::Tensor("moe_uniq raw alloc".into(), e.to_string()))?;
    unsafe { cur::memset_d8_sync(dev, 0, words * 4) }
        .map_err(|e| LoadError::Tensor("moe_uniq zero".into(), e.to_string()))?;
    let path = paddock_models::dev_var!("PADDOCK_MOE_UNIQ").unwrap_or_default();
    tracing::info!("moe_uniq routing diagnostic ARMED (raw dev buffer, 5s dumper) -> {path}");
    let ctx = exec.stream.context().clone();
    std::thread::spawn(move || {
        if ctx.bind_to_thread().is_err() {
            return;
        }
        let mut host = vec![0u32; words];
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            if unsafe { cur::memcpy_dtoh_sync(&mut host, dev) }.is_err() {
                continue;
            }
            if path.is_empty() {
                continue;
            }
            // {"bands":{"le64":{"invocations":..,"pairs":..,"hist":{..},
            //  "pairs_sum":{..}},..},"corrupt":..} - nonzero buckets only.
            // corrupt: a hist bucket can never exceed its band's count -
            // the tripwire that caught the pool placements.
            let corrupt = (0..4).any(|b| {
                let inv = host[b * 260 + 258];
                (0..=128).any(|u| host[b * 260 + u] > inv)
            });
            let mut s = String::with_capacity(8192);
            s.push_str("{\"bands\":{");
            for (i, name) in ["le64", "le256", "le1024", "gt1024"]
                .into_iter()
                .enumerate()
            {
                let base = i * 260;
                if i != 0 {
                    s.push(',');
                }
                s.push_str(&format!(
                    "\"{name}\":{{\"invocations\":{},\"pairs\":{},\"hist\":{{",
                    host[base + 258],
                    host[base + 259]
                ));
                let mut first = true;
                for u in 0..=128usize {
                    if host[base + u] != 0 {
                        if !first {
                            s.push(',');
                        }
                        s.push_str(&format!("\"{u}\":{}", host[base + u]));
                        first = false;
                    }
                }
                s.push_str("},\"pairs_sum\":{");
                let mut first = true;
                for u in 0..=128usize {
                    if host[base + 129 + u] != 0 {
                        if !first {
                            s.push(',');
                        }
                        s.push_str(&format!("\"{u}\":{}", host[base + 129 + u]));
                        first = false;
                    }
                }
                s.push_str("}}");
            }
            s.push_str(&format!("}},\"corrupt\":{corrupt}}}"));
            let _ = std::fs::write(&path, s);
        }
    });
    Ok(dev)
}
