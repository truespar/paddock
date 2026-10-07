//! Checkpoint-dir loader - every residency byte-exact to the file (charter:
//! bf16 parity first). Big GEMM planes keep their bf16 bytes on device;
//! small vectors widen to f32 (exact - bf16 is the top half of f32); routed
//! experts ride `nvf4_moe_upload` with the checkpoint's own nibbles.

use std::sync::Arc;

use crate::gpu::{DeviceTensor, GpuError, GpuExecutor, Nvf4MoePlane, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{
    bf16_bytes, bf16_concat_plane, bf16_plane, bf16_to_f16_exact, bf16_to_f32, f32_tensor,
    nvf4_expert_stack,
};
use paddock_models::ggml_type::GgmlType;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::*;
use super::{DenseClass, DensePlane, dense_class_from_env};

pub(super) fn dt(
    exec: &GpuExecutor,
    v: Vec<f32>,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuError> {
    Ok(DeviceTensor {
        buf: exec.to_device(&v)?,
        dims,
    })
}

/// f32 exact widen of a small tensor.
fn f32_dt(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuModelError> {
    let n: usize = dims.iter().product();
    dt(exec, f32_tensor(st, name, n)?, dims).map_err(GpuModelError::from)
}

/// Gemma (1+w) fold for the norms the pack's PLAIN-w kernels consume
/// (attention q/k per-head RMSNorm). The checkpoint stores raw `w`, and
/// `rmsnorm_batch` computes `x*inv*w` - so `w+1` on the host is the reference
/// (`examples/q38fn_host_forward.rs` folds the same way before calling the
/// plain-w reference). The hyper-connection and PLE norms do not come through
/// here: their kernel (`q4x_group_norm_1p`) applies the (1+w) FMA itself.
fn f32_dt_1p(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuModelError> {
    let n: usize = dims.iter().product();
    let v: Vec<f32> = f32_tensor(st, name, n)?
        .into_iter()
        .map(|w| w + 1.0)
        .collect();
    dt(exec, v, dims).map_err(GpuModelError::from)
}

/// conv1d [rows, 1, k] -> device f32 [rows, k] (squeeze the middle axis).
fn conv_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    rows: usize,
    k: usize,
) -> Result<DeviceTensor, GpuModelError> {
    f32_dt(exec, st, name, vec![rows, k]).map_err(|e| match e {
        GpuModelError::Unsupported(m) => GpuModelError::Unsupported(format!("{m} (conv [r,1,k])")),
        other => other,
    })
}

/// MXFP8 (OCP microscaling) straight off the checkpoint, when the file ships
/// its dense planes that way: `weight` is e4m3 and `weight_scale` is one
/// ue8m0 byte per 32 values along K.
///
/// Detected, never elected. `DenseClass` picks among representations of a
/// bf16 source plane (keep it, narrow it to f16, add a twin, squeeze it to
/// per-row e4m3); MXFP8 is not a representation we choose, it is what the
/// checkpoint holds, and there is no bf16 underneath to fall back to. So this
/// runs before the class election and the election never sees these planes.
///
/// Byte passthrough: `RepackedMxfp4 {data, scale}` is this pair's layout -
/// out-row-major payload, scale plane `[out_dim, in_dim/32]` - which
/// `f8_gemv_at_off` already documents as "data += row_off*in, scale +=
/// row_off*in/32". Nothing is dequantized, requantized or rounded on the way
/// in, so the served weights are the checkpoint's own bits.
fn mxfp8_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Option<Result<DensePlane, GpuModelError>> {
    use paddock_models::safetensors::StDtype;
    let (t, wb) = st.bytes(name)?;
    if t.dtype != StDtype::F8E4m3 {
        return None;
    }
    // modelopt names the companion `<tensor>_scale`, i.e. `....weight_scale`
    let scale_name = format!("{name}_scale");
    let (ts, sb) = st.bytes(&scale_name)?;
    // k is the input dim; the scale plane is [out, in/32]
    let want_scale = [n, k / 32];
    if ts.dtype != StDtype::U8 || !k.is_multiple_of(32) || ts.shape != want_scale {
        return Some(Err(GpuModelError::Unsupported(format!(
            "{scale_name}: {:?} {:?} - want u8 {want_scale:?} (MXFP8 block [1, 32])",
            ts.dtype, ts.shape
        ))));
    }
    if wb.len() != n * k || sb.len() != n * k / 32 {
        return Some(Err(GpuModelError::Unsupported(format!(
            "{name}: {} payload / {} scale bytes, want {} / {}",
            wb.len(),
            sb.len(),
            n * k,
            n * k / 32
        ))));
    }
    Some((|| {
        let data = exec.to_device_u8(wb).map_err(GpuModelError::from)?;
        let scale = exec.to_device_u8(sb).map_err(GpuModelError::from)?;
        Ok(DensePlane::Mxf8 {
            plane: crate::gpu::RepackedMxfp4 { data, scale },
            in_dim: k,
            out_dim: n,
        })
    })())
}

/// Whether this plane is MXFP8 in the file. The bf16-only fusions (row
/// concats, the inject fold) have to ask before they reach for the bytes:
/// their whole trick is a byte concatenation, which a payload+scale pair
/// cannot serve.
fn is_mxfp8(st: &ShardedSafetensors, name: &str) -> bool {
    use paddock_models::safetensors::StDtype;
    st.bytes(name)
        .is_some_and(|(t, _)| t.dtype == StDtype::F8E4m3)
        && st.bytes(&format!("{name}_scale")).is_some()
}

/// Load one dense projection in the elected class. bf16 is the parity class
/// and the default; the 8-bit lane is opt-in (see `DenseClass`). A checkpoint
/// that ships MXFP8 dense planes bypasses the election entirely - see
/// [`mxfp8_plane`].
fn dense(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Result<DensePlane, GpuModelError> {
    if let Some(r) = mxfp8_plane(exec, st, name, n, k) {
        return r;
    }
    match dense_class_from_env() {
        DenseClass::Bf16 => Ok(DensePlane::Bf16(bf16_plane(exec, st, name, n, k)?)),
        DenseClass::F16 => {
            if !exec.has_f16_gemm() {
                return Err(GpuModelError::Unsupported(
                    "PADDOCK_Q38FN_DENSE=f16 but the pack has no f16_gemm (slot 383)".into(),
                ));
            }
            // bf16 -> f16 is exact for every value inside f16's normal range
            // (bf16 carries 8 mantissa bits, f16 carries 11), and
            // `narrow_to_f16` REFUSES the plane if any weight overflows, so a
            // checkpoint this does not hold for fails at load rather than
            // silently losing its tail.
            let raw = bf16_bytes(st, name, n * k)?;
            Ok(DensePlane::F16 {
                w: exec
                    .f16_to_device(&bf16_to_f16_exact(raw, name)?)
                    .map_err(GpuModelError::from)?,
                in_dim: k,
                out_dim: n,
            })
        }
        DenseClass::Dual => {
            if !exec.has_f16_gemm() {
                return Err(GpuModelError::Unsupported(
                    "PADDOCK_Q38FN_DENSE=dual but the pack has no f16_gemm (slot 383)".into(),
                ));
            }
            let raw = bf16_bytes(st, name, n * k)?;
            Ok(DensePlane::Dual {
                w16: exec
                    .f16_to_device(&bf16_to_f16_exact(raw, name)?)
                    .map_err(GpuModelError::from)?,
                w: QuantTensor {
                    bytes: exec.to_device_u8(raw).map_err(GpuModelError::from)?,
                    ty: GgmlType::Bf16,
                    dims: vec![k, n],
                },
                in_dim: k,
                out_dim: n,
            })
        }
        DenseClass::F8Row => {
            if !exec.has_bf16_to_f8row() {
                return Err(GpuModelError::Unsupported(
                    "PADDOCK_Q38FN_DENSE=f8row but the pack has no bf16_to_f8row".into(),
                ));
            }
            // Straight from the checkpoint's own bf16 - no Q8 hop, which would
            // double-quantize (the edge `bf16_to_f8row` exists to close).
            let raw = bf16_bytes(st, name, n * k)?;
            let plane = exec.bf16_to_f8row(raw, k, n).map_err(GpuModelError::from)?;
            Ok(DensePlane::F8Row {
                plane,
                in_dim: k,
                out_dim: n,
            })
        }
    }
}

/// The lm_head rides the same class election as every other dense plane -
/// it is 1.27 GB of the per-token traffic on its own.
pub(crate) fn dense_head(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Result<DensePlane, GpuModelError> {
    dense(exec, st, name, n, k)
}

pub(crate) fn hc_weights(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    pfx: &str,
    inject: bool,
) -> Result<HcW, GpuModelError> {
    let hw = c.hc_width();
    let (lr, hc) = (c.hc_lowrank, c.hc_count);
    let down_name = format!("{pfx}.input_mix_weight_down.weight");
    let inj_name = format!("{pfx}.block_inject_weight.weight");
    // Fold the inject rows onto the down plane when we may: both projections
    // read the same normalized state, so this turns two launches into one at
    // batch 1 (and two ROW-SEGMENT reads of one residency above it). Refused
    // when the dense class is 8-bit - appending f32 inject weights to a
    // quantized plane would quantize them, a numerics change a launch fold is
    // not entitled to make - and when the checkpoint does not store the
    // inject as bf16, since the concatenation is a BYTE concatenation.
    let inj_is_bf16 = st
        .bytes(&inj_name)
        .map(|(t, _)| t.dtype == StDtype::Bf16)
        .unwrap_or(false);
    let fold = inject
        && matches!(dense_class_from_env(), DenseClass::Bf16 | DenseClass::Dual)
        && inj_is_bf16;
    // one-shot witness: the unfolded path costs 1.24 ms/tick at c8 (two
    // pd_matvec_f32_batch launches, 18.7 us + 7.3 us, for an 82 KB plane),
    // so which branch this takes is worth knowing at a glance.
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[hc-fold] fold={fold} inject={inject} inj_is_bf16={inj_is_bf16}                  class={:?} inj_name={inj_name}",
                dense_class_from_env()
            );
        }
    }
    let norm = f32_dt(exec, st, &format!("{pfx}.hc_norm.weight"), vec![hw])?;
    let up = dense(
        exec,
        st,
        &format!("{pfx}.input_mix_weight_up.weight"),
        hw,
        lr,
    )?;
    // Permuted twin for the FUSED mix epilogue (slots 530-531). Built once at
    // load; declines to None when the pack lacks the slots or the shape does
    // not qualify, and the forward keeps the two-launch path in that case.
    // the Dual class keeps its bf16 half, so the fused mix epilogue survives
    let up_hcmix = match &up {
        DensePlane::Bf16(w) | DensePlane::Dual { w, .. } => exec
            .bf16_hcmix_permute(w, hw / hc, hc)
            .map_err(GpuModelError::from)?,
        _ => None,
    };
    if fold {
        let mut raw = bf16_bytes(st, &down_name, lr * hw)?.to_vec();
        raw.extend_from_slice(bf16_bytes(st, &inj_name, hc * hw)?);
        let plane = QuantTensor {
            bytes: exec.to_device_u8(&raw).map_err(GpuModelError::from)?,
            ty: GgmlType::Bf16,
            dims: vec![hw, lr + hc],
        };
        // the folded hc down plane is a byte concatenation, so it is bf16-only;
        // under Dual it keeps the bf16 half and simply never takes the f16
        // twin (it is a 2-segment store, which has no f16 entry anyway)
        // LOW-M ISLAND (slots 571/572), folded lane only: the inject rows
        // must live in the same plane so the island can read them as their
        // own aligned gemm window.
        // the down-only arm needs down_p42 too, so build when either is armed
        let (down_p42, up_p42) = if super::hc_island_on() || super::hc_down_p42_on() {
            build_hc_island(exec, &plane, &up, lr, hc, hw)?
        } else {
            (None, None)
        };
        return Ok(HcW {
            norm,
            down: DensePlane::Bf16(plane),
            lowrank: lr,
            inject_rows: hc,
            up,
            up_hcmix,
            down_p42,
            up_p42,
            inject: None,
        });
    }
    Ok(HcW {
        norm,
        down: dense(exec, st, &down_name, lr, hw)?,
        lowrank: lr,
        inject_rows: 0,
        up,
        up_hcmix,
        // island needs the folded inject rows; unfolded lanes keep the chain
        down_p42: None,
        up_p42: None,
        inject: if inject {
            Some(f32_dt(exec, st, &inj_name, vec![hc, hw])?)
        } else {
            None
        },
    })
}

fn moe_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    pfx: &str,
    role: &str,
    rows: usize,
    in_dim: usize,
) -> Result<Nvf4MoePlane, GpuModelError> {
    nvf4_expert_stack(
        exec,
        st,
        |e| format!("{pfx}.mlp.experts.{e}.{role}"),
        c.n_expert,
        rows,
        in_dim,
    )
}

/// bf16 twin of a host f32 plane for the TGV lane (slot 547): built only
/// when the lane is armed, uploaded as raw bf16 bytes (the TGV wrapper
/// takes CudaSlice<u8>, mirroring QuantTensor::bytes).
/// bf16 twin with the ROW COUNT padded up to `align` (zero rows). The
/// vendored low-M dense GEMM carries the output width in its MMA-M tile, so a
/// 513-row router has to reach 576 before it can run at all; the padding rows
/// are zeros and the consumers read the real rows by stride.
/// Build the LOW-M HC island's two planes (slots 571/572):
///   down_p42 = `down` ([lr+hc, hw] bf16) padded to a multiple of 64 rows, so
///              the inject block at rows [lr, lr+hc) can be read as its own
///              gemm window (the gemm carries N in its MMA-M tile).
///   up_p42   = `up` ([hw, lr] bf16) in the GATE epilogue's row order - branch
///              s of hidden d at row d*hc+s - padded to kpad columns.
fn build_hc_island(
    exec: &GpuExecutor,
    down: &QuantTensor,
    up: &DensePlane,
    lr: usize,
    hc: usize,
    hw: usize,
) -> Result<
    (
        Option<cudarc::driver::CudaSlice<u8>>,
        Option<cudarc::driver::CudaSlice<u8>>,
    ),
    GpuModelError,
> {
    let Some(upq) = super::plane_bytes(up) else {
        return Ok((None, None));
    };
    let kpad = lr.div_ceil(128) * 128;
    let rows_dst = (lr + hc).div_ceil(64) * 64;
    let mut dp = exec
        .alloc_u8(rows_dst * hw * 2)
        .map_err(GpuModelError::from)?;
    exec.bf16_pad_rows(&down.bytes, &mut dp, lr + hc, rows_dst, hw)
        .map_err(GpuModelError::from)?;
    let mut upp = exec.alloc_u8(hw * kpad * 2).map_err(GpuModelError::from)?;
    exec.bf16_hc_perm_pad(&upq.bytes, &mut upp, hw / hc, hc, lr, kpad)
        .map_err(GpuModelError::from)?;
    // one-shot self-check of both prep kernels: the island's kernels verify
    // clean on synthetic data, so a divergence in the walk points here.
    if super::dense_witness_on() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            let bf = |b: &[u8], i: usize| {
                half::bf16::from_bits(u16::from_le_bytes([b[i * 2], b[i * 2 + 1]])).to_f32()
            };
            let src_d = exec
                .to_host_u8_len(&down.bytes, 16)
                .map_err(GpuModelError::from)?;
            let got_d = exec.to_host_u8_len(&dp, 16).map_err(GpuModelError::from)?;
            // inject row lr must land at row lr of the padded plane
            let si = exec
                .to_host_u8_len(&down.bytes, (lr * hw + 4) * 2)
                .map_err(GpuModelError::from)?;
            let gi = exec
                .to_host_u8_len(&dp, (lr * hw + 4) * 2)
                .map_err(GpuModelError::from)?;
            // up: dst row d*hc+s col j == src row s*hidden+d col j
            let (hs, d, sbr, j) = (hw / hc, 3usize, 2usize, 5usize);
            let su = exec
                .to_host_u8_len(&upq.bytes, ((sbr * hs + d) * lr + j + 1) * 2)
                .map_err(GpuModelError::from)?;
            let gu = exec
                .to_host_u8_len(&upp, ((d * hc + sbr) * kpad + j + 1) * 2)
                .map_err(GpuModelError::from)?;
            eprintln!(
                "[hc-island] down[0]={:.5}/{:.5} inject[lr]={:.5}/{:.5} up[{}]={:.5}/{:.5} \
                 (src/built)",
                bf(&src_d, 0),
                bf(&got_d, 0),
                bf(&si, lr * hw),
                bf(&gi, lr * hw),
                d * hc + sbr,
                bf(&su, (sbr * hs + d) * lr + j),
                bf(&gu, (d * hc + sbr) * kpad + j),
            );
        }
    }
    Ok((Some(dp), Some(upp)))
}

/// Load one decoder layer (everything except the PLE table - see `load_ple`).
pub fn load_layer(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    li: usize,
) -> Result<Qwen4ExpLayer, GpuModelError> {
    load_layer_named(
        exec,
        st,
        c,
        &format!("model.language_model.layers.{li}"),
        c.blocks[li],
    )
}

/// One routed-expert plane of a 128x128 BLOCK-SCALED FP8 checkpoint, as the
/// Q8_0 block stream (`[f16 scale | 32 int8]`) that `repack_q8_blocks` takes.
///
/// LAYOUT. The checkpoint stores each expert separately as `[out, in]` e4m3
/// with a BF16 `weight_scale_inv` of `[out/128, in/128]`, and dequant is
/// `fp8 * scale_inv` (the name is the inverse of the quantize divisor;
/// verified against this checkpoint - multiply gives trained-weight range,
/// divide gives 1e6). GGUF expert planes are `[in, out, n_expert]` walked as a
/// flat stream of `(e * out + o)` rows, and a safetensors expert's row `o` is
/// already `in` contiguous values, so concatenating experts in order is that
/// stream. A 32-element Q8_0 block never straddles a 128-column scale block
/// (128 is a multiple of 32), so every block carries exactly one FP8 scale.
///
/// Why a transcode, and what the SOTA target is. The pack has no MoE GEMM for
/// this format: `f8bs_*` is e4m3 activations over MXFP4 weights and `f8row_*`
/// is FP8 with PER-ROW scales, neither of which is 128x128-block-scaled FP8
/// weights. Serving these experts natively means a new kernel. That is the
/// SOTA answer and it is the follow-up; it is not what this head needs first,
/// because this head is a drafter - the verify walks every drafted row through
/// the full-precision target and judges it, so the head's numerics move the
/// acceptance rate and can never move the emitted text (see the module note on
/// `base`, same argument). The Q8_0 expert seat this feeds is the one the GGUF
/// MTP head already rides (`ExpertSeats::Q8`), measured at 74% acceptance on
/// this family, so it is a proven seat rather than an untested shortcut.
/// Re-measure acceptance on this lane; if it lands materially under the GGUF
/// head's, the native-format kernel is what buys it back.
fn f8bs_experts_to_q8(
    st: &ShardedSafetensors,
    pfx: &str,
    role: &str,
    n_expert: usize,
    rows: usize,
    in_dim: usize,
) -> Result<Vec<u8>, GpuModelError> {
    const BS: usize = 128;
    const QK: usize = 32;
    if !in_dim.is_multiple_of(QK) {
        return Err(GpuModelError::Unsupported(format!(
            "{pfx} {role}: in_dim {in_dim} is not a multiple of the Q8_0 block"
        )));
    }
    let sc_cols = in_dim.div_ceil(BS);
    let mut out = Vec::with_capacity(n_expert * rows * in_dim / QK * 34);
    for e in 0..n_expert {
        let wn = format!("{pfx}.mlp.experts.{e}.{role}.weight");
        let sn = format!("{wn}_scale_inv");
        let bad = |what: &str| GpuModelError::Unsupported(format!("{wn}: {what}"));
        let (wt, wb) = st.bytes(&wn).ok_or_else(|| bad("missing"))?;
        if wt.dtype != StDtype::F8E4m3 || wt.shape != [rows, in_dim] {
            return Err(bad(&format!(
                "want F8E4m3 [{rows}, {in_dim}], got {:?} {:?}",
                wt.dtype, wt.shape
            )));
        }
        let (stt, sb) = st.bytes(&sn).ok_or_else(|| bad("missing scale_inv"))?;
        if stt.dtype != StDtype::Bf16 {
            return Err(bad(&format!("scale_inv is {:?}, want Bf16", stt.dtype)));
        }
        let sc = crate::gpu_model::st_load::bf16_to_f32(sb);
        if sc.len() != rows.div_ceil(BS) * sc_cols {
            return Err(bad(&format!(
                "scale_inv has {} entries, want {}",
                sc.len(),
                rows.div_ceil(BS) * sc_cols
            )));
        }
        for o in 0..rows {
            let row = &wb[o * in_dim..(o + 1) * in_dim];
            let srow = (o / BS) * sc_cols;
            for b in 0..in_dim / QK {
                let i0 = b * QK;
                // one scale for the whole block - see the layout note
                let s = sc[srow + i0 / BS];
                let mut v = [0f32; QK];
                let mut amax = 0f32;
                for (j, v) in v.iter_mut().enumerate() {
                    *v = paddock_models::modelopt::e4m3_to_f32(row[i0 + j]) * s;
                    amax = amax.max(v.abs());
                }
                // Q8_0: d = amax/127, qs = round(x/d); an all-zero block keeps
                // d = 0 and zero quants, which is what ggml writes too
                let d = amax / 127.0;
                out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for v in v {
                    out.push(((v * inv).round().clamp(-127.0, 127.0) as i8) as u8);
                }
            }
        }
    }
    Ok(out)
}

/// The IN-FILE MTP head, from the safetensors checkpoint itself.
///
/// The GGUF lane sideloads a head (`--mtp`) because unsloth's UD export strips
/// the block; a safetensors checkpoint that declares `text_config.mtp` ships it
/// under `mtp.*` and needs no second file. NVIDIA's official NVFP4 export is
/// the worked case: `mtp` + `mtp_num_hidden_layers` in the config, the weights
/// in `model-fp8-mtp-ple.safetensors`, and before this the lane loaded them
/// not at all - `spec_capable()` answered false and every token decoded at the
/// no-spec rate while the bytes sat in the file (measured 2026-09-19 on a
/// GB10: 18.58 tok/s with only the n-gram fallback drafting).
///
/// The layout is the GGUF's, spelled out instead of packed: `nextn.eh_proj`
/// arrives pre-split as `fc_embedding` / `fc_hidden`, so the load-time column
/// split the GGUF path performs has nothing to do here. The head block is one
/// full-attention decoder layer, which is why it goes through
/// `load_layer_named` rather than a copy.
pub fn load_mtp_st(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
) -> Result<super::load_gguf::MtpWeights, GpuModelError> {
    let h = c.hidden;
    Ok(super::load_gguf::MtpWeights {
        // always full attention - `mtp` is a block, not an entry in
        // `layer_types`, so the kind cannot be read from the config's list
        layer: load_layer_named(exec, st, c, "mtp.layers.0", Qwen4ExpBlock::Attention)?,
        eh_e: dense(exec, st, "mtp.fc_embedding.weight", h, h)?,
        eh_h: dense(exec, st, "mtp.fc_hidden.weight", h, h)?,
        // `1p`, like the attention q/k norms and unlike the hyper-connection
        // ones. This family's norms are zero-centered: the checkpoint stores
        // `w` and the scale is `(1 + w)`. The pack adds the 1 itself for the
        // hyper-connection / PLE / indexer norms, so those load raw - but the
        // nextn glue's two norms go to a plain `rmsnorm_batch`, which does
        // not, so raw `w` scales the head's state by ~0 and its logits come
        // out near-uniform. That is what "128 drafted, 0 accepted, proposals
        // scattered over the whole 248k vocab" was (2026-09-19). The GGUF head
        // reads `f32_dt` because llama.cpp's conversion bakes the +1 in.
        enorm: f32_dt_1p(exec, st, "mtp.pre_fc_norm_embedding.weight", vec![h])?,
        // One statistic over the whole 4-stream row (hc_width), ungrouped -
        // the same reading the GGUF head takes
        hnorm: f32_dt_1p(
            exec,
            st,
            "mtp.pre_fc_norm_hidden.weight",
            vec![c.hc_width()],
        )?,
        // the final mixer's shape, no inject
        head_mix: hc_weights(exec, st, c, "mtp.hyper_connection_mixer", false)?,
    })
}

/// `load_layer` with the tensor prefix and the mixer kind given explicitly.
///
/// The decoder's own layers are `model.language_model.layers.{i}` and take
/// their kind from `layer_types`; the MTP head is the same block shape under
/// `mtp.layers.0` and is always full attention (`text_config.mtp`), so it
/// cannot read either from `li`. Splitting the prefix out is what lets the
/// safetensors MTP reuse this loader instead of growing a second copy of it -
/// the head is a decoder block, and a divergent copy would drift the moment
/// one of them gained a seat.
pub fn load_layer_named(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    p: &str,
    kind: Qwen4ExpBlock,
) -> Result<Qwen4ExpLayer, GpuModelError> {
    let h = c.hidden;
    let mixer = match kind {
        Qwen4ExpBlock::Gdn => {
            let g = format!("{p}.linear_attn");
            // a then b, which is delta_gate_ab's fused layout; hoisted so
            // the bf16 twin (TGV lane) is built from the same host bytes.
            let mut ab_v = f32_tensor(st, &format!("{g}.in_proj_a.weight"), c.gdn_v_heads * h)?;
            ab_v.extend(f32_tensor(
                st,
                &format!("{g}.in_proj_b.weight"),
                c.gdn_v_heads * h,
            )?);
            let ab16 = None;
            MixerW::Gdn(GdnW {
                qkv: dense(
                    exec,
                    st,
                    &format!("{g}.in_proj_qkv.weight"),
                    c.gdn_qkv_rows(),
                    h,
                )?,
                z: dense(
                    exec,
                    st,
                    &format!("{g}.in_proj_z.weight"),
                    c.gdn_z_rows(),
                    h,
                )?,
                // byte concat - refused on MXFP8, see `is_mxfp8`
                zqkv: if super::fuse_gdn_zq_on() && !is_mxfp8(st, &format!("{g}.in_proj_z.weight"))
                {
                    Some(bf16_concat_plane(
                        exec,
                        st,
                        &[
                            (&format!("{g}.in_proj_z.weight"), c.gdn_z_rows()),
                            (&format!("{g}.in_proj_qkv.weight"), c.gdn_qkv_rows()),
                        ],
                        h,
                    )?)
                } else {
                    None
                },
                ab: dt(exec, ab_v, vec![2 * c.gdn_v_heads, h])?,
                ab16,
                conv: conv_plane(
                    exec,
                    st,
                    &format!("{g}.conv1d.weight"),
                    c.gdn_qkv_rows(),
                    c.gdn_conv,
                )?,
                a_log: f32_dt(exec, st, &format!("{g}.A_log"), vec![c.gdn_v_heads])?,
                // pd_delta_gate computes g = ssm_a * softplus(a + dt_bias), so
                // it wants the -exp(A_log) fold the reference does inline. Kept
                // BESIDE the raw A_log (which stays byte-exact residency) -
                // 48 floats, and a derived plane is cheaper than a kernel arm.
                ssm_a: dt(
                    exec,
                    f32_tensor(st, &format!("{g}.A_log"), c.gdn_v_heads)?
                        .into_iter()
                        .map(|a| -a.exp())
                        .collect(),
                    vec![c.gdn_v_heads],
                )?,
                dt_bias: f32_dt(exec, st, &format!("{g}.dt_bias"), vec![c.gdn_v_heads])?,
                norm: f32_dt(exec, st, &format!("{g}.norm.weight"), vec![c.gdn_v_dim])?,
                out: dense(exec, st, &format!("{g}.out_proj.weight"), h, c.gdn_z_rows())?,
                tiled_heads: false,
            })
        }
        Qwen4ExpBlock::Attention => {
            let a = format!("{p}.self_attn");
            let kv = c.n_kv_heads * c.head_dim;
            MixerW::Attn(AttnW {
                q: dense(exec, st, &format!("{a}.q_proj.weight"), c.attn_q_rows(), h)?,
                k: dense(exec, st, &format!("{a}.k_proj.weight"), kv, h)?,
                v: dense(exec, st, &format!("{a}.v_proj.weight"), kv, h)?,
                // byte concat - refused on MXFP8, see `is_mxfp8`
                qkv_f: if super::fuse_attn_qkv_on() && !is_mxfp8(st, &format!("{a}.q_proj.weight"))
                {
                    Some(bf16_concat_plane(
                        exec,
                        st,
                        &[
                            (&format!("{a}.q_proj.weight"), c.attn_q_rows()),
                            (&format!("{a}.k_proj.weight"), kv),
                            (&format!("{a}.v_proj.weight"), kv),
                        ],
                        h,
                    )?)
                } else {
                    None
                },
                o: dense(exec, st, &format!("{a}.o_proj.weight"), h, c.attn_o_in())?,
                q_norm: f32_dt_1p(exec, st, &format!("{a}.q_norm.weight"), vec![c.head_dim])?,
                k_norm: f32_dt_1p(exec, st, &format!("{a}.k_norm.weight"), vec![c.head_dim])?,
                idx_qk: dense(
                    exec,
                    st,
                    &format!("{a}.indexer.index_qk_proj.weight"),
                    (c.idx_heads + c.idx_kv_heads) * c.idx_head_dim,
                    h,
                )?,
                idx_q_norm: f32_dt(
                    exec,
                    st,
                    &format!("{a}.indexer.q_layernorm.weight"),
                    vec![c.idx_head_dim],
                )?,
                idx_k_norm: f32_dt(
                    exec,
                    st,
                    &format!("{a}.indexer.k_layernorm.weight"),
                    vec![c.idx_head_dim],
                )?,
            })
        }
    };
    // router + shared gate row hoisted so the bf16 twin (TGV lane) is
    // built from the same host bytes.
    let mut router_v = f32_tensor(st, &format!("{p}.mlp.gate.weight"), c.n_expert * h)?;
    router_v.extend(f32_tensor(
        st,
        &format!("{p}.mlp.shared_expert_gate.weight"),
        h,
    )?);
    // padded to 64 rows so the low-M dense GEMM (slot 566) can take it; the
    // batch-1 gemv reads the real 513 rows by stride.
    let router16 = None;
    // The seat follows what the checkpoint actually ships, not what the caller
    // expects. NVIDIA's NVFP4 export quantizes the DECODER's routed experts to
    // NVFP4 but leaves the MTP head's at block-scaled FP8 (they live in
    // `model-fp8-mtp-ple.safetensors`), so one layer loader has to seat both -
    // and probing beats a parameter, because a future checkpoint that moves
    // either way then needs no caller change.
    let e0_dtype = st
        .bytes(&format!("{p}.mlp.experts.0.gate_proj.weight"))
        .map(|(t, _)| t.dtype);
    let seats = if e0_dtype == Some(StDtype::F8E4m3) {
        // 128x128 block-scaled FP8 -> the Q8_0 expert seat (see
        // `f8bs_experts_to_q8` for the transcode and why it is one)
        let q8 = |role: &str, rows: usize, in_dim: usize| {
            let raw = f8bs_experts_to_q8(st, p, role, c.n_expert, rows, in_dim)?;
            exec.repack_q8_blocks(&raw, vec![in_dim, rows, c.n_expert])
                .map_err(GpuModelError::from)
        };
        super::ExpertSeats::Q8 {
            gate: q8("gate_proj", c.moe_ff, h)?,
            up: q8("up_proj", c.moe_ff, h)?,
            down: q8("down_proj", h, c.moe_ff)?,
        }
    } else {
        let (gate, up, down) = (
            moe_plane(exec, st, c, p, "gate_proj", c.moe_ff, h)?,
            moe_plane(exec, st, c, p, "up_proj", c.moe_ff, h)?,
            moe_plane(exec, st, c, p, "down_proj", h, c.moe_ff)?,
        );
        super::ExpertSeats::Nvf4 { gate, up, down }
    };
    let moe = MoeW {
        // the shared expert's scalar gate rides as row n_expert
        router: dt(exec, router_v, vec![c.n_expert + 1, h])?,
        router16,
        seats,
        sh_gate: dense(
            exec,
            st,
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            c.shared_ff,
            h,
        )?,
        // the gate|up row concat is a byte concat of two bf16 planes; an
        // MXFP8 pair has a scale plane beside each payload, so there is
        // nothing to concatenate and the two-call path is correct
        sh_gu: if super::fuse_sh_on()
            && !is_mxfp8(st, &format!("{p}.mlp.shared_expert.gate_proj.weight"))
        {
            Some(bf16_concat_plane(
                exec,
                st,
                &[
                    (
                        &format!("{p}.mlp.shared_expert.gate_proj.weight"),
                        c.shared_ff,
                    ),
                    (
                        &format!("{p}.mlp.shared_expert.up_proj.weight"),
                        c.shared_ff,
                    ),
                ],
                h,
            )?)
        } else {
            None
        },
        sh_up: dense(
            exec,
            st,
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            c.shared_ff,
            h,
        )?,
        sh_down: dense(
            exec,
            st,
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            h,
            c.shared_ff,
        )?,
    };
    Ok(Qwen4ExpLayer {
        attn_hc: hc_weights(exec, st, c, &format!("{p}.attn_hyper_connection"), true)?,
        mlp_hc: hc_weights(exec, st, c, &format!("{p}.mlp_hyper_connection"), true)?,
        mixer,
        moe,
        ple: None, // filled by load_ple on PLE layers (51 GB table)
    })
}

/// PLE projections + hash buffers (no table - see `load_ple_table`).
pub fn load_ple_projections(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    li: usize,
) -> Result<PleW, GpuModelError> {
    let p = format!("model.language_model.layers.{li}.ple");
    let hw = c.hc_width();
    let emb = format!("{p}.ple_embedding");
    let i64s = |name: &str, want: usize| -> Result<Vec<i64>, GpuModelError> {
        let (t, b) = st
            .bytes(name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        if t.dtype != StDtype::I64 || b.len() != want * 8 {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: want I64 x{want}"
            )));
        }
        Ok(b.as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect())
    };
    // Table scale. Two shapes in the wild, and which one is present also says
    // what the table's payload is:
    //   * `ngram_embedding.weight_scale`, one scalar - the FP8 table (e4m3
    //     bytes, NVIDIA's export). bf16 in this repo, F32 in the FP8 repo.
    //   * `ngram_embedding.weight_scale_2`, one f32 - the NVFP4 table
    //     (e2m1 nibbles with per-16 e4m3 group scales beside each shard;
    //     Mia-AiLab's export). This is the global factor of a two-level
    //     scale, so it is not interchangeable with the scalar above - the
    //     gather multiplies it by the row's own group scale.
    // The gather branches on the shard dtype, so it needs no flag from here.
    let table_scale = {
        let scalar = format!("{emb}.ngram_embedding.weight_scale");
        let global = format!("{emb}.ngram_embedding.weight_scale_2");
        let (t, b) = st
            .bytes(&scalar)
            .or_else(|| st.bytes(&global))
            .ok_or_else(|| {
                GpuModelError::Unsupported(format!("{scalar} / {global}: neither present"))
            })?;
        match t.dtype {
            StDtype::Bf16 => bf16_to_f32(b)[0],
            StDtype::F32 => f32::from_le_bytes(b[..4].try_into().expect("f32 scalar")),
            other => {
                return Err(GpuModelError::Unsupported(format!(
                    "{scalar} / {global}: dtype {other:?}, want bf16/f32 scalar"
                )));
            }
        }
    };
    Ok(PleW {
        key: dense(exec, st, &format!("{p}.key_proj.weight"), hw, c.hidden)?,
        value: dense(
            exec,
            st,
            &format!("{p}.value_proj.weight"),
            c.hidden,
            c.hidden,
        )?,
        conv: conv_plane(exec, st, &format!("{p}.conv1d.weight"), hw, c.ple_conv)?,
        norm_key: f32_dt(exec, st, &format!("{p}.norm_key.weight"), vec![hw])?,
        norm_query: f32_dt(exec, st, &format!("{p}.norm_query.weight"), vec![hw])?,
        norm_conv: f32_dt(exec, st, &format!("{p}.norm_conv.weight"), vec![hw])?,
        table: None,
        table_rows: 0,
        table_scale,
        multipliers: i64s(&format!("{emb}.layer_multipliers"), c.ngram_size)?,
        head_vocab: i64s(&format!("{emb}.ngram_heads_vocab_sizes"), c.ple_heads())?,
        head_offset: i64s(&format!("{emb}.ngram_heads_offsets"), c.ple_heads())?,
    })
}

/// Upload the 51.2 GB n-gram table (128 fp8 shards, concatenated in order).
///
/// Streamed shard by shard straight out of the mmap into one device
/// allocation: the obvious `Vec<u8>` concat would want 51.2 GB of host RAM
/// on top of the 51.2 GB on device, and it buys nothing - each shard is
/// already contiguous and 128-row aligned by construction.
///
/// This is what makes the PLE gather a device op. vLLM has always held the
/// table this way (`NgramEmbedding.oe_embedder` is a
/// `VocabParallelEmbedding`, i.e. a device Parameter, gathered with an
/// index_select in `embed_batched`); our host-mmap gather was the anomaly,
/// and it cost 0.9-48 s prefill ticks on the serve ladder.
pub fn load_ple_table(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    li: usize,
    ple: &mut PleW,
) -> Result<(), GpuModelError> {
    let emb = format!("model.language_model.layers.{li}.ple.ple_embedding");
    let width = c.ple_embed / c.ple_heads();
    // one pass for the geometry (and to refuse a re-sharded checkpoint before
    // committing 51 GB of device memory), one to copy
    let mut rows = 0usize;
    let mut bytes = 0usize;
    for sh in 0..c.ngram_split {
        let name = format!("{emb}.ngram_embedding.shard_{sh}.weight");
        let (t, b) = st
            .bytes(&name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        if t.dtype != StDtype::F8E4m3 || t.shape.len() != 2 || t.shape[1] != width {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: want F8 [*, {width}], got {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        if b.len() != t.shape[0] * width {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: {} bytes for [{}, {width}]",
                b.len(),
                t.shape[0]
            )));
        }
        rows += t.shape[0];
        bytes += b.len();
    }
    let mut dev = exec.alloc_u8(bytes).map_err(GpuModelError::from)?;
    let mut off = 0usize;
    for sh in 0..c.ngram_split {
        let name = format!("{emb}.ngram_embedding.shard_{sh}.weight");
        let (_, b) = st
            .bytes(&name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        exec.upload_u8_at(b, &mut dev, off)
            .map_err(GpuModelError::from)?;
        off += b.len();
    }
    exec.synchronize().map_err(GpuModelError::from)?;
    ple.table = Some(dev);
    ple.table_rows = rows;
    Ok(())
}
