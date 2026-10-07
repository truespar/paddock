//! Reading plain (unquantized) tensors out of a safetensors checkpoint.
//!
//! The safetensors-primary lanes all need the same three things - widen a bf16
//! tensor to f32, hand back raw bf16 bytes for a device-resident bf16 plane,
//! and say so loudly when a tensor is missing or the wrong dtype - so they live
//! here instead of once per family.
//!
//! This exists because the codebase was already growing copies: `bf16_to_f32`
//! had three independent definitions when granite needed a fourth. The rule
//! is to grep the API rather than the symbol name; the two
//! stragglers (`nemotron/dflash.rs`, `qwen3_asr/aligner.rs`) predate this and
//! should fold in when either is next touched.

use paddock_models::ggml_type::GgmlType;
use paddock_models::modelopt::nvfp4_view;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use crate::gpu::{GpuExecutor, Nvf4MoePlane, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Widen bf16 to f32 exactly: bf16 is the top 16 bits of an f32, so this is a
/// shift, never a rounding.
pub(crate) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

/// Raw bf16 bytes for a device-resident bf16 plane. Values are identical to
/// the f32 widening - the widen just moves into the consuming kernel's
/// registers - so this is a residency choice, not a numeric one.
pub(crate) fn bf16_bytes<'a>(
    st: &'a ShardedSafetensors,
    name: &str,
    want_elems: usize,
) -> Result<&'a [u8], GpuModelError> {
    let (t, bytes) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: tensor missing")))?;
    if t.dtype != StDtype::Bf16 {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: expected bf16, got {:?}",
            t.dtype
        )));
    }
    if bytes.len() != want_elems * 2 {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: {} bytes, expected {want_elems} bf16 elements",
            bytes.len()
        )));
    }
    Ok(bytes)
}

/// bf16 -> f16, BIT-LEVEL and exact, straight off the checkpoint bytes.
///
/// bf16 is `s | 8e | 7m` biased 127; f16 is `s | 5e | 10m` biased 15. Every
/// bf16 mantissa bit fits (7 <= 10) and the exponent is a rebias, so for any
/// value inside f16's normal range the conversion is exact - which is what
/// makes the f16 tensor-core lane the same numbers rather than a precision
/// trade. Out-of-range is checked, not clamped: overflow refuses the plane,
/// and the subnormal tail falls back to the rounding convert (values below
/// 2^-14 contribute less than the f32 accumulator's own rounding, the same
/// reasoning `narrow_to_f16` records).
///
/// The obvious `bf16 -> f32 -> f16` spelling of this costs four minutes of
/// load on qwen4exp's 3.2G dense elements, which is why it is written out.
/// Lives here because the second family to want it (dinov3) arrived.
pub(crate) fn bf16_to_f16_exact(raw: &[u8], what: &str) -> Result<Vec<half::f16>, GpuModelError> {
    let mut over = 0usize;
    let mut out = Vec::with_capacity(raw.len() / 2);
    for b in raw.as_chunks::<2>().0 {
        let v = u16::from_le_bytes(*b);
        let sign = v & 0x8000;
        let e = ((v >> 7) & 0xff) as i32;
        let m = v & 0x7f;
        if (113..=142).contains(&e) {
            // normal in f16: exponent rebias 127 -> 15, mantissa left-aligned
            out.push(half::f16::from_bits(
                sign | (((e - 112) as u16) << 10) | (m << 3),
            ));
        } else if e == 0 {
            out.push(half::f16::from_bits(sign)); // +/-0 (bf16 subnormals flush)
        } else {
            // over- or underflow: let the rounding convert decide, and count
            // the overflows so the caller can refuse the plane
            let f = f32::from_bits((v as u32) << 16);
            let h = half::f16::from_f32(f);
            if f.is_finite() && !h.is_finite() {
                over += 1;
            }
            out.push(h);
        }
    }
    if over > 0 {
        return Err(GpuModelError::Unsupported(format!(
            "{what}: {over} of {} weights overflow f16 (|w| > 65504) - this plane cannot \
             carry the f16 tensor-core lane",
            raw.len() / 2
        )));
    }
    Ok(out)
}

/// Read a tensor as f32, widening bf16 exactly; f32 passes through.
/// OCP E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. 0x7F/0xFF are NaN;
/// exponent 0 is subnormal with no implicit leading 1.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0x0F && man == 7.0 {
        return f32::NAN;
    }
    if exp == 0 {
        // subnormal: 2^(1-bias) * man/8
        sign * (man / 8.0) * (2.0f32).powi(1 - 7)
    } else {
        sign * (1.0 + man / 8.0) * (2.0f32).powi(exp - 7)
    }
}

/// OCP E8M0 shared scale: a biased exponent byte, 0xFF is NaN.
fn ue8m0_to_f32(b: u8) -> f32 {
    if b == 0xFF {
        f32::NAN
    } else {
        (2.0f32).powi(b as i32 - 127)
    }
}

pub(crate) fn f32_tensor(
    st: &ShardedSafetensors,
    name: &str,
    want_elems: usize,
) -> Result<Vec<f32>, GpuModelError> {
    let (t, bytes) = st
        .bytes(name)
        .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: tensor missing")))?;
    let v = match t.dtype {
        StDtype::Bf16 => bf16_to_f32(bytes),
        StDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        // MXFP8 dequantized on the host. These are the planes the engine
        // consumes as f32 (the GDN in_proj_a/b pair the delta_gate_ab layout
        // fuses, and the norms), not as a weight class - there is no f8 lane
        // to hand them to, so the only alternatives are dequantize here or
        // refuse a checkpoint we otherwise serve. They are tiny: in_proj_a is
        // [48, 2560], 123k values, against the 99 GiB the file carries. The
        // big planes never reach this function - `dense()` passes their bytes
        // straight through (see `mxfp8_plane`), so nothing here walks back a
        // quantization the kernel could have read directly.
        StDtype::F8E4m3 => {
            let (ts, sb) = st.bytes(&format!("{name}_scale")).ok_or_else(|| {
                GpuModelError::Unsupported(format!("{name}: f8e4m3 with no {name}_scale"))
            })?;
            if ts.dtype != StDtype::U8 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}_scale: {:?}, want u8 (ue8m0)",
                    ts.dtype
                )));
            }
            if bytes.len() != sb.len() * 32 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: {} payload bytes against {} scales - want one ue8m0 per 32 \
                     (MXFP8 block [1, 32])",
                    bytes.len(),
                    sb.len()
                )));
            }
            bytes
                .iter()
                .enumerate()
                .map(|(i, &b)| e4m3_to_f32(b) * ue8m0_to_f32(sb[i / 32]))
                .collect()
        }
        other => {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: expected bf16/f32/f8e4m3, got {other:?}"
            )));
        }
    };
    if v.len() != want_elems {
        return Err(GpuModelError::Unsupported(format!(
            "{name}: {} elements, expected {want_elems}",
            v.len()
        )));
    }
    Ok(v)
}

/// bf16 plane resident as shipped: checkpoint bytes on device, dims [k, n]
/// (in_dim-major, the QuantTensor convention the bf16 GEMV/GEMM lanes read).
pub(crate) fn bf16_plane(
    exec: &GpuExecutor,
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

/// Load-time row concat of bf16 planes into one residency - the rival's
/// MergedColumnParallelLinear layout (qwen3_5.py). Rows stack in `parts`
/// order; dims stay the QuantTensor [k, n] convention. Byte-exact: the fused
/// plane is the checkpoint bytes of each part, back to back.
pub(crate) fn bf16_concat_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    parts: &[(&str, usize)],
    k: usize,
) -> Result<QuantTensor, GpuModelError> {
    let n: usize = parts.iter().map(|p| p.1).sum();
    let mut raw: Vec<u8> = Vec::with_capacity(n * k * 2);
    for (name, rows) in parts {
        raw.extend_from_slice(bf16_bytes(st, name, rows * k)?);
    }
    Ok(QuantTensor {
        bytes: exec.to_device_u8(&raw).map_err(GpuModelError::from)?,
        ty: GgmlType::Bf16,
        dims: vec![k, n],
    })
}

/// A layer's routed experts of one role (gate / up / down), NVFP4 off the
/// checkpoint's own nibbles: every expert's packed + scale bytes stacked in
/// expert order (expert `e`'s row `r` at `e * rows + r`) with its global
/// scale, then one `nvf4_moe_upload`. `name(e)` is the expert's tensor
/// prefix (`...experts.{e}.gate_proj`); either export dialect reads.
pub(crate) fn nvf4_expert_stack(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: impl Fn(usize) -> String,
    n_expert: usize,
    rows: usize,
    in_dim: usize,
) -> Result<Nvf4MoePlane, GpuModelError> {
    let mut cat_p: Vec<u8> = Vec::with_capacity(n_expert * rows * in_dim / 2);
    let mut cat_s: Vec<u8> = Vec::with_capacity(n_expert * rows * in_dim / 16);
    let mut s2 = Vec::with_capacity(n_expert);
    for e in 0..n_expert {
        let pfx = name(e);
        let v = nvfp4_view(st, &pfx)
            .map_err(|err| GpuModelError::Unsupported(format!("{pfx}: {err}")))?;
        if (v.n, v.k) != (rows, in_dim) {
            return Err(GpuModelError::Unsupported(format!(
                "{pfx} is [{}, {}], expected [{rows}, {in_dim}]",
                v.n, v.k
            )));
        }
        cat_p.extend_from_slice(v.packed);
        cat_s.extend_from_slice(v.scales);
        s2.push(v.scale2);
    }
    exec.nvf4_moe_upload(&cat_p, &cat_s, &s2, n_expert, rows, in_dim)
        .map_err(GpuModelError::from)
}
