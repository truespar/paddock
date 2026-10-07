//! Granite from a safetensors CHECKPOINT DIRECTORY - the NVFP4 lane.
//!
//! `load.rs` reads a GGUF; this reads what IBM actually ships for Blackwell:
//! `ibm-granite/granite-4.2-{8b,30b}-nvfp4`, a compressed-tensors export with
//! no GGUF anywhere in the repo. Same decoder, same `Hparams`, same forward -
//! only the weight CLASS differs, which is what [`GraniteW`] exists for.
//!
//! What the checkpoint quantizes, verified against the shipped 30b index
//! (1923 tensors):
//!
//! - the seven per-layer projections (`q,k,v,o,gate,up,down`) carry the
//!   llm-compressor fp4 triple - `weight_packed` [n, k/2] e2m1 nibbles,
//!   `weight_scale` [n, k/16] e4m3, and `weight_global_scale`. Served W4A16
//!   **byte-for-byte as shipped**: no requantization anywhere in this file.
//! - `model.embed_tokens.weight`, `lm_head.weight` and every norm stay bf16.
//!   That is IBM's recipe, not an oversight, and we keep it - quantizing the
//!   head to reach an existing code path would be a quality change nobody
//!   asked for.
//!
//! `input_global_scale` rides along in the checkpoint for a W4A4 activation
//! lane. We deliberately ignore it and serve W4A16, which keeps the weights
//! bit-exact to the file - the same election the qwen3.8 and nemotron NVFP4
//! lanes made.

use std::path::Path;
use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::ggml_type::GgmlType;
use paddock_models::granite::GraniteConfig;
use paddock_models::modelopt::{e4m3_to_f32, fp8_channel_view, nvfp4_view};
use paddock_models::safetensors::ShardedSafetensors;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_bytes, f32_tensor};

use super::*;

/// f32 -> e4m3 (round-to-nearest-even, saturate at 448, sign preserved;
/// +/-0 canonicalize to 0x00). Inverse of `e4m3_to_f32` on every canonical
/// byte - asserted once per process in `f8lin_requant`.
fn f32_to_e4m3(v: f32) -> u8 {
    if v == 0.0 || v.is_nan() {
        return 0;
    }
    let sign = if v < 0.0 { 0x80u8 } else { 0 };
    let a = v.abs();
    // subnormal band: below 2^-6, steps of 2^-9
    if a < 0.015_625 {
        let t = a * 512.0; // a / 2^-9
        let mut m = t.floor();
        let frac = t - m;
        if frac > 0.5 || (frac == 0.5 && (m as u32) & 1 == 1) {
            m += 1.0;
        }
        let m = m as u32; // 0..=8; 8 rolls into exp 1 mant 0 = 0x08
        return sign | (m as u8);
    }
    if a >= 464.0 {
        return sign | 0x7E; // saturate at 448 (exp 15, mant 6)
    }
    let mut e = a.log2().floor() as i32;
    // guard fp fuzz at binade edges
    if a < 2f32.powi(e) {
        e -= 1;
    } else if a >= 2f32.powi(e + 1) {
        e += 1;
    }
    let t = (a / 2f32.powi(e) - 1.0) * 8.0; // mantissa steps
    let mut m = t.floor();
    let frac = t - m;
    if frac > 0.5 || (frac == 0.5 && (m as u32) & 1 == 1) {
        m += 1.0;
    }
    let mut mi = m as i32;
    if mi == 8 {
        mi = 0;
        e += 1;
    }
    if e > 8 || (e == 8 && mi > 6) {
        return sign | 0x7E;
    }
    sign | (((e + 7) as u8) << 3) | (mi as u8)
}

/// Fold arbitrary per-row f32 scales into (e4m3 bytes, biased pow2 exponent):
/// e = ceil(log2(scale)) so the residual m = scale/2^(e-127) is in (0.5, 1] -
/// values only shrink (no saturation), RN-even costs <= half an e4m3 ULP, and
/// a pow2 scale is a verbatim row copy. 256-entry LUT per non-pow2 row.
fn f8lin_requant(bytes: &[u8], scales: &[f32], k: usize) -> (Vec<u8>, Vec<u8>) {
    static CHECK: std::sync::Once = std::sync::Once::new();
    CHECK.call_once(|| {
        for b in 0..=0xFEu8 {
            if b & 0x7f == 0x7f || b == 0x80 {
                continue; // NaN codes and -0 canonicalize
            }
            assert_eq!(
                f32_to_e4m3(e4m3_to_f32(b)),
                b,
                "e4m3 roundtrip broke at byte {b:#x} - the lin requant would corrupt weights"
            );
        }
    });
    let n = scales.len();
    let mut out = vec![0u8; bytes.len()];
    let mut wse = vec![127u8; n];
    let nth = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8)
        .min(32);
    let rows_per = n.div_ceil(nth);
    std::thread::scope(|scope| {
        let mut rest: &mut [u8] = &mut out;
        let mut erest: &mut [u8] = &mut wse;
        let mut r0 = 0usize;
        while r0 < n {
            let rows = rows_per.min(n - r0);
            let (chunk, tail) = rest.split_at_mut(rows * k);
            let (echunk, etail) = erest.split_at_mut(rows);
            rest = tail;
            erest = etail;
            let src = &bytes[r0 * k..(r0 + rows) * k];
            let scs = &scales[r0..r0 + rows];
            scope.spawn(move || {
                for r in 0..rows {
                    let s = scs[r];
                    if !(s > 0.0 && s.is_finite()) {
                        // dead/degenerate row: zero bytes, neutral exponent
                        chunk[r * k..(r + 1) * k].fill(0);
                        echunk[r] = 127;
                        continue;
                    }
                    let bits = s.to_bits();
                    if bits & 0x7f_ffff == 0 {
                        // exact power of two: verbatim bytes
                        chunk[r * k..(r + 1) * k].copy_from_slice(&src[r * k..(r + 1) * k]);
                        echunk[r] = (bits >> 23).min(254) as u8;
                        continue;
                    }
                    let e = ((bits >> 23) + 1).min(254);
                    let m = s / f32::from_bits((e as u32) << 23);
                    let mut lut = [0u8; 256];
                    for (b, l) in lut.iter_mut().enumerate() {
                        let v = e4m3_to_f32(b as u8);
                        *l = if v.is_nan() { 0 } else { f32_to_e4m3(v * m) };
                    }
                    for (d, &b) in chunk[r * k..(r + 1) * k]
                        .iter_mut()
                        .zip(&src[r * k..(r + 1) * k])
                    {
                        *d = lut[b as usize];
                    }
                    echunk[r] = e as u8;
                }
            });
            r0 += rows;
        }
    });
    (out, wse)
}

/// Row order that turns HF's q/k layout into the one our rope kernels read.
///
/// **This is the difference between working and fluent garbage, and nothing
/// about it fails loudly.** granite runs llama.cpp's `LLAMA_ROPE_TYPE_NORM` -
/// rotating INTERLEAVED pairs `(2k, 2k+1)` - while HF stores q/k for the
/// half-split NEOX convention. llama.cpp's converter reconciles them by
/// PERMUTING the q/k output rows on the way into the GGUF (`LlamaModel.permute`,
/// which granite's converter inherits), so the GGUF our kernels were built and
/// parity-checked against is already in interleaved order. A safetensors
/// checkpoint is not, so this applies the same permutation at load.
///
/// Measured rather than assumed: per-row weight energy of
/// `blk.0.attn_q` in the Q8_0 GGUF against `q_proj` in the NVFP4 checkpoint
/// correlates **0.9999 under this permutation and 0.544 without it** (k_proj:
/// 0.9998 vs 0.430). Row energy survives requantization, which is what makes
/// the two files comparable at all.
///
/// `heads` is the head count the RESHAPE uses: `n_heads` for q, but
/// `n_kv_heads` for k - upstream swaps it in for the GQA case, and using
/// `n_heads` for k would permute by the wrong block size.
fn rope_row_perm(heads: usize, rows: usize) -> Vec<usize> {
    // numpy: arange(rows).reshape(heads, 2, half).swapaxes(1, 2).reshape(rows)
    let half = rows / heads / 2;
    let mut p = vec![0usize; rows];
    for h in 0..heads {
        for i in 0..half {
            for j in 0..2 {
                p[h * 2 * half + i * 2 + j] = h * 2 * half + j * half + i;
            }
        }
    }
    p
}

/// Reorder whole rows of a row-major `[rows, stride]` byte plane.
///
/// Safe on the packed fp4 nibbles and on the e4m3 block scales alike: both are
/// row-major with every row contiguous, so a row permutation never touches the
/// bytes within a row - no re-encode, no requantization, values bit-identical.
fn permute_rows(src: &[u8], perm: &[usize], stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    for (dst_row, &src_row) in perm.iter().enumerate() {
        out[dst_row * stride..(dst_row + 1) * stride]
            .copy_from_slice(&src[src_row * stride..(src_row + 1) * stride]);
    }
    out
}

impl GpuGranite {
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        let cfg = GraniteConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("granite config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("granite shards: {e}")))?;

        // The fp4 kernels convert e2m1 through e4m3, so they need sm_89+. Say
        // that out loud rather than serving something else: unlike the qwen3.5
        // overlay there is no Q8 base to fall back to here - this checkpoint is
        // the only copy of the weights.
        if !exec.has_nvf4_ckpt() {
            let (maj, min) = exec.compute_capability();
            return Err(GpuModelError::Unsupported(format!(
                "granite NVFP4: this GPU is sm_{maj}{min} and the W4A16 nvf4 kernels need \
                 sm_89 or newer. There is no GGUF in this checkpoint to fall back to - \
                 serve the Q8_0 GGUF build of the same model instead."
            )));
        }
        // Gate on the checkpoint's own byte total. Summed from the tensor
        // directory rather than from file sizes so the shard headers and any
        // padding stay out of it -- this is the number that has to fit.
        let ckpt_bytes: u64 = st
            .names()
            .filter_map(|n| st.bytes(n).map(|(_, b)| b.len() as u64))
            .sum();
        exec.vram_load_gate(ckpt_bytes, "granite")
            .map_err(GpuModelError::WontFit)?;

        let (n_layer, n_embd, n_heads) = (cfg.n_layer, cfg.hidden, cfg.n_heads);
        let (n_kv_heads, head_dim, n_ff) = (cfg.n_kv_heads, cfg.head_dim, cfg.n_ff);
        let n_vocab = cfg.vocab;
        let gb = |b: u64| b as f64 / (1u64 << 30) as f64;

        // Plain rope, exactly as the GGUF lane builds it: ext_factor 0
        // collapses YarnRope's ramp and freq_scale 1 leaves the base alone.
        let rope = YarnRope::new(
            head_dim,
            cfg.rope_theta,
            1.0,
            cfg.max_pos,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();

        // ---- embeddings: bf16 rows, gathered and widened in-kernel --------
        if !exec.has_embed_gather_bf16() {
            return Err(GpuModelError::Unsupported(
                "granite NVFP4: the kernel pack has no bf16 embedding gather - rebuild the \
                 pack, or serve the Q8_0 GGUF build"
                    .into(),
            ));
        }
        let raw = bf16_bytes(&st, "model.embed_tokens.weight", n_vocab * n_embd)?;
        let tok_embd = TokEmbd::Q8(QuantTensor {
            bytes: exec.to_device_u8(raw).map_err(GpuModelError::from)?,
            ty: GgmlType::Bf16,
            dims: vec![n_embd, n_vocab],
        });
        tracing::info!(
            "granite VRAM  input embeddings (bf16, as shipped)  {:>7.2} GB",
            gb(tok_embd.resident_bytes() as u64)
        );

        let dequant_bf16 = paddock_models::dev_var_os!("PADDOCK_GRANITE_NVF4_BF16").is_some();
        if dequant_bf16 {
            tracing::warn!(
                "granite: PADDOCK_GRANITE_NVF4_BF16 - fp4 planes dequantized to bf16 (bisect \
                 instrument, ~1.7x the VRAM; same values, different kernels)"
            );
        }

        // ---- layers --------------------------------------------------------
        let dt = |v: Vec<f32>, dims: Vec<usize>| -> Result<DeviceTensor, GpuModelError> {
            Ok(DeviceTensor {
                buf: exec.to_device(&v).map_err(GpuModelError::from)?,
                dims,
            })
        };
        // One projection: NVFP4 when the checkpoint quantized it, bf16 when it
        // left it alone. Deciding per TENSOR rather than per model is what lets
        // a future mixed recipe load without a code change -- and `granite
        // 4.2`'s own recipe already leaves the head unquantized, so the mixed
        // case is real, not hypothetical.
        // `rope_heads` is Some only for q and k, which are the only planes whose
        // ROW ORDER the rope convention constrains - v never ropes and o is on
        // the other side of attention.
        let plane = |name: &str,
                     want_out: usize,
                     want_in: usize,
                     rope_heads: Option<usize>|
         -> Result<GraniteW, GpuModelError> {
            let perm = rope_heads.map(|h| rope_row_perm(h, want_out));
            match nvfp4_view(&st, name) {
                Ok(v) => {
                    if (v.n, v.k) != (want_out, want_in) {
                        return Err(GpuModelError::Unsupported(format!(
                            "{name}: nvfp4 plane [{}, {}] != expected [{want_out}, {want_in}]",
                            v.n, v.k
                        )));
                    }
                    // Dev bisect (PADDOCK_GRANITE_NVF4_BF16=1): serve the same
                    // values through the bf16 kernels instead of the fp4 ones,
                    // using modelopt's pinned reference dequant. Identical
                    // numbers, different kernel family - so if output is right
                    // here and wrong without it, the fp4 lane is at fault, and
                    // if it is wrong both ways the bug is structural (rope,
                    // norms, head) and not the quantization at all.
                    if dequant_bf16 {
                        let mut bytes = Vec::with_capacity(v.n * v.k * 2);
                        for dst in 0..v.n {
                            let src = perm.as_ref().map_or(dst, |p| p[dst]);
                            for x in v.dequant_row_f32(src) {
                                bytes
                                    .extend_from_slice(&((x.to_bits() >> 16) as u16).to_le_bytes());
                            }
                        }
                        return Ok(GraniteW::Bf16(QuantTensor {
                            bytes: exec.to_device_u8(&bytes).map_err(GpuModelError::from)?,
                            ty: GgmlType::Bf16,
                            dims: vec![v.k, v.n],
                        }));
                    }
                    let up = match &perm {
                        // one nibble pair per byte, one e4m3 per 16 elements
                        Some(p) => exec.nvf4_upload(
                            &permute_rows(v.packed, p, v.k / 2),
                            &permute_rows(v.scales, p, v.k / 16),
                            v.scale2,
                            v.n,
                            v.k,
                        ),
                        None => exec.nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k),
                    };
                    Ok(GraniteW::Nvf4(up.map_err(GpuModelError::from)?))
                }
                // Not NVFP4 -> try the fp8 export: e4m3 bytes with one scale per
                // output row (`weight_scale` BF16 [n, 1]). Same family, same
                // rule -- bytes byte-exact, scale applied in the epilogue.
                Err(_) if fp8_channel_view(&st, name).is_ok() => {
                    let v = fp8_channel_view(&st, name)
                        .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}")))?;
                    if (v.n, v.k) != (want_out, want_in) {
                        return Err(GpuModelError::Unsupported(format!(
                            "{name}: fp8 plane [{}, {}] != expected [{want_out}, {want_in}]",
                            v.n, v.k
                        )));
                    }
                    // q/k rope order applies to every class, not just fp4:
                    // one e4m3 byte per element, one f32 scale per row.
                    let owned_w;
                    let owned_s;
                    let (wb, sc): (&[u8], &[f32]) = match &perm {
                        Some(pm) => {
                            owned_w = permute_rows(v.weight, pm, v.k);
                            owned_s = pm.iter().map(|&i| v.scales[i]).collect::<Vec<f32>>();
                            (&owned_w, &owned_s)
                        }
                        None => (v.weight, &v.scales),
                    };
                    // tuned lin lane: strip boxes serve b=1 (f8lin_gemv),
                    // decode widths (f8_gemm_lin) and the 33..1024 band
                    // (lin_kt, ~94% of the DRAM roof on this die) from one
                    // plane. PADDOCK_G42_F8ROW=1 = load-time A/B pin.
                    // Alternative layout: ROW-MAJOR w8 (the fp8-native shape
                    // qwen38 uses) - f8d ks at decode widths measured
                    // 1.02-1.05x the Q8 rung; f8_gemm_w8 takes the wide band.
                    // PADDOCK_G42_F8LINBOX=1 = the lin-box A/B (kt3 waves at
                    // 42us/GEMM, but legacy-lin decode costs ~+0.8ms/tick and
                    // its ticket-less b=1 gemv measured 15ms ITL).
                    // DEFAULT: the f8row class - measured best of the three
                    // fp8 arms end to end. f8d runs 750 GB/s against Q8-ks's
                    // 1630 on granite decode shapes, which is what sinks the
                    // w8+f8d arm.
                    // PADDOCK_G42_F8W8=1 re-arms the tuned-family experiment.
                    let f8_ok = exec.has_f8d_gemm_mma_ks()
                        && exec.has_f8_gemm_w8()
                        && v.k % 128 == 0
                        && (v.k / 32) % 2 == 0
                        && paddock_models::dev_var_os!("PADDOCK_G42_F8W8").is_some();
                    if f8_ok {
                        let (data, wse) = f8lin_requant(wb, sc, v.k);
                        let mut sc32 = vec![0u8; v.n * (v.k / 32)];
                        for (r, ch) in sc32.chunks_mut(v.k / 32).enumerate() {
                            ch.fill(wse[r]);
                        }
                        let raw = exec.to_device_u8(&data).map_err(GpuModelError::from)?;
                        let s32 = exec.to_device_u8(&sc32).map_err(GpuModelError::from)?;
                        let w8 = crate::gpu::RepackedMxfp4 {
                            data: raw,
                            scale: s32,
                        };
                        let plane = if paddock_models::dev_var_os!("PADDOCK_G42_F8LINBOX").is_some()
                            && exec.has_f8_lin()
                            && exec.has_f8lin_gemv()
                        {
                            exec.f8w_repack_lin(w8, v.k, v.n)
                                .map_err(GpuModelError::from)?
                        } else {
                            w8
                        };
                        return Ok(GraniteW::F8Lin {
                            plane,
                            out_dim: v.n,
                            in_dim: v.k,
                        });
                    }
                    let plane = exec.fp8_ckpt_to_f8row_rows(wb, sc, v.k, v.n);
                    Ok(GraniteW::Fp8 {
                        plane: plane.map_err(GpuModelError::from)?,
                        out_dim: v.n,
                        in_dim: v.k,
                    })
                }
                Err(_) => {
                    let raw = bf16_bytes(&st, &format!("{name}.weight"), want_out * want_in)?;
                    let owned;
                    let bytes = match &perm {
                        Some(p) => {
                            owned = permute_rows(raw, p, want_in * 2);
                            &owned[..]
                        }
                        None => raw,
                    };
                    Ok(GraniteW::Bf16(QuantTensor {
                        bytes: exec.to_device_u8(bytes).map_err(GpuModelError::from)?,
                        ty: GgmlType::Bf16,
                        dims: vec![want_in, want_out],
                    }))
                }
            }
        };

        let mut layers = Vec::with_capacity(n_layer);
        let (mut attn_bytes, mut ffn_bytes) = (0u64, 0u64);
        let q_out = n_heads * head_dim;
        let kv_out = n_kv_heads * head_dim;
        for li in 0..n_layer {
            let p = format!("model.layers.{li}");
            let attn_norm = dt(
                f32_tensor(&st, &format!("{p}.input_layernorm.weight"), n_embd)?,
                vec![n_embd],
            )?;
            // fused [q|k|v] NVFP4 plane: the gate|up merge's twin on
            // the attention side, and memory-neutral -- the three per-projection
            // planes are not uploaded when it exists. Exact for the same reason
            // (one weight_global_scale shared by q, k and v, checked per layer;
            // a mismatch keeps the split planes). q and k rows take the same
            // NORM-rope permutation `plane` applies to them, v none. Needs the
            // pack's partials-consuming rope kernel (slot 524) -- an older pack
            // keeps the split planes. Kill: PADDOCK_G42_NO_QKV_NV4.
            let fused_qkv = (!dequant_bf16
                && exec.has_qkv_rope_from_parts()
                && paddock_models::dev_var_os!("PADDOCK_G42_NO_QKV_NV4").is_none())
            .then(|| {
                let q = nvfp4_view(&st, &format!("{p}.self_attn.q_proj")).ok()?;
                let k = nvfp4_view(&st, &format!("{p}.self_attn.k_proj")).ok()?;
                let v = nvfp4_view(&st, &format!("{p}.self_attn.v_proj")).ok()?;
                if q.scale2 != k.scale2
                    || k.scale2 != v.scale2
                    || (q.n, q.k) != (q_out, n_embd)
                    || (k.n, k.k) != (kv_out, n_embd)
                    || (v.n, v.k) != (kv_out, n_embd)
                {
                    return None;
                }
                let pq = rope_row_perm(n_heads, q_out);
                let pk = rope_row_perm(n_kv_heads, kv_out);
                let rows = q_out + 2 * kv_out;
                let mut packed = Vec::with_capacity(rows * (n_embd / 2));
                packed.extend_from_slice(&permute_rows(q.packed, &pq, n_embd / 2));
                packed.extend_from_slice(&permute_rows(k.packed, &pk, n_embd / 2));
                packed.extend_from_slice(v.packed);
                let mut scales = Vec::with_capacity(rows * (n_embd / 16));
                scales.extend_from_slice(&permute_rows(q.scales, &pq, n_embd / 16));
                scales.extend_from_slice(&permute_rows(k.scales, &pk, n_embd / 16));
                scales.extend_from_slice(v.scales);
                Some(exec.nvf4_upload(&packed, &scales, q.scale2, rows, n_embd))
            })
            .flatten()
            .transpose()
            .map_err(GpuModelError::from)?;
            let (wq, wk, wv) = match &fused_qkv {
                Some(_) => (
                    GraniteW::Nvf4Fused {
                        out_dim: q_out,
                        in_dim: n_embd,
                    },
                    GraniteW::Nvf4Fused {
                        out_dim: kv_out,
                        in_dim: n_embd,
                    },
                    GraniteW::Nvf4Fused {
                        out_dim: kv_out,
                        in_dim: n_embd,
                    },
                ),
                None => (
                    plane(
                        &format!("{p}.self_attn.q_proj"),
                        q_out,
                        n_embd,
                        Some(n_heads),
                    )?,
                    plane(
                        &format!("{p}.self_attn.k_proj"),
                        kv_out,
                        n_embd,
                        Some(n_kv_heads),
                    )?,
                    plane(&format!("{p}.self_attn.v_proj"), kv_out, n_embd, None)?,
                ),
            };
            let wo = plane(&format!("{p}.self_attn.o_proj"), n_embd, q_out, None)?;
            attn_bytes += attn_norm.buf.len() as u64 * 4
                + wq.bytes()
                + wk.bytes()
                + wv.bytes()
                + wo.bytes()
                + fused_qkv
                    .as_ref()
                    .map_or(0, |f| (f.data.len() + f.scale.len()) as u64);

            let ffn_norm = dt(
                f32_tensor(&st, &format!("{p}.post_attention_layernorm.weight"), n_embd)?,
                vec![n_embd],
            )?;
            // gate|up as one [2*n_ff, n_embd] plane when the checkpoint lets
            // us: nvfp4 carries one weight_global_scale per tensor, so this is
            // only exact if gate's equals up's. Granite's do (40/40 on the 8b,
            // 62/62 on the 30b) but that is a property of this export, not of
            // the format -- a mismatch falls back to split rather than
            // rescaling one side's e4m3 block scales. See GraniteLayer::gate_up
            // for the measurement that motivates it.
            let merged = (!dequant_bf16)
                .then(|| {
                    let g = nvfp4_view(&st, &format!("{p}.mlp.gate_proj")).ok()?;
                    let u = nvfp4_view(&st, &format!("{p}.mlp.up_proj")).ok()?;
                    if g.scale2 != u.scale2
                        || (g.n, g.k) != (n_ff, n_embd)
                        || (u.n, u.k) != (n_ff, n_embd)
                    {
                        return None;
                    }
                    // INTERLEAVED rows: (gate_j, up_j) adjacent, so the
                    // prefill GEMM's swiglu + nvf4-quant epilogue (slot 533) sees
                    // each pair in one warp's row block and the f32 [rows, 2ff]
                    // landing never exists; every consumer of the plane's output
                    // reads pairs (the `_il` twins, slots 534-536). Bit-identical to
                    // the plain chain (bench/nv4_swq_cmp.cu) but OPT-IN
                    // (PADDOCK_NV4_SWQ=1): measured slower -- the silu + quant math
                    // per output inside the 1-CTA/SM GEMM (25% occupancy) costs
                    // more than the f32 landing's round trip it removes (8b/4096
                    // rows 1357 vs 1108 us, 30b 3415 vs 2789; bench/nv4_swq_time.cu).
                    // A persistent f4t that overlaps one tile's epilogue with the
                    // next tile's mainloop is what would make it pay.
                    let pairs = exec.has_nvf4_gemm_f4t_swq()
                        && exec.has_swiglu_fused_nvf4_il()
                        && (2 * n_ff) % 256 == 0
                        && n_embd % 256 == 0
                        && paddock_models::dev_var!("PADDOCK_NV4_SWQ")
                            .map(|v| v == "1")
                            .unwrap_or(false);
                    let wrow = n_embd / 2;
                    let srow = n_embd / 16;
                    let mut packed = Vec::with_capacity(g.packed.len() + u.packed.len());
                    let mut scales = Vec::with_capacity(g.scales.len() + u.scales.len());
                    if pairs {
                        for j in 0..n_ff {
                            packed.extend_from_slice(&g.packed[j * wrow..(j + 1) * wrow]);
                            packed.extend_from_slice(&u.packed[j * wrow..(j + 1) * wrow]);
                            scales.extend_from_slice(&g.scales[j * srow..(j + 1) * srow]);
                            scales.extend_from_slice(&u.scales[j * srow..(j + 1) * srow]);
                        }
                    } else {
                        packed.extend_from_slice(g.packed);
                        packed.extend_from_slice(u.packed);
                        scales.extend_from_slice(g.scales);
                        scales.extend_from_slice(u.scales);
                    }
                    Some(
                        exec.nvf4_upload(&packed, &scales, g.scale2, 2 * n_ff, n_embd)
                            .map(|mut p| {
                                p.gu_pairs = pairs;
                                p
                            }),
                    )
                })
                .flatten()
                .transpose()
                .map_err(GpuModelError::from)?
                .map(GraniteW::Nvf4);
            let (gate, up) = match &merged {
                Some(_) => (None, None),
                None => (
                    Some(plane(&format!("{p}.mlp.gate_proj"), n_ff, n_embd, None)?),
                    Some(plane(&format!("{p}.mlp.up_proj"), n_ff, n_embd, None)?),
                ),
            };
            let down = plane(&format!("{p}.mlp.down_proj"), n_embd, n_ff, None)?;
            ffn_bytes += ffn_norm.buf.len() as u64 * 4
                + down.bytes()
                + merged.as_ref().map_or_else(
                    || {
                        gate.as_ref().map_or(0, |g| g.bytes())
                            + up.as_ref().map_or(0, |u| u.bytes())
                    },
                    |m| m.bytes(),
                );

            // fused wqkv f8row plane (rows q|k|v): exact concat - the
            // per-row scale class fuses losslessly, which int8's per-32
            // exactness law forbids. PADDOCK_G42_NO_QKV_FUSE=1 skips.
            let qkv_f8 = match (&wq, &wk, &wv) {
                (
                    GraniteW::Fp8 { plane: pq, .. },
                    GraniteW::Fp8 { plane: pk, .. },
                    GraniteW::Fp8 { plane: pv, .. },
                ) if paddock_models::dev_var_os!("PADDOCK_G42_NO_QKV_FUSE").is_none() => {
                    let fuse = || -> Result<crate::gpu::F8RowPlane, String> {
                        let k_in = n_embd;
                        let rows = pq.scale.len() + pk.scale.len() + pv.scale.len();
                        let mut data: cudarc::driver::CudaSlice<u8> = exec
                            .stream
                            .alloc_zeros(rows * k_in)
                            .map_err(|e| e.to_string())?;
                        let mut scale: cudarc::driver::CudaSlice<f32> =
                            exec.stream.alloc_zeros(rows).map_err(|e| e.to_string())?;
                        let mut off = 0usize;
                        for pl in [pq, pk, pv] {
                            let n = pl.scale.len();
                            let mut dv = data.slice_mut(off * k_in..(off + n) * k_in);
                            exec.stream
                                .memcpy_dtod(&pl.data, &mut dv)
                                .map_err(|e| e.to_string())?;
                            let mut sv = scale.slice_mut(off..off + n);
                            exec.stream
                                .memcpy_dtod(&pl.scale, &mut sv)
                                .map_err(|e| e.to_string())?;
                            off += n;
                        }
                        Ok(crate::gpu::F8RowPlane { data, scale })
                    };
                    match fuse() {
                        Ok(pl) => Some(pl),
                        Err(e) => {
                            tracing::warn!("granite qkv_f8 fuse skipped: {e}");
                            None
                        }
                    }
                }
                _ => None,
            };
            layers.push(GraniteLayer {
                attn_norm,
                wq,
                wk,
                wv,
                qkv_f8,
                qkv_nv4: fused_qkv,
                wo,
                ffn_norm,
                gate,
                up,
                gate_up: merged,
                down,
            });
        }
        tracing::info!(
            "granite VRAM  attention (q/k/v/o)                  {:>7.2} GB",
            gb(attn_bytes)
        );
        let n_qkv_fused = layers.iter().filter(|l| l.qkv_nv4.is_some()).count();
        if n_qkv_fused > 0 {
            tracing::info!(
                "granite nvf4: fused q|k|v plane on {n_qkv_fused}/{n_layer} layers (split planes not uploaded)"
            );
        }
        tracing::info!(
            "granite VRAM  dense FFN (gate/up/down)             {:>7.2} GB",
            gb(ffn_bytes)
        );

        // ---- head ----------------------------------------------------------
        let output_norm = dt(f32_tensor(&st, "model.norm.weight", n_embd)?, vec![n_embd])?;
        // 4.2 unties the head, so `lm_head.weight` is a real tensor. Branch on
        // PRESENCE for the same reason the GGUF lane does: a tied export omits
        // it and reuses the embedding matrix, and quietly serving the wrong one
        // still produces fluent text.
        let lm_head = if st.bytes("lm_head.weight").is_some() {
            plane("lm_head", n_vocab, n_embd, None)?
        } else {
            tracing::info!("granite: no lm_head.weight - tied head, reusing embed_tokens");
            let raw = bf16_bytes(&st, "model.embed_tokens.weight", n_vocab * n_embd)?;
            GraniteW::Bf16(QuantTensor {
                bytes: exec.to_device_u8(raw).map_err(GpuModelError::from)?,
                ty: GgmlType::Bf16,
                dims: vec![n_embd, n_vocab],
            })
        };
        let head_bytes = output_norm.buf.len() as u64 * 4 + lm_head.bytes();
        tracing::info!(
            "granite VRAM  output head + final norm             {:>7.2} GB",
            gb(head_bytes)
        );

        let weights_bytes = tok_embd.resident_bytes() as u64 + attn_bytes + ffn_bytes + head_bytes;
        exec.trim_mem_pool();
        // The summary names the checkpoint CLASS from what was actually built,
        // not a fixed string: this loader serves IBM's NVFP4 and FP8 exports
        // (and a bf16 one) through the same plane closure, and the fp8 lane
        // once logged "NVFP4 W4A16" too -- a label that did not
        // reflect the configuration. The exact byte count rides along so a
        // registry `weight_bytes` can be measured, not rounded from GB.
        let n_fp8 = layers
            .iter()
            .filter(|l| matches!(l.down, GraniteW::Fp8 { .. }))
            .count();
        let n_nvf4 = layers
            .iter()
            .filter(|l| matches!(l.down, GraniteW::Nvf4 { .. }))
            .count();
        let checkpoint_class = match (n_fp8 > 0, n_nvf4 > 0) {
            (true, false) => "FP8 e4m3 checkpoint (f8row lane)",
            (false, true) => "NVFP4 W4A16 checkpoint",
            (false, false) => "bf16 checkpoint",
            (true, true) => "mixed NVFP4/FP8 checkpoint",
        };
        tracing::info!(
            "granite VRAM  = model resident total               {:>7.2} GB ({} B)  \
             ({n_layer} layers, heads {n_heads}/{n_kv_heads}×{head_dim}, ff {n_ff}, \
             rope θ {:.0}, scales e{} r{} l{} a{}, {})",
            gb(weights_bytes),
            weights_bytes,
            cfg.rope_theta,
            cfg.embedding_scale,
            cfg.residual_scale,
            cfg.logit_scale,
            cfg.attention_scale,
            checkpoint_class,
        );

        Ok(Self {
            exec,
            hp: Hparams {
                n_layer,
                n_embd,
                n_heads,
                n_kv_heads,
                head_dim,
                n_ff,
                n_vocab,
                eps: cfg.eps,
                rope,
                embedding_scale: cfg.embedding_scale,
                residual_scale: cfg.residual_scale,
                logit_scale: cfg.logit_scale,
                attention_scale: cfg.attention_scale,
                // text-only checkpoint: no vision streams to inject. `-1` per
                // layer is "nothing to inject here", which is exactly what the
                // GGUF lane builds when a file carries no `deepstack_mapping`
                // - an empty vec is a different thing and the layer loop reads
                // this by index.
                deepstack: vec![-1; n_layer],
            },
            tok_embd,
            layers,
            output_norm,
            lm_head,
            max_ctx,
            weights_bytes,
            content_id: (
                crate::kv_tier::fingerprint::weights_safetensors(&st),
                crate::kv_tier::fingerprint::tokenizer_dir(dir),
            ),
            // KV8 (fp8-e4m3) by default since 2026-10-04 - the class this
            // family's boards and rivals run, with fp8 arms on every serving
            // path; a die that cannot store fp8 gets f16 back (serving.rs)
            // and an explicit kv_cache_dtype wins
            kv_dtype: KvDtype::Fp8E4m3,
            decode: None,
            scratch: None,
            batch: None,
            chunked: Vec::new(),
            last_reused: Vec::new(),
            seal_hist: Vec::new(),
            seal_ok: Vec::new(),
            vision: None,
            audio: None,
            img_pad_id: None,
            audio_pad_id: None,
            media: Default::default(),
            img_cache: Vec::new(),
            img_cache_bytes: 0,
            img_cache_clock: 0,
            img_cache_reused: 0,
            pipe: None,
            enc: std::collections::VecDeque::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The permutation is transcribed from numpy
    /// (`arange(n).reshape(heads, 2, half).swapaxes(1, 2).reshape(n)`), and an
    /// off-by-one in the loop nest is invisible - it produces a model that
    /// still generates fluent text, just wrong text. So pin the small case that
    /// can be checked by hand.
    #[test]
    fn rope_row_perm_matches_the_numpy_transcription() {
        // heads=2, rows=8 -> half=2
        //   idx            = [[[0,1],[2,3]], [[4,5],[6,7]]]
        //   swapaxes(1,2)  = [[[0,2],[1,3]], [[4,6],[5,7]]]
        assert_eq!(rope_row_perm(2, 8), vec![0, 2, 1, 3, 4, 6, 5, 7]);
        // one head is a plain even/odd deinterleave
        assert_eq!(rope_row_perm(1, 6), vec![0, 3, 1, 4, 2, 5]);
        // every entry must be hit exactly once, at every shipped q/k shape
        for (heads, rows) in [(32usize, 4096usize), (8, 1024), (32, 4096), (8, 1024)] {
            let p = rope_row_perm(heads, rows);
            let mut seen = vec![false; rows];
            for &i in &p {
                assert!(!seen[i], "row {i} produced twice - not a permutation");
                seen[i] = true;
            }
            assert!(seen.iter().all(|&b| b), "permutation drops a row");
        }
    }

    /// Row reordering must never disturb bytes within a row - that is what
    /// makes it safe to apply to packed fp4 nibbles and e4m3 scales without
    /// re-encoding anything.
    #[test]
    fn permute_rows_moves_rows_whole() {
        let src: Vec<u8> = (0..12).collect(); // 4 rows x 3 bytes
        let out = permute_rows(&src, &[3, 0, 1, 2], 3);
        assert_eq!(out, vec![9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
