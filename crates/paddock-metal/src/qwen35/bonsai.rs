//! Native packed ternary lane. Only embedding lookup rounds to F16; F32
//! auxiliary tensors promote the decoder, attention cache and residuals to F32.
use super::*;
use paddock_models::{
    bonsai::BonsaiConfig,
    safetensors::{ShardedSafetensors, StDtype, qwen35_hf_name},
};
use std::collections::BTreeMap;

pub(super) const AFFINE2: u32 = 0x102;

#[cfg(test)]
thread_local! {
    pub(super) static BASELINE_PROJECTIONS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_PREFILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(format!("Bonsai MLX: {}", s.into()))
}

fn hf_name(name: &str) -> Result<String> {
    Ok(match name {
        "token_embd.weight" => "language_model.model.embed_tokens.weight".into(),
        "output_norm.weight" => "language_model.model.norm.weight".into(),
        "output.weight" => "language_model.lm_head.weight".into(),
        _ => qwen35_hf_name(name)
            .and_then(|n| {
                n.strip_prefix("model.language_model.")
                    .map(|n| format!("language_model.model.{n}"))
            })
            .ok_or_else(|| error(format!("unmapped {name}")))?,
    })
}

fn bytes<'a>(
    source: &'a ShardedSafetensors,
    name: &str,
    dtype: StDtype,
    shape: &[usize],
) -> Result<&'a [u8]> {
    let (info, data) = source
        .bytes(name)
        .ok_or_else(|| error(format!("missing {name}")))?;
    if info.dtype != dtype || info.shape != shape {
        return Err(error(format!(
            "{name}: expected {dtype:?} {shape:?}, got {:?} {:?}",
            info.dtype, info.shape
        )));
    }
    Ok(data)
}

pub(super) fn validate_tensors(source: &ShardedSafetensors, config: &BonsaiConfig) -> Result<()> {
    if source.names().any(|n| n.contains("mtp.")) {
        return Err(error("MTP is not part of this checkpoint"));
    }
    for (module, width) in paddock_models::bonsai::packed_modules() {
        let stem = format!("language_model.{module}");
        let signs = bytes(source, &format!("{stem}.signs"), StDtype::F32, &[width])?;
        if signs
            .chunks_exact(4)
            .zip(&config.signs[&width])
            .any(|(a, b)| f32::from_le_bytes(a.try_into().expect("four-byte sign")) != *b)
        {
            return Err(error(format!("{stem}: signs disagree with hadamard.json")));
        }
        let (info, codes) = source
            .bytes(&format!("{stem}.weight"))
            .ok_or_else(|| error(format!("missing {stem}")))?;
        if info.dtype != StDtype::U32 || info.shape.len() != 2 || info.shape[1] != width / 16 {
            return Err(error(format!("invalid packed {stem}")));
        }
        // Code 3 is not ternary. Reject rather than silently decode a fourth level.
        if codes.chunks_exact(4).any(|b| {
            let word = u32::from_le_bytes(b.try_into().expect("four-byte packed word"));
            word & (word >> 1) & 0x5555_5555 != 0
        }) {
            return Err(error(format!("{stem}: non-ternary code")));
        }
        let shape = [info.shape[0], width / 128];
        let scales = bytes(source, &format!("{stem}.scales"), StDtype::F16, &shape)?;
        let biases = bytes(source, &format!("{stem}.biases"), StDtype::F16, &shape)?;
        if scales
            .chunks_exact(2)
            .zip(biases.chunks_exact(2))
            .any(|(s, b)| {
                let s = u16::from_le_bytes([s[0], s[1]]);
                let b = u16::from_le_bytes([b[0], b[1]]);
                s & 0x7c00 == 0x7c00 || b != s ^ 0x8000
            })
        {
            return Err(error(format!(
                "{stem}: requires finite scales and bias = -scale"
            )));
        }
    }
    Ok(())
}

pub(super) fn load_weight(
    device: &MetalDevice,
    source: &ShardedSafetensors,
    name: &str,
    dims: &[usize],
) -> Result<Weight> {
    let hf = hf_name(name)?;
    let (info, _) = source
        .bytes(&hf)
        .ok_or_else(|| error(format!("missing {hf}")))?;
    let packed = info.dtype == StDtype::U32;
    let buffer = if packed {
        if dims.len() != 2 {
            return Err(error("packed tensor must be a matrix"));
        }
        let (k, n) = (dims[0], dims[1]);
        let stem = hf
            .strip_suffix(".weight")
            .ok_or_else(|| error("packed tensor is not a weight"))?;
        // Bias is provably redundant, validated before any allocation. Keep
        // native 2-bit codes and exact FP16 scales, no full-weight expansion.
        device.upload_parts(&[
            bytes(source, &hf, StDtype::U32, &[n, k / 16])?,
            bytes(
                source,
                &format!("{stem}.scales"),
                StDtype::F16,
                &[n, k / 128],
            )?,
        ])?
    } else {
        let shape = if name.ends_with("ssm_conv1d.weight") {
            vec![dims[1], dims[0], 1]
        } else {
            dims.iter().copied().rev().collect()
        };
        let data = bytes(source, &hf, StDtype::F32, &shape)?;
        if data
            .chunks_exact(4)
            .any(|b| !f32::from_le_bytes(b.try_into().expect("four-byte scalar")).is_finite())
        {
            return Err(error(format!("non-finite {hf}")));
        }
        let buffer = device.upload_parts(&[data])?;
        if name.ends_with("ssm_a") {
            let cmd = device.begin()?;
            cmd.dispatch(
                "bonsai_a",
                &[&buffer],
                &[dims[0] as u32],
                [dims[0].div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
        }
        buffer
    };
    Ok(Weight {
        buffer,
        ty: if packed { AFFINE2 } else { 0 },
        k: dims[0],
        n: *dims.get(1).unwrap_or(&1),
    })
}

pub(super) struct Bonsai {
    signs: BTreeMap<usize, Buffer>,
    rotated: Buffer,
}

impl Bonsai {
    pub(super) fn new(device: &MetalDevice, config: &BonsaiConfig) -> Result<Self> {
        let mut signs = BTreeMap::new();
        for (&width, values) in &config.signs {
            let buffer = device.upload_parts(&[&values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()])?;
            signs.insert(width, buffer);
        }
        Ok(Self {
            signs,
            rotated: device.alloc(CHUNK * 17408 * 4)?,
        })
    }
    pub(super) fn embed(
        &self,
        cmd: &Commands<'_>,
        weight: &Weight,
        ids: &Buffer,
        output: &Buffer,
        rows: usize,
    ) {
        cmd.dispatch(
            "bonsai_embed",
            &[&weight.buffer, ids, &self.signs[&weight.k], output],
            &[weight.k as u32, weight.n as u32, rows as u32],
            [weight.k / 1024, rows, 1],
            256,
        );
    }
    pub(super) fn project(
        &self,
        cmd: &Commands<'_>,
        planes: &[(&Weight, &Buffer)],
        input: &Buffer,
        rows: usize,
        spans: Option<&[(usize, usize, usize)]>,
    ) {
        let k = planes[0].0.k;
        assert!(planes.iter().all(|(w, _)| w.k == k) && (1..=CHUNK).contains(&rows));
        if let Some(spans) = spans {
            let mut end = 0;
            for &(first, count, logical) in spans {
                assert!(first == end && count > 0 && first + count <= rows);
                assert!(logical == 1 || logical == CHUNK);
                end += count;
            }
            assert_eq!(end, rows);
        }
        if planes.iter().any(|(w, _)| w.ty == AFFINE2) {
            cmd.dispatch(
                "bonsai_rotate",
                &[input, &self.signs[&k], &self.rotated],
                &[k as u32, rows as u32],
                [k / 1024, rows, 1],
                256,
            );
        }
        if let Some(spans) = spans {
            for &(first, count, logical) in spans {
                self.project_span(cmd, planes, input, first, count, logical == CHUNK);
            }
        } else {
            self.project_span(cmd, planes, input, 0, rows, false);
        }
    }
    fn project_span(
        &self,
        cmd: &Commands<'_>,
        planes: &[(&Weight, &Buffer)],
        input: &Buffer,
        first: usize,
        rows: usize,
        prefill: bool,
    ) {
        let k = planes[0].0.k;
        let packed = planes.iter().take_while(|(w, _)| w.ty == AFFINE2).count();
        // These dispatch/occupancy elections are qualified on Apple10 (M5).
        // Preserve the existing route on older GPUs until measured there.
        let fused = !prefill && cmd.tensor_accelerated() && rows <= 4 && (2..=3).contains(&packed);
        #[cfg(test)]
        let fused = fused && !BASELINE_PROJECTIONS.with(|v| v.get());
        #[cfg(test)]
        let fused = fused && !ternary_add_tests::ADD_PROJECTIONS.with(|v| v.get());
        if fused {
            let (a, ao) = planes[0];
            let (b, bo) = planes[1];
            let (c, co) = planes[if packed == 3 { 2 } else { 1 }];
            cmd.dispatch_at(
                [
                    "bonsai_multi1",
                    "bonsai_multi2",
                    "bonsai_multi3",
                    "bonsai_multi4",
                ][rows - 1],
                &[&a.buffer, &b.buffer, &c.buffer, &self.rotated, ao, bo, co],
                &[
                    0,
                    0,
                    0,
                    first * k * 4,
                    first * a.n * 4,
                    first * b.n * 4,
                    first * c.n * 4,
                ],
                &[
                    k as u32,
                    a.n as u32,
                    b.n as u32,
                    if packed == 3 { c.n as u32 } else { 0 },
                    rows as u32,
                ],
                [
                    planes[..packed].iter().map(|(w, _)| w.n.div_ceil(16)).sum(),
                    1,
                    1,
                ],
                128,
            );
        }
        for &(w, out) in &planes[if fused { packed } else { 0 }..] {
            let input = if w.ty == AFFINE2 {
                &self.rotated
            } else {
                input
            };
            let optimized = w.ty == AFFINE2 && cmd.tensor_accelerated();
            #[cfg(test)]
            let optimized = optimized && !BASELINE_PROJECTIONS.with(|v| v.get());
            let (kernel, cols, tile) = if prefill && w.ty == AFFINE2 {
                if rows <= 16 {
                    ("bonsai_prefill16", 32, 16)
                } else if rows <= 32 {
                    ("bonsai_prefill32", 32, 32)
                } else {
                    ("bonsai_prefill64", 32, 64)
                }
            } else if prefill {
                // Unquantized F32 gates keep strict arithmetic, including
                // single-token prompt suffixes restored from the cache.
                ("bonsai_mm32", 16, 32)
            } else if optimized && rows <= 4 {
                (
                    [
                        "bonsai_full1",
                        "bonsai_full2",
                        "bonsai_full3",
                        "bonsai_full4",
                    ][rows - 1],
                    16,
                    rows,
                )
            } else if optimized && w.n >= 5120 && rows > 4 {
                if rows <= 32 {
                    ("bonsai_tile32x32x64", 32, 32)
                } else {
                    ("bonsai_tile64x32x64", 32, 64)
                }
            } else if rows == 1 {
                ("bonsai_vectors1", 16, 1)
            } else if rows <= 4 {
                ("bonsai_vectors4", 16, 4)
            } else if rows <= 32 {
                ("bonsai_mm32", 16, 32)
            } else {
                ("bonsai_mm64", 16, 64)
            };
            #[cfg(test)]
            let (kernel, cols, tile) = if w.ty == AFFINE2 {
                ternary_add_tests::election(false, rows).unwrap_or((kernel, cols, tile))
            } else {
                (kernel, cols, tile)
            };
            cmd.dispatch_at(
                kernel,
                &[&w.buffer, input, out],
                &[0, first * k * 4, first * w.n * 4],
                &[k as u32, w.n as u32, rows as u32, w.ty],
                [w.n.div_ceil(cols), rows.div_ceil(tile), 1],
                128,
            );
        }
    }
}

impl Qwen35 {
    pub(super) fn bonsai_attention(
        &self,
        cmd: &Commands<'_>,
        w: &FullAttention,
        rows: usize,
        tiles: usize,
        decode_rows: usize,
        decode_length: usize,
    ) {
        let s = &self.scratch;
        self.project(
            cmd,
            &[(&w.q, &s.qraw), (&w.k, &s.k), (&w.v, &s.v)],
            &s.norm,
            rows,
            &s.gemm,
        );
        let p = [
            24,
            4,
            self.page_stride as u32,
            self.rope.to_bits(),
            self.eps.to_bits(),
            self.rotary as u32,
        ];
        cmd.dispatch(
            "qwen_qnorm_rope",
            &[&s.qraw, &w.q_norm.buffer, &s.mrope, &s.q],
            &p,
            [24, rows, 1],
            32,
        );
        cmd.dispatch(
            "bonsai_knorm_store",
            &[
                &s.k,
                &s.v,
                &w.k_norm.buffer,
                &s.meta,
                &s.pages,
                &w.keys,
                &w.values,
                &s.mrope,
            ],
            &p,
            [4, rows, 1],
            32,
        );
        if tiles > 0 {
            cmd.dispatch(
                "bonsai_attention_prefill",
                &[
                    &s.q,
                    &w.keys,
                    &w.values,
                    &s.meta,
                    &s.pages,
                    &s.attn,
                    &s.attn_tiles,
                    &s.limits,
                ],
                &[24, 4, self.page_stride as u32, (1.0f32 / 16.0).to_bits()],
                [24, tiles, 1],
                128,
            );
        }
        if decode_rows > 0 {
            let splits = decode_length.div_ceil(128).clamp(16, MAX_SPLITS);
            cmd.dispatch(
                "bonsai_attention_decode",
                &[
                    &s.q,
                    &w.keys,
                    &w.values,
                    &s.meta,
                    &s.pages,
                    &s.decode_rows,
                    &s.attn_parts,
                    &s.limits,
                ],
                &[
                    24,
                    4,
                    self.page_stride as u32,
                    (1.0f32 / 16.0).to_bits(),
                    splits as u32,
                    16,
                ],
                [4, decode_rows, splits],
                128,
            );
            cmd.dispatch(
                "qwen_attention_merge",
                &[&s.attn_parts, &s.attn, &s.decode_rows],
                &[24, splits as u32],
                [24 * decode_rows, 1, 1],
                32,
            );
        }
        cmd.dispatch(
            "qwen_attn_gate",
            &[&s.attn, &s.qraw],
            &[(rows * 24 * 256) as u32],
            [(rows * 24 * 256).div_ceil(256), 1, 1],
            256,
        );
        self.project(cmd, &[(&w.o, &s.delta)], &s.attn, rows, &s.gemm);
    }
}

#[cfg(test)]
mod phase_tests {
    use super::*;

    #[test]
    fn mixed_projection_roles_preserve_every_row_and_guard() {
        let device = MetalDevice::new(None).unwrap();
        if !device.tensor_accelerated() {
            return;
        }
        let (k, rows) = (1024usize, 78usize);
        let upload = |v: &[f32]| {
            device
                .upload_parts(&[&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()])
                .unwrap()
        };
        let lane = Bonsai {
            signs: BTreeMap::from([(
                k,
                upload(
                    &(0..k)
                        .map(|i| if i % 3 == 0 { -1. } else { 1. })
                        .collect::<Vec<_>>(),
                ),
            )]),
            rotated: device.alloc(k * rows * 4).unwrap(),
        };
        let input = upload(
            &(0..rows * k)
                .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
                .collect::<Vec<_>>(),
        );
        let weights = [19usize, 35, 67].map(|n| {
            let mut bytes = (0..k * n / 16)
                .flat_map(|i| {
                    (0..16)
                        .fold(0u32, |w, j| w | (((i * 37 + j * 11) % 3) as u32) << (j * 2))
                        .to_le_bytes()
                })
                .collect::<Vec<_>>();
            bytes.extend((0..k * n / 128).flat_map(|i| {
                half::f16::from_f32((i % 103 + 1) as f32 / 1031.)
                    .to_bits()
                    .to_le_bytes()
            }));
            Weight {
                buffer: device.upload_parts(&[&bytes]).unwrap(),
                k,
                n,
                ty: AFFINE2,
            }
        });
        let gate = Weight {
            buffer: upload(
                &(0..k * 67)
                    .map(|i| ((i * 17 % 257) as f32 - 128.) / 1023.3)
                    .collect::<Vec<_>>(),
            ),
            k,
            n: 67,
            ty: 0,
        };
        let spans = [
            (0, 2, 1),
            (2, 7, CHUNK),
            (9, 1, 1),
            (10, 64, CHUNK),
            (74, 3, 1),
            (77, 1, CHUNK),
        ];
        let singletons = spans
            .iter()
            .flat_map(|&(first, count, role)| (first..first + count).map(move |r| (r, 1, role)))
            .collect::<Vec<_>>();
        let sentinel = 12345.0f32;
        for third in [&weights[2], &gate] {
            let mut expected = Vec::new();
            for roles in [&singletons[..], &spans[..]] {
                let outputs = [19usize, 35, 67].map(|n| upload(&vec![sentinel; rows * n + 16]));
                let cmd = device.begin().unwrap();
                lane.project(
                    &cmd,
                    &[
                        (&weights[0], &outputs[0]),
                        (&weights[1], &outputs[1]),
                        (third, &outputs[2]),
                    ],
                    &input,
                    rows,
                    Some(roles),
                );
                cmd.finish().unwrap();
                let actual = [19usize, 35, 67]
                    .into_iter()
                    .zip(&outputs)
                    .map(|(n, o)| {
                        let v = unsafe { o.read_f32(0, rows * n + 16) };
                        assert!(
                            v[..rows * n]
                                .iter()
                                .all(|x| x.is_finite() && *x != sentinel)
                        );
                        assert!(v[rows * n..].iter().all(|x| *x == sentinel));
                        v
                    })
                    .collect::<Vec<_>>();
                if expected.is_empty() {
                    expected = actual;
                } else {
                    assert_eq!(
                        actual, expected,
                        "mixed phases changed row arithmetic, third type={}",
                        third.ty
                    );
                }
            }
        }
    }
}
