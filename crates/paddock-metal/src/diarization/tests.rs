use super::*;
use half::bf16;

/// Counter-free, same-device A/B: run with --release and --ignored, with
/// PADDOCK_DIAR_MODEL and PADDOCK_DIAR_FIXTURE naming the existing model and
/// oracle directory. The selector is not compiled into serving binaries.
#[test]
#[ignore = "requires local diarization checkpoint, oracle and exclusive benchmark GPU use"]
fn prepared_producers_complete_stream_abba() {
    use paddock_engine::diarization::{Backend, Stream};
    use paddock_models::diarization::Preset;
    use std::time::Instant;

    let path = std::env::var("PADDOCK_DIAR_MODEL").unwrap();
    let fixture = std::env::var("PADDOCK_DIAR_FIXTURE").unwrap();
    let read = |name: &str| {
        let b = std::fs::read(Path::new(&fixture).join(name)).unwrap();
        assert!(b.len().is_multiple_of(4));
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let pcm = read("pcm.f32");
    let mut model = Diarization::load(Path::new(&path), None).unwrap();
    assert!(
        !model.bf && model.device.tensor_accelerated(),
        "M5 Q8 comparison"
    );
    let frontend = model.frontend().unwrap();
    for (name, preset) in [
        ("offline", Preset::Offline),
        ("low", Preset::Low),
        ("very_low", Preset::VeryLow),
        ("ultra_low", Preset::UltraLow),
    ] {
        let oracle = read(&format!("{name}.f32"));
        let mut digest = None;
        // Two warmups, then two ABBA cycles. New requests/reset state each
        // time; the two transport partitions remain balanced per variant.
        for (run, prepared) in [
            false, true, false, true, true, false, false, true, true, false,
        ]
        .into_iter()
        .enumerate()
        {
            model.prepare_q8_consumers = prepared;
            let chunk = if run % 2 == 0 { 16001 } else { 1103 };
            let mut stream = Stream::new(preset);
            let mut out = Vec::new();
            let start = Instant::now();
            for part in pcm.chunks(chunk) {
                out.extend(stream.feed(&mut model, &frontend, part, false).unwrap());
            }
            out.extend(stream.feed(&mut model, &frontend, &[], true).unwrap());
            let wall = start.elapsed().as_secs_f64();
            let flat = out.into_iter().flatten().collect::<Vec<_>>();
            assert_eq!(flat.len(), oracle.len());
            assert!(!flat.is_empty() && flat.iter().chain(&oracle).all(|x| x.is_finite()));
            let rmse = (flat
                .iter()
                .zip(&oracle)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / flat.len() as f64)
                .sqrt();
            let flips = flat
                .iter()
                .zip(&oracle)
                .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
                .count();
            assert!(rmse < 0.005 && flips as f64 / flat.len() as f64 <= 0.001);
            let hash = blake3::hash(&bytes(&flat)).to_hex().to_string();
            if let Some(previous) = &digest {
                assert_eq!(
                    previous, &hash,
                    "{name}: producer/partition changed probabilities"
                );
            }
            digest = Some(hash.clone());
            println!(
                "{}",
                serde_json::json!({"preset":name,"run":run,"warmup":run<2,
                "prepared":prepared,"chunk":chunk,"wall_seconds":wall,"gpu_seconds":stream.gpu_seconds,
                "probabilities_blake3":hash,"rmse":rmse,"decision_differences":flips,
                "weight_bytes":model.weight_bytes(),"workspace_bytes":model.workspace_bytes()})
            );
        }
    }
}

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + salt) % 251) as f32 / 127. - 0.93)
        .collect()
}

fn same_bits(a: &Buffer, b: &Buffer, count: usize, label: &str) {
    let a = unsafe { a.read_u32(count) };
    let b = unsafe { b.read_u32(count) };
    assert!(
        a == b,
        "{label}: {} bit mismatches",
        a.iter().zip(&b).filter(|(a, b)| a != b).count()
    );
}

#[test]
fn attention_prepares_exact_q8_operands_through_shrinking_windows() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let scratch_bytes = MAX_ROWS.div_ceil(128) * 128 * 512 * 2 + 256;
    let old_half = d.upload(&vec![0x7fu8; scratch_bytes]).unwrap();
    let new_half = d.upload(&vec![0x7fu8; scratch_bytes]).unwrap();
    for rows in [684usize, 16, 381, 129, 1, 128, 17] {
        let count = rows * 512;
        let q = d.upload(&bytes(&pattern(count, 13))).unwrap();
        for valid in [1, rows.saturating_sub(3).max(1), rows] {
            let masked = |salt| {
                let mut x = pattern(count, salt);
                for head in 0..8 {
                    x[(head * rows + valid) * 64..(head + 1) * rows * 64].fill(f32::NAN);
                }
                d.upload(&bytes(&x)).unwrap()
            };
            let k = masked(17);
            let v = masked(29);
            let old = d.upload(&bytes(&pattern(count + 64, 53))).unwrap();
            let new = d.upload(&bytes(&pattern(count + 64, 53))).unwrap();
            let p = [
                8,
                64,
                rows as u32,
                valid as u32,
                0,
                0,
                8,
                0,
                rows as u32,
                rows as u32,
            ];
            let c = d.begin().unwrap();
            c.dispatch(
                "diar_attention_f32",
                &[&q, &k, &v, &old],
                &p,
                [8, rows.div_ceil(32), 1],
                128,
            );
            c.dispatch(
                "linear_input_padded",
                &[&old, &old_half],
                &[512, 0, rows as u32],
                [rows.div_ceil(128) * 128 * 2, 1, 1],
                256,
            );
            c.dispatch(
                "diar_attention_q8_input",
                &[&q, &k, &v, &new, &new_half],
                &p,
                [8, rows.div_ceil(32), 1],
                128,
            );
            c.finish().unwrap();
            same_bits(&old, &new, count + 64, "attention F32 + guard");
            same_bits(
                &old_half,
                &new_half,
                scratch_bytes / 4,
                "attention F16 + padding + guard",
            );
            assert!(
                unsafe { new.read_f32(0, count) }
                    .iter()
                    .all(|v| v.is_finite())
            );
        }
    }
}

#[test]
fn gelu_prepares_exact_q8_operands_without_overwriting_live_input() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let (k, n) = (512usize, 2048usize);
    let mut weights = Vec::with_capacity(k * n / 32 * 34);
    for block in 0..k * n / 32 {
        weights.extend(f16::from_f32((block % 7 + 1) as f32 * 0.001).to_le_bytes());
        weights.extend((0..32).map(|i| ((block * 13 + i * 7) % 256) as u8));
    }
    let weights = d.upload(&weights).unwrap();
    let bias = d.upload(&bytes(&pattern(n, 23))).unwrap();
    let scratch_bytes = MAX_ROWS.div_ceil(128) * 128 * n * 2 + 256;
    let old_half = d.upload(&vec![0x7fu8; scratch_bytes]).unwrap();
    let new_half = d.upload(&vec![0x7fu8; scratch_bytes]).unwrap();
    for rows in [684usize, 16, 381, 129, 1, 128, 17] {
        let input = d.upload(&bytes(&pattern(rows * k, 19))).unwrap();
        let operand = d.alloc(rows.div_ceil(128) * 128 * k * 2).unwrap();
        let count = rows * n;
        let old = d.upload(&bytes(&pattern(count + 64, 71))).unwrap();
        let new = d.upload(&bytes(&pattern(count + 64, 71))).unwrap();
        let p = [k as u32, n as u32, rows as u32, 1, 3];
        let c = d.begin().unwrap();
        c.dispatch(
            "linear_input_padded",
            &[&input, &operand],
            &p,
            [(operand.len() / 2).div_ceil(256), 1, 1],
            256,
        );
        c.dispatch(
            "diar_q8_project32_prepare",
            &[&weights, &operand, &new, &bias, &new_half],
            &p,
            [n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
        // Run the unfused producer afterwards using the same operand; an
        // accidental overwrite of the live input also breaks the F32 equality.
        c.dispatch(
            "diar_q8_project32",
            &[&weights, &operand, &old, &bias],
            &p,
            [n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
        c.dispatch(
            "linear_input_padded",
            &[&old, &old_half],
            &[n as u32, 0, rows as u32],
            [(rows.div_ceil(128) * 128 * n).div_ceil(256), 1, 1],
            256,
        );
        c.finish().unwrap();
        same_bits(&old, &new, count + 64, "GELU F32 + guard");
        same_bits(
            &old_half,
            &new_half,
            scratch_bytes / 4,
            "GELU F16 + padding + guard",
        );
    }
}

#[test]
fn normalization_prepares_exact_q8_operands_and_clears_padding() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let weight = d.upload(&bytes(&pattern(512, 41))).unwrap();
    let bias = d.upload(&bytes(&pattern(512, 83))).unwrap();
    // Reuse scratch through shrinking/growing windows. Padding must be zero,
    // never an earlier window's values; F32 output/guards must stay untouched.
    let scratch_bytes = MAX_ROWS.div_ceil(128) * 128 * 512 * 2 + 256;
    let poison = vec![0x7fu8; scratch_bytes];
    let prepared_old = d.upload(&poison).unwrap();
    let prepared_new = d.upload(&poison).unwrap();
    for residual in [false, true] {
        for rows in [684usize, 16, 382, 1, 129, 15, 128, 17, 127] {
            let count = rows * 512;
            let initial = pattern(count + 64, 51);
            let old_x = d.upload(&bytes(&initial)).unwrap();
            let new_x = d.upload(&bytes(&initial)).unwrap();
            let delta = d.upload(&bytes(&pattern(count + 64, 11))).unwrap();
            let old_norm = d.upload(&bytes(&initial)).unwrap();
            let new_norm = d.upload(&bytes(&initial)).unwrap();
            let c = d.begin().unwrap();
            if residual {
                c.dispatch(
                    "diar_residual_norm_f32",
                    &[&delta, &old_x, &weight, &bias, &old_norm],
                    &[],
                    [rows, 1, 1],
                    256,
                );
            } else {
                c.dispatch(
                    "diar_norm",
                    &[&old_x, &weight, &bias, &old_norm],
                    &[0],
                    [rows, 1, 1],
                    256,
                );
            }
            c.dispatch(
                "linear_input_padded",
                &[&old_norm, &prepared_old],
                &[512, 0, rows as u32],
                [rows.div_ceil(128) * 128 * 2, 1, 1],
                256,
            );
            if residual {
                c.dispatch(
                    "diar_residual_norm_q8_input",
                    &[&delta, &new_x, &weight, &bias, &new_norm, &prepared_new],
                    &[rows as u32],
                    [rows.div_ceil(128) * 128, 1, 1],
                    256,
                );
            } else {
                c.dispatch(
                    "diar_norm_q8_input",
                    &[&new_x, &weight, &bias, &new_norm, &prepared_new],
                    &[rows as u32],
                    [rows.div_ceil(128) * 128, 1, 1],
                    256,
                );
            }
            c.finish().unwrap();
            same_bits(&old_x, &new_x, count + 64, "residual + guard");
            same_bits(&old_norm, &new_norm, count + 64, "F32 norm + guard");
            same_bits(
                &prepared_old,
                &prepared_new,
                scratch_bytes / 4,
                &format!("prepared F16 + row padding + guard: residual={residual}, M={rows}"),
            );
        }
    }
}

#[test]
fn residual_norm_preserves_both_arithmetic_contracts() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let weight = d.upload(&bytes(&pattern(512, 41))).unwrap();
    let bias = d.upload(&bytes(&pattern(512, 83))).unwrap();
    for bf in [false, true] {
        for rows in [1usize, 31, 382, 684] {
            let count = rows * 512;
            let residual = pattern(count + 64, 51);
            let delta = pattern(count + 64, 11);
            let old_x = d.upload(&bytes(&residual)).unwrap();
            let new_x = d.upload(&bytes(&residual)).unwrap();
            let old_delta = d.upload(&bytes(&delta)).unwrap();
            let new_delta = d.upload(&bytes(&delta)).unwrap();
            let old_norm = d.upload(&bytes(&residual)).unwrap();
            let new_norm = d.upload(&bytes(&residual)).unwrap();
            let c = d.begin().unwrap();
            c.dispatch(
                "diar_post",
                &[&old_delta, &bias, &old_x],
                &[count as u32, 512, u32::from(bf), 2],
                [count.div_ceil(256), 1, 1],
                256,
            );
            let norm_params = if bf { [512, 1e-5f32.to_bits()] } else { [0, 0] };
            c.dispatch(
                if bf { "gmlx_layer_norm" } else { "diar_norm" },
                &[&old_x, &weight, &bias, &old_norm],
                &norm_params,
                [rows, 1, 1],
                if bf { 128 } else { 256 },
            );
            c.dispatch(
                if bf {
                    "diar_residual_norm_bf16"
                } else {
                    "diar_residual_norm_f32"
                },
                &[&new_delta, &new_x, &weight, &bias, &new_norm],
                &[],
                [rows, 1, 1],
                if bf { 128 } else { 256 },
            );
            c.finish().unwrap();
            same_bits(&old_x, &new_x, count + 64, "residual + guard");
            same_bits(&old_norm, &new_norm, count + 64, "norm + guard");
        }
    }
}

#[test]
fn packed_q8_epilogues_preserve_operands_rounding_and_tails() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    for (k, n) in [
        (512usize, 1536usize),
        (512, 2048),
        (2048, 512),
        (512, 192),
        (192, 8),
    ] {
        let mut weights = Vec::with_capacity(k * n / 32 * 34);
        for block in 0..k * n / 32 {
            weights.extend(f16::from_f32((block % 7 + 1) as f32 * 0.001).to_le_bytes());
            weights.extend((0..32).map(|i| ((block * 13 + i * 7) % 256) as u8));
        }
        let weights = d.upload(&weights).unwrap();
        let bias = d.upload(&bytes(&pattern(n, 23))).unwrap();
        for rows in [16usize, 17, 31, 32, 65, 381] {
            let count = rows * n;
            let input = d.upload(&bytes(&pattern(rows * k, 19))).unwrap();
            let prepared = d
                .alloc(k.div_ceil(128) * 128 * rows.div_ceil(128) * 128 * 2)
                .unwrap();
            let initial = pattern(count + 64, 71);
            for (op, biased) in [(0, false), (0, true), (2, true), (3, true)] {
                let old = d.upload(&bytes(&initial)).unwrap();
                let residual = d.upload(&bytes(&initial)).unwrap();
                let new = d.upload(&bytes(&initial)).unwrap();
                let c = d.begin().unwrap();
                let params = [k as u32, n as u32, rows as u32, 8, 1f32.to_bits()];
                c.dispatch(
                    "linear_input_padded",
                    &[&input, &prepared],
                    &params,
                    [(prepared.len() / 2).div_ceil(256), 1, 1],
                    256,
                );
                c.dispatch(
                    "linear_quant_tile32",
                    &[&weights, &prepared, &old],
                    &params,
                    [n.div_ceil(64), rows.div_ceil(32), 1],
                    128,
                );
                if biased {
                    c.dispatch(
                        "diar_post",
                        &[&old, &bias, &old],
                        &[count as u32, n as u32, 0, 1],
                        [count.div_ceil(256), 1, 1],
                        256,
                    );
                }
                if op != 0 {
                    c.dispatch(
                        "diar_post",
                        &[&old, &bias, &residual],
                        &[count as u32, n as u32, 0, op],
                        [count.div_ceil(256), 1, 1],
                        256,
                    );
                }
                c.dispatch(
                    "diar_q8_project32",
                    &[&weights, &prepared, &new, &bias],
                    &[k as u32, n as u32, rows as u32, u32::from(biased), op],
                    [n.div_ceil(64), rows.div_ceil(32), 1],
                    128,
                );
                c.finish().unwrap();
                same_bits(
                    if op == 2 { &residual } else { &old },
                    &new,
                    count + 64,
                    &format!("Q8 M={rows} N={n} K={k} op={op} bias={biased}"),
                );
            }
        }
    }
}

#[test]
fn attention_packing_preserves_masked_keys_and_output_bits() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    for bf in [false, true] {
        for rows in [1usize, 31, 33, 382, 684] {
            let count = rows * 512;
            let upload = |values: &[f32]| {
                d.upload(&if bf {
                    values
                        .iter()
                        .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
                        .collect()
                } else {
                    bytes(values)
                })
                .unwrap()
            };
            let packed = |values: &[f32]| -> Vec<f32> {
                let mut output = vec![0.; count];
                for row in 0..rows {
                    for head in 0..8 {
                        for col in 0..64 {
                            output[(head * rows + row) * 64 + col] =
                                values[row * 512 + head * 64 + col];
                        }
                    }
                }
                output
            };
            for valid in [1, rows.saturating_sub(3).max(1), rows] {
                let q = pattern(count, 13);
                let mut k = pattern(count, 17);
                let mut v = pattern(count, 29);
                k[valid * 512..].fill(f32::NAN);
                v[valid * 512..].fill(f32::NAN);
                let oq = upload(&q);
                let ok = upload(&k);
                let ov = upload(&v);
                let nq = upload(&packed(&q));
                let nk = upload(&packed(&k));
                let nv = upload(&packed(&v));
                let old = d.upload(&bytes(&pattern(count + 64, 53))).unwrap();
                let new = d.upload(&bytes(&pattern(count + 64, 53))).unwrap();
                let c = d.begin().unwrap();
                let params = [
                    8,
                    64,
                    rows as u32,
                    valid as u32,
                    0,
                    0,
                    8,
                    0,
                    rows as u32,
                    rows as u32,
                ];
                c.dispatch(
                    if bf {
                        "diar_attention_row_check"
                    } else {
                        "kumo_attention_tile64"
                    },
                    &[&oq, &ok, &ov, &old],
                    &params,
                    [8, rows.div_ceil(32), 1],
                    128,
                );
                c.dispatch(
                    if bf {
                        "diar_attention_bf16"
                    } else {
                        "diar_attention_f32"
                    },
                    &[&nq, &nk, &nv, &new],
                    &params,
                    [8, rows.div_ceil(32), 1],
                    128,
                );
                c.finish().unwrap();
                assert!(
                    unsafe { new.read_f32(0, count) }
                        .iter()
                        .all(|v| v.is_finite())
                );
                same_bits(
                    &old,
                    &new,
                    count + 64,
                    &format!("attention bf={bf} rows={rows} valid={valid}"),
                );
            }
            let qkv = pattern(rows * 1536, 19);
            let input = d.upload(&bytes(&qkv)).unwrap();
            let q = d.alloc(count * 4).unwrap();
            let k = d.alloc(count * 4).unwrap();
            let v = d.alloc(count * 4).unwrap();
            let view = d.alloc(count * 4).unwrap();
            let c = d.begin().unwrap();
            c.dispatch(
                "diar_heads",
                &[&input, &q, &k, &v],
                &[rows as u32, u32::from(bf)],
                [count.div_ceil(256), 1, 1],
                256,
            );
            c.dispatch(
                "diar_heads_trace",
                &[&v, &view],
                &[rows as u32, u32::from(bf)],
                [count.div_ceil(256), 1, 1],
                256,
            );
            c.finish().unwrap();
            let expected: Vec<_> = (0..count)
                .map(|i| {
                    let x = qkv[i / 512 * 1536 + 1024 + i % 512];
                    if bf { bf16::from_f32(x).to_f32() } else { x }
                })
                .collect();
            assert_eq!(
                unsafe { view.read_f32(0, count) },
                expected,
                "V packing and trace inversion"
            );
        }
    }
}

#[test]
fn fused_projections_preserve_materialized_bf16_boundaries_and_tails() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let values = |n: usize, salt: usize| -> Vec<f32> {
        (0..n)
            .map(|i| bf16::from_f32(((i * 37 + salt) % 251) as f32 / 127. - 0.93).to_f32())
            .collect()
    };
    // Actual encoder/head shapes, including the narrow eight-speaker tail
    // and maximum upsampled output. Compare with the unfused serving graph,
    // not a lower-precision CPU approximation of tensor accumulation.
    for (k, n, op) in [
        (1024, 512, 0),
        (512, 1536, 0),
        (512, 512, 2),
        (512, 2048, 3),
        (2048, 512, 2),
        (576, 1536, 4),
        (192, 192, 4),
        (192, 8, 5),
    ] {
        let weights = d
            .upload(
                &values(k * n, 7)
                    .iter()
                    .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let bias = d.upload(&bytes(&values(n, 59))).unwrap();
        for rows in [1usize, 31, 64, 65, 381, 684] {
            let rows = if k == 192 { rows * 8 } else { rows };
            let count = rows * n;
            let input = d.upload(&bytes(&values(rows * k, 19))).unwrap();
            let initial = values(count + 64, 71);
            let temp = d.upload(&bytes(&initial)).unwrap();
            let expected = d.upload(&bytes(&initial)).unwrap();
            let actual = d.upload(&bytes(&initial)).unwrap();
            let bias_on = u32::from(op != 0);
            for (kernel, tile) in [("diar_project32", 32), ("diar_project64", 64)] {
                unsafe {
                    actual.write_u32(&initial.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                }
                let c = d.begin().unwrap();
                c.dispatch(
                    "gmlx_vmm64",
                    &[&weights, &input, &temp, &bias],
                    &[k as u32, n as u32, rows as u32, bias_on],
                    [n.div_ceil(64), rows.div_ceil(64), 1],
                    128,
                );
                if op != 0 {
                    unsafe {
                        expected
                            .write_u32(&initial.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    }
                    c.dispatch(
                        "diar_post",
                        &[&temp, &bias, &expected],
                        &[count as u32, n as u32, 1, op],
                        [count.div_ceil(256), 1, 1],
                        256,
                    );
                }
                c.dispatch(
                    kernel,
                    &[&weights, &input, &actual, &bias],
                    &[k as u32, n as u32, rows as u32, bias_on, op],
                    [n.div_ceil(64), rows.div_ceil(tile), 1],
                    128,
                );
                c.finish().unwrap();
                let reference = if op == 2 { &expected } else { &temp };
                let a = unsafe { actual.read_u32(count + 64) };
                let b = unsafe { reference.read_u32(count + 64) };
                assert!(
                    a == b,
                    "{kernel}: M={rows}, N={n}, K={k}, op={op}; mismatches={}",
                    a.iter().zip(&b).filter(|(a, b)| a != b).count()
                );
            }
        }
    }
}
