use crate::device::{Buffer, MetalDevice};
use paddock_models::safetensors::{SafetensorsFile, StDtype};
use std::path::Path;

#[test]
fn recurrent_input_reuse_preserves_outputs_state_and_checkpoints() {
    recurrent_input_reuse(false, false);
}

#[test]
#[ignore = "GPU kernel election only; NOT end-to-end serving throughput"]
fn recurrent_input_reuse_election() {
    recurrent_input_reuse(true, false);
}

#[test]
fn bonsai_recurrent_packing_preserves_f32_state_and_checkpoints() {
    recurrent_input_reuse(false, true);
}

#[test]
#[ignore = "Bonsai F32 packed recurrence: alternating stage costs, not serving throughput"]
fn bonsai_recurrent_packing_cost() {
    recurrent_input_reuse(true, true);
}

fn recurrent_input_reuse(measure: bool, full: bool) {
    let device = MetalDevice::new(Some(512 << 20)).expect("Metal device for recurrent reuse check");
    let upload_u = |values: &[u32]| {
        device
            .upload(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("upload recurrent U32 test metadata")
    };
    let upload_f = |values: &[f32]| {
        device
            .upload(
                &values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("upload recurrent F32 test input")
    };
    let value = |i: usize| {
        let bits = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
        if full {
            ((bits ^ bits.rotate_left(13)) % 65536) as f32 / 513717. - 0.0625
        } else {
            half::bf16::from_f32(((bits ^ bits.rotate_left(13)) % 2048) as f32 / 16384. - 0.0625)
                .to_f32()
        }
    };
    let shapes = [
        vec![(0usize, 1usize, 0usize)],
        vec![(2, 2, 37)],
        vec![(2, 4, 37)],
        vec![(2, 8, 37)],
        vec![(2, 16, 37)],
        vec![(3, 1, 37), (0, 1, 128), (1, 1, 13), (2, 1, 4095)],
        vec![(1, 31, 11), (0, 33, 0)],
        vec![(2, 32, 73)],
        vec![(0, 512, 0)],
        vec![(0, 1024, 0)],
        vec![(0, 256, 0), (1, 256, 0), (2, 256, 0), (3, 256, 0)],
        vec![(0, 128, 0), (1, 128, 0), (2, 128, 0), (3, 128, 0)],
        vec![(2, 1, 71), (0, 127, 0), (3, 384, 19)],
        vec![(3, 257, 511), (1, 13, 31)],
    ];
    for shape in shapes {
        let rows: usize = shape.iter().map(|v| v.1).sum();
        let qkv = upload_f(&(0..rows * 10240).map(value).collect::<Vec<_>>());
        let gates = upload_f(
            &(0..rows * 48)
                .flat_map(|i| match i % 131 {
                    0 => [-110., 0.], // decay underflow and a closed update gate
                    1 => [0., 1.],    // no decay and a fully open update gate
                    _ => [-0.03 - value(i).abs(), 0.5 + value(i + 31)],
                })
                .collect::<Vec<_>>(),
        );
        let mut spans = Vec::new();
        let mut metadata = Vec::new();
        let mut first = 0;
        for &(slot, count, pos) in &shape {
            spans.extend([first as u32, count as u32, slot as u32, 0]);
            metadata.extend((pos..pos + count).flat_map(|p| [slot as u32, p as u32]));
            first += count;
        }
        let spans = upload_u(&spans);
        let meta = upload_u(&metadata);
        let mut cuts = vec![0; rows];
        // Two independent destinations; mixed slots must not race for a cut.
        cuts[rows / 2] = 5;
        cuts[rows - 1] = 6;
        let cuts = upload_u(&cuts);
        // Restored states are F32, not BF16. Exercise cancellation and a
        // spread of exponents, plus untouched layer/slot/tail sentinels.
        let initial: Vec<_> = (0..2 * 7 * 48 * 128 * 128 + 16)
            .map(|i| {
                let bits = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
                f32::from_bits((bits & 0x807fffff) | ((115 + bits % 15) << 23))
            })
            .collect();
        let names = if full {
            if device.tensor_accelerated() {
                &["bonsai_dn_recurrent", "bonsai_dn_recurrent_packed"][..]
            } else {
                &["bonsai_dn_recurrent"][..]
            }
        } else if device.tensor_accelerated() {
            &[
                "mlx_dn_recurrent",
                "mlx_dn_recurrent_quad",
                "mlx_dn_recurrent_packed",
            ][..]
        } else {
            &["mlx_dn_recurrent", "mlx_dn_recurrent_quad"][..]
        };
        let states: Vec<_> = names.iter().map(|_| upload_f(&initial)).collect();
        let outputs: Vec<_> = names
            .iter()
            .map(|_| upload_f(&vec![f32::NAN; rows * 48 * 128 + 16]))
            .collect();
        let p = [16, 48, 10240, rows as u32, 7, 1, 0];
        let run = |route: usize| {
            let cmd = device.begin().expect("begin recurrent candidate dispatch");
            cmd.dispatch(
                names[route],
                &[
                    &qkv,
                    &gates,
                    &states[route],
                    &spans,
                    &meta,
                    &outputs[route],
                    &cuts,
                ],
                &p,
                [
                    if full {
                        [8, 4][route]
                    } else {
                        [32, 8, 4][route]
                    },
                    48,
                    shape.len(),
                ],
                128,
            );
            cmd.finish().expect("finish recurrent candidate dispatch")
        };
        for route in 0..names.len() {
            run(route);
        }
        for (label, buffers, count) in [
            ("state", &states, initial.len()),
            ("output", &outputs, rows * 48 * 128 + 16),
        ] {
            let expected = unsafe { buffers[0].read_f32(0, count) };
            let live = if label == "output" { count - 16 } else { count };
            assert!(expected[..live].iter().all(|v| v.is_finite()));
            for route in 1..names.len() {
                let got = unsafe { buffers[route].read_f32(0, count) };
                let differences = got
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(differences, 0, "{label} {shape:?} {}", names[route]);
            }
        }
        if measure {
            let mut times = vec![Vec::new(); names.len()];
            for repeat in 0..9 {
                for i in 0..names.len() {
                    let route = (i + repeat) % names.len();
                    times[route].push(run(route) * 1000.);
                }
            }
            for (name, times) in names.iter().zip(&mut times) {
                times.sort_by(f64::total_cmp);
                eprintln!(
                    "MLX_RECURRENT_REUSE shape={shape:?} kernel={name} median_ms={:.4}",
                    times[4]
                );
            }
        }
    }
}

fn differences(buffer: &Buffer, fixture: &SafetensorsFile, key: &str) -> (usize, f32) {
    let (info, bytes) = fixture.bytes(key).expect("GPU fixture tensor");
    assert_eq!(info.dtype, StDtype::F32);
    // SAFETY: caller completed the GPU command; host only compares results.
    let actual = unsafe { buffer.read_f32(0, bytes.len() / 4) };
    let mut unequal = 0;
    let mut max_error = 0f32;
    for (index, (value, encoded)) in actual.iter().zip(bytes.chunks_exact(4)).enumerate() {
        let expected = f32::from_le_bytes(encoded.try_into().expect("four-byte F32 fixture value"));
        assert!(value.is_finite() && expected.is_finite());
        assert_eq!(value.to_bits() & 0xffff, 0, "non-BF16 output");
        unequal += usize::from(value.to_bits() != expected.to_bits());
        if value.to_bits() != expected.to_bits() && unequal <= 8 {
            eprintln!("{key}[{index}]: native={value}, reference={expected}");
        }
        max_error = max_error.max((value - expected).abs());
    }
    (unequal, max_error)
}

#[test]
#[ignore = "requires PADDOCK_MLX_OPERATIONS GPU oracle; operation-boundary isolation"]
fn native_operation_boundaries_match_mlx_gpu() {
    let root = std::env::var("PADDOCK_MLX_OPERATIONS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let device = MetalDevice::new(None).unwrap();
    let mut failures = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let op = case["op"].as_str().unwrap();
        let rows = case["rows"].as_u64().unwrap() as usize;
        let fixture = SafetensorsFile::open(&root.join(case["file"].as_str().unwrap())).unwrap();
        let upload = |key| device.upload(fixture.bytes(key).unwrap().1).unwrap();
        let metadata = |values: Vec<u32>| {
            device
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        let x = upload("x");
        let output = device.alloc(fixture.bytes("y").unwrap().1.len()).unwrap();
        let cmd = device.begin().unwrap();
        let result = match op {
            "rms" => {
                let w = upload("w");
                cmd.dispatch(
                    "mlx_rms",
                    &[&x, &w, &output],
                    &[5120, 0, 1e-6f32.to_bits()],
                    [rows, 1, 1],
                    1024,
                );
                &output
            }
            "residual_rms" => {
                let w = upload("w");
                let delta = upload("delta");
                cmd.dispatch(
                    "mlx_residual_rms",
                    &[&x, &delta, &w, &output],
                    &[5120, 0, 1e-6f32.to_bits()],
                    [rows, 1, 1],
                    1024,
                );
                &output
            }
            "swiglu" => {
                let up = upload("up");
                cmd.dispatch(
                    "mlx_swiglu",
                    &[&x, &up],
                    &[(rows * 17408) as u32],
                    [(rows * 17408).div_ceil(256), 1, 1],
                    256,
                );
                &x
            }
            "dn_qk_norm" => {
                cmd.dispatch(
                    "mlx_dn_qk_norm",
                    &[&x],
                    &[16, 48, 10240, rows as u32],
                    [32, rows, 1],
                    32,
                );
                &x
            }
            "dn_gated_norm" => {
                let z = upload("z");
                let w = upload("w");
                cmd.dispatch(
                    "mlx_dn_gated_norm",
                    &[&x, &z, &w],
                    &[16, 48, 10240, rows as u32, 0, 0, 1e-6f32.to_bits()],
                    [48, rows, 1],
                    32,
                );
                &x
            }
            "attn_gate" => {
                let qgate = upload("qgate");
                cmd.dispatch(
                    "mlx_attn_gate",
                    &[&x, &qgate],
                    &[(rows * 6144) as u32],
                    [(rows * 6144).div_ceil(256), 1, 1],
                    256,
                );
                &x
            }
            "qnorm" | "qnorm_rope" => {
                let w = upload("w");
                let positions = metadata(
                    (0..rows)
                        .flat_map(|r| [r as u32 + 9, r as u32 + 9, r as u32 + 9, 0])
                        .collect(),
                );
                let theta = manifest["rope_theta"].as_f64().unwrap() as f32;
                cmd.dispatch(
                    "mlx_qnorm_rope",
                    &[&x, &w, &positions, &output],
                    &[
                        24,
                        4,
                        0,
                        theta.to_bits(),
                        1e-6f32.to_bits(),
                        if op == "qnorm" { 0 } else { 64 },
                    ],
                    [24, rows, 1],
                    32,
                );
                &output
            }
            "dn_conv" => {
                let w = upload("w");
                let history = device.alloc(3 * 10240 * 4).unwrap();
                let meta = metadata((0..rows).flat_map(|r| [0, r as u32]).collect());
                let bounds = metadata((0..rows).flat_map(|_| [0, rows as u32]).collect());
                cmd.dispatch(
                    "mlx_dn_conv",
                    &[&x, &w, &history, &meta, &bounds, &output],
                    &[16, 48, 10240, rows as u32, 1, 0, 0],
                    [(rows * 10240).div_ceil(256), 1, 1],
                    256,
                );
                &output
            }
            "dn_update" => {
                let alpha = upload("alpha");
                let beta = upload("beta");
                let a = upload("a");
                let dt = upload("dt");
                let gates = device.alloc(rows * 48 * 2 * 4).unwrap();
                let state = device.alloc(48 * 128 * 128 * 4).unwrap();
                let meta = metadata((0..rows).flat_map(|r| [0, r as u32]).collect());
                let spans = metadata(vec![0, rows as u32, 0, 0]);
                let checkpoints = metadata(vec![0; rows]);
                let p = [16, 48, 10240, rows as u32, 1, 0, 0];
                cmd.dispatch(
                    "mlx_dn_gates",
                    &[&alpha, &beta, &a, &dt, &gates],
                    &p,
                    [(rows * 48).div_ceil(256), 1, 1],
                    256,
                );
                cmd.dispatch(
                    "mlx_dn_recurrent",
                    &[&x, &gates, &state, &spans, &meta, &output, &checkpoints],
                    &p,
                    [32, 48, 1],
                    128,
                );
                &output
            }
            _ => panic!("unrecognized operation {op}"),
        };
        cmd.finish().unwrap();
        let (unequal, error) = differences(result, &fixture, "y");
        eprintln!("{op} m={rows}: unequal={unequal}, max_error={error}");
        if unequal != 0 {
            failures.push(format!(
                "{op} m={rows}: {unequal} unequal, max_error={error}"
            ));
        }
        if op == "residual_rms" {
            assert_eq!(differences(&x, &fixture, "residual").0, 0);
        }
    }
    assert!(failures.is_empty(), "operation mismatches: {failures:?}");
}

/// Reports arithmetic disagreement; successful execution is not a parity gate.
/// Keep this separate from the strict operation-boundary test above.
#[test]
#[ignore = "requires --attention-only GPU fixtures; diagnostic, not a parity qualification"]
fn native_attention_oracle_diagnostic() {
    let root = std::env::var("PADDOCK_MLX_OPERATIONS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let device = MetalDevice::new(None).unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        assert_eq!(case["op"], "attn_prefill");
        let m = case["rows"].as_u64().unwrap() as usize;
        let f = SafetensorsFile::open(&root.join(case["file"].as_str().unwrap())).unwrap();
        let x = device.upload(f.bytes("x").unwrap().1).unwrap();
        let k = device.upload(f.bytes("k_bf16").unwrap().1).unwrap();
        let v = device.upload(f.bytes("v_bf16").unwrap().1).unwrap();
        let metadata = |values: Vec<u32>| {
            device
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
        };
        let meta = metadata((0..m).flat_map(|r| [0, r as u32]).collect());
        let pages = metadata((0..m.div_ceil(16) as u32).collect());
        let limits = metadata((0..m as u32).collect());
        let tiles = metadata(
            (0..m)
                .step_by(32)
                .flat_map(|r| [r as u32, (m - r).min(32) as u32])
                .collect(),
        );
        let q = device.alloc((m + 32) * 6144 * 2).unwrap();
        let out = device.alloc(m * 6144 * 4).unwrap();
        let zero = metadata(vec![0; m * 6144]);
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "mlx_attention_query",
            &[&x, &q],
            &[6144, 0, m as u32],
            [((m + 32) * 6144).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "mlx_attention_prefill",
            &[&q, &k, &v, &meta, &pages, &out, &tiles, &limits],
            &[24, 4, m.div_ceil(16) as u32, (1.0f32 / 16.0).to_bits()],
            [24, m.div_ceil(32), 1],
            128,
        );
        // Mirror the BF16 attention-result boundary before the gate.
        cmd.dispatch(
            "mlx_residual",
            &[&out, &zero],
            &[(m * 6144) as u32],
            [(m * 6144).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let (unequal, max_error) = differences(&out, &f, "y");
        let (fused_unequal, fused_max_error) = differences(&out, &f, "y_fused");
        let (_, native_f32_error) = differences(&out, &f, "y_f32");
        let reference = device.upload(f.bytes("y").unwrap().1).unwrap();
        let (_, default_f32_error) = differences(&reference, &f, "y_f32");
        eprintln!(
            "ATTENTION_ORACLE {}",
            serde_json::json!({"rows":m,"elements":m*6144,
            "unequal":unequal,"max_error":max_error,"bit_exact":unequal==0,
            "forced_fused_unequal":fused_unequal,"forced_fused_max_error":fused_max_error,
            "native_vs_f32_max_error":native_f32_error,"default_vs_f32_max_error":default_f32_error})
        );
    }
}
