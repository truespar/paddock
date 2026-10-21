use crate::{
    affine,
    device::{Buffer, Commands, MetalDevice},
    weights::Weight,
};
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors, StDtype};
use std::path::Path;

#[test]
fn affine_small_admission_is_exact_and_bounded() {
    small_admission_control(false, false);
}

#[test]
fn affine_tiny_prompt_is_exact_and_bounded() {
    small_admission_control(true, false);
}

#[test]
fn affine_gemma_prefill_is_exact_and_bounded() {
    small_admission_control(false, true);
}

fn small_admission_control(tiny_only: bool, gemma: bool) {
    let d = MetalDevice::new(None).expect("Metal projection test GPU operation");
    if !d.tensor_accelerated() {
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::BASELINE_ADMISSION_FOR_TEST.with(|v| v.set(false));
            affine::BASELINE_TAIL_FOR_TEST.with(|v| v.set(false));
            affine::BASELINE_STAGING_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let shapes = if gemma {
        vec![(5376, 1057), (8192, 5376), (16384, 5376), (21504, 5376)]
    } else {
        vec![
            (5120usize, 48usize),
            (5120, 65),
            (5120, 17408),
            (6144, 5120),
            (17408, 5120),
        ]
    };
    for (k, n) in shapes {
        let mut bytes: Vec<_> = (0..k * n / 8)
            .flat_map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes())
            .collect();
        for bias in [false, true] {
            bytes.extend((0..k * n / 64).flat_map(|i| {
                let scale = (i % 251 + 1) as f32 / 9973.;
                half::bf16::from_f32(if bias { -7.5 * scale } else { scale })
                    .to_bits()
                    .to_le_bytes()
            }));
        }
        let weight = Weight {
            buffer: d
                .upload(&bytes)
                .expect("Metal projection test GPU operation"),
            ty: affine::AFFINE4,
            k,
            n,
        };
        let sizes: &[usize] = if tiny_only {
            &[1, 4, 8, 12]
        } else {
            &[1, 4, 8, 12, 13, 16, 17, 31, 32, 33, 480, 496, 511, 512]
        };
        for &rows in sizes {
            let input = d
                .upload(
                    &(0..k * rows)
                        .flat_map(|i| (((i * 37) % 1999) as f32 / 113. - 999. / 113.).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .expect("Metal projection test GPU operation");
            let output = d
                .alloc((rows * n + 32) * 4)
                .expect("Metal projection test GPU operation");
            // Tiny prompt rows keep the 512-row contraction, including its
            // padded tensor input. Adaptive decode scratch would be too small.
            let scratch_bytes = affine::workspace_bytes(k, n, rows.max(32));
            let scratch = d
                .alloc(scratch_bytes + 128)
                .expect("Metal projection test GPU operation");
            let mut expected = None;
            for baseline in [true, false] {
                affine::BASELINE_ADMISSION_FOR_TEST.with(|v| v.set(baseline));
                affine::BASELINE_TAIL_FOR_TEST.with(|v| v.set(baseline));
                affine::BASELINE_STAGING_FOR_TEST.with(|v| v.set(baseline));
                unsafe {
                    output.write_u32(&vec![u32::MAX; rows * n + 32]);
                    scratch.write_u32(&vec![u32::MAX; (scratch_bytes + 128) / 4]);
                }
                let cmd = d.begin().expect("Metal projection test GPU operation");
                let cmd = if rows == 512 {
                    cmd.with_affine_prefill_rows(512)
                } else {
                    cmd
                };
                if rows < 13 {
                    affine::project_stable(
                        &cmd,
                        &[(&weight, &output)],
                        &input,
                        rows,
                        &scratch,
                        &[(0, rows, 512)],
                        false,
                    );
                } else {
                    affine::project(&cmd, &[(&weight, &output)], &input, rows, &scratch);
                }
                cmd.finish().expect("Metal projection test GPU operation");
                let actual = unsafe { output.read_f32(0, rows * n + 32) };
                assert!(
                    actual[..rows * n].iter().all(|v| v.is_finite()),
                    "k={k} n={n} rows={rows} baseline={baseline}"
                );
                assert!(actual[rows * n..].iter().all(|v| v.to_bits() == u32::MAX));
                assert!(
                    unsafe { scratch.read_f32(scratch_bytes / 4, 32) }
                        .iter()
                        .all(|v| v.to_bits() == u32::MAX)
                );
                let bits: Vec<_> = actual[..rows * n].iter().map(|v| v.to_bits()).collect();
                if let Some(expected) = &expected {
                    assert!(&bits == expected, "small prefill k={k},n={n},rows={rows}");
                } else {
                    expected = Some(bits);
                }
            }
        }
    }
}

#[test]
fn affine_compact_ffn_is_exact_and_bounded() {
    compact_ffn_control(false);
}

#[test]
#[ignore = "rotated GPU FFN timings, not serving performance"]
fn affine_ragged_ffn_cost() {
    compact_ffn_control(true);
}

fn compact_ffn_control(timed: bool) {
    let d = MetalDevice::new(None).expect("Metal projection test GPU operation");
    if !d.tensor_accelerated() {
        return;
    }
    let weight = |k: usize, n: usize, salt: usize| {
        let mut bytes: Vec<_> = (0..k * n / 8)
            .flat_map(|i| {
                (i as u32)
                    .wrapping_add(salt as u32)
                    .wrapping_mul(2654435761)
                    .to_le_bytes()
            })
            .collect();
        for bias in [false, true] {
            bytes.extend((0..k * n / 64).flat_map(|i| {
                let scale = (i % 251 + 1) as f32 / 9973.;
                // Center the weights so the activation tests both signs instead
                // of saturating every gate to zero under a negative-only fixture.
                half::bf16::from_f32(if bias { -7.5 * scale } else { scale })
                    .to_bits()
                    .to_le_bytes()
            }));
        }
        Weight {
            buffer: d
                .upload(&bytes)
                .expect("Metal projection test GPU operation"),
            ty: affine::AFFINE4,
            k,
            n,
        }
    };
    let (k, n, rows) = (5120, 17408, 512);
    let gate = weight(k, n, 3);
    let up = weight(k, n, 17);
    let down = weight(n, k, 29);
    let input = d
        .upload(
            &(0..rows * k)
                .flat_map(|i| (((i * 37) % 1999) as f32 / 113. - 999. / 113.).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .expect("Metal projection test GPU operation");
    let gate_out = d
        .alloc((rows * n + 32) * 4)
        .expect("Metal projection test GPU operation");
    let up_out = d
        .alloc((rows * n + 32) * 4)
        .expect("Metal projection test GPU operation");
    let output = d
        .alloc((rows * k + 32) * 4)
        .expect("Metal projection test GPU operation");
    let scratch_bytes =
        affine::workspace_bytes(k, n, rows).max(affine::workspace_bytes(n, k, rows));
    let scratch = d
        .alloc(scratch_bytes + 128)
        .expect("Metal projection test GPU operation");
    let weights = [&gate, &up, &down];
    let outputs = [&gate_out, &up_out, &output];
    for rows in [480, 481, 497, 511, 512] {
        let mut expected = Vec::new();
        let mut activation = Vec::new();
        for route in 0..3 {
            let compact = route > 0;
            unsafe {
                gate_out.write_u32(&vec![u32::MAX; rows * n + 32]);
                up_out.write_u32(&vec![u32::MAX; rows * n + 32]);
                output.write_u32(&vec![u32::MAX; rows * k + 32]);
                scratch.write_u32(&vec![u32::MAX; (scratch_bytes + 128) / 4]);
            }
            let cmd = d.begin().expect("Metal projection test GPU operation");
            let cmd = if route == 2 || route == 0 {
                cmd.with_affine_prefill_rows(512)
            } else {
                cmd
            };
            if compact {
                assert!(affine::prefill_ffn(
                    &cmd, weights, &input, outputs, rows, &scratch
                ));
            } else {
                affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(true));
                affine::project(
                    &cmd,
                    &[(weights[0], &gate_out), (weights[1], &up_out)],
                    &input,
                    rows,
                    &scratch,
                );
                cmd.dispatch(
                    "mlx_swiglu",
                    &[&gate_out, &up_out],
                    &[(rows * n) as u32],
                    [(rows * n).div_ceil(256), 1, 1],
                    256,
                );
                affine::project(&cmd, &[(weights[2], &output)], &gate_out, rows, &scratch);
                affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(false));
            }
            cmd.finish().expect("Metal projection test GPU operation");
            let values = unsafe { output.read_f32(0, rows * k + 32) };
            assert!(
                values[..rows * k].iter().all(|v| v.is_finite()),
                "compact route={route}"
            );
            assert!(values[..rows * k].iter().any(|v| *v != 0.));
            assert!(values[rows * k..].iter().all(|v| v.to_bits() == u32::MAX));
            assert!(
                unsafe { scratch.read_f32(scratch_bytes / 4, 32) }
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX)
            );
            let packed = unsafe { scratch.read_f32(0, rows * n / 2) }
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>();
            if compact {
                assert!(
                    values[..rows * k]
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(expected.iter().map(|v: &f32| v.to_bits())),
                    "compact FFN projection changed"
                );
                assert!(packed == activation, "compact activation input changed");
                for buf in [&gate_out, &up_out] {
                    assert!(
                        unsafe { buf.read_f32(rows * n / 2, rows * n / 2 + 32) }
                            .iter()
                            .all(|v| v.to_bits() == u32::MAX)
                    );
                }
            } else {
                expected = values[..rows * k].to_vec();
                activation = packed;
            }
        }
        // The padded down-projection input must be zero, not stale workspace.
        let padding = (rows.next_multiple_of(128) - rows) * n;
        assert!(
            unsafe { scratch.read_f32(rows * n / 2, padding / 2) }
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        if timed {
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for order in 0..2 {
                    let route = (round + order) % 2;
                    let cmd = d
                        .begin()
                        .expect("Metal projection test GPU operation")
                        .with_affine_prefill_rows(512);
                    if route == 0 {
                        affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(true));
                        affine::project(
                            &cmd,
                            &[(weights[0], &gate_out), (weights[1], &up_out)],
                            &input,
                            rows,
                            &scratch,
                        );
                        cmd.dispatch(
                            "mlx_swiglu",
                            &[&gate_out, &up_out],
                            &[(rows * n) as u32],
                            [(rows * n).div_ceil(256), 1, 1],
                            256,
                        );
                        affine::project(&cmd, &[(weights[2], &output)], &gate_out, rows, &scratch);
                        affine::BASELINE_RAGGED_FOR_TEST.with(|v| v.set(false));
                    } else {
                        assert!(affine::prefill_ffn(
                            &cmd, weights, &input, outputs, rows, &scratch
                        ));
                    }
                    times[route]
                        .push(cmd.finish().expect("Metal projection test GPU operation") * 1000.);
                    assert!(
                        unsafe { output.read_f32(0, rows * k) }
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
            eprintln!(
                "RAGGED_FFN_COST {}",
                serde_json::json!({"rows":rows,"gpu_ms":times})
            );
        }
    }
    let cmd = d.begin().expect("Metal projection test GPU operation");
    for rows in [1, 4, 32, 128, 256, 479, 513, 1024] {
        assert!(!affine::prefill_ffn(
            &cmd, weights, &input, outputs, rows, &scratch
        ));
    }
    cmd.finish().expect("Metal projection test GPU operation");
}

#[test]
fn affine_shared_input_projections_are_exact() {
    let d = MetalDevice::new(None).unwrap();
    // Ragged fused-domain seams plus the actual dense-27B DeltaNet planes.
    for (k, ns) in [(512usize, [65, 257, 31, 48]), (5120, [10240, 6144, 48, 48])] {
        let weights: Vec<_> = ns
            .iter()
            .enumerate()
            .map(|(plane, &n)| {
                let mut bytes: Vec<_> = (0..k * n / 8)
                    .flat_map(|i| {
                        ((i + plane * 131) as u32)
                            .wrapping_mul(2654435761)
                            .to_le_bytes()
                    })
                    .collect();
                for bias in [false, true] {
                    bytes.extend((0..k * n / 64).flat_map(|i| {
                        let value = if bias {
                            -((i % 127 + 1) as f32) / 113.
                        } else {
                            (i % 251 + 1) as f32 / 9973.
                        };
                        half::bf16::from_f32(value).to_bits().to_le_bytes()
                    }));
                }
                Weight {
                    buffer: d.upload(&bytes).unwrap(),
                    ty: affine::AFFINE4,
                    k,
                    n,
                }
            })
            .collect();
        for m in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 13, 32, 129, 512] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| (((i * 37) % 1999) as f32 / 113. - 8.).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let scratch_bytes = ns
                .iter()
                .map(|&n| affine::workspace_bytes(k, n, m))
                .max()
                .unwrap();
            let scratch = d.alloc(scratch_bytes + 128).unwrap();
            let outputs: Vec<_> = ns
                .iter()
                .map(|&n| d.alloc((m * n + 32) * 4).unwrap())
                .collect();
            let planes: Vec<_> = weights.iter().zip(&outputs).collect();
            for decode in [None, Some(false), Some(true)] {
                if decode.is_some() && m > 8 {
                    continue;
                }
                let mut reference = Vec::new();
                for shared in [false, true] {
                    unsafe {
                        scratch.write_u32(&vec![u32::MAX; (scratch_bytes + 128) / 4]);
                        for (out, &n) in outputs.iter().zip(&ns) {
                            out.write_u32(&vec![u32::MAX; m * n + 32]);
                        }
                    }
                    let cmd = d.begin().unwrap();
                    let groups = if shared { 4 } else { 2 };
                    for group in planes.chunks(groups) {
                        if let Some(single) = decode {
                            affine::project_verify(&cmd, group, &x, m, &scratch, single);
                        } else {
                            affine::project(&cmd, group, &x, m, &scratch);
                        }
                    }
                    cmd.finish().unwrap();
                    assert!(
                        unsafe { scratch.read_f32(scratch_bytes / 4, 32) }
                            .iter()
                            .all(|v| v.to_bits() == u32::MAX)
                    );
                    for (i, (out, &n)) in outputs.iter().zip(&ns).enumerate() {
                        let got = unsafe { out.read_f32(0, m * n + 32) };
                        assert!(got[..m * n].iter().all(|v| v.is_finite()));
                        assert!(got[m * n..].iter().all(|v| v.to_bits() == u32::MAX));
                        let bits: Vec<_> = got.iter().map(|v| v.to_bits()).collect();
                        if shared {
                            assert!(
                                bits == reference[i],
                                "k={k} m={m} plane={i} decode={decode:?}"
                            );
                        } else {
                            reference.push(bits);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn affine_prefill_padded_layout_is_exact() {
    prefill_layout_gate(false);
}

#[test]
#[ignore = "GPU timing election on full model projection dimensions"]
fn affine_prefill_layout_benchmark() {
    prefill_layout_gate(true);
}

fn prefill_layout_gate(timed: bool) {
    prefill_layout_control(timed, PrefillLayoutProbe::General);
}

#[test]
#[ignore = "GPU timing of small admission projections"]
fn affine_small_admission_benchmark() {
    prefill_layout_control(true, PrefillLayoutProbe::Admission);
}

#[test]
#[ignore = "GPU timing of the dense-27B mixer output projection"]
fn affine_mixer_output_benchmark() {
    prefill_layout_control(true, PrefillLayoutProbe::MixerOutput);
}

#[derive(Clone, Copy, PartialEq)]
enum PrefillLayoutProbe {
    General,
    Admission,
    MixerOutput,
    WideStaging,
    Gemma,
}

#[test]
#[ignore = "Gemma MLX shape election; exact GPU outputs and paired timings"]
fn affine_gemma_prefill_benchmark() {
    prefill_layout_control(true, PrefillLayoutProbe::Gemma);
}

#[test]
#[ignore = "GPU timing of bulk staging; exact outputs required"]
fn affine_wide_staging_benchmark() {
    prefill_layout_control(true, PrefillLayoutProbe::WideStaging);
}

fn prefill_layout_control(timed: bool, probe: PrefillLayoutProbe) {
    let kernels = [
        ("mlx_affine_tile64", 64, 32),
        ("mlx_affine_prefill64", 64, 32),
        ("mlx_affine_prefill_load32", 64, 32),
        ("mlx_affine_prefill_rows256", 256, 32),
        ("mlx_affine_prefill_store256", 256, 32),
        ("mlx_affine_prefill_wide128", 128, 32),
        ("mlx_affine_prefill_deep128", 128, 32),
    ];
    let device = MetalDevice::new(None).expect("Metal device for prefill layout check");
    if probe == PrefillLayoutProbe::WideStaging && !device.tensor_accelerated() {
        eprintln!("wide-staging diagnostic requires Apple10/M5");
        return;
    }
    let kernels = &kernels[..if device.tensor_accelerated() {
        kernels.len()
    } else {
        4
    }];
    let small_kernels = [
        ("mlx_affine_tile32", 32, 32),
        ("mlx_affine_prefill_load32_m32", 32, 32),
        ("mlx_affine_prefill_load32", 64, 32),
    ];
    let kernels = if probe == PrefillLayoutProbe::Admission {
        &small_kernels[..]
    } else {
        kernels
    };
    let wide_kernels = [
        ("mlx_affine_prefill_store256", 256, 32),
        ("mlx_affine_prefill_wide128", 128, 32),
        ("mlx_affine_prefill_deep128", 128, 32),
    ];
    let kernels = if probe == PrefillLayoutProbe::WideStaging {
        &wide_kernels[..]
    } else {
        kernels
    };
    let shapes = if probe == PrefillLayoutProbe::Gemma {
        vec![
            (5376, vec![21504, 21504]),
            (21504, vec![5376]),
            (5376, vec![8192, 4096, 4096]),
            (8192, vec![5376]),
            (5376, vec![16384, 2048]),
            (16384, vec![5376]),
        ]
    } else if probe == PrefillLayoutProbe::MixerOutput {
        vec![(6144, vec![5120])]
    } else if timed {
        vec![
            (5120, vec![17408, 17408]),
            (17408, vec![5120]),
            (5120, vec![12288, 1024, 1024]),
            (5120, vec![10240, 6144]),
            (6144, vec![5120]),
            (512, vec![257, 65, 31]),
            (512, vec![512, 1024, 32]),
        ]
    } else {
        vec![
            (64, vec![1, 31, 33]),
            (512, vec![512, 1024, 32]),
            (512, vec![257, 65, 31]),
            (5120, vec![257, 65, 31]),
            (17408, vec![33]),
        ]
    };
    for (k, ns) in shapes {
        let weights: Vec<_> = ns
            .iter()
            .map(|&n| {
                let mut bytes: Vec<_> = (0..k * n / 8)
                    .flat_map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes())
                    .collect();
                for bias in [false, true] {
                    bytes.extend((0..k * n / 64).flat_map(|i| {
                        let v = if bias {
                            -((i % 499 + 1) as f32) / 113.
                        } else {
                            (i % 997 + 1) as f32 / 9973.
                        };
                        half::bf16::from_f32(v).to_bits().to_le_bytes()
                    }));
                }
                Weight {
                    buffer: device.upload(&bytes).expect("upload affine test weights"),
                    ty: affine::AFFINE4,
                    k,
                    n,
                }
            })
            .collect();
        let rows: &[usize] = if probe == PrefillLayoutProbe::Gemma {
            &[128, 512]
        } else if probe == PrefillLayoutProbe::MixerOutput
            || probe == PrefillLayoutProbe::WideStaging
        {
            &[512]
        } else if probe == PrefillLayoutProbe::Admission {
            &[13, 32]
        } else if timed {
            &[128, 256, 512, 1024]
        } else {
            &[
                33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 513,
            ]
        };
        for &m in rows {
            let input = device
                .upload(
                    &(0..m * k)
                        .flat_map(|i| (((i * 37) % 1999) as f32 / 113. - 8.).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .expect("upload prefill test input");
            let elements = k.div_ceil(128) * 128 * m.div_ceil(128) * 128;
            let scratch = device
                .alloc(elements * 2)
                .expect("allocate prefill scratch");
            let outputs: Vec<_> = ns
                .iter()
                .map(|&n| {
                    device
                        .upload(
                            &(0..m * n + 32)
                                .flat_map(|_| f32::NAN.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .expect("upload poisoned prefill output")
                })
                .collect();
            let cmd = device.begin().expect("begin prefill input conversion");
            cmd.dispatch(
                "mlx_input",
                &[&input, &scratch],
                &[k as u32, m as u32],
                [elements.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().expect("finish prefill input conversion");
            let dispatch = |cmd: &Commands<'_>, route: usize| {
                let (kernel, tile, columns) = kernels[route];
                let second = weights.get(1).unwrap_or(&weights[0]);
                let third = weights.get(2).unwrap_or(second);
                let o1 = outputs.get(1).unwrap_or(&outputs[0]);
                let o2 = outputs.get(2).unwrap_or(o1);
                cmd.dispatch(
                    kernel,
                    &[
                        &weights[0].buffer,
                        &second.buffer,
                        &third.buffer,
                        &scratch,
                        &outputs[0],
                        o1,
                        o2,
                    ],
                    &[
                        k as u32,
                        ns[0] as u32,
                        ns.get(1).copied().unwrap_or(0) as u32,
                        ns.get(2).copied().unwrap_or(0) as u32,
                        m as u32,
                    ],
                    [
                        ns.iter().map(|n| n.div_ceil(columns)).sum::<usize>(),
                        m.div_ceil(tile),
                        1,
                    ],
                    128,
                );
            };
            let mut reference = Vec::new();
            for (route, &(kernel, _, _)) in kernels.iter().enumerate() {
                // Bulk routes consume whole K tiles; ragged K retains the
                // existing bounded route rather than reading a zero-weight tail.
                if (kernel == "mlx_affine_prefill_wide128" && k % 256 != 0)
                    || (kernel == "mlx_affine_prefill_deep128" && k % 384 != 0)
                {
                    continue;
                }
                // Independently poison every candidate; a missing store must fail.
                for (o, &n) in outputs.iter().zip(&ns) {
                    unsafe {
                        o.write_u32(&vec![u32::MAX; m * n + 32]);
                    }
                }
                let cmd = device.begin().expect("begin prefill candidate dispatch");
                dispatch(&cmd, route);
                cmd.finish().expect("finish prefill candidate dispatch");
                for (i, (o, &n)) in outputs.iter().zip(&ns).enumerate() {
                    let actual = unsafe { o.read_f32(0, m * n + 32) };
                    assert!(
                        actual[..m * n].iter().all(|x| x.is_finite()),
                        "nonfinite k={k} n={n} m={m} kernel={kernel}"
                    );
                    assert!(
                        actual[..m * n].iter().all(|x| x.to_bits() & 0xffff == 0),
                        "unrounded BF16 boundary: k={k},n={n},m={m},route={route}"
                    );
                    assert!(actual[m * n..].iter().all(|x| x.is_nan()));
                    if route == 0 {
                        reference.push(actual[..m * n].to_vec());
                    } else {
                        let differences: Vec<_> = actual[..m * n]
                            .iter()
                            .zip(&reference[i])
                            .enumerate()
                            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
                            .take(4)
                            .collect();
                        assert!(
                            differences.is_empty(),
                            "k={k},n={n},m={m},route={route}: {differences:?}"
                        );
                    }
                }
            }
            if !timed {
                continue;
            }
            let run = |route| {
                let cmd = device.begin().expect("begin timed prefill dispatch");
                for _ in 0..8 {
                    dispatch(&cmd, route);
                }
                cmd.finish().expect("finish timed prefill dispatch") / 8.0
            };
            let warm = std::time::Instant::now();
            while warm.elapsed().as_millis() < 200 {
                for (route, &(kernel, _, _)) in kernels.iter().enumerate() {
                    if (kernel == "mlx_affine_prefill_wide128" && k % 256 != 0)
                        || (kernel == "mlx_affine_prefill_deep128" && k % 384 != 0)
                    {
                        continue;
                    }
                    run(route);
                }
            }
            let mut times = vec![Vec::new(); kernels.len()];
            for round in 0..9 {
                for i in 0..kernels.len() {
                    let route = (round + i) % kernels.len();
                    if (kernels[route].0 == "mlx_affine_prefill_wide128" && k % 256 != 0)
                        || (kernels[route].0 == "mlx_affine_prefill_deep128" && k % 384 != 0)
                    {
                        continue;
                    }
                    times[route].push(run(route) * 1e6);
                }
            }
            let raw = times.clone();
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "PREFILL_LAYOUT {}",
                serde_json::json!({"k":k,"ns":ns,"m":m,"kernels":kernels,"median_us":times.iter().map(|t|t.get(4)).collect::<Vec<_>>(),"samples_us":raw})
            );
        }
    }
}

// Pre-election and F32-input arms are diagnostic only. There is no serving
// environment switch. The real dispatcher always uses the elected BF16 path.
fn diagnostic(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, m: usize, stream: bool) {
    assert!((2..=12).contains(&m));
    let rows = if stream {
        m.div_ceil(m.div_ceil(5))
    } else {
        m.min(5)
    };
    let name = format!("mlx_affine{}{rows}", if stream { "_stream" } else { "" });
    cmd.dispatch(
        &name,
        &[&w.buffer, &w.buffer, &w.buffer, x, y, y, y],
        &[w.k as u32, w.n as u32, 0, 0, m as u32],
        [w.n.div_ceil(8), m.div_ceil(rows), 1],
        if stream { 64 } else { 128 },
    );
}

fn check_output(output: &Buffer, fixture: &SafetensorsFile, m: usize, n: usize, label: &str) {
    // SAFETY: every caller completed its GPU command before reading.
    let actual = unsafe { output.read_f32(0, m * n + 32) };
    assert!(
        actual[m * n..].iter().all(|v| v.is_nan()),
        "{label}: output guard"
    );
    let (info, bytes) = fixture.bytes("y").expect("GPU fixture output");
    assert_eq!(info.dtype, StDtype::F32);
    assert_eq!(info.shape, [m, n]);
    let expected = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("four-byte F32 fixture value")));
    let mut max_error = 0f32;
    let mut max_value = 0f32;
    let mut unequal = 0;
    for (a, b) in actual[..m * n].iter().zip(expected) {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}: unwritten/nonfinite output"
        );
        assert_eq!(a.to_bits() & 0xffff, 0, "{label}: output is not BF16");
        max_error = max_error.max((a - b).abs());
        max_value = max_value.max(b.abs());
        unequal += usize::from(a.to_bits() != b.to_bits());
    }
    eprintln!(
        "{label} m={m} max_error={max_error} max_value={max_value} unequal={unequal}/{}",
        m * n
    );
    // The narrow projection caught a real BF16-reduction error. Require exact
    // GPU-oracle equality there; do not hide it behind a relative tolerance.
    if n == 48 {
        assert_eq!(unequal, 0, "{label}: split-K arithmetic regression");
    } else {
        // Same pre-existing format/layout bound. Complete generation parity
        // is separate: passing this does not qualify changed token choices.
        assert!(
            max_error <= max_value * 0.008 + 0.0001,
            "{label}: projection mismatch"
        );
    }
}

#[test]
#[ignore = "requires reference fixtures in PADDOCK_MLX_FIXTURES; GPU oracle only"]
fn real_checkpoint_projections_match_mlx_gpu() {
    let dir = std::env::var("PADDOCK_MLX_FIXTURES").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let device = MetalDevice::new(None).unwrap();
    let scratch = device.alloc(32 << 20).unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let (k, n, m) = (
            case["k"].as_u64().unwrap() as usize,
            case["n"].as_u64().unwrap() as usize,
            case["rows"].as_u64().unwrap() as usize,
        );
        let fixture = SafetensorsFile::open(&dir.join(case["file"].as_str().unwrap())).unwrap();
        let (info, bytes) = fixture.bytes("x").unwrap();
        assert_eq!(info.dtype, StDtype::F32);
        assert_eq!(info.shape, [m, k]);
        let x = device.upload(bytes).unwrap();
        let w = affine::load(&device, &source, case["weight"].as_str().unwrap(), k, n).unwrap();
        let poison: Vec<_> = (0..m * n + 32)
            .flat_map(|_| f32::NAN.to_le_bytes())
            .collect();
        let output = device.upload(&poison).unwrap();
        let cmd = device.begin().unwrap();
        let cmd = if m == 512 {
            cmd.with_affine_prefill_rows(512)
        } else {
            cmd
        };
        affine::project(&cmd, &[(&w, &output)], &x, m, &scratch);
        cmd.finish().unwrap();
        check_output(&output, &fixture, m, n, case["weight"].as_str().unwrap());
    }

    // Unequal plane widths exercise fused-domain seams and ragged vector
    // tiles. The three fixture inputs are identical GPU-generated values.
    for m in [4, 7, 13, 129, 512] {
        let cases: Vec<_> = manifest["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["rows"] == m && c["k"] == 5120 && c["n"] != 248320)
            .collect();
        assert_eq!(cases.len(), 3);
        let fixtures: Vec<_> = cases
            .iter()
            .map(|c| SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap())
            .collect();
        let x = device.upload(fixtures[0].bytes("x").unwrap().1).unwrap();
        let weights: Vec<_> = cases
            .iter()
            .map(|c| {
                affine::load(
                    &device,
                    &source,
                    c["weight"].as_str().unwrap(),
                    5120,
                    c["n"].as_u64().unwrap() as usize,
                )
                .unwrap()
            })
            .collect();
        let outputs: Vec<_> = weights
            .iter()
            .map(|w| {
                device
                    .upload(
                        &(0..m as usize * w.n + 32)
                            .flat_map(|_| f32::NAN.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
            })
            .collect();
        let planes: Vec<_> = weights.iter().zip(&outputs).collect();
        let cmd = device.begin().unwrap();
        let cmd = if m == 512 {
            cmd.with_affine_prefill_rows(512)
        } else {
            cmd
        };
        affine::project(&cmd, &planes, &x, m as usize, &scratch);
        cmd.finish().unwrap();
        for ((w, out), fixture) in planes.iter().zip(&fixtures) {
            check_output(out, fixture, m as usize, w.n, "fused");
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_MLX_FIXTURES; reports shape sensitivity, NOT a parity pass"]
fn real_checkpoint_batch_shape_diagnostic() {
    let dir = std::env::var("PADDOCK_MLX_FIXTURES").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let device = MetalDevice::new(None).unwrap();
    for case in manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["rows"] == 512)
    {
        let k = case["k"].as_u64().unwrap() as usize;
        let n = case["n"].as_u64().unwrap() as usize;
        let fixture = SafetensorsFile::open(&dir.join(case["file"].as_str().unwrap())).unwrap();
        let first = &fixture.bytes("x").unwrap().1[..k * 4];
        let x = device.upload(&first.repeat(512)).unwrap();
        let w = affine::load(&device, &source, case["weight"].as_str().unwrap(), k, n).unwrap();
        let scratch = device.alloc(affine::workspace_bytes(k, n, 512)).unwrap();
        let out = device.alloc(512 * n * 4).unwrap();
        let mut reference = Vec::new();
        for rows in [1, 4, 32, 128, 512] {
            let cmd = device.begin().unwrap();
            affine::project(&cmd, &[(&w, &out)], &x, rows, &scratch);
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, n) };
            assert!(got.iter().all(|v| v.is_finite()));
            if rows == 1 {
                reference = got.clone();
            }
            let unequal = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
            let max_abs = got
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "AFFINE_BATCH_SHAPE {}",
                serde_json::json!({
                    "weight": case["weight"], "k": k, "n": n, "rows": rows,
                    "unequal_to_single": unequal, "max_abs": max_abs
                })
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_MLX_FIXTURES; exact column verifier and warm GPU timing"]
fn affine_copy_verifier_election() {
    let dir = std::env::var("PADDOCK_MLX_FIXTURES").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let device = MetalDevice::new(None).unwrap();
    for case in manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["rows"] == 512 || (c["rows"] == 4 && c["n"] == 248320))
    {
        let k = case["k"].as_u64().unwrap() as usize;
        let n = case["n"].as_u64().unwrap() as usize;
        let fixture = SafetensorsFile::open(&dir.join(case["file"].as_str().unwrap())).unwrap();
        let bytes = fixture.bytes("x").unwrap().1;
        let x = device
            .upload(&bytes.repeat(if case["rows"] == 4 { 2 } else { 1 }))
            .unwrap();
        let w = affine::load(&device, &source, case["weight"].as_str().unwrap(), k, n).unwrap();
        let bias = device.alloc(6 * k / 16 * 4).unwrap();
        let out = device.alloc((6 * n + 32) * 4).unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "mlx_affine_bias",
            &[&x, &bias],
            &[k as u32, 6],
            [(6 * k / 16).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        for m in [2usize, 3, 4, 5, 6] {
            let group = if m <= 4 { m } else { 3 };
            let routes = if group == 4 { 2 } else { 3 };
            let dispatch = |cmd: &Commands<'_>, route: usize| {
                let kernel = format!(
                    "mlx_affine_verify_{}{group}",
                    ["single", "narrow", "half"][route]
                );
                cmd.dispatch(
                    &kernel,
                    &[&w.buffer, &w.buffer, &w.buffer, &x, &out, &out, &out, &bias],
                    &[k as u32, n as u32, 0, 0, m as u32],
                    [n.div_ceil([16, 4, 8][route]), m.div_ceil(group), 1],
                    128,
                );
            };
            let mut reference = Vec::new();
            for route in 0..routes {
                unsafe {
                    out.write_u32(&vec![u32::MAX; m * n + 32]);
                }
                let cmd = device.begin().unwrap();
                dispatch(&cmd, route);
                cmd.finish().unwrap();
                let got = unsafe { out.read_f32(0, m * n + 32) };
                assert!(got[..m * n].iter().all(|v| v.is_finite()));
                assert!(got[m * n..].iter().all(|v| v.is_nan()));
                if route != 0 {
                    let unequal = got[..m * n]
                        .iter()
                        .zip(&reference)
                        .filter(|(a, b)| a != b)
                        .count();
                    assert_eq!(unequal, 0, "column verify K={k} N={n} M={m} route={route}");
                } else {
                    reference = got[..m * n].to_vec();
                }
            }
            let run = |route| {
                let cmd = device.begin().unwrap();
                for _ in 0..16 {
                    dispatch(&cmd, route);
                }
                cmd.finish().unwrap() / 16.
            };
            for _ in 0..8 {
                for route in 0..routes {
                    run(route);
                }
            }
            let mut times = vec![Vec::new(); routes];
            for round in 0..9 {
                for i in 0..routes {
                    let route = (round + i) % routes;
                    times[route].push(run(route) * 1e6);
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "COPY_VERIFIER {}",
                serde_json::json!({"weight":case["weight"],"k":k,"n":n,"rows":m,"median_us":times.iter().map(|t|t[4]).collect::<Vec<_>>(),"samples_us":times})
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_MLX_FIXTURES; real-weight row-invariant contraction and warm GPU timings, not HTTP qualification"]
fn affine_fast_contract_probe() {
    let dir = std::env::var("PADDOCK_MLX_FIXTURES").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let device = MetalDevice::new(None).unwrap();
    for case in manifest["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["rows"] == 512 || (c["rows"] == 4 && c["n"] == 248320))
    {
        let k = case["k"].as_u64().unwrap() as usize;
        let n = case["n"].as_u64().unwrap() as usize;
        let fixture = SafetensorsFile::open(&dir.join(case["file"].as_str().unwrap())).unwrap();
        let first = &fixture.bytes("x").unwrap().1[..k * 4];
        let x = device.upload(&first.repeat(12)).unwrap();
        let w = affine::load(&device, &source, case["weight"].as_str().unwrap(), k, n).unwrap();
        let compact = device.alloc(12 * k * 2).unwrap();
        let out = device.alloc((12 * n + 32) * 4).unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "mlx_input_compact",
            &[&x, &compact],
            &[k as u32, 12],
            [(12 * k).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let mut reference = Vec::new();
        for m in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12] {
            let r = m.div_ceil(m.div_ceil(5));
            let dispatch = |cmd: &Commands<'_>| {
                cmd.dispatch(
                    &format!("mlx_affine_stable{r}"),
                    &[&w.buffer, &w.buffer, &w.buffer, &compact, &out, &out, &out],
                    &[k as u32, n as u32, 0, 0, m as u32],
                    [n.div_ceil(8), m.div_ceil(r), 1],
                    64,
                );
            };
            unsafe {
                out.write_u32(&vec![u32::MAX; m * n + 32]);
            }
            let cmd = device.begin().unwrap();
            dispatch(&cmd);
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, m * n + 32) };
            assert!(got[..m * n].iter().all(|v| v.is_finite()));
            assert!(got[m * n..].iter().all(|v| v.to_bits() == u32::MAX));
            if m == 1 {
                reference = got[..n].to_vec();
            }
            for (row, values) in got[..m * n].chunks(n).enumerate() {
                let unequal = values
                    .iter()
                    .zip(&reference)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(unequal, 0, "fast contraction K={k} N={n} M={m} row={row}");
            }
            let mut times = Vec::new();
            for round in 0..12 {
                let cmd = device.begin().unwrap();
                for _ in 0..8 {
                    dispatch(&cmd);
                }
                let time = cmd.finish().unwrap() / 8.;
                if round >= 3 {
                    times.push(time * 1e6);
                }
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "FAST_CONTRACT {}",
                serde_json::json!({"k":k,"n":n,"rows":m,"median_us":times[4]})
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_MLX_FIXTURES; warm-cache GPU diagnostic, NOT serving timing"]
fn affine_projection_election() {
    let dir = std::env::var("PADDOCK_MLX_FIXTURES").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let device = MetalDevice::new(None).unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let m = case["rows"].as_u64().unwrap() as usize;
        if ![2, 3, 4, 5, 8, 12].contains(&m) {
            continue;
        }
        let k = case["k"].as_u64().unwrap() as usize;
        let n = case["n"].as_u64().unwrap() as usize;
        let fixture = SafetensorsFile::open(&dir.join(case["file"].as_str().unwrap())).unwrap();
        let x = device.upload(fixture.bytes("x").unwrap().1).unwrap();
        let w = affine::load(&device, &source, case["weight"].as_str().unwrap(), k, n).unwrap();
        let y = device.alloc(m * n * 4).unwrap();
        let scratch = device.alloc(32 << 20).unwrap();
        let run = |route: usize| {
            let cmd = device.begin().unwrap();
            for _ in 0..16 {
                if route == 2 {
                    affine::project(&cmd, &[(&w, &y)], &x, m, &scratch);
                } else {
                    diagnostic(&cmd, &w, &x, &y, m, route == 1);
                }
            }
            cmd.finish().unwrap() / 16.0
        };
        let warm = std::time::Instant::now();
        while warm.elapsed().as_millis() < 200 {
            run(0);
            run(1);
            run(2);
        }
        let mut times = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..7 {
            for i in 0..3 {
                let route = (i + round) % 3;
                times[route].push(run(route) * 1e6);
            }
        }
        for row in &mut times {
            row.sort_by(f64::total_cmp);
        }
        eprintln!(
            "AFFINE_ELECTION {}",
            serde_json::json!({
            "weight":case["weight"],"m":m,"k":k,"n":n,"baseline_us":times[0][3],
            "stream_f32_us":times[1][3],"serving_bf16_us":times[2][3],
            "speedup":times[0][3]/times[2][3],"samples_us":times})
        );
    }
}
