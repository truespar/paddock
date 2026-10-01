use super::affine;
use crate::device::MetalDevice;
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors};
use std::path::Path;

fn poison_output(y: &crate::device::Buffer, count: usize) {
    // Previous command completed. An unwritten candidate element must not
    // inherit a passing value from the baseline it is compared against.
    unsafe { y.write_u32(&vec![f32::NAN.to_bits(); count + 32]) };
}

fn dense_grid(kernel: &str, n: usize, rows: usize) -> [usize; 3] {
    [
        n.div_ceil(if kernel.contains("_wide") { 64 } else { 32 }),
        rows.div_ceil(32),
        1,
    ]
}

#[test]
fn hc_combine_norm_preserves_residual_norm_and_guards() {
    hc_combine_norm_cases(false);
}

#[test]
#[ignore = "rotated GPU costs for fused residual update and normalization"]
fn hc_combine_norm_execution_cost() {
    hc_combine_norm_cases(true);
}

fn hc_combine_norm_cases(measure: bool) {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    let values = |n, salt| {
        (0..n)
            .flat_map(|i| {
                half::bf16::from_f32(((i * 29 + salt) % 137) as f32 / 97. - 0.7)
                    .to_f32()
                    .to_le_bytes()
            })
            .collect::<Vec<_>>()
    };
    let w = d.upload(&values(10240, 71)).unwrap();
    for rows in [1usize, 4, 8, 13, 257, 2048] {
        let input = values(rows * 10240 + 32, 19);
        let h = d.upload(&input).unwrap();
        let delta = d.upload(&values(rows * 2560, 37)).unwrap();
        let inject = d.upload(&values(rows * 4, 67)).unwrap();
        let norm = d.alloc((rows * 10240 + 32) * 4).unwrap();
        let mut expected: Option<(Vec<u32>, Vec<u32>)> = None;
        let mut times: [Vec<f64>; 2] = Default::default();
        for round in 0..if measure { 9 } else { 1 } {
            for index in 0..2 {
                let route = (round + index) % 2;
                unsafe {
                    h.write_u32(
                        &input
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| u32::from_le_bytes(*b))
                            .collect::<Vec<_>>(),
                    );
                }
                poison_output(&norm, rows * 10240);
                let cmd = d.begin().unwrap();
                for _ in 0..if measure { 16 } else { 1 } {
                    if route == 0 {
                        cmd.dispatch(
                            "q4b_hc_combine",
                            &[&h, &delta, &inject],
                            &[2560, rows as u32],
                            [(rows * 10240).div_ceil(256), 1, 1],
                            256,
                        );
                        cmd.dispatch(
                            "q4b_norm",
                            &[&h, &w, &norm],
                            &[2560, 4, 1e-6f32.to_bits()],
                            [4, rows, 1],
                            256,
                        );
                    } else {
                        cmd.dispatch(
                            "q4b_hc_combine_norm",
                            &[&h, &delta, &inject, &w, &norm],
                            &[2560, 4, 1e-6f32.to_bits()],
                            [4, rows, 1],
                            256,
                        );
                    }
                }
                let gpu = cmd.finish().unwrap();
                let output = unsafe { norm.read_f32(0, rows * 10240 + 32) };
                assert!(output[..rows * 10240].iter().all(|v| v.is_finite()));
                assert!(output[rows * 10240..].iter().all(|v| v.is_nan()));
                let residual = unsafe { h.read_u32(rows * 10240 + 32) };
                assert!(
                    residual[rows * 10240..]
                        .iter()
                        .zip(input[rows * 10240 * 4..].as_chunks::<4>().0)
                        .all(|(&a, b)| a == u32::from_le_bytes(*b))
                );
                let got = (
                    residual,
                    output.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                );
                if let Some(want) = &expected {
                    assert!(
                        &got == want,
                        "combine/norm rows={rows} round={round} route={route}"
                    );
                } else {
                    expected = Some(got);
                }
                if round > 0 {
                    times[route].push(gpu / 16.);
                }
            }
        }
        if measure {
            eprintln!(
                "FLASH_HC_COMBINE_NORM {}",
                serde_json::json!({"rows":rows,"gpu_seconds":times,"exact":true})
            );
        }
    }
}

#[test]
fn expert_prefill_variants_preserve_bits_masks_and_tail_guards() {
    expert_prefill_variant_cases(false);
}

#[test]
#[ignore = "checkpoint-backed expert staging exactness and rotated GPU costs"]
fn expert_prefill_variants_execution_cost() {
    expert_prefill_variant_cases(true);
}

fn expert_prefill_variant_cases(measure: bool) {
    let d = MetalDevice::new(Some(2 << 30)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let w = if measure {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
        let source = super::mlx::Source::open(Path::new(&path)).unwrap();
        source
            .weight(
                &d,
                &format!("{}.layers.0.mlp.switch_mlp.gate_proj", super::mlx::ROOT),
            )
            .unwrap()
    } else {
        let (k, n) = (256, 80 * 512);
        let mut raw = (0..k * n / 2)
            .map(|i| (i * 37 + i / 137 + 19) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / 32).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            k,
            n,
            ty: affine::A4G32,
        }
    };
    let (k, n) = (w.k, w.n / 512);
    let kernels = [
        "q4a_expert_mm_group32_packed",
        "q4a_expert_mm_k128_n32",
        "q4a_expert_mm_k128_n64",
        "q4a_expert_mm_load4",
        "q4a_expert_mm_load8",
        "q4a_expert_mm_sg1",
        "q4a_expert_mm_sg2",
        "q4a_expert_mm_pad4",
        "q4a_expert_mm_pad16",
        "q4a_expert_mm_register",
        "q4a_expert_mm_masked",
        "q4a_expert_mm_rows64",
    ];
    let mut failures = Vec::new();
    for rows in if measure {
        vec![205, 512, 1024, 2048]
    } else {
        vec![1, 9, 33, 205, 2048]
    } {
        for concentrated in [false, true] {
            let entries = rows * 10;
            let ids = d
                .upload(
                    &(0..entries)
                        .flat_map(|i| {
                            (if concentrated && i / 10 % 2 == 0 {
                                i % 10
                            } else {
                                (i / 10 * 37 + i % 10 * 17) % 512
                            } as u32)
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let lists = d.alloc(512 * entries * 4).unwrap();
            let counts = d.alloc(512 * 4).unwrap();
            let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
            let wide_tiles = d.alloc(tiles.len()).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "moe_align",
                &[&ids, &lists, &counts],
                &[entries as u32],
                [512, 1, 1],
                256,
            );
            cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
            cmd.dispatch(
                "q4a_expert_tiles64",
                &[&counts, &wide_tiles],
                &[512],
                [1, 1, 1],
                512,
            );
            cmd.finish().unwrap();
            for per_entry in if measure {
                vec![false]
            } else {
                vec![false, true]
            } {
                let input_rows = if per_entry { entries } else { rows };
                let x = d
                    .upload(
                        &(0..input_rows * k)
                            .flat_map(|i| (((i * 29 % 137) as f32 - 68.) / 97.).to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let packed = d.alloc(input_rows.next_multiple_of(32) * k * 2).unwrap();
                let y = d.alloc((entries * n + 32) * 4).unwrap();
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "q4a_input",
                    &[&x, &packed],
                    &[k as u32, n as u32, input_rows as u32, 0, 0, 0, 0],
                    [(input_rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                    256,
                );
                cmd.finish().unwrap();
                let mut p = vec![
                    k as u32,
                    n as u32,
                    entries as u32,
                    u32::from(per_entry),
                    512,
                ];
                p.extend([u32::MAX; affine::MAX_ROWS.div_ceil(32)]);
                for row in (0..rows).step_by(7) {
                    p[5 + row / 32] &= !(1 << (row % 32));
                }
                let mut expected: Option<Vec<u32>> = None;
                let mut times: [Vec<f64>; 12] = Default::default();
                for round in 0..if measure { 7 } else { 1 } {
                    for index in 0..kernels.len() {
                        let route = (round + index) % kernels.len();
                        poison_output(&y, entries * n);
                        let cmd = d.begin().unwrap();
                        cmd.dispatch(
                            kernels[route],
                            &[
                                &w.buffer,
                                &packed,
                                &lists,
                                &counts,
                                if route == 11 { &wide_tiles } else { &tiles },
                                &y,
                            ],
                            &p,
                            [
                                n.div_ceil(if route == 1 { 32 } else { 64 }),
                                entries.div_ceil(32) + 512,
                                1,
                            ],
                            if route == 5 {
                                32
                            } else if route == 6 {
                                64
                            } else {
                                128
                            },
                        );
                        let gpu = cmd.finish().unwrap();
                        let got = unsafe { y.read_f32(0, entries * n + 32) };
                        assert!(got[entries * n..].iter().all(|v| v.is_nan()));
                        for row in 0..rows {
                            let span = &got[row * 10 * n..(row + 1) * 10 * n];
                            assert!(span.iter().all(|v| if row % 7 == 0 {
                                v.is_nan()
                            } else {
                                v.is_finite()
                            }));
                        }
                        let got = got[..entries * n]
                            .iter()
                            .map(|x| x.to_bits())
                            .collect::<Vec<_>>();
                        if let Some(ref expected) = expected {
                            let unequal = got.iter().zip(expected).filter(|(a, b)| a != b).count();
                            if unequal != 0 {
                                failures.push((
                                    rows,
                                    concentrated,
                                    per_entry,
                                    round,
                                    route,
                                    unequal,
                                ));
                            }
                        } else {
                            expected = Some(got);
                        }
                        if round > 0 {
                            times[route].push(gpu);
                        }
                    }
                }
                if measure {
                    eprintln!(
                        "FLASH_EXPERT_PREFILL_VARIANTS {}",
                        serde_json::json!({"rows": rows, "concentrated": concentrated, "per_entry": per_entry, "kernels": kernels, "gpu_seconds": times})
                    );
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "expert staging changed output bits: {failures:?}"
    );
}

#[test]
fn small_prompt_packed_wide_preserves_bits_and_offset_guards() {
    small_prompt_wide_cases(false);
}

#[test]
#[ignore = "exact shared-unpacking small-prompt costs; not a serving benchmark"]
fn small_prompt_wide_rows_execution_cost() {
    small_prompt_wide_cases(true);
}

#[test]
fn wide_row_reuse_keeps_parallelism_and_covers_partial_groups() {
    for (rows, columns, expected) in [
        (1, 10240, 1),
        (2, 10240, 2),
        (3, 10240, 2),
        (4, 10240, 4),
        (4, 320, 1),
        (4, 640, 2),
        (9, 320, 2),
        (9, 640, 4),
        (12, 320, 2),
        (21, 320, 4),
        (49, 4, 1),
    ] {
        assert_eq!(affine::wide_row_group(rows, columns), expected);
    }
    for rows in 1..=2048_usize {
        for columns in [4, 13, 320, 512, 640, 2560, 10240] {
            let width = affine::wide_row_group(rows, columns);
            assert!([1, 2, 4].contains(&width));
            if width > 1 {
                assert!((rows / width) * columns.div_ceil(8) >= 128);
            }
            let covered = (0..rows.div_ceil(width))
                .flat_map(|group| group * width..((group + 1) * width).min(rows))
                .collect::<Vec<_>>();
            assert_eq!(covered, (0..rows).collect::<Vec<_>>());
        }
    }
}

fn small_prompt_wide_cases(measure: bool) {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    for (k, n, ty) in [
        (2560, 640, affine::A4G32),
        (10240, 320, affine::A4G32),
        (320, 10240, affine::A4G32),
        (2560, 512, affine::A8G64),
        (2560, 10240, affine::A4G32),
        (6144, 2560, affine::A4G32),
        (64, 4, affine::A4G32),
        (64, 13, affine::A4G32),
        (64, 13, affine::A8G64),
    ] {
        let (bits, group) = affine::format(ty);
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + i / 137 + 19) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / group).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        let w = d.upload(&raw).unwrap();
        for (rows, start) in [
            (1, 0),
            (2, 3),
            (3, 1),
            (4, 3),
            (5, 1),
            (7, 3),
            (8, 0),
            (9, 14),
            (11, 0),
            (12, 37),
            (13, 7),
            (21, 14),
            (49, 7),
        ] {
            if measure && (rows < 2 || n < 64) {
                continue;
            }
            let input = (0..(start + rows) * k)
                .flat_map(|i| {
                    // The model's small-prompt projection contract is BF16
                    // activations stored in F32, not arbitrary F32 inputs.
                    half::bf16::from_f32((i % 137) as f32 / 97. - 0.7)
                        .to_f32()
                        .to_le_bytes()
                })
                .collect::<Vec<_>>();
            let x = d.upload(&input).unwrap();
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let mut expected = None;
            let mut kernels = vec![
                ("q4a_wide", 1),
                (
                    if bits == 4 {
                        "q4a_wide4_packed"
                    } else {
                        "q4a_wide8_packed"
                    },
                    1,
                ),
                (
                    if bits == 4 {
                        "q4a_wide4_rows2"
                    } else {
                        "q4a_wide8_rows2"
                    },
                    2,
                ),
                (
                    if bits == 4 {
                        "q4a_wide4_rows4"
                    } else {
                        "q4a_wide8_rows4"
                    },
                    4,
                ),
            ];
            if measure {
                kernels.remove(0);
            }
            let mut times = vec![Vec::new(); kernels.len()];
            for round in 0..if measure { 7 } else { 1 } {
                for route in (0..kernels.len()).map(|r| (r + round) % kernels.len()) {
                    let (kernel, width) = kernels[route];
                    poison_output(&y, count);
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernel,
                        &[&w, &x, &y],
                        &[
                            k as u32,
                            n as u32,
                            rows as u32,
                            bits as u32,
                            group as u32,
                            1,
                            start as u32,
                        ],
                        [n.div_ceil(8), rows.div_ceil(width), 1],
                        64,
                    );
                    let elapsed = cmd.finish().unwrap();
                    if round > 0 {
                        times[route].push(elapsed);
                    }
                    let got = unsafe { y.read_f32(0, count + 32) };
                    assert!(
                        got[..start * n]
                            .iter()
                            .chain(&got[count..])
                            .all(|x| x.is_nan())
                    );
                    let got = got[start * n..count]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>();
                    if let Some(expected) = &expected {
                        assert!(
                            &got == expected,
                            "kernel={kernel} K={k} N={n} bits={bits} rows={rows} start={start}"
                        );
                    } else {
                        expected = Some(got);
                    }
                }
            }
            if measure {
                eprintln!(
                    "FLASH_WIDE_ROWS_COST {}",
                    serde_json::json!({"k":k,"n":n,"bits":bits,
                "rows":rows,"start":start,"kernels":kernels,"gpu_seconds":times,"exact":true})
                );
            }
        }
    }
}

#[test]
fn expert_gate_up_vector_preserves_singleton_arithmetic_and_guards() {
    expert_gate_up_vector_cases(false);
}

#[test]
fn hc_down_vector_preserves_projection_and_activation_bits() {
    hc_down_vector_cases(false);
}

#[test]
fn hc_up_mix_vector_preserves_gate_and_mixed_bits() {
    hc_up_mix_vector_cases(false);
}

#[test]
#[ignore = "checkpoint-backed HC up/mix exact outputs and rotated costs"]
fn hc_up_mix_vector_execution_cost() {
    hc_up_mix_vector_cases(true);
}

fn hc_up_mix_vector_cases(measure: bool) {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let w = if measure {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
        let source = super::mlx::Source::open(Path::new(&path)).unwrap();
        source
            .weight(
                &d,
                &format!(
                    "{}.layers.0.attn_hyper_connection.input_mix_weight_up",
                    super::mlx::ROOT
                ),
            )
            .unwrap()
    } else {
        let size = 320 * 10240;
        let mut raw = (0..size / 2)
            .map(|i| (i * 31 + i / 127) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..size / 32).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            k: 320,
            n: 10240,
            ty: affine::A4G32,
        }
    };
    for rows in [1, 4, 8] {
        let values = |n| {
            (0..n)
                .flat_map(|i| {
                    half::bf16::from_f32((i % 137) as f32 / 97. - 0.7)
                        .to_f32()
                        .to_le_bytes()
                })
                .collect::<Vec<_>>()
        };
        let x = d.upload(&values(rows * 320)).unwrap();
        let norm = d.upload(&values(rows * 10240)).unwrap();
        let gate = d.alloc((rows * 10240 + 32) * 4).unwrap();
        let mixed = d.alloc((rows * 2560 + 32) * 4).unwrap();
        let mut expected = None;
        let mut times: [Vec<f64>; 2] = Default::default();
        for round in 0..if measure { 9 } else { 1 } {
            for index in 0..2 {
                let route = (round + index) % 2;
                for (y, count) in [(&gate, rows * 10240), (&mixed, rows * 2560)] {
                    poison_output(y, count);
                }
                let cmd = d.begin().unwrap().with_independent_rows(true);
                for _ in 0..if measure { 32 } else { 1 } {
                    if route == 0 {
                        affine::project(&cmd, &w, &x, &gate, rows);
                        cmd.dispatch(
                            "q4b_hc_mix",
                            &[&norm, &gate, &mixed],
                            &[2560, rows as u32],
                            [(rows * 2560).div_ceil(256), 1, 1],
                            256,
                        );
                    } else {
                        cmd.dispatch(
                            "q4a_hc_up_mix_vector",
                            &[&w.buffer, &x, &norm, &gate, &mixed],
                            &[rows as u32],
                            [640, rows, 1],
                            128,
                        );
                    }
                }
                let gpu = cmd.finish().unwrap();
                let got = [(&gate, rows * 10240), (&mixed, rows * 2560)].map(|(y, count)| {
                    let v = unsafe { y.read_u32(count + 32) };
                    assert!(v[count..].iter().all(|&v| v == f32::NAN.to_bits()));
                    v[..count].to_vec()
                });
                if let Some(expected) = &expected {
                    assert_eq!(&got, expected, "rows={rows} route={route}");
                } else {
                    expected = Some(got);
                }
                if round > 0 {
                    times[route].push(gpu / 32.);
                }
            }
        }
        if measure {
            eprintln!(
                "FLASH_HC_UP_VECTOR {}",
                serde_json::json!({"rows":rows,"gpu_seconds":times,"exact":true})
            );
        }
    }
}

#[test]
#[ignore = "checkpoint-backed HC fusion exact outputs and rotated costs"]
fn hc_down_vector_execution_cost() {
    hc_down_vector_cases(true);
}

fn hc_down_vector_cases(measure: bool) {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let k = 10240;
    let weights = if measure {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
        let source = super::mlx::Source::open(Path::new(&path)).unwrap();
        ["input_mix_weight_down", "block_inject_weight"]
            .into_iter()
            .map(|suffix| {
                source
                    .weight(
                        &d,
                        &format!(
                            "{}.layers.0.attn_hyper_connection.{suffix}",
                            super::mlx::ROOT
                        ),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>()
    } else {
        [320, 4]
            .into_iter()
            .enumerate()
            .map(|(index, n)| {
                let size = k * n;
                let mut raw = (0..size / 2)
                    .map(|i| (i * 31 + i / 127 + index * 71) as u8)
                    .collect::<Vec<_>>();
                for bias in [false, true] {
                    raw.extend((0..size / 32).flat_map(|i| {
                        half::bf16::from_f32(if bias {
                            -0.04 + (i % 17) as f32 * 0.001
                        } else {
                            0.002 + (i % 31) as f32 * 0.00002
                        })
                        .to_le_bytes()
                    }));
                }
                crate::weights::Weight {
                    buffer: d.upload(&raw).unwrap(),
                    k,
                    n,
                    ty: affine::A4G32,
                }
            })
            .collect::<Vec<_>>()
    };
    for rows in [1, 4, 8] {
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| {
                        half::bf16::from_f32((i % 137) as f32 / 97. - 0.7)
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let outputs = [
            d.alloc((rows * 320 + 32) * 4).unwrap(),
            d.alloc((rows * 4 + 32) * 4).unwrap(),
        ];
        let mut expected = None;
        let mut times: [Vec<f64>; 2] = Default::default();
        for round in 0..if measure { 9 } else { 1 } {
            for index in 0..2 {
                let route = (round + index) % 2;
                for (w, y) in weights.iter().zip(&outputs) {
                    poison_output(y, rows * w.n);
                }
                let cmd = d.begin().unwrap().with_independent_rows(true);
                for _ in 0..if measure { 32 } else { 1 } {
                    if route == 0 {
                        for (i, (w, y)) in weights.iter().zip(&outputs).enumerate() {
                            affine::project(&cmd, w, &x, y, rows);
                            cmd.dispatch(
                                if i == 0 {
                                    "q4b_scale_silu"
                                } else {
                                    "q4b_injection"
                                },
                                &[y],
                                &[(rows * w.n) as u32],
                                [(rows * w.n).div_ceil(256), 1, 1],
                                256,
                            );
                        }
                    } else {
                        cmd.dispatch(
                            "q4a_hc_down_vector",
                            &[
                                &weights[0].buffer,
                                &weights[1].buffer,
                                &x,
                                &outputs[0],
                                &outputs[1],
                            ],
                            &[rows as u32],
                            [42, rows, 1],
                            64,
                        );
                    }
                }
                let gpu = cmd.finish().unwrap();
                let got = weights
                    .iter()
                    .zip(&outputs)
                    .map(|(w, y)| {
                        let v = unsafe { y.read_u32(rows * w.n + 32) };
                        assert!(v[rows * w.n..].iter().all(|&v| v == f32::NAN.to_bits()));
                        v[..rows * w.n].to_vec()
                    })
                    .collect::<Vec<_>>();
                if let Some(expected) = &expected {
                    assert_eq!(&got, expected, "rows={rows} route={route}");
                } else {
                    expected = Some(got);
                }
                if round > 0 {
                    times[route].push(gpu / 32.);
                }
            }
        }
        if measure {
            eprintln!(
                "FLASH_HC_VECTOR {}",
                serde_json::json!({"rows":rows,"gpu_seconds":times,"exact":true})
            );
        }
    }
}

#[test]
#[ignore = "checkpoint-backed singleton expert fusion costs; not serving timings"]
fn expert_gate_up_vector_execution_cost() {
    expert_gate_up_vector_cases(true);
}

fn expert_gate_up_vector_cases(measure: bool) {
    expert_gate_up_cases(measure, false);
}

#[test]
fn expert_gate_up_ordered_preserves_routes_and_guards() {
    expert_gate_up_cases(false, true);
}

#[test]
#[ignore = "checkpoint-backed routed prompt gate/up fusion; exact outputs and rotated component GPU costs"]
fn expert_gate_up_ordered_execution_cost() {
    expert_gate_up_cases(true, true);
}

fn expert_gate_up_cases(measure: bool, ordered: bool) {
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    let k = 2560;
    let n = 640;
    let experts = if measure { 512 } else { 17 };
    let weights = if measure {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
        let source = super::mlx::Source::open(Path::new(&path)).unwrap();
        [
            "switch_mlp.gate_proj",
            "switch_mlp.up_proj",
            "shared_expert.gate_proj",
            "shared_expert.up_proj",
        ]
        .into_iter()
        .map(|suffix| {
            source
                .weight(&d, &format!("{}.layers.0.mlp.{suffix}", super::mlx::ROOT))
                .unwrap()
        })
        .collect::<Vec<_>>()
    } else {
        [experts, experts, 1, 1]
            .into_iter()
            .enumerate()
            .map(|(index, e)| {
                let size = k * n * e;
                let mut raw = (0..size / 2)
                    .map(|i| (i * 31 + i / 127 + index * 71) as u8)
                    .collect::<Vec<_>>();
                for bias in [false, true] {
                    raw.extend((0..size / 32).flat_map(|i| {
                        half::bf16::from_f32(if bias {
                            -0.04 + (i % 17) as f32 * 0.001
                        } else {
                            0.002 + (i % 31) as f32 * 0.00002
                        })
                        .to_le_bytes()
                    }));
                }
                crate::weights::Weight {
                    buffer: d.upload(&raw).unwrap(),
                    k,
                    n: n * e,
                    ty: affine::A4G32,
                }
            })
            .collect::<Vec<_>>()
    };
    let row_counts: &[usize] = if ordered {
        if measure {
            &[7, 14, 49, 96, 128]
        } else {
            &[7, 13, 49, 128]
        }
    } else if measure {
        &[1, 4, 8]
    } else {
        &[1, 2, 3, 4, 7, 8]
    };
    for &rows in row_counts {
        for routing in 0..if measure { 2 } else { 3 } {
            let skewed = routing == 1;
            let x = d
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            half::bf16::from_f32((i % 137) as f32 / 97. - 0.7)
                                .to_f32()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let ids = d
                .upload(
                    &(0..rows * 10)
                        .flat_map(|i| {
                            if routing == 2 && i % 7 == 0 {
                                u32::MAX
                            } else if skewed {
                                (experts - 1) as u32
                            } else {
                                ((i * 43 + i / 10 * 13) % experts) as u32
                            }
                            .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let outputs = [rows * 10 * n, rows * 10 * n, rows * n, rows * n]
                .map(|count| d.alloc((count + 32) * 4).unwrap());
            for (out, count) in
                outputs
                    .iter()
                    .zip([rows * 10 * n, rows * 10 * n, rows * n, rows * n])
            {
                poison_output(out, count);
            }
            let order = d.alloc(rows * 10 * 4).unwrap();
            if ordered {
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "q4a_expert_order",
                    &[&ids, &order],
                    &[(rows * 10) as u32],
                    [1, 1, 1],
                    256,
                );
                cmd.finish().unwrap();
            }
            let baseline = |cmd: &crate::device::Commands<'_>, shared_only: bool| {
                for plane in if shared_only { 2 } else { 0 }..4 {
                    let w = &weights[plane];
                    if plane < 2 {
                        cmd.dispatch(
                            if ordered {
                                "q4a_expert4_fast_ordered"
                            } else {
                                "q4a_expert4_fast"
                            },
                            &[
                                &w.buffer,
                                &x,
                                &ids,
                                &outputs[plane],
                                if ordered { &order } else { &ids },
                            ],
                            &[k as u32, n as u32, (rows * 10) as u32, 0, experts as u32],
                            if ordered {
                                [n.div_ceil(16) * 8, (rows * 10).div_ceil(8), 1]
                            } else {
                                [n.div_ceil(16), rows * 10, 1]
                            },
                            128,
                        );
                    } else {
                        cmd.dispatch(
                            "q4a_mv4_fast",
                            &[&w.buffer, &x, &outputs[plane]],
                            &[k as u32, n as u32, rows as u32, 4, 32, 1, 0],
                            [n.div_ceil(16), rows, 1],
                            128,
                        );
                    }
                }
                for (gate, up, count) in [(0, 1, rows * 10 * n), (2, 3, rows * n)] {
                    if shared_only && gate == 0 {
                        continue;
                    }
                    cmd.dispatch(
                        "mlx_swiglu",
                        &[&outputs[gate], &outputs[up]],
                        &[count as u32],
                        [count.div_ceil(256), 1, 1],
                        256,
                    );
                }
            };
            let cmd = d.begin().unwrap();
            baseline(&cmd, false);
            cmd.finish().unwrap();
            let expected = [unsafe { outputs[0].read_u32(rows * 10 * n) }, unsafe {
                outputs[2].read_u32(rows * n)
            }];
            let allocated = d.allocated_bytes();
            let variants: &[(&str, usize, usize)] = if ordered {
                &[
                    ("baseline", 4, 128),
                    ("q4a_expert_gate_up_ordered", 4, 64),
                    ("q4a_expert_gate_up_ordered", 4, 128),
                    ("q4a_expert_gate_up_ordered_pair2", 2, 64),
                    ("q4a_expert_gate_up_ordered_pair4", 4, 64),
                    ("q4a_expert_gate_up_ordered_pair4", 4, 128),
                ]
            } else {
                &[
                    ("baseline", 4, 128),
                    ("q4a_expert_gate_up_vector", 4, 64),
                    ("q4a_expert_gate_up_vector", 4, 128),
                    ("q4a_expert_gate_up_pair2", 2, 64),
                    ("q4a_expert_gate_up_pair4", 4, 64),
                    ("q4a_expert_gate_up_pair4", 4, 128),
                ]
            };
            let mut times = vec![Vec::new(); variants.len()];
            for round in 0..if measure { 9 } else { 1 } {
                for index in 0..variants.len() {
                    let route = (index + round) % variants.len();
                    for (out, count) in [(&outputs[0], rows * 10 * n), (&outputs[2], rows * n)] {
                        poison_output(out, count);
                    }
                    let cmd = d.begin().unwrap();
                    for _ in 0..if measure { 16 } else { 1 } {
                        if route == 0 {
                            baseline(&cmd, false);
                        } else {
                            let (name, columns, threads) = variants[route];
                            let mut buffers = vec![
                                &weights[0].buffer,
                                &weights[1].buffer,
                                &weights[2].buffer,
                                &weights[3].buffer,
                                &x,
                                &ids,
                                &outputs[0],
                                &outputs[2],
                            ];
                            let stripes = if name.contains("pair") { 4 } else { 8 };
                            if ordered {
                                baseline(&cmd, true);
                                buffers.push(&order);
                            }
                            cmd.dispatch(
                                name,
                                &buffers,
                                &[k as u32, n as u32, rows as u32, experts as u32],
                                if ordered {
                                    [
                                        n.div_ceil(threads / 32 * columns) * stripes,
                                        (rows * 10).div_ceil(8) + 1,
                                        1,
                                    ]
                                } else {
                                    [n.div_ceil(threads / 32 * columns), rows * 11 + 1, 1]
                                },
                                threads,
                            );
                        }
                    }
                    let gpu = cmd.finish().unwrap();
                    for (i, count) in [rows * 10 * n, rows * n].into_iter().enumerate() {
                        let got = unsafe { outputs[i * 2].read_u32(count + 32) };
                        assert!(
                            got[..count] == expected[i],
                            "rows={rows} route={route} routing={routing}"
                        );
                        assert!(got[count..].iter().all(|&v| v == f32::NAN.to_bits()));
                    }
                    if round > 0 {
                        times[route].push(gpu / 16.);
                    }
                }
            }
            assert_eq!(d.allocated_bytes(), allocated);
            if measure {
                eprintln!(
                    "FLASH_EXPERT_VECTOR {}",
                    serde_json::json!({"rows":rows,"skewed":skewed,"ordered":ordered,"variants":variants,"gpu_seconds":times,"exact":true})
                );
            }
        }
    }
}

#[test]
fn joined_input_projection_preserves_tiles_spans_and_output_guards() {
    joined_input_cases(false);
}

#[test]
fn expert_down_vector_preserves_outputs_invalid_routes_and_guards() {
    expert_down_vector_cases(false);
}

#[test]
#[ignore = "checkpoint-backed expert down costs, exact routed/shared outputs and guards; not a serving bar"]
fn expert_down_vector_execution_cost() {
    expert_down_vector_cases(true);
}

fn expert_down_vector_cases(measure: bool) {
    let d = MetalDevice::new(Some(2 << 30)).unwrap();
    for (k, n) in [(640_usize, 2560_usize), (64, 13)] {
        if measure && n != 2560 {
            continue;
        }
        let experts = if measure { 512 } else { 17 };
        let weights = if measure {
            let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
            let source = super::mlx::Source::open(Path::new(&path)).unwrap();
            ["switch_mlp.down_proj", "shared_expert.down_proj"]
                .into_iter()
                .map(|suffix| {
                    source
                        .weight(&d, &format!("{}.layers.0.mlp.{suffix}", super::mlx::ROOT))
                        .unwrap()
                })
                .collect::<Vec<_>>()
        } else {
            [experts, 1]
                .into_iter()
                .enumerate()
                .map(|(plane, e)| {
                    let size = k * n * e;
                    let mut raw = (0..size / 2)
                        .map(|i| (i * 31 + i / 127 + plane * 71) as u8)
                        .collect::<Vec<_>>();
                    for bias in [false, true] {
                        raw.extend((0..size / 32).flat_map(|i| {
                            half::bf16::from_f32(if bias {
                                -0.04 + (i % 17) as f32 * 0.001
                            } else {
                                0.002 + (i % 31) as f32 * 0.00002
                            })
                            .to_le_bytes()
                        }));
                    }
                    crate::weights::Weight {
                        buffer: d.upload(&raw).unwrap(),
                        k,
                        n: n * e,
                        ty: affine::A4G32,
                    }
                })
                .collect::<Vec<_>>()
        };
        for rows in [1, 4, 8] {
            for routing in 0..if measure { 2 } else { 3 } {
                let inputs = [rows * 10 * k, rows * k].map(|count| {
                    d.upload(
                        &(0..count)
                            .flat_map(|i| {
                                half::bf16::from_f32((i % 137) as f32 / 97. - 0.7)
                                    .to_f32()
                                    .to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap()
                });
                let ids = d
                    .upload(
                        &(0..rows * 10)
                            .flat_map(|i| {
                                let id = if routing == 2 && i % 7 == 0 {
                                    u32::MAX
                                } else if routing == 1 {
                                    (experts - 1) as u32
                                } else {
                                    ((i * 43 + i / 10 * 13) % experts) as u32
                                };
                                id.to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let counts = [rows * 10 * n, rows * n];
                let outputs = counts.map(|count| d.alloc((count + 32) * 4).unwrap());
                let baseline = |cmd: &crate::device::Commands<'_>| {
                    cmd.dispatch(
                        "q4a_expert4",
                        &[&weights[0].buffer, &inputs[0], &ids, &outputs[0], &ids],
                        &[k as u32, n as u32, (rows * 10) as u32, 1, experts as u32],
                        [n.div_ceil(16), rows * 10, 1],
                        128,
                    );
                    cmd.dispatch(
                        "q4a_mv4",
                        &[&weights[1].buffer, &inputs[1], &outputs[1]],
                        &[k as u32, n as u32, rows as u32, 4, 32, 1, 0],
                        [n.div_ceil(16), rows, 1],
                        128,
                    );
                };
                for (out, count) in outputs.iter().zip(counts) {
                    poison_output(out, count);
                }
                let cmd = d.begin().unwrap();
                baseline(&cmd);
                cmd.finish().unwrap();
                let expected = [0, 1].map(|i| unsafe { outputs[i].read_u32(counts[i] + 32) });
                let variants = [
                    ("baseline", 4, 128),
                    ("q4a_expert_down_vector4", 4, 64),
                    ("q4a_expert_down_vector8", 8, 32),
                    ("q4a_expert_down_vector8", 8, 64),
                    ("q4a_expert_down_vector8", 8, 128),
                    ("q4a_expert_down_vector16", 16, 64),
                ];
                let mut times = vec![Vec::new(); variants.len()];
                let allocated = d.allocated_bytes();
                for round in 0..if measure { 9 } else { 1 } {
                    for route in (0..variants.len()).map(|i| (i + round) % variants.len()) {
                        let (name, width, threads) = variants[route];
                        for (out, count) in outputs.iter().zip(counts) {
                            poison_output(out, count);
                        }
                        let cmd = d.begin().unwrap();
                        for _ in 0..if measure { 16 } else { 1 } {
                            if route == 0 {
                                baseline(&cmd);
                            } else {
                                cmd.dispatch(
                                    name,
                                    &[
                                        &weights[0].buffer,
                                        &weights[1].buffer,
                                        &inputs[0],
                                        &inputs[1],
                                        &ids,
                                        &outputs[0],
                                        &outputs[1],
                                    ],
                                    &[k as u32, n as u32, rows as u32, experts as u32],
                                    [n.div_ceil(threads / 32 * width), rows * 11 + 1, 1],
                                    threads,
                                );
                            }
                        }
                        let gpu = cmd.finish().unwrap();
                        for i in 0..2 {
                            assert!(
                                unsafe { outputs[i].read_u32(counts[i] + 32) } == expected[i],
                                "K={k} N={n} rows={rows} route={route} routing={routing} plane={i}"
                            );
                            assert!(
                                expected[i][counts[i]..]
                                    .iter()
                                    .all(|&v| v == f32::NAN.to_bits())
                            );
                        }
                        if round > 0 {
                            times[route].push(gpu / 16.);
                        }
                    }
                }
                assert_eq!(d.allocated_bytes(), allocated);
                if measure {
                    eprintln!(
                        "FLASH_DOWN_VECTOR {}",
                        serde_json::json!({"rows":rows,"routing":routing,"variants":variants,"gpu_seconds":times,"exact":true})
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "checkpoint-backed rotated joined projection costs; exact output required, not serving timing"]
fn joined_input_execution_cost() {
    joined_input_cases(true);
}

fn joined_input_cases(measure: bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(true));
            affine::JOINED_SLAB_FOR_TEST.with(|v| v.set(false));
            affine::JOINED_ROW_GROUP_FOR_TEST.with(|v| v.set(0));
        }
    }
    let _reset = Reset;
    affine::JOINED_SLAB_FOR_TEST.with(|v| v.set(true));
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let k = 2560;
    let weights = if measure {
        let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
        let source = super::mlx::Source::open(Path::new(&path)).unwrap();
        ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"]
            .into_iter()
            .map(|suffix| {
                source
                    .weight(
                        &d,
                        &format!("{}.layers.0.linear_attn.{suffix}", super::mlx::ROOT),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>()
    } else {
        [10240, 6144, 48, 48]
            .into_iter()
            .enumerate()
            .map(|(index, n)| {
                let mut raw = (0..k * n / 2)
                    .map(|i| (i * 31 + i / 127 + index * 71) as u8)
                    .collect::<Vec<_>>();
                for bias in [false, true] {
                    raw.extend((0..k * n / 32).flat_map(|i| {
                        half::bf16::from_f32(if bias {
                            -0.04 + (i % 17) as f32 * 0.001
                        } else {
                            0.002 + (i % 31) as f32 * 0.00002
                        })
                        .to_le_bytes()
                    }));
                }
                crate::weights::Weight {
                    buffer: d.upload(&raw).unwrap(),
                    k,
                    n,
                    ty: affine::A4G32,
                }
            })
            .collect::<Vec<_>>()
    };
    for spans in [
        vec![(0, 1, 1)],
        vec![(0, 13, 13)], // unequal split-K elections: fallback
        vec![(0, 65, 65)],
        vec![(0, 1, 1), (1, 33, 128), (34, 31, 64)],
        vec![(0, 511, 512)],
        vec![(0, 512, 512)],
        vec![(0, 1024, 1024)],
        vec![(0, 768, 1024), (768, 769, 1024)],
        vec![
            (0, 512, 1024),
            (512, 512, 1024),
            (1024, 512, 1024),
            (1536, 512, 1024),
        ],
    ] {
        let rows = spans.iter().map(|s| s.1).sum::<usize>();
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let outputs = weights
            .iter()
            .map(|w| d.alloc((rows * w.n + 32) * 4).unwrap())
            .collect::<Vec<_>>();
        let group = weights.iter().zip(&outputs).collect::<Vec<_>>();
        // The narrow arena forces row slicing; full arena crosses the virtual
        // QKV/Z boundary inside a slab and includes a ragged final row tile.
        let arenas = if measure {
            vec![affine::WORKSPACE_BYTES]
        } else {
            vec![512 * k * 2, affine::WORKSPACE_BYTES]
        };
        for arena in arenas {
            let scratch = d.alloc(arena).unwrap();
            affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(false));
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(&spans);
            affine::project_group(&cmd, &group, &x, rows);
            cmd.finish().unwrap();
            let expected = group
                .iter()
                .map(|(w, y)| unsafe { y.read_u32(rows * w.n) })
                .collect::<Vec<_>>();
            let allocated = d.allocated_bytes();
            let mut times: [Vec<f64>; 5] = Default::default();
            for round in 0..if measure { 9 } else { 1 } {
                for index in 0..5 {
                    let route = (index + round) % 5;
                    for (w, y) in &group {
                        poison_output(y, rows * w.n);
                    }
                    affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(route != 0));
                    affine::JOINED_ROW_GROUP_FOR_TEST.with(|v| v.set([1, 1, 4, 8, 16][route]));
                    let cmd = d
                        .begin()
                        .unwrap()
                        .with_projection_workspace(&scratch)
                        .with_projection_rows(&spans);
                    affine::project_group(&cmd, &group, &x, rows);
                    let gpu = cmd.finish().unwrap();
                    for ((w, y), expected) in group.iter().zip(&expected) {
                        let got = unsafe { y.read_u32(rows * w.n + 32) };
                        assert!(
                            &got[..rows * w.n] == expected,
                            "rows={rows} arena={arena} n={} route={route}",
                            w.n
                        );
                        assert!(got[rows * w.n..].iter().all(|&v| v == f32::NAN.to_bits()));
                    }
                    if round > 0 {
                        times[route].push(gpu);
                    }
                }
            }
            assert_eq!(d.allocated_bytes(), allocated);
            if measure {
                eprintln!(
                    "FLASH_JOINED_INPUT {}",
                    serde_json::json!({"rows":rows, "spans":spans, "gpu_seconds":times, "exact":true})
                );
            }
        }
    }
}

#[test]
fn grouped_input_staging_preserves_contracts_arena_fallback_and_mutation() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(true));
    use crate::weights::Weight;
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let k = 320;
    let weights = [
        (515, affine::A4G32),
        (640, affine::A8G64),
        (48, affine::A4G32),
        (640, affine::A4G32),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (n, ty))| {
        let (bits, group) = affine::format(ty);
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 31 + i / 127 + index * 71) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / group).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        Weight {
            buffer: d.upload(&raw).unwrap(),
            ty,
            k,
            n,
        }
    })
    .collect::<Vec<_>>();
    for spans in [
        vec![(0, 1, 1)],
        vec![(0, 13, 128)],
        vec![(0, 65, 65)],
        vec![(0, 1, 1), (1, 33, 128), (34, 31, 64)],
        vec![(0, 128, 1024), (128, 128, 1024)],
    ] {
        let rows = spans.iter().map(|s| s.1).sum::<usize>();
        let x = d.alloc(rows * k * 4).unwrap();
        let outputs = weights
            .iter()
            .map(|w| d.alloc((rows * w.n + 32) * 4).unwrap())
            .collect::<Vec<_>>();
        let group = weights.iter().zip(&outputs).collect::<Vec<_>>();
        // Tiny scratch must invalidate sharing before inline split-K overwrites
        // the arena. Large scratch exercises staged split-K and dense reuse.
        for arena in [4096usize, 96 << 10, 512 << 10, 8 << 20] {
            let scratch = d.alloc(arena).unwrap();
            // A tiny arena may not hold original split-K partials at all.
            let fits = weights.iter().all(|w| {
                spans.iter().all(|&(_, count, logical)| {
                    let (kind, parts) = affine::contraction(k, w.n, w.ty, logical);
                    kind != 2 || parts == 1 || count * w.n * parts * 2 <= arena
                })
            });
            if !fits {
                continue;
            }
            for epoch in 0..2 {
                unsafe {
                    x.write_u32(
                        &(0..rows * k)
                            .map(|i| (((i + epoch * 39) % 137) as f32 / 97. - 0.7).to_bits())
                            .collect::<Vec<_>>(),
                    )
                };
                let cmd = d
                    .begin()
                    .unwrap()
                    .with_projection_workspace(&scratch)
                    .with_projection_rows(&spans);
                for (w, y) in &group {
                    affine::project(&cmd, w, &x, y, rows);
                }
                cmd.finish().unwrap();
                let expected = group
                    .iter()
                    .map(|(w, y)| unsafe { y.read_u32(rows * w.n) })
                    .collect::<Vec<_>>();
                for (w, y) in &group {
                    poison_output(y, rows * w.n);
                }
                let allocated = d.allocated_bytes();
                let cmd = d
                    .begin()
                    .unwrap()
                    .with_projection_workspace(&scratch)
                    .with_projection_rows(&spans);
                affine::project_group(&cmd, &group, &x, rows);
                cmd.finish().unwrap();
                for ((w, y), expected) in group.iter().zip(&expected) {
                    let got = unsafe { y.read_u32(rows * w.n + 32) };
                    assert_eq!(
                        &got[..rows * w.n],
                        expected,
                        "rows={rows} arena={arena} epoch={epoch} n={}",
                        w.n
                    );
                    assert!(got[rows * w.n..].iter().all(|&v| v == f32::NAN.to_bits()));
                }
                assert_eq!(d.allocated_bytes(), allocated);
            }
        }
    }
}

#[test]
fn weight_slabs_preserve_ragged_slices_offsets_and_guards() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    for k in [256usize, 320] {
        let n = 515;
        let mut raw = (0..k * n / 2)
            .map(|i| (i * 37 + i / 127 + 19) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / 32).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        let w = d.upload(&raw).unwrap();
        for rows in [1usize, 31, 32, 33, 63, 64, 65, 129] {
            let start = 3;
            let x = d
                .upload(
                    &(0..(start + rows) * k)
                        .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, 4, 32, 1, start as u32];
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_mm4_packed",
                &[&w, &x, &y],
                &p,
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            let expected = unsafe { y.read_f32(0, count + 32) };
            // 32 and 128-row arenas force independent physical row slices;
            // 515 output columns force full and ragged weight slabs.
            for capacity in [31usize, 32, 128] {
                let words = (256 + capacity) * k / 2;
                let scratch = d.alloc((words + 16) * 4).unwrap();
                unsafe {
                    scratch.write_u32(&vec![0xdeadbeef; words + 16]);
                }
                poison_output(&y, count);
                let allocated = d.allocated_bytes();
                let cmd = d.begin().unwrap();
                let accepted = affine::project_slab(&cmd, &w, &x, &scratch, &y, &p);
                assert_eq!(accepted, capacity >= 32);
                cmd.finish().unwrap();
                assert_eq!(allocated, d.allocated_bytes());
                let got = unsafe { y.read_f32(0, count + 32) };
                if accepted {
                    assert_eq!(&got[start * n..count], &expected[start * n..count]);
                } else {
                    assert!(got.iter().all(|v| v.is_nan()));
                }
                assert!(
                    got[..start * n]
                        .iter()
                        .chain(&got[count..])
                        .all(|v| v.is_nan())
                );
                let scratch = unsafe { scratch.read_u32(words + 16) };
                assert!(scratch[words..].iter().all(|&v| v == 0xdeadbeef));
            }
        }
    }
}

#[test]
#[ignore = "bounded weight-slab cost including conversion; exact output required"]
fn dense_slab_execution_cost() {
    let d = MetalDevice::new(Some(400 << 20)).unwrap();
    assert!(d.tensor_accelerated());
    for (k, n) in [
        (2560usize, 10240usize),
        (6144, 2560),
        (320, 10240),
        (10240, 320),
        (2560, 6144),
        (2560, 640),
        (640, 2560),
    ] {
        let mut raw = (0..k * n / 2)
            .map(|i| (i * 37 + i / 127 + 19) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / 32).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        let w = d.upload(&raw).unwrap();
        for rows in [64usize, 512, 1024, 2048] {
            let start = 3;
            let x = d
                .upload(
                    &(0..(start + rows) * k)
                        .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let staged = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
            let scratch = d.alloc(affine::WORKSPACE_BYTES).unwrap();
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, 4, 32, 1, start as u32];
            let mut expected = None;
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..7 {
                for route in [round % 2, (round + 1) % 2] {
                    poison_output(&y, count);
                    let cmd = d.begin().unwrap();
                    if route == 0 {
                        cmd.dispatch(
                            "q4a_input",
                            &[&x, &staged],
                            &p,
                            [(rows * k).div_ceil(256), 1, 1],
                            256,
                        );
                        cmd.dispatch(
                            if k.is_multiple_of(128) {
                                "q4a_mm4_device128_pad8"
                            } else {
                                "q4a_mm4_device64_pad8"
                            },
                            &[&w, &staged, &y],
                            &p,
                            [n.div_ceil(32), rows.div_ceil(32), 1],
                            128,
                        );
                    } else {
                        assert!(affine::project_slab(&cmd, &w, &x, &scratch, &y, &p));
                    }
                    let gpu = cmd.finish().unwrap();
                    let got = unsafe { y.read_f32(0, count + 32) };
                    assert!(
                        got[..start * n]
                            .iter()
                            .chain(&got[count..])
                            .all(|v| v.is_nan())
                    );
                    let got = got[start * n..count]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>();
                    if let Some(ref want) = expected {
                        assert!(got == *want, "slab k={k} n={n} rows={rows} route={route}");
                    } else {
                        expected = Some(got);
                    }
                    if round > 0 {
                        times[route].push(gpu);
                    }
                }
            }
            eprintln!(
                "FLASH_DENSE_SLAB {}",
                serde_json::json!({"k":k,"n":n,"rows":rows,"gpu_seconds":times,"exact":true})
            );
        }
    }
}

#[test]
fn group32_expert_loaders_preserve_mixed_rows_and_output_guards() {
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let (k, n) = (64usize, 64usize);
    let mut raw = (0..512 * k * n / 2)
        .map(|i| (i * 37 + 19) as u8)
        .collect::<Vec<_>>();
    for value in [0.003f32, -0.04] {
        raw.extend(
            (0..512 * k * n / 32).flat_map(|_| ((value.to_bits() >> 16) as u16).to_le_bytes()),
        );
    }
    let w = d.upload(&raw).unwrap();
    for rows in [1usize, 9, 33, 129, 2048] {
        let entries = rows * 10;
        let ids = (0..entries)
            .map(|i| {
                if i / 10 % 7 == 0 {
                    (i % 10) as u32
                } else {
                    ((i / 10 * 37 + i % 10 * 17) % 512) as u32
                }
            })
            .collect::<Vec<_>>();
        let ids = d
            .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let lists = d.alloc(512 * entries * 4).unwrap();
        let counts = d.alloc(512 * 4).unwrap();
        let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
        // Simulate gate/up's live F32 output prefix and dead suffix alias.
        // NaN canaries separate the two views; direct TensorOps must not
        // write packed input, cross an expert boundary or overwrite either.
        let sorted_offset = (entries * n + 32) * 4;
        let sorted_end = sorted_offset / 4 + entries * k / 2;
        let y = d.alloc((sorted_end + 32) * 4).unwrap();
        let offsets = d.alloc((512 + 16) * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[entries as u32],
            [512, 1, 1],
            256,
        );
        cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
        cmd.finish().unwrap();
        for per_entry in [false, true] {
            let input_rows = if per_entry { entries } else { rows };
            let x = d
                .upload(
                    &(0..input_rows * k)
                        .flat_map(|i| (((i * 29 % 137) as f32 - 68.) / 97.).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let staged = d.alloc(input_rows.next_multiple_of(32) * k * 2).unwrap();
            let mut p = vec![
                k as u32,
                n as u32,
                entries as u32,
                u32::from(per_entry),
                512,
            ];
            p.extend([u32::MAX; affine::MAX_ROWS.div_ceil(32)]);
            for row in (0..rows).step_by(7) {
                p[5 + row / 32] &= !(1 << (row % 32));
            }
            let mut expected = None;
            for kernel in [
                "q4a_expert_mm_tail",
                "q4a_expert_mm_group32",
                "q4a_expert_mm_group32_pad",
                "q4a_expert_mm_group32_packed",
                "q4a_expert_mm_direct",
                "q4a_expert_mm_direct64",
            ] {
                poison_output(&y, sorted_end);
                unsafe {
                    offsets.write_u32(&[0xdeadbeef; 528]);
                    tiles.write_u32(&vec![0xdeadbeef; tiles.len() / 4]);
                }
                let cmd = d.begin().unwrap();
                if !kernel.ends_with("64") {
                    cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
                }
                let input = if kernel.ends_with("_packed") {
                    cmd.dispatch(
                        "q4a_input",
                        &[&x, &staged],
                        &[k as u32, n as u32, input_rows as u32, 0, 0, 0, 0],
                        [(input_rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                        256,
                    );
                    &staged
                } else {
                    &x
                };
                if kernel.contains("_direct") {
                    let wide = kernel.ends_with("64");
                    if wide {
                        cmd.dispatch(
                            "q4a_expert_plan64",
                            &[&counts, &offsets, &tiles],
                            &[512],
                            [1, 1, 1],
                            512,
                        );
                    } else {
                        cmd.dispatch(
                            "q4a_expert_offsets",
                            &[&counts, &offsets],
                            &[512],
                            [1, 1, 1],
                            512,
                        );
                    }
                    cmd.dispatch_at(
                        if wide {
                            "q4a_expert_pack64"
                        } else {
                            "q4a_expert_pack"
                        },
                        &[&x, &lists, &counts, &tiles, &offsets, &y],
                        &[0, 0, 0, 0, 0, sorted_offset],
                        &p,
                        [k.div_ceil(512), entries.div_ceil(32) + 512, 1],
                        128,
                    );
                    cmd.dispatch_at(
                        kernel,
                        &[&w, &y, &lists, &counts, &tiles, &offsets, &y],
                        &[0, sorted_offset, 0, 0, 0, 0, 0],
                        &p,
                        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                        128,
                    );
                } else {
                    cmd.dispatch(
                        kernel,
                        &[&w, input, &lists, &counts, &tiles, &y],
                        &p,
                        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                        128,
                    );
                }
                cmd.dispatch(
                    "q4a_expert_vector_masked",
                    &[&w, &x, &ids, &y],
                    &p,
                    [n.div_ceil(16), entries, 1],
                    128,
                );
                cmd.finish().unwrap();
                let got = unsafe { y.read_f32(0, entries * n + 32) };
                assert!(got[..entries * n].iter().all(|v| v.is_finite()));
                assert!(got[entries * n..].iter().all(|v| v.is_nan()));
                assert!(
                    unsafe { y.read_f32(sorted_end, 32) }
                        .iter()
                        .all(|v| v.is_nan())
                );
                if kernel.contains("_direct") {
                    let counts = unsafe { counts.read_u32(512) };
                    let offsets = unsafe { offsets.read_u32(528) };
                    let lists = unsafe { lists.read_u32(512 * entries) };
                    let packed = unsafe { y.read_u32(sorted_end) };
                    let source = unsafe { x.read_f32(0, input_rows * k) };
                    let mut total = 0;
                    for expert in 0..512 {
                        assert_eq!(offsets[expert] as usize, total);
                        for row in 0..counts[expert] as usize {
                            let entry = lists[expert * entries + row] as usize;
                            let src = if per_entry { entry } else { entry / 10 } * k;
                            for col in 0..k {
                                let index = (total + row) * k + col;
                                let word = packed[sorted_offset / 4 + index / 2];
                                let bf = (word >> ((index % 2) * 16)) as u16;
                                assert_eq!(bf, half::bf16::from_f32(source[src + col]).to_bits());
                            }
                        }
                        total += counts[expert] as usize;
                    }
                    assert_eq!(total, entries);
                    assert!(offsets[512..].iter().all(|&v| v == 0xdeadbeef));
                    if kernel.ends_with("64") {
                        let plan = unsafe { tiles.read_u32(tiles.len() / 4) };
                        let mut expected = Vec::new();
                        for (expert, &count) in counts.iter().enumerate() {
                            let mut first = 0;
                            while first < count {
                                expected.extend([expert as u32, first]);
                                first += if count - first > 48 { 64 } else { 32 };
                            }
                        }
                        assert_eq!(plan[0] as usize * 2, expected.len());
                        assert_eq!(&plan[1..1 + expected.len()], expected);
                        assert!(plan[1 + expected.len()..].iter().all(|&v| v == 0xdeadbeef));
                    }
                }
                if let Some(expected) = &expected {
                    assert!(
                        &got[..entries * n] == expected,
                        "{kernel} rows={rows} per_entry={per_entry}"
                    );
                } else {
                    expected = Some(got[..entries * n].to_vec());
                }
            }
        }
    }
}

#[test]
fn expert_rows64_plan_covers_tile_boundaries_and_empty_experts() {
    let d = MetalDevice::new(Some(1 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let boundaries = [
        0u32, 1, 8, 9, 16, 17, 32, 33, 48, 49, 63, 64, 65, 96, 97, 112, 113, 128, 129,
    ];
    let counts = (0..512)
        .map(|i| boundaries[i % boundaries.len()])
        .collect::<Vec<_>>();
    let input = d
        .upload(
            &counts
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let offsets = d.alloc(528 * 4).unwrap();
    let tiles = d.alloc((1 + 2 * 512 * 3 + 16) * 4).unwrap();
    unsafe {
        offsets.write_u32(&[0xdeadbeef; 528]);
        tiles.write_u32(&vec![0xdeadbeef; tiles.len() / 4]);
    }
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_plan64",
        &[&input, &offsets, &tiles],
        &[512],
        [1, 1, 1],
        512,
    );
    cmd.finish().unwrap();
    let offsets = unsafe { offsets.read_u32(528) };
    let tiles = unsafe { tiles.read_u32(tiles.len() / 4) };
    let mut total = 0;
    let mut expected = Vec::new();
    for (expert, &count) in counts.iter().enumerate() {
        assert_eq!(offsets[expert], total);
        total += count;
        let mut first = 0;
        while first < count {
            expected.extend([expert as u32, first]);
            first += if count - first > 48 { 64 } else { 32 };
        }
    }
    assert_eq!(tiles[0] as usize * 2, expected.len());
    assert_eq!(&tiles[1..1 + expected.len()], expected);
    assert!(tiles[1 + expected.len()..].iter().all(|&v| v == 0xdeadbeef));
    assert!(offsets[512..].iter().all(|&v| v == 0xdeadbeef));
}

#[test]
fn fused_expert_gate_up_preserves_bf16_boundaries_masks_and_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let (k, n) = (128usize, 80usize);
    let weight = |seed: usize| {
        let mut raw = (0..512 * k * n / 2)
            .map(|i| (i.wrapping_mul(37 + seed) ^ (i >> 5) ^ seed) as u8)
            .collect::<Vec<_>>();
        for value in [0.013f32, -0.08] {
            raw.extend((0..512 * k * n / 32).flat_map(|i| {
                ((value
                    .mul_add(1. + (i % 7) as f32 / 8., seed as f32 / 1000.)
                    .to_bits()
                    >> 16) as u16)
                    .to_le_bytes()
            }));
        }
        d.upload(&raw).unwrap()
    };
    let gate = weight(1);
    let up = weight(3);
    for rows in [1usize, 9, 33, 129, 1024, 2048] {
        let entries = rows * 10;
        let ids = d
            .upload(
                &(0..entries)
                    .flat_map(|i| {
                        (if i / 10 % 7 == 0 {
                            i % 10
                        } else {
                            (i / 10 * 37 + i % 10 * 17) % 512
                        } as u32)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| (((i * 29 % 137) as f32 - 68.) / 97.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let packed = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
        let lists = d.alloc(512 * entries * 4).unwrap();
        let counts = d.alloc(512 * 4).unwrap();
        let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
        let y = d.alloc((entries * n + 32) * 4).unwrap();
        let u = d.alloc((entries * n + 32) * 4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[entries as u32],
            [512, 1, 1],
            256,
        );
        cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
        cmd.dispatch(
            "q4a_input",
            &[&x, &packed],
            &[k as u32, n as u32, rows as u32, 0, 0, 0, 0],
            [(rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        for mixed in [false, true] {
            let mut p = vec![k as u32, n as u32, entries as u32, 0, 512];
            p.extend([u32::MAX; affine::MAX_ROWS.div_ceil(32)]);
            if mixed {
                for row in (0..rows).step_by(7) {
                    p[5 + row / 32] &= !(1 << (row % 32));
                }
            }
            let mut expected = None;
            for fused in [0, 1, 2] {
                poison_output(&y, entries * n);
                poison_output(&u, entries * n);
                let cmd = d.begin().unwrap();
                if fused == 1 {
                    cmd.dispatch(
                        "q4a_expert_gate_up_packed",
                        &[&gate, &up, &packed, &lists, &counts, &tiles, &y],
                        &p,
                        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                        128,
                    );
                } else if fused == 2 {
                    cmd.dispatch(
                        "q4a_expert_gate_up_dispatch",
                        &[&gate, &up, &packed, &lists, &counts, &tiles, &y, &u],
                        &p,
                        [n.div_ceil(64), entries.div_ceil(32) + 512, 2],
                        128,
                    );
                } else {
                    for (w, out) in [(&gate, &y), (&up, &u)] {
                        cmd.dispatch(
                            "q4a_expert_mm_group32_packed",
                            &[w, &packed, &lists, &counts, &tiles, out],
                            &p,
                            [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                            128,
                        );
                    }
                }
                if mixed {
                    for (w, out) in [(&gate, &y), (&up, &u)] {
                        cmd.dispatch(
                            "q4a_expert_vector_masked",
                            &[w, &x, &ids, out],
                            &p,
                            [n.div_ceil(16), entries, 1],
                            128,
                        );
                    }
                }
                if fused == 1 {
                    if mixed {
                        cmd.dispatch(
                            "q4a_expert_swiglu_masked",
                            &[&y, &u],
                            &p,
                            [(entries * n).div_ceil(256), 1, 1],
                            256,
                        );
                    }
                } else {
                    cmd.dispatch(
                        "mlx_swiglu",
                        &[&y, &u],
                        &[(entries * n) as u32],
                        [(entries * n).div_ceil(256), 1, 1],
                        256,
                    );
                }
                cmd.finish().unwrap();
                let got = unsafe { y.read_f32(0, entries * n + 32) };
                assert!(got[..entries * n].iter().all(|v| v.is_finite()));
                assert!(got[entries * n..].iter().all(|v| v.is_nan()));
                assert!(
                    unsafe { u.read_f32(entries * n, 32) }
                        .iter()
                        .all(|v| v.is_nan())
                );
                if let Some(expected) = &expected {
                    assert!(&got[..entries * n] == expected, "rows={rows} mixed={mixed}");
                } else {
                    expected = Some(got[..entries * n].to_vec());
                }
            }
        }
    }
}

#[test]
fn unequal_projection_contracts_preserve_all_rows_and_guards() {
    let d = MetalDevice::new(Some(96 << 20)).unwrap();
    let spans = [
        (0, 1, 1),
        (1, 17, 700),
        (18, 15, 750),
        (33, 31, 800),
        (64, 64, 1024),
        (128, 31, 900),
        (159, 1, 1),
    ];
    let rows = 160;
    let short_spans = [
        (0, 4, 4),
        (4, 8, 8),
        (12, 8, 12),
        (20, 64, 64),
        (84, 64, 64),
        (148, 12, 12),
    ];
    let decode_spans = [
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (4, 64, 64),
        (68, 64, 64),
        (132, 28, 32),
    ];
    let scratch = d.alloc(affine::WORKSPACE_BYTES).unwrap();
    for (k, n, ty) in [
        (10240, 320, affine::A4G32),
        (2560, 6144, affine::A4G32),
        (2560, 512, affine::A8G64),
        (10240, 4, affine::A4G32),
        (320, 10240, affine::A4G32),
    ] {
        let (bits, group) = affine::format(ty);
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for plane in 0..2 {
            raw.extend((0..k * n / group).flat_map(|i| {
                half::bf16::from_f32(if plane == 0 {
                    0.0003 + (i % 17) as f32 * 0.0001
                } else {
                    -0.004 + (i % 7) as f32 * 0.0002
                })
                .to_le_bytes()
            }));
        }
        let w = crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            ty,
            k,
            n,
        };
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let y = d.alloc((rows * n + 32) * 4).unwrap();
        for spans in [&spans[..], &short_spans, &decode_spans] {
            poison_output(&y, rows * n);
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            for &(start, count, logical) in spans {
                affine::project_span(&cmd, &w, &x, &y, count, start, logical);
            }
            cmd.finish().unwrap();
            let expected = unsafe { y.read_f32(0, rows * n) };
            poison_output(&y, rows * n);
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(spans);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let got = unsafe { y.read_f32(0, rows * n + 32) };
            assert!(
                got[..rows * n] == expected,
                "merged projection changed K={k} N={n} type={ty}"
            );
            assert!(got[rows * n..].iter().all(|v| v.is_nan()));
        }
    }
}

#[test]
fn wide_physical_projection_preserves_logical_contraction_and_guards() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    let rows = 2048;
    let spans = [
        (0, 512, 1024),
        (512, 512, 1024),
        (1024, 512, 1024),
        (1536, 512, 1024),
    ];
    let scratch = d.alloc(affine::workspace_bytes(rows)).unwrap();
    for (k, n, ty) in [
        (10240, 320, affine::A4G32),
        (2560, 6144, affine::A4G32),
        (2560, 512, affine::A8G64),
        (320, 10240, affine::A4G32),
    ] {
        let (bits, group) = affine::format(ty);
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for value in [0.003f32, -0.04] {
            raw.extend((0..k * n / group).flat_map(|_| half::bf16::from_f32(value).to_le_bytes()));
        }
        let w = crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            k,
            n,
            ty,
        };
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let y = d.alloc((rows * n + 32) * 4).unwrap();
        poison_output(&y, rows * n);
        let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
        for &(start, count, logical) in &spans {
            affine::project_span(&cmd, &w, &x, &y, count, start, logical);
        }
        cmd.finish().unwrap();
        let expected = unsafe { y.read_f32(0, rows * n) };
        poison_output(&y, rows * n);
        let cmd = d
            .begin()
            .unwrap()
            .with_projection_workspace(&scratch)
            .with_projection_rows(&spans);
        affine::project(&cmd, &w, &x, &y, rows);
        cmd.finish().unwrap();
        let got = unsafe { y.read_f32(0, rows * n + 32) };
        assert!(
            got[..rows * n] == expected,
            "wide physical pass changed K={k} N={n} type={ty}"
        );
        assert!(got[rows * n..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn device_input_staging_preserves_offset_tail_and_guards() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let n = 515usize;
    for (k, bits, group, kernel, baseline) in [
        (320usize, 4, 32, "q4a_mm4_device64", "q4a_mm4_packed"),
        (320, 8, 64, "q4a_mm8_device64", "q4a_mm8_packed"),
        (320, 4, 32, "q4a_mm4_device64_group32", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_group32", "q4a_mm4_packed"),
        (320, 4, 32, "q4a_mm4_device64_pad8", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_pad8", "q4a_mm4_packed"),
        (320, 4, 32, "q4a_mm4_device64_pad16", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_pad16", "q4a_mm4_packed"),
        (256, 4, 32, "q4a_mm4_device128_wide", "q4a_mm4_packed"),
    ] {
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for value in [0.003f32, -0.04] {
            raw.extend(
                (0..k * n / group)
                    .flat_map(|_| half::bf16::from_f32(value).to_bits().to_le_bytes()),
            );
        }
        let w = d.upload(&raw).unwrap();
        for rows in [1usize, 31, 32, 33, 63, 64, 65, 129] {
            let start = 3usize;
            let values = (0..(start + rows) * k)
                .map(|i| (i % 137) as f32 / 97. - 0.7)
                .collect::<Vec<_>>();
            let x = d
                .upload(
                    &values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let padded = rows.next_multiple_of(32);
            let stage_words = padded * k / 2;
            let stage = d.alloc((stage_words + 16) * 4).unwrap();
            unsafe {
                stage.write_u32(&vec![0xdeadbeef; stage_words + 16]);
            }
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let p = [
                k as u32,
                n as u32,
                rows as u32,
                bits as u32,
                group as u32,
                1,
                start as u32,
            ];
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                baseline,
                &[&w, &x, &y],
                &p,
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            cmd.finish().unwrap();
            let expected = unsafe { y.read_f32(0, count + 32) };
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_input",
                &[&x, &stage],
                &p,
                [(padded * k).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                kernel,
                &[&w, &stage, &y],
                &p,
                dense_grid(kernel, n, rows),
                128,
            );
            cmd.finish().unwrap();
            let got = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(&got[start * n..count], &expected[start * n..count]);
            assert!(
                got[..start * n]
                    .iter()
                    .chain(&got[count..])
                    .all(|v| v.is_nan())
            );
            let staged = unsafe { stage.read_u32(stage_words + 16) };
            assert!(staged[stage_words..].iter().all(|&v| v == 0xdeadbeef));
            assert!(staged[rows * k / 2..stage_words].iter().all(|&v| v == 0));
        }
    }
}

#[test]
#[ignore = "rotated dense projection layout costs; exact output required, not serving timing"]
fn dense_layout_execution_cost() {
    let d = MetalDevice::new(Some(320 << 20)).unwrap();
    assert!(d.tensor_accelerated());
    for (k, n) in [
        (2560usize, 10240usize),
        (6144, 2560),
        (320, 10240),
        (10240, 320),
        (2560, 512),
        (2560, 6144),
        (2560, 640),
        (640, 2560),
    ] {
        let mut raw = (0..k * n / 2)
            .map(|i| (i * 37 + i / 127 + 19) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..k * n / 32).flat_map(|i| {
                let v = if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                };
                half::bf16::from_f32(v).to_le_bytes()
            }));
        }
        let w = d.upload(&raw).unwrap();
        for rows in [64usize, 512, 1024, 2048] {
            let start = 3;
            let x = d
                .upload(
                    &(0..(start + rows) * k)
                        .flat_map(|i| (((i % 137) as f32 / 97.) - 0.7).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let stage = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
            let count = (start + rows) * n;
            let y = d.alloc((count + 32) * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, 4, 32, 1, start as u32];
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_input",
                &[&x, &stage],
                &p,
                [(rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let kernels: &[&str] = if k.is_multiple_of(128) {
                &[
                    "q4a_mm4_device128_group32",
                    "q4a_mm4_device128_pad8",
                    "q4a_mm4_device128_pad16",
                    "q4a_mm4_device128_wide",
                ]
            } else {
                &[
                    "q4a_mm4_device64_group32",
                    "q4a_mm4_device64_pad8",
                    "q4a_mm4_device64_pad16",
                ]
            };
            let mut times = kernels.iter().map(|_| Vec::new()).collect::<Vec<_>>();
            let mut expected = None;
            for round in 0..7 {
                for route in (0..kernels.len()).map(|i| (i + round) % kernels.len()) {
                    poison_output(&y, count);
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernels[route],
                        &[&w, &stage, &y],
                        &p,
                        dense_grid(kernels[route], n, rows),
                        128,
                    );
                    let gpu = cmd.finish().unwrap();
                    let got = unsafe { y.read_f32(0, count + 32) };
                    assert!(
                        got[..start * n]
                            .iter()
                            .chain(&got[count..])
                            .all(|x| x.is_nan())
                    );
                    let got = got[start * n..count]
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>();
                    if let Some(ref want) = expected {
                        assert!(
                            got == *want,
                            "layout changed k={k} n={n} rows={rows} route={route}"
                        );
                    } else {
                        expected = Some(got);
                    }
                    if round > 0 {
                        times[route].push(gpu);
                    }
                }
            }
            eprintln!(
                "FLASH_DENSE_LAYOUT {}",
                serde_json::json!({"k":k,"n":n,"rows":rows,"kernels":kernels,"gpu_seconds":times,"exact":true})
            );
        }
    }
}

#[test]
fn split_staging_preserves_contracts_offsets_and_bounded_arena() {
    split_staging_cases(false);
}

#[test]
#[ignore = "rotated GPU split-projection cost; not a serving benchmark"]
fn split_staging_execution_cost() {
    split_staging_cases(true);
}

fn split_staging_cases(measure: bool) {
    for (bits, group, ty) in [(4, 32, affine::A4G32), (8, 64, affine::A8G64)] {
        split_staging_layout_cases(measure, bits, group, ty);
    }
}

fn split_staging_layout_cases(measure: bool, bits: usize, group: usize, ty: u32) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(false));
            affine::PADDED_TILES_FOR_TEST.with(|v| v.set(true));
            affine::STAGED_ROUTER_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    let d = MetalDevice::new(Some(160 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    affine::STAGED_ROUTER_FOR_TEST.with(|v| v.set(true));
    for (k, n, rows, logical) in [
        (10240usize, 320usize, 64usize, 128usize),
        (10240, 320, 129, 512),
        (10240, 320, 512, 512),
        (10240, 320, 1024, 512),
        (10240, 320, 2048, 512),
        (10240, 320, 812, 812),
        (10240, 320, 1024, 1024),
        (10240, 320, 2048, 1024),
        (2560, 515, 129, 256),
        (2560, 515, 511, 256),
        (2560, 512, 512, 512),
        (2560, 512, 1024, 512),
        (2560, 512, 2048, 512),
        (320, 96, 64, 128),
        (640, 160, 129, 256),
        (640, 2560, 64, 128),
    ] {
        let parts = affine::contraction(k, n, ty, logical).1;
        let mut raw = (0..k * n * bits / 8)
            .map(|i| (i * 37 + 19) as u8)
            .collect::<Vec<_>>();
        for plane in 0..2 {
            raw.extend((0..k * n / group).flat_map(|i| {
                let value = if plane == 0 {
                    (1 + i % 29) as f32 * 0.0003
                } else {
                    -0.04 + (i % 17) as f32 * 0.002
                };
                half::bf16::from_f32(value).to_bits().to_le_bytes()
            }));
        }
        let w = crate::weights::Weight {
            buffer: d.upload(&raw).unwrap(),
            ty,
            k,
            n,
        };
        let start = 3;
        let x = d
            .upload(
                &(0..(start + rows) * k)
                    .flat_map(|i| ((i % 137) as f32 / 97. - 0.7).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let count = (start + rows) * n;
        let y = d.alloc((count + 32) * 4).unwrap();
        let arena = d.alloc(affine::workspace_bytes(2048)).unwrap();
        affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(true));
        affine::PADDED_TILES_FOR_TEST.with(|v| v.set(false));
        poison_output(&y, count);
        let cmd = d.begin().unwrap().with_projection_workspace(&arena);
        affine::project_span(&cmd, &w, &x, &y, rows, start, logical);
        cmd.finish().unwrap();
        let expected = unsafe { y.read_f32(0, count + 32) };
        // Include an arena that forces multiple physical slices and a small
        // arena where input staging must fall back to the original split.
        let arenas = if measure {
            vec![arena.len()]
        } else {
            vec![
                arena.len(),
                rows * n * parts * 2 + 128,
                32 * (k + n * parts) * 2,
            ]
        };
        for bytes in arenas {
            // The fallback's partial-only arena must fit too.
            let bytes = bytes.max(rows * n * parts * 2);
            let scratch = d.alloc(bytes).unwrap();
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            for round in 0..if measure { 8 } else { 1 } {
                for index in 0..3 {
                    let route = (index + round) % 3;
                    affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(route == 0));
                    affine::PADDED_TILES_FOR_TEST.with(|v| v.set(route == 2));
                    poison_output(&y, count);
                    let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                    affine::project_span(&cmd, &w, &x, &y, rows, start, logical);
                    let seconds = cmd.finish().unwrap();
                    let got = unsafe { y.read_f32(0, count + 32) };
                    assert_eq!(
                        &got[start * n..count],
                        &expected[start * n..count],
                        "split staging K={k} N={n} rows={rows} logical={logical} arena={bytes} route={route}"
                    );
                    assert!(
                        got[..start * n]
                            .iter()
                            .chain(&got[count..])
                            .all(|v| v.is_nan())
                    );
                    if round > 0 {
                        times[route].push(seconds);
                    }
                }
            }
            if measure {
                eprintln!(
                    "FLASH_SPLIT_STAGING {}",
                    serde_json::json!({
                    "bits":bits,"group":group,"k":k,"n":n,"rows":rows,"logical":logical,"parts":parts,
                    "arena":bytes,"gpu_seconds":times,"exact":true})
                );
            }
        }
    }
}

fn grouped_matrix_case(
    d: &MetalDevice,
    w: &crate::weights::Weight,
    x: &crate::device::Buffer,
    ids: &crate::device::Buffer,
    y: &crate::device::Buffer,
    expected: &[u8],
    base: &str,
    rows: usize,
    count: usize,
    per_entry: bool,
) {
    let k = w.k;
    let n = w.n / 512;
    assert_eq!(expected.len(), count * 4);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mv",
        &[&w.buffer, x, ids, y],
        &[
            k as u32,
            n as u32,
            (rows * 10) as u32,
            u32::from(per_entry),
            512,
        ],
        [n.div_ceil(16), rows * 10, 1],
        128,
    );
    let baseline_seconds = cmd.finish().unwrap();
    let vector = unsafe { y.read_f32(0, count) };
    poison_output(y, count);
    let entries = rows * 10;
    let lists = d.alloc(512 * entries * 4).unwrap();
    let counts = d.alloc(512 * 4).unwrap();
    let tiles = d.alloc((1 + 2 * (entries.div_ceil(32) + 512)) * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "moe_align",
        &[ids, &lists, &counts],
        &[entries as u32],
        [512, 1, 1],
        256,
    );
    cmd.dispatch("iq_tiles512", &[&counts, &tiles], &[512], [1, 1, 1], 512);
    let mut params = vec![
        k as u32,
        n as u32,
        entries as u32,
        u32::from(per_entry),
        512,
    ];
    params.extend([u32::MAX; affine::MAX_ROWS.div_ceil(32)]);
    cmd.dispatch(
        "q4a_expert_mm",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(32), entries.div_ceil(32) + 512, 1],
        128,
    );
    let seconds = cmd.finish().unwrap();
    let matrix = unsafe { y.read_f32(0, count + 32) };
    assert!(matrix[count..].iter().all(|v| v.is_nan()));
    poison_output(y, count);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mm_wide",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
        128,
    );
    let wide_seconds = cmd.finish().unwrap();
    let wide = unsafe { y.read_f32(0, count + 32) };
    assert!(wide[count..].iter().all(|v| v.is_nan()));
    assert!(
        matrix[..count] == wide[..count],
        "wide expert matrix changed arithmetic"
    );
    eprintln!(
        "AFFINE_GROUPED_WIDE {base} rows={rows} per_entry={per_entry} matrix_s={seconds} wide_s={wide_seconds}"
    );
    let kernels = [
        "q4a_expert_mm_wide",
        "q4a_expert_mm_tail",
        "q4a_expert_mm_group32",
        "q4a_expert_mm_group32_pad",
        "q4a_expert_mm_group32_packed",
        "q4a_expert_mm_direct",
        "q4a_expert_mm_direct64",
    ];
    let mut times = std::array::from_fn::<_, 7, _>(|_| Vec::new());
    let input_rows = if per_entry { entries } else { rows };
    let staged = d.alloc(input_rows.div_ceil(32) * 32 * k * 2).unwrap();
    let sorted = d.alloc(entries * k * 2).unwrap();
    let offsets = d.alloc(512 * 4).unwrap();
    let wide_tiles = d.alloc(tiles.len()).unwrap();
    for round in 0..7 {
        for i in 0..kernels.len() {
            let route = (i + round) % kernels.len();
            poison_output(y, count);
            let cmd = d.begin().unwrap();
            let input = if route == 4 {
                cmd.dispatch(
                    "q4a_input",
                    &[x, &staged],
                    &[k as u32, n as u32, input_rows as u32, 0, 0, 0, 0],
                    [(input_rows.div_ceil(32) * 32 * k).div_ceil(256), 1, 1],
                    256,
                );
                &staged
            } else {
                x
            };
            if route >= 5 {
                let tiles = if route == 6 {
                    cmd.dispatch(
                        "q4a_expert_plan64",
                        &[&counts, &offsets, &wide_tiles],
                        &[512],
                        [1, 1, 1],
                        512,
                    );
                    &wide_tiles
                } else {
                    cmd.dispatch(
                        "q4a_expert_offsets",
                        &[&counts, &offsets],
                        &[512],
                        [1, 1, 1],
                        512,
                    );
                    &tiles
                };
                cmd.dispatch(
                    if route == 6 {
                        "q4a_expert_pack64"
                    } else {
                        "q4a_expert_pack"
                    },
                    &[x, &lists, &counts, tiles, &offsets, &sorted],
                    &params,
                    [k.div_ceil(512), entries.div_ceil(32) + 512, 1],
                    128,
                );
                cmd.dispatch(
                    kernels[route],
                    &[&w.buffer, &sorted, &lists, &counts, tiles, &offsets, y],
                    &params,
                    [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                    128,
                );
            } else {
                cmd.dispatch(
                    kernels[route],
                    &[&w.buffer, input, &lists, &counts, &tiles, y],
                    &params,
                    [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
                    128,
                );
            }
            let elapsed = cmd.finish().unwrap();
            let candidate = unsafe { y.read_f32(0, count + 32) };
            assert!(candidate[count..].iter().all(|v| v.is_nan()));
            assert!(
                candidate[..count] == matrix[..count],
                "expert tile changed arithmetic: {} {base} rows={rows} per_entry={per_entry}",
                kernels[route]
            );
            if round > 0 {
                times[route].push(elapsed);
            }
        }
    }
    eprintln!(
        "AFFINE_EXPERT_TILES {}",
        serde_json::json!({"base":base,"rows":rows,"per_entry":per_entry,"kernels":kernels,"gpu_seconds":times})
    );
    for row in (0..rows).step_by(7) {
        params[5 + row / 32] &= !(1 << (row % 32));
    }
    poison_output(y, count);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4a_expert_mm_tail",
        &[&w.buffer, x, &lists, &counts, &tiles, y],
        &params,
        [n.div_ceil(64), entries.div_ceil(32) + 512, 1],
        128,
    );
    cmd.dispatch(
        "q4a_expert_vector_masked",
        &[&w.buffer, x, ids, y],
        &params,
        [n.div_ceil(16), entries, 1],
        128,
    );
    cmd.finish().unwrap();
    let mixed = unsafe { y.read_f32(0, count + 32) };
    assert!(mixed[count..].iter().all(|v| v.is_nan()));
    for row in 0..rows {
        let expected = if row % 7 == 0 { &vector } else { &matrix };
        let span = row * 10 * n..(row + 1) * 10 * n;
        assert!(
            mixed[span.clone()] == expected[span],
            "mixed expert contract changed row {row}"
        );
    }
    let mut max_error = 0f32;
    let mut unequal = 0;
    let mut peak = 0f32;
    for (&a, b) in matrix[..count].iter().zip(
        expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b)),
    ) {
        assert!(a.is_finite() && b.is_finite());
        max_error = max_error.max((a - b).abs());
        peak = peak.max(b.abs());
        unequal += usize::from(a != b);
    }
    eprintln!(
        "AFFINE_GROUPED {base} rows={rows} baseline_s={baseline_seconds} matrix_s={seconds} error={max_error} peak={peak} unequal={unequal}/{count}"
    );
    assert_eq!(
        unequal, 0,
        "same-checkpoint grouped expert operation mismatch"
    );
}

#[test]
#[ignore = "same-checkpoint MLX GPU fixtures: PADDOCK_FLASH_NEXT_MLX_AFFINE_REFERENCE"]
fn flash_next_affine_matches_mlx_gpu() {
    let dir = std::env::var("PADDOCK_FLASH_NEXT_MLX_AFFINE_REFERENCE").unwrap();
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let source =
        ShardedSafetensors::open_dir(Path::new(manifest["model"].as_str().unwrap())).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let scratch = d.alloc(affine::WORKSPACE_BYTES).unwrap();
    let mut failures = Vec::new();
    for c in manifest["cases"].as_array().unwrap() {
        let k = c["k"].as_u64().unwrap() as usize;
        let n = c["n"].as_u64().unwrap() as usize;
        let rows = c["rows"].as_u64().unwrap() as usize;
        let experts = c["expert"].as_bool().unwrap();
        let shape = if experts { vec![512, n, k] } else { vec![n, k] };
        let ty = if c["bits"] == 8 {
            affine::A8G64
        } else {
            affine::A4G32
        };
        let base = c["base"].as_str().unwrap();
        eprintln!(
            "AFFINE_CASE file={} input={}",
            c["file"],
            c.get("input").and_then(|v| v.as_str()).unwrap_or("sine")
        );
        let w = affine::load(&d, &source, base, &shape, ty).unwrap();
        let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
        let x = d.upload(f.bytes("x").unwrap().1).unwrap();
        let count = rows * n * if experts { 10 } else { 1 };
        let y = d
            .upload(
                &(0..count + 32)
                    .flat_map(|_| f32::NAN.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let ids = if experts {
            Some(d.upload(f.bytes("ids").unwrap().1).unwrap())
        } else {
            None
        };
        if experts && rows > 128 {
            // Repeat the independent 1024-row fixture to exercise the actual
            // 2048-row serving election. Repeat its complete reference too;
            // never silently zip a half-length expected output.
            for copies in 1..=if rows == 1024 { 2 } else { 1 } {
                for per_entry in [false, true] {
                    let Some((_, raw)) = f.bytes(if per_entry { "x_entry" } else { "x" }) else {
                        continue;
                    };
                    let input = d.upload(&raw.repeat(copies)).unwrap();
                    let ids = d.upload(&f.bytes("ids").unwrap().1.repeat(copies)).unwrap();
                    let expected = f
                        .bytes(if per_entry {
                            "grouped_entry"
                        } else {
                            "grouped"
                        })
                        .unwrap()
                        .1
                        .repeat(copies);
                    let y = d.alloc((count * copies + 32) * 4).unwrap();
                    grouped_matrix_case(
                        &d,
                        &w,
                        &input,
                        &ids,
                        &y,
                        &expected,
                        base,
                        rows * copies,
                        count * copies,
                        per_entry,
                    );
                }
            }
            continue;
        }
        let cmd = d.begin().unwrap();
        if let Some(ids) = &ids {
            cmd.dispatch(
                "q4a_expert_mv",
                &[&w.buffer, &x, ids, &y],
                &[k as u32, n as u32, (rows * 10) as u32, 0, 512],
                [n.div_ceil(16), rows * 10, 1],
                128,
            );
        } else if rows >= 13 && n > 48 {
            // Keep the original scalar-staged matrix as an independent
            // exactness seam for split-K and rejected tile candidates.
            let group = c["group"].as_u64().unwrap() as usize;
            let mut parts = (512 / (n.div_ceil(32) * rows.div_ceil(32)))
                .min(k / group.max(32))
                .max(1);
            while !k.is_multiple_of(parts * group.max(32)) {
                parts -= 1;
            }
            cmd.dispatch(
                "q4a_mm",
                &[&w.buffer, &x, &y],
                &[
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    group as u32,
                    parts as u32,
                    0,
                ],
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
        } else {
            affine::project(&cmd, &w, &x, &y, rows);
        }
        let baseline_seconds = cmd.finish().unwrap();
        let actual = unsafe { y.read_f32(0, count + 32) };
        eprintln!("AFFINE_TIME {base} rows={rows} experts={experts} gpu_s={baseline_seconds}");
        if let Some(ids) = &ids {
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            affine::experts(&cmd, &w, &x, ids, &y, rows * 10, false);
            let specialized_seconds = cmd.finish().unwrap();
            let specialized = unsafe { y.read_f32(0, count) };
            assert_eq!(
                &actual[..count],
                &specialized,
                "expert step specialization changed arithmetic {base} rows={rows}"
            );
            eprintln!(
                "AFFINE_EXPERT_ENTRY {base} rows={rows} baseline_s={baseline_seconds} specialized_s={specialized_seconds}"
            );
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_expert_order",
                &[ids, &scratch],
                &[(rows * 10) as u32],
                [1, 1, 1],
                256,
            );
            affine::experts_ordered(&cmd, &w, &x, ids, &y, rows * 10, false, Some(&scratch));
            let ordered_seconds = cmd.finish().unwrap();
            let ordered = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &ordered[..count],
                "expert ordering changed arithmetic {base} rows={rows}"
            );
            assert!(ordered[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_ORDER {base} rows={rows} baseline_s={baseline_seconds} ordered_s={ordered_seconds}"
            );
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if k.is_multiple_of(512) {
                    "q4a_expert4_fast_pair"
                } else {
                    "q4a_expert4_pair"
                },
                &[&w.buffer, &x, ids, &y, &scratch],
                &[k as u32, n as u32, (rows * 10) as u32, 0, 512],
                [n.div_ceil(16) * 4, (rows * 10).div_ceil(8), 1],
                128,
            );
            let pair_seconds = cmd.finish().unwrap();
            let paired = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &paired[..count],
                "expert pair changed arithmetic {base} rows={rows}"
            );
            assert!(paired[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_PAIR {base} rows={rows} baseline_s={baseline_seconds} ordered_s={ordered_seconds} pair_s={pair_seconds}"
            );
        }
        if !experts && rows == 4 {
            let cmd = d.begin().unwrap().with_independent_rows(true);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let batched = unsafe { y.read_f32(0, count) };
            for row in 0..rows {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[row * k * 4..(row + 1) * k * 4])
                    .unwrap();
                let out = d.alloc(n * 4).unwrap();
                let cmd = d.begin().unwrap();
                affine::project(&cmd, &w, &input, &out, 1);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, n) },
                    batched[row * n..(row + 1) * n],
                    "independent rows changed singleton contraction {base} row={row}"
                );
            }
        }
        if !experts && rows > 1 {
            // Independent launches retain the pre-coalescing implementation
            // as the oracle, including unaligned starts and tiny physical
            // slices of a large logical contract.
            let mut spans = Vec::new();
            let mut at = 0;
            for len in [1, 7, 23, 33, rows / 4, rows] {
                let count = len.min(rows - at);
                if count > 0 {
                    spans.push((at, count, rows));
                    at += count;
                }
            }
            poison_output(&y, count);
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(&spans);
            affine::project(&cmd, &w, &x, &y, rows);
            let merged_seconds = cmd.finish().unwrap();
            let merged = unsafe { y.read_f32(0, count + 32) };
            assert!(merged[count..].iter().all(|v| v.is_nan()));
            poison_output(&y, count);
            let mut separate_seconds = 0.;
            for &span in &spans {
                let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                affine::project_span(&cmd, &w, &x, &y, span.1, span.0, span.2);
                separate_seconds += cmd.finish().unwrap();
            }
            let separate = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &merged[..count],
                &separate[..count],
                "dense coalescing changed {base} rows={rows}"
            );
            assert!(separate[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_COALESCED {base} rows={rows} merged_s={merged_seconds} separate_s={separate_seconds}"
            );
        }
        if !experts && [4, 128].contains(&rows) {
            let spans = if rows == 4 {
                vec![(0, 1, 1), (1, 3, 3)]
            } else {
                vec![(0, 1, 1), (1, 33, 33), (34, 94, 94)]
            };
            let cmd = d
                .begin()
                .unwrap()
                .with_projection_workspace(&scratch)
                .with_projection_rows(&spans);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let mixed = unsafe { y.read_f32(0, count + 32) };
            assert!(mixed[count..].iter().all(|v| v.is_nan()));
            for &(start, len, _) in &spans {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[start * k * 4..(start + len) * k * 4])
                    .unwrap();
                let out = d.alloc(len * n * 4).unwrap();
                let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
                affine::project(&cmd, &w, &input, &out, len);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, len * n) },
                    mixed[start * n..(start + len) * n],
                    "ragged projection changed per-sequence contraction {base} span={start}/{len}"
                );
            }
        }
        if !experts && rows >= 13 && n > 48 {
            if affine::contraction(k, n, ty, rows).1 == 1 && k.is_multiple_of(64) {
                let staged = d.alloc(rows.next_multiple_of(32) * k * 2).unwrap();
                let kernel = match (ty, k.is_multiple_of(128)) {
                    (affine::A4G32, true) => "q4a_mm4_device128",
                    (affine::A4G32, false) => "q4a_mm4_device64",
                    (_, true) => "q4a_mm8_device128",
                    (_, false) => "q4a_mm8_device64",
                };
                let group_kernel = match (ty, k.is_multiple_of(128)) {
                    (affine::A4G32, true) => "q4a_mm4_device128_group32",
                    (affine::A4G32, false) => "q4a_mm4_device64_group32",
                    _ => kernel,
                };
                let params = [
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    c["group"].as_u64().unwrap() as u32,
                    1,
                    0,
                ];
                let mut times = [Vec::new(), Vec::new(), Vec::new()];
                for round in 0..7 {
                    for index in 0..3 {
                        let route = (index + round) % 3;
                        poison_output(&y, count);
                        let cmd = d.begin().unwrap();
                        if route > 0 {
                            cmd.dispatch(
                                "q4a_input",
                                &[&x, &staged],
                                &params,
                                [(rows.next_multiple_of(32) * k).div_ceil(256), 1, 1],
                                256,
                            );
                            cmd.dispatch(
                                if route == 2 { group_kernel } else { kernel },
                                &[&w.buffer, &staged, &y],
                                &params,
                                [n.div_ceil(32), rows.div_ceil(32), 1],
                                128,
                            );
                        } else {
                            cmd.dispatch(
                                if ty == affine::A4G32 {
                                    "q4a_mm4_packed"
                                } else {
                                    "q4a_mm8_packed"
                                },
                                &[&w.buffer, &x, &y],
                                &params,
                                [n.div_ceil(32), rows.div_ceil(32), 1],
                                128,
                            );
                        }
                        let elapsed = cmd.finish().unwrap();
                        let got = unsafe { y.read_f32(0, count + 32) };
                        assert!(got[count..].iter().all(|v| v.is_nan()));
                        assert!(
                            got[..count] == actual[..count],
                            "device input changed {base} rows={rows} route={route}"
                        );
                        if round > 0 {
                            times[route].push(elapsed);
                        }
                    }
                }
                eprintln!(
                    "AFFINE_DEVICE_INPUT {}",
                    serde_json::json!({"base":base,"rows":rows,"gpu_seconds":times})
                );
            }
            poison_output(&y, count);
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            affine::project(&cmd, &w, &x, &y, rows);
            let split_seconds = cmd.finish().unwrap();
            let parallel = unsafe { y.read_f32(0, count + 32) };
            assert_eq!(
                &actual[..count],
                &parallel[..count],
                "parallel split-K changed native BF16 arithmetic: {base} rows={rows}"
            );
            assert!(parallel[count..].iter().all(|v| v.is_nan()));
            eprintln!(
                "AFFINE_SPLIT {base} rows={rows} baseline_s={baseline_seconds} parallel_s={split_seconds}"
            );
        }
        if !experts && rows >= 812 && n > 48 {
            // All checkpoint planes here have a single K partition. Retain
            // the reordered packed kernel as a GPU cache-ordering ablation.
            poison_output(&y, count);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                if c["bits"] == 4 {
                    "q4a_mm4_reuse"
                } else {
                    "q4a_mm8_reuse"
                },
                &[&w.buffer, &x, &y],
                &[
                    k as u32,
                    n as u32,
                    rows as u32,
                    c["bits"].as_u64().unwrap() as u32,
                    c["group"].as_u64().unwrap() as u32,
                    1,
                    0,
                ],
                [n.div_ceil(32), rows.div_ceil(32), 1],
                128,
            );
            let packed_seconds = cmd.finish().unwrap();
            let packed = unsafe { y.read_f32(0, count + 32) };
            assert!(
                actual[..count] == packed[..count],
                "packed dense changed {base}"
            );
            assert!(packed[count..].iter().all(|v| v.is_nan()));
            eprintln!("AFFINE_REUSE {base} rows={rows} gpu_s={packed_seconds}");
        }
        if !experts && [4, 128].contains(&rows) {
            let cmd = d.begin().unwrap().with_projection_workspace(&scratch);
            affine::project(&cmd, &w, &x, &y, rows);
            cmd.finish().unwrap();
            let whole = unsafe { y.read_f32(0, count) };
            let slices = if rows == 4 {
                vec![(0, 1), (1, 3)]
            } else {
                vec![(0, 1), (1, 7), (8, 23), (31, 33), (64, 63), (127, 1)]
            };
            for (start, len) in slices {
                let input = d
                    .upload(&f.bytes("x").unwrap().1[start * k * 4..(start + len) * k * 4])
                    .unwrap();
                let out = d.alloc(len * n * 4).unwrap();
                let spans = [(0, len, rows)];
                let cmd = d
                    .begin()
                    .unwrap()
                    .with_projection_workspace(&scratch)
                    .with_projection_rows(&spans);
                affine::project(&cmd, &w, &input, &out, len);
                cmd.finish().unwrap();
                assert_eq!(
                    unsafe { out.read_f32(0, len * n) },
                    whole[start * n..(start + len) * n],
                    "logical projection changed when sliced: {base} span={start}/{len}/{rows}"
                );
            }
        }
        assert!(
            actual[count..].iter().all(|v| v.is_nan()),
            "output guard {base}"
        );
        let expected = f
            .bytes("y")
            .unwrap()
            .1
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b));
        let mut error = 0f32;
        let mut peak = 0f32;
        let mut unequal = 0;
        for (&a, b) in actual[..count].iter().zip(expected) {
            assert!(a.is_finite() && b.is_finite());
            assert_eq!(a.to_bits() & 0xffff, 0);
            error = error.max((a - b).abs());
            peak = peak.max(b.abs());
            unequal += usize::from(a != b);
        }
        eprintln!("AFFINE {base} m={rows} error={error} peak={peak} unequal={unequal}/{count}");
        // An operation-format bound is not a generation-parity claim. Tiny
        // row-invariant projections must match exactly; all model outputs
        // still need their separate greedy/logit qualification.
        if ((n <= 48 || rows == 1 || experts) && unequal != 0) || error > peak * 0.008 + 0.0001 {
            failures.push(format!(
                "{base} rows={rows} error={error} unequal={unequal}"
            ));
        }
    }
    let c = &manifest["gather"];
    let base = c["base"].as_str().unwrap();
    let (info, _) = source.bytes(&format!("{base}.weight")).unwrap();
    let w = affine::load(&d, &source, base, &[info.shape[0], 160], affine::A4G32).unwrap();
    let f = SafetensorsFile::open(&dir.join(c["file"].as_str().unwrap())).unwrap();
    let ids = d.upload(f.bytes("ids").unwrap().1).unwrap();
    let y = d.alloc(7 * 160 * 4).unwrap();
    let cmd = d.begin().unwrap();
    affine::gather(&cmd, &w, &ids, &y, 7);
    cmd.finish().unwrap();
    let actual = unsafe { y.read_f32(0, 7 * 160) };
    let expected = f
        .bytes("y")
        .unwrap()
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "same-checkpoint PLE shard gather");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn flash_next_mlx_expert_order_is_stable_bounded_permutation() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    for entries in [1, 10, 40, 64, 90, 330, 1280, 1290, 1600, 1920, 2040, 2048] {
        for invalid in [false, true] {
            let ids = (0..entries)
                .map(|i| {
                    if invalid && i % 13 == 0 {
                        512
                    } else {
                        (i * 73 + 511) % 512
                    }
                })
                .collect::<Vec<u32>>();
            let input = d
                .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                .unwrap();
            let out = d.upload(&vec![0xA5; (entries as usize + 16) * 4]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "q4a_expert_order",
                &[&input, &out],
                &[entries],
                [1, 1, 1],
                256,
            );
            cmd.finish().unwrap();
            let actual = unsafe { out.read_u32(entries as usize + 16) };
            // Control-index verification only; no host model arithmetic.
            let mut expected = (0..entries)
                .filter(|&i| ids[i as usize] < 512)
                .collect::<Vec<_>>();
            expected.sort_by_key(|&i| (ids[i as usize], i));
            expected.resize(entries as usize, u32::MAX);
            assert_eq!(&actual[..entries as usize], expected);
            assert!(actual[entries as usize..].iter().all(|&v| v == 0xA5A5A5A5));
        }
    }
}

#[test]
fn extended_expert_order_preserves_bits_routes_and_guards() {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    // Full 512-expert address layout, with fewer independent output columns.
    // Full-model tests additionally check the actual 640/2560-wide weights.
    let n: usize = 16;
    for k in [640usize, 2560] {
        let size = k * n * 512;
        let mut raw = (0..size / 2)
            .map(|i| (i * 37 + i / 131) as u8)
            .collect::<Vec<_>>();
        for bias in [false, true] {
            raw.extend((0..size / 32).flat_map(|i| {
                half::bf16::from_f32(if bias {
                    -0.04 + (i % 17) as f32 * 0.001
                } else {
                    0.002 + (i % 31) as f32 * 0.00002
                })
                .to_le_bytes()
            }));
        }
        let w = d.upload(&raw).unwrap();
        for rows in [129usize, 160, 192, 204] {
            let entries = rows * 10;
            let count = entries * n;
            let x = d
                .upload(
                    &(0..entries * k)
                        .flat_map(|i| {
                            half::bf16::from_f32((i % 137) as f32 / 97. - 0.7)
                                .to_f32()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            for routing in 0..3 {
                let ids = (0..entries)
                    .map(|i| match routing {
                        1 => 511,
                        2 if i % 7 == 0 => u32::MAX,
                        _ => ((i * 73 + i / 10 * 17) % 512) as u32,
                    })
                    .collect::<Vec<_>>();
                let input = d
                    .upload(&ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap();
                let order = d.upload(&vec![0xA5; (entries + 32) * 4]).unwrap();
                let y = d.alloc((count + 32) * 4).unwrap();
                let allocated = d.allocated_bytes();
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "q4a_expert_order",
                    &[&input, &order],
                    &[entries as u32],
                    [1, 1, 1],
                    256,
                );
                cmd.finish().unwrap();
                assert!(
                    unsafe { order.read_u32(entries + 32) }[entries..]
                        .iter()
                        .all(|&v| v == 0xA5A5A5A5)
                );
                for per_entry in [false, true] {
                    let mut expected = None;
                    for route in 0..3 {
                        poison_output(&y, count);
                        let cmd = d.begin().unwrap();
                        let name = match (k, route) {
                            (2560, 0) => "q4a_expert4_fast",
                            (2560, 1) => "q4a_expert4_fast_ordered",
                            (2560, _) => "q4a_expert4_fast_pair",
                            (_, 0) => "q4a_expert4",
                            (_, 1) => "q4a_expert4_ordered",
                            _ => "q4a_expert4_pair",
                        };
                        cmd.dispatch(
                            name,
                            &[&w, &x, &input, &y, &order],
                            &[
                                k as u32,
                                n as u32,
                                entries as u32,
                                u32::from(per_entry),
                                512,
                            ],
                            match route {
                                0 => [n / 16, entries, 1],
                                1 => [n / 16 * 8, entries.div_ceil(8), 1],
                                _ => [n / 16 * 4, entries.div_ceil(8), 1],
                            },
                            128,
                        );
                        cmd.finish().unwrap();
                        let actual = unsafe { y.read_u32(count + 32) };
                        assert!(actual[count..].iter().all(|&v| v == f32::NAN.to_bits()));
                        for (i, &id) in ids.iter().enumerate() {
                            assert!(actual[i * n..(i + 1) * n].iter().all(|&v| {
                                if id >= 512 {
                                    v == f32::NAN.to_bits()
                                } else {
                                    f32::from_bits(v).is_finite()
                                }
                            }));
                        }
                        if let Some(expected) = &expected {
                            assert!(
                                &actual == expected,
                                "k={k} rows={rows} routing={routing} per_entry={per_entry} route={route}"
                            );
                        } else {
                            expected = Some(actual);
                        }
                    }
                }
                assert_eq!(d.allocated_bytes(), allocated);
                assert_eq!(unsafe { input.read_u32(entries) }, ids);
            }
        }
    }
}
