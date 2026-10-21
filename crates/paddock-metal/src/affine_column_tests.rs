use crate::device::MetalDevice;

#[test]
fn affine_column_reuse_is_exact_and_bounded() {
    probe(false);
}

#[test]
#[ignore = "Gemma one-vector column reuse: paired GPU cost, not serving throughput"]
fn affine_column_reuse_cost() {
    probe(true);
}

fn probe(timed: bool) {
    column_probe(timed, false);
}

#[test]
fn affine_wide_columns_are_exact_and_bounded() {
    column_probe(false, true);
}

#[test]
#[ignore = "Gemma wide-input column occupancy: GPU diagnostic, not serving throughput"]
fn affine_wide_columns_cost() {
    column_probe(true, true);
}

fn column_probe(timed: bool, wide: bool) {
    let d = MetalDevice::new(None).expect("Metal column-reuse device");
    let shapes = if wide {
        (if timed {
            vec![8192, 16384, 21504]
        } else {
            vec![512, 1536, 8192, 16384, 21504]
        })
        .into_iter()
        .map(|k| (k, if timed { vec![5376] } else { vec![65, 17, 1] }))
        .collect()
    } else if timed {
        vec![
            (5376, vec![21504, 21504]),
            (5376, vec![8192, 4096, 4096]),
            (5376, vec![262144]),
        ]
    } else {
        vec![
            (64, vec![1, 33, 65]),
            (192, vec![17]),
            (256, vec![1, 17, 33]),
            (320, vec![33, 65]),
            (512, vec![65]),
            (5376, vec![65, 33, 17]),
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
                        half::bf16::from_f32(if bias {
                            -((i % 499 + 1) as f32) / 113.
                        } else {
                            (i % 997 + 1) as f32 / 9973.
                        })
                        .to_bits()
                        .to_le_bytes()
                    }));
                }
                d.upload(&bytes).expect("column-reuse weights")
            })
            .collect();
        let x = d
            .upload(
                &(0..k)
                    .flat_map(|i| (((i * 37) % 1999) as f32 / 113. - 8.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("column-reuse activations");
        let outputs: Vec<_> = ns
            .iter()
            .map(|&n| {
                d.alloc((n + 32) * 4)
                    .expect("column-reuse output and guards")
            })
            .collect();
        let mut expected = Vec::new();
        let (variants, columns) = if wide {
            (
                [
                    "mlx_affine_single",
                    "mlx_affine_single_columns2",
                    "mlx_affine_single_columns1",
                ],
                [16, 8, 4],
            )
        } else {
            (
                ["mlx_affine1", "mlx_affine_columns4", "mlx_affine_columns1"],
                [32, 64, 16],
            )
        };
        let mut variants: Vec<_> = variants.into_iter().zip(columns).collect();
        if !wide {
            variants.push(("mlx_affine_pipeline4", 16));
            if k.is_multiple_of(256) {
                variants.push(("mlx_affine_pipeline_full4", 16));
            }
        }
        let mut times = vec![Vec::new(); variants.len()];
        for round in 0..if timed { 10 } else { 2 } {
            for order in 0..variants.len() {
                let at = (order + round) % variants.len();
                unsafe {
                    for (o, &n) in outputs.iter().zip(&ns) {
                        o.write_u32(&vec![u32::MAX; n + 32]);
                    }
                }
                let cmd = d.begin().expect("column-reuse command");
                for _ in 0..if timed { 8 } else { 1 } {
                    let w1 = weights.get(1).unwrap_or(&weights[0]);
                    let w2 = weights.get(2).unwrap_or(w1);
                    let o1 = outputs.get(1).unwrap_or(&outputs[0]);
                    let o2 = outputs.get(2).unwrap_or(o1);
                    let bufs = [&weights[0], w1, w2, &x, &outputs[0], o1, o2];
                    cmd.dispatch_at(
                        variants[at].0,
                        &bufs,
                        &[0, 0, 0, 0, 32, 32, 32],
                        &[
                            k as u32,
                            ns[0] as u32,
                            ns.get(1).copied().unwrap_or(0) as u32,
                            ns.get(2).copied().unwrap_or(0) as u32,
                            1,
                        ],
                        [ns.iter().map(|n| n.div_ceil(variants[at].1)).sum(), 1, 1],
                        128,
                    );
                }
                let elapsed = cmd.finish().expect("column-reuse GPU completion")
                    / if timed { 8. } else { 1. };
                let actual: Vec<_> = outputs
                    .iter()
                    .zip(&ns)
                    .map(|(out, &n)| {
                        let values = unsafe { out.read_u32(n + 32) };
                        assert!(
                            values[8..8 + n]
                                .iter()
                                .all(|v| f32::from_bits(*v).is_finite())
                        );
                        assert!(
                            values[..8]
                                .iter()
                                .chain(&values[8 + n..])
                                .all(|v| *v == u32::MAX)
                        );
                        values
                    })
                    .collect();
                if expected.is_empty() {
                    expected = actual;
                } else {
                    let differences: Vec<_> = actual
                        .iter()
                        .flatten()
                        .zip(expected.iter().flatten())
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .take(8)
                        .collect();
                    assert!(
                        differences.is_empty(),
                        "k={k} ns={ns:?} candidate={at}: {differences:?}"
                    );
                }
                if round > 0 {
                    times[at].push(elapsed * 1e6);
                }
            }
        }
        if timed {
            eprintln!(
                "AFFINE_COLUMN_COST {}",
                serde_json::json!({"k":k,"ns":ns,"variants":variants,"times_us":times})
            );
        }
    }
}
