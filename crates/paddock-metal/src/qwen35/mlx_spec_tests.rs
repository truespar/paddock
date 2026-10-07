use super::*;

fn same_logits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let differences: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (a, b))| !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits())
        .map(|(i, (a, b))| (i, *a, *b))
        .collect();
    assert!(
        differences.is_empty(),
        "{label}: {} different logits; max error {}; first {:?}",
        differences.len(),
        differences
            .iter()
            .map(|(_, a, b)| (a - b).abs())
            .fold(0.0f32, f32::max),
        &differences[..differences.len().min(4)]
    );
}

fn fresh(model: &mut Qwen35, slot: usize, prompt: &[u32]) -> Vec<f32> {
    model.reset();
    for cache in &mut model.cache {
        cache.table.clear(&mut model.pool);
        cache.history.clear();
    }
    model.forward_prefill(slot, prompt).unwrap()
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL and PADDOCK_METAL_DFLASH_MODEL"]
fn mlx_zero_draft_decode_preserves_picks_state_and_drafter() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    model.attach_dflash(Path::new(&draft)).unwrap();
    for length in [31usize, 255, 2047] {
        for slots in [vec![0], vec![0, 3], vec![0, 2, 3], vec![0, 1, 2, 3]] {
            let prompt: Vec<u32> = (1000..1000 + length as u32).collect();
            let init = |model: &mut Qwen35| {
                fresh(model, slots[0], &prompt);
                for &slot in &slots[1..] {
                    model.forward_prefill(slot, &prompt).unwrap();
                }
            };
            let reqs: Vec<_> = slots
                .iter()
                .map(|&slot| (slot, length, vec![5000 + slot as u32]))
                .collect();
            let next: Vec<_> = slots
                .iter()
                .map(|&slot| (slot, 8000 + slot as u32, (length + 1) as u32))
                .collect();
            let pendings: Vec<_> = slots
                .iter()
                .map(|&slot| (slot, 9000 + slot as u32))
                .collect();
            let selected: Vec<_> = (0..slots.len()).collect();
            init(&mut model);
            let logits = model.verify(&reqs).unwrap();
            let picks: Vec<_> = logits
                .chunks(model.vocab)
                .map(|r| {
                    let mut best = 0;
                    for i in 1..r.len() {
                        if r[i] > r[best] {
                            best = i;
                        }
                    }
                    best as u32
                })
                .collect();
            model.commit_verify(&vec![1; slots.len()]).unwrap();
            let expected = model.execute(&next, &selected).unwrap();
            let expected_draft = model.dflash_draft(&pendings, 1).unwrap();
            init(&mut model);
            let got = model.forward_spec_batch(&reqs).unwrap().unwrap();
            assert_eq!(got, picks, "length={length},slots={slots:?}");
            assert!(model.spec.as_ref().unwrap().round.is_none());
            for &slot in &slots {
                assert_eq!(
                    model.slots[slot].history.last(),
                    Some(&(5000 + slot as u32))
                );
            }
            let got = model.execute(&next, &selected).unwrap();
            same_logits(
                &got,
                &expected,
                &format!("direct greedy continuation length={length},slots={slots:?}"),
            );
            assert_eq!(
                model.dflash_draft(&pendings, 1).unwrap(),
                expected_draft,
                "draft conditioning"
            );
            let history = model.slots[slots[0]].history.clone();
            assert!(
                model
                    .decode_picks(&[(slots[0], length + 2, vec![1, 2])])
                    .is_err()
            );
            assert_eq!(model.slots[slots[0]].history, history);
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL and PADDOCK_METAL_DFLASH_MODEL"]
fn mlx_copy_and_neural_sources_preserve_committed_state() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 1024, 1, None).unwrap();
    model.attach_dflash(Path::new(&draft)).unwrap();
    let mut prompt: Vec<u32> = (1000..1500).collect();
    prompt.extend([
        10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 99, 10, 11, 12, 13, 14,
    ]);
    let sequence = [15, 21, 22, 23, 24, 25];
    for accepted in 1..=sequence.len() {
        fresh(&mut model, 0, &prompt);
        let mut expected = Vec::new();
        for &t in &sequence {
            expected.extend(model.forward(t).unwrap());
        }
        fresh(&mut model, 0, &prompt);
        let proposals = model
            .spec_draft_batch(&[(0, 15)], lookup::MAX_DRAFT)
            .unwrap()
            .unwrap();
        assert_eq!(proposals, vec![vec![21, 22, 23, 24, 25]]);
        let got = model
            .forward_spec_verify(&[(0, prompt.len(), sequence.to_vec())])
            .unwrap()
            .unwrap();
        same_logits(
            &got,
            &expected,
            "copy-source verification versus serial target",
        );
        model.spec_commit(&[accepted as u32]).unwrap();
        assert_eq!(
            &model.slots[0].history[prompt.len()..],
            &sequence[..accepted]
        );
        // No source match: the model drafter is capped, but its trained
        // noncausal window and conditioning remain intact after rollback.
        let neural = model
            .spec_draft_batch(&[(0, 54321)], lookup::MAX_DRAFT)
            .unwrap()
            .unwrap();
        let cap = if model.device.tensor_accelerated() {
            3
        } else {
            1
        };
        assert_eq!(neural[0].len(), cap);
        let raw = model.dflash_draft(&[(0, 54321)], cap).unwrap().unwrap();
        assert_eq!(neural[0], raw[0][..cap]);
        let continuation = model.forward(54321).unwrap();
        let committed = [prompt.as_slice(), &sequence[..accepted]].concat();
        fresh(&mut model, 0, &prompt);
        for &token in &sequence[..accepted] {
            model.forward(token).unwrap();
        }
        assert_eq!(model.slots[0].history, committed);
        let reference = model.forward(54321).unwrap();
        same_logits(
            &continuation,
            &reference,
            "copy-source rollback continuation",
        );
    }
}

#[test]
fn mlx_direct_prefill_pages_preserve_attention() {
    direct_prefill_pages(false);
}

#[test]
#[ignore = "GPU diagnostic only; never compare instrumented timings to serving"]
fn mlx_direct_prefill_pages_benchmark() {
    direct_prefill_pages(true);
}

fn direct_prefill_pages(measure: bool) {
    let device = MetalDevice::new(Some(256 << 20)).unwrap();
    let (heads, kv, stride) = (24usize, 4usize, 1060usize);
    let upload_u = |v: &[u32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let value = |i: usize, salt: u32| {
        let bits = (i as u32).wrapping_mul(1664525).wrapping_add(salt);
        half::bf16::from_f32(((bits ^ bits.rotate_left(13)) % 8192) as f32 / 2048. - 2.)
    };
    let keys = device
        .upload(
            &(0..stride * 16 * kv * 256)
                .flat_map(|i| value(i, 1234).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let values = device
        .upload(
            &(0..stride * 16 * kv * 256)
                .flat_map(|i| value(i, 5678).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    // Two slots share the pool but traverse different physical-page orders.
    let pages = upload_u(
        &(0..stride as u32)
            .rev()
            .chain((0..stride as u32).map(|i| (i + 73) % stride as u32))
            .collect::<Vec<_>>(),
    );
    for (prefix, rows) in [
        (0usize, 1usize),
        (0, 13),
        (0, 16),
        (0, 31),
        (0, 32),
        (0, 33),
        (3, 37),
        (31, 65),
        (997, 129),
        (3071, 512),
        (3584, 511),
        (4095, 1),
        (8192, 512),
        (16384, 511),
    ] {
        let query = device
            .upload(
                &(0..(rows + 32) * heads * 256)
                    .flat_map(|i| value(i, 1013904223).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let meta = upload_u(
            &(0..rows)
                .flat_map(|r| [((r / 32) % 2) as u32, (prefix + r) as u32])
                .collect::<Vec<_>>(),
        );
        let limits = upload_u(&(0..rows).map(|r| (prefix + r) as u32).collect::<Vec<_>>());
        let tiles = upload_u(
            &(0..rows.div_ceil(32))
                .rev()
                .flat_map(|i| [(i * 32) as u32, (rows - i * 32).min(32) as u32])
                .collect::<Vec<_>>(),
        );
        let poison: Vec<_> = (0..rows * heads * 256 + 32)
            .flat_map(|_| f32::NAN.to_le_bytes())
            .collect();
        let query_f32 = device
            .upload(
                &(0..(rows + 32) * heads * 256)
                    .flat_map(|i| value(i, 1013904223).to_f32().to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let grouped_query = device
            .alloc(rows.div_ceil(32) * 32 * heads * 256 * 2)
            .unwrap();
        let output = device.upload(&poison).unwrap();
        let run = |name: &str, destination: &Buffer| {
            let cmd = device.begin().unwrap();
            let grouped = name == "mlx_attention_prefill_gqa";
            if grouped {
                cmd.dispatch(
                    "mlx_attention_query_grouped",
                    &[&query_f32, &grouped_query, &tiles],
                    &[rows.div_ceil(32) as u32],
                    [rows.div_ceil(32) * 32 * heads, 1, 1],
                    256,
                );
            }
            cmd.dispatch(
                name,
                &[
                    if grouped { &grouped_query } else { &query },
                    &keys,
                    &values,
                    &meta,
                    &pages,
                    destination,
                    &tiles,
                    &limits,
                ],
                &[
                    heads as u32,
                    kv as u32,
                    stride as u32,
                    (1f32 / 16.).to_bits(),
                ],
                if grouped {
                    [kv, rows.div_ceil(32) * 4, 1]
                } else {
                    [heads, rows.div_ceil(32), 1]
                },
                if grouped { 256 } else { 128 },
            );
            cmd.finish().unwrap()
        };
        run("mlx_attention_prefill", &output);
        let expected = unsafe { output.read_f32(0, rows * heads * 256) };
        assert!(expected.iter().all(|v| v.is_finite()));
        let all_routes = [
            "mlx_attention_prefill_direct",
            "mlx_attention_prefill_indexed",
            "mlx_attention_prefill_gqa",
        ];
        let routes = &all_routes[..if device.tensor_accelerated() { 3 } else { 2 }];
        for route in routes {
            // Independent poison prevents a missing candidate write from
            // inheriting a correct result left by the baseline dispatch.
            let candidate = device.upload(&poison).unwrap();
            run(route, &candidate);
            let actual = unsafe { candidate.read_f32(0, rows * heads * 256 + 32) };
            assert!(actual[rows * heads * 256..].iter().all(|v| v.is_nan()));
            same_logits(&actual[..rows * heads * 256], &expected, route);
        }
        if !measure {
            continue;
        }
        let candidate = device.upload(&poison).unwrap();
        let mut times = vec![Vec::new(); routes.len()];
        for r in 0..7 {
            for i in 0..routes.len() {
                let arm = (r + i) % routes.len();
                times[arm].push(run(routes[arm], &candidate) * 1e6);
            }
        }
        let median: Vec<_> = times
            .iter()
            .map(|samples| {
                let mut sorted = samples.clone();
                sorted.sort_by(f64::total_cmp);
                sorted[sorted.len() / 2]
            })
            .collect();
        eprintln!(
            "MLX_ATTENTION_PAGES {}",
            serde_json::json!({"prefix":prefix,"rows":rows,"routes":routes,"median_us":median,"samples_us":times})
        );
    }
}

#[test]
fn mlx_verify_attention_preserves_per_row_partition_boundaries() {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let (heads, kv, rows, stride) = (24usize, 4usize, 8usize, 194usize);
    let upload_u = |v: &[u32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let value = |i: usize, salt: u32| {
        let x = (i as u32).wrapping_mul(1664525).wrapping_add(salt);
        half::bf16::from_f32(((x ^ x.rotate_left(13)) % 2048) as f32 / 1024.0 - 1.0)
    };
    let q = device
        .upload(
            &(0..rows * heads * 256)
                .flat_map(|i| value(i, 1013904223).to_f32().to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let keys = device
        .upload(
            &(0..stride * 16 * kv * 256)
                .flat_map(|i| value(i, 1234).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let values = device
        .upload(
            &(0..stride * 16 * kv * 256)
                .flat_map(|i| value(i, 5678).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let pages = upload_u(&(0..stride as u32).rev().collect::<Vec<_>>());
    let parts = device.alloc(rows * heads * MAX_SPLITS * 258 * 4).unwrap();
    let out = device.alloc(rows * heads * 256 * 4).unwrap();
    let selected = upload_u(&(0..rows as u32).collect::<Vec<_>>());
    for prefix in [511usize, 1023, 2047, 3071] {
        let meta = upload_u(
            &(0..rows)
                .flat_map(|r| [0, (prefix + r) as u32])
                .collect::<Vec<_>>(),
        );
        let limits = upload_u(&(0..rows).map(|r| (prefix + r) as u32).collect::<Vec<_>>());
        for floor in [4usize, 16] {
            let splits = (prefix + rows).div_ceil(128).max(floor).min(MAX_SPLITS);
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "mlx_attention_verify",
                &[
                    &q, &keys, &values, &meta, &pages, &selected, &parts, &limits,
                ],
                &[
                    heads as u32,
                    kv as u32,
                    stride as u32,
                    (1.0f32 / 16.0).to_bits(),
                    splits as u32,
                    floor as u32,
                ],
                [kv, rows, splits],
                128,
            );
            cmd.dispatch(
                "qwen_attention_merge",
                &[&parts, &out, &selected],
                &[heads as u32, splits as u32],
                [heads * rows, 1, 1],
                32,
            );
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, rows * heads * 256) };
            for row in 0..rows {
                let one = upload_u(&[row as u32]);
                let splits = (prefix + row + 1).div_ceil(128).max(floor).min(MAX_SPLITS);
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    "mlx_attention_decode",
                    &[&q, &keys, &values, &meta, &pages, &one, &parts, &limits],
                    &[
                        heads as u32,
                        kv as u32,
                        stride as u32,
                        (1.0f32 / 16.0).to_bits(),
                        splits as u32,
                    ],
                    [kv, 1, splits],
                    128,
                );
                cmd.dispatch(
                    "qwen_attention_merge",
                    &[&parts, &out, &one],
                    &[heads as u32, splits as u32],
                    [heads, 1, 1],
                    32,
                );
                cmd.finish().unwrap();
            }
            same_logits(
                &got,
                &unsafe { out.read_f32(0, got.len()) },
                &format!("attention prefix={prefix} floor={floor}"),
            );
        }
    }
}

#[test]
fn mlx_decode_attention_is_batch_and_length_invariant() {
    mlx_decode_attention_cohorts(false);
}

#[test]
#[ignore = "warm GPU attention cost only; full-model and serving gates are separate"]
fn mlx_decode_attention_cohort_cost() {
    mlx_decode_attention_cohorts(true);
}

fn mlx_decode_attention_cohorts(timing: bool) {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let (heads, kv, rows, stride) = (24usize, 4usize, 8usize, 256usize);
    let upload_u = |v: &[u32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let value = |i: usize, salt: u32| {
        let x = (i as u32).wrapping_mul(1664525).wrapping_add(salt);
        half::bf16::from_f32(((x ^ x.rotate_left(13)) % 2048) as f32 / 1024.0 - 1.0)
    };
    let q = device
        .upload(
            &(0..rows * heads * 256)
                .flat_map(|i| value(i, 1013904223).to_f32().to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let keys = device
        .upload(
            &(0..stride * 16 * kv * 256)
                .flat_map(|i| value(i, 1234).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let values = device
        .upload(
            &(0..stride * 16 * kv * 256)
                .flat_map(|i| value(i, 5678).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    // Two logical slots view reversed/interleaved physical pages. Selection
    // order and peer length must not leak into a row's reduction tree.
    let pages = upload_u(
        &(0..stride as u32)
            .rev()
            .chain((0..stride as u32).map(|i| (i * 73) % stride as u32))
            .collect::<Vec<_>>(),
    );
    let part_values = rows * heads * MAX_SPLITS * 258;
    let output_values = rows * heads * 256;
    let parts = device.alloc((part_values + 32) * 4).unwrap();
    let out = device.alloc((output_values + 32) * 4).unwrap();
    let cases = [
        [1usize, 31, 127, 511, 2047, 2048, 3584, 4096],
        [512; 8],
        [3584; 8],
    ];
    for lengths in cases {
        let meta = upload_u(
            &(0..rows)
                .flat_map(|r| [(r % 2) as u32, (lengths[r] - 1) as u32])
                .collect::<Vec<_>>(),
        );
        let limits = upload_u(&lengths.map(|n| (n - 1) as u32));
        let dispatch = |cmd: &Commands<'_>, ids: &[u32], selected: &Buffer, stable: bool| {
            let floor = if stable {
                16
            } else {
                16usize.div_ceil(ids.len())
            };
            let splits = ids
                .iter()
                .map(|&r| lengths[r as usize])
                .max()
                .unwrap()
                .div_ceil(128)
                .max(floor)
                .min(MAX_SPLITS);
            cmd.dispatch(
                if stable {
                    "mlx_attention_stable"
                } else {
                    "mlx_attention_decode"
                },
                &[&q, &keys, &values, &meta, &pages, selected, &parts, &limits],
                &[
                    heads as u32,
                    kv as u32,
                    stride as u32,
                    (1.0f32 / 16.0).to_bits(),
                    splits as u32,
                    floor as u32,
                ],
                [kv, ids.len(), splits],
                128,
            );
            cmd.dispatch(
                "qwen_attention_merge",
                &[&parts, &out, selected],
                &[heads as u32, splits as u32],
                [heads * ids.len(), 1, 1],
                32,
            );
        };
        // Independent single-row production kernel is the exact oracle.
        for row in 0..rows as u32 {
            let selected = upload_u(&[row]);
            let cmd = device.begin().unwrap();
            dispatch(&cmd, &[row], &selected, false);
            cmd.finish().unwrap();
        }
        let reference = unsafe { out.read_f32(0, output_values) };
        assert!(reference.iter().all(|v| v.is_finite()));
        for ids in [
            vec![3],
            vec![7, 0],
            vec![3, 0, 6, 2],
            (0..8).rev().collect(),
        ] {
            let selected = upload_u(&ids);
            unsafe {
                out.write_u32(&vec![u32::MAX; output_values + 32]);
                parts.write_u32(&vec![u32::MAX; part_values + 32]);
            }
            let cmd = device.begin().unwrap();
            dispatch(&cmd, &ids, &selected, true);
            cmd.finish().unwrap();
            let got = unsafe { out.read_f32(0, output_values + 32) };
            for row in 0..rows {
                let range = row * heads * 256..(row + 1) * heads * 256;
                if ids.contains(&(row as u32)) {
                    same_logits(
                        &got[range.clone()],
                        &reference[range],
                        &format!("row={row} lengths={lengths:?} selected={ids:?}"),
                    );
                } else {
                    assert!(got[range].iter().all(|v| v.to_bits() == u32::MAX));
                }
            }
            assert!(got[output_values..].iter().all(|v| v.to_bits() == u32::MAX));
            assert!(
                unsafe { parts.read_f32(part_values, 32) }
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX)
            );
            if timing {
                let run = |stable| {
                    let cmd = device.begin().unwrap();
                    for _ in 0..32 {
                        dispatch(&cmd, &ids, &selected, stable);
                    }
                    cmd.finish().unwrap() * 1e6 / 32.
                };
                for _ in 0..4 {
                    run(false);
                    run(true);
                }
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..9 {
                    for i in 0..2 {
                        let route = (round + i) % 2;
                        times[route].push(run(route == 1));
                    }
                }
                eprintln!(
                    "MLX_STABLE_ATTENTION {}",
                    serde_json::json!({
                        "lengths":lengths,"selected":ids,"routes":["baseline","stable"],"gpu_us":times
                    })
                );
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL and PADDOCK_METAL_DFLASH_MODEL; real GPU transactions"]
fn mlx_dflash_verification_matches_decode_at_every_rejection_boundary() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    verification_boundaries(&path, &draft, false);
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL and PADDOCK_METAL_DFLASH_MODEL; packed candidate rollback gate"]
fn mlx_packed_dflash_verification_matches_decode_at_every_rejection_boundary() {
    crate::affine::BASELINE_PACKED_FOR_TEST.with(|v| v.set(false));
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    verification_boundaries(&path, &draft, false);
    crate::affine::BASELINE_PACKED_FOR_TEST.with(|v| v.set(false));
}

#[test]
#[ignore = "requires PADDOCK_SPLASH_MODEL; real packed target/draft GPU transactions"]
fn splash_verification_matches_decode_at_every_rejection_boundary() {
    let path = std::env::var("PADDOCK_SPLASH_MODEL").unwrap();
    verification_boundaries(&path, &path, true);
}

#[test]
#[ignore = "requires PADDOCK_SPLASH_MODEL; real packed single-user prefill"]
fn splash_wide_cold_prefill_matches_bounded_prompt_and_replay() {
    let path = std::env::var("PADDOCK_SPLASH_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 1, None).unwrap();
    model.attach_dflash(Path::new(&path)).unwrap();
    let prompt: Vec<u32> = (1000..3053).collect();
    let expected = fresh(&mut model, 0, &prompt);
    model.reset();
    for cache in &mut model.cache {
        cache.table.clear(&mut model.pool);
        cache.history.clear();
    }
    model.prefill_begin(0, prompt.clone()).unwrap();
    let (_, done) = model.forward_mixed(&[], 8192).unwrap();
    assert!(done.is_empty());
    assert_eq!(model.slots[0].history.len(), 2048);
    let (_, done) = model.forward_mixed(&[], 8192).unwrap();
    same_logits(
        &done[0].1,
        &expected,
        "wide cold wave versus bounded prefill",
    );
    model.reset();
    let replay = model.forward_prefill(0, &prompt).unwrap();
    assert!(model.slots[0].reused > 0);
    same_logits(&replay, &expected, "wide-wave cached replay");
}

fn verification_boundaries(path: &str, draft: &str, splash: bool) {
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    assert!(model.attach_mtp(Path::new(&path)).is_err());
    model.attach_dflash(Path::new(&draft)).unwrap();
    assert!(model.spec_capable());
    assert!(model.attach_dflash(Path::new(&draft)).is_err());
    assert_eq!(
        model.spec_batch_draft_budget(1),
        Some(if splash { 7 } else { lookup::MAX_DRAFT })
    );
    assert_eq!(
        model.spec_batch_draft_budget(4),
        Some(if splash { 7 } else { 0 })
    );
    let tokens: Vec<u32> = (2000..2009).collect();
    for (length, accepted_lengths, slots) in [
        (79, (1..=8).collect::<Vec<_>>(), vec![0, 3]),
        // Cross both a KV page and the c=1 split-attention election boundary.
        (2047, vec![1, 3, 8], vec![0]),
        // Cross the Splash long-decode election within a verification block.
        (2043, vec![1, 3, 8], vec![0]),
    ] {
        let prompt: Vec<u32> = (1000..1000 + length).collect();
        let first = fresh(&mut model, 0, &prompt);
        if splash && length == 2047 {
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            model.prefill_begin(0, prompt.clone()).unwrap();
            let (_, mut done) = model.forward_mixed(&[], 8192).unwrap();
            // On a wide server the admission quantum is intentionally short.
            while done.is_empty() {
                done = model.forward_mixed(&[], 8192).unwrap().1;
            }
            same_logits(
                &done[0].1,
                &first,
                "cold admission chunks versus direct prefill",
            );
            fresh(&mut model, 0, &prompt);
        }
        let mut expected = Vec::new();
        for &token in &tokens {
            expected.push(model.forward(token).unwrap());
        }
        if splash && length == 79 {
            // Real c=4 verification elects the 32-row tensor tile. Include a
            // ragged cohort so row padding cannot affect target arithmetic.
            for counts in [[8usize, 8, 8, 8], [1, 3, 5, 8]] {
                fresh(&mut model, 0, &prompt);
                for slot in 1..4 {
                    model.forward_prefill(slot, &prompt).unwrap();
                }
                let reqs: Vec<_> = counts
                    .iter()
                    .enumerate()
                    .map(|(slot, &n)| (slot, prompt.len(), tokens[..n].to_vec()))
                    .collect();
                let got = model.forward_spec_verify(&reqs).unwrap().unwrap();
                let mut base = 0;
                for (slot, &count) in counts.iter().enumerate() {
                    for row in 0..count {
                        same_logits(
                            &got[(base + row) * model.vocab..(base + row + 1) * model.vocab],
                            &expected[row],
                            &format!("c4 slot={slot} count={count} row={row}"),
                        );
                    }
                    base += count;
                }
                let committed: Vec<_> = counts.iter().map(|&n| n as u32).collect();
                model.spec_commit(&committed).unwrap();
                for (slot, &count) in counts.iter().enumerate() {
                    let next = model
                        .execute(
                            &[(slot, tokens[count], (prompt.len() + count) as u32)],
                            &[0],
                        )
                        .unwrap();
                    same_logits(
                        &next,
                        &expected[count],
                        &format!("c4 committed slot={slot} count={count}"),
                    );
                }
            }
        }
        // The adaptive controller also elects narrow rounds. Their kernel
        // specializations must preserve the same logits, not just argmaxes.
        for count in 1..=8 {
            fresh(&mut model, 0, &prompt);
            let verified = model
                .forward_spec_verify(&[(0, prompt.len(), tokens[..count].to_vec())])
                .unwrap()
                .unwrap();
            for (r, logits) in verified.chunks(model.vocab).enumerate() {
                same_logits(logits, &expected[r], &format!("width={count} row={r}"));
            }
            model.spec_commit(&[count as u32]).unwrap();
            let continued = model.forward(tokens[count]).unwrap();
            same_logits(
                &continued,
                &expected[count],
                &format!("width={count} continued"),
            );
        }
        for slot in slots {
            for &accepted in &accepted_lengths {
                same_logits(&fresh(&mut model, slot, &prompt), &first, "fresh prefill");
                let request = [(slot, prompt.len(), tokens[..8].to_vec())];
                let before = model.slots[slot].history.clone();
                let verified = model.forward_spec_verify(&request).unwrap().unwrap();
                for (r, logits) in verified.chunks(model.vocab).enumerate() {
                    same_logits(logits, &expected[r], &format!("slot={slot} row={r}"));
                }
                assert_eq!(model.slots[slot].history, before);
                assert!(
                    model
                        .execute(&[(slot, 9, prompt.len() as u32)], &[0])
                        .is_err()
                );
                assert!(model.spec_commit(&[0]).is_err());
                assert!(model.spec_commit(&[9]).is_err());
                assert!(model.spec_commit(&[]).is_err());
                model.spec_commit(&[accepted as u32]).unwrap();
                assert!(model.spec_commit(&[1]).is_err());
                let continued = model
                    .execute(
                        &[(slot, tokens[accepted], (prompt.len() + accepted) as u32)],
                        &[0],
                    )
                    .unwrap();
                same_logits(
                    &continued,
                    &expected[accepted],
                    &format!("slot={slot} accepted={accepted}"),
                );
            }
        }
        // Cancellation discards a transaction; later slot reuse starts cleanly.
        fresh(&mut model, 3, &prompt);
        model
            .forward_spec_verify(&[(3, prompt.len(), tokens[..8].to_vec())])
            .unwrap()
            .unwrap();
        same_logits(&fresh(&mut model, 0, &prompt), &first, "reset prefill");
        let proposals = model.spec_draft_batch(&[(0, 2000)], 1).unwrap().unwrap();
        let trained = model.dflash_draft(&[(0, 2000)], 1).unwrap().unwrap();
        assert_eq!(
            trained[0].len(),
            if !splash && model.device.tensor_accelerated() {
                1
            } else {
                7
            }
        );
        // The hidden-state window remains eight positions even when the
        // head/selector emits only a prefix; dflash's direct test checks its
        // hidden states and vocabulary logits against the full-window path.
        assert_eq!(proposals[0].len(), if splash { 7 } else { 1 });
        assert_eq!(proposals[0], trained[0][..proposals[0].len()]);
    }
}
