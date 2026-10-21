use super::*;

#[test]
fn attention_reuse_is_exact_and_bounded() {
    attention_reuse(false);
}

#[test]
#[ignore = "Bonsai bounded value-load attention GPU diagnostic"]
fn attention_reuse_cost() {
    attention_reuse(true);
}

fn attention_reuse(timed: bool) {
    let d = MetalDevice::new(None).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    for live in [1usize, 4] {
        for length in if timed {
            vec![512usize, 2048, 8192]
        } else {
            vec![1usize, 31, 129, 1025]
        } {
            let stride = length.div_ceil(16) + 1;
            let make = |count: usize, salt: usize| {
                d.upload(
                    &(0..count)
                        .flat_map(|i| {
                            (((i * salt + i / 256) % 1999) as f32 / 1317. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let keys = make(stride * live * 16 * 4 * 256, 17);
            let values = make(stride * live * 16 * 4 * 256, 37);
            let query = make((live + 2) * 24 * 256, 19);
            let mut metadata = vec![0u32; (live + 2) * 2];
            let mut limits = vec![0u32; live + 2];
            let mut selected = Vec::new();
            for slot in 0..live {
                let row = live - slot;
                selected.push(row as u32);
                metadata[row * 2] = slot as u32;
                metadata[row * 2 + 1] = (length - 1) as u32;
                limits[row] = (length.saturating_sub(slot * 7).max(1) - 1) as u32;
            }
            let meta = upload(&metadata);
            let rows = upload(&selected);
            let limits = upload(&limits);
            let pages = upload(
                &(0..stride * live)
                    .rev()
                    .map(|p| p as u32)
                    .collect::<Vec<_>>(),
            );
            for splits in if timed {
                vec![length.div_ceil(128).clamp(16, 32)]
            } else {
                vec![16usize, 17, 32]
            } {
                let elements = live * 24 * splits * 258;
                let out = d.alloc((elements + 32) * 4).unwrap();
                let mut expected: Option<Vec<u32>> = None;
                let mut seconds = [Vec::new(), Vec::new()];
                for round in 0..if timed { 7 } else { 1 } {
                    for order in 0..2 {
                        let mode = (order + round) % 2;
                        unsafe {
                            out.write_u32(&vec![u32::MAX; elements + 32]);
                        }
                        let cmd = d.begin().unwrap();
                        let repeats = if timed { 16 } else { 1 };
                        for _ in 0..repeats {
                            cmd.dispatch_at(
                                ["bonsai_attention_decode", "bonsai_attention_value2"][mode],
                                &[&query, &keys, &values, &meta, &pages, &rows, &out, &limits],
                                &[0, 0, 0, 0, 0, 0, 32, 0],
                                &[
                                    24,
                                    4,
                                    stride as u32,
                                    (1.0f32 / 16.).to_bits(),
                                    splits as u32,
                                    16,
                                ],
                                [4, live, splits],
                                128,
                            );
                        }
                        let elapsed = cmd.finish().unwrap() / repeats as f64;
                        let actual = unsafe { out.read_u32(elements + 32) };
                        assert!(
                            actual[..8]
                                .iter()
                                .chain(&actual[elements + 8..])
                                .all(|v| *v == u32::MAX)
                        );
                        if let Some(expected) = &expected {
                            let differences =
                                actual.iter().zip(expected).filter(|(a, b)| a != b).count();
                            assert_eq!(
                                differences, 0,
                                "live={live} length={length} splits={splits} mode={mode}"
                            );
                        } else {
                            expected = Some(actual);
                        }
                        if round > 0 {
                            seconds[mode].push(elapsed);
                        }
                    }
                }
                if timed {
                    eprintln!(
                        "BONSAI_ATTENTION_AHEAD {}",
                        serde_json::json!({"live":live,"length":length,"splits":splits,"seconds":seconds})
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Bonsai MLX: alternating in-process full-logit and execution-cost comparison"]
fn mixed_projection_paired_cost() {
    paired_cost(PairedChange::Mixed);
}

#[test]
#[ignore = "requires Bonsai MLX: half-staged prefill costs and full-logit equality"]
fn half_staging_paired_cost() {
    paired_cost(PairedChange::Half);
}

#[test]
#[ignore = "requires M5 Bonsai MLX: short-prefill half-staging costs and full-logit equality"]
fn short_half_staging_paired_cost() {
    paired_cost(PairedChange::ShortHalf);
}

#[test]
#[ignore = "requires M5 Bonsai MLX: deeper prompt staging and full-logit equality"]
fn deep_staging_paired_cost() {
    paired_cost(PairedChange::Deep);
}

enum PairedChange {
    Recurrent,
    Mixed,
    Half,
    ShortHalf,
    Deep,
    Attention,
}

#[test]
#[ignore = "requires Bonsai MLX: packed F32 recurrence, full logits and paired costs"]
fn recurrent_packed_paired_cost() {
    paired_cost(PairedChange::Recurrent);
}

#[test]
#[ignore = "requires Bonsai MLX: bounded value-load attention, full logits and paired costs"]
fn attention_ahead_paired_cost() {
    paired_cost(PairedChange::Attention);
}

#[test]
#[ignore = "requires Bonsai MLX: alternate exact attention on identical independent slots"]
fn attention_ahead_paired_steps() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            BASELINE_ATTENTION_AHEAD.set(false);
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_BONSAI_MLX").unwrap();
    let mut model = Qwen35::load(std::path::Path::new(&path), 8192, 2, None).unwrap();
    let allocated = model.device.allocated_bytes();
    for length in [512u32, 2048, 4096] {
        model.reset();
        let prompt = (0..length)
            .map(|i| 1000 + (i * 17) % 16000)
            .collect::<Vec<_>>();
        let mut initial = None;
        for slot in 0..2 {
            while model.evict_checkpoint() {}
            let row = model.forward_prefill(slot, &prompt).unwrap();
            if let Some(expected) = &initial {
                assert_eq!(&row, expected, "cold controls differ");
            } else {
                initial = Some(row);
            }
        }
        let mut wall = [Vec::new(), Vec::new()];
        let mut gpu = [Vec::new(), Vec::new()];
        for step in 0..64u32 {
            let mut expected = None;
            for order in 0..2 {
                let slot = (step as usize + order) % 2;
                BASELINE_ATTENTION_AHEAD.set(slot == 0);
                let start = std::time::Instant::now();
                let row = model
                    .execute(&[(slot, 7000 + step, length + step)], &[0])
                    .unwrap();
                let seconds = start.elapsed().as_secs_f64();
                assert!(row.iter().all(|v| v.is_finite()));
                let bits = row.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                if let Some(previous) = &expected {
                    assert_eq!(&bits, previous, "prompt={length} step={step}");
                } else {
                    expected = Some(bits);
                }
                if step >= 8 {
                    wall[slot].push(seconds);
                    gpu[slot].push(model.last_gpu_seconds);
                }
                assert_eq!(model.device.allocated_bytes(), allocated);
            }
        }
        eprintln!(
            "BONSAI_PAIRED_STEPS {}",
            serde_json::json!({"prompt":length,"wall_seconds":wall,"gpu_seconds":gpu,"allocated":allocated})
        );
    }
}

fn paired_cost(change: PairedChange) {
    use paddock_engine::generator::Generator;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            BASELINE_MIXED.set(false);
            BASELINE_HALF_STAGING.set(false);
            BASELINE_SHORT_HALF_STAGING.set(false);
            BASELINE_DEEP_STAGING.set(false);
            BASELINE_ATTENTION_AHEAD.set(false);
            forward::BASELINE_RECURRENT_FOR_TEST.set(false);
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_BONSAI_MLX").expect("Bonsai MLX checkpoint");
    let mut model = Qwen35::load(std::path::Path::new(&path), 4096, 4, None).unwrap();
    let half_staging = matches!(change, PairedChange::Half);
    let short_half_staging = matches!(change, PairedChange::ShortHalf);
    let deep_staging = matches!(change, PairedChange::Deep);
    let attention = matches!(change, PairedChange::Attention);
    let recurrent = matches!(change, PairedChange::Recurrent);
    for length in if short_half_staging {
        vec![17u32, 33, 65, 129, 257]
    } else if half_staging || deep_staging || attention || recurrent {
        vec![512u32, 2048]
    } else {
        vec![512]
    } {
        for live in [1, 4] {
            let mut expected = None;
            for round in
                0..if half_staging || short_half_staging || deep_staging || attention || recurrent {
                    3
                } else {
                    5
                }
            {
                for at in 0..2 {
                    let baseline = (at + round) % 2 == 0;
                    if recurrent {
                        forward::BASELINE_RECURRENT_FOR_TEST.set(baseline);
                    } else if attention {
                        BASELINE_ATTENTION_AHEAD.set(baseline);
                    } else if deep_staging {
                        BASELINE_DEEP_STAGING.set(baseline);
                    } else if short_half_staging {
                        BASELINE_SHORT_HALF_STAGING.set(baseline);
                    } else if half_staging {
                        BASELINE_HALF_STAGING.set(baseline);
                    } else {
                        BASELINE_MIXED.set(baseline);
                    }
                    model.reset();
                    while model.evict_checkpoint() {}
                    for slot in 0..live {
                        model
                            .prefill_begin(
                                slot,
                                (0..length)
                                    .map(|i| 1000 + (i * 17 + slot as u32 * 41) % 16000)
                                    .collect(),
                            )
                            .unwrap();
                    }
                    let mut rows = vec![Vec::new(); live];
                    let start = std::time::Instant::now();
                    let mut done = 0;
                    while done < live {
                        for (slot, row, _) in model.forward_mixed(&[], 512).unwrap().1 {
                            rows[slot] = row;
                            done += 1;
                        }
                    }
                    let prefill = start.elapsed().as_secs_f64();
                    let mut hash = blake3::Hasher::new();
                    for value in rows.iter().flatten() {
                        hash.update(&value.to_bits().to_le_bytes());
                    }
                    let mut times = Vec::new();
                    for step in 0..if short_half_staging { 8 } else { 32 } {
                        let tokens: Vec<_> =
                            (0..live).map(|slot| 7000 + step + slot as u32).collect();
                        let start = std::time::Instant::now();
                        let row = model
                            .forward_batch(&tokens, &vec![length + step; live])
                            .unwrap();
                        times.push(start.elapsed().as_secs_f64());
                        assert!(row.iter().all(|v| v.is_finite()));
                        for value in row {
                            hash.update(&value.to_bits().to_le_bytes());
                        }
                    }
                    let digest = hash.finalize().to_hex().to_string();
                    let identity = (digest.clone(), model.device_mem_used());
                    if let Some(expected) = &expected {
                        assert_eq!(&identity, expected);
                    } else {
                        expected = Some(identity);
                    }
                    eprintln!(
                        "BONSAI_PAIRED_COST {}",
                        serde_json::json!({"round":round,"live":live,"tokens":length,"half_staging":half_staging,"short_half_staging":short_half_staging,"deep_staging":deep_staging,"packed_recurrent":recurrent,"attention_ahead":attention,"baseline":baseline,"prefill_s":prefill,"decode_s":times,"logits_blake3":digest,"allocated":model.device_mem_used()})
                    );
                }
            }
        }
    }
}
