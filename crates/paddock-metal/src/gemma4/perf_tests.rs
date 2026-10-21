use super::*;

#[test]
#[ignore = "requires Gemma MLX: isolate batch-dependent projection and attention arithmetic"]
fn mlx_batch_arithmetic_diagnostic() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            forward::FIXED_DECODE_SPLITS_FOR_TEST.set(false);
            forward::SINGLE_DECODE_PROJECTION_FOR_TEST.set(false);
            forward::BATCH_TRACE_FOR_TEST.with(|v| *v.borrow_mut() = None);
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_MM_MLX").unwrap();
    let mut model = Gemma4::load(Path::new(&path), 4096, 5, None).unwrap();
    assert!(
        model.mlx
            && model.width == 5376
            && model.moe_scratch.is_none()
            && model.diffusion.is_none()
    );
    for mode in 0..4 {
        forward::FIXED_DECODE_SPLITS_FOR_TEST.set(mode & 1 != 0);
        forward::SINGLE_DECODE_PROJECTION_FOR_TEST.set(mode & 2 != 0);
        model.reset();
        let prompt = (0..1024u32)
            .map(|i| 1000 + (i * 17) % 16000)
            .collect::<Vec<_>>();
        let mut initial = None;
        for slot in 0..5 {
            while model.evict() {}
            let row = model.forward_prefill(slot, &prompt).unwrap();
            if let Some(expected) = &initial {
                assert_eq!(&row, expected);
            } else {
                initial = Some(row);
            }
        }
        let mut changes = 0usize;
        let mut max_abs = 0.0f32;
        let mut times = [Vec::new(), Vec::new()];
        let mut picks = 0usize;
        for step in 0..16u32 {
            let mut rows = [Vec::new(), Vec::new()];
            let mut traces = [Vec::new(), Vec::new()];
            for at in 0..2 {
                let batch = (step as usize + at) % 2;
                let input = if batch == 0 {
                    vec![(0, 7000 + step, 1024 + step)]
                } else {
                    // Slot 1 follows the isolated control. Other peers join
                    // and leave with their own KV positions.
                    (1..=2 + step as usize % 3)
                        .map(|slot| (slot, 7000 + step, model.slots[slot].history.len() as u32))
                        .collect()
                };
                let start = std::time::Instant::now();
                if step == 0 {
                    forward::BATCH_TRACE_FOR_TEST.with(|v| *v.borrow_mut() = Some(Vec::new()));
                }
                rows[batch] = model.execute(&input, &[0]).unwrap();
                times[batch].push(start.elapsed().as_secs_f64());
                if step == 0 {
                    traces[batch] =
                        forward::BATCH_TRACE_FOR_TEST.with(|v| v.borrow_mut().take().unwrap());
                }
            }
            if step == 0 {
                for ((name, a), (_, b)) in traces[0].iter().zip(&traces[1]) {
                    eprintln!(
                        "GEMMA_BATCH_STAGE mode={mode} stage={name} different={} max={}",
                        a.iter()
                            .zip(b)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count(),
                        a.iter()
                            .zip(b)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max)
                    );
                }
            }
            changes += rows[0]
                .iter()
                .zip(&rows[1])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            max_abs = rows[0]
                .iter()
                .zip(&rows[1])
                .map(|(a, b)| (a - b).abs())
                .fold(max_abs, f32::max);
            let pick = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            picks += usize::from(pick(&rows[0]) == pick(&rows[1]));
        }
        eprintln!(
            "GEMMA_BATCH_ARITHMETIC {}",
            serde_json::json!({"mode":mode,"batch_cycle":[2,3,4],"different":changes,"max_abs":max_abs,"argmax_matches":picks,"seconds":times})
        );
        if mode == 3 {
            assert_eq!(
                changes, 0,
                "fixed attention and one-vector projections must isolate batch occupancy"
            );
        }
    }
}

#[test]
fn decode_ahead_is_exact_and_bounded() {
    decode_ahead(false);
}

#[test]
#[ignore = "Gemma bounded attention load overlap: alternating GPU timings, not serving throughput"]
fn decode_ahead_cost() {
    decode_ahead(true);
}

fn decode_ahead(timed: bool) {
    let d = MetalDevice::new(None).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let (hd, kh, window) = (256usize, 16usize, 1024usize);
    for live in [1usize, 4] {
        for length in if timed {
            vec![512usize, 2048, 8192]
        } else {
            vec![1usize, 31, 129, 1025]
        } {
            let stride = length.div_ceil(16) + 1;
            let ring = if window > 0 { 1536 } else { 0 };
            let tokens = if window > 0 {
                ring * live
            } else {
                stride * live * 16
            };
            let kv = |salt: usize| {
                d.upload(
                    &(0..tokens * kh * hd)
                        .flat_map(|i| {
                            half::bf16::from_f32(((i * salt + i / hd) % 101) as f32 / 97. - 0.5)
                                .to_bits()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let keys = kv(13);
            let values = kv(37);
            let query = d
                .upload(
                    &(0..(live + 2) * 32 * hd)
                        .flat_map(|i| {
                            half::bf16::from_f32(((i * 19) % 127) as f32 / 117. - 0.5)
                                .to_f32()
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut metadata = vec![0u32; (live + 2) * 2];
            let mut selected = Vec::new();
            for slot in 0..live {
                let row = live - slot;
                selected.push(row as u32);
                metadata[row * 2] = slot as u32;
                metadata[row * 2 + 1] = (length.saturating_sub(slot * 7).max(1) - 1) as u32;
            }
            let meta = upload(&metadata);
            let rows = upload(&selected);
            let pages = upload(
                &(0..stride * live)
                    .rev()
                    .map(|p| p as u32)
                    .collect::<Vec<_>>(),
            );
            for splits in if timed {
                vec![
                    length
                        .min(if window > 0 { window } else { length })
                        .div_ceil(128)
                        .max(16usize.div_ceil(live))
                        .clamp(1, 32),
                ]
            } else {
                vec![1usize, 3, 16, 32]
            } {
                let elements = live * 32 * splits * (hd + 2);
                let output = d.alloc((elements + 32) * 4).unwrap();
                let mut expected: Option<Vec<u32>> = None;
                let variants = 2;
                let mut seconds = vec![Vec::new(); variants];
                for round in 0..if timed { 7 } else { 1 } {
                    for order in 0..variants {
                        let mode = (order + round) % variants;
                        unsafe {
                            output.write_u32(&vec![u32::MAX; elements + 32]);
                        }
                        let cmd = d.begin().unwrap();
                        let repeats = if timed { 16 } else { 1 };
                        for _ in 0..repeats {
                            cmd.dispatch_at(
                                match mode {
                                    0 => "gmlx_decode256",
                                    _ => "gmlx_value2_256",
                                },
                                &[&query, &keys, &values, &meta, &pages, &rows, &output],
                                &[0, 0, 0, 0, 0, 0, 32],
                                &[
                                    32,
                                    kh as u32,
                                    stride as u32,
                                    window as u32,
                                    ring as u32,
                                    splits as u32,
                                ],
                                [kh, live, splits],
                                128,
                            );
                        }
                        let elapsed = cmd.finish().unwrap() / repeats as f64;
                        let actual = unsafe { output.read_u32(elements + 32) };
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
                                "hd={hd} live={live} length={length} splits={splits} mode={mode}"
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
                        "GEMMA_DECODE_AHEAD {}",
                        serde_json::json!({"hd":hd,"live":live,"length":length,"splits":splits,"seconds":seconds})
                    );
                }
            }
        }
    }
}

#[test]
fn rounded_merge_is_exact_and_bounded() {
    rounded_merge_check(false);
}

#[test]
#[ignore = "paired attention merge/store costs, not serving throughput"]
fn rounded_merge_cost() {
    rounded_merge_check(true);
}

fn rounded_merge_check(timed: bool) {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    for hd in [256usize, 512] {
        for live in [1usize, 4] {
            let indices = [4u32, 0, 2, 1];
            let rows = d
                .upload(
                    &indices[..live]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let size = 6 * HEADS * hd;
            let sentinel = 12352.0f32.to_bits(); // exactly BF16, including the untouched rows
            let outputs = [
                d.alloc((size + 16) * 4).unwrap(),
                d.alloc((size + 16) * 4).unwrap(),
            ];
            for splits in [1usize, 3, 16, 32] {
                let parts = d
                    .upload(
                        &(0..live * HEADS * splits * (hd + 2))
                            .flat_map(|i| {
                                let split = i / (hd + 2) % splits;
                                let empty = split > 0 && split % 3 != 0;
                                let value = match i % (hd + 2) {
                                    at if at == hd => {
                                        if empty {
                                            f32::NEG_INFINITY
                                        } else {
                                            (i % 19) as f32 / 7. - 2.
                                        }
                                    }
                                    at if at == hd + 1 => {
                                        if empty {
                                            0.
                                        } else {
                                            (i % 7 + 1) as f32 / 3.
                                        }
                                    }
                                    _ => ((i * 13 + i / (hd + 2)) % 997) as f32 / 117. - 3.,
                                };
                                value.to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..if timed { 9 } else { 1 } {
                    for order in 0..2 {
                        let at = (order + round) % 2;
                        unsafe {
                            outputs[at].write_u32(&vec![sentinel; size + 16]);
                        }
                        let cmd = d.begin().unwrap();
                        let repeats = if timed { 60 } else { 1 };
                        for _ in 0..repeats {
                            cmd.dispatch_at(
                                if at == 0 {
                                    "gemma_merge_shared"
                                } else {
                                    "gmlx_merge_rounded"
                                },
                                &[&parts, &outputs[at], &rows],
                                &[0, 32, 0],
                                &[HEADS as u32, splits as u32, hd as u32],
                                [HEADS * live, 1, 1],
                                32,
                            );
                            if at == 0 {
                                cmd.dispatch_at(
                                    "gmlx_round",
                                    &[&outputs[at]],
                                    &[32],
                                    &[size as u32],
                                    [size.div_ceil(256), 1, 1],
                                    256,
                                );
                            }
                        }
                        let seconds = cmd.finish().unwrap() / repeats as f64;
                        if round > 0 {
                            times[at].push(seconds);
                        }
                    }
                    let old = unsafe { outputs[0].read_u32(size + 16) };
                    let new = unsafe { outputs[1].read_u32(size + 16) };
                    let differences = old.iter().zip(&new).filter(|(a, b)| a != b).count();
                    assert_eq!(differences, 0, "hd={hd} live={live} splits={splits}");
                    assert!(
                        new[..8]
                            .iter()
                            .chain(&new[size + 8..])
                            .all(|v| *v == sentinel)
                    );
                    for row in 0..6 {
                        if !indices[..live].contains(&(row as u32)) {
                            assert!(
                                new[8 + row * HEADS * hd..8 + (row + 1) * HEADS * hd]
                                    .iter()
                                    .all(|v| *v == sentinel)
                            );
                        }
                    }
                    assert!(new.iter().all(|v| f32::from_bits(*v).is_finite()));
                }
                if timed {
                    eprintln!(
                        "ROUNDED_MERGE_COST {}",
                        serde_json::json!({"hd":hd,"live":live,"splits":splits,"seconds":times})
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Gemma MLX: in-process paired model costs and full-logit equality"]
fn mlx_speed_paired_cost() {
    mlx_paired_cost(PairedChange::SmallColumns, 512);
}

#[test]
#[ignore = "requires Gemma MLX: paired wide-input projection election and full-logit equality"]
fn mlx_wide_columns_paired_cost() {
    mlx_paired_cost(PairedChange::WideColumns, 512);
}

#[test]
#[ignore = "requires Gemma MLX: in-process fused attention rounding comparison"]
fn mlx_rounded_merge_paired_cost() {
    mlx_paired_cost(PairedChange::RoundedMerge, 512);
}

#[test]
#[ignore = "requires Gemma MLX: paired pipelined projection cost and full-logit equality"]
fn mlx_pipeline_paired_cost() {
    mlx_paired_cost(PairedChange::Pipeline, 512);
}

#[test]
#[ignore = "requires Gemma MLX: long-prompt paired pipelined projections"]
fn mlx_pipeline_long_paired_cost() {
    mlx_paired_cost(PairedChange::Pipeline, 2048);
}

enum PairedChange {
    SmallColumns,
    WideColumns,
    RoundedMerge,
    Pipeline,
    Aligned,
    Attention,
}

#[test]
#[ignore = "requires Gemma MLX: bounded value-load overlap, complete logits and paired cost"]
fn mlx_decode_ahead_paired_cost() {
    mlx_paired_cost(PairedChange::Attention, 2048);
}

#[test]
#[ignore = "requires Gemma MLX: bounded value-load overlap on identical independent slots"]
fn mlx_decode_ahead_paired_steps() {
    mlx_paired_steps(PairedChange::Attention);
}

#[test]
#[ignore = "requires Gemma MLX: paired aligned projection costs and full vocabulary equality"]
fn mlx_aligned_paired_cost() {
    mlx_paired_cost(PairedChange::Aligned, 2048);
}

#[test]
#[ignore = "requires Gemma MLX: alternate aligned kernels on independent identical live slots"]
fn mlx_aligned_paired_steps() {
    mlx_paired_steps(PairedChange::Aligned);
}

fn mlx_paired_steps(change: PairedChange) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            crate::affine::BASELINE_GEMMA_ALIGNED_FOR_TEST.set(false);
            forward::BASELINE_DECODE_AHEAD_FOR_TEST.set(false);
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_MM_MLX").unwrap();
    let mut model = Gemma4::load(Path::new(&path), 8192, 2, None).unwrap();
    let allocated = model.device.allocated_bytes();
    for length in [512u32, 2048, 4096] {
        model.reset();
        let prompt = (0..length)
            .map(|i| 1000 + (i * 17) % 16000)
            .collect::<Vec<_>>();
        let mut initial: Option<Vec<f32>> = None;
        for slot in 0..2 {
            // Both controls must be cold: warm reuse can select a different
            // final-prompt contraction even when token histories match.
            while model.evict() {}
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
            let mut expected: Option<Vec<u32>> = None;
            for order in 0..2 {
                let slot = (step as usize + order) % 2;
                if matches!(change, PairedChange::Attention) {
                    forward::BASELINE_DECODE_AHEAD_FOR_TEST.set(slot == 0);
                } else {
                    crate::affine::BASELINE_GEMMA_ALIGNED_FOR_TEST.set(slot == 0);
                }
                let start = std::time::Instant::now();
                let row = model
                    .execute(&[(slot, 7000 + step, length + step)], &[0])
                    .unwrap();
                let seconds = start.elapsed().as_secs_f64();
                if step >= 8 {
                    wall[slot].push(seconds);
                    gpu[slot].push(model.last_gpu_seconds);
                }
                assert!(row.iter().all(|v| v.is_finite()));
                let bits = row.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                if let Some(previous) = &expected {
                    let differences = bits.iter().zip(previous).filter(|(a, b)| a != b).count();
                    assert_eq!(differences, 0, "prompt={length} step={step}");
                } else {
                    expected = Some(bits);
                }
                assert_eq!(model.device.allocated_bytes(), allocated);
            }
        }
        eprintln!(
            "GEMMA_PAIRED_STEPS {}",
            serde_json::json!({"change":if matches!(change, PairedChange::Attention) {"decode_ahead"} else {"aligned_projection"},"prompt":length,"wall_seconds":wall,"gpu_seconds":gpu,"allocated":allocated})
        );
    }
}

fn mlx_paired_cost(change: PairedChange, prompt: u32) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            crate::affine::BASELINE_GEMMA_SPEED_FOR_TEST.set(false);
            crate::affine::BASELINE_GEMMA_PIPELINE_FOR_TEST.set(false);
            crate::affine::BASELINE_GEMMA_ALIGNED_FOR_TEST.set(false);
            crate::affine::BASELINE_GEMMA_WIDE_FOR_TEST.set(false);
            forward::BASELINE_ROUNDED_MERGE_FOR_TEST.set(false);
            forward::BASELINE_DECODE_AHEAD_FOR_TEST.set(false);
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_MM_MLX").unwrap();
    let mut m = Gemma4::load(Path::new(&path), 4096, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    for live in [1, 4] {
        let mut expected = None;
        for round in 0..5 {
            for at in 0..2 {
                let old = (at + round) % 2 == 0;
                match change {
                    PairedChange::Attention => forward::BASELINE_DECODE_AHEAD_FOR_TEST.set(old),
                    PairedChange::Aligned => {
                        crate::affine::BASELINE_GEMMA_ALIGNED_FOR_TEST.set(old)
                    }
                    PairedChange::Pipeline => {
                        crate::affine::BASELINE_GEMMA_PIPELINE_FOR_TEST.set(old)
                    }
                    PairedChange::RoundedMerge => forward::BASELINE_ROUNDED_MERGE_FOR_TEST.set(old),
                    PairedChange::WideColumns => {
                        crate::affine::BASELINE_GEMMA_WIDE_FOR_TEST.set(old)
                    }
                    PairedChange::SmallColumns => {
                        crate::affine::BASELINE_GEMMA_SPEED_FOR_TEST.set(old)
                    }
                }
                m.reset();
                while m.evict() {}
                for slot in 0..live {
                    m.prefill_begin(
                        slot,
                        (0..prompt)
                            .map(|i| 1000 + (i * 17 + slot as u32 * 41) % 16000)
                            .collect(),
                    )
                    .unwrap();
                }
                let mut logits = vec![Vec::new(); live];
                let start = std::time::Instant::now();
                let mut done = 0;
                while done < live {
                    for (slot, row, _) in m.forward_mixed(&[], 512).unwrap().1 {
                        logits[slot] = row;
                        done += 1;
                    }
                }
                let prefill = start.elapsed().as_secs_f64();
                let mut hash = blake3::Hasher::new();
                for row in &logits {
                    for v in row {
                        hash.update(&v.to_bits().to_le_bytes());
                    }
                }
                let mut times = Vec::new();
                for step in 0..16 {
                    let tokens: Vec<_> = (0..live).map(|slot| 7000 + step + slot as u32).collect();
                    let start = std::time::Instant::now();
                    let row = m
                        .forward_batch(&tokens, &vec![prompt + step; live])
                        .unwrap();
                    times.push(start.elapsed().as_secs_f64());
                    assert!(row.iter().all(|v| v.is_finite()));
                    for v in row {
                        hash.update(&v.to_bits().to_le_bytes());
                    }
                }
                let digest = hash.finalize().to_hex().to_string();
                assert_eq!(m.device.allocated_bytes(), allocated);
                if let Some(expected) = &expected {
                    assert_eq!(&digest, expected);
                } else {
                    expected = Some(digest.clone());
                }
                eprintln!(
                    "GEMMA_PAIRED_COST {}",
                    serde_json::json!({"round":round,"live":live,"prompt":prompt,"baseline":old,"prefill_s":prefill,"decode_s":times,"logits_blake3":digest,"allocated":m.device.allocated_bytes()})
                );
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; GPU prefill tile election"]
fn prefill_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for names in [
        vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ],
        vec!["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
        vec!["blk.0.ffn_down.weight"],
    ] {
        let weights: Vec<_> = names
            .iter()
            .map(|name| {
                let shape = &map.tensor_info(name).unwrap().dims;
                Weight::load(&d, &map, name, &[shape[0] as usize, shape[1] as usize]).unwrap()
            })
            .collect();
        if !weights.iter().all(|w| matches!(w.ty, 12 | 13 | 14 | 23)) {
            continue;
        }
        let k = weights[0].k;
        let workspace = d.alloc(384 * k * 2).unwrap();
        for m in [144usize, 192, 257, 279, 288] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let b: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let base: Vec<_> = weights.iter().zip(&a).collect();
            let narrow: Vec<_> = weights.iter().zip(&b).collect();
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for offset in 0..2 {
                    let at = if round % 2 == 0 { offset } else { 1 - offset };
                    let cmd = d.begin().unwrap();
                    for _ in 0..16 {
                        if at == 1 {
                            forward::prefill96_projection(&cmd, &narrow, &x, m, &workspace);
                        } else if weights.len() == 1 {
                            weights[0].linear(&cmd, &x, &a[0], m, 1., &workspace);
                        } else {
                            projections(&cmd, &base, &x, m, &workspace);
                        }
                    }
                    let ms = cmd.finish().unwrap() * 1000. / 16.;
                    if round > 1 {
                        times[at].push(ms);
                    }
                }
            }
            for (i, w) in weights.iter().enumerate() {
                assert_eq!(
                    unsafe { a[i].read_f32(0, m * w.n) },
                    unsafe { b[i].read_f32(0, m * w.n) },
                    "prefill m={m} {}",
                    names[i]
                );
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "prefill {:?} m={m}: old {:.4} ms tile96 {:.4} ms",
                names, times[0][3], times[1][3]
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; GPU fused projection election"]
fn grouped_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for names in [
        vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ],
        vec!["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"],
    ] {
        let weights: Vec<_> = names
            .iter()
            .map(|name| {
                let shape = &map.tensor_info(name).unwrap().dims;
                Weight::load(&d, &map, name, &[shape[0] as usize, shape[1] as usize]).unwrap()
            })
            .collect();
        println!(
            "fused {:?} types {:?}",
            names,
            weights.iter().map(|w| w.ty).collect::<Vec<_>>()
        );
        if !weights.iter().all(|w| matches!(w.ty, 12..=14)) {
            continue;
        }
        let k = weights[0].k;
        let workspace = d.alloc(128 * k * 2).unwrap();
        for m in [3usize, 4] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let b: Vec<_> = weights
                .iter()
                .map(|w| d.alloc(m * w.n * 4).unwrap())
                .collect();
            let base: Vec<_> = weights.iter().zip(&a).collect();
            let pair: Vec<_> = weights.iter().zip(&b).collect();
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..9 {
                for offset in 0..2 {
                    let at = if round % 2 == 0 { offset } else { 1 - offset };
                    let cmd = d.begin().unwrap();
                    for _ in 0..16 {
                        if at == 0 {
                            projections(&cmd, &base, &x, m, &workspace);
                        } else {
                            forward::pair_projection(&cmd, &pair, &x, m);
                        }
                    }
                    let ms = cmd.finish().unwrap() * 1000. / 16.;
                    if round > 1 {
                        times[at].push(ms);
                    }
                }
            }
            for (i, w) in weights.iter().enumerate() {
                assert_eq!(
                    unsafe { a[i].read_f32(0, m * w.n) },
                    unsafe { b[i].read_f32(0, m * w.n) },
                    "fused m={m} {}",
                    names[i]
                );
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            println!(
                "fused {:?} m={m}: old {:.4} ms pair {:.4} ms",
                names, times[0][3], times[1][3]
            );
        }
    }
}

#[test]
#[ignore = "GPU merge election, not serving throughput"]
fn merge_election() {
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    for hd in [256usize, 512] {
        for rows in [1usize, 4, 32] {
            let selected = d
                .upload(
                    &(0..rows as u32)
                        .rev()
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = d.alloc(rows * HEADS * hd * 4).unwrap();
            let b = d.alloc(a.len()).unwrap();
            for splits in [4usize, 16, 32] {
                let parts = d
                    .upload(
                        &(0..rows * HEADS * splits * (hd + 2))
                            .flat_map(|i| {
                                let v = if i % (hd + 2) == hd + 1 {
                                    1. + (i % 7) as f32
                                } else {
                                    ((i * 13 + i / (hd + 2)) % 31) as f32 / 17. - 1.
                                };
                                v.to_le_bytes()
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..9 {
                    for offset in 0..2 {
                        let at = if round % 2 == 0 { offset } else { 1 - offset };
                        let cmd = d.begin().unwrap();
                        // Amortize command submission over 60 dependent dispatches,
                        // just as a target decode pays one merge per layer.
                        for _ in 0..60 {
                            cmd.dispatch(
                                if at == 0 {
                                    "gemma_merge"
                                } else {
                                    "gemma_merge_shared"
                                },
                                &[&parts, if at == 0 { &a } else { &b }, &selected],
                                &[HEADS as u32, splits as u32, hd as u32],
                                [HEADS * rows, 1, 1],
                                32,
                            );
                        }
                        let ms = cmd.finish().unwrap() * 1000. / 60.;
                        if round > 1 {
                            times[at].push(ms);
                        }
                    }
                }
                let reference = unsafe { a.read_f32(0, rows * HEADS * hd) };
                let actual = unsafe { b.read_f32(0, reference.len()) };
                assert!(actual.iter().all(|x| x.is_finite()));
                assert_eq!(reference, actual, "hd={hd} rows={rows} splits={splits}");
                for t in &mut times {
                    t.sort_by(f64::total_cmp);
                }
                println!(
                    "merge hd={hd} rows={rows} splits={splits}: old {:.5} ms shared {:.5} ms",
                    times[0][3], times[1][3]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; GPU verification projection election"]
fn verification_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(Some(4 << 30)).unwrap();
    for (name, k, n) in [
        ("blk.0.attn_q.weight", 5376usize, 8192usize),
        ("blk.0.ffn_down.weight", 21504, 5376),
        ("token_embd.weight", 5376, 262144),
    ] {
        let w = Weight::load(&d, &map, name, &[k, n]).unwrap();
        for m in [2usize, 3, 4, 5, 6, 7, 8, 16, 32] {
            let x = d
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = d.alloc(m * n * 4).unwrap();
            let b = d.alloc(a.len()).unwrap();
            let p = [k as u32, n as u32, m as u32, w.ty, 1f32.to_bits()];
            let base = match m {
                1 => ("linear_kquant1", 4, 1),
                2 => ("linear_kquant2", 16, 2),
                3 => ("linear_kquant3", 16, 3),
                4 => ("linear_kquant4", 16, 4),
                5..=8 if w.ty != 14 => ("gemma_verify8", 16, 8),
                _ => ("gemma_f32_32", 16, 32),
            };
            let mut candidates = vec![base];
            if m <= 4 && matches!(w.ty, 12..=14) {
                candidates.push((
                    match m {
                        2 => "gemma_pair2",
                        3 => "gemma_pair3",
                        _ => "gemma_pair4",
                    },
                    32,
                    m,
                ));
            }
            if (5..=8).contains(&m) {
                candidates.push((
                    match m {
                        5 => "gemma_full5",
                        6 => "gemma_full6",
                        7 => "gemma_full7",
                        _ => "gemma_full8",
                    },
                    16,
                    m,
                ));
            }
            if m >= 4 {
                candidates.push(("gemma_staged_f32_32", 16, 32));
            }
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                base.0,
                &[&w.buffer, &x, &a],
                &p,
                [n.div_ceil(base.1), m.div_ceil(base.2), 1],
                128,
            );
            cmd.finish().unwrap();
            let reference = unsafe { a.read_f32(0, m * n) };
            let magnitude = reference.iter().map(|v| v.abs()).fold(1f32, f32::max);
            let mut times = vec![Vec::new(); candidates.len()];
            let mut errors = vec![0f32; candidates.len()];
            // A few microsecond dispatches are not enough to reach sustained
            // GPU clocks. Warm a real work train before alternating candidates.
            let cmd = d.begin().unwrap();
            for _ in 0..32 {
                cmd.dispatch(
                    base.0,
                    &[&w.buffer, &x, &a],
                    &p,
                    [n.div_ceil(base.1), m.div_ceil(base.2), 1],
                    128,
                );
            }
            cmd.finish().unwrap();
            for round in 0..9 {
                // Alternate order to avoid granting one candidate all the cold clocks.
                for offset in 0..candidates.len() {
                    let at = if round % 2 == 0 {
                        offset
                    } else {
                        candidates.len() - 1 - offset
                    };
                    let (kernel, cols, rows) = candidates[at];
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, &b],
                        &p,
                        [n.div_ceil(cols), m.div_ceil(rows), 1],
                        128,
                    );
                    let ms = cmd.finish().unwrap() * 1000.;
                    if round > 1 {
                        times[at].push(ms);
                    }
                    if round == 0 {
                        let actual = unsafe { b.read_f32(0, m * n) };
                        assert!(actual.iter().all(|v| v.is_finite()));
                        errors[at] = reference
                            .iter()
                            .zip(&actual)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        assert!(
                            errors[at] / magnitude < 0.00001,
                            "{name} {kernel} m={m}: error={}",
                            errors[at]
                        );
                    }
                }
            }
            for (i, (kernel, _, _)) in candidates.iter().enumerate() {
                times[i].sort_by(f64::total_cmp);
                println!(
                    "{name} m={m} {kernel}: {:.4} ms error={}",
                    times[i][3], errors[i]
                );
            }
        }
    }
}

#[test]
#[ignore = "GPU attention cost diagnostic, not serving throughput"]
fn verification_attention_election() {
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let (hd, kh, ring) = (256usize, 16usize, 1536usize);
    for c in [1usize, 4] {
        for count in [4usize, 8] {
            let rows = c * count;
            let q = d
                .upload(
                    &(0..rows * HEADS * hd)
                        .flat_map(|i| {
                            (((i * 7 + i / hd) % 17) as f32 / 257. - 8. / 257.).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let halfs = |factor: usize| {
                (0..c * ring * kh * hd)
                    .flat_map(|i| {
                        half::f16::from_f32(((i * factor + i / hd) % 19) as f32 / 256. - 9. / 256.)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>()
            };
            let k = d.upload(&halfs(3)).unwrap();
            let v = d.upload(&halfs(11)).unwrap();
            let selected = upload(&(0..rows as u32).collect::<Vec<_>>());
            let pages = upload(&[0]);
            let tiles = upload(
                &(0..c)
                    .flat_map(|i| [(i * count) as u32, count as u32])
                    .collect::<Vec<_>>(),
            );
            let parts = d.alloc(rows * HEADS * SPLITS * (hd + 2) * 4).unwrap();
            let a = d.alloc(rows * HEADS * hd * 4).unwrap();
            let b = d.alloc(a.len()).unwrap();
            for prefix in [128usize, 1024] {
                let meta = upload(
                    &(0..c)
                        .flat_map(|s| (0..count).flat_map(move |i| [s as u32, (prefix + i) as u32]))
                        .collect::<Vec<_>>(),
                );
                let p = [HEADS as u32, kh as u32, 1, 1024, ring as u32];
                let splits = (prefix + count)
                    .min(1024)
                    .div_ceil(128)
                    .max(16usize.div_ceil(rows))
                    .clamp(1, SPLITS);
                let mut params = p.to_vec();
                params.push(splits as u32);
                let mut times = [Vec::new(), Vec::new()];
                for round in 0..9 {
                    for (mode, time) in times.iter_mut().enumerate() {
                        let cmd = d.begin().unwrap();
                        if mode == 0 {
                            cmd.dispatch(
                                "gemma_decode256",
                                &[&q, &k, &v, &meta, &pages, &selected, &parts],
                                &params,
                                [kh, rows, splits],
                                128,
                            );
                            cmd.dispatch(
                                "gemma_merge",
                                &[&parts, &a, &selected],
                                &[HEADS as u32, splits as u32, hd as u32],
                                [HEADS * rows, 1, 1],
                                32,
                            );
                        } else {
                            cmd.dispatch(
                                "gemma_verify_attn256",
                                &[&q, &k, &v, &meta, &pages, &b, &tiles],
                                &p,
                                [HEADS, c, 1],
                                128,
                            );
                        }
                        let ms = cmd.finish().unwrap() * 1000.;
                        if round > 1 {
                            time.push(ms);
                        }
                    }
                }
                let expected = unsafe { a.read_f32(0, rows * HEADS * hd) };
                let actual = unsafe { b.read_f32(0, expected.len()) };
                assert!(actual.iter().all(|x| x.is_finite()));
                let error = expected
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(error < 0.00001, "attention error {error}");
                for time in &mut times {
                    time.sort_by(f64::total_cmp);
                }
                println!(
                    "SWA c={c} block={count} ctx={prefix}: per-query {:.4} ms, F32 matrix {:.4} ms",
                    times[0][3], times[1][3]
                );
            }
        }
    }
}

// A GPU-to-GPU projection probe using the unchanged real checkpoint, never
// a host matrix reference. Timings are diagnostics, not a serving score.
#[test]
#[ignore = "requires PADDOCK_GEMMA4_GGUF; projection election diagnostic"]
fn f32_matrix_projection_election() {
    let path = std::env::var("PADDOCK_GEMMA4_GGUF").unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let device = MetalDevice::new(Some(4 << 30)).unwrap();
    for (name, k, n) in [
        ("blk.0.attn_q.weight", 5376, 8192),
        ("blk.0.ffn_down.weight", 21504, 5376),
        ("token_embd.weight", 5376, 262144),
    ] {
        let w = Weight::load(&device, &map, name, &[k, n]).unwrap();
        for rows in [4usize, 8, 16, 32] {
            let x = device
                .upload(
                    &(0..rows * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31. - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = device.alloc(rows * n * 4).unwrap();
            let b = device.alloc(rows * n * 4).unwrap();
            let p = [k as u32, n as u32, rows as u32, w.ty, 1f32.to_bits()];
            let baseline = if rows <= 4 {
                "linear_kquant4"
            } else if rows <= 8 {
                "gemma_verify8"
            } else {
                "gemma_verify16"
            };
            let tile = rows.min(16);
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                baseline,
                &[&w.buffer, &x, &a],
                &p,
                [n.div_ceil(16), rows.div_ceil(tile), 1],
                128,
            );
            cmd.finish().unwrap();
            let reference = unsafe { a.read_f32(0, rows * n) };
            for (kernel, bm) in [
                (baseline, tile),
                ("gemma_f32_8", 8),
                ("gemma_f32_16", 16),
                ("gemma_f32_32", 32),
            ] {
                let mut times = Vec::new();
                for _ in 0..5 {
                    let cmd = device.begin().unwrap();
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, &x, &b],
                        &p,
                        [n.div_ceil(16), rows.div_ceil(bm), 1],
                        128,
                    );
                    times.push(cmd.finish().unwrap() * 1000.);
                }
                let actual = unsafe { b.read_f32(0, rows * n) };
                let error = reference
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(actual.iter().all(|x| x.is_finite()));
                assert!(error < 0.05, "{name} {kernel} rows={rows} error={error}");
                times.sort_by(f64::total_cmp);
                println!(
                    "{name} rows={rows} {kernel}: {:.3} ms, max error {error}",
                    times[2]
                );
            }
        }
    }
}
