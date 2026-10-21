use super::*;

#[test]
#[ignore = "requires Bonsai MLX; short generated tails restore exact page-boundary state"]
fn bonsai_short_generated_prefix_is_exact() {
    let path = std::env::var("PADDOCK_BONSAI_MLX").unwrap();
    for live in [1, 4] {
        let mut m = Qwen35::load(Path::new(&path), 1024, live, None).unwrap();
        let allocated = m.device.allocated_bytes();
        let capacity = m.cache.len();
        let prompts = (0..live)
            .map(|slot| {
                (0..113)
                    .map(|i| 1000 + i + slot as u32 * 397)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (slot, prompt) in prompts.iter().enumerate() {
            m.forward_prefill(slot, prompt).unwrap();
            assert_eq!(
                retention::mark_anchor(&mut m.cache, prompt, &[], 113),
                Some(112)
            );
        }
        for pos in 113..144 {
            m.forward_batch(&vec![7000 + pos; live], &vec![pos; live])
                .unwrap();
        }
        let histories = m.slots[..live]
            .iter()
            .map(|s| s.history.clone())
            .collect::<Vec<_>>();
        for h in &histories {
            assert!(m.cache.iter().any(|c| c.history == *h && !c.reserved));
        }
        for s in &mut m.slots[..live] {
            s.prefill_end = 145;
        }
        let expected = m
            .forward_batch(&vec![7999; live], &vec![144; live])
            .unwrap();
        m.release_inactive_slots(&[false; 4]);
        for (slot, h) in histories.iter().enumerate() {
            let mut next = h.clone();
            next.push(7999);
            assert_eq!(m.prepare(slot, &next).unwrap(), 144);
        }
        let actual = m
            .forward_batch(&vec![7999; live], &vec![144; live])
            .unwrap();
        assert_eq!(
            actual, expected,
            "page-boundary restore changed logits c={live}"
        );
        assert_eq!(m.device.allocated_bytes(), allocated);
        assert_eq!(m.cache.len(), capacity);
        for prompt in prompts {
            assert!(m.cache.iter().any(|c| c.history == prompt[..112]));
        }
    }
}

#[test]
#[ignore = "requires Bonsai MLX; unanchored completions keep both prompt and generated-tail reuse"]
fn bonsai_generated_prefix_preserves_raw_prompt_reuse() {
    let path = std::env::var("PADDOCK_BONSAI_MLX").unwrap();
    let mut m = Qwen35::load(Path::new(&path), 1024, 1, None).unwrap();
    let prompt = (1000..1113).collect::<Vec<_>>();
    m.forward_prefill(0, &prompt).unwrap();
    let expected = m.forward_prefill(0, &prompt).unwrap();
    assert_eq!(m.take_prefill_reused(0), 112);
    assert!(m.cache.iter().all(|c| !c.anchor));
    for pos in 113..256 {
        m.forward_batch(&[7000 + pos], &[pos]).unwrap();
    }
    assert!(m.cache.iter().any(|c| c.history.len() == 256));
    assert!(m.cache.iter().any(|c| c.history == prompt[..112]));
    m.release_inactive_slots(&[false]);
    let actual = m.forward_prefill(0, &prompt).unwrap();
    assert_eq!(m.take_prefill_reused(0), 112);
    assert_eq!(
        actual, expected,
        "a generated tail must not change prompt reuse"
    );
}

#[test]
#[ignore = "requires Bonsai MLX; exact live-state continuation, release, COW and cancellation"]
fn bonsai_generated_prefix_is_bounded_and_exact() {
    let path = std::env::var("PADDOCK_BONSAI_MLX").unwrap();
    for live in [1, 4] {
        let mut m = Qwen35::load(Path::new(&path), 4096, live, None).unwrap();
        let allocated = m.device.allocated_bytes();
        let capacity = m.cache.len();
        for slot in 0..live {
            let prompt = (0..113)
                .map(|i| 1000 + i + slot as u32 * 397)
                .collect::<Vec<_>>();
            m.forward_prefill(slot, &prompt).unwrap();
            assert_eq!(
                retention::mark_anchor(&mut m.cache, &prompt, &[], 113),
                Some(112)
            );
        }
        for pos in 113..256 {
            m.forward_batch(&vec![7000 + pos; live], &vec![pos; live])
                .unwrap();
        }
        let histories = m.slots[..live]
            .iter()
            .map(|s| s.history.clone())
            .collect::<Vec<_>>();
        for h in &histories {
            assert!(m.cache.iter().any(|c| c.history == *h && !c.reserved));
        }
        // Both arms process the next request's suffix as prompt work. The
        // oracle is uninterrupted live state, NOT a differently rounded cold
        // recomputation of the generated prefix.
        for s in &mut m.slots[..live] {
            s.prefill_end = 257;
        }
        let expected = m
            .forward_batch(&vec![7999; live], &vec![256; live])
            .unwrap();
        m.release_inactive_slots(&[false; 4]);
        for (slot, h) in histories.iter().enumerate() {
            let mut next = h.clone();
            next.push(7999);
            assert_eq!(m.prepare(slot, &next).unwrap(), 256);
        }
        let actual = m
            .forward_batch(&vec![7999; live], &vec![256; live])
            .unwrap();
        assert_eq!(
            actual, expected,
            "restoring generated state changed logits c={live}"
        );
        if live == 1 {
            // The real two-entry single-user cache retains its original
            // anchor when the generated suffix branches before the tail.
            let mut branch = histories[0].clone();
            branch[200] = 12345;
            branch.push(7999);
            assert_eq!(m.prepare(0, &branch).unwrap(), 112);
        }
        // A cancelled new request must not publish unexecuted tokens.
        m.release_inactive_slots(&[false; 4]);
        let cancelled = (19000..19301).collect::<Vec<_>>();
        m.prefill_begin(live - 1, cancelled.clone()).unwrap();
        assert!(m.prefill_abort(live - 1));
        assert!(!m.cache.iter().any(|c| c.history == cancelled));
        assert_eq!(m.cache.len(), capacity);
        assert_eq!(m.device.allocated_bytes(), allocated);
        assert!(m.cache.iter().all(|c| !c.reserved));
        eprintln!("BONSAI_CONTINUATION_EXACT c={live} tokens=256 allocated={allocated}");
    }
}

#[test]
fn bonsai_prompt_projection_is_slice_stable() {
    let device = MetalDevice::new(None).unwrap();
    if !device.tensor_accelerated() {
        return;
    }
    let (k, n, rows) = (512usize, 67usize, 129usize);
    let input = (0..k * rows)
        .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
        .collect::<Vec<_>>();
    let x = device
        .upload_parts(&[&input
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()])
        .unwrap();
    let mut bytes = (0..k * n / 16)
        .flat_map(|i| {
            (0..16)
                .fold(0u32, |word, j| {
                    word | (((i * 37 + j * 11) % 3) as u32) << (j * 2)
                })
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    bytes.extend((0..k * n / 128).flat_map(|i| {
        half::f16::from_f32((i % 103 + 1) as f32 / 1031.)
            .to_bits()
            .to_le_bytes()
    }));
    let w = device.upload_parts(&[&bytes]).unwrap();
    let sentinel = 12345.0f32;
    let out = device.alloc((n * rows + 16) * 4).unwrap();
    let mut expected = Vec::new();
    for (staging, slice) in [129usize, 128, 64, 33, 32, 13, 4, 1]
        .into_iter()
        .flat_map(|slice| [(0, slice), (1, slice), (2, slice)])
    {
        unsafe {
            out.write_u32(&vec![sentinel.to_bits(); n * rows + 16]);
        }
        let cmd = device.begin().unwrap();
        for first in (0..rows).step_by(slice) {
            let count = slice.min(rows - first);
            let bm = if staging == 2 {
                128
            } else if count <= 16 {
                16
            } else if count <= 32 {
                32
            } else if count >= 128 {
                128
            } else {
                64
            };
            cmd.dispatch_at(
                if staging == 2 {
                    "bonsai_prehalf128x256"
                } else if staging == 1 {
                    match bm {
                        16 => "bonsai_prefill16",
                        32 => "bonsai_prehalf32x128",
                        64 => "bonsai_prehalf64x128",
                        _ => "bonsai_prehalf128x128",
                    }
                } else if bm == 16 {
                    "bonsai_prefill16"
                } else if bm == 32 {
                    "bonsai_prefill32"
                } else if bm == 128 {
                    "bonsai_prefill128"
                } else {
                    "bonsai_prefill64"
                },
                &[&w, &x, &out],
                &[0, first * k * 4, first * n * 4],
                &[k as u32, n as u32, count as u32, bonsai::AFFINE2],
                [n.div_ceil(32), count.div_ceil(bm), 1],
                128,
            );
        }
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, n * rows + 16) };
        assert!(
            actual[..n * rows]
                .iter()
                .all(|v| v.is_finite() && *v != sentinel)
        );
        assert!(actual[n * rows..].iter().all(|v| *v == sentinel));
        if expected.is_empty() {
            let mut max_abs = 0.0f64;
            for row in 0..rows {
                for col in 0..n {
                    let mut reference = 0.0f64;
                    for j in 0..k {
                        let i = col * k + j;
                        let trit = ((i / 16 * 37 + i % 16 * 11) % 3) as f64 - 1.;
                        let scale =
                            half::f16::from_f32((i / 128 % 103 + 1) as f32 / 1031.).to_f64();
                        reference += f64::from(input[row * k + j]) * trit * scale;
                    }
                    max_abs = max_abs.max((f64::from(actual[row * n + col]) - reference).abs());
                }
            }
            assert!(max_abs < 0.003, "prompt projection error {max_abs}");
            eprintln!("BONSAI_PROMPT_PROJECTION_MAX_ABS {max_abs}");
            expected = actual;
        } else {
            assert_eq!(
                actual, expected,
                "prompt contraction changed at slice={slice}, staging={staging}"
            );
        }
    }
}

#[test]
fn bonsai_specialized_projections_are_exact_and_bounded() {
    let device = MetalDevice::new(None).unwrap();
    let k = 512usize;
    let input = (0..k * 512)
        .map(|i| ((i * 139 % 997) as f32 - 498.) / 317.31)
        .collect::<Vec<_>>();
    let x = device
        .upload_parts(&[&input
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()])
        .unwrap();
    let dimensions = [19usize, 35, 67];
    let weights = dimensions.map(|n| {
        let mut bytes = (0..k * n / 16)
            .flat_map(|i| {
                (0..16)
                    .fold(0u32, |word, j| {
                        word | (((i * 37 + j * 11) % 3) as u32) << (j * 2)
                    })
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        bytes.extend((0..k * n / 128).flat_map(|i| {
            half::f16::from_f32((i % 103 + 1) as f32 / 1031.)
                .to_bits()
                .to_le_bytes()
        }));
        device.upload_parts(&[&bytes]).unwrap()
    });
    let sentinel = 12345.0f32;
    for rows in [
        1usize, 2, 3, 4, 5, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512,
    ] {
        let outputs = dimensions.map(|n| {
            device
                .upload_parts(&[&vec![sentinel; n * rows + 16]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()])
                .unwrap()
        });
        let baseline = if rows == 1 {
            "bonsai_vectors1"
        } else if rows <= 4 {
            "bonsai_vectors4"
        } else if rows <= 32 {
            "bonsai_mm32"
        } else {
            "bonsai_mm64"
        };
        let bm = if rows == 1 {
            1
        } else if rows <= 4 {
            4
        } else if rows <= 32 {
            32
        } else {
            64
        };
        let cmd = device.begin().unwrap();
        for i in 0..3 {
            cmd.dispatch(
                baseline,
                &[&weights[i], &x, &outputs[i]],
                &[k as u32, dimensions[i] as u32, rows as u32, bonsai::AFFINE2],
                [dimensions[i].div_ceil(16), rows.div_ceil(bm), 1],
                128,
            );
        }
        cmd.finish().unwrap();
        // SAFETY: all producer commands have completed; include guard words.
        let expected = (0..3)
            .map(|i| unsafe { outputs[i].read_f32(0, dimensions[i] * rows + 16) })
            .collect::<Vec<_>>();
        for i in 0..3 {
            unsafe {
                outputs[i].write_u32(&vec![sentinel.to_bits(); dimensions[i] * rows + 16]);
            }
        }
        let kernel = if rows <= 4 {
            [
                "bonsai_bits1",
                "bonsai_bits2",
                "bonsai_bits3",
                "bonsai_bits4",
            ][rows - 1]
        } else if rows <= 32 {
            "bonsai_tile32x32x64"
        } else {
            "bonsai_tile64x32x64"
        };
        let cmd = device.begin().unwrap();
        for i in 0..3 {
            cmd.dispatch(
                kernel,
                &[&weights[i], &x, &outputs[i]],
                &[k as u32, dimensions[i] as u32, rows as u32, bonsai::AFFINE2],
                [
                    dimensions[i].div_ceil(if rows <= 4 { 16 } else { 32 }),
                    rows.div_ceil(bm),
                    1,
                ],
                128,
            );
        }
        cmd.finish().unwrap();
        for i in 0..3 {
            let actual = unsafe { outputs[i].read_f32(0, dimensions[i] * rows + 16) };
            assert_eq!(actual, expected[i], "{kernel} n={}", dimensions[i]);
            assert!(
                actual[dimensions[i] * rows..]
                    .iter()
                    .all(|v| *v == sentinel)
            );
        }
        if rows > 4 {
            continue;
        }
        for planes in [2, 3] {
            for i in 0..3 {
                unsafe {
                    outputs[i].write_u32(&vec![sentinel.to_bits(); dimensions[i] * rows + 16]);
                }
            }
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                [
                    "bonsai_bits_multi1",
                    "bonsai_bits_multi2",
                    "bonsai_bits_multi3",
                    "bonsai_bits_multi4",
                ][rows - 1],
                &[
                    &weights[0],
                    &weights[1],
                    &weights[2],
                    &x,
                    &outputs[0],
                    &outputs[1],
                    &outputs[2],
                ],
                &[
                    k as u32,
                    dimensions[0] as u32,
                    dimensions[1] as u32,
                    if planes == 3 { dimensions[2] as u32 } else { 0 },
                    rows as u32,
                ],
                [
                    dimensions[..planes].iter().map(|n| n.div_ceil(16)).sum(),
                    1,
                    1,
                ],
                128,
            );
            cmd.finish().unwrap();
            for i in 0..3 {
                let actual = unsafe { outputs[i].read_f32(0, dimensions[i] * rows + 16) };
                if i < planes {
                    assert_eq!(
                        actual, expected[i],
                        "multi rows={rows} planes={planes} i={i}"
                    );
                } else {
                    assert!(actual.iter().all(|v| *v == sentinel));
                }
            }
        }
    }
}

#[test]
#[ignore = "requires downloaded Bonsai; paired cold-cache execution timing and exact arithmetic, not HTTP"]
fn bonsai_execution_cost() {
    execution_cost(false);
}

#[test]
#[ignore = "requires downloaded Bonsai; paired phase-local prefill cost, not HTTP; old/new arithmetic differ"]
fn bonsai_prefill_execution_cost() {
    execution_cost(true);
}

#[test]
#[ignore = "requires Bonsai; repeated warm-prefix suffix cost, not HTTP"]
fn bonsai_cached_suffix_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            bonsai::BASELINE_PREFILL.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 1, None).unwrap();
    for tail in [1usize, 2, 4, 8, 16] {
        for old in [true, false] {
            bonsai::BASELINE_PREFILL.with(|v| v.set(old));
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            let prompt = (0..512 + tail)
                .map(|i| 1000 + (i % 500) as u32)
                .collect::<Vec<_>>();
            let expected = model.forward_prefill(0, &prompt).unwrap();
            let mut times = Vec::new();
            for _ in 0..4 {
                model.reset();
                let start = std::time::Instant::now();
                let got = model.forward_prefill(0, &prompt).unwrap();
                times.push(start.elapsed().as_secs_f64());
                assert_eq!(model.slots[0].reused, 512);
                assert_eq!(
                    got, expected,
                    "cache suffix changed logits: old={old} tail={tail}"
                );
            }
            eprintln!(
                "BONSAI_SUFFIX_COST {}",
                serde_json::json!({"baseline":old,"tail":tail,"warm_s":times})
            );
        }
    }
}

fn execution_cost(prefill: bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            bonsai::BASELINE_PROJECTIONS.with(|v| v.set(false));
            bonsai::BASELINE_PREFILL.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut signatures = std::collections::BTreeMap::new();
    for round in 0..4 {
        for (live, length) in [(1usize, 32usize), (4, 32), (1, 1024), (4, 1024)] {
            for turn in 0..2 {
                let baseline = (turn + round) % 2 == 0;
                bonsai::BASELINE_PROJECTIONS.with(|v| v.set(baseline && !prefill));
                bonsai::BASELINE_PREFILL.with(|v| v.set(baseline || !prefill));
                model.reset();
                for cache in &mut model.cache {
                    cache.table.clear(&mut model.pool);
                    cache.history.clear();
                }
                for slot in 0..live {
                    model
                        .prefill_begin(
                            slot,
                            (0..length)
                                .map(|i| 1000 + ((i + slot * 17) % 500) as u32)
                                .collect(),
                        )
                        .unwrap();
                }
                let mut logits = vec![Vec::new(); live];
                let start = std::time::Instant::now();
                while !model.pending.is_empty() {
                    for (slot, values, _) in model.forward_mixed(&[], 512).unwrap().1 {
                        logits[slot] = values;
                    }
                }
                let ttft = start.elapsed().as_secs_f64();
                let mut hash = blake3::Hasher::new();
                let mut gpu = Vec::new();
                let mut wall = Vec::new();
                for step in 0..32 {
                    for values in &logits {
                        for value in values {
                            assert!(value.is_finite());
                            hash.update(&value.to_bits().to_le_bytes());
                        }
                    }
                    let rows = (0..live)
                        .map(|slot| {
                            (
                                slot,
                                7000 + step as u32,
                                model.slots[slot].history.len() as u32,
                            )
                        })
                        .collect::<Vec<_>>();
                    let start = std::time::Instant::now();
                    let (values, _) = model.forward_mixed(&rows, 0).unwrap();
                    wall.push(start.elapsed().as_secs_f64());
                    gpu.push(model.last_gpu_seconds);
                    logits = values
                        .chunks_exact(model.vocab)
                        .map(|x| x.to_vec())
                        .collect();
                }
                let signature = hash.finalize().to_hex().to_string();
                let expected = signatures
                    .entry((live, length, baseline && prefill))
                    .or_insert_with(|| signature.clone());
                eprintln!(
                    "BONSAI_EXECUTION {}",
                    serde_json::json!({"round":round,"baseline":baseline,"prefill_comparison":prefill,"live":live,"prompt_tokens":length,"ttft_s":ttft,"decode_gpu_s":gpu,"decode_wall_s":wall,"logits_blake3":signature,"exact":&signature==expected,"allocated_bytes":model.device.allocated_bytes()})
                );
                assert_eq!(
                    &signature, expected,
                    "baseline={baseline} live={live} length={length}"
                );
            }
        }
    }
}

#[test]
#[ignore = "real-shape packed projection diagnostic; not serving throughput"]
fn bonsai_projection_cost() {
    let device = MetalDevice::new(None).unwrap();
    for (k, n) in [
        (5120usize, 17408usize),
        (17408, 5120),
        (6144, 5120),
        (5120, 10240),
        (5120, 1024),
        (5120, 248320),
    ] {
        let mut seed = 41u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let mut bytes = Vec::with_capacity(k * n / 4 + k * n / 64);
        for _ in 0..k * n / 16 {
            let word = (0..16).fold(0, |word, j| word | (random() % 3) << (j * 2));
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for _ in 0..k * n / 128 {
            bytes.extend_from_slice(
                &half::f16::from_f32((random() % 100 + 1) as f32 / 8192.)
                    .to_bits()
                    .to_le_bytes(),
            );
        }
        let w = device.upload_parts(&[&bytes]).unwrap();
        drop(bytes);
        let input = (0..k * 512)
            .map(|_| (random() % 65536) as f32 / 32768. - 1.)
            .collect::<Vec<_>>();
        let x = device
            .upload_parts(&[&input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()])
            .unwrap();
        for m in [1usize, 4, 32, 128, 512] {
            if n > 17408 && m > 4 {
                continue;
            }
            let out = device.alloc(m * n * 4).unwrap();
            let mut kernels: Vec<(&str, usize, usize)> = if m <= 4 {
                vec![("bonsai_vectors1", 16, 1), ("bonsai_vectors4", 16, 4)]
            } else {
                vec![
                    ("bonsai_mm32", 16, 32),
                    ("bonsai_mm64", 16, 64),
                    ("bonsai_tile32x32x64", 32, 32),
                    ("bonsai_tile64x32x64", 32, 64),
                ]
            };
            if m == 1 {
                kernels.push(("bonsai_full1", 16, 1));
                kernels.push(("bonsai_bits1", 16, 1));
            }
            if m == 4 {
                kernels.push(("bonsai_full4", 16, 4));
                kernels.push(("bonsai_bits4", 16, 4));
            }
            if n <= 17408 {
                kernels.extend([
                    ("bonsai_prefill16", 32, 16),
                    ("bonsai_prefill32", 32, 32),
                    ("bonsai_prefill64", 32, 64),
                    ("bonsai_prefill128", 32, 128),
                    ("bonsai_prehalf128x128", 32, 128),
                    ("bonsai_prehalf32x128", 32, 32),
                    ("bonsai_prehalf64x128", 32, 64),
                    ("bonsai_prehalf128x256", 32, 128),
                ]);
            }
            let mut reference = Vec::new();
            let prefill_reference = if n <= 17408 {
                let cmd = device.begin().unwrap();
                cmd.dispatch(
                    "bonsai_prefill128",
                    &[&w, &x, &out],
                    &[k as u32, n as u32, m as u32, bonsai::AFFINE2],
                    [n.div_ceil(32), m.div_ceil(128), 1],
                    128,
                );
                cmd.finish().unwrap();
                // SAFETY: completed dispatch covers all output rows.
                unsafe { out.read_f32(0, n * m) }
            } else {
                Vec::new()
            };
            for round in 0..4 {
                for order in 0..kernels.len() {
                    let (kernel, bn, bm) = kernels[(order + round) % kernels.len()];
                    let cmd = device.begin().unwrap();
                    for _ in 0..3 {
                        cmd.dispatch(
                            kernel,
                            &[&w, &x, &out],
                            &[k as u32, n as u32, m as u32, bonsai::AFFINE2],
                            [n.div_ceil(bn), m.div_ceil(bm), 1],
                            128,
                        );
                    }
                    let ms = cmd.finish().unwrap() * 1000. / 3.;
                    // SAFETY: completed dispatch covers the entire output.
                    let actual = unsafe { out.read_f32(0, n * m) };
                    assert!(actual.iter().all(|v| v.is_finite()), "{kernel}");
                    if kernel.starts_with("bonsai_prehalf") {
                        let changed = actual
                            .iter()
                            .zip(&prefill_reference)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        assert_eq!(
                            changed, 0,
                            "{kernel} k={k} n={n} m={m} changes current prefill arithmetic"
                        );
                    }
                    if reference.is_empty() {
                        reference = actual.clone();
                    }
                    let max_abs = actual
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if m <= 4
                        && !kernel.starts_with("bonsai_prefill")
                        && !kernel.starts_with("bonsai_prehalf")
                    {
                        assert_eq!(actual, reference, "{kernel}");
                    } else {
                        assert!(max_abs < 0.001, "{kernel} max_abs={max_abs}");
                    }
                    if round > 0 {
                        eprintln!(
                            "BONSAI_PROJECTION {}",
                            serde_json::json!({"k":k,"n":n,"m":m,"kernel":kernel,"round":round,"ms":ms,"max_abs":max_abs})
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bonsai_rotation_projection_and_embedding() {
    objc2::rc::autoreleasepool(|_| {
        let device = MetalDevice::new(None).unwrap();
        let (k, n, rows) = (1024, 19, 7);
        let input: Vec<f32> = (0..k * rows)
            .map(|i| ((i * 17 % 137) as f32 - 68.) / 128.)
            .collect();
        let signs: Vec<f32> = (0..k).map(|i| if i % 3 == 0 { -1. } else { 1. }).collect();
        let upload = |v: &[f32]| {
            device
                .upload_parts(&[&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()])
                .unwrap()
        };
        let x = upload(&input);
        let sign = upload(&signs);
        let out = device.alloc(k * rows * 4).unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "bonsai_rotate",
            &[&x, &sign, &out],
            &[k as u32, rows as u32],
            [1, rows, 1],
            256,
        );
        cmd.finish().unwrap();
        let fwht = |row: &mut [f32]| {
            for bit in 0..10 {
                let s = 1 << bit;
                for base in (0..k).step_by(s * 2) {
                    for j in 0..s {
                        let (a, b) = (row[base + j], row[base + j + s]);
                        row[base + j] = a + b;
                        row[base + j + s] = a - b;
                    }
                }
            }
            for v in row {
                *v /= 32.;
            }
        };
        let mut expected = input.clone();
        for row in expected.chunks_exact_mut(k) {
            for (x, s) in row.iter_mut().zip(&signs) {
                *x *= s;
            }
            fwht(row);
        }
        // SAFETY: command completed and all ranges match owned allocations.
        assert_eq!(unsafe { out.read_f32(0, k * rows) }, expected);
        let codes: Vec<u32> = (0..k * n / 16)
            .map(|i| (0..16).fold(0, |w, j| w | (((i * 16 + j) % 3) as u32) << (j * 2)))
            .collect();
        let mut packed: Vec<u8> = codes.iter().flat_map(|c| c.to_le_bytes()).collect();
        packed.extend(std::iter::repeat_n(0x3800u16, k * n / 128).flat_map(|s| s.to_le_bytes()));
        let w = device.upload_parts(&[&packed]).unwrap();
        let output = device.alloc(n * rows * 4).unwrap();
        for kernel in [
            "bonsai_mv",
            "bonsai_vectors1",
            "bonsai_vectors4",
            "bonsai_mm32",
            "bonsai_mm64",
        ] {
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                kernel,
                &[&w, &x, &output],
                &[k as u32, n as u32, rows as u32, bonsai::AFFINE2],
                if kernel == "bonsai_mv" {
                    [n.div_ceil(4), rows, 1]
                } else if kernel == "bonsai_vectors1" {
                    [n.div_ceil(16), rows, 1]
                } else if kernel == "bonsai_vectors4" {
                    [n.div_ceil(16), rows.div_ceil(4), 1]
                } else {
                    [n.div_ceil(16), 1, 1]
                },
                128,
            );
            cmd.finish().unwrap();
            let actual = unsafe { output.read_f32(0, n * rows) };
            for row in 0..rows {
                for col in 0..n {
                    let reference: f32 = (0..k)
                        .map(|j| input[row * k + j] * (((col * k + j) % 3) as f32 - 1.) * 0.5)
                        .sum();
                    assert_eq!(
                        actual[row * n + col],
                        reference,
                        "{kernel} row={row} col={col}"
                    );
                }
            }
        }
        let ids = device
            .upload_parts(&[&[0u32, 18]
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<_>>()])
            .unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "bonsai_embed",
            &[&w, &ids, &sign, &out],
            &[k as u32, n as u32, 2],
            [1, 2, 1],
            256,
        );
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, k * 2) };
        for (row, id) in [0, 18].into_iter().enumerate() {
            let mut expected: Vec<f32> = (0..k)
                .map(|i| (((id * k + i) % 3) as f32 - 1.) * 0.5)
                .collect();
            fwht(&mut expected);
            for (i, v) in expected.iter_mut().enumerate() {
                *v = half::f16::from_f32(*v * signs[i]).to_f32();
            }
            assert_eq!(&actual[row * k..(row + 1) * k], expected);
        }
    });
}

#[test]
#[ignore = "requires PADDOCK_METAL_BONSAI_MODEL and downloaded checkpoint"]
fn bonsai_model_smoke() {
    let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let prompt = "<|im_start|>user\nWhat is 2 + 3? Reply with just the number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = tokenizer.encode(prompt).unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let start = std::time::Instant::now();
    let mut logits = model.forward_prefill(0, &tokens).unwrap();
    let ttft = start.elapsed().as_secs_f64();
    let mut generated = Vec::new();
    let mut decode = Vec::new();
    for _ in 0..32 {
        assert!(logits.iter().all(|v| v.is_finite()));
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        generated.push(next);
        if tokenizer.stop_ids().contains(&next) {
            break;
        }
        logits = model.forward(next).unwrap();
        decode.push(model.last_gpu_seconds);
    }
    eprintln!(
        "BONSAI_SMOKE {}",
        serde_json::json!({"prompt_tokens":tokens,"generated":generated,
        "text":tokenizer.decode(&generated, false).unwrap(),"ttft_s":ttft,"decode_gpu_s":decode,"weight_bytes":model.weight_bytes})
    );
    assert!(tokenizer.decode(&generated, false).unwrap().contains('5'));
}

#[test]
#[ignore = "requires PADDOCK_METAL_BONSAI_MODEL and PADDOCK_BONSAI_REFERENCE"]
pub(super) fn bonsai_complete_generation_reference() {
    let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
    let reference = std::path::PathBuf::from(std::env::var("PADDOCK_BONSAI_REFERENCE").unwrap());
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("results.json")).unwrap()).unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let cases = data["cases"].as_array().unwrap();
    let max_tokens = data["max_new_tokens"].as_u64().unwrap() as usize;
    assert!((1..=256).contains(&max_tokens) && cases.len().is_multiple_of(4));
    let prompts: Vec<Vec<u32>> = cases
        .iter()
        .map(|c| {
            let ids = tokenizer.encode(c["prompt"].as_str().unwrap()).unwrap();
            assert_eq!(serde_json::json!(ids), c["tokens"]);
            ids
        })
        .collect();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    let mut mismatches = Vec::new();
    for live in [1, 4] {
        for first in (0..cases.len()).step_by(live) {
            model.reset();
            for cache in &mut model.cache {
                cache.table.clear(&mut model.pool);
                cache.history.clear();
            }
            let mut outputs: Vec<Vec<u32>> = vec![Vec::new(); live];
            let mut logits = Vec::new();
            let start = std::time::Instant::now();
            if live == 1 {
                logits.push(model.forward_prefill(0, &prompts[first]).unwrap());
            } else {
                for slot in 0..live {
                    model
                        .prefill_begin(slot, prompts[first + slot].clone())
                        .unwrap();
                }
                logits.resize_with(live, Vec::new);
                while !model.pending.is_empty() {
                    for (slot, values, _) in model.forward_mixed(&[], 512).unwrap().1 {
                        logits[slot] = values;
                    }
                }
            }
            let ttft = start.elapsed().as_secs_f64();
            let mut gpu = Vec::new();
            for _ in 0..max_tokens {
                let mut decodes = Vec::new();
                for slot in 0..live {
                    if outputs[slot]
                        .last()
                        .is_some_and(|t| tokenizer.stop_ids().contains(t))
                    {
                        continue;
                    }
                    assert!(logits[slot].iter().all(|v| v.is_finite()));
                    let token = logits[slot]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0 as u32;
                    outputs[slot].push(token);
                    if !tokenizer.stop_ids().contains(&token) {
                        decodes.push((slot, token, model.slots[slot].history.len() as u32));
                    }
                }
                if decodes.is_empty() {
                    break;
                }
                let (values, _) = model.forward_mixed(&decodes, 0).unwrap();
                gpu.push(model.last_gpu_seconds);
                for (i, &(slot, _, _)) in decodes.iter().enumerate() {
                    logits[slot] = values[i * model.vocab..(i + 1) * model.vocab].to_vec();
                }
            }
            for (slot, output) in outputs.iter().enumerate() {
                eprintln!(
                    "BONSAI_REFERENCE {}",
                    serde_json::json!({"live":live,"case":first+slot,"ttft_s":ttft,"gpu_s":gpu,"generated":output,"text":tokenizer.decode(output,false).unwrap()})
                );
                if serde_json::json!(output) != cases[first + slot]["generated"] {
                    mismatches.push((live, first + slot));
                }
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "generation mismatches: {mismatches:?}"
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_BONSAI_MODEL; native bundled-tower semantic smoke, not full vision parity"]
fn bonsai_bundled_vision_smoke() {
    use paddock_engine::service::MmChunk;
    let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 1, None).unwrap();
    model.attach_vision(Path::new(&path)).unwrap();
    for (rgb, answer) in [
        ([255u8, 0, 0], "red"),
        ([0, 255, 0], "green"),
        ([0, 0, 255], "blue"),
    ] {
        let chunks=vec![
            MmChunk::Text(tokenizer.encode("<|im_start|>user\n<|vision_start|>").unwrap()),
            MmChunk::Image {rgb: std::iter::repeat_n(rgb,256*256).flatten().collect(),w:256,h:256},
            MmChunk::Text(tokenizer.encode("<|vision_end|>\nWhat color is the image? Reply with one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").unwrap()),
        ];
        model.reset();
        let start = std::time::Instant::now();
        let (mut logits, _) = model.prefill_images(0, &chunks).unwrap();
        let ttft = start.elapsed().as_secs_f64();
        let mut output = Vec::new();
        for _ in 0..16 {
            assert!(logits.iter().all(|v| v.is_finite()));
            let token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            output.push(token);
            if tokenizer.stop_ids().contains(&token) {
                break;
            }
            logits = model.forward(token).unwrap();
        }
        let text = tokenizer.decode(&output, false).unwrap();
        eprintln!(
            "BONSAI_VISION {}",
            serde_json::json!({"expected":answer,"text":text,"ttft_s":ttft,"weight_bytes":model.weight_bytes,"allocated_bytes":model.device.allocated_bytes()})
        );
        assert!(
            text.to_lowercase().contains(answer),
            "expected {answer}, got {text}"
        );
    }
}
