use super::*;
use paddock_engine::generator::Generator;

#[test]
#[ignore = "requires Gemma MLX; failed publication must not leave stale ring metadata"]
fn mlx_prefix_publication_failure_invalidates_old_snapshot() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            prefix::FAIL_PUBLISH_FOR_TEST.set(false);
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_MM_MLX").unwrap();
    let mut m = Gemma4::load(Path::new(&path), 512, 1, None).unwrap();
    let tokens = (1000..1128).collect::<Vec<_>>();
    let expected = m.forward_prefill(0, &tokens).unwrap();
    assert_eq!(m.cache[0].history, tokens);
    prefix::FAIL_PUBLISH_FOR_TEST.set(true);
    assert!(m.publish(0).is_err());
    assert!(m.cache[0].history.is_empty());
    assert!(m.cache[0].images.is_empty());
    assert!(m.cache[0].table.blocks().is_empty());
    prefix::FAIL_PUBLISH_FOR_TEST.set(false);
    let actual = m.forward_prefill(0, &tokens).unwrap();
    assert_eq!(m.take_prefill_reused(0), 0);
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "requires Gemma MLX and independent oMLX oracle; batched generation and restore"]
fn nax_batched_generation_and_restore() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            forward::NAX_PREFILL_FOR_TEST.set(None);
        }
    }
    let _reset = Reset;
    let model = std::env::var("PADDOCK_MM_MLX").unwrap();
    let root = std::env::var("PADDOCK_GEMMA_ATTN_ORACLE").unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(Path::new(&root).join("manifest.json")).unwrap())
            .unwrap();
    let ints = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let stops = ints(&manifest["stop"]);
    let max_tokens = manifest["max_tokens"].as_u64().unwrap_or(128) as usize;
    let mut m = Gemma4::load(Path::new(&model), 8192, 4, None).unwrap();
    let best = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0 as u32
    };
    let mut regressions = Vec::new();
    for indices in [[0, 1, 2, 3], [5, 6, 7, 8]] {
        let cases = indices.map(|i| &manifest["cases"][i]);
        let ids = cases.map(|v| ints(&v["ids"]));
        let expected = cases.map(|v| ints(&v["generated"]));
        let mut old = vec![Vec::new(); 4];
        for fast in [false, true] {
            forward::NAX_PREFILL_FOR_TEST.set(Some(fast));
            m.reset();
            while m.evict() {}
            for (slot, prompt) in ids.iter().enumerate() {
                m.prefill_begin(slot, prompt.clone()).unwrap();
            }
            let start = std::time::Instant::now();
            let mut ready = 0;
            let mut logits = vec![Vec::new(); 4];
            while ready < 4 {
                for (slot, row, _) in m.forward_mixed(&[], 512).unwrap().1 {
                    logits[slot] = row;
                    ready += 1;
                }
            }
            let prefill = start.elapsed().as_secs_f64();
            let mut generated = vec![Vec::new(); 4];
            let mut done = [false; 4];
            for _ in 0..max_tokens {
                let mut decodes = Vec::new();
                for slot in 0..4 {
                    if !done[slot] {
                        assert!(logits[slot].iter().all(|v| v.is_finite()));
                        let token = best(&logits[slot]);
                        generated[slot].push(token);
                        done[slot] = stops.contains(&token);
                        if !done[slot] {
                            decodes.push((
                                slot,
                                token,
                                (ids[slot].len() + generated[slot].len() - 1) as u32,
                            ));
                        }
                    }
                }
                if decodes.is_empty() {
                    break;
                }
                let (next, prefills) = m.forward_mixed(&decodes, 512).unwrap();
                assert!(prefills.is_empty());
                for (index, (slot, _, _)) in decodes.iter().enumerate() {
                    logits[*slot] = next[index * m.vocab..(index + 1) * m.vocab].to_vec();
                }
            }
            for slot in 0..4 {
                eprintln!(
                    "GEMMA_NAX_BATCH {}",
                    serde_json::json!({"name":cases[slot]["name"],"fast":fast,"prefill_s":prefill,"matches_oracle":generated[slot]==expected[slot],"matches_old":!fast || generated[slot]==old[slot],"generated":generated[slot]})
                );
                // Existing decode arithmetic can diverge across batch sizes.
                // Do not label that parity: retain the per-case oracle result,
                // and require any changed candidate sequence to match it.
                if fast && generated[slot] != expected[slot] && generated[slot] != old[slot] {
                    regressions.push(indices[slot]);
                }
            }
            if !fast {
                old = generated;
            }
        }
    }
    assert!(
        regressions.is_empty(),
        "new reference regressions: {regressions:?}"
    );
    forward::NAX_PREFILL_FOR_TEST.set(Some(true));
    m.reset();
    while m.evict() {}
    let tokens = ints(&manifest["cases"][6]["ids"]);
    let prefix = &tokens[..tokens.len() - 1];
    assert!(prefix.len() > m.ring);
    m.forward_prefill(0, prefix).unwrap();
    let a = m
        .execute(&[(0, *tokens.last().unwrap(), prefix.len() as u32)], &[0])
        .unwrap();
    let b = m.forward_prefill(1, &tokens).unwrap();
    assert_eq!(m.take_prefill_reused(1), prefix.len());
    assert_eq!(a, b);
    m.prefill_begin(3, tokens.clone()).unwrap();
    assert!(m.prefill_abort(3));
    assert!(m.slots[3].history.is_empty());
    let mut logits = a;
    for step in 0..16 {
        let token = best(&logits);
        let pos = (tokens.len() + step) as u32;
        let a = m.execute(&[(0, token, pos)], &[0]).unwrap();
        let b = m.execute(&[(1, token, pos)], &[0]).unwrap();
        assert_eq!(a, b);
        logits = a;
    }
}

#[test]
fn nax_attention_paged_ring_ragged_and_masked_cache() {
    let d = MetalDevice::new(None).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    for hd in [256usize, 512] {
        // The final case has entirely masked first key tiles for later rows,
        // with large negative scores that expose a fabricated softmax maximum.
        for (rows, offset, window, ring, negative) in [
            (1usize, 0usize, 0usize, 0usize, false),
            (17, 31, 0, 0, false),
            (33, 159, 17, 64, false),
            (32, 129, 17, 96, true),
        ] {
            let heads = 4;
            let kh = 2;
            let slots = [3usize, 1];
            let page_stride = (offset + rows).div_ceil(16).next_power_of_two();
            let total_pages = 4 * page_stride;
            let pages: Vec<_> = (0..total_pages)
                .map(|p| ((p * 5 + 3) % total_pages) as u32)
                .collect();
            let physical = |slot: usize, t: usize| {
                if ring > 0 {
                    slot * ring + t % ring
                } else {
                    pages[slot * page_stride + t / 16] as usize * 16 + t % 16
                }
            };
            let tokens = if ring > 0 { 4 * ring } else { 16 * total_pages };
            let mut keys = vec![half::bf16::NAN; tokens * kh * hd];
            let mut values = keys.clone();
            let qvalue = |slot: usize, row: usize, head: usize, j: usize| {
                half::bf16::from_f32(if negative {
                    1.
                } else {
                    (((slot * 13 + row * 7 + head * 19 + j * 31) % 113) as f32 - 56.) / 128.
                })
            };
            let kvalue = |slot: usize, t: usize, head: usize, j: usize| {
                half::bf16::from_f32(if negative {
                    -4.
                } else {
                    (((slot * 29 + t * 11 + head * 7 + j * 17) % 127) as f32 - 63.) / 128.
                })
            };
            let vvalue = |slot: usize, t: usize, head: usize, j: usize| {
                half::bf16::from_f32(
                    (((slot * 3 + t * 13 + head * 17 + j * 7) % 131) as f32 - 65.) / 128.,
                )
            };
            let low = if window > 0 {
                (offset + 1).saturating_sub(window)
            } else {
                0
            };
            let mut query = Vec::new();
            let mut meta = Vec::new();
            let mut tiles = Vec::new();
            for (index, &slot) in slots.iter().enumerate() {
                for t in low..offset + rows {
                    for head in 0..kh {
                        for j in 0..hd {
                            let at = (physical(slot, t) * kh + head) * hd + j;
                            keys[at] = kvalue(slot, t, head, j);
                            values[at] = vvalue(slot, t, head, j);
                        }
                    }
                }
                for row in 0..rows {
                    meta.extend([slot as u32, (offset + row) as u32]);
                    for head in 0..heads {
                        for j in 0..hd {
                            query.push(qvalue(slot, row, head, j));
                        }
                    }
                }
                for row in (0..rows).step_by(32) {
                    tiles.extend([(index * rows + row) as u32, (rows - row).min(32) as u32]);
                }
            }
            let bf = |v: &[half::bf16]| {
                d.upload(
                    &v.iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let (q, k, v) = (bf(&query), bf(&keys), bf(&values));
            let (meta, pages, tiles_buffer) = (upload(&meta), upload(&pages), upload(&tiles));
            let count = slots.len() * rows * heads * hd;
            let out = d.upload(&vec![0xff; count * 4 + 64]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                &format!("gmlx_nax_prefill{hd}"),
                &[&q, &k, &v, &meta, &pages, &out, &tiles_buffer],
                &[
                    heads as u32,
                    kh as u32,
                    page_stride as u32,
                    window as u32,
                    ring as u32,
                ],
                [heads, tiles.len() / 2, if hd == 512 { 2 } else { 1 }],
                128,
            );
            cmd.finish().unwrap();
            // SAFETY: completed output stores cover count scalars; guard stays allocated.
            let actual = unsafe { out.read_f32(0, count + 16) };
            assert!(actual[..count].iter().all(|v| v.is_finite()));
            assert!(
                actual[count..].iter().all(|v| v.to_bits() == u32::MAX),
                "tail overwritten"
            );
            for (index, &slot) in slots.iter().enumerate() {
                for row in 0..rows {
                    for head in 0..heads {
                        let pos = offset + row;
                        let low = if window > 0 {
                            (pos + 1).saturating_sub(window)
                        } else {
                            0
                        };
                        let dots: Vec<_> = (low..=pos)
                            .map(|t| {
                                (0..hd)
                                    .map(|j| {
                                        qvalue(slot, row, head, j).to_f64()
                                            * kvalue(slot, t, head / (heads / kh), j).to_f64()
                                    })
                                    .sum::<f64>()
                            })
                            .collect();
                        let max = dots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let probabilities: Vec<_> = dots.iter().map(|v| (v - max).exp()).collect();
                        let sum: f64 = probabilities.iter().sum();
                        for j in [0, 7, 127, 128, hd - 1] {
                            let expected = probabilities
                                .iter()
                                .enumerate()
                                .map(|(t, p)| {
                                    p * vvalue(slot, t + low, head / (heads / kh), j).to_f64()
                                })
                                .sum::<f64>()
                                / sum;
                            let got =
                                f64::from(actual[((index * rows + row) * heads + head) * hd + j]);
                            assert!(
                                (got - expected).abs() <= 3e-6 + expected.abs() * 0.004,
                                "hd={hd} rows={rows} slot={slot} row={row} head={head} dim={j} ring={ring}: {got} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires Gemma and oMLX oracle; reference nonregression, not a universal parity gate"]
fn nax_full_generation_probe() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            forward::NAX_PREFILL_FOR_TEST.set(None);
        }
    }
    let _reset = Reset;
    let model = std::env::var("PADDOCK_MM_MLX").unwrap();
    let root = std::env::var("PADDOCK_GEMMA_ATTN_ORACLE").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let stops: Vec<u32> = manifest["stop"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let max_tokens = manifest["max_tokens"].as_u64().unwrap_or(128) as usize;
    let mut failures = Vec::new();
    let mut m = Gemma4::load(Path::new(&model), 8192, 4, None).unwrap();
    for case in manifest["cases"].as_array().unwrap() {
        let ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let oracle: Vec<u32> = case["generated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut expected = None;
        for fast in [false, true] {
            forward::NAX_PREFILL_FOR_TEST.set(Some(fast));
            m.reset();
            while m.evict() {}
            let started = std::time::Instant::now();
            let mut logits = m.forward_prefill(0, &ids).unwrap();
            let prefill = started.elapsed().as_secs_f64();
            let mut generated = Vec::new();
            let mut costs = Vec::new();
            for step in 0..max_tokens {
                assert!(logits.iter().all(|v| v.is_finite()));
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .unwrap()
                    .0 as u32;
                generated.push(token);
                if stops.contains(&token) {
                    break;
                }
                let start = std::time::Instant::now();
                logits = m
                    .forward_batch(&[token], &[(ids.len() + step) as u32])
                    .unwrap();
                costs.push(start.elapsed().as_secs_f64());
            }
            let matches_old = expected.as_ref().is_none_or(|e| e == &generated);
            if expected.is_none() {
                expected = Some(generated.clone());
            }
            eprintln!(
                "GEMMA_NAX_GENERATION {}",
                serde_json::json!({"name":case["name"],"fast":fast,"prompt":ids.len(),"prefill_s":prefill,"decode_s":costs,"matches_old":matches_old,"matches_oracle":generated==oracle,"generated":generated,"allocated":m.device.allocated_bytes()})
            );
            // Preserve every case the baseline gets right. Keep existing
            // mismatches in the emitted evidence instead of calling them
            // parity or changing the fixture to hide them.
            if fast && expected.as_ref() == Some(&oracle) && generated != oracle {
                failures.push(case["name"].clone());
            }
        }
    }
    assert!(
        failures.is_empty(),
        "new reference regressions: {failures:?}"
    );
}

#[test]
#[ignore = "GPU timing and arithmetic diagnostic"]
fn nax_attention_cost() {
    let d = MetalDevice::new(None).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    let u32buf = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    for hd in [256usize, 512] {
        let kh = if hd == 256 { 16 } else { 4 };
        for (rows, offset, window) in [
            (17usize, 0usize, 0usize),
            (128, 512, 0),
            (512, 1536, 0),
            (512, 1536, 1024),
        ] {
            let length = offset + rows;
            let values = |n: usize, seed: usize| {
                (0..n)
                    .map(|i| {
                        half::bf16::from_f32((((i * 37 + seed) % 1999) as f32 / 997.0 - 1.0) * 0.2)
                    })
                    .collect::<Vec<_>>()
            };
            let qv = values(rows * 32 * hd, 3);
            let q = d
                .upload(
                    &qv.iter()
                        .flat_map(|v| v.to_f32().to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let qb = d
                .upload(
                    &qv.iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let kv = d
                .upload(
                    &values(length * kh * hd, 41)
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let vv = d
                .upload(
                    &values(length * kh * hd, 73)
                        .iter()
                        .flat_map(|v| v.to_bits().to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let meta = u32buf(
                &(0..rows)
                    .flat_map(|r| [0, (offset + r) as u32])
                    .collect::<Vec<_>>(),
            );
            let pages = u32buf(&(0..length.div_ceil(16) as u32).collect::<Vec<_>>());
            let tiles = u32buf(
                &(0..rows)
                    .step_by(32)
                    .flat_map(|r| [r as u32, (rows - r).min(32) as u32])
                    .collect::<Vec<_>>(),
            );
            let variants = ["", "nax_"];
            let out: Vec<_> = (0..variants.len())
                .map(|_| d.alloc(rows * 32 * hd * 4).unwrap())
                .collect();
            let mut times = vec![Vec::new(); variants.len()];
            for round in 0..5 {
                for at in 0..variants.len() {
                    let which = (round + at) % variants.len();
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        &format!("gmlx_{}prefill{hd}", variants[which]),
                        &[
                            if which == 1 { &qb } else { &q },
                            &kv,
                            &vv,
                            &meta,
                            &pages,
                            &out[which],
                            &tiles,
                        ],
                        &[32, kh as u32, length.div_ceil(16) as u32, window as u32, 0],
                        [
                            32,
                            rows.div_ceil(32),
                            if which == 1 && hd == 512 { 2 } else { 1 },
                        ],
                        128,
                    );
                    let elapsed = cmd.finish().unwrap();
                    if round > 0 {
                        times[which].push(elapsed);
                    }
                }
            }
            let a: Vec<_> = unsafe { out[0].read_f32(0, rows * 32 * hd) }
                .into_iter()
                .map(|v| half::bf16::from_f32(v).to_f32())
                .collect();
            for which in 1..variants.len() {
                let b = unsafe { out[which].read_f32(0, rows * 32 * hd) };
                assert!(a.iter().chain(b.iter()).all(|v| v.is_finite()));
                let different = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let rounded = a
                    .iter()
                    .zip(&b)
                    .filter(|(a, b)| half::bf16::from_f32(**a) != half::bf16::from_f32(**b))
                    .count();
                let max = a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "GEMMA_COMPACT_ATTN {}",
                    serde_json::json!({"candidate":variants[which],"hd":hd,"rows":rows,"offset":offset,"window":window,"different":different,"rounded_different":rounded,"max":max,"times":[times[0],times[which]]})
                );
                let value = |i: usize, seed: usize| {
                    half::bf16::from_f32((((i * 37 + seed) % 1999) as f32 / 997.0 - 1.0) * 0.2)
                        .to_f64()
                };
                for row in [0, rows / 2, rows - 1] {
                    for head in [0, 7, 31] {
                        let pos = offset + row;
                        let low =
                            (pos + 1).saturating_sub(if window == 0 { pos + 1 } else { window });
                        let dots: Vec<_> = (low..=pos)
                            .map(|t| {
                                (0..hd)
                                    .map(|j| {
                                        value((row * 32 + head) * hd + j, 3)
                                            * value((t * kh + head / (32 / kh)) * hd + j, 41)
                                    })
                                    .sum::<f64>()
                            })
                            .collect();
                        let high = dots.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let probabilities: Vec<_> = dots.iter().map(|x| (x - high).exp()).collect();
                        let denominator = probabilities.iter().sum::<f64>();
                        for dim in [0, 127, 128, hd - 1] {
                            let expected = probabilities
                                .iter()
                                .enumerate()
                                .map(|(t, p)| {
                                    p * value(((t + low) * kh + head / (32 / kh)) * hd + dim, 73)
                                })
                                .sum::<f64>()
                                / denominator;
                            let actual = f64::from(b[(row * 32 + head) * hd + dim]);
                            assert!(
                                (actual - expected).abs() <= 2e-6 + expected.abs() * 0.004,
                                "F64 attention hd={hd} row={row} head={head} dim={dim}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}
