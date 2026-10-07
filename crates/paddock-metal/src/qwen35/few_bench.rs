//! Same-checkpoint end-to-end decode diagnostic, not HTTP/admission timing.
//! Fixed neural depths deliberately bypass the production source/depth policy.
//! EOS is respected; all compared token IDs and timing samples are retained.
use super::*;
use paddock_tokenizer::GgufTokenizer;
use std::time::Instant;

fn clear(model: &mut Qwen35) {
    model.reset();
    for entry in &mut model.cache {
        entry.table.clear(&mut model.pool);
        entry.history.clear();
    }
}

fn pick(row: &[f32]) -> u32 {
    assert!(row.iter().all(|v| v.is_finite()));
    row.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0 as u32
}

#[test]
#[ignore = "requires Qwen MLX and DFlash fixtures; repeated full-model GPU benchmark"]
fn mlx_few_decode_rung() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    let prefix_rung = std::env::var_os("PADDOCK_FEW_PREFIX").is_some();
    let only_depth = std::env::var("PADDOCK_FEW_DEPTH")
        .ok()
        .map(|s| s.parse::<usize>().unwrap());
    let only_c = std::env::var("PADDOCK_FEW_CONCURRENCY")
        .ok()
        .map(|s| s.parse::<usize>().unwrap());
    let rounds: usize = std::env::var("PADDOCK_FEW_ROUNDS")
        .unwrap_or("3".into())
        .parse()
        .unwrap();
    let max_tokens: usize = std::env::var("PADDOCK_FEW_TOKENS")
        .unwrap_or("128".into())
        .parse()
        .unwrap();
    assert!(rounds > 0 && max_tokens >= 2);
    assert!(only_depth.is_none() || prefix_rung);
    assert!(only_depth.is_none_or(|k| k <= 3));
    assert!(only_c.is_none_or(|c| matches!(c, 1 | 4)));
    let tokenizer = GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let stops = tokenizer.stop_ids();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    model.attach_dflash(Path::new(&draft)).unwrap();
    for (case, length, c) in [
        ("code", 128usize, 1usize),
        ("prose", 128, 1),
        ("code", 2048, 1),
        ("code", 2048, 4),
    ] {
        if only_c.is_some_and(|v| v != c) {
            continue;
        }
        let task = if case == "code" {
            "Write a complete Python LRU cache class with type hints, tests, and an explanation. Return working code, not pseudocode."
        } else {
            "Explain how a forest ecosystem recovers after a wildfire, with concrete examples of the different stages."
        };
        let mut prompts = Vec::new();
        for slot in 0..c {
            let prefix = format!("<|im_start|>user\nRequest {slot}. Background notes: ");
            let suffix =
                format!("\nTask: {task}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            let mut prompt = tokenizer.encode(&prefix).unwrap();
            let end = tokenizer.encode(&suffix).unwrap();
            let filler = tokenizer
                .encode(" The system uses a bounded cache and preserves the order of requests.")
                .unwrap();
            while prompt.len() + end.len() < length {
                let take = (length - prompt.len() - end.len()).min(filler.len());
                prompt.extend_from_slice(&filler[..take]);
            }
            prompt.extend(end);
            assert_eq!(prompt.len(), length);
            prompts.push(prompt);
        }
        let variants: &[(bool, usize)] = if prefix_rung {
            &[
                (false, 0),
                (true, 0),
                (false, 1),
                (true, 1),
                (false, 2),
                (true, 2),
                (false, 3),
                (true, 3),
            ]
        } else {
            &[
                (false, 0usize),
                (true, 0),
                (false, 1),
                (false, 3),
                (true, 2),
                (true, 3),
            ]
        };
        let mut expected: Option<Vec<Vec<u32>>> = None;
        for round in 0..=rounds {
            // Rotate and reverse: rotation alone leaves adjacent A/B pairs
            // in the same order in most rounds, concealing clock drift.
            for shift in 0..variants.len() {
                let step = if round % 2 == 0 {
                    shift
                } else {
                    variants.len() - 1 - shift
                };
                let variant = (step + round) % variants.len();
                let (candidate, depth) = variants[variant];
                if c == 1 && only_depth.is_some_and(|v| v != depth) {
                    continue;
                }
                // The exploratory sweep already established that concurrent
                // neural speculation loses. Repeated qualification retains
                // ordinary batching instead of timing an unelected policy.
                if c > 1 && depth != 0 {
                    continue;
                }
                crate::affine::BASELINE_PACKED_FOR_TEST.with(|v| v.set(!prefix_rung && !candidate));
                dflash::FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.set(!prefix_rung || !candidate));
                clear(&mut model);
                // Disable draft execution/tap capture for ordinary decode,
                // while retaining its allocation to avoid repeated loads.
                let held = if depth == 0 {
                    model.dflash.take()
                } else {
                    None
                };
                let init = Instant::now();
                let mut outputs = Vec::new();
                for (slot, prompt) in prompts.iter().enumerate() {
                    outputs.push(vec![pick(&model.forward_prefill(slot, prompt).unwrap())]);
                }
                let prefill_s = init.elapsed().as_secs_f64();
                let start = Instant::now();
                let mut last = vec![0.; c];
                let mut gaps = Vec::new();
                let mut draft_s = 0.;
                let mut verify_s = 0.;
                let mut proposed = 0;
                let mut accepted = 0;
                let mut steps = 0;
                loop {
                    let live: Vec<_> = outputs
                        .iter()
                        .enumerate()
                        .filter(|(_, o)| o.len() < max_tokens && !stops.contains(o.last().unwrap()))
                        .map(|(slot, o)| (slot, *o.last().unwrap()))
                        .collect();
                    if live.is_empty() {
                        break;
                    }
                    let before = Instant::now();
                    let drafts = if depth == 0 {
                        vec![Vec::new(); live.len()]
                    } else {
                        model.dflash_draft(&live, depth).unwrap().unwrap()
                    };
                    draft_s += before.elapsed().as_secs_f64();
                    let reqs: Vec<_> = live
                        .iter()
                        .zip(drafts)
                        .map(|(&(slot, t), draft)| {
                            let tokens = std::iter::once(t)
                                .chain(
                                    draft
                                        .into_iter()
                                        .take(depth.min(max_tokens - outputs[slot].len() - 1)),
                                )
                                .collect();
                            (slot, model.slots[slot].history.len(), tokens)
                        })
                        .collect();
                    let before = Instant::now();
                    let picks = if depth == 0 {
                        model.decode_picks(&reqs).unwrap()
                    } else {
                        model.forward_spec_batch(&reqs).unwrap().unwrap()
                    };
                    verify_s += before.elapsed().as_secs_f64();
                    let now = start.elapsed().as_secs_f64();
                    let mut offset = 0;
                    for (slot, _, chunk) in reqs {
                        let count = 1 + chunk[1..]
                            .iter()
                            .zip(&picks[offset..])
                            .take_while(|(a, b)| a == b)
                            .count();
                        proposed += chunk.len() - 1;
                        accepted += count - 1;
                        gaps.push(now - last[slot]);
                        last[slot] = now;
                        for &token in &picks[offset..offset + count] {
                            outputs[slot].push(token);
                            if stops.contains(&token) {
                                break;
                            }
                        }
                        offset += chunk.len();
                    }
                    steps += 1;
                }
                let seconds = start.elapsed().as_secs_f64();
                if depth == 0 {
                    model.dflash = held;
                }
                let parity = expected.as_ref().is_none_or(|e| *e == outputs);
                if expected.is_none() {
                    expected = Some(outputs.clone());
                }
                gaps.sort_by(f64::total_cmp);
                let generated = outputs.iter().map(|o| o.len() - 1).sum::<usize>();
                eprintln!(
                    "FEW_MODEL {}",
                    serde_json::json!({"case":case,"prompt_tokens":length,"concurrency":c,"candidate":candidate,"prefix_rung":prefix_rung,
                    "draft_depth":depth,"round":round,"warmup":round==0,"max_tokens":max_tokens,"seconds":seconds,
                    "decode_tps":generated as f64/seconds,"prefill_seconds":prefill_s,"draft_seconds":draft_s,"verify_seconds":verify_s,
                    "gap_p99_ms":gaps[((gaps.len() as f64*0.99).ceil() as usize).saturating_sub(1)]*1000.,
                    "steps":steps,"proposed":proposed,"accepted":accepted,"parity":parity,"outputs":outputs,
                    "stopped":outputs.iter().map(|o|stops.contains(o.last().unwrap())).collect::<Vec<_>>(),"allocated_bytes":model.device_mem_used()})
                );
                assert!(parity, "candidate/depth changed greedy output");
            }
        }
    }
    crate::affine::BASELINE_PACKED_FOR_TEST.with(|v| v.set(false));
    dflash::FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.set(false));
}

#[test]
#[ignore = "requires PADDOCK_FEW_GGUF; production vector vs existing narrow MPP tile"]
fn gguf_few_row_matrix_probe() {
    let path = std::env::var("PADDOCK_FEW_GGUF").unwrap();
    let source = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let d = MetalDevice::new(None).unwrap();
    for (name, k, n) in [
        ("blk.0.ffn_gate.weight", 5120usize, 17408usize),
        ("blk.0.ffn_down.weight", 17408, 5120),
    ] {
        let w = Weight::load(&d, &source, name, &[k, n]).unwrap();
        let x = d
            .upload(
                &(0..16 * k)
                    .flat_map(|i| (((i * 37 % 1999) as f32 - 999.) / 113.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let y = d.alloc(16 * n * 4).unwrap();
        let scratch = d.alloc(k * 128 * 2 + 4 * 16 * n * 4).unwrap();
        for rows in [2usize, 3, 4, 5, 8, 16] {
            let dispatch = |cmd: &Commands<'_>, matrix: bool| {
                if matrix {
                    cmd.dispatch(
                        "linear_input_padded",
                        &[&x, &scratch],
                        &[k as u32, n as u32, rows as u32, w.ty, 1f32.to_bits()],
                        [(k * 128).div_ceil(256), 1, 1],
                        256,
                    );
                    w.linear_prepared(cmd, &scratch, &y, rows, 1.);
                } else {
                    projection::project(cmd, &[(&w, &y)], &x, rows, &scratch);
                }
            };
            let mut expected = Vec::new();
            let mut different = 0;
            let mut times = [Vec::new(), Vec::new()];
            for matrix in [false, true] {
                let cmd = d.begin().unwrap();
                dispatch(&cmd, matrix);
                cmd.finish().unwrap();
                let got = unsafe { y.read_f32(0, rows * n) };
                assert!(got.iter().all(|v| v.is_finite()));
                if matrix {
                    different = got
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a.to_bits() != f32::to_bits(**b))
                        .count();
                } else {
                    expected = got;
                }
            }
            for round in 0..9 {
                for shift in 0..2 {
                    let route = (round + shift) % 2;
                    let cmd = d.begin().unwrap();
                    for _ in 0..8 {
                        dispatch(&cmd, route == 1);
                    }
                    let us = cmd.finish().unwrap() * 1e6 / 8.;
                    if round >= 2 {
                        times[route].push(us);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "GGUF_FEW {}",
                serde_json::json!({"weight":name,"type":w.ty,"k":k,"n":n,"rows":rows,
                "median_us":[times[0][3],times[1][3]],"different":different,"elements":rows*n,"samples_us":times})
            );
        }
    }
}
