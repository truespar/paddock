use super::*;
use paddock_engine::generator::Generator;

fn greedy(m: &FlashNext, rows: usize) -> Vec<u32> {
    let cmd = m.device.begin().unwrap();
    cmd.dispatch(
        "spec_argmax",
        &[&m.scratch.logits, &m.scratch.ids],
        &[VOCAB as u32],
        [rows, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    unsafe { m.scratch.ids.read_u32(rows) }
}

#[test]
fn flash_next_mlx_gpu_margin_ties_and_teacher_score() {
    let d = MetalDevice::new(Some(8 << 20)).unwrap();
    let mut values = vec![0f32; 1003];
    values[9] = 4.;
    values[777] = 4.;
    values[1002] = -2.;
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let ids = d.alloc(8).unwrap();
    let scores = d.alloc(12).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "q4b_margin",
        &[&x, &ids, &scores],
        &[1003, 1002],
        [1, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { ids.read_u32(2) }, [9, 777]);
    assert_eq!(unsafe { scores.read_f32(0, 3) }, [4., 4., -2.]);
}

#[test]
#[ignore = "111 GB cost diagnostic; external watchdog required, counters are not serving timings"]
fn flash_next_mlx_prompt_cost() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    let prompt = reference["cases"][1]["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    eprintln!("MLX_COST prefill rows={}", prompt.len());
    m.prefill(0, &prompt).unwrap();
    for token in reference["cases"][1]["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .take(3)
    {
        eprintln!("MLX_COST decode");
        m.forward(token.as_u64().unwrap() as u32).unwrap();
    }
}

#[test]
#[ignore = "111 GB GPU logit diagnostic; external memory watchdog required"]
fn flash_next_mlx_teacher_forced_margins() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    // Both the retained 128-row gate and the wide-prefill candidate can be
    // checked against their actual same-checkpoint upstream chunk contract.
    let chunk = reference["chunk"].as_u64().unwrap() as usize;
    assert!([CHUNK, 256, 512, MLX_CHUNK].contains(&chunk));
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    m.chunk = chunk;
    for case in reference["cases"].as_array().unwrap() {
        let vector = |key: &str| {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        };
        let prompt = vector("prompt_ids");
        let tokens = vector("token_ids");
        m.reset();
        let start = std::time::Instant::now();
        m.prefill(0, &prompt).unwrap();
        let prefill_s = start.elapsed().as_secs_f64();
        let mut rows = vec![];
        for (step, &token) in tokens.iter().enumerate() {
            let cmd = m.device.begin().unwrap();
            cmd.dispatch(
                "q4b_margin",
                &[&m.scratch.logits, &m.scratch.ids, &m.scratch.delta],
                &[VOCAB as u32, token],
                [1, 1, 1],
                256,
            );
            cmd.finish().unwrap();
            // Forward has completed; both control IDs and layer-output
            // scratch are dead until the next walk. Respect the exact grant.
            let actual = unsafe { m.scratch.ids.read_u32(2) };
            let values = unsafe { m.scratch.delta.read_f32(0, 3) };
            assert!(values.iter().all(|v| v.is_finite()));
            rows.push(serde_json::json!({"step":step, "native_ids":actual,
                "native_values":values, "teacher_id":token,
                "reference":case["logits"][step], "gpu_s":m.last_gpu_seconds}));
            // Always feed the reference token, even after a mismatch, so
            // subsequent margins compare the same token history on GPU.
            if step + 1 < tokens.len() {
                m.forward(token).unwrap();
            }
        }
        eprintln!(
            "MLX_MARGINS {}",
            serde_json::json!({"case":case["name"],
            "prefill_s":prefill_s, "rows":rows})
        );
    }
}

#[test]
#[ignore = "111 GB native MLX full-model GPU gate; run under external memory watchdog"]
fn flash_next_mlx_full_generation_control() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let source =
        std::fs::read_to_string(std::env::var("PADDOCK_FLASH_NEXT_MLX_CONTROL").unwrap()).unwrap();
    let reference: serde_json::Value = source
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|v| v.get("batch_prompt_ids").is_some())
        .expect("MLX-VLM GPU batch control");
    let vectors = |key: &str| {
        reference[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let prompts = vectors("batch_prompt_ids");
    let expected = vectors("serial_token_ids");
    let started = std::time::Instant::now();
    let mut m = FlashNext::load(Path::new(&path), 256, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    eprintln!(
        "MLX_NATIVE loaded_s={} resident_bytes={allocated}",
        started.elapsed().as_secs_f64()
    );
    let mut failures = vec![];
    for (i, prompt) in prompts.iter().enumerate() {
        m.reset();
        let start = std::time::Instant::now();
        m.prefill(0, prompt).unwrap();
        eprintln!(
            "MLX_NATIVE prefill case={i} wall_s={} gpu_s={}",
            start.elapsed().as_secs_f64(),
            m.last_gpu_seconds
        );
        let mut actual = vec![];
        for _ in 0..32 {
            let token = greedy(&m, 1)[0];
            actual.push(token);
            if [248044, 248046].contains(&token) {
                break;
            }
            m.forward(token).unwrap();
            eprintln!("MLX_NATIVE decode case={i} gpu_s={}", m.last_gpu_seconds);
        }
        eprintln!(
            "MLX_NATIVE serial case={i} elapsed_s={} tokens={actual:?} expected={:?}",
            start.elapsed().as_secs_f64(),
            expected[i]
        );
        if actual != expected[i] {
            failures.push(format!("serial {i}"));
        }
    }
    m.reset();
    // One ragged prefill walk, then compact only live decode rows. GPU
    // argmax samples the identical final logits used by the serving trait.
    for (slot, prompt) in prompts.iter().enumerate() {
        m.prepare(slot, prompt).unwrap();
    }
    let rows = prompts
        .iter()
        .enumerate()
        .flat_map(|(slot, prompt)| {
            prompt
                .iter()
                .enumerate()
                .map(move |(pos, &token)| (slot, token, pos as u32))
        })
        .collect::<Vec<_>>();
    let outputs = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| (r.2 as usize + 1 == prompts[r.0].len()).then_some(i))
        .collect::<Vec<_>>();
    m.execute(&rows, &outputs).unwrap();
    let mut tokens = greedy(&m, 4);
    let mut active = (0..4).collect::<Vec<_>>();
    let mut actual = vec![vec![]; 4];
    for _ in 0..32 {
        let mut rows = vec![];
        for (&slot, &token) in active.iter().zip(&tokens) {
            actual[slot].push(token);
            if ![248044, 248046].contains(&token) {
                rows.push((slot, token, m.slots[slot].length as u32));
            }
        }
        if rows.is_empty() {
            break;
        }
        active = rows.iter().map(|r| r.0).collect();
        m.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())
            .unwrap();
        tokens = greedy(&m, rows.len());
    }
    eprintln!("MLX_NATIVE ragged tokens={actual:?}");
    if actual != expected {
        failures.push("ragged".into());
    }
    m.reset();
    for (slot, prompt) in prompts.iter().enumerate() {
        m.prefill_begin(slot, prompt.clone()).unwrap();
    }
    let mut decodes = vec![];
    let mut actual = vec![vec![]; 4];
    for _ in 0..40 {
        let (_, done) = m.forward_mixed(&decodes, 33).unwrap();
        let slots = decodes
            .iter()
            .map(|r| r.0)
            .chain(done.iter().map(|r| r.0))
            .collect::<Vec<_>>();
        decodes.clear();
        if !slots.is_empty() {
            let sampled = greedy(&m, slots.len());
            for (slot, token) in slots.into_iter().zip(sampled) {
                actual[slot].push(token);
                if ![248044, 248046].contains(&token) {
                    decodes.push((slot, token, m.slots[slot].length as u32));
                }
            }
        }
        if decodes.is_empty() && m.pending.is_empty() {
            break;
        }
    }
    eprintln!("MLX_NATIVE chunked_mixed tokens={actual:?}");
    if actual != expected {
        failures.push("chunked mixed".into());
    }
    m.reset();
    m.prefill_begin(3, prompts[0].repeat(5)).unwrap();
    m.forward_mixed(&[], 31).unwrap();
    assert!(m.prefill_abort(3));
    assert_eq!(m.slots[3].length, 0);
    m.prefill(3, &prompts[0]).unwrap();
    assert_eq!(
        greedy(&m, 1),
        vec![expected[0][0]],
        "cancelled slot must reset PLE/GDN/QSA state"
    );
    let before = m.slots[3].length;
    assert!(
        m.execute(&[(3, 123, before as u32), (0, VOCAB as u32, 0)], &[0])
            .is_err()
    );
    assert_eq!(m.slots[3].length, before);
    assert!(!m.poisoned);
    let free = m.pool.free_blocks();
    let logits = unsafe { m.scratch.logits.read_f32(0, VOCAB) };
    for contracts in [vec![], vec![0], vec![MLX_CHUNK + 1], vec![1, 1]] {
        assert!(
            m.execute_contracts(&[(3, 123, before as u32)], &[0], Some(&contracts))
                .is_err()
        );
        assert_eq!(m.slots[3].length, before);
        assert_eq!(m.pool.free_blocks(), free);
        assert!(!m.poisoned);
        assert_eq!(unsafe { m.scratch.logits.read_f32(0, VOCAB) }, logits);
    }
    assert_eq!(allocated, m.device.allocated_bytes());
    assert!(
        failures.is_empty(),
        "same-checkpoint MLX generation failures: {failures:?}"
    );
}

#[test]
#[ignore = "111 GB mixed decode replay; external memory watchdog required"]
fn flash_next_mlx_mixed_decode_preserves_serial_logits() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    let ids = |case: usize, key: &str| {
        reference["cases"][case][key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let prompt = ids(0, "prompt_ids");
    let neighbour = ids(1, "prompt_ids");
    let tokens = ids(0, "token_ids");
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    let mut serial = vec![m.prefill(0, &prompt).unwrap()];
    for &token in &tokens[..tokens.len() - 1] {
        serial.push(m.forward(token).unwrap());
    }
    m.reset();
    assert_eq!(m.prefill(0, &prompt).unwrap(), serial[0]);
    m.prefill_begin(1, neighbour.clone()).unwrap();
    m.prefill_begin(2, prompt.clone()).unwrap();
    let mut mismatches = Vec::new();
    for step in 1..tokens.len() {
        // Admission, completion, cancellation, and reuse of other slots must
        // not alter this sequence's cached state or projection contraction.
        if [11, 23, 47].contains(&step) {
            m.prefill_abort(1);
            m.prefill_begin(1, neighbour.clone()).unwrap();
        }
        let (actual, _) = m
            .forward_mixed(&[(0, tokens[step - 1], m.slots[0].length as u32)], CHUNK)
            .unwrap();
        if actual != serial[step] {
            mismatches.push(step);
        }
        assert_eq!(m.device.allocated_bytes(), allocated);
    }
    eprintln!(
        "MLX_MIXED_REPLAY positions={} vocabulary={} unequal_steps={mismatches:?}",
        tokens.len(),
        VOCAB
    );
    assert!(
        mismatches.is_empty(),
        "mixed prefill changed serial decode logits"
    );
}

#[test]
fn whole_walk_status_survives_reused_layer_scratch() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let flags = d.upload(&u32::MAX.to_le_bytes()).unwrap();
    let bad = d.upload(&0u32.to_le_bytes()).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch("q4x_status", &[&flags, &bad], &[1, 512, 2], [1, 1, 1], 256);
    // The next layer overwrites its scratch with a healthy value. The
    // accumulated whole-walk status must still retain the earlier failure.
    cmd.dispatch(
        "nemo_state_copy",
        &[&flags],
        &[1, 0, u32::MAX],
        [1, 1, 1],
        256,
    );
    cmd.dispatch("q4x_status", &[&flags, &bad], &[1, 0, 4], [1, 1, 1], 256);
    cmd.finish().unwrap();
    assert_eq!(unsafe { bad.read_u32(1) }, [2]);
    assert_eq!(unsafe { flags.read_u32(1) }, [0]);
}

#[test]
#[ignore = "111 GB logical prefill replay; external memory watchdog required"]
fn flash_next_mlx_sliced_prefill_preserves_serial_logits() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_MLX_MARGINS").unwrap()).unwrap(),
    )
    .unwrap();
    let ids = |case: usize, key: &str| {
        reference["cases"][case][key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let prompt = ids(1, "prompt_ids");
    let neighbour = ids(0, "prompt_ids");
    let tokens = ids(1, "token_ids");
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    let mut serial = vec![m.prefill(0, &prompt).unwrap()];
    for &token in &tokens[..15] {
        serial.push(m.forward(token).unwrap());
    }
    m.reset();
    m.prefill_begin(0, prompt.clone()).unwrap();
    m.prefill_begin(1, neighbour.clone()).unwrap();
    let mut tick = 0;
    let first = loop {
        let budget = [1, 3, 8, 13, 32, 63, 125, 128][tick % 8];
        let (decodes, complete) = m.forward_mixed(&[], budget).unwrap();
        assert!(decodes.is_empty());
        assert_eq!(allocated, m.device.allocated_bytes());
        if tick == 3 {
            m.prefill_abort(1);
            m.prefill_begin(1, neighbour.clone()).unwrap();
        }
        if let Some((_, logits, used)) = complete.into_iter().find(|c| c.0 == 0) {
            assert_eq!(used, prompt.len());
            break logits;
        }
        tick += 1;
        assert!(tick < 128, "sliced prefill failed to make progress");
    };
    assert_eq!(
        first, serial[0],
        "sliced prompt changed full-vocabulary completion logits"
    );
    for (i, &token) in tokens[..15].iter().enumerate() {
        assert_eq!(
            m.forward(token).unwrap(),
            serial[i + 1],
            "sliced prompt changed recurrent/cache state at {i}"
        );
    }
    eprintln!(
        "MLX_SLICED_PREFILL prompt={} positions={} vocabulary={VOCAB} ticks={} exact=true",
        prompt.len(),
        serial.len(),
        tick + 1
    );
}

#[test]
fn checked_full_model_memory_bounds() {
    for (ctx, batch) in [
        (0, 1),
        (262145, 1),
        (1, 0),
        (1, 65),
        (usize::MAX, usize::MAX),
    ] {
        assert!(FlashNext::memory(ctx, batch).is_err());
        assert!(FlashNext::memory_rows(ctx, batch, MLX_CHUNK).is_err());
    }
    for rows in [0, super::super::affine::MAX_ROWS + 1, usize::MAX] {
        assert!(FlashNext::memory_rows(4096, 4, rows).is_err());
    }
    for (ctx, batch) in [(1, 1), (4096, 4), (262144, 64)] {
        let (c, s) = FlashNext::memory(ctx, batch).unwrap();
        eprintln!("Flash Next context={ctx} batch={batch} cache={c} scratch={s}");
        assert!(c > 0 && s > 0);
        let (mlx_cache, mlx_scratch) = FlashNext::memory_rows(ctx, batch, MLX_CHUNK).unwrap();
        assert_eq!(
            mlx_cache, c,
            "prefill capacity must not alter cache ownership"
        );
        assert!(mlx_scratch > s);
    }
}

#[test]
#[ignore = "read-only capacity diagnostic; requires the elected MLX checkpoint headers"]
fn mlx_agent_context_capacity_plan() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").expect("MLX model directory");
    let plan = super::super::mlx::FlashNextMlxPlan::inspect(Path::new(&path)).unwrap();
    for batch in [1, 4] {
        for context in [4096, 8192, 12288, 16384, 24576, 32768] {
            for rows in [MLX_CHUNK, 512, 256, CHUNK] {
                let (cache, scratch) = FlashNext::mlx_memory_rows(context, batch, rows).unwrap();
                let scratch = scratch + super::super::affine::workspace_bytes(rows) as u64;
                for entries in [2 * batch, batch, 0] {
                    let prefix_bytes = prefix::PrefixCache::bytes(context, entries);
                    let required = plan.resident_weight_bytes + cache + scratch + prefix_bytes;
                    eprintln!(
                        "FLASH_AGENT_PLAN context={context} batch={batch} rows={rows} entries={entries} weights={} cache={cache} prefix={prefix_bytes} scratch={scratch} required={required}",
                        plan.resident_weight_bytes
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "112 GB complete-state prefix replay; external memory watchdog required"]
fn flash_next_mlx_prefix_reuse_preserves_complete_state() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .ok()
        .map(|v| v.parse::<u64>().expect("positive budget bytes"));
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, budget).unwrap();
    eprintln!(
        "FLASH_PREFIX_CAPACITY rows={} allocated={}",
        m.chunk,
        m.device.allocated_bytes()
    );
    let allocated = m.device.allocated_bytes();
    let prompt = |branch: u32, n: usize| {
        (0..n)
            .map(|i| {
                if i % 257 == 0 {
                    248044
                } else {
                    100 + (i as u32 * 7 + branch * 113) % 16000
                }
            })
            .collect::<Vec<_>>()
    };
    // Cross-slot restores on both sides of logical and sparse-index boundaries.
    // Full-vocabulary F32 equality, plus teacher-forced decode, validates every
    // carried state rather than accepting merely the same first greedy token.
    for n in [m.chunk + 1, m.chunk + 17, 2307] {
        let tokens = prompt(1, n);
        m.reset();
        m.prefix.clear(&mut m.pool);
        // Disable capture as well as lookup for the cold reference, retaining
        // the same allocated capacity and arithmetic shape. This also tests
        // whether introducing the backup-page cut changes cold computation.
        let prefix = std::mem::take(&mut m.prefix);
        let start = std::time::Instant::now();
        let mut cold = vec![m.prefill(0, &tokens).unwrap()];
        let cold_ms = start.elapsed().as_secs_f64() * 1000.;
        assert_eq!(m.take_prefill_reused(0), 0);
        for t in [321, 248044, 753, 902] {
            cold.push(m.forward(t).unwrap());
        }
        m.prefix = prefix;
        m.reset();
        assert_eq!(
            m.prefill(0, &tokens).unwrap(),
            cold[0],
            "capture changed cold logits"
        );
        m.reset();
        let reused = prefix::cuts(tokens.len(), m.chunk)[1];
        let start = std::time::Instant::now();
        let warm = m.prefill(2, &tokens).unwrap();
        let warm_ms = start.elapsed().as_secs_f64() * 1000.;
        assert_eq!(m.take_prefill_reused(2), reused);
        assert_eq!(m.take_prefill_reused(2), 0);
        assert_eq!(warm, cold[0], "prefix changed complete logits at {n}");
        for (i, t) in [321, 248044, 753, 902].into_iter().enumerate() {
            let pos = m.slots[2].length as u32;
            assert_eq!(m.execute(&[(2, t, pos)], &[0]).unwrap(), cold[i + 1]);
        }
        if n == 2307 {
            let mut edited = tokens.clone();
            edited[reused - 1] = 1337;
            m.reset();
            m.prefix.clear(&mut m.pool);
            let expected = m.prefill(0, &edited).unwrap();
            m.reset();
            m.prefix.clear(&mut m.pool);
            m.prefill(0, &tokens).unwrap();
            m.reset();
            assert_eq!(m.prefill(3, &edited).unwrap(), expected);
            assert_eq!(
                m.take_prefill_reused(3),
                prefix::cuts(tokens.len(), m.chunk)[0]
            );
            // Re-establish the original branch for the cancellation gate.
            m.prefill(0, &tokens).unwrap();
        }
        // Cancellation must not lose validated snapshots, and scheduler slicing
        // a restored suffix must keep the cold prompt's original shape contract.
        m.prefill_begin(3, tokens.clone()).unwrap();
        assert_eq!(m.take_prefill_reused(3), reused);
        assert!(m.prefill_abort(3));
        m.prefill_begin(1, tokens.clone()).unwrap();
        assert_eq!(m.take_prefill_reused(1), reused);
        let mut ticks = 0;
        loop {
            let (_, completed) = m.forward_mixed(&[], [1, 13, 32, 63][ticks % 4]).unwrap();
            ticks += 1;
            assert!(ticks < 128);
            if let Some((_, logits, used)) = completed.into_iter().find(|v| v.0 == 1) {
                assert_eq!(used, n);
                assert_eq!(
                    logits, cold[0],
                    "sliced prefix suffix changed logits at {n}"
                );
                break;
            }
        }
        assert_eq!(m.device.allocated_bytes(), allocated);
        eprintln!(
            "FLASH_PREFIX_EXACT prompt={n} reused={reused} cold_ms={cold_ms:.3} warm_ms={warm_ms:.3} vocabulary={VOCAB} decode_steps=4"
        );
    }
    // Distinct conversations fill the pool; one advancing agent must not
    // displace the other waiting conversations' latest checkpoints.
    m.reset();
    m.prefix.clear(&mut m.pool);
    let mut prompts = (0..4).map(|i| prompt(i, m.chunk + 17)).collect::<Vec<_>>();
    let mut references = Vec::new();
    for (i, tokens) in prompts.iter().enumerate() {
        references.push(m.prefill(i, tokens).unwrap());
    }
    for _ in 0..3 {
        prompts[0].extend([33; 512]);
        m.prefill(0, &prompts[0]).unwrap();
    }
    m.reset();
    for i in 1..4 {
        assert_eq!(m.prefill(i, &prompts[i]).unwrap(), references[i]);
        assert!(m.take_prefill_reused(i) > 0, "fast branch evicted peer {i}");
    }
    assert_eq!(allocated, m.device.allocated_bytes());
    m.reset();
    m.prefix.clear(&mut m.pool);
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
    m.poisoned = true;
    assert!(
        m.prepare(0, &prompts[0]).is_err(),
        "poison is not a cache miss"
    );
}

#[test]
#[ignore = "112 GB natural-generation cache gate; elected model, prefix fixtures and watchdog required"]
fn flash_next_mlx_prefix_natural_generations_and_arrivals() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let prompts = fixtures["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            v["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(prompts.len(), 4);
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .ok()
        .map(|v| v.parse::<u64>().expect("positive budget bytes"));
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, budget).unwrap();
    eprintln!(
        "FLASH_PREFIX_CAPACITY rows={} allocated={}",
        m.chunk,
        m.device.allocated_bytes()
    );
    let allocated = m.device.allocated_bytes();
    let argmax = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
            .unwrap()
            .0 as u32
    };
    let ended = |t: u32| [248044, 248046].contains(&t);
    let mut reference = Vec::new();
    for prompt in &prompts {
        m.reset();
        m.prefix.clear(&mut m.pool);
        let prefix = std::mem::take(&mut m.prefix);
        let mut logits = m.prefill(0, prompt).unwrap();
        assert_eq!(
            m.take_prefill_reused(0),
            0,
            "fixture prefixes must be distinct"
        );
        let mut tokens = Vec::new();
        for _ in 0..64 {
            let t = argmax(&logits);
            tokens.push(t);
            if ended(t) {
                break;
            }
            logits = m.forward(t).unwrap();
        }
        assert!(
            ended(*tokens.last().unwrap()),
            "a capped generation is not a natural-EOS gate"
        );
        reference.push(tokens);
        m.prefix = prefix;
    }
    for warm in [false, true] {
        m.reset();
        m.prefix.clear(&mut m.pool);
        if warm {
            for (slot, prompt) in prompts.iter().enumerate() {
                let logits = m.prefill(slot, prompt).unwrap();
                assert_eq!(
                    argmax(&logits),
                    reference[slot][0],
                    "capture changed first token"
                );
            }
        }
        m.reset();
        let mut actual = vec![Vec::new(); 4];
        let mut ready = [false; 4];
        // The fourth client arrives late, after the first cohort starts processing.
        for (slot, prompt) in prompts.iter().enumerate().take(3) {
            m.prefill_begin(slot, prompt.clone()).unwrap();
            assert_eq!(
                m.take_prefill_reused(slot),
                if warm {
                    m.prompt_plan(prompt).cuts()[1]
                } else {
                    0
                }
            );
        }
        for tick in 0..512 {
            if tick == 2 {
                m.prefill_begin(3, prompts[3].clone()).unwrap();
                assert_eq!(m.take_prefill_reused(3) > 0, warm);
                assert!(m.prefill_abort(3));
                m.prefill_begin(3, prompts[3].clone()).unwrap();
                assert_eq!(m.take_prefill_reused(3) > 0, warm);
            }
            let decodes = (0..4)
                .filter_map(|slot| {
                    let &token = actual[slot].last()?;
                    (ready[slot] && !ended(token)).then_some((
                        slot,
                        token,
                        m.slots[slot].length as u32,
                    ))
                })
                .collect::<Vec<_>>();
            let budget = if warm {
                [13, 128, 512]
            } else {
                [m.capacity, 511, 33]
            };
            let (logits, complete) = m.forward_mixed(&decodes, budget[tick % 3]).unwrap();
            for (i, &(slot, _, _)) in decodes.iter().enumerate() {
                actual[slot].push(argmax(&logits[i * VOCAB..(i + 1) * VOCAB]));
            }
            for (slot, logits, rows) in complete {
                assert_eq!(rows, prompts[slot].len());
                actual[slot].push(argmax(&logits));
                ready[slot] = true;
            }
            assert!(actual.iter().all(|v| v.len() <= 64));
            assert_eq!(allocated, m.device.allocated_bytes());
            if actual.iter().all(|v| v.last().is_some_and(|&t| ended(t))) {
                break;
            }
        }
        assert_eq!(
            actual, reference,
            "restores/late arrivals changed complete generations"
        );
        eprintln!(
            "FLASH_PREFIX_NATURAL warm={warm} exact=4/4 natural_eos=true late_arrival=true cancellation=true"
        );
    }
}

#[test]
#[ignore = "full checkpoint and memory watchdog; exact full logits and alternating execution cost, not serving throughput"]
fn flash_next_mlx_coalesced_packed_execution() {
    use super::super::{affine, deltanet, moe};
    let controls = |route| {
        affine::SEPARATE_SPANS_FOR_TEST.with(|v| v.set(route == 0));
        deltanet::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(route < 2));
        moe::BASELINE_EXPERT_TAIL_FOR_TEST.with(|v| v.set(route < 3));
        affine::INLINE_INPUT_FOR_TEST.with(|v| v.set(route < 4));
    };
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::SEPARATE_SPANS_FOR_TEST.with(|v| v.set(false));
            deltanet::BASELINE_RECURRENT_FOR_TEST.with(|v| v.set(false));
            moe::BASELINE_EXPERT_TAIL_FOR_TEST.with(|v| v.set(false));
            affine::INLINE_INPUT_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    assert_eq!(m.chunk, 1024);
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        let mut expected = None;
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for round in 0..3 {
            for i in 0..5 {
                let route = (i + round) % 5;
                controls(route);
                m.reset();
                m.prefix.clear(&mut m.pool);
                let mut gpu = 0.;
                let mut logits = Vec::new();
                let step = 1024 / slots;
                for offset in (0..2048).step_by(step) {
                    let rows = (0..slots)
                        .flat_map(|slot| {
                            (offset..offset + step)
                                .map(move |p| (slot, 1000 + p as u32 + slot as u32 * 17, p as u32))
                        })
                        .collect::<Vec<_>>();
                    let output = if offset + step == 2048 {
                        (0..slots).map(|s| (s + 1) * step - 1).collect::<Vec<_>>()
                    } else {
                        vec![]
                    };
                    let result = m
                        .execute_contracts(&rows, &output, Some(&vec![1024; slots]))
                        .unwrap();
                    gpu += m.last_gpu_seconds;
                    logits.extend(result.into_iter().map(f32::to_bits));
                }
                times[route].push(gpu);
                for pos in 2048..2051 {
                    let rows = (0..slots)
                        .map(|s| (s, 23 + s as u32, pos))
                        .collect::<Vec<_>>();
                    logits.extend(
                        m.execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                }
                if let Some(reference) = &expected {
                    assert!(
                        &logits == reference,
                        "changed full-vocabulary prefill/continuation: c={slots} round={round} route={route}"
                    );
                } else {
                    expected = Some(logits);
                }
                assert_eq!(allocated, m.device.allocated_bytes());
                eprintln!(
                    "FLASH_COALESCED_PACKED_SAMPLE c={slots} round={round} route={route} gpu_s={gpu}"
                );
            }
        }
        eprintln!(
            "FLASH_COALESCED_PACKED {}",
            serde_json::json!({"slots":slots,"gpu_seconds":times,"full_logits_exact":true})
        );
    }
}

#[test]
#[ignore = "full checkpoint and memory watchdog; physical 1024/2048 rows, unchanged logical arithmetic, full logits and cost"]
fn flash_next_mlx_wide_batch_execution_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            mlx_load::WIDE_BATCH_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    mlx_load::WIDE_BATCH_FOR_TEST.with(|v| v.set(true));
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let prompts = fixtures["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(s, p)| {
            p["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .cycle()
                .take(4096 + s * 17)
                .map(|t| t.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut m = FlashNext::load(Path::new(&path), 8192, 4, Some(budget)).unwrap();
    assert_eq!((m.capacity, m.chunk), (2048, 1024));
    let allocated = m.device.allocated_bytes();
    let _prefix = std::mem::take(&mut m.prefix);
    m.prefill(0, &prompts[0]).unwrap();
    let mut reference = None;
    for round in 0..3 {
        for index in 0..2 {
            let capacity = if (round + index) % 2 == 0 { 1024 } else { 2048 };
            m.capacity = capacity;
            m.reset();
            for (slot, prompt) in prompts.iter().enumerate() {
                m.prefill_begin(slot, prompt.clone()).unwrap();
            }
            let started = std::time::Instant::now();
            let mut gpu = 0.;
            let mut ticks = 0;
            let mut first = [0.; 4];
            let mut last = [0.; 4];
            let mut gaps = Vec::new();
            let mut steps = [0; 4];
            let mut logits = vec![Vec::new(); 4];
            for _ in 0..512 {
                let decodes = (0..4)
                    .filter(|&s| (1..=16).contains(&steps[s]))
                    .map(|s| (s, 100 + s as u32 * 17 + steps[s], m.slots[s].length as u32))
                    .collect::<Vec<_>>();
                let (out, complete) = m.forward_mixed(&decodes, capacity).unwrap();
                gpu += m.last_gpu_seconds;
                ticks += 1;
                let elapsed = started.elapsed().as_secs_f64();
                for (i, &(s, _, _)) in decodes.iter().enumerate() {
                    logits[s].extend(out[i * VOCAB..(i + 1) * VOCAB].iter().map(|v| v.to_bits()));
                    gaps.push(elapsed - last[s]);
                    last[s] = elapsed;
                    steps[s] += 1;
                }
                for (s, out, _) in complete {
                    logits[s].extend(out.into_iter().map(f32::to_bits));
                    first[s] = elapsed;
                    last[s] = elapsed;
                    steps[s] = 1;
                }
                if steps.iter().all(|&v| v == 17) {
                    break;
                }
            }
            assert_eq!(steps, [17; 4]);
            if let Some(expected) = &reference {
                assert!(
                    &logits == expected,
                    "wide physical pass changed full logits"
                );
            } else {
                reference = Some(logits);
            }
            assert_eq!(allocated, m.device.allocated_bytes());
            gaps.sort_by(f64::total_cmp);
            eprintln!(
                "FLASH_WIDE_PHYSICAL {}",
                serde_json::json!({"round":round,"capacity":capacity,
                "gpu_seconds":gpu,"seconds":started.elapsed().as_secs_f64(),"first_seconds":first,
                "max_gap_s":gaps.last(),"ticks":ticks,"allocated_bytes":allocated,"full_logits_exact":true})
            );
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; natural complete generations, restores, late arrivals and cancellation at 2048 physical rows"]
fn flash_next_mlx_wide_batch_generation_and_cache_parity() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            mlx_load::WIDE_BATCH_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    mlx_load::WIDE_BATCH_FOR_TEST.with(|v| v.set(true));
    flash_next_mlx_prefix_natural_generations_and_arrivals();
    flash_next_mlx_prefix_reuse_preserves_complete_state();
}

#[test]
#[ignore = "full checkpoint and watchdog; on-chip decode attention cost and exact full-vocabulary continuations"]
fn flash_next_mlx_local_attention_execution_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            qsa::LOCAL_ATTENTION_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let prompts = fixtures["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut m = FlashNext::load(Path::new(&path), 8192, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    for batch in [1, 4] {
        let mut reference: Option<Vec<Vec<u32>>> = None;
        for round in 0..3 {
            for turn in 0..2 {
                let local = (round + turn) % 2 == 1;
                qsa::LOCAL_ATTENTION_FOR_TEST.with(|v| v.set(local));
                m.reset();
                let mut logits = Vec::new();
                for (slot, prompt) in prompts.iter().enumerate().take(batch) {
                    logits.push(
                        m.prefill(slot, prompt)
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>(),
                    );
                }
                let started = std::time::Instant::now();
                let mut gpu = Vec::new();
                for step in 0..32 {
                    let rows = (0..batch)
                        .map(|slot| {
                            (
                                slot,
                                100 + slot as u32 * 17 + step,
                                m.slots[slot].length as u32,
                            )
                        })
                        .collect::<Vec<_>>();
                    let out = m.execute(&rows, &(0..batch).collect::<Vec<_>>()).unwrap();
                    gpu.push(m.last_gpu_seconds);
                    logits.push(out.into_iter().map(f32::to_bits).collect::<Vec<_>>());
                }
                let wall = started.elapsed().as_secs_f64();
                if let Some(expected) = &reference {
                    assert!(
                        expected == &logits,
                        "local attention changed full logits: c={batch} local={local}"
                    );
                } else {
                    reference = Some(logits);
                }
                assert_eq!(m.device.allocated_bytes(), allocated);
                eprintln!(
                    "FLASH_LOCAL_ATTENTION {}",
                    serde_json::json!({"batch":batch,"round":round,"local":local,
                    "wall_seconds":wall,"gpu_seconds":gpu,"allocated_bytes":allocated,"full_logits_exact":true})
                );
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint, prefix fixtures and watchdog; fixed-work admission cost with exact vocabulary checks"]
fn flash_next_mlx_admission_execution_cost() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            serving::POLICY_FOR_TEST.with(|v| v.set(0));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let prompts = fixtures["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(slot, p)| {
            p["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .cycle()
                .take(4096 + slot * 17)
                .map(|t| t.as_u64().unwrap() as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(prompts.len(), 4);
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 8192, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    m.prefix.clear(&mut m.pool);
    let _prefix = std::mem::take(&mut m.prefix);
    let mut reference = None;
    for round in 0..2 {
        for index in 0..3 {
            let policy = (index + round) % 3;
            serving::POLICY_FOR_TEST.with(|v| v.set(policy));
            m.reset();
            for (slot, prompt) in prompts.iter().enumerate() {
                m.prefill_begin(slot, prompt.clone()).unwrap();
            }
            let started = std::time::Instant::now();
            let mut gpu = 0.;
            let mut first = [0.; 4];
            let mut last = [0.; 4];
            let mut gaps = Vec::new();
            let mut logits = vec![Vec::new(); 4];
            let mut steps = [0; 4];
            for _ in 0..512 {
                let decodes = (0..4)
                    .filter(|&s| (1..=16).contains(&steps[s]))
                    .map(|s| (s, 100 + s as u32 * 17 + steps[s], m.slots[s].length as u32))
                    .collect::<Vec<_>>();
                let (out, complete) = m.forward_mixed(&decodes, m.chunk).unwrap();
                gpu += m.last_gpu_seconds;
                let elapsed = started.elapsed().as_secs_f64();
                for (i, &(s, _, _)) in decodes.iter().enumerate() {
                    logits[s].extend(out[i * VOCAB..(i + 1) * VOCAB].iter().map(|v| v.to_bits()));
                    gaps.push(elapsed - last[s]);
                    last[s] = elapsed;
                    steps[s] += 1;
                }
                for (s, out, _) in complete {
                    logits[s].extend(out.into_iter().map(f32::to_bits));
                    first[s] = elapsed;
                    last[s] = elapsed;
                    steps[s] = 1;
                }
                if steps.iter().all(|&n| n == 17) {
                    break;
                }
            }
            assert_eq!(steps, [17; 4]);
            if let Some(reference) = &reference {
                assert!(
                    &logits == reference,
                    "admission changed full-vocabulary state: policy={policy} round={round}"
                );
            } else {
                reference = Some(logits);
            }
            assert_eq!(allocated, m.device.allocated_bytes());
            gaps.sort_by(f64::total_cmp);
            eprintln!(
                "FLASH_ADMISSION {}",
                serde_json::json!({"round":round,"policy":policy,
                "seconds":started.elapsed().as_secs_f64(),"gpu_seconds":gpu,"first_seconds":first,
                "max_gap_s":gaps.last(),"p99_gap_s":gaps[gaps.len()*99/100],"full_logits_exact":true})
            );
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; fixed ragged projection cost and exact full-vocabulary continuations"]
fn flash_next_mlx_compatible_spans_execution_cost() {
    use super::super::affine;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::SHAPE_ONLY_SPANS_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 2048, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    let mut reference = None;
    for round in 0..3 {
        for i in 0..2 {
            let route = (i + round) % 2;
            affine::SHAPE_ONLY_SPANS_FOR_TEST.with(|v| v.set(route == 0));
            m.reset();
            m.prefix.clear(&mut m.pool);
            let started = std::time::Instant::now();
            let mut gpu = 0.;
            let mut logits = Vec::new();
            for offset in (0..512).step_by(32) {
                let rows = (0..4)
                    .flat_map(|s| {
                        (offset..offset + 32)
                            .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                    })
                    .collect::<Vec<_>>();
                let outputs = if offset == 480 {
                    vec![31, 63, 95, 127]
                } else {
                    vec![]
                };
                logits.extend(
                    m.execute_contracts(&rows, &outputs, Some(&[700, 750, 800, 1024]))
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits),
                );
                gpu += m.last_gpu_seconds;
            }
            for pos in 512..516 {
                let rows = (0..4).map(|s| (s, 23 + s as u32, pos)).collect::<Vec<_>>();
                logits.extend(
                    m.execute_contracts(&rows, &[0, 1, 2, 3], Some(&[1; 4]))
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits),
                );
                gpu += m.last_gpu_seconds;
            }
            if let Some(reference) = &reference {
                assert!(
                    &logits == reference,
                    "compatible span merge changed logits: round={round} route={route}"
                );
            } else {
                reference = Some(logits);
            }
            assert_eq!(allocated, m.device.allocated_bytes());
            eprintln!(
                "FLASH_COMPATIBLE_SPANS {}",
                serde_json::json!({"round":round,"route":route,
                "gpu_seconds":gpu,"wall_seconds":started.elapsed().as_secs_f64(),"full_logits_exact":true})
            );
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; fused expert gate/up complete vocabulary and GPU execution A/B"]
fn flash_next_mlx_fused_gate_up_execution_cost() {
    use super::super::moe;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            moe::FUSED_GATE_UP_FOR_TEST.with(|v| v.set(0));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        let mut reference = None;
        for round in 0..3 {
            for index in 0..3 {
                let fused = (round + index) % 3;
                moe::FUSED_GATE_UP_FOR_TEST.with(|v| v.set(fused));
                m.reset();
                m.prefix.clear(&mut m.pool);
                let started = std::time::Instant::now();
                let mut gpu = 0.;
                let mut logits = Vec::new();
                let step = 1024 / slots;
                for offset in (0..2048).step_by(step) {
                    let rows = (0..slots)
                        .flat_map(|s| {
                            (offset..offset + step)
                                .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                        })
                        .collect::<Vec<_>>();
                    let outputs = if offset + step == 2048 {
                        (0..slots).map(|s| (s + 1) * step - 1).collect::<Vec<_>>()
                    } else {
                        vec![]
                    };
                    logits.extend(
                        m.execute_contracts(&rows, &outputs, Some(&vec![1024; slots]))
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    gpu += m.last_gpu_seconds;
                }
                for pos in 2048..2052 {
                    let rows = (0..slots)
                        .map(|s| (s, 23 + s as u32, pos))
                        .collect::<Vec<_>>();
                    logits.extend(
                        m.execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    gpu += m.last_gpu_seconds;
                }
                if let Some(expected) = &reference {
                    assert!(
                        &logits == expected,
                        "fused={fused} slots={slots} round={round}"
                    );
                } else {
                    reference = Some(logits);
                }
                assert_eq!(allocated, m.device.allocated_bytes());
                eprintln!(
                    "FLASH_FUSED_GATE_UP {}",
                    serde_json::json!({"slots":slots,"round":round,"fused":fused,
                    "gpu_seconds":gpu,"wall_seconds":started.elapsed().as_secs_f64(),"full_logits_exact":true})
                );
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint and memory watchdog; alternating whole-model loader costs and exact full logits"]
fn flash_next_mlx_projection_loader_execution_cost() {
    use super::super::{affine, moe};
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            moe::EXPERT_LOADER_FOR_TEST.with(|v| v.set(2));
            affine::PLAIN_DENSE_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        let mut reference = None;
        for round in 0..3 {
            for index in 0..3 {
                let route = (index + round) % 3;
                moe::EXPERT_LOADER_FOR_TEST.with(|v| v.set(if route == 0 { 0 } else { 2 }));
                affine::PLAIN_DENSE_FOR_TEST.with(|v| v.set(route < 2));
                m.reset();
                m.prefix.clear(&mut m.pool);
                let started = std::time::Instant::now();
                let mut gpu = 0.;
                let mut logits = Vec::new();
                let step = 1024 / slots;
                for offset in (0..2048).step_by(step) {
                    let rows = (0..slots)
                        .flat_map(|s| {
                            (offset..offset + step)
                                .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                        })
                        .collect::<Vec<_>>();
                    let outputs = if offset + step == 2048 {
                        (0..slots).map(|s| (s + 1) * step - 1).collect::<Vec<_>>()
                    } else {
                        vec![]
                    };
                    logits.extend(
                        m.execute_contracts(&rows, &outputs, Some(&vec![1024; slots]))
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    gpu += m.last_gpu_seconds;
                }
                for pos in 2048..2052 {
                    let rows = (0..slots)
                        .map(|s| (s, 23 + s as u32, pos))
                        .collect::<Vec<_>>();
                    logits.extend(
                        m.execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    gpu += m.last_gpu_seconds;
                }
                if let Some(expected) = &reference {
                    assert!(
                        &logits == expected,
                        "projection loader changed logits: c={slots} round={round} route={route}"
                    );
                } else {
                    reference = Some(logits);
                }
                assert_eq!(allocated, m.device.allocated_bytes());
                eprintln!(
                    "FLASH_PROJECTION_LOADER {}",
                    serde_json::json!({"slots":slots,
                    "round":round, "route":route, "gpu_seconds":gpu,
                    "wall_seconds":started.elapsed().as_secs_f64(), "full_logits_exact":true})
                );
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint and memory watchdog; staged projections/padded attention full-logit parity and rotated costs"]
fn flash_next_mlx_prefill_optimization_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::Staging, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; gathered attention full-logit parity and rotated costs"]
fn flash_next_mlx_attention_gather_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::AttentionGather, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; long prefill gathered attention exact logits and rotated costs"]
fn flash_next_mlx_attention_gather_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::AttentionGather, true);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; padded projections exact long-prefill logits and rotated costs"]
fn flash_next_mlx_projection_padding_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::ProjectionPadding, true);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; weight reuse exact full logits and rotated costs"]
fn flash_next_mlx_projection_reuse_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::ProjectionReuse, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; long weight reuse exact full logits and rotated costs"]
fn flash_next_mlx_projection_reuse_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::ProjectionReuse, true);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; attention value pitch exact full logits and costs"]
fn flash_next_mlx_attention_value_pitch_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::AttentionValuePitch, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; long attention value pitch full logits and costs"]
fn flash_next_mlx_attention_value_pitch_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::AttentionValuePitch, true);
}

#[derive(Clone, Copy, Debug)]
enum PrefillOptimization {
    Staging,
    AttentionGather,
    ProjectionPadding,
    ProjectionReuse,
    AttentionValuePitch,
    DirectExpert,
    ExpertRows64,
    RouterSplit,
    CacheOnlyTail,
    SharedInput,
    JoinedInput,
    VectorGateUp,
    PairedGateUp,
    DecodeFusion,
    CombineNorm,
    DirectAttention,
}

#[test]
#[ignore = "full checkpoint and watchdog; direct contiguous attention exact vocabularies and rotated costs"]
fn flash_next_mlx_direct_attention_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::DirectAttention, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; fused residual/normalization exact vocabularies and rotated costs"]
fn flash_next_mlx_combine_norm_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::CombineNorm, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; singleton expert fusion exact vocabularies and rotated costs"]
fn flash_next_mlx_vector_gate_up_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::VectorGateUp, false);
}

#[test]
#[ignore = "checkpoint-backed paired expert reuse, unchanged incoming optimizations and exact full vocabulary"]
fn flash_next_mlx_paired_gate_up_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::PairedGateUp, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; decode fusion exact vocabularies and rotated costs"]
fn flash_next_mlx_decode_fusion_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::DecodeFusion, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; cache-only terminal layer exact vocabularies and rotated costs"]
fn flash_next_mlx_cache_only_tail_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::CacheOnlyTail, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; long cache-only terminal layer exact vocabularies and rotated costs"]
fn flash_next_mlx_cache_only_tail_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::CacheOnlyTail, true);
}

#[test]
#[ignore = "full checkpoint and watchdog; shared DeltaNet preparation exact vocabularies and rotated costs"]
fn flash_next_mlx_shared_input_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::SharedInput, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; long shared DeltaNet preparation exact vocabularies and rotated costs"]
fn flash_next_mlx_shared_input_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::SharedInput, true);
}

#[test]
#[ignore = "full checkpoint and watchdog; joined DeltaNet projection grids with exact vocabularies and rotated costs"]
fn flash_next_mlx_joined_input_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::JoinedInput, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; long joined projection grids with exact vocabularies and rotated costs"]
fn flash_next_mlx_joined_input_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::JoinedInput, true);
}

#[test]
#[ignore = "full checkpoint and watchdog; bounded PLE read-ahead under the real mixed prefill scheduler"]
fn flash_next_mlx_lookahead_execution_cost() {
    lookahead_execution_cost(false);
}

#[test]
#[ignore = "full checkpoint and watchdog; long bounded PLE read-ahead under the real mixed prefill scheduler"]
fn flash_next_mlx_lookahead_long_execution_cost() {
    lookahead_execution_cost(true);
}

fn lookahead_execution_cost(long: bool) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            forward::PLE_LOOKAHEAD_FOR_TEST.with(|v| v.set(false));
            forward::ADAPTIVE_PLE_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let prompt_tokens = if long { 8705 } else { 2561 };
    let mut m = FlashNext::load(
        Path::new(&path),
        if long { 12288 } else { 4096 },
        4,
        Some(budget),
    )
    .unwrap();
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        let mut reference = None;
        for round in 0..if long { 3 } else { 4 } {
            for route in (0..2).map(|i| (i + round) % 2) {
                forward::ADAPTIVE_PLE_FOR_TEST.with(|v| v.set(route == 1));
                m.reset();
                m.prefix.clear(&mut m.pool);
                for slot in 0..slots {
                    m.prefill_begin(
                        slot,
                        (0..prompt_tokens)
                            .map(|p| 1000 + p + slot as u32 * 17)
                            .collect(),
                    )
                    .unwrap();
                }
                let started = std::time::Instant::now();
                let mut gpu = 0.;
                let mut done = 0;
                let mut logits = vec![Vec::new(); slots];
                while done < slots {
                    let (_, complete) = m.forward_mixed(&[], m.capacity).unwrap();
                    gpu += m.last_gpu_seconds;
                    for (slot, out, _) in complete {
                        logits[slot].extend(out.into_iter().map(f32::to_bits));
                        done += 1;
                    }
                }
                let prefill_wall = started.elapsed().as_secs_f64();
                let prefill_gpu = gpu;
                for position in prompt_tokens..prompt_tokens + 16 {
                    let rows = (0..slots)
                        .map(|s| (s, 23 + s as u32, position))
                        .collect::<Vec<_>>();
                    let out = m.execute(&rows, &(0..slots).collect::<Vec<_>>()).unwrap();
                    for (s, row) in out.chunks_exact(VOCAB).enumerate() {
                        logits[s].extend(row.iter().map(|v| v.to_bits()));
                    }
                }
                if let Some(expected) = &reference {
                    assert!(
                        &logits == expected,
                        "lookahead changes vocabulary c={slots} round={round} route={route}"
                    );
                } else {
                    reference = Some(logits);
                }
                assert_eq!(allocated, m.device.allocated_bytes());
                eprintln!(
                    "FLASH_LOOKAHEAD {}",
                    serde_json::json!({"slots":slots,"round":round,"route":route,"prompt_tokens":prompt_tokens,"adaptive":true,
                    "prefill_wall_seconds":prefill_wall,"prefill_gpu_seconds":prefill_gpu,"full_logits_exact":true,"allocated":allocated})
                );
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint and memory watchdog; staged affine-8 router full logits and costs"]
fn flash_next_mlx_router_split_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::RouterSplit, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; long staged affine-8 router full logits and costs"]
fn flash_next_mlx_router_split_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::RouterSplit, true);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; 64-row expert exact full logits and costs"]
fn flash_next_mlx_expert_rows64_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::ExpertRows64, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; long 64-row expert exact full logits and costs"]
fn flash_next_mlx_expert_rows64_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::ExpertRows64, true);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; sorted expert input exact full logits and costs"]
fn flash_next_mlx_direct_expert_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::DirectExpert, false);
}

#[test]
#[ignore = "full checkpoint and memory watchdog; long sorted expert input exact full logits and costs"]
fn flash_next_mlx_direct_expert_long_execution_cost() {
    prefill_optimization_execution_cost(PrefillOptimization::DirectExpert, true);
}

fn prefill_optimization_execution_cost(comparison: PrefillOptimization, long: bool) {
    use super::super::{affine, moe, qsa};
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::PLAIN_SPLIT_FOR_TEST.with(|v| v.set(false));
            qsa::PADDED_ATTENTION_FOR_TEST.with(|v| v.set(true));
            qsa::GATHER_ATTENTION_FOR_TEST.with(|v| v.set(true));
            qsa::VALUE_PITCH_FOR_TEST.with(|v| v.set(true));
            qsa::DIRECT_RUNS_FOR_TEST.with(|v| v.set(true));
            affine::PADDED_TILES_FOR_TEST.with(|v| v.set(true));
            affine::TILE_REUSE_FOR_TEST.with(|v| v.set(true));
            moe::DIRECT_EXPERT_FOR_TEST.with(|v| v.set(true));
            moe::EXPERT_ROWS64_FOR_TEST.with(|v| v.set(false));
            super::super::residual::HC_COMBINE_NORM_FOR_TEST.with(|v| v.set(true));
            affine::STAGED_ROUTER_FOR_TEST.with(|v| v.set(true));
            forward::CACHE_ONLY_TAIL_FOR_TEST.with(|v| v.set(true));
            affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(false));
            affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(true));
            affine::JOINED_SLAB_FOR_TEST.with(|v| v.set(false));
            moe::VECTOR_GATE_UP_FOR_TEST.with(|v| v.set(true));
            moe::PAIRED_GATE_UP_FOR_TEST.with(|v| v.set(true));
            super::super::residual::HC_VECTOR_FOR_TEST.with(|v| v.set(true));
            super::super::residual::HC_UP_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let context = if long { 12288 } else { 4096 };
    let full_tokens = if long { 8192 } else { 2048 };
    let mut m = FlashNext::load(Path::new(&path), context, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        let mut reference = None;
        let gather = matches!(comparison, PrefillOptimization::AttentionGather);
        let projection = matches!(comparison, PrefillOptimization::ProjectionPadding);
        let reuse = matches!(comparison, PrefillOptimization::ProjectionReuse);
        let pitch = matches!(comparison, PrefillOptimization::AttentionValuePitch);
        let wide = matches!(comparison, PrefillOptimization::ExpertRows64);
        let router = matches!(comparison, PrefillOptimization::RouterSplit);
        let cache_only = matches!(comparison, PrefillOptimization::CacheOnlyTail);
        let shared = matches!(comparison, PrefillOptimization::SharedInput);
        let joined = matches!(comparison, PrefillOptimization::JoinedInput);
        let fusion = matches!(comparison, PrefillOptimization::DecodeFusion);
        let combine_norm = matches!(comparison, PrefillOptimization::CombineNorm);
        let direct_attention = matches!(comparison, PrefillOptimization::DirectAttention);
        let paired = matches!(comparison, PrefillOptimization::PairedGateUp);
        let vector = fusion || matches!(comparison, PrefillOptimization::VectorGateUp);
        let direct = paired
            || direct_attention
            || combine_norm
            || vector
            || joined
            || shared
            || cache_only
            || router
            || wide
            || matches!(comparison, PrefillOptimization::DirectExpert);
        let routes = if gather || projection || reuse || pitch || direct {
            2
        } else {
            3
        };
        for round in 0..if (joined || paired) && !long { 9 } else { 3 } {
            for index in 0..routes {
                let route = (index + round) % routes;
                qsa::DIRECT_RUNS_FOR_TEST
                    .with(|v| v.set(paired || (direct_attention && route == 1)));
                super::super::residual::HC_COMBINE_NORM_FOR_TEST
                    .with(|v| v.set(paired || (combine_norm && route == 1)));
                forward::CACHE_ONLY_TAIL_FOR_TEST.with(|v| v.set(!cache_only || route == 1));
                affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(shared && route == 1));
                affine::JOINED_INPUT_FOR_TEST.with(|v| {
                    v.set(
                        paired
                            || direct_attention
                            || combine_norm
                            || vector
                            || (joined && route == 1),
                    )
                });
                moe::VECTOR_GATE_UP_FOR_TEST.with(|v| v.set(!vector || route == 1));
                moe::PAIRED_GATE_UP_FOR_TEST.with(|v| v.set(paired && route == 1));
                super::super::residual::HC_VECTOR_FOR_TEST.with(|v| v.set(!fusion || route == 1));
                super::super::residual::HC_UP_FOR_TEST.with(|v| v.set(!fusion || route == 1));
                affine::STAGED_ROUTER_FOR_TEST.with(|v| v.set(!router || route == 1));
                moe::EXPERT_ROWS64_FOR_TEST.with(|v| v.set(wide && route == 1));
                moe::DIRECT_EXPERT_FOR_TEST.with(|v| {
                    v.set(
                        paired
                            || direct_attention
                            || combine_norm
                            || vector
                            || joined
                            || shared
                            || cache_only
                            || router
                            || wide
                            || (direct && route == 1),
                    )
                });
                qsa::VALUE_PITCH_FOR_TEST.with(|v| v.set(direct || (pitch && route == 1)));
                affine::TILE_REUSE_FOR_TEST
                    .with(|v| v.set(direct || pitch || (reuse && route == 1)));
                affine::PLAIN_SPLIT_FOR_TEST.with(|v| {
                    v.set(!direct && !gather && !projection && !reuse && !pitch && route == 0)
                });
                affine::PADDED_TILES_FOR_TEST
                    .with(|v| v.set(direct || pitch || reuse || (projection && route == 1)));
                qsa::PADDED_ATTENTION_FOR_TEST.with(|v| {
                    v.set(direct || gather || projection || reuse || pitch || route == 2)
                });
                qsa::GATHER_ATTENTION_FOR_TEST.with(|v| {
                    v.set(direct || pitch || reuse || projection || (gather && route == 1))
                });
                m.reset();
                m.prefix.clear(&mut m.pool);
                let started = std::time::Instant::now();
                let mut gpu = 0.;
                let mut logits = Vec::new();
                // First full logical chunks, then a split-K 512-row tail.
                // Four slots share the full 2048-row physical arena.
                let step = if slots == 1 { 1024 } else { 512 };
                for (offset, count, logical) in (0..full_tokens)
                    .step_by(step)
                    .map(|offset| (offset, step, 1024))
                    .chain([(full_tokens, 512, 512)])
                {
                    let rows = (0..slots)
                        .flat_map(|s| {
                            (offset..offset + count)
                                .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                        })
                        .collect::<Vec<_>>();
                    let outputs = if cache_only && offset < full_tokens {
                        Vec::new()
                    } else {
                        (0..slots).map(|s| (s + 1) * count - 1).collect::<Vec<_>>()
                    };
                    logits.extend(
                        m.execute_contracts(&rows, &outputs, Some(&vec![logical; slots]))
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    gpu += m.last_gpu_seconds;
                }
                let prefill_gpu = gpu;
                for pos in full_tokens as u32 + 512..full_tokens as u32 + 544 {
                    let rows = (0..slots)
                        .map(|s| (s, 23 + s as u32, pos))
                        .collect::<Vec<_>>();
                    logits.extend(
                        m.execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    gpu += m.last_gpu_seconds;
                }
                if let Some(expected) = &reference {
                    assert!(
                        &logits == expected,
                        "{comparison:?} changed full logits c={slots} round={round} route={route}"
                    );
                } else {
                    reference = Some(logits);
                }
                assert_eq!(allocated, m.device.allocated_bytes());
                eprintln!(
                    "FLASH_SPLIT_MODEL {}",
                    serde_json::json!({"slots":slots,
                    "round":round,"route":route,"prefill_gpu_seconds":prefill_gpu,
                    "gather_comparison":gather,"prompt_tokens":full_tokens+512,
                    "comparison":format!("{comparison:?}"),
                    "warmup":(joined || paired) && !long && round < 2,
                    "gpu_seconds":gpu,"wall_seconds":started.elapsed().as_secs_f64(),
                    "full_logits_exact":true,"allocated":allocated})
                );
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; matched-token c=1 execution versus the installed oMLX adapter, not serving"]
fn flash_next_mlx_fixed_work_cost() {
    fixed_work_cost(0);
}

#[test]
#[ignore = "full checkpoint and watchdog; counter-free encoder granularity A/B and exact vocabulary"]
fn flash_next_mlx_encoder_granularity_cost() {
    fixed_work_cost(1);
}

#[test]
#[ignore = "full checkpoint and watchdog; bounded-thread pipeline compilation A/B and exact vocabulary"]
fn flash_next_mlx_pipeline_thread_limit_cost() {
    fixed_work_cost(2);
}

#[test]
#[ignore = "full checkpoint and watchdog; grouped prompt chunks retain logical contractions, full vocabulary A/B"]
fn flash_next_mlx_grouped_prompt_cost() {
    fixed_work_cost(3);
}

fn fixed_work_cost(experiment: u8) {
    use crate::device::{ISOLATE_ENCODERS_FOR_TEST, LIMIT_PIPELINE_THREADS_FOR_TEST};
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ISOLATE_ENCODERS_FOR_TEST.with(|v| v.set(false));
            LIMIT_PIPELINE_THREADS_FOR_TEST.with(|v| v.set(false));
            prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let profile = std::env::var_os("PADDOCK_METAL_PROFILE").is_some();
    assert!(experiment == 0 || !profile, "A/B must not use counters");
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let seed = fixtures["prompts"][0]["token_ids"].as_array().unwrap();
    assert!(!seed.is_empty());
    let continuation = (100..132u32).collect::<Vec<_>>();
    LIMIT_PIPELINE_THREADS_FOR_TEST.with(|v| v.set(experiment == 2));
    // One allocation for both controls. Wider scratch is elected only inside
    // the existing grant, with unchanged residency and retained-prefix count.
    prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.set(experiment == 3));
    let mut m = FlashNext::load(Path::new(&path), 16384, 1, Some(budget)).unwrap();
    if experiment == 3 {
        assert_eq!(m.capacity, 2048);
    }
    m.prefix.clear(&mut m.pool);
    let _prefix = std::mem::take(&mut m.prefix);
    for length in if profile {
        vec![4096]
    } else {
        vec![4096, 8192]
    } {
        let ids = seed
            .iter()
            .cycle()
            .take(length)
            .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(ids.iter().all(|&t| (t as usize) < VOCAB));
        eprintln!(
            "FLASH_FIXED_INPUT {}",
            serde_json::json!({"length":length,"ids":ids,"continuation":continuation})
        );
        let mut reference = None;
        for (round, isolate) in (0..if profile { 2 } else { 4 }).flat_map(|round| {
            if experiment != 0 {
                vec![(round, round % 2 == 1), (round, round % 2 != 1)]
            } else {
                vec![(round, false)]
            }
        }) {
            ISOLATE_ENCODERS_FOR_TEST.with(|v| v.set(experiment == 1 && isolate));
            LIMIT_PIPELINE_THREADS_FOR_TEST.with(|v| v.set(experiment == 2 && isolate));
            m.reset();
            m.slots[0].plan = prompt::Plan::grid(length, MLX_CHUNK);
            let mut waves = Vec::new();
            let mut gpu = 0.;
            let started = std::time::Instant::now();
            let physical_chunk = if experiment == 3 && isolate {
                2048
            } else {
                MLX_CHUNK
            };
            for offset in (0..length - 1).step_by(physical_chunk) {
                let end = (offset + physical_chunk).min(length - 1);
                let rows = (offset..end)
                    .map(|p| (0, ids[p], p as u32))
                    .collect::<Vec<_>>();
                let wave = std::time::Instant::now();
                if experiment == 3 && isolate {
                    m.execute_planned(&rows, &[]).unwrap();
                } else {
                    m.execute(&rows, &[]).unwrap();
                }
                waves.push(wave.elapsed().as_secs_f64() * 1000.);
                gpu += m.last_gpu_seconds;
            }
            ISOLATE_ENCODERS_FOR_TEST.with(|v| v.set(false));
            LIMIT_PIPELINE_THREADS_FOR_TEST.with(|v| v.set(false));
            let first = m
                .execute(&[(0, ids[length - 1], (length - 1) as u32)], &[0])
                .unwrap();
            gpu += m.last_gpu_seconds;
            let prefill_ms = started.elapsed().as_secs_f64() * 1000.;
            let mut logits = first.clone();
            let mut decode_ms = Vec::new();
            let mut decode_gpu_ms = Vec::new();
            for (step, &token) in continuation.iter().enumerate() {
                let begin = std::time::Instant::now();
                let output = m
                    .execute(&[(0, token, (length + step) as u32)], &[0])
                    .unwrap();
                decode_ms.push(begin.elapsed().as_secs_f64() * 1000.);
                decode_gpu_ms.push(m.last_gpu_seconds * 1000.);
                logits.extend(output);
            }
            assert_eq!(logits.len(), VOCAB * (1 + continuation.len()));
            assert!(logits.iter().all(|x| x.is_finite()));
            let signature = blake3::hash(
                &logits
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            if let Some(previous) = reference {
                assert_eq!(signature, previous, "repeated execution changed vocabulary");
            }
            reference = Some(signature);
            let first_greedy = first
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap()
                .0;
            eprintln!(
                "FLASH_FIXED_COST {}",
                serde_json::json!({"engine":"paddock","length":length,"chunk":MLX_CHUNK,
                    "round":round,"mode":if profile {"stages"} else if round == 0 {"warm"} else {"control"},
                    "isolated_prefill":experiment == 1 && isolate,"encoder_comparison":experiment == 1,
                    "limited_threads":experiment == 2 && isolate,"pipeline_comparison":experiment == 2,
                    "physical_chunk":physical_chunk,"grouped_prompt_comparison":experiment == 3,
                    "prefill_ms":prefill_ms,"prefill_gpu_ms":gpu*1000.,"prefill_wave_ms":waves,
                    "decode_ms":decode_ms,"decode_gpu_ms":decode_gpu_ms,
                    "full_logits_blake3":signature.to_hex().to_string(),"repeat_exact":true,
                    "first_greedy":first_greedy,"allocated_bytes":m.device.allocated_bytes()})
            );
        }
    }
}

#[test]
#[ignore = "full checkpoint, watchdog and PADDOCK_METAL_PROFILE; instrumented stage attribution, not serving timing"]
fn flash_next_mlx_prefill_stage_attribution() {
    assert!(std::env::var_os("PADDOCK_METAL_PROFILE").is_some());
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    for slots in [1, 4] {
        m.reset();
        m.prefix.clear(&mut m.pool);
        let step = if slots == 1 { 1024 } else { 512 };
        for (offset, count, logical) in (0..2048)
            .step_by(step)
            .map(|offset| (offset, step, 1024))
            .chain([(2048, 512, 512)])
        {
            let rows = (0..slots)
                .flat_map(|s| {
                    (offset..offset + count)
                        .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                })
                .collect::<Vec<_>>();
            eprintln!(
                "FLASH_STAGE slots={slots} position={offset} rows={} logical={logical}",
                rows.len()
            );
            m.execute_contracts(&rows, &[], Some(&vec![logical; slots]))
                .unwrap();
        }
        // Mirror the small, unequal message-boundary tails visible in the
        // SDK trace. Their logical contracts must not be replaced by the
        // aggregate physical row count when attributing projection cost.
        let tail_lengths = [14, 9, 14, 12];
        let rows = (0..slots)
            .flat_map(|s| {
                (2560..2560 + tail_lengths[s])
                    .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
            })
            .collect::<Vec<_>>();
        eprintln!(
            "FLASH_STAGE slots={slots} phase=message_tail rows={} logical={:?}",
            rows.len(),
            &tail_lengths[..slots]
        );
        m.execute_contracts(&rows, &[], Some(&tail_lengths[..slots]))
            .unwrap();
        let outputs = (0..slots).collect::<Vec<_>>();
        for step in 0..4 {
            let rows = (0..slots)
                .map(|s| (s, 100 + s as u32 * 17 + step, m.slots[s].length as u32))
                .collect::<Vec<_>>();
            eprintln!("FLASH_STAGE slots={slots} phase=decode step={step} rows={slots}");
            let logits = m.execute(&rows, &outputs).unwrap();
            assert_eq!(logits.len(), slots * VOCAB);
            assert!(logits.iter().all(|x| x.is_finite()));
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; unequal message-tail contractions, full vocabulary bits and rotated GPU costs"]
fn flash_next_mlx_packed_message_tail_execution_cost() {
    message_tail_execution_cost(false, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; shared unpacking against packed-wide, complete vocabulary and rotated costs"]
fn flash_next_mlx_shared_unpack_tail_execution_cost() {
    message_tail_execution_cost(true, false);
}

#[test]
#[ignore = "full checkpoint and watchdog; routed prompt fusion with unchanged shared experts and full vocabulary"]
fn flash_next_mlx_routed_prompt_execution_cost() {
    message_tail_execution_cost(false, true);
}

fn message_tail_execution_cost(reuse: bool, prompt_fusion: bool) {
    use super::super::affine;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            affine::PACKED_WIDE_FOR_TEST.with(|v| v.set(true));
            affine::ROW_REUSE_FOR_TEST.with(|v| v.set(true));
            super::super::moe::PROMPT_GATE_UP_FOR_TEST.with(|v| v.set(4));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        let mut expected = None;
        let routes = if prompt_fusion { 4 } else { 2 };
        for round in 0..3 {
            for index in 0..routes {
                let route = (index + round) % routes;
                let packed = route == 1;
                affine::PACKED_WIDE_FOR_TEST.with(|v| v.set(prompt_fusion || packed || reuse));
                affine::ROW_REUSE_FOR_TEST.with(|v| v.set(prompt_fusion || (packed && reuse)));
                super::super::moe::PROMPT_GATE_UP_FOR_TEST
                    .with(|v| v.set(if prompt_fusion { route as u8 } else { 4 }));
                m.reset();
                m.prefix.clear(&mut m.pool);
                let step = if slots == 1 { 1024 } else { 512 };
                let mut prefill_gpu = 0.;
                for offset in (0..2048).step_by(step) {
                    let rows = (0..slots)
                        .flat_map(|s| {
                            (offset..offset + step)
                                .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                        })
                        .collect::<Vec<_>>();
                    m.execute_contracts(&rows, &[], Some(&vec![1024; slots]))
                        .unwrap();
                    prefill_gpu += m.last_gpu_seconds;
                }
                let lengths = [12, 9, 14, 12];
                let rows = (0..slots)
                    .flat_map(|s| {
                        (2048..2048 + lengths[s])
                            .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                    })
                    .collect::<Vec<_>>();
                let mut end = 0;
                let outputs = lengths[..slots]
                    .iter()
                    .map(|n| {
                        end += n;
                        end - 1
                    })
                    .collect::<Vec<_>>();
                let began = std::time::Instant::now();
                let mut logits = m
                    .execute_contracts(&rows, &outputs, Some(&lengths[..slots]))
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>();
                let tail_gpu = m.last_gpu_seconds;
                let tail_wall = began.elapsed().as_secs_f64();
                let mut decode_gpu = 0.;
                for step in 0..8 {
                    let rows = (0..slots)
                        .map(|s| (s, 100 + s as u32 * 17 + step, m.slots[s].length as u32))
                        .collect::<Vec<_>>();
                    logits.extend(
                        m.execute(&rows, &(0..slots).collect::<Vec<_>>())
                            .unwrap()
                            .into_iter()
                            .map(f32::to_bits),
                    );
                    decode_gpu += m.last_gpu_seconds;
                }
                if let Some(expected) = &expected {
                    assert!(
                        &logits == expected,
                        "packed tail changed vocabulary: slots={slots} round={round} packed={packed}"
                    );
                } else {
                    expected = Some(logits);
                }
                assert_eq!(m.device.allocated_bytes(), allocated);
                eprintln!(
                    "FLASH_PACKED_TAIL {}",
                    serde_json::json!({"slots":slots,"round":round,
                    "packed":packed,"shared_unpack_comparison":reuse,"prompt_fusion":prompt_fusion,"route":route,"tail_gpu_seconds":tail_gpu,"tail_wall_seconds":tail_wall,
                    "prefill_gpu_seconds":prefill_gpu,"decode_step_gpu_seconds":decode_gpu/8.,
                    "rows":rows.len(),"logical_rows":&lengths[..slots],"full_logits_exact":true,"allocated":allocated})
                );
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; changed logical tail lengths must preserve cold full-vocabulary outputs"]
fn flash_next_mlx_tail_classes_preserve_cold_logits() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    assert_eq!(m.chunk, 1024);
    let allocated = m.device.allocated_bytes();
    for (source_len, target_len, reuse_tail) in [
        (1700, 1800, true),
        (1700, 2020, false),
        (2020, 2200, true),
        (1800, 1700, true),
        (1510, 1530, true),
        (1290, 1510, false),
        (751, 800, true),
        (751, 900, false),
    ] {
        let prompt = |n: usize| (0..n).map(|i| 1000 + i as u32 % 3000).collect::<Vec<_>>();
        let source = prompt(source_len);
        let target = prompt(target_len);
        m.reset();
        m.prefix.clear(&mut m.pool);
        let prefix = std::mem::take(&mut m.prefix);
        let mut reference = vec![m.prefill(0, &target).unwrap()];
        for t in [23, 248044, 42] {
            reference.push(m.forward(t).unwrap());
        }
        m.prefix = prefix;
        m.reset();
        m.prefill(0, &source).unwrap();
        m.reset();
        let got = m.prefill(3, &target).unwrap();
        let reused = m.take_prefill_reused(3);
        let cuts = prefix::cuts(source.len(), m.chunk);
        // A shorter prompt cannot restore state beyond its own history.
        let expected = if reuse_tail && cuts[1] < target_len {
            cuts[1]
        } else {
            cuts[0]
        };
        assert_eq!(reused, expected, "source={source_len} target={target_len}");
        assert!(
            got == reference[0],
            "tail restore changed cold logits: {source_len}->{target_len}"
        );
        for (i, t) in [23, 248044, 42].into_iter().enumerate() {
            let pos = m.slots[3].length as u32;
            assert!(
                m.execute(&[(3, t, pos)], &[0]).unwrap() == reference[i + 1],
                "tail restore changed continuation {i}: {source_len}->{target_len}"
            );
        }
        assert_eq!(allocated, m.device.allocated_bytes());
        eprintln!(
            "FLASH_TAIL_CLASS source={source_len} target={target_len} reused={reused} full_logits_exact=true"
        );
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; rotated prompt-fusion costs from identical restored GPU state"]
fn flash_next_mlx_routed_prompt_restored_execution_cost() {
    routed_prompt_restored_cases(0);
}

#[test]
#[ignore = "full checkpoint and watchdog; production prompt-fusion election versus previous runner arithmetic"]
fn flash_next_mlx_routed_prompt_qualified_execution_cost() {
    routed_prompt_restored_cases(1);
}

#[test]
#[ignore = "full checkpoint and watchdog; medium vector tails, extended ordering versus identical restored state"]
fn flash_next_mlx_routed_prompt_order_range_cost() {
    routed_prompt_restored_cases(2);
}

fn routed_prompt_restored_cases(experiment: u8) {
    use super::super::moe;
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            moe::PROMPT_GATE_UP_FOR_TEST.with(|v| v.set(4));
            moe::EXTENDED_ORDER_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    for slots in [1, 4] {
        m.reset();
        m.prefix.clear(&mut m.pool);
        let tokens = |slot: usize, length: usize| {
            (0..length)
                .map(|p| 1000 + p as u32 + slot as u32 * 17)
                .collect::<Vec<_>>()
        };
        let step = if slots == 1 { 1024 } else { 512 };
        for offset in (0..2048).step_by(step) {
            let rows = (0..slots)
                .flat_map(|s| {
                    (offset..offset + step)
                        .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                })
                .collect::<Vec<_>>();
            m.execute_contracts(&rows, &[], Some(&vec![1024; slots]))
                .unwrap();
        }
        for slot in 0..slots {
            let prompt = tokens(slot, 2049);
            m.slots[slot].plan = prompt::Plan::grid(prompt.len(), 1024);
            m.capture_prefix(slot, &prompt).unwrap();
        }
        let cases: &[[usize; 4]] = if experiment == 2 {
            &[
                [128, 1, 1, 1],
                [129, 1, 1, 1],
                [160, 1, 1, 1],
                [192, 4, 4, 4],
                [204, 1, 1, 1],
                [96, 32, 32, 32],
            ]
        } else {
            &[
                [7, 7, 7, 7],
                [12, 9, 14, 12],
                [32, 17, 24, 23],
                [96, 1, 1, 1],
                [128, 32, 32, 32],
            ]
        };
        for lengths in cases {
            let mut expected = None;
            let mut expected_decode = None;
            let routes: &[usize] = match experiment {
                1 => &[0, 4],
                2 => &[0, 1],
                _ => &[0, 1, 2, 3],
            };
            for round in 0..7 {
                for index in 0..routes.len() {
                    let route = routes[(index + round) % routes.len()];
                    moe::PROMPT_GATE_UP_FOR_TEST.with(|v| {
                        v.set(if experiment == 2 { 4 } else { route as u8 });
                    });
                    moe::EXTENDED_ORDER_FOR_TEST.with(|v| v.set(experiment == 2 && route == 1));
                    m.reset();
                    for (slot, &length) in lengths[..slots].iter().enumerate() {
                        let prompt = tokens(slot, 2048 + length + 1);
                        m.slots[slot].plan = prompt::Plan::grid(prompt.len(), 1024);
                        assert_eq!(m.restore_prefix(slot, &prompt).unwrap(), 2048);
                    }
                    let rows = (0..slots)
                        .flat_map(|s| {
                            (2048..2048 + lengths[s])
                                .map(move |p| (s, 1000 + p as u32 + s as u32 * 17, p as u32))
                        })
                        .collect::<Vec<_>>();
                    let mut end = 0;
                    let outputs = lengths[..slots]
                        .iter()
                        .map(|n| {
                            end += n;
                            end - 1
                        })
                        .collect::<Vec<_>>();
                    let began = std::time::Instant::now();
                    let logits = m
                        .execute_contracts(&rows, &outputs, Some(&lengths[..slots]))
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    let wall = began.elapsed().as_secs_f64();
                    if let Some(expected) = &expected {
                        assert!(
                            &logits == expected,
                            "vocabulary changed: slots={slots} round={round} route={route}"
                        );
                    } else {
                        expected = Some(logits);
                    }
                    assert_eq!(m.device.allocated_bytes(), allocated);
                    let tail_gpu_seconds = m.last_gpu_seconds;
                    if experiment == 2 {
                        let mut hash = blake3::Hasher::new();
                        for step in 0..8 {
                            let rows = (0..slots)
                                .map(|slot| {
                                    (
                                        slot,
                                        100 + step + slot as u32,
                                        (2048 + lengths[slot]) as u32 + step,
                                    )
                                })
                                .collect::<Vec<_>>();
                            let logits = m.execute(&rows, &(0..slots).collect::<Vec<_>>()).unwrap();
                            for value in logits {
                                hash.update(&value.to_bits().to_le_bytes());
                            }
                        }
                        if let Some(expected) = expected_decode {
                            assert_eq!(
                                hash.finalize(),
                                expected,
                                "continuation state changed: slots={slots} round={round} route={route}"
                            );
                        } else {
                            expected_decode = Some(hash.finalize());
                        }
                        assert_eq!(m.device.allocated_bytes(), allocated);
                    }
                    eprintln!(
                        "FLASH_RESTORED_PROMPT {}",
                        serde_json::json!({"slots":slots,"round":round,"route":route,"experiment":experiment,"logical_rows":&lengths[..slots],"tail_gpu_seconds":tail_gpu_seconds,"wall_seconds":wall,"full_logits_exact":true,"decode_steps":if experiment==2 {8} else {0},"allocated":allocated})
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; fixed-contract accuracy, complete generations and cross-length cache experiment"]
fn flash_next_mlx_canonical_prefill_diagnostic() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            serving::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(false));
        }
    }
    let _reset = Reset;
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    let allocated = m.device.allocated_bytes();
    let argmax = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
            .unwrap()
            .0 as u32
    };
    for fixture in fixtures["prompts"].as_array().unwrap() {
        let tokens = fixture["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let mut reference: Option<(Vec<f32>, Vec<u32>)> = None;
        for canonical in [false, true] {
            serving::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(canonical));
            m.reset();
            m.prefix.clear(&mut m.pool);
            let prefix = std::mem::take(&mut m.prefix);
            let started = std::time::Instant::now();
            let mut logits = m.prefill(0, &tokens).unwrap();
            let prefill_seconds = started.elapsed().as_secs_f64();
            assert!(logits.iter().all(|v| v.is_finite()));
            let first = logits.clone();
            let mut output = Vec::new();
            for _ in 0..128 {
                let t = argmax(&logits);
                output.push(t);
                if [248044, 248046].contains(&t) {
                    break;
                }
                logits = m.forward(t).unwrap();
            }
            m.prefix = prefix;
            assert!(
                [248044, 248046].contains(output.last().unwrap()),
                "natural EOS required"
            );
            let (relative_l2, max_abs, first_equal, generation_equal) =
                if let Some((a, b)) = &reference {
                    let error = a
                        .iter()
                        .zip(&first)
                        .map(|(&a, &b)| f64::from(a - b).powi(2))
                        .sum::<f64>();
                    let norm = a.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
                    (
                        error.sqrt() / norm.sqrt(),
                        a.iter()
                            .zip(&first)
                            .map(|(&a, &b)| (a - b).abs())
                            .fold(0f32, f32::max),
                        argmax(a) == argmax(&first),
                        b == &output,
                    )
                } else {
                    (0., 0., true, true)
                };
            eprintln!(
                "FLASH_CANONICAL_ACCURACY {}",
                serde_json::json!({
                    "prompt_tokens":tokens.len(), "canonical":canonical, "prefill_seconds":prefill_seconds,
                    "relative_l2":relative_l2, "max_abs":max_abs, "first_equal":first_equal,
                    "generation_equal":generation_equal, "output_tokens":output
                })
            );
            if !canonical {
                reference = Some((first, output));
            }
        }
    }
    // These exact-prefix extensions crossed contraction classes in the
    // observed SDK workload. A candidate must preserve its OWN cold graph's
    // complete vocabulary and subsequent recurrence, not only its argmax.
    for (source_len, target_len) in [(751, 900), (1700, 2020), (1290, 1510)] {
        let source = (0..source_len)
            .map(|i| 1000 + i as u32 % 3000)
            .collect::<Vec<_>>();
        let target = (0..target_len)
            .map(|i| 1000 + i as u32 % 3000)
            .collect::<Vec<_>>();
        for canonical in [false, true] {
            serving::CANONICAL_PREFILL_FOR_TEST.with(|v| v.set(canonical));
            m.reset();
            m.prefix.clear(&mut m.pool);
            let prefix = std::mem::take(&mut m.prefix);
            let mut reference = vec![m.prefill(0, &target).unwrap()];
            for t in [23, 248044, 42] {
                reference.push(m.forward(t).unwrap());
            }
            m.prefix = prefix;
            m.reset();
            m.prefill(0, &source).unwrap();
            m.reset();
            let started = std::time::Instant::now();
            let got = m.prefill(3, &target).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            let reused = m.take_prefill_reused(3);
            assert!(
                got == reference[0],
                "cold/warm mismatch canonical={canonical} {source_len}->{target_len}"
            );
            for (i, t) in [23, 248044, 42].into_iter().enumerate() {
                let pos = m.slots[3].length as u32;
                assert!(m.execute(&[(3, t, pos)], &[0]).unwrap() == reference[i + 1]);
            }
            if canonical {
                assert_eq!(reused, prefix::cuts(source_len, m.chunk)[1]);
            }
            assert_eq!(allocated, m.device.allocated_bytes());
            eprintln!(
                "FLASH_CANONICAL_RESTORE {}",
                serde_json::json!({
                    "canonical":canonical, "source_tokens":source_len, "target_tokens":target_len,
                    "reused":reused, "seconds":seconds, "full_logits_exact":true
                })
            );
        }
    }
}

#[test]
#[ignore = "full checkpoint and watchdog; message-prefix full vocabularies, natural follow-ups and mixed arrivals"]
fn flash_next_mlx_message_prefix_generations() {
    message_prefix_generations(0);
}

#[test]
#[ignore = "full checkpoint and watchdog; grouped prefill natural-EOS, prefix restores, arrivals and cancellation"]
fn flash_next_mlx_grouped_prompt_generations() {
    message_prefix_generations(1);
}

#[test]
#[ignore = "full checkpoint and watchdog; extended expert order natural-EOS, prefix restores, arrivals and cancellation"]
fn flash_next_mlx_extended_order_generations() {
    message_prefix_generations(2);
}

fn message_prefix_generations(experiment: u8) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            forward::CACHE_ONLY_TAIL_FOR_TEST.with(|v| v.set(true));
            super::super::affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(false));
            super::super::affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(true));
            super::super::affine::JOINED_SLAB_FOR_TEST.with(|v| v.set(false));
            forward::PLE_LOOKAHEAD_FOR_TEST.with(|v| v.set(false));
            forward::ADAPTIVE_PLE_FOR_TEST.with(|v| v.set(true));
            super::super::moe::VECTOR_GATE_UP_FOR_TEST.with(|v| v.set(true));
            super::super::moe::PAIRED_GATE_UP_FOR_TEST.with(|v| v.set(true));
            super::super::moe::PROMPT_GATE_UP_FOR_TEST.with(|v| v.set(4));
            super::super::residual::HC_VECTOR_FOR_TEST.with(|v| v.set(true));
            super::super::residual::HC_UP_FOR_TEST.with(|v| v.set(true));
            super::super::affine::PACKED_WIDE_FOR_TEST.with(|v| v.set(true));
            super::super::affine::ROW_REUSE_FOR_TEST.with(|v| v.set(true));
            super::super::residual::HC_COMBINE_NORM_FOR_TEST.with(|v| v.set(true));
            super::super::qsa::DIRECT_RUNS_FOR_TEST.with(|v| v.set(true));
            prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.set(false));
            super::super::moe::EXTENDED_ORDER_FOR_TEST.with(|v| v.set(true));
        }
    }
    let _reset = Reset;
    // Independent old full-tail cold execution remains the oracle. Both new
    // cold and restored/mixed execution must preserve every vocabulary bit.
    forward::CACHE_ONLY_TAIL_FOR_TEST.with(|v| v.set(false));
    super::super::affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(false));
    super::super::affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(false));
    super::super::affine::JOINED_SLAB_FOR_TEST.with(|v| v.set(false));
    forward::PLE_LOOKAHEAD_FOR_TEST.with(|v| v.set(false));
    forward::ADAPTIVE_PLE_FOR_TEST.with(|v| v.set(false));
    super::super::moe::VECTOR_GATE_UP_FOR_TEST.with(|v| v.set(false));
    super::super::moe::PAIRED_GATE_UP_FOR_TEST.with(|v| v.set(false));
    super::super::moe::PROMPT_GATE_UP_FOR_TEST.with(|v| v.set(0));
    super::super::residual::HC_VECTOR_FOR_TEST.with(|v| v.set(false));
    super::super::residual::HC_UP_FOR_TEST.with(|v| v.set(false));
    super::super::affine::PACKED_WIDE_FOR_TEST.with(|v| v.set(false));
    super::super::affine::ROW_REUSE_FOR_TEST.with(|v| v.set(false));
    super::super::residual::HC_COMBINE_NORM_FOR_TEST.with(|v| v.set(false));
    super::super::qsa::DIRECT_RUNS_FOR_TEST.with(|v| v.set(false));
    prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.set(false));
    super::super::moe::EXTENDED_ORDER_FOR_TEST.with(|v| v.set(false));
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .unwrap()
        .parse()
        .unwrap();
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&path)).unwrap();
    let mut m = FlashNext::load(Path::new(&path), 4096, 4, Some(budget)).unwrap();
    assert!(
        m.markers.is_some(),
        "the real tokenizer must elect the message plan"
    );
    let allocated = m.device.allocated_bytes();
    let argmax = |logits: &[f32]| {
        logits
            .iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
            .unwrap()
            .0 as u32
    };
    let generate = |m: &mut FlashNext, mut logits: Vec<f32>| {
        let mut tokens = Vec::new();
        let mut vocabularies = Vec::new();
        for _ in 0..64 {
            let t = argmax(&logits);
            tokens.push(t);
            vocabularies.push(logits);
            if [248044, 248046].contains(&t) {
                return (tokens, vocabularies);
            }
            logits = m.forward(t).unwrap();
        }
        panic!("natural EOS required, not a truncated parity test");
    };
    let mut sources = Vec::new();
    let mut targets = Vec::new();
    let mut expected = Vec::new();
    for fixture in fixtures["prompts"].as_array().unwrap() {
        let source = fixture["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        m.reset();
        m.prefix.clear(&mut m.pool);
        let first = m.prefill(0, &source).unwrap();
        let (reply, _) = generate(&mut m, first);
        let mut target = source.clone();
        target.extend(reply);
        target.extend(tokenizer.encode("\n<|im_start|>user\nRepeat your previous answer, with no extra words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").unwrap());
        m.reset();
        m.prefix.clear(&mut m.pool);
        let prefix = std::mem::take(&mut m.prefix);
        let first = m.prefill(0, &target).unwrap();
        expected.push(generate(&mut m, first));
        m.prefix = prefix;
        sources.push(source);
        targets.push(target);
    }
    assert_eq!(sources.len(), 4);
    forward::CACHE_ONLY_TAIL_FOR_TEST.with(|v| v.set(true));
    for (experimental, warm) in [
        (0, false),
        (0, true),
        (1, false),
        (1, true),
        (2, false),
        (2, true),
        (3, false),
        (3, true),
        (4, false),
        (4, true),
        (5, false),
        (5, true),
        (6, false),
        (6, true),
        (7, false),
        (7, true),
        (8, false),
        (8, true),
        (9, false),
        (9, true),
        (10, false),
        (10, true),
        (11, false),
        (11, true),
    ]
    .into_iter()
    .filter(|(mode, _)| match experiment {
        1 => matches!(mode, 9 | 10),
        2 => matches!(mode, 9 | 11),
        _ => true,
    }) {
        super::super::affine::SHARED_INPUT_FOR_TEST.with(|v| v.set(experimental == 1));
        forward::PLE_LOOKAHEAD_FOR_TEST.with(|v| v.set(experimental == 1));
        forward::ADAPTIVE_PLE_FOR_TEST.with(|v| v.set(experimental >= 2));
        super::super::affine::JOINED_INPUT_FOR_TEST.with(|v| v.set(experimental >= 2));
        super::super::affine::JOINED_SLAB_FOR_TEST.with(|v| v.set(experimental == 3));
        super::super::moe::VECTOR_GATE_UP_FOR_TEST.with(|v| v.set(experimental >= 4));
        super::super::residual::HC_VECTOR_FOR_TEST.with(|v| v.set(experimental >= 4));
        super::super::residual::HC_UP_FOR_TEST.with(|v| v.set(experimental >= 4));
        super::super::affine::PACKED_WIDE_FOR_TEST.with(|v| v.set(experimental >= 5));
        super::super::residual::HC_COMBINE_NORM_FOR_TEST.with(|v| v.set(experimental >= 6));
        super::super::qsa::DIRECT_RUNS_FOR_TEST.with(|v| v.set(experimental >= 6));
        super::super::affine::ROW_REUSE_FOR_TEST.with(|v| v.set(experimental >= 7));
        super::super::moe::PAIRED_GATE_UP_FOR_TEST.with(|v| v.set(experimental >= 8));
        super::super::moe::PROMPT_GATE_UP_FOR_TEST
            .with(|v| v.set(if experimental >= 9 { 4 } else { 0 }));
        prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.set(experimental == 10));
        super::super::moe::EXTENDED_ORDER_FOR_TEST.with(|v| v.set(experimental == 11));
        m.reset();
        m.prefix.clear(&mut m.pool);
        if warm {
            for (slot, source) in sources.iter().enumerate() {
                m.prefill(slot, source).unwrap();
            }
        }
        // Cold single-user execution must also exercise same-slot grouping;
        // concurrent grants alone can fill the arena with one chunk per slot.
        if experiment != 0 {
            for (slot, target) in targets.iter().enumerate() {
                m.reset();
                let first = m.prefill(0, target).unwrap();
                let result = generate(&mut m, first);
                assert_eq!(result.0, expected[slot].0);
                assert!(
                    result
                        .1
                        .iter()
                        .flatten()
                        .map(|v| v.to_bits())
                        .eq(expected[slot].1.iter().flatten().map(|v| v.to_bits()))
                );
            }
            m.reset();
            m.prefix.clear(&mut m.pool);
            if warm {
                for (slot, source) in sources.iter().enumerate() {
                    m.prefill(slot, source).unwrap();
                }
            }
            eprintln!(
                "FLASH_SERIAL_GENERATIONS experimental={experimental} warm={warm} exact=4/4 full_vocabularies=true natural_eos=true"
            );
        }
        m.reset();
        let mut actual = vec![Vec::new(); 4];
        let mut ready = [false; 4];
        for (slot, target) in targets.iter().enumerate().take(3) {
            m.prefill_begin(slot, target.clone()).unwrap();
            let reused = m.take_prefill_reused(slot);
            if warm {
                assert!(reused >= m.prompt_plan(&sources[slot]).cuts()[0] && reused > 0);
            } else {
                assert_eq!(reused, 0);
            }
            eprintln!("FLASH_MESSAGE_START warm={warm} slot={slot} reused={reused}");
        }
        for tick in 0..1024 {
            if tick == 2 {
                m.prefill_begin(3, targets[3].clone()).unwrap();
                assert_eq!(m.take_prefill_reused(3) > 0, warm);
                assert!(m.prefill_abort(3));
                m.prefill_begin(3, targets[3].clone()).unwrap();
            }
            let decodes = (0..4)
                .filter_map(|slot| {
                    let &token = actual[slot].last()?;
                    (ready[slot] && ![248044, 248046].contains(&token)).then_some((
                        slot,
                        token,
                        m.slots[slot].length as u32,
                    ))
                })
                .collect::<Vec<_>>();
            let (logits, complete) = m
                .forward_mixed(&decodes, [2048, 511, 33, 13][tick % 4])
                .unwrap();
            for (i, &(slot, _, _)) in decodes.iter().enumerate() {
                let row = &logits[i * VOCAB..(i + 1) * VOCAB];
                assert!(
                    row.iter()
                        .map(|v| v.to_bits())
                        .eq(expected[slot].1[actual[slot].len()]
                            .iter()
                            .map(|v| v.to_bits())),
                    "follow-up decode vocabulary differs warm={warm} slot={slot}"
                );
                actual[slot].push(argmax(row));
            }
            for (slot, logits, _) in complete {
                assert!(
                    logits
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(expected[slot].1[0].iter().map(|v| v.to_bits())),
                    "follow-up prefill vocabulary differs warm={warm} slot={slot}"
                );
                actual[slot].push(argmax(&logits));
                ready[slot] = true;
            }
            assert_eq!(allocated, m.device.allocated_bytes());
            assert!(actual.iter().all(|v| v.len() <= 64));
            if actual
                .iter()
                .all(|v| v.last().is_some_and(|t| [248044, 248046].contains(t)))
            {
                break;
            }
        }
        for (got, reference) in actual.iter().zip(&expected) {
            assert_eq!(*got, reference.0);
        }
        eprintln!(
            "FLASH_MESSAGE_GENERATIONS experimental={experimental} warm={warm} exact=4/4 full_vocabularies=true natural_eos=true arrivals=true cancel=true"
        );
    }
    m.reset();
    m.prefix.clear(&mut m.pool);
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
}

#[test]
#[ignore = "112 GB wide-prefix generation gate; elected model, fixtures and watchdog required"]
fn flash_next_mlx_prefix_wide_serial_generations() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap();
    let fixtures: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_FLASH_NEXT_PREFIX_FIXTURES").unwrap()).unwrap(),
    )
    .unwrap();
    let budget = std::env::var("PADDOCK_FLASH_NEXT_TEST_BUDGET_BYTES")
        .ok()
        .map(|v| v.parse::<u64>().expect("positive budget bytes"));
    let mut m = FlashNext::load(Path::new(&path), 4096, 1, budget).unwrap();
    assert!(
        m.chunk >= 256,
        "this gate must exercise grouped matrix arithmetic"
    );
    let allocated = m.device.allocated_bytes();
    eprintln!("FLASH_PREFIX_WIDE rows={} allocated={allocated}", m.chunk);
    let generate = |m: &mut FlashNext, mut logits: Vec<f32>| {
        let mut tokens = Vec::new();
        for _ in 0..64 {
            let t = logits
                .iter()
                .enumerate()
                .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
                .unwrap()
                .0 as u32;
            tokens.push(t);
            if [248044, 248046].contains(&t) {
                return tokens;
            }
            logits = m.forward(t).unwrap();
        }
        panic!("a capped generation is not a natural-EOS gate");
    };
    let prompts = fixtures["prompts"].as_array().unwrap();
    assert_eq!(prompts.len(), 4);
    for fixture in prompts {
        let prompt = fixture["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        m.reset();
        m.prefix.clear(&mut m.pool);
        let prefix = std::mem::take(&mut m.prefix);
        let cold = m.prefill(0, &prompt).unwrap();
        let reference = generate(&mut m, cold.clone());
        m.prefix = prefix;
        m.reset();
        assert_eq!(
            m.prefill(0, &prompt).unwrap(),
            cold,
            "wide capture changed logits"
        );
        // Overwrite live recurrence and PLE history before restoring. The saved
        // GPU state must be independent, not an alias of the original slot.
        m.reset();
        m.prefill(0, &[17, 42, 248044]).unwrap();
        m.reset();
        let warm = m.prefill(0, &prompt).unwrap();
        let reused = m.take_prefill_reused(0);
        assert_eq!(reused, m.prompt_plan(&prompt).cuts()[1]);
        assert_eq!(warm, cold, "wide restore changed full-vocabulary logits");
        assert_eq!(
            generate(&mut m, warm),
            reference,
            "wide complete generation changed"
        );
        assert_eq!(m.device.allocated_bytes(), allocated);
        eprintln!(
            "FLASH_PREFIX_WIDE_EXACT prompt={} reused={reused} natural_eos=true",
            prompt.len()
        );
    }
}

#[test]
fn mlx_prefill_capacity_preserves_explicit_cache_budget() {
    let weight_bytes = 1 << 20;
    for expected in [CHUNK, 256, 512, MLX_CHUNK] {
        let (cache, scratch) = FlashNext::mlx_memory_rows(256, 4, expected).unwrap();
        let scratch = scratch + super::super::affine::workspace_bytes(expected) as u64;
        let budget = weight_bytes + cache + scratch;
        let (device, chunk, actual_cache, actual_scratch, entries, paged) =
            FlashNext::mlx_device(256, 4, weight_bytes, 0, Some(budget)).unwrap();
        assert!(!paged);
        let (base_cache, base_scratch) = FlashNext::mlx_memory_rows(256, 4, chunk).unwrap();
        assert_eq!(
            actual_cache,
            base_cache + prefix::PrefixCache::bytes(256, entries)
        );
        assert_eq!(
            actual_scratch,
            base_scratch + super::super::affine::workspace_bytes(chunk) as u64
        );
        assert!(chunk <= expected);
        assert!(weight_bytes + actual_cache + actual_scratch <= budget);
        assert_eq!(device.budget_bytes(), budget);
        assert_eq!(device.allocated_bytes(), 0);
        if entries == 0 {
            assert_eq!(
                chunk, expected,
                "a cache miss must preserve legacy capacity"
            );
        }
    }
    assert!(FlashNext::mlx_device(256, 4, weight_bytes, 0, Some(1)).is_err());
    assert!(FlashNext::mlx_device(256, 4, u64::MAX, 0, None).is_err());
}

#[test]
fn mlx_wide_upgrade_preserves_residency_retention_and_grant() {
    wide_upgrade_preserves_grant(4);
}

#[test]
fn mlx_grouped_prompt_upgrade_preserves_residency_retention_and_grant() {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(prompt::GROUPED_PROMPT_FOR_TEST.with(|v| v.replace(true)));
    wide_upgrade_preserves_grant(1);
}

fn wide_upgrade_preserves_grant(batch: usize) {
    let (context, weights, ple) = (256, 1u64 << 30, 1u64 << 29);
    let (cache, scratch) = FlashNext::mlx_memory_rows(context, batch, MLX_CHUNK).unwrap();
    let entries = batch * 2;
    let cache = cache + prefix::PrefixCache::bytes(context, entries);
    let scratch = scratch + super::super::affine::workspace_bytes(MLX_CHUNK) as u64;
    let (wc, ws) =
        FlashNext::mlx_memory_rows(context, batch, super::super::affine::MAX_ROWS).unwrap();
    assert_eq!(wc + prefix::PrefixCache::bytes(context, entries), cache);
    let wide_scratch =
        ws + super::super::affine::workspace_bytes(super::super::affine::MAX_ROWS) as u64;
    for (space, expected) in [
        (scratch, MLX_CHUNK),
        (wide_scratch, super::super::affine::MAX_ROWS),
    ] {
        let budget = weights + cache + space;
        let (d, rows, actual_cache, actual_scratch, actual_entries, paged) =
            FlashNext::mlx_device(context, batch, weights, ple, Some(budget)).unwrap();
        assert!(
            !paged,
            "wider rows must never force resident weights onto disk"
        );
        assert_eq!(actual_entries, entries);
        assert_eq!(actual_cache, cache);
        assert_eq!(
            rows,
            if d.tensor_accelerated() {
                expected
            } else {
                MLX_CHUNK
            }
        );
        assert_eq!(d.budget_bytes(), budget);
        assert!(weights + actual_cache + actual_scratch <= budget);
        assert_eq!(d.allocated_bytes(), 0);
    }
}

#[test]
fn mlx_file_backed_ple_preserves_grant_and_wide_prefill() {
    let weights = 110_626_145_280;
    let ple = 32_000_153_600;
    let budget = 106334 * 1024 * 1024;
    for batch in [1, 4] {
        let (d, rows, cache, scratch, entries, paged) =
            FlashNext::mlx_device(8192, batch, weights, ple, Some(budget)).unwrap();
        assert!(paged);
        assert_eq!(
            rows,
            if (batch > 1 || prompt::grouping()) && d.tensor_accelerated() {
                super::super::affine::MAX_ROWS
            } else {
                MLX_CHUNK
            }
        );
        assert_eq!(entries, batch * 2);
        assert_eq!(d.budget_bytes(), budget);
        assert_eq!(d.allocated_bytes(), 0);
        assert!(weights - ple + cache + scratch <= budget);
    }
    assert!(FlashNext::mlx_device(8192, 1, weights, ple, Some(weights - ple)).is_err());
    assert!(FlashNext::mlx_device(8192, 1, 1, ple, Some(budget)).is_err());
}

#[test]
#[ignore = "requires elected 82GB GGUF and M5 GPU; not independent model parity"]
fn full_walk_load_reset_mixed_cancel_and_fail_closed() {
    let path = std::env::var("PADDOCK_FLASH_NEXT_MODEL").expect("model");
    let mut m = FlashNext::load(Path::new(&path), 256, 4, None).unwrap();
    let allocated = m.device.allocated_bytes();
    let (cache, scratch) = FlashNext::memory(256, 4).unwrap();
    assert_eq!(m.weight_bytes, 81_950_799_360);
    assert_eq!(allocated, m.weight_bytes + cache + scratch);
    let prompt = [
        17, 42, 88, 248044, 27, 100, 101, 33, 248044, 75, 61, 19, 71, 13, 10, 99, 31,
    ];
    let first = m.prefill(0, &prompt).unwrap();
    assert_eq!(first.len(), VOCAB);
    assert!(first.iter().all(|x| x.is_finite()));
    eprintln!(
        "full walk finite; allocated={allocated}, last_gpu_ms={}",
        m.last_gpu_seconds * 1000.
    );
    let next = m.forward(123).unwrap();
    m.reset();
    let repeated = m.prefill(0, &prompt).unwrap();
    assert_eq!(
        first, repeated,
        "same-shaped reset/replay must be byte identical"
    );
    assert_eq!(next, m.forward(123).unwrap());
    // Invalid later rows must not advance an earlier valid slot or mutate its
    // GPU caches. Replay continuation against the same-shaped clean walk.
    assert!(
        m.execute(&[(0, 124, 18), (1, VOCAB as u32, 0)], &[0])
            .is_err()
    );
    assert_eq!(m.slots[0].length, 18);
    assert!(!m.poisoned);
    let continuation = m.forward(124).unwrap();
    m.reset();
    m.prefill(0, &prompt).unwrap();
    m.forward(123).unwrap();
    assert_eq!(continuation, m.forward(124).unwrap());
    m.reset();
    for i in 0..4 {
        m.prefill_begin(i, prompt.to_vec()).unwrap();
    }
    assert!(m.prefill_begin(0, prompt.to_vec()).is_err());
    let mut done = Vec::new();
    while !m.pending.is_empty() {
        let (decode, complete) = m.forward_mixed(&[], 33).unwrap();
        assert!(decode.is_empty());
        done.extend(complete);
    }
    assert_eq!(done.len(), 4);
    for (slot, logits, work) in done {
        assert_eq!(work, prompt.len());
        assert_eq!(m.slots[slot].length, prompt.len());
        let error = logits
            .iter()
            .zip(&first)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        eprintln!("full walk schedule diagnostic slot={slot} max_logit_diff={error}");
        // Diagnostic arithmetic tolerance only. Independent greedy generation
        // parity is a separate same-GGUF llama.cpp gate, never replaced by this.
        assert!(
            logits
                .iter()
                .zip(&first)
                .all(|(x, y)| x.is_finite() && (x - y).abs() <= 0.02 + 0.001 * y.abs())
        );
    }
    m.prefill_begin(2, vec![17; 129]).unwrap();
    let _ = m.forward_mixed(&[(0, 123, 17)], 31).unwrap();
    assert!(m.prefill_abort(2));
    assert_eq!(m.slots[2].length, 0);
    m.prefill_begin(2, prompt.to_vec()).unwrap();
    let (_, done) = m.forward_mixed(&[], 128).unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].0, 2);
    assert_eq!(done[0].1, first);
    assert_eq!(m.device.allocated_bytes(), allocated);
    m.release_inactive_slots(&[false; 4]);
    assert_eq!(m.pool.free_blocks(), m.pool.capacity() as usize);
    // Simulate a latched whole-walk failure, not a fabricated Metal error.
    m.poisoned = true;
    m.reset();
    m.prefill_abort(0);
    assert!(m.forward(17).unwrap_err().to_string().contains("poisoned"));
    assert!(
        m.prefill_begin(0, prompt.to_vec())
            .unwrap_err()
            .to_string()
            .contains("poisoned")
    );
}
