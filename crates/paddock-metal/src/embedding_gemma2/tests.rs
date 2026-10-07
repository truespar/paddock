use super::*;

#[test]
#[ignore = "real checkpoint; PADDOCK_EG2_MODEL"]
fn embeddinggemma2_batch_boundaries_and_allocation_recovery() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let mut m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    let sequence = |n: usize| {
        (0..n)
            .map(|i| match i {
                0 => 2,
                i if i == n - 1 => 1,
                i => (i * 7919 % 10000 + 300) as u32,
            })
            .collect::<Vec<_>>()
    };
    let seqs: Vec<_> = [31, 32, 33, 511, 512, 513, 1023, 1024, 1025]
        .into_iter()
        .map(sequence)
        .collect();
    let p = m.embed_submit(&seqs, 0).unwrap();
    let expected = m.embed_collect(&p).unwrap();
    drop(p);
    for (seq, expected) in seqs.iter().zip(&expected) {
        let p = m.embed_submit(std::slice::from_ref(seq), 0).unwrap();
        assert_eq!(
            m.embed_collect(&p).unwrap()[0],
            *expected,
            "length {}",
            seq.len()
        );
    }
    let reversed: Vec<_> = seqs.iter().rev().cloned().collect();
    let p = m.embed_submit(&reversed, 0).unwrap();
    let mut result = m.embed_collect(&p).unwrap();
    result.reverse();
    assert_eq!(result, expected, "arrival order changed embeddings");
    drop(p);
    let longest = [sequence(CONTEXT)];
    let p = m.embed_submit(&longest, 0).unwrap();
    let full = m.embed_collect(&p).unwrap();
    drop(p);
    m.reclaim_idle();
    let p = m.embed_submit(&longest, 0).unwrap();
    assert_eq!(
        m.embed_collect(&p).unwrap(),
        full,
        "8K changed after reclaim"
    );
    drop(p);
    let weights = m.weight_bytes;
    drop(m);

    // A user budget that holds the weights and short requests, but not a long
    // scratch generation. Refusal must release partial allocations, preserve
    // the weights, and leave the endpoint able to answer smaller requests.
    let mut m = EmbeddingGemma2::load(Path::new(&path), 8192, Some(weights + (32 << 20))).unwrap();
    let short = [sequence(31)];
    let p = m.embed_submit(&short, 0).unwrap();
    let before = m.embed_collect(&p).unwrap();
    drop(p);
    let error = match m.embed_submit(&longest, 0) {
        Ok(_) => panic!("oversized scratch fit a deliberately small budget"),
        Err(e) => e,
    };
    assert!(error.contains("budget"), "{error}");
    assert!(m.scratch.is_none());
    assert_eq!(m.device.allocated_bytes(), weights);
    let p = m.embed_submit(&short, 0).unwrap();
    assert_eq!(m.embed_collect(&p).unwrap(), before);
}

#[test]
#[ignore = "real checkpoint; PADDOCK_EG2_MODEL"]
fn embeddinggemma2_workspace_lifetime() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let mut m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    let weights = m.device.allocated_bytes();
    assert_eq!(weights, m.weight_bytes);
    let short = [vec![2, 1000, 1]];
    let p = m.embed_submit(&short, 0).unwrap();
    let expected = m.embed_collect(&p).unwrap();
    drop(p);
    let small = m.device.allocated_bytes();
    assert!(small - weights < 8 << 20);
    assert_eq!(m.scratch.as_ref().unwrap().rows, 64);
    // Grow while a previous command still owns the smaller generation, then
    // enqueue a third command reusing the larger allocation in queue order.
    let p = m.embed_submit(&short, 0).unwrap();
    let q = m.embed_submit(&[vec![1000; 982]], 0).unwrap();
    let r = m.embed_submit(&short, 0).unwrap();
    assert_eq!(m.scratch.as_ref().unwrap().rows, 1024);
    m.reclaim_idle();
    assert!(m.scratch.is_none());
    assert!(m.device.allocated_bytes() > small);
    assert_eq!(m.embed_collect(&p).unwrap(), expected);
    assert_eq!(m.embed_collect(&r).unwrap(), expected);
    drop(q); // Cancellation still fences the buffers, even without collect.
    drop(p);
    drop(r);
    assert_eq!(m.device.allocated_bytes(), weights);
    assert!(m.idle_reclaim_after().is_none());
    let p = m.embed_submit(&short, 0).unwrap();
    assert_eq!(m.embed_collect(&p).unwrap(), expected);
    drop(p);
    m.reclaim_idle();
    assert_eq!(m.device.allocated_bytes(), weights);
    eprintln!(
        "workspace: weights={weights}, short={small}, reclaimed={}",
        m.device.allocated_bytes()
    );
}

#[test]
#[ignore = "same-checkpoint MLX GPU operation captures; PADDOCK_EG2_OPERATIONS"]
fn embeddinggemma2_rotary_positions() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let fixtures = std::env::var("PADDOCK_EG2_OPERATIONS").unwrap();
    let m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    for (layer, hd, theta) in [(0, 256, 10000f32), (5, 512, 1000000f32)] {
        let f = paddock_models::safetensors::SafetensorsFile::open(
            &Path::new(&fixtures).join(format!("rope{hd}.safetensors")),
        )
        .unwrap();
        let (_, input) = f.bytes("x").unwrap();
        let (_, expected) = f.bytes("y").unwrap();
        let (_, positions) = f.bytes("positions").unwrap();
        let rows = positions.len() / 4;
        let meta = positions
            .chunks_exact(4)
            .flat_map(|b| {
                let p = f32::from_le_bytes(b.try_into().unwrap()) as u32;
                [0, p].into_iter().flat_map(u32::to_le_bytes)
            })
            .collect::<Vec<_>>();
        let x = m.device.upload(input).unwrap();
        let meta = m.device.upload(&meta).unwrap();
        let y = m.device.alloc(expected.len()).unwrap();
        let c = m.device.begin().unwrap();
        c.dispatch(
            "eg2_heads",
            &[&x, &m.layers[layer].qn.buffer, &meta, &y],
            &[4, hd, 0, 1, theta.to_bits(), 0],
            [4, rows, 1],
            hd as usize / 4,
        );
        c.finish().unwrap();
        let actual = unsafe { y.read_f32(0, expected.len() / 4) };
        let expected = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let unequal = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("rotary{hd}: {unequal}/{} differ, max={max}", actual.len());
        assert_eq!(unequal, 0);
    }
}

#[test]
#[ignore = "same-checkpoint MLX GPU operation captures; PADDOCK_EG2_OPERATIONS"]
fn embeddinggemma2_operations() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let fixtures = std::env::var("PADDOCK_EG2_OPERATIONS").unwrap();
    let m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    for op in [
        "pre", "ple", "ple_norm", "q", "k", "v", "q_heads", "k_heads", "v_heads", "geglu",
        "ple_gate",
    ] {
        let f = paddock_models::safetensors::SafetensorsFile::open(
            &Path::new(&fixtures).join(format!("{op}.safetensors")),
        )
        .unwrap();
        let (info, data) = f.bytes("x").unwrap();
        let rows = info.shape[..info.shape.len() - 1].iter().product::<usize>();
        let x = m.device.upload(data).unwrap();
        let (_, expected) = f.bytes("y").unwrap();
        let out = m.device.alloc(expected.len()).unwrap();
        let c = m.device.begin().unwrap();
        let l = &m.layers[0];
        let mut hold = Vec::new();
        match op {
            "pre" | "ple_norm" => m.norm(
                &c,
                &x,
                if op == "pre" { &l.pre } else { &m.ple_norm },
                &out,
                rows,
                1.,
                None,
            ),
            "ple" | "q" | "k" | "v" => m.linear(
                &c,
                match op {
                    "ple" => &m.ple,
                    "q" => &l.q,
                    "k" => &l.k,
                    _ => &l.v,
                },
                &x,
                &out,
                rows,
            ),
            "geglu" => {
                hold.push(m.device.upload(f.bytes("up").unwrap().1).unwrap());
                c.dispatch(
                    "gmlx_geglu",
                    &[&x, &hold[0]],
                    &[(data.len() / 4) as u32],
                    [(data.len() / 4).div_ceil(256), 1, 1],
                    256,
                );
            }
            "ple_gate" => {
                let (_, values) = f.bytes("ple").unwrap();
                let packed = values
                    .chunks_exact(WIDTH * 4)
                    .flat_map(|row| row.repeat(LAYERS))
                    .collect::<Vec<_>>();
                hold.push(m.device.upload(&packed).unwrap());
                c.dispatch(
                    "eg2_ple_gate",
                    &[&x, &hold[0]],
                    &[rows as u32, 0, 1],
                    [(rows * WIDTH).div_ceil(256), 1, 1],
                    256,
                );
            }
            _ => {
                let count = info.shape[info.shape.len() - 2];
                let meta = (0..rows)
                    .flat_map(|r| [(r / count * count) as u32, (r % count) as u32])
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>();
                hold.push(m.device.upload(&meta).unwrap());
                let heads = if op == "q_heads" { 4 } else { 2 };
                c.dispatch(
                    "eg2_heads",
                    &[
                        &x,
                        if op == "q_heads" {
                            &l.qn.buffer
                        } else {
                            &l.kn.buffer
                        },
                        &hold[0],
                        &out,
                    ],
                    &[
                        heads,
                        256,
                        0,
                        1,
                        if op == "v_heads" {
                            0
                        } else {
                            10000f32.to_bits()
                        },
                        0,
                    ],
                    [heads as usize, rows, 1],
                    64,
                );
            }
        }
        c.finish().unwrap();
        let actual = unsafe {
            if matches!(op, "geglu" | "ple_gate") {
                x.read_f32(0, expected.len() / 4)
            } else {
                out.read_f32(0, expected.len() / 4)
            }
        };
        let expected = expected
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let diff = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        eprintln!("{op}: max={max:.8} unequal={diff}/{}", actual.len());
        assert!(actual.iter().all(|v| v.is_finite()));
        if matches!(
            op,
            "pre" | "ple_norm" | "q_heads" | "k_heads" | "v_heads" | "geglu" | "ple_gate"
        ) {
            assert_eq!(
                diff, 0,
                "isolated {op} no longer matches upstream GPU arithmetic"
            );
        }
    }
}

#[test]
#[ignore = "real same-checkpoint GPU reference; PADDOCK_EG2_MODEL and PADDOCK_EG2_FIXTURE"]
fn embeddinggemma2_reference() {
    let path = std::env::var("PADDOCK_EG2_MODEL").expect("model");
    let fixture = std::env::var("PADDOCK_EG2_FIXTURE").expect("fixture");
    let data: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture).expect("read fixture")).expect("json");
    let mut m = EmbeddingGemma2::load(Path::new(&path), 8192, None).expect("load");
    assert!(m.validate(&[]).is_err());
    assert!(m.validate(&[vec![]]).is_err());
    assert!(m.validate(&[vec![VOCAB as u32]]).is_err());
    assert!(m.validate(&[vec![258880]]).is_err());
    assert!(m.validate(&[vec![1; CONTEXT + 1]]).is_err());
    assert!(
        m.embed_submit_dimensions(&[vec![2, 1]], 0, Some(127))
            .is_err()
    );
    let mut failures = Vec::new();
    for case in data["cases"].as_array().expect("cases") {
        if std::env::var("PADDOCK_EG2_CASE").is_ok_and(|v| case["name"] != v) {
            continue;
        }
        let ids: Vec<Vec<u32>> = serde_json::from_value(case["ids"].clone()).expect("ids");
        let refs: Vec<Vec<f32>> =
            serde_json::from_value(case["embeddings"].clone()).expect("embeddings");
        let start = std::time::Instant::now();
        let p = m.embed_submit(&ids, 0).expect("submit");
        p.completion.wait().expect("completion");
        if let (Some(t), Ok(path)) = (&p.trace, std::env::var("PADDOCK_EG2_TRACE")) {
            let n = ids.iter().map(Vec::len).sum::<usize>();
            let data = unsafe { t.read_f32(0, (LAYERS + 1) * n * WIDTH) };
            std::fs::write(
                path,
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .expect("trace");
        }
        let values = m.embed_collect(&p).expect("collect");
        assert_eq!(values.len(), refs.len(), "reference row count changed");
        assert_eq!(values.len(), ids.len(), "embedding row count changed");
        assert!(refs.iter().all(|v| v.len() == DIM));
        assert!(values.iter().all(|v| v.len() == DIM));
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let mut samples = Vec::new();
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let p = m.embed_submit(&ids, 0).expect("repeat submit");
            let next = m.embed_collect(&p).expect("repeat collect");
            samples.push(start.elapsed().as_secs_f64() * 1000.);
            assert_eq!(
                next, values,
                "repeated request changed the native embedding"
            );
        }
        eprintln!("{} warm batch milliseconds: {samples:?}", case["name"]);
        for (i, (a, b)) in values.iter().zip(&refs).enumerate() {
            let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
            let norm: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
            let refnorm: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
            let cos = dot / (norm * refnorm).sqrt();
            let err = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{} row {i}: cosine={cos:.9}, max_abs={err:.6}, batch_ms={ms:.3}",
                case["name"]
            );
            if let Some(serial) = case["serial_embeddings"][i].as_array() {
                let dot: f64 = a
                    .iter()
                    .zip(serial)
                    .map(|(a, b)| *a as f64 * b.as_f64().unwrap())
                    .sum();
                let norm: f64 = serial.iter().map(|v| v.as_f64().unwrap().powi(2)).sum();
                eprintln!("vs MLX serial: {:.9}", dot / norm.sqrt());
            }
            assert!((norm - 1.).abs() < 1e-4);
            if cos <= 0.9999 || err >= 0.004 {
                failures.push(format!("{} row {i}", case["name"]));
            }
        }
        for (i, ids) in ids.iter().enumerate() {
            if ids.len() > 32 {
                continue;
            }
            let p = m
                .embed_submit(std::slice::from_ref(ids), 0)
                .expect("serial submit");
            let single = m.embed_collect(&p).expect("serial collect");
            assert_eq!(&single[0], &values[i], "batch changed the native embedding");
        }
        if case["name"] == "short" {
            for dim in [128, 256, 512, 768] {
                let p = m
                    .embed_submit_dimensions(&ids, 0, Some(dim))
                    .expect("MRL submit");
                let resized = m.embed_collect(&p).expect("MRL collect");
                assert_eq!(resized.len(), values.len());
                for (small, full) in resized.iter().zip(&values) {
                    assert_eq!(small.len(), dim);
                    let ss: f64 = small.iter().map(|&v| (v as f64).powi(2)).sum();
                    let fs: f64 = full[..dim].iter().map(|&v| (v as f64).powi(2)).sum();
                    let dot: f64 = small
                        .iter()
                        .zip(full)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    assert!((ss - 1.).abs() < 1e-4);
                    assert!(dot / (ss * fs).sqrt() > 0.999999);
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "same-weight comparison failures: {failures:?}"
    );
}
