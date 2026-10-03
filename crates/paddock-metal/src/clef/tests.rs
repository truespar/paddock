use super::*;
use paddock_engine::clef_decision::ClefQuestion;

#[test]
fn plan_is_bounded_and_request_isolated() {
    let q = ClefQuestion {
        qtype: 1,
        span: (0, 2),
        options: vec![(1, 3), (3, 4)],
    };
    let ids = vec![1; 19];
    let a = ClefRequest {
        ids: &ids,
        questions: std::slice::from_ref(&q),
        images: &[],
    };
    let p = plan::Plan::new(
        &[
            ClefRequest {
                ids: &ids,
                questions: std::slice::from_ref(&q),
                images: &[],
            },
            a,
        ],
        100,
    )
    .unwrap();
    assert_eq!(p.runs, [0, 19, 19, 19]);
    assert_eq!(p.globals, [18, 19, 37, 38]);
    assert_eq!(
        p.causal,
        [0, 16, 0, 19, 16, 3, 0, 19, 19, 16, 19, 19, 35, 3, 19, 19]
    );
    assert_eq!(p.options, [1, 3, 3, 4, 20, 22, 22, 23]);
    assert_eq!(p.self_tiles, [0, 1, 0, 1, 1, 1, 1, 1]);
    assert_eq!(p.qof, [0, 0, 1, 1]);
    assert!(plan::Plan::new(&[], 100).is_err());
    assert!(
        plan::Plan::new(
            &[ClefRequest {
                ids: &[101],
                questions: &[q],
                images: &[],
            }],
            100
        )
        .is_err()
    );
}

#[test]
fn plan_rejects_every_workspace_boundary_before_submission() {
    let ids = vec![1; MAX_ROWS];
    let question = ClefQuestion {
        qtype: 1,
        span: (0, 1),
        options: vec![(0, 1)],
    };
    let check = |ids: &[u32], questions: &[ClefQuestion]| {
        plan::Plan::new(
            &[ClefRequest {
                ids,
                questions,
                images: &[],
            }],
            100,
        )
    };
    assert!(check(&ids, std::slice::from_ref(&question)).is_ok());
    assert!(check(&vec![1; MAX_ROWS + 1], std::slice::from_ref(&question)).is_err());
    assert!(check(&[], std::slice::from_ref(&question)).is_err());
    assert!(check(&[1], &[]).is_err());
    let questions = (0..=MAX_QUESTIONS)
        .map(|_| ClefQuestion {
            qtype: 1,
            span: (0, 1),
            options: vec![(0, 1)],
        })
        .collect::<Vec<_>>();
    assert!(check(&[1], &questions[..MAX_QUESTIONS]).is_ok());
    assert!(check(&[1], &questions).is_err());
    for (qtype, span, options) in [
        (3, (0, 1), vec![(0, 1)]),
        (1, (0, 0), vec![(0, 1)]),
        (1, (0, 2), vec![(0, 1)]),
        (1, (0, 1), vec![]),
        (1, (0, 1), vec![(1, 1)]),
        (1, (0, 1), vec![(0, usize::MAX)]),
        (1, (0, 1), vec![(0, 1); MAX_OPTIONS + 1]),
    ] {
        assert!(
            check(
                &[1],
                &[ClefQuestion {
                    qtype,
                    span,
                    options
                }]
            )
            .is_err()
        );
    }
    let requests = (0..=MAX_REQUESTS)
        .map(|_| ClefRequest {
            ids: &[1],
            questions: std::slice::from_ref(&question),
            images: &[],
        })
        .collect::<Vec<_>>();
    assert!(plan::Plan::new(&requests[..MAX_REQUESTS], 100).is_ok());
    assert!(plan::Plan::new(&requests, 100).is_err());
}

#[test]
fn compiled_kernels_and_fused_projection_are_finite() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let k = 128;
    let n = 128;
    let m = 33;
    let weight = vec![0x3f80u16; k * n]; // exactly representable BF16 one
    let w = d
        .upload(
            &weight
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let x = upload(&d, &vec![1.; k * m]).unwrap();
    let y = d.alloc(m * n * 4).unwrap();
    let l = Linear {
        weight: w.into(),
        bias: None,
        k,
        n,
    };
    let cmd = d.begin().unwrap();
    l.run(&cmd, &x, &y, m, 0);
    cmd.finish().unwrap();
    let actual = unsafe { y.read_f32(0, m * n) };
    assert!(actual.iter().all(|&v| v == 128.));
    let cmd = d.begin().unwrap();
    l.run(&cmd, &x, &y, m, 3);
    cmd.finish().unwrap();
    let actual = unsafe { y.read_f32(0, m * n / 2) };
    assert!(actual.iter().all(|&v| v == 16384.));
    let packed = d.alloc(m * k * 4).unwrap();
    let cmd = d.begin().unwrap();
    point(
        &cmd,
        "clef_prepare",
        &[&x, &packed],
        &[(m * k) as u32],
        m * k,
    );
    l.parts(&cmd, &packed, &y, m, 0);
    cmd.finish().unwrap();
    let actual = unsafe { y.read_f32(0, m * n) };
    assert!(actual.iter().all(|&v| v == 128.));
    let down = Linear {
        weight: d
            .upload(
                &vec![0x3f80u16; 64 * 16]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .into(),
        bias: None,
        k: 64,
        n: 16,
    };
    let result = d.alloc(m * 16 * 4).unwrap();
    let cmd = d.begin().unwrap();
    l.parts(&cmd, &packed, &y, m, 3);
    down.parts(&cmd, &y, &result, m, 0);
    cmd.finish().unwrap();
    let actual = unsafe { result.read_f32(0, m * 16) };
    assert!(actual.iter().all(|&v| v == 1_048_576.));

    // Exercise grouped tile ordering with a partial final group and row/column
    // tails. The two traversals must touch every output exactly once and keep
    // the same reduction, including non-BF16-representable activation values.
    let (k, n, m) = (4096, 1088, 579);
    let weights = (0..k * n)
        .flat_map(|i| (0x3b00u16 + (i % 512) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let l = Linear {
        weight: d.upload(&weights).unwrap().into(),
        bias: None,
        k,
        n,
    };
    let x = upload(
        &d,
        &(0..m * k)
            .map(|i| ((i % 199) as f32 - 99.) / 127.)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let packed = d.alloc(m * k * 4).unwrap();
    let grouped = d.alloc(m * n * 4).unwrap();
    let linear = d.alloc(m * n * 4).unwrap();
    let strict = d.alloc(m * n * 4).unwrap();
    let cmd = d.begin().unwrap();
    point(
        &cmd,
        "clef_prepare",
        &[&x, &packed],
        &[(m * k) as u32],
        m * k,
    );
    l.parts(&cmd, &packed, &grouped, m, 0);
    forward::LINEAR_WALK_FOR_TEST.with(|v| v.set(true));
    l.parts(&cmd, &packed, &linear, m, 0);
    forward::LINEAR_WALK_FOR_TEST.with(|v| v.set(false));
    l.run(&cmd, &x, &strict, m, 0);
    cmd.finish().unwrap();
    let grouped = unsafe { grouped.read_f32(0, m * n) };
    let linear = unsafe { linear.read_f32(0, m * n) };
    let strict = unsafe { strict.read_f32(0, m * n) };
    assert!(grouped.iter().all(|v| v.is_finite()));
    assert_eq!(grouped, linear);
    let error = grouped
        .iter()
        .zip(&strict)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    let energy = strict.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
    assert!((error / energy).sqrt() < 2e-5);
}

#[test]
#[ignore = "requires CLEF_DIR and 23 GiB model memory"]
fn real_checkpoint_is_batch_stable() {
    let dir = std::env::var("CLEF_DIR").expect("CLEF_DIR");
    let mut model = Clef::load(Path::new(&dir), None).unwrap();
    let q = ClefQuestion {
        qtype: 1,
        span: (4, 8),
        options: vec![(10, 14), (16, 20)],
    };
    let ids = (1..=24).collect::<Vec<u32>>();
    let longer = (1..=37).collect::<Vec<u32>>();
    let a = ClefRequest {
        ids: &ids,
        questions: std::slice::from_ref(&q),
        images: &[],
    };
    let b = ClefRequest {
        ids: &longer,
        questions: std::slice::from_ref(&q),
        images: &[],
    };
    let alone = model.forward(std::slice::from_ref(&a)).unwrap();
    let mixed = model.forward(&[b, a]).unwrap();
    assert_eq!(alone[0], mixed[1]);
    eprintln!(
        "Clef logits: {:?}; weight/workspace: {:?}",
        alone,
        model.info()
    );
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn probability(values: &[f32]) -> Vec<f64> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let ex = values
        .iter()
        .map(|&x| (x as f64 - max).exp())
        .collect::<Vec<_>>();
    let sum = ex.iter().sum::<f64>();
    ex.iter().map(|x| x / sum).collect()
}

#[test]
#[ignore = "requires CLEF_DIR and same-weight GPU oracle outputs in golden-metal/f32"]
fn official_reference_and_ragged_batch_parity() {
    let root = std::path::PathBuf::from(std::env::var("CLEF_DIR").expect("CLEF_DIR"));
    let mut model = Clef::load(&root, None).unwrap();
    // read at run time, not embedded, so a tree without the bench fixtures
    // still builds (the test is ignored and names its inputs anyway)
    let fixtures_path = std::env::var_os("CLEF_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/bench/clef_fixtures.jsonl")
        });
    let fixtures = std::fs::read_to_string(&fixtures_path).unwrap_or_else(|e| {
        panic!(
            "the Clef fixtures at {}: {e} (set CLEF_FIXTURES)",
            fixtures_path.display()
        )
    });
    let fixtures = fixtures.as_str();
    let mut packed_ids = Vec::new();
    let mut packed_questions = Vec::new();
    let mut packed_expected = Vec::new();
    for line in fixtures.lines().filter(|l| !l.trim().is_empty()) {
        let f: serde_json::Value = serde_json::from_str(line).unwrap();
        let id = f["id"].as_str().unwrap();
        let gold: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("golden-metal/f32/{id}.json")))
                .expect("complete GPU oracle fixture"),
        )
        .unwrap();
        let ids = gold["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let questions = gold["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let span = |x: &serde_json::Value| {
                    (
                        x[0].as_u64().unwrap() as usize,
                        x[1].as_u64().unwrap() as usize,
                    )
                };
                ClefQuestion {
                    qtype: v["type"].as_u64().unwrap() as u32,
                    span: span(&v["span"]),
                    options: v["option_spans"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(span)
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let req = ClefRequest {
            ids: &ids,
            questions: &questions,
            images: &[],
        };
        let start = std::time::Instant::now();
        let actual = model.forward(std::slice::from_ref(&req)).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.;
        forward::UNBLOCKED_FOR_TEST.with(|v| v.set(true));
        let baseline = model.forward(std::slice::from_ref(&req));
        forward::UNBLOCKED_FOR_TEST.with(|v| v.set(false));
        assert_eq!(
            actual,
            baseline.unwrap(),
            "{id}: blocked projection changed logits"
        );
        let mut logit_error = 0f64;
        let mut prob_error = 0f64;
        let mut flips = 0;
        assert_eq!(actual[0].len(), gold["questions"].as_array().unwrap().len());
        for (a, b) in actual[0].iter().zip(gold["questions"].as_array().unwrap()) {
            let logits = b["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();
            assert_eq!(a.len(), logits.len());
            logit_error = logit_error.max(
                a.iter()
                    .zip(&logits)
                    .map(|(x, y)| f64::from((x - y).abs()))
                    .fold(0., f64::max),
            );
            prob_error = prob_error.max(
                probability(a)
                    .iter()
                    .zip(probability(&logits))
                    .map(|(x, y)| (x - y).abs())
                    .fold(0., f64::max),
            );
            let argmax = |x: &[f32]| {
                x.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            flips += usize::from(argmax(a) != argmax(&logits));
        }
        eprintln!(
            "{}",
            serde_json::json!({"id":id,"rows":ids.len(),"ms":ms,"max_logit_error":logit_error,"max_probability_error":prob_error,"flips":flips})
        );
        assert!(
            prob_error < 0.00005 && logit_error < 0.0002 && flips == 0,
            "{id}: probability={prob_error}, logits={logit_error}, flips={flips}"
        );
        // Isolate backbone error against the exact same reference.
        let plan = plan::Plan::new(std::slice::from_ref(&req), model.config.vocab).unwrap();
        model.ws.metadata.write(&plan);
        let cmd = model.device.begin().unwrap();
        model.backbone(&cmd, &plan, &[]);
        cmd.finish().unwrap();
        let hidden = unsafe { model.ws.norm.read_f32(0, ids.len() * model.config.hidden) };
        let reference = read_f32(&root.join(format!("golden-metal/f32/{id}.hidden.f32")));
        assert_eq!(hidden.len(), reference.len());
        let relative = (hidden
            .iter()
            .zip(&reference)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
            .sum::<f64>()
            / reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>())
        .sqrt();
        eprintln!("{id}: hidden relative RMS {relative:.8}");
        assert!(relative < 0.0005, "{id}: hidden {relative}");
        if packed_ids.iter().map(Vec::len).sum::<usize>() + ids.len() <= MAX_ROWS {
            packed_ids.push(ids);
            packed_questions.push(questions);
            packed_expected.push(actual[0].clone());
        }
    }
    let reqs = packed_ids
        .iter()
        .zip(&packed_questions)
        .map(|(ids, questions)| ClefRequest {
            ids,
            questions,
            images: &[],
        })
        .collect::<Vec<_>>();
    let actual = model.forward(&reqs).unwrap();
    assert_eq!(
        actual, packed_expected,
        "packed logits must be bit-identical"
    );
}
