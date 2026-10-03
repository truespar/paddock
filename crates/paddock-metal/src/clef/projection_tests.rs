//! GPU-only projection experiments. Alternated order avoids treating a warmed
//! cache or a later thermal state as a kernel improvement. Not serving knobs.
use super::*;

#[test]
#[ignore = "manual single-fixture GPU hidden diagnostic; CLEF_MODEL, CLEF_CASE, CLEF_HIDDEN"]
fn quantized_hidden_diagnostic() {
    use paddock_engine::clef_decision::ClefQuestion;
    let model = std::path::PathBuf::from(std::env::var("CLEF_MODEL").unwrap());
    let case = std::env::var("CLEF_CASE").unwrap();
    let mut model = Clef::load(&model, None).unwrap();
    let gold: serde_json::Value = serde_json::from_slice(&std::fs::read(case).unwrap()).unwrap();
    let ids = gold["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<_>>();
    let span = |v: &serde_json::Value| {
        (
            v[0].as_u64().unwrap() as usize,
            v[1].as_u64().unwrap() as usize,
        )
    };
    let qs = gold["questions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| ClefQuestion {
            qtype: q["type"].as_u64().unwrap() as u32,
            span: span(&q["span"]),
            options: q["option_spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(span)
                .collect(),
        })
        .collect::<Vec<_>>();
    let count = (model.layers.len() + 6) * ids.len() * model.config.hidden;
    model.trace = Some(model.device.alloc(count * 4).unwrap());
    let logits = model
        .forward(&[ClefRequest {
            ids: &ids,
            questions: &qs,
            images: &[],
        }])
        .unwrap();
    let hidden = unsafe { model.ws.norm.read_f32(0, ids.len() * model.config.hidden) };
    std::fs::write(
        std::env::var("CLEF_HIDDEN").unwrap(),
        hidden
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    eprintln!("logits {logits:?}");
    let trace = unsafe { model.trace.as_ref().unwrap().read_f32(0, count) };
    std::fs::write(
        format!("{}.layers.f32", std::env::var("CLEF_HIDDEN").unwrap()),
        trace
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires CLEF_MODEL and same-checkpoint GPU CLEF_GOLDEN directory"]
fn quantized_checkpoint_reference_and_ragged_batch_parity() {
    use paddock_engine::clef_decision::ClefQuestion;
    let model = std::path::PathBuf::from(std::env::var("CLEF_MODEL").expect("CLEF_MODEL"));
    let golden = std::path::PathBuf::from(std::env::var("CLEF_GOLDEN").expect("CLEF_GOLDEN"));
    let mut model = Clef::load(&model, None).unwrap();
    let mut paths = std::fs::read_dir(golden)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect::<Vec<_>>();
    paths.sort();
    assert!(!paths.is_empty());
    let mut packed = Vec::new();
    let filter = std::env::var("CLEF_CASE_FILTER").ok();
    let mut checked = 0;
    let mut failed = Vec::new();
    for path in paths {
        if filter
            .as_deref()
            .is_some_and(|name| path.file_stem().and_then(|v| v.to_str()) != Some(name))
        {
            continue;
        }
        let gold: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        checked += 1;
        let ids = gold["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let span = |v: &serde_json::Value| {
            (
                v[0].as_u64().unwrap() as usize,
                v[1].as_u64().unwrap() as usize,
            )
        };
        let qs = gold["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| ClefQuestion {
                qtype: q["type"].as_u64().unwrap() as u32,
                span: span(&q["span"]),
                options: q["option_spans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(span)
                    .collect(),
            })
            .collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let actual = model
            .forward(&[ClefRequest {
                ids: &ids,
                questions: &qs,
                images: &[],
            }])
            .unwrap()
            .remove(0);
        let mut error = 0f64;
        let mut prob = 0f64;
        let mut flips = 0;
        assert_eq!(actual.len(), gold["questions"].as_array().unwrap().len());
        for (a, q) in actual.iter().zip(gold["questions"].as_array().unwrap()) {
            let b = q["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>();
            let softmax = |v: &[f64]| {
                let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum = v.iter().map(|v| (v - max).exp()).sum::<f64>();
                v.iter().map(|v| (v - max).exp() / sum).collect::<Vec<_>>()
            };
            let a = a.iter().map(|&v| v as f64).collect::<Vec<_>>();
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            for (x, y) in a.iter().zip(&b) {
                error = error.max((x - y).abs());
            }
            for (x, y) in softmax(&a).iter().zip(softmax(&b)) {
                prob = prob.max((x - y).abs());
            }
            let argmax = |v: &[f64]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            flips += usize::from(argmax(&a) != argmax(&b));
        }
        eprintln!(
            "{}",
            serde_json::json!({"case":gold["id"],"rows":ids.len(),"ms":start.elapsed().as_secs_f64()*1000.,"max_logit_error":error,"max_probability_error":prob,"flips":flips,"weight_bytes":model.info().weight_bytes,"workspace_bytes":model.info().workspace_bytes})
        );
        // Finish every case and batch isolation before failing: one long case
        // must not conceal missing coverage of the subsequent fixtures.
        if error >= 0.0002 || prob >= 0.00005 || flips != 0 {
            failed.push(format!(
                "{}: logit {error}, probability {prob}, flips {flips}",
                path.display()
            ));
        }
        if packed.len() < 4 && ids.len() < 1500 {
            packed.push((ids, qs, actual));
        }
    }
    assert!(checked > 0, "case filter matched no golden fixture");
    if filter.is_some() && packed.is_empty() {
        assert!(failed.is_empty(), "{}", failed.join("\n"));
        return;
    }
    let requests = packed
        .iter()
        .map(|(ids, questions, _)| ClefRequest {
            ids,
            questions,
            images: &[],
        })
        .collect::<Vec<_>>();
    let actual = model.forward(&requests).unwrap();
    assert_eq!(actual.len(), packed.len());
    for (a, (_, _, expected)) in actual.iter().zip(&packed) {
        assert_eq!(a, expected, "ragged packed decision differs");
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn packed_q8_and_affine8_match_gpu_reference_on_tails_and_epilogues() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    for (k, n, m) in [(64usize, 66usize, 3usize), (128, 130, 65)] {
        // Powers-of-two scales make these weights exactly BF16 representable.
        // The reference is another GPU projection, never CPU inference.
        let signed = (0..n * k)
            .map(|i| ((i % 127) as i16 - 63) as i8)
            .collect::<Vec<_>>();
        let bf = signed
            .iter()
            .flat_map(|&q| half::bf16::from_f32(q as f32 / 32.).to_le_bytes())
            .collect::<Vec<_>>();
        let mut q8 = Vec::new();
        for block in signed.chunks_exact(32) {
            q8.extend(half::f16::from_f32(1. / 32.).to_le_bytes());
            q8.extend(block.iter().map(|&q| q as u8));
        }
        let side = |value: f32| {
            d.upload(
                &(0..n * k / 64)
                    .flat_map(|_| half::bf16::from_f32(value).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let planes = [
            quant::Plane {
                data: d.upload(&q8).unwrap(),
                kind: 1,
                affine: None,
            },
            quant::Plane {
                data: d
                    .upload(
                        &signed
                            .iter()
                            .map(|&q| (q as i16 + 128) as u8)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                kind: 2,
                affine: Some((side(1. / 32.), side(-4.))),
            },
        ];
        let x = upload(
            &d,
            &(0..m * k)
                .map(|i| ((i % 29) as f32 - 14.) / 16.)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let parts = d.alloc(m * k * 4).unwrap();
        let c = d.begin().unwrap();
        point(&c, "clef_prepare", &[&x, &parts], &[(m * k) as u32], m * k);
        c.finish().unwrap();
        let reference = Linear {
            weight: d.upload(&bf).unwrap().into(),
            bias: None,
            k,
            n,
        };
        for plane in planes {
            let candidate = Linear {
                weight: plane,
                bias: None,
                k,
                n,
            };
            for epi in [0, 1, 2, 3] {
                let expected = upload(&d, &vec![0.25; m * n]).unwrap();
                let actual = upload(&d, &vec![0.25; m * n]).unwrap();
                let c = d.begin().unwrap();
                reference.run(&c, &x, &expected, m, epi);
                candidate.run(&c, &x, &actual, m, epi);
                c.finish().unwrap();
                let count = if epi == 3 { m * n / 2 } else { m * n };
                assert_eq!(
                    unsafe { actual.read_f32(0, count) },
                    unsafe { expected.read_f32(0, count) },
                    "kind {}, epi {epi}",
                    candidate.weight.kind
                );
            }
            for epi in [0, 1, 3] {
                let expected = upload(&d, &vec![0.25; m * n]).unwrap();
                let actual = upload(&d, &vec![0.25; m * n]).unwrap();
                let c = d.begin().unwrap();
                reference.parts(&c, &parts, &expected, m, epi);
                candidate.parts(&c, &parts, &actual, m, epi);
                c.finish().unwrap();
                let count = if epi == 3 { m * n / 2 } else { m * n };
                assert_eq!(
                    unsafe { actual.read_u32(count) },
                    unsafe { expected.read_u32(count) },
                    "parts kind {}, epi {epi}",
                    candidate.weight.kind
                );
            }
            let ids = d
                .upload(
                    &[1u32, 0, 2]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let spans = d
                .upload(
                    &[0u32, 2]
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            for spans in [None, Some(&spans)] {
                let rows = if spans.is_some() { 1 } else { 3 };
                let a = d.alloc(rows * k * 4).unwrap();
                let b = d.alloc(rows * k * 4).unwrap();
                let c = d.begin().unwrap();
                reference.weight.gather(&c, &ids, spans, &a, k, rows);
                candidate.weight.gather(&c, &ids, spans, &b, k, rows);
                c.finish().unwrap();
                assert_eq!(unsafe { a.read_u32(rows * k) }, unsafe {
                    b.read_u32(rows * k)
                });
            }
        }
    }
}

#[test]
fn blocked_projections_preserve_tails_and_epilogues() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    // Incomplete K stripes, partial output tiles, and both sides of an M tile.
    // Compare raw words: the fused epilogue stores two BF16 planes, not F32.
    for (k, n, m) in [
        (130, 66, 65),
        (1025, 130, 3),
        (257, 64, 33),
        (12288, 64, 65),
    ] {
        let w = d
            .upload(
                &(0..k * n)
                    .flat_map(|i| {
                        ((0x3b00 + (i % 512) as u16) | ((((i / 31) % 2) as u16) << 15))
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x = upload(
            &d,
            &(0..m * k)
                .map(|i| ((i % 199) as f32 - 99.) / 127.)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let parts = d.alloc(m * k * 4).unwrap();
        let c = d.begin().unwrap();
        point(&c, "clef_prepare", &[&x, &parts], &[(m * k) as u32], m * k);
        c.finish().unwrap();
        for epi in [0, 1, 3] {
            let mut reference = None;
            for kernel in ["clef_mm_parts", "clef_mm_parts_k256", "clef_mm_parts_k1024"] {
                let y = upload(&d, &vec![0.25; m * n]).unwrap();
                let c = d.begin().unwrap();
                c.dispatch(
                    kernel,
                    &[&w, &parts, &y],
                    &[k as u32, n as u32, m as u32, epi],
                    [n.div_ceil(64), m.div_ceil(64), 1],
                    128,
                );
                c.finish().unwrap();
                let values = unsafe { y.read_u32(if epi == 3 { m * n / 2 } else { m * n }) };
                if let Some(expected) = &reference {
                    assert_eq!(&values, expected, "{kernel} K{k} N{n} M{m} epi{epi}");
                } else {
                    reference = Some(values);
                }
            }
        }
    }
}

#[test]
#[ignore = "manual alternating full-model timing; requires CLEF_DIR"]
fn full_pass_blocking_cost() {
    use paddock_engine::clef_decision::ClefQuestion;
    let root = std::path::PathBuf::from(std::env::var("CLEF_DIR").expect("CLEF_DIR"));
    let mut model = Clef::load(&root, None).unwrap();
    for (case, batch) in [
        ("readme_invoice", 1),
        ("many_questions", 1),
        ("readme_invoice", 4),
        ("many_questions", 4),
        ("long_contract", 1),
    ] {
        let gold: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join(format!("golden-metal/f32/{case}.json"))).unwrap(),
        )
        .unwrap();
        let ids = gold["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let questions = gold["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let span = |v: &serde_json::Value| {
                    (
                        v[0].as_u64().unwrap() as usize,
                        v[1].as_u64().unwrap() as usize,
                    )
                };
                ClefQuestion {
                    qtype: q["type"].as_u64().unwrap() as u32,
                    span: span(&q["span"]),
                    options: q["option_spans"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(span)
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let requests = (0..batch)
            .map(|_| ClefRequest {
                ids: &ids,
                questions: &questions,
                images: &[],
            })
            .collect::<Vec<_>>();
        let expected = model.forward(&requests).unwrap();
        let mut times = [Vec::new(), Vec::new()];
        let mut thermal = [Vec::new(), Vec::new()];
        for round in 0..5 {
            for offset in 0..2 {
                let index = (round + offset) % 2;
                forward::UNBLOCKED_FOR_TEST.with(|v| v.set(index == 0));
                let before = objc2_foundation::NSProcessInfo::processInfo()
                    .thermalState()
                    .0;
                let start = std::time::Instant::now();
                let result = model.forward(&requests);
                let ms = start.elapsed().as_secs_f64() * 1000.;
                forward::UNBLOCKED_FOR_TEST.with(|v| v.set(false));
                assert_eq!(expected, result.unwrap(), "{case}, batch={batch}");
                if round > 0 {
                    times[index].push(ms);
                    thermal[index].push((
                        before,
                        objc2_foundation::NSProcessInfo::processInfo()
                            .thermalState()
                            .0,
                    ));
                }
            }
        }
        eprintln!(
            "{}",
            serde_json::json!({"case":case,"batch":batch,"baseline_ms":times[0],"blocked_ms":times[1],
                "baseline_thermal":thermal[0],"blocked_thermal":thermal[1]})
        );
    }
}

#[test]
#[ignore = "manual projection timing; run without other GPU workloads"]
fn projection_cost_and_reduction_contract() {
    let d = MetalDevice::new(Some(1 << 30)).unwrap();
    let kernels = ["clef_mm_parts", "clef_mm_parts_k256", "clef_mm_parts_k1024"];
    for (k, n, m) in [
        (4096, 24576, 4368),
        (12288, 4096, 4368),
        (4096, 8192, 4368),
        (4096, 24576, 1092),
        (4096, 24576, 260),
    ] {
        let w = d
            .upload(
                &(0..k * n)
                    .flat_map(|i| {
                        // Signed deterministic BF16 inputs. Both matmuls run on GPU.
                        let bits = (0x3b00 + (i % 512) as u16) | ((((i / 31) % 2) as u16) << 15);
                        bits.to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let x = upload(
            &d,
            &(0..m * k)
                .map(|i| ((i % 199) as f32 - 99.) / 127.)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let parts = d.alloc(m * k * 4).unwrap();
        let y = d.alloc(m * n * 4).unwrap();
        let cmd = d.begin().unwrap();
        point(
            &cmd,
            "clef_prepare",
            &[&x, &parts],
            &[(m * k) as u32],
            m * k,
        );
        cmd.finish().unwrap();
        let dispatch = |kernel| {
            let c = d.begin().unwrap();
            c.dispatch(
                kernel,
                &[&w, &parts, &y],
                &[k as u32, n as u32, m as u32, 0],
                [n.div_ceil(64), m.div_ceil(64), 1],
                128,
            );
            c.finish().unwrap();
        };
        dispatch(kernels[0]);
        let reference = unsafe { y.read_f32(0, m * n) };
        let mut timing = vec![Vec::new(); kernels.len()];
        for round in 0..6 {
            for offset in 0..kernels.len() {
                let index = (offset + round) % kernels.len();
                let start = std::time::Instant::now();
                dispatch(kernels[index]);
                timing[index].push(start.elapsed().as_secs_f64() * 1000.);
                if round == 0 {
                    let actual = unsafe { y.read_f32(0, m * n) };
                    let error = actual
                        .iter()
                        .zip(&reference)
                        .map(|(a, b)| f64::from(a - b).powi(2))
                        .sum::<f64>();
                    let energy = reference.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
                    let relative = (error / energy).sqrt();
                    eprintln!(
                        "{} k{k} n{n} m{m}: relative {relative}, exact {}",
                        kernels[index],
                        actual == reference
                    );
                    assert!(
                        actual == reference,
                        "{} changed the reduction",
                        kernels[index]
                    );
                }
            }
        }
        for (kernel, ms) in kernels.iter().zip(timing) {
            eprintln!(
                "{}",
                serde_json::json!({"kernel":kernel,"k":k,"n":n,"m":m,"ms":ms})
            );
        }
    }
}
