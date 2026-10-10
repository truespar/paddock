use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};

fn pick(row: &[f32]) -> u32 {
    assert!(row.iter().all(|v| v.is_finite()));
    row.iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0 as u32
}

fn image_prompt(color: [u8; 3], side: usize) -> Vec<MmChunk> {
    vec![
        MmChunk::Text(vec![100; 50]),
        MmChunk::Image {
            rgb: (0..side * side).flat_map(|_| color).collect(),
            w: side,
            h: side,
        },
        MmChunk::Text(vec![200; 50]),
    ]
}

#[test]
#[ignore = "requires PADDOCK_SPLASH_MODEL; native bundled vision and DFlash2"]
fn splash_bundled_vision_changes_logits_and_survives_replay_and_speculation() {
    let path = std::env::var("PADDOCK_SPLASH_MODEL").unwrap();
    let mut m = Qwen35::load(Path::new(&path), 2048, 2, None).unwrap();
    m.attach_vision(Path::new(&path)).unwrap();
    m.attach_dflash(Path::new(&path)).unwrap();
    println!(
        "Splash resident weights={} kv={} total={:?}",
        m.weight_bytes,
        m.kv_bytes,
        m.device_mem_used()
    );
    let red = image_prompt([255, 0, 0], 256);
    let blue = image_prompt([0, 0, 255], 288);
    let (a, n) = m.prefill_images(0, &red).unwrap();
    let first = pick(&a);
    let expected = m.execute(&[(0, first, n as u32)], &[0]).unwrap();
    m.reset();
    let (b, _) = m.prefill_images(1, &blue).unwrap();
    pick(&b);
    assert_ne!(a, b, "vision must affect target logits");
    m.reset();
    let (again, replayed) = m.prefill_images(0, &red).unwrap();
    assert_eq!(n, replayed);
    assert!(
        a == again,
        "image replay changed logits; max error {}",
        a.iter()
            .zip(&again)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
    let draft = m.spec_draft_batch(&[(0, first)], 7).unwrap().unwrap();
    let chunk = std::iter::once(first)
        .chain(draft[0].iter().copied())
        .collect();
    let verified = m.forward_spec_verify(&[(0, n, chunk)]).unwrap().unwrap();
    assert!(
        verified[..m.vocab] == expected,
        "image-conditioned verification changed target"
    );
    m.spec_commit(&[1]).unwrap();
    m.reset();
    let (cancelled, _) = m.prefill_images(0, &red).unwrap();
    assert!(
        cancelled == a,
        "reset after speculative image prefill changed logits"
    );
}

/// Reference-produced GPU embeddings enter only this ignored test. There is
/// no replay switch, oracle, or alternate vision provider in the runner.
#[test]
#[ignore = "canonical greedy-parity.py --vision-backbone-debug diagnostic"]
fn vision_backbone_boundary_capture() {
    let Some(fixture) = std::env::var_os("PADDOCK_METAL_BOUNDARY_FIXTURE") else {
        return; // ordinary all-ignored model tests need no external capture
    };
    let f: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    let ids = |key: &str| -> Vec<u32> {
        f[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
            .collect()
    };
    let chunks = vec![
        MmChunk::Text(ids("before")),
        MmChunk::Image {
            rgb: std::fs::read(f["rgb"].as_str().unwrap()).unwrap(),
            w: f["w"].as_u64().unwrap() as usize,
            h: f["h"].as_u64().unwrap() as usize,
        },
        MmChunk::Text(ids("after")),
    ];
    let continuation = ids("continuation");
    assert!(!continuation.is_empty());
    let embedding = std::fs::read(f["reference_embedding"].as_str().unwrap()).unwrap();
    assert!(embedding.len() >= 8);
    let n = u32::from_le_bytes(embedding[..4].try_into().unwrap()) as usize;
    let width = u32::from_le_bytes(embedding[4..8].try_into().unwrap()) as usize;
    assert_eq!(embedding.len(), 8 + n * width * 4);
    assert!(
        embedding[8..]
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
    );
    let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").expect("target");
    let mm = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("vision");
    // Match the four-slot parity runner, including its first admission grant.
    let mut m = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    m.attach_vision(Path::new(&mm)).unwrap();
    assert_eq!(width, m.width);
    let mut capture = serde_json::Map::new();
    for (reference, serial) in [(false, false), (true, false), (false, true), (true, true)] {
        m.diagnostic_serial_prefill = serial;
        m.reset();
        while m.evict_checkpoint() {}
        for (_, r) in m.admit_images(vec![(0, chunks.clone())]) {
            assert!(!matches!(r, MmAdmit::Failed(_)));
        }
        while m.encoding_pending() {
            for (_, r) in m.step_images() {
                assert!(!matches!(r, MmAdmit::Failed(_)));
            }
        }
        let layout = m.slots[0].mm.as_mut().unwrap();
        assert_eq!(layout.images.len(), 1);
        let output = &mut layout.images[0];
        assert_eq!(output.nx * output.ny, n);
        if reference {
            // Replace the request-owned plane, not the encoder cache. All
            // subsequent math is the same native Metal language backbone.
            output.embd = m.device.upload(&embedding[8..]).unwrap();
        }
        let mut logits = loop {
            let (_, done) = m.forward_mixed(&[], CHUNK).unwrap();
            if let Some((_, out, _)) = done.into_iter().next() {
                break out;
            }
        };
        let mut rows = Vec::new();
        for (i, &token) in continuation.iter().enumerate() {
            assert!(logits.iter().all(|v| v.is_finite()));
            let mut order = (0..logits.len()).collect::<Vec<_>>();
            order.select_nth_unstable_by(5, |&a, &b| {
                logits[b].total_cmp(&logits[a]).then(a.cmp(&b))
            });
            order[..5].sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            rows.push(
                order[..5]
                    .iter()
                    .map(|&j| (j, logits[j]))
                    .collect::<Vec<_>>(),
            );
            if i + 1 < continuation.len() {
                logits = m.forward(token).unwrap();
            }
        }
        capture.insert(
            if serial && reference {
                "reference_embeddings_serial"
            } else if serial {
                "native_embeddings_serial"
            } else if reference {
                "reference_embeddings"
            } else {
                "native_embeddings"
            }
            .into(),
            serde_json::json!(rows),
        );
    }
    eprintln!("VISION_BACKBONE {}", serde_json::Value::Object(capture));
}

#[test]
#[ignore = "requires real Qwen/mmproj/drafter paths and an M5"]
fn causal_image_chunks_preserve_logits_with_decode_spec_and_cancel_interference() {
    causal_image_lifecycle(&["off", "mtp", "dflash"]);
}

#[test]
#[ignore = "requires an elected Qwen target with in-file MTP and its own vision tower"]
fn causal_image_lifecycle_with_infile_mtp_without_external_drafter() {
    // Qwen 3.6's catalog has in-file nextn, but no elected DFlash companion.
    // Qualify that composition with the same lifecycle assertions, without
    // borrowing another checkpoint's drafter or relaxing any comparison.
    causal_image_lifecycle(&["off", "mtp"]);
}

#[test]
#[ignore = "requires PTQ1 target and BF16 tower in PADDOCK_METAL_QWEN_MODEL / PADDOCK_METAL_MMPROJ"]
fn ptq1_causal_image_lifecycle_without_speculation() {
    causal_image_lifecycle(&["off"]);
}

#[test]
#[ignore = "requires LightOn GGUF/tower or MLX directory in PADDOCK_METAL_QWEN_MODEL / PADDOCK_METAL_MMPROJ"]
fn lighton_tied_head_and_causal_image_lifecycle() {
    let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").expect("LightOn target");
    let m = Qwen35::load(Path::new(&path), 2048, 4, None).unwrap();
    assert!(matches!(
        m.geometry,
        Geometry::DENSE_08B | Geometry::DENSE_4B
    ));
    assert!(
        m.head.is_none(),
        "tied checkpoint must not allocate a second vocabulary plane"
    );
    assert!(std::ptr::eq(m.head(), &m.embedding));
    drop(m);
    causal_image_lifecycle(&["off"]);
}

#[test]
#[ignore = "requires LightOn MLX in PADDOCK_METAL_QWEN_MODEL / PADDOCK_METAL_MMPROJ"]
fn lighton_mixed_image_followers_do_not_reapply_handoff_cap() {
    let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").unwrap();
    let tower = std::env::var_os("PADDOCK_METAL_MMPROJ").unwrap();
    let mut m = Qwen35::load(Path::new(&path), 2048, 4, None).unwrap();
    m.attach_vision(Path::new(&tower)).unwrap();
    let mut rider = pick(&m.prefill(3, &[100; 64]).unwrap());
    let requests = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        .into_iter()
        .enumerate()
        .map(|(slot, color)| (slot, image_prompt(color, 1024)))
        .collect();
    assert_eq!(m.admit_images(requests).len(), 3);
    while m.encoding_pending() {
        m.step_images();
    }
    assert_eq!(m.pending.len(), 3);
    // Identical leading text may already be restored from the rider's cache.
    let follower_offsets = [m.pending[1].offset, m.pending[2].offset];
    for expected in [7, 127, 127] {
        let before = m.pending.front().unwrap().offset;
        let pos = m.slots[3].history.len() as u32;
        let (logits, done) = m.forward_mixed(&[(3, rider, pos)], 512).unwrap();
        rider = pick(&logits);
        assert!(done.is_empty());
        assert_eq!(m.pending.front().unwrap().offset - before, expected);
        assert_eq!(
            m.pending[1].offset, follower_offsets[0],
            "follower must not pin the head to seven rows"
        );
        assert_eq!(m.pending[2].offset, follower_offsets[1]);
    }
    assert!(m.prefill_abort(1));
    let before = m.pending.front().unwrap().offset;
    let pos = m.slots[3].history.len() as u32;
    let (_, done) = m.forward_mixed(&[(3, rider, pos)], 512).unwrap();
    assert!(done.is_empty());
    assert_eq!(m.pending.front().unwrap().offset - before, 127);
    assert_eq!(m.pending[1].slot, 2);
    assert_eq!(m.pending[1].offset, follower_offsets[1]);
    assert!(m.cache.iter().all(|c| !c.reserved));
}

fn causal_image_lifecycle(modes: &[&str]) {
    let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").expect("target");
    let mm = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("vision");
    for &mode in modes {
        let mut m = Qwen35::load(Path::new(&path), 2048, 4, None).unwrap();
        m.attach_vision(Path::new(&mm)).unwrap();
        eprintln!(
            "resident model weights={} KV/state={} device={:?}",
            m.weight_bytes,
            m.kv_bytes,
            m.device_mem_used()
        );
        match mode {
            "mtp" => m.attach_mtp(Path::new(&path)).unwrap(),
            "dflash" => m
                .attach_dflash(Path::new(
                    &std::env::var_os("PADDOCK_METAL_DFLASH_MODEL").expect("drafter"),
                ))
                .unwrap(),
            _ => (),
        }
        let mut prompt = image_prompt([255, 0, 0], 1024);
        prompt[0] = MmChunk::Text(vec![100; 64]);
        prompt[2] = MmChunk::Text(vec![200]);
        let mut captures = Vec::new();
        let mut rider = 0;
        for slot in [0, 1] {
            while m.evict_checkpoint() {}
            rider = pick(&m.prefill(3, &[100; 64]).unwrap());
            m.admit_images(vec![(slot, prompt.clone())]);
            while m.encoding_pending() {
                m.step_images();
            }
            let mm = m.slots[slot].mm.as_ref().unwrap();
            // Equal rotary time must not expose later spatial tokens.
            for row in 64..1088 {
                assert_eq!(m.rope_position(slot, row)[0], 64);
                assert_eq!(mm.limit(row), row as u32);
            }
            let mut ticks = 0;
            loop {
                let before = m.slots[slot].history.len();
                let pos = m.slots[3].history.len() as u32;
                let (out, done) = m.forward_mixed(&[(3, rider, pos)], 512).unwrap();
                rider = pick(&out);
                ticks += 1;
                let after = m.slots[slot].history.len();
                assert!(
                    after > before && after - before <= 127,
                    "bounded, nonstarving image chunk"
                );
                assert!(
                    m.cache.iter().all(|c| !c.reserved),
                    "no half-finished GPU plan"
                );
                if let Some((_, logits, n)) = done.into_iter().next() {
                    assert_eq!(n, 1089);
                    captures.push(logits);
                    break;
                }
                assert!(m.prefill(slot, &[100]).is_err());
                assert!(!m.spec_ensure_warm(slot, &[], after as u32).unwrap());
                // Same image chunk boundaries, but unrelated speculative
                // passes overwrite shared metadata/taps between those chunks.
                if slot == 1 && mode != "off" && ticks % 3 == 0 {
                    let pos = m.slots[3].history.len();
                    let draft = m.spec_draft_batch(&[(3, rider)], 2).unwrap().unwrap();
                    let mut chunk = vec![rider];
                    chunk.extend(&draft[0]);
                    let verified = m.verify(&[(3, pos, chunk)]).unwrap();
                    m.commit_verify(&[1]).unwrap();
                    rider = pick(&verified[..m.vocab]);
                }
                assert!(ticks < 40);
            }
            assert_eq!(ticks, 10, "7-row handoff then 127-row hybrid grants");
        }
        assert!(
            captures[0] == captures[1],
            "{mode}: interleaved image chunks; max error {}",
            captures[0]
                .iter()
                .zip(&captures[1])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        );
        let mut next = pick(&captures[0]);
        for pos in 1089..1105 {
            let a = m.execute(&[(0, next, pos)], &[0]).unwrap();
            let b = m.execute(&[(1, next, pos)], &[0]).unwrap();
            assert_eq!(
                a, b,
                "{mode}: full continuation after chunked image at {pos}"
            );
            next = pick(&a);
        }
        // Cancel inside an image's causal rows, then immediately reuse the
        // slot. No suspended command or partial activation can survive.
        while m.evict_checkpoint() {}
        m.admit_images(vec![(2, prompt)]);
        while m.encoding_pending() {
            m.step_images();
        }
        let pos = m.slots[3].history.len() as u32;
        let (out, _) = m.forward_mixed(&[(3, rider, pos)], 512).unwrap();
        assert_eq!(m.slots[2].history.len(), 7, "small encoder handoff grant");
        let (out, _) = m.forward_mixed(&[(3, pick(&out), pos + 1)], 512).unwrap();
        m.forward_mixed(&[(3, pick(&out), pos + 2)], 512).unwrap();
        assert_eq!(m.slots[2].history.len(), 261, "7 + 127 + 127 causal rows");
        assert!(m.prefill_abort(2));
        assert!(m.pending.is_empty());
        assert!(m.cache.iter().all(|c| !c.reserved));
        pick(&m.prefill(2, &[100; 24]).unwrap());
        eprintln!(
            "{mode}: causal image chunks, exact 16-step continuation, speculation and cancellation pass"
        );
    }
}
#[test]
#[ignore = "requires real Qwen/mmproj/drafter paths and an M5"]
fn multimodal_cache_mixed_slots_cancellation_and_speculation() {
    let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").expect("target");
    let mm = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("vision");
    for mode in ["mtp", "dflash"] {
        let mut m = Qwen35::load(Path::new(&path), 2048, 4, None).unwrap();
        m.attach_vision(Path::new(&mm)).unwrap();
        if mode == "mtp" {
            m.attach_mtp(Path::new(&path)).unwrap();
        } else {
            let df = std::env::var_os("PADDOCK_METAL_DFLASH_MODEL").expect("drafter");
            m.attach_dflash(Path::new(&df)).unwrap();
        }
        assert!(m.spec_draft_kv_space());
        let red = image_prompt([255, 0, 0], 256);
        let blue = image_prompt([0, 0, 255], 288);
        let (a, na) = m.prefill_images(0, &red).unwrap();
        let (b, nb) = m.prefill_images(1, &blue).unwrap();
        assert_eq!((na, nb), (164, 181));
        assert_eq!(m.rope_position(0, na), [108; 4]);
        assert_eq!(m.rope_position(1, nb), [109; 4]);
        let admissions = m.admit_images(vec![(2, red.clone()), (3, blue.clone())]);
        assert!(
            m.prefill_images(0, &red).is_err(),
            "exclusive fallback must not steal queued completions"
        );
        assert!(
            admissions
                .iter()
                .all(|(_, a)| matches!(a, MmAdmit::Encoding))
        );
        while m.encoding_pending() {
            for (_, a) in m.step_images() {
                assert!(matches!(a, MmAdmit::Queued));
            }
        }
        let mut done = Vec::new();
        while !m.pending.is_empty() {
            done.extend(m.forward_mixed(&[], 512).unwrap().1);
        }
        assert_eq!(done.len(), 2);
        assert!(m.image_cache_reuses() >= 2);
        for (s, logits, _) in done {
            assert_eq!(
                pick(&logits),
                pick(if s == 2 { &a } else { &b }),
                "{mode} slot {s}"
            );
        }
        assert!(
            m.take_prefill_reused(2) > 0,
            "exact image prefix should resume"
        );
        // Same token placeholders, different raw image: must not borrow a
        // cached image's KV or recurrent checkpoint.
        let green = image_prompt([0, 255, 0], 256);
        let (g, _) = m.prefill_images(3, &green).unwrap();
        assert_eq!(m.take_prefill_reused(3), 0);
        assert!(a.iter().zip(&g).any(|(a, g)| (a - g).abs() > 0.1));
        // Identical image-conditioned target state: speculative verification
        // followed by a one-row commit vs one ordinary target step.
        let (fresh, _) = m.prefill_images(2, &red).unwrap();
        let next = pick(&fresh);
        let draft = m.spec_draft_batch(&[(2, next)], 3).unwrap().unwrap();
        let mut chunk = vec![next];
        chunk.extend(&draft[0]);
        let verify = m.verify(&[(2, na, chunk)]).unwrap();
        m.commit_verify(&[1]).unwrap();
        m.prefill_images(0, &red).unwrap();
        let plain = m.execute(&[(0, next, na as u32)], &[0]).unwrap();
        assert_eq!(
            pick(&plain),
            pick(&verify[..m.vocab]),
            "{mode} verification after vision"
        );
        let follow = pick(&plain);
        let after = m
            .execute(
                &[(0, follow, na as u32 + 1), (2, follow, na as u32 + 1)],
                &[0, 1],
            )
            .unwrap();
        assert_eq!(
            pick(&after[..m.vocab]),
            pick(&after[m.vocab..]),
            "{mode} rollback after image"
        );
        // A large image crosses the text quantum without inflating the row
        // workspace. MTP / DFlash conditioning survives every causal cut.
        let large = image_prompt([255, 0, 0], 768);
        let r = m.admit_images(vec![(3, large.clone())]);
        assert!(matches!(r[0].1, MmAdmit::Encoding));
        // Deterministically exhaust the encoder allowance. Setup still
        // happens, and an existing job must advance even at zero budget;
        // neither check may depend on how quickly this Mac runs the tower.
        m.last_gpu_seconds = 0.176;
        assert!(m.step_images().is_empty());
        assert!(m.step_images().is_empty());
        assert!(m.encoding_pending());
        m.prefill_abort(3);
        assert!(!m.encoding_pending());
        let (large_logits, n) = m.prefill_images(3, &large).unwrap();
        assert_eq!(n, 676);
        pick(&large_logits);
        assert_eq!(m.row_capacity, CHUNK);
        assert_eq!(m.rope_position(3, n), [124; 4]);
        let invalid = vec![MmChunk::Image {
            rgb: vec![0; 3],
            w: 512,
            h: 512,
        }];
        assert!(matches!(
            m.admit_images(vec![(0, invalid)])[0].1,
            MmAdmit::Failed(_)
        ));
        let text = m.forward_prefill(0, &[100, 200, 300]).unwrap();
        pick(&text);
        assert!(m.slots[0].mm.is_none());
        eprintln!("{mode}: multimodal state/cache/speculation gates passed");
    }
}
