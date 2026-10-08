use super::*;
use paddock_engine::{encoder::embedding_gemma2::*, service::MmChunk};

#[test]
#[ignore = "real MLX checkpoint; independently loadable bundled tower accounting"]
fn embeddinggemma2_mlx_catalog_memory() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let mut text_bytes = None;
    for (image, audio) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut model = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
        assert!(model.mlx);
        assert_eq!(
            *text_bytes.get_or_insert(model.weight_bytes),
            model.weight_bytes
        );
        if image || audio {
            model
                .attach_mlx_media(Path::new(&path), 280, image, audio)
                .unwrap();
        }
        assert_eq!(model.media_kinds(), (image, audio, image));
        assert_eq!(model.device.allocated_bytes(), model.weight_bytes);
        eprintln!(
            "bundled image={image} audio={audio}: text={} resident={}",
            text_bytes.unwrap(),
            model.weight_bytes
        );
        let pending = model.embed_submit(&[vec![2, 1000, 1]], 0).unwrap();
        let output = model.embed_collect(&pending).unwrap();
        assert!(output[0].iter().all(|x| x.is_finite()));
        drop(pending);
        model.reclaim_idle();
        assert_eq!(model.device.allocated_bytes(), model.weight_bytes);
    }
}

fn f32_file(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "real audio weights and PCM fixtures; packed causal-domain regression"]
fn embeddinggemma2_audio_packed_domains() {
    let d = MetalDevice::new(None).unwrap();
    let map = paddock_models::mapped::MappedGguf::open(Path::new(
        &std::env::var("PADDOCK_EG2_AUDIO").unwrap(),
    ))
    .unwrap();
    let tower = audio::Audio::load(&d, &map).unwrap();
    let root = std::env::var("PADDOCK_EG2_MEDIA_PREPARED").unwrap();
    let samples: Vec<_> = ["speech30", "tiny-0.3s", "chirp", "speech6"]
        .iter()
        .map(|n| f32_file(&Path::new(&root).join(format!("audio/{n}.pcm.f32"))))
        .collect();
    let refs: Vec<_> = samples.iter().map(Vec::as_slice).collect();
    let packed = tower.encode_batch(&d, &refs).unwrap();
    for (clip, out) in refs.iter().zip(packed) {
        let single = tower.encode_batch(&d, &[clip]).unwrap();
        let n = audio_tokens(clip.len()).unwrap() * WIDTH;
        let a = unsafe { out.read_f32(0, n) };
        let b = unsafe { single[0].read_f32(0, n) };
        assert_eq!(a, b, "another clip changed this clip's feature rows");
    }
    let start = std::time::Instant::now();
    for _ in 0..4 {
        tower.encode_batch(&d, &[refs[0]]).unwrap();
    }
    let serial = start.elapsed();
    let start = std::time::Instant::now();
    tower.encode_batch(&d, &[refs[0]; 4]).unwrap();
    eprintln!(
        "four 30-second clips: serial {:.2} ms, packed {:.2} ms",
        serial.as_secs_f64() * 1000.,
        start.elapsed().as_secs_f64() * 1000.
    );
}

#[test]
#[ignore = "real media weights; isolated projection tile election"]
fn embeddinggemma2_media_projection_tiles() {
    use paddock_models::mapped::MappedGguf;
    let d = MetalDevice::new(None).unwrap();
    for (env, name, k, n, ty, rows) in [
        (
            "PADDOCK_EG2_VISION",
            "v.blk.0.ffn_up.weight",
            768,
            3072,
            30,
            2394,
        ),
        (
            "PADDOCK_EG2_VISION",
            "v.blk.0.ffn_down.weight",
            3072,
            768,
            30,
            2394,
        ),
        (
            "PADDOCK_EG2_AUDIO",
            "a.blk.0.ffn_up.weight",
            1024,
            4096,
            1,
            750,
        ),
        (
            "PADDOCK_EG2_AUDIO",
            "a.blk.0.ffn_down.weight",
            4096,
            1024,
            1,
            750,
        ),
        (
            "PADDOCK_EG2_AUDIO",
            "a.blk.0.ffn_up.weight",
            1024,
            4096,
            1,
            3000,
        ),
    ] {
        let m = MappedGguf::open(Path::new(&std::env::var(env).unwrap())).unwrap();
        let w = vision::weight(&d, &m, name, &[k, n], ty).unwrap();
        let x = d
            .upload(
                &vec![
                    if ty == 30 {
                        1f32.to_le_bytes().to_vec()
                    } else {
                        0x3c00u16.to_le_bytes().to_vec()
                    };
                    rows * k
                ]
                .concat(),
            )
            .unwrap();
        let y = d.alloc(rows * n * 4).unwrap();
        let candidates = if ty == 30 {
            [
                ("vis_bmm_fast64", 64),
                ("vis_bmm_fast32", 32),
                ("eg2v_bmm128", 128),
            ]
        } else {
            [("vis_mm64", 64), ("vis_mm32", 32), ("eg2a_mm128", 128)]
        };
        let mut gold = Vec::new();
        for (kernel, tile) in candidates {
            let mut times = Vec::new();
            for _ in 0..6 {
                let c = d.begin().unwrap();
                for _ in 0..16 {
                    c.dispatch(
                        kernel,
                        &[&w.buffer, &x, &y, &w.buffer],
                        &[k as u32, n as u32, rows as u32, 0],
                        [n.div_ceil(64), rows.div_ceil(tile), 1],
                        128,
                    );
                }
                times.push(c.finish().unwrap() * 1000. / 16.);
            }
            let got = unsafe { y.read_f32(0, rows * n) };
            if gold.is_empty() {
                gold = got;
            } else {
                let max = gold
                    .iter()
                    .zip(got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(max < 0.0001, "{kernel}: {max}");
            }
            times.remove(0);
            times.sort_by(f64::total_cmp);
            eprintln!("{name} M={rows} {kernel}: median {:.4} ms", times[2]);
        }
    }
}

#[test]
#[ignore = "same-GGUF GPU media reference plus HF processor fixtures"]
fn embeddinggemma2_media_reference() {
    let file = std::env::var("PADDOCK_EG2_MEDIA_REFERENCE").unwrap();
    let prep = std::env::var("PADDOCK_EG2_MEDIA_PREPARED").unwrap();
    let prep = Path::new(&prep);
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let mut m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    let budget = data["budget"].as_u64().unwrap() as usize;
    if m.mlx {
        m.attach_mlx_media(Path::new(&path), budget, true, true)
            .unwrap();
    } else {
        for variable in ["PADDOCK_EG2_VISION", "PADDOCK_EG2_AUDIO"] {
            m.attach_mmproj(Path::new(&std::env::var(variable).unwrap()), budget, true)
                .unwrap();
        }
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(prep.join("manifest.json")).unwrap()).unwrap();
    let audio = m.audio.as_mut().unwrap();
    let mut failures = Vec::new();
    if let Some(clips) = data["clips"].as_object() {
        for name in clips.keys() {
            let samples = f32_file(&prep.join(format!("audio/{name}.pcm.f32")));
            let expected = f32_file(&prep.join(format!("audio/{name}.hf-mel.f32")));
            let got = audio.log_mel(&m.device, &samples).unwrap();
            assert_eq!(got.len(), expected.len());
            let err = got
                .iter()
                .zip(&expected)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            let rms = (got
                .iter()
                .zip(&expected)
                .map(|(x, y)| f64::from(x - y).powi(2))
                .sum::<f64>()
                / got.len() as f64)
                .sqrt();
            eprintln!("frontend {name}: max={err:.8}, rms={rms:.8}");
            if err > 0.00001 || rms > 0.000002 {
                failures.push(format!("frontend {name}"));
            }
        }
    }
    // The shipped path follows HF's log(mel + .001). Isolate the tower
    // against llama.cpp by matching its known, different log(max(mel,.001)).
    audio.llama_floor = !m.mlx;
    for case in data["cases"].as_array().unwrap() {
        let seqs: Vec<Vec<u32>> = serde_json::from_value(case["ids"].clone()).unwrap();
        let mut media = Vec::new();
        for item in case["items"].as_array().unwrap() {
            let mut inputs = Vec::new();
            for part in item.as_array().into_iter().flatten() {
                if let Some(name) = part["image"].as_str() {
                    let shape = &manifest["images"][name]["size"];
                    let stem = Path::new(name).file_stem().unwrap().to_str().unwrap();
                    inputs.push(MmChunk::Image {
                        rgb: std::fs::read(prep.join(format!("{stem}.rgb"))).unwrap(),
                        w: shape[0].as_u64().unwrap() as usize,
                        h: shape[1].as_u64().unwrap() as usize,
                    });
                } else if let Some(name) = part["audio"].as_str() {
                    inputs.push(MmChunk::Audio {
                        samples: f32_file(&prep.join(format!("audio/{name}.pcm.f32"))),
                        mel: None,
                    });
                }
            }
            media.push(inputs);
        }
        let expected: Vec<Vec<f32>> = serde_json::from_value(case["embeddings"].clone()).unwrap();
        let started = std::time::Instant::now();
        if let Some(expected) = case.get("features") {
            let expected: Vec<Vec<f32>> = serde_json::from_value(expected.clone()).unwrap();
            assert!(!expected.is_empty());
            assert!(expected.iter().all(|row| row.len() == WIDTH));
            let inputs = m.encode_media(&seqs, &media).unwrap();
            assert_eq!(inputs.len(), 1, "tower fixtures must isolate one medium");
            assert_eq!(inputs[0].run.len, expected.len(), "tower row count changed");
            let got = unsafe { inputs[0].data.read_f32(0, expected.len() * WIDTH) };
            let ref_flat: Vec<_> = expected.into_iter().flatten().collect();
            assert!(got.iter().chain(&ref_flat).all(|v| v.is_finite()));
            let dot: f64 = got
                .iter()
                .zip(&ref_flat)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(&got) * norm(&ref_flat));
            eprintln!("{} tower cosine={cos:.9}", case["name"]);
            if !cos.is_finite() || cos <= 0.9995 {
                failures.push(format!("{} tower: {cos}", case["name"]));
            }
        }
        let p = m.embed_submit_media(&seqs, &media, 0, None).unwrap();
        let got = m.embed_collect(&p).unwrap();
        assert_eq!(got.len(), seqs.len());
        assert_eq!(got.len(), expected.len());
        for (i, (g, r)) in got.iter().zip(&expected).enumerate() {
            assert_eq!((g.len(), r.len()), (DIM, DIM));
            assert!(g.iter().chain(r).all(|v| v.is_finite()));
            let dot: f64 = g
                .iter()
                .zip(r)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let norm = |v: &[f32]| v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (norm(g) * norm(r));
            let err = g
                .iter()
                .zip(r)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{} row {i}: cosine={cos:.9} max={err:.6} elapsed_ms={:.2}",
                case["name"],
                started.elapsed().as_secs_f64() * 1000.
            );
            if !cos.is_finite() || cos <= 0.9995 || err >= 0.004 {
                failures.push(format!("{} row {i}: {cos} / {err}", case["name"]));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
#[ignore = "real GGUF and companions; PADDOCK_EG2_MODEL, PADDOCK_EG2_VISION, PADDOCK_EG2_AUDIO"]
fn embeddinggemma2_media_lifetime_and_admission() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let mut m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    for variable in ["PADDOCK_EG2_VISION", "PADDOCK_EG2_AUDIO"] {
        m.attach_mmproj(Path::new(&std::env::var(variable).unwrap()), 70, true)
            .unwrap();
    }
    assert_eq!(m.media_kinds(), (true, true, true));
    let weights = m.weight_bytes;
    assert_eq!(m.device.allocated_bytes(), weights);
    let count = image_tokens(48, 48, 70).unwrap();
    let image = || MmChunk::Image {
        w: 48,
        h: 48,
        rgb: (0..48 * 48 * 3).map(|i| (i % 251) as u8).collect(),
    };
    let mut seq = vec![2, BOI_TOKEN];
    seq.extend(std::iter::repeat_n(IMAGE_TOKEN, count));
    seq.extend([EOI_TOKEN, 1]);
    let mut aud = vec![2, BOA_TOKEN];
    aud.extend(std::iter::repeat_n(
        AUDIO_TOKEN,
        audio_tokens(16000).unwrap(),
    ));
    aud.extend([EOA_TOKEN, 1]);
    let audio = || MmChunk::Audio {
        samples: vec![0.; 16000],
        mel: None,
    };
    let mut video = vec![2];
    for _ in 0..2 {
        video.push(BOI_TOKEN);
        video.extend(std::iter::repeat_n(VIDEO_TOKEN, count));
        video.push(EOI_TOKEN);
    }
    video.push(1);
    let sequences = vec![seq.clone(), aud.clone(), vec![2, 1000, 1], video];
    let media = vec![vec![image()], vec![audio()], vec![], vec![image(), image()]];
    let p = m
        .embed_submit_media(&sequences, &media, 0, Some(768))
        .unwrap();
    let got = m.embed_collect(&p).unwrap();
    drop(p);
    for (i, (s, medium)) in sequences.iter().zip(media).enumerate() {
        let p = m
            .embed_submit_media(std::slice::from_ref(s), &[medium], 0, None)
            .unwrap();
        assert_eq!(
            m.embed_collect(&p).unwrap()[0],
            got[i],
            "media changed in mixed batch"
        );
    }
    for v in &got {
        assert!(v.iter().all(|v| v.is_finite()));
        assert!((v.iter().map(|x| x * x).sum::<f32>() - 1.).abs() < 1e-5);
    }
    // Batch invariance alone cannot detect a disconnected media injection.
    // Keep the token IDs fixed and require actual pixels/PCM to matter.
    for (s, original, changed) in [
        (
            &seq,
            &got[0],
            MmChunk::Image {
                w: 48,
                h: 48,
                rgb: vec![0; 48 * 48 * 3],
            },
        ),
        (
            &aud,
            &got[1],
            MmChunk::Audio {
                samples: (0..16000).map(|i| (i as f32 * 0.1).sin() * 0.5).collect(),
                mel: None,
            },
        ),
    ] {
        let p = m
            .embed_submit_media(std::slice::from_ref(s), &[vec![changed]], 0, None)
            .unwrap();
        // A pending request owns its GPU inputs/scratch even if idle cleanup
        // runs before collection (as can happen after client cancellation).
        m.reclaim_idle();
        let output = m.embed_collect(&p).unwrap();
        let distance: f32 = output[0]
            .iter()
            .zip(original)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(
            distance > 0.001,
            "media content did not affect the embedding"
        );
    }
    m.reclaim_idle();
    assert_eq!(m.device.allocated_bytes(), weights);
    assert!(m.validate(&[seq.clone()]).is_err());
    assert!(m.validate_media(&[seq.clone()], &[vec![audio()]]).is_err());
    assert!(m.validate_media(&[seq.clone()], &[]).is_err());
    assert!(
        m.validate_media(&[vec![2, BOI_TOKEN, 1]], &[vec![]])
            .is_err()
    );
    assert!(
        m.validate_media(&[seq.clone()], &[vec![], vec![image()]])
            .is_err()
    );
    seq[0] = VOCAB as u32;
    assert!(m.validate_media(&[seq], &[vec![image()]]).is_err());
    assert!(
        m.validate_media(
            &[aud],
            &[vec![MmChunk::Audio {
                samples: vec![f32::NAN; 16000],
                mel: None
            }]]
        )
        .is_err()
    );
    assert_eq!(
        m.device.allocated_bytes(),
        weights,
        "invalid admission allocated memory"
    );
}

#[test]
#[ignore = "real media weights; verifies the catalog's bounded Metal workspace envelope"]
fn embeddinggemma2_media_workspace_envelope() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    // Same conservative envelope the manager admits: weights plus the Metal
    // text workspace and optional companions' existing workspace allowances.
    let weights = if Path::new(&path).is_dir() {
        288_165_984 + 367_075_328 + 588_618_624
    } else {
        294_064_224 + 368_268_992 + 613_806_752
    };
    let budget = weights + 805_306_368 + 169_869_312 + 33_554_432;
    let mut m = EmbeddingGemma2::load(Path::new(&path), CONTEXT, Some(budget)).unwrap();
    if m.mlx {
        m.attach_mlx_media(Path::new(&path), 1120, true, true)
            .unwrap();
    } else {
        for variable in ["PADDOCK_EG2_VISION", "PADDOCK_EG2_AUDIO"] {
            m.attach_mmproj(Path::new(&std::env::var(variable).unwrap()), 1120, true)
                .unwrap();
        }
    }
    let weights = m.device.allocated_bytes();
    drop(m.workspace(CONTEXT).unwrap());
    assert_eq!(m.device.allocated_bytes() - weights, 788_529_152);
    // Worst-budget picture after a maximum text workspace was cached.
    let mut ids = vec![2, BOI_TOKEN];
    ids.extend(std::iter::repeat_n(
        IMAGE_TOKEN,
        image_tokens(48, 48, 1120).unwrap(),
    ));
    ids.extend([EOI_TOKEN, 1]);
    let pending = m
        .embed_submit_media(
            &[ids],
            &[vec![MmChunk::Image {
                w: 48,
                h: 48,
                rgb: vec![127; 48 * 48 * 3],
            }]],
            0,
            None,
        )
        .unwrap();
    assert!(
        m.embed_collect(&pending).unwrap()[0]
            .iter()
            .all(|x| x.is_finite())
    );
    drop(pending);
    // A retained small text generation may coexist with the packed tower.
    // Nine maximum clips cross the 4096-row projection-group boundary and
    // then need a larger text generation. Both phases must fit the envelope.
    drop(m.workspace(3072).unwrap());
    let mut ids = vec![2, BOA_TOKEN];
    ids.extend(std::iter::repeat_n(
        AUDIO_TOKEN,
        audio_tokens(480000).unwrap(),
    ));
    ids.extend([EOA_TOKEN, 1]);
    let pending = m
        .embed_submit_media(
            &vec![ids; 9],
            &(0..9)
                .map(|_| {
                    vec![MmChunk::Audio {
                        samples: vec![0.; 480000],
                        mel: None,
                    }]
                })
                .collect::<Vec<_>>(),
            0,
            None,
        )
        .unwrap();
    let outputs = m.embed_collect(&pending).unwrap();
    assert_eq!(outputs.len(), 9);
    for output in &outputs {
        assert!(output.iter().all(|v| v.is_finite()));
        assert_eq!(
            output, &outputs[0],
            "projection group changed the embedding"
        );
    }
    drop(pending);
    m.reclaim_idle();
    assert_eq!(m.device.allocated_bytes(), weights);
}
