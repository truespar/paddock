//! Reference-input replay only. No checkpoint/oracle dependency in serving.
use super::*;

#[test]
#[ignore = "requires language capture with rotary positions"]
fn lighton_language_rotary_contract() {
    let root = std::env::var_os("PADDOCK_METAL_LANGUAGE_SEAMS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let model = Qwen35::load(
        Path::new(manifest["model"].as_str().unwrap()),
        4096,
        1,
        None,
    )
    .unwrap();
    let d = &model.device;
    let read = |name: &str| std::fs::read(root.join(format!("layer-3-{name}.f32"))).unwrap();
    let positions = read("position");
    let rows = positions.len() / 12;
    let pos: Vec<_> = positions
        .chunks_exact(12)
        .flat_map(|v| {
            let mut p = [0u32; 4];
            for (j, b) in v.chunks_exact(4).enumerate() {
                let f = f32::from_le_bytes(b.try_into().unwrap());
                assert!(f >= 0. && f.fract() == 0.);
                p[j] = f as u32;
            }
            p.into_iter().flat_map(u32::to_le_bytes)
        })
        .collect();
    let pos = d.upload(&pos).unwrap();
    let q = d.upload(&read("self_attn.q_proj-output")).unwrap();
    let k = d.upload(&read("self_attn.k_proj-output")).unwrap();
    let v = d.upload(&read("self_attn.v_proj-output")).unwrap();
    let meta = d
        .upload(
            &(0..rows as u32)
                .flat_map(|r| [0u32, r].into_iter().flat_map(u32::to_le_bytes))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let pages = d
        .upload(
            &(0..rows.div_ceil(16) as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let oq = d.alloc(rows * model.geometry.heads * 256 * 4).unwrap();
    let ok = d.alloc(rows * model.geometry.kv_heads * 256 * 2).unwrap();
    let ov = d.alloc(ok.len()).unwrap();
    let Mixer::Full(w) = &model.layers[3].mixer else {
        panic!("full attention")
    };
    let p = [
        model.geometry.heads as u32,
        model.geometry.kv_heads as u32,
        rows.div_ceil(16) as u32,
        model.rope.to_bits(),
        model.eps.to_bits(),
        model.rotary as u32,
    ];
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "mlx_qnorm_rope",
        &[&q, &w.q_norm.buffer, &pos, &oq],
        &p,
        [model.geometry.heads, rows, 1],
        32,
    );
    cmd.dispatch(
        "mlx_knorm_store",
        &[&k, &v, &w.k_norm.buffer, &meta, &pages, &ok, &ov, &pos],
        &p,
        [model.geometry.kv_heads, rows, 1],
        32,
    );
    cmd.finish().unwrap();
    let mut differences = 0;
    for (name, buffer, bf16) in [("query", &oq, false), ("key", &ok, true)] {
        let expected: Vec<_> = read(name)
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let actual = if bf16 {
            unsafe { buffer.read_u32(expected.len() / 2) }
                .into_iter()
                .flat_map(|w| {
                    [
                        half::bf16::from_bits(w as u16).to_f32(),
                        half::bf16::from_bits((w >> 16) as u16).to_f32(),
                    ]
                })
                .collect::<Vec<_>>()
        } else {
            unsafe { buffer.read_f32(0, expected.len()) }
        };
        let different = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "LANGUAGE_ROTARY {name} different={different}/{} max={max}",
            expected.len()
        );
        let outside_rotary = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(i, (a, b))| i % 256 >= model.rotary && a.to_bits() != b.to_bits())
            .count();
        eprintln!("LANGUAGE_ROTARY {name} outside_rotary={outside_rotary}");
        differences += different;
    }
    if std::env::var_os("PADDOCK_METAL_REQUIRE_PARITY").is_some() {
        assert_eq!(differences, 0);
    }
}

#[test]
fn paged_register_attention_preserves_ragged_tiles_pages_and_guards() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    for heads in [8usize, 16] {
        let kv_heads = 2;
        for length in [1usize, 15, 16, 17, 31, 32, 33, 129, 257] {
            let blocks = length.div_ceil(16);
            let padded = blocks * 16;
            let qbytes: Vec<_> = (0..length * heads * 256)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * 13 % 127) as f32 - 63.) / 32.).to_le_bytes()
                })
                .collect();
            let q = d.upload(&qbytes).unwrap();
            let meta = d
                .upload(
                    &(0..length as u32)
                        .flat_map(|r| [1u32, r].into_iter().flat_map(u32::to_le_bytes))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let limits = d
                .upload(
                    &(0..length as u32)
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let mut reference: Option<Vec<u32>> = None;
            for (tile_rows, reverse) in [(32, false), (1, true), (7, false), (16, true), (31, true)]
            {
                let page_map: Vec<_> = (0..blocks)
                    .map(|i| if reverse { blocks - 1 - i } else { i })
                    .collect();
                let pages = d
                    .upload(
                        &std::iter::repeat_n(u32::MAX, blocks)
                            .chain(page_map.iter().map(|&v| v as u32))
                            .flat_map(u32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let planes: Vec<_> = [7usize, 11]
                    .into_iter()
                    .map(|salt| {
                        let mut raw = vec![0u8; padded * kv_heads * 256 * 2];
                        for r in 0..length {
                            let physical = page_map[r / 16] * 16 + r % 16;
                            for col in 0..kv_heads * 256 {
                                let i = r * kv_heads * 256 + col;
                                let b = half::bf16::from_f32(((i * salt % 127) as f32 - 63.) / 32.)
                                    .to_le_bytes();
                                let at = (physical * kv_heads * 256 + col) * 2;
                                raw[at..at + 2].copy_from_slice(&b);
                            }
                        }
                        d.upload(&raw).unwrap()
                    })
                    .collect();
                let tile_count = length.div_ceil(tile_rows);
                let tiles = d
                    .upload(
                        &(0..length)
                            .step_by(tile_rows)
                            .flat_map(|r| {
                                [r as u32, (length - r).min(tile_rows) as u32]
                                    .into_iter()
                                    .flat_map(u32::to_le_bytes)
                            })
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                let count = length * heads * 256;
                let out = d.upload(&vec![0xff; (count + 16) * 4]).unwrap();
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "mlx_attention_prefill_nax",
                    &[
                        &q, &planes[0], &planes[1], &meta, &pages, &out, &tiles, &limits,
                    ],
                    &[
                        heads as u32,
                        kv_heads as u32,
                        blocks as u32,
                        (1f32 / 16.).to_bits(),
                    ],
                    [heads, tile_count, 1],
                    128,
                );
                cmd.finish().unwrap();
                let values = unsafe { out.read_f32(0, count) };
                assert!(values.iter().all(|v| v.is_finite()));
                assert!(
                    unsafe { out.read_f32(count, 16) }
                        .iter()
                        .all(|v| v.is_nan())
                );
                let bits: Vec<_> = values.iter().map(|v| v.to_bits()).collect();
                if let Some(expected) = &reference {
                    assert!(
                        bits == *expected,
                        "attention changed heads={heads} length={length} tile={tile_rows} reverse={reverse}: {:?}",
                        bits.iter()
                            .zip(expected)
                            .enumerate()
                            .filter(|(_, (a, b))| a != b)
                            .take(12)
                            .map(|(i, (&a, &b))| (i, f32::from_bits(a), f32::from_bits(b)))
                            .collect::<Vec<_>>()
                    );
                } else {
                    reference = Some(bits);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires --reference-results language capture; numerical ablation, not serving qualification"]
fn lighton_language_generation_ablation() {
    use paddock_engine::service::MmChunk;
    let root = std::env::var_os("PADDOCK_METAL_LANGUAGE_SEAMS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let path = Path::new(manifest["model"].as_str().unwrap());
    let f = &manifest["generation"];
    let ids = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>()
    };
    let chunks = vec![
        MmChunk::Text(ids(&f["before"])),
        MmChunk::Image {
            rgb: std::fs::read(root.join("image.rgb")).unwrap(),
            w: f["size"][0].as_u64().unwrap() as usize,
            h: f["size"][1].as_u64().unwrap() as usize,
        },
        MmChunk::Text(ids(&f["after"])),
    ];
    let reference = ids(&f["reference"]["ids"]);
    for (candidate, projection, decode) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, false, true),
        (false, true, true),
        (true, true, true),
    ] {
        let mut model = Qwen35::load(path, 4096, 4, None).unwrap();
        model.attach_vision(path).unwrap();
        attention::BASELINE_LANGUAGE_NAX_FOR_TEST.with(|v| v.set(!candidate));
        attention::FORCE_LANGUAGE_NAX_FOR_TEST.with(|v| v.set(candidate));
        crate::affine::REFERENCE_CHUNKS_FOR_TEST.with(|v| v.set(projection));
        crate::affine::REFERENCE_DECODE_FOR_TEST.with(|v| v.set(decode));
        let (mut logits, prompt) = model.prefill_images(0, &chunks).unwrap();
        assert_eq!(
            prompt,
            f["reference"]["usage"]["prompt_tokens"].as_u64().unwrap() as usize
        );
        let start = std::time::Instant::now();
        let mut generated = Vec::new();
        let mut stopped = false;
        for _ in 0..700 {
            let token = logits
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .unwrap()
                .0 as u32;
            generated.push(token);
            if token == 248046 {
                stopped = true;
                break;
            }
            logits = model.forward(token).unwrap();
        }
        attention::BASELINE_LANGUAGE_NAX_FOR_TEST.with(|v| v.set(false));
        attention::FORCE_LANGUAGE_NAX_FOR_TEST.with(|v| v.set(false));
        crate::affine::REFERENCE_CHUNKS_FOR_TEST.with(|v| v.set(false));
        crate::affine::REFERENCE_DECODE_FOR_TEST.with(|v| v.set(false));
        eprintln!(
            "LANGUAGE_GENERATION {}",
            serde_json::json!({"nax":candidate,"projection":projection,"decode":decode,"tokens":generated,"reference":reference,"exact":generated==reference,"stopped":stopped,"decode_seconds":start.elapsed().as_secs_f64()})
        );
        assert!(stopped, "generation cap, not natural EOS");
    }
}

#[test]
#[ignore = "requires metal-lighton-language.py reference capture"]
fn lighton_language_operation_contracts() {
    let root = std::env::var_os("PADDOCK_METAL_LANGUAGE_SEAMS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let model = Qwen35::load(
        Path::new(manifest["model"].as_str().unwrap()),
        4096,
        1,
        None,
    )
    .unwrap();
    let d = &model.device;
    let mut total = 0;
    for (name, info) in manifest["operations"].as_object().unwrap() {
        let rest = name.strip_prefix("layer-").unwrap();
        let (index, op) = rest.split_once('-').unwrap();
        let layer = &model.layers[index.parse::<usize>().unwrap()];
        let rows = info["input"]["shape"][0].as_u64().unwrap() as usize;
        let cols = info["output"]["shape"][1].as_u64().unwrap() as usize;
        let input = d
            .upload(&std::fs::read(root.join(format!("{name}-input.f32"))).unwrap())
            .unwrap();
        let output = d.alloc(rows * cols * 4).unwrap();
        let raw = std::fs::read(root.join(format!("{name}-output.f32"))).unwrap();
        assert_eq!(raw.len(), rows * cols * 4);
        let expected: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let spans = [(0, rows, CHUNK)];
        let reference_rows = info["input"]["full_shape"][1].as_u64().unwrap() as usize;
        let reference_contract = std::env::var_os("PADDOCK_METAL_REFERENCE_PROJECTION").is_some();
        let cmd = d.begin().unwrap();
        let cmd = if reference_contract {
            cmd.with_affine_prefill_rows(reference_rows)
        } else {
            cmd.with_projection_rows(&spans)
        };
        if matches!(op, "input_layernorm" | "post_attention_layernorm") {
            let norm = if op == "input_layernorm" {
                &layer.norm
            } else {
                &layer.post_norm
            };
            cmd.dispatch(
                "mlx_rms",
                &[&input, &norm.buffer, &output],
                &[model.width as u32, norm.ty, model.eps.to_bits()],
                [rows, 1, 1],
                (model.width.div_ceil(128) * 32).min(1024),
            );
        } else {
            let w = match op {
                "mlp.gate_proj" => &layer.gate,
                "mlp.up_proj" => &layer.up,
                "mlp.down_proj" => &layer.down,
                _ => match &layer.mixer {
                    Mixer::Linear(w) => match op {
                        "linear_attn.in_proj_qkv" => &w.qkv,
                        "linear_attn.in_proj_z" => &w.z,
                        "linear_attn.in_proj_a" => &w.alpha,
                        "linear_attn.in_proj_b" => &w.beta,
                        "linear_attn.out_proj" => &w.out,
                        _ => panic!("unknown operation {op}"),
                    },
                    Mixer::Full(w) => match op {
                        "self_attn.q_proj" => &w.q,
                        "self_attn.k_proj" => &w.k,
                        "self_attn.v_proj" => &w.v,
                        "self_attn.o_proj" => &w.o,
                        _ => panic!("unknown operation {op}"),
                    },
                },
            };
            assert_eq!(w.n, cols);
            model.project(&cmd, &[(w, &output)], &input, rows, &model.scratch.gemm);
        }
        cmd.finish().unwrap();
        let actual = unsafe { output.read_f32(0, expected.len()) };
        let different = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let squared: f64 = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|b| f64::from(*b).powi(2)).sum();
        assert!(actual.iter().all(|v| v.is_finite()));
        eprintln!(
            "LANGUAGE_CONTRACT {name} different={different}/{} max={max} relative_l2={}",
            expected.len(),
            (squared / norm.max(1e-30)).sqrt()
        );
        total += different;
    }
    if std::env::var_os("PADDOCK_METAL_REQUIRE_PARITY").is_some() {
        assert_eq!(total, 0, "language reference-input operation differences");
    }
}

#[test]
#[ignore = "requires metal-lighton-language.py attention capture"]
fn lighton_language_attention_contract() {
    let root = std::env::var_os("PADDOCK_METAL_LANGUAGE_SEAMS").unwrap();
    let root = Path::new(&root);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let config =
        paddock_models::mlx::QwenConfig::read(Path::new(manifest["model"].as_str().unwrap()))
            .unwrap();
    let d = MetalDevice::new(Some(256 << 20)).unwrap();
    let mut differences = 0;
    for index in manifest["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .filter(|i| (i + 1) % 4 == 0)
    {
        let name = format!("layer-{index}");
        let rows = manifest["attention"][format!("{name}-query")]["shape"][0]
            .as_u64()
            .unwrap() as usize;
        let heads = config.heads;
        let kv_heads = config.kv_heads;
        let read = |suffix: &str| -> Vec<f32> {
            std::fs::read(root.join(format!("{name}-{suffix}.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let bf = |values: &[f32]| {
            values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect::<Vec<_>>()
        };
        let mut queries = bf(&read("query"));
        queries.resize((rows + 32) * heads * 256 * 2, 0);
        let q = d.upload(&queries).unwrap();
        let k = d.upload(&bf(&read("key"))).unwrap();
        let v = d.upload(&bf(&read("value"))).unwrap();
        let meta = d
            .upload(
                &(0..rows as u32)
                    .flat_map(|r| [0u32, r].into_iter().flat_map(u32::to_le_bytes))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let pages = d
            .upload(
                &(0..rows.div_ceil(16) as u32)
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let limits = d
            .upload(
                &(0..rows as u32)
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let tiles = d
            .upload(
                &(0..rows)
                    .step_by(32)
                    .flat_map(|r| {
                        [r as u32, (rows - r).min(32) as u32]
                            .into_iter()
                            .flat_map(u32::to_le_bytes)
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let output = d.alloc(rows * heads * 256 * 4).unwrap();
        let cmd = d.begin().unwrap();
        let kernel = if std::env::var_os("PADDOCK_METAL_LANGUAGE_NAX").is_some() {
            "mlx_attention_prefill_nax"
        } else {
            "mlx_attention_prefill_direct"
        };
        cmd.dispatch(
            kernel,
            &[&q, &k, &v, &meta, &pages, &output, &tiles, &limits],
            &[
                heads as u32,
                kv_heads as u32,
                rows.div_ceil(16) as u32,
                (1f32 / 16.).to_bits(),
            ],
            [heads, rows.div_ceil(32), 1],
            128,
        );
        cmd.finish().unwrap();
        let expected = read("attention");
        let actual = unsafe { output.read_f32(0, expected.len()) };
        assert!(actual.iter().all(|v| v.is_finite()));
        let different = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| half::bf16::from_f32(**a).to_f32() != **b)
            .count();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (half::bf16::from_f32(*a).to_f32() - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "LANGUAGE_ATTENTION {name} different={different}/{} max={max}",
            expected.len()
        );
        differences += different;
    }
    if std::env::var_os("PADDOCK_METAL_REQUIRE_PARITY").is_some() {
        assert_eq!(
            differences, 0,
            "language attention reference-input differences"
        );
    }
}
