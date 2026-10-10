// GPU and full-model validation lives here; CPU inference is not a reference.
use super::*;

#[test]
fn head256_prefill_matches_split_gqa_with_shuffled_pages() {
    for (g, prefix, strict) in [
        Geometry::DENSE_08B,
        Geometry::DENSE_4B,
        Geometry::DENSE_9B,
        Geometry::DENSE_27B,
        Geometry::MOE_35B,
    ]
    .into_iter()
    .flat_map(|g| [3usize, 997].map(|p| (g, p)))
    .flat_map(|(g, p)| [false, true].map(|strict| (g, p, strict)))
    {
        let heads = g.heads;
        let device = MetalDevice::new(Some(192 << 20)).unwrap();
        let rows = 37usize;
        let stride = (prefix + rows).div_ceil(16).max(4);
        let blocks = stride * 2;
        let upload_u = |v: &[u32]| {
            device
                .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let q = device
            .upload(
                &(0..rows * heads * 256)
                    .flat_map(|i| {
                        (((i * 7 + i / 256) % 17) as f32 / if strict { 257.0 } else { 256.0 }
                            - 8.0 / 256.0)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let half_bytes = |mult: usize| {
            (0..blocks * 16 * g.kv_heads * 256)
                .flat_map(|i| {
                    half::f16::from_f32(((i * mult + i / 256) % 19) as f32 / 256.0 - 9.0 / 256.0)
                        .to_le_bytes()
                })
                .collect::<Vec<_>>()
        };
        let keys = device.upload(&half_bytes(3)).unwrap();
        let values = device.upload(&half_bytes(11)).unwrap();
        let meta = upload_u(
            &(0..rows)
                .flat_map(|r| [1, (r + prefix) as u32])
                .collect::<Vec<_>>(),
        );
        let pages = upload_u(
            &(0..stride)
                .map(|i| (i * 2) as u32)
                .chain((0..stride).rev().map(|i| (i * 2 + 1) as u32))
                .collect::<Vec<_>>(),
        );
        let selected = upload_u(&(0..rows as u32).rev().collect::<Vec<_>>());
        let limits = upload_u(
            &(0..rows as u32)
                .map(|r| r + prefix as u32)
                .collect::<Vec<_>>(),
        );
        let tiles = upload_u(&[32, 5, 0, 32]);
        let query = device.alloc((rows + 32) * heads * 256 * 2).unwrap();
        let tensor = device.alloc(rows * heads * 256 * 4).unwrap();
        let vector = device.alloc(tensor.len()).unwrap();
        let parts = device.alloc(rows * heads * MAX_SPLITS * 258 * 4).unwrap();
        // Two query tiles, four independent partitions per head. No group
        // may alias another group's device-backed K/V staging.
        let staging = device
            .alloc(2 * heads * 4 * 32 * 256 * if strict { 4 } else { 2 })
            .unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "attention_query",
            &[&q, &query],
            &[(heads * 256) as u32, 0, rows as u32],
            [((rows + 32) * heads * 256).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            if strict {
                "qwen_attention_prefill_strict"
            } else {
                "qwen_attention_prefill"
            },
            &[
                if strict { &q } else { &query },
                &keys,
                &values,
                &meta,
                &pages,
                &tensor,
                &tiles,
                &limits,
                &staging,
            ],
            &[
                heads as u32,
                g.kv_heads as u32,
                stride as u32,
                (1.0f32 / 16.0).to_bits(),
            ],
            [heads, 2, 1],
            128,
        );
        cmd.finish().unwrap();
        let expected = unsafe { tensor.read_f32(0, rows * heads * 256) };
        assert!(expected.iter().all(|v| v.is_finite()));
        // Both short, partly empty partitions and a long ragged prefix. Retain
        // the original unsplit tensor kernel and independent split-GQA kernel
        // as GPU arithmetic comparisons under the unchanged tolerance.
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            if strict {
                "qwen_attention_prefill_split_strict"
            } else {
                "qwen_attention_prefill_split"
            },
            &[
                if strict { &q } else { &query },
                &keys,
                &values,
                &meta,
                &pages,
                &parts,
                &tiles,
                &limits,
                &staging,
            ],
            &[
                heads as u32,
                g.kv_heads as u32,
                stride as u32,
                (1.0f32 / 16.0).to_bits(),
            ],
            [heads, 2, 4],
            128,
        );
        cmd.dispatch(
            "qwen_attention_prefill_join",
            &[&parts, &vector, &tiles],
            &[heads as u32],
            [heads, 2, 32],
            32,
        );
        cmd.finish().unwrap();
        let actual = unsafe { vector.read_f32(0, expected.len()) };
        assert!(actual.iter().all(|v| v.is_finite()));
        let error = expected
            .iter()
            .zip(actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 0.00001, "append attention prefix {prefix}: {error}");
        for splits in [1usize, 3, 17, 32] {
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                g.decode_kernel(),
                &[
                    &q, &keys, &values, &meta, &pages, &selected, &parts, &limits,
                ],
                &[
                    heads as u32,
                    g.kv_heads as u32,
                    stride as u32,
                    (1.0f32 / 16.0).to_bits(),
                    splits as u32,
                ],
                [g.kv_heads, rows, splits],
                128,
            );
            cmd.dispatch(
                "qwen_attention_merge",
                &[&parts, &vector, &selected],
                &[heads as u32, splits as u32],
                [heads * rows, 1, 1],
                32,
            );
            cmd.finish().unwrap();
            let actual = unsafe { vector.read_f32(0, expected.len()) };
            assert!(actual.iter().all(|v| v.is_finite()));
            let max = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max < 0.00001, "head-256 attention {splits} splits: {max}");
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL and an M5"]
fn qwen_checkpoint_resume_holes_mixed_and_cancellation() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").expect("Qwen GGUF path");
    let mut model = Qwen35::load(Path::new(&path), 1024, 4, None).unwrap();
    let p: Vec<u32> = (1000..1537).collect();
    let baseline = model.forward_prefill(0, &p).unwrap();
    let next = model.forward(11751).unwrap();
    let cached = model
        .cache
        .iter()
        .position(|c| c.history.len() == 528)
        .unwrap()
        + model.slots.len();
    assert_eq!(model.prepare(2, &p).unwrap(), 528);
    for layer in 0..model.geometry.linear_layers() {
        for (buffer, stride) in [
            (&model.state, model.geometry.state()),
            (&model.conv, model.geometry.conv() * 3),
        ] {
            let saved =
                unsafe { buffer.read_f32((layer * model.state_slots + cached) * stride, stride) };
            let restored =
                unsafe { buffer.read_f32((layer * model.state_slots + 2) * stride, stride) };
            assert!(
                saved == restored,
                "exact checkpoint bytes, layer {layer}, stride {stride}"
            );
        }
    }
    let resumed = model.forward_prefill(2, &p).unwrap();
    assert_eq!(model.take_prefill_reused(2), 528);
    // Cold prefill uses F16 TensorOps; the cached 9-row suffix uses FP32
    // SIMD projections/recurrence. Require the existing schedule tolerance
    // and identical greedy selection, not impossible cross-route bit parity.
    let compare = |a: &[f32], b: &[f32]| {
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        eprintln!("checkpoint schedule max logit difference {max}");
        assert!(b.iter().all(|v| v.is_finite()));
        assert!(max < 0.08, "checkpoint schedule error {max}");
        assert_eq!(top(a), top(b));
    };
    compare(&baseline, &resumed);
    let batch = model.forward_batch(&[0, 0, 11751], &[0, 0, 537]).unwrap();
    assert!(batch[..2 * model.vocab].iter().all(|v| *v == 0.0));
    compare(&next, &batch[2 * model.vocab..]);

    let p2: Vec<u32> = (2000..2073).collect();
    let reference = model.forward_prefill(1, &p2).unwrap();
    // Clear checkpoints so the aborted request must actually run GPU work.
    for c in &mut model.cache {
        c.table.clear(&mut model.pool);
        c.history.clear();
    }
    model.prefill_begin(1, p2.clone()).unwrap();
    let (decodes, done) = model.forward_mixed(&[(2, 13, 538)], 7).unwrap();
    assert_eq!(decodes.len(), model.vocab);
    assert!(done.is_empty());
    assert!(model.prefill_abort(1));
    model.prefill_begin(1, p2).unwrap();
    let mut done = Vec::new();
    for _ in 0..20 {
        done.extend(model.forward_mixed(&[], 7).unwrap().1);
        if !done.is_empty() {
            break;
        }
    }
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].2, 73);
    let max = reference
        .iter()
        .zip(&done[0].1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(done[0].1.iter().all(|v| v.is_finite()));
    assert!(max < 0.12, "chunk/serial Qwen logit difference {max}");
    model.release_inactive_slots(&[false; 4]);
    assert!(
        model
            .forward_batch(&[0; 4], &[0; 4])
            .unwrap()
            .iter()
            .all(|v| *v == 0.0)
    );
    assert!(model.forward_prefill(0, &vec![1; 1025]).is_err());
    assert!(model.forward_mixed(&[(4, 1, 0)], 1).is_err());
    assert!(model.forward_mixed(&[(0, 1, 0), (0, 2, 2)], 0).is_err());
    assert!(model.device_mem_used().unwrap() <= model.device.budget_bytes());
}

#[test]
fn kquant_formats_and_packed_rungs_match_gpu_vector() {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let upload_u = |v: &[u32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let mut blocks = Vec::new();
    // Format fixtures have analytically known values. This tests the GGUF
    // bit fields independently from the shared decoder used by both GEMMs.
    for ty in [12u32, 13, 14, 23] {
        let size = match ty {
            12 => 144,
            13 => 176,
            14 => 210,
            _ => 136,
        };
        let mut block = vec![0u8; size];
        let expected: Vec<f32> = match ty {
            12 | 13 => {
                block[..4].copy_from_slice(&[0, 0x3c, 0, 0x3c]);
                block[4..12].fill(1);
                block[12..16].fill(0x11);
                let start = if ty == 12 { 16 } else { 48 };
                block[start..].fill(0x21);
                if ty == 13 {
                    block[16..48].fill(0xaa);
                }
                (0..256)
                    .map(|i| {
                        if i / 32 % 2 == 0 {
                            0.0
                        } else if ty == 12 {
                            1.0
                        } else {
                            17.0
                        }
                    })
                    .collect()
            }
            14 => {
                block[..128].fill(0x21);
                block[128..192].fill(0xe4);
                block[192..208].fill(1);
                block[208..].copy_from_slice(&[0, 0x3c]);
                (0..256)
                    .map(|i| [-31.0, -15.0, 2.0, 18.0][i / 32 % 4])
                    .collect()
            }
            _ => {
                block[..2].copy_from_slice(&[0, 0x3c]);
                block[4..8].fill(0x22);
                block[8..].fill(0x21);
                (0..256)
                    .map(|i| if i % 32 < 16 { 3120.0 } else { 2490.0 })
                    .collect()
            }
        };
        let w = device.upload(&block).unwrap();
        let ids = upload_u(&[0]);
        let decoded = device.alloc(256 * 4).unwrap();
        let cmd = device.begin().unwrap();
        cmd.dispatch(
            "embed",
            &[&w, &ids, &decoded],
            &[256, 1, ty, 1.0f32.to_bits()],
            [1, 1, 1],
            256,
        );
        cmd.finish().unwrap();
        assert_eq!(unsafe { decoded.read_f32(0, 256) }, expected, "format {ty}");
        blocks.push((ty, block));
    }
    for (ty, block) in blocks {
        for (k, n, m) in [
            (256usize, 35usize, 1usize),
            (512, 64, 2),
            (768, 35, 3),
            (512, 48, 4),
            (256, 35, 9),
            (512, 48, 17),
            (256, 64, 32),
            (512, 35, 33),
            (512, 35, 63),
            (512, 35, 64),
            (512, 35, 65),
            (512, 35, 95),
            (256, 48, 96),
            (512, 35, 97),
            (512, 35, 129),
        ] {
            let w = device.upload(&block.repeat(k * n / 256)).unwrap();
            let x = device
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 7 + i / k) % 13) as f32 / 128.0 - 6.0 / 128.0).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let a = device.alloc(n * m * 4).unwrap();
            let b = device.alloc(n * m * 4).unwrap();
            let padded = device
                .alloc(m.div_ceil(128) * 128 * k.div_ceil(128) * 128 * 2)
                .unwrap();
            let p = [k as u32, n as u32, m as u32, ty, 1.0f32.to_bits()];
            let cmd = device.begin().unwrap();
            cmd.dispatch("linear", &[&w, &x, &a], &p, [n.div_ceil(4), m, 1], 128);
            let weight = Weight {
                buffer: w,
                ty,
                k,
                n,
            };
            weight.linear(&cmd, &x, &b, m, 1.0, &padded);
            cmd.finish().unwrap();
            assert_eq!(
                unsafe { a.read_f32(0, m * n) },
                unsafe { b.read_f32(0, m * n) },
                "{ty}/{k}/{n}/{m}"
            );
            for domains in [2, 3] {
                let c = device.alloc(b.len()).unwrap();
                let d = device.alloc(b.len()).unwrap();
                let weights = [(&weight, &b), (&weight, &c), (&weight, &d)];
                let cmd = device.begin().unwrap();
                projections(&cmd, &weights[..domains], &x, m, &padded);
                cmd.finish().unwrap();
                for (_, out) in &weights[..domains] {
                    assert_eq!(
                        unsafe { a.read_f32(0, m * n) },
                        unsafe { out.read_f32(0, m * n) },
                        "domains {ty}/{m}/{domains}"
                    );
                }
            }
        }
    }
}

#[test]
fn kquant_varied_superblocks_match_gpu_arithmetic() {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    for (k, n, counts, split_scratch) in [
        (
            512usize,
            67usize,
            &[
                1usize, 2, 3, 4, 5, 7, 8, 9, 15, 17, 33, 47, 48, 49, 65, 80, 95, 96, 97, 129, 273,
            ][..],
            false,
        ),
        (5120, 1041, &[5usize, 8, 17, 33, 64][..], false),
        (5120, 1041, &[5usize, 8, 17, 33, 64][..], true),
        (
            8192,
            1041,
            &[1usize, 2, 3, 4, 5, 7, 8, 9, 17, 32, 33, 64][..],
            true,
        ),
    ] {
        for (ty, size) in [(12u32, 144usize), (13, 176), (14, 210), (23, 136)] {
            let mut bytes: Vec<u8> = (0..k * n / 256 * size)
                .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
                .collect();
            for block in bytes.chunks_exact_mut(size) {
                let d = if ty == 14 { 208 } else { 0 };
                block[d..d + 2].copy_from_slice(&half::f16::from_f32(1.0 / 1024.0).to_le_bytes());
                if ty == 12 || ty == 13 {
                    block[2..4].copy_from_slice(&half::f16::from_f32(1.0 / 4096.0).to_le_bytes());
                }
            }
            let weight = Weight {
                buffer: device.upload(&bytes).unwrap(),
                ty,
                k,
                n,
            };
            for &m in counts {
                let values: Vec<u8> = (0..m * k)
                    .flat_map(|i| (((i * 17 + i / k * 3) % 31) as f32 / 31.0 - 0.5).to_le_bytes())
                    .collect();
                let x = device.upload(&values).unwrap();
                let a = device.alloc(m * n * 4).unwrap();
                let b = device.alloc(a.len()).unwrap();
                let padded = device
                    .alloc(
                        m.div_ceil(128) * 128 * k * 2
                            + if split_scratch { 4 * m * n * 4 } else { 0 },
                    )
                    .unwrap();
                let cmd = device.begin().unwrap();
                // Decode keeps the original F32 inputs, including values not
                // representable in F16. Compare against the independent GPU
                // vector path without masking an accidental activation cut.
                if m < 5 {
                    cmd.dispatch(
                        "linear",
                        &[&weight.buffer, &x, &a],
                        &[k as u32, n as u32, m as u32, ty, 1.0f32.to_bits()],
                        [n.div_ceil(4), m, 1],
                        128,
                    );
                } else {
                    // The reference expands only this test matrix on the GPU;
                    // production keeps original quant bytes and bounded tiles.
                    let wf = device.alloc(k * n * 2).unwrap();
                    cmd.dispatch(
                        "linear_prepare",
                        &[&weight.buffer, &x, &wf, &padded],
                        &[k as u32, n as u32, m as u32, ty],
                        [(k * n.max(m)).div_ceil(256), 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        "linear_mpp",
                        &[&wf, &padded, &a],
                        &[k as u32, n as u32, m as u32, 1, 1.0f32.to_bits()],
                        [n.div_ceil(64), m.div_ceil(32), 1],
                        128,
                    );
                }
                weight.linear(&cmd, &x, &b, m, 1.0, &padded);
                cmd.finish().unwrap();
                let a = unsafe { a.read_f32(0, m * n) };
                let b = unsafe { b.read_f32(0, m * n) };
                assert!(a.iter().chain(&b).all(|x| x.is_finite()));
                let error = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    error < 0.0005,
                    "format {ty}, rows {m}, max GPU error {error}"
                );
            }
        }
    }
}

#[test]
fn transient_prefill_expansion_matches_tiled_gpu_for_all_kquants() {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let (k, n, m) = (768usize, 1041usize, 1041usize);
    for (ty, size) in [(12u32, 144usize), (13, 176), (14, 210), (23, 136)] {
        let mut bytes: Vec<u8> = (0..k * n / 256 * size)
            .map(|i| ((i * 73 + i / 19 * 37 + 11) % 256) as u8)
            .collect();
        for b in bytes.chunks_exact_mut(size) {
            let at = if ty == 14 { 208 } else { 0 };
            b[at..at + 2].copy_from_slice(&half::f16::from_f32(1.0 / 1024.0).to_le_bytes());
            if matches!(ty, 12 | 13) {
                b[2..4].copy_from_slice(&half::f16::from_f32(1.0 / 4096.0).to_le_bytes());
            }
        }
        let w = Weight {
            buffer: device.upload(&bytes).unwrap(),
            ty,
            k,
            n,
        };
        let x = device
            .upload(
                &(0..m * k)
                    .flat_map(|i| (((i * 17 + i / k * 3) % 31) as f32 / 31.0 - 0.5).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let tiled = device.alloc(m * n * 4).unwrap();
        let expanded = device.alloc(m * n * 4).unwrap();
        let pad = device.alloc(m.div_ceil(128) * 128 * k * 2).unwrap();
        let slab = device.alloc(pad.len() + k * n * 2).unwrap();
        let cmd = device.begin().unwrap();
        w.linear(&cmd, &x, &tiled, m, 1.0, &pad);
        w.linear(&cmd, &x, &expanded, m, 1.0, &slab);
        cmd.finish().unwrap();
        let expected = unsafe { tiled.read_f32(0, m * n) };
        let check = |out: &Buffer| {
            let actual = unsafe { out.read_f32(0, m * n) };
            assert!(actual.iter().all(|v| v.is_finite()));
            let error = expected
                .iter()
                .zip(actual)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 0.0005, "transient format {ty}: max error {error}");
        };
        check(&expanded);
        let second = device.alloc(m * n * 4).unwrap();
        let third = device.alloc(m * n * 4).unwrap();
        let domains = [(&w, &expanded), (&w, &second), (&w, &third)];
        let cmd = device.begin().unwrap();
        projections(&cmd, &domains, &x, m, &slab);
        cmd.finish().unwrap();
        for (_, out) in domains {
            check(out);
        }
    }
}

#[test]
#[ignore = "real target projection diagnostic; not a serving or parity gate"]
fn real_projection_route_diagnostic() {
    let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").expect("target");
    let map = paddock_models::mapped::MappedGguf::open(Path::new(&path)).unwrap();
    let device = MetalDevice::new(None).unwrap();
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.3.attn_q.weight",
    ] {
        let dims = map
            .gguf()
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap()
            .dims
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>();
        let w = Weight::load(&device, &map, name, &dims).unwrap();
        for m in [256usize, 576, 1101] {
            let (k, n) = (w.k, w.n);
            let x = device
                .upload(
                    &(0..m * k)
                        .flat_map(|i| {
                            (((i * 17 + i / k * 3) % 31) as f32 / 31.0 - 0.5).to_le_bytes()
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            let pad = device
                .alloc(m.div_ceil(128) * 128 * k.div_ceil(128) * 128 * 2)
                .unwrap();
            let slab = device.alloc(pad.len() + k * n * 2).unwrap();
            let a = device.alloc(m * n * 4).unwrap();
            let b = device.alloc(a.len()).unwrap();
            let mut times = Vec::new();
            for _ in 0..4 {
                let cmd = device.begin().unwrap();
                w.linear(&cmd, &x, &a, m, 1.0, &pad);
                let tiled = cmd.finish().unwrap() * 1000.0;
                let cmd = device.begin().unwrap();
                w.linear(&cmd, &x, &b, m, 1.0, &slab);
                let expanded = cmd.finish().unwrap() * 1000.0;
                times.push([tiled, expanded]);
            }
            let actual = unsafe { a.read_f32(0, m * n) };
            let expected = unsafe { b.read_f32(0, m * n) };
            assert!(actual.iter().chain(&expected).all(|v| v.is_finite()));
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "PROJECTION_ROUTE {}",
                serde_json::json!({"name":name,"k":k,"n":n,"m":m,"times_tiled_expanded_ms":times,"max_delta":error})
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_QWEN_MODEL and an M5"]
fn qwen_four_row_decode_and_ragged_prefill_match_serial_gpu() {
    let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").expect("Qwen GGUF path");
    let mut model = Qwen35::load(Path::new(&path), 1024, 4, None).unwrap();
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32
    };
    let compare = |a: &[f32], b: &[f32]| {
        let error = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(b.iter().all(|x| x.is_finite()));
        eprintln!(
            "Qwen batch/serial max error {error}; top {}/{}",
            top(a),
            top(b)
        );
        assert!(error < 0.08, "Qwen batch/serial GPU error {error}");
        assert_eq!(top(a), top(b));
    };
    for lengths in [[37, 25, 41, 33], [537, 552, 560, 569]] {
        let prompts: Vec<Vec<u32>> = lengths
            .iter()
            .enumerate()
            .map(|(s, &n)| (0..n).map(|i| 1000 + (s * 700 + i) as u32).collect())
            .collect();
        let mut reference = Vec::new();
        for p in &prompts {
            let logits = model.forward_prefill(0, p).unwrap();
            let next = model.forward(top(&logits)).unwrap();
            reference.push((logits, next));
        }
        for staggered in [false, true] {
            model.reset();
            for c in &mut model.cache {
                c.table.clear(&mut model.pool);
                c.history.clear();
            }
            for (s, p) in prompts.iter().enumerate() {
                model.prefill_begin(s, p.clone()).unwrap();
                if staggered && s == 0 {
                    let (decode, completed) = model.forward_mixed(&[], 512).unwrap();
                    assert!(decode.is_empty() && completed.is_empty());
                    assert_eq!(model.pending[0].offset, 32);
                    assert_eq!(model.slots[0].history.len(), 32);
                }
            }
            let mut finished = Vec::new();
            for _ in 0..8 {
                finished.extend(model.forward_mixed(&[], 512).unwrap().1);
                if finished.len() == 4 {
                    break;
                }
            }
            assert_eq!(finished.len(), 4);
            let mut tokens = vec![0; 4];
            for (s, logits, n) in finished {
                assert_eq!(n, lengths[s]);
                eprintln!("prefill slot={s}, tokens={n}, staggered={staggered}");
                compare(&reference[s].0, &logits);
                tokens[s] = top(&logits);
            }
            let batch = model
                .forward_batch(&tokens, &lengths.map(|n| n as u32))
                .unwrap();
            for (s, row) in batch.chunks_exact(model.vocab).enumerate() {
                eprintln!(
                    "decode slot={s}, tokens={}, staggered={staggered}",
                    lengths[s]
                );
                compare(&reference[s].1, row);
            }
        }
    }
}

#[test]
fn deltanet_chunked_matches_gpu_recurrence_with_ragged_resumed_spans() {
    for vh in [2, 4, 6] {
        for strict in [false, true] {
            deltanet_chunked_geometry(vh, strict);
        }
    }
}

fn deltanet_chunked_geometry(vh: usize, strict: bool) {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let kh = 2usize;
    let width = (kh * 2 + vh) * 128;
    let rows = 55usize;
    let upload_u = |v: &[u32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let values: Vec<f32> = (0..rows * width)
        .map(|i| ((i * 17 + i / 128 * 3) % 31) as f32 / 128.0 - 15.0 / 128.0)
        .collect();
    // Four disjoint in-graph snapshots, including a cut inside a 32-row
    // chunk, two cuts in one span, and a one-row resumed span.
    let initial: Vec<f32> = (0..7 * vh * 128 * 128)
        .map(|i| ((i * 7 + i / 128) % 17) as f32 / 2048.0 - 8.0 / 2048.0)
        .collect();
    let upload_f = |v: &[f32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let qkv = upload_f(&values);
    let a = upload_f(&initial);
    let b = upload_f(&initial);
    let gates = upload_f(
        &(0..rows * vh)
            .flat_map(|i| [-0.015625 * (1 + i % 3) as f32, 0.375])
            .collect::<Vec<_>>(),
    );
    let spans = upload_u(&[0, 17, 2, 0, 17, 37, 0, 2, 54, 1, 1, 4]);
    let chunks = upload_u(&[0, 9, 2, 4, 9, 8, 2, 5, 17, 16, 0, 6, 33, 21, 0, 0]);
    let mut cuts = vec![0; rows];
    for (row, destination) in [(8, 4), (16, 5), (32, 6), (54, 7)] {
        cuts[row] = destination;
    }
    let checkpoints = upload_u(&cuts);
    let meta = upload_u(
        &(0..rows)
            .flat_map(|r| {
                if r < 17 {
                    [2, 200 + r as u32]
                } else if r < 54 {
                    [0, (r - 17) as u32]
                } else {
                    [1, 777]
                }
            })
            .collect::<Vec<_>>(),
    );
    let reference = device.alloc(rows * vh * 128 * 4).unwrap();
    let actual = device.alloc(reference.len()).unwrap();
    let prepared = device
        .alloc(4 * vh * 17408 * if strict { 4 } else { 2 })
        .unwrap();
    let mut p = [kh as u32, vh as u32, width as u32, rows as u32, 7, 0, 1];
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "dn_recurrent",
        &[&qkv, &gates, &a, &spans, &meta, &reference, &checkpoints],
        &p,
        [8, vh, 3],
        128,
    );
    cmd.finish().unwrap();
    p[6] = 0;
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "dn_recurrent",
        &[&qkv, &gates, &b, &spans, &meta, &actual, &checkpoints],
        &p,
        [8, vh, 3],
        128,
    );
    cmd.dispatch(
        if strict {
            "dn_chunk_dots_strict"
        } else {
            "dn_chunk_dots"
        },
        &[&qkv, &gates, &chunks, &prepared],
        &p,
        [vh, 4, 1],
        128,
    );
    cmd.dispatch(
        if strict {
            "dn_chunk_prepare_strict"
        } else {
            "dn_chunk_prepare"
        },
        &[&qkv, &gates, &chunks, &prepared],
        &p,
        [vh, 4, 1],
        128,
    );
    cmd.dispatch(
        if strict {
            "dn_chunk_walk_strict"
        } else {
            "dn_chunk_walk"
        },
        &[&prepared, &gates, &spans, &chunks, &meta, &b, &actual],
        &p,
        [8, vh, 3],
        128,
    );
    cmd.finish().unwrap();
    let mut good = true;
    for (name, x, y) in [("output", &reference, &actual), ("state", &a, &b)] {
        let x = unsafe { x.read_f32(0, x.len() / 4) };
        let y = unsafe { y.read_f32(0, y.len() / 4) };
        assert!(y.iter().all(|v| v.is_finite()), "nonfinite {name}");
        let max = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rms =
            (x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / x.len() as f32).sqrt();
        eprintln!("{name}, strict={strict}: max {max}, rms {rms}");
        good &= if strict {
            max < 0.000005 && rms < 0.0000005
        } else {
            max < 0.0005 && rms < 0.00005
        };
    }
    assert!(good, "chunked DeltaNet arithmetic mismatch");
}

#[test]
fn deltanet_conv_checkpoints_preserve_incoming_and_interior_windows() {
    let device = MetalDevice::new(Some(128 << 20)).unwrap();
    let width = 257usize; // Ragged channel tail and nonzero layer stride.
    let slots = 7;
    let input: Vec<f32> = (0..8 * width).map(|i| i as f32).collect();
    let initial: Vec<f32> = (0..2 * slots * 3 * width)
        .map(|i| -(i as f32) - 1.0)
        .collect();
    let upload_f = |v: &[f32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let upload_u = |v: &[u32]| {
        device
            .upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let x = upload_f(&input);
    let history = upload_f(&initial);
    let spans = upload_u(&[0, 5, 2, 0, 5, 2, 0, 0, 7, 1, 1, 0]);
    let meta = upload_u(&[2, 9, 2, 10, 2, 11, 2, 12, 2, 13, 0, 0, 0, 1, 1, 99]);
    let cuts = upload_u(&[0, 4, 3, 5, 5, 6, 0, 0, 7, 7, 0, 0]);
    let cmd = device.begin().unwrap();
    cmd.dispatch(
        "dn_conv_commit",
        &[&x, &history, &spans, &meta, &cuts],
        &[0, 0, width as u32, 8, slots as u32, 1],
        [width.div_ceil(256), 3, 1],
        256,
    );
    cmd.finish().unwrap();
    let result = unsafe { history.read_f32(0, initial.len()) };
    assert_eq!(&result[..slots * 3 * width], &initial[..slots * 3 * width]);
    // These are storage/window fixtures, not a CPU inference reference.
    for (target, source, first, boundary, fresh) in [
        (2, 2, 0, 4, false),
        (0, 0, 5, 6, true),
        (1, 1, 7, 7, false),
        (3, 2, 0, 0, false),
        (4, 2, 0, 3, false),
        (5, 0, 5, 5, true),
        (6, 1, 7, 7, false),
    ] {
        for j in 0..3 {
            let row = boundary as isize + j as isize - 2;
            let start = ((slots + target) * 3 + j) * width;
            let expected = if row >= first as isize {
                input[row as usize * width..(row as usize + 1) * width].to_vec()
            } else if fresh {
                vec![0.0; width]
            } else {
                let old = ((slots + source) * 3 + (row - first as isize + 3) as usize) * width;
                initial[old..old + width].to_vec()
            };
            assert_eq!(
                &result[start..start + width],
                expected,
                "slot {target}, window {j}"
            );
        }
    }
}
