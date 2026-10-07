use super::*;
use paddock_engine::generator::Generator;

fn floats(d: &MetalDevice, values: impl IntoIterator<Item = f32>) -> Buffer {
    d.upload(
        &values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn ints(d: &MetalDevice, values: impl IntoIterator<Item = u32>) -> Buffer {
    d.upload(
        &values
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn raw_logit_bias_routes_top_six_without_normalizing_weights() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let logits = floats(&d, [0.; EXPERTS * 2]);
    let bias = floats(&d, (0..EXPERTS).map(|i| if i == 383 { 10. } else { 0. }));
    let ids = ints(&d, [u32::MAX; 16]);
    let weights = floats(&d, [f32::NAN; 16]);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "kolibri_route",
        &[&logits, &bias, &ids, &weights],
        &[],
        [2, 1, 1],
        32,
    );
    cmd.finish().unwrap();
    assert_eq!(
        unsafe { ids.read_u32(12) },
        [383, 0, 1, 2, 3, 4, 383, 0, 1, 2, 3, 4]
    );
    assert_eq!(unsafe { weights.read_f32(0, 12) }, [0.5; 12]);
    assert!(
        unsafe { weights.read_f32(12, 4) }
            .iter()
            .all(|x| x.is_nan())
    );
    // This adversarial pair reverses the order if sigmoid is applied BEFORE
    // selection bias. Membership depends on logits+bias, never sigmoid+bias.
    let logits = floats(&d, (0..EXPERTS).map(|i| if i == 0 { 10. } else { -10. }));
    let bias = floats(&d, (0..EXPERTS).map(|i| if i == 383 { 2. } else { 0. }));
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "kolibri_route",
        &[&logits, &bias, &ids, &weights],
        &[],
        [1, 1, 1],
        32,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { ids.read_u32(2) }, [0, 383]);
}

#[test]
fn compaction_covers_expert_383_and_keeps_stable_entry_order() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let ids = ints(&d, (0..102).map(|i| if i % 2 == 0 { 383 } else { 257 }));
    let lists = d.alloc(EXPERTS * 102 * 4).unwrap();
    let counts = d.alloc(EXPERTS * 4).unwrap();
    let tiles = ints(&d, [u32::MAX; 64]);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "moe_align",
        &[&ids, &lists, &counts],
        &[102],
        [EXPERTS, 1, 1],
        256,
    );
    cmd.dispatch("kolibri_tiles", &[&counts, &tiles], &[], [1, 1, 1], 512);
    cmd.finish().unwrap();
    let tiles = unsafe { tiles.read_u32(20) };
    assert_eq!(
        tiles[..17],
        [
            8, 257, 0, 257, 16, 257, 32, 257, 48, 383, 0, 383, 16, 383, 32, 383, 48
        ]
    );
    assert_eq!(tiles[17], u32::MAX);
    let lists = unsafe { lists.read_u32(EXPERTS * 102) };
    assert_eq!(
        &lists[383 * 102..383 * 102 + 51],
        &(0..102).step_by(2).collect::<Vec<_>>()
    );
}

// Storage/indexing fixture: all codes equal one, scale=1, bias=0. The
// oracle is the separately implemented GPU grouped contraction, not a CPU
// inference path. Ragged rows exercise the compaction tail and guards.
#[test]
fn packed_group64_expert_vector_and_grouped_agree() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let (k, n, rows) = (256, 64, 17);
    let elements = EXPERTS * k * n;
    let mut bytes = vec![0x11; elements / 2];
    bytes.extend((0..elements / 64).flat_map(|_| 0x3f80u16.to_le_bytes()));
    bytes.extend(vec![0; elements / 64 * 2]);
    let w = d.upload(&bytes).unwrap();
    let x = floats(&d, (0..rows * k).map(|i| (i % 4) as f32 * 0.25));
    let ids = ints(&d, (0..rows * ACTIVE).map(|i| (i % 3 + 381) as u32));
    let lists = d.alloc(EXPERTS * rows * ACTIVE * 4).unwrap();
    let counts = d.alloc(EXPERTS * 4).unwrap();
    let cap = (rows * ACTIVE).div_ceil(16) + EXPERTS;
    let tiles = d.alloc((1 + 2 * cap) * 4).unwrap();
    let a = floats(&d, vec![f32::NAN; rows * ACTIVE * n + 32]);
    let b = floats(&d, vec![f32::NAN; rows * ACTIVE * n + 32]);
    let p = [k as u32, n as u32, rows as u32, 0];
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "kolibri_expert_mv",
        &[&w, &x, &ids, &a],
        &p,
        [n.div_ceil(16), rows * ACTIVE, 1],
        128,
    );
    cmd.dispatch(
        "moe_align",
        &[&ids, &lists, &counts],
        &[(rows * ACTIVE) as u32],
        [EXPERTS, 1, 1],
        256,
    );
    cmd.dispatch("kolibri_tiles", &[&counts, &tiles], &[], [1, 1, 1], 512);
    cmd.dispatch(
        "kolibri_grouped",
        &[&w, &x, &lists, &counts, &tiles, &b],
        &p,
        [n.div_ceil(32), cap, 1],
        128,
    );
    cmd.finish().unwrap();
    assert_eq!(unsafe { a.read_f32(0, rows * ACTIVE * n) }, unsafe {
        b.read_f32(0, rows * ACTIVE * n)
    });
    assert!(
        unsafe { b.read_f32(rows * ACTIVE * n, 32) }
            .iter()
            .all(|x| x.is_nan())
    );
}

#[test]
fn bf16_paged_attention_keeps_exactly_513_keys() {
    let d = MetalDevice::new(Some(64 << 20)).unwrap();
    let length: usize = 514;
    let capacity = length.next_multiple_of(16);
    let q = floats(&d, [0.; HEADS * HEAD_DIM]);
    let keys = d.upload(&vec![0; capacity * KVWIDTH * 2]).unwrap();
    // Position zero MUST be masked; position one MUST still be visible.
    // Uniform logits turn the known-value fixture into a boundary check.
    let values = d
        .upload(
            &(0..capacity * KVWIDTH)
                .flat_map(|i| {
                    half::bf16::from_f32(match i / KVWIDTH {
                        0 => 4096.,
                        1 => 257.,
                        _ => 1.,
                    })
                    .to_bits()
                    .to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let pages = ints(&d, (0..capacity / 16).map(|i| i as u32));
    let meta = ints(&d, [0, (length - 1) as u32]);
    let rows = ints(&d, [0]);
    let tiles = ints(&d, [0, 1]);
    let parts = d.alloc(HEADS * SPLITS * (HEAD_DIM + 2) * 4).unwrap();
    let decode = d.alloc(HEADS * HEAD_DIM * 4).unwrap();
    let prefill = d.alloc(HEADS * HEAD_DIM * 4).unwrap();
    for window in [0, WINDOW, WINDOW - 1] {
        let p = [
            HEADS as u32,
            KV_HEADS as u32,
            (capacity / 16) as u32,
            window as u32,
            0,
            SPLITS as u32,
        ];
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "kolibri_decode",
            &[&q, &keys, &values, &meta, &pages, &rows, &parts],
            &p,
            [KV_HEADS, 1, SPLITS],
            128,
        );
        cmd.dispatch(
            "gemma_merge",
            &[&parts, &decode, &rows],
            &[HEADS as u32, SPLITS as u32, HEAD_DIM as u32],
            [HEADS, 1, 1],
            32,
        );
        cmd.dispatch(
            "kolibri_round",
            &[&decode],
            &[(HEADS * HEAD_DIM) as u32],
            [(HEADS * HEAD_DIM).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "kolibri_prefill",
            &[&q, &keys, &values, &meta, &pages, &prefill, &tiles],
            &p,
            [HEADS, 1, 1],
            128,
        );
        cmd.dispatch(
            "kolibri_round",
            &[&prefill],
            &[(HEADS * HEAD_DIM) as u32],
            [(HEADS * HEAD_DIM).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let expected = half::bf16::from_f32(match window {
            0 => (4096. + 256. + 512.) / 514.,
            WINDOW => 768. / 513.,
            _ => 1.,
        })
        .to_f32();
        assert!(
            unsafe { decode.read_f32(0, HEADS * HEAD_DIM) }
                .iter()
                .all(|&v| v == expected)
        );
        assert!(
            unsafe { prefill.read_f32(0, HEADS * HEAD_DIM) }
                .iter()
                .all(|&v| v == expected),
            "window={window}"
        );
    }
}

#[test]
fn packed_staging_matches_scalar_gpu_for_ragged_experts_and_affine_biases() {
    let d = MetalDevice::new(Some(128 << 20)).unwrap();
    let (k, n, rows) = (256usize, 64usize, 17usize);
    let count = EXPERTS * k * n;
    // Nonconstant signed affine groups and all nibble positions. This is
    // fixture construction only; both contractions execute on GPU.
    let mut bytes = (0..count / 2)
        .map(|i| i.wrapping_mul(73).wrapping_add(i / 31) as u8)
        .collect::<Vec<_>>();
    for is_bias in [false, true] {
        bytes.extend((0..count / 64).flat_map(|i| {
            let value = if is_bias {
                (i % 23) as f32 * 0.0625 - 0.6875
            } else {
                ((i % 11) as f32 - 5.) * 0.03125
            };
            half::bf16::from_f32(value).to_bits().to_le_bytes()
        }));
    }
    let w = d.upload(&bytes).unwrap();
    let x = floats(
        &d,
        (0..rows * ACTIVE * k).map(|i| ((i * 17 % 97) as f32 - 48.) * 0.015625),
    );
    let ids = ints(&d, (0..rows * ACTIVE).map(|i| [0, 1, 383, 257][i % 4]));
    let lists = d.alloc(EXPERTS * rows * ACTIVE * 4).unwrap();
    let counts = d.alloc(EXPERTS * 4).unwrap();
    let cap = (rows * ACTIVE).div_ceil(16) + EXPERTS;
    let tiles = d.alloc((1 + 2 * cap) * 4).unwrap();
    for per_entry in [0, 1] {
        let a = floats(&d, vec![f32::NAN; rows * ACTIVE * n + 32]);
        let b = floats(&d, vec![f32::NAN; rows * ACTIVE * n + 32]);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "moe_align",
            &[&ids, &lists, &counts],
            &[(rows * ACTIVE) as u32],
            [EXPERTS, 1, 1],
            256,
        );
        cmd.dispatch("kolibri_tiles", &[&counts, &tiles], &[], [1, 1, 1], 512);
        for (name, y) in [("kolibri_grouped_scalar", &a), ("kolibri_grouped", &b)] {
            cmd.dispatch(
                name,
                &[&w, &x, &lists, &counts, &tiles, y],
                &[k as u32, n as u32, rows as u32, per_entry],
                [n.div_ceil(32), cap, 1],
                128,
            );
        }
        cmd.finish().unwrap();
        // SAFETY: both outputs completed; the guard remains within allocations.
        let (scalar, packed) = unsafe {
            (
                a.read_f32(0, rows * ACTIVE * n),
                b.read_f32(0, rows * ACTIVE * n),
            )
        };
        assert!(packed.iter().all(|v| v.is_finite()));
        assert_eq!(scalar, packed);
        assert!(
            unsafe { b.read_f32(rows * ACTIVE * n, 32) }
                .iter()
                .all(|v| v.is_nan())
        );
    }
}

#[test]
#[ignore = "requires PADDOCK_KOLIBRI_MLX, ~50 GiB model memory"]
fn checkpoint_generation() {
    let path = std::env::var("PADDOCK_KOLIBRI_MLX").expect("model path");
    let path = std::path::Path::new(&path);
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(path).unwrap();
    let tokens = tokenizer.encode("The capital of Germany is").unwrap();
    let mut model = Kolibri::load(path, 2048, 4, None).unwrap();
    let mut logits = model.forward_prefill(0, &tokens).unwrap();
    let mut generated = Vec::new();
    for _ in 0..32 {
        assert!(logits.iter().all(|v| v.is_finite()));
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        generated.push(token);
        logits = model.forward(token).unwrap();
    }
    eprintln!("KOLIBRI_GREEDY {generated:?}");
    eprintln!("KOLIBRI_TEXT {:?}", tokenizer.decode(&generated, false));
}
