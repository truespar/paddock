//! Compare the short output projection to the original complete GPU draft.
//! The trained noncausal backbone is never shortened, including near the
//! attention-window boundary and when a previously long proposal is reused.
use super::*;

fn equal_bits(actual: &[u32], expected: &[u32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    let different = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    assert_eq!(different, 0, "{label}");
}

#[test]
fn dflash_selector_prefix_respects_block_stride_and_output_guards() {
    let d = MetalDevice::new(None).unwrap();
    let block = 8usize;
    let blocks = 2usize;
    let rows = block * blocks;
    let top_data: Vec<u32> = (0..rows * 16)
        .flat_map(|i| [(i % 47) as u32, ((i % 17) as f32 / 19.).to_bits()])
        .collect();
    let upload_u = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let top = upload_u(&top_data);
    let codebook = |salt: usize| {
        d.upload(
            &(0..47 * 256)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * salt % 239) as f32 - 119.) / 997.)
                        .to_bits()
                        .to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let pred = codebook(37);
    let succ = codebook(61);
    let h = d
        .upload(
            &(0..rows * 256)
                .flat_map(|i| (((i * 13 % 97) as f32 - 48.) / 211.).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let tokens = upload_u(&(0..rows as u32).collect::<Vec<_>>());
    let out = d.alloc((rows + 16) * 4).unwrap();
    let buffers = [&top, &pred, &succ, &h, &tokens, &out];
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "df_select",
        &buffers,
        &[block as u32, 30, 30],
        [blocks, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    let expected = unsafe { out.read_u32(rows) };
    for selected in 1..=block {
        unsafe {
            out.write_u32(&vec![u32::MAX; rows + 16]);
        }
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "df_select_prefix",
            &buffers,
            &[block as u32, 30, 30, selected as u32],
            [blocks, 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let got = unsafe { out.read_u32(rows + 16) };
        for b in 0..blocks {
            assert_eq!(
                &got[b * block..b * block + selected],
                &expected[b * block..b * block + selected]
            );
            assert!(
                got[b * block + selected..(b + 1) * block]
                    .iter()
                    .all(|v| *v == u32::MAX)
            );
        }
        assert!(got[rows..].iter().all(|v| *v == u32::MAX));
    }
}

#[test]
#[ignore = "requires Qwen MLX + DFlash fixtures; hidden/logit/prefix GPU equality"]
fn mlx_dflash_short_head_preserves_hidden_logits_and_selection() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let draft = std::env::var("PADDOCK_METAL_DFLASH_MODEL").unwrap();
    let mut model = Qwen35::load(Path::new(&path), 4096, 4, None).unwrap();
    model.attach_dflash(Path::new(&draft)).unwrap();
    assert!(
        model.device.tensor_accelerated(),
        "M5 election qualification"
    );
    for length in [31usize, 511, 2047, 3583] {
        model.reset();
        for entry in &mut model.cache {
            entry.table.clear(&mut model.pool);
            entry.history.clear();
        }
        let prompt: Vec<u32> = (1000..1000 + length as u32).collect();
        model.forward_prefill(3, &prompt).unwrap();
        for k in [7usize, 1, 3, 2, 1, 7] {
            FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.set(true));
            let full = model.dflash_draft(&[(3, 54321)], k).unwrap().unwrap();
            assert_eq!(full[0].len(), 7);
            let d = model.dflash.as_ref().unwrap();
            let keep = if k <= 3 { 4 } else { 8 };
            // SAFETY: dflash_draft synchronizes its GPU completion.
            let hidden = unsafe { d.x.read_u32(8 * model.width) };
            let normalized = unsafe { d.norm.read_u32(8 * model.width) };
            let logits = unsafe { d.logits.read_u32(keep * model.vocab) };
            let candidates = unsafe { d.top.read_u32(keep * 32) };
            let history = model.slots[3].history.clone();
            // Poison output storage to detect accidental suffix reads/writes.
            unsafe {
                d.logits.write_u32(&vec![u32::MAX; 8 * model.vocab]);
                d.top.write_u32(&[u32::MAX; 8 * 32]);
                d.out.write_u32(&[u32::MAX; 8]);
            }
            FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.set(false));
            let got = model.dflash_draft(&[(3, 54321)], k).unwrap().unwrap();
            assert_eq!(got[0], full[0][..k.min(7)]);
            assert_eq!(model.slots[3].history, history);
            let d = model.dflash.as_ref().unwrap();
            // SAFETY: candidate completion has also been awaited.
            equal_bits(&unsafe { d.x.read_u32(8 * model.width) }, &hidden, "hidden");
            equal_bits(
                &unsafe { d.norm.read_u32(8 * model.width) },
                &normalized,
                "norm",
            );
            equal_bits(
                &unsafe { d.logits.read_u32(keep * model.vocab) },
                &logits,
                "logits",
            );
            equal_bits(&unsafe { d.top.read_u32(keep * 32) }, &candidates, "top16");
            if k <= 3 {
                assert!(
                    unsafe { d.logits.read_u32(8 * model.vocab) }[keep * model.vocab..]
                        .iter()
                        .all(|v| *v == u32::MAX)
                );
                assert!(
                    unsafe { d.top.read_u32(8 * 32) }[keep * 32..]
                        .iter()
                        .all(|v| *v == u32::MAX)
                );
                assert!(
                    unsafe { d.out.read_u32(8) }[k + 1..]
                        .iter()
                        .all(|v| *v == u32::MAX)
                );
            }
        }
        // More than one active drafter retains its existing head geometry.
        model.forward_prefill(0, &prompt).unwrap();
        FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.set(true));
        let expected = model.dflash_draft(&[(3, 54321), (0, 12345)], 3).unwrap();
        FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.set(false));
        assert_eq!(
            model.dflash_draft(&[(3, 54321), (0, 12345)], 3).unwrap(),
            expected
        );
    }
    assert!(model.dflash_draft(&[(0, 1)], 0).unwrap().is_none());
    assert!(model.dflash_draft(&[(4, 1)], 1).unwrap().is_none());
    assert!(model.dflash_draft(&[(0, 1), (0, 1)], 1).unwrap().is_none());
    model.reset();
    assert!(model.dflash_draft(&[(0, 1)], 1).unwrap().is_none());
}
