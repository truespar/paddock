//! Analytic/layout and GPU metamorphic tests. These do not replace the
//! independent Meta checkpoint gate or run an in-house CPU reference model.
use super::text::TextWorkspace;
use super::*;
use objc2_metal::MTLBuffer;
use paddock_models::sam3::Sam3TextConfig;

fn halves(d: &MetalDevice, values: &[f32]) -> Buffer {
    d.upload(
        &values
            .iter()
            .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
fn read_half(b: &Buffer) -> Vec<f32> {
    tests::bytes(b)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| half::f16::from_le_bytes(*x).to_f32())
        .collect()
}

#[test]
fn text_workspace_is_exact_bounded_and_released() {
    for cap in [0, 33, usize::MAX] {
        assert!(Sam3Text::resident_bytes_required(cap).is_err());
    }
    for cap in [1, 4] {
        let bytes = TextWorkspace::required_bytes(cap).unwrap();
        let d = MetalDevice::new(Some(bytes)).unwrap();
        let ws = TextWorkspace::new(&d, cap).unwrap();
        assert_eq!(d.allocated_bytes(), bytes);
        assert!(d.alloc(1).is_err());
        drop(ws);
        assert_eq!(d.allocated_bytes(), 0);
    }
}

#[test]
fn text_embedding_preserves_f32_and_resets_positions_per_prompt() {
    let d = MetalDevice::new(None).unwrap();
    let token = upload(&d, &[vec![1.0001; 1024], vec![-2.0002; 1024]].concat()).unwrap();
    let position_values: Vec<_> = (0..32 * 1024)
        .map(|i| (i / 1024) as f32 * 0.03125)
        .collect();
    let position = upload(&d, &position_values).unwrap();
    let ids: Vec<u32> = (0..64).map(|i| (i / 32) as u32).collect();
    let ids_gpu = d
        .upload(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    let x = d.alloc(64 * 1024 * 4).unwrap();
    let c = d.begin().unwrap();
    point(
        &c,
        "sam3_text_embed",
        &[&token, &position, &ids_gpu, &x],
        &[64],
        64 * 1024,
    );
    c.finish().unwrap();
    let got = unsafe { x.read_f32(0, 64 * 1024) };
    for (i, value) in got.iter().enumerate() {
        let expected =
            if i < 32 * 1024 { 1.0001f32 } else { -2.0002f32 } + position_values[i % (32 * 1024)];
        assert_eq!(value.to_bits(), expected.to_bits(), "embedding {i}");
    }
    assert_ne!(got[0], half::f16::from_f32(got[0]).to_f32());
}

#[test]
fn text_qkv_rounds_before_bias_without_vision_permutation() {
    let d = MetalDevice::new(None).unwrap();
    let values: Vec<_> = (0..2 * 3072)
        .map(|i| {
            if i % 3072 < 2048 {
                1.0006
            } else {
                (i % 1024) as f32
            }
        })
        .collect();
    let input = halves(&d, &values);
    let bias = upload(&d, &vec![-0.0006; 3072]).unwrap();
    let q = d.alloc(2 * 1024 * 2).unwrap();
    let k = d.alloc(q.len()).unwrap();
    let v = d.alloc(q.len()).unwrap();
    let c = d.begin().unwrap();
    point(
        &c,
        "sam3_text_qkv",
        &[&input, &bias, &q, &k, &v],
        &[2],
        2 * 1024,
    );
    c.finish().unwrap();
    let half = |x| half::f16::from_f32(x).to_f32();
    for (which, out, scale) in [(0, &q, 0.125), (1, &k, 1.), (2, &v, 1.)] {
        for (i, got) in read_half(out).iter().enumerate() {
            let x = values[(i / 1024) * 3072 + which * 1024 + i % 1024];
            assert_eq!(*got, half((half(x) - 0.0006) * scale), "{which} / {i}");
        }
    }
}

fn attention(d: &MetalDevice, q: &Buffer, k: &Buffer, v: &Buffer, out: &Buffer, prompts: usize) {
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_text_attention",
        &[q, k, v, out],
        &[prompts as u32],
        [prompts * 16, 1, 1],
        256,
    );
    c.finish().unwrap();
}

#[test]
fn text_attention_is_causal_and_isolates_heads_and_prompts() {
    let d = MetalDevice::new(None).unwrap();
    let n = 4 * 32 * 1024;
    let zero = halves(&d, &vec![0.; n]);
    let values: Vec<_> = (0..n)
        .map(|i| (i / (32 * 1024) * 64 + i % 1024 / 64 * 2 + i / 1024 % 32) as f32)
        .collect();
    let v = halves(&d, &values);
    let out = d.alloc(n * 2).unwrap();
    attention(&d, &zero, &zero, &v, &out, 4);
    for (i, got) in read_half(&out).iter().enumerate() {
        // With equal logits the causal average of 0..t is exactly t/2.
        let expected =
            (i / (32 * 1024) * 64 + i % 1024 / 64 * 2) as f32 + (i / 1024 % 32) as f32 * 0.5;
        assert_eq!(*got, expected, "attention {i}");
    }
    let original = tests::bytes(&out);
    let changed: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, v)| if i / 1024 % 32 == 31 { 8192. } else { *v })
        .collect();
    let changed = halves(&d, &changed);
    attention(&d, &zero, &zero, &changed, &out, 4);
    let after = tests::bytes(&out);
    for prompt in 0..4 {
        let a = prompt * 32 * 1024 * 2;
        assert_eq!(
            &after[a..a + 31 * 1024 * 2],
            &original[a..a + 31 * 1024 * 2],
            "future row leaked"
        );
    }
}

#[test]
fn text_attention_nonuniform_softmax_is_stable_at_large_logits() {
    let d = MetalDevice::new(None).unwrap();
    let mut q = vec![0.; 32 * 1024];
    let mut k = q.clone();
    let mut v = q.clone();
    for row in 0..32 {
        for head in 0..16 {
            q[row * 1024 + head * 64] = 1.;
            k[row * 1024 + head * 64] = if row == 0 { 4100. } else { 4096. };
            for col in 0..64 {
                v[row * 1024 + head * 64 + col] = if row == 0 { 1. } else { 0. };
            }
        }
    }
    let (q, k, v) = (halves(&d, &q), halves(&d, &k), halves(&d, &v));
    let out = d.alloc(32 * 1024 * 2).unwrap();
    attention(&d, &q, &k, &v, &out, 1);
    for (i, value) in read_half(&out).iter().enumerate() {
        // One e^4 weight and t unit weights; an analytic softmax identity,
        // not a reference implementation of attention on the host.
        let expected = half::f16::from_f32(1. / (1. + (i / 1024) as f32 * (-4f32).exp())).to_f32();
        assert_eq!(*value, expected, "softmax {i}");
    }
}

#[test]
fn text_erf_gelu_epilogue_differs_from_vision_tanh_and_keeps_large_activations() {
    let d = MetalDevice::new(None).unwrap();
    let mut eye = vec![0.; 64 * 64];
    for i in 0..64 {
        eye[i * 64 + i] = 1.;
    }
    let m = Matrix {
        w: halves(&d, &eye),
        k: 64,
        n: 64,
    };
    let xs = [
        0.1, -0.1, 0.5, -0.5, 1., -1., 2., -2., 3., -3., 8., -8., 1062., -1062.,
    ];
    let x: Vec<_> = (0..64).map(|i| xs[i % xs.len()]).collect();
    let x_gpu = halves(&d, &x);
    let bias = upload(&d, &vec![0.; 64]).unwrap();
    let out = d.alloc(64 * 2).unwrap();
    let tanh = d.alloc(64 * 2).unwrap();
    let series_values: Vec<_> = x.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect();
    let series = upload(&d, &series_values).unwrap();
    let c = d.begin().unwrap();
    vision::mm(&c, &m, &bias, &x_gpu, &out, 1, 5);
    vision::mm(&c, &m, &bias, &x_gpu, &tanh, 1, 3);
    point(&c, "mv_gelu_series_check", &[&series], &[64], 64);
    c.finish().unwrap();
    let got = read_half(&out);
    let reference = unsafe { series.read_f32(0, 64) };
    for (g, r) in got.iter().zip(reference) {
        assert!(
            (g - r).abs() <= 0.0005 * r.abs().max(0.001) + 2e-7,
            "{g} vs {r}"
        );
    }
    assert_ne!(tests::bytes(&out), tests::bytes(&tanh));
    assert_eq!(got[12], 1062.);
    assert_eq!(got[13], 0.);
}

// A nontrivial two-block GPU graph with official widths, but synthetic
// weights and a tiny vocabulary. It tests scheduling/arithmetic invariance,
// not checkpoint accuracy. Loader tests still insist on official geometry.
fn synthetic(cap: usize) -> Sam3Text {
    let d = MetalDevice::new(None).unwrap();
    let norm = || Norm {
        w: upload(&d, &vec![1.; 1024]).unwrap(),
        b: upload(&d, &vec![0.01; 1024]).unwrap(),
    };
    let conv = |k, n, scale| {
        let mut w = vec![0.; k * n];
        for o in 0..n {
            w[o * k + (o * 17 + 3) % k] = scale;
            w[o * k + (o * 13 + 19) % k] += scale * 0.5;
        }
        Conv {
            w: Matrix {
                w: halves(&d, &w),
                k,
                n,
            },
            b: upload(
                &d,
                &(0..n).map(|i| (i % 7) as f32 * 0.01).collect::<Vec<_>>(),
            )
            .unwrap(),
        }
    };
    let token_values: Vec<_> = (0..8 * 1024)
        .map(|i| ((i * 13 + i / 1024 * 7) % 97) as f32 * 0.0123 - 0.5)
        .collect();
    let position_values: Vec<_> = (0..32 * 1024)
        .map(|i| ((i * 7 + i / 1024) % 31) as f32 * 0.0011)
        .collect();
    let token = upload(&d, &token_values).unwrap();
    let position = upload(&d, &position_values).unwrap();
    let blocks = (0..2)
        .map(|_| Block {
            n1: norm(),
            n2: norm(),
            qkv: conv(1024, 3072, 0.25),
            out: conv(1024, 1024, 0.125),
            up: conv(1024, 4096, 0.5),
            down: conv(4096, 1024, 0.25),
        })
        .collect();
    let final_norm = norm();
    let resizer = conv(1024, 256, 0.25);
    let weight_bytes = d.allocated_bytes();
    let ws = TextWorkspace::new(&d, cap).unwrap();
    Sam3Text {
        device: d,
        cfg: Sam3TextConfig {
            vocab: 8,
            context: 32,
            hidden: 1024,
            n_layer: 2,
            n_heads: 16,
            intermediate: 4096,
            d_model: 256,
        },
        token,
        position,
        blocks,
        final_norm,
        resizer,
        ws,
        weight_bytes,
        encoded_prompts: 0,
    }
}

#[test]
fn text_graph_replays_across_batches_and_invalidates_failed_requests() {
    let mut model = synthetic(4);
    let bytes = model.device.allocated_bytes();
    assert!(model.read_features().is_err());
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|p| (0..32).map(|i| (i * 3 + p) % 8).collect())
        .collect();
    let mut singles = Vec::new();
    for p in &prompts {
        model.encode(p, 1).unwrap();
        singles.push(model.read_features().unwrap());
    }
    assert_ne!(singles[0], singles[1]);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for order in [[0, 1, 2, 3], [3, 1, 0, 2], [0, 0, 1, 1]] {
        for count in [4, 2, 1, 3, 4] {
            let ids: Vec<_> = order[..count]
                .iter()
                .flat_map(|&i| prompts[i].iter().copied())
                .collect();
            model.encode(&ids, count).unwrap();
            let batch = model.read_features().unwrap();
            assert_eq!(batch.len(), count * 32 * 256);
            for (i, &index) in order[..count].iter().enumerate() {
                assert_eq!(
                    bits(&batch[i * 32 * 256..(i + 1) * 32 * 256]),
                    bits(&singles[index]),
                    "batch {count}, row {i}"
                );
            }
            assert_eq!(model.device.allocated_bytes(), bytes);
        }
    }
    for (ids, count) in [
        (vec![], 0),
        (vec![0; 160], 5),
        (vec![0; 31], 1),
        (vec![8; 32], 1),
        (vec![u32::MAX; 32], 1),
    ] {
        assert!(model.encode(&ids, count).is_err());
        assert!(model.read_features().is_err());
        assert_eq!(model.device.allocated_bytes(), bytes);
        model.encode(&prompts[0], 1).unwrap();
        assert_eq!(bits(&model.read_features().unwrap()), bits(&singles[0]));
    }
    // The exact same request remains reusable after a failed request, but
    // an actual numerical failure must also refuse to publish stale output.
    unsafe {
        *model.position.raw.contents().as_ptr().cast::<f32>() = f32::NAN;
    }
    assert!(model.encode(&prompts[0], 1).is_err());
    assert!(model.read_features().is_err());
    drop(model);
}

#[test]
fn text_loader_refuses_nonofficial_geometry() {
    let good = Sam3TextConfig {
        vocab: 49408,
        context: 32,
        hidden: 1024,
        n_layer: 24,
        n_heads: 16,
        intermediate: 4096,
        d_model: 256,
    };
    text_load::validate_geometry(&good).unwrap();
    for field in 0..7 {
        let mut c = good.clone();
        *match field {
            0 => &mut c.vocab,
            1 => &mut c.context,
            2 => &mut c.hidden,
            3 => &mut c.n_layer,
            4 => &mut c.n_heads,
            5 => &mut c.intermediate,
            _ => &mut c.d_model,
        } = 0;
        assert!(text_load::validate_geometry(&c).is_err());
    }
}
