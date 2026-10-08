//! Output-tile elections may improve occupancy, never change the arithmetic.
use super::*;

#[test]
#[ignore = "real MLX8 checkpoint; request projection batch contract and guards"]
fn embeddinggemma2_request_projection_guards() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    let lengths = [2, 12, 13, 31, 32, 33, 63, 64, 65, 128, 129];
    let seqs: Vec<_> = lengths.iter().map(|&n| vec![2; n]).collect();
    let rows: usize = lengths.iter().sum();
    for w in [
        &m.ple,
        &m.output,
        &m.layers[0].q,
        &m.layers[0].k,
        &m.layers[0].down,
    ] {
        assert_eq!(w.ty, 0x108);
        let values: Vec<f32> = (0..rows * w.k)
            .map(|i| {
                [0.00390625f32, -0.75, 0.5, 1.25, -1., 0.03125, 0., 0.125][(i * 7 + i / 19) % 8]
            })
            .collect();
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x = m.device.upload(&bytes).unwrap();
        let out = m.device.alloc((rows * w.n + 64) * 4).unwrap();
        let sentinel = vec![0xdeadbeefu32; rows * w.n + 64];
        unsafe {
            out.write_u32(&sentinel);
        }
        let c = m.device.begin().unwrap();
        m.text_linear(&c, w, &x, &out, &seqs);
        c.finish().unwrap();
        let actual = unsafe { out.read_u32(rows * w.n + 64) };
        assert_eq!(&actual[rows * w.n..], &sentinel[rows * w.n..]);
        let mut first = 0;
        for seq in &seqs {
            let x = m
                .device
                .upload(&bytes[first * w.k * 4..(first + seq.len()) * w.k * 4])
                .unwrap();
            let out = m.device.alloc((seq.len() * w.n + 64) * 4).unwrap();
            unsafe {
                out.write_u32(&sentinel[..seq.len() * w.n + 64]);
            }
            let c = m.device.begin().unwrap();
            m.text_linear(&c, w, &x, &out, std::slice::from_ref(seq));
            c.finish().unwrap();
            let single = unsafe { out.read_u32(seq.len() * w.n + 64) };
            assert_eq!(
                &single[..seq.len() * w.n],
                &actual[first * w.n..(first + seq.len()) * w.n],
                "request projection changed: K={}, N={}, M={}",
                w.k,
                w.n,
                seq.len()
            );
            assert!(single[seq.len() * w.n..].iter().all(|&v| v == 0xdeadbeef));
            first += seq.len();
        }
    }
}

#[test]
#[ignore = "same-checkpoint first-layer captures; PADDOCK_EG2_STAGES"]
fn embeddinggemma2_projection_stages() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let stages = std::env::var("PADDOCK_EG2_STAGES").unwrap();
    let m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    let f = paddock_models::safetensors::SafetensorsFile::open(Path::new(&stages)).unwrap();
    let index = std::env::var("PADDOCK_EG2_LAYER")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    let l = &m.layers[index];
    let mut failures = Vec::new();
    for (name, input, w) in [
        ("q", "pre", &l.q),
        ("k", "pre", &l.k),
        ("v", "pre", &l.v),
        ("o", "attn", &l.o),
        ("ff_down", "geglu", &l.down),
        ("ple_gate", "ff_residual", &l.ple_gate),
        ("ple_out", "ple_gelu", &l.ple_out),
        ("ff_gate", "ff_pre", &l.gate),
        ("ff_up", "ff_pre", &l.up),
        ("ple", "embed", &m.ple),
    ] {
        if name == "ple" && index != 0 {
            continue;
        }
        let (info, bytes) = f.bytes(input).unwrap();
        let rows = info.shape[0];
        let x = m.device.upload(bytes).unwrap();
        let y = m.device.alloc(rows * w.n * 4).unwrap();
        let c = m.device.begin().unwrap();
        m.text_linear(&c, w, &x, &y, &[vec![0; rows]]);
        c.finish().unwrap();
        let actual = unsafe { y.read_f32(0, rows * w.n) };
        let expected: Vec<_> = f
            .bytes(name)
            .unwrap()
            .1
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let diff = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{name}: {diff}/{} max={max}", actual.len());
        if diff > 0 {
            failures.push(name);
        }
    }
    assert!(failures.is_empty(), "projection differences: {failures:?}");
}

#[test]
#[ignore = "real checkpoint; PADDOCK_EG2_MODEL"]
fn embeddinggemma2_narrow_and_wide_projections_are_identical() {
    let path = std::env::var("PADDOCK_EG2_MODEL").unwrap();
    let m = EmbeddingGemma2::load(Path::new(&path), 8192, None).unwrap();
    let mut checked = 0;
    for w in [
        &m.ple,
        &m.output,
        &m.layers[0].q,
        &m.layers[5].q,
        &m.layers[0].k,
        &m.layers[0].gate,
        &m.layers[0].down,
    ] {
        let (wide, narrow) = match w.ty {
            0x108 => ("eg2_project_a8", "eg2_project_a8_narrow"),
            8 => ("eg2_project_q8", "eg2_project_q8_narrow"),
            // GGUF retains some projection matrices in a raw dtype; their
            // dispatch has not changed and does not select the narrow tile.
            _ => continue,
        };
        checked += 1;
        for rows in [1usize, 15, 16, 17, 31, 32, 33, 127, 128, 129] {
            let input: Vec<u8> = (0..rows * w.k)
                .flat_map(|i| {
                    // Exact dyadic values with cancellation and uneven groups.
                    [0.00390625f32, -0.75, 0.5, 1.25, -1., 0.03125, 0., 0.125][(i * 7 + i / 19) % 8]
                        .to_le_bytes()
                })
                .collect();
            let x = m.device.upload(&input).unwrap();
            let len = rows * w.n;
            let a = m.device.alloc((len + 64) * 4).unwrap();
            let b = m.device.alloc((len + 64) * 4).unwrap();
            let sentinel = vec![0xdeadbeefu32; len + 64];
            unsafe {
                a.write_u32(&sentinel);
                b.write_u32(&sentinel);
            }
            let c = m.device.begin().unwrap();
            let p = [w.k as u32, w.n as u32, rows as u32];
            c.dispatch(
                wide,
                &[&w.buffer, &x, &a],
                &p,
                [w.n.div_ceil(64), rows.div_ceil(32), 1],
                128,
            );
            c.dispatch(
                narrow,
                &[&w.buffer, &x, &b],
                &p,
                [w.n.div_ceil(16), rows.div_ceil(16), 1],
                128,
            );
            c.finish().unwrap();
            let a = unsafe { a.read_u32(len + 64) };
            let b = unsafe { b.read_u32(len + 64) };
            assert_eq!(a, b, "projection changed: M={rows}, N={}, K={}", w.n, w.k);
            assert_eq!(
                &a[len..],
                &sentinel[len..],
                "projection overwrote row guard"
            );
        }
    }
    assert!(
        checked >= 5,
        "expected the MLX8 or Q8 checkpoint projections"
    );
}
