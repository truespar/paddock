//! Output-tile elections may improve occupancy, never change the arithmetic.
use super::*;

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
