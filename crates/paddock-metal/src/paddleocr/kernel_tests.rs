use super::*;
use objc2_metal::MTLBuffer;
const E: usize = 1152;
fn halves(b: &Buffer, n: usize) -> Vec<f32> {
    // SAFETY: tests finish every command before inspecting its output.
    unsafe {
        std::slice::from_raw_parts(b.raw.contents().as_ptr().cast::<half::f16>(), n)
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }
}

#[test]
fn padded_attention_matches_independent_gpu_scan() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    // Three KV tiles in the middle image exercise repeated online rescaling
    // and accumulation; both image boundaries remain deliberately ragged.
    let sizes = [37usize, 672, 13];
    let rows: usize = sizes.iter().sum();
    let plane = |seed: usize| {
        d.upload(
            &(0..(rows + 64) * 1280)
                .flat_map(|i| {
                    let v = if i % 80 < 72 && i / 1280 < rows {
                        ((i * seed % 31) as f32 - 15.0) / 64.0
                    } else {
                        0.0
                    };
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (q, k, v) = (plane(3), plane(7), plane(11));
    let mut tiles = Vec::new();
    let mut bounds = Vec::new();
    let mut first = 0;
    for n in sizes {
        for row in (0..n).step_by(32) {
            tiles.extend([
                (first + row) as u32,
                (n - row).min(32) as u32,
                first as u32,
                n as u32,
            ]);
        }
        for _ in 0..n {
            bounds.extend([first as u32, (first + n) as u32]);
        }
        first += n;
    }
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let t = upload(&tiles);
    let b = upload(&bounds);
    let out = d
        .upload(
            &(0..(rows + 32) * E)
                .flat_map(|_| half::f16::NAN.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let reference = d.alloc(rows * E * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "pocr_attention",
        &[&q, &k, &v, &out, &t],
        &[1],
        [16, tiles.len() / 4, 1],
        64,
    );
    cmd.dispatch(
        "vis_attention_check",
        &[&q, &k, &v, &reference, &b],
        &[72, 16, 80],
        [16, rows, 1],
        32,
    );
    cmd.finish().unwrap();
    let actual = halves(&out, rows * E);
    let expected = unsafe { reference.read_f32(0, rows * E) };
    assert!(actual.iter().all(|v| v.is_finite()));
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 0.0001, "GPU attention error {max}");
    let full = d.alloc(rows * E * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "pocr_attention",
        &[&q, &k, &v, &full, &t],
        &[0],
        [16, tiles.len() / 4, 1],
        64,
    );
    cmd.finish().unwrap();
    let actual = unsafe { full.read_f32(0, rows * E) };
    assert!(actual.iter().all(|v| v.is_finite()));
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max < 0.000002,
        "F32 probability/output GPU attention error {max}"
    );
    assert!(
        halves(&out, (rows + 32) * E)[rows * E..]
            .iter()
            .all(|v| v.is_nan()),
        "tail guard"
    );
}
