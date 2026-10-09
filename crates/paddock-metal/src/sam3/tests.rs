use super::*;
use objc2_metal::MTLBuffer;

pub(super) fn bytes(b: &Buffer) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(b.raw.contents().as_ptr().cast(), b.len()).to_vec() }
}

#[test]
fn sam3_norm_keeps_f32_residuals_and_ln_pre_boundary() {
    let d = MetalDevice::new(None).unwrap();
    // Constant rows have zero variance: the exact expected affine output
    // is beta. Vary beta by lane, with a value that cannot survive F16.
    let x = upload(&d, &vec![7.; 2 * 1024]).unwrap();
    let add = d.upload(&vec![0; 2 * 1024 * 2]).unwrap();
    let bias = upload(&d, &vec![0.25; 1024]).unwrap();
    let weight = upload(&d, &vec![2.; 1024]).unwrap();
    let beta: Vec<_> = (0..1024)
        .map(|i| if i % 2 == 0 { 1.0001f32 } else { -0.5 })
        .collect();
    let beta_gpu = upload(&d, &beta).unwrap();
    let out = d.alloc(2 * 1024 * 2).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_norm",
        &[&x, &add, &bias, &weight, &beta_gpu, &out],
        &[1024, 1],
        [2, 1, 1],
        256,
    );
    c.finish().unwrap();
    assert!(unsafe { x.read_f32(0, 2048) }.iter().all(|x| *x == 7.25));
    let expected: Vec<_> = (0..2048)
        .flat_map(|i| half::f16::from_f32(beta[i % 1024]).to_le_bytes())
        .collect();
    assert_eq!(bytes(&out), expected);
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_norm",
        &[&x, &add, &bias, &weight, &beta_gpu, &x],
        &[1024, 2],
        [2, 1, 1],
        256,
    );
    c.finish().unwrap();
    assert_eq!(
        unsafe { x.read_f32(0, 2048) },
        beta.repeat(2),
        "ln_pre rounded away F32 state"
    );
}

#[test]
fn sam3_convolution_checks_every_tap_and_padding() {
    let d = MetalDevice::new(None).unwrap();
    let side = 7usize;
    let channels = 256;
    let mut weights = vec![0u16; 9 * channels * channels];
    // Each output channel chooses a different tap and input channel. This
    // tests all 9 taps, every border and tap/channel permutation exactly.
    for o in 0..channels {
        weights[(o * 9 + o % 9) * channels + (o + 17) % channels] = 0x3c00;
    }
    let w = d
        .upload(
            &weights
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let values: Vec<_> = (0..side * side * channels)
        .map(|i| half::f16::from_f32((i % 31) as f32 - 15.))
        .collect();
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let bias = upload(&d, &vec![1.; channels]).unwrap();
    let out = d.alloc(values.len() * 4).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_conv3",
        &[&w, &x, &out, &bias],
        &[side as u32, channels as u32],
        [4, (side * side).div_ceil(32), 1],
        128,
    );
    c.finish().unwrap();
    let got = unsafe { out.read_f32(0, values.len()) };
    for r in 0..side * side {
        for o in 0..channels {
            let y = (r / side) as isize + (o % 9 / 3) as isize - 1;
            let x = (r % side) as isize + (o % 3) as isize - 1;
            let expected = if (0..side as isize).contains(&x) && (0..side as isize).contains(&y) {
                values[(y as usize * side + x as usize) * channels + (o + 17) % channels].to_f32()
                    + 1.
            } else {
                1.
            };
            assert_eq!(got[r * channels + o], expected, "row {r}, channel {o}");
        }
    }
}

#[test]
fn sam3_qkv_bias_precedes_rounding_and_rope_has_no_inplace_race() {
    let d = MetalDevice::new(None).unwrap();
    let mut acc = vec![0.; 2 * 3072];
    let mut bias = vec![0.; 3072];
    for (i, b) in bias.iter_mut().enumerate() {
        *b = if i < 2048 { -0.0006 } else { 3. };
    }
    for r in 0..2 {
        for c in 0..2048 {
            acc[r * 3072 + c] = 1.0006;
        }
    }
    let input = upload(&d, &acc).unwrap();
    let b = upload(&d, &bias).unwrap();
    // First row rotates by 0; second by pi/2 exactly (analytic table).
    let rope: Vec<_> = (0..64)
        .flat_map(|i| if i < 32 { [1., 0.] } else { [0., 1.] })
        .collect();
    let rope = upload(&d, &rope).unwrap();
    let q = d.alloc(2 * 1024 * 2).unwrap();
    let k = d.alloc(q.len()).unwrap();
    let v = d.alloc(q.len()).unwrap();
    let c = d.begin().unwrap();
    point(
        &c,
        "sam3_qkv",
        &[&input, &b, &rope, &q, &k, &v],
        &[2, 2],
        2 * 1024,
    );
    c.finish().unwrap();
    for (out, scale) in [(&q, 0.125f32), (&k, 1.)] {
        let raw = bytes(out);
        for i in 0..2048 {
            let expected = if i >= 1024 && i % 64 < 32 {
                -scale
            } else {
                scale
            };
            assert_eq!(
                u16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into().unwrap()),
                half::f16::from_f32(expected).to_bits()
            );
        }
    }
    assert!(
        bytes(&v)
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes(*b) == 0x4200)
    );
}

#[test]
fn sam3_patch_rounding_and_window_order_are_explicit() {
    let d = MetalDevice::new(None).unwrap();
    // 28x28, four 14x14 windows; black/white/midgrey in each RGB triplet.
    let rgb: [u8; 28 * 28 * 3] = std::array::from_fn(|i| [0, 255, 128][i % 3]);
    let src = d.upload(&rgb).unwrap();
    let out = d.alloc(4 * 592 * 2).unwrap();
    for video in [0, 1] {
        let c = d.begin().unwrap();
        point(
            &c,
            "sam3_patch",
            &[&src, &out],
            &[28, 14, 1, 592, video],
            4 * 592,
        );
        c.finish().unwrap();
        let raw = bytes(&out);
        for row in 0..4 {
            for col in 0..592 {
                let got = u16::from_le_bytes(
                    raw[(row * 592 + col) * 2..(row * 592 + col) * 2 + 2]
                        .try_into()
                        .unwrap(),
                );
                let expected = match col {
                    0..196 => 0xbc00,
                    196..392 => 0x3c00,
                    392..588 => {
                        if video == 0 {
                            0x1c04
                        } else {
                            0x1c00
                        }
                    }
                    _ => 0,
                };
                assert_eq!(got, expected, "video {video}, row {row}, column {col}");
            }
        }
    }
    // The first patch of window 1 must be spatial column 2, not column 1.
    let rgb: Vec<_> = (0..56 * 56 * 3)
        .map(|i| if (i / 3 % 56) / 14 == 2 { 255 } else { 0 })
        .collect();
    let src = d.upload(&rgb).unwrap();
    let out = d.alloc(16 * 592 * 2).unwrap();
    let c = d.begin().unwrap();
    point(
        &c,
        "sam3_patch",
        &[&src, &out],
        &[56, 14, 2, 592, 0],
        16 * 592,
    );
    c.finish().unwrap();
    let raw = bytes(&out);
    assert_eq!(
        u16::from_le_bytes(raw[4 * 592 * 2..4 * 592 * 2 + 2].try_into().unwrap()),
        0x3c00
    );
    assert_eq!(
        u16::from_le_bytes(raw[592 * 2..592 * 2 + 2].try_into().unwrap()),
        0xbc00
    );
}

#[test]
fn sam3_all_uint8_levels_keep_the_cuda_image_video_distinction() {
    let d = MetalDevice::new(None).unwrap();
    let rgb: Vec<_> = (0..256).flat_map(|i| [i as u8; 3]).collect();
    let src = d.upload(&rgb).unwrap();
    let picture = d.alloc(256 * 8 * 2).unwrap();
    let video = d.alloc(picture.len()).unwrap();
    let c = d.begin().unwrap();
    for (out, mode) in [(&picture, 0), (&video, 1)] {
        point(
            &c,
            "sam3_patch",
            &[&src, out],
            &[16, 1, 16, 8, mode],
            256 * 8,
        );
    }
    c.finish().unwrap();
    let image = bytes(&picture);
    let frames = bytes(&video);
    let mut differences = 0;
    for i in 0..256 {
        differences += usize::from(image[i * 16..i * 16 + 2] != frames[i * 16..i * 16 + 2]);
        for raw in [&image, &frames] {
            assert_eq!(&raw[i * 16..i * 16 + 2], &raw[i * 16 + 2..i * 16 + 4]);
            assert_eq!(&raw[i * 16..i * 16 + 2], &raw[i * 16 + 4..i * 16 + 6]);
            assert!(raw[i * 16 + 6..i * 16 + 16].iter().all(|v| *v == 0));
        }
    }
    // Exhaustive CUDA bring-up records this exact count (vit.cuh). This
    // catches accidentally collapsing either path to a shared expression;
    // byte-for-byte comparison of real reference frames is a separate gate.
    assert_eq!(differences, 140);
}

#[test]
fn sam3_matrix_and_convolution_preserve_half_boundaries() {
    let d = MetalDevice::new(None).unwrap();
    // Implicit convolution with a centre identity filter is exactly identity,
    // including row tails, borders and every SIMD group's destination mapping.
    let side = 7;
    let channels = 256;
    let mut weights = vec![0u16; 9 * channels * channels];
    for i in 0..channels {
        weights[(i * 9 + 4) * channels + i] = 0x3c00;
    }
    let weight = d
        .upload(
            &weights
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let values: Vec<_> = (0..side * side * channels)
        .map(|i| half::f16::from_f32((i % 17) as f32 - 8.))
        .collect();
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let bias = upload(&d, &vec![0.; channels]).unwrap();
    let out = d.alloc(values.len() * 4).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_conv3",
        &[&weight, &x, &out, &bias],
        &[side as u32, channels as u32],
        [4, (side * side).div_ceil(32), 1],
        128,
    );
    c.finish().unwrap();
    assert_eq!(
        unsafe { out.read_f32(0, values.len()) },
        values.iter().map(|x| x.to_f32()).collect::<Vec<_>>()
    );
    // Out-of-place F16 matrix landing, ragged row count and output tile.
    let n = 72;
    let k = 64;
    let rows = 3;
    let mut identity = vec![0u16; k * n];
    for i in 0..k {
        identity[i * k + i] = 0x3c00;
    }
    let w = d
        .upload(
            &identity
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = d.alloc(rows * n * 2).unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_mm",
        &[&w, &x, &out, &bias],
        &[k as u32, n as u32, rows as u32, 1],
        [2, 1, 1],
        128,
    );
    c.finish().unwrap();
    let raw = bytes(&out);
    for r in 0..rows {
        for col in 0..n {
            let bits = u16::from_le_bytes(
                raw[(r * n + col) * 2..(r * n + col) * 2 + 2]
                    .try_into()
                    .unwrap(),
            );
            assert_eq!(
                bits,
                if col < k {
                    values[r * k + col].to_bits()
                } else {
                    0
                }
            );
        }
    }
}

#[test]
fn sam3_window_attention_does_not_leak_between_pictures_or_windows() {
    let d = MetalDevice::new(None).unwrap();
    let rows = 64;
    let width = 1024;
    let zero = d.upload(&vec![0; rows * width * 2]).unwrap();
    let v: Vec<_> = (0..rows * width)
        .flat_map(|i| half::f16::from_f32(if i / width < 32 { 2. } else { 7. }).to_le_bytes())
        .collect();
    let v = d.upload(&v).unwrap();
    let out = d.alloc(rows * width * 2).unwrap();
    let plan: Vec<u32> = vec![0, 32, 0, 32, 32, 32, 32, 32];
    let plan = d
        .upload(
            &plan
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let c = d.begin().unwrap();
    c.dispatch(
        "sam3_attention",
        &[&zero, &zero, &v, &out, &plan],
        &[1],
        [16, 2, 1],
        64,
    );
    c.finish().unwrap();
    assert_eq!(bytes(&out), bytes(&v));
}

#[test]
fn sam3_attention_covers_real_window_and_global_extents() {
    let d = MetalDevice::new(None).unwrap();
    let zero = d.upload(&vec![0; 5184 * 1024 * 2]).unwrap();
    let v: Vec<_> = (0..5184 * 1024)
        .flat_map(|i| half::f16::from_f32((i % 1024 / 64 + 1) as f32).to_le_bytes())
        .collect();
    let v = d.upload(&v).unwrap();
    let out = d.alloc(v.len()).unwrap();
    for global in [false, true] {
        let tiles = super::load::tiles(&d, global).unwrap();
        let c = d.begin().unwrap();
        c.dispatch(
            "sam3_attention",
            &[&zero, &zero, &v, &out, &tiles],
            &[1],
            [16, 162, 1],
            64,
        );
        c.finish().unwrap();
        assert_eq!(
            bytes(&out),
            bytes(&v),
            "global={global}: real sequence/head geometry"
        );
    }
}
