use super::tests::bytes;
use super::*;

#[test]
fn resize_keeps_picture_and_video_rounding_distinct() {
    let d = MetalDevice::new(None).unwrap();
    let mut input = image::Input::default();
    let out = d.alloc(3).unwrap();
    // A 2x2 grayscale picture [0,1;0,1] averages to 0.5. Picture rounds
    // ties to even at the END (0); Pillow rounds each axis half-up (1).
    let rgb = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1];
    for (kind, expected) in [(Sam3InputKind::Picture, 0), (Sam3InputKind::VideoFrame, 1)] {
        input.land(&d, &rgb, 2, 2, 1, kind, &out).unwrap();
        assert_eq!(bytes(&out), [expected; 3]);
    }
}

#[test]
fn resize_reuses_staging_handles_identity_axes_and_rejects_bad_geometry() {
    let d = MetalDevice::new(None).unwrap();
    let base = d.allocated_bytes();
    {
        let mut input = image::Input::default();
        let side = 17;
        let out = d.alloc(side * side * 3).unwrap();
        for kind in [Sam3InputKind::Picture, Sam3InputKind::VideoFrame] {
            for (w, h) in [(17, 17), (17, 23), (31, 17), (31, 23), (2, 3)] {
                let rgb: Vec<_> = (0..w * h * 3).map(|i| [0, 128, 255][i % 3]).collect();
                input.land(&d, &rgb, w, h, side, kind, &out).unwrap();
                let expected: Vec<_> = (0..side * side * 3).map(|i| [0, 128, 255][i % 3]).collect();
                assert_eq!(bytes(&out), expected, "{kind:?} {w}x{h}");
                let retained = d.allocated_bytes();
                for _ in 0..3 {
                    input.land(&d, &rgb, w, h, side, kind, &out).unwrap();
                    assert_eq!(d.allocated_bytes(), retained, "resize staging grew");
                    assert_eq!(bytes(&out), expected);
                }
                assert_eq!(retained - base - out.len() as u64, input.bytes());
            }
        }
        let before = d.allocated_bytes();
        for (rgb, w, h, side) in [
            (&[][..], 0, 0, 17),
            (&[0][..], 1, 1, 17),
            (&[][..], usize::MAX, 2, 17),
            (&[][..], 1, 1, usize::MAX),
        ] {
            assert!(
                input
                    .land(&d, rgb, w, h, side, Sam3InputKind::Picture, &out)
                    .is_err()
            );
        }
        assert_eq!(
            d.allocated_bytes(),
            before,
            "invalid input allocated memory"
        );
    }
    assert_eq!(d.allocated_bytes(), base, "input staging leaked");
}

#[test]
fn resize_identity_is_byte_exact_for_all_uint8_levels() {
    let d = MetalDevice::new(None).unwrap();
    let rgb: Vec<_> = (0..16 * 16 * 3).map(|i| i as u8).collect();
    let out = d.alloc(rgb.len()).unwrap();
    let mut input = image::Input::default();
    for kind in [Sam3InputKind::Picture, Sam3InputKind::VideoFrame] {
        input.land(&d, &rgb, 16, 16, 16, kind, &out).unwrap();
        assert_eq!(bytes(&out), rgb);
    }
}
