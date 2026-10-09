//! Exact pixel/stem checks using the EXISTING CUDA qualification fixtures.
//! These do not require weights. Run explicitly with --ignored after copying
//! approved reference files; missing files must fail, never count as a pass.
use super::tests::bytes;
use super::*;
use paddock_models::safetensors::{SafetensorsFile, StDtype};

#[test]
#[ignore = "requires SAM3_GOLDENS and SAM3_ASSETS (official reference pictures)"]
fn picture_resize_matches_meta_bytes() {
    let gold = std::path::PathBuf::from(std::env::var_os("SAM3_GOLDENS").expect("SAM3_GOLDENS"));
    let assets = std::path::PathBuf::from(std::env::var_os("SAM3_ASSETS").expect("SAM3_ASSETS"));
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(gold.join("manifest.json")).unwrap()).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let dst = d.alloc(1008 * 1008 * 3).unwrap();
    let mut input = image::Input::default();
    for stem in ["groceries", "test_image", "truck"] {
        let name = manifest["images"][stem]["file"]
            .as_str()
            .expect("image filename");
        let rgb = paddock_jpeg::decode_rgb(&std::fs::read(assets.join(name)).unwrap(), 64_000_000)
            .unwrap();
        input
            .land(
                &d,
                &rgb.rgb,
                rgb.width,
                rgb.height,
                1008,
                Sam3InputKind::Picture,
                &dst,
            )
            .unwrap();
        let expected =
            super::golden_tests::picture(&gold.join("image").join(stem).join("input.safetensors"));
        let ours = bytes(&dst);
        let differing = ours.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{stem}: resized pixels differ from Meta; do not weaken to a tolerance"
        );
    }
}

#[test]
#[ignore = "requires SAM3_GOLDENS/video_frames/0001.safetensors (Meta CUDA video-loader fixtures)"]
fn video_resize_and_normalization_match_meta_bits() {
    let gold = std::path::PathBuf::from(std::env::var_os("SAM3_GOLDENS").expect("SAM3_GOLDENS"));
    let frames = SafetensorsFile::open(&gold.join("video_frames/0001.safetensors")).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let dst = d.alloc(1008 * 1008 * 3).unwrap();
    let patches = d.alloc(5184 * 592 * 2).unwrap();
    let mut input = image::Input::default();
    // Require the full 24-frame contract; no truncation at the first missing
    // tensor, which could otherwise turn a partial fixture into a green test.
    for frame in 0..24 {
        let (meta, rgb) = frames
            .bytes(&format!("f{frame:04}"))
            .expect("decoded RGB frame");
        assert_eq!(meta.dtype, StDtype::U8);
        assert_eq!(meta.shape.len(), 3);
        assert_eq!(meta.shape[2], 3);
        let h = meta.shape[0];
        let w = meta.shape[1];
        input
            .land(&d, rgb, w, h, 1008, Sam3InputKind::VideoFrame, &dst)
            .unwrap();
        let (shape, expected) = frames
            .bytes(&format!("r{frame:04}"))
            .expect("resized frame");
        assert_eq!(shape.dtype, StDtype::U8);
        assert_eq!(shape.shape, [1008, 1008, 3]);
        let ours = bytes(&dst);
        assert_eq!(ours.len(), expected.len());
        assert_eq!(
            ours.iter().zip(expected).filter(|(a, b)| a != b).count(),
            0,
            "frame {frame}: resize bytes"
        );
        let c = d.begin().unwrap();
        point(
            &c,
            "sam3_patch",
            &[&dst, &patches],
            &[1008, 14, 24, 592, 1],
            5184 * 592,
        );
        c.finish().unwrap();
        let raw = bytes(&patches);
        let (shape, expected) = frames
            .bytes(&format!("n{frame:04}"))
            .expect("normalized F16 frame");
        assert_eq!(shape.dtype, StDtype::F16);
        assert_eq!(shape.shape, [3, 1008, 1008]);
        let mut differing = 0;
        for row in 0..5184 {
            let win = row / 576;
            let cell = row % 576;
            let gy = win / 3 * 24 + cell / 24;
            let gx = win % 3 * 24 + cell % 24;
            for col in 0..592 {
                let ours = &raw[(row * 592 + col) * 2..(row * 592 + col) * 2 + 2];
                if col >= 588 {
                    differing += usize::from(ours != [0, 0]);
                    continue;
                }
                let channel = col / 196;
                let y = gy * 14 + col % 196 / 14;
                let x = gx * 14 + col % 14;
                let index = ((channel * 1008 + y) * 1008 + x) * 2;
                differing += usize::from(ours != &expected[index..index + 2]);
            }
        }
        assert_eq!(differing, 0, "frame {frame}: normalized patch bits");
    }
}
