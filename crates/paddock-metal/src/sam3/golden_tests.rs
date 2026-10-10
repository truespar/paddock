//! The same independent gate as engine/tests/gpu_sam3_golden.rs. Fixtures
//! come from Meta's own CUDA implementation, NOT a CPU reference or a new
//! oracle implemented here. Explicitly ignored, never silently "passed"
//! when checkpoint/reference data is absent. No SAM Materials in the repo.
use super::*;
use paddock_models::safetensors::{SafetensorsFile, StDtype};

pub(super) fn tensor(path: &Path, name: &str, shape: &[usize]) -> Vec<f32> {
    let file = SafetensorsFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (t, bytes) = file
        .bytes(name)
        .unwrap_or_else(|| panic!("{} lacks {name}", path.display()));
    assert_eq!(t.shape, shape, "{name}");
    let values: Vec<_> = match t.dtype {
        StDtype::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect(),
        StDtype::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| f32::from_bits(u32::from(u16::from_le_bytes(*x)) << 16))
            .collect(),
        other => panic!("unsupported golden dtype {other:?}"),
    };
    assert_eq!(values.len(), shape.iter().product::<usize>());
    assert!(values.iter().all(|v| v.is_finite()), "nonfinite {name}");
    values
}

// Reconstruct the EXACT uint8 input from Meta's saved normalized pixel
// tensor, checking every value. This is fixture decoding, not model math.
pub(super) fn picture(path: &Path) -> Vec<u8> {
    let values = tensor(path, "pixel", &[1, 3, 1008, 1008]);
    let mut rgb = vec![0; values.len()];
    for ch in 0..3 {
        for i in 0..1008 * 1008 {
            let value = values[ch * 1008 * 1008 + i];
            let byte = ((value * 0.5 + 0.5) * 255.).round().clamp(0., 255.) as u8;
            assert_eq!(
                (f32::from(byte) * (1. / 255.) - 0.5) * 2.,
                value,
                "non-invertible golden pixel"
            );
            rgb[i * 3 + ch] = byte;
        }
    }
    rgb
}

fn cell(r: usize, side: usize, window_major: bool) -> (usize, usize) {
    if window_major {
        let wi = r / 576;
        (
            wi / (side / 24) * 24 + r % 576 / 24,
            wi % (side / 24) * 24 + r % 24,
        )
    } else {
        (r / side, r % side)
    }
}

fn check(
    ours: &[f32],
    fp32: &[f32],
    bf16: &[f32],
    side: usize,
    channels: usize,
    window: bool,
    label: &str,
) {
    assert_eq!(ours.len(), side * side * channels);
    assert_eq!(fp32.len(), ours.len());
    assert_eq!(bf16.len(), ours.len());
    assert!(
        ours.iter().all(|x| x.is_finite()),
        "{label}: nonfinite output"
    );
    let (mut metal, mut meta, mut denominator) = (0f64, 0f64, 0f64);
    for r in 0..side * side {
        let (y, x) = cell(r, side, window);
        for ch in 0..channels {
            let index = (ch * side + y) * side + x;
            let truth = f64::from(fp32[index]);
            metal += (f64::from(ours[r * channels + ch]) - truth).powi(2);
            meta += (f64::from(bf16[index]) - truth).powi(2);
            denominator += truth * truth;
        }
    }
    let metal = (metal / denominator.max(1e-300)).sqrt();
    let meta = (meta / denominator.max(1e-300)).sqrt();
    eprintln!("{label}: Metal rel-RMS {metal:.6e}; Meta BF16 rel-RMS {meta:.6e}");
    assert!(
        metal <= meta,
        "{label}: exceeds Meta's own BF16 distance from FP32"
    );
}

fn hash_plane(model: &Sam3Vision, plane: Sam3VisionPlane) -> blake3::Hash {
    let values = model.read_plane(plane).unwrap();
    let mut hash = blake3::Hasher::new();
    for value in values {
        hash.update(&value.to_bits().to_le_bytes());
    }
    hash.finalize()
}

#[test]
#[ignore = "requires approved SAM3_DIR checkpoint and SAM3_GOLDENS Meta CUDA reference tensors"]
fn image_encoder_meets_cuda_reference_gate_and_replays_exactly() {
    let dir = std::env::var_os("SAM3_DIR")
        .expect("set SAM3_DIR to the approved facebook/sam3 checkpoint");
    let gold = std::env::var_os("SAM3_GOLDENS")
        .expect("set SAM3_GOLDENS to Meta CUDA fixtures, not Paddock-generated outputs");
    let gold = std::path::PathBuf::from(gold);
    let _: serde_json::Value = serde_json::from_slice(
        &std::fs::read(gold.join("manifest.json")).expect("reference manifest"),
    )
    .expect("parse reference manifest");
    let mut model = Sam3Vision::load(Path::new(&dir), None).expect("load SAM 3 vision");
    assert!(model.read_plane(Sam3VisionPlane::Trunk).is_err());
    assert!(model.ensure_tracker().is_err());
    eprintln!(
        "SAM 3 Metal weights {} MiB; workspace {} MiB",
        model.weight_bytes() >> 20,
        model.workspace_bytes() >> 20
    );
    for stem in ["groceries", "test_image", "truck"] {
        let image = gold.join("image").join(stem);
        let rgb = picture(&image.join("input.safetensors"));
        let start = std::time::Instant::now();
        model.encode(&rgb, false, false).unwrap();
        eprintln!(
            "{stem}: encoder + detector neck {:.3} seconds wall",
            start.elapsed().as_secs_f64()
        );
        let fp32 = image.join("backbone.fp32.safetensors");
        let bf16 = image.join("backbone.bf16.safetensors");
        for (plane, name, side, channels, window) in [
            (Sam3VisionPlane::Trunk, "trunk", 72, 1024, true),
            (Sam3VisionPlane::Detector(0), "fpn0", 288, 256, false),
            (Sam3VisionPlane::Detector(1), "fpn1", 144, 256, false),
            (Sam3VisionPlane::Detector(2), "fpn2", 72, 256, false),
        ] {
            check(
                &model.read_plane(plane).unwrap(),
                &tensor(&fp32, name, &[1, channels, side, side]),
                &tensor(&bf16, name, &[1, channels, side, side]),
                side,
                channels,
                window,
                &format!("{stem}/{name}"),
            );
        }
        if stem == "truck" {
            let before: Vec<_> = (0..3)
                .map(|i| hash_plane(&model, Sam3VisionPlane::Detector(i)))
                .collect();
            assert!(model.read_plane(Sam3VisionPlane::TrackerSkip0).is_err());
            model.ensure_tracker().unwrap();
            model.ensure_tracker().unwrap(); // idempotent, no encoder replay
            for (i, h) in before.iter().enumerate() {
                assert_eq!(*h, hash_plane(&model, Sam3VisionPlane::Detector(i)));
            }
            let tracker = gold.join("pvs").join(stem);
            for (plane, name, side, channels) in [
                (Sam3VisionPlane::TrackerSkip0, "fpn0", 288, 32),
                (Sam3VisionPlane::TrackerSkip1, "fpn1", 144, 64),
                (Sam3VisionPlane::Tracker(2), "fpn2", 72, 256),
            ] {
                check(
                    &model.read_plane(plane).unwrap(),
                    &tensor(
                        &tracker.join("sam2_neck.fp32.safetensors"),
                        name,
                        &[1, channels, side, side],
                    ),
                    &tensor(
                        &tracker.join("sam2_neck.bf16.safetensors"),
                        name,
                        &[1, channels, side, side],
                    ),
                    side,
                    channels,
                    false,
                    &format!("{stem}/tracker/{name}"),
                );
            }
            let planes = [
                Sam3VisionPlane::Trunk,
                Sam3VisionPlane::Detector(0),
                Sam3VisionPlane::Detector(1),
                Sam3VisionPlane::Detector(2),
                Sam3VisionPlane::TrackerSkip0,
                Sam3VisionPlane::TrackerSkip1,
                Sam3VisionPlane::Tracker(2),
            ];
            let expected: Vec<_> = planes.iter().map(|p| hash_plane(&model, *p)).collect();
            let resident = model.device.allocated_bytes();
            for _ in 0..3 {
                model.encode(&rgb, false, true).unwrap();
                for (p, h) in planes.iter().zip(&expected) {
                    assert_eq!(*h, hash_plane(&model, *p), "replay {p:?}");
                }
                assert_eq!(
                    model.device.allocated_bytes(),
                    resident,
                    "encoder grew its workspace"
                );
            }
        }
        assert!(model.encode(&rgb[..1], false, false).is_err());
        assert!(
            model.read_plane(Sam3VisionPlane::Trunk).is_err(),
            "failed encode exposed stale features"
        );
        assert!(model.ensure_tracker().is_err());
    }
}
