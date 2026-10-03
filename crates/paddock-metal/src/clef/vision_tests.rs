use super::*;
use paddock_engine::clef_decision::{ClefImage, ClefQuestion};
#[test]
fn image_plan_is_bounded_and_rotary_positions_are_request_local() {
    let ids = vec![1; 20];
    let question = ClefQuestion {
        qtype: 1,
        span: (12, 14),
        options: vec![(14, 16), (17, 19)],
    };
    let mut image = ClefImage {
        rgb: vec![0; 64 * 96 * 3],
        width: 96,
        height: 64,
        resized: (64, 96),
        row: 2,
    };
    let plan = |im: &ClefImage| {
        plan::Plan::new(
            &[
                ClefRequest {
                    ids: &ids,
                    questions: std::slice::from_ref(&question),
                    images: std::slice::from_ref(im),
                },
                ClefRequest {
                    ids: &ids,
                    questions: std::slice::from_ref(&question),
                    images: &[],
                },
            ],
            100,
        )
    };
    let p = plan(&image).unwrap();
    assert_eq!(
        &p.positions[6..24],
        &[2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 3, 3, 2, 3, 4]
    );
    assert_eq!(&p.positions[24..27], &[5, 5, 5]);
    assert_eq!(&p.positions[60..63], &[0, 0, 0]);
    image.row = 18;
    assert!(plan(&image).is_err());
    image.row = 2;
    image.resized = (usize::MAX, 32);
    assert!(plan(&image).is_err());
    image.resized = (64, 96);
    image.rgb.pop();
    assert!(plan(&image).is_err());
}
#[test]
#[ignore = "requires CLEF_DIR and CLEF_VISION_GOLD from independent GPU image oracle"]
fn same_weight_image_decisions_and_packed_isolation() {
    let root = std::path::PathBuf::from(std::env::var("CLEF_DIR").unwrap());
    let gold = std::path::PathBuf::from(std::env::var("CLEF_VISION_GOLD").unwrap());
    let companion = std::env::var("CLEF_COMPANION").ok();
    let only = std::env::var("CLEF_VISION_CASE").ok();
    let mut model =
        Clef::load_with_companion(&root, companion.as_deref().map(Path::new), None).unwrap();
    eprintln!(
        "loaded: weights={} workspace={}",
        model.weight_bytes, model.workspace_bytes
    );
    let pad = model.vision_config().expect("tower loaded").0.image_token;
    let mut files = std::fs::read_dir(&gold)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .filter(|p| {
            only.as_ref()
                .is_none_or(|name| p.file_stem().and_then(|s| s.to_str()) == Some(name.as_str()))
        })
        .collect::<Vec<_>>();
    files.sort();
    assert!(!files.is_empty());
    for path in files {
        let g: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let id = g["id"].as_str().unwrap();
        let ids = g["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let questions = g["questions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let span = |v: &serde_json::Value| {
                    (
                        v[0].as_u64().unwrap() as usize,
                        v[1].as_u64().unwrap() as usize,
                    )
                };
                ClefQuestion {
                    qtype: v["type"].as_u64().unwrap() as u32,
                    span: span(&v["span"]),
                    options: v["option_spans"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(span)
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let mut start = 0;
        let images = g["media"]["images"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let grid = &g["media"]["image_grid_thw"][i];
                let row = start + ids[start..].iter().position(|&t| t == pad).unwrap();
                let image = ClefImage {
                    rgb: std::fs::read(gold.join(format!("{id}.rgb{i}.u8"))).unwrap(),
                    width: v["width"].as_u64().unwrap() as usize,
                    height: v["height"].as_u64().unwrap() as usize,
                    resized: (
                        grid[1].as_u64().unwrap() as usize * 16,
                        grid[2].as_u64().unwrap() as usize * 16,
                    ),
                    row,
                };
                start = row + image.tokens();
                image
            })
            .collect::<Vec<_>>();
        let request = || ClefRequest {
            ids: &ids,
            questions: &questions,
            images: &images,
        };
        let read = |name: &str| {
            std::fs::read(gold.join(name))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let expected = read(&format!("{id}.pixels.f32"));
        let mut patch_offset = 0;
        for im in &images {
            let rgb = vision::resize::pixels(&model.device, im, model.vision.as_ref().unwrap().mlx)
                .unwrap();
            let rows = im.tokens() * 4;
            let patches = model.device.alloc(rows * 1536 * 4).unwrap();
            let cmd = model.device.begin().unwrap();
            point(
                &cmd,
                "clef_vis_patches",
                &[&rgb, &patches],
                &[im.resized.1 as u32, im.resized.0 as u32],
                rows * 1536,
            );
            cmd.finish().unwrap();
            let actual = unsafe { patches.read_f32(0, rows * 1536) };
            let expected = &expected[patch_offset..patch_offset + actual.len()];
            let error = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                actual.iter().all(|v| v.is_finite()) && error < 3e-7,
                "{id}: processor mismatch {error}"
            );
            patch_offset += actual.len();
        }
        let start = std::time::Instant::now();
        let result = model.forward(&[request()]).unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.;
        let expected = g["questions"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|q| {
                q["logits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
            })
            .collect::<Vec<_>>();
        let actual = result[0].iter().flatten().copied().collect::<Vec<_>>();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "{id}: tokens={} ms={ms:.1} max_logit_error={error:.8} actual={actual:?} expected={expected:?}",
            ids.len()
        );
        assert!(
            error < 2e-4,
            "{id}: strict image reference mismatch {error}"
        );
        for (actual, question) in result[0].iter().zip(g["questions"].as_array().unwrap()) {
            let expected = question["probabilities"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>();
            let maximum = actual.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = actual
                .iter()
                .map(|&v| f64::from(v - maximum).exp())
                .collect::<Vec<_>>();
            let sum: f64 = weights.iter().sum();
            let probs = weights.iter().map(|v| v / sum).collect::<Vec<_>>();
            let winner = |values: &[f64]| {
                values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            assert_eq!(winner(&probs), winner(&expected), "{id}: decision flip");
            assert!(
                probs
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| (a - b).abs() < 5e-5),
                "{id}: probability mismatch"
            );
        }
        if ids.len() * 2 <= MAX_ROWS {
            let packed = model.forward(&[request(), request()]).unwrap();
            assert_eq!(result[0], packed[0], "{id}: packing changed request 0");
            assert_eq!(result[0], packed[1], "{id}: packing changed request 1");
        }
        assert!(
            model.device.allocated_bytes()
                <= model.weight_bytes + workspace::Workspace::bytes(&model.config) + (16 << 20),
            "{id}: vision scratch retained after the pass"
        );
    }
}
