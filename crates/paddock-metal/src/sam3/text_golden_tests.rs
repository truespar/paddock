//! Independent CUDA/Meta fixtures, using the same contract as the existing
//! CUDA end-to-end gate. No relaxed tolerance or self-generated oracle.
use super::*;
use paddock_tokenizer::sam3::{Sam3TokenizeError, Sam3Tokenizer};
use std::path::PathBuf;

fn inputs() -> (PathBuf, PathBuf) {
    (
        std::env::var_os("SAM3_DIR")
            .expect("set SAM3_DIR to approved SAM 3 weights")
            .into(),
        std::env::var_os("SAM3_GOLDENS")
            .expect("set SAM3_GOLDENS to independent Meta CUDA fixtures")
            .into(),
    )
}
fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore = "requires approved SAM3_DIR tokenizer and all 46 SAM3_GOLDENS/tokens.json reference cases"]
fn text_tokenizer_matches_all_meta_fixtures_without_silent_truncation() {
    let (dir, gold) = inputs();
    let tokenizer = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let fixture = json(&gold.join("tokens.json"));
    let prompts = fixture["prompts"].as_array().expect("reference prompts");
    assert_eq!(prompts.len(), 46, "reference fixture is incomplete");
    let ids = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
            .collect::<Vec<_>>()
    };
    for p in prompts {
        let text = p["prompt"].as_str().unwrap();
        assert_eq!(tokenizer.bpe(text).unwrap(), ids(&p["bpe"]), "BPE {text:?}");
        match tokenizer.encode(text) {
            Ok(t) => {
                assert!(!p["too_long"].as_bool().unwrap());
                assert_eq!(t.ids.as_slice(), ids(&p["ids"]), "layout {text:?}");
                assert_eq!(t.valid, tokenizer.bpe(text).unwrap().len() + 2);
            }
            Err(Sam3TokenizeError::TooLong { .. }) => assert!(p["too_long"].as_bool().unwrap()),
            Err(e) => panic!("{text:?}: {e}"),
        }
    }
}

#[test]
#[ignore = "requires approved SAM3_DIR and all nine Meta image/detector FP32+BF16 fixtures"]
fn text_tower_meets_meta_reference_bar_and_is_batch_bit_stable() {
    let (dir, gold) = inputs();
    let manifest = json(&gold.join("manifest.json"));
    let cases: Vec<_> = manifest["cases"]
        .as_array()
        .expect("reference cases")
        .iter()
        .filter(|c| c["mode"].as_str() == Some("fp32"))
        .collect();
    assert_eq!(cases.len(), 9, "reference image prompt suite is incomplete");
    let tokenizer = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    let required = Sam3Text::resident_bytes_required(4).unwrap();
    let mut model =
        Sam3Text::load(&dir, 4, Some(required)).expect("load at the declared reservation");
    assert_eq!(model.device.allocated_bytes(), required);
    let mut prompts = Vec::new();
    let mut singles = Vec::new();
    for case in &cases {
        let file = case["file"].as_str().expect("fixture filename");
        let stem = file
            .strip_suffix(".fp32.safetensors")
            .expect("fp32 fixture suffix");
        let words = case["encoded_text"]
            .as_str()
            .expect("exact reference-encoded text");
        let tokens = tokenizer.encode(words).unwrap();
        let start = std::time::Instant::now();
        model.encode(&tokens.ids, 1).unwrap();
        let elapsed = start.elapsed();
        let ours = model.read_features().unwrap();
        // Meta returns sequence-first tokens. With batch=1 this flattens to
        // the same row-major layout, but assert the shape instead of guessing.
        let fp32 = golden_tests::tensor(&gold.join(file), "language_features", &[32, 1, 256]);
        let bf16 = golden_tests::tensor(
            &gold.join(format!("{stem}.bf16.safetensors")),
            "language_features",
            &[32, 1, 256],
        );
        let count = tokens.valid * 256;
        let dist = |a: &[f32]| {
            a[..count]
                .iter()
                .zip(&fp32[..count])
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>()
        };
        let energy: f64 = fp32[..count].iter().map(|x| f64::from(*x).powi(2)).sum();
        assert!(energy > 0. && ours.iter().all(|x| x.is_finite()));
        let metal = (dist(&ours) / energy).sqrt();
        let meta = (dist(&bf16) / energy).sqrt();
        eprintln!(
            "{stem} {words:?}: Metal rel-RMS {metal:.6e}, Meta BF16 {meta:.6e}; {:.3} ms",
            elapsed.as_secs_f64() * 1000.
        );
        assert!(
            metal <= meta,
            "{stem}: text exceeds Meta BF16's distance from FP32"
        );
        prompts.push(tokens.ids);
        singles.push(bits(&ours));
    }
    // All prompts individually, four-way, duplicated, reordered, and after
    // shrinking/re-expanding a batch. Every row including padding is checked.
    for round in 0..3 {
        for cap in [4, 1, 3, 2, 4] {
            for start in 0..prompts.len() {
                let indices: Vec<_> = (0..cap)
                    .map(|i| (start + i * (round + 1)) % prompts.len())
                    .collect();
                let ids: Vec<_> = indices.iter().flat_map(|&i| prompts[i]).collect();
                model.encode(&ids, cap).unwrap();
                let got = bits(&model.read_features().unwrap());
                for (row, &index) in indices.iter().enumerate() {
                    assert_eq!(
                        &got[row * 32 * 256..(row + 1) * 32 * 256],
                        singles[index],
                        "round {round}, batch {cap}, prompt {index}"
                    );
                }
                assert_eq!(model.device.allocated_bytes(), required);
            }
        }
    }
    model.encode(&prompts[0].repeat(4), 4).unwrap();
    assert!(model.encode(&[u32::MAX; 32], 1).is_err());
    assert!(model.read_features().is_err());
    model.encode(&prompts[0], 1).unwrap();
    assert_eq!(bits(&model.read_features().unwrap()), singles[0]);
}
