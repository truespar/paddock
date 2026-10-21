//! Gemma 4 12B (unified) against an F32 reference on the SAME weights.
//!
//! The reference is transformers' `Gemma4UnifiedForConditionalGeneration`
//! built from the registry files themselves - the Q8_0 text GGUF dequantized,
//! the mmproj's tensors with the converter's CHW permutation and pos-table
//! transpose undone - recorded by the 12B oracle (`gemma4_12b_oracle.py`)
//! into `target/gemma4u-reference` (or `GEMMA4U_REF`):
//!
//! - `embed/<medium>.f64`: the picture / audio embedder rows alone, in F64
//!   from the processor's own pixel values and frames. The engine's rows are
//!   F32 class, so they must agree to ~1e-5 of the row scale - this is where
//!   a wrong patch order, norm order or position axis would show.
//! - `fixture.json` + `<case>.logits0.f32`: per prompt (text, pictures,
//!   clips), the last prompt row's logits and the greedy continuation. The
//!   engine runs Q8_0 as W8A8 with an fp8 KV cache, so logits agree by
//!   cosine and top-1, and the reference's greedy stream is teacher-forced
//!   through the engine step by step up to its first stop token (past it
//!   the reference only rambles on out of distribution): every step the
//!   reference decided with a clear margin must pick the same token.
//!
//! Heavy (12B Q8_0 + mmproj): PADDOCK_HEAVY_TESTS=1, --release.
//! GEMMA4U_DIR names the model folder.

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_engine::gpu_model::gemma4::unified::UnifiedEmbedder;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;

fn ref_dir() -> PathBuf {
    std::env::var_os("GEMMA4U_REF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gemma4u-reference")
        })
}

fn files() -> Option<(PathBuf, PathBuf)> {
    if !common::heavy() {
        return None;
    }
    let dir = common::model_dir("GEMMA4U_DIR", &["gemma-4-12B-it-GGUF"])?;
    let model = dir.join("gemma-4-12b-it-Q8_0.gguf");
    let mmproj = dir.join("mmproj-BF16.gguf");
    if !model.exists() || !mmproj.exists() || !ref_dir().join("fixture.json").exists() {
        common::missing("the 12B Q8_0 + mmproj, or the recorded reference");
        return None;
    }
    Some((model, mmproj))
}

/// A picture as the processor's PIL decode gives it (`Image.open(...)
/// .convert("RGB")`, no EXIF transpose): JPEG through libjpeg-turbo's
/// arithmetic, everything else through the image codecs, alpha dropped.
fn picture(path: &Path) -> (Vec<u8>, usize, usize) {
    let bytes = std::fs::read(path).expect("test picture");
    let rgb = if paddock_jpeg::sniff(&bytes) {
        let im = paddock_jpeg::decode_rgb(&bytes, 64_000_000).expect("jpeg");
        image::RgbImage::from_raw(im.width as u32, im.height as u32, im.rgb).expect("plane")
    } else {
        image::load_from_memory(&bytes).expect("decode").into_rgb8()
    };
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    (rgb.into_raw(), w, h)
}

fn clip(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("test clip");
    let wav = paddock_engine::audio::decode::decode_audio(&bytes).expect("decode");
    paddock_engine::audio::resample::resample(&wav.samples, wav.sample_rate, 16000)
        .expect("resample")
}

fn read_f64(path: &Path) -> Vec<f64> {
    std::fs::read(path)
        .expect("reference rows")
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .expect("reference logits")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn amax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

#[test]
fn embedder_rows_match_the_f64_reference() {
    let Some((_, mmproj)) = files() else {
        return;
    };
    let exec = common::gpu_arc().expect("gpu");
    let map = MappedGguf::open(&mmproj).expect("open mmproj");
    let e = UnifiedEmbedder::load(exec.clone(), &map, None, true).expect("load embedders");
    let media = ref_dir().join("media");
    for name in [
        "house.jpg",
        "receipt.png",
        "chart_rgba.png",
        "fleurs1.wav",
        "speech6.wav",
    ] {
        let path = media.join(name);
        let (rows, n) = if name.ends_with(".wav") {
            let s = clip(&path);
            let n = UnifiedEmbedder::audio_rows(s.len());
            (e.encode_audio(&s, 0, n).expect("encode clip"), n)
        } else {
            let (rgb, w, h) = picture(&path);
            e.encode_picture(&rgb, w, h).expect("encode picture")
        };
        let got = exec.to_host_len(&rows, n * 3840).expect("download");
        let want = read_f64(&ref_dir().join("embed").join(format!("{name}.f64")));
        assert_eq!(want.len(), got.len(), "{name}: row count");
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f64, |m, (&g, &w)| m.max((g as f64 - w).abs()));
        eprintln!("{name}: {n} rows, max |d| {worst:.3e} of scale {scale:.3}");
        assert!(
            worst <= 1e-4 * scale,
            "{name}: max |d| {worst:.3e} past 1e-4 of the row scale {scale:.3}"
        );
    }
}

#[test]
fn prompts_follow_the_f32_reference() {
    let Some((model, mmproj)) = files() else {
        return;
    };
    let exec = common::gpu_arc().expect("gpu");
    let map = MappedGguf::open(&model).expect("open gguf");
    let mut m = GpuGemma4::load(exec, &map, 8192).expect("load");
    let mm = MappedGguf::open(&mmproj).expect("open mmproj");
    m.attach_media(&mm, None, true).expect("attach media");
    Generator::enable_batch(&mut m, 2).expect("enable_batch");
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_dir().join("fixture.json")).expect("fixture"))
            .expect("json");
    let media = ref_dir().join("media");
    let mut failures = Vec::new();
    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let mut chunks = Vec::new();
        for part in case["parts"].as_array().expect("parts") {
            let file =
                |k: &str| media.join(Path::new(part[k].as_str().unwrap()).file_name().unwrap());
            if let Some(ids) = part.get("ids") {
                let ids: Vec<u32> = ids
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                chunks.push(MmChunk::Text(ids));
            } else if part.get("image").is_some() {
                let (rgb, w, h) = picture(&file("image"));
                chunks.push(MmChunk::Image { rgb, w, h });
            } else {
                chunks.push(MmChunk::Audio {
                    samples: clip(&file("audio")),
                    mel: None,
                });
            }
        }
        let (logits, rows) =
            Generator::forward_prefill_multimodal(&mut m, 0, &chunks).expect("prefill");
        assert_eq!(
            rows,
            case["rows"].as_u64().unwrap() as usize,
            "{name}: prompt rows"
        );
        let want = read_f32(&ref_dir().join(format!("{name}.logits0.f32")));
        let cos = cosine(&logits, &want);
        // the generation_config's eos set: <eos>, <turn|>, <|tool_response>
        let mut greedy: Vec<u32> = case["greedy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        if let Some(stop) = greedy.iter().position(|t| [1, 106, 50].contains(t)) {
            greedy.truncate(stop + 1);
        }
        let margins: Vec<f64> = case["margins"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        // teacher-forced: the engine's pick at every step of the reference's
        // own stream; a step decided by > 1.0 logit is not a near-tie
        let mut pick = amax(&logits) as u32;
        let (mut agree, mut clear_miss) = (0usize, Vec::new());
        for (i, &t) in greedy.iter().enumerate() {
            if pick == t {
                agree += 1;
            } else if margins[i] > 1.0 {
                clear_miss.push(i);
            }
            if i + 1 == greedy.len() {
                break;
            }
            let next = Generator::forward_batch(&mut m, &[t], &[(rows + i) as u32]).expect("step");
            pick = amax(&next[..want.len()]) as u32;
        }
        eprintln!(
            "{name}: {rows} rows, logits cosine {cos:.5}, first {} vs {}, teacher-forced \
             {agree}/{} (clear misses {clear_miss:?})",
            amax(&logits),
            amax(&want),
            greedy.len()
        );
        // the cosine is over all 262k logits, most of them a W8A8 + fp8-KV
        // tail no sampler reaches - a sanity floor, the picks are the gate
        if cos < 0.98 || amax(&logits) != amax(&want) || !clear_miss.is_empty() {
            failures.push(name.to_owned());
        }
    }
    assert!(failures.is_empty(), "off the reference: {failures:?}");
}
