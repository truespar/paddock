//! EmbeddingGemma 2's CUDA text encoder against llama.cpp on the identical
//! Q8_0 GGUF, plus the encoder contracts a reference cannot see.
//!
//! The reference: the newest prebuilt llama-server serving the same file
//! (`--embedding`, mean pooling from the GGUF, L2-normalized), recorded by the
//! GGUF oracle script into `target/embeddinggemma2-reference/gguf8-cuda.json`
//! (or `PADDOCK_EG2_FIXTURE`) after reproducing itself across two server
//! processes. Cases: one 18-token query, five ragged 15-18-token texts in one
//! request, a 982-token text crossing the sliding window and a 2,882-token
//! one.
//!
//! The gate is a tolerance set from the reference's own noise, not bit
//! equality. On CUDA both sides quantize activations to int8 (32-value
//! blocks) against the same Q8_0 weights, and this model amplifies a one-step
//! rounding difference anywhere upstream: llama.cpp's OWN CPU and CUDA
//! backends, same file, same tokens, disagree by up to 1.44e-4 in 1 - cosine
//! on these cases, and swapping only this lane's attention between f16 and
//! exact f32 moves its vectors by up to 1.8e-4 (GB10, 2026-10-07). This lane
//! sits at <= 2.6e-4 from llama.cpp's CUDA backend. So: cosine above 0.9995
//! and every component within 0.004. The Metal lane's 0.9999 holds there
//! because neither Metal side quantizes activations; it is below this class's
//! floor. A real defect - a wrong rope base, a mask on the wrong layers, a
//! swapped GLU - lands far outside either bar.
//!
//! What the reference cannot see, gated here:
//!   - batch invariance: a sequence's vector is bit-identical alone, packed
//!     with others and packed in reverse order - across the attention tiles'
//!     16/64-row edges, the sliding window's 512 and the mmq tile's 128;
//!   - Matryoshka: 128 / 256 / 512 are the 768 vector's prefix, re-normalized;
//!   - admission refuses media placeholders and overlong sequences;
//!   - a full 8,192-token sequence is finite and repeats exactly, before and
//!     after the idle scratch is dropped.

mod common;

use paddock_engine::gpu_model::embedding_gemma2::GpuEmbeddingGemma2;
use paddock_models::mapped::MappedGguf;

fn load(context: usize) -> Option<GpuEmbeddingGemma2> {
    let path = common::model("PADDOCK_EG2_MODEL", common::EMBEDDINGGEMMA2_Q8)?;
    let exec = common::gpu_arc()?;
    let map = MappedGguf::open(&path).expect("open gguf");
    Some(GpuEmbeddingGemma2::load(exec, &map, context).expect("load"))
}

struct Case {
    name: String,
    ids: Vec<Vec<u32>>,
    embeddings: Vec<Vec<f32>>,
}

fn fixture() -> Option<Vec<Case>> {
    let path = std::env::var_os("PADDOCK_EG2_FIXTURE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/embeddinggemma2-reference/gguf8-cuda.json")
        });
    let Ok(text) = std::fs::read_to_string(&path) else {
        common::missing(&format!(
            "EmbeddingGemma 2 reference fixture {} (record it with the GGUF oracle script)",
            path.display()
        ));
        return None;
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    let cases = v["cases"].as_array().expect("cases");
    Some(
        cases
            .iter()
            .map(|c| Case {
                name: c["name"].as_str().expect("name").into(),
                ids: serde_json::from_value(c["ids"].clone()).expect("ids"),
                embeddings: serde_json::from_value(c["embeddings"].clone()).expect("embeddings"),
            })
            .collect(),
    )
}

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|r| r.iter().map(|x| x.to_bits()).collect())
        .collect()
}

#[test]
fn embeddinggemma2_matches_llamacpp_on_the_same_gguf() {
    let Some(cases) = fixture() else { return };
    let Some(mut m) = load(8192) else { return };
    let (mut worst_cos, mut worst_err) = (1f64, 0f64);
    let mut dump = Vec::new();
    for case in &cases {
        let got = m.embed(&case.ids, None).expect("embed");
        dump.push(serde_json::json!({"name": case.name, "ids": case.ids, "embeddings": got}));
        assert_eq!(got.len(), case.embeddings.len(), "{}", case.name);
        for (i, (g, r)) in got.iter().zip(&case.embeddings).enumerate() {
            assert_eq!(g.len(), 768);
            let dot: f64 = g.iter().zip(r).map(|(a, b)| *a as f64 * *b as f64).sum();
            let ng: f64 = g.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nr: f64 = r.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (ng * nr);
            let err = g
                .iter()
                .zip(r)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0f64, f64::max);
            println!(
                "{} row {i}: cosine {cos:.9} max component error {err:.6}",
                case.name
            );
            assert!((ng - 1.0).abs() < 1e-4, "{} row {i}: norm {ng}", case.name);
            worst_cos = worst_cos.min(cos);
            worst_err = worst_err.max(err);
        }
    }
    // PADDOCK_EG2_DUMP: write these vectors in the oracle's fixture shape, for
    // cross-reference checks (e.g. against llama.cpp's CPU backend)
    if let Some(path) = std::env::var_os("PADDOCK_EG2_DUMP") {
        let doc = serde_json::json!({"endpoint": "paddock-cuda", "cases": dump});
        std::fs::write(path, doc.to_string()).expect("dump");
    }
    // every row is reported before the verdict, so a miss shows its pattern
    println!("worst cosine {worst_cos:.9}, worst component error {worst_err:.6}");
    assert!(worst_cos > 0.9995, "worst cosine {worst_cos}");
    assert!(worst_err < 0.004, "worst component error {worst_err}");
}

#[test]
fn embeddinggemma2_vectors_do_not_depend_on_the_pass() {
    let Some(cases) = fixture() else { return };
    let Some(mut m) = load(8192) else { return };
    let long = &cases
        .iter()
        .find(|c| c.name == "long")
        .expect("long case")
        .ids[0];
    // edges of the hd 512 tile (16), the hd 256 tile (64), the mmq tile
    // (128), the sliding window (512 / 1024), and the padded one-row pass
    let lens = [
        1usize, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513,
    ];
    let lens = lens.iter().chain(&[1023, 1024, 1025]);
    let seqs: Vec<Vec<u32>> = lens.map(|&n| long[..n].to_vec()).collect();
    let alone: Vec<Vec<f32>> = seqs
        .iter()
        .map(|s| {
            m.embed(std::slice::from_ref(s), None)
                .expect("alone")
                .remove(0)
        })
        .collect();
    let packed = m.embed(&seqs, None).expect("packed");
    let mut rev = seqs.clone();
    rev.reverse();
    let mut reversed = m.embed(&rev, None).expect("reversed");
    reversed.reverse();
    assert_eq!(bits(&alone), bits(&packed), "packed differs from alone");
    assert_eq!(bits(&alone), bits(&reversed), "reversed differs from alone");
    // and the fixture's own ragged request, against each text alone
    let ragged = &cases
        .iter()
        .find(|c| c.name == "ragged")
        .expect("ragged")
        .ids;
    let together = m.embed(ragged, None).expect("ragged");
    for (s, v) in ragged.iter().zip(&together) {
        let one = m.embed(std::slice::from_ref(s), None).expect("one");
        assert_eq!(bits(&one), bits(std::slice::from_ref(v)));
    }
}

#[test]
fn embeddinggemma2_matryoshka_is_the_renormalized_prefix() {
    let Some(cases) = fixture() else { return };
    let Some(mut m) = load(8192) else { return };
    let ids = &cases
        .iter()
        .find(|c| c.name == "ragged")
        .expect("ragged")
        .ids;
    let full = m.embed(ids, None).expect("768");
    for dims in [128usize, 256, 512, 768] {
        let short = m.embed(ids, Some(dims)).expect("prefix");
        for (s, f) in short.iter().zip(&full) {
            assert_eq!(s.len(), dims);
            let norm = f[..dims].iter().map(|v| v * v).sum::<f32>().sqrt();
            for (a, b) in s.iter().zip(&f[..dims]) {
                assert!(
                    (a - b / norm).abs() < 2e-6,
                    "dims {dims}: {a} vs {}",
                    b / norm
                );
            }
        }
    }
    assert!(GpuEmbeddingGemma2::validate_dimensions(Some(300)).is_err());
}

#[test]
fn embeddinggemma2_refuses_media_and_overlong_input() {
    let Some(m) = load(1024) else { return };
    assert!(
        m.validate(&[vec![2, 258880, 1]]).is_err(),
        "image placeholder"
    );
    assert!(
        m.validate(&[vec![2, 258884, 1]]).is_err(),
        "audio placeholder"
    );
    assert!(m.validate(&[vec![]]).is_err(), "empty sequence");
    assert!(m.validate(&[vec![7; 1025]]).is_err(), "past the context");
    assert!(m.validate(&[vec![262144]]).is_err(), "past the vocabulary");
    assert!(m.validate(&[vec![2, 100, 1]]).is_ok());
}

#[test]
fn embeddinggemma2_full_context_repeats_across_a_scratch_reclaim() {
    let Some(cases) = fixture() else { return };
    let Some(mut m) = load(8192) else { return };
    let long = &cases.iter().find(|c| c.name == "long").expect("long").ids[0];
    let ids: Vec<u32> = long.iter().cycle().take(8192).copied().collect();
    let a = m.embed(std::slice::from_ref(&ids), None).expect("8k");
    assert!(a[0].iter().all(|v| v.is_finite()));
    let b = m.embed(std::slice::from_ref(&ids), None).expect("8k again");
    assert_eq!(bits(&a), bits(&b));
    m.reclaim_idle();
    assert!(m.idle_reclaim_after().is_none());
    let c = m
        .embed(std::slice::from_ref(&ids), None)
        .expect("after reclaim");
    assert_eq!(bits(&a), bits(&c));
}

// ---- pictures ----------------------------------------------------------------
//
// The reference: the same llama-server with the mmproj (`clip.use_gelu = true`
// added - without it llama.cpp runs the tower's MLP on GELU_QUICK, not the
// checkpoint's tanh GELU, and lands up to 3.8e-4 away in 1 - cosine) given
// the Hugging Face processor's resize of each picture as a PNG, recorded by
// the media oracle script into `media280-llama.json` beside the processor's
// PNGs and the original files (`media280-prep/`, `originals/`). Here the
// ORIGINAL files go in, so the resize is under test too - byte for byte
// against the processor's output.

use paddock_engine::service::MmChunk;

fn reference_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/embeddinggemma2-reference")
}

fn load_with_pictures(context: usize) -> Option<GpuEmbeddingGemma2> {
    let mmproj = common::model("PADDOCK_EG2_MMPROJ", common::EMBEDDINGGEMMA2_MMPROJ)?;
    let mut m = load(context)?;
    let map = MappedGguf::open(&mmproj).expect("open mmproj");
    m.attach_mmproj(&map, 280, true)
        .expect("attach the picture tower");
    Some(m)
}

/// A test picture as the processor's Pillow decodes it: a JPEG through
/// libjpeg-turbo's arithmetic (`paddock_jpeg`), anything else through the
/// image codecs, upright by EXIF orientation, alpha dropped.
fn picture(name: &str) -> MmChunk {
    use image::metadata::Orientation;
    let path = reference_dir().join("media280-prep/originals").join(name);
    let bytes = std::fs::read(&path).expect("test picture");
    let img = if paddock_jpeg::sniff(&bytes) {
        let im = paddock_jpeg::decode_rgb(&bytes, 64_000_000).expect("jpeg");
        let rgb =
            image::RgbImage::from_raw(im.width as u32, im.height as u32, im.rgb).expect("plane");
        let mut img = image::DynamicImage::ImageRgb8(rgb);
        if let Some(o) = im.exif.as_deref().and_then(Orientation::from_exif_chunk) {
            img.apply_orientation(o);
        }
        img
    } else {
        use image::ImageDecoder as _;
        let mut d = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .expect("format")
            .into_decoder()
            .expect("decoder");
        let o = d.orientation().unwrap_or(Orientation::NoTransforms);
        let mut img = image::DynamicImage::from_decoder(d).expect("decode");
        img.apply_orientation(o);
        img
    };
    let rgb = img.into_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    MmChunk::Image {
        rgb: rgb.into_raw(),
        w,
        h,
    }
}

struct MediaCase {
    name: String,
    ids: Vec<Vec<u32>>,
    media: Vec<Vec<MmChunk>>,
    /// Per item: does it carry a synthetic clip (see [`SYNTHETIC`])?
    synthetic: Vec<bool>,
    embeddings: Vec<Vec<f32>>,
}

/// The ASR fixtures' generated signals - tones, a chirp, AM, silence -
/// against the LibriSpeech speech. Their spectra are mostly bins at the
/// rounding floor of the FFT, which every precision class rounds
/// differently (see the audio gates).
const SYNTHETIC: [&str; 6] = [
    "chirp",
    "am-12.5s",
    "tone-2s",
    "silence-1s",
    "tiny-0.3s",
    "exact-30s",
];

/// A test clip as written by the media oracle's `audio` step (16 kHz mono).
fn clip(name: &str) -> MmChunk {
    let path = reference_dir().join(format!("media280-prep/audio/{name}.wav"));
    let bytes = std::fs::read(&path).expect("test clip");
    let wav = paddock_engine::audio::decode::decode_audio(&bytes).expect("decode clip");
    assert_eq!(wav.sample_rate, 16000, "{name}");
    MmChunk::Audio {
        samples: wav.samples,
        mel: None,
    }
}

fn media_fixture() -> Option<(serde_json::Value, Vec<MediaCase>)> {
    media_fixture_at("PADDOCK_EG2_MEDIA_FIXTURE", "media280-llama.json")
}

fn media_fixture_at(env: &str, file: &str) -> Option<(serde_json::Value, Vec<MediaCase>)> {
    let path = std::env::var_os(env)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| reference_dir().join(file));
    let Ok(text) = std::fs::read_to_string(&path) else {
        common::missing(&format!(
            "EmbeddingGemma 2 media fixture {} (record it with the media oracle script)",
            path.display()
        ));
        return None;
    };
    let v: serde_json::Value = serde_json::from_str(&text).expect("fixture json");
    let cases = v["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| {
            // an item's media, in order: a string item has none
            let media = c["items"]
                .as_array()
                .expect("items")
                .iter()
                .map(|item| {
                    item.as_array().map_or_else(Vec::new, |parts| {
                        parts
                            .iter()
                            .filter_map(|p| match (p["image"].as_str(), p["audio"].as_str()) {
                                (Some(f), _) => Some(picture(f)),
                                (_, Some(a)) => Some(clip(a)),
                                _ => None,
                            })
                            .collect()
                    })
                })
                .collect();
            let synthetic = c["items"]
                .as_array()
                .expect("items")
                .iter()
                .map(|item| {
                    item.as_array().is_some_and(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| p["audio"].as_str())
                            .any(|a| SYNTHETIC.contains(&a))
                    })
                })
                .collect();
            MediaCase {
                name: c["name"].as_str().expect("name").into(),
                ids: serde_json::from_value(c["ids"].clone()).expect("ids"),
                media,
                synthetic,
                embeddings: serde_json::from_value(c["embeddings"].clone()).expect("embeddings"),
            }
        })
        .collect();
    Some((v, cases))
}

#[test]
fn embeddinggemma2_pictures_match_llamacpp_on_the_same_gguf() {
    let Some((fixture, cases)) = media_fixture() else {
        return;
    };
    let Some(mut m) = load_with_pictures(8192) else {
        return;
    };
    // the device resize is the processor's, byte for byte
    let prep = reference_dir().join("media280-prep");
    for (name, info) in fixture["images"].as_object().expect("images") {
        let MmChunk::Image { rgb, w, h } = picture(name) else {
            unreachable!()
        };
        let want = image::open(prep.join(info["png"].as_str().expect("png")))
            .expect("processor png")
            .into_rgb8();
        let (got, tw, th) = m
            .image_tower()
            .expect("tower")
            .resize(&rgb, w, h)
            .expect("resize");
        assert_eq!(
            (tw, th),
            (want.width() as usize, want.height() as usize),
            "{name}"
        );
        assert!(
            got == want.as_raw().as_slice(),
            "{name}: resized bytes differ"
        );
        assert_eq!(
            m.image_tokens(w, h).expect("tokens"),
            info["tokens"].as_u64().expect("tokens") as usize,
            "{name}"
        );
    }
    let (mut worst_cos, mut worst_err) = (1f64, 0f64);
    for case in &cases {
        let got = m.embed_media(&case.ids, &case.media, None).expect("embed");
        for (i, (g, r)) in got.iter().zip(&case.embeddings).enumerate() {
            let dot: f64 = g.iter().zip(r).map(|(a, b)| *a as f64 * *b as f64).sum();
            let ng: f64 = g.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let nr: f64 = r.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (ng * nr);
            let err = g
                .iter()
                .zip(r)
                .map(|(a, b)| (*a as f64 - *b as f64).abs())
                .fold(0f64, f64::max);
            println!(
                "{} row {i}: cosine {cos:.9} max component error {err:.6}",
                case.name
            );
            assert!((ng - 1.0).abs() < 1e-4, "{} row {i}: norm {ng}", case.name);
            worst_cos = worst_cos.min(cos);
            worst_err = worst_err.max(err);
        }
    }
    // the text gate's bar: the same int8-activation class on both sides
    println!("worst cosine {worst_cos:.9}, worst component error {worst_err:.6}");
    assert!(worst_cos > 0.9995, "worst cosine {worst_cos}");
    assert!(worst_err < 0.004, "worst component error {worst_err}");
}

#[test]
fn embeddinggemma2_picture_vectors_do_not_depend_on_the_pass() {
    let Some((_, cases)) = media_fixture() else {
        return;
    };
    let Some(mut m) = load_with_pictures(8192) else {
        return;
    };
    let mut seqs = Vec::new();
    let mut media = Vec::new();
    for c in &cases {
        seqs.extend(c.ids.iter().cloned());
        media.extend(c.media.iter().cloned());
    }
    let alone: Vec<Vec<f32>> = seqs
        .iter()
        .zip(&media)
        .map(|(s, md)| {
            m.embed_media(std::slice::from_ref(s), std::slice::from_ref(md), None)
                .expect("alone")
                .remove(0)
        })
        .collect();
    let packed = m.embed_media(&seqs, &media, None).expect("packed");
    seqs.reverse();
    media.reverse();
    let mut reversed = m.embed_media(&seqs, &media, None).expect("reversed");
    reversed.reverse();
    assert_eq!(bits(&alone), bits(&packed), "packed differs from alone");
    assert_eq!(bits(&alone), bits(&reversed), "reversed differs from alone");
}

#[test]
fn embeddinggemma2_refuses_mismatched_media() {
    use paddock_engine::gpu_model::embedding_gemma2::media::{
        AUDIO_TOKEN, BOI_TOKEN, EOI_TOKEN, IMAGE_TOKEN,
    };
    let house = picture("house.jpg");
    let block = |n: usize| {
        let mut s = vec![2, BOI_TOKEN];
        s.extend(std::iter::repeat_n(IMAGE_TOKEN, n));
        s.extend([EOI_TOKEN, 1]);
        s
    };
    // a text-only endpoint refuses pictures outright
    if let Some(text_only) = load(1024) {
        let err = text_only
            .validate_media(&[block(266)], &[vec![house.clone()]])
            .expect_err("no tower");
        assert!(err.contains("no pictures"), "{err}");
    }
    let Some(m) = load_with_pictures(1024) else {
        return;
    };
    assert!(
        m.validate_media(&[block(266)], &[vec![house.clone()]])
            .is_ok()
    );
    assert!(
        m.validate_media(&[block(265)], &[vec![house.clone()]])
            .is_err(),
        "short run"
    );
    assert!(
        m.validate_media(&[block(266)], &[vec![]]).is_err(),
        "run without its picture"
    );
    assert!(
        m.validate_media(&[vec![2, 100, 1]], &[vec![house.clone()]])
            .is_err(),
        "picture without its run"
    );
    let audio = MmChunk::Audio {
        samples: vec![0.0; 16000],
        mel: None,
    };
    assert!(
        m.validate_media(&[vec![2, AUDIO_TOKEN, 1]], &[vec![audio]])
            .is_err(),
        "audio before its tower"
    );
    // a bare placeholder id stays refused on the text path
    assert!(m.validate(&[block(266)]).is_err());
}

// ---- audio -------------------------------------------------------------------
//
// The reference: the same llama-server and patched mmproj given the clips
// (`media-audio-llama.json`: LibriSpeech speech at 6 / 15 / 30 s and the
// synthetic ASR fixtures, alone, interleaved with text and a picture, and in
// a mixed batch). Its frontend floors the mel with ln(max(mel, 1e-3)) where
// the processor adds the 1e-3, so the vector gate runs the tower on llama's
// floor; the frontend itself is held to the processor's own log-mel (written
// by the oracle's `audio --features` from transformers), and the served
// vectors to transformers' own pipeline on the official checkpoint in f32
// (`media-audio-hf.json`, the tie-breaker; skipped when not recorded).
//
// The bars: speech-like items (speech, noise, interleaved) as the text gate,
// cosine above 0.9995 and components within 0.004. The synthetic signals
// get 0.9975 and 0.01 - not slack but the model's own precision class on
// them: transformers in bf16 against transformers in f32, same checkpoint,
// lands at 0.99763 / 0.00935 on silence and 0.99935 on the chirp (GB10,
// 2026-10-07), while on speech it agrees to 0.99998. A real defect (a
// wrong mask, rel-shift or clamp) moves speech far past 0.9995.

/// Every row of `got` against `cases`' vectors, the bars above; prints all.
fn audio_gate(label: &str, cases: &[MediaCase], got: &[Vec<Vec<f32>>]) -> Result<(), String> {
    let mut fail = Vec::new();
    println!("{label}:");
    for (case, rows) in cases.iter().zip(got) {
        for (i, (g, r)) in rows.iter().zip(&case.embeddings).enumerate() {
            let (cos, err) = cosine(g, r);
            let (bar, comp) = if case.synthetic[i] {
                (0.9975, 0.01)
            } else {
                (0.9995, 0.004)
            };
            println!(
                "  {} row {i}: cosine {cos:.9} max component error {err:.6}",
                case.name
            );
            if cos <= bar || err >= comp {
                fail.push(format!("{} row {i}: {cos:.6} / {err:.4}", case.name));
            }
        }
    }
    if fail.is_empty() {
        Ok(())
    } else {
        Err(fail.join(", "))
    }
}

fn audio_fixture() -> Option<(serde_json::Value, Vec<MediaCase>)> {
    media_fixture_at("PADDOCK_EG2_AUDIO_FIXTURE", "media-audio-llama.json")
}

fn cosine(g: &[f32], r: &[f32]) -> (f64, f64) {
    let dot: f64 = g.iter().zip(r).map(|(a, b)| *a as f64 * *b as f64).sum();
    let ng: f64 = g.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nr: f64 = r.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let err = g
        .iter()
        .zip(r)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0f64, f64::max);
    (dot / (ng * nr), err)
}

#[test]
fn embeddinggemma2_audio_frontend_matches_the_processor() {
    let Some((fixture, _)) = audio_fixture() else {
        return;
    };
    let Some(mut m) = load_with_pictures(1024) else {
        return;
    };
    let tower = m.audio_tower().expect("audio tower");
    let mut worst = 0f64;
    for (name, info) in fixture["clips"].as_object().expect("clips") {
        let Some(hf) = info["hf_mel"].as_str() else {
            common::missing("the processor's log-mel (record the clips with `audio --features`)");
            return;
        };
        let want: Vec<f32> = std::fs::read(reference_dir().join("media280-prep").join(hf))
            .expect("hf mel")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let MmChunk::Audio { samples, .. } = clip(name) else {
            unreachable!()
        };
        let got = tower.log_mel(&samples).expect("log-mel");
        assert_eq!(got.len(), want.len(), "{name}: frames");
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0f64, f64::max);
        println!(
            "{name}: {} frames, worst |log-mel diff| {err:.3e}",
            got.len() / 128
        );
        worst = worst.max(err);
    }
    // f32 FFT and filterbank against numpy's (float64 filterbank and log):
    // speech agrees to 1.2e-4. A pure tone's far bins hold nothing but its
    // peak's FFT rounding, which two FFTs round differently, and the log
    // stretches that near the 1e-3 floor: 3.5e-3 on the tones and the chirp
    assert!(worst < 5e-3, "worst log-mel difference {worst}");
}

#[test]
fn embeddinggemma2_audio_matches_llamacpp_on_the_same_gguf() {
    let Some((_, cases)) = audio_fixture() else {
        return;
    };
    let Some(mut m) = load_with_pictures(8192) else {
        return;
    };
    let run = |m: &mut GpuEmbeddingGemma2| -> Vec<Vec<Vec<f32>>> {
        cases
            .iter()
            .map(|c| m.embed_media(&c.ids, &c.media, None).expect("embed"))
            .collect()
    };
    // context: the processor's frontend, as served - the floors part on the
    // synthetic clips (llama.cpp's own vectors sit 0.979 from transformers'
    // on the 30 s synthetic clip)
    let served = run(&mut m);
    let _ = audio_gate("processor frontend (served, context only)", &cases, &served);
    m.audio_tower().expect("audio").set_llama_floor(true);
    let gated = run(&mut m);
    audio_gate("llama.cpp's floor (the gate)", &cases, &gated).expect("audio parity");
}

#[test]
fn embeddinggemma2_audio_matches_the_processor_pipeline() {
    let Some((_, cases)) = media_fixture_at("PADDOCK_EG2_AUDIO_HF_FIXTURE", "media-audio-hf.json")
    else {
        return;
    };
    let Some(mut m) = load_with_pictures(8192) else {
        return;
    };
    let got: Vec<Vec<Vec<f32>>> = cases
        .iter()
        .map(|c| m.embed_media(&c.ids, &c.media, None).expect("embed"))
        .collect();
    audio_gate("served, against transformers f32", &cases, &got).expect("processor parity");
}

#[test]
fn embeddinggemma2_audio_vectors_do_not_depend_on_the_pass() {
    let Some((_, cases)) = audio_fixture() else {
        return;
    };
    let Some(mut m) = load_with_pictures(8192) else {
        return;
    };
    let mut seqs = Vec::new();
    let mut media = Vec::new();
    for c in cases.iter().filter(|c| !c.name.contains("30")) {
        seqs.extend(c.ids.iter().cloned());
        media.extend(c.media.iter().cloned());
    }
    let alone: Vec<Vec<f32>> = seqs
        .iter()
        .zip(&media)
        .map(|(s, md)| {
            m.embed_media(std::slice::from_ref(s), std::slice::from_ref(md), None)
                .expect("alone")
                .remove(0)
        })
        .collect();
    let packed = m.embed_media(&seqs, &media, None).expect("packed");
    seqs.reverse();
    media.reverse();
    let mut reversed = m.embed_media(&seqs, &media, None).expect("reversed");
    reversed.reverse();
    assert_eq!(bits(&alone), bits(&packed), "packed differs from alone");
    assert_eq!(bits(&alone), bits(&reversed), "reversed differs from alone");
}

#[test]
fn embeddinggemma2_refuses_mismatched_audio() {
    use paddock_engine::gpu_model::embedding_gemma2::audio::{AUDIO_MAX_SAMPLES, audio_tokens};
    use paddock_engine::gpu_model::embedding_gemma2::media::{AUDIO_TOKEN, BOA_TOKEN, EOA_TOKEN};
    let block = |n: usize| {
        let mut s = vec![2, BOA_TOKEN];
        s.extend(std::iter::repeat_n(AUDIO_TOKEN, n));
        s.extend([EOA_TOKEN, 1]);
        s
    };
    let one = |n: usize| MmChunk::Audio {
        samples: vec![0.0; n],
        mel: None,
    };
    // the processor's arithmetic: 1 s is 99 frames, 25 tokens; 30 s is 750
    assert_eq!(audio_tokens(16_000), Ok(25));
    assert_eq!(audio_tokens(AUDIO_MAX_SAMPLES), Ok(750));
    assert_eq!(audio_tokens(161), Ok(1));
    assert!(audio_tokens(160).is_err(), "under one frame");
    assert!(audio_tokens(AUDIO_MAX_SAMPLES + 1).is_err(), "past 30 s");
    let Some(m) = load_with_pictures(1024) else {
        return;
    };
    assert!(m.validate_media(&[block(25)], &[vec![one(16_000)]]).is_ok());
    assert!(
        m.validate_media(&[block(24)], &[vec![one(16_000)]])
            .is_err(),
        "short run"
    );
    assert!(
        m.validate_media(&[block(25)], &[vec![]]).is_err(),
        "run without its clip"
    );
    let long = one(AUDIO_MAX_SAMPLES + 16_000);
    assert!(
        m.validate_media(&[block(775)], &[vec![long]]).is_err(),
        "past 30 s"
    );
    assert!(
        m.validate_media(&[block(25)], &[vec![picture("house.jpg")]])
            .is_err(),
        "a picture in an audio run"
    );
}

/// The catalog serves the projector cut in two (a picture file and an audio
/// file, each its own download): attached one after the other they must
/// embed exactly as the upstream file holding both - a cut, not a
/// conversion. Also: `audio = false` leaves the audio tower in a combined
/// file unloaded.
#[test]
fn embeddinggemma2_split_towers_embed_like_the_combined_file() {
    let Some((_, cases)) = audio_fixture() else {
        return;
    };
    let (Some(vision), Some(audio)) = (
        common::model(
            "PADDOCK_EG2_MMPROJ_VISION",
            common::EMBEDDINGGEMMA2_MMPROJ_VISION,
        ),
        common::model(
            "PADDOCK_EG2_MMPROJ_AUDIO",
            common::EMBEDDINGGEMMA2_MMPROJ_AUDIO,
        ),
    ) else {
        return;
    };
    let Some(mut combined) = load_with_pictures(8192) else {
        return;
    };
    let mut split = load(8192).expect("load");
    for file in [&vision, &audio] {
        let map = MappedGguf::open(file).expect("open cut");
        split.attach_mmproj(&map, 280, true).expect("attach cut");
    }
    assert!(split.serves_images() && split.serves_audio());
    let case = cases
        .iter()
        .find(|c| c.name == "picture_and_clip")
        .expect("picture_and_clip");
    let a = combined
        .embed_media(&case.ids, &case.media, None)
        .expect("combined");
    let b = split
        .embed_media(&case.ids, &case.media, None)
        .expect("split");
    assert_eq!(bits(&a), bits(&b), "the cut changed a vector");

    let mut no_audio = load(1024).expect("load");
    let mmproj =
        common::model("PADDOCK_EG2_MMPROJ", common::EMBEDDINGGEMMA2_MMPROJ).expect("mmproj");
    no_audio
        .attach_mmproj(&MappedGguf::open(&mmproj).expect("open"), 280, false)
        .expect("attach");
    assert!(no_audio.serves_images() && !no_audio.serves_audio());
}

/// `EG2_GOLDEN=<dir>`: record (`EG2_GOLDEN_WRITE=1`) or check every vector
/// of a spread of passes as raw f32 bits - each text case alone, all of them
/// packed, 8 and 32 copies of the query, each picture / clip case alone and
/// all of them packed - so a kernel change that claims to land the same bits
/// (a GEMM tile election, an attention grid split) is held to the build
/// before it across every row count the elections cross.
#[test]
fn embeddinggemma2_matches_recorded_goldens() {
    let Some(dir) = std::env::var_os("EG2_GOLDEN").map(std::path::PathBuf::from) else {
        return;
    };
    let write = std::env::var_os("EG2_GOLDEN_WRITE").is_some();
    let Some(cases) = fixture() else { return };
    let Some(mut m) = load_with_pictures(8192) else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("golden dir");
    let mut passes: Vec<(String, Vec<Vec<u32>>, Option<Vec<Vec<MmChunk>>>)> = Vec::new();
    for c in &cases {
        passes.push((c.name.clone(), c.ids.clone(), None));
    }
    passes.push((
        "text_packed".into(),
        cases.iter().flat_map(|c| c.ids.clone()).collect(),
        None,
    ));
    let query = cases.iter().find(|c| c.name == "short").expect("short").ids[0].clone();
    for n in [8usize, 32] {
        passes.push((format!("queries{n}"), vec![query.clone(); n], None));
    }
    let mut media_cases = Vec::new();
    if let Some((_, c)) = media_fixture() {
        media_cases.extend(c);
    }
    if let Some((_, c)) = audio_fixture() {
        media_cases.extend(c);
    }
    for c in &media_cases {
        passes.push((
            format!("media_{}", c.name),
            c.ids.clone(),
            Some(c.media.clone()),
        ));
    }
    if !media_cases.is_empty() {
        passes.push((
            "media_packed".into(),
            media_cases.iter().flat_map(|c| c.ids.clone()).collect(),
            Some(media_cases.iter().flat_map(|c| c.media.clone()).collect()),
        ));
    }
    for (name, ids, media) in &passes {
        let got = match media {
            Some(md) => m.embed_media(ids, md, None).expect("embed"),
            None => m.embed(ids, None).expect("embed"),
        };
        let bytes: Vec<u8> = got.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let file = dir.join(format!("{name}.f32"));
        if write {
            std::fs::write(&file, bytes).expect("write golden");
        } else {
            let want = std::fs::read(&file).expect("golden");
            assert!(want == bytes, "{name} moved");
        }
    }
    println!(
        "{} passes {}",
        passes.len(),
        if write { "recorded" } else { "match" }
    );
}

/// Slot 829 spreads an attention tile's warps over 2 or 4 blocks; every row
/// must land the bits slot 809's grid lands - on packed sequences of mixed
/// lengths (ragged tiles, a 1,500-token one past the 512-key window), on the
/// sliding hd 256 layers and the full hd 512 ones.
#[test]
fn embeddinggemma2_split_attention_is_bit_identical() {
    use paddock_engine::gpu::{EG2_ATTN_ROWS, EG2_TILE_SHIFT};
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_eg2_attn_s() {
        common::missing("a kernel pack with slot 829 (the split attention grid)");
        return;
    }
    let lens = [270usize, 37, 1500, 5, 700, 64];
    let rows: usize = lens.iter().sum();
    let mut cu = vec![0u32];
    for l in lens {
        cu.push(cu.last().unwrap() + l as u32);
    }
    // a fixed LCG: values on the scale the head pass lands (unit-norm rows
    // scaled, so scores spread across the softmax's range)
    let mut seed = 0x2545_f491_u64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let d_cu = exec.to_device_u32(&cu).expect("cu");
    for (hd, window) in [(256usize, 512usize), (256, 0), (512, 0)] {
        let per = if hd == 256 {
            EG2_ATTN_ROWS[0]
        } else {
            EG2_ATTN_ROWS[1]
        };
        let mut tiles = Vec::new();
        for (s, &l) in lens.iter().enumerate() {
            for t in 0..l.div_ceil(per) {
                tiles.push(((s as u32) << EG2_TILE_SHIFT) | t as u32);
            }
        }
        let d_tiles = exec.to_device_u32(&tiles).expect("tiles");
        let q = exec
            .to_device_f16(&rnd(rows * 4 * hd, 0.25), "q")
            .expect("q");
        let k = exec.to_device_f16(&rnd(rows * 512, 1.0), "k").expect("k");
        let v = exec.to_device_f16(&rnd(rows * 512, 1.0), "v").expect("v");
        let mut outs = Vec::new();
        for split in [1usize, 2, 4] {
            let mut out = exec.stream.alloc_zeros::<f32>(rows * 4 * hd).expect("out");
            exec.eg2_attn(
                &q,
                &k,
                &v,
                &d_cu,
                &d_tiles,
                tiles.len(),
                &mut out,
                rows,
                hd,
                window,
                split,
            )
            .expect("attn");
            outs.push(exec.to_host_len(&out, rows * 4 * hd).expect("host"));
        }
        for (i, o) in outs.iter().enumerate().skip(1) {
            let differ = o
                .iter()
                .zip(&outs[0])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "hd {hd} window {window} split {}: {differ} of {} differ",
                [1, 2, 4][i],
                o.len()
            );
            assert_eq!(differ, 0, "the split grid moved a row");
        }
        assert!(outs[0].iter().all(|x| x.is_finite()));
    }
}
