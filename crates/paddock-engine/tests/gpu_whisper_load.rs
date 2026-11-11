//! Loader milestone for the whisper family: open the real
//! KB-Whisper GGUF (our conversion - our whisper converter),
//! upload all 1260 planes resident, and check the declared geometry + decode
//! contract survived the trip. The encoder/decoder graphs are gated
//! elsewhere; this test is the "weights resident with an honest VRAM ledger"
//! gate.
//!
//! Heavy (uploads ~3.2 GB) - gated on the model file, a built pack, and a
//! CUDA device, all skip-cleanly like the sibling load tests.

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::whisper::GpuWhisper;
use paddock_models::mapped::MappedGguf;

/// Every Nordic sibling that is present, not just the first one brought up.
/// KB-Whisper ships `proj_out` as its own plane; NB-Whisper and Røst omit it
/// (the head is tied to the token embedding), so loading all three is what
/// actually covers the tie fallback. `WHISPER_GGUF` overrides with one file.
///
/// Whichever are present is the set - a box with one checkpoint still gates
/// that one, and finding none is what `common` reports as a skip.
fn model_paths() -> Vec<std::path::PathBuf> {
    if let Ok(p) = std::env::var("WHISPER_GGUF") {
        return vec![std::path::PathBuf::from(p)];
    }
    common::WHISPER_NORDIC
        .iter()
        .filter_map(|names| {
            common::model_roots()
                .iter()
                .find_map(|r| names.iter().map(|n| r.join(n)).find(|p| p.exists()))
        })
        .collect()
}

#[test]
fn loads_whisper_and_geometry_matches() {
    let paths: Vec<_> = model_paths().into_iter().filter(|p| p.exists()).collect();
    if paths.is_empty() {
        common::missing("no whisper checkpoint found (set WHISPER_GGUF for one file)");
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    for path in paths {
        let map = MappedGguf::open(&path).expect("open gguf");
        assert_eq!(map.gguf().architecture(), Some("whisper"), "arch string");

        // 4096 is the runner's default serve ctx - the loader must cap at the
        // trained 448-position table instead of refusing to serve.
        let m = GpuWhisper::load(Arc::clone(&exec), &map, 4096).expect("load whisper");

        // whisper-large-v3 class facts every Nordic fine-tune shares
        assert_eq!(m.n_layers(), (32, 32), "encoder/decoder depth");
        assert!(m.vocab() > 50_000, "vocab {}", m.vocab());
        // universal whisper decode contract: sot <|startoftranscript|> = 50258,
        // eot <|endoftext|> = 50257 (stamped from the checkpoint's own
        // generation_config - equality here proves the ids survived conversion)
        assert_eq!(m.contract_tokens(), (50258, 50257), "sot/eot ids");
        // the ledger must account for a large-v3's ~3.2 GB of f16 planes -
        // a silently-skipped tensor group would show up as a shortfall here
        let gib = m.weights_bytes() as f64 / (1u64 << 30) as f64;
        assert!(
            (2.5..4.0).contains(&gib),
            "resident {gib:.2} GiB out of the large-v3 class"
        );
        eprintln!(
            "{}: resident {gib:.2} GiB, vocab {}",
            path.file_name().unwrap().to_string_lossy(),
            m.vocab()
        );
    }
}

/// A window holding a sliver of audio used to come back as a confident
/// `<|nospeech|>` TOKEN: nothing suppressed it, the no-speech guard keeps a
/// confident window, and the word-timing pass then refused the transcript
/// (production: every clip whose tail window held under ~3 s failed word
/// timestamps with "got the special token 50363"). The reference masks the
/// prompt block and `<|nospeech|>` before every softmax; so does the pick
/// kernel now. A transcript may hold text, timestamps and `<|eot|>` only -
/// timestamps included even on the no-timestamps prompt, where the reference
/// leaves them unsuppressed and the transcriber strips them (a 0.1 s tone
/// comes back as a lone `<|0.00|>`, i.e. an empty transcript).
#[test]
fn a_sliver_window_never_yields_a_special_token() {
    let Some(path) = model_paths().into_iter().find(|p| p.exists()) else {
        common::missing("no whisper checkpoint found (set WHISPER_GGUF for one file)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let mut m = GpuWhisper::load(Arc::clone(&exec), &map, 448).expect("load whisper");
    let (_, eot) = m.contract_tokens();
    let ts_begin = m.time_scale().begin; // `<|0.00|>`
    let tone = {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/audio-formats/tone.wav"),
        )
        .expect("tone fixture");
        let wav = paddock_engine::audio::decode::decode_audio(&bytes).expect("decode tone");
        paddock_engine::audio::resample::resample(&wav.samples, wav.sample_rate, 16000)
            .expect("resample tone")
    };
    // 0.1 s of silence, 0.1 s of tone, 2 s of tone, and a 30.1 s clip whose
    // second window is the sliver - the shapes that failed in production
    let silence = vec![0.0f32; 1600];
    let mut long = vec![0.0f32; 16000 * 30];
    long.extend_from_slice(&tone[..tone.len().min(1600)]);
    for (name, clip) in [
        ("0.1 s silence", silence.as_slice()),
        ("0.1 s tone", &tone[..tone.len().min(1600)]),
        ("2 s tone", &tone[..tone.len().min(32000)]),
        ("30.1 s", long.as_slice()),
    ] {
        let (_, windows) = m.transcribe(clip, Some("sv"), 64).expect(name);
        for (w, ids) in windows.iter().enumerate() {
            if let Some(&t) = ids.iter().find(|&&t| t >= eot && t < ts_begin) {
                panic!("{name}: window {w} emitted special token {t} (eot {eot}): {ids:?}");
            }
        }
        eprintln!("{name}: {} window(s), tokens {:?}", windows.len(), windows);
    }
}
