//! Gradium Phonon 31M on CUDA against the model's F64 equations.
//!
//! The fixture directory (`PADDOCK_PHONON_REFERENCE_F64`, or
//! `phonon-oracle-f64` under a model root) is written by the independent
//! PyTorch equations on the same checkpoint, run with `--f64` (the same
//! validation-only script writes the Metal gate's F32 fixture without it):
//! the load-time constants, FlowLM's output on the fixed
//! sentence, 64 zero-noise frames' latents and EOS logits, and Mimi's
//! waveform of those frames decoded as one full-sequence causal pass.
//!
//! Why F64 and not the same script's F32 output (the Metal gate's fixture):
//! the autoregressive chain amplifies rounding through a stretch of frames
//! (33-45 on this sentence), where the F32 reference itself drifts 1.5e-4
//! from F64 - past the Metal lane's 1e-4 frame bound - and its waveform
//! 6.7e-4. Measured against an F32 run, an F32 engine is graded on the
//! reference's rounding as much as its own.
//!
//! Two frame checks, for the same reason:
//!   - teacher forced (each frame reads the reference's previous latent):
//!     a frame's own step, latents and EOS within 1e-5 - F32 class, ten
//!     times tighter than the old frame bound (measured: <= 6.2e-6);
//!   - free running (each frame reads its own previous latent): the
//!     trajectory, within 2e-4 a frame - just above the F32 reference's
//!     own 1.5e-4 at the sensitive stretch, where a tighter bound would
//!     grade which way rounding amplified, not the arithmetic - and the
//!     waveform within the Metal lane's 0.002.
//!
//! Set `PADDOCK_PHONON_REFERENCE_F32` to print the F32 reference's own
//! distance beside ours.
//!
//! Beyond the fixture: whole-utterance generation keeps the EOS/tail contract
//! and the decoder's samples, batch size and cancellation never change a
//! sample, and a rerun is bit-identical.
//!
//! `PADDOCK_HEAVY_TESTS=1` runs these.

mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use paddock_engine::gpu_model::phonon::{CODEC_BATCH, GpuPhonon};
use paddock_engine::phonon::{Backend, Noise, Options};
use paddock_models::phonon::{FRAME_SAMPLES, MAX_FRAMES, SAMPLE_RATE};

const TEXT: &str = "Hello. This is Paddock speaking with its own native speech engine.";

/// A fixture tensor, widened to f64 (`.f64` files, or the F32 run's `.f32`).
fn read(dir: &Path, name: &str) -> Vec<f64> {
    let f64_file = dir.join(format!("{name}.f64"));
    if let Ok(bytes) = std::fs::read(&f64_file) {
        return bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
    }
    let bytes = std::fs::read(dir.join(format!("{name}.f32")))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join(name).display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f64::from(f32::from_le_bytes(*c)))
        .collect()
}

/// Relative L2 of `actual` against the fixture's `name`.
fn rel_l2(dir: &Path, name: &str, actual: &[f32]) -> f64 {
    let expected = read(dir, name);
    assert_eq!(actual.len(), expected.len(), "{name}: length");
    let err = actual
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (f64::from(a) - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let mag = expected.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-8);
    err / mag
}

/// [`rel_l2`], asserted under `tolerance`.
fn check(dir: &Path, name: &str, actual: &[f32], tolerance: f64) -> f64 {
    let rel = rel_l2(dir, name, actual);
    assert!(
        rel < tolerance,
        "{name}: relative L2 {rel:e} >= {tolerance:e}"
    );
    rel
}

fn show(v: &[f64]) -> String {
    v.iter()
        .map(|e| format!("{e:.1e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn load() -> Option<(GpuPhonon, PathBuf)> {
    if !common::heavy() {
        return None;
    }
    let model = common::model_dir("PADDOCK_PHONON_MODEL", &["Phonon-31M-EN"])?;
    let exec = common::gpu_arc()?;
    let t = Instant::now();
    let m = GpuPhonon::load(exec, &model).expect("the registry checkpoint loads");
    eprintln!(
        "loaded in {:.3} s: weights {} B, workspace {} B",
        t.elapsed().as_secs_f64(),
        m.weight_bytes(),
        m.workspace_bytes()
    );
    Some((m, model))
}

fn generate(m: &mut GpuPhonon, text: &str, options: Options) -> Vec<f32> {
    let mut pcm = Vec::new();
    m.generate(text, options, &mut |x| {
        pcm.extend(x);
        true
    })
    .expect("a valid request generates to its end of speech");
    pcm
}

#[test]
fn phonon_matches_the_f64_equations() {
    let Some((mut m, _)) = load() else { return };
    let Some(dir) = common::model_dir("PADDOCK_PHONON_REFERENCE_F64", &["phonon-oracle-f64"])
    else {
        return;
    };
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["arithmetic"], "F64",
        "the fixture must be the --f64 run"
    );
    let [condition, time, voice, bos] = m.constants().unwrap();
    for (name, v) in [
        ("condition", &condition),
        ("time", &time),
        ("voice", &voice),
        ("bos", &bos),
    ] {
        eprintln!("{name}: {:e}", check(&dir, name, v, 1e-5));
    }
    m.reset(TEXT).unwrap();
    eprintln!(
        "hidden: {:e}",
        check(&dir, "hidden", &m.hidden().unwrap(), 1e-4)
    );
    let frames = manifest["frames"].as_u64().unwrap() as usize;
    assert!(
        (33..=MAX_FRAMES).contains(&frames),
        "the fixture must cross the decoder's ring wrap"
    );
    let f32_ref = std::env::var_os("PADDOCK_PHONON_REFERENCE_F32").map(PathBuf::from);
    let (mut latents, mut endings, mut theirs) = (Vec::new(), Vec::new(), Vec::new());
    let mut pcm = Vec::new();
    for i in 0..frames {
        let eos = m.sample_batch(&[[0.0; 32]], i == 0).unwrap();
        let name = format!("latent-{i}");
        latents.push(rel_l2(&dir, &name, &m.latents(1).unwrap()));
        endings.push(rel_l2(&dir, &format!("eos-{i}"), &eos));
        if let Some(r) = &f32_ref {
            let v: Vec<f32> = read(r, &name).iter().map(|&v| v as f32).collect();
            theirs.push(rel_l2(&dir, &name, &v));
        }
        pcm.extend(m.decode_batch(1).unwrap());
    }
    eprintln!("latents: {}", show(&latents));
    eprintln!("eos: {}", show(&endings));
    if let Some(r) = &f32_ref {
        eprintln!("F32 reference latents: {}", show(&theirs));
        let v: Vec<f32> = read(r, "pcm").iter().map(|&v| v as f32).collect();
        eprintln!("F32 reference pcm: {:e}", rel_l2(&dir, "pcm", &v));
    }
    // teacher forced: each frame reads the reference's previous latent, so a
    // frame's error is its own step's arithmetic, not the chain's
    m.reset(TEXT).unwrap();
    let (mut forced, mut forced_eos) = (Vec::new(), Vec::new());
    for i in 0..frames {
        if i > 0 {
            let prev: Vec<f32> = read(&dir, &format!("latent-{}", i - 1))
                .iter()
                .map(|&v| v as f32)
                .collect();
            m.force_prev(&prev).unwrap();
        }
        let eos = m.sample_batch(&[[0.0; 32]], i == 0).unwrap();
        forced.push(rel_l2(&dir, &format!("latent-{i}"), &m.latents(1).unwrap()));
        forced_eos.push(rel_l2(&dir, &format!("eos-{i}"), &eos));
    }
    eprintln!("teacher-forced latents: {}", show(&forced));
    eprintln!("teacher-forced eos: {}", show(&forced_eos));
    for (i, (l, e)) in forced.iter().zip(&forced_eos).enumerate() {
        assert!(
            *l < 1e-5 && *e < 1e-5,
            "teacher-forced frame {i}: latent {l:e}, eos {e:e} (bound 1e-5)"
        );
    }
    for (i, (l, e)) in latents.iter().zip(&endings).enumerate() {
        assert!(
            *l < 2e-4 && *e < 2e-4,
            "free-running frame {i}: latent {l:e}, eos {e:e} (bound 2e-4)"
        );
    }
    eprintln!("pcm: {:e}", check(&dir, "pcm", &pcm, 0.002));
    // the whole-utterance stop/tail contract at temperature 0
    let first_eos = (0..frames)
        .find(|i| read(&dir, &format!("eos-{i}"))[0] > 4.0)
        .expect("the fixture must reach end of speech");
    let zero = Options {
        seed: 0,
        temperature: 0.0,
    };
    let generated = generate(&mut m, TEXT, zero);
    assert_eq!(generated.len(), (first_eos + 2) * FRAME_SAMPLES);
    assert!(
        generated == pcm[..generated.len()],
        "batched generation must keep the frame-by-frame decoder's samples"
    );
    let a = generate(&mut m, TEXT, Options::default());
    let b = generate(&mut m, TEXT, Options::default());
    assert!(a == b, "a rerun must be bit-identical");
}

#[test]
fn phonon_batches_and_cancellation_keep_every_sample() {
    let Some((mut m, _)) = load() else { return };
    let text = "Batch boundaries must not change the sound of this voice.";
    let options = Options {
        seed: 42,
        temperature: 0.3,
    };
    let mut noise = Noise::new(options);
    // three decoder ring wraps, partial batches included
    let noises: Vec<_> = (0..97).map(|_| noise.frame()).collect();
    m.reset(text).unwrap();
    let mut expected = Vec::new();
    let mut endings = Vec::new();
    for (i, n) in noises.iter().enumerate() {
        endings.extend(m.sample_batch(std::slice::from_ref(n), i == 0).unwrap());
        expected.extend(m.decode_batch(1).unwrap());
    }
    for batch in [2, 3, 4, 7, CODEC_BATCH] {
        m.reset(text).unwrap();
        let mut eos = m.sample_batch(&noises[..1], true).unwrap();
        let mut pcm = m.decode_batch(1).unwrap();
        for chunk in noises[1..].chunks(batch) {
            eos.extend(m.sample_batch(chunk, false).unwrap());
            pcm.extend(m.decode_batch(chunk.len()).unwrap());
        }
        assert_eq!(eos, endings, "EOS, batch {batch}");
        assert!(
            pcm == expected,
            "decoder samples changed with batch {batch}"
        );
    }
    for (text, options) in [
        (
            "Hello!",
            Options {
                seed: 0,
                temperature: 0.0,
            },
        ),
        ("A short sentence with a different ending.", options),
        (
            "When a local assistant reads a longer answer aloud, every word should survive the \
             final partial batch, and a cancelled request must not leak its voice or its \
             convolution history into the next answer.",
            Options {
                seed: 99,
                temperature: 0.3,
            },
        ),
    ] {
        // the frame-by-frame stream with the stop rule applied by hand
        m.reset(text).unwrap();
        let mut noise = Noise::new(options);
        let tail = if text.split_whitespace().count() <= 4 {
            3
        } else {
            1
        };
        let mut want = Vec::new();
        let mut stop = MAX_FRAMES + 1;
        for frame in 0..MAX_FRAMES {
            let eos = m.sample_batch(&[noise.frame()], frame == 0).unwrap()[0];
            want.extend(m.decode_batch(1).unwrap());
            if eos > 4.0 {
                stop = stop.min(frame + 1 + tail);
            }
            if frame + 1 == stop {
                break;
            }
        }
        assert!(stop <= MAX_FRAMES, "fixture did not finish");
        for cancel_at in [1, 5, 10] {
            let mut emitted = 0;
            m.generate(
                "Cancel this different utterance before completion.",
                options,
                &mut |_| {
                    emitted += 1;
                    emitted < cancel_at
                },
            )
            .unwrap();
            assert_eq!(emitted, cancel_at);
            let got = generate(&mut m, text, options);
            assert!(
                got == want,
                "EOS trimming or cancellation changed PCM: {text}"
            );
        }
    }
}

/// Latency and speed on the rival harness's three inputs (8 / 30 / 78
/// words): load to ready, time to the first complete 80 ms frame, and audio
/// seconds a wall second. A measurement, not a gate.
#[test]
fn phonon_speed() {
    let Some((mut m, _)) = load() else { return };
    let texts = [
        ("short", "Hello there, this is a local speech test."),
        (
            "medium",
            "The assistant can speak a complete answer while keeping every word on this \
             computer, allowing the listener to hear the first sentence before the rest of \
             the response is ready.",
        ),
        (
            "long",
            "When a local assistant reads a longer response aloud, the speech engine should \
             begin promptly and keep producing clear audio without pauses, while the \
             application remains responsive to new requests and allows the listener to stop \
             playback at any time, so this passage checks both the delay before the first \
             sound and the time needed to finish a continuous explanation about private \
             computing, reliable software, useful measurements, and the importance of \
             comparing identical models under carefully recorded conditions.",
        ),
    ];
    // warm every shape once
    for (_, t) in texts {
        generate(&mut m, t, Options::default());
    }
    for (name, text) in texts {
        let mut first = Vec::new();
        let mut speed = Vec::new();
        for rep in 0..7 {
            let options = Options {
                seed: rep,
                temperature: 0.3,
            };
            let start = Instant::now();
            let mut at = None;
            let mut samples = 0;
            m.generate(text, options, &mut |x| {
                at.get_or_insert_with(|| start.elapsed().as_secs_f64());
                samples += x.len();
                true
            })
            .unwrap();
            let wall = start.elapsed().as_secs_f64();
            first.push(at.unwrap() * 1e3);
            speed.push(samples as f64 / SAMPLE_RATE as f64 / wall);
        }
        first.sort_by(f64::total_cmp);
        speed.sort_by(f64::total_cmp);
        eprintln!(
            "{{\"text\":\"{name}\",\"first_frame_ms\":{:.3},\"realtime\":{:.2}}}",
            first[3], speed[3]
        );
    }
}
