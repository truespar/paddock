//! Nemotron 3.5 Transcribe 30B-A4B NVFP4: the language model on the speech
//! LM checkpoint (NVFP4 attention, bf16 head).
//!
//! Heavy (21.5 GB of weights): `PADDOCK_HEAVY_TESTS=1`. The checkpoint comes
//! from `TRANSCRIBE_DIR` or the models root.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "TRANSCRIBE_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4";

fn setup() -> Option<GpuNemotron> {
    if !common::heavy() {
        return None;
    }
    let exec = common::gpu_arc()?;
    if !exec.has_mamba2() || !exec.has_nvf4_moe() {
        common::missing("pack lacks the Nemotron kernels (mamba2, nvf4 moe)");
        return None;
    }
    let dir = common::model_dir(CKPT_ENV, &[CKPT_DIR])?;
    let t0 = std::time::Instant::now();
    let model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    Some(model)
}

fn argmax(l: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

/// The chat turn "What is the capital of France? Answer in one sentence."
/// with reasoning off, as the checkpoint's template renders it.
const CHAT: [u32; 28] = [
    10, 25708, 1010, 11, 1010, 10, 3263, 1010, 7493, 1395, 1278, 8961, 1307, 5498, 1063, 3450,
    1294, 1925, 19286, 1046, 11, 1010, 10, 1503, 19464, 1010, 12, 13,
];

#[test]
fn transcribe_language_model_answers_text() {
    let Some(mut model) = setup() else {
        return;
    };
    let mut logits = Vec::new();
    for &tok in &CHAT {
        logits = model.forward(tok).expect("prompt step");
    }
    let mut ours = Vec::new();
    for _ in 0..24 {
        let tok = argmax(&logits);
        ours.push(tok);
        if tok == 11 {
            break;
        }
        logits = model.forward(tok).expect("decode step");
    }
    println!("greedy: {ours:?}");
    assert!(ours.contains(&11), "no <|im_end|> within 24 tokens");
}

/// KL(p || q) of the softmaxes of two logit rows, and p's top-1 margin.
fn kl_and_margin(p: &[f32], q: &[f32]) -> (f64, f32) {
    let lse = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lp, lq) = (lse(p), lse(q));
    let kl = p
        .iter()
        .zip(q)
        .map(|(&a, &b)| {
            let la = a as f64 - lp;
            la.exp() * (la - (b as f64 - lq))
        })
        .sum::<f64>();
    let mut v: Vec<f32> = p.to_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    (kl.max(0.0), v[0] - v[1])
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let (words, _) = b.as_chunks::<4>();
    words.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

fn read_u32(p: &std::path::Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let (words, _) = b.as_chunks::<4>();
    words.iter().map(|&c| u32::from_le_bytes(c)).collect()
}

/// ||a - b|| / ||b|| over the whole plane, and the worst row's.
fn rel_err(a: &[f32], b: &[f32], width: usize) -> (f64, f64) {
    assert_eq!(a.len(), b.len(), "plane sizes differ");
    let (mut num, mut den, mut worst) = (0f64, 0f64, 0f64);
    for (ra, rb) in a.chunks(width).zip(b.chunks(width)) {
        let n: f64 = ra
            .iter()
            .zip(rb)
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
            .sum();
        let d: f64 = rb.iter().map(|y| f64::from(*y).powi(2)).sum();
        num += n;
        den += d;
        if d > 0.0 {
            worst = worst.max((n / d).sqrt());
        }
    }
    ((num / den.max(1e-30)).sqrt(), worst)
}

/// The speech encoder against the F64 Transformers oracle on the same
/// weights (the oracle script beside this lane writes the fixtures under
/// `<models>/transcribe-oracle/audio{,-w8a8}/<clip>/`, or
/// `TRANSCRIBE_AUDIO_ORACLE`), in the class the lane runs:
///
/// - W8A8 (the checkpoint's, served by default): against the oracle with the
///   same input quantization. Element-wise this can only be statistical - a
///   static-scale quantizer is discontinuous, so two W8A8 evaluations that
///   differ anywhere (here f32 sums against F64) part at an occasional e4m3
///   code and 32 blocks carry it to the class's own noise: the W8A8 oracle
///   sits median 1.3-1.7e-2 per row from the unquantized one, p99 0.13-0.6,
///   its worst (low-norm, near-silent) rows past 6x. The bands below hold
///   this lane inside that spread; the kernels themselves are held to the
///   arithmetic in `gpu_transcribe_w8a8.rs`.
/// - f16 (PADDOCK_TR_ENC_F16): against the oracle on the dequantized
///   weights, a 2^-11 class.
#[test]
fn transcribe_speech_encoder_matches_the_oracle() {
    let Some(mut model) = setup() else {
        return;
    };
    let w8a8 = model.speech_w8a8_probe();
    let dir = if w8a8 {
        "transcribe-oracle/audio-w8a8"
    } else {
        "transcribe-oracle/audio"
    };
    let root = std::env::var("TRANSCRIBE_AUDIO_ORACLE").map_or_else(
        |_| {
            common::model_roots()
                .into_iter()
                .map(|r| r.join(dir))
                .find(|p| p.is_dir())
        },
        |p| Some(std::path::PathBuf::from(p)),
    );
    let Some(root) = root else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    let mut clips: Vec<_> = std::fs::read_dir(&root)
        .expect("fixture root")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("rows.f32").is_file())
        .collect();
    clips.sort();
    assert!(!clips.is_empty(), "no fixtures under {}", root.display());
    let mut failed = Vec::new();
    for dir in clips {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let pcm = read_f32(&dir.join("pcm.f32"));
        let t0 = std::time::Instant::now();
        let (rows, st) = model.audio_probe(&pcm).expect("encode");
        let secs = t0.elapsed().as_secs_f64();
        let mel_ref = read_f32(&dir.join("mel.f32"));
        let states_ref = read_f32(&dir.join("states.f32"));
        let probs_ref = read_f32(&dir.join("probs.f32"));
        let codes_ref = read_u32(&dir.join("codes.u32"));
        let rows_ref = read_f32(&dir.join("rows.f32"));
        // the valid rows only: a padding row (the last stacked group past the
        // audio) is computed by the reference and dropped by both
        let n_tok = rows_ref.len() / 2688;
        let states_ref = &states_ref[..n_tok * 1280];
        let probs_ref = &probs_ref[..n_tok * 64];
        let codes_ref = &codes_ref[..n_tok];

        let mel_max = st
            .mel
            .iter()
            .zip(&mel_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let probs: Vec<f32> = st.probs[..n_tok * 8].iter().flatten().copied().collect();
        let p_max = probs
            .iter()
            .zip(probs_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let flips = st.codes[..n_tok]
            .iter()
            .zip(codes_ref)
            .filter(|(a, b)| a != b)
            .count();
        let (s_rel, s_worst) = rel_err(&st.states[..n_tok * 1280], states_ref, 1280);
        let (r_rel, r_worst) = rel_err(&rows, &rows_ref, 2688);
        println!(
            "{name}: {} samples, {} tokens in {secs:.2}s | mel max {mel_max:.2e} | speaker p max \
             {p_max:.2e}, {flips}/{} pattern flips ({} active) | states rel {s_rel:.2e} (worst row \
             {s_worst:.2e}) | rows rel {r_rel:.2e} (worst row {r_worst:.2e})",
            pcm.len(),
            rows.len() / 2688,
            codes_ref.len(),
            codes_ref.iter().filter(|&&c| c != 0).count(),
        );
        // per-row errors: a few rows are ill-conditioned in the reference
        // itself (silence between utterances - its own f32 run moves them
        // 100-1000x more than a typical row), so the band is on the row
        // distribution, the worst rows reported
        let mut per: Vec<f64> = st.states[..n_tok * 1280]
            .chunks(1280)
            .zip(states_ref.chunks(1280))
            .map(|(a, b)| rel_err(a, b, 1280).0)
            .collect();
        per.sort_by(f64::total_cmp);
        let (p50, p99) = (per[per.len() / 2], per[per.len() * 99 / 100]);
        println!(
            "  state rows: median {p50:.2e} p99 {p99:.2e} max {:.2e}",
            per[per.len() - 1]
        );
        assert_eq!(rows.len(), rows_ref.len(), "{name}: token count");
        // (state rows median, p99, projected rows) bands per class
        let (b50, b99, brows) = if w8a8 {
            (3e-2, 1.0, 0.15)
        } else {
            (2e-3, 3e-2, 3e-2)
        };
        for (bad, what) in [
            (mel_max >= 1e-3, "mel"),
            (p50 >= b50 || p99 >= b99, "speech states"),
            (flips * 100 > codes_ref.len(), "activity patterns"),
            (r_rel >= brows, "projected rows"),
        ] {
            if bad {
                failed.push(format!("{name}: {what} off the oracle"));
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The default transcription request around one recording: the template's
/// user turn up to the audio (its instruction and the separating space), and
/// from the end of the turn to the open reply with reasoning off.
const ASR_HEAD: [u32; 43] = [
    10, 25708, 1010, 11, 1010, 10, 3263, 1010, 32535, 15661, 1046, 26242, 1058, 1435, 29930, 1046,
    1488, 3592, 1058, 10564, 94503, 1044, 3907, 12369, 1428, 1321, 124440, 1046, 6392, 33361, 1058,
    34315, 65656, 1321, 8961, 3229, 1046, 27039, 1058, 21482, 2342, 1046, 1032,
];
const ASR_TAIL: [u32; 8] = [11, 1010, 10, 1503, 19464, 1010, 12, 13];

/// A recording through the whole lane: the serial multimodal prefill, then
/// greedy decode to `<|im_end|>`. Prints the ids (the decode is the
/// tokenizer's) for the clip the fixtures name.
#[test]
fn transcribe_recording_greedy() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    for clip in ["en10", "en95"] {
        let pcm = read_f32(&root.join(clip).join("pcm.f32"));
        let chunks = vec![
            paddock_engine::service::MmChunk::Text(ASR_HEAD.to_vec()),
            paddock_engine::service::MmChunk::Audio {
                samples: pcm,
                mel: None,
            },
            paddock_engine::service::MmChunk::Text(ASR_TAIL.to_vec()),
        ];
        let t0 = std::time::Instant::now();
        let (mut logits, n) = model
            .forward_multimodal(&chunks)
            .expect("prefill")
            .expect("audio prompt");
        let ttft = t0.elapsed().as_secs_f64();
        let mut ours = Vec::new();
        for _ in 0..512 {
            let tok = argmax(&logits);
            ours.push(tok);
            if tok == 11 {
                break;
            }
            logits = model.forward(tok).expect("decode step");
        }
        println!(
            "{clip}: {n} prompt rows, prefill {ttft:.2}s, {} tokens in {:.2}s\n  {ours:?}",
            ours.len(),
            t0.elapsed().as_secs_f64()
        );
        assert!(ours.contains(&11), "{clip}: no <|im_end|>");
    }
}

/// The FP8 MTP block drafts against a recording's transcript: one slot runs
/// draft + verify rounds, the other the plain greedy reference; the spec
/// stream must stay on the greedy one (a fork only on a reference near-tie)
/// and the drafts must land. The block sees the audio rows on its append
/// passes too, as the trunk does.
#[test]
fn transcribe_mtp_drafts_accept() {
    // the drafter is this gate's subject: state the operator's choice
    // SAFETY: before any model loads, on the test's only thread
    unsafe {
        std::env::remove_var("PADDOCK_NO_SPEC");
        std::env::set_var("PADDOCK_SPEC", "on");
    }
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    assert!(
        model.spec_capable(),
        "the MTP block must make Transcribe spec-capable"
    );
    let pcm = read_f32(&root.join("en95").join("pcm.f32"));
    let chunks = vec![
        paddock_engine::service::MmChunk::Text(ASR_HEAD.to_vec()),
        paddock_engine::service::MmChunk::Audio {
            samples: pcm,
            mel: None,
        },
        paddock_engine::service::MmChunk::Text(ASR_TAIL.to_vec()),
    ];
    let (l1, n) = model
        .forward_prefill_multimodal(1, &chunks)
        .expect("prefill 1");
    let (l0, _) = model
        .forward_prefill_multimodal(0, &chunks)
        .expect("prefill 0");
    assert_eq!(argmax(&l1), argmax(&l0), "slot boundary picks disagree");
    let margin = |l: &[f32]| {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in l {
            if v > a {
                b = a;
                a = v;
            } else if v > b {
                b = v;
            }
        }
        a - b
    };
    const ROUNDS: usize = 40;
    let k = 3usize;
    let mut b = vec![argmax(&l0)];
    let mut m = vec![margin(&l0)];
    for i in 0..ROUNDS * (k + 1) + 2 {
        let l = model
            .forward_batch(&[b[i]], &[(n + i) as u32])
            .expect("decode");
        b.push(argmax(&l));
        m.push(margin(&l));
    }
    let (mut committed, mut pending, mut pos) = (Vec::new(), b[0], n);
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let mut parted: Option<usize> = None;
    let t = std::time::Instant::now();
    for round in 0..ROUNDS {
        if pending == 11 {
            break;
        }
        let drafts = model
            .spec_draft_batch(&[(1usize, pending)], k)
            .expect("draft")
            .expect("mtp drafter engaged");
        let d = &drafts[0];
        assert_eq!(d.len(), k, "round {round}: MTP declined (cold coverage?)");
        let mut chunk = vec![pending];
        chunk.extend_from_slice(d);
        let picks = model
            .forward_spec_batch(&[(1usize, pos, chunk.clone())])
            .expect("verify round")
            .expect("engaged");
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        drafted += k;
        accepted += a;
        committed.extend_from_slice(&chunk[..=a]);
        if parted.is_none()
            && let Some(i) = (0..committed.len()).find(|&i| committed[i] != b[i])
        {
            parted = Some(i);
            // the rounds score in the prompt class (W4A4: every MoE input
            // re-quantized to NVFP4) and the greedy stream decodes in the
            // W16 class (bf16 activations): the two may part where the
            // greedy margin is about a logit - 1.09 on this clip once the
            // prompt folds bf16 partials (2026-11-14; with f32 partials it
            // never parts) - while a broken verify parts on any margin
            assert!(
                m[i] < 1.5,
                "round {round}: spec stream parts from greedy at {i} on a {:.3} margin",
                m[i]
            );
        }
        pending = picks[a];
        pos += a + 1;
    }
    let rate = accepted as f64 / drafted.max(1) as f64;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "MTP acceptance: {accepted}/{drafted} = {rate:.3} at K={k}; {} committed in {secs:.2}s \
         ({:.1} tok/s on one slot); parted from greedy at {parted:?}",
        committed.len(),
        committed.len() as f64 / secs
    );
    assert!(rate >= 0.5, "MTP acceptance {rate:.3} below the 0.5 floor");
}

/// Teacher-forced agreement with the official stack: vLLM 0.32 + NeMo's
/// plugin on this checkpoint transcribed FLEURS clips greedily (the
/// fixture keeps the prompt halves, each clip's samples and vLLM's output
/// ids); every one of those ids is fed back here and our top-1 is scored
/// against it. The two engines differ in class (vLLM: bf16 with the
/// encoder's activations quantized to e4m3), so a miss is expected only
/// where the reference itself is close - the misses' margins are reported.
#[test]
fn transcribe_teacher_forced_against_vllm() {
    teacher_forced(false);
}

/// [`transcribe_teacher_forced_against_vllm`] through the batch lane: the
/// slot prefill and one-row batched decode steps.
#[test]
fn transcribe_teacher_forced_against_vllm_batch_lane() {
    teacher_forced(true);
}

fn teacher_forced(batch: bool) {
    let Some(mut model) = setup() else {
        return;
    };
    if batch {
        assert!(model.batch_enable_probe(1).expect("enable") >= 1);
    }
    let Some(dir) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/tf"))
        .find(|p| p.join("vllm-fleurs60.json").is_file())
    else {
        common::missing("no transcribe-oracle/tf fixture");
        return;
    };
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("vllm-fleurs60.json")).unwrap()).unwrap();
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let (head, tail) = (ids(&fx["head"]), ids(&fx["tail"]));
    let (mut total, mut agree, mut exact) = (0usize, 0usize, 0usize);
    let mut misses: Vec<f32> = Vec::new();
    let clips = fx["clips"].as_array().unwrap();
    for c in clips {
        let name = c["name"].as_str().unwrap();
        let toks = ids(&c["tokens"]);
        let pcm = read_f32(&dir.join(format!("{name}.f32")));
        let chunks = vec![
            paddock_engine::service::MmChunk::Text(head.clone()),
            paddock_engine::service::MmChunk::Audio {
                samples: pcm,
                mel: None,
            },
            paddock_engine::service::MmChunk::Text(tail.clone()),
        ];
        let (mut logits, n) = if batch {
            model
                .forward_prefill_multimodal(0, &chunks)
                .expect("slot prefill")
        } else {
            model
                .forward_multimodal(&chunks)
                .expect("prefill")
                .expect("audio prompt")
        };
        let mut clip_ok = true;
        for (i, &t) in toks.iter().enumerate() {
            let ours = argmax(&logits);
            total += 1;
            if ours == t {
                agree += 1;
            } else {
                clip_ok = false;
                misses.push(logits[ours as usize] - logits[t as usize]);
            }
            if i + 1 < toks.len() {
                logits = if batch {
                    model
                        .forward_batch(&[t], &[(n + i) as u32])
                        .expect("forced step")
                } else {
                    model.forward(t).expect("forced step")
                };
            }
        }
        exact += usize::from(clip_ok);
    }
    misses.sort_by(f32::total_cmp);
    let rate = agree as f64 / total as f64;
    let near = misses.iter().filter(|&&m| m < 1.0).count();
    println!(
        "teacher-forced top-1 vs vLLM ({}): {agree}/{total} = {:.2}% ({exact}/{} clips with no miss); \
         {} misses, {near} within 1 logit, margins {:?}",
        if batch { "batch lane" } else { "serial lane" },
        100.0 * rate,
        clips.len(),
        misses.len(),
        misses
    );
    assert!(rate >= 0.97, "teacher-forced agreement {rate:.4} under 97%");
}

/// A wave of recording prompts admitted together (one speech-encoder call,
/// shared prefill passes) lands each prompt where admitting it alone does:
/// the same first pick, the same distribution. The encoder's rows are the
/// same bits either way (`transcribe_encoder_rows_batch_invariant`); the
/// language model's prompt passes are not - a wave's passes are wider and
/// some lanes take other row-count arms, and the W4A4 prompt class
/// requantizes what moves - so a prompt sitting on a near-tie can land
/// either side of it (GB10 2026-11-15, 60 FLEURS clips in waves of 8: 59-60
/// picks agree in either encoder class, KL median ~1e-8, the worst items
/// 0.4-0.6 - a digits-vs-words number at the first token). The gate is that
/// a pick may differ only across a near-tie: in one lane the other's pick
/// is the runner-up within 2 logits, and the distributions of every other
/// item agree.
#[test]
fn transcribe_audio_wave_matches_single_admission() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/tf"))
        .find(|p| p.join("vllm-fleurs60.json").is_file())
    else {
        common::missing("no transcribe-oracle/tf fixture");
        return;
    };
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("vllm-fleurs60.json")).unwrap()).unwrap();
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let (head, tail) = (ids(&fx["head"]), ids(&fx["tail"]));
    assert_eq!(model.batch_enable_probe(8).expect("enable"), 8);
    let items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)> = fx["clips"]
        .as_array()
        .unwrap()
        .iter()
        .step_by(7)
        .take(8)
        .enumerate()
        .map(|(slot, c)| {
            let pcm = read_f32(&root.join(format!("{}.f32", c["name"].as_str().unwrap())));
            (
                slot,
                vec![
                    paddock_engine::service::MmChunk::Text(head.clone()),
                    paddock_engine::service::MmChunk::Audio {
                        samples: pcm,
                        mel: None,
                    },
                    paddock_engine::service::MmChunk::Text(tail.clone()),
                ],
            )
        })
        .collect();
    let t0 = std::time::Instant::now();
    let wave = model.forward_prefill_multimodal_batch(items.clone());
    let t_wave = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let single: Vec<(Vec<f32>, usize)> = items
        .iter()
        .map(|(slot, chunks)| {
            model
                .forward_prefill_multimodal(*slot, chunks)
                .expect("single admission")
        })
        .collect();
    let t_single = t1.elapsed().as_secs_f64();
    // compared as distributions: the raw logits of the two lanes can move
    // by a near-constant shift the softmax ignores
    let mut same = 0usize;
    let mut worst = 0f64;
    let mut moved = Vec::new();
    // `b`'s pick is `a`'s runner-up within 2 logits
    let tie = |a: &[f32], b: &[f32]| {
        let (pa, pb) = (argmax(a) as usize, argmax(b) as usize);
        let second = (0..a.len())
            .filter(|&i| i != pa)
            .max_by(|&x, &y| a[x].total_cmp(&a[y]))
            .unwrap();
        second == pb && a[pa] - a[pb] < 2.0
    };
    for (k, ((_, w), (l, n))) in wave.iter().zip(&single).enumerate() {
        let (lw, nw) = w.as_ref().expect("wave admission");
        assert_eq!(nw, n, "row counts differ");
        let agree = argmax(lw) == argmax(l);
        same += usize::from(agree);
        let (kl, margin) = kl_and_margin(l, lw);
        println!(
            "  item: {n} rows, KL(single||wave) {kl:.2e}, single top-1 margin {margin:.2}, wave \
             {:.2}",
            kl_and_margin(lw, l).1
        );
        if !agree && !(tie(l, lw) || tie(lw, l)) {
            moved.push(format!(
                "item {k}: picks {} / {} off a near-tie",
                argmax(l),
                argmax(lw)
            ));
        }
        if agree && margin >= 2.0 && kl_and_margin(lw, l).1 >= 2.0 {
            worst = worst.max(kl);
        }
    }
    println!(
        "wave of {}: {same} first picks agree, worst KL off the near-ties {worst:.2e}; wave \
         {t_wave:.2}s vs one at a time {t_single:.2}s",
        items.len()
    );
    assert!(
        moved.is_empty(),
        "a wave admission moved a first pick: {moved:?}"
    );
    assert!(
        worst < 0.1,
        "a wave admission moved a distribution (KL {worst:.3})"
    );
}

/// Repeat runs of one recording are the same bits: the speech encoder's
/// rows, and the serial prefill's last-row logits.
#[test]
fn transcribe_is_deterministic() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    let pcm = read_f32(&root.join("en10").join("pcm.f32"));
    let (a, _) = model.audio_probe(&pcm).expect("encode a");
    let (b, _) = model.audio_probe(&pcm).expect("encode b");
    let enc_gap = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let chunks = vec![
        paddock_engine::service::MmChunk::Text(ASR_HEAD.to_vec()),
        paddock_engine::service::MmChunk::Audio {
            samples: pcm,
            mel: None,
        },
        paddock_engine::service::MmChunk::Text(ASR_TAIL.to_vec()),
    ];
    let la = model.forward_multimodal(&chunks).unwrap().unwrap().0;
    let lb = model.forward_multimodal(&chunks).unwrap().unwrap().0;
    let lgap = la
        .iter()
        .zip(&lb)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    println!("encoder repeat gap {enc_gap:e}; serial prefill repeat gap {lgap:e}");
    // a text prompt's serial logits, for the lane comparison below
    let mut lt = Vec::new();
    model.reset();
    for &t in &CHAT {
        lt = model.forward(t).expect("text step");
    }
    // the batch lane: the same prompt into slot 0 twice, then slot 1
    assert!(model.batch_enable_probe(4).expect("enable") >= 2);
    let st = model.forward_prefill(2, &CHAT).expect("text slot prefill");
    let tt = argmax(&lt) as usize;
    println!(
        "text: serial top {tt} {} vs slot {} (max gap {:e})",
        lt[tt],
        st[tt],
        lt.iter()
            .zip(&st)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    );
    let s0 = model.forward_prefill_multimodal(0, &chunks).unwrap().0;
    let s0b = model.forward_prefill_multimodal(0, &chunks).unwrap().0;
    let s1 = model.forward_prefill_multimodal(1, &chunks).unwrap().0;
    let g = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let top = argmax(&la) as usize;
    println!(
        "slot repeat gap {:e}, slot0 vs slot1 {:e}, slot vs serial {:e} (top {top}: serial {} slot {})",
        g(&s0, &s0b),
        g(&s0, &s1),
        g(&s0, &la),
        la[top],
        s0[top]
    );
    assert_eq!(enc_gap, 0.0, "the speech encoder is not deterministic");
    assert_eq!(lgap, 0.0, "the serial prefill is not deterministic");
}

/// Recording prompts on the CHUNKED queue (mixed ticks carry their rows)
/// finish where a single admission lands them.
#[test]
fn transcribe_chunked_admission_matches_single() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/tf"))
        .find(|p| p.join("vllm-fleurs60.json").is_file())
    else {
        common::missing("no transcribe-oracle/tf fixture");
        return;
    };
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("vllm-fleurs60.json")).unwrap()).unwrap();
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let (head, tail) = (ids(&fx["head"]), ids(&fx["tail"]));
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    let items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)> = fx["clips"]
        .as_array()
        .unwrap()
        .iter()
        .step_by(13)
        .take(4)
        .enumerate()
        .map(|(slot, c)| {
            let pcm = read_f32(&root.join(format!("{}.f32", c["name"].as_str().unwrap())));
            (
                slot,
                vec![
                    paddock_engine::service::MmChunk::Text(head.clone()),
                    paddock_engine::service::MmChunk::Audio {
                        samples: pcm,
                        mel: None,
                    },
                    paddock_engine::service::MmChunk::Text(tail.clone()),
                ],
            )
        })
        .collect();
    assert!(model.supports_chunked_multimodal());
    let verdicts = model.prefill_begin_multimodal(items.clone());
    for (k, v) in &verdicts {
        assert!(
            matches!(v, paddock_engine::generator::MmAdmit::Queued),
            "slot {k}: not queued"
        );
    }
    let mut done: std::collections::HashMap<usize, (Vec<f32>, usize)> = Default::default();
    let mut ticks = 0;
    while done.len() < items.len() {
        let (_, fin) = model.forward_mixed(&[], 256).expect("mixed tick");
        for (slot, l, n) in fin {
            done.insert(slot, (l, n));
        }
        ticks += 1;
        assert!(ticks < 200, "the queue never drained");
    }
    let mut worst = 0f64;
    for (slot, chunks) in &items {
        let (l, n) = model
            .forward_prefill_multimodal(*slot, chunks)
            .expect("single admission");
        let (lc, nc) = &done[slot];
        assert_eq!(*nc, n, "row counts differ");
        let (kl, margin) = kl_and_margin(&l, lc);
        println!(
            "  slot {slot}: {n} rows, picks {} / {}, KL {kl:.2e}, margin {margin:.2}",
            argmax(&l),
            argmax(lc)
        );
        worst = worst.max(kl);
    }
    println!("chunked admission: {ticks} ticks at 256 rows, worst KL {worst:.2e}");
    assert!(
        worst < 0.1,
        "the chunked queue moved a distribution (KL {worst:.3})"
    );
}

/// The chunked queue over four FLEURS recordings at each budget, once with
/// `var` set and once without: every slot's last logits, state and conv
/// window must be bit-identical.
fn chunked_queue_ab(model: &mut GpuNemotron, var: &str, budgets: &[usize]) {
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/tf"))
        .find(|p| p.join("vllm-fleurs60.json").is_file())
    else {
        common::missing("no transcribe-oracle/tf fixture");
        return;
    };
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("vllm-fleurs60.json")).unwrap()).unwrap();
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect()
    };
    let (head, tail) = (ids(&fx["head"]), ids(&fx["tail"]));
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    let items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)> = fx["clips"]
        .as_array()
        .unwrap()
        .iter()
        .step_by(7)
        .take(4)
        .enumerate()
        .map(|(slot, c)| {
            let pcm = read_f32(&root.join(format!("{}.f32", c["name"].as_str().unwrap())));
            (
                slot,
                vec![
                    paddock_engine::service::MmChunk::Text(head.clone()),
                    paddock_engine::service::MmChunk::Audio {
                        samples: pcm,
                        mel: None,
                    },
                    paddock_engine::service::MmChunk::Text(tail.clone()),
                ],
            )
        })
        .collect();
    let mut run = |set: bool, budget: usize| {
        // SAFETY: the walk reads the switch per call; the test drives every
        // launch from this thread
        unsafe {
            if set {
                std::env::set_var(var, "1");
            } else {
                std::env::remove_var(var);
            }
        }
        model.prefill_begin_multimodal(items.clone());
        let mut done: std::collections::BTreeMap<usize, (Vec<f32>, usize)> = Default::default();
        let mut ticks = 0;
        while done.len() < items.len() {
            let (_, fin) = model.forward_mixed(&[], budget).expect("mixed tick");
            for (slot, l, n) in fin {
                done.insert(slot, (l, n));
            }
            ticks += 1;
            assert!(ticks < 400, "the queue never drained");
        }
        let states: Vec<_> = (0..items.len())
            .map(|s| model.state_dump_probe(s))
            .collect();
        (done, states, ticks)
    };
    let same = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
    for &budget in budgets {
        let (la, sa, ticks) = run(true, budget);
        let (lb, sb, _) = run(false, budget);
        for (slot, (l, n)) in &la {
            let (l2, n2) = &lb[slot];
            assert_eq!(n, n2, "slot {slot}: row counts differ");
            assert!(
                same(l, l2),
                "{var} budget {budget} slot {slot}: logits differ"
            );
        }
        for (s, (a, b)) in sa.iter().zip(&sb).enumerate() {
            for ((la, st_a, w_a), (lb, st_b, w_b)) in a.iter().zip(b) {
                assert_eq!(la, lb);
                assert!(
                    same(st_a, st_b),
                    "{var} budget {budget} slot {s} layer {la}: state differs"
                );
                assert!(
                    same(w_a, w_b),
                    "{var} budget {budget} slot {s} layer {la}: conv window differs"
                );
            }
        }
        let rows: Vec<usize> = la.values().map(|v| v.1).collect();
        println!(
            "{var} budget {budget}: {ticks} ticks, rows {rows:?}: logits, states and windows bit-identical"
        );
    }
    unsafe { std::env::remove_var(var) };
}

/// A prompt pass's mamba advance in two launches a layer (slots 954-955)
/// against the per-run walk, short runs on the serial conv and long ones
/// chunked.
#[test]
fn transcribe_prompt_runs_match_per_run_launches() {
    let Some(mut model) = setup() else {
        return;
    };
    if !model.prompt_runs_probe() {
        common::missing("pack lacks slots 954-955");
        return;
    }
    chunked_queue_ab(&mut model, "PADDOCK_NO_MAMBA_PROMPT_RUNS", &[96, 512]);
}

/// A verify round's mamba advance in two launches a layer (slots 947-948)
/// against the per-run launches it replaced: the same batched MTP rounds over
/// three recordings, once per path from fresh admissions - every pick, and
/// every slot's state and conv window after the rounds, bit-identical.
#[test]
fn transcribe_round_runs_match_per_run_launches() {
    // SAFETY: before any model loads, on the test's only thread
    unsafe {
        std::env::remove_var("PADDOCK_NO_SPEC");
        std::env::set_var("PADDOCK_SPEC", "on");
        // the verify walk's two advances (a fused round has only the table)
        std::env::set_var("PADDOCK_TR_NO_FUSED", "1");
    }
    let Some(mut model) = setup() else {
        return;
    };
    if !model.verify_runs_probe() {
        common::missing("pack lacks slots 947-948");
        return;
    }
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    let clips = ["en95", "en10", "edge"];
    let chunks: Vec<Vec<paddock_engine::service::MmChunk>> = clips
        .iter()
        .map(|c| {
            vec![
                paddock_engine::service::MmChunk::Text(ASR_HEAD.to_vec()),
                paddock_engine::service::MmChunk::Audio {
                    samples: read_f32(&root.join(c).join("pcm.f32")),
                    mel: None,
                },
                paddock_engine::service::MmChunk::Text(ASR_TAIL.to_vec()),
            ]
        })
        .collect();
    const ROUNDS: usize = 8;
    let mut run = |per_run: bool| {
        // SAFETY: the walk reads the switch per call; the test drives every
        // launch from this thread
        unsafe {
            if per_run {
                std::env::set_var("PADDOCK_NO_MAMBA_RUNS", "1");
            } else {
                std::env::remove_var("PADDOCK_NO_MAMBA_RUNS");
            }
        }
        let mut live: Vec<(usize, u32, usize)> = Vec::new();
        for (s, ch) in chunks.iter().enumerate() {
            let (l, n) = model.forward_prefill_multimodal(s, ch).expect("prefill");
            live.push((s, argmax(&l), n));
        }
        let mut picks_all = Vec::new();
        for round in 0..ROUNDS {
            let pend: Vec<(usize, u32)> = live.iter().map(|&(s, p, _)| (s, p)).collect();
            let drafts = model
                .spec_draft_batch(&pend, 3)
                .expect("draft")
                .expect("mtp drafter engaged");
            let reqs: Vec<(usize, usize, Vec<u32>)> = live
                .iter()
                .zip(&drafts)
                .map(|(&(s, p, pos), d)| {
                    let mut c = vec![p];
                    c.extend_from_slice(d);
                    (s, pos, c)
                })
                .collect();
            let picks = model
                .forward_spec_batch(&reqs)
                .expect("verify round")
                .unwrap_or_else(|| panic!("round {round} declined"));
            let mut off = 0usize;
            for (i, (_, _, c)) in reqs.iter().enumerate() {
                let mut a = 0usize;
                while a + 1 < c.len() && c[a + 1] == picks[off + a] {
                    a += 1;
                }
                live[i].1 = picks[off + a];
                live[i].2 += a + 1;
                off += c.len();
            }
            picks_all.push(picks);
        }
        let states: Vec<_> = (0..clips.len())
            .map(|s| model.state_dump_probe(s))
            .collect();
        (picks_all, states)
    };
    let (pa, sa) = run(true);
    let (pb, sb) = run(false);
    unsafe {
        std::env::remove_var("PADDOCK_NO_MAMBA_RUNS");
        std::env::remove_var("PADDOCK_TR_NO_FUSED");
    }
    assert_eq!(
        pa, pb,
        "round picks differ between the per-run and run-table advances"
    );
    for (s, (a, b)) in sa.iter().zip(&sb).enumerate() {
        for ((la, st_a, w_a), (lb, st_b, w_b)) in a.iter().zip(b) {
            assert_eq!(la, lb);
            let same =
                |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
            assert!(same(st_a, st_b), "slot {s} layer {la}: state differs");
            assert!(same(w_a, w_b), "slot {s} layer {la}: conv window differs");
        }
    }
    println!(
        "{ROUNDS} rounds x {} slots: picks, states and windows bit-identical across the two advances",
        clips.len()
    );
}

/// The bf16 head's tile class at every call width: a 128-row call (its column
/// tiles on x so a row tile's weight boxes stream once), four 32-row calls and
/// one-row calls (the vocab arm's one-row admission) - every logit
/// bit-identical, the property that lets a one-row tick, a prompt tail and a
/// verify round's rows score alike.
#[test]
fn transcribe_head_tile_rows_bit_identical() {
    let Some(model) = setup() else {
        return;
    };
    let hidden = 2688usize;
    let rows = 128usize;
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let x: Vec<f32> = (0..rows * hidden)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
        })
        .collect();
    let wide = model.head_tile_probe(&x, rows).expect("bf16 head");
    let vocab = wide.len() / rows;
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
    for q in 0..4 {
        let part = model
            .head_tile_probe(&x[q * 32 * hidden..(q + 1) * 32 * hidden], 32)
            .expect("32 rows");
        assert!(
            same(&part, &wide[q * 32 * vocab..(q + 1) * 32 * vocab]),
            "rows {}..{}: a 32-row call differs from the 128-row call",
            q * 32,
            (q + 1) * 32
        );
    }
    for r in [0usize, 31, 32, 77, 127] {
        let one = model
            .head_tile_probe(&x[r * hidden..(r + 1) * hidden], 1)
            .expect("one row");
        assert!(
            same(&one, &wide[r * vocab..(r + 1) * vocab]),
            "row {r}: the one-row tile differs from the 128-row call"
        );
    }
    println!("bf16 head tile: 128 / 32 / 1-row calls bit-identical over {vocab} logits");
}

/// The speaker branch on the device (one diarizer window: pre-encode, the
/// predict and the activity patterns of slot 949, no readback) against the
/// host driver it replaced for those recordings: probabilities, patterns and
/// the projected rows bit-identical.
#[test]
fn transcribe_device_speaker_branch_matches_the_host_driver() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    for clip in ["en10", "edge", "en95", "mix200"] {
        let pcm = read_f32(&root.join(clip).join("pcm.f32"));
        // SAFETY: the encode reads the switch per call; one thread drives it
        unsafe { std::env::set_var("PADDOCK_TR_HOST_DIAR", "1") };
        let (rh, sh) = model.audio_probe(&pcm).expect("host driver");
        unsafe { std::env::remove_var("PADDOCK_TR_HOST_DIAR") };
        let (rd, sd) = model.audio_probe(&pcm).expect("device branch");
        let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let flat = |p: &[[f32; 8]]| p.iter().flatten().copied().collect::<Vec<f32>>();
        assert_eq!(sh.codes, sd.codes, "{clip}: activity patterns differ");
        assert!(
            bits(&flat(&sh.probs), &flat(&sd.probs)),
            "{clip}: probabilities differ"
        );
        assert!(bits(&rh, &rd), "{clip}: projected rows differ");
        println!(
            "{clip}: {} rows, device speaker branch == host driver",
            sd.codes.len()
        );
    }
}

/// The speaker branch of several one-window recordings packed into shared
/// diarizer passes (slot 950) against each recording's own diarizer window
/// and against the host driver - the speech branch packed the same way in all
/// three (one encode of the same recordings), so the projected rows differ
/// only if an activity pattern does: bit-identical three ways.
#[test]
fn transcribe_packed_speaker_branch_matches_one_window() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    let long = read_f32(&root.join("en95").join("pcm.f32"));
    let mut pcms = vec![
        read_f32(&root.join("en10").join("pcm.f32")),
        read_f32(&root.join("edge").join("pcm.f32")),
    ];
    // FLEURS-length cuts of the long recording, odd lengths included
    for (a, b) in [
        (0usize, 160_000usize),
        (160_000, 331_337),
        (331_337, 410_000),
    ] {
        pcms.push(long[a..b.min(long.len())].to_vec());
    }
    let bits = |a: &[f32], b: &[f32]| {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    };
    let packed = model.audio_rows_probe(&pcms).expect("packed");
    // SAFETY: the encode reads the switches per call; one thread drives it
    unsafe { std::env::set_var("PADDOCK_TR_NO_DIAR_PACK", "1") };
    let alone = model.audio_rows_probe(&pcms).expect("one window each");
    unsafe {
        std::env::remove_var("PADDOCK_TR_NO_DIAR_PACK");
        std::env::set_var("PADDOCK_TR_HOST_DIAR", "1");
    }
    let host = model.audio_rows_probe(&pcms).expect("host driver");
    unsafe { std::env::remove_var("PADDOCK_TR_HOST_DIAR") };
    for i in 0..pcms.len() {
        assert!(
            bits(&packed[i], &alone[i]),
            "recording {i}: packed rows differ from alone"
        );
        assert!(
            bits(&packed[i], &host[i]),
            "recording {i}: packed rows differ from the host driver"
        );
    }
    println!(
        "{} recordings: packed speaker branch == one window == host driver",
        pcms.len()
    );
}

/// The speech encoder is batch-invariant: a recording's rows from a packed
/// wave are the bits it gets encoded alone. The W8A8 class makes this load-
/// bearing - a static-scale quantizer turns any difference into a different
/// e4m3 code somewhere and 32 blocks carry it to the class's own noise -
/// so a wave admission must not see a different encoder than a lone one.
#[test]
fn transcribe_encoder_rows_batch_invariant() {
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/tf"))
        .find(|p| p.join("vllm-fleurs60.json").is_file())
    else {
        common::missing("no transcribe-oracle/tf fixture");
        return;
    };
    let fx: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("vllm-fleurs60.json")).unwrap()).unwrap();
    let pcms: Vec<Vec<f32>> = fx["clips"]
        .as_array()
        .unwrap()
        .iter()
        .step_by(7)
        .take(8)
        .map(|c| read_f32(&root.join(format!("{}.f32", c["name"].as_str().unwrap()))))
        .collect();
    let packed = model.audio_rows_probe(&pcms).expect("packed");
    let mut moved = Vec::new();
    for (i, pcm) in pcms.iter().enumerate() {
        let alone = model
            .audio_rows_probe(std::slice::from_ref(pcm))
            .expect("alone");
        let same = alone[0].len() == packed[i].len()
            && alone[0]
                .iter()
                .zip(&packed[i])
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            let d = alone[0]
                .iter()
                .zip(&packed[i])
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            moved.push(format!("recording {i}: max |d| {d:.3e}"));
        }
    }
    println!(
        "{} recordings, {} moved by packing: {moved:?}",
        pcms.len(),
        moved.len()
    );
    assert!(
        moved.is_empty(),
        "packed rows differ from lone rows: {moved:?}"
    );
}

/// A fused round (fused.rs) with a queued recording's chunk riding it
/// against the same round alone and the chunk as a span of its own after it:
/// the picks of every round up to and including the fused one, the riding
/// recording's first-token logits and its mamba state bit-identical. The
/// decoding slots' states are reported, not gated: a round of 8 rows alone
/// and one of 168 beside a chunk take some prompt-class lanes in other
/// row-count arms (GB10 2026-11-14: ~1e-3 of the plane's range from
/// mid-depth on), while each round's picks, kept rows and replay come from
/// its own walk - so the rounds AFTER the fused one decode from states that
/// differ that much, and their picks are reported too (they part at the
/// first near-tie: with the W8A8 encoder's rows, 2026-11-15, one slot parts
/// in the second round after).
#[test]
fn transcribe_fused_round_matches_round_then_span() {
    // SAFETY: before any model loads, on the test's only thread
    unsafe {
        std::env::remove_var("PADDOCK_NO_SPEC");
        std::env::set_var("PADDOCK_SPEC", "on");
    }
    let Some(mut model) = setup() else {
        return;
    };
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/audio"))
        .find(|p| p.is_dir())
    else {
        common::missing("no transcribe-oracle/audio fixtures");
        return;
    };
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    let chunks = |c: &str| {
        vec![
            paddock_engine::service::MmChunk::Text(ASR_HEAD.to_vec()),
            paddock_engine::service::MmChunk::Audio {
                samples: read_f32(&root.join(c).join("pcm.f32")),
                mel: None,
            },
            paddock_engine::service::MmChunk::Text(ASR_TAIL.to_vec()),
        ]
    };
    let greedy = |n: usize| vec![paddock_engine::sampler::DevicePlan::Greedy; n];
    let mut run = |fused: bool| {
        // two decoding slots a few rounds in, a third recording queued
        let mut live: Vec<(usize, u32, usize)> = Vec::new();
        for (s, c) in ["en10", "edge"].iter().enumerate() {
            let (l, n) = model
                .forward_prefill_multimodal(s, &chunks(c))
                .expect("prefill");
            live.push((s, argmax(&l), n));
        }
        let mut picks_all = Vec::new();
        let round = |model: &mut paddock_engine::gpu_model::nemotron::GpuNemotron,
                     live: &mut Vec<(usize, u32, usize)>,
                     with_chunk: bool| {
            let pend: Vec<(usize, u32)> = live.iter().map(|&(s, p, _)| (s, p)).collect();
            let drafts = model
                .spec_draft_batch(&pend, 3)
                .expect("draft")
                .expect("mtp drafter engaged");
            let reqs: Vec<(usize, usize, Vec<u32>)> = live
                .iter()
                .zip(&drafts)
                .map(|(&(s, p, pos), d)| {
                    let mut c = vec![p];
                    c.extend_from_slice(d);
                    (s, pos, c)
                })
                .collect();
            let total: usize = reqs.iter().map(|r| r.2.len()).sum();
            let (picks, fin) = if with_chunk {
                let (p, f) = model
                    .forward_mixed_spec_plans(&reqs, 4096, &greedy(total), &[])
                    .expect("fused round");
                (p.expect("engaged"), f)
            } else {
                (
                    model
                        .forward_spec_batch_plans(&reqs, &greedy(total))
                        .expect("round")
                        .expect("engaged"),
                    Vec::new(),
                )
            };
            let mut off = 0usize;
            for (i, (_, _, c)) in reqs.iter().enumerate() {
                let mut a = 0usize;
                while a + 1 < c.len() && c[a + 1] == picks[off + a] {
                    a += 1;
                }
                live[i].1 = picks[off + a];
                live[i].2 += a + 1;
                off += c.len();
            }
            (picks, fin)
        };
        for _ in 0..3 {
            picks_all.push(round(&mut model, &mut live, false).0);
        }
        let v = model.prefill_begin_multimodal(vec![(2, chunks("en10"))]);
        assert!(matches!(v[0].1, paddock_engine::generator::MmAdmit::Queued));
        // the round below takes the prompt only once its encode has landed
        model.settle_encodes_probe().expect("encode");
        let first: Vec<f32> = if fused {
            let (p, fin) = round(&mut model, &mut live, true);
            picks_all.push(p);
            match fin
                .into_iter()
                .find(|f| f.0 == 2)
                .expect("the chunk finished")
                .1
            {
                paddock_engine::generator::FinishSample::Logits(l) => l,
                _ => panic!("host-sampled finisher expected"),
            }
        } else {
            picks_all.push(round(&mut model, &mut live, false).0);
            let (_, fin) = model.forward_mixed(&[], 4096).expect("span");
            fin.into_iter()
                .find(|f| f.0 == 2)
                .expect("the chunk finished")
                .1
        };
        for _ in 0..2 {
            picks_all.push(round(&mut model, &mut live, false).0);
        }
        let states: Vec<_> = (0..3).map(|s| model.state_dump_probe(s)).collect();
        (picks_all, first, states)
    };
    let (pa, la, sa) = run(true);
    let (pb, lb, sb) = run(false);
    // three rounds before the riding chunk and the fused round itself
    assert_eq!(
        pa[..4],
        pb[..4],
        "round picks differ between fused and round-then-span"
    );
    let after = pa[4..].iter().zip(&pb[4..]).filter(|(a, b)| a == b).count();
    println!(
        "rounds after the fused one: {after}/{} pick lists agree",
        pa.len() - 4
    );
    let bits = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        bits(&la, &lb),
        "the riding recording's first-token logits differ"
    );
    let mut worst = 0f32;
    let mut differ = 0usize;
    for (s, (a, b)) in sa.iter().zip(&sb).enumerate() {
        let (mut d, mut abs, mut mag, mut first) = (0usize, 0f32, 0f32, None);
        for ((li, st_a, w_a), (_, st_b, w_b)) in a.iter().zip(b) {
            for (x, y) in st_a.iter().chain(w_a).zip(st_b.iter().chain(w_b)) {
                mag = mag.max(x.abs());
                if x.to_bits() != y.to_bits() {
                    d += 1;
                    first.get_or_insert(*li);
                    abs = abs.max((x - y).abs());
                    worst = worst.max((x - y).abs() / x.abs().max(y.abs()).max(1e-3));
                }
            }
        }
        differ += d;
        println!(
            "  slot {s}: {d} elements differ from layer {first:?}, worst abs {abs:e} (plane max {mag:e})"
        );
        if s == 2 {
            assert_eq!(d, 0, "the riding recording's state differs");
        }
    }
    println!(
        "fused round == round then span: picks, first-token logits; states {differ} elements \
         differ, worst relative {worst:e}"
    );
}
