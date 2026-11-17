//! Nemotron 3.5 Transcribe's word timestamps and speaker labels on the GPU:
//! the CTC timestamp head against NeMo's F32 head on the same states, and
//! the whole path - our encoder's capture, the head, the aligner - against
//! NeMo's aligner (exact) and vLLM's served `ctc_timestamps` (tolerance) on
//! the same transcripts.
//! Fixtures: the transcribe CTC reference script over the F64 oracle dumps.
//!
//! Heavy (21.5 GB of weights): `PADDOCK_HEAVY_TESTS=1`.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::ctc_align;
use paddock_engine::gpu_model::nemotron::GpuNemotron;
use serde_json::Value;

const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4";

struct Fixture {
    model: GpuNemotron,
    root: std::path::PathBuf,
    params: ctc_align::AlignParams,
}

fn setup() -> Option<Fixture> {
    if !common::heavy() {
        return None;
    }
    let root = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/ctc"))
        .find(|p| p.join("ref").is_dir() && p.join("oracle").is_dir())?;
    let exec = common::gpu_arc()?;
    if !exec.has_tr_ctc() {
        common::missing("pack lacks the CTC timestamp head (slots 958-960)");
        return None;
    }
    let dir = common::model_dir("TRANSCRIBE_DIR", &[CKPT_DIR])?;
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let params = ctc_align::AlignParams::from_config(&config["ctc_timestamps"]).unwrap();
    let model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    Some(Fixture {
        model,
        root,
        params,
    })
}

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn clips(root: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(root.join("ref"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    v.sort();
    v
}

/// Word-level distance of two `ctc_timestamps` objects: (same words and
/// speakers, the largest start/end gap in seconds, the mean gap, the share
/// of edges more than 40 ms apart).
fn word_gap(a: &Value, b: &Value) -> (bool, f64, f64, f64) {
    let (wa, wb) = (
        a["words"].as_array().unwrap(),
        b["words"].as_array().unwrap(),
    );
    let same = wa.len() == wb.len()
        && wa
            .iter()
            .zip(wb)
            .all(|(x, y)| x["word"] == y["word"] && x["speaker"] == y["speaker"]);
    let mut worst = 0f64;
    let mut sum = 0f64;
    let (mut n, mut far) = (0usize, 0usize);
    for (x, y) in wa.iter().zip(wb) {
        for k in ["start", "end"] {
            let d = (x[k].as_f64().unwrap() - y[k].as_f64().unwrap()).abs();
            worst = worst.max(d);
            sum += d;
            n += 1;
            far += usize::from(d > 0.04 + 1e-9);
        }
    }
    let n = n.max(1) as f64;
    (same, worst, sum / n, far as f64 / n)
}

/// Diarization segments: (same speakers in the same order, the largest
/// start / end gap in seconds; infinite when the counts differ).
fn dia_gap(a: &Value, b: &Value) -> (bool, f64) {
    let (sa, sb) = (
        a["diarization"].as_array().unwrap(),
        b["diarization"].as_array().unwrap(),
    );
    if sa.len() != sb.len() {
        return (false, f64::INFINITY);
    }
    let same = sa.iter().zip(sb).all(|(x, y)| x["speaker"] == y["speaker"]);
    let mut worst = 0f64;
    for (x, y) in sa.iter().zip(sb) {
        for k in ["start", "end"] {
            worst = worst.max((x[k].as_f64().unwrap() - y[k].as_f64().unwrap()).abs());
        }
    }
    (same, worst)
}

#[test]
fn transcribe_ctc_head_and_timestamps_match_the_reference() {
    let Some(mut fx) = setup() else {
        return;
    };
    let vllm: Value =
        serde_json::from_slice(&std::fs::read(fx.root.join("vllm-ctc.json")).unwrap()).unwrap();
    let mut failures = Vec::new();
    for name in clips(&fx.root) {
        let r: Value = serde_json::from_slice(
            &std::fs::read(fx.root.join("ref").join(&name).join("ref.json")).unwrap(),
        )
        .unwrap();
        let t_enc = r["t_enc"].as_u64().unwrap() as usize;
        let cols: Vec<u32> = serde_json::from_value(r["columns"].clone()).unwrap();
        let o = fx.root.join("oracle").join(&name);
        let states = read_f32(&o.join("states.f32"));
        let states = &states[..t_enc * 1280];

        // 1. the head on the oracle's states vs NeMo's F32 head
        let ours = fx.model.ctc_columns_probe(states, &cols).unwrap();
        let want = read_f32(&fx.root.join("ref").join(&name).join("logp.f32"));
        assert_eq!(ours.len(), want.len(), "{name}: logp shape");
        let max_abs = ours
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        // 2. our head's columns through the aligner on the reference's own
        // speaker inputs: the reference's object
        let spm = paddock_tokenizer::spm::SpmBpe::from_proto(
            &std::fs::read(fx.root.join("ctc-spm.model")).unwrap(),
        )
        .unwrap();
        let transcript = r["transcript"].as_str().unwrap();
        let prep =
            ctc_align::prepare(transcript, &spm, spm.piece_count() as u32, 2 * t_enc).unwrap();
        let spk: Vec<[f32; 8]> = read_f32(&fx.root.join("ref").join(&name).join("spk80.f32"))
            .as_chunks::<8>()
            .0
            .to_vec();
        let labels = std::fs::read(fx.root.join("ref").join(&name).join("labels10.u8")).unwrap();
        let aligned = ctc_align::align(
            &prep,
            &ctc_align::Inputs {
                logp: &ours,
                ctc_frames: 2 * t_enc,
                speaker80: &spk,
                labels10: &labels,
                duration: r["duration"].as_f64().unwrap(),
            },
            &fx.params,
        )
        .unwrap();
        let nemo = &r["ctc_timestamps"];
        let head_exact = aligned == *nemo;

        // 3. the whole path from the recording (our encoder's states and
        // speaker probabilities), the served transcript
        let pcm = read_f32(&o.join("pcm.f32"));
        let served = &vllm[format!("{name}.wav")];
        let text = served["choices"][0]["message"]["content"].as_str().unwrap();
        let e2e = fx.model.ctc_timestamps_probe(&pcm, text).unwrap();
        let e2e_exact = e2e == *nemo;
        let (same_n, worst_n, mean_n, far_n) = word_gap(&e2e, nemo);
        let dia_n = e2e["diarization"] == nemo["diarization"];
        let (dspk_n, dgap_n) = dia_gap(&e2e, nemo);
        let (dspk_v, dgap_v) = dia_gap(&e2e, &served["ctc_timestamps"]);
        println!(
            "  {name} diarization: vs NeMo same speakers {dspk_n}, edges max {dgap_n:.3}s; vs vLLM \
             same speakers {dspk_v}, edges max {dgap_v:.3}s"
        );
        let (same_v, worst_v, mean_v, far_v) = word_gap(&e2e, &served["ctc_timestamps"]);
        println!(
            "{name}: head max|d| {max_abs:.2e}, head->aligner {}, end to end vs NeMo {} (words+\
             speakers {}, edges max {worst_n:.3}s mean {mean_n:.4}s past 40 ms {:.1}%, \
             diarization {}); vs vLLM words+speakers {} edges max {worst_v:.3}s mean \
             {mean_v:.4}s past 40 ms {:.1}%",
            if head_exact { "identical" } else { "DIFFERS" },
            if e2e_exact { "identical" } else { "differs" },
            if same_n { "same" } else { "DIFFER" },
            100.0 * far_n,
            if dia_n { "same" } else { "differs" },
            if same_v { "same" } else { "DIFFER" },
            100.0 * far_v,
        );
        // f32 on both sides, only the accumulation order differs (measured
        // <= 9.9e-4 over the 9 clips)
        if max_abs > 2e-3 {
            failures.push(format!("{name}: head max |d| {max_abs}"));
        }
        if !head_exact {
            failures.push(format!(
                "{name}: our head through the aligner differs from NeMo's"
            ));
        }
        // the f16 class: our encoder's states and speaker probabilities sit
        // close enough to the F64 oracle that the whole object is NeMo's, bit
        // for bit. The W8A8 class (the checkpoint's, served): its states sit
        // at that class's own noise from the oracle's (the speech gate), and a
        // recording inside one window has its speaker branch in bf16, so
        // NeMo's object on the oracle's states and probabilities is a
        // reference by tolerance - the same words and speakers, the word
        // edges within 10 ms on average, the diarization segments the same
        // speakers within one 10 ms frame
        let w8a8 = fx.model.speech_w8a8_probe();
        if !w8a8 && !e2e_exact {
            failures.push(format!("{name}: end to end {e2e} vs NeMo {nemo}"));
        }
        if w8a8 && (!same_n || !dspk_n || dgap_n > 0.0105 || mean_n > 0.01 || far_n > 0.1) {
            failures.push(format!("{name}: end to end off NeMo's words"));
        }
        // vLLM's own perception encoder (bf16 / fp8) moves some word edges:
        // the words and speakers hold, the edges on average within 10 ms
        // (NeMo's head in bf16 on the oracle states lands on the f32 object,
        // so the spread is vLLM's encoder, not the head's precision; the
        // German clips carry most of it, inside the 10% allowed past 40 ms)
        if !same_v || mean_v > 0.01 || far_v > 0.1 {
            failures.push(format!("{name}: off vLLM's words"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
