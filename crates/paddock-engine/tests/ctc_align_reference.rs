//! The CTC timestamp aligner against NVIDIA's: NeMo Speech's aligner and
//! the vLLM plugin's public result, run by the transcribe CTC reference script
//! in F32 over fixed inputs (F64-oracle speaker probabilities, the NeMo head's
//! log probabilities). Fed the same inputs, the transcript must prepare to the
//! same columns and align to the same `ctc_timestamps` object.
#![allow(clippy::unwrap_used)]
#![cfg(feature = "cuda")]

mod common;

use paddock_engine::ctc_align;

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn ctc_align_matches_nemo() {
    let Some(root) = common::model_roots()
        .into_iter()
        .map(|r| r.join("transcribe-oracle/ctc"))
        .find(|p| p.join("ref").is_dir())
    else {
        common::missing("no transcribe-oracle/ctc/ref fixture");
        return;
    };
    let Some(ckpt) = common::model_roots()
        .into_iter()
        .map(|r| r.join("NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"))
        .find(|p| p.join("config.json").is_file())
    else {
        common::missing("no Transcribe checkpoint config");
        return;
    };
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ckpt.join("config.json")).unwrap()).unwrap();
    let params = ctc_align::AlignParams::from_config(&config["ctc_timestamps"]).unwrap();
    let sp = paddock_tokenizer::spm::SpmBpe::from_proto(
        &std::fs::read(root.join("ctc-spm.model")).unwrap(),
    )
    .unwrap();
    let blank = sp.piece_count() as u32;
    let mut clips: Vec<_> = std::fs::read_dir(root.join("ref"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    clips.sort();
    let mut bad = 0usize;
    for d in clips {
        let r: serde_json::Value =
            serde_json::from_slice(&std::fs::read(d.join("ref.json")).unwrap()).unwrap();
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let t_enc = r["t_enc"].as_u64().unwrap() as usize;
        let transcript = r["transcript"].as_str().unwrap();
        let prep = ctc_align::prepare(transcript, &sp, blank, 2 * t_enc).unwrap();
        let want_cols: Vec<u32> = serde_json::from_value(r["columns"].clone()).unwrap();
        assert_eq!(prep.columns, want_cols, "{name}: columns");
        let logp = read_f32(&d.join("logp.f32"));
        let spk: Vec<[f32; 8]> = read_f32(&d.join("spk80.f32")).as_chunks::<8>().0.to_vec();
        let labels = std::fs::read(d.join("labels10.u8")).unwrap();
        let got = ctc_align::align(
            &prep,
            &ctc_align::Inputs {
                logp: &logp,
                ctc_frames: 2 * t_enc,
                speaker80: &spk,
                labels10: &labels,
                duration: r["duration"].as_f64().unwrap(),
            },
            &params,
        )
        .unwrap();
        let want = &r["ctc_timestamps"];
        let (gw, ww) = (
            got["words"].as_array().unwrap(),
            want["words"].as_array().unwrap(),
        );
        let words_same = gw == ww;
        let rest_same = got["diarization"] == want["diarization"]
            && got["speaker_tag_to_diarization_speaker"]
                == want["speaker_tag_to_diarization_speaker"];
        println!(
            "{name}: {} words (ref {}), words {}, diarization+mapping {}",
            gw.len(),
            ww.len(),
            if words_same { "identical" } else { "DIFFER" },
            if rest_same { "identical" } else { "DIFFER" }
        );
        if !words_same {
            for (i, (a, b)) in gw.iter().zip(ww).enumerate() {
                if a != b {
                    println!("   first word diff at {i}: ours {a} ref {b}");
                    break;
                }
            }
        }
        if !rest_same {
            println!(
                "   mapping ours {} ref {}",
                got["speaker_tag_to_diarization_speaker"],
                want["speaker_tag_to_diarization_speaker"]
            );
        }
        bad += usize::from(!(words_same && rest_same));
    }
    assert_eq!(bad, 0, "{bad} clips differ from the reference");
}
