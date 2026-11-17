//! The SentencePiece BPE encoder against the library itself: a fixture of
//! `encode_as_ids` / `piece_to_id` / `decode` over FLEURS transcripts (whole
//! and per word), Transcribe's speaker-tagged outputs and Unicode edge cases,
//! written by the reference library on the checkpoint's own proto
//! (`transcribe-oracle/ctc` under the models root).
#![allow(clippy::unwrap_used)]

use crate::spm::SpmBpe;

fn fixture() -> Option<(SpmBpe, serde_json::Value)> {
    let roots = [
        std::env::var("PADDOCK_MODELS").ok(),
        std::env::var("HOME")
            .ok()
            .map(|h| format!("{h}/paddock/models")),
    ];
    for r in roots.into_iter().flatten() {
        let dir = std::path::Path::new(&r).join("transcribe-oracle/ctc");
        if let (Ok(proto), Ok(cases)) = (
            std::fs::read(dir.join("ctc-spm.model")),
            std::fs::read(dir.join("spm-cases.json")),
        ) {
            return Some((
                SpmBpe::from_proto(&proto).unwrap(),
                serde_json::from_slice(&cases).unwrap(),
            ));
        }
    }
    eprintln!("SKIP: no transcribe-oracle/ctc fixture");
    None
}

#[test]
fn spm_bpe_matches_sentencepiece() {
    let Some((sp, fx)) = fixture() else { return };
    assert_eq!(sp.piece_count(), 23862);
    let cases = fx["cases"].as_array().unwrap();
    let mut bad = Vec::new();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = sp.encode(text);
        if got != want {
            bad.push(format!("{text:?}: got {got:?} want {want:?}"));
        }
    }
    for b in bad.iter().take(8) {
        eprintln!("{b}");
    }
    assert!(
        bad.is_empty(),
        "{} of {} encodes differ",
        bad.len(),
        cases.len()
    );
    for (piece, id) in fx["piece_to_id"].as_object().unwrap() {
        assert_eq!(
            sp.piece_to_id(piece),
            id.as_u64().unwrap() as u32,
            "piece_to_id({piece:?})"
        );
    }
    for d in fx["decode"].as_array().unwrap() {
        let ids: Vec<u32> = d["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            sp.decode(&ids),
            d["text"].as_str().unwrap(),
            "decode({ids:?})"
        );
    }
    println!(
        "{} encodes, piece_to_id and decode match sentencepiece",
        cases.len()
    );
}
