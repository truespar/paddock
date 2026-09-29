//! gemma4 picture prefills must not read keys they have not written.
//!
//! A picture's rows attend NON-CAUSALLY to its last row, so a cut inside one -
//! a prefill pass boundary, or an SWA sub-span boundary the ring cannot
//! absorb - leaves its first rows reading KV rows that hold whatever the slot
//! held before. That is checked directly: each prompt is prefilled twice at a
//! 4096-row pass, after two different "poison" prompts that leave different
//! KV in the rows past the boundary, and the two prefills must be BIT-
//! IDENTICAL. A cold prefill that reads only its own keys cannot tell the
//! poisons apart; one that reads stale rows does.
//!
//! Comparing against a one-pass prefill instead would not work: moving the
//! pass boundary changes GEMM row counts, and that noise compounds through
//! the layers (measured: ~1% hidden-state spread by the last layer on a text
//! row, enough to flip a near-tie answer) whether or not a picture is cut.
//!
//! `G4_MM_IMAGE_TOKENS` raises the picture cap (the gemma4 tower honours the
//! runner's `max_image_tokens`) so a picture longer than an SWA sub-span can
//! be checked the same way.
//!
//! Very heavy (gemma-4-26B-A4B Q8_0 + mmproj): PADDOCK_HEAVY_TESTS=1,
//! --release. GEMMA4_MM_DIR names the model dir.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::gemma4::GpuGemma4;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PASS_ROWS: usize = 4096;

fn load(image_tokens: Option<usize>) -> Option<(GpuGemma4, GgufTokenizer)> {
    if !common::heavy() {
        return None;
    }
    let dir = common::model_dir("GEMMA4_MM_DIR", &["gemma-4-26B-A4B-it-GGUF"])?;
    let pick = |want_mmproj: bool| {
        std::fs::read_dir(&dir).ok().and_then(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path())).find(|p| {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                n.ends_with(".gguf") && n.contains("mmproj") == want_mmproj
            })
        })
    };
    let (Some(model_path), Some(mmproj_path)) = (pick(false), pick(true)) else {
        common::missing(&format!("no backbone + mmproj pair in {}", dir.display()));
        return None;
    };
    let exec = common::gpu_arc()?;
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuGemma4::load(exec, &map, 8192).expect("load");
    let mm = MappedGguf::open(&mmproj_path).expect("open mmproj");
    m.attach_vision_with(&mm, image_tokens)
        .expect("attach vision");
    Generator::enable_batch(&mut m, 2).expect("enable_batch");
    let pass = m.set_prefill_pass_rows(PASS_ROWS).expect("narrow the pass");
    assert_eq!(
        pass, PASS_ROWS,
        "the pass floor moved above the test's boundary"
    );
    Some((m, tok))
}

/// Four solid quadrants, so the picture has content the model can name.
fn quadrants(w: usize, h: usize, colors: [[u8; 3]; 4]) -> MmChunk {
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let q = (usize::from(y >= h / 2) << 1) | usize::from(x >= w / 2);
            rgb.extend_from_slice(&colors[q]);
        }
    }
    MmChunk::Image { rgb, w, h }
}

fn amax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

#[test]
fn a_picture_prefill_reads_only_keys_it_wrote() {
    // SAFETY: first thing in this binary's only test, before the engine (or
    // any thread it starts) reads the environment. With the prefix cache on,
    // the second prefill of a prompt would resume from the first.
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    let image_tokens: Option<usize> = std::env::var("G4_MM_IMAGE_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok());
    let Some((mut m, tok)) = load(image_tokens) else {
        return;
    };
    let unit = "Ledger line: the depot at the harbour shipped crates of timber, wool \
                and salted fish north along the coast road before the first snow. ";
    let doc = tok.encode(&unit.repeat(260)).expect("enc");
    let other = tok
        .encode(&"Weather log: squalls from the west, then fog, then a calm night. ".repeat(420))
        .expect("enc");
    assert!(
        doc.len() > 5600 && other.len() > 5600,
        "documents too short"
    );
    let question = tok
        .encode(" What colours are in the picture, and what are the ledger lines about?")
        .expect("enc");
    let picture = quadrants(
        896,
        896,
        [[220, 30, 30], [30, 30, 220], [30, 180, 30], [240, 220, 40]],
    );
    let with_picture = |before: usize| {
        vec![
            MmChunk::Text(doc[..before].to_vec()),
            picture.clone(),
            MmChunk::Text(question.clone()),
        ]
    };
    // the picture's soft-token count, so the placements below are exact
    let (_, rows) =
        Generator::forward_prefill_multimodal(&mut m, 0, &with_picture(16)).expect("probe");
    let n_img = rows - 16 - 2 - question.len();
    eprintln!("one picture = {n_img} soft tokens");
    let straddle_at = PASS_ROWS - n_img / 2;
    let cases = [
        // (name, prompt, whether a boundary falls inside its picture)
        (
            "picture inside pass 1",
            with_picture(PASS_ROWS - n_img - 400),
            false,
        ),
        (
            "picture after the boundary",
            with_picture(PASS_ROWS + 100),
            false,
        ),
        (
            "picture straddling the boundary",
            with_picture(straddle_at),
            true,
        ),
    ];
    let poisons = [
        vec![MmChunk::Text(doc[..PASS_ROWS + n_img + 600].to_vec())],
        vec![MmChunk::Text(other[..PASS_ROWS + n_img + 600].to_vec())],
    ];
    let mut failures = Vec::new();
    for (name, prompt, cut) in &cases {
        let mut outs = Vec::new();
        for poison in &poisons {
            Generator::forward_prefill_multimodal(&mut m, 0, poison).expect("poison");
            let (logits, _) =
                Generator::forward_prefill_multimodal(&mut m, 0, prompt).expect("mm prefill");
            outs.push(logits);
        }
        let diff = outs[0]
            .iter()
            .zip(&outs[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{name}: max |logit| difference across poisons {diff:.3e}, top {} vs {}{}",
            amax(&outs[0]),
            amax(&outs[1]),
            if *cut {
                "  <- the boundary falls inside the picture"
            } else {
                ""
            }
        );
        if diff != 0.0 {
            failures.push(format!("{name}: {diff:.3e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "a picture prefill read keys it had not written (its result depends on \
         what the slot held before): {failures:?}"
    );
}
