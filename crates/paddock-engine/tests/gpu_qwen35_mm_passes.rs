//! Picture prompts longer than one planned prefill chunk run in planned
//! passes (`prefix_cache::mm_pass_ends`) instead of one pass as long as the
//! prompt, which regrew the serving scratch to the prompt's length and kept it.
//! The slot path must still produce the single path's greedy stream - the
//! single path prefills the whole prompt in one pass, so it is the oracle for
//! the pass cuts.
//!
//! The chunk is forced small (PADDOCK_CHUNK_TICK_ROWS=512, read once per
//! process - hence its own test binary) so a ~2.1K-row prompt takes several
//! passes, and the first picture is placed so a 512-row boundary lands inside
//! it and has to walk back to its start. A twelve-picture prompt then checks
//! that pictures are encoded pass by pass inside the picture store's budget.
//!
//! Very heavy (a Qwen3.6/3.8-27B Q8_0 + mmproj, ~30 GB residency):
//! PADDOCK_HEAVY_TESTS=1, --release. QWEN36_MM_DIR names the model dir.

mod common;

use paddock_engine::generator::{Generator, MmAdmit};
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

const PASS_ROWS: usize = 512;

fn setup() -> Option<(GpuQwen35, GgufTokenizer)> {
    if !common::heavy() {
        return None;
    }
    let dir = common::model_dir("QWEN36_MM_DIR", common::QWEN36_27B_DIR)?;
    let model_path = std::fs::read_dir(&dir).ok().and_then(|rd| {
        rd.filter_map(|e| e.ok().map(|e| e.path())).find(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.ends_with("Q8_0.gguf") && !n.starts_with("mmproj")
        })
    });
    let mmproj_path = ["mmproj-F16.gguf", "mmproj-BF16.gguf"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists());
    let (Some(model_path), Some(mmproj_path)) = (model_path, mmproj_path) else {
        common::missing(&format!(
            "no Q8_0 backbone + mmproj pair in {}",
            dir.display()
        ));
        return None;
    };
    let exec = common::gpu_arc()?;
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load 27B");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    let mm = MappedGguf::open(&mmproj_path).expect("open mmproj");
    m.attach_vision(&mm).expect("attach vision");
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

fn top(v: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut i: Vec<usize> = (0..v.len()).collect();
    i.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    i.into_iter().take(k).map(|j| (j as u32, v[j])).collect()
}

fn amax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

#[test]
fn a_long_picture_prompt_in_planned_passes_matches_the_single_path() {
    // SAFETY: first thing in this test binary's only test, before the engine
    // (or any thread it starts) reads the environment. The other two keep the
    // Q8_0 FFN and projection planes resident: the one-pass oracle decodes on
    // the serial lane, which has no e4m3 arm, and the load drops those planes
    // otherwise. Both sides of the comparison run under the same settings.
    unsafe {
        std::env::set_var("PADDOCK_CHUNK_TICK_ROWS", PASS_ROWS.to_string());
        std::env::set_var("PADDOCK_QWEN35_F8_FFN_PF_MIN", "2");
        std::env::set_var("PADDOCK_QWEN35_W8_MIN", "64");
    }
    let Some((mut m, tok)) = setup() else { return };
    let vocab = m.vocab;
    let n_new = 12usize;

    let unit = "Ledger line: the depot at the harbour shipped crates of timber, wool \
                and salted fish north along the coast road before the first snow. ";
    let doc = tok.encode(&unit.repeat(80)).expect("enc");
    assert!(doc.len() > 1800, "doc too short: {} tokens", doc.len());
    // 1000 rows, then a 512x384 picture (16x12 = 192 rows at [1000, 1192)):
    // the 1024-row pass boundary lands inside it and must walk back to 1000
    let question = tok
        .encode(" Describe the colours of both pictures, then say what the ledger lines are about.")
        .expect("enc");
    // `skip` shifts the document by a few tokens: a second prompt with a
    // different first page, so the radix cannot resume it from the first
    let prompt = |skip: usize| {
        vec![
            MmChunk::Text(doc[skip..skip + 1000].to_vec()),
            quadrants(
                512,
                384,
                [[220, 30, 30], [30, 30, 220], [30, 180, 30], [240, 220, 40]],
            ),
            MmChunk::Text(doc[skip + 1000..skip + 1700].to_vec()),
            quadrants(
                384,
                512,
                [[250, 250, 250], [10, 10, 10], [250, 130, 0], [120, 0, 160]],
            ),
            MmChunk::Text(question.clone()),
        ]
    };
    let (chunks, chunks_b) = (prompt(0), prompt(7));

    // ---- single path: the whole prompt in one pass (the oracle), for both
    let (logits, rows_ref) = m.forward_multimodal_chunks(&chunks).expect("single mm");
    assert!(
        rows_ref > 4 * PASS_ROWS,
        "prompt is {rows_ref} rows - too short to take several {PASS_ROWS}-row passes"
    );
    let mut want = vec![amax(&logits)];
    for _ in 1..n_new {
        let l = m.forward_one(*want.last().unwrap()).expect("fwd");
        want.push(amax(&l));
    }
    eprintln!("single: {:?}", tok.decode(&want, false));
    let (logits_b, rows_b) = m.forward_multimodal_chunks(&chunks_b).expect("single mm b");

    // ---- slot path: planned passes
    m.enable_batch(2).expect("enable_batch");
    let budget = Generator::vision_budget(&m).expect("vision budget");
    assert_eq!(
        budget.max_tokens as usize, PASS_ROWS,
        "one picture must be capped at the planned pass"
    );
    let (l0, rows0) = m.forward_prefill_slot_mm(0, &chunks).expect("slot mm");
    assert_eq!(
        rows0, rows_ref,
        "the slot path prefilled a different prompt"
    );
    assert!(
        !m.prefill_scratch_fits(rows0),
        "the serving scratch was regrown to the whole {rows0}-row prompt"
    );
    let mut got = vec![amax(&l0)];
    for pos in rows0 as u32..(rows0 + n_new - 1) as u32 {
        let l = m
            .forward_batch(&[*got.last().unwrap()], &[pos])
            .expect("batch step");
        got.push(amax(&l[..vocab]));
    }
    eprintln!("passes: {:?}", tok.decode(&got, false));
    assert_eq!(
        got, want,
        "planned passes diverged from the one-pass prefill"
    );

    // ---- the batched wave routes an over-chunk prompt to the same passes: two
    // cold picture prompts at once (the long one is `chunks_b`, whose first
    // page differs, so the radix cannot resume it), the long one must answer
    // as its one-pass reference did
    let short = vec![
        MmChunk::Text(tok.encode("Picture: ").expect("enc")),
        quadrants(
            256,
            160,
            [[255, 0, 0], [0, 0, 255], [255, 0, 0], [0, 0, 255]],
        ),
        MmChunk::Text(tok.encode(" The colours are").expect("enc")),
    ];
    let mut wave = m.forward_prefill_multimodal_batch(vec![(0, chunks_b), (1, short)]);
    wave.sort_by_key(|(k, _)| *k);
    let (lw, rows_w) = wave.remove(0).1.expect("wave slot 0");
    assert_eq!(rows_w, rows_b);
    // The one-pass oracle decodes on the serial Q8_0 lane and the slot path on
    // the e4m3 serving lanes, so their logits sit ~0.1-0.4 apart - enough to
    // flip a near tie (measured: this prompt's reference leads by 0.33). The
    // exact check of the pass cuts is prompt A's stream above; here the long
    // prompt only has to answer sanely and, above all, not regrow the scratch.
    let ref_top3: Vec<u32> = top(&logits_b, 3).into_iter().map(|(t, _)| t).collect();
    assert!(
        ref_top3.contains(&amax(&lw)),
        "the batched wave's long prompt answered {} - not among the one-pass top 3 {ref_top3:?}",
        amax(&lw)
    );
    assert!(
        !m.prefill_scratch_fits(rows_w),
        "the batched wave regrew the serving scratch to the whole prompt"
    );
    let (held, budget) = m.picture_store_bytes();
    assert!(
        held <= budget,
        "picture store {held} B over its {budget} B budget"
    );

    // ---- pictures are encoded as their pass comes up: twelve distinct
    // pictures whose embeddings together outgrow the store's budget (two
    // planned passes of rows) prefill, and the store stays inside it - they
    // used to be encoded all up front and held twice until the prefill ended
    let mut many = Vec::new();
    for i in 0..12u8 {
        many.push(MmChunk::Text(
            doc[40 * i as usize..40 * i as usize + 40].to_vec(),
        ));
        many.push(quadrants(
            512,
            384,
            [
                [20 * i, 0, 200],
                [0, 200, 20 * i],
                [200, 20 * i, 0],
                [90, 90, 20 * i],
            ],
        ));
    }
    many.push(MmChunk::Text(question.clone()));
    let per_token = budget / (2 * PASS_ROWS as u64);
    let pictures_bytes = 12 * 192 * per_token;
    assert!(
        pictures_bytes > budget,
        "the prompt's pictures ({pictures_bytes} B) must outgrow the {budget} B budget"
    );
    let (lm, rows_m) = m.forward_prefill_slot_mm(1, &many).expect("many pictures");
    let (held, _) = m.picture_store_bytes();
    assert!(
        held <= budget,
        "{held} B held after a {pictures_bytes} B picture prompt - pictures were not \
         released pass by pass"
    );
    eprintln!(
        "many pictures: {rows_m} rows, top {:?}, store {held}/{budget} B",
        top(&lm, 3)
    );
    eprintln!("MM PASSES OK: {rows_ref} rows in {PASS_ROWS}-row passes == one pass");

    // ---- the stepped lane (`Generator::prefill_begin_multimodal` +
    // `encode_step`): the same planned passes, one unit a call - a pass's
    // pictures, then the pass - with ANOTHER slot decoding between the units,
    // as the scheduler's ticks do. It must land the blocking slot path's
    // logits bit for bit: the units share nothing across a call but the
    // slot's own KV / state and the job's pictures, so a decode tick that
    // reuses every scratch plane in between cannot reach them. The step
    // budget is capped by the 512-row planned pass here, so the two paths cut
    // the same passes. Each side starts from a fresh prefix cache.
    m.enable_batch(2).expect("fresh batch state");
    let (l_block, rows_block) = m.forward_prefill_slot_mm(1, &chunks).expect("blocking");
    m.enable_batch(2).expect("fresh batch state");
    assert!(
        Generator::supports_chunked_multimodal(&m),
        "qwen35 with vision should take the stepped picture lane"
    );
    let text = tok.encode(&unit.repeat(3)).expect("enc");
    let mut t0 = amax(&m.forward_prefill_slot(0, &text).expect("text slot"));
    let mut p0 = text.len() as u32;
    let verdicts = Generator::prefill_begin_multimodal(&mut m, vec![(1, chunks.clone())]);
    assert!(
        matches!(verdicts[..], [(1, MmAdmit::Encoding)]),
        "the picture prompt should be held for stepping"
    );
    let (mut units, mut stepped) = (0usize, None);
    while stepped.is_none() {
        assert!(
            Generator::encoding_pending(&m),
            "held job vanished unreported"
        );
        for (slot, v) in Generator::encode_step(&mut m) {
            assert_eq!(slot, 1);
            match v {
                MmAdmit::Prefilled { logits, rows } => stepped = Some((logits, rows)),
                MmAdmit::Failed(e) => panic!("stepped unit failed: {e}"),
                _ => panic!("unexpected verdict"),
            }
        }
        units += 1;
        assert!(units < 200, "the stepped lane never finished");
        // the other slot's decode tick between the units
        let l = m.forward_batch(&[t0], &[p0]).expect("slot 0 decode");
        t0 = amax(&l[..vocab]);
        p0 += 1;
    }
    let (l_step, rows_step) = stepped.expect("finished");
    assert_eq!(
        rows_step, rows_block,
        "stepped prefill covered different rows"
    );
    let same = l_step
        .iter()
        .zip(&l_block)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(
        same,
        "the stepped lane's logits differ from the blocking path's (units {units})"
    );
    assert!(
        !Generator::encoding_pending(&m),
        "a finished job stayed held"
    );
    eprintln!(
        "MM STEPS OK: {rows_step} rows in {units} units with slot 0 decoding between them == \
         the blocking pass, bit for bit"
    );
}
