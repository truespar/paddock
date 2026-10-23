//! Nemotron 3.5 Parse against its F64 oracle - the release's own
//! Transformers code run in float64 (the repo's nemotron-parse oracle script
//! writes the fixtures), llama.cpp having no runner for this graph.
//!
//! The tower legs (this file's first gate):
//!   - pixels: the processor's resize + pad reproduced byte for byte (the
//!     resized content equals the oracle's), and the device patch stem's f16
//!     rows equal the oracle's f32 patches rounded to nearest;
//!   - the position field: PyTorch's CUDA antialias + grid_sample in F32,
//!     reproduced on the host - within F32 rounding of the oracle's;
//!   - the decoder prefix: the whole tower + neck against F64. The bar is the
//!     vendor's own BF16 serving class (`prefix_bf16.f32` when the fixture
//!     carries it): an f16-operand tower with f32 residual must land CLOSER to
//!     F64 than BF16 does, on every page.
//!
//! Fixtures: `<models>/nemotron-parse-oracle/` (or NEMOTRON_PARSE_REFERENCE_DIR),
//! checkpoint `<models>/Nemotron-3.5-Parse/` (or NEMOTRON_PARSE_DIR).

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::nemotron_parse::{RadioTower, prep_page};
use paddock_models::nemotron_parse::NemotronParseConfig;
use paddock_models::safetensors::ShardedSafetensors;

fn say(msg: &str) {
    eprintln!("nemotron-parse: {msg}");
}

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// rel L2 of `got` against `want` (both widened to f64)
fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w as f64).powi(2);
        den += (w as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

fn max_abs(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w as f64).abs())
        .fold(0.0, f64::max)
}

struct Fixture {
    dir: PathBuf,
    name: String,
    meta: serde_json::Value,
}

fn fixtures() -> Option<(PathBuf, Vec<Fixture>)> {
    let fix = std::env::var_os("NEMOTRON_PARSE_REFERENCE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            common::model_roots()
                .into_iter()
                .map(|r| r.join("nemotron-parse-oracle"))
                .find(|p| p.join("manifest.json").is_file())
        });
    let Some(fix) = fix else {
        common::missing("nemotron-parse oracle fixtures (nemotron-parse-oracle/)");
        return None;
    };
    let Some(model) = common::model_dir("NEMOTRON_PARSE_DIR", &["Nemotron-3.5-Parse"]) else {
        common::missing("the Nemotron-3.5-Parse checkpoint");
        return None;
    };
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fix.join("manifest.json")).expect("manifest"))
            .expect("parse manifest");
    let pages = man["pages"]
        .as_array()
        .expect("manifest.pages")
        .iter()
        .map(|p| {
            let name = p.as_str().expect("page name").to_owned();
            let dir = fix.join(&name);
            let meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("meta"))
                .expect("parse meta");
            Fixture { dir, name, meta }
        })
        .collect();
    Some((model, pages))
}

fn wh(v: &serde_json::Value) -> (usize, usize) {
    (
        v[0].as_u64().expect("w") as usize,
        v[1].as_u64().expect("h") as usize,
    )
}

/// The host pixel path alone (no GPU): resize + pad against the processor.
#[test]
fn pixels_match_the_processor() {
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let cfg = NemotronParseConfig::read(&model).expect("read the checkpoint contract");
    for f in &pages {
        let (w, h) = wh(&f.meta["input_wh"]);
        let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
        let page = prep_page(&rgb, w, h, &cfg);
        let want_content = (
            f.meta["content_size"][0].as_u64().unwrap() as usize,
            f.meta["content_size"][1].as_u64().unwrap() as usize,
        );
        let want_canvas = (
            f.meta["image_size"][0].as_u64().unwrap() as usize,
            f.meta["image_size"][1].as_u64().unwrap() as usize,
        );
        assert_eq!(page.content_hw, want_content, "{}: content size", f.name);
        assert_eq!(page.canvas_hw, want_canvas, "{}: canvas size", f.name);
        let resized = std::fs::read(f.dir.join("resized.rgb")).expect("resized.rgb");
        let (ch, cw) = page.content_hw;
        let ww = page.canvas_hw.1;
        let mut bad = 0usize;
        for y in 0..ch {
            let a = &page.canvas[y * ww * 3..y * ww * 3 + cw * 3];
            let b = &resized[y * cw * 3..(y + 1) * cw * 3];
            bad += a.iter().zip(b).filter(|(x, y)| x != y).count();
        }
        // padding is white
        let white = (0..page.canvas_hw.0)
            .flat_map(|y| (0..ww).map(move |x| (y, x)))
            .filter(|&(y, x)| y >= ch || x >= cw)
            .all(|(y, x)| page.canvas[(y * ww + x) * 3..(y * ww + x) * 3 + 3] == [255, 255, 255]);
        say(&format!(
            "{}: {}x{} -> content {:?} canvas {:?}, {bad} resized bytes differ",
            f.name, w, h, page.content_hw, page.canvas_hw
        ));
        assert_eq!(bad, 0, "{}: resize is not the processor's", f.name);
        assert!(white, "{}: padding is not white", f.name);
    }
}

#[test]
fn tower_matches_the_f64_oracle() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let cfg = NemotronParseConfig::read(&model).expect("read the checkpoint contract");
    let st = ShardedSafetensors::open_dir(&model).expect("open safetensors");
    let mut tower = RadioTower::load(exec, &st, &cfg, 14_000).expect("load the tower");
    let mut worst = 0f64;
    for f in &pages {
        let (w, h) = wh(&f.meta["input_wh"]);
        let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
        let page = prep_page(&rgb, w, h, &cfg);

        // position field
        let pe = tower.field_host(&page).expect("field");
        let want_pe = f32s(&f.dir.join("posemb.f32"));
        let pe_diff = pe.iter().zip(&want_pe).filter(|(a, b)| a != b).count();
        let pe_max = max_abs(&pe, &want_pe);

        // encode, then the stem's rows and the prefix
        let out = tower.encode(&[&page]).expect("encode");
        let px = tower.patches_host(&page).expect("patches");
        let want_px = f32s(&f.dir.join("pixels.f32"));
        let px_bad = px
            .iter()
            .zip(&want_px)
            .filter(|(a, b)| **a != half::f16::from_f32(**b))
            .count();
        let got = common_host(&tower, &out[0]);
        let want = f32s(&f.dir.join("prefix.f32"));
        assert_eq!(got.len(), want.len(), "{}: prefix rows", f.name);
        let e = rel_l2(&got, &want);
        let bf = f.dir.join("prefix_bf16.f32");
        let bar = bf.is_file().then(|| rel_l2(&f32s(&bf), &want));
        say(&format!(
            "{}: field {pe_diff} of {} floats differ (max {pe_max:.2e}); stem {px_bad} of {} \
             halves differ; prefix rel L2 {e:.3e} (max abs {:.3e}){}",
            f.name,
            pe.len(),
            px.len(),
            max_abs(&got, &want),
            bar.map_or(String::new(), |b| format!(", vendor BF16 {b:.3e}"))
        ));
        assert_eq!(px_bad, 0, "{}: patch stem", f.name);
        assert!(pe_max < 1e-5, "{}: position field off by {pe_max}", f.name);
        if let Some(b) = bar {
            assert!(
                e < b,
                "{}: prefix {e:.3e} not closer to F64 than BF16 ({b:.3e})",
                f.name
            );
        }
        worst = worst.max(e);
    }
    // A sanity ceiling, not the gate (the per-page BF16 bar above is):
    // structural mistakes land near 1. The f16 class moves page to page
    // with any rounding change - slot 893 reading the qkv landing directly
    // took two report pages 1.6-1.8e-2 -> 2.3-2.5e-2 while the 20-page mean
    // fell 1.14e-2 -> 1.09e-2 (the vendor's BF16 is 7.6e-2..1.5e-1).
    assert!(
        worst < 3e-2,
        "prefix rel L2 {worst:.3e} - structurally wrong"
    );
}

fn common_host(
    tower: &RadioTower,
    p: &paddock_engine::gpu_model::nemotron_parse::PrefixPlane,
) -> Vec<f32> {
    tower.prefix_host(p).expect("prefix to host")
}

/// x and y coordinate ids (the tokenizer's `<x_*>` / `<y_*>` runs).
fn coord(id: u32) -> Option<(char, i64)> {
    match id {
        50011..=51034 => Some(('x', (id - 50011) as i64)),
        51035..=52314 => Some(('y', (id - 51035) as i64)),
        _ => None,
    }
}

/// The decoder legs. Teacher-forced on the oracle's own ids, every position
/// whose F64 top-2 margin clears a near-tie floor must pick the oracle's id;
/// free-running, every non-coordinate token must equal F64's and a coordinate
/// may sit at most 2 bins off (the vendor's own BF16 run flips adjacent bins
/// on the first page: 4 of 554 tokens, all by one bin).
#[test]
fn decoder_matches_the_f64_oracle() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let mut m = paddock_engine::gpu_model::nemotron_parse::GpuNemotronParse::load_dir(
        exec,
        &model,
        16_384,
        paddock_engine::gpu::KvDtype::Fp16,
    )
    .expect("load");
    paddock_engine::generator::Generator::enable_batch(&mut m, 2).expect("enable");
    const FLOOR: f64 = 0.05;
    let (mut tf_bad, mut text_bad, mut far) = (0usize, 0usize, 0usize);
    for f in &pages {
        let (w, h) = wh(&f.meta["input_wh"]);
        let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
        let prompt: Vec<u32> = f.meta["prompt_ids"]
            .as_array()
            .expect("prompt_ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let want: Vec<u32> = f.meta["gen_ids"]
            .as_array()
            .expect("gen_ids")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        let margins: Vec<f64> = f.meta["margins"]
            .as_array()
            .expect("margins")
            .iter()
            .map(|v| v.as_f64().expect("margin"))
            .collect();
        let chunks = vec![
            paddock_engine::service::MmChunk::Image {
                rgb: rgb.clone(),
                w,
                h,
            },
            paddock_engine::service::MmChunk::Text(prompt.clone()),
        ];
        let t0 = std::time::Instant::now();
        let tf = m
            .decode_page_greedy(0, &chunks, want.len(), Some(&want))
            .expect("teacher-forced");
        let tf_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut tf_page = 0usize;
        let mut tf_tie = 0usize;
        for (i, (&a, &b)) in tf.iter().zip(&want).enumerate() {
            if a != b {
                if margins[i] > FLOOR {
                    tf_page += 1;
                    eprintln!(
                        "  {}: teacher-forced pos {i}: got {a} want {b} (F64 margin {:.4})",
                        f.name, margins[i]
                    );
                } else {
                    tf_tie += 1;
                }
            }
        }
        let t1 = std::time::Instant::now();
        let free = m
            .decode_page_greedy(0, &chunks, want.len() + 64, None)
            .expect("free-running");
        let free_ms = t1.elapsed().as_secs_f64() * 1e3;
        let (mut tb, mut near, mut fr) = (0usize, 0usize, 0usize);
        for (&a, &b) in free.iter().zip(&want) {
            if a == b {
                continue;
            }
            match (coord(a), coord(b)) {
                (Some((ax, av)), Some((bx, bv))) if ax == bx && (av - bv).abs() <= 2 => near += 1,
                (Some(_), Some(_)) => fr += 1,
                _ => tb += 1,
            }
        }
        say(&format!(
            "{}: teacher-forced {} of {} differ ({tf_tie} at F64 near-ties) [{tf_ms:.0} ms]; \
             free-running {} vs {} tokens: {tb} text differ, {near} coords within 2 bins, {fr} \
             coords farther [{free_ms:.0} ms]",
            f.name,
            tf_page + tf_tie,
            want.len(),
            free.len(),
            want.len()
        ));
        tf_bad += tf_page;
        text_bad += tb;
        far += fr;
    }
    assert_eq!(
        tf_bad, 0,
        "teacher-forced picks off F64 away from near-ties"
    );
    assert_eq!(text_bad, 0, "free-running text tokens off F64");
    assert_eq!(far, 0, "free-running coordinates more than 2 bins off F64");
}

/// MTP rounds: the emitted ids against F64 (same bar as greedy) and against
/// this engine's own plain greedy decode, with the acceptance and speed.
#[test]
fn mtp_spec_matches_the_f64_oracle() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let mut m = paddock_engine::gpu_model::nemotron_parse::GpuNemotronParse::load_dir(
        exec,
        &model,
        16_384,
        paddock_engine::gpu::KvDtype::Fp16,
    )
    .expect("load");
    paddock_engine::generator::Generator::enable_batch(&mut m, 2).expect("enable");
    let (mut text_bad, mut far, mut rounds, mut acc) = (0usize, 0usize, 0usize, 0usize);
    let mut same_as_plain = 0usize;
    for f in &pages {
        let (w, h) = wh(&f.meta["input_wh"]);
        let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
        let ids = |k: &str| -> Vec<u32> {
            f.meta[k]
                .as_array()
                .expect(k)
                .iter()
                .map(|v| v.as_u64().expect("id") as u32)
                .collect()
        };
        let (prompt, want) = (ids("prompt_ids"), ids("gen_ids"));
        let chunks = vec![
            paddock_engine::service::MmChunk::Image { rgb, w, h },
            paddock_engine::service::MmChunk::Text(prompt),
        ];
        let plain = m
            .decode_page_greedy(0, &chunks, want.len() + 64, None)
            .expect("plain");
        let t = std::time::Instant::now();
        let run = m
            .decode_page_spec(0, &chunks, want.len() + 64)
            .expect("spec");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (mut tb, mut near, mut fr) = (0usize, 0usize, 0usize);
        for (&a, &b) in run.tokens.iter().zip(&want) {
            if a == b {
                continue;
            }
            match (coord(a), coord(b)) {
                (Some((ax, av)), Some((bx, bv))) if ax == bx && (av - bv).abs() <= 2 => near += 1,
                (Some(_), Some(_)) => fr += 1,
                _ => tb += 1,
            }
        }
        let same = run.tokens == plain;
        same_as_plain += same as usize;
        say(&format!(
            "{}: spec {} tokens in {} rounds ({:.1}% accepted) [{ms:.0} ms]: {tb} text differ from \
             F64, {near} coords within 2 bins, {fr} farther; {} plain greedy",
            f.name,
            run.tokens.len(),
            run.rounds,
            100.0 * run.accepted as f64 / run.rounds.max(1) as f64,
            if same { "identical to" } else { "DIFFERS from" }
        ));
        text_bad += tb;
        far += fr;
        rounds += run.rounds;
        acc += run.accepted;
    }
    say(&format!(
        "MTP acceptance {:.2}% over {rounds} rounds; {same_as_plain}/{} pages identical to plain greedy",
        100.0 * acc as f64 / rounds.max(1) as f64,
        pages.len()
    ));
    assert_eq!(text_bad, 0, "spec text tokens off F64");
    assert_eq!(far, 0, "spec coordinates more than 2 bins off F64");
}

/// One tower pass of six page shapes: every page's prefix bit-identical to the
/// same page encoded alone (a pass packs pages of different sizes and must not
/// let them touch), and no position field evicted while its pass still needs it
/// (a pass of five shapes used to panic on the lookup).
#[test]
fn tower_packs_many_page_shapes_bit_identically() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let cfg = NemotronParseConfig::read(&model).expect("cfg");
    let st = ShardedSafetensors::open_dir(&model).expect("safetensors");
    let mut tower = RadioTower::load(
        exec,
        &st,
        &cfg,
        paddock_engine::gpu_model::nemotron_parse::TOWER_PASS_ROWS,
    )
    .expect("tower");
    let f = pages
        .iter()
        .find(|f| f.name == "page_000")
        .expect("page_000");
    let (w, h) = wh(&f.meta["input_wh"]);
    let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
    // six crops of the page, six shapes
    let crop = |cw: usize, ch: usize| {
        let mut out = Vec::with_capacity(cw * ch * 3);
        for y in 0..ch {
            out.extend_from_slice(&rgb[(y * w) * 3..(y * w + cw) * 3]);
        }
        prep_page(&out, cw, ch, &cfg)
    };
    let shapes = [
        (w, h / 3),
        (w - 64, h / 3),
        (w / 2, h / 2),
        (w / 2 + 40, h / 4),
        (w / 3, h / 3),
        (w, h / 5),
    ];
    let preps: Vec<_> = shapes.iter().map(|&(cw, ch)| crop(cw, ch)).collect();
    let alone: Vec<Vec<f32>> = preps
        .iter()
        .map(|p| {
            let planes = tower.encode(&[p]).expect("encode alone");
            tower.prefix_host(&planes[0]).expect("read")
        })
        .collect();
    let group: Vec<_> = preps.iter().collect();
    let planes = tower.encode(&group).expect("encode the pass");
    let mut moved = 0;
    for (i, (pl, want)) in planes.iter().zip(&alone).enumerate() {
        let got = tower.prefix_host(pl).expect("read");
        let off = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rows: Vec<usize> = (0..got.len() / 1024)
            .filter(|r| {
                (0..1024).any(|c| got[r * 1024 + c].to_bits() != want[r * 1024 + c].to_bits())
            })
            .collect();
        say(&format!(
            "page shape {:?} ({} rows): {off} of {} prefix floats differ packed vs alone, worst \
             {worst:.3e}, rows {:?}",
            shapes[i],
            got.len() / 1024,
            got.len(),
            &rows[..rows.len().min(12)]
        ));
        moved += off;
    }
    assert_eq!(moved, 0, "a page's prefix moved when packed");
}

/// FNV-1a over a logits row's bits: two rows hash alike only when every f32
/// is the same bit pattern (the gate compares bits, not values).
fn row_hash(row: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in row {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The greedy pick of a (penalized) logits row, lowest id on ties.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// One request of an invariance run: its slot, its chunks and the scheduler
/// tick it arrives at.
struct InvReq {
    slot: usize,
    chunks: Vec<paddock_engine::service::MmChunk>,
    at: usize,
}

/// Drive the family the way the service does - admission, the tower's
/// encode steps, mixed ticks with `budget` prefill rows - and return every
/// slot's emitted ids with the bit hash of the logits row that produced each.
/// `spec`: once a slot decodes, it runs MTP rounds alone (one slot at a time
/// keeps the round shapes the service's; the other slots stay parked).
fn drive_inv(
    m: &mut paddock_engine::gpu_model::nemotron_parse::GpuNemotronParse,
    reqs: &[InvReq],
    budget: usize,
    steps: usize,
    spec: bool,
) -> std::collections::HashMap<usize, Vec<(u32, u64)>> {
    use paddock_engine::generator::{Generator, MmAdmit};
    let vocab = Generator::vocab(m);
    let mut out: std::collections::HashMap<usize, Vec<(u32, u64)>> = Default::default();
    // per slot: (input token, its position) once decoding; None = done
    let mut live: std::collections::BTreeMap<usize, Option<(u32, u32)>> = Default::default();
    let mut waiting = reqs.len();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(900);
    // a tick = one scheduler iteration that did work; idle polls (the tower
    // pass still running) do not count, so arrivals land at the same point
    // of the stream run to run
    let mut tick = 0usize;
    let mut admitted = 0usize;
    while std::time::Instant::now() < deadline {
        while admitted < reqs.len() && reqs[admitted].at <= tick {
            let r = &reqs[admitted];
            for (_, a) in m.prefill_begin_multimodal(vec![(r.slot, r.chunks.clone())]) {
                assert!(!matches!(a, MmAdmit::Failed(_)), "admission failed");
            }
            admitted += 1;
        }
        for (_, a) in m.encode_step() {
            assert!(!matches!(a, MmAdmit::Failed(_)), "encode failed");
        }
        let emit = |out: &mut std::collections::HashMap<usize, Vec<(u32, u64)>>,
                    slot: usize,
                    tok: u32,
                    h: u64|
         -> bool {
            let v = out.entry(slot).or_default();
            v.push((tok, h));
            tok == 2 || v.len() >= steps
        };
        if spec {
            let ready: Vec<(usize, u32, u32)> = live
                .iter()
                .filter_map(|(&s, st)| st.map(|(t, p)| (s, t, p)))
                .collect();
            if let Some(&(slot, tok, pos)) = ready.first() {
                // one MTP round for the first decoding slot
                let d = m
                    .spec_draft_batch(&[(slot, tok)], 1)
                    .expect("draft")
                    .expect("spec ready")[0][0];
                let picks = m
                    .forward_spec_batch(&[(slot, pos as usize, vec![tok, d])])
                    .expect("verify")
                    .expect("spec ready");
                let rows = m.read_batch_logits(2).expect("logits");
                let mut done = emit(&mut out, slot, picks[0], row_hash(&rows[..vocab]));
                let mut next = (picks[0], pos + 1);
                if !done && d == picks[0] {
                    done = emit(&mut out, slot, picks[1], row_hash(&rows[vocab..]));
                    next = (picks[1], pos + 2);
                }
                live.insert(slot, (!done).then_some(next));
                tick += 1;
                continue;
            }
        }
        let decodes: Vec<(usize, u32, u32)> = if spec {
            Vec::new()
        } else {
            live.iter()
                .filter_map(|(&s, st)| st.map(|(t, p)| (s, t, p)))
                .collect()
        };
        let (logits, fin) = m.forward_mixed(&decodes, budget).expect("tick");
        if decodes.is_empty() && fin.is_empty() && m.prefill_queue().is_empty() {
            // nothing ran: the tower's pass is still in flight
            std::thread::sleep(std::time::Duration::from_micros(200));
        } else {
            tick += 1;
        }
        for (i, &(slot, _, pos)) in decodes.iter().enumerate() {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let pick = argmax(row);
            let done = emit(&mut out, slot, pick, row_hash(row));
            live.insert(slot, (!done).then_some((pick, pos + 1)));
        }
        for (slot, row, n_rows) in fin {
            let pick = argmax(&row);
            let done = emit(&mut out, slot, pick, row_hash(&row));
            live.insert(slot, (!done).then_some((pick, n_rows as u32)));
            waiting -= 1;
        }
        if waiting == 0 && live.values().all(|s| s.is_none()) {
            return out;
        }
    }
    panic!("invariance run did not finish");
}

/// G4: a page's tokens AND the bits of every logits row behind them are the
/// same alone, chunked differently beside five other pages (packed tower
/// passes, mixed ticks with other requests' prefill rows, decode bands of
/// six), in another slot, and under MTP rounds.
#[test]
fn batch_invariance() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let mut m = paddock_engine::gpu_model::nemotron_parse::GpuNemotronParse::load_dir(
        exec,
        &model,
        16_384,
        paddock_engine::gpu::KvDtype::Fp16,
    )
    .expect("load");
    paddock_engine::generator::Generator::enable_batch(&mut m, 16).expect("enable");
    let chunks = |name: &str| {
        let f = pages.iter().find(|f| f.name == name).expect(name);
        let (w, h) = wh(&f.meta["input_wh"]);
        let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
        let prompt: Vec<u32> = f.meta["prompt_ids"]
            .as_array()
            .expect("prompt")
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect();
        vec![
            paddock_engine::service::MmChunk::Image { rgb, w, h },
            paddock_engine::service::MmChunk::Text(prompt),
        ]
    };
    const STEPS: usize = 400;
    let a = "page_000";
    let solo = drive_inv(
        &mut m,
        &[InvReq {
            slot: 0,
            chunks: chunks(a),
            at: 0,
        }],
        1 << 20,
        STEPS,
        false,
    );
    let want = &solo[&0];
    // beside five pages: two arrive with it (packed tower passes), three
    // later (their prefill rows ride the mixed ticks of its decode band);
    // 509-row chunks cut every prompt at other rows than the solo run's
    let others = ["page_004", "page_008", "page_012", "page_016", "nordvik-2"];
    let mut reqs = vec![InvReq {
        slot: 5,
        chunks: chunks(a),
        at: 0,
    }];
    for (i, o) in others.iter().enumerate() {
        reqs.push(InvReq {
            slot: [0, 1, 2, 3, 4][i],
            chunks: chunks(o),
            at: if i < 2 { 0 } else { 30 + 15 * i },
        });
    }
    let busy = drive_inv(&mut m, &reqs, 509, STEPS, false);
    let spec = drive_inv(
        &mut m,
        &[InvReq {
            slot: 9,
            chunks: chunks(a),
            at: 0,
        }],
        777,
        STEPS,
        true,
    );
    // 7-row chunks: every prefill GEMM in the tiny-M regime (where the
    // elected GEMM would take its GEMV band), beside another page
    let dribble = drive_inv(
        &mut m,
        &[
            InvReq {
                slot: 11,
                chunks: chunks("page_020"),
                at: 0,
            },
            InvReq {
                slot: 7,
                chunks: chunks(a),
                at: 0,
            },
        ],
        7,
        STEPS,
        false,
    );
    let first_off = |got: &[(u32, u64)]| {
        got.iter()
            .zip(want)
            .position(|(g, w)| g != w)
            .map_or("none".to_owned(), |i| format!("step {i}"))
    };
    say(&format!(
        "{a}: {} steps solo; first difference: busy (6 pages, 509-row chunks, slot 5) {}, MTP \
         rounds (slot 9, 777-row chunks) {}, 7-row chunks beside another page (slot 7) {}",
        want.len(),
        first_off(&busy[&5]),
        first_off(&spec[&9]),
        first_off(&dribble[&7]),
    ));
    assert_eq!(&busy[&5], want, "a page's bits moved with its batch");
    assert_eq!(&spec[&9], want, "a page's bits moved under MTP rounds");
    assert_eq!(
        &dribble[&7], want,
        "a page's bits moved with its prompt's chunking"
    );
}

/// Phase timings on this die (ignored: a bench, not a gate). Tower passes at
/// one and two A4 pages, a page's prefill, decode ticks and verify rounds by
/// width with a page-sized KV behind every slot.
#[test]
#[ignore]
fn bench_phases() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((model, pages)) = fixtures() else {
        return;
    };
    let mut m = paddock_engine::gpu_model::nemotron_parse::GpuNemotronParse::load_dir(
        exec,
        &model,
        16_384,
        paddock_engine::gpu::KvDtype::Fp16,
    )
    .expect("load");
    let before = paddock_engine::generator::Generator::device_mem_used(&m);
    paddock_engine::generator::Generator::enable_batch(&mut m, 32).expect("enable");
    say(&format!(
        "resident: weights {:?} B, KV pool {:?} B, device in use {:?} B at load -> {:?} B with \
         the 32-slot batch state",
        paddock_engine::generator::Generator::weights_mem_bytes(&m),
        paddock_engine::generator::Generator::kv_mem_bytes(&m),
        before,
        paddock_engine::generator::Generator::device_mem_used(&m),
    ));
    let f = pages
        .iter()
        .find(|f| f.name == "page_000")
        .expect("page_000");
    let (w, h) = wh(&f.meta["input_wh"]);
    let rgb = std::fs::read(f.dir.join("input.rgb")).expect("input.rgb");
    let cfg = NemotronParseConfig::read(&model).expect("cfg");
    let page = prep_page(&rgb, w, h, &cfg);
    let t = std::time::Instant::now();
    let pp = prep_page(&rgb, w, h, &cfg);
    say(&format!(
        "host prep {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    ));
    let only_r: Option<usize> = std::env::var("NP_BENCH_R")
        .ok()
        .and_then(|v| v.parse().ok());
    for n in [1usize, 2] {
        if only_r.is_some() {
            break;
        }
        let group: Vec<&paddock_engine::gpu_model::nemotron_parse::PagePrep> =
            std::iter::repeat_n(&page, n).collect();
        let _ = m.tower_mut().encode(&group).expect("warm");
        m.tower_mut().sync().expect("sync");
        let t = std::time::Instant::now();
        for _ in 0..3 {
            let _ = m.tower_mut().encode(&group).expect("encode");
        }
        m.tower_mut().sync().expect("sync");
        say(&format!(
            "tower pass x{n}: {:.1} ms a page",
            t.elapsed().as_secs_f64() * 1e3 / 3.0 / n as f64
        ));
    }
    drop(pp);
    // a page's prefill: decode_page_greedy with one generated token
    let prompt: Vec<u32> = f.meta["prompt_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let chunks = vec![
        paddock_engine::service::MmChunk::Image {
            rgb: rgb.clone(),
            w,
            h,
        },
        paddock_engine::service::MmChunk::Text(prompt),
    ];
    for s in 0..32 {
        m.decode_page_greedy(s, &chunks, 1, None)
            .expect("fill slot");
    }
    let t = std::time::Instant::now();
    m.decode_page_greedy(0, &chunks, 1, None).expect("prefill");
    say(&format!(
        "encode + prefill one page: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3
    ));
    let rs = only_r.map_or(vec![1usize, 4, 8, 16, 32], |r| vec![r]);
    for r in rs {
        let d = m.bench_decode_tick(r, 2960).expect("tick");
        let v = m.bench_verify_round(r, 2960).expect("verify");
        say(&format!(
            "r={r:2}: decode tick {d:.2} ms ({:.0} tok/s), verify round {v:.2} ms",
            r as f64 * 1e3 / d
        ));
    }
}
