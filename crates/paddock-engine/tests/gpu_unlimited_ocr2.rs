//! Unlimited-OCR-2 (SR-SWA on the DeepSeek-OCR body) against the reference's
//! own `infer` path, dumped by the oracle script on the battery pages: the
//! prompt layout, the prefill head, and teacher-forced decode top-1 through
//! the page-count guard and the router's ring.
//!
//! Cases (one oracle directory each): one page in base mode, one page in
//! gundam (tiles) mode, three pages and forty pages in base mode - the last
//! one runs ~39K positions, past what one v1 sequence holds - and three pages
//! tiled with three different geometries (a portrait page, a two-page
//! spread, a page small enough to take the no-tile bail). The reference is
//! BF16 end to end; this lane is a Q8_0 decoder with f32 activations and the
//! f16 tower, so the bars are agreement bars, not bit bars.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use paddock_engine::gpu_model::deepseek_ocr::GpuDeepseekOcr;
use paddock_engine::service::{MmChunk, OcrCropMode};
use paddock_models::mapped::MappedGguf;

const BOS: u32 = 0;
const PAGE: u32 = 128_827;
const EOS: u32 = 1;
/// "Multi page parsing." - the checkpoint's only trained instruction
const INSTR: [u32; 4] = [37460, 4366, 76466, 16];

fn oracle_root() -> Option<PathBuf> {
    common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("uocr2-oracle"))
        .find(|p| p.join("p1-base").join("meta.json").exists())
}

fn gguf_pair() -> Option<(PathBuf, PathBuf)> {
    common::model_roots().iter().find_map(|r| {
        let d = r.join("Unlimited-OCR-2-GGUF");
        let (m, p) = (
            d.join("Unlimited-OCR-2-Q8_0.gguf"),
            d.join("mmproj-Unlimited-OCR-F16.gguf"),
        );
        (m.exists() && p.exists()).then_some((m, p))
    })
}

fn read_f32s(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[bi] {
            bi = i;
        }
    }
    bi as u32
}

/// max|Δ| over the reference's top-50 logits.
fn head_dmax(oracle: &[f32], got: &[f32]) -> f32 {
    let mut idx: Vec<usize> = (0..oracle.len()).collect();
    idx.sort_by(|&a, &b| oracle[b].partial_cmp(&oracle[a]).unwrap());
    idx[..50]
        .iter()
        .map(|&i| (got[i] - oracle[i]).abs())
        .fold(0f32, f32::max)
}

struct Case {
    s: usize,
    gen_ids: Vec<u32>,
    /// top-1 of the reference's raw logits per generated step
    top1: Vec<u32>,
    logits_pf: Vec<f32>,
    chunks: Vec<MmChunk>,
}

fn load_case(dir: &Path) -> Case {
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let s = meta["S"].as_u64().unwrap() as usize;
    let gen_ids: Vec<u32> = meta["gen_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let top1: Vec<u32> = meta["top5"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t[0][0].as_u64().unwrap() as u32)
        .collect();
    let crop = meta["crop"].as_bool().unwrap();
    let sizes = meta["page_sizes"].as_array().unwrap();
    let mut chunks = Vec::new();
    if !crop {
        chunks.push(MmChunk::OcrCrop(OcrCropMode::Base));
    } else if sizes.len() > 1 {
        // several pages tiled is a request override, as for the reference
        chunks.push(MmChunk::OcrCrop(OcrCropMode::Gundam));
    }
    chunks.push(MmChunk::Text(vec![BOS]));
    for (k, sz) in sizes.iter().enumerate() {
        let (w, h) = (
            sz[0].as_u64().unwrap() as usize,
            sz[1].as_u64().unwrap() as usize,
        );
        let rgb = std::fs::read(dir.join(format!("page_{k}.rgb"))).unwrap();
        assert_eq!(rgb.len(), w * h * 3);
        chunks.push(MmChunk::Text(vec![PAGE]));
        chunks.push(MmChunk::Image { rgb, w, h });
    }
    chunks.push(MmChunk::Text(INSTR.to_vec()));
    Case {
        s,
        gen_ids,
        top1,
        logits_pf: read_f32s(&dir.join("logits_prefill.f32")),
        chunks,
    }
}

/// Prefill one case on `slot`: prompt rows and the head against the reference.
fn prefill(m: &mut GpuDeepseekOcr, slot: usize, c: &Case, name: &str) {
    let (logits, rows) = m
        .multimodal_prefill_slot(slot, &c.chunks)
        .expect("mm prefill");
    assert_eq!(rows, c.s, "{name}: prompt rows vs the reference");
    let d = head_dmax(&c.logits_pf, &logits);
    eprintln!("{name}: prefill head50 max|Δ| {d:.3}");
    assert_eq!(
        argmax(&logits),
        c.gen_ids[0],
        "{name}: prefill argmax (the first <PAGE>)"
    );
}

/// Teacher-forced decode over the whole reference output: its generated ids
/// fed back, our top-1 against the reference's raw top-1 at every step that
/// has one (the guard bans EOS until the page count is reached; the reference
/// logged its logits before its own guard, so an EOS there would show as a miss).
fn teacher_forced(m: &mut GpuDeepseekOcr, slot: usize, c: &Case, name: &str) -> (usize, usize) {
    prefill(m, slot, c, name);
    let (mut agree, mut n) = (0usize, 0usize);
    for (s, &tok) in c.gen_ids.iter().enumerate() {
        m.batch_step_slots(&[tok], &[(c.s + s) as u32], &[slot as u32])
            .expect("decode step");
        let l = m.read_batch_logits(1).expect("logits");
        // step s's logits pick token s + 1 (the reference's top-1 there)
        if let Some(&want) = c.top1.get(s + 1) {
            agree += usize::from(argmax(&l) == want);
            n += 1;
        }
    }
    eprintln!("{name}: teacher-forced top-1 {agree}/{n}");
    (agree, n)
}

/// Free-running greedy decode of several cases side by side in one batch (one
/// row a slot): each must emit one <PAGE> a page and stop on EOS, as the
/// reference did. Q8_0 against BF16 parts the streams somewhere, so the shared
/// prefix is reported, not gated. `holes` adds a (token 0, position 0) row a
/// step for each listed slot - what the serving loop's dense step feeds every
/// slot that is not prefilled - and must change nothing. Returns the outputs.
fn free_run(
    m: &mut GpuDeepseekOcr,
    cases: &[(usize, &Case, &str)],
    holes: &[u32],
) -> Vec<Vec<u32>> {
    for (slot, c, name) in cases {
        prefill(m, *slot, c, name);
    }
    let mut out: Vec<Vec<u32>> = cases.iter().map(|(_, c, _)| vec![c.gen_ids[0]]).collect();
    let cap: Vec<usize> = cases.iter().map(|(_, c, _)| c.gen_ids.len() * 2).collect();
    loop {
        let live: Vec<usize> = (0..cases.len())
            .filter(|&i| out[i].last() != Some(&EOS) && out[i].len() < cap[i])
            .collect();
        if live.is_empty() {
            break;
        }
        let mut toks: Vec<u32> = live.iter().map(|&i| *out[i].last().unwrap()).collect();
        let mut pos: Vec<u32> = live
            .iter()
            .map(|&i| (cases[i].1.s + out[i].len() - 1) as u32)
            .collect();
        let mut slots: Vec<u32> = live.iter().map(|&i| cases[i].0 as u32).collect();
        for &h in holes {
            toks.push(0);
            pos.push(0);
            slots.push(h);
        }
        m.batch_step_slots(&toks, &pos, &slots)
            .expect("decode step");
        let l = m.read_batch_logits(slots.len()).expect("logits");
        let v = l.len() / slots.len();
        for (r, &i) in live.iter().enumerate() {
            out[i].push(argmax(&l[r * v..(r + 1) * v]));
        }
    }
    for (i, (_, c, name)) in cases.iter().enumerate() {
        let o = &out[i];
        let shared = o.iter().zip(&c.gen_ids).take_while(|(a, b)| a == b).count();
        let diff: Vec<usize> = (0..o.len().min(c.gen_ids.len()))
            .filter(|&k| o[k] != c.gen_ids[k])
            .collect();
        // one inserted token shifts every later position, but each <PAGE>
        // realigns the streams: count agreement page by page
        let split = |v: &[u32]| -> Vec<Vec<u32>> {
            let mut pages = vec![Vec::new()];
            for &t in v {
                if t == PAGE {
                    pages.push(Vec::new());
                }
                pages.last_mut().unwrap().push(t);
            }
            pages
        };
        let (po, pr) = (split(o), split(&c.gen_ids));
        let paged: usize = po
            .iter()
            .zip(&pr)
            .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| x == y).count())
            .sum();
        eprintln!(
            "{name}: page-aligned agreement {paged}/{} ({:.2}%)",
            c.gen_ids.len(),
            100.0 * paged as f64 / c.gen_ids.len() as f64
        );
        eprintln!(
            "{name}: {} positions differ, first {:?}",
            diff.len(),
            diff.iter()
                .take(8)
                .map(|&k| (k, o[k], c.gen_ids[k]))
                .collect::<Vec<_>>()
        );
        let pages = |v: &[u32]| v.iter().filter(|&&t| t == PAGE).count();
        eprintln!(
            "{name}: free run {} tokens (reference {}), shared prefix {shared}, <PAGE> {} (reference {})",
            o.len(),
            c.gen_ids.len(),
            pages(o),
            pages(&c.gen_ids)
        );
        assert_eq!(o.last(), Some(&EOS), "{name}: free run stops on EOS");
        assert!(
            paged as f64 >= 0.95 * c.gen_ids.len() as f64,
            "{name}: page-aligned agreement {paged}/{}",
            c.gen_ids.len()
        );
        assert_eq!(pages(o), pages(&c.gen_ids), "{name}: one <PAGE> a page");
    }
    out
}

#[test]
fn unlimited_ocr2_tracks_the_reference() {
    let Some(root) = oracle_root() else {
        common::missing("no Unlimited-OCR-2 oracle (ocr-battery/uocr2-oracle)");
        return;
    };
    let Some((model, mmproj)) = gguf_pair() else {
        common::missing("no Unlimited-OCR-2 GGUF pair");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    let mut m = GpuDeepseekOcr::load(Arc::clone(&exec), &map, 65_536).expect("load decoder");
    assert!(m.hp.dash.is_some(), "the v2 file carries the router keys");
    let mm = MappedGguf::open(&mmproj).expect("open mmproj");
    m.attach_vision(&mm).expect("attach vision");
    // slot 2 is never admitted: it only ever sees hole rows
    assert_eq!(m.enable_batch(3).expect("enable_batch"), 3);
    // p40-base is the long one: ~11K prompt rows and ~28K generated, past
    // the 32K a v1 sequence could hold
    let names = [
        "p1-base",
        "p1-gundam",
        "p3-base",
        "p3-gundam-mixed",
        "p40-base",
    ];
    let cases: Vec<(&str, Case)> = names
        .iter()
        .filter(|n| root.join(n).join("meta.json").exists())
        .map(|&n| (n, load_case(&root.join(n))))
        .collect();
    let case = |name: &str| cases.iter().find(|(n, _)| *n == name).map(|(n, c)| (*n, c));
    let mut all = (0usize, 0usize);
    for (i, (name, c)) in cases.iter().enumerate() {
        let (a, n) = teacher_forced(&mut m, i % 2, c, name);
        all = (all.0 + a, all.1 + n);
        assert!(
            a as f64 / n as f64 >= 0.95,
            "{name}: teacher-forced top-1 {a}/{n}"
        );
    }
    eprintln!("all cases: teacher-forced top-1 {}/{}", all.0, all.1);
    // the two one-page cases decode side by side
    let side: Vec<(usize, &Case, &str)> = ["p1-base", "p1-gundam"]
        .iter()
        .filter_map(|n| case(n))
        .enumerate()
        .map(|(i, (n, c))| (i, c, n))
        .collect();
    free_run(&mut m, &side, &[]);
    if let Some((n, c)) = case("p3-base") {
        let plain = free_run(&mut m, &[(0, c, n)], &[]).remove(0);
        // again beside a finished slot (1) and a never-admitted one (2)
        let holed = free_run(&mut m, &[(0, c, n)], &[1, 2]).remove(0);
        let same = plain.iter().zip(&holed).filter(|(a, b)| a == b).count();
        eprintln!(
            "{n}: with hole rows {same}/{} positions as without",
            plain.len()
        );
        assert!(
            same as f64 >= 0.95 * plain.len() as f64,
            "hole rows moved the read"
        );
    }
    // three pages tiled, each its own grid: 2x3, 4x3 and the small-image bail
    if let Some((n, c)) = case("p3-gundam-mixed") {
        free_run(&mut m, &[(1, c, n)], &[]);
    }
    if let Some((n, c)) = case("p40-base") {
        free_run(&mut m, &[(0, c, n)], &[]);
    }
}
