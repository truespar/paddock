//! Does grouping pictures into one prefill pass make a picture cheaper?
//!
//! Distinct pictures (no picture-store reuse), prefilled two ways on the same
//! model: one per pass (`forward_prefill_slot_mm`, the stepped lane's shape for
//! a lone or large picture) and together (`forward_prefill_multimodal_batch`,
//! the batched wave: grouped tower calls plus batched passes). Two sizes: a
//! phone photo at the default `detail` cap (~4K rows) and a small one (~200
//! rows). Prints per-picture times. It also checks that the grouped pass
//! answers as the per-pass one does (first token per picture), since a speedup
//! that changed the answers would be no speedup.
//!
//! A measurement, not a gate: PADDOCK_HEAVY_TESTS=1, --release, --nocapture.
//! QWEN36_MM_DIR names the model dir (a Qwen3.6/3.8-27B Q8_0 + mmproj).

mod common;

use std::time::Instant;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn setup(max_batch: usize) -> Option<(GpuQwen35, GgufTokenizer)> {
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
        common::missing(&format!("no Q8_0 backbone + mmproj in {}", dir.display()));
        return None;
    };
    let exec = common::gpu_arc()?;
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen35::load(exec, &map, 16384).expect("load 27B");
    let mm = MappedGguf::open(&mmproj_path).expect("open mmproj");
    m.attach_vision(&mm).expect("attach vision");
    m.enable_batch(max_batch).expect("enable_batch");
    Some((m, tok))
}

/// A distinct picture per seed: smooth color fields, so the tower sees real
/// structure rather than noise, and no two hash alike.
fn picture(w: usize, h: usize, seed: u32) -> MmChunk {
    let mut rgb = vec![0u8; w * h * 3];
    let s = seed as f32 * 0.37;
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let i = (y * w + x) * 3;
            rgb[i] = ((fx * 6.0 + s).sin() * 120.0 + 128.0) as u8;
            rgb[i + 1] = ((fy * 5.0 + s * 1.7).cos() * 120.0 + 128.0) as u8;
            rgb[i + 2] = (((fx + fy) * 4.0 + s * 2.3).sin() * 120.0 + 128.0) as u8;
        }
    }
    MmChunk::Image { rgb, w, h }
}

fn prompt(tok: &GgufTokenizer, pic: MmChunk) -> Vec<MmChunk> {
    vec![
        MmChunk::Text(tok.encode("<|im_start|>user\n").expect("enc")),
        pic,
        MmChunk::Text(
            tok.encode("Describe this photo in two sentences.<|im_end|>\n<|im_start|>assistant\n")
                .expect("enc"),
        ),
    ]
}

fn amax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Per-picture seconds: `n` distinct pictures one per pass, then `n` more
/// grouped, then the first `n` grouped again for the answer check (their
/// tower outputs now come from the picture store, which does not change what
/// the pass computes).
fn measure(m: &mut GpuQwen35, tok: &GgufTokenizer, w: usize, h: usize, n: usize, seed0: u32) {
    let mk = |k: u32| prompt(tok, picture(w, h, seed0 + k));
    // warm the kernels and the pass scratch at this shape
    let _ = m.forward_prefill_slot_mm(0, &mk(1000)).expect("warm");
    let serial: Vec<Vec<MmChunk>> = (0..n as u32).map(mk).collect();
    let t = Instant::now();
    let mut ref_first = Vec::new();
    let mut rows = 0;
    for (slot, c) in serial.iter().enumerate() {
        let (l, r) = m.forward_prefill_slot_mm(slot, c).expect("serial");
        ref_first.push(amax(&l));
        rows = r;
    }
    let ts = t.elapsed().as_secs_f64() / n as f64;
    let grouped: Vec<(usize, Vec<MmChunk>)> = (0..n).map(|k| (k, mk(100 + k as u32))).collect();
    let t = Instant::now();
    let res = m.forward_prefill_multimodal_batch(grouped);
    let tg = t.elapsed().as_secs_f64() / n as f64;
    assert!(res.iter().all(|(_, r)| r.is_ok()), "grouped prefill failed");
    let again: Vec<(usize, Vec<MmChunk>)> = serial.into_iter().enumerate().collect();
    let res = m.forward_prefill_multimodal_batch(again);
    let same = res
        .iter()
        .zip(&ref_first)
        .filter(|((_, r), want)| amax(&r.as_ref().expect("regroup").0) == **want)
        .count();
    eprintln!(
        "GROUP {w}x{h} ({rows} rows/prompt) x{n}: one per pass {ts:.3} s/picture, grouped \
         {tg:.3} s/picture ({:+.1}%), first token equal {same}/{n}",
        (tg / ts - 1.0) * 100.0
    );
}

#[test]
fn grouping_pictures_per_pass() {
    let Some((mut m, tok)) = setup(8) else {
        return;
    };
    // the size the runner hands a 12 MP photo at detail "auto" (~4K rows)
    measure(&mut m, &tok, 2368, 1760, 4, 1);
    // a small photo (~200 rows), where a pass's fixed cost dominates
    measure(&mut m, &tok, 448, 448, 8, 50);
}
