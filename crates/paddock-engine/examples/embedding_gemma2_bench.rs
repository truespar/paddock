//! EmbeddingGemma 2 pass timer: wall time of one blocking `embed` (submit +
//! collect) over the oracle fixture's workloads at several pack widths -
//! the encoder's GPU cost without HTTP, tokenization or the scheduler.
//!
//! Usage: EG2_GGUF=<embeddinggemma-2-Q8_0.gguf> EG2_FIXTURE=<oracle json>
//!        PADDOCK_PACK=... [REPS=50] [ONLY="1 query"] embedding_gemma2_bench
//! With EG2_MMPROJ=<mmproj-BF16.gguf> and EG2_PICTURES=<dir> (the media
//! oracle's `originals/`) it also times picture passes at the 280-token
//! budget: the resize alone, one picture's whole pass, and eight pictures in
//! one pass.
// A development probe: it runs on a box its author is looking at, and a
// failure should stop it where it happened rather than be reported.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::embedding_gemma2::{GpuEmbeddingGemma2, media};
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;

fn main() {
    let pack = std::env::var("PADDOCK_PACK").expect("set PADDOCK_PACK");
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let exec = Arc::new(GpuExecutor::new(0, pack.as_ref()).expect("executor"));
    let map =
        MappedGguf::open(std::env::var("EG2_GGUF").expect("set EG2_GGUF").as_ref()).expect("gguf");
    let mut m = GpuEmbeddingGemma2::load(exec, &map, 8192).expect("load");
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("EG2_FIXTURE").expect("set EG2_FIXTURE"))
            .expect("fixture"),
    )
    .expect("json");
    let case = |name: &str| -> Vec<Vec<u32>> {
        let c = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        serde_json::from_value(c["ids"].clone()).unwrap()
    };
    let query = case("short");
    let docs = case("ragged");
    let mut runs: Vec<(String, Vec<Vec<u32>>)> = vec![
        ("1 query".into(), query.clone()),
        ("5 docs".into(), docs.clone()),
    ];
    for n in [8usize, 32, 128] {
        runs.push((format!("{n} queries"), vec![query[0].clone(); n]));
    }
    runs.push(("window".into(), case("window")));
    runs.push(("long".into(), case("long")));
    let mut media: Vec<Vec<Vec<MmChunk>>> = vec![Vec::new(); runs.len()];
    if let Ok(mmproj) = std::env::var("EG2_MMPROJ") {
        m.attach_mmproj(
            &MappedGguf::open(mmproj.as_ref()).expect("mmproj"),
            280,
            true,
        )
        .expect("attach");
        let dir =
            std::path::PathBuf::from(std::env::var("EG2_PICTURES").expect("set EG2_PICTURES"));
        let pics: Vec<MmChunk> = [
            "house.jpg",
            "receipt.png",
            "chart_rgba.png",
            "banner_gray.png",
        ]
        .iter()
        .map(|f| {
            let im = image::open(dir.join(f)).unwrap().into_rgb8();
            let (w, h) = (im.width() as usize, im.height() as usize);
            MmChunk::Image {
                rgb: im.into_raw(),
                w,
                h,
            }
        })
        .collect();
        let seq = |c: &MmChunk| {
            let MmChunk::Image { w, h, .. } = c else {
                unreachable!()
            };
            let mut s = vec![2, media::BOI_TOKEN];
            s.extend(std::iter::repeat_n(
                media::IMAGE_TOKEN,
                m.image_tokens(*w, *h).unwrap(),
            ));
            s.extend([media::EOI_TOKEN, 1]);
            s
        };
        let eight: Vec<MmChunk> = pics.iter().cycle().take(8).cloned().collect();
        runs.push(("1 picture".into(), vec![seq(&pics[0])]));
        media.push(vec![vec![pics[0].clone()]]);
        runs.push(("8 pictures".into(), eight.iter().map(seq).collect()));
        media.push(eight.iter().map(|c| vec![c.clone()]).collect());
        let MmChunk::Image { rgb, w, h } = &pics[0] else {
            unreachable!()
        };
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            m.image_tower().unwrap().resize(rgb, *w, *h).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("      resize: {w} x {h} picture {ms:8.3} ms");
    }
    // ONLY=<name>: time just that workload (a profiler run wants one shape)
    let only = std::env::var("ONLY").ok();
    for ((name, seqs), media) in runs
        .iter()
        .zip(&media)
        .filter(|((n, _), _)| only.as_ref().is_none_or(|o| o == n))
    {
        for _ in 0..3 {
            m.embed_media(seqs, media, None).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            m.embed_media(seqs, media, None).unwrap();
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        let rows: usize = seqs.iter().map(Vec::len).sum();
        println!(
            "{name:>12}: {rows:5} rows {ms:8.3} ms/pass  {:9.0} tok/s",
            rows as f64 / ms * 1e3
        );
    }
}
