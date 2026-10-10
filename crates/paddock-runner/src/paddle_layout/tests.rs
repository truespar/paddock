//! The pipeline's host steps against the reference's own run over the OCR
//! battery (PaddleX 3.7.2's PaddleOCR-VL 1.6 pipeline, reading through this
//! runner): `<models>/ocr-battery/doclayout-official/` holds each page's
//! PNG, its `res.json` (layout boxes + the parsed blocks), its `.md`, and
//! `requests.jsonl` - every region request the pipeline sent, in order.
//! Skipped when the directory is absent.

use std::path::PathBuf;

use paddock_engine::gpu_model::doclayout::{LABELS, LayoutBox};
use serde_json::Value;

use super::{markdown, prep};

fn official() -> Option<PathBuf> {
    let root = std::env::var_os("PADDOCK_MODELS")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("paddock/models")))?;
    let d = root.join("ocr-battery").join("doclayout-official");
    d.join("requests.jsonl").is_file().then_some(d)
}

fn pages(dir: &std::path::Path) -> Vec<(String, Value)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".res.json"))
        .collect();
    v.sort();
    v.iter()
        .map(|p| {
            let stem = p
                .file_name()
                .unwrap()
                .to_string_lossy()
                .replace(".res.json", "");
            (
                stem,
                serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap(),
            )
        })
        .collect()
}

fn bbox(v: &Value) -> [i32; 4] {
    let a: Vec<i32> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as i32)
        .collect();
    [a[0], a[1], a[2], a[3]]
}

#[test]
fn markdown_matches_the_reference_byte_for_byte() {
    let Some(dir) = official() else {
        eprintln!("skipped: no official PaddleOCR-VL run");
        return;
    };
    let mut n = 0;
    for (stem, j) in pages(&dir) {
        let blocks: Vec<(String, String, [i32; 4])> = j["parsing_res_list"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                (
                    b["block_label"].as_str().unwrap().to_owned(),
                    b["block_content"].as_str().unwrap_or("").to_owned(),
                    bbox(&b["block_bbox"]),
                )
            })
            .collect();
        let md: Vec<markdown::MdBlock> = blocks
            .iter()
            .map(|(l, c, b)| markdown::MdBlock {
                label: l,
                content: c,
                bbox: *b,
            })
            .collect();
        let got = markdown::page(&md, j["width"].as_u64().unwrap() as usize);
        let want = std::fs::read_to_string(dir.join(format!("{stem}.md"))).unwrap();
        assert_eq!(got, want, "{stem}");
        n += 1;
    }
    assert!(n > 0);
}

#[test]
fn regions_and_requests_match_the_reference() {
    let Some(dir) = official() else {
        eprintln!("skipped: no official PaddleOCR-VL run");
        return;
    };
    let reqs: Vec<Value> = std::fs::read_to_string(dir.join("requests.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let mut sent = Vec::new();
    for (stem, j) in pages(&dir) {
        let boxes: Vec<LayoutBox> = j["layout_det_res"]["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                let cls = b["cls_id"].as_u64().unwrap() as usize;
                assert_eq!(LABELS[cls], b["label"].as_str().unwrap());
                LayoutBox {
                    cls,
                    score: b["score"].as_f64().unwrap() as f32,
                    bbox: bbox(&b["coordinate"]),
                    order: b["order"].as_u64().map(|o| o as u32),
                    query: 0,
                    polygon: None,
                }
            })
            .collect();
        let page = image::open(dir.join(format!("{stem}.png")))
            .unwrap()
            .to_rgb8();
        let blocks = prep::merge(prep::blocks(&page, &prep::filter_overlap(&boxes)));
        let want: Vec<(String, [i32; 4])> = j["parsing_res_list"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| {
                (
                    b["block_label"].as_str().unwrap().to_owned(),
                    bbox(&b["block_bbox"]),
                )
            })
            .collect();
        let got: Vec<(String, [i32; 4])> = blocks
            .iter()
            .map(|b| (b.label.to_owned(), b.bbox))
            .collect();
        assert_eq!(got, want, "{stem}: blocks");
        for b in &blocks {
            if let Some((img, task)) = prep::request(b) {
                sent.push((img.width() as u64, img.height() as u64, task));
            }
        }
    }
    // the reference's client sends a batch concurrently, so its log is in
    // arrival order: the requests are compared as a set
    let mut want: Vec<(u64, u64, String)> = reqs
        .iter()
        .map(|r| {
            let im = &r["image"];
            (
                im[0].as_u64().unwrap(),
                im[1].as_u64().unwrap(),
                r["text"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    let mut got: Vec<(u64, u64, String)> = sent
        .iter()
        .map(|(w, h, t)| (*w, *h, (*t).to_owned()))
        .collect();
    want.sort();
    got.sort();
    assert_eq!(got.len(), want.len(), "request count");
    assert_eq!(got, want, "crops and prompts");
}

/// PaddleX's own auto-shape run (`ocr-battery/doclayout-official/polygons/`):
/// its layout boxes with their outlines through our overlap filter and
/// outline crop - the same blocks, and every crop byte-identical to its
/// `CropByBoxes`, whited pixels and all.
#[test]
fn outline_crops_match_the_reference() {
    use sha2::Digest as _;
    let Some(dir) = official() else {
        eprintln!("skipped: no official PaddleOCR-VL run");
        return;
    };
    let dir = dir.join("polygons");
    if !dir.is_dir() {
        eprintln!("skipped: no PaddleX polygon dump");
        return;
    }
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    pages.sort();
    let (mut crops, mut whited) = (0, 0);
    for page in &pages {
        let stem = page.file_stem().unwrap().to_string_lossy().to_string();
        let j: Value = serde_json::from_slice(&std::fs::read(page).unwrap()).unwrap();
        let boxes: Vec<LayoutBox> = j["layout"]["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| LayoutBox {
                cls: b["cls_id"].as_u64().unwrap() as usize,
                score: b["score"].as_f64().unwrap() as f32,
                bbox: bbox(&b["coordinate"]),
                order: b["order"].as_u64().map(|o| o as u32),
                query: 0,
                polygon: Some(
                    b["polygon_points"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|p| [p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32])
                        .collect(),
                ),
            })
            .collect();
        let img = image::open(dir.parent().unwrap().join(format!("{stem}.png")))
            .unwrap()
            .to_rgb8();
        let kept = prep::filter_overlap(&boxes);
        let want = j["crops"].as_array().unwrap();
        assert_eq!(kept.len(), want.len(), "{stem}: filtered boxes");
        for (b, w) in kept.iter().zip(want) {
            assert_eq!(b.bbox, bbox(&w["coordinate"]), "{stem}: box");
            let c = prep::crop_outline(&img, b.bbox, b.polygon.as_ref());
            let sha = sha2::Sha256::digest(c.as_raw());
            let hex: String = sha.iter().take(8).map(|v| format!("{v:02x}")).collect();
            let plain = prep::crop(&img, b.bbox);
            let n = c
                .pixels()
                .zip(plain.pixels())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                (hex.as_str(), n as u64),
                (w["sha"].as_str().unwrap(), w["whited"].as_u64().unwrap()),
                "{stem} {:?}",
                b.bbox
            );
            crops += 1;
            whited += usize::from(n > 0);
        }
    }
    eprintln!(
        "{} pages: {crops} crops identical to the reference's ({whited} with pixels whited out)",
        pages.len()
    );
    assert!(crops > 0);
}
