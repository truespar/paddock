//! PP-DocLayoutV3 on CUDA against golden taps from the Transformers reference
//! (itself checked against PaddleX's own inference: same detections, scores
//! within 1e-3, boxes within half a pixel), fed the pipeline's own pixels -
//! the OpenCV `INTER_CUBIC` 800 x 800 resize the golden directory carries.
//!
//! Golden layout: `<models>/ocr-battery/doclayout/<page>/manifest.json` plus
//! one raw little-endian f32 file per tap, feature maps NHWC. Checkpoint:
//! `<models>/PP-DocLayoutV3_safetensors` (or `PP_DOCLAYOUT_DIR`).
//!
//! Gated on: CUDA device + built pack + both directories.

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::doclayout::GpuDocLayout;

fn golden_pages() -> Vec<PathBuf> {
    let Some(root) = common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("doclayout"))
        .find(|p| p.is_dir())
    else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    v.retain(|p| p.join("manifest.json").exists());
    v.sort();
    v
}

fn checkpoint() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("PP_DOCLAYOUT_DIR") {
        return Some(PathBuf::from(d));
    }
    common::model_roots()
        .iter()
        .map(|r| r.join("PP-DocLayoutV3_safetensors"))
        .find(|p| p.join("model.safetensors").exists())
}

fn tap(dir: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest json");
    let t = &man["taps"][name];
    let shape: Vec<usize> = t["shape"]
        .as_array()
        .unwrap_or_else(|| panic!("no tap {name}"))
        .iter()
        .map(|v| v.as_u64().expect("dim") as usize)
        .collect();
    let bytes = std::fs::read(dir.join(t["file"].as_str().expect("file"))).expect("tap file");
    let v = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (shape, v)
}

/// relative L2 and the largest absolute difference
fn diff(got: &[f32], want: &[f32]) -> (f64, f32) {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f32);
    for (g, w) in got.iter().zip(want) {
        let d = (g - w) as f64;
        num += d * d;
        den += (*w as f64) * (*w as f64);
        mx = mx.max((g - w).abs());
    }
    ((num / den.max(1e-30)).sqrt(), mx)
}

#[test]
fn backbone_and_encoder_match_the_reference() {
    let pages = golden_pages();
    if pages.is_empty() {
        common::missing("no PP-DocLayoutV3 golden taps");
        return;
    }
    let Some(dir) = checkpoint() else {
        common::missing("no PP-DocLayoutV3 checkpoint");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let m = GpuDocLayout::load(exec.clone(), &dir).expect("load");
    eprintln!("weights {:.1} MB", m.weight_bytes as f64 / 1e6);
    for page in &pages {
        let rgb = std::fs::read(page.join("input_u8.rgb")).unwrap();
        let (out, enc) = m.encode(&rgb).expect("encode");
        let name = page.file_name().unwrap().to_string_lossy().to_string();
        let mut planes: Vec<(String, &paddock_engine::gpu_model::doclayout::Plane, f64)> =
            Vec::new();
        for (i, l) in enc.levels.iter().enumerate() {
            planes.push((format!("encoder.last_hidden_state.{i}"), l, 3e-2));
        }
        planes.push(("encoder.mask_feat".into(), &enc.mask_feat, 3e-2));
        for (tapn, pl, tol) in planes {
            let (shape, want) = tap(page, &tapn);
            assert_eq!(shape, vec![1, pl.h, pl.w, pl.c], "{tapn} shape");
            let got: Vec<f32> = exec
                .to_host_f16_len(&pl.data, pl.h * pl.w * pl.c)
                .unwrap()
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let (rel, mx) = diff(&got, &want);
            eprintln!(
                "{name} {tapn} {}x{}x{}: rel L2 {rel:.2e}, max |d| {mx:.3}",
                pl.h, pl.w, pl.c
            );
            assert!(rel < tol, "{tapn}: rel L2 {rel}");
        }
        for (i, st) in out.stages.iter().enumerate() {
            let (shape, want) = tap(page, &format!("backbone.{i}.0"));
            assert_eq!(shape, vec![1, st.h, st.w, st.c], "stage {i} shape");
            let got: Vec<f32> = exec
                .to_host_f16_len(&st.data, st.h * st.w * st.c)
                .unwrap()
                .iter()
                .map(|v| v.to_f32())
                .collect();
            let (rel, mx) = diff(&got, &want);
            eprintln!(
                "{} stage {i} {}x{}x{}: rel L2 {rel:.2e}, max |d| {mx:.3}",
                page.file_name().unwrap().to_string_lossy(),
                st.h,
                st.w,
                st.c
            );
            assert!(rel < 2e-2, "stage {i}: rel L2 {rel}");
        }
    }
}

/// The memory rows the reference's query selection kept, in its rank order:
/// each `enc_topk_logits` row found bit-for-bit among `enc_outputs_class`.
fn reference_topk(page: &Path) -> Vec<u32> {
    let (_, all) = tap(page, "out.enc_outputs_class");
    let (_, top) = tap(page, "out.enc_topk_logits");
    let key = |r: &[f32]| r.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
    let mut rows: std::collections::HashMap<Vec<u32>, Vec<u32>> = Default::default();
    for (i, r) in all.as_chunks::<25>().0.iter().enumerate().rev() {
        rows.entry(key(r)).or_default().push(i as u32);
    }
    top.as_chunks::<25>()
        .0
        .iter()
        .map(|r| {
            rows.get_mut(&key(r))
                .and_then(|v| v.pop())
                .expect("topk row")
        })
        .collect()
}

#[test]
fn decoder_matches_the_reference() {
    let pages = golden_pages();
    if pages.is_empty() {
        common::missing("no PP-DocLayoutV3 golden taps");
        return;
    }
    let Some(dir) = checkpoint() else {
        common::missing("no PP-DocLayoutV3 checkpoint");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let m = GpuDocLayout::load(exec.clone(), &dir).expect("load");
    let q = paddock_engine::gpu_model::doclayout::QUERIES;
    for page in &pages {
        let name = page.file_name().unwrap().to_string_lossy().to_string();
        let rgb = std::fs::read(page.join("input_u8.rgb")).unwrap();
        let (_, _, dec) = m.detect(&rgb).expect("detect");
        if let Some(d) = std::env::var_os("PD_DL_DUMP") {
            // the decoder's outputs for an offline look (f32 / u32 LE)
            let d = PathBuf::from(d).join(&name);
            std::fs::create_dir_all(&d).unwrap();
            let le = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            let idx: Vec<u8> = dec.topk.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(d.join("topk.u32"), idx).unwrap();
            std::fs::write(d.join("init_ref.f32"), le(&dec.init_ref)).unwrap();
            for (l, v) in dec.layers.iter().enumerate() {
                std::fs::write(d.join(format!("layer{l}.f32")), le(v)).unwrap();
            }
        }
        // align our queries to the reference's by memory row: f16 noise may
        // swap near-tied members at the top-300 boundary
        let want_idx = reference_topk(page);
        let pos: std::collections::HashMap<u32, usize> =
            want_idx.iter().enumerate().map(|(p, &r)| (r, p)).collect();
        let pairs: Vec<(usize, usize)> = dec
            .topk
            .iter()
            .enumerate()
            .filter_map(|(i, r)| pos.get(r).map(|&p| (i, p)))
            .collect();
        let same_rank = pairs.iter().filter(|(i, p)| i == p).count();
        eprintln!(
            "{name} selection: {} / {q} rows shared, {same_rank} at the same rank",
            pairs.len()
        );
        assert!(
            pairs.len() >= q - 6,
            "selection drifted: {} shared",
            pairs.len()
        );
        let gather = |v: &[f32], w: usize, ours: bool| -> Vec<f32> {
            pairs
                .iter()
                .flat_map(|&(i, p)| {
                    let r = if ours { i } else { p };
                    v[r * w..(r + 1) * w].to_vec()
                })
                .collect()
        };
        let check = |what: &str, got: &[f32], want: &[f32], w: usize, tol: f64| {
            let (rel, mx) = diff(&gather(got, w, true), &gather(want, w, false));
            eprintln!("{name} {what}: rel L2 {rel:.2e}, max |d| {mx:.4}");
            assert!(rel < tol, "{what}: rel L2 {rel}");
        };
        // the initial boxes (the reference keeps them pre-sigmoid): a mask's
        // bounding box, so one pixel near zero logit far from the rest can
        // stretch a background query's box - a gross-break bound only
        let (_, init) = tap(page, "out.init_reference_points");
        let init: Vec<f32> = init.iter().map(|u| 1.0 / (1.0 + (-u).exp())).collect();
        check("init_ref", &dec.init_ref, &init, 4, 0.25);
        // the hidden states drift more than the encoder did, and not from
        // the decoder's arithmetic: replayed through the reference with OUR
        // selection, layer 0 lands at 4e-3 (median row 1e-3) - the rest is
        // the handful of queries swapped at the top-300 boundary, which
        // self-attention spreads to their neighbours. Low-ranked background
        // queries carry most of it, so the whole-array numbers only catch a
        // gross break; the rows that become detections are held tight below.
        for (l, got) in dec.layers.iter().enumerate() {
            let (_, want) = tap(page, &format!("decoder.layer{l}"));
            check(&format!("decoder.layer{l}"), got, &want, 256, 0.25);
        }
        let (_, logits) = tap(page, "out.logits");
        check("logits", &dec.logits, &logits, 25, 0.25);
        let (_, boxes) = tap(page, "out.pred_boxes");
        check("pred_boxes", &dec.boxes, &boxes, 4, 0.25);
        let masks = exec.to_host(&dec.masks).unwrap();
        let (_, want_masks) = tap(page, "out.out_masks");
        // what the postprocess will keep: every reference query scoring past
        // 0.3 must be ours too, its score within 0.02, its box within a
        // pixel of the 800 x 800 input, and its mask (thresholded at 0.5,
        // what the polygons are cut from) within 1 % of its pixels
        let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
        let (mut kept, mut worst_px, mut worst_mask) = (0, 0f32, 0f64);
        for &(i, p) in &pairs {
            let ws = logits[p * 25..(p + 1) * 25]
                .iter()
                .copied()
                .fold(f32::MIN, f32::max);
            if sig(ws) <= 0.3 {
                continue;
            }
            kept += 1;
            let gs = dec.logits[i * 25..(i + 1) * 25]
                .iter()
                .copied()
                .fold(f32::MIN, f32::max);
            let db = (0..4)
                .map(|k| (dec.boxes[i * 4 + k] - boxes[p * 4 + k]).abs() * 800.0)
                .fold(0f32, f32::max);
            let px = 200 * 200;
            let (gm, wm) = (
                &masks[i * px..(i + 1) * px],
                &want_masks[p * px..(p + 1) * px],
            );
            let on = wm.iter().filter(|&&v| v > 0.0).count().max(1);
            let flips = gm
                .iter()
                .zip(wm)
                .filter(|(g, w)| (**g > 0.0) != (**w > 0.0))
                .count();
            let mf = flips as f64 / on as f64;
            worst_px = worst_px.max(db);
            worst_mask = worst_mask.max(mf);
            assert!(
                (sig(gs) - sig(ws)).abs() < 0.02 && db < 1.0 && mf < 0.01,
                "{name} query {p}: score {} vs {}, box off {db:.2} px, mask {:.2} % flipped",
                sig(gs),
                sig(ws),
                mf * 100.0
            );
        }
        eprintln!(
            "{name}: {kept} detections past 0.3 agree (box within {worst_px:.2} px, mask \
             within {:.2} % of its pixels)",
            worst_mask * 100.0
        );
        let lost = want_idx
            .iter()
            .enumerate()
            .filter(|(p, _)| {
                let ws = logits[p * 25..(p + 1) * 25]
                    .iter()
                    .copied()
                    .fold(f32::MIN, f32::max);
                sig(ws) > 0.3 && !pairs.iter().any(|&(_, pp)| pp == *p)
            })
            .count();
        assert_eq!(
            lost, 0,
            "{name}: {lost} detections fell out of the selection"
        );
    }
}

/// The official run's outline of box `i` on `page` (its `res.json`).
fn official_outline(page: &Path, i: usize) -> Vec<[f32; 2]> {
    let stem = page
        .file_stem()
        .expect("page name")
        .to_string_lossy()
        .to_string();
    let res: serde_json::Value = serde_json::from_slice(
        &std::fs::read(page.with_file_name(format!("{stem}.res.json"))).expect("res.json"),
    )
    .expect("res json");
    res["layout_det_res"]["boxes"][i]["polygon_points"]
        .as_array()
        .expect("polygon_points")
        .iter()
        .map(|p| {
            [
                p[0].as_f64().expect("x") as f32,
                p[1].as_f64().expect("y") as f32,
            ]
        })
        .collect()
}

/// The official pipeline's own detections over the battery pages:
/// `<models>/ocr-battery/doclayout-official/page_NNN.{png,boxes.json}`
/// (PaddleX's PaddleOCR-VL 1.6 pipeline, `layout_det_res`).
fn official_pages() -> Vec<PathBuf> {
    let Some(root) = common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("doclayout-official"))
        .find(|p| p.is_dir())
    else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    v.retain(|p| p.extension().is_some_and(|e| e == "png"));
    v.sort();
    v
}

#[test]
fn layout_matches_the_official_pipeline() {
    use paddock_engine::gpu_model::doclayout::INPUT;
    let pages = official_pages();
    if pages.is_empty() {
        common::missing("no official PP-DocLayoutV3 detections");
        return;
    }
    let Some(dir) = checkpoint() else {
        common::missing("no PP-DocLayoutV3 checkpoint");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let m = GpuDocLayout::load(exec.clone(), &dir).expect("load");
    let (mut boxes, mut exact, mut near, mut pages_same) = (0, 0, 0, 0);
    let (mut outlines_same, mut shaped, mut shaped_same) = (0, 0, 0);
    let mut outline_diffs: Vec<String> = Vec::new();
    let mut worst = 0i32;
    for page in &pages {
        let name = page.file_stem().unwrap().to_string_lossy().to_string();
        let img = image::open(page).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let t0 = std::time::Instant::now();
        let (got, peak) = exec
            .pool_peak_during(|| Ok(m.layout(img.as_raw(), w, h).expect("layout")))
            .unwrap();
        let _ = INPUT;
        if page == &pages[0] || page == pages.last().unwrap() {
            eprintln!(
                "{name}: forward {:.1} ms, weights {:.1} MB, transient peak {:.1} MB",
                t0.elapsed().as_secs_f64() * 1e3,
                m.weight_bytes as f64 / 1e6,
                peak as f64 / 1e6
            );
        }
        let want: serde_json::Value =
            serde_json::from_slice(&std::fs::read(page.with_extension("boxes.json")).unwrap())
                .unwrap();
        let want = want["boxes"].as_array().unwrap();
        boxes += want.len();
        let mut same = got.len() == want.len();
        for (i, wb) in want.iter().enumerate() {
            let cls = wb["cls_id"].as_u64().unwrap() as usize;
            let c: Vec<i32> = wb["coordinate"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as i32)
                .collect();
            let order = wb["order"].as_u64().map(|o| o as u32);
            let outline: Vec<[f32; 2]> = official_outline(page, i);
            let Some(g) = got.get(i) else {
                eprintln!("{name} #{i}: missing {} {c:?}", wb["label"]);
                same = false;
                continue;
            };
            let d = (0..4).map(|k| (g.bbox[k] - c[k]).abs()).max().unwrap();
            let ds = (g.score as f64 - wb["score"].as_f64().unwrap()).abs();
            if g.cls == cls && g.order == order && d == 0 {
                exact += 1;
            }
            let plain = outline.len() == 4
                && outline[0] == [c[0] as f32, c[1] as f32]
                && outline[2] == [c[2] as f32, c[3] as f32];
            shaped += usize::from(!plain);
            if g.polygon.as_deref() == Some(outline.as_slice()) {
                outlines_same += 1;
                shaped_same += usize::from(!plain);
            } else if g.cls == cls && d == 0 && outline_diffs.len() < 6 {
                outline_diffs.push(format!("{name} #{i}: got {:?} want {outline:?}", g.polygon));
            }
            if g.cls == cls && g.order == order && d <= 2 && ds < 0.02 {
                near += 1;
                worst = worst.max(d);
            } else {
                same = false;
                eprintln!(
                    "{name} #{i}: got {} {:?} {:.3} ord {:?}, want {} {c:?} {:.3} ord {order:?}",
                    g.label(),
                    g.bbox,
                    g.score,
                    g.order,
                    wb["label"],
                    wb["score"].as_f64().unwrap()
                );
            }
        }
        for g in got.iter().skip(want.len()) {
            eprintln!("{name}: extra {} {:?} {:.3}", g.label(), g.bbox, g.score);
        }
        pages_same += same as usize;
    }
    eprintln!(
        "{} pages ({pages_same} identical lists): {boxes} official boxes, {exact} exact, \
         {near} within 2 px (worst {worst} px)",
        pages.len()
    );
    for d in &outline_diffs {
        eprintln!("{d}");
    }
    eprintln!(
        "outlines: {outlines_same} of {boxes} identical to the official run's ({shaped_same} of its \
         {shaped} that are not the plain box)"
    );
    // measured at landing: 64 of 64 lists identical, 794 of 794 boxes within
    // a pixel (776 exact); the margin is for a kernel change that moves an
    // f16 rounding, not for drift
    assert!(
        near * 100 >= boxes * 99,
        "only {near} of {boxes} boxes agree"
    );
    assert!(
        pages_same * 100 >= pages.len() * 95,
        "only {pages_same} lists identical"
    );
    // measured at landing: the outline is identical wherever the box is
    // (776 of 776); the boxes a pixel off carry outlines a pixel off
    assert!(
        outlines_same * 100 >= exact * 99,
        "only {outlines_same} outlines identical"
    );
}
