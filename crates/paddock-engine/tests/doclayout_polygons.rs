//! The region outlines (PaddleX's "auto" layout shape) against PaddleX
//! itself, on PaddleX's own inputs: for every battery page, the boxes and
//! the static graph's masks PaddleX handed `extract_polygon_points_by_masks`,
//! and what came out - with the stages (OpenCV's contours, approxPolyDP,
//! the custom vertices, the min-area quad) for a mismatch to name its step.
//! No GPU: this gates the geometry alone, network noise kept out.
//!
//! Data: `<models>/ocr-battery/doclayout-official/polygons/page_NNN.{json,masks.u8}`.

mod common;

use std::path::PathBuf;

use paddock_engine::gpu_model::doclayout::{geom, polygon};

fn pages() -> Vec<PathBuf> {
    let Some(dir) = common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery/doclayout-official/polygons"))
        .find(|p| p.is_dir())
    else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("polygon dump dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    v.sort();
    v
}

fn pts_i(v: &serde_json::Value) -> Vec<geom::Pt> {
    v.as_array()
        .expect("points")
        .iter()
        .map(|p| {
            [
                p[0].as_i64().expect("x") as i32,
                p[1].as_i64().expect("y") as i32,
            ]
        })
        .collect()
}

fn pts_f(v: &serde_json::Value) -> Vec<[f32; 2]> {
    v.as_array()
        .expect("points")
        .iter()
        .map(|p| {
            [
                p[0].as_f64().expect("x") as f32,
                p[1].as_f64().expect("y") as f32,
            ]
        })
        .collect()
}

#[test]
fn outlines_match_paddlex_on_its_own_masks() {
    let pages = pages();
    if pages.is_empty() {
        common::missing("no PaddleX polygon dump");
        return;
    }
    let (mut total, mut same, mut quads) = (0, 0, 0);
    let mut first_bad: Vec<String> = Vec::new();
    for page in &pages {
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(page).unwrap()).unwrap();
        let name = page.file_stem().unwrap().to_string_lossy().to_string();
        let shape: Vec<usize> = j["mask_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let raw = std::fs::read(page.with_extension("masks.u8")).unwrap();
        let side = shape[1];
        let masks: Vec<Vec<u8>> = raw.chunks_exact(side * side).map(<[u8]>::to_vec).collect();
        let boxes: Vec<[f32; 4]> = j["boxes_in"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| [2, 3, 4, 5].map(|k| b[k].as_f64().unwrap() as f32))
            .collect();
        let s = &j["scale_ratio"];
        let scale = (s[0].as_f64().unwrap(), s[1].as_f64().unwrap());
        let got = polygon::outlines(&boxes, &masks, side, scale);
        let want: Vec<Vec<[f32; 2]>> = j["polygons"]
            .as_array()
            .unwrap()
            .iter()
            .map(pts_f)
            .collect();
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            total += 1;
            let rect = boxes[i];
            let is_rect = w.len() == 4 && w[0] == [rect[0], rect[1]] && w[2] == [rect[2], rect[3]];
            quads += usize::from(!is_rect);
            if g == w {
                same += 1;
                continue;
            }
            // name the first stage that parts
            let st = &j["stages"][i];
            let mut why = String::from("normalize");
            if let Some(cs) = st.get("contours") {
                let [x0, y0, x1, y1] = boxes[i].map(|v| v as i32);
                let (bw, bh) = ((x1 - x0) as usize, (y1 - y0) as usize);
                let c = st["crop"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                let mut crop = Vec::new();
                for y in c[1]..c[3] {
                    crop.extend_from_slice(&masks[i][y * side + c[0]..y * side + c[2]]);
                }
                let resized = geom::resize_nearest(&crop, c[2] - c[0], c[3] - c[1], bw, bh);
                let ours = geom::external_contours(&resized, bw, bh);
                let theirs: Vec<Vec<geom::Pt>> = cs.as_array().unwrap().iter().map(pts_i).collect();
                if ours != theirs {
                    why = format!("contours ({} vs {})", ours.len(), theirs.len());
                } else if let Some(ap) = st.get("approx") {
                    let cnt = ours
                        .iter()
                        .max_by(|a, b| geom::contour_area(a).total_cmp(&geom::contour_area(b)))
                        .unwrap();
                    let a = geom::approx_poly_dp(cnt, 0.004 * geom::arc_length(cnt));
                    if a != pts_i(ap) {
                        why = format!("approx {a:?} vs {ap}");
                    } else if st.get("quad").is_some_and(|q| !q.is_null()) {
                        why = format!("custom/quad: {}", st["quad"]);
                    }
                }
            }
            if first_bad.len() < 8 {
                first_bad.push(format!("{name} #{i}: got {g:?} want {w:?} [{why}]"));
            }
        }
    }
    for b in &first_bad {
        eprintln!("{b}");
    }
    eprintln!(
        "{} pages: {same} of {total} outlines identical ({quads} not the plain box)",
        pages.len()
    );
    assert_eq!(same, total, "outlines differ from PaddleX on its own masks");
}
