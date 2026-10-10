//! Each region's outline - PaddleX's `layout_shape_mode = "auto"`, the
//! PaddleOCR-VL pipeline's default: the detection's mask (`sigmoid > 0.5`,
//! 200 x 200 over the 800 x 800 input) is cut to the box, stretched to the
//! box's pixels, traced, simplified and kept as
//!
//! - the box itself, when the mask's minimum-area rectangle covers it
//!   (IoU >= 0.95) - most regions;
//! - that rectangle (a quad, possibly tilted), when the outline fits it
//!   (IoU >= 0.8) and the previous region's outline does not touch this box;
//! - else the outline itself.
//!
//! The arithmetic follows the reference's types through NumPy 2's promotion
//! rules (`extract_polygon_points_by_masks`, `mask2polygon`,
//! `extract_custom_vertices`, `_normalize_layout_polygon`,
//! `convert_polygon_to_quad`), on the OpenCV routines `geom` ports. The
//! upstream function reads the widest box as `max(x2 - y1)` - an indexing
//! slip kept as is, since it decides how densely long edges are resampled.

use super::geom;

/// A region's outline in page pixels, clockwise from the top-left for a
/// rectangle or quad.
pub type Polygon = Vec<[f32; 2]>;

/// The vertex-spacing bound as NumPy holds it: a box width (an int, which
/// scales into a float64) or the widest-box term (a float32, which stays
/// one).
#[derive(Clone, Copy)]
enum Spacing {
    Int(i32),
    F32(f32),
}

fn rect_of(b: &[f32; 4]) -> Polygon {
    let [x0, y0, x1, y1] = b.map(|v| v as i32 as f32);
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn to_f64(p: &[[f32; 2]]) -> Vec<[f64; 2]> {
    p.iter().map(|q| [q[0] as f64, q[1] as f64]).collect()
}

fn norm(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// `extract_custom_vertices`: convex vertices kept, concave runs of two or
/// more kept where their angle is wide, long edges resampled; a vertex at
/// a ~45 degree convex corner pushed out along its bisector.
fn custom_vertices(poly: &[geom::Pt], spacing: Spacing) -> Vec<[f64; 2]> {
    let n = poly.len();
    let max_dist: f64 = match spacing {
        Spacing::Int(v) => v as f64 * 0.3,
        Spacing::F32(v) => (v * 0.3f32) as f64,
    };
    struct Info {
        convex: bool,
        angle: f64,
        v1: [f64; 2],
        v2: [f64; 2],
    }
    let p = |i: usize| poly[i].map(|v| v as f64);
    let info: Vec<Info> = (0..n)
        .map(|i| {
            let (pp, pc, pn) = (poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]);
            let a = [pc[0] - pp[0], pc[1] - pp[1]];
            let b = [pn[0] - pc[0], pn[1] - pc[1]];
            let convex = a[0] * b[1] - a[1] * b[0] < 0;
            let v1 = [(pp[0] - pc[0]) as f64, (pp[1] - pc[1]) as f64];
            let v2 = [(pn[0] - pc[0]) as f64, (pn[1] - pc[1]) as f64];
            let (n1, n2) = (norm(v1), norm(v2));
            let u1 = [v1[0] / n1, v1[1] / n1];
            let u2 = [v2[0] / n2, v2[1] / n2];
            let dot = (u1[0] * u2[0] + u1[1] * u2[1]).clamp(-1.0, 1.0);
            Info {
                convex,
                angle: dot.acos().to_degrees(),
                v1,
                v2,
            }
        })
        .collect();
    let concave: Vec<usize> = (0..n).filter(|&i| !info[i].convex).collect();
    let mut preserve: Vec<usize> = Vec::new();
    if let Some(&first) = concave.first() {
        let mut groups: Vec<usize> = Vec::new();
        let mut cur = vec![first];
        for k in 1..concave.len() {
            if concave[k] - concave[k - 1] == 1 || (concave[k - 1] == n - 1 && concave[k] == 0) {
                cur.push(concave[k]);
            } else {
                if cur.len() >= 2 {
                    groups.extend(&cur);
                }
                cur = vec![concave[k]];
            }
        }
        if cur.len() >= 2 {
            groups.extend(&cur);
        }
        if concave.len() >= 2 && concave[0] == 0 && concave[concave.len() - 1] == n - 1 {
            if groups.contains(&0) && groups.contains(&(n - 1)) {
                preserve.extend(groups);
            }
        } else {
            preserve.extend(groups);
        }
    }
    let kept: Vec<usize> = (0..n)
        .filter(|&i| info[i].convex || (preserve.contains(&i) && info[i].angle >= 120.0))
        .collect();
    let mut fin: Vec<usize> = Vec::new();
    for k in 0..kept.len() {
        let (cur, next) = (kept[k], kept[(k + 1) % kept.len()]);
        fin.push(cur);
        let (a, b) = (p(cur), p(next));
        let dist = norm([a[0] - b[0], a[1] - b[1]]);
        if dist > max_dist {
            let between: Vec<usize> = if next > cur {
                (cur + 1..next).collect()
            } else {
                (cur + 1..n).chain(0..next).collect()
            };
            if !between.is_empty() {
                let needed = (dist / max_dist).ceil() as usize - 1;
                if between.len() <= needed {
                    fin.extend(&between);
                } else {
                    let step = between.len() as f64 / needed as f64;
                    fin.extend((0..needed).map(|i| between[(i as f64 * step) as usize]));
                }
            }
        }
    }
    fin.sort_unstable();
    fin.dedup();
    fin.iter()
        .map(|&i| {
            let inf = &info[i];
            let pc = p(i);
            if inf.convex && (inf.angle - 45.0).abs() < 1.0 {
                let (n1, n2) = (norm(inf.v1), norm(inf.v2));
                let mut dir = [
                    inf.v1[0] / n1 + inf.v2[0] / n2,
                    inf.v1[1] / n1 + inf.v2[1] / n2,
                ];
                let nd = norm(dir);
                dir = [dir[0] / nd, dir[1] / nd];
                let d = (n1 + n2) / 2.0;
                [pc[0] + dir[0] * d, pc[1] + dir[1] * d]
            } else {
                pc
            }
        })
        .collect()
}

/// `convert_polygon_to_quad`: the minimum-area rectangle's corners, ordered
/// by angle about their mean and rolled to start at the smallest x + y.
fn to_quad(poly: &[[f32; 2]]) -> Option<Polygon> {
    if poly.len() < 3 {
        return None;
    }
    let q = geom::box_points(&geom::min_area_rect(poly));
    let mut c = q[0];
    for p in &q[1..] {
        c = [c[0] + p[0], c[1] + p[1]];
    }
    c = [c[0] / 4.0, c[1] / 4.0];
    let ang: Vec<f32> = q.iter().map(|p| (p[1] - c[1]).atan2(p[0] - c[0])).collect();
    let mut idx: Vec<usize> = (0..4).collect();
    idx.sort_by(|&a, &b| {
        ang[a]
            .partial_cmp(&ang[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let q: Vec<[f32; 2]> = idx.iter().map(|&i| q[i]).collect();
    let sums: Vec<f32> = q.iter().map(|p| p[0] + p[1]).collect();
    let mut tl = 0;
    for i in 1..4 {
        if sums[i] < sums[tl] {
            tl = i;
        }
    }
    Some((0..4).map(|k| q[(k + tl) % 4]).collect())
}

/// `_normalize_layout_polygon` in "auto" mode.
fn normalize(b: &[f32; 4], poly: Option<Vec<[f64; 2]>>, prev: Option<&Polygon>) -> Polygon {
    let rect = rect_of(b);
    let Some(poly) = poly else { return rect };
    let poly: Polygon = poly.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
    if poly.len() < 4 {
        return rect;
    }
    if let Some(quad) = to_quad(&poly) {
        let (r, q) = (to_f64(&rect), to_f64(&quad));
        if geom::overlap(&r, &q, false) >= 0.95 {
            return rect;
        }
        let iou_poly = geom::overlap(&to_f64(&poly), &q, false);
        let iou_prev = prev.map_or(0.0, |p| geom::overlap(&to_f64(p), &r, true));
        if iou_poly >= 0.8 && iou_prev < 0.01 {
            return quad;
        }
    }
    poly
}

/// Every box's outline (`extract_polygon_points_by_masks`): `boxes` the
/// postprocessed corners in page pixels (rounded, before clipping), `masks`
/// each box's 0/1 mask, `scale` the page-to-input ratios (800 / W, 800 / H).
pub fn outlines(
    boxes: &[[f32; 4]],
    masks: &[Vec<u8>],
    side: usize,
    scale: (f64, f64),
) -> Vec<Polygon> {
    let (sw, sh) = (scale.0 / 4.0, scale.1 / 4.0);
    // the reference's `max(boxes[:, 4] - boxes[:, 3])`: x2 - y1
    let widest = boxes
        .iter()
        .map(|b| b[2] - b[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<Polygon> = Vec::with_capacity(boxes.len());
    for (b, mask) in boxes.iter().zip(masks) {
        let [x0, y0, x1, y1] = b.map(|v| v as i32);
        let (bw, bh) = (x1 - x0, y1 - y0);
        let poly = (|| {
            if bw <= 0 || bh <= 0 {
                return Err(());
            }
            let cut = |a: i32, s: f64| {
                ((a as f64 * s).round_ties_even() as i64).clamp(0, side as i64) as usize
            };
            let (cx0, cx1, cy0, cy1) = (cut(x0, sw), cut(x1, sw), cut(y0, sh), cut(y1, sh));
            if cx1 <= cx0 || cy1 <= cy0 {
                return Err(());
            }
            let (cw, ch) = (cx1 - cx0, cy1 - cy0);
            let mut crop = Vec::with_capacity(cw * ch);
            for y in cy0..cy1 {
                crop.extend_from_slice(&mask[y * side + cx0..y * side + cx1]);
            }
            if crop.iter().all(|&v| v == 0) {
                return Err(());
            }
            let (bw, bh) = (bw as usize, bh as usize);
            let resized = geom::resize_nearest(&crop, cw, ch, bw, bh);
            let spacing = if bw as f64 > (widest * 0.6f32) as f64 {
                Spacing::Int(bw as i32)
            } else {
                Spacing::F32(widest)
            };
            // `mask2polygon`: the largest outer contour, simplified
            let cs = geom::external_contours(&resized, bw, bh);
            let mut best: Option<(&Vec<geom::Pt>, f64)> = None;
            for c in &cs {
                let a = geom::contour_area(c);
                if best.is_none_or(|(_, m)| a > m) {
                    best = Some((c, a));
                }
            }
            let Some((cnt, _)) = best else {
                return Ok(None);
            };
            let approx = geom::approx_poly_dp(cnt, 0.004 * geom::arc_length(cnt));
            let pts = custom_vertices(&approx, spacing);
            Ok(Some(
                pts.into_iter()
                    .map(|p| [p[0] + x0 as f64, p[1] + y0 as f64])
                    .collect::<Vec<_>>(),
            ))
        })();
        let polygon = match poly {
            Err(()) => rect_of(b),
            Ok(p) => normalize(b, p.filter(|v: &Vec<[f64; 2]>| !v.is_empty()), out.last()),
        };
        out.push(polygon);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_mask_keeps_the_box() {
        // the mask covers the box exactly: its rectangle is the box
        let side = 200;
        let mut m = vec![0u8; side * side];
        for y in 20..40 {
            for x in 10..90 {
                m[y * side + x] = 1;
            }
        }
        let b = [40.0, 80.0, 360.0, 160.0];
        let p = outlines(&[b], &[m], side, (1.0, 1.0));
        assert_eq!(p[0], rect_of(&b));
    }

    #[test]
    fn an_empty_mask_keeps_the_box() {
        let side = 200;
        let b = [40.0, 80.0, 360.0, 160.0];
        let p = outlines(&[b], &[vec![0u8; side * side]], side, (1.0, 1.0));
        assert_eq!(p[0], rect_of(&b));
    }
}
