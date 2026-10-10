//! The detector's postprocess - PaddleX's, step for step: the exported
//! graph's own tail, then `LayoutAnalysisProcess.apply` at the PaddleOCR-VL
//! 1.6 settings (threshold 0.3, NMS on, unclip 1.0, the per-class merge
//! table), then the order numbering. Host-side selection over at most 7,500
//! (query, class) pairs - the same class of work as a sampler.
//!
//! - Reading order: votes ascending, each query's rank its `order_seq`.
//! - Candidates: sigmoid scores over all 300 x 25 pairs, the top 300 - one
//!   query can come back under two labels (NMS then folds the twin: the same
//!   box has IoU 1).
//! - Boxes: (cx, cy, w, h) -> corners, times the page size, in f32 as the
//!   graph runs it; then `np.round` (half to even).
//! - Filter `score > 0.3`; class-aware greedy NMS (an IoU of 0.6 or more
//!   within a class, 0.98 across classes, the +1 pixel area convention);
//!   drop "image" boxes covering most of the page (0.82 of a landscape
//!   page, 0.93 otherwise) unless that would drop everything; remove every
//!   box 90 % inside a chart, display_formula, doc_title, inline_formula or
//!   paragraph_title box; sort by `order_seq`; clip by truncation.
//!
//! Arithmetic follows NumPy 2's float32 promotion (the boxes are a float32
//! array, so areas and IoUs are float32); every area is an integer under
//! 2^24, so only the final divide rounds.

use super::DecoderOut;

/// The checkpoint's 25 classes, by id (`inference.yml`'s `label_list` - the
/// names the pipeline keys on; HF's `id2label` folds several together).
pub const LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

const CLASSES: usize = LABELS.len();
const THRESHOLD: f32 = 0.3;
const IMAGE: usize = 14;
/// the classes whose boxes swallow what they contain ("large" merge mode)
const LARGE: [usize; 5] = [3, 5, 6, 15, 17];
/// labels the 1-based `order` numbering skips (PaddleX SKIP_ORDER_LABELS)
const SKIP_ORDER: [&str; 11] = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
];

/// One detected region, in page pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct LayoutBox {
    pub cls: usize,
    pub score: f32,
    /// x1, y1, x2, y2 - clipped to the page, x2 / y2 exclusive
    pub bbox: [i32; 4],
    /// 1-based reading position among the numbered labels
    pub order: Option<u32>,
    /// the decoder query it came from (its mask row)
    pub query: usize,
    /// its outline in page pixels (the reference's "auto" shape), when the
    /// masks were read; None = the box itself
    pub polygon: Option<super::polygon::Polygon>,
}

impl LayoutBox {
    pub fn label(&self) -> &'static str {
        LABELS[self.cls]
    }
}

/// A candidate between the steps: the graph's 7-column row.
#[derive(Clone, Copy)]
pub(super) struct Cand {
    cls: usize,
    score: f32,
    b: [f32; 4],
    seq: u32,
    query: usize,
}

/// `order_seq`: each query's rank by ascending votes (ties by index - the
/// graph's argsort is unstable, so any tie order is the reference's).
pub fn order_seq(votes: &[f32]) -> Vec<u32> {
    let mut ptr: Vec<usize> = (0..votes.len()).collect();
    ptr.sort_by(|&a, &b| votes[a].total_cmp(&votes[b]).then(a.cmp(&b)));
    let mut seq = vec![0u32; votes.len()];
    for (r, &q) in ptr.iter().enumerate() {
        seq[q] = r as u32;
    }
    seq
}

/// IoU with the +1 pixel convention.
fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0]) + 1.0).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1]) + 1.0).max(0.0);
    let inter = iw * ih;
    let aa = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0);
    let ab = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
    inter / (aa + ab - inter)
}

/// `a` lies at least 90 % inside `b` (plain areas, no +1).
fn contained(a: &[f32; 4], b: &[f32; 4]) -> bool {
    let area = (a[2] - a[0]) * (a[3] - a[1]);
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let r = if area > 0.0 { iw * ih / area } else { 0.0 };
    r >= 0.9
}

/// The page's regions in reading order from one forward, as rectangles.
pub fn postprocess(dec: &DecoderOut, wh: (usize, usize)) -> Vec<LayoutBox> {
    finish(select(dec, wh), None, wh)
}

/// The surviving candidates in reading order, before clipping - where the
/// reference cuts the outlines (it does so before it clips, and a box the
/// clip then drops still served as the next one's predecessor).
pub(super) fn select(dec: &DecoderOut, (w, h): (usize, usize)) -> Vec<Cand> {
    let seq = order_seq(&dec.votes);
    let (fw, fh) = (w as f32, h as f32);
    // class-wise top 300 over the (query, class) pairs
    let scores: Vec<f32> = dec
        .logits
        .iter()
        .map(|&l| 1.0 / (1.0 + (-l).exp()))
        .collect();
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    idx.truncate(super::QUERIES);
    let mut cands: Vec<Cand> = idx
        .iter()
        .map(|&i| {
            let q = i / CLASSES;
            let bx = &dec.boxes[q * 4..q * 4 + 4];
            let (hw, hh) = (bx[2] * 0.5, bx[3] * 0.5);
            let b = [
                ((bx[0] - hw) * fw).round_ties_even(),
                ((bx[1] - hh) * fh).round_ties_even(),
                ((bx[0] + hw) * fw).round_ties_even(),
                ((bx[1] + hh) * fh).round_ties_even(),
            ];
            Cand {
                cls: i % CLASSES,
                score: scores[i],
                b,
                seq: seq[q],
                query: q,
            }
        })
        .filter(|c| c.score > THRESHOLD)
        .collect();
    // NMS, greedy by descending score (`argsort(scores)[::-1]`: equal scores
    // come later-first)
    let mut order: Vec<usize> = (0..cands.len()).collect();
    order.sort_by(|&a, &b| cands[b].score.total_cmp(&cands[a].score).then(b.cmp(&a)));
    let mut kept = Vec::new();
    while let Some((&cur, rest)) = order.split_first() {
        kept.push(cur);
        let c = cands[cur];
        order = rest
            .iter()
            .copied()
            .filter(|&i| {
                let t = if cands[i].cls == c.cls {
                    0.6f32
                } else {
                    0.98f32
                };
                iou(&c.b, &cands[i].b) < t
            })
            .collect();
    }
    cands = kept.iter().map(|&i| cands[i]).collect();
    // page-sized pictures are backgrounds, not regions
    if cands.len() > 1 {
        let thres = if w > h { 0.82 } else { 0.93 };
        let page = (w * h) as f64;
        let small: Vec<Cand> = cands
            .iter()
            .copied()
            .filter(|c| {
                if c.cls != IMAGE {
                    return true;
                }
                let (x0, y0) = (c.b[0].max(0.0), c.b[1].max(0.0));
                let (x1, y1) = (c.b[2].min(fw), c.b[3].min(fh));
                (((x1 - x0) * (y1 - y0)) as f64) <= thres * page
            })
            .collect();
        if !small.is_empty() {
            cands = small;
        }
    }
    // "large" merge: one keep mask over the pre-merge set
    let keep: Vec<bool> = (0..cands.len())
        .map(|i| {
            !cands
                .iter()
                .enumerate()
                .any(|(j, o)| j != i && LARGE.contains(&o.cls) && contained(&cands[i].b, &o.b))
        })
        .collect();
    let mut cands: Vec<Cand> = cands
        .into_iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(c))
        .collect();
    cands.sort_by_key(|c| c.seq);
    cands
}

impl Cand {
    /// the corners, rounded, before clipping
    pub(super) fn corners(&self) -> [f32; 4] {
        self.b
    }

    pub(super) fn query(&self) -> usize {
        self.query
    }
}

/// Clip (by truncation), drop the degenerate, number - with each box's
/// outline when the masks were read.
pub(super) fn finish(
    cands: Vec<Cand>,
    outlines: Option<Vec<super::polygon::Polygon>>,
    (w, h): (usize, usize),
) -> Vec<LayoutBox> {
    let (fw, fh) = (w as f32, h as f32);
    let mut outlines = outlines.map(Vec::into_iter);
    let mut n = 0;
    cands
        .iter()
        .filter_map(|c| {
            let polygon = outlines.as_mut().and_then(Iterator::next);
            let x0 = c.b[0].max(0.0) as i32;
            let y0 = c.b[1].max(0.0) as i32;
            let x1 = c.b[2].min(fw) as i32;
            let y1 = c.b[3].min(fh) as i32;
            if x1 <= x0 || y1 <= y0 {
                return None;
            }
            let order = (!SKIP_ORDER.contains(&LABELS[c.cls])).then(|| {
                n += 1;
                n
            });
            Some(LayoutBox {
                cls: c.cls,
                score: c.score,
                bbox: [x0, y0, x1, y1],
                order,
                query: c.query,
                polygon,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_seq_ranks_by_ascending_votes() {
        assert_eq!(order_seq(&[2.0, 0.5, 1.0]), vec![2, 0, 1]);
    }

    #[test]
    fn iou_uses_the_plus_one_convention() {
        let a = [0.0, 0.0, 9.0, 9.0];
        assert_eq!(iou(&a, &a), 1.0);
        // 10 x 10 vs its right half: 50 / 100
        assert_eq!(iou(&a, &[5.0, 0.0, 9.0, 9.0]), 0.5);
        assert!(contained(&[1.0, 1.0, 5.0, 5.0], &a));
        assert!(!contained(&a, &[1.0, 1.0, 5.0, 5.0]));
    }
}
