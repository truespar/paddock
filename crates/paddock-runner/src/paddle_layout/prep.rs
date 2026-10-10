//! From layout boxes to what the recognizer reads - PaddleX's PaddleOCR-VL
//! 1.6 page preparation, step for step:
//!
//! 1. `filter_overlap_boxes`: "reference" boxes go, so does anything under
//!    6 px a side; of two boxes overlapping past 0.7 of the smaller the
//!    smaller goes (an inline formula past 0.5 goes either way), unless their
//!    outlines overlap less than that, or the pair is a picture-like box
//!    beside a different label (a table beside text is not spared);
//! 2. crops: `page[y1:y2, x1:x2]`, no padding, everything outside the
//!    region's outline painted white (`cv::fillPoly` of the outline's
//!    truncated corners - a plain box's outline covers its whole crop);
//! 3. `merge_blocks`: consecutive "text" blocks that sit side by side or
//!    stack aligned on one edge beside a picture or table become ONE crop,
//!    stacked on a white canvas - unless the stack would be 3x taller than
//!    wide;
//! 4. per-label task prompts; a formula crop is first trimmed to its ink
//!    (`crop_margin`).
//!
//! The crops are the page's own RGB - the reference's server path sends the
//! same pixels (as JPEG), its local path sends them channel-swapped.

use image::RgbImage;
use paddock_engine::gpu_model::doclayout::{LayoutBox, geom, polygon::Polygon};

/// Labels kept as pictures, never read (PaddleX `image_labels`, chart and
/// seal recognition off as in 1.6).
pub const IMAGE_LABELS: [&str; 5] = ["image", "header_image", "footer_image", "chart", "seal"];

/// One block of the page in reading order.
pub struct Block {
    pub label: &'static str,
    pub bbox: [i32; 4],
    /// what the recognizer reads: None for a picture block's twin that was
    /// merged into the group's first crop
    pub img: Option<RgbImage>,
    /// a merged group's first block index (None outside a group)
    pub group: Option<usize>,
    /// the region's outline in page pixels, when the layout read its mask
    pub polygon: Option<Polygon>,
}

fn to_f64(p: &Polygon) -> Vec<[f64; 2]> {
    p.iter().map(|q| [q[0] as f64, q[1] as f64]).collect()
}

/// `filter_overlap_boxes` (the outline check where the boxes carry them).
pub fn filter_overlap(boxes: &[LayoutBox]) -> Vec<LayoutBox> {
    let boxes: Vec<&LayoutBox> = boxes.iter().filter(|b| b.label() != "reference").collect();
    let n = boxes.len();
    let f = |b: &LayoutBox| b.bbox.map(f64::from);
    let area = |b: &LayoutBox| {
        let c = f(b);
        ((c[2] - c[0]) * (c[3] - c[1])).abs()
    };
    let overlap = |a: &LayoutBox, b: &LayoutBox| {
        let (p, q) = (f(a), f(b));
        let iw = (p[2].min(q[2]) - p[0].max(q[0])).max(0.0);
        let ih = (p[3].min(q[3]) - p[1].max(q[1])).max(0.0);
        let small = area(a).min(area(b));
        if small > 0.0 { iw * ih / small } else { 0.0 }
    };
    let pictures = ["image", "table", "seal", "chart"];
    let mut dropped = vec![false; n];
    for i in 0..n {
        let c = f(boxes[i]);
        if c[2] - c[0] < 6.0 || c[3] - c[1] < 6.0 {
            dropped[i] = true;
        }
        for j in i + 1..n {
            if dropped[i] || dropped[j] {
                continue;
            }
            let r = overlap(boxes[i], boxes[j]);
            let (li, lj) = (boxes[i].label(), boxes[j].label());
            if (li == "inline_formula" || lj == "inline_formula") && r > 0.5 {
                dropped[i] |= li == "inline_formula";
                dropped[j] |= lj == "inline_formula";
                continue;
            }
            if r > 0.7 {
                if let (Some(pi), Some(pj)) = (&boxes[i].polygon, &boxes[j].polygon)
                    && geom::overlap(&to_f64(pi), &to_f64(pj), true) < 0.7
                {
                    continue;
                }
                if li != lj && (pictures.contains(&li) || pictures.contains(&lj)) {
                    let has_table = li == "table" || lj == "table";
                    if !has_table || (pictures.contains(&li) && pictures.contains(&lj)) {
                        continue;
                    }
                }
                if area(boxes[i]) >= area(boxes[j]) {
                    dropped[j] = true;
                } else {
                    dropped[i] = true;
                }
            }
        }
    }
    boxes
        .into_iter()
        .zip(dropped)
        .filter(|&(_, d)| !d)
        .map(|(b, _)| b.clone())
        .collect()
}

/// The crop of one box (coordinates already clipped to the page).
pub fn crop(page: &RgbImage, b: [i32; 4]) -> RgbImage {
    let [x0, y0, x1, y1] = b.map(|v| v.max(0) as u32);
    let (x1, y1) = (x1.min(page.width()), y1.min(page.height()));
    image::imageops::crop_imm(page, x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0)).to_image()
}

/// `CropByBoxes` with an outline: the crop, white outside the outline.
pub fn crop_outline(page: &RgbImage, b: [i32; 4], outline: Option<&Polygon>) -> RgbImage {
    let mut img = crop(page, b);
    let Some(outline) = outline else { return img };
    let (w, h) = (img.width() as usize, img.height() as usize);
    // `np.array(points, dtype=np.int32)` truncates, then the crop's origin
    let pts: Vec<geom::Pt> = outline
        .iter()
        .map(|p| [p[0] as i32 - b[0], p[1] as i32 - b[1]])
        .collect();
    let mut inside = vec![0u8; w * h];
    geom::fill_poly(&mut inside, w, h, &pts, 1);
    for (px, &keep) in img.pixels_mut().zip(&inside) {
        if keep == 0 {
            *px = image::Rgb([255, 255, 255]);
        }
    }
    img
}

/// Every box as a block with its crop.
pub fn blocks(page: &RgbImage, boxes: &[LayoutBox]) -> Vec<Block> {
    boxes
        .iter()
        .map(|b| Block {
            label: b.label(),
            bbox: b.bbox,
            img: Some(crop_outline(page, b.bbox, b.polygon.as_ref())),
            group: None,
            polygon: b.polygon.clone(),
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Align {
    Left,
    Right,
    Center,
}

/// `merge_images`: stacked top to bottom on a white canvas as wide as the
/// widest, each step aligning the stack so far and the next crop.
fn stack(imgs: &[&RgbImage], aligns: &[Align]) -> RgbImage {
    let mut x = vec![0u32; imgs.len()];
    let mut width = imgs[0].width();
    for i in 1..imgs.len() {
        let w2 = imgs[i].width();
        let step = width.max(w2);
        let (a, b) = match aligns[i - 1] {
            Align::Center => ((step - width) / 2, (step - w2) / 2),
            Align::Right => (step - width, step - w2),
            Align::Left => (0, 0),
        };
        for xk in x.iter_mut().take(i) {
            *xk += a;
        }
        x[i] = b;
        width = step;
    }
    let height: u32 = imgs.iter().map(|i| i.height()).sum();
    let mut canvas = RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
    let mut y = 0;
    for (img, &xo) in imgs.iter().zip(&x) {
        image::imageops::replace(&mut canvas, *img, i64::from(xo), i64::from(y));
        y += img.height();
    }
    canvas
}

/// `merge_blocks` with the pipeline's non-merge labels (pictures + tables).
pub fn merge(blocks: Vec<Block>) -> Vec<Block> {
    let non_merge = |l: &str| IMAGE_LABELS.contains(&l) || l == "table";
    let mergeable: Vec<usize> = (0..blocks.len())
        .filter(|&i| !non_merge(blocks[i].label))
        .collect();
    let aligned = |a: i32, b: i32| (a - b).abs() <= 5;
    // the union of two boxes touches a picture or table
    let beside_other = |i: usize, p: usize| {
        let (a, b) = (blocks[p].bbox, blocks[i].bbox);
        let u = [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ];
        blocks.iter().enumerate().any(|(k, o)| {
            k != i && k != p && non_merge(o.label) && {
                let iw = (u[2].min(o.bbox[2]) - u[0].max(o.bbox[0])).max(0);
                let ih = (u[3].min(o.bbox[3]) - u[1].max(o.bbox[1])).max(0);
                iw * ih > 0
            }
        })
    };
    let mut groups: Vec<(Vec<usize>, Vec<Align>)> = Vec::new();
    for (k, &i) in mergeable.iter().enumerate() {
        if k == 0 {
            groups.push((vec![i], Vec::new()));
            continue;
        }
        let p = mergeable[k - 1];
        let (b, pb) = (blocks[i].bbox, blocks[p].bbox);
        let text = blocks[i].label == "text" && blocks[p].label == "text";
        // horizontal projection IoU > 0 is just a positive overlap
        let h_overlap = b[2].min(pb[2]) - b[0].max(pb[0]) > 0;
        let cross = !h_overlap
            && text
            && b[0] > pb[2]
            && b[1] < pb[3]
            && f64::from(b[0] - pb[2]) < f64::from((pb[2] - pb[0]).max(b[2] - b[0])) * 0.3;
        let updown = h_overlap
            && text
            && b[3] >= pb[1]
            && f64::from((b[1] - pb[3]).abs()) < f64::from((pb[3] - pb[1]).max(b[3] - b[1])) * 0.5
            && (aligned(b[0], pb[0]) ^ aligned(b[2], pb[2]))
            && beside_other(i, p);
        let align = if cross {
            Some(Align::Center)
        } else if updown {
            Some(if aligned(b[0], pb[0]) {
                Align::Left
            } else if aligned(b[2], pb[2]) {
                Align::Right
            } else {
                Align::Center
            })
        } else {
            None
        };
        match align {
            Some(a) => {
                let g = groups.last_mut().expect("a group is open");
                g.0.push(i);
                g.1.push(a);
            }
            None => groups.push((vec![i], Vec::new())),
        }
    }
    let mut blocks: Vec<Option<Block>> = blocks.into_iter().map(Some).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < blocks.len() {
        if let Some((members, aligns)) = groups
            .iter()
            .find(|(m, _)| m[0] == i && blocks[m[0]].is_some())
        {
            let (start, end) = (members[0], *members.last().expect("non-empty"));
            let imgs: Vec<&RgbImage> = members
                .iter()
                .map(|&m| {
                    blocks[m]
                        .as_ref()
                        .and_then(|b| b.img.as_ref())
                        .expect("cropped")
                })
                .collect();
            let w = imgs.iter().map(|i| i.width()).max().unwrap_or(0);
            let h: u32 = imgs.iter().map(|i| i.height()).sum();
            let tall = w == 0 || f64::from(h) / f64::from(w) >= 3.0;
            let merged = (!tall && members.len() > 1).then(|| stack(&imgs, aligns));
            for (k, &m) in members.iter().enumerate() {
                let mut b = blocks[m].take().expect("unused");
                if let Some(img) = &merged {
                    b.img = (k == 0).then(|| img.clone());
                    b.group = Some(start);
                }
                out.push(b);
            }
            // pictures and tables between the group's members follow it
            out.extend(
                blocks
                    .iter_mut()
                    .take(end)
                    .skip(start + 1)
                    .filter_map(Option::take),
            );
            i = end + 1;
            continue;
        }
        if blocks[i].as_ref().is_some_and(|b| non_merge(b.label)) {
            out.extend(blocks[i].take());
        }
        i += 1;
    }
    out
}

/// The task prompt a block is read with; None = kept as a picture.
pub fn prompt(label: &str) -> Option<&'static str> {
    if IMAGE_LABELS.contains(&label) {
        None
    } else if label == "table" {
        Some("Table Recognition:")
    } else if label.contains("formula") && label != "formula_number" {
        Some("Formula Recognition:")
    } else {
        Some("OCR:")
    }
}

/// `crop_margin`: a formula crop trimmed to the bounding box of its ink -
/// OpenCV's fixed-point grey, stretched to 0..255 (truncating, as the
/// reference's LUT does), every pixel at 200 or darker counted.
pub fn crop_margin(img: &RgbImage) -> RgbImage {
    let grey: Vec<u8> = img
        .pixels()
        .map(|p| {
            let [r, g, b] = p.0.map(u32::from);
            ((b * 1868 + g * 9617 + r * 4899 + (1 << 13)) >> 14) as u8
        })
        .collect();
    let (Some(&lo), Some(&hi)) = (grey.iter().min(), grey.iter().max()) else {
        return img.clone();
    };
    if lo == hi {
        return img.clone();
    }
    let span = f64::from(hi - lo);
    let w = img.width() as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (k, &v) in grey.iter().enumerate() {
        let s = (f64::from(v - lo) / span * 255.0) as u32;
        if s <= 200 {
            let (x, y) = (k % w, k / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x0 == usize::MAX {
        return img.clone();
    }
    image::imageops::crop_imm(
        img,
        x0 as u32,
        y0 as u32,
        (x1 - x0 + 1) as u32,
        (y1 - y0 + 1) as u32,
    )
    .to_image()
}

/// What one block is sent as: its crop (a formula's trimmed to its ink when
/// that leaves more than 2 px a side) and its prompt.
pub fn request(b: &Block) -> Option<(RgbImage, &'static str)> {
    let p = prompt(b.label)?;
    let img = b.img.as_ref()?;
    if p == "Formula Recognition:" {
        let t = crop_margin(img);
        if t.width() > 2 && t.height() > 2 {
            return Some((t, p));
        }
    }
    Some((img.clone(), p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lb(cls: usize, bbox: [i32; 4]) -> LayoutBox {
        LayoutBox {
            cls,
            score: 0.9,
            bbox,
            order: None,
            query: 0,
            polygon: None,
        }
    }

    #[test]
    fn overlap_filter_keeps_the_larger_and_spares_pictures() {
        // text 22, image 14, table 21, inline_formula 15, reference 18
        let b = [
            lb(22, [0, 0, 100, 100]),
            lb(22, [10, 10, 90, 90]),
            lb(14, [0, 200, 100, 300]),
            lb(22, [5, 205, 95, 295]),
            lb(18, [0, 400, 100, 500]),
            lb(15, [0, 600, 50, 650]),
            lb(22, [0, 600, 100, 700]),
            lb(22, [0, 800, 4, 900]),
        ];
        let kept: Vec<[i32; 4]> = filter_overlap(&b).iter().map(|b| b.bbox).collect();
        assert_eq!(
            kept,
            vec![
                [0, 0, 100, 100],
                [0, 200, 100, 300],
                [5, 205, 95, 295],
                [0, 600, 100, 700]
            ]
        );
    }

    #[test]
    fn prompts_follow_the_label() {
        assert_eq!(prompt("table"), Some("Table Recognition:"));
        assert_eq!(prompt("display_formula"), Some("Formula Recognition:"));
        assert_eq!(prompt("formula_number"), Some("OCR:"));
        assert_eq!(prompt("chart"), None);
        assert_eq!(prompt("paragraph_title"), Some("OCR:"));
    }

    #[test]
    fn crop_margin_finds_the_ink() {
        let mut img = RgbImage::from_pixel(20, 10, image::Rgb([255, 255, 255]));
        for x in 5..9 {
            for y in 3..6 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let t = crop_margin(&img);
        assert_eq!((t.width(), t.height()), (4, 3));
    }

    #[test]
    fn side_by_side_text_merges_into_one_crop() {
        let page = RgbImage::from_pixel(400, 100, image::Rgb([255, 255, 255]));
        let b = [
            lb(22, [0, 0, 100, 50]),
            lb(22, [110, 0, 210, 50]),
            lb(17, [0, 60, 100, 90]),
        ];
        let merged = merge(blocks(&page, &b));
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].group, Some(0));
        let img = merged[0].img.as_ref().unwrap();
        assert_eq!((img.width(), img.height()), (100, 100));
        assert!(merged[1].img.is_none());
        assert!(merged[2].group.is_none());
    }
}
