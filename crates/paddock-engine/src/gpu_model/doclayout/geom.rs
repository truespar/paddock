//! The geometry PaddleX's layout pipeline borrows from OpenCV 4.10 and
//! Shapely, ported routine by routine so the polygons it cuts from the masks,
//! and the crops it whites out around them, come out the same: host-side
//! work on a few 200 x 200 masks and a dozen polygons a page.
//!
//! - `resize_nearest`: `cv::resize(INTER_NEAREST)` (`resizeNN`: source index
//!   `floor(x / (dst / src))`, clamped);
//! - `external_contours`: `cv::findContours(RETR_EXTERNAL,
//!   CHAIN_APPROX_SIMPLE)`, the 4.10 scanner (`contours_new.cpp`): the
//!   one-pixel zero border, the raster scan, Suzuki-Abe border following
//!   with the inline simple-chain compression, contours returned newest
//!   first (the root's child list is built by prepending);
//! - `contour_area`, `arc_length`, `approx_poly_dp`: `shapedescr.cpp` /
//!   `approx.cpp` (the length's per-segment root is a float32 one);
//! - `convex_hull`, `min_area_rect`, `box_points`: Sklansky's hull, the
//!   rotating calipers and `RotatedRect::points`, in float32 as there;
//! - `fill_poly`: `cv::fillPoly` with LINE_8 and no shift - every edge drawn
//!   with the 8-connected line iterator (clipped to the image), then the
//!   edge-table scan fill;
//! - `overlap`: Shapely's polygon intersection over union / over the smaller
//!   area, by clipping against a convex polygon (a non-convex pair is cut
//!   into triangles first). Self-intersecting input is taken as given, where
//!   Shapely would first repair it with `buffer(0)`.

/// An integer point (x, y).
pub type Pt = [i32; 2];

// ---------------------------------------------------------------- resize

/// `cv::resize(src, (dw, dh), INTER_NEAREST)` on one u8 channel.
pub fn resize_nearest(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let ofs = |d: usize, s: usize| -> Vec<usize> {
        let ifx = 1.0 / (d as f64 / s as f64);
        (0..d)
            .map(|x| ((x as f64 * ifx).floor() as usize).min(s - 1))
            .collect()
    };
    let (xo, yo) = (ofs(dw, sw), ofs(dh, sh));
    let mut out = Vec::with_capacity(dw * dh);
    for &sy in &yo {
        let row = &src[sy * sw..(sy + 1) * sw];
        out.extend(xo.iter().map(|&sx| row[sx]));
    }
    out
}

// ---------------------------------------------------------------- contours

const DELTAS: [[i32; 2]; 8] = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
];
const MASK8_RIGHT: i8 = -128; // 0x80
const MASK8_NEW: i8 = 2;
const MASK8_FLAGS: i8 = -2; // 0xFE

struct Scanner {
    img: Vec<i8>,
    step: i32,
    w: i32,
    h: i32,
    pt: [i32; 2],
    lnbd: [i32; 2],
    contours: Vec<Vec<Pt>>,
}

impl Scanner {
    fn at(&self, x: i32, y: i32) -> i8 {
        self.img[(y * self.step + x) as usize]
    }

    fn delta(&self, s: i8) -> i32 {
        let d = DELTAS[(s % 8) as usize];
        d[0] + d[1] * self.step
    }

    /// `icvFetchContourEx<schar>` for an outer border, simple chain.
    fn fetch(&mut self, start: [i32; 2]) -> Vec<Pt> {
        let i0 = start[1] * self.step + start[0];
        let mut pt = [start[0] - 1, start[1] - 1];
        let mut pts = Vec::new();
        let mut s_end: i8 = 4;
        let mut s: i8 = s_end;
        let mut i1;
        loop {
            s = (s - 1) & 7;
            i1 = i0 + self.delta(s);
            if self.img[i1 as usize] != 0 || s == s_end {
                break;
            }
        }
        if s == s_end {
            self.img[i0 as usize] = MASK8_NEW | MASK8_RIGHT;
            pts.push(pt);
            return pts;
        }
        let mut i3 = i0;
        let mut i4;
        let mut prev_s = s ^ 4;
        loop {
            s_end = s;
            s = s.min(15);
            loop {
                s += 1;
                i4 = i3 + self.delta(s);
                if self.img[i4 as usize] != 0 || s >= 15 {
                    break;
                }
            }
            s &= 7;
            if ((s as i32 - 1) as u32) < (s_end as i32 as u32) {
                self.img[i3 as usize] = MASK8_NEW | MASK8_RIGHT;
            } else if self.img[i3 as usize] == 1 {
                self.img[i3 as usize] = MASK8_NEW;
            }
            if s != prev_s {
                pts.push(pt);
            }
            prev_s = s;
            pt[0] += DELTAS[s as usize][0];
            pt[1] += DELTAS[s as usize][1];
            if i4 == i0 && i3 == i1 {
                break;
            }
            i3 = i4;
            s = (s + 4) & 7;
        }
        pts
    }

    fn contour_scan(&mut self, prev: i8, p: i8, last: &mut [i32; 2], x: i32, y: i32) -> bool {
        let mut is_hole = false;
        if !(prev == 0 && p == 1) {
            if p != 0 || prev < 1 {
                return false;
            }
            if prev & MASK8_FLAGS != 0 {
                last[0] = x - 1;
            }
            is_hole = true;
        }
        // RETR_EXTERNAL: no holes, nothing inside an outer border
        if is_hole || self.at(last[0], last[1]) > 0 {
            return false;
        }
        last[0] = x;
        let c = self.fetch([x, y]);
        // the root's children are prepended: newest first
        self.contours.insert(0, c);
        self.pt = [x + 1, y];
        true
    }

    fn find_next(&mut self) -> bool {
        let (mut x, mut y) = (self.pt[0], self.pt[1]);
        let (width, height) = (self.w - 1, self.h - 1);
        let mut last = self.lnbd;
        let mut prev = self.at(x - 1, y);
        while y < height {
            let mut p = 0i8;
            while x < width {
                while x < width {
                    p = self.at(x, y);
                    if p != prev {
                        break;
                    }
                    x += 1;
                }
                if x >= width {
                    break;
                }
                if self.contour_scan(prev, p, &mut last, x, y) {
                    self.lnbd = last;
                    return true;
                }
                prev = p;
                if prev & MASK8_FLAGS != 0 {
                    last[0] = x;
                }
                x += 1;
            }
            let _ = p;
            last = [0, y + 1];
            x = 1;
            prev = 0;
            y += 1;
        }
        false
    }
}

/// `cv::findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`: every
/// outer border of the nonzero pixels, newest-found first.
pub fn external_contours(mask: &[u8], w: usize, h: usize) -> Vec<Vec<Pt>> {
    let (pw, ph) = (w + 2, h + 2);
    let mut img = vec![0i8; pw * ph];
    for y in 0..h {
        for x in 0..w {
            img[(y + 1) * pw + x + 1] = i8::from(mask[y * w + x] != 0);
        }
    }
    let mut sc = Scanner {
        img,
        step: pw as i32,
        w: pw as i32,
        h: ph as i32,
        pt: [1, 1],
        lnbd: [0, 1],
        contours: Vec::new(),
    };
    while sc.find_next() {}
    sc.contours
}

/// `cv::contourArea` (unoriented).
pub fn contour_area(c: &[Pt]) -> f64 {
    let mut a = 0f64;
    let mut prev = c[c.len() - 1].map(|v| v as f32);
    for p in c {
        let p = p.map(|v| v as f32);
        a += prev[0] as f64 * p[1] as f64 - prev[1] as f64 * p[0] as f64;
        prev = p;
    }
    (a * 0.5).abs()
}

/// `cv::arcLength(c, closed = true)`: each segment's length a float32 root.
pub fn arc_length(c: &[Pt]) -> f64 {
    if c.len() <= 1 {
        return 0.0;
    }
    let mut per = 0f64;
    let mut prev = c[c.len() - 1].map(|v| v as f32);
    for p in c {
        let p = p.map(|v| v as f32);
        let (dx, dy) = (p[0] - prev[0], p[1] - prev[1]);
        per += (dx * dx + dy * dy).sqrt() as f64;
        prev = p;
    }
    per
}

/// `cv::approxPolyDP(c, eps, closed = true)` over integer points.
pub fn approx_poly_dp(src: &[Pt], eps: f64) -> Vec<Pt> {
    let count0 = src.len();
    if count0 == 0 {
        return Vec::new();
    }
    let count = count0;
    let eps = eps * eps;
    let mut dst: Vec<Pt> = vec![[0, 0]; count0];
    let mut new_count = 0usize;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let read = |pos: &mut usize| -> Pt {
        let p = src[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        p
    };
    // 1. two approximately farthest points
    let mut pos = 0usize;
    let mut right_start = 0usize;
    let mut start_pt: Pt = [-1000000, -1000000];
    let mut le_eps = false;
    for _ in 0..3 {
        let mut max_dist = 0f64;
        pos = (pos + right_start) % count;
        start_pt = read(&mut pos);
        for j in 1..count {
            let pt = read(&mut pos);
            let dx = (pt[0] - start_pt[0]) as f64;
            let dy = (pt[1] - start_pt[1]) as f64;
            let dist = dx * dx + dy * dy;
            if dist > max_dist {
                max_dist = dist;
                right_start = j;
            }
        }
        le_eps = max_dist <= eps;
    }
    // 2. the stack
    if !le_eps {
        let slice_start = pos % count;
        let right_end = slice_start;
        right_start = (right_start + slice_start) % count;
        stack.push((right_start, right_end));
        stack.push((slice_start, right_start));
    } else {
        dst[new_count] = start_pt;
        new_count += 1;
    }
    // 3. the recursion
    while let Some((s0, s1)) = stack.pop() {
        let end_pt = src[s1];
        let mut pos = s0;
        let mut start_pt = read(&mut pos);
        let le;
        let mut rs = 0usize;
        if pos != s1 {
            let dx = (end_pt[0] - start_pt[0]) as f64;
            let dy = (end_pt[1] - start_pt[1]) as f64;
            let mut max_dist = 0f64;
            while pos != s1 {
                let pt = read(&mut pos);
                let dist = (((pt[1] - start_pt[1]) as f64) * dx
                    - ((pt[0] - start_pt[0]) as f64) * dy)
                    .abs();
                if dist > max_dist {
                    max_dist = dist;
                    rs = (pos + count - 1) % count;
                }
            }
            le = max_dist * max_dist <= eps * (dx * dx + dy * dy);
        } else {
            le = true;
            start_pt = src[s0];
        }
        if le {
            dst[new_count] = start_pt;
            new_count += 1;
        } else {
            stack.push((rs, s1));
            stack.push((s0, rs));
        }
    }
    // 4. drop near-collinear points
    let count = new_count;
    let read_dst = |dst: &[Pt], pos: &mut usize| -> Pt {
        let p = dst[*pos];
        *pos += 1;
        if *pos >= count {
            *pos = 0;
        }
        p
    };
    let mut pos = count - 1;
    let mut start_pt = read_dst(&dst, &mut pos);
    let mut wpos = pos;
    let mut pt = read_dst(&dst, &mut pos);
    let mut i = 0;
    while i < count && new_count > 2 {
        let end_pt = read_dst(&dst, &mut pos);
        let dx = (end_pt[0] - start_pt[0]) as f64;
        let dy = (end_pt[1] - start_pt[1]) as f64;
        let dist =
            (((pt[0] - start_pt[0]) as f64) * dy - ((pt[1] - start_pt[1]) as f64) * dx).abs();
        let sip = (pt[0] - start_pt[0]) * (end_pt[0] - pt[0])
            + (pt[1] - start_pt[1]) * (end_pt[1] - pt[1]);
        if dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx != 0.0 && dy != 0.0 && sip >= 0 {
            new_count -= 1;
            start_pt = end_pt;
            dst[wpos] = start_pt;
            wpos += 1;
            if wpos >= count {
                wpos = 0;
            }
            pt = read_dst(&dst, &mut pos);
            i += 2;
            continue;
        }
        start_pt = pt;
        dst[wpos] = start_pt;
        wpos += 1;
        if wpos >= count {
            wpos = 0;
        }
        pt = end_pt;
        i += 1;
    }
    dst.truncate(new_count);
    dst
}

// ---------------------------------------------------------------- min-area rect

fn sign_f32(v: f32) -> i32 {
    i32::from(v > 0.0) - i32::from(v < 0.0)
}

fn sign_f64(v: f64) -> i32 {
    i32::from(v > 0.0) - i32::from(v < 0.0)
}

/// Sklansky's scan over the x-sorted points from `start` toward `end`.
fn sklansky(
    a: &[[f32; 2]],
    order: &[usize],
    start: i32,
    end: i32,
    stack: &mut [i32],
    nsign: i32,
    sign2: i32,
) -> i32 {
    let p = |i: i32| a[order[i as usize]];
    let incr = if end > start { 1 } else { -1 };
    let (mut pprev, mut pcur) = (start, start + incr);
    let mut pnext = pcur + incr;
    let mut size = 3usize;
    if start == end || (p(start)[0] == p(end)[0] && p(start)[1] == p(end)[1]) {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    let end = end + incr;
    while pnext != end {
        let cury = p(pcur)[1];
        let nexty = p(pnext)[1];
        let by = nexty - cury;
        if sign_f32(by) != nsign {
            let ax = p(pcur)[0] - p(pprev)[0];
            let bx = p(pnext)[0] - p(pcur)[0];
            let ay = cury - p(pprev)[1];
            let convexity = ay as f64 * bx as f64 - ax as f64 * by as f64;
            if sign_f64(convexity) == sign2 && (ax != 0.0 || ay != 0.0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[size] = pnext;
                size += 1;
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[size - 2] = pnext;
                pcur = pprev;
                pprev = stack[size - 4];
                size -= 1;
            }
        } else {
            pnext += incr;
            stack[size - 1] = pnext;
        }
    }
    size as i32 - 1
}

/// `cv::convexHull(points, clockwise = false, returnPoints = true)`.
pub fn convex_hull(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let total = pts.len();
    if total == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&i, &j| {
        let (a, b) = (pts[i], pts[j]);
        a[0].partial_cmp(&b[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal))
            .then(i.cmp(&j))
    });
    let (mut miny, mut maxy) = (0usize, 0usize);
    for i in 1..total {
        let y = pts[order[i]][1];
        if pts[order[miny]][1] > y {
            miny = i;
        }
        if pts[order[maxy]][1] < y {
            maxy = i;
        }
    }
    let mut hull: Vec<usize> = Vec::new();
    let mut stack = vec![0i32; total + 2];
    let first = pts[order[0]];
    let last = pts[order[total - 1]];
    if first[0] == last[0] && first[1] == last[1] {
        hull.push(order[0]);
    } else {
        // upper half
        let tl = sklansky(pts, &order, 0, maxy as i32, &mut stack, -1, 1);
        let tl_stack: Vec<i32> = stack[..tl as usize].to_vec();
        let tr = sklansky(
            pts,
            &order,
            total as i32 - 1,
            maxy as i32,
            &mut stack[tl as usize..],
            -1,
            -1,
        );
        let tr_stack: Vec<i32> = stack[tl as usize..(tl + tr) as usize].to_vec();
        // counter-clockwise: the two halves trade places
        let (tl_stack, tr_stack) = (tr_stack, tl_stack);
        let (tl, tr) = (tr, tl);
        for i in 0..(tl - 1).max(0) {
            hull.push(order[tl_stack[i as usize] as usize]);
        }
        for i in (1..tr).rev() {
            hull.push(order[tr_stack[i as usize] as usize]);
        }
        let stop_idx = if tr > 2 {
            tr_stack[1]
        } else if tl > 2 {
            tl_stack[(tl - 2) as usize]
        } else {
            -1
        };
        // lower half
        let mut bl = sklansky(pts, &order, 0, miny as i32, &mut stack, 1, -1);
        let bl_stack: Vec<i32> = stack[..bl as usize].to_vec();
        let mut br = sklansky(
            pts,
            &order,
            total as i32 - 1,
            miny as i32,
            &mut stack[bl as usize..],
            1,
            1,
        );
        let br_stack: Vec<i32> = stack[bl as usize..(bl + br) as usize].to_vec();
        if stop_idx >= 0 {
            let check = if bl > 2 {
                bl_stack[1]
            } else if bl + br > 2 {
                br_stack[(2 - bl) as usize]
            } else {
                -1
            };
            let same = |a: i32, b: i32| {
                let (pa, pb) = (pts[order[a as usize]], pts[order[b as usize]]);
                pa[0] == pb[0] && pa[1] == pb[1]
            };
            if check == stop_idx || (check >= 0 && same(check, stop_idx)) {
                bl = bl.min(2);
                br = br.min(2);
            }
        }
        for i in 0..(bl - 1).max(0) {
            hull.push(order[bl_stack[i as usize] as usize]);
        }
        for i in (1..br).rev() {
            hull.push(order[br_stack[i as usize] as usize]);
        }
        // a cyclic shift toward an ascending or descending index order
        let nout = hull.len();
        if nout >= 3 {
            let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
            for i in 1..nout {
                let idx = hull[i];
                lt += usize::from(hull[i - 1] < idx);
                if lt > 1 && lt + 2 <= i {
                    break;
                }
                if idx < hull[min_idx] {
                    min_idx = i;
                }
                if idx > hull[max_idx] {
                    max_idx = i;
                }
            }
            let mmdist = max_idx.abs_diff(min_idx);
            if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
                let ascending = (max_idx + 1) % nout == min_idx;
                let i0 = if ascending { min_idx } else { max_idx };
                if i0 > 0 {
                    let mut out = vec![0usize; nout];
                    let mut j = i0;
                    let mut i = 0;
                    while i < nout {
                        let cur = hull[j];
                        out[i] = cur;
                        let nj = if j + 1 < nout { j + 1 } else { 0 };
                        if i < nout - 1 && ascending != (cur < hull[nj]) {
                            break;
                        }
                        j = nj;
                        i += 1;
                    }
                    if i == nout {
                        hull = out;
                    }
                }
            }
        }
    }
    hull.into_iter().map(|i| pts[i]).collect()
}

/// `cv::RotatedRect`: center, (width, height), angle in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotatedRect {
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub angle: f32,
}

/// The rotating calipers over a convex hull (`CALIPERS_MINAREARECT`): the
/// corner point and the two side vectors.
fn calipers(points: &[[f32; 2]]) -> [[f32; 2]; 3] {
    let n = points.len();
    let mut minarea = f32::MAX;
    let mut vect = vec![[0f32; 2]; n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0[0], pt0[0], pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            left_x = pt0[0];
            left = i;
        }
        if pt0[0] > right_x {
            right_x = pt0[0];
            right = i;
        }
        if pt0[1] > top_y {
            top_y = pt0[1];
            top = i;
        }
        if pt0[1] < bottom_y {
            bottom_y = pt0[1];
            bottom = i;
        }
        let pt = points[if i + 1 < n { i + 1 } else { 0 }];
        let dx = (pt[0] - pt0[0]) as f64;
        let dy = (pt[1] - pt0[1]) as f64;
        vect[i] = [dx as f32, dy as f32];
        inv_len[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    // (OpenCV seeds the base vector with the hull's orientation, but in
    // this mode the first rotation overwrites it before it is read)
    let mut seq = [bottom, right, top, left];
    let mut best = (0usize, 0f32, 0f32, 0f32, 0f32, 0usize);
    let cw = |v: [f32; 2]| [v[1], -v[0]];
    let ccw = |v: [f32; 2]| [-v[1], v[0]];
    let r180 = |v: [f32; 2]| [-v[0], -v[1]];
    let first_right = |v1: [f32; 2], v2: [f32; 2]| {
        let t = cw(v1);
        t[0].mul_add(v2[0], t[1] * v2[1]) < 0.0
    };
    for _ in 0..n {
        let rot = [
            vect[seq[0]],
            cw(vect[seq[1]]),
            r180(vect[seq[2]]),
            ccw(vect[seq[3]]),
        ];
        let mut main = 0usize;
        for i in 1..4 {
            if first_right(rot[i], rot[main]) {
                main = i;
            }
        }
        let pi = seq[main];
        let (lx, ly) = (vect[pi][0] * inv_len[pi], vect[pi][1] * inv_len[pi]);
        let (base_a, base_b) = match main {
            0 => (lx, ly),
            1 => (ly, -lx),
            2 => (-lx, -ly),
            _ => (-ly, lx),
        };
        seq[main] += 1;
        if seq[main] == n {
            seq[main] = 0;
        }
        let dx = points[seq[1]][0] - points[seq[3]][0];
        let dy = points[seq[1]][1] - points[seq[3]][1];
        let width = dx.mul_add(base_a, dy * base_b);
        let dx = points[seq[2]][0] - points[seq[0]][0];
        let dy = points[seq[2]][1] - points[seq[0]][1];
        let height = (-dx).mul_add(base_b, dy * base_a);
        let area = width * height;
        if area <= minarea {
            minarea = area;
            best = (seq[3], base_a, width, base_b, height, seq[0]);
        }
    }
    let (li, a1, w, b1, h, bi) = best;
    let (a2, b2) = (-b1, a1);
    let c1 = a1.mul_add(points[li][0], points[li][1] * b1);
    let c2 = a2.mul_add(points[bi][0], points[bi][1] * b2);
    let idet = 1.0f32 / a1.mul_add(b2, -(a2 * b1));
    let px = c1.mul_add(b2, -(c2 * b1)) * idet;
    let py = a1.mul_add(c2, -(a2 * c1)) * idet;
    [[px, py], [a1 * w, b1 * w], [a2 * h, b2 * h]]
}

/// `cv::minAreaRect` over float32 points.
pub fn min_area_rect(pts: &[[f32; 2]]) -> RotatedRect {
    let hull = convex_hull(pts);
    let mut r = RotatedRect {
        center: [0.0, 0.0],
        size: [0.0, 0.0],
        angle: 0.0,
    };
    let n = hull.len();
    if n > 2 {
        let out = calipers(&hull);
        r.center = [
            out[0][0] + (out[1][0] + out[2][0]) * 0.5,
            out[0][1] + (out[1][1] + out[2][1]) * 0.5,
        ];
        r.size = [
            ((out[1][0] as f64) * (out[1][0] as f64) + (out[1][1] as f64) * (out[1][1] as f64))
                .sqrt() as f32,
            ((out[2][0] as f64) * (out[2][0] as f64) + (out[2][1] as f64) * (out[2][1] as f64))
                .sqrt() as f32,
        ];
        r.angle = (out[1][1] as f64).atan2(out[1][0] as f64) as f32;
    } else if n == 2 {
        r.center = [
            (hull[0][0] + hull[1][0]) * 0.5,
            (hull[0][1] + hull[1][1]) * 0.5,
        ];
        let dx = (hull[1][0] - hull[0][0]) as f64;
        let dy = (hull[1][1] - hull[0][1]) as f64;
        r.size = [(dx * dx + dy * dy).sqrt() as f32, 0.0];
        r.angle = dy.atan2(dx) as f32;
    } else if n == 1 {
        r.center = hull[0];
    }
    // `box.angle*180/CV_PI`: float times int stays float, then the double pi
    r.angle = ((r.angle * 180.0f32) as f64 / std::f64::consts::PI) as f32;
    r
}

/// `cv::boxPoints` (`RotatedRect::points`).
pub fn box_points(r: &RotatedRect) -> [[f32; 2]; 4] {
    let ang = r.angle as f64 * std::f64::consts::PI / 180.0;
    let b = ang.cos() as f32 * 0.5;
    let a = ang.sin() as f32 * 0.5;
    let [cx, cy] = r.center;
    let [w, h] = r.size;
    let p0 = [
        (-b).mul_add(w, (-a).mul_add(h, cx)),
        (-a).mul_add(w, b.mul_add(h, cy)),
    ];
    let p1 = [
        (-b).mul_add(w, a.mul_add(h, cx)),
        (-a).mul_add(w, (-b).mul_add(h, cy)),
    ];
    [
        p0,
        p1,
        [2.0 * cx - p0[0], 2.0 * cy - p0[1]],
        [2.0 * cx - p1[0], 2.0 * cy - p1[1]],
    ]
}

// ---------------------------------------------------------------- fill

/// `cv::clipLine` on an image of `w` x `h`; false when nothing is left.
fn clip_line(w: i64, h: i64, p1: &mut [i64; 2], p2: &mut [i64; 2]) -> bool {
    let (right, bottom) = (w - 1, h - 1);
    if w <= 0 || h <= 0 {
        return false;
    }
    let code = |p: &[i64; 2]| -> i32 {
        i32::from(p[0] < 0)
            + i32::from(p[0] > right) * 2
            + i32::from(p[1] < 0) * 4
            + i32::from(p[1] > bottom) * 8
    };
    let (mut c1, mut c2) = (code(p1), code(p2));
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            p1[0] += ((a - p1[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p1[1] = a;
            c1 = i32::from(p1[0] < 0) + i32::from(p1[0] > right) * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            p2[0] += ((a - p2[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p2[1] = a;
            c2 = i32::from(p2[0] < 0) + i32::from(p2[0] > right) * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                p1[1] +=
                    ((a - p1[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p1[0] = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                p2[1] +=
                    ((a - p2[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p2[0] = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

/// `Line(img, p1, p2, LINE_8)`: the 8-connected line iterator, left to
/// right, clipped to the image.
fn line8(img: &mut [u8], w: usize, h: usize, p1: [i64; 2], p2: [i64; 2], val: u8) {
    let (mut p1, mut p2) = (p1, p2);
    // the iterator clips in int; a coordinate beyond i32 cannot occur here
    let outside = |p: &[i64; 2]| p[0] < 0 || p[0] >= w as i64 || p[1] < 0 || p[1] >= h as i64;
    if (outside(&p1) || outside(&p2)) && !clip_line(w as i64, h as i64, &mut p1, &mut p2) {
        return;
    }
    let (mut dx, mut dy) = (p2[0] - p1[0], p2[1] - p1[1]);
    let (mut delta_x, mut delta_y) = (1i64, 1i64);
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        p1 = p2;
    }
    if dy < 0 {
        dy = -dy;
        delta_y = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut delta_x, &mut delta_y);
    }
    let mut err = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    let (mut minus_shift, mut plus_shift, mut minus_step, mut plus_step) =
        (delta_x, 0i64, 0i64, delta_y);
    let count = dx + 1;
    if vert {
        std::mem::swap(&mut plus_step, &mut plus_shift);
        std::mem::swap(&mut minus_step, &mut minus_shift);
    }
    let mut p = p1;
    for _ in 0..count {
        img[p[1] as usize * w + p[0] as usize] = val;
        let neg = err < 0;
        err += minus_delta + if neg { plus_delta } else { 0 };
        p[0] += minus_shift + if neg { plus_shift } else { 0 };
        p[1] += minus_step + if neg { plus_step } else { 0 };
    }
}

#[derive(Clone, Copy)]
struct Edge {
    y0: i32,
    y1: i32,
    x: i64,
    dx: i64,
    next: usize,
}

const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;
const NIL: usize = usize::MAX;

/// `cv::fillPoly(img, [pts], val)` with LINE_8, no shift, no offset.
pub fn fill_poly(img: &mut [u8], w: usize, h: usize, pts: &[Pt], val: u8) {
    let count = pts.len();
    if count == 0 {
        return;
    }
    // CollectPolyEdges
    let mut edges: Vec<Edge> = Vec::with_capacity(count + 1);
    let conv = |p: Pt| [(p[0] as i64) << XY_SHIFT, p[1] as i64];
    let mut pt0 = conv(pts[count - 1]);
    for &v in pts {
        let pt1 = conv(v);
        let mut t0 = [(pt0[0] + (XY_ONE >> 1)) >> XY_SHIFT, pt0[1]];
        let mut t1 = [(pt1[0] + (XY_ONE >> 1)) >> XY_SHIFT, pt1[1]];
        line8(img, w, h, t0, t1, val);
        let (mut pt0c, mut pt1c) = (pt0, pt1);
        let out = |t: &[i64; 2]| t[0] < 0 || t[0] >= w as i64 || t[1] < 0 || t[1] >= h as i64;
        if out(&t0) || out(&t1) {
            clip_line(w as i64, h as i64, &mut t0, &mut t1);
            if t0[1] != t1[1] {
                pt0c = [t0[0] << XY_SHIFT, t0[1]];
                pt1c = [t1[0] << XY_SHIFT, t1[1]];
            }
        } else {
            pt0c[0] += XY_ONE >> 1;
            pt1c[0] += XY_ONE >> 1;
        }
        if pt0[1] != pt1[1] {
            let dx = (pt1c[0] - pt0c[0]) / (pt1c[1] - pt0c[1]);
            let e = if pt0[1] < pt1[1] {
                Edge {
                    y0: pt0[1] as i32,
                    y1: pt1[1] as i32,
                    x: pt0c[0] + (pt0[1] - pt0c[1]) * dx,
                    dx,
                    next: NIL,
                }
            } else {
                Edge {
                    y0: pt1[1] as i32,
                    y1: pt0[1] as i32,
                    x: pt1c[0] + (pt1[1] - pt1c[1]) * dx,
                    dx,
                    next: NIL,
                }
            };
            edges.push(e);
        }
        pt0 = pt1;
    }
    fill_edges(img, w, h, edges, val);
}

/// `FillEdgeCollection` (non-AA): the active-edge scan.
fn fill_edges(img: &mut [u8], w: usize, h: usize, mut edges: Vec<Edge>, val: u8) {
    let total = edges.len();
    if total < 2 {
        return;
    }
    let (mut y_max, mut y_min) = (i32::MIN, i32::MAX);
    let (mut x_max, mut x_min) = (-1i64, i64::MAX);
    for e in &edges {
        let x1 = e.x + (e.y1 - e.y0) as i64 * e.dx;
        y_min = y_min.min(e.y0);
        y_max = y_max.max(e.y1);
        x_min = x_min.min(e.x).min(x1);
        x_max = x_max.max(e.x).max(x1);
    }
    if y_max < 0 || y_min >= h as i32 || x_max < 0 || x_min >= (w as i64) << XY_SHIFT {
        return;
    }
    edges.sort_by(|a, b| a.y0.cmp(&b.y0).then(a.x.cmp(&b.x)).then(a.dx.cmp(&b.dx)));
    // the sentinel, then the list head (`tmp`) as an extra node
    edges.push(Edge {
        y0: i32::MAX,
        y1: 0,
        x: 0,
        dx: 0,
        next: NIL,
    });
    let head = edges.len();
    edges.push(Edge {
        y0: i32::MAX,
        y1: 0,
        x: 0,
        dx: 0,
        next: NIL,
    });
    let mut i = 0usize;
    let mut e = 0usize;
    let y_max = y_max.min(h as i32);
    let mut y = edges[e].y0;
    while y < y_max {
        let mut draw = 0;
        let clipline = y < 0;
        let mut prelast = head;
        let mut last = edges[head].next;
        while last != NIL || edges[e].y0 == y {
            if last != NIL && edges[last].y1 == y {
                // the edge ends here
                edges[prelast].next = edges[last].next;
                last = edges[last].next;
                continue;
            }
            let keep_prelast = prelast;
            if last != NIL && (edges[e].y0 > y || edges[last].x < edges[e].x) {
                prelast = last;
                last = edges[last].next;
            } else if i < total {
                // the edge starts here
                edges[prelast].next = e;
                edges[e].next = last;
                prelast = e;
                i += 1;
                e = i;
            } else {
                break;
            }
            if draw != 0 {
                if !clipline {
                    let (x1, x2) = if edges[keep_prelast].x > edges[prelast].x {
                        (
                            edges[prelast].x >> XY_SHIFT,
                            edges[keep_prelast].x >> XY_SHIFT,
                        )
                    } else {
                        (
                            edges[keep_prelast].x >> XY_SHIFT,
                            edges[prelast].x >> XY_SHIFT,
                        )
                    };
                    if x1 < w as i64 && x2 >= 0 {
                        let x1 = x1.max(0) as usize;
                        let x2 = x2.min(w as i64 - 1) as usize;
                        let row = y as usize * w;
                        img[row + x1..=row + x2].fill(val);
                    }
                }
                let (kd, pd) = (edges[keep_prelast].dx, edges[prelast].dx);
                edges[keep_prelast].x += kd;
                edges[prelast].x += pd;
            }
            draw ^= 1;
        }
        // re-sort the active list by x (bubble sort)
        let mut keep_prelast = NIL;
        loop {
            let mut prelast = head;
            let mut last = edges[head].next;
            let mut last_exchange = NIL;
            while last != keep_prelast && last != NIL && edges[last].next != NIL {
                let te = edges[last].next;
                if edges[last].x > edges[te].x {
                    edges[prelast].next = te;
                    edges[last].next = edges[te].next;
                    edges[te].next = last;
                    prelast = te;
                    last_exchange = prelast;
                } else {
                    prelast = last;
                    last = te;
                }
            }
            if last_exchange == NIL {
                break;
            }
            keep_prelast = last_exchange;
            if keep_prelast == edges[head].next || keep_prelast == head {
                break;
            }
        }
        y += 1;
    }
}

// ---------------------------------------------------------------- overlap

fn area2(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    (0..n)
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % n]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum()
}

/// A polygon's area (Shapely `Polygon(p).area`).
pub fn polygon_area(p: &[[f64; 2]]) -> f64 {
    if p.len() < 3 {
        0.0
    } else {
        (area2(p) / 2.0).abs()
    }
}

fn is_convex(p: &[[f64; 2]]) -> bool {
    let n = p.len();
    if n < 4 {
        return true;
    }
    let mut sign = 0f64;
    for i in 0..n {
        let (a, b, c) = (p[i], p[(i + 1) % n], p[(i + 2) % n]);
        let cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cr != 0.0 {
            if sign != 0.0 && cr.signum() != sign {
                return false;
            }
            sign = cr.signum();
        }
    }
    true
}

/// Sutherland-Hodgman: `subject` clipped by the convex `clip`.
fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut clip = clip.to_vec();
    if area2(&clip) < 0.0 {
        clip.reverse();
    }
    let mut out = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if out.is_empty() {
            break;
        }
        let (a, b) = (clip[i], clip[(i + 1) % n]);
        let side = |p: [f64; 2]| (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        let input = std::mem::take(&mut out);
        let m = input.len();
        for k in 0..m {
            let (cur, prev) = (input[k], input[(k + m - 1) % m]);
            let (sc, sp) = (side(cur), side(prev));
            if sc >= 0.0 {
                if sp < 0.0 {
                    let t = sp / (sp - sc);
                    out.push([
                        prev[0] + t * (cur[0] - prev[0]),
                        prev[1] + t * (cur[1] - prev[1]),
                    ]);
                }
                out.push(cur);
            } else if sp >= 0.0 {
                let t = sp / (sp - sc);
                out.push([
                    prev[0] + t * (cur[0] - prev[0]),
                    prev[1] + t * (cur[1] - prev[1]),
                ]);
            }
        }
    }
    out
}

/// Ear-clipping triangulation of a simple polygon.
fn triangles(p: &[[f64; 2]]) -> Vec<[[f64; 2]; 3]> {
    let mut v: Vec<[f64; 2]> = p.to_vec();
    if area2(&v) < 0.0 {
        v.reverse();
    }
    let mut out = Vec::new();
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let mut guard = 0;
    while v.len() > 3 && guard < 10_000 {
        guard += 1;
        let n = v.len();
        let mut clipped = false;
        for i in 0..n {
            let (a, b, c) = (v[(i + n - 1) % n], v[i], v[(i + 1) % n]);
            if cross(a, b, c) <= 0.0 {
                continue;
            }
            let inside = v.iter().enumerate().any(|(k, &q)| {
                k != i
                    && k != (i + n - 1) % n
                    && k != (i + 1) % n
                    && cross(a, b, q) >= 0.0
                    && cross(b, c, q) >= 0.0
                    && cross(c, a, q) >= 0.0
            });
            if !inside {
                out.push([a, b, c]);
                v.remove(i);
                clipped = true;
                break;
            }
        }
        if !clipped {
            break;
        }
    }
    if v.len() == 3 {
        out.push([v[0], v[1], v[2]]);
    }
    out
}

/// The area two polygons share.
pub fn intersection_area(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    if a.len() < 3 || b.len() < 3 {
        return 0.0;
    }
    if is_convex(b) {
        polygon_area(&clip_convex(a, b))
    } else if is_convex(a) {
        polygon_area(&clip_convex(b, a))
    } else {
        triangles(a)
            .iter()
            .map(|t| polygon_area(&clip_convex(b, t)))
            .sum()
    }
}

/// PaddleX's `calculate_polygon_overlap_ratio`: intersection over the union
/// (`small = false`) or over the smaller area.
pub fn overlap(a: &[[f64; 2]], b: &[[f64; 2]], small: bool) -> f64 {
    let inter = intersection_area(a, b);
    let (aa, ab) = (polygon_area(a), polygon_area(b));
    let den = if small { aa.min(ab) } else { aa + ab - inter };
    if den > 0.0 { inter / den } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_filled_square_has_its_outer_contour() {
        let (w, h) = (6, 5);
        let mut m = vec![0u8; w * h];
        for y in 1..4 {
            for x in 1..5 {
                m[y * w + x] = 1;
            }
        }
        let c = external_contours(&m, w, h);
        assert_eq!(c, vec![vec![[1, 1], [1, 3], [4, 3], [4, 1]]]);
        assert_eq!(contour_area(&c[0]), 6.0);
        assert_eq!(arc_length(&c[0]), 10.0);
    }

    #[test]
    fn nested_blobs_report_only_outer_borders_newest_first() {
        let (w, h) = (9, 9);
        let mut m = vec![0u8; w * h];
        for y in 0..9 {
            for x in 0..9 {
                // a ring with a dot inside, and a separate dot below-right
                let ring = (1..=5).contains(&x)
                    && (1..=5).contains(&y)
                    && !((2..=4).contains(&x) && (2..=4).contains(&y));
                if ring || (x == 3 && y == 3) || (x == 7 && y == 7) {
                    m[y * w + x] = 1;
                }
            }
        }
        let c = external_contours(&m, w, h);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0], vec![[7, 7]]);
        assert_eq!(c[1][0], [1, 1]);
    }

    #[test]
    fn min_area_rect_of_an_axis_box() {
        // cv2 4.10: ((2, 1), (2, 4), 90) and these corners - cos(pi / 2)
        // leaves its -6.1e-17 on the first x, as there
        let r = min_area_rect(&[[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]);
        assert_eq!(
            r,
            RotatedRect {
                center: [2.0, 1.0],
                size: [2.0, 4.0],
                angle: 90.0
            }
        );
        assert_eq!(
            box_points(&r),
            [[-6.123234e-17, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]
        );
    }

    #[test]
    fn fill_covers_the_polygon_and_its_edges() {
        let (w, h) = (6, 5);
        let mut m = vec![0u8; w * h];
        fill_poly(&mut m, w, h, &[[1, 1], [4, 1], [4, 3], [1, 3]], 1);
        let n: usize = m.iter().map(|&v| v as usize).sum();
        assert_eq!(n, 12);
        assert_eq!(m[w + 1], 1);
        assert_eq!(m[3 * w + 4], 1);
    }

    #[test]
    fn overlap_of_squares() {
        let a = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let b = [[1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [1.0, 2.0]];
        assert!((overlap(&a, &b, false) - 2.0 / 6.0).abs() < 1e-12);
        assert!((overlap(&a, &b, true) - 0.5).abs() < 1e-12);
        // a concave L against a square
        let l = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let s = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]];
        assert!((intersection_area(&l, &s) - 0.75).abs() < 1e-12);
        assert!((intersection_area(&s, &l) - 0.75).abs() < 1e-12);
    }
}
