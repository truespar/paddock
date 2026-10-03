use super::*;

/// Validate the entire pass before uploading any metadata or submitting work.
/// Spans refer to the original IDs, including lexical means: overlapping spans
/// cannot cause an unbounded duplicated-token allocation.
#[derive(Default, Debug)]
pub(super) struct Plan {
    pub ids: Vec<u32>,
    pub runs: Vec<u32>,
    pub bounds: Vec<u32>,
    pub questions: Vec<u32>,
    pub options: Vec<u32>,
    pub globals: Vec<u32>,
    pub qof: Vec<u32>,
    pub rof: Vec<u32>,
    pub types: Vec<u32>,
    pub qopts: Vec<u32>,
    pub causal: Vec<u32>,
    pub option_tiles: Vec<u32>,
    pub field_tiles: Vec<u32>,
    pub self_tiles: Vec<u32>,
    /// Interleaved (time, height, width) positions, request-local even in a batch.
    pub positions: Vec<u32>,
}
fn tiles(out: &mut Vec<u32>, first: usize, count: usize, key: usize, keys: usize) {
    for row in (0..count).step_by(16) {
        out.extend([
            (first + row) as u32,
            (count - row).min(16) as u32,
            key as u32,
            keys as u32,
        ]);
    }
}
impl Plan {
    pub fn new(reqs: &[ClefRequest<'_>], vocab: usize) -> Result<Self> {
        if reqs.is_empty() || reqs.len() > MAX_REQUESTS {
            return Err(error("request count outside 1..256"));
        }
        let mut p = Self::default();
        for (ri, r) in reqs.iter().enumerate() {
            let start = p.ids.len();
            let n = r.ids.len();
            if n == 0
                || n > MAX_ROWS - start
                || r.questions.is_empty()
                || r.questions.len() > MAX_QUESTIONS - p.types.len()
                || r.ids.iter().any(|&t| t as usize >= vocab)
            {
                return Err(error(format!(
                    "request {ri}: invalid row/question budget or token ID"
                )));
            }
            let qfirst = p.types.len();
            let ofirst = p.qof.len();
            p.ids.extend_from_slice(r.ids);
            let mut row = 0;
            let mut pos = 0u32;
            for image in r.images {
                let (rh, rw) = image.resized;
                let pixels = image
                    .width
                    .checked_mul(image.height)
                    .and_then(|n| n.checked_mul(3));
                let tokens = (rh / 32).checked_mul(rw / 32);
                if pixels != Some(image.rgb.len())
                    || image.width == 0
                    || image.height == 0
                    || image.width.max(image.height) > 131072
                    || image.rgb.len() > 192 << 20
                    || rh == 0
                    || rw == 0
                    || rh % 32 != 0
                    || rw % 32 != 0
                    || tokens.is_none_or(|t| t > n || image.row > n - t)
                    || image.row < row
                {
                    return Err(error("invalid, overlapping or out-of-bounds image rows"));
                }
                while row < image.row {
                    p.positions.extend([pos; 3]);
                    row += 1;
                    pos += 1;
                }
                for y in 0..rh / 32 {
                    for x in 0..rw / 32 {
                        p.positions.extend([pos, pos + y as u32, pos + x as u32]);
                    }
                }
                row += tokens.expect("validated image extent");
                pos += (rh.max(rw) / 32) as u32;
            }
            while row < n {
                p.positions.extend([pos; 3]);
                row += 1;
                pos += 1;
            }
            p.runs.extend([start as u32, n as u32]);
            for _ in 0..n {
                p.bounds.extend([start as u32, (start + n) as u32]);
            }
            p.globals
                .extend([(start + n - 1) as u32, (start + n) as u32]);
            tiles(&mut p.causal, start, n, start, n);
            for q in r.questions {
                let valid = |(a, b): (usize, usize)| a < b && b <= n;
                if q.qtype > 2
                    || !valid(q.span)
                    || q.options.is_empty()
                    || q.options.len() > MAX_OPTIONS - p.qof.len()
                    || !q.options.iter().all(|&s| valid(s))
                {
                    return Err(error(format!(
                        "request {ri}: invalid question type/span/options"
                    )));
                }
                let qi = p.types.len();
                p.questions
                    .extend([(start + q.span.0) as u32, (start + q.span.1) as u32]);
                p.types.push(q.qtype);
                p.rof.push(ri as u32);
                p.qopts.extend([p.qof.len() as u32, q.options.len() as u32]);
                for &(a, b) in &q.options {
                    p.options.extend([(start + a) as u32, (start + b) as u32]);
                    p.qof.push(qi as u32);
                }
            }
            let nq = p.types.len() - qfirst;
            let no = p.qof.len() - ofirst;
            tiles(&mut p.option_tiles, ofirst, no, start, n);
            tiles(&mut p.field_tiles, qfirst, nq, start, n);
            tiles(&mut p.self_tiles, qfirst, nq, qfirst, nq);
        }
        Ok(p)
    }
}
