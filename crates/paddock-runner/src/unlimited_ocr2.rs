//! Unlimited-OCR-2's block markup -> grounded regions.
//!
//! The checkpoint writes neither Markdown nor v1's det/ref records. A document
//! is one string of `<PAGE>`-opened pages, each a run of layout blocks:
//!
//! ```text
//! <PAGE><BLOCK>title<LEVEL2>[82, 59, 806, 91]<FORK>Only All There
//! <BLOCK>text[81, 136, 918, 170]<FORK>That there when many ...
//! <BLOCK>cross_table[..]<FORK><table>..</table><PAGE><BLOCK>append_table[..]<FORK>..
//! ```
//!
//! The grammar is the one the authors' own Markdown converter reads: after
//! `<BLOCK>` a label (`[A-Za-z_]+`, lowercased), then a `<LEVELn>` and a box
//! in either order (each optional), then `<FORK>`; the block's text runs to
//! the next `<BLOCK>` or `<PAGE>`. A box may carry more than four numbers
//! (the first four are the box) and full-width commas. `append_<t>` continues
//! the open `cross_<t>` block across a page break, and a `<MERGE>` in the
//! text before a block joins that block to the previous column - both show as
//! `continues`. The joining itself (chains, a split table's repeated header
//! rows) is rendering and stays the client's: every block is a region as the
//! model wrote it.
//!
//! Boxes stay on the family's 0-999 grid (the checkpoint's own drawing code
//! scales by `/ 999`), the shared `regions` space.

use crate::deepseek_ocr::Region;

const PAGE: &str = "<PAGE>";
const BLOCK: &str = "<BLOCK>";
const FORK: &str = "<FORK>";
const MERGE: &str = "<MERGE>";
const LEVEL: &str = "<LEVEL";

/// Every block of a raw Unlimited-OCR-2 output, pages numbered from 0 in
/// request order. Empty when the text carries no `<BLOCK>` record.
pub fn parse_blocks(raw: &str) -> Vec<Region> {
    if !raw.contains(BLOCK) {
        return Vec::new();
    }
    let mut segs: Vec<&str> = raw.split(PAGE).collect();
    // the output opens on its first <PAGE>; text before it is not a page
    if segs.len() > 1 && segs[0].trim().is_empty() {
        segs.remove(0);
    }
    let mut out = Vec::new();
    for (page, seg) in segs.iter().enumerate() {
        let heads: Vec<(usize, Head)> = seg
            .match_indices(BLOCK)
            .filter_map(|(i, _)| head(&seg[i + BLOCK.len()..]).map(|h| (i, h)))
            .collect();
        for (j, (at, h)) in heads.iter().enumerate() {
            let at = *at;
            let body = at + BLOCK.len() + h.len;
            let end = heads.get(j + 1).map_or(seg.len(), |n| n.0);
            // a <MERGE> anywhere in the previous block's text joins this one
            let merged = j > 0 && {
                let prev = heads[j - 1].0 + BLOCK.len() + heads[j - 1].1.len;
                seg[prev..at].contains(MERGE)
            };
            let text = seg[body..end].replace(MERGE, "");
            let text = text.trim();
            out.push(Region {
                label: h.label.clone(),
                boxes: h.bbox.into_iter().collect(),
                text: (!text.is_empty()).then(|| text.to_owned()),
                quads: Vec::new(),
                continues: merged || h.label.starts_with("append_"),
                level: h.level,
                page: Some(page),
            });
        }
    }
    out
}

/// One block header: what follows `<BLOCK>` up to and including `<FORK>`.
struct Head {
    label: String,
    level: Option<u32>,
    bbox: Option<[i64; 4]>,
    /// bytes the header spans
    len: usize,
}

/// Parse a header, or None when no label follows (the marker is then text).
fn head(s: &str) -> Option<Head> {
    let mut c = Cursor { s, at: 0 };
    c.ws();
    let label = c.take(|ch| ch.is_ascii_alphabetic() || ch == '_');
    if label.is_empty() {
        return None;
    }
    let (mut level, mut bbox) = (None, None);
    // a level and a box, either order, each at most once
    for _ in 0..2 {
        c.ws();
        if level.is_none() && c.rest().starts_with(LEVEL) {
            c.at += LEVEL.len();
            c.ws();
            level = c.take(|ch| ch.is_ascii_digit()).parse().ok();
            c.ws();
            if c.rest().starts_with(['>', '}']) {
                c.at += 1;
            }
        } else if bbox.is_none() && c.rest().starts_with('[') {
            match parse_box(c.rest()) {
                Some((b, n)) => {
                    bbox = Some(b);
                    c.at += n;
                }
                None => break,
            }
        } else {
            break;
        }
    }
    c.ws();
    if c.rest().starts_with(FORK) {
        c.at += FORK.len();
    }
    Some(Head {
        label: label.to_ascii_lowercase(),
        level,
        bbox,
        len: c.at,
    })
}

/// `[x0, y0, x1, y1 (, more)*]` at the start of `s`: the first four numbers
/// and the bytes the list spans, or None when it is not four-plus integers.
fn parse_box(s: &str) -> Option<([i64; 4], usize)> {
    let close = s.find(']')?;
    let nums: Vec<i64> = s[1..close]
        .split([',', '，'])
        .map(|t| t.trim().parse().ok())
        .collect::<Option<_>>()?;
    (nums.len() >= 4).then(|| ([nums[0], nums[1], nums[2], nums[3]], close + 1))
}

struct Cursor<'a> {
    s: &'a str,
    at: usize,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> &'a str {
        &self.s[self.at..]
    }

    fn ws(&mut self) {
        let r = self.rest();
        self.at += r.len() - r.trim_start().len();
    }

    fn take(&mut self, ok: impl Fn(char) -> bool) -> &'a str {
        let r = self.rest();
        let n = r.find(|ch: char| !ok(ch)).unwrap_or(r.len());
        self.at += n;
        &r[..n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two pages of the reference's own output on the battery (trimmed).
    const TWO_PAGES: &str = "<PAGE><BLOCK>title[81, 60, 390, 88]<FORK>Do At Without\n\
        <BLOCK>table[81, 109, 919, 233]<FORK><table><tr><td>Beyond</td></tr></table>\n\
        <BLOCK>page_number[491, 971, 529, 985]<FORK>279\n\
        <PAGE><BLOCK>title<LEVEL2>[82, 59, 806, 91]<FORK>Only All There Such Schedule She\n\
        <BLOCK>title<LEVEL3>[81, 109, 400, 129]<FORK>At Quarter With About\n\
        <BLOCK>text[81, 136, 918, 170]<FORK>That there when many supplier.";

    #[test]
    fn blocks_carry_page_level_box_and_text() {
        let rs = parse_blocks(TWO_PAGES);
        assert_eq!(rs.len(), 6);
        assert_eq!(rs[0].label, "title");
        assert_eq!(rs[0].boxes, vec![[81, 60, 390, 88]]);
        assert_eq!(rs[0].text.as_deref(), Some("Do At Without"));
        assert_eq!((rs[0].page, rs[0].level), (Some(0), None));
        assert!(rs[1].text.as_deref().unwrap().starts_with("<table>"));
        assert_eq!(rs[2].label, "page_number");
        assert_eq!(rs[3].page, Some(1));
        assert_eq!(rs[3].level, Some(2));
        assert_eq!(rs[3].boxes, vec![[82, 59, 806, 91]]);
        assert_eq!(rs[4].level, Some(3));
        assert_eq!(
            rs[5].text.as_deref(),
            Some("That there when many supplier.")
        );
        assert!(rs.iter().all(|r| !r.continues));
        let v = rs[3].to_json();
        assert_eq!(
            (v["page"].as_u64(), v["level"].as_u64()),
            (Some(1), Some(2))
        );
    }

    #[test]
    fn either_order_extra_numbers_and_continuations() {
        let raw = "<PAGE><BLOCK>title[1, 2, 3, 4]<LEVEL1><FORK>Head\n\
                   <BLOCK>cross_table[5，6，7，8，9]<FORK><table></table><MERGE>\n\
                   <BLOCK>text[1, 1, 2, 2]<FORK>joined column\
                   <PAGE><BLOCK>append_table[0, 0, 9, 9]<FORK><table></table>";
        let rs = parse_blocks(raw);
        assert_eq!(rs.len(), 4);
        assert_eq!((rs[0].level, rs[0].boxes[0]), (Some(1), [1, 2, 3, 4]));
        assert_eq!(
            rs[1].boxes,
            vec![[5, 6, 7, 8]],
            "first four of five, full-width commas"
        );
        assert_eq!(
            rs[1].text.as_deref(),
            Some("<table></table>"),
            "<MERGE> is not text"
        );
        assert!(rs[2].continues, "a <MERGE> before it joins the column");
        assert!(
            rs[3].continues && rs[3].page == Some(1),
            "append_ across the break"
        );
    }

    #[test]
    fn no_block_markup_is_no_regions() {
        assert!(parse_blocks("plain text [1, 2, 3, 4]").is_empty());
        assert!(parse_blocks("<|det|>title [1, 2, 3, 4]<|/det|>x").is_empty());
        // a marker with no label is text, a box-less block keeps its text
        let rs = parse_blocks("<PAGE><BLOCK> [1,2]<BLOCK>text<FORK>no box");
        assert_eq!(rs.len(), 1);
        assert!(rs[0].boxes.is_empty());
        assert_eq!(rs[0].text.as_deref(), Some("no box"));
    }
}
