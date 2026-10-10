//! What the pipeline does to each region's answer before it is a block's
//! content - PaddleX's PaddleOCR-VL post-processing in its own order:
//!
//! 1. repetition truncation (`truncate_repetitive_content`, armed past 50
//!    characters, 5,000 for a table): a long single line ending in five or
//!    more copies of one phrase keeps what came before them, a line that is
//!    one unit repeated ten or more times becomes the unit, and an answer
//!    whose lines are 80 % one line becomes that line;
//! 2. LaTeX delimiters, only when a pair is present: every `$` dropped,
//!    `\(`/`\)` -> ` $ `/` $`, `\[`/`\]` -> ` $$ ` (a formula number then
//!    loses its `$` again);
//! 3. a table's OTSL (`<fcel> <ecel> <lcel> <ucel> <xcel> <nl>`) -> HTML,
//!    rows padded to the width that costs the fewest edits, spans read off
//!    the merge tags; an answer that yields no cell stays as it was.
//!
//! Every length here is in characters, as Python counts them.

const NL: &str = "<nl>";
const FCEL: &str = "<fcel>";
const ECEL: &str = "<ecel>";
const LCEL: &str = "<lcel>";
const UCEL: &str = "<ucel>";
const XCEL: &str = "<xcel>";
const TAGS: [&str; 6] = [NL, FCEL, ECEL, LCEL, UCEL, XCEL];

/// One region's answer as the block's content.
pub fn clean(label: &str, raw: &str) -> String {
    let min_count = if label == "table" { 5000 } else { 50 };
    let mut s = truncate_repetitive(raw, min_count);
    if (s.contains("\\(") && s.contains("\\)")) || (s.contains("\\[") && s.contains("\\]")) {
        s = s
            .replace('$', "")
            .replace("\\(", " $ ")
            .replace("\\)", " $")
            .replace("\\[\\[", "\\[")
            .replace("\\]\\]", "\\]")
            .replace("\\[", " $$ ")
            .replace("\\]", " $$ ");
        if label == "formula_number" {
            s = s.replace('$', "");
        }
    }
    if label == "table" {
        let html = otsl_to_html(&s);
        if !html.is_empty() {
            s = html;
        }
    }
    s
}

/// Python's `str.strip()` (Unicode whitespace both ends).
fn py_strip(s: &str) -> &str {
    s.trim_matches(char::is_whitespace)
}

fn truncate_repetitive(content: &str, min_count: usize) -> String {
    let (line_threshold, char_threshold, min_len) = (10usize, 10usize, 10usize);
    if content.chars().count() < min_count {
        return content.to_owned();
    }
    let stripped = py_strip(content);
    if stripped.is_empty() {
        return content.to_owned();
    }
    let chars: Vec<char> = stripped.chars().collect();
    let single_line = !stripped.contains('\n');
    // a long line ending in a repeated phrase
    if single_line
        && chars.len() > 100
        && let Some((prefix, unit, count)) = repeating_suffix(&chars, 8, 5)
        && (unit * count) as f64 > chars.len() as f64 * 0.5
    {
        return prefix.iter().collect();
    }
    // the whole line one unit over and over
    if single_line
        && chars.len() > min_len
        && let Some(unit) = shortest_period(&chars)
        && chars.len() / unit >= char_threshold
    {
        return chars[..unit].iter().collect();
    }
    // one line dominating
    let lines: Vec<&str> = content
        .split('\n')
        .map(py_strip)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() < line_threshold {
        return content.to_owned();
    }
    // Counter.most_common(1): the highest count, first seen on a tie
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for l in &lines {
        match counts.iter_mut().find(|(k, _)| k == l) {
            Some(e) => e.1 += 1,
            None => counts.push((l, 1)),
        }
    }
    let (line, count) = counts.iter().fold(
        ("", 0),
        |best, &(k, c)| if c > best.1 { (k, c) } else { best },
    );
    if count >= line_threshold && count as f64 / lines.len() as f64 >= 0.8 {
        return line.to_owned();
    }
    content.to_owned()
}

/// `find_repeating_suffix`: the longest unit (of `min_len` or more chars)
/// the string ends in `min_repeats` copies of, as (prefix, unit length,
/// copies).
fn repeating_suffix(
    s: &[char],
    min_len: usize,
    min_repeats: usize,
) -> Option<(&[char], usize, usize)> {
    let n = s.len();
    for i in (min_len..=n / min_repeats).rev() {
        let unit = &s[n - i..];
        if (1..min_repeats).all(|k| &s[n - (k + 1) * i..n - k * i] == unit) {
            let mut count = 0;
            while (count + 1) * i <= n && &s[n - (count + 1) * i..n - count * i] == unit {
                count += 1;
            }
            return Some((&s[..n - count * i], i, count));
        }
    }
    None
}

/// `find_shortest_repeating_substring`: the shortest period that tiles the
/// whole string.
fn shortest_period(s: &[char]) -> Option<usize> {
    let n = s.len();
    (1..=n / 2).find(|&i| n.is_multiple_of(i) && s.chunks_exact(i).all(|c| c == &s[..i]))
}

/// Split at the six tags: (the tags in order, the pieces - tags and the
/// non-blank text between them, `re.split` with the group kept).
fn split_tags(s: &str) -> (Vec<&str>, Vec<&str>) {
    let (mut tags, mut parts) = (Vec::new(), Vec::new());
    let mut rest = s;
    let mut text_start = 0;
    let mut at = 0;
    while at < rest.len() {
        if let Some(t) = TAGS.iter().find(|t| rest[at..].starts_with(**t)) {
            let text = &rest[text_start..at];
            if !py_strip(text).is_empty() {
                parts.push(text);
            }
            tags.push(*t);
            parts.push(*t);
            rest = &rest[at + t.len()..];
            text_start = 0;
            at = 0;
        } else {
            at += rest[at..].chars().next().map_or(1, char::len_utf8);
        }
    }
    let text = &rest[text_start..];
    if !py_strip(text).is_empty() {
        parts.push(text);
    }
    (tags, parts)
}

/// `otsl_pad_to_sqr_v2`: every row to the one width that costs the fewest
/// added or dropped cells (never narrower than the last filled cell).
fn pad_to_rect(s: &str) -> String {
    let s = py_strip(s);
    if !s.contains(NL) {
        return format!("{s}{NL}");
    }
    let mut rows: Vec<(Vec<&str>, usize)> = Vec::new();
    for line in s.split(NL) {
        if line.is_empty() {
            continue;
        }
        // each cell: a tag and the text up to the next tag (text before the
        // first tag is not a cell)
        let mut cells = Vec::new();
        let mut starts: Vec<usize> = Vec::new();
        let mut at = 0;
        while at < line.len() {
            if TAGS.iter().any(|t| line[at..].starts_with(*t)) {
                starts.push(at);
                at += line[at..].find('>').map_or(1, |e| e + 1);
            } else {
                at += line[at..].chars().next().map_or(1, char::len_utf8);
            }
        }
        for (k, &st) in starts.iter().enumerate() {
            let end = starts.get(k + 1).copied().unwrap_or(line.len());
            cells.push(&line[st..end]);
        }
        if cells.is_empty() {
            continue;
        }
        let min_len = cells
            .iter()
            .rposition(|c| c.starts_with(FCEL))
            .map_or(0, |i| i + 1);
        rows.push((cells, min_len));
    }
    if rows.is_empty() {
        return NL.to_owned();
    }
    let lo = rows.iter().map(|r| r.1).max().unwrap_or(0);
    let hi = lo.max(rows.iter().map(|r| r.0.len()).max().unwrap_or(0));
    let mut best = (usize::MAX, hi);
    for w in lo..=hi {
        let cost: usize = rows.iter().map(|r| r.0.len().abs_diff(w)).sum();
        if cost < best.0 {
            best = (cost, w);
        }
    }
    let width = best.1;
    let mut out = String::new();
    for (k, (cells, _)) in rows.iter().enumerate() {
        if k > 0 {
            out.push_str(NL);
        }
        for c in cells.iter().take(width) {
            out.push_str(c);
        }
        for _ in cells.len()..width {
            out.push_str(ECEL);
        }
    }
    out.push_str(NL);
    out
}

struct Cell {
    text: String,
    rows: (usize, usize),
    cols: (usize, usize),
}

/// `convert_otsl_to_html`, the parse included warts and all (a `<fcel>`
/// straight before another tag takes that tag as its text, as upstream
/// does) - the point is the reference's HTML, not a better parser.
pub fn otsl_to_html(otsl: &str) -> String {
    let padded = pad_to_rect(otsl);
    let (tags, texts) = split_tags(&padded);
    // rows of tags between <nl>s (runs of <nl> collapse)
    let mut grid: Vec<Vec<&str>> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for t in &tags {
        if *t == NL {
            if !cur.is_empty() {
                grid.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(t);
        }
    }
    if !cur.is_empty() {
        grid.push(cur);
    }
    let mut texts: Vec<&str> = texts;
    if !grid.is_empty() {
        let max_cols = grid.iter().map(Vec::len).max().unwrap_or(0);
        for row in &mut grid {
            row.resize(max_cols, ECEL);
        }
        // re-thread the texts through the completed grid
        let mut out = Vec::new();
        let mut ti = 0;
        for row in &grid {
            for &tok in row {
                out.push(tok);
                if ti < texts.len() && texts[ti] == tok {
                    ti += 1;
                    if ti < texts.len() && !TAGS.contains(&texts[ti]) {
                        out.push(texts[ti]);
                        ti += 1;
                    }
                }
            }
            out.push(NL);
            if ti < texts.len() && texts[ti] == NL {
                ti += 1;
            }
        }
        texts = out;
    }
    let run = |r: usize, c: usize, right: bool, which: [&str; 2]| -> usize {
        let (mut r, mut c, mut span) = (r, c, 0);
        loop {
            let Some(t) = grid.get(r).and_then(|row| row.get(c)) else {
                return span;
            };
            if !which.contains(t) {
                return span;
            }
            span += 1;
            if right {
                c += 1;
            } else {
                r += 1;
            }
        }
    };
    let mut cells = Vec::new();
    let (mut r, mut c) = (0, 0);
    for (i, &t) in texts.iter().enumerate() {
        if t == FCEL || t == ECEL {
            let (mut rs, mut cs, mut off) = (1, 1, 1);
            let mut text = "";
            if t != ECEL {
                text = texts.get(i + 1).copied().unwrap_or("");
                off = 2;
            }
            let right = texts.get(i + off).copied().unwrap_or("");
            let below = grid
                .get(r + 1)
                .and_then(|row| row.get(c))
                .copied()
                .unwrap_or("");
            if right == LCEL || right == XCEL {
                cs += run(r, c + 1, true, [LCEL, XCEL]);
            }
            if below == UCEL || below == XCEL {
                rs += run(r + 1, c, false, [UCEL, XCEL]);
            }
            cells.push(Cell {
                text: py_strip(text).to_owned(),
                rows: (r, r + rs),
                cols: (c, c + cs),
            });
        }
        if [FCEL, ECEL, LCEL, UCEL, XCEL].contains(&t) {
            c += 1;
        }
        if t == NL {
            r += 1;
            c = 0;
        }
    }
    if cells.is_empty() {
        return String::new();
    }
    let (nr, nc) = (grid.len(), grid.iter().map(Vec::len).max().unwrap_or(0));
    // the grid view: each position names the cell covering it (None = the
    // default empty 1 x 1 cell), later cells over earlier
    let mut at: Vec<Vec<Option<usize>>> = vec![vec![None; nc]; nr];
    for (k, cell) in cells.iter().enumerate() {
        for row in at
            .iter_mut()
            .take(cell.rows.1.min(nr))
            .skip(cell.rows.0.min(nr))
        {
            for slot in row
                .iter_mut()
                .take(cell.cols.1.min(nc))
                .skip(cell.cols.0.min(nc))
            {
                *slot = Some(k);
            }
        }
    }
    let mut body = String::new();
    for (i, row) in at.iter().enumerate() {
        body.push_str("<tr>");
        for (j, slot) in row.iter().enumerate() {
            let Some(k) = *slot else {
                body.push_str("<td></td>");
                continue;
            };
            let cell = &cells[k];
            if cell.rows.0 != i || cell.cols.0 != j {
                continue;
            }
            let mut open = String::from("td");
            let (rs, cs) = (cell.rows.1 - cell.rows.0, cell.cols.1 - cell.cols.0);
            if rs > 1 {
                open.push_str(&format!(" rowspan=\"{rs}\""));
            }
            if cs > 1 {
                open.push_str(&format!(" colspan=\"{cs}\""));
            }
            body.push_str(&format!("<{open}>{}</td>", escape(py_strip(&cell.text))));
        }
        body.push_str("</tr>");
    }
    format!("<table>{body}</table>")
}

/// Python's `html.escape(s, quote=True)`.
fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#x27;"),
            c => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn otsl_spans_and_padding() {
        let html = otsl_to_html("<fcel>A<lcel><fcel>B<nl><fcel>C<fcel>D<fcel>E<nl>");
        assert_eq!(
            html,
            "<table><tr><td colspan=\"2\">A</td><td>B</td></tr><tr><td>C</td><td>D</td>\
             <td>E</td></tr></table>"
        );
        let html = otsl_to_html("<fcel>A<fcel>B<nl><ucel><fcel>C<nl>");
        assert_eq!(
            html,
            "<table><tr><td rowspan=\"2\">A</td><td>B</td></tr><tr><td>C</td></tr></table>"
        );
        // a short row is padded with empty cells, text is escaped
        let html = otsl_to_html("<fcel>a<b<fcel>c<nl><fcel>d<nl>");
        assert_eq!(
            html,
            "<table><tr><td>a&lt;b</td><td>c</td></tr><tr><td>d</td><td></td></tr></table>"
        );
        assert_eq!(otsl_to_html("plain text"), "");
    }

    #[test]
    fn repetition_truncation() {
        // the strip moves the unit's space to its front: " abc def gh" x 20,
        // found as its 4-copy multiple (the longest unit is tried first)
        let unit = "abc def gh ";
        let s = format!("start of the line here {}", unit.repeat(20));
        assert_eq!(truncate_repetitive(&s, 50), "start of the line here");
        assert_eq!(truncate_repetitive(&"xy".repeat(40), 50), "xy");
        let lines = "same line\n".repeat(12);
        assert_eq!(truncate_repetitive(&lines, 50), "same line");
        assert_eq!(truncate_repetitive("short", 50), "short");
    }

    #[test]
    fn latex_delimiters_only_with_a_pair() {
        assert_eq!(clean("text", "a \\(x\\) b"), "a  $ x $ b");
        assert_eq!(clean("display_formula", "\\[x^2\\]"), " $$ x^2 $$ ");
        assert_eq!(clean("text", "costs $5 \\( only"), "costs $5 \\( only");
        assert_eq!(clean("formula_number", "\\((1)\\)"), "  (1) ");
    }
}
