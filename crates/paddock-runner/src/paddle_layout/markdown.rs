//! A page's blocks as Markdown - PaddleX's `MarkdownConverter` with the
//! PaddleOCR-VL handlers (pretty, formula numbers hidden): each block
//! through its label's formatter, joined by blank lines, labels without a
//! formatter skipped. 1.6 drops page furniture from the Markdown (number,
//! footnote, header, footer, their pictures, aside text) though it still
//! reads it.

/// What the converter needs of one block.
pub struct MdBlock<'a> {
    pub label: &'a str,
    pub content: &'a str,
    pub bbox: [i32; 4],
}

/// PaddleOCR-VL 1.6's `markdown_ignore_labels`.
const IGNORE: [&str; 7] = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
];

/// The Chinese numerals the reference's title pattern names (one..ten,
/// hundred, thousand, ten thousand, hundred million, zero, and the financial
/// forms of one..ten).
const CN_NUMERALS: &str = "\u{4e00}\u{4e8c}\u{4e09}\u{56db}\u{4e94}\u{516d}\u{4e03}\u{516b}\u{4e5d}\u{5341}\u{767e}\u{5343}\u{4e07}\u{4ebf}\u{96f6}\u{58f9}\u{8d30}\u{53c1}\u{8086}\u{4f0d}\u{9646}\u{67d2}\u{634c}\u{7396}\u{62fe}";

fn collapse(s: &str) -> String {
    s.replace("-\n", "").replace('\n', " ")
}

fn centered(s: &str) -> String {
    format!("<div style=\"text-align: center;\">{}</div>\n", collapse(s))
}

/// The picture reference the reference pipeline writes (its crops are saved
/// under these names; the region list carries the boxes to cut them from).
pub fn image_path(label: &str, b: [i32; 4]) -> String {
    format!(
        "imgs/img_in_{label}_box_{}_{}_{}_{}.jpg",
        b[0], b[1], b[2], b[3]
    )
}

/// The title-numbering regex's first group at the head of `s` (after its
/// leading whitespace): Arabic `1.2.3.`, a parenthesized number, Chinese
/// numerals, or a Roman numeral up to X followed by `.` or a space - tried
/// in the pattern's order.
fn numbering(s: &str) -> Option<usize> {
    let b: Vec<char> = s.chars().collect();
    let len = |n: usize| b[..n].iter().map(|c| c.len_utf8()).sum::<usize>();
    let cn = |c: char| CN_NUMERALS.contains(c);
    let digit19 = |c: char| ('1'..='9').contains(&c);
    // [1-9][0-9]*(?:\.[1-9][0-9]*)*[.<ideographic comma>]?
    if b.first().copied().is_some_and(digit19) {
        let mut i = 1;
        while b.get(i).is_some_and(char::is_ascii_digit) {
            i += 1;
        }
        while b.get(i) == Some(&'.') && b.get(i + 1).copied().is_some_and(digit19) {
            i += 2;
            while b.get(i).is_some_and(char::is_ascii_digit) {
                i += 1;
            }
        }
        if matches!(b.get(i), Some('.' | '\u{3001}')) {
            i += 1;
        }
        return Some(len(i));
    }
    // [(<fullwidth (>](?:[1-9][0-9]*|[cn]+)[)<fullwidth )>]
    if matches!(b.first(), Some('(' | '\u{ff08}')) {
        let mut i = 1;
        if b.get(1).copied().is_some_and(digit19) {
            i = 2;
            while b.get(i).is_some_and(char::is_ascii_digit) {
                i += 1;
            }
        } else {
            while b.get(i).copied().is_some_and(cn) {
                i += 1;
            }
        }
        if i > 1 && matches!(b.get(i), Some(')' | '\u{ff09}')) {
            return Some(len(i + 1));
        }
    }
    // [cn]+[<ideographic comma>.]?
    if b.first().copied().is_some_and(cn) {
        let mut i = 1;
        while b.get(i).copied().is_some_and(cn) {
            i += 1;
        }
        if matches!(b.get(i), Some('.' | '\u{3001}')) {
            i += 1;
        }
        return Some(len(i));
    }
    // (?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)
    for r in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"] {
        if let Some(rest) = s.strip_prefix(r)
            && let Some(next) = rest.chars().next()
            && (next == '.' || next.is_whitespace())
        {
            return Some(r.len() + next.len_utf8());
        }
    }
    None
}

/// `format_title`: numbering normalized to `N rest`, trailing dots off, one
/// heading level per dot below `##`.
fn title(content: &str) -> String {
    let mut t = content.to_owned();
    let body = content.trim_start_matches(char::is_whitespace);
    if let Some(n) = numbering(body) {
        let rest = body[n..].trim_start_matches(char::is_whitespace);
        // `(.*)$` stops at a line end; only a final newline may follow it
        let line = rest.strip_suffix('\n').unwrap_or(rest);
        if !line.contains('\n') {
            t = format!("{} {}", body[..n].trim_matches(char::is_whitespace), line);
        }
    }
    let t = t.trim_end_matches('.');
    let level = if t.contains('.') {
        t.matches('.').count() + 1
    } else {
        1
    };
    collapse(&format!("#{} {t}", "#".repeat(level)))
}

/// `format_first_line`: the first non-blank piece, if it is one of the
/// templates (case-insensitively), through `f`.
fn first_line(content: &str, templates: &[&str], f: impl Fn(&str) -> String, sep: &str) -> String {
    let mut parts: Vec<String> = content.split(sep).map(str::to_owned).collect();
    if let Some(p) = parts.iter_mut().find(|p| !p.trim().is_empty())
        && templates.contains(&p.to_lowercase().as_str())
    {
        *p = f(p);
    }
    parts.join(sep)
}

/// One block's Markdown, None for a label without a formatter.
pub fn render(b: &MdBlock, page_width: usize) -> Option<String> {
    if IGNORE.contains(&b.label) {
        return None;
    }
    let c = b.content;
    Some(match b.label {
        "paragraph_title" => title(c),
        "doc_title" => collapse(&format!("# {c}")),
        "figure_title" => centered(c),
        "vision_footnote" | "text" | "vertical_text" | "reference_content" => {
            c.replace("\n\n", "\n").replace('\n', "\n\n")
        }
        // the Chinese "abstract" heading, or the English one
        "abstract" => first_line(
            c,
            &["\u{6458}\u{8981}", "abstract"],
            |l| format!("## {l}\n"),
            " ",
        ),
        "content" => c.replace("-\n", "  \n").replace('\n', "  \n"),
        "image" | "chart" | "seal" => {
            let width = (b.bbox[2] - b.bbox[0]) as f64;
            let scale = (width / page_width as f64 * 100.0) as i64;
            centered(&format!(
                "<img src=\"{}\" alt=\"Image\" width=\"{scale}%\" />",
                image_path(b.label, b.bbox)
            ))
        }
        "display_formula" | "inline_formula" => c.to_owned(),
        "table" => format!(
            "\n{}",
            c.replace(
                "<table>",
                "<table border=1 style='margin: auto; word-wrap: break-word;'>"
            )
            .replace(
                "<th>",
                "<th style='text-align: center; word-wrap: break-word;'>"
            )
            .replace(
                "<td>",
                "<td style='text-align: center; word-wrap: break-word;'>"
            )
        ),
        "algorithm" => c.trim_matches('\n').to_owned(),
        _ => return None,
    })
}

/// The page's Markdown from its blocks in reading order (the stream builds
/// the same string block by block; the gate checks it whole).
#[cfg(test)]
pub fn page(blocks: &[MdBlock], page_width: usize) -> String {
    let mut md = String::new();
    for b in blocks {
        if let Some(piece) = render(b, page_width) {
            if !md.is_empty() {
                md.push_str("\n\n");
            }
            md.push_str(&piece);
        }
    }
    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_take_their_level_from_the_numbering() {
        assert_eq!(title("Margin More Beyond"), "## Margin More Beyond");
        assert_eq!(title("1.2 Foo"), "### 1.2 Foo");
        assert_eq!(title("3.Results."), "### 3. Results");
        assert_eq!(title("IV. Methods"), "### IV. Methods");
        assert_eq!(
            title("\u{ff08}\u{4e09}\u{ff09}\u{603b}\u{7ed3}"),
            "## \u{ff08}\u{4e09}\u{ff09} \u{603b}\u{7ed3}"
        );
        assert_eq!(title("In the beginning"), "## In the beginning");
    }

    #[test]
    fn furniture_is_skipped_and_blocks_join_by_blank_lines() {
        let b = |label, content| MdBlock {
            label,
            content,
            bbox: [0, 0, 500, 100],
        };
        let md = page(
            &[
                b("header", "Page head"),
                b("doc_title", "A Title"),
                b("text", "line one\nline two"),
                b("formula_number", "(1)"),
                b("image", ""),
            ],
            1000,
        );
        assert_eq!(
            md,
            "# A Title\n\nline one\n\nline two\n\n<div style=\"text-align: center;\"><img \
             src=\"imgs/img_in_image_box_0_0_500_100.jpg\" alt=\"Image\" width=\"50%\" /></div>\n"
        );
    }
}
