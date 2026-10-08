//! `tojson` as chat templates are written against it: Python's `json.dumps`.
//!
//! Hugging Face renders chat templates with its own `tojson` filter -
//! `json.dumps(x, ensure_ascii=False, indent=None, separators=None,
//! sort_keys=False)` - and llama.cpp's template engine reproduces the same
//! output (Python's separators, the mapping's own key order, no escaping
//! beyond JSON's). minijinja's builtin differs on all three: compact
//! separators (`{"a":1}` where the training data reads `{"a": 1}`), and
//! `<`, `>`, `&` and `'` HTML-escaped to `<` and friends - so every
//! tool definition and every history tool call reached the model in bytes
//! it was never trained on (Claude Code's tool descriptions are full of
//! apostrophes and `<example>` tags). This is the Python form, kwargs and
//! all: `ensure_ascii`, `indent`, `separators`, `sort_keys`, positional in
//! that order as in transformers' signature.
//!
//! Floats follow Python's `repr` (`1.0`, `1e-05`, `1e+16`), the template
//! authors' environment; llama.cpp prints C++'s six-significant-digit
//! default there, a quirk not worth copying.

use minijinja::value::{Kwargs, Rest, Value, ValueKind};
use minijinja::{Error, ErrorKind};

/// The filter: `value | tojson(ensure_ascii=False, indent=None,
/// separators=None, sort_keys=False)`. Positional arguments ride `args`
/// (minijinja caps a filter at five parameters); keyword arguments arrive as
/// its last element.
pub(super) fn tojson(value: Value, args: Rest<Value>) -> Result<Value, Error> {
    let mut pos: Vec<Value> = args.0;
    let kwargs = match pos.last() {
        Some(v) if v.is_kwargs() => Kwargs::try_from(pos.pop().expect("checked"))?,
        _ => Kwargs::from_iter(std::iter::empty::<(String, Value)>()),
    };
    if pos.len() > 4 {
        return Err(Error::new(
            ErrorKind::TooManyArguments,
            "tojson takes ensure_ascii, indent, separators, sort_keys",
        ));
    }
    let mut pos = pos.into_iter();
    let (ensure_ascii, indent, separators, sort_keys) =
        (pos.next(), pos.next(), pos.next(), pos.next());
    let pick = |pos: Option<Value>, name: &str| -> Result<Option<Value>, Error> {
        match pos {
            Some(v) => Ok(Some(v)),
            None => kwargs.get::<Option<Value>>(name),
        }
    };
    let ensure_ascii = pick(ensure_ascii, "ensure_ascii")?.is_some_and(|v| v.is_true());
    let indent = match pick(indent, "indent")? {
        None => None,
        Some(v) if v.is_none() || v.is_undefined() => None,
        // an int is that many spaces (negative: none, but still newlines);
        // a string is the indent itself, as in Python
        Some(v) if v.is_integer() => Some(" ".repeat(v.as_i64().unwrap_or(0).max(0) as usize)),
        Some(v) if v.kind() == ValueKind::String => Some(v.as_str().unwrap_or("").to_owned()),
        Some(v) => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!(
                    "tojson: indent must be an integer or a string, got {}",
                    v.kind()
                ),
            ));
        }
    };
    let separators = pick(separators, "separators")?;
    // Python: with an indent the DEFAULT item separator drops its space
    let default_item = if indent.is_some() { "," } else { ", " };
    let (item_sep, key_sep) = match separators {
        Some(s) if !(s.is_none() || s.is_undefined()) => {
            let parts: Vec<String> = s
                .try_iter()?
                .map(|p| p.as_str().map(str::to_owned))
                .collect::<Option<_>>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson: separators must be strings",
                    )
                })?;
            if parts.len() != 2 {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson: separators must be an (item, key) pair",
                ));
            }
            (parts[0].clone(), parts[1].clone())
        }
        _ => (default_item.to_owned(), ": ".to_owned()),
    };
    let sort_keys = pick(sort_keys, "sort_keys")?.is_some_and(|v| v.is_true());
    kwargs.assert_all_used()?;
    let opts = Opts {
        ensure_ascii,
        indent,
        item_sep,
        key_sep,
        sort_keys,
    };
    let mut out = String::new();
    dump(&value, &opts, 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

struct Opts {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

fn newline(opts: &Opts, level: usize, out: &mut String) {
    if let Some(ind) = &opts.indent {
        out.push('\n');
        for _ in 0..level {
            out.push_str(ind);
        }
    }
}

fn dump(v: &Value, opts: &Opts, level: usize, out: &mut String) -> Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined | ValueKind::None => out.push_str("null"),
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if v.is_integer() {
                out.push_str(&v.to_string());
            } else {
                out.push_str(&py_float(f64::try_from(v.clone())?));
            }
        }
        ValueKind::String => py_string(v.as_str().unwrap_or(""), opts.ensure_ascii, out),
        ValueKind::Map => {
            let mut keys: Vec<Value> = v.try_iter()?.collect();
            if keys.is_empty() {
                out.push_str("{}");
                return Ok(());
            }
            if opts.sort_keys {
                keys.sort_by_key(key_text);
            }
            out.push('{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                newline(opts, level + 1, out);
                py_string(&key_text(k), opts.ensure_ascii, out);
                out.push_str(&opts.key_sep);
                dump(&v.get_item(k)?, opts, level + 1, out)?;
            }
            newline(opts, level, out);
            out.push('}');
        }
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<Value> = v.try_iter()?.collect();
            if items.is_empty() {
                out.push_str("[]");
                return Ok(());
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&opts.item_sep);
                }
                newline(opts, level + 1, out);
                dump(item, opts, level + 1, out)?;
            }
            newline(opts, level, out);
            out.push(']');
        }
        // bytes and plain objects have no JSON form in Python either
        _ => {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson: a {} value is not JSON serializable", v.kind()),
            ));
        }
    }
    Ok(())
}

/// A mapping key as Python writes it: strings as they are, numbers, bools and
/// None stringified the way `json.dumps` coerces them.
fn key_text(k: &Value) -> String {
    match k.kind() {
        ValueKind::String => k.as_str().unwrap_or("").to_owned(),
        ValueKind::Bool => (if k.is_true() { "true" } else { "false" }).to_owned(),
        ValueKind::None | ValueKind::Undefined => "null".to_owned(),
        ValueKind::Number if !k.is_integer() => f64::try_from(k.clone())
            .map(py_float)
            .unwrap_or_else(|_| k.to_string()),
        _ => k.to_string(),
    }
}

/// `json.dumps` string escaping: `"`, `\` and the C0 controls (with the five
/// short forms), and - under `ensure_ascii` - everything outside printable
/// ASCII as `\uXXXX`, astral code points as a surrogate pair. Lowercase hex,
/// as Python writes it.
fn py_string(s: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest round-trip digits, fixed notation
/// while the decimal exponent sits in [-4, 16), else `d.ddde+XX` with at
/// least two exponent digits; an integral value keeps its `.0`.
fn py_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_owned();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    // Rust's LowerExp is the shortest round-trip form: "1.2345e-5"
    let e = format!("{f:e}");
    let (mant, exp) = e
        .split_once('e')
        .expect("LowerExp always carries an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent is an integer");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut s = String::new();
    if neg {
        s.push('-');
    }
    if (-4..16).contains(&exp) {
        // fixed: the point sits exp + 1 digits into `digits`
        let point = exp + 1;
        if point <= 0 {
            s.push_str("0.");
            for _ in 0..(-point) {
                s.push('0');
            }
            s.push_str(&digits);
        } else {
            let p = point as usize;
            if digits.len() <= p {
                s.push_str(&digits);
                for _ in digits.len()..p {
                    s.push('0');
                }
                s.push_str(".0");
            } else {
                s.push_str(&digits[..p]);
                s.push('.');
                s.push_str(&digits[p..]);
            }
        }
    } else {
        s.push_str(&digits[..1]);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if exp < 0 { '-' } else { '+' });
        s.push_str(&format!("{:02}", exp.abs()));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::py_float;

    #[test]
    fn floats_print_as_python_repr() {
        for (f, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-2.5, "-2.5"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e100, "1.2345e+100"),
            (123.456, "123.456"),
            (-0.0, "-0.0"),
            (f64::INFINITY, "Infinity"),
        ] {
            assert_eq!(py_float(f), want, "{f}");
        }
    }
}
