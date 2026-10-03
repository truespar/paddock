//! JSON the way Python holds and writes it - because the decision models'
//! sequences are built from Python's text of the request, not from ours.
//!
//! Laya reads a structured `state` as `json.dumps(state, ensure_ascii=False)`,
//! a dict-valued criterion as the same with the same separators, a non-string
//! instruction likewise, and a list label as `str(label)`; Clef renders every
//! non-string value with `sort_keys=True` and the compact `(",", ":")`
//! separators. Every byte of those strings is a token the model reads, so
//! "equivalent JSON" is not good enough: the key order must be the caller's
//! (Python dicts keep insertion order; `serde_json::Map` sorts), `12.50`
//! must come out `12.5`, `1e16` as `1e+16`, `True` as `true` in JSON and
//! `True` as a label, and escapes exactly as Python's encoder writes them.
//!
//! So this is a small order-preserving parser over the request's own bytes
//! and a writer that follows CPython's `json.encoder` and `float.__repr__`.
//! It only ever sees text serde_json has already accepted.

use std::fmt::Write as _;

/// A JSON value as `json.loads` would hand it to Laya.
#[derive(Debug, Clone, PartialEq)]
pub enum PyVal {
    Null,
    Bool(bool),
    /// an integer literal: Python keeps it exact at any size, so the digits
    /// are kept as written ("-0" normalised to "0", as `int` does)
    Int(String),
    Float(f64),
    Str(String),
    List(Vec<PyVal>),
    /// insertion order; a repeated key keeps its FIRST position and its LAST
    /// value, as a Python dict built by `json.loads` does
    Dict(Vec<(String, PyVal)>),
}

impl PyVal {
    pub fn get(&self, key: &str) -> Option<&PyVal> {
        match self {
            PyVal::Dict(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyVal::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `json.dumps(v, ensure_ascii=False)` - default separators `", "` and
    /// `": "` (which is also what `separators=(", ", ": ")` spells).
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out, false);
        out
    }

    /// `json.dumps(v, ensure_ascii=False, separators=(",", ":"),
    /// sort_keys=True)`. Python sorts the keys as strings, i.e. by code
    /// point, which is the byte order of their UTF-8.
    pub fn dumps_sorted_compact(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out, true);
        out
    }

    fn write_json(&self, out: &mut String, sorted_compact: bool) {
        let (item_sep, key_sep) = if sorted_compact {
            (",", ":")
        } else {
            (", ", ": ")
        };
        match self {
            PyVal::Null => out.push_str("null"),
            PyVal::Bool(true) => out.push_str("true"),
            PyVal::Bool(false) => out.push_str("false"),
            PyVal::Int(d) => out.push_str(d),
            PyVal::Float(f) => out.push_str(&float_json(*f)),
            PyVal::Str(s) => write_str(s, out),
            PyVal::List(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(item_sep);
                    }
                    v.write_json(out, sorted_compact);
                }
                out.push(']');
            }
            PyVal::Dict(kv) => {
                let mut order: Vec<&(String, PyVal)> = kv.iter().collect();
                if sorted_compact {
                    order.sort_by(|a, b| a.0.cmp(&b.0));
                }
                out.push('{');
                for (i, (k, v)) in order.into_iter().enumerate() {
                    if i > 0 {
                        out.push_str(item_sep);
                    }
                    write_str(k, out);
                    out.push_str(key_sep);
                    v.write_json(out, sorted_compact);
                }
                out.push('}');
            }
        }
    }

    /// `str(v)` - how a list label becomes option text. Containers are
    /// refused before this is asked (a label must be a scalar).
    pub fn py_str(&self) -> String {
        match self {
            PyVal::Null => "None".into(),
            PyVal::Bool(true) => "True".into(),
            PyVal::Bool(false) => "False".into(),
            PyVal::Int(d) => d.clone(),
            PyVal::Float(f) => float_repr(*f),
            PyVal::Str(s) => s.clone(),
            other => other.dumps(),
        }
    }

    /// The same value as serde_json holds it, for answers that echo a label.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            PyVal::Null => serde_json::Value::Null,
            PyVal::Bool(b) => serde_json::Value::Bool(*b),
            PyVal::Int(d) => d
                .parse::<i64>()
                .map(serde_json::Value::from)
                .or_else(|_| d.parse::<u64>().map(serde_json::Value::from))
                .unwrap_or_else(|_| serde_json::Value::String(d.clone())),
            PyVal::Float(f) => serde_json::Number::from_f64(*f)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            PyVal::Str(s) => serde_json::Value::String(s.clone()),
            PyVal::List(items) => {
                serde_json::Value::Array(items.iter().map(PyVal::to_json).collect())
            }
            PyVal::Dict(kv) => serde_json::Value::Object(
                kv.iter().map(|(k, v)| (k.clone(), v.to_json())).collect(),
            ),
        }
    }
}

/// CPython's `encode_basestring` (the ensure_ascii=False one): `"` and `\`
/// escaped, `\n \r \t \b \f` by name, every other control below 0x20 as a
/// lower-case `\u00XX`; nothing else - not DEL, not U+2028.
fn write_str(s: &str, out: &mut String) {
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
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `json.dumps` of a float: `float.__repr__`, and the three non-finite
/// spellings Python's encoder allows by default.
fn float_json(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f == f64::INFINITY {
        "Infinity".into()
    } else if f == f64::NEG_INFINITY {
        "-Infinity".into()
    } else {
        float_repr(f)
    }
}

/// `float.__repr__`: the shortest digits that round-trip, then CPython's
/// 'r' layout - scientific when the decimal exponent is below -4 or at least
/// 16 (`1e-05`, `1e+16`, two exponent digits minimum), positional otherwise,
/// and a positional integer keeps its `.0`.
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` is the shortest round-trip digits in scientific form
    let sci = format!("{:e}", f.abs());
    let (mant, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("an integer exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let (first, rest) = digits.split_at(1);
        let m = if rest.is_empty() {
            first.to_owned()
        } else {
            format!("{first}.{rest}")
        };
        let es = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{m}e{es}{:02}", exp.abs());
    }
    // positional: the point sits after digit (exp + 1)
    let point = exp + 1;
    let s = if point <= 0 {
        format!("0.{}{digits}", "0".repeat((-point) as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        let (a, b) = digits.split_at(point as usize);
        format!("{a}.{b}")
    };
    format!("{sign}{s}")
}

/// Parse JSON text, keeping object key order. `None` on anything malformed -
/// the text has already passed serde_json, so that is a logic error upstream.
pub fn parse(text: &str) -> Option<PyVal> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.ws();
    (p.i == p.s.len()).then_some(v)
}

/// The value at `key` of the top-level object in `body`, order kept.
pub fn field(body: &[u8], key: &str) -> Option<PyVal> {
    let text = std::str::from_utf8(body).ok()?;
    match parse(text)? {
        PyVal::Dict(kv) => kv.into_iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
        _ => None,
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    const MAX_DEPTH: usize = 512;

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &[u8]) -> bool {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Option<PyVal> {
        self.ws();
        match *self.s.get(self.i)? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(PyVal::Str),
            b't' => self.eat(b"true").then_some(PyVal::Bool(true)),
            b'f' => self.eat(b"false").then_some(PyVal::Bool(false)),
            b'n' => self.eat(b"null").then_some(PyVal::Null),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Option<PyVal> {
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            return None;
        }
        self.i += 1;
        let mut kv: Vec<(String, PyVal)> = Vec::new();
        self.ws();
        if self.eat(b"}") {
            self.depth -= 1;
            return Some(PyVal::Dict(kv));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            if !self.eat(b":") {
                return None;
            }
            let v = self.value()?;
            match kv.iter_mut().find(|(e, _)| *e == k) {
                Some(slot) => slot.1 = v,
                None => kv.push((k, v)),
            }
            self.ws();
            if self.eat(b",") {
                continue;
            }
            if self.eat(b"}") {
                self.depth -= 1;
                return Some(PyVal::Dict(kv));
            }
            return None;
        }
    }

    fn array(&mut self) -> Option<PyVal> {
        self.depth += 1;
        if self.depth > Self::MAX_DEPTH {
            return None;
        }
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.eat(b"]") {
            self.depth -= 1;
            return Some(PyVal::List(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            if self.eat(b",") {
                continue;
            }
            if self.eat(b"]") {
                self.depth -= 1;
                return Some(PyVal::List(items));
            }
            return None;
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = std::str::from_utf8(self.s.get(self.i..self.i + 4)?).ok()?;
        let v = u32::from_str_radix(h, 16).ok()?;
        self.i += 4;
        Some(v)
    }

    fn string(&mut self) -> Option<String> {
        if !self.eat(b"\"") {
            return None;
        }
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.s.len() && self.s[self.i] != b'"' && self.s[self.i] != b'\\' {
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.s[start..self.i]).ok()?);
            match *self.s.get(self.i)? {
                b'"' => {
                    self.i += 1;
                    return Some(out);
                }
                _ => {
                    self.i += 1;
                    let e = *self.s.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi)
                                && self.s[self.i..].starts_with(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    self.i = save;
                                    hi
                                }
                            } else {
                                hi
                            };
                            // a lone surrogate has no Rust char; Python would
                            // carry it and its tokenizer could not encode it
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return None,
                    }
                }
            }
        }
    }

    fn number(&mut self) -> Option<PyVal> {
        let start = self.i;
        let mut float = false;
        while self.i < self.s.len() {
            match self.s[self.i] {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => float = true,
                _ => break,
            }
            self.i += 1;
        }
        let lit = std::str::from_utf8(&self.s[start..self.i]).ok()?;
        if lit.is_empty() {
            return None;
        }
        if float {
            // `float(lit)` - an overflow is inf in Python too
            lit.parse::<f64>().ok().map(PyVal::Float)
        } else {
            let lit = if lit == "-0" { "0" } else { lit };
            Some(PyVal::Int(lit.to_owned()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_repr() {
        for (f, want) in [
            (12.5, "12.5"),
            (100.0, "100.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (1e22, "1e+22"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (-2.5e-7, "-2.5e-07"),
            (1e100, "1e+100"),
            (-0.0, "-0.0"),
            (5e-324, "5e-324"),
            (123456.789, "123456.789"),
        ] {
            assert_eq!(float_repr(f), want, "{f:e}");
        }
    }

    #[test]
    fn dumps_follows_the_python_encoder() {
        let v = parse(
            r#"{"z": 1, "a": [1, 2.0, "x"], "big": 10000000000000000, "e": 1e16,
               "s": "tab\there \"q\" \u00e9 \ud83d\ude00 \u0001", "n": null, "t": true, "neg": -0}"#,
        )
        .unwrap();
        assert_eq!(
            v.dumps(),
            "{\"z\": 1, \"a\": [1, 2.0, \"x\"], \"big\": 10000000000000000, \"e\": 1e+16, \
             \"s\": \"tab\\there \\\"q\\\" \u{e9} \u{1F600} \\u0001\", \"n\": null, \"t\": true, \
             \"neg\": 0}"
        );
        // a repeated key keeps its first place and its last value
        assert_eq!(
            parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap().dumps(),
            "{\"a\": 3, \"b\": 2}"
        );
        assert_eq!(parse("[]").unwrap().dumps(), "[]");
        assert_eq!(parse("{}").unwrap().dumps(), "{}");
    }

    #[test]
    fn sorted_compact_follows_sort_keys() {
        let v = parse(
            r#"{"z": 1e21, "a": [1.5, -0.0, 3, true, null, "\u00f6"], "\u00c4": "x",
               "m": {"k": 0.1, "big": 12345678901234567890, "small": 1e-7}}"#,
        )
        .unwrap();
        assert_eq!(
            v.dumps_sorted_compact(),
            "{\"a\":[1.5,-0.0,3,true,null,\"\u{f6}\"],\"m\":{\"big\":12345678901234567890,\
             \"k\":0.1,\"small\":1e-07},\"z\":1e+21,\"\u{c4}\":\"x\"}"
        );
    }

    #[test]
    fn labels_are_python_str() {
        let v = parse(r#"[1, 2.5, "x", null, true, 1e2]"#).unwrap();
        let PyVal::List(items) = v else { panic!() };
        let s: Vec<String> = items.iter().map(PyVal::py_str).collect();
        assert_eq!(s, ["1", "2.5", "x", "None", "True", "100.0"]);
    }

    #[test]
    fn field_reads_the_top_level_value() {
        let body = br#"{"questions": {}, "state": {"b": 1, "a": 2}}"#;
        assert_eq!(
            field(body, "state").unwrap().dumps(),
            "{\"b\": 1, \"a\": 2}"
        );
        assert!(field(body, "nope").is_none());
    }
}
