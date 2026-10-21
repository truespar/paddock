//! One pass over a batch's input: validate every line and record where each
//! request body sits, so the run can later read a single item by offset and
//! send its body verbatim.
//!
//! What fails the batch here is what OpenAI fails at validation: a line that
//! is not JSON, a missing or repeated custom_id, a url or model other than the
//! batch's, a streaming request. Problems inside a request (a picture the
//! model cannot read, a prompt over the context) are the item's own and
//! surface as that item's error when it runs. Better to refuse a 10 000-line
//! file in a minute than to find line 9 999 malformed six hours in.
//!
//! Memory is bounded by the longest line ([`super::MAX_LINE`]): the base64
//! inside a body is scanned, never materialized into a `Value`.

use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use serde::Deserialize;
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::Value;
use serde_json::value::RawValue;

use super::{MAX_LINE, MAX_REQUESTS};

/// An index row: (idx, custom_id, body offset, body length).
pub(crate) type Row = (i64, String, i64, i64);

/// Stop listing errors after this many; the count is still reported.
const MAX_ERRORS: usize = 100;

/// Rows are handed to the store in chunks this size (one transaction each).
pub(crate) const CHUNK: usize = 2_000;

#[derive(Debug, Default)]
pub(crate) struct Outcome {
    pub total: i64,
    /// OpenAI's `errors.data` entries; empty means the input is valid.
    pub errors: Vec<Value>,
}

#[derive(Deserialize)]
struct LineProbe<'a> {
    #[serde(borrow, default)]
    custom_id: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    method: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    url: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    body: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct BodyProbe<'a> {
    #[serde(borrow, default)]
    model: Option<Cow<'a, str>>,
    #[serde(default)]
    stream: Option<bool>,
}

fn err(code: &str, line: usize, message: String) -> Value {
    serde_json::json!({"code": code, "message": message, "param": null, "line": line})
}

/// Scan `path`, a JSONL of batch lines, for a batch on `endpoint`. `model` is
/// the one the batch was created for (from its first line); every line must
/// name it. Rows are handed to `sink` in chunks of [`CHUNK`] as they are found,
/// so a valid 100 000-line index is never held whole.
pub(crate) fn index_jsonl(
    path: &Path,
    endpoint: &str,
    model: &str,
    mut sink: impl FnMut(&[Row]) -> Result<(), String>,
) -> Result<Outcome, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut out = Outcome::default();
    let mut seen: HashSet<String> = HashSet::new();
    let mut chunk: Vec<Row> = Vec::with_capacity(CHUNK);
    let mut buf: Vec<u8> = Vec::new();
    let mut off: u64 = 0;
    let mut lineno = 0usize;
    let mut extra_errors = 0usize;
    let mut push_err = |out: &mut Outcome, e: Value| {
        if out.errors.len() < MAX_ERRORS {
            out.errors.push(e);
        } else {
            extra_errors += 1;
        }
    };
    loop {
        buf.clear();
        let start = off;
        let (n, too_long) = read_line_capped(&mut r, &mut buf, MAX_LINE)?;
        if n == 0 {
            break;
        }
        off += n as u64;
        lineno += 1;
        if too_long {
            push_err(
                &mut out,
                err(
                    "line_too_large",
                    lineno,
                    format!(
                        "line {lineno} is over the {} MB request limit",
                        MAX_LINE >> 20
                    ),
                ),
            );
            continue;
        }
        let text = trim_line(&buf);
        if text.is_empty() {
            continue;
        }
        let probe: LineProbe<'_> = match serde_json::from_slice(text) {
            Ok(p) => p,
            Err(e) => {
                push_err(
                    &mut out,
                    err(
                        "invalid_json_line",
                        lineno,
                        format!("line {lineno} is not valid JSON: {e}"),
                    ),
                );
                continue;
            }
        };
        let Some(cid) = probe.custom_id.filter(|c| !c.is_empty()) else {
            push_err(
                &mut out,
                err(
                    "missing_custom_id",
                    lineno,
                    format!("line {lineno} has no custom_id"),
                ),
            );
            continue;
        };
        if probe.method.as_deref() != Some("POST") {
            push_err(
                &mut out,
                err(
                    "invalid_method",
                    lineno,
                    format!("line {lineno}: method must be \"POST\""),
                ),
            );
        }
        if probe.url.as_deref() != Some(endpoint) {
            push_err(
                &mut out,
                err(
                    "mismatched_endpoint",
                    lineno,
                    format!(
                        "line {lineno}: url {:?} does not match the batch's endpoint {endpoint}",
                        probe.url.as_deref().unwrap_or("")
                    ),
                ),
            );
        }
        let Some(body) = probe.body.filter(|b| b.get().starts_with('{')) else {
            push_err(
                &mut out,
                err(
                    "invalid_body",
                    lineno,
                    format!("line {lineno}: body must be a JSON object"),
                ),
            );
            continue;
        };
        match serde_json::from_str::<BodyProbe<'_>>(body.get()) {
            Ok(b) => {
                let m = b.model.as_deref().unwrap_or("");
                if !super::same_model(m, model) {
                    push_err(
                        &mut out,
                        err(
                            "mismatched_model",
                            lineno,
                            format!(
                                "line {lineno}: model {m:?} differs from the batch's {model:?} - \
                                 one batch runs on one model"
                            ),
                        ),
                    );
                }
                if b.stream == Some(true) {
                    push_err(
                        &mut out,
                        err(
                            "stream_not_supported",
                            lineno,
                            format!("line {lineno}: a batch request cannot stream"),
                        ),
                    );
                }
            }
            Err(e) => push_err(
                &mut out,
                err("invalid_body", lineno, format!("line {lineno}: body: {e}")),
            ),
        }
        if !seen.insert(cid.to_string()) {
            push_err(
                &mut out,
                err(
                    "duplicate_custom_id",
                    lineno,
                    format!("line {lineno}: custom_id {cid:?} appears more than once"),
                ),
            );
            continue;
        }
        if seen.len() > MAX_REQUESTS {
            push_err(
                &mut out,
                err(
                    "too_many_requests",
                    lineno,
                    format!("a batch holds at most {MAX_REQUESTS} requests"),
                ),
            );
            break;
        }
        // the body's place in the file: its slice of the line buffer, shifted
        // by where the line started (both point into `buf`)
        let rel = body.get().as_ptr() as usize - buf.as_ptr() as usize;
        if out.errors.is_empty() {
            chunk.push((
                out.total,
                cid.into_owned(),
                (start + rel as u64) as i64,
                body.get().len() as i64,
            ));
            if chunk.len() == CHUNK {
                sink(&chunk)?;
                chunk.clear();
            }
        }
        out.total += 1;
    }
    if out.total == 0 && out.errors.is_empty() {
        out.errors.push(err(
            "empty_file",
            0,
            "the input file has no requests".into(),
        ));
    }
    if extra_errors > 0 {
        out.errors.push(err(
            "more_errors",
            lineno,
            format!("{extra_errors} more errors not listed"),
        ));
    }
    if out.errors.is_empty() && !chunk.is_empty() {
        sink(&chunk)?;
    }
    Ok(out)
}

/// The model named by the first request line, read without scanning the file.
pub(crate) fn first_model(path: &Path) -> Result<Option<String>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        let (n, too_long) = read_line_capped(&mut r, &mut buf, MAX_LINE)?;
        if n == 0 || too_long {
            return Ok(None);
        }
        let text = trim_line(&buf);
        if text.is_empty() {
            continue;
        }
        let Ok(p) = serde_json::from_slice::<LineProbe<'_>>(text) else {
            return Ok(None);
        };
        return Ok(p
            .body
            .and_then(|b| serde_json::from_str::<BodyProbe<'_>>(b.get()).ok())
            .and_then(|b| b.model.map(Cow::into_owned)));
    }
}

/// A line without its terminator or surrounding blanks.
fn trim_line(buf: &[u8]) -> &[u8] {
    let s = buf
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(buf.len());
    let e = buf
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(s, |i| i + 1);
    &buf[s..e.max(s)]
}

/// Read one line into `buf` (newline included), never holding more than `cap`
/// bytes of it. Returns the bytes consumed and whether the line was over the
/// cap (then `buf` holds only its start and the rest was skipped).
fn read_line_capped<R: BufRead>(
    r: &mut R,
    buf: &mut Vec<u8>,
    cap: usize,
) -> Result<(usize, bool), String> {
    let mut n = 0usize;
    let mut over = false;
    loop {
        let avail = r.fill_buf().map_err(|e| e.to_string())?;
        if avail.is_empty() {
            return Ok((n, over));
        }
        let (take, done) = match avail.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (avail.len(), false),
        };
        if !over {
            if buf.len() + take > cap {
                over = true;
            } else {
                buf.extend_from_slice(&avail[..take]);
            }
        }
        r.consume(take);
        n += take;
        if done {
            return Ok((n, over));
        }
    }
}

/// An Anthropic create body, converted to batch lines as it is parsed.
pub(crate) struct Converted {
    pub total: i64,
    pub model: String,
    /// The `paddock` extension object, if the body carried one.
    pub paddock: Value,
}

/// Parse an Anthropic `POST /v1/messages/batches` body from `src` and write
/// each request to `dst` as a batch line (`{custom_id, method, url, body}`), one
/// element of `requests` in memory at a time. Errors are the request's 400.
pub(crate) fn convert_anthropic(src: impl Read, dst: impl Write) -> Result<Converted, String> {
    let mut de = serde_json::Deserializer::from_reader(BufReader::with_capacity(1 << 20, src));
    let mut sink = LineSink {
        w: std::io::BufWriter::with_capacity(1 << 20, dst),
        seen: HashSet::new(),
        total: 0,
        model: None,
        paddock: Value::Null,
        failure: None,
    };
    let parsed = CreateSeed(&mut sink).deserialize(&mut de);
    if let Some(f) = sink.failure.take() {
        return Err(f);
    }
    parsed.map_err(|e| format!("invalid request body: {e}"))?;
    de.end().map_err(|e| format!("invalid request body: {e}"))?;
    sink.w.flush().map_err(|e| e.to_string())?;
    if sink.total == 0 {
        return Err("requests: at least one request is required".into());
    }
    Ok(Converted {
        total: sink.total,
        model: sink.model.unwrap_or_default(),
        paddock: sink.paddock,
    })
}

struct LineSink<W: Write> {
    w: std::io::BufWriter<W>,
    seen: HashSet<String>,
    total: i64,
    model: Option<String>,
    paddock: Value,
    /// A validation failure, kept verbatim (serde would otherwise prefix it
    /// with a position nobody asked for).
    failure: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicRequest {
    custom_id: String,
    params: Box<RawValue>,
}

impl<W: Write> LineSink<W> {
    fn push(&mut self, r: AnthropicRequest) -> Result<(), String> {
        let i = self.total;
        let valid_id = (1..=64).contains(&r.custom_id.len())
            && r.custom_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if !valid_id {
            return Err(format!(
                "requests.{i}.custom_id: {:?} must match ^[a-zA-Z0-9_-]{{1,64}}$",
                r.custom_id
            ));
        }
        if !self.seen.insert(r.custom_id.clone()) {
            return Err(format!(
                "requests.{i}.custom_id: {:?} is used more than once",
                r.custom_id
            ));
        }
        if self.seen.len() > MAX_REQUESTS {
            return Err(format!(
                "requests: a batch holds at most {MAX_REQUESTS} requests"
            ));
        }
        let raw = r.params.get();
        if !raw.starts_with('{') {
            return Err(format!("requests.{i}.params: must be an object"));
        }
        let probe: BodyProbe<'_> =
            serde_json::from_str(raw).map_err(|e| format!("requests.{i}.params: {e}"))?;
        let m = probe
            .model
            .as_deref()
            .filter(|m| !m.is_empty())
            .ok_or_else(|| format!("requests.{i}.params.model: Field required"))?;
        match &self.model {
            None => self.model = Some(m.to_owned()),
            Some(first) if !super::same_model(first, m) => {
                return Err(format!(
                    "requests.{i}.params.model: {m:?} differs from {first:?} - one batch runs on \
                     one model"
                ));
            }
            Some(_) => {}
        }
        if probe.stream == Some(true) {
            return Err(format!(
                "requests.{i}.params.stream: a batch request cannot stream"
            ));
        }
        // a pretty-printed request would break the one-line framing: those are
        // re-serialized compact, everything else is copied byte for byte
        let compact;
        let body: &str = if raw.contains(['\n', '\r']) {
            let v: Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
            compact = v.to_string();
            &compact
        } else {
            raw
        };
        let cid = serde_json::to_string(&r.custom_id).map_err(|e| e.to_string())?;
        let io = |e: std::io::Error| format!("writing the batch input: {e}");
        self.w
            .write_all(b"{\"custom_id\":")
            .and_then(|_| self.w.write_all(cid.as_bytes()))
            .and_then(|_| {
                self.w
                    .write_all(b",\"method\":\"POST\",\"url\":\"/v1/messages\",\"body\":")
            })
            .and_then(|_| self.w.write_all(body.as_bytes()))
            .and_then(|_| self.w.write_all(b"}\n"))
            .map_err(io)?;
        self.total += 1;
        Ok(())
    }
}

struct CreateSeed<'s, W: Write>(&'s mut LineSink<W>);

impl<'de, W: Write> DeserializeSeed<'de> for CreateSeed<'_, W> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_map(self)
    }
}

impl<'de, W: Write> Visitor<'de> for CreateSeed<'_, W> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an object with a requests array")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut saw = false;
        while let Some(key) = map.next_key::<Cow<'de, str>>()? {
            match key.as_ref() {
                "requests" => {
                    saw = true;
                    map.next_value_seed(RequestsSeed(&mut *self.0))?;
                }
                "paddock" => self.0.paddock = map.next_value::<Value>()?,
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        if !saw {
            self.0.failure = Some("requests: Field required".into());
            return Err(serde::de::Error::custom("requests: Field required"));
        }
        Ok(())
    }
}

struct RequestsSeed<'s, W: Write>(&'s mut LineSink<W>);

impl<'de, W: Write> DeserializeSeed<'de> for RequestsSeed<'_, W> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de, W: Write> Visitor<'de> for RequestsSeed<'_, W> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an array of {custom_id, params}")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while let Some(r) = seq.next_element::<AnthropicRequest>()? {
            if let Err(e) = self.0.push(r) {
                self.0.failure = Some(e.clone());
                return Err(serde::de::Error::custom(e));
            }
        }
        Ok(())
    }
}
