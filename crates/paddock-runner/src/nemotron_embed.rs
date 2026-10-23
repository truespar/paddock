//! Nemotron 3.5 Embed 8B / Rerank 8B: the checkpoints' input contracts.
//! Computation stays in the engine (`gpu_model::nemotron_embed`) behind the
//! shared encoder scheduler; this module renders requests exactly as each
//! checkpoint's `chat_template.jinja` and Pixtral processor would.
//!
//! - **Embed** (`/v1/embeddings`): an optional system prefix (`task`:
//!   `query` -> `query:`, `document` / `passage` -> `passage:`) then the
//!   user content - every picture placeholder first, space-joined, then the
//!   text parts, space-joined. No BOS / EOS.
//! - **Rerank** (`/v1/rerank`): `[IMG] ` when the document has a picture (at
//!   most one), then `query: {query}\n\npassage: {document}` - the document's
//!   text parts joined by a blank line - and the whole thing stripped when a
//!   picture leads it. The score is the head's raw logit (the release's
//!   Sentence Transformers head is `Identity`): higher is more relevant.
//!
//! User text is marked the way both templates mark it: a U+200C after every
//! `[` and `<`, so no text can spell a special token. A picture expands to the
//! processor's blocks: per merged row its `[IMG]` tokens and an
//! `[IMG_BREAK]`, the last one `[IMG_END]`; the tokenizer splits at special
//! tokens, so the text between pictures tokenizes as runs of its own, which
//! is what Hugging Face does too.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use paddock_engine::service::MmChunk;
use paddock_models::nemotron_embed::{Kind, NemotronEmbedConfig};
use serde_json::Value;

use paddock_tokenizer::GgufTokenizer;

use crate::serving::{EmbedModel, ServeError};

/// The family's serving facts the handlers need.
pub struct Family {
    pub cfg: NemotronEmbedConfig,
    pub img: u32,
    pub brk: u32,
    pub end: u32,
    pub pictures: bool,
}

impl Family {
    pub fn kind(&self) -> Kind {
        self.cfg.kind
    }
}

/// The checkpoint directory for `path` (the directory itself or a
/// `.safetensors` file inside it), when its config names this family.
pub(crate) fn directory(path: &Path) -> Option<PathBuf> {
    let dir = if path.is_dir() {
        path
    } else if path.extension().is_some_and(|x| x == "safetensors") {
        path.parent()?
    } else {
        return None;
    };
    NemotronEmbedConfig::is_ours(dir).then(|| dir.to_path_buf())
}

/// Load either checkpoint on CUDA; `pictures` builds the Pixtral tower.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load(
    id: String,
    dir: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    context: usize,
    budget: Option<u64>,
    pictures: bool,
) -> Result<EmbedModel, ServeError> {
    if device != "cuda" {
        return Err(ServeError::Engine(format!(
            "Nemotron 3.5 Embed / Rerank is served on CUDA (got {device:?})"
        )));
    }
    let cfg =
        NemotronEmbedConfig::read(dir).map_err(|e| ServeError::Open(dir.into(), e.to_string()))?;
    let tokenizer =
        GgufTokenizer::from_hf_dir(dir).map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    let id_of = |t: &str| {
        tokenizer
            .token_to_id(t)
            .ok_or_else(|| ServeError::Tokenizer(format!("the tokenizer has no {t}")))
    };
    let (img, brk, end) = (id_of("[IMG]")?, id_of("[IMG_BREAK]")?, id_of("[IMG_END]")?);
    if img != cfg.image_token_id {
        return Err(ServeError::Tokenizer(format!(
            "[IMG] is {img} in the tokenizer, {} in the config",
            cfg.image_token_id
        )));
    }
    let context = context.clamp(1, cfg.max_ctx);
    let metrics = Arc::new(paddock_engine::metrics::EngineMetrics::default());
    let kind = cfg.kind;
    let (d, p) = (dir.to_path_buf(), pack.map(Path::to_path_buf));
    let encoder = paddock_engine::encoder::Encoder::spawn(
        move || {
            let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, p.as_deref())
                .map_err(|e| e.to_string())?;
            crate::serving::note_device_cc(&exec);
            if let Some(b) = budget {
                exec.set_vram_budget(b);
            }
            paddock_engine::gpu_model::nemotron_embed::GpuNemotronEmbed::load_dir(
                Arc::new(exec),
                &d,
                context,
                pictures,
            )
            .map_err(|e| e.to_string())
        },
        Some(metrics.clone()),
    )
    .map_err(ServeError::Engine)?;
    let pictures = encoder.serves_images();
    Ok(EmbedModel {
        id,
        encoder,
        tokenizer: Arc::new(tokenizer),
        eos: None,
        yes_id: None,
        no_id: None,
        is_reranker: kind == Kind::Rerank,
        embedding_gemma2: false,
        max_input_tokens: context,
        image_budget: None,
        audio: false,
        nemotron: Some(Family {
            cfg,
            img,
            brk,
            end,
            pictures,
        }),
        metrics,
    })
}

/// The embedder's system prefix for a `task`.
pub(crate) fn task_prefix(task: &str) -> Result<&'static str, String> {
    Ok(match task {
        "query" | "search_query" => "query:",
        "document" | "passage" | "search_document" => "passage:",
        other => {
            return Err(format!(
                "unknown Nemotron 3.5 Embed task {other:?} (query, document or passage)"
            ));
        }
    })
}

/// Both templates' user-text marking: U+200C after every `[` and `<`.
pub(crate) fn mark(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        out.push(ch);
        if ch == '[' || ch == '<' {
            out.push('\u{200c}');
        }
    }
    out
}

/// Python's `str.isspace` (Jinja's `trim` is `str.strip`): Rust's
/// `char::is_whitespace` plus the four ASCII separators U+001C..U+001F.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// One rendered input in order: text runs and pictures.
enum Piece {
    Text(String),
    Picture { rgb: Vec<u8>, w: usize, h: usize },
}

/// An item's parts: texts and pictures, in the order sent.
struct Parts {
    texts: Vec<String>,
    pictures: Vec<(Vec<u8>, usize, usize)>,
}

/// Decode an item's content parts (`text` / `input_text`, `image_url` /
/// `input_image` with a data: URI, decoded as the reference's Pillow does).
fn parts_of(parts: &[&Value], pixels_left: &mut u64, pictures: bool) -> Result<Parts, String> {
    let mut out = Parts {
        texts: Vec::new(),
        pictures: Vec::new(),
    };
    for part in parts {
        if let Value::String(s) = part {
            out.texts.push(s.clone());
            continue;
        }
        match part.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" | "input_text" => out.texts.push(
                part.get("text")
                    .and_then(Value::as_str)
                    .ok_or("a text part needs `text`")?
                    .to_owned(),
            ),
            "image_url" | "input_image" => {
                if !pictures {
                    return Err("this endpoint serves no pictures".into());
                }
                let v = part.get("image_url");
                let url = v
                    .and_then(|v| v.get("url").and_then(Value::as_str).or(v.as_str()))
                    .ok_or("an image part needs `image_url`")?;
                let detail = v
                    .and_then(|v| v.get("detail"))
                    .or(part.get("detail"))
                    .and_then(Value::as_str);
                if detail.is_some_and(|d| d != "auto") {
                    return Err(format!(
                        "image detail {detail:?} is not served: pictures take the processor's \
                         longest edge"
                    ));
                }
                out.pictures.push(decode(url, pixels_left)?);
            }
            other => {
                return Err(format!(
                    "content part type {other:?} is not served (text, image_url)"
                ));
            }
        }
    }
    Ok(out)
}

fn decode(url: &str, pixels_left: &mut u64) -> Result<(Vec<u8>, usize, usize), String> {
    let (rgb, w, h) =
        crate::reference_image::decode_image_url_reference_limited(url, *pixels_left)?;
    *pixels_left = pixels_left.saturating_sub((w * h) as u64);
    Ok((rgb, w, h))
}

/// Tokenize rendered pieces: text runs, and each picture as its processor
/// blocks.
fn tokenize(
    pieces: Vec<Piece>,
    tok: &GgufTokenizer,
    fam: &Family,
    limit: usize,
) -> Result<(Vec<u32>, Vec<MmChunk>), String> {
    let mut seq = Vec::new();
    let mut media = Vec::new();
    let unit = fam.cfg.patch * fam.cfg.merge;
    for p in pieces {
        match p {
            Piece::Text(t) => {
                if !t.is_empty() {
                    seq.extend(tok.encode(&t).map_err(|e| e.to_string())?);
                }
            }
            Piece::Picture { rgb, w, h } => {
                let (tw, th) = fam.cfg.picture_target(w, h)?;
                let (cols, rows) = (tw / unit, th / unit);
                for r in 0..rows {
                    seq.extend(std::iter::repeat_n(fam.img, cols));
                    seq.push(if r + 1 == rows { fam.end } else { fam.brk });
                }
                media.push(MmChunk::Image { rgb, w, h });
            }
        }
    }
    if seq.is_empty() {
        return Err("an input renders to no tokens".into());
    }
    if seq.len() > limit {
        return Err(format!(
            "{} tokens including picture tokens; the limit is {limit}",
            seq.len()
        ));
    }
    Ok((seq, media))
}

/// `/v1/embeddings` input through the embedder's template: each item a
/// string or a `{"content": [parts]}` object (one vector an item).
pub(crate) fn embed_inputs(
    input: &Value,
    tok: &GgufTokenizer,
    fam: &Family,
    limit: usize,
    task: Option<&str>,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<MmChunk>>), String> {
    let prefix = task.map(task_prefix).transpose()?;
    let items: Vec<&Value> = match input {
        Value::Array(items) if items.is_empty() => return Err("input is empty".into()),
        Value::Array(items) => items.iter().collect(),
        one => vec![one],
    };
    let mut pixels_left = crate::reference_image::REFERENCE_MAX_PIXELS;
    let mut seqs = Vec::with_capacity(items.len());
    let mut media = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let at = |e: String| format!("input[{i}]: {e}");
        let parts = match item {
            Value::String(s) => Parts {
                texts: vec![s.clone()],
                pictures: Vec::new(),
            },
            Value::Object(o) => {
                let list: Vec<&Value> = match o.get("content") {
                    Some(Value::Array(p)) if !p.is_empty() => p.iter().collect(),
                    Some(s @ Value::String(_)) => vec![s],
                    _ => return Err(at("an object item needs a non-empty `content`".into())),
                };
                parts_of(&list, &mut pixels_left, fam.pictures).map_err(at)?
            }
            _ => {
                return Err(at(
                    "an item must be a string or a `{\"content\": [...]}` object".into(),
                ));
            }
        };
        // render_user_content: pictures first, then texts, space-joined
        let mut pieces = Vec::new();
        let mut text = prefix.map(|p| format!("{p} ")).unwrap_or_default();
        let mut first = true;
        for (rgb, w, h) in parts.pictures {
            if !first {
                text.push(' ');
            }
            pieces.push(Piece::Text(std::mem::take(&mut text)));
            pieces.push(Piece::Picture { rgb, w, h });
            first = false;
        }
        for t in &parts.texts {
            if !first {
                text.push(' ');
            }
            text.push_str(&mark(t));
            first = false;
        }
        pieces.push(Piece::Text(text));
        let (seq, chunks) = tokenize(pieces, tok, fam, limit).map_err(at)?;
        seqs.push(seq);
        media.push(chunks);
    }
    Ok((seqs, media))
}

/// Whether `/v1/rerank` documents carry pictures (decode them off the async
/// runtime).
pub(crate) fn has_pictures(docs: &[Value]) -> bool {
    docs.iter().any(|d| {
        d.get("image").is_some()
            || d.get("content").and_then(Value::as_array).is_some_and(|p| {
                p.iter().any(|x| {
                    x.get("type")
                        .and_then(Value::as_str)
                        .is_some_and(|t| t != "text" && t != "input_text")
                })
            })
    })
}

/// `/v1/rerank` pairs through the reranker's template. A document is a
/// string, `{"text": ..}` (optionally with `"image": <data: URI>`, the
/// Jina form), or a `{"content": [parts]}` item.
pub(crate) fn rerank_inputs(
    query: &str,
    docs: &[Value],
    tok: &GgufTokenizer,
    fam: &Family,
    limit: usize,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<MmChunk>>), String> {
    let mut pixels_left = crate::reference_image::REFERENCE_MAX_PIXELS;
    let q = mark(query);
    let mut seqs = Vec::with_capacity(docs.len());
    let mut media = Vec::with_capacity(docs.len());
    for (i, doc) in docs.iter().enumerate() {
        let at = |e: String| format!("documents[{i}]: {e}");
        let parts = match doc {
            Value::String(s) => Parts {
                texts: vec![s.clone()],
                pictures: Vec::new(),
            },
            Value::Object(o) if o.contains_key("content") => {
                let list: Vec<&Value> = match o.get("content") {
                    Some(Value::Array(p)) if !p.is_empty() => p.iter().collect(),
                    Some(s @ Value::String(_)) => vec![s],
                    _ => return Err(at("`content` must be a non-empty list or a string".into())),
                };
                parts_of(&list, &mut pixels_left, fam.pictures).map_err(at)?
            }
            Value::Object(o) => {
                let mut p = Parts {
                    texts: Vec::new(),
                    pictures: Vec::new(),
                };
                if let Some(t) = o.get("text") {
                    p.texts.push(
                        t.as_str()
                            .ok_or_else(|| at("`text` must be a string".into()))?
                            .to_owned(),
                    );
                }
                if let Some(u) = o.get("image") {
                    if !fam.pictures {
                        return Err(at("this endpoint serves no pictures".into()));
                    }
                    let url = u
                        .as_str()
                        .ok_or_else(|| at("`image` must be a data: URI".into()))?;
                    p.pictures.push(decode(url, &mut pixels_left).map_err(at)?);
                }
                if p.texts.is_empty() && p.pictures.is_empty() {
                    return Err(at(
                        "a document object needs `text`, `image` or `content`".into()
                    ));
                }
                p
            }
            _ => return Err(at("a document must be a string or an object".into())),
        };
        if parts.pictures.len() > 1 {
            return Err(at("at most one picture per document".into()));
        }
        // text_content: the document's text parts joined by a blank line
        let d = parts
            .texts
            .iter()
            .map(|t| mark(t))
            .collect::<Vec<_>>()
            .join("\n\n");
        let body = format!("query: {q}\n\npassage: {d}");
        let pieces = match parts.pictures.into_iter().next() {
            // `rendered | trim`: "[IMG] " leads, so only the tail can strip
            Some((rgb, w, h)) => vec![
                Piece::Picture { rgb, w, h },
                Piece::Text(format!(" {body}").trim_end_matches(py_space).to_owned()),
            ],
            None => vec![Piece::Text(body)],
        };
        let (seq, chunks) = tokenize(pieces, tok, fam, limit).map_err(at)?;
        seqs.push(seq);
        media.push(chunks);
    }
    Ok((seqs, media))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PADDOCK_MODELS_DIR`, else `~/paddock/models`.
    fn models() -> PathBuf {
        std::env::var_os("PADDOCK_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join("paddock/models")
            })
    }

    fn data_uri(path: &str) -> String {
        use base64::Engine as _;
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(b)
        )
    }

    /// Every oracle case rendered through the templates here lands the
    /// processor's own ids (the fixtures' `input_ids.u32`), pictures' token
    /// blocks included. Skips when the checkpoints or fixtures are absent.
    #[test]
    fn templates_land_the_processors_ids() {
        let root = models();
        for (ckpt, sub) in [
            ("Nemotron-3.5-Embed-8B-BF16", "embed"),
            ("Nemotron-3.5-Rerank-8B-BF16", "rerank"),
        ] {
            let dir = root.join(ckpt);
            let fix = root.join("nemotron-embed-oracle").join(sub);
            if !dir.join("config.json").is_file() || !fix.join("manifest.json").is_file() {
                eprintln!("skip: {} or its fixtures are absent", dir.display());
                continue;
            }
            let cfg = NemotronEmbedConfig::read(&dir).expect("config");
            let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
            let id = |t: &str| tok.token_to_id(t).expect(t);
            let fam = Family {
                img: id("[IMG]"),
                brk: id("[IMG_BREAK]"),
                end: id("[IMG_END]"),
                cfg,
                pictures: true,
            };
            let man: Value =
                serde_json::from_slice(&std::fs::read(fix.join("manifest.json")).unwrap()).unwrap();
            for case in man["cases"].as_array().unwrap() {
                let name = case.as_str().unwrap();
                let cd = fix.join(name);
                let meta: Value =
                    serde_json::from_slice(&std::fs::read(cd.join("meta.json")).unwrap()).unwrap();
                let want: Vec<u32> = std::fs::read(cd.join("input_ids.u32"))
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                let text = meta["text"]
                    .as_str()
                    .unwrap_or("")
                    .repeat(meta["long_repeat"].as_u64().unwrap_or(1) as usize);
                let mut parts = Vec::new();
                if let Some(img) = meta["image"].as_str() {
                    parts.push(serde_json::json!({"type": "image_url",
                                                  "image_url": {"url": data_uri(img)}}));
                }
                if !text.is_empty() {
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
                let item = serde_json::json!({"content": parts});
                let (seqs, _) = if sub == "embed" {
                    let task = if meta["kind"] == "query" {
                        "query"
                    } else {
                        "passage"
                    };
                    embed_inputs(&item, &tok, &fam, 1 << 20, Some(task)).expect("render")
                } else {
                    rerank_inputs(
                        meta["query"].as_str().unwrap(),
                        &[item],
                        &tok,
                        &fam,
                        1 << 20,
                    )
                    .expect("render")
                };
                assert_eq!(
                    seqs[0], want,
                    "{sub}/{name}: ids differ from the processor's"
                );
            }
        }
    }

    #[test]
    fn user_text_is_marked_like_the_templates() {
        assert_eq!(mark("a[IMG]<s>b"), "a[\u{200c}IMG]<\u{200c}s>b");
        assert_eq!(mark("plain"), "plain");
        assert!(py_space('\u{1f}') && py_space(' ') && !py_space('\u{200c}'));
        assert_eq!(task_prefix("search_query").unwrap(), "query:");
        assert_eq!(task_prefix("document").unwrap(), "passage:");
        assert!(task_prefix("clustering").is_err());
    }
}
