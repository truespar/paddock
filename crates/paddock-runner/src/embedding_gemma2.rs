//! Model-specific encoder admission and input contract; computation stays in
//! the device backends (native Metal, the CUDA encoder in paddock-engine) and
//! the shared ragged encoder scheduler.
use crate::serving::{EmbedModel, ServeError};
use std::path::{Path, PathBuf};
mod video;

pub(crate) fn directory(path: &Path) -> Option<PathBuf> {
    let dir = if path.is_dir() {
        path
    } else if path.extension().is_some_and(|x| x == "safetensors") {
        path.parent()?
    } else {
        return None;
    };
    let config = dir.join("config.json");
    if std::fs::metadata(&config).ok()?.len() > 1 << 20 {
        return None;
    }
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(config).ok()?).ok()?;
    (cfg["model_type"] == "embedding_gemma2").then(|| dir.to_path_buf())
}

/// The picture soft-token budget: `max_image_tokens` when set (one of the
/// processor's supported budgets), else the checkpoint's 280.
pub(crate) fn image_budget(configured: Option<u32>) -> Result<usize, ServeError> {
    use paddock_engine::gpu_model::embedding_gemma2::media::{
        DEFAULT_IMAGE_TOKENS, IMAGE_TOKEN_BUDGETS,
    };
    let Some(b) = configured else {
        return Ok(DEFAULT_IMAGE_TOKENS);
    };
    let b = b as usize;
    if IMAGE_TOKEN_BUDGETS.contains(&b) {
        Ok(b)
    } else {
        Err(ServeError::Engine(format!(
            "EmbeddingGemma 2 pictures take a budget of {IMAGE_TOKEN_BUDGETS:?} tokens (got {b})"
        )))
    }
}

/// The media towers of the GGUF lane (CUDA and Metal): the projector files - the
/// upstream one carrying both towers, or the catalog's picture and audio
/// cuts - the picture soft-token budget, and whether to load an audio tower.
pub(crate) struct Media {
    pub files: Vec<PathBuf>,
    pub image_budget: usize,
    // Only the Metal MLX package reads this: its one bundle holds both towers.
    // The GGUF lanes already pick towers by which files are in `files`.
    #[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
    pub image: bool,
    pub audio: bool,
}

/// `media`: the towers to attach, if any.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    context: usize,
    budget: Option<u64>,
    media: Option<Media>,
) -> Result<EmbedModel, ServeError> {
    use paddock_tokenizer::GgufTokenizer;
    use std::sync::Arc;
    let tokenizer = if path.is_dir() {
        GgufTokenizer::from_hf_dir(path)
    } else {
        let map = paddock_models::mapped::MappedGguf::open(path)
            .map_err(|e| ServeError::Open(path.into(), e.to_string()))?;
        GgufTokenizer::from_gguf(map.gguf())
    }
    .map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    if tokenizer.bos_id != Some(2) || tokenizer.eos_id != Some(1) {
        return Err(ServeError::Tokenizer(
            "EmbeddingGemma 2 requires its BOS=2/EOS=1 tokenizer".into(),
        ));
    }
    let context = context.min(8192);
    let metrics = Arc::new(paddock_engine::metrics::EngineMetrics::default());
    let image_budget = media.as_ref().map(|m| m.image_budget);
    let encoder = match device {
        "cuda" => cuda_encoder(path, gpu, pack, context, budget, media, &metrics)?,
        "metal" => metal_encoder(path, gpu, pack, context, budget, media, &metrics)?,
        other => {
            return Err(ServeError::Engine(format!(
                "EmbeddingGemma 2 needs cuda or metal (got {other:?})"
            )));
        }
    };
    // the towers the encoder actually built: an older kernel pack runs the
    // picture tower but not the audio one
    let image_budget = image_budget.filter(|_| encoder.serves_images());
    let audio = encoder.serves_audio();
    Ok(EmbedModel {
        id,
        encoder,
        tokenizer: Arc::new(tokenizer),
        eos: Some(1),
        yes_id: None,
        no_id: None,
        is_reranker: false,
        embedding_gemma2: true,
        max_input_tokens: context,
        image_budget,
        audio,
        metrics,
    })
}

/// The CUDA encoder serves the Q8_0 GGUF (the MLX checkpoint is Metal's),
/// with the picture tower when an mmproj is given.
fn cuda_encoder(
    path: &Path,
    gpu: usize,
    pack: Option<&Path>,
    context: usize,
    budget: Option<u64>,
    media: Option<Media>,
    metrics: &std::sync::Arc<paddock_engine::metrics::EngineMetrics>,
) -> Result<paddock_engine::encoder::Encoder, ServeError> {
    if path.is_dir() {
        return Err(ServeError::Engine(
            "EmbeddingGemma 2 on CUDA serves the Q8_0 GGUF; the MLX checkpoint is Metal's".into(),
        ));
    }
    let path = path.to_path_buf();
    let pack = pack.map(Path::to_path_buf);
    paddock_engine::encoder::Encoder::spawn(
        move || {
            let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
                .map_err(|e| e.to_string())?;
            crate::serving::note_device_cc(&exec);
            if let Some(b) = budget {
                exec.set_vram_budget(b);
            }
            let map = paddock_models::mapped::MappedGguf::open(&path).map_err(|e| e.to_string())?;
            let mut model = paddock_engine::gpu_model::embedding_gemma2::GpuEmbeddingGemma2::load(
                std::sync::Arc::new(exec),
                &map,
                context,
            )
            .map_err(|e| e.to_string())?;
            for file in media.iter().flat_map(|m| &m.files) {
                let (budget, audio) = media
                    .as_ref()
                    .map_or((0, false), |m| (m.image_budget, m.audio));
                let mm = paddock_models::mapped::MappedGguf::open(file)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
                model
                    .attach_mmproj(&mm, budget, audio)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
            }
            Ok(model)
        },
        Some(metrics.clone()),
    )
    .map_err(ServeError::Engine)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_encoder(
    path: &Path,
    gpu: usize,
    pack: Option<&Path>,
    context: usize,
    budget: Option<u64>,
    media: Option<Media>,
    metrics: &std::sync::Arc<paddock_engine::metrics::EngineMetrics>,
) -> Result<paddock_engine::encoder::Encoder, ServeError> {
    if gpu != 0 || pack.is_some() {
        return Err(ServeError::Engine(
            "EmbeddingGemma 2 currently requires native Metal with the system Apple GPU".into(),
        ));
    }
    let path = path.to_path_buf();
    paddock_engine::encoder::Encoder::spawn(
        move || {
            let mut model = paddock_metal::EmbeddingGemma2::load(&path, context, budget)
                .map_err(|e| e.to_string())?;
            if let Some(media) = media {
                if path.is_dir() {
                    model
                        .attach_mlx_media(&path, media.image_budget, media.image, media.audio)
                        .map_err(|e| e.to_string())?;
                } else {
                    for file in media.files {
                        model
                            .attach_mmproj(&file, media.image_budget, media.audio)
                            .map_err(|e| format!("{}: {e}", file.display()))?;
                    }
                }
            }
            Ok(model)
        },
        Some(metrics.clone()),
    )
    .map_err(ServeError::Engine)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_encoder(
    _: &Path,
    _: usize,
    _: Option<&Path>,
    _: usize,
    _: Option<u64>,
    _: Option<Media>,
    _: &std::sync::Arc<paddock_engine::metrics::EngineMetrics>,
) -> Result<paddock_engine::encoder::Encoder, ServeError> {
    Err(ServeError::Engine(
        "this runner has no EmbeddingGemma 2 Metal backend".into(),
    ))
}

pub(crate) fn task_prefix(task: &str) -> Result<&'static str, String> {
    Ok(match task {
        "query" | "search_query" => "task: search result | query: ",
        "document" => "title: none | text: ",
        "classification" => "task: classification | query: ",
        "clustering" => "task: clustering | query: ",
        "code_retrieval" => "task: code retrieval | query: ",
        "fact_checking" => "task: fact checking | query: ",
        "question_answering" => "task: question answering | query: ",
        "sentence_similarity" => "task: sentence similarity | query: ",
        _ => return Err("unknown EmbeddingGemma 2 task".into()),
    })
}

/// Whether an `input` carries content-part items (the media form). Plain
/// strings and token ids stay on the text path.
pub(crate) fn has_media_items(input: &serde_json::Value) -> bool {
    match input {
        serde_json::Value::Object(_) => true,
        serde_json::Value::Array(items) => items.iter().any(serde_json::Value::is_object),
        _ => false,
    }
}

/// The media form of `/v1/embeddings` input: each item a string or a
/// `{"content": [parts]}` object, the parts in the chat shape (`text` /
/// `input_text`, `image_url` / `input_image` with a data: URI, decoded as the
/// reference processor's Pillow decodes it, `input_audio` / a data:audio
/// `audio_url`, decoded to 16 kHz mono). One item, one
/// vector, the parts in the order given - the model card's interleaving,
/// where each `<|image|>` in the text is filled in order.
///
/// The sequence is what Hugging Face's processor builds: `<bos>`, then the
/// text runs and media blocks (`<|image>` + N x `<|image|>` + `<image|>`, N
/// from the picture's size at this endpoint's budget; `<|audio>` + N x
/// `<|audio|>` + `<audio|>`, 25 a second), then `<eos>`. Text
/// between two media is one run - the tokenizer splits at special tokens, so
/// tokenizing a run alone is what HF does too. The task prefix applies to
/// items without media only ("Prefixes apply to text only. Pass images,
/// video, and audio without any prefix" - the model card).
pub(crate) fn media_inputs(
    input: &serde_json::Value,
    m: &EmbedModel,
    prefix: &str,
) -> Result<(Vec<Vec<u32>>, Vec<Vec<paddock_engine::service::MmChunk>>), String> {
    use paddock_engine::encoder::embedding_gemma2::*;
    use serde_json::Value;
    let items: Vec<&Value> = match input {
        Value::Array(items) if items.is_empty() => return Err("input is empty".into()),
        Value::Array(items) => items.iter().collect(),
        one => vec![one],
    };
    let tok = |s: &str| m.tokenizer.encode(s).map_err(|e| e.to_string());
    // one request's pictures share the reference decoder's pixel ceiling
    let mut pixels_left = crate::reference_image::REFERENCE_MAX_PIXELS;
    let mut seqs = Vec::with_capacity(items.len());
    let mut media = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let at = |e: String| format!("input[{i}]: {e}");
        let mut seq = vec![2u32];
        let mut chunks = Vec::new();
        match item {
            Value::String(s) => seq.extend(tok(&format!("{prefix}{s}")).map_err(at)?),
            Value::Object(o) => {
                let parts: Vec<&Value> = match o.get("content") {
                    Some(Value::Array(p)) if !p.is_empty() => p.iter().collect(),
                    Some(s @ Value::String(_)) => vec![s],
                    _ => return Err(at("an object item needs a non-empty `content`".into())),
                };
                // adjacent text parts form one run; a medium closes it
                let mut run = String::new();
                let mut text_only = true;
                let mut runs = Vec::new();
                for part in parts {
                    if let Value::String(s) = part {
                        run.push_str(s);
                        continue;
                    }
                    let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
                    match kind {
                        "text" | "input_text" => run.push_str(
                            part.get("text")
                                .and_then(Value::as_str)
                                .ok_or_else(|| at("a text part needs `text`".into()))?,
                        ),
                        "image_url" | "input_image" => {
                            let v = part.get("image_url");
                            let url = v
                                .and_then(|v| v.get("url").and_then(Value::as_str).or(v.as_str()))
                                .ok_or_else(|| at("an image part needs `image_url`".into()))?;
                            let detail = v
                                .and_then(|v| v.get("detail"))
                                .or(part.get("detail"))
                                .and_then(Value::as_str);
                            if detail.is_some_and(|d| d != "auto") {
                                return Err(at(format!(
                                    "image detail {detail:?} is not served: EmbeddingGemma 2 \
                                     pictures take the endpoint's budget (`max_image_tokens`)"
                                )));
                            }
                            // the processor's own decoder's bytes (libjpeg-turbo for
                            // a JPEG), upright, never resized here
                            let (rgb, w, h) =
                                crate::reference_image::decode_image_url_reference_limited(
                                    url,
                                    pixels_left,
                                )
                                .map_err(at)?;
                            pixels_left = pixels_left.saturating_sub((w * h) as u64);
                            text_only = false;
                            let chunk = paddock_engine::service::MmChunk::Image { rgb, w, h };
                            runs.push((std::mem::take(&mut run), Some((IMAGE_TOKEN, chunk))));
                        }
                        "input_audio" | "audio_url" => {
                            if !m.audio {
                                return Err(at("this endpoint serves no audio (its mmproj or kernel pack has no audio tower)".into()));
                            }
                            text_only = false;
                            runs.push((
                                std::mem::take(&mut run),
                                Some((AUDIO_TOKEN, clip(part).map_err(at)?)),
                            ));
                        }
                        "input_video" => {
                            if !m.encoder.serves_video() {
                                return Err(at("this endpoint does not serve video frames".into()));
                            }
                            let frames = video::decode(part, &mut pixels_left).map_err(at)?;
                            text_only = false;
                            for (before, frame) in frames {
                                run.push_str(&before);
                                runs.push((std::mem::take(&mut run), Some((VIDEO_TOKEN, frame))));
                            }
                        }
                        other => {
                            return Err(at(format!(
                                "content part type {other:?} is not served (text, image_url, input_audio, input_video)"
                            )));
                        }
                    }
                }
                runs.push((run, None));
                if text_only {
                    // one text run, prefixed like a string item
                    let text = format!("{prefix}{}", runs.pop().map(|r| r.0).unwrap_or_default());
                    seq.extend(tok(&text).map_err(at)?);
                } else {
                    for (text, chunk) in runs {
                        if !text.is_empty() {
                            seq.extend(tok(&text).map_err(at)?);
                        }
                        let Some((soft_token, chunk)) = chunk else {
                            continue;
                        };
                        let (open, soft, close, n) = match &chunk {
                            paddock_engine::service::MmChunk::Image { w, h, .. } => {
                                let budget = m.image_budget.ok_or_else(|| {
                                    at("this endpoint serves no pictures (no picture tower)".into())
                                })?;
                                let budget = if soft_token == VIDEO_TOKEN {
                                    VIDEO_FRAME_TOKENS
                                } else {
                                    budget
                                };
                                let n = image_tokens(*w, *h, budget).map_err(at)?;
                                (BOI_TOKEN, soft_token, EOI_TOKEN, n)
                            }
                            paddock_engine::service::MmChunk::Audio { samples, .. } => (
                                BOA_TOKEN,
                                AUDIO_TOKEN,
                                EOA_TOKEN,
                                audio_tokens(samples.len()).map_err(at)?,
                            ),
                            _ => unreachable!("only pictures and clips are parsed"),
                        };
                        seq.push(open);
                        seq.extend(std::iter::repeat_n(soft, n));
                        seq.push(close);
                        chunks.push(chunk);
                    }
                }
            }
            _ => {
                return Err(at(
                    "an item must be a string or a `{\"content\": [...]}` object".into(),
                ));
            }
        }
        seq.push(1);
        if seq.len() > m.max_input_tokens {
            return Err(at(format!(
                "{} tokens including media and special tokens; the limit is {}",
                seq.len(),
                m.max_input_tokens
            )));
        }
        seqs.push(seq);
        media.push(chunks);
    }
    Ok((seqs, media))
}

/// One audio part to 16 kHz mono samples - `/v1/audio/transcriptions`'
/// decode (any container it takes) and resampler. At most 30 s: the decode
/// stops a second past that, and `audio_tokens` refuses the rest honestly.
fn clip(part: &serde_json::Value) -> Result<paddock_engine::service::MmChunk, String> {
    use base64::Engine as _;
    let refs = crate::chat::find_audio(&[serde_json::json!({ "content": [part] })])?;
    let r = refs
        .into_iter()
        .next()
        .ok_or("an audio part with no audio")?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(r.b64.trim())
        .map_err(|e| format!("audio base64: {e}"))?;
    let wav = paddock_engine::audio::decode::decode_audio_limited(&bytes, 31)
        .map_err(|e| format!("audio decode (declared format {:?}): {e}", r.format))?;
    if wav.samples.is_empty() {
        return Err("audio part holds no samples".into());
    }
    let samples = paddock_engine::audio::resample::resample(&wav.samples, wav.sample_rate, 16000)?;
    Ok(paddock_engine::service::MmChunk::Audio { samples, mel: None })
}
