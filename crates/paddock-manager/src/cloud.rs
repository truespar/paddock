//! Cloud models - the manager's BYO-key external-provider client (doc §1:
//! "the manager calls runners only as its own API client (chat/compare, web
//! search, MCP host, BYO-key external providers, benchmarks)"). Users register
//! provider endpoints (OpenAI, Anthropic, OpenRouter, any OpenAI-compatible
//! server) with their own API key; the enabled models join the Studio's
//! pickers as Cloud models for chat/compare.
//!
//! The Studio speaks exactly one dialect - the OpenAI Responses event stream
//! it already uses for runners - so this module translates on the way out and
//! back: a Responses request becomes the provider's native call
//! (chat/completions for OpenAI-compatible, /v1/messages for Anthropic,
//! passthrough for OpenAI itself), and the provider's SSE becomes the small
//! Responses event set the Studio consumes (output_text/reasoning_text deltas,
//! completed/incomplete/failed with usage). Keys live in the manager DB and
//! ride only manager->provider; the browser never sees one.
//!
//! This is not the forbidden inference proxy: like the runner relays, it lives
//! under /api behind manager auth as the Studio's private seam - /v1/* still
//! does not exist on the manager port.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde_json::{Value, json};

use crate::routes::AppState;

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// One pooled client for every provider call. A fresh Client per request made
/// every fetch a COLD first contact - and a first TCP/TLS to
/// api.openai.com can intermittently stall ~20s (seen: 19.8s, then 179ms once
/// warm), which reads as "operation timed out". With a pool only the
/// first call pays the handshake, and connect_timeout fails fast enough that
/// send_retry's second attempt still fits the caller's patience.
static HTTP: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(8))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("TLS backend available")
});

/// Send an idempotent GET, retrying once when the failure is a timeout or a
/// connect error - the observed stall is first-contact only, so the immediate
/// second attempt usually lands.
async fn send_retry(req: reqwest::RequestBuilder) -> Result<reqwest::Response, reqwest::Error> {
    let retry = req.try_clone();
    match req.send().await {
        Err(e) if (e.is_timeout() || e.is_connect()) && retry.is_some() => {
            retry.expect("checked").send().await
        }
        r => r,
    }
}
/// Anthropic requires max_tokens; when the Studio didn't send one, cap
/// generously rather than truncating at some tiny provider default.
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 4096;

// ── CRUD ────────────────────────────────────────────────────────────────────

/// `GET /api/cloud` - endpoints without keys (`hasKey` only).
pub async fn list(State(state): State<Arc<AppState>>) -> Response {
    match state.db.list_cloud_endpoints() {
        Ok(rows) => Json(Value::Array(rows)).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `POST /api/cloud` - create `{name, kind, baseUrl, apiKey?, models?}`.
pub async fn create(State(state): State<Arc<AppState>>, Json(doc): Json<Value>) -> Response {
    match state.db.create_cloud_endpoint(&doc) {
        Ok(row) => Json(row).into_response(),
        Err(crate::store::StoreError::Bad(m)) => err(StatusCode::BAD_REQUEST, m),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `PATCH /api/cloud/{id}` - partial update; an absent/empty apiKey keeps the
/// stored one (the edit form round-trips without it).
pub async fn update(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(doc): Json<Value>,
) -> Response {
    match state.db.update_cloud_endpoint(&id, &doc) {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `DELETE /api/cloud/{id}`.
pub async fn delete(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.delete_cloud_endpoint(&id) {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ── provider model list ─────────────────────────────────────────────────────

/// `GET /api/cloud/{id}/models` - the provider's own model list, normalized to
/// `GET /api/cloud/browse` - the OpenRouter catalog with no stored endpoint:
/// their list endpoint is public, and the Cloud page leads with it (search
/// first, key only to chat -). Same normalized shape as the
/// per-endpoint route, always trending-ordered.
pub async fn browse() -> Response {
    let res = match send_retry(
        HTTP.get("https://openrouter.ai/api/v1/models?sort=most-popular")
            .timeout(std::time::Duration::from_secs(20)),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("OpenRouter not answering: {}", errchain(&e)),
            );
        }
    };
    let status = res.status();
    let body: Value = match res.json().await {
        Ok(v) => v,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("OpenRouter answered non-JSON: {e}"),
            );
        }
    };
    if !status.is_success() {
        let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        return (code, Json(body)).into_response();
    }
    let mut list = body
        .get("data")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(normalize_model).collect::<Vec<_>>())
        .unwrap_or_default();
    list.extend(openrouter_speech_models("https://openrouter.ai/api/v1", "").await);
    Json(json!({ "models": list, "ranked": true })).into_response()
}

/// OpenRouter's speech models, which its DEFAULT list does not contain.
///
/// Measured against the live API: `/models` returns 406 rows and
/// not one of them is a transcription model - `openai/whisper-large-v3`,
/// `nvidia/parakeet-tdt-0.6b-v3`, `google/chirp-3` and the rest appear only
/// under `?output_modalities=transcription`, which returns its own 14. The
/// default list is text-output models, and nothing says so. So the catalog is
/// two calls, or the picker simply has no speech models to show however well
/// they are flagged.
///
/// A failure here costs the speech rows and nothing else: the main catalog is
/// the answer to the request, and losing 400 models because a second call
/// timed out would be the worse trade.
async fn openrouter_speech_models(base: &str, key: &str) -> Vec<Value> {
    let req = HTTP
        .get(format!("{base}/models?output_modalities=transcription"))
        .timeout(std::time::Duration::from_secs(20));
    let req = if key.is_empty() {
        req
    } else {
        req.bearer_auth(key)
    };
    let Ok(res) = send_retry(req).await else {
        tracing::warn!("openrouter speech-model list did not answer; catalog has none");
        return Vec::new();
    };
    if !res.status().is_success() {
        tracing::warn!(status = %res.status(), "openrouter speech-model list refused");
        return Vec::new();
    }
    let Ok(body) = res.json::<Value>().await else {
        return Vec::new();
    };
    body.get("data")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(normalize_model).collect())
        .unwrap_or_default()
}

/// `GET /api/cloud/browse/endpoints?model=author/slug` - OpenRouter's
/// official per-model provider list (public, like the catalog). One model is
/// served by many providers, each with its own price, context, quantization
/// and live throughput - the expandable row's data.
pub async fn browse_endpoints(
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let Some(model) = q.get("model").filter(|s| !s.is_empty()) else {
        return err(StatusCode::BAD_REQUEST, "missing ?model=".into());
    };
    let res = match send_retry(
        HTTP.get(format!(
            "https://openrouter.ai/api/v1/models/{model}/endpoints"
        ))
        .timeout(std::time::Duration::from_secs(15)),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("OpenRouter not answering: {}", errchain(&e)),
            );
        }
    };
    let status = res.status();
    let body: Value = match res.json().await {
        Ok(v) => v,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("OpenRouter answered non-JSON: {e}"),
            );
        }
    };
    if !status.is_success() {
        let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        return (code, Json(body)).into_response();
    }
    let providers: Vec<Value> = body
        .get("data")
        .and_then(|d| d.get("endpoints"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let name = e
                        .get("provider_name")
                        .or_else(|| e.get("name"))
                        .and_then(Value::as_str)?;
                    let price = |k: &str| {
                        e.get("pricing")
                            .and_then(|p| p.get(k))
                            .and_then(Value::as_str)
                            .and_then(|s| s.parse::<f64>().ok())
                            .filter(|v| v.is_finite() && *v >= 0.0)
                    };
                    let mut out = json!({ "name": name });
                    // the tag is the endpoint's UNIQUE identity - the same
                    // brand appears many times as regional/tier variants
                    // (amazon-bedrock vs amazon-bedrock/us-east-1), and the
                    // tag is also the slug provider routing pins on
                    if let Some(t) = e
                        .get("tag")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                    {
                        out["tag"] = json!(t);
                    }
                    if let Some(c) = e.get("context_length").and_then(Value::as_u64) {
                        out["ctx"] = json!(c);
                    }
                    if let Some(p) = price("prompt") {
                        out["promptPrice"] = json!(p);
                    }
                    if let Some(p) = price("completion") {
                        out["completionPrice"] = json!(p);
                    }
                    if let Some(q) = e
                        .get("quantization")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty() && *s != "unknown")
                    {
                        out["quant"] = json!(q);
                    }
                    if let Some(m) = e.get("max_completion_tokens").and_then(Value::as_u64) {
                        out["maxOut"] = json!(m);
                    }
                    if let Some(t) = e.get("throughput_last_30m").and_then(Value::as_f64) {
                        out["tps"] = json!(t.round());
                    }
                    Some(out)
                })
                .collect()
        })
        .unwrap_or_default();
    Json(json!({ "providers": providers })).into_response()
}

/// `{models: [...], ranked: bool}` for the enable-picker (`ranked` = the
/// order is last-week popularity, so the Studio may label it "Trending").
/// Doubles as the key check: a bad key surfaces as the provider's 401 message.
pub async fn models(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let (kind, base, key) = match load_secret(&state, &id).await {
        Ok(secret) => secret,
        Err(response) => return response,
    };
    // The key is optional here, unlike the chat relay: OpenRouter's list
    // endpoint is public (browse-before-you-paste-a-key is the intended
    // first run), and a keyless request to any other provider surfaces that
    // provider's own 401 - the honest message. Auth rides when a key exists.
    let ranked = is_openrouter(&base);
    let client = &*HTTP;
    let req = match kind.as_str() {
        // both OpenAI kinds list on GET {base}/models with a Bearer key;
        // OpenRouter additionally takes the DOCUMENTED sort param, which
        // makes the default picker view genuine last-week usage
        "openai" | "openai-compat" => {
            let url = if ranked {
                format!("{base}/models?sort=most-popular")
            } else {
                format!("{base}/models")
            };
            let r = client.get(url);
            if key.is_empty() {
                r
            } else {
                r.bearer_auth(&key)
            }
        }
        "anthropic" => {
            let r = client
                .get(format!("{base}/models"))
                .header("anthropic-version", ANTHROPIC_VERSION);
            if key.is_empty() {
                r
            } else {
                r.header("x-api-key", &key)
            }
        }
        other => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("unknown kind \"{other}\""),
            );
        }
    };
    let res = match send_retry(req.timeout(std::time::Duration::from_secs(20))).await {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("provider not answering: {}", errchain(&e)),
            );
        }
    };
    let status = res.status();
    let body: Value = match res.json().await {
        Ok(v) => v,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("provider answered non-JSON: {e}"),
            );
        }
    };
    if !status.is_success() {
        // pass the provider's own message through (both OpenAI and Anthropic
        // use {error:{message}}, which is what the Studio reads)
        let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        return (code, Json(body)).into_response();
    }
    let deny_non_chat = kind == "openai";
    let mut list = body
        .get("data")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter(|m| {
                    !deny_non_chat
                        || m.get("id")
                            .and_then(Value::as_str)
                            .is_some_and(openai_chat_model)
                })
                .filter_map(normalize_model)
                .map(|mut m| {
                    stamp_native_reasoning(&kind, &mut m);
                    stamp_native_asr(&kind, &mut m);
                    m
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    // A SAVED OpenRouter endpoint hits this route rather than `browse`, and
    // OpenRouter keeps its speech models out of the default list - same two
    // calls, same reason (see `openrouter_speech_models`).
    if ranked {
        list.extend(openrouter_speech_models(&base, &key).await);
    }
    Json(json!({ "models": list, "ranked": ranked })).into_response()
}

/// OpenAI's /v1/models lists everything the key can call - TTS, embeddings,
/// moderation, image and legacy completion models alongside the chat ones.
/// A pick we cannot serve is only ever a broken lane, so it never reaches the
/// picker.
///
/// `whisper` and `transcribe` are deliberately not on this list: a transcriber
/// is a real lane now, not a pick that would break. The rest stay
/// - nothing here turns an embedding or an image model into a conversation.
fn openai_chat_model(id: &str) -> bool {
    const UNSERVABLE: &[&str] = &[
        "tts",
        "embedding",
        "moderation",
        "davinci",
        "babbage",
        "gpt-image",
        "dall-e",
        "realtime",
        "instruct",
    ];
    !UNSERVABLE.iter().any(|t| id.contains(t))
}

/// Native lists carry no modality metadata at all, so a bare `whisper-1` would
/// arrive as `kind: 'chat'` and earn a refusal the first time someone typed at
/// it. OpenAI's speech ids are well known and stable; stamp them the same way
/// `stamp_native_reasoning` stamps thinking.
///
/// Only for the NATIVE lists - OpenRouter states `output_modalities` itself
/// and `normalize_model` reads the truth rather than guessing from a name.
fn stamp_native_asr(kind: &str, m: &mut Value) {
    if kind != "openai" {
        return;
    }
    let Some(id) = m.get("id").and_then(Value::as_str) else {
        return;
    };
    if id.contains("whisper") || id.contains("transcribe") {
        m["asr"] = json!(true);
    }
}

/// Native lists carry no capability metadata, but the reasoning families are
/// well known: OpenAI's gpt-5*/o* always think (effort-controlled), current
/// Claude models think when asked (extended thinking). The stamp feeds the
/// Studio's per-lane thinking control - without it the compare toggle never
/// appears for these picks.
fn stamp_native_reasoning(kind: &str, m: &mut Value) {
    let Some(id) = m.get("id").and_then(Value::as_str) else {
        return;
    };
    let capable = match kind {
        "openai" => id.starts_with("gpt-5") || ["o1", "o3", "o4"].iter().any(|p| id.starts_with(p)),
        "anthropic" => id.starts_with("claude"),
        _ => return,
    };
    if capable {
        m["reasoning"] = json!(true);
    }
}

/// OpenRouter gets special treatment its official API invites: the documented
/// `sort=most-popular` param ("most tokens processed in the last week") turns
/// the picker's default view into real trending data. A plain substring check
/// must not be used for routing or credential checks: only the canonical
/// API base receives OpenRouter-specific behavior.
fn is_openrouter(base: &str) -> bool {
    crate::connections::is_openrouter(base)
}

/// One provider model row -> the normalized picker shape. OpenRouter carries
/// the richest metadata (name, pricing, context, modalities, reasoning);
/// OpenAI has bare ids + created; Anthropic has display_name. Everything
/// beyond `id` is optional and the picker degrades per field.
pub(crate) fn normalize_model(m: &Value) -> Option<Value> {
    let id = m.get("id").and_then(Value::as_str)?;
    let display = m
        .get("name")
        .or_else(|| m.get("display_name"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());
    // 0 is not a context window, it is the absence of one - every
    // transcription model reports `context_length: 0` because the question
    // does not apply to it, and "0 ctx" in the picker is a worse lie than no
    // number at all.
    let ctx = m
        .get("context_length")
        .and_then(Value::as_u64)
        .filter(|c| *c > 0);
    // Speech-to-text: audio in, a transcript out, and it does not chat. The
    // discriminator is the provider's own, not a name guess - OpenRouter tags
    // these `output_modalities: ["transcription"]`, which is also what
    // `GET /models?output_modalities=transcription` filters on. The Studio
    // turns this into `kind: 'transcriber'`, which is what routes a clip to
    // /audio/transcriptions instead of the chat wire.
    let asr = m
        .get("architecture")
        .and_then(|a| a.get("output_modalities"))
        .and_then(Value::as_array)
        .is_some_and(|mods| mods.iter().any(|x| x.as_str() == Some("transcription")));
    // The model's own OUTPUT ceiling, which is a different number from the
    // context window and routinely far smaller (deepseek-v4-flash-0731: 1M
    // context, 384k output). The Studio's "Model maximum" reply length used to
    // be derived from the context alone, so it asked a 1M-context model for
    // ~1M output tokens and the provider 400'd the whole send on
    // prompt+output > context. OpenRouter carries it under
    // `top_provider`; the per-endpoint list carries it bare.
    let max_out = m
        .get("top_provider")
        .and_then(|t| t.get("max_completion_tokens"))
        .or_else(|| m.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .filter(|v| *v > 0);
    let vision = m
        .get("architecture")
        .and_then(|a| a.get("input_modalities"))
        .and_then(Value::as_array)
        .map(|mods| mods.iter().any(|x| x.as_str() == Some("image")));
    // per-token USD strings ("0.00001"); the Studio renders them as $/M.
    // OpenRouter's router pseudo-models price as "-1" - a DYNAMIC-pricing
    // sentinel, not a number to show ("$-1000000" happened) - negatives are
    // dropped so the row simply carries no price.
    let price = |k: &str| {
        m.get("pricing")
            .and_then(|p| p.get(k))
            .and_then(Value::as_str)
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
    };
    let prompt_price = price("prompt");
    let completion_price = price("completion");
    // reasoning support: OpenRouter's `reasoning` block, or `reasoning` in
    // the supported parameter list
    let reasoning = m.get("reasoning").is_some_and(|r| !r.is_null())
        || m.get("supported_parameters")
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some("reasoning")));
    let created = m.get("created").and_then(Value::as_u64);
    let free =
        id.ends_with(":free") || (prompt_price == Some(0.0) && completion_price == Some(0.0));
    // a short description tail feeds the picker's SEARCH (matching "coding",
    // "fast", "agentic"...), not the UI - cap it so 400 models stay light
    let blurb = m
        .get("description")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take(200).collect::<String>());
    let mut out = json!({ "id": id });
    if let Some(d) = display {
        out["display"] = json!(d);
    }
    if let Some(c) = ctx {
        out["ctx"] = json!(c);
    }
    if let Some(m) = max_out {
        out["maxOut"] = json!(m);
    }
    if asr {
        out["asr"] = json!(true);
    }
    if let Some(v) = vision {
        out["vision"] = json!(v);
    }
    if let Some(p) = prompt_price {
        out["promptPrice"] = json!(p);
    }
    if let Some(p) = completion_price {
        out["completionPrice"] = json!(p);
    }
    if reasoning {
        out["reasoning"] = json!(true);
    }
    if let Some(parameters) = m.get("supported_parameters").and_then(Value::as_array) {
        out["tools"] = json!(parameters.iter().any(|p| p.as_str() == Some("tools")));
    }
    if let Some(c) = created {
        out["created"] = json!(c);
    }
    if free {
        out["free"] = json!(true);
    }
    if let Some(b) = blurb {
        out["blurb"] = json!(b);
    }
    Some(out)
}

pub(crate) fn normalize_provider_model(kind: &str, m: &Value) -> Option<Value> {
    if kind == "openai"
        && !m
            .get("id")
            .and_then(Value::as_str)
            .is_some_and(openai_chat_model)
    {
        return None;
    }
    let mut value = normalize_model(m)?;
    stamp_native_reasoning(kind, &mut value);
    stamp_native_asr(kind, &mut value);
    Some(value)
}

/// `POST /api/cloud/{id}/check` - does the saved key actually authenticate,
/// and does an OpenAI/Anthropic-shaped API answer at the base URL? Runs after
/// every key save and from the endpoint menu. OpenRouter needs its own probe
/// (their documented GET {base}/key key-info endpoint) because its model list
/// is public and answers 200 to any key. Returns `{ok, message}` - the
/// message is the provider's own words on failure.
pub async fn check(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let (kind, base, key) = match load_secret(&state, &id).await {
        Ok(secret) => secret,
        Err(response) => return response,
    };
    let client = &*HTTP;
    let openrouter = is_openrouter(&base);
    if key.is_empty() && openrouter {
        return Json(json!({"ok": false, "message": "No API key saved to test."})).into_response();
    }
    let req = match kind.as_str() {
        // the key-info endpoint is the only authenticated cheap probe there
        "openai" | "openai-compat" if openrouter => {
            client.get(format!("{base}/key")).bearer_auth(&key)
        }
        "openai" | "openai-compat" => {
            let r = client.get(format!("{base}/models"));
            if key.is_empty() {
                r
            } else {
                r.bearer_auth(&key)
            }
        }
        "anthropic" => {
            let r = client
                .get(format!("{base}/models"))
                .header("anthropic-version", ANTHROPIC_VERSION);
            if key.is_empty() {
                r
            } else {
                r.header("x-api-key", &key)
            }
        }
        other => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("unknown kind \"{other}\""),
            );
        }
    };
    let res = match req.timeout(std::time::Duration::from_secs(15)).send().await {
        Ok(r) => r,
        Err(e) => {
            return Json(json!({"ok": false, "message": format!("Nothing answered at {base}: {}", errchain(&e))}))
                .into_response();
        }
    };
    let status = res.status();
    let body: Value = res.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        let msg = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("the server answered HTTP {status}"));
        // OpenRouter answers a MALFORMED token with "Missing Authentication
        // header" (their words for "that is not the shape of our keys") -
        // translate it into something a user can act on. A well-formed wrong
        // key gets their "User not found." and passes through as is.
        let msg = if openrouter && msg.contains("Missing Authentication") {
            "That does not look like an OpenRouter key. Theirs start with sk-or-v1-.".to_owned()
        } else {
            msg
        };
        return Json(json!({"ok": false, "message": msg})).into_response();
    }
    // 200 alone isn't compatibility - a random web server says 200 too. The
    // key-info probe carries `data`; a model list carries a `data` ARRAY.
    let compatible = if openrouter {
        body.get("data").is_some()
    } else {
        body.get("data").and_then(Value::as_array).is_some()
    };
    if !compatible {
        return Json(json!({
            "ok": false,
            "message": "Something answered, but not an OpenAI-style API. Check the base URL."
        }))
        .into_response();
    }
    let msg = if key.is_empty() {
        "Answers without a key."
    } else {
        "Key works."
    };
    Json(json!({"ok": true, "message": msg})).into_response()
}

/// `POST /api/cloud/{id}/v1/audio/transcriptions` - a clip to a provider's
/// speech-to-text endpoint, keyed server-side.
///
/// The body is `multipart/form-data` and is forwarded BYTE FOR BYTE, content
/// type included: a multipart body whose `boundary=` parameter is rewritten or
/// dropped is unparseable, which is the same reason the runner relay
/// (`relay_transcriptions`) does it this way. Nothing here inspects the form,
/// so `file`, `model`, `language` and `response_format` are the caller's
/// business and the Studio speaks the OpenAI dialect straight through.
///
/// Anthropic has no transcription endpoint at all, so that kind is refused
/// here rather than 404'ing against their base URL with the user's key
/// attached.
///
/// `?model=` is for the USAGE LEDGER only. The form's own `model` field is
/// what the provider reads and stays authoritative; this route never touches
/// it. The ledger needs the name too, and parsing multipart to recover one
/// field would mean exactly the inspection this relay avoids - so the caller
/// says it twice, once for the provider and once for us.
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let (kind, base, key) = match load_secret(&state, &id).await {
        Ok(secret) => secret,
        Err(response) => return response,
    };
    if kind == "anthropic" {
        return err(
            StatusCode::BAD_REQUEST,
            "Anthropic has no speech-to-text endpoint. Use an OpenAI-style endpoint \
             (OpenAI or OpenRouter) for transcription."
                .into(),
        );
    }
    if key.is_empty() && !state.db.connection_allows_unauthenticated(&id) {
        return err(
            StatusCode::BAD_REQUEST,
            "no API key saved for this endpoint".into(),
        );
    }
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_owned();
    let request = HTTP
        .post(format!("{base}/audio/transcriptions"))
        .header(axum::http::header::CONTENT_TYPE, ct)
        .body(body);
    let request = if key.is_empty() {
        request
    } else {
        request.bearer_auth(&key)
    };
    let res = match request.send().await {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("provider not answering: {e}"),
            );
        }
    };
    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let text = res.text().await.unwrap_or_default();
    // The provider's own words on failure, not a status code of ours - a 25 MB
    // limit or an unsupported format is something the user can act on, and
    // only they can say which it was.
    if !status.is_success() {
        let detail = serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| {
                v.pointer("/error/message")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| text.chars().take(400).collect());
        return err(status, format!("transcription failed: {detail}"));
    }
    // One ledger row, same as every chat request gets from the SSE tap - a
    // transcription is not SSE, so it books itself here. Without this the
    // Studio would show the money on the turn and the endpoint's totals would
    // quietly disagree with the bill.
    if let Ok(v) = serde_json::from_str::<Value>(&text) {
        let u = &v["usage"];
        let input = u["input_tokens"].as_u64().unwrap_or(0);
        let output = u["output_tokens"].as_u64().unwrap_or(0);
        let cost = u["cost"].as_f64();
        // Seconds are what a whisper-class model actually bills, and it
        // reports no tokens at all; `duration` is the fallback for a provider
        // that times the clip without pricing it.
        let seconds = u["seconds"].as_f64().or_else(|| v["duration"].as_f64());
        if input > 0 || output > 0 || cost.is_some() || seconds.is_some() {
            let model = q.get("model").map(String::as_str).unwrap_or("(unnamed)");
            if let Err(e) = state
                .db
                .insert_cloud_usage_row(&id, model, input, output, 0, cost, seconds)
            {
                tracing::warn!(endpoint = %id, "cloud transcription usage row failed: {e}");
            }
        }
    }
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        text,
    )
        .into_response()
}

// ── the chat relay ──────────────────────────────────────────────────────────

/// `POST /api/cloud/{id}/v1/responses` - the Studio's chat seam for a cloud
/// lane. Same contract as the runner relay (a Responses request in, Responses
/// SSE out), with the dialect translation in between.
pub async fn responses(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    body: axum::body::Bytes,
) -> Response {
    let (kind, base, key) = match load_secret(&state, &id).await {
        Ok(secret) => secret,
        Err(response) => return response,
    };
    if key.is_empty() && !state.db.connection_allows_unauthenticated(&id) {
        return err(
            StatusCode::BAD_REQUEST,
            "no API key saved for this endpoint".into(),
        );
    }
    let req_body: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")),
    };
    // The builtin clock tool, read before the body builders (every builder
    // strips `tools`; the loop below re-declares what it serves). A junk
    // timezone dies here as a 400, never at the provider.
    let clock = match crate::cloud_loop::extract_clock(&req_body) {
        Ok(c) => c,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    let model = bare_model(
        req_body.get("model").and_then(Value::as_str).unwrap_or(""),
        &id,
    );
    // A pick pinned to one provider rides as "model@Provider" (the Studio's
    // per-provider breakdown) - translate to OpenRouter's documented
    // provider-routing preference. No fallbacks: the user chose that
    // provider's price/quant deliberately, silently routing elsewhere would
    // betray the choice.
    let (model, pinned) = if is_openrouter(&base) {
        match model.rsplit_once('@') {
            Some((m, p)) if !m.is_empty() && !p.is_empty() => (m.to_owned(), Some(p.to_owned())),
            _ => (model, None),
        }
    } else {
        (model, None)
    };
    let built = match kind.as_str() {
        "openai" => openai_body(&model, &req_body).map(|b| ("responses", b)),
        // OpenRouter speaks Responses natively and its
        // surface carries the full dial set + provider routing - ride it and
        // stream through verbatim. Every other OpenAI-compatible server
        // keeps chat/completions, the universal compat surface.
        "openai-compat" if is_openrouter(&base) => {
            openrouter_responses_body(&model, &req_body, pinned.as_deref())
                .map(|b| ("responses", b))
        }
        "openai-compat" => compat_body(&model, &req_body, pinned.as_deref(), false)
            .map(|b| ("chat/completions", b)),
        "anthropic" => anthropic_body(&model, &req_body).map(|b| ("messages", b)),
        other => Err(format!("unknown kind \"{other}\"")),
    };
    let (path, mut out_body) = match built {
        Ok(x) => x,
        Err(m) => return err(StatusCode::BAD_REQUEST, m),
    };
    // Output-cap belt: the Studio clamps its reply budget to the pick's
    // published max-output, but a pick stored before that number was recorded
    // plans from the context window alone and overshoots by a hair - observed
    // 128411 vs claude-sonnet-5's 128000, the whole send dead on the
    // provider's 400. The stored metadata lives here, so the relay enforces
    // it whatever the client computed.
    if let Some(cap) = state.db.cloud_endpoint_out_cap(&id, &model) {
        for field in ["max_output_tokens", "max_tokens"] {
            if let Some(v) = out_body.get(field).and_then(Value::as_u64)
                && v > cap
            {
                out_body[field] = json!(cap);
            }
        }
    }

    // Connectors on a Responses-wire lane run the manager-hosted MCP agent
    // loop: tools listed and executed here (credentials never
    // leave the box), declared to the provider as function tools, runner-
    // shaped mcp items streamed to the Studio. Other lanes still strip tools
    // until their adapters land.
    if path != "chat/completions" {
        let specs = crate::cloud_loop::extract_specs(&req_body);
        if !specs.is_empty() || clock.is_some() {
            let url = format!("{base}/{path}");
            let key2 = key.clone();
            let anthropic = path == "messages";
            let post = move |b: Value| {
                let r = HTTP.post(&url).json(&b);
                if key2.is_empty() {
                    if anthropic {
                        r.header("anthropic-version", ANTHROPIC_VERSION)
                    } else {
                        r
                    }
                } else if anthropic {
                    r.header("x-api-key", &key2)
                        .header("anthropic-version", ANTHROPIC_VERSION)
                } else {
                    r.bearer_auth(&key2)
                        .header("X-OpenRouter-Title", "Paddock")
                        .header("X-Title", "Paddock")
                }
            };
            // The caller's own tool-call ceiling, read off their request: the
            // loop below is what spends it, and only the Responses wire has
            // the field to say it with (anthropic_body drops it, and every
            // dialect strips it before the provider sees it).
            let max_tool_calls = req_body
                .get("max_tool_calls")
                .and_then(Value::as_u64)
                .map(|n| n as usize);
            let (tx, rx) = futures::channel::mpsc::unbounded::<String>();
            if anthropic {
                tokio::spawn(crate::cloud_loop::run_anthropic(
                    specs,
                    clock,
                    out_body,
                    max_tool_calls,
                    post,
                    tx,
                ));
            } else {
                tokio::spawn(crate::cloud_loop::run(
                    specs,
                    clock,
                    out_body,
                    max_tool_calls,
                    post,
                    tx,
                ));
            }
            let mut out = Response::new(metered_body(
                rx,
                state.db.clone(),
                id.clone(),
                model.clone(),
            ));
            out.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("text/event-stream"),
            );
            return out;
        }
    }

    // The pooled client: no total timeout (a long generation streams for
    // minutes; its connect_timeout keeps a dead host from hanging the lane),
    // and connection reuse takes the cold TCP/TLS handshake out of every
    // chat send's TTFT.
    let mut req = HTTP.post(format!("{base}/{path}")).json(&out_body);
    req = match kind.as_str() {
        "anthropic" if key.is_empty() => req.header("anthropic-version", ANTHROPIC_VERSION),
        _ if key.is_empty() => req,
        "anthropic" => req
            .header("x-api-key", &key)
            .header("anthropic-version", ANTHROPIC_VERSION),
        // OpenRouter app attribution (their docs): X-OpenRouter-Title is the
        // current standard, X-Title the legacy spelling - send both; harmless
        // on other providers. HTTP-Referer (the app's permanent identity URL)
        // is deliberately not set until the product URL is decided - it
        // becomes the immutable app id in their rankings.
        _ => req
            .bearer_auth(&key)
            .header("X-OpenRouter-Title", "Paddock")
            .header("X-Title", "Paddock"),
    };
    let res = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return err(
                StatusCode::BAD_GATEWAY,
                format!("provider not answering: {}", errchain(&e)),
            );
        }
    };
    if !res.status().is_success() {
        // the provider's own error message - both dialects use
        // {error:{message}}, exactly what the Studio's error path reads.
        // OpenRouter's outer message can be a useless wrapper, so the real
        // detail is hoisted into it when present.
        let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let error = crate::provider_error::response(res).await;
        let body = axum::body::Body::from(json!({"error":error}).to_string());
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json"),
        );
        return out;
    }

    // A Responses-wire provider (OpenAI, OpenRouter) already speaks the
    // Studio's dialect: stream through verbatim. The rest run the
    // frame-by-frame translation pump.
    if path == "responses" {
        let mut out = Response::new(metered_bytes(
            res,
            state.db.clone(),
            id.clone(),
            model.clone(),
        ));
        out.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/event-stream"),
        );
        return out;
    }
    let translator: Translator = if kind == "anthropic" {
        Translator::Anthropic(AnthropicStream::default())
    } else {
        Translator::Compat(CompatStream::default())
    };
    let (tx, rx) = futures::channel::mpsc::unbounded::<String>();
    tokio::spawn(pump(res, translator, tx));
    let mut out = Response::new(metered_body(
        rx,
        state.db.clone(),
        id.clone(),
        model.clone(),
    ));
    out.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/event-stream"),
    );
    out
}

#[allow(clippy::result_large_err)]
async fn load_secret(state: &AppState, id: &str) -> Result<(String, String, String), Response> {
    // Keychain may block on OS access control. Never run it on a Tokio I/O
    // worker, and never misreport a locked/unavailable key as a deleted model.
    let db = state.db.clone();
    let id = id.to_owned();
    match tokio::task::spawn_blocking(move || db.cloud_endpoint_secret(&id)).await {
        Ok(Ok(Some(value))) => Ok(value),
        Ok(Ok(None)) => Err(err(StatusCode::NOT_FOUND, "The saved connection no longer exists.".into())),
        _ => Err(err(StatusCode::SERVICE_UNAVAILABLE, "The saved credential is unavailable. Unlock Keychain or check the connection in Manager.".into())),
    }
}

// ── cloud usage ledger tap  ──────────────────────────────────────
// Every cloud lane's studio-bound stream is Responses-dialect SSE whose
// terminal (response.completed/incomplete) carries usage - tokens on every
// provider, per-request cost on OpenRouter. The tap watches the OUTBOUND
// stream (one seam covers verbatim passthrough, the translator pumps and the
// MCP agent loop alike) and writes one cloud_usage row when the stream ends.
// The loop pre-aggregates its rounds into the terminal it forwards, so the
// row for a tool turn is the whole turn, not the last round.
struct UsageScan {
    buf: Vec<u8>,
    usage: Option<Value>,
}

impl UsageScan {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            usage: None,
        }
    }

    fn feed(&mut self, chunk: &[u8]) {
        self.buf.extend_from_slice(chunk);
        // a terminal frame embeds the response's whole output; anything past
        // this cap is not a frame we care to assemble
        const CAP: usize = 8 << 20;
        if self.buf.len() > CAP {
            self.buf.clear();
            return;
        }
        while let Some(i) = self.buf.windows(2).position(|w| w == b"\n\n") {
            let frame: Vec<u8> = self.buf.drain(..i + 2).collect();
            let Ok(text) = std::str::from_utf8(&frame) else {
                continue;
            };
            let data: String = text
                .lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .collect::<Vec<_>>()
                .join("");
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(&data) else {
                continue;
            };
            match v["type"].as_str().unwrap_or("") {
                "response.completed" | "response.incomplete"
                    if !v["response"]["usage"].is_null() =>
                {
                    self.usage = Some(v["response"]["usage"].clone());
                }
                _ => {}
            }
        }
    }

    fn finish(self, db: &crate::store::Store, endpoint: &str, model: &str) {
        let Some(u) = self.usage else { return };
        let input = u["input_tokens"].as_u64().unwrap_or(0);
        let output = u["output_tokens"].as_u64().unwrap_or(0);
        let reasoning = u["output_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .unwrap_or(0);
        let cost = u["cost"].as_f64();
        if input == 0 && output == 0 && cost.is_none() {
            return; // a failed/empty turn is not a ledger row
        }
        if let Err(e) = db.insert_cloud_usage(endpoint, model, input, output, reasoning, cost) {
            tracing::warn!(%endpoint, "cloud usage row failed: {e}");
        }
    }
}

/// Wrap a string-channel body (translator pumps, the MCP loop) with the tap.
fn metered_body(
    rx: futures::channel::mpsc::UnboundedReceiver<String>,
    db: std::sync::Arc<crate::store::Store>,
    endpoint: String,
    model: String,
) -> axum::body::Body {
    use futures::StreamExt;
    let (tx2, rx2) = futures::channel::mpsc::unbounded::<String>();
    tokio::spawn(async move {
        let mut scan = UsageScan::new();
        let mut rx = rx;
        while let Some(s) = rx.next().await {
            scan.feed(s.as_bytes());
            // a gone client stops forwarding but keeps draining, so the turn
            // that already billed still lands in the ledger
            let _ = tx2.unbounded_send(s);
        }
        scan.finish(&db, &endpoint, &model);
    });
    axum::body::Body::from_stream(rx2.map(Ok::<_, std::convert::Infallible>))
}

/// Wrap a provider byte stream (the verbatim Responses passthrough) with the
/// tap. Bytes forward untouched; the scanner assembles frames itself, so
/// multi-byte characters split across chunks never corrupt the passthrough.
fn metered_bytes(
    res: reqwest::Response,
    db: std::sync::Arc<crate::store::Store>,
    endpoint: String,
    model: String,
) -> axum::body::Body {
    use futures::StreamExt;
    let (tx, rx) = futures::channel::mpsc::unbounded::<axum::body::Bytes>();
    tokio::spawn(async move {
        let mut scan = UsageScan::new();
        let mut s = res.bytes_stream();
        while let Some(chunk) = s.next().await {
            let Ok(b) = chunk else { break };
            scan.feed(&b);
            let _ = tx.unbounded_send(b);
        }
        scan.finish(&db, &endpoint, &model);
    });
    axum::body::Body::from_stream(rx.map(Ok::<_, std::convert::Infallible>))
}

/// `GET /api/cloud/usage` - the ledger's per-endpoint totals (all time +
/// trailing 24h). Costs are provider-reported; endpoints whose provider
/// reports none show cost null, never zero.
pub async fn usage(State(state): State<Arc<AppState>>) -> Response {
    match state.db.cloud_usage_summary() {
        Ok(rows) => Json(json!({"endpoints": rows})).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `POST /api/cloud/mcp-approvals/{id}` - the Studio's approval card, cloud
/// edition: resolves a call the agent loop parked. Same contract as the
/// runner relay path.
pub async fn approval(Path(id): Path<String>, Json(doc): Json<Value>) -> Response {
    let approve = doc.get("approve").and_then(Value::as_bool).unwrap_or(false);
    if crate::cloud_loop::resolve_approval(&id, approve) {
        Json(json!({"ok": true, "approved": approve})).into_response()
    } else {
        err(
            StatusCode::NOT_FOUND,
            "no pending approval with that id".into(),
        )
    }
}

fn err(status: StatusCode, msg: String) -> Response {
    (
        status,
        Json(json!({"error": {"type": "cloud_error", "message": msg}})),
    )
        .into_response()
}

/// reqwest's Display hides the interesting part: "error sending request for
/// url" says nothing about why - a transient blip at api.openai.com reads as a
/// mystery. Walk the source chain so a DNS failure, connect
/// timeout or TLS error names itself in the message the user sees.
fn errchain(e: &dyn std::error::Error) -> String {
    let mut parts = vec![e.to_string()];
    let mut cur = e.source();
    while let Some(s) = cur {
        let t = s.to_string();
        if parts.last() != Some(&t) {
            parts.push(t);
        }
        cur = s.source();
    }
    parts.join(": ")
}

/// Strip the Studio's `cloud:{endpoint}:` id prefix if the client sent its own
/// composite id. Split on the KNOWN prefix, never on ':' - provider model ids
/// carry colons of their own (OpenRouter's `:free` variants).
fn bare_model(model: &str, endpoint_id: &str) -> String {
    let prefix = format!("cloud:{endpoint_id}:");
    model.strip_prefix(&prefix).unwrap_or(model).to_owned()
}

// ── request translation ─────────────────────────────────────────────────────

/// Paddock-only fields that must not reach any provider: per-request server
/// config (`chat_template_kwargs`, `file_metadata`, `pdf_mode`, `max_pages`)
/// and the server-tools list (a cloud lane advertises none, but strip
/// defensively - an OpenAI `mcp` tool with only a server_label would 400).
const LOCAL_ONLY_FIELDS: &[&str] = &[
    "chat_template_kwargs",
    "file_metadata",
    "pdf_mode",
    "max_pages",
    "tools",
];

/// The OpenAI families that always think: they take reasoning.effort and
/// REJECT explicit temperature/top_p.
fn openai_reasoning_family(model: &str) -> bool {
    model.starts_with("gpt-5") || ["o1", "o3", "o4"].iter().any(|p| model.starts_with(p))
}

fn validate_output_cap(body: &Value) -> Result<(), String> {
    let Some(value) = body.get("max_output_tokens") else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .map(|_| ())
        .ok_or_else(|| "max_output_tokens must be a non-negative integer".to_owned())
}

/// OpenAI native: the Studio's Responses body is the wire format - sanitize
/// rather than rebuild. Sampling arrives only when the user set a dial (the
/// sampler popover; untouched means absent), so explicit temperature/top_p
/// ride to the models that take them and are dropped for the reasoning
/// families that reject them (the popover says so). top_k and seed don't
/// exist on the Responses API at all.
fn openai_body(model: &str, body: &Value) -> Result<Value, String> {
    validate_output_cap(body)?;
    let mut out = body.clone();
    let obj = out
        .as_object_mut()
        .ok_or("request body must be a JSON object")?;
    obj.insert("model".into(), json!(model));
    // a stateless relay must not leave a transcript in the provider's
    // response storage (OpenAI stores by default)
    obj.insert("store".into(), json!(false));
    for k in LOCAL_ONLY_FIELDS {
        obj.remove(*k);
    }
    // Of the dial set, OpenAI's Responses API accepts temperature and top_p
    // only - the rest are paddock/llama.cpp-family extensions that would 400.
    for k in [
        "top_k",
        "seed",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ] {
        obj.remove(k);
    }
    if openai_reasoning_family(model) {
        for k in ["temperature", "top_p"] {
            obj.remove(k);
        }
    }
    // Reasoning summaries must be ASKED for on OpenAI (reasoning.summary) or
    // the model thinks in silence; the param is rejected by non-reasoning
    // models, so gate on the id families that reason (gpt-5*, o1/o3/o4*).
    // The Studio's effort choice (reasoning.effort) rides through untouched -
    // summary is merged in beside it, never over it.
    if openai_reasoning_family(model) {
        let r = obj.entry("reasoning").or_insert_with(|| json!({}));
        if let Some(ro) = r.as_object_mut()
            && !ro.contains_key("summary")
        {
            ro.insert("summary".into(), json!("auto"));
        }
    }
    if let Some(items) = obj.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            let Some(parts) = item.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                check_file_part(part)?;
                if let Some(p) = part.as_object_mut() {
                    // part-level paddock extensions (page ranges, pdf route)
                    p.remove("pages");
                    p.remove("pdf_mode");
                }
            }
        }
    }
    Ok(out)
}

/// OpenRouter's native Responses lane (schemas unchanged from
/// the beta): same sanitize-not-rebuild as the OpenAI leg - the Studio's
/// Responses body is the wire format - but OpenRouter's surface carries the
/// full dial set (top_k/min_p ride verbatim; the multiplicative repeat knob
/// travels under their repetition_penalty spelling), the unified `reasoning`
/// param for the thinking toggle, and provider pinning. Their API is
/// stateless - store/previous_response_id are rejected - so neither ever
/// rides. chat/completions remains the documented fallback lane (compat_body
/// with openrouter=true) if this flip ever needs reverting.
fn openrouter_responses_body(
    model: &str,
    body: &Value,
    pinned_provider: Option<&str>,
) -> Result<Value, String> {
    validate_output_cap(body)?;
    let mut out = body.clone();
    let obj = out
        .as_object_mut()
        .ok_or("request body must be a JSON object")?;
    obj.insert("model".into(), json!(model));
    obj.remove("store");
    obj.remove("previous_response_id");
    // the Studio's thinking intent, read before the local-only strip removes
    // its carrier
    let think = body
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool);
    for k in LOCAL_ONLY_FIELDS {
        obj.remove(*k);
    }
    if let Some(v) = obj.remove("repeat_penalty")
        && !v.is_null()
    {
        obj.insert("repetition_penalty".into(), v);
    }
    // Their Responses surface documents the EFFORT form of the unified
    // reasoning param (the {enabled} form is the chat/completions spelling
    // and is silently dropped here - zero reasoning
    // events). summary "auto" (the OpenAI Responses spelling) is what makes
    // the thinking VISIBLE - without it Anthropic adaptive models return the
    // reasoning item as signature-only with an empty summary (effort alone =
    // 0 reasoning chars, +summary = live
    // reasoning_text deltas). Toggle on = both; off = omit (provider
    // default).
    if think == Some(true) {
        obj.insert(
            "reasoning".into(),
            json!({"effort": "high", "summary": "auto"}),
        );
    }
    // The Studio's `web_search` tool (the OpenAI Responses shape its local
    // runners execute themselves) becomes OpenRouter's web plugin - their
    // Exa-backed search that works on every model. The tool entry itself is
    // stripped: it is ours to translate, not the provider's to refuse.
    // Results come back as url_citation annotations on the message rather
    // than web_search_call items - inline citations, no search card.
    // Read off the ORIGINAL body, not `obj`: `tools` joined LOCAL_ONLY_FIELDS
    // after this translation was written, so the strip above had already
    // removed the carrier and the plugin never engaged - the same
    // read-before-strip rule as `think`.
    let wanted_web = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|ts| {
            ts.iter()
                .any(|t| t.get("type").and_then(Value::as_str) == Some("web_search"))
        });
    if wanted_web {
        obj.insert("plugins".into(), json!([{"id": "web"}]));
    }
    if let Some(p) = pinned_provider {
        obj.insert(
            "provider".into(),
            json!({"order": [p], "allow_fallbacks": false}),
        );
    }
    if let Some(items) = obj.get_mut("input").and_then(Value::as_array_mut) {
        for item in items {
            let Some(parts) = item.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts {
                check_file_part(part)?;
                if let Some(p) = part.as_object_mut() {
                    p.remove("pages");
                    p.remove("pdf_mode");
                }
            }
        }
    }
    Ok(out)
}

/// OpenAI-compatible (OpenRouter, or any /chat/completions server): rebuild as
/// a chat.completions request. `pinned_provider` is OpenRouter's routing
/// preference for a provider-pinned pick; `openrouter` switches how the
/// Studio's thinking intent travels.
fn compat_body(
    model: &str,
    body: &Value,
    pinned_provider: Option<&str>,
    openrouter: bool,
) -> Result<Value, String> {
    validate_output_cap(body)?;
    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = body.get("instructions").and_then(Value::as_str)
        && !sys.is_empty()
    {
        messages.push(json!({"role": "system", "content": sys}));
    }
    for item in body
        .get("input")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = item.get("content").unwrap_or(&Value::Null);
        if let Some(s) = content.as_str() {
            messages.push(json!({"role": role, "content": s}));
            continue;
        }
        let mut parts: Vec<Value> = Vec::new();
        for part in content.as_array().map(Vec::as_slice).unwrap_or(&[]) {
            check_file_part(part)?;
            match part.get("type").and_then(Value::as_str) {
                Some("input_text") => {
                    parts.push(json!({"type": "text", "text": part.get("text").and_then(Value::as_str).unwrap_or("")}));
                }
                Some("input_image") => {
                    let url = part.get("image_url").and_then(Value::as_str).unwrap_or("");
                    let detail = part.get("detail").and_then(Value::as_str).unwrap_or("auto");
                    parts.push(
                        json!({"type": "image_url", "image_url": {"url": url, "detail": detail}}),
                    );
                }
                Some("input_file") => {
                    parts.push(json!({"type": "file", "file": {
                        "filename": part.get("filename").and_then(Value::as_str).unwrap_or("file.pdf"),
                        "file_data": part.get("file_data").and_then(Value::as_str).unwrap_or(""),
                    }}));
                }
                _ => {}
            }
        }
        messages.push(json!({"role": role, "content": parts}));
    }
    let mut out = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        // without this most servers omit token counts from the stream
        "stream_options": {"include_usage": true},
    });
    for k in [
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
    ] {
        if let Some(v) = body.get(k)
            && !v.is_null()
        {
            out[k] = v.clone();
        }
    }
    // llama.cpp-family servers call it repeat_penalty; OpenRouter (and the
    // vLLM class behind it) spell the same multiplicative knob
    // repetition_penalty - renamed in transit, never dropped.
    if let Some(v) = body.get("repeat_penalty")
        && !v.is_null()
    {
        out[if openrouter {
            "repetition_penalty"
        } else {
            "repeat_penalty"
        }] = v.clone();
    }
    if let Some(m) = body.get("max_output_tokens").and_then(Value::as_u64) {
        out["max_tokens"] = json!(m);
    }
    if let Some(p) = pinned_provider {
        out["provider"] = json!({"order": [p], "allow_fallbacks": false});
    }
    // The Studio's thinking intent (chat_template_kwargs.enable_thinking,
    // the local qwen dialect) must not just vanish on cloud lanes: Gemma 4
    // on OpenRouter declares reasoning default_enabled:false and never
    // thought because nobody ASKED. OpenRouter's unified
    // `reasoning` param carries it (unsupported models drop the param);
    // other OpenAI-compatible servers (vLLM et al) honor
    // chat_template_kwargs verbatim, so it forwards as is.
    if let Some(think) = body
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool)
    {
        if openrouter {
            out["reasoning"] = json!({"enabled": think});
        } else {
            out["chat_template_kwargs"] = body["chat_template_kwargs"].clone();
        }
    }
    Ok(out)
}

/// Which thinking dialect an Anthropic model speaks. Opus/Sonnet 4.6+ and
/// the whole 5 family take ADAPTIVE (`{"type":"adaptive"}`, effort via
/// output_config); the legacy enabled+budget shape 400s on them. Everything
/// older - sonnet-4-5, opus-4-1, haiku-4-5, the 3.x line - only knows
/// enabled+budget_tokens. Version read: the number pair right after the tier
/// word; the 3.x ids invert the order (claude-3-5-sonnet) so they miss the
/// prefix and land on legacy, which is right, and date stamps
/// (claude-sonnet-4-20250514) are too big for the minor slot. Fable/Mythos
/// exist only as 5-family models (incl. un-numbered "preview" ids).
fn anthropic_adaptive_thinking(model: &str) -> bool {
    if model.starts_with("claude-fable-") || model.starts_with("claude-mythos-") {
        return true;
    }
    ["opus", "sonnet", "haiku"].iter().any(|tier| {
        model
            .strip_prefix(&format!("claude-{tier}-"))
            .is_some_and(|rest| {
                let mut nums = rest.split('-').map_while(|s| s.parse::<u32>().ok());
                let major = nums.next().unwrap_or(0);
                let minor = nums.next().filter(|n| *n < 100).unwrap_or(0);
                major > 4 || (major == 4 && minor >= 6)
            })
    })
}

/// Anthropic /v1/messages: system string + strictly alternating messages with
/// typed content blocks; consecutive same-role turns are merged (the API
/// rejects them, and a compare history can legitimately produce user->user
/// after failed turns are filtered out).
fn anthropic_body(model: &str, body: &Value) -> Result<Value, String> {
    validate_output_cap(body)?;
    let mut messages: Vec<Value> = Vec::new();
    for item in body
        .get("input")
        .and_then(Value::as_array)
        .unwrap_or(&Vec::new())
    {
        let role = match item.get("role").and_then(Value::as_str) {
            Some("assistant") => "assistant",
            _ => "user",
        };
        let content = item.get("content").unwrap_or(&Value::Null);
        let mut blocks: Vec<Value> = Vec::new();
        if let Some(s) = content.as_str() {
            if !s.is_empty() {
                blocks.push(json!({"type": "text", "text": s}));
            }
        } else {
            for part in content.as_array().map(Vec::as_slice).unwrap_or(&[]) {
                check_file_part(part)?;
                match part.get("type").and_then(Value::as_str) {
                    Some("input_text") => {
                        let t = part.get("text").and_then(Value::as_str).unwrap_or("");
                        if !t.is_empty() {
                            blocks.push(json!({"type": "text", "text": t}));
                        }
                    }
                    Some("input_image") => {
                        let url = part.get("image_url").and_then(Value::as_str).unwrap_or("");
                        let Some((media, data)) = parse_data_uri(url) else {
                            return Err(
                                "image attachments must be inline data for a cloud endpoint".into(),
                            );
                        };
                        blocks.push(json!({"type": "image", "source": {
                            "type": "base64", "media_type": media, "data": data,
                        }}));
                    }
                    Some("input_file") => {
                        let uri = part.get("file_data").and_then(Value::as_str).unwrap_or("");
                        let Some((_, data)) = parse_data_uri(uri) else {
                            return Err(
                                "file attachments must be inline data for a cloud endpoint".into(),
                            );
                        };
                        blocks.push(json!({"type": "document", "source": {
                            "type": "base64", "media_type": "application/pdf", "data": data,
                        }}));
                    }
                    _ => {}
                }
            }
        }
        if blocks.is_empty() {
            continue;
        }
        match messages.last_mut() {
            Some(last) if last["role"] == role => {
                if let Some(arr) = last["content"].as_array_mut() {
                    arr.extend(blocks);
                }
            }
            _ => messages.push(json!({"role": role, "content": blocks})),
        }
    }
    let mut out = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "max_tokens": body.get("max_output_tokens").and_then(Value::as_u64)
            .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS),
    });
    if let Some(sys) = body.get("instructions").and_then(Value::as_str)
        && !sys.is_empty()
    {
        out["system"] = json!(sys);
    }
    // The Studio's thinking toggle. Two Anthropic dialects, per their
    // platform docs: Opus/Sonnet 4.6+ and the whole 5 family
    // take ADAPTIVE - the model decides depth itself, there is no budget
    // (budget_tokens 400s: "thinking.type.enabled is not supported for this
    // model"), display must be requested because it defaults to "omitted"
    // there (an empty thought fold otherwise), and toggle off rides
    // explicitly as {"type":"disabled"} since the 5 family thinks by
    // default. Older models want the legacy enabled shape with an explicit
    // budget (min 1024, strictly under max_tokens): half the output cap
    // leaves room for the answer; a cap too small to fit the minimum budget
    // plus an answer keeps thinking off rather than erroring.
    let mut thinking_on = false;
    let think = body
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool);
    if anthropic_adaptive_thinking(model) {
        match think {
            Some(true) => {
                out["thinking"] = json!({"type": "adaptive", "display": "summarized"});
                thinking_on = true;
            }
            Some(false) => {
                out["thinking"] = json!({"type": "disabled"});
            }
            None => {}
        }
    } else if think == Some(true) {
        let max = out["max_tokens"]
            .as_u64()
            .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS);
        if max >= 2048 {
            let budget = (max / 2).clamp(1024, 32_000);
            out["thinking"] = json!({"type": "enabled", "budget_tokens": budget});
            thinking_on = true;
        }
    }
    // Sampling arrives only when the user set a dial (untouched means absent),
    // so explicit values ride - except under extended thinking, which
    // documents that they must be unset (the popover says so). A model that
    // rejects a dial (Sonnet 5: "`temperature` is deprecated") errors in its
    // own words, loudly, and the user unsets it - never a silent drop of an
    // explicit choice. Anthropic's temperature scale tops out at 1.0 where
    // OpenAI's runs to 2.0: clamp instead of erroring the whole lane.
    if !thinking_on {
        if let Some(t) = body.get("temperature").and_then(Value::as_f64) {
            out["temperature"] = json!(t.min(1.0));
        }
        if let Some(v) = body.get("top_p")
            && !v.is_null()
        {
            out["top_p"] = v.clone();
        }
        // top_k 0 is the dial's off position (same meaning as absent) - and
        // Anthropic's top_k wants a positive integer, so 0 stays home.
        if let Some(k) = body.get("top_k").and_then(Value::as_u64)
            && k > 0
        {
            out["top_k"] = json!(k);
        }
    }
    Ok(out)
}

/// Cloud providers read PDFs natively and nothing else - a runner extracts
/// .docx/.xlsx/... locally, a provider can't. Refuse loudly instead of sending
/// bytes the model will never see (no-silent-failures).
fn check_file_part(part: &Value) -> Result<(), String> {
    if part.get("type").and_then(Value::as_str) != Some("input_file") {
        return Ok(());
    }
    let name = part.get("filename").and_then(Value::as_str).unwrap_or("");
    let is_pdf = name.to_ascii_lowercase().ends_with(".pdf")
        || part
            .get("file_data")
            .and_then(Value::as_str)
            .is_some_and(|d| d.starts_with("data:application/pdf"));
    if is_pdf {
        Ok(())
    } else {
        Err(format!(
            "This cloud endpoint reads PDFs only - \"{name}\" can't be attached here. Local models extract it."
        ))
    }
}

/// `data:<media>;base64,<payload>` -> (media, payload).
fn parse_data_uri(s: &str) -> Option<(&str, &str)> {
    let rest = s.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media = meta.strip_suffix(";base64")?;
    Some((media, data))
}

// ── stream translation ──────────────────────────────────────────────────────

enum Translator {
    Compat(CompatStream),
    Anthropic(AnthropicStream),
}

impl Translator {
    fn on_data(&mut self, data: &str) -> Vec<String> {
        match self {
            Translator::Compat(s) => s.on_data(data),
            Translator::Anthropic(s) => s.on_data(data),
        }
    }
    /// The upstream closed: make sure the Studio still gets a terminal event
    /// (a stream that just stops would leave the turn hanging with no usage
    /// and no error).
    fn finish(&mut self) -> Vec<String> {
        match self {
            Translator::Compat(s) => s.terminal(),
            Translator::Anthropic(s) => s.terminal(),
        }
    }
    fn is_done(&self) -> bool {
        match self {
            Translator::Compat(s) => s.done,
            Translator::Anthropic(s) => s.done,
        }
    }
}

/// Read the provider's SSE body, translate frame by frame, forward as
/// Responses SSE frames. Frames are split at the byte level so a UTF-8
/// character (or a CRLF pair) falling across network chunks can't corrupt.
async fn pump(
    res: reqwest::Response,
    mut tr: Translator,
    tx: futures::channel::mpsc::UnboundedSender<String>,
) {
    use futures::StreamExt;
    let mut stream = res.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let send = |tx: &futures::channel::mpsc::UnboundedSender<String>, events: Vec<String>| {
        for ev in events {
            let _ = tx.unbounded_send(format!("data: {ev}\n\n"));
        }
    };
    while let Some(chunk) = stream.next().await {
        // the Studio hung up (Stop, closed tab): returning drops `stream`,
        // which closes the provider connection so it stops generating - and
        // stops billing - instead of streaming into a dead channel
        if tx.is_closed() {
            return;
        }
        let Ok(bytes) = chunk else {
            // a cut after the terminal event is just the socket closing -
            // only an unterminated answer is worth an error in the thread
            if !tr.is_done() {
                send(
                    &tx,
                    vec![failed_event(
                        "connection to the provider was lost mid-answer",
                    )],
                );
            }
            return;
        };
        buf.extend_from_slice(&bytes);
        while let Some((at, len)) = frame_delim(&buf) {
            let frame = String::from_utf8_lossy(&buf[..at]).into_owned();
            buf.drain(..at + len);
            if let Some(data) = frame_data(&frame) {
                send(&tx, tr.on_data(&data));
            }
        }
    }
    if let Some(data) = frame_data(&String::from_utf8_lossy(&buf)) {
        send(&tx, tr.on_data(&data));
    }
    send(&tx, tr.finish());
}

/// Earliest SSE frame delimiter: `\n\n` or `\r\n\r\n`, whichever starts first.
fn frame_delim(b: &[u8]) -> Option<(usize, usize)> {
    for i in 0..b.len().saturating_sub(1) {
        if b[i..].starts_with(b"\r\n\r\n") {
            return Some((i, 4));
        }
        if b[i] == b'\n' && b[i + 1] == b'\n' {
            return Some((i, 2));
        }
    }
    None
}

/// Join a frame's `data:` lines (the Studio-side readSse twin).
fn frame_data(frame: &str) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            parts.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

fn delta_event(kind: &str, delta: &str) -> String {
    json!({"type": kind, "delta": delta}).to_string()
}

fn failed_event(msg: &str) -> String {
    json!({"type": "response.failed", "response": {"error": {"message": msg}}}).to_string()
}

/// chat.completions SSE -> Responses events. Reasoning arrives as
/// `delta.reasoning` (OpenRouter's normalized field) or `delta.reasoning_content`
/// (the DeepSeek-style field vLLM/SGLang serve); both map to reasoning_text.
#[derive(Default)]
struct CompatStream {
    usage: Option<Value>,
    hit_length: bool,
    done: bool,
    /// OpenRouter stamps every chunk with the provider that actually served
    /// the request. Surfacing it is not decoration: the :free gemma routed
    /// between two hosts of which one silently ignored the reasoning param,
    /// and "thinks sometimes" is undiagnosable until the provider is
    /// visible.
    provider: Option<String>,
}

impl CompatStream {
    fn on_data(&mut self, data: &str) -> Vec<String> {
        if data.trim() == "[DONE]" {
            return self.terminal();
        }
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return vec![];
        };
        if let Some(p) = v
            .get("provider")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            self.provider = Some(p.to_string());
        }
        if let Some(e) = v.get("error") {
            self.done = true;
            let error = crate::provider_error::normalize(e, None);
            return vec![
                json!({"type":"response.failed","response":{"status":"failed","error":error}})
                    .to_string(),
            ];
        }
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(json!({
                "input_tokens": u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0),
                "output_tokens": u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0),
                "output_tokens_details": {
                    "reasoning_tokens": u
                        .get("completion_tokens_details")
                        .and_then(|d| d.get("reasoning_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                },
            }));
        }
        let mut out = Vec::new();
        if let Some(choice) = v
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        {
            if let Some(d) = choice.get("delta") {
                for (field, kind) in [
                    ("content", "response.output_text.delta"),
                    ("reasoning", "response.reasoning_text.delta"),
                    ("reasoning_content", "response.reasoning_text.delta"),
                ] {
                    if let Some(s) = d.get(field).and_then(Value::as_str)
                        && !s.is_empty()
                    {
                        out.push(delta_event(kind, s));
                    }
                }
            }
            if choice.get("finish_reason").and_then(Value::as_str) == Some("length") {
                self.hit_length = true;
            }
        }
        out
    }

    fn terminal(&mut self) -> Vec<String> {
        if self.done {
            return vec![];
        }
        self.done = true;
        let usage = self.usage.take().unwrap_or_else(|| json!({}));
        let mut resp = json!({"usage": usage});
        if let Some(p) = self.provider.take() {
            resp["provider"] = json!(p);
        }
        if self.hit_length {
            resp["incomplete_details"] = json!({"reason": "max_output_tokens"});
            vec![json!({"type": "response.incomplete", "response": resp}).to_string()]
        } else {
            vec![json!({"type": "response.completed", "response": resp}).to_string()]
        }
    }
}

/// Anthropic /v1/messages SSE -> Responses events. input_tokens ride
/// message_start, output_tokens the message_delta frames, thinking blocks map
/// to reasoning_text.
#[derive(Default)]
struct AnthropicStream {
    input_tokens: u64,
    output_tokens: u64,
    hit_max: bool,
    done: bool,
}

impl AnthropicStream {
    fn on_data(&mut self, data: &str) -> Vec<String> {
        let Ok(v) = serde_json::from_str::<Value>(data) else {
            return vec![];
        };
        match v.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                self.input_tokens = v
                    .get("message")
                    .and_then(|m| m.get("usage"))
                    .and_then(|u| u.get("input_tokens"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                vec![]
            }
            Some("content_block_delta") => {
                let Some(d) = v.get("delta") else {
                    return vec![];
                };
                match d.get("type").and_then(Value::as_str) {
                    Some("text_delta") => d
                        .get("text")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(|s| vec![delta_event("response.output_text.delta", s)])
                        .unwrap_or_default(),
                    Some("thinking_delta") => d
                        .get("thinking")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(|s| vec![delta_event("response.reasoning_text.delta", s)])
                        .unwrap_or_default(),
                    _ => vec![],
                }
            }
            Some("message_delta") => {
                if let Some(t) = v
                    .get("usage")
                    .and_then(|u| u.get("output_tokens"))
                    .and_then(Value::as_u64)
                {
                    self.output_tokens = t;
                }
                if v.get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                    == Some("max_tokens")
                {
                    self.hit_max = true;
                }
                vec![]
            }
            Some("message_stop") => self.terminal(),
            Some("error") => {
                self.done = true;
                let msg = v
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or("the provider reported an error");
                vec![failed_event(msg)]
            }
            _ => vec![], // ping etc.
        }
    }

    fn terminal(&mut self) -> Vec<String> {
        if self.done {
            return vec![];
        }
        self.done = true;
        let usage = json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        });
        if self.hit_max {
            vec![
                json!({"type": "response.incomplete", "response": {
                    "usage": usage,
                    "incomplete_details": {"reason": "max_output_tokens"},
                }})
                .to_string(),
            ]
        } else {
            vec![json!({"type": "response.completed", "response": {"usage": usage}}).to_string()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloud_adapters_reject_malformed_output_caps() {
        for cap in [json!(1.5), json!(-1), json!("128"), json!(true)] {
            let body = json!({"model": "m", "input": [], "max_output_tokens": cap});
            assert!(
                serde_json::from_value::<paddock_api::responses::ResponsesRequest>(body.clone())
                    .is_err()
            );
            let rejected = [
                openai_body("m", &body).is_err(),
                openrouter_responses_body("m", &body, None).is_err(),
                compat_body("m", &body, None, false).is_err(),
                anthropic_body("m", &body).is_err(),
            ];
            assert_eq!(rejected, [true; 4]);
        }
    }

    #[test]
    fn cloud_adapters_accept_null_or_zero_output_caps() {
        for cap in [Value::Null, json!(0)] {
            let body = json!({"model": "m", "input": [], "max_output_tokens": cap});
            assert!(
                serde_json::from_value::<paddock_api::responses::ResponsesRequest>(body.clone())
                    .is_ok()
            );
            assert!(openai_body("m", &body).is_ok());
            assert!(openrouter_responses_body("m", &body, None).is_ok());
            assert!(compat_body("m", &body, None, false).is_ok());
            assert!(anthropic_body("m", &body).is_ok());
        }
    }

    #[test]
    fn bare_model_strips_only_the_known_prefix() {
        // OpenRouter variant suffixes carry colons - never split on ':'
        assert_eq!(
            bare_model("cloud:ep1:meta/llama-3:free", "ep1"),
            "meta/llama-3:free"
        );
        assert_eq!(bare_model("gpt-5.2", "ep1"), "gpt-5.2");
        assert_eq!(
            bare_model("cloud:other:gpt-5.2", "ep1"),
            "cloud:other:gpt-5.2"
        );
    }

    #[test]
    fn openai_body_sanitizes_in_place() {
        let body = json!({
            "model": "cloud:e:gpt-5.2", "stream": true, "temperature": 0.7, "top_k": 40,
            "seed": 7, "min_p": 0.05, "repeat_penalty": 1.1, "presence_penalty": 1.5,
            "frequency_penalty": 0.2,
            "chat_template_kwargs": {"enable_thinking": true}, "pdf_mode": "text",
            "tools": [{"type": "mcp", "server_label": "x"}],
            "input": [{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,QUJD", "pages": 3, "pdf_mode": "text"},
            ]}],
        });
        let out = openai_body("gpt-5.2", &body).unwrap();
        assert_eq!(out["model"], "gpt-5.2");
        assert_eq!(out["store"], false);
        assert_eq!(out["stream"], true);
        for k in [
            "temperature",
            "top_k",
            "seed",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "chat_template_kwargs",
            "pdf_mode",
            "tools",
        ] {
            assert!(out.get(k).is_none(), "{k} must be stripped");
        }
        let part = &out["input"][0]["content"][1];
        assert!(part.get("pages").is_none() && part.get("pdf_mode").is_none());
        assert_eq!(part["filename"], "a.pdf");
        // non-reasoning models take explicit temperature/top_p on Responses
        let body =
            json!({"model": "m", "temperature": 0.5, "top_p": 0.9, "top_k": 40, "input": []});
        let out = openai_body("gpt-4o", &body).unwrap();
        assert_eq!(out["temperature"], 0.5);
        assert_eq!(out["top_p"], 0.9);
        assert!(out.get("top_k").is_none(), "top_k is not a Responses field");
    }

    #[test]
    fn context_management_rides_the_responses_legs() {
        // Phase-4 check (context-management plan): a caller's
        // context_management / truncation opt-ins must reach Responses-wire
        // providers verbatim - the providers apply their own compaction, the
        // relay only sanitizes. The rebuild legs are exempt by shape:
        // chat/completions has no such params, and Anthropic's dialect is a
        // different config (edits array) an OpenAI-shaped entry can't ride.
        let body = json!({
            "model": "m", "input": "hi",
            "context_management": [{"type": "compaction", "compact_threshold": 5000}],
            "truncation": "auto",
        });
        let out = openai_body("gpt-5.2", &body).unwrap();
        assert_eq!(out["context_management"][0]["compact_threshold"], 5000);
        assert_eq!(out["truncation"], "auto");
        let out = openrouter_responses_body("q", &body, None).unwrap();
        assert_eq!(out["context_management"][0]["type"], "compaction");
        assert_eq!(out["truncation"], "auto");
    }

    #[test]
    fn provider_pin_becomes_routing_preference() {
        let body =
            json!({"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}]});
        let out =
            compat_body("deepseek/deepseek-v4-flash", &body, Some("DeepInfra"), true).unwrap();
        assert_eq!(out["provider"]["order"][0], "DeepInfra");
        assert_eq!(out["provider"]["allow_fallbacks"], false);
        assert!(
            compat_body("m", &body, None, false)
                .unwrap()
                .get("provider")
                .is_none()
        );
    }

    #[test]
    fn openai_reasoning_summary_only_for_reasoning_families() {
        let body = json!({"model": "m", "input": []});
        assert_eq!(
            openai_body("gpt-5.2", &body).unwrap()["reasoning"]["summary"],
            "auto"
        );
        assert_eq!(
            openai_body("o3-mini", &body).unwrap()["reasoning"]["summary"],
            "auto"
        );
        assert!(
            openai_body("gpt-4.1", &body)
                .unwrap()
                .get("reasoning")
                .is_none()
        );
    }

    #[test]
    fn normalize_model_carries_openrouter_metadata() {
        let m = json!({
            "id": "deepseek/deepseek-v4-flash:free",
            "name": "DeepSeek V4 Flash (free)",
            "created": 1786000000u64,
            "description": "fast agentic coding model. ".repeat(20),
            "context_length": 163840,
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["reasoning", "tools"],
        });
        let out = normalize_model(&m).unwrap();
        assert_eq!(out["display"], "DeepSeek V4 Flash (free)");
        assert_eq!(out["ctx"], 163840);
        assert_eq!(out["vision"], true);
        assert_eq!(out["promptPrice"], 0.0);
        assert_eq!(out["reasoning"], true);
        assert_eq!(out["tools"], true);
        assert_eq!(out["free"], true);
        assert!(out["blurb"].as_str().unwrap().chars().count() <= 200);
        // a bare OpenAI-style row degrades to id + created, nothing invented
        let out = normalize_model(&json!({"id": "gpt-5.2", "created": 5})).unwrap();
        assert_eq!(out["created"], 5);
        assert!(out.get("display").is_none());
        assert!(out.get("tools").is_none());
        assert!(
            out.get("maxOut").is_none(),
            "no output ceiling is honest; a guessed one is not"
        );
        assert!(out.get("promptPrice").is_none());
        assert!(out.get("free").is_none());
        // The output ceiling is its own number, nested under top_provider and
        // routinely a fraction of the context window - the Studio's reply cap
        // reads it so "Model maximum" cannot ask for more than the model can
        // emit (deepseek-v4-flash-0731: 1M context, 384k output).
        let out = normalize_model(&json!({
            "id": "deepseek/deepseek-v4-flash-0731",
            "context_length": 1048576,
            "top_provider": {"context_length": 1048576, "max_completion_tokens": 384000},
        }))
        .unwrap();
        assert_eq!(out["ctx"], 1048576);
        assert_eq!(out["maxOut"], 384000);
        // ...and the per-endpoint list carries it bare, so both shapes read.
        let out = normalize_model(&json!({"id": "m", "max_completion_tokens": 8192})).unwrap();
        assert_eq!(out["maxOut"], 8192);

        // A speech model: audio in, a TRANSCRIPT out. The Studio turns `asr`
        // into kind:'transcriber', which is what routes a clip to
        // /audio/transcriptions instead of the chat wire.
        let out = normalize_model(&json!({
            "id": "openai/whisper-large-v3",
            "name": "OpenAI: Whisper Large V3",
            "context_length": 0,
            "architecture": {
                "input_modalities": ["audio"],
                "output_modalities": ["transcription"],
            },
        }))
        .unwrap();
        assert_eq!(out["asr"], true);
        assert_eq!(out["vision"], false, "audio in is not image in");
        // ...and a context window of 0 is the ABSENCE of one, not a fact. Every
        // transcription model reports 0 because the question does not apply,
        // and "0 ctx" in the picker is a worse lie than no number.
        assert!(out.get("ctx").is_none(), "0 is not a context window: {out}");

        // A chat model is not one, however it is spelled.
        let out = normalize_model(&json!({
            "id": "c", "architecture": {"output_modalities": ["text"]},
        }))
        .unwrap();
        assert!(out.get("asr").is_none());
    }

    #[test]
    fn normalized_prices_reject_nonfinite_and_dynamic_values() {
        for price in ["-1", "NaN", "inf", "1e999"] {
            let value = normalize_model(&json!({
                "id": "a/b", "pricing": {"prompt": price, "completion": price},
                "supported_parameters": []
            }))
            .unwrap();
            assert!(value.get("promptPrice").is_none());
            assert!(value.get("completionPrice").is_none());
            assert!(value.get("free").is_none());
            assert_eq!(value["tools"], false);
        }
    }

    /// A NATIVE OpenAI list states no modalities at all, so the speech models
    /// have to be recognised by their (stable, well-known) ids or they arrive
    /// as chat models and refuse the first thing typed at them.
    #[test]
    fn openai_native_speech_models_are_offered_and_marked() {
        for id in ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] {
            assert!(openai_chat_model(id), "{id} must reach the picker");
            let mut m = json!({ "id": id });
            stamp_native_asr("openai", &mut m);
            assert_eq!(m["asr"], true, "{id} must be marked as speech");
        }
        // ...while what we still cannot serve stays out.
        for id in [
            "tts-1",
            "text-embedding-3-large",
            "dall-e-3",
            "gpt-4o-realtime-preview",
        ] {
            assert!(
                !openai_chat_model(id),
                "{id} would only ever be a broken lane"
            );
        }
        // A chat model is not stamped, and no other provider is guessed at by
        // name - OpenRouter states its modalities and normalize_model reads them.
        let mut m = json!({ "id": "gpt-5.2" });
        stamp_native_asr("openai", &mut m);
        assert!(m.get("asr").is_none());
        let mut m = json!({ "id": "some-whisper-clone" });
        stamp_native_asr("openai-compat", &mut m);
        assert!(
            m.get("asr").is_none(),
            "only the native OpenAI list is guessed at"
        );
        // OpenRouter's router pseudo-models price as "-1" = dynamic pricing -
        // never a number to display, and definitely not free
        let out = normalize_model(&json!({
            "id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"},
        }))
        .unwrap();
        assert!(out.get("promptPrice").is_none());
        assert!(out.get("completionPrice").is_none());
        assert!(out.get("free").is_none());
    }

    #[test]
    fn openrouter_detection_gates_the_sort_param() {
        assert!(is_openrouter("https://openrouter.ai/api/v1"));
        assert!(!is_openrouter("https://api.openai.com/v1"));
        assert!(!is_openrouter("http://127.0.0.1:11700/v1"));
    }

    #[test]
    fn compat_body_builds_chat_completions() {
        let body = json!({
            "model": "m", "instructions": "be brief", "temperature": 0.5, "top_k": 40,
            "min_p": 0.05, "presence_penalty": 1.5, "repeat_penalty": 1.1,
            "max_output_tokens": 256,
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "message", "role": "assistant", "content": "hi"},
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"},
                ]},
            ],
        });
        let out = compat_body("org/model:free", &body, None, false).unwrap();
        assert_eq!(out["model"], "org/model:free");
        assert_eq!(out["messages"][0]["role"], "system");
        assert_eq!(out["messages"][1]["content"], "hello");
        assert_eq!(out["messages"][3]["content"][1]["type"], "image_url");
        assert_eq!(out["max_tokens"], 256);
        assert_eq!(out["top_k"], 40);
        assert_eq!(out["min_p"], 0.05);
        assert_eq!(out["presence_penalty"], 1.5);
        // llama.cpp-family compat servers keep the llama.cpp spelling...
        assert_eq!(out["repeat_penalty"], 1.1);
        assert!(out.get("repetition_penalty").is_none());
        assert_eq!(out["stream_options"]["include_usage"], true);
        // ...and OpenRouter (vLLM-class) gets the same knob renamed, not dropped
        let or = compat_body("org/model:free", &body, None, true).unwrap();
        assert_eq!(or["repetition_penalty"], 1.1);
        assert!(or.get("repeat_penalty").is_none());
    }

    #[test]
    fn thinking_intent_travels_per_dialect() {
        // OpenRouter: the qwen-dialect enable_thinking becomes their unified
        // `reasoning` param (Gemma 4 declares default_enabled:false and only
        // thinks when asked); other compat servers get chat_template_kwargs
        // verbatim (vLLM et al honor it).
        let body = |think: bool| {
            json!({
                "model": "m", "chat_template_kwargs": {"enable_thinking": think},
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            })
        };
        let out = compat_body("google/gemma-4-26b-a4b-it", &body(true), None, true).unwrap();
        assert_eq!(out["reasoning"]["enabled"], true);
        assert!(out.get("chat_template_kwargs").is_none());
        let out = compat_body("google/gemma-4-26b-a4b-it", &body(false), None, true).unwrap();
        assert_eq!(out["reasoning"]["enabled"], false);
        let out = compat_body("qwen3.5-9b", &body(true), None, false).unwrap();
        assert_eq!(out["chat_template_kwargs"]["enable_thinking"], true);
        assert!(out.get("reasoning").is_none());
        // no intent sent -> neither param appears on either dialect
        let bare =
            json!({"model": "m", "input": [{"type": "message", "role": "user", "content": "hi"}]});
        let out = compat_body("m", &bare, None, true).unwrap();
        assert!(out.get("reasoning").is_none() && out.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn anthropic_body_merges_consecutive_roles_and_honors_defaults() {
        let body = json!({
            "model": "m", "instructions": "sys", "temperature": 1.8, "top_p": 0.8,
            "input": [
                {"type": "message", "role": "user", "content": "one"},
                {"type": "message", "role": "user", "content": "two"},
                {"type": "message", "role": "assistant", "content": "ok"},
            ],
        });
        let out = anthropic_body("claude-x", &body).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2, "consecutive user turns merge");
        assert_eq!(msgs[0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(out["system"], "sys");
        // explicit dials ride (clamped to Anthropic's 1.0 scale); a model
        // that rejects one errors in its own words rather than us silently
        // dropping an explicit choice
        assert_eq!(out["temperature"], 1.0);
        assert_eq!(out["top_p"], 0.8);
        assert_eq!(out["max_tokens"], ANTHROPIC_DEFAULT_MAX_TOKENS);
        // top_k 0 is the dial's off position (= absent); a positive one rides;
        // the extension dials never do (Anthropic has no such knobs)
        let with = |k: &str, v: serde_json::Value| {
            let mut b = body.clone();
            b[k] = v;
            anthropic_body("claude-x", &b).unwrap()
        };
        assert!(with("top_k", json!(0)).get("top_k").is_none());
        assert_eq!(with("top_k", json!(20))["top_k"], 20);
        assert!(with("min_p", json!(0.05)).get("min_p").is_none());
        assert!(
            with("presence_penalty", json!(1.5))
                .get("presence_penalty")
                .is_none()
        );
        assert!(
            with("repeat_penalty", json!(1.1))
                .get("repeat_penalty")
                .is_none()
        );
    }

    #[test]
    fn openrouter_responses_keeps_dials_reasoning_and_pin() {
        // The native Responses lane: full dial set rides,
        // repeat knob renamed to their spelling, thinking toggle becomes the
        // unified reasoning param, pinned picks route, paddock-only fields
        // and statefulness never reach the wire.
        let body = json!({
            "model": "m", "input": "hi", "stream": true, "store": true,
            "temperature": 0.7, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1,
            "chat_template_kwargs": {"enable_thinking": true},
            "file_metadata": "off", "tools": [{"type": "mcp", "server_label": "x"}],
        });
        let out =
            openrouter_responses_body("qwen/qwen3.5-9b", &body, Some("deepinfra/bf16")).unwrap();
        assert_eq!(out["model"], "qwen/qwen3.5-9b");
        assert_eq!(out["temperature"], 0.7);
        assert_eq!(out["top_k"], 40);
        assert_eq!(out["min_p"], 0.05);
        assert_eq!(out["repetition_penalty"], 1.1);
        assert!(out.get("repeat_penalty").is_none());
        assert_eq!(out["reasoning"]["effort"], "high");
        // summary "auto" is what makes thinking VISIBLE on their surface -
        // effort alone returns signature-only reasoning items (probed)
        assert_eq!(out["reasoning"]["summary"], "auto");
        assert_eq!(out["provider"]["order"][0], "deepinfra/bf16");
        assert_eq!(out["provider"]["allow_fallbacks"], false);
        assert!(out.get("store").is_none());
        assert!(out.get("chat_template_kwargs").is_none());
        assert!(out.get("file_metadata").is_none());
        assert!(out.get("tools").is_none());
        // untouched dials stay absent, thinking-off travels explicitly
        let body = json!({
            "model": "m", "input": "hi",
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let out = openrouter_responses_body("m", &body, None).unwrap();
        // toggle off = omit (their Responses reasoning param has no
        // enabled/exclude form; absent = provider default)
        assert!(out.get("reasoning").is_none());
        assert!(out.get("provider").is_none());
        assert!(out.get("top_k").is_none());
    }

    #[test]
    fn anthropic_thinking_suppresses_explicit_sampling() {
        // thinking documents that sampler params must be unset - both dialects
        let body = json!({
            "model": "m", "temperature": 0.7, "top_p": 0.8, "max_output_tokens": 32768,
            "chat_template_kwargs": {"enable_thinking": true},
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        });
        let out = anthropic_body("claude-sonnet-5", &body).unwrap();
        assert_eq!(out["thinking"]["type"], "adaptive");
        assert!(out.get("temperature").is_none() && out.get("top_p").is_none());
        let out = anthropic_body("claude-sonnet-4-5", &body).unwrap();
        assert_eq!(out["thinking"]["type"], "enabled");
        assert!(out.get("temperature").is_none() && out.get("top_p").is_none());
    }

    #[test]
    fn anthropic_thinking_toggle_maps_to_extended_thinking() {
        // the LEGACY dialect (pre-4.6): enabled + an explicit budget
        let body = |think: bool, max: u64| {
            json!({
                "model": "m", "max_output_tokens": max,
                "chat_template_kwargs": {"enable_thinking": think},
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            })
        };
        let out = anthropic_body("claude-sonnet-4-5", &body(true, 32768)).unwrap();
        assert_eq!(out["thinking"]["type"], "enabled");
        assert_eq!(out["thinking"]["budget_tokens"], 16384);
        // toggle off -> no thinking block at all
        assert!(
            anthropic_body("m", &body(false, 32768))
                .unwrap()
                .get("thinking")
                .is_none()
        );
        // a cap too small for the minimum budget + an answer leaves thinking off
        assert!(
            anthropic_body("m", &body(true, 1024))
                .unwrap()
                .get("thinking")
                .is_none()
        );
        // the budget is bounded even under huge caps
        let out = anthropic_body("m", &body(true, 128_000)).unwrap();
        assert_eq!(out["thinking"]["budget_tokens"], 32_000);
    }

    #[test]
    fn anthropic_claude5_takes_adaptive_thinking() {
        // The compare-lane error this guards: Sonnet 5 400s on the legacy
        // enabled+budget shape. 4.6+/5.x get adaptive (no budget, display
        // requested - it defaults to "omitted" there); off rides explicitly
        // as disabled because the 5 family thinks by default.
        let body = |think: bool| {
            json!({
                "model": "m", "max_output_tokens": 32768,
                "chat_template_kwargs": {"enable_thinking": think},
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            })
        };
        let out = anthropic_body("claude-sonnet-5", &body(true)).unwrap();
        assert_eq!(out["thinking"]["type"], "adaptive");
        assert_eq!(out["thinking"]["display"], "summarized");
        assert!(out["thinking"].get("budget_tokens").is_none());
        let out = anthropic_body("claude-sonnet-5", &body(false)).unwrap();
        assert_eq!(out["thinking"]["type"], "disabled");
    }

    #[test]
    fn anthropic_thinking_dialect_gate() {
        for id in [
            "claude-sonnet-5",
            "claude-opus-5",
            "claude-sonnet-5-latest",
            "claude-opus-4-8",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-fable-5",
            "claude-mythos-preview",
        ] {
            assert!(anthropic_adaptive_thinking(id), "{id} should be adaptive");
        }
        for id in [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-5",
            "gpt-5.2",
            "m",
        ] {
            assert!(!anthropic_adaptive_thinking(id), "{id} should be legacy");
        }
    }

    #[test]
    fn openai_effort_rides_and_summary_merges_beside_it() {
        let body = json!({"model": "m", "reasoning": {"effort": "high"}, "input": []});
        let out = openai_body("gpt-5.2", &body).unwrap();
        assert_eq!(out["reasoning"]["effort"], "high");
        assert_eq!(out["reasoning"]["summary"], "auto");
    }

    /// Was `openai_list_keeps_only_chat_models_and_stamps_reasoning`, and it
    /// used to assert whisper-1 and gpt-4o-transcribe were kept out. They are
    /// offered now: the rule was never "chat only", it was "only
    /// what we can serve", and a transcriber became something we can serve.
    /// `openai_native_speech_models_are_offered_and_marked` owns that side.
    #[test]
    fn openai_list_drops_what_cannot_be_served_and_stamps_reasoning() {
        for id in [
            "tts-1-hd",
            "text-embedding-3-small",
            "omni-moderation-latest",
            "davinci-002",
            "gpt-image-1",
            "gpt-3.5-turbo-instruct",
        ] {
            assert!(!openai_chat_model(id), "{id} cannot be served as a lane");
        }
        for id in [
            "gpt-5.2",
            "gpt-4o",
            "o3-mini",
            "gpt-4.1-nano",
            "gpt-5-chat-latest",
        ] {
            assert!(openai_chat_model(id), "{id} chats");
        }
        let mut m = json!({"id": "o3-mini"});
        stamp_native_reasoning("openai", &mut m);
        assert_eq!(m["reasoning"], true);
        let mut m = json!({"id": "gpt-4o"});
        stamp_native_reasoning("openai", &mut m);
        assert!(m.get("reasoning").is_none());
        let mut m = json!({"id": "claude-sonnet-5"});
        stamp_native_reasoning("anthropic", &mut m);
        assert_eq!(m["reasoning"], true);
    }

    #[test]
    fn non_pdf_files_are_refused_loudly() {
        let body = json!({"model": "m", "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_file", "filename": "notes.docx", "file_data": "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,AA=="},
        ]}]});
        for out in [
            openai_body("m", &body).err(),
            compat_body("m", &body, None, false).err(),
            anthropic_body("m", &body).err(),
        ] {
            let msg = out.expect("must refuse");
            assert!(msg.contains("notes.docx"), "names the file: {msg}");
        }
    }

    #[test]
    fn compat_stream_translates_deltas_usage_and_done() {
        let mut s = CompatStream::default();
        let out = s.on_data(r#"{"choices":[{"delta":{"reasoning":"hm"}}]}"#);
        assert!(out[0].contains("reasoning_text.delta"));
        let out = s.on_data(r#"{"choices":[{"delta":{"content":"Hi"}}]}"#);
        assert!(out[0].contains("output_text.delta") && out[0].contains("Hi"));
        assert!(s.on_data(r#"{"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5,"completion_tokens_details":{"reasoning_tokens":2}}}"#).is_empty());
        let done = s.on_data("[DONE]");
        assert!(done[0].contains("response.completed"));
        assert!(done[0].contains("\"input_tokens\":3") && done[0].contains("\"output_tokens\":5"));
        assert!(done[0].contains("\"reasoning_tokens\":2"));
        // upstream close after [done] must not double-terminate
        assert!(s.terminal().is_empty());
    }

    #[test]
    fn provider_errors_surface_the_raw_detail() {
        // the 429 that hid behind "Provider returned error"
        let e = json!({
            "message": "Provider returned error", "code": 429,
            "metadata": {"raw": "google/gemma-4-31b-it:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations", "provider_name": "Google AI Studio"},
        });
        let normalized = crate::provider_error::normalize(&e, None);
        let msg = normalized["message"].as_str().unwrap();
        assert!(msg.starts_with("Google AI Studio: "));
        assert!(
            msg.contains("retry shortly"),
            "keeps the actionable part: {msg}"
        );
        assert!(
            !msg.contains("http") && !msg.contains("add your own key"),
            "drops the link dump: {msg}"
        );
        // Missing detail and a raw containing only a URL use the outer message.
        for e in [
            json!({"message": "boom"}),
            json!({"message": "boom", "metadata": {"raw": "https://x.ai"}}),
        ] {
            assert_eq!(
                crate::provider_error::normalize(&e, None)["message"],
                "boom"
            );
        }
        // the streaming error path prefers the detail too
        let mut s = CompatStream::default();
        let out = s.on_data(r#"{"error":{"message":"Provider returned error","metadata":{"raw":"model is overloaded"}}}"#);
        assert!(out[0].contains("model is overloaded"));
    }

    #[test]
    fn compat_stream_carries_the_serving_provider() {
        // OpenRouter's :free routing proved who served matters: darkbloom
        // claimed reasoning support and silently ignored it while
        // google-ai-studio honored it. The provider stamp rides the chunks
        // and must land on the terminal event.
        let mut s = CompatStream::default();
        let _ = s.on_data(r#"{"provider":"Darkbloom","choices":[{"delta":{"content":"x"}}]}"#);
        let done = s.on_data("[DONE]");
        assert!(done[0].contains("\"provider\":\"Darkbloom\""));
        // servers that don't stamp one (vLLM et al) add no field at all
        let mut s = CompatStream::default();
        let _ = s.on_data(r#"{"choices":[{"delta":{"content":"x"}}]}"#);
        assert!(!s.on_data("[DONE]")[0].contains("provider"));
    }

    #[test]
    fn compat_stream_length_maps_to_incomplete() {
        let mut s = CompatStream::default();
        let _ = s.on_data(r#"{"choices":[{"delta":{"content":"x"},"finish_reason":"length"}]}"#);
        let done = s.on_data("[DONE]");
        assert!(done[0].contains("response.incomplete"));
        assert!(done[0].contains("max_output_tokens"));
    }

    #[test]
    fn compat_stream_without_done_still_terminates() {
        let mut s = CompatStream::default();
        let _ = s.on_data(r#"{"choices":[{"delta":{"content":"x"}}]}"#);
        let done = s.terminal();
        assert!(done[0].contains("response.completed"));
    }

    #[test]
    fn anthropic_stream_full_sequence() {
        let mut s = AnthropicStream::default();
        assert!(
            s.on_data(r#"{"type":"message_start","message":{"usage":{"input_tokens":11}}}"#)
                .is_empty()
        );
        assert!(
            s.on_data(
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}"#
            )
            .is_empty()
        );
        let out = s.on_data(r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"mm"}}"#);
        assert!(out[0].contains("reasoning_text.delta"));
        let out = s.on_data(r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}"#);
        assert!(out[0].contains("output_text.delta") && out[0].contains("Hello"));
        assert!(s.on_data(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#).is_empty());
        let done = s.on_data(r#"{"type":"message_stop"}"#);
        assert!(done[0].contains("response.completed"));
        assert!(done[0].contains("\"input_tokens\":11") && done[0].contains("\"output_tokens\":9"));
        assert!(
            s.terminal().is_empty(),
            "no double terminal after message_stop"
        );
    }

    #[test]
    fn anthropic_stream_max_tokens_maps_to_incomplete() {
        let mut s = AnthropicStream::default();
        let _ = s.on_data(r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":4}}"#);
        let done = s.on_data(r#"{"type":"message_stop"}"#);
        assert!(done[0].contains("response.incomplete"));
    }

    #[test]
    fn frame_split_handles_both_delimiters_and_multiline_data() {
        assert_eq!(frame_delim(b"a\n\nb"), Some((1, 2)));
        assert_eq!(frame_delim(b"a\r\n\r\nb"), Some((1, 4)));
        assert_eq!(frame_delim(b"ab"), None);
        assert_eq!(
            frame_data("event: x\ndata: one\ndata: two"),
            Some("one\ntwo".into())
        );
        assert_eq!(frame_data(": comment"), None);
    }

    #[test]
    fn data_uri_parses() {
        assert_eq!(
            parse_data_uri("data:image/png;base64,AA=="),
            Some(("image/png", "AA=="))
        );
        assert_eq!(parse_data_uri("http://x/y.png"), None);
    }
}
