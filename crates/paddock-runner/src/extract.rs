//! Request extractors with spec-shaped rejections. axum's stock `Json`
//! answers a deserialization failure with 422 + plain text; the real OpenAI
//! and Anthropic APIs answer 400 with their error envelope - including for
//! unknown fields, which our request structs deny, so every unimplemented
//! spec parameter surfaces here by NAME instead of being silently ignored.

use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use serde_json::json;

/// serde's unknown-field error reads "unknown field `x`, expected one of ...";
/// reshape it to the OpenAI wording. Other messages pass through.
fn shape(raw: &str) -> String {
    if let Some(i) = raw.find("unknown field `") {
        let rest = &raw[i + "unknown field `".len()..];
        if let Some(j) = rest.find('`') {
            return format!("Unrecognized request argument supplied: {}", &rest[..j]);
        }
    }
    raw.to_owned()
}

/// OpenAI-enveloped JSON extractor.
pub struct OaiJson<T>(pub T);

impl<S, T> FromRequest<S> for OaiJson<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(OaiJson(v)),
            Err(rej) => Err((
                rejection_status(&rej),
                Json(ErrorBody::new(
                    "invalid_request_error",
                    shape(&rejection_text(&rej)),
                )),
            )
                .into_response()),
        }
    }
}

/// Anthropic-enveloped JSON extractor ({"type":"error","error":{...}}).
pub struct AnthJson<T>(pub T);

impl<S, T> FromRequest<S> for AnthJson<T>
where
    S: Send + Sync,
    T: serde::de::DeserializeOwned,
{
    type Rejection = Response;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(AnthJson(v)),
            Err(rej) => Err((
                rejection_status(&rej),
                Json(json!({"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": shape(&rejection_text(&rej)),
                }})),
            )
                .into_response()),
        }
    }
}

fn rejection_status(rej: &JsonRejection) -> StatusCode {
    // Syntax/schema failures use the API's 400 envelope, but a transport
    // size limit is still 413. In particular, table endpoints have a much
    // smaller limit than image/chat payloads; never disguise it as bad JSON.
    if rej.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    }
}

fn rejection_text(rej: &JsonRejection) -> String {
    match rej {
        JsonRejection::JsonDataError(e) => e.body_text(),
        JsonRejection::JsonSyntaxError(e) => e.body_text(),
        JsonRejection::MissingJsonContentType(_) => {
            "requests must have content-type: application/json".to_owned()
        }
        other => other.body_text(),
    }
}

#[cfg(test)]
mod tests {
    use super::shape;

    #[test]
    fn unknown_field_message_is_openai_worded() {
        let raw = "Failed to deserialize the JSON body into the target type: \
                   unknown field `logit_bias`, expected one of `model`, `messages`";
        assert_eq!(
            shape(raw),
            "Unrecognized request argument supplied: logit_bias"
        );
        assert_eq!(
            shape("expected value at line 1"),
            "expected value at line 1"
        );
    }
}
