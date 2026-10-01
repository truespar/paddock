//! Public table API behavior with a deterministic backend; no GPU required.
#![allow(clippy::unwrap_used)]
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use paddock_engine::tabular::{Info, Output, Tabular, TabularBackend};
use paddock_models::kumo::{KumoConfig, Table, Task};
use paddock_runner::{
    routes::{AppState, router},
    tabular::TabularModel,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

struct Fake;
impl TabularBackend for Fake {
    type Context = ();
    fn cache_budget(&self) -> u64 {
        1 << 20
    }
    fn context_bytes(&self, _r: usize, _c: usize) -> u64 {
        16
    }
    fn fit(&mut self, t: &Table<'_>) -> Result<((), Output), String> {
        t.validate_context(&Task::Classification)?;
        Ok((
            (),
            Output {
                values: Vec::new(),
                gpu_seconds: 0.,
                workspace_bytes: 0,
            },
        ))
    }
    fn query(&mut self, _ctx: &(), _x: &[f32], rows: usize) -> Result<Output, String> {
        Ok(Output {
            values: vec![0.; rows * 10],
            gpu_seconds: 0.,
            workspace_bytes: 256,
        })
    }
    fn info(&self) -> Info {
        Info {
            config: KumoConfig {
                task: Task::Classification,
                size: "small".into(),
                cell: 128,
                embedding_layers: 4,
                inducing: 128,
                hidden: 512,
                layers: 12,
                heads: 8,
                query_kv_heads: 8,
            },
            weight_bytes: 1234,
        }
    }
    fn predict(&mut self, t: &Table<'_>) -> Result<Output, String> {
        Ok(Output {
            values: vec![0.; t.query_rows * 10],
            gpu_seconds: 0.001,
            workspace_bytes: 256,
        })
    }
}
fn state() -> AppState {
    let mut s = AppState::for_tests(None);
    s.tabular = Some(TabularModel {
        id: "kumo-small-classifier".into(),
        service: Tabular::spawn(|| Ok(Fake)).unwrap(),
    });
    s
}
fn body() -> Value {
    json!({"model":"kumo-small-classifier","preprocessing":"prepared","context":[[1.],[2.]],"targets":[0.,1.],"query":[[null]],"categorical":[false]})
}
#[tokio::test]
async fn raw_context_lifecycle_and_credentials() {
    let mut s = state();
    s.auth_key = Some("tabular-secret".into());
    let app = router(Arc::new(s));
    async fn invoke(
        app: &axum::Router,
        method: &str,
        path: &str,
        body: Value,
        authorized: bool,
    ) -> (StatusCode, Value) {
        let mut req = Request::builder()
            .method(method)
            .uri(path)
            .header("content-type", "application/json");
        if authorized {
            req = req.header("authorization", "Bearer tabular-secret");
        }
        let res = app
            .clone()
            .oneshot(req.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    let raw = json!({"preprocessing":"sdm_v1","context":[[1,"a"],[2,"b"],[3,null]],"targets":["yes","no","yes"],"categorical":[false,true],"num_estimators":4});
    assert_eq!(
        invoke(&app, "POST", "/v1/tabular/contexts", raw.clone(), false)
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    let (status, fit) = invoke(&app, "POST", "/v1/tabular/contexts", raw, true).await;
    assert_eq!(status, StatusCode::OK, "{fit}");
    let id = fit["context_id"].as_str().unwrap();
    let query = json!({"context_id":id,"query":[[4,"unknown"]]});
    let (status, result) =
        invoke(&app, "POST", "/v1/tabular/predictions", query.clone(), true).await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["classes"], json!(["yes", "no"]));
    assert_eq!(result["predictions"][0]["label"], "yes");
    assert_eq!(result["cache_hit"], true);
    let bad = json!({"context_id":id,"query":[[4]]});
    assert_eq!(
        invoke(&app, "POST", "/v1/tabular/predictions", bad, true)
            .await
            .0,
        StatusCode::BAD_REQUEST
    );
    let path = format!("/v1/tabular/contexts/{id}");
    assert_eq!(
        invoke(&app, "DELETE", &path, Value::Null, false).await.0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        invoke(&app, "DELETE", &path, Value::Null, true).await.0,
        StatusCode::OK
    );
    assert_eq!(
        invoke(&app, "POST", "/v1/tabular/predictions", query, true)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
}
async fn call(
    s: AppState,
    uri: &str,
    body: Option<String>,
    key: Option<&str>,
) -> (StatusCode, Value) {
    let mut req = if body.is_some() {
        Request::post(uri)
    } else {
        Request::get(uri)
    }
    .header("content-type", "application/json");
    if let Some(key) = key {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    let res = router(Arc::new(s))
        .oneshot(req.body(Body::from(body.unwrap_or_default())).unwrap())
        .await
        .unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}
#[tokio::test]
async fn prediction_and_discovery_are_not_chat() {
    let (status, v) = call(
        state(),
        "/v1/tabular/predictions",
        Some(body().to_string()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(v["predictions"][0]["probabilities"], json!([0.5, 0.5]));
    assert_eq!(v["usage"]["context_rows"], 2);
    assert!(v["usage"].get("total_tokens").is_none());
    let (_, models) = call(state(), "/v1/models", None, None).await;
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    let model = &models["data"][0];
    assert_eq!(model["id"], "kumo-small-classifier");
    assert!(model.to_string().contains("tabular_prediction"));
    let (_, server) = call(state(), "/api/server", None, None).await;
    assert_eq!(server["tabular"]["capabilities"]["ensemble"], true);
    let (status, _) = call(
        state(),
        "/v1/chat/completions",
        Some(json!({"messages":[{"role":"user","content":"hi"}]}).to_string()),
        None,
    )
    .await;
    assert_ne!(status, StatusCode::OK);
}
#[tokio::test]
async fn authentication_validation_and_drain_are_enforced() {
    let mut s = state();
    s.auth_key = Some("test-tabular-key".into());
    let (status, _) = call(s, "/v1/tabular/predictions", Some(body().to_string()), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let mut invalid = body();
    invalid["model"] = "different".into();
    let (status, _) = call(
        state(),
        "/v1/tabular/predictions",
        Some(invalid.to_string()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let mut invalid = body();
    invalid["preprocessing"] = "auto".into();
    let (status, _) = call(
        state(),
        "/v1/tabular/predictions",
        Some(invalid.to_string()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let s = state();
    s.drain.begin();
    let (status, _) = call(s, "/v1/tabular/predictions", Some(body().to_string()), None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}
#[tokio::test]
async fn table_body_limit_is_smaller_than_image_limit() {
    let mut raw = body().to_string();
    raw.push_str(&" ".repeat(8 * 1024 * 1024));
    let (status, _) = call(state(), "/v1/tabular/predictions", Some(raw), None).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}
