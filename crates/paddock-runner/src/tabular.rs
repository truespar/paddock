//! Fitted or prepared table predictions. Not a chat/Reads capability: Kumo
//! consumes structured rows and outcomes, not natural-language instructions.
use crate::{extract::OaiJson, routes::AppState};
use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use paddock_engine::tabular::{Input, Operation, RecipeOutput, Tabular};
use paddock_models::kumo::recipe::{Cell, RawTable};
use paddock_models::kumo::{KumoConfig, Task};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc, time::Instant};

pub struct TabularModel {
    pub id: String,
    pub service: Tabular,
}
impl TabularModel {
    pub fn capabilities(&self) -> Value {
        let c = &self.service.info().config;
        json!({"tabular_prediction":true,"task":task_name(&c.task),"preprocessing":["prepared","sdm_v1"],
            "max_context_rows":4096,"max_query_rows":1024,"max_columns":500,"max_cells":131072,
            "max_classes":10,"quantiles":if c.task==Task::Regression {999} else {0},
            "context_cache":self.service.cache_budget_bytes()>0,"cache_budget_bytes":self.service.cache_budget_bytes(),"ensemble":true,"default_estimators":8,"max_estimators":16,
            "max_cached_contexts":4,"context_idle_ttl_seconds":600,"cache_eviction":"lru"})
    }
}
fn task_name(task: &Task) -> &'static str {
    match task {
        Task::Classification => "classification",
        Task::Regression => "regression",
    }
}

pub fn is_kumo(path: &Path) -> bool {
    path.is_dir()
        && std::fs::read(path.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .is_some_and(|v| v["model_type"] == "kumo_tabular")
}
pub fn load(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    budget: Option<u64>,
) -> Result<TabularModel, String> {
    // Parse before any device work so malformed exports are not mistaken
    // for unsupported devices or dispatched to a chat loader.
    let cfg = KumoConfig::read(path).map_err(|e| e.to_string())?;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if device == "metal" {
        let path = path.to_owned();
        let service = Tabular::spawn(move || {
            paddock_metal::Kumo::load(&path, budget).map_err(|e| e.to_string())
        })?;
        return Ok(TabularModel { id, service });
    }
    if device != "cuda" {
        return Err(format!(
            "Kumo-Tabular needs the native cuda or metal backend (got {device:?})"
        ));
    }
    let (path, pack) = (path.to_owned(), pack.map(Path::to_path_buf));
    let service = Tabular::spawn(move || {
        use paddock_engine::tabular::TabularBackend;
        let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
            .map_err(|e| e.to_string())?;
        crate::serving::note_device_cc(&exec);
        if let Some(b) = budget {
            exec.set_vram_budget(b);
        }
        let mut m = paddock_engine::gpu_model::kumo::GpuKumo::load(Arc::new(exec), &path)
            .map_err(|e| e.to_string())?;
        // One tiny table before the first request: every kernel's first
        // launch pays its module load, and that belongs to startup.
        let y = if cfg.task == Task::Classification {
            [0., 1.]
        } else {
            [0.5, -0.5]
        };
        m.predict(&paddock_models::kumo::Table {
            x: &[0.25, 1., -0.5, 0., 0.75, 1.],
            y: &y,
            categorical: &[false, true],
            query_rows: 1,
        })
        .map_err(|e| format!("Kumo-Tabular warm-up: {e}"))?;
        tracing::info!(
            size = %cfg.size,
            task = task_name(&cfg.task),
            weight_bytes = m.weight_bytes(),
            context_cache_bytes = m.cache_budget(),
            "Kumo-Tabular loaded on cuda"
        );
        Ok(m)
    })?;
    Ok(TabularModel { id, service })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model: Option<String>,
    /// This typed request is the explicit prepared-table path. Raw fitted
    /// recipes use RecipeRequest instead; never silently reinterpret features
    /// or regression target units in this contract.
    pub preprocessing: String,
    pub context: Vec<Vec<Option<f32>>>,
    pub targets: Vec<f32>,
    pub query: Vec<Vec<Option<f32>>>,
    pub categorical: Vec<bool>,
}
impl Request {
    fn into_input(self, cfg: &KumoConfig) -> Result<(Input, usize), String> {
        if self.preprocessing != "prepared" {
            return Err("preprocessing must be 'prepared' or 'sdm_v1'".into());
        }
        let cols = self.categorical.len();
        if self.context.len() != self.targets.len()
            || self
                .context
                .iter()
                .chain(&self.query)
                .any(|r| r.len() != cols)
        {
            return Err("context, targets, query and categorical column counts must agree".into());
        }
        let query_rows = self.query.len();
        let input = Input {
            x: self
                .context
                .into_iter()
                .chain(self.query)
                .flatten()
                .map(|v| v.unwrap_or(f32::NAN))
                .collect(),
            y: self.targets,
            categorical: self.categorical,
            query_rows,
        };
        input.table().validate(&cfg.task)?;
        let classes = if cfg.task == Task::Classification {
            let mut present = [false; 10];
            for &y in &input.y {
                present[y as usize] = true;
            }
            let n = present.iter().filter(|&&v| v).count();
            if present[..n].iter().any(|&v| !v) {
                return Err(
                    "classification targets must be contiguous class codes 0..N-1 (N <= 10)".into(),
                );
            }
            n
        } else {
            0
        };
        Ok((input, classes))
    }
}
fn err(code: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (code, Json(paddock_api::ErrorBody::new(kind, message))).into_response()
}
fn predictions(values: &[f32], task: &Task, classes: usize) -> Value {
    match task {
        Task::Classification => Value::Array(
            values
                .as_chunks::<10>()
                .0
                .iter()
                .map(|all| {
                    let logits = &all[..classes];
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp = logits
                        .iter()
                        .map(|&x| f64::from(x - max).exp())
                        .collect::<Vec<_>>();
                    let sum = exp.iter().sum::<f64>();
                    let probabilities = exp.iter().map(|x| x / sum).collect::<Vec<_>>();
                    let class = logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                        .expect("validated nonempty class set")
                        .0;
                    json!({"class":class,"probabilities":probabilities,"logits":logits})
                })
                .collect(),
        ),
        Task::Regression => Value::Array(
            values
                .as_chunks::<999>()
                .0
                .iter()
                .map(|raw| {
                    let mut q = raw.to_vec();
                    q.sort_by(f32::total_cmp);
                    json!({"median":q[499],"quantiles":q})
                })
                .collect(),
        ),
    }
}
pub async fn handle(
    State(state): State<Arc<AppState>>,
    OaiJson(payload): OaiJson<Value>,
) -> Response {
    if payload.get("context_id").is_some() {
        return match serde_json::from_value::<CachedRequest>(payload) {
            Ok(req) => {
                run_recipe(
                    &state,
                    req.model,
                    Operation::Query {
                        id: req.context_id,
                        query: req.query,
                    },
                )
                .await
            }
            Err(e) => err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                e.to_string(),
            ),
        };
    }
    if payload["preprocessing"] == "sdm_v1" {
        return match serde_json::from_value::<RecipeRequest>(payload) {
            Ok(req) => {
                run_recipe(
                    &state,
                    req.model,
                    Operation::Predict {
                        table: RawTable {
                            context: req.context,
                            targets: req.targets,
                            categorical: req.categorical,
                        },
                        query: req.query,
                        estimators: req.num_estimators,
                        seed: req.seed,
                    },
                )
                .await
            }
            Err(e) => err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                e.to_string(),
            ),
        };
    }
    let req = match serde_json::from_value::<Request>(payload) {
        Ok(v) => v,
        Err(e) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                e.to_string(),
            );
        }
    };
    let Some(model) = &state.tabular else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "this runner has no tabular model loaded",
        );
    };
    if req.model.as_ref().is_some_and(|id| id != &model.id) {
        return err(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "requested model is not served by this runner",
        );
    }
    let cfg = &model.service.info().config;
    let (input, classes) = match req.into_input(cfg) {
        Ok(x) => x,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let context = input.y.len();
    let query = input.query_rows;
    let cols = input.categorical.len();
    let start = Instant::now();
    let out = match model.service.predict(input).await {
        Ok(v) => v,
        Err(e) => {
            let status = match &e {
                paddock_engine::tabular::Error::Invalid(_) => StatusCode::BAD_REQUEST,
                paddock_engine::tabular::Error::Busy
                | paddock_engine::tabular::Error::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                paddock_engine::tabular::Error::Backend(_) => StatusCode::INTERNAL_SERVER_ERROR,
                paddock_engine::tabular::Error::NotFound => StatusCode::NOT_FOUND,
            };
            return err(status, "tabular_error", e.to_string());
        }
    };
    if out.values.len() != query * cfg.outputs() || out.values.iter().any(|v| !v.is_finite()) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "tabular_error",
            "invalid prediction output",
        );
    }
    Json(json!({"object":"tabular.prediction","model":model.id,"task":task_name(&cfg.task),"preprocessing":"prepared",
        "predictions":predictions(&out.values,&cfg.task,classes),
        "quantile_levels":if cfg.task==Task::Regression {Some((1..1000).map(|i|f64::from(i)/1000.).collect::<Vec<_>>())} else {None},
        "usage":{"context_rows":context,"query_rows":query,"columns":cols,"gpu_ms":out.gpu_seconds*1000.,
            "elapsed_ms":start.elapsed().as_secs_f64()*1000.,"workspace_bytes":out.workspace_bytes}})).into_response()
}

fn default_estimators() -> usize {
    8
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeRequest {
    model: Option<String>,
    preprocessing: String,
    context: Vec<Vec<Cell>>,
    targets: Vec<Cell>,
    categorical: Vec<bool>,
    #[serde(default)]
    query: Vec<Vec<Cell>>,
    #[serde(default = "default_estimators")]
    num_estimators: usize,
    #[serde(default)]
    seed: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedRequest {
    model: Option<String>,
    context_id: String,
    query: Vec<Vec<Cell>>,
}
pub async fn fit(
    State(state): State<Arc<AppState>>,
    OaiJson(req): OaiJson<RecipeRequest>,
) -> Response {
    if req.preprocessing != "sdm_v1" || !req.query.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "fit requires preprocessing 'sdm_v1' and no query rows",
        );
    }
    run_recipe(
        &state,
        req.model,
        Operation::Fit {
            id: format!("ctx_{}", uuid::Uuid::new_v4().simple()),
            table: RawTable {
                context: req.context,
                targets: req.targets,
                categorical: req.categorical,
            },
            estimators: req.num_estimators,
            seed: req.seed,
        },
    )
    .await
}
pub async fn release(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    run_recipe(&state, None, Operation::Release { id }).await
}
async fn run_recipe(state: &AppState, model: Option<String>, op: Operation) -> Response {
    let Some(served) = &state.tabular else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "this runner has no tabular model loaded",
        );
    };
    if model.as_ref().is_some_and(|id| id != &served.id) {
        return err(
            StatusCode::NOT_FOUND,
            "model_not_found",
            "requested model is not served by this runner",
        );
    }
    let deleting = matches!(&op, Operation::Release { .. });
    let cached = matches!(&op, Operation::Query { .. });
    let start = Instant::now();
    let out = match served.service.recipe(op).await {
        Ok(v) => v,
        Err(e) => {
            use paddock_engine::tabular::Error;
            let status = match &e {
                Error::Invalid(_) => StatusCode::BAD_REQUEST,
                Error::NotFound => StatusCode::NOT_FOUND,
                Error::Busy | Error::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                Error::Backend(_) => StatusCode::INTERNAL_SERVER_ERROR,
            };
            return err(status, "tabular_error", e.to_string());
        }
    };
    if deleting {
        return Json(json!({"object":"tabular.context","id":out.context_id,"deleted":true}))
            .into_response();
    }
    recipe_response(served, out, cached, start.elapsed().as_secs_f64())
}
fn recipe_response(
    served: &TabularModel,
    out: RecipeOutput,
    cached: bool,
    elapsed: f64,
) -> Response {
    let task = &served.service.info().config.task;
    let mut rows = predictions(&out.output.values, task, out.classes.len());
    if *task == Task::Classification {
        for row in rows.as_array_mut().expect("prediction array") {
            if let Some(i) = row["class"].as_u64() {
                row["label"] = json!(out.classes[i as usize]);
            }
        }
    }
    Json(json!({"object":if out.query_rows==0 {"tabular.context"} else {"tabular.prediction"},"model":served.id,
        "task":task_name(task),"preprocessing":"sdm_v1","recipe_seed":out.seed,"num_estimators":out.estimators,
        "context_id":out.context_id,"cache_hit":cached,"cache_bytes":out.cache_bytes,
        "classes":out.classes,"predictions":rows,
        "quantile_levels":if *task==Task::Regression {Some((1..1000).map(|i|f64::from(i)/1000.).collect::<Vec<_>>())} else {None},
        "usage":{"context_rows":out.context_rows,"query_rows":out.query_rows,"columns":out.columns,
            "gpu_ms":out.output.gpu_seconds*1000.,"elapsed_ms":elapsed*1000.,"workspace_bytes":out.output.workspace_bytes}})).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> KumoConfig {
        KumoConfig {
            task: Task::Classification,
            size: "small".into(),
            cell: 128,
            embedding_layers: 4,
            inducing: 128,
            hidden: 512,
            layers: 12,
            heads: 8,
            query_kv_heads: 8,
        }
    }
    fn req() -> Value {
        json!({"preprocessing":"prepared","context":[[1.],[2.]],"targets":[0.,1.],"query":[[null]],"categorical":[false]})
    }
    #[test]
    fn prepared_contract_is_required_and_strict() {
        let input = serde_json::from_value::<Request>(req())
            .unwrap()
            .into_input(&cfg())
            .unwrap();
        assert_eq!(input.1, 2);
        assert!(input.0.x[2].is_nan());
        let mut v = req();
        v.as_object_mut().unwrap().remove("preprocessing");
        assert!(serde_json::from_value::<Request>(v).is_err());
        let mut v = req();
        v["num_estimators"] = 8.into();
        assert!(serde_json::from_value::<Request>(v).is_err());
        let mut v = req();
        v["targets"] = json!([0, 2]);
        assert!(
            serde_json::from_value::<Request>(v)
                .unwrap()
                .into_input(&cfg())
                .is_err()
        );
        let mut v = req();
        v["query"] = json!([[1, 2]]);
        assert!(
            serde_json::from_value::<Request>(v)
                .unwrap()
                .into_input(&cfg())
                .is_err()
        );
    }
    #[test]
    fn probability_normalization_and_quantile_levels() {
        let logits = vec![1000.; 10];
        let p = predictions(&logits, &Task::Classification, 2);
        assert_eq!(p[0]["class"], 0);
        assert_eq!(p[0]["probabilities"], json!([0.5, 0.5]));
        let q = (0..999).rev().map(|i| i as f32).collect::<Vec<_>>();
        let p = predictions(&q, &Task::Regression, 0);
        assert_eq!(p[0]["median"], 499.);
        assert_eq!(p[0]["quantiles"][0], 0.);
    }
}
