//! Native-client contract evidence from the real router, without a listener,
//! model load, supervisor task or user database. The optional export is consumed
//! by Swift tests; it prevents a hand-maintained JSON fixture becoming the API.
use std::sync::Arc;

use axum::{body::Body, http::Request};
use paddock_manager::{registry::Registry, routes::AppState};
use tower::ServiceExt;

#[test]
fn native_client_read_routes() {
    // Runner discovery is outside AppState's in-memory store. Scope the Unix
    // admin namespace before starting the async runtime, in a child process:
    // mutating the test harness's environment would race its other threads.
    #[cfg(unix)]
    if std::env::var_os("PADDOCK_MACOS_CONTRACT_CHILD").is_none() {
        let isolated = tempfile::Builder::new()
            .prefix("paddock-contract-")
            .tempdir_in("/tmp")
            .expect("isolated admin namespace");
        let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
            .args(["--exact", "native_client_read_routes", "--nocapture"])
            .env("PADDOCK_MACOS_CONTRACT_CHILD", "1")
            .env("XDG_RUNTIME_DIR", isolated.path())
            .env("PADDOCK_DATA", isolated.path().join("data"))
            .output()
            .expect("isolated contract process");
        assert!(
            output.status.success(),
            "isolated contract failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("contract runtime")
        .block_on(assert_native_client_read_routes());
}

async fn assert_native_client_read_routes() {
    let temp = tempfile::tempdir().expect("temporary test directory");
    let mut state = AppState::for_tests();
    state.registry = Arc::new(Registry::new(temp.path().join("models")).with_backend("metal"));
    state.readiness = Arc::new(paddock_manager::readiness::probe_for_backend("metal"));
    state.auth_key = Some("macos-contract-test".into());
    state
        .db
        .insert_activity(
            "native-contract",
            11540,
            1,
            &[serde_json::json!({
                "seq":1,"ts_ms":1000,"status":200,"gen_ai.request.model":"fixture",
                "paddock.ttft_ms":12.5,"gen_ai.usage.output_tokens":128
            })],
        )
        .expect("activity fixture");
    let router = paddock_manager::routes::router(Arc::new(state));
    let export = std::env::var_os("PADDOCK_MACOS_CONTRACT_DIR").map(std::path::PathBuf::from);
    if let Some(dir) = &export {
        assert!(
            dir.is_absolute(),
            "fixture output must be an explicit absolute path"
        );
        std::fs::create_dir_all(dir).expect("fixture directory");
    }

    for (path, file) in [
        ("/api/server", "server.json"),
        ("/api/readiness", "readiness.json"),
        ("/api/models/catalog", "catalog.json"),
        ("/api/runners", "runners.json"),
        ("/api/usage/history?from=0&to=10000", "usage.json"),
        ("/api/activity?limit=200", "activity.json"),
        ("/api/cache", "cache.json"),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::get(path)
                    .header("authorization", "Bearer macos-contract-test")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("router response");
        assert_eq!(response.status(), 200, "{path}");
        let bytes = axum::body::to_bytes(response.into_body(), 8 * 1024 * 1024)
            .await
            .expect("bounded body");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("JSON body");
        match path {
            "/api/server" => assert_eq!(json["role"], "manager"),
            "/api/readiness" => assert_eq!(json["backend"], "metal"),
            "/api/models/catalog" => {
                assert_eq!(json["schema"], 3);
                assert!(
                    !json["models"]
                        .as_array()
                        .expect("catalog models")
                        .is_empty()
                );
                let models = json["models"].as_array().expect("catalog models");
                let diarization = models
                    .iter()
                    .find(|m| m["id"] == "nemotron-3-diarization")
                    .expect("Diarization reaches both Studio catalogs");
                assert_eq!(
                    diarization["capability"],
                    serde_json::json!(["diarization"])
                );
                for a in diarization["artifacts"].as_array().expect("artifacts") {
                    assert_eq!(a["backend_supported"], true);
                    assert_eq!(a["runtime"]["default_spec"], "off");
                    assert_eq!(a["runtime"]["memory"]["workspace_bytes"], 36133888);
                    let files = a["files"].as_array().expect("downloadable R2 bundle");
                    assert!(files.iter().all(|f| f["url"].as_str().expect("url")
                        == format!(
                            "https://models.truespar.io/models/{}",
                            f["dest"].as_str().expect("dest")
                        )));
                    assert!(
                        files
                            .iter()
                            .any(|f| f["dest"].as_str().expect("dest").ends_with("/LICENSE"))
                    );
                }
                let diffusion = models
                    .iter()
                    .find(|m| m["id"] == "diffusiongemma-26b-a4b")
                    .expect("DiffusionGemma reaches the native and web catalogs");
                for id in ["q8", "q4", "mlx-4bit"] {
                    let artifact = diffusion["artifacts"]
                        .as_array()
                        .expect("DiffusionGemma artifacts")
                        .iter()
                        .find(|a| a["id"] == id)
                        .expect("Metal DiffusionGemma artifact");
                    assert_eq!(artifact["backend_supported"], true);
                    assert_eq!(artifact["runtime"]["default_max_batch"], 1);
                    assert_eq!(artifact["runtime"]["default_spec"], "off");
                    // the MLX checkpoint carries its own tower; the GGUF
                    // weights take the vision companion (backend_tests agrees)
                    assert_eq!(artifact["runtime"]["embedded_vision"], id == "mlx-4bit");
                    let files = artifact["files"]
                        .as_array()
                        .expect("DiffusionGemma download manifest");
                    assert_eq!(files.len(), if id == "mlx-4bit" { 12 } else { 1 });
                    assert!(files.iter().all(|f| {
                        f["url"].as_str().is_some_and(|url| {
                            url.starts_with("https://models.truespar.io/models/")
                        })
                    }));
                }
                let qwen = models
                    .iter()
                    .find(|m| m["id"] == "qwen3.8-27b")
                    .expect("Qwen must reach the macOS catalog");
                assert_eq!(qwen["specs"]["published_at"], "2026-08-14");
                assert_ne!(qwen["specs"]["published_at"], qwen["revision"]);
                let splash = qwen["artifacts"]
                    .as_array()
                    .expect("Qwen artifacts are an array")
                    .iter()
                    .find(|a| a["id"] == "splash-4bit")
                    .expect("Splash must reach the macOS catalog");
                assert_eq!(splash["backend_supported"], true);
                assert_eq!(splash["format"], "splash-packed-q4");
                assert_eq!(splash["runtime"]["embedded_vision"], true);
                assert_eq!(splash["runtime"]["default_spec"], "adaptive");
                assert_eq!(splash["runtime"]["default_max_batch"], 1);
                assert_eq!(splash["runtime"]["default_max_ctx"], 32768);
                let files = splash["files"].as_array().expect("Splash file manifest");
                assert_eq!(files.len(), 81);
                assert!(files.iter().all(|f| {
                    f["url"]
                        .as_str()
                        .expect("catalog file URL")
                        .starts_with("https://models.truespar.io/models/Qwen3.8-27B-Splash/")
                }));
                let bonsai = models
                    .iter()
                    .find(|m| m["id"] == "bonsai-2-27b")
                    .expect("Bonsai must reach the macOS catalog");
                assert_eq!(bonsai["vendor"], "Prism ML");
                assert_eq!(bonsai["specs"]["published_at"], "2026-09-17");
                assert_eq!(
                    bonsai["specs"]["published_source"],
                    "https://prismml.com/news/bonsai-2-27b"
                );
                assert_ne!(bonsai["specs"]["published_at"], bonsai["revision"]);
                let ternary = bonsai["artifacts"]
                    .as_array()
                    .expect("Bonsai artifacts")
                    .iter()
                    .find(|a| a["id"] == "mlx-2bit")
                    .expect("Native ternary MLX export");
                assert_eq!(ternary["backend_supported"], true);
                assert_eq!(ternary["runtime"]["embedded_vision"], true);
                assert_eq!(ternary["runtime"]["kv_cache_dtype"], "f32");
                assert_eq!(ternary["runtime"]["default_max_batch"], 1);
                assert_eq!(ternary["runtime"]["default_max_ctx"], 32768);
                assert_eq!(ternary["runtime"]["default_spec"], "off");
                assert_eq!(
                    ternary["source"]["revision"],
                    "3f926b415992eaa2ae9dd7b573706494d6bbf787"
                );
                let files = ternary["files"].as_array().expect("Bonsai file manifest");
                assert_eq!(files.len(), 11);
                assert!(files.iter().all(|f| {
                    f["url"].as_str().expect("catalog file URL").starts_with(
                        "https://models.truespar.io/models/Ternary-Bonsai-2-27B-mlx-2bit/",
                    ) && f["sha256"].as_str().is_some_and(|sha| sha.len() == 64)
                }));
                for model in models {
                    if let Some(date) = model["specs"]["published_at"].as_str() {
                        assert_eq!(date.len(), 10, "{} publication day", model["id"]);
                        assert!(
                            model["specs"]["published_source"]
                                .as_str()
                                .is_some_and(|s| s.starts_with("https://"))
                        );
                    }
                }
            }
            "/api/runners" => assert_eq!(json, serde_json::json!([])),
            "/api/usage/history?from=0&to=10000" => assert!(json["buckets"].is_array()),
            "/api/activity?limit=200" => {
                assert_eq!(json["events"][0]["gen_ai.request.model"], "fixture")
            }
            "/api/cache" => assert_eq!(json["servers"], serde_json::json!([])),
            _ => unreachable!(),
        }
        if let Some(dir) = &export {
            std::fs::write(dir.join(file), &bytes).expect("write derived contract fixture");
        }
    }
}

#[tokio::test]
async fn native_reads_use_the_same_revision_checked_store_as_web() {
    let mut state = AppState::for_tests();
    state.auth_key = Some("read-contract".into());
    let router = paddock_manager::routes::router(Arc::new(state));
    let call = |method: &'static str, path: String, body: serde_json::Value| {
        let router = router.clone();
        async move {
            let response = router
                .oneshot(
                    Request::builder()
                        .method(method)
                        .uri(path)
                        .header("authorization", "Bearer read-contract")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status().as_u16();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
                .await
                .unwrap();
            (
                status,
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_default(),
            )
        }
    };
    let body = serde_json::json!({"id":"native-reads-contract","name":"Triage","revision":"",
        "body":r#"{"questions":{"q1":{"type":"noul","instructions":"Is it urgent?"}},"samples":"auto"}"#});
    let (status, saved) = call("POST", "/api/reads".into(), body.clone()).await;
    assert_eq!(status, 200);
    assert_eq!(saved["set"]["name"], "Triage");
    let (status, conflict) = call("POST", "/api/reads".into(), body).await;
    assert_eq!(status, 409);
    assert!(
        conflict["error"]["message"]
            .as_str()
            .unwrap()
            .contains("draft is kept")
    );
    let (_, listed) = call("GET", "/api/reads".into(), serde_json::Value::Null).await;
    assert_eq!(listed.as_array().unwrap(), &[saved["set"].clone()]);
    let (status, _) = call(
        "DELETE",
        format!(
            "/api/reads/native-reads-contract?revision={}",
            saved["set"]["revision"].as_str().unwrap()
        ),
        serde_json::Value::Null,
    )
    .await;
    assert_eq!(status, 204);
}
