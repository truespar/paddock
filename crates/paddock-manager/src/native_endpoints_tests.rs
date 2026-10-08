use super::*;

#[test]
fn editing_a_memory_limit_explicitly_clears_automatic_provenance() {
    let raw = crate::automatic_budget::pin(RAW, 2048);
    assert!(crate::automatic_budget::is_automatic(&raw));
    let unrelated = patch(&raw, 13493, &[Change::MaxCtx(Some(8192))], false).unwrap();
    assert!(crate::automatic_budget::is_automatic(&unrelated));
    // Even the same number becomes a hard limit when deliberately selected.
    let fixed = patch(&raw, 13493, &[Change::VramBudget(Some(2048))], false).unwrap();
    assert!(!crate::automatic_budget::is_automatic(&fixed));
    assert_eq!(crate::automatic_budget::pin(&fixed, 4096), fixed);
    let automatic = patch(&fixed, 13493, &[Change::VramBudget(None)], false).unwrap();
    assert!(
        toml::from_str::<toml::Value>(&automatic)
            .unwrap()
            .get("vram_budget")
            .is_none()
    );
}

#[test]
fn residency_is_strict_and_unsupported_models_cannot_claim_lazy_start() {
    assert!(!on_demand_config(RAW));
    for policy in [
        "load='on_demand'",
        "load='invalid'",
        "load_timeout_seconds=0",
        "unload_after_idel_seconds=60",
        "unload_after_idle_seconds=-1",
    ] {
        let raw = format!("{RAW}\n[residency]\n{policy}\n");
        assert!(validate_residency(&toml::from_str(&raw).unwrap()).is_err());
        assert!(!on_demand_config(&raw));
    }
    let raw = format!("{RAW}\n[residency]\nload='at_startup'\nload_timeout_seconds=120\n");
    assert!(
        !validate_residency(&toml::from_str(&raw).unwrap())
            .unwrap()
            .enabled()
    );
    let edited = patch(
        &raw,
        13493,
        &[Change::Residency(
            paddock_admin::residency::Config::default(),
        )],
        false,
    )
    .unwrap();
    assert!(edited.contains("retained comment"));
    assert!(edited.contains("synthetic-runner-secret"));
    assert!(
        patch(
            RAW,
            13493,
            &[Change::Residency(paddock_admin::residency::Config {
                load: paddock_admin::residency::LoadPolicy::OnDemand,
                ..Default::default()
            })],
            false
        )
        .is_err()
    );
}

#[test]
fn log_output_withholds_saved_live_and_labeled_credentials() {
    let (_dir, state) = isolated_state();
    let path = fixture(&state);
    std::fs::write(&path, "port=13493\napi_key='saved-private-key'\n[mcp.headers]\nAuthorization='Basic secret-header'\n").unwrap();
    let text = "INFO ready\nliteral saved-private-key\nliteral old-live-key\nAPI_KEY=unknown-key\nAuthorization: unknown\nBasic secret-header\n";
    let safe = safe_log_text(&state.supervisor, 13493, text, Some("old-live-key"));
    assert!(safe.starts_with("INFO ready\n"));
    assert_eq!(safe.matches("withheld").count(), 5);
    for secret in [
        "saved-private-key",
        "old-live-key",
        "unknown-key",
        "secret-header",
    ] {
        assert!(!safe.contains(secret));
    }
    std::fs::write(&path, "invalid [config").unwrap();
    assert!(safe_log_text(&state.supervisor, 13493, "sensitive\n", None).contains("withheld"));
}

fn isolated_state() -> (tempfile::TempDir, Arc<AppState>) {
    let dir = tempfile::tempdir().unwrap();
    let mut state = AppState::for_tests();
    state.supervisor = Arc::new(Supervisor::new(
        crate::supervisor::SpawnDefaults {
            runner_bin: None,
            device: "metal".into(),
            kernel_pack: None,
            models_dirs: vec![],
            logs_dir: dir.path().join("logs"),
            runners_dir: dir.path().join("runners"),
            work_dir: dir.path().into(),
            base_port: 41540,
            health_timeout: std::time::Duration::from_millis(100),
        },
        state.registry.clone(),
        None,
        None,
    ));
    (dir, Arc::new(state))
}

const RAW: &str = "# retained comment\nport = 13493\nhost = '127.0.0.1'\nmodel = 'fixture.gguf'\ndevice = 'metal'\nmax_ctx = 4096\napi_key = 'synthetic-runner-secret'\ncustom_flag = true\n[forensics]\nenabled = false\nauto = 'images'\n[[mcp_servers]]\nserver_label = 'fixture'\nserver_url = 'https://example.invalid/mcp'\n[mcp_servers.headers]\nAuthorization = 'synthetic-mcp-secret'\n";

fn creation_state() -> (tempfile::TempDir, Arc<AppState>) {
    creation_state_with_tower(false, false)
}

fn creation_state_with_tower(embedded: bool, audio: bool) -> (tempfile::TempDir, Arc<AppState>) {
    creation_state_with_media(embedded, audio, false)
}

fn creation_state_with_media(
    embedded: bool,
    audio: bool,
    split: bool,
) -> (tempfile::TempDir, Arc<AppState>) {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    std::fs::create_dir(&models).unwrap();
    std::fs::write(models.join("fixture.gguf"), b"fixture").unwrap();
    let mut catalog = json!({"schema":3,"model":[{
        "id":"fixture", "display":"Fixture", "capability":["chat","tools"],
        "artifact":[{"id":"q4","kind":"weights","format":"gguf","label":"Fixture Q4", "default":true,
            "runtime":{"backends":["metal"],"default_max_batch":32,"kv_cache_dtype":"f16"},
            "file":[{"dest":"fixture.gguf","size":7,"sha256":"","url":"https://example.invalid/never-download"}]}]
    }]});
    if embedded {
        catalog["model"][0]["capability"] = json!(["chat", "vision"]);
        catalog["model"][0]["artifact"][0]["runtime"]["embedded_vision"] = json!(true);
    }
    if audio {
        catalog["model"][0]["capability"] = json!(["transcription"]);
        std::fs::write(models.join("audio.gguf"), b"fixture").unwrap();
        catalog["model"][0]["artifact"].as_array_mut().unwrap().push(json!({
            "id":"audio","kind":"audio","format":"gguf","label":"Speech encoder", "default":true,"required":true,
            "runtime":{"backends":["metal"]},
            "file":[{"dest":"audio.gguf","size":7,"sha256":"","url":"https://example.invalid/no-download"}]
        }));
    }
    if split {
        catalog["model"][0]["capability"] = json!(["embeddings"]);
        catalog["model"][0]["artifact"][1]["required"] = json!(false);
        catalog["model"][0]["artifact"][1]["default"] = json!(false);
        std::fs::write(models.join("vision.gguf"), b"fixture").unwrap();
        catalog["model"][0]["artifact"].as_array_mut().unwrap().push(json!({
            "id":"vision","kind":"vision","format":"gguf","label":"Pictures", "default":true,
            "runtime":{"backends":["metal"]},
            "file":[{"dest":"vision.gguf","size":7,"sha256":"","url":"https://example.invalid/no-download"}]
        }));
    }
    let catalog = serde_json::from_value(catalog).unwrap();
    let registry = Arc::new(
        crate::registry::Registry::from_catalog(catalog, models.clone()).with_backend("metal"),
    );
    let mut state = AppState::for_tests();
    state.registry = registry.clone();
    state.supervisor = Arc::new(Supervisor::new(
        crate::supervisor::SpawnDefaults {
            runner_bin: None,
            device: "metal".into(),
            kernel_pack: None,
            models_dirs: vec![models],
            logs_dir: dir.path().join("logs"),
            runners_dir: dir.path().join("runners"),
            work_dir: dir.path().into(),
            base_port: 41540,
            health_timeout: std::time::Duration::from_millis(10),
        },
        registry,
        None,
        None,
    ));
    (dir, Arc::new(state))
}

#[tokio::test]
async fn native_embedding_split_towers_follow_independent_switches_and_survive_edit() {
    let (_dir, state) = creation_state_with_media(false, true, true);
    for vision in [false, true] {
        for audio in [false, true] {
            let choice: Composition = serde_json::from_value(json!({
                "model":"fixture", "artifact":"q4", "vision":vision,"audio":audio
            }))
            .unwrap();
            let port = create_config(
                &state,
                "fixture",
                "q4",
                None,
                &[Change::Composition(choice)],
                false,
                None,
            )
            .await
            .unwrap();
            let (raw, _) = state.supervisor.read_config_file(port).unwrap();
            let doc = parse(&raw, port).unwrap();
            assert_eq!(
                doc.get("mmproj").is_some(),
                vision,
                "{vision}/{audio}: {doc:?}"
            );
            assert_eq!(
                doc.get("audio_mmproj").is_some(),
                audio,
                "{vision}/{audio}: {doc:?}"
            );
            let off: Composition = serde_json::from_value(json!({
                "model":"fixture", "artifact":"q4", "vision":false,"audio":false
            }))
            .unwrap();
            let edited = composition::resolve(&state, port, &raw, &off)
                .await
                .unwrap();
            let doc = parse(&edited, port).unwrap();
            assert!(doc.get("mmproj").is_none() && doc.get("audio_mmproj").is_none());
        }
    }
}

#[tokio::test]
async fn creation_preserves_embedded_vision_and_required_audio_without_an_optional_toggle() {
    for audio in [false, true] {
        let (_dir, state) = creation_state_with_tower(!audio, audio);
        let changes: Vec<Change> = serde_json::from_value(json!([
            {"field":"composition","value":{"model":"fixture","artifact":"q4","vision":false,"drafter":null}}
        ])).unwrap();
        let port = create_config(&state, "fixture", "q4", None, &changes, false, None)
            .await
            .unwrap();
        let (raw, _) = state.supervisor.read_config_file(port).unwrap();
        if audio {
            assert!(
                parse(&raw, port).unwrap()["mmproj"]
                    .as_str()
                    .unwrap()
                    .ends_with("audio.gguf")
            );
        }
    }
}

#[tokio::test]
async fn creation_preparation_is_read_only_and_uses_the_edit_schema() {
    let (_dir, state) = creation_state();
    let draft = prepare(&state, "fixture", "q4").await.unwrap();
    assert_eq!(draft["port"], 0);
    assert!(draft["revision"].is_null());
    assert_eq!(draft["settings"]["max_batch"], 1);
    assert_eq!(draft["settings"]["max_ctx"], 32768);
    assert!(
        draft["settings"]["runtime_options"]
            .as_array()
            .unwrap()
            .len()
            > 10
    );
    assert!(!state.supervisor.servers_dir().exists());
    assert!(!state.supervisor.has_records_for_test().await);
    assert!(!draft.to_string().contains("never-download"));
}

#[tokio::test]
async fn new_context_default_reaches_saved_config_without_rewriting_explicit_choices() {
    let (_dir, state) = creation_state();
    for context in [None, Some(8192)] {
        let changes = context
            .map(|v| vec![Change::MaxCtx(Some(v))])
            .unwrap_or_default();
        let port = create_config(&state, "fixture", "q4", None, &changes, false, None)
            .await
            .unwrap();
        let saved = projection(&state.supervisor, port).unwrap();
        assert_eq!(saved["settings"]["max_ctx"], context.unwrap_or(32768));
        assert_eq!(saved["settings"]["max_batch"], 1);
    }
    let (_dir, speech) = creation_state_with_tower(false, true);
    let draft = prepare(&speech, "fixture", "q4").await.unwrap();
    assert_eq!(draft["settings"]["max_ctx"], 4096);
}

#[tokio::test]
async fn creation_publishes_complete_settings_once_and_edit_reads_identical_values() {
    let (_dir, state) = creation_state();
    let changes: Vec<Change> = serde_json::from_value(json!([
        {"field":"max_ctx","value":8192}, {"field":"max_batch","value":1},
        {"field":"spec","value":"off"}, {"field":"host","value":"127.0.0.1"},
        {"field":"vram_budget","value":24576}, {"field":"kv_cache_dtype","value":"f16"},
        {"field":"composition","value":{"model":"fixture","artifact":"q4","vision":false,"drafter":null}},
        {"field":"runtime","value":{"max_tokens":3072,"temp":0.7}},
        {"field":"kv_offload","value":{"enabled":false,"ram_gb":0,"nvme_gb":0}}
    ])).unwrap();
    let tools = serde_json::from_value(
        json!({"provider":"exa","key":"synthetic-search-key","connectors":[]}),
    )
    .unwrap();
    let port = create_config(&state, "fixture", "q4", None, &changes, false, Some(tools))
        .await
        .unwrap();
    assert!(port >= 1024);
    assert!(
        !state.supervisor.has_records_for_test().await,
        "Publication must not spawn before all settings are ready"
    );
    let (raw, revision) = state.supervisor.read_config_file(port).unwrap();
    let doc = parse(&raw, port).unwrap();
    assert_eq!(doc["max_tokens"].as_integer(), Some(3072));
    assert_eq!(doc["temp"].as_float(), Some(0.7));
    assert_eq!(doc["vram_budget"].as_integer(), Some(24576));
    assert_eq!(
        doc["web_search_api_key"].as_str(),
        Some("synthetic-search-key")
    );
    assert!(doc["api_key"].as_str().unwrap().starts_with("pd-"));
    let edit = projection(&state.supervisor, port).unwrap();
    assert_eq!(edit["settings"]["max_ctx"], 8192);
    assert_eq!(edit["settings"]["max_batch"], 1);
    assert_eq!(edit["settings"]["vram_budget"], 24576);
    assert!(!edit.to_string().contains("synthetic-search-key"));
    assert!(
        create_config(&state, "fixture", "q4", Some(port), &changes, false, None)
            .await
            .is_err()
    );
    assert_eq!(
        state.supervisor.config_file_hash(port).as_deref(),
        Some(revision.as_str())
    );
    let second = create_config(&state, "fixture", "q4", None, &[], false, None)
        .await
        .unwrap();
    assert_ne!(port, second);
    assert_eq!(
        projection(&state.supervisor, second).unwrap()["settings"]["max_batch"],
        1
    );
}

#[tokio::test]
async fn creation_rejects_invalid_advanced_network_and_stale_connectors_before_publication() {
    let (_dir, state) = creation_state();
    for value in [
        json!([{"field":"max_batch","value":0}]),
        json!([{"field":"max_batch","value":1},{"field":"max_batch","value":4}]),
        json!([{"field":"runtime","value":{"max_tokens":"wrong type"}}]),
        json!([{"field":"host","value":"0.0.0.0"}]),
        json!([{"field":"kv_offload","value":{"enabled":true,"ram_gb":1,"nvme_gb":0}}]),
    ] {
        let changes: Vec<Change> = serde_json::from_value(value).unwrap();
        assert!(
            create_config(&state, "fixture", "q4", None, &changes, false, None)
                .await
                .is_err()
        );
        assert!(!state.supervisor.servers_dir().exists());
    }
    let tools = serde_json::from_value(json!({"provider":"exa","key":"","connectors":[]})).unwrap();
    assert!(
        create_config(&state, "fixture", "q4", None, &[], false, Some(tools))
            .await
            .is_err()
    );
    let tools = serde_json::from_value(
        json!({"provider":"","key":"","connectors":[{"id":"missing","revision":1}]}),
    )
    .unwrap();
    assert!(
        create_config(&state, "fixture", "q4", None, &[], false, Some(tools))
            .await
            .is_err()
    );
    assert!(!state.supervisor.servers_dir().exists());
}

#[tokio::test]
async fn creation_attaches_reviewed_connectors_without_exporting_credentials_or_touching_other_files()
 {
    let (_dir, state) = creation_state();
    let row = state
        .db
        .create_connector(
            &json!({"label":"fixture-tools", "url":"https://example.invalid/mcp",
        "headers":{"Authorization":"Bearer synthetic-private-header"}}),
        )
        .unwrap();
    let id = row["id"].as_str().unwrap();
    let row = state.db.get_connector(id).unwrap().unwrap();
    let tools = serde_json::from_value(
        json!({"provider":"","key":"","connectors":[{"id":id,"revision":row["revision"]}]}),
    )
    .unwrap();
    let port = create_config(&state, "fixture", "q4", None, &[], false, Some(tools))
        .await
        .unwrap();
    let (raw, _) = state.supervisor.read_config_file(port).unwrap();
    let doc = parse(&raw, port).unwrap();
    assert_eq!(doc["mcp_servers"][0]["connector_id"].as_str(), Some(id));
    assert_eq!(
        doc["mcp_servers"][0]["headers"]["Authorization"].as_str(),
        Some("Bearer synthetic-private-header")
    );
    assert_eq!(
        state.db.get_connector(id).unwrap().unwrap()["ports"],
        json!([port])
    );
    assert!(
        !projection(&state.supervisor, port)
            .unwrap()
            .to_string()
            .contains("synthetic-private-header")
    );
    let stale = serde_json::from_value(
        json!({"provider":"","key":"","connectors":[{"id":id,"revision":row["revision"]}]}),
    )
    .unwrap();
    assert!(
        create_config(&state, "fixture", "q4", None, &[], false, Some(stale))
            .await
            .is_err()
    );
    assert_eq!(
        std::fs::read_dir(state.supervisor.servers_dir())
            .unwrap()
            .count(),
        1
    );
}

#[test]
fn offload_edits_preserve_private_paths_and_validate_budgets() {
    let raw = format!(
        "{RAW}\n[kv_offload]\nenabled=false\nram_gb=1.0\nnvme_path='/private/custom/cache'\n"
    );
    let result = patch(
        &raw,
        13493,
        &[Change::KvOffload(offload::Settings {
            enabled: true,
            ram_gb: 2.0,
            nvme_gb: 32.0,
        })],
        false,
    )
    .unwrap();
    let doc = parse(&result, 13493).unwrap();
    assert_eq!(
        doc["kv_offload"]["nvme_path"].as_str(),
        Some("/private/custom/cache")
    );
    assert_eq!(doc["api_key"].as_str(), Some("synthetic-runner-secret"));
    assert_eq!(doc["kv_offload"]["ram_gb"].as_float(), Some(2.0));
    assert!(!offload::projection(&doc).to_string().contains("private"));
    assert!(!crate::backend_contract::path_kv_offload(
        std::path::Path::new(doc["model"].as_str().unwrap())
    ));
    for ram in [f64::NAN, f64::INFINITY, -1., 0., 0.25, 1025.] {
        assert!(
            patch(
                &raw,
                13493,
                &[Change::KvOffload(offload::Settings {
                    enabled: true,
                    ram_gb: ram,
                    nvme_gb: 32.
                })],
                false
            )
            .is_err()
        );
    }
}

fn fixture(state: &AppState) -> std::path::PathBuf {
    let path = state.supervisor.server_config_path(13493);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, RAW).unwrap();
    path
}

#[test]
fn changed_fields_preserve_secrets_custom_keys_and_nested_configuration() {
    let text = patch(
        RAW,
        13493,
        &[Change::MaxBatch(Some(8)), Change::Forensics(true)],
        false,
    )
    .unwrap();
    let value: toml::Value = toml::from_str(&text).unwrap();
    assert!(text.contains("# retained comment"));
    assert_eq!(value["max_batch"].as_integer(), Some(8));
    assert_eq!(value["api_key"].as_str(), Some("synthetic-runner-secret"));
    assert_eq!(value["custom_flag"].as_bool(), Some(true));
    assert_eq!(value["forensics"]["auto"].as_str(), Some("images"));
    assert_eq!(
        value["mcp_servers"][0]["headers"]["Authorization"].as_str(),
        Some("synthetic-mcp-secret")
    );
    assert!(value["mcp_servers"][0].get("max_batch").is_none());
}

#[test]
fn null_restores_default_without_erasing_other_settings() {
    let text = patch(RAW, 13493, &[Change::MaxCtx(None)], false).unwrap();
    let value: toml::Value = toml::from_str(&text).unwrap();
    assert!(value.get("max_ctx").is_none());
    assert!(value["custom_flag"].as_bool().unwrap());
}

#[test]
fn simple_form_uses_web_speculation_values_and_metal_memory_guards() {
    for policy in ["on", "off", "adaptive", "auto", "ladder"] {
        let result = patch(RAW, 13493, &[Change::Spec(Some(policy.into()))], false).unwrap();
        assert_eq!(
            parse(&result, 13493).unwrap()["spec"].as_str(),
            Some(policy)
        );
    }
    assert!(
        patch(
            RAW,
            13493,
            &[Change::KvCacheDtype(Some("fp8_e4m3".into()))],
            false
        )
        .is_err()
    );
    assert!(patch(RAW, 13493, &[Change::VramBudget(Some(0))], false).is_err());
    let result = patch(
        RAW,
        13493,
        &[
            Change::KvCacheDtype(Some("auto".into())),
            Change::VramBudget(Some(8192)),
        ],
        false,
    )
    .unwrap();
    let doc = parse(&result, 13493).unwrap();
    assert_eq!(doc["vram_budget"].as_integer(), Some(8192));
    assert_eq!(doc["api_key"].as_str(), Some("synthetic-runner-secret"));
    let legacy = RAW.replacen("port = 13493", "no_spec = true\nport = 13493", 1);
    let enabled = patch(&legacy, 13493, &[Change::Spec(Some("on".into()))], false).unwrap();
    assert!(parse(&enabled, 13493).unwrap().get("no_spec").is_none());
}

#[tokio::test]
async fn native_bonsai_f32_edits_follow_the_catalog_without_changing_other_models() {
    let (_dir, state) = isolated_state();
    let changes = [Change::KvCacheDtype(Some("f32".into()))];
    let raw = RAW.to_owned() + "\n[catalog]\nmodel = 'bonsai-2-27b'\nartifact = 'mlx-2bit'\n";
    let patched = patch(&raw, 13493, &changes, false).unwrap();
    let resolved = resolve_changes(&state, 13493, patched, &changes)
        .await
        .unwrap();
    assert_eq!(
        parse(&resolved, 13493).unwrap()["kv_cache_dtype"].as_str(),
        Some("f32")
    );
    let unknown = patch(RAW, 13493, &changes, false).unwrap();
    assert!(
        resolve_changes(&state, 13493, unknown, &changes)
            .await
            .is_err()
    );
    let original = raw
        .replace("bonsai-2-27b", "qwen3.8-27b")
        .replace("mlx-2bit", "mlx-4bit");
    let patched = patch(&original, 13493, &changes, false).unwrap();
    assert!(
        resolve_changes(&state, 13493, patched, &changes)
            .await
            .is_err()
    );
}

#[test]
fn invalid_duplicate_and_arbitrary_fields_are_rejected() {
    for changes in [
        vec![],
        vec![Change::MaxCtx(Some(0))],
        vec![Change::MaxBatch(Some(257))],
        vec![Change::Spec(Some("999".into()))],
        vec![Change::MaxCtx(None), Change::MaxCtx(Some(4096))],
    ] {
        assert!(patch(RAW, 13493, &changes, false).is_err());
    }
    assert!(patch(RAW, 13494, &[Change::MaxCtx(Some(8192))], false).is_err());
    for value in [
        json!({"field":"model","value":"/tmp/foreign.gguf"}),
        json!({"field":"max_ctx","value":4096,"url":"evil"}),
    ] {
        assert!(serde_json::from_value::<Change>(value).is_err());
    }
}

#[test]
fn network_bind_requires_consent_and_a_key_and_never_weakens_auth() {
    let changes = [Change::Host("0.0.0.0".parse().unwrap())];
    assert!(patch(RAW, 13493, &changes, false).is_err());
    assert!(patch(RAW, 13493, &changes, true).is_ok());
    let keyless = RAW.replace("api_key = 'synthetic-runner-secret'\n", "");
    assert!(patch(&keyless, 13493, &changes, true).is_err());
    assert!(patch(RAW, 13493, &[Change::ApiKey("short".into())], false).is_err());
    assert!(
        patch(
            RAW,
            13493,
            &[Change::ApiKey("a-secret-with\na-newline".into())],
            false
        )
        .is_err()
    );
}

#[test]
fn projection_is_credential_free_and_revision_matches_values() {
    let (_dir, state) = isolated_state();
    fixture(&state);
    let value = projection(&state.supervisor, 13493).unwrap();
    assert!(!value.to_string().contains("synthetic"));
    assert!(!value.to_string().contains("Authorization"));
    assert_eq!(value["max_ctx"], 4096);
    assert_eq!(value["settings"]["has_api_key"], true);
    assert_eq!(
        value["revision"].as_str(),
        state.supervisor.config_file_hash(13493).as_deref()
    );
}

#[test]
fn advanced_schema_projects_only_reviewed_values_and_legacy_spec_disable() {
    let (_dir, state) = isolated_state();
    let path = fixture(&state);
    std::fs::write(
        &path,
        RAW.replacen(
            "port =",
            "no_spec = true\ntemp = 0.7\nseed = 123\nport =",
            1,
        ),
    )
    .unwrap();
    let p = projection(&state.supervisor, 13493).unwrap();
    assert_eq!(p["settings"]["no_spec"], true);
    let fields = p["settings"]["runtime_options"].as_array().unwrap();
    assert_eq!(fields.len(), 16);
    assert_eq!(
        fields.iter().find(|f| f["id"] == "temp").unwrap()["value"],
        0.7
    );
    assert_eq!(
        fields.iter().find(|f| f["id"] == "seed").unwrap()["value"],
        123
    );
    for private in [
        "synthetic",
        "Authorization",
        "api_key",
        "custom_flag",
        "model_dirs",
    ] {
        assert!(!serde_json::to_string(fields).unwrap().contains(private));
    }
}

#[test]
fn advanced_edits_are_typed_bounded_and_preserve_private_config() {
    let changes: serde_json::Map<String, Value> = serde_json::from_value(json!({
        "temp":0.4, "max_tokens":8192, "seed":123456789, "no_metrics":false,
        "pdf_max_pages":40, "served_model_name":"local-coding", "top_p":null,
    }))
    .unwrap();
    let original = RAW.replacen("port =", "top_p = 0.9\nport =", 1);
    let text = patch(&original, 13493, &[Change::Runtime(changes)], false).unwrap();
    let doc = parse(&text, 13493).unwrap();
    assert_eq!(doc["temp"].as_float(), Some(0.4));
    assert_eq!(doc["max_tokens"].as_integer(), Some(8192));
    assert!(doc.get("top_p").is_none());
    assert_eq!(doc["served_model_name"].as_str(), Some("local-coding"));
    assert_eq!(doc["api_key"].as_str(), Some("synthetic-runner-secret"));
    assert_eq!(
        doc["mcp_servers"][0]["headers"]["Authorization"].as_str(),
        Some("synthetic-mcp-secret")
    );
    for invalid in [
        json!({"temp":3}),
        json!({"top_p":-0.1}),
        json!({"seed":1.5}),
        json!({"no_metrics":"false"}),
        json!({"max_tokens":0}),
        json!({"model":"/tmp/foreign.gguf"}),
        json!({"api_key":null}),
        json!({"kv_offload":{"enabled":true}}),
        json!({"no_auth":true}),
        json!({"served_model_name":"bad\nname"}),
    ] {
        assert!(
            patch(
                RAW,
                13493,
                &[Change::Runtime(invalid.as_object().unwrap().clone())],
                false
            )
            .is_err()
        );
    }
}

#[tokio::test]
async fn deferred_save_is_durable_never_spawns_and_rejects_stale_edits() {
    let (_dir, state) = isolated_state();
    let path = fixture(&state);
    let revision = state.supervisor.config_file_hash(13493).unwrap();
    edit(
        state.clone(),
        13493,
        revision.clone(),
        None,
        vec![Change::MaxCtx(Some(8192))],
        Apply::Defer,
        false,
    )
    .await
    .unwrap();
    assert!(
        !state
            .supervisor
            .list()
            .await
            .iter()
            .any(|r| r.port == 13493)
    );
    assert_eq!(
        projection(&state.supervisor, 13493).unwrap()["max_ctx"],
        8192
    );
    let saved = std::fs::read_to_string(&path).unwrap();
    assert!(
        edit(
            state.clone(),
            13493,
            revision,
            None,
            vec![Change::MaxCtx(Some(16384))],
            Apply::Defer,
            false
        )
        .await
        .is_err()
    );
    assert_eq!(std::fs::read_to_string(path).unwrap(), saved);
}

#[tokio::test]
async fn restart_without_the_reviewed_process_never_writes_or_starts() {
    let (_dir, state) = isolated_state();
    let path = fixture(&state);
    let revision = state.supervisor.config_file_hash(13493).unwrap();
    assert!(
        edit(
            state.clone(),
            13493,
            revision,
            Some(42),
            vec![Change::MaxCtx(Some(8192))],
            Apply::Restart,
            false
        )
        .await
        .is_err()
    );
    assert_eq!(std::fs::read_to_string(path).unwrap(), RAW);
    assert!(
        !state
            .supervisor
            .list()
            .await
            .iter()
            .any(|r| r.port == 13493)
    );
}

#[tokio::test]
async fn shared_writer_rechecks_reviewed_process_before_publishing() {
    let (_dir, state) = isolated_state();
    let path = fixture(&state);
    let revision = state.supervisor.config_file_hash(13493).unwrap();
    let content = patch(RAW, 13493, &[Change::MaxCtx(Some(8192))], false).unwrap();
    let result = state
        .supervisor
        .write_config_file(13493, &content, Some(&revision), 1, Some(u32::MAX))
        .await;
    assert!(result.unwrap_err().contains("reviewed runner"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), RAW);
}

#[tokio::test]
async fn remove_requires_current_revision_and_only_removes_its_config() {
    let (_dir, state) = isolated_state();
    let path = fixture(&state);
    let other = state.supervisor.server_config_path(13494);
    std::fs::write(&other, "untouched").unwrap();
    assert!(remove(state.clone(), 13493, "old").await.is_err());
    assert!(path.exists());
    let revision = state.supervisor.config_file_hash(13493).unwrap();
    remove(state, 13493, &revision).await.unwrap();
    assert!(!path.exists());
    assert_eq!(std::fs::read_to_string(other).unwrap(), "untouched");
}
