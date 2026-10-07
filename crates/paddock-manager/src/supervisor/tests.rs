use super::*;

/// Exercise the installed-model resolver, not preview's missing-download
/// fallback. These tiny files satisfy catalog presence checks only; no
/// weights are downloaded or loaded by these control-plane tests.
fn installed_model_supervisor(
    dir: &Path,
    model_id: &str,
    default_spec: Option<&str>,
) -> Supervisor {
    installed_model_supervisor_on(dir, model_id, default_spec, "metal")
}

/// The same on a chosen backend - a CUDA-only row has no Metal artifacts.
fn installed_model_supervisor_on(
    dir: &Path,
    model_id: &str,
    default_spec: Option<&str>,
    backend: &str,
) -> Supervisor {
    let models = dir.join("models");
    let source = crate::registry::Registry::new(models.clone()).with_backend(backend);
    let mut model = source.catalog_of(model_id).unwrap().clone();
    for artifact in &mut model.artifacts {
        if let Some(policy) = default_spec {
            artifact.runtime.default_spec = Some(policy.into());
        }
        for file in &mut artifact.files {
            file.size = 1;
            let path = models.join(&file.dest);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
    }
    let registry = crate::registry::Registry::from_catalog(
        crate::registry::Catalog {
            schema: 3,
            models: vec![model],
        },
        models.clone(),
    )
    .with_backend(backend);
    Supervisor::new(
        SpawnDefaults {
            runner_bin: None,
            runners_dir: dir.join("runners"),
            device: backend.into(),
            kernel_pack: None,
            models_dirs: vec![models],
            logs_dir: dir.join("logs"),
            work_dir: dir.into(),
            base_port: 18100,
            health_timeout: Duration::from_secs(1),
        },
        Arc::new(registry),
        None,
        None,
    )
}

#[tokio::test]
async fn invalid_metal_edits_never_replace_saved_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let sup = installed_model_supervisor(dir.path(), "gemma-4-31b", None);
    let valid = sup
        .preview_config(SpawnSpec {
            model: "gemma-4-31b".into(),
            artifact: Some("mlx-4bit".into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let port = 18100;
    sup.write_config_file_deferred(port, &valid, None).unwrap();
    let bad = format!("{valid}\n[kv_offload]\nenabled = true\nram_gb = 8\nnvme_gb = 0\n");
    assert!(sup.write_config_file_deferred(port, &bad, None).is_err());
    assert!(
        sup.write_config_file(port, &bad, None, 100, None)
            .await
            .is_err()
    );
    assert_eq!(sup.read_config_file(port).unwrap().0, valid);
    assert!(!sup.is_serving(port).await);
}

/// Vision OFF has to reach the file as `vision = false`: the runner loads
/// a tower it finds beside the weights whenever the file names none, so
/// leaving the mmproj line out served images with the switch off. The
/// projection and the file-derived spec must both read the off back, and
/// a re-render from that spec must not resolve the tower again.
#[tokio::test]
async fn vision_off_is_written_read_back_and_survives_a_re_render() {
    let dir = tempfile::tempdir().unwrap();
    let sup = installed_model_supervisor(dir.path(), "qwen3.5-9b", None);
    let spec = |vision| SpawnSpec {
        model: "qwen3.5-9b".into(),
        artifact: Some("q8".into()),
        vision,
        ..Default::default()
    };
    let on = sup.preview_config(spec(None)).await.unwrap();
    let v: toml::Value = toml::from_str(&on).unwrap();
    assert!(v.get("mmproj").is_some(), "{on}");
    assert!(v.get("vision").is_none(), "{on}");

    let off = sup.preview_config(spec(Some(false))).await.unwrap();
    let v: toml::Value = toml::from_str(&off).unwrap();
    assert!(v.get("mmproj").is_none(), "{off}");
    assert_eq!(v["vision"].as_bool(), Some(false), "{off}");

    assert!(sup.project_config_text(&on).unwrap().vision);
    assert!(!sup.project_config_text(&off).unwrap().vision);
    let from_file = sup.spec_from_config_text(&off).unwrap();
    assert_eq!(from_file.vision, Some(false));
    assert_eq!(sup.spec_from_config_text(&on).unwrap().vision, None);
    let again = sup.preview_config(from_file).await.unwrap();
    let v: toml::Value = toml::from_str(&again).unwrap();
    assert!(v.get("mmproj").is_none(), "{again}");
    assert_eq!(v["vision"].as_bool(), Some(false), "{again}");
}

#[tokio::test]
async fn installed_non_speculative_models_honor_catalog_off() {
    for (model, artifact) in [
        ("bonsai-2-27b", "mlx-2bit"),
        ("gemma-4-31b", "mlx-4bit"),
        ("muse-glimmer-30b", "mlx-4bit"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let sup = installed_model_supervisor(dir.path(), model, None);
        let resolved = sup
            .resolve_model(model, Some(artifact), false, None, None)
            .await
            .unwrap();
        assert!(resolved.mtp.is_none(), "{model}");
        assert!(resolved.drafter.is_none(), "{model}");
        assert!(resolved.spec_desc.is_none(), "{model}");
        let request = SpawnSpec {
            model: model.into(),
            artifact: Some(artifact.into()),
            ..Default::default()
        };
        let preview = sup.preview_config(request.clone()).await.unwrap();
        let config: toml::Value = toml::from_str(&preview).unwrap();
        assert_eq!(config["spec"].as_str(), Some("off"), "{model}");
        assert!(config.get("mtp").is_none(), "{model}");
        sup.render_spec_config(18100, request).await.unwrap();
        // Deliberately enabling an unsupported mechanism must still fail.
        for policy in ["adaptive", "4"] {
            assert!(matches!(
                sup.resolve_model(model, Some(artifact), false, Some(policy), None)
                    .await,
                Err(SpawnError::Unsupported(_))
            ));
        }
    }
}

#[tokio::test]
async fn installed_speculative_model_default_off_does_not_load_a_drafter() {
    for policy in ["off", " OFF ", "false", "no", "none", "0"] {
        let dir = tempfile::tempdir().unwrap();
        let sup = installed_model_supervisor(dir.path(), "qwen3.8-27b", Some(policy));
        let resolve = |want| sup.resolve_model("qwen3.8-27b", Some("mlx-4bit"), false, want, None);
        let inherited = resolve(None).await.unwrap();
        assert!(inherited.mtp.is_none(), "{policy}");
        assert!(inherited.drafter.is_none(), "{policy}");
        assert_eq!(inherited.spec_desc.as_deref(), Some("off"), "{policy}");

        let enabled = resolve(Some("adaptive")).await.unwrap();
        assert!(enabled.mtp.is_some(), "explicit policy overrides {policy}");
        assert!(enabled.drafter.is_some());
        assert!(enabled.spec_desc.unwrap().contains("adaptive"));

        let disabled = resolve(Some("off")).await.unwrap();
        assert!(disabled.mtp.is_none());
        assert_eq!(disabled.spec_desc.as_deref(), Some("off"));
    }
}

#[tokio::test]
async fn automatic_port_skips_saved_loading_and_unrelated_listeners() {
    let dir = tempfile::tempdir().unwrap();
    let socket = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let base = socket.local_addr().unwrap().port();
    let sup = Supervisor::new(
        SpawnDefaults {
            runner_bin: None,
            runners_dir: dir.path().join("runners"),
            device: "metal".into(),
            kernel_pack: None,
            models_dirs: vec![],
            logs_dir: dir.path().join("logs"),
            work_dir: dir.path().into(),
            base_port: base,
            health_timeout: Duration::from_secs(1),
        },
        Arc::new(crate::registry::Registry::new(dir.path().join("models"))),
        None,
        None,
    );
    let first = sup.allocate_port().await.unwrap();
    assert_ne!(first, base, "never take over another service's listener");
    let config = sup.server_config_path(first);
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    std::fs::write(&config, "# saved endpoint fixture\n").unwrap();
    let second = sup.allocate_port().await.unwrap();
    assert_ne!(second, first, "a stopped model's address stays reserved");
    sup.spawning.lock().unwrap().insert(second);
    let third = sup.allocate_port().await.unwrap();
    assert!(![base, first, second].contains(&third));
    assert_eq!(
        std::fs::read_to_string(config).unwrap(),
        "# saved endpoint fixture\n"
    );
    assert!(socket.local_addr().is_ok(), "unrelated listener stays open");
}

/// A refusal that does not name what it is refusing over sent a user
/// hunting for an hour. The blocker has to survive into the message.
#[test]
fn a_taken_port_says_what_is_holding_it() {
    let sock = "/run/user/1000/paddock/runner-11540.sock";
    let msg = SpawnError::PortTaken(11540, format!("something is listening on {sock}")).to_string();
    assert!(msg.contains("11540"), "{msg}");
    assert!(
        msg.contains(sock),
        "the operator cannot act on what is not named: {msg}"
    );

    let msg = SpawnError::PortTaken(11540, "this manager has a runner on it, pid 4242".into())
        .to_string();
    assert!(msg.contains("pid 4242"), "{msg}");
}

fn bare_record(pid: u32) -> Record {
    Record {
        origin: Origin::Adopted,
        child: None,
        model: None,
        spec_desc: None,
        pid,
        pinned: false,
        spec: None,
        api_key: None,
    }
}

/// `list()` and `configured()` answer "is anything on this port" for two
/// different surfaces, and when they disagreed the endpoint rendered on
/// neither. One rule, both callers.
#[test]
fn a_port_counts_as_occupied_from_either_side_alone() {
    let mut recs = HashMap::new();
    assert!(!port_has_endpoint(&recs, &[], 11540), "nothing anywhere");

    // Socket only: a runner we did not start, or one that outlived our
    // record. This is the case that used to satisfy `configured().running`
    // while producing no runner row at all.
    assert!(port_has_endpoint(&recs, &[11540], 11540));

    // Record only: ours, still booting, socket not up yet.
    recs.insert(11540, bare_record(4242));
    assert!(port_has_endpoint(&recs, &[], 11540));
    assert!(port_has_endpoint(&recs, &[11540], 11540));

    // Neighbouring ports are not implicated by either signal.
    assert!(!port_has_endpoint(&recs, &[11540], 11541));
}

/// The zombie-record rule, which had no rule before: quiet is not gone.
#[test]
fn a_silent_record_is_only_forgotten_when_it_is_provably_gone() {
    // Gone: nothing enumerates and the process we launched has exited, or
    // there was never a process of ours to begin with (adopted/attached).
    assert!(silent_record_is_gone(false, ChildState::Exited));
    assert!(silent_record_is_gone(false, ChildState::NoHandle));

    // Booting: our child is alive, its socket is not up yet. Dropping this
    // record strands the process handle and we could never stop it again.
    assert!(!silent_record_is_gone(false, ChildState::Alive));

    // Hung: the socket is still there but identify does not answer. That is
    // a real state an operator has to be able to see, so it keeps its row.
    assert!(!silent_record_is_gone(true, ChildState::Alive));
    assert!(!silent_record_is_gone(true, ChildState::Exited));
    assert!(!silent_record_is_gone(true, ChildState::NoHandle));
}

#[test]
fn version_dirs_order_numerically_not_lexically() {
    assert!(parse_version_dir("1.10.0").unwrap() > parse_version_dir("1.9.0").unwrap());
    assert!(parse_version_dir("2.0").unwrap() > parse_version_dir("1.99.99").unwrap());
    assert!(parse_version_dir("v1.2.3").is_none());
    assert!(parse_version_dir("1.2.3-rc1").is_none());
    assert!(parse_version_dir("").is_none());
}

#[test]
fn newest_artifact_wins_and_binaryless_dirs_are_skipped() {
    let exe_name = if cfg!(windows) {
        "paddock-runner.exe"
    } else {
        "paddock-runner"
    };
    let dir = std::env::temp_dir().join(format!("paddock-runners-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for v in ["1.4.0", "1.10.2", "1.9.9"] {
        std::fs::create_dir_all(dir.join(v)).unwrap();
        std::fs::write(dir.join(v).join(exe_name), b"stub").unwrap();
    }
    // newer version dir but no executable inside - must not be elected
    std::fs::create_dir_all(dir.join("2.0.0")).unwrap();
    std::fs::create_dir_all(dir.join("not-a-version")).unwrap();

    let (v, bin) = newest_runner_artifact(&dir).expect("an artifact");
    assert_eq!(v, "1.10.2");
    assert!(bin.ends_with(std::path::Path::new("1.10.2").join(exe_name)));
    let _ = std::fs::remove_dir_all(dir);
}

/// Every key the renderer writes must be DECLARED owned, or `merge_owned_keys`
/// treats it as hand-edited state: the manager would then be unable to change
/// or clear it through the Simple tab, silently and only for that one key.
///
/// Read off the source rather than by rendering, so a key is caught the moment
/// it is typed - no spec has to be able to reach it first.
#[test]
fn render_emits_only_owned_keys() {
    let src = include_str!("../supervisor.rs");
    let start = src.find("fn render_server_config").expect("the renderer");
    // its body ends where the next item at impl indentation begins
    let rest = &src[start..];
    let end = rest[1..]
        .find("\n    /// ")
        .map(|i| i + 1)
        .unwrap_or(rest.len());
    let body = &rest[..end];

    let mut missing: Vec<&str> = Vec::new();
    for (i, _) in body.match_indices("t.insert(\"") {
        let after = &body[i + "t.insert(\"".len()..];
        let key = &after[..after.find('"').expect("a closed key literal")];
        if !OWNED_CONFIG_KEYS.contains(&key) {
            missing.push(key);
        }
    }
    assert!(
        missing.is_empty(),
        "render_server_config writes {missing:?}, which OWNED_CONFIG_KEYS does not declare - \
         add them there or the Simple tab can never clear them"
    );
    // and the scan actually found the renderer, not an empty slice
    assert!(
        body.contains("t.insert(\"model\""),
        "the key scan matched nothing - the body split moved"
    );
}

/// The `[forensics]` owned key round-trips exactly as render writes it and
/// project reads it - the two halves of the Simple-tab contract. The
/// renderer normalizes a bare `{enabled:true}` to the product default; the
/// projector must read every field back. A hand-set scope survives.
#[test]
fn forensics_block_round_trips_render_shape_and_project_shape() {
    // What render_server_config serializes for a bare enable.
    let normalized = ForensicsSpec {
        enabled: true,
        auto: Some("all".into()),
        tool: Some(true),
        device: None,
    };
    let text = toml::to_string_pretty(&toml::Value::try_from(&normalized).unwrap()).unwrap();
    assert!(text.contains("enabled = true"), "{text}");
    assert!(text.contains("auto = \"all\""), "{text}");
    assert!(text.contains("tool = true"), "{text}");
    assert!(
        !text.contains("device"),
        "device omitted when sharing the model GPU: {text}"
    );

    // What project_config_text reads back out of a `[forensics]` block -
    // including a hand-set scope and a cross-GPU device pin.
    let file = "[forensics]\nenabled = true\nauto = \"images\"\ntool = false\ndevice = 1\n";
    let v: toml::Value = toml::from_str(file).unwrap();
    let parsed: ForensicsSpec = v.get("forensics").cloned().unwrap().try_into().unwrap();
    assert!(parsed.enabled);
    assert_eq!(parsed.auto.as_deref(), Some("images"));
    assert_eq!(parsed.tool, Some(false));
    assert_eq!(parsed.device, Some(1));
}

/// `[kv_offload]` must survive the render/project round trip, and the two
/// halves of the disk tier must travel together: a path with no budget or
/// a budget with no path arms nothing and warns at every start, so the
/// renderer never writes half a pair.
#[test]
fn kv_offload_block_round_trips_and_the_disk_budget_stands_alone() {
    let full = KvOffloadSpec {
        enabled: true,
        ram_gb: 24.0,
        nvme_gb: 200.0,
        nvme_path: Some("D:/paddock-cache".into()),
    };
    let text = toml::to_string_pretty(&toml::Value::try_from(&full).unwrap()).unwrap();
    assert!(text.contains("enabled = true"), "{text}");
    assert!(text.contains("ram_gb = 24.0"), "{text}");
    assert!(text.contains("nvme_gb = 200.0"), "{text}");
    assert!(text.contains("nvme_path"), "{text}");

    // RAM only: the disk keys stay out of the file entirely rather than
    // appearing as zeroes a reader would have to interpret
    let ram_only = KvOffloadSpec {
        enabled: true,
        ram_gb: 8.0,
        ..Default::default()
    };
    let text = toml::to_string_pretty(&toml::Value::try_from(&ram_only).unwrap()).unwrap();
    assert!(text.contains("ram_gb = 8.0"), "{text}");
    assert!(
        !text.contains("nvme_gb"),
        "an unset disk budget is absent, not 0: {text}"
    );
    assert!(!text.contains("nvme_path"), "{text}");

    // a disk budget with no folder is complete on its own - the runner
    // defaults the location, so this is the commonest shape and must not
    // be mistaken for half a tier
    let no_path = KvOffloadSpec {
        enabled: true,
        ram_gb: 8.0,
        nvme_gb: 64.0,
        nvme_path: None,
    };
    let text = toml::to_string_pretty(&toml::Value::try_from(&no_path).unwrap()).unwrap();
    assert!(
        text.contains("nvme_gb = 64.0"),
        "the budget survives on its own: {text}"
    );
    assert!(
        !text.contains("nvme_path"),
        "no folder means no key: {text}"
    );

    // and it reads back out of a hand-written file
    let file = "[kv_offload]\nenabled = true\nram_gb = 12.5\nnvme_gb = 64.0\n\
                nvme_path = \"/var/cache/paddock\"\n";
    let v: toml::Value = toml::from_str(file).unwrap();
    let parsed: KvOffloadSpec = v.get("kv_offload").cloned().unwrap().try_into().unwrap();
    assert!(parsed.enabled);
    assert_eq!(parsed.ram_gb, 12.5);
    assert_eq!(parsed.nvme_gb, 64.0);
    assert_eq!(parsed.nvme_path.as_deref(), Some("/var/cache/paddock"));
}

/// The backfill a legacy endpoint gets on its next start: additive only,
/// appended after an array-of-tables (where a bare key could not go), and
/// never written twice.
#[test]
fn stamping_adds_the_block_once_and_changes_nothing_else() {
    let dir = std::env::temp_dir().join(format!("pd-stamp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("11540.toml");
    let before = "\
# a note the operator wrote
model = 'E:\\models\\Tiny-Q8_0.gguf'
max_ctx = 4096

[[mcp_servers]]
server_label = \"tic\"
";
    std::fs::write(&path, before).unwrap();
    let spec = SpawnSpec {
        model: "tiny".into(),
        artifact: Some("q8".into()),
        ..Default::default()
    };
    Supervisor::stamp_catalog_identity(&path, &spec);

    let after = std::fs::read_to_string(&path).unwrap();
    let v: toml::Value = toml::from_str(&after).unwrap();
    assert_eq!(v["catalog"]["model"].as_str(), Some("tiny"));
    assert_eq!(v["catalog"]["artifact"].as_str(), Some("q8"));
    // everything the operator had is untouched, comment included
    assert_eq!(v["model"].as_str(), Some(r"E:\models\Tiny-Q8_0.gguf"));
    assert_eq!(v["max_ctx"].as_integer(), Some(4096));
    assert_eq!(v["mcp_servers"][0]["server_label"].as_str(), Some("tic"));
    assert!(after.contains("# a note the operator wrote"));

    // a second start must not touch it, and a path-shaped model has no
    // identity to record
    Supervisor::stamp_catalog_identity(&path, &spec);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), after);
    std::fs::write(&path, before).unwrap();
    Supervisor::stamp_catalog_identity(
        &path,
        &SpawnSpec {
            model: r"E:\models\Tiny-Q8_0.gguf".into(),
            ..Default::default()
        },
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
    let _ = std::fs::remove_dir_all(dir);
}

/// The CLI's `switch` changes one field and resends the rest, which only
/// works if a spec survives serde in both directions.
///
/// It used to send just model/max_ctx/max_batch, and the switch route reads
/// an absent OWNED key as "cleared" - so the verb silently stripped an
/// endpoint's kv_cache_dtype, spec policy, MCP connectors and web-search
/// settings for the crime of not mentioning them. The verb now rebuilds its
/// request from `GET /api/servers/{port}/file`'s `spec`, so anything lost
/// in this round trip is lost on every swap.
#[test]
fn a_spec_round_trips_so_a_swap_can_keep_what_it_did_not_mention() {
    let original = SpawnSpec {
        model: "tiny".into(),
        host: Some(std::net::Ipv4Addr::LOCALHOST.into()),
        max_ctx: Some(8192),
        max_batch: Some(4),
        kv_cache_dtype: Some("f16".into()),
        spec_policy: Some("off".into()),
        api_key: Some("pd-keepme".into()),
        web_search_provider: Some("brave".into()),
        mcp_servers: vec![serde_json::json!({ "server_label": "tic" })],
        ..Default::default()
    };
    let json = serde_json::to_value(&original).expect("a spec must serialize");
    // The wire name the switch route reads, not the Rust field name.
    assert_eq!(
        json["spec"].as_str(),
        Some("off"),
        "spec_policy must serialize as `spec`"
    );

    let back: SpawnSpec = serde_json::from_value(json).expect("and deserialize");
    assert_eq!(back.host, Some(std::net::Ipv4Addr::LOCALHOST.into()));
    // the fields the old verb dropped
    assert_eq!(back.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(back.spec_policy.as_deref(), Some("off"));
    assert_eq!(back.api_key.as_deref(), Some("pd-keepme"));
    assert_eq!(back.web_search_provider.as_deref(), Some("brave"));
    assert_eq!(back.mcp_servers.len(), 1, "connectors must survive a swap");
    // and the envelope, which the verb may legitimately override
    assert_eq!(back.max_ctx, Some(8192));
    assert_eq!(back.max_batch, Some(4));
}

/// The upgrade path every existing endpoint takes: a file written before
/// the `[catalog]` block existed has none, and the first save has to add one.
///
/// This is the exact shape `merge_owned_keys` warns about - inserting a new
/// item into a document that already ends in an array-of-tables, where TOML
/// would read a naively-appended key as a member of that last table. A
/// config with MCP servers attached is not an edge case, so the merge is
/// pinned here rather than trusted.
#[test]
fn merge_adds_a_catalog_block_to_a_file_that_ends_in_mcp_servers() {
    let current = "\
model = 'E:\\models\\Tiny-Q8_0.gguf'
max_ctx = 4096

[[mcp_servers]]
server_label = \"tic\"
server_url = \"https://mcp.tic.io\"
";
    // the render re-emits mcp_servers because the spec still has them -
    // absent in the render would mean the operator turned them off
    let rendered = "\
model = 'E:\\models\\Tiny-Q8_0.gguf'
max_ctx = 8192

[catalog]
model = \"tiny\"
artifact = \"q8\"

[[mcp_servers]]
server_label = \"tic\"
server_url = \"https://mcp.tic.io\"
";
    let out = merge_owned_keys(current, rendered).unwrap();
    let v: toml::Value = toml::from_str(&out).unwrap();
    assert_eq!(v["catalog"]["model"].as_str(), Some("tiny"));
    assert_eq!(v["catalog"]["artifact"].as_str(), Some("q8"));
    assert_eq!(v["max_ctx"].as_integer(), Some(8192));
    // the block landed as its own table, not swallowed into mcp_servers
    assert_eq!(v["mcp_servers"].as_array().map(Vec::len), Some(1));
    assert_eq!(v["mcp_servers"][0]["server_label"].as_str(), Some("tic"));
    assert!(
        v["mcp_servers"][0].get("model").is_none(),
        "catalog keys leaked into the MCP entry"
    );
}

/// Absent-in-render means DELETE for every owned key, and `catalog` is no
/// exception: point an endpoint at a GGUF the catalog does not know and its
/// stale identity has to go, or the editor keeps offering a model that is
/// no longer being served.
#[test]
fn merge_clears_a_catalog_block_the_render_dropped() {
    let current =
        "model = \"/models/old.gguf\"\n\n[catalog]\nmodel = \"tiny\"\nartifact = \"q8\"\n";
    let rendered = "model = \"/models/imported.gguf\"\n";
    let out = merge_owned_keys(current, rendered).unwrap();
    let v: toml::Value = toml::from_str(&out).unwrap();
    assert_eq!(v["model"].as_str(), Some("/models/imported.gguf"));
    assert!(
        v.get("catalog").is_none(),
        "the catalog block should have been cleared"
    );
}

#[test]
fn merge_keeps_foreign_keys_and_clears_owned_ones() {
    let current = "\
# my own note
model = \"/models/old.gguf\"
max_ctx = 4096
mmproj = \"/models/tower.gguf\"
log_file = \"/tmp/mine.log\"
";
    // vision turned off (no mmproj), context raised, a flag we do not own
    let rendered = "model = \"/models/new.gguf\"\nmax_ctx = 8192\n";
    let out = merge_owned_keys(current, rendered).unwrap();
    let v: toml::Value = toml::from_str(&out).unwrap();
    assert_eq!(v["model"].as_str(), Some("/models/new.gguf"));
    assert_eq!(v["max_ctx"].as_integer(), Some(8192));
    // absent in the render = off, not "leave the old one"
    assert!(v.get("mmproj").is_none(), "mmproj should have been cleared");
    // never ours, never touched
    assert_eq!(v["log_file"].as_str(), Some("/tmp/mine.log"));
    assert!(
        out.contains("# my own note"),
        "the user's comment should survive"
    );
}

/// The positional trap: a new scalar appended to a document that ends in an
/// array-of-tables reads as a member of the last table. The merge must notice
/// and fall back rather than hand back text whose MEANING changed.
#[test]
fn merge_falls_back_rather_than_mangle_a_trailing_table() {
    let current = "model = \"/m.gguf\"\n\n[[mcp_servers]]\nserver_label = \"github\"\n";
    let rendered =
        "model = \"/m.gguf\"\nmax_ctx = 8192\n\n[[mcp_servers]]\nserver_label = \"github\"\n";
    let out = merge_owned_keys(current, rendered).unwrap();
    let v: toml::Value = toml::from_str(&out).unwrap();
    // max_ctx is a ROOT key, not a field of the mcp_servers entry
    assert_eq!(v["max_ctx"].as_integer(), Some(8192));
    assert_eq!(v["mcp_servers"].as_array().map(Vec::len), Some(1));
    assert!(v["mcp_servers"][0].get("max_ctx").is_none());
}

/// EmbeddingGemma 2's towers are two artifacts with two switches: the picture
/// tower rides `mmproj` by default, the audio tower `audio_mmproj` only when
/// asked - and its off is SAID (`audio = false`), since the runner loads an
/// audio tower it finds beside the weights. All three shapes read back
/// through the projection and the file-derived spec, and re-render the same.
#[tokio::test]
async fn split_towers_render_by_their_own_switches_and_read_back() {
    let dir = tempfile::tempdir().unwrap();
    let sup = installed_model_supervisor_on(dir.path(), "embeddinggemma-2", None, "cuda");
    let spec = |vision, audio| SpawnSpec {
        model: "embeddinggemma-2".into(),
        artifact: Some("q8".into()),
        vision,
        audio,
        ..Default::default()
    };
    let parse = |text: &str| -> toml::Value { toml::from_str(text).unwrap() };
    let ends = |v: &toml::Value, key: &str, file: &str| {
        v.get(key)
            .and_then(toml::Value::as_str)
            .is_some_and(|p| p.ends_with(file))
    };
    let default = sup.preview_config(spec(None, None)).await.unwrap();
    let v = parse(&default);
    assert!(
        ends(&v, "mmproj", "embeddinggemma-2-mmproj-vision-BF16.gguf"),
        "{default}"
    );
    assert!(v.get("audio_mmproj").is_none(), "{default}");
    assert_eq!(v["audio"].as_bool(), Some(false), "{default}");

    let both = sup.preview_config(spec(None, Some(true))).await.unwrap();
    let v = parse(&both);
    assert!(
        ends(&v, "mmproj", "embeddinggemma-2-mmproj-vision-BF16.gguf"),
        "{both}"
    );
    assert!(
        ends(
            &v,
            "audio_mmproj",
            "embeddinggemma-2-mmproj-audio-BF16.gguf"
        ),
        "{both}"
    );
    assert!(v.get("audio").is_none(), "{both}");

    let audio_only = sup
        .preview_config(spec(Some(false), Some(true)))
        .await
        .unwrap();
    let v = parse(&audio_only);
    assert!(v.get("mmproj").is_none(), "{audio_only}");
    assert_eq!(v["vision"].as_bool(), Some(false), "{audio_only}");
    assert!(
        ends(
            &v,
            "audio_mmproj",
            "embeddinggemma-2-mmproj-audio-BF16.gguf"
        ),
        "{audio_only}"
    );

    for (text, vision, audio) in [
        (&default, true, false),
        (&both, true, true),
        (&audio_only, false, true),
    ] {
        let p = sup.project_config_text(text).unwrap();
        assert_eq!((p.vision, p.audio), (vision, audio), "{text}");
        let again = sup
            .preview_config(sup.spec_from_config_text(text).unwrap())
            .await
            .unwrap();
        let v = parse(&again);
        assert_eq!(v.get("mmproj").is_some(), vision, "{again}");
        assert_eq!(v.get("audio_mmproj").is_some(), audio, "{again}");
    }
}
