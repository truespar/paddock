//! Control-plane integration only: never starts a runner or loads model weights.
use std::{path::Path, sync::Arc, time::Duration};

use paddock_manager::{
    registry::Registry,
    supervisor::{SpawnDefaults, SpawnSpec, Supervisor},
};

fn supervisor(dir: &Path, device: &str) -> Supervisor {
    Supervisor::new(
        SpawnDefaults {
            runner_bin: None,
            runners_dir: dir.join("runners"),
            device: device.into(),
            kernel_pack: None,
            models_dirs: vec![dir.join("models")],
            logs_dir: dir.join("logs"),
            work_dir: dir.to_path_buf(),
            base_port: 18100,
            health_timeout: Duration::from_secs(1),
        },
        Arc::new(Registry::new(dir.join("models")).with_backend(device)),
        None,
        None,
    )
}

fn spec() -> SpawnSpec {
    SpawnSpec {
        model: "qwen3.8-27b".into(),
        artifact: Some("mlx-4bit".into()),
        port: Some(18100),
        ..Default::default()
    }
}

fn flash_next_spec() -> SpawnSpec {
    SpawnSpec {
        model: "qwen3.8-flash-next".into(),
        ..spec()
    }
}

#[tokio::test]
async fn diarization_previews_both_formats_without_chat_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    for (artifact, path) in [
        ("mlx-bf16", "Nemotron-3-Diarization-MLX"),
        (
            "q8",
            "Nemotron-3-Diarization-GGUF/Nemotron-3-Diarization.q8_0.gguf",
        ),
    ] {
        let request = SpawnSpec {
            model: "nemotron-3-diarization".into(),
            artifact: Some(artifact.into()),
            ..Default::default()
        };
        let raw = metal.preview_config(request.clone()).await.unwrap();
        let config: toml::Value = toml::from_str(&raw).unwrap();
        assert_eq!(
            config["model"].as_str(),
            dir.path().join("models").join(path).to_str()
        );
        assert_eq!(config["spec"].as_str(), Some("off"));
        assert_eq!(config["max_ctx"].as_integer(), Some(684));
        assert_eq!(config["max_batch"].as_integer(), Some(1));
        assert_eq!(config["kv_cache_dtype"].as_str(), Some("auto"));
        assert!(config.get("mmproj").is_none());
        // the same artifact previews on CUDA with the same envelope
        let raw = supervisor(dir.path(), "cuda")
            .preview_config(request)
            .await
            .unwrap();
        let cuda: toml::Value = toml::from_str(&raw).unwrap();
        assert_eq!(cuda["model"], config["model"]);
        assert_eq!(cuda["spec"].as_str(), Some("off"));
        assert_eq!(cuda["max_ctx"].as_integer(), Some(684));
        assert_eq!(cuda["max_batch"].as_integer(), Some(1));
        assert!(cuda.get("mmproj").is_none());
    }
}

#[tokio::test]
async fn bonsai_preview_elects_native_ternary_bundle_and_exact_kv_precision() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    let request = SpawnSpec {
        model: "bonsai-2-27b".into(),
        artifact: Some("mlx-2bit".into()),
        vision: Some(true),
        ..spec()
    };
    let raw = metal.preview_config(request.clone()).await.unwrap();
    let config: toml::Value = toml::from_str(&raw).unwrap();
    assert_eq!(
        config["model"].as_str(),
        dir.path()
            .join("models")
            .join("Ternary-Bonsai-2-27B-mlx-2bit")
            .to_str()
    );
    assert_eq!(config["max_batch"].as_integer(), Some(1));
    assert_eq!(config["max_ctx"].as_integer(), Some(32768));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("f32"));
    assert_eq!(config["spec"].as_str(), Some("off"));
    for key in ["mmproj", "mtp", "fp8_native"] {
        assert!(config.get(key).is_none(), "unexpected {key}");
    }
    let projected = metal.project_config_text(&raw).unwrap();
    assert!(projected.vision);
    assert_eq!(projected.artifact.as_deref(), Some("mlx-2bit"));
    let registry = Registry::new(dir.path().join("models")).with_backend("metal");
    let model = registry.catalog_of("bonsai-2-27b").unwrap();
    let artifact = model.default_weights_for_backend("metal", None).unwrap();
    assert_eq!(artifact.id, "mlx-2bit");
    assert_eq!(
        artifact
            .runtime
            .estimate_kv_dtype(paddock_estimator::KvDtype::F16),
        paddock_estimator::KvDtype::F32
    );
    assert_eq!(artifact.shape.as_ref().unwrap().weight_bytes, 8169821120);
    assert!(artifact.runtime.memory.is_some());
    assert_eq!(artifact.files.len(), 11);
    for file in &artifact.files {
        assert_eq!(
            file.url,
            format!("https://models.truespar.io/models/{}", file.dest)
        );
        assert_eq!(file.sha256.len(), 64);
        assert!(!file.dest.ends_with(".py"));
    }
    let cuda = supervisor(dir.path(), "cuda");
    assert!(cuda.preview_config(request.clone()).await.is_err());
    for bad in [
        SpawnSpec {
            kv_cache_dtype: Some("f16".into()),
            ..request.clone()
        },
        SpawnSpec {
            max_batch: Some(5),
            ..request.clone()
        },
        SpawnSpec {
            vision: Some(false),
            ..request
        },
    ] {
        assert!(metal.preview_config(bad).await.is_err());
    }
    let cuda_registry = registry.with_backend("cuda");
    assert_eq!(
        cuda_registry
            .catalog_of("bonsai-2-27b")
            .unwrap()
            .default_weights_for_backend("cuda", None)
            .unwrap()
            .id,
        "ptq1"
    );
}

#[tokio::test]
async fn splash_preview_keeps_one_native_package_and_one_user_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    let s = SpawnSpec {
        artifact: Some("splash-4bit".into()),
        vision: Some(true),
        ..spec()
    };
    let raw = metal.preview_config(s.clone()).await.unwrap();
    let config: toml::Value = toml::from_str(&raw).unwrap();
    assert_eq!(
        config["model"].as_str(),
        dir.path()
            .join("models")
            .join("Qwen3.8-27B-Splash")
            .to_str()
    );
    assert_eq!(config["max_batch"].as_integer(), Some(1));
    assert_eq!(config["max_ctx"].as_integer(), Some(32768));
    assert_eq!(config["spec"].as_str(), Some("adaptive"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("auto"));
    for key in ["mmproj", "mtp", "fp8_native"] {
        assert!(config.get(key).is_none());
    }
    let projected = metal.project_config_text(&raw).unwrap();
    assert!(projected.vision);
    assert_eq!(projected.artifact.as_deref(), Some("splash-4bit"));
    assert!(
        supervisor(dir.path(), "cuda")
            .preview_config(s)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn multimodal_mlx_preview_and_edit_keep_embedded_vision() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    let cuda = supervisor(dir.path(), "cuda");
    for (model, folder) in [
        ("gemma-4-31b", "Gemma-4-31B-IT-MLX-4bit"),
        ("muse-glimmer-30b", "Muse-Glimmer-30B-MLX-4bit"),
    ] {
        let s = SpawnSpec {
            model: model.into(),
            vision: Some(true),
            ..spec()
        };
        let raw = metal.preview_config(s.clone()).await.unwrap();
        let c: toml::Value = toml::from_str(&raw).unwrap();
        assert_eq!(c["device"].as_str(), Some("metal"));
        assert_eq!(c["max_ctx"].as_integer(), Some(4096));
        assert_eq!(c["max_batch"].as_integer(), Some(4));
        assert_eq!(c["kv_cache_dtype"].as_str(), Some("auto"));
        assert_eq!(c["spec"].as_str(), Some("off"));
        assert_eq!(
            c["model"].as_str(),
            dir.path().join("models").join(folder).to_str()
        );
        for key in ["mmproj", "mtp", "fp8_native"] {
            assert!(c.get(key).is_none());
        }
        let p = metal.project_config_text(&raw).unwrap();
        assert!(
            p.vision,
            "an embedded tower must survive Simple/Edit projection"
        );
        assert_eq!(p.artifact.as_deref(), Some("mlx-4bit"));
        assert!(cuda.preview_config(s.clone()).await.is_err());
        for bad in [
            SpawnSpec {
                vision: Some(false),
                ..s.clone()
            },
            SpawnSpec {
                max_ctx: Some(4097),
                ..s.clone()
            },
            SpawnSpec {
                max_batch: Some(5),
                ..s.clone()
            },
            SpawnSpec {
                kv_cache_dtype: Some("fp8_e4m3".into()),
                ..s.clone()
            },
            SpawnSpec {
                drafter: Some("drafter".into()),
                ..s.clone()
            },
            SpawnSpec {
                spec_policy: Some("on".into()),
                ..s.clone()
            },
        ] {
            assert!(metal.preview_config(bad).await.is_err());
        }
    }
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[tokio::test]
async fn flash_next_mlx_preview_uses_the_measured_envelope_and_directory() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let c: toml::Value =
        toml::from_str(&sup.preview_config(flash_next_spec()).await.unwrap()).unwrap();
    assert_eq!(c["device"].as_str(), Some("metal"));
    assert_eq!(c["max_ctx"].as_integer(), Some(4096));
    assert_eq!(c["max_batch"].as_integer(), Some(4));
    assert_eq!(c["kv_cache_dtype"].as_str(), Some("auto"));
    assert_eq!(c["spec"].as_str(), Some("off"));
    assert_eq!(
        c["model"].as_str(),
        dir.path()
            .join("models")
            .join("Qwen3.8-Flash-Next-MLX-4bit")
            .to_str()
    );
    assert_eq!(c["catalog"]["artifact"].as_str(), Some("mlx-4bit"));
    for key in ["mmproj", "mtp", "fp8_native", "kernel_pack"] {
        assert!(c.get(key).is_none(), "unexpected {key}");
    }
    let mut smaller = flash_next_spec();
    smaller.max_ctx = Some(2048);
    smaller.max_batch = Some(2);
    let c: toml::Value = toml::from_str(&sup.preview_config(smaller).await.unwrap()).unwrap();
    assert_eq!(c["max_ctx"].as_integer(), Some(2048));
    assert_eq!(c["max_batch"].as_integer(), Some(2));
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());

    // New configurations pin implicit defaults too. The generic 27B envelope
    // is 32768 x 1 for local MLX; Flash-Next GGUF retains its 4K/four-slot limit.
    // Explicit and existing saved configurations are never silently rewritten.
    for (s, context, batch) in [
        (spec(), 32768, 1),
        (
            SpawnSpec {
                artifact: Some("iq3".into()),
                ..flash_next_spec()
            },
            4096,
            4,
        ),
    ] {
        let c: toml::Value = toml::from_str(&sup.preview_config(s).await.unwrap()).unwrap();
        assert_eq!(c["max_ctx"].as_integer(), Some(context));
        assert_eq!(c["max_batch"].as_integer(), Some(batch));
    }
}

#[tokio::test]
async fn flash_next_mlx_preview_accepts_default_and_refuses_incompatible_options() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut implicit = flash_next_spec();
    implicit.artifact = None;
    assert!(sup.preview_config(implicit).await.is_ok());
    assert!(
        supervisor(dir.path(), "cuda")
            .preview_config(flash_next_spec())
            .await
            .is_err()
    );
    for (ctx, batch) in [(0, 4), (4097, 4), (4096, 0), (4096, 5)] {
        let mut s = flash_next_spec();
        s.max_ctx = Some(ctx);
        s.max_batch = Some(batch);
        assert!(
            sup.preview_config(s).await.is_err(),
            "accepted {ctx}/{batch}"
        );
    }
    for dtype in ["f16", "bf16", "fp8_e4m3"] {
        let mut s = flash_next_spec();
        s.kv_cache_dtype = Some(dtype.into());
        assert!(sup.preview_config(s).await.is_err());
    }
    for policy in ["on", "auto", "ladder"] {
        let mut s = flash_next_spec();
        s.spec_policy = Some(policy.into());
        assert!(sup.preview_config(s).await.is_err());
    }
    for s in [
        SpawnSpec {
            vision: Some(true),
            ..flash_next_spec()
        },
        SpawnSpec {
            drafter: Some("dflash2".into()),
            ..flash_next_spec()
        },
        SpawnSpec {
            fp8_native: true,
            ..flash_next_spec()
        },
    ] {
        assert!(sup.preview_config(s).await.is_err());
    }
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[tokio::test]
async fn whisper_metal_previews_are_bounded_and_companion_free() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    for id in [
        "kb-whisper-large",
        "nb-whisper-large",
        "roest-v3-whisper-1.5b",
    ] {
        let mut s = SpawnSpec {
            model: id.into(),
            ..Default::default()
        };
        let default_config: toml::Value =
            toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
        assert_eq!(default_config["max_ctx"].as_integer(), Some(448));
        assert_eq!(default_config["max_batch"].as_integer(), Some(16));
        s.artifact = Some("f16".into());
        s.max_ctx = Some(448);
        s.max_batch = Some(4);
        let c: toml::Value = toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
        assert_eq!(c["kv_cache_dtype"].as_str(), Some("f16"));
        assert_eq!(c["spec"].as_str(), Some("off"));
        assert!(c.get("mmproj").is_none() && c.get("mtp").is_none());
        for (ctx, batch) in [(449, 4), (448, 17)] {
            let mut bad = s.clone();
            bad.max_ctx = Some(ctx);
            bad.max_batch = Some(batch);
            assert!(sup.preview_config(bad).await.is_err());
        }
        s.kv_cache_dtype = Some("fp8_e4m3".into());
        assert!(sup.preview_config(s).await.is_err());
    }
}

#[tokio::test]
async fn granite_speech_metal_previews_are_pair_safe() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    for id in ["granite-speech-4.1-2b", "granite-speech-4.1-2b-plus"] {
        let mut s = SpawnSpec {
            model: id.into(),
            ..Default::default()
        };
        assert!(sup.preview_config(s.clone()).await.is_ok());
        s.artifact = Some("q8".into());
        s.max_ctx = Some(4096);
        s.max_batch = Some(4);
        let c: toml::Value = toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
        assert!(
            c["mmproj"]
                .as_str()
                .unwrap()
                .ends_with(&format!("{id}-GGUF/mmproj-model-f16.gguf"))
        );
        assert_eq!(c["kv_cache_dtype"].as_str(), Some("f16"));
        assert_eq!(c["spec"].as_str(), Some("off"));
        assert!(c.get("mtp").is_none());
        for (ctx, batch) in [(4097, 4), (4096, 17)] {
            let mut bad = s.clone();
            bad.max_ctx = Some(ctx);
            bad.max_batch = Some(batch);
            assert!(sup.preview_config(bad).await.is_err());
        }
        s.kv_cache_dtype = Some("fp8_e4m3".into());
        assert!(sup.preview_config(s).await.is_err());
    }
}

#[tokio::test]
async fn qwen3_aligner_metal_preview_is_companion_free() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut s = SpawnSpec {
        model: "qwen3-forced-aligner-0.6b".into(),
        ..Default::default()
    };
    assert!(sup.preview_config(s.clone()).await.is_ok());
    s.artifact = Some("bf16".into());
    s.max_ctx = Some(4096);
    s.max_batch = Some(4);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
    assert!(
        config["model"]
            .as_str()
            .unwrap()
            .ends_with("model.safetensors")
    );
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["spec"].as_str(), Some("off"));
    assert!(config.get("mmproj").is_none());
    for (ctx, batch) in [(8193, 4), (4096, 5)] {
        let mut bad = s.clone();
        bad.max_ctx = Some(ctx);
        bad.max_batch = Some(batch);
        assert!(sup.preview_config(bad).await.is_err());
    }
}

#[tokio::test]
async fn qwen3_asr_metal_preview_requires_audio_and_f16() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut s = SpawnSpec {
        model: "qwen3-asr-1.7b".into(),
        ..Default::default()
    };
    assert!(sup.preview_config(s.clone()).await.is_ok());
    s.artifact = Some("q8".into());
    s.max_ctx = Some(4096);
    s.max_batch = Some(4);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
    assert_eq!(config["spec"].as_str(), Some("off"));
    assert!(
        config["mmproj"]
            .as_str()
            .unwrap()
            .ends_with("mmproj-Qwen3-ASR-1.7B-bf16.gguf")
    );
    assert!(config.get("mtp").is_none());
    for (ctx, batch) in [(8193, 4), (4096, 17)] {
        let mut wrong = s.clone();
        wrong.max_ctx = Some(ctx);
        wrong.max_batch = Some(batch);
        assert!(sup.preview_config(wrong).await.is_err());
    }
    s.kv_cache_dtype = Some("fp8_e4m3".into());
    assert!(sup.preview_config(s).await.is_err());
}

#[tokio::test]
async fn unlimited_ocr_metal_preview_requires_q8_pair_and_f16() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut s = SpawnSpec {
        model: "unlimited-ocr".into(),
        ..Default::default()
    };
    assert!(sup.preview_config(s.clone()).await.is_ok());
    s.artifact = Some("q8".into());
    s.max_ctx = Some(4096);
    s.max_batch = Some(4);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
    assert_eq!(config["spec"].as_str(), Some("off"));
    assert!(
        config["mmproj"]
            .as_str()
            .unwrap()
            .ends_with("mmproj-Unlimited-OCR-F16.gguf")
    );
    assert!(config.get("mtp").is_none());
    for (ctx, batch) in [(32769, 4), (4096, 17)] {
        let mut wrong = s.clone();
        wrong.max_ctx = Some(ctx);
        wrong.max_batch = Some(batch);
        assert!(sup.preview_config(wrong).await.is_err());
    }
    s.kv_cache_dtype = Some("fp8_e4m3".into());
    assert!(sup.preview_config(s).await.is_err());
}

#[tokio::test]
async fn paddleocr_metal_preview_requires_elected_pair_and_f16() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut s = SpawnSpec {
        model: "paddleocr-vl-1.6".into(),
        ..Default::default()
    };
    assert!(sup.preview_config(s.clone()).await.is_ok());
    s.artifact = Some("bf16".into());
    s.max_ctx = Some(4096);
    s.max_batch = Some(4);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
    assert_eq!(config["spec"].as_str(), Some("off"));
    assert!(
        config["mmproj"]
            .as_str()
            .unwrap()
            .ends_with("PaddleOCR-VL-1.6-GGUF-mmproj.gguf")
    );
    assert!(config.get("mtp").is_none());
    for (ctx, batch) in [(32769, 4), (4096, 17)] {
        let mut wrong = s.clone();
        wrong.max_ctx = Some(ctx);
        wrong.max_batch = Some(batch);
        assert!(sup.preview_config(wrong).await.is_err());
    }
    s.kv_cache_dtype = Some("fp8_e4m3".into());
    assert!(sup.preview_config(s).await.is_err());
}

#[tokio::test]
async fn metal_gemma_moe_preview_selects_its_own_tower_and_assistant() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut s = SpawnSpec {
        model: "gemma-4-26b-a4b".into(),
        ..Default::default()
    };
    assert!(sup.preview_config(s.clone()).await.is_ok());
    s.artifact = Some("q8".into());
    s.vision = Some(true);
    s.spec_policy = Some("on".into());
    s.max_ctx = Some(4096);
    s.max_batch = Some(4);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
    assert_eq!(config["spec"].as_str(), Some("on"));
    for (key, suffix) in [
        ("mmproj", "gemma-4-26B-A4B-it-mmproj-BF16.gguf"),
        ("mtp", "gemma-4-26B-A4B-it-mtp.gguf"),
    ] {
        assert!(config[key].as_str().unwrap().ends_with(suffix));
    }
    s.kv_cache_dtype = Some("fp8_e4m3".into());
    assert!(sup.preview_config(s).await.is_err());
}

#[tokio::test]
async fn metal_qwen_moe_preview_uses_q8_vision_infile_mtp_and_f16() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let mut s = SpawnSpec {
        model: "qwen3.6-35b-a3b".into(),
        ..Default::default()
    };
    assert!(sup.preview_config(s.clone()).await.is_ok());
    s.artifact = Some("q8".into());
    s.vision = Some(true);
    s.spec_policy = Some("on".into());
    s.max_ctx = Some(4096);
    s.max_batch = Some(4);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
    assert_eq!(config["spec"].as_str(), Some("on"));
    assert!(
        config["mmproj"]
            .as_str()
            .unwrap()
            .ends_with("Qwen3.6-35B-A3B-MTP-GGUF/mmproj-BF16.gguf")
    );
    assert!(config.get("mtp").is_none());
    assert!(config.get("fp8_native").is_none());
    let mut no_vision = s.clone();
    no_vision.vision = Some(false);
    let config: toml::Value =
        toml::from_str(&sup.preview_config(no_vision).await.unwrap()).unwrap();
    assert!(config.get("mmproj").is_none());
    for dtype in ["fp8_e4m3", "bf16"] {
        let mut wrong = s.clone();
        wrong.kv_cache_dtype = Some(dtype.into());
        assert!(sup.preview_config(wrong).await.is_err());
    }
    s.artifact = Some("q4".into());
    assert!(sup.preview_config(s).await.is_err());
}

#[tokio::test]
async fn text_moe_metal_preview_accepts_default_and_disables_companions() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    for (model, artifact) in [
        ("gpt-oss-20b", "mxfp4"),
        ("gpt-oss-120b", "mxfp4"),
        ("laguna-xs-2.1", "q4"),
        ("laguna-s-2.1", "q4"),
        ("nemotron-3.5-lightning-30b", "q8"),
    ] {
        let mut s = SpawnSpec {
            model: model.into(),
            ..Default::default()
        };
        assert!(sup.preview_config(s.clone()).await.is_ok());
        s.artifact = Some(artifact.into());
        let config: toml::Value =
            toml::from_str(&sup.preview_config(s.clone()).await.unwrap()).unwrap();
        assert_eq!(config["device"].as_str(), Some("metal"));
        assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
        assert_eq!(config["spec"].as_str(), Some("off"));
        for key in ["mmproj", "mtp", "fp8_native", "kernel_pack"] {
            assert!(config.get(key).is_none());
        }
        let mut too_long = s.clone();
        too_long.max_ctx = Some(32769);
        assert!(
            sup.preview_config(too_long)
                .await
                .unwrap_err()
                .to_string()
                .contains("max_ctx <= 32768")
        );
        let mut allowed = s.clone();
        allowed.max_ctx = Some(32768);
        assert!(sup.preview_config(allowed).await.is_ok());
        for batch in [0, 65] {
            let mut too_wide = s.clone();
            too_wide.max_batch = Some(batch);
            assert!(
                sup.preview_config(too_wide)
                    .await
                    .unwrap_err()
                    .to_string()
                    .contains("max_batch in 1..=64")
            );
        }
        // CUDA's checkpoint contract is unchanged; only Metal has this cap.
        let mut cuda = s.clone();
        cuda.max_ctx = Some(131072);
        assert!(
            supervisor(dir.path(), "cuda")
                .preview_config(cuda)
                .await
                .is_ok()
        );
        s.kv_cache_dtype = Some("fp8_e4m3".into());
        assert!(sup.preview_config(s).await.is_err());
    }
}

#[tokio::test]
async fn undownloaded_mlx_preview_defaults_to_adaptive_dflash2_and_native_kv() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let text = sup.preview_config(spec()).await.unwrap();
    let config: toml::Value = toml::from_str(&text).unwrap();
    assert_eq!(config["device"].as_str(), Some("metal"));
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("auto"));
    assert_eq!(config["spec"].as_str(), Some("adaptive"));
    assert_eq!(config["max_batch"].as_integer(), Some(1));
    assert_eq!(
        config["model"].as_str(),
        Some(
            dir.path()
                .join("models")
                .join("Qwen3.8-27B-MLX-4bit")
                .to_str()
                .unwrap()
        )
    );
    assert_eq!(config["catalog"]["artifact"].as_str(), Some("mlx-4bit"));
    assert_eq!(
        config["mtp"].as_str(),
        dir.path()
            .join("models")
            // the catalog `dest`, joined as the registry joins it
            .join("Qwen3.8-27B-GGUF/dflash2-Q4_K_M.gguf")
            .to_str()
    );
    for key in ["mmproj", "fp8_native", "kernel_pack"] {
        assert!(config.get(key).is_none(), "unexpected {key}");
    }
    assert!(
        std::fs::read_dir(dir.path()).unwrap().next().is_none(),
        "preview writes nothing"
    );
}

#[tokio::test]
async fn qwen_mlx_explicit_off_survives_default_on_and_does_not_load_a_drafter() {
    let dir = tempfile::tempdir().unwrap();
    let sup = supervisor(dir.path(), "metal");
    let raw = sup
        .preview_config(SpawnSpec {
            spec_policy: Some("off".into()),
            ..spec()
        })
        .await
        .unwrap();
    let config: toml::Value = toml::from_str(&raw).unwrap();
    assert_eq!(config["spec"].as_str(), Some("off"));
    assert!(config.get("mtp").is_none());
    let projected = sup.project_config_text(&raw).unwrap();
    assert_eq!(projected.spec.as_deref(), Some("off"));
}

#[tokio::test]
async fn preview_refuses_incompatible_backend_cache_and_companion_requests() {
    let dir = tempfile::tempdir().unwrap();
    let cuda = supervisor(dir.path(), "cuda");
    assert!(
        cuda.preview_config(spec())
            .await
            .unwrap_err()
            .to_string()
            .contains("requires backend metal")
    );
    let metal = supervisor(dir.path(), "metal");
    for dtype in ["f16", "fp8_e4m3"] {
        let mut s = spec();
        s.kv_cache_dtype = Some(dtype.into());
        assert!(
            metal
                .preview_config(s)
                .await
                .unwrap_err()
                .to_string()
                .contains("kv_cache_dtype")
        );
    }
    for policy in ["on", "auto", "ladder"] {
        let mut s = spec();
        s.spec_policy = Some(policy.into());
        let config: toml::Value = toml::from_str(&metal.preview_config(s).await.unwrap()).unwrap();
        assert_eq!(config["spec"].as_str(), Some(policy));
        assert!(config.get("mtp").is_some());
    }
    let mut vision = spec();
    vision.vision = Some(true);
    assert!(
        metal
            .preview_config(vision)
            .await
            .unwrap_err()
            .to_string()
            .contains("text only")
    );
    let mut drafter = spec();
    drafter.drafter = Some("unrelated-drafter".into());
    assert!(
        metal
            .preview_config(drafter)
            .await
            .unwrap_err()
            .to_string()
            .contains("companion")
    );
    let mut fp8 = spec();
    fp8.fp8_native = true;
    assert!(
        metal
            .preview_config(fp8)
            .await
            .unwrap_err()
            .to_string()
            .contains("companion")
    );
}

#[tokio::test]
async fn metal_gguf_previews_use_f16_without_changing_cuda_defaults() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    for (model, artifact) in [
        ("gemma-4-31b", "q8"),
        ("qwen3.8-27b", "q4"),
        ("granite-4.2-8b", "q8"),
    ] {
        let s = SpawnSpec {
            model: model.into(),
            artifact: Some(artifact.into()),
            ..Default::default()
        };
        let text = metal.preview_config(s.clone()).await.unwrap();
        let config: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"), "{model}");
        if model.starts_with("granite") {
            assert_eq!(config["spec"].as_str(), Some("off"));
        }
        let mut auto = s.clone();
        auto.kv_cache_dtype = Some("auto".into());
        assert!(metal.preview_config(auto).await.is_ok());
        let mut fp8 = s.clone();
        fp8.kv_cache_dtype = Some("fp8_e4m3".into());
        assert!(
            metal
                .preview_config(fp8)
                .await
                .unwrap_err()
                .to_string()
                .contains("kv_cache_dtype")
        );
        let mut snapshot = s;
        snapshot.fp8_native = true;
        assert!(metal.preview_config(snapshot).await.is_err());
    }
    let cuda = supervisor(dir.path(), "cuda");
    let s = SpawnSpec {
        model: "gemma-4-31b".into(),
        artifact: Some("q8".into()),
        kv_cache_dtype: Some("fp8_e4m3".into()),
        ..Default::default()
    };
    let text = cuda.preview_config(s).await.unwrap();
    let config: toml::Value = toml::from_str(&text).unwrap();
    assert_eq!(config["kv_cache_dtype"].as_str(), Some("fp8_e4m3"));
}

#[tokio::test]
async fn metal_preview_refuses_missing_models_formats_and_muse_dflash1() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    for (model, artifact) in [
        ("qwen3.6-35b-a3b", "q4"),
        ("qwen3.8-27b", "q3"),
        ("granite-4.2-8b", "nvfp4"),
    ] {
        let s = SpawnSpec {
            model: model.into(),
            artifact: Some(artifact.into()),
            ..Default::default()
        };
        assert!(metal.preview_config(s).await.is_err(), "{model}/{artifact}");
    }
    let s = SpawnSpec {
        model: "muse-glimmer-30b".into(),
        artifact: Some("q8".into()),
        drafter: Some("drafter".into()),
        ..Default::default()
    };
    assert!(metal.preview_config(s).await.is_err());
}

#[tokio::test]
async fn metal_qwen9b_preview_keeps_vision_and_infile_mtp_without_external_draft() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    for artifact in ["q8", "q4"] {
        let text = metal
            .preview_config(SpawnSpec {
                model: "qwen3.5-9b".into(),
                artifact: Some(artifact.into()),
                vision: Some(true),
                spec_policy: Some("on".into()),
                ..Default::default()
            })
            .await
            .unwrap();
        let config: toml::Value = toml::from_str(&text).unwrap();
        assert_eq!(config["device"].as_str(), Some("metal"));
        assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
        assert_eq!(config["spec"].as_str(), Some("on"));
        assert_eq!(config["catalog"]["artifact"].as_str(), Some(artifact));
        assert!(
            config["mmproj"]
                .as_str()
                .unwrap()
                .ends_with("Qwen3.5-9B-MTP-GGUF/mmproj-BF16.gguf")
        );
        assert!(config.get("mtp").is_none(), "MTP must use the in-file head");
        assert!(config.get("fp8_native").is_none());
    }
}

#[tokio::test]
async fn metal_qwen3_retrieval_preview_uses_same_gguf_without_companions() {
    let dir = tempfile::tempdir().unwrap();
    let metal = supervisor(dir.path(), "metal");
    for kind in ["embedding", "reranker"] {
        for size in ["0.6", "4", "8"] {
            let text = metal
                .preview_config(SpawnSpec {
                    model: format!("qwen3-{kind}-{size}b"),
                    ..Default::default()
                })
                .await
                .unwrap();
            let config: toml::Value = toml::from_str(&text).unwrap();
            assert_eq!(config["device"].as_str(), Some("metal"));
            assert_eq!(config["kv_cache_dtype"].as_str(), Some("f16"));
            assert!(config["model"].as_str().unwrap().ends_with(".gguf"));
            for k in ["mmproj", "mtp", "fp8_native"] {
                assert!(config.get(k).is_none(), "{k}");
            }
        }
    }
}
