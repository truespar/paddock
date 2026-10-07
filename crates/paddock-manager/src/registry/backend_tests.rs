use super::*;

#[test]
fn diarization_catalog_serves_metal_and_cuda_non_chat_with_measured_bounded_memory() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let model = metal.catalog_of("nemotron-3-diarization").unwrap();
    assert_eq!(model.capability, ["diarization"]);
    assert_eq!(
        crate::estimate::kind_for(&model.capability),
        paddock_estimator::ModelKind::Encoder
    );
    // the trained BF16 values are the default on both lanes; Q8_0 is the
    // smaller option
    for backend in ["metal", "cuda"] {
        assert_eq!(
            model.default_weights_for_backend(backend, None).unwrap().id,
            "mlx-bf16"
        );
    }
    assert!(model.default_weights_for_backend("vulkan", None).is_none());
    for id in ["mlx-bf16", "q8"] {
        let a = model.artifact(id).unwrap();
        assert!(a.runtime.supports_backend("metal"));
        assert!(a.runtime.supports_backend("cuda"));
        assert_eq!(a.runtime.checkpoint_dir, id == "mlx-bf16");
        assert_eq!(a.runtime.default_spec.as_deref(), Some("off"));
        assert_eq!(a.workspace, Some(36_133_888));
        assert_eq!(a.runtime.memory.as_ref().unwrap().kv_reserve_sequences, 0);
        assert!(a.shape.as_ref().unwrap().kv_layers.is_empty());
        assert_eq!(metal.default_envelope(&model.id, Some(id)), (684, 1));
        for f in &a.files {
            assert_eq!(
                f.url,
                format!("https://models.truespar.io/models/{}", f.dest)
            );
            assert!(f.size > 0 && f.sha256.len() == 64);
        }
        assert_eq!(a.files.len(), if id == "mlx-bf16" { 5 } else { 3 });
        assert!(a.files.iter().any(|f| f.dest.ends_with("/LICENSE")));
        assert!(a.files.iter().any(|f| f.dest.ends_with("/README.md")));
        if id == "mlx-bf16" {
            assert!(
                a.files
                    .iter()
                    .any(|f| f.dest.ends_with("/NVIDIA_MODEL_CARD.md"))
            );
        }
        let source = a
            .source
            .as_ref()
            .expect("mirror retains upstream provenance");
        assert_eq!(source.base_model, "nvidia/Nemotron-3-Diarization");
        assert_eq!(source.license, "openmdw-1.1");
        assert_eq!(source.license_url, "https://openmdw.ai/license/1-1/");
        let (repo, revision) = if id == "mlx-bf16" {
            (
                "mlx-community/Nemotron-3-Diarization",
                "59ed2dbfc1346dcea9d423c71306a3a2499c568f",
            )
        } else {
            (
                "nvidia/Nemotron-3-Diarization",
                "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
            )
        };
        assert_eq!(source.repo, repo);
        assert_eq!(source.revision, revision);
        let caps = metal
            .capability_of(a.entry_path(metal.models_dir()).unwrap().to_str().unwrap())
            .unwrap();
        assert_eq!(caps, ["diarization"]);
    }
}

#[test]
fn laya_metal_uses_original_bundle_and_encoder_memory_not_chat_kv() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let model = metal.catalog_of("laya").unwrap();
    let artifact = model.default_weights_for_backend("metal", None).unwrap();
    assert_eq!(artifact.id, "f16");
    assert_eq!(artifact.capabilities(model), ["decision"]);
    assert!(artifact.runtime.checkpoint_dir);
    assert_eq!(artifact.runtime.kv_cache_dtype.as_deref(), Some("auto"));
    assert_eq!(artifact.files.len(), 16);
    let memory = artifact.runtime.memory.as_ref().unwrap();
    assert_eq!(memory.weight_bytes, 2_330_839_064);
    assert_eq!(memory.workspace_bytes, Some(109_651_460));
    assert_eq!(memory.kv_reserve_sequences, 0);
    assert!(artifact.shape.as_ref().unwrap().kv_layers.is_empty());
    assert!(
        artifact
            .entry_path(metal.models_dir())
            .unwrap()
            .ends_with("laya")
    );
    let cuda = metal.with_backend("cuda");
    let artifact = cuda.catalog_of("laya").unwrap().default_weights().unwrap();
    assert_eq!(artifact.workspace, Some(337_865_736));
    assert!(artifact.runtime.memory.is_none());
}

#[test]
fn clef_flash_cuda_keeps_its_q8_gguf_and_vision_companion() {
    let cuda = Registry::new("./models".into()).with_backend("cuda");
    let model = cuda.catalog_of("clef-flash").unwrap();
    assert_eq!(
        model
            .weights()
            .filter(|a| a.runtime.supports_backend("cuda"))
            .map(|a| a.id.as_str())
            .collect::<Vec<_>>(),
        ["q8"]
    );
    let artifact = model.default_weights_for_backend("cuda", None).unwrap();
    assert_eq!(artifact.quant.as_deref(), Some("Q8_0"));
    assert!(artifact.runtime.memory.is_none());
    assert_eq!(artifact.workspace, Some(4_730_142_720));
    assert_eq!(artifact.shape.as_ref().unwrap().weight_bytes, 9_658_079_232);
    assert_eq!(artifact.capabilities(model), ["decision", "vision"]);
    let bundle: Vec<&str> = model
        .default_bundle_for_backend("cuda", None)
        .iter()
        .map(|a| a.id.as_str())
        .collect();
    assert_eq!(bundle, ["q8", "vision"]);
    assert!(
        model
            .artifacts
            .iter()
            .filter(|a| a.runtime.supports_backend("cuda"))
            .flat_map(|a| &a.files)
            .all(|f| {
                f.url
                    .starts_with("https://models.truespar.io/models/Clef-Flash-GGUF/")
            })
    );
}

#[test]
fn clef_27b_cuda_keeps_its_q8_gguf_and_vision_companion() {
    let cuda = Registry::new("./models".into()).with_backend("cuda");
    let model = cuda.catalog_of("clef").unwrap();
    let artifact = model.default_weights_for_backend("cuda", None).unwrap();
    assert_eq!(artifact.id, "q8");
    assert_eq!(artifact.workspace, Some(6_672_105_472));
    assert_eq!(
        artifact.shape.as_ref().unwrap().weight_bytes,
        28_733_401_088
    );
    let bundle: Vec<&str> = model
        .default_bundle_for_backend("cuda", None)
        .iter()
        .map(|a| a.id.as_str())
        .collect();
    assert_eq!(bundle, ["q8", "vision"]);
    assert!(artifact.files.iter().all(|f| {
        f.url
            .starts_with("https://models.truespar.io/models/Clef-GGUF/")
    }));
}

#[test]
fn both_clefs_offer_native_metal_q8_and_mlx8_text_and_vision_decisions_without_chat() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    for (id, q8_bytes, mlx_bytes, files, revision) in [
        (
            "clef-flash",
            9_650_477_056,
            10_677_937_088,
            13,
            "dfa0993decb4f8507a0eae01afd1b2d33a4bb734",
        ),
        (
            "clef",
            28_725_405_696,
            29_768_205_248,
            17,
            "ffcdb6132b3cc94e523860321dd9ed58bc1ee9a1",
        ),
    ] {
        let model = metal.catalog_of(id).unwrap();
        assert_eq!(
            model.default_weights_for_backend("metal", None).unwrap().id,
            "q8"
        );
        assert_eq!(
            model
                .default_bundle_for_backend("metal", None)
                .iter()
                .map(|a| a.id.as_str())
                .collect::<Vec<_>>(),
            ["q8", "vision"]
        );
        assert_eq!(
            model.weights().map(|a| a.id.as_str()).collect::<Vec<_>>(),
            ["q8", "mlx8"]
        );
        for (artifact_id, bytes) in [("q8", q8_bytes), ("mlx8", mlx_bytes)] {
            let a = model.artifact(artifact_id).unwrap();
            assert!(a.runtime.supports_backend("metal"));
            assert_eq!(a.capabilities(model), ["decision", "vision"]);
            assert_eq!(a.runtime.embedded_vision, artifact_id == "mlx8");
            assert_eq!(
                a.runtime.companions,
                Some(if artifact_id == "q8" {
                    vec!["vision".into()]
                } else {
                    vec![]
                })
            );
            assert_eq!(a.runtime.default_spec.as_deref(), Some("off"));
            let memory = a.runtime.memory.as_ref().unwrap();
            assert_eq!(memory.weight_bytes, bytes);
            assert_eq!(memory.kv_reserve_sequences, 0);
            assert_eq!((memory.max_ctx, memory.max_batch), (16384, 256));
            assert!(a.shape.as_ref().unwrap().kv_layers.is_empty());
            assert_eq!(metal.default_envelope(id, Some(artifact_id)), (16384, 1));
            let entry = a.entry_path(metal.models_dir()).unwrap();
            assert_eq!(
                metal.capability_of(entry.to_str().unwrap()).unwrap(),
                ["decision", "vision"]
            );
        }
        let mlx = model.artifact("mlx8").unwrap();
        let tower = model.artifact("vision").unwrap();
        assert!(tower.runtime.supports_backend("metal"));
        assert!(tower.workspace.unwrap() > 3_900_000_000);
        assert!(mlx.runtime.checkpoint_dir);
        assert!(!mlx.runtime.supports_backend("cuda"));
        assert_eq!(mlx.files.len(), files);
        let source = mlx.source.as_ref().unwrap();
        assert_eq!(source.repo, format!("mlx-community/{id}-8bit"));
        assert_eq!(source.revision, revision);
        assert_eq!(source.base_model, format!("Cloudflare/{id}"));
        for file in &mlx.files {
            assert_eq!(
                file.url,
                format!(
                    "https://models.truespar.io/models/mlx-community/{revision}/{}",
                    file.dest
                )
            );
            assert!(file.size > 0 && file.sha256.len() == 64);
            assert!(!file.dest.ends_with(".py"));
        }
        for name in [
            "LICENSE",
            "README.md",
            "joint_head.safetensors",
            "config.json",
        ] {
            assert!(
                mlx.files
                    .iter()
                    .any(|f| f.dest.ends_with(&format!("/{name}")))
            );
        }
    }
}

#[test]
fn kumo_serves_every_checkpoint_on_both_backends_as_a_single_pass_table_model() {
    for backend in ["cuda", "metal"] {
        let reg = Registry::new("./models".into()).with_backend(backend);
        let model = reg.catalog_of("kumo-tabular").unwrap();
        assert_eq!(
            crate::estimate::kind_for(&model.capability),
            paddock_estimator::ModelKind::Encoder
        );
        assert_eq!(
            model.default_weights_for_backend(backend, None).unwrap().id,
            "large-classification"
        );
        for size in ["small", "medium", "large"] {
            for task in ["classification", "regression"] {
                let a = model.artifact(&format!("{size}-{task}")).unwrap();
                assert!(a.runtime.supports_backend(backend));
                assert_eq!(a.capabilities(model), ["tabular"]);
                assert!(a.runtime.checkpoint_dir);
                // the runner is handed the export directory, whose name is
                // the served model id
                assert!(
                    a.entry_path(reg.models_dir())
                        .unwrap()
                        .ends_with(format!("kumo-tabular-{size}-{task}"))
                );
                assert!(a.files[0].dest.ends_with("model.safetensors"));
                let shape = a.shape.as_ref().unwrap();
                assert!(shape.kv_layers.is_empty() && shape.weight_bytes > 100 << 20);
                assert!(a.workspace.unwrap() > 2 << 30);
            }
        }
    }
}

#[test]
fn diffusion_metal_catalog_exposes_vision_in_three_formats_without_cuda_drift() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let model = metal.catalog_of("diffusiongemma-26b-a4b").unwrap();
    assert_eq!(
        model.default_weights_for_backend("metal", None).unwrap().id,
        "mlx-4bit"
    );
    for id in ["q8", "q4", "mlx-4bit"] {
        let a = model.artifact(id).unwrap();
        assert!(a.runtime.supports_backend("metal"));
        assert_eq!(a.capabilities(model), ["chat", "vision", "reasoning"]);
        assert_eq!(a.runtime.default_spec.as_deref(), Some("off"));
        assert_eq!(metal.default_envelope(&model.id, Some(id)), (32768, 1));
        assert_eq!(a.runtime.memory.as_ref().unwrap().max_batch, 8);
        assert_eq!(a.runtime.embedded_vision, id == "mlx-4bit");
        assert!(!crate::backend_contract::metal_kv_offload(model, a));
        let (_, tower, drafter) = metal.planned_paths(&model.id, Some(id)).unwrap();
        assert_eq!(tower.is_some(), id != "mlx-4bit");
        assert!(drafter.is_none());
        assert_eq!(a.runtime.checkpoint_dir, id == "mlx-4bit");
        assert!(a.files.iter().all(|f| f.size > 0 && f.sha256.len() == 64));
        assert!(
            a.files
                .iter()
                .all(|f| { f.url == format!("https://models.truespar.io/models/{}", f.dest) })
        );
    }
    assert_eq!(
        model
            .artifact("q4")
            .unwrap()
            .source
            .as_ref()
            .unwrap()
            .revision,
        "f4183a2c7a354128d02545752303c4354d165bf0"
    );
    let a = model.artifact("mlx-4bit").unwrap();
    assert_eq!(a.files.len(), 12);
    assert_eq!(
        a.source.as_ref().unwrap().revision,
        "a7a81407613811e8ba63af92ac0d852b809e191f"
    );
    for name in ["processor_config.json", "README.md"] {
        assert!(a.files.iter().any(|f| f.dest.ends_with(name)));
    }
    assert!(
        a.entry_path(metal.models_dir())
            .unwrap()
            .ends_with("diffusiongemma-26B-A4B-it-MLX-4bit")
    );
    // CUDA serves the Q4_K_M too, on the k-quant lane, priced from its own
    // measured shape: the Metal probe above never reaches CUDA, and the CUDA
    // measurement never reaches Metal
    let metal_q4 = model.artifact("q4").unwrap();
    assert_eq!(
        metal_q4.shape.as_ref().unwrap().weight_bytes,
        16_806_810_208
    );
    assert_eq!(metal_q4.workspace, Some(4_294_967_296));
    let cuda = metal.with_backend("cuda");
    let model = cuda.catalog_of("diffusiongemma-26b-a4b").unwrap();
    assert_eq!(model.default_weights().unwrap().id, "q8");
    for id in ["q8", "q4"] {
        assert_eq!(cuda.default_envelope(&model.id, Some(id)), (4096, 32));
        assert!(cuda.planned_paths(&model.id, Some(id)).is_some(), "{id}");
    }
    let q4 = model.artifact("q4").unwrap();
    assert_eq!(q4.shape.as_ref().unwrap().weight_bytes, 19_101_029_376);
    assert_eq!(q4.workspace, Some(6_338_927_036));
    assert!(q4.runtime.kv_cache_dtype.is_none());
    assert!(cuda.planned_paths(&model.id, Some("mlx-4bit")).is_none());
}

#[test]
fn qwen_image_metal_resolves_embedded_mlx_and_gguf_editing_tower() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let model = metal.catalog_of("qwen-image-2.1").unwrap();
    let weights = model.default_weights_for_backend("metal", None).unwrap();
    assert_eq!(weights.id, "mlx-4bit");
    assert_eq!(weights.capabilities(model), ["image-generation"]);
    assert!(weights.runtime.checkpoint_dir && weights.runtime.embedded_vision);
    assert_eq!(weights.runtime.companions.as_deref(), Some([].as_slice()));
    assert_eq!(weights.files.len(), 23);
    for name in [
        "LICENSE",
        "Notice",
        "model_index.json",
        "processor/tokenizer.json",
        "text_encoder/model.safetensors",
        "transformer/model.safetensors",
        "vae/model.safetensors",
    ] {
        assert!(
            weights.files.iter().any(|f| f.dest.ends_with(name)),
            "missing {name}"
        );
    }
    assert!(weights.files.iter().all(
        |f| f.url.contains("4db4e8c0c0e7a1debf0320415bec8388e888494c") && f.sha256.len() == 64
    ));
    assert!(
        weights
            .entry_path(metal.models_dir())
            .unwrap()
            .ends_with("Qwen-Image-2.1-MLX-4bit")
    );
    assert_eq!(metal.planned_lane_companions(&model.id, None), (None, None));
    let (_, vision, draft) = metal.planned_paths(&model.id, None).unwrap();
    assert!(vision.is_none() && draft.is_none());
    let (text, vae) = metal.planned_lane_companions(&model.id, Some("q4"));
    assert!(text.unwrap().ends_with("Qwen3VL-8B-Instruct-Q4_K_M.gguf"));
    assert!(
        vae.unwrap()
            .ends_with("vae/diffusion_pytorch_model.safetensors")
    );
    assert_eq!(metal.default_envelope(&model.id, None).1, 1);
    assert!(
        metal
            .planned_paths(&model.id, Some("q4"))
            .unwrap()
            .1
            .unwrap()
            .ends_with("mmproj-Qwen3VL-8B-Instruct-F16.gguf")
    );
    assert!(
        !model
            .artifact("q8")
            .unwrap()
            .runtime
            .supports_backend("metal")
    );
    let cuda = metal.with_backend("cuda");
    assert_eq!(
        cuda.catalog_of("qwen-image-2.1")
            .unwrap()
            .default_weights()
            .unwrap()
            .id,
        "q8"
    );
    assert!(
        cuda.planned_paths("qwen-image-2.1", Some("q4"))
            .unwrap()
            .1
            .is_some()
    );
}

#[test]
fn minicpm_metal_gguf_and_official_mlx_have_complete_local_contracts() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let model = metal.catalog_of("minicpm5-2b").unwrap();
    assert_eq!(
        model.default_weights_for_backend("metal", None).unwrap().id,
        "q8"
    );
    for id in ["q8", "q4", "mlx-4bit"] {
        let a = model.artifact(id).unwrap();
        assert!(a.runtime.supports_backend("metal"));
        assert_eq!(a.runtime.checkpoint_dir, id == "mlx-4bit");
        assert_eq!(metal.default_envelope("minicpm5-2b", Some(id)), (32768, 1));
        let (path, vision, draft) = metal.planned_paths("minicpm5-2b", Some(id)).unwrap();
        assert!(vision.is_none() && draft.is_none());
        assert_eq!(a.capabilities(model), ["chat", "tools", "reasoning"]);
        assert!(crate::backend_contract::metal_kv_offload(model, a));
        if id == "mlx-4bit" {
            assert!(path.ends_with("MiniCPM5-2B-MLX"));
            assert_eq!(a.source.as_ref().unwrap().repo, "openbmb/MiniCPM5-2B-MLX");
            assert_eq!(a.files.len(), 7);
            for f in &a.files {
                assert!(f.url.contains("/8a9ad7539ac86281d0ac2b017ba04a5de53fe9a3/"));
                assert_eq!(f.sha256.len(), 64);
                assert!(f.size > 0);
            }
        } else {
            assert_eq!(path.extension().unwrap(), "gguf");
        }
    }
    let cuda = metal.with_backend("cuda");
    assert!(
        cuda.planned_paths("minicpm5-2b", Some("mlx-4bit"))
            .is_none()
    );
    assert_eq!(cuda.default_envelope("minicpm5-2b", Some("q8")), (4096, 32));
}

#[test]
fn metal_chat_defaults_are_32k_with_export_caps_and_no_cuda_or_speech_drift() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    for (model, artifact, context) in [
        ("qwen3.8-27b", "mlx-4bit", 32768),
        ("qwen3.8-27b", "q4", 32768),
        ("qwen3.8-27b", "splash-4bit", 32768),
        ("bonsai-2-27b", "mlx-2bit", 32768),
        ("bonsai-2-27b", "ptq1", 32768),
        ("gpt-oss-20b", "mxfp4", 32768),
        ("gemma-4-31b", "mlx-4bit", 4096),
        ("qwen3.8-flash-next", "mlx-4bit", 4096),
        ("kb-whisper-large", "f16", 448),
        ("qwen3-asr-1.7b", "q8", 4096),
        ("qwen3-embedding-0.6b", "q8", 4096),
    ] {
        assert_eq!(metal.default_envelope(model, Some(artifact)).0, context);
    }
    let cuda = metal.with_backend("cuda");
    assert_eq!(
        cuda.default_envelope("bonsai-2-27b", Some("ptq1")),
        (4096, 32)
    );
    // qwen3.8's CUDA target is its own election (backend_overrides.cuda,
    // 2026-10-04): the long agent window, fitted to the card at admission.
    // It must not leak into Metal, which is what the next check holds.
    assert_eq!(
        cuda.default_envelope("qwen3.8-27b", Some("q4")),
        (262144, 4)
    );
    let metal = cuda.with_backend("metal");
    assert_eq!(
        metal.default_envelope("qwen3.8-27b", Some("mlx-4bit")),
        (32768, 1)
    );
}

#[test]
fn bonsai_gguf_metal_adds_optional_vision_and_keeps_mlx_default() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let model = metal.catalog_of("bonsai-2-27b").unwrap();
    assert_eq!(
        model.default_weights_for_backend("metal", None).unwrap().id,
        "mlx-2bit"
    );
    let a = model.artifact("ptq1").unwrap();
    assert!(a.runtime.supports_backend("metal"));
    assert_eq!(a.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(a.runtime.companions, Some(vec!["vision".into()]));
    assert_eq!(
        a.runtime.capability,
        Some(vec![
            "chat".into(),
            "vision".into(),
            "tools".into(),
            "reasoning".into()
        ])
    );
    assert!(!a.runtime.checkpoint_dir && !a.runtime.embedded_vision);
    assert!(
        !a.runtime
            .capability
            .as_ref()
            .unwrap()
            .iter()
            .any(|c| c == "speculative")
    );
    let memory = a.runtime.memory.as_ref().unwrap();
    assert_eq!(memory.weight_bytes, 5935527936);
    assert_eq!(memory.max_batch, 4);
    assert_eq!(
        metal.default_envelope("bonsai-2-27b", Some("ptq1")),
        (32768, 1)
    );
    assert!(metal.planned_paths("bonsai-2-27b", Some("ptq1")).is_some());
    assert!(
        model
            .artifact("vision")
            .unwrap()
            .runtime
            .supports_backend("metal")
    );
    assert!(!model.artifact("vision").unwrap().required);
    assert!(
        metal
            .planned_paths("bonsai-2-27b", Some("ptq1"))
            .unwrap()
            .1
            .is_some()
    );
    assert_eq!(
        model
            .default_bundle_for_backend("metal", None)
            .iter()
            .map(|a| a.id.as_str())
            .collect::<Vec<_>>(),
        ["mlx-2bit"]
    );
    let cuda = metal.with_backend("cuda");
    let model = cuda.catalog_of("bonsai-2-27b").unwrap();
    assert!(model.capability.iter().any(|c| c == "vision"));
    assert_eq!(
        model
            .default_bundle_for_backend("cuda", None)
            .iter()
            .map(|a| a.id.as_str())
            .collect::<Vec<_>>(),
        ["ptq1", "vision"]
    );
    assert_eq!(
        model.default_weights_for_backend("cuda", None).unwrap().id,
        "ptq1"
    );
    assert_eq!(
        model
            .artifact("ptq1")
            .unwrap()
            .shape
            .as_ref()
            .unwrap()
            .weight_bytes,
        5982828640
    );
}

#[test]
fn metal_chat_default_respects_small_model_context_and_explicit_export_recommendation() {
    let registry = Registry::new("./models".into());
    let original = registry.catalog_of("qwen3.8-27b").unwrap();
    for (recommended, ceiling, expected) in [
        (None, 8192, 8192),
        (Some(65536), 8192, 8192),
        (Some(16384), 262144, 16384),
    ] {
        let mut model = original.clone();
        model.artifacts.retain(|a| a.id == "q4");
        let artifact = &mut model.artifacts[0];
        artifact.shape.as_mut().unwrap().max_ctx = ceiling;
        artifact.runtime.default_max_ctx = recommended;
        let projected = Registry::from_catalog(
            Catalog {
                schema: 3,
                models: vec![model],
            },
            "./models".into(),
        )
        .with_backend("metal");
        assert_eq!(
            projected.default_envelope("qwen3.8-27b", Some("q4")).0,
            expected
        );
    }
}

#[test]
fn bonsai_and_aligner_shared_basename_never_swap_model_identity() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    for (folder, model, artifact) in [
        ("Ternary-Bonsai-2-27B-mlx-2bit", "bonsai-2-27b", "mlx-2bit"),
        (
            "Qwen3-ForcedAligner-0.6B-hf",
            "qwen3-forced-aligner-0.6b",
            "bf16",
        ),
    ] {
        for path in [
            format!("/other/models/{folder}/model.safetensors"),
            format!(r"E:\models\{folder}\model.safetensors"),
            folder.to_owned(),
        ] {
            assert_eq!(
                reg.identify_weights(Path::new(&path)),
                Some((model.into(), artifact.into())),
                "{path}"
            );
        }
        assert_eq!(
            reg.identity_for(
                Some((model, Some(artifact))),
                Path::new("model.safetensors")
            ),
            Some((model.into(), Some(artifact.into())))
        );
    }
    assert_eq!(reg.identify_weights(Path::new("model.safetensors")), None);
    assert_eq!(
        reg.identify_weights(Path::new("/unknown/model.safetensors")),
        None
    );
}

#[test]
fn whisper_metal_exact_f16_cache_and_workspace_preserve_cuda() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let cuda = Registry::new("./models".into()).with_backend("cuda");
    for (id, bytes) in [
        ("kb-whisper-large", 3227136000),
        ("nb-whisper-large", 3094359040),
        ("roest-v3-whisper-1.5b", 3094359040),
    ] {
        let m = metal.catalog_of(id).unwrap();
        assert!(m.default_weights_for_backend("metal", None).is_some());
        let a = m.artifact("f16").unwrap();
        assert!(a.runtime.supports_backend("metal"));
        assert_eq!(
            a.runtime.qualification,
            Some(runtime::Qualification::Qualified)
        );
        assert_eq!(a.runtime.kv_cache_dtype.as_deref(), Some("f16"));
        assert_eq!(a.runtime.companions, Some(vec![]));
        let shape = a.shape.as_ref().unwrap();
        assert_eq!(shape.weight_bytes, bytes);
        let cross = shape.cross_kv.as_ref().unwrap();
        assert_eq!(
            (cross.layers, cross.frames, cross.k_dim, cross.v_dim),
            (32, 1500, 1280, 1280)
        );
        let mem = a.runtime.memory.as_ref().unwrap();
        assert_eq!((mem.max_ctx, mem.max_batch), (448, 16));
        assert_eq!(mem.kv_reserve_sequences, 0);
        assert!(!mem.full_paged_kv);
        assert_eq!(a.workspace, Some(576716800));
        let original = cuda.catalog_of(id).unwrap();
        let original = original.default_weights_for_backend("cuda", None).unwrap();
        assert_eq!(original.shape.as_ref().unwrap().weight_bytes, 3359912960);
        assert!(original.workspace.is_none());
    }
}

#[test]
fn granite_speech_metal_exact_pairs_preserve_cuda() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    for (id, weight, tower) in [
        ("granite-speech-4.1-2b", 1952592128, 1154200944),
        ("granite-speech-4.1-2b-plus", 1734224000, 1162589552),
    ] {
        let m = metal.catalog_of(id).unwrap();
        assert!(m.default_weights_for_backend("metal", None).is_some());
        let w = m.artifact("q8").unwrap();
        assert!(w.runtime.supports_backend("metal"));
        assert_eq!(
            w.runtime.qualification,
            Some(runtime::Qualification::Qualified)
        );
        assert_eq!(w.runtime.kv_cache_dtype.as_deref(), Some("f16"));
        assert_eq!(w.runtime.companions, Some(vec!["audio".to_string()]));
        assert_eq!(w.shape.as_ref().unwrap().weight_bytes, weight);
        assert_eq!(w.workspace, Some(186808384));
        let mem = w.runtime.memory.as_ref().unwrap();
        assert_eq!((mem.max_ctx, mem.max_batch), (4096, 16));
        assert!(mem.full_paged_kv);
        assert_eq!(mem.kv_reserve_sequences, 1);
        let a = m.artifact("audio").unwrap();
        assert!(a.required && a.runtime.supports_backend("metal"));
        assert_eq!(a.runtime.memory.as_ref().unwrap().weight_bytes, tower);
        assert_eq!(a.workspace, Some(1073741824));
        let cuda = Registry::new("./models".into()).with_backend("cuda");
        let m = cuda.catalog_of(id).unwrap();
        let w = m.default_weights_for_backend("cuda", None).unwrap();
        assert_eq!(w.id, "q8");
        assert_eq!(w.shape.as_ref().unwrap().weight_bytes, 1952592128);
        assert!(w.workspace.is_none());
    }
}

#[test]
fn qwen3_aligner_metal_retains_bf16_and_no_generative_kv() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let m = metal.catalog_of("qwen3-forced-aligner-0.6b").unwrap();
    assert!(m.default_weights_for_backend("metal", None).is_some());
    let a = m.artifact("bf16").unwrap();
    assert!(a.runtime.supports_backend("metal"));
    assert_eq!(
        a.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(a.runtime.companions, Some(vec![]));
    assert_eq!(a.workspace, Some(1110904832));
    let s = a.shape.as_ref().unwrap();
    assert_eq!(s.weight_bytes, 1836292160);
    assert_eq!(s.kind, paddock_estimator::ModelKind::Encoder);
    assert!(s.kv_layers.is_empty());
    let cuda = metal.with_backend("cuda");
    let m = cuda.catalog_of("qwen3-forced-aligner-0.6b").unwrap();
    let a = m.default_weights_for_backend("cuda", None).unwrap();
    assert_eq!(a.id, "bf16");
    assert!(a.shape.is_none());
    assert!(a.workspace.is_none());
}

#[test]
fn qwen3_asr_metal_requires_exact_audio_pair_and_preserves_cuda() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let m = metal.catalog_of("qwen3-asr-1.7b").unwrap();
    assert!(m.default_weights_for_backend("metal", None).is_some());
    let w = m.artifact("q8").unwrap();
    assert!(w.runtime.supports_backend("metal"));
    assert_eq!(
        w.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(w.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(
        w.runtime.companions.as_deref(),
        Some(["audio".to_string()].as_slice())
    );
    assert_eq!(w.shape.as_ref().unwrap().weight_bytes, 2159087616);
    assert_eq!(w.workspace, Some(128258112));
    let mem = w.runtime.memory.as_ref().unwrap();
    assert_eq!((mem.max_ctx, mem.max_batch), (8192, 16));
    let a = m.artifact("audio").unwrap();
    assert!(a.required && a.runtime.supports_backend("metal"));
    assert_eq!(a.workspace, Some(536870912));
    let cuda = metal.with_backend("cuda");
    let m = cuda.catalog_of("qwen3-asr-1.7b").unwrap();
    assert_eq!(
        m.default_weights_for_backend("cuda", None).unwrap().id,
        "q8"
    );
    assert_eq!(
        m.artifact("q8")
            .unwrap()
            .shape
            .as_ref()
            .unwrap()
            .weight_bytes,
        2165035104
    );
    assert_eq!(m.artifact("audio").unwrap().workspace, None);
}

#[test]
fn unlimited_ocr_metal_has_an_explicit_required_pair_and_preserves_cuda() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let m = metal.catalog_of("unlimited-ocr").unwrap();
    assert!(m.default_weights_for_backend("metal", None).is_some());
    let w = m.artifact("q8").unwrap();
    assert!(w.runtime.supports_backend("metal"));
    assert_eq!(
        w.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(w.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(
        w.runtime.companions.as_deref(),
        Some(["mmproj".to_string()].as_slice())
    );
    assert_eq!(w.shape.as_ref().unwrap().weight_bytes, 3120896000);
    assert_eq!(w.workspace, Some(136120644));
    let mem = w.runtime.memory.as_ref().unwrap();
    assert_eq!((mem.max_ctx, mem.max_batch), (32768, 16));
    let v = m.artifact("mmproj").unwrap();
    assert!(v.required && v.runtime.supports_backend("metal"));
    assert_eq!(v.workspace, Some(2415919104));
    let cuda = metal.with_backend("cuda");
    let m = cuda.catalog_of("unlimited-ocr").unwrap();
    assert_eq!(
        m.default_weights_for_backend("cuda", None).unwrap().id,
        "q8"
    );
    assert_eq!(
        m.artifact("q8")
            .unwrap()
            .shape
            .as_ref()
            .unwrap()
            .weight_bytes,
        3126134688
    );
    assert_eq!(m.artifact("mmproj").unwrap().workspace, Some(996147200));
}

#[test]
fn paddleocr_metal_is_an_explicit_document_pair_with_native_memory() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    let m = metal.catalog_of("paddleocr-vl-1.6").unwrap();
    assert!(m.default_weights_for_backend("metal", None).is_some());
    let w = m.artifact("bf16").unwrap();
    assert!(w.runtime.supports_backend("metal"));
    assert_eq!(
        w.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(
        w.runtime.capability.as_deref(),
        Some(["chat".to_string(), "documents".to_string()].as_slice())
    );
    assert_eq!(
        w.runtime.companions.as_deref(),
        Some(["mmproj".to_string()].as_slice())
    );
    assert_eq!(w.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    let mem = w.runtime.memory.as_ref().unwrap();
    assert_eq!((mem.max_ctx, mem.max_batch), (32768, 16));
    assert_eq!(w.shape.as_ref().unwrap().weight_bytes, 933384192);
    assert_eq!(w.workspace, Some(103239744));
    let v = m.artifact("mmproj").unwrap();
    assert!(v.required && v.runtime.supports_backend("metal"));
    assert_eq!(v.workspace, Some(2860253184));
    let restored = metal.with_backend("cuda");
    let m = restored.catalog_of("paddleocr-vl-1.6").unwrap();
    assert_eq!(
        m.default_weights_for_backend("cuda", None).unwrap().id,
        "bf16"
    );
    assert_eq!(m.artifact("mmproj").unwrap().workspace, Some(1950351360));
    assert_eq!(
        m.artifact("bf16").unwrap().shape.as_ref().unwrap().max_ctx,
        131072
    );
}

#[test]
fn omitted_backend_metadata_preserves_cuda_without_advertising_metal() {
    for contract in [
        ArtifactRuntime::default(),
        toml::from_str::<ArtifactRuntime>("").unwrap(),
    ] {
        assert!(contract.supports_backend("cuda"));
        assert!(!contract.supports_backend("metal"));
        assert!(!contract.supports_backend("future-backend"));
    }
    let disabled: ArtifactRuntime = toml::from_str("backends = []").unwrap();
    assert!(!disabled.supports_backend("cuda"));
}

#[test]
fn catalog_gates_missing_architectures_and_weight_formats() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    for (model, artifact) in [
        ("qwen3.6-35b-a3b", "q4"),
        ("qwen3.8-27b", "q3"),
        ("qwen3.8-27b", "nvfp4"),
        ("granite-4.2-8b", "nvfp4"),
        ("granite-4.2-30b", "fp8"),
        ("muse-glimmer-30b", "drafter"),
        ("nemotron-3.5-lightning-30b", "nvfp4"),
    ] {
        assert!(
            !reg.catalog_of(model)
                .unwrap()
                .artifact(artifact)
                .unwrap()
                .runtime
                .supports_backend("metal")
        );
        assert!(reg.start_pull(model, Some(&[artifact.into()])).is_err());
    }
}

#[test]
fn flash_next_metal_prices_resident_ple_without_changing_cuda() {
    let cuda = Registry::new("./models".into());
    let original = cuda.catalog_of("qwen3.8-flash-next").unwrap();
    assert_eq!(
        original
            .default_weights_for_backend("cuda", None)
            .unwrap()
            .id,
        "iq3"
    );
    let original_artifact = original.artifact("iq3").unwrap();
    assert!(original_artifact.runtime.memory.is_none());
    assert_eq!(
        original_artifact.shape.as_ref().unwrap().weight_bytes,
        56559859968
    );
    let first_sha = original_artifact.files[0].sha256.clone();
    let metal = cuda.with_backend("metal");
    let model = metal.catalog_of("qwen3.8-flash-next").unwrap();
    assert!(model.default_weights_for_backend("metal", None).is_some());
    assert!(metal.planned_paths(&model.id, None).is_some());
    assert!(!model.default_bundle_for_backend("metal", None).is_empty());
    let a = model.artifact("iq3").unwrap();
    assert!(a.runtime.supports_backend("metal"));
    assert_eq!(
        a.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(a.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(a.runtime.companions.as_deref(), Some([].as_slice()));
    assert!(metal.planned_paths(&model.id, Some("iq3")).is_some());
    assert_eq!(a.files.len(), 3);
    assert_eq!(a.files[0].sha256, first_sha);
    let mem = a.runtime.memory.as_ref().unwrap();
    assert_eq!((mem.max_ctx, mem.max_batch), (4096, 4));
    assert_eq!(a.workspace, Some(200949944));
    let mut shape = a.shape.clone().unwrap().into_model_shape(0, 0);
    mem.apply(&mut shape);
    mem.apply(&mut shape);
    assert_eq!(shape.weight_bytes, 81950799360);
    assert_eq!(shape.nextn_bytes, 0);
    assert_eq!(shape.kv_reserve_sequences, 0);
    let kv = shape.kv_per_sequence(4096, paddock_estimator::KvDtype::F16) * 4;
    let recurrent = 36 * 4 * (786432 + 30720) * 4;
    assert_eq!(
        shape.weight_bytes + kv + recurrent + shape.workspace_bytes,
        83025082040
    );
    // Backend projection must restore CUDA's own class and long context.
    let cuda = metal.with_backend("cuda");
    let a = cuda
        .catalog_of("qwen3.8-flash-next")
        .unwrap()
        .artifact("iq3")
        .unwrap();
    assert!(a.runtime.memory.is_none());
    assert_eq!(a.shape.as_ref().unwrap().max_ctx, 262144);
    assert_eq!(a.shape.as_ref().unwrap().weight_bytes, 56559859968);
}

#[test]
fn metal_gemma_moe_uses_exact_companions() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let model = reg.catalog_of("gemma-4-26b-a4b").unwrap();
    assert!(model.default_weights_for_backend("metal", None).is_some());
    assert!(!model.default_bundle_for_backend("metal", None).is_empty());
    let q8 = model.artifact("q8").unwrap();
    assert_eq!(
        q8.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(q8.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(
        q8.runtime.companions.as_deref(),
        Some(["vision".into(), "drafter".into()].as_slice())
    );
    for id in ["q8", "vision", "drafter"] {
        assert!(
            model
                .artifact(id)
                .unwrap()
                .runtime
                .supports_backend("metal")
        );
    }
    assert!(reg.planned_paths(&model.id, Some("q8")).is_some());
    let cuda = reg.with_backend("cuda");
    let model = cuda.catalog_of("gemma-4-26b-a4b").unwrap();
    let q8 = model.default_weights_for_backend("cuda", None).unwrap();
    assert_eq!(q8.id, "q8");
    assert!(q8.runtime.kv_cache_dtype.is_none());
    assert!(q8.runtime.qualification.is_none());
    assert_eq!(q8.shape.as_ref().unwrap().weight_bytes, 27645605376);
}

#[test]
fn metal_qwen_moe_only_exposes_elected_q8_and_its_own_companion() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let model = reg.catalog_of("qwen3.6-35b-a3b").unwrap();
    assert!(model.default_weights_for_backend("metal", None).is_some());
    assert!(!model.default_bundle_for_backend("metal", None).is_empty());
    let q8 = model.artifact("q8").unwrap();
    assert!(q8.runtime.supports_backend("metal"));
    assert_eq!(
        q8.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(q8.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(
        q8.runtime.companions.as_deref(),
        Some(["vision".into()].as_slice())
    );
    assert!(
        model
            .artifact("vision")
            .unwrap()
            .runtime
            .supports_backend("metal")
    );
    assert!(
        !model
            .artifact("q4")
            .unwrap()
            .runtime
            .supports_backend("metal")
    );
    assert!(reg.planned_paths(&model.id, Some("q8")).is_some());
    let cuda = reg.with_backend("cuda");
    let model = cuda.catalog_of("qwen3.6-35b-a3b").unwrap();
    assert_eq!(
        model.default_weights_for_backend("cuda", None).unwrap().id,
        "q8"
    );
    assert!(
        model
            .artifact("q8")
            .unwrap()
            .runtime
            .kv_cache_dtype
            .is_none()
    );
    assert!(
        model
            .artifact("q8")
            .unwrap()
            .runtime
            .qualification
            .is_none()
    );
    assert_eq!(
        model
            .artifact("q8")
            .unwrap()
            .shape
            .as_ref()
            .unwrap()
            .max_ctx,
        262144
    );
}

#[test]
fn metal_gpt_oss_defaults_preserve_memory_contracts() {
    for id in ["gpt-oss-20b", "gpt-oss-120b"] {
        let cuda = Registry::new("./models".into());
        assert_eq!(
            cuda.catalog_of(id)
                .unwrap()
                .default_weights_for_backend("cuda", None)
                .unwrap()
                .id,
            "mxfp4"
        );
        let metal = cuda.with_backend("metal");
        let model = metal.catalog_of(id).unwrap();
        assert!(model.default_weights_for_backend("metal", None).is_some());
        assert!(metal.planned_paths(id, None).is_some());
        assert!(!model.default_bundle_for_backend("metal", None).is_empty());
        let artifact = model.artifact("mxfp4").unwrap();
        assert!(artifact.runtime.supports_backend("metal"));
        assert_eq!(
            artifact.runtime.qualification,
            Some(runtime::Qualification::Qualified)
        );
        assert_eq!(artifact.runtime.kv_cache_dtype.as_deref(), Some("f16"));
        assert_eq!(artifact.runtime.companions.as_deref(), Some([].as_slice()));
        assert!(metal.planned_paths(id, Some("mxfp4")).is_some());
    }
}

#[test]
fn metal_nemotron_prices_only_native_trunk_and_actual_recurrent_blocks() {
    let cuda = Registry::new("./models".into());
    let original = cuda
        .catalog_of("nemotron-3.5-lightning-30b")
        .unwrap()
        .artifact("q8")
        .unwrap();
    assert_eq!(original.workspace, Some(1899342848));
    assert_eq!(original.shape.as_ref().unwrap().weight_bytes, 34997444864);
    assert_eq!(original.shape.as_ref().unwrap().nextn_bytes, 1419146240);
    assert!(original.runtime.memory.is_none());
    let metal = cuda.with_backend("metal");
    let model = metal.catalog_of("nemotron-3.5-lightning-30b").unwrap();
    assert!(model.default_weights_for_backend("metal", None).is_some());
    assert!(!model.default_bundle_for_backend("metal", None).is_empty());
    assert!(metal.planned_paths(&model.id, Some("q8")).is_some());
    let a = model.artifact("q8").unwrap();
    assert_eq!(
        a.runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(a.runtime.kv_cache_dtype.as_deref(), Some("f16"));
    assert_eq!(a.runtime.companions, Some(Vec::new()));
    assert!(
        !a.runtime
            .capability
            .as_ref()
            .unwrap()
            .iter()
            .any(|c| c == "speculative" || c == "vision")
    );
    assert!(
        !model
            .artifact("nvfp4")
            .unwrap()
            .runtime
            .supports_backend("metal")
    );
    let memory = a.runtime.memory.as_ref().unwrap();
    let published = a.shape.clone().unwrap();
    assert_eq!(published.weight_bytes, 33577599744);
    assert_eq!(published.nextn_bytes, 0);
    assert_eq!(published.max_ctx, 32768);
    assert_eq!(a.workspace, Some(677817604));
    let mut shape = published.into_model_shape(0, 1899342848);
    memory.apply(&mut shape);
    memory.apply(&mut shape);
    assert_eq!(shape.nextn_bytes, 0);
    assert_eq!(shape.workspace_bytes, 677817604);
    let r = shape.recurrent.unwrap();
    assert_eq!(
        (
            r.layers,
            r.state_elems,
            r.conv_elems,
            r.conv_dim,
            r.elem_bytes
        ),
        (23, 524288, 18432, 6144, 4)
    );
    let kv = shape.kv_per_sequence(4096, paddock_estimator::KvDtype::F16) * 5;
    let recurrent = r.layers * (r.state_elems + r.conv_elems) * r.elem_bytes * 5;
    assert_eq!(kv + recurrent, 375480320);
    // Switching an already projected registry must restore CUDA, including
    // its workspace, rather than retaining the Metal override in source data.
    let restored = metal.with_backend("cuda");
    let q8 = restored
        .catalog_of("nemotron-3.5-lightning-30b")
        .unwrap()
        .artifact("q8")
        .unwrap();
    assert_eq!(q8.workspace, Some(1899342848));
    assert!(q8.runtime.memory.is_none());
    assert_eq!(q8.shape.as_ref().unwrap().nextn_bytes, 1419146240);
}

#[test]
fn metal_gpt_oss_memory_is_backend_specific_and_survives_projection() {
    for (id, weights, kv) in [
        ("gpt-oss-20b", 12096558336, 1006632960),
        ("gpt-oss-120b", 63374323968, 1509949440),
    ] {
        let cuda = Registry::new("./models".into());
        let original = cuda.catalog_of(id).unwrap().artifact("mxfp4").unwrap();
        assert!(original.runtime.memory.is_none());
        assert_eq!(original.shape.as_ref().unwrap().max_ctx, 131072);
        assert_eq!(
            original.shape.as_ref().unwrap().kv_layers[0].window,
            Some(128)
        );
        let metal = cuda.with_backend("metal");
        let a = metal.catalog_of(id).unwrap().artifact("mxfp4").unwrap();
        let memory = a.runtime.memory.as_ref().unwrap();
        let published = a.shape.clone().unwrap();
        assert_eq!(published.max_ctx, 32768);
        assert_eq!(published.weight_bytes, weights);
        assert_eq!(memory.max_batch, 64);
        assert!(published.kv_layers.iter().all(|l| l.window.is_none()));
        let mut shape = published.into_model_shape(0, 0);
        memory.apply(&mut shape);
        memory.apply(&mut shape); // Resolution is idempotent, not additive.
        assert_eq!(shape.kv_reserve_sequences, 1);
        assert_eq!(
            a.runtime
                .estimate_kv_dtype(paddock_estimator::KvDtype::Fp8E4m3),
            paddock_estimator::KvDtype::F16
        );
        assert_eq!(
            ArtifactRuntime::default().estimate_kv_dtype(paddock_estimator::KvDtype::Fp8E4m3),
            paddock_estimator::KvDtype::Fp8E4m3
        );
        assert_eq!(
            shape.kv_per_sequence(4096, paddock_estimator::KvDtype::F16) * 5,
            kv
        );
    }
}

#[test]
fn metal_laguna_is_explicit_text_only_with_native_residency() {
    for (id, cuda_weight, metal_weight, kv) in [
        ("laguna-xs-2.1", 21747788800, 20270574592, 3355443200),
        ("laguna-s-2.1", 76927235072, 73391436800, 4026531840),
    ] {
        let cuda = Registry::new("./models".into());
        let original = cuda.catalog_of(id).unwrap();
        assert_eq!(
            original
                .default_weights_for_backend("cuda", None)
                .unwrap()
                .id,
            "q4"
        );
        assert!(
            original
                .artifact("dflash")
                .unwrap()
                .runtime
                .supports_backend("cuda")
        );
        let q4 = original.artifact("q4").unwrap();
        assert!(q4.runtime.memory.is_none());
        assert_eq!(q4.shape.as_ref().unwrap().weight_bytes, cuda_weight);
        if id == "laguna-xs-2.1" {
            assert_eq!(q4.shape.as_ref().unwrap().kv_layers[1].window, Some(512));
        } else {
            assert_eq!(q4.files.len(), 3);
            assert!(q4.files[0].dest.ends_with("00001-of-00003.gguf"));
        }
        let metal = cuda.with_backend("metal");
        let model = metal.catalog_of(id).unwrap();
        assert!(model.default_weights_for_backend("metal", None).is_some());
        assert!(!model.default_bundle_for_backend("metal", None).is_empty());
        assert!(metal.planned_paths(&model.id, Some("q4")).is_some());
        assert!(
            !model
                .artifact("dflash")
                .unwrap()
                .runtime
                .supports_backend("metal")
        );
        assert!(
            metal
                .start_pull(&model.id, Some(&["dflash".into()]))
                .is_err()
        );
        let q4 = model.artifact("q4").unwrap();
        assert_eq!(
            q4.runtime.qualification,
            Some(runtime::Qualification::Qualified)
        );
        assert_eq!(q4.runtime.kv_cache_dtype.as_deref(), Some("f16"));
        assert_eq!(q4.runtime.companions.as_deref(), Some([].as_slice()));
        assert!(
            !q4.runtime
                .capability
                .as_ref()
                .unwrap()
                .iter()
                .any(|s| s == "speculative")
        );
        let memory = q4.runtime.memory.as_ref().unwrap();
        let mut shape = q4.shape.clone().unwrap().into_model_shape(0, 0);
        memory.apply(&mut shape);
        memory.apply(&mut shape);
        assert_eq!(shape.weight_bytes, metal_weight);
        assert_eq!(shape.max_ctx, 32768);
        assert_eq!(memory.max_batch, 64);
        assert_eq!(shape.kv_reserve_sequences, 1);
        assert!(shape.kv_layers.iter().all(|l| l.window.is_none()));
        assert_eq!(
            shape.kv_per_sequence(4096, paddock_estimator::KvDtype::F16) * 5,
            kv
        );
    }
}

#[test]
fn metal_qwen3_retrieval_artifacts_are_available_without_companions() {
    let metal = Registry::new("./models".into()).with_backend("metal");
    for kind in ["embedding", "reranker"] {
        for size in ["0.6", "4", "8"] {
            let id = format!("qwen3-{kind}-{size}b");
            let m = metal.catalog_of(&id).unwrap();
            let w = m.default_weights_for_backend("metal", None).unwrap();
            assert_eq!(w.id, "q8");
            assert_eq!(
                w.runtime.qualification,
                Some(runtime::Qualification::Qualified)
            );
            assert_eq!(w.runtime.kv_cache_dtype.as_deref(), Some("f16"));
            assert_eq!(w.runtime.companions.as_deref(), Some([].as_slice()));
            assert!(!w.runtime.checkpoint_dir);
            assert!(metal.planned_paths(&id, None).is_some());
        }
    }
}

#[test]
fn metal_projection_preserves_cuda_contracts_and_is_reversible() {
    let cuda = Registry::new("./models".into());
    let before = serde_json::to_value(cuda.catalog()).unwrap();
    let metal = cuda.with_backend("metal");
    for model in metal.catalog().models.iter() {
        for a in model
            .weights()
            .filter(|a| a.runtime.supports_backend("metal"))
        {
            assert!(!a.runtime.experimental, "{}/{}", model.id, a.id);
            assert_eq!(
                a.runtime.kv_cache_dtype.as_deref(),
                Some(if model.id == "bonsai-2-27b" && a.id == "mlx-2bit" {
                    "f32"
                } else if a.runtime.checkpoint_dir
                    || model
                        .capability
                        .iter()
                        .any(|c| c == "diarization" || c == "decision")
                {
                    // Single-pass audio/decision graphs have no decode KV.
                    "auto"
                } else {
                    "f16"
                })
            );
            assert!(a.runtime.companions.is_some(), "{}/{}", model.id, a.id);
        }
    }
    let muse = metal.catalog_of("muse-glimmer-30b").unwrap();
    let q8 = muse.artifact("q8").unwrap();
    assert!(!q8.runtime.allows_companion("drafter"));
    assert!(q8.runtime.allows_companion("drafter2"));
    let restored = metal.with_backend("cuda");
    assert_eq!(serde_json::to_value(restored.catalog()).unwrap(), before);
    for model in &restored.catalog().models {
        for artifact in &model.artifacts {
            assert_eq!(
                artifact.runtime.supports_backend("cuda"),
                !matches!(
                    artifact.id.as_str(),
                    "mlx-4bit" | "mlx-2bit" | "mlx8" | "splash-4bit" | "mlx-mixed-4-8bit"
                ),
                "{}/{}",
                model.id,
                artifact.id
            );
        }
    }
}

#[test]
fn qwen36_elects_the_tested_q4_composition_on_metal_only() {
    let cuda = Registry::new("./models".into());
    assert_eq!(
        cuda.catalog_of("qwen3.6-27b")
            .unwrap()
            .default_weights_for_backend("cuda", None)
            .unwrap()
            .id,
        "q8"
    );
    let metal = cuda.with_backend("metal");
    let model = metal.catalog_of("qwen3.6-27b").unwrap();
    let bundle = model.default_bundle_for_backend("metal", None);
    assert_eq!(
        bundle.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
        ["q4", "vision"]
    );
    assert!(model.mtp_in_file);
    assert_eq!(
        bundle[0].runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert_eq!(
        model.artifact("q8").unwrap().runtime.qualification,
        Some(runtime::Qualification::Qualified)
    );
    assert!(!bundle[0].runtime.allows_companion("drafter2"));
}

#[test]
fn qwen35_9b_keeps_q8_as_the_measured_metal_text_vision_mtp_default() {
    for backend in ["cuda", "metal"] {
        let registry = Registry::new("./models".into()).with_backend(backend);
        let model = registry.catalog_of("qwen3.5-9b").unwrap();
        let bundle = model.default_bundle_for_backend(backend, None);
        assert_eq!(
            bundle.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
            ["q8", "vision"]
        );
        assert!(model.mtp_in_file);
        if backend == "metal" {
            assert_eq!(
                bundle[0].runtime.qualification,
                Some(runtime::Qualification::Qualified)
            );
            assert_eq!(
                model.artifact("q4").unwrap().runtime.qualification,
                Some(runtime::Qualification::Qualified)
            );
            assert_eq!(bundle[0].runtime.kv_cache_dtype.as_deref(), Some("f16"));
            assert!(bundle[0].runtime.allows_companion("vision"));
            assert!(!bundle[0].runtime.allows_companion("drafter2"));
        }
    }
}

#[tokio::test]
async fn installed_alternative_cannot_displace_the_declared_default() {
    let dir = tempfile::tempdir().unwrap();
    let mut model = Registry::new(dir.path().into())
        .catalog_of("qwen3.8-27b")
        .unwrap()
        .clone();
    for a in &mut model.artifacts {
        for file in &mut a.files {
            file.size = 4;
        }
    }
    let reg = Registry::from_catalog(
        Catalog {
            schema: 3,
            models: vec![model],
        },
        dir.path().into(),
    )
    .with_backend("metal");
    let model = reg.catalog_of("qwen3.8-27b").unwrap();
    for a in model.weights().filter(|a| a.id == "q8" || a.id == "q4") {
        for file in &a.files {
            let path = dir.path().join(&file.dest);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"test").unwrap();
        }
    }
    assert_eq!(
        model.default_weights_for_backend("metal", None).unwrap().id,
        "q4"
    );
    let resolved = reg
        .resolve(&model.id, None, false, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        resolved.weights,
        model
            .artifact("q4")
            .unwrap()
            .entry_path(dir.path())
            .unwrap()
    );
    let view = reg.catalog_annotated();
    let q8 = view["models"][0]["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["id"] == "q8")
        .unwrap();
    assert_eq!(q8["runtime"]["qualification"], "qualified");
    assert_eq!(q8["runtime"]["kv_cache_dtype"], "f16");
}
