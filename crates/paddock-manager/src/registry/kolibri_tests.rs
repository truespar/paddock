//! The downloadable bundle and both clients must share the native loader contract.
use super::*;
use std::time::Duration;

const MODEL: &str = "kolibri-1";
const ARTIFACT: &str = "mlx-mixed-4-8bit";
const DIRECTORY: &str = "Kolibri-1-MLX-mixed-4-8-bit";
const REVISION: &str = "20a42da14e58c0f9de5af5a50b6766874c259b0c";
// CUDA's two lanes: Hob-forge's Q4_K_M GGUF (+ Aleph Alpha's own LICENSE, which
// that repo does not carry) and primitive-ai's NVFP4 checkpoint directory
const Q4: &str = "q4";
const Q4_REVISION: &str = "47fb91b2420efeaa9c3df1259cd6b0c151a85048";
const LICENSE_REVISION: &str = "35bc4d3be745502227a67247de77d70e691614ee";
const NVFP4: &str = "nvfp4";
const NVFP4_DIRECTORY: &str = "Kolibri-1-NVFP4";
const NVFP4_REVISION: &str = "893bda0c210cf48a3f32ba5d40f4b65d1df82a09";

#[tokio::test]
#[ignore = "reads public R2 URLs; no credentials, model download or GPU load"]
async fn kolibri_public_mirror_accepts_the_real_registry_http_client() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let model = reg.catalog_of(MODEL).unwrap();
    for file in [ARTIFACT, Q4, NVFP4]
        .iter()
        .flat_map(|id| &model.artifact(id).unwrap().files)
    {
        let head = reg
            .client
            .head(&file.url)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
        assert_eq!(
            head.headers()[reqwest::header::CONTENT_LENGTH],
            file.size.to_string(),
            "{}",
            file.dest
        );
        if file.size < 10_000_000 {
            let body = reg
                .client
                .get(&file.url)
                .timeout(Duration::from_secs(60))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .bytes()
                .await
                .unwrap();
            assert_eq!(body.len() as u64, file.size);
            assert_eq!(hex(&Sha256::digest(&body)), file.sha256, "{}", file.dest);
        } else {
            let start = file.size - 65536;
            let range = format!("bytes={start}-{}", file.size - 1);
            let response = reg
                .client
                .get(&file.url)
                .timeout(Duration::from_secs(60))
                .header(reqwest::header::RANGE, range)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::PARTIAL_CONTENT);
            assert_eq!(
                response.headers()[reqwest::header::CONTENT_RANGE],
                format!("bytes {start}-{}/{}", file.size - 1, file.size)
            );
            assert_eq!(response.bytes().await.unwrap().len(), 65536);
        }
    }
}

#[test]
fn kolibri_pins_all_shards_tokenizer_template_and_license_on_r2() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let model = reg.catalog_of(MODEL).unwrap();
    let a = model.default_weights_for_backend("metal", None).unwrap();
    assert_eq!(model.vendor.as_deref(), Some("Aleph Alpha"));
    assert_eq!(a.id, ARTIFACT);
    assert_eq!(a.quant.as_deref(), Some("MLX-AFFINE-4/8-G64"));
    assert_eq!(a.files.len(), 16);
    assert!(
        a.files[0]
            .dest
            .ends_with("/model-00001-of-00009.safetensors")
    );
    assert_eq!(a.total_size(), 45_292_398_023);
    let source = a.source.as_ref().unwrap();
    assert_eq!(source.repo, "eins78/Kolibri-1-mlx-mixed-4-8-bit");
    assert_eq!(source.base_model, "Aleph-Alpha/Kolibri-1");
    assert_eq!(source.revision, REVISION);
    assert_eq!(source.license, "apache-2.0");
    assert!(source.license_url.contains(REVISION));
    let names: std::collections::BTreeSet<_> = a
        .files
        .iter()
        .map(|f| {
            assert_eq!(
                f.url,
                format!(
                    "https://models.truespar.io/models/eins78/{REVISION}/{}",
                    f.dest
                )
            );
            assert_eq!(f.sha256.len(), 64);
            assert!(f.size > 0 && f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
            f.dest.strip_prefix(&format!("{DIRECTORY}/")).unwrap()
        })
        .collect();
    assert_eq!(names.len(), 16);
    for shard in 1..=9 {
        assert!(names.contains(format!("model-{shard:05}-of-00009.safetensors").as_str()));
    }
    for name in [
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        assert!(names.contains(name), "{name}");
    }
}

#[test]
fn kolibri_launch_and_memory_contract_do_not_inherit_cuda_or_sliding_ring_defaults() {
    let reg = Registry::new("./models".into()).with_backend("metal");
    let model = reg.catalog_of(MODEL).unwrap();
    let a = model.artifact(ARTIFACT).unwrap();
    assert_eq!(a.runtime.backends, ["metal"]);
    assert_eq!(a.runtime.default_envelope(), (32768, 1));
    assert_eq!(a.runtime.default_spec.as_deref(), Some("off"));
    assert!(a.runtime.checkpoint_dir && !a.runtime.embedded_vision);
    assert_eq!(a.runtime.companions, Some(vec![]));
    assert_eq!(a.capabilities(model), ["chat", "reasoning", "tools"]);
    assert!(!crate::backend_contract::metal_kv_offload(model, a));
    assert_eq!(
        a.runtime
            .estimate_kv_dtype(paddock_estimator::KvDtype::Fp8E4m3),
        paddock_estimator::KvDtype::F16
    );
    let memory = a.runtime.memory.as_ref().unwrap();
    assert_eq!(memory.weight_bytes, 45_381_770_240);
    assert!(memory.full_paged_kv);
    assert_eq!(memory.kv_reserve_sequences, 1);
    assert_eq!(memory.workspace_bytes, Some(268_435_456));
    let shape = a.shape.as_ref().unwrap();
    assert_eq!(shape.max_ctx, 262144);
    assert_eq!(shape.kv_layers.iter().map(|l| l.count).sum::<u64>(), 50);
    assert!(
        shape
            .kv_layers
            .iter()
            .all(|l| l.k_dim == 512 && l.v_dim == 512 && l.window.is_none())
    );
    let mut resident = shape.clone().into_model_shape(0, a.workspace.unwrap());
    memory.apply(&mut resident);
    // Match the real 32K/c=1 smoke allocation, including the reserve slot;
    // pricing only the ten full-attention layers would undercount by 5x.
    let kv = resident.kv_per_sequence(32768, paddock_estimator::KvDtype::F16)
        * (1 + resident.kv_reserve_sequences);
    assert_eq!(kv, 6_710_886_400);
    assert!(resident.weight_bytes + kv + resident.workspace_bytes >= 52_245_375_240);
    assert_eq!(
        a.entry_path(reg.models_dir()).unwrap(),
        reg.models_dir().join(DIRECTORY)
    );
    // CUDA elects its own weights (the GGUF), and the MLX directory is never
    // pullable there
    let gguf = reg
        .models_dir()
        .join("Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf");
    let cuda = reg.with_backend("cuda");
    assert_eq!(cuda.planned_paths(MODEL, None).unwrap().0, gguf);
    assert!(cuda.start_pull(MODEL, Some(&[ARTIFACT.into()])).is_err());
}

#[test]
fn kolibri_cuda_defaults_to_the_gguf_on_every_card_and_offers_nvfp4_on_blackwell() {
    let reg = Registry::new("./models".into()).with_backend("cuda");
    let model = reg.catalog_of(MODEL).unwrap();
    // The GGUF is the default on Blackwell too: on the Spark it decodes ~1.9x
    // faster than NVFP4, whose non-expert planes are all BF16
    for cc in [None, Some([8, 6]), Some([12, 1])] {
        assert_eq!(
            model.default_weights_for_backend("cuda", cc).unwrap().id,
            Q4
        );
    }
    assert_eq!(
        model.default_weights_for_backend("metal", None).unwrap().id,
        ARTIFACT
    );
    let q4 = model.artifact(Q4).unwrap();
    let nv = model.artifact(NVFP4).unwrap();
    assert!(q4.default && !nv.default);
    assert!(q4.min_cc.is_none());
    assert_eq!(nv.min_cc, Some([12, 0]));
    assert!(nv.fits_cc(Some([12, 1])) && !nv.fits_cc(Some([8, 9])) && !nv.fits_cc(None));
    for a in [q4, nv] {
        assert_eq!(a.runtime.backends, ["cuda"]);
        assert_eq!(a.runtime.default_envelope(), (262144, 4));
        assert_eq!(a.runtime.default_spec.as_deref(), Some("off"));
        assert_eq!(a.runtime.companions, Some(vec![]));
        assert!(a.runtime.memory.is_none() && a.runtime.backend_overrides.is_empty());
        assert_eq!(a.capabilities(model), ["chat", "reasoning", "tools"]);
        // KV8, the family default; only the Metal artifact pins BF16
        assert_eq!(
            a.runtime
                .estimate_kv_dtype(paddock_estimator::KvDtype::Fp8E4m3),
            paddock_estimator::KvDtype::Fp8E4m3
        );
        assert!(a.workspace.is_some_and(|w| w > 0));
        let shape = a.shape.as_ref().unwrap();
        assert_eq!(shape.max_ctx, 262144);
        assert_eq!(shape.vocab, 128000);
        assert!(shape.weight_bytes > a.total_size() - (1 << 30));
        // CUDA keeps the 40 sliding layers in rings, priced at the window
        let windows: Vec<_> = shape
            .kv_layers
            .iter()
            .map(|l| (l.window, l.count))
            .collect();
        assert_eq!(windows, [(Some(513), 40), (None, 10)]);
        assert!(
            shape
                .kv_layers
                .iter()
                .all(|l| l.k_dim == 512 && l.v_dim == 512)
        );
        let source = a.source.as_ref().unwrap();
        assert_eq!(source.license, "apache-2.0");
        assert!(source.base_model.starts_with("Aleph-Alpha/Kolibri-1"));
        for f in &a.files {
            assert!(
                f.url.starts_with("https://models.truespar.io/models/"),
                "{}",
                f.url
            );
            assert!(f.url.ends_with(&format!("/{}", f.dest)), "{}", f.url);
            assert_eq!(f.sha256.len(), 64);
            assert!(f.size > 0 && f.sha256.bytes().all(|b| b.is_ascii_hexdigit()));
        }
    }

    assert_eq!(q4.format, "gguf");
    assert_eq!(q4.quant.as_deref(), Some("Q4_K_M"));
    assert!(!q4.runtime.checkpoint_dir);
    let source = q4.source.as_ref().unwrap();
    assert_eq!(source.repo, "Hob-forge/Kolibri-1-GGUF");
    assert_eq!(source.revision, Q4_REVISION);
    let dests: Vec<_> = q4.files.iter().map(|f| f.dest.as_str()).collect();
    assert_eq!(
        dests,
        [
            "Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf",
            "Kolibri-1-GGUF/README.md",
            "Kolibri-1-GGUF/LICENSE"
        ]
    );
    assert_eq!(
        q4.files[0].sha256,
        "c2ac1301424441ef210b6de50ce25e8ccf69f86494df53d6ba52ed558456062e"
    );
    assert_eq!(q4.files[0].size, 47_454_113_472);
    assert!(
        q4.files[0]
            .url
            .contains(&format!("/Hob-forge/{Q4_REVISION}/"))
    );
    assert!(
        q4.files[2]
            .url
            .contains(&format!("/Aleph-Alpha/{LICENSE_REVISION}/"))
    );
    assert_eq!(
        q4.entry_path(reg.models_dir()).unwrap(),
        reg.models_dir()
            .join("Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf")
    );

    assert_eq!(nv.format, "safetensors");
    assert_eq!(nv.quant.as_deref(), Some("NVFP4"));
    assert!(nv.runtime.checkpoint_dir);
    let source = nv.source.as_ref().unwrap();
    assert_eq!(source.repo, "primitive-ai/Kolibri-1-NVFP4");
    assert_eq!(source.revision, NVFP4_REVISION);
    assert!(source.license_url.contains(NVFP4_REVISION));
    let names: std::collections::BTreeSet<_> = nv
        .files
        .iter()
        .map(|f| {
            assert!(f.url.contains(&format!("/primitive-ai/{NVFP4_REVISION}/")));
            f.dest.strip_prefix(&format!("{NVFP4_DIRECTORY}/")).unwrap()
        })
        .collect();
    assert_eq!(names.len(), 59);
    for layer in 0..50 {
        assert!(names.contains(format!("experts-layer{layer:02}.safetensors").as_str()));
    }
    for name in [
        "carry-bf16-000.safetensors",
        "carry-bf16-001.safetensors",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ] {
        assert!(names.contains(name), "{name}");
    }
    assert_eq!(nv.total_size(), 47_737_936_884);
    assert_eq!(
        nv.entry_path(reg.models_dir()).unwrap(),
        reg.models_dir().join(NVFP4_DIRECTORY)
    );
}

#[tokio::test]
async fn kolibri_installed_directory_is_reused_without_any_network_download() {
    let dir = tempfile::tempdir().unwrap();
    let mut model = Registry::new(dir.path().into())
        .catalog_of(MODEL)
        .unwrap()
        .clone();
    for f in &mut model.artifacts[0].files {
        f.size = 4;
        f.sha256 = hex(&Sha256::digest(b"test"));
        f.url = "http://127.0.0.1:1/no-download-allowed".into();
        let file = dir.path().join(&f.dest);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(file, b"test").unwrap();
    }
    let reg = Registry::from_catalog(
        Catalog {
            schema: 3,
            models: vec![model],
        },
        dir.path().into(),
    )
    .with_backend("metal");
    let resolved = reg.resolve(MODEL, None, true, None).await.unwrap().unwrap();
    assert_eq!(resolved.weights, dir.path().join(DIRECTORY));
    assert!(resolved.mmproj.is_none() && resolved.mtp.is_none() && resolved.fp8_snapshot.is_none());
    assert!(!resolved.speculative && !resolved.drafter_declared);
    assert_eq!(
        reg.identify_weights(&resolved.weights),
        Some((MODEL.into(), ARTIFACT.into()))
    );
    let json = reg.catalog_annotated();
    let row = &json["models"][0];
    assert_eq!(row["installed"], true);
    assert_eq!(row["vendor"], "Aleph Alpha");
    assert_eq!(row["specs"]["published_at"], "2026-10-03");
    assert_eq!(row["artifacts"][0]["backend_supported"], true);
    assert_eq!(row["artifacts"][0]["kv_offload_supported"], false);
}
