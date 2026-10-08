//! EmbeddingGemma 2: unsloth's text GGUF, and our byte-for-byte cut of its
//! projector into a picture tower and an audio tower, each its own download.
use super::*;
use std::time::Duration;

const MODEL: &str = "embeddinggemma-2";
const REVISION: &str = "ba3888272494be64ed88c9eb536ddc61a1be73d5";
const PICTURES: u64 = 368_268_992 + 169_869_312;
const AUDIO: u64 = 613_806_752 + 33_554_432;

#[test]
fn embeddinggemma2_mlx_bundle_is_pinned_metal_only_and_keeps_gguf_default() {
    for backend in ["metal", "cuda"] {
        let registry = Registry::new("./models".into()).with_backend(backend);
        let model = registry.catalog_of(MODEL).unwrap();
        assert_eq!(
            model.default_weights_for_backend(backend, None).unwrap().id,
            "q8"
        );
        let mlx = model.artifact("mlx8").unwrap();
        assert!(!mlx.default && mlx.runtime.checkpoint_dir && !mlx.runtime.embedded_vision);
        assert_eq!(mlx.runtime.backends, ["metal"]);
        assert_eq!(mlx.runtime.companions.as_deref(), Some([].as_slice()));
        assert_eq!(mlx.runtime.kv_cache_dtype.as_deref(), Some("auto"));
        assert_eq!(mlx.runtime.default_max_ctx, Some(8192));
        assert_eq!(mlx.runtime.default_max_batch, Some(1));
        assert_eq!(
            mlx.source.as_ref().unwrap().repo,
            "mlx-community/embeddinggemma-2-8bit"
        );
        assert_eq!(mlx.files.len(), 17);
        assert_eq!(mlx.total_size(), 1_266_595_204);
        assert_eq!(
            mlx.files[0].sha256,
            "6b7e97f9687ad422ab3625a1cdf6002892187d7dd04d01b2a09253fa1e5d4cfd"
        );
        for name in [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "processor_config.json",
            "preprocessor_config.json",
            "README.md",
            "LICENSE",
            "PADDOCK-PROVENANCE.txt",
            "1_Pooling/config.json",
            "2_Normalize/config.json",
        ] {
            assert!(
                mlx.files
                    .iter()
                    .any(|f| f.dest == format!("EmbeddingGemma-2-MLX-8bit/{name}"))
            );
        }
        for file in &mlx.files {
            assert_eq!(
                file.url,
                format!(
                    "https://models.truespar.io/models/mlx-community/7505ef2f8ddef45efef6d060865f27989b3c9cec/{}",
                    file.dest
                )
            );
            assert!(file.size > 0 && file.sha256.len() == 64);
        }
        let paths = registry.planned_paths(MODEL, Some("mlx8"));
        if backend == "cuda" {
            assert!(paths.is_none());
            continue;
        }
        let (path, vision, drafter) = paths.unwrap();
        assert_eq!(path, PathBuf::from("./models/EmbeddingGemma-2-MLX-8bit"));
        assert!(vision.is_none() && drafter.is_none());
        assert!(registry.audio_tower(MODEL, Some("mlx8")).is_none());
        let towers = mlx.runtime.optional_towers.as_ref().unwrap();
        assert!(towers.vision.as_ref().unwrap().default && !towers.audio.as_ref().unwrap().default);
        let price = |image, audio| {
            crate::estimate::towers_bytes_for(model, &registry, Some(mlx), image, audio)
        };
        assert_eq!(price(false, Some(false)), 0);
        assert_eq!(price(true, None), 367_075_328 + 169_869_312);
        assert_eq!(price(false, Some(true)), 588_618_624 + 33_554_432);
        assert_eq!(
            price(true, Some(true)),
            367_075_328 + 169_869_312 + 588_618_624 + 33_554_432
        );
    }
}

#[tokio::test]
async fn embeddinggemma2_mlx_resolves_without_gguf_companions_installed() {
    let dir = tempfile::tempdir().unwrap();
    let mut model = Registry::new(dir.path().into())
        .catalog_of(MODEL)
        .unwrap()
        .clone();
    let mlx = model.artifacts.iter_mut().find(|a| a.id == "mlx8").unwrap();
    for file in &mut mlx.files {
        file.size = 1;
        let path = dir.path().join(&file.dest);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }
    let registry = Registry::from_catalog(
        Catalog {
            schema: 3,
            models: vec![model],
        },
        dir.path().into(),
    )
    .with_backend("metal");
    let resolved = registry
        .resolve(MODEL, Some("mlx8"), false, None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        resolved.weights,
        dir.path().join("EmbeddingGemma-2-MLX-8bit")
    );
    assert!(resolved.mmproj.is_none() && resolved.mtp.is_none());
}

#[test]
fn embeddinggemma2_pins_the_gguf_and_both_tower_cuts_on_r2() {
    let reg = Registry::new("./models".into()).with_backend("cuda");
    let m = reg.catalog_of(MODEL).unwrap();
    assert_eq!(m.vendor.as_deref(), Some("Google"));
    // an embedding row: its towers are inputs, not image chat or speech to text
    assert_eq!(m.capability, ["embeddings"]);
    assert!(m.split_towers());
    let q8 = m.default_weights_for_backend("cuda", None).unwrap();
    assert_eq!((q8.id.as_str(), q8.quant.as_deref()), ("q8", Some("Q8_0")));
    assert_eq!(q8.shape.as_ref().unwrap().max_ctx, 8192);
    let vision = m.artifact("vision").unwrap();
    let audio = m.artifact("audio").unwrap();
    assert_eq!(
        (vision.kind, audio.kind),
        (ArtifactKind::Vision, ArtifactKind::Audio)
    );
    // pictures in the default download, audio opt-in, neither required
    assert!(vision.default && !vision.required);
    assert!(!audio.default && !audio.required);
    assert_eq!(m.split_audio_tower().map(|a| a.id.as_str()), Some("audio"));
    for (a, file, size, sha) in [
        (
            q8,
            "embeddinggemma-2-Q8_0.gguf",
            309_855_520,
            "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc",
        ),
        (
            vision,
            "embeddinggemma-2-mmproj-vision-BF16.gguf",
            368_268_992,
            "0ad80061179c86c867b64fad424d3d81942ce6e27326eeb23e9505d553c23078",
        ),
        (
            audio,
            "embeddinggemma-2-mmproj-audio-BF16.gguf",
            613_806_752,
            "1aec404e75971296c155d9e62a8d0cc1d8efceda54065f7ebb7aacf6a813d0a2",
        ),
    ] {
        let source = a.source.as_ref().unwrap();
        assert_eq!(source.repo, "unsloth/embeddinggemma-2-GGUF");
        assert_eq!(source.revision, REVISION);
        assert_eq!(source.base_model, "google/embeddinggemma-2");
        assert_eq!(source.license, "apache-2.0");
        assert_eq!(a.files.len(), 1, "{}", a.id);
        let f = &a.files[0];
        assert_eq!(f.dest, format!("EmbeddingGemma-2-GGUF/{file}"));
        assert_eq!(
            f.url,
            format!("https://models.truespar.io/models/{}", f.dest)
        );
        assert_eq!((f.size, f.sha256.as_str()), (size, sha));
        assert!(a.runtime.supports_backend("metal"), "{}", a.id);
    }
}

#[test]
fn embeddinggemma2_bundle_pulls_pictures_and_leaves_audio_opt_in() {
    for backend in ["cuda", "metal"] {
        let reg = Registry::new("./models".into()).with_backend(backend);
        let m = reg.catalog_of(MODEL).unwrap();
        let ids: Vec<_> = m
            .default_bundle_for_backend(backend, Some([12, 1]))
            .iter()
            .map(|a| a.id.as_str())
            .collect();
        assert_eq!(ids, ["q8", "vision"]);
    }
}

#[test]
fn embeddinggemma2_prices_each_tower_by_its_own_switch() {
    let reg = Registry::new("./this-dir-does-not-exist".into()).with_backend("cuda");
    let m = reg.catalog_of(MODEL).unwrap();
    let w = m.artifact("q8");
    let price = |vision, audio| crate::estimate::towers_bytes_for(m, &reg, w, vision, audio);
    // audio absent = its catalog default (off), as the supervisor serves it
    assert_eq!(price(true, None), PICTURES);
    assert_eq!(price(true, Some(true)), PICTURES + AUDIO);
    assert_eq!(price(false, Some(true)), AUDIO);
    assert_eq!(price(false, Some(false)), 0);
}

#[test]
fn embeddinggemma2_resolves_the_picture_tower_as_mmproj_and_names_the_audio_tower() {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    let mut model = Registry::new(models.clone())
        .catalog_of(MODEL)
        .unwrap()
        .clone();
    // weights and pictures on disk, audio not downloaded
    for a in model.artifacts.iter_mut().filter(|a| a.id != "audio") {
        for f in &mut a.files {
            f.size = 1;
            let p = models.join(&f.dest);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
    }
    let reg = Registry::from_catalog(
        Catalog {
            schema: 3,
            models: vec![model],
        },
        models.clone(),
    )
    .with_backend("cuda");
    let (_, mmproj, _) = reg.planned_paths(MODEL, None).unwrap();
    assert!(
        mmproj
            .unwrap()
            .ends_with("embeddinggemma-2-mmproj-vision-BF16.gguf")
    );
    let (path, installed, default, _) = reg.audio_tower(MODEL, None).unwrap();
    assert!(path.ends_with("embeddinggemma-2-mmproj-audio-BF16.gguf"));
    assert!(!installed && !default);
}

#[tokio::test]
#[ignore = "reads public R2 URLs; no credentials, model download or GPU load"]
async fn embeddinggemma2_public_mirror_serves_every_file() {
    let reg = Registry::new("./models".into()).with_backend("cuda");
    let m = reg.catalog_of(MODEL).unwrap();
    for f in m.artifacts.iter().flat_map(|a| &a.files) {
        let head = reg
            .client
            .head(&f.url)
            .timeout(Duration::from_secs(60))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
        assert_eq!(
            head.headers()[reqwest::header::CONTENT_LENGTH],
            f.size.to_string(),
            "{}",
            f.dest
        );
    }
}
