use super::*;

/// Speculative capability and its implementation must travel together.
#[test]
fn speculative_capability_matches_a_drafter_or_in_file_heads() {
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    let mut claims_without_means = Vec::new();
    let mut drafter_without_claim = Vec::new();
    for m in &reg.catalog().models {
        let claims = m.capability.iter().any(|c| c == "speculative");
        let drafter = m.artifacts.iter().any(|a| a.kind == ArtifactKind::Drafter);
        if claims && !drafter && !m.mtp_in_file {
            claims_without_means.push(m.id.clone());
        }
        if drafter && !claims {
            drafter_without_claim.push(m.id.clone());
        }
    }
    assert!(
        claims_without_means.is_empty() && drafter_without_claim.is_empty(),
        "claim `speculative` with neither in-file heads nor a drafter: \
         {claims_without_means:?}; catalogue a drafter without claiming \
         `speculative`: {drafter_without_claim:?}"
    );
}

#[test]
fn nvfp4_is_the_default_only_where_it_runs() {
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    for id in ["granite-4.2-8b", "granite-4.2-30b"] {
        let m = reg.catalog().models.iter().find(|m| m.id == id).expect(id);
        // Blackwell (sm_120): the NVFP4 lane is the default.
        assert_eq!(
            m.default_weights_for(Some([12, 0]))
                .and_then(|a| a.quant.as_deref()),
            Some("NVFP4"),
            "{id}: NVFP4 is the default on Blackwell",
        );
        // Ampere (sm_86): NVFP4's min_cc is unmet, so the floorless Q8_0 wins.
        assert_eq!(
            m.default_weights_for(Some([8, 6]))
                .and_then(|a| a.quant.as_deref()),
            Some("Q8_0"),
            "{id}: falls back to Q8_0 off Blackwell",
        );
        // No card / unknown cc: never hand out a gated default.
        assert_eq!(
            m.default_weights_for(None).and_then(|a| a.quant.as_deref()),
            Some("Q8_0"),
            "{id}: falls back to Q8_0 when cc is unknown",
        );
        // The nominal (cc-agnostic) default is still the marked one.
        assert_eq!(
            m.default_weights().and_then(|a| a.quant.as_deref()),
            Some("NVFP4"),
            "{id}: nominal default is the marked NVFP4",
        );
    }
}

/// Nemotron's Blackwell election (decided 2026-10-04): NVFP4 weights plus the
/// DSpark drafter where sm_120 can run both, the in-file-MTP Q8_0 lane with no
/// drafter pulled anywhere else. The drafter's min_cc has to gate the bundle
/// AND the spawn wiring, or an Ampere box downloads 1.3 GB it cannot use.
#[test]
fn nemotron_pairs_nvfp4_with_dspark_on_blackwell_only() {
    let id = "nemotron-3.5-lightning-30b";
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    let m = reg.catalog_of(id).unwrap();
    let ids = |cc: Option<[u32; 2]>| -> Vec<String> {
        m.default_bundle_for_backend("cuda", cc)
            .iter()
            .map(|a| a.id.clone())
            .collect()
    };
    assert_eq!(ids(Some([12, 1])), ["nvfp4", "dspark"]);
    assert_eq!(ids(Some([8, 6])), ["q8"]);
    assert_eq!(ids(None), ["q8"]);
    let plain = |cc: Option<[u32; 2]>| -> Vec<String> {
        m.default_bundle_for(cc)
            .iter()
            .map(|a| a.id.clone())
            .collect()
    };
    assert_eq!(plain(Some([12, 1])), ids(Some([12, 1])));
    assert_eq!(plain(Some([8, 6])), ids(Some([8, 6])));

    let drafter = |cc: Option<[u32; 2]>| {
        Registry::new(std::path::PathBuf::from("./models"))
            .with_cc(cc)
            .planned_paths(id, None)
            .expect("nemotron resolves on cuda")
            .2
    };
    let bw = drafter(Some([12, 1])).expect("Blackwell wires the drafter");
    assert!(bw.to_string_lossy().contains("DSpark"), "{}", bw.display());
    assert_eq!(
        drafter(Some([8, 6])),
        None,
        "Q8_0 drafts with its in-file head"
    );
}
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;

/// Over the real catalog: a model that declares a default mmproj
/// companion must get one back in its composition - whichever sense it
/// serves. Written after Qwen3-ASR shipped unservable: the manager
/// downloaded its speech encoder, resolved Vision only, passed no
/// `--mmproj`, and the runner refused with "pass its audio mmproj" for a
/// file already on disk. Asserting the class rather than that one model
/// is the point - the next sense (video, whatever) fails here first.
#[test]
fn every_default_mmproj_companion_reaches_the_composition() {
    let reg = Registry::new(std::path::PathBuf::from("./this-dir-does-not-exist"));
    let mut checked = 0;
    for m in &reg.catalog().models {
        let Some((_, mmproj, _)) = reg.planned_paths(&m.id, None) else {
            continue;
        };
        let declares = m.artifacts.iter().any(|a| a.kind.is_mmproj() && a.default);
        assert_eq!(
            declares,
            mmproj.is_some(),
            "{}: declares a default mmproj companion = {declares}, composition carries one = {}",
            m.id,
            mmproj.is_some()
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "the embedded catalog resolved no models at all"
    );
}

// A tiny origin that serves a fixed buffer and honours `Range: bytes=a-b`
// (206 + Content-Range), so we exercise the parallel path.
async fn serve(State(data): State<Arc<Vec<u8>>>, headers: HeaderMap) -> impl IntoResponse {
    let total = data.len() as u64;
    if let Some(r) = headers.get(axum::http::header::RANGE) {
        let spec = r.to_str().unwrap_or("").trim_start_matches("bytes=");
        let (a, b) = spec.split_once('-').unwrap_or(("0", ""));
        let start: u64 = a.parse().unwrap_or(0);
        let end: u64 = if b.is_empty() {
            total - 1
        } else {
            b.parse::<u64>().unwrap_or(total - 1).min(total - 1)
        };
        let slice = data[start as usize..=end as usize].to_vec();
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::CONTENT_RANGE,
            format!("bytes {start}-{end}/{total}").parse().unwrap(),
        );
        h.insert(axum::http::header::ACCEPT_RANGES, "bytes".parse().unwrap());
        (StatusCode::PARTIAL_CONTENT, h, slice).into_response()
    } else {
        (StatusCode::OK, data.to_vec()).into_response()
    }
}

pub(super) async fn spawn_origin(data: Vec<u8>) -> String {
    let app = axum::Router::new()
        .route("/f", axum::routing::get(serve))
        .with_state(Arc::new(data));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/f")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_range_download_verifies_and_publishes() {
    // ~40 MiB of a deterministic pattern -> several segments across workers
    let data: Vec<u8> = (0..40 * 1024 * 1024u32)
        .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
        .collect();
    let sha = hex(&Sha256::digest(&data));
    let size = data.len() as u64;
    let url = spawn_origin(data.clone()).await;

    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.gguf");
    let client = reqwest::Client::new();
    let progress = Arc::new(AtomicU64::new(0));

    download_file(&client, &url, &dest, &sha, size, progress.clone(), None)
        .await
        .expect("download");

    assert!(dest.exists(), "final file published");
    assert!(!part_path(&dest).exists(), "part file cleaned up");
    assert_eq!(
        progress.load(Ordering::Relaxed),
        size,
        "progress reached total"
    );
    assert_eq!(std::fs::read(&dest).unwrap(), data, "bytes match exactly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wrong_checksum_is_rejected() {
    let data: Vec<u8> = vec![7u8; 3 * 1024 * 1024];
    let size = data.len() as u64;
    let url = spawn_origin(data).await;
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("bad.gguf");
    let client = reqwest::Client::new();
    let bad_sha = "0".repeat(64);
    let err = download_file(
        &client,
        &url,
        &dest,
        &bad_sha,
        size,
        Arc::new(AtomicU64::new(0)),
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, DlError::Checksum { .. }), "got {err:?}");
    assert!(!dest.exists(), "a bad download is never published");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn missing_origin_file_is_a_clean_not_found() {
    // origin serves only /f; any other path 404s - as R2 would for a file
    // that was force-deleted while still listed in the manifest.
    let base = spawn_origin(vec![1u8; 4096]).await;
    let gone = base.replace("/f", "/deleted-from-r2.gguf");
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("x.gguf");
    let err = download_file(
        &reqwest::Client::new(),
        &gone,
        &dest,
        &"0".repeat(64),
        4096,
        Arc::new(AtomicU64::new(0)),
        None,
    )
    .await
    .unwrap_err();
    assert!(
        matches!(err, DlError::NotFound { .. }),
        "gone file -> NotFound, got {err:?}"
    );
    assert!(!dest.exists(), "nothing published");
    assert!(
        !part_path(&dest).exists(),
        "no .part left behind (failed fast before disk write)"
    );
}

/// Two models, so "the file names a different model" is expressible.
fn two_model_registry() -> Registry {
    let art = |id: &str, kind: ArtifactKind, dest: &str| CatalogArtifact {
        id: id.into(),
        kind,
        format: "gguf".into(),
        runtime: ArtifactRuntime::default(),
        source: None,
        label: "Full quality".into(),
        quant: None,
        default: id == "q8",
        required: false,
        min_cc: None,
        workspace: None,
        shape: None,
        files: vec![CatalogFile {
            url: String::new(),
            dest: dest.into(),
            sha256: String::new(),
            size: 1,
        }],
    };
    let model = |id: &str, artifacts: Vec<CatalogArtifact>| CatalogModel {
        id: id.into(),
        display: id.into(),
        vendor: None,
        family: None,
        mtp_in_file: false,
        capability: vec!["chat".into()],
        revision: None,
        license: None,
        kv_default: None,
        specs: Default::default(),
        artifacts,
    };
    let catalog = Catalog {
        schema: 3,
        models: vec![
            model(
                "tiny",
                vec![
                    art("q8", ArtifactKind::Weights, "tiny-GGUF/Tiny-Q8_0.gguf"),
                    art("q4", ArtifactKind::Weights, "tiny-GGUF/Tiny-Q4_K_M.gguf"),
                    // a vision companion must not identify as weights
                    art("vision", ArtifactKind::Vision, "tiny-GGUF/mmproj-BF16.gguf"),
                ],
            ),
            model(
                "other",
                vec![art("q4", ArtifactKind::Weights, "other-GGUF/Other-Q4.gguf")],
            ),
        ],
    };
    Registry::from_catalog(catalog, PathBuf::from("unused"))
}

#[test]
fn identify_weights_maps_a_path_back_to_catalog_identity() {
    let reg = two_model_registry();
    // case-insensitive, matched by file name regardless of the dir it
    // actually lives in (an election path uses the real install root)
    assert_eq!(
        reg.identify_weights(Path::new(r"E:\models\tiny-GGUF/tiny-q8_0.gguf")),
        Some(("tiny".into(), "q8".into()))
    );
    // a runner's file-derived id has no extension - the stem still matches
    assert_eq!(
        reg.identify_weights(Path::new("Tiny-Q8_0")),
        Some(("tiny".into(), "q8".into()))
    );
    // directory-shaped serving (a safetensors checkpoint dir like the
    // forced aligner) reports the directory name as its id - the dest's
    // parent matches it back to the model
    assert_eq!(
        reg.identify_weights(Path::new("tiny-gguf")),
        Some(("tiny".into(), "q8".into()))
    );
    assert_eq!(
        reg.identify_weights(Path::new(r"E:\models\tiny-GGUF/mmproj-BF16.gguf")),
        None
    );
    assert_eq!(reg.identify_weights(Path::new("something-else.gguf")), None);
}

/// The four ways a config file's `[catalog]` block and its `model` path can
/// stand to each other. The fourth is the one that motivates
/// the block at all: a file the catalog cannot recognize by name.
#[test]
fn identity_for_reconciles_the_declaration_with_the_weights() {
    let reg = two_model_registry();
    // Mixed separators deliberately, exactly as the sibling test above does:
    // `\` is not a path separator on unix, so an all-backslash literal makes
    // `file_name()` return the whole string and `identify_weights` answer
    // None for a path that is perfectly recognizable on Windows. That is
    // not a harmless platform quirk here - with `by_file` forced to None,
    // most assertions below stop testing reconciliation at all and pass
    // through the "declaration stands" arm instead. Only the `retired`
    // case, where the declaration is discarded, ever noticed.
    let q8 = Path::new(r"E:\models\tiny-GGUF/Tiny-Q8_0.gguf");

    // agree -> the declaration names the model, the file names the artifact
    assert_eq!(
        reg.identity_for(Some(("tiny", Some("q8"))), q8),
        Some(("tiny".into(), Some("q8".into())))
    );
    // same model, other quant: repointing `model` does not require editing
    // the block, and the artifact follows the bytes
    assert_eq!(
        reg.identity_for(Some(("tiny", Some("q8"))), Path::new("Tiny-Q4_K_M.gguf")),
        Some(("tiny".into(), Some("q4".into())))
    );
    // the file is a different model - somebody repointed `model` and left
    // the block behind. Serving `other` while claiming `tiny` is the one
    // outcome worth overruling the declaration for.
    assert_eq!(
        reg.identity_for(Some(("tiny", Some("q8"))), Path::new("Other-Q4.gguf")),
        Some(("other".into(), Some("q4".into())))
    );
    // The case identify_weights can never serve: renamed, copied, imported
    // in place. The declaration stands - an endpoint does not lose its
    // identity because someone reorganized their disk.
    assert_eq!(
        reg.identity_for(
            Some(("tiny", Some("q4"))),
            Path::new(r"D:\keep\my-copy.gguf")
        ),
        Some(("tiny".into(), Some("q4".into())))
    );
    // an id the catalog has never heard of is discarded rather than passed
    // through: every consumer assumes catalog id or path, and a third kind
    // would offer the user a selection they cannot select
    assert_eq!(
        reg.identity_for(Some(("retired", None)), Path::new("my-copy.gguf")),
        None
    );
    assert_eq!(
        reg.identity_for(Some(("retired", None)), q8),
        Some(("tiny".into(), Some("q8".into())))
    );
    // no block at all = every config file written before then, and the
    // answer is exactly what those files got then
    assert_eq!(
        reg.identity_for(None, q8),
        Some(("tiny".into(), Some("q8".into())))
    );
    assert_eq!(reg.identity_for(None, Path::new("my-copy.gguf")), None);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn registry_pull_from_manifest_downloads_to_dest() {
    let data: Vec<u8> = (0..5 * 1024 * 1024u32).map(|i| (i >> 3) as u8).collect();
    let sha = hex(&Sha256::digest(&data));
    let size = data.len() as u64;
    let url = spawn_origin(data.clone()).await; // serves the bytes at /f, honours Range

    // a one-model manifest pointing at the local origin (as models.toml would
    // at real R2), with an explicit dest - no remote catalog is ever fetched.
    let catalog = Catalog {
        schema: 3,
        models: vec![CatalogModel {
            id: "tiny".into(),
            display: "Tiny".into(),
            vendor: None,
            family: None,
            mtp_in_file: false,
            capability: vec!["chat".into()],
            revision: None,
            license: Some("apache-2.0".into()),
            kv_default: None,
            specs: Default::default(),
            artifacts: vec![CatalogArtifact {
                id: "q8".into(),
                kind: ArtifactKind::Weights,
                format: "gguf".into(),
                runtime: ArtifactRuntime::default(),
                source: None,
                label: "Full quality".into(),
                quant: Some("Q8_0".into()),
                default: true,
                required: false,
                min_cc: None,
                workspace: None,
                shape: None,
                files: vec![CatalogFile {
                    url: url.clone(),
                    dest: "tiny-GGUF/tiny.gguf".into(),
                    sha256: sha.clone(),
                    size,
                }],
            }],
        }],
    };
    let dir = tempfile::tempdir().unwrap();
    let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());
    assert_eq!(reg.catalog().models[0].id, "tiny");
    assert!(
        !reg.is_installed(&reg.catalog().models[0]),
        "not installed before pull"
    );

    let jid = reg.start_pull("tiny", None).expect("start pull");
    // poll to completion. 30 s, not the 4.5 it used to be: this is a hang
    // guard, and under a loaded suite with temp on a slow disk a tiny local
    // download can take longer than an idle machine would suggest.
    let mut done = false;
    for _ in 0..2000 {
        let st = reg
            .job(&jid)
            .unwrap()
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        match st {
            PullStatus::Done => {
                done = true;
                break;
            }
            PullStatus::Error { message } => panic!("pull failed: {message}"),
            PullStatus::Cancelled => panic!("nothing cancelled this pull"),
            PullStatus::Running => {}
        }
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    }
    assert!(done, "pull reached Done");
    let job = reg.job(&jid).unwrap();
    assert_eq!(
        job.downloaded.load(Ordering::Relaxed),
        size,
        "progress = total"
    );
    let landed = dir.path().join("tiny-GGUF").join("tiny.gguf");
    assert!(landed.exists(), "model file landed at its dest");
    assert_eq!(std::fs::read(&landed).unwrap(), data, "bytes verified");
    assert!(
        reg.is_installed(&reg.catalog().models[0]),
        "installed after pull"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resolve_pulls_missing_files_and_splits_weights_from_mmproj() {
    let data: Vec<u8> = (0..2 * 1024 * 1024u32).map(|i| (i >> 2) as u8).collect();
    let sha = hex(&Sha256::digest(&data));
    let size = data.len() as u64;
    let url = spawn_origin(data.clone()).await;

    // a weights + vision model (schema 3 artifacts) - both files point at
    // the local origin.
    let catalog = Catalog {
        schema: 3,
        models: vec![CatalogModel {
            id: "vis".into(),
            display: "Vis".into(),
            vendor: None,
            family: None,
            mtp_in_file: false,
            capability: vec!["chat".into(), "vision".into()],
            revision: None,
            license: None,
            kv_default: None,
            specs: Default::default(),
            artifacts: vec![
                CatalogArtifact {
                    id: "q8".into(),
                    kind: ArtifactKind::Weights,
                    format: "gguf".into(),
                    runtime: ArtifactRuntime::default(),
                    source: None,
                    label: "Full quality".into(),
                    quant: Some("Q8_0".into()),
                    default: true,
                    required: false,
                    min_cc: None,
                    workspace: None,
                    shape: None,
                    files: vec![CatalogFile {
                        url: url.clone(),
                        dest: "Vis-GGUF/vis-Q8_0.gguf".into(),
                        sha256: sha.clone(),
                        size,
                    }],
                },
                CatalogArtifact {
                    id: "vision".into(),
                    kind: ArtifactKind::Vision,
                    format: "gguf".into(),
                    runtime: ArtifactRuntime::default(),
                    source: None,
                    label: "Vision".into(),
                    quant: None,
                    default: true,
                    required: false,
                    min_cc: None,
                    workspace: None,
                    shape: None,
                    files: vec![CatalogFile {
                        url: url.clone(),
                        dest: "Vis-GGUF/vis-mmproj-F16.gguf".into(),
                        sha256: sha.clone(),
                        size,
                    }],
                },
            ],
        }],
    };
    let dir = tempfile::tempdir().unwrap();
    let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());

    // an unknown name resolves to None -> caller treats it as a path
    assert!(
        reg.resolve("not-a-model", None, true, None)
            .await
            .unwrap()
            .is_none()
    );

    // the deploy contract: pull=false on a not-installed model is an
    // honest error naming the fix, never a silent download
    let err = reg.resolve("vis", None, false, None).await.unwrap_err();
    assert!(
        err.to_string().contains("not downloaded"),
        "honest no-pull error: {err}"
    );

    // pull=true fetches the composition and returns split paths
    let r = reg
        .resolve("vis", None, true, None)
        .await
        .unwrap()
        .expect("known id resolves");
    assert!(r.weights.exists(), "weights pulled to disk");
    assert!(
        !r.weights.to_string_lossy().contains("mmproj"),
        "weights is the weights artifact"
    );
    let mm = r.mmproj.expect("vision companion detected");
    assert!(mm.exists(), "mmproj pulled to disk");
    assert!(
        mm.to_string_lossy().contains("mmproj"),
        "mmproj is the vision artifact"
    );

    // now installed: the no-pull path resolves the same composition
    let r2 = reg
        .resolve("vis", Some("q8"), false, None)
        .await
        .unwrap()
        .expect("resolves installed");
    assert_eq!(r2.weights, r.weights);
}

/// Two catalogued drafters (muse ships DFlash1 and DFlash2) must elect
/// deliberately, not by declaration order. Plain first-match made the
/// choice invisible and the ordering load-bearing.
#[tokio::test]
async fn drafter_election_prefers_the_pin_then_the_default() {
    let dir = tempfile::tempdir().unwrap();
    let drafter = |id: &str, dest: &str, default: bool| CatalogArtifact {
        id: id.into(),
        kind: ArtifactKind::Drafter,
        format: "gguf".into(),
        runtime: ArtifactRuntime::default(),
        source: None,
        label: format!("Speed drafter ({id})"),
        quant: None,
        default,
        required: false,
        min_cc: None,
        workspace: None,
        shape: None,
        files: vec![CatalogFile {
            url: "http://invalid.invalid/x".into(),
            dest: dest.into(),
            sha256: "0".repeat(64),
            size: 3,
        }],
    };
    let catalog = Catalog {
        schema: 3,
        models: vec![CatalogModel {
            id: "m".into(),
            display: "M".into(),
            vendor: None,
            family: None,
            mtp_in_file: false,
            capability: vec!["chat".into(), "speculative".into()],
            revision: None,
            license: None,
            kv_default: None,
            specs: Default::default(),
            artifacts: vec![
                CatalogArtifact {
                    id: "q8".into(),
                    kind: ArtifactKind::Weights,
                    format: "gguf".into(),
                    runtime: ArtifactRuntime::default(),
                    source: None,
                    label: "Full quality".into(),
                    quant: Some("Q8_0".into()),
                    default: true,
                    required: false,
                    min_cc: None,
                    workspace: None,
                    shape: None,
                    files: vec![CatalogFile {
                        url: "http://invalid.invalid/w".into(),
                        dest: "M/w.gguf".into(),
                        sha256: "0".repeat(64),
                        size: 3,
                    }],
                },
                // v2 declared first and default; v1 second. Order must not
                // be what decides.
                drafter("d2", "M/d2.gguf", true),
                drafter("d1", "M/d1.gguf", false),
            ],
        }],
    };
    let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());
    let put = |rel: &str| {
        let p = dir.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"abc").unwrap();
    };
    put("M/w.gguf");

    // Only v1 on disk: the default is unavailable, so the installed one is
    // wired rather than nothing - an endpoint should not lose speculation
    // because a newer drafter exists that was never downloaded.
    put("M/d1.gguf");
    let r = reg
        .resolve("m", Some("q8"), false, None)
        .await
        .unwrap()
        .expect("resolves");
    assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d1"));

    // both on disk: the default wins, not the first declared or the first
    // installed
    put("M/d2.gguf");
    let r = reg
        .resolve("m", Some("q8"), false, None)
        .await
        .unwrap()
        .expect("resolves");
    assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d2"));

    // an explicit pin beats the default, and is wired without being
    // default: asking for it is the same consent `default` expresses
    let r = reg
        .resolve("m", Some("q8"), false, Some("d1"))
        .await
        .unwrap()
        .expect("resolves");
    assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d1"));
    assert!(
        r.mtp
            .expect("pin wires without asking")
            .ends_with("d1.gguf")
    );

    // a pin naming an artifact this model does not have falls back rather
    // than serving nothing
    let r = reg
        .resolve("m", Some("q8"), false, Some("nope"))
        .await
        .unwrap()
        .expect("resolves");
    assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d2"));
}

/// The corner that used to go silent (a follow-up): a pin naming
/// a real catalogued artifact whose bytes are not downloaded. The election
/// skipped the installed check on the pin arm, and the three consumers each
/// patched around it separately - so `drafter_any` wired the installed
/// sibling while `drafter_pick` reported nothing, and the "which drafter
/// did On get me" surface said nothing in the one case where the answer is
/// least guessable. One election feeds all three fields now.
#[tokio::test]
async fn a_dead_pin_falls_back_and_the_fallback_is_named() {
    let dir = tempfile::tempdir().unwrap();
    let drafter = |id: &str, dest: &str, default: bool| CatalogArtifact {
        id: id.into(),
        kind: ArtifactKind::Drafter,
        format: "gguf".into(),
        runtime: ArtifactRuntime::default(),
        source: None,
        label: format!("Speed drafter ({id})"),
        quant: None,
        default,
        required: false,
        min_cc: None,
        workspace: None,
        shape: None,
        files: vec![CatalogFile {
            url: "http://invalid.invalid/x".into(),
            dest: dest.into(),
            sha256: "0".repeat(64),
            size: 3,
        }],
    };
    let catalog = Catalog {
        schema: 3,
        models: vec![CatalogModel {
            id: "m".into(),
            display: "M".into(),
            vendor: None,
            family: None,
            mtp_in_file: false,
            capability: vec!["chat".into(), "speculative".into()],
            revision: None,
            license: None,
            kv_default: None,
            specs: Default::default(),
            artifacts: vec![
                CatalogArtifact {
                    id: "q8".into(),
                    kind: ArtifactKind::Weights,
                    format: "gguf".into(),
                    runtime: ArtifactRuntime::default(),
                    source: None,
                    label: "Full quality".into(),
                    quant: Some("Q8_0".into()),
                    default: true,
                    required: false,
                    min_cc: None,
                    workspace: None,
                    shape: None,
                    files: vec![CatalogFile {
                        url: "http://invalid.invalid/w".into(),
                        dest: "M/w.gguf".into(),
                        sha256: "0".repeat(64),
                        size: 3,
                    }],
                },
                drafter("d2", "M/d2.gguf", true),
                drafter("d1", "M/d1.gguf", false),
            ],
        }],
    };
    let reg = Registry::from_catalog(catalog, dir.path().to_path_buf());
    let put = |rel: &str| {
        let p = dir.path().join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"abc").unwrap();
    };
    put("M/w.gguf");
    put("M/d1.gguf"); // Only the non-default sibling is on disk

    // Pin the default (d2) while its bytes are missing: an explicit "on"
    // wires the installed sibling, and names it - wired and named must be
    // the same artifact.
    let r = reg
        .resolve("m", Some("q8"), false, Some("d2"))
        .await
        .unwrap()
        .expect("resolves");
    assert_eq!(r.drafter_pick.as_ref().map(|(i, _)| i.as_str()), Some("d1"));
    assert!(
        r.drafter_any
            .as_ref()
            .expect("explicit on wires the sibling")
            .ends_with("d1.gguf"),
        "drafter_any must wire what drafter_pick names"
    );
    // ...but the default lane stays empty: a dead pin was consent for d2,
    // not for silently enabling the non-default d1 (never default-on).
    assert!(
        r.mtp.is_none(),
        "a dead pin must not default-enable a non-default sibling"
    );
}

// The blessed-models manifest ships in the binary - it must always parse and
// every entry must be complete, so a hand-edited models.toml can't ship broken.
// /api/servers answers the capability of a stopped endpoint from here, and
// the composer's mic decides from that whether starting it would give you
// a transcriber. The lookup has to survive the shape a config
// file actually stores: `model` is normally the resolved weights path, not
// the catalog id, so a by-id-only match would report no capability for
// every configured endpoint and the mic would offer nothing.
#[test]
fn capability_resolves_by_id_and_by_weights_path() {
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    let speech = reg
        .catalog()
        .models
        .iter()
        .find(|m| m.capability.iter().any(|c| c == "transcription"))
        .expect("catalog ships at least one speech model");

    let by_id = reg
        .capability_of(&speech.id)
        .expect("resolves by catalog id");
    assert!(
        by_id.iter().any(|c| c == "transcription"),
        "{}: speech by id",
        speech.id
    );

    // ...and by the weights filename, which is what servers/<port>.toml holds.
    let file = speech
        .default_weights()
        .and_then(|a| a.files.first())
        .map(|f| f.dest.clone())
        .expect("speech model has a default weights file");
    let path = format!("/some/models/dir/{file}");
    let by_path = reg
        .capability_of(&path)
        .unwrap_or_else(|| panic!("{path}: resolves by weights path"));
    assert!(
        by_path.iter().any(|c| c == "transcription"),
        "{path}: speech by path"
    );

    // A model the catalog has never heard of stays unknown rather than
    // being guessed at - the mic then leaves it out instead of offering a
    // "speech model" that would not work once started.
    assert!(reg.capability_of("/models/somebody-elses.gguf").is_none());
}

/// Every GGUF weights artifact publishes a shape.
///
/// "Always publish the shape" is only a rule if something enforces it, and
/// this is the half that can be enforced with no GPU and no models on disk:
/// the block is PRESENT. `the shapes generator --check` is the other half -
/// it re-probes installed files and catches a block that drifted from the
/// bytes it describes.
///
/// The consequence of a missing block is not cosmetic: the picker has no
/// second path any more (`approxResident` is gone), so an unpriced artifact
/// shows a dash where a fit verdict belongs.
#[test]
fn every_gguf_weights_artifact_publishes_a_shape() {
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    let mut naked = Vec::new();
    for m in &reg.catalog().models {
        for a in m.weights() {
            // safetensors is exempt and named, not silently skipped:
            // `probe_path` reads GGUF only, so the generator has no
            // geometry source for those and refuses to invent one. The
            // exemption disappears the day the runner can report its own
            // shape after a load.
            if a.format != "gguf" {
                continue;
            }
            if a.shape.is_none() {
                naked.push(format!("{}/{}", m.id, a.id));
            }
        }
    }
    assert!(
        naked.is_empty(),
        "these GGUF weights artifacts publish no shape, so will-it-fit cannot price them: \
         {naked:?}\nregenerate with the shapes generator"
    );
}

#[test]
fn a_published_shape_round_trips_through_the_estimator() {
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    let a = reg
        .catalog()
        .models
        .iter()
        .flat_map(|m| m.weights())
        .find(|a| a.shape.is_some())
        .expect("at least one artifact publishes a shape");
    let s = a.shape.clone().unwrap();
    let weight_bytes = s.weight_bytes;
    let kv_runs: u64 = s.kv_layers.iter().map(|r| r.count).sum();
    let shape = s.into_model_shape(1234, 5678);
    assert_eq!(
        shape.weight_bytes, weight_bytes,
        "weights survive the completion"
    );
    assert_eq!(shape.tower_bytes, 1234, "tower comes from the caller");
    assert_eq!(
        shape.workspace_bytes, 5678,
        "workspace comes from the caller"
    );
    // The published form collapses identical consecutive blocks into runs;
    // the estimator still prices block by block, so the expansion has to
    // give every one of them back.
    assert_eq!(
        shape.kv_layers.len() as u64,
        kv_runs,
        "every KV block in the runs is expanded"
    );
}

#[test]
fn embedded_manifest_parses_and_is_well_formed() {
    let reg = Registry::new(std::path::PathBuf::from("./models"));
    assert!(!reg.catalog().models.is_empty(), "manifest lists models");
    for m in &reg.catalog().models {
        assert!(!m.capability.is_empty(), "{}: has a capability", m.id);
        assert!(
            m.weights().next().is_some(),
            "{}: has a weights artifact",
            m.id
        );
        assert!(
            m.default_weights().is_some(),
            "{}: has a default weights choice",
            m.id
        );
        for a in &m.artifacts {
            assert!(!a.files.is_empty(), "{}/{}: artifact has files", m.id, a.id);
            for f in &a.files {
                assert!(f.url.starts_with("http"), "{}: absolute url", m.id);
                assert_eq!(f.sha256.len(), 64, "{}: sha256 present", m.id);
                assert!(f.size > 0, "{}: nonzero size", m.id);
                assert!(
                    !f.dest.is_empty() && !f.dest.starts_with('/'),
                    "{}: relative dest",
                    m.id
                );
            }
        }
    }
    // A weights artifact can span several files, for two different
    // reasons, and only one of them is sharding.
    //
    // The spawn path hands the runner `files.first()` and the engine takes
    // it from there - for a gguf-split family the loader walks the
    // remaining shards from that first shard's own metadata. So whatever
    // else the artifact carries, file[0] has to be the thing the engine
    // can open: shard 1 of a split, or the single .gguf otherwise.
    // Listing a middle shard first would fail at load, and it is a
    // hand-editing mistake nothing else here would catch - the sizes and
    // hashes would all be correct.
    //
    // The other reason is a notice riding with the weights: Røst's licence
    // is use-restricted and its text has to be undownloadable-without-the-
    // model, so LICENSE.txt sits in the same artifact rather than in an
    // optional companion someone could decline. That is not a shard and
    // must not be read as one.
    for m in &reg.catalog().models {
        for a in m.weights().filter(|a| a.files.len() > 1) {
            let first = &a.files[0].dest;
            let sharded = a.files.iter().any(|f| f.dest.contains("-of-"));
            if a.format == "splash-packed-q4" {
                assert!(a.runtime.checkpoint_dir);
                assert_eq!(first, "Qwen3.8-27B-Splash/manifest.json");
                assert_eq!(a.files[0].sha256, paddock_models::splash::MANIFEST_SHA256);
            } else if a.runtime.checkpoint_dir
                && a.capabilities(m).iter().any(|c| c == "image-generation")
            {
                // A diffusion directory has multiple component subfolders;
                // its root index, not a nested shard's parent, is the entry.
                assert!(first.ends_with("/model_index.json"));
                let root = first.strip_suffix("model_index.json").unwrap();
                assert!(a.files.iter().all(|f| f.dest.starts_with(root)));
                for component in [
                    "transformer/config.json",
                    "text_encoder/config.json",
                    "vae/config.json",
                    "processor/tokenizer.json",
                ] {
                    assert!(
                        a.files
                            .iter()
                            .any(|f| f.dest == format!("{root}{component}"))
                    );
                }
            } else if sharded {
                assert!(
                    first.contains("-00001-of-"),
                    "{}/{}: a sharded weights artifact must list shard 1 first, not {first}",
                    m.id,
                    a.id
                );
            } else {
                assert!(
                    first.ends_with(".gguf") || first.ends_with(".safetensors"),
                    "{}/{}: file[0] is what the runner is handed - it must be the \
                     loadable weights, not {first}",
                    m.id,
                    a.id
                );
            }
        }
    }
    // the annotated view the Studio consumes is valid JSON with the fields it needs
    let v = reg.catalog_annotated();
    assert!(
        v["models"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    );
    assert!(v["models"][0]["installed"].is_boolean());
    assert!(v["models"][0]["total_size"].is_number());
    assert!(
        v["models"][0]["artifacts"].is_array(),
        "serialized as `artifacts`, not `artifact`"
    );
    assert!(
        v["models"][0]["artifacts"][0]["installed"].is_boolean(),
        "piece-level install state"
    );
}
