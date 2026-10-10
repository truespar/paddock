//! PaddleOCR-VL's layout companion (PP-DocLayoutV3): pulled with the reader
//! on CUDA into the reader's own folder - where the runner discovers it -
//! left off Metal, which serves whole pages, and charged like the other
//! switch-less resident companions.

use super::*;

const MODEL: &str = "paddleocr-vl-1.6";

#[test]
fn the_layout_reader_rides_the_cuda_bundle_only() {
    for (backend, want) in [
        ("cuda", vec!["bf16", "mmproj", "layout"]),
        ("metal", vec!["bf16", "mmproj"]),
    ] {
        let reg = Registry::new("./models".into()).with_backend(backend);
        let m = reg.catalog_of(MODEL).unwrap();
        let ids: Vec<_> = m
            .default_bundle_for_backend(backend, Some([12, 1]))
            .iter()
            .map(|a| a.id.as_str())
            .collect();
        assert_eq!(ids, want, "{backend}");
    }
}

#[test]
fn the_layout_reader_lands_where_the_runner_discovers_it() {
    let reg = Registry::new("./models".into()).with_backend("cuda");
    let m = reg.catalog_of(MODEL).unwrap();
    let layout = m.artifact("layout").unwrap();
    assert_eq!(layout.kind, ArtifactKind::Layout);
    assert!(!layout.kind.is_lane_companion() && !layout.kind.is_mmproj());
    let folder = m.artifact("bf16").unwrap().files[0]
        .dest
        .rsplit_once('/')
        .unwrap()
        .0;
    for f in &layout.files {
        // a `PP-DocLayoutV3*` directory inside the weights' folder
        assert!(
            f.dest.starts_with(&format!("{folder}/PP-DocLayoutV3/")),
            "{}",
            f.dest
        );
    }
    let names: Vec<_> = layout
        .files
        .iter()
        .map(|f| f.dest.rsplit('/').next().unwrap())
        .collect();
    for needed in ["model.safetensors", "config.json", "LICENSE", "README.md"] {
        assert!(names.contains(&needed), "{needed}");
    }
    assert_eq!(layout.source.as_ref().unwrap().license, "apache-2.0");
}

#[test]
fn the_layout_reader_is_charged_with_its_workspace() {
    let reg = Registry::new("./this-dir-does-not-exist".into()).with_backend("cuda");
    let m = reg.catalog_of(MODEL).unwrap();
    let layout = m.artifact("layout").unwrap();
    let charged = crate::estimate::lane_companion_bytes_for(m, &reg, m.artifact("bf16"));
    assert_eq!(charged, layout.total_size() + layout.workspace.unwrap());
}
