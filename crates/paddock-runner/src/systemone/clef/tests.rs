//! The runner half of Clef's parity gate: every fixture's ids and spans,
//! rebuilt here from the request, against the reference run's own
//! (the oracle run, written to `<model>/golden/f32`) - the image fixtures'
//! too, their pictures decoded as the endpoint decodes them, against the
//! pixels the reference's decoder gave.
//!
//! Needs `CLEF_DIR` (the checkpoint, with the oracle's runs in
//! `golden/f32/*.json`, or in `CLEF_GOLDEN_DIR`); the fixture lists are the
//! golden directory's when it has them, the repository's otherwise. Without
//! `CLEF_DIR` the test says so and passes, like every model-backed gate here.
//! With `CLEF_GGUF` the tokenizer is the GGUF's own (its vocabulary and
//! merges) and the runs are the GGUF oracle's (`golden/gguf-f32`): a GGUF
//! must encode every fixture exactly as the release's tokenizer.json does.

use super::super::pyjson::{self, PyVal};
use super::encode::{ClefTok, MAX_LENGTH, encode, media_ids};
use super::question::{self, render};

const IMAGE_PAD: u32 = 248056;

#[test]
fn pixel_overrides_cannot_expand_beyond_the_context() {
    use serde_json::json;
    let vision = super::ClefVision {
        pad: IMAGE_PAD,
        min_pixels: 65536,
        max_pixels: 16777216,
    };
    for bounds in [
        json!({"min_pixels": 1u64 << 50, "max_pixels": 1u64 << 50}),
        json!({"min_pixels": 1024, "max_pixels": u64::MAX}),
        json!({"size": {"shortest_edge": 1024, "longest_edge": 1u64 << 50}}),
        json!({"min_pixels": 1, "max_pixels": 100}),
        json!({"min_pixels": 1024}),
    ] {
        let body = json!({"media_kwargs": bounds});
        let error = super::pixel_bounds(body.as_object().unwrap(), &vision).unwrap_err();
        assert_eq!(error.status(), axum::http::StatusCode::UNPROCESSABLE_ENTITY);
    }
    let defaults = super::pixel_bounds(&serde_json::Map::new(), &vision).unwrap();
    assert_eq!(defaults, (65536, 16777216));
    let override_ = json!({"media_kwargs":{"size":{"shortest_edge":1024,"longest_edge":262144}}});
    assert_eq!(
        super::pixel_bounds(override_.as_object().unwrap(), &vision).unwrap(),
        (1024, 262144)
    );
}

#[test]
fn sequences_match_the_reference() {
    let Some(dir) = std::env::var_os("CLEF_DIR").map(std::path::PathBuf::from) else {
        eprintln!("SKIP: CLEF_DIR is not set");
        return;
    };
    let gguf = std::env::var_os("CLEF_GGUF").map(std::path::PathBuf::from);
    let tok = match &gguf {
        Some(g) => ClefTok::from_gguf(
            paddock_models::mapped::MappedGguf::open(g)
                .expect("the GGUF")
                .gguf(),
        )
        .expect("the GGUF's tokenizer"),
        None => ClefTok::load(&dir).expect("tokenizer"),
    };
    let sub = if gguf.is_some() { "gguf-f32" } else { "f32" };
    let golden = std::env::var_os("CLEF_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| dir.join("golden"));
    // The fixtures are read at RUN time, never embedded: the development
    // tree keeps them with the bench scripts, and a tree without those (the
    // public one) must still build - its run skips until a golden directory
    // holding fixtures.jsonl is named. The pictures stay beside the file that
    // names them, in the golden directory's copy or the bench one.
    let bench = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/bench");
    let fixtures_path = if golden.join("fixtures.jsonl").exists() {
        golden.join("fixtures.jsonl")
    } else {
        bench.join("clef_fixtures.jsonl")
    };
    let Ok(fixtures) = std::fs::read_to_string(&fixtures_path) else {
        eprintln!(
            "SKIP: no Clef fixtures at {} - set CLEF_GOLDEN_DIR to a directory holding fixtures.jsonl",
            fixtures_path.display()
        );
        return;
    };
    let (images_path, image_root) = if golden.join("image_fixtures.jsonl").exists() {
        (golden.join("image_fixtures.jsonl"), golden.clone())
    } else {
        (bench.join("clef_image_fixtures.jsonl"), bench)
    };
    let images = std::fs::read_to_string(&images_path).unwrap_or_default();
    let lines = fixtures
        .lines()
        .map(|l| (l, false))
        .chain(images.lines().map(|l| (l, true)));
    let mut checked = 0usize;
    for (line, image) in lines.filter(|(l, _)| !l.trim().is_empty()) {
        let req = pyjson::parse(line).expect("a fixture");
        let id = req.get("id").and_then(PyVal::as_str).expect("an id");
        let gold_path = golden.join(format!("{sub}/{id}.json"));
        if image && !gold_path.exists() {
            // a golden set from before the image fixtures
            eprintln!("SKIP {id}: no reference run in {}", golden.display());
            continue;
        }
        let gold: serde_json::Value =
            serde_json::from_slice(&std::fs::read(gold_path).expect("golden json"))
                .expect("golden parses");
        let qs = question::parse_all(req.get("questions").expect("questions")).expect("valid");
        let state = render(req.get("state").expect("state"));
        let tokens = images_as_decoded(&golden.join(sub), &image_root, id, line, &gold);
        let (media, at) = media_ids(&tok, &tokens, IMAGE_PAD).expect("media");
        // the reference's own cut, to compare a state it shortened
        let enc = encode(
            &tok,
            &qs,
            &state,
            (&media, &at),
            MAX_LENGTH,
            Some(usize::MAX),
        )
        .expect("encodes");
        let want: Vec<u32> = gold["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(enc.ids.len(), want.len(), "{id}: length");
        if let Some(i) = enc.ids.iter().zip(&want).position(|(a, b)| a != b) {
            panic!(
                "{id}: first id difference at {i}: {} vs {}",
                enc.ids[i], want[i]
            );
        }
        let gq = gold["questions"].as_array().unwrap();
        assert_eq!(gq.len(), qs.len(), "{id}: question count");
        for ((q, s), g) in qs.iter().zip(&enc.spans).zip(gq) {
            assert_eq!(g["id"], q.id.as_str(), "{id}: question order");
            assert_eq!(g["type"], q.kind.qtype(), "{id}/{}: type", q.id);
            let span = |v: &serde_json::Value| {
                (
                    v[0].as_u64().unwrap() as usize,
                    v[1].as_u64().unwrap() as usize,
                )
            };
            assert_eq!(s.question, span(&g["span"]), "{id}/{}: question span", q.id);
            let gs: Vec<_> = g["option_spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(span)
                .collect();
            assert_eq!(s.options, gs, "{id}/{}: option spans", q.id);
            let gi: Vec<&str> = g["option_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap())
                .collect();
            let oi: Vec<&str> = q.options.iter().map(|o| o.id.as_str()).collect();
            assert_eq!(oi, gi, "{id}/{}: option ids", q.id);
        }
        checked += 1;
    }
    assert!(checked > 0, "no fixtures");
    eprintln!("{checked} fixtures token-exact");
}

/// A fixture's images decoded as the endpoint decodes them (a base64 data
/// URL through `decode_image_url_reference`: JPEG as libjpeg-turbo, EXIF
/// upright, alpha dropped), held to the size and the bytes of the
/// reference's decode (`<golden run>/<id>.rgb<k>.u8`). Returns each image's
/// tokens at the fixture's pixel bounds.
fn images_as_decoded(
    golden: &std::path::Path,
    root: &std::path::Path,
    id: &str,
    line: &str,
    gold: &serde_json::Value,
) -> Vec<usize> {
    use base64::Engine as _;
    let req: serde_json::Value = serde_json::from_str(line).expect("fixture json");
    let Some(paths) = req["images"].as_array() else {
        return Vec::new();
    };
    let (mut lo, mut hi) = (65536u64, 16777216u64);
    if let Some(size) = req["media_kwargs"].get("size") {
        lo = size["shortest_edge"].as_u64().expect("size");
        hi = size["longest_edge"].as_u64().expect("size");
    }
    let sizes = gold["media"]["images"].as_array().expect("decoded sizes");
    let grids = gold["media"]["image_grid_thw"].as_array().expect("grids");
    paths
        .iter()
        .enumerate()
        .map(|(k, p)| {
            let p = p.as_str().expect("path");
            let bytes = std::fs::read(root.join(p)).expect("image file");
            let url = format!(
                "data:application/octet-stream;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            );
            let (rgb, w, h) =
                crate::reference_image::decode_image_url_reference(&url).expect("decodes");
            assert_eq!(
                (h as u64, w as u64),
                (
                    sizes[k]["height"].as_u64().expect("height"),
                    sizes[k]["width"].as_u64().expect("width")
                ),
                "{id}: image {k} size"
            );
            // CLEF_DECODED_OUT: keep this decode, for the engine gate to run
            // on (CLEF_RGB_DIR) - the decoder's effect on the answers
            if let Some(out) = std::env::var_os("CLEF_DECODED_OUT") {
                let out = std::path::PathBuf::from(out);
                std::fs::create_dir_all(&out).expect("CLEF_DECODED_OUT");
                std::fs::write(out.join(format!("{id}.rgb{k}.u8")), &rgb).expect("write");
            }
            let want = std::fs::read(golden.join(format!("{id}.rgb{k}.u8"))).expect("rgb");
            let (mut differ, mut worst) = (0usize, 0u8);
            for (a, b) in rgb.iter().zip(&want) {
                if a != b {
                    differ += 1;
                    worst = worst.max(a.abs_diff(*b));
                }
            }
            assert_eq!(
                differ, 0,
                "{id}: image {k} ({p}) decodes {differ} bytes off the reference's (worst {worst})"
            );
            let (rh, rw) =
                paddock_models::clef::smart_resize(h as u64, w as u64, 32, lo, hi).expect("resize");
            let g = &grids[k];
            assert_eq!(
                (rh / 16, rw / 16),
                (g[1].as_u64().expect("grid"), g[2].as_u64().expect("grid")),
                "{id}: image {k} grid"
            );
            ((rh / 32) * (rw / 32)) as usize
        })
        .collect()
}
