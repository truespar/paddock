//! Kumo-Tabular on CUDA against the original model's own outputs.
//!
//! There is no llama.cpp for a tabular foundation model, so the oracle is the
//! original: NVIDIA's `_KumoTabular` from the pinned structured-data-models
//! revision, PyTorch F32 with no autocast, run over five prepared tables per
//! checkpoint - `(context, query, columns)` = (1, 2, 1), (17, 5, 7),
//! (65, 9, 13), (257, 17, 31), (513, 33, 65), with numerical and categorical
//! columns, missing query cells and an all-missing context column. The export
//! step writes each checkpoint directory (`config.json`, the lossless F32
//! `model.safetensors`) and its `oracle.json` beside it.
//!
//! The gates, per case:
//!   - every raw output (ten logits or 999 quantiles a query row) within
//!     `1e-4 + 1e-5 * |reference|` - the Metal lane's gate, unchanged;
//!   - bit-exact query isolation: a query row predicted alone equals the
//!     same row predicted with the others (the graph is row-invariant by
//!     construction, so this is equality, not a tolerance);
//!   - a repeat after a different shape is bit-identical;
//!   - a fitted context replays every query bit-identically to the direct
//!     pass, alone and together, and a second replay does not move;
//!   - the fused blocks (the served path) and the op-by-op reference blocks
//!     agree to the bit, direct and across a replay (a context fitted by one
//!     path, replayed by the other);
//!   - an ensemble's members run as one pass give each member the bits its
//!     own pass gives, direct and through a fit and replay (the members test,
//!     and the recipe test on the real recipe's members).
//!
//! Needs `KUMO_REFERENCE_DIR` (default: the checkout's
//! `target/kumo-reference`) holding `<size>-<task>/` directories; uploads
//! whole checkpoints, so it runs under PADDOCK_HEAVY_TESTS. `KUMO_BENCH=1`
//! adds warm timings: seven samples after two warm-ups, direct and cached
//! (`KUMO_BENCH_REF=1` also times the reference path, alternating);
//! `KUMO_ONLY=<size>-<task>` and `KUMO_CONTEXT=<rows>` narrow the sweep.

mod common;

use std::path::PathBuf;

use paddock_engine::gpu_model::kumo::GpuKumo;
use paddock_models::kumo::Table;

fn say(msg: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{msg}");
}

fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("a number list")
        .iter()
        .map(|x| x.as_f64().map_or(f32::NAN, |n| n as f32))
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
fn kumo_matches_the_original_model() {
    if !common::heavy() {
        return;
    }
    let root = std::env::var_os("KUMO_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/kumo-reference")
        });
    let dirs: Vec<PathBuf> = ["small", "medium", "large"]
        .iter()
        .flat_map(|s| ["classification", "regression"].map(|t| root.join(format!("{s}-{t}"))))
        .filter(|d| d.join("oracle.json").exists())
        .filter(|d| std::env::var("KUMO_ONLY").map_or(true, |o| d.ends_with(o)))
        .collect();
    if dirs.is_empty() {
        common::missing(&format!(
            "no Kumo exports with oracles under {}",
            root.display()
        ));
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_kumo() {
        common::missing("this pack predates the Kumo-Tabular lane (slots 698-708)");
        return;
    }
    let bench = std::env::var_os("KUMO_BENCH").is_some();
    let mut misses = 0usize;
    for dir in &dirs {
        let mut m = GpuKumo::load(exec.clone(), dir).expect("load the checkpoint");
        // KUMO_REFERENCE=1: time and profile the reference blocks instead
        let base_path = std::env::var_os("KUMO_REFERENCE").is_some();
        m.set_reference_path(base_path);
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("oracle.json")).unwrap()).unwrap();
        for case in v["cases"].as_array().unwrap() {
            let x = f32s(&case["x"]);
            let y = f32s(&case["y"]);
            let cat: Vec<bool> = case["categorical"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_bool().unwrap())
                .collect();
            let (nc, q, cols) = (
                y.len(),
                case["query_rows"].as_u64().unwrap() as usize,
                cat.len(),
            );
            if std::env::var("KUMO_CONTEXT").is_ok_and(|c| c != nc.to_string()) {
                continue;
            }
            let expected = f32s(&case["output"]);
            let t = Table {
                x: &x,
                y: &y,
                categorical: &cat,
                query_rows: q,
            };
            let out = m.predict(&t).expect("predict");
            assert_eq!(out.values.len(), expected.len(), "{name}: output length");
            let mut max = 0f32;
            let mut outside = 0usize;
            for (i, (&a, &b)) in out.values.iter().zip(&expected).enumerate() {
                let err = (a - b).abs();
                max = max.max(err);
                if err > 1e-4 + 1e-5 * b.abs() {
                    if outside == 0 {
                        say(&format!(
                            "KUMO_MISS {name} {nc}/{q}/{cols} output {i}: {a} vs {b}"
                        ));
                    }
                    outside += 1;
                }
            }
            misses += outside;
            // KUMO_DIAG_ORACLE=<file beside oracle.json>: report (never gate)
            // the distance to another evaluation of the same cases
            if let Ok(f) = std::env::var("KUMO_DIAG_ORACLE") {
                let d: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(dir.join(f)).unwrap()).unwrap();
                let alt = d["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|c| c["context_rows"].as_u64() == Some(nc as u64))
                    .map(|c| {
                        c["output"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect::<Vec<_>>()
                    })
                    .unwrap();
                let ours = out
                    .values
                    .iter()
                    .zip(&alt)
                    .map(|(a, b)| (f64::from(*a) - b).abs())
                    .fold(0., f64::max);
                let oracle = expected
                    .iter()
                    .zip(&alt)
                    .map(|(a, b)| (f64::from(*a) - b).abs())
                    .fold(0., f64::max);
                say(&format!(
                    "KUMO_DIAG {name} context={nc} ours_vs_alt={ours:.3e} oracle_vs_alt={oracle:.3e}"
                ));
            }
            // one query row alone: bit-identical to its place in the batch
            let one = m
                .predict(&Table {
                    x: &x[..(nc + 1) * cols],
                    y: &y,
                    categorical: &cat,
                    query_rows: 1,
                })
                .expect("single-row predict");
            let per = expected.len() / q;
            assert_eq!(one.values, out.values[..per], "{name}: query isolation");
            let again = m.predict(&t).expect("repeat");
            assert_eq!(
                again.values, out.values,
                "{name}: repeat after another shape"
            );
            let (ctx, fit) = m.fit(&t).expect("fit");
            assert!(fit.values.is_empty());
            let queries = &x[nc * cols..];
            let cached = m.query(&ctx, queries, q).expect("replay");
            assert_eq!(cached.values, out.values, "{name}: replay vs direct");
            let cached_one = m.query(&ctx, &queries[..cols], 1).expect("replay one");
            assert_eq!(
                cached_one.values,
                out.values[..per],
                "{name}: replay one row"
            );
            let again = m.query(&ctx, queries, q).expect("replay again");
            assert_eq!(again.values, cached.values, "{name}: replay must not move");
            if m.set_reference_path(true) {
                let r = m.predict(&t).expect("reference predict");
                assert_eq!(r.values, out.values, "{name}: fused vs reference, direct");
                let r = m.query(&ctx, queries, q).expect("reference replay");
                assert_eq!(r.values, out.values, "{name}: fused fit, reference replay");
                let (rctx, _) = m.fit(&t).expect("reference fit");
                m.set_reference_path(base_path);
                let r = m.query(&rctx, queries, q).expect("fused replay");
                assert_eq!(r.values, out.values, "{name}: reference fit, fused replay");
            } else {
                common::missing("this pack predates the fused Kumo passes (slots 709-713)");
            }
            let mut line = format!(
                "KUMO {name} context={nc} query={q} cols={cols} max_abs={max:.3e} outside={outside} \
                 direct_ms={:.2} fit_ms={:.2} cached_ms={:.2} workspace={} ctx_bytes={}",
                out.gpu_seconds * 1e3,
                fit.gpu_seconds * 1e3,
                cached.gpu_seconds * 1e3,
                out.workspace_bytes,
                ctx.bytes()
            );
            if bench {
                let (mut direct, mut replay) = (Vec::new(), Vec::new());
                // KUMO_BENCH=direct or =cached times one side alone (profiles)
                let which = std::env::var("KUMO_BENCH").unwrap_or_default();
                for i in 0..9 {
                    let s = std::time::Instant::now();
                    if which != "cached" {
                        m.predict(&t).unwrap();
                    }
                    let d = s.elapsed().as_secs_f64() * 1e3;
                    let s = std::time::Instant::now();
                    if which != "direct" {
                        m.query(&ctx, queries, q).unwrap();
                    }
                    let r = s.elapsed().as_secs_f64() * 1e3;
                    if i >= 2 {
                        direct.push(d);
                        replay.push(r);
                    }
                }
                line += &format!(
                    " bench_direct_ms={:.2} bench_cached_ms={:.2}",
                    median(direct),
                    median(replay)
                );
                if std::env::var_os("KUMO_BENCH_REF").is_some() {
                    // fused and reference alternate, so drift favours neither
                    let mut t4 = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                    for i in 0..9 {
                        for reference in [i % 2 == 0, i % 2 != 0] {
                            m.set_reference_path(reference);
                            let s0 = std::time::Instant::now();
                            m.predict(&t).unwrap();
                            let d = s0.elapsed().as_secs_f64() * 1e3;
                            let s0 = std::time::Instant::now();
                            m.query(&ctx, queries, q).unwrap();
                            let r = s0.elapsed().as_secs_f64() * 1e3;
                            if i >= 2 {
                                t4[usize::from(reference) * 2].push(d);
                                t4[usize::from(reference) * 2 + 1].push(r);
                            }
                        }
                    }
                    m.set_reference_path(false);
                    let [fd, fc, rd, rc] = t4.map(median);
                    line += &format!(
                        " ab_fused_direct_ms={fd:.2} ab_ref_direct_ms={rd:.2} ab_fused_cached_ms={fc:.2} ab_ref_cached_ms={rc:.2}"
                    );
                }
            }
            say(&line);
        }
    }
    assert_eq!(misses, 0, "outputs outside the reference gate");
}

/// An ensemble's members in one pass (slots 714-717) against the same
/// members one at a time: every output bit-identical, direct and through a
/// fit and replay, on the fused and the reference path. The members are each
/// prepared table with its columns rotated (member k starts at column k), so
/// a mix-up between members moves bits; the 513-row table does not fit one
/// pass for four members and splits into groups, and a synthetic 16-member
/// table replays 1024 query rows in slices.
#[test]
fn kumo_members_match_one_at_a_time() {
    if !common::heavy() {
        return;
    }
    let root = std::env::var_os("KUMO_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/kumo-reference")
        });
    let dirs: Vec<PathBuf> = ["small", "medium", "large"]
        .iter()
        .flat_map(|s| ["classification", "regression"].map(|t| root.join(format!("{s}-{t}"))))
        .filter(|d| d.join("oracle.json").exists())
        .filter(|d| std::env::var("KUMO_ONLY").map_or(true, |o| d.ends_with(o)))
        .collect();
    if dirs.is_empty() {
        common::missing(&format!("no Kumo exports under {}", root.display()));
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_kumo_members() {
        common::missing("this pack predates Kumo member passes (slots 714-717)");
        return;
    }
    // member k: the table with its columns rotated by k
    let rotate = |x: &[f32], cat: &[bool], k: usize| -> (Vec<f32>, Vec<bool>) {
        let cols = cat.len();
        (
            x.chunks(cols)
                .flat_map(|row| (0..cols).map(move |j| row[(j + k) % cols]))
                .collect(),
            (0..cols).map(|j| cat[(j + k) % cols]).collect(),
        )
    };
    let check = |m: &mut GpuKumo,
                 name: &str,
                 x: &[f32],
                 y: &[f32],
                 cat: &[bool],
                 q: usize,
                 members: usize| {
        let (nc, cols) = (y.len(), cat.len());
        let mems: Vec<(Vec<f32>, Vec<bool>)> = (0..members).map(|k| rotate(x, cat, k)).collect();
        let tables: Vec<Table> = mems
            .iter()
            .map(|(x, c)| Table {
                x,
                y,
                categorical: c,
                query_rows: q,
            })
            .collect();
        let qs: Vec<&[f32]> = mems.iter().map(|(x, _)| &x[nc * cols..]).collect();
        for reference in [false, true] {
            if m.set_reference_path(reference) != reference {
                continue;
            }
            let path = if reference { "reference" } else { "fused" };
            let one: Vec<Vec<f32>> = tables
                .iter()
                .map(|t| m.predict(t).unwrap().values)
                .collect();
            let many = m.predict_many(&tables).unwrap();
            let (ctxs, _) = m.fit_many(&tables).unwrap();
            let mut at = 0;
            for ctx in &ctxs {
                for (j, o) in m
                    .query_many(ctx, &qs[at..at + ctx.members()], q)
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    assert_eq!(
                        o.values,
                        one[at + j],
                        "{name} {path}: member {} batched replay",
                        at + j
                    );
                }
                at += ctx.members();
            }
            for (k, (a, b)) in one.iter().zip(&many).enumerate() {
                assert_eq!(a, &b.values, "{name} {path}: member {k} batched vs alone");
            }
            say(&format!(
                "KUMO_MEMBERS {name} context={nc} query={q} cols={cols} members={members} fit_passes={} path={path} ok",
                ctxs.len()
            ));
        }
        m.set_reference_path(false);
    };
    for (n, dir) in dirs.iter().enumerate() {
        let mut m = GpuKumo::load(exec.clone(), dir).expect("load the checkpoint");
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("oracle.json")).unwrap()).unwrap();
        for case in v["cases"].as_array().unwrap() {
            let cat: Vec<bool> = case["categorical"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_bool().unwrap())
                .collect();
            let q = case["query_rows"].as_u64().unwrap() as usize;
            check(
                &mut m,
                &name,
                &f32s(&case["x"]),
                &f32s(&case["y"]),
                &cat,
                q,
                4,
            );
        }
        if n == 0 {
            // 50 context rows, 10 columns, 16 members: one fitted pass, and a
            // 1024-row query replayed in slices of what 16 members may carry
            let (nc, q, cols) = (50usize, 1024usize, 10usize);
            let mut seed = 0x2545_f491_4f6c_dd1du64;
            let x: Vec<f32> = (0..(nc + q) * cols)
                .map(|i| {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if i % 37 == 5 {
                        f32::NAN
                    } else {
                        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
                    }
                })
                .collect();
            let y: Vec<f32> = (0..nc).map(|i| (i % 3) as f32).collect();
            let cat: Vec<bool> = (0..cols).map(|j| j % 4 == 3).collect();
            check(&mut m, &name, &x, &y, &cat, q, 16);
        }
    }
}

/// The whole `sdm_v1` recipe against the original pipeline: SDM's fitted
/// processors and the original model on CPU, eight ensemble members over two
/// raw tables (65 and 129 context rows) with missing and unseen categories,
/// a >50-category column, constant and all-missing columns, huge offsets
/// and outliers, label shifts and regression unit restoration. Needs
/// `recipe-oracle.json` beside each export.
///
/// Gated, tolerance unchanged (`1e-4 + 1e-5 * |reference|`): the product -
/// every reduced regression prediction, in standardized units - and every
/// classification member's raw logits; and every member replays bit-
/// identically to its direct pass.
///
/// Reported, not gated: a regression member's 999 raw quantiles against the
/// CPU F32 oracle. Some members of these deliberately hostile tables are
/// ill-conditioned (rank-Gaussian and power-transformed features with
/// 1e20-scale offsets), and there two F32 evaluations part by more than the
/// gate while the ensemble's trimmed mean agrees. Measured against the same
/// pipeline in F64 (`KUMO_RECIPE_DIAG=recipe-oracle-f64.json`, the export
/// script's `--double`), this lane's worst member error is at or below the
/// CPU F32 oracle's own in 5 of 6 cases (large regression, 129 rows: 5.4e-4
/// against 1.1e-3) and 1.11e-4 against 1.00e-4 in the sixth - F32 noise, not
/// a lane defect. The Metal lane and PyTorch MPS miss the same members.
#[test]
fn kumo_recipe_matches_the_original_pipeline() {
    use paddock_models::kumo::Task;
    use paddock_models::kumo::recipe::{Cell, Fitted, RawTable};
    if !common::heavy() {
        return;
    }
    let root = std::env::var_os("KUMO_REFERENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/kumo-reference")
        });
    let dirs: Vec<PathBuf> = ["small", "medium", "large"]
        .iter()
        .flat_map(|s| ["classification", "regression"].map(|t| root.join(format!("{s}-{t}"))))
        .filter(|d| d.join("recipe-oracle.json").exists())
        .filter(|d| std::env::var("KUMO_ONLY").map_or(true, |o| d.ends_with(o)))
        .collect();
    if dirs.is_empty() {
        common::missing(&format!("no Kumo recipe oracles under {}", root.display()));
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_kumo() {
        common::missing("this pack predates the Kumo-Tabular lane (slots 698-708)");
        return;
    }
    let mut failures = 0usize;
    for dir in &dirs {
        let mut m = GpuKumo::load(exec.clone(), dir).expect("load the checkpoint");
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("recipe-oracle.json")).unwrap())
                .unwrap();
        for case in v["cases"].as_array().unwrap() {
            let req = &case["request"];
            let raw = RawTable {
                context: serde_json::from_value(req["context"].clone()).unwrap(),
                targets: serde_json::from_value(req["targets"].clone()).unwrap(),
                categorical: serde_json::from_value(req["categorical"].clone()).unwrap(),
            };
            let query: Vec<Vec<Cell>> = serde_json::from_value(req["query"].clone()).unwrap();
            let f = Fitted::fit(
                &raw,
                &m.config.task,
                req["num_estimators"].as_u64().unwrap() as usize,
                req["seed"].as_u64().unwrap(),
            )
            .unwrap();
            let mut outputs = Vec::new();
            let (mut max, mut outside) = (0f32, 0usize);
            let classification = m.config.task == Task::Classification;
            // each member's context + query rows and its query rows alone
            let qs: Vec<Vec<f32>> = f.members.iter().map(|mb| mb.transform(&query)).collect();
            let xs: Vec<Vec<f32>> = f
                .members
                .iter()
                .zip(&qs)
                .map(|(mb, q)| [mb.context.as_slice(), q].concat())
                .collect();
            let tables: Vec<Table> = f
                .members
                .iter()
                .zip(&xs)
                .map(|(mb, x)| Table {
                    x,
                    y: &mb.y,
                    categorical: &mb.categorical,
                    query_rows: query.len(),
                })
                .collect();
            for (i, (member, expected)) in f
                .members
                .iter()
                .zip(case["members"].as_array().unwrap())
                .enumerate()
            {
                let q = member.transform(&query);
                let mut x = member.context.clone();
                x.extend_from_slice(&q);
                let t = Table {
                    x: &x,
                    y: &member.y,
                    categorical: &member.categorical,
                    query_rows: query.len(),
                };
                let out = m.predict(&t).unwrap();
                let (ctx, _) = m.fit(&t).unwrap();
                let cached = m.query(&ctx, &q, query.len()).unwrap();
                assert_eq!(
                    out.values, cached.values,
                    "{name}: member {i} direct vs replay"
                );
                if m.set_reference_path(true) {
                    let r = m.predict(&t).unwrap();
                    assert_eq!(
                        r.values, out.values,
                        "{name}: member {i} fused vs reference"
                    );
                    m.set_reference_path(false);
                }
                for (j, (a, b)) in out.values.iter().zip(f32s(&expected["output"])).enumerate() {
                    max = max.max((a - b).abs());
                    if (a - b).abs() > 1e-4 + 1e-5 * b.abs() {
                        if outside == 0 {
                            say(&format!(
                                "KUMO_RECIPE_MISS {name} member={i} output={j}: {a} vs {b}"
                            ));
                        }
                        outside += 1;
                    }
                }
                // KUMO_RECIPE_DIAG=<file beside the oracle>: per member, the
                // distance of ours and of the oracle to another evaluation
                if let Ok(file) = std::env::var("KUMO_RECIPE_DIAG") {
                    let d: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(dir.join(file)).unwrap()).unwrap();
                    let c = d["cases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|c| c["request"]["context"] == case["request"]["context"])
                        .unwrap();
                    let alt = f32s(&c["members"][i]["output"]);
                    let dist = |v: &[f32]| {
                        v.iter()
                            .zip(&alt)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max)
                    };
                    say(&format!(
                        "KUMO_RECIPE_DIAG {name} context={} member={i} ours_vs_alt={:.3e} oracle_vs_alt={:.3e}",
                        f.context_rows,
                        dist(&out.values),
                        dist(&f32s(&expected["output"]))
                    ));
                }
                outputs.push(out.values);
            }
            // the members in one pass (slots 714-717): the bits they gave one
            // at a time, direct and through a fit and replay
            if exec.has_kumo_members() {
                let batch = m.predict_many(&tables).unwrap();
                let (ctxs, _) = m.fit_many(&tables).unwrap();
                let qv: Vec<&[f32]> = qs.iter().map(Vec::as_slice).collect();
                let mut replay = Vec::new();
                let mut at = 0;
                for ctx in &ctxs {
                    replay.extend(
                        m.query_many(ctx, &qv[at..at + ctx.members()], query.len())
                            .unwrap(),
                    );
                    at += ctx.members();
                }
                for (i, ((b, r), a)) in batch.iter().zip(&replay).zip(&outputs).enumerate() {
                    assert_eq!(&b.values, a, "{name}: member {i} batched vs alone");
                    assert_eq!(&r.values, a, "{name}: member {i} batched replay vs alone");
                }
                if std::env::var_os("KUMO_BENCH").is_some() {
                    // the recipe's GPU work as the service runs it: members
                    // one at a time against one batch, direct and cached
                    let time = |m: &mut GpuKumo, f: &mut dyn FnMut(&mut GpuKumo)| {
                        for _ in 0..2 {
                            f(m);
                        }
                        median(
                            (0..7)
                                .map(|_| {
                                    let t0 = std::time::Instant::now();
                                    f(m);
                                    t0.elapsed().as_secs_f64() * 1e3
                                })
                                .collect(),
                        )
                    };
                    let one = time(&mut m, &mut |m| {
                        for t in &tables {
                            m.predict(t).unwrap();
                        }
                    });
                    let many = time(&mut m, &mut |m| {
                        m.predict_many(&tables).unwrap();
                    });
                    let singles: Vec<_> = tables.iter().map(|t| m.fit(t).unwrap().0).collect();
                    let cached_one = time(&mut m, &mut |m| {
                        for (ctx, q) in singles.iter().zip(&qv) {
                            m.query(ctx, q, query.len()).unwrap();
                        }
                    });
                    let cached_many = time(&mut m, &mut |m| {
                        let mut at = 0;
                        for ctx in &ctxs {
                            m.query_many(ctx, &qv[at..at + ctx.members()], query.len())
                                .unwrap();
                            at += ctx.members();
                        }
                    });
                    say(&format!(
                        "KUMO_RECIPE_BENCH {name} context={} query={} cols={} members={} passes={} one_ms={one:.2} batch_ms={many:.2} cached_one_ms={cached_one:.2} cached_batch_ms={cached_many:.2}",
                        f.context_rows,
                        query.len(),
                        f.members[0].categorical.len(),
                        f.members.len(),
                        ctxs.len()
                    ));
                }
            }
            let reduced = f.reduce(&outputs, &m.config.task, query.len()).unwrap();
            let mut reduced_outside = 0usize;
            if m.config.task == Task::Regression {
                for (a, b) in reduced.iter().zip(case["predictions"].as_array().unwrap()) {
                    let b = b.as_f64().unwrap();
                    let diff = (f64::from(*a) - b).abs() / f.target_scale;
                    if diff > 1e-4 + 1e-5 * ((b - f.target_mean) / f.target_scale).abs() {
                        reduced_outside += 1;
                    }
                }
            }
            if let (Ok(file), Task::Regression) =
                (std::env::var("KUMO_RECIPE_DIAG"), &m.config.task)
            {
                let d: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(dir.join(file)).unwrap()).unwrap();
                let c = d["cases"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|c| c["request"]["context"] == case["request"]["context"])
                    .unwrap();
                let alt: Vec<f64> = c["predictions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let ours = reduced
                    .iter()
                    .zip(&alt)
                    .map(|(a, b)| (f64::from(*a) - b).abs())
                    .fold(0., f64::max);
                let oracle = case["predictions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(&alt)
                    .map(|(a, b)| (a.as_f64().unwrap() - b).abs())
                    .fold(0., f64::max);
                say(&format!(
                    "KUMO_RECIPE_DIAG_REDUCED {name} context={} ours_vs_alt={ours:.3e} oracle_vs_alt={oracle:.3e} (target units)",
                    f.context_rows
                ));
            }
            say(&format!(
                "KUMO_RECIPE {name} context={} members={} max_abs={max:.3e} outside={outside} reduced_outside={reduced_outside}",
                f.context_rows,
                f.members.len()
            ));
            failures += reduced_outside + if classification { outside } else { 0 };
        }
    }
    assert_eq!(
        failures, 0,
        "full recipe outputs outside the reference gate"
    );
}
