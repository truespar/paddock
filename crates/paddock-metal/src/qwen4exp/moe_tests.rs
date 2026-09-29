use super::*;
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> (PathBuf, Value) {
    let dir = PathBuf::from(std::env::var("PADDOCK_FLASH_NEXT_MOE_REFERENCE").unwrap());
    let m: Value =
        serde_json::from_slice(&std::fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(m["complete"], true);
    assert_eq!(m["device"], "mps");
    (dir, m)
}
fn floats(p: PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
fn uints(p: PathBuf) -> Vec<u32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}
fn upload(d: &MetalDevice, x: &[f32]) -> Buffer {
    d.upload(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn close(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let e = (a - b).abs();
        max = max.max(e);
        // Declared before the first GPU run. Allows F32 reduction order
        // through HC/router/expert contractions, never a changed expert set.
        assert!(
            a.is_finite() && b.is_finite() && e <= 2e-4 + 3e-5 * b.abs(),
            "{label}[{i}]: {a} vs {b}, error {e}"
        );
    }
    max
}
fn rows<T: Copy>(all: &[T], ids: &[usize], width: usize) -> Vec<T> {
    ids.iter()
        .flat_map(|&r| all[r * width..(r + 1) * width].iter().copied())
        .collect()
}

#[test]
fn memory_bounds_are_exact_and_fail_before_allocation() {
    let d = MetalDevice::new(Some(
        Workspace::bytes(super::super::affine::MAX_ROWS).unwrap() as u64,
    ))
    .unwrap();
    for n in [0, super::super::affine::MAX_ROWS + 1, usize::MAX] {
        assert!(Workspace::bytes(n).is_err());
        assert!(Workspace::new(&d, n).is_err());
        assert_eq!(d.allocated_bytes(), 0);
    }
    for n in [1, 4, 9, 31, 128, 256, 512, 1024, 2048] {
        let s = Workspace::new(&d, n).unwrap();
        assert_eq!(d.allocated_bytes(), Workspace::bytes(n).unwrap() as u64);
        drop(s);
        assert_eq!(d.allocated_bytes(), 0);
    }
}

#[test]
#[ignore = "real elected GGUF + independent MPS full-FFN fixtures"]
fn complete_hc_routed_shared_ffn_matches_mps_across_schedules() {
    let (dir, m) = fixture();
    let map = MappedGguf::open(&PathBuf::from(
        m["model_files"][0]["path"].as_str().unwrap(),
    ))
    .unwrap();
    let d = MetalDevice::new(Some(3 << 30)).unwrap();
    assert!(Weights::load(&d, &map, 48).is_err());
    assert_eq!(d.allocated_bytes(), 0);
    for case in m["cases"].as_array().unwrap() {
        let layer = case["layer"].as_u64().unwrap() as usize;
        let key = case["id"].as_str().unwrap();
        let load = |name| floats(dir.join(format!("{key}.{name}.f32")));
        let (hidden, mixed, output, combined, act, down, shared, scale, prob) = (
            load("hidden"),
            load("mixed"),
            load("output"),
            load("combined"),
            load("act"),
            load("down"),
            load("shared"),
            load("shared_scale"),
            load("weights"),
        );
        let selected = uints(dir.join(format!("{key}.ids.u32")));
        let logits = load("logits");
        let w = Weights::load(&d, &map, layer).unwrap();
        let before = d.allocated_bytes();
        let s = Workspace::new(&d, 128).unwrap();
        let hc = residual::Workspace::new(&d, 128).unwrap();
        assert_eq!(
            d.allocated_bytes() - before,
            (Workspace::bytes(128).unwrap() + residual::Workspace::bytes(128).unwrap()) as u64
        );
        // Isolate routing with identical scores, independently of HC/matmul
        // reduction order. Every rank/tie must match exactly. The shared-gate
        // dummy does not affect topk; the full graph checks its real value.
        for first in [0, 128] {
            let n = (133 - first).min(128);
            let input = logits[first * 512..(first + n) * 512]
                .chunks(512)
                .flat_map(|r| r.iter().copied().chain(std::iter::once(0.)))
                .collect::<Vec<_>>();
            let input = upload(&d, &input);
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "iq_route512",
                &[&input, &s.ids, &s.weights, &s.shared_scale, &s.invalid],
                &[n as u32],
                [n, 1, 1],
                32,
            );
            cmd.submit().unwrap().wait().unwrap();
            assert_eq!(
                unsafe { s.ids.read_u32(n * 10) },
                selected[first * 10..(first + n) * 10]
            );
            close(
                &unsafe { s.weights.read_f32(0, n * 10) },
                &prob[first * 10..(first + n) * 10],
                "identical-input routing weights",
            );
        }
        // Permute logical rows: these stateless FFNs may not depend on row
        // ordering, request identity or whether a hot companion was admitted.
        let order = (0..133).map(|i| (i * 37 + 132) % 133).collect::<Vec<_>>();
        for chunk in [1, 4, 8, 9, 16, 31, 32, 33, 64, 128] {
            let mut error = 0f32;
            let mut reorders = 0;
            for ids in order.chunks(chunk) {
                let n = ids.len();
                let mut h = rows(&hidden, ids, residual::WIDE);
                h.extend([9876.5; 17]);
                let h = upload(&d, &h);
                let cmd = d.begin().unwrap();
                w.encode_ffn(&cmd, &h, &hc, &s, n, false).unwrap();
                cmd.submit().unwrap().wait().unwrap();
                assert_eq!(unsafe { s.invalid.read_u32(n) }, vec![0; n]);
                let actual_ids = unsafe { s.ids.read_u32(n * 10) };
                let expected_ids = rows(&selected, ids, 10);
                let mut aligned_prob = Vec::with_capacity(n * 10);
                for (r, &source) in ids.iter().enumerate() {
                    let a = &actual_ids[r * 10..(r + 1) * 10];
                    let e = &expected_ids[r * 10..(r + 1) * 10];
                    reorders += usize::from(a != e);
                    let (mut aset, mut eset) = (a.to_vec(), e.to_vec());
                    aset.sort_unstable();
                    eset.sort_unstable();
                    assert_eq!(
                        aset, eset,
                        "selected expert set layer={layer} chunk={chunk} source={source}"
                    );
                    // A rank swap inside the selected set is not an admission
                    // change. Keep exact set equality and match weights by id;
                    // the unchanged full output gate still prices fold order.
                    for expert in a {
                        aligned_prob.push(
                            prob[source * 10 + e.iter().position(|id| id == expert).unwrap()],
                        );
                    }
                }
                error = error.max(close(
                    &unsafe { s.output.read_f32(0, n * WIDTH) },
                    &rows(&output, ids, WIDTH),
                    "FFN output",
                ));
                close(
                    &unsafe { hc.mixed.read_f32(0, n * WIDTH) },
                    &rows(&mixed, ids, WIDTH),
                    "HC mixed",
                );
                close(
                    &unsafe { s.weights.read_f32(0, n * 10) },
                    &aligned_prob,
                    "router probabilities",
                );
                close(
                    &unsafe { s.shared_scale.read_f32(0, n) },
                    &rows(&scale, ids, 1),
                    "shared gate",
                );
                close(
                    &unsafe { s.shared_output.read_f32(0, n * WIDTH) },
                    &rows(&shared, ids, WIDTH),
                    "shared output",
                );
                close(
                    &unsafe { h.read_f32(0, n * residual::WIDE) },
                    &rows(&combined, ids, residual::WIDE),
                    "HC combined",
                );
                assert_eq!(unsafe { h.read_f32(n * residual::WIDE, 17) }, [9876.5; 17]);
            }
            eprintln!(
                "Flash Next full FFN layer={layer} chunk={chunk} max_abs={error} internal_rank_reorders={reorders}"
            );
        }
        // Independently gate fused activation/down and both occupancy arms,
        // including >64 hot rows. Use the exact MPS mixed input here so this
        // distinguishes expert arithmetic from upstream HC perturbations.
        for (grouped, tensor) in [(false, false), (true, false), (true, true)] {
            let x = upload(&d, &mixed[..128 * WIDTH]);
            unsafe {
                s.ids.write_u32(&selected[..128 * 10]);
            }
            let cmd = d.begin().unwrap();
            w.encode_experts(&cmd, &x, &s, 128, grouped, tensor);
            cmd.submit().unwrap().wait().unwrap();
            let ae = close(
                &unsafe { s.act.read_f32(0, 128 * 10 * FF) },
                &act[..128 * 10 * FF],
                "fused expert activation",
            );
            let de = close(
                &unsafe { s.down.read_f32(0, 128 * 10 * WIDTH) },
                &down[..128 * 10 * WIDTH],
                "expert down",
            );
            eprintln!(
                "Flash Next experts layer={layer} grouped={grouped} tensor={tensor} act={ae} down={de}"
            );
        }
        // Fail invalid bounds before encoding any work, not just before the
        // last projection. Caller retains the unchanged input and output.
        let h = upload(&d, &hidden[..residual::WIDE]);
        let prior = unsafe { s.output.read_u32(128 * WIDTH) };
        let cmd = d.begin().unwrap();
        for n in [0, 2, 129, usize::MAX] {
            assert!(w.encode_ffn(&cmd, &h, &hc, &s, n, false).is_err());
        }
        cmd.submit().unwrap().wait().unwrap();
        assert_eq!(
            unsafe { h.read_f32(0, residual::WIDE) },
            hidden[..residual::WIDE]
        );
        assert_eq!(unsafe { s.output.read_u32(128 * WIDTH) }, prior);
        let small_hc = residual::Workspace::new(&d, 1).unwrap();
        let two = upload(&d, &hidden[..2 * residual::WIDE]);
        let cmd = d.begin().unwrap();
        assert!(w.encode_ffn(&cmd, &two, &small_hc, &s, 2, false).is_err());
        cmd.submit().unwrap().wait().unwrap();
        assert_eq!(
            unsafe { two.read_f32(0, 2 * residual::WIDE) },
            hidden[..2 * residual::WIDE]
        );
        assert_eq!(unsafe { s.output.read_u32(128 * WIDTH) }, prior);
        drop((small_hc, two));
        // Nonfinite activations must not yield an out-of-range expert address
        // or corrupt healthy neighbors, on both SIMD and grouped paths.
        for n in [4, 9] {
            let mut input = mixed[..n * WIDTH].to_vec();
            input[WIDTH] = f32::NAN;
            let x = upload(&d, &input);
            let cmd = d.begin().unwrap();
            w.encode(&cmd, &x, &s, n).unwrap();
            cmd.submit().unwrap().wait().unwrap();
            let bad = unsafe { s.invalid.read_u32(n) };
            assert_eq!(bad.iter().sum::<u32>(), 1);
            assert_eq!(bad[1], 1);
            let out = unsafe { s.output.read_f32(0, n * WIDTH) };
            for (row, values) in out.chunks(WIDTH).enumerate() {
                if row == 1 {
                    assert!(values.iter().all(|v| v.is_nan()));
                } else {
                    assert!(values.iter().all(|v| v.is_finite()));
                }
            }
        }
        drop((h, hc, s, w));
        assert_eq!(d.allocated_bytes(), 0);
    }
}

#[test]
#[ignore = "internal route diagnostic only; run without Metal validation, no rival claim"]
fn time_identical_expert_routes() {
    let (dir, m) = fixture();
    let map = MappedGguf::open(&PathBuf::from(
        m["model_files"][0]["path"].as_str().unwrap(),
    ))
    .unwrap();
    let d = MetalDevice::new(Some(3 << 30)).unwrap();
    let w = Weights::load(&d, &map, 0).unwrap();
    let s = Workspace::new(&d, 128).unwrap();
    let mixed = floats(dir.join("l0.mixed.f32"));
    let selected = uints(dir.join("l0.ids.u32"));
    for (label, order) in [
        ("spread", (0..128).map(|i| i % 64).collect::<Vec<_>>()),
        ("hot", vec![0; 128]),
    ] {
        for n in [1, 4, 9, 32, 128] {
            let x = upload(&d, &rows(&mixed, &order[..n], WIDTH));
            unsafe {
                s.ids.write_u32(&rows(&selected, &order[..n], 10));
            }
            let routes = [
                ("simd", false, false),
                ("adaptive", true, false),
                ("tensor", true, true),
            ];
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            // Interleave/rotate and warm all routes. Fixed-order runs can
            // charge the first route for device ramp-up and cache coldness.
            for round in 0..15 {
                for j in 0..3 {
                    let index = (round + j) % 3;
                    let (_, grouped, tensor) = routes[index];
                    let start = std::time::Instant::now();
                    let cmd = d.begin().unwrap();
                    w.encode_experts(&cmd, &x, &s, n, grouped, tensor);
                    cmd.submit().unwrap().wait().unwrap();
                    if round >= 3 {
                        times[index].push(start.elapsed().as_secs_f64() * 1000.);
                    }
                }
            }
            for (i, (name, _, _)) in routes.iter().enumerate() {
                let ms = &mut times[i];
                ms.sort_by(f64::total_cmp);
                eprintln!(
                    "MoE diagnostic {label} rows={n} route={name} median_ms={:.4}",
                    (ms[5] + ms[6]) * 0.5
                );
            }
        }
    }
}

#[test]
#[ignore = "elected full GGUF; sequentially uploads every FFN layer, not model execution"]
fn all_48_ffn_weights_load_and_release_at_exact_raw_size() {
    let (_, m) = fixture();
    let map = MappedGguf::open(&PathBuf::from(
        m["model_files"][0]["path"].as_str().unwrap(),
    ))
    .unwrap();
    let d = MetalDevice::new(Some(3 << 30)).unwrap();
    for layer in 0..48 {
        let prefix = format!("blk.{layer}.");
        let expected = map
            .tensor_infos()
            .filter(|t| {
                t.name
                    .strip_prefix(&prefix)
                    .is_some_and(|name| name.starts_with("ffn_") || name.starts_with("hc_ffn_"))
            })
            .map(|t| map.tensor_bytes(&t.name).unwrap().1.len() as u64)
            .sum::<u64>();
        let w = Weights::load(&d, &map, layer).unwrap();
        assert_eq!(d.allocated_bytes(), expected, "layer={layer}");
        drop(w);
        assert_eq!(d.allocated_bytes(), 0);
        eprintln!("Flash Next FFN load layer={layer} exact_bytes={expected}");
    }
}
