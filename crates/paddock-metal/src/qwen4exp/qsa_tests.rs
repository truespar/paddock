use super::{qsa, residual::WIDTH};
use crate::device::{Buffer, MetalDevice};
use paddock_models::mapped::MappedGguf;
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> (PathBuf, Value) {
    let dir = PathBuf::from(std::env::var("PADDOCK_FLASH_NEXT_QSA_REFERENCE").unwrap());
    let m: Value =
        serde_json::from_slice(&std::fs::read(dir.join("results.json")).unwrap()).unwrap();
    assert_eq!(m["complete"], true);
    assert_eq!(m["device"], "mps");
    assert_eq!(m["index_cache"], "f32");
    assert_eq!(m["kv_cache"], "f16");
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
fn upload(d: &MetalDevice, v: &[f32]) -> Buffer {
    d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn upload_u(d: &MetalDevice, v: &[u32]) -> Buffer {
    d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .unwrap()
}
fn close(a: &[f32], b: &[f32], label: &str) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let e = (a - b).abs();
        max = max.max(e);
        // Predeclared allowance includes RoPE transcendental/reduction order
        // and downstream F16-cache rounding, not full-generation parity.
        assert!(
            a.is_finite() && b.is_finite() && e <= 2e-4 + 3e-5 * b.abs(),
            "{label}[{i}]: {a} vs {b}, error {e}"
        );
    }
    max
}

#[test]
#[ignore = "requires independent MPS QSA fixtures"]
fn radix_selection_matches_gpu_stable_topk_at_full_context() {
    let (dir, m) = fixture();
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    for c in m["selection"].as_array().unwrap() {
        let id = c["id"].as_u64().unwrap();
        let blocks = c["blocks"].as_u64().unwrap() as usize;
        let scores = upload(&d, &floats(dir.join(format!("select-{id}.scores.f32"))));
        // Zero blocks uses a one-token incomplete tail, still no complete block.
        let meta = upload_u(&d, &[0, (blocks * 4).saturating_sub(1) as u32, 0, 1]);
        let ids = upload_u(&d, &vec![0x12345678; 512 + 17]);
        let count = d.alloc(4).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4s_select",
            &[&scores, &meta, &ids, &count],
            &[1, blocks.max(1) as u32, 1, 1, 4],
            [1, 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        let got = unsafe { ids.read_u32(529) };
        assert_eq!(
            &got[..512],
            uints(dir.join(format!("select-{id}.ids.u32"))),
            "blocks={blocks}, case={id}"
        );
        assert_eq!(&got[512..], &[0x12345678; 17]);
        assert_eq!(
            unsafe { count.read_u32(1) }[0],
            c["count"].as_u64().unwrap() as u32
        );
        eprintln!("QSA exact radix selection: case={id} blocks={blocks}");
    }
    assert_eq!(d.allocated_bytes(), 0);
}

#[test]
fn selection_zero_ties_and_invalid_scores_fail_closed() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let meta = upload_u(&d, &[0, 2051, 0, 1]);
    let ids = d.alloc(512 * 4).unwrap();
    let count = d.alloc(4).unwrap();
    for bad in [None, Some(f32::NAN), Some(f32::INFINITY), Some(-1.)] {
        let mut values: Vec<_> = (0..513)
            .map(|i| if i % 2 == 0 { 0. } else { -0. })
            .collect();
        if let Some(bad) = bad {
            values[512] = bad;
        }
        let scores = upload(&d, &values);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4s_select",
            &[&scores, &meta, &ids, &count],
            &[1, 513, 1, 1, 4],
            [1, 1, 1],
            256,
        );
        cmd.submit().unwrap().wait().unwrap();
        if bad.is_some() {
            assert_eq!(unsafe { count.read_u32(1) }, [u32::MAX]);
            assert_eq!(unsafe { ids.read_u32(512) }, vec![u32::MAX; 512]);
        } else {
            assert_eq!(unsafe { count.read_u32(1) }, [512]);
            assert_eq!(unsafe { ids.read_u32(512) }, (0..512).collect::<Vec<_>>());
        }
    }
}

#[test]
fn compact_mlx_attention_matches_legacy_partitions_exactly() {
    attention_staging_contracts(false);
}

#[test]
fn cache_only_finite_checks_written_paged_rows_and_completed_index_blocks() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let meta = upload_u(&d, &[1, 3, 0, 2, 1, 4, 0, 2, 0, 16, 2, 3]);
    let pages = upload_u(&d, &[2, 1, 3, 0]);
    let bad = upload_u(&d, &[0]);
    let p = [2, 8, 3, 2, 4];
    for case in 0..7 {
        let mut keys = vec![half::bf16::ONE; 4 * 16 * 512];
        let mut values = keys.clone();
        let mut raw = vec![1.; 3 * 128];
        let mut pooled = vec![1.; 2 * 8 * 128];
        match case {
            1 => keys[(3 * 16 + 3) * 512 + 511] = half::bf16::NAN,
            2 => values[16 * 512] = half::bf16::INFINITY,
            3 => raw[128 + 127] = f32::NEG_INFINITY,
            4 => pooled[8 * 128] = f32::NAN,
            // Neither an untouched page nor an incomplete index block is read.
            5 => {
                keys[0] = half::bf16::NAN;
                pooled[9 * 128] = f32::NAN;
            }
            6 => unsafe { bad.write_u32(&[4]) },
            _ => (),
        }
        if case != 6 {
            unsafe { bad.write_u32(&[0]) };
        }
        let keys = d
            .upload(
                &keys
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let values = d
            .upload(
                &values
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let raw = upload(&d, &raw);
        let pooled = upload(&d, &pooled);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "q4b_cache_finite",
            &[&keys, &values, &raw, &pooled, &meta, &pages, &bad],
            &p,
            [6, 1, 1],
            256,
        );
        cmd.finish().unwrap();
        assert_eq!(
            unsafe { bad.read_u32(1) }[0],
            match case {
                1..=4 => 8,
                6 => 4,
                _ => 0,
            },
            "case={case}"
        );
    }
}

#[test]
#[ignore = "isolated GPU timing for exact Flash Next attention staging routes"]
fn mlx_attention_staging_cost() {
    attention_staging_contracts(true);
}

#[test]
fn local_mlx_attention_preserves_partials_and_guards() {
    attention_local_contracts(false, false);
}

#[test]
#[ignore = "isolated on-chip attention timing; not a serving benchmark"]
fn mlx_attention_local_cost() {
    attention_local_contracts(true, false);
}

#[test]
fn direct_mlx_attention_preserves_partials_and_guards() {
    attention_local_contracts(false, true);
}

#[test]
#[ignore = "exact contiguous-run attention GPU costs; not a serving benchmark"]
fn mlx_attention_direct_runs_cost() {
    attention_local_contracts(true, true);
}

fn attention_local_contracts(benchmark: bool, direct: bool) {
    let d = MetalDevice::new(Some(768 << 20)).unwrap();
    let capacity = 2048usize;
    let slots = 64usize;
    let pages_per_slot = 1024usize;
    let length = pages_per_slot * 16;
    let values = |count: usize, scale: f32| {
        (0..count)
            .map(|i| (i as f32 * scale).sin() * 0.5)
            .collect::<Vec<_>>()
    };
    let bf = |scale| {
        d.upload(
            &values(length * qsa::KV, scale)
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let k = bf(0.013);
    let v = bf(0.017);
    let pages = upload_u(
        &d,
        &(0..slots * pages_per_slot)
            // Every slot has a different permutation: wrong-slot reads must
            // not pass simply because all tables point to identical pages.
            .map(|i| {
                if direct && (i / pages_per_slot).is_multiple_of(2) {
                    return ((i % pages_per_slot + i / pages_per_slot * 32) % pages_per_slot)
                        as u32;
                }
                ((pages_per_slot - 1 - i % pages_per_slot + i / pages_per_slot * 37)
                    % pages_per_slot) as u32
            })
            .collect::<Vec<_>>(),
    );
    let units = capacity + slots * 8 * 3;
    let scratch_len = units * 2 * 8192;
    let parts_len = units * qsa::HEADS * 258;
    let scratch = upload(&d, &vec![-8765.; scratch_len + 17]);
    let parts = upload(&d, &vec![f32::NAN; parts_len + 17]);
    for (n, bf_query) in [1, 4, 8, 17, 32, 64, 128, 512, 1024, 2048]
        .into_iter()
        .flat_map(|n| [(n, false), (n, true)])
    {
        let mut query_values = values(n * qsa::Q, 0.019);
        if bf_query {
            for value in &mut query_values {
                *value = half::bf16::from_f32(*value).to_f32();
            }
        }
        let query = upload(&d, &query_values);
        let qg = upload(&d, &values(n * qsa::Q * 2, 0.023));
        let out = upload(&d, &vec![-9876.; n * qsa::Q + 17]);
        for (position, ragged) in [0usize, 7, 127, 2047, 2050, 8191, 16383]
            .into_iter()
            .map(|p| (p, false))
            .chain([(2050, true), (16383, true)])
        {
            if (n > 64 && ![127, 8191, 16383].contains(&position)) || (benchmark && ragged) {
                continue;
            }
            let position_at = |i: usize| position - if ragged { i % 19 } else { 0 };
            let meta = upload_u(
                &d,
                &(0..n)
                    .flat_map(|i| {
                        [
                            (slots - 1 - i % slots) as u32,
                            position_at(i) as u32,
                            i as u32,
                            (i + 1) as u32,
                        ]
                    })
                    .collect::<Vec<_>>(),
            );
            let selected = upload_u(
                &d,
                &(0..n * 512)
                    .map(|i| {
                        let blocks = (position_at(i / 512) + 1) / 4;
                        if blocks > 512 {
                            ((i % 512) * blocks / 512 + (i / 512) % (blocks / 512)) as u32
                        } else {
                            (i % 512).min(blocks.saturating_sub(1)) as u32
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let counts = upload_u(
                &d,
                &(0..n)
                    .map(|i| ((position_at(i) + 1) / 4).min(512) as u32)
                    .collect::<Vec<_>>(),
            );
            // Four-part decode, sliced one-part prefill, and mixed masks,
            // including slots above 31. Never re-elect using physical rows.
            for mask in [[u32::MAX; 2], [0; 2], [0x55555555, 0xaaaaaaaa]] {
                if n > 64 && mask != [0; 2] {
                    continue;
                }
                let p = [
                    pages_per_slot as u32,
                    (pages_per_slot * 4) as u32,
                    n as u32,
                    slots as u32,
                    4,
                    mask[0],
                    mask[1],
                    capacity as u32,
                ];
                let mut reference: Option<(Vec<u32>, Vec<u32>)> = None;
                let kernels: &[&str] = if direct {
                    &[
                        "q4b_attention_local_vpad1",
                        "q4b_attention_direct_values",
                        "q4b_attention_direct_keys",
                        "q4b_attention_direct_runs",
                    ]
                } else {
                    &[
                        "q4b_attention_compact",
                        "q4b_attention_local",
                        "q4b_attention_local_pad",
                        "q4b_attention_local_wide",
                        "q4b_attention_local_gather",
                        "q4b_attention_local_vpad1",
                    ]
                };
                let mut times = vec![Vec::new(); kernels.len()];
                let routes = kernels.len();
                for round in 0..if benchmark { 7 } else { 1 } {
                    for route in (0..routes).map(|r| (r + round) % routes) {
                        if round == 0 {
                            // Poison independently after the previous command
                            // has completed. A missing write must not inherit
                            // the oracle's partials or joined output.
                            unsafe {
                                parts.write_u32(&vec![
                                    f32::NAN.to_bits() + route as u32;
                                    parts_len + 17
                                ]);
                                out.write_u32(&vec![(-9876f32).to_bits(); n * qsa::Q + 17]);
                            }
                        }
                        let cmd = d.begin().unwrap();
                        cmd.dispatch(
                            kernels[route],
                            &[
                                &query, &k, &v, &meta, &pages, &selected, &counts, &scratch, &parts,
                            ],
                            &p,
                            [2, n, 4],
                            128,
                        );
                        cmd.dispatch(
                            "q4b_join_gate_compact",
                            &[&parts, &qg, &out, &meta],
                            &p,
                            [n * qsa::HEADS, 1, 1],
                            32,
                        );
                        let elapsed = cmd.finish().unwrap();
                        if round > 0 {
                            times[route].push(elapsed);
                        }
                        let actual = unsafe { out.read_u32(n * qsa::Q + 17) };
                        assert_eq!(&actual[n * qsa::Q..], &[(-9876f32).to_bits(); 17]);
                        assert!(
                            unsafe { parts.read_f32(parts_len, 17) }
                                .iter()
                                .all(|v| v.is_nan())
                        );
                        let mut active = Vec::new();
                        for row in 0..n {
                            let slot = slots - 1 - row % slots;
                            let splits = if mask[slot / 32] & (1 << (slot % 32)) != 0 {
                                4
                            } else {
                                1
                            };
                            for split in 0..splits {
                                let unit = if split == 0 {
                                    row
                                } else {
                                    capacity + 3 * (slot * 8) + split - 1
                                };
                                active.extend(
                                    unsafe {
                                        parts.read_f32(unit * qsa::HEADS * 258, qsa::HEADS * 258)
                                    }
                                    .into_iter()
                                    .map(f32::to_bits),
                                );
                            }
                        }
                        if let Some((expected, partials)) = &reference {
                            if actual != *expected || active != *partials {
                                let counts = (0..258)
                                    .map(|offset| {
                                        active
                                            .iter()
                                            .zip(partials)
                                            .enumerate()
                                            .filter(|(i, (a, b))| i % 258 == offset && a != b)
                                            .count()
                                    })
                                    .collect::<Vec<_>>();
                                eprintln!(
                                    "DIRECT_DIFFERENCES kernel={} output={} maxima={} denominator={} partial_values={} max_error={}",
                                    kernels[route],
                                    actual.iter().zip(expected).filter(|(a, b)| a != b).count(),
                                    counts[256],
                                    counts[257],
                                    counts[..256].iter().sum::<usize>(),
                                    active
                                        .iter()
                                        .zip(partials)
                                        .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b))
                                            .abs())
                                        .fold(0f32, f32::max)
                                );
                            }
                            assert!(
                                actual == *expected,
                                "output n={n} position={position} ragged={ragged} route={route}"
                            );
                            assert!(
                                active == *partials,
                                "partials n={n} position={position} ragged={ragged} route={route}"
                            );
                        } else {
                            reference = Some((actual, active));
                        }
                    }
                }
                assert_eq!(unsafe { out.read_f32(n * qsa::Q, 17) }, [-9876.; 17]);
                assert_eq!(unsafe { scratch.read_f32(scratch_len, 17) }, [-8765.; 17]);
                assert!(
                    unsafe { parts.read_f32(parts_len, 17) }
                        .iter()
                        .all(|v| v.is_nan())
                );
                if benchmark {
                    eprintln!(
                        "FLASH_LOCAL_COST {}",
                        serde_json::json!({"rows":n,"position":position,"ragged":ragged,"bf16_query":bf_query,
                    "mask":mask,"kernels":kernels,"gpu_seconds":times,"partials_exact":true})
                    );
                }
            }
        }
    }
}

fn attention_staging_contracts(benchmark: bool) {
    // Independent legacy storage/kernel is retained as the arithmetic oracle.
    // Sparse boundary, tail rows, four-part decode, sliced one-part prefill,
    // non-monotonic slots and both halves of the logical mask are covered.
    let d = MetalDevice::new(Some(768 << 20)).unwrap();
    let pages_per_slot = 256usize;
    let length = pages_per_slot * 16;
    let values = |count: usize, scale: f32| {
        (0..count)
            .map(|i| (i as f32 * scale).sin() * 0.5)
            .collect::<Vec<_>>()
    };
    let bf = |count: usize, scale| {
        d.upload(
            &values(count, scale)
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let k = bf(length * qsa::KV, 0.013);
    let v = bf(length * qsa::KV, 0.017);
    for (capacity, slots, order) in [
        (256usize, 4usize, [3usize, 1, 2, 0]),
        (1024, 64, [63, 1, 33, 0]),
    ] {
        let pages = upload_u(
            &d,
            &(0..slots * pages_per_slot)
                .map(|i| (pages_per_slot - 1 - i % pages_per_slot) as u32)
                .collect::<Vec<_>>(),
        );
        let n = if benchmark { capacity } else { 99 };
        let query = upload(&d, &values(n * qsa::Q, 0.019));
        let qg = upload(&d, &values(n * qsa::Q * 2, 0.023));
        let legacy_scratch = d.alloc(n * 2 * 4 * 8192 * 4).unwrap();
        let legacy_parts = d.alloc(n * qsa::HEADS * 4 * 258 * 4).unwrap();
        let units = capacity + slots * 8 * 3;
        let scratch_len = units * 2 * 8192;
        let parts_len = units * qsa::HEADS * 258;
        let compact_scratch = upload(&d, &vec![-8765.; scratch_len + 17]);
        let compact_parts = upload(&d, &vec![f32::NAN; parts_len + 17]);
        for position in [7usize, 128, 2047, 2050, 4095] {
            let mut meta = Vec::new();
            let spans = if benchmark {
                [capacity / 4; 4]
            } else {
                [8, 1, 17, 73]
            };
            let mut start = 0;
            for (slot, count) in order.into_iter().zip(spans) {
                for _ in 0..count {
                    meta.extend([
                        slot as u32,
                        position as u32,
                        start as u32,
                        (start + count) as u32,
                    ]);
                }
                start += count;
            }
            let meta = upload_u(&d, &meta);
            let blocks = (position + 1) / 4;
            let selected = upload_u(
                &d,
                &(0..n * 512)
                    .map(|i| (i % 512).min(blocks.saturating_sub(1)) as u32)
                    .collect::<Vec<_>>(),
            );
            let counts = upload_u(&d, &vec![blocks.min(512) as u32; n]);
            let mut p = [
                pages_per_slot as u32,
                (pages_per_slot * 4) as u32,
                n as u32,
                slots as u32,
                4,
                0,
                0,
                capacity as u32,
            ];
            // The eight-row span decodes, while the one physical row belongs
            // to a wide logical prefill and must NOT use four partitions.
            if !benchmark {
                p[5 + order[0] / 32] = 1 << (order[0] % 32);
            }
            let old = upload(&d, &vec![-9876.; n * qsa::Q + 17]);
            let new = upload(&d, &vec![-9876.; n * qsa::Q + 17]);
            let mut times = [Vec::new(), Vec::new()];
            let kernels = ["q4b_attention_contract", "q4b_attention_compact"];
            for round in 0..if benchmark { 6 } else { 1 } {
                for route in (0..2).map(|r| (r + round) % 2) {
                    let (scratch, parts) = if route == 0 {
                        (&legacy_scratch, &legacy_parts)
                    } else {
                        (&compact_scratch, &compact_parts)
                    };
                    let cmd = d.begin().unwrap();
                    cmd.dispatch(
                        kernels[route],
                        &[
                            &query, &k, &v, &meta, &pages, &selected, &counts, scratch, parts,
                        ],
                        &p,
                        [2, n, 4],
                        128,
                    );
                    let elapsed = cmd.finish().unwrap();
                    if round > 0 {
                        times[route].push(elapsed);
                    }
                    let cmd = d.begin().unwrap();
                    if route == 0 {
                        cmd.dispatch(
                            "q4b_join_gate",
                            &[parts, &qg, &old],
                            &p,
                            [n * qsa::HEADS, 1, 1],
                            32,
                        );
                    } else {
                        cmd.dispatch(
                            "q4b_join_gate_compact",
                            &[parts, &qg, &new, &meta],
                            &p,
                            [n * qsa::HEADS, 1, 1],
                            32,
                        );
                    }
                    cmd.finish().unwrap();
                    if route != 0 {
                        let actual = unsafe { new.read_f32(0, n * qsa::Q + 17) };
                        let expected = unsafe { old.read_f32(0, n * qsa::Q + 17) };
                        let diffs = actual
                            .iter()
                            .zip(&expected)
                            .filter(|(a, b)| a.to_bits() != b.to_bits())
                            .count();
                        assert_eq!(diffs, 0, "route={route} position={position} rows={n}");
                    }
                }
            }
            if benchmark {
                eprintln!(
                    "FLASH_ATTENTION {}",
                    serde_json::json!({"rows":n,"position":position,"kernels":kernels,"gpu_seconds":times})
                );
            }
            let expected = unsafe { old.read_f32(0, n * qsa::Q + 17) };
            let actual = unsafe { new.read_f32(0, n * qsa::Q + 17) };
            assert!(actual.iter().all(|v| v.is_finite()));
            assert_eq!(actual, expected, "slots={slots} position={position}");
            assert_eq!(&actual[n * qsa::Q..], &[-9876.; 17]);
            assert_eq!(
                unsafe { compact_scratch.read_f32(scratch_len, 17) },
                [-8765.; 17]
            );
            assert!(
                unsafe { compact_parts.read_f32(parts_len, 17) }
                    .iter()
                    .all(|v| v.is_nan())
            );
        }
    }
}

#[test]
fn mlx_workspaces_have_exact_bounded_allocation_ledgers() {
    let d = MetalDevice::new(Some(1 << 30)).unwrap();
    for (rows, slots) in [(1, 1), (8, 1), (128, 64), (256, 4), (512, 4), (1024, 64)] {
        let before = d.allocated_bytes();
        let q = qsa::Workspace::new_mlx(&d, rows, slots, 256).unwrap();
        assert_eq!(
            d.allocated_bytes() - before,
            qsa::Workspace::mlx_bytes(rows, slots, 256) as u64
        );
        assert!(
            qsa::Workspace::mlx_bytes(rows, slots, 256) <= qsa::Workspace::bytes(rows, slots, 256)
        );
        drop(q);
        let dn = super::deltanet::Workspace::new_mlx(&d, rows, slots).unwrap();
        assert_eq!(
            d.allocated_bytes() - before,
            super::deltanet::Workspace::mlx_bytes(rows, slots) as u64
        );
        drop(dn);
        assert_eq!(d.allocated_bytes(), before);
    }
}

struct Stream {
    length: usize,
    probes: Vec<usize>,
    input: Vec<f32>,
    query: Vec<f32>,
    index_query: Vec<f32>,
    raw: Vec<f32>,
    pooled: Vec<f32>,
    output: Vec<f32>,
    selected: Vec<u32>,
    counts: Vec<u32>,
}
fn run(
    d: &MetalDevice,
    s: &mut qsa::State,
    w: &qsa::Weights,
    streams: &[Stream],
    rows: &[(usize, usize)],
) -> f32 {
    let x = upload(
        d,
        &rows
            .iter()
            .flat_map(|&(slot, pos)| {
                streams[slot].input[pos * WIDTH..(pos + 1) * WIDTH]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>(),
    );
    let y = upload(d, &vec![-9876.5; rows.len() * WIDTH + 17]);
    s.run(d, w, rows, &x, &y).unwrap();
    let mut worst = 0f32;
    let counts = unsafe { s.scratch.counts.read_u32(rows.len()) };
    let ids = unsafe { s.scratch.selected.read_u32(rows.len() * 512) };
    for (r, &(slot, pos)) in rows.iter().enumerate() {
        let t = &streams[slot];
        for (b, expected, width, label) in [
            (&s.scratch.query, &t.query, 6144, "query"),
            (&s.scratch.index_query, &t.index_query, 512, "index query"),
            (&s.scratch.raw, &t.raw, 128, "raw index keys"),
        ] {
            close(
                &unsafe { b.read_f32(r * width, width) },
                &expected[pos * width..(pos + 1) * width],
                label,
            );
        }
        if let Some(probe) = t.probes.iter().position(|&p| p == pos) {
            assert_eq!(counts[r], t.counts[probe]);
            assert_eq!(
                &ids[r * 512..(r + 1) * 512],
                &t.selected[probe * 512..(probe + 1) * 512],
                "selected blocks slot={slot} pos={pos}"
            );
            worst = worst.max(close(
                &unsafe { y.read_f32(r * WIDTH, WIDTH) },
                &t.output[probe * WIDTH..(probe + 1) * WIDTH],
                "attention output",
            ));
        }
    }
    assert_eq!(unsafe { y.read_f32(rows.len() * WIDTH, 17) }, [-9876.5; 17]);
    worst
}

#[test]
#[ignore = "requires elected Flash Next GGUF and independent MPS QSA fixtures"]
fn actual_qsa_sparse_boundary_mixed_prefill_decode_reset_and_fork() {
    let (dir, m) = fixture();
    let map = MappedGguf::open(&PathBuf::from(
        std::env::var("PADDOCK_FLASH_NEXT_MODEL").unwrap(),
    ))
    .unwrap();
    let d = MetalDevice::new(Some(512 << 20)).unwrap();
    for layer in [0, 2, 48, usize::MAX] {
        assert!(qsa::Weights::load(&d, &map, layer).is_err());
    }
    for layer in [3, 47] {
        let w = qsa::Weights::load(&d, &map, layer).unwrap();
        let streams: Vec<_> = m["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c["layer"] == layer)
            .map(|c| {
                let id = c["id"].as_str().unwrap();
                let f = |suffix| floats(dir.join(format!("{id}.{suffix}.f32")));
                Stream {
                    length: c["length"].as_u64().unwrap() as usize,
                    probes: c["probes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect(),
                    input: f("input"),
                    query: f("query"),
                    index_query: f("index_query"),
                    raw: f("raw"),
                    pooled: f("pooled"),
                    output: f("output"),
                    selected: uints(dir.join(format!("{id}.selected.u32"))),
                    counts: uints(dir.join(format!("{id}.counts.u32"))),
                }
            })
            .collect();
        let before = d.allocated_bytes();
        let mut s = qsa::State::new(&d, &w, 128, 5, 2061).unwrap();
        assert_eq!(
            d.allocated_bytes() - before,
            qsa::State::bytes(128, 5, 2061).unwrap() as u64
        );
        let page_stride = 2061usize.div_ceil(16);
        let block_stride = page_stride * 4;
        let mut prefix_worst = 0f32;
        for first in (0..2040).step_by(128) {
            prefix_worst = prefix_worst.max(run(
                &d,
                &mut s,
                &w,
                &streams,
                &(first..(first + 128).min(2040))
                    .map(|pos| (0, pos))
                    .collect::<Vec<_>>(),
            ));
        }
        s.copy_slot(&d, 0, 4).unwrap();
        for chunk in [1, 4, 9, 16, 31, 128] {
            for slot in 0..4 {
                s.reset(&d, slot).unwrap();
            }
            s.copy_slot(&d, 4, 0).unwrap();
            let mut pos = [2040, 0, 0, 0];
            let mut round = 0;
            let mut worst = prefix_worst;
            while pos.iter().zip(&streams).any(|(&p, t)| p < t.length) {
                let mut rows = vec![];
                for slot in [3, 1, 0, 2] {
                    let count = (streams[slot].length - pos[slot]).min(if slot == round % 4 {
                        1
                    } else {
                        chunk
                    });
                    rows.extend((pos[slot]..pos[slot] + count).map(|p| (slot, p)));
                    pos[slot] += count;
                }
                worst = worst.max(run(&d, &mut s, &w, &streams, &rows));
                round += 1;
            }
            for (slot, t) in streams.iter().enumerate() {
                close(
                    &unsafe {
                        s.cache
                            .pooled
                            .read_f32(slot * block_stride * 128, t.pooled.len())
                    },
                    &t.pooled,
                    "cached pooled keys",
                );
            }
            let keys = unsafe { s.cache.keys.read_u32(s.cache.keys.len() / 4) };
            let values = unsafe { s.cache.values.read_u32(s.cache.values.len() / 4) };
            let pooled = unsafe { s.cache.pooled.read_u32(s.cache.pooled.len() / 4) };
            let ring = unsafe { s.cache.ring.read_u32(s.cache.ring.len() / 4) };
            let lengths = s.lengths.clone();
            let x = upload(&d, &streams[0].input[..WIDTH]);
            let y = upload(&d, &vec![-9876.5; WIDTH]);
            assert!(s.run(&d, &w, &[(0, 2061)], &x, &y).is_err());
            assert!(s.run(&d, &w, &[(1, 37), (2, usize::MAX)], &x, &y).is_err());
            assert!(s.reset(&d, 5).is_err());
            assert!(s.copy_slot(&d, 0, 5).is_err());
            assert!(s.copy_slot(&d, 0, 0).is_err());
            assert_eq!(s.lengths, lengths);
            assert_eq!(
                unsafe { s.cache.keys.read_u32(s.cache.keys.len() / 4) },
                keys
            );
            assert_eq!(
                unsafe { s.cache.values.read_u32(s.cache.values.len() / 4) },
                values
            );
            assert_eq!(
                unsafe { s.cache.pooled.read_u32(s.cache.pooled.len() / 4) },
                pooled
            );
            assert_eq!(
                unsafe { s.cache.ring.read_u32(s.cache.ring.len() / 4) },
                ring
            );
            assert_eq!(unsafe { y.read_f32(0, WIDTH) }, vec![-9876.5; WIDTH]);
            s.reset(&d, 1).unwrap();
            // Complete bytes of all other slots remain unchanged, including
            // the immutable saved prefix. Page permutation is reversed.
            let now = unsafe { s.cache.keys.read_u32(s.cache.keys.len() / 4) };
            for slot in [0, 2, 3, 4] {
                let start = (4 - slot) * page_stride * 16 * 512 / 2;
                let len = page_stride * 16 * 512 / 2;
                assert_eq!(&now[start..start + len], &keys[start..start + len]);
            }
            // Restore a prefix ending inside a block. Continuing the fork
            // needs the raw ring, not just normalized completed blocks/KV.
            s.reset(&d, 0).unwrap();
            run(&d, &mut s, &w, &streams, &[(0, 0), (0, 1), (0, 2)]);
            s.copy_slot(&d, 0, 1).unwrap();
            let t = &streams[0];
            let x = upload(&d, &t.input[3 * WIDTH..7 * WIDTH]);
            let y = d.alloc(4 * WIDTH * 4).unwrap();
            s.run(&d, &w, &(3..7).map(|p| (1, p)).collect::<Vec<_>>(), &x, &y)
                .unwrap();
            let probe = t.probes.iter().position(|&p| p == 3).unwrap();
            close(
                &unsafe { y.read_f32(0, WIDTH) },
                &t.output[probe * WIDTH..(probe + 1) * WIDTH],
                "incomplete-block fork",
            );
            eprintln!(
                "QSA layer={layer} chunk={chunk}: {round} mixed batches; exact selections; max output error={worst:e}; cache/reset/fork pass"
            );
        }
    }
    assert_eq!(d.allocated_bytes(), 0);
}
