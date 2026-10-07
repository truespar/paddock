//! W16 decode-class gates for nemotron: rows that stand in for decode steps
//! (a one-row tick, a decode tick's rows, a spec verify round's) run one
//! batch-invariant class, so a row's bits never depend on what shares its
//! launch. Each kernel is bit-compared per row, R rows at once against the
//! same row alone, on real checkpoint planes where the class reads them:
//!
//!   1. nvf4_moe_{up_relu2,down_part}_w16 - the W4A16 expert pair
//!   2. dense_w16 on the FP8 mamba projections (e4m3 widened to f16)
//!   3. dense_w16 on the bf16 attention planes, fused q|k|v segments included
//!   4. attn_rows_partial_fixed - the fixed key splits, a verify group's row
//!      against the one-row group a decode tick attends through
//!   5. moe_route_w16 - the routing front in one launch against the router,
//!      top-k, bf16 cast and align launches it replaces
//!
//! The row counts straddle every launch-shape seam the kernels have: the
//! in-kernel activation cast (<= 8 rows) against the pre-cast plane, the
//! 32-row blocks, a ragged tail, and the class's 64-row ceiling.
//! CUDA + pack gated; 1-3 need the Nemotron NVFP4 checkpoint.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_engine::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor};
use paddock_models::ggml_type::GgmlType;
use paddock_models::modelopt::{fp8_view, nvfp4_view};
use paddock_models::safetensors::ShardedSafetensors;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";

fn checkpoint() -> Option<ShardedSafetensors> {
    let dir = common::model_dir(CKPT_ENV, &[CKPT_DIR])?;
    ShardedSafetensors::open_dir(&dir).ok()
}

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn bf16_up(exec: &GpuExecutor, x: &[f32]) -> CudaSlice<half::bf16> {
    let d = exec.to_device(x).unwrap();
    let mut o = exec.stream_alloc_bf16(x.len()).unwrap();
    exec.convert_f32_bf16(&d, &mut o, x.len()).unwrap();
    o
}

/// Eight real experts of layer 1 (tiled), one doubling as the shared expert
/// on its own planes; k = 3 picks a row with expert 0 on every row, so a
/// 32-row block fills at 32 rows and spills past it.
#[test]
fn w16_experts_rows_match_one_row_bitexact() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe_w16() {
        common::missing("pack has no W16 expert pair (cc != 12?)");
        return;
    }
    let n_e = 8usize;
    let (mut up_p, mut up_s, mut up_s2) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dn_p, mut dn_s, mut dn_s2) = (Vec::new(), Vec::new(), Vec::new());
    let mut views = Vec::new();
    for e in 0..n_e {
        let u = nvfp4_view(&st, &format!("backbone.layers.1.mixer.experts.{e}.up_proj"))
            .expect("up view");
        let d = nvfp4_view(
            &st,
            &format!("backbone.layers.1.mixer.experts.{e}.down_proj"),
        )
        .expect("down view");
        up_p.extend_from_slice(u.packed);
        up_s.extend_from_slice(u.scales);
        up_s2.push(u.scale2);
        dn_p.extend_from_slice(d.packed);
        dn_s.extend_from_slice(d.scales);
        dn_s2.push(d.scale2);
        views.push((u, d));
    }
    let (ff, in_dim, embd) = (views[0].0.n, views[0].0.k, views[0].1.n);
    let sh = &views[5];
    let k = 3usize;
    let up = exec
        .nvf4_moe_upload_tiled(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up");
    let dn = exec
        .nvf4_moe_upload_tiled(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn");
    let shu = exec
        .nvf4_moe_upload_tiled(sh.0.packed, sh.0.scales, &[sh.0.scale2], 1, ff, in_dim)
        .expect("shu");
    let shd = exec
        .nvf4_moe_upload_tiled(sh.1.packed, sh.1.scales, &[sh.1.scale2], 1, embd, ff)
        .expect("shd");
    let aw = k * ff + ff;
    let nb = 64usize;
    // one launch pair over `rows` rows; returns (activations, partials)
    let run = |idx: &[u32], w: &[f32], x: &[f32], rows: usize| -> (Vec<u16>, Vec<f32>) {
        let d_idx = exec.to_device_u32(idx).unwrap();
        let d_w = exec.to_device(w).unwrap();
        let x16 = bf16_up(&exec, x);
        let mut srow = exec.alloc_u32(nb * 32).unwrap();
        let mut sslot = exec.alloc_u32(nb * 32).unwrap();
        let mut bexp = exec.alloc_u32(nb).unwrap();
        exec.moe_align_bm(
            &d_idx, &mut srow, &mut sslot, &mut bexp, rows, k, n_e, 32, nb,
        )
        .expect("align");
        let mut act = exec.stream_alloc_bf16(rows * aw).unwrap();
        let mut part = exec.alloc(rows * (k + 1) * embd).unwrap();
        exec.nvf4_moe_up_relu2_w16(&up, &shu, &srow, &sslot, &bexp, &x16, &mut act, k, nb, rows)
            .expect("up w16");
        exec.nvf4_moe_down_part_w16(
            &dn, &shd, &srow, &sslot, &bexp, &d_w, &act, &mut part, k, nb, rows,
        )
        .expect("down w16");
        let act = exec.to_host_bf16(&act).unwrap();
        (
            act.iter().map(|v| v.to_bits()).collect(),
            exec.to_host(&part).unwrap(),
        )
    };
    for rows in [2usize, 5, 8, 13, 33, 64] {
        let idx: Vec<u32> = (0..rows)
            .flat_map(|r| [0u32, 1 + (r % 7) as u32, 1 + ((r + 3) % 7) as u32])
            .collect();
        let topk_w: Vec<f32> = det(rows * k, 7 + rows as u64)
            .iter()
            .map(|v| 0.2 + 0.1 * v)
            .collect();
        let x = det(rows * in_dim, 91 + rows as u64);
        let (act, part) = run(&idx, &topk_w, &x, rows);
        assert!(
            part.iter().all(|v| v.is_finite()),
            "{rows} rows: non-finite partials"
        );
        let pw = (k + 1) * embd;
        for r in 0..rows {
            let (a1, p1) = run(
                &idx[r * k..(r + 1) * k],
                &topk_w[r * k..(r + 1) * k],
                &x[r * in_dim..(r + 1) * in_dim],
                1,
            );
            assert_eq!(
                &act[r * aw..(r + 1) * aw],
                &a1[..],
                "{rows} rows: row {r}'s activations depend on the rows beside it"
            );
            assert!(
                bits_eq(&part[r * pw..(r + 1) * pw], &p1),
                "{rows} rows: row {r}'s partials depend on the rows beside it"
            );
        }
        println!("[w16] experts: {rows} rows bit-exact with each row alone");
    }
}

/// The FP8 mamba projections (in_proj [10304, 2688], out_proj [2688, 4096])
/// across the in-kernel cast (<= 8 rows) and the pre-cast plane past it.
#[test]
fn w16_dense_e4m3_rows_match_one_row_bitexact() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_mamba2() || !exec.has_dense_w16() {
        common::missing("pack has no dense_w16");
        return;
    }
    for name in [
        "backbone.layers.0.mixer.in_proj",
        "backbone.layers.0.mixer.out_proj",
    ] {
        let v = fp8_view(&st, name).expect("fp8 view");
        let plane = exec
            .fp8_ckpt_to_f8row(v.weight, v.weight_scale, v.k, v.n)
            .expect("upload");
        let run = |x: &[f32], rows: usize| -> Vec<f32> {
            let d_x = exec.to_device(x).unwrap();
            let mut x16 = exec.alloc_f16(rows * v.k).unwrap();
            let mut d_y = exec.alloc(rows * v.n).unwrap();
            exec.dense_w16_e4m3(&plane, v.k, v.n, &d_x, &mut x16, &mut d_y, v.n, rows)
                .expect("dense w16");
            exec.to_host(&d_y).unwrap()
        };
        for rows in [2usize, 7, 8, 9, 32, 41, 64] {
            let x = det(rows * v.k, 30 + rows as u64);
            let y = run(&x, rows);
            assert!(
                y.iter().all(|v| v.is_finite()),
                "{name}: non-finite at {rows} rows"
            );
            for r in 0..rows {
                let y1 = run(&x[r * v.k..(r + 1) * v.k], 1);
                assert!(
                    bits_eq(&y[r * v.n..(r + 1) * v.n], &y1),
                    "{name}: {rows} rows, row {r} depends on the rows beside it"
                );
            }
        }
        println!("[w16] {name} [{}, {}]: bit-exact at 2..64 rows", v.n, v.k);
    }
}

/// The bf16 attention planes of layer 5: the fused q|k|v plane (one convert,
/// three segment calls - the verify walk's shape) and o_proj. A segment call
/// must be the fused plane's columns bit for bit, and every row must be the
/// row alone.
#[test]
fn w16_dense_bf16_rows_match_one_row_bitexact() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_dense_w16() || !exec.has_bf16_dense() {
        common::missing("pack has no dense_w16");
        return;
    }
    let raw = |n: &str| st.bytes(n).expect("tensor").1.to_vec();
    let (q, kk, vv) = (
        raw("backbone.layers.5.mixer.q_proj.weight"),
        raw("backbone.layers.5.mixer.k_proj.weight"),
        raw("backbone.layers.5.mixer.v_proj.weight"),
    );
    let o = raw("backbone.layers.5.mixer.o_proj.weight");
    let embd = 2688usize;
    let (q_dim, kv_dim) = (q.len() / 2 / embd, kk.len() / 2 / embd);
    let fused: Vec<u8> = [q, kk, vv].concat();
    let wqkv = QuantTensor {
        bytes: exec.to_device_u8(&fused).unwrap(),
        ty: GgmlType::Bf16,
        dims: vec![embd, q_dim + 2 * kv_dim],
    };
    let wo = QuantTensor {
        bytes: exec.to_device_u8(&o).unwrap(),
        ty: GgmlType::Bf16,
        dims: vec![q_dim, embd],
    };
    for (name, w, in_dim, out_dim) in [
        ("wqkv", &wqkv, embd, q_dim + 2 * kv_dim),
        ("wo", &wo, q_dim, embd),
    ] {
        let run = |x: &[f32], rows: usize| -> Vec<f32> {
            let d_x = exec.to_device(x).unwrap();
            let mut x16 = exec.stream_alloc_bf16(rows * in_dim).unwrap();
            let mut d_y = exec.alloc(rows * out_dim).unwrap();
            exec.dense_w16_bf16(w, 0, out_dim, &d_x, &mut x16, &mut d_y, out_dim, rows)
                .expect("dense w16");
            exec.to_host(&d_y).unwrap()
        };
        for rows in [2usize, 7, 8, 9, 32, 41, 64] {
            let x = det(rows * in_dim, 50 + rows as u64);
            let y = run(&x, rows);
            assert!(
                y.iter().all(|v| v.is_finite()),
                "{name}: non-finite at {rows} rows"
            );
            for r in 0..rows {
                let y1 = run(&x[r * in_dim..(r + 1) * in_dim], 1);
                assert!(
                    bits_eq(&y[r * out_dim..(r + 1) * out_dim], &y1),
                    "{name}: {rows} rows, row {r} depends on the rows beside it"
                );
            }
            if name == "wqkv" {
                // the verify walk's shape: one cast, then q, k, v segments
                let x16 = bf16_up(&exec, &x);
                for (lo, n) in [(0, q_dim), (q_dim, kv_dim), (q_dim + kv_dim, kv_dim)] {
                    let mut d_s = exec.alloc(rows * n).unwrap();
                    exec.dense_w16_bf16_pre(w, lo, n, &x16, &mut d_s, n, rows)
                        .expect("segment");
                    let s = exec.to_host(&d_s).unwrap();
                    for r in 0..rows {
                        assert!(
                            bits_eq(
                                &s[r * n..(r + 1) * n],
                                &y[r * out_dim + lo..r * out_dim + lo + n]
                            ),
                            "wqkv: {rows} rows, segment at {lo} parts from the fused call at row {r}"
                        );
                    }
                }
                if exec.has_dense_w16_seg() {
                    // the serving shape: the fused plane in one launch into
                    // three outputs
                    let d_x = exec.to_device(&x).unwrap();
                    let mut x16s = exec.stream_alloc_bf16(rows * in_dim).unwrap();
                    let mut d_q = exec.alloc(rows * q_dim).unwrap();
                    let mut d_k = exec.alloc(rows * kv_dim).unwrap();
                    let mut d_v = exec.alloc(rows * kv_dim).unwrap();
                    exec.dense_w16_bf16_seg(
                        w, q_dim, kv_dim, &d_x, &mut x16s, &mut d_q, &mut d_k, &mut d_v, rows,
                    )
                    .expect("segmented launch");
                    let outs = [
                        (0, q_dim, exec.to_host(&d_q).unwrap()),
                        (q_dim, kv_dim, exec.to_host(&d_k).unwrap()),
                        (q_dim + kv_dim, kv_dim, exec.to_host(&d_v).unwrap()),
                    ];
                    for (lo, n, s) in &outs {
                        for r in 0..rows {
                            assert!(
                                bits_eq(
                                    &s[r * n..(r + 1) * n],
                                    &y[r * out_dim + lo..r * out_dim + lo + n]
                                ),
                                "wqkv: {rows} rows, the segmented launch's output at {lo} parts \
                                 from the plane's columns at row {r}"
                            );
                        }
                    }
                }
            }
        }
        let seg = if name == "wqkv" {
            ", q/k/v segments and the segmented launch too"
        } else {
            ""
        };
        println!("[w16] {name} [{out_dim}, {in_dim}]: bit-exact at 2..64 rows{seg}");
    }
}

/// Fixed key splits: a verify group's row and the same row as its own group
/// (the decode tick's one-row groups) must combine to the same bits, in both
/// cache formats, at nemotron's geometry. The shared-law kernel is the
/// accuracy yardstick - same math, different split law, so they agree to the
/// summation-order class and no closer.
#[test]
fn w16_attn_fixed_splits_rows_invariant() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_attn_rows_partial_fixed() || !exec.has_attn_rows_partial() {
        common::missing("pack has no fixed-split rows partial");
        return;
    }
    let (n_heads, n_kv_heads, head_dim) = (32usize, 2usize, 128usize);
    let kv_dim = n_kv_heads * head_dim;
    let qdim = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_ctx = 2048usize;
    let bps = max_ctx / 16;
    let n_slots = 2usize;
    // identity block table: pool index == slot * max_ctx + pos
    let bt_host: Vec<u32> = (0..(n_slots * bps) as u32).collect();
    let d_bt = exec.stream.clone_htod(&bt_host).unwrap();
    // nemotron has no sinks - the plane serving allocates
    let d_s = exec.alloc_no_sinks(n_heads).unwrap();
    let kv_host = |seed: u64, dt: KvDtype| -> CudaSlice<u8> {
        let v = det(n_slots * max_ctx * kv_dim, seed);
        let bytes: Vec<u8> = match dt {
            KvDtype::Fp16 => v
                .iter()
                .flat_map(|&x| f16::from_f32(x).to_le_bytes())
                .collect(),
            // e4m3 codes with |v| < 2 and never the NaN pattern (s.1111.111)
            KvDtype::Fp8E4m3 => v
                .iter()
                .map(|&x| ((x.to_bits() >> 24) as u8 & 0x80) | ((x.to_bits() >> 8) as u8 % 0x40))
                .collect(),
        };
        exec.stream.clone_htod(&bytes).unwrap()
    };
    // (rows as (slot, position), verify groups as (first row, rows))
    let verify: Vec<(u32, u32)> = (0..8).map(|i| (1, 1500 + i)).collect();
    let two: Vec<(u32, u32)> = (0..4)
        .map(|i| (1, 300 + i))
        .chain((0..5).map(|i| (0, 1000 + i)))
        .collect();
    let short: Vec<(u32, u32)> = (0..3).map(|i| (0, 5 + i)).collect();
    let cases: [(&str, &[(u32, u32)], Vec<u32>); 3] = [
        ("verify chunk", &verify, vec![0, 8]),
        ("two slots", &two, vec![0, 4, 4, 5]),
        ("inside one split", &short, vec![0, 3]),
    ];
    for dt in [KvDtype::Fp16, KvDtype::Fp8E4m3] {
        let d_k = kv_host(11, dt);
        let d_v = kv_host(12, dt);
        for (name, rows, groups) in &cases {
            let n_rows = rows.len();
            let q = det(n_rows * qdim, 21);
            let d_q = exec.to_device(&q).unwrap();
            let d_pos = exec
                .stream
                .clone_htod(&rows.iter().map(|r| r.1).collect::<Vec<u32>>())
                .unwrap();
            let d_slots = exec
                .stream
                .clone_htod(&rows.iter().map(|r| r.0).collect::<Vec<u32>>())
                .unwrap();
            let solo: Vec<u32> = (0..n_rows as u32).flat_map(|i| [i, 1]).collect();
            // the serving split law, and a coarser one
            for split in [256usize, 512] {
                let ns = max_ctx.div_ceil(split);
                let run = |groups: &[u32], fixed: bool| -> Vec<f32> {
                    let n_splits = if fixed { ns } else { 7 };
                    let d_groups = exec.stream.clone_htod(groups).unwrap();
                    let mut d_o = exec.alloc(n_heads * n_rows * n_splits * head_dim).unwrap();
                    let mut d_ml = exec.alloc(n_heads * n_rows * n_splits * 2).unwrap();
                    let mut d_out = exec.alloc(n_rows * qdim).unwrap();
                    let args = (&d_q, &d_k, &d_v);
                    if fixed {
                        exec.attn_rows_partial_fixed(
                            args.0,
                            args.1,
                            args.2,
                            &mut d_o,
                            &mut d_ml,
                            &d_pos,
                            &d_slots,
                            &d_groups,
                            groups.len() / 2,
                            Some((&d_bt, bps)),
                            max_ctx,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            kv_dim,
                            n_rows,
                            n_splits,
                            0,
                            scale,
                            dt,
                            split,
                            None,
                        )
                        .expect("fixed partial");
                    } else {
                        exec.attn_rows_partial(
                            args.0,
                            args.1,
                            args.2,
                            &mut d_o,
                            &mut d_ml,
                            &d_pos,
                            &d_slots,
                            &d_groups,
                            groups.len() / 2,
                            Some((&d_bt, bps)),
                            max_ctx,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            kv_dim,
                            n_rows,
                            n_splits,
                            0,
                            scale,
                            dt,
                        )
                        .expect("shared partial");
                    }
                    exec.attn_combine_batch(
                        &d_o, &d_ml, &d_s, &mut d_out, n_heads, head_dim, n_splits, n_rows,
                    )
                    .expect("combine");
                    exec.to_host(&d_out).unwrap()
                };
                let grouped = run(groups, true);
                let alone = run(&solo, true);
                assert!(
                    bits_eq(&grouped, &alone),
                    "{dt:?} {name} split {split}: a row's bits depend on its group"
                );
                let shared = run(groups, false);
                let peak = shared.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let maxd = grouped
                    .iter()
                    .zip(&shared)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    grouped.iter().all(|v| v.is_finite()) && maxd <= 1e-4 * peak.max(1.0),
                    "{dt:?} {name} split {split}: {maxd:.3e} off the shared law (peak {peak:.3})"
                );
                println!(
                    "[w16] attn {dt:?} {name} split {split}: rows invariant, {maxd:.1e} off the shared law"
                );
            }
        }
    }
}

/// Slot 745 (two warps a row, f32 fold of the halves): under both split
/// laws a row's bits must not depend on its group - a verify group's row and
/// the same row alone (a decode tick) combine to the same bits - and the
/// fold must agree with the one-warp kernel on the same law to the
/// summation-order class. e4m3 paged only (745 refuses anything else),
/// nemotron's geometry, groups of up to six rows.
#[test]
fn w16_attn_kh_rows_invariant() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_attn_rows_partial_kh() || !exec.has_attn_rows_partial_pow2() {
        common::missing("pack has no two-warps-a-row rows partial");
        return;
    }
    let (n_heads, n_kv_heads, head_dim) = (32usize, 2usize, 128usize);
    let kv_dim = n_kv_heads * head_dim;
    let qdim = n_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_ctx = 8192usize;
    let bps = max_ctx / 16;
    let n_slots = 2usize;
    // a scattered block table (the serve pool hands out freed pages)
    let nb = (n_slots * bps) as u32;
    let bt_host: Vec<u32> = (0..nb).map(|j| (j * 7919) % nb).collect();
    let d_bt = exec.stream.clone_htod(&bt_host).unwrap();
    let d_s = exec.alloc_no_sinks(n_heads).unwrap();
    let kv = |seed: u64| -> CudaSlice<u8> {
        let v = det(n_slots * max_ctx * kv_dim, seed);
        let bytes: Vec<u8> = v
            .iter()
            .map(|&x| ((x.to_bits() >> 24) as u8 & 0x80) | ((x.to_bits() >> 8) as u8 % 0x40))
            .collect();
        exec.stream.clone_htod(&bytes).unwrap()
    };
    let (d_k, d_v) = (kv(31), kv(32));
    // (rows as (slot, position), groups as (first row, rows <= 6))
    let verify: Vec<(u32, u32)> = (0..8).map(|i| (1, 7000 + i)).collect();
    let dspark: Vec<(u32, u32)> = (0..6).map(|i| (0, 3001 + i)).collect();
    let two: Vec<(u32, u32)> = (0..4)
        .map(|i| (1, 300 + i))
        .chain((0..5).map(|i| (0, 1000 + i)))
        .collect();
    let short: Vec<(u32, u32)> = (0..3).map(|i| (0, 5 + i)).collect();
    let cases: [(&str, &[(u32, u32)], Vec<u32>); 4] = [
        ("8-row verify as 6 + 2", &verify, vec![0, 6, 6, 2]),
        ("DSpark round", &dspark, vec![0, 6]),
        ("two slots", &two, vec![0, 4, 4, 5]),
        ("inside one split", &short, vec![0, 3]),
    ];
    for (name, rows, groups) in &cases {
        let n_rows = rows.len();
        let q = det(n_rows * qdim, 41);
        let d_q = exec.to_device(&q).unwrap();
        let d_pos = exec
            .stream
            .clone_htod(&rows.iter().map(|r| r.1).collect::<Vec<u32>>())
            .unwrap();
        let d_slots = exec
            .stream
            .clone_htod(&rows.iter().map(|r| r.0).collect::<Vec<u32>>())
            .unwrap();
        let solo: Vec<u32> = (0..n_rows as u32).flat_map(|i| [i, 1]).collect();
        let gmax = groups.chunks(2).map(|g| g[1] as usize).max().unwrap();
        // 683's fixed law at the serving split and a coarser one; 744's pow2
        // law (split_keys 0) at its 128-split budget; 745's TILE law at a
        // 120-split budget (the one-warp reference runs 744 there - the
        // same 256-key splits at this depth)
        for (split, n_splits) in [
            (512usize, max_ctx / 512),
            (1024, max_ctx / 1024),
            (0, 128),
            (0x4000_0000 | 120, 128),
        ] {
            let run = |groups: &[u32], kh: Option<usize>, law: usize| -> Vec<f32> {
                let d_groups = exec.stream.clone_htod(groups).unwrap();
                let mut d_o = exec.alloc(n_heads * n_rows * n_splits * head_dim).unwrap();
                let mut d_ml = exec.alloc(n_heads * n_rows * n_splits * 2).unwrap();
                let mut d_out = exec.alloc(n_rows * qdim).unwrap();
                exec.attn_rows_partial_fixed(
                    &d_q,
                    &d_k,
                    &d_v,
                    &mut d_o,
                    &mut d_ml,
                    &d_pos,
                    &d_slots,
                    &d_groups,
                    groups.len() / 2,
                    Some((&d_bt, bps)),
                    max_ctx,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    n_rows,
                    n_splits,
                    0,
                    scale,
                    KvDtype::Fp8E4m3,
                    law,
                    kh,
                )
                .expect("rows partial");
                exec.attn_combine_batch(
                    &d_o, &d_ml, &d_s, &mut d_out, n_heads, head_dim, n_splits, n_rows,
                )
                .expect("combine");
                exec.to_host(&d_out).unwrap()
            };
            let grouped = run(groups, Some(gmax), split);
            let alone = run(&solo, Some(1), split);
            assert!(
                bits_eq(&grouped, &alone),
                "{name} split {split}: a row's bits depend on its group under 745"
            );
            // the one-warp kernel has no TILE law: below 30K keys its splits
            // are 744's 256-key ones
            let ref_law = if split & 0x4000_0000 != 0 { 0 } else { split };
            let one_warp = run(&solo, None, ref_law);
            let peak = one_warp.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let maxd = grouped
                .iter()
                .zip(&one_warp)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                grouped.iter().all(|v| v.is_finite()) && maxd <= 1e-4 * peak.max(1.0),
                "{name} split {split}: {maxd:.3e} off the one-warp kernel (peak {peak:.3})"
            );
            println!(
                "[w16] attn kh {name} split {split}: rows invariant, {maxd:.1e} off the one-warp kernel"
            );
        }
    }
}

/// The routing front: one launch against the four it replaces. Its router
/// runs ONE summation order at every row count - the unfused launcher's
/// decode-width body, whereas that launcher moves to a lane-strided tile
/// kernel (another order) from 16 rows - so each row's logits must be the
/// same bits alone and among R rows, and the unfused router's below 16.
/// Top-k picks and weights, the bf16 cast and the 32-row sorting must be
/// what the unfused kernels make of those logits (the sorting compared per
/// expert - which of an expert's blocks a row lands in is the align's
/// atomic scatter order). Two
/// calls back to back at every row count prove the tickets come back zero.
/// Synthetic router at nemotron's shape (128 experts over 2688, top-6, a
/// selection bias), rows across both token tilings.
#[test]
fn w16_route_matches_the_unfused_router_bitexact() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_moe_route_w16() {
        common::missing("pack has no W16 routing front");
        return;
    }
    let (in_dim, n_e, k) = (2688usize, 128usize, 6usize);
    let router = DeviceTensor {
        buf: exec
            .to_device(
                &det(in_dim * n_e, 7)
                    .iter()
                    .map(|v| v * 0.05)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        dims: vec![in_dim, n_e],
    };
    let bias = exec
        .to_device(&det(n_e, 8).iter().map(|v| v * 0.01).collect::<Vec<_>>())
        .unwrap();
    let scale = 2.5f32;
    let mut tickets = exec
        .alloc_u32(GpuExecutor::moe_route_w16_tickets(64))
        .unwrap();
    struct Routed {
        logits: Vec<f32>,
        idx: Vec<u32>,
        w: Vec<f32>,
        x16: Vec<u16>,
        blocks: Vec<(u32, usize, Vec<(u32, u32)>)>,
    }
    // per expert: its block count and the (row, pick) pairs across them -
    // which of an expert's blocks a pair lands in is the scatter's atomic
    // order (an expert past 32 picks spills), so the grouping is compared
    let blocks =
        |nb: usize, sr: &[u32], ss: &[u32], be: &[u32]| -> Vec<(u32, usize, Vec<(u32, u32)>)> {
            let mut per: std::collections::BTreeMap<u32, (usize, Vec<(u32, u32)>)> =
                std::collections::BTreeMap::new();
            for b in 0..nb {
                if be[b] == u32::MAX {
                    continue;
                }
                let e = per.entry(be[b]).or_default();
                e.0 += 1;
                e.1.extend(
                    (0..32)
                        .map(|j| (sr[b * 32 + j], ss[b * 32 + j]))
                        .filter(|p| p.0 != u32::MAX),
                );
            }
            per.into_iter()
                .map(|(e, (n, mut v))| {
                    v.sort_unstable();
                    (e, n, v)
                })
                .collect()
        };
    let nb_of = |rows: usize| n_e.min(rows * k) + rows * k / 32;
    let mut route = |x: &[f32], rows: usize| -> Routed {
        let nb = nb_of(rows);
        let d_x = exec.to_device(x).unwrap();
        let mut lg = exec.alloc(rows * n_e).unwrap();
        let mut x16 = exec.stream_alloc_bf16(rows * in_dim).unwrap();
        let mut idx = exec.alloc_u32(rows * k).unwrap();
        let mut w = exec.alloc(rows * k).unwrap();
        let (mut sr, mut ss, mut be) = (
            exec.alloc_u32(nb * 32).unwrap(),
            exec.alloc_u32(nb * 32).unwrap(),
            exec.alloc_u32(nb).unwrap(),
        );
        exec.moe_route_w16(
            &router,
            &d_x,
            &mut lg,
            &bias,
            scale,
            k,
            &mut x16,
            &mut idx,
            &mut w,
            &mut sr,
            &mut ss,
            &mut be,
            nb,
            rows,
            &mut tickets,
        )
        .expect("routing front");
        assert!(
            exec.to_host_u32(&tickets).unwrap().iter().all(|&t| t == 0),
            "{rows} rows: tickets left non-zero"
        );
        Routed {
            logits: exec.to_host(&lg).unwrap(),
            idx: exec.to_host_u32(&idx).unwrap(),
            w: exec.to_host(&w).unwrap(),
            x16: exec
                .to_host_bf16(&x16)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            blocks: blocks(
                nb,
                &exec.to_host_u32(&sr).unwrap(),
                &exec.to_host_u32(&ss).unwrap(),
                &exec.to_host_u32(&be).unwrap(),
            ),
        }
    };
    for rows in [1usize, 2, 5, 8, 9, 16, 33, 64] {
        let nb = nb_of(rows);
        let x = det(rows * in_dim, 90 + rows as u64);
        let d_x = exec.to_device(&x).unwrap();
        let got = route(&x, rows);
        let again = route(&x, rows);
        assert!(
            bits_eq(&got.logits, &again.logits) && got.idx == again.idx,
            "{rows} rows: a second launch parts from the first"
        );
        // the class property: a row's logits alone == among the rows
        for r in 0..rows {
            let one = route(&x[r * in_dim..(r + 1) * in_dim], 1);
            assert!(
                bits_eq(&got.logits[r * n_e..(r + 1) * n_e], &one.logits),
                "{rows} rows: row {r}'s router logits depend on the rows beside it"
            );
        }
        // the unfused router at decode widths (its batch-body arm)
        if rows < 16 {
            let mut lg0 = exec.alloc(rows * n_e).unwrap();
            exec.matvec_f32_batch(&router, &d_x, &mut lg0, rows)
                .unwrap();
            assert!(
                bits_eq(&got.logits, &exec.to_host(&lg0).unwrap()),
                "{rows} rows: logits part from the unfused router's"
            );
        }
        // top-k, cast and align: the unfused kernels on the same logits
        let d_lg = exec.to_device(&got.logits).unwrap();
        let mut idx0 = exec.alloc_u32(rows * k).unwrap();
        let mut w0 = exec.alloc(rows * k).unwrap();
        exec.moe_topk_sigmoid_batch(&d_lg, &bias, scale, n_e, k, &mut idx0, &mut w0, rows)
            .unwrap();
        assert_eq!(
            got.idx,
            exec.to_host_u32(&idx0).unwrap(),
            "{rows} rows: picks"
        );
        assert!(
            bits_eq(&got.w, &exec.to_host(&w0).unwrap()),
            "{rows} rows: weights"
        );
        let mut x16_0 = exec.stream_alloc_bf16(rows * in_dim).unwrap();
        exec.convert_f32_bf16(&d_x, &mut x16_0, rows * in_dim)
            .unwrap();
        let want16: Vec<u16> = exec
            .to_host_bf16(&x16_0)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got.x16, want16, "{rows} rows: bf16 cast");
        let (mut sr0, mut ss0, mut be0) = (
            exec.alloc_u32(nb * 32).unwrap(),
            exec.alloc_u32(nb * 32).unwrap(),
            exec.alloc_u32(nb).unwrap(),
        );
        exec.moe_align(&idx0, &mut sr0, &mut ss0, &mut be0, rows, k, n_e, nb)
            .unwrap();
        let want_b = blocks(
            nb,
            &exec.to_host_u32(&sr0).unwrap(),
            &exec.to_host_u32(&ss0).unwrap(),
            &exec.to_host_u32(&be0).unwrap(),
        );
        assert_eq!(got.blocks, want_b, "{rows} rows: sorted blocks");
    }
    println!(
        "[w16] routing front: logits row-invariant at 1..64 rows (== the unfused router below \
         16), picks/weights/cast/align == the unfused kernels, tickets reset"
    );
}
