//! K-quant (Q4_K / Q5_K / Q6_K / IQ4_XS) GPU-path parity on real UD-file
//! weights, against an in-test CPU reference dequant (independent port of the
//! GGUF format spec - same math the kernels implement). Bisects the whole
//! stage-1 ladder per format:
//!   1. upload() raw-layout dequant           -> f32 table
//!   2. repack_kquant + kquant_dequant_rp     -> must match (1)
//!   3. kquant_gather (embedding rows)        -> must match CPU rows
//!   4. kquant_gemv                           -> CPU dot, rel-err gated
//!   5. dequant_rp + gemm_f32 (prefill interim) -> CPU matvec, rel-err gated
//!
//! Gated on: CUDA device + built pack + the UD-Q4_K_XL download.

mod common;

use paddock_engine::gpu::GpuExecutor;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;

const IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

fn f16(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

fn scale_min_k4(s: &[u8], j: usize) -> (u32, u32) {
    if j < 4 {
        ((s[j] & 63) as u32, (s[j + 4] & 63) as u32)
    } else {
        (
            ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)) as u32,
            ((s[j + 4] >> 4) | ((s[j] >> 6) << 4)) as u32,
        )
    }
}

/// CPU reference dequant of one whole k-quant tensor (row-major, GGUF order).
fn cpu_dequant(bytes: &[u8], ty: GgmlType, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    let n_super = n / 256;
    match ty {
        GgmlType::Q4K | GgmlType::Q5K => {
            let (bs, q5) = if ty == GgmlType::Q5K {
                (176, true)
            } else {
                (144, false)
            };
            for b in 0..n_super {
                let s = &bytes[b * bs..];
                let (d, dmin) = (f16(&s[0..]), f16(&s[2..]));
                let scales = &s[4..16];
                // Q5_K source order: qh[32] PRECEDES qs[128] (ggml struct)
                let qs = if q5 { &s[48..176] } else { &s[16..144] };
                let qh = if q5 { &s[16..48] } else { &s[0..0] };
                for j in 0..8 {
                    let (sc, m) = scale_min_k4(scales, j);
                    let dj = d * sc as f32;
                    let mj = dmin * m as f32;
                    let qg = &qs[(j >> 1) * 32..];
                    for l in 0..32 {
                        let mut v = if j & 1 == 1 { qg[l] >> 4 } else { qg[l] & 0xF } as u32;
                        if q5 && (qh[l] >> j) & 1 == 1 {
                            v += 16;
                        }
                        y[b * 256 + j * 32 + l] = dj * v as f32 - mj;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            for b in 0..n_super {
                let s = &bytes[b * 210..];
                let (ql, qh) = (&s[0..128], &s[128..192]);
                let sc = &s[192..208];
                let d = f16(&s[208..]);
                for nh in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let (qlo, qlh) = (ql[nh * 64 + l], ql[nh * 64 + 32 + l]);
                        let h = qh[nh * 32 + l];
                        let q = [
                            ((qlo & 0xF) as i32 | ((h & 3) as i32) << 4) - 32,
                            ((qlh & 0xF) as i32 | (((h >> 2) & 3) as i32) << 4) - 32,
                            ((qlo >> 4) as i32 | (((h >> 4) & 3) as i32) << 4) - 32,
                            ((qlh >> 4) as i32 | (((h >> 6) & 3) as i32) << 4) - 32,
                        ];
                        for (r, qv) in q.iter().enumerate() {
                            y[b * 256 + nh * 128 + r * 32 + l] =
                                d * (sc[nh * 8 + r * 2 + is] as i8) as f32 * *qv as f32;
                        }
                    }
                }
            }
        }
        GgmlType::Iq4Xs => {
            for b in 0..n_super {
                let s = &bytes[b * 136..];
                let d = f16(&s[0..]);
                let sh = u16::from_le_bytes([s[2], s[3]]);
                let sl = &s[4..8];
                let qs = &s[8..136];
                for ib in 0..8 {
                    let ls = (((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) as i32
                        | (((sh >> (2 * ib)) & 3) as i32) << 4)
                        - 32;
                    let dl = d * ls as f32;
                    let q = &qs[ib * 16..];
                    for j in 0..16 {
                        y[b * 256 + ib * 32 + j] = dl * IQ4NL[(q[j] & 0xF) as usize] as f32;
                        y[b * 256 + ib * 32 + 16 + j] = dl * IQ4NL[(q[j] >> 4) as usize] as f32;
                    }
                }
            }
        }
        GgmlType::Q4_0 => {
            // 32-weight blocks: f16 d + 16 nibble bytes; byte k holds weight
            // k (low) and k+16 (high); value = d*(q-8)
            for b in 0..n / 32 {
                let blk = &bytes[b * 18..b * 18 + 18];
                let d = f16(&blk[0..]);
                for k in 0..16 {
                    y[b * 32 + k] = d * ((blk[2 + k] & 0xF) as i32 - 8) as f32;
                    y[b * 32 + 16 + k] = d * ((blk[2 + k] >> 4) as i32 - 8) as f32;
                }
            }
        }
        other => panic!("not a k-quant: {other:?}"),
    }
    y
}

fn deterministic_input(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

/// CPU mirror of pd_quantize_q8_mmq's per-32-block math: amax/127 scale,
/// round-nearest-even, clamp to +-127. Returns (int8 values, per-block scales).
fn cpu_quantize_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut scl = vec![0f32; nb];
    for b in 0..nb {
        let blk = &x[b * 32..b * 32 + 32];
        let a = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let s = a * (1.0 / 127.0);
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        scl[b] = s;
        for (l, v) in blk.iter().enumerate() {
            let qi = (v * inv).round_ties_even() as i32;
            q[b * 32 + l] = qi.clamp(-127, 127) as i8;
        }
    }
    (q, scl)
}

/// Decode one k-quant super-block into the W4A8 kernel's operands: 256
/// CENTERED int8 weights + per-32 (dj, mu) - or per-16 dj with mu = 0 for
/// Q6_K (its scales are per-16; the kernel splits the mma accordingly).
/// Mirrors the tile-staging math exactly (f32 products, same centering).
fn cpu_w4a8_super(s: &[u8], ty: GgmlType) -> ([i8; 256], [f32; 16], [f32; 8]) {
    let mut q = [0i8; 256];
    let mut dj16 = [0f32; 16]; // per-16 scale slots (per-32 formats duplicate)
    let mut mu = [0f32; 8];
    match ty {
        GgmlType::Q4K | GgmlType::Q5K => {
            let q5 = ty == GgmlType::Q5K;
            let (d, dmin) = (f16(&s[0..]), f16(&s[2..]));
            let scales = &s[4..16];
            let qs = if q5 { &s[48..176] } else { &s[16..144] };
            let qh = if q5 { &s[16..48] } else { &s[0..0] };
            let c = if q5 { 16i32 } else { 8i32 };
            for j in 0..8 {
                let (sc, m) = scale_min_k4(scales, j);
                let dj = d * sc as f32;
                dj16[2 * j] = dj;
                dj16[2 * j + 1] = dj;
                mu[j] = c as f32 * dj - dmin * m as f32;
                let qg = &qs[(j >> 1) * 32..];
                for l in 0..32 {
                    let mut v = if j & 1 == 1 { qg[l] >> 4 } else { qg[l] & 0xF } as i32;
                    if q5 && (qh[l] >> j) & 1 == 1 {
                        v += 16;
                    }
                    q[j * 32 + l] = (v - c) as i8;
                }
            }
        }
        GgmlType::Q6K => {
            let (ql, qh) = (&s[0..128], &s[128..192]);
            let sc = &s[192..208];
            let d = f16(&s[208..]);
            for i in 0..16 {
                dj16[i] = d * (sc[i] as i8) as f32;
            }
            for nh in 0..2 {
                for l in 0..32 {
                    let (qlo, qlh) = (ql[nh * 64 + l], ql[nh * 64 + 32 + l]);
                    let h = qh[nh * 32 + l];
                    let qv = [
                        ((qlo & 0xF) as i32 | ((h & 3) as i32) << 4) - 32,
                        ((qlh & 0xF) as i32 | (((h >> 2) & 3) as i32) << 4) - 32,
                        ((qlo >> 4) as i32 | (((h >> 4) & 3) as i32) << 4) - 32,
                        ((qlh >> 4) as i32 | (((h >> 6) & 3) as i32) << 4) - 32,
                    ];
                    for (r, v) in qv.iter().enumerate() {
                        q[nh * 128 + r * 32 + l] = *v as i8;
                    }
                }
            }
        }
        GgmlType::Iq4Xs => {
            let d = f16(&s[0..]);
            let sh = u16::from_le_bytes([s[2], s[3]]);
            let sl = &s[4..8];
            let qs = &s[8..136];
            for ib in 0..8 {
                let ls = (((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) as i32
                    | (((sh >> (2 * ib)) & 3) as i32) << 4)
                    - 32;
                let dl = d * ls as f32;
                dj16[2 * ib] = dl;
                dj16[2 * ib + 1] = dl;
                for j in 0..16 {
                    q[ib * 32 + j] = IQ4NL[(qs[ib * 16 + j] & 0xF) as usize];
                    q[ib * 32 + 16 + j] = IQ4NL[(qs[ib * 16 + j] >> 4) as usize];
                }
            }
        }
        GgmlType::Q4_0 => {
            // eight independent f16 block scales; centered q = nibble - 8,
            // dj = dsub[j]. The value is the centered form, so mu stays 0 -
            // a nonzero term would re-add the 8-center a second time.
            for j in 0..8 {
                let blk = &s[j * 18..j * 18 + 18];
                let dj = f16(&blk[0..]);
                dj16[2 * j] = dj;
                dj16[2 * j + 1] = dj;
                for k in 0..16 {
                    q[j * 32 + k] = ((blk[2 + k] & 0xF) as i32 - 8) as i8;
                    q[j * 32 + 16 + k] = ((blk[2 + k] >> 4) as i32 - 8) as i8;
                }
            }
        }
        other => panic!("not a k-quant: {other:?}"),
    }
    (q, dj16, mu)
}

/// Stage-2 W4A8 GEMM vs a CPU model of the same quantized-activation math
/// (exact int dots + f32 scale application, f64 accumulation), plus a class
/// check against the exact-f32 reference. Batch off the 128 boundary to
/// exercise the pad-column path.
#[test]
fn w4a8_matches_quantized_reference() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");

    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    let batch = 144usize; // 128 < batch < 256: pad columns live
    let check_rows = 384usize; // CPU-ref cost containment; GEMM writes all

    for (name, ty) in cases {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, ty, "{name}: file type changed?");
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let n_super = in_dim / 256;
        let src_b = bytes.len() / (n_super * out_dim);

        let x = deterministic_input(in_dim * batch, 7);
        let d_x = exec.to_device(&x).expect("x");
        let n_chunks = in_dim.div_ceil(128);
        let batch_pad = batch.div_ceil(128) * 128;
        let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
        exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
            .expect("quantize");
        let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
        exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch)
            .expect("sums");
        let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);
        let mut d_y = exec.alloc(out_dim * batch).expect("y");
        exec.kquant_gemm_w4a8(&kq, &d_yq, needs_sums.then_some(&d_sums), &mut d_y, batch)
            .expect("w4a8");
        let y_gpu = exec.to_host(&d_y).expect("y host");

        // CPU: quantize each activation row, then the W4A8 math in f64
        let rows = check_rows.min(out_dim);
        let mut xq = vec![0i8; in_dim * batch];
        let mut xs = vec![0f32; (in_dim / 32) * batch];
        let mut sums = vec![0f32; (in_dim / 32) * batch];
        for c in 0..batch {
            let (q, s) = cpu_quantize_q8(&x[c * in_dim..(c + 1) * in_dim]);
            for b in 0..in_dim / 32 {
                let ssum: i32 = q[b * 32..b * 32 + 32].iter().map(|&v| v as i32).sum();
                sums[c * (in_dim / 32) + b] = s[b] * ssum as f32;
            }
            xq[c * in_dim..(c + 1) * in_dim].copy_from_slice(&q);
            xs[c * (in_dim / 32)..(c + 1) * (in_dim / 32)].copy_from_slice(&s);
        }
        let mut y_ref = vec![0f32; rows * batch];
        let mut wq = vec![[0i8; 256]; n_super];
        let mut wdj = vec![[0f32; 16]; n_super];
        let mut wmu = vec![[0f32; 8]; n_super];
        for o in 0..rows {
            for sblk in 0..n_super {
                let s = &bytes[(o * n_super + sblk) * src_b..];
                let (q, dj, mu) = cpu_w4a8_super(s, ty);
                wq[sblk] = q;
                wdj[sblk] = dj;
                wmu[sblk] = mu;
            }
            for c in 0..batch {
                let xrow = &xq[c * in_dim..];
                let srow = &xs[c * (in_dim / 32)..];
                let sumrow = &sums[c * (in_dim / 32)..];
                let mut acc = 0f64;
                for sblk in 0..n_super {
                    for g16 in 0..16 {
                        let k0 = sblk * 256 + g16 * 16;
                        let dot: i32 = (0..16)
                            .map(|k| wq[sblk][g16 * 16 + k] as i32 * xrow[k0 + k] as i32)
                            .sum();
                        let db = srow[(sblk * 256 + g16 * 16) / 32];
                        acc += (wdj[sblk][g16] * db) as f64 * dot as f64;
                    }
                    if needs_sums {
                        for g in 0..8 {
                            acc += wmu[sblk][g] as f64 * sumrow[sblk * 8 + g] as f64;
                        }
                    }
                }
                y_ref[c * rows + o] = acc as f32;
            }
        }
        let gpu_sub: Vec<f32> = (0..batch)
            .flat_map(|c| y_gpu[c * out_dim..c * out_dim + rows].to_vec())
            .collect();
        let e = rel_err(&gpu_sub, &y_ref);
        eprintln!("{name} [{ty:?}] w4a8 vs quantized CPU ref rel_err {e:.2e}");
        assert!(
            e < 5e-4,
            "{name}: W4A8 decorrelated from its own math class ({e:.2e})"
        );

        // class check: quantized activations vs the exact-f32 product
        let w_f32 = cpu_dequant(bytes, ty, in_dim * out_dim);
        let mut y_exact = vec![0f32; rows * batch];
        for o in 0..rows {
            let wrow = &w_f32[o * in_dim..];
            for c in 0..batch {
                let xrow = &x[c * in_dim..];
                let mut acc = 0f64;
                for k in 0..in_dim {
                    acc += wrow[k] as f64 * xrow[k] as f64;
                }
                y_exact[c * rows + o] = acc as f32;
            }
        }
        let ec = rel_err(&gpu_sub, &y_exact);
        eprintln!("{name} [{ty:?}] w4a8 vs exact-f32 class gap {ec:.2e}");
        assert!(
            ec < 5e-2,
            "{name}: quantization-class error too large ({ec:.2e})"
        );
    }
}

/// `kquant_gemm_w4a8_pipe` (the cp.async-overlapped rung built to
/// close granite-30b's prefill gap on sm_120a - see
/// `packs/cuda/src/quant/kquant_w4a8.cuh`) against `kquant_gemm_w4a8` itself,
/// not a CPU reference: the pipe kernel keeps v1's exact tile_x layout and
/// MMA fold order, only moving the raw weight+scale byte source off a
/// synchronous global load onto a prefetched shared buffer, so a real bug
/// would show as a real divergence, not a shifted rel_err - the two should
/// agree near machine-epsilon-tight, not just within the quantization-class
/// tolerance the CPU-reference tests above use.
///
/// Real tensors off the granite-4.1-30b Q4_K_M file cover Q4_K and Q6_K, the
/// two types that file actually ships. Q5_K and
/// IQ4_XS have no elected model file to draw from (same gap
/// `w4a8_matches_quantized_reference` above already notes for IQ4_XS) -
/// their pipe-vs-v1 agreement is exercised by the shared per-DT code path
/// (same template, same branches) rather than a from-scratch synthetic
/// tensor; call this out rather than silently skip it.
#[test]
fn w4a8_pipe_matches_w4a8() {
    let Some(model) = common::model("GRANITE_30B_GGUF", common::GRANITE_30B_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemm_w4a8_pipe() {
        eprintln!("pack lacks kquant_gemm_w4a8_pipe - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    let cases = [
        ("blk.0.attn_output.weight", GgmlType::Q4K),
        ("blk.0.attn_v.weight", GgmlType::Q6K),
    ];
    // 65: just past the >64 rung gate. 128/144: exactly one tile / one padded
    // column. 1024: granite's real PF_ROWS_DEFAULT prefill chunk.
    let batches = [65usize, 128, 144, 1024];

    for (name, ty) in cases {
        let (info, _bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, ty, "{name}: file type changed?");
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);

        for &batch in &batches {
            let x = deterministic_input(in_dim * batch, 11);
            let d_x = exec.to_device(&x).expect("x");
            let n_chunks = in_dim.div_ceil(128);
            let batch_pad = batch.div_ceil(128) * 128;
            let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
                .expect("quantize");
            let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
            exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch)
                .expect("sums");

            let mut d_y_v1 = exec.alloc(out_dim * batch).expect("y v1");
            exec.kquant_gemm_w4a8(
                &kq,
                &d_yq,
                needs_sums.then_some(&d_sums),
                &mut d_y_v1,
                batch,
            )
            .expect("w4a8 v1");
            let y_v1 = exec.to_host(&d_y_v1).expect("y v1 host");

            let mut d_y_pipe = exec.alloc(out_dim * batch).expect("y pipe");
            exec.kquant_gemm_w4a8_pipe(
                &kq,
                &d_yq,
                needs_sums.then_some(&d_sums),
                &mut d_y_pipe,
                batch,
            )
            .expect("w4a8 pipe");
            let y_pipe = exec.to_host(&d_y_pipe).expect("y pipe host");

            let e = rel_err(&y_pipe, &y_v1);
            eprintln!("{name} [{ty:?}] b={batch} pipe vs v1 rel_err {e:.2e}");
            assert!(
                e < 1e-5,
                "{name} b={batch}: pipe kernel diverged from v1 ({e:.2e})"
            );
        }
    }
}

/// `kquant_gemm_w4a8_pipe2` (the genuinely-double-buffered rung:
/// a real 2-deep raw byte ring plus the half-width tile_x that affords it)
/// against `kquant_gemm_w4a8` (v1) directly, same discipline as
/// `w4a8_pipe_matches_w4a8` above: this restructures the unpack into two
/// half-superblock passes, so a real bug in the `ci`/`j` half-split math or
/// the ping-pong buffer indexing would show as a real divergence, not a
/// shifted rel_err.
#[test]
fn w4a8_pipe2_matches_w4a8() {
    let Some(model) = common::model("GRANITE_30B_GGUF", common::GRANITE_30B_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemm_w4a8_pipe2() {
        eprintln!("pack lacks kquant_gemm_w4a8_pipe2 - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    let cases = [
        ("blk.0.attn_output.weight", GgmlType::Q4K),
        ("blk.0.attn_v.weight", GgmlType::Q6K),
    ];
    let batches = [65usize, 128, 144, 1024];

    for (name, ty) in cases {
        let (info, _bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, ty, "{name}: file type changed?");
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);

        for &batch in &batches {
            let x = deterministic_input(in_dim * batch, 11);
            let d_x = exec.to_device(&x).expect("x");
            let n_chunks = in_dim.div_ceil(128);
            let batch_pad = batch.div_ceil(128) * 128;
            let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
                .expect("quantize");
            let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
            exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch)
                .expect("sums");

            let mut d_y_v1 = exec.alloc(out_dim * batch).expect("y v1");
            exec.kquant_gemm_w4a8(
                &kq,
                &d_yq,
                needs_sums.then_some(&d_sums),
                &mut d_y_v1,
                batch,
            )
            .expect("w4a8 v1");
            let y_v1 = exec.to_host(&d_y_v1).expect("y v1 host");

            let mut d_y_hi = exec.alloc(out_dim * batch).expect("y hi");
            exec.kquant_gemm_w4a8_pipe2(
                &kq,
                &d_yq,
                needs_sums.then_some(&d_sums),
                &mut d_y_hi,
                batch,
            )
            .expect("w4a8 pipe hi");
            let y_hi = exec.to_host(&d_y_hi).expect("y hi host");

            let e = rel_err(&y_hi, &y_v1);
            eprintln!("{name} [{ty:?}] b={batch} pipe2 vs v1 rel_err {e:.2e}");
            assert!(
                e < 1e-5,
                "{name} b={batch}: pipe2 kernel diverged from v1 ({e:.2e})"
            );
        }
    }
}

/// v1 / pipe / pipe2 throughput on the real granite-30b FFN shapes, which
/// dominate prefill time: `ffn_gate`/`ffn_up` (Q4K,
/// in=4096/out=32768 - the grid=2048-at-b=1024 case where the plain pipe
/// kernel leaves compute throughput on the table) and `ffn_down`
/// (Q6K, in=32768/out=4096, grid-starved the other way). Batches sweep
/// small (grid-starved) to the real PF_ROWS_DEFAULT (1024) prefill chunk.
/// `cargo test ... w4a8_pipe2_gemm_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn w4a8_pipe2_gemm_bench() {
    let Some(model) = common::model("GRANITE_30B_GGUF", common::GRANITE_30B_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open");
    let has_pipe = exec.has_kquant_gemm_w4a8_pipe();
    let has_pipe2 = exec.has_kquant_gemm_w4a8_pipe2();

    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
    ];
    for batch in [128usize, 256, 1024] {
        for (name, ty) in cases {
            let kq = exec.repack_kquant(&map, name).expect("repack");
            let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
            let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);
            let x = deterministic_input(in_dim * batch, 7);
            let d_x = exec.to_device(&x).expect("x");
            let n_chunks = in_dim.div_ceil(128);
            let batch_pad = batch.div_ceil(128) * 128;
            let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
            exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
                .unwrap();
            exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch).unwrap();

            let macs = in_dim as f64 * out_dim as f64 * batch as f64;
            let reps = 50;

            let mut d_y_v1 = exec.alloc(out_dim * batch).expect("y v1");
            exec.kquant_gemm_w4a8(
                &kq,
                &d_yq,
                needs_sums.then_some(&d_sums),
                &mut d_y_v1,
                batch,
            )
            .unwrap();
            exec.stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                exec.kquant_gemm_w4a8(
                    &kq,
                    &d_yq,
                    needs_sums.then_some(&d_sums),
                    &mut d_y_v1,
                    batch,
                )
                .unwrap();
            }
            exec.stream.synchronize().unwrap();
            let t_v1 = t0.elapsed().as_secs_f64() / reps as f64;

            let t_pipe = if has_pipe {
                let mut d_y_pipe = exec.alloc(out_dim * batch).expect("y pipe");
                exec.kquant_gemm_w4a8_pipe(
                    &kq,
                    &d_yq,
                    needs_sums.then_some(&d_sums),
                    &mut d_y_pipe,
                    batch,
                )
                .unwrap();
                exec.stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    exec.kquant_gemm_w4a8_pipe(
                        &kq,
                        &d_yq,
                        needs_sums.then_some(&d_sums),
                        &mut d_y_pipe,
                        batch,
                    )
                    .unwrap();
                }
                exec.stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() / reps as f64
            } else {
                f64::NAN
            };

            let t_pipe2 = if has_pipe2 {
                let mut d_y_hi = exec.alloc(out_dim * batch).expect("y hi");
                exec.kquant_gemm_w4a8_pipe2(
                    &kq,
                    &d_yq,
                    needs_sums.then_some(&d_sums),
                    &mut d_y_hi,
                    batch,
                )
                .unwrap();
                exec.stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    exec.kquant_gemm_w4a8_pipe2(
                        &kq,
                        &d_yq,
                        needs_sums.then_some(&d_sums),
                        &mut d_y_hi,
                        batch,
                    )
                    .unwrap();
                }
                exec.stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() / reps as f64
            } else {
                f64::NAN
            };

            eprintln!(
                "{:6} b={batch:4} {name:22} v1 {:7.1}us ({:5.1} TOPS)  pipe {:7.1}us ({:5.1} TOPS, {:+.1}%)  hi {:7.1}us ({:5.1} TOPS, {:+.1}%)",
                format!("{ty:?}"),
                t_v1 * 1e6,
                2.0 * macs / t_v1 / 1e12,
                t_pipe * 1e6,
                2.0 * macs / t_pipe / 1e12,
                (t_v1 / t_pipe - 1.0) * 100.0,
                t_pipe2 * 1e6,
                2.0 * macs / t_pipe2 / 1e12,
                (t_pipe / t_pipe2 - 1.0) * 100.0,
            );
        }
    }
}

/// The dp4a decode-batch GEMM vs the same quantized-math CPU model. Batches
/// straddle the b<=4/RT boundaries (5) and the RT=16 tile boundary (33).
#[test]
fn dp4a_matches_quantized_reference() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");

    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    let check_rows = 384usize;

    for batch in [5usize, 33] {
        for (name, ty) in cases {
            let (info, bytes) = map.tensor_bytes(name).expect("tensor");
            assert_eq!(info.ggml_type, ty, "{name}: file type changed?");
            let kq = exec.repack_kquant(&map, name).expect("repack");
            let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
            let n_super = in_dim / 256;
            let src_b = bytes.len() / (n_super * out_dim);

            let x = deterministic_input(in_dim * batch, 11);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_xq = exec.alloc_i8(in_dim * batch).expect("xq");
            let mut d_xs = exec.alloc(in_dim / 32 * batch).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim * batch)
                .expect("quantize");
            let mut d_sums = exec.alloc(in_dim / 16 * batch).expect("sums");
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
                .expect("sums");
            let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            exec.kquant_gemm_dp4a(
                &kq,
                &d_xq,
                &d_xs,
                needs_sums.then_some(&d_sums),
                &mut d_y,
                batch,
            )
            .expect("dp4a");
            let y_gpu = exec.to_host(&d_y).expect("y host");

            let rows = check_rows.min(out_dim);
            let mut xq = vec![0i8; in_dim * batch];
            let mut xs = vec![0f32; (in_dim / 32) * batch];
            let mut sums = vec![0f32; (in_dim / 32) * batch];
            for c in 0..batch {
                let (q, s) = cpu_quantize_q8(&x[c * in_dim..(c + 1) * in_dim]);
                for b in 0..in_dim / 32 {
                    let ssum: i32 = q[b * 32..b * 32 + 32].iter().map(|&v| v as i32).sum();
                    sums[c * (in_dim / 32) + b] = s[b] * ssum as f32;
                }
                xq[c * in_dim..(c + 1) * in_dim].copy_from_slice(&q);
                xs[c * (in_dim / 32)..(c + 1) * (in_dim / 32)].copy_from_slice(&s);
            }
            let mut y_ref = vec![0f32; rows * batch];
            let mut wq = vec![[0i8; 256]; n_super];
            let mut wdj = vec![[0f32; 16]; n_super];
            let mut wmu = vec![[0f32; 8]; n_super];
            for o in 0..rows {
                for sblk in 0..n_super {
                    let s = &bytes[(o * n_super + sblk) * src_b..];
                    let (q, dj, mu) = cpu_w4a8_super(s, ty);
                    wq[sblk] = q;
                    wdj[sblk] = dj;
                    wmu[sblk] = mu;
                }
                for c in 0..batch {
                    let xrow = &xq[c * in_dim..];
                    let srow = &xs[c * (in_dim / 32)..];
                    let sumrow = &sums[c * (in_dim / 32)..];
                    let mut acc = 0f64;
                    for sblk in 0..n_super {
                        for g16 in 0..16 {
                            let k0 = sblk * 256 + g16 * 16;
                            let dot: i32 = (0..16)
                                .map(|k| wq[sblk][g16 * 16 + k] as i32 * xrow[k0 + k] as i32)
                                .sum();
                            let db = srow[(sblk * 256 + g16 * 16) / 32];
                            acc += (wdj[sblk][g16] * db) as f64 * dot as f64;
                        }
                        if needs_sums {
                            for g in 0..8 {
                                acc += wmu[sblk][g] as f64 * sumrow[sblk * 8 + g] as f64;
                            }
                        }
                    }
                    y_ref[c * rows + o] = acc as f32;
                }
            }
            let gpu_sub: Vec<f32> = (0..batch)
                .flat_map(|c| y_gpu[c * out_dim..c * out_dim + rows].to_vec())
                .collect();
            let e = rel_err(&gpu_sub, &y_ref);
            eprintln!("{name} [{ty:?}] b={batch} dp4a vs quantized CPU ref rel_err {e:.2e}");
            assert!(e < 5e-4, "{name} b={batch}: dp4a decorrelated ({e:.2e})");
        }
    }
}

/// K-split mma GEMM (the 17..64 decode-batch rung) vs the same quantized CPU
/// reference the dp4a test uses. Same numeric class as dp4a (exact int dots,
/// f32 scale application) - only the K-fold order differs (z-slice partial
/// planes + fixed-order combine), so the gate is the dp4a one.
#[test]
fn mma_ks_matches_quantized_reference() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_mma_ks() {
        eprintln!("pack lacks kquant_gemm_mma_ks - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    let check_rows = 384usize;

    // 5 exercises the BN16 rung, 17 BN32, 33/64 BN64 (+ nz > 1 on every shape)
    for batch in [5usize, 17, 33, 64] {
        for (name, ty) in cases {
            let (info, bytes) = map.tensor_bytes(name).expect("tensor");
            assert_eq!(info.ggml_type, ty, "{name}: file type changed?");
            let kq = exec.repack_kquant(&map, name).expect("repack");
            let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
            let n_super = in_dim / 256;
            let src_b = bytes.len() / (n_super * out_dim);

            let x = deterministic_input(in_dim * batch, 11);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_xq = exec.alloc_i8(in_dim * batch).expect("xq");
            let mut d_xs = exec.alloc(in_dim / 32 * batch).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim * batch)
                .expect("quantize");
            let mut d_sums = exec.alloc(in_dim / 16 * batch).expect("sums");
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
                .expect("sums");
            let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);
            let mut d_part = exec.alloc(8 * out_dim * batch).expect("part");
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            exec.kquant_gemm_mma_ks(
                &kq,
                &d_xq,
                &d_xs,
                needs_sums.then_some(&d_sums),
                &mut d_part,
                &mut d_y,
                batch,
            )
            .expect("mma_ks");
            let y_gpu = exec.to_host(&d_y).expect("y host");

            let rows = check_rows.min(out_dim);
            let mut xq = vec![0i8; in_dim * batch];
            let mut xs = vec![0f32; (in_dim / 32) * batch];
            let mut sums = vec![0f32; (in_dim / 32) * batch];
            for c in 0..batch {
                let (q, s) = cpu_quantize_q8(&x[c * in_dim..(c + 1) * in_dim]);
                for b in 0..in_dim / 32 {
                    let ssum: i32 = q[b * 32..b * 32 + 32].iter().map(|&v| v as i32).sum();
                    sums[c * (in_dim / 32) + b] = s[b] * ssum as f32;
                }
                xq[c * in_dim..(c + 1) * in_dim].copy_from_slice(&q);
                xs[c * (in_dim / 32)..(c + 1) * (in_dim / 32)].copy_from_slice(&s);
            }
            let mut y_ref = vec![0f32; rows * batch];
            let mut wq = vec![[0i8; 256]; n_super];
            let mut wdj = vec![[0f32; 16]; n_super];
            let mut wmu = vec![[0f32; 8]; n_super];
            for o in 0..rows {
                for sblk in 0..n_super {
                    let s = &bytes[(o * n_super + sblk) * src_b..];
                    let (q, dj, mu) = cpu_w4a8_super(s, ty);
                    wq[sblk] = q;
                    wdj[sblk] = dj;
                    wmu[sblk] = mu;
                }
                for c in 0..batch {
                    let xrow = &xq[c * in_dim..];
                    let srow = &xs[c * (in_dim / 32)..];
                    let sumrow = &sums[c * (in_dim / 32)..];
                    let mut acc = 0f64;
                    for sblk in 0..n_super {
                        for g16 in 0..16 {
                            let k0 = sblk * 256 + g16 * 16;
                            let dot: i32 = (0..16)
                                .map(|k| wq[sblk][g16 * 16 + k] as i32 * xrow[k0 + k] as i32)
                                .sum();
                            let db = srow[(sblk * 256 + g16 * 16) / 32];
                            acc += (wdj[sblk][g16] * db) as f64 * dot as f64;
                        }
                        if needs_sums {
                            for g in 0..8 {
                                acc += wmu[sblk][g] as f64 * sumrow[sblk * 8 + g] as f64;
                            }
                        }
                    }
                    y_ref[c * rows + o] = acc as f32;
                }
            }
            let gpu_sub: Vec<f32> = (0..batch)
                .flat_map(|c| y_gpu[c * out_dim..c * out_dim + rows].to_vec())
                .collect();
            let e = rel_err(&gpu_sub, &y_ref);
            eprintln!("{name} [{ty:?}] b={batch} mma_ks vs quantized CPU ref rel_err {e:.2e}");
            assert!(e < 5e-4, "{name} b={batch}: mma_ks decorrelated ({e:.2e})");
        }
    }
}

/// K-split mma GEMM bandwidth at the 17..64 serving widths (vs the dp4a bench
/// above at the same shapes). `cargo test ... mma_ks_gemm_bench -- --ignored
/// --nocapture`
#[test]
#[ignore]
fn mma_ks_gemm_bench() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_mma_ks() {
        eprintln!("pack lacks kquant_gemm_mma_ks - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open");
    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    for batch in [2usize, 4, 8, 17, 32, 64] {
        for (name, tag) in cases {
            let kq = exec.repack_kquant(&map, name).expect("repack");
            let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
            let wbytes = (kq.data.len() + kq.scales.len()) as f64;
            let x = deterministic_input(in_dim * batch, 3);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_xq = exec.alloc_i8(in_dim * batch).expect("xq");
            let mut d_xs = exec.alloc(in_dim / 32 * batch).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim * batch)
                .expect("q");
            let mut d_sums = exec.alloc(in_dim / 16 * batch).expect("s");
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
                .expect("ss");
            let needs = matches!(tag, GgmlType::Q4K | GgmlType::Q5K);
            let mut d_part = exec.alloc(8 * out_dim * batch).expect("part");
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            exec.kquant_gemm_mma_ks(
                &kq,
                &d_xq,
                &d_xs,
                needs.then_some(&d_sums),
                &mut d_part,
                &mut d_y,
                batch,
            )
            .expect("warm");
            exec.stream.synchronize().unwrap();
            let reps = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                exec.kquant_gemm_mma_ks(
                    &kq,
                    &d_xq,
                    &d_xs,
                    needs.then_some(&d_sums),
                    &mut d_part,
                    &mut d_y,
                    batch,
                )
                .expect("gemm");
            }
            exec.stream.synchronize().unwrap();
            let dt = t0.elapsed().as_secs_f64() / reps as f64;
            eprintln!(
                "{:6} b={batch:2} {name:26} {:7.1} us  {:6.0} GB/s weight-effective",
                format!("{tag:?}"),
                dt * 1e6,
                wbytes / dt / 1e9
            );
        }
    }
}

/// dp4a decode-batch GEMM bandwidth at serving widths.
/// `cargo test ... dp4a_gemm_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn dp4a_gemm_bench() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open");
    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    for batch in [1usize, 2, 8, 16, 32] {
        for (name, tag) in cases {
            let kq = exec.repack_kquant(&map, name).expect("repack");
            let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
            let wbytes = (kq.data.len() + kq.scales.len()) as f64;
            let x = deterministic_input(in_dim * batch, 3);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_xq = exec.alloc_i8(in_dim * batch).expect("xq");
            let mut d_xs = exec.alloc(in_dim / 32 * batch).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim * batch)
                .expect("q");
            let mut d_sums = exec.alloc(in_dim / 16 * batch).expect("s");
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
                .expect("ss");
            let needs = matches!(tag, GgmlType::Q4K | GgmlType::Q5K);
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            exec.kquant_gemm_dp4a(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y, batch)
                .expect("warm");
            exec.stream.synchronize().unwrap();
            let reps = 200;
            let mut dt = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    exec.kquant_gemm_dp4a(
                        &kq,
                        &d_xq,
                        &d_xs,
                        needs.then_some(&d_sums),
                        &mut d_y,
                        batch,
                    )
                    .expect("gemv");
                }
                exec.stream.synchronize().unwrap();
                dt = dt.min(t0.elapsed().as_secs_f64() / reps as f64);
            }
            eprintln!(
                "{:6} b={batch:2} {name:26} {:7.1} us  {:6.0} GB/s weight-effective",
                format!("{tag:?}"),
                dt * 1e6,
                wbytes / dt / 1e9
            );
        }
    }
}

/// W4A8 vs the stage-1 interim (dequant_rp + gemm_f32) at prefill batches.
/// `cargo test ... w4a8_gemm_bench -- --ignored --nocapture`
#[test]
#[ignore]
fn w4a8_gemm_bench() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open");

    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    for batch in [128usize, 512, 1024] {
        for (name, ty) in cases {
            let kq = exec.repack_kquant(&map, name).expect("repack");
            let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
            let x = deterministic_input(in_dim * batch, 7);
            let d_x = exec.to_device(&x).expect("x");
            let n_chunks = in_dim.div_ceil(128);
            let batch_pad = batch.div_ceil(128) * 128;
            let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
            let needs_sums = matches!(ty, GgmlType::Q4K | GgmlType::Q5K);
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            let mut d_wdq = exec.alloc(in_dim * out_dim).expect("wdq");

            let w4a8 = |d_yq: &mut _, d_sums: &mut _, d_y: &mut _| {
                exec.quantize_q8_mmq(&d_x, d_yq, in_dim, batch).unwrap();
                exec.mmq_sums(d_yq, d_sums, in_dim, batch).unwrap();
                exec.kquant_gemm_w4a8(&kq, d_yq, needs_sums.then_some(&*d_sums), d_y, batch)
                    .unwrap();
            };
            w4a8(&mut d_yq, &mut d_sums, &mut d_y);
            exec.stream.synchronize().unwrap();
            let reps = 50;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                w4a8(&mut d_yq, &mut d_sums, &mut d_y);
            }
            exec.stream.synchronize().unwrap();
            let t_w4a8 = t0.elapsed().as_secs_f64() / reps as f64;

            // stage-1 interim on the same shapes
            exec.kquant_dequant_rp(&kq, &mut d_wdq).unwrap();
            exec.gemm_f32(&d_wdq, in_dim, out_dim, &d_x, &mut d_y, batch)
                .unwrap();
            exec.stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                exec.kquant_dequant_rp(&kq, &mut d_wdq).unwrap();
                exec.gemm_f32(&d_wdq, in_dim, out_dim, &d_x, &mut d_y, batch)
                    .unwrap();
            }
            exec.stream.synchronize().unwrap();
            let t_f32 = t0.elapsed().as_secs_f64() / reps as f64;

            let macs = in_dim as f64 * out_dim as f64 * batch as f64;
            eprintln!(
                "{:6} b={batch:4} {name:26} w4a8 {:8.1} us ({:6.1} TOPS)  f32-interim {:8.1} us  speedup {:.1}x",
                format!("{ty:?}"),
                t_w4a8 * 1e6,
                2.0 * macs / t_w4a8 / 1e12,
                t_f32 * 1e6,
                t_f32 / t_w4a8
            );
        }
    }
}

/// Bandwidth microbench: kquant_gemv vs the Q8 repacked GEMV on real tensors.
/// `cargo test ... kquant_gemv_bandwidth -- --ignored --nocapture`
#[test]
#[ignore]
fn kquant_gemv_bandwidth() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open");
    let q8_map = MappedGguf::open(&model.with_file_name("Qwen3.5-9B-Q8_0.gguf")).expect("q8");

    let cases = [
        ("blk.0.ffn_gate.weight", "Q4K"),
        ("blk.1.ffn_down.weight", "Q5K"),
        ("blk.0.ffn_down.weight", "Q6K"),
        ("output.weight", "Q6K-head"),
    ];
    for (name, tag) in cases {
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let bytes = (kq.data.len() + kq.scales.len()) as f64;
        let x = deterministic_input(in_dim, 3);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(out_dim).expect("y");
        exec.kquant_gemv(&kq, &d_x, &mut d_y).expect("warm");
        exec.stream.synchronize().unwrap();
        // best-of-3 segments: clock-ramp / neighbor noise moved single-shot
        // numbers by +-15%, well above the effects being measured
        let reps = 200;
        let mut dt = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                exec.kquant_gemv(&kq, &d_x, &mut d_y).expect("gemv");
            }
            exec.stream.synchronize().unwrap();
            dt = dt.min(t0.elapsed().as_secs_f64() / reps as f64);
        }
        eprintln!(
            "{tag:10} {name:26} [{in_dim:6}x{out_dim:6}] {:7.1} us  {:6.0} GB/s",
            dt * 1e6,
            bytes / dt / 1e9
        );
        // the W4A8 serving GEMV on the same tensor
        if exec.has_kquant_gemv_w4a8() {
            let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
            let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim)
                .expect("q");
            let mut d_sums = exec.alloc(in_dim / 16).expect("s");
            let needs = matches!(kq.ty, GgmlType::Q4K | GgmlType::Q5K);
            if needs {
                exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, 1)
                    .expect("ss");
            }
            exec.kquant_gemv_w4a8(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y)
                .expect("warm");
            exec.stream.synchronize().unwrap();
            let mut dt8 = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    exec.kquant_gemv_w4a8(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y)
                        .expect("gemv");
                }
                exec.stream.synchronize().unwrap();
                dt8 = dt8.min(t0.elapsed().as_secs_f64() / reps as f64);
            }
            eprintln!(
                "{:10} {name:26} [{in_dim:6}x{out_dim:6}] {:7.1} us  {:6.0} GB/s",
                format!("{tag}-w4a8"),
                dt8 * 1e6,
                bytes / dt8 / 1e9
            );
        }
        // the multi-column GEMV at the spec-verify widths (weight-effective
        // GB/s - flops scale with r but the weight read shouldn't)
        if exec.has_kquant_gemv_w4a8_nc() {
            for r in [2usize, 4, 5] {
                if !GpuExecutor::kquant_gemv_w4a8_nc_fits(&kq, r) {
                    continue;
                }
                let mut xall = Vec::with_capacity(r * in_dim);
                for c in 0..r {
                    xall.extend_from_slice(&deterministic_input(in_dim, 3 + c as u64));
                }
                let d_xa = exec.to_device(&xall).expect("xa");
                let mut d_xq = exec.alloc_i8(r * in_dim).expect("xq");
                let mut d_xs = exec.alloc(r * in_dim / 32).expect("xs");
                exec.quantize_q8(&d_xa, &mut d_xq, &mut d_xs, r * in_dim)
                    .expect("q");
                let mut d_sums = exec.alloc(r * in_dim / 16).expect("s");
                let needs = matches!(kq.ty, GgmlType::Q4K | GgmlType::Q5K);
                if needs {
                    exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, r)
                        .expect("ss");
                }
                let mut d_yn = exec.alloc(r * out_dim).expect("yn");
                exec.kquant_gemv_w4a8_nc(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_yn, r)
                    .expect("warm");
                exec.stream.synchronize().unwrap();
                let mut dtn = f64::MAX;
                for _ in 0..3 {
                    let t0 = std::time::Instant::now();
                    for _ in 0..reps {
                        exec.kquant_gemv_w4a8_nc(
                            &kq,
                            &d_xq,
                            &d_xs,
                            needs.then_some(&d_sums),
                            &mut d_yn,
                            r,
                        )
                        .expect("gemv");
                    }
                    exec.stream.synchronize().unwrap();
                    dtn = dtn.min(t0.elapsed().as_secs_f64() / reps as f64);
                }
                eprintln!(
                    "{:10} {name:26} [{in_dim:6}x{out_dim:6}] {:7.1} us  {:6.0} GB/s",
                    format!("{tag}-nc{r}"),
                    dtn * 1e6,
                    bytes / dtn / 1e9
                );
            }
        }
    }
    // Q8_0 baseline on the equivalent tensor
    for name in ["blk.0.ffn_gate.weight", "output.weight"] {
        let q8 = exec.repack_q8(&q8_map, name).expect("repack q8");
        let (in_dim, out_dim) = (q8.dims[0], q8.dims[1]);
        let bytes = (q8.data.len() + q8.scale.len()) as f64;
        let x = deterministic_input(in_dim, 3);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(out_dim).expect("y");
        exec.q8_0_gemv_repacked(&q8, None, &d_x, &mut d_y)
            .expect("warm");
        exec.stream.synchronize().unwrap();
        let reps = 200;
        let mut dt = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                exec.q8_0_gemv_repacked(&q8, None, &d_x, &mut d_y)
                    .expect("gemv");
            }
            exec.stream.synchronize().unwrap();
            dt = dt.min(t0.elapsed().as_secs_f64() / reps as f64);
        }
        eprintln!(
            "{:10} {name:26} [{in_dim:6}x{out_dim:6}] {:7.1} us  {:6.0} GB/s",
            "Q8_0-ref",
            dt * 1e6,
            bytes / dt / 1e9
        );
    }
}

#[test]
fn kquant_paths_match_cpu_reference_on_ud_file() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant() {
        panic!("pack lacks the k-quant family");
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    // one representative tensor per format (verified types in this UD file)
    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];

    // Independent cross-quant anchor: the Q8_0 export of the same model,
    // dequanted through the long-standing load_f32 path. Guards against a
    // spec misreading that is self-consistent between the kernels and the
    // in-test CPU port (the Q5_K qh/qs source-order bug survived exactly
    // that hole on first landing).
    let q8_map = {
        let q8 = model.with_file_name("Qwen3.5-9B-Q8_0.gguf");
        q8.exists().then(|| MappedGguf::open(&q8).expect("open q8"))
    };

    for (name, want_ty) in cases {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, want_ty, "{name}: file type changed?");
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let n: usize = dims.iter().product();
        let (in_dim, out_dim) = (dims[0], dims[1]);
        let cpu = cpu_dequant(bytes, want_ty, n);

        if let Some(q8m) = &q8_map {
            let anchor = paddock_engine::reference::load_f32(q8m, name).expect("q8 anchor");
            let e = rel_err(&cpu, &anchor.data);
            eprintln!("{name} [{want_ty:?}] vs Q8_0-file anchor rel_err {e:.2e}");
            assert!(
                e < 0.15,
                "{name}: CPU reference decorrelated from the Q8_0 export ({e:.2e}) - \
                 format spec misread"
            );
        }

        // 1) raw-layout dequant (upload's kq arm)
        let gpu_f32 = exec.upload(&map, name).expect("upload dequant");
        let raw = exec.to_host(&gpu_f32.buf).expect("to host");
        let e1 = rel_err(&raw, &cpu);
        eprintln!("{name} [{want_ty:?}] raw-dequant rel_err {e1:.2e}");
        assert!(e1 < 1e-6, "{name}: raw dequant mismatch ({e1:.2e})");

        // 2) repack + dequant_rp must reproduce the same table
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let mut d_rp = exec.alloc(n).expect("alloc rp");
        exec.kquant_dequant_rp(&kq, &mut d_rp).expect("dequant_rp");
        let rp = exec.to_host(&d_rp).expect("to host");
        let e2 = rel_err(&rp, &cpu);
        eprintln!("{name} [{want_ty:?}] repack+rp-dequant rel_err {e2:.2e}");
        assert!(e2 < 1e-6, "{name}: repacked dequant mismatch ({e2:.2e})");

        // 3) gemv vs CPU dot
        let x = deterministic_input(in_dim, 42);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(out_dim).expect("y");
        exec.kquant_gemv(&kq, &d_x, &mut d_y).expect("gemv");
        let gy = exec.to_host(&d_y).expect("y host");
        let mut cy = vec![0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0f64;
            for i in 0..in_dim {
                acc += (cpu[o * in_dim + i] as f64) * (x[i] as f64);
            }
            cy[o] = acc as f32;
        }
        let e3 = rel_err(&gy, &cy);
        eprintln!("{name} [{want_ty:?}] gemv rel_err {e3:.2e}");
        assert!(e3 < 1e-4, "{name}: gemv mismatch ({e3:.2e})");

        // 4) prefill interim: dequant_rp + the pd_gemm_f32 export at a few
        // batch shapes. That export's elected default is the
        // 3xTF32 tensor-core arm, so this rung measures that class, and the
        // gate is set from its measured band with the two-level accumulation
        // drain: 4.5-5.4e-7 across all three tensors. 1e-5 is
        // ~20x margin and still trips every known regression class: single
        // tf32 reads 2.8e-4, and an un-drained mma C-chain reads K*2^-27
        // (3.8e-5 at K=4096 - the DPU truncation bias this gate caught).
        for batch in [2usize, 33, 128] {
            let xb = deterministic_input(batch * in_dim, 7 + batch as u64);
            let d_xb = exec.to_device(&xb).expect("xb");
            let mut d_yb = exec.alloc(batch * out_dim).expect("yb");
            exec.gemm_f32(&d_rp, in_dim, out_dim, &d_xb, &mut d_yb, batch)
                .expect("gemm_f32");
            let gyb = exec.to_host(&d_yb).expect("yb host");
            let mut cyb = vec![0f32; batch * out_dim];
            for r in 0..batch {
                for o in 0..out_dim {
                    let mut acc = 0f64;
                    for i in 0..in_dim {
                        acc += (cpu[o * in_dim + i] as f64) * (xb[r * in_dim + i] as f64);
                    }
                    cyb[r * out_dim + o] = acc as f32;
                }
            }
            let e4 = rel_err(&gyb, &cyb);
            eprintln!("{name} [{want_ty:?}] gemm_f32 b={batch} rel_err {e4:.2e}");
            assert!(e4 < 1e-5, "{name}: gemm_f32 b={batch} mismatch ({e4:.2e})");
        }
    }

    // 5) embedding gather on the Q4_K token table
    let (info, bytes) = map.tensor_bytes("token_embd.weight").expect("embd");
    assert_eq!(info.ggml_type, GgmlType::Q4K);
    let embd = info.dims[0] as usize;
    let kq = exec
        .repack_kquant(&map, "token_embd.weight")
        .expect("repack embd");
    let tokens: Vec<u32> = vec![0, 1, 17, 4096, 151935];
    let d_tok = exec.to_device_u32(&tokens).expect("tokens");
    let mut d_out = exec.alloc(tokens.len() * embd).expect("out");
    exec.kquant_gather(&kq, &d_tok, &mut d_out, embd, tokens.len())
        .expect("gather");
    let g = exec.to_host(&d_out).expect("host");
    for (i, &t) in tokens.iter().enumerate() {
        let row_bytes = &bytes[(t as usize) * (embd / 256) * 144..];
        let cpu_row = cpu_dequant(row_bytes, GgmlType::Q4K, embd);
        let e = rel_err(&g[i * embd..(i + 1) * embd], &cpu_row);
        eprintln!("token_embd row {t} gather rel_err {e:.2e}");
        assert!(e < 1e-6, "gather row {t} mismatch ({e:.2e})");
    }
}

/// Exact-int dot of one k-quant row against a quantized activation row -
/// the same W4A8 model as the dense tests (int dots, f32 scale application,
/// f64 accumulation; the Q4/Q5 mu term rides per-32 activation sums).
fn kq_row_dot(bytes: &[u8], ty: GgmlType, src_b: usize, row: usize, xq: &[i8], xs: &[f32]) -> f64 {
    let in_dim = xq.len();
    let n_super = in_dim / 256;
    let mut acc = 0f64;
    for sblk in 0..n_super {
        let (q, dj, mu) = cpu_w4a8_super(&bytes[(row * n_super + sblk) * src_b..], ty);
        for g16 in 0..16 {
            let k0 = sblk * 256 + g16 * 16;
            let dot: i32 = (0..16)
                .map(|k| q[g16 * 16 + k] as i32 * xq[k0 + k] as i32)
                .sum();
            acc += (dj[g16] * xs[k0 / 32]) as f64 * dot as f64;
        }
        if matches!(ty, GgmlType::Q4K | GgmlType::Q5K) {
            for (g, &mu_g) in mu.iter().enumerate() {
                let k0 = sblk * 256 + g * 32;
                let s32: i32 = (0..32).map(|k| xq[k0 + k] as i32).sum();
                acc += (mu_g * xs[k0 / 32]) as f64 * s32 as f64;
            }
        }
    }
    acc
}

/// W4A8 b=1 GEMV SMALL-OUT (<2048) rung - the ROWS=2 variant the qwen/laguna
/// wk/wv and shexp planes ride. The original test's three
/// cases all have out >= 2048 (the ROWS=4 rung), so the <2> rung shipped
/// UNGATED - and it was the shexp gate_up [2048 -> 1024] Q4_K plane served
/// by it that turned out to be flipping decode catastrophically.
#[test]
fn kq_w4a8_gemv_small_out_rung() {
    let Some(model) = common::model("LAGUNA_GGUF", common::LAGUNA_XS_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemv_w4a8() {
        eprintln!("pack lacks kquant_gemv_w4a8 - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    // small-out (<2048) planes of each dtype present in the file
    let cases = [
        "blk.1.ffn_gate_shexp.weight", // Q4_K [2048, 512] expected
        "blk.1.ffn_up_shexp.weight",
        "blk.1.attn_k.weight", // [2048, 1024]
        "blk.1.attn_v.weight",
    ];
    for name in cases {
        let Ok((info, bytes)) = map.tensor_bytes(name) else {
            eprintln!("{name}: not in file - skipping");
            continue;
        };
        let want_ty = info.ggml_type;
        if !matches!(want_ty, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K) {
            eprintln!("{name}: {want_ty:?} not a k-quant - skipping");
            continue;
        }
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let src_b = bytes.len() / (in_dim / 256 * out_dim);

        let x = deterministic_input(in_dim, 31);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
        let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim)
            .expect("quantize");
        let mut d_sums = exec.alloc(in_dim / 16).expect("sums");
        let needs = matches!(want_ty, GgmlType::Q4K | GgmlType::Q5K);
        if needs {
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, 1)
                .expect("sums");
        }
        let mut d_y = exec.alloc(out_dim).expect("y");
        exec.kquant_gemv_w4a8(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y)
            .expect("w4a8 gemv");
        let gy = exec.to_host(&d_y).expect("y host");

        let (xq, xs) = cpu_quantize_q8(&x);
        let mut cy = vec![0f32; out_dim];
        for (o, c) in cy.iter_mut().enumerate() {
            *c = kq_row_dot(bytes, want_ty, src_b, o, &xq, &xs) as f32;
        }
        let e = rel_err(&gy, &cy);
        eprintln!("{name} [{want_ty:?}] {in_dim}x{out_dim} w4a8 gemv rel_err {e:.2e}");
        assert!(e < 5e-4, "{name}: small-out w4a8 gemv mismatch ({e:.2e})");
    }
}

/// W4A8 b=1 decode GEMV (the mmvq-class serving default) vs the exact-int
/// CPU model on real UD tensors - all four formats.
#[test]
fn kq_w4a8_gemv_matches_int_reference() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemv_w4a8() {
        eprintln!("pack lacks kquant_gemv_w4a8 - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
        // IQ4_XS: the 2026-07 unsloth UD recipes carry no IQ4_XS tensors (the
        // elected *-MTP-GGUF line is Q4K/Q5K/Q6K + Q8_0) - the real-file case is
        // retired; the kernels stay for imported files. Synthetic coverage TODO.
    ];
    for (name, want_ty) in cases {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, want_ty);
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let src_b = bytes.len() / (in_dim / 256 * out_dim);

        let x = deterministic_input(in_dim, 31);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
        let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim)
            .expect("quantize");
        let mut d_sums = exec.alloc(in_dim / 16).expect("sums");
        let needs = matches!(want_ty, GgmlType::Q4K | GgmlType::Q5K);
        if needs {
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, 1)
                .expect("sums");
        }
        let mut d_y = exec.alloc(out_dim).expect("y");
        exec.kquant_gemv_w4a8(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y)
            .expect("w4a8 gemv");
        let gy = exec.to_host(&d_y).expect("y host");

        // CPU: mirror the quantize, then the exact-int row dot
        let (xq, xs) = cpu_quantize_q8(&x);
        let mut cy = vec![0f32; out_dim];
        for (o, c) in cy.iter_mut().enumerate() {
            *c = kq_row_dot(bytes, want_ty, src_b, o, &xq, &xs) as f32;
        }
        let e = rel_err(&gy, &cy);
        eprintln!("{name} [{want_ty:?}] w4a8 gemv rel_err {e:.2e}");
        assert!(e < 5e-4, "{name}: w4a8 gemv mismatch ({e:.2e})");
    }
}

/// Multi-segment W4A8 GEMV (granite's decode QKV / gate|up one-launch merge,
/// mixed k-quant dtypes). Two numeric gates, mirroring the nc test's split:
/// planes whose solo launcher election is already `<4,128>` (q at out 4096,
/// gate/up at out 32768 - the mu out_dim>=2048 bucket) must be BIT-identical
/// to the single kernel; k/v run `<4,512>` solo, so their merged `<4,128>`
/// rows are a sanctioned TPR regrouping - those gate against the exact-int
/// CPU reference (the same 5e-4 anchor the single kernel's own test uses).
#[test]
fn kq_w4a8_multi_matches_single() {
    let Some(model) = common::model("GRANITE_30B_GGUF", common::GRANITE_30B_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemv_w4a8_multi() {
        eprintln!("pack lacks kquant_gemv_w4a8_multi - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let wq = exec.repack_kquant(&map, "blk.0.attn_q.weight").expect("q");
    let wk = exec.repack_kquant(&map, "blk.0.attn_k.weight").expect("k");
    let wv = exec.repack_kquant(&map, "blk.0.attn_v.weight").expect("v");
    assert_eq!(wq.ty, GgmlType::Q4K);
    assert_eq!(wv.ty, GgmlType::Q6K, "Q4_K_M pairs Q6_K v with Q4_K q/k");
    let in_dim = wq.dims[0];

    let x = deterministic_input(in_dim, 47);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
    let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
    let mut d_sums = exec.alloc(in_dim / 16).expect("sums");
    exec.quantize_q8_sums(&d_x, &mut d_xq, &mut d_xs, &mut d_sums, in_dim)
        .expect("quantize");

    // single-kernel outputs at the production election
    let mut d_q1 = exec.alloc(wq.dims[1]).expect("q1");
    let mut d_k1 = exec.alloc(wk.dims[1]).expect("k1");
    let mut d_v1 = exec.alloc(wv.dims[1]).expect("v1");
    exec.kquant_gemv_w4a8(&wq, &d_xq, &d_xs, Some(&d_sums), &mut d_q1)
        .expect("q single");
    exec.kquant_gemv_w4a8(&wk, &d_xq, &d_xs, Some(&d_sums), &mut d_k1)
        .expect("k single");
    exec.kquant_gemv_w4a8(&wv, &d_xq, &d_xs, None, &mut d_v1)
        .expect("v single");

    let mut d_q2 = exec.alloc(wq.dims[1]).expect("q2");
    let mut d_k2 = exec.alloc(wk.dims[1]).expect("k2");
    let mut d_v2 = exec.alloc(wv.dims[1]).expect("v2");
    exec.kquant_gemv_w4a8_multi(
        &mut [(&wq, &mut d_q2), (&wk, &mut d_k2), (&wv, &mut d_v2)],
        &d_xq,
        &d_xs,
        &d_sums,
    )
    .expect("qkv multi");

    let (q1, q2) = (exec.to_host(&d_q1).unwrap(), exec.to_host(&d_q2).unwrap());
    assert_eq!(
        q1, q2,
        "q plane must be BIT-identical (same <4,128> config)"
    );

    // k/v: merged TPR=32 vs solo TPR=128 - anchor against the exact-int CPU
    // reference, same bound as the single kernel's own gate
    let (xq_h, xs_h) = cpu_quantize_q8(&x);
    for (name, w, d_y) in [
        ("blk.0.attn_k.weight", &wk, &d_k2),
        ("blk.0.attn_v.weight", &wv, &d_v2),
    ] {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, w.ty);
        let src_b = bytes.len() / (in_dim / 256 * w.dims[1]);
        let gy = exec.to_host(d_y).unwrap();
        let mut cy = vec![0f32; w.dims[1]];
        for (o, c) in cy.iter_mut().enumerate() {
            *c = kq_row_dot(bytes, w.ty, src_b, o, &xq_h, &xs_h) as f32;
        }
        let e = rel_err(&gy, &cy);
        eprintln!("{name} [{:?}] multi gemv rel_err {e:.2e}", w.ty);
        assert!(e < 5e-4, "{name}: multi gemv mismatch ({e:.2e})");
    }

    // gate|up: both Q4_K in the <4,128> bucket - 2-segment merge, bit-exact
    let wg = exec
        .repack_kquant(&map, "blk.0.ffn_gate.weight")
        .expect("gate");
    let wu = exec.repack_kquant(&map, "blk.0.ffn_up.weight").expect("up");
    let mut d_g1 = exec.alloc(wg.dims[1]).expect("g1");
    let mut d_u1 = exec.alloc(wu.dims[1]).expect("u1");
    exec.kquant_gemv_w4a8(&wg, &d_xq, &d_xs, Some(&d_sums), &mut d_g1)
        .expect("gate single");
    exec.kquant_gemv_w4a8(&wu, &d_xq, &d_xs, Some(&d_sums), &mut d_u1)
        .expect("up single");
    let mut d_g2 = exec.alloc(wg.dims[1]).expect("g2");
    let mut d_u2 = exec.alloc(wu.dims[1]).expect("u2");
    exec.kquant_gemv_w4a8_multi(
        &mut [(&wg, &mut d_g2), (&wu, &mut d_u2)],
        &d_xq,
        &d_xs,
        &d_sums,
    )
    .expect("gate|up multi");
    let (g1, g2) = (exec.to_host(&d_g1).unwrap(), exec.to_host(&d_g2).unwrap());
    let (u1, u2) = (exec.to_host(&d_u1).unwrap(), exec.to_host(&d_u2).unwrap());
    assert_eq!(
        g1, g2,
        "gate plane must be BIT-identical (same <4,128> config)"
    );
    assert_eq!(
        u1, u2,
        "up plane must be BIT-identical (same <4,128> config)"
    );
}

/// Multi-column W4A8 GEMV (the spec-verify r-class). Two numeric classes,
/// split by FORMAT (not by r):
///
/// - Q6K/IQ4_XS (non-mu): BIT-identity against the single-column GEMV run
///   per column. Both launchers land the same threads-per-row in both
///   out_dim buckets (nc ROWS=4/256t = TPR 64 vs b=1 <4,256>; nc ROWS=2 =
///   TPR 128 vs b=1 <4,512>), so the chunk walk and fold order coincide
///   exactly.
/// - Q4K/Q5K (mu): tolerance parity against the single-column GEMV AND the
///   exact-int CPU reference (the correctness anchor). The mu NT
///   election (out_dim >= 2048 picks NT in {128,256,512} to maximize
///   resident threads, per-shape and per-die) regroups the b=1 GEMV's f32
///   accumulation - the launcher's own comment names this the sanctioned
///   TPR-regrouping class - so bit-identity with the fixed-TPR nc kernel
///   holds only when the election happens to land TPR 64, which is a
///   per-shape/per-arch accident, not a contract. This gate sat dead
///   (broken model path) when that election landed and came back RED on the
///   stale bit-identity claim: most rows differed, at reorder-noise size
///   The r >= 4 arm additionally uses the per-row mu FOLD
///   (a second sanctioned reorder), same gating either way.
#[test]
fn kq_w4a8_nc_bitmatches_single_column() {
    let Some(model) = common::model("QWEN35_UD_GGUF", common::QWEN35_9B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemv_w4a8_nc() {
        eprintln!("pack lacks kquant_gemv_w4a8_nc - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let cases = [
        ("blk.0.ffn_gate.weight", GgmlType::Q4K),
        ("blk.1.ffn_down.weight", GgmlType::Q5K),
        ("blk.0.ffn_down.weight", GgmlType::Q6K),
    ];
    for (name, want_ty) in cases {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor");
        assert_eq!(info.ggml_type, want_ty);
        let kq = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (kq.dims[0], kq.dims[1]);
        let src_b = bytes.len() / (in_dim / 256 * out_dim);
        let needs = matches!(want_ty, GgmlType::Q4K | GgmlType::Q5K);
        // all launcher-legal widths, INCLUDING non-elected (format, r) pairs
        // - fits() is dispatch policy, the kernel must be correct everywhere
        for r in [2usize, 4, 5] {
            // r distinct activation rows, strided layout
            let mut xall = Vec::with_capacity(r * in_dim);
            for c in 0..r {
                xall.extend_from_slice(&deterministic_input(in_dim, 31 + c as u64));
            }
            let d_x = exec.to_device(&xall).expect("x");
            let mut d_xq = exec.alloc_i8(r * in_dim).expect("xq");
            let mut d_xs = exec.alloc(r * in_dim / 32).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, r * in_dim)
                .expect("quantize");
            let mut d_sums = exec.alloc(r * in_dim / 16).expect("sums");
            if needs {
                exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, r)
                    .expect("sums");
            }
            let mut d_y = exec.alloc(r * out_dim).expect("y");
            exec.kquant_gemv_w4a8_nc(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y, r)
                .expect("nc gemv");
            let ync = exec.to_host(&d_y).expect("y host");
            // single-column reference: re-quantize the column's f32 row -
            // per-32 blocks never straddle rows (in_dim % 256 == 0), so the
            // quantize is bitwise the strided run's row c
            for c in 0..r {
                let d_cx = exec
                    .to_device(&xall[c * in_dim..(c + 1) * in_dim])
                    .expect("cx");
                let mut d_cq = exec.alloc_i8(in_dim).expect("cq");
                let mut d_cs = exec.alloc(in_dim / 32).expect("cs");
                exec.quantize_q8(&d_cx, &mut d_cq, &mut d_cs, in_dim)
                    .expect("cq quant");
                let mut d_cm = exec.alloc(in_dim / 16).expect("cm");
                if needs {
                    exec.q8_sums_strided(&d_cq, &mut d_cm, in_dim, 1)
                        .expect("cm sums");
                }
                let mut d_cy = exec.alloc(out_dim).expect("cy");
                exec.kquant_gemv_w4a8(&kq, &d_cq, &d_cs, needs.then_some(&d_cm), &mut d_cy)
                    .expect("single gemv");
                let y1 = exec.to_host(&d_cy).expect("cy host");
                let nc_col = &ync[c * out_dim..(c + 1) * out_dim];
                if needs {
                    // mu formats: TPR-regrouping class vs b=1 (and at r >= 4
                    // additionally the per-row mu fold) - tolerance gates,
                    // anchored on the exact-int CPU reference
                    let e1 = rel_err(nc_col, &y1);
                    let (cxq, cxs) = cpu_quantize_q8(&xall[c * in_dim..(c + 1) * in_dim]);
                    let mut cy = vec![0f32; out_dim];
                    for (o, v) in cy.iter_mut().enumerate() {
                        *v = kq_row_dot(bytes, want_ty, src_b, o, &cxq, &cxs) as f32;
                    }
                    let e2 = rel_err(nc_col, &cy);
                    eprintln!(
                        "{name} r={r} col {c}: mu rel_err {e1:.2e} (vs b=1) \
                         {e2:.2e} (vs CPU)"
                    );
                    assert!(e1 < 1e-4, "{name} r={r} col {c}: vs b=1 GEMV ({e1:.2e})");
                    assert!(e2 < 5e-4, "{name} r={r} col {c}: vs CPU ref ({e2:.2e})");
                } else {
                    let diff = nc_col
                        .iter()
                        .zip(&y1)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        diff, 0,
                        "{name} r={r} col {c}: {diff} elements differ from the single-column GEMV"
                    );
                }
            }
            let class = if needs {
                "tolerance (mu class)"
            } else {
                "BIT-identical"
            };
            eprintln!("{name} [{want_ty:?}] nc r={r}: {class} vs single-column ✓");
        }
    }
}

/// Fused ab matvec + delta gate vs the exact two-launch pair - the fused
/// kernel claims the matvec's per-element summation schedule and the gate's
/// expressions verbatim, so the gate is BIT-identity on g and beta
/// (synthetic tensors, no model file needed).
#[test]
fn matvec_ab_gate_bitmatches_pair() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_matvec_ab_gate() {
        eprintln!("pack lacks matvec_ab_gate - skipping");
        return;
    }
    let (in_dim, n_heads) = (2048usize, 48usize);
    let out_dim = 2 * n_heads;
    let wv = deterministic_input(in_dim * out_dim, 7);
    let ab = paddock_engine::gpu::DeviceTensor {
        buf: exec.to_device(&wv).expect("w"),
        dims: vec![in_dim, out_dim],
    };
    let ssm_a = exec
        .to_device(&deterministic_input(n_heads, 11))
        .expect("a");
    let dt_bias = exec
        .to_device(&deterministic_input(n_heads, 13))
        .expect("dt");
    for n in [1usize, 2, 5, 8] {
        let x = exec
            .to_device(&deterministic_input(n * in_dim, 17))
            .expect("x");
        // exact pair
        let mut d_ab = exec.alloc(n * out_dim).expect("ab");
        let mut g0 = exec.alloc(n * n_heads).expect("g0");
        let mut b0 = exec.alloc(n * n_heads).expect("b0");
        exec.matvec_f32_batch(&ab, &x, &mut d_ab, n)
            .expect("matvec");
        exec.delta_gate_ab(&d_ab, &ssm_a, &dt_bias, &mut g0, &mut b0, n, n_heads)
            .expect("gate");
        // fused
        let mut g1 = exec.alloc(n * n_heads).expect("g1");
        let mut b1 = exec.alloc(n * n_heads).expect("b1");
        exec.matvec_ab_gate(&ab, &x, &ssm_a, &dt_bias, &mut g1, &mut b1, n, n_heads)
            .expect("fused");
        let (hg0, hb0) = (exec.to_host(&g0).unwrap(), exec.to_host(&b0).unwrap());
        let (hg1, hb1) = (exec.to_host(&g1).unwrap(), exec.to_host(&b1).unwrap());
        let dg = hg0
            .iter()
            .zip(&hg1)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let db = hb0
            .iter()
            .zip(&hb1)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(dg + db, 0, "n={n}: {dg} g / {db} beta elements differ");
        eprintln!("ab_gate n={n}: BIT-identical to the pair ✓");
    }
}

/// k-quant MoE expert pair (token-batched decode class) on the 35B-A3B UD
/// file's real expert tensors, vs the exact-int CPU model above. The CPU
/// re-quantizes the identical f32 inputs with the mirrored math, so each
/// kernel is isolated: gate_up gated on its fused output, down on the
/// combined proj (its quantized operands come from the GPU's fused values).
#[test]
fn kq_moe_pair_matches_quantized_reference() {
    let Some(model) = common::model("QWEN36_MOE_UD_GGUF", common::QWEN36_35B_A3B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_moe() {
        eprintln!("pack lacks the k-quant MoE pair - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    // first block whose gate/up/down expert tensors are all k-quant
    let mut picked = None;
    for i in 0..8 {
        let names = [
            format!("blk.{i}.ffn_gate_exps.weight"),
            format!("blk.{i}.ffn_up_exps.weight"),
            format!("blk.{i}.ffn_down_exps.weight"),
        ];
        let tys: Vec<_> = names
            .iter()
            .filter_map(|n| map.tensor_bytes(n).ok().map(|(info, _)| info.ggml_type))
            .collect();
        if tys.len() == 3
            && tys.iter().all(|t| {
                matches!(
                    t,
                    GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Iq4Xs
                )
            })
        {
            picked = Some((names, tys));
            break;
        }
    }
    let Some((names, tys)) = picked else {
        eprintln!("no all-k-quant expert block in the first 8 layers - skipping");
        return;
    };
    let gate = exec.repack_kquant(&map, &names[0]).expect("repack gate");
    let up = exec.repack_kquant(&map, &names[1]).expect("repack up");
    let down = exec.repack_kquant(&map, &names[2]).expect("repack down");
    let (embd, ff, ne) = (gate.dims[0], gate.dims[1], gate.dims[2]);
    assert_eq!(down.dims[0], ff);
    assert_eq!(down.dims[1], embd);
    eprintln!("expert block {} [{embd}x{ff}x{ne}] types {tys:?}", names[0]);

    let batch = 3usize;
    let n_active = 8usize;
    // deterministic routing across distinct experts (incl. repeats)
    let idx_h: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 7) % ne as u32)
        .collect();
    let topk_h: Vec<f32> = (0..batch * n_active)
        .map(|i| 1.0 / (1.0 + (i % n_active) as f32))
        .collect();
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_topk = exec.to_device(&topk_h).expect("topk");

    // activations: GPU quantize + strided sums; CPU mirrors the same math
    let x = deterministic_input(batch * embd, 11);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq = exec.alloc_i8(batch * embd).expect("xq");
    let mut d_xs = exec.alloc(batch * embd / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * embd)
        .expect("quantize");
    let mut d_sums = exec
        .alloc(batch * embd.max(n_active * ff) / 16)
        .expect("sums");
    exec.q8_sums_strided(&d_xq, &mut d_sums, embd, batch)
        .expect("sums");

    let mut d_fused = exec.alloc(batch * n_active * ff).expect("fused");
    exec.kquant_moe_gate_up(
        &gate,
        &up,
        &d_idx,
        &d_xq,
        &d_xs,
        Some(&d_sums),
        &mut d_fused,
        n_active,
        batch,
    )
    .expect("gate_up");
    let fused_gpu = exec.to_host(&d_fused).expect("fused host");

    let (_, gate_bytes) = map.tensor_bytes(&names[0]).expect("gate bytes");
    let (_, up_bytes) = map.tensor_bytes(&names[1]).expect("up bytes");
    let (_, down_bytes) = map.tensor_bytes(&names[2]).expect("down bytes");
    let mut xq = vec![0i8; batch * embd];
    let mut xs = vec![0f32; batch * embd / 32];
    for c in 0..batch {
        let (q, s) = cpu_quantize_q8(&x[c * embd..(c + 1) * embd]);
        xq[c * embd..(c + 1) * embd].copy_from_slice(&q);
        xs[c * (embd / 32)..(c + 1) * (embd / 32)].copy_from_slice(&s);
    }
    let gsrc = gate_bytes.len() / (embd / 256 * ff * ne);
    let usrc = up_bytes.len() / (embd / 256 * ff * ne);
    let dsrc = down_bytes.len() / (ff / 256 * embd * ne);
    let mut fused_ref = vec![0f32; batch * n_active * ff];
    for b in 0..batch {
        let xqr = &xq[b * embd..(b + 1) * embd];
        let xsr = &xs[b * (embd / 32)..(b + 1) * (embd / 32)];
        for slot in 0..n_active {
            let e = idx_h[b * n_active + slot] as usize;
            for o in 0..ff {
                let g = kq_row_dot(gate_bytes, tys[0], gsrc, e * ff + o, xqr, xsr) as f32;
                let u = kq_row_dot(up_bytes, tys[1], usrc, e * ff + o, xqr, xsr) as f32;
                fused_ref[(b * n_active + slot) * ff + o] = (g / (1.0 + (-g).exp())) * u;
            }
        }
    }
    let e1 = rel_err(&fused_gpu, &fused_ref);
    eprintln!("kq moe gate_up vs quantized CPU ref rel_err {e1:.2e}");
    assert!(e1 < 5e-4, "gate_up mismatch ({e1:.2e})");

    // down stage: both sides quantize the GPU fused values
    let mut d_fq = exec.alloc_i8(batch * n_active * ff).expect("fq");
    let mut d_fs = exec.alloc(batch * n_active * ff / 32).expect("fs");
    exec.quantize_q8(&d_fused, &mut d_fq, &mut d_fs, batch * n_active * ff)
        .expect("fq quant");
    exec.q8_sums_strided(&d_fq, &mut d_sums, ff, batch * n_active)
        .expect("fsums");
    let mut d_out = exec.alloc(batch * embd).expect("out");
    exec.kquant_moe_down(
        &down,
        &d_idx,
        &d_topk,
        &d_fq,
        &d_fs,
        Some(&d_sums),
        &mut d_out,
        n_active,
        batch,
    )
    .expect("down");
    let out_gpu = exec.to_host(&d_out).expect("out host");

    let (fq, fs) = cpu_quantize_q8(&fused_gpu);
    let mut out_ref = vec![0f32; batch * embd];
    for b in 0..batch {
        for o in 0..embd {
            let mut v = 0f64;
            for slot in 0..n_active {
                let srow = b * n_active + slot;
                let e = idx_h[srow] as usize;
                v += topk_h[srow] as f64
                    * kq_row_dot(
                        down_bytes,
                        tys[2],
                        dsrc,
                        e * embd + o,
                        &fq[srow * ff..(srow + 1) * ff],
                        &fs[srow * (ff / 32)..(srow + 1) * (ff / 32)],
                    );
            }
            out_ref[b * embd + o] = v as f32;
        }
    }
    let e2 = rel_err(&out_gpu, &out_ref);
    eprintln!("kq moe down vs quantized CPU ref rel_err {e2:.2e}");
    assert!(e2 < 5e-4, "down mismatch ({e2:.2e})");
}

/// Sorted k-quant MoE mma pair (the prefill/serving class) on the same real
/// 35B-A3B UD expert tensors, against the exact-int CPU model. Routing is
/// concentrated on few experts so live sorted blocks carry MULTIPLE rows plus
/// PAD tails (the layout the kernels must get right). gate_up's output is
/// int8-quantized in-kernel, so its gate covers quantization noise (~0.4% of
/// block amax) and PAD rows are asserted exactly zero; the down stage
/// consumes the GPU's own fq/fs, so its gate is the exact-int 5e-4.
#[test]
fn kq_moe_sorted_pair_matches_quantized_reference() {
    let Some(model) = common::model("QWEN36_MOE_UD_GGUF", common::QWEN36_35B_A3B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_moe_mma() {
        eprintln!("pack lacks the sorted k-quant MoE mma pair - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    // first block whose expert tensors are all k-quant AND whose gate/up
    // types match (the single-dtype pair kernel's contract)
    let mut picked = None;
    for i in 0..8 {
        let names = [
            format!("blk.{i}.ffn_gate_exps.weight"),
            format!("blk.{i}.ffn_up_exps.weight"),
            format!("blk.{i}.ffn_down_exps.weight"),
        ];
        let tys: Vec<_> = names
            .iter()
            .filter_map(|n| map.tensor_bytes(n).ok().map(|(info, _)| info.ggml_type))
            .collect();
        if tys.len() == 3
            && tys.iter().all(|t| {
                matches!(
                    t,
                    GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Iq4Xs
                )
            })
            && tys[0] == tys[1]
        {
            picked = Some((names, tys));
            break;
        }
    }
    let Some((names, tys)) = picked else {
        eprintln!("no matched-pair k-quant expert block in the first 8 layers - skipping");
        return;
    };
    let gate = exec.repack_kquant(&map, &names[0]).expect("repack gate");
    let up = exec.repack_kquant(&map, &names[1]).expect("repack up");
    let down = exec.repack_kquant(&map, &names[2]).expect("repack down");
    let (embd, ff, ne) = (gate.dims[0], gate.dims[1], gate.dims[2]);
    eprintln!("expert block {} [{embd}x{ff}x{ne}] types {tys:?}", names[0]);

    let batch = 8usize;
    let n_active = 8usize;
    // concentrate routing on 13 experts: ~5 rows per live block + PAD tails
    let idx_h: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 7) % 13)
        .collect();
    let topk_h: Vec<f32> = (0..batch * n_active)
        .map(|i| 1.0 / (1.0 + (i % n_active) as f32))
        .collect();
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_topk = exec.to_device(&topk_h).expect("topk");

    let x = deterministic_input(batch * embd, 23);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq = exec.alloc_i8(batch * embd).expect("xq");
    let mut d_xs = exec.alloc(batch * embd / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * embd)
        .expect("quantize");

    let max_blocks = (batch * n_active + ne * 31).div_ceil(32);
    let mut d_srow = exec.alloc_u32(max_blocks * 32).expect("srow");
    let mut d_sslot = exec.alloc_u32(max_blocks * 32).expect("sslot");
    let mut d_bexp = exec.alloc_u32(max_blocks).expect("bexp");
    exec.moe_align(
        &d_idx,
        &mut d_srow,
        &mut d_sslot,
        &mut d_bexp,
        batch,
        n_active,
        ne,
        max_blocks,
    )
    .expect("align");
    let srow_h = exec.to_host_u32(&d_srow).expect("srow host");
    let sslot_h = exec.to_host_u32(&d_sslot).expect("sslot host");
    let bexp_h = exec.to_host_u32(&d_bexp).expect("bexp host");

    let mut d_sums = exec
        .alloc((batch * embd / 16).max(max_blocks * 32 * ff / 16))
        .expect("sums");
    let needs_gu = matches!(tys[0], GgmlType::Q4K | GgmlType::Q5K);
    if needs_gu {
        exec.q8_sums_strided(&d_xq, &mut d_sums, embd, batch)
            .expect("xsums");
    }
    let mut d_fq = exec.alloc_i8(max_blocks * 32 * ff).expect("fq");
    let mut d_fs = exec.alloc(max_blocks * 32 * ff / 32).expect("fs");
    exec.kquant_moe_gate_up_mma(
        &gate,
        &up,
        &d_srow,
        &d_bexp,
        &d_xq,
        &d_xs,
        needs_gu.then_some(&d_sums),
        &mut d_fq,
        &mut d_fs,
        max_blocks,
    )
    .expect("gate_up mma");
    let fq_gpu = exec.to_host_i8(&d_fq).expect("fq host");
    let fs_gpu = exec.to_host(&d_fs).expect("fs host");

    // CPU mirror of the quantized activations (same math as quantize_q8)
    let (_, gate_bytes) = map.tensor_bytes(&names[0]).expect("gate bytes");
    let (_, up_bytes) = map.tensor_bytes(&names[1]).expect("up bytes");
    let (_, down_bytes) = map.tensor_bytes(&names[2]).expect("down bytes");
    let mut xq = vec![0i8; batch * embd];
    let mut xs = vec![0f32; batch * embd / 32];
    for c in 0..batch {
        let (q, s) = cpu_quantize_q8(&x[c * embd..(c + 1) * embd]);
        xq[c * embd..(c + 1) * embd].copy_from_slice(&q);
        xs[c * (embd / 32)..(c + 1) * (embd / 32)].copy_from_slice(&s);
    }
    let gsrc = gate_bytes.len() / (embd / 256 * ff * ne);
    let usrc = up_bytes.len() / (embd / 256 * ff * ne);
    let dsrc = down_bytes.len() / (ff / 256 * embd * ne);

    // gate_up: dequantized GPU fq/fs vs the exact CPU silu ref over live
    // sorted rows; PAD rows must be exact zeros (the flat fsums pass and the
    // down stage's direct row reads rely on that)
    let mut live_gpu = Vec::new();
    let mut live_ref = Vec::new();
    let mut live_rows = 0usize;
    for (blk, &e) in bexp_h.iter().enumerate() {
        if e == u32::MAX {
            continue;
        }
        for col in 0..32 {
            let sp = blk * 32 + col;
            let r = srow_h[sp];
            if r == u32::MAX {
                for o in 0..ff {
                    assert_eq!(fq_gpu[sp * ff + o], 0, "PAD row {sp} not zeroed at {o}");
                }
                for sb in 0..ff / 32 {
                    assert_eq!(fs_gpu[sp * (ff / 32) + sb], 0.0, "PAD scale {sp}/{sb}");
                }
                continue;
            }
            live_rows += 1;
            let r = r as usize;
            let xqr = &xq[r * embd..(r + 1) * embd];
            let xsr = &xs[r * (embd / 32)..(r + 1) * (embd / 32)];
            for o in 0..ff {
                let g = kq_row_dot(gate_bytes, tys[0], gsrc, e as usize * ff + o, xqr, xsr) as f32;
                let u = kq_row_dot(up_bytes, tys[1], usrc, e as usize * ff + o, xqr, xsr) as f32;
                live_ref.push((g / (1.0 + (-g).exp())) * u);
                live_gpu.push(fq_gpu[sp * ff + o] as f32 * fs_gpu[sp * (ff / 32) + o / 32]);
            }
        }
    }
    assert_eq!(
        live_rows,
        batch * n_active,
        "every routed pair appears once"
    );
    let e1 = rel_err(&live_gpu, &live_ref);
    eprintln!("kq moe sorted gate_up (dequantized) rel_err {e1:.2e} over {live_rows} rows");
    assert!(e1 < 1e-2, "sorted gate_up mismatch ({e1:.2e})");

    // down: exact-int ref off the GPU's own fq/fs (isolates the mma + scatter)
    let needs_dn = matches!(tys[2], GgmlType::Q4K | GgmlType::Q5K);
    if needs_dn {
        exec.q8_sums_strided(&d_fq, &mut d_sums, ff, max_blocks * 32)
            .expect("fsums");
    }
    let mut d_part = exec.alloc(batch * n_active * embd).expect("part");
    exec.kquant_moe_down_mma(
        &down,
        &d_srow,
        &d_sslot,
        &d_bexp,
        &d_topk,
        &d_fq,
        &d_fs,
        needs_dn.then_some(&d_sums),
        &mut d_part,
        n_active,
        max_blocks,
    )
    .expect("down mma");
    let part_gpu = exec.to_host(&d_part).expect("part host");

    let fs_h = &fs_gpu;
    let mut part_ref = vec![0f32; batch * n_active * embd];
    for (blk, &e) in bexp_h.iter().enumerate() {
        if e == u32::MAX {
            continue;
        }
        for col in 0..32 {
            let sp = blk * 32 + col;
            let tok = srow_h[sp];
            if tok == u32::MAX {
                continue;
            }
            let pair = tok as usize * n_active + sslot_h[sp] as usize;
            let fqr: Vec<i8> = fq_gpu[sp * ff..(sp + 1) * ff].to_vec();
            let fsr = &fs_h[sp * (ff / 32)..(sp + 1) * (ff / 32)];
            for o in 0..embd {
                part_ref[pair * embd + o] = (topk_h[pair] as f64
                    * kq_row_dot(down_bytes, tys[2], dsrc, e as usize * embd + o, &fqr, fsr))
                    as f32;
            }
        }
    }
    let e2 = rel_err(&part_gpu, &part_ref);
    eprintln!("kq moe sorted down partials rel_err {e2:.2e}");
    assert!(e2 < 5e-4, "sorted down mismatch ({e2:.2e})");
}

///  fused-GLU GEMV: gate+up+SwiGLU one-launch must be BIT-identical
/// to the split path (multi<4,128> gate|up, then swiglu) - the GLU kernel's
/// per-row dots are the identical row walks at the same TPR and its epilogue
/// is the character-identical silu expression, so this is a memcmp gate, not
/// a tolerance gate. Real granite-30b gate/up tensors (Q4_K, in 4096).
#[test]
fn kq_w4a8_glu_matches_split() {
    let Some(model) = common::model("GRANITE_30B_GGUF", common::GRANITE_30B_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_gemv_w4a8_glu() || !exec.has_kquant_gemv_w4a8_multi() {
        eprintln!("pack lacks glu/multi GEMV - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let wg = exec
        .repack_kquant(&map, "blk.0.ffn_gate.weight")
        .expect("gate");
    let wu = exec.repack_kquant(&map, "blk.0.ffn_up.weight").expect("up");
    assert_eq!(wg.ty, GgmlType::Q4K);
    assert_eq!(wu.ty, GgmlType::Q4K);
    let in_dim = wg.dims[0];
    let n_ff = wg.dims[1];

    let x = deterministic_input(in_dim, 91);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
    let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
    let mut d_sums = exec.alloc(in_dim / 16).expect("sums");
    exec.quantize_q8_sums(&d_x, &mut d_xq, &mut d_xs, &mut d_sums, in_dim)
        .expect("quantize");

    // split reference: multi(gate,up) then swiglu-in-place on gate
    let mut d_g = exec.alloc(n_ff).expect("g");
    let mut d_u = exec.alloc(n_ff).expect("u");
    exec.kquant_gemv_w4a8_multi(
        &mut [(&wg, &mut d_g), (&wu, &mut d_u)],
        &d_xq,
        &d_xs,
        &d_sums,
    )
    .expect("gate|up multi");
    exec.swiglu(&mut d_g, &d_u, n_ff).expect("swiglu");

    // fused
    let mut d_h = exec.alloc(n_ff).expect("h");
    exec.kquant_gemv_w4a8_glu(&wg, &wu, &d_xq, &d_xs, &d_sums, &mut d_h)
        .expect("glu fused");

    let (href, hglu) = (exec.to_host(&d_g).unwrap(), exec.to_host(&d_h).unwrap());
    assert_eq!(
        href, hglu,
        "fused GLU must be BIT-identical to multi+swiglu"
    );
}

// ---- Q4_0 (QAT lineage): synthesized-tensor coverage ---------------------
// No real Q4_0 file is pinned for the gates (the QAT gemma UD files are the
// field source), so this ladder runs on deterministic synthesized blocks -
// the same coverage shape the IQ4_XS retirement note asks for.

/// Deterministic raw Q4_0 blocks: nibbles sweep all 16 values, block scales
/// vary in magnitude and sign-of-exponent across the tensor.
fn synth_q40(n_blocks: usize, seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_blocks * 18);
    let mut state = seed;
    for _ in 0..n_blocks {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // f16 scales spread across ~1e-3..2.0, both signs
        let mag = ((state >> 40) & 0x3FF) as f32 / 512.0 + 1e-3;
        let d = if (state >> 51) & 1 == 1 { -mag } else { mag };
        out.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for k in 0..16u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(k);
            out.push((state >> 33) as u8);
        }
    }
    out
}

/// The whole stage-1 + stage-2 ladder on a synthesized Q4_0 tensor:
/// dequant_rp exact, gather exact, exact GEMV vs f64 CPU dot, W4A8 vs the
/// CPU quantized-activation model - the same bisect the real-file cases run.
#[test]
fn q40_synth_full_ladder() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant() || !exec.has_kquant_q40() {
        return; // pack predates the Q4_0 capability slot
    }
    let (in_dim, out_dim) = (768usize, 512usize);
    let n = in_dim * out_dim;
    let bytes = synth_q40(n / 32, 41);
    let cpu = cpu_dequant(&bytes, GgmlType::Q4_0, n);

    let kq = exec
        .repack_kquant_raw(&bytes, vec![in_dim, out_dim], GgmlType::Q4_0, "synth_q40")
        .expect("repack");

    // dequant from the repacked streams: per-term products are the exact
    // dj*v - 8*dj form, bit-identical to d*(q-8) (both terms exact in f32)
    let mut d_f = exec.alloc(n).expect("f32 dst");
    exec.kquant_dequant_rp(&kq, &mut d_f).expect("dequant_rp");
    let rp = exec.to_host(&d_f).expect("host");
    assert_eq!(rp.len(), cpu.len());
    for (i, (a, b)) in rp.iter().zip(&cpu).enumerate() {
        assert!(a == b, "dequant_rp[{i}]: {a} != {b}");
    }

    // embedding gather: rows exact
    let toks: Vec<u32> = vec![0, 3, out_dim as u32 - 1, 17, 255];
    let d_toks = exec.to_device_u32(&toks).expect("tokens");
    let mut d_rows = exec.alloc(toks.len() * in_dim).expect("rows");
    exec.kquant_gather(&kq, &d_toks, &mut d_rows, in_dim, toks.len())
        .expect("gather");
    let rows = exec.to_host(&d_rows).expect("rows host");
    for (i, t) in toks.iter().enumerate() {
        let r = *t as usize;
        for l in 0..in_dim {
            let want = cpu[r * in_dim + l];
            let got = rows[i * in_dim + l];
            assert!(got == want, "gather row {r} [{l}]: {got} != {want}");
        }
    }

    // exact-f32 GEMV vs f64 CPU dot (commutative grouping differs)
    let x = deterministic_input(in_dim, 11);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_y = exec.alloc(out_dim).expect("y");
    exec.kquant_gemv(&kq, &d_x, &mut d_y).expect("gemv");
    let y = exec.to_host(&d_y).expect("y host");
    let mut y_ref = vec![0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = 0f64;
        for l in 0..in_dim {
            acc += (cpu[o * in_dim + l] as f64) * (x[l] as f64);
        }
        y_ref[o] = acc as f32;
    }
    let e = rel_err(&y, &y_ref);
    assert!(e < 2e-6, "exact gemv rel err {e}");

    // stage-2 W4A8 vs the CPU quantized-activation model (mu format: the
    // centered -8 offset rides the activation sums)
    let batch = 144usize;
    let xb = deterministic_input(in_dim * batch, 7);
    let d_xb = exec.to_device(&xb).expect("xb");
    let n_chunks = in_dim.div_ceil(128);
    let batch_pad = batch.div_ceil(128) * 128;
    let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
    exec.quantize_q8_mmq(&d_xb, &mut d_yq, in_dim, batch)
        .expect("quantize");
    let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
    exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch)
        .expect("sums");
    let mut d_yb = exec.alloc(out_dim * batch).expect("yb");
    exec.kquant_gemm_w4a8(&kq, &d_yq, Some(&d_sums), &mut d_yb, batch)
        .expect("w4a8");
    let y_gpu = exec.to_host(&d_yb).expect("yb host");

    let n_super = in_dim / 256;
    let src_b = 144usize;
    let mut w4 = vec![0f32; out_dim * batch];
    for o in 0..out_dim {
        let mut wq = vec![0i8; in_dim];
        let mut dj16 = vec![0f32; n_super * 16];
        let mut mu = vec![0f32; n_super * 8];
        for sb in 0..n_super {
            let s = &bytes[(o * n_super + sb) * src_b..];
            let (q, d16, m8) = cpu_w4a8_super(s, GgmlType::Q4_0);
            wq[sb * 256..(sb + 1) * 256].copy_from_slice(&q);
            dj16[sb * 16..(sb + 1) * 16].copy_from_slice(&d16);
            mu[sb * 8..(sb + 1) * 8].copy_from_slice(&m8);
        }
        for c in 0..batch {
            let (xq, xs) = cpu_quantize_q8(&xb[c * in_dim..(c + 1) * in_dim]);
            let mut acc = 0f64;
            for blk in 0..in_dim / 32 {
                let mut dot = 0i64;
                let mut sum = 0i64;
                for l in 0..32 {
                    dot += (wq[blk * 32 + l] as i64) * (xq[blk * 32 + l] as i64);
                    sum += xq[blk * 32 + l] as i64;
                }
                let dj = dj16[blk * 2] as f64;
                let m = mu[blk] as f64;
                let sc = xs[blk] as f64;
                acc += dj * (sc * dot as f64) + m * (sc * sum as f64);
            }
            w4[c * out_dim + o] = acc as f32;
        }
    }
    let e2 = rel_err(&y_gpu, &w4);
    assert!(e2 < 2e-5, "w4a8 rel err {e2}");
}

/// The load fallback's contract: transcoding Q4_0 -> Q8_0 dequants
/// BIT-IDENTICALLY (same f16 scale, int8 = nibble - 8). Host-only.
#[test]
fn q40_transcode_is_exact() {
    let bytes = synth_q40(96, 5);
    let q40 = cpu_dequant(&bytes, GgmlType::Q4_0, 96 * 32);
    let q8 = paddock_engine::gpu::q40_to_q8_blocks(&bytes);
    assert_eq!(q8.len(), 96 * 34);
    for b in 0..96 {
        let blk = &q8[b * 34..(b + 1) * 34];
        let d = f16(&blk[0..]);
        for l in 0..32 {
            let v = blk[2 + l] as i8;
            let got = d * v as f32;
            assert!(got == q40[b * 32 + l], "block {b} weight {l}");
        }
    }
}

/// MoE expert offload, phase A: the same expert block served from a
/// device-mapped host mirror must produce BIT-identical gate_up and down
/// outputs to the VRAM-resident plane. Same repack, same kernels, same
/// addressing - only the memory the bytes sit in differs.
#[test]
fn kq_moe_pair_host_mapped_bitmatches_resident() {
    let Some(model) = common::model("QWEN36_MOE_UD_GGUF", common::QWEN36_35B_A3B_UD_Q4) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_moe() {
        eprintln!("pack lacks the k-quant MoE pair - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let names = [
        "blk.0.ffn_gate_exps.weight".to_string(),
        "blk.0.ffn_up_exps.weight".to_string(),
        "blk.0.ffn_down_exps.weight".to_string(),
    ];
    let res: Vec<_> = names
        .iter()
        .map(|n| exec.repack_kquant(&map, n).expect("resident repack"))
        .collect();
    let host: Vec<_> = names
        .iter()
        .map(|n| {
            exec.try_repack_kquant_host_mapped(&map, n)
                .expect("host-mapped repack")
                .expect("k-quant expert tensor")
        })
        .collect();
    let (embd, ff, ne) = (res[0].dims[0], res[0].dims[1], res[0].dims[2]);
    eprintln!(
        "host-mapped expert block [{embd}x{ff}x{ne}] {:.1} MB in host memory",
        host.iter().map(|h| h.host_bytes()).sum::<u64>() as f64 / 1e6
    );

    let batch = 3usize;
    let n_active = 8usize;
    let idx_h: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 7) % ne as u32)
        .collect();
    let topk_h: Vec<f32> = (0..batch * n_active)
        .map(|i| 1.0 / (1.0 + (i % n_active) as f32))
        .collect();
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_topk = exec.to_device(&topk_h).expect("topk");
    let x = deterministic_input(batch * embd, 11);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq = exec.alloc_i8(batch * embd).expect("xq");
    let mut d_xs = exec.alloc(batch * embd / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * embd)
        .expect("quantize");
    let mut d_sums = exec
        .alloc(batch * embd.max(n_active * ff) / 16)
        .expect("sums");
    exec.q8_sums_strided(&d_xq, &mut d_sums, embd, batch)
        .expect("sums");

    let run = |gate: &paddock_engine::gpu::RepackedKQ,
               up: &paddock_engine::gpu::RepackedKQ,
               down: &paddock_engine::gpu::RepackedKQ|
     -> (Vec<f32>, Vec<f32>) {
        let mut d_fused = exec.alloc(batch * n_active * ff).expect("fused");
        exec.kquant_moe_gate_up(
            gate,
            up,
            &d_idx,
            &d_xq,
            &d_xs,
            Some(&d_sums),
            &mut d_fused,
            n_active,
            batch,
        )
        .expect("gate_up");
        let fused = exec.to_host(&d_fused).expect("fused host");
        let mut d_fq = exec.alloc_i8(batch * n_active * ff).expect("fq");
        let mut d_fs = exec.alloc(batch * n_active * ff / 32).expect("fs");
        exec.quantize_q8(&d_fused, &mut d_fq, &mut d_fs, batch * n_active * ff)
            .expect("fq quant");
        let mut d_fsums = exec.alloc(batch * n_active * ff / 16).expect("fsums");
        exec.q8_sums_strided(&d_fq, &mut d_fsums, ff, batch * n_active)
            .expect("fsums");
        let mut d_out = exec.alloc(batch * embd).expect("out");
        exec.kquant_moe_down(
            down,
            &d_idx,
            &d_topk,
            &d_fq,
            &d_fs,
            Some(&d_fsums),
            &mut d_out,
            n_active,
            batch,
        )
        .expect("down");
        (fused, exec.to_host(&d_out).expect("out host"))
    };
    let (fused_r, out_r) = run(&res[0], &res[1], &res[2]);
    let (fused_h, out_h) = run(&host[0], &host[1], &host[2]);
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    eprintln!(
        "host-mapped vs resident: gate_up bitwise={} down bitwise={}",
        same(&fused_r, &fused_h),
        same(&out_r, &out_h)
    );
    assert!(
        same(&fused_r, &fused_h),
        "gate_up differs between host-mapped and resident planes"
    );
    assert!(
        same(&out_r, &out_h),
        "down differs between host-mapped and resident planes"
    );
}

/// The i-quant family (quant/iquant.cuh): (1) the pack's raw dequant is
/// BIT-identical to the ggml reference port for every i-quant type the file
/// carries; (2) the token-batched MoE pair over repacked i-quant expert
/// seats matches a float reference built from that dequant (exact int dots
/// times f32 scales on the GPU, f64 on the CPU - tolerance covers order).
#[test]
fn iq_family_matches_reference() {
    use paddock_kernels::reference::iq::{dequant_iq, iq_block_bytes};
    let Some(model) = common::model("QWEN36_MOE_IQ2_GGUF", common::QWEN36_35B_A3B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_iq() {
        eprintln!("pack lacks the i-quant family - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");

    // (1) one expert tensor per i-quant type present
    let mut seen = std::collections::BTreeMap::new();
    for t in map.tensor_infos() {
        if iq_block_bytes(t.raw_type).is_some() && t.name.contains("_exps") {
            seen.entry(t.raw_type).or_insert_with(|| t.name.clone());
        }
    }
    assert!(
        !seen.is_empty(),
        "no i-quant expert tensors in {}",
        model.display()
    );
    for (raw, name) in &seen {
        let (info, bytes) = map.tensor_bytes(name).expect("tensor bytes");
        let n = info.element_count() as usize;
        let block = iq_block_bytes(*raw).expect("i-quant");
        let take = (4096usize * 256).min(n);
        let mut cpu = vec![0f32; take];
        dequant_iq(*raw, &bytes[..take / 256 * block], &mut cpu).expect("cpu dequant");
        let t = exec.upload(&map, name).expect("gpu dequant");
        let gpu = exec.to_host_len(&t.buf, take).expect("dtoh");
        let mism = gpu
            .iter()
            .zip(&cpu)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        eprintln!(
            "{name} {:?}: {take} weights, {mism} bit mismatches",
            info.ggml_type
        );
        assert_eq!(
            mism, 0,
            "{name}: GPU dequant differs from the ggml reference port"
        );
    }

    // (2) MoE pair on every distinct (gate, up, down) type triple in the file
    let mut triples: std::collections::BTreeMap<Vec<u32>, [String; 3]> = Default::default();
    for i in 0..64 {
        let names = [
            format!("blk.{i}.ffn_gate_exps.weight"),
            format!("blk.{i}.ffn_up_exps.weight"),
            format!("blk.{i}.ffn_down_exps.weight"),
        ];
        let raws: Vec<u32> = names
            .iter()
            .filter_map(|n| map.tensor_info(n).map(|t| t.raw_type))
            .collect();
        if raws.len() == 3 && raws.iter().all(|&r| iq_block_bytes(r).is_some()) {
            triples.entry(raws).or_insert(names);
        }
    }
    assert!(
        !triples.is_empty(),
        "no all-i-quant expert block - nothing to pair-test"
    );
    for (raws, names) in &triples {
        iq_pair_check(&exec, &map, names, raws);
    }
}

fn iq_pair_check(exec: &GpuExecutor, map: &MappedGguf, names: &[String; 3], raws: &[u32]) {
    use paddock_kernels::reference::iq::{dequant_iq, iq_block_bytes};
    let gate = exec.repack_kquant(map, &names[0]).expect("repack gate");
    let up = exec.repack_kquant(map, &names[1]).expect("repack up");
    let down = exec.repack_kquant(map, &names[2]).expect("repack down");
    let (embd, ff, ne) = (gate.dims[0], gate.dims[1], gate.dims[2]);
    eprintln!(
        "i-quant expert block {} [{embd}x{ff}x{ne}] raw types {raws:?}",
        names[0]
    );

    let batch = 3usize;
    let n_active = 8usize;
    let idx_h: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 7) % ne as u32)
        .collect();
    let topk_h: Vec<f32> = (0..batch * n_active)
        .map(|i| 1.0 / (1.0 + (i % n_active) as f32))
        .collect();
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_topk = exec.to_device(&topk_h).expect("topk");
    let x = deterministic_input(batch * embd, 11);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq = exec.alloc_i8(batch * embd).expect("xq");
    let mut d_xs = exec.alloc(batch * embd / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * embd)
        .expect("quantize");
    // formats with a per-16 mu term need the activation sums - the pack
    // refuses a launch without them (Q2_K seats did exactly that on the MTP
    // repo's UD-IQ2_XXS file, whose blk.40 experts are Q2_K/Q3_K)
    let raw_needs_sums = |raw: u32| matches!(raw, 2 | 10 | 12 | 13); // Q4_0, Q2_K, Q4_K, Q5_K
    let mut d_sums = exec.alloc(batch * embd / 16).expect("sums");
    let gu_needs = raw_needs_sums(raws[0]) || raw_needs_sums(raws[1]);
    if gu_needs {
        exec.q8_sums_strided(&d_xq, &mut d_sums, embd, batch)
            .expect("sums");
    }
    let mut d_fused = exec.alloc(batch * n_active * ff).expect("fused");
    exec.kquant_moe_gate_up(
        &gate,
        &up,
        &d_idx,
        &d_xq,
        &d_xs,
        gu_needs.then_some(&d_sums),
        &mut d_fused,
        n_active,
        batch,
    )
    .expect("gate_up");
    let fused_gpu = exec.to_host(&d_fused).expect("fused host");

    // reference: dequantized expert rows (one expert = `rows` rows of in_dim)
    let expert_rows = |name: &str, raw: u32, e: usize, rows: usize, in_dim: usize| -> Vec<f32> {
        let (_, bytes) = map.tensor_bytes(name).expect("bytes");
        let block = iq_block_bytes(raw).expect("i-quant");
        let per_row = in_dim / 256 * block;
        let mut out = vec![0f32; rows * in_dim];
        dequant_iq(
            raw,
            &bytes[e * rows * per_row..(e + 1) * rows * per_row],
            &mut out,
        )
        .expect("expert rows");
        out
    };
    let mut xq = vec![0i8; batch * embd];
    let mut xs = vec![0f32; batch * embd / 32];
    for c in 0..batch {
        let (q, s) = cpu_quantize_q8(&x[c * embd..(c + 1) * embd]);
        xq[c * embd..(c + 1) * embd].copy_from_slice(&q);
        xs[c * (embd / 32)..(c + 1) * (embd / 32)].copy_from_slice(&s);
    }
    let dot = |w: &[f32], xq: &[i8], xs: &[f32]| -> f64 {
        w.iter()
            .zip(xq)
            .enumerate()
            .map(|(k, (&wk, &q))| wk as f64 * q as f64 * xs[k / 32] as f64)
            .sum()
    };
    let mut fused_ref = vec![0f32; batch * n_active * ff];
    let mut cache: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)> = Default::default();
    for b in 0..batch {
        let xqr = &xq[b * embd..(b + 1) * embd];
        let xsr = &xs[b * (embd / 32)..(b + 1) * (embd / 32)];
        for slot in 0..n_active {
            let e = idx_h[b * n_active + slot] as usize;
            let (g, u) = cache.entry(e).or_insert_with(|| {
                (
                    expert_rows(&names[0], raws[0], e, ff, embd),
                    expert_rows(&names[1], raws[1], e, ff, embd),
                )
            });
            for o in 0..ff {
                let gv = dot(&g[o * embd..(o + 1) * embd], xqr, xsr);
                let uv = dot(&u[o * embd..(o + 1) * embd], xqr, xsr);
                fused_ref[(b * n_active + slot) * ff + o] =
                    ((gv / (1.0 + (-gv).exp())) * uv) as f32;
            }
        }
    }
    let e1 = rel_err(&fused_gpu, &fused_ref);
    eprintln!("i-quant moe gate_up vs dequant reference rel_err {e1:.2e}");
    assert!(e1 < 5e-4, "gate_up mismatch ({e1:.2e})");

    let mut d_fq = exec.alloc_i8(batch * n_active * ff).expect("fq");
    let mut d_fs = exec.alloc(batch * n_active * ff / 32).expect("fs");
    exec.quantize_q8(&d_fused, &mut d_fq, &mut d_fs, batch * n_active * ff)
        .expect("fq quant");
    let mut d_out = exec.alloc(batch * embd).expect("out");
    let d_needs = raw_needs_sums(raws[2]);
    let mut d_fsums = exec.alloc(batch * n_active * ff / 16).expect("fsums");
    if d_needs {
        exec.q8_sums_strided(&d_fq, &mut d_fsums, ff, batch * n_active)
            .expect("fsums");
    }
    exec.kquant_moe_down(
        &down,
        &d_idx,
        &d_topk,
        &d_fq,
        &d_fs,
        d_needs.then_some(&d_fsums),
        &mut d_out,
        n_active,
        batch,
    )
    .expect("down");
    let out_gpu = exec.to_host(&d_out).expect("out host");
    let (fq, fs) = cpu_quantize_q8(&fused_gpu);
    let mut dcache: std::collections::HashMap<usize, Vec<f32>> = Default::default();
    let mut out_ref = vec![0f32; batch * embd];
    for b in 0..batch {
        for slot in 0..n_active {
            let srow = b * n_active + slot;
            let e = idx_h[srow] as usize;
            let dw = dcache
                .entry(e)
                .or_insert_with(|| expert_rows(&names[2], raws[2], e, embd, ff));
            let fqr = &fq[srow * ff..(srow + 1) * ff];
            let fsr = &fs[srow * (ff / 32)..(srow + 1) * (ff / 32)];
            for o in 0..embd {
                out_ref[b * embd + o] +=
                    (topk_h[srow] as f64 * dot(&dw[o * ff..(o + 1) * ff], fqr, fsr)) as f32;
            }
        }
    }
    let e2 = rel_err(&out_gpu, &out_ref);
    eprintln!("i-quant moe down vs dequant reference rel_err {e2:.2e}");
    assert!(e2 < 5e-4, "down mismatch ({e2:.2e})");
}

/// The shared expert's k-quant seats (Q5_K [2048 -> 512], Q6_K [512 -> 2048]
/// in the UD-IQ2 files) through every lane the shexp dispatch can take:
/// b=1 fused GEMV, the nc GEMV, the mma K-split, the dp4a GEMM.
#[test]
fn shexp_kquant_lanes_match_reference() {
    let Some(model) = common::model("QWEN36_MOE_IQ2_GGUF", common::QWEN36_35B_A3B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model).expect("open gguf");
    for name in [
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
        "blk.0.attn_gate.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_out.weight",
        "output.weight",
    ] {
        let (info, _) = map.tensor_bytes(name).expect("bytes");
        let (in_dim, out_dim) = (info.dims[0] as usize, info.dims[1] as usize);
        let wf = exec
            .to_host(&exec.upload(&map, name).expect("dequant").buf)
            .expect("host");
        let kq = exec.repack_kquant(&map, name).expect("repack");
        eprintln!("{name} {:?} [{in_dim} -> {out_dim}]", info.ggml_type);
        let dotf = |x: &[f32], o: usize| -> f32 {
            (0..in_dim)
                .map(|i| wf[o * in_dim + i] as f64 * x[i] as f64)
                .sum::<f64>() as f32
        };
        // b=1 fused GEMV (gemv_any's k-quant arm)
        let x = deterministic_input(in_dim, 3);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(out_dim).expect("y");
        exec.kquant_gemv(&kq, &d_x, &mut d_y).expect("gemv");
        let gy = exec.to_host(&d_y).expect("y");
        let cy: Vec<f32> = (0..out_dim).map(|o| dotf(&x, o)).collect();
        let e = rel_err(&gy, &cy);
        eprintln!("  kquant_gemv rel_err {e:.2e}");
        assert!(e < 1e-4, "{name}: kquant_gemv ({e:.2e})");
        // the int8 lanes: quantize on GPU, reference with the CPU mirror
        let rs: &[usize] = if out_dim > 65536 {
            &[2, 5]
        } else {
            &[2, 5, 30]
        };
        for &r in rs {
            let xb = deterministic_input(r * in_dim, 9 + r as u64);
            let d_xb = exec.to_device(&xb).expect("xb");
            let mut d_xq = exec.alloc_i8(r * in_dim).expect("xq");
            let mut d_xs = exec.alloc(r * in_dim / 32).expect("xs");
            exec.quantize_q8(&d_xb, &mut d_xq, &mut d_xs, r * in_dim)
                .expect("quant");
            exec.synchronize().expect("sync after quantize");
            let mut d_sums = exec.alloc(r * in_dim / 16).expect("sums");
            exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, r)
                .expect("sums");
            exec.synchronize().expect("sync after sums");
            let needs = matches!(kq.ty, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_0);
            let mut cref = vec![0f32; r * out_dim];
            for row in 0..r {
                let (q, s) = cpu_quantize_q8(&xb[row * in_dim..(row + 1) * in_dim]);
                let xdq: Vec<f32> = q
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v as f32 * s[i / 32])
                    .collect();
                for o in 0..out_dim {
                    cref[row * out_dim + o] = dotf(&xdq, o);
                }
            }
            let mut lanes: Vec<(&str, Vec<f32>)> = Vec::new();
            let mut d_yb = exec.alloc(r * out_dim).expect("yb");
            exec.kquant_gemm_dp4a(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_yb, r)
                .expect("dp4a");
            exec.synchronize().expect("sync after dp4a");
            lanes.push(("gemm_dp4a", exec.to_host(&d_yb).expect("h")));
            if (3..=64).contains(&r) && exec.has_kquant_mma_ks() {
                let mut part = exec.alloc(8 * 64 * out_dim).expect("part");
                let mut d_yk = exec.alloc(r * out_dim).expect("yk");
                exec.kquant_gemm_mma_ks(
                    &kq,
                    &d_xq,
                    &d_xs,
                    needs.then_some(&d_sums),
                    &mut part,
                    &mut d_yk,
                    r,
                )
                .expect("mma_ks");
                lanes.push(("gemm_mma_ks", exec.to_host(&d_yk).expect("h")));
            }
            if std::env::var_os("PADDOCK_NO_KQ_NC").is_none()
                && exec.has_kquant_gemv_w4a8_nc()
                && GpuExecutor::kquant_gemv_w4a8_nc_fits(&kq, r)
            {
                let mut d_yn = exec.alloc(r * out_dim).expect("yn");
                exec.kquant_gemv_w4a8_nc(&kq, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_yn, r)
                    .expect("nc");
                lanes.push(("gemv_w4a8_nc", exec.to_host(&d_yn).expect("h")));
            }
            for (lane, y) in &lanes {
                let e = rel_err(y, &cref);
                eprintln!("  r={r} {lane} rel_err {e:.2e}");
                assert!(e < 5e-4, "{name}: {lane} r={r} ({e:.2e})");
            }
        }
    }
}

/// Rows that are not a whole number of superblocks (Qwen3.8-Flash-Next's
/// expert `down`: in_dim 640 = 2.5 superblocks, IQ4_NL only): the repack lays
/// the row's 32-blocks flat with no padding and the token-batched MoE down
/// strides rows by in_dim/32 blocks. Synthesized IQ4_NL rows (any 18 bytes
/// are a valid block), f64 CPU reference over the raw blocks; the repacked
/// dequant equals the raw dequant element for element.
#[test]
fn kq_moe_down_partial_superblock_rows_match_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_moe() || !exec.has_kquant_iq() {
        eprintln!("pack lacks the k-quant MoE pair / i-quant seats - skipping");
        return;
    }
    const KV: [i8; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let (ff, embd, ne) = (640usize, 24usize, 5usize);
    let blocks = ff / 32;
    // synth_q40 emits {f16 d, 16 bytes}: the IQ4_NL block layout exactly
    let raw = synth_q40(ne * embd * blocks, 77);
    let down = exec
        .repack_kquant_raw(
            &raw,
            vec![ff, embd, ne],
            GgmlType::Iq4Nl,
            "synthetic down [640, 24, 5]",
        )
        .expect("repack flat rows");
    assert_eq!(
        down.data.len(),
        ne * embd * blocks * 16,
        "flat data stream, no padding"
    );
    assert_eq!(down.scales.len(), ne * embd * blocks * 2, "flat d stream");
    assert_eq!(down.dims, vec![ff, embd, ne]);
    let batch = 3usize;
    let n_active = 4usize;
    let idx_h: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i as u32).wrapping_mul(2654435761) >> 7) % ne as u32)
        .collect();
    let topk_h: Vec<f32> = (0..batch * n_active)
        .map(|i| 1.0 / (1.0 + (i % n_active) as f32))
        .collect();
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_topk = exec.to_device(&topk_h).expect("topk");
    let act = deterministic_input(batch * n_active * ff, 5);
    let d_act = exec.to_device(&act).expect("act");
    let mut d_fq = exec.alloc_i8(batch * n_active * ff).expect("fq");
    let mut d_fs = exec.alloc(batch * n_active * ff / 32).expect("fs");
    exec.quantize_q8(&d_act, &mut d_fq, &mut d_fs, batch * n_active * ff)
        .expect("fq quant");
    let mut d_out = exec.alloc(batch * embd).expect("out");
    exec.kquant_moe_down(
        &down, &d_idx, &d_topk, &d_fq, &d_fs, None, &mut d_out, n_active, batch,
    )
    .expect("down on flat rows");
    let out_gpu = exec.to_host(&d_out).expect("out host");
    let (fq, fs) = cpu_quantize_q8(&act);
    let deq_row = |row: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(ff);
        for j in 0..blocks {
            let blk = &raw[(row * blocks + j) * 18..(row * blocks + j + 1) * 18];
            let d = f16(&blk[0..2]);
            let lo: Vec<f32> = (0..16)
                .map(|i| KV[(blk[2 + i] & 0xf) as usize] as f32 * d)
                .collect();
            let hi: Vec<f32> = (0..16)
                .map(|i| KV[(blk[2 + i] >> 4) as usize] as f32 * d)
                .collect();
            v.extend(lo);
            v.extend(hi);
        }
        v
    };
    let mut out_ref = vec![0f32; batch * embd];
    for b in 0..batch {
        for o in 0..embd {
            let mut v = 0f64;
            for slot in 0..n_active {
                let srow = b * n_active + slot;
                let e = idx_h[srow] as usize;
                let w = deq_row(e * embd + o);
                let mut dot = 0f64;
                for i in 0..ff {
                    dot += w[i] as f64
                        * (fq[srow * ff + i] as f32 * fs[srow * (ff / 32) + i / 32]) as f64;
                }
                v += topk_h[srow] as f64 * dot;
            }
            out_ref[b * embd + o] = v as f32;
        }
    }
    let e2 = rel_err(&out_gpu, &out_ref);
    eprintln!("kq moe down on 640-wide flat IQ4_NL rows vs f64 ref rel_err {e2:.2e}");
    assert!(e2 < 5e-4, "flat-row down mismatch ({e2:.2e})");
    let mut d_dq = exec.alloc(ne * embd * ff).expect("dq");
    exec.kquant_dequant_rp(&down, &mut d_dq)
        .expect("dequant rp");
    let dq = exec.to_host(&d_dq).expect("dq host");
    for r in 0..ne * embd {
        let want = deq_row(r);
        assert!(
            dq[r * ff..(r + 1) * ff] == want[..],
            "row {r}: repacked dequant differs from raw"
        );
    }
}

/// The i-quant family on the DENSE lanes (quant/iquant_dense.cuh): every
/// dense entry point the qwen35 walk takes, on a file whose dense tensors
/// are i-quant (Qwen3.5-9B UD-IQ2_XXS), against an f64 dot over the
/// repacked dequant. The f32 GEMV and the gather are exact classes; the
/// int8-activation lanes are checked against the same int8-quantized
/// activations, so they are exact too - the activation quantization itself
/// is the caller's choice, not the lane's.
#[test]
fn iq_dense_lanes_match_reference() {
    let Some(model) = common::model("QWEN35_IQ2_GGUF", common::QWEN35_9B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_iq() || !exec.has_kquant_iq_dense() {
        eprintln!("pack lacks the dense i-quant lanes (slots 539/540) - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let is_iq = |t: GgmlType| {
        matches!(
            t,
            GgmlType::Iq1S
                | GgmlType::Iq1M
                | GgmlType::Iq2Xxs
                | GgmlType::Iq2Xs
                | GgmlType::Iq2S
                | GgmlType::Iq3Xxs
                | GgmlType::Iq3S
                | GgmlType::Iq4Nl
                | GgmlType::Q2K
                | GgmlType::Q3K
        )
    };
    let batch = 5usize;
    let mut checked = 0usize;
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.4.ffn_up.weight",
        "output.weight",
    ] {
        let Ok((info, _)) = map.tensor_bytes(name) else {
            continue;
        };
        if !is_iq(info.ggml_type) {
            eprintln!("{name}: {:?}, not i-quant - skipped", info.ggml_type);
            continue;
        }
        let w = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        // the reference weights come off the RAW bytes through the Rust port
        // of ggml's dequant - independent of the repack and of every kernel
        let (_, raw) = map.tensor_bytes(name).expect("raw");
        let mut deq = vec![0f32; in_dim * out_dim];
        paddock_kernels::reference::iq::dequant_iq(w.ty.raw(), raw, &mut deq).expect("raw dequant");
        // and the repacked dequant must agree with it (its own i-quant branch)
        let mut d_dq = exec.alloc(in_dim * out_dim).expect("dq");
        exec.kquant_dequant_rp(&w, &mut d_dq).expect("dequant");
        let deq_rp = exec.to_host(&d_dq).expect("h");
        let e_rp = rel_err(&deq_rp, &deq);
        assert!(
            e_rp < 1e-6,
            "{name}: repacked dequant vs raw dequant ({e_rp:.2e})"
        );
        let x = deterministic_input(batch * in_dim, 21);
        let mut xq = vec![0i8; batch * in_dim];
        let mut xs = vec![0f32; batch * in_dim / 32];
        for r in 0..batch {
            let (qq, ss) = cpu_quantize_q8(&x[r * in_dim..(r + 1) * in_dim]);
            xq[r * in_dim..(r + 1) * in_dim].copy_from_slice(&qq);
            xs[r * (in_dim / 32)..(r + 1) * (in_dim / 32)].copy_from_slice(&ss);
        }
        let mut yref = vec![0f32; batch * out_dim];
        let mut yref_q = vec![0f32; batch * out_dim];
        for r in 0..batch {
            for o in 0..out_dim {
                let (mut a, mut aq) = (0f64, 0f64);
                for i in 0..in_dim {
                    let wv = deq[o * in_dim + i] as f64;
                    a += wv * x[r * in_dim + i] as f64;
                    aq += wv * (xq[r * in_dim + i] as f32 * xs[r * (in_dim / 32) + i / 32]) as f64;
                }
                yref[r * out_dim + o] = a as f32;
                yref_q[r * out_dim + o] = aq as f32;
            }
        }
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq = exec.alloc_i8(batch * in_dim).expect("xq");
        let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
            .expect("quant");
        // Q2_K carries a per-16 min term: the int8 lanes take the sums
        let needs = w.ty == GgmlType::Q2K;
        let mut d_sums = exec.alloc(batch * in_dim / 16).expect("sums");
        exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, batch)
            .expect("sums");
        let mut y_gemv = vec![0f32; batch * out_dim];
        let mut xr = exec.alloc(in_dim).expect("xr");
        let mut yr = exec.alloc(out_dim).expect("yr");
        for r in 0..batch {
            exec.copy_region(&d_x, r * in_dim, &mut xr, 0, in_dim)
                .expect("copy");
            exec.kquant_gemv(&w, &xr, &mut yr).expect("gemv f32");
            y_gemv[r * out_dim..(r + 1) * out_dim].copy_from_slice(&exec.to_host(&yr).expect("h"));
        }
        let mut d_y1 = exec.alloc(out_dim).expect("y1");
        let mut d_xq1 = exec.alloc_i8(in_dim).expect("xq1");
        let mut d_xs1 = exec.alloc(in_dim / 32).expect("xs1");
        exec.copy_region(&d_xq, 0, &mut d_xq1, 0, in_dim)
            .expect("copy");
        exec.copy_region(&d_xs, 0, &mut d_xs1, 0, in_dim / 32)
            .expect("copy");
        let mut d_sums1 = exec.alloc(in_dim / 16).expect("sums1");
        exec.copy_region(&d_sums, 0, &mut d_sums1, 0, in_dim / 16)
            .expect("copy");
        exec.kquant_gemv_w4a8(&w, &d_xq1, &d_xs1, needs.then_some(&d_sums1), &mut d_y1)
            .expect("w4a8");
        let y_w4a8 = exec.to_host(&d_y1).expect("h");
        let mut d_y = exec.alloc(batch * out_dim).expect("y");
        exec.kquant_gemm_dp4a(&w, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y, batch)
            .expect("dp4a");
        let y_dp4a = exec.to_host(&d_y).expect("h");
        let mut d_y3 = exec.alloc(3 * out_dim).expect("y3");
        exec.kquant_gemv_w4a8_nc(&w, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y3, 3)
            .expect("nc");
        let y_nc = exec.to_host(&d_y3).expect("h");
        let e_gemv = rel_err(&y_gemv, &yref);
        let e_w4a8 = rel_err(&y_w4a8, &yref_q[..out_dim]);
        let e_dp4a = rel_err(&y_dp4a, &yref_q);
        let e_nc = rel_err(&y_nc, &yref_q[..3 * out_dim]);
        eprintln!(
            "{name} {:?} [{in_dim}->{out_dim}]: gemv_f32 {e_gemv:.2e} | w4a8 {e_w4a8:.2e} dp4a {e_dp4a:.2e} nc {e_nc:.2e}",
            w.ty
        );
        if e_gemv > 2e-5 || e_dp4a > 2e-5 {
            eprintln!("  ref    {:?}", &yref[..4]);
            eprintln!("  gemv   {:?}", &y_gemv[..4]);
            eprintln!("  ref_q  {:?}", &yref_q[..4]);
            eprintln!("  w4a8   {:?}", &y_w4a8[..4]);
            eprintln!("  dp4a   {:?}", &y_dp4a[..4]);
            eprintln!("  deq[0..4] {:?}  x[0..4] {:?}", &deq[..4], &x[..4]);
        }
        assert!(e_gemv < 2e-5, "{name}: f32 gemv ({e_gemv:.2e})");
        assert!(
            e_w4a8 < 2e-5 && e_dp4a < 2e-5 && e_nc < 2e-5,
            "{name}: int8 lanes ({e_w4a8:.2e} {e_dp4a:.2e} {e_nc:.2e})"
        );
        checked += 1;
    }
    // the fused gate|up pair equals silu(gate . x) * (up . x) off the single lanes
    if let (Ok((gi, _)), Ok((ui, _))) = (
        map.tensor_bytes("blk.0.ffn_gate.weight"),
        map.tensor_bytes("blk.0.ffn_up.weight"),
    ) && is_iq(gi.ggml_type)
        && is_iq(ui.ggml_type)
    {
        let g = exec
            .repack_kquant(&map, "blk.0.ffn_gate.weight")
            .expect("gate");
        let u = exec.repack_kquant(&map, "blk.0.ffn_up.weight").expect("up");
        let (in_dim, out_dim) = (g.dims[0], g.dims[1]);
        let x = deterministic_input(in_dim, 33);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
        let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
        exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, in_dim)
            .expect("quant");
        let mut d_sums = exec.alloc(in_dim / 16).expect("sums");
        exec.q8_sums_strided(&d_xq, &mut d_sums, in_dim, 1)
            .expect("sums");
        let mut d_yg = exec.alloc(out_dim).expect("yg");
        let mut d_yu = exec.alloc(out_dim).expect("yu");
        let mut d_glu = exec.alloc(out_dim).expect("glu");
        let (ng, nu) = (g.ty == GgmlType::Q2K, u.ty == GgmlType::Q2K);
        exec.kquant_gemv_w4a8(&g, &d_xq, &d_xs, ng.then_some(&d_sums), &mut d_yg)
            .expect("g");
        exec.kquant_gemv_w4a8(&u, &d_xq, &d_xs, nu.then_some(&d_sums), &mut d_yu)
            .expect("u");
        exec.kquant_gemv_w4a8_glu(&g, &u, &d_xq, &d_xs, &d_sums, &mut d_glu)
            .expect("glu");
        let (yg, yu, glu) = (
            exec.to_host(&d_yg).expect("h"),
            exec.to_host(&d_yu).expect("h"),
            exec.to_host(&d_glu).expect("h"),
        );
        let want: Vec<f32> = yg
            .iter()
            .zip(&yu)
            .map(|(&a, &b)| (a / (1.0 + (-a).exp())) * b)
            .collect();
        let e = rel_err(&glu, &want);
        eprintln!("blk.0 ffn gate|up glu vs split lanes rel_err {e:.2e}");
        assert!(e < 1e-5, "glu pair ({e:.2e})");
        checked += 1;
    }
    // the token gather: rows equal the dequant
    if let Ok((ti, _)) = map.tensor_bytes("token_embd.weight")
        && is_iq(ti.ggml_type)
    {
        let t = exec.repack_kquant(&map, "token_embd.weight").expect("embd");
        let (embd, vocab) = (t.dims[0], t.dims[1]);
        let toks: Vec<u32> = vec![0, 1, 17, 1000, (vocab - 1) as u32];
        let d_tok = exec.to_device_u32(&toks).expect("tok");
        let mut d_out = exec.alloc(toks.len() * embd).expect("out");
        exec.kquant_gather(&t, &d_tok, &mut d_out, embd, toks.len())
            .expect("gather");
        let got = exec.to_host(&d_out).expect("h");
        let (_, raw) = map.tensor_bytes("token_embd.weight").expect("raw");
        let mut deq = vec![0f32; embd * vocab];
        paddock_kernels::reference::iq::dequant_iq(t.ty.raw(), raw, &mut deq).expect("raw dequant");
        for (i, &tk) in toks.iter().enumerate() {
            let row = &deq[tk as usize * embd..(tk as usize + 1) * embd];
            assert!(
                got[i * embd..(i + 1) * embd] == *row,
                "token {tk}: gather differs from dequant"
            );
        }
        eprintln!("token_embd {:?} gather: {} rows exact", t.ty, toks.len());
        checked += 1;
    }
    assert!(
        checked > 0,
        "no i-quant dense tensor found in {}",
        model.display()
    );
}

/// Bandwidth of the dense i-quant lanes vs the k-quant GEMV on the same
/// shapes: bytes of the repacked plane per call over the measured time.
/// `cargo test ... iq_dense_gemv_bench -- --ignored --nocapture`.
#[test]
#[ignore]
fn iq_dense_gemv_bench() {
    let Some(exec) = common::gpu() else {
        return;
    };
    // Four layers' copies of each plane, cycled per launch, so every launch
    // streams from DRAM (54-70 MB per set vs the 5060 Ti's 32 MB L2); the
    // single-plane numbers are the L2-resident ceiling.
    const LAYERS: [usize; 4] = [0, 8, 16, 24];
    let mut planes: Vec<(String, Vec<paddock_engine::gpu::RepackedKQ>)> = Vec::new();
    let mut add = |label: &str, map: &MappedGguf, base: &str| {
        let mut set = Vec::new();
        for l in LAYERS {
            let n = format!("blk.{l}.{base}");
            match exec.repack_kquant(map, &n) {
                Ok(w) => set.push(w),
                Err(e) => eprintln!("{n}: {e}"),
            }
        }
        if !set.is_empty() {
            planes.push((format!("{label} {base}"), set));
        }
    };
    if let Some(m) = common::model("QWEN35_IQ2_GGUF", common::QWEN35_9B_UD_IQ2) {
        let map = MappedGguf::open(&m).expect("open");
        for n in ["ffn_gate.weight", "ffn_down.weight", "attn_q.weight"] {
            add("9B-IQ2", &map, n);
        }
    }
    if let Some(m) = common::model("QWEN36_MOE_IQ2_GGUF", common::QWEN36_35B_A3B_UD_IQ2) {
        let map = MappedGguf::open(&m).expect("open");
        for n in [
            "attn_qkv.weight",
            "ffn_gate_shexp.weight",
            "ssm_out.weight",
            "attn_q.weight",
        ] {
            add("35B-IQ2-dense", &map, n);
        }
    }
    for (name, set) in &planes {
        let w = &set[0];
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let bytes = (w.data.len() + w.scales.len()) as f64;
        let x = deterministic_input(in_dim, 3);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_y = exec.alloc(out_dim).expect("y");
        let mut d_xq = exec.alloc_i8(in_dim).expect("xq");
        let mut d_xs = exec.alloc(in_dim / 32).expect("xs");
        let mut d_sums = exec.alloc(in_dim / 16).expect("sums");
        exec.quantize_q8_sums(&d_x, &mut d_xq, &mut d_xs, &mut d_sums, in_dim)
            .expect("q");
        let needs = matches!(
            w.ty,
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_0 | GgmlType::Q2K
        );
        let time = |f: &mut dyn FnMut(&paddock_engine::gpu::RepackedKQ), cold: bool| -> f64 {
            let pick = |i: usize| if cold { &set[i % set.len()] } else { &set[0] };
            // best of 3 x 100 launches: the card's clocks ramp under load and
            // a single short burst reads 20-40% low
            let iters = 100;
            let mut best = f64::MAX;
            for _ in 0..3 {
                for i in 0..8 {
                    f(pick(i));
                }
                exec.stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for i in 0..iters {
                    f(pick(i));
                }
                exec.stream.synchronize().unwrap();
                best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
            }
            best
        };
        let mut f32_lane =
            |w: &paddock_engine::gpu::RepackedKQ| exec.kquant_gemv(w, &d_x, &mut d_y).unwrap();
        let t_f32 = time(&mut f32_lane, false);
        let t_f32c = time(&mut f32_lane, true);
        let mut w4a8_lane = |w: &paddock_engine::gpu::RepackedKQ| {
            exec.kquant_gemv_w4a8(w, &d_xq, &d_xs, needs.then_some(&d_sums), &mut d_y)
                .unwrap()
        };
        let t_w4a8 = time(&mut w4a8_lane, false);
        let t_w4a8c = time(&mut w4a8_lane, true);
        eprintln!(
            "{name} {:?} [{in_dim}->{out_dim}] {:.1} MB: gemv_f32 {:.1} us = {:.0} GB/s (cold {:.0}) | w4a8 {:.1} us = {:.0} GB/s (cold {:.1} us = {:.0})",
            w.ty,
            bytes / 1e6,
            t_f32 * 1e6,
            bytes / t_f32 / 1e9,
            bytes / t_f32c / 1e9,
            t_w4a8 * 1e6,
            bytes / t_w4a8 / 1e9,
            t_w4a8c * 1e6,
            bytes / t_w4a8c / 1e9
        );
    }
}

/// The >64-row W4A8 tile (`kquant_gemm_w4a8_pipe2`, slot 580) on dense
/// i-quant planes: against the raw-bytes CPU dequant at 65 rows (the first
/// width the tile serves, one tile column with pad) and against the dp4a
/// lane - the same exact-int class off the same per-32 quantization - at
/// 144 and 1024 rows (two tile columns, ragged; the prefill-chunk width).
/// The v1 / pipe launchers forward the type to pipe2, so they are checked
/// to agree bit for bit.
#[test]
fn iq_w4a8_tile_matches_reference() {
    let Some(model) = common::model("QWEN35_IQ2_GGUF", common::QWEN35_9B_UD_IQ2) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_kquant_iq() || !exec.has_kquant_iq_dense() || !exec.has_kquant_iq_tile() {
        eprintln!("pack lacks the i-quant tile rung (slot 580) - skipping");
        return;
    }
    let map = MappedGguf::open(&model).expect("open gguf");
    let mut checked = 0usize;
    // the 9B UD-IQ2_XXS file: IQ2_XXS qkv / gate / up, IQ3_XXS attn_gate,
    // Q2_K / IQ2_S / IQ3_S down planes by layer, Q3_K token_embd
    for name in [
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.1.ffn_down.weight",
        "blk.18.ffn_down.weight",
        "token_embd.weight",
    ] {
        // PADDOCK_TEST_PLANES=a,b narrows the walk (compute-sanitizer runs)
        if let Ok(list) = std::env::var("PADDOCK_TEST_PLANES")
            && !list.split(',').any(|p| p == name)
        {
            continue;
        }
        let Ok((info, raw)) = map.tensor_bytes(name) else {
            continue;
        };
        let is_iq = matches!(
            info.ggml_type,
            GgmlType::Iq1S
                | GgmlType::Iq1M
                | GgmlType::Iq2Xxs
                | GgmlType::Iq2Xs
                | GgmlType::Iq2S
                | GgmlType::Iq3Xxs
                | GgmlType::Iq3S
                | GgmlType::Iq4Nl
                | GgmlType::Q2K
                | GgmlType::Q3K
        );
        if !is_iq {
            eprintln!("{name}: {:?}, not i-quant - skipped", info.ggml_type);
            continue;
        }
        let w = exec.repack_kquant(&map, name).expect("repack");
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let needs = w.ty == GgmlType::Q2K;
        let n_chunks = in_dim.div_ceil(128);

        // 65 rows against the CPU reference on the dequantized activations
        // (planes up to 64M weights - the vocab-wide ones take the GPU checks)
        if in_dim * out_dim <= 64 << 20 {
            let batch = 65usize;
            let mut deq = vec![0f32; in_dim * out_dim];
            paddock_kernels::reference::iq::dequant_iq(w.ty.raw(), raw, &mut deq)
                .expect("raw dequant");
            let x = deterministic_input(batch * in_dim, 31);
            let mut xdq = vec![0f32; batch * in_dim];
            for r in 0..batch {
                let (qq, ss) = cpu_quantize_q8(&x[r * in_dim..(r + 1) * in_dim]);
                for i in 0..in_dim {
                    xdq[r * in_dim + i] = qq[i] as f32 * ss[i / 32];
                }
            }
            let mut yref = vec![0f32; batch * out_dim];
            for r in 0..batch {
                for o in 0..out_dim {
                    let mut a = 0f64;
                    for i in 0..in_dim {
                        a += deq[o * in_dim + i] as f64 * xdq[r * in_dim + i] as f64;
                    }
                    yref[r * out_dim + o] = a as f32;
                }
            }
            let d_x = exec.to_device(&x).expect("x");
            let batch_pad = batch.div_ceil(128) * 128;
            let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
                .expect("quantize");
            let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
            exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch)
                .expect("sums");
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            exec.kquant_gemm_w4a8_pipe2(&w, &d_yq, needs.then_some(&d_sums), &mut d_y, batch)
                .expect("pipe2");
            let y = exec.to_host(&d_y).expect("y host");
            let e = rel_err(&y, &yref);
            eprintln!(
                "{name} {:?} [{in_dim}->{out_dim}] b={batch}: pipe2 vs cpu {e:.2e}",
                w.ty
            );
            if e > 2e-5 {
                eprintln!("  ref   {:?}", &yref[..4]);
                eprintln!("  pipe2 {:?}", &y[..4]);
            }
            assert!(
                e < 2e-5,
                "{name} b={batch}: pipe2 tile vs CPU reference ({e:.2e})"
            );
        }

        // 144 / 1024 rows against the dp4a lane, and v1 / pipe forward to pipe2
        for batch in [144usize, 1024] {
            let x = deterministic_input(batch * in_dim, 37);
            let d_x = exec.to_device(&x).expect("x");
            let mut d_xq = exec.alloc_i8(batch * in_dim).expect("xq");
            let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
            exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
                .expect("quant");
            let mut d_ssums = exec.alloc(batch * in_dim / 16).expect("ssums");
            exec.q8_sums_strided(&d_xq, &mut d_ssums, in_dim, batch)
                .expect("ssums");
            let mut d_y0 = exec.alloc(out_dim * batch).expect("y0");
            exec.kquant_gemm_dp4a(
                &w,
                &d_xq,
                &d_xs,
                needs.then_some(&d_ssums),
                &mut d_y0,
                batch,
            )
            .expect("dp4a");
            let y_dp4a = exec.to_host(&d_y0).expect("h");

            let batch_pad = batch.div_ceil(128) * 128;
            let mut d_yq = exec.alloc_u8(n_chunks * batch_pad * 144).expect("yq");
            exec.quantize_q8_mmq(&d_x, &mut d_yq, in_dim, batch)
                .expect("quantize");
            let mut d_sums = exec.alloc(n_chunks * batch_pad * 4).expect("sums");
            exec.mmq_sums(&d_yq, &mut d_sums, in_dim, batch)
                .expect("sums");
            let mut d_y = exec.alloc(out_dim * batch).expect("y");
            exec.kquant_gemm_w4a8_pipe2(&w, &d_yq, needs.then_some(&d_sums), &mut d_y, batch)
                .expect("pipe2");
            let y = exec.to_host(&d_y).expect("y host");
            let e = rel_err(&y, &y_dp4a);
            eprintln!("{name} {:?} b={batch}: pipe2 vs dp4a {e:.2e}", w.ty);
            assert!(
                e < 1e-5,
                "{name} b={batch}: pipe2 tile vs dp4a lane ({e:.2e})"
            );

            let mut d_y1 = exec.alloc(out_dim * batch).expect("y1");
            exec.kquant_gemm_w4a8(&w, &d_yq, needs.then_some(&d_sums), &mut d_y1, batch)
                .expect("w4a8 v1 -> pipe2");
            let y1 = exec.to_host(&d_y1).expect("h");
            assert!(
                y1 == y,
                "{name} b={batch}: the v1 launcher did not forward to pipe2"
            );
            let mut d_y2 = exec.alloc(out_dim * batch).expect("y2");
            exec.kquant_gemm_w4a8_pipe(&w, &d_yq, needs.then_some(&d_sums), &mut d_y2, batch)
                .expect("w4a8 pipe -> pipe2");
            let y2 = exec.to_host(&d_y2).expect("h");
            assert!(
                y2 == y,
                "{name} b={batch}: the pipe launcher did not forward to pipe2"
            );
        }
        checked += 1;
    }
    assert!(checked > 0, "no i-quant plane in the file");
}
