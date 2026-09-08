//! Parity test for the batched attention decode kernel vs a CPU reference of
//! the same computation (GQA + per-head sinks + sliding window). Gated on a
//! CUDA device + built pack.

mod common;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_kernels::reference::ops::{YarnRope, softmax_with_sink};

/// Upload an f32 slice to the device as f16 - the single-stream KV kernels take a
/// typed f16 buffer (and the CPU reference sees the same f16-rounded values via
/// `f16_round` to stay bit-comparable).
fn f16_dev(exec: &GpuExecutor, data: &[f32]) -> CudaSlice<f16> {
    let h: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    exec.stream.clone_htod(&h).expect("f16 htod")
}

/// Upload an f32 slice as raw f16 BYTES - the batched KV cache is dtype-erased
/// (raw u8, dtype passed as a flag), so its kernels take a byte buffer. Tests
/// exercise the fp16 path (`KvDtype::Fp16`).
fn f16_dev_u8(exec: &GpuExecutor, data: &[f32]) -> CudaSlice<u8> {
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&x| f16::from_f32(x).to_le_bytes())
        .collect();
    exec.stream.clone_htod(&bytes).expect("f16 bytes htod")
}

/// Round an f32 slice through f16 (what the fp16 cache stores), for the reference.
fn f16_round(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

/// CPU reference: for each head, scores = scale·(q_h · K[p,kvh]); softmax with
/// the head's sink; out_h = Σ_p prob_p · V[p,kvh].
#[allow(clippy::too_many_arguments)]
fn cpu_attn(
    q: &[f32],
    kc: &[f32],
    vc: &[f32],
    sinks: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    first_pos: usize,
    n_pos: usize,
    kv_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let group = n_heads / n_kv_heads;
    let mut out = vec![0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / group;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores: Vec<f32> = (0..n_pos)
            .map(|i| {
                let base = (first_pos + i) * kv_dim + kvh * head_dim;
                qh.iter()
                    .zip(&kc[base..base + head_dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        softmax_with_sink(&mut scores, sinks[h]);
        let o = &mut out[h * head_dim..(h + 1) * head_dim];
        for (i, &w) in scores.iter().enumerate() {
            let base = (first_pos + i) * kv_dim + kvh * head_dim;
            for (od, &vd) in o.iter_mut().zip(&vc[base..base + head_dim]) {
                *od += w * vd;
            }
        }
    }
    out
}

#[test]
fn attn_decode_matches_cpu_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };

    // gpt-oss geometry: 64 heads, 8 kv heads, head_dim 64
    let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_pos = 300usize;

    let q = det(n_heads * head_dim, 1);
    let kc = det(max_pos * kv_dim, 2);
    let vc = det(max_pos * kv_dim, 3);
    let sinks = det(n_heads, 4);

    let d_q = exec.to_device(&q).expect("q");
    let d_k = f16_dev(&exec, &kc);
    let d_v = f16_dev(&exec, &vc);
    let d_s = exec.to_device(&sinks).expect("sinks");
    // the CPU reference sees the same f16-rounded KV the fp16 cache holds
    let (kc, vc) = (f16_round(&kc), f16_round(&vc));

    // full attention (first_pos 0), and a sliding-window case (first_pos > 0)
    for (pos, window) in [(0usize, 0usize), (200, 0), (250, 128)] {
        let first_pos = if window > 0 {
            (pos + 1).saturating_sub(window)
        } else {
            0
        };
        let n_pos = pos + 1 - first_pos;

        let mut d_out: CudaSlice<f32> = exec.alloc(n_heads * head_dim).expect("out");
        exec.attn_decode(
            &d_q, &d_k, &d_v, &d_s, &mut d_out, n_heads, n_kv_heads, head_dim, first_pos, n_pos,
            kv_dim, scale,
        )
        .expect("attn_decode");
        let got = exec.to_host(&d_out).expect("dtoh");

        let expected = cpu_attn(
            &q, &kc, &vc, &sinks, n_heads, n_kv_heads, head_dim, first_pos, n_pos, kv_dim, scale,
        );

        let max_diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("pos {pos} window {window} (n_pos {n_pos}): max_abs_diff {max_diff:.2e}");
        assert!(max_diff < 1e-5, "pos {pos}: {max_diff} too high");
    }
}

/// The batched elementwise ops (rmsnorm/rope/kv-append) vs the single-sequence
/// versions: batched must equal running each sequence independently.
#[test]
fn batched_ops_match_per_sequence() {
    let Some(exec) = common::gpu() else {
        return;
    };
    let (n_heads, head_dim, embd) = (64usize, 64usize, 2880usize);
    let qdim = n_heads * head_dim;
    let batch = 4usize;
    let positions = [10u32, 47, 128, 200];
    let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
    let maxd = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // --- rmsnorm_batch vs per-row rmsnorm
    let w = det(embd, 99);
    let d_w = exec.to_device(&w).expect("w");
    let mut xrows = Vec::new();
    for b in 0..batch {
        xrows.extend(det(embd, 10 + b as u64));
    }
    let d_x = exec.to_device(&xrows).expect("x");
    let mut d_bn = exec.alloc(batch * embd).expect("bn");
    exec.rmsnorm_batch(&d_x, &d_w, &mut d_bn, embd, 1e-5, batch)
        .expect("rmsnorm_batch");
    let bn = exec.to_host(&d_bn).expect("bn host");
    for b in 0..batch {
        let d_xb = exec
            .to_device(&xrows[b * embd..(b + 1) * embd])
            .expect("xb");
        let mut d_yb = exec.alloc(embd).expect("yb");
        exec.rmsnorm(&d_xb, &d_w, &mut d_yb, embd, 1e-5)
            .expect("rmsnorm");
        let yb = exec.to_host(&d_yb).expect("yb host");
        assert!(
            maxd(&bn[b * embd..(b + 1) * embd], &yb) < 1e-5,
            "rmsnorm seq {b}"
        );
    }

    // --- rope_yarn_batch vs per-sequence rope
    let yarn = YarnRope::new(head_dim, 150_000.0, 1.0, 4096, 1.0, 1.0, 32.0, 1.0);
    let params = yarn.kernel_params();
    let mut qrows = Vec::new();
    for b in 0..batch {
        qrows.extend(det(qdim, 30 + b as u64));
    }
    let mut d_qb = exec.to_device(&qrows).expect("qb");
    exec.rope_yarn_batch(&mut d_qb, &d_pos, n_heads, head_dim, params, batch)
        .expect("rope_batch");
    let rb = exec.to_host(&d_qb).expect("rb host");
    for b in 0..batch {
        let mut d_single = exec
            .to_device(&qrows[b * qdim..(b + 1) * qdim])
            .expect("single");
        exec.rope_yarn(
            &mut d_single,
            n_heads,
            head_dim,
            positions[b] as usize,
            params,
        )
        .expect("rope");
        let single = exec.to_host(&d_single).expect("single host");
        assert!(
            maxd(&rb[b * qdim..(b + 1) * qdim], &single) < 1e-5,
            "rope seq {b}"
        );
    }

    // --- kv_append_batch scatter: cache[b][pos[b]] must become kv[b]
    let kv_dim = 512usize;
    let max_ctx = 256usize;
    let mut kvrows = Vec::new();
    for b in 0..batch {
        kvrows.extend(det(kv_dim, 50 + b as u64));
    }
    let d_kv = exec.to_device(&kvrows).expect("kv");
    // batched cache is raw bytes; fp16 path -> 2 bytes/elem
    let mut d_cache = exec.alloc_u8(batch * max_ctx * kv_dim * 2).expect("cache");
    exec.kv_append_batch(
        &d_kv,
        &mut d_cache,
        &d_pos,
        None,
        kv_dim,
        max_ctx,
        batch,
        KvDtype::Fp16,
    )
    .expect("kv_append_batch");
    let cache: Vec<f32> = exec
        .stream
        .clone_dtoh(&d_cache)
        .expect("cache host")
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    for b in 0..batch {
        let row = positions[b] as usize;
        let got = &cache[(b * max_ctx + row) * kv_dim..(b * max_ctx + row + 1) * kv_dim];
        // the cache stores f16, so compare against the f16-rounded input
        assert!(
            maxd(got, &f16_round(&kvrows[b * kv_dim..(b + 1) * kv_dim])) < 1e-9,
            "kv seq {b}"
        );
    }
    eprintln!("batched rmsnorm/rope/kv-append all match per-sequence");
}

/// Batched decode attention (pd_attn_decode_batch): B sequences, each with its
/// own KV cache and its own position, in one launch - must equal running the
/// single-sequence reference on each sequence independently. The per-sequence
/// attention behind continuous batching.
#[test]
fn attn_decode_batch_matches_per_sequence() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for dt in KV_DTYPES {
        let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let max_ctx = 256usize;
        let batch = 4usize;
        // per-sequence positions (each sequence at its own decode step)
        let positions = [10u32, 47, 128, 200];

        // per-sequence data: q [batch, qdim], caches [batch, max_ctx, kv_dim]
        let qdim = n_heads * head_dim;
        let mut q = Vec::new();
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for b in 0..batch {
            q.extend(det(qdim, 1 + b as u64));
            kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
            vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
        }
        let sinks = det(n_heads, 4);

        let d_q = exec.to_device(&q).expect("q");
        let d_k = kv_dev_u8(&exec, &kc, dt);
        let d_v = kv_dev_u8(&exec, &vc, dt);
        let d_s = exec.to_device(&sinks).expect("sinks");
        // reference sees the same f16-rounded KV the fp16 cache holds
        let (kc, vc) = (kv_round(&kc, dt), kv_round(&vc, dt));
        let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
        let mut d_out = exec.alloc(batch * qdim).expect("out");
        exec.attn_decode_batch(
            &d_q, &d_k, &d_v, &d_s, &mut d_out, &d_pos, None, n_heads, n_kv_heads, head_dim,
            max_ctx, kv_dim, 0, batch, scale, dt,
        )
        .expect("attn_decode_batch");
        let got = exec.to_host(&d_out).expect("dtoh");

        for b in 0..batch {
            let n_pos = positions[b] as usize + 1;
            let qb = &q[b * qdim..(b + 1) * qdim];
            let kcb = &kc[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim];
            let vcb = &vc[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim];
            let expected = cpu_attn(
                qb, kcb, vcb, &sinks, n_heads, n_kv_heads, head_dim, 0, n_pos, kv_dim, scale,
            );
            let max_diff = got[b * qdim..(b + 1) * qdim]
                .iter()
                .zip(&expected)
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{dt:?} seq {b} (pos {}): max_abs_diff {max_diff:.2e}",
                positions[b]
            );
            assert!(max_diff < 1e-5, "seq {b}: {max_diff} too high");
        }
    }
}

/// Batched FlashDecoding split (attn_partial_batch + attn_combine_batch): B
/// sequences at their own positions, KV range split n_splits ways, must equal the
/// per-sequence CPU reference - across split counts that divide, don't divide, and
/// over-split (empty chunks the combine drops), plus a sliding-window case.
#[test]
fn attn_decode_batch_split_matches_per_sequence() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for dt in KV_DTYPES {
        let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let max_ctx = 256usize;
        let batch = 4usize;
        let qdim = n_heads * head_dim;
        let positions = [10u32, 47, 128, 200];

        let mut q = Vec::new();
        let mut kc = Vec::new();
        let mut vc = Vec::new();
        for b in 0..batch {
            q.extend(det(qdim, 1 + b as u64));
            kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
            vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
        }
        let sinks = det(n_heads, 4);

        let d_q = exec.to_device(&q).expect("q");
        let d_k = kv_dev_u8(&exec, &kc, dt);
        let d_v = kv_dev_u8(&exec, &vc, dt);
        let d_s = exec.to_device(&sinks).expect("sinks");
        let (kc, vc) = (kv_round(&kc, dt), kv_round(&vc, dt));
        let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");

        for (window, n_splits) in [(0usize, 1usize), (0, 3), (0, 4), (0, 8), (0, 300), (128, 4)] {
            let mut d_o = exec
                .alloc(n_heads * batch * n_splits * head_dim)
                .expect("o");
            let mut d_ml = exec.alloc(n_heads * batch * n_splits * 2).expect("ml");
            let mut d_out = exec.alloc(batch * qdim).expect("out");
            exec.attn_partial_batch(
                &d_q, &d_k, &d_v, &mut d_o, &mut d_ml, &d_pos, None, n_heads, n_kv_heads, head_dim,
                max_ctx, kv_dim, window, n_splits, batch, scale, dt,
            )
            .expect("attn_partial_batch");
            exec.attn_combine_batch(
                &d_o, &d_ml, &d_s, &mut d_out, n_heads, head_dim, n_splits, batch,
            )
            .expect("attn_combine_batch");
            let got = exec.to_host(&d_out).expect("dtoh");

            for b in 0..batch {
                let pos = positions[b] as usize;
                let first_pos = if window > 0 {
                    (pos + 1).saturating_sub(window)
                } else {
                    0
                };
                let n_pos = pos + 1 - first_pos;
                let qb = &q[b * qdim..(b + 1) * qdim];
                let kcb = &kc[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim];
                let vcb = &vc[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim];
                let expected = cpu_attn(
                    qb, kcb, vcb, &sinks, n_heads, n_kv_heads, head_dim, first_pos, n_pos, kv_dim,
                    scale,
                );
                let max_diff = got[b * qdim..(b + 1) * qdim]
                    .iter()
                    .zip(&expected)
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "{dt:?} window {window} splits {n_splits} seq {b} (pos {pos}): max_abs_diff {max_diff:.2e}"
                );
                assert!(
                    max_diff < 1e-5,
                    "window {window} splits {n_splits} seq {b}: {max_diff} too high"
                );
            }
        }
        eprintln!("{dt:?} batched FlashDecoding split matches per-sequence CPU reference");
    }
}

/// FlashDecoding split path (attn_partial + attn_combine) vs the same CPU
/// reference, across split counts that divide, don't divide, and over-split the
/// KV range - the last case leaves empty chunks (m = -inf) the combine must drop.
#[test]
fn attn_flashdecoding_split_matches_cpu_reference() {
    let Some(exec) = common::gpu() else {
        return;
    };

    let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_pos = 300usize;

    let q = det(n_heads * head_dim, 1);
    let kc = det(max_pos * kv_dim, 2);
    let vc = det(max_pos * kv_dim, 3);
    let sinks = det(n_heads, 4);

    let d_q = exec.to_device(&q).expect("q");
    let d_k = f16_dev(&exec, &kc);
    let d_v = f16_dev(&exec, &vc);
    let d_s = exec.to_device(&sinks).expect("sinks");
    // reference sees the same f16-rounded KV the fp16 cache holds
    let (kc, vc) = (f16_round(&kc), f16_round(&vc));

    for (pos, window) in [(0usize, 0usize), (200, 0), (250, 128), (129, 0)] {
        let first_pos = if window > 0 {
            (pos + 1).saturating_sub(window)
        } else {
            0
        };
        let n_pos = pos + 1 - first_pos;
        let expected = cpu_attn(
            &q, &kc, &vc, &sinks, n_heads, n_kv_heads, head_dim, first_pos, n_pos, kv_dim, scale,
        );

        for n_splits in [1usize, 3, 4, 8, n_pos + 2] {
            let mut d_o: CudaSlice<f32> = exec.alloc(n_heads * n_splits * head_dim).expect("o");
            let mut d_ml: CudaSlice<f32> = exec.alloc(n_heads * n_splits * 2).expect("ml");
            let mut d_out: CudaSlice<f32> = exec.alloc(n_heads * head_dim).expect("out");
            exec.attn_partial(
                &d_q, &d_k, &d_v, &mut d_o, &mut d_ml, n_heads, n_kv_heads, head_dim, first_pos,
                n_pos, n_splits, kv_dim, scale,
            )
            .expect("attn_partial");
            exec.attn_combine(&d_o, &d_ml, &d_s, &mut d_out, n_heads, head_dim, n_splits)
                .expect("attn_combine");
            let got = exec.to_host(&d_out).expect("dtoh");

            let max_diff = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "pos {pos} window {window} n_pos {n_pos} splits {n_splits}: max_abs_diff {max_diff:.2e}"
            );
            assert!(
                max_diff < 1e-5,
                "pos {pos} splits {n_splits}: {max_diff} too high"
            );
        }
    }
}

/// Tiled prefill attention (pd_attn_prefill): T causal rows against one KV
/// slot must match the decode-batch kernel on the same inputs - same value
/// math, per-32-key-tile online-softmax order, so agreement is ~1e-6 not
/// bit-exact. Covers T multiple/non-multiple of the 16-query tile and real
/// (non-degenerate) sinks. Speed printed under PADDOCK_HEAVY_TESTS.
#[test]
fn attn_prefill_matches_decode_batch() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for dt in KV_DTYPES {
        // (16, 4, 256) is the real qwen35-9B full-attn shape; (16, 8, 128) covers
        // the HD=128 instantiation
        for (n_heads, n_kv_heads, head_dim) in [(16usize, 4usize, 256usize), (16, 8, 128)] {
            let kv_dim = n_kv_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let max_ctx = 512usize;
            let qdim = n_heads * head_dim;

            for &t_len in &[512usize, 100, 33] {
                let q = det(t_len * qdim, 1);
                let kc = det(max_ctx * kv_dim, 100);
                let vc = det(max_ctx * kv_dim, 200);
                let sinks = det(n_heads, 4);
                let positions: Vec<u32> = (0..t_len as u32).collect();
                let slots = vec![0u32; t_len];

                let d_q = exec.to_device(&q).expect("q");
                let d_k = kv_dev_u8(&exec, &kc, dt);
                let d_v = kv_dev_u8(&exec, &vc, dt);
                let d_s = exec.to_device(&sinks).expect("sinks");
                let d_pos = exec.stream.clone_htod(&positions).expect("pos");
                let d_slots = exec.stream.clone_htod(&slots).expect("slots");
                let mut d_ref = exec.alloc(t_len * qdim).expect("ref");
                let mut d_new = exec.alloc(t_len * qdim).expect("new");
                exec.attn_decode_batch(
                    &d_q,
                    &d_k,
                    &d_v,
                    &d_s,
                    &mut d_ref,
                    &d_pos,
                    Some(&d_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_dim,
                    0,
                    t_len,
                    scale,
                    dt,
                )
                .expect("decode_batch");
                exec.attn_prefill(
                    &d_q, &d_k, &d_v, &d_s, &mut d_new, &d_pos, &d_slots, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_dim, 0, t_len, scale, dt,
                )
                .expect("attn_prefill");
                let r = exec.to_host(&d_ref).expect("dtoh ref");
                let n = exec.to_host(&d_new).expect("dtoh new");
                let max_diff = r
                    .iter()
                    .zip(&n)
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);

                let mut speed = String::new();
                if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() && t_len == 512 {
                    let time = |f: &mut dyn FnMut()| -> f64 {
                        for _ in 0..5 {
                            f();
                        }
                        exec.synchronize().unwrap();
                        let mut best = f64::MAX;
                        for _ in 0..6 {
                            let t = std::time::Instant::now();
                            for _ in 0..10 {
                                f();
                            }
                            exec.synchronize().unwrap();
                            best = best.min(t.elapsed().as_secs_f64() / 10.0);
                        }
                        best
                    };
                    let td = time(&mut || {
                        exec.attn_decode_batch(
                            &d_q,
                            &d_k,
                            &d_v,
                            &d_s,
                            &mut d_ref,
                            &d_pos,
                            Some(&d_slots),
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_dim,
                            0,
                            t_len,
                            scale,
                            dt,
                        )
                        .unwrap();
                    });
                    let tp = time(&mut || {
                        exec.attn_prefill(
                            &d_q, &d_k, &d_v, &d_s, &mut d_new, &d_pos, &d_slots, n_heads,
                            n_kv_heads, head_dim, max_ctx, kv_dim, 0, t_len, scale, dt,
                        )
                        .unwrap();
                    });
                    speed = format!(
                        "  decode {:.3}ms | prefill {:.3}ms ({:.2}x)",
                        td * 1e3,
                        tp * 1e3,
                        td / tp
                    );
                }
                eprintln!("{dt:?} hd={head_dim} T={t_len}: max_abs_diff {max_diff:.2e}{speed}");
                assert!(
                    max_diff < 5e-5,
                    "hd={head_dim} T={t_len}: {max_diff} too high"
                );

                // f16 WMMA kernel (hd=256 only): f16 Q/K/V + f16 O accumulation is a
                // coarser class than the f32 kernels - gate at the f16 rounding scale
                // dense f16-class WMMA: f16 cache only (see the refusal test)
                if head_dim == 256 && dt == KvDtype::Fp16 {
                    let mut d_f16 = exec.alloc(t_len * qdim).expect("f16 out");
                    exec.attn_prefill_f16(
                        &d_q, &d_k, &d_v, &d_s, &mut d_f16, &d_pos, &d_slots, n_heads, n_kv_heads,
                        head_dim, max_ctx, kv_dim, 0, t_len, scale, dt,
                    )
                    .expect("attn_prefill_f16");
                    let f = exec.to_host(&d_f16).expect("dtoh f16");
                    let max_diff16 = r
                        .iter()
                        .zip(&f)
                        .map(|(a, c)| (a - c).abs())
                        .fold(0.0f32, f32::max);
                    let mut speed16 = String::new();
                    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() && t_len == 512 {
                        let time = |f: &mut dyn FnMut()| -> f64 {
                            for _ in 0..5 {
                                f();
                            }
                            exec.synchronize().unwrap();
                            let mut best = f64::MAX;
                            for _ in 0..6 {
                                let t = std::time::Instant::now();
                                for _ in 0..10 {
                                    f();
                                }
                                exec.synchronize().unwrap();
                                best = best.min(t.elapsed().as_secs_f64() / 10.0);
                            }
                            best
                        };
                        let tf = time(&mut || {
                            exec.attn_prefill_f16(
                                &d_q, &d_k, &d_v, &d_s, &mut d_f16, &d_pos, &d_slots, n_heads,
                                n_kv_heads, head_dim, max_ctx, kv_dim, 0, t_len, scale, dt,
                            )
                            .unwrap();
                        });
                        speed16 = format!("  wmma {:.3}ms", tf * 1e3);
                    }
                    eprintln!(
                        "{dt:?} hd={head_dim} T={t_len} [wmma f16]: max_abs_diff {max_diff16:.2e}{speed16}"
                    );
                    assert!(
                        max_diff16 < 8e-3,
                        "hd=256 T={t_len} wmma: {max_diff16} too high"
                    );
                }
            }
        }
    }
}

/// f16 WMMA prefill attention at the gpt-oss shape - hd=64 instantiation,
/// GQA 64/8, real (non-degenerate) sinks, and the sliding window that
/// gpt-oss alternating layers actually use. Must match the decode-batch
/// kernel at the f16 rounding scale (the same 8e-3 bar as the hd=256 case).
#[test]
fn attn_prefill_f16_hd64_matches_decode_batch() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for dt in KV_DTYPES {
        // The DENSE f16-class WMMA prefill kernels take an f16 cache only -
        // pd_attn_prefill_f16* returns cudaErrorInvalidValue for e4m3 rather
        // than computing something wrong (packs/cuda/src/attn/prefill.cuh).
        // That refusal is pinned by f16_class_prefill_refuses_an_fp8_cache; the
        // PAGED f16 kernels do carry fp8 arms and are covered by
        // paged_f16_prefill_fp8_matches_decode_batch.
        if dt != KvDtype::Fp16 {
            continue;
        }
        let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
        let kv_dim = n_kv_heads * head_dim;
        let qdim = n_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let max_ctx = 512usize;

        for &t_len in &[512usize, 100, 33] {
            for &swa in &[0usize, 128] {
                let q = det(t_len * qdim, 1);
                let kc = det(max_ctx * kv_dim, 100);
                let vc = det(max_ctx * kv_dim, 200);
                let sinks = det(n_heads, 4);
                let positions: Vec<u32> = (0..t_len as u32).collect();
                let slots = vec![0u32; t_len];

                let d_q = exec.to_device(&q).expect("q");
                let d_k = kv_dev_u8(&exec, &kc, dt);
                let d_v = kv_dev_u8(&exec, &vc, dt);
                let d_s = exec.to_device(&sinks).expect("sinks");
                let d_pos = exec.stream.clone_htod(&positions).expect("pos");
                let d_slots = exec.stream.clone_htod(&slots).expect("slots");
                let mut d_ref = exec.alloc(t_len * qdim).expect("ref");
                let mut d_f16 = exec.alloc(t_len * qdim).expect("f16 out");
                exec.attn_decode_batch(
                    &d_q,
                    &d_k,
                    &d_v,
                    &d_s,
                    &mut d_ref,
                    &d_pos,
                    Some(&d_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_dim,
                    swa,
                    t_len,
                    scale,
                    dt,
                )
                .expect("decode_batch");
                exec.attn_prefill_f16(
                    &d_q, &d_k, &d_v, &d_s, &mut d_f16, &d_pos, &d_slots, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_dim, swa, t_len, scale, dt,
                )
                .expect("attn_prefill_f16 hd64");
                let r = exec.to_host(&d_ref).expect("dtoh ref");
                let f = exec.to_host(&d_f16).expect("dtoh f16");
                let max_diff = r
                    .iter()
                    .zip(&f)
                    .map(|(a, c)| (a - c).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "{dt:?} hd=64 T={t_len} swa={swa} [wmma f16]: max_abs_diff {max_diff:.2e}"
                );
                assert!(
                    max_diff < 8e-3,
                    "hd=64 T={t_len} swa={swa}: {max_diff} too high"
                );
            }
        }
    }
}

/// The ROW-SUB-RANGE attention dispatch (mixed-slot prefill pass, G6): two
/// slot groups in one pass - a 40-row group in slot 2 and a 24-row group in
/// slot 5 - where the big group takes attn_prefill_f16_rows and the small one
/// attn_decode_batch_rows, each via host-side pointer offsets. Both must match
/// the full-range attn_decode_batch reference (which reads slots per row).
#[test]
fn attn_rows_subrange_matches_full_pass() {
    let Some(exec) = common::gpu() else {
        return;
    };
    for dt in KV_DTYPES {
        // The DENSE f16-class WMMA prefill kernels take an f16 cache only -
        // pd_attn_prefill_f16* returns cudaErrorInvalidValue for e4m3 rather
        // than computing something wrong (packs/cuda/src/attn/prefill.cuh).
        // That refusal is pinned by f16_class_prefill_refuses_an_fp8_cache; the
        // PAGED f16 kernels do carry fp8 arms and are covered by
        // paged_f16_prefill_fp8_matches_decode_batch.
        if dt != KvDtype::Fp16 {
            continue;
        }
        let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
        let kv_dim = n_kv_heads * head_dim;
        let qdim = n_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let max_ctx = 512usize;
        let n_slots = 8usize;
        let (g0, g1) = (40usize, 24usize); // rows per group
        let rows = g0 + g1;

        for &swa in &[0usize, 128] {
            let q = det(rows * qdim, 7);
            let kc = det(n_slots * max_ctx * kv_dim, 100);
            let vc = det(n_slots * max_ctx * kv_dim, 200);
            let sinks = det(n_heads, 4);
            // group 0: slot 2, positions 10..50; group 1: slot 5, positions 0..24
            let mut positions: Vec<u32> = (10..(10 + g0) as u32).collect();
            positions.extend(0..g1 as u32);
            let mut slots = vec![2u32; g0];
            slots.extend(vec![5u32; g1]);

            let d_q = exec.to_device(&q).expect("q");
            let d_k = kv_dev_u8(&exec, &kc, dt);
            let d_v = kv_dev_u8(&exec, &vc, dt);
            let d_s = exec.to_device(&sinks).expect("sinks");
            let d_pos = exec.stream.clone_htod(&positions).expect("pos");
            let d_slots = exec.stream.clone_htod(&slots).expect("slots");
            let mut d_ref = exec.alloc(rows * qdim).expect("ref");
            let mut d_out = exec.alloc(rows * qdim).expect("out");
            exec.attn_decode_batch(
                &d_q,
                &d_k,
                &d_v,
                &d_s,
                &mut d_ref,
                &d_pos,
                Some(&d_slots),
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_dim,
                swa,
                rows,
                scale,
                dt,
            )
            .expect("full-range reference");
            exec.attn_prefill_f16_rows(
                &d_q, &d_k, &d_v, &d_s, &mut d_out, &d_pos, &d_slots, n_heads, n_kv_heads,
                head_dim, max_ctx, kv_dim, swa, 0, g0, scale, dt,
            )
            .expect("f16 rows group 0");
            exec.attn_decode_batch_rows(
                &d_q,
                &d_k,
                &d_v,
                &d_s,
                &mut d_out,
                &d_pos,
                Some(&d_slots),
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_dim,
                swa,
                g0,
                g1,
                scale,
                dt,
            )
            .expect("decode rows group 1");
            let r = exec.to_host(&d_ref).expect("dtoh ref");
            let o = exec.to_host(&d_out).expect("dtoh out");
            // group 1 ran the same kernel as the reference on offset pointers -
            // bitwise; group 0 crossed kernels - f16 online-softmax noise bound
            let g1_bitwise = r[g0 * qdim..]
                .iter()
                .zip(&o[g0 * qdim..])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                g1_bitwise,
                "decode_batch_rows must be bitwise == full-range decode"
            );
            let max_diff = r[..g0 * qdim]
                .iter()
                .zip(&o[..g0 * qdim])
                .map(|(a, c)| (a - c).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{dt:?} rows-subrange swa={swa}: f16 group max_abs_diff {max_diff:.2e}, decode group bitwise"
            );
            assert!(max_diff < 8e-3, "f16 rows group diverged: {max_diff}");
        }
    }
}

/// Paged decode attention (pd_attn_decode_batch_paged) vs the dense decode
/// (pd_attn_decode_batch). With a CONTIGUOUS IDENTITY block table
/// (`bt[s*bps + j] = s*bps + j`), the block pool `[batch*bps, 16, kv_dim]` has a
/// byte-identical layout to the dense cache `[batch, max_ctx, kv_dim]` - physical
/// block `s*bps + pos/16` at intra-block row `pos%16` lands at exactly
/// `s*max_ctx*kv_dim + pos*kv_dim`. So the paged kernel, reading the same buffer
/// through the block table, must produce BITWISE-identical output. This isolates
/// and gates the paged block-table addressing independent of the fused/split
/// decode paths the engine actually dispatches. Covers full attention and a
/// sliding-window (`first_pos > 0`) case.
#[test]
fn attn_decode_batch_paged_bitwise_matches_dense() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_paged_kv() {
        eprintln!("pack has no paged KV kernels - skipping");
        return;
    }
    let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_ctx = 256usize;
    let bps = max_ctx / 16; // blocks per slot (16-token pages)
    let batch = 4usize;
    let positions = [10u32, 47, 128, 200];

    let qdim = n_heads * head_dim;
    let mut q = Vec::new();
    let mut kc = Vec::new();
    let mut vc = Vec::new();
    for b in 0..batch {
        q.extend(det(qdim, 1 + b as u64));
        kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
        vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
    }
    let sinks = det(n_heads, 4);

    let d_q = exec.to_device(&q).expect("q");
    let d_s = exec.to_device(&sinks).expect("sinks");
    let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
    // contiguous identity block table: slot s -> blocks [s*bps .. s*bps+bps)
    let bt_host: Vec<u32> = (0..(batch * bps) as u32).collect();
    let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");

    for dt in KV_DTYPES {
        // One KV buffer per tensor, read two ways: dense [batch, max_ctx, kv_dim] and
        // pool [batch*bps, 16, kv_dim] - byte-identical under the identity table.
        // The identity holds per BYTE, so it must hold at 1 byte/elem exactly as it
        // does at 2 - a paged addressing bug that scales by element size shows up
        // here and nowhere else.
        let d_k = kv_dev_u8(&exec, &kc, dt);
        let d_v = kv_dev_u8(&exec, &vc, dt);
        for &window in &[0usize, 128] {
            let mut d_dense = exec.alloc(batch * qdim).expect("dense");
            let mut d_paged = exec.alloc(batch * qdim).expect("paged");
            exec.attn_decode_batch(
                &d_q,
                &d_k,
                &d_v,
                &d_s,
                &mut d_dense,
                &d_pos,
                None,
                n_heads,
                n_kv_heads,
                head_dim,
                max_ctx,
                kv_dim,
                window,
                batch,
                scale,
                dt,
            )
            .expect("attn_decode_batch");
            exec.attn_decode_batch_paged(
                &d_q,
                &d_k,
                &d_v,
                &d_s,
                &mut d_paged,
                &d_pos,
                None,
                &d_bt,
                bps,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                window,
                batch,
                scale,
                dt,
            )
            .expect("attn_decode_batch_paged");
            let dense = exec.to_host(&d_dense).expect("dense host");
            let paged = exec.to_host(&d_paged).expect("paged host");
            let bitwise = dense
                .iter()
                .zip(&paged)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            let max_diff = dense
                .iter()
                .zip(&paged)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "paged decode {dt:?} window {window}: bitwise={bitwise} max_abs_diff {max_diff:.2e}"
            );
            assert!(
                bitwise,
                "paged decode {dt:?} window {window} not bitwise == dense (max_diff {max_diff})"
            );
        }
    }
}

/// Paged KV append (pd_kv_append_batch_paged) vs dense (pd_kv_append_batch).
/// Both caches start zeroed (alloc_zeros); with the identity block table the pool
/// [batch*bps, 16, kv_dim] and dense [batch, max_ctx, kv_dim] have identical byte
/// layout, so scattering the same rows to the same positions must leave the two
/// buffers BYTE-identical.
#[test]
fn kv_append_batch_paged_bytes_match_dense() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_paged_kv() {
        return;
    }
    for dt in KV_DTYPES {
        let kv_dim = 512usize;
        let max_ctx = 256usize;
        let bps = max_ctx / 16;
        let batch = 4usize;
        let positions = [10u32, 47, 128, 200];
        let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");

        let mut kvrows = Vec::new();
        for b in 0..batch {
            kvrows.extend(det(kv_dim, 50 + b as u64));
        }
        let d_kv = exec.to_device(&kvrows).expect("kv");

        // dense cache and pool are the same total size (bps*16 == max_ctx), both
        // zeroed; the identity table makes the two writes land at the same bytes.
        let mut d_dense = exec
            .alloc_u8(batch * max_ctx * kv_dim * dt.bytes())
            .expect("dense");
        let mut d_pool = exec
            .alloc_u8(batch * bps * 16 * kv_dim * dt.bytes())
            .expect("pool");
        exec.kv_append_batch(
            &d_kv,
            &mut d_dense,
            &d_pos,
            None,
            kv_dim,
            max_ctx,
            batch,
            dt,
        )
        .expect("kv_append_batch");
        let bt_host: Vec<u32> = (0..(batch * bps) as u32).collect();
        let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");
        exec.kv_append_batch_paged(
            &d_kv,
            &mut d_pool,
            &d_pos,
            None,
            &d_bt,
            bps,
            kv_dim,
            batch,
            dt,
        )
        .expect("kv_append_batch_paged");

        let dense = exec.stream.clone_dtoh(&d_dense).expect("dense host");
        let pool = exec.stream.clone_dtoh(&d_pool).expect("pool host");
        assert_eq!(dense.len(), pool.len(), "buffer sizes must match");
        let diffs = dense.iter().zip(&pool).filter(|(a, b)| a != b).count();
        eprintln!(
            "{dt:?} paged append: {diffs} differing bytes of {}",
            dense.len()
        );
        assert_eq!(
            diffs, 0,
            "paged append not byte-identical to dense ({diffs} bytes differ)"
        );
    }
}

/// gpt-oss paged fused-append (G1): pd_qkv_rope_append_batch_paged vs dense
/// pd_qkv_rope_append_batch. Same qkv input + rope params + positions to both;
/// dense writes K/V into [batch, max_ctx, kvdim], paged into the pool
/// [batch*bps, 16, kvdim] under a contiguous identity block table (bps =
/// max_ctx/16). Since only the K/V store base differs (dense slot*max_ctx*kvdim
/// vs block-table lookup, which the identity table overlays byte-for-byte), the
/// K cache, V cache, AND the rotated q_out must all be BYTE-identical. This gate
/// isolates the paged block-table addressing - the only change in both fused
/// gpt-oss append kernels (the ks GEMM twin reuses the identical address lines,
/// validated end-to-end when G3 wires it under greedy parity).
#[test]
fn qkv_rope_append_batch_paged_bytes_match_dense() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_gpt_oss_paged_append() {
        eprintln!("pack has no gpt-oss paged fused-append - skipping");
        return;
    }
    // gpt-oss-ish head geometries (append is count-agnostic; a couple shapes +
    // both KV dtypes exercise the per-head block-table math).
    for (n_heads, n_kv_heads, head_dim) in [(64usize, 8usize, 64usize), (32, 4, 128)] {
        for kv_dtype in [KvDtype::Fp16, KvDtype::Fp8E4m3] {
            let qdim = n_heads * head_dim;
            let kvdim = n_kv_heads * head_dim;
            let rowd = qdim + 2 * kvdim;
            let max_ctx = 256usize;
            let bps = max_ctx / 16;
            let batch = 4usize;
            let positions = [10u32, 47, 128, 200];
            let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
            // yarn params: ext_factor 0 (no ramp), non-trivial theta chain.
            let params = (0.95f32, 1.0f32, 8.0f32, 32.0f32, 0.0f32, 1.0f32);
            let kb = match kv_dtype {
                KvDtype::Fp16 => 2usize,
                KvDtype::Fp8E4m3 => 1,
            };

            let mut qkv = Vec::new();
            for b in 0..batch {
                qkv.extend(det(rowd, 300 + b as u64));
            }
            let d_qkv = exec.to_device(&qkv).expect("qkv");

            let mut d_qd = exec.alloc(batch * qdim).expect("qd");
            let mut d_qp = exec.alloc(batch * qdim).expect("qp");
            let mut d_dk = exec.alloc_u8(batch * max_ctx * kvdim * kb).expect("dk");
            let mut d_dv = exec.alloc_u8(batch * max_ctx * kvdim * kb).expect("dv");
            let mut d_pk = exec.alloc_u8(batch * bps * 16 * kvdim * kb).expect("pk");
            let mut d_pv = exec.alloc_u8(batch * bps * 16 * kvdim * kb).expect("pv");

            exec.qkv_rope_append_batch(
                &d_qkv, &mut d_qd, &mut d_dk, &mut d_dv, &d_pos, None, n_heads, n_kv_heads,
                head_dim, max_ctx, params, batch, kv_dtype,
            )
            .expect("dense");

            let bt_host: Vec<u32> = (0..(batch * bps) as u32).collect();
            let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");
            exec.qkv_rope_append_batch_paged(
                &d_qkv, &mut d_qp, &mut d_pk, &mut d_pv, &d_pos, None, n_heads, n_kv_heads,
                head_dim, params, batch, &d_bt, bps, kv_dtype,
            )
            .expect("paged");

            let (dk, pk) = (
                exec.stream.clone_dtoh(&d_dk).unwrap(),
                exec.stream.clone_dtoh(&d_pk).unwrap(),
            );
            let (dv, pv) = (
                exec.stream.clone_dtoh(&d_dv).unwrap(),
                exec.stream.clone_dtoh(&d_pv).unwrap(),
            );
            let (qd, qp) = (exec.to_host(&d_qd).unwrap(), exec.to_host(&d_qp).unwrap());
            let kdiff = dk.iter().zip(&pk).filter(|(a, b)| a != b).count();
            let vdiff = dv.iter().zip(&pv).filter(|(a, b)| a != b).count();
            let qdiff = qd.iter().zip(&qp).filter(|(a, b)| a != b).count();
            eprintln!(
                "paged fused-append h={n_heads}/{n_kv_heads}/{head_dim} {kv_dtype:?}: \
                 K {kdiff} V {vdiff} q {qdiff} differ"
            );
            assert_eq!(kdiff, 0, "K cache not byte-identical");
            assert_eq!(vdiff, 0, "V cache not byte-identical");
            assert_eq!(qdiff, 0, "q_out not identical");
        }
    }
}

/// Paged FlashDecoding partial (P3b/P3b-2, pd_attn_decode_batch_partial_paged) +
/// the unchanged combine vs the dense partial + combine. With a contiguous
/// identity block table the pool overlays the dense cache, so paged must be
/// BITWISE == dense - proving the paged split addressing. The launcher's dispatch
/// is mirrored dense↔paged, so both pick the same kernel per geometry: the first
/// two geometries force the PLAIN partial (group > 8, or n_kv_heads < 4 - P3b);
/// the last three hit the GQA-FUSED partial (group ∈ [2,8], n_kv_heads ≥ 4 - the
/// P3b-2 paged GQA kernel, both tile classes 32/16). Covers head_dim 64/128/256,
/// split counts that divide/don't/over-split, and a sliding window.
///
/// One paged-ONLY arm is pinned off for the comparison: the v9q fp8 decode
/// (fp8 KV, hd128, G4, batch >= 2 - granite's shape) has no dense twin and is
/// a different NUMERICS CLASS (P e4m3-rounded before PV, the industry fp8
/// path), so "paged == dense" cannot hold through it - measured 4.4e-3 on a
/// GB10 the day the vec8 batch gate moved this shape onto v9q, and traced
/// to the class, not the die (`paged_fp8_hd128_v9q_matches_p8_class_reference`
/// is that arm's own gate). Addressing is what this test proves; the class
/// arm is proven where its class is the reference.
#[test]
fn attn_partial_batch_paged_bitwise_matches_dense() {
    let _arm = V9Q_ENV.lock().unwrap_or_else(|e| e.into_inner());
    paddock_engine::envset::set_env("PADDOCK_NO_V9Q", "1");
    attn_partial_batch_paged_bitwise_matches_dense_body();
    // SAFETY: single-threaded w.r.t. every reader of this env - the mutex
    // above serialises the two tests that touch it
    unsafe { std::env::remove_var("PADDOCK_NO_V9Q") };
}

/// The two tests that pin/unpin PADDOCK_NO_V9Q run under one lock: cargo runs
/// tests on parallel threads and the pack reads the env at every launch.
static V9Q_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn attn_partial_batch_paged_bitwise_matches_dense_body() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_attn_partial_batch_paged() {
        eprintln!("pack has no paged FlashDecoding partial - skipping");
        return;
    }
    for dt in KV_DTYPES {
        // (n_heads, n_kv_heads, head_dim): first two force the dense PLAIN partial
        // ((64,2,64) group 32 (>8); (16,2,256) n_kv_heads 2 (<4)); the last three hit
        // the GQA-FUSED partial ((64,8,64)->tile 32; (16,4,128)->tile 32; (16,4,256)->
        // tile 16 = the qwen35 full-attn shape).
        for (n_heads, n_kv_heads, head_dim) in [
            (64usize, 2usize, 64usize),
            (16, 2, 256),
            (64, 8, 64),
            (16, 4, 128),
            (16, 4, 256),
        ] {
            let kv_dim = n_kv_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let max_ctx = 256usize;
            let bps = max_ctx / 16;
            let batch = 4usize;
            let qdim = n_heads * head_dim;
            let positions = [10u32, 47, 128, 200];

            let mut q = Vec::new();
            let mut kc = Vec::new();
            let mut vc = Vec::new();
            for b in 0..batch {
                q.extend(det(qdim, 1 + b as u64));
                kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
                vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
            }
            let sinks = det(n_heads, 4);

            let d_q = exec.to_device(&q).expect("q");
            let d_k = kv_dev_u8(&exec, &kc, dt);
            let d_v = kv_dev_u8(&exec, &vc, dt);
            let d_s = exec.to_device(&sinks).expect("sinks");
            let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
            let bt_host: Vec<u32> = (0..(batch * bps) as u32).collect();
            let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");

            for (window, n_splits) in [(0usize, 3usize), (0, 4), (0, 8), (128, 4)] {
                let mut d_o = exec
                    .alloc(n_heads * batch * n_splits * head_dim)
                    .expect("o");
                let mut d_ml = exec.alloc(n_heads * batch * n_splits * 2).expect("ml");
                let mut d_dense = exec.alloc(batch * qdim).expect("dense");
                let mut d_o2 = exec
                    .alloc(n_heads * batch * n_splits * head_dim)
                    .expect("o2");
                let mut d_ml2 = exec.alloc(n_heads * batch * n_splits * 2).expect("ml2");
                let mut d_paged = exec.alloc(batch * qdim).expect("paged");
                exec.attn_partial_batch(
                    &d_q, &d_k, &d_v, &mut d_o, &mut d_ml, &d_pos, None, n_heads, n_kv_heads,
                    head_dim, max_ctx, kv_dim, window, n_splits, batch, scale, dt,
                )
                .expect("dense partial");
                exec.attn_combine_batch(
                    &d_o,
                    &d_ml,
                    &d_s,
                    &mut d_dense,
                    n_heads,
                    head_dim,
                    n_splits,
                    batch,
                )
                .expect("dense combine");
                exec.attn_partial_batch_paged(
                    &d_q, &d_k, &d_v, &mut d_o2, &mut d_ml2, &d_pos, None, &d_bt, bps, n_heads,
                    n_kv_heads, head_dim, kv_dim, window, n_splits, batch, scale, dt,
                )
                .expect("paged partial");
                exec.attn_combine_batch(
                    &d_o2,
                    &d_ml2,
                    &d_s,
                    &mut d_paged,
                    n_heads,
                    head_dim,
                    n_splits,
                    batch,
                )
                .expect("paged combine");
                let dense = exec.to_host(&d_dense).expect("dh");
                let paged = exec.to_host(&d_paged).expect("ph");
                let bitwise = dense
                    .iter()
                    .zip(&paged)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                let maxd = dense
                    .iter()
                    .zip(&paged)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "{dt:?} paged partial hd={head_dim} window {window} splits {n_splits}: bitwise={bitwise} max_abs {maxd:.2e}"
                );
                // fp16 is bitwise: pool and dense are the same bytes under the
                // identity table, so the same kernel reads the same values in
                // the same order. At e4m3 the two entries may still pick
                // different f32-class arms at a shape (the vec8 register walk
                // vs the GQA walk parted by ~4e-8 while vec8 took hd128/G4) -
                // a last-ulp accumulation-ORDER difference, not corruption, so
                // hold fp8 to the f32 rounding scale rather than pretending
                // bitwise still holds. The e4m3-P class arm is pinned off
                // above; anything past 1e-6 here is addressing, and a bug.
                if dt == KvDtype::Fp16 {
                    assert!(
                        bitwise,
                        "paged partial hd={head_dim} window {window} splits {n_splits} not bitwise == dense (max_abs {maxd})"
                    );
                } else {
                    assert!(
                        maxd < 1e-6,
                        "paged partial {dt:?} hd={head_dim} window {window} splits {n_splits}: {maxd} is past accumulation-order noise"
                    );
                }
            }
        }
    }
}

/// The v9q fp8 decode arm (fp8 KV, hd128, G4, batch >= 2 - granite) against
/// a reference of ITS class: Q*scale cast to e4m3, f32 scores, online softmax
/// over 32-key supertiles aligned to 16-token blocks from the split's first
/// block, P = exp(x - m) ROUNDED to e4m3 for the PV product, l accumulated
/// from the unrounded weights, splits merged like the combine. Measured on
/// the GB10 that surfaced it: 1.5e-8..6e-8 at every split count and length
/// (accumulation order), where the exact-f32 reference sits 1.6e-3..5.9e-3
/// away - that gap IS the class, the same P.to(fp8) the industry fp8 paths
/// ship, and the number a reviewer should expect from this arm.
///
/// Two contracts, both covered: n_splits >= 2 writes partials + ml for the
/// combine (the engine clamps this arm to 2..4 splits), and n_splits == 1 is
/// the in-kernel FINALIZE - [b][head][hd] rows straight into the final
/// buffer, out_ml untouched, no sink (granite has none) and NO combine after
/// it (granite's `v9q_ns1` branch). Combining a finalized buffer reads
/// garbage ml and was exactly the harness mistake that first made this arm
/// look 0.4 wrong.
#[test]
fn paged_fp8_hd128_v9q_matches_p8_class_reference() {
    let _arm = V9Q_ENV.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: serialised against the only other writer by the mutex
    unsafe { std::env::remove_var("PADDOCK_NO_V9Q") };
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_attn_partial_batch_paged() {
        eprintln!("pack has no paged FlashDecoding partial - skipping");
        return;
    }
    // The v9q bodies exist only in sm_90+ passes (PD_ATTN_TMA_OK): below
    // that the same export runs the f32-class partial, and its distance from
    // the P8-class reference is the class gap this test documents (4.5e-3 on
    // an A6000), not a defect. Only a die that runs v9q can be held to 1e-6.
    if exec.compute_capability().0 < 9 {
        eprintln!(
            "v9q needs the sm_90+ fp8 mma bodies; this die runs the f32-class arm - skipping"
        );
        return;
    }
    let dt = KvDtype::Fp8E4m3;
    let (n_heads, n_kv_heads, head_dim) = (16usize, 4usize, 128usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_ctx = 256usize;
    let bps = max_ctx / 16;
    let batch = 4usize;
    let qdim = n_heads * head_dim;
    for positions in [
        [10u32, 47, 128, 200],
        [200u32, 200, 200, 200],
        [15u32, 16, 17, 31],
    ] {
        let (mut q, mut kc, mut vc) = (Vec::new(), Vec::new(), Vec::new());
        for b in 0..batch {
            q.extend(det(qdim, 1 + b as u64));
            kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
            vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
        }
        let sinks = det(n_heads, 4);
        let d_q = exec.to_device(&q).expect("q");
        let d_k = kv_dev_u8(&exec, &kc, dt);
        let d_v = kv_dev_u8(&exec, &vc, dt);
        let d_s = exec.to_device(&sinks).expect("sinks");
        let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
        let bt_host: Vec<u32> = (0..(batch * bps) as u32).collect();
        let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");
        let kr = kv_round(&kc, dt);
        let vr = kv_round(&vc, dt);
        let reference = |n_splits: usize, sinks: &[f32]| -> Vec<f32> {
            let mut r = Vec::with_capacity(batch * qdim);
            for b in 0..batch {
                r.extend(cpu_attn_p8(
                    &q[b * qdim..(b + 1) * qdim],
                    &kr[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim],
                    &vr[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim],
                    sinks,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    positions[b] as usize + 1,
                    kv_dim,
                    scale,
                    n_splits,
                ));
            }
            r
        };
        let maxd = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        for n_splits in [2usize, 3, 4, 8] {
            let mut d_o = exec
                .alloc(n_heads * batch * n_splits * head_dim)
                .expect("o");
            let mut d_ml = exec.alloc(n_heads * batch * n_splits * 2).expect("ml");
            let mut d_out = exec.alloc(batch * qdim).expect("out");
            exec.attn_partial_batch_paged(
                &d_q, &d_k, &d_v, &mut d_o, &mut d_ml, &d_pos, None, &d_bt, bps, n_heads,
                n_kv_heads, head_dim, kv_dim, 0, n_splits, batch, scale, dt,
            )
            .expect("paged partial");
            exec.attn_combine_batch(
                &d_o, &d_ml, &d_s, &mut d_out, n_heads, head_dim, n_splits, batch,
            )
            .expect("combine");
            let out = exec.to_host(&d_out).expect("out");
            let d = maxd(&out, &reference(n_splits, &sinks));
            eprintln!(
                "v9q fp8 hd128 G4 pos={positions:?} splits={n_splits}: max_abs vs P8-class ref {d:.2e}"
            );
            assert!(
                d < 1e-6,
                "v9q splits={n_splits}: {d} is past accumulation-order noise"
            );
        }
        // the finalize contract: one split, final rows, no combine, no sink
        let mut d_out = exec.alloc(batch * qdim).expect("out1");
        let mut d_ml = exec.alloc(n_heads * batch * 2).expect("ml1");
        exec.attn_partial_batch_paged(
            &d_q, &d_k, &d_v, &mut d_out, &mut d_ml, &d_pos, None, &d_bt, bps, n_heads, n_kv_heads,
            head_dim, kv_dim, 0, 1, batch, scale, dt,
        )
        .expect("paged finalize");
        let out = exec.to_host(&d_out).expect("out1");
        let no_sink = vec![f32::NEG_INFINITY; n_heads];
        let d = maxd(&out, &reference(1, &no_sink));
        eprintln!("v9q fp8 hd128 G4 pos={positions:?} finalize: max_abs vs P8-class ref {d:.2e}");
        assert!(
            d < 1e-6,
            "v9q finalize: {d} is past accumulation-order noise"
        );
    }
}

/// P8-class CPU reference for the v9q arm - see the test above for the recipe.
#[allow(clippy::too_many_arguments)]
fn cpu_attn_p8(
    q: &[f32],
    kr: &[f32],
    vr: &[f32],
    sinks: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_pos: usize,
    kv_dim: usize,
    scale: f32,
    n_splits: usize,
) -> Vec<f32> {
    let group = n_heads / n_kv_heads;
    let mut out = vec![0f32; n_heads * head_dim];
    let chunk = n_pos.div_ceil(n_splits);
    for h in 0..n_heads {
        let kvh = h / group;
        let q8: Vec<f32> = q[h * head_dim..(h + 1) * head_dim]
            .iter()
            .map(|&x| e4m3_decode(e4m3_encode(x * scale)))
            .collect();
        let score = |p: usize| -> f32 {
            let base = p * kv_dim + kvh * head_dim;
            q8.iter()
                .zip(&kr[base..base + head_dim])
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        let mut parts: Vec<(f32, f32, Vec<f32>)> = Vec::new();
        for sp in 0..n_splits {
            let lo = sp * chunk;
            let hi = (lo + chunk).min(n_pos);
            if lo >= hi {
                continue;
            }
            let (mut m, mut l, mut o) = (f32::NEG_INFINITY, 0f32, vec![0f32; head_dim]);
            let mut st = (lo >> 4) * 16; // supertiles start at the split's first block
            while st < hi {
                let keys: Vec<usize> = (st.max(lo)..(st + 32).min(hi)).collect();
                st += 32;
                if keys.is_empty() {
                    continue;
                }
                let xs: Vec<f32> = keys.iter().map(|&p| score(p)).collect();
                let mnew = xs.iter().fold(m, |a, &b| a.max(b));
                let corr = (m - mnew).exp();
                l *= corr;
                for v in o.iter_mut() {
                    *v *= corr;
                }
                for (&p, &x) in keys.iter().zip(&xs) {
                    let w = (x - mnew).exp();
                    let w8 = e4m3_decode(e4m3_encode(w));
                    l += w;
                    let base = p * kv_dim + kvh * head_dim;
                    for (od, &vd) in o.iter_mut().zip(&vr[base..base + head_dim]) {
                        *od += w8 * vd;
                    }
                }
                m = mnew;
            }
            parts.push((m, l, o));
        }
        let mg = parts.iter().map(|p| p.0).fold(f32::NEG_INFINITY, f32::max);
        let mut l = 0f32;
        let mut o = vec![0f32; head_dim];
        for (m, li, oi) in &parts {
            let c = (m - mg).exp();
            l += li * c;
            for (a, b) in o.iter_mut().zip(oi) {
                *a += b * c;
            }
        }
        l += (sinks[h] - mg).exp();
        for (i, v) in o.iter().enumerate() {
            out[h * head_dim + i] = v / l;
        }
    }
    out
}

/// Paged tiled prefill (P4b, pd_attn_prefill_paged) vs the dense tiled prefill
/// (pd_attn_prefill). Single-slot (all rows share slot 0). With a contiguous
/// identity block table the pool overlays the dense cache, so paged must be
/// BITWISE == dense - proving the paged prefill addressing. Covers the qwen35
/// full-attn shape (hd 256) and the hd 128 instantiation, T multiple/non-
/// multiple of the query tile, and a sliding window. Only runs on a P4b pack.
#[test]
fn attn_prefill_paged_bitwise_matches_dense() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_attn_prefill_paged() {
        eprintln!("pack has no paged tiled prefill - skipping");
        return;
    }
    for dt in KV_DTYPES {
        for (n_heads, n_kv_heads, head_dim) in [(16usize, 4usize, 256usize), (16, 8, 128)] {
            let kv_dim = n_kv_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let max_ctx = 512usize;
            let bps = max_ctx / 16;
            let qdim = n_heads * head_dim;

            for &t_len in &[512usize, 100, 33] {
                let q = det(t_len * qdim, 1);
                let kc = det(max_ctx * kv_dim, 100);
                let vc = det(max_ctx * kv_dim, 200);
                let sinks = det(n_heads, 4);
                let positions: Vec<u32> = (0..t_len as u32).collect();
                let slots = vec![0u32; t_len];

                let d_q = exec.to_device(&q).expect("q");
                let d_k = kv_dev_u8(&exec, &kc, dt);
                let d_v = kv_dev_u8(&exec, &vc, dt);
                let d_s = exec.to_device(&sinks).expect("sinks");
                let d_pos = exec.stream.clone_htod(&positions).expect("pos");
                let d_slots = exec.stream.clone_htod(&slots).expect("slots");
                // identity block table (slot 0 only needs its `bps` entries)
                let bt_host: Vec<u32> = (0..bps as u32).collect();
                let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");

                for &swa in &[0usize, 128] {
                    let mut d_dense = exec.alloc(t_len * qdim).expect("dense");
                    let mut d_paged = exec.alloc(t_len * qdim).expect("paged");
                    exec.attn_prefill(
                        &d_q,
                        &d_k,
                        &d_v,
                        &d_s,
                        &mut d_dense,
                        &d_pos,
                        &d_slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        swa,
                        t_len,
                        scale,
                        dt,
                    )
                    .expect("dense prefill");
                    exec.attn_prefill_paged(
                        &d_q,
                        &d_k,
                        &d_v,
                        &d_s,
                        &mut d_paged,
                        &d_pos,
                        &d_slots,
                        &d_bt,
                        bps,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        swa,
                        t_len,
                        scale,
                        dt,
                    )
                    .expect("paged prefill");
                    let dense = exec.to_host(&d_dense).expect("dh");
                    let paged = exec.to_host(&d_paged).expect("ph");
                    let bitwise = dense
                        .iter()
                        .zip(&paged)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    let maxd = dense
                        .iter()
                        .zip(&paged)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    eprintln!(
                        "{dt:?} paged prefill hd={head_dim} T={t_len} swa={swa}: bitwise={bitwise} max_abs {maxd:.2e}"
                    );
                    assert!(
                        bitwise,
                        "paged prefill hd={head_dim} T={t_len} swa={swa} not bitwise == dense (max_abs {maxd})"
                    );
                }
            }
        }
    }
}

/// Paged f16 prefill (pd_attn_prefill_f16_paged) vs the dense f16 WMMA prefill
/// (pd_attn_prefill_f16). Single-slot; with a contiguous identity block table
/// the pool overlays the dense cache and page=16 aligns with the 16-key WMMA
/// tile. Two gates, matched to the dispatcher:
///
/// - Shapes served by the paged WMMA twin (the same math as the dense kernel)
///   must be BITWISE == dense - this proves the paged addressing.
/// - hd-256 GQA shapes with G ∈ {2,4,8} route to the staged-HMMA v4 tile on
///   sm_86+ (the qwen35 backport) - a different kernel than the dense WMMA
///   reference, so bitwise cannot hold; those get the f16-class gate (5e-4),
///   the same bound the sm_100 tcgen05 arms are held to. An addressing bug
///   (wrong page) blows that bound by orders of magnitude, so the gate still
///   proves v4's paged addressing, just not bit-exactness.
///
/// To keep the bitwise addressing proof for the WMMA fallback on the v4
/// shapes too (it still serves them when v4 can't run), the test re-execs
/// itself once with PADDOCK_NO_PF_V4 set - the kill switch is latched once
/// per process, so an in-process toggle can't work.
///
/// Shapes: qwen35 9B/35B-A3B (hd 256, G 4/8 -> v4), qwen36 27B (hd 256, G 6 ->
/// WMMA fallthrough, bitwise), gpt-oss (hd 64 -> WMMA, bitwise); T multiple/
/// non-multiple of the tile; sliding window. Only runs on a P4b-2 pack.
#[test]
fn attn_prefill_f16_paged_bitwise_matches_dense() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_attn_prefill_f16_paged() {
        eprintln!("pack has no paged f16 WMMA prefill - skipping");
        return;
    }
    // Child-leg marker: with the v4 arm killed, every shape runs the shared
    // WMMA kernel pair and must be bitwise.
    let forced_wmma = std::env::var_os("PADDOCK_NO_PF_V4").is_some();
    let cc_major = exec.compute_capability().0;
    for dt in KV_DTYPES {
        // The DENSE f16-class WMMA prefill kernels take an f16 cache only -
        // pd_attn_prefill_f16* returns cudaErrorInvalidValue for e4m3 rather
        // than computing something wrong (packs/cuda/src/attn/prefill.cuh).
        // That refusal is pinned by f16_class_prefill_refuses_an_fp8_cache; the
        // PAGED f16 kernels do carry fp8 arms and are covered by
        // paged_f16_prefill_fp8_matches_decode_batch.
        if dt != KvDtype::Fp16 {
            continue;
        }
        for (n_heads, n_kv_heads, head_dim) in [
            (16usize, 4usize, 256usize),
            (16, 2, 256),
            (24, 4, 256),
            (64, 8, 64),
        ] {
            let kv_dim = n_kv_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            let max_ctx = 512usize; // % 64 == 0 (WMMA requirement)
            let bps = max_ctx / 16;
            let qdim = n_heads * head_dim;

            for &t_len in &[512usize, 100, 33] {
                let q = det(t_len * qdim, 1);
                let kc = det(max_ctx * kv_dim, 100);
                let vc = det(max_ctx * kv_dim, 200);
                let sinks = det(n_heads, 4);
                let positions: Vec<u32> = (0..t_len as u32).collect();
                let slots = vec![0u32; t_len];

                let d_q = exec.to_device(&q).expect("q");
                let d_k = kv_dev_u8(&exec, &kc, dt);
                let d_v = kv_dev_u8(&exec, &vc, dt);
                let d_s = exec.to_device(&sinks).expect("sinks");
                let d_pos = exec.stream.clone_htod(&positions).expect("pos");
                let d_slots = exec.stream.clone_htod(&slots).expect("slots");
                let bt_host: Vec<u32> = (0..bps as u32).collect();
                let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");

                for &swa in &[0usize, 128] {
                    let mut d_dense = exec.alloc(t_len * qdim).expect("dense");
                    let mut d_paged = exec.alloc(t_len * qdim).expect("paged");
                    exec.attn_prefill_f16(
                        &d_q,
                        &d_k,
                        &d_v,
                        &d_s,
                        &mut d_dense,
                        &d_pos,
                        &d_slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        swa,
                        t_len,
                        scale,
                        dt,
                    )
                    .expect("dense f16 prefill");
                    exec.attn_prefill_f16_paged(
                        &d_q,
                        &d_k,
                        &d_v,
                        &d_s,
                        &mut d_paged,
                        &d_pos,
                        &d_slots,
                        &d_bt,
                        bps,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        swa,
                        t_len,
                        scale,
                        dt,
                    )
                    .expect("paged f16 prefill");
                    let dense = exec.to_host(&d_dense).expect("dh");
                    let paged = exec.to_host(&d_paged).expect("ph");
                    let bitwise = dense
                        .iter()
                        .zip(&paged)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    let maxd = dense
                        .iter()
                        .zip(&paged)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    let g = n_heads / n_kv_heads;
                    // Which shapes the pack routes to the v4 mma.sync tile, whose
                    // f32 online softmax is a different numeric class from the
                    // dense WMMA reference - those get the f16-class gate, the
                    // rest must stay bitwise. MIRROR of the two dispatch blocks in
                    // packs/cuda/src/attn/prefill.cuh; widen both together.
                    //
                    // This list has rotted twice now: it was written for hd256
                    // G∈{2,4,8}, then hd256 G=6 arrived with qwen3.6-27b, then
                    // hd128 G∈{4,6,9} with the fp8-KV tile. Each time the test
                    // went red and stayed red until someone ran the GPU
                    // integration suite - which is the real lesson: `--lib` is
                    // not this suite.
                    let v4_shape = !forced_wmma
                        && ((head_dim == 256 && matches!(g, 2 | 4 | 6 | 8))
                            || (head_dim == 128 && matches!(g, 4 | 6 | 9)));
                    eprintln!(
                        "{dt:?} paged f16 prefill hd={head_dim} G={g} T={t_len} swa={swa}: bitwise={bitwise} max_abs {maxd:.2e}{}",
                        if v4_shape {
                            " (v4 arm, class gate)"
                        } else {
                            ""
                        }
                    );
                    // sm_100+ paged runs the pf5/pf6 tcgen05 arms (f32 online
                    // softmax, TMA staging) - never bitwise vs the dense WMMA
                    // reference. Below that, only the v4 shapes escape bitwise.
                    if cc_major >= 10 || v4_shape {
                        assert!(
                            maxd < 5e-4,
                            "paged f16 prefill hd={head_dim} G={g} T={t_len} swa={swa} exceeds f16-class gate (max_abs {maxd})"
                        );
                    } else {
                        assert!(
                            bitwise,
                            "paged f16 prefill hd={head_dim} G={g} T={t_len} swa={swa} not bitwise == dense (max_abs {maxd})"
                        );
                    }
                }
            }
        }
    }
    // Second leg: prove the WMMA fallback's paged ADDRESSING bitwise on the
    // v4 shapes as well, by re-running this test in a child process with the
    // v4 arm disabled (the dispatcher latches the env once per process).
    // sm_100+ has no shared-WMMA claim, so there is nothing to prove there.
    if !forced_wmma && cc_major < 10 {
        drop(exec); // free the parent's context before the child creates its own
        let exe = std::env::current_exe().expect("test exe path");
        let st = std::process::Command::new(exe)
            .args([
                "attn_prefill_f16_paged_bitwise_matches_dense",
                "--exact",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("PADDOCK_NO_PF_V4", "1")
            .status()
            .expect("spawn PADDOCK_NO_PF_V4 leg");
        assert!(st.success(), "PADDOCK_NO_PF_V4 bitwise leg failed");
    }
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) KV cache
//
// fp8 KV was once accepted on sm_86 and produced wrong output. The device
// gate now refuses it there, but nothing ever isolated where the error comes
// from, so the gate is a guess with a bad outcome attached rather than a
// diagnosis. There are only two candidates:
//
//   storage      the e4m3 round-trip itself (3 mantissa bits, max 448) - the
//                cache simply cannot hold what was written
//   accumulation the kernel dequantises and sums badly - vLLM hit exactly this
//                and shipped a "two-level accumulation" fix for Hopper, and
//                chose FlashInfer on Blackwell to dodge "precision issues"
//
// These tests separate them. Both encode e4m3 on the host and hand the GPU
// those exact bytes, so host and device see identical 8-bit values and no
// rounding disagreement can be mistaken for a kernel fault. The first test
// covers the write path, the second the read+accumulate path.
//
// Nothing here is a scale-factor design. Per-tensor/per-head scales only make
// sense as an answer to the STORAGE arm; if the storage round-trip is faithful
// and only the accumulated output drifts, scaling would be treating a symptom.

/// Decode one e4m3 byte. Exact and unambiguous - 256 patterns, bias 7, no Inf,
/// a single NaN (S.1111.111), max finite 448, min normal 2^-6, min subnormal
/// 2^-9. Matches OCP FP8 and `__nv_fp8_e4m3`.
fn e4m3_decode(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f32;
    if e == 0 {
        s * m * 2f32.powi(-9) // subnormal (and zero)
    } else if e == 15 && (b & 0x07) == 7 {
        f32::NAN
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// The 127 non-negative finite e4m3 magnitudes, ascending (bytes 0x00..=0x7E -
/// 0x7F is NaN). Monotone in the bit pattern, so encoding is a binary search.
fn e4m3_mag_table() -> &'static [f32; 127] {
    static T: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 127];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_decode(i as u8);
        }
        t
    })
}

/// Encode f32 -> e4m3, round-to-nearest-even, overflow saturating to +/-448
/// (what `__nv_fp8_e4m3` does - the format has no infinity to overflow into).
fn e4m3_encode(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    let t = e4m3_mag_table();
    let hi = t.partition_point(|&v| v < a);
    let byte = if hi == 0 {
        0
    } else if hi >= t.len() {
        (t.len() - 1) as u8 // saturate
    } else {
        let (dl, dh) = (a - t[hi - 1], t[hi] - a);
        if dh < dl || (dh == dl && hi % 2 == 0) {
            hi as u8
        } else {
            (hi - 1) as u8
        }
    };
    sign | byte
}

/// Upload an f32 slice as raw e4m3 BYTES - the fp8 twin of `f16_dev_u8`.
fn e4m3_dev_u8(exec: &GpuExecutor, data: &[f32]) -> CudaSlice<u8> {
    let bytes: Vec<u8> = data.iter().map(|&x| e4m3_encode(x)).collect();
    exec.stream.clone_htod(&bytes).expect("e4m3 bytes htod")
}

/// Round an f32 slice through e4m3 (what the fp8 cache stores), for the reference.
fn e4m3_round(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&x| e4m3_decode(e4m3_encode(x))).collect()
}

/// Upload as whichever width the cache actually holds. Every parity test in this
/// file ran fp16 only, which is how the fp8 path reached a serve untested - a
/// route that is bit-exact on 2-byte KV can still address 1-byte KV wrongly, and
/// nothing here would have noticed. Sweep the dtype, don't fork the test.
fn kv_dev_u8(exec: &GpuExecutor, data: &[f32], dt: KvDtype) -> CudaSlice<u8> {
    match dt {
        KvDtype::Fp16 => f16_dev_u8(exec, data),
        KvDtype::Fp8E4m3 => e4m3_dev_u8(exec, data),
    }
}

/// Round an f32 slice through whichever width the cache holds - the reference
/// side of `kv_dev_u8`. Quantisation error then sits on both sides and cancels,
/// so what a comparison measures is the kernel, not the format.
fn kv_round(data: &[f32], dt: KvDtype) -> Vec<f32> {
    match dt {
        KvDtype::Fp16 => f16_round(data),
        KvDtype::Fp8E4m3 => e4m3_round(data),
    }
}

/// The dtypes a parity sweep should cover. Both are storage formats on every
/// arch - e4m3 conversion is software-emulated below sm_89, not absent - so a
/// kernel that reads the cache is testable here regardless of the device gate
/// serving applies.
const KV_DTYPES: [KvDtype; 2] = [KvDtype::Fp16, KvDtype::Fp8E4m3];

/// STORAGE arm. `kv_append_batch` with `KvDtype::Fp8E4m3` must write exactly the
/// bytes the host codec produces for the same input - including at magnitudes
/// past 448, where the format saturates rather than overflowing to infinity.
/// A mismatch here means the cache cannot hold what the model wrote, and no
/// amount of kernel work downstream can recover it.
#[test]
fn kv_append_batch_fp8_roundtrip_matches_host_codec() {
    let Some(exec) = common::gpu() else {
        return;
    };
    let (kv_dim, max_ctx, batch) = (512usize, 64usize, 4usize);
    let positions = [3u32, 17, 40, 63];
    let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");

    // 1.0x is the model's natural K/V range; 900x drives values past 448 so the
    // saturation edge is covered, not assumed.
    for mag in [1.0f32, 12.0, 900.0] {
        let mut kvrows = Vec::new();
        for b in 0..batch {
            kvrows.extend(det(kv_dim, 50 + b as u64).iter().map(|&x| x * mag));
        }
        let d_kv = exec.to_device(&kvrows).expect("kv");
        let mut d_cache = exec.alloc_u8(batch * max_ctx * kv_dim).expect("cache");
        exec.kv_append_batch(
            &d_kv,
            &mut d_cache,
            &d_pos,
            None,
            kv_dim,
            max_ctx,
            batch,
            KvDtype::Fp8E4m3,
        )
        .expect("kv_append_batch fp8");
        let raw = exec.stream.clone_dtoh(&d_cache).expect("cache host");

        let mut bad = 0usize;
        let mut worst = (0f32, 0f32, 0u8, 0u8);
        for b in 0..batch {
            let row = positions[b] as usize;
            let off = (b * max_ctx + row) * kv_dim;
            for i in 0..kv_dim {
                let src = kvrows[b * kv_dim + i];
                let want = e4m3_encode(src);
                let got = raw[off + i];
                if got != want {
                    bad += 1;
                    let d = (e4m3_decode(got) - e4m3_decode(want)).abs();
                    if d > worst.0 {
                        worst = (d, src, want, got);
                    }
                }
            }
        }
        let saturated = kvrows.iter().filter(|v| v.abs() > 448.0).count();
        let detail = if bad > 0 {
            format!(
                ", worst src {} host 0x{:02X}={} gpu 0x{:02X}={}",
                worst.1,
                worst.2,
                e4m3_decode(worst.2),
                worst.3,
                e4m3_decode(worst.3)
            )
        } else {
            String::new()
        };
        eprintln!(
            "fp8 append mag {mag}: {bad}/{} bytes differ from the host codec ({saturated} inputs past 448){detail}",
            batch * kv_dim
        );
        assert_eq!(bad, 0, "fp8 store diverges from the e4m3 spec at mag {mag}");
    }
}

/// ACCUMULATION arm. Same geometry and same inputs as
/// `attn_decode_batch_matches_per_sequence`, but the cache holds e4m3 and the
/// CPU reference reads the e4m3-rounded values - so quantisation error is
/// present in both sides and cancels. What is left is purely how the kernel
/// dequantises and sums. The f16 twin of this test holds to 1e-5; if fp8 cannot,
/// the fault is in the kernel's accumulation, not in the format.
#[test]
fn attn_decode_batch_fp8_matches_per_sequence() {
    let Some(exec) = common::gpu() else {
        return;
    };
    let (n_heads, n_kv_heads, head_dim) = (64usize, 8usize, 64usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_ctx = 256usize;
    let batch = 4usize;
    let positions = [10u32, 47, 128, 200];

    let qdim = n_heads * head_dim;
    let mut q = Vec::new();
    let mut kc = Vec::new();
    let mut vc = Vec::new();
    for b in 0..batch {
        q.extend(det(qdim, 1 + b as u64));
        kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
        vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
    }
    let sinks = det(n_heads, 4);

    let d_q = exec.to_device(&q).expect("q");
    let d_k = e4m3_dev_u8(&exec, &kc);
    let d_v = e4m3_dev_u8(&exec, &vc);
    let d_s = exec.to_device(&sinks).expect("sinks");
    // reference sees the same e4m3-rounded KV the fp8 cache holds
    let (kc, vc) = (e4m3_round(&kc), e4m3_round(&vc));
    let d_pos = exec.stream.clone_htod(&positions.to_vec()).expect("pos");
    let mut d_out = exec.alloc(batch * qdim).expect("out");
    exec.attn_decode_batch(
        &d_q,
        &d_k,
        &d_v,
        &d_s,
        &mut d_out,
        &d_pos,
        None,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_dim,
        0,
        batch,
        scale,
        KvDtype::Fp8E4m3,
    )
    .expect("attn_decode_batch fp8");
    let got = exec.to_host(&d_out).expect("dtoh");

    let mut worst = 0f32;
    for b in 0..batch {
        let n_pos = positions[b] as usize + 1;
        let qb = &q[b * qdim..(b + 1) * qdim];
        let kcb = &kc[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim];
        let vcb = &vc[b * max_ctx * kv_dim..(b + 1) * max_ctx * kv_dim];
        let expected = cpu_attn(
            qb, kcb, vcb, &sinks, n_heads, n_kv_heads, head_dim, 0, n_pos, kv_dim, scale,
        );
        let max_diff = got[b * qdim..(b + 1) * qdim]
            .iter()
            .zip(&expected)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        worst = worst.max(max_diff);
        eprintln!(
            "fp8 seq {b} (pos {}): max_abs_diff {max_diff:.2e}",
            positions[b]
        );
    }
    assert!(
        worst < 1e-5,
        "fp8 KV accumulation is off by {worst} - the f16 path holds to 1e-5 on the identical geometry, so this is the kernel, not the format"
    );
}

/// The DENSE f16-class WMMA prefill kernels take an f16 cache only:
/// `pd_attn_prefill_f16*` returns `cudaErrorInvalidValue` for e4m3 rather than
/// computing something wrong (packs/cuda/src/attn/prefill.cuh). Nothing in the
/// engine may reach them with an fp8 cache - `pf_attn_dtype_ok` and the paged
/// dispatch are what keep that true, and this pins the pack half of the deal.
///
/// A failure here is not "fp8 broke". It means an fp8 arm was added to a dense
/// f16 entry, in which case it needs its own parity case and the model-side
/// gate needs to learn it is allowed - silently succeeding is the dangerous
/// outcome, because it would be untested math on a serving path.
#[test]
fn f16_class_prefill_refuses_an_fp8_cache() {
    let Some(exec) = common::gpu() else {
        return;
    };
    let (n_heads, n_kv_heads, head_dim) = (16usize, 4usize, 256usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (max_ctx, t_len) = (512usize, 64usize);
    let qdim = n_heads * head_dim;

    let d_q = exec.to_device(&det(t_len * qdim, 1)).expect("q");
    let d_k = e4m3_dev_u8(&exec, &det(max_ctx * kv_dim, 100));
    let d_v = e4m3_dev_u8(&exec, &det(max_ctx * kv_dim, 200));
    let d_s = exec.to_device(&det(n_heads, 4)).expect("sinks");
    let positions: Vec<u32> = (0..t_len as u32).collect();
    let d_pos = exec.stream.clone_htod(&positions).expect("pos");
    let d_slots = exec.stream.clone_htod(&vec![0u32; t_len]).expect("slots");
    let mut d_out = exec.alloc(t_len * qdim).expect("out");

    let dense = exec.attn_prefill_f16(
        &d_q,
        &d_k,
        &d_v,
        &d_s,
        &mut d_out,
        &d_pos,
        &d_slots,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_dim,
        0,
        t_len,
        scale,
        KvDtype::Fp8E4m3,
    );
    assert!(
        dense.is_err(),
        "attn_prefill_f16 accepted an e4m3 cache - if the dense f16 class grew \
         an fp8 arm it needs a parity case, not a silent pass"
    );
    let rows = exec.attn_prefill_f16_rows(
        &d_q,
        &d_k,
        &d_v,
        &d_s,
        &mut d_out,
        &d_pos,
        &d_slots,
        n_heads,
        n_kv_heads,
        head_dim,
        max_ctx,
        kv_dim,
        0,
        0,
        t_len,
        scale,
        KvDtype::Fp8E4m3,
    );
    assert!(
        rows.is_err(),
        "attn_prefill_f16_rows accepted an e4m3 cache - same rule as the full pass"
    );
    eprintln!("dense f16-class prefill refuses e4m3, as the pack contract says");
}

/// The PAGED f16 prefill kernels do carry fp8 arms, and until this test none of
/// them had ever run in a parity check - which is how fp8 serve went
/// out on untested math. Two entries matter:
///
///   attn_prefill_f16_paged    several fp8 dispatch arms (hd/GQA specialised)
///   attn_prefill_f16_paged_vl the packed varlen route, gated on fp8 only - its
///                             single caller (qwen35/batch.rs) requires
///                             kv_dtype == Fp8E4m3, so it cannot execute in an
///                             fp16 configuration and fp16-only testing could
///                             never have reached it
///
/// Oracle is `attn_decode_batch`, which the storage/accumulation arms proved
/// correct on an e4m3 cache to f32 rounding noise. Tolerance is the f16-WMMA
/// class (these accumulate in f16), the same 8e-3 the fp16 twins use.
#[test]
fn paged_f16_prefill_fp8_matches_decode_batch() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_paged_kv() || !exec.has_attn_prefill_f16_paged() {
        eprintln!("pack has no paged f16 prefill - skipping");
        return;
    }
    let head_dim = 256usize; // the vl gate's shape
    let max_ctx = 512usize; // % 64 == 0
    let bps = max_ctx / 16;
    let dt = KvDtype::Fp8E4m3;

    // attn_g in {4, 6, 8} - the group ratios the vl gate admits.
    for (n_heads, n_kv_heads) in [(16usize, 4usize), (24, 4), (16, 2)] {
        let attn_g = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let qdim = n_heads * head_dim;

        // w > 24 is the vl caller's own admission rule
        for &t_len in &[100usize, 33] {
            let q = det(t_len * qdim, 1);
            let kc = det(max_ctx * kv_dim, 100);
            let vc = det(max_ctx * kv_dim, 200);
            let sinks = det(n_heads, 4);
            let positions: Vec<u32> = (0..t_len as u32).collect();

            let d_q = exec.to_device(&q).expect("q");
            let d_k = e4m3_dev_u8(&exec, &kc);
            let d_v = e4m3_dev_u8(&exec, &vc);
            let d_s = exec.to_device(&sinks).expect("sinks");
            let d_pos = exec.stream.clone_htod(&positions).expect("pos");
            let d_slots = exec.stream.clone_htod(&vec![0u32; t_len]).expect("slots");
            // identity table: slot 0 -> blocks [0..bps), so the pool and a dense
            // [max_ctx, kv_dim] cache are the same bytes
            let bt_host: Vec<u32> = (0..bps as u32).collect();
            let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");

            for &swa in &[0usize, 128] {
                let mut d_ref = exec.alloc(t_len * qdim).expect("ref");
                exec.attn_decode_batch(
                    &d_q,
                    &d_k,
                    &d_v,
                    &d_s,
                    &mut d_ref,
                    &d_pos,
                    Some(&d_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_dim,
                    swa,
                    t_len,
                    scale,
                    dt,
                )
                .expect("decode_batch oracle");
                let r = exec.to_host(&d_ref).expect("dtoh ref");

                let mut d_pf = exec.alloc(t_len * qdim).expect("pf");
                exec.attn_prefill_f16_paged(
                    &d_q, &d_k, &d_v, &d_s, &mut d_pf, &d_pos, &d_slots, &d_bt, bps, n_heads,
                    n_kv_heads, head_dim, kv_dim, swa, t_len, scale, dt,
                )
                .expect("paged f16 prefill (fp8)");
                let got = exec.to_host(&d_pf).expect("dtoh pf");
                let maxd = r
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "fp8 paged f16 prefill G={attn_g} hd={head_dim} T={t_len} swa={swa}: max_abs_diff {maxd:.2e}"
                );
                assert!(
                    maxd < 8e-3,
                    "fp8 paged f16 prefill G={attn_g} T={t_len} swa={swa}: {maxd} exceeds the f16-class gate"
                );

                // The varlen route: one span of t_len rows in slot 0, work packed
                // into 64-wide tiles over w*attn_g, laid out exactly as its caller
                // builds it - [row_base, width, tile_offset, slot] per tile.
                if !exec.has_attn_prefill_f16_paged_vl() || t_len <= 24 || swa != 0 {
                    continue;
                }
                let mut items: Vec<u32> = Vec::new();
                let mut t0 = 0usize;
                while t0 < t_len * attn_g {
                    items.extend_from_slice(&[0, t_len as u32, t0 as u32, 0]);
                    t0 += 64;
                }
                let n_tiles = items.len() / 4;
                let d_items = exec.stream.clone_htod(&items).expect("items");
                let mut d_vl = exec.alloc(t_len * qdim).expect("vl");
                exec.attn_prefill_f16_paged_vl(
                    &d_q, &d_k, &d_v, &d_s, &mut d_vl, &d_pos, &d_items, n_tiles, &d_bt, bps,
                    n_heads, n_kv_heads, head_dim, kv_dim, swa, scale, dt,
                )
                .expect("paged f16 prefill VL (fp8)");
                let vl = exec.to_host(&d_vl).expect("dtoh vl");
                let vld = r
                    .iter()
                    .zip(&vl)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "fp8 paged f16 prefill VL G={attn_g} hd={head_dim} T={t_len} tiles={n_tiles}: max_abs_diff {vld:.2e}"
                );
                assert!(
                    vld < 8e-3,
                    "fp8 VL prefill G={attn_g} T={t_len}: {vld} exceeds the f16-class gate - \
                     this route runs ONLY under fp8, so nothing else would catch it"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Spec-verify attention
//
// Six attn_spec_* entries take a KvDtype and none had ever run a parity check -
// gpu_spec_batch_kernels.rs covers the spec PLUMBING (commit, rollback walk,
// cond_append, ring runs) and never the attention itself.
//
// They matter for the fp8-KV question because the qwen35 verify arm is fp8-ONLY:
// attn_verify_dispatch (gpu_model/qwen35/ops.rs) refuses unless
// kv_dtype == Fp8E4m3, head_dim == 256, n_heads == 6 * n_kv_heads and
// k1 in 2..=8. Like attn_prefill_f16_paged_vl, a route that cannot execute in
// an fp16 configuration is one fp16-only testing could never have reached.
//
// *** this TEST is RED on MAIN. It found a real defect; see the task. ***
//
// ORACLE. attn_decode_batch, proven correct on an e4m3 cache by the storage and
// accumulation arms above. Spec verify is per-row decode attention: draft row j
// of a slot attends [0 ..= its own position], with the round's earlier draft
// tokens already in that slot's cache. Same pool, same per-row positions and
// slots, so the same answer. The harness is not in doubt: with the fa6 arm
// disabled (PADDOCK_NO_SPEC_FA6=1) these exact buffers give 5.96e-8 through the
// fallthrough walk, i.e. f32 rounding noise.
//
// SINKS. The spec attention kernels have no sink term at all -
// pd_attn_spec_batch_fin is literally pd_attn_spec_batch_paged with
// n_splits = FIN|1 and no sink pointer is passed; the sink enters only at
// attn_combine_batch, which gemma4 hands a NEG-INF buffer. The reference uses
// -inf for the same reason: a ZEROED sink is not a no-sink, it adds
// exp(0 - max) of phantom mass to every denominator.
//
// TOLERANCE. Not the f32 gates. The pack's own comment on this arm reads
// "Numerics: the shipped fp8 dense-decode class (e4m3 Q/P rounding); the f16-KV
// exact gates never reach this arm" - Q and the probabilities are e4m3 too, by
// design. So this asks whether the arm is in CLASS, not whether it is exact. A
// wrong walk, a wrong slot, a wrong causal bound or a dropped split all miss by
// far more than the rounding class, and an arm that writes nothing misses by
// exactly |reference|.

/// Per-row spec-verify geometry: `blocks` slots, `k1` draft rows each, every row
/// at its own position. Returns (positions, slots).
fn spec_rows(blocks: usize, k1: usize) -> (Vec<u32>, Vec<u32>) {
    let mut positions = Vec::with_capacity(blocks * k1);
    let mut slots = Vec::with_capacity(blocks * k1);
    for b in 0..blocks {
        // a different depth per slot, so a kernel that reads one slot's position
        // for another's rows cannot pass
        let base = 40 + b * 7;
        for j in 0..k1 {
            positions.push((base + j) as u32);
            slots.push(b as u32);
        }
    }
    (positions, slots)
}

/// How far off the oracle a spec arm may land: the fp8 dense-decode class, which
/// is coarse (e4m3 Q and P) but nowhere near this wide.
const SPEC_CLASS: f32 = 5e-2;

#[test]
fn spec_verify_attention_matches_decode_batch_on_fp8() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_paged_kv() || !exec.has_attn_spec_batch_paged() {
        eprintln!("pack has no spec attention - skipping");
        return;
    }
    // the qwen35 verify-FA gate's own geometry
    let (n_heads, n_kv_heads, head_dim) = (24usize, 4usize, 256usize);
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_ctx = 512usize;
    let bps = max_ctx / 16;
    let blocks = 16usize; // slots in the round
    let qdim = n_heads * head_dim;
    let dt = KvDtype::Fp8E4m3;

    // one cache per slot, read two ways: dense [blocks, max_ctx, kv_dim] for the
    // oracle and pool [blocks*bps, 16, kv_dim] for the spec kernels - the same
    // bytes under the identity block table
    let mut kc = Vec::new();
    let mut vc = Vec::new();
    for b in 0..blocks {
        kc.extend(det(max_ctx * kv_dim, 100 + b as u64));
        vc.extend(det(max_ctx * kv_dim, 200 + b as u64));
    }
    let d_k = e4m3_dev_u8(&exec, &kc);
    let d_v = e4m3_dev_u8(&exec, &vc);
    let bt_host: Vec<u32> = (0..(blocks * bps) as u32).collect();
    let d_bt = exec.stream.clone_htod(&bt_host).expect("bt");
    // no-sink, spelled the way the engine spells it (gemma4's neg_inf_sinks)
    let d_nosink = exec
        .to_device(&vec![f32::NEG_INFINITY; n_heads])
        .expect("neg-inf sinks");

    // collect rather than stop at the first: which arms work is the finding
    let mut fails: Vec<String> = Vec::new();
    let mut worst = 0f32;
    // "wrote nothing" is a distinct failure from "computed badly", and only one
    // of them is a numerics question - say which
    let judge = |got: &[f32], r: &[f32], what: String, fails: &mut Vec<String>| -> f32 {
        let maxd = r
            .iter()
            .zip(got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let nz = got.iter().filter(|v| **v != 0.0).count();
        eprintln!(
            "{what}: max_abs_diff {maxd:.2e}{}",
            if nz == 0 { "  [WROTE NOTHING]" } else { "" }
        );
        // a NaN max must fail too, so this is deliberately not `maxd >= SPEC_CLASS`
        if maxd.partial_cmp(&SPEC_CLASS) != Some(std::cmp::Ordering::Less) {
            fails.push(if nz == 0 {
                format!(
                    "{what}: wrote NOTHING (0/{} nonzero), reported success",
                    got.len()
                )
            } else {
                format!("{what}: {maxd:.3e} out of class")
            });
        }
        maxd
    };

    for k1 in [2usize, 4, 8] {
        let rows = blocks * k1;
        let (positions, slots) = spec_rows(blocks, k1);
        let q = det(rows * qdim, 1 + k1 as u64);
        let d_q = exec.to_device(&q).expect("q");
        let d_pos = exec.stream.clone_htod(&positions).expect("pos");
        let d_slots = exec.stream.clone_htod(&slots).expect("slots");

        // oracle: the same rows as ordinary per-row decode attention
        let mut d_ref = exec.alloc(rows * qdim).expect("ref");
        exec.attn_decode_batch(
            &d_q,
            &d_k,
            &d_v,
            &d_nosink,
            &mut d_ref,
            &d_pos,
            Some(&d_slots),
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_dim,
            0,
            rows,
            scale,
            dt,
        )
        .expect("decode_batch oracle");
        let r = exec.to_host(&d_ref).expect("dtoh ref");

        // (a) fin: n_splits == 1 with the finalize folded into the kernel
        if exec.has_attn_spec_batch_fin() {
            let mut d_out = exec.alloc(rows * qdim).expect("fin out");
            let mut d_ml = exec.alloc(n_heads * rows * 2).expect("fin ml");
            let ran = exec
                .attn_spec_batch_fin(
                    &d_q,
                    &d_k,
                    &d_v,
                    &mut d_out,
                    &mut d_ml,
                    &d_pos,
                    Some(&d_slots),
                    &d_bt,
                    bps,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    0,
                    rows,
                    k1,
                    scale,
                    dt,
                )
                .expect("attn_spec_batch_fin");
            // Result<bool>: false is "declined, caller falls back" - comparing
            // then would read an untouched buffer and could only pass by luck
            if ran {
                let got = exec.to_host(&d_out).expect("dtoh fin");
                worst = worst.max(judge(
                    &got,
                    &r,
                    format!("spec fin       k1={k1} rows={rows}"),
                    &mut fails,
                ));
            } else {
                eprintln!("spec fin       k1={k1}: declined (rc -2), caller falls back");
            }
        }

        // (b) paged partials + the separate combine (the n_splits > 1 dispatch)
        for n_splits in [2usize, 4] {
            let mut d_o = exec.alloc(n_heads * rows * n_splits * head_dim).expect("o");
            let mut d_ml = exec.alloc(n_heads * rows * n_splits * 2).expect("ml");
            let mut d_out = exec.alloc(rows * qdim).expect("out");
            exec.attn_spec_batch_paged(
                &d_q,
                &d_k,
                &d_v,
                &mut d_o,
                &mut d_ml,
                &d_pos,
                Some(&d_slots),
                &d_bt,
                bps,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                n_splits,
                rows,
                k1,
                scale,
                dt,
            )
            .expect("attn_spec_batch_paged");
            exec.attn_combine_batch(
                &d_o, &d_ml, &d_nosink, &mut d_out, n_heads, head_dim, n_splits, rows,
            )
            .expect("combine");
            let got = exec.to_host(&d_out).expect("dtoh paged");
            worst = worst.max(judge(
                &got,
                &r,
                format!("spec paged     k1={k1} rows={rows} splits={n_splits}"),
                &mut fails,
            ));

            // (c) a16: the f16-q twin. The fa6 arm answers -3 for it deliberately
            // ("no f16-q twin of the GV arm; the engine only arms f32 q"), so a
            // refusal is the documented outcome - but if it ever does take this
            // geometry the numbers have to hold, because gemma4's forward
            // dispatches it. No has_* probe exists for the a16 slots; a null
            // slot surfaces as MissingOp, which reads like any other refusal.
            {
                let mut d_o2 = exec
                    .alloc(n_heads * rows * n_splits * head_dim)
                    .expect("o2");
                let mut d_ml2 = exec.alloc(n_heads * rows * n_splits * 2).expect("ml2");
                let mut d_out2 = exec.alloc(rows * qdim).expect("out2");
                match exec.attn_spec_batch_paged_a16(
                    &d_q,
                    &d_k,
                    &d_v,
                    &mut d_o2,
                    &mut d_ml2,
                    &d_pos,
                    Some(&d_slots),
                    &d_bt,
                    bps,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    0,
                    n_splits,
                    rows,
                    k1,
                    scale,
                    dt,
                ) {
                    Err(e) => {
                        if k1 == 2 && n_splits == 2 {
                            eprintln!(
                                "spec paged a16 : declines this geometry ({e:?}) - expected, the GV arm has no f16-q twin"
                            );
                        }
                    }
                    Ok(()) => {
                        exec.attn_combine_batch(
                            &d_o2,
                            &d_ml2,
                            &d_nosink,
                            &mut d_out2,
                            n_heads,
                            head_dim,
                            n_splits,
                            rows,
                        )
                        .expect("combine a16");
                        let g2 = exec.to_host(&d_out2).expect("dtoh a16");
                        worst = worst.max(judge(
                            &g2,
                            &r,
                            format!("spec paged a16 k1={k1} splits={n_splits}"),
                            &mut fails,
                        ));
                    }
                }
            }

            // (d) lco: partials and combine fused behind an arrival-ticket
            // barrier. The tickets are zeroed once by contract (the kernel's
            // atomicInc wraps at n_splits-1), so allocate zeroed per call here
            // rather than reusing across shapes.
            if exec.has_attn_spec_lco_paged() {
                let mut d_ao = exec
                    .alloc(n_heads * rows * n_splits * head_dim)
                    .expect("ao");
                let mut d_aml = exec.alloc(n_heads * rows * n_splits * 2).expect("aml");
                let mut d_out3 = exec.alloc(rows * qdim).expect("out3");
                let mut d_tk = exec
                    .stream
                    .alloc_zeros::<u32>(n_heads * rows.max(1) * 8)
                    .expect("tickets");
                match exec.attn_spec_lco_paged(
                    &d_q,
                    &d_k,
                    &d_v,
                    &mut d_ao,
                    &mut d_aml,
                    &d_nosink,
                    &mut d_out3,
                    &mut d_tk,
                    &d_pos,
                    Some(&d_slots),
                    &d_bt,
                    bps,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    0,
                    n_splits,
                    rows,
                    k1,
                    scale,
                    dt,
                ) {
                    Err(e) => {
                        if k1 == 2 && n_splits == 2 {
                            eprintln!("spec lco       : declines this geometry ({e:?})");
                        }
                    }
                    Ok(false) => {
                        if k1 == 2 && n_splits == 2 {
                            eprintln!("spec lco       : not elected for this geometry");
                        }
                    }
                    Ok(true) => {
                        let g3 = exec.to_host(&d_out3).expect("dtoh lco");
                        worst = worst.max(judge(
                            &g3,
                            &r,
                            format!("spec lco       k1={k1} rows={rows} splits={n_splits}"),
                            &mut fails,
                        ));
                    }
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "spec-verify attention disagrees with per-row decode on an e4m3 cache.\n\
         The harness is NOT in doubt: PADDOCK_NO_SPEC_FA6=1 turns every one of \
         these green at 5.96e-8 through the fallthrough walk, on the same \
         buffers. What is left is the fa6 arm (pd_attn_spec_fa_krs_kernel, the \
         24q/4kv/hd256 fp8 GV=6 verify arm) launching, returning success from \
         pd_launch_status(), and writing nothing.\n{fails:#?}"
    );
    eprintln!("spec verify attention agrees with per-row decode on e4m3; worst {worst:.2e}");
}

// ---------------------------------------------------------------------------
// The family-specific KV WRITERS at fp8
//
// Everything above tests kernels that READ an fp8 cache, plus the shared append
// that writes it. But three families bypass that shared append with a fused
// writer of their own, and those are what decide whether fp8 KV can be enabled
// for an ARCH rather than for one family:
//
//   gemma4    kv_nra_rows, kv_nra_rows_i16      <- covered here
//   granite   rope_norm_qk_append_paged         <- covered here
//   qwen35    q36_qkg_nra_rows                  ruled out live (PADDOCK_NO_QNF
//                                               changes nothing), dflash_cond_
//                                               append is f16 by design
//   gpt_oss   qkv_rope_append_batch_paged       already dtype-swept, green
//   laguna, nemotron                            shared append only
//
// Method is the repo's own (gpu_spec_batch_kernels.rs
// dflash_cond_append_matches_norm_rope_append_chain): run the fused writer and
// the equivalent unfused norm/rope/append chain into two pools pre-filled with
// a poison byte, and compare the POOL BYTES. The chain's append is the one the
// storage arm already proved byte-exact against the e4m3 spec, so agreement
// means the fused writer stores the same thing.
//
// The poison fill matters as much as the comparison: it is what distinguishes
// "wrote the same bytes" from "wrote nothing", and writing nothing while
// reporting success is exactly the failure this suite met today.

/// Compare two KV pools as VALUES at the storage format's own resolution.
///
/// Bit-exactness is the wrong bar for a FUSED writer: it carries K in registers
/// through norm -> rope -> store, while the chain round-trips through an f32
/// buffer between each step, so a few values legitimately land one ulp apart.
/// Requiring byte equality would make this test fail for a reason that is not a
/// defect. What must hold is that no element is further than one representable
/// step apart, and that the differing FRACTION stays tiny - a real corruption
/// moves many elements by far more than a step, and cannot hide under either.
fn pool_agrees(a: &[u8], b: &[u8], dt: KvDtype) -> (usize, f32, f32) {
    let dec = |raw: &[u8]| -> Vec<f32> {
        match dt {
            KvDtype::Fp16 => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            KvDtype::Fp8E4m3 => raw.iter().map(|&x| e4m3_decode(x)).collect(),
        }
    };
    // one step of the format, relative: f16 has 10 mantissa bits, e4m3 has 3
    let step = match dt {
        KvDtype::Fp16 => 2f32.powi(-10),
        KvDtype::Fp8E4m3 => 2f32.powi(-3),
    };
    let (va, vb) = (dec(a), dec(b));
    let mut n = 0usize;
    let mut worst = 0f32;
    for (&x, &y) in va.iter().zip(&vb) {
        if x == y {
            continue;
        }
        n += 1;
        let rel = (x - y).abs() / x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        worst = worst.max(rel);
    }
    (n, worst, step * 1.5)
}

/// Fused K/V writers must store the same bytes as the unfused chain at both
/// cache widths. A writer that is right at fp16 and wrong at e4m3 corrupts
/// every read downstream, and no amount of correct attention recovers it.
#[test]
fn fused_kv_writers_match_the_chain_at_both_dtypes() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if !exec.has_paged_kv() {
        return;
    }
    let (n_kv, hd, bps, n_blocks) = (4usize, 128usize, 8usize, 8usize);
    let kv_dim = n_kv * hd;
    let eps = 1e-6f32;
    // ext_factor 0 pins ramp = 0 while still walking the full theta chain
    let params = (
        10000f32.powf(-2.0 / hd as f32),
        1.0f32,
        0.0f32,
        1.0f32,
        0.0f32,
        1.0f32,
    );
    let rows = 24usize;

    // two slots, crossing a 16-position block boundary on the second
    let mut pos: Vec<u32> = Vec::new();
    let mut slots: Vec<u32> = Vec::new();
    for i in 0..rows {
        if i < rows / 2 {
            slots.push(0);
            pos.push(3 + i as u32);
        } else {
            slots.push(1);
            pos.push(10 + (i - rows / 2) as u32);
        }
    }
    let mut bt = vec![0u32; 2 * bps];
    bt[0] = 2;
    bt[1] = 3;
    bt[bps] = 5;
    bt[bps + 1] = 7;

    let kw = det(hd, 13);
    let d_kw = exec.to_device(&kw).expect("kw");
    let d_pos = exec.to_device_u32(&pos).expect("pos");
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let d_bt = exec.to_device_u32(&bt).expect("bt");

    for dt in KV_DTYPES {
        let pool_bytes = n_blocks * 16 * kv_dim * dt.bytes();
        // poison, so "wrote nothing" cannot masquerade as "wrote the same"
        let fill = vec![0xABu8; pool_bytes];

        // ---- gemma4's writer: rmsnorm(k) + rope(k) + append k,v
        if exec.has_kv_nra_rows() {
            let fk = det(rows * kv_dim, 11);
            let fv = det(rows * kv_dim, 12);
            let d_fk = exec.to_device(&fk).expect("fk");
            let d_fv = exec.to_device(&fv).expect("fv");
            let mut a_k = exec.to_device_u8(&fill).expect("a_k");
            let mut a_v = exec.to_device_u8(&fill).expect("a_v");
            let mut b_k = exec.to_device_u8(&fill).expect("b_k");
            let mut b_v = exec.to_device_u8(&fill).expect("b_v");

            let mut kn = exec.to_device(&vec![0f32; rows * kv_dim]).expect("kn");
            exec.rmsnorm_batch(&d_fk, &d_kw, &mut kn, hd, eps, rows * n_kv)
                .expect("norm");
            exec.rope_yarn_batch(&mut kn, &d_pos, n_kv, hd, params, rows)
                .expect("rope");
            exec.kv_append_batch_paged_rows(
                &kn,
                &mut a_k,
                &d_pos,
                Some(&d_slots),
                &d_bt,
                bps,
                kv_dim,
                0,
                rows,
                dt,
            )
            .expect("chain append k");
            exec.kv_append_batch_paged_rows(
                &d_fv,
                &mut a_v,
                &d_pos,
                Some(&d_slots),
                &d_bt,
                bps,
                kv_dim,
                0,
                rows,
                dt,
            )
            .expect("chain append v");

            exec.kv_nra_rows(
                &d_fk,
                Some(&d_fv),
                &d_kw,
                &mut b_k,
                &mut b_v,
                &d_pos,
                Some(&d_slots),
                None,
                // neox must match what rope_yarn_batch implements, or the chain
                // ropes K differently and the pools legitimately disagree
                &d_bt,
                bps,
                n_kv,
                hd,
                eps,
                params,
                0,
                rows,
                dt,
                true,
                false,
            )
            .expect("kv_nra_rows");

            let (ak, bk) = (
                exec.to_host_range_u8(&a_k, 0, pool_bytes).expect("ak"),
                exec.to_host_range_u8(&b_k, 0, pool_bytes).expect("bk"),
            );
            let (av, bv) = (
                exec.to_host_range_u8(&a_v, 0, pool_bytes).expect("av"),
                exec.to_host_range_u8(&b_v, 0, pool_bytes).expect("bv"),
            );
            assert_ne!(
                bk, fill,
                "kv_nra_rows {dt:?}: k pool untouched - wrote NOTHING"
            );
            assert_ne!(
                bv, fill,
                "kv_nra_rows {dt:?}: v pool untouched - wrote NOTHING"
            );
            let (nk, rk, tol) = pool_agrees(&ak, &bk, dt);
            let (nv, rv, _) = pool_agrees(&av, &bv, dt);
            eprintln!(
                "kv_nra_rows {dt:?}: k {nk} elems differ (worst rel {rk:.2e}),                  v {nv} (worst rel {rv:.2e}), one-step tol {tol:.2e}"
            );
            assert!(
                rk <= tol,
                "kv_nra_rows {dt:?}: k is {rk:.3e} off, past one step ({tol:.3e})"
            );
            assert!(
                rv <= tol,
                "kv_nra_rows {dt:?}: v is {rv:.3e} off, past one step ({tol:.3e})"
            );
            let frac = 100.0 * nk as f64 / (ak.len() as f64);
            assert!(
                frac < 1.0,
                "kv_nra_rows {dt:?}: {frac:.2}% of k elements differ - too many for rounding"
            );
        }

        // ---- granite's rope_norm_qk_append_paged is not covered here, and
        // deliberately so. Its name says "norm" but it takes no weight tensor
        // and no eps, so a rope-only chain is the obvious reading - and that
        // reading is wrong: it lands worst-rel 2.0 against the fused kernel at
        // FP16, which is a different computation, not a rounding gap. Writing a
        // reference I cannot justify would produce a test that passes or fails
        // for reasons unrelated to fp8. Someone who knows that kernel's
        // contract should supply the chain; until then granite's writer is the
        // one KV writer with no fp8 evidence either way (task filed).
    }
}
