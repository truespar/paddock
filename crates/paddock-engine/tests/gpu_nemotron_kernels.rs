//! Unit gates for the nemotron_h_moe kernel set: each new
//! kernel against a host reference before any graph work trusts it.
//!
//!   1. mamba_conv_step        -> f64 reference, rel-to-mag (window bit-exact)
//!   2. mamba2_scan_seq        -> f64 host reference, rel-to-mag gated
//!   3. mamba_rmsnorm_gated_g  -> f64 host reference, rel gated
//!   4. f8r_gemv               -> real checkpoint fp8 plane, f64 reference
//!   5. nvf4 moe up/down       -> real expert planes, dequant_row host ref
//!   6. quantize_q8_relu2      -> BIT-exact vs relu2-into-f32 + quantize_q8
//!   7. Q8 decode-band MoE     -> real Q8 expert planes, vs the sorted route
//!
//! CUDA + pack gated; 4-5 additionally need the Nemotron NVFP4 checkpoint.

mod common;

use paddock_models::modelopt::{e4m3_to_f32, fp8_view, nvfp4_view};
use paddock_models::safetensors::ShardedSafetensors;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
/// The Q8 GGUF lane's own override (the NVFP4 dir cannot stand in for it).
const NEMO_Q8_ENV: &str = "NEMOTRON_Q8_GGUF";

// nemotron mamba geometry
const H: usize = 64;
const HD: usize = 64;
const S: usize = 128;
const G: usize = 8;
const D_INNER: usize = H * HD;
const CONV_DIM: usize = D_INNER + 2 * G * S;
const K_CONV: usize = 4;

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

#[test]
fn mamba_conv_step_matches_reference() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() {
        common::missing("pack has no mamba2 kernels");
        return;
    }
    let win0 = det(3 * CONV_DIM, 1);
    let x_new = det(CONV_DIM, 2);
    let w = det(CONV_DIM * K_CONV, 3);
    let b = det(CONV_DIM, 4);

    let mut d_win = exec.to_device(&win0).expect("win");
    let d_x = exec.to_device(&x_new).expect("x");
    let d_w = exec.to_device(&w).expect("w");
    let d_b = exec.to_device(&b).expect("b");
    let mut d_out = exec.alloc(CONV_DIM).expect("out");
    exec.mamba_conv_step(
        &mut d_win, &d_x, 0, &d_w, &d_b, &mut d_out, CONV_DIM, K_CONV,
    )
    .expect("conv step");
    let out = exec.to_host(&d_out).expect("out host");
    let win = exec.to_host(&d_win).expect("win host");

    for c in 0..CONV_DIM {
        // f64 host reference, gated rel-to-ACCUMULATION-magnitude: the
        // device contracts mul+add to fmaf and its expf is ULP-off libm, so
        // a near-cancelling acc (|acc| << sum|terms|) can never bit-match a
        // host mirror - but format/index bugs are O(mag) and still fail.
        let vals = [
            win0[c],
            win0[CONV_DIM + c],
            win0[2 * CONV_DIM + c],
            x_new[c],
        ];
        let mut acc = b[c] as f64;
        let mut mag = (b[c] as f64).abs();
        for j in 0..K_CONV {
            let term = w[c * K_CONV + j] as f64 * vals[j] as f64;
            acc += term;
            mag += term.abs();
        }
        let want = acc / (1.0 + (-acc).exp());
        let rel = (out[c] as f64 - want).abs() / mag.max(1e-6);
        assert!(rel < 1e-6, "conv out[{c}]: got {} want {want}", out[c]);
        // window advanced: rows 1..3 shifted down, new token appended
        assert_eq!(win[c].to_bits(), win0[CONV_DIM + c].to_bits());
        assert_eq!(
            win[CONV_DIM + c].to_bits(),
            win0[2 * CONV_DIM + c].to_bits()
        );
        assert_eq!(win[2 * CONV_DIM + c].to_bits(), x_new[c].to_bits());
    }
    println!("conv step: {CONV_DIM}ch k{K_CONV} pass (window shift bit-exact)");
}

#[test]
fn mamba_conv_seq_matches_serial_steps() {
    // The bulk span kernel must be BIT-exact vs stepping the serial kernel
    // token by token (same FMA order per token), including the final window
    // state - the bulk-prefill path's state handoff to decode depends on it.
    let Some(exec) = common::gpu() else { return };
    // The GGUF lane's gate, not the fp8 one: conv_seq runs in both lanes'
    // bulk prefill, but the fp8 pair is nulled below sm_89 - so asking the
    // fp8 question here skipped this test on exactly the Ampere cards whose
    // Q8_0 prefill runs this kernel (the same wrong-lane mistake
    // has_nemotron_prefill_gguf's comment records for the engine).
    if !exec.has_mamba2() || !exec.has_nemotron_prefill_gguf() {
        common::missing("pack has no mamba conv_seq kernel");
        return;
    }
    const X_OFF: usize = 33;
    const STRIDE: usize = CONV_DIM + 77; // fused-row layout: junk around the span
    // 13 rows = the serial kernel; 200 and 1022 = the T-chunked twin the
    // launcher elects from 128 rows (64-row chunks, a ragged last chunk, a
    // halo read from xbc for every chunk but the first, the window written
    // back by the last chunk only) - GB10 2026-09-11.
    for t_len in [13usize, 200, 1022] {
        #[allow(non_snake_case)]
        let T = t_len;
        let win0 = det(3 * CONV_DIM, 11);
        let xbc = det(T * STRIDE, 12);
        let w = det(CONV_DIM * K_CONV, 13);
        let b = det(CONV_DIM, 14);
        let d_xbc = exec.to_device(&xbc).expect("xbc");
        let d_w = exec.to_device(&w).expect("w");
        let d_b = exec.to_device(&b).expect("b");

        // serial reference: T conv steps, each fed the token's span slice
        let mut d_win_s = exec.to_device(&win0).expect("win serial");
        let mut d_step = exec.alloc(CONV_DIM).expect("step out");
        let mut serial_out = Vec::with_capacity(T * CONV_DIM);
        for t in 0..T {
            exec.mamba_conv_step(
                &mut d_win_s,
                &d_xbc,
                t * STRIDE + X_OFF,
                &d_w,
                &d_b,
                &mut d_step,
                CONV_DIM,
                K_CONV,
            )
            .expect("conv step");
            serial_out.extend(exec.to_host(&d_step).expect("step host"));
        }
        let win_serial = exec.to_host(&d_win_s).expect("win serial host");

        // bulk: one launch over the span
        let mut d_win_b = exec.to_device(&win0).expect("win bulk");
        let mut d_out = exec.alloc(T * CONV_DIM).expect("bulk out");
        exec.mamba_conv_seq(
            &mut d_win_b,
            &d_xbc,
            X_OFF,
            STRIDE,
            &d_w,
            &d_b,
            &mut d_out,
            CONV_DIM,
            K_CONV,
            T,
        )
        .expect("conv seq");
        let bulk_out = exec.to_host(&d_out).expect("bulk host");
        let win_bulk = exec.to_host(&d_win_b).expect("win bulk host");

        for i in 0..T * CONV_DIM {
            assert_eq!(
                bulk_out[i].to_bits(),
                serial_out[i].to_bits(),
                "out[{i}] (t {}, c {}): bulk {} vs serial {}",
                i / CONV_DIM,
                i % CONV_DIM,
                bulk_out[i],
                serial_out[i]
            );
        }
        for c in 0..2 * CONV_DIM {
            assert_eq!(
                win_bulk[c].to_bits(),
                win_serial[c].to_bits(),
                "window[{c}]"
            );
        }
        println!("conv seq: {T} tokens bit-exact vs serial steps (incl. final window)");
    }
}

#[test]
fn mamba2_scan_seq_matches_f64_reference() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() {
        common::missing("pack has no mamba2 kernels");
        return;
    }
    let t_len = 5usize;
    let state0 = det(H * HD * S, 10);
    let xbc = det(t_len * CONV_DIM, 11);
    let dt_stride = CONV_DIM + H; // emulate a fused-row layout
    let dt_raw = det(t_len * dt_stride, 12);
    // A = -exp(A_log) is negative; keep it in a sane range
    let a: Vec<f32> = det(H, 13).iter().map(|v| -v.abs() - 0.05).collect();
    let d: Vec<f32> = det(H, 14);
    let dt_bias: Vec<f32> = det(H, 15);

    // the arena layout is [h, S, i] (i-minor, for lane coalescing);
    // state0 holds the LOGICAL (h, i, j) values the f64 reference walks -
    // upload the transposed image
    let mut dev_img = vec![0f32; H * HD * S];
    for h in 0..H {
        for i in 0..HD {
            for j in 0..S {
                dev_img[(h * S + j) * HD + i] = state0[(h * HD + i) * S + j];
            }
        }
    }
    let mut d_state = exec.to_device(&dev_img).expect("state");
    let d_xbc = exec.to_device(&xbc).expect("xbc");
    let d_dt = exec.to_device(&dt_raw).expect("dt");
    let d_a = exec.to_device(&a).expect("a");
    let d_d = exec.to_device(&d).expect("d");
    let d_db = exec.to_device(&dt_bias).expect("db");
    let mut d_y = exec.alloc(t_len * D_INNER).expect("y");
    exec.mamba2_scan_seq(
        &mut d_state,
        &d_xbc,
        &d_dt,
        0,
        dt_stride,
        &d_a,
        &d_d,
        &d_db,
        &mut d_y,
        t_len,
        H,
        HD,
        S,
        G,
    )
    .expect("scan");
    let y = exec.to_host(&d_y).expect("y host");
    let state = exec.to_host(&d_state).expect("state host");

    // f64 reference
    let mut st: Vec<f64> = state0.iter().map(|v| *v as f64).collect();
    let mut worst = 0f64;
    for t in 0..t_len {
        let row = &xbc[t * CONV_DIM..(t + 1) * CONV_DIM];
        for h in 0..H {
            let g = h / (H / G); // repeat_interleave
            let v = (dt_raw[t * dt_stride + h] + dt_bias[h]) as f64;
            let dt = if v <= 20.0 { v.exp().ln_1p() } else { v };
            let decay = (dt * a[h] as f64).exp();
            for i in 0..HD {
                let x_ti = row[h * HD + i] as f64;
                let contrib = dt * x_ti;
                let mut acc = 0f64;
                let mut mag = 0f64;
                for j in 0..S {
                    let sb = row[D_INNER + g * S + j] as f64;
                    let sc = row[D_INNER + (G + g) * S + j] as f64;
                    let idx = (h * HD + i) * S + j;
                    st[idx] = decay * st[idx] + contrib * sb;
                    acc += st[idx] * sc;
                    mag += (st[idx] * sc).abs();
                }
                let want = acc + d[h] as f64 * x_ti;
                let got = y[t * D_INNER + h * HD + i] as f64;
                let rel = (got - want).abs() / mag.max(1e-6);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-5,
                    "y[t{t} h{h} i{i}]: got {got} want {want} rel {rel}"
                );
            }
        }
    }
    // Final state closeness: the update terms are O(1), so the honest gate
    // is ABSOLUTE error scaled to O(1 + |value|) - a near-cancelling entry
    // (|s| ~ 1e-4) carries the same f32 rounding as a large one, and a
    // rel-to-tiny gate would flag ordinary rounding as failure. Device
    // buffer is [h, S, i]; the reference stays logical (h, i, j).
    for h in 0..H {
        for i in 0..HD {
            for j in 0..S {
                let g0 = state[(h * S + j) * HD + i];
                let w0 = st[(h * HD + i) * S + j];
                let err = (g0 as f64 - w0).abs();
                assert!(
                    err < 1e-4 * (1.0 + w0.abs()),
                    "state[h{h} i{i} j{j}]: got {g0} want {w0}"
                );
            }
        }
    }
    println!("scan: T={t_len} worst y rel {worst:.2e}");
}

#[test]
fn mamba_rmsnorm_gated_g_matches_f64_reference() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() {
        common::missing("pack has no mamba2 kernels");
        return;
    }
    let t_len = 3usize;
    let z_stride = D_INNER + 77; // emulate fused rows
    let x = det(t_len * D_INNER, 20);
    let z = det(t_len * z_stride, 21);
    let w = det(D_INNER, 22);
    let eps = 1e-5f32;

    let d_x = exec.to_device(&x).expect("x");
    let d_z = exec.to_device(&z).expect("z");
    let d_w = exec.to_device(&w).expect("w");
    let mut d_out = exec.alloc(t_len * D_INNER).expect("out");
    exec.mamba_rmsnorm_gated_g(
        &d_x, &d_z, 0, z_stride, &d_w, &mut d_out, t_len, D_INNER, G, eps,
    )
    .expect("norm");
    let out = exec.to_host(&d_out).expect("out host");

    let gsize = D_INNER / G;
    for t in 0..t_len {
        for grp in 0..G {
            let mut ss = 0f64;
            let mut gated = vec![0f64; gsize];
            for (j, g) in gated.iter_mut().enumerate() {
                let c = grp * gsize + j;
                let xv = x[t * D_INNER + c] as f64;
                let zv = z[t * z_stride + c] as f64;
                let gv = xv * (zv / (1.0 + (-zv).exp()));
                *g = gv;
                ss += gv * gv;
            }
            let inv = 1.0 / (ss / gsize as f64 + eps as f64).sqrt();
            for (j, &gj) in gated.iter().enumerate() {
                let c = grp * gsize + j;
                let want = gj * inv * w[c] as f64;
                let got = out[t * D_INNER + c] as f64;
                assert!(
                    (got - want).abs() / want.abs().max(1e-4) < 1e-4,
                    "norm[t{t} g{grp} j{j}]: got {got} want {want}"
                );
            }
        }
    }
    println!("grouped gated norm: T={t_len} groups {G} x {gsize} pass");
}

#[test]
fn f8r_gemv_matches_f64_reference_on_real_plane() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_mamba2() {
        common::missing("pack has no mamba2 kernels");
        return;
    }
    let v = fp8_view(&st, "backbone.layers.0.mixer.in_proj").expect("in_proj view");
    let plane = exec
        .fp8_ckpt_to_f8row(v.weight, v.weight_scale, v.k, v.n)
        .expect("upload");
    let x = det(v.k, 30);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_y = exec.alloc(v.n).expect("y");
    exec.f8r_gemv(&plane, &d_x, &mut d_y, v.k, v.n)
        .expect("gemv");
    let y = exec.to_host(&d_y).expect("y host");

    let mut worst = 0f64;
    for o in (0..v.n).step_by(97) {
        let mut acc = 0f64;
        let mut mag = 0f64;
        for (&wq, &xk) in v.weight[o * v.k..(o + 1) * v.k].iter().zip(&x) {
            let wv = e4m3_to_f32(wq) as f64 * v.weight_scale as f64;
            acc += wv * xk as f64;
            mag += (wv * xk as f64).abs();
        }
        let rel = (y[o] as f64 - acc).abs() / mag.max(1e-6);
        worst = worst.max(rel);
        assert!(rel < 1e-5, "f8r row {o}: got {} want {acc} rel {rel}", y[o]);
    }
    println!("f8r_gemv in_proj [{}, {}]: worst rel {worst:.2e}", v.n, v.k);
}

#[test]
fn nvf4_moe_kernels_match_host_reference() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe() {
        common::missing("pack has no nvf4 moe consumers (cc != 12.0?)");
        return;
    }
    // 4-expert mini-planes from layer 1, real bytes, concatenated the way
    // the loader will (expert e's rows at e*ff).
    let n_e = 4usize;
    let mut up_p = Vec::new();
    let mut up_s = Vec::new();
    let mut up_s2 = Vec::new();
    let mut dn_p = Vec::new();
    let mut dn_s = Vec::new();
    let mut dn_s2 = Vec::new();
    let mut up_views = Vec::new();
    let mut dn_views = Vec::new();
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
        up_views.push(u);
        dn_views.push(d);
    }
    let (ff, embd) = (up_views[0].n, dn_views[0].n);
    let up = exec
        .nvf4_moe_upload(&up_p, &up_s, &up_s2, n_e, ff, up_views[0].k)
        .expect("up");
    let dn = exec
        .nvf4_moe_upload(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn");

    let k = 3usize;
    let idx: Vec<u32> = vec![2, 0, 3];
    let topk_w: Vec<f32> = vec![0.5, 0.3, 0.2];
    let x = det(up_views[0].k, 40);

    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&topk_w).expect("w");
    let d_x = exec.to_device(&x).expect("x");
    let mut d_up = exec.alloc(k * ff).expect("up out");
    let mut d_y = exec.alloc(embd).expect("y");
    exec.nvf4_moe_up_relu2(&up, &d_idx, &d_x, &mut d_up, k, 1)
        .expect("up gemv");
    exec.nvf4_moe_down_acc(&dn, &d_idx, &d_w, &d_up, &mut d_y, None, k, 1, false)
        .expect("down gemv");
    let up_out = exec.to_host(&d_up).expect("up host");
    let y = exec.to_host(&d_y).expect("y host");

    // host reference through the pinned dequant
    let mut href = vec![0f64; embd];
    for (slot, (&e, &wt)) in idx.iter().zip(topk_w.iter()).enumerate() {
        let u = &up_views[e as usize];
        let dnv = &dn_views[e as usize];
        let mut hu = vec![0f64; ff];
        for r in (0..ff).step_by(37) {
            let row = u.dequant_row_f32(r);
            let acc: f64 = row
                .iter()
                .zip(x.iter())
                .map(|(w, x)| *w as f64 * *x as f64)
                .sum();
            let v = acc.max(0.0);
            hu[r] = v * v;
            let got = up_out[slot * ff + r] as f64;
            let rel = (got - hu[r]).abs() / hu[r].abs().max(1e-5);
            assert!(
                rel < 1e-3,
                "up[slot {slot} r {r}]: got {got} want {}",
                hu[r]
            );
        }
        // full up rows for the down reference (host, exact dequant)
        for (r, h) in hu.iter_mut().enumerate() {
            if *h == 0.0 {
                let row = u.dequant_row_f32(r);
                let acc: f64 = row
                    .iter()
                    .zip(x.iter())
                    .map(|(w, x)| *w as f64 * *x as f64)
                    .sum();
                let v = acc.max(0.0);
                *h = v * v;
            }
        }
        // fold the GPU's own up outputs into the down reference instead?
        // No: the reference stays independent - hu is all-host.
        for o in (0..embd).step_by(53) {
            let row = dnv.dequant_row_f32(o);
            let dot: f64 = row.iter().zip(hu.iter()).map(|(w, x)| *w as f64 * *x).sum();
            href[o] += wt as f64 * dot;
        }
    }
    for o in (0..embd).step_by(53) {
        let got = y[o] as f64;
        let rel = (got - href[o]).abs() / href[o].abs().max(1e-4);
        assert!(rel < 1e-3, "down[{o}]: got {got} want {}", href[o]);
    }
    println!("nvf4 moe up+down: k={k} over {n_e} real experts pass");
}

// ---- sorted-tile MoE MMA gates -------------------------------------
// Both gates are BIT-exact, anchored on the already-verified dense
// pd_mxfp4_gemm_nv4: the bs kernels keep its per-acc K-accumulation order
// (kt ascending, k64 ascending) and MMA columns are independent, so a
// token's column in the 32-wide tile must equal the same token through the
// dense kernel. The up epilogue's relu2 + per-16 quantize is anchored by
// running the same device quantizer (pd_quantize_nvf4) on host-computed
// relu2 values of the dense output - same f32 inputs, same scale pick,
// byte-equal planes.

/// The exact f32 epilogue the up_bs kernel applies before its quantize:
/// v = relu(acc * scale2)^2, plain (non-FMA) f32 ops.
fn relu2_s2(y: &[f32], s2: f32) -> Vec<f32> {
    y.iter()
        .map(|&a| {
            let r = (a * s2).max(0.0f32);
            r * r
        })
        .collect()
}

fn htod_i8(exec: &paddock_engine::gpu::GpuExecutor, host: &[i8]) -> cudarc::driver::CudaSlice<i8> {
    let mut d = exec.alloc_i8(host.len()).expect("alloc i8");
    {
        let mut v = d.try_slice_mut(0..host.len()).expect("i8 view");
        exec.stream.memcpy_htod(host, &mut v).expect("htod i8");
    }
    d
}

#[test]
fn nvf4_moe_bs_pair_matches_dense_nv4_bitexact() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe_bs() || !exec.has_mxfp4_gemm_nv4() {
        common::missing("pack has no nvf4 moe bs kernels (cc != 12.0?)");
        return;
    }
    let u = nvfp4_view(&st, "backbone.layers.1.mixer.experts.0.up_proj").expect("up view");
    let d = nvfp4_view(&st, "backbone.layers.1.mixer.experts.0.down_proj").expect("down view");
    let (ff, in_dim, embd) = (u.n, u.k, d.n);
    assert_eq!(d.k, ff);
    let up = exec
        .nvf4_moe_upload(u.packed, u.scales, &[u.scale2], 1, ff, in_dim)
        .expect("up");
    let dn = exec
        .nvf4_moe_upload(d.packed, d.scales, &[d.scale2], 1, embd, ff)
        .expect("dn");
    // the same bytes as dense reference planes
    let up_rp = paddock_engine::gpu::RepackedMxfp4 {
        data: exec.to_device_u8(u.packed).expect("up data"),
        scale: exec.to_device_u8(u.scales).expect("up scale"),
    };
    let dn_rp = paddock_engine::gpu::RepackedMxfp4 {
        data: exec.to_device_u8(d.packed).expect("dn data"),
        scale: exec.to_device_u8(d.scales).expect("dn scale"),
    };

    // t = 20 < 32: the single block carries 12 PAD columns (pad-emission
    // coverage). Identity sorted layout, one block, expert 0.
    let t = 20usize;
    let x = det(t * in_dim, 91);
    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq4 = exec.alloc_i8(t * in_dim / 2).expect("xq4");
    let mut d_xs4 = exec.alloc_u8(t * in_dim / 16).expect("xs4");
    exec.quantize_nvf4(&d_x, &mut d_xq4, &mut d_xs4, t * in_dim)
        .expect("quant x");

    const PAD: u32 = u32::MAX;
    let mut srow = vec![PAD; 32];
    for (i, r) in srow.iter_mut().enumerate().take(t) {
        *r = i as u32;
    }
    let d_srow = exec.to_device_u32(&srow).expect("srow");
    let d_sslot = exec.to_device_u32(&[0u32; 32]).expect("sslot");
    let d_bexp = exec.to_device_u32(&[0u32]).expect("bexp");

    // up_bs vs dense + host relu2 + device quantizer
    let mut d_fq = exec.alloc_u8(32 * ff / 2).expect("fq");
    let mut d_fs = exec.alloc_u8(32 * ff / 16).expect("fs");
    exec.nvf4_moe_up_relu2_bs(
        &up, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, 1,
    )
    .expect("up bs");
    let mut d_yu = exec.alloc(t * ff).expect("yu");
    exec.mxfp4_gemm_nv4(&up_rp, &d_xq4, &d_xs4, &mut d_yu, in_dim, ff, t)
        .expect("dense up");
    let yu = exec.to_host(&d_yu).expect("yu host");
    let v = relu2_s2(&yu, u.scale2);
    let d_v = exec.to_device(&v).expect("v");
    let mut d_qref = exec.alloc_i8(t * ff / 2).expect("qref");
    let mut d_sref = exec.alloc_u8(t * ff / 16).expect("sref");
    exec.quantize_nvf4(&d_v, &mut d_qref, &mut d_sref, t * ff)
        .expect("quant ref");
    let fq = exec
        .to_host_range_u8(&d_fq, 0, 32 * ff / 2)
        .expect("fq host");
    let fs = exec
        .to_host_range_u8(&d_fs, 0, 32 * ff / 16)
        .expect("fs host");
    let qref = exec.to_host_i8(&d_qref).expect("qref host");
    let sref = exec
        .to_host_range_u8(&d_sref, 0, t * ff / 16)
        .expect("sref host");
    assert_eq!(
        &fq[..t * ff / 2],
        &qref.iter().map(|&b| b as u8).collect::<Vec<_>>()[..]
    );
    assert_eq!(&fs[..t * ff / 16], &sref[..]);
    assert!(
        fq[t * ff / 2..].iter().all(|&b| b == 0),
        "PAD fq rows must be zero"
    );
    assert!(
        fs[t * ff / 16..].iter().all(|&b| b == 0),
        "PAD fs rows must be zero"
    );

    // down_bs (weight 1.0, np=1) vs dense * scale2
    let mut d_part = exec.alloc(t * embd).expect("part");
    exec.nvf4_moe_down_bs(
        &dn,
        &d_srow,
        &d_sslot,
        &d_bexp,
        None,
        &d_fq,
        &d_fs,
        &mut d_part,
        1,
        1,
        0,
        1,
    )
    .expect("down bs");
    // dense B input: the up_bs fq/fs bytes verbatim (i8 upload of the u8 plane)
    let fq_i8: Vec<i8> = fq[..t * ff / 2].iter().map(|&b| b as i8).collect();
    let d_fq_i8 = htod_i8(&exec, &fq_i8);
    let d_fs_ref = exec.to_device_u8(&fs[..t * ff / 16]).expect("fs ref");
    let mut d_yd = exec.alloc(t * embd).expect("yd");
    exec.mxfp4_gemm_nv4(&dn_rp, &d_fq_i8, &d_fs_ref, &mut d_yd, ff, embd, t)
        .expect("dense dn");
    let part = exec.to_host(&d_part).expect("part host");
    let yd = exec.to_host(&d_yd).expect("yd host");
    let mut bad = 0usize;
    for i in 0..t * embd {
        let want = yd[i] * d.scale2;
        if part[i].to_bits() != want.to_bits() {
            bad += 1;
            if bad <= 4 {
                eprintln!("part[{i}] {} != dense*s2 {}", part[i], want);
            }
        }
    }
    assert_eq!(bad, 0, "down_bs must be bit-exact vs dense nv4 * scale2");
    println!("nvf4 moe bs pair: bit-exact vs dense nv4 on [{ff},{in_dim}]/[{embd},{ff}], t={t}");
}

#[test]
fn nvf4_moe_bs_composition_matches_per_token_dense() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe_bs() || !exec.has_mxfp4_gemm_nv4() {
        common::missing("pack has no nvf4 moe bs kernels (cc != 12.0?)");
        return;
    }
    // 4 real routed experts + expert 1's planes doubling as the "shared"
    // expert - exercises moe_align, the sorted gather, the per-(token, slot)
    // scatter with slot_off, and the fixed-order combine, all bit-exact.
    let n_e = 4usize;
    let mut views = Vec::new();
    for e in 0..n_e {
        let u = nvfp4_view(&st, &format!("backbone.layers.1.mixer.experts.{e}.up_proj"))
            .expect("up view");
        let d = nvfp4_view(
            &st,
            &format!("backbone.layers.1.mixer.experts.{e}.down_proj"),
        )
        .expect("down view");
        views.push((u, d));
    }
    let (ff, in_dim, embd) = (views[0].0.n, views[0].0.k, views[0].1.n);
    let mut up_p = Vec::new();
    let mut up_s = Vec::new();
    let mut up_s2 = Vec::new();
    let mut dn_p = Vec::new();
    let mut dn_s = Vec::new();
    let mut dn_s2 = Vec::new();
    for (u, d) in &views {
        up_p.extend_from_slice(u.packed);
        up_s.extend_from_slice(u.scales);
        up_s2.push(u.scale2);
        dn_p.extend_from_slice(d.packed);
        dn_s.extend_from_slice(d.scales);
        dn_s2.push(d.scale2);
    }
    let up = exec
        .nvf4_moe_upload(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up");
    let dn = exec
        .nvf4_moe_upload(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn");
    let sh = &views[1];
    let sh_up = exec
        .nvf4_moe_upload(sh.0.packed, sh.0.scales, &[sh.0.scale2], 1, ff, in_dim)
        .expect("sh up");
    let sh_dn = exec
        .nvf4_moe_upload(sh.1.packed, sh.1.scales, &[sh.1.scale2], 1, embd, ff)
        .expect("sh dn");
    // per-expert dense reference planes (same bytes)
    let dense: Vec<_> = views
        .iter()
        .map(|(u, d)| {
            (
                paddock_engine::gpu::RepackedMxfp4 {
                    data: exec.to_device_u8(u.packed).expect("du"),
                    scale: exec.to_device_u8(u.scales).expect("su"),
                },
                paddock_engine::gpu::RepackedMxfp4 {
                    data: exec.to_device_u8(d.packed).expect("dd"),
                    scale: exec.to_device_u8(d.scales).expect("sd"),
                },
            )
        })
        .collect();

    let (t, k) = (9usize, 3usize);
    let np = k + 1;
    // routing: spread across all 4 experts, distinct picks per token
    let idx: Vec<u32> = (0..t * k).map(|p| ((p * 7 + p / k) % n_e) as u32).collect();
    let topk_w: Vec<f32> = det(t * k, 55).iter().map(|v| 0.05 + v.abs()).collect();
    let x = det(t * in_dim, 77);
    let res0 = det(t * embd, 33);

    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq4 = exec.alloc_i8(t * in_dim / 2).expect("xq4");
    let mut d_xs4 = exec.alloc_u8(t * in_dim / 16).expect("xs4");
    exec.quantize_nvf4(&d_x, &mut d_xq4, &mut d_xs4, t * in_dim)
        .expect("quant x");
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&topk_w).expect("w");

    // routed align + tiles
    let nb_r = t * k / 32 + n_e;
    let mut d_srow = exec.alloc_u32(nb_r * 32).expect("srow");
    let mut d_sslot = exec.alloc_u32(nb_r * 32).expect("sslot");
    let mut d_bexp = exec.alloc_u32(nb_r).expect("bexp");
    exec.moe_align(
        &d_idx,
        &mut d_srow,
        &mut d_sslot,
        &mut d_bexp,
        t,
        k,
        n_e,
        nb_r,
    )
    .expect("align");
    let mut d_fq = exec.alloc_u8(nb_r * 32 * ff / 2).expect("fq");
    let mut d_fs = exec.alloc_u8(nb_r * 32 * ff / 16).expect("fs");
    exec.nvf4_moe_up_relu2_bs(
        &up, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, nb_r,
    )
    .expect("up bs");
    let mut d_part = exec.alloc(t * np * embd).expect("part");
    exec.nvf4_moe_down_bs(
        &dn,
        &d_srow,
        &d_sslot,
        &d_bexp,
        Some(&d_w),
        &d_fq,
        &d_fs,
        &mut d_part,
        k,
        np,
        0,
        nb_r,
    )
    .expect("down bs");

    // shared: identity align over t tokens (idx zeros), slot_off = k
    let d_sh_idx = exec.to_device_u32(&vec![0u32; t]).expect("sh idx");
    let nb_s = t / 32 + 1;
    let mut d_srow_s = exec.alloc_u32(nb_s * 32).expect("srow s");
    let mut d_sslot_s = exec.alloc_u32(nb_s * 32).expect("sslot s");
    let mut d_bexp_s = exec.alloc_u32(nb_s).expect("bexp s");
    exec.moe_align(
        &d_sh_idx,
        &mut d_srow_s,
        &mut d_sslot_s,
        &mut d_bexp_s,
        t,
        1,
        1,
        nb_s,
    )
    .expect("sh align");
    let mut d_fq_s = exec.alloc_u8(nb_s * 32 * ff / 2).expect("fq s");
    let mut d_fs_s = exec.alloc_u8(nb_s * 32 * ff / 16).expect("fs s");
    exec.nvf4_moe_up_relu2_bs(
        &sh_up,
        &d_srow_s,
        &d_bexp_s,
        &d_xq4,
        &d_xs4,
        &mut d_fq_s,
        &mut d_fs_s,
        nb_s,
    )
    .expect("sh up bs");
    exec.nvf4_moe_down_bs(
        &sh_dn,
        &d_srow_s,
        &d_sslot_s,
        &d_bexp_s,
        None,
        &d_fq_s,
        &d_fs_s,
        &mut d_part,
        1,
        np,
        k,
        nb_s,
    )
    .expect("sh down bs");

    let mut d_y = exec.to_device(&res0).expect("res");
    exec.moe_slot_combine(&d_part, &mut d_y, embd, np, t)
        .expect("combine");
    let y = exec.to_host(&d_y).expect("y host");

    // reference: per-(token, slot) batch-1 dense chain, folded in the
    // combine kernel's exact order (f32, ascending slot, then residual add)
    let xq4 = exec.to_host_i8(&d_xq4).expect("xq4 host");
    let xs4 = exec
        .to_host_range_u8(&d_xs4, 0, t * in_dim / 16)
        .expect("xs4 host");
    let per_token = |tok: usize, e: usize, wt: f32| -> Vec<f32> {
        let (urp, drp) = &dense[e];
        let d_xq1 = htod_i8(&exec, &xq4[tok * in_dim / 2..(tok + 1) * in_dim / 2]);
        let d_xs1 = exec
            .to_device_u8(&xs4[tok * in_dim / 16..(tok + 1) * in_dim / 16])
            .expect("xs1");
        let mut d_yu = exec.alloc(ff).expect("yu1");
        exec.mxfp4_gemm_nv4(urp, &d_xq1, &d_xs1, &mut d_yu, in_dim, ff, 1)
            .expect("dense up1");
        let yu = exec.to_host(&d_yu).expect("yu1 host");
        let v = relu2_s2(&yu, views[e].0.scale2);
        let d_v = exec.to_device(&v).expect("v1");
        let mut d_q1 = exec.alloc_i8(ff / 2).expect("q1");
        let mut d_s1 = exec.alloc_u8(ff / 16).expect("s1");
        exec.quantize_nvf4(&d_v, &mut d_q1, &mut d_s1, ff)
            .expect("quant1");
        let mut d_yd = exec.alloc(embd).expect("yd1");
        exec.mxfp4_gemm_nv4(drp, &d_q1, &d_s1, &mut d_yd, ff, embd, 1)
            .expect("dense dn1");
        let w = wt * views[e].1.scale2;
        exec.to_host(&d_yd)
            .expect("yd1 host")
            .iter()
            .map(|&v| v * w)
            .collect()
    };
    let mut bad = 0usize;
    for tok in 0..t {
        let mut parts = Vec::new();
        for j in 0..k {
            let e = idx[tok * k + j] as usize;
            parts.push(per_token(tok, e, topk_w[tok * k + j]));
        }
        // shared rides expert 1's planes at weight 1.0 (its own 1-plane
        // upload above), slot k - the kernel's topk_w=NULL path
        parts.push(per_token(tok, 1, 1.0));
        for o in 0..embd {
            let mut acc = 0.0f32;
            for p in &parts {
                acc += p[o];
            }
            let want = res0[tok * embd + o] + acc;
            if y[tok * embd + o].to_bits() != want.to_bits() {
                bad += 1;
                if bad <= 4 {
                    eprintln!("y[{tok},{o}] {} != ref {}", y[tok * embd + o], want);
                }
            }
        }
    }
    assert_eq!(
        bad, 0,
        "bs composition must be bit-exact vs per-token dense chain"
    );
    println!("nvf4 moe bs composition: t={t} k={k} + shared, bit-exact");
}

// ---- tiled-layout skinny pair gate  ---------------------
// The serve dispatch's skinny-decode shape end to end: routed picks through
// moe_align_bm(8) + the BM=8 _st pair over TILED planes, shared expert
// through the BM=32 _stw pair over its tiled planes - final residual must be
// BIT-EXACT vs the shipped row-major bs chain on the same routing (the _st
// family keeps the bs pair's kt/k64 accumulate order verbatim; the tiled
// repack is a pure byte permutation). t*k is chosen so per-expert fill
// EXCEEDS 8 and the align actually SPLITS experts into multiple blocks -
// the uniform-fill bench never exercised that path.
#[test]
fn nvf4_moe_st_skinny_chain_matches_bs_bitexact() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe_bs() || !exec.has_nvf4_moe_st() {
        common::missing("pack has no tiled nvf4 moe kernels (cc != 12.0?)");
        return;
    }
    let n_e = 4usize;
    let mut views = Vec::new();
    for e in 0..n_e {
        let u = nvfp4_view(&st, &format!("backbone.layers.1.mixer.experts.{e}.up_proj"))
            .expect("up view");
        let d = nvfp4_view(
            &st,
            &format!("backbone.layers.1.mixer.experts.{e}.down_proj"),
        )
        .expect("down view");
        views.push((u, d));
    }
    let (ff, in_dim, embd) = (views[0].0.n, views[0].0.k, views[0].1.n);
    assert!(
        ff % 64 == 0 && in_dim % 64 == 0 && embd % 64 == 0,
        "tiled dims"
    );
    let mut up_p = Vec::new();
    let mut up_s = Vec::new();
    let mut up_s2 = Vec::new();
    let mut dn_p = Vec::new();
    let mut dn_s = Vec::new();
    let mut dn_s2 = Vec::new();
    for (u, d) in &views {
        up_p.extend_from_slice(u.packed);
        up_s.extend_from_slice(u.scales);
        up_s2.push(u.scale2);
        dn_p.extend_from_slice(d.packed);
        dn_s.extend_from_slice(d.scales);
        dn_s2.push(d.scale2);
    }
    // both layouts resident side by side (test-only; serving keeps one)
    let up_r = exec
        .nvf4_moe_upload(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up r");
    let dn_r = exec
        .nvf4_moe_upload(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn r");
    let up_t = exec
        .nvf4_moe_upload_tiled(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up t");
    let dn_t = exec
        .nvf4_moe_upload_tiled(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn t");
    let sh = &views[1];
    let shu_r = exec
        .nvf4_moe_upload(sh.0.packed, sh.0.scales, &[sh.0.scale2], 1, ff, in_dim)
        .expect("shu r");
    let shd_r = exec
        .nvf4_moe_upload(sh.1.packed, sh.1.scales, &[sh.1.scale2], 1, embd, ff)
        .expect("shd r");
    let shu_t = exec
        .nvf4_moe_upload_tiled(sh.0.packed, sh.0.scales, &[sh.0.scale2], 1, ff, in_dim)
        .expect("shu t");
    let shd_t = exec
        .nvf4_moe_upload_tiled(sh.1.packed, sh.1.scales, &[sh.1.scale2], 1, embd, ff)
        .expect("shd t");

    // t*k = 36 over 4 experts -> ~9 picks/expert: BM=8 must split blocks
    let (t, k) = (12usize, 3usize);
    let np = k + 1;
    let idx: Vec<u32> = (0..t * k).map(|p| ((p * 7 + p / k) % n_e) as u32).collect();
    let topk_w: Vec<f32> = det(t * k, 55).iter().map(|v| 0.05 + v.abs()).collect();
    let x = det(t * in_dim, 77);
    let res0 = det(t * embd, 33);

    let d_x = exec.to_device(&x).expect("x");
    let mut d_xq4 = exec.alloc_i8(t * in_dim / 2).expect("xq4");
    let mut d_xs4 = exec.alloc_u8(t * in_dim / 16).expect("xs4");
    exec.quantize_nvf4(&d_x, &mut d_xq4, &mut d_xs4, t * in_dim)
        .expect("quant x");
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&topk_w).expect("w");
    let d_sh_idx = exec.to_device_u32(&vec![0u32; t]).expect("sh idx");

    // one chain runner per (layout, bm-shape); returns the folded residual
    let run = |tiled: bool| -> Vec<f32> {
        let (up, dn, shu, shd) = if tiled {
            (&up_t, &dn_t, &shu_t, &shd_t)
        } else {
            (&up_r, &dn_r, &shu_r, &shd_r)
        };
        let bm = if tiled { 8usize } else { 32usize };
        let nb_r = t * k / bm + n_e + 1;
        let mut d_srow = exec.alloc_u32(nb_r * 32).expect("srow");
        let mut d_sslot = exec.alloc_u32(nb_r * 32).expect("sslot");
        let mut d_bexp = exec.alloc_u32(nb_r).expect("bexp");
        let mut d_fq = exec.alloc_u8(nb_r * 32 * ff / 2).expect("fq");
        let mut d_fs = exec.alloc_u8(nb_r * 32 * ff / 16).expect("fs");
        let mut d_part = exec.alloc(t * np * embd).expect("part");
        if tiled {
            exec.moe_align_bm(
                &d_idx,
                &mut d_srow,
                &mut d_sslot,
                &mut d_bexp,
                t,
                k,
                n_e,
                8,
                nb_r,
            )
            .expect("align bm8");
            exec.nvf4_moe_up_relu2_st(
                up, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, nb_r, 8,
            )
            .expect("up st");
            exec.nvf4_moe_down_st(
                dn,
                &d_srow,
                &d_sslot,
                &d_bexp,
                Some(&d_w),
                &d_fq,
                &d_fs,
                &mut d_part,
                k,
                np,
                0,
                nb_r,
                8,
            )
            .expect("down st");
        } else {
            exec.moe_align(
                &d_idx,
                &mut d_srow,
                &mut d_sslot,
                &mut d_bexp,
                t,
                k,
                n_e,
                nb_r,
            )
            .expect("align");
            exec.nvf4_moe_up_relu2_bs(
                up, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, nb_r,
            )
            .expect("up bs");
            exec.nvf4_moe_down_bs(
                dn,
                &d_srow,
                &d_sslot,
                &d_bexp,
                Some(&d_w),
                &d_fq,
                &d_fs,
                &mut d_part,
                k,
                np,
                0,
                nb_r,
            )
            .expect("down bs");
        }
        // shared: BM=32 both layouts (the serve dispatch's shared shape)
        let nb_s = t / 32 + 1;
        let mut d_srow_s = exec.alloc_u32(nb_s * 32).expect("srow s");
        let mut d_sslot_s = exec.alloc_u32(nb_s * 32).expect("sslot s");
        let mut d_bexp_s = exec.alloc_u32(nb_s).expect("bexp s");
        exec.moe_align(
            &d_sh_idx,
            &mut d_srow_s,
            &mut d_sslot_s,
            &mut d_bexp_s,
            t,
            1,
            1,
            nb_s,
        )
        .expect("sh align");
        let mut d_fq_s = exec.alloc_u8(nb_s * 32 * ff / 2).expect("fq s");
        let mut d_fs_s = exec.alloc_u8(nb_s * 32 * ff / 16).expect("fs s");
        if tiled {
            exec.nvf4_moe_up_relu2_st(
                shu,
                &d_srow_s,
                &d_bexp_s,
                &d_xq4,
                &d_xs4,
                &mut d_fq_s,
                &mut d_fs_s,
                nb_s,
                32,
            )
            .expect("sh up stw");
            exec.nvf4_moe_down_st(
                shd,
                &d_srow_s,
                &d_sslot_s,
                &d_bexp_s,
                None,
                &d_fq_s,
                &d_fs_s,
                &mut d_part,
                1,
                np,
                k,
                nb_s,
                32,
            )
            .expect("sh down stw");
        } else {
            exec.nvf4_moe_up_relu2_bs(
                shu,
                &d_srow_s,
                &d_bexp_s,
                &d_xq4,
                &d_xs4,
                &mut d_fq_s,
                &mut d_fs_s,
                nb_s,
            )
            .expect("sh up bs");
            exec.nvf4_moe_down_bs(
                shd,
                &d_srow_s,
                &d_sslot_s,
                &d_bexp_s,
                None,
                &d_fq_s,
                &d_fs_s,
                &mut d_part,
                1,
                np,
                k,
                nb_s,
            )
            .expect("sh down bs");
        }
        let mut d_y = exec.to_device(&res0).expect("res");
        exec.moe_slot_combine(&d_part, &mut d_y, embd, np, t)
            .expect("combine");
        exec.to_host(&d_y).expect("y host")
    };
    let y_row = run(false);
    let y_til = run(true);
    let bad = y_row
        .iter()
        .zip(&y_til)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        bad, 0,
        "skinny tiled chain must be bit-exact vs the row-major bs chain"
    );
    // wide-tiled twin on the ROUTED experts too (the prefill shape): st(32)
    // over tiled planes vs the row chain
    {
        let nb_r = t * k / 32 + n_e;
        let mut d_srow = exec.alloc_u32(nb_r * 32).expect("srow w");
        let mut d_sslot = exec.alloc_u32(nb_r * 32).expect("sslot w");
        let mut d_bexp = exec.alloc_u32(nb_r).expect("bexp w");
        exec.moe_align(
            &d_idx,
            &mut d_srow,
            &mut d_sslot,
            &mut d_bexp,
            t,
            k,
            n_e,
            nb_r,
        )
        .expect("align w");
        let mut d_fq = exec.alloc_u8(nb_r * 32 * ff / 2).expect("fq w");
        let mut d_fs = exec.alloc_u8(nb_r * 32 * ff / 16).expect("fs w");
        exec.nvf4_moe_up_relu2_st(
            &up_t, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, nb_r, 32,
        )
        .expect("up stw");
        let mut d_part = exec.alloc(t * np * embd).expect("part w");
        exec.nvf4_moe_down_st(
            &dn_t,
            &d_srow,
            &d_sslot,
            &d_bexp,
            Some(&d_w),
            &d_fq,
            &d_fs,
            &mut d_part,
            k,
            np,
            0,
            nb_r,
            32,
        )
        .expect("down stw");
        let nb_s = t / 32 + 1;
        let mut d_srow_s = exec.alloc_u32(nb_s * 32).expect("srow ws");
        let mut d_sslot_s = exec.alloc_u32(nb_s * 32).expect("sslot ws");
        let mut d_bexp_s = exec.alloc_u32(nb_s).expect("bexp ws");
        exec.moe_align(
            &d_sh_idx,
            &mut d_srow_s,
            &mut d_sslot_s,
            &mut d_bexp_s,
            t,
            1,
            1,
            nb_s,
        )
        .expect("sh align ws");
        let mut d_fq_s = exec.alloc_u8(nb_s * 32 * ff / 2).expect("fq ws");
        let mut d_fs_s = exec.alloc_u8(nb_s * 32 * ff / 16).expect("fs ws");
        exec.nvf4_moe_up_relu2_st(
            &shu_t,
            &d_srow_s,
            &d_bexp_s,
            &d_xq4,
            &d_xs4,
            &mut d_fq_s,
            &mut d_fs_s,
            nb_s,
            32,
        )
        .expect("sh up ws");
        exec.nvf4_moe_down_st(
            &shd_t,
            &d_srow_s,
            &d_sslot_s,
            &d_bexp_s,
            None,
            &d_fq_s,
            &d_fs_s,
            &mut d_part,
            1,
            np,
            k,
            nb_s,
            32,
        )
        .expect("sh down ws");
        let mut d_y = exec.to_device(&res0).expect("res w");
        exec.moe_slot_combine(&d_part, &mut d_y, embd, np, t)
            .expect("combine w");
        let y_wide = exec.to_host(&d_y).expect("y wide host");
        let badw = y_row
            .iter()
            .zip(&y_wide)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            badw, 0,
            "wide tiled chain must be bit-exact vs the row-major bs chain"
        );
    }
    println!("[nvf4-st] skinny+wide tiled chains: t={t} k={k} (fill>8, blocks split), bit-exact");
}

/// Lever 14's door: the shared expert served as ns_sh = 2 pseudo-experts
/// inside the routed launch (the loader's fold-in: up = a row split, down =
/// a K split, one combine slot each) against the separate chain (routed
/// pair + the standalone shared plane into its own slot). Three chains on
/// the real shared expert of layer 1: A = separate (BM=8 routed + BM=32
/// standalone shared), B = fold at BM=8 (the skinny decode shape), C = fold
/// at BM=32 (the mixed/prefill shape the PPL lane ships). B and C must be
/// bit-exact (same per-token K order); B vs A is the split-K regroup of the
/// shared down (two f32 halves summed in the combine instead of one chain)
/// and must sit at f32 reassociation, not at a logprob-visible distance.
#[test]
fn nvf4_moe_sh_fold8_matches_separate_chain() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe_st() {
        common::missing("pack has no tiled nvf4 moe kernels (cc != 12.0?)");
        return;
    }
    let n_e = 4usize;
    let mut views = Vec::new();
    for e in 0..n_e {
        let u = nvfp4_view(&st, &format!("backbone.layers.1.mixer.experts.{e}.up_proj"))
            .expect("up view");
        let d = nvfp4_view(
            &st,
            &format!("backbone.layers.1.mixer.experts.{e}.down_proj"),
        )
        .expect("down view");
        views.push((u, d));
    }
    let shu = nvfp4_view(&st, "backbone.layers.1.mixer.shared_experts.up_proj").expect("sh up");
    let shd = nvfp4_view(&st, "backbone.layers.1.mixer.shared_experts.down_proj").expect("sh down");
    let (ff, in_dim, embd) = (views[0].0.n, views[0].0.k, views[0].1.n);
    assert_eq!(
        (shu.n, shu.k),
        (2 * ff, in_dim),
        "shared up = 2 x moe_ff rows"
    );
    assert_eq!(
        (shd.n, shd.k),
        (embd, 2 * ff),
        "shared down = 2 x moe_ff cols"
    );
    let ns = 2usize;
    // routed planes
    let (mut up_p, mut up_s, mut up_s2) = (Vec::new(), Vec::new(), Vec::new());
    let (mut dn_p, mut dn_s, mut dn_s2) = (Vec::new(), Vec::new(), Vec::new());
    for (u, d) in &views {
        up_p.extend_from_slice(u.packed);
        up_s.extend_from_slice(u.scales);
        up_s2.push(u.scale2);
        dn_p.extend_from_slice(d.packed);
        dn_s.extend_from_slice(d.scales);
        dn_s2.push(d.scale2);
    }
    let up_r = exec
        .nvf4_moe_upload_tiled(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up r");
    let dn_r = exec
        .nvf4_moe_upload_tiled(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn r");
    // standalone shared plane (one expert, ff = 2 x moe_ff)
    let shu_t = exec
        .nvf4_moe_upload_tiled(shu.packed, shu.scales, &[shu.scale2], 1, 2 * ff, in_dim)
        .expect("shu t");
    let shd_t = exec
        .nvf4_moe_upload_tiled(shd.packed, shd.scales, &[shd.scale2], 1, embd, 2 * ff)
        .expect("shd t");
    // fold-in planes: routed + the two halves, exactly as load.rs builds them
    let (mut fu_p, mut fu_s, mut fu_s2) = (up_p.clone(), up_s.clone(), up_s2.clone());
    fu_p.extend_from_slice(shu.packed); // row split: consecutive spans
    fu_s.extend_from_slice(shu.scales);
    for _ in 0..ns {
        fu_s2.push(shu.scale2);
    }
    let (mut fd_p, mut fd_s, mut fd_s2) = (dn_p.clone(), dn_s.clone(), dn_s2.clone());
    {
        // K split: pseudo-expert h takes columns [h*ff, (h+1)*ff) of every row
        let rb = ff / 2;
        let sb = ff / 16;
        let vrb = shd.k / 2;
        let vsb = shd.k / 16;
        for h in 0..ns {
            for r in 0..embd {
                fd_p.extend_from_slice(&shd.packed[r * vrb + h * rb..r * vrb + (h + 1) * rb]);
                fd_s.extend_from_slice(&shd.scales[r * vsb + h * sb..r * vsb + (h + 1) * sb]);
            }
        }
        for _ in 0..ns {
            fd_s2.push(shd.scale2);
        }
    }
    let up_f = exec
        .nvf4_moe_upload_tiled(&fu_p, &fu_s, &fu_s2, n_e + ns, ff, in_dim)
        .expect("up f");
    let dn_f = exec
        .nvf4_moe_upload_tiled(&fd_p, &fd_s, &fd_s2, n_e + ns, embd, ff)
        .expect("dn f");

    for &t in &[8usize, 32usize] {
        let k = 3usize;
        let kw = k + ns;
        // routed picks over the 4 experts, weights positive; shared picks 4, 5 at 1.0
        let idx_r: Vec<u32> = (0..t * k).map(|p| ((p * 7 + p / k) % n_e) as u32).collect();
        let w_r: Vec<f32> = det(t * k, 55).iter().map(|v| 0.05 + v.abs()).collect();
        let mut idx_f = Vec::with_capacity(t * kw);
        let mut w_f = Vec::with_capacity(t * kw);
        for tok in 0..t {
            idx_f.extend_from_slice(&idx_r[tok * k..(tok + 1) * k]);
            w_f.extend_from_slice(&w_r[tok * k..(tok + 1) * k]);
            for h in 0..ns {
                idx_f.push((n_e + h) as u32);
                w_f.push(1.0);
            }
        }
        let x = det(t * in_dim, 77);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_xq4 = exec.alloc_i8(t * in_dim / 2).expect("xq4");
        let mut d_xs4 = exec.alloc_u8(t * in_dim / 16).expect("xs4");
        exec.quantize_nvf4(&d_x, &mut d_xq4, &mut d_xs4, t * in_dim)
            .expect("quant x");
        let d_idx_r = exec.to_device_u32(&idx_r).expect("idx r");
        let d_w_r = exec.to_device(&w_r).expect("w r");
        let d_idx_f = exec.to_device_u32(&idx_f).expect("idx f");
        let d_w_f = exec.to_device(&w_f).expect("w f");
        let d_sh_idx = exec.to_device_u32(&vec![0u32; t]).expect("sh idx");
        let zeros = |np: usize| exec.to_device(&vec![0.0f32; t * np * embd]).expect("part");

        // A: separate - routed BM=8 into slots 0..k, standalone shared BM=32 into slot k
        let y_a = {
            let np = k + 1;
            let rows_cap = t * k + 8 * n_e;
            let nb = rows_cap / 8;
            let mut d_srow = exec.alloc_u32(rows_cap.max(32)).expect("srow");
            let mut d_sslot = exec.alloc_u32(rows_cap.max(32)).expect("sslot");
            let mut d_bexp = exec.alloc_u32(nb).expect("bexp");
            let mut d_fq = exec.alloc_u8(rows_cap * ff / 2).expect("fq");
            let mut d_fs = exec.alloc_u8(rows_cap * ff / 16).expect("fs");
            let mut d_part = zeros(np);
            exec.moe_align_bm(
                &d_idx_r,
                &mut d_srow,
                &mut d_sslot,
                &mut d_bexp,
                t,
                k,
                n_e,
                8,
                nb,
            )
            .expect("align r");
            exec.nvf4_moe_up_relu2_st(
                &up_r, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, nb, 8,
            )
            .expect("up r");
            exec.nvf4_moe_down_st(
                &dn_r,
                &d_srow,
                &d_sslot,
                &d_bexp,
                Some(&d_w_r),
                &d_fq,
                &d_fs,
                &mut d_part,
                k,
                np,
                0,
                nb,
                8,
            )
            .expect("down r");
            let nb_s = t / 32 + 1;
            let mut d_srow_s = exec.alloc_u32(nb_s * 32).expect("srow s");
            let mut d_sslot_s = exec.alloc_u32(nb_s * 32).expect("sslot s");
            let mut d_bexp_s = exec.alloc_u32(nb_s).expect("bexp s");
            exec.moe_align(
                &d_sh_idx,
                &mut d_srow_s,
                &mut d_sslot_s,
                &mut d_bexp_s,
                t,
                1,
                1,
                nb_s,
            )
            .expect("sh align");
            let mut d_fq_s = exec.alloc_u8(nb_s * 32 * (2 * ff) / 2).expect("fq s");
            let mut d_fs_s = exec.alloc_u8(nb_s * 32 * (2 * ff) / 16).expect("fs s");
            exec.nvf4_moe_up_relu2_st(
                &shu_t,
                &d_srow_s,
                &d_bexp_s,
                &d_xq4,
                &d_xs4,
                &mut d_fq_s,
                &mut d_fs_s,
                nb_s,
                32,
            )
            .expect("sh up");
            exec.nvf4_moe_down_st(
                &shd_t,
                &d_srow_s,
                &d_sslot_s,
                &d_bexp_s,
                None,
                &d_fq_s,
                &d_fs_s,
                &mut d_part,
                1,
                np,
                k,
                nb_s,
                32,
            )
            .expect("sh down");
            let mut d_y = exec.to_device(&vec![0.0f32; t * embd]).expect("y");
            exec.moe_slot_combine(&d_part, &mut d_y, embd, np, t)
                .expect("combine a");
            exec.to_host(&d_y).expect("y a")
        };
        // B / C: the fold at BM=8 and at BM=32
        let fold = |bm: usize| -> Vec<f32> {
            let np = kw;
            let rows_cap = t * kw + bm * (n_e + ns);
            let nb = rows_cap / bm;
            let mut d_srow = exec.alloc_u32(rows_cap.max(32)).expect("srow");
            let mut d_sslot = exec.alloc_u32(rows_cap.max(32)).expect("sslot");
            let mut d_bexp = exec.alloc_u32(nb).expect("bexp");
            let mut d_fq = exec.alloc_u8(rows_cap * ff / 2).expect("fq");
            let mut d_fs = exec.alloc_u8(rows_cap * ff / 16).expect("fs");
            let mut d_part = zeros(np);
            if bm == 32 {
                exec.moe_align(
                    &d_idx_f,
                    &mut d_srow,
                    &mut d_sslot,
                    &mut d_bexp,
                    t,
                    kw,
                    n_e + ns,
                    nb,
                )
                .expect("align f32");
            } else {
                exec.moe_align_bm(
                    &d_idx_f,
                    &mut d_srow,
                    &mut d_sslot,
                    &mut d_bexp,
                    t,
                    kw,
                    n_e + ns,
                    bm,
                    nb,
                )
                .expect("align f8");
            }
            exec.nvf4_moe_up_relu2_st(
                &up_f, &d_srow, &d_bexp, &d_xq4, &d_xs4, &mut d_fq, &mut d_fs, nb, bm,
            )
            .expect("up f");
            exec.nvf4_moe_down_st(
                &dn_f,
                &d_srow,
                &d_sslot,
                &d_bexp,
                Some(&d_w_f),
                &d_fq,
                &d_fs,
                &mut d_part,
                kw,
                np,
                0,
                nb,
                bm,
            )
            .expect("down f");
            let mut d_y = exec.to_device(&vec![0.0f32; t * embd]).expect("y");
            exec.moe_slot_combine(&d_part, &mut d_y, embd, np, t)
                .expect("combine f");
            exec.to_host(&d_y).expect("y f")
        };
        let y_b = fold(8);
        let y_c = fold(32);
        let bc = y_b
            .iter()
            .zip(&y_c)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let amax = y_a.iter().fold(0f32, |m, v| m.max(v.abs()));
        let (mut dmax, mut dsum) = (0f32, 0f64);
        for (a, b) in y_a.iter().zip(&y_b) {
            let d = (a - b).abs();
            dmax = dmax.max(d);
            dsum += d as f64;
        }
        println!(
            "[nvf4-fold8] t={t}: fold BM=8 vs BM=32 differing values {bc}/{}; fold vs separate max|d| {dmax:.3e} (max|y| {amax:.3e}, rel {:.2e}) mean|d| {:.3e}",
            y_b.len(),
            dmax / amax.max(1e-30),
            dsum / y_a.len() as f64
        );
        assert_eq!(
            bc, 0,
            "the fold at BM=8 and at BM=32 must be bit-exact (same per-token K order)"
        );
        assert!(
            dmax <= 1e-3 * amax.max(1e-30),
            "the fold moved the output beyond f32 reassociation: max|d| {dmax:.3e} vs max|y| {amax:.3e}"
        );
    }
}

// ---- tiled r=1 twins gate: the mt class rules (rel-to-rms + determinism) ---
#[test]
fn nvf4_moe_mtt_chain_matches_mt() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe_mt() || !exec.has_nvf4_moe_st() {
        common::missing("pack has no tiled nvf4 moe kernels (cc != 12.0?)");
        return;
    }
    let n_e = 4usize;
    let mut up_p = Vec::new();
    let mut up_s = Vec::new();
    let mut up_s2 = Vec::new();
    let mut dn_p = Vec::new();
    let mut dn_s = Vec::new();
    let mut dn_s2 = Vec::new();
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
    let sh = &views[1];
    let (ff_s, k) = (ff, 3usize);
    let up_r = exec
        .nvf4_moe_upload(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up r");
    let dn_r = exec
        .nvf4_moe_upload(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn r");
    let shu_r = exec
        .nvf4_moe_upload(sh.0.packed, sh.0.scales, &[sh.0.scale2], 1, ff_s, in_dim)
        .expect("shu r");
    let shd_r = exec
        .nvf4_moe_upload(sh.1.packed, sh.1.scales, &[sh.1.scale2], 1, embd, ff_s)
        .expect("shd r");
    let up_t = exec
        .nvf4_moe_upload_tiled(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up t");
    let dn_t = exec
        .nvf4_moe_upload_tiled(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn t");
    let shu_t = exec
        .nvf4_moe_upload_tiled(sh.0.packed, sh.0.scales, &[sh.0.scale2], 1, ff_s, in_dim)
        .expect("shu t");
    let shd_t = exec
        .nvf4_moe_upload_tiled(sh.1.packed, sh.1.scales, &[sh.1.scale2], 1, embd, ff_s)
        .expect("shd t");

    let idx: Vec<u32> = vec![0, 2, 3];
    let topk_w: Vec<f32> = vec![0.4, 0.35, 0.25];
    let x = det(in_dim, 91);
    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&topk_w).expect("w");
    let d_x = exec.to_device(&x).expect("x");

    let run_mt = |tiled: bool| -> (Vec<f32>, Vec<f32>) {
        let (up, dn, shu, shd) = if tiled {
            (&up_t, &dn_t, &shu_t, &shd_t)
        } else {
            (&up_r, &dn_r, &shu_r, &shd_r)
        };
        let mut d_act = exec.alloc(k * ff + ff_s).expect("act");
        let mut d_part = exec.alloc((k + 1) * embd).expect("part");
        if tiled {
            exec.nvf4_moe_up_relu2_mtt(up, shu, &d_idx, &d_x, &mut d_act, k)
                .expect("up mtt");
            exec.nvf4_moe_down_part_tt(dn, shd, &d_idx, &d_w, &d_act, &mut d_part, k)
                .expect("dn mtt");
        } else {
            exec.nvf4_moe_up_relu2_mt(up, shu, &d_idx, &d_x, &mut d_act, k)
                .expect("up mt");
            exec.nvf4_moe_down_part(dn, shd, &d_idx, &d_w, &d_act, &mut d_part, k)
                .expect("dn mt");
        }
        (
            exec.to_host(&d_act).expect("act host"),
            exec.to_host(&d_part).expect("part host"),
        )
    };
    let (act_r, part_r) = run_mt(false);
    let (act_t, part_t) = run_mt(true);
    // determinism: the tiled chain must reproduce itself bit-exactly
    let (act_t2, part_t2) = run_mt(true);
    assert!(
        act_t
            .iter()
            .zip(&act_t2)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && part_t
                .iter()
                .zip(&part_t2)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "mtt chain must be run-to-run deterministic"
    );
    // value: rel-to-rms vs the row-major mt chain (regrouped sums)
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let rms = (b.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / b.len() as f64)
            .sqrt()
            .max(1e-20);
        a.iter()
            .zip(b)
            .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
            .fold(0.0f64, f64::max)
            / rms
    };
    let ra = rel(&act_t, &act_r);
    let rp = rel(&part_t, &part_r);
    assert!(
        ra < 1e-4 && rp < 1e-4,
        "mtt rel-to-rms too high: act {ra} part {rp}"
    );
    println!("[nvf4-st] mtt twins: act rel {ra:.2e} part rel {rp:.2e}, deterministic");
}

// ---- decode rung: fused multi-task GEMV gates ------------------------------
// Rung 4b moved both mt kernels to CTA-per-task split-K x4 (the wave-tail
// fix), so their sums are deterministic fixed-order but REGROUPED vs the
// GEMV pair - every value gate here is rel-to-rms against the pair chain
// (the rung-4a precedent class; the token battery arbitrates end to end),
// and bit-exactness is retained only where it must hold: run-to-run
// determinism of the whole mt chain.
#[test]
fn nvf4_moe_mt_chain_matches_gemv_pair() {
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_nvf4_moe() || !exec.has_nvf4_moe_mt() {
        common::missing("pack has no nvf4 moe mt consumers (cc != 12.0?)");
        return;
    }
    let n_e = 4usize;
    let mut up_p = Vec::new();
    let mut up_s = Vec::new();
    let mut up_s2 = Vec::new();
    let mut dn_p = Vec::new();
    let mut dn_s = Vec::new();
    let mut dn_s2 = Vec::new();
    let (mut ff, mut embd, mut in_dim) = (0, 0, 0);
    for e in 0..n_e {
        let u = nvfp4_view(&st, &format!("backbone.layers.1.mixer.experts.{e}.up_proj"))
            .expect("up view");
        let d = nvfp4_view(
            &st,
            &format!("backbone.layers.1.mixer.experts.{e}.down_proj"),
        )
        .expect("down view");
        (ff, embd, in_dim) = (u.n, d.n, u.k);
        up_p.extend_from_slice(u.packed);
        up_s.extend_from_slice(u.scales);
        up_s2.push(u.scale2);
        dn_p.extend_from_slice(d.packed);
        dn_s.extend_from_slice(d.scales);
        dn_s2.push(d.scale2);
    }
    let up = exec
        .nvf4_moe_upload(&up_p, &up_s, &up_s2, n_e, ff, in_dim)
        .expect("up");
    let dn = exec
        .nvf4_moe_upload(&dn_p, &dn_s, &dn_s2, n_e, embd, ff)
        .expect("dn");
    // the real shared planes - ff_s (3712) differs from the routed ff, so the
    // fused task split is exercised at the serving shape
    let su = nvfp4_view(&st, "backbone.layers.1.mixer.shared_experts.up_proj").expect("sh up");
    let sd = nvfp4_view(&st, "backbone.layers.1.mixer.shared_experts.down_proj").expect("sh dn");
    let ff_s = su.n;
    let sh_up = exec
        .nvf4_moe_upload(su.packed, su.scales, &[su.scale2], 1, ff_s, in_dim)
        .expect("shu");
    let sh_dn = exec
        .nvf4_moe_upload(sd.packed, sd.scales, &[sd.scale2], 1, embd, ff_s)
        .expect("shd");

    let k = 6usize;
    let idx: Vec<u32> = vec![2, 0, 3, 1, 0, 2];
    let topk_w: Vec<f32> = vec![0.3, 0.25, 0.2, 0.15, 0.06, 0.04];
    let x = det(in_dim, 91);
    let residual = det(embd, 92);

    let d_idx = exec.to_device_u32(&idx).expect("idx");
    let d_w = exec.to_device(&topk_w).expect("w");
    let d_sh_idx = exec.to_device_u32(&[0]).expect("sh idx");
    let d_sh_w = exec.to_device(&[1.0f32]).expect("sh w");
    let d_x = exec.to_device(&x).expect("x");

    // Old chain: the 4-launch pair + residual add
    let mut d_up_o = exec.alloc(k * ff).expect("up out");
    let mut d_shu_o = exec.alloc(ff_s).expect("sh up out");
    let mut d_proj = exec.alloc(embd).expect("proj");
    exec.nvf4_moe_up_relu2(&up, &d_idx, &d_x, &mut d_up_o, k, 1)
        .expect("up pair");
    exec.nvf4_moe_up_relu2(&sh_up, &d_sh_idx, &d_x, &mut d_shu_o, 1, 1)
        .expect("sh up pair");
    exec.nvf4_moe_down_acc(&dn, &d_idx, &d_w, &d_up_o, &mut d_proj, None, k, 1, false)
        .expect("dn pair");
    exec.nvf4_moe_down_acc(
        &sh_dn,
        &d_sh_idx,
        &d_sh_w,
        &d_shu_o,
        &mut d_proj,
        None,
        1,
        1,
        true,
    )
    .expect("sh dn pair");
    let mut d_y_old = exec.to_device(&residual).expect("res old");
    exec.add(&mut d_y_old, &d_proj, embd).expect("res add");
    let y_old = exec.to_host(&d_y_old).expect("y old");
    let up_pair = exec.to_host(&d_up_o).expect("up host");
    let shu_pair = exec.to_host(&d_shu_o).expect("shu host");

    // New: fused up - split-K x4 regrouping, gated rel-to-rms vs the pair
    let mut d_act = exec.alloc(k * ff + ff_s).expect("act");
    exec.nvf4_moe_up_relu2_mt(&up, &sh_up, &d_idx, &d_x, &mut d_act, k)
        .expect("up mt");
    let act = exec.to_host(&d_act).expect("act host");
    let pair_all: Vec<f32> = up_pair.iter().chain(shu_pair.iter()).copied().collect();
    let act_rms = (pair_all
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        / pair_all.len() as f64)
        .sqrt();
    let mut up_rel = 0f64;
    for r in 0..k * ff + ff_s {
        let rel = (act[r] as f64 - pair_all[r] as f64).abs() / act_rms;
        up_rel = up_rel.max(rel);
        assert!(
            rel < 1e-5,
            "up_mt act row {r}: {} vs {} (rel-to-rms {rel:.3e})",
            act[r],
            pair_all[r]
        );
    }

    // New: slot-split down - each slot plane vs a k=1 down_acc launch fed
    // the same act row (rel-to-rms: the split-K fold regroups the sum)
    let mut d_part = exec.alloc((k + 1) * embd).expect("part");
    exec.nvf4_moe_down_part(&dn, &sh_dn, &d_idx, &d_w, &d_act, &mut d_part, k)
        .expect("dn mt");
    let part = exec.to_host(&d_part).expect("part host");
    for slot in 0..=k {
        let (plane, e, wt, xr, kk): (&_, u32, f32, &[f32], usize) = if slot < k {
            (
                &dn,
                idx[slot],
                topk_w[slot],
                &act[slot * ff..(slot + 1) * ff],
                ff,
            )
        } else {
            (&sh_dn, 0, 1.0, &act[k * ff..], ff_s)
        };
        let d_i1 = exec.to_device_u32(&[e]).expect("i1");
        let d_w1 = exec.to_device(&[wt]).expect("w1");
        let d_xr = exec.to_device(xr).expect("xr");
        let mut d_ref = exec.alloc(embd).expect("ref");
        exec.nvf4_moe_down_acc(plane, &d_i1, &d_w1, &d_xr, &mut d_ref, None, 1, 1, false)
            .expect("k1 ref");
        let href = exec.to_host(&d_ref).expect("ref host");
        let _ = kk;
        let srms =
            (href.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / embd as f64).sqrt();
        for r in 0..embd {
            let (a, b) = (part[slot * embd + r] as f64, href[r] as f64);
            let rel = (a - b).abs() / srms;
            assert!(
                rel < 1e-5,
                "down_part slot {slot} row {r}: {a} vs {b} (rel-to-rms {rel:.3e})"
            );
        }
    }

    // full chain: combine into the residual; deterministic (bit-equal on
    // re-run) and tight vs the pair chain (cross-slot fold order only)
    let mut d_y_new = exec.to_device(&residual).expect("res new");
    exec.moe_slot_combine(&d_part, &mut d_y_new, embd, k + 1, 1)
        .expect("combine");
    let y_new = exec.to_host(&d_y_new).expect("y new");
    let mut d_act2 = exec.alloc(k * ff + ff_s).expect("act2");
    exec.nvf4_moe_up_relu2_mt(&up, &sh_up, &d_idx, &d_x, &mut d_act2, k)
        .expect("up mt 2");
    let mut d_part2 = exec.alloc((k + 1) * embd).expect("part2");
    exec.nvf4_moe_down_part(&dn, &sh_dn, &d_idx, &d_w, &d_act2, &mut d_part2, k)
        .expect("dn mt 2");
    let mut d_y_new2 = exec.to_device(&residual).expect("res new2");
    exec.moe_slot_combine(&d_part2, &mut d_y_new2, embd, k + 1, 1)
        .expect("combine 2");
    let y_new2 = exec.to_host(&d_y_new2).expect("y new2");
    // near-zero outputs make per-element relative error read as cancellation
    // noise; the house drift convention is diff relative to the vector rms
    let rms = (y_old.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / embd as f64).sqrt();
    let mut max_rel = 0f64;
    for r in 0..embd {
        assert_eq!(
            y_new[r].to_bits(),
            y_new2[r].to_bits(),
            "mt chain must be deterministic"
        );
        let (a, b) = (y_new[r] as f64, y_old[r] as f64);
        let rel = (a - b).abs() / rms;
        max_rel = max_rel.max(rel);
        assert!(
            rel < 1e-5,
            "mt chain vs pair chain at {r}: {a} vs {b} (rel-to-rms {rel:.3e})"
        );
    }
    println!(
        "nvf4 moe mt chain: up max rel-to-rms {up_rel:.3e}, {} slot planes gated, chain max rel-to-rms {max_rel:.3e}, deterministic",
        k + 1
    );
}

/// 7. The bf16-resident decode lane: the four attention
///    planes served from checkpoint bytes must match the f32-widened matvec they
///    replace. Products are identical (bf16 -> f32 widening is exact); only the
///    warp-local summation grouping differs (16-element packs / 4-warp combine
///    vs 256-thread stride / 8-warp combine) - the sanctioned reorder class, so
///    the gate is rel-to-rms + determinism, and the embed-gather twin (no
///    summation at all) must be BIT-exact.
///    The bf16 mma K-split arm sums partials through one static
///    device plane (pd_bf16ks_part) under the pack's single-engine-stream
///    serving contract - the same contract pd_smp_scr carries, and the same
///    test hazard: two tests driving bf16 GEMMs at batch > 8 on their own
///    streams would interleave partial writes. Serialize them the way
///    gpu_sample_rows does.
fn bf16ks_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

#[test]
fn bf16_attn_planes_match_f32_matvec() {
    let _scr = bf16ks_lock();
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    assert!(
        exec.has_bf16_dense(),
        "pack has no bf16 dense lane - rebuild packs/cuda"
    );
    assert!(
        exec.has_embed_gather_bf16(),
        "pack has no bf16 embed gather - rebuild packs/cuda"
    );

    fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    }

    // layer 5 is the first attention layer; q_proj [4096, 2688] and
    // o_proj [2688, 4096] cover both serve shapes (k/v are q_proj's walk
    // at fewer rows)
    for (name, out_dim, in_dim) in [
        (
            "backbone.layers.5.mixer.q_proj.weight",
            4096usize,
            2688usize,
        ),
        ("backbone.layers.5.mixer.o_proj.weight", 2688, 4096),
    ] {
        let (t, raw) = st.bytes(name).expect("plane present");
        assert_eq!(
            t.dtype,
            paddock_models::safetensors::StDtype::Bf16,
            "{name} not bf16"
        );
        assert_eq!(raw.len(), out_dim * in_dim * 2, "{name} size");

        let w_f32 = exec.to_device(&bf16_to_f32(raw)).expect("f32 plane");
        let qt = paddock_engine::gpu::QuantTensor {
            bytes: exec.to_device_u8(raw).expect("bf16 plane"),
            ty: paddock_models::ggml_type::GgmlType::Bf16,
            dims: vec![in_dim, out_dim],
        };
        let d_x = exec.to_device(&det(in_dim, 91)).expect("x");

        // the exact serve arms: matvec_f32_batch(batch=1) vs bf16_gemv
        let mut d_y32 = exec.alloc(out_dim).expect("y32");
        exec.matvec_f32_raw(&w_f32, in_dim, out_dim, &d_x, &mut d_y32, 1)
            .expect("f32 arm");
        let y32 = exec.to_host(&d_y32).expect("y32 host");
        let mut d_yb = exec.alloc(out_dim).expect("yb");
        exec.bf16_gemv(&qt, None, &d_x, &mut d_yb)
            .expect("bf16 arm");
        let yb = exec.to_host(&d_yb).expect("yb host");
        let mut d_yb2 = exec.alloc(out_dim).expect("yb2");
        exec.bf16_gemv(&qt, None, &d_x, &mut d_yb2)
            .expect("bf16 arm 2");
        let yb2 = exec.to_host(&d_yb2).expect("yb2 host");

        let rms =
            (y32.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / out_dim as f64).sqrt();
        let mut max_rel = 0f64;
        for o in 0..out_dim {
            assert_eq!(
                yb[o].to_bits(),
                yb2[o].to_bits(),
                "{name} row {o}: not deterministic"
            );
            let rel = (yb[o] as f64 - y32[o] as f64).abs() / rms;
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1e-5,
                "{name} row {o}: bf16 {} vs f32 {} (rel-to-rms {rel:.3e})",
                yb[o],
                y32[o]
            );
        }

        // embed-gather twin over the same rows (a projection plane is just a
        // row table to a gather): x1.0 scale on exactly-widened values must
        // land bit-identical to the f32 gather
        let toks = [0u32, 7, 1234, out_dim as u32 - 1];
        let d_toks = exec.to_device_u32(&toks).expect("toks");
        let mut d_g32 = exec.alloc(toks.len() * in_dim).expect("g32");
        exec.embed_gather_batch(&w_f32, &d_toks, &mut d_g32, in_dim, toks.len())
            .expect("g f32");
        let g32 = exec.to_host(&d_g32).expect("g32 host");
        let mut d_gb = exec.alloc(toks.len() * in_dim).expect("gb");
        exec.embed_gather_bf16(&qt, &d_toks, &mut d_gb, in_dim, toks.len(), 1.0)
            .expect("g bf16");
        let gb = exec.to_host(&d_gb).expect("gb host");
        for (i, (a, b)) in gb.iter().zip(&g32).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{name} gather elem {i}: {a} vs {b}"
            );
        }
        println!("{name}: gemv max rel-to-rms {max_rel:.3e}, gather bit-exact");
    }
}

/// The chunked SSD prefill scan (mamba/ssd.cuh) writes its 5-kernel chain
/// through pack-STATIC device scratch, so two concurrent SSD launches from
/// parallel test threads would interleave. Serialize the SSD-regime tests
/// the way bf16ks/gpu_sample_rows do.
fn ssd_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// The SSD-regime twin of mamba2_scan_seq_matches_f64_reference: T=300
/// (above the measured MIN_T=256 election floor) crosses two full
/// 128-token chunks into a 44-token partial chunk, so the chunk chain, the
/// Gram masking and the pad-identity all sit in the checked span. The y
/// tolerance is 1e-4 (vs the serial gate's 1e-5): with
/// this test's decay depths the inclusive log-decay cumsum reaches O(100),
/// and exp(fl(cum_t) - fl(cum_s)) carries the cumsum's absolute rounding
/// as relative error - conditioning of the reformulation, not a defect
/// (real prompts sit shallower). States keep the abs-scaled 1e-4 gate.
#[test]
fn mamba2_scan_seq_ssd_matches_f64_reference() {
    let _scr = ssd_lock();
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() {
        common::missing("pack has no mamba2 kernels");
        return;
    }
    let t_len = 300usize;
    let state0 = det(H * HD * S, 30);
    let xbc = det(t_len * CONV_DIM, 31);
    let dt_stride = CONV_DIM + H;
    let dt_raw = det(t_len * dt_stride, 32);
    let a: Vec<f32> = det(H, 33).iter().map(|v| -v.abs() - 0.05).collect();
    let d: Vec<f32> = det(H, 34);
    let dt_bias: Vec<f32> = det(H, 35);

    let mut dev_img = vec![0f32; H * HD * S];
    for h in 0..H {
        for i in 0..HD {
            for j in 0..S {
                dev_img[(h * S + j) * HD + i] = state0[(h * HD + i) * S + j];
            }
        }
    }
    let mut d_state = exec.to_device(&dev_img).expect("state");
    let d_xbc = exec.to_device(&xbc).expect("xbc");
    let d_dt = exec.to_device(&dt_raw).expect("dt");
    let d_a = exec.to_device(&a).expect("a");
    let d_d = exec.to_device(&d).expect("d");
    let d_db = exec.to_device(&dt_bias).expect("db");
    let mut d_y = exec.alloc(t_len * D_INNER).expect("y");
    exec.mamba2_scan_seq(
        &mut d_state,
        &d_xbc,
        &d_dt,
        0,
        dt_stride,
        &d_a,
        &d_d,
        &d_db,
        &mut d_y,
        t_len,
        H,
        HD,
        S,
        G,
    )
    .expect("scan ssd");
    let y = exec.to_host(&d_y).expect("y host");
    let state = exec.to_host(&d_state).expect("state host");

    // determinism first: the chain is fixed-order, no atomics - a second
    // run from the same seed must be bit-identical in y AND state
    let mut d_state2 = exec.to_device(&dev_img).expect("state2");
    let mut d_y2 = exec.alloc(t_len * D_INNER).expect("y2");
    exec.mamba2_scan_seq(
        &mut d_state2,
        &d_xbc,
        &d_dt,
        0,
        dt_stride,
        &d_a,
        &d_d,
        &d_db,
        &mut d_y2,
        t_len,
        H,
        HD,
        S,
        G,
    )
    .expect("scan ssd rerun");
    let y2 = exec.to_host(&d_y2).expect("y2 host");
    let st2 = exec.to_host(&d_state2).expect("st2 host");
    assert!(
        y2.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()),
        "SSD scan y not deterministic across runs"
    );
    assert!(
        st2.iter()
            .zip(&state)
            .all(|(p, q)| p.to_bits() == q.to_bits()),
        "SSD scan state not deterministic across runs"
    );

    // f64 reference (identical loop to the serial gate)
    let mut st: Vec<f64> = state0.iter().map(|v| *v as f64).collect();
    let mut worst = 0f64;
    for t in 0..t_len {
        let row = &xbc[t * CONV_DIM..(t + 1) * CONV_DIM];
        for h in 0..H {
            let g = h / (H / G);
            let v = (dt_raw[t * dt_stride + h] + dt_bias[h]) as f64;
            let dt = if v <= 20.0 { v.exp().ln_1p() } else { v };
            let decay = (dt * a[h] as f64).exp();
            for i in 0..HD {
                let x_ti = row[h * HD + i] as f64;
                let contrib = dt * x_ti;
                let mut acc = 0f64;
                let mut mag = 0f64;
                for j in 0..S {
                    let sb = row[D_INNER + g * S + j] as f64;
                    let sc = row[D_INNER + (G + g) * S + j] as f64;
                    let idx = (h * HD + i) * S + j;
                    st[idx] = decay * st[idx] + contrib * sb;
                    acc += st[idx] * sc;
                    mag += (st[idx] * sc).abs();
                }
                let want = acc + d[h] as f64 * x_ti;
                let got = y[t * D_INNER + h * HD + i] as f64;
                let rel = (got - want).abs() / mag.max(1e-6);
                worst = worst.max(rel);
                assert!(
                    rel < 1e-4,
                    "ssd y[t{t} h{h} i{i}]: got {got} want {want} rel {rel}"
                );
            }
        }
    }
    for h in 0..H {
        for i in 0..HD {
            for j in 0..S {
                let g0 = state[(h * S + j) * HD + i];
                let w0 = st[(h * HD + i) * S + j];
                let err = (g0 as f64 - w0).abs();
                assert!(
                    err < 1e-4 * (1.0 + w0.abs()),
                    "ssd state[h{h} i{i} j{j}]: got {g0} want {w0}"
                );
            }
        }
    }
    println!("SSD scan: T={t_len} worst y rel {worst:.2e}, deterministic");
}

/// SSD f32-vs-f16 twin claim on an f16-exact seed (the f16 class gate's
/// step-1 argument carried to the chunked path): both arms load identical
/// values, all arithmetic is f32 in shared scratch, and only the FINAL
/// arena store rounds - so y must be BIT-IDENTICAL between the f32 and
/// f16 SSD arms, and the f16 state must be exactly rn(f32 state).
#[test]
fn mamba2_scan_seq_ssd_f16_twin_matches_f32() {
    use half::f16 as h16;
    let _scr = ssd_lock();
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() || !exec.has_mamba2_f16_state() {
        common::missing("pack has no f16 mamba2 kernels");
        return;
    }
    let t_len = 300usize; // >= the SSD threshold (256), partial third chunk
    let state_elems = H * HD * S;
    // f16-exact seed (k/256, |k|<=200): both arenas hold identical numbers
    let seed: Vec<f32> = (0..state_elems)
        .map(|i| ((i % 401) as f32 - 200.0) / 256.0)
        .collect();
    let xbc = det(t_len * CONV_DIM, 40);
    let dt_stride = CONV_DIM + H;
    let dt_raw = det(t_len * dt_stride, 41);
    let a: Vec<f32> = det(H, 42).iter().map(|v| -v.abs() - 0.05).collect();
    let d: Vec<f32> = det(H, 43);
    let dt_bias: Vec<f32> = det(H, 44);

    let d_xbc = exec.to_device(&xbc).expect("xbc");
    let d_dt = exec.to_device(&dt_raw).expect("dt");
    let d_a = exec.to_device(&a).expect("a");
    let d_d = exec.to_device(&d).expect("d");
    let d_db = exec.to_device(&dt_bias).expect("db");

    let mut s32 = exec.to_device(&seed).expect("s32");
    let mut y32 = exec.alloc(t_len * D_INNER).expect("y32");
    exec.mamba2_scan_seq(
        &mut s32, &d_xbc, &d_dt, 0, dt_stride, &d_a, &d_d, &d_db, &mut y32, t_len, H, HD, S, G,
    )
    .expect("ssd f32");
    let h32 = exec.to_host(&y32).expect("y32 host");
    let st32 = exec.to_host(&s32).expect("s32 host");

    let seed16: Vec<h16> = seed.iter().map(|&v| h16::from_f32(v)).collect();
    let mut s16 = exec.alloc_f16(state_elems).expect("s16");
    {
        let mut v = s16.try_slice_mut(0..state_elems).expect("s16 view");
        exec.stream.memcpy_htod(&seed16, &mut v).expect("htod s16");
    }
    let mut y16 = exec.alloc(t_len * D_INNER).expect("y16");
    exec.mamba2_scan_seq_at_f16(
        &mut s16, 0, &d_xbc, 0, &d_dt, 0, dt_stride, &d_a, &d_d, &d_db, &mut y16, 0, t_len, H, HD,
        S, G,
    )
    .expect("ssd f16");
    let h16y = exec.to_host(&y16).expect("y16 host");
    assert_eq!(
        h32, h16y,
        "SSD y must be BIT-IDENTICAL between f32 and f16 arms on an f16-exact \
         seed - the chain is f32 in scratch and only the arena store rounds"
    );
    let st16 = exec.to_host_f16_len(&s16, state_elems).expect("s16 host");
    for (e, (w, g)) in st32.iter().zip(&st16).enumerate() {
        assert_eq!(
            h16::from_f32(*w).to_bits(),
            g.to_bits(),
            "SSD f16 state elem {e} must be exactly rn(f32 state): {w} vs {g:?}"
        );
    }
    println!("SSD f16 twin: y bit-identical, state = rn(f32), T={t_len}");
}

/// Rung 8 stage A: the batched decode steps must be BIT-exact per row
/// against loops of their serial twins over shuffled (distinct) slot ids -
/// the continuous-batching tick's contract. Real nemotron mamba geometry.
/// (A rung-19 j-split briefly relaxed the y gate to regroup-class; the split
/// was falsified dead-even by bench/nemo_scan_split_bench.cu and reverted,
/// so bit-exactness is back. The determinism re-run below stays as a
/// standing guard for any future regrouping attempt.)
#[test]
fn batched_decode_steps_match_serial_loops() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() || !exec.has_mamba2_batch() {
        common::missing("pack has no batched mamba2 step kernels");
        return;
    }
    const NSLOTS: usize = 4;
    const B: usize = 3;
    let slots = [2u32, 0, 3];
    // fused rows [B, stride]: [pad | x B C (conv span) | dt lanes]
    let x_off = 16usize;
    let stride = x_off + CONV_DIM + H;
    let rows = det(B * stride, 11);
    let arena0 = det(NSLOTS * (K_CONV - 1) * CONV_DIM, 12);
    let w = det(CONV_DIM * K_CONV, 13);
    let b = det(CONV_DIM, 14);

    // ---- conv step over the windows arena ----------------------------------
    let mut d_arena = exec.to_device(&arena0).expect("arena");
    let d_rows = exec.to_device(&rows).expect("rows");
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let d_w = exec.to_device(&w).expect("w");
    let d_b = exec.to_device(&b).expect("b");
    let mut d_out = exec.alloc(B * CONV_DIM).expect("out");
    exec.mamba_conv_step_batch(
        &mut d_arena,
        &d_rows,
        x_off,
        stride,
        &d_slots,
        &d_w,
        &d_b,
        &mut d_out,
        CONV_DIM,
        K_CONV,
        B,
    )
    .expect("conv batch");
    let got_arena = exec.to_host(&d_arena).expect("arena out");
    let got_out = exec.to_host(&d_out).expect("out");

    let wlen = (K_CONV - 1) * CONV_DIM;
    let mut wins: Vec<_> = (0..NSLOTS)
        .map(|s| {
            exec.to_device(&arena0[s * wlen..(s + 1) * wlen])
                .expect("win")
        })
        .collect();
    let mut d_o1 = exec.alloc(CONV_DIM).expect("o1");
    for (r, &s) in slots.iter().enumerate() {
        exec.mamba_conv_step(
            &mut wins[s as usize],
            &d_rows,
            r * stride + x_off,
            &d_w,
            &d_b,
            &mut d_o1,
            CONV_DIM,
            K_CONV,
        )
        .expect("conv serial");
        let o1 = exec.to_host(&d_o1).expect("o1");
        assert!(
            o1.iter()
                .zip(&got_out[r * CONV_DIM..(r + 1) * CONV_DIM])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "conv out row {r} not bit-exact"
        );
    }
    for s in 0..NSLOTS {
        let ws = exec.to_host(&wins[s]).expect("win out");
        assert!(
            ws.iter()
                .zip(&got_arena[s * wlen..(s + 1) * wlen])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "conv window slot {s} not bit-exact"
        );
    }

    // ---- scan step over the states arena -----------------------------------
    let st_len = H * HD * S;
    let states0 = det(NSLOTS * st_len, 21);
    let mut d_states = exec.to_device(&states0).expect("states");
    let a = det(H, 22);
    let dsk = det(H, 23);
    let dtb = det(H, 24);
    let d_a = exec.to_device(&a).expect("a");
    let d_d = exec.to_device(&dsk).expect("d");
    let d_dtb = exec.to_device(&dtb).expect("dtb");
    let mut d_y = exec.alloc(B * D_INNER).expect("y");
    exec.mamba2_scan_step_batch(
        &mut d_states,
        &d_out,
        &d_rows,
        x_off + CONV_DIM,
        stride,
        &d_slots,
        &d_a,
        &d_d,
        &d_dtb,
        &mut d_y,
        B,
        H,
        HD,
        S,
        G,
    )
    .expect("scan batch");
    let got_states = exec.to_host(&d_states).expect("states out");
    let got_y = exec.to_host(&d_y).expect("y out");

    let mut sstates: Vec<_> = (0..NSLOTS)
        .map(|s| {
            exec.to_device(&states0[s * st_len..(s + 1) * st_len])
                .expect("st")
        })
        .collect();
    let mut d_y1 = exec.alloc(D_INNER).expect("y1");
    for (r, &s) in slots.iter().enumerate() {
        let xrow = exec
            .to_device(&got_out[r * CONV_DIM..(r + 1) * CONV_DIM])
            .expect("xr");
        exec.mamba2_scan_seq(
            &mut sstates[s as usize],
            &xrow,
            &d_rows,
            r * stride + x_off + CONV_DIM,
            stride,
            &d_a,
            &d_d,
            &d_dtb,
            &mut d_y1,
            1,
            H,
            HD,
            S,
            G,
        )
        .expect("scan serial");
        let y1 = exec.to_host(&d_y1).expect("y1");
        assert!(
            y1.iter()
                .zip(&got_y[r * D_INNER..(r + 1) * D_INNER])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "scan y row {r} not bit-exact"
        );
    }
    for s in 0..NSLOTS {
        let st = exec.to_host(&sstates[s]).expect("st out");
        assert!(
            st.iter()
                .zip(&got_states[s * st_len..(s + 1) * st_len])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "scan state slot {s} not bit-exact"
        );
    }

    // determinism: the batched step must be run-to-run deterministic (no
    // atomics, no launch-order sensitivity) - a second run from the same
    // seed must be bit-identical in both y and states. Kept from the
    // reverted rung-19 gate as a standing invariant.
    let mut d_states2 = exec.to_device(&states0).expect("states2");
    let mut d_y2 = exec.alloc(B * D_INNER).expect("y2");
    exec.mamba2_scan_step_batch(
        &mut d_states2,
        &d_out,
        &d_rows,
        x_off + CONV_DIM,
        stride,
        &d_slots,
        &d_a,
        &d_d,
        &d_dtb,
        &mut d_y2,
        B,
        H,
        HD,
        S,
        G,
    )
    .expect("scan batch rerun");
    let y2 = exec.to_host(&d_y2).expect("y2");
    let st2 = exec.to_host(&d_states2).expect("st2");
    assert!(
        y2.iter()
            .zip(&got_y)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "batched scan y not deterministic across runs"
    );
    assert!(
        st2.iter()
            .zip(&got_states)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "batched scan states not deterministic across runs"
    );

    // ---- row-batched nvf4 GEMV (cc12-gated) --------------------------------
    // n=64 exercises the mr twin's BN=1 (thin) arm; n=4232 crosses the
    // launcher's >=4096 width gate into the BN=4 wide arm with a ragged
    // 8-not-32 output tail - both must stay bit-exact vs serial.
    if exec.has_nvf4_ckpt() {
        for n in [64usize, 4232] {
            let k = 256usize;
            let mut lcg = 7u64;
            let mut nb = || {
                lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
                (lcg >> 33) as u8
            };
            let packed: Vec<u8> = (0..n * k / 2).map(|_| nb()).collect();
            let scales: Vec<u8> = (0..n * k / 16).map(|_| 0x28 + (nb() % 0x20)).collect();
            let plane = exec
                .nvf4_upload(&packed, &scales, 0.37, n, k)
                .expect("plane");
            // batches straddling the mr twin's 16-row group: masked
            // partial group, one exact group, two groups with a 1-row tail, two
            // exact groups - every arm must stay bit-exact vs the serial GEMV.
            // 2..=8 ride the small-batch arms (4- and 8-row tiles, KS=2, BN=2
            // where the grid covers the die): full and ragged tiles of each
            for bt in [2, B, 4, 5, 8, 16, 17, 32] {
                let xs = det(bt * k, 31);
                let d_xs = exec.to_device(&xs).expect("xs");
                let mut d_yb = exec.alloc(bt * n).expect("yb");
                exec.nvf4_gemv_batch(&plane, &d_xs, &mut d_yb, None, bt)
                    .expect("gemv batch");
                let got_yb = exec.to_host(&d_yb).expect("yb out");
                let mut d_ys = exec.alloc(n).expect("ys");
                for r in 0..bt {
                    let xr = exec.to_device(&xs[r * k..(r + 1) * k]).expect("xr");
                    exec.nvf4_gemv(&plane, &xr, &mut d_ys, None)
                        .expect("gemv serial");
                    let ys = exec.to_host(&d_ys).expect("ys");
                    assert!(
                        ys.iter()
                            .zip(&got_yb[r * n..(r + 1) * n])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "gemv batch n {n} bt {bt} row {r} not bit-exact"
                    );
                }
            }
        }
    } else {
        common::missing("pack has no nvf4 consumers for the gemv-batch leg");
    }

    println!(
        "batched decode steps: conv/scan/gemv all bit-exact vs serial loops \
         + batched scan deterministic across runs"
    );
}

// ---- shared fold-in topk (k+ns-wide rows) -----------------------
// The _sh variant must write exactly the plain kernel's routed picks and
// normalized weights in columns 0..k (same selection + same gather-sum
// order, so bitwise equality is the bar) plus the constant shared
// pseudo-expert picks (sh0+i, w=1.0) in columns k..k+ns.
#[test]
fn moe_topk_sigmoid_sh_matches_plain_plus_constants() {
    let Some(exec) = common::gpu() else { return };
    let (ne, k, ns, batch) = (128usize, 6usize, 2usize, 5usize);
    let logits = det(batch * ne, 90);
    let bias = det(ne, 91);
    let d_l = exec.to_device(&logits).expect("logits");
    let d_b = exec.to_device(&bias).expect("bias");
    let mut d_i6 = exec.alloc_u32(batch * k).expect("i6");
    let mut d_w6 = exec.alloc(batch * k).expect("w6");
    let mut d_i8 = exec.alloc_u32(batch * (k + ns)).expect("i8");
    let mut d_w8 = exec.alloc(batch * (k + ns)).expect("w8");
    exec.moe_topk_sigmoid_batch(&d_l, &d_b, 2.5, ne, k, &mut d_i6, &mut d_w6, batch)
        .expect("plain topk");
    if exec
        .moe_topk_sigmoid_batch_sh(&d_l, &d_b, 2.5, ne, k, ns, ne, &mut d_i8, &mut d_w8, batch)
        .is_err()
    {
        common::missing("pack has no moe_topk_sigmoid_batch_sh");
        return;
    }
    let i6 = exec.to_host_u32(&d_i6).expect("i6 host");
    let w6 = exec.to_host(&d_w6).expect("w6 host");
    let i8v = exec.to_host_u32(&d_i8).expect("i8 host");
    let w8 = exec.to_host(&d_w8).expect("w8 host");
    for b in 0..batch {
        for s in 0..k {
            assert_eq!(i8v[b * (k + ns) + s], i6[b * k + s], "idx row {b} slot {s}");
            assert_eq!(
                w8[b * (k + ns) + s].to_bits(),
                w6[b * k + s].to_bits(),
                "w row {b} slot {s}"
            );
        }
        for s in 0..ns {
            assert_eq!(
                i8v[b * (k + ns) + k + s],
                (ne + s) as u32,
                "sh idx row {b} slot {s}"
            );
            assert_eq!(w8[b * (k + ns) + k + s], 1.0f32, "sh w row {b} slot {s}");
        }
    }
    println!("topk_sh: routed columns bitwise == plain, shared columns constant");
}

// ---- tensor-core NVFP4 head (bf16 mma class) -------------
// nvf4_gemm_tc is a numeric CLASS change vs the scalar lane - the f32->bf16
// activation cast plus mma k16-tree reassociation - so bit-equality on
// general inputs is not the bar. The gate pins it two ways instead:
//   lattice leg: scales are powers of two in [1/4, 4], x rows are small
//     integers, weights arbitrary nibbles. Every product is then a multiple
//     of 2^-5 with row sums far below 2^24, so every accumulation order is
//     exact in f32 and the bf16 x cast is exact - the tc result must be
//     BIT-EXACT vs the scalar batch GEMV. Any fragment-layout or staging
//     bug lands whole wrong values here, not rounding noise.
//   gaussian leg: realistic values; the bf16 cast contributes ~2^-9
//     relative per element, so a 1% of-row-max tolerance is generous while
//     still orders of magnitude tighter than any real defect.
// Geometry stresses the tile edges: n=4232 is ragged vs the BM=128 row tile
// (33 full + an 8-row tail), k=2688 is 21 exact KT=128 stages while k=2624
// leaves a 64-wide ragged final KT=128 stage (%32 upload gate held), and batches 1/9/32/33 cover a
// zero-padded column tile, a partial tile, an exact tile, and a second
// grid.y tile of width one.
//
// Since the lm_head rung the wrapper routes K%128==0 to the
// PERSISTENT raw-ring twin (pd_nvf4_gemm_tcp_kernel), so the k=2688 legs
// gate that arm while k=2624 keeps gating the one-shot tc fallback. The
// twins' direct bit-identity contract (same decode, same mma chain) is
// memcmp'd at the vocab shape in bench/nv4mr_head_probe.cu.
#[test]
fn nvf4_gemm_tc_matches_scalar_class() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_nvf4_gemm_tc() {
        common::missing("pack has no nvf4_gemm_tc");
        return;
    }
    let n = 4232usize;
    let mut lcg = 11u64;
    let mut nb = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        (lcg >> 33) as u8
    };
    for k in [2688usize, 2624] {
        let packed: Vec<u8> = (0..n * k / 2).map(|_| nb()).collect();
        // lattice leg scales: e4m3 powers of two 0x28..0x48 = 0.25..4.0
        let scales: Vec<u8> = (0..n * k / 16).map(|_| 0x28 + 8 * (nb() % 5)).collect();
        let plane = exec
            .nvf4_upload(&packed, &scales, 0.5, n, k)
            .expect("plane");
        for bt in [1usize, 9, 32, 33] {
            // integers in [-4, 4]: exact in bf16, exact partial sums
            let xs: Vec<f32> = (0..bt * k).map(|_| (nb() % 9) as f32 - 4.0).collect();
            let d_xs = exec.to_device(&xs).expect("xs");
            let mut d_yt = exec.alloc(bt * n).expect("yt");
            let mut d_yr = exec.alloc(bt * n).expect("yr");
            exec.nvf4_gemm_tc(&plane, &d_xs, &mut d_yt, None, bt)
                .expect("tc");
            exec.nvf4_gemv_batch(&plane, &d_xs, &mut d_yr, None, bt)
                .expect("ref");
            let yt = exec.to_host(&d_yt).expect("yt host");
            let yr = exec.to_host(&d_yr).expect("yr host");
            for i in 0..bt * n {
                assert!(
                    yt[i].to_bits() == yr[i].to_bits(),
                    "lattice k {k} bt {bt} elem {i}: tc {} vs ref {}",
                    yt[i],
                    yr[i]
                );
            }
        }
    }
    // gaussian leg: realistic scales + activations at the serve batch
    let (n, k, bt) = (4232usize, 2688usize, 32usize);
    let packed: Vec<u8> = (0..n * k / 2).map(|_| nb()).collect();
    let scales: Vec<u8> = (0..n * k / 16).map(|_| 0x28 + (nb() % 0x20)).collect();
    let plane = exec
        .nvf4_upload(&packed, &scales, 0.37, n, k)
        .expect("plane");
    let xs = det(bt * k, 47);
    let d_xs = exec.to_device(&xs).expect("xs");
    let mut d_yt = exec.alloc(bt * n).expect("yt");
    let mut d_yr = exec.alloc(bt * n).expect("yr");
    exec.nvf4_gemm_tc(&plane, &d_xs, &mut d_yt, None, bt)
        .expect("tc");
    exec.nvf4_gemv_batch(&plane, &d_xs, &mut d_yr, None, bt)
        .expect("ref");
    let yt = exec.to_host(&d_yt).expect("yt host");
    let yr = exec.to_host(&d_yr).expect("yr host");
    for r in 0..bt {
        let row = &yr[r * n..(r + 1) * n];
        let m = row.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let tol = 1e-2 * m + 1e-3;
        for i in 0..n {
            let d = (yt[r * n + i] - row[i]).abs();
            assert!(
                d <= tol,
                "gaussian row {r} elem {i}: tc {} vs ref {} (tol {tol})",
                yt[r * n + i],
                row[i]
            );
        }
    }
    println!(
        "nvf4_gemm_tc: lattice bit-exact vs scalar class, gaussian within bf16-cast tolerance"
    );
}

/// Tile-major plane residency (the lm_head repack rung): every
/// consumer class over a `nvf4_upload_tiled` plane must be BIT-identical to
/// the same class over the row-major upload of the same triple - the `_tm`
/// twins only move the weight/scale addressing, never the walk. n is not a
/// multiple of 128 (4232 = 33*128 + 8) so the zero-filled pad rows are
/// live in the tc arm's full-tile stages; random e4m3 scales cover the
/// scale-record addressing (a constant scale plane repacks to itself).
#[test]
fn nvf4_tm_plane_matches_rowmajor() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_nvf4_tm() {
        common::missing("pack has no tile-major nvf4 twins - rebuild packs/cuda");
        return;
    }
    let (n, k) = (4232usize, 2688usize);
    let mut lcg = 23u64;
    let mut nb = || {
        lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
        (lcg >> 33) as u8
    };
    let packed: Vec<u8> = (0..n * k / 2).map(|_| nb()).collect();
    let scales: Vec<u8> = (0..n * k / 16).map(|_| 0x28 + (nb() % 0x20)).collect();
    let rm = exec
        .nvf4_upload(&packed, &scales, 0.37, n, k)
        .expect("row-major");
    let tm = exec
        .nvf4_upload_tiled(&packed, &scales, 0.37, n, k)
        .expect("tiled");
    assert_eq!(rm.layout, paddock_engine::gpu::Nvf4Layout::Row);
    assert_eq!(tm.layout, paddock_engine::gpu::Nvf4Layout::Tiled);
    // fragment plane: same triple, mma-fragment-ordered blocks
    let tf = exec.has_nvf4_tf().then(|| {
        exec.nvf4_upload_frag(&packed, &scales, 0.37, n, k)
            .expect("frag")
    });
    let planes: Vec<(&str, &paddock_engine::gpu::Nvf4Plane)> = std::iter::once(("tm", &tm))
        .chain(tf.iter().map(|p| ("tf", p)))
        .collect();

    for (name, pl) in &planes {
        // batch lane: bt=1 rides the per-row twin, bt>=2 the mr twin (its
        // 4- and 8-row small-batch tiles included)
        for bt in [1usize, 2, 3, 4, 6, 8, 9, 32] {
            let xs = det(bt * k, 0x7a0 + bt as u64);
            let d_xs = exec.to_device(&xs).expect("xs");
            let mut d_yr = exec.alloc(bt * n).expect("yr");
            let mut d_yt = exec.alloc(bt * n).expect("yt");
            exec.nvf4_gemv_batch(&rm, &d_xs, &mut d_yr, None, bt)
                .expect("rm batch");
            exec.nvf4_gemv_batch(pl, &d_xs, &mut d_yt, None, bt)
                .expect("twin batch");
            let yr = exec.to_host(&d_yr).expect("yr host");
            let yt = exec.to_host(&d_yt).expect("yt host");
            for i in 0..bt * n {
                assert!(
                    yr[i].to_bits() == yt[i].to_bits(),
                    "gemv_batch bt {bt} elem {i}: rm {} vs {name} {}",
                    yr[i],
                    yt[i]
                );
            }
        }
        // single-row entry: non-row wrappers reroute to the batch twin at 1
        {
            let xs = det(k, 0x7b1);
            let d_xs = exec.to_device(&xs).expect("xs");
            let mut d_yr = exec.alloc(n).expect("yr");
            let mut d_yt = exec.alloc(n).expect("yt");
            exec.nvf4_gemv(&rm, &d_xs, &mut d_yr, None)
                .expect("rm gemv");
            exec.nvf4_gemv(pl, &d_xs, &mut d_yt, None)
                .expect("twin gemv");
            let yr = exec.to_host(&d_yr).expect("yr host");
            let yt = exec.to_host(&d_yt).expect("yt host");
            for i in 0..n {
                assert!(
                    yr[i].to_bits() == yt[i].to_bits(),
                    "gemv elem {i}: rm {} vs {name} {}",
                    yr[i],
                    yt[i]
                );
            }
        }
        // tensor-core head class: every layout's arm is the same math in
        // the same order (tcp REPK for tiled, tcv FRAG for fragment)
        if exec.has_nvf4_gemm_tc() {
            for bt in [9usize, 32] {
                let xs = det(bt * k, 0x7c0 + bt as u64);
                let d_xs = exec.to_device(&xs).expect("xs");
                let mut d_yr = exec.alloc(bt * n).expect("yr");
                let mut d_yt = exec.alloc(bt * n).expect("yt");
                exec.nvf4_gemm_tc(&rm, &d_xs, &mut d_yr, None, bt)
                    .expect("rm tc");
                exec.nvf4_gemm_tc(pl, &d_xs, &mut d_yt, None, bt)
                    .expect("twin tc");
                let yr = exec.to_host(&d_yr).expect("yr host");
                let yt = exec.to_host(&d_yt).expect("yt host");
                for i in 0..bt * n {
                    assert!(
                        yr[i].to_bits() == yt[i].to_bits(),
                        "tc bt {bt} elem {i}: rm {} vs {name} {}",
                        yr[i],
                        yt[i]
                    );
                }
            }
        }
    }
    println!(
        "nvf4 layout twins ({}): gemv/gemv_batch/mr/tc all bit-exact vs row-major",
        planes.iter().map(|(n, _)| *n).collect::<Vec<_>>().join("+")
    );
}

/// Thin-k/v rung: the fused q|k|v decode GEMM must be
/// BIT-identical, per out-row, to the plain batched GEMM over the same
/// concatenated [q;k;v] plane - the segmented store only reroutes the
/// epilogue, and mma configs never reorder the per-element k-walk. Real
/// checkpoint planes (layer 5), all three launcher bands + the ragged gridY
/// edge. The 2..=8 band compares against the multi-row GEMV class instead
/// (f32 products on host-bf16-cast x - same products, warp-reduce order)
/// with the plane test's tolerance.
///
/// The reference is the plain GEMM over the fused plane, not the three
/// per-segment GEMMs: the K-split election (`pd_bf16ks_nz`) decides "split
/// or not" from the launch's own grid against the die's SM count, so the
/// 4608-row fused plane and a 256-row k/v segment can land on opposite sides
/// of that early-out. They do on GB10 (48 SMs): the fused plane's 144 CTAs
/// fill the die and stay unsplit while the k segment's 8 CTAs split into 6
/// slabs - a different f32 regroup, 1 ulp apart (fused -1.0522887 vs segment
/// -1.0522894 at bt 9). On 148/188-SM dies both split and the per-segment
/// identity held by coincidence of shape. The per-segment comparison stays
/// as the regroup-class check (rel-to-rms 5e-5) so a thin-plane routing or
/// layout bug still shows as O(1); the plain-over-fused reference is
/// bit-exact at every width - the fused launcher walks the plain one's tier
/// ladder over the fused row count, prefill widths included.
#[test]
fn bf16_qkv_fused_matches_segment_gemms() {
    let _scr = bf16ks_lock();
    let Some(exec) = common::gpu() else { return };
    let Some(st) = checkpoint() else {
        common::missing("no nemotron checkpoint");
        return;
    };
    if !exec.has_bf16_qkv_gemm() {
        common::missing("pack has no bf16_qkv_gemm_mma - rebuild packs/cuda");
        return;
    }
    let (q_dim, kv_dim, hid) = (4096usize, 256usize, 2688usize);
    let seg = |name: &str, out: usize| -> Vec<u8> {
        let (t, raw) = st.bytes(name).expect("plane present");
        assert_eq!(t.dtype, paddock_models::safetensors::StDtype::Bf16);
        assert_eq!(raw.len(), out * hid * 2, "{name} size");
        raw.to_vec()
    };
    let qraw = seg("backbone.layers.5.mixer.q_proj.weight", q_dim);
    let kraw = seg("backbone.layers.5.mixer.k_proj.weight", kv_dim);
    let vraw = seg("backbone.layers.5.mixer.v_proj.weight", kv_dim);
    let mut fraw = qraw.clone();
    fraw.extend_from_slice(&kraw);
    fraw.extend_from_slice(&vraw);
    let qt = |raw: &[u8], out: usize| paddock_engine::gpu::QuantTensor {
        bytes: exec.to_device_u8(raw).expect("plane"),
        ty: paddock_models::ggml_type::GgmlType::Bf16,
        dims: vec![hid, out],
    };
    let fused = qt(&fraw, q_dim + 2 * kv_dim);
    let planes = [qt(&qraw, q_dim), qt(&kraw, kv_dim), qt(&vraw, kv_dim)];

    // launcher bands: <=8 (n32e), <=16 (n32d), <=32 (n32f), <=64 or a
    // sub-wave fat grid (64x64), the fat 128x128 tile (prefill chunks)
    for bt in [2usize, 8, 9, 16, 17, 32, 33, 64, 65, 300, 2048] {
        let x = det(bt * hid, 0x9c0 + bt as u64);
        let d_x = exec.to_device(&x).expect("x");
        let mut d_q = exec.alloc(bt * q_dim).expect("q");
        let mut d_k = exec.alloc(bt * kv_dim).expect("k");
        let mut d_v = exec.alloc(bt * kv_dim).expect("v");
        exec.bf16_qkv_gemm(
            &fused, &d_x, &mut d_q, &mut d_k, &mut d_v, q_dim, kv_dim, bt,
        )
        .expect("fused");
        let got = [
            exec.to_host(&d_q).expect("q host"),
            exec.to_host(&d_k).expect("k host"),
            exec.to_host(&d_v).expect("v host"),
        ];
        // second fused run: deterministic bit-for-bit
        exec.bf16_qkv_gemm(
            &fused, &d_x, &mut d_q, &mut d_k, &mut d_v, q_dim, kv_dim, bt,
        )
        .expect("fused 2");
        let got2 = exec.to_host(&d_q).expect("q host 2");
        for i in 0..bt * q_dim {
            assert_eq!(
                got[0][i].to_bits(),
                got2[i].to_bits(),
                "bt {bt}: not deterministic"
            );
        }
        if bt > 8 {
            // same mma class, same plane, same grid -> same K-split election
            // -> bit-exact: the plain GEMM over the concatenated plane,
            // sliced per segment. Both launchers share one tier ladder
            // (BM=32 to 16 rows, BM=64/BN=32 to 32, 64x64 to 64 rows or a
            // sub-wave fat grid, 128x128 above) over the same row count, so
            // the grids - and the fill early-out - agree at every width.
            let fdim = q_dim + 2 * kv_dim;
            let mut d_f = exec.alloc(bt * fdim).expect("fused y");
            exec.bf16_gemm(&fused, None, &d_x, &mut d_f, bt)
                .expect("plain over fused");
            let full = exec.to_host(&d_f).expect("fused y host");
            for (p, (off, out)) in [(0, q_dim), (q_dim, kv_dim), (q_dim + kv_dim, kv_dim)]
                .into_iter()
                .enumerate()
            {
                for c in 0..bt {
                    for r in 0..out {
                        let want = full[c * fdim + off + r];
                        let have = got[p][c * out + r];
                        assert_eq!(
                            have.to_bits(),
                            want.to_bits(),
                            "bt {bt} plane {p} row {c} col {r}: fused {have} vs plain-over-fused {want}"
                        );
                    }
                }
            }
            // per-segment GEMMs: the same values up to the K-split regroup
            // (the segment's grid may elect a different nz than the fused
            // plane's on a small die - see the doc comment).
            for (p, (plane, out)) in planes.iter().zip([q_dim, kv_dim, kv_dim]).enumerate() {
                let mut d_y = exec.alloc(bt * out).expect("y");
                exec.bf16_gemm(plane, None, &d_x, &mut d_y, bt)
                    .expect("segment");
                let want = exec.to_host(&d_y).expect("y host");
                let rms = (want.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                    / want.len() as f64)
                    .sqrt()
                    .max(1e-20);
                let mut exact = 0usize;
                for i in 0..bt * out {
                    if got[p][i].to_bits() == want[i].to_bits() {
                        exact += 1;
                    }
                    let rel = (got[p][i] as f64 - want[i] as f64).abs() / rms;
                    assert!(
                        rel < 5e-5,
                        "bt {bt} plane {p} elem {i}: fused {} vs segment {} (rel-to-rms {rel:.3e})",
                        got[p][i],
                        want[i]
                    );
                }
                println!(
                    "bf16_qkv bt {bt} plane {p}: {exact}/{} bit-exact vs the segment GEMM",
                    bt * out
                );
            }
        } else {
            // 2..=8: the segment path is the multi-row GEMV (f32 products).
            // Cast x to bf16 on the host first (round-to-nearest-even, the
            // mma's smem-stage cast) so the products match and only the
            // reduce order differs - the bf16-plane test's 1e-5 class.
            let xc: Vec<f32> = x
                .iter()
                .map(|v| {
                    let b = v.to_bits();
                    f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
                })
                .collect();
            let d_xc = exec.to_device(&xc).expect("xc");
            for (p, out) in [q_dim, kv_dim, kv_dim].into_iter().enumerate() {
                let mut d_y = exec.alloc(bt * out).expect("y");
                exec.bf16_gemm(&planes[p], None, &d_xc, &mut d_y, bt)
                    .expect("segment mr");
                let want = exec.to_host(&d_y).expect("y host");
                let rms = (want.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()
                    / want.len() as f64)
                    .sqrt()
                    .max(1e-20);
                for i in 0..bt * out {
                    let rel = (got[p][i] as f64 - want[i] as f64).abs() / rms;
                    // mma tree-reduce vs mr linear accumulate over K=2688:
                    // a genuinely different summation order, so ~1e-5 drift
                    // is the expected class (measured 1.26e-5); a routing
                    // or layout bug shows as O(1), not O(1e-5)
                    assert!(
                        rel < 5e-5,
                        "bt {bt} plane {p} elem {i}: fused {} vs mr {} (rel-to-rms {rel:.3e})",
                        got[p][i],
                        want[i]
                    );
                }
            }
        }
    }
    println!(
        "bf16_qkv fused: bit-exact vs the plain GEMM over the fused plane (bt>8), \
         regroup class vs the segment GEMMs, 1e-5 vs mr class (bt<=8), deterministic"
    );
}

/// Glue rung: the fused add + rmsnorm + nvf4-quantize kernel must
/// be BIT-EXACT to the three launches it replaces, on all three of its
/// outputs. It is elected purely to remove 46 launches per decode tick, so a
/// single differing byte means it is not the same computation and the whole
/// premise is void - hence exact equality, not a tolerance.
///
/// The failure this guards is subtle: the kernel keeps `pd_rmsnorm_batch`'s
/// reduction only if it accumulates in the same elected mode AND the launcher
/// picks the same nth, and keeps the nvf4 16-block scale groups only if the
/// 2-lane pairing survives the strided walk. All of it is silent when wrong -
/// the sums merely regroup, and a regrouped sum moves the f32 mean only when
/// it sits near a rounding boundary. The kernel shipped with the f64
/// accumulate hard-coded after df became the default and this test still
/// passed: on uniform rows the two classes part on 0.54% of rows, and it ran
/// 38. A residual stream's outlier channels lift that to 9.4% (GB10 probe,
/// 2026-09-26), so the rows here carry four and there are ~420 of them.
#[test]
fn add_rmsnorm_quant_nvf4_matches_three_kernel_chain_bitexact() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_add_rmsnorm_quant_nvf4() {
        common::missing("pack has no add_rmsnorm_quant_nvf4_batch");
        return;
    }
    const EMBD: usize = 2688; // nemotron hidden; %32 == 0 keeps the 16-blocks whole
    let eps = 1e-5f32;
    // t=1 and t=32 are served decode widths and t=63 the last one; t=5 catches
    // a row-stride slip that a power-of-two batch would hide; 64 and 256 run
    // the wide-width branch.
    for &t in &[1usize, 5, 32, 63, 64, 256] {
        let mut x = det(t * EMBD, 900 + t as u64);
        for row in x.chunks_mut(EMBD) {
            for c in [7, 311, 1500, 2600] {
                row[c] *= 300.0;
            }
        }
        let proj = det(t * EMBD, 1900 + t as u64);
        let w = det(EMBD, 77);

        // reference: the three launches, in the order forward.rs runs them
        let mut d_x = exec.to_device(&x).expect("x");
        let d_proj = exec.to_device(&proj).expect("proj");
        let d_w = exec.to_device(&w).expect("w");
        let mut d_xn = exec.alloc(t * EMBD).expect("xn");
        let mut d_q = exec.alloc_i8(t * EMBD / 2).expect("q");
        let mut d_s = exec.alloc_u8(t * EMBD / 16).expect("s");
        exec.add(&mut d_x, &d_proj, t * EMBD).expect("add");
        exec.rmsnorm_batch(&d_x, &d_w, &mut d_xn, EMBD, eps, t)
            .expect("norm");
        exec.quantize_nvf4(&d_xn, &mut d_q, &mut d_s, t * EMBD)
            .expect("quant");
        let want_x = exec.to_host(&d_x).expect("x host");
        let want_xn = exec.to_host(&d_xn).expect("xn host");
        let want_q = exec.to_host_i8(&d_q).expect("q host");
        let want_s = exec
            .to_host_range_u8(&d_s, 0, t * EMBD / 16)
            .expect("s host");

        // fused
        let mut f_x = exec.to_device(&x).expect("fx");
        let mut f_xn = exec.alloc(t * EMBD).expect("fxn");
        let mut f_q = exec.alloc_i8(t * EMBD / 2).expect("fq");
        let mut f_s = exec.alloc_u8(t * EMBD / 16).expect("fs");
        exec.add_rmsnorm_quant_nvf4_batch(
            &mut f_x,
            Some(&d_proj),
            &d_w,
            &mut f_xn,
            &mut f_q,
            &mut f_s,
            EMBD,
            eps,
            t,
        )
        .expect("fused");
        let got_x = exec.to_host(&f_x).expect("fx host");
        let got_xn = exec.to_host(&f_xn).expect("fxn host");
        let got_q = exec.to_host_i8(&f_q).expect("fq host");
        let got_s = exec
            .to_host_range_u8(&f_s, 0, t * EMBD / 16)
            .expect("fs host");

        // x: the residual the next layer reads. bit-exact or the tower drifts.
        assert_eq!(got_x, want_x, "t={t}: residual x differs");
        // xn: the f32 normed row the router still consumes.
        assert_eq!(got_xn, want_xn, "t={t}: normed row differs");
        // q/scale: the nvf4 planes the MoE up kernel consumes.
        assert_eq!(got_q, want_q, "t={t}: nvf4 packed plane differs");
        assert_eq!(got_s, want_s, "t={t}: nvf4 scale plane differs");
        println!("add+rmsnorm+quant_nvf4 t={t}: all 4 planes bit-exact vs the 3-kernel chain");
    }
}

/// f16 SSM-state class (scan rung, ABI 443-445). Three claims, and
/// the first is the sharp one because it is exact rather than tolerant.
///
///  1. The SEQ walk's interior arithmetic is bit-identical to the f32 twin.
///     The walk holds state in registers for the whole span and touches the
///     arena only at the two ends, so if the seed is exactly representable in
///     f16 the initial load is lossless and `y` must match F32 BIT-FOR-BIT.
///     Any drift here means the port changed the math, not the storage.
///  2. The DECODE step rounds once per token, so it is compared on a band,
///     not exactly - and the band is checked against the class floor, since
///     f16 carries 10 mantissa bits.
///  3. A snap row flat-copied back over the live state restores it exactly,
///     which is what a partial spec accept relies on.
#[test]
fn mamba2_f16_state_matches_f32_class() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2() || !exec.has_mamba2_f16_state() {
        common::missing("pack lacks the f16 SSM-state class (slots 443-445)");
        return;
    }
    use half::f16 as h16;
    let (nh, hd, ds, ng) = (H, HD, S, G);
    let state_elems = nh * hd * ds;
    let conv_dim = CONV_DIM;

    // Seed state with values that are exact in f16 (k/256, |k|<=512): then
    // f32 and f16 arenas hold the identical numbers and claim 1 is testable.
    let seed: Vec<f32> = (0..state_elems)
        .map(|i| ((i % 401) as f32 - 200.0) / 256.0)
        .collect();
    for (i, &v) in seed.iter().enumerate().take(64) {
        assert_eq!(h16::from_f32(v).to_f32(), v, "seed[{i}] not exact in f16");
    }
    let xbc = det(8 * conv_dim, 4242);
    let dtr = det(8 * nh, 4243);
    let a: Vec<f32> = (0..nh).map(|i| -0.5 - (i % 7) as f32 * 0.1).collect();
    let dd = det(nh, 4245);
    let bias: Vec<f32> = det(nh, 4246).iter().map(|v| v * 0.1).collect();

    let d_xbc = exec.to_device(&xbc).expect("xbc");
    let d_dt = exec.to_device(&dtr).expect("dt");
    let d_a = exec.to_device(&a).expect("a");
    let d_d = exec.to_device(&dd).expect("d");
    let d_b = exec.to_device(&bias).expect("bias");

    // ---- claim 1: seq walk, bit-exact -------------------------------------
    let n_tok = 5usize;
    let mut s32 = exec.to_device(&seed).expect("s32");
    let mut y32 = exec.alloc(n_tok * nh * hd).expect("y32");
    exec.mamba2_scan_seq_at(
        &mut s32, 0, &d_xbc, 0, &d_dt, 0, nh, &d_a, &d_d, &d_b, &mut y32, 0, n_tok, nh, hd, ds, ng,
    )
    .expect("seq f32");
    let h32 = exec.to_host(&y32).expect("y32 host");

    let seed16: Vec<h16> = seed.iter().map(|&v| h16::from_f32(v)).collect();
    let mut s16 = exec.alloc_f16(state_elems).expect("s16");
    {
        let mut v = s16.try_slice_mut(0..state_elems).expect("s16 view");
        exec.stream.memcpy_htod(&seed16, &mut v).expect("htod s16");
    }
    let mut y16 = exec.alloc(n_tok * nh * hd).expect("y16");
    exec.mamba2_scan_seq_at_f16(
        &mut s16, 0, &d_xbc, 0, &d_dt, 0, nh, &d_a, &d_d, &d_b, &mut y16, 0, n_tok, nh, hd, ds, ng,
    )
    .expect("seq f16");
    let h16y = exec.to_host(&y16).expect("y16 host");
    assert_eq!(
        h32, h16y,
        "seq walk: f16-state y must be BIT-IDENTICAL to f32 on an f16-exact \
         seed - the walk is register-resident, so only the arena hand-off may round"
    );

    // ---- claim 2: decode step, class band ---------------------------------
    let slots: Vec<u32> = vec![0];
    let d_slots = exec.to_device_u32(&slots).expect("slots");
    let mut t32 = exec.to_device(&seed).expect("t32");
    let mut ty32 = exec.alloc(nh * hd).expect("ty32");
    exec.mamba2_scan_step_batch(
        &mut t32, &d_xbc, &d_dt, 0, nh, &d_slots, &d_a, &d_d, &d_b, &mut ty32, 1, nh, hd, ds, ng,
    )
    .expect("step f32");
    let a32 = exec.to_host(&ty32).expect("ty32 host");

    let mut t16 = exec.alloc_f16(state_elems).expect("t16");
    {
        let mut v = t16.try_slice_mut(0..state_elems).expect("t16 view");
        exec.stream.memcpy_htod(&seed16, &mut v).expect("htod t16");
    }
    let mut ty16 = exec.alloc(nh * hd).expect("ty16");
    exec.mamba2_scan_step_batch_f16(
        &mut t16, &d_xbc, &d_dt, 0, nh, &d_slots, &d_a, &d_d, &d_b, &mut ty16, 1, nh, hd, ds, ng,
    )
    .expect("step f16");
    let b16 = exec.to_host(&ty16).expect("ty16 host");

    let rms_rel = |x: &[f32], y: &[f32]| {
        let (mut se, mut sr) = (0f64, 0f64);
        for (a, b) in x.iter().zip(y.iter()) {
            se += ((*a - *b) as f64).powi(2);
            sr += (*a as f64).powi(2);
        }
        (se / x.len() as f64).sqrt() / (sr / x.len() as f64).sqrt().max(1e-30)
    };
    // Step 1 must be exactly zero and that is not a strong result: with an
    // f16-exact seed the load is lossless, `s` is computed in f32 identically
    // by both arms, and `y` reads the PRE-round `s`. Nothing has been read
    // back through f16 yet. Asserting a band here would pass on any port.
    assert_eq!(
        rms_rel(&a32, &b16),
        0.0,
        "step 1 must be exact - an f16-exact seed rounds only on STORE, and y \
         reads the pre-round state; nonzero here means the port rounds early"
    );

    // The real test is the CHAIN: from step 2 the f16 arm reads back what it
    // rounded, which is the compounding this class has to be judged on. Run
    // both arms forward over the same inputs and watch the divergence grow.
    let mut prev = 0.0f64;
    for step in 2..=8usize {
        let off = (step - 1) % 8;
        exec.mamba2_scan_step_batch(
            &mut t32,
            &d_xbc,
            &d_dt,
            off * nh,
            nh,
            &d_slots,
            &d_a,
            &d_d,
            &d_b,
            &mut ty32,
            1,
            nh,
            hd,
            ds,
            ng,
        )
        .expect("step f32");
        exec.mamba2_scan_step_batch_f16(
            &mut t16,
            &d_xbc,
            &d_dt,
            off * nh,
            nh,
            &d_slots,
            &d_a,
            &d_d,
            &d_b,
            &mut ty16,
            1,
            nh,
            hd,
            ds,
            ng,
        )
        .expect("step f16");
        let c32 = exec.to_host(&ty32).expect("c32");
        let c16 = exec.to_host(&ty16).expect("c16");
        let rel = rms_rel(&c32, &c16);
        println!("  step {step}: rms-rel {rel:.3e}");
        // 2^-11 is the f16 mantissa floor; a correct chain sits near it and
        // grows slowly under the decay. Orders above it means a mispaired
        // half2 lane or a stale read, not a precision class.
        assert!(
            rel < 1e-2,
            "decode chain step {step}: rms-rel {rel:.3e} - that is not the f16 \
             class floor (2^-11 ~ 4.9e-4), it is a defect"
        );
        assert!(
            step == 2 || rel >= prev * 0.05,
            "step {step} error collapsed - arms desynced?"
        );
        prev = rel;
    }
    println!("f16 SSM state: seq BIT-EXACT, decode step 1 exact, chain within class");
}

// ---- the decode-band MoE route ----------------------------
// Every kernel RUNG needs its own parity case - a route that only the serving
// tick exercises is a route nothing gates. These two cover the pair the r>1
// Q8 arm now takes, against the sorted tile it replaced.

/// The engine's live-block bound, mirrored so the gate launches what the tick
/// launches (gpu_model/nemotron/batch.rs::moe_live_blocks).
fn live_blocks(rows: usize, picks: usize, experts: usize) -> usize {
    experts.min(rows * picks) * rows.div_ceil(32)
}

#[test]
fn quantize_q8_relu2_is_bitexact_vs_relu2_then_quantize() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_quantize_q8_relu2() {
        common::missing("pack has no quantize_q8_relu2");
        return;
    }
    // spans both signs and a whole zero block, so the relu clamp and the
    // scale==0 guard are both exercised
    let n = 32 * 512;
    let mut x = det(n, 77);
    for v in x.iter_mut().take(32) {
        *v = -1.0;
    }
    let d_x = exec.to_device(&x).expect("x");
    let mut q = exec.alloc_i8(n).expect("q");
    let mut s = exec.alloc(n / 32).expect("s");
    exec.quantize_q8_relu2(&d_x, &mut q, &mut s, n)
        .expect("fused");
    let (qf, sf) = (
        exec.to_host_i8(&q).expect("qf"),
        exec.to_host(&s).expect("sf"),
    );

    let relu2: Vec<f32> = x
        .iter()
        .map(|v| {
            let t = v.max(0.0);
            t * t
        })
        .collect();
    let d_r = exec.to_device(&relu2).expect("r");
    exec.quantize_q8(&d_r, &mut q, &mut s, n).expect("plain");
    let (qp, sp) = (
        exec.to_host_i8(&q).expect("qp"),
        exec.to_host(&s).expect("sp"),
    );

    let qd = qf.iter().zip(&qp).filter(|(a, b)| a != b).count();
    let sd = sf.iter().zip(&sp).filter(|(a, b)| a != b).count();
    assert_eq!(
        (qd, sd),
        (0, 0),
        "fused relu2 quantize must be BIT-identical: {qd} int8, {sd} scales"
    );
    println!("quantize_q8_relu2: bit-identical over {n} elements");
}

#[test]
fn q8_moe_decode_band_matches_the_sorted_route() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_q8_0_moe_relu2_dec2() || !exec.has_quantize_q8_relu2() {
        common::missing("pack has no relu2 decode-band pair");
        return;
    }
    let Some(path) = common::model(NEMO_Q8_ENV, common::NEMOTRON_30B_Q8) else {
        common::missing("nemotron Q8_0 GGUF");
        return;
    };
    if !common::heavy() {
        return;
    }
    let map = match paddock_models::mapped::MappedGguf::open(&path) {
        Ok(m) => m,
        Err(e) => panic!("open {}: {e}", path.display()),
    };
    let up = exec
        .repack_q8(&map, "blk.1.ffn_up_exps.weight")
        .expect("up");
    let down = exec
        .repack_q8(&map, "blk.1.ffn_down_exps.weight")
        .expect("down");
    let sh_up = exec
        .repack_q8(&map, "blk.1.ffn_up_shexp.weight")
        .expect("sh_up");
    let sh_down = exec
        .repack_q8(&map, "blk.1.ffn_down_shexp.weight")
        .expect("sh_down");
    let (embd, moe_ff, n_expert) = (up.dims[0], up.dims[1], up.dims[2]);
    let shared_ff = sh_up.dims[1];
    let (n_active, r) = (6usize, 4usize);

    // distinct picks per row, spread over every expert - the real routing
    let mut idx_h = vec![0u32; r * n_active];
    for b in 0..r {
        let mut seen: Vec<u32> = Vec::new();
        let mut h = (b as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(1);
        while seen.len() < n_active {
            h = h
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = ((h >> 33) % n_expert as u64) as u32;
            if !seen.contains(&e) {
                seen.push(e);
            }
        }
        idx_h[b * n_active..(b + 1) * n_active].copy_from_slice(&seen);
    }
    let wt: Vec<f32> = (0..r * n_active)
        .map(|i| 0.1 + (i % 7) as f32 * 0.05)
        .collect();
    let d_idx = exec.to_device_u32(&idx_h).expect("idx");
    let d_w = exec.to_device(&wt).expect("w");
    let d_shi = exec.to_device_u32(&vec![0u32; r]).expect("shi");
    let d_shw = exec.to_device(&vec![1.0f32; r]).expect("shw");

    let xf = exec.to_device(&det(r * embd, 21)).expect("x");
    let mut xq = exec.alloc_i8(r * embd).expect("xq");
    let mut xs = exec.alloc(r * embd / 32).expect("xs");
    exec.quantize_q8(&xf, &mut xq, &mut xs, r * embd)
        .expect("qx");

    let nbr = live_blocks(r, n_active, n_expert);
    let nbs = live_blocks(r, 1, 1);
    let mut srow = exec.alloc_u32(nbr.max(nbs) * 32).expect("srow");
    let mut sslot = exec.alloc_u32(nbr.max(nbs) * 32).expect("sslot");
    let mut bexp = exec.alloc_u32(nbr.max(nbs)).expect("bexp");
    let mut fu = exec.alloc(nbr * 32 * moe_ff.max(shared_ff)).expect("fu");
    let mut fq = exec.alloc_i8(nbr * 32 * moe_ff.max(shared_ff)).expect("fq");
    let mut fs = exec
        .alloc(nbr * 32 * moe_ff.max(shared_ff) / 32)
        .expect("fs");
    let mut part = exec.alloc(8 * 64 * shared_ff.max(embd)).expect("part");
    let mut want = exec.alloc(r * embd).expect("want");
    let mut proj = exec.alloc(r * embd).expect("proj");
    exec.zero_region(&mut want, 0, r * embd).expect("zero");

    // reference: the sorted tile for both halves - what the r>1 arm ran before
    exec.moe_align(
        &d_idx, &mut srow, &mut sslot, &mut bexp, r, n_active, n_expert, nbr,
    )
    .expect("align");
    exec.q8_0_moe_up_relu2_sorted(&up, &srow, &bexp, &xq, &xs, &mut fu, nbr)
        .expect("up sorted");
    exec.quantize_q8(&fu, &mut fq, &mut fs, nbr * 32 * moe_ff)
        .expect("q");
    let mut rpart = exec.alloc(r * n_active * embd).expect("rpart");
    exec.q8_0_moe_down_sorted(
        &down, &srow, &sslot, &bexp, &d_w, &fq, &fs, &mut rpart, n_active, nbr,
    )
    .expect("down sorted");
    exec.moe_slot_combine(&rpart, &mut want, embd, n_active, r)
        .expect("combine");
    exec.moe_align(&d_shi, &mut srow, &mut sslot, &mut bexp, r, 1, 1, nbs)
        .expect("align sh");
    exec.q8_0_moe_up_relu2_sorted(&sh_up, &srow, &bexp, &xq, &xs, &mut fu, nbs)
        .expect("sh up");
    exec.quantize_q8(&fu, &mut fq, &mut fs, nbs * 32 * shared_ff)
        .expect("q sh");
    exec.q8_0_moe_down_sorted(
        &sh_down, &srow, &sslot, &bexp, &d_shw, &fq, &fs, &mut proj, 1, nbs,
    )
    .expect("sh down");
    exec.moe_slot_combine(&proj, &mut want, embd, 1, r)
        .expect("combine sh");
    let want_h = exec.to_host(&want).expect("want");

    // route under test: dec2 routed pair + the dense ladder for the shared FFN
    let mut got = exec.alloc(r * embd).expect("got");
    let mut shp = exec.alloc(r * embd).expect("shp");
    exec.q8_0_moe_up_relu2_dec2(&up, &d_idx, &xq, &xs, &mut fu, n_active, r, 0)
        .expect("up dec2");
    exec.quantize_q8(&fu, &mut fq, &mut fs, r * n_active * moe_ff)
        .expect("q dec2");
    exec.q8_0_moe_dn_dec2(&down, &d_idx, &d_w, &fq, &fs, &mut got, n_active, r)
        .expect("dn dec2");
    exec.q8_0_gemm_mma_ks(&sh_up, &xq, &xs, &mut part, &mut fu, r)
        .expect("sh up ks");
    exec.quantize_q8_relu2(&fu, &mut fq, &mut fs, r * shared_ff)
        .expect("q relu2");
    exec.q8_0_gemm_mma_ks(&sh_down, &fq, &fs, &mut part, &mut shp, r)
        .expect("sh down ks");
    exec.add(&mut got, &shp, r * embd).expect("add");
    let got_h = exec.to_host(&got).expect("got");

    let (mut md, mut mv) = (0f32, 0f32);
    for (a, b) in want_h.iter().zip(&got_h) {
        md = md.max((a - b).abs());
        mv = mv.max(a.abs());
    }
    let rel = md / mv.max(1e-30);
    println!("decode-band MoE vs sorted: max|d| {md:.3e} / max|y| {mv:.3e} = {rel:.3e}");
    // both routes fold the same topk weights over the same experts and differ
    // only in reduction grouping, so this is an f32-epsilon check. A layout or
    // indexing error reads as O(1), not O(1e-7).
    assert!(
        rel < 1e-5,
        "decode-band MoE disagrees with the sorted route: rel {rel:.3e}"
    );
}

/// Rollback by replay (slots 746-747) against the snapshot walk (444) it
/// replaces in the verify round. The keep walk's y must be the snap walk's
/// y BIT FOR BIT and leave the state untouched (it stays the pre-round state
/// until the commit); a replay of n kept rows must land on snap row n - 1 bit
/// for bit, for every n - into another buffer, in place, and beside other
/// replays in the same launch; a rows = 0 descriptor leaves its target alone.
/// A few heads ride past the softplus cut so both dt branches replay.
#[test]
fn mamba2_keep_walk_and_replay_match_the_snap_walk_bitexact() {
    use cudarc::driver::{CudaSlice, DevicePtr};
    use half::f16 as h16;
    let Some(exec) = common::gpu() else { return };
    if !exec.has_mamba2_f16_state() || !exec.has_mamba2_rescan() {
        common::missing("pack lacks rollback by replay (slots 746-747)");
        return;
    }
    let (nh, hd, ds, ng) = (H, HD, S, G);
    let se = nh * hd * ds;
    let n_tok = 7usize;
    let kr = paddock_engine::gpu::mamba2_keep_row(nh, hd, ds, ng);
    let seed: Vec<h16> = det(se, 7001)
        .iter()
        .map(|&v| h16::from_f32(v * 0.75))
        .collect();
    let xbc = det(n_tok * CONV_DIM, 7002);
    let dtr: Vec<f32> = det(n_tok * nh, 7003)
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 13 == 0 { 24.0 + v } else { v * 3.0 })
        .collect();
    let a: Vec<f32> = (0..nh).map(|i| -0.3 - (i % 9) as f32 * 0.17).collect();
    let dd = det(nh, 7005);
    let bias: Vec<f32> = det(nh, 7006).iter().map(|v| v * 0.5).collect();
    let d_xbc = exec.to_device(&xbc).expect("xbc");
    let d_dt = exec.to_device(&dtr).expect("dt");
    let d_a = exec.to_device(&a).expect("a");
    let d_d = exec.to_device(&dd).expect("d");
    let d_b = exec.to_device(&bias).expect("bias");
    let up16 = |v: &[h16], copies: usize| -> CudaSlice<h16> {
        let mut b = exec.alloc_f16(v.len() * copies).expect("f16 plane");
        for c in 0..copies {
            let mut w = b
                .try_slice_mut(c * v.len()..(c + 1) * v.len())
                .expect("f16 view");
            exec.stream.memcpy_htod(v, &mut w).expect("htod f16");
        }
        b
    };
    let bits = |b: &CudaSlice<h16>, off: usize, n: usize| -> Vec<u16> {
        let h = exec.to_host_f16_len(b, off + n).expect("dtoh f16");
        h[off..off + n].iter().map(|x| x.to_bits()).collect()
    };

    // the reference: the snapshot walk the verify ran until now
    let mut s_snap = up16(&seed, 1);
    let mut snap = exec.alloc_f16(n_tok * se).expect("snap");
    let mut y_snap = exec.alloc(n_tok * D_INNER).expect("y snap");
    exec.mamba2_scan_seq_snap_at_f16(
        &mut s_snap,
        0,
        &d_xbc,
        0,
        &d_dt,
        0,
        nh,
        &d_a,
        &d_d,
        &d_b,
        &mut y_snap,
        0,
        &mut snap,
        0,
        n_tok,
        nh,
        hd,
        ds,
        ng,
    )
    .expect("snap walk");

    // the keep walk on slot 1 of a two-slot arena, its rows kept at row 2
    let arena = up16(&seed, 2);
    let mut keep = exec.alloc((2 + n_tok) * kr).expect("keep");
    let mut y_keep = exec.alloc(n_tok * D_INNER).expect("y keep");
    exec.mamba2_scan_seq_keep_at_f16(
        &arena,
        se,
        &d_xbc,
        0,
        &d_dt,
        0,
        nh,
        &d_a,
        &d_d,
        &d_b,
        &mut y_keep,
        0,
        &mut keep,
        2 * kr,
        n_tok,
        nh,
        hd,
        ds,
        ng,
    )
    .expect("keep walk");
    let ys = exec.to_host(&y_snap).expect("y snap host");
    let yk = exec.to_host(&y_keep).expect("y keep host");
    assert!(
        ys.iter().zip(&yk).all(|(p, q)| p.to_bits() == q.to_bits()),
        "the keep walk's y must be the snap walk's bit for bit"
    );
    let seed_bits: Vec<u16> = seed.iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        bits(&arena, se, se),
        seed_bits,
        "the keep walk wrote the live state"
    );

    let ptr = |b: &CudaSlice<h16>, off: usize| -> u64 {
        let (p, _g) = b.device_ptr(&exec.stream);
        p + (off * 2) as u64
    };
    let (kp, _g1) = keep.device_ptr(&exec.stream);
    let kp = kp + (2 * kr * 4) as u64;
    let (ap, _g2) = d_a.device_ptr(&exec.stream);
    let (bp, _g3) = d_b.device_ptr(&exec.stream);
    let mut desc = exec.alloc_u64(6 * 4).expect("descs");
    let tmp = exec.alloc_f16(se).expect("tmp");
    for n in 1..=n_tok {
        let d = [ptr(&arena, se), ptr(&tmp, 0), kp, ap, bp, n as u64];
        exec.mamba2_rescan_upload(&mut desc, &d, nh, hd, ds, ng)
            .expect("replay");
        assert_eq!(
            bits(&tmp, 0, se),
            bits(&snap, (n - 1) * se, se),
            "replay of {n} rows differs from snap row {}",
            n - 1
        );
    }
    // in place on slot 1, slot 0 into tmp at 3 rows, and a rows = 0 no-op,
    // one launch
    let idle = up16(&vec![h16::from_f32(0.5); se], 1);
    let d = [
        ptr(&arena, se),
        ptr(&arena, se),
        kp,
        ap,
        bp,
        n_tok as u64,
        ptr(&arena, 0),
        ptr(&tmp, 0),
        kp,
        ap,
        bp,
        3,
        ptr(&arena, 0),
        ptr(&idle, 0),
        kp,
        ap,
        bp,
        0,
    ];
    exec.mamba2_rescan_upload(&mut desc, &d, nh, hd, ds, ng)
        .expect("replays");
    assert_eq!(
        bits(&arena, se, se),
        bits(&s_snap, 0, se),
        "in-place replay of the whole round differs from the snap walk's final state"
    );
    assert_eq!(
        bits(&tmp, 0, se),
        bits(&snap, 2 * se, se),
        "the batched 3-row replay differs"
    );
    assert_eq!(bits(&arena, 0, se), seed_bits, "a replay source changed");
    assert!(
        bits(&idle, 0, se)
            .iter()
            .all(|&b| b == h16::from_f32(0.5).to_bits()),
        "a rows = 0 replay wrote its target"
    );
    println!(
        "rollback by replay: keep-walk y == snap y, replay n rows == snap row n-1 for n = 1..{n_tok}, in place and batched"
    );
}
