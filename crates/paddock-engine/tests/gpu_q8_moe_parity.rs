//! CPU-reference parity for the Q8_0 routed-expert MoE kernels
//! (pd_q8_0_moe_gate_up_dp4a / pd_q8_0_moe_down_dp4a / pd_shexp_gate_add -
//! the qwen3.6-A3B FFN). The GPU quantizes activations; the reference reads
//! those exact int8 blocks back, so the diff isolates the kernels' integer
//! dots + f32 scale products (tight tolerance, no quantization slack).

mod common;

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_engine::gpu::{GpuExecutor, RepackedQ8};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
}

/// CPU q8_0 quantize into the REPACKED layout (int8 stream + f16 scales),
/// same math as pd_quantize_q8 / pd_q8_0_repack: per-32 absmax / 127.
fn quant_rows(vals: &[f32]) -> (Vec<i8>, Vec<half::f16>) {
    let mut data = Vec::with_capacity(vals.len());
    let mut scales = Vec::with_capacity(vals.len() / 32);
    for block in vals.chunks(32) {
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scales.push(half::f16::from_f32(d));
        for &v in block {
            data.push((v * inv).round().clamp(-127.0, 127.0) as i8);
        }
    }
    (data, scales)
}

fn upload_repacked(
    exec: &GpuExecutor,
    vals: &[f32],
    dims: Vec<usize>,
) -> (RepackedQ8, Vec<i8>, Vec<half::f16>) {
    let (data, scales) = quant_rows(vals);
    let bytes: Vec<u8> = data.iter().map(|&b| b as u8).collect();
    let sbytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    let t = RepackedQ8 {
        data: exec.to_device_u8(&bytes).expect("data"),
        scale: exec.to_device_u8(&sbytes).expect("scale"),
        dims,
    };
    (t, data, scales)
}

/// dequantized dot of a repacked row against int8 activations with scales
fn ref_dot(w: &[i8], ws: &[half::f16], xq: &[i8], xs: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..w.len() / 32 {
        let mut s = 0i32;
        for i in 0..32 {
            s += (w[b * 32 + i] as i32) * (xq[b * 32 + i] as i32);
        }
        acc += ws[b].to_f32() * xs[b] * s as f32;
    }
    acc
}

#[test]
fn q8_moe_matches_cpu() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let (n_expert, n_active, in_dim, ff, batch) = (16usize, 4usize, 128usize, 64usize, 3usize);
    let mut seed = 0xdeadbeefcafef00du64;

    // expert weights (gate/up [in, ff, E] rows of in_dim; down [ff, in, E])
    let gate_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let up_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let down_f: Vec<f32> = (0..n_expert * in_dim * ff)
        .map(|_| lcg(&mut seed))
        .collect();
    let (gate, gate_q, gate_s) = upload_repacked(&exec, &gate_f, vec![in_dim, ff, n_expert]);
    let (up, up_q, up_s) = upload_repacked(&exec, &up_f, vec![in_dim, ff, n_expert]);
    let (down, down_q, down_s) = upload_repacked(&exec, &down_f, vec![ff, in_dim, n_expert]);

    // activations + routing
    let x: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
    let idx_host: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i * 5 + 3) % n_expert) as u32)
        .collect();
    let w_host: Vec<f32> = (0..batch * n_active)
        .map(|i| 0.1 + 0.05 * (i % 7) as f32)
        .collect();

    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx_host).expect("idx");
    let d_w = exec.to_device(&w_host).expect("w");
    let mut d_xq: CudaSlice<i8> = exec.alloc_i8(batch * in_dim).expect("xq");
    let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
        .expect("quant");
    let mut d_fused = exec.alloc(batch * n_active * ff).expect("fused");
    exec.q8_0_moe_gate_up(
        &gate,
        &up,
        &d_idx,
        &d_xq,
        &d_xs,
        &mut d_fused,
        n_active,
        batch,
    )
    .expect("gate_up");

    // read the GPU's exact int8 activations back for the reference
    let xq_host: Vec<i8> = exec.to_host_i8(&d_xq).expect("xq back");
    let xs_host = exec.to_host(&d_xs).expect("xs back");
    let fused_gpu = exec.to_host(&d_fused).expect("fused back");

    let mut max_diff = 0.0f32;
    let mut fused_ref = vec![0.0f32; batch * n_active * ff];
    for b in 0..batch {
        let xq_b = &xq_host[b * in_dim..][..in_dim];
        let xs_b = &xs_host[b * in_dim / 32..][..in_dim / 32];
        for s in 0..n_active {
            let e = idx_host[b * n_active + s] as usize;
            for o in 0..ff {
                let row = (e * ff + o) * in_dim;
                let g = ref_dot(
                    &gate_q[row..][..in_dim],
                    &gate_s[row / 32..][..in_dim / 32],
                    xq_b,
                    xs_b,
                );
                let u = ref_dot(
                    &up_q[row..][..in_dim],
                    &up_s[row / 32..][..in_dim / 32],
                    xq_b,
                    xs_b,
                );
                let r = (g / (1.0 + (-g).exp())) * u;
                fused_ref[(b * n_active + s) * ff + o] = r;
                max_diff = max_diff.max((r - fused_gpu[(b * n_active + s) * ff + o]).abs());
            }
        }
    }
    eprintln!("q8 moe gate_up parity: max_abs_diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "gate_up diverges: {max_diff}");

    // down + weighted combine
    let mut d_fq: CudaSlice<i8> = exec.alloc_i8(batch * n_active * ff).expect("fq");
    let mut d_fs = exec.alloc(batch * n_active * ff / 32).expect("fs");
    exec.quantize_q8(&d_fused, &mut d_fq, &mut d_fs, batch * n_active * ff)
        .expect("quant f");
    let mut d_out = exec.alloc(batch * in_dim).expect("out");
    exec.q8_0_moe_down(
        &down, &d_idx, &d_w, &d_fq, &d_fs, &mut d_out, n_active, batch,
    )
    .expect("down");
    let fq_host: Vec<i8> = exec.to_host_i8(&d_fq).expect("fq back");
    let fs_host = exec.to_host(&d_fs).expect("fs back");
    let out_gpu = exec.to_host(&d_out).expect("out back");

    let mut max_diff = 0.0f32;
    let mut out_ref = vec![0.0f32; batch * in_dim];
    for b in 0..batch {
        for o in 0..in_dim {
            let mut acc = 0.0f32;
            for s in 0..n_active {
                let e = idx_host[b * n_active + s] as usize;
                let row = (e * in_dim + o) * ff;
                let srow = (b * n_active + s) * ff;
                acc += w_host[b * n_active + s]
                    * ref_dot(
                        &down_q[row..][..ff],
                        &down_s[row / 32..][..ff / 32],
                        &fq_host[srow..][..ff],
                        &fs_host[srow / 32..][..ff / 32],
                    );
            }
            out_ref[b * in_dim + o] = acc;
            max_diff = max_diff.max((acc - out_gpu[b * in_dim + o]).abs());
        }
    }
    eprintln!("q8 moe down parity: max_abs_diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "down diverges: {max_diff}");

    // shared-expert sigmoid gate fold: dst += sigmoid(x.w) * src
    let wg: Vec<f32> = (0..in_dim).map(|_| lcg(&mut seed)).collect();
    let src: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
    let d_wg = exec.to_device(&wg).expect("wg");
    let d_src = exec.to_device(&src).expect("src");
    exec.shexp_gate_add(&mut d_out, &d_src, &d_x, &d_wg, in_dim, in_dim, batch)
        .expect("shexp");
    let folded = exec.to_host(&d_out).expect("folded");
    let mut max_diff = 0.0f32;
    for b in 0..batch {
        let dot: f32 = x[b * in_dim..][..in_dim]
            .iter()
            .zip(&wg)
            .map(|(a, w)| a * w)
            .sum();
        let g = 1.0 / (1.0 + (-dot).exp());
        for o in 0..in_dim {
            let r = out_ref[b * in_dim + o] + g * src[b * in_dim + o];
            max_diff = max_diff.max((r - folded[b * in_dim + o]).abs());
        }
    }
    eprintln!("shexp gate fold parity: max_abs_diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "shexp fold diverges: {max_diff}");
}

/// Sorted-class parity: same CPU reference, routed through moe_align +
/// pd_q8_0_moe_gate_up_sorted + pd_q8_0_moe_down_sorted + moe_slot_combine.
/// in_dim/ff meet the kernels' %256 staging requirement; expert count is
/// small so per-expert tails exercise PAD rows heavily.
#[test]
fn q8_moe_sorted_matches_cpu() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };

    let (n_expert, n_active, in_dim, ff, batch) = (8usize, 4usize, 256usize, 256usize, 37usize);
    let mut seed = 0x5eed5eed5eed5eedu64;
    let gate_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let up_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let down_f: Vec<f32> = (0..n_expert * in_dim * ff)
        .map(|_| lcg(&mut seed))
        .collect();
    let (gate, gate_q, gate_s) = upload_repacked(&exec, &gate_f, vec![in_dim, ff, n_expert]);
    let (up, up_q, up_s) = upload_repacked(&exec, &up_f, vec![in_dim, ff, n_expert]);
    let (down, down_q, down_s) = upload_repacked(&exec, &down_f, vec![ff, in_dim, n_expert]);

    let x: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
    let idx_host: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i * 7 + 2) % n_expert) as u32)
        .collect();
    let w_host: Vec<f32> = (0..batch * n_active)
        .map(|i| 0.05 + 0.03 * (i % 11) as f32)
        .collect();

    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx_host).expect("idx");
    let d_w = exec.to_device(&w_host).expect("w");
    let mut d_xq: CudaSlice<i8> = exec.alloc_i8(batch * in_dim).expect("xq");
    let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
        .expect("quant");

    let max_blocks = (batch * n_active + n_expert * 31).div_ceil(32);
    let mut srow = exec.alloc_u32(max_blocks * 32).expect("srow");
    let mut sslot = exec.alloc_u32(max_blocks * 32).expect("sslot");
    let mut bexp = exec.alloc_u32(max_blocks).expect("bexp");
    exec.moe_align(
        &d_idx, &mut srow, &mut sslot, &mut bexp, batch, n_active, n_expert, max_blocks,
    )
    .expect("align");
    let mut d_fused = exec.alloc(max_blocks * 32 * ff).expect("fused");
    exec.q8_0_moe_gate_up_sorted(
        &gate,
        &up,
        &srow,
        &bexp,
        &d_xq,
        &d_xs,
        &mut d_fused,
        max_blocks,
    )
    .expect("gate_up sorted");
    let mut d_fq: CudaSlice<i8> = exec.alloc_i8(max_blocks * 32 * ff).expect("fq");
    let mut d_fs = exec.alloc(max_blocks * 32 * ff / 32).expect("fs");
    exec.quantize_q8(&d_fused, &mut d_fq, &mut d_fs, max_blocks * 32 * ff)
        .expect("quant f");
    let mut d_part = exec.alloc(batch * n_active * in_dim).expect("part");
    exec.q8_0_moe_down_sorted(
        &down,
        &srow,
        &sslot,
        &bexp,
        &d_w,
        &d_fq,
        &d_fs,
        &mut d_part,
        n_active,
        max_blocks,
    )
    .expect("down sorted");
    let mut d_out = exec.alloc(batch * in_dim).expect("out");
    exec.stream.memset_zeros(&mut d_out).expect("zero");
    exec.moe_slot_combine(&d_part, &mut d_out, in_dim, n_active, batch)
        .expect("combine");
    let out_gpu = exec.to_host(&d_out).expect("out back");

    // CPU reference over the GPU's exact int8 stages (activations + the
    // sorted fused rows), mapped through the sorted layout read back
    let xq_host: Vec<i8> = exec.to_host_i8(&d_xq).expect("xq back");
    let xs_host = exec.to_host(&d_xs).expect("xs back");
    let srow_host = exec.to_host_u32(&srow).expect("srow back");
    let sslot_host = exec.to_host_u32(&sslot).expect("sslot back");
    let bexp_host = exec.to_host_u32(&bexp).expect("bexp back");
    let fq_host: Vec<i8> = exec.to_host_i8(&d_fq).expect("fq back");
    let fs_host = exec.to_host(&d_fs).expect("fs back");

    const PAD: u32 = 0xFFFF_FFFF;
    let mut out_ref = vec![0.0f32; batch * in_dim];
    let mut covered = vec![false; batch * n_active];
    for blk in 0..max_blocks {
        let e = bexp_host[blk];
        if e == PAD {
            continue;
        }
        for r in 0..32 {
            let srw = srow_host[blk * 32 + r];
            if srw == PAD {
                continue;
            }
            let slot = sslot_host[blk * 32 + r] as usize;
            let pair = srw as usize * n_active + slot;
            assert_eq!(
                idx_host[pair], e,
                "sorted layout routed to the wrong expert"
            );
            covered[pair] = true;
            let frow = blk * 32 + r;
            for o in 0..in_dim {
                let wrow = (e as usize * in_dim + o) * ff;
                out_ref[srw as usize * in_dim + o] += w_host[pair]
                    * ref_dot(
                        &down_q[wrow..][..ff],
                        &down_s[wrow / 32..][..ff / 32],
                        &fq_host[frow * ff..][..ff],
                        &fs_host[frow * ff / 32..][..ff / 32],
                    );
            }
        }
    }
    assert!(
        covered.iter().all(|&c| c),
        "sorted layout dropped a (token, slot) pair"
    );
    let mut max_diff = 0.0f32;
    for i in 0..batch * in_dim {
        max_diff = max_diff.max((out_ref[i] - out_gpu[i]).abs());
    }
    eprintln!("q8 sorted moe end-to-end parity: max_abs_diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "sorted moe diverges: {max_diff}");

    // and the fused rows themselves vs the CPU gate_up reference
    let fused_gpu = exec.to_host(&d_fused).expect("fused back");
    let mut max_diff = 0.0f32;
    for blk in 0..max_blocks {
        let e = bexp_host[blk];
        for r in 0..32 {
            let srw = srow_host[blk * 32 + r];
            if e == PAD || srw == PAD {
                continue;
            }
            let xq_b = &xq_host[srw as usize * in_dim..][..in_dim];
            let xs_b = &xs_host[srw as usize * in_dim / 32..][..in_dim / 32];
            for o in 0..ff {
                let wrow = (e as usize * ff + o) * in_dim;
                let g = ref_dot(
                    &gate_q[wrow..][..in_dim],
                    &gate_s[wrow / 32..][..in_dim / 32],
                    xq_b,
                    xs_b,
                );
                let u = ref_dot(
                    &up_q[wrow..][..in_dim],
                    &up_s[wrow / 32..][..in_dim / 32],
                    xq_b,
                    xs_b,
                );
                let r_ref = (g / (1.0 + (-g).exp())) * u;
                max_diff = max_diff.max((r_ref - fused_gpu[(blk * 32 + r) * ff + o]).abs());
            }
        }
    }
    eprintln!("q8 sorted gate_up parity: max_abs_diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "sorted gate_up diverges: {max_diff}");
}

/// int8-MMA sorted pair parity: the mma gate_up quantizes in registers, so
/// (a) its dequantized output must sit within half a quant step of the CPU
/// f32 reference, and (b) the down+combine result over the mma's own int8
/// stages must match the CPU reference tightly (isolates the down kernel).
#[test]
fn q8_moe_mma_matches_cpu() {
    let Some(exec) = common::gpu() else {
        return;
    };
    if exec.compute_capability().0 < 8 {
        eprintln!("no int8 mma on this device - skipping");
        return;
    }
    let exec = Arc::new(exec);

    // Three shapes, because the K-walk guards only fire on some of them:
    //   (256, 256) - the clean case, K a whole number of 256-wide chunks
    //   (256, 704) - gemma-4-26B-A4B's real expert width. The down half walks
    //                K=704 as 3 chunks of 8 blocks = 24 > 22, so the last
    //                chunk's tail blocks are masked out mid-scale-word.
    //   (512, 160) - down's scale rows are then 5 blocks, an ODD length, the
    //                one case where a paired 32-bit scale load would be
    //                misaligned and has to fall back to two 16-bit reads.
    //                ff=160 also leaves gate_up's last 64-row strip ragged.
    for (in_dim, ff) in [(256usize, 256usize), (256, 704), (512, 160)] {
        mma_case(&exec, 8, 4, in_dim, ff, 37);
    }
}

fn mma_case(
    exec: &Arc<GpuExecutor>,
    n_expert: usize,
    n_active: usize,
    in_dim: usize,
    ff: usize,
    batch: usize,
) {
    let exec = exec.clone();
    let mut seed = 0x1234abcd5678eff0u64;
    let gate_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let up_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let down_f: Vec<f32> = (0..n_expert * in_dim * ff)
        .map(|_| lcg(&mut seed))
        .collect();
    let (gate, gate_q, gate_s) = upload_repacked(&exec, &gate_f, vec![in_dim, ff, n_expert]);
    let (up, up_q, up_s) = upload_repacked(&exec, &up_f, vec![in_dim, ff, n_expert]);
    let (down, down_q, down_s) = upload_repacked(&exec, &down_f, vec![ff, in_dim, n_expert]);

    let x: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
    let idx_host: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i * 3 + 1) % n_expert) as u32)
        .collect();
    let w_host: Vec<f32> = (0..batch * n_active)
        .map(|i| 0.05 + 0.04 * (i % 9) as f32)
        .collect();
    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx_host).expect("idx");
    let d_w = exec.to_device(&w_host).expect("w");
    let mut d_xq: CudaSlice<i8> = exec.alloc_i8(batch * in_dim).expect("xq");
    let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
        .expect("quant");

    // Validate both block tiles: 32 (serving/decode, double-buffered) and 64
    // (wider prefill, single-buffered). BM only regroups tokens into blocks -
    // each output row is an independent K-sum and PAD rows are zeros - so both
    // must match the same CPU reference. bm=64 exercises the token-half loop.
    const PAD: u32 = 0xFFFF_FFFF;
    for bm in [32usize, 64usize] {
        let max_blocks = (batch * n_active + n_expert * (bm - 1)).div_ceil(bm);
        let mut srow = exec.alloc_u32(max_blocks * bm).expect("srow");
        let mut sslot = exec.alloc_u32(max_blocks * bm).expect("sslot");
        let mut bexp = exec.alloc_u32(max_blocks).expect("bexp");
        if bm == 64 {
            exec.moe_align_bm(
                &d_idx, &mut srow, &mut sslot, &mut bexp, batch, n_active, n_expert, bm, max_blocks,
            )
            .expect("align_bm");
        } else {
            exec.moe_align(
                &d_idx, &mut srow, &mut sslot, &mut bexp, batch, n_active, n_expert, max_blocks,
            )
            .expect("align");
        }
        let mut d_fq: CudaSlice<i8> = exec.alloc_i8(max_blocks * bm * ff).expect("fq");
        let mut d_fs = exec.alloc(max_blocks * bm * ff / 32).expect("fs");
        // PAD blocks return before writing anything, so without this the
        // staging buffers keep whatever was in the pool. Harmless for the
        // result (every consumer re-checks PAD) but it makes fq/fs
        // run-dependent, which would make the bit-exactness dump below
        // compare garbage. Zero first so the comparison is over defined bytes.
        exec.stream.memset_zeros(&mut d_fq).expect("zero fq");
        exec.stream.memset_zeros(&mut d_fs).expect("zero fs");
        exec.q8_0_moe_gate_up_mma(
            &gate, &up, &srow, &bexp, &d_xq, &d_xs, &mut d_fq, &mut d_fs, max_blocks, bm,
        )
        .expect("gate_up mma");
        // Read the gate_up stage back here, at the point of definition, rather
        // than after the down/part/out allocations -- reading it late made it
        // vary run to run on shapes it has no business varying on.
        let fq_host: Vec<i8> = exec.to_host_i8(&d_fq).expect("fq back");
        let fs_host = exec.to_host(&d_fs).expect("fs back");
        let mut d_part = exec.alloc(batch * n_active * in_dim).expect("part");
        exec.q8_0_moe_down_mma(
            &down,
            &srow,
            &sslot,
            &bexp,
            &d_w,
            &d_fq,
            &d_fs,
            &mut d_part,
            n_active,
            max_blocks,
            bm,
        )
        .expect("down mma");
        let mut d_out = exec.alloc(batch * in_dim).expect("out");
        exec.stream.memset_zeros(&mut d_out).expect("zero");
        exec.moe_slot_combine(&d_part, &mut d_out, in_dim, n_active, batch)
            .expect("combine");

        let xq_host: Vec<i8> = exec.to_host_i8(&d_xq).expect("xq back");
        let xs_host = exec.to_host(&d_xs).expect("xs back");
        let srow_host = exec.to_host_u32(&srow).expect("srow back");
        let sslot_host = exec.to_host_u32(&sslot).expect("sslot back");
        let bexp_host = exec.to_host_u32(&bexp).expect("bexp back");
        let out_gpu = exec.to_host(&d_out).expect("out back");

        // Two-run bit-exactness harness for MoE kernel variants. Any rung that
        // only re-partitions work (different tile, different block->output
        // map) leaves each output's K walk -- and so its accumulation order --
        // untouched, and must therefore agree BIT for bit with the current
        // kernel, not merely within the quant tolerances asserted below. That
        // is a far sharper gate than the tolerances, and it is how the RW=32
        // strip variant was cleared before being benchmarked (R2).
        // Kernel-selecting env vars are read once per process (function-local
        // statics in the launchers), so this is two runs and a cmp rather than
        // one in-process A/B:
        //   PADDOCK_MOE_DUMP_DIR=/tmp/a cargo test ... q8_moe_mma_matches_cpu
        //   <VARIANT_ENV>=1 PADDOCK_MOE_DUMP_DIR=/tmp/b cargo test ... (same)
        //   cmp -s /tmp/a/out_<shape>.bin /tmp/b/out_<shape>.bin
        //
        // COMPARE out_*, not fq/fs/srow. pd_moe_align_kernel scatters a token
        // into its expert's block with atomicAdd(&fill[e]), so which slot a
        // token lands in inside that block is race order, not a function of
        // the input: two identical runs produce the same multiset of token ids
        // per block in a different permutation (measured: ~a quarter of the
        // entries reorder, always same set, always inside live blocks;
        // compute-sanitizer initcheck is clean, so this is order, not
        // uninitialised memory). fq/fs are laid out by SLOT, so they inherit
        // that permutation and are not reproducible byte-wise. The combined
        // output is keyed by TOKEN (part[(token*n_active + slot)*embd + r])
        // and folded in fixed slot order, so it is exact and is the only
        // sound substrate for this gate. srow/bexp are dumped for diagnosis.
        if let Ok(dir) = std::env::var("PADDOCK_MOE_DUMP_DIR") {
            let tag = format!("{in_dim}x{ff}_bm{bm}");
            std::fs::create_dir_all(&dir).expect("dump dir");
            let raw = |v: &[u8], name: &str| {
                std::fs::write(std::path::Path::new(&dir).join(name), v).expect("dump");
            };
            raw(
                &srow_host
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
                &format!("srow_{tag}.bin"),
            );
            raw(
                &bexp_host
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
                &format!("bexp_{tag}.bin"),
            );
            raw(
                &fq_host.iter().map(|&b| b as u8).collect::<Vec<u8>>(),
                &format!("fq_{tag}.bin"),
            );
            raw(
                &fs_host
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>(),
                &format!("fs_{tag}.bin"),
            );
            raw(
                &out_gpu
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect::<Vec<u8>>(),
                &format!("out_{tag}.bin"),
            );
        }

        // (a) gate_up: dequantized in-register-quantized output within half a
        // quant step of the CPU f32 fused reference (mma f32-regroup drift is
        // orders below the quant step at these magnitudes)
        let mut worst_ratio = 0.0f32;
        for blk in 0..max_blocks {
            let e = bexp_host[blk];
            for r in 0..bm {
                let srw = srow_host[blk * bm + r];
                if e == PAD || srw == PAD {
                    continue;
                }
                let xq_b = &xq_host[srw as usize * in_dim..][..in_dim];
                let xs_b = &xs_host[srw as usize * in_dim / 32..][..in_dim / 32];
                let frow = blk * bm + r;
                for o in 0..ff {
                    let wrow = (e as usize * ff + o) * in_dim;
                    let g = ref_dot(
                        &gate_q[wrow..][..in_dim],
                        &gate_s[wrow / 32..][..in_dim / 32],
                        xq_b,
                        xs_b,
                    );
                    let u = ref_dot(
                        &up_q[wrow..][..in_dim],
                        &up_s[wrow / 32..][..in_dim / 32],
                        xq_b,
                        xs_b,
                    );
                    let r_ref = (g / (1.0 + (-g).exp())) * u;
                    let scale = fs_host[frow * ff / 32 + o / 32];
                    let deq = fq_host[frow * ff + o] as f32 * scale;
                    let tol = 0.5 * scale.abs() + 1e-3 * r_ref.abs() + 1e-4;
                    let diff = (deq - r_ref).abs();
                    worst_ratio = worst_ratio.max(diff / tol.max(1e-9));
                    assert!(
                        diff <= tol,
                        "bm={bm} mma gate_up out of quant tolerance: {deq} vs {r_ref} (tol {tol})"
                    );
                }
            }
        }
        eprintln!(
            "mma gate_up parity (in_dim={in_dim} ff={ff} bm={bm}): worst diff/tolerance ratio {worst_ratio:.3}"
        );

        // (b) down+combine over the mma's own int8 stages: tight
        let mut out_ref = vec![0.0f32; batch * in_dim];
        for blk in 0..max_blocks {
            let e = bexp_host[blk];
            if e == PAD {
                continue;
            }
            for r in 0..bm {
                let srw = srow_host[blk * bm + r];
                if srw == PAD {
                    continue;
                }
                let slot = sslot_host[blk * bm + r] as usize;
                let pair = srw as usize * n_active + slot;
                let frow = blk * bm + r;
                for o in 0..in_dim {
                    let wrow = (e as usize * in_dim + o) * ff;
                    out_ref[srw as usize * in_dim + o] += w_host[pair]
                        * ref_dot(
                            &down_q[wrow..][..ff],
                            &down_s[wrow / 32..][..ff / 32],
                            &fq_host[frow * ff..][..ff],
                            &fs_host[frow * ff / 32..][..ff / 32],
                        );
                }
            }
        }
        // RELATIVE, not absolute: the down output is a K=ff sum, so both its
        // magnitude and its f32 regrouping error grow with ff. The old 1e-4
        // absolute bar was written when ff=256 was the only shape and reads
        // 1.07e-4 at ff=704 on an UNCHANGED kernel - an absolute constant is
        // simply the wrong shape of gate here.
        let mut max_diff = 0.0f32;
        let mut max_ref = 0.0f32;
        for i in 0..batch * in_dim {
            max_diff = max_diff.max((out_ref[i] - out_gpu[i]).abs());
            max_ref = max_ref.max(out_ref[i].abs());
        }
        let rel = max_diff / max_ref.max(1e-9);
        eprintln!(
            "mma down+combine parity (in_dim={in_dim} ff={ff} bm={bm}): \
             max_abs_diff {max_diff:.2e} max_abs_ref {max_ref:.2e} rel {rel:.2e}"
        );
        // Measured 1.56e-7 / 1.68e-7 / 1.81e-7 across the three shapes (both
        // bm) - flat, i.e. f32-epsilon regrouping noise and nothing else. 1e-6
        // keeps ~5x headroom while staying far tighter than the absolute bar
        // it replaces.
        assert!(
            rel < 1e-6,
            "in_dim={in_dim} ff={ff} bm={bm} mma down diverges: rel {rel} (abs {max_diff})"
        );
    }
}

/// OCP e4m3 (E4M3FN) byte -> f32. No infinities: 0x7F/0xFF are the NaNs, and
/// the max finite is 448 (e=15, m=6).
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f32;
    if e == 0 {
        sign * m * (2.0f32).powi(-9) // subnormal: 2^-6 * m/8
    } else {
        sign * (1.0 + m / 8.0) * (2.0f32).powi(e - 7)
    }
}

/// Flat-scale e4m3 expert gate_up (change A) against a CPU reference
/// built from the ACTUAL device planes - the requantized e4m3 weights and the
/// e4m3 activations are read back and decoded here, so this gate is sharp on
/// layout/indexing (a wrong row, a swapped k-half or a mis-mapped D fragment
/// shows up immediately) and does not conflate the kernel with the precision
/// class. How lossy e4m3-per-row is against Q8_0 is a separate question that
/// only greedy parity on a real model can answer; this test deliberately says
/// nothing about it.
#[test]
fn f8row_moe_gate_up_matches_cpu() {
    let Some(exec) = common::gpu() else {
        return;
    };
    let (cc_major, cc_minor) = exec.compute_capability();
    if cc_major < 9 && !(cc_major == 8 && cc_minor >= 9) {
        eprintln!("no e4m3 mma on this device - skipping");
        return;
    }
    if !exec.has_f8row_moe() {
        eprintln!("pack lacks the flat-scale e4m3 expert lane - skipping");
        return;
    }
    let exec = Arc::new(exec);
    // Same three shapes as the Q8 mma gate: the clean case, A4B's real expert
    // width (ragged K guard), and the odd-scale-row/ragged-strip case.
    for (in_dim, ff) in [(256usize, 256usize), (256, 704), (512, 160)] {
        f8row_case(&exec, 8, 4, in_dim, ff, 37);
    }
}

fn f8row_case(
    exec: &Arc<GpuExecutor>,
    n_expert: usize,
    n_active: usize,
    in_dim: usize,
    ff: usize,
    batch: usize,
) {
    let mut seed = 0x51ce_2f8a_9d31_0007u64;
    let gate_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let up_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let (gate, _, _) = upload_repacked(exec, &gate_f, vec![in_dim, ff, n_expert]);
    let (up, _, _) = upload_repacked(exec, &up_f, vec![in_dim, ff, n_expert]);
    let rows = n_expert * ff;
    let gate_r = exec.q8_0_to_f8row_rows(&gate, rows).expect("gate -> f8row");
    let up_r = exec.q8_0_to_f8row_rows(&up, rows).expect("up -> f8row");
    // decode the planes the KERNEL will actually read
    let gwd = exec
        .to_host_range_u8(&gate_r.data, 0, rows * in_dim)
        .expect("gate f8 back");
    let gws = exec.to_host(&gate_r.scale).expect("gate rs back");
    let uwd = exec
        .to_host_range_u8(&up_r.data, 0, rows * in_dim)
        .expect("up f8 back");
    let uws = exec.to_host(&up_r.scale).expect("up rs back");

    // down half: [ff, in_dim, n_expert] -> one flat (n_expert*in_dim)-row
    // stream, K = ff
    let down_f: Vec<f32> = (0..n_expert * in_dim * ff)
        .map(|_| lcg(&mut seed))
        .collect();
    let (down, _, _) = upload_repacked(exec, &down_f, vec![ff, in_dim, n_expert]);
    let drows = n_expert * in_dim;
    let down_r = exec
        .q8_0_to_f8row_rows(&down, drows)
        .expect("down -> f8row");
    let dwd = exec
        .to_host_range_u8(&down_r.data, 0, drows * ff)
        .expect("down f8 back");
    let dws = exec.to_host(&down_r.scale).expect("down rs back");

    let x: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
    let idx_host: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i * 3 + 1) % n_expert) as u32)
        .collect();
    let w_host: Vec<f32> = (0..batch * n_active)
        .map(|i| 0.05 + 0.04 * (i % 9) as f32)
        .collect();
    let d_w = exec.to_device(&w_host).expect("w");
    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx_host).expect("idx");
    let mut d_xq = exec.alloc_u8(batch * in_dim).expect("xq");
    let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
    exec.quantize_e4m3_b32f(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
        .expect("quant e4m3");
    let xq_host = exec
        .to_host_range_u8(&d_xq, 0, batch * in_dim)
        .expect("xq back");
    let xs_host = exec.to_host(&d_xs).expect("xs back");

    const PAD: u32 = 0xFFFF_FFFF;
    for bm in [32usize, 64usize] {
        let max_blocks = (batch * n_active + n_expert * (bm - 1)).div_ceil(bm);
        let mut srow = exec.alloc_u32(max_blocks * bm).expect("srow");
        let mut sslot = exec.alloc_u32(max_blocks * bm).expect("sslot");
        let mut bexp = exec.alloc_u32(max_blocks).expect("bexp");
        if bm == 64 {
            exec.moe_align_bm(
                &d_idx, &mut srow, &mut sslot, &mut bexp, batch, n_active, n_expert, bm, max_blocks,
            )
            .expect("align_bm");
        } else {
            exec.moe_align(
                &d_idx, &mut srow, &mut sslot, &mut bexp, batch, n_active, n_expert, max_blocks,
            )
            .expect("align");
        }
        let mut d_fq: CudaSlice<i8> = exec.alloc_i8(max_blocks * bm * ff).expect("fq");
        let mut d_fs = exec.alloc(max_blocks * bm * ff / 32).expect("fs");
        exec.stream.memset_zeros(&mut d_fq).expect("zero fq");
        exec.stream.memset_zeros(&mut d_fs).expect("zero fs");
        // e4m3-out epilogue: this is the arm the flat-scale down half needs,
        // and it is the one the serve path runs when both halves are on, so it
        // is the one worth gating. The int8-out twin is the same GEMM.
        exec.f8row_moe_gate_up_mma_geglu_f8(
            &gate_r, &up_r, &srow, &bexp, &d_xq, &d_xs, &mut d_fq, &mut d_fs, in_dim, ff,
            max_blocks, bm,
        )
        .expect("f8row gate_up f8-out");
        let fq_host: Vec<i8> = exec.to_host_i8(&d_fq).expect("fq back");
        let fs_host = exec.to_host(&d_fs).expect("fs back");
        let srow_host = exec.to_host_u32(&srow).expect("srow back");
        let bexp_host = exec.to_host_u32(&bexp).expect("bexp back");

        // reference: per-32-scaled e4m3 activations x per-ROW-scaled e4m3
        // weights, gelu_tanh(gate)*up, compared after the kernel's own int8
        // requantize (so the bar is half a quant step, same as the Q8 twin).
        let dot = |wd: &[u8], wscale: f32, wrow: usize, srw: usize| -> f32 {
            let mut acc = 0.0f32;
            for b in 0..in_dim / 32 {
                let mut s = 0.0f32;
                for i in 0..32 {
                    s += e4m3_to_f32(wd[wrow + b * 32 + i])
                        * e4m3_to_f32(xq_host[srw * in_dim + b * 32 + i]);
                }
                acc += s * xs_host[srw * in_dim / 32 + b];
            }
            acc * wscale
        };
        let mut worst_ratio = 0.0f32;
        for blk in 0..max_blocks {
            let e = bexp_host[blk];
            if e == PAD {
                continue;
            }
            for r in 0..bm {
                let srw = srow_host[blk * bm + r];
                if srw == PAD {
                    continue;
                }
                let frow = blk * bm + r;
                for o in 0..ff {
                    let row = e as usize * ff + o;
                    let g = dot(&gwd, gws[row], row * in_dim, srw as usize);
                    let u = dot(&uwd, uws[row], row * in_dim, srw as usize);
                    let gelu =
                        0.5 * g * (1.0 + (0.797_884_6 * g * (1.0 + 0.044715 * g * g)).tanh());
                    let r_ref = gelu * u;
                    let scale = fs_host[frow * ff / 32 + o / 32];
                    // e4m3 out: the bar is the encoding's own error, which is
                    // RELATIVE (half an ulp = 2^-4) down to the point where a
                    // value stops being representable and flushes to zero --
                    // below half the smallest subnormal, 2^-10 * scale. Both
                    // terms are needed: a pure relative bound leaves no
                    // allowance at all for a value that legitimately rounds to
                    // 0, which is exactly where this first fired.
                    let deq = e4m3_to_f32(fq_host[frow * ff + o] as u8) * scale;
                    let tol = 0.0626 * r_ref.abs()
                        + 0.5 * (2.0f32).powi(-9) * scale.abs()
                        + 1e-3 * r_ref.abs()
                        + 1e-6;
                    let diff = (deq - r_ref).abs();
                    worst_ratio = worst_ratio.max(diff / tol.max(1e-9));
                    assert!(
                        diff <= tol,
                        "in_dim={in_dim} ff={ff} bm={bm} f8row gate_up out of quant \
                         tolerance: {deq} vs {r_ref} (tol {tol})"
                    );
                }
            }
        }
        eprintln!(
            "f8row gate_up parity (in_dim={in_dim} ff={ff} bm={bm}): \
             worst diff/tolerance ratio {worst_ratio:.3}"
        );

        // ---- down half over the same e4m3 fq/fs the gate_up just wrote ----
        let mut d_part = exec.alloc(batch * n_active * in_dim).expect("part");
        exec.f8row_moe_down_mma(
            &down_r,
            &srow,
            &sslot,
            &bexp,
            &d_w,
            &d_fq,
            &d_fs,
            &mut d_part,
            ff,
            in_dim,
            n_active,
            max_blocks,
            bm,
        )
        .expect("f8row down");
        let mut d_out = exec.alloc(batch * in_dim).expect("out");
        exec.stream.memset_zeros(&mut d_out).expect("zero");
        exec.moe_slot_combine(&d_part, &mut d_out, in_dim, n_active, batch)
            .expect("combine");
        let out_gpu = exec.to_host(&d_out).expect("out back");
        let sslot_host = exec.to_host_u32(&sslot).expect("sslot back");

        let mut out_ref = vec![0.0f32; batch * in_dim];
        for blk in 0..max_blocks {
            let e = bexp_host[blk];
            if e == PAD {
                continue;
            }
            for r in 0..bm {
                let srw = srow_host[blk * bm + r];
                if srw == PAD {
                    continue;
                }
                let slot = sslot_host[blk * bm + r] as usize;
                let pair = srw as usize * n_active + slot;
                let frow = blk * bm + r;
                for o in 0..in_dim {
                    let row = e as usize * in_dim + o;
                    let mut acc = 0.0f32;
                    for b in 0..ff / 32 {
                        let mut s = 0.0f32;
                        for i in 0..32 {
                            s += e4m3_to_f32(dwd[row * ff + b * 32 + i])
                                * e4m3_to_f32(fq_host[frow * ff + b * 32 + i] as u8);
                        }
                        acc += s * fs_host[frow * ff / 32 + b];
                    }
                    out_ref[srw as usize * in_dim + o] += w_host[pair] * acc * dws[row];
                }
            }
        }
        let mut max_diff = 0.0f32;
        let mut max_ref = 0.0f32;
        for i in 0..batch * in_dim {
            max_diff = max_diff.max((out_ref[i] - out_gpu[i]).abs());
            max_ref = max_ref.max(out_ref[i].abs());
        }
        let rel = max_diff / max_ref.max(1e-9);
        eprintln!(
            "f8row down+combine parity (in_dim={in_dim} ff={ff} bm={bm}): \
             max_abs_diff {max_diff:.2e} max_abs_ref {max_ref:.2e} rel {rel:.2e}"
        );
        // Same bar as the Q8 down gate: both sides do the identical e4m3
        // products, so the only difference left is f32 regrouping noise.
        assert!(
            rel < 1e-6,
            "in_dim={in_dim} ff={ff} bm={bm} f8row down diverges: rel {rel} (abs {max_diff})"
        );
    }
}

/// The nemotron_h_moe Q8_0 kernels - single-plane up with
/// squared-relu (no gate matrix), token-batched AND sorted, plus the
/// K-tail-guarded down_sorted. Dims are deliberately 32- but not 256-aligned
/// (in_dim 160, ff 96) so the sorted kernels' zero-staged tail chunks are on
/// the path - a fully-aligned run would leave the guards untested.
#[test]
fn q8_moe_up_relu2_matches_cpu() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_q8_0_moe_relu2() {
        common::missing("pack lacks q8_0_moe_up_relu2 (stale .so?)");
        return;
    }

    let (n_expert, n_active, in_dim, ff, batch) = (8usize, 6usize, 160usize, 96usize, 37usize);
    let mut seed = 0x172_172_172_172u64;
    let up_f: Vec<f32> = (0..n_expert * ff * in_dim)
        .map(|_| lcg(&mut seed))
        .collect();
    let down_f: Vec<f32> = (0..n_expert * in_dim * ff)
        .map(|_| lcg(&mut seed))
        .collect();
    let (up, up_q, up_s) = upload_repacked(&exec, &up_f, vec![in_dim, ff, n_expert]);
    let (down, down_q, down_s) = upload_repacked(&exec, &down_f, vec![ff, in_dim, n_expert]);

    let x: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
    let idx_host: Vec<u32> = (0..batch * n_active)
        .map(|i| ((i * 5 + 1) % n_expert) as u32)
        .collect();
    let w_host: Vec<f32> = (0..batch * n_active)
        .map(|i| 0.05 + 0.04 * (i % 9) as f32)
        .collect();
    let d_x = exec.to_device(&x).expect("x");
    let d_idx = exec.to_device_u32(&idx_host).expect("idx");
    let d_w = exec.to_device(&w_host).expect("w");
    let mut d_xq: CudaSlice<i8> = exec.alloc_i8(batch * in_dim).expect("xq");
    let mut d_xs = exec.alloc(batch * in_dim / 32).expect("xs");
    exec.quantize_q8(&d_x, &mut d_xq, &mut d_xs, batch * in_dim)
        .expect("quant");
    let xq_host: Vec<i8> = exec.to_host_i8(&d_xq).expect("xq back");
    let xs_host = exec.to_host(&d_xs).expect("xs back");

    let relu2 = |v: f32| {
        let r = v.max(0.0);
        r * r
    };

    // ---- token-batched pair ----
    let mut d_up_out = exec.alloc(batch * n_active * ff).expect("up out");
    exec.q8_0_moe_up_relu2(&up, &d_idx, &d_xq, &d_xs, &mut d_up_out, n_active, batch)
        .expect("up relu2");
    let up_gpu = exec.to_host(&d_up_out).expect("up back");
    let mut up_ref = vec![0.0f32; batch * n_active * ff];
    let mut max_diff = 0.0f32;
    for b in 0..batch {
        let xq_b = &xq_host[b * in_dim..][..in_dim];
        let xs_b = &xs_host[b * in_dim / 32..][..in_dim / 32];
        for s in 0..n_active {
            let e = idx_host[b * n_active + s] as usize;
            for o in 0..ff {
                let row = (e * ff + o) * in_dim;
                let u = ref_dot(
                    &up_q[row..][..in_dim],
                    &up_s[row / 32..][..in_dim / 32],
                    xq_b,
                    xs_b,
                );
                let r = relu2(u);
                up_ref[(b * n_active + s) * ff + o] = r;
                max_diff = max_diff.max((r - up_gpu[(b * n_active + s) * ff + o]).abs());
            }
        }
    }
    eprintln!("q8 up_relu2 token-batched parity: max_abs_diff {max_diff:.2e}");
    assert!(max_diff < 1e-4, "up_relu2 diverges: {max_diff}");

    let mut d_fq: CudaSlice<i8> = exec.alloc_i8(batch * n_active * ff).expect("fq");
    let mut d_fs = exec.alloc(batch * n_active * ff / 32).expect("fs");
    exec.quantize_q8(&d_up_out, &mut d_fq, &mut d_fs, batch * n_active * ff)
        .expect("quant f");
    let mut d_out = exec.alloc(batch * in_dim).expect("out");
    exec.q8_0_moe_down(
        &down, &d_idx, &d_w, &d_fq, &d_fs, &mut d_out, n_active, batch,
    )
    .expect("down");
    let out_tb = exec.to_host(&d_out).expect("out back");

    // ---- sorted pair on the same routing (tail-guarded kernels) ----
    let max_blocks = (batch * n_active + n_expert * 31).div_ceil(32);
    let mut srow = exec.alloc_u32(max_blocks * 32).expect("srow");
    let mut sslot = exec.alloc_u32(max_blocks * 32).expect("sslot");
    let mut bexp = exec.alloc_u32(max_blocks).expect("bexp");
    exec.moe_align(
        &d_idx, &mut srow, &mut sslot, &mut bexp, batch, n_active, n_expert, max_blocks,
    )
    .expect("align");
    let mut d_fused = exec.alloc(max_blocks * 32 * ff).expect("fused");
    exec.q8_0_moe_up_relu2_sorted(&up, &srow, &bexp, &d_xq, &d_xs, &mut d_fused, max_blocks)
        .expect("up relu2 sorted");
    let srow_host = exec.to_host_u32(&srow).expect("srow back");
    let sslot_host = exec.to_host_u32(&sslot).expect("sslot back");
    let bexp_host = exec.to_host_u32(&bexp).expect("bexp back");
    let fused_gpu = exec.to_host(&d_fused).expect("fused back");

    const PAD: u32 = 0xFFFF_FFFF;
    let mut max_diff = 0.0f32;
    for blk in 0..max_blocks {
        if bexp_host[blk] == PAD {
            continue;
        }
        for r in 0..32 {
            let srw = srow_host[blk * 32 + r];
            if srw == PAD {
                continue;
            }
            let slot = sslot_host[blk * 32 + r] as usize;
            let pair = srw as usize * n_active + slot;
            for o in 0..ff {
                let want = up_ref[pair * ff + o];
                let got = fused_gpu[(blk * 32 + r) * ff + o];
                max_diff = max_diff.max((want - got).abs());
            }
        }
    }
    eprintln!(
        "q8 up_relu2 sorted parity (K tail {}): max_abs_diff {max_diff:.2e}",
        in_dim % 256
    );
    assert!(max_diff < 1e-4, "sorted up_relu2 diverges: {max_diff}");

    let mut d_fq2: CudaSlice<i8> = exec.alloc_i8(max_blocks * 32 * ff).expect("fq2");
    let mut d_fs2 = exec.alloc(max_blocks * 32 * ff / 32).expect("fs2");
    exec.quantize_q8(&d_fused, &mut d_fq2, &mut d_fs2, max_blocks * 32 * ff)
        .expect("quant f2");
    let mut d_part = exec.alloc(batch * n_active * in_dim).expect("part");
    exec.q8_0_moe_down_sorted(
        &down,
        &srow,
        &sslot,
        &bexp,
        &d_w,
        &d_fq2,
        &d_fs2,
        &mut d_part,
        n_active,
        max_blocks,
    )
    .expect("down sorted");
    let mut d_out_s = exec.alloc(batch * in_dim).expect("out s");
    exec.stream.memset_zeros(&mut d_out_s).expect("zero");
    exec.moe_slot_combine(&d_part, &mut d_out_s, in_dim, n_active, batch)
        .expect("combine");
    let out_sorted = exec.to_host(&d_out_s).expect("out sorted back");

    // exact CPU reference over the sorted quantized fused rows (the down's
    // real input), exercising the ff-tail guard end to end
    let fq2_host: Vec<i8> = exec.to_host_i8(&d_fq2).expect("fq2 back");
    let fs2_host = exec.to_host(&d_fs2).expect("fs2 back");
    let mut out_ref = vec![0.0f32; batch * in_dim];
    for blk in 0..max_blocks {
        let e = bexp_host[blk];
        if e == PAD {
            continue;
        }
        for r in 0..32 {
            let srw = srow_host[blk * 32 + r];
            if srw == PAD {
                continue;
            }
            let slot = sslot_host[blk * 32 + r] as usize;
            let pair = srw as usize * n_active + slot;
            let frow = blk * 32 + r;
            for o in 0..in_dim {
                let wrow = (e as usize * in_dim + o) * ff;
                out_ref[srw as usize * in_dim + o] += w_host[pair]
                    * ref_dot(
                        &down_q[wrow..][..ff],
                        &down_s[wrow / 32..][..ff / 32],
                        &fq2_host[frow * ff..][..ff],
                        &fs2_host[frow * ff / 32..][..ff / 32],
                    );
            }
        }
    }
    let mut max_diff = 0.0f32;
    for i in 0..batch * in_dim {
        max_diff = max_diff.max((out_ref[i] - out_sorted[i]).abs());
    }
    eprintln!(
        "q8 relu2 sorted down parity (ff tail {}): max_abs_diff {max_diff:.2e}",
        ff % 256
    );
    assert!(
        max_diff < 1e-4,
        "tail-guarded down_sorted diverges: {max_diff}"
    );

    // the two lanes quantize the fused plane independently, so they are the
    // same CLASS, not the same bits - a loose band catches gross breakage
    // (wrong expert row / dropped tail) without faking exactness
    let mut max_band = 0.0f32;
    for i in 0..batch * in_dim {
        max_band = max_band.max((out_tb[i] - out_sorted[i]).abs());
    }
    eprintln!("q8 relu2 token-batched vs sorted band: {max_band:.2e}");
    assert!(
        max_band < 0.05,
        "lanes disagree beyond requantization noise: {max_band}"
    );

    // ---- the int8-MMA pair on the same routing: BITWISE vs the dp4a pair ----
    // (slot 743's relu^2 up writes fq/fs in its epilogue; the down is the
    // shared mma down). Same k32 integer dots, same scale fold order, same
    // relu^2 and per-32 rounding - so the quantized plane, and with it the
    // combined output, must be the same bits, tails included (in_dim 160 and
    // ff 96 put a ragged K chunk and a half-empty 64-row strip on the path).
    if !exec.has_q8_moe_relu2_mma() {
        common::missing("pack lacks q8_0_moe_up_relu2_mma (slot 743)");
        return;
    }
    let mut d_fq3: CudaSlice<i8> = exec.alloc_i8(max_blocks * 32 * ff).expect("fq3");
    let mut d_fs3 = exec.alloc(max_blocks * 32 * ff / 32).expect("fs3");
    exec.q8_0_moe_up_relu2_mma(
        &up, &srow, &bexp, &d_xq, &d_xs, &mut d_fq3, &mut d_fs3, max_blocks, 32,
    )
    .expect("up relu2 mma");
    let fq3_host: Vec<i8> = exec.to_host_i8(&d_fq3).expect("fq3 back");
    let fs3_host = exec.to_host(&d_fs3).expect("fs3 back");
    let fq_mismatch = fq3_host
        .iter()
        .zip(&fq2_host)
        .filter(|(a, b)| a != b)
        .count();
    let fs_mismatch = fs3_host
        .iter()
        .zip(&fs2_host)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("q8 relu2 mma vs dp4a: fq {fq_mismatch} / fs {fs_mismatch} mismatches");
    assert_eq!(
        (fq_mismatch, fs_mismatch),
        (0, 0),
        "relu2 mma up is not bitwise"
    );
    let mut d_part3 = exec.alloc(batch * n_active * in_dim).expect("part3");
    exec.q8_0_moe_down_mma(
        &down,
        &srow,
        &sslot,
        &bexp,
        &d_w,
        &d_fq3,
        &d_fs3,
        &mut d_part3,
        n_active,
        max_blocks,
        32,
    )
    .expect("down mma");
    let mut d_out_m = exec.alloc(batch * in_dim).expect("out m");
    exec.stream.memset_zeros(&mut d_out_m).expect("zero");
    exec.moe_slot_combine(&d_part3, &mut d_out_m, in_dim, n_active, batch)
        .expect("combine m");
    let out_mma = exec.to_host(&d_out_m).expect("out mma back");
    let out_mismatch = out_mma
        .iter()
        .zip(&out_sorted)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!("q8 relu2 mma pair vs dp4a pair: {out_mismatch} output mismatches");
    assert_eq!(out_mismatch, 0, "relu2 mma pair is not bitwise");
}
