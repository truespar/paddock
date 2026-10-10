//! `pd_embed_gather_q8r` (slot 832) against `pd_embed_gather_q8`, bit for bit:
//! the same Q8_0 table gathered once from its raw 34-byte blocks and once
//! from the planes `pd_q8_0_repack` writes - what a tied qwen35 file now reads
//! its token rows from (the LM head's own plane) instead of a second copy.
//!
//! The repacked side goes through the real repack kernel, so the test pins
//! the layout contract the gather relies on (block order unchanged, int8
//! plane + f16 scale plane), not a hand-built image of it. Tokens include
//! repeats, the first and last rows and an out-of-order run; both scales the
//! engine uses (1.0, and gemma4's sqrt(n_embd)) are checked.
//!
//! Gated on: CUDA device + built pack.

mod common;

use half::f16;
use paddock_engine::gpu::QuantTensor;
use paddock_models::ggml_type::GgmlType;

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

/// A raw GGUF Q8_0 block stream for an `[embd, rows]` table.
fn raw_q8(embd: usize, rows: usize) -> Vec<u8> {
    let nb = embd / 32 * rows;
    let q = det(nb * 32, 11 + embd as u64);
    let d = det(nb, 17 + rows as u64);
    let mut out = Vec::with_capacity(nb * 34);
    for b in 0..nb {
        out.extend_from_slice(&f16::from_f32(d[b] * 0.03 + 0.009).to_le_bytes());
        out.extend(
            q[b * 32..(b + 1) * 32]
                .iter()
                .map(|v| (v * 254.0) as i8 as u8),
        );
    }
    out
}

#[test]
fn repacked_gather_is_bit_identical_to_the_raw_gather() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_embed_gather_q8r() {
        panic!("the pack has no slot 832 (embed_gather_q8r) - rebuild packs/cuda");
    }
    for (embd, rows) in [(2560usize, 1031usize), (1024, 517), (96, 7)] {
        let raw = raw_q8(embd, rows);
        let table = QuantTensor {
            bytes: exec.to_device_u8(&raw).expect("raw"),
            ty: GgmlType::Q8_0,
            dims: vec![embd, rows],
        };
        let rep = exec
            .repack_q8_blocks(&raw, vec![embd, rows])
            .expect("repack");
        let last = rows as u32 - 1;
        let toks: Vec<u32> = vec![0, last, 3, 3, last / 2, 1, last, 0, 2];
        let d_toks = exec.to_device_u32(&toks).expect("tokens");
        for scale in [1.0f32, (embd as f32).sqrt()] {
            let n = toks.len();
            let mut a = exec.alloc(n * embd).expect("a");
            let mut b = exec.alloc(n * embd).expect("b");
            exec.embed_gather_q8(&table, &d_toks, &mut a, embd, n, scale)
                .expect("raw gather");
            exec.embed_gather_q8r(&rep, &d_toks, &mut b, embd, n, scale)
                .expect("repacked gather");
            let (a, b) = (exec.to_host(&a).unwrap(), exec.to_host(&b).unwrap());
            let diffs = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diffs, 0,
                "embd {embd} rows {rows} scale {scale}: {diffs} values differ"
            );
            assert!(a.iter().any(|v| *v != 0.0), "the table gathered to zeros");
        }
    }
}

#[test]
fn a_row_width_off_the_block_grid_is_refused() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_embed_gather_q8r() {
        return;
    }
    let raw = raw_q8(64, 4);
    let rep = exec.repack_q8_blocks(&raw, vec![64, 4]).expect("repack");
    let d_toks = exec.to_device_u32(&[1]).expect("tokens");
    let mut out = exec.alloc(64).expect("out");
    // the table is 64 wide; asking for 48 must be an error, not a misread
    assert!(
        exec.embed_gather_q8r(&rep, &d_toks, &mut out, 48, 1, 1.0)
            .is_err()
    );
}
