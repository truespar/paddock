//! The pinned small-upload ring (`GpuExecutor::htod`): uploads queued behind
//! a long stream op, wrapping the ring while every DMA is still pending,
//! must each land their own bytes.
#![allow(clippy::unwrap_used)]

mod common;

#[test]
fn htod_ring_keeps_pending_spans_through_wraps() {
    let Some(exec) = common::gpu() else { return };
    // In bytes, against the 16 MB ring: 254 x 64 KB fill a lap to 15.9 MB;
    // ten 1.5 MB spans then wrap and stop at 15 MB, short of the first
    // lap's tail; the next 1.5 MB wraps again onto the second lap's first
    // span while the first lap's tail is still the oldest in flight - the
    // case where an overlap check of the oldest span alone reuses pinned
    // bytes a queued DMA has not read yet.
    let mut sizes: Vec<usize> = vec![64 << 10; 254];
    sizes.extend(std::iter::repeat_n(1536 << 10, 11));
    // and an irregular tail through a few more wraps
    sizes.extend((0..24).map(|i| (1 << 20) + (i * 37_123) % (1 << 19)));
    let words: Vec<usize> = sizes.iter().map(|b| b / 4).collect();
    let mut dst: Vec<cudarc::driver::CudaSlice<u32>> = words
        .iter()
        .map(|&n| exec.stream.alloc_zeros::<u32>(n).unwrap())
        .collect();
    let mut hold = exec.stream.alloc_zeros::<u8>(3 << 30).unwrap();
    let word = |rep: u32, i: usize, j: usize| (j as u32) ^ ((i as u32) << 20) ^ (rep << 29);
    for rep in 0..2u32 {
        // a stream op every upload below queues behind (~10 ms)
        exec.stream.memset_zeros(&mut hold).unwrap();
        for (i, (&n, d)) in words.iter().zip(dst.iter_mut()).enumerate() {
            let v: Vec<u32> = (0..n).map(|j| word(rep, i, j)).collect();
            exec.htod(&v, d).unwrap();
        }
        exec.stream.synchronize().unwrap();
        for (i, (&n, d)) in words.iter().zip(&dst).enumerate() {
            let h = exec.stream.clone_dtoh(d).unwrap();
            let bad = (0..n).filter(|&j| h[j] != word(rep, i, j)).count();
            assert_eq!(bad, 0, "rep {rep} upload {i}: {bad} of {n} words wrong");
        }
    }
    println!(
        "htod ring: 2 x {} uploads ({} MB a pass) behind a 3 GB memset, every word landed",
        sizes.len(),
        sizes.iter().sum::<usize>() >> 20
    );
}
