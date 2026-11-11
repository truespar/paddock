//! Parity for `argmax_top2_rows`  - the kernel every whisper
//! decode step now ends on.
//!
//! Four properties matter, and they are different in kind:
//!
//! 1. The PICK must not MOVE. It is the same greedy token `argmax_rows`
//!    returns, ties included. If asking for confidence could change which
//!    token a transcript contains, the feature would be a correctness hazard
//!    rather than a readout - so the two kernels are run side by side on the
//!    same logits and required to agree exactly.
//!
//! 2. The numbers must be the log-softmax. Compared against a host
//!    log-sum-exp, not against a previous run of the same kernel.
//!
//! 3. The RUNNER-UP must be the SECOND-BEST, exactly. A top-2 that gets rank 1
//!    right and rank 2 wrong passes every gate the confidence readout had
//!    before this rung existed - the margin would simply be a plausible number
//!    that is not the margin. It gets its own case because of that, not
//!    because top-2 is hard.
//!
//! 4. The entropy must be the Renyi-2 of the actual distribution, including at
//!    the two ends (a peaked row reads ~0, a uniform row reads log n) - which
//!    is the property Shannon over a 51866-vocab tail gets backwards, and the
//!    reason this column exists.
//!
//! Light (no model load).

mod common;

use paddock_engine::gpu::GpuExecutor;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0
        })
        .collect()
}

/// Host log-softmax at one index, in f64 so the oracle is not the thing under
/// test.
fn host_logprob(row: &[f32], at: usize) -> f64 {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
    (row[at] as f64 - m) - s.ln()
}

/// Renyi-2 (collision) entropy in nats, in f64.
fn host_entropy2(row: &[f32]) -> f64 {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = row.iter().map(|&v| (v as f64 - m).exp()).sum();
    let s2: f64 = row.iter().map(|&v| ((v as f64 - m) * 2.0).exp()).sum();
    2.0 * s.ln() - s2.ln()
}

/// `(top1, top2)` with the kernel's tie rule: lowest index wins at both ranks.
fn host_top2(row: &[f32]) -> (usize, usize) {
    let mut a = 0usize;
    let mut b = usize::MAX;
    for i in 1..row.len() {
        if row[i] > row[a] {
            b = a;
            a = i;
        } else if b == usize::MAX || row[i] > row[b] {
            b = i;
        }
    }
    (a, b)
}

fn exec() -> Option<GpuExecutor> {
    common::gpu()
}

#[test]
fn matches_the_host_log_softmax_and_never_moves_the_pick() {
    let Some(e) = exec() else { return };
    // whisper's real vocabulary width, and a batch spanning the shapes the
    // decode pool actually runs at
    let vocab = 51866usize;
    let probe = 50363u32; // `<|nospeech|>`'s id on every released whisper
    for rows in [1usize, 3, 8, 32] {
        let host: Vec<f32> = det(rows * vocab, 0x5eed ^ rows as u64);
        let d_logits = e.to_device(&host).expect("upload");
        let mut d_idx = e.to_device_u32(&vec![0u32; rows]).expect("idx");
        let mut d_alt = e.to_device_u32(&vec![0u32; rows]).expect("alt");
        let mut d_plain = e.to_device_u32(&vec![0u32; rows]).expect("plain");
        let mut d_stats = e.alloc(rows * 4).expect("stats");

        e.argmax_top2_rows(
            &d_logits,
            &mut d_idx,
            &mut d_alt,
            &mut d_stats,
            probe,
            None,
            rows,
            vocab,
        )
        .expect("argmax_top2_rows");
        e.argmax_rows(&d_logits, &mut d_plain, rows, vocab)
            .expect("argmax_rows");
        let idx = e.to_host_u32(&d_idx).expect("idx back");
        let alt = e.to_host_u32(&d_alt).expect("alt back");
        let plain = e.to_host_u32(&d_plain).expect("plain back");
        let stats = e.to_host_len(&d_stats, rows * 4).expect("stats back");

        for r in 0..rows {
            let row = &host[r * vocab..(r + 1) * vocab];
            let (want1, want2) = host_top2(row);
            assert_eq!(idx[r], plain[r], "rows={rows} r={r}: the pick moved");
            assert_eq!(idx[r] as usize, want1, "rows={rows} r={r}: not the argmax");
            assert_eq!(
                alt[r] as usize, want2,
                "rows={rows} r={r}: not the runner-up"
            );

            let want_lp = host_logprob(row, want1);
            let got_lp = stats[r * 4] as f64;
            assert!(
                (got_lp - want_lp).abs() < 2e-3,
                "rows={rows} r={r}: log p {got_lp} vs host {want_lp}"
            );
            // a log-probability is never positive, and this one is a real
            // distribution over 51866 tokens so it is nowhere near 0 either
            assert!(
                got_lp < 0.0,
                "rows={rows} r={r}: log p {got_lp} is not a log-probability"
            );

            let want_probe = host_logprob(row, probe as usize).exp();
            let got_probe = stats[r * 4 + 1] as f64;
            assert!(
                (got_probe - want_probe).abs() < 1e-5,
                "rows={rows} r={r}: p(probe) {got_probe} vs host {want_probe}"
            );

            let want_lp2 = host_logprob(row, want2);
            let got_lp2 = stats[r * 4 + 2] as f64;
            assert!(
                (got_lp2 - want_lp2).abs() < 2e-3,
                "rows={rows} r={r}: log p2 {got_lp2} vs host {want_lp2}"
            );
            // the whole point of the second rank: it is below the first, so a
            // margin is never negative
            assert!(
                got_lp2 <= got_lp,
                "rows={rows} r={r}: runner-up outranks the pick"
            );

            let want_h2 = host_entropy2(row);
            let got_h2 = stats[r * 4 + 3] as f64;
            assert!(
                (got_h2 - want_h2).abs() < 2e-3,
                "rows={rows} r={r}: H2 {got_h2} vs host {want_h2}"
            );
        }
    }
}

#[test]
fn a_peaked_row_reads_as_certain_and_a_flat_one_as_not() {
    let Some(e) = exec() else { return };
    // The property the UI's colour bands rest on: confidence has to track how
    // concentrated the distribution actually is, not just be *a* number.
    let vocab = 4096usize;
    let mut host = vec![0.0f32; 2 * vocab]; // row 1 stays uniform
    host[7] = 30.0; // row 0: one token takes essentially all the mass
    let d_logits = e.to_device(&host).expect("upload");
    let mut d_idx = e.to_device_u32(&[0u32; 2]).expect("idx");
    let mut d_alt = e.to_device_u32(&[0u32; 2]).expect("alt");
    let mut d_stats = e.alloc(8).expect("stats");
    e.argmax_top2_rows(
        &d_logits,
        &mut d_idx,
        &mut d_alt,
        &mut d_stats,
        vocab as u32,
        None,
        2,
        vocab,
    )
    .expect("kernel");
    let idx = e.to_host_u32(&d_idx).expect("idx back");
    let alt = e.to_host_u32(&d_alt).expect("alt back");
    let stats = e.to_host_len(&d_stats, 8).expect("stats back");

    assert_eq!(idx[0], 7);
    assert!(
        stats[0].exp() > 0.999,
        "peaked row read {} ",
        stats[0].exp()
    );
    // uniform over 4096: every token is 1/4096, so ln(1/4096) = -8.317
    assert!(
        (stats[4] - (1.0f32 / vocab as f32).ln()).abs() < 1e-3,
        "flat row read {}, want {}",
        stats[4],
        (1.0f32 / vocab as f32).ln()
    );
    // probe out of range means "no probe", not "probability zero by accident"
    assert_eq!(stats[1], 0.0);
    assert_eq!(stats[5], 0.0);

    // Entropy at the two ends. This is the column's whole job: a peaked row is
    // ~0 nats and a uniform one is ln(vocab), and any tail-dominated measure
    // would put them the other way round.
    assert!(
        stats[3].abs() < 1e-3,
        "peaked row H2 {} should be ~0",
        stats[3]
    );
    assert!(
        (stats[7] - (vocab as f32).ln()).abs() < 1e-3,
        "uniform row H2 {}, want {}",
        stats[7],
        (vocab as f32).ln()
    );

    // Ties resolve to the LOWEST index at both ranks. Row 1 is uniform, so
    // every logit is a tie and the answer is fully determined by the rule -
    // which makes this the sharpest possible check that the cross-warp merge
    // did not reorder anything.
    assert_eq!(
        idx[1], 0,
        "uniform row picked {} instead of index 0",
        idx[1]
    );
    assert_eq!(
        alt[1], 1,
        "uniform row's runner-up was {} instead of 1",
        alt[1]
    );
    // and the peaked row's runner-up is the lowest-index loser, all of which
    // are still tied with each other at 0.0
    assert_eq!(
        alt[0], 0,
        "peaked row's runner-up was {} instead of 0",
        alt[0]
    );
}

#[test]
fn a_two_way_call_and_a_diffuse_row_are_told_apart() {
    let Some(e) = exec() else { return };
    // The reason the runner-up is worth a kernel change at all. Both rows have
    // a middling top-1 probability, so max-prob alone rates them the same; the
    // margin says one was a coin flip between two words and the other was
    // merely spread thin, and only the first is where ASR errors live.
    let vocab = 4096usize;
    let mut host = vec![-20.0f32; 2 * vocab];
    // row 0: two candidates neck and neck, everything else nowhere
    host[100] = 1.00;
    host[200] = 0.99;
    // row 1: one leader over a broad shallow field. -8.5 across 4095 rivals
    // puts ~45% of the mass in the tail, so the leader lands near row 0's top-1
    // while no single rival comes close to it.
    for i in 0..vocab {
        host[vocab + i] = -8.5;
    }
    host[vocab + 300] = 0.0;
    let d_logits = e.to_device(&host).expect("upload");
    let mut d_idx = e.to_device_u32(&[0u32; 2]).expect("idx");
    let mut d_alt = e.to_device_u32(&[0u32; 2]).expect("alt");
    let mut d_stats = e.alloc(8).expect("stats");
    e.argmax_top2_rows(
        &d_logits,
        &mut d_idx,
        &mut d_alt,
        &mut d_stats,
        vocab as u32,
        None,
        2,
        vocab,
    )
    .expect("kernel");
    let idx = e.to_host_u32(&d_idx).expect("idx back");
    let alt = e.to_host_u32(&d_alt).expect("alt back");
    let stats = e.to_host_len(&d_stats, 8).expect("stats back");

    assert_eq!((idx[0], alt[0]), (100, 200));
    assert_eq!((idx[1], alt[1]), (300, 0));

    let margin = |r: usize| stats[r * 4].exp() - stats[r * 4 + 2].exp();
    // the coin flip: two tokens ~0.5 each, so the margin is nearly nothing
    assert!(
        margin(0) < 0.02,
        "two-way row margin {} should be ~0",
        margin(0)
    );
    // the diffuse row: the leader beats every individual rival outright
    assert!(
        margin(1) > 0.3,
        "diffuse row margin {} should be clear",
        margin(1)
    );
    // ...and max-prob cannot tell them apart, which is the finding this test
    // pins: without the second rank both rows look equally (un)certain
    let p1 = |r: usize| stats[r * 4].exp();
    assert!(
        (p1(0) - p1(1)).abs() < 0.15,
        "the two rows' top-1 probabilities ({}, {}) are supposed to be close - \
         if they drift apart this test stops proving the margin adds anything",
        p1(0),
        p1(1)
    );

    // The two signals are COMPLEMENTARY, which is why both are wanted: the
    // margin fires on row 0 (torn between two words) and is quiet on row 1,
    // while the entropy does the opposite (row 1 has half its mass smeared
    // across 4095 tokens). Either one alone misses a real kind of uncertainty.
    let h2 = |r: usize| stats[r * 4 + 3];
    assert!(
        h2(1) > h2(0),
        "the diffuse row's entropy ({}) should exceed the two-way row's ({})",
        h2(1),
        h2(0)
    );
}

#[test]
fn suppressed_ids_are_never_picked_and_the_probe_still_sees_them() {
    let Some(e) = exec() else { return };
    // The whisper case this exists for: a near-empty window where the model's
    // own argmax is `<|nospeech|>`. The reference masks the prompt block and
    // `<|nospeech|>` before the softmax, so the pick, the runner-up, their
    // log-probabilities and H2 are those of the SUPPRESSED distribution -
    // while `no_speech_prob` keeps reading the model's unsuppressed belief,
    // which is what the no-speech guard is defined on.
    let vocab = 51866usize;
    let probe = 50363u32;
    let mask: Vec<u32> = vec![50258, 50361, 50362, 50363, 50364, 50360, 50359];
    let rows = 3usize;
    let mut host: Vec<f32> = det(rows * vocab, 0xa11ce);
    // row 0: `<|nospeech|>` wins outright (far above the ~e^12.8 the 51k
    // random tail sums to), a text token is second
    host[50363] = 20.0;
    host[1000] = 9.0;
    // row 1: `<|nospeech|>` AND `<|startoftranscript|>` beat the field
    host[vocab + 50363] = 20.0;
    host[vocab + 50258] = 19.0;
    // row 2: nothing suppressed is near the top
    host[2 * vocab + 777] = 11.0;
    let d_logits = e.to_device(&host).expect("upload");
    let d_mask = e.to_device_u32(&mask).expect("mask");
    let mut d_idx = e.to_device_u32(&vec![0u32; rows]).expect("idx");
    let mut d_alt = e.to_device_u32(&vec![0u32; rows]).expect("alt");
    let mut d_stats = e.alloc(rows * 4).expect("stats");
    e.argmax_top2_rows(
        &d_logits,
        &mut d_idx,
        &mut d_alt,
        &mut d_stats,
        probe,
        Some(&d_mask),
        rows,
        vocab,
    )
    .expect("kernel");
    let idx = e.to_host_u32(&d_idx).expect("idx back");
    let alt = e.to_host_u32(&d_alt).expect("alt back");
    let stats = e.to_host_len(&d_stats, rows * 4).expect("stats back");

    for r in 0..rows {
        let row = &host[r * vocab..(r + 1) * vocab];
        // the oracle: the same row with the mask applied as -inf
        let mut masked = row.to_vec();
        for &m in &mask {
            masked[m as usize] = f32::NEG_INFINITY;
        }
        let (want1, want2) = host_top2(&masked);
        assert_eq!(idx[r] as usize, want1, "r={r}: pick");
        assert_eq!(alt[r] as usize, want2, "r={r}: runner-up");
        assert!(
            !mask.contains(&idx[r]) && !mask.contains(&alt[r]),
            "r={r}: a suppressed id came out"
        );
        let lp = host_logprob(&masked, want1);
        assert!(
            (stats[r * 4] as f64 - lp).abs() < 2e-3,
            "r={r}: log p {} vs {lp}",
            stats[r * 4]
        );
        let lp2 = host_logprob(&masked, want2);
        assert!(
            (stats[r * 4 + 2] as f64 - lp2).abs() < 2e-3,
            "r={r}: log p2 {} vs {lp2}",
            stats[r * 4 + 2]
        );
        let h2 = host_entropy2(&masked);
        assert!(
            (stats[r * 4 + 3] as f64 - h2).abs() < 2e-3,
            "r={r}: H2 {} vs {h2}",
            stats[r * 4 + 3]
        );
        // the probe reads the UNSUPPRESSED row
        let want_probe = host_logprob(row, probe as usize).exp();
        let got_probe = stats[r * 4 + 1] as f64;
        assert!(
            (got_probe - want_probe).abs() < 1e-5,
            "r={r}: p(probe) {got_probe} vs {want_probe}"
        );
    }
    assert_eq!(idx[0], 1000, "row 0 falls through to the best text token");
    assert!(
        stats[1] > 0.9,
        "row 0's no_speech_prob {} must stay the model's own",
        stats[1]
    );
}
