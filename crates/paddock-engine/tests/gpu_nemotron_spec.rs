//! Nemotron verify-round gates: the three ways a speculative round resolves
//! its rows - device argmax (greedy), device sampling with one plan per row,
//! and raw logits for the service's host sampler with the commit handed
//! back through `spec_commit` - must drive the trunk identically, and a
//! commit that closes a page must leave the reply checkpoint a later turn
//! resumes from. No drafter is needed: the chunks are hand-built, which is
//! also what lets a test commit a round partway.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;
use paddock_engine::sampler::DevicePlan;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";
const MAX_CTX: usize = 2048;
/// 700 = 43 pages + 12 rows: an 8-row round from here closes page 44 at
/// its fourth row (cut 704)
const PROMPT_LEN: usize = 700;
const PAGE: usize = 16;

fn argmax(l: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

fn oracle_prompt(n: usize) -> Option<Vec<u32>> {
    let path = std::env::var("NEMOTRON_ORACLE").unwrap_or_else(|_| ORACLE.into());
    let raw = std::fs::read(&path)
        .or_else(|_| {
            std::fs::read(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../scripts/nemotron/oracle-wikitext-ids.json"
            ))
        })
        .ok()?;
    let oracle: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let seed: Vec<u32> = oracle["prompt_ids"]
        .as_array()?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    Some((0..n).map(|i| seed[i % seed.len()]).collect())
}

/// Load the target with 4 slots and the verify kernel set, or skip.
fn load() -> Option<(GpuNemotron, Vec<u32>)> {
    let exec = common::gpu_arc()?;
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
        || !exec.has_nemotron_prefill_f8()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
        || !exec.has_sample_rows()
    {
        common::missing("pack lacks the nemotron verify kernel set");
        return None;
    }
    let dir = common::model_dir(CKPT_ENV, &[CKPT_DIR])?;
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return None;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, MAX_CTX).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    Some((model, prompt))
}

/// Slot 0 prefills cold and decodes the greedy stream the chunks borrow;
/// slots 1..=3 then resume the same prompt from the same checkpoint over the
/// same tail rows, so they start bitwise equal (asserted - it is the premise
/// of every comparison below). Returns the stream, first token included.
fn three_equal_slots(model: &mut GpuNemotron, prompt: &[u32]) -> Vec<u32> {
    let cold = model.forward_prefill(0, prompt).expect("cold prefill");
    let mut b = vec![argmax(&cold)];
    for i in 0..10 {
        let (step, _) = model
            .forward_mixed_sampled(
                &[(0usize, b[i], (PROMPT_LEN + i) as u32)],
                usize::MAX,
                &[paddock_engine::generator::RowSample::Device(
                    DevicePlan::Greedy,
                )],
                &[],
            )
            .expect("baseline decode");
        b.push(step.ids[0]);
    }
    for slot in 1..=3 {
        let l = model
            .forward_prefill(slot, prompt)
            .expect("resumed prefill");
        assert_eq!(argmax(&l), b[0], "slot {slot} resumed off the stream");
    }
    let s1 = model.state_dump_probe(1);
    assert_eq!(s1, model.state_dump_probe(2), "slots 1/2 start apart");
    assert_eq!(s1, model.state_dump_probe(3), "slots 1/3 start apart");
    b
}

fn accepted(chunk: &[u32], picks: &[u32]) -> usize {
    let mut a = 0usize;
    while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
        a += 1;
    }
    a
}

#[test]
fn sampled_rounds_drive_the_trunk_like_the_greedy_round() {
    let Some((mut model, prompt)) = load() else {
        return;
    };
    let b = three_equal_slots(&mut model, &prompt);
    let vocab = model.vocab();

    // round 1: drafts off the greedy stream with a wrong token at draft 4,
    // so the round is partially accepted whatever the verify class picks
    // round 2: the full stream from the next pending, to exercise a longer
    // accept on top of round 1's rollback
    let mut pos = PROMPT_LEN;
    let mut pending = b[0];
    for round in 0..3 {
        let mut chunk = vec![pending];
        let base = pos - PROMPT_LEN;
        chunk.extend((1..8).map(|i| b.get(base + i).copied().unwrap_or(b[1])));
        if round == 0 {
            chunk[4] = chunk[4].wrapping_add(1) % vocab as u32;
        }
        let req = |slot: usize| vec![(slot, pos, chunk.clone())];

        let greedy = model
            .forward_spec_batch(&req(1))
            .expect("greedy verify")
            .expect("greedy engaged");
        let a = accepted(&chunk, &greedy);

        let rows = model
            .forward_spec_verify(&req(2))
            .expect("host verify")
            .expect("host engaged");
        assert_eq!(rows.len(), chunk.len() * vocab);
        let host: Vec<u32> = rows.chunks(vocab).map(argmax).collect();
        assert_eq!(
            host, greedy,
            "round {round}: host rows argmax off the device argmax"
        );
        model.spec_commit(&[(a + 1) as u32]).expect("host commit");

        // round 0 the classic plan, then the truncation plan at a nucleus
        // of one (mode 6 over the full vocab - still the argmax)
        let plan = if round == 0 {
            DevicePlan::Greedy
        } else {
            DevicePlan::TruncCat {
                inv_t: 1.0,
                u: 0.5,
                k: 0,
                top_p: 1e-9,
                min_p: 0.0,
            }
        };
        if matches!(plan, DevicePlan::TruncCat { .. }) && !model.supports_device_trunc() {
            common::missing("pack lacks device truncation sampling");
            return;
        }
        let planned = model
            .forward_spec_batch_plans(&req(3), &vec![plan; chunk.len()])
            .expect("plans verify")
            .expect("plans engaged");
        assert_eq!(
            planned, greedy,
            "round {round}: planned picks off the argmax"
        );

        let s1 = model.state_dump_probe(1);
        assert_eq!(
            s1,
            model.state_dump_probe(2),
            "round {round}: host commit left a different state"
        );
        assert_eq!(
            s1,
            model.state_dump_probe(3),
            "round {round}: planned commit left a different state"
        );
        eprintln!(
            "round {round}: {} of {} drafts accepted",
            a,
            chunk.len() - 1
        );
        pos += a + 1;
        pending = greedy[a];
    }

    // a round with no draft in it is a decode tick the slow way: declined,
    // on every entry, so the service takes the captured tick instead
    assert!(
        model
            .forward_spec_batch(&[(1usize, pos, vec![pending])])
            .expect("draft-less greedy")
            .is_none(),
        "a draft-less greedy round must decline"
    );
    assert!(
        model
            .forward_spec_verify(&[(2usize, pos, vec![pending])])
            .expect("draft-less host verify")
            .is_none(),
        "a draft-less host verify must decline"
    );

    // a verify the service never commits is abandoned by the next walk
    let two = vec![pending, b[1]];
    model
        .forward_spec_verify(&[(2usize, pos, two.clone())])
        .expect("open verify")
        .expect("engaged");
    model
        .forward_spec_batch(&[(1usize, pos, two)])
        .expect("greedy after an abandoned verify")
        .expect("engaged");
    assert!(
        model.spec_commit(&[1]).is_err(),
        "a commit after the round was abandoned must fail"
    );
}

#[test]
fn spec_commit_checkpoints_the_reply_at_the_page_edge() {
    let Some((mut model, prompt)) = load() else {
        return;
    };
    let b = three_equal_slots(&mut model, &prompt);
    let cut = PROMPT_LEN.div_ceil(PAGE) * PAGE;
    let edge = cut - PROMPT_LEN; // rows through the edge: 4
    // off the greedy stream: slot 0's decode already filed a checkpoint
    // under that key, and a commit's state is its fed tokens' whatever the
    // model would have picked
    let chunk: Vec<u32> = std::iter::once(b[0]).chain(2001..2008).collect();

    // slot 1 commits the whole round - the page closes at its 4th row, so
    // the reply checkpoint comes from that row's snapshot, mid-round
    model
        .forward_spec_verify(&[(1usize, PROMPT_LEN, chunk.clone())])
        .expect("verify 1")
        .expect("engaged");
    model.spec_commit(&[chunk.len() as u32]).expect("commit 1");
    let (c1, blob) = model
        .reply_ckpt_probe(1)
        .expect("a commit through a page edge must checkpoint the reply");
    assert_eq!(c1, cut, "checkpoint at the wrong cut");

    // slot 2 commits exactly through the edge: its live state after the
    // rollback is the state at the cut, rebuilt by the rollback's own code
    model
        .forward_spec_verify(&[(2usize, PROMPT_LEN, chunk.clone())])
        .expect("verify 2")
        .expect("engaged");
    model.spec_commit(&[edge as u32]).expect("commit 2");
    let live: Vec<f32> = model
        .state_dump_probe(2)
        .into_iter()
        .flat_map(|(_, s, w)| s.into_iter().chain(w))
        .collect();
    assert_eq!(blob.len(), live.len(), "blob layout off the state dump");
    let diff = blob
        .iter()
        .zip(&live)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        diff, 0,
        "mid-round checkpoint differs from the state at the cut in {diff} values"
    );

    // a later turn that extends the committed reply resumes at the cut
    let mut next: Vec<u32> = prompt.clone();
    next.extend(&chunk[..edge]);
    next.push(chunk[edge]);
    model.forward_prefill(3, &next).expect("next turn");
    assert_eq!(
        model.take_prefill_reused(3),
        cut,
        "the next turn must resume at the reply checkpoint"
    );
}

/// A verify row stands for one-row decode: it must score its token the way
/// the r=1 tick does. Two slots resume the same prompt from the same
/// checkpoint (bitwise equal state, asserted); one decodes eight tokens a row
/// at a time, the other verifies the same eight in one round, and the rows'
/// logits are compared. Reports max |d| and the decode's top-1/top-2 margin
/// per row; asserts every pick agrees wherever the margin clears the drift.
#[test]
fn verify_rows_score_like_one_row_decode() {
    let Some((mut model, prompt)) = load() else {
        return;
    };
    let vocab = model.vocab();
    model.forward_prefill(0, &prompt).expect("cold prefill");
    let l1 = model.forward_prefill(1, &prompt).expect("resume 1");
    model.forward_prefill(2, &prompt).expect("resume 2");
    assert_eq!(
        model.state_dump_probe(1),
        model.state_dump_probe(2),
        "slots 1/2 start apart"
    );
    let mut b = vec![argmax(&l1)];
    let mut dec: Vec<Vec<f32>> = Vec::new();
    for i in 0..8 {
        let (l, _) = model
            .forward_mixed(&[(1usize, b[i], (PROMPT_LEN + i) as u32)], usize::MAX)
            .expect("r=1 decode");
        b.push(argmax(&l));
        dec.push(l);
    }
    let ver = model
        .forward_spec_verify(&[(2usize, PROMPT_LEN, b[..8].to_vec())])
        .expect("verify")
        .expect("engaged");
    assert_eq!(ver.len(), 8 * vocab);
    let mut worst = 0.0f32;
    for (i, d) in dec.iter().enumerate() {
        let v = &ver[i * vocab..(i + 1) * vocab];
        let md = d
            .iter()
            .zip(v)
            .map(|(a, c)| (a - c).abs())
            .fold(0.0f32, f32::max);
        let mut top = d.to_vec();
        top.sort_by(|a, c| c.total_cmp(a));
        let margin = top[0] - top[1];
        eprintln!(
            "row {i}: max |d| {md:.5}  decode margin {margin:.4}  picks {} / {}",
            b[i + 1],
            argmax(v)
        );
        worst = worst.max(md);
        if margin > 2.0 * md {
            assert_eq!(argmax(v), b[i + 1], "row {i}: verify picked off decode");
        }
    }
    model.spec_commit(&[8]).expect("commit");
    eprintln!("verify vs r=1 decode: worst max |d| {worst:.5}");
    // the W16 decode class makes a verify row the one-row tick's own
    // computation, not an approximation of it
    if model.w16_class_probe(8) {
        assert_eq!(worst, 0.0, "a verify row left the one-row decode class");
    }
}

/// The W16 class's other half: a decode tick's row scores the same whatever
/// shares the tick. Slots 1 and 3 resume equal and step the same tokens -
/// slot 1 in two-row ticks beside slot 2 on other tokens, slot 3 alone - so
/// their logits must match bit for bit every step and their states after.
/// The two-row tick runs the fused add+norm prologue, the one-row tick the
/// plain chain: this is also the gate that keeps those one norm.
#[test]
fn decode_rows_score_the_same_alone_or_shared() {
    let Some((mut model, prompt)) = load() else {
        return;
    };
    if !model.w16_class_probe(2) {
        common::missing("pack lacks the W16 decode class");
        return;
    }
    let vocab = model.vocab();
    let l0 = model.forward_prefill(0, &prompt).expect("cold prefill");
    for s in 1..=3 {
        model.forward_prefill(s, &prompt).expect("resume");
    }
    assert_eq!(
        model.state_dump_probe(1),
        model.state_dump_probe(3),
        "slots 1/3 start apart"
    );
    let mut t = argmax(&l0);
    for i in 0..8usize {
        let pos = (PROMPT_LEN + i) as u32;
        let other = (t + 1 + 97 * i as u32) % vocab as u32;
        let (two, _) = model
            .forward_mixed(&[(1usize, t, pos), (2usize, other, pos)], usize::MAX)
            .expect("two-row tick");
        let (one, _) = model
            .forward_mixed(&[(3usize, t, pos)], usize::MAX)
            .expect("one-row tick");
        assert!(
            two[..vocab]
                .iter()
                .zip(&one)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "step {i}: a decode row's logits depend on the row beside it"
        );
        t = argmax(&one);
    }
    assert_eq!(
        model.state_dump_probe(1),
        model.state_dump_probe(3),
        "slots 1/3 parted after eight ticks"
    );
    eprintln!("decode rows: 8 ticks bit-exact alone vs shared");
}

/// ... and beside a prompt being admitted. A mixed tick's decode band runs
/// the class's kernels over the prefill chunk's lanes (the chunk takes the
/// prefill class over every row, the band's rows are recomputed over it),
/// so a stream's bits do not depend on a prompt joining the tick. Slots 1
/// and 3 resume equal; slot 3 decodes eight greedy steps in one-row ticks,
/// then slot 2 starts a chunked prefill of another prompt and slot 1 steps
/// the same tokens, each tick beside a 48-row chunk of it: every step's
/// logits must be slot 3's bit for bit, and the two slots' states equal
/// after.
#[test]
fn decode_rows_score_the_same_beside_a_prefill_chunk() {
    let Some((mut model, prompt)) = load() else {
        return;
    };
    if !model.w16_class_probe(1) {
        common::missing("pack lacks the W16 decode class");
        return;
    }
    let vocab = model.vocab();
    let l0 = model.forward_prefill(0, &prompt).expect("cold prefill");
    for s in [1usize, 3] {
        model.forward_prefill(s, &prompt).expect("resume");
    }
    assert_eq!(
        model.state_dump_probe(1),
        model.state_dump_probe(3),
        "slots 1/3 start apart"
    );
    let mut toks = vec![argmax(&l0)];
    let mut alone = Vec::new();
    for i in 0..8usize {
        let pos = (PROMPT_LEN + i) as u32;
        let (one, fin) = model
            .forward_mixed(&[(3usize, toks[i], pos)], usize::MAX)
            .expect("one-row tick");
        assert!(fin.is_empty());
        toks.push(argmax(&one));
        alone.push(one);
    }
    // another text (the prompt reversed - no prefix to resume off), longer
    // than eight 48-row chunks so every tick below carries one
    let other: Vec<u32> = prompt.iter().rev().take(600).copied().collect();
    model.prefill_begin(2, other).expect("prefill_begin");
    for i in 0..8usize {
        let pos = (PROMPT_LEN + i) as u32;
        let (band, fin) = model
            .forward_mixed(&[(1usize, toks[i], pos)], 48)
            .expect("mixed tick");
        assert!(
            fin.is_empty(),
            "tick {i}: the chunked prompt finished early"
        );
        assert!(
            band[..vocab]
                .iter()
                .zip(&alone[i])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "step {i}: a decode row's logits depend on the prefill chunk beside it"
        );
    }
    assert_eq!(
        model.state_dump_probe(1),
        model.state_dump_probe(3),
        "slots 1/3 parted after eight ticks"
    );
    eprintln!("decode band: 8 mixed ticks bit-exact with the one-row ticks");
}
