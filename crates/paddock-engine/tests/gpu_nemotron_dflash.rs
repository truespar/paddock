//! DFlash drafter gates for nemotron (C2): attach the official
//! `nvidia/...-NVFP4-DFlash` checkpoint to the NVFP4 target (its trained
//! pairing), verify feature coverage flows from the batched walks, drafts
//! are deterministic, and - the invariant that matters - a full spec loop
//! (draft -> trunk verify -> accept walk -> commit) produces exactly the
//! no-spec greedy stream. Acceptance quality is reported, not asserted
//! (it is the model's business); correctness is asserted.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "NEMOTRON_NVFP4_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
const DFLASH_ENV: &str = "NEMOTRON_DFLASH_DIR";
const DFLASH_DIR: &str = "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";
const PROMPT_LEN: usize = 700;

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
    let raw = std::fs::read(&path).ok()?;
    let oracle: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let seed: Vec<u32> = oracle["prompt_ids"]
        .as_array()?
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    Some((0..n).map(|i| seed[i % seed.len()]).collect())
}

#[test]
fn dflash_spec_loop_matches_greedy() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv()
        || !exec.has_mamba2_batch()
        || !exec.has_nvf4_gemv_batch()
        || !exec.has_nvf4_ckpt()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
    {
        common::missing("pack lacks the nemotron dflash kernel set");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let df_dir = std::env::var(DFLASH_ENV).unwrap_or_else(|_| DFLASH_DIR.into());
    if !std::path::Path::new(&df_dir)
        .join("model.safetensors")
        .exists()
    {
        eprintln!("skip: no dflash checkpoint at {df_dir}");
        return;
    }
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    model
        .attach_dflash(std::path::Path::new(&df_dir))
        .expect("attach dflash");
    assert!(
        model.spec_capable(),
        "drafter attached + verify kernels => spec capable"
    );
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    // slot 0 walks the prompt cold and plants the checkpoints; slots 1 and 2
    // then resume it from the same checkpoint over the same tail rows, so
    // they start bitwise equal - asserted, it is the premise. A resume is
    // NOT the cold walk's state: the tail rows ride a narrower prefill pass
    // (different GEMM widths, different rounding), and near a greedy near-tie
    // that parts the streams without either being wrong. So the no-spec
    // baseline decodes on slot 1 and the spec loop runs on its twin, slot 2,
    // which drafts warm off the rows its pages carry.
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    let b0 = argmax(&l0);
    assert!(
        model
            .spec_ensure_warm(0, &[], (PROMPT_LEN - 1) as u32)
            .expect("warm probe"),
        "slot 0 must be feature-warm after its full prefill"
    );
    let l1 = model.forward_prefill(1, &prompt).expect("prefill 1");
    let l2 = model.forward_prefill(2, &prompt).expect("prefill 2");
    assert_eq!(argmax(&l1), b0);
    assert_eq!(
        model.state_dump_probe(1),
        model.state_dump_probe(2),
        "slots 1/2 start apart"
    );
    assert!(
        l1.iter().zip(&l2).all(|(a, c)| a.to_bits() == c.to_bits()),
        "slots 1/2 resumed to different logits"
    );
    assert!(
        model
            .spec_ensure_warm(2, &[], (PROMPT_LEN - 1) as u32)
            .expect("warm probe"),
        "slot 2 must draft warm off the resumed pages"
    );
    let mut b = vec![argmax(&l1)];
    for i in 0..32 {
        let (step, _) = model
            .forward_mixed_sampled(
                &[(1usize, b[i], (PROMPT_LEN + i) as u32)],
                usize::MAX,
                &[paddock_engine::generator::RowSample::Device(
                    paddock_engine::sampler::DevicePlan::Greedy,
                )],
                &[],
            )
            .expect("baseline decode");
        b.push(step.ids[0]);
    }

    // drafts must be deterministic (the laguna selftest property)
    let d1 = model
        .spec_draft_batch(&[(2usize, b[0])], 4)
        .expect("draft")
        .expect("engaged");
    let d2 = model
        .spec_draft_batch(&[(2usize, b[0])], 4)
        .expect("draft 2")
        .expect("engaged");
    assert_eq!(d1, d2, "repeat drafts diverged (ring append race?)");
    assert_eq!(d1[0].len(), 4);
    let hits = d1[0].iter().zip(&b[1..5]).filter(|(a, c)| a == c).count();
    eprintln!(
        "first-round drafts {:?} vs truth {:?} - {hits}/4 on-stream",
        d1[0],
        &b[1..5]
    );

    // full spec loop on slot 2: committed stream must equal the no-spec
    // stream REGARDLESS of draft quality - verify rows are the one-row
    // decode class, so this is exact, not a band
    let mut committed: Vec<u32> = Vec::new();
    let mut pending = b[0];
    let mut pos = PROMPT_LEN;
    let mut drafted = 0usize;
    let mut accepted = 0usize;
    let mut rounds = 0usize;
    while committed.len() < 24 {
        rounds += 1;
        let k = 7usize;
        let drafts = model
            .spec_draft_batch(&[(2usize, pending)], k)
            .expect("draft")
            .expect("engaged");
        let mut chunk = vec![pending];
        chunk.extend(&drafts[0]);
        drafted += drafts[0].len();
        let picks = model
            .forward_spec_batch(&[(2usize, pos, chunk.clone())])
            .expect("verify")
            .expect("engaged");
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        accepted += a;
        // service semantics: rows 0..=a of the chunk are committed, and the
        // pick after the last accepted row becomes the next pending
        for i in 0..=a {
            committed.push(if i == 0 { chunk[0] } else { chunk[i] });
        }
        pending = picks[a];
        pos += a + 1;
        // the committed tokens so far must be the no-spec stream
        assert_eq!(
            &committed[..],
            &b[..committed.len()],
            "spec-committed stream diverged from the no-spec greedy stream"
        );
        assert_eq!(pending, b[committed.len()], "next pending off-stream");
    }
    eprintln!(
        "spec loop: {} committed in {rounds} rounds (acceptance length {:.2}), drafted {drafted}, accepted {accepted} ({:.0}%)",
        committed.len(),
        committed.len() as f64 / rounds as f64,
        100.0 * accepted as f64 / drafted.max(1) as f64
    );
}

/// The drafter's rows ride the pool pages: a conversation's next turn,
/// prefix-resumed in ANY slot, adopts the drafter's rows with the pages and
/// drafts warm from the resume point - the agent loop, where a turn often
/// lands in a slot that served someone else meanwhile. Both slots resumed
/// over the same pages must draft the same block, and a prompt that parts
/// from the cached tokens stays warm below the part.
#[test]
fn dflash_coverage_rides_the_pages() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv() || !exec.has_spec_verify_mamba() || !exec.has_argmax_rows() {
        common::missing("pack lacks the nemotron dflash kernel set");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let df_dir = std::env::var(DFLASH_ENV).unwrap_or_else(|_| DFLASH_DIR.into());
    if !std::path::Path::new(&df_dir)
        .join("model.safetensors")
        .exists()
    {
        eprintln!("skip: no dflash checkpoint at {df_dir}");
        return;
    }
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    model
        .attach_dflash(std::path::Path::new(&df_dir))
        .expect("attach dflash");
    assert_eq!(model.batch_enable_probe(2).expect("enable"), 2);
    let depth = model.spec_fixed_draft_depth().expect("a fixed depth");
    let warm = |m: &mut GpuNemotron, slot: usize, len: usize| {
        m.spec_ensure_warm(slot, &[], (len - 1) as u32)
            .expect("warm probe")
    };

    // turn 1 on slot 0: the prompt, then a reply decoded through the pages
    let l0 = model.forward_prefill(0, &prompt).expect("turn 1");
    let mut seq = prompt.clone();
    let mut tok = argmax(&l0);
    for i in 0..24 {
        seq.push(tok);
        let (step, _) = model
            .forward_mixed_sampled(
                &[(0usize, tok, (PROMPT_LEN + i) as u32)],
                usize::MAX,
                &[paddock_engine::generator::RowSample::Device(
                    paddock_engine::sampler::DevicePlan::Greedy,
                )],
                &[],
            )
            .expect("turn 1 decode");
        tok = step.ids[0];
    }
    assert!(
        warm(&mut model, 0, seq.len()),
        "decode ticks must extend the coverage"
    );
    model.release_inactive_slots(&[false, false]);

    // turn 2 on slot 0: the conversation so far + a new message
    let mut turn2 = seq.clone();
    turn2.extend((0..40).map(|i| 3000 + i as u32));
    model.forward_prefill(0, &turn2).expect("turn 2");
    let reused = model.take_prefill_reused(0);
    assert!(
        reused >= PROMPT_LEN - 16,
        "turn 2 must resume from the cache (at {reused})"
    );
    assert!(
        warm(&mut model, 0, turn2.len()),
        "a same-slot resume at {reused} left the drafter cold"
    );

    // the turn again, now from turn 2's own checkpoint and with its writer
    // idle: first into the OTHER slot, whose pages never held it, then into
    // slot 0 - both adopt the same pages, drafter rows included, walk the
    // same tail, and so draft warm and draft the same block
    model.release_inactive_slots(&[false, false]);
    let t1 = argmax(
        &model
            .forward_prefill(1, &turn2)
            .expect("turn 2, other slot"),
    );
    let other = model.take_prefill_reused(1);
    assert!(other > reused, "turn 2's own checkpoint (at {other})");
    assert!(
        warm(&mut model, 1, turn2.len()),
        "a cross-slot resume at {other} left the drafter cold"
    );
    let t0 = argmax(&model.forward_prefill(0, &turn2).expect("turn 2 again"));
    assert_eq!(model.take_prefill_reused(0), other);
    assert_eq!(t0, t1);
    let d = model
        .spec_draft_batch(&[(0usize, t0), (1usize, t1)], depth)
        .expect("draft")
        .expect("engaged");
    assert_eq!(d[0].len(), depth);
    assert_eq!(d[0], d[1], "the adopted pages must draft like their writer");

    // a turn on slot 0 that parts from its tokens inside the prompt keeps the
    // agreeing span: the resume lands below the part and stays warm
    model.release_inactive_slots(&[false, false]);
    let mut parted = prompt[..680].to_vec();
    parted.extend((0..30).map(|i| 5000 + i as u32));
    model.forward_prefill(0, &parted).expect("parted turn");
    let at = model.take_prefill_reused(0);
    assert!(
        at > 0 && at <= 680,
        "parted turn resumes below the part (at {at})"
    );
    assert!(
        warm(&mut model, 0, parted.len()),
        "the agreeing span must stay warm"
    );
    eprintln!(
        "coverage: same-slot resume at {reused} warm, cross-slot at {other} warm \
         (drafts {:?}), parted resume at {at} warm",
        d[0]
    );
}

const DSPARK_ENV: &str = "NEMOTRON_DSPARK_DIR";
const DSPARK_DIR: &str = "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark";

/// DSpark rides the same seat: the causal sliding-window block with sinks,
/// then the Markov head's left-to-right walk. Drafts are deterministic and
/// exactly the depth asked for (a causal block drafts no more than it
/// verifies), and the spec loop commits a valid stream. Acceptance is
/// reported, not asserted - it is the checkpoint's business, and a wrong
/// sink, rope, window or Markov wiring shows as a collapse there first.
#[test]
fn dspark_drafts_and_rounds() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_paged_kv() || !exec.has_spec_verify_mamba() || !exec.has_argmax_rows() {
        common::missing("pack lacks the nemotron dflash kernel set");
        return;
    }
    let Some(dir) = common::model_dir(CKPT_ENV, &[CKPT_DIR]) else {
        return;
    };
    let ds_dir = std::env::var(DSPARK_ENV).unwrap_or_else(|_| DSPARK_DIR.into());
    if !std::path::Path::new(&ds_dir)
        .join("model.safetensors")
        .exists()
    {
        eprintln!("skip: no dspark checkpoint at {ds_dir}");
        return;
    }
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let mut model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    model
        .attach_dflash(std::path::Path::new(&ds_dir))
        .expect("attach dspark");
    assert!(model.spec_capable());
    assert_eq!(model.spec_block_width(), Some(8), "the trained block");
    let depth = model.spec_fixed_draft_depth().expect("a fixed depth");
    assert_eq!(model.batch_enable_probe(2).expect("enable"), 2);
    let l0 = model.forward_prefill(0, &prompt).expect("prefill");
    let b0 = argmax(&l0);

    let d1 = model
        .spec_draft_batch(&[(0usize, b0)], depth)
        .expect("draft")
        .expect("engaged");
    let d2 = model
        .spec_draft_batch(&[(0usize, b0)], depth)
        .expect("draft 2")
        .expect("engaged");
    assert_eq!(d1, d2, "repeat drafts diverged");
    assert_eq!(
        d1[0].len(),
        depth,
        "a causal block drafts exactly its depth"
    );
    let short = model
        .spec_draft_batch(&[(0usize, b0)], 2)
        .expect("draft 3")
        .expect("engaged");
    assert_eq!(
        short[0],
        d1[0][..2],
        "a shorter causal block is the longer one's prefix"
    );

    let (mut pending, mut pos) = (b0, PROMPT_LEN);
    let (mut rounds, mut committed, mut accepted) = (0usize, 0usize, 0usize);
    while committed < 64 {
        let drafts = model
            .spec_draft_batch(&[(0usize, pending)], depth)
            .expect("draft")
            .expect("engaged");
        let mut chunk = vec![pending];
        chunk.extend(&drafts[0]);
        let picks = model
            .forward_spec_batch(&[(0usize, pos, chunk.clone())])
            .expect("verify")
            .expect("engaged");
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        rounds += 1;
        accepted += a;
        committed += a + 1;
        pending = picks[a];
        pos += a + 1;
        assert!(
            model
                .spec_ensure_warm(0, &[], (pos - 1) as u32)
                .expect("warm"),
            "the round's commit must keep the drafter warm"
        );
    }
    eprintln!(
        "dspark depth {depth}: {committed} tokens in {rounds} rounds ({:.2} a round), {:.0}% of drafts",
        committed as f64 / rounds as f64,
        100.0 * accepted as f64 / (rounds * depth) as f64
    );
}
