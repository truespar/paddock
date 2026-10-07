//! Flash-Next speculation rounds (heavy: the real GGUF + the MTP head).
//!
//! `QWEN38FN_GGUF` = shard 1 of the model, `QWEN38FN_MTP` = the draft head
//! (e.g. `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`); skipped without either.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b })
}

/// The HOST-SAMPLED verify round (`forward_spec_verify` + `spec_commit`) -
/// the round every tool-carrying request speculates through, since a
/// constraint keeps a slot out of the greedy and device rounds - against the
/// greedy device round (`forward_spec_batch`). Two slots walk one prompt cold,
/// so they start bit-identical; slot 0 then speculates through the greedy
/// round and slot 1 through the sampled one with the host taking each row's
/// argmax. They must draft the same, pick the same, commit the same, and
/// leave the same carried state (the next decode's logits, bit for bit). The
/// prompt is past the 2051-token window, so the verify walks attend through
/// QSA.
#[test]
fn gguf_sampled_verify_commits_what_the_greedy_round_does() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(mtp) = std::env::var_os("QWEN38FN_MTP").map(std::path::PathBuf::from) else {
        common::missing("QWEN38FN_MTP");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let base = tok
        .encode(
            "The dispatcher's log lists each courier, the parcels they carried, the depot \
             they left from and the time they signed the handover sheet. ",
        )
        .expect("encode");
    let p: Vec<u32> = base.iter().copied().cycle().take(2200).collect();

    // the prefix cache off: both slots walk the prompt cold, in one walk each
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 2).expect("load gguf");
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(3 << 30))
        .expect("cache");
    m.attach_mtp(&mtp).expect("attach mtp");
    let l0 = m.prefill_slot(0, &p).expect("prefill 0");
    let l1 = m.prefill_slot(1, &p).expect("prefill 1");
    assert!(l0 == l1, "the two slots do not start bit-identical");

    let depth = 3usize;
    let mut pend = [argmax(&l0) as u32; 2];
    let (mut out0, mut out1) = (Vec::new(), Vec::new());
    let mut accepted = 0usize;
    for round in 0..16 {
        let draft = |m: &mut Qwen4ExpGpu, slot: usize, pending: u32| -> Vec<u32> {
            let d = m
                .spec_draft_batch(&[(slot, pending)], depth)
                .expect("draft")
                .map(|mut d| d.remove(0))
                .unwrap_or_default();
            let mut chunk = vec![pending];
            chunk.extend(d.iter().copied().take(depth));
            chunk
        };
        // slot 0: the greedy device round
        let pos0 = m.slot_position(0);
        let c0 = draft(&mut m, 0, pend[0]);
        let picks = m
            .forward_spec_batch(&[(0, pos0, c0.clone())])
            .expect("greedy round")
            .expect("the greedy round declined");
        let mut a = 0usize;
        while a + 1 < c0.len() && c0[a + 1] == picks[a] {
            a += 1;
        }
        out0.extend_from_slice(&picks[..=a]);
        pend[0] = picks[a];
        accepted += a;

        // slot 1: the host-sampled round, the host taking each row's argmax
        let pos1 = m.slot_position(1);
        let c1 = draft(&mut m, 1, pend[1]);
        assert_eq!(c1, c0, "round {round}: the head drafted differently");
        let rows = m
            .forward_spec_verify(&[(1, pos1, c1.clone())])
            .expect("sampled round")
            .expect("the sampled round declined");
        let vocab = rows.len() / c1.len();
        let hp: Vec<u32> = rows.chunks(vocab).map(|r| argmax(r) as u32).collect();
        assert_eq!(
            hp, picks,
            "round {round}: the sampled round's rows pick differently"
        );
        let mut b = 0usize;
        while b + 1 < c1.len() && c1[b + 1] == hp[b] {
            b += 1;
        }
        m.spec_commit(&[(b + 1) as u32]).expect("commit");
        out1.extend_from_slice(&hp[..=b]);
        pend[1] = hp[b];
        assert_eq!(
            m.slot_position(1),
            m.slot_position(0),
            "round {round}: the two routes committed different rows"
        );
    }
    assert_eq!(out1, out0, "the two routes emitted different streams");
    assert!(
        accepted > 0,
        "no draft was ever accepted: nothing speculated"
    );
    // the carried state each route left, read through the next decode
    let n0 = m.decode_step_batch(&[(0, pend[0])]).expect("decode 0");
    let n1 = m.decode_step_batch(&[(1, pend[1])]).expect("decode 1");
    assert!(
        n0[0] == n1[0],
        "the two routes left different carried state"
    );
    eprintln!(
        "SAMPLED ROUND: 16 rounds at depth {depth} from position {}: {} tokens, {accepted} drafts \
         accepted; picks, commits and the next decode bit-identical to the greedy round",
        p.len(),
        out0.len()
    );
}

/// The reply checkpoint of a speculating reply. A reply decoded through
/// greedy verify rounds must leave the checkpoint the decode ticks leave for
/// the same tokens - including at a page a round passed MID-round, where the
/// state at the boundary is rebuilt from the round's rollback planes - so
/// the next turn, run after the slot is released as the scheduler does,
/// resumes at the same point with bit-identical logits. The rounds stop
/// right after a round passes a page mid-round, so that is the rolling
/// checkpoint the next turn reads. Two loads: each files its own pages.
#[test]
fn gguf_spec_reply_checkpoint_matches_the_decode_ticks() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(mtp) = std::env::var_os("QWEN38FN_MTP").map(std::path::PathBuf::from) else {
        common::missing("QWEN38FN_MTP");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let base = tok
        .encode(
            "The dispatcher's log lists each courier, the parcels they carried, the depot \
             they left from and the time they signed the handover sheet. ",
        )
        .expect("encode");
    let p: Vec<u32> = base.iter().copied().cycle().take(700).collect();
    let msg = tok
        .encode(" Now add the next courier to the log.")
        .expect("encode");
    let load = |with_mtp: bool| -> Qwen4ExpGpu {
        common::qwen4exp_exact_kv();
        let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 4096, 1).expect("load gguf");
        let headroom = exec.vram_headroom().unwrap_or(0);
        m.enable_moe_cache(headroom.saturating_sub(3 << 30))
            .expect("cache");
        if with_mtp {
            m.attach_mtp(&mtp).expect("attach mtp");
        }
        m
    };
    // the next turn: the history (prompt + the tokens the reply fed) and a
    // new message, in the released slot
    let turn2 = |m: &mut Qwen4ExpGpu, reply: &[u32]| -> (usize, Vec<f32>) {
        m.release_inactive_slots(&[false]);
        let mut t2 = p.clone();
        t2.extend_from_slice(reply);
        t2.extend_from_slice(&msg);
        let l = m.prefill_slot(0, &t2).expect("turn 2");
        (m.take_prefill_reused(0), l)
    };

    // load 1: speculate until a round has passed a page mid-round (twice, the
    // second one the last round)
    let (reply, pend, at_spec, l_spec) = {
        let mut m = load(true);
        let l = m.prefill_slot(0, &p).expect("prefill");
        let mut pend = argmax(&l) as u32;
        let mut reply: Vec<u32> = Vec::new();
        let (depth, mut mid, mut accepted) = (3usize, 0usize, 0usize);
        for _ in 0..96 {
            let pos0 = m.slot_position(0);
            let d = m
                .spec_draft_batch(&[(0, pend)], depth)
                .expect("draft")
                .map(|mut d| d.remove(0))
                .unwrap_or_default();
            let mut chunk = vec![pend];
            chunk.extend(d.iter().copied().take(depth));
            let picks = m
                .forward_spec_batch(&[(0, pos0, chunk.clone())])
                .expect("round")
                .expect("the round declined");
            let mut a = 0usize;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
                a += 1;
            }
            reply.extend_from_slice(&chunk[..=a]);
            pend = picks[a];
            accepted += a;
            let pos = m.slot_position(0);
            if pos / 16 > pos0 / 16 && pos % 16 != 0 {
                mid += 1;
                if mid >= 2 {
                    break;
                }
            }
        }
        assert!(
            mid >= 2,
            "no round passed a page mid-round ({accepted} drafts accepted)"
        );
        let (at, l2) = turn2(&mut m, &reply);
        (reply, pend, at, l2)
    };

    // load 2: the same reply through decode ticks
    let (at_dec, l_dec) = {
        let mut m = load(false);
        let l = m.prefill_slot(0, &p).expect("prefill");
        let mut next = argmax(&l) as u32;
        for (i, &t) in reply.iter().enumerate() {
            assert_eq!(next, t, "decode tick {i} picks differently from the rounds");
            let d = m.decode_step_batch(&[(0, t)]).expect("decode");
            next = argmax(&d[0]) as u32;
        }
        assert_eq!(next, pend, "the decode ticks end on another pending token");
        turn2(&mut m, &reply)
    };
    let end = p.len() + reply.len();
    eprintln!(
        "SPEC REPLY CKPT: reply of {} tokens to {end}; next turn resumed at {at_spec} (spec) / \
         {at_dec} (decode ticks); logits bit-identical: {}",
        reply.len(),
        l_spec == l_dec
    );
    assert_eq!(
        at_spec,
        end / 16 * 16,
        "the speculating reply's checkpoint is not at its last page"
    );
    assert_eq!(
        at_spec, at_dec,
        "the two replies checkpoint at different points"
    );
    assert!(
        l_spec == l_dec,
        "a resume from the in-round checkpoint is not the decode ticks' resume"
    );
}
