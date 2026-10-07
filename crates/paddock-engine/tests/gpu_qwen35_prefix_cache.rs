//! Qwen3.5 hybrid prefix-cache gates. A hybrid model can only RESUME a cached
//! prefix at a DeltaNet state checkpoint (the recurrent state has no rollback),
//! so the cache pairs full-attn KV pages with state snapshots at the last two
//! page boundaries of every prefill.
//!
//! Gate 1 (bit-exact): re-prefilling the same prompt resumes from its own
//! checkpoint - the resumed chunk replays the cold run's final chunk with a
//! bit-exact restored state over byte-identical KV pages, so the logits must
//! match exactly.
//!
//! Gate 2 (multi-turn shape): a prompt sharing all but its trailing tokens
//! with a cached one (the re-rendered-history case) must reuse a checkpoint
//! and stay greedy-identical to the pinned-off (PADDOCK_NO_PREFIX_CACHE)
//! single-chunk path. The resume geometry differs from the cold run's, so the
//! DeltaNet chunked-scan grouping differs - greedy + loose L2 is the honest
//! bar (the S2/G2 gate lesson), on a clear-winner prompt.
//!
//! Gate 3 (the reply checkpoint, stage F): a turn that DECODES past a page
//! boundary leaves a checkpoint at the reply's last boundary, so the next
//! turn (prompt + reply + a new message) resumes there - past the prompt's
//! own cuts - and stays greedy-identical to the pinned-off path. Decode
//! advances the state one token at a time where the pinned prefill scans
//! the reply in chunks, so this is the multi-turn gate's class (greedy +
//! loose L2), not gate 1's.
//!
//! Gate 4 (checkpoint pages, issue #33): checkpoints are pages of the KV
//! pool, so a context that needs them takes them - the cache yields, the
//! request never fails for it.
//!
//! Heavy GPU test: PADDOCK_HEAVY_TESTS=1, --release, --test-threads=1.

mod common;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn setup() -> Option<(GpuQwen35, GgufTokenizer)> {
    if !common::heavy() {
        return None;
    }
    let path = common::model("QWEN35_GGUF", common::QWEN35_9B_Q8)?;
    let exec = common::gpu_arc()?;
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load 9B");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    Some((m, tok))
}

/// ~`n`-token prompt of varied real text (clear-winner continuations).
fn long_prompt(tok: &GgufTokenizer, n: usize) -> Vec<u32> {
    let base = tok
        .encode(
            "The reference manual describes a distributed consensus protocol in which \
             every participant maintains a monotonically increasing term counter and \
             exchanges signed heartbeat messages over authenticated channels. ",
        )
        .expect("enc");
    base.iter().copied().cycle().take(n).collect()
}

fn amax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn rel(x: &[f32], y: &[f32]) -> f32 {
    let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
    let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

#[test]
fn resume_from_own_checkpoint_is_bit_exact() {
    let Some((mut m, tok)) = setup() else { return };
    m.enable_batch(4).expect("enable_batch");

    let b = long_prompt(&tok, 203); // non-page-aligned deliberately
    let cold = m.forward_prefill_slot(0, &b).expect("cold prefill");
    assert_eq!(m.take_prefill_reused(0), 0, "cold run must not reuse");

    let warm = m.forward_prefill_slot(1, &b).expect("warm prefill");
    let reused = m.take_prefill_reused(1);
    let b1 = (b.len() - 1) / 16 * 16;
    eprintln!(
        "PREFIX CACHE: reused {reused} of {} tokens (checkpoint at {b1})",
        b.len()
    );
    assert_eq!(reused, b1, "must resume at the deepest checkpoint");
    // restored state is a bit-exact snapshot + KV pages are byte-identical +
    // the resumed chunk has the cold run's exact geometry => logits identical
    let n_diff = cold.iter().zip(&warm).filter(|(a, b)| a != b).count();
    assert_eq!(
        n_diff,
        0,
        "resume must be BIT-EXACT; {} of {} logits differ (rel {:.2e})",
        n_diff,
        cold.len(),
        rel(&warm, &cold)
    );
}

/// Loose L2 bound, per the established gate policy (gpu_gpt_oss_parity's
/// F16_KV_REL): a resume changes the DeltaNet chunked-scan grouping, so
/// logits legitimately drift ~1e-2 while the greedy token holds; 3e-1 still
/// catches O(1) breakage (wrong window / state / slot mapping).
const LOOSE_REL: f32 = 3e-1;

#[test]
fn multi_turn_shape_reuses_and_matches_pinned_reference() {
    let Some((mut m, tok)) = setup() else { return };

    let a = long_prompt(&tok, 199);
    // the multi-turn shape: everything but the trailing generation header is
    // re-sent verbatim, then diverges (new turn) - three independent tails
    // make the greedy gate three independent trials
    let tails = [
        " The committee voted to adopt the second proposal because",
        " In summary, the protocol guarantees safety by requiring",
        " The final tally showed forty-two delegates voting in favor of",
    ];
    let bs: Vec<Vec<u32>> = tails
        .iter()
        .map(|t| {
            let mut b: Vec<u32> = a[..a.len() - 6].to_vec();
            b.extend(tok.encode(t).expect("enc"));
            b
        })
        .collect();

    // pinned references: the pre-cache single-chunk path
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    m.enable_batch(4).expect("enable_batch pinned");
    let refs: Vec<Vec<f32>> = bs
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let r = m.forward_prefill_slot(i, b).expect("pinned prefill");
            assert_eq!(m.take_prefill_reused(i), 0, "pinned path must not reuse");
            r
        })
        .collect();
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };

    // cached run: A fills the cache (checkpoints at its last two page
    // boundaries), each B resumes from the deepest one under the shared prefix
    m.enable_batch(4).expect("enable_batch cached");
    let _ = m.forward_prefill_slot(0, &a).expect("prefill A");
    assert_eq!(m.take_prefill_reused(0), 0);
    for (i, (b, reference)) in bs.iter().zip(&refs).enumerate() {
        let got = m.forward_prefill_slot(1 + i, b).expect("prefill B");
        let reused = m.take_prefill_reused(1 + i);
        let r = rel(&got, reference);
        eprintln!(
            "PREFIX CACHE multi-turn[{i}]: reused {reused} of {} shared; rel {:.2e}; \
             greedy {} vs {}",
            a.len() - 6,
            r,
            amax(&got),
            amax(reference)
        );
        assert!(reused >= 16, "expected checkpoint reuse, got {reused}");
        assert!(
            reused <= a.len() - 6,
            "reuse cannot exceed the shared prefix"
        );
        assert_eq!(
            amax(&got),
            amax(reference),
            "greedy token flipped (tail {i})"
        );
        assert!(r < LOOSE_REL, "diverged: rel {r} (tail {i})");
    }
}

/// Greedy decode `n` tokens on `slot` through the sampled batch tick (the
/// serving decode path), returning the reply.
fn decode_greedy(m: &mut GpuQwen35, slot: usize, first: u32, pos0: usize, n: usize) -> Vec<u32> {
    use paddock_engine::generator::RowSample;
    use paddock_engine::sampler::DevicePlan;
    let mut reply = vec![first];
    let mut tok = first;
    for j in 0..n {
        let mut tokens = vec![0u32; slot + 1];
        let mut positions = vec![0u32; slot + 1];
        tokens[slot] = tok;
        positions[slot] = (pos0 + j) as u32;
        let mut plans = vec![RowSample::Hole; slot + 1];
        plans[slot] = RowSample::Device(DevicePlan::Greedy);
        let step = m
            .forward_batch_sampled(&tokens, &positions, &plans)
            .expect("decode tick");
        tok = step.ids[slot];
        reply.push(tok);
    }
    reply.pop();
    reply
}

#[test]
fn reply_checkpoint_resumes_the_next_turn_at_the_reply() {
    let Some((mut m, tok)) = setup() else { return };
    let a = long_prompt(&tok, 199);
    let tail = tok
        .encode(" The committee then turned to the question of quorum, and")
        .expect("enc");

    // turn 1 on the cached engine: prefill A, decode a reply that crosses
    // at least two page boundaries (the tracked state snapshots at each)
    m.enable_batch(4).expect("enable_batch cached");
    let logits = m.forward_prefill_slot(0, &a).expect("prefill A");
    let reply = decode_greedy(&mut m, 0, amax(&logits) as u32, a.len(), 45);
    let mut b: Vec<u32> = a.clone();
    b.extend_from_slice(&reply);
    b.extend_from_slice(&tail);
    let reply_cut = (a.len() + reply.len()) / 16 * 16;
    assert!(
        reply_cut > (a.len() - 1) / 16 * 16,
        "the reply must cross a boundary"
    );

    // turn 2: resumes at the reply's last boundary, past the prompt's cuts
    m.release_inactive_slots(&[false, false, false, false]);
    let got = m.forward_prefill_slot(1, &b).expect("prefill B");
    let reused = m.take_prefill_reused(1);
    eprintln!(
        "REPLY CKPT: reused {reused} of {} (prompt {} + reply {} + tail {}); reply cut {reply_cut}",
        b.len(),
        a.len(),
        reply.len(),
        tail.len()
    );
    assert_eq!(
        reused, reply_cut,
        "must resume at the reply's last page boundary"
    );

    // the pinned-off reference for the same turn-2 prompt
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    m.enable_batch(4).expect("enable_batch pinned");
    let reference = m.forward_prefill_slot(2, &b).expect("pinned prefill");
    assert_eq!(m.take_prefill_reused(2), 0, "pinned path must not reuse");
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };

    let r = rel(&got, &reference);
    eprintln!(
        "REPLY CKPT: rel {r:.2e}; greedy {} vs {}",
        amax(&got),
        amax(&reference)
    );
    assert_eq!(amax(&got), amax(&reference), "greedy token flipped");
    assert!(r < LOOSE_REL, "diverged: rel {r}");
}

/// Gate 4 (issue #33): checkpoints live in the KV pool's own pages and hand
/// them back to a live context. The pool is pinned to one slot's window plus
/// a quarter - or, where one checkpoint outweighs that quarter (Bonsai 27B:
/// ~300 pages), to A's pages plus one checkpoint: prompt A's checkpoints take
/// pages, then a prompt B that needs nearly the whole window gets them - by
/// evicting A's checkpoints, never by failing - and B is still what the
/// cache-off path computes. A has lost its resume point to it. Greedy + loose
/// L2 like gate 2: the cached path chunks a cold prefill at its checkpoint
/// cuts, the pinned-off path does not.
#[test]
fn checkpoint_pages_yield_to_a_growing_context() {
    let Some((mut m, tok)) = setup() else { return };
    let b = long_prompt(&tok, 3900); // 244 of the window's 256 pages
    let base = tok
        .encode(
            "Harbour records from the northern coast list every vessel by tonnage, \
             port of origin and the cargo declared at the customs house on arrival. ",
        )
        .expect("enc");
    let a: Vec<u32> = base.iter().copied().cycle().take(400).collect();

    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    m.enable_batch(2).expect("enable_batch pinned");
    let reference = m.forward_prefill_slot(0, &b).expect("pinned prefill");
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };

    // one checkpoint's pages are the model's: read them off an unpinned pool
    m.enable_batch(2).expect("enable_batch");
    let ppc = m.ckpt_pages_probe();
    assert!(ppc > 0, "prefix cache on, yet no checkpoint pages");
    let pool = 320usize.max(a.len().div_ceil(16) + ppc + 8);
    unsafe { std::env::set_var("PADDOCK_KV_POOL_BLOCKS", pool.to_string()) };
    m.enable_batch(2).expect("enable_batch with a tight pool");
    unsafe { std::env::remove_var("PADDOCK_KV_POOL_BLOCKS") };

    // A leaves a checkpoint behind: its re-prefill resumes
    let _ = m.forward_prefill_slot(0, &a).expect("prefill A");
    m.release_inactive_slots(&[false, false]);
    let _ = m.forward_prefill_slot(1, &a).expect("re-prefill A");
    let a_reused = m.take_prefill_reused(1);
    assert!(
        a_reused >= 16,
        "A must have checkpointed, reused {a_reused}"
    );
    m.release_inactive_slots(&[false, false]);

    // B's context needs the pages A's checkpoints hold
    let got = m
        .forward_prefill_slot(0, &b)
        .expect("B must take the checkpoint pages, not run the pool dry");
    assert_eq!(m.take_prefill_reused(0), 0, "B shares nothing with A");
    let r = rel(&got, &reference);
    eprintln!(
        "CKPT PAGES: A resumed at {a_reused}; B cold under pressure rel {r:.2e}, greedy {} vs {}",
        amax(&got),
        amax(&reference)
    );
    assert_eq!(amax(&got), amax(&reference), "greedy token flipped");
    assert!(r < LOOSE_REL, "diverged: rel {r}");

    // ...and A paid for it
    m.release_inactive_slots(&[false, false]);
    let _ = m.forward_prefill_slot(1, &a).expect("prefill A again");
    let after = m.take_prefill_reused(1);
    eprintln!("CKPT PAGES: A after B resumed at {after}");
    assert!(
        after < a_reused,
        "A's checkpoint should have gone to B's context"
    );
}
