//! Flash-Next chunked prefill (forward/chunked.rs): the scheduler's mixed
//! tick against the walks it replaces. Heavy: `QWEN38FN_GGUF=<first shard>`
//! names the file, as in gpu_qwen4exp_gguf.rs.
//!
//! On a unified-memory box set `QWEN38FN_MOE_DEVICE=1` (the lane that
//! serves there; the host-mapped expert lane pins ~50 GB and takes the
//! better part of an hour to load).

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn moe_lane() {
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b })
}

fn rel(x: &[f32], y: &[f32]) -> f32 {
    let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
    let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn load(
    exec: &std::sync::Arc<paddock_engine::gpu::GpuExecutor>,
    path: &std::path::Path,
) -> Qwen4ExpGpu {
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(exec, path, 4096, 3).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");
    m
}

/// Drive `slot`'s queued prompt to its end through mixed ticks of `budget`
/// rows while `rider` (slot, first token) decodes greedily in every tick.
/// Returns the prompt's last logits, the rider's tokens, and the tick count.
fn chunk_with_rider(
    m: &mut Qwen4ExpGpu,
    slot: usize,
    budget: usize,
    rider: Option<(usize, u32)>,
) -> (Vec<f32>, Vec<u32>, usize) {
    let mut toks = Vec::new();
    let mut cur = rider;
    let mut ticks = 0usize;
    loop {
        let decodes: Vec<(usize, u32, u32)> = cur
            .map(|(s, t)| vec![(s, t, m.slot_position(s) as u32)])
            .unwrap_or_default();
        let before = m.prefill_queue();
        let (logits, finished) = m.forward_mixed(&decodes, budget).expect("mixed tick");
        ticks += 1;
        let after = m.prefill_queue();
        assert_ne!(before, after, "a mixed tick made no prefill progress");
        if let Some((s, _)) = cur {
            let t = argmax(&logits) as u32;
            toks.push(t);
            cur = Some((s, t));
        }
        if let Some((_, l, _)) = finished.into_iter().find(|f| f.0 == slot) {
            return (l, toks, ticks);
        }
        assert!(ticks < 10_000, "prefill never finished");
    }
}

/// ~220 tokens of varied text, deliberately not page-aligned.
fn prompt(tok: &GgufTokenizer, text: &str, n: usize) -> Vec<u32> {
    let base = tok.encode(text).expect("enc");
    base.iter().copied().cycle().take(n).collect()
}

/// A prompt prefilled through the mixed tick - odd-sized spans, a decode row
/// riding every one - lands on the distribution the single-slot walk gives
/// it, and the rider decodes what it decodes alone. Cache OFF, so both sides
/// are cold walks and span boundaries are the budget's alone.
///
/// Not bit-exact, and asserting that would be wrong: a span boundary moves
/// the GDN chunked scan's grouping and the dense GEMMs see different row
/// counts (the prefix gate's class). Asserted: the argmax, a loose L2, the
/// greedy continuation's first token, and the rider's greedy stream.
#[test]
fn gguf_chunked_prefill_matches_single_slot() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let a = prompt(
        &tok,
        "The reference manual describes a distributed consensus protocol in which \
         every participant maintains a monotonically increasing term counter and \
         exchanges signed heartbeat messages over authenticated channels. ",
        221,
    );
    let r = tok
        .encode("Write a short story about a lighthouse keeper who collects clocks.")
        .expect("enc");
    const K: usize = 8;
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    let mut m = load(&exec, &path);
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
    assert!(m.supports_chunked_prefill(), "chunked prefill not served");

    // references: the prompt through the single-slot walk, the rider alone
    let reference = m.prefill_slot(0, &a).expect("single-slot prefill");
    let mut chain_ref = vec![argmax(&reference) as u32];
    for _ in 1..K {
        let l = m
            .decode_step_batch(&[(0, *chain_ref.last().unwrap())])
            .expect("decode");
        chain_ref.push(argmax(&l[0]) as u32);
    }
    let lr = m.prefill_slot(1, &r).expect("rider prefill");
    let mut rider_ref = vec![argmax(&lr) as u32];
    for _ in 0..64 {
        let l = m
            .decode_step_batch(&[(1, *rider_ref.last().unwrap())])
            .expect("decode");
        rider_ref.push(argmax(&l[0]) as u32);
    }

    // the same, chunked: the rider re-admitted fresh, the prompt queued
    let lr = m.prefill_slot(1, &r).expect("rider prefill again");
    assert_eq!(argmax(&lr) as u32, rider_ref[0]);
    m.prefill_begin(0, a.clone()).expect("prefill_begin");
    let (got, rider, ticks) = chunk_with_rider(&mut m, 0, 37, Some((1, rider_ref[0])));
    let rr = rel(&got, &reference);
    eprintln!(
        "CHUNKED: {} rows in {ticks} ticks of 37 with a rider; argmax {} vs {}; rel {rr:.2e}",
        a.len(),
        argmax(&got),
        argmax(&reference)
    );
    assert_eq!(
        argmax(&got),
        argmax(&reference),
        "chunked prefill moved the argmax"
    );
    assert!(rr < 3e-1, "chunked prefill diverged: rel {rr}");
    // the control that says what that distance IS: the same 37-row spans
    // with no rider - single-slot walks continuing the slot, the geometry a
    // checkpoint-cut prefill already has. If the mixed walk (decode rows
    // leading, the decode entries for them) leaked anything into the prompt,
    // the rider run would sit well outside this one.
    m.prefill_begin(2, a.clone())
        .expect("prefill_begin control");
    let (ctrl, _, cticks) = chunk_with_rider(&mut m, 2, 37, None);
    let rc = rel(&ctrl, &reference);
    let rx = rel(&got, &ctrl);
    eprintln!(
        "CONTROL: {cticks} riderless ticks of 37; rel vs single-slot {rc:.2e}, \
         rider run vs control {rx:.2e}"
    );
    assert_eq!(argmax(&ctrl), argmax(&reference));
    assert!(
        rr < 1.5 * rc + 5e-2,
        "the rider run ({rr}) sits outside the span geometry's own distance ({rc})"
    );
    // the carried state: greedy steps off the chunked slot
    let mut chain = vec![argmax(&got) as u32];
    for _ in 1..K {
        let l = m
            .decode_step_batch(&[(0, *chain.last().unwrap())])
            .expect("decode");
        chain.push(argmax(&l[0]) as u32);
    }
    let same = chain
        .iter()
        .zip(&chain_ref)
        .take_while(|(x, y)| x == y)
        .count();
    eprintln!("CHUNKED chain {chain:?} vs single-slot {chain_ref:?} ({same}/{K} agree)");
    assert_eq!(chain[0], chain_ref[0]);
    // the rider rode prefill-class walks: its stream is the decode tick's
    let n = rider.len();
    let same = rider
        .iter()
        .zip(&rider_ref[1..=n])
        .take_while(|(x, y)| x == y)
        .count();
    eprintln!("RIDER {same}/{n} tokens match its solo decode");
    assert!(
        same >= n.min(8),
        "the rider's greedy stream parted from its solo decode after {same} tokens"
    );
}

/// The cache paths the mixed tick owns (cache ON):
/// - an idle lone prompt walks alone and takes its cuts IN the walk - the
///   same walk `prefill_slot` runs. An exact re-send of such a prompt walks
///   cold by the cache's own rule (prefix.rs `resume`), so it must come back
///   BIT-IDENTICAL; a turn extending it resumes at its deepest cut;
/// - a prompt chunked beside a rider stops its spans at the cuts and files
///   a checkpoint at each, so a later prefill of it resumes at the deepest.
#[test]
fn gguf_chunked_prefill_files_its_checkpoints() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    moe_lane();
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let a = prompt(
        &tok,
        "The reference manual describes a distributed consensus protocol in which \
         every participant maintains a monotonically increasing term counter and \
         exchanges signed heartbeat messages over authenticated channels. ",
        203,
    );
    let b = prompt(
        &tok,
        "Tidal energy converters extract kinetic power from moving water; the \
         survey compares barrage, stream and lagoon designs across twelve sites. ",
        187,
    );
    let mut m = load(&exec, &path);

    // idle, lone, whole budget: one tick, the in-walk cuts
    m.prefill_begin(0, a.clone()).expect("prefill_begin a");
    let (la, _, ticks) = chunk_with_rider(&mut m, 0, 4096, None);
    assert_eq!(ticks, 1, "an idle lone prompt that fits is one walk");
    let warm = m.prefill_slot(1, &a).expect("re-send a");
    assert_eq!(m.take_prefill_reused(1), 0, "an exact re-send walks cold");
    let n_diff = warm.iter().zip(&la).filter(|(x, y)| x != y).count();
    eprintln!("IDLE: the re-send of the lone chunked prompt differs in {n_diff} logits");
    assert_eq!(n_diff, 0, "the idle mixed tick must be prefill_slot's walk");
    // a turn that extends it resumes at the in-walk checkpoint it filed
    let mut a2 = a.clone();
    a2.extend(
        tok.encode(" The committee then turned to quorum.")
            .expect("enc"),
    );
    let _ = m.prefill_slot(2, &a2).expect("extended turn");
    let b1 = (a.len() - 1) / 16 * 16;
    assert_eq!(
        m.take_prefill_reused(2),
        b1,
        "the extended turn must resume at the chunked prompt's deepest cut"
    );

    // beside a rider (slot 1 decoding a's continuation): spans stop at cuts
    m.prefill_begin(2, b.clone()).expect("prefill_begin b");
    let (lb, _, ticks) = chunk_with_rider(&mut m, 2, 41, Some((1, argmax(&warm) as u32)));
    let warm_b = m.prefill_slot(0, &b).expect("resume b");
    let b1 = (b.len() - 1) / 16 * 16;
    let reused = m.take_prefill_reused(0);
    let r = rel(&warm_b, &lb);
    eprintln!(
        "RIDER: b in {ticks} ticks; resume reused {reused} (deepest cut {b1}); \
         argmax {} vs {}; rel {r:.2e}",
        argmax(&warm_b),
        argmax(&lb)
    );
    assert_eq!(
        reused, b1,
        "the chunked prompt's cut checkpoints were not filed"
    );
    assert_eq!(argmax(&warm_b), argmax(&lb), "resume moved the argmax");
    // The history under both is identical (the resume restores the chunked
    // walk's own state at the cut), so the logit distance is the last
    // rows' walk CLASS alone: the resume walks them single-slot (f16
    // tensor-core attention, segmented GDN walk), the chunked tick walked
    // them as a run beside a rider (tiled attention, runs recurrence).
    // Measured 0.30 here and 0.20-0.26 between the same classes in
    // `gguf_chunked_prefill_matches_single_slot`, so the L2 bounds only a
    // catastrophe. What says the checkpoint is RIGHT is the carried state:
    // greedy continuations off the resumed slot and off the chunked slot.
    assert!(r < 5e-1, "resume diverged: rel {r}");
    const K: usize = 8;
    let greedy = |m: &mut Qwen4ExpGpu, slot: usize, first: u32| -> Vec<u32> {
        let mut ch = vec![first];
        for _ in 1..K {
            let l = m
                .decode_step_batch(&[(slot, *ch.last().unwrap())])
                .expect("decode");
            ch.push(argmax(&l[0]) as u32);
        }
        ch
    };
    let from_chunked = greedy(&mut m, 2, argmax(&lb) as u32);
    let from_resume = greedy(&mut m, 0, argmax(&warm_b) as u32);
    let same = from_chunked
        .iter()
        .zip(&from_resume)
        .take_while(|(x, y)| x == y)
        .count();
    eprintln!("RESUME chain {from_resume:?} vs chunked {from_chunked:?} ({same}/{K} agree)");
    assert!(
        same >= 4,
        "the resumed slot's continuation parted from the chunked slot's after {same} tokens"
    );
}
