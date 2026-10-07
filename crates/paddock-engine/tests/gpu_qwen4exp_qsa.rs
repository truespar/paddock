//! Flash-Next QSA, rung R1 (the Flash-Next QSA design note):
//! the indexer's compressed key cache on the real checkpoint. Heavy:
//! `QWEN38FN_GGUF=<first shard>` names the file, as in gpu_qwen4exp_gguf.rs;
//! on a unified-memory box set `QWEN38FN_MOE_DEVICE=1`.
//!
//! The kernels are op-gated against the reference (gpu_qwen4exp_ops.rs);
//! this gates the WIRING: every walk shape the lane serves must leave the
//! same block keys - a prompt walked whole, the same tokens decoded one at a
//! time (the block a decode row closes reads its first keys off the ring
//! earlier walks filed), odd-sized chunked spans (blocks straddle spans), and
//! a prefix-cache resume (the adopted pages must hold the keys they were
//! given, bit for bit). One test, two loads in sequence: a second resident
//! copy of this model does not fit the boxes it runs on.

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

/// Drive `slot`'s queued prompt to its end through riderless mixed ticks of
/// `budget` rows (spans that end mid-block).
fn chunk(m: &mut Qwen4ExpGpu, slot: usize, budget: usize) -> Vec<f32> {
    for _ in 0..10_000 {
        let (_, finished) = m.forward_mixed(&[], budget).expect("mixed tick");
        if let Some((_, l, _)) = finished.into_iter().find(|f| f.0 == slot) {
            return l;
        }
    }
    panic!("prefill never finished");
}

const HD: usize = 128;

/// The attention layers this checkpoint has (every 4th of 48).
fn attn_layers() -> Vec<usize> {
    (3..48).step_by(4).collect()
}

/// Per-block rel L2 of `got` against `want` ([blocks, HD] each).
fn block_rels(got: &[f32], want: &[f32]) -> Vec<f32> {
    got.chunks(HD)
        .zip(want.chunks(HD))
        .map(|(g, w)| rel(g, w))
        .collect()
}

/// One resident copy of this model at a time: the heavy tests in this binary
/// take turns (a parallel test runner would load two).
static HEAVY: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn gguf_qsa_index_cache_agrees_across_walks() {
    if !common::heavy() {
        return;
    }
    let _one = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    if !exec.has_qsa_indexer() {
        common::missing("pack has no QSA indexer kernels (rebuild packs/cuda)");
        return;
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let base = tok
        .encode(
            "The reference manual describes a distributed consensus protocol in which \
             every participant maintains a monotonically increasing term counter and \
             exchanges signed heartbeat messages over authenticated channels. ",
        )
        .expect("enc");
    // 221 tokens: 55 blocks close in the prompt, token 220 is the tail
    let a: Vec<u32> = base.iter().copied().cycle().take(221).collect();
    let layers = attn_layers();

    // ---- load A, prefix cache off: whole / decoded / chunked walks
    let full = {
        unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
        let mut m = load(&exec, &path);
        unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
        // slot 0: the prompt, then three decoded tokens - the third closes
        // block 55 from 220 (the prompt walk's ring entry) and 221-222 (the
        // decode walks' ones)
        let l = m.prefill_slot(0, &a).expect("prefill");
        let mut toks = vec![argmax(&l) as u32];
        for _ in 0..2 {
            let l = m
                .decode_step_batch(&[(0, *toks.last().unwrap())])
                .expect("decode");
            toks.push(argmax(&l[0]) as u32);
        }
        m.decode_step_batch(&[(0, toks[2])]).expect("decode");
        assert_eq!(m.slot_position(0), 224);
        let mut full = a.clone();
        full.extend(&toks);
        // slot 1: the same 224 tokens in one walk; slot 2: in 37-row spans
        m.prefill_slot(1, &full).expect("prefill full");
        m.prefill_begin(2, full.clone()).expect("prefill_begin");
        chunk(&mut m, 2, 37);
        for &li in &layers {
            let k = |s| {
                m.qsa_index_keys(li, s, 0, 56)
                    .expect("read")
                    .expect("attention layer has an index cache")
            };
            let (k0, k1, k2) = (k(0), k(1), k(2));
            assert!(
                k0.iter().chain(&k1).chain(&k2).all(|v| v.is_finite()),
                "layer {li}: a written block key is not finite"
            );
            let (r0, r2) = (block_rels(&k0, &k1), block_rels(&k2, &k1));
            let worst = |r: &[f32]| r.iter().copied().fold(0.0f32, f32::max);
            // What a WRONG key looks like here: the distance between one
            // block's key and the next one's in the same walk - a pooling
            // source error (a stale ring entry, a neighbour's raw key, a span
            // boundary read from the wrong side) lands at that scale.
            let mut nb: Vec<f32> = k1
                .chunks(HD)
                .zip(k1.chunks(HD).skip(1))
                .map(|(x, y)| rel(y, x))
                .collect();
            nb.sort_by(f32::total_cmp);
            let median_nb = nb[nb.len() / 2];
            eprintln!(
                "layer {li:2}: decoded-vs-whole worst {:.2e} (block 55 {:.2e}), \
                 chunked-vs-whole worst {:.2e}; neighbouring blocks' median {median_nb:.2e}",
                worst(&r0),
                r0[55],
                worst(&r2)
            );
            // The walk geometry moves the activations under the keys: ~1e-2
            // at the first attention layer, then more with every layer (37-row
            // spans move this lane's logits ~2e-1 - the chunked gate's
            // control), until a span boundary's own drift is as big as a wrong
            // key's. The pooling is the same code in every layer, so the first
            // layer is where it is judged: every block - the one decode closed
            // off the ring, the ones straddling a span boundary - within a
            // tenth of the distance to a wrong key. Deeper layers are printed
            // for the record.
            if li == layers[0] {
                assert!(
                    worst(&r0).max(worst(&r2)) < 0.1 * median_nb,
                    "layer {li}: walk shapes disagree on block keys before the geometry could"
                );
            }
        }
        full
    };

    // ---- load B, prefix cache on: a resume hands the keys back
    let mut m = load(&exec, &path);
    m.prefill_slot(0, &a).expect("prefill");
    let resumed = {
        m.prefill_slot(1, &full).expect("prefill resumed");
        m.take_prefill_reused(1)
    };
    eprintln!("prefix resume at {resumed}");
    assert!(
        resumed >= 16,
        "the full prompt did not resume off the cached one"
    );
    let copied = resumed / 4;
    for &li in &layers {
        let k0 = m
            .qsa_index_keys(li, 0, 0, copied)
            .expect("read")
            .expect("cache");
        let k1 = m
            .qsa_index_keys(li, 1, 0, copied)
            .expect("read")
            .expect("cache");
        assert!(
            k0 == k1,
            "layer {li}: the resumed slot's first {copied} block keys are not the published ones"
        );
    }
}

/// R4a: a prompt longer than the walk budget walks in pieces split at
/// absolute multiples of it. Against the whole walk that is a geometry
/// change (the chunked gate's class: the argmax and the first greedy token
/// hold, the logits move); between the two bounded paths - the single-slot
/// prefill and riderless chunked ticks - it is the SAME geometry, so they
/// must agree to the last digits.
#[test]
fn gguf_bounded_walks_match_the_whole_walk() {
    if !common::heavy() {
        return;
    }
    let _one = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
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
            "Harbour logs record every vessel's arrival, its cargo manifest, the pilot on \
             duty and the berth assigned; discrepancies are flagged for the customs office. ",
        )
        .expect("enc");
    let p: Vec<u32> = base.iter().copied().cycle().take(1500).collect();
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    let mut m = load(&exec, &path);
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
    let chain = |m: &mut Qwen4ExpGpu, slot: usize, first: &[f32]| -> Vec<u32> {
        let mut t = vec![argmax(first) as u32];
        for _ in 1..8 {
            let l = m
                .decode_step_batch(&[(slot, *t.last().unwrap())])
                .expect("decode");
            t.push(argmax(&l[0]) as u32);
        }
        t
    };
    // slot 0: one walk of 1500 rows
    let whole = m.prefill_slot(0, &p).expect("whole");
    let chain_whole = chain(&mut m, 0, &whole);
    // slot 1: the same prompt in 256-row walks; slot 2: riderless chunked ticks
    m.cap_walk_rows(256).expect("cap");
    let bounded = m.prefill_slot(1, &p).expect("bounded");
    let chain_bounded = chain(&mut m, 1, &bounded);
    m.prefill_begin(2, p.clone()).expect("prefill_begin");
    let chunked = chunk(&mut m, 2, 4096);
    let (rb, rc) = (rel(&bounded, &whole), rel(&chunked, &bounded));
    let same = chain_bounded
        .iter()
        .zip(&chain_whole)
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "BOUNDED: 1500 rows in 256-row walks vs one walk: argmax {} vs {}, rel {rb:.2e}, \
         chain {same}/8; chunked vs bounded rel {rc:.2e} (bit-exact: {})",
        argmax(&bounded),
        argmax(&whole),
        chunked == bounded
    );
    assert_eq!(
        argmax(&bounded),
        argmax(&whole),
        "bounded walks moved the argmax"
    );
    assert!(rb < 3e-1, "bounded walks diverged: rel {rb}");
    assert_eq!(chain_bounded[0], chain_whole[0]);
    assert!(
        rc < 1e-4,
        "the two bounded paths walked different geometries: rel {rc}"
    );
    // the index keys the bounded walk filed agree with the whole walk's at
    // the first attention layer (R1's judge), every block of the prompt
    let k0 = m
        .qsa_index_keys(3, 0, 0, 375)
        .expect("read")
        .expect("cache");
    let k1 = m
        .qsa_index_keys(3, 1, 0, 375)
        .expect("read")
        .expect("cache");
    let worst = block_rels(&k1, &k0).into_iter().fold(0.0f32, f32::max);
    eprintln!("BOUNDED: layer 3 index keys vs the whole walk: worst block rel {worst:.2e}");
    assert!(
        worst < 7e-2,
        "bounded walks filed different index keys: {worst}"
    );
}

/// R4b: inside the window (every row sees <= 2051 tokens) QSA selects every
/// block, so the sparse path IS dense attention - walked by other kernels
/// (the SIMT gather against the tensor-core prefill / decode kernels). The
/// gate is the first attention layer, where both paths read the same
/// inputs: a wrong selection (a block dropped or doubled) moves its output
/// O(1e-1), the kernels' own distance is ~1e-5 (v4 prefill) to ~5e-3 (the
/// f16-accumulating WMMA tile). Downstream nothing is bounded: on this
/// prompt (one sentence cycled to 1500 tokens) ANY perturbation there -
/// sparse vs dense, or two dense prefill kernels - saturates to a 20-60%
/// hidden-state spread by the middle layers (measured per attention layer,
/// 2026-09-28: the two dense kernels differ from each other exactly as much
/// as either differs from sparse), and the final argmax is a near tie
/// (`<|im_end|>` vs " A", 0.1-0.35 logits on the dense paths), so the
/// logits, argmax and greedy chain are printed, not asserted - end-to-end
/// quality is the llama.cpp parity legs'. Past the window the two are
/// different models by design; that distance is printed too.
#[test]
fn gguf_qsa_sparse_matches_dense_inside_the_window() {
    if !common::heavy() {
        return;
    }
    let _one = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    use paddock_engine::gpu_model::qwen4exp::QsaMode;
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    let base = tok
        .encode(
            "A lighthouse keeper logs the passing ships each night: the hour, the \
             heading, the flag, the weather and anything unusual about the lights. ",
        )
        .expect("enc");
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    let mut m = load(&exec, &path);
    unsafe { std::env::remove_var("PADDOCK_NO_PREFIX_CACHE") };
    let run = |m: &mut Qwen4ExpGpu, slot: usize, p: &[u32]| -> (Vec<f32>, Vec<u32>) {
        let l = m.prefill_slot(slot, p).expect("prefill");
        let mut t = vec![argmax(&l) as u32];
        for _ in 1..8 {
            let d = m
                .decode_step_batch(&[(slot, *t.last().unwrap())])
                .expect("decode");
            t.push(argmax(&d[0]) as u32);
        }
        (l, t)
    };
    // inside the window, both modes, the first attention layer's output
    // dumped (the dev sink, one layer, one tag) - the one place the attention
    // kernels are judged alone: downstream the model's top-10-of-512 routing
    // amplifies any rounding difference (the walk-geometry gates see the same
    // amplification: 1.4e-1 to 2e-1 at the logits)
    let p: Vec<u32> = base.iter().copied().cycle().take(1500).collect();
    let tmp = std::env::temp_dir().join(format!("paddock-qsa-gate-{}", std::process::id()));
    let walk = |m: &mut Qwen4ExpGpu, slot: usize, mode: QsaMode, sub: &str| {
        let d = tmp.join(sub);
        std::fs::create_dir_all(&d).expect("tmp");
        unsafe {
            std::env::set_var("PADDOCK_Q38FN_DUMP", &d);
            std::env::set_var("PADDOCK_Q38FN_DUMP_N", "1500");
            std::env::set_var("PADDOCK_Q38FN_DUMP_TAGS", "mix_out");
            std::env::set_var("PADDOCK_Q38FN_DUMP_LAYERS", "3");
        }
        assert!(m.set_qsa_mode(mode), "this pack has no QSA kernels");
        let l = m.prefill_slot(slot, &p).expect("prefill");
        for v in [
            "PADDOCK_Q38FN_DUMP",
            "PADDOCK_Q38FN_DUMP_N",
            "PADDOCK_Q38FN_DUMP_TAGS",
            "PADDOCK_Q38FN_DUMP_LAYERS",
        ] {
            unsafe { std::env::remove_var(v) };
        }
        let raw = std::fs::read(d.join("L3.mix_out.bin")).expect("layer 3 dump");
        let a3: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let mut t = vec![argmax(&l) as u32];
        for _ in 1..8 {
            let d = m
                .decode_step_batch(&[(slot, *t.last().unwrap())])
                .expect("decode");
            t.push(argmax(&d[0]) as u32);
        }
        (l, t, a3)
    };
    let (ld, td, ad) = walk(&mut m, 0, QsaMode::Dense, "dense");
    let (ls, ts, asp) = walk(&mut m, 1, QsaMode::Sparse, "sparse");
    let _ = std::fs::remove_dir_all(&tmp);
    let (r, r3) = (rel(&ls, &ld), rel(&asp, &ad));
    let same = ts.iter().zip(&td).take_while(|(a, b)| a == b).count();
    eprintln!(
        "INSIDE (1500): layer-3 attention output sparse vs dense rel {r3:.2e}; logits argmax {} \
         vs {}, rel {r:.2e}; chain {same}/8",
        argmax(&ls),
        argmax(&ld)
    );
    // the sparse kernel sits 4.5e-8 from f64 (op gate); the dense kernels
    // ~1e-5 (v4) to ~5e-3 (the WMMA tile's f16 O) - a wrong selection inside
    // the window (dropping or duplicating tokens) moves this O(1e-1)
    assert!(
        r3 < 1e-2,
        "the first attention layer's output moved: rel {r3}"
    );
    // the logits, argmax and chain are printed above, not asserted: past
    // layer 3 the two walks see different inputs and this prompt amplifies
    // any difference (see the doc)
    // past it: QSA (Auto) against pinned dense - different models; recorded
    let q: Vec<u32> = base.iter().copied().cycle().take(3000).collect();
    m.set_qsa_mode(QsaMode::Dense);
    let (qd, _) = run(&mut m, 0, &q);
    m.set_qsa_mode(QsaMode::Auto);
    let (qs, _) = run(&mut m, 1, &q);
    assert!(
        qs.iter().all(|v| v.is_finite()),
        "QSA logits past the window are not finite"
    );
    eprintln!(
        "PAST (3000): QSA vs dense argmax {} vs {}, rel {:.2e}",
        argmax(&qs),
        argmax(&qd),
        rel(&qs, &qd)
    );
}

/// Zero-copy resume across a release. A continued conversation resumes from
/// the radix: the pages its turn filed (the prompt, the reply's closed pages)
/// are adopted by refcount, the state restored from the reply checkpoint's
/// pool pages. Its slot's pages go back to the pool when the slot idles, and
/// the radix keeps what it filed - so the next turn, run in ANOTHER slot
/// after the release, must resume at the same point as the unreleased
/// same-slot resume, with BIT-identical logits (same pages, same checkpoint,
/// same walk geometry). Two loads: a turn-2 walk files its own checkpoints,
/// after which the same prompt elsewhere is an exact re-send (cold).
#[test]
fn gguf_resume_survives_a_release_into_another_slot() {
    if !common::heavy() {
        return;
    }
    let _one = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(path) = common::model("QWEN38FN_GGUF", &[]) else {
        common::missing("QWEN38FN_GGUF");
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
    let line = tok
        .encode(
            "Ledger entry: the courier signed for parcel AB-17 at the north depot, weight 4.5 kg, \
             route Umea to Malmo, note standard handling. ",
        )
        .expect("encode");
    let mut a: Vec<u32> = Vec::new();
    while a.len() < 1500 {
        a.extend(&line);
    }
    a.truncate(1500);
    let msg = tok
        .encode("Now list every parcel that went through the north depot and its weight.")
        .expect("encode");

    // one conversation: the prompt and a 40-token greedy reply in slot 0
    // (its decode closes pages, so the reply checkpoint rolls), then the
    // history + reply + a new message in `slot2`, slot 0 released first when
    // asked. Returns (resume point, the next turn's logits).
    let turn2 = |release: bool, slot2: usize| -> (usize, Vec<f32>) {
        let mut m = load(&exec, &path);
        let l = m.prefill_slot(0, &a).expect("prefill");
        let mut reply = vec![argmax(&l) as u32];
        for _ in 1..40 {
            let d = m
                .decode_step_batch(&[(0, *reply.last().unwrap())])
                .expect("decode");
            reply.push(argmax(&d[0]) as u32);
        }
        if release {
            m.release_inactive_slots(&[false, false, false]);
        }
        let mut b = a.clone();
        b.extend(&reply);
        b.extend(&msg);
        let l2 = m.prefill_slot(slot2, &b).expect("prefill turn 2");
        (m.take_prefill_reused(slot2), l2)
    };
    let (at_same, l_same) = turn2(false, 0);
    let (at_moved, l_moved) = turn2(true, 1);
    eprintln!(
        "RESUME: same slot at {at_same}, released and moved at {at_moved}; logits bit-identical: {}",
        l_same == l_moved
    );
    assert!(
        at_same > a.len(),
        "the resume did not reach into the reply ({at_same})"
    );
    assert_eq!(at_moved, at_same, "the moved resume point differs");
    assert!(
        l_moved == l_same,
        "a resume after the release is not the same-slot resume"
    );
}
