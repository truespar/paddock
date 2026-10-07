//! gpt-oss-20b GPU cross-path consistency: batched vs single-stream, prefill
//! vs decode, cache vs recompute - one numeric class agreeing with itself.
//! (There is no CPU-reference greedy test here - same-weights llama.cpp
//! parity is the oracle.)
//! Heavy + gated (needs the model, the pack, a GPU, and PADDOCK_HEAVY_TESTS=1;
//! run --release).
//!
//!   cargo test -p paddock-engine --release --test gpu_gpt_oss_parity -- --nocapture

mod common;

use paddock_engine::gpu::{GpuExecutor, KvDtype};
use paddock_engine::gpu_model::gpt_oss::GpuGptOss;
use paddock_engine::spec::NgramDraft;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// Relative-error bound for batched-vs-single-stream logit comparisons.
///
/// The KV cache is fp16 (2× the decode-attention bandwidth). The batched and
/// single-stream paths compute K/V through *different* GEMM kernels (gemm vs
/// gemv) whose f16 roundings diverge at rounding boundaries, and that difference
/// compounds over a long prefill - so these paths agree to ~1e-2, not the ~1e-6
/// f32 gave. This is inherent fp16-KV behavior (every production engine ships
/// fp16/bf16 KV); the real correctness gate is the greedy-token (argmax) match
/// asserted alongside, plus the attention-kernel parity tests (~5e-8) and decode
/// parity (~4e-5). Running many heavy models in one process adds cross-context
/// noise on top, so this bound is generous; heavy tests are most precise run
/// one-per-process. Since the exact-f32 pin was dropped these tests
/// run the real serving lanes, so cross-path comparisons also cross numeric
/// classes: b=1 rides dp4a, 2..=dp4a_max the batched dp4a MoE, prefill chunks
/// the int8 mmq or fp8 block-scale sorted MoE - all llama.cpp-class, all with
/// fixed-order (non-atomic) folds, each anchored externally by the same-weights
/// llama.cpp greedy gate. The greedy-token asserts are the real gate; this L2
/// is a loose sanity bound (a wrong attention window or slot mapping is O(1)
/// rel_err, far above the ~1e-2..1e-1 cross-class rounding spread).
const F16_KV_REL: f32 = 3e-1;

// (ExactB1Pin used to live here: it pinned PADDOCK_NO_MMQ /
// PADDOCK_NO_DP4A_B1 so cross-path tests compared "one exact-f32 class" -
// but since the g||u interleave those pinned MoE lanes misread
// the fused plane identically on both sides of every comparison, so the
// tests were green on matching garbage. The lanes and the pin are gone;
// the real lanes compared here are deterministic per path.)

/// forward_batch of B identical sequences must match single-stream forward_one:
/// same prefill, then one batched decode step where every row is the same
/// sequence at the same position -> B identical logit rows, each == single-stream.
/// The end-to-end validation of the batched decode path. Two-tier gate since
/// the exact-f32 pin collapse: the B rows are one computation
/// repeated, so they must agree with each other bit-for-bit (row-mixing/slot
/// bugs, no numeric noise involved) - while row-vs-single crosses numeric
/// classes (batched dp4a_b/mma vs b=1 dp4a) and gates greedy + loose L2.
#[test]
fn forward_batch_matches_single_stream() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let prompt = tok
        .encode("The three laws of robotics are")
        .expect("encode");

    // single-stream: prefill, then the next-token logits
    let mut single = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    single.set_kv_dtype(KvDtype::Fp16);
    single.reset();
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = single.forward_one(t).expect("fwd");
    }
    let next = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0u32, |(i, _)| i as u32);
    let single_next_logits = single.forward_one(next).expect("fwd"); // logits after `next`

    // batched: B copies of the same sequence, replayed step-by-step into the
    // per-slot caches. The last step (processing `next` at position prompt.len())
    // must produce the same logits the single path measured.
    let batch = 4usize;
    single.enable_batch(batch).expect("enable_batch");
    let full: Vec<u32> = prompt
        .iter()
        .copied()
        .chain(std::iter::once(next))
        .collect();
    let mut bl = Vec::new();
    for (pos, &t) in full.iter().enumerate() {
        let row: Vec<u32> = vec![t; batch];
        let positions: Vec<u32> = vec![pos as u32; batch];
        bl = single.forward_batch(&row, &positions).expect("fwd_batch");
    }
    let vocab = single_next_logits.len();
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    // identical rows, one computation: bit-for-bit agreement or a row leaked
    for r in 1..batch {
        assert_eq!(
            bl[r * vocab..(r + 1) * vocab],
            bl[0..vocab],
            "row {r} differs from row 0 on identical inputs"
        );
    }
    for r in 0..batch {
        let row = &bl[r * vocab..(r + 1) * vocab];
        let num: f64 = row
            .iter()
            .zip(&single_next_logits)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let den: f64 = single_next_logits.iter().map(|x| (*x as f64).powi(2)).sum();
        let rel = (num.sqrt() / den.sqrt().max(1e-12)) as f32;
        eprintln!("batch row {r}: rel_err vs single-stream {rel:.2e}");
        assert_eq!(
            amax(row),
            amax(&single_next_logits),
            "row {r} greedy != single-stream"
        );
        assert!(rel < F16_KV_REL, "row {r}: {rel} too high");
    }
    eprintln!("forward_batch matches single-stream across {batch} rows");
}

/// Prefix caching: a second prompt sharing a long prefix with a cached one must,
/// after copying the cached prefix KV + prefilling only its divergent tail, produce
/// the same logits as a full single-stream run. Validates the LCP + KV-copy +
/// start_pos-prefill path (the shared-system-prompt win).
#[test]
fn forward_prefill_prefix_cache_reuse() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    // shared prefix that is NON-page-aligned (70 tokens) + long distinct suffixes,
    // so the diverging page is cached and the token-granular tail match engages
    // (the shared tokens 64..70 must be reclaimed from a sibling page).
    let base: Vec<u32> = tok
        .encode("The following is a detailed technical specification describing a distributed consensus protocol")
        .expect("enc");
    let prefix: Vec<u32> = base.iter().cycle().take(70).copied().collect();
    // distinct VARIED-text tails (not a single repeated token): a repeated-token
    // tail makes the next-token a near-tie that fp16 KV legitimately flips, which
    // would make the greedy gate meaningless. Real text gives a clear-winner token.
    let tail_a: Vec<u32> = tok
        .encode(" and the algorithm proceeds through three distinct phases in sequence")
        .expect("enc");
    let tail_b: Vec<u32> = tok
        .encode(" whereas the alternative formulation instead relies on a shared global clock")
        .expect("enc");
    let a: Vec<u32> = prefix
        .iter()
        .copied()
        .chain(tail_a.iter().cycle().take(24).copied())
        .collect();
    let b: Vec<u32> = prefix
        .iter()
        .copied()
        .chain(tail_b.iter().cycle().take(24).copied())
        .collect();

    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    let single_last = |m: &mut GpuGptOss, seq: &[u32]| -> Vec<f32> {
        m.reset();
        let mut l = Vec::new();
        for &t in seq {
            l = m.forward_one(t).expect("fwd");
        }
        l
    };
    let ref_a = single_last(&mut m, &a);
    let ref_b = single_last(&mut m, &b);

    m.enable_batch(4).expect("enable_batch");
    // slot 0: prompt A -> fills the (empty) cache
    let got_a = m.forward_prefill(0, &a).expect("prefill a");
    // slot 1: prompt B -> must HIT the cache on the shared prefix, prefill only tail
    let got_b = m.forward_prefill(1, &b).expect("prefill b");
    let ra = rel(&got_a, &ref_a);
    let rb = rel(&got_b, &ref_b);
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let _rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    eprintln!("cache: A rel_err {ra:.2e} (fills cache), B rel_err {rb:.2e} (reuses prefix)");
    eprintln!(
        "  greedy A: prefill {} single {} | B: prefill {} single {}",
        amax(&got_a),
        amax(&ref_a),
        amax(&got_b),
        amax(&ref_b)
    );
    // greedy-token match is the real gate; L2 is loose under fp16 KV (F16_KV_REL)
    assert_eq!(
        amax(&got_a),
        amax(&ref_a),
        "A greedy token wrong (cache fill)"
    );
    assert_eq!(
        amax(&got_b),
        amax(&ref_b),
        "B greedy token wrong (cache reuse)"
    );
    assert!(ra < F16_KV_REL, "A diverged: {ra}");
    assert!(rb < F16_KV_REL, "B (cached-prefix reuse) diverged: {rb}");
    eprintln!("prefix-cache reuse matches single-stream");
}

/// Batched-tail prefill: several prompts prefilled in one weight-amortized pass
/// (their divergent tails concatenated, mapped to their slots/positions) must each
/// match a single-stream run. Covers (a) two prompts with no shared prefix batched
/// cold, and (b) two prompts reusing a pre-cached shared prefix batched together -
/// including different tail lengths (so last-row extraction is exercised).
#[test]
fn forward_prefill_batch_matches_single_stream() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mk = |s: &str, n: usize| -> Vec<u32> {
        tok.encode(s)
            .expect("enc")
            .iter()
            .cycle()
            .take(n)
            .copied()
            .collect()
    };
    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    // no-shared-prefix pair (different lengths -> last rows at different indices)
    let x = mk(
        "A story about sailing ships crossing the wide ocean at night",
        40,
    );
    let y = mk(
        "Notes on compiler optimization passes and register allocation strategies here",
        55,
    );
    // shared-prefix trio: p pre-cached, then q and r batched (reuse + tail batch)
    let p = mk(
        "System prompt describing the assistant persona and safety policy in detail",
        60,
    );
    let q: Vec<u32> = p
        .iter()
        .copied()
        .chain(std::iter::repeat_n(220u32, 5))
        .collect();
    let r: Vec<u32> = p
        .iter()
        .copied()
        .chain(std::iter::repeat_n(700u32, 9))
        .collect();

    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    let single = |m: &mut GpuGptOss, seq: &[u32]| -> Vec<f32> {
        m.reset();
        let mut l = Vec::new();
        for &t in seq {
            l = m.forward_one(t).expect("fwd");
        }
        l
    };
    let (rx, ry, rq, rr) = (
        single(&mut m, &x),
        single(&mut m, &y),
        single(&mut m, &q),
        single(&mut m, &r),
    );

    m.enable_batch(4).expect("enable_batch");
    // cold batched full-prefill of two distinct prompts in one pass
    let cold = m
        .forward_prefill_batch(&[(0, x.clone()), (1, y.clone())])
        .expect("batch cold");
    let (ex, ey) = (rel(&cold[0], &rx), rel(&cold[1], &ry));
    // pre-cache p's prefix, then batch q & r (both reuse it), different tail lengths
    m.forward_prefill(2, &p).expect("cache p");
    let warm = m
        .forward_prefill_batch(&[(0, q.clone()), (3, r.clone())])
        .expect("batch warm");
    let (eq, er) = (rel(&warm[0], &rq), rel(&warm[1], &rr));
    eprintln!("cold batch: X {ex:.2e} Y {ey:.2e}; warm batch (reuse): Q {eq:.2e} R {er:.2e}");
    // greedy-token match is the real gate; L2 is loose under fp16 KV (F16_KV_REL)
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let _rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    for (name, got, refl) in [
        ("X", &cold[0], &rx),
        ("Y", &cold[1], &ry),
        ("Q", &warm[0], &rq),
        ("R", &warm[1], &rr),
    ] {
        assert_eq!(
            amax(got),
            amax(refl),
            "{name} greedy token wrong in batched prefill"
        );
    }
    for (name, e) in [("X", ex), ("Y", ey), ("Q", eq), ("R", er)] {
        assert!(e < F16_KV_REL, "{name} diverged in batched prefill: {e}");
    }
    eprintln!("batched-tail prefill matches single-stream (cold + cache-reuse)");
}

/// RadixAttention multi-entry: two distinct prefixes cached at once, and a later
/// prompt reusing each - the win over a single-entry cache (which would thrash).
/// Validates both correctness (each reused prompt matches single-stream) AND that
/// reuse actually happened (the cache's reused-page counter climbs).
///
/// The prefixes must exceed swa_window (128): the P5c resume only adopts blocks
/// at an SWA-window checkpoint, and checkpoints exist at >= swa_window-token
/// boundaries (paged_prefix_resume rejects pos < swa_window outright). The
/// original 64-token prefixes predate that redesign and could never count a
/// reused page - found red during a sweep, and pre-existing rather than a
/// regression.
#[test]
fn forward_prefill_radix_multi_entry() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mk = |s: &str| -> Vec<u32> {
        let base = tok.encode(s).expect("enc");
        base.iter().cycle().take(160).copied().collect()
    };
    let p1 = mk("Chapter one begins with a description of the mountain range and its rivers");
    let p2 =
        mk("The quarterly financial report summarizes revenue costs and projected earnings growth");
    let a: Vec<u32> = p1.iter().copied().chain([220u32, 11, 12]).collect();
    let b: Vec<u32> = p2.iter().copied().chain([220u32, 13, 14]).collect();
    let c: Vec<u32> = p1.iter().copied().chain([220u32, 15]).collect(); // shares p1 with A
    let d: Vec<u32> = p2.iter().copied().chain([220u32, 16]).collect(); // shares p2 with B

    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    let single = |m: &mut GpuGptOss, seq: &[u32]| -> Vec<f32> {
        m.reset();
        let mut l = Vec::new();
        for &t in seq {
            l = m.forward_one(t).expect("fwd");
        }
        l
    };
    let ref_c = single(&mut m, &c);
    let ref_d = single(&mut m, &d);

    m.enable_batch(4).expect("enable_batch");
    m.forward_prefill(0, &a).expect("a"); // caches p1 branch
    m.forward_prefill(1, &b).expect("b"); // caches p2 branch (both now resident)
    let r0 = m.prefix_cache_reused_blocks();
    let got_c = m.forward_prefill(2, &c).expect("c"); // must reuse p1
    let r1 = m.prefix_cache_reused_blocks();
    let got_d = m.forward_prefill(3, &d).expect("d"); // must reuse p2
    let r2 = m.prefix_cache_reused_blocks();

    let rc = rel(&got_c, &ref_c);
    let rd = rel(&got_d, &ref_d);
    eprintln!(
        "C reused {} pages rel_err {rc:.2e}; D reused {} pages rel_err {rd:.2e}",
        r1 - r0,
        r2 - r1
    );
    // greedy-token match is the real gate; L2 is loose under fp16 KV (F16_KV_REL)
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let _rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    assert_eq!(
        amax(&got_c),
        amax(&ref_c),
        "C greedy token wrong (page reuse)"
    );
    assert_eq!(
        amax(&got_d),
        amax(&ref_d),
        "D greedy token wrong (page reuse)"
    );
    assert!(rc < F16_KV_REL, "C diverged: {rc}");
    assert!(rd < F16_KV_REL, "D diverged: {rd}");
    assert!(
        r1 - r0 >= 2,
        "C did not reuse p1 (multi-entry failed): {} pages",
        r1 - r0
    );
    assert!(
        r2 - r1 >= 2,
        "D did not reuse p2 (multi-entry failed): {} pages",
        r2 - r1
    );
    eprintln!("radix multi-entry: both prefixes resident and reused, parity holds");
}

/// Parallel prefill must match token-by-token single-stream. A >256-token prompt
/// (exercises the multi-chunk causal path) is prefilled in one call; its last
/// logits, the greedy next token, AND the following decode step must all match a
/// single-stream forward_one replay - proving both the returned logits and the
/// KV the prefill wrote are correct.
#[test]
fn forward_prefill_matches_single_stream() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let base = tok
        .encode("The history of the Roman empire is long and storied and")
        .expect("enc");
    // > PREFILL_CHUNK (256) so the chunked causal path is exercised
    let prompt: Vec<u32> = base.iter().cycle().take(300).copied().collect();

    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0u32, |(i, _)| i as u32)
    };
    let rel = |a: &[f32], b: &[f32]| -> f32 {
        let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };

    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    // single-stream reference: last logits, then the step after the greedy token
    m.reset();
    let mut single_last = Vec::new();
    for &t in &prompt {
        single_last = m.forward_one(t).expect("fwd");
    }
    let next = argmax(&single_last);
    let single_next = m.forward_one(next).expect("fwd");

    // parallel prefill into slot 0, then one decode step at position |prompt|
    m.enable_batch(2).expect("enable_batch");
    let vocab = m.vocab;
    let pl = m.forward_prefill(0, &prompt).expect("prefill");
    let r_last = rel(&pl, &single_last);
    eprintln!(
        "prefill last-logits rel_err {r_last:.2e}, next single={next} prefill={}",
        argmax(&pl)
    );
    // greedy token is the real gate (fp16 KV -> looser L2, see F16_KV_REL)
    assert_eq!(argmax(&pl), next, "prefill greedy token wrong");
    assert!(
        r_last < F16_KV_REL,
        "prefill last logits diverged: {r_last}"
    );

    let dl = m
        .forward_batch(&[next, 0], &[prompt.len() as u32, 0])
        .expect("decode");
    let r_next = rel(&dl[0..vocab], &single_next);
    eprintln!("post-prefill decode rel_err {r_next:.2e}");
    assert!(
        r_next < F16_KV_REL,
        "decode after prefill diverged: {r_next}"
    );
    eprintln!("parallel prefill matches single-stream (300-token, multi-chunk)");
}

/// The scheduler runs sequences at different positions in the same batched step
/// (staggered admission). This validates that per-row positions work: two
/// distinct sequences, one already decoding at a high position while the other
/// prefills from 0, must each still greedily match their independent single-
/// stream runs. A bug that used row 0's position for all rows (or mis-ranged the
/// attention) would diverge here but not in the uniform-position test.
#[test]
fn forward_batch_heterogeneous_positions() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let a = tok.encode("The capital of France is").expect("enc a");
    let b = tok
        .encode("Water boils at a temperature of")
        .expect("enc b");

    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1))
            .map_or(0u32, |(i, _)| i as u32)
    };

    // Independent single-stream greedy references (own KV cache, reset between).
    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    let greedy = |m: &mut GpuGptOss, prompt: &[u32], n: usize| -> Vec<u32> {
        m.reset();
        let mut logits = Vec::new();
        for &t in prompt {
            logits = m.forward_one(t).expect("fwd");
        }
        let mut out = Vec::new();
        for _ in 0..n {
            let nx = argmax(&logits);
            out.push(nx);
            logits = m.forward_one(nx).expect("fwd");
        }
        out
    };
    let ref_a = greedy(&mut m, &a, 1);
    let ref_b = greedy(&mut m, &b, 1);

    // Batched: slot 0 runs A alone first (advancing its position), then slot 1
    // starts B from position 0 - so every joint step has the two rows at
    // different positions. Row 0 = A, row 1 = B; dummy (0,0) for an idle row.
    //
    // Gate design (exact-f32 pin collapse): the batched lanes and
    // forward_one are different numeric classes now, and A's continuation goes
    // marginal a couple of tokens past " Paris" - greedy STREAM equality across
    // classes is a knife-edge, not a property. So the stream gate is
    // SAME-CLASS: run the identical slot-0 schedule twice - once with slot 1
    // idle the whole way, once with slot 1 prefilling B mid-stream - and
    // require slot 0's stream to match bit-for-bit between the runs. Rows are
    // computed independently, so any difference means slot 1's positions or KV
    // leaked into slot 0 (exactly the bug this test exists to catch, and
    // sharper than the old cross-class check). The single-stream references
    // gate only each prompt's clear-winner first token across classes.
    m.enable_batch(2).expect("enable_batch");
    let vocab = m.vocab;
    let run = |m: &mut GpuGptOss, with_b: bool| -> (Vec<u32>, u32) {
        // slot 0 prefill A (slot 1 idle); keep the last row-0 logits
        let mut pos0 = 0u32;
        let mut l = Vec::new();
        for &t in &a {
            l = m.forward_batch(&[t, 0], &[pos0, 0]).expect("fwd_batch");
            pos0 += 1;
        }
        let mut last0 = argmax(&l[0..vocab]); // A's first greedy token
        // slot 0 decodes 2 tokens alone (gets ahead)
        let mut a_got = Vec::new();
        for _ in 0..2 {
            a_got.push(last0);
            l = m.forward_batch(&[last0, 0], &[pos0, 0]).expect("fb");
            pos0 += 1;
            last0 = argmax(&l[0..vocab]);
        }
        // joint phase: slot 1 either stays idle or prefills B from pos 0
        // while slot 0 keeps decoding at pos0
        for (pos1, &bt) in (0u32..).zip(b.iter()) {
            let (t1, p1) = if with_b { (bt, pos1) } else { (0, 0) };
            l = m
                .forward_batch(&[last0, t1], &[pos0, p1])
                .expect("fb joint");
            a_got.push(last0);
            last0 = argmax(&l[0..vocab]);
            pos0 += 1;
        }
        // after the last joint step, row 1's logits (having consumed all of B)
        // predict B's first greedy token
        (a_got, argmax(&l[vocab..2 * vocab]))
    };
    let (iso_stream, _) = run(&mut m, false); // slot 1 idle throughout
    let (het_stream, b_first) = run(&mut m, true); // slot 1 prefills B mid-stream

    eprintln!(
        "iso={iso_stream:?}\n het={het_stream:?}\n ref_a[0]={} | ref_b[0]={} b_first={b_first}",
        ref_a[0], ref_b[0]
    );
    assert_eq!(
        het_stream, iso_stream,
        "slot 0 (A) diverged when slot 1 prefilled at a different position"
    );
    assert_eq!(
        het_stream[0], ref_a[0],
        "A first greedy token wrong (batched vs single)"
    );
    assert_eq!(
        b_first, ref_b[0],
        "slot 1 (B) first token wrong at heterogeneous position"
    );
    eprintln!("heterogeneous-position batched decode: slot isolation exact, first tokens match");
}

/// Profiling target: a steady stream of B=64 forward_batch steps (the sorted-MoE
/// regime) with nothing else, so `ncu -k regex:sorted` captures clean B=64 launches.
/// Not an assertion - just a harness for the profiler.
#[test]
fn profile_moe_b64() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let mut gpu = GpuGptOss::load(std::sync::Arc::new(exec), &map, 1024).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    gpu.set_kv_dtype(KvDtype::Fp16);
    gpu.enable_batch(64).expect("enable_batch");
    let toks: Vec<u32> = vec![100u32; 64];
    for s in 0..40u32 {
        let pos: Vec<u32> = vec![s; 64];
        gpu.forward_batch(&toks, &pos).expect("fwd");
    }
}

/// Aggregate throughput of the batched decode path as B scales - the weight-
/// amortization win. Weights are read once per step and B tokens come out, so
/// aggregate tok/s should climb with B (vs the serial engine's flat rate).
#[test]
fn batched_aggregate_throughput() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let mut gpu = GpuGptOss::load(std::sync::Arc::new(exec), &map, 1024).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    gpu.set_kv_dtype(KvDtype::Fp16);
    let max_batch = 64usize;
    gpu.enable_batch(max_batch).expect("enable_batch");
    let steps = 64usize;
    eprintln!("batched decode aggregate throughput (gpt-oss-20b):");
    for &b in &[1usize, 2, 4, 8, 16, 32, 64] {
        // warm at a fixed low position
        let toks: Vec<u32> = vec![100u32; b];
        for p in 0..8u32 {
            let pos: Vec<u32> = vec![p; b];
            gpu.forward_batch(&toks, &pos).expect("warm");
        }
        let t0 = std::time::Instant::now();
        for s in 0..steps {
            let pos: Vec<u32> = vec![8 + s as u32; b];
            gpu.forward_batch(&toks, &pos).expect("fwd");
        }
        let dt = t0.elapsed().as_secs_f64();
        let per_step_ms = dt * 1e3 / steps as f64;
        let aggregate = (b * steps) as f64 / dt;
        eprintln!(
            "  B={b:>2}: {per_step_ms:6.2} ms/step | aggregate {aggregate:7.1} tok/s | per-seq {:.1} tok/s",
            aggregate / b as f64
        );
    }
}

/// Diagnostic (heavy): wall time to prefill a long prompt - the TTFT-critical
/// path. Times forward_prefill for a ~400-token prompt (the B0 shape).
#[test]
fn prefill_timing() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 1024).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    m.enable_batch(16).expect("enable_batch");
    // COLD prefill: every iteration gets a DISTINCT prompt (varied from token 0)
    // so the radix prefix cache never matches. Re-running one prompt measured
    // cache-hit TTFT (~flat 24 ms at any length), not prefill.
    let prompt = |seed: u64, len: usize| -> Vec<u32> {
        (0..len)
            .map(|i| {
                let h = (seed.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15)) >> 33;
                (h % 100_000) as u32
            })
            .collect()
    };
    let mut seed = 1u64;
    for &plen in &[128usize, 400, 800] {
        m.forward_prefill(0, &prompt(seed, plen)).expect("warm");
        seed += 1;
        let iters = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            m.forward_prefill(0, &prompt(seed, plen)).expect("prefill");
            seed += 1;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        eprintln!(
            "  prefill {plen:>4} tokens: {ms:7.2} ms ({:.0} tok/s)",
            plen as f64 / ms * 1e3
        );
    }
}

/// Diagnostic (heavy): decode launch/sync-bound vs GPU-bound. If no-sync is much
/// faster than synced-per-token, per-launch/sync overhead is exposed and CUDA
/// graphs will help; if equal, we're GPU-bound and graphs would be neutral.
#[test]
fn decode_launch_bound_probe() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let prompt = tok
        .encode("The three laws of robotics are")
        .expect("encode");
    let mut gpu = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("gpu load");
    // the exact class this test is stated in (the KV8 default is lossy)
    gpu.set_kv_dtype(KvDtype::Fp16);
    // small N fills GPU bubbles between a few tokens (graphs-help signal) without
    // overflowing the WDDM command queue (which large N does, confounding it).
    for n in [2usize, 4, 8, 32, 128] {
        let (synced, nosync) = gpu.bench_launch_bound(&prompt, n).expect("bench");
        eprintln!(
            "N={n:>3}: synced {synced:.3} ms/tok | no-sync {nosync:.3} ms/tok | reclaimable {:.1}%",
            (synced - nosync) / synced * 100.0
        );
    }
}

/// B=1 decode latency vs context length, with the batched FlashDecoding split on
/// vs off - confirms the split (path-unify phase 1) fills the GPU and keeps
/// long-context B=1 decode from scaling with context. Latency depends only on how
/// many KV positions attention walks (n_pos), not their content, so we decode at
/// high positions against a zeroed cache - the kernels do the same work.
#[test]
fn b1_long_context_decode_latency() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let mut gpu = GpuGptOss::load(std::sync::Arc::new(exec), &map, 4096).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    gpu.set_kv_dtype(KvDtype::Fp16);
    gpu.enable_batch(1).expect("enable_batch");
    let steps = 32usize;

    let measure = |gpu: &mut GpuGptOss, ctx: usize, on: bool| -> f64 {
        paddock_engine::gpu_model::gpt_oss::set_attn_split(on);
        for i in 0..8u32 {
            gpu.forward_batch(&[100], &[(ctx + i as usize) as u32])
                .expect("warm");
        }
        let t0 = std::time::Instant::now();
        for i in 0..steps {
            gpu.forward_batch(&[100], &[(ctx + 8 + i) as u32])
                .expect("dec");
        }
        t0.elapsed().as_secs_f64() * 1e3 / steps as f64
    };

    eprintln!("B=1 decode latency vs context (gpt-oss-20b, A6000):");
    eprintln!("   ctx | split-off ms/tok | split-on ms/tok | speedup");
    for &ctx in &[256usize, 512, 1024, 2048, 3072] {
        let off = measure(&mut gpu, ctx, false);
        let on = measure(&mut gpu, ctx, true);
        eprintln!("  {ctx:>4} | {off:>16.2} | {on:>15.2} | {:.2}x", off / on);
    }
    paddock_engine::gpu_model::gpt_oss::set_attn_split(true);
}

/// fp8 E4M3 KV is an OPT-IN, lossy mode - this test doesn't gate on correctness
/// (fp8 flips some greedy tokens by design); it confirms the fp8 path runs end to
/// end and quantifies the drift vs the fp16 default, so the tradeoff is on record.
#[test]
fn fp8_kv_runs_and_quantifies_drift() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let prompt = tok
        .encode("The history of the Roman empire is long and storied and")
        .expect("enc");

    let steps = 24usize;

    let greedy = |m: &mut GpuGptOss, prompt: &[u32]| -> Vec<u32> {
        m.reset();
        let mut last = Vec::new();
        for &t in prompt {
            last = m.forward_one(t).expect("fwd");
        }
        let amax = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i) as u32
        };
        let mut out = Vec::new();
        let mut nxt = amax(&last);
        for _ in 0..steps {
            out.push(nxt);
            let l = m.forward_one(nxt).expect("fwd");
            nxt = amax(&l);
        }
        out
    };

    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    let vocab = m.vocab;
    // fp16 default reference
    let fp16 = greedy(&mut m, &prompt);
    // opt into fp8 KV and decode the same prompt
    m.set_kv_dtype(KvDtype::Fp8E4m3);
    let fp8 = greedy(&mut m, &prompt);

    let matched = fp16.iter().zip(&fp8).take_while(|(a, b)| a == b).count();
    let same = fp16.iter().zip(&fp8).filter(|(a, b)| a == b).count();
    eprintln!(
        "fp8 KV drift: {}/{} greedy tokens match fp16 ({} agree before first divergence)",
        same, steps, matched
    );
    eprintln!("  fp16: {:?}", &fp16[..8.min(fp16.len())]);
    eprintln!("  fp8 : {:?}", &fp8[..8.min(fp8.len())]);
    // fp8 is lossy-by-design; the gate is only that it produced valid tokens
    assert!(
        fp8.iter().all(|&t| (t as usize) < vocab),
        "fp8 produced invalid token ids"
    );
    assert_eq!(fp8.len(), steps, "fp8 decode did not run to completion");
}

// (moe_tc_matches_and_benchmarks lived here until: it A/B'd the
// f32-sorted vs tensor-core-sorted MoE lanes through forward_batch. Serving
// no longer routes there - batched MoE rides the int8 mmq / fp8 block-scale
// sorted kernels - and the TC-vs-f32 kernel parity it asserted lives at the
// kernel level in gpu_moe_parity::sorted_mmq_moe_matches_sorted_f32.)

/// Heavy: `generate_greedy_spec` must (1) be deterministic run-to-run and
/// (2) match `generate_greedy` on clear-margin text. The templated-JSON
/// prompt gives the n-gram drafter real accept coverage (repeated structure)
/// AND reject coverage (per-line values), with greedy margins that dwarf the
/// batch-vs-b1 numeric-class gap, so exact equality is the bar. Verbatim-
/// repeat prompts are excluded: they sit on knife edges between the classes
/// (measured stream divergence), the same policy as story prompts in the
/// b9895 prefill gate.
#[test]
fn spec_greedy_matches_plain_greedy() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    // spec-vs-plain exactness is an int8-class property: the block-scale fp8
    // prefill shifts hidden states enough to flip near-tie tokens between the
    // dp4a decode and mmq verify kernels. Pin one class end to end.
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(false);
    let mut gpu = GpuGptOss::load(std::sync::Arc::new(exec), &map, 2048).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    gpu.set_kv_dtype(KvDtype::Fp16);
    let text = "Convert each item to a JSON object with fields name, id and \
                price, one per line:\napple 1 3.50\nbanana 2 1.25\ncherry 3 \
                8.00\ndamson 4 2.75\nelderberry 5 9.10\nfig 6 4.20\ngrape 7 \
                2.30\n\n{\"name\": \"apple\", \"id\": 1, \"price\": 3.50}\n";
    let prompt = tok.encode(text).expect("encode");
    let n = 96;
    let plain = gpu.generate_greedy(&prompt, n).expect("plain");
    let spec1 = gpu.generate_greedy_spec(&prompt, n, 7).expect("spec 1");
    let spec2 = gpu.generate_greedy_spec(&prompt, n, 7).expect("spec 2");
    assert_eq!(spec1, spec2, "spec decode must be deterministic");
    assert_eq!(
        spec1, plain,
        "spec stream diverges from plain greedy on clear-margin text"
    );
    // different draft length must not change the stream either
    let spec3 = gpu.generate_greedy_spec(&prompt, n, 3).expect("spec k=3");
    assert_eq!(spec3, plain, "spec stream depends on n_draft");
    eprintln!("spec == plain over {n} tokens (k=7 and k=3), deterministic");
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(true);
}

/// Heavy: the multi-slot spec-batch round (`forward_spec_batch` with ragged
/// per-slot chunks) must (1) be deterministic and (2) reproduce each slot's
/// plain `forward_batch` greedy stream on clear-margin text - the templated
/// JSON workload, same bar as `spec_greedy_matches_plain_greedy`. Slots get
/// different item lists so their streams (and rounds' row counts) diverge,
/// exercising ragged accept/commit across slots in one pass.
#[test]
fn spec_batch_matches_plain_batch() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    // int8-class pin: see spec_greedy_matches_plain_greedy
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(false);
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut gpu = GpuGptOss::load(std::sync::Arc::new(exec), &map, 2048).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    gpu.set_kv_dtype(KvDtype::Fp16);
    let b = 3usize;
    gpu.enable_batch(b).expect("enable_batch");
    let vocab = 201088usize;
    let items = [
        "apple 1 3.50\nbanana 2 1.25\ncherry 3 8.00\ndamson 4 2.75\nfig 6 4.20\n",
        "kiwi 11 2.10\nlemon 12 0.80\nmango 13 5.40\nnectarine 14 3.30\nolive 15 7.70\n",
        "pear 21 1.90\nquince 22 6.60\nraisin 23 0.40\nsloe 24 9.90\ntomato 25 2.20\n",
    ];
    let prompts: Vec<Vec<u32>> = (0..b)
        .map(|s| {
            let text = format!(
                "Convert each item to a JSON object with fields name, id and \
                 price, one per line:\n{}\n{{\"name\": \"x\", \"id\": 0, \"price\": 0.0}}\n",
                items[s]
            );
            tok.encode(&text).expect("encode")
        })
        .collect();
    let argmax = |l: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, v) in l.iter().enumerate() {
            if *v > l[best] {
                best = i;
            }
        }
        best as u32
    };
    let n = 40usize;

    // plain per-tick greedy batch streams
    let mut pending = vec![0u32; b];
    let mut pos = vec![0u32; b];
    for (s, p) in prompts.iter().enumerate() {
        let logits = gpu.forward_prefill(s, p).expect("prefill");
        pending[s] = argmax(&logits);
        pos[s] = p.len() as u32;
    }
    let mut plain: Vec<Vec<u32>> = (0..b).map(|s| vec![pending[s]]).collect();
    while plain[0].len() < n {
        let logits = gpu.forward_batch(&pending, &pos).expect("fwd");
        for s in 0..b {
            pos[s] += 1;
            pending[s] = argmax(&logits[s * vocab..(s + 1) * vocab]);
            plain[s].push(pending[s]);
        }
    }

    // spec-batch streams, twice (determinism)
    let run_spec = |gpu: &mut GpuGptOss| -> Vec<Vec<u32>> {
        let mut pending = vec![0u32; b];
        let mut pos = vec![0usize; b];
        let mut drs: Vec<NgramDraft> = (0..b).map(|_| NgramDraft::default()).collect();
        for (s, p) in prompts.iter().enumerate() {
            let logits = gpu.forward_prefill(s, p).expect("prefill");
            pending[s] = argmax(&logits);
            pos[s] = p.len();
            for &t in p.iter().chain([pending[s]].iter()) {
                drs[s].push(t);
            }
        }
        let mut out: Vec<Vec<u32>> = (0..b).map(|s| vec![pending[s]]).collect();
        while out.iter().any(|st| st.len() < n) {
            let mut reqs = Vec::new();
            let mut live = Vec::new();
            for s in 0..b {
                if out[s].len() >= n {
                    continue;
                }
                let mut chunk = vec![pending[s]];
                chunk.extend(drs[s].draft(3));
                reqs.push((s, pos[s], chunk));
                live.push(s);
            }
            let picks = gpu.forward_spec_batch(&reqs).expect("spec batch");
            let mut base = 0usize;
            for (ri, s) in live.iter().copied().enumerate() {
                let chunk = &reqs[ri].2;
                let mut a = 0usize;
                while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                    a += 1;
                }
                for &t in picks[base..=base + a].iter() {
                    out[s].push(t);
                    drs[s].push(t);
                }
                pos[s] += a + 1;
                pending[s] = picks[base + a];
                base += chunk.len();
            }
        }
        out
    };
    let spec1 = run_spec(&mut gpu);
    let spec2 = run_spec(&mut gpu);
    for s in 0..b {
        assert_eq!(
            spec1[s][..n],
            spec2[s][..n],
            "slot {s}: spec-batch nondeterministic"
        );
        assert_eq!(
            spec1[s][..n],
            plain[s][..n],
            "slot {s}: spec-batch stream diverges from plain batch greedy"
        );
    }
    eprintln!("spec-batch == plain batch greedy over {n} tokens x {b} slots, deterministic");
    paddock_engine::gpu_model::gpt_oss::set_moe_bs(true);
}

/// P5c paged prefix reuse (pool mode): a prompt sharing a >swa_window prefix with
/// a cached one must, after zero-copy adopting the full-attn blocks AND restoring
/// the trailing SWA window from the checkpoint, produce the same greedy token as a
/// cold single-stream run. This is the SWA-checkpoint correctness gate - a wrong
/// window restore or block share is O(1) rel_err and flips the clear-winner token.
#[test]
fn paged_prefix_reuse_swa_checkpoint() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    // shared prefix > swa_window (128) tokens of real text (clear-winner greedy),
    // then two distinct real-text tails.
    // COHERENT (non-repeated) prefix > swa_window tokens, so the greedy next
    // token is numerically stable - the prefill and single-stream sides run
    // different numeric classes (mmq/bs chunks vs b=1 dp4a), whose ~f16-level
    // rounding gap flips near-tie tokens. Only a CLEAR-WINNER token is a
    // valid gate.
    let prefix: Vec<u32> = tok
        .encode("Geography is the study of places and the relationships between people and their \
                 environments. It examines the physical features of the Earth, such as mountains, \
                 rivers, valleys, deserts, and coastlines, as well as the many human societies that \
                 are spread across them. Students of geography learn about the seven continents, the \
                 great oceans, the varied climates of different regions, and the capital cities of \
                 the world's many nations, which serve as important centers of government, commerce, \
                 and culture. They also study how populations migrate, how borders shift over time, \
                 and how trade connects distant lands. France, a country located in Western Europe, \
                 is one such nation, with a long and storied history that stretches back for many \
                 centuries into the distant past of the European continent and its neighbors.")
        .expect("enc");
    assert!(
        prefix.len() > 130,
        "prefix must exceed swa_window (got {})",
        prefix.len()
    );
    // distinct tails that each force a CONFIDENT clear-winner completion, so the
    // greedy gate is robust to MoE noise yet still discriminating (A≠B).
    let tail_a: Vec<u32> = tok.encode(" The capital city of France is").expect("enc");
    let tail_b: Vec<u32> = tok
        .encode(" The largest planet in our solar system is")
        .expect("enc");
    let a: Vec<u32> = prefix.iter().copied().chain(tail_a).collect();
    let b: Vec<u32> = prefix.iter().copied().chain(tail_b).collect();
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };

    // (the serving MoE lanes all fold in fixed order - no atomic
    // scatter - so each path is deterministic run-to-run; the old set_moe_tc
    // pin targeted the deleted f32-sorted lane)
    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    // cold single-stream references (no pool, no reuse)
    let single_last = |m: &mut GpuGptOss, seq: &[u32]| -> Vec<f32> {
        m.reset();
        let mut l = Vec::new();
        for &t in seq {
            l = m.forward_one(t).expect("fwd");
        }
        l
    };
    let ref_a = single_last(&mut m, &a);
    let ref_b = single_last(&mut m, &b);
    // the two tails must diverge, else a pass proves nothing about the boundary KV
    assert_ne!(
        amax(&ref_a),
        amax(&ref_b),
        "test not discriminating: A,B predict same token"
    );

    // enable the paged budget pool -> forward_prefill_batch takes the P5c path
    // SAFETY: heavy GPU tests run serial (--test-threads=1)
    unsafe { std::env::set_var("PADDOCK_KV_POOL_BLOCKS", "2048") };
    m.enable_batch(4).expect("enable_batch");
    // slot 0: prompt A -> cold full prefill, fills the radix + SWA checkpoints
    let got_a = m
        .forward_prefill_batch(&[(0, a.clone())])
        .expect("prefill a")
        .remove(0);
    // slot 1: prompt B -> shares the 176-token prefix, RESUMES from a checkpoint
    let got_b = m
        .forward_prefill_batch(&[(1, b.clone())])
        .expect("prefill b")
        .remove(0);
    unsafe { std::env::remove_var("PADDOCK_KV_POOL_BLOCKS") };

    let (ra, rb) = (rel(&got_a, &ref_a), rel(&got_b, &ref_b));
    eprintln!(
        "p5c: A pool {} cold {} rel {ra:.2e} | B pool {} cold {} rel {rb:.2e}",
        amax(&got_a),
        amax(&ref_a),
        amax(&got_b),
        amax(&ref_b)
    );
    // rel_err is the robust gate: a wrong SWA-window restore or block share is
    // O(1) rel_err (skip-SWA measured token 1585, ~1.0 rel); a correct reuse
    // tracks cold within fp16-KV noise. Argmax can flip on near-tie tokens under
    // the nondeterministic sorted-MoE atomic scatter, so it is informational.
    // relative gate: reuse must not add much error over the cold-fill baseline
    // (path noise dominates the absolute rel). A broken restore is O(1) (skip-SWA
    // measured 0.90). ra itself bounds real cold-path breakage.
    assert!(ra < F16_KV_REL, "A diverged (pool cold fill): {ra}");
    assert!(
        rb < ra.max(F16_KV_REL) * 1.5,
        "B diverged (P5c SWA-checkpoint reuse): rb {rb} vs ra {ra}"
    );
    eprintln!("P5c paged prefix reuse + SWA checkpoint matches cold single-stream");
}

/// P5c reuse on the CHUNKED (forward_mixed) path - the DEFAULT serving path.
/// prefill_begin + forward_mixed must resume a shared prefix (full-attn share +
/// SWA-window restore) and match the cold single-stream greedy token.
#[test]
fn paged_prefix_reuse_mixed_path() {
    if !common::heavy() {
        return;
    }
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let Some(exec) = common::gpu() else {
        return;
    };
    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let prefix: Vec<u32> = tok
        .encode("Geography is the study of places and the relationships between people and their \
                 environments. It examines the physical features of the Earth, such as mountains, \
                 rivers, valleys, deserts, and coastlines, as well as the many human societies that \
                 are spread across them. Students of geography learn about the seven continents, the \
                 great oceans, the varied climates of different regions, and the capital cities of \
                 the world's many nations, which serve as important centers of government, commerce, \
                 and culture. They also study how populations migrate, how borders shift over time, \
                 and how trade connects distant lands. France, a country located in Western Europe, \
                 is one such nation, with a long and storied history that stretches back for many \
                 centuries into the distant past of the European continent and its neighbors.")
        .expect("enc");
    assert!(
        prefix.len() > 130,
        "prefix must exceed swa_window (got {})",
        prefix.len()
    );
    let tail_a: Vec<u32> = tok.encode(" The capital city of France is").expect("enc");
    let tail_b: Vec<u32> = tok
        .encode(" The largest planet in our solar system is")
        .expect("enc");
    let a: Vec<u32> = prefix.iter().copied().chain(tail_a).collect();
    let b: Vec<u32> = prefix.iter().copied().chain(tail_b).collect();
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    };
    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };

    let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(KvDtype::Fp16);
    let single_last = |m: &mut GpuGptOss, seq: &[u32]| -> Vec<f32> {
        m.reset();
        let mut l = Vec::new();
        for &t in seq {
            l = m.forward_one(t).expect("fwd");
        }
        l
    };
    let ref_a = single_last(&mut m, &a);
    let ref_b = single_last(&mut m, &b);
    assert_ne!(amax(&ref_a), amax(&ref_b), "test not discriminating");

    // SAFETY: heavy GPU tests run serial
    unsafe { std::env::set_var("PADDOCK_KV_POOL_BLOCKS", "2048") };
    m.enable_batch(4).expect("enable_batch");
    // drive a chunked prefill to completion on `slot`, returning last-token logits
    let mixed_last = |m: &mut GpuGptOss, slot: usize, seq: &[u32]| -> Vec<f32> {
        m.prefill_begin(slot, seq.to_vec()).expect("prefill_begin");
        loop {
            let (_dec, fin) = m.forward_mixed(&[], 8192).expect("forward_mixed");
            if let Some((_s, logits, _n)) = fin.into_iter().find(|f| f.0 == slot) {
                return logits;
            }
        }
    };
    let got_a = mixed_last(&mut m, 0, &a); // cold fill via chunked path
    let got_b = mixed_last(&mut m, 1, &b); // resumes shared prefix via chunked path
    unsafe { std::env::remove_var("PADDOCK_KV_POOL_BLOCKS") };

    let (ra, rb) = (rel(&got_a, &ref_a), rel(&got_b, &ref_b));
    eprintln!(
        "p5c-mixed: A {} cold {} rel {ra:.2e} | B {} cold {} rel {rb:.2e}",
        amax(&got_a),
        amax(&ref_a),
        amax(&got_b),
        amax(&ref_b)
    );
    // forward_mixed vs forward_one has a higher inherent rel than the batch path,
    // so gate A loosely and B RELATIVE to the cold-fill baseline (reuse error is
    // rb-ra; a broken restore is O(1)).
    assert!(
        ra < F16_KV_REL * 1.6,
        "A diverged (chunked cold fill): {ra}"
    );
    assert!(
        rb < ra.max(F16_KV_REL) * 1.5,
        "B diverged (chunked P5c reuse): rb {rb} vs ra {ra}"
    );
    eprintln!("P5c chunked-path prefix reuse matches cold single-stream");
}

/// Speculative verify under the budget pool grows the slot's paged block
/// table to cover the whole draft span before the verify reads it. This is the
/// core wiring: `forward_spec_batch_inner` calls `ensure_pool_rows` for every
/// draft position, so a span crossing a 16-token block boundary allocates the
/// next block. Without it the post-boundary positions alias an unallocated
/// (zero) block-table entry -> physical block 0 -> they write AND read the same
/// wrong block (self-consistent, so the immediate logits barely move) while
/// silently corrupting the KV of the sequence's first 16 tokens. The logit
/// impact is tiny and buried under sm120 verify-MoE noise, so this is validated
/// WHITE-BOX: the block table must grow by exactly the boundary-crossing blocks
/// with the growth on, and not grow with it off (`PADDOCK_NO_SPEC_POOL_GROW`).
/// A companion coherence check confirms the verify still produces in-vocab picks
/// that mostly match the confident cold continuation.
#[test]
fn spec_batch_under_pool_grows_span() {
    if !common::heavy() {
        return;
    }
    // keeps the pack PATH: the runs below each build their own executor
    let Some(pack) = common::pack() else {
        return;
    };
    let Some(model_path) = common::model("PADDOCK_MODEL", common::GPT_OSS_20B) else {
        return;
    };
    let amax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i) as u32
    };
    let prompt_txt = "Here is a list of the counting numbers, written out one after another in ascending \
         order, each number exactly one greater than the number that came before it. We will \
         simply keep counting upward for a long while without stopping: 1, 2, 3, 4, 5, 6, 7, \
         8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, \
         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, \
         51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, \
         72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, \
         93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110";
    const K: usize = 4;
    const BLK: usize = 16;

    let map = MappedGguf::open(&model_path).expect("open");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tok");
    let prompt: Vec<u32> = tok.encode(prompt_txt).expect("enc");
    // split so the K+1-row draft span [s, s+K] crosses a block boundary
    let s = {
        let mut s = ((prompt.len().saturating_sub(2)) / BLK) * BLK + (BLK - 1);
        while s + K >= prompt.len() {
            s = s.saturating_sub(BLK);
        }
        s
    };
    assert_ne!(
        s / BLK,
        (s + K) / BLK,
        "span must cross a block boundary (s={s})"
    );
    let chunk: Vec<u32> = prompt[s..=s + K].to_vec();
    let blocks_pre = s.div_ceil(BLK); // blocks covering [0, s)
    let blocks_span = (s + K + 1).div_ceil(BLK); // blocks covering [0, s+K]
    assert!(blocks_span > blocks_pre, "span adds no block - bad split");

    // one fresh pool run; returns (blocks_after_prefill, blocks_after_verify, verify_logits)
    let run = |grow: bool| -> (usize, usize, Vec<f32>) {
        let exec = GpuExecutor::new(0, &pack).expect("exec");
        let map = MappedGguf::open(&model_path).expect("open");
        unsafe { std::env::set_var("PADDOCK_KV_POOL_BLOCKS", "2048") };
        if grow {
            unsafe { std::env::remove_var("PADDOCK_NO_SPEC_POOL_GROW") };
        } else {
            unsafe { std::env::set_var("PADDOCK_NO_SPEC_POOL_GROW", "1") };
        }
        let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load");
        // the exact class this test is stated in (the KV8 default is lossy)
        m.set_kv_dtype(KvDtype::Fp16);
        m.enable_batch(2).expect("batch");
        assert!(m.pool_active());
        m.forward_prefill(0, &prompt[..s]).expect("prefill");
        let after_prefill = m.pool_slot_blocks(0).expect("pooled");
        let logits = m
            .spec_batch_logits(&[(0, s, chunk.clone())])
            .expect("verify");
        let after_verify = m.pool_slot_blocks(0).expect("pooled");
        unsafe { std::env::remove_var("PADDOCK_KV_POOL_BLOCKS") };
        unsafe { std::env::remove_var("PADDOCK_NO_SPEC_POOL_GROW") };
        (after_prefill, after_verify, logits)
    };

    // STABLE cold reference for ROW 0: forward_one over prompt[0..=s] (b=1 path is
    // low-noise). Row 0 sits at position s in the block the prefill already filled
    // and on a confident counting token, so its verify logits are reproducible -
    // a trustworthy gate that the verify actually reads the KV (not just a grown
    // but ignored table). Deeper rows are too MoE-noisy to gate on.
    let ref0: Vec<f32> = {
        let exec = GpuExecutor::new(0, &pack).expect("exec");
        let map = MappedGguf::open(&model_path).expect("open");
        let mut m = GpuGptOss::load(std::sync::Arc::new(exec), &map, 512).expect("load ref");
        // the exact class this test is stated in (the KV8 default is lossy)
        m.set_kv_dtype(KvDtype::Fp16);
        m.reset();
        let mut l = Vec::new();
        for &t in &prompt[..=s] {
            l = m.forward_one(t).expect("fwd");
        }
        l
    };

    let (gp_pre, gp_post, gp_logits) = run(true);
    let (ng_pre, ng_post, _ng_logits) = run(false);
    let vocab = ref0.len();
    eprintln!("expect prefill={blocks_pre} span={blocks_span}");
    eprintln!("with-grow: prefill {gp_pre} -> verify {gp_post}",);
    eprintln!("no-grow:   prefill {ng_pre} -> verify {ng_post}");

    // both configs prefill the same [0, s) -> same starting block count
    assert_eq!(gp_pre, blocks_pre, "prefill block count off");
    assert_eq!(ng_pre, blocks_pre, "prefill block count off");
    // With growth: the verify extends the table to cover the whole draft span.
    assert_eq!(
        gp_post, blocks_span,
        "spec verify did NOT grow the table across the block boundary"
    );
    // Without growth: the table is left un-grown (the boundary block is missing) -
    // proves the growth is what extends the span, not something incidental.
    assert_eq!(
        ng_post, blocks_pre,
        "growth ran despite PADDOCK_NO_SPEC_POOL_GROW"
    );

    // coherence: the confident pre-boundary row 0 tracks the cold single-stream
    // reference within fp16-KV slack, confirming the verify reads real KV.
    let rel = |x: &[f32], y: &[f32]| -> f32 {
        let num: f64 = x.iter().zip(y).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = y.iter().map(|p| (*p as f64).powi(2)).sum();
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    };
    let r0 = rel(&gp_logits[0..vocab], &ref0);
    eprintln!(
        "coherence: row-0 verify-vs-cold rel {r0:.3e} (pick {} vs cold {})",
        amax(&gp_logits[0..vocab]),
        amax(&ref0)
    );
    assert!(
        r0 < F16_KV_REL,
        "confident row-0 verify diverged from cold: {r0} (F16_KV_REL {F16_KV_REL})"
    );
    eprintln!(
        "spec verify grows the pool span across a block boundary (load-bearing) and reads correct KV"
    );
}
