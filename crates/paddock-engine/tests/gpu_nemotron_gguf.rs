//! GGUF-lane gates for nemotron: the unsloth Q8_0 file must
//! parse into the same geometry the NVFP4 lane serves, load onto the
//! device, and run the serial spine coherently - bulk prefill against the
//! token-by-token walk with the family's standard near-exact band (top-1 at
//! the boundary + identical greedy continuation + mean|Δ|/rms). The
//! INDEPENDENT correctness reference for this lane is llama.cpp consuming
//! the identical file (greedy parity over a fixed prompt set) - this test is
//! the internal-consistency gate that runs without a server.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;
use paddock_models::mapped::MappedGguf;
use paddock_models::nemotron::{NemotronBlock, NemotronConfig};

const GGUF_ENV: &str = "NEMOTRON_GGUF";
const GGUF: &str = "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf";
const ORACLE: &str = "/models/nemotron-battery/oracle/decoder-oracle.json";
const PROMPT_LEN: usize = 700;
const GREEDY_STEPS: usize = 24;

/// Every gate in this file loads a full ~20 GB nemotron Q8_0 onto the device.
/// libtest runs `#[test]` fns on parallel threads, so several of those loads
/// overlap and the card OOMs - seen as four spurious
/// `load gguf: CUDA_ERROR_OUT_OF_MEMORY` failures in a whole-suite run whose
/// tests all pass individually and under `--test-threads=1`. Same shape as
/// gpu_sample_rows.rs's static-scratch race: take a process-wide lock and
/// hold it for the model's entire GPU lifetime, so only one resident model
/// exists at a time. A spurious red here is worse than the serialization -
/// it reads exactly like a real load regression.
fn exec_locked() -> Option<(
    std::sync::MutexGuard<'static, ()>,
    std::sync::Arc<paddock_engine::gpu::GpuExecutor>,
)> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    let guard = LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    common::gpu_arc().map(|e| (guard, e))
}

fn gguf_path() -> Option<std::path::PathBuf> {
    let p = std::env::var(GGUF_ENV).unwrap_or_else(|_| GGUF.into());
    let p = std::path::PathBuf::from(p);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("skip: no nemotron Q8_0 gguf at {}", p.display());
        None
    }
}

/// How close the winner was: top1 - top2 over the row. A spec round verifies
/// k+1 rows in one launch where the reference decodes one row at a time, and
/// the reductions reassociate with the width - so at a near-tie the two can
/// pick different tokens without either being wrong. A divergence is only
/// worth chasing when this margin is not tiny, so the asserts carry it.
fn top2_margin(l: &[f32]) -> f32 {
    let mut a = f32::NEG_INFINITY;
    let mut b = f32::NEG_INFINITY;
    for &v in l {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

/// A reference top1-top2 gap at or below this is a TIE: the paths compared
/// here differ in batch width, and that reassociation can order two
/// candidates this close either way. Measured on this lane: observed ties
/// 5.6e-3, real divergences 5e-2 to 3e-1.
const TIE_MARGIN: f32 = 1e-2;

/// Compare two greedy continuations. A mismatch fails unless the REFERENCE's
/// own top two were tied there, in which case the comparison ends at that
/// token (everything after it follows the token that was picked) and says so.
/// `min_match` is what the gate must prove before a tie is allowed to end it.
fn same_until_tie(reference: &[u32], other: &[u32], margin: &[f32], min_match: usize, what: &str) {
    for i in 0..reference.len().min(other.len()) {
        if reference[i] == other[i] {
            continue;
        }
        let m = margin.get(i).copied().unwrap_or(f32::NAN);
        assert!(
            m <= TIE_MARGIN,
            "{what}: diverged at {i} ({} vs {}), reference margin {m:.3e} - too wide for a tie\n               reference: {reference:?}\n  other:     {other:?}",
            reference[i],
            other[i]
        );
        assert!(
            i >= min_match,
            "{what}: parted at {i}, before {min_match} tokens proved anything (margin {m:.3e})"
        );
        eprintln!(
            "{what}: {i} tokens identical, then the reference's own top-2 tied \
             (margin {m:.3e}) - the comparison ends there"
        );
        return;
    }
}

/// Per-step logits of one path against the reference's, at the same
/// positions and on the same fed tokens. Mean |d|/rms over the row and the
/// top-1 agreement, the same yardstick and bounds the NVFP4 lane's
/// batch-vs-serial gates use.
fn teacher_forced(reference: &[Vec<f32>], other: &[Vec<f32>], what: &str) {
    let n = reference.len().min(other.len());
    assert!(n > 0, "{what}: nothing to compare");
    let (mut sum, mut worst, mut top1) = (0f64, 0f64, 0usize);
    for i in 0..n {
        let (r, o) = (&reference[i], &other[i]);
        let rms =
            (r.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / r.len() as f64).sqrt();
        let d = r
            .iter()
            .zip(o)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .sum::<f64>()
            / r.len() as f64
            / rms.max(1e-3);
        sum += d;
        if d > worst {
            worst = d;
        }
        if argmax(r) == argmax(o) {
            top1 += 1;
        }
    }
    let mean = sum / n as f64;
    eprintln!("{what}: teacher-forced mean |d|/rms {mean:.5}, worst {worst:.5}, top-1 {top1}/{n}");
    assert!(
        mean < 0.10 && top1 * 10 >= n * 9,
        "{what}: drifted from the serial lane - mean |d|/rms {mean:.4}, worst {worst:.4}, \
         top-1 {top1}/{n}"
    );
}

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
    // ...else the wikitext ids checked in beside the battery scripts, so this
    // file stops skipping itself on every box without the battery dump.
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

#[test]
fn gguf_config_matches_the_nvfp4_geometry() {
    let Some(path) = gguf_path() else { return };
    let map = MappedGguf::open(&path).expect("mmap");
    let c = NemotronConfig::from_gguf(map.gguf()).expect("from_gguf");
    assert_eq!((c.hidden, c.n_layer, c.vocab), (2688, 52, 131072));
    assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim), (32, 2, 128));
    assert_eq!(
        (c.d_inner(), c.conv_dim(), c.in_proj_rows()),
        (4096, 6144, 10304)
    );
    assert_eq!(
        (c.n_expert, c.n_active, c.moe_ff, c.shared_ff),
        (128, 6, 1856, 3712)
    );
    assert!((c.routed_scale - 2.5).abs() < 1e-6);
    let attn: Vec<usize> = c
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == NemotronBlock::Attention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(attn, vec![5, 12, 19, 26, 33, 42]);
    assert_eq!(
        c.blocks
            .iter()
            .filter(|b| **b == NemotronBlock::Mamba)
            .count(),
        23
    );
    assert_eq!(
        c.blocks
            .iter()
            .filter(|b| **b == NemotronBlock::Moe)
            .count(),
        23
    );
    // the eos SET: the gguf stamps 11 (<|im_end|>), the vocab scan adds
    // 2 (</s>) - both stop decode, same as the HF generation_config
    assert_eq!(c.eos_ids, vec![2, 11]);
    assert_eq!(c.bos_id, 1);
}

#[test]
fn gguf_bulk_prefill_matches_serial() {
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2() || !exec.has_q8_0_moe_relu2() {
        common::missing("pack lacks the q8 nemotron kernel set (stale .so?)");
        return;
    }
    let Some(path) = gguf_path() else { return };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);

    // ---- serial reference walk + greedy continuation ----------------------
    model.reset();
    let mut logits_s = Vec::new();
    for &t in &prompt {
        logits_s = model.forward(t).expect("serial forward");
    }
    let mut ids_s = Vec::with_capacity(GREEDY_STEPS);
    let mut ser_steps = vec![logits_s.clone()];
    let mut l = logits_s.clone();
    for _ in 0..GREEDY_STEPS {
        let tok = argmax(&l);
        ids_s.push(tok);
        l = model.forward(tok).expect("serial decode");
        ser_steps.push(l.clone());
    }

    // ---- bulk prefill + the same greedy continuation ----------------------
    model.reset();
    let logits_b = model.forward_prefill_stream(&prompt).expect("bulk prefill");
    let mut bulk_steps = vec![logits_b.clone()];
    for &tok in &ids_s {
        // teacher-forced: the serial lane's token, so the two walks cannot
        // fork at a near-tie and then compare different continuations
        bulk_steps.push(model.forward(tok).expect("bulk-side decode"));
    }

    // same near-exact class as the NVFP4 lane's bulk-vs-serial gate: the
    // prefill runs the mmq int8 GEMM ladder where decode runs the repacked
    // GEMV - identical products, regrouped summation
    assert_eq!(
        argmax(&logits_s),
        argmax(&logits_b),
        "prompt-boundary top-1 disagrees between serial and bulk"
    );
    teacher_forced(&ser_steps, &bulk_steps, "bulk prefill vs serial");

    let rms = (logits_s
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / logits_s.len() as f64)
        .sqrt();
    let mean_ad = logits_s
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / logits_s.len() as f64;
    eprintln!(
        "gguf bulk-vs-serial: mean|d| {mean_ad:.4} rms {rms:.4} ratio {:.4}",
        mean_ad / rms
    );
    assert!(
        mean_ad / rms < 0.10,
        "logit drift {mean_ad} vs rms {rms} out of band"
    );
    eprintln!("greedy ids: {ids_s:?}");
}

/// Stage-B gate: the Q8 batch lane (paged KV + arenas + the Q8 dispatch in
/// the shared layer walk) must reproduce the serial spine - slot prefill +
/// r=1 graph decode in the serial decode's numeric class (repacked GEMVs +
/// the token-batched relu2 pair), the family's standard boundary band, and
/// an identical greedy continuation. Plus a coalesced wave + an r=3 tick
/// smoke (rows must not replay: the mamba advance is not idempotent).
#[test]
fn gguf_batch_lane_matches_serial() {
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2()
        || !exec.has_mamba2_batch()
        || !exec.has_paged_kv()
        || !exec.has_q8_0_moe_relu2()
    {
        common::missing("pack lacks the q8 nemotron batch kernel set (stale .so?)");
        return;
    }
    let Some(path) = gguf_path() else { return };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);

    // serial spine
    model.reset();
    let mut logits_s = Vec::new();
    for &t in &prompt {
        logits_s = model.forward(t).expect("serial forward");
    }
    let mut ids_s = Vec::with_capacity(GREEDY_STEPS);
    let mut ser_steps = vec![logits_s.clone()];
    let mut l = logits_s.clone();
    for _ in 0..GREEDY_STEPS {
        let tok = argmax(&l);
        ids_s.push(tok);
        l = model.forward(tok).expect("serial decode");
        ser_steps.push(l.clone());
    }

    // batch lane
    let slots = model.batch_enable_probe(4).expect("enable_batch");
    assert_eq!(
        slots, 4,
        "the 32.6 GiB Q8 model must still seat 4 slots at ctx 4096"
    );
    let logits_b = model.forward_prefill(0, &prompt).expect("batch prefill");
    let mut bat_steps = vec![logits_b.clone()];
    for (i, &tok) in ids_s.iter().enumerate() {
        // teacher-forced on the serial lane's ids, as above
        bat_steps.push(
            model
                .forward_batch(&[tok], &[(PROMPT_LEN + i) as u32])
                .expect("batch decode"),
        );
    }
    // the last step's logits: the r=3 tick below continues from here
    let l = bat_steps.last().expect("decoded").clone();

    let rms = (logits_s
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / logits_s.len() as f64)
        .sqrt();
    let mean_ad = logits_s
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / logits_s.len() as f64;
    eprintln!(
        "gguf batch-vs-serial: mean|d| {mean_ad:.4} rms {rms:.4} ratio {:.4}",
        mean_ad / rms
    );
    eprintln!("greedy serial (the stream fed to both lanes): {ids_s:?}");
    assert_eq!(
        argmax(&logits_s),
        argmax(&logits_b),
        "boundary top-1 flipped"
    );
    assert!(mean_ad / rms.max(1e-3) < 0.10, "boundary drift out of band");
    teacher_forced(&ser_steps, &bat_steps, "batch lane vs serial");

    // coalesced wave: same prompt in slot 1 + two shorter ones
    let items = vec![
        (1usize, prompt.clone()),
        (2usize, prompt[..97].to_vec()),
        (3usize, prompt[..333].to_vec()),
    ];
    let out = model.forward_prefill_batch(&items).expect("coalesced wave");
    assert_eq!(out.len(), 3);
    assert_eq!(
        argmax(&out[0]),
        argmax(&bat_steps[0]),
        "same prompt through the coalesced wave flipped its boundary pick"
    );

    // r=3 decode tick: only next unconsumed tokens per slot
    let toks3 = [argmax(&l), argmax(&out[0]), argmax(&out[1])];
    let pos3 = [(PROMPT_LEN + GREEDY_STEPS) as u32, prompt.len() as u32, 97];
    let logits3 = model.forward_batch(&toks3, &pos3).expect("r=3 decode");
    let vocab = logits3.len() / 3;
    assert_eq!(vocab * 3, logits3.len());
    assert!(
        logits3.iter().all(|v| v.is_finite()),
        "r=3 tick produced non-finite logits"
    );
}

/// Stage-B gate 2: the radix prefix cache + mamba state snapshots serve the
/// Q8 class through the same machinery (checkpoints are f32 arena blobs -
/// class-agnostic - but the resumed-tail walk runs the Q8 dispatch, so the
/// structural gate is the greedy-stream equality after an exact-repeat
/// resume). Plus the fp8-KV flip smoke: the lane rebuilds at the new byte
/// width, boundary top-1 holds, greedy-8 stays in the kv8 band, and the kv
/// accounting shrinks.
#[test]
fn gguf_prefix_resume_and_fp8_kv_smoke() {
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2()
        || !exec.has_mamba2_batch()
        || !exec.has_paged_kv()
        || !exec.has_q8_0_moe_relu2()
    {
        common::missing("pack lacks the q8 nemotron batch kernel set (stale .so?)");
        return;
    }
    let Some(path) = gguf_path() else { return };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    let greedy8 = |model: &mut GpuNemotron, first: &[f32], pos0: usize| {
        let mut ids = Vec::new();
        let mut tok = argmax(first);
        for i in 0..8 {
            ids.push(tok);
            let l = model
                .forward_batch(&[tok], &[(pos0 + i) as u32])
                .expect("decode tick");
            tok = argmax(&l);
        }
        ids
    };

    // cold prefill (slot 0) plants checkpoints at the last two page cuts
    let cold = model.forward_prefill(0, &prompt).expect("cold prefill");
    assert_eq!(model.take_prefill_reused(0), 0, "first sight cannot reuse");
    let cold_ids = greedy8(&mut model, &cold, PROMPT_LEN);

    // exact repeat (slot 1) must resume at the deep checkpoint
    let warm = model.forward_prefill(1, &prompt).expect("warm prefill");
    let reused = model.take_prefill_reused(1);
    assert_eq!(
        reused, 688,
        "exact repeat must resume at the deep checkpoint"
    );
    // The in-file MTP drafts from a resumed prompt too: its cells ride the
    // pages slot 1 just adopted from slot 0, so the drafter is warm through
    // the prompt end. With a dense per-slot cache it went cold at every
    // resume - every agent turn after the first drafted nothing. (Slot 0
    // has decoded past its prompt by now, so it is not asked here.)
    assert!(
        model
            .spec_ensure_warm(1, &[], (PROMPT_LEN - 1) as u32)
            .expect("warm probe"),
        "MTP drafter cold after a prefix resume"
    );
    assert_eq!(
        argmax(&cold),
        argmax(&warm),
        "resume flipped the boundary pick"
    );
    let rms =
        (cold.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / cold.len() as f64).sqrt();
    let mean_ad = cold
        .iter()
        .zip(&warm)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / cold.len() as f64;
    eprintln!("q8 resume boundary: mean|d| {mean_ad:.4} vs rms {rms:.3}");
    assert!(
        mean_ad / rms.max(1e-3) < 0.10,
        "resumed logits out of the reorder band"
    );
    // structural no-corruption gate: slot 1's greedy stream from the resumed
    // state must equal slot 0's cold stream. Decode both on slot 1's cursor.
    // (forward_batch with identity rows decodes slot 0 - re-run the cold
    // stream shape on slot 1 via mixed rows instead)
    let warm_ids = {
        let mut ids = Vec::new();
        let mut tok = argmax(&warm);
        for i in 0..8 {
            ids.push(tok);
            let (step, _) = model
                .forward_mixed_sampled(
                    &[(1usize, tok, (PROMPT_LEN + i) as u32)],
                    usize::MAX,
                    &[paddock_engine::generator::RowSample::Device(
                        paddock_engine::sampler::DevicePlan::Greedy,
                    )],
                    &[],
                )
                .expect("slot1 decode tick");
            tok = step.ids[0];
        }
        ids
    };
    assert_eq!(cold_ids, warm_ids, "greedy stream diverged after resume");

    // fp8-KV flip: rebuild the lane at the lossy KV class
    let l16 = cold;
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp8E4m3);
    assert_eq!(model.batch_enable_probe(4).expect("enable fp8"), 4);
    let l8 = model.forward_prefill(0, &prompt).expect("fp8 prefill");
    let kv8 = model.kv_mem_bytes().expect("fp8 kv accounting");
    let ids8 = greedy8(&mut model, &l8, PROMPT_LEN);
    let mean_ad8 = l16
        .iter()
        .zip(&l8)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / l16.len() as f64;
    let same = cold_ids.iter().zip(&ids8).filter(|(a, b)| a == b).count();
    eprintln!(
        "q8+kv8: boundary mean|d| {mean_ad8:.4} ({:.2}% of rms), greedy8 {same}/8, kv {kv8} B",
        100.0 * mean_ad8 / rms
    );
    assert_eq!(argmax(&l16), argmax(&l8), "kv8 flipped the boundary top-1");
    // kv8 is a LOSSY class - the band is the NVFP4 lane's measured envelope
    assert!(
        mean_ad8 / rms < 0.12,
        "kv8 drift out of the measured class band"
    );
    assert!(same >= 6, "kv8 greedy-8 diverged structurally ({same}/8)");
}

/// Spec-core gate (C1): the trunk verify round must (a) produce
/// the exact greedy continuation as picks when fed true drafts, (b) reject
/// wrong drafts at the right position, and - the sharp part - (c) leave the
/// mamba state/conv windows exactly at the accepted row after a partial
/// accept, proven by the next round's picks still matching the no-spec
/// greedy stream. Runs on the GGUF lane (the machinery is class-shared).
#[test]
fn gguf_spec_verify_round_rolls_back_state() {
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2()
        || !exec.has_mamba2_batch()
        || !exec.has_paged_kv()
        || !exec.has_q8_0_moe_relu2()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
    {
        common::missing("pack lacks the spec verify kernel set (stale .so?)");
        return;
    }
    let Some(path) = gguf_path() else { return };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    // no-spec baseline stream on slot 0: b0..b9
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    let mut b = vec![argmax(&l0)];
    for i in 0..9 {
        let l = model
            .forward_batch(&[b[i]], &[(PROMPT_LEN + i) as u32])
            .expect("decode");
        b.push(argmax(&l));
    }

    // fresh same-state slots via the (already-gated) prefix resume
    let l1 = model.forward_prefill(1, &prompt).expect("prefill 1");
    assert_eq!(argmax(&l1), b[0]);
    let l2 = model.forward_prefill(2, &prompt).expect("prefill 2");
    assert_eq!(argmax(&l2), b[0]);

    // round 1, two ragged reqs in one verify: slot 1 all-true drafts (full
    // accept), slot 2 a wrong first draft (partial accept + rollback)
    let wrong = |t: u32| if t == 0 { 1 } else { t - 1 };
    let reqs = vec![
        (1usize, PROMPT_LEN, vec![b[0], b[1], b[2], b[3]]),
        (2usize, PROMPT_LEN, vec![b[0], wrong(b[1]), wrong(b[2])]),
    ];
    let picks = model
        .forward_spec_batch(&reqs)
        .expect("verify round")
        .expect("engaged");
    assert_eq!(
        &picks[0..4],
        &[b[1], b[2], b[3], b[4]],
        "slot 1 picks off the true stream"
    );
    assert_eq!(
        picks[4], b[1],
        "slot 2 pending row must still pick the true next token"
    );

    // round 2 on slot 1 (state at P+4 after the full accept): a wrong draft
    let reqs = vec![(1usize, PROMPT_LEN + 4, vec![b[4], wrong(b[5]), wrong(b[6])])];
    let picks = model
        .forward_spec_batch(&reqs)
        .expect("round 2")
        .expect("engaged");
    assert_eq!(
        picks[0], b[5],
        "round-2 pending pick validates the round-1 state advance"
    );

    // round 3 on slot 1 (rollback happened after round 2's reject): true
    // drafts again - if the rollback missed, these picks diverge
    let reqs = vec![(1usize, PROMPT_LEN + 5, vec![b[5], b[6], b[7]])];
    let picks = model
        .forward_spec_batch(&reqs)
        .expect("round 3")
        .expect("engaged");
    assert_eq!(
        &picks[0..2],
        &[b[6], b[7]],
        "post-rollback picks diverged from the no-spec stream"
    );

    // slot 2 continues after its partial accept (state must sit at P+1)
    let (step, _) = model
        .forward_mixed_sampled(
            &[(2usize, b[1], (PROMPT_LEN + 1) as u32)],
            usize::MAX,
            &[paddock_engine::generator::RowSample::Device(
                paddock_engine::sampler::DevicePlan::Greedy,
            )],
            &[],
        )
        .expect("slot 2 decode");
    assert_eq!(step.ids[0], b[2], "slot 2 diverged after its rollback");
}

/// Serve-cadence spec gate: a live spec-on serve forked its greedy stream
/// vs no-spec on nearly every prompt tried - every fork a
/// verify-produced pick (accepted draft or bonus), clustered 1-3 rounds
/// after partial accepts, with chunk lengths varying round to round from
/// the k_now controller - while the short fixed-k loops above stay exact.
/// This replays that cadence in-engine: 200 committed tokens, chunk length
/// cycling, draft quality cycling full/none/half accepted, the service's
/// exact commit walk, stream equality asserted at every round.
#[test]
fn gguf_spec_serve_cadence_matches_greedy() {
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2()
        || !exec.has_mamba2_batch()
        || !exec.has_paged_kv()
        || !exec.has_q8_0_moe_relu2()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
    {
        common::missing("pack lacks the spec verify kernel set (stale .so?)");
        return;
    }
    let Some(path) = gguf_path() else { return };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    const N: usize = 200;
    // no-spec stream on slot 0
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    let mut b = vec![argmax(&l0)];
    let mut margin = vec![top2_margin(&l0)];
    for i in 0..N + 16 {
        let l = model
            .forward_batch(&[b[i]], &[(PROMPT_LEN + i) as u32])
            .expect("decode");
        b.push(argmax(&l));
        margin.push(top2_margin(&l));
    }

    // spec slot: same resume class the rollback gate already certifies
    let l1 = model.forward_prefill(1, &prompt).expect("prefill 1");
    assert_eq!(argmax(&l1), b[0], "slot 1 boundary off before any spec ran");

    let wrong = |t: u32| if t == 0 { 1 } else { t - 1 };
    // serve-observed chunk lengths were 2..=8 => 1..=7 drafts per round
    let ks = [3usize, 6, 1, 4, 7, 2, 5, 1, 7, 3];
    let mut committed: Vec<u32> = Vec::new();
    let mut pending = b[0];
    let mut pos = PROMPT_LEN;
    let mut round = 0usize;
    /// A tie in the first rounds means the gate proved nothing.
    const MIN_ROUNDS: usize = 5;
    let mut parted: Option<(usize, usize, f32)> = None;
    while committed.len() < N {
        let k = ks[round % ks.len()];
        let mut chunk = vec![pending];
        for j in 0..k {
            let t = b[committed.len() + 1 + j];
            // cycle the acceptance class: full accept / all rejected (the
            // serve's frequent acc=1 rounds) / mid partial
            chunk.push(match round % 3 {
                0 => t,
                1 => wrong(t),
                _ if j < k / 2 => t,
                _ => wrong(t),
            });
        }
        let picks = model
            .forward_spec_batch(&[(1usize, pos, chunk.clone())])
            .expect("verify round")
            .expect("engaged");
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        committed.extend_from_slice(&chunk[..=a]);
        let cls = round % 3;
        let acc = a + 1;
        // Where the reference's own top-2 are this close, the width
        // difference between a 1-row decode and a (k+1)-row verify is enough
        // to swap them, and the reference stream stops being the answer from
        // there on. Anything wider than this is a real divergence.
        let tie = |i: usize| margin.get(i).copied().unwrap_or(f32::NAN) <= TIE_MARGIN;
        if committed[..] != b[..committed.len()] {
            let i = committed
                .iter()
                .zip(&b)
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            assert!(
                tie(i),
                "round {round}: spec stream diverged (k={k} class={cls} acc={acc},                  reference margin {:.3e} at position {i} - too wide for a tie)",
                margin.get(i).copied().unwrap_or(f32::NAN)
            );
            parted = Some((round, i, margin[i]));
            break;
        }
        if picks[a] != b[committed.len()] {
            let i = committed.len();
            assert!(
                tie(i),
                "round {round}: bonus pick off-stream (k={k} class={cls} acc={acc},                  reference margin {:.3e} - too wide for a tie)",
                margin.get(i).copied().unwrap_or(f32::NAN)
            );
            parted = Some((round, i, margin[i]));
            break;
        }
        pending = picks[a];
        pos += a + 1;
        round += 1;
    }
    match parted {
        None => eprintln!(
            "serve-cadence spec loop: {} committed over {round} rounds, all on-stream",
            committed.len()
        ),
        Some((r, i, m)) => {
            let committed_len = committed.len();
            eprintln!(
                "serve-cadence spec loop: {committed_len} committed over {r} rounds on-stream, then the \
                 reference's own top-2 tied at position {i} (margin {m:.3e}) and the streams \
                 parted - verification stops there"
            );
            assert!(
                r >= MIN_ROUNDS,
                "the streams parted at round {r}, before {MIN_ROUNDS} rounds proved anything \
                 (margin {m:.3e} at position {i})"
            );
        }
    }
}

/// C3 gate: the in-file MTP block (blk.52 nextn) drafts usefully. Drafts
/// are quality-only - the verify re-judges every token - so this is not a
/// byte gate; the discriminator is the ACCEPTANCE RATE. A correct block
/// (vLLM's nemotron_h_mtp.py order, RMSNorm end norm, the (token_i, h_{i-1})
/// pairing) accepts a large fraction of its drafts against the target's own
/// greedy walk; a norm/shift/pairing error collapses it to noise (~1/vocab
/// per position, i.e. ~0). Floor 0.25 sits far above noise and safely below
/// the pooled acceptance a healthy drafter reaches on this target. The
/// stream-equality assert doubles as proof the MTP walk hooks never perturb
/// the trunk (same in-engine regime the serve-cadence gate certifies).
#[test]
fn gguf_mtp_drafts_accept() {
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2()
        || !exec.has_mamba2_batch()
        || !exec.has_paged_kv()
        || !exec.has_q8_0_moe_relu2()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
    {
        common::missing("pack lacks the spec verify kernel set (stale .so?)");
        return;
    }
    // Spec is off by default on dies under 128 SMs, and the loader installs
    // that as PROCESS env the first time any test here loads a model - so on
    // such a die this gate, whose whole subject is the drafter, skipped
    // itself on a side effect of whichever test ran before it. State the
    // operator's choice instead, which is the same door the default's own
    // comment points at (`--spec on` / PADDOCK_SPEC beats the default).
    // Honour a human's explicit PADDOCK_NO_SPEC only when nothing has loaded
    // yet, i.e. when the loader cannot have been the one to set it.
    if std::env::var_os("PADDOCK_NO_SPEC").is_some() && std::env::var_os("PADDOCK_SPEC").is_none() {
        unsafe { std::env::remove_var("PADDOCK_NO_SPEC") };
    }
    unsafe { std::env::set_var("PADDOCK_SPEC", "on") };
    let Some(path) = gguf_path() else { return };
    let Some(prompt) = oracle_prompt(PROMPT_LEN) else {
        common::missing("no oracle dump for prompt ids");
        return;
    };
    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    assert!(
        model.spec_capable(),
        "in-file MTP must make the GGUF lane spec-capable"
    );

    // Spec slot first: the full prefill walk warms the MTP (coverage + h
    // chain) via the rows_pass_body hook. Order matters - a radix-hit
    // resume adopts KV without walking the prefix, and a drafter cannot
    // warm rows it never saw (the DFlash stance; prefix-hit requests serve
    // dense), so the warm slot must be the one that walks cold.
    const ROUNDS: usize = 25;
    const K: usize = 7;
    let l1 = model.forward_prefill(1, &prompt).expect("prefill 1");

    // no-spec stream on slot 0 (radix hit is fine here - the reference
    // needs state exactness, not drafter warmth; the resume gate certifies
    // that)
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    assert_eq!(
        argmax(&l1),
        argmax(&l0),
        "slot boundary picks disagree before any spec ran"
    );
    let mut b = vec![argmax(&l0)];
    let mut margin = vec![top2_margin(&l0)];
    for i in 0..ROUNDS * (K + 1) + 2 {
        let l = model
            .forward_batch(&[b[i]], &[(PROMPT_LEN + i) as u32])
            .expect("decode");
        b.push(argmax(&l));
        margin.push(top2_margin(&l));
    }

    let mut committed: Vec<u32> = Vec::new();
    let mut pending = b[0];
    let mut pos = PROMPT_LEN;
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let mut parted: Option<usize> = None;
    for round in 0..ROUNDS {
        let drafts = model
            .spec_draft_batch(&[(1usize, pending)], K)
            .expect("draft")
            .expect("mtp drafter engaged");
        let d = &drafts[0];
        assert_eq!(d.len(), K, "round {round}: MTP declined (cold coverage?)");
        let mut chunk = vec![pending];
        chunk.extend_from_slice(d);
        let picks = model
            .forward_spec_batch(&[(1usize, pos, chunk.clone())])
            .expect("verify round")
            .expect("engaged");
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        drafted += K;
        accepted += a;
        committed.extend_from_slice(&chunk[..=a]);
        // Acceptance is this gate's subject and does not depend on the
        // reference, so a tie ends the byte comparison but not the rounds.
        if parted.is_none() {
            same_until_tie(
                &b[..committed.len()],
                &committed,
                &margin,
                8,
                &format!("round {round}: spec stream (acc={a}/{K})"),
            );
            if committed[..] != b[..committed.len()] {
                parted = Some(round);
            }
        }
        pending = picks[a];
        pos += a + 1;
    }
    let rate = accepted as f64 / drafted as f64;
    eprintln!(
        "MTP acceptance: {accepted}/{drafted} = {rate:.3} over {ROUNDS} rounds ({} committed)",
        committed.len()
    );
    assert!(
        rate >= 0.25,
        "MTP acceptance {rate:.3} below the 0.25 floor - block math suspect"
    );
}

/// Differential replay of a live serve's spec rounds. Feed it the
/// `[spec2a-ids]` dump (PADDOCK_SPEC_DEBUG_IDS=1 serve) via PADDOCK_REPLAY:
/// it prefills the dumped prompt, computes the no-spec greedy stream, then
/// replays every dumped chunk through forward_spec_batch - phase A bare,
/// phase B with the serve's interleaved drafter call before each round -
/// and reports the first round where the engine's picks differ from the
/// serve's (serve-only state drift) or from the no-spec stream (engine
/// verify defect). Diagnostic: prints findings, never asserts stream
/// equality itself.
#[test]
fn gguf_spec_replay_dump() {
    let Some(file) = std::env::var_os("PADDOCK_REPLAY") else {
        eprintln!("skip: PADDOCK_REPLAY unset");
        return;
    };
    let Some((_gpu_lock, exec)) = exec_locked() else {
        return;
    };
    if !exec.has_mamba2()
        || !exec.has_mamba2_batch()
        || !exec.has_paged_kv()
        || !exec.has_q8_0_moe_relu2()
        || !exec.has_spec_verify_mamba()
        || !exec.has_argmax_rows()
    {
        common::missing("pack lacks the spec verify kernel set (stale .so?)");
        return;
    }
    let Some(path) = gguf_path() else { return };

    fn ids(s: &str) -> Vec<u32> {
        s.trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect()
    }
    let raw = std::fs::read_to_string(&file).expect("replay file");
    let mut prompt: Vec<u32> = Vec::new();
    let mut rounds: Vec<(usize, Vec<u32>, Vec<u32>)> = Vec::new();
    for line in raw.lines() {
        if let Some(rest) = line.split_once("prompt=").map(|(_, r)| r) {
            prompt = ids(rest);
        } else if let Some((head, picks)) = line.split_once(" picks=") {
            let pos: usize = head
                .split_once("pos=")
                .and_then(|(_, r)| r.split_whitespace().next())
                .and_then(|v| v.parse().ok())
                .expect("pos");
            let chunk = head
                .split_once("chunk=")
                .map(|(_, r)| ids(r))
                .expect("chunk");
            rounds.push((pos, chunk, ids(picks)));
        }
    }
    assert!(
        !prompt.is_empty() && !rounds.is_empty(),
        "replay file parsed empty"
    );
    let plen = prompt.len();
    eprintln!("replay: prompt {plen} toks, {} rounds", rounds.len());

    let map = MappedGguf::open(&path).expect("mmap");
    let mut model = GpuNemotron::load(exec, &map, 4096).expect("load gguf");
    // the exact class this test is stated in (the KV8 default is lossy)
    model.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    drop(map);
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);

    // no-spec greedy stream (slot 0)
    let n_stream: usize = rounds.iter().map(|(_, c, _)| c.len()).sum::<usize>() + 8;
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    let mut b = vec![argmax(&l0)];
    for i in 0..n_stream {
        let l = model
            .forward_batch(&[b[i]], &[(plen + i) as u32])
            .expect("decode");
        b.push(argmax(&l));
    }

    // the serve's committed stream, re-derived from the dump
    let mut serve_stream = vec![rounds[0].1[0]];
    for (_, chunk, picks) in &rounds {
        let mut a = 0usize;
        while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
            a += 1;
        }
        serve_stream.extend_from_slice(&picks[..=a]);
    }
    match serve_stream.iter().zip(&b).position(|(s, e)| s != e) {
        Some(i) => eprintln!(
            "serve stream FORKS no-spec at tok {i}: serve={} nospec={}",
            serve_stream[i], b[i]
        ),
        None => eprintln!(
            "serve stream == no-spec stream for {} toks",
            serve_stream.len()
        ),
    }

    // phase A: bare replay (slot 1); phase B: drafter interleaved (slot 2)
    let df_dir = std::env::var("NEMOTRON_DFLASH_DIR")
        .unwrap_or_else(|_| "/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash".into());
    let has_drafter = std::path::Path::new(&df_dir)
        .join("model.safetensors")
        .exists();
    for phase in ["bare", "drafted"] {
        if phase == "drafted" {
            if !has_drafter {
                eprintln!("phase B skipped: no drafter at {df_dir}");
                continue;
            }
            model
                .attach_dflash(std::path::Path::new(&df_dir))
                .expect("attach dflash");
        }
        let slot = if phase == "bare" { 1usize } else { 2 };
        let l = model
            .forward_prefill(slot, &prompt)
            .expect("replay prefill");
        if argmax(&l) != rounds[0].1[0] {
            eprintln!(
                "[{phase}] boundary pick {} != serve's {}",
                argmax(&l),
                rounds[0].1[0]
            );
        }
        let mut clean = true;
        for (ri, (pos, chunk, srv_picks)) in rounds.iter().enumerate() {
            if phase == "drafted" {
                // the serve drafts k_budget=7 before every round, then
                // truncates; replay the call, discard the drafts
                let _ = model
                    .spec_draft_batch(&[(slot, chunk[0])], 7)
                    .expect("draft");
            }
            let picks = model
                .forward_spec_batch(&[(slot, *pos, chunk.clone())])
                .expect("verify")
                .expect("engaged");
            if &picks != srv_picks {
                let d = picks
                    .iter()
                    .zip(srv_picks)
                    .position(|(a, c)| a != c)
                    .unwrap();
                eprintln!(
                    "[{phase}] round {ri} (pos {pos}) DIVERGES from serve at row {d}: engine={:?} serve={:?} chunk={:?}",
                    picks, srv_picks, chunk
                );
                clean = false;
                break;
            }
        }
        if clean {
            eprintln!(
                "[{phase}] replay matches the serve's picks on all {} rounds",
                rounds.len()
            );
        }
    }

    // ── bisect: which round corrupts the state? ────────────────────────────
    // Rebuild ground-truth r=1 state at round r (prefill + decode the
    // committed prefix) and replay rounds r..=R; the tok-F flip vanishing
    // means the corrupting round left the suffix. Requires the first fork.
    if std::env::var_os("PADDOCK_REPLAY_BISECT").is_none() {
        return;
    }
    let Some(fork_tok) = serve_stream.iter().zip(&b).position(|(s, e)| s != e) else {
        eprintln!("bisect: no fork - nothing to do");
        return;
    };
    // accepted count per round + the round containing the fork token
    let accs: Vec<usize> = rounds
        .iter()
        .map(|(_, chunk, picks)| {
            let mut a = 0usize;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[a] {
                a += 1;
            }
            a + 1
        })
        .collect();
    let mut cum = 1usize; // stream tok 0 = the prefill pick
    let mut fork_round = 0usize;
    let mut fork_row = 0usize;
    for (ri, &a) in accs.iter().enumerate() {
        if fork_tok < cum + a {
            fork_round = ri;
            fork_row = fork_tok - cum;
            break;
        }
        cum += a;
    }
    let want = b[fork_tok];
    let got = serve_stream[fork_tok];
    eprintln!(
        "bisect: fork tok {fork_tok} lives in round {fork_round} row {fork_row} (want {want}, serve {got})"
    );

    // replay rounds r..=fork_round from a fresh r=1 state; true = flip present
    let probe = |r: usize, model: &mut GpuNemotron| -> bool {
        let n_r = 1 + accs[..r].iter().sum::<usize>();
        let l = model.forward_prefill(3, &prompt).expect("probe prefill");
        assert_eq!(argmax(&l), serve_stream[0]);
        for (j, &t) in serve_stream[..n_r - 1].iter().enumerate() {
            let _ = model
                .forward_mixed_sampled(
                    &[(3usize, t, (plen + j) as u32)],
                    usize::MAX,
                    &[paddock_engine::generator::RowSample::Device(
                        paddock_engine::sampler::DevicePlan::Greedy,
                    )],
                    &[],
                )
                .expect("probe decode");
        }
        assert_eq!(
            rounds[r].0,
            plen + n_r - 1,
            "probe pos misaligned at round {r}"
        );
        for (pos, chunk, _) in &rounds[r..=fork_round] {
            let picks = model
                .forward_spec_batch(&[(3usize, *pos, chunk.clone())])
                .expect("probe verify")
                .expect("engaged");
            if *pos == rounds[fork_round].0 {
                return picks[fork_row] == got;
            }
        }
        unreachable!("fork round not reached");
    };

    assert!(
        probe(0, &mut model),
        "bisect precondition: full suffix must reproduce the flip"
    );
    let (mut lo, mut hi) = (0usize, fork_round); // flip(lo)=true; hi untested
    if probe(fork_round, &mut model) {
        eprintln!(
            "bisect: the flip survives a FRESH state + round {fork_round} alone - not prior-state corruption; it is round {fork_round}'s own verify walk. chunk={:?}",
            rounds[fork_round].1
        );
        return;
    }
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if probe(mid, &mut model) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    eprintln!(
        "bisect: corruption onset = round {lo} (last suffix start that still flips): pos={} chunk={:?} picks={:?} acc={}",
        rounds[lo].0, rounds[lo].1, rounds[lo].2, accs[lo]
    );
    eprintln!(
        "        first clean suffix start = round {hi}: pos={} chunk={:?} acc={}",
        rounds[hi].0, rounds[hi].1, accs[hi]
    );

    // ── state diff at the corrupting round ─────────────────────────────────
    // Advance through round `lo` three ways and compare the mamba arenas:
    //   A = verify round (the corrupting path)
    //   B = r=1 decode of the same committed tokens (ground truth)
    //   C = one bulk prefill over [prompt ∥ committed] (the r>1 class twin)
    // Structural corruption shows as O(1) relative error on some layer;
    // benign kernel-class noise sits orders of magnitude lower.
    let n_lo = 1 + accs[..lo].iter().sum::<usize>();
    let toks_after: Vec<u32> = serve_stream[..n_lo - 1 + accs[lo]].to_vec();
    let rel = |x: &[f32], y: &[f32]| -> f64 {
        let mut d2 = 0f64;
        let mut n2 = 0f64;
        for (a, b) in x.iter().zip(y) {
            d2 += ((*a - *b) as f64).powi(2);
            n2 += (*b as f64).powi(2);
        }
        (d2 / n2.max(1e-30)).sqrt()
    };
    // B first (ground truth)
    let l = model.forward_prefill(3, &prompt).expect("B prefill");
    assert_eq!(argmax(&l), serve_stream[0]);
    for (j, &t) in toks_after.iter().enumerate() {
        let _ = model
            .forward_mixed_sampled(
                &[(3usize, t, (plen + j) as u32)],
                usize::MAX,
                &[paddock_engine::generator::RowSample::Device(
                    paddock_engine::sampler::DevicePlan::Greedy,
                )],
                &[],
            )
            .expect("B decode");
    }
    let sb = model.state_dump_probe(3);
    // A: replay up to and through round lo on a fresh slot
    let _ = model.forward_prefill(3, &prompt).expect("A prefill");
    for (pos, chunk, _) in &rounds[..=lo] {
        let _ = model
            .forward_spec_batch(&[(3usize, *pos, chunk.clone())])
            .expect("A verify")
            .expect("engaged");
    }
    let sa = model.state_dump_probe(3);
    // C: bulk prefill over the concatenation
    let mut cat = prompt.clone();
    cat.extend_from_slice(&toks_after);
    let _ = model.forward_prefill(3, &cat).expect("C prefill");
    let sc_ = model.state_dump_probe(3);
    eprintln!("state diff after round {lo} (rel-L2 vs the r=1 ground truth B):");
    for ((li, ssa, swa), ((_, ssb, swb), (_, ssc, swc))) in sa.iter().zip(sb.iter().zip(sc_.iter()))
    {
        let (dsa, dwa) = (rel(ssa, ssb), rel(swa, swb));
        let (dsc, dwc) = (rel(ssc, ssb), rel(swc, swb));
        if dsa > 1e-4 || dwa > 1e-4 || dsc > 1e-4 || dwc > 1e-4 {
            eprintln!(
                "  layer {li:2}: A(ssm {dsa:.3e} win {dwa:.3e})  C(ssm {dsc:.3e} win {dwc:.3e})"
            );
        }
    }
}
