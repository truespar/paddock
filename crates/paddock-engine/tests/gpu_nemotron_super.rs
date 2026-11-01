//! Nemotron 3.5 Super VL 120B-A12B NVFP4 (text) vs a vLLM reference dump
//! (the Super oracle script beside the Lightning one): the same weights, so the
//! gates are the Lightning lane's - per-layer stage sums as a BAND (two
//! engines, two residual classes; structure errors are orders of
//! magnitude), the prompt-boundary top-1 and the greedy ids sharp - plus
//! teacher-forced top-1 agreement over longer texts, the measure that
//! scales past one short prompt.
//!
//! Heavy (75 GB of weights): `PADDOCK_HEAVY_TESTS=1`. The oracle path comes
//! from `SUPER_VL_ORACLE` or sits beside the checkpoint as
//! `super-vl-oracle/super-vl-oracle.json`. KV is fp8 on both sides (the
//! checkpoint's own spec, what the dump ran).

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::nemotron::GpuNemotron;

const CKPT_ENV: &str = "SUPER_VL_DIR";
const CKPT_DIR: &str = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4";

fn setup() -> Option<(GpuNemotron, serde_json::Value)> {
    if !common::heavy() {
        return None;
    }
    let exec = common::gpu_arc()?;
    if !exec.has_mamba2() || !exec.has_nvf4_moe() || !exec.has_moe_topk_sigmoid_wide() {
        common::missing("pack lacks the Super kernels (mamba2, nvf4 moe, slot 903)");
        return None;
    }
    let dir = common::model_dir(CKPT_ENV, &[CKPT_DIR])?;
    let oracle_path = std::env::var("SUPER_VL_ORACLE").unwrap_or_else(|_| {
        dir.parent()
            .expect("models root")
            .join("super-vl-oracle/super-vl-oracle.json")
            .display()
            .to_string()
    });
    let Ok(raw) = std::fs::read(&oracle_path) else {
        common::missing(&format!("no oracle dump at {oracle_path}"));
        return None;
    };
    let oracle: serde_json::Value = serde_json::from_slice(&raw).expect("oracle json");
    let t0 = std::time::Instant::now();
    let model = GpuNemotron::load_dir(exec, &dir, 4096).expect("load");
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    Some((model, oracle))
}

fn ids(v: &serde_json::Value) -> Vec<u32> {
    v.as_array()
        .expect("id array")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
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

#[test]
fn super_vl_matches_the_vllm_reference() {
    let Some((mut model, oracle)) = setup() else {
        return;
    };
    let prompt_ids = ids(&oracle["prompt_ids"]);
    let greedy_ids = ids(&oracle["greedy_ids"]);
    let n_layer = oracle["n_layers"].as_u64().expect("n_layers") as usize;
    assert_eq!(n_layer, 88);

    // ---- the prompt, token by token, accumulating per-layer stage sums ----
    let mut sums = vec![0f64; n_layer];
    let mut abs_sums = vec![0f64; n_layer];
    let mut logits: Vec<f32> = Vec::new();
    for &tok in &prompt_ids {
        let mut stages: Vec<Vec<f32>> = Vec::with_capacity(n_layer);
        logits = model.forward_probed(tok, &mut stages).expect("forward");
        assert_eq!(stages.len(), n_layer);
        for (li, hs) in stages.iter().enumerate() {
            for &v in hs {
                sums[li] += v as f64;
                abs_sums[li] += (v as f64).abs();
            }
        }
    }
    let mut failed = 0usize;
    let mut worst = (0f64, 0usize);
    for li in 0..n_layer {
        let o = &oracle["stage_sums"][li.to_string()];
        let o_abs = o["abs_sum"].as_f64().expect("abs_sum");
        let o_sum = o["sum"].as_f64().expect("sum");
        let rel_abs = (abs_sums[li] - o_abs).abs() / o_abs.max(1e-3);
        let rel_sum = (sums[li] - o_sum).abs() / o_abs.max(1e-3);
        if rel_abs > worst.0 {
            worst = (rel_abs, li);
        }
        let out = rel_abs >= 0.15 || rel_sum >= 0.15;
        failed += usize::from(out);
        println!(
            "layer {li:2}: abs_sum {:>12.2} vs {o_abs:>12.2} (rel {rel_abs:.4})  sum {:>10.2} vs {o_sum:>10.2}{}",
            abs_sums[li],
            sums[li],
            if out { " <-- OUT OF BAND" } else { "" }
        );
    }
    println!(
        "stage sums: worst abs_sum rel {:.4} at layer {}",
        worst.0, worst.1
    );

    // ---- greedy decode against the reference ids ----------------------------
    let mut ours = Vec::with_capacity(greedy_ids.len());
    for step in 0..greedy_ids.len() {
        let tok = argmax(&logits);
        ours.push(tok);
        if step + 1 < greedy_ids.len() {
            logits = model.forward(tok).expect("decode step");
        }
    }
    let lead = ours
        .iter()
        .zip(&greedy_ids)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "greedy: first {lead}/{} agree\n  ours:   {ours:?}\n  oracle: {greedy_ids:?}",
        greedy_ids.len()
    );
    assert_eq!(failed, 0, "{failed} layers out of the 15% stage band");
    assert_eq!(ours[0], greedy_ids[0], "prompt-boundary top-1 differs");
    // the streams may part only on a near-tie in the reference: our pick
    // within TIE nats of its top-1 (the dump's own prompt has an exact tie
    // at step 3, ' quick' and ' capital' both at -1.539, which vLLM breaks
    // by index). Past the parting point they are different texts.
    if lead < greedy_ids.len() {
        const TIE: f64 = 0.25;
        let top5 = oracle["decode_steps"][lead]["top5"]
            .as_array()
            .expect("top5");
        let best = top5[0]["logprob"].as_f64().expect("logprob");
        let ours_lp = top5
            .iter()
            .find(|t| t["id"].as_u64() == Some(ours[lead] as u64))
            .and_then(|t| t["logprob"].as_f64());
        println!(
            "parted at step {lead}: ours {} at {ours_lp:?}, reference top-1 {} at {best:.3}",
            ours[lead], greedy_ids[lead]
        );
        assert!(
            ours_lp.is_some_and(|lp| best - lp <= TIE),
            "greedy parted at step {lead} on a pick {:.3} nats under the reference's top-1",
            ours_lp.map_or(f64::INFINITY, |lp| best - lp)
        );
    }
}

/// Teacher-forced top-1 over the dump's passages (prose, code, a chat
/// transcript): at every position, does our argmax match the reference's?
/// Two engines on the same weights disagree only on near-ties, so this
/// sits in the high 90s when the graph is right and collapses when it is not.
#[test]
fn super_vl_teacher_forced_top1() {
    let Some((mut model, oracle)) = setup() else {
        return;
    };
    let limit: usize = std::env::var("SUPER_VL_TF_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let forced = oracle["teacher_forced"]
        .as_object()
        .expect("teacher_forced");
    let (mut agree_all, mut total_all) = (0usize, 0usize);
    for (name, part) in forced {
        let toks = ids(&part["ids"]);
        let top5 = part["top5"].as_array().expect("top5");
        model.reset();
        let n = toks.len().min(limit);
        let (mut agree, mut total) = (0usize, 0usize);
        // at each disagreement: how far under the reference's top-1 our pick
        // sits in ITS distribution (None: outside its top-5) - near-ties are
        // two engines' rounding, a wide gap is a graph error
        let mut gaps: Vec<Option<f64>> = Vec::new();
        for (i, &tok) in toks.iter().enumerate().take(n) {
            let logits = model.forward(tok).expect("forward");
            // the reference's position i+1 holds its distribution over token
            // i+1 given tokens 0..=i - ours after feeding token i
            let Some(next) = top5.get(i + 1).and_then(|v| v.as_array()) else {
                continue;
            };
            let want = next[0]["id"].as_u64().expect("id") as u32;
            let got = argmax(&logits);
            total += 1;
            if got == want {
                agree += 1;
            } else {
                let best = next[0]["logprob"].as_f64().expect("logprob");
                gaps.push(
                    next.iter()
                        .find(|t| t["id"].as_u64() == Some(got as u64))
                        .and_then(|t| t["logprob"].as_f64())
                        .map(|lp| best - lp),
                );
            }
        }
        let near = gaps.iter().filter(|g| g.is_some_and(|g| g <= 0.5)).count();
        let outside = gaps.iter().filter(|g| g.is_none()).count();
        println!(
            "teacher-forced {name}: {agree}/{total} top-1 ({:.2}%); of {} disagreements {near} \
             within 0.5 nats of the reference's top-1, {outside} outside its top-5",
            100.0 * agree as f64 / total.max(1) as f64,
            gaps.len()
        );
        agree_all += agree;
        total_all += total;
    }
    let rate = agree_all as f64 / total_all.max(1) as f64;
    println!(
        "teacher-forced all: {agree_all}/{total_all} ({:.2}%)",
        100.0 * rate
    );
    assert!(
        rate >= 0.93,
        "teacher-forced top-1 {:.2}% under 93%",
        100.0 * rate
    );
}

/// Greedy chat answers through the checkpoint's template (reasoning off):
/// the dump's rendered prompt ids in, the first reply tokens compared.
#[test]
fn super_vl_chat_greedy() {
    let Some((mut model, oracle)) = setup() else {
        return;
    };
    let chats = oracle["chat"].as_array().expect("chat");
    let mut exact = 0usize;
    for c in chats {
        let prompt = ids(&c["prompt_ids"]);
        let want = ids(&c["greedy_ids"]);
        model.reset();
        let mut logits = model.forward_prefill_stream(&prompt).expect("prefill");
        let mut ours = Vec::with_capacity(want.len());
        for step in 0..want.len() {
            let tok = argmax(&logits);
            ours.push(tok);
            if step + 1 < want.len() {
                logits = model.forward(tok).expect("decode");
            }
        }
        let lead = ours.iter().zip(&want).take_while(|(a, b)| a == b).count();
        exact += usize::from(lead == want.len());
        println!(
            "{:?}: first {lead}/{} agree (reference {:?})",
            c["question"].as_str().unwrap_or(""),
            want.len(),
            c["text"].as_str().unwrap_or("")
        );
    }
    println!("chat greedy: {exact}/{} exact", chats.len());
    assert!(
        exact * 2 >= chats.len(),
        "fewer than half the chat answers agree"
    );
}

/// The prefill class at the prompt boundary: each teacher-forced passage
/// prefilled whole to a ladder of cut points, our first pick against the
/// reference's distribution there. The one-row gates above decode token by
/// token on the exact row class; this one measures what a served prompt's
/// bulk prefill - the batch lane, the sorted W4A4 expert pair where the pack
/// has it - hands the first answer token. Every cut is cold (prefix cache
/// off: a cut would otherwise resume from the one before it).
/// `PADDOCK_NO_NVF4_MOE_BS=1` runs the same ladder on the exact class for
/// the A/B.
#[test]
fn super_vl_prefill_boundaries() {
    let Some((mut model, oracle)) = setup() else {
        return;
    };
    // SAFETY: set before the batch lane exists, on the test's only thread
    // that reads the environment (one heavy test per process)
    unsafe { std::env::set_var("PADDOCK_NO_PREFIX_CACHE", "1") };
    model.enable_batch(1).expect("batch lane");
    let forced = oracle["teacher_forced"]
        .as_object()
        .expect("teacher_forced");
    const TIE_NATS: f64 = 0.05;
    let (mut agree, mut total, mut gap_sum) = (0usize, 0usize, 0f64);
    let (mut outside, mut ties) = (0usize, 0usize);
    let mut rows_ms = (0usize, 0f64);
    for (name, part) in forced {
        let toks = ids(&part["ids"]);
        let top5 = part["top5"].as_array().expect("top5");
        let mut line = Vec::new();
        for cut in (128..=toks.len()).step_by(128) {
            // the reference's position `cut` holds its distribution over
            // token `cut` given tokens 0..cut
            let Some(next) = top5.get(cut).and_then(|v| v.as_array()) else {
                continue;
            };
            model.reset();
            let t = std::time::Instant::now();
            let logits = model.forward_prefill_stream(&toks[..cut]).expect("prefill");
            let ms = t.elapsed().as_secs_f64() * 1e3;
            rows_ms.0 += cut;
            rows_ms.1 += ms;
            let got = argmax(&logits);
            let best = next[0]["logprob"].as_f64().expect("logprob");
            total += 1;
            if got == next[0]["id"].as_u64().expect("id") as u32 {
                agree += 1;
                line.push(format!("=({ms:.0})"));
            } else {
                match next
                    .iter()
                    .find(|t| t["id"].as_u64() == Some(got as u64))
                    .and_then(|t| t["logprob"].as_f64())
                {
                    Some(lp) => {
                        gap_sum += best - lp;
                        // the reference's logits are bf16: picks closer than
                        // its resolution are a tie it broke by index
                        ties += usize::from(best - lp <= TIE_NATS);
                        line.push(format!("{:.2}({ms:.0})", best - lp));
                    }
                    None => {
                        outside += 1;
                        line.push(format!("out({ms:.0})"));
                    }
                }
            }
        }
        println!("{name}: {}", line.join(" "));
    }
    let rate = (agree + ties) as f64 / total.max(1) as f64;
    println!(
        "prefill boundaries: {agree}/{total} top-1, {ties} more at reference ties ({:.1}% with \
         them), mean miss gap {:.3} nats, {outside} outside the reference's top-5; {:.0} rows/s",
        100.0 * rate,
        gap_sum / (total - agree - outside).max(1) as f64,
        rows_ms.0 as f64 / (rows_ms.1 / 1e3).max(1e-9)
    );
    assert!(
        rate >= 0.90,
        "prefill boundary top-1 {:.1}% under 90%",
        100.0 * rate
    );
    assert_eq!(outside, 0, "a boundary pick outside the reference's top-5");
}

/// The in-file MTP block (`mtp.layers.0` attention + `.1` latent MoE, vLLM's
/// nemotron_h_mtp order) drafts usefully: drafts are quality-only - the verify
/// re-judges every token - so the discriminator is the ACCEPTANCE RATE
/// against the trunk's own greedy walk (a norm / pairing / class error
/// collapses it toward 1/vocab; floor 0.25), and the committed stream must be
/// the no-spec greedy stream up to the first near-tie (the MTP hooks never
/// perturb the trunk).
#[test]
fn super_vl_mtp_drafts_accept() {
    // the drafter is this gate's subject: state the operator's choice
    // SAFETY: before any model loads, on the test's only thread
    unsafe {
        std::env::remove_var("PADDOCK_NO_SPEC");
        std::env::set_var("PADDOCK_SPEC", "on");
    }
    let Some((mut model, oracle)) = setup() else {
        return;
    };
    assert_eq!(model.batch_enable_probe(4).expect("enable"), 4);
    assert!(
        model.spec_capable(),
        "the in-file MTP must make Super spec-capable"
    );
    let prompt: Vec<u32> = ids(&oracle["teacher_forced"]["prose_a"]["ids"])
        .into_iter()
        .take(512)
        .collect();
    const ROUNDS: usize = 25;
    const K: usize = 3;
    // the spec slot walks cold first (a drafter warms only rows it saw)
    let l1 = model.forward_prefill(1, &prompt).expect("prefill 1");
    let l0 = model.forward_prefill(0, &prompt).expect("prefill 0");
    assert_eq!(argmax(&l1), argmax(&l0), "slot boundary picks disagree");
    let margin = |l: &[f32]| {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in l {
            if v > a {
                b = a;
                a = v;
            } else if v > b {
                b = v;
            }
        }
        a - b
    };
    // the no-spec greedy reference on slot 0
    let n = prompt.len();
    let mut b = vec![argmax(&l0)];
    let mut m = vec![margin(&l0)];
    for i in 0..ROUNDS * (K + 1) + 2 {
        let l = model
            .forward_batch(&[b[i]], &[(n + i) as u32])
            .expect("decode");
        b.push(argmax(&l));
        m.push(margin(&l));
    }
    let (mut committed, mut pending, mut pos) = (Vec::new(), b[0], n);
    let (mut drafted, mut accepted) = (0usize, 0usize);
    let mut parted: Option<usize> = None;
    let t = std::time::Instant::now();
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
        if parted.is_none()
            && let Some(i) = (0..committed.len()).find(|&i| committed[i] != b[i])
        {
            parted = Some(i);
            // a fork is legitimate only where the reference itself is a
            // near-tie (the verify rows and one-row decode share a class, so
            // the two streams agree to rounding)
            assert!(
                m[i] < 0.5,
                "round {round}: spec stream parts from greedy at {i} on a {:.3} margin",
                m[i]
            );
        }
        pending = picks[a];
        pos += a + 1;
    }
    let rate = accepted as f64 / drafted as f64;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "MTP acceptance: {accepted}/{drafted} = {rate:.3} over {ROUNDS} rounds at K={K}; {} \
         committed in {secs:.2}s ({:.1} tok/s on one slot); parted from greedy at {parted:?}",
        committed.len(),
        committed.len() as f64 / secs
    );
    assert!(
        rate >= 0.25,
        "MTP acceptance {rate:.3} below the 0.25 floor"
    );
}
