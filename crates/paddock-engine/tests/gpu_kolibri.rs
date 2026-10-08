//! Kolibri 1 (Aleph Alpha) on the laguna body's Kolibri flavor.
//!
//! - `kolibri_router_*`: slot 748's sigmoid_logit_add router against its
//!   definition (Aleph Alpha's vLLM plugin: top-k on logits + bias, weights =
//!   sigmoid of the selected raw logits, never renormalized), and proof the
//!   test tells it apart from the Laguna router on the same planes. Light.
//! - `kolibri_q4km_*` (PADDOCK_HEAVY_TESTS + the Q4_K_M file): the file loads,
//!   a known-answer prompt answers, and the batched lane's greedy stream
//!   tracks the serial exact-f32 lane's (a divergence must sit on a near-tie
//!   of the serial logits - the two lanes are different numeric classes).
//! - `kolibri_nvfp4_*` (heavy + the `primitive-ai/Kolibri-1-NVFP4` dir): the
//!   same two gates on the compressed-tensors NVFP4 build, the SWA prefix
//!   cache's back-off resume (a rewritten tail resumes behind the cut), and
//!   the reply checkpoint a tool call pins (the next turn resumes past the
//!   reply).
//!
//! No llama.cpp release reads `kolibri1` yet, so there is no same-weights
//! black-box reference for this family; the architecture cross-check is the
//! official FP8 checkpoint under Aleph Alpha's own vLLM plugin.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::laguna::GpuLaguna;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[test]
fn kolibri_router_selects_on_logits_plus_bias() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_moe_topk_logit_sigmoid() {
        common::missing("pack has no moe_topk_logit_sigmoid_batch (slot 748)");
        return;
    }
    // (experts, k, rows, bias amplitude, routed scale): Kolibri's 384/6, the
    // 8- and 16-wide lanes, a ragged count and a zero bias
    let cases = [
        (384usize, 6usize, 9usize, 4.0f32, 1.0f32),
        (256, 8, 5, 4.0, 1.0),
        (512, 16, 3, 4.0, 2.5),
        (100, 6, 4, 0.0, 1.0),
        (385, 6, 2, 1.0, 1.0),
    ];
    for (ne, k, rows, amp, scale) in cases {
        let logits: Vec<f32> = det(rows * ne, 7 + ne as u64)
            .into_iter()
            .map(|v| 3.0 * v)
            .collect();
        let bias: Vec<f32> = det(ne, 11).into_iter().map(|v| amp * v).collect();
        let d_l = exec.to_device(&logits).expect("logits");
        let d_b = exec.to_device(&bias).expect("bias");
        let mut d_i = exec.alloc_u32(rows * k).expect("idx");
        let mut d_w = exec.alloc(rows * k).expect("w");
        exec.moe_topk_logit_sigmoid_batch(&d_l, &d_b, scale, ne, k, &mut d_i, &mut d_w, rows)
            .expect("router");
        let idx = exec.stream.clone_dtoh(&d_i).expect("idx back");
        let w = exec.stream.clone_dtoh(&d_w).expect("w back");
        for b in 0..rows {
            let row = &logits[b * ne..(b + 1) * ne];
            let mut order: Vec<usize> = (0..ne).collect();
            // f32 adds round identically on host and device (no fma here)
            order.sort_by(|&x, &y| {
                (row[y] + bias[y])
                    .partial_cmp(&(row[x] + bias[x]))
                    .expect("finite")
                    .then(x.cmp(&y))
            });
            for s in 0..k {
                let got = idx[b * k + s] as usize;
                assert_eq!(got, order[s], "ne {ne} row {b} pick {s}");
                let want = sigmoid(row[got]) * scale;
                let err = (w[b * k + s] - want).abs();
                assert!(
                    err <= 2e-7 * want.abs().max(1.0),
                    "ne {ne} row {b} pick {s}: w {} vs {want}",
                    w[b * k + s]
                );
            }
        }
    }
    println!("slot 748 router: picks exact, weights within 2e-7 on all cases");
}

#[test]
fn kolibri_router_is_not_the_laguna_router() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_moe_topk_logit_sigmoid() {
        common::missing("pack has no moe_topk_logit_sigmoid_batch (slot 748)");
        return;
    }
    // logits and a bias of real magnitude: selecting on sigmoid(l) + b
    // squashes the logit term into (0, 1) and lets the bias dominate, so the
    // two classes must choose different experts. 256 experts - the Laguna
    // kernel's ceiling.
    let (ne, k, rows) = (256usize, 6usize, 8usize);
    let logits: Vec<f32> = det(rows * ne, 3).into_iter().map(|v| 6.0 * v).collect();
    let bias: Vec<f32> = det(ne, 5).into_iter().map(|v| 2.0 * v).collect();
    let d_l = exec.to_device(&logits).expect("logits");
    let d_b = exec.to_device(&bias).expect("bias");
    let mut d_ik = exec.alloc_u32(rows * k).expect("idx");
    let mut d_wk = exec.alloc(rows * k).expect("w");
    let mut d_il = exec.alloc_u32(rows * k).expect("idx laguna");
    let mut d_wl = exec.alloc(rows * k).expect("w laguna");
    exec.moe_topk_logit_sigmoid_batch(&d_l, &d_b, 1.0, ne, k, &mut d_ik, &mut d_wk, rows)
        .expect("kolibri router");
    if exec
        .moe_topk_sigmoid_batch(&d_l, &d_b, 1.0, ne, k, &mut d_il, &mut d_wl, rows)
        .is_err()
    {
        common::missing("pack has no moe_topk_sigmoid_batch");
        return;
    }
    let a = exec.stream.clone_dtoh(&d_ik).expect("a");
    let b = exec.stream.clone_dtoh(&d_il).expect("b");
    let differ = (0..rows)
        .filter(|&r| {
            let mut x = a[r * k..(r + 1) * k].to_vec();
            let mut y = b[r * k..(r + 1) * k].to_vec();
            x.sort_unstable();
            y.sort_unstable();
            x != y
        })
        .count();
    assert!(
        differ >= rows / 2,
        "only {differ}/{rows} rows choose differently - the fixture cannot tell the routers apart"
    );
    println!("router classes choose differently on {differ}/{rows} rows");
}

/// The GGUF-built tokenizer (pre "kolibri1" -> the qwen2 split) must encode
/// exactly as Aleph Alpha's tokenizer.json does. KOLIBRI_HF_DIR names a
/// directory holding the official tokenizer.json (any Kolibri-1 checkout).
#[test]
fn kolibri_gguf_tokenizer_matches_hf() {
    let Some(path) = common::model("KOLIBRI_GGUF", common::KOLIBRI_Q4KM) else {
        return;
    };
    let Some(hf) = std::env::var_os("KOLIBRI_HF_DIR").map(std::path::PathBuf::from) else {
        common::missing("KOLIBRI_HF_DIR (official tokenizer.json) unset");
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let g = GgufTokenizer::from_gguf(map.gguf()).expect("gguf tokenizer");
    let h = GgufTokenizer::from_hf_dir(&hf).expect("hf tokenizer");
    let texts = [
        "Die Hauptstadt von Frankreich ist",
        "Gr\u{fc}\u{df}e aus M\u{fc}nchen! \u{dc}ber 1234567 B\u{e4}ume, 3.14159 und 2026-10-06.",
        "The quick brown fox jumps over the lazy dog.\n\n  Indented   spaces\ttab",
        "def f(x):\n    return [i ** 2 for i in range(x)]  # Kommentar",
        "<|im_start|>user\nWie ist das Wetter?<|im_end|>\n<|im_start|>assistant\n<think>\n",
        "emoji \u{1f426} \u{4e2d}\u{6587} \u{414}\u{410} na\u{ef}ve caf\u{e9} \u{2014} \u{201e}Anf\u{fc}hrungszeichen\u{201c}",
    ];
    for t in texts {
        let a = g.encode(t).expect("gguf encode");
        let b = h.encode(t).expect("hf encode");
        assert_eq!(a, b, "tokenizers part on {t:?}");
    }
    println!("gguf tokenizer == tokenizer.json on {} texts", texts.len());
}

/// Greedy argmax and the top1 - top2 margin of one logits row.
fn top2(l: &[f32]) -> (u32, f32) {
    let (mut b, mut bv, mut sv) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in l.iter().enumerate() {
        if v > bv {
            sv = bv;
            bv = v;
            b = i;
        } else if v > sv {
            sv = v;
        }
    }
    (b as u32, bv - sv)
}

#[test]
fn kolibri_q4km_answers_and_lanes_agree() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("KOLIBRI_GGUF", common::KOLIBRI_Q4KM) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuLaguna::load(exec, &map, 2048).expect("load");
    answers_and_lanes_agree(&mut m, &tok);
}

/// The NVFP4 safetensors build (BF16 attention / shared expert / head, NVFP4
/// routed experts): the serial lane runs them W4A16, the batched prefill W4A4.
#[test]
fn kolibri_nvfp4_answers_and_lanes_agree() {
    if !common::heavy() {
        return;
    }
    let Some(dir) = common::model("KOLIBRI_NVFP4", common::KOLIBRI_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let mut m = GpuLaguna::load_kolibri_nvfp4(exec, &dir, 2048).expect("load");
    answers_and_lanes_agree(&mut m, &tok);
}

fn answers_and_lanes_agree(m: &mut GpuLaguna, tok: &GgufTokenizer) {
    assert!(!tok.add_bos, "kolibri1 leads with no BOS");
    assert!(
        tok.stop_ids().contains(&127906) && tok.stop_ids().contains(&127901),
        "<|im_end|> and <|endoftext|> both stop: {:?}",
        tok.stop_ids()
    );
    // the template's reasoning-off rendering (a raw-text continuation is not
    // what a chat-tuned reasoning model answers; at greedy it loops)
    let prompt = tok
        .encode(
            "<|im_start|>system\n# Reasoning effort\n\nReasoning is disabled. Proceed straight \
             to answering according to the user's instructions.<|im_end|>\n<|im_start|>user\n\
             Wie hei\u{df}t die Hauptstadt von Frankreich? Antworte mit einem Satz.<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n",
        )
        .expect("encode");
    let n_gen = 48usize;
    let stops = tok.stop_ids();

    // serial lane: exact-f32 GEMVs over a dense per-layer KV; greedy to the
    // first stop token
    let mut logits = Vec::new();
    for &t in &prompt {
        logits = m.forward(t).expect("serial prefill");
    }
    let first_serial = logits.clone();
    let mut serial = Vec::new();
    let mut margins = Vec::new();
    while serial.len() < n_gen {
        let (t, gap) = top2(&logits);
        serial.push(t);
        margins.push(gap);
        if stops.contains(&t) {
            break;
        }
        logits = m.forward(t).expect("serial decode");
    }
    let text = tok.decode(&serial, true).expect("decode");
    println!("serial ({} tokens): {text:?}", serial.len());
    assert!(
        text.contains("Paris"),
        "serial lane does not answer: {text:?}"
    );
    assert!(
        serial.last().is_some_and(|t| stops.contains(t)),
        "serial lane never stopped within {n_gen} tokens"
    );

    // batched lane on the same weights (enable_batch drops the serial state):
    // W4A8 projections, paged KV, the decode-class MoE
    m.enable_batch(2).expect("enable_batch");
    let mut logits = m.forward_prefill(0, &prompt).expect("batched prefill");
    let dmax = first_serial
        .iter()
        .zip(&logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("first-token logits: serial vs batched max |diff| {dmax:.4}");
    let mut batched = Vec::new();
    while batched.len() < serial.len() {
        let (t, _) = top2(&logits);
        batched.push(t);
        if stops.contains(&t) {
            break;
        }
        let pos = (prompt.len() + batched.len() - 1) as u32;
        logits = m.forward_batch(&[t], &[pos]).expect("batched decode");
    }
    let btext = tok.decode(&batched, true).expect("decode");
    println!("batched ({} tokens): {btext:?}", batched.len());
    assert!(
        btext.contains("Paris"),
        "batched lane does not answer: {btext:?}"
    );
    // the two lanes are different numeric classes: they may part, but only
    // where the serial stream itself was close to a tie
    match (0..serial.len().min(batched.len())).find(|&i| serial[i] != batched[i]) {
        Some(i) => {
            println!("lanes part at token {i} (serial margin {:.4})", margins[i]);
            assert!(
                margins[i] < 1.0,
                "lanes part at token {i} on a {:.3} margin - not a near-tie",
                margins[i]
            );
        }
        None => {
            assert_eq!(serial.len(), batched.len(), "one lane stopped early");
            println!(
                "lanes agree on all {} tokens through the stop",
                serial.len()
            );
        }
    }
}

/// The batched prefill (W4A8 projections, the hd128 repack-pane G=12 tile,
/// the SWA rings and their 512-row span ladder) against the serial exact
/// lane on a prompt that crosses the 513-key window three times over: the
/// last row's logits must pick the same token and sit close.
#[test]
fn kolibri_q4km_long_prefill_tracks_serial() {
    if !common::heavy() {
        return;
    }
    let Some(path) = common::model("KOLIBRI_GGUF", common::KOLIBRI_Q4KM) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuLaguna::load(exec, &map, 4096).expect("load");
    long_prefill_tracks_serial(&mut m, &tok);
}

/// The same on the NVFP4 build: the batched prefill's routed experts run W4A4
/// (activations quantized to nvfp4) against the serial lane's W4A16.
#[test]
fn kolibri_nvfp4_long_prefill_tracks_serial() {
    if !common::heavy() {
        return;
    }
    let Some(dir) = common::model("KOLIBRI_NVFP4", common::KOLIBRI_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let mut m = GpuLaguna::load_kolibri_nvfp4(exec, &dir, 4096).expect("load");
    long_prefill_tracks_serial(&mut m, &tok);
}

fn long_prefill_tracks_serial(m: &mut GpuLaguna, tok: &GgufTokenizer) {
    let mut text = String::from("<|im_start|>user\nHier ist ein Protokoll:\n");
    for i in 0..120 {
        text.push_str(&format!(
            "Eintrag {i}: Paket P-{} von Ulm nach Kiel, Gewicht {} kg.\n",
            1000 + 37 * i,
            1 + i % 17
        ));
    }
    text.push_str("Welches Gewicht hatte Eintrag 7?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt = tok.encode(&text).expect("encode");
    assert!(prompt.len() > 3 * 513, "prompt {} tokens", prompt.len());

    let mut serial = Vec::new();
    for &t in &prompt {
        serial = m.forward(t).expect("serial");
    }
    m.enable_batch(2).expect("enable_batch");
    let batched = m.forward_prefill(0, &prompt).expect("batched prefill");
    let (ts, gap) = top2(&serial);
    let (tb, _) = top2(&batched);
    let dmax = serial
        .iter()
        .zip(&batched)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "{} tokens: serial top1 {ts} (margin {gap:.3}) batched top1 {tb}, max |diff| {dmax:.4}",
        prompt.len()
    );
    assert_eq!(ts, tb, "last-row pick differs (serial margin {gap:.3})");
    assert!(dmax < 1.5, "last-row logits part by {dmax}");
}

/// The SWA prefix cache's BACK-OFF checkpoint on the NVFP4 build: a prompt
/// whose last 40 tokens were rewritten (the same document, another question)
/// resumes at the first prompt's back-off cut (~256 tokens behind its last
/// page) instead of re-prefilling whole - the trailing checkpoint sits past
/// the divergence. The reference is the serial exact lane (it touches no
/// cache): the resumed prefill must pick what it picks. The logit bound is
/// the W4A16-vs-W4A4 class's, loosened for a resumed tail: per-token nvfp4
/// activation codes are discontinuous, so a re-chunked walk's ulp-level
/// differences can flip a code (measured cold-vs-resumed: 1.98 max over the
/// vocabulary at a 37.8 margin, 0.04 on the tokens a greedy stream emits).
#[test]
fn kolibri_nvfp4_prefix_backoff_resumes() {
    if !common::heavy() {
        return;
    }
    let Some(dir) = common::model("KOLIBRI_NVFP4", common::KOLIBRI_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let mut text = String::from("<|im_start|>user\nHier ist ein Protokoll:\n");
    for i in 0..120 {
        text.push_str(&format!(
            "Eintrag {i}: Paket P-{} von Ulm nach Kiel, Gewicht {} kg.\n",
            1000 + 37 * i,
            1 + i % 17
        ));
    }
    let a = tok.encode(&text).expect("encode");
    let mut b = a[..a.len() - 40].to_vec();
    b.extend(
        tok.encode("Welches Gewicht hatte Eintrag 7?<|im_end|>\n<|im_start|>assistant\n")
            .expect("encode"),
    );
    let mut m = GpuLaguna::load_kolibri_nvfp4(exec, &dir, 4096).expect("load");
    let mut serial = Vec::new();
    for &t in &b {
        serial = m.forward(t).expect("serial b");
    }
    m.enable_batch(2).expect("enable_batch");
    m.forward_prefill(0, &a).expect("prefill a");
    let warm = m.forward_prefill(1, &b).expect("warm b");
    let reused = m.take_prefill_reused(1);
    // a's checkpoints: its last page boundary and the back-off cut ~256
    // tokens behind it; b diverges 40 tokens before a's end
    let last = (a.len() - 1) / 16 * 16;
    let backoff = (last - 256) / 16 * 16;
    println!(
        "a {} tokens (cuts {backoff}, {last}), b {} sharing {}: resumed at {reused}",
        a.len(),
        b.len(),
        a.len() - 40
    );
    assert_eq!(reused, backoff, "b must resume at a's back-off cut");
    let (ts, gap) = top2(&serial);
    let (tw, _) = top2(&warm);
    let dmax = serial
        .iter()
        .zip(&warm)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("serial top1 {ts} (margin {gap:.3}) resumed top1 {tw}, max |diff| {dmax:.4}");
    assert_eq!(
        ts, tw,
        "resumed b picks differently (serial margin {gap:.3})"
    );
    assert!(dmax < 3.0, "resumed logits part by {dmax}");
}

/// The reply checkpoint (`Generator::reply_pin_at`): a reply that opens a
/// tool call files itself under the prefix cache at the call, so the next
/// turn - this prompt, the reply, then the tool's result - resumes past the
/// reply instead of at the prompt's last page. The reply is the serial
/// lane's greedy one (the exact-f32 reference), teacher-forced through the
/// batched lane; b ends inside the next reasoning block, where the next
/// token's distribution is broad, and the resumed logits must match the
/// serial lane's as closely as a cold batched prefill does.
#[test]
fn kolibri_nvfp4_reply_pin_resumes_past_the_reply() {
    if !common::heavy() {
        return;
    }
    let Some(dir) = common::model("KOLIBRI_NVFP4", common::KOLIBRI_NVFP4) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let tok = GgufTokenizer::from_hf_dir(&dir).expect("tokenizer");
    let mut text = String::from("<|im_start|>user\nHier ist ein Protokoll:\n");
    for i in 0..60 {
        text.push_str(&format!(
            "Eintrag {i}: Paket P-{} von Ulm nach Kiel, Gewicht {} kg.\n",
            1000 + 37 * i,
            1 + i % 17
        ));
    }
    text.push_str("Fasse das Protokoll kurz zusammen.<|im_end|>\n<|im_start|>assistant\n");
    let a = tok.encode(&text).expect("encode");
    let mut m = GpuLaguna::load_kolibri_nvfp4(exec, &dir, 4096).expect("load");
    // serial: the prompt, a greedy reply of 70 tokens fed, then a tool result
    let mut logits = Vec::new();
    for &t in &a {
        logits = m.forward(t).expect("serial a");
    }
    let mut history = a.clone();
    for _ in 0..70 {
        let t = top2(&logits).0;
        history.push(t);
        logits = m.forward(t).expect("serial reply");
    }
    let pos = history.len() as u32; // every reply token fed
    let mut b = history.clone();
    b.extend(
        tok.encode("<|im_end|>\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
            .expect("encode"),
    );
    let mut serial = logits;
    for &t in &b[history.len()..] {
        serial = m.forward(t).expect("serial b");
    }
    // batched: prefill the prompt in slot 0, teacher-force the reply, pin
    m.enable_batch(2).expect("enable_batch");
    m.forward_prefill(0, &a).expect("prefill a");
    for (p, &t) in history.iter().enumerate().skip(a.len()) {
        m.forward_batch(&[t], &[p as u32]).expect("reply row");
    }
    m.reply_pin_at(0, &history, pos);
    let warm = m.forward_prefill(1, &b).expect("warm b");
    let reused = m.take_prefill_reused(1);
    let prompt_cut = (a.len() - 1) / 16 * 16;
    let pin_cut = pos as usize / 16 * 16;
    println!(
        "a {} tokens (last cut {prompt_cut}), reply to {pos} (pin cut {pin_cut}), b {}: resumed at {reused}",
        a.len(),
        b.len()
    );
    let (ts, gap) = top2(&serial);
    let (tw, _) = top2(&warm);
    let dmax = serial
        .iter()
        .zip(&warm)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let (kl, top5) = kl_top5(&serial, &warm);
    println!(
        "serial top1 {ts} (margin {gap:.3}) resumed top1 {tw}, max |diff| {dmax:.4}, \
         KL {kl:.2e}, top-5 shared {top5}"
    );
    assert_eq!(
        reused, pin_cut,
        "b must resume at the reply's pin, past the prompt"
    );
    // the class the cold batched prefill of b sits in against the serial
    // lane (GB10, 2026-10-08: KL 3.4e-2, max |diff| 1.13; resumed 3.7e-2, 1.35)
    assert!(
        ts == tw || gap < 0.3,
        "resumed b picks differently (serial margin {gap:.3})"
    );
    assert!(kl < 0.1, "resumed distribution parts by KL {kl:.3e}");
    assert!(dmax < 3.0, "resumed logits part by {dmax}");
}

/// KL(p || q) of two logit rows' softmaxes, and how many of p's top 5 are q's.
fn kl_top5(p: &[f32], q: &[f32]) -> (f64, usize) {
    let sm = |l: &[f32]| {
        let m = l.iter().copied().fold(f32::MIN, f32::max) as f64;
        let e: Vec<f64> = l.iter().map(|&x| (x as f64 - m).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|x| x / z).collect::<Vec<f64>>()
    };
    let (pp, qq) = (sm(p), sm(q));
    let kl = pp
        .iter()
        .zip(&qq)
        .filter(|(a, _)| **a > 0.0)
        .map(|(a, b)| a * (a / b.max(1e-300)).ln())
        .sum();
    let top = |v: &[f32]| {
        let mut i: Vec<usize> = (0..v.len()).collect();
        i.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        i.truncate(5);
        i
    };
    let (tp, tq) = (top(p), top(q));
    (kl, tp.iter().filter(|i| tq.contains(i)).count())
}
