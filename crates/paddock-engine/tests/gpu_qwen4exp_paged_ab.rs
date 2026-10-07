//! Flash-Next's batch walks, deterministically, for an A/B across builds: a
//! fixed prefill wave (three runs in one 3967-row walk, one past the
//! dense-exact window so QSA serves), batched decode ticks, mixed ticks
//! carrying three decode rows beside a prompt that takes several spans,
//! then a four-wide decode, then three second turns (the history, its reply
//! and a new message) resumed from the prefix cache - two in their own
//! slots, one in another slot - a resume at an in-walk cut, and the chunked
//! flow the server runs (a long prompt, its reply, the slot released, turns
//! resumed at the reply and at both in-walk cuts). A second dump replays the
//! served single-slot shape, where the pool holds exactly full context and
//! the checkpoints it wants. Every logits row's hash goes to `Q4X_AB_OUT`
//! (one line each). The same file run against two builds must print the same
//! lines when the builds compute the same bits - how paging the live KV was
//! checked against the dense strips it replaced (the paged kernels are
//! bit-identical twins). Heavy: `QWEN38FN_GGUF=<first shard>`;
//! `QWEN38FN_MOE_DEVICE=1` on a unified-memory box.

mod common;

use std::fmt::Write as _;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b })
}

/// FNV-1a over the row's bits: equal rows, equal lines.
fn hash(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
    })
}

fn prompt(tok: &GgufTokenizer, text: &str, n: usize) -> Vec<u32> {
    let base = tok.encode(text).expect("enc");
    base.iter().copied().cycle().take(n).collect()
}

#[test]
fn gguf_batch_walks_dump() {
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
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 32768, 4).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");

    let mut out = String::new();
    let mut line = |tag: &str, v: &[f32]| {
        writeln!(out, "{tag} argmax {} hash {:016x}", argmax(v), hash(v)).unwrap();
    };
    let a = prompt(
        &tok,
        "A parcel network links twelve cities; each depot keeps a ledger of crates, carriers and days. ",
        67,
    );
    let b = prompt(
        &tok,
        "The harbour master's log records every vessel, its cargo of timber or iron ore, the tide and the \
         pilot who brought it in, day after day through the long northern winter. ",
        2400,
    );
    let c = prompt(
        &tok,
        "In the archive, letters between two cartographers argue about the true course of a river that \
         changes its bed after every spring flood. ",
        1500,
    );
    let d = prompt(
        &tok,
        "A lighthouse keeper on a limestone island writes down the colour of the sea at dawn. ",
        2600,
    );
    // the wave: three runs in one walk
    let first = m
        .prefill_slots(&[(0, a.clone()), (1, b.clone()), (2, c.clone())])
        .expect("wave");
    // each slot's history: its prompt, then every token fed back to it
    let mut hist: Vec<Vec<u32>> = vec![a.clone(), b.clone(), c.clone(), d.clone()];
    let mut rows: Vec<(usize, u32)> = Vec::new();
    for (i, l) in first.iter().enumerate() {
        line(&format!("wave run {i}"), l);
        rows.push((i, argmax(l) as u32));
    }
    // batched decode ticks
    for step in 0..12 {
        for &(s, t) in &rows {
            hist[s].push(t);
        }
        let ls = m.decode_step_batch(&rows).expect("decode");
        for (r, l) in ls.iter().enumerate() {
            line(&format!("decode3 step {step} row {r}"), l);
            rows[r].1 = argmax(l) as u32;
        }
    }
    // mixed ticks: slot 3's prompt in 700-row spans beside three decode rows
    m.prefill_begin(3, d.clone()).expect("prefill_begin");
    let mut tick = 0;
    let d_last = loop {
        let decodes: Vec<(usize, u32, u32)> = rows
            .iter()
            .map(|&(s, t)| (s, t, m.slot_position(s) as u32))
            .collect();
        for &(s, t) in &rows {
            hist[s].push(t);
        }
        let (logits, finished) = m.forward_mixed(&decodes, 700).expect("mixed tick");
        let vocab = logits.len() / decodes.len();
        for (r, l) in logits.chunks(vocab).enumerate() {
            line(&format!("mixed tick {tick} row {r}"), l);
            rows[r].1 = argmax(l) as u32;
        }
        tick += 1;
        if let Some((_, l, _)) = finished.into_iter().find(|f| f.0 == 3) {
            break l;
        }
        assert!(tick < 100, "prefill never finished");
    };
    line("mixed prompt last", &d_last);
    rows.push((3, argmax(&d_last) as u32));
    // four-wide decode
    for step in 0..6 {
        for &(s, t) in &rows {
            hist[s].push(t);
        }
        let ls = m.decode_step_batch(&rows).expect("decode");
        for (r, l) in ls.iter().enumerate() {
            line(&format!("decode4 step {step} row {r}"), l);
            rows[r].1 = argmax(l) as u32;
        }
    }
    // second turns off the prefix cache: slots 0 and 1 continue their own
    // conversations, slot 2's continues in slot 3
    let msg = tok
        .encode(" Now add the next entry to the ledger.")
        .expect("enc");
    for (conv, slot) in [(0usize, 0usize), (1, 1), (2, 3)] {
        let mut t2 = hist[conv].clone();
        t2.extend(&msg);
        let l = m.prefill_slot(slot, &t2).expect("turn 2");
        let reused = m.take_prefill_reused(slot);
        line(
            &format!("turn2 conv {conv} in slot {slot} resumed at {reused}"),
            &l,
        );
    }
    // a resume at an IN-WALK cut: a fresh 2400-token prompt (its cuts at
    // 2368 / 2384 are taken inside its walk), then the same prompt cut short
    // with another ending resumes at the upper cut
    let e = prompt(
        &tok,
        "The glassblower's apprentice counted every bubble trapped in the green vase, and the \
         master wrote each count into a ledger bound in sealskin. ",
        2400,
    );
    let l = m.prefill_slot(2, &e).expect("in-walk fresh");
    line("inwalk fresh slot 2", &l);
    let mut e2 = e[..2390].to_vec();
    e2.extend(&msg);
    let l = m.prefill_slot(0, &e2).expect("in-walk resume");
    let reused = m.take_prefill_reused(0);
    line(&format!("inwalk resume slot 0 at {reused}"), &l);
    // the server's flow: a long prompt through chunked prefill (mixed ticks
    // of 2048 rows, in-walk cuts on the finishing span), a greedy reply
    // whose decode closes pages, the slot released as the scheduler does
    // between turns, then the next turn chunked the same way
    let chunked = |m: &mut Qwen4ExpGpu, slot: usize, toks: Vec<u32>| -> Vec<f32> {
        m.prefill_begin(slot, toks).expect("prefill_begin");
        loop {
            let (_, finished) = m.forward_mixed(&[], 2048).expect("mixed tick");
            if let Some((_, l, _)) = finished.into_iter().find(|f| f.0 == slot) {
                break l;
            }
        }
    };
    let f = prompt(
        &tok,
        "A cartographer's diary: every evening she redrew the coastline from the soundings the \
         fishermen brought back, and every morning the tide had moved it again. ",
        9000,
    );
    let l = chunked(&mut m, 1, f.clone());
    line("chunked turn1 slot 1", &l);
    let mut hist1 = f.clone();
    let mut t = argmax(&l) as u32;
    for step in 0..40 {
        hist1.push(t);
        let d = m.decode_step_batch(&[(1, t)]).expect("decode");
        line(&format!("chunked reply step {step}"), &d[0]);
        t = argmax(&d[0]) as u32;
    }
    m.release_inactive_slots(&[true, false, true, true]);
    let mut t2 = hist1.clone();
    t2.extend(&msg);
    let l = chunked(&mut m, 1, t2);
    let reused = m.take_prefill_reused(1);
    line(&format!("chunked turn2 slot 1 at {reused}"), &l);
    // the served multi-turn shape: the re-rendered history diverges inside
    // the prompt's last page, so the next turn resumes at one of the two
    // in-walk cuts the chunked walk's finishing span took (8976, 8992)
    for (keep, slot) in [(8998usize, 1usize), (8985, 1)] {
        m.release_inactive_slots(&[true, false, true, true]);
        let mut t3 = f[..keep].to_vec();
        t3.extend(&msg);
        let l = chunked(&mut m, slot, t3);
        let reused = m.take_prefill_reused(slot);
        line(
            &format!("chunked in-walk resume slot {slot} at {reused}"),
            &l,
        );
    }
    let dest = std::env::var("Q4X_AB_OUT").unwrap_or_else(|_| "q4x_ab.txt".into());
    std::fs::write(&dest, &out).expect("write dump");
    eprintln!("{} lines -> {dest}", out.lines().count());
}

/// The served single-slot shape: one slot at 32K, so the plan's pool is full
/// context plus the checkpoints it wants and nothing more - every checkpoint
/// past the index space and every page past the pool is taken back
/// (`make_room`, the index steal). The scheduler's calls in its order: each
/// request chunked (`prefill_begin`, riderless mixed ticks), its reply
/// decoded as one-row mixed ticks, the slot released when it ends. Every
/// prompt opens with the same header, whose pages the first request files
/// and every later one shares. Turn 2 re-renders the long turn's history so
/// it diverges inside the prompt's last page and resumes at the upper in-walk
/// cut - on the header pages the warm request's walk wrote, not the ones the
/// long turn computed itself. The walks are not batch-invariant (a row's KV
/// depends, past the last ulp, on how many rows walked beside it), so these
/// lines differ from a build that resumed a conversation on its own pages
/// (phase B's same-slot checkpoints) while every line before them agrees.
#[test]
fn gguf_served_shape_dump() {
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
    // the cache's own account of every checkpoint and resume, when asked
    if std::env::var_os("PADDOCK_PREFIX_STATS").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .try_init();
    }
    if std::env::var_os("QWEN38FN_MOE_DEVICE").is_none() {
        unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    }
    let map = MappedGguf::open(&path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    drop(map);
    common::qwen4exp_exact_kv();
    let mut m = Qwen4ExpGpu::load_gguf_with_slots(&exec, &path, 32768, 1).expect("load gguf");
    let headroom = exec.vram_headroom().unwrap_or(0);
    m.enable_moe_cache(headroom.saturating_sub(512 << 20))
        .expect("cache");

    let mut out = String::new();
    let mut line = |tag: &str, v: &[f32]| {
        writeln!(out, "{tag} argmax {} hash {:016x}", argmax(v), hash(v)).unwrap();
    };
    // one request the scheduler's way: returns the prompt's logits and the
    // reply's, the reply tokens appended to `hist`
    let request = |m: &mut Qwen4ExpGpu, toks: Vec<u32>, reply: usize, hist: &mut Vec<u32>| {
        let mut ls = Vec::new();
        m.prefill_begin(0, toks.clone()).expect("prefill_begin");
        let l = loop {
            let (_, finished) = m.forward_mixed(&[], 4096).expect("mixed tick");
            if let Some((_, l, _)) = finished.into_iter().find(|f| f.0 == 0) {
                break l;
            }
        };
        let reused = m.take_prefill_reused(0);
        hist.clear();
        hist.extend(&toks);
        let mut t = argmax(&l) as u32;
        ls.push((format!("prompt {} resumed at {reused}", toks.len()), l));
        for step in 0..reply {
            hist.push(t);
            let p = m.slot_position(0) as u32;
            let (d, _) = m.forward_mixed(&[(0, t, p)], 4096).expect("decode tick");
            t = argmax(&d) as u32;
            ls.push((format!("  reply step {step}"), d));
        }
        m.release_inactive_slots(&[false]);
        ls
    };
    let mut hist = Vec::new();
    let warm = prompt(
        &tok,
        "A parcel network links twelve cities; each depot keeps a ledger of crates, carriers and days. ",
        250,
    );
    let short = prompt(
        &tok,
        "Explain how a parcel-routing system could be designed. ",
        70,
    );
    let long = prompt(
        &tok,
        "Record: the depot in Kalmar shipped 95 crates of salted cod to Sundsvall on day 269, \
         carrier code 5289. The depot in Visby shipped timber to Falun on day 12. ",
        16530,
    );
    let msg = tok
        .encode(" Now list every record that shipped timber, by record number.")
        .expect("enc");
    // every request opens with the same header (a chat template's system
    // turn), so its pages are filed once and later prompts share them
    let header = prompt(
        &tok,
        "You are a careful assistant. Answer from the records only. ",
        40,
    );
    let with_header = |body: Vec<u32>| -> Vec<u32> { header.iter().copied().chain(body).collect() };
    let (warm, short, long) = (with_header(warm), with_header(short), with_header(long));
    for (name, toks, reply) in [
        ("warm", warm, 4usize),
        ("short", short, 200),
        ("long", long.clone(), 160),
    ] {
        for (tag, l) in request(&mut m, toks, reply, &mut hist) {
            line(&format!("{name} {tag}"), &l);
        }
    }
    let mut t2 = long[..16566].to_vec();
    t2.extend(&msg);
    for (tag, l) in request(&mut m, t2, 40, &mut hist) {
        line(&format!("turn2 {tag}"), &l);
    }
    // beside the first dump's file, not over it
    let dest = std::env::var("Q4X_AB_OUT").unwrap_or_else(|_| "q4x_ab.txt".into()) + ".served";
    std::fs::write(&dest, &out).expect("write dump");
    eprintln!("{} lines -> {dest}", out.lines().count());
}
