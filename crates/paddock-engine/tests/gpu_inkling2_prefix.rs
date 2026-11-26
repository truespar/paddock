//! Inkling 2 Mini's prefix cache on CUDA (slot 1014 + the radix over the
//! full visits' blocks), heavy: the official release (`INKLING2_DIR`) on
//! token sequences from the teacher-forced goldens (`INKLING2_GOLDENS`).
//!
//! - `inkling2_conv_capture_is_the_carried_state`: slot 1014 against its
//!   definition - a run's last three inputs had it ended at the cut, folded
//!   from 8 partials as slot 1004 folds them, the old state where the cut
//!   is under three rows in - bit for bit. Light.
//! - `inkling2_prefix_resume`: a prompt lands (its cuts captured mid-pass),
//!   its slot decodes a reply (rolling reply checkpoints); then, on the other
//!   slot, a next turn that re-sends the reply resumes at the reply's last
//!   checkpoint and a next turn that drops it resumes at the prompt's end -
//!   each against the same prompt prefilled cold: the prompt's last logits
//!   and a greedy continuation part only at near-ties. Then every block goes
//!   back to the pool once the slots let go.

mod common;

use paddock_engine::generator::Generator;
use paddock_engine::gpu_model::inkling2::GpuInkling2;

fn i32s(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

fn top1(l: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::MIN);
    let mut second = f32::MIN;
    for (i, &v) in l.iter().enumerate() {
        if v > best.1 {
            second = best.1;
            best = (i, v);
        } else if v > second {
            second = v;
        }
    }
    (best.0, best.1 - second)
}

/// A queued prompt on `slot` through mixed ticks (no decode rows) until it
/// lands: its last logits.
fn land(m: &mut GpuInkling2, slot: usize) -> Vec<f32> {
    loop {
        let (_, fin) = m.forward_mixed(&[], 4096).expect("mixed tick");
        if let Some((_, l, _)) = fin.into_iter().find(|f| f.0 == slot) {
            return l;
        }
        assert!(
            !m.prefill_queue().is_empty(),
            "slot {slot}'s prompt left unfinished"
        );
    }
}

/// `n` greedy steps on `slot` from `logits` (the slot holds `pos` rows):
/// each step's (top-1, margin), and the tokens fed.
fn greedy(
    m: &mut GpuInkling2,
    slot: usize,
    mut pos: usize,
    mut logits: Vec<f32>,
    n: usize,
) -> (Vec<(usize, f32)>, Vec<u32>) {
    let v = paddock_models::inkling2::VOCAB_REAL;
    let (mut tops, mut fed) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let t = top1(&logits[..v]);
        tops.push(t);
        fed.push(t.0 as u32);
        let (out, _) = m
            .forward_mixed(&[(slot, t.0 as u32, pos as u32)], 0)
            .expect("decode");
        logits = out;
        pos += 1;
    }
    (tops, fed)
}

/// Partings between two runs, each allowed only where one side called it
/// by under `near` - the chaos band of this looped model at its own noise.
fn partings(a: &[(usize, f32)], b: &[(usize, f32)], near: f32, what: &str) -> usize {
    let mut hard = 0;
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        if x.0 != y.0 {
            let m = x.1.min(y.1);
            println!(
                "  {what} step {k}: {} (margin {:.3}) vs {} (margin {:.3})",
                x.0, x.1, y.0, y.1
            );
            if m >= near {
                hard += 1;
            }
            // the streams part from here: compare no further
            break;
        }
    }
    hard
}

#[test]
fn inkling2_conv_capture_is_the_carried_state() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_inkling2_conv_capture() {
        common::missing("pack has no slot 1014");
        return;
    }
    let det = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    };
    for (k, c, xstride) in [(1usize, 1024usize, 1088usize), (8, 2304, 2304)] {
        let rows = 40;
        let slots = 3;
        let x = det(rows * k * xstride, 7 + k as u64);
        let state = det(slots * 3 * c, 11);
        // (r0, kk, slot, out offset): deep in a run, at its third row, at
        // one row in (two old rows), at the run's start (the old state)
        let items: Vec<[u32; 4]> = vec![
            [4, 20, 1, 0],
            [10, 3, 0, (3 * c) as u32],
            [25, 1, 2, (6 * c) as u32],
            [30, 0, 1, (9 * c) as u32],
        ];
        let dst = 5;
        let n_out = dst + 12 * c;
        let dx = exec.to_device(&x).unwrap();
        let ds = exec.to_device(&state).unwrap();
        let flat: Vec<u32> = items.iter().flatten().copied().collect();
        let di = exec.to_device_u32(&flat).unwrap();
        let mut out = exec.to_device(&vec![0.0f32; n_out]).unwrap();
        exec.inkling2_conv_capture((&dx, k, xstride), c, &ds, (&di, items.len()), &mut out, dst)
            .unwrap();
        let got = exec.to_host(&out).unwrap();
        let mut want = vec![0.0f32; n_out];
        for it in &items {
            let (r0, kk, slot, off) =
                (it[0] as usize, it[1] as i64, it[2] as usize, it[3] as usize);
            for j in 0..3usize {
                let t = kk - 3 + j as i64;
                for ch in 0..c {
                    let v = if t >= 0 {
                        let row = r0 + t as usize;
                        if k == 1 {
                            x[row * xstride + ch]
                        } else {
                            (0..k).fold(0.0f32, |acc, q| acc + x[(row * k + q) * xstride + ch])
                        }
                    } else {
                        state[slot * 3 * c + (t + 3) as usize * c + ch]
                    };
                    want[off + dst + j * c + ch] = v;
                }
            }
        }
        let differ = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differ, 0,
            "k = {k}: {differ} floats differ from the definition"
        );
    }
}

#[test]
fn inkling2_prefix_resume() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_conv_capture() {
        common::missing("pack has no Inkling 2 prefix-cache ops (slots 977-1014)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(gold) = std::env::var_os("INKLING2_GOLDENS") else {
        common::missing("INKLING2_GOLDENS");
        return;
    };
    let g = paddock_models::safetensors::SafetensorsFile::open(std::path::Path::new(&gold))
        .expect("goldens");
    let seq = |s: usize, n: usize| -> Vec<u32> {
        let (_, ib) = g.bytes(&format!("s{s}.ids")).expect("ids");
        let ids: Vec<u32> = i32s(ib).into_iter().map(|t| t as u32).collect();
        ids[..n.min(ids.len())].to_vec()
    };
    // a 3000-token document (both its cuts land inside the one pass) and
    // another text for the follow-ups
    let a = seq(2, 3000);
    let other = seq(0, 300);
    assert!(a.len() == 3000 && other.len() == 300, "goldens too short");
    let mut m = GpuInkling2::load(exec.clone(), &dir, 16384, 4096, 1).expect("load");
    assert!(m.enable_batch(2).expect("enable batch") >= 2);
    let capacity = m.pool_free_blocks().expect("pool");
    let near = 0.5;

    // ---- turn 1 on slot 0: the prompt lands, then a reply
    m.prefill_begin(0, a.clone()).expect("queue");
    let la = land(&mut m, 0);
    assert_eq!(m.take_prefill_reused(0), 0, "a first prompt is cold");
    let n_reply = 40;
    let (_, reply) = greedy(&mut m, 0, a.len(), la, n_reply);
    // every reply token was fed (a reply's stop token never is, so a next
    // turn re-sends exactly what the slot holds): its rolling checkpoint sits
    // at the last block boundary the reply reached
    let reply_ckpt = (a.len() + n_reply) / 16 * 16;

    // ---- turn 2 on slot 1, re-sending the reply as generated
    let mut c = a.clone();
    c.extend(&reply);
    c.extend(&other);
    m.prefill_begin(1, c.clone()).expect("queue");
    // the restore, exactly: slot 1 now holds what slot 0 held when the
    // reply's checkpoint was taken (slot 0 has not moved since)
    let (conv0, win0) = m.slot_state(0).expect("slot 0 state");
    let (conv1, win1) = m.slot_state(1).expect("slot 1 state");
    let conv_differ = conv0
        .iter()
        .zip(&conv1)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let win_differ = win0.iter().zip(&win1).filter(|(x, y)| x != y).count();
    println!(
        "  restored state: {conv_differ} of {} conv floats, {win_differ} of {} window bytes differ",
        conv0.len(),
        win0.len()
    );
    assert!(
        conv_differ == 0 && win_differ == 0 && win0.len() == win1.len(),
        "the restore is not the checkpointed state"
    );
    let lc_res = land(&mut m, 1);
    let got = m.take_prefill_reused(1);
    println!(
        "turn 2 (the reply re-sent): resumed at {got} of {}",
        c.len()
    );
    assert_eq!(
        got, reply_ckpt,
        "the next turn resumes at the reply's checkpoint"
    );
    let (res_steps, _) = greedy(&mut m, 1, c.len(), lc_res.clone(), 16);
    // the same prompt cold on slot 1
    let lc_cold = m.prefill_slot(1, &c).expect("cold prefill");
    let (cold_steps, _) = greedy(&mut m, 1, c.len(), lc_cold.clone(), 16);
    let v = paddock_models::inkling2::VOCAB_REAL;
    let (r, k) = (top1(&lc_res[..v]), top1(&lc_cold[..v]));
    let dmax = lc_res[..v]
        .iter()
        .zip(&lc_cold[..v])
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  last logits: resumed {} ({:.3}) cold {} ({:.3}), max |diff| {dmax:.3}",
        r.0, r.1, k.0, k.1
    );
    let hard2 = partings(&res_steps, &cold_steps, near, "turn 2");

    // ---- turn 3 on slot 1, the reply dropped: resumes at the prompt's end
    let mut d = a.clone();
    d.extend(&other);
    m.prefill_begin(1, d.clone()).expect("queue");
    let ld_res = land(&mut m, 1);
    let got = m.take_prefill_reused(1);
    println!(
        "turn 3 (the reply dropped): resumed at {got} of {}",
        d.len()
    );
    assert_eq!(
        got,
        a.len() / 16 * 16,
        "a reply-less next turn resumes at the prompt's end"
    );
    let (res3, _) = greedy(&mut m, 1, d.len(), ld_res, 16);
    let ld_cold = m.prefill_slot(1, &d).expect("cold prefill");
    let (cold3, _) = greedy(&mut m, 1, d.len(), ld_cold, 16);
    let hard3 = partings(&res3, &cold3, near, "turn 3");

    assert_eq!(
        hard2 + hard3,
        0,
        "a resumed prompt parted from cold away from a near-tie"
    );

    // ---- every block comes back once the slots let go
    m.release_inactive_slots(&[false, false]);
    assert_eq!(
        m.pool_free_blocks().expect("pool"),
        capacity,
        "blocks leaked past the cache"
    );
}
