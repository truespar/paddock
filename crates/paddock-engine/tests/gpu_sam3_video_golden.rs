//! SAM 3's video path, stage by stage, against Meta's own video predictor
//! (facebookresearch/sam3 pinned at 2345a4ad10) on the A6000.
//!
//! The reference is `harness/make_video_stage_goldens.py`: Meta's predictor
//! run in full precision on clip 0001 ("person", 270 frames), every tracker
//! stage call on frames 0-23 dumped with its inputs and outputs, and the same
//! call re-run on the same inputs under Meta's real bf16 autocast. Each stage
//! here is fed Meta's fp32 inputs and must land no further from Meta's fp32
//! output than that bf16 run of the same call did - Meta's own precision
//! class is the bar, one call at a time.
//!
//! Needs `SAM3_GOLDENS` (the goldens root; its `video_stages/0001-person`)
//! and the checkpoint (`SAM3_DIR`). Skips, saying so, without them.

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::sam3::{GpuSam3MemEnc, MEM_DIM};
use paddock_models::safetensors::{SafetensorsFile, StDtype};

fn say(msg: &str) {
    eprintln!("{msg}");
}

/// The stage dumps of one clip, if the goldens are there.
fn stages(clip: &str) -> Option<(PathBuf, serde_json::Value)> {
    let root = std::env::var_os("SAM3_GOLDENS").map(PathBuf::from)?;
    let dir = root.join("video_stages").join(clip);
    let index = std::fs::read(dir.join("index.json")).ok()?;
    let v = serde_json::from_slice(&index).expect("parse the stage index");
    Some((dir, v))
}

/// A golden tensor as f32 (the bf16 twins widened exactly).
fn tensor(path: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let f = SafetensorsFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (t, b) = f
        .bytes(name)
        .unwrap_or_else(|| panic!("{}: no tensor {name}", path.display()));
    let v = match t.dtype {
        StDtype::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        StDtype::Bf16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        other => panic!("{name}: {other:?}"),
    };
    (t.shape.clone(), v)
}

/// Object `b`'s `[c][s][s]` slice of an NCHW tensor, as `[s*s][c]` rows.
fn nchw_rows(v: &[f32], b: usize, c: usize, s: usize) -> Vec<f32> {
    let base = b * c * s * s;
    let mut out = vec![0f32; c * s * s];
    for ch in 0..c {
        for p in 0..s * s {
            out[p * c + ch] = v[base + ch * s * s + p];
        }
    }
    out
}

/// Relative RMS distance of `a` from `b`.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

/// The memory encoder (`_encode_new_memory`) on every dumped call: the
/// frame's tracker feature and the objects' masks in (1152^2 sigmoid on the
/// tracked frames, 1008^2 binarized where objects are born or
/// re-conditioned), the 64-channel memory out, an object at a time against
/// Meta fp32 with Meta's bf16 run of the same call as the bar; and the
/// memory's position table against Meta's.
#[test]
fn memory_encoder_meets_metas_own_bf16_bar() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some((gold, index)) = stages("0001-person") else {
        common::missing("SAM3_GOLDENS/video_stages/0001-person (Meta's stage dumps) not there");
        return;
    };
    let calls: Vec<&serde_json::Value> = index["calls"]
        .as_array()
        .expect("calls")
        .iter()
        .filter(|c| c["stage"] == "memenc")
        .collect();
    assert!(!calls.is_empty(), "no memory-encoder calls in the dumps");
    let mut enc = GpuSam3MemEnc::load_dir(exec.clone(), &dir, 4).expect("load the memory encoder");
    say(&format!(
        "sam3 memory encoder: weights {} MiB, {} calls",
        enc.weight_bytes() >> 20,
        calls.len()
    ));
    let px = 72 * 72;
    let mut failures = Vec::new();
    let mut worst = f64::INFINITY;
    for (i, call) in calls.iter().enumerate() {
        let file = gold.join(call["file"].as_str().expect("file"));
        let frame = call["frame"].as_u64().expect("frame");
        let binarize = call["meta"]["in.is_mask_from_pts"]
            .as_bool()
            .expect("is_mask_from_pts");
        // the feature arrives [5184][objects][256], one copy an object
        let (fs, feat) = tensor(&file, "in.current_vision_feats.2");
        let nb = fs[1];
        let pix: Vec<f32> = (0..px)
            .flat_map(|r| feat[(r * nb) * 256..(r * nb + 1) * 256].iter().copied())
            .collect();
        for b in 1..nb {
            let same = (0..px).all(|r| {
                feat[(r * nb + b) * 256..(r * nb + b + 1) * 256] == pix[r * 256..(r + 1) * 256]
            });
            assert!(
                same,
                "{file:?}: object {b}'s feature differs from object 0's"
            );
        }
        let (ms, masks) = tensor(&file, "in.pred_masks_high_res");
        let side = ms[3];
        let (_, scores) = tensor(&file, "in.object_score_logits");
        let appearing: Vec<bool> = scores.iter().map(|s| *s > 0.0).collect();
        let pix_d = exec.to_device(&pix).expect("feature");
        let masks_d = exec.to_device(&masks).expect("masks");
        enc.encode(&pix_d, &masks_d, side, binarize, &appearing)
            .expect("encode");
        let ours = exec
            .to_host_len(enc.memory(), nb * px * MEM_DIM)
            .expect("memory");
        let (_, g32) = tensor(&file, "out.0");
        let (_, g16) = tensor(&file, "bf16.0");
        let mut line = Vec::new();
        for b in 0..nb {
            let g = nchw_rows(&g32, b, MEM_DIM, 72);
            let t = nchw_rows(&g16, b, MEM_DIM, 72);
            let o = &ours[b * px * MEM_DIM..(b + 1) * px * MEM_DIM];
            let (e, bar) = (rel(o, &g), rel(&t, &g));
            worst = worst.min(bar / e.max(1e-300));
            line.push(format!("{e:.2e}/{bar:.2e}"));
            if e > bar {
                failures.push(format!(
                    "call {i} (frame {frame}, object {b}): {e:.3e} > Meta bf16 {bar:.3e}"
                ));
            }
        }
        say(&format!(
            "  memenc {i:2} frame {frame:2} {side}^2{} : ours/Meta-bf16 {}",
            if binarize { " binarized" } else { "" },
            line.join("  ")
        ));
        if i == 0 {
            let (_, gpos) = tensor(&file, "out.1.0");
            let g = nchw_rows(&gpos, 0, MEM_DIM, 72);
            let ours = exec.to_host_len(enc.pos(), px * MEM_DIM).expect("pos");
            let d = ours
                .iter()
                .zip(&g)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            say(&format!(
                "  memory pos table: worst |ours - Meta's| {d:.3e}"
            ));
            if d > 1e-5 {
                failures.push(format!("pos table off by {d:.3e}"));
            }
        }
    }
    say(&format!(
        "  memory encoder: at worst {worst:.1}x under Meta's bf16 distance"
    ));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Object `b`'s rows of a `[seq][objects][dim]` tensor (Meta's sequence-first
/// layout), as `[seq][dim]`.
fn seq_rows(v: &[f32], seq: usize, nb: usize, b: usize, dim: usize) -> Vec<f32> {
    (0..seq)
        .flat_map(|r| {
            v[(r * nb + b) * dim..(r * nb + b + 1) * dim]
                .iter()
                .copied()
        })
        .collect()
}

/// The memory attention (`transformer.encoder`) on every dumped call: the
/// frame's tracker feature and one object's bank (its memory frames and
/// object pointers, with their positions) in, the conditioned 72^2 feature
/// out, an object at a time against Meta fp32 with Meta's bf16 run of the
/// same call as the bar; and the input's position table against Meta's.
#[test]
fn memory_attention_meets_metas_own_bf16_bar() {
    use paddock_engine::gpu_model::sam3::GpuSam3MemAttn;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some((gold, index)) = stages("0001-person") else {
        common::missing("SAM3_GOLDENS/video_stages/0001-person (Meta's stage dumps) not there");
        return;
    };
    let calls: Vec<&serde_json::Value> = index["calls"]
        .as_array()
        .expect("calls")
        .iter()
        .filter(|c| c["stage"] == "memattn")
        .collect();
    assert!(!calls.is_empty(), "no memory-attention calls in the dumps");
    let mut ma = GpuSam3MemAttn::load_dir(exec.clone(), &dir).expect("load the memory attention");
    say(&format!(
        "sam3 memory attention: weights {} MiB, {} calls",
        ma.weight_bytes() >> 20,
        calls.len()
    ));
    let (px, d) = (72 * 72, 256);
    let mut failures = Vec::new();
    let mut worst = f64::INFINITY;
    for (i, call) in calls.iter().enumerate() {
        let file = gold.join(call["file"].as_str().expect("file"));
        let frame = call["frame"].as_u64().expect("frame");
        let nptr = call["meta"]["in.num_obj_ptr_tokens"]
            .as_u64()
            .expect("num_obj_ptr_tokens") as usize;
        let (ss, src) = tensor(&file, "in.src.0");
        let nb = ss[1];
        let (ps, prompt) = tensor(&file, "in.prompt");
        let nk = ps[0];
        let (_, ppos) = tensor(&file, "in.prompt_pos");
        let (_, g32) = tensor(&file, "out.memory");
        let (_, g16) = tensor(&file, "bf16.memory");
        if i == 0 {
            let (_, spos) = tensor(&file, "in.src_pos.0");
            let theirs = seq_rows(&spos, px, nb, 0, d);
            let ours = exec.to_host_len(ma.pos01(), px * d).expect("pos");
            let dmax = ours
                .iter()
                .zip(&theirs)
                .map(|(a, b)| (a / 0.1 - b).abs())
                .fold(0f32, f32::max);
            say(&format!(
                "  neck pos table: worst |ours - Meta's| {dmax:.3e}"
            ));
            if dmax > 1e-5 {
                failures.push(format!("neck pos table off by {dmax:.3e}"));
            }
        }
        let mut line = Vec::new();
        for b in 0..nb {
            let s = seq_rows(&src, px, nb, b, d);
            let m = seq_rows(&prompt, nk, nb, b, 64);
            let mp = seq_rows(&ppos, nk, nb, b, 64);
            let kin: Vec<f32> = m.iter().zip(&mp).map(|(a, p)| a + p).collect();
            let src_d = exec.to_device(&s).expect("src");
            let kin_d = exec.to_device_f16(&kin, "kin").expect("kin");
            let v_d = exec.to_device_f16(&m, "vmem").expect("vmem");
            ma.run(&src_d, &kin_d, &v_d, nk, nk - nptr)
                .expect("memory attention");
            let ours = exec.to_host_len(ma.output(), px * d).expect("out");
            let g = seq_rows(&g32, px, nb, b, d);
            let t = seq_rows(&g16, px, nb, b, d);
            let (e, bar) = (rel(&ours, &g), rel(&t, &g));
            worst = worst.min(bar / e.max(1e-300));
            line.push(format!("{e:.2e}/{bar:.2e}"));
            if e > bar {
                failures.push(format!(
                    "call {i} (frame {frame}, object {b}): {e:.3e} > Meta bf16 {bar:.3e}"
                ));
            }
        }
        say(&format!(
            "  memattn {i:2} frame {frame:2} {nk} keys ({nptr} pointer): ours/Meta-bf16 {}",
            line.join("  ")
        ));
    }
    say(&format!(
        "  memory attention: at worst {worst:.1}x under Meta's bf16 distance"
    ));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The tracker's SAM heads (`_forward_sam_heads`) on every dumped call: a
/// tracked frame's memory-conditioned feature with no prompt (three
/// candidates, the best by predicted IoU), or a newborn object's frame with
/// its downsampled mask as the prompt (one) - the low-res masks (-1024 where
/// the object is gone), the predicted IoUs, the object pointer and the
/// object-score logit, an object at a time against Meta fp32 with Meta's bf16
/// run of the same call as the bar.
#[test]
fn tracker_heads_meet_metas_own_bf16_bar() {
    use paddock_engine::gpu_model::sam3::{GpuSam3Pvs, PvsFeatures, PvsMask, PvsPrompt};
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some((gold, index)) = stages("0001-person") else {
        common::missing("SAM3_GOLDENS/video_stages/0001-person (Meta's stage dumps) not there");
        return;
    };
    let calls: Vec<&serde_json::Value> = index["calls"]
        .as_array()
        .expect("calls")
        .iter()
        .filter(|c| c["stage"] == "samheads")
        .collect();
    assert!(!calls.is_empty(), "no SAM-head calls in the dumps");
    let mut heads = GpuSam3Pvs::load_dir(exec.clone(), &dir).expect("load the tracker heads");
    let (side, plane) = (288usize, 288 * 288);
    let mut failures = Vec::new();
    let mut worst = f64::INFINITY;
    for (i, call) in calls.iter().enumerate() {
        let file = gold.join(call["file"].as_str().expect("file"));
        let frame = call["frame"].as_u64().expect("frame");
        let meta = &call["meta"];
        assert!(
            meta["in.point_inputs"].is_null(),
            "call {i}: clicks in a dump"
        );
        let multimask = meta["in.multimask_output"].as_bool().expect("multimask");
        let (fs, feat) = tensor(&file, "in.backbone_features");
        let nb = fs[0];
        let (_, s0) = tensor(&file, "in.high_res_features.0");
        let (_, s1) = tensor(&file, "in.high_res_features.1");
        let mask_in = SafetensorsFile::open(&file)
            .expect("dump")
            .bytes("in.mask_inputs")
            .map(|_| tensor(&file, "in.mask_inputs").1);
        let (gs, g32) = tensor(&file, "out.0");
        let m_out = gs[1];
        let (_, g16) = tensor(&file, "bf16.0");
        let (_, iou32) = tensor(&file, "out.2");
        let (_, iou16) = tensor(&file, "bf16.2");
        let (_, p32) = tensor(&file, "out.5");
        let (_, p16) = tensor(&file, "bf16.5");
        let (_, o32) = tensor(&file, "out.6");
        let (_, o16) = tensor(&file, "bf16.6");
        let mut line = Vec::new();
        for b in 0..nb {
            let f_d = exec.to_device(&nchw_rows(&feat, b, 256, 72)).expect("feat");
            let s0_d = exec.to_device(&nchw_rows(&s0, b, 32, side)).expect("s0");
            let s1_d = exec
                .to_device(&nchw_rows(&s1, b, 64, side / 2))
                .expect("s1");
            let mplane = mask_in
                .as_ref()
                .map(|m| m[b * plane..(b + 1) * plane].to_vec());
            let prompt = PvsPrompt {
                points: Vec::new(),
                bbox: None,
                mask: mplane.as_deref().map(PvsMask::Host),
                multimask,
            };
            let feats = PvsFeatures {
                feat: &f_d,
                no_memory: false,
                s0: &s0_d,
                s1: &s1_d,
                track: true,
            };
            let res = heads.predict_on(feats, &prompt).expect("heads");
            let logits = exec.to_host_len(heads.logits(), plane * 4).expect("logits");
            let ptr = exec.to_host_len(heads.pointer(), 256).expect("pointer");
            // Meta's channels: masks 1..3 with three candidates, its choice with one
            let ks: Vec<usize> = if multimask {
                (1..=m_out).collect()
            } else {
                vec![res.candidates[0].0]
            };
            let ours: Vec<f32> = ks
                .iter()
                .flat_map(|&k| (0..plane).map(move |p| (p, k)))
                .map(|(p, k)| logits[p * 4 + k])
                .collect();
            let g = &g32[b * m_out * plane..(b + 1) * m_out * plane];
            let t = &g16[b * m_out * plane..(b + 1) * m_out * plane];
            let (e, bar) = (rel(&ours, g), rel(t, g));
            let pe = rel(&ptr, &p32[b * 256..(b + 1) * 256]);
            let pbar = rel(&p16[b * 256..(b + 1) * 256], &p32[b * 256..(b + 1) * 256]);
            let di = ks
                .iter()
                .enumerate()
                .map(|(j, &k)| (res.iou[k] - iou32[b * m_out + j]).abs())
                .fold(0f32, f32::max);
            let dbar = (0..m_out)
                .map(|j| (iou16[b * m_out + j] - iou32[b * m_out + j]).abs())
                .fold(0f32, f32::max)
                .max(0.002);
            let dl = (res.object_logit - o32[b]).abs();
            let lbar = (o16[b] - o32[b]).abs().max(0.02);
            worst = worst.min(bar / e.max(1e-300)).min(pbar / pe.max(1e-300));
            line.push(format!("{e:.1e}/{bar:.1e} ptr {pe:.1e}/{pbar:.1e}"));
            if e > bar || pe > pbar || di > dbar || dl > lbar {
                failures.push(format!(
                    "call {i} (frame {frame}, object {b}): masks {e:.3e}/{bar:.3e}, pointer \
                     {pe:.3e}/{pbar:.3e}, iou |d| {di:.4}/{dbar:.4}, object logit |d| \
                     {dl:.3}/{lbar:.3}"
                ));
            }
        }
        say(&format!(
            "  samheads {i:2} frame {frame:2} {}: {}",
            if multimask { "track " } else { "birth " },
            line.join("  ")
        ));
    }
    say(&format!(
        "  tracker heads: at worst {worst:.1}x under Meta's bf16 distance"
    ));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Half the f16 step at `x`: what rounding to f16 may move it.
fn f16_half_step(x: f32) -> f32 {
    let h = half::f16::from_f32(x.abs());
    let next = half::f16::from_bits(h.to_bits() + 1);
    0.5 * (next.to_f32() - h.to_f32())
}

/// The memory bank (`_prepare_memory_conditioned_features`'s assembly)
/// replayed over the dumped frames: the births, every tracked frame's
/// pointers and scores, the re-conditionings and every memory write, fed in
/// the order Meta made them with Meta's own stage outputs as the values. On
/// each tracked frame the bank's choice of memory frames and pointers, their
/// temporal positions and the two f16 planes the memory attention reads must
/// be the `prompt` / `prompt_pos` Meta handed it. The values are Meta's, so
/// the value plane must match exactly. The key input `M + pos` carries our
/// position tables (the sine table, the pointers' sine + projection), fp32
/// math of their own: it must land within f16's rounding of Meta's fp32 sum
/// plus `POS_TOL` parts of the position's own size (at least 1). Counting
/// f16 steps instead blows up near zero; the pointer positions run to ~5
/// and a 256-term fp32 dot product is a few 1e-6 off the exact value there,
/// Meta's included - two orders under bf16 still. On a few frames the
/// memory attention then runs on the bank as built and is held to Meta's
/// bf16 bar, as the attention's own gate holds it on Meta's inputs.
#[test]
fn memory_bank_builds_metas_prompt() {
    use half::f16;
    use paddock_engine::gpu_model::sam3::{GpuSam3BankKv, GpuSam3MemAttn, Sam3Bank};
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some((gold, index)) = stages("0001-person") else {
        common::missing("SAM3_GOLDENS/video_stages/0001-person (Meta's stage dumps) not there");
        return;
    };
    let last = index["dumped_frames"][1].as_u64().expect("dumped_frames") as u32;
    let calls: Vec<&serde_json::Value> = index["calls"].as_array().expect("calls").iter().collect();
    let num_frames = calls
        .iter()
        .find(|c| c["stage"] == "memcond")
        .and_then(|c| c["meta"]["in.num_frames"].as_u64())
        .map(|n| n as u32);
    let mut kv = GpuSam3BankKv::load_dir(exec.clone(), &dir).expect("load the bank tables");
    let mut ma = GpuSam3MemAttn::load_dir(exec.clone(), &dir).expect("load the memory attention");
    const RUN_ATTENTION: [u32; 5] = [2, 7, 16, 17, 23];
    let (px, d, w) = (72 * 72, 256, 64);
    let shape = |c: &serde_json::Value, t: &str| -> Vec<usize> {
        c["tensors"][t][0]
            .as_array()
            .unwrap_or_else(|| panic!("{}: no {t}", c["file"]))
            .iter()
            .map(|x| x.as_u64().expect("dim") as usize)
            .collect()
    };

    #[derive(PartialEq, Debug)]
    enum Phase {
        /// detection through the tracker's propagation
        Track,
        /// the update plan: re-conditionings, then the memory update
        Plan,
        /// its execution: births
        Exec,
    }
    let mut phase = Phase::Exec;
    let mut bank: Option<Sam3Bank> = None;
    let mut recond: Vec<usize> = Vec::new();
    let mut births = 0usize;
    let mut failures = Vec::new();
    const POS_TOL: f32 = 2e-6;
    let (mut checked, mut v_off, mut kin_off) = (0usize, 0usize, 0usize);
    let mut worst_attn = f64::INFINITY;
    // the worst key-input excess over f16 rounding, memory rows and pointer
    // rows apart: (excess, frame, object, row, Meta's sum, ours)
    let mut worst_k = [(0f32, 0u32, 0usize, 0usize, 0f32, 0f32); 2];
    for (ci, call) in calls.iter().enumerate() {
        let stage = call["stage"].as_str().expect("stage");
        let f = call["frame"].as_u64().expect("frame") as u32;
        if f > last {
            break;
        }
        let file = gold.join(call["file"].as_str().expect("file"));
        match stage {
            "det" => {
                // a pass over the frame begins; its plan says, after the
                // fact, which tracks the detections re-condition - Meta's
                // trk_id_to_max_iou_high_conf_det, built in ascending
                // detection order, those whose tracker score is over 0.8
                phase = Phase::Track;
                let next = |st: &str| {
                    calls[ci..]
                        .iter()
                        .find(|c| c["stage"] == st)
                        .copied()
                        .unwrap_or_else(|| panic!("frame {f}: no {st} after the detection"))
                };
                let (prop, plan) = (next("prop"), next("plan"));
                assert_eq!(plan["frame"].as_u64(), Some(f as u64), "frame {f}'s plan");
                let (_, scores) = tensor(&gold.join(prop["file"].as_str().unwrap()), "out.1");
                let mut m: Vec<(u64, usize)> = plan["meta"]
                    .as_object()
                    .expect("plan meta")
                    .iter()
                    .filter_map(|(k, v)| {
                        let id = k.strip_prefix("out.0.trk_id_to_max_iou_high_conf_det.")?;
                        Some((v.as_u64()?, id.parse().ok()?))
                    })
                    .collect();
                m.sort_unstable();
                assert!(
                    m.windows(2).all(|p| p[0].0 != p[1].0),
                    "frame {f}: two tracks on one detection - the order is not recoverable"
                );
                recond = m
                    .into_iter()
                    .filter(|&(_, id)| scores.get(id).is_some_and(|s| *s > 0.8))
                    .map(|(_, id)| id)
                    .collect();
                let removed = plan["meta"]["out.0.obj_ids_newly_removed"].as_array();
                assert!(
                    removed.is_none_or(|r| r.is_empty()),
                    "frame {f}: an object removed - the replay does not cover removals"
                );
            }
            "prop" => phase = Phase::Plan,
            "plan" => {
                phase = Phase::Exec;
                let born = shape(call, "out.0.new_det_obj_ids")[0];
                if born > 0 {
                    assert!(
                        bank.is_none(),
                        "frame {f}: births into a second tracker state - the replay covers one"
                    );
                    bank = Some(Sam3Bank::new(exec.clone(), born).expect("bank"));
                    births = 0;
                }
            }
            "outputs" => {
                if let Some(b) = bank.as_mut() {
                    b.prune(f);
                }
            }
            "samheads" if call["tensors"].get("in.mask_inputs").is_none() => {
                // a tracked frame's heads (a mask-as-output pass carries its
                // mask in, and its pointer is the maskout call's)
                let b = bank.as_mut().expect("a tracked frame before any birth");
                let (_, ious) = tensor(&file, "out.2");
                let (_, ptrs) = tensor(&file, "out.5");
                let (_, logits) = tensor(&file, "out.6");
                let nb = logits.len();
                assert_eq!(nb, b.objects(), "frame {f}: objects in the heads' call");
                let m = ious.len() / nb;
                for o in 0..nb {
                    let best = ious[o * m..(o + 1) * m]
                        .iter()
                        .copied()
                        .fold(f32::MIN, f32::max);
                    let ptr = exec.to_device(&ptrs[o * d..(o + 1) * d]).expect("pointer");
                    b.track(f, o, &ptr, logits[o], best).expect("track");
                }
            }
            "maskout" => {
                let b = bank.as_mut().expect("a mask pass before any birth");
                let (_, ptr) = tensor(&file, "out.5");
                let obj = match phase {
                    Phase::Plan => {
                        assert!(
                            !recond.is_empty(),
                            "frame {f}: a re-conditioning nobody planned"
                        );
                        recond.remove(0)
                    }
                    Phase::Exec => {
                        births += 1;
                        births - 1
                    }
                    Phase::Track => panic!("frame {f}: a mask pass during propagation"),
                };
                let ptr = exec.to_device(&ptr[..d]).expect("pointer");
                b.condition(f, obj, &ptr).expect("condition");
            }
            "memenc" => {
                let b = bank.as_mut().expect("a memory before any birth");
                let (ms, mem) = tensor(&file, "out.0");
                assert_eq!(ms[0], b.objects(), "frame {f}: objects in the memory call");
                for o in 0..ms[0] {
                    let rows = exec.to_device(&nchw_rows(&mem, o, w, 72)).expect("memory");
                    b.set_memory(f, o, &rows, 0).expect("set memory");
                }
            }
            "memattn" => {
                let b = bank.as_ref().expect("a tracked frame before any birth");
                let plan = b.plan(f, num_frames).expect("plan");
                let (nk, nrope) = plan.keys();
                let ps = shape(call, "in.prompt");
                let nptr = call["meta"]["in.num_obj_ptr_tokens"]
                    .as_u64()
                    .expect("num_obj_ptr_tokens") as usize;
                let mems: Vec<String> = plan
                    .memories
                    .iter()
                    .map(|&(c, g, r)| format!("{}{g}@{r}", if c { "c" } else { "" }))
                    .collect();
                say(&format!(
                    "  frame {f:2}: {} memory frames [{}], {} pointers, {nk} keys (Meta {})",
                    plan.memories.len(),
                    mems.join(" "),
                    plan.pointers.len(),
                    ps[0]
                ));
                if nk != ps[0] || nk - nrope != nptr || ps[1] != b.objects() {
                    failures.push(format!(
                        "frame {f}: {nk} keys, {} pointer tokens for {} objects; Meta {} keys, \
                         {nptr} pointer tokens for {}",
                        nk - nrope,
                        b.objects(),
                        ps[0],
                        ps[1]
                    ));
                    continue;
                }
                let (_, prompt) = tensor(&file, "in.prompt");
                let (_, ppos) = tensor(&file, "in.prompt_pos");
                let attn = RUN_ATTENTION.contains(&f);
                let (src, g32, g16) = if attn {
                    (
                        tensor(&file, "in.src.0").1,
                        tensor(&file, "out.memory").1,
                        tensor(&file, "bf16.memory").1,
                    )
                } else {
                    Default::default()
                };
                let nb = ps[1];
                let mut line = Vec::new();
                for o in 0..nb {
                    let (k2, r2) = kv.fill(b, &plan, o).expect("fill");
                    assert_eq!((k2, r2), (nk, nrope));
                    let kin = exec.to_host_f16_len(kv.kin(), nk * w).expect("kin");
                    let v = exec.to_host_f16_len(kv.vmem(), nk * w).expect("v");
                    let m = seq_rows(&prompt, nk, nb, o, w);
                    let mp = seq_rows(&ppos, nk, nb, o, w);
                    let (mut v_bad, mut k_step, mut k_far) = (0usize, 0usize, 0f32);
                    for i in 0..nk * w {
                        if v[i] != f16::from_f32(m[i]) {
                            v_bad += 1;
                        }
                        let theirs = m[i] + mp[i];
                        let ours = kin[i].to_f32();
                        k_step += (kin[i] != f16::from_f32(theirs)) as usize;
                        // in units of the position's size
                        let excess =
                            ((ours - theirs).abs() - f16_half_step(theirs)) / mp[i].abs().max(1.0);
                        k_far = k_far.max(excess);
                        let kind = (i >= nrope * w) as usize;
                        if excess > worst_k[kind].0 {
                            worst_k[kind] = (excess, f, o, i / w, theirs, ours);
                        }
                    }
                    checked += nk * w;
                    v_off += v_bad;
                    kin_off += k_step;
                    let mut cell = format!("v {v_bad} off, kin {k_step} rounded apart");
                    if v_bad > 0 || k_far > POS_TOL {
                        failures.push(format!(
                            "frame {f}, object {o}: {v_bad} value entries differ, key input \
                             {k_far:.2e} of the position past f16 rounding at worst"
                        ));
                    }
                    if attn {
                        let s = seq_rows(&src, px, nb, o, d);
                        let src_d = exec.to_device(&s).expect("src");
                        ma.run(&src_d, kv.kin(), kv.vmem(), nk, nrope)
                            .expect("memory attention");
                        let ours = exec.to_host_len(ma.output(), px * d).expect("out");
                        let g = seq_rows(&g32, px, nb, o, d);
                        let t = seq_rows(&g16, px, nb, o, d);
                        let (e, bar) = (rel(&ours, &g), rel(&t, &g));
                        worst_attn = worst_attn.min(bar / e.max(1e-300));
                        cell.push_str(&format!(", attention {e:.2e}/{bar:.2e}"));
                        if e > bar {
                            failures.push(format!(
                                "frame {f}, object {o}: attention on the bank {e:.3e} > Meta \
                                 bf16 {bar:.3e}"
                            ));
                        }
                    }
                    line.push(cell);
                }
                say(&format!("            {}", line.join("; ")));
            }
            _ => {}
        }
    }
    assert!(checked > 0, "no tracked frames in the dumps");
    for (kind, (e, f, o, r, theirs, ours)) in ["memory", "pointer"].iter().zip(worst_k) {
        say(&format!(
            "  key input, {kind} rows: at worst {e:.2e} of the position past f16 rounding \
             (frame {f}, object {o}, row {r}: Meta {theirs:.6e}, ours {ours:.6e})"
        ));
    }
    say(&format!(
        "  memory bank: {checked} entries, {v_off} values off, {kin_off} key-input entries a \
         f16 step off ({:.2e}); attention on the bank at worst {worst_attn:.1}x under Meta's \
         bf16 distance",
        kin_off as f64 / checked as f64
    ));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A golden tensor's raw bytes and shape, whatever its dtype (the dumps'
/// int64 ids, bool masks, uint8 packed bits).
fn raw(path: &Path, name: &str) -> Option<(Vec<usize>, Vec<u8>)> {
    let f = SafetensorsFile::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (t, b) = f.bytes(name)?;
    Some((t.shape.clone(), b.to_vec()))
}

/// An integer golden tensor, int64 or int32 (Meta's dumps carry both), as
/// i64; empty when absent.
fn ints(path: &Path, name: &str) -> Vec<i64> {
    let Some((shape, b)) = raw(path, name) else {
        return Vec::new();
    };
    let n: usize = shape.iter().product();
    match b.len().checked_div(n) {
        Some(8) => b
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes(*c))
            .collect(),
        Some(4) => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as i64)
            .collect(),
        _ if n == 0 => Vec::new(),
        _ => panic!("{name}: {} bytes for {n} integers", b.len()),
    }
}

/// A JSON list of integers from a dump's metadata (empty when absent).
fn meta_ints(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default()
}

/// Meta's video decisions replayed over all 270 frames of the clip from
/// Meta's own per-frame inputs: each frame's detections (after its NMS) and
/// the tracker's propagated masks and object scores go into our planner and
/// mask ops, and every decision must be Meta's - the births and their ids,
/// the matches, the unmatched and occluded tracks, the hot-start
/// bookkeeping, which tracks re-condition, the occlusion suppression - then
/// the frame's built masks at the video's size, and finally, through the
/// 15-frame hot-start buffer, the outputs Meta yielded (ids, probabilities,
/// boxes, masks) in the fp32 decision golden. The inputs are Meta's, so the
/// decisions must match exactly; the masks are our bilinear upsample of
/// Meta's logits, held to a handful of pixels a frame.
#[test]
fn video_decisions_follow_metas_plan() {
    use std::collections::VecDeque;

    use paddock_engine::gpu_model::sam3::{
        GpuSam3VideoMasks, HOTSTART_DELAY, LOW_PX, Sam3FrameRecord, Sam3MaskSource, Sam3Plan,
        Sam3PlanIn, Sam3VideoPlanner,
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((gold, index)) = stages("0001-person") else {
        common::missing("SAM3_GOLDENS/video_stages/0001-person (Meta's stage dumps) not there");
        return;
    };
    let root = PathBuf::from(std::env::var_os("SAM3_GOLDENS").expect("SAM3_GOLDENS"));
    let decisions = root.join("video/0001/person.fp32.safetensors");
    let calls: Vec<&serde_json::Value> = index["calls"].as_array().expect("calls").iter().collect();
    let last_frame = calls
        .iter()
        .filter_map(|c| c["frame"].as_u64())
        .max()
        .expect("frames") as u32;
    let (h, w) = (720usize, 1280usize);
    let mut masks = GpuSam3VideoMasks::new(exec.clone()).expect("video masks");
    let mut planner = Sam3VideoPlanner::new();

    // the current pass: its detections and its propagated tracks
    let mut dets: (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    let mut trks: (Vec<f32>, Vec<f32>) = (Vec::new(), Vec::new());
    let mut plan: Option<Sam3Plan> = None;
    let mut held: VecDeque<(Sam3FrameRecord, Vec<Vec<f32>>)> = VecDeque::new();
    let mut stream = false;
    let mut failures: Vec<String> = Vec::new();
    let (mut plans, mut frames_out, mut recond_total) = (0usize, 0usize, 0usize);
    let (mut built_px_off, mut built_worst) = (0usize, 0usize);
    let (mut out_px_off, mut out_worst, mut box_worst) = (0usize, 0usize, 0f32);
    let upload = |v: &[f32]| {
        exec.to_device(if v.is_empty() { &[0.0] } else { v })
            .expect("planes")
    };

    let yield_frame = |rec: &Sam3FrameRecord,
                       planes: &[Vec<f32>],
                       planner: &Sam3VideoPlanner,
                       masks: &mut GpuSam3VideoMasks,
                       failures: &mut Vec<String>|
     -> (usize, usize, f32) {
        let f = rec.frame;
        let flat: Vec<f32> = planes.iter().flatten().copied().collect();
        let dev = upload(&flat);
        let objects: Vec<_> = rec
            .objects
            .iter()
            .enumerate()
            .map(|(i, o)| (o.id, o.prob, o.tracker_prob, &dev, i * LOW_PX))
            .collect();
        let hidden = |id: i64| planner.hidden().contains(&id);
        let out = masks.outputs(&objects, &hidden, h, w).expect("outputs");
        let key = |s: &str| format!("f{f:04}.{s}");
        let ids = ints(&decisions, &key("ids"));
        let (_, probs) = if ids.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            tensor(&decisions, &key("probs"))
        };
        let ours: Vec<i64> = out.iter().map(|o| o.id).collect();
        if ours != ids {
            failures.push(format!("frame {f}: outputs {ours:?}, Meta {ids:?}"));
            return (0, 0, 0.0);
        }
        if ids.is_empty() {
            return (0, 0, 0.0);
        }
        let (_, boxes) = tensor(&decisions, &key("boxes_xywh"));
        let (_, packed) = raw(&decisions, &key("masks_packed")).expect("masks");
        let row = packed.len() / ids.len();
        let (mut off, mut worst, mut bw) = (0usize, 0usize, 0f32);
        for (i, o) in out.iter().enumerate() {
            if o.prob != probs[i] {
                failures.push(format!(
                    "frame {f}, object {}: probability {} vs Meta {}",
                    o.id, o.prob, probs[i]
                ));
            }
            for k in 0..4 {
                bw = bw.max((o.bbox_xywh[k] - boxes[i * 4 + k]).abs());
            }
            let m = masks.read_mask(i).expect("mask");
            let mut n = 0usize;
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let theirs = (packed[i * row + p / 8] >> (7 - p % 8)) & 1;
                    n += (theirs != m[x * h + y]) as usize;
                }
            }
            off += n;
            worst = worst.max(n);
        }
        (off, worst, bw)
    };

    for (ci, call) in calls.iter().enumerate() {
        let stage = call["stage"].as_str().expect("stage");
        let f = call["frame"].as_u64().expect("frame") as u32;
        let file = gold.join(call["file"].as_str().expect("file"));
        match stage {
            "det" => dets = (tensor(&file, "out.scores").1, tensor(&file, "out.mask").1),
            "prop" => {
                let (ls, logits) = tensor(&file, "out.1");
                trks = if ls[0] == 0 {
                    (Vec::new(), Vec::new())
                } else {
                    (logits, tensor(&file, "out.0").1)
                };
            }
            "plan" => {
                let (nd, nt) = (dets.0.len(), trks.0.len());
                let dp = upload(&dets.1);
                let tp = upload(&trks.1);
                let dt = masks.ious((&dp, 0, nd), Some((&tp, 0, nt))).expect("ious");
                let tt = masks.ious((&tp, 0, nt), None).expect("ious");
                let nonempty: Vec<bool> = tt.area_a.iter().map(|&a| a > 0).collect();
                let p = planner.plan(&Sam3PlanIn {
                    frame: f,
                    det_scores: &dets.0,
                    det_trk_iou: &dt.iou,
                    trk_nonempty: &nonempty,
                    trk_trk_iou: &tt.iou,
                    trk_logits: &trks.0,
                });
                plans += 1;
                let m = &call["meta"];
                let mut bad = Vec::new();
                let new_dets: Vec<i64> = p.new_dets.iter().map(|&d| d as i64).collect();
                if new_dets != ints(&file, "out.0.new_det_fa_inds") {
                    bad.push(format!("births {new_dets:?}"));
                }
                if p.new_ids != ints(&file, "out.0.new_det_obj_ids") {
                    bad.push(format!("new ids {:?}", p.new_ids));
                }
                if p.unmatched != ints(&file, "out.0.unmatched_trk_obj_ids") {
                    bad.push(format!("unmatched {:?}", p.unmatched));
                }
                for (d, list) in p.det_matched.iter().enumerate() {
                    let theirs = ints(&file, &format!("out.0.det_to_matched_trk_obj_ids.{d}"));
                    if *list != theirs {
                        bad.push(format!("detection {d} matched {list:?} vs {theirs:?}"));
                    }
                }
                let mut hc: Vec<(i64, u64)> = p
                    .high_conf_det
                    .iter()
                    .map(|&(i, d)| (i, d as u64))
                    .collect();
                hc.sort_unstable();
                let mut theirs: Vec<(i64, u64)> = m
                    .as_object()
                    .expect("meta")
                    .iter()
                    .filter_map(|(k, v)| {
                        let id = k.strip_prefix("out.0.trk_id_to_max_iou_high_conf_det.")?;
                        Some((id.parse().ok()?, v.as_u64()?))
                    })
                    .collect();
                theirs.sort_unstable();
                if hc != theirs {
                    bad.push(format!("high-confidence matches {hc:?} vs {theirs:?}"));
                }
                if p.removed != meta_ints(&m["out.0.obj_ids_newly_removed"]) {
                    bad.push(format!("removed {:?}", p.removed));
                }
                if planner.tracks() != ints(&file, "out.1.obj_ids_all_gpu") {
                    bad.push(format!("tracks {:?}", planner.tracks()));
                }
                if m["out.1.max_obj_id"].as_i64() != Some(planner.max_obj_id()) {
                    bad.push(format!("max id {}", planner.max_obj_id()));
                }
                let removed: Vec<i64> = planner.hidden().iter().copied().collect();
                if removed != meta_ints(&m["out.1.rank0_metadata.removed_obj_ids"]) {
                    bad.push(format!("removed set {removed:?}"));
                }
                for (k, v) in m.as_object().expect("meta") {
                    let id = |p: &str| k.strip_prefix(p).and_then(|s| s.parse::<i64>().ok());
                    if let Some(id) = id("out.1.obj_id_to_score.") {
                        let ours = planner.score(id).unwrap_or(f32::NAN);
                        if v.as_f64() != Some(ours as f64) {
                            bad.push(format!("object {id} score {ours} vs {v}"));
                        }
                    } else if let Some(id) = id("out.1.rank0_metadata.obj_first_frame_idx.") {
                        if v.as_u64() != planner.first_frame(id).map(|x| x as u64) {
                            bad.push(format!("object {id} first frame"));
                        }
                    } else if let Some(id) = id("out.1.rank0_metadata.trk_keep_alive.") {
                        if v.as_i64() != planner.keep_alive(id).map(|x| x as i64) {
                            bad.push(format!(
                                "object {id} keep-alive {:?} vs {v}",
                                planner.keep_alive(id)
                            ));
                        }
                    } else if let Some(id) = id("out.1.rank0_metadata.unmatched_frame_inds.") {
                        let ours: Vec<i64> = planner
                            .unmatched_frames(id)
                            .unwrap_or(&[])
                            .iter()
                            .map(|&x| x as i64)
                            .collect();
                        if ours != meta_ints(v) {
                            bad.push(format!("object {id} unmatched frames"));
                        }
                    } else if let Some(pair) =
                        k.strip_prefix("out.1.rank0_metadata.overlap_pair_to_frame_inds.(")
                    {
                        let mut it = pair
                            .trim_end_matches(')')
                            .split(", ")
                            .map(|s| s.parse::<i64>().unwrap());
                        let (a, b) = (it.next().unwrap(), it.next().unwrap());
                        let ours: Vec<i64> = planner
                            .overlap_frames(a, b)
                            .unwrap_or(&[])
                            .iter()
                            .map(|&x| x as i64)
                            .collect();
                        if ours != meta_ints(v) {
                            bad.push(format!("overlap ({a}, {b}) frames {ours:?} vs {v}"));
                        }
                    }
                }
                for &id in planner.tracks() {
                    let theirs = ints(&file, &format!("out.1.obj_id_to_last_occluded.{id}"));
                    let ours = planner.last_occluded(id);
                    if !theirs.is_empty() && Some(theirs[0]) != ours {
                        bad.push(format!("object {id} last occluded {ours:?} vs {theirs:?}"));
                    }
                }
                // the re-conditionings Meta ran: its mask-as-output passes
                // between this pass's propagation and its plan
                let start = calls[..ci]
                    .iter()
                    .rposition(|c| c["stage"] == "prop")
                    .unwrap_or(0);
                let maskouts = calls[start..ci]
                    .iter()
                    .filter(|c| c["stage"] == "maskout")
                    .count();
                let dumped = calls[start..ci].iter().any(|c| c["stage"] == "memenc");
                if dumped && maskouts != p.recondition.len() {
                    bad.push(format!(
                        "{} re-conditionings vs Meta's {maskouts}",
                        p.recondition.len()
                    ));
                }
                recond_total += p.recondition.len();
                if !bad.is_empty() {
                    failures.push(format!("frame {f} plan: {}", bad.join("; ")));
                }
                // the suppressed tracks go to -10 before anything reads them
                for (t, &s) in p.suppress.iter().enumerate() {
                    if s {
                        trks.1[t * LOW_PX..(t + 1) * LOW_PX].fill(-10.0);
                    }
                }
                plan = Some(p);
            }
            "outputs" => {
                let p = plan.take().expect("a plan before the outputs");
                // the frame's planes as Meta builds them: the tracks', and a
                // birth's detection mask after the hole / sprinkle fill
                let mut planes: Vec<Vec<f32>> = Vec::new();
                for t in 0..p.tracks.len() {
                    planes.push(trks.1[t * LOW_PX..(t + 1) * LOW_PX].to_vec());
                }
                if !p.new_dets.is_empty() {
                    let born: Vec<f32> = p
                        .new_dets
                        .iter()
                        .flat_map(|&d| dets.1[d * LOW_PX..(d + 1) * LOW_PX].iter().copied())
                        .collect();
                    let mut bp = upload(&born);
                    masks.clean(&mut bp, 0, p.new_dets.len()).expect("clean");
                    let back = exec.to_host_len(&bp, born.len()).expect("born");
                    for k in 0..p.new_dets.len() {
                        planes.push(back[k * LOW_PX..(k + 1) * LOW_PX].to_vec());
                    }
                }
                let ids: Vec<i64> = p.tracks.iter().chain(&p.new_ids).copied().collect();
                let flat: Vec<f32> = planes.iter().flatten().copied().collect();
                let dev = upload(&flat);
                let list: Vec<(&cudarc::driver::CudaSlice<f32>, usize)> =
                    (0..ids.len()).map(|i| (&dev, i * LOW_PX)).collect();
                masks.render(&list, h, w).expect("render");
                for (i, id) in ids.iter().enumerate() {
                    let Some((_, theirs)) = raw(&file, &format!("out.{id}")) else {
                        failures.push(format!("frame {f}: Meta built no mask for object {id}"));
                        continue;
                    };
                    let m = masks.read_mask(i).expect("mask");
                    let mut n = 0usize;
                    for y in 0..h {
                        for x in 0..w {
                            n += ((theirs[y * w + x] != 0) as u8 != m[x * h + y]) as usize;
                        }
                    }
                    built_px_off += n;
                    built_worst = built_worst.max(n);
                }
                let rec = planner.finish(&p, &trks.0);
                // the record's planes in its object order
                let rec_planes: Vec<Vec<f32>> = rec
                    .objects
                    .iter()
                    .map(|o| match o.source {
                        Sam3MaskSource::Track(t) => planes[t].clone(),
                        Sam3MaskSource::Detection(d) => {
                            let k = p.new_dets.iter().position(|&x| x == d).expect("birth");
                            planes[p.tracks.len() + k].clone()
                        }
                    })
                    .collect();
                if !stream {
                    // frame 0's first pass is add_prompt's, outside the stream
                    stream = true;
                    continue;
                }
                held.push_back((rec, rec_planes));
                let flush = f == last_frame;
                while held.len() >= HOTSTART_DELAY as usize || (flush && !held.is_empty()) {
                    let (rec, planes) = held.pop_front().expect("held");
                    let (off, worst, bw) =
                        yield_frame(&rec, &planes, &planner, &mut masks, &mut failures);
                    out_px_off += off;
                    out_worst = out_worst.max(worst);
                    box_worst = box_worst.max(bw);
                    frames_out += 1;
                    if !flush {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    say(&format!(
        "  video decisions: {plans} plans, {recond_total} re-conditionings, {frames_out} frames out; \
         built masks {built_px_off} pixels off (worst {built_worst} an object), outputs \
         {out_px_off} pixels off (worst {out_worst}), boxes off by {box_worst:.2e}"
    ));
    assert_eq!(frames_out, last_frame as usize + 1, "frames yielded");
    // our bilinear upsample is not torch's instruction for instruction: a
    // pixel whose interpolated logit sits on 0 can land either side
    if built_worst > 16 || out_worst > 16 {
        failures.push(format!(
            "masks: {built_worst} / {out_worst} pixels off on one object at worst"
        ));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The frames in, as Meta's JPEG-folder loader makes them, on clip 0001's
/// first 24 frames from PIL's decode (`harness/make_video_frames.py`):
/// Pillow's bilinear resize to 1008^2 must be Pillow's to the byte, and the
/// patch stem's normalization of it - to_tensor's / 255, fp16 storage, fp16
/// (x - 0.5) / 0.5 - Meta's fp16 frame tensor to the bit.
#[test]
fn video_frames_match_metas_loader() {
    use half::f16;
    use paddock_engine::gpu_model::sam3::GpuSam3FrameIn;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(root) = std::env::var_os("SAM3_GOLDENS").map(PathBuf::from) else {
        common::missing("SAM3_GOLDENS not set");
        return;
    };
    let file = root.join("video_frames/0001.safetensors");
    if !file.exists() {
        common::missing(
            "SAM3_GOLDENS/video_frames/0001.safetensors (make_video_frames.py) not there",
        );
        return;
    }
    let (side, patch, win, kp) = (1008usize, 14usize, 24usize, 592usize);
    let (g, pp) = (side / patch, patch * patch);
    let mut frames = GpuSam3FrameIn::new(exec.clone(), side).expect("frames");
    let mut plane = exec.alloc_u8(side * side * 3).expect("plane");
    let mut rows = exec.alloc_f16(g * g * kp).expect("rows");
    let mut failures = Vec::new();
    let mut checked = 0;
    for f in 0.. {
        let Some((fs, rgb)) = raw(&file, &format!("f{f:04}")) else {
            break;
        };
        let Some((_, want)) = raw(&file, &format!("r{f:04}")) else {
            break;
        };
        let (h, w) = (fs[0], fs[1]);
        frames.land(&rgb, h, w, &mut plane).expect("land");
        let ours = exec.to_host_u8_len(&plane, side * side * 3).expect("plane");
        let off = ours.iter().zip(&want).filter(|(a, b)| a != b).count();
        if off > 0 {
            failures.push(format!(
                "frame {f}: {off} resized bytes differ from Pillow's"
            ));
        }
        exec.sam3_patch_rows_norm(&plane, &mut rows, 1, side, patch, win, 3, kp, true)
            .expect("stem");
        let r16 = exec.to_host_f16_len(&rows, g * g * kp).expect("rows");
        let (_, n) = raw(&file, &format!("n{f:04}")).expect("tensor");
        let n: Vec<u16> = n
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect();
        let mut bad = 0usize;
        for r in 0..g * g {
            // the stem's window-major row -> its grid cell
            let (wi, l) = (r / (win * win), r % (win * win));
            let gy = (wi / (g / win)) * win + l / win;
            let gx = (wi % (g / win)) * win + l % win;
            for col in 0..3 * pp {
                let (c, rem) = (col / pp, col % pp);
                let (y, x) = (gy * patch + rem / patch, gx * patch + rem % patch);
                let theirs = n[(c * side + y) * side + x];
                bad += (r16[r * kp + col].to_bits() != theirs) as usize;
            }
            bad += (3 * pp..kp)
                .filter(|&col| r16[r * kp + col] != f16::ZERO)
                .count();
        }
        if bad > 0 {
            failures.push(format!(
                "frame {f}: {bad} stem values differ from Meta's frame"
            ));
        }
        checked += 1;
    }
    say(&format!(
        "  video frames: {checked} frames, Pillow resize and fp16 normalization checked bit for bit"
    ));
    assert!(checked > 0, "no resized frames in the golden");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Meta's video detector on all 270 frames of clip 0001 ("person"): our
/// frame input (Pillow's resize, the fp16 normalization) and the picture
/// detector, the joint score's logit round trip, the 0.5 threshold and the
/// mask NMS, against the detections Meta's fp32 run kept on each frame
/// (`det` dumps). Same detections in the same order, the scores and boxes
/// close and the 288^2 masks agreeing in sign almost everywhere. This stage
/// has no bf16 twin; the tolerances are well above what it measures (it
/// prints the worst of each) - the end-to-end gate is the real bar.
#[test]
fn video_detector_meets_metas_detections() {
    use paddock_engine::gpu_model::sam3::{GpuSam3, LOW_PX};
    use paddock_tokenizer::sam3::Sam3Tokenizer;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some((gold, index)) = stages("0001-person") else {
        common::missing("SAM3_GOLDENS/video_stages/0001-person (Meta's stage dumps) not there");
        return;
    };
    let root = PathBuf::from(std::env::var_os("SAM3_GOLDENS").expect("SAM3_GOLDENS"));
    let frames = root.join("video_frames/0001.safetensors");
    if !frames.exists() {
        common::missing(
            "SAM3_GOLDENS/video_frames/0001.safetensors (make_video_frames.py) not there",
        );
        return;
    }
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let tk = tok.encode("person").expect("tokenize");
    let mut sam = GpuSam3::load_dir(exec.clone(), &dir, 1280 * 720, 4).expect("load sam3");
    let calls: Vec<&serde_json::Value> = index["calls"].as_array().expect("calls").iter().collect();
    let mut seen = std::collections::BTreeSet::new();
    let mut failures = Vec::new();
    let (mut n_frames, mut n_dets) = (0usize, 0usize);
    let (mut score_worst, mut box_worst, mut sign_worst) = (0f32, 0f32, 0f64);
    for call in &calls {
        if call["stage"] != "det" {
            continue;
        }
        let f = call["frame"].as_u64().expect("frame") as u32;
        if !seen.insert(f) {
            continue;
        }
        let file = gold.join(call["file"].as_str().expect("file"));
        let (_, theirs_s) = tensor(&file, "out.scores");
        let (_, theirs_b) = tensor(&file, "out.bbox");
        let (_, theirs_m) = tensor(&file, "out.mask");
        let (fs, rgb) = raw(&frames, &format!("f{f:04}")).expect("frame");
        let d = sam
            .video_detect(&rgb, (fs[0], fs[1]), &tk.ids, tk.valid)
            .expect("detect");
        n_frames += 1;
        if d.scores.len() != theirs_s.len() {
            failures.push(format!(
                "frame {f}: {} detections {:?} vs Meta's {:?}",
                d.scores.len(),
                d.scores,
                theirs_s
            ));
            continue;
        }
        let n = d.scores.len();
        n_dets += n;
        if n == 0 {
            continue;
        }
        let ours_m = exec
            .to_host_len(sam.video_det_planes().expect("planes"), n * LOW_PX)
            .expect("masks");
        for k in 0..n {
            let ds = (d.scores[k] - theirs_s[k]).abs();
            score_worst = score_worst.max(ds);
            let db = (0..4)
                .map(|j| (d.boxes_xyxy[k][j] - theirs_b[k * 4 + j]).abs())
                .fold(0f32, f32::max);
            box_worst = box_worst.max(db);
            let (a, b) = (
                &ours_m[k * LOW_PX..(k + 1) * LOW_PX],
                &theirs_m[k * LOW_PX..(k + 1) * LOW_PX],
            );
            let union = a
                .iter()
                .zip(b)
                .filter(|(x, y)| **x > 0.0 || **y > 0.0)
                .count();
            let diff = a
                .iter()
                .zip(b)
                .filter(|(x, y)| (**x > 0.0) != (**y > 0.0))
                .count();
            let sign = diff as f64 / union.max(1) as f64;
            sign_worst = sign_worst.max(sign);
            // a pixel or two on a tiny mask is not a share of anything
            if ds > 5e-3 || db > 5e-3 || diff > union / 100 + 2 {
                failures.push(format!(
                    "frame {f}, detection {k}: score {} vs {} ({ds:.2e}), box {db:.2e}, mask \
                     sign {diff} of {union} pixels",
                    d.scores[k], theirs_s[k]
                ));
            }
        }
    }
    say(&format!(
        "  video detector: {n_frames} frames, {n_dets} detections; worst score {score_worst:.2e}, \
         box {box_worst:.2e}, mask sign {:.3}% of the union",
        sign_worst * 100.0
    ));
    assert!(n_frames > 0, "no detections in the dumps");
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A decision golden's frame: ids and row-major masks (bit-unpacked).
fn golden_frame(path: &Path, f: u32, px: usize) -> (Vec<i64>, Vec<Vec<bool>>) {
    let ids = ints(path, &format!("f{f:04}.ids"));
    if ids.is_empty() {
        return (ids, Vec::new());
    }
    let (_, packed) = raw(path, &format!("f{f:04}.masks_packed")).expect("masks");
    let row = packed.len() / ids.len();
    let masks = (0..ids.len())
        .map(|i| {
            (0..px)
                .map(|p| (packed[i * row + p / 8] >> (7 - p % 8)) & 1 == 1)
                .collect()
        })
        .collect();
    (ids, masks)
}

/// SAM 3 tracking clip 0001 ("person", 270 frames) end to end: the session
/// from PIL's decoded frames - detection, the bank and the memory attention,
/// the heads, the decisions, births and re-conditioning, the memory update,
/// the hot-start buffer - against Meta's fp32 run at decision level. The bar
/// is Meta's own precision: its bf16 run lands on the fp32 object set on 260
/// of 270 frames, with a mean mask IoU of 0.986 over the objects both have
/// (two empty masks counting as agreeing); ours must agree with fp32 at
/// least as often and at least as closely.
#[test]
fn video_tracks_like_metas_own_bf16() {
    use paddock_engine::gpu_model::sam3::GpuSam3;
    use paddock_tokenizer::sam3::Sam3Tokenizer;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(root) = std::env::var_os("SAM3_GOLDENS").map(PathBuf::from) else {
        common::missing("SAM3_GOLDENS not set");
        return;
    };
    let frames = root.join("video_frames/0001.safetensors");
    let fp32 = root.join("video/0001/person.fp32.safetensors");
    let bf16 = root.join("video/0001/person.bf16.safetensors");
    if !frames.exists() || !fp32.exists() || !bf16.exists() {
        common::missing("SAM3_GOLDENS video_frames / video decision goldens not there");
        return;
    }
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let tk = tok.encode("person").expect("tokenize");
    let mut sam = GpuSam3::load_dir(exec.clone(), &dir, 1280 * 720, 4).expect("load sam3");
    let (h, w) = (720usize, 1280usize);
    let nframes = 270u32;
    let mut s = sam
        .video_start(&[(tk.ids, tk.valid)], (h, w), Some(nframes), false)
        .expect("start");
    let mut out = Vec::new();
    let t0 = std::time::Instant::now();
    for f in 0..nframes {
        let (_, rgb) = raw(&frames, &format!("f{f:04}")).expect("frame");
        let step = sam.video_frame(&mut s, &rgb).expect("frame");
        out.extend(step.frames);
    }
    out.extend(sam.video_finish(&mut s).expect("finish"));
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(out.len(), nframes as usize, "frames out");

    // ours against fp32, and Meta bf16 against fp32, frame by frame
    let px = h * w;
    let (mut same, mut same16) = (0usize, 0usize);
    let (mut ious, mut ious16) = (Vec::new(), Vec::new());
    let mut differ = Vec::new();
    let mut low = Vec::new();
    for fr in &out {
        let f = fr.frame;
        let (gids, gmasks) = golden_frame(&fp32, f, px);
        let (bids, bmasks) = golden_frame(&bf16, f, px);
        let ids: Vec<i64> = fr.objects.iter().map(|o| o.id).collect();
        // two empty masks agree: Meta keeps an object whose pixels the
        // output's one-owner pass then hands to another
        let iou = |a: &[bool], b: &[bool]| {
            let (mut i, mut u) = (0usize, 0usize);
            for (x, y) in a.iter().zip(b) {
                i += (*x && *y) as usize;
                u += (*x || *y) as usize;
            }
            if u == 0 { 1.0 } else { i as f64 / u as f64 }
        };
        if ids == gids {
            same += 1;
            for (k, o) in fr.objects.iter().enumerate() {
                // our RLE is column-major, zeros first
                let mut m = vec![false; px];
                let (mut pos, mut on) = (0usize, false);
                for &c in &o.rle {
                    for p in pos..pos + c as usize {
                        let (x, y) = (p / h, p % h);
                        m[y * w + x] = on;
                    }
                    pos += c as usize;
                    on = !on;
                }
                let v = iou(&m, &gmasks[k]);
                if v < 0.9 {
                    let area = |x: &[bool]| x.iter().filter(|b| **b).count();
                    low.push(format!(
                        "frame {f} object {}: IoU {v:.3}, ours {} px, Meta {} px",
                        o.id,
                        area(&m),
                        area(&gmasks[k])
                    ));
                }
                ious.push(v);
            }
        } else {
            differ.push(format!("{f}: {ids:?} vs {gids:?}"));
        }
        if bids == gids {
            same16 += 1;
            for k in 0..gids.len() {
                ious16.push(iou(&bmasks[k], &gmasks[k]));
            }
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let (m, m16) = (mean(&ious), mean(&ious16));
    let min = ious.iter().copied().fold(1.0, f64::min);
    say(&format!(
        "  video end to end: {nframes} frames in {dt:.1} s ({:.2} fps); object set = fp32's on \
         {same} frames (Meta bf16 {same16}); mean mask IoU {m:.4} (Meta bf16 {m16:.4}), worst \
         {min:.4}",
        nframes as f64 / dt
    ));
    for d in differ.iter().take(20) {
        say(&format!("    differs at frame {d}"));
    }
    for l in &low {
        say(&format!("    {l}"));
    }
    assert!(
        same >= same16 && m >= m16,
        "against Meta fp32: {same} frames agree (bf16 {same16}), mean IoU {m:.4} (bf16 {m16:.4})"
    );
}

/// Several concepts in one session track each as its own session would:
/// clip 0001's first 40 frames with "person" and "tree" together, every
/// final frame and every preview the same objects - same masks to the pixel,
/// same scores and boxes - as a "person" session and a "tree" session, the
/// ids interleaved (`local * 2 + concept`). The frame is encoded once for
/// both; nothing a concept's pass leaves behind may reach the other's.
#[test]
fn video_concepts_track_as_their_own_sessions() {
    use paddock_engine::gpu_model::sam3::{GpuSam3, Sam3VideoFrame};
    use paddock_tokenizer::sam3::Sam3Tokenizer;
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some(dir) = common::model_dir("SAM3_DIR", &["sam3"]) else {
        return;
    };
    let Some(root) = std::env::var_os("SAM3_GOLDENS").map(PathBuf::from) else {
        common::missing("SAM3_GOLDENS not set");
        return;
    };
    let frames = root.join("video_frames/0001.safetensors");
    if !frames.exists() {
        common::missing("SAM3_GOLDENS video_frames not there");
        return;
    }
    let tok = Sam3Tokenizer::from_file(&dir.join("tokenizer.json")).expect("tokenizer");
    let concepts: Vec<_> = ["person", "tree"]
        .iter()
        .map(|t| {
            let tk = tok.encode(t).expect("tokenize");
            (tk.ids, tk.valid)
        })
        .collect();
    let mut sam = GpuSam3::load_dir(exec.clone(), &dir, 1280 * 720, 4).expect("load sam3");
    let (h, w) = (720usize, 1280usize);
    let nframes = 40u32;
    // a session's final frames, then its previews, frame by frame
    let mut run = |prompts: &[([u32; 32], usize)]| -> (Vec<Sam3VideoFrame>, Vec<Sam3VideoFrame>) {
        let mut s = sam.video_start(prompts, (h, w), None, true).expect("start");
        let (mut out, mut previews) = (Vec::new(), Vec::new());
        for f in 0..nframes {
            let (_, rgb) = raw(&frames, &format!("f{f:04}")).expect("frame");
            let step = sam.video_frame(&mut s, &rgb).expect("frame");
            out.extend(step.frames);
            previews.push(step.preview.expect("preview asked for"));
        }
        out.extend(sam.video_finish(&mut s).expect("finish"));
        (out, previews)
    };
    let both = run(&concepts);
    let alone = [run(&concepts[..1]), run(&concepts[1..])];
    let n = concepts.len() as i64;
    let mut bad = Vec::new();
    let mut seen = [0usize; 2];
    for (what, b, a) in [
        ("final", &both.0, [&alone[0].0, &alone[1].0]),
        ("preview", &both.1, [&alone[0].1, &alone[1].1]),
    ] {
        assert_eq!(b.len(), a[0].len(), "{what} frames");
        for (i, fr) in b.iter().enumerate() {
            for (k, single) in a.iter().enumerate() {
                let theirs = &single[i].objects;
                let ours: Vec<_> = fr.objects.iter().filter(|o| o.concept == k).collect();
                seen[k] += ours.len();
                let same = ours.len() == theirs.len()
                    && ours.iter().zip(theirs.iter()).all(|(o, t)| {
                        o.id == t.id * n + k as i64
                            && o.prob.to_bits() == t.prob.to_bits()
                            && o.bbox_px == t.bbox_px
                            && o.area == t.area
                            && o.rle == t.rle
                    });
                if !same {
                    bad.push(format!(
                        "{what} frame {}: concept {k} {:?} vs alone {:?}",
                        fr.frame,
                        ours.iter().map(|o| (o.id, o.area)).collect::<Vec<_>>(),
                        theirs.iter().map(|o| (o.id, o.area)).collect::<Vec<_>>()
                    ));
                }
            }
        }
    }
    say(&format!(
        "  two concepts in one session: {nframes} frames, {} person and {} tree objects across \
         finals and previews, {} differ",
        seen[0],
        seen[1],
        bad.len()
    ));
    for b in bad.iter().take(10) {
        say(&format!("    {b}"));
    }
    assert!(
        seen[0] > 0 && seen[1] > 0,
        "both concepts must find something to compare"
    );
    assert!(bad.is_empty(), "{} frames differ", bad.len());
}
