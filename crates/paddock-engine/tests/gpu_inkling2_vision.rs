//! Inkling 2 Mini's picture tower on CUDA (slots 1011-1013 + the half tower
//! interface), heavy: the official release (`INKLING2_DIR`) against the
//! reference model's goldens (`INKLING2_VISION_REF`, the vision reference
//! script's output - transformers' own processor and tower in float64 and
//! in bfloat16, fed the pictures' exact RGB bytes).
//!
//! - `inkling2_tower_vs_f64`: every picture's language-model rows against
//!   the float64 oracle; the lane must sit well inside the reference's own
//!   bfloat16 error. Then all pictures in one pass against each alone: a
//!   picture's rows must not depend on what shares its pass.
//! - `inkling2_pictures_vs_reference`: picture prompts teacher-forced
//!   against the reference model (`INKLING2_PICTURE_GOLDENS`, the reference
//!   script's `--image-prompt` output: its own processor, tower and greedy
//!   answer): top-1 at every position after the picture, gated like the
//!   text lane's - the agreement band (`INKLING2_FLOOR`, default 0.94) and
//!   the clear misses (rows the reference calls by a margin of 0.5 or more).

mod common;

use paddock_engine::gpu_model::inkling2::vision::{Picture, Tower};
use paddock_models::safetensors::{SafetensorsFile, ShardedSafetensors};

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>()
        .0
        .iter()
        .map(|c| i64::from_le_bytes(*c))
        .collect()
}

/// |a - b| / |b| over the whole plane.
fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        num += (*x as f64 - *y as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den).sqrt()
}

struct Gold {
    rgb: Vec<u8>,
    w: usize,
    h: usize,
    grid: (usize, usize),
    f64_rows: Vec<f32>,
    bf16_rows: Vec<f32>,
}

fn goldens(g: &SafetensorsFile) -> Vec<Gold> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some((_, rgb)) = g.bytes(&format!("rgb.{i}")) {
        let size = i64s(g.bytes(&format!("size.{i}")).expect("size").1);
        let grid = i64s(g.bytes(&format!("grid.{i}")).expect("grid").1);
        out.push(Gold {
            rgb: rgb.to_vec(),
            w: size[0] as usize,
            h: size[1] as usize,
            grid: (grid[0] as usize, grid[1] as usize),
            f64_rows: f32s(g.bytes(&format!("f64.{i}")).expect("f64 rows").1),
            bf16_rows: f32s(g.bytes(&format!("bf16.{i}")).expect("bf16 rows").1),
        });
        i += 1;
    }
    out
}

#[test]
fn inkling2_tower_vs_f64() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2_vision() {
        common::missing("pack has no Inkling 2 tower ops (slots 1011-1013)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(path) = std::env::var_os("INKLING2_VISION_REF") else {
        common::missing("INKLING2_VISION_REF (the vision reference script's goldens)");
        return;
    };
    let g = SafetensorsFile::open(std::path::Path::new(&path)).expect("goldens");
    let golds = goldens(&g);
    assert!(!golds.is_empty(), "no pictures in the goldens");
    let st = ShardedSafetensors::open_dir(&dir).expect("checkpoint");
    let mut tower = Tower::load(exec.clone(), &st)
        .expect("load")
        .expect("the release carries a tower");
    let width = paddock_models::inkling2::WIDTH;
    let pics: Vec<Picture<'_>> = golds
        .iter()
        .map(|g| Picture {
            rgb: &g.rgb,
            w: g.w,
            h: g.h,
        })
        .collect();
    let mut alone = Vec::new();
    for (i, (p, gold)) in pics.iter().zip(&golds).enumerate() {
        assert_eq!(p.grid(), gold.grid, "picture {i}: the token grid");
        let t0 = std::time::Instant::now();
        let rows = tower.encode(std::slice::from_ref(p)).expect("encode");
        exec.synchronize().expect("sync");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let got = exec
            .to_host_len(&rows[0].rows, rows[0].n * width)
            .expect("rows");
        assert_eq!(got.len(), gold.f64_rows.len(), "picture {i}: rows");
        let ours = rel_rms(&got, &gold.f64_rows);
        let theirs = rel_rms(&gold.bf16_rows, &gold.f64_rows);
        println!(
            "picture {i}: {} x {} ({} tokens) {ms:.1} ms - rel rms vs f64: ours {ours:.3e}, \
             reference bf16 {theirs:.3e} ({:.1}x closer)",
            p.w,
            p.h,
            p.tokens(),
            theirs / ours
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "picture {i}: non-finite rows"
        );
        assert!(
            ours * 4.0 < theirs,
            "picture {i}: {ours:.3e} off f64 is not well inside the reference's bf16 {theirs:.3e}"
        );
        alone.push(got);
    }
    // every picture in one pass
    let rows = tower.encode(&pics).expect("encode all");
    let mut differ = 0usize;
    for (i, (r, a)) in rows.iter().zip(&alone).enumerate() {
        let got = exec.to_host_len(&r.rows, r.n * width).expect("rows");
        let d = got
            .iter()
            .zip(a)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        if d != 0 {
            println!(
                "picture {i}: {d} of {} floats differ packed, rel {:.3e}",
                got.len(),
                rel_rms(&got, a)
            );
        }
        differ += d;
    }
    assert_eq!(differ, 0, "a picture's rows depend on what shares its pass");
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect()
}

#[test]
fn inkling2_pictures_vs_reference() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_inkling2() || !exec.has_inkling2_vision() {
        common::missing("pack has no Inkling 2 ops (slots 977-1013)");
        return;
    }
    let Some(dir) = common::model_dir("INKLING2_DIR", &["Inkling-2-Mini"]) else {
        common::missing("Inkling-2-Mini");
        return;
    };
    let Some(path) = std::env::var_os("INKLING2_PICTURE_GOLDENS") else {
        common::missing("INKLING2_PICTURE_GOLDENS (the reference script's --image-prompt goldens)");
        return;
    };
    let floor: f64 = std::env::var("INKLING2_FLOOR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.94);
    let g = SafetensorsFile::open(std::path::Path::new(&path)).expect("goldens");
    let mut m = paddock_engine::gpu_model::inkling2::GpuInkling2::load(
        exec.clone(),
        &dir,
        8192,
        2048,
        paddock_engine::gpu::INKLING2_KV_F32,
    )
    .expect("load");
    let image = paddock_models::inkling2::IMAGE_TOKEN;
    let (mut agree, mut total, mut clear) = (0usize, 0usize, 0usize);
    let mut s = 0;
    while let Some((_, ids)) = g.bytes(&format!("s{s}.ids")) {
        let ids: Vec<u32> = i32s(ids).into_iter().map(|t| t as u32).collect();
        let Some((_, rgb)) = g.bytes(&format!("s{s}.rgb")) else {
            s += 1;
            continue;
        };
        let size = i64s(g.bytes(&format!("s{s}.size")).expect("size").1);
        let a = ids
            .iter()
            .position(|&t| t == image)
            .expect("a placeholder run");
        let b = a + ids[a..].iter().take_while(|&&t| t == image).count();
        let chunks = vec![
            paddock_engine::service::MmChunk::Text(ids[..a].to_vec()),
            paddock_engine::service::MmChunk::Image {
                rgb: rgb.to_vec(),
                w: size[0] as usize,
                h: size[1] as usize,
            },
            paddock_engine::service::MmChunk::Text(ids[b..].to_vec()),
        ];
        let ti = i32s(g.bytes(&format!("s{s}.top_ids")).expect("top ids").1);
        let tl = f32s(g.bytes(&format!("s{s}.top_logits")).expect("top logits").1);
        m.reset().expect("reset");
        let mut ours = vec![0usize; ids.len()];
        let laid = m
            .forward_each_mm(
                &chunks,
                paddock_engine::gpu_model::inkling2::Logits::All,
                |r, l| {
                    ours[r] = l
                        .iter()
                        .enumerate()
                        .fold(
                            (0, f32::MIN),
                            |best, (i, &v)| if v > best.1 { (i, v) } else { best },
                        )
                        .0;
                },
            )
            .expect("forward");
        assert_eq!(
            laid, ids,
            "prompt {s}: the laid-out prompt is the reference's"
        );
        let (mut ag, mut n, mut cl) = (0usize, 0usize, 0usize);
        // every row from the picture's last placeholder on predicts text
        for r in b - 1..ids.len() - 1 {
            let want = ti[r * 8] as usize;
            let margin = tl[r * 8] - tl[r * 8 + 1];
            n += 1;
            if ours[r] == want {
                ag += 1;
            } else if margin >= 0.5 {
                cl += 1;
            }
        }
        println!(
            "prompt {s}: {} ids ({} picture rows), top-1 {ag}/{n} = {:.2}%, clear misses {cl}",
            ids.len(),
            b - a,
            100.0 * ag as f64 / n as f64
        );
        agree += ag;
        total += n;
        clear += cl;
        s += 1;
    }
    assert!(total > 0, "no picture prompts in the goldens");
    let rate = agree as f64 / total as f64;
    println!(
        "all: {agree}/{total} = {:.2}%, clear misses {clear}",
        100.0 * rate
    );
    assert!(rate >= floor, "top-1 {rate:.4} under the {floor} band");
    assert!(
        clear * 50 <= total,
        "{clear} clear misses in {total} rows - the picture rows are off"
    );
}
