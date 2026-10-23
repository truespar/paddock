//! Nemotron 3.5 Embed 8B and Rerank 8B against their F64 oracle - the
//! release's own Transformers code (`Mistral3Model`, the reranker's
//! `model.py`) run in float64 on the BF16 weights widened exactly (the
//! repo's nemotron-embed oracle script writes the fixtures). The bar on
//! every case is the vendor's own serving class: the same code in BF16
//! (`embed-bf16/`, `rerank-bf16/`). An f16-operand lane with f32 residual
//! must land CLOSER to F64 than BF16 does.
//!
//! Legs:
//!   - pixels: the processor's uint8 resize + fused normalize reproduced -
//!     the stem's f16 rows equal the oracle's f32 pixels rounded to nearest;
//!   - the picture rows (tower + projector) against F64;
//!   - the pooled mean and the unit vector (Embed), the logit (Rerank);
//!   - batch invariance: every case alone, all together, and in reverse
//!     order land the same bits.
//!
//! Fixtures: `<models>/nemotron-embed-oracle/` (or NEMOTRON_EMBED_REFERENCE_DIR),
//! checkpoints `<models>/Nemotron-3.5-{Embed,Rerank}-8B-BF16/`. Heavy: each
//! leg uploads a 16 GB model (PADDOCK_HEAVY_TESTS=1).

mod common;

use std::path::{Path, PathBuf};

use paddock_engine::gpu_model::nemotron_embed::{GpuNemotronEmbed, Output, Picture};
use paddock_engine::service::MmChunk;

fn say(msg: &str) {
    eprintln!("nemotron-embed: {msg}");
}

fn read_le<const N: usize, T>(path: &Path, f: fn([u8; N]) -> T) -> Vec<T> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .as_chunks::<N>()
        .0
        .iter()
        .map(|c| f(*c))
        .collect()
}

fn f64s(path: &Path) -> Vec<f64> {
    read_le(path, f64::from_le_bytes)
}

fn f32s(path: &Path) -> Vec<f32> {
    read_le(path, f32::from_le_bytes)
}

/// rel L2 of `got` against `want`
fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        num += (g - w).powi(2);
        den += w.powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

struct Case {
    name: String,
    dir: PathBuf,
    meta: serde_json::Value,
    ids: Vec<u32>,
    picture: Option<(Vec<u8>, usize, usize)>,
}

impl Case {
    fn media(&self) -> Vec<MmChunk> {
        self.picture
            .iter()
            .map(|(rgb, w, h)| MmChunk::Image {
                rgb: rgb.clone(),
                w: *w,
                h: *h,
            })
            .collect()
    }
}

fn fixtures(sub: &str, ckpt: &str) -> Option<(PathBuf, PathBuf, Vec<Case>)> {
    let root = std::env::var_os("NEMOTRON_EMBED_REFERENCE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            common::model_roots()
                .into_iter()
                .map(|r| r.join("nemotron-embed-oracle"))
                .find(|p| p.is_dir())
        });
    let Some(root) = root.filter(|r| r.join(sub).join("manifest.json").is_file()) else {
        common::missing("nemotron-embed oracle fixtures (nemotron-embed-oracle/)");
        return None;
    };
    let Some(model) = common::model_dir("", &[ckpt]) else {
        common::missing(&format!("the {ckpt} checkpoint"));
        return None;
    };
    let fix = root.join(sub);
    let man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fix.join("manifest.json")).expect("manifest"))
            .expect("parse manifest");
    let cases = man["cases"]
        .as_array()
        .expect("manifest.cases")
        .iter()
        .map(|c| {
            let name = c.as_str().expect("case name").to_owned();
            let dir = fix.join(&name);
            let meta: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("meta"))
                    .expect("parse meta");
            let ids = read_le(&dir.join("input_ids.u32"), u32::from_le_bytes);
            let picture = meta["input_wh"].as_array().map(|wh| {
                let (w, h) = (
                    wh[0].as_u64().expect("input width") as usize,
                    wh[1].as_u64().expect("input height") as usize,
                );
                (
                    std::fs::read(dir.join("input.rgb")).expect("input.rgb"),
                    w,
                    h,
                )
            });
            Case {
                name,
                dir,
                meta,
                ids,
                picture,
            }
        })
        .collect();
    Some((root, model, cases))
}

fn load(exec: std::sync::Arc<paddock_engine::gpu::GpuExecutor>, model: &Path) -> GpuNemotronEmbed {
    GpuNemotronEmbed::load_dir(exec, model, 32768, true).expect("load the checkpoint")
}

/// The pixel leg: the stem's rows against the oracle's `[3][H][W]` f32
/// pixels rounded to f16, patch (pr, pc) column (c, ky, kx).
fn pixels_match(m: &mut GpuNemotronEmbed, c: &Case) {
    let (rgb, w, h) = c.picture.as_ref().expect("a picture case");
    let rows = m
        .picture_patch_rows(&Picture { rgb, w: *w, h: *h })
        .expect("patch rows");
    let want = f32s(&c.dir.join("pixels.f32"));
    let (tw, th) = m.cfg.picture_target(*w, *h).expect("picture target");
    assert_eq!(want.len(), 3 * tw * th, "{}: oracle pixel plane", c.name);
    let cols = tw / 14;
    let mut bad = 0usize;
    for (p, row) in rows.as_chunks::<592>().0.iter().enumerate() {
        let (pr, pc) = (p / cols, p % cols);
        for (i, &v) in row[..588].iter().enumerate() {
            let (ch, r) = (i / 196, i % 196);
            let (y, x) = (pr * 14 + r / 14, pc * 14 + r % 14);
            let o = want[(ch * th + y) * tw + x];
            bad += (v != half::f16::from_f32(o)) as usize;
        }
        bad += row[588..].iter().filter(|v| v.to_f32() != 0.0).count();
    }
    say(&format!(
        "{}: stem {bad} of {} halves differ from the processor's pixels",
        c.name,
        rows.len()
    ));
    assert_eq!(bad, 0, "{}: the pixel path", c.name);
}

#[test]
fn embed_matches_the_f64_oracle_and_is_batch_invariant() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((root, model, cases)) = fixtures("embed", "Nemotron-3.5-Embed-8B-BF16") else {
        return;
    };
    let mut m = load(exec, &model);
    let bf = root.join("embed-bf16");
    let mut worst = 0f64;
    for c in &cases {
        if let Some((rgb, w, h)) = &c.picture {
            pixels_match(&mut m, c);
            let got: Vec<f64> = m
                .picture_features(&[Picture { rgb, w: *w, h: *h }])
                .expect("picture rows")
                .iter()
                .map(|&v| v as f64)
                .collect();
            let want: Vec<f64> = f32s(&c.dir.join("vision.f32"))
                .iter()
                .map(|&v| v as f64)
                .collect();
            let bar: Vec<f64> = f32s(&bf.join(&c.name).join("vision.f32"))
                .iter()
                .map(|&v| v as f64)
                .collect();
            let (e, b) = (rel_l2(&got, &want), rel_l2(&bar, &want));
            say(&format!(
                "{}: picture rows rel L2 {e:.3e} (vendor BF16 {b:.3e})",
                c.name
            ));
            assert!(
                e < b,
                "{}: picture rows {e:.3e} not closer than BF16 {b:.3e}",
                c.name
            );
        }
        let seqs = [c.ids.clone()];
        let media = [c.media()];
        let mean: Vec<f64> = m.run(&seqs, &media, Output::Mean).expect("mean")[0]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let emb: Vec<f64> = m
            .run(&seqs, &media, Output::Embed { dims: 4096 })
            .expect("embed")[0]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let want_mean = f64s(&c.dir.join("hidden_mean.f64"));
        let want = f64s(&c.dir.join("embedding.f64"));
        let bar = f64s(&bf.join(&c.name).join("embedding.f64"));
        let cos = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let (e, b) = (rel_l2(&emb, &want), rel_l2(&bar, &want));
        say(&format!(
            "{} ({} tokens): mean rel L2 {:.3e}; vector rel L2 {e:.3e}, 1 - cos {:.2e} \
             (vendor BF16 {b:.3e}, {:.2e})",
            c.name,
            c.ids.len(),
            rel_l2(&mean, &want_mean),
            1.0 - cos(&emb, &want),
            1.0 - cos(&bar, &want),
        ));
        assert!(
            e < b,
            "{}: vector {e:.3e} not closer to F64 than BF16 {b:.3e}",
            c.name
        );
        worst = worst.max(e);
    }
    // a sanity ceiling, not the gate (the per-case BF16 bar is): a
    // structural mistake lands near 1
    assert!(
        worst < 2e-2,
        "vector rel L2 {worst:.3e} - structurally wrong"
    );

    // batch invariance: alone, all together, reversed - the same bits
    let alone: Vec<Vec<f32>> = cases
        .iter()
        .map(|c| {
            m.run(
                std::slice::from_ref(&c.ids),
                &[c.media()],
                Output::Embed { dims: 4096 },
            )
            .expect("alone")
            .remove(0)
        })
        .collect();
    let (seqs, media): (Vec<_>, Vec<_>) = cases.iter().map(|c| (c.ids.clone(), c.media())).unzip();
    let together = m
        .run(&seqs, &media, Output::Embed { dims: 4096 })
        .expect("together");
    let (mut rs, mut rm) = (seqs.clone(), media.clone());
    rs.reverse();
    rm.reverse();
    let mut reversed = m
        .run(&rs, &rm, Output::Embed { dims: 4096 })
        .expect("reversed");
    reversed.reverse();
    for (i, c) in cases.iter().enumerate() {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&alone[i]),
            bits(&together[i]),
            "{}: alone vs packed",
            c.name
        );
        assert_eq!(
            bits(&alone[i]),
            bits(&reversed[i]),
            "{}: alone vs reversed",
            c.name
        );
    }
    say(&format!(
        "batch invariance: {} cases alone = packed = reversed",
        cases.len()
    ));

    // Matryoshka: a prefix re-normalized
    let c = &cases[0];
    let full = &alone[0];
    let d512 = m
        .run(
            std::slice::from_ref(&c.ids),
            &[c.media()],
            Output::Embed { dims: 512 },
        )
        .expect("512")
        .remove(0);
    let n: f64 = full[..512]
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let dev = d512
        .iter()
        .zip(&full[..512])
        .map(|(&a, &b)| (a as f64 - b as f64 / n).abs())
        .fold(0.0, f64::max);
    assert!(dev < 1e-6, "512-dim prefix off by {dev:.2e}");
}

#[test]
fn rerank_matches_the_f64_oracle() {
    if !common::heavy() {
        return;
    }
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((root, model, cases)) = fixtures("rerank", "Nemotron-3.5-Rerank-8B-BF16") else {
        return;
    };
    let mut m = load(exec, &model);
    let bf = root.join("rerank-bf16");
    for c in &cases {
        let got = m
            .run(std::slice::from_ref(&c.ids), &[c.media()], Output::Score)
            .expect("score")[0][0] as f64;
        let want = c.meta["logit"].as_f64().expect("logit");
        let bar: serde_json::Value = serde_json::from_slice(
            &std::fs::read(bf.join(&c.name).join("meta.json")).expect("bf16 meta"),
        )
        .unwrap();
        let b = bar["logit"].as_f64().unwrap();
        say(&format!(
            "{} ({} tokens): logit {got:.6} vs F64 {want:.6} (|d| {:.2e}; vendor BF16 {b:.6}, \
             |d| {:.2e})",
            c.name,
            c.ids.len(),
            (got - want).abs(),
            (b - want).abs()
        ));
        // closer than BF16, or inside f32 rounding of it where BF16 is
        // already that close
        assert!(
            (got - want).abs() <= (b - want).abs().max(1e-4),
            "{}: logit off by {:.2e}, BF16 by {:.2e}",
            c.name,
            (got - want).abs(),
            (b - want).abs()
        );
    }
}

/// Pass throughput on the shapes retrieval serves: short queries in bulk,
/// passages, long documents, picture pages. Every item is distinct (windows
/// of the 20K case's ids; a fresh set of public pages each repetition), so
/// a rival that caches by content is timed on the same work. Not a gate.
///   PADDOCK_HEAVY_TESTS=1 cargo test --release --test gpu_nemotron_embed bench -- --ignored --nocapture
#[test]
#[ignore]
fn bench_passes() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let Some((root, model, cases)) = fixtures("embed", "Nemotron-3.5-Embed-8B-BF16") else {
        return;
    };
    let mut m = load(exec, &model);
    let long = &cases.iter().find(|c| c.name == "p_20k").expect("p_20k").ids;
    let page = cases
        .iter()
        .find(|c| c.name == "img_page")
        .expect("img_page");
    // item i of n tokens: a window of the long case's ids (the body does
    // not care what the tokens say, and no two items are equal)
    let win = |i: usize, n: usize| {
        let o = (i * 7919) % (long.len() - n);
        long[o..o + n].to_vec()
    };
    let text = |k: usize, n: usize| -> Vec<Vec<u32>> { (0..k).map(|i| win(i, n)).collect() };
    let pages = root.join("bench-pages");
    let pic = |i: usize| -> Vec<MmChunk> {
        let rgb = std::fs::read(pages.join(format!("page_{i:03}.rgb"))).expect("bench page");
        vec![MmChunk::Image {
            rgb,
            w: 1448,
            h: 2048,
        }]
    };
    // rep 0 warms; reps 1-3 are timed, each on its own inputs
    type Shape<'a> = (
        &'static str,
        Box<dyn Fn(usize) -> (Vec<Vec<u32>>, Vec<Vec<MmChunk>>) + 'a>,
    );
    let shapes: Vec<Shape<'_>> = vec![
        (
            "1 query x 16",
            Box::new(move |r| (vec![win(r, 16)], vec![vec![]])),
        ),
        (
            "64 queries x 16",
            Box::new(move |_| (text(64, 16), vec![vec![]; 64])),
        ),
        (
            "16 passages x 512",
            Box::new(move |_| (text(16, 512), vec![vec![]; 16])),
        ),
        (
            "64 passages x 512",
            Box::new(move |_| (text(64, 512), vec![vec![]; 64])),
        ),
        (
            "8 passages x 2048",
            Box::new(move |_| (text(8, 2048), vec![vec![]; 8])),
        ),
        (
            "1 document x 16384",
            Box::new(move |_| (text(1, 16384), vec![vec![]])),
        ),
        (
            "1 page picture",
            Box::new({
                let ids = page.ids.clone();
                move |r| (vec![ids.clone()], vec![pic(r)])
            }),
        ),
        (
            "8 page pictures",
            Box::new({
                let ids = page.ids.clone();
                move |r| {
                    (
                        vec![ids.clone(); 8],
                        (0..8).map(|i| pic(8 * r + i)).collect(),
                    )
                }
            }),
        ),
    ];
    // NE_BENCH_ONLY=<substring> times one shape (for a profiler run)
    let only = std::env::var("NE_BENCH_ONLY").ok();
    for (name, make) in &shapes {
        if only.as_ref().is_some_and(|o| !name.contains(o.as_str())) {
            continue;
        }
        let (seqs, media) = make(0);
        let rows: usize = seqs.iter().map(Vec::len).sum();
        m.run(&seqs, &media, Output::Embed { dims: 4096 })
            .expect("warm");
        let mut best = f64::MAX;
        for r in 1..4 {
            let (seqs, media) = make(r);
            let t = std::time::Instant::now();
            m.run(&seqs, &media, Output::Embed { dims: 4096 })
                .expect("pass");
            best = best.min(t.elapsed().as_secs_f64());
        }
        say(&format!(
            "{name}: {rows} rows {:.1} ms = {:.0} rows/s, {:.1} items/s",
            best * 1e3,
            rows as f64 / best,
            seqs.len() as f64 / best
        ));
    }
}

/// The body's projections onto the residual stream (slot 894's two
/// landings) against the plain f32 landing on the same shapes - what the
/// bias + residual epilogue costs in place. Not a gate.
#[test]
#[ignore]
fn probe_residual_landing() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    // pseudo-random operands: constant ones toggle few bits, draw less power
    // and clock higher than real activations do
    let noise = |n: usize, scale: f32, seed: u64| -> Vec<half::f16> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                half::f16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale)
            })
            .collect()
    };
    let rows_all: usize = std::env::var("NE_PROBE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384);
    for (name, k, n, rows) in [
        ("o", 4096usize, 4096usize, rows_all),
        ("down", 14336, 4096, rows_all),
    ] {
        let w = paddock_engine::gpu::HalfTensor {
            buf: exec.f16_to_device(&noise(k * n, 0.04, 1)).unwrap(),
            dims: vec![k, n],
        };
        let x = exec.f16_to_device(&noise(rows * k, 2.0, 2)).unwrap();
        let mut y = exec.alloc(rows * n).unwrap();
        let zeros = exec.to_device(&vec![0.0f32; n]).unwrap();
        for (tag, bias) in [("plain", None), ("residual", Some(&zeros))] {
            exec.f16_gemm_ring_rows(&w, &x, 0, &mut y, 0, rows, bias)
                .unwrap();
            exec.synchronize().unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                exec.f16_gemm_ring_rows(&w, &x, 0, &mut y, 0, rows, bias)
                    .unwrap();
                exec.synchronize().unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            say(&format!(
                "{name} {tag}: {:.2} ms = {:.1} TF/s",
                best * 1e3,
                2.0 * (k * n * rows) as f64 / best / 1e12
            ));
        }
    }
}
