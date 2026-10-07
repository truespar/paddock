//! S8 gates: multimodal requests riding batch slots. The slot mm prefill must
//! produce the single-path's greedy stream, a text slot decoding CONCURRENTLY
//! with an image slot must be unaffected (mixed-batch mrope deltas), and the
//! image-embedding cache must serve a re-sent image without touching the
//! vision tower.
//!
//! Very heavy (Qwen3.6-27B + mmproj, ~28 GB residency): PADDOCK_HEAVY_TESTS=1,
//! --release --test-threads=1.

mod common;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_engine::service::MmChunk;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

fn setup() -> Option<(GpuQwen35, GgufTokenizer)> {
    if !common::heavy() {
        return None;
    }
    // QWEN36_MM_DIR points the gate at another model dir (e.g. the 35B-A3B -
    // same tower, different merger width); default stays the 27B.
    let dir = common::model_dir("QWEN36_MM_DIR", common::QWEN36_27B_DIR)?;
    // the backbone GGUF is the directory's sole non-mmproj Q8_0 file
    let model_path = std::fs::read_dir(&dir).ok().and_then(|rd| {
        rd.filter_map(|e| e.ok().map(|e| e.path())).find(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.ends_with("Q8_0.gguf") && !n.starts_with("mmproj")
        })
    });
    // F16 or BF16 mmproj (unsloth ships BF16 for the 27B)
    let mmproj_path = ["mmproj-F16.gguf", "mmproj-BF16.gguf"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists());
    let (Some(model_path), Some(mmproj_path)) = (model_path, mmproj_path) else {
        common::missing(&format!(
            "no Q8_0 backbone + mmproj pair in {}",
            dir.display()
        ));
        return None;
    };
    let exec = common::gpu_arc()?;
    let map = MappedGguf::open(&model_path).expect("open gguf");
    let tok = GgufTokenizer::from_gguf(map.gguf()).expect("tokenizer");
    let mut m = GpuQwen35::load(exec, &map, 4096).expect("load 27B");
    // the exact class this test is stated in (the KV8 default is lossy)
    m.set_kv_dtype(paddock_engine::gpu::KvDtype::Fp16);
    let mm = MappedGguf::open(&mmproj_path).expect("open mmproj");
    m.attach_vision(&mm).expect("attach vision");
    Some((m, tok))
}

/// 256x160 interleaved RGB8: solid red left half, solid blue right half.
fn red_blue_rgb() -> (Vec<u8>, usize, usize) {
    let (w, h) = (256usize, 160usize);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for _y in 0..h {
        for x in 0..w {
            if x < w / 2 {
                rgb.extend_from_slice(&[255, 0, 0]);
            } else {
                rgb.extend_from_slice(&[0, 0, 255]);
            }
        }
    }
    (rgb, w, h)
}

/// 160x256 interleaved RGB8 (portrait - a different merged grid than
/// red_blue): solid green top half, solid yellow bottom half. Distinct bytes so
/// the image cache keys it separately.
fn green_yellow_rgb() -> (Vec<u8>, usize, usize) {
    let (w, h) = (160usize, 256usize);
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for _x in 0..w {
            if y < h / 2 {
                rgb.extend_from_slice(&[0, 255, 0]);
            } else {
                rgb.extend_from_slice(&[255, 255, 0]);
            }
        }
    }
    (rgb, w, h)
}

fn amax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i) as u32
}

#[test]
fn slot_mm_matches_single_path_text_unaffected_and_cache_serves() {
    let Some((mut m, tok)) = setup() else { return };
    let vocab = m.vocab;
    let n_new = 12usize;

    let before = tok.encode("Picture 1: ").expect("enc");
    let after = tok
        .encode(" The two dominant solid colors in this picture are")
        .expect("enc");
    let (rgb, w, h) = red_blue_rgb();
    let chunks = vec![
        MmChunk::Text(before.clone()),
        MmChunk::Image { rgb, w, h },
        MmChunk::Text(after.clone()),
    ];
    let text_prompt = tok
        .encode("The capital of France is the city of")
        .expect("enc");

    // ---- single-path references (mm stream + text stream)
    let mut mm_ref = Vec::new();
    {
        let (logits, rows) = m.forward_multimodal_chunks(&chunks).expect("single mm");
        // The exclusive path reports what it PREFILLED, not the prompt's text
        // (which billed an image prompt as its text alone). 256x160
        // at patch 16 merged 2x2 is an 8x5 soft-token run, so the count has to
        // clear the text by at least that. (The first cut of this assertion
        // said `> 100` - the real image the bug was found on was 1471 rows -
        // and only a heavy-gate run says what this picture is worth.)
        let text_rows = before.len() + after.len();
        assert!(
            rows >= text_rows + 40,
            "mm prefill reported {rows} rows; expected {text_rows} text + >=40 image"
        );
        let mut t = amax(&logits);
        mm_ref.push(t);
        for _ in 1..n_new {
            let l = m.forward_one(t).expect("fwd");
            t = amax(&l);
            mm_ref.push(t);
        }
    }
    let mut text_ref = Vec::new();
    {
        m.reset();
        let mut l = Vec::new();
        for &t in &text_prompt {
            l = m.forward_one(t).expect("fwd");
        }
        let mut t = amax(&l);
        text_ref.push(t);
        for _ in 1..n_new {
            l = m.forward_one(t).expect("fwd");
            t = amax(&l);
            text_ref.push(t);
        }
    }
    eprintln!("mm ref: {:?}", tok.decode(&mm_ref, false));
    eprintln!("text ref: {:?}", tok.decode(&text_ref, false));

    // ---- slot path: image in slot 0, text in slot 1, decoded together (b=2).
    // The single-path reference above already cached the image's embeddings,
    // so this prefill must be served across paths from the cache.
    assert_eq!(
        m.image_cache_reuses(),
        0,
        "reference encode was the first sight"
    );
    m.enable_batch(2).expect("enable_batch");
    let (l0, rows0) = m.forward_prefill_slot_mm(0, &chunks).expect("slot mm");
    assert_eq!(
        m.image_cache_reuses(),
        1,
        "slot prefill must reuse the cached embeddings"
    );
    let l1 = m.forward_prefill_slot(1, &text_prompt).expect("slot text");
    let mut got0 = vec![amax(&l0)];
    let mut got1 = vec![amax(&l1)];
    let (mut p0, mut p1) = (rows0 as u32, text_prompt.len() as u32);
    for _ in 1..n_new {
        let l = m
            .forward_batch(&[*got0.last().unwrap(), *got1.last().unwrap()], &[p0, p1])
            .expect("batch step");
        got0.push(amax(&l[..vocab]));
        got1.push(amax(&l[vocab..2 * vocab]));
        p0 += 1;
        p1 += 1;
    }
    eprintln!("mm slot: {:?}", tok.decode(&got0, false));
    eprintln!("text slot: {:?}", tok.decode(&got1, false));
    assert_eq!(got0, mm_ref, "mm slot stream diverged from the single path");
    assert_eq!(
        got1, text_ref,
        "text slot contaminated by the concurrent image slot"
    );

    // ---- image-embedding cache: the same bytes re-sent must skip the tower
    let (l0b, rows0b) = m
        .forward_prefill_slot_mm(0, &chunks)
        .expect("slot mm resend");
    assert_eq!(rows0b, rows0);
    assert_eq!(
        m.image_cache_reuses(),
        2,
        "identical image must be served from the embedding cache"
    );
    assert_eq!(
        amax(&l0b),
        got0[0],
        "cache-served prefill changed the greedy token"
    );
    eprintln!("VISION SERVING OK: slot==single, concurrent text intact, cache reused");
}

/// Multi-image gate: two interleaved images (distinct grids) must (a) prefill +
/// decode a coherent greedy stream on the exclusive path and (b) produce the
/// identical stream on the batched-slot path, and the per-image cache must
/// serve both on a re-send. This is the end-to-end proof that lifting the
/// one-image cap lays out the second image's embeddings/mrope/bound correctly.
#[test]
fn two_images_slot_matches_single_path_and_cache_serves_both() {
    let Some((mut m, tok)) = setup() else { return };
    let vocab = m.vocab;
    let n_new = 16usize;

    // "Picture 1: <img A> Picture 2: <img B> Q:" - two images, three text spans.
    let t0 = tok.encode("Picture 1: ").expect("enc");
    let t1 = tok.encode(" Picture 2: ").expect("enc");
    let t2 = tok
        .encode(" The dominant colors are, in order for picture 1 then picture 2:")
        .expect("enc");
    let (rgb_a, wa, ha) = red_blue_rgb();
    let (rgb_b, wb, hb) = green_yellow_rgb();
    let chunks = vec![
        MmChunk::Text(t0),
        MmChunk::Image {
            rgb: rgb_a,
            w: wa,
            h: ha,
        },
        MmChunk::Text(t1),
        MmChunk::Image {
            rgb: rgb_b,
            w: wb,
            h: hb,
        },
        MmChunk::Text(t2),
    ];

    // ---- exclusive path reference: prefill both images, greedy-decode
    let reuses_before = m.image_cache_reuses();
    let mut mm_ref = Vec::new();
    {
        let (logits, _rows) = m.forward_multimodal_chunks(&chunks).expect("two-image mm");
        let mut t = amax(&logits);
        mm_ref.push(t);
        for _ in 1..n_new {
            let l = m.forward_one(t).expect("fwd");
            t = amax(&l);
            mm_ref.push(t);
        }
    }
    // both images are first-sight here - no cache reuse yet
    assert_eq!(
        m.image_cache_reuses(),
        reuses_before,
        "reference encode is first sight of both"
    );
    eprintln!("two-image ref: {:?}", tok.decode(&mm_ref, false));

    // ---- batched-slot path: same two images in slot 0, must match exactly and
    // reuse both cached embeddings (2 reuses in one prefill)
    m.enable_batch(2).expect("enable_batch");
    let (l0, rows0) = m
        .forward_prefill_slot_mm(0, &chunks)
        .expect("two-image slot mm");
    assert_eq!(
        m.image_cache_reuses(),
        reuses_before + 2,
        "slot prefill must reuse both cached images"
    );
    let mut got0 = vec![amax(&l0)];
    for (p0, _) in (rows0 as u32..).zip(1..n_new) {
        let l = m
            .forward_batch(&[*got0.last().unwrap()], &[p0])
            .expect("batch step");
        got0.push(amax(&l[..vocab]));
    }
    eprintln!("two-image slot: {:?}", tok.decode(&got0, false));
    assert_eq!(
        got0, mm_ref,
        "two-image slot stream diverged from the exclusive path"
    );
    eprintln!("MULTI-IMAGE OK: 2 images, slot==single, both cache-served");
}

/// Tower-only gate (no 27B backbone): `encode_batch` must give every image the
/// answer it would have got alone. Only mmproj + pack needed.
///
/// This gate used to be a single numeric band against the serial encode
/// ("rel ~4e-6, pure f32 reorder noise"). That band was measured on a tower
/// that no longer exists: at one point every vision GEMM was f32
/// (`matvec_batch`), moved the tower to f16 weights with f16
/// activation staging to stop widening the mmproj in VRAM. Rounding the
/// activations to f16 between GEMMs raises the numeric floor by ~1000x, so the
/// band has been failing ever since - on a tower that is provably correct
/// Measured on an A6000, 27 layers, 308 rows:
///
///   one cuBLAS f16 GEMM 1280x1280 at 308 rows vs 616 rows  rel 1.55e-6
///   the same seed after 27 rounds of (GEMM -> round to f16)  rel 1.43e-3
///   the real tower, batch-of-2 vs serial   noise image       rel 1.06e-3
///                                          gradient image    rel 1.08e-3
///                                          this test's image rel 2.71e-3
///
/// The synthetic chain (random weights, no model) lands in the same place as
/// the real tower, its growth curve SATURATES - 5.1e-4 by step 6, 1.4e-3 by
/// step 26 - and the real number barely moves across images that share nothing.
/// Both are the signature of a quantization floor, not of a propagating error.
/// Chasing that band tighter would mean batch-invariant GEMMs (fixed split-K
/// regardless of M), a real throughput cost for a difference already below the
/// tower's own f16 resolution. Granite's tower documents the same ~0.2%
/// (gpu_model/granite/encode.rs).
///
/// So the gate asserts what a correct batched encode actually promises, and
/// those three things are bitwise:
///   1. determinism - the same call twice;
///   2. batch-position invariance - the same image at slot 0 and slot 1;
///   3. cross-image independence - image 0's rows must not move when the
///      other image in the batch is replaced wholesale.
///      Every plumbing break this gate exists to catch (row concat, per-image pos
///      blocks, the `vision_attn_at` window, the merger's 4-row split) breaks 2 or
///      3 by a nonzero bit, and neither can be moved by a cuBLAS heuristic. The
///      numeric comparison against serial stays as a coarse backstop: a genuine
///      break is O(1) relative, three orders above the floor.
#[test]
fn batched_encode_matches_serial() {
    use paddock_engine::gpu_model::qwen35::vision::VisionModel;
    if !common::heavy() {
        return;
    }
    let Some(dir) = common::model_dir("QWEN36_MM_DIR", common::QWEN36_27B_DIR) else {
        return;
    };
    let Some(mmproj_path) = ["mmproj-F16.gguf", "mmproj-BF16.gguf"]
        .iter()
        .map(|n| dir.join(n))
        .find(|p| p.exists())
    else {
        common::missing(&format!("no mmproj in {}", dir.display()));
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let mm = MappedGguf::open(&mmproj_path).expect("open mmproj");
    let vm = VisionModel::load(exec.clone(), &mm).expect("load tower");

    let (a, w, h) = red_blue_rgb();
    let mut a2 = a.clone();
    a2[0] = 1; // distinct pixels, same dims - a genuinely different image
    a2[4] = 200;
    // a3 shares A's dims but none of its pixels: the control for invariant 3,
    // where a2's two-byte edit would be too weak to prove anything
    let a3: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            if i % w < w / 2 {
                [0, 255, 0]
            } else {
                [255, 255, 0]
            }
        })
        .collect();
    let (ia, tw, th) = vm.preprocess_rgb(&a, w, h);
    let (ia2, tw2, th2) = vm.preprocess_rgb(&a2, w, h);
    let (ia3, tw3, th3) = vm.preprocess_rgb(&a3, w, h);
    assert_eq!((tw, th), (tw2, th2));
    assert_eq!((tw, th), (tw3, th3));

    let sa = vm.encode(&ia, tw, th).expect("serial a");
    let sa2 = vm.encode(&ia2, tw, th).expect("serial a2");
    let batch = vm
        .encode_batch(&[(ia.as_slice(), tw, th), (ia2.as_slice(), tw, th)])
        .expect("batch of two");
    assert_eq!(batch.len(), 2);
    assert_eq!((batch[0].nx, batch[0].ny), (sa.nx, sa.ny));
    assert_eq!((batch[1].nx, batch[1].ny), (sa2.nx, sa2.ny));
    let host = |o: &paddock_engine::gpu_model::qwen35::vision::VisionOutput| {
        exec.to_host(&o.embd).expect("to_host")
    };
    let (ba, ba2, ha, ha2) = (host(&batch[0]), host(&batch[1]), host(&sa), host(&sa2));
    assert_eq!(ba.len(), ha.len());
    let diff = |a: &[f32], b: &[f32]| {
        let (mut mx, mut n_ne, mut l2n, mut l2d) = (0f32, 0usize, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            if x.to_bits() != y.to_bits() {
                n_ne += 1;
            }
            mx = mx.max(d);
            l2n += (d as f64) * (d as f64);
            l2d += (*y as f64) * (*y as f64);
        }
        (mx, n_ne, (l2n / l2d.max(1e-30)).sqrt())
    };

    // 1. determinism - everything below reads a nonzero bit as a break, so the
    //    kernels have to be repeatable before any of it means anything
    let (_, ne, _) = diff(&host(&vm.encode(&ia, tw, th).expect("serial a again")), &ha);
    assert_eq!(
        ne, 0,
        "encode is not deterministic: {ne} elements differ between two calls"
    );

    // 2. batch-position invariance - the same image at slot 0 and slot 1 of one
    //    call. Row concat, the per-image pos block, the attention row window and
    //    the merger's 4-row split are all positional; any of them slipping shows
    //    up here and nothing else can move these bits.
    let bb = vm
        .encode_batch(&[(ia.as_slice(), tw, th), (ia.as_slice(), tw, th)])
        .expect("batch of A,A");
    let (bb0, bb1) = (host(&bb[0]), host(&bb[1]));
    let (mx, ne, _) = diff(&bb0, &bb1);
    assert_eq!(
        ne, 0,
        "batch slot 0 != slot 1 for the same image ({ne} elements, max {mx:.3e})"
    );

    // 3. cross-image independence - replace the other image wholesale and image
    //    0's rows must not move by one bit. Reading across the image boundary
    //    (attention window, pos stride, a GEMM row count) lands here.
    let bx = vm
        .encode_batch(&[(ia.as_slice(), tw, th), (ia3.as_slice(), tw, th)])
        .expect("batch of A,A3");
    let (mx, ne, _) = diff(&host(&bx[0]), &ba);
    assert_eq!(
        ne, 0,
        "image 0 moved when image 1 changed ({ne} elements, max {mx:.3e})"
    );

    // 4. coarse backstop against serial. Not bitwise and never will be: cuBLAS
    //    picks a different kernel at 2n rows, and the tower re-rounds its
    //    activations to f16 between GEMMs, which parks that seed on the f16
    //    floor (~1e-3 - see the doc comment). A real break is O(1) relative.
    let (mx, ne, rel) = diff(&ba, &ha);
    eprintln!("A : max {mx:.3e} ne {ne}/{} rel {rel:.3e}", ba.len());
    let (mx2, ne2, rel2) = diff(&ba2, &ha2);
    eprintln!("A2: max {mx2:.3e} ne {ne2}/{} rel {rel2:.3e}", ba2.len());
    let peak = ha.iter().fold(0f32, |m, v| m.max(v.abs()));
    assert!(
        rel < 1e-2 && rel2 < 1e-2,
        "batched encode rel err {rel:.3e}/{rel2:.3e} is O(1), not the f16 floor"
    );
    assert!(
        mx < 0.25 * peak && mx2 < 0.25 * peak,
        "batched encode max abs err {mx:.3e}/{mx2:.3e} against peak {peak:.3e}"
    );
    eprintln!(
        "batched encode: bitwise position-invariant and image-independent; \
         vs serial rel {rel:.2e}/{rel2:.2e} (f16 activation floor) over {} elements x2",
        ba.len()
    );
}

/// Batched-mm-prefill gate: two concurrent image requests through the
/// Generator batch API (one tower pass + one batched prefill pass) must
/// produce the same greedy streams as the serial per-slot path. Logits shift
/// within the f32 GEMM row-dispatch band under batching (as all batched
/// prefill does); the greedy stream is the arbiter.
#[test]
fn batched_mm_prefill_matches_serial() {
    use paddock_engine::generator::Generator;
    let Some((mut m, tok)) = setup() else { return };
    let vocab = m.vocab;
    let n_new = 10usize;

    let mk = |img: (Vec<u8>, usize, usize), q: &str| {
        vec![
            MmChunk::Text(tok.encode("Picture: ").expect("enc")),
            MmChunk::Image {
                rgb: img.0,
                w: img.1,
                h: img.2,
            },
            MmChunk::Text(tok.encode(q).expect("enc")),
        ]
    };
    let ca = mk(red_blue_rgb(), " The two dominant solid colors are");
    let cb = mk(green_yellow_rgb(), " The two dominant solid colors are");

    m.enable_batch(2).expect("enable_batch");
    // serial reference: per-slot mm prefills + joint decode
    let (la, rows_a) = m.forward_prefill_slot_mm(0, &ca).expect("serial a");
    let (lb, rows_b) = m.forward_prefill_slot_mm(1, &cb).expect("serial b");
    let mut ref_a = vec![amax(&la)];
    let mut ref_b = vec![amax(&lb)];
    let (mut pa, mut pb) = (rows_a as u32, rows_b as u32);
    for _ in 1..n_new {
        let l = m
            .forward_batch(&[*ref_a.last().unwrap(), *ref_b.last().unwrap()], &[pa, pb])
            .expect("batch step");
        ref_a.push(amax(&l[..vocab]));
        ref_b.push(amax(&l[vocab..2 * vocab]));
        pa += 1;
        pb += 1;
    }
    eprintln!("serial a: {:?}", tok.decode(&ref_a, false));
    eprintln!("serial b: {:?}", tok.decode(&ref_b, false));

    // batched path (images now cache-served -> same embeddings; the prefill
    // itself is the one concatenated pass under test)
    let items = vec![(0usize, ca.clone()), (1usize, cb.clone())];
    let res = Generator::forward_prefill_multimodal_batch(&mut m, items);
    assert_eq!(res.len(), 2);
    let (l0, r0) = res[0].1.as_ref().expect("batched a").clone();
    let (l1, r1) = res[1].1.as_ref().expect("batched b").clone();
    assert_eq!((r0, r1), (rows_a, rows_b), "row counts must match serial");
    let mut got_a = vec![amax(&l0)];
    let mut got_b = vec![amax(&l1)];
    let (mut pa, mut pb) = (r0 as u32, r1 as u32);
    for _ in 1..n_new {
        let l = m
            .forward_batch(&[*got_a.last().unwrap(), *got_b.last().unwrap()], &[pa, pb])
            .expect("batch step");
        got_a.push(amax(&l[..vocab]));
        got_b.push(amax(&l[vocab..2 * vocab]));
        pa += 1;
        pb += 1;
    }
    eprintln!("batched a: {:?}", tok.decode(&got_a, false));
    eprintln!("batched b: {:?}", tok.decode(&got_b, false));
    // Compare through the END-OF-MESSAGE token: past it the model predicts
    // continuations of an already-ended message on near-flat logits, where the
    // f32 GEMM row-dispatch band (batched r = ta+tb vs serial r = tb)
    // legitimately flips an argmax (observed: token 8 of 10, answers
    // identical). The semantic payload must match exactly.
    let cut = |s: &[u32], r: &[u32]| {
        // both streams end their answer with the same im_end id the serial
        // stream produced right after the answer text
        let eom = r.iter().position(|&t| {
            tok.decode(&[t], false)
                .is_ok_and(|d| d.contains("<|im_end|>"))
        });
        match eom {
            Some(i) => s[..=i.min(s.len() - 1)].to_vec(),
            None => s.to_vec(),
        }
    };
    assert_eq!(
        cut(&got_a, &ref_a),
        cut(&ref_a, &ref_a),
        "slot 0 answer diverged under batched mm prefill"
    );
    assert_eq!(
        cut(&got_b, &ref_b),
        cut(&ref_b, &ref_b),
        "slot 1 answer diverged under batched mm prefill"
    );
    eprintln!("BATCHED MM PREFILL OK: both answers match serial through end-of-message");
}
