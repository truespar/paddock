//! DeepEncoder end-to-end oracle gate: our tower vs the
//! checkpoint's own deepencoder.py (f32, CUDA) on identical deterministic
//! inputs, both of the family's real geometries.
//!
//! The oracle artifacts are produced out of tree
//! into `<models>/ocr-battery/oracle/` - ~30 MB of f32, deliberately outside
//! the repo. The gate skips cleanly when they are absent and fails under
//! `PADDOCK_STRICT_GATES=1` like every other gate.
//!
//! Numeric class: the oracle runs f32 end to end; our tower runs f16 weight
//! planes with f32 accumulation (the class every mmproj tower here runs in).
//! The gate is therefore CLASS-tolerance - mean |Δ| and relative L2 - plus a
//! per-run cosine, not an exact match. `PADDOCK_OCR_VIS_DUMP=1` on this test
//! prints our per-stage sums; `stages.json` next to the artifacts carries the
//! reference's, hook-matched to the same tags, for bisecting any failure to a
//! layer instead of a tower.

mod common;

use std::sync::Arc;

use paddock_engine::gpu_model::deepseek_ocr::encode::PatchSrc;
use paddock_engine::gpu_model::deepseek_ocr::vision::DeepEncoder;
use paddock_models::mapped::MappedGguf;

fn oracle_dir() -> Option<std::path::PathBuf> {
    common::model_roots()
        .iter()
        .map(|r| r.join("ocr-battery").join("oracle"))
        .find(|p| p.join("global_feats.bin").exists())
}

fn mmproj_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("UNLIMITED_OCR_MMPROJ") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    common::model_roots().iter().find_map(|r| {
        let p = r
            .join("Unlimited-OCR-GGUF")
            .join("mmproj-Unlimited-OCR-F16.gguf");
        p.exists().then_some(p)
    })
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The oracle stores pixels [B, 3, px, px] (torch CHW); `patch_rows` eats the
/// interleaved HWC our image ingestion produces. Convert per view.
fn chw_to_hwc(chw: &[f32], views: usize, px: usize) -> Vec<f32> {
    let plane = px * px;
    let mut out = vec![0f32; chw.len()];
    for b in 0..views {
        let src = &chw[b * 3 * plane..][..3 * plane];
        let dst = &mut out[b * 3 * plane..][..3 * plane];
        for c in 0..3 {
            for i in 0..plane {
                dst[i * 3 + c] = src[c * plane + i];
            }
        }
    }
    out
}

struct Diff {
    max_abs: f32,
    mean_abs: f64,
    rel_l2: f64,
    cosine: f64,
}

fn diff(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len());
    let (mut max_abs, mut sum_abs, mut d2, mut w2, mut g2, mut dot) =
        (0f32, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        d2 += (d as f64) * (d as f64);
        w2 += (w as f64) * (w as f64);
        g2 += (g as f64) * (g as f64);
        dot += (g as f64) * (w as f64);
    }
    Diff {
        max_abs,
        mean_abs: sum_abs / got.len() as f64,
        rel_l2: (d2 / w2).sqrt(),
        cosine: dot / (w2.sqrt() * g2.sqrt()),
    }
}

fn run_geometry(enc: &mut DeepEncoder, dir: &std::path::Path, run: &str, views: usize, px: usize) {
    let pixels = chw_to_hwc(&read_f32(&dir.join(format!("{run}_pixels.bin"))), views, px);
    let want = read_f32(&dir.join(format!("{run}_feats.bin")));
    let rows = enc.patch_rows(&pixels, views, px);
    let out = enc.encode(PatchSrc::F32(&rows), views, px).expect("encode");
    let got = enc_to_host(enc, &out.embd);
    assert_eq!(got.len(), want.len(), "{run}: feature count");
    let d = diff(&got, &want);
    // FNV-1a over the feature bits: the anchor a numerically-neutral tower
    // change (a fused epilogue, a dropped pass) is checked against - it must
    // not move
    let bits = got.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x0100_0000_01b3)
    });
    eprintln!(
        "{run}: max|Δ| {:.4}  mean|Δ| {:.5}  relL2 {:.5}  cos {:.6}  bits {bits:016x}",
        d.max_abs, d.mean_abs, d.rel_l2, d.cosine
    );
    // f16-weight/f32-accumulate class against an f32 oracle, 36 GEMM-deep.
    // The gate values are the measured class behavior with headroom (first
    // runs: relL2 ~2e-3, cos > 0.99999); a real graph bug - a wrong eps, a
    // swapped activation, a shifted window - moves relL2 by ORDERS, not
    // percent.
    assert!(
        d.rel_l2 < 0.01,
        "{run}: relative L2 {} over the class gate",
        d.rel_l2
    );
    assert!(
        d.cosine > 0.999,
        "{run}: cosine {} under the class gate",
        d.cosine
    );
}

fn enc_to_host(enc: &DeepEncoder, buf: &cudarc::driver::CudaSlice<f32>) -> Vec<f32> {
    enc.exec_ref().to_host(buf).expect("readback")
}

/// The u8 stem (pixels uploaded raw, normalize + im2row + f16 on
/// device) must land the exact tower output the f32 stem lands - the whole
/// graph after `s16` is deterministic, so bit-equal features prove the stem
/// itself is bit-exact. The reference arm builds the old host path by hand:
/// normalize_rgb8's expression verbatim, then `patch_rows`.
#[test]
fn u8_stem_matches_the_f32_stem() {
    let Some(path) = mmproj_path() else {
        common::missing("no Unlimited-OCR mmproj (set UNLIMITED_OCR_MMPROJ)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open mmproj");
    let mut enc = DeepEncoder::load(Arc::clone(&exec), &map).expect("load deepencoder");
    let (mean, std) = (enc.hp.image_mean, enc.hp.image_std);
    for (views, px) in [(1usize, 1024usize), (2, 640)] {
        let n = views * px * px * 3;
        let mut s = 0x00c0_ffeeu64;
        let pixels: Vec<u8> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        let norm: Vec<f32> = pixels
            .iter()
            .enumerate()
            .map(|(i, &b)| (b as f32 / 255.0 - mean[i % 3]) / std[i % 3])
            .collect();
        let rows = enc.patch_rows(&norm, views, px);
        let f32_arm = enc
            .encode(PatchSrc::F32(&rows), views, px)
            .expect("f32 stem");
        // the workspace backs `embd` - read arm A out before running arm B
        let want = enc_to_host(&enc, &f32_arm.embd);
        let u8_arm = enc
            .encode(PatchSrc::U8(&pixels), views, px)
            .expect("u8 stem");
        let got = enc_to_host(&enc, &u8_arm.embd);
        assert_eq!(got.len(), want.len(), "{views}x{px}: feature count");
        let diverged = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(
            diverged,
            0,
            "{views}x{px}: {diverged} of {} values differ",
            got.len()
        );
    }
}

/// The tower's seams landed in its GEMMs (bias + residual in the projections,
/// the FFN up landed f16 with its activation, f16 layernorms, the q|k|v split
/// in one pass) must land the composed tower's exact bits, at every view
/// count a request can chunk to - 1 and 8 views stress the GEMMs' smallest
/// and widest CLIP grids, where a shape may leave the fused tile for the
/// scratch fallback.
#[test]
fn fused_seams_match_the_composed_tower() {
    let Some(path) = mmproj_path() else {
        common::missing("no Unlimited-OCR mmproj (set UNLIMITED_OCR_MMPROJ)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open mmproj");
    let mut enc = DeepEncoder::load(Arc::clone(&exec), &map).expect("load deepencoder");
    if !exec.has_deepencoder_fused() {
        common::missing("pack without the DeepEncoder tower landings (slots 940-941)");
        return;
    }
    for (views, px) in [(1usize, 1024usize), (1, 640), (3, 640), (8, 640)] {
        let n = views * px * px * 3;
        let mut s = 0x5eed_0f0cu64 ^ (views * px) as u64;
        let pixels: Vec<u8> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect();
        enc.set_composed_seams(false);
        let fused = enc.encode(PatchSrc::U8(&pixels), views, px).expect("fused");
        let got = enc_to_host(&enc, &fused.embd);
        enc.set_composed_seams(true);
        let composed = enc
            .encode(PatchSrc::U8(&pixels), views, px)
            .expect("composed");
        let want = enc_to_host(&enc, &composed.embd);
        enc.set_composed_seams(false);
        let diverged = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(
            diverged,
            0,
            "{views}x{px}: {diverged} of {} values differ",
            got.len()
        );
    }
}

#[test]
fn deepencoder_matches_the_reference_on_both_geometries() {
    let Some(dir) = oracle_dir() else {
        common::missing("no DeepEncoder oracle");
        return;
    };
    let Some(path) = mmproj_path() else {
        common::missing("no Unlimited-OCR mmproj (set UNLIMITED_OCR_MMPROJ)");
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let map = MappedGguf::open(&path).expect("open mmproj");
    let mut enc = DeepEncoder::load(Arc::clone(&exec), &map).expect("load deepencoder");

    run_geometry(&mut enc, &dir, "global", 1, 1024);
    run_geometry(&mut enc, &dir, "crops", 3, 640);
}
