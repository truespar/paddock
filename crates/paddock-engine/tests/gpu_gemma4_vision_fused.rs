//! Gemma 4's vision tower: the fused pass (slots 820-825) must land the same
//! bits as the unfused chain it replaces, on every tower geometry we serve -
//! EmbeddingGemma 2's (768 wide, 16 layers, head dim 64, no standardize) and
//! the Gemma 4 chat models' (26B A4B: 1152 wide, 27 layers, head dim 72,
//! standardized tail). The fused kernels repeat the chain's arithmetic
//! operation for operation (gemma4v.cuh), so this is equality, not a
//! tolerance: a single differing bit fails it. Pictures at three grids,
//! wide, tall and square, through the tower's own (bilinear) resize.
//!
//! Env: PADDOCK_MODELS (the models dir), PADDOCK_PACK.

mod common;

use paddock_engine::gpu_model::gemma4::vision::{Resized, VisionModel};
use paddock_models::mapped::MappedGguf;

const MMPROJS: &[&[&str]] = &[
    common::EMBEDDINGGEMMA2_MMPROJ,
    &["gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-mmproj-BF16.gguf"],
];

/// A deterministic test picture: smooth gradients with a hard edge, so the
/// tower sees structure at every scale.
fn picture(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let edge = if (x * 7 + y * 3 + seed as usize) % 97 < 40 {
                60
            } else {
                0
            };
            out.push(((x * 255 / w.max(1)) as u32 ^ seed) as u8);
            out.push(((y * 255 / h.max(1)) as u32 + seed) as u8 ^ edge);
            out.push((((x + y) * 31 + seed as usize * 17) % 256) as u8);
        }
    }
    out
}

#[test]
fn fused_tower_is_bit_identical_to_the_unfused_chain() {
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    if !exec.has_g4v() {
        common::missing("a kernel pack with slots 820-825 (the fused gemma4v tower)");
        return;
    }
    let mut ran = 0;
    for candidates in MMPROJS {
        let Some(path) = common::model("PADDOCK_G4V_MMPROJ", candidates) else {
            continue;
        };
        let map = MappedGguf::open(&path).expect("open mmproj");
        let tower = VisionModel::load(exec.clone(), &map, None).expect("load tower");
        for (w, h, seed) in [(1200, 900, 1), (420, 760, 2), (512, 512, 3)] {
            let (resized, tw, th) = tower.resize_rgb(&picture(w, h, seed), w, h);
            let fused = tower
                .encode_resized(Resized::Host(&resized), tw, th)
                .expect("fused");
            let chain = tower
                .encode_resized_unfused(Resized::Host(&resized), tw, th)
                .expect("chain");
            assert_eq!(fused.n_tokens, chain.n_tokens);
            let a = exec
                .to_host_len(&fused.embd, fused.n_tokens * tower.llm_embd())
                .expect("a");
            let b = exec
                .to_host_len(&chain.embd, chain.n_tokens * tower.llm_embd())
                .expect("b");
            let differ = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            println!(
                "{} {w}x{h} -> {tw}x{th}: {} tokens, {differ} of {} values differ",
                path.file_name().unwrap().to_string_lossy(),
                fused.n_tokens,
                a.len()
            );
            assert_eq!(differ, 0, "the fused pass moved the tower's output");
            ran += 1;
        }
    }
    if ran == 0 {
        common::missing("an EmbeddingGemma 2 or Gemma 4 mmproj");
    }
}

/// `G4V_GOLDEN=<dir>`: write (`G4V_GOLDEN_WRITE=1`) or check every picture's
/// fused output against raw f32 files - how a change that has no in-tree
/// reference of its own (a weight re-lay) is held to the build before it.
#[test]
fn fused_tower_matches_recorded_goldens() {
    let Some(dir) = std::env::var_os("G4V_GOLDEN").map(std::path::PathBuf::from) else {
        return;
    };
    let write = std::env::var_os("G4V_GOLDEN_WRITE").is_some();
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    std::fs::create_dir_all(&dir).expect("golden dir");
    for candidates in MMPROJS {
        let Some(path) = common::model("PADDOCK_G4V_MMPROJ", candidates) else {
            continue;
        };
        let map = MappedGguf::open(&path).expect("open mmproj");
        let tower = VisionModel::load(exec.clone(), &map, None).expect("load tower");
        for (w, h, seed) in [(1200, 900, 1), (420, 760, 2), (512, 512, 3)] {
            let (resized, tw, th) = tower.resize_rgb(&picture(w, h, seed), w, h);
            let o = tower
                .encode_resized(Resized::Host(&resized), tw, th)
                .expect("encode");
            let got = exec
                .to_host_len(&o.embd, o.n_tokens * tower.llm_embd())
                .expect("rows");
            let file = dir.join(format!(
                "{}-{w}x{h}.f32",
                path.file_stem().unwrap().to_string_lossy()
            ));
            let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
            if write {
                std::fs::write(&file, bytes).expect("write golden");
            } else {
                let want = std::fs::read(&file).expect("golden");
                assert!(want == bytes, "{} moved", file.display());
                println!("{} matches its golden", file.display());
            }
        }
    }
}
