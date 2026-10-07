//! Shared gate plumbing for the engine's integration tests.
//!
//! Every GPU gate needs the same two inputs - the kernel pack, and usually a
//! model file - and every gate used to resolve them itself. That is how
//! `gpu_whisper_kernels` came to report "12 passed" in 0.00 s having executed
//! nothing: it reads `PADDOCK_PACK`, the sweep exported `PADDOCK_TEST_PACK`,
//! and all twelve tests fell straight through their `else { return }`. Three
//! live spellings of "where is the pack" across 38 files, plus model paths
//! hardcoded to boxes that no longer exist (`/models/...`,
//! `C:/dev/models/...`). A green gate that never ran is worse than a red one:
//! red gets looked at.
//!
//! So: one name per input, a default that works on a bare checkout, and
//! `PADDOCK_STRICT_GATES=1` to turn every skip into a failure.
//!
//! - **pack** - `PADDOCK_PACK`, else the single pack built in
//!   `packs/cuda/build/`. The two old names still work and say that they are
//!   old, so an existing invocation keeps running instead of going quietly
//!   inert.
//! - **models** - `PADDOCK_MODELS` (a path list, authoritative), else
//!   `model_dirs` from the
//!   checkout's own `paddock.toml`, else `<data root>/models` via the one
//!   resolver in paddock-admin  - never a private `USERPROFILE`
//!   join, which is what put every gpt-oss gate permanently to sleep on a box
//!   whose models live on E:.
//!
//! Measured on the dev box before this landed, pack built and GPU present:
//! **68 of 141** engine tests reported ok having executed nothing.
//!
//! Skips print STRAIGHT to fd 2 rather than through `eprintln!`, because
//! libtest captures the macros and discards the capture when a test passes -
//! the whole reason a skipping gate was indistinguishable from a working one.
#![allow(dead_code)] // each test binary uses a different subset

use std::io::Write;
use std::path::{Path, PathBuf};

use paddock_engine::gpu::{GpuError, GpuExecutor};

/// The name. The other two are what the tests and scripts grew before anyone
/// lined them up.
const PACK_ENV: &str = "PADDOCK_PACK";
const PACK_ENV_OLD: [&str; 2] = ["PADDOCK_TEST_PACK", "PADDOCK_CUDA_PACK"];

/// Repo root, from the crate being compiled.
fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Straight to fd 2, deliberately not `eprintln!` - see the module note.
fn notice(msg: &str) {
    let _ = writeln!(std::io::stderr(), "{msg}");
}

/// `PADDOCK_STRICT_GATES=1` - "I expect these gates to RUN". Set it in any
/// sweep whose green you intend to believe.
fn strict() -> bool {
    matches!(
        std::env::var("PADDOCK_STRICT_GATES").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes")
    )
}

/// The one place a gate gives up. `None` so callers can `let Some(x) = ... else
/// { return }`; under strict mode not being able to run is the failure.
fn unavailable<T>(what: &str) -> Option<T> {
    assert!(
        !strict(),
        "gate cannot run: {what}\n(PADDOCK_STRICT_GATES is set - a skipped gate counts as failed)"
    );
    notice(&format!("SKIP: {what}"));
    None
}

/// Report an input this module can't resolve on the gate's behalf - a set of
/// files, a pack capability probe - with the same visibility and the same
/// strict-mode failure as everything else here.
pub fn missing(what: &str) {
    let _: Option<()> = unavailable(what);
}

/// Packs built in the checkout, platform extension only. `pd-cuda-sm86-t.lib`
/// and friends are link artifacts, not packs.
fn built_packs() -> Vec<PathBuf> {
    let dir = workspace().join("packs/cuda/build");
    let ext = if cfg!(windows) {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()) == Some(ext)
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("pd-cuda-"))
        })
        .collect();
    out.sort();
    out
}

/// The kernel pack the gates run against.
///
/// An env var that names a file which is not there is a mistake, not an absent
/// optional input, so it fails rather than skipping - you said where it was.
pub fn pack() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(PACK_ENV).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(p);
        assert!(p.exists(), "{PACK_ENV} names {} - not there", p.display());
        return Some(p);
    }
    for old in PACK_ENV_OLD {
        if let Some(p) = std::env::var_os(old).filter(|v| !v.is_empty()) {
            let p = PathBuf::from(p);
            // once per process, not once per test - every gate in a binary
            // calls this and the same line 12 times is noise, not information
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                notice(&format!(
                    "NOTE: {old} is the old name for {PACK_ENV} - honouring it"
                ));
            });
            assert!(p.exists(), "{old} names {} - not there", p.display());
            return Some(p);
        }
    }
    match built_packs().as_slice() {
        [one] => Some(one.clone()),
        [] => unavailable(&format!(
            "no kernel pack: {PACK_ENV} unset and nothing built in packs/cuda/build \
             (build it with packs/cuda/build.ps1)"
        )),
        many => {
            // Picking one silently is how you validate the wrong arch for an
            // afternoon. Newest-mtime would be a guess; say so instead.
            let names: Vec<_> = many
                .iter()
                .filter_map(|p| p.file_name()?.to_str())
                .collect();
            unavailable(&format!(
                "several packs built ({}) - set {PACK_ENV} to name one",
                names.join(", ")
            ))
        }
    }
}

/// Pack + CUDA device, the pair almost every gate opens with.
///
/// Crucially this does not swallow every error as "no CUDA - skipping", which
/// is what the hand-rolled versions did: a pack that fails to LOAD (stale ABI,
/// wrong magic, truncated table) is a defect and fails here. Only an absent or
/// unusable device is a skip.
/// Flash-Next (qwen4exp) pools at KV8 by default, and its KV class is one
/// process-global pick read from the environment at the first load
/// (gpu_model/qwen4exp/forward.rs `KV`). Its f32-reference gates are stated
/// at the exact f16 class, so their files call this before loading: it pins
/// f16 for the whole test process, once, before any load can read the pick.
pub fn qwen4exp_exact_kv() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: set once, before the first qwen4exp load in this process; the
    // engine reads it once (OnceLock) and nothing else in the suite reads it
    ONCE.call_once(|| unsafe { std::env::set_var("PADDOCK_Q38FN_KV8", "0") });
}

pub fn gpu() -> Option<GpuExecutor> {
    let pack = pack()?;
    match GpuExecutor::new(0, &pack) {
        Ok(e) => Some(e),
        Err(e @ GpuError::Pack(_)) => panic!("kernel pack {} is bad: {e}", pack.display()),
        Err(GpuError::Driver(e)) => unavailable(&format!("no usable CUDA device ({e})")),
        Err(GpuError::Unsupported(e)) => unavailable(&e),
        Err(e) => panic!("executor init failed on {}: {e}", pack.display()),
    }
}

/// Just the device, for the gates that drive a `KernelPack` directly instead
/// of going through the executor.
pub fn cuda() -> Option<std::sync::Arc<cudarc::driver::CudaContext>> {
    match cudarc::driver::CudaContext::new(0) {
        Ok(c) => Some(c),
        Err(e) => unavailable(&format!("no usable CUDA device ({e})")),
    }
}

/// Same, wrapped - the model loaders all take `Arc<GpuExecutor>`.
pub fn gpu_arc() -> Option<std::sync::Arc<GpuExecutor>> {
    gpu().map(std::sync::Arc::new)
}

/// `PADDOCK_HEAVY_TESTS` - the deliberate opt-in for gates that upload a whole
/// model. Skipping one is a CHOICE, not a missing input, so it stays a skip
/// even under strict mode. It just says so now, where a bare `return` did not.
pub fn heavy() -> bool {
    if std::env::var_os("PADDOCK_HEAVY_TESTS").is_some() {
        return true;
    }
    notice("SKIP: heavy gate (set PADDOCK_HEAVY_TESTS=1 to run it)");
    false
}

// The files the gates want, named the way the registry lays them out plus the
// bare name as a fallback. These follow the elected model line  -
// the `*-MTP-GGUF` repos for qwen3.5/3.6 - which is also why several of these
// gates had been dead for weeks: they pointed at pinned HF snapshot dirs and
// `C:/dev/models` paths from before the line was elected.
pub const GPT_OSS_20B: &[&str] = &[
    "gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf",
    "gpt-oss-20b-mxfp4.gguf",
];
pub const QWEN35_9B_Q8: &[&str] = &[
    "Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-Q8_0.gguf",
    "Qwen3.5-9B-Q8_0.gguf",
];
pub const QWEN35_9B_UD_Q4: &[&str] = &[
    "Qwen3.5-9B-MTP-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
    "Qwen3.5-9B-UD-Q4_K_XL.gguf",
];
pub const QWEN36_27B_Q8: &[&str] = &[
    "Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q8_0.gguf",
    "Qwen3.6-27B-Q8_0.gguf",
];
pub const QWEN36_27B_DIR: &[&str] = &["Qwen3.6-27B-MTP-GGUF"];
pub const QWEN36_MMPROJ: &[&str] = &["Qwen3.6-27B-MTP-GGUF/mmproj-BF16.gguf", "mmproj-BF16.gguf"];
pub const QWEN36_35B_A3B_Q8: &[&str] = &[
    "Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf",
    "Qwen3.6-35B-A3B-Q8_0.gguf",
];
pub const QWEN36_35B_A3B_UD_Q4: &[&str] = &[
    "Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
];
/// A DENSE i-quant file: every projection IQ2_XXS / IQ2_S / IQ3_S, the head
/// and the embedding included (the dense i-quant lanes, slot 578).
pub const QWEN35_9B_UD_IQ2: &[&str] = &[
    "Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-IQ2_XXS.gguf",
    "Qwen3.5-9B-IQ2/Qwen3.5-9B-UD-IQ2_XXS.gguf",
    "Qwen3.5-9B-UD-IQ2_XXS.gguf",
];

/// The i-quant demo file: IQ2_XXS / IQ2_S / IQ3_S expert seats, Q5_K/Q6_K dense.
pub const QWEN36_35B_A3B_UD_IQ2: &[&str] = &[
    "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
    "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
];
pub const GRANITE_8B_Q8: &[&str] = &[
    "granite-4.1-8b-GGUF/granite-4.1-8b-Q8_0.gguf",
    "granite-4.1-8b-Q8_0.gguf",
];
pub const GRANITE_30B_Q8: &[&str] = &[
    "granite-4.1-30b-GGUF/granite-4.1-30b-Q8_0.gguf",
    "granite-4.1-30b-Q8_0.gguf",
];
pub const GRANITE_30B_Q4: &[&str] = &[
    "granite-4.1-30b-GGUF/granite-4.1-30b-Q4_K_M.gguf",
    "granite-4.1-30b-Q4_K_M.gguf",
];
/// Host file for the bf16 dense-plane gate (`gpu_bf16_plane`).
///
/// This used to sit on a `UD-Q8_K_XL` file, which mixed bf16 `token_embd` /
/// `output` / `attn_k` / `attn_v` in among Q8_0. Once that quant was dropped
/// in favour of plain `Q8_0` the gate had nothing to load, and a skipping gate
/// covers nothing.
///
/// So it is homed on the **mmproj**, the practical source of real bf16 weights
/// here: 303 BF16 tensors next to 506 F32. Two things to be honest about:
///
///   - A `UD-Q4_K_XL` file will not do instead, tempting as "it carries the
///     same bf16 head" sounds. Those heads are quantized (`Q6_K` or `Q8_0`
///     depending on the model) and some such files carry no BF16 tensor at
///     all - the only BF16 left in an A3B file is a couple of MoE router
///     planes (`blk.N.ffn_gate_inp{,_shexp}`).
///   - The serving vision tower down-converts these planes to f16, so this
///     file is a *source of real bf16 weights*, not
///     a claim about which kernel the tower runs. The gate's subject is the
///     bf16 dense lane itself, which stays live for any gemma4-family file
///     carrying a bf16 tensor (`Plane::Bf16` in gemma4/mod.rs) even though no
///     currently-resident checkpoint takes that arm.
pub const MUSE_GLIMMER_MMPROJ: &[&str] = &[
    "Muse-Glimmer-30B-GGUF/mmproj-Muse-Glimmer-30B-BF16.gguf",
    "mmproj-Muse-Glimmer-30B-BF16.gguf",
];
/// The nemotron GGUF lane's serving file - also the llama.cpp same-weights
/// parity reference for this family. The NVFP4 checkpoint is a different
/// artifact entirely (`NEMOTRON_NVFP4_DIR`), so a Q8-lane gate cannot borrow it.
pub const NEMOTRON_30B_Q8: &[&str] = &[
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
];
pub const LAGUNA_XS_Q4: &[&str] = &[
    "Laguna-XS-2.1-GGUF/Laguna-XS-2.1-Q4_K_M.gguf",
    "Laguna-XS-2.1-Q4_K_M.gguf",
];
/// Kolibri 1's Q4_K_M (the laguna body's Kolibri flavor) - a pinned
/// local-development file; no llama.cpp release reads `kolibri1` yet.
pub const KOLIBRI_Q4KM: &[&str] = &[
    "Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf",
    "Kolibri-1-Q4_K_M.gguf",
];
/// The compressed-tensors NVFP4 checkpoint directory.
pub const KOLIBRI_NVFP4: &[&str] = &["Kolibri-1-NVFP4"];
/// EmbeddingGemma 2's text Q8_0 (unsloth, the CUDA lane's checkpoint).
pub const EMBEDDINGGEMMA2_Q8: &[&str] = &[
    "EmbeddingGemma-2-GGUF/embeddinggemma-2-Q8_0.gguf",
    "embeddinggemma-2-Q8_0.gguf",
];
/// Its mmproj (the same repo): the gemma4v picture and gemma4a audio towers.
pub const EMBEDDINGGEMMA2_MMPROJ: &[&str] = &["EmbeddingGemma-2-GGUF/mmproj-BF16.gguf"];
/// The catalog's cut of that file: the picture tower and the audio tower,
/// each its own download.
pub const EMBEDDINGGEMMA2_MMPROJ_VISION: &[&str] =
    &["EmbeddingGemma-2-GGUF/embeddinggemma-2-mmproj-vision-BF16.gguf"];
pub const EMBEDDINGGEMMA2_MMPROJ_AUDIO: &[&str] =
    &["EmbeddingGemma-2-GGUF/embeddinggemma-2-mmproj-audio-BF16.gguf"];
/// All three Nordic whisper fine-tunes: KB ships `proj_out` as its own plane,
/// NB and Røst tie it to the embedding, so loading the set is what covers the
/// tie fallback.
pub const WHISPER_NORDIC: [&[&str]; 3] = [
    &[
        "kb-whisper-large/kb-whisper-large-f16.gguf",
        "kb-whisper-large-f16.gguf",
    ],
    &[
        "nb-whisper-large/nb-whisper-large-f16.gguf",
        "nb-whisper-large-f16.gguf",
    ],
    &[
        "roest-v3-whisper-1.5b/roest-v3-whisper-1.5b-f16.gguf",
        "roest-v3-whisper-1.5b-f16.gguf",
    ],
];

/// Directories that may hold GGUFs, most specific first.
///
/// `PADDOCK_MODELS` REPLACES the ladder rather than prepending to it - same
/// contract as `PADDOCK_DATA` in the real resolver, and the only way a sweep
/// can say "look here and nowhere else" (which is what makes a strict-mode
/// negative test possible at all).
pub fn model_roots() -> Vec<PathBuf> {
    if let Some(list) = std::env::var_os("PADDOCK_MODELS").filter(|v| !v.is_empty()) {
        return std::env::split_paths(&list).collect();
    }
    let mut roots = Vec::new();
    // The checkout's own dev config already says where the models live
    // (`model_dirs`), and it is the same file the dev runner reads - so a gate
    // and the server it is gating agree without anyone exporting anything.
    let cfg = workspace().join("paddock.toml");
    if let Ok(text) = std::fs::read_to_string(&cfg)
        && let Ok(doc) = text.parse::<toml::Table>()
        && let Some(dirs) = doc.get("model_dirs").and_then(|v| v.as_array())
    {
        roots.extend(dirs.iter().filter_map(|d| d.as_str()).map(PathBuf::from));
    }
    roots.push(paddock_admin::data_root().join("models"));
    roots
}

/// Find a model file.
///
/// `env_var` is the per-family override the tests already had (`QWEN35_GGUF`,
/// `WHISPER_GGUF`, ...) - set it and that exact file is used. Otherwise each
/// candidate is tried as a path under every root, then by bare file name a few
/// levels down, because the registry lays models out as `<repo-dir>/<file>`.
pub fn model(env_var: &str, candidates: &[&str]) -> Option<PathBuf> {
    if !env_var.is_empty()
        && let Some(p) = std::env::var_os(env_var).filter(|v| !v.is_empty())
    {
        let p = PathBuf::from(p);
        assert!(p.exists(), "{env_var} names {} - not there", p.display());
        return Some(p);
    }
    let roots = model_roots();
    for root in &roots {
        for cand in candidates {
            let direct = root.join(cand);
            if direct.exists() {
                return Some(direct);
            }
        }
    }
    for root in &roots {
        for cand in candidates {
            let name = Path::new(cand).file_name()?;
            if let Some(hit) = find_named(root, name.to_str()?, 3) {
                return Some(hit);
            }
        }
    }
    let where_looked: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
    let hint = if env_var.is_empty() {
        String::new()
    } else {
        format!(" (or set {env_var} to the file)")
    };
    unavailable(&format!(
        "model not found: {} - looked under {}{hint}",
        candidates.join(" / "),
        where_looked.join(", ")
    ))
}

/// Same, for a gate that wants the whole model DIRECTORY (a backbone plus its
/// mmproj) rather than one file.
pub fn model_dir(env_var: &str, candidates: &[&str]) -> Option<PathBuf> {
    if !env_var.is_empty()
        && let Some(p) = std::env::var_os(env_var).filter(|v| !v.is_empty())
    {
        let p = PathBuf::from(p);
        assert!(
            p.is_dir(),
            "{env_var} names {} - not a directory",
            p.display()
        );
        return Some(p);
    }
    let roots = model_roots();
    for root in &roots {
        for cand in candidates {
            let d = root.join(cand);
            if d.is_dir() {
                return Some(d);
            }
        }
    }
    unavailable(&format!(
        "model directory not found: {} - looked under {}",
        candidates.join(" / "),
        roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// Breadth-limited hunt for a file name. Depth 3 covers `<root>/<repo>/<file>`
/// and one level of sharding without walking a whole model drive.
fn find_named(dir: &Path, name: &str, depth: usize) -> Option<PathBuf> {
    if depth == 0 {
        return None;
    }
    let entries = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Some(p);
        }
    }
    subdirs.sort();
    subdirs
        .into_iter()
        .find_map(|d| find_named(&d, name, depth - 1))
}
