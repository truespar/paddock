//! The paddock **runner** - the data plane. One process, one served model, one
//! port, the full API surface (OpenAI chat + Responses, Anthropic messages,
//! embeddings/rerank). Headless and stateless on disk: no SQLite, no Studio
//! assets, no NVML - those live in the manager (`paddock`), which supervises
//! runners over the local admin surface. See

pub mod admin;
pub mod alignments;
pub mod chat;
pub use paddock_mcp::clock;
pub mod chat_template;
pub mod completions;
pub mod config;
pub mod constrained;
pub mod context_management;
pub mod deepseek_ocr;
pub mod doc;
pub mod drain;
pub mod embeddings;
pub mod events;
pub mod extract;
pub mod filters;
pub mod forced_align;
pub mod forensics;
pub mod harmony;
pub mod language;
pub mod messages;
pub mod metrics;
pub mod muse;
pub mod paddle_ocr;
pub mod parsers;
pub mod pdf;
pub mod ratelimit;
pub mod realtime;
pub mod reasoning;
pub mod responses;
pub mod routes;
pub mod service;
pub mod serving;
pub mod startup;
pub mod stats;
pub mod subtitles;
pub mod tiffdoc;
pub use paddock_mcp::{loop_budget, tool_search};
pub mod transcriptions;
pub mod websearch;

use std::sync::Arc;

use config::Config;
use paddock_models::ModelStore;
use paddock_models::mapped::MappedGguf;
use routes::AppState;

/// Resolve a bare model name against the installed files in `model_dirs`.
/// The runner never downloads - catalog pulls are the manager's job - so an
/// uninstalled name is a loud error naming the fix, not a silent wait.
/// Companions (mmproj / MTP drafter) are picked up from the model's own
/// directory, mirroring the layout the manager's catalog pulls produce.
fn resolve_local_model(cfg: &mut Config) -> Result<(), String> {
    let Some(model) = cfg.model.clone() else {
        return Ok(());
    };
    if model.exists() {
        // A direct weights path gets the same companion discovery a
        // by-name resolve does. This branch used to return before the scan,
        // so `--model <dir>/X.gguf` silently served an mm model without its
        // sibling `X-mmproj.gguf` (caught live on PaddleOCR-VL) -
        // the two entry forms must behave identically.
        discover_companions(cfg, &model);
        return Ok(());
    }
    // Not a path on disk - treat it as an installed-model id and scan.
    let name = model.to_string_lossy().to_string();
    let store = ModelStore::new(cfg.model_dirs.clone());
    let found = store
        .list()
        .map_err(|e| format!("model scan failed: {e}"))?
        .into_iter()
        .find(|m| m.id == name);
    let Some(found) = found else {
        return Err(format!(
            "model {name:?} is neither a file nor an installed model in {:?}. \
             The runner does not download models - pull it with the manager \
             (`paddock pull {name}`) or pass a GGUF path.",
            cfg.model_dirs
        ));
    };
    tracing::info!(name = %name, weights = %found.path.display(), "resolved installed model by name");
    discover_companions(cfg, &found.path);
    cfg.model = Some(found.path);
    Ok(())
}

/// Fill unset companion paths (mmproj / MTP drafter) from the weights'
/// directory - the layout catalog pulls produce. Explicit `--mmproj`/`--mtp`
/// always win; a multi-model directory yields the first match in readdir
/// order, so mixed layouts should keep one model per directory.
fn discover_companions(cfg: &mut Config, weights: &std::path::Path) {
    if cfg.mmproj.is_some() && cfg.mtp.is_some() {
        return;
    }
    // A checkpoint DIRECTORY (safetensors-primary lane) carries everything
    // inside itself; scanning its PARENT would treat unrelated sibling
    // models' companions as this model's.
    if weights.is_dir() {
        return;
    }
    let Some(dir) = weights.parent() else { return };
    let weights_name = weights.file_name().map(|n| n.to_os_string());
    let mut mmproj = None;
    let mut mtp = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            // never resolve the weights file itself as its own companion
            // (e.g. a main GGUF whose NAME contains "mmproj")
            if Some(e.file_name()) == weights_name {
                continue;
            }
            let fname = e.file_name().to_string_lossy().to_lowercase();
            if !fname.ends_with(".gguf") {
                continue;
            }
            if fname.contains("mmproj") {
                mmproj.get_or_insert(e.path());
            } else if fname.starts_with("mtp") || fname.contains("-mtp.") {
                mtp.get_or_insert(e.path());
            }
        }
    }
    if cfg.mmproj.is_none()
        && let Some(p) = mmproj
    {
        tracing::info!(mmproj = %p.display(), "companion mmproj discovered beside the weights");
        cfg.mmproj = Some(p);
    }
    if cfg.mtp.is_none()
        && let Some(p) = mtp
    {
        tracing::info!(mtp = %p.display(), "companion MTP drafter discovered beside the weights");
        cfg.mtp = Some(p);
    }
}

/// Bind and serve until the process is stopped. Caller sets up tracing.
pub async fn run(
    mut cfg: Config,
    mut banner: startup::Banner,
) -> Result<(), Box<dyn std::error::Error>> {
    resolve_local_model(&mut cfg)?;

    // [kv_offload]: the shipped tier election, armed once before any model
    // loads (families read it at enable_batch). Budgets only, by design.
    //
    // Every way of asking for offload and not getting it is REPORTED. These
    // combinations used to fall through to `None` in silence, which is the
    // precise shape of failure the no-silent-failures principle exists to
    // forbid: the user writes a config, the runner accepts it, and the
    // feature is simply not there.
    let kv = &cfg.kv_offload;
    if !kv.enabled && (kv.ram_gb > 0.0 || kv.nvme_gb > 0.0 || kv.nvme_path.is_some()) {
        tracing::warn!(
            "[kv_offload] has budgets set but enabled = false - no prefix cache \
             offload will run. Set enabled = true, or remove the budgets."
        );
    }
    if kv.enabled && kv.ram_gb <= 0.0 {
        tracing::warn!(
            "[kv_offload] enabled = true but ram_gb = 0 - nothing is armed. The RAM \
             tier is the entry point; the disk tier stores through it, so ram_gb \
             must be set for either to run."
        );
    }
    if kv.enabled && kv.nvme_path.is_some() && kv.nvme_gb <= 0.0 {
        tracing::warn!(
            "[kv_offload] nvme_path is set but nvme_gb = 0 - the disk tier has no \
             budget and will not arm. Set nvme_gb to the space you want it to use."
        );
    }
    let ram_armed = kv.enabled && kv.ram_gb > 0.0;
    // The BUDGET is the decision; the location has a sane default. A disk
    // budget with no path used to arm nothing and merely say so, which made
    // the commonest setup ("give it 200 GB") a two-field affair for no
    // reason. It lands under the box's data root - the same resolver the
    // manager and runner share, so a portable copy keeps its cache with it.
    let nvme_armed = match (kv.enabled, kv.nvme_gb) {
        (true, gb) if gb > 0.0 && ram_armed => {
            // The data ROOT, not root/kv-cache: the store appends its own
            // `kv-cache` segment (NvmeStore::dir_for), which is what keeps a
            // path pointed at a populated directory from being swept. So this
            // field means "the folder to put the cache in", and it means the
            // same thing whether it was defaulted or typed - otherwise a user
            // who pasted the shown default would land in kv-cache/kv-cache.
            let path = kv
                .nvme_path
                .clone()
                .unwrap_or_else(paddock_admin::data_root);
            Some((path, gb))
        }
        _ => None,
    };
    if let Some((path, gb)) = &nvme_armed {
        // A quota bigger than the volume is a promise the disk cannot keep.
        // Reported, not refused: the store evicts on quota, so an optimistic
        // budget degrades to a smaller cache rather than failing to serve.
        match fs4::available_space(path).ok() {
            Some(free) => {
                let want = (gb * (1u64 << 30) as f64) as u64;
                if want > free {
                    tracing::warn!(
                        want_gib = gb,
                        free_gib = free as f64 / (1u64 << 30) as f64,
                        path = %path.display(),
                        "[kv_offload] nvme_gb exceeds the free space on that volume - \
                         the cache will evict against a quota it can never reach"
                    );
                }
            }
            None => tracing::warn!(
                path = %path.display(),
                "[kv_offload] could not read free space for nvme_path - the disk tier \
                 will still arm, but its budget is unchecked"
            ),
        }
    }
    paddock_engine::kv_tier::pool_tier::set_tier_ram_bytes(
        ram_armed.then_some((kv.ram_gb * (1u64 << 30) as f64) as u64),
    );
    paddock_engine::kv_tier::pool_tier::set_tier_nvme(
        nvme_armed
            .clone()
            .map(|(p, gb)| (p, (gb * (1u64 << 30) as f64) as u64)),
    );
    // [moe_offload]: armed the same way, read by the MoE families at load
    // (host-map the expert planes) and at enable_batch (seat the slot cache).
    let mo = &cfg.moe_offload;
    if !mo.enabled && mo.vram_gb.is_some() {
        tracing::warn!(
            "[moe_offload] vram_gb is set but enabled = false - experts stay VRAM-resident \
             and a model that does not fit will be refused. Set enabled = true."
        );
    }
    paddock_engine::gpu::set_moe_offload(paddock_engine::gpu::MoeOffloadCfg {
        enabled: mo.enabled,
        vram_bytes: mo
            .vram_gb
            .map(|g| (g.max(0.0) * (1u64 << 30) as f64) as u64),
    });
    if mo.enabled {
        tracing::info!(
            vram_gb = mo.vram_gb,
            "[moe_offload] MoE expert offload armed (slot cache auto-sized from the KV plan's leftover unless vram_gb caps it)"
        );
    }
    if let Some(mib) = cfg.graph_scratch_mib {
        paddock_engine::kv_plan::set_graph_scratch_mib(mib);
        tracing::info!(
            graph_scratch_mib = mib,
            "KV-plan graph/scratch reserve overridden (defaults: gpt-oss 3072 MiB graph/prefill scratch, qwen35 768 MiB graph pools + headroom)"
        );
    }
    if ram_armed {
        tracing::info!(
            ram_gb = kv.ram_gb,
            nvme_gb = (kv.nvme_gb > 0.0).then_some(kv.nvme_gb),
            // where it actually landed, defaulted or not - a location the
            // operator did not type is exactly the one worth stating
            nvme_path = nvme_armed.as_ref().map(|(p, _)| p.display().to_string()),
            "[kv_offload] KV offloading armed"
        );
    }

    // `gpu` config -> CUDA ordinal, resolved natively against the driver (a
    // UUID pin is enumeration-order-proof). A real config field - no
    // CUDA_VISIBLE_DEVICES anywhere, so a manager-started, service-started,
    // and hand-started server all pin identically from the same file.
    let gpu_ordinal = match (&cfg.gpu, cfg.device.as_str()) {
        (Some(sel), "cuda") => {
            let ord = paddock_engine::cuda::resolve_device(sel)
                .map_err(|e| format!("gpu = {sel:?}: {e}"))?;
            tracing::info!(gpu = %sel, ordinal = ord, "gpu selector resolved against the CUDA driver");
            ord
        }
        (Some(sel), other) => {
            tracing::warn!(gpu = %sel, device = other, "gpu selector ignored - device is not cuda");
            0
        }
        (None, _) => 0,
    };

    // Serve on the 16-token KV page grid: every paged family manages KV in
    // 16-token blocks, and an off-grid max_ctx used to silently disqualify
    // paging entirely (dense fallback, no prefix cache - a trap class the
    // dense lane's removal closed). Rounding up grants slightly more
    // context than asked - never less, and never a regime change.
    if !cfg.max_ctx.is_multiple_of(16) {
        let rounded = cfg.max_ctx.div_ceil(16) * 16;
        tracing::info!(
            "max_ctx {} rounded up to {} (16-token KV page grid)",
            cfg.max_ctx,
            rounded
        );
        cfg.max_ctx = rounded;
    }

    // load the served model synchronously at startup (loud failure if it can't).
    // Encoder architectures (qwen3) serve /v1/embeddings + /v1/rerank; the rest
    // are generative and serve chat/completions.
    let (mut serving, mut embedder, mut asr, mut aligner) = (None, None, None, None);
    // the resolved off policy, surfaced on admin identify (SpecInfo.off)
    let mut spec_policy_off = false;
    if let Some(path) = &cfg.model {
        // A DIRECTORY has no extension to strip, and `file_stem` would cut it
        // at the last dot - which is inside the version on every checkpoint
        // directory we serve: `granite-4.2-8b-nvfp4` came out as `granite-4`
        // and `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` as
        // `NVIDIA-Nemotron-3`. A GGUF still wants the stem, so that the
        // `.gguf` suffix goes away.
        let id = if path.is_dir() {
            path.file_name()
        } else {
            path.file_stem()
        }
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_owned());
        let arch = MappedGguf::open(path)
            .ok()
            .and_then(|m| m.gguf().architecture().map(str::to_owned))
            .unwrap_or_default();
        tracing::info!(model = %path.display(), device = %cfg.device, %arch, "loading model");
        // arch `qwen3` is ambiguous: bare = the embeddings/rerank encoder;
        // paired with an AUDIO mmproj = the Qwen3-ASR generative family
        // (routed through serving::load like every other generator).
        let audio_companion = cfg.mmproj.as_deref().is_some_and(serving::mmproj_is_audio);
        // Normalize --kv-cache-dtype into the env transport the engine reads
        // (the PADDOCK_MAX_OUTPUT_TOKENS pattern) before the arch branch:
        // whisper serves through `load_asr`, which is not the generative
        // lane, and leaving this inside the generative branch made the flag a
        // silent no-op for the one family whose KV bytes dominate its wall.
        match cfg.kv_cache_dtype.as_str() {
            "auto" => {}
            "f16" => unsafe {
                // gemma4 and both ASR families default to fp8 - this is the
                // way back to exact f16. G4_KV16 is gemma4's own historical
                // spelling, kept so existing scripts keep working; the
                // generic one is what every other family reads.
                std::env::set_var("PADDOCK_G4_KV16", "1");
                std::env::set_var("PADDOCK_KV_CACHE_DTYPE", "f16");
            },
            "fp8_e4m3" => unsafe { std::env::set_var("PADDOCK_KV_CACHE_DTYPE", "fp8_e4m3") },
            other => {
                return Err(
                    format!("--kv-cache-dtype {other:?}: expected auto, f16, or fp8_e4m3").into(),
                );
            }
        }
        if let Some(dir) = serving::aligner_dir(path) {
            // The safetensors-primary route: a checkpoint dir
            // (or its entry-point .safetensors file - the form a catalog
            // spawn hands over) whose config.json names the forced-aligner
            // arch. Trained context is 8192; a larger server-wide --max-ctx
            // simply clamps here rather than refusing a model that never
            // needs more.
            //
            // The default id is the DIRECTORY's full name, not the stem the
            // other families use: file_stem on "Qwen3-ForcedAligner-0.6B-hf"
            // eats ".6B-hf" as an extension and serves the model as
            // "Qwen3-ForcedAligner-0", which fails the honest-naming bar.
            let dir_id = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or(id);
            let m = serving::load_aligner(
                cfg.served_model_name.clone().unwrap_or(dir_id),
                &dir,
                &cfg.device,
                gpu_ordinal,
                cfg.kernel_pack.as_deref(),
                cfg.max_ctx.min(8192),
                cfg.vram_budget.map(|mib| mib << 20),
            )?;
            tracing::info!(model = %m.id, "forced-alignment model ready");
            aligner = Some(m);
        } else if serving::is_asr_arch(&arch) {
            // whisper is speech-to-text only: encoder-decoder, no text
            // prompt, no chat surface - it serves /v1/audio/transcriptions
            // and nothing else, so it never reaches the generative lane.
            let m = serving::load_asr(
                cfg.served_model_name.clone().unwrap_or(id),
                path,
                &cfg.device,
                gpu_ordinal,
                cfg.kernel_pack.as_deref(),
                cfg.max_ctx,
                cfg.max_batch,
                cfg.vram_budget.map(|mib| mib << 20),
            )?;
            tracing::info!(model = %m.id, "speech-to-text model ready");
            asr = Some(m);
        } else if serving::is_encoder_arch(&arch) && !audio_companion {
            let m = serving::load_embedder(
                id,
                path,
                &cfg.device,
                gpu_ordinal,
                cfg.kernel_pack.as_deref(),
                cfg.max_ctx,
                cfg.vram_budget.map(|mib| mib << 20),
            )?;
            // Load-time quality calibration: pick the fastest block-scale
            // class profile this model holds retrieval quality on (FP4
            // tolerance is model-specific - measured 8B-clean profiles fail
            // on 4B), instead of serving the most conservative mix
            // everywhere. Deterministic; a few encode passes of load time,
            // so the verdict is CACHED on disk keyed by (model, pack)
            // fingerprints. PADDOCK_BS_CALIB=off (or any explicit
            // PADDOCK_BS_* pin, which the engine records) skips it;
            // =force ignores the cache and re-measures.
            let calib_mode = paddock_models::dev_var!("PADDOCK_BS_CALIB").unwrap_or_default();
            if calib_mode != "off" {
                let (cache, cached) = serving::CalibCache::probe(path, cfg.kernel_pack.as_deref());
                let mut applied = false;
                if calib_mode != "force"
                    && let Some((profile, smooth)) = cached
                {
                    match m.encoder.apply_profile(profile.clone(), smooth).await {
                        Ok(true) => {
                            tracing::info!(model = %m.id, profile, "block-scale profile from cache");
                            applied = true;
                        }
                        Ok(false) => {} // pinned/unknown/missing stats: fall through
                        Err(e) => {
                            tracing::warn!(model = %m.id, error = %e, "cached profile apply failed; recalibrating")
                        }
                    }
                }
                if !applied {
                    // rerankers get the rank-quality metric (their embedding
                    // space is not a retrieval space - the embed-recall task
                    // measured pure noise on one); embedders get embed recall
                    let verdict = if m.is_reranker {
                        match (m.yes_id, m.no_id) {
                            (Some(yes), Some(no)) => {
                                let groups = serving::calib_rerank_corpus();
                                let group = groups[0].1.len();
                                let mut seqs = Vec::with_capacity(groups.len() * group);
                                let mut rel = Vec::with_capacity(groups.len());
                                for (query, docs, rel_idx) in &groups {
                                    for doc in docs {
                                        // exactly the /v1/rerank tokenization (no EOS)
                                        let p = embeddings::rerank_prompt(
                                            embeddings::DEFAULT_INSTRUCT,
                                            query,
                                            doc,
                                        );
                                        seqs.push(
                                            m.tokenizer.encode(&p).map_err(|e| e.to_string())?,
                                        );
                                    }
                                    rel.push(*rel_idx);
                                }
                                Some(m.encoder.calibrate_rerank(seqs, yes, no, group, rel).await)
                            }
                            _ => {
                                tracing::warn!(model = %m.id, "reranker lacks yes/no tokens; skipping calibration");
                                None
                            }
                        }
                    } else {
                        let (texts, n_docs, rel) = serving::calib_corpus();
                        let seqs: Vec<Vec<u32>> = texts
                            .iter()
                            .map(|t| {
                                let mut e = m.tokenizer.encode(t).map_err(|e| e.to_string())?;
                                if let Some(eos) = m.eos {
                                    e.push(eos);
                                }
                                Ok::<_, String>(e)
                            })
                            .collect::<Result<_, _>>()?;
                        Some(m.encoder.calibrate(seqs, n_docs, rel).await)
                    };
                    match verdict {
                        Some(Ok(profile)) => {
                            tracing::info!(model = %m.id, profile, "block-scale calibration");
                            // "pinned" = env-selected classes, not a verdict
                            if profile != "pinned" {
                                // smooth profiles cache their s-vectors so
                                // warm starts skip the stats encode entirely
                                // (all smooth rung names carry "smooth" or
                                // the f8s marker)
                                let smooth =
                                    if profile.contains("smooth") || profile.contains("f8s") {
                                        m.encoder.export_smooth().await.ok().flatten()
                                    } else {
                                        None
                                    };
                                cache.store(profile, smooth);
                            }
                        }
                        Some(Err(e)) => {
                            tracing::warn!(model = %m.id, error = %e, "block-scale calibration failed; keeping the static default")
                        }
                        None => {}
                    }
                }
            }
            tracing::info!(model = %m.id, "encoder ready");
            embedder = Some(m);
        } else {
            let mut spec_off = cfg.no_spec;
            if let Some(s) = &cfg.spec {
                // Validate here, at startup, where the operator is looking -
                // the engine's own parse falls back to the default with a
                // warning, and a typo'd policy that silently serves something
                // else is exactly the silent failure the principles forbid.
                let p: paddock_engine::spec_policy::SpecPolicy = s
                    .parse()
                    .map_err(|e: String| format!("config `spec`: {e}"))?;
                if cfg.no_spec && !p.is_off() {
                    return Err(format!(
                        "config sets both no_spec = true and spec = {s:?} - pick one \
                         (no_spec is the older spelling of spec = \"off\")"
                    )
                    .into());
                }
                spec_off = spec_off || p.is_off();
                unsafe { std::env::set_var("PADDOCK_SPEC", p.to_string()) };
            }
            // The engine-wide speculative/MTP kill switch, set from the
            // RESOLVED policy - `no_spec` alone was not enough.
            //
            // `spec_off` below only suppresses a SIDELOADED drafter file, and
            // that is the whole story for a model whose drafter is a separate
            // GGUF (muse's dflash). A model carrying MTP IN-FILE (qwen3.5/3.6
            // `nextn`) has no `cfg.mtp` to withhold: its MTP block loads with
            // the weights, and the only thing that stops it is this switch -
            // which used to be set only by the legacy `no_spec` boolean. So
            // `spec = "off"` loaded the MTP block anyway and the KV planner
            // then reserved its draft state (qwen35/batch.rs gates `spec_est`
            // on `serve_spec_on()`, i.e. on PADDOCK_QWEN35_SPEC, which this
            // kill switch clears).
            //
            // Measured on the 9B Q8 at 4096x32: 2.08 GiB reserved for a
            // drafter the scheduler had already been told never to use, out of
            // a 21.1 GiB budget that then had 1.53 GiB left for a KV pool
            // needing 4.00 - i.e. the endpoint refused to start over VRAM held
            // for a feature that was off. The config doc calls "off" "the only
            // setting that gives the VRAM back"; now it is.
            if spec_off {
                unsafe { std::env::set_var("PADDOCK_NO_SPEC", "1") };
            }
            // spec "off" documents "no drafter loaded" - honor it. Loading
            // anyway is not inert: attach_mtp trades the nospec F8CUT lane
            // away process-wide (set_f8cut_spec_off) and pays the drafter's
            // VRAM. Companion auto-discovery on direct paths made
            // this reachable env-free, and it silently cost the gemma nospec
            // lane throughput until it was caught.
            let mtp = if spec_off {
                if cfg.mtp.is_some() {
                    tracing::info!(
                        "spec policy is off - MTP drafter left unloaded \
                         (spec \"off\" loads no drafter)"
                    );
                }
                None
            } else {
                cfg.mtp.as_deref()
            };
            let id = cfg.served_model_name.clone().unwrap_or(id);
            let m = serving::load(
                id,
                path,
                &cfg.device,
                gpu_ordinal,
                cfg.kernel_pack.as_deref(),
                cfg.max_ctx,
                cfg.max_batch,
                cfg.mmproj.as_deref(),
                mtp,
                cfg.fp8_native.as_deref(),
                // config carries MiB (nvidia-smi units); the engine takes bytes
                cfg.vram_budget.map(|mib| mib << 20),
            )?;
            tracing::info!(model = %m.id, "model ready");
            spec_policy_off = spec_off;
            serving = Some(m);
        }
    }

    // Auth policy: an explicit key wins; a non-loopback bind with no key
    // generates and requires one (in-process - the runner is stateless and has
    // no key store; the manager configures keys at spawn per doc §5.1).
    // --no-auth is the explicit opt-out for network binds - never silent.
    if cfg.no_auth && !cfg.host.is_loopback() {
        tracing::warn!(
            "auth DISABLED on a network bind (--no-auth): every peer that can \
             reach this address has full API access"
        );
    }
    let auth_key = match cfg.api_key.as_deref().filter(|s| !s.is_empty()) {
        Some(k) => Some(k.to_owned()),
        None if cfg.no_auth => None,
        None if !cfg.host.is_loopback() => Some(format!("pk-{}", uuid::Uuid::new_v4().simple())),
        None => None,
    };
    banner.auth_key = auth_key.clone();

    // Engine stats sampler: the runner's self-report (tok/s, batch, KV, phase,
    // allocator-ledger VRAM). No NVML here - device telemetry is the manager's
    // job: the inside view and the outside view must come from
    // different processes or the reconciliation cross-check is worthless.
    // Encoder runners have no generative engine, so they fell through to
    // `engine: null` and published no memory at all. They do hold
    // weights, so they get a metrics handle carrying the memory rows.
    let engine_metrics = serving
        .as_ref()
        .map(|s| s.engine.metrics())
        .or_else(|| embedder.as_ref().map(|e| Arc::clone(&e.metrics)))
        .or_else(|| asr.as_ref().map(|a| Arc::clone(&a.metrics)));
    let stats = crate::stats::start(engine_metrics.clone());
    // Held for the graceful-shutdown path below: on SIGINT/SIGTERM the engine
    // thread drops the generator (freeing all device memory) before the
    // process exits. Dying without freeing leaves the driver to reclaim the
    // dead ~27 GB context asynchronously, which stalls every other CUDA
    // process on the card for the next ~1-2 minutes - measured as ~850 ms
    // whole-wave TTFT stalls in a neighboring server.
    let engine_shutdown = serving.as_ref().map(|s| s.engine.clone());

    // Generation identity: a UUID minted once per PROCESS START,
    // never persisted - a restart is precisely the boundary where counters
    // and event sequences reset, so the id must change with it. This is what
    // the manager keys activity/usage on; `started_at_unix` stays for reset
    // detection only (keying on it collided at second resolution).
    let instance_id = uuid::Uuid::new_v4().to_string();

    // /metrics registry: second sink on the numbers the event
    // ring already measures, plus scrape-time engine gauges. Gated
    // independently of the ring - --no-events must not kill /metrics.
    let metrics = if cfg.no_metrics {
        crate::metrics::Metrics::disabled()
    } else {
        crate::metrics::Metrics::new(
            instance_id.clone(),
            crate::metrics::ModelIds {
                serving: serving.as_ref().map(|m| m.id.clone()),
                embedder: embedder.as_ref().map(|e| e.id.clone()),
                asr: asr.as_ref().map(|a| a.id.clone()),
            },
            engine_metrics,
        )
    };
    // Snapshot ring: a minute-cadence self-snapshot of
    // the counter set, so a manager that was away can reconstruct its blind
    // window instead of recording one opaque gap. Rides the metrics gate -
    // with --no-metrics the task never spawns and the ring stays empty.
    let snapshots = crate::metrics::SnapshotRing::new();
    if metrics.enabled() {
        crate::metrics::start_snapshots(metrics.clone(), snapshots.clone());
    }
    // The metrics-auth posture, said out loud where it bites: a Prometheus-
    // style scraper may send no auth header at all, so a network bench will
    // silently lose its server-metrics half unless the scrape is opened.
    if !cfg.no_metrics
        && !cfg.host.is_loopback()
        && auth_key.is_some()
        && cfg.metrics_auth != Some(false)
    {
        tracing::info!(
            "/metrics: open for loopback scrapes, API key required from the network \
             (a remote benchmark run needs `metrics_auth = false` - its scraper cannot send a key)"
        );
    }

    // Web-search provider for the server-executed web_search tool (API
    // conformance). Declared config, not a DB row - the runner is stateless;
    // it joins the live config view below (control-plane, no-restart).
    let web_search = cfg.web_search();

    // PDF handling: text extraction (sift) and page rendering (pdfium) are both
    // compiled in, so neither can be missing at runtime. This used to be a
    // three-way probe over a sidecar library, whose "not found" arm was the
    // whole of  - a degraded request was the first sign anything was
    // wrong. Now the only variable is whether the MODEL can read page images,
    // which the vision capability already answers.
    let pdf_cfg = crate::pdf::PdfConfig::from_config(&cfg);
    tracing::info!(
        max_pages = pdf_cfg.max_pages,
        long_edge = pdf_cfg.long_edge,
        "PDF input ready: page rendering (pdfium, linked in) for vision models, text extraction (sift) for the rest"
    );

    // Request filters (doc §13): validated here so a malformed variant/preset
    // is a startup error, never a preset that silently fails to apply.
    let filters = crate::filters::Filters::build(
        cfg.aliases.clone(),
        cfg.variants.clone(),
        cfg.strip_params.clone(),
        cfg.force_params.clone(),
    )
    .map_err(|e| format!("invalid filter config: {e}"))?;
    if !filters.aliases.is_empty() || filters.needs_body_transform() {
        tracing::info!(
            aliases = ?filters.aliases,
            variants = ?filters.variants.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(),
            strip = ?filters.strip,
            force = ?filters.force.keys().collect::<Vec<_>>(),
            "request filters active"
        );
    }

    // What an un-dialled request will be sampled at, and why. Logged rather
    // than left implicit because the answer is now model-dependent
    // - "why does this model wander" has to be answerable from the log
    // without reading the table.
    let sampling = crate::routes::SamplingDefaults::for_model(
        &cfg,
        serving.as_ref().map(|s| s.arch.as_str()),
        serving.as_ref().and_then(|s| s.published_sampling),
    );
    if serving.is_some() {
        let d = sampling.resolve(true);
        tracing::info!(
            temperature = d.temp,
            top_k = d.top_k,
            top_p = d.top_p,
            min_p = d.min_p,
            source = sampling.provenance(),
            cite = sampling.provenance_detail().unwrap_or("-"),
            "default sampling for requests that send none"
        );
    }

    let state = Arc::new(AppState {
        spec_off: spec_policy_off,
        auth_key,
        trusted_proxy: cfg.trusted_proxy,
        instance_id,
        metrics,
        snapshots,
        metrics_auth: cfg.metrics_auth,
        serving,
        sampling,
        embedder,
        asr,
        aligner,
        max_ctx: cfg.max_ctx,
        vad_gate: cfg.vad_gate,
        max_batch: cfg.max_batch,
        default_max_output_tokens: cfg.max_tokens.unwrap_or(1024),
        max_output_ceiling: cfg.max_output_ceiling,
        // live view: server-tool and web-search changes in the config file
        // apply on the next request (control-plane state), never via a
        // model restart
        live: crate::routes::LiveConfig::new(
            banner.config_path.clone().map(Into::into),
            crate::routes::LiveSnapshot {
                mcp_servers: cfg.mcp_servers.clone(),
                web_search,
            },
        ),
        rate_limiter: std::sync::Arc::new(crate::ratelimit::RateLimiter::new(
            crate::ratelimit::Limits {
                per_minute: cfg.ratelimit_per_minute,
                per_day: cfg.ratelimit_per_day,
            },
            cfg.trusted_proxy,
        )),
        mcp: std::sync::Arc::new(paddock_mcp::McpManager::new()),
        approvals: std::sync::Arc::new(crate::responses::ApprovalGate::default()),
        approval_store: std::sync::Arc::new(crate::responses::ApprovalStore::default()),
        stats,
        pdf: pdf_cfg,
        drain: Arc::new(crate::drain::DrainCtl::default()),
        // Event ring (doc §8.1): metadata level, RAM only, on by default;
        // --no-events is the off switch (§8.7's stated knob).
        events: if cfg.no_events {
            crate::events::EventRing::disabled()
        } else {
            crate::events::EventRing::new()
        },
        session_headers: cfg.session_headers.clone(),
        filters: Arc::new(filters),
        concurrency_limit: cfg.concurrency_limit,
        // Forensic preprocessing gate ([forensics]). None unless enabled, so
        // there is zero request-path cost when off.
        forensics: crate::forensics::ForensicRuntime::build(&cfg.forensics),
    });

    let addr = std::net::SocketAddr::new(cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::debug!(%addr, model_dirs = ?cfg.model_dirs, "paddock runner listening");

    // Admin surface (doc §5.1): local pipe/socket only, keyed by the inference
    // port we now provably own (TCP bound above - so an admin-name collision
    // can only be a corpse, never a live runner). Spawned, not awaited: a
    // failed admin surface degrades supervision, never serving.
    let admin_state = Arc::new(crate::admin::AdminState {
        app: state.clone(),
        port: cfg.port,
        started_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        started: std::time::Instant::now(),
    });
    let admin_port = cfg.port;
    tokio::spawn(async move {
        if let Err(e) =
            paddock_admin::server::serve(admin_port, crate::admin::router(admin_state)).await
        {
            tracing::warn!(%e, "admin surface unavailable (manager supervision degraded; serving continues)");
        }
    });
    // Bound successfully - print the banner with the address we actually hold.
    startup::print_startup_banner(&cfg, &banner);

    // TCP_NODELAY: without it, Nagle + delayed-ACK interaction adds a fixed
    // ~30-40 ms stall to multi-segment responses (embedding payloads are
    // megabytes) - measured as a flat per-request tax on the embed bench.
    use axum::serve::ListenerExt;
    let listener = listener.tap_io(|tcp| {
        let _ = tcp.set_nodelay(true);
    });
    // `with_connect_info` so the rate limiter can fall back to the socket peer
    // when it isn't behind a trusted proxy (see `ratelimit::client_key`).
    let serve = axum::serve(
        listener,
        routes::router(state).into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal());
    // Bound the connection drain: an SSE stream a client never closes must not
    // hold the process past shutdown (the second listener sees the same signal,
    // then allows 10 s of drain before proceeding to the engine free).
    tokio::select! {
        r = serve => r?,
        () = async {
            shutdown_signal().await;
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        } => {
            tracing::warn!("shutdown: connections still open after 10 s - proceeding");
        }
    }
    if let Some(engine) = engine_shutdown {
        tracing::info!("shutdown: draining engine and freeing device memory");
        let clean = tokio::task::spawn_blocking(move || {
            engine.shutdown(std::time::Duration::from_secs(30))
        })
        .await
        .unwrap_or(false);
        if clean {
            tracing::info!("shutdown: device memory freed - exiting");
        } else {
            tracing::warn!(
                "shutdown: engine did not confirm the free within 30 s - exiting anyway"
            );
        }
    }
    Ok(())
}

/// Resolves when the process receives SIGINT (Ctrl+C) or SIGTERM. Multiple
/// listeners each observe the same delivery (tokio signal streams are
/// broadcast), so the serve hook and the drain bound above can both wait.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term =
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(%e, "SIGTERM handler unavailable; Ctrl+C only");
                    let _ = tokio::signal::ctrl_c().await;
                    return;
                }
            };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
