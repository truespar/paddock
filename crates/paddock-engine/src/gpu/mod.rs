//! GPU execution building blocks: device weight store, per-op kernel calls,
//! GEMM. The gpu_model module assembles these into model graphs.
//!
//! cuBLAS is the vendor-baseline GEMM permitted by the kernel policy; our own
//! quantized GEMM kernels replace it where it can't go (fused dequant paths).
//!
//! Split into op-domain submodules: one `impl GpuExecutor` block per file.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use paddock_kernels::KernelPack;
use paddock_kernels::abi::KernelTableV1;

mod arch;
/// Public for `ts_flags` - the whisper timestamp grammar's per-row bits are
/// built host-side (see `gpu_model::whisper::ts_state`) and consumed by the
/// kernel, so both halves have to name the same constants.
pub mod asr;
mod attention;
mod basic_ops;
mod batch_ops;
mod bf16;
mod deltanet;
mod error;
mod fp4;
mod fp8;
mod fused_gemv;
mod graph;
mod host_plane;
pub use host_plane::{HostMappedKq, HostMirror};
mod moe_cache;
pub use moe_cache::{
    ExpertCache, MOE_CACHE_NONE, MoeOffloadCfg, moe_cache_slots_pin, moe_offload, set_moe_offload,
};
mod kquant;
mod types;
pub use kquant::q40_to_q8_blocks;
mod mamba;
mod moe_bs;
mod moe_mxfp4;
mod moe_q8;
mod q8_gemm;
mod qwen4exp;
mod sampling;
mod tier_xfer;
mod transfer;
mod upload;

pub use error::GpuError;
use error::drv;
pub(crate) use error::from_driver;
pub use graph::{CapturedGraph, end_capture_no_flags};
pub use types::*;

/// Owns the CUDA context, stream, cuBLAS handle, and kernel pack - the
/// execution substrate model graphs build on.
pub struct GpuExecutor {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    /// Side stream for event-gated device->host reads: collecting one
    /// batch's pooled output must not queue behind the next batch's kernels
    /// on the compute stream (the encoder pipelines submits).
    copy_stream: Arc<CudaStream>,
    /// Second COMPUTE stream for decode-tick sub-DAG overlap (gemma4 A4B:
    /// the shared dense-FFN branch runs beside the routed MoE branch; both
    /// read post-attention x, and meet at the two-branch tail). Same
    /// greatest priority as `stream` - the short branch it carries must
    /// dispatch promptly, not queue behind a full routed wave.
    side_stream: Arc<CudaStream>,
    /// When set, `stream_ptr()` hands out `side_stream` - flipped by the
    /// side_fork/side_end guard around the forked branch's launches.
    /// Atomic (not Cell) because the executor is shared across threads;
    /// in practice only the model walk's thread touches it.
    side_armed: std::sync::atomic::AtomicBool,
    /// A dense i-quant plane was loaded here. INTERIM seam for the missing
    /// i-quant mma tile lane: prefill over 64 rows then needs the row-major
    /// int8 activations as well as the mmq tiles (qwen35/ops.rs
    /// `prefill_quant`), and only the loader knows a plane's type. Goes
    /// with the tile lane when it lands.
    dense_iq_seen: std::sync::atomic::AtomicBool,
    /// A dense i-quant plane whose rows are not whole super-blocks (IQ4_NL's
    /// flat 32-block rows) was loaded: the tile rung cannot take it, so the
    /// over-64-row producers must keep the row-major int8 pair even on a pack
    /// that carries the i-quant tile marker.
    dense_iq_flat_seen: std::sync::atomic::AtomicBool,
    /// The forked branch's completion event, recorded by `side_end` and
    /// stream-waited by `side_join` before the joint consumer launches.
    /// Parked here so the event outlives graph capture.
    side_pending: std::sync::Mutex<Option<cudarc::driver::CudaEvent>>,
    // Keeps the library loaded (the table points into it) when the pack was
    // dlopen'd, and names the source in preflight's refusal either way.
    pack: std::sync::Arc<KernelPack>,
    kernels: KernelTableV1,
    /// Device SM count - launch-shape heuristics (attention split fill) scale
    /// with it instead of baking one card's geometry into a constant.
    sm_count: usize,
    /// Compute capability (major, minor) - kernel-class routing (e.g. the
    /// sm_120a block-scale MoE) keys on it.
    cc: (u32, u32),
    /// Unified-memory (integrated) die - DGX Spark GB10, Jetson: the GPU has
    /// no memory of its own, so "free VRAM" is an OS question, not a driver
    /// one. Read at construction from CU_DEVICE_ATTRIBUTE_INTEGRATED, never
    /// inferred from the compute capability (`device_mem_info` explains).
    integrated: bool,
    /// Hard VRAM budget in bytes (0 = none). Set once by the runner from its
    /// config file (`vram_budget`, the manager writes it at admission) before
    /// the model loads. Every free-VRAM sizer goes through `vram_headroom`,
    /// which clamps device-free to `budget - ledger` - so two runners each
    /// keep to their granted slice instead of both sizing against the same
    /// shared "free" and summing past the card, which froze the machine.
    vram_budget: std::sync::atomic::AtomicU64,
    /// One reusable host->device staging buffer for the whole load, grown to
    /// the largest tensor seen and released by `release_staging` when the load
    /// stamps its weight line.
    ///
    /// It used to be a fresh `clone_htod` per tensor, and that is a
    /// fragmentation engine: every weight plane was allocated while a
    /// same-sized staging buffer sat beside it, and the staging was then freed
    /// - so each hole ended up between two live planes, and `cuMemPoolTrimTo`
    ///   can only return a block when nothing in it is live. Thousands of
    ///   tensors, thousands of holes. Measured retained-not-live across families
    ///   before this: gemma-4-31B 4.85 GB (14.6% of live), qwen3.8-27B 2.12 GB
    ///   (7.6%), against ~0.5-0.8 GB on loads that happen to do less repacking.
    ///   One buffer reused means zero interleaved frees during load.
    ///
    /// Mutex rather than RefCell because `GpuExecutor` is shared across
    /// threads; re-entrancy is not a concern because no `with_staged_raw`
    /// closure calls back into it (all ten call sites are sequential).
    staging: std::sync::Mutex<Option<cudarc::driver::CudaSlice<u8>>>,
    /// Grow-only scratch for the e4m3 intermediate inside `q8_0_to_f8w_lin`.
    ///
    /// The Q8 -> e4m3 -> tile-linear chain used to allocate the e4m3 plane,
    /// allocate the linear output above it, then drop the e4m3 - leaving a hole
    /// under a live plane that `cuMemPoolTrimTo` can never return. Per tensor,
    /// every layer. Measured on gemma-4-31B: the f8a phase strands 4.06 GB, of
    /// which the repack is ~1.6 GB and the fused-qkv build ~1.2 GB
    /// (`PADDOCK_NO_F8LIN=1` -> 3.37 GB, `PADDOCK_G4_NO_QKVFUSE=1` -> 3.74,
    /// both -> 1.16, against a 0.89 floor with the whole phase off).
    ///
    /// Because the intermediate never escapes `q8_0_to_f8w_lin` - the repack
    /// reads it by in_dim/out_dim, not by length - this one can be oversized
    /// and reused for every tensor, unlike a recycled `RepackedMxfp4` handed
    /// back to a caller that reads `data.len()`. One buffer, grown once,
    /// released with the staging at the end of load. An earlier attempt kept a
    /// pool of eight exact-sized planes instead and REGRESSED qwen35 (1.15 ->
    /// 2.52 GB), because releasing eight large planes at the end is the same
    /// batch-free antipattern one level down.
    conv_scratch:
        std::sync::Mutex<Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<u8>)>>,
}
impl GpuExecutor {
    /// See the `dense_iq_seen` field.
    pub fn dense_iq_seen(&self) -> bool {
        self.dense_iq_seen
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(crate) fn note_dense_iq(&self) {
        self.dense_iq_seen
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
    /// See the `dense_iq_flat_seen` field.
    pub fn dense_iq_flat_seen(&self) -> bool {
        self.dense_iq_flat_seen
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    pub(crate) fn note_dense_iq_flat(&self) {
        self.dense_iq_flat_seen
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Load a pack from a path. The bring-up/bench/example entry point, and
    /// what a dev build has always used.
    pub fn new(ordinal: usize, pack_path: &std::path::Path) -> Result<Self, GpuError> {
        Self::with_pack(ordinal, Some(pack_path))
    }

    /// `None` means the kernels compiled into this binary - a release build
    /// (`--features static-pack`), where the install is two binaries and there
    /// is no pack file to point at. Without that feature `None` is a
    /// configuration error, said in full rather than as a missing-file
    /// mystery.
    pub fn with_pack(
        ordinal: usize,
        pack_path: Option<&std::path::Path>,
    ) -> Result<Self, GpuError> {
        // A preload used to run here, mapping paddock's own CUDA libraries by
        // absolute path so cudarc's by-name request for `cublas64_13.dll` could
        // not be answered from PATH by whatever Toolkit the box happened to
        // have. There is no such request any more: cudarc is built with the
        // `driver` feature alone, and the binaries do not contain the string
        // "cublas64" at all - they could not ask. The only library the engine
        // opens is the CUDA DRIVER, which belongs to the display driver and has
        // exactly one copy on any machine.
        //
        // And if that one copy is not there, say so: cudarc panics on its
        // first call when the library cannot be opened, and a panic here is
        // the runner dying at startup with a dlopen trace instead of a line
        // naming the driver.
        if !crate::cuda::driver_present() {
            return Err(GpuError::Driver(crate::cuda::NO_DRIVER.into()));
        }
        let ctx = CudaContext::new(ordinal).map_err(drv)?;
        // A real (non-blocking) stream, not ctx.default_stream(): the null/legacy
        // default stream cannot be captured - cuStreamBeginCapture on it returns
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED - and CUDA-graph capture of the
        // per-token decode is the launch-overhead lever. Everything runs on this one
        // stream, so cudarc's cross-stream event tracking is pure overhead; disable it
        // now (before any buffer is allocated) so device_ptr guards are no-ops - which
        // also makes capture legal (its cuStreamWaitEvent calls are what forbid it).
        // glibc's dynamic mmap threshold caps at 32 MiB: any Vec bigger than
        // that (the batched-logits readback is b x vocab f32 = 33.8 MB at
        // B=42 on gpt-oss) is mmap'd fresh and munmap'd on drop every step,
        // and the ~8k soft page faults during the dtoh memcpy ran it at
        // 1.8 GB/s (+17 ms/step, a sharp cliff at B >= 42). Raising the
        // threshold keeps those steady-state buffers on the heap, where they
        // recycle warm pages (~19 GB/s). Pinned staging is not the answer on
        // this class of host: direct DMA measured 1.5 GB/s (virtualized
        // PCIe), and cudarc's alloc_pinned is write-combined (uncached host
        // reads) anyway.
        // glibc-only tuning: Windows has no mallopt (and no such cliff - its
        // allocator recycles large blocks without the mmap round trip).
        // SAFETY: process-wide allocator tuning, no aliasing or lifetime impact
        #[cfg(target_os = "linux")]
        unsafe {
            libc::mallopt(libc::M_MMAP_THRESHOLD, 128 << 20);
            // ...and without this, the freed heap top gets trimmed (madvised
            // away) on every drop and the pages fault right back in
            libc::mallopt(libc::M_TRIM_THRESHOLD, 256 << 20);
        }
        // The MAIN compute stream takes the GREATEST stream priority: with a
        // single stream this changes nothing (priority orders CTA dispatch
        // between streams), and when the overlap decode lane is forked (its
        // streams keep the default = least priority), span/prefill kernels
        // win every free-SM dispatch slot - the decode graph fills tail
        // waves and launch-glue gaps instead of time-slicing whole kernels
        // (route-B v1 measured the time-slice: spans stretched ~2x, c32
        // -4%). Falls back to default priority if the range query fails.
        ctx.bind_to_thread().map_err(drv)?;
        let stream = match cudarc::driver::result::stream::get_priority_range() {
            Ok((_least, greatest)) => ctx.new_stream_with_priority(greatest).map_err(drv)?,
            Err(_) => ctx.new_stream().map_err(drv)?,
        };
        let copy_stream = ctx.new_stream().map_err(drv)?;
        let side_stream = match cudarc::driver::result::stream::get_priority_range() {
            Ok((_least, greatest)) => ctx.new_stream_with_priority(greatest).map_err(drv)?,
            Err(_) => ctx.new_stream().map_err(drv)?,
        };
        // SAFETY: single compute stream - there is no concurrent cross-stream buffer use
        // for the event bookkeeping to guard, so dropping it is correctness-preserving.
        unsafe { ctx.disable_event_tracking() };
        let pack = std::sync::Arc::new(match pack_path {
            Some(p) => KernelPack::load(p)?,
            #[cfg(feature = "static-pack")]
            None => KernelPack::builtin()?,
            #[cfg(not(feature = "static-pack"))]
            None => {
                return Err(GpuError::Unsupported(
                    "no kernel pack configured, and this build has none compiled in. \
                     Set `kernel_pack` in the runner's config (or --kernel-pack) to a pack \
                     built by packs/cuda/build.ps1 / build.sh. Release builds carry their \
                     kernels internally and need neither."
                        .to_owned(),
                ));
            }
        });
        let kernels = pack.kernels_v1()?;
        // A pack older than this build fills fewer table entries and the rest
        // read as None. Safe by design, and invisible until a request happens
        // to need one - which is how a months-stale pack in a portable install
        // first announced itself as a kernel name in front of a user,
        // mid-transcript. Said once, here, where whoever runs the
        // box can act on it. Applies to a built-in pack too: the archive is a
        // separate nvcc run, so linked-in is not the same as same-vintage.
        if let Ok(fit) = pack.table_fit()
            && fit.is_stale()
        {
            tracing::warn!(
                pack = %pack.origin().display(),
                declared = fit.declared,
                expected = fit.expected,
                "kernel pack is OLDER than this engine build: {} of {} table entries are absent, \
                 and anything that needs one will refuse at the point of use. Rebuild the pack \
                 (packs/cuda/build.ps1 on Windows, build.sh elsewhere) and restart this runner.",
                fit.missing_entries(),
                fit.expected / std::mem::size_of::<usize>(),
            );
        }
        let sm_count = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .map_err(drv)? as usize;
        use cudarc::driver::sys::CUdevice_attribute as Attr;
        let cc = (
            ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .map_err(drv)? as u32,
            ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .map_err(drv)? as u32,
        );
        let integrated = ctx
            .attribute(Attr::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .map_err(drv)?
            != 0;
        let exec = Self {
            ctx,
            stream,
            copy_stream,
            side_stream,
            side_armed: std::sync::atomic::AtomicBool::new(false),
            dense_iq_seen: std::sync::atomic::AtomicBool::new(false),
            dense_iq_flat_seen: std::sync::atomic::AtomicBool::new(false),
            side_pending: std::sync::Mutex::new(None),
            pack,
            kernels,
            sm_count,
            cc,
            integrated,
            vram_budget: std::sync::atomic::AtomicU64::new(0),
            staging: std::sync::Mutex::new(None),
            conv_scratch: std::sync::Mutex::new(None),
        };
        exec.preflight()?;
        Ok(exec)
    }

    /// Reject unsupported GPU/pack combinations at load with a complete
    /// sentence instead of a raw CUDA error at the first model launch.
    /// Two failure classes:
    /// a pre-Ampere card (below the pack's sm_80 floor), and a pack whose
    /// fatbin lacks this device's arch (surfaces as CUDA 209
    /// no-kernel-image / 222 unsupported-PTX on a trial launch).
    fn preflight(&self) -> Result<(), GpuError> {
        let device = self.ctx.name().unwrap_or_else(|_| "unknown GPU".into());
        let (maj, min) = self.cc;
        if maj < 8 {
            return Err(GpuError::Unsupported(format!(
                "{device} is sm_{maj}{min} - paddock's CUDA packs need Ampere or newer \
                 (sm_80+): the Q8_0 serving path is built on the int8 dp4a/mma ladder \
                 that first ships there. Pre-Ampere cards are not supported."
            )));
        }
        // Validated-arch policy: exact-(major,minor) match against the closed
        // bring-up campaigns serves silently; any other Ampere-or-newer die
        // serves under the UNVALIDATED warning (see gpu/arch.rs). Runs before
        // the trial launch so the stamp precedes any no-image refusal - and
        // so a GB10/Spark, where plain sm_120 SASS forward-loads and the
        // probe below passes, is named before an sm_120a-only family fails.
        match arch::gate(self.cc, &device) {
            arch::Gate::Validated => {}
            arch::Gate::Unvalidated(warn) => tracing::warn!("{warn}"),
        }
        // trial launch of the always-present elementwise add: proves this
        // pack's fatbin carries an image for this device before any model load
        let mut a = self.alloc(4)?;
        let b = self.alloc(4)?;
        let launched = self.add(&mut a, &b, 4).and_then(|()| self.synchronize());
        if let Err(e) = launched {
            // A built-in pack cannot be "rebuilt with this arch" by whoever is
            // running it - the arch list was fixed when the binary was built,
            // so say the thing they can do (point at a pack that has it).
            let fix = if self.pack.is_builtin() {
                format!(
                    "This build's kernels cover only the validated architectures. Build a pack \
                     for it (packs/cuda/build.sh {maj}{min}) and pass --kernel-pack."
                )
            } else {
                format!("Rebuild the pack with this arch included: packs/cuda/build.sh {maj}{min}")
            };
            return Err(GpuError::Unsupported(format!(
                "kernel pack {} has no working image for {device} (sm_{maj}{min}): {e}. {fix}",
                self.pack.origin().display()
            )));
        }
        Ok(())
    }

    /// Hard VRAM admission at model load: refuse when the weights alone plus
    /// a modest working floor exceed free VRAM. Without this, Windows WDDM
    /// lets the allocations "succeed" by paging into system RAM - and the
    /// whole MACHINE freezes (48 GB card, 63 GB committed). The
    /// manager runs the same arithmetic earlier with fleet knowledge; this is
    /// the engine's own last line (direct runner invocations, adopted
    /// setups). PADDOCK_ALLOW_VRAM_OVERCOMMIT bypasses, loudly. Must run on
    /// the engine thread (context current).
    pub fn vram_load_gate(&self, weights_bytes: u64, model: &str) -> Result<(), String> {
        if std::env::var_os("PADDOCK_ALLOW_VRAM_OVERCOMMIT").is_some() {
            tracing::warn!(
                model,
                "VRAM load gate BYPASSED (PADDOCK_ALLOW_VRAM_OVERCOMMIT)"
            );
            return Ok(());
        }
        // KV/scratch land after load and pools size against what remains;
        // the floor stays modest so the gate is honest, never clever
        const FLOOR: u64 = 1 << 30;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        // the configured budget binds first: it was granted with the whole
        // fleet in view, so exceeding it is a config problem even when the
        // card happens to have free bytes right now
        if let Some(budget) = self.vram_budget()
            && weights_bytes + FLOOR > budget
        {
            return Err(format!(
                "{model} will not fit its configured VRAM budget: weights need {:.1} GiB but vram_budget grants {:.1} GiB. Raise vram_budget in the server's config file (or remove the line), stop another model to free its budget, or pick a smaller quant.",
                gib(weights_bytes),
                gib(budget),
            ));
        }
        let Some((free, total)) = self.device_mem_info() else {
            return Ok(()); // no honest number - the allocations themselves decide
        };
        if weights_bytes + FLOOR > free {
            if self.integrated {
                // MemAvailable is a FLOOR on a unified-memory die, not the
                // truth. The NVIDIA driver (>= 590, system-memory pools)
                // parks the pages an exiting CUDA process frees in its own
                // shrinker-backed pools: /proc/meminfo books them as used,
                // MemAvailable does not count them - and the driver serves
                // the next allocation from them without touching the
                // kernel's free pages. Measured 2026-09-07 on GB10 right
                // after a 27B runner exited: MemAvailable 25 GiB, a 40 GiB
                // cudaMalloc + memset succeeded in 0.27 s, MemAvailable
                // moved by 0.5 GiB. The gate refused every load for the
                // next several minutes on that reading. So before refusing,
                // ask the one party that knows: a trial allocation of the
                // weights' size. Success is the proof (the pages go straight
                // back and the load takes them for real); failure keeps the
                // refusal, which is the honest answer on a box that is
                // truly full.
                let need = weights_bytes + FLOOR;
                match unsafe { cudarc::driver::result::malloc_sync(need as usize) } {
                    Ok(ptr) => {
                        // SAFETY: ptr came from malloc_sync just above and is
                        // freed exactly once, on the same context.
                        if let Err(e) = unsafe { cudarc::driver::result::free_sync(ptr) } {
                            tracing::warn!(error = %e, "trial allocation: free failed");
                        }
                        tracing::warn!(
                            model,
                            need_gib = gib(need),
                            available_gib = gib(free),
                            "unified-memory die: MemAvailable would refuse this load, but a \
                             trial allocation of the weights' size succeeded - the driver's \
                             retained pools cover it. Loading."
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        return Err(format!(
                            "{model} will not fit: weights need {:.1} GiB but only {:.1} GiB of this box's {:.1} GiB unified memory is available (MemAvailable), and a trial allocation of {:.1} GiB was refused by the driver ({e}). Another process likely holds the rest - stop it first, or pick a smaller quant. Refusing to load: on unified memory an oversubscribed load thrashes the whole machine rather than failing cleanly.",
                            gib(weights_bytes),
                            gib(free),
                            gib(total),
                            gib(need),
                        ));
                    }
                }
            }
            return Err(format!(
                "{model} will not fit: weights need {:.1} GiB but only {:.1} GiB of {:.1} GiB VRAM is free. Another model likely holds the rest - stop it first, or pick a smaller quant. Refusing to load: oversubscribed VRAM pages into system RAM and can freeze the machine.",
                gib(weights_bytes),
                gib(free),
                gib(total),
            ));
        }
        Ok(())
    }

    /// `(free, total)` device bytes as every sizer must read them.
    ///
    /// Discrete cards: cuMemGetInfo, unchanged. Integrated (unified-memory)
    /// dies - DGX Spark GB10, Jetson - answer differently: the driver reports
    /// total = the box's RAM and free = the kernel's MemFree, which counts the
    /// page cache as USED and ignores the pages the driver itself parks in
    /// its freed-allocation pools (open kernel module >= 590,
    /// NVreg_EnableSystemMemoryPools). Measured 2026-09-07 on a Spark: free
    /// read 0.9 GiB while 112 GiB of cudaMalloc succeeded, and the load gate
    /// refused a 9 GB model on a 121 GB box. Every engine that serves on this
    /// class of box reads the OS instead - NVIDIA's own Spark guidance (the
    /// getAvailableMemory snippet in the DGX Spark known-issues page),
    /// llama.cpp (ggml PR #17368), vLLM (PR #35356), SGLang (PR #11287) -
    /// so that is the reading here: free = MemAvailable (a hugetlb-configured
    /// box: HugePages_Free x Hugepagesize, per the same snippet), total =
    /// the smaller of the driver's total and MemTotal. SwapFree is
    /// deliberately not added: past the limit this class of box thrashes and
    /// wedges rather than failing an allocation.
    ///
    /// Still pessimistic right after a large CUDA process exits: the pooled
    /// pages sit in no /proc/meminfo category, so MemAvailable does not see
    /// them until the kernel's shrinker runs (memory pressure, or
    /// drop_caches=2). The gate's refusal says so; nothing here tries to be
    /// clever about it.
    pub fn device_mem_info(&self) -> Option<(u64, u64)> {
        let (free, total) = cudarc::driver::result::mem_get_info().ok()?;
        let (free, total) = (free as u64, total as u64);
        if !self.integrated {
            return Some((free, total));
        }
        static SAID: std::sync::Once = std::sync::Once::new();
        match unified_available_memory() {
            Some((avail, mem_total)) => {
                let total = total.min(mem_total);
                SAID.call_once(|| {
                    tracing::info!(
                        box_gib = total as f64 / (1u64 << 30) as f64,
                        available_gib = avail as f64 / (1u64 << 30) as f64,
                        driver_free_gib = free as f64 / (1u64 << 30) as f64,
                        "unified-memory die: free VRAM is read as the OS's MemAvailable \
                         (the driver's own free figure counts the page cache as used)"
                    );
                });
                Some((avail, total))
            }
            None => {
                SAID.call_once(|| {
                    tracing::warn!(
                        "unified-memory die but /proc/meminfo could not be read - \
                         falling back to the driver's free figure, which undercounts"
                    );
                });
                Some((free, total))
            }
        }
    }

    /// Share of the CARD one runner may occupy when its config sets no
    /// explicit `vram_budget`. 0.90 is vLLM's `--gpu-memory-utilization`
    /// default and the same bargain: the serving process gets the card's bulk,
    /// the rest stays for the display server, allocator slack, CUDA context
    /// and a co-resident runner.
    ///
    /// It is a CEILING on this process, not a reservation of the remainder -
    /// families still subtract their own `VRAM_HEADROOM` slack under it, and
    /// the addressable clamp (`slots * bps`) still applies above it. On a
    /// 49.1 GB A6000 that puts a gemma4-family runner (6 GiB own headroom) at
    /// ~38 GB, which is where muse already sat; granite (1 GiB) drops from the
    /// 48.0 GB to ~43 GB. Override per server with `vram_budget`
    /// in the config file - the engine reads no env for this, by design.
    const DEFAULT_VRAM_UTILIZATION: f64 = 0.90;

    /// Configure the hard VRAM budget (bytes) this executor's model must live
    /// inside. Called once by the runner before load, from the config file's
    /// `vram_budget` - the engine itself never reads env or files for it.
    pub fn set_vram_budget(&self, bytes: u64) {
        self.vram_budget
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// The configured VRAM budget, if any.
    pub fn vram_budget(&self) -> Option<u64> {
        match self.vram_budget.load(std::sync::atomic::Ordering::Relaxed) {
            0 => None,
            b => Some(b),
        }
    }

    /// Bytes a sizer may still take: what remains of this runner's budget
    /// (`budget - pool ledger`), clamped to device-free. The seam every
    /// free-VRAM-based sizer (KV pool, checkpoint pool, width clamp, opt-in
    /// fusion planes) must read instead of raw cuMemGetInfo - raw free is the
    /// whole card's, and two co-resident runners each sizing against it is
    /// exactly how a card oversubscribes after both passed admission.
    /// `None` = the driver gave no number (callers keep their old
    /// no-measurement behavior); `Some(0)` = a real, exhausted reading.
    /// Callers that need pool-held frees excluded should trim_mem_pool first,
    /// same as before.
    ///
    /// With no configured budget this used to hand back the whole card's free
    /// bytes, which is: the paged pool then took everything going,
    /// `max_ctx` became a per-slot RESERVATION instead of a per-sequence
    /// ceiling, and granite-4.1-30b drove this A6000 to 48.0 of 49.1 GB with a
    /// desktop session on it. The default is now a utilization fraction of the
    /// CARD, exactly as the explicit-budget branch already worked - the two
    /// are the same expression now, one configured and one derived.
    ///
    /// Deriving from TOTAL rather than free is also what makes pool sizing
    /// REPRODUCIBLE, and that is not a nicety: the same config measured ~17%
    /// apart across server loads purely because the desktop's VRAM use
    /// drifted 1.9-4.9 GB between starts and each load therefore claimed a
    /// different pool. Sizing off a fixed fraction of the card removes that
    /// input; `free` stays only as a safety clamp so we still never
    /// over-commit a card someone else is using.
    pub fn vram_headroom(&self) -> Option<u64> {
        let (free, total) = self.device_mem_info()?;
        let (budget, explicit) = match self.vram_budget() {
            Some(b) => (b, true),
            None => (
                (total as f64 * Self::DEFAULT_VRAM_UTILIZATION) as u64,
                false,
            ),
        };
        let allowance = budget.saturating_sub(self.process_mem_used().unwrap_or(0));
        // Say it once when the derived default is what binds - a pool quietly
        // smaller than the card is exactly the kind of thing the no-silent-
        // failures rule exists for, and the number is otherwise invisible.
        if !explicit && allowance < free {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::info!(
                    card_gib = total as f64 / (1u64 << 30) as f64,
                    utilization = Self::DEFAULT_VRAM_UTILIZATION,
                    budget_gib = budget as f64 / (1u64 << 30) as f64,
                    ours_gib = self.process_mem_used().unwrap_or(0) as f64 / (1u64 << 30) as f64,
                    headroom_gib = allowance as f64 / (1u64 << 30) as f64,
                    "VRAM sizing bounded by the default budget, not by free VRAM \
                     (set vram_budget in the server config to override)"
                );
            });
        }
        Some(free.min(allowance))
    }

    /// Device bytes this PROCESS holds live in its stream-ordered mempool -
    /// weights + KV/state pools + resident scratch, measured rather than
    /// estimated. Per-process by construction (the pool is ours alone). The
    /// previous reading - free-at-init minus free-now, device-GLOBAL via
    /// cuMemGetInfo - attributed co-resident processes' allocations to this
    /// model the moment a second runner shared the card: two runners' ledgers
    /// summed past physical VRAM (found on the manager's fleet table).
    /// CUDA context/modules/graphs and cuBLAS workspaces are not
    /// pool allocations and stay excluded, as before. Must be called on a
    /// thread where this context is current (the engine thread).
    pub fn process_mem_used(&self) -> Option<u64> {
        let mut used: u64 = 0;
        // SAFETY: the pool handle belongs to this device; the out-param is the
        // attribute's documented cuuint64_t.
        unsafe {
            let dev = cudarc::driver::result::device::get(self.ctx.ordinal() as i32).ok()?;
            let pool = cudarc::driver::result::device::get_mem_pool(dev).ok()?;
            cudarc::driver::result::mem_pool::get_attribute(
                pool,
                cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                std::ptr::from_mut(&mut used).cast(),
            )
            .ok()?;
        }
        (used > 0).then_some(used)
    }

    /// Device bytes this process's mempool holds from the DRIVER - live
    /// allocations plus freed-but-retained blocks. The companion to
    /// `process_mem_used`: reserved-minus-used is the pool's internal
    /// fragmentation, i.e. bytes a `trim_to(0)` could not hand back because
    /// each retained block still has a live allocation somewhere in it.
    ///
    /// Worth having as its own reading because the free-VRAM delta a loader
    /// ledgers conflates three different things - live planes, pool
    /// fragmentation, and the CUDA context/modules/cuBLAS workspaces that are
    /// not pool allocations at all. Only `used`, `reserved` and the driver's
    /// free-VRAM view together separate them.
    /// Hand the load-time staging buffer back. Safe at any point - the next
    /// `with_staged_raw` just allocates a fresh one - but the right moment is
    /// when uploading is done, before any pool-sized allocation sizes itself
    /// against free VRAM.
    pub fn release_staging(&self) {
        if let Ok(mut slot) = self.staging.lock() {
            *slot = None;
        }
        if let Ok(mut slot) = self.conv_scratch.lock() {
            *slot = None;
        }
    }

    pub fn pool_reserved_bytes(&self) -> Option<u64> {
        let mut reserved: u64 = 0;
        // SAFETY: same contract as `process_mem_used` - our device's pool, and
        // the out-param is the attribute's documented cuuint64_t.
        unsafe {
            let dev = cudarc::driver::result::device::get(self.ctx.ordinal() as i32).ok()?;
            let pool = cudarc::driver::result::device::get_mem_pool(dev).ok()?;
            cudarc::driver::result::mem_pool::get_attribute(
                pool,
                cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                std::ptr::from_mut(&mut reserved).cast(),
            )
            .ok()?;
        }
        (reserved > 0).then_some(reserved)
    }

    /// `process_mem_used`, sampled at a point where the number is TRUSTWORTHY.
    /// This is the one call a loader should use to stamp its resident-weight
    /// line; `process_mem_used` alone is a live telemetry read.
    ///
    /// The synchronize is load-bearing, not hygiene. `cuMemFreeAsync` is
    /// stream-ORDERED: bytes a repack/reclaim freed are still counted as used
    /// until the stream actually reaches the free. Sampling without it
    /// overstated gemma-4-31b's weights by ~18 GiB (the whole f32 staging
    /// plane) - the load simply hadn't got there yet. The trim then returns
    /// the reclaimed pages to the OS so the driver's free-VRAM number agrees
    /// with ours; without it the two views of the same card disagree by
    /// whatever the pool is sitting on.
    ///
    /// Costs one stream sync plus a `cuMemPoolTrimTo` - fine once at load,
    /// wrong in a serving loop (the next alloc re-acquires from the OS).
    pub fn settled_mem_used(&self) -> Option<u64> {
        // The load is done uploading by the time it stamps its weight line, so
        // the staging buffer goes back before anything pool-sized (KV, prefill
        // scratch) sizes itself against free VRAM. A loader that keeps going
        // after this simply grows a fresh one.
        self.release_staging();
        self.trim_mem_pool(); // syncs, then releases reclaimed pages
        let used = self.process_mem_used();
        // Family-agnostic residency split, emitted here because every loader
        // calls this exactly once to stamp its resident-weight line - so every
        // family gets the instrument without a per-family edit.
        //
        // `live` is what the pool has handed out; `retained` is what it holds
        // from the driver but nothing is using - memory a trim could not give
        // back, because each retained block still has a live allocation in it.
        // A load that only allocates sits near 2-3% retained; a load that
        // allocates a source plane and frees it after its replacement exists
        // strands far more, because every hole ends up between two live
        // planes. Measured on qwen3.8-27B: 21% before the never-upload REPLACE,
        // 7.6% after, against 2.5% on a no-reclaim control
        //
        // Worth a line rather than a probe because the number is invisible in
        // every other view: the driver counts retained bytes as used, so a
        // free-VRAM ledger blames them on the weights, and ~26 GB of duplicate
        // planes once hid under exactly that arithmetic.
        if let (Some(u), Some(r)) = (used, self.pool_reserved_bytes()) {
            let gb = |b: u64| b as f64 / 1e9;
            tracing::info!(
                "VRAM residency split: pool live {:.2} GB · retained-not-live {:.2} GB ({:.1}% \
                 of live) · pool reserved {:.2} GB",
                gb(u),
                gb(r.saturating_sub(u)),
                if u > 0 {
                    r.saturating_sub(u) as f64 / u as f64 * 100.0
                } else {
                    0.0
                },
                gb(r),
            );
        }
        used
    }

    /// The sync half of `settled_mem_used` without the trim. Correct for a
    /// DELTA across a scope that swaps one set of planes for another: the sync
    /// is what makes the counter true, and the trim only matters when the
    /// driver's free-VRAM number has to agree with ours. Skipping it keeps a
    /// repeated caller (the encoder's calibration ladder re-quantizes ~700
    /// planes per rung) off the release-and-re-acquire-from-the-OS path.
    pub fn synced_mem_used(&self) -> Option<u64> {
        let _ = self.synchronize();
        self.process_mem_used()
    }

    /// Compute capability (major, minor), e.g. (12, 0) on Blackwell consumer.
    pub fn compute_capability(&self) -> (u32, u32) {
        self.cc
    }

    /// Whether the kernels came from this binary rather than a pack file.
    /// Worth being able to ask: a runner started with `--kernel-pack` on a
    /// build that also has kernels inside is using the file, and a support
    /// question about "which kernels are you running" has one true answer.
    pub fn pack_is_builtin(&self) -> bool {
        self.pack.is_builtin()
    }

    /// The loaded pack's semantic version, for feature-gating engine-elected
    /// defaults on pack GENERATIONS: an older .so can carry the same exports
    /// with older kernel bodies, which a symbol probe cannot distinguish.
    pub fn pack_version(&self) -> [u32; 3] {
        self.pack.info().pack_version
    }

    /// Device SM count (e.g. 84 on an A6000, 188 on an RTX PRO 6000 Blackwell).
    pub fn sm_count(&self) -> usize {
        self.sm_count
    }

    /// Context-wide drain (cuCtxSynchronize): every stream in the context,
    /// including forked lanes and the copy stream. Debug-probe surface.
    pub fn device_sync(&self) -> Result<(), GpuError> {
        self.ctx.synchronize().map_err(drv)
    }

    /// Fork a second execution lane on the same context and device memory:
    /// its own compute + copy streams and cuBLAS handle, sharing the loaded
    /// pack and kernel table. Work on the two lanes overlaps on the GPU -
    /// the dual-lane encoder uses this so two wave-starved small batches
    /// backfill each other's idle SMs. The parent stream is drained first so
    /// everything already uploaded (weights, quantized planes) is visible to
    /// the new lane; later cross-lane mutations must synchronize explicitly
    /// A fresh independent stream on this context - the whisper admission
    /// graph replays beside the decode tick from one (P38).
    pub(crate) fn new_side_stream(
        &self,
    ) -> Result<std::sync::Arc<cudarc::driver::CudaStream>, GpuError> {
        self.ctx.new_stream().map_err(drv)
    }

    /// Fork: position `side_stream` behind everything enqueued on the
    /// compute stream so far, then route subsequent launches to it. Under
    /// graph capture the record/wait pair becomes plain DAG edges (the
    /// decode tick records the fork into its graph).
    pub fn side_fork(&self) -> Result<(), GpuError> {
        let ev = self.stream.record_event(None).map_err(drv)?;
        self.side_stream.wait(&ev).map_err(drv)?;
        self.side_armed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// End the forked section: launches return to the compute stream; the
    /// side branch's completion event parks for `side_join`.
    pub fn side_end(&self) -> Result<(), GpuError> {
        self.side_armed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let ev = self.side_stream.record_event(None).map_err(drv)?;
        *self
            .side_pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(ev);
        Ok(())
    }

    /// Join: the compute stream waits device-side on the parked side event.
    /// No-op when nothing is pending - call unconditionally before any
    /// consumer of both branches.
    pub fn side_join(&self) -> Result<(), GpuError> {
        if let Some(ev) = self
            .side_pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            self.stream.wait(&ev).map_err(drv)?;
        }
        Ok(())
    }

    /// (event tracking is disabled by design).
    pub fn fork_stream(&self) -> Result<Self, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        // default (least) priority - the parent's greatest-priority compute
        // stream must win dispatch over forked lanes (see `new`)
        let stream = self.ctx.new_stream().map_err(drv)?;
        let copy_stream = self.ctx.new_stream().map_err(drv)?;
        let side_stream = self.ctx.new_stream().map_err(drv)?;
        Ok(Self {
            ctx: self.ctx.clone(),
            stream,
            copy_stream,
            side_stream,
            side_armed: std::sync::atomic::AtomicBool::new(false),
            dense_iq_seen: std::sync::atomic::AtomicBool::new(false),
            dense_iq_flat_seen: std::sync::atomic::AtomicBool::new(false),
            side_pending: std::sync::Mutex::new(None),
            pack: self.pack.clone(),
            kernels: self.kernels,
            sm_count: self.sm_count,
            cc: self.cc,
            integrated: self.integrated,
            // same process, same mempool ledger - the lanes share one budget
            vram_budget: std::sync::atomic::AtomicU64::new(
                self.vram_budget.load(std::sync::atomic::Ordering::Relaxed),
            ),
            // the forked lane does its own staging; nothing is shared
            staging: std::sync::Mutex::new(None),
            conv_scratch: std::sync::Mutex::new(None),
        })
    }

    /// The pack's kernel table (missing entries are None, never null jumps).
    pub fn kernels(&self) -> Result<KernelTableV1, GpuError> {
        Ok(self.kernels)
    }

    fn stream_ptr(&self) -> *mut core::ffi::c_void {
        if self.side_armed.load(std::sync::atomic::Ordering::Relaxed) {
            return self.side_stream.cu_stream() as *mut core::ffi::c_void;
        }
        self.stream.cu_stream() as *mut core::ffi::c_void
    }

    /// DeltaNet recurrent-state element size in BYTES: 4 (f32, exact) or 2
    /// under PADDOCK_DN_STATE_BF16 / PADDOCK_DN_STATE_F16 (the compounding
    /// numeric trade - kernels still compute f32; only the stored
    /// state/snapshot bytes halve; F16 keeps 10 mantissa bits vs bf16's 7).
    /// Element INDICES are dtype-agnostic; every state byte-offset must go
    /// through this. The pack mirrors the same envs (pd_dns_state_class).
    pub fn dn_state_esz() -> u64 {
        static ESZ: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        *ESZ.get_or_init(|| {
            // F16 is value-aware ("0" pins f32) - it is a model DEFAULT set
            // by qwen35's default_envs; BF16 stays a presence-only probe.
            // F8 (e4m3, 1 byte) is a value-aware probe and wins over both -
            // precedence mirrors the pack's pd_dns_state_class.
            let f8 = paddock_models::dev_var!("PADDOCK_DN_STATE_F8")
                .map(|v| v != "0")
                .unwrap_or(false);
            if f8 {
                return 1;
            }
            let f16 = std::env::var("PADDOCK_DN_STATE_F16")
                .map(|v| v != "0")
                .unwrap_or(false);
            if f16 || paddock_models::dev_var_os!("PADDOCK_DN_STATE_BF16").is_some() {
                2
            } else {
                4
            }
        })
    }
}

/// `(MemAvailable, MemTotal)` in bytes for an integrated die, from
/// /proc/meminfo - the reading NVIDIA publishes for the DGX Spark and the one
/// llama.cpp, vLLM and SGLang all take. A box with hugetlb pages configured
/// reports `HugePages_Free x Hugepagesize` instead, because that is the only
/// memory CUDA can then use (same rule as NVIDIA's snippet). `None` where
/// there is no /proc/meminfo, and on any other OS: no integrated NVIDIA die
/// exists outside Linux today, and the caller keeps the driver's figure.
fn unified_available_memory() -> Option<(u64, u64)> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut avail = None;
    let mut huge_total = 0u64;
    let mut huge_free = 0u64;
    let mut huge_size = 0u64;
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let kb: u64 = rest
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        match key {
            "MemTotal" => total = Some(kb * 1024),
            "MemAvailable" => avail = Some(kb * 1024),
            "HugePages_Total" => huge_total = kb, // a count, not kB
            "HugePages_Free" => huge_free = kb,
            "Hugepagesize" => huge_size = kb * 1024,
            _ => {}
        }
    }
    let (total, mut avail) = (total?, avail?);
    if huge_total > 0 && huge_size > 0 {
        avail = huge_free * huge_size;
    }
    Some((avail, total))
}
