//! Device telemetry via NVML or public Metal APIs. No helper processes or
//! elevated privileges. Metal capacity and runner-local observations stay
//! separate from NVML's machine-wide GPU usage and process attribution.
//!
//! NVML runs in exactly one process per box: the manager (doc §9). Runners link
//! no NVML - their inside view (allocator ledger, engine counters) comes from
//! their own `/api/stats`, and keeping the two views in different processes is
//! what makes the reconciliation cross-check meaningful.
//!
//! Isolation contract (must never regress):
//! - Runs on its **own OS thread**, low cadence. The manager has no inference
//!   to perturb, but the sampler must never block a request handler either.
//! - Every metric is `Option`: NVML returns `NotSupported` for absent sensors
//!   (passive datacenter cards have no fan, WDDM hides per-process util, etc.),
//!   which becomes `None`. Works uniformly across every NVIDIA arch we plan to
//!   run (Ampere -> Ada -> Hopper -> Blackwell) with no per-card code.
//! - If NVML is unavailable (no driver / no NVIDIA GPU), the snapshot is
//!   `{ available: false }` and the rest of the manager is unaffected.
//!
//! Per-runner attribution: every sample carries the device's per-PID memory
//! list (compute + graphics - WDDM can classify either way); the reconciler
//! task joins it against supervised runner PIDs and each runner's allocator
//! self-report to produce the §9 drift gauge - the memstress methodology as
//! an always-on dashboard.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::watch;

#[cfg(target_os = "macos")]
mod pressure;

/// One process holding memory on a device (NVML's outside view). `mem` is
/// None when the driver hides per-process bytes (some WDDM configurations).
#[derive(Debug, Clone, Serialize)]
pub struct GpuProc {
    pub pid: u32,
    pub mem: Option<u64>,
}

/// One device's latest metrics. Absent sensors serialize as `null`.
#[derive(Debug, Clone, Serialize, Default)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub uuid: Option<String>,
    /// PCI bus id - the stable key to correlate with a runner's CUDA device.
    pub pci: Option<String>,
    /// GPU core utilization, percent.
    pub util_gpu: Option<u32>,
    /// Memory-controller utilization, percent.
    pub util_mem: Option<u32>,
    /// Bytes.
    pub mem_used: Option<u64>,
    pub mem_total: Option<u64>,
    pub temp_c: Option<u32>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub fan_pct: Option<u32>,
    /// Processes holding memory on this device (per-PID attribution input).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub procs: Vec<GpuProc>,
    /// Apple unified-memory capacity, not dedicated VRAM or GPU usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal: Option<MetalHardware>,
    /// An NVIDIA part with no memory of its own (DGX Spark's GB10): NVML
    /// reports no framebuffer, so `mem_total` / `mem_used` are the machine's
    /// RAM as the OS books it - used = total - MemAvailable, the runner load
    /// gate's own reading.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub host_memory: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetalHardware {
    pub unified_memory_total: Option<u64>,
    pub recommended_working_set: u64,
    /// System thermal pressure, not GPU temperature.
    pub thermal_pressure: Option<&'static str>,
    /// Last public dispatch memory-pressure event; unknown before first event.
    pub memory_pressure: Option<&'static str>,
    /// Runtime capability discovery, not a claim of device utilization.
    pub counter_sets: Vec<String>,
}

/// A runner's process-local measurements, never NVML-equivalent attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalRunnerMetrics {
    pub allocated_bytes: u64,
    pub completed_commands: u64,
    pub gpu_seconds_total: f64,
    pub last_command_ms: Option<f64>,
}

/// A full-fleet sample. `available: false` means no device sampler is available.
#[derive(Debug, Clone, Serialize)]
pub struct GpuSnapshot {
    pub available: bool,
    /// Unix seconds when sampled.
    pub ts: u64,
    pub gpus: Vec<GpuInfo>,
}

impl GpuSnapshot {
    fn unavailable() -> Self {
        Self {
            available: false,
            ts: now_secs(),
            gpus: Vec::new(),
        }
    }
}

/// Handle held by `AppState`; hands out the latest snapshot and a stream.
#[derive(Clone)]
pub struct Telemetry {
    rx: watch::Receiver<Arc<GpuSnapshot>>,
}

impl Telemetry {
    /// The most recent snapshot (never blocks).
    pub fn latest(&self) -> Arc<GpuSnapshot> {
        self.rx.borrow().clone()
    }

    /// A fresh receiver for a streaming client (WebSocket). Each subscriber
    /// bumps the sender's receiver count, which ramps the sample cadence up.
    pub fn subscribe(&self) -> watch::Receiver<Arc<GpuSnapshot>> {
        self.rx.clone()
    }

    /// A disabled handle (tests / no-sampler): always reports unavailable.
    pub fn disabled() -> Self {
        let (tx, rx) = watch::channel(Arc::new(GpuSnapshot::unavailable()));
        // Keep the sender alive forever so the receiver stays valid.
        std::mem::forget(tx);
        Telemetry { rx }
    }
}

/// Spawn the sampler on a dedicated thread and return a handle. Sampling is
/// entirely independent of request handling - the returned receiver only ever
/// *reads* the latest published snapshot.
pub fn start(backend: &str) -> Telemetry {
    let (tx, rx) = watch::channel(Arc::new(GpuSnapshot::unavailable()));
    let builder = std::thread::Builder::new().name("gpu-telemetry".to_owned());
    let metal = backend == "metal";
    if let Err(e) = builder.spawn(move || {
        if metal {
            run_metal(tx);
        } else {
            run(tx);
        }
    }) {
        tracing::warn!(%e, "could not spawn GPU telemetry thread; metrics disabled");
    }
    Telemetry { rx }
}

#[cfg(target_os = "macos")]
fn run_metal(tx: watch::Sender<Arc<GpuSnapshot>>) {
    use objc2_metal::{MTLCounterSet, MTLDevice};
    let Some(device) = objc2_metal::MTLCreateSystemDefaultDevice() else {
        return;
    };
    if !device.hasUnifiedMemory() {
        return;
    }
    let pressure = pressure::Pressure::new();
    let counter_sets: Vec<String> = device
        .counterSets()
        .map(|sets| sets.iter().map(|s| s.name().to_string()).collect())
        .unwrap_or_default();
    loop {
        let snap = GpuSnapshot {
            available: true,
            ts: now_secs(),
            gpus: vec![GpuInfo {
                name: device.name().to_string(),
                metal: Some(MetalHardware {
                    unified_memory_total: crate::metal_memory::physical_bytes(),
                    recommended_working_set: device.recommendedMaxWorkingSetSize(),
                    thermal_pressure: pressure::thermal(),
                    memory_pressure: pressure.memory(),
                    counter_sets: counter_sets.clone(),
                }),
                // Never pass the manager's currentAllocatedSize off as the
                // fleet's memory, or unified RAM capacity off as VRAM.
                ..GpuInfo::default()
            }],
        };
        if tx.send(Arc::new(snap)).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(if tx.receiver_count() > 1 {
            400
        } else {
            2000
        }));
    }
}

#[cfg(not(target_os = "macos"))]
fn run_metal(_tx: watch::Sender<Arc<GpuSnapshot>>) {}

fn run(tx: watch::Sender<Arc<GpuSnapshot>>) {
    // NVML is optional. If it's missing (a container without the management
    // lib, a non-NVIDIA host), device metrics are simply absent - honestly
    // reported as `available: false`, never guessed.
    let nvml = match crate::nvml::init() {
        Ok(n) => {
            tracing::info!(
                gpus = n.device_count().unwrap_or(0),
                "GPU telemetry started (NVML)"
            );
            n
        }
        Err(e) => {
            tracing::info!(%e, "NVML unavailable - device telemetry disabled");
            return;
        }
    };

    loop {
        let snap = Arc::new(sample(&nvml));
        // Err only when every receiver has dropped (manager shutting down).
        if tx.send(snap).is_err() {
            break;
        }
        // Ramp: fast while a stream client is connected (receiver_count > 1 -
        // the AppState handle always holds one), idle-slow otherwise. NVML
        // sampling is sub-ms either way.
        let ms = if tx.receiver_count() > 1 { 400 } else { 2000 };
        std::thread::sleep(Duration::from_millis(ms));
    }
}

fn sample(nvml: &nvml_wrapper::Nvml) -> GpuSnapshot {
    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
    use nvml_wrapper::enums::device::UsedGpuMemory;

    let count = nvml.device_count().unwrap_or(0);
    let mut gpus = Vec::with_capacity(count as usize);
    for i in 0..count {
        let Ok(d) = nvml.device_by_index(i) else {
            continue;
        };
        // Each read is independently capability-probed -> None when unsupported.
        let util = d.utilization_rates().ok();
        // A part that computes in the host's memory has no framebuffer for
        // NVML to report; its device memory is the OS's (see
        // `nvml::host_memory_device`). Without this every fit check on a DGX
        // Spark read "no GPU telemetry".
        let (mem, host_memory) = match d.memory_info() {
            Ok(m) => (Some((m.used, m.total)), false),
            Err(nvml_wrapper::error::NvmlError::NotSupported)
                if crate::nvml::host_memory_device(nvml, &d) =>
            {
                let m = paddock_models::meminfo::MemInfo::read();
                (
                    m.map(|m| (m.total.saturating_sub(m.usable()), m.total)),
                    m.is_some(),
                )
            }
            Err(_) => (None, false),
        };
        // Per-PID memory: compute + graphics lists (WDDM classifies CUDA work
        // as either), de-duplicated by pid keeping the larger figure.
        let mut procs: Vec<GpuProc> = Vec::new();
        let mut raw = d.running_compute_processes().unwrap_or_default();
        raw.extend(d.running_graphics_processes().unwrap_or_default());
        for p in raw {
            let mem = match p.used_gpu_memory {
                UsedGpuMemory::Used(b) => Some(b),
                UsedGpuMemory::Unavailable => None,
            };
            match procs.iter_mut().find(|e| e.pid == p.pid) {
                Some(e) => e.mem = e.mem.max(mem),
                None => procs.push(GpuProc { pid: p.pid, mem }),
            }
        }
        gpus.push(GpuInfo {
            index: i,
            name: d.name().unwrap_or_default(),
            uuid: d.uuid().ok(),
            pci: d.pci_info().ok().map(|p| p.bus_id),
            util_gpu: util.as_ref().map(|u| u.gpu),
            util_mem: util.as_ref().map(|u| u.memory),
            mem_used: mem.map(|(used, _)| used),
            mem_total: mem.map(|(_, total)| total),
            temp_c: d.temperature(TemperatureSensor::Gpu).ok(),
            // NVML reports milliwatts.
            power_w: d.power_usage().ok().map(|mw| f64::from(mw) / 1000.0),
            power_limit_w: d
                .enforced_power_limit()
                .ok()
                .map(|mw| f64::from(mw) / 1000.0),
            sm_clock_mhz: d.clock_info(Clock::SM).ok(),
            mem_clock_mhz: d.clock_info(Clock::Memory).ok(),
            fan_pct: d.fan_speed(0).ok(),
            procs,
            metal: None,
            host_memory,
        });
    }
    GpuSnapshot {
        available: true,
        ts: now_secs(),
        gpus,
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_steady_drift_warns_once_then_on_change_then_clears() {
        let mut w = super::DriftWarnings::default();
        let gib = 1i64 << 30;
        assert_eq!(w.note(7, 0, false), None, "healthy: nothing to say");
        assert_eq!(
            w.note(7, 20 * gib, true),
            Some(true),
            "leaving tolerance warns"
        );
        assert_eq!(
            w.note(7, 20 * gib, true),
            None,
            "the same drift again: silent"
        );
        assert_eq!(
            w.note(7, 20 * gib + gib / 2, true),
            None,
            "under a GiB of movement: silent"
        );
        assert_eq!(
            w.note(7, 21 * gib + 1, true),
            Some(true),
            "a GiB of movement warns again"
        );
        assert_eq!(
            w.note(7, gib / 4, false),
            Some(false),
            "back in tolerance: one all-clear"
        );
        assert_eq!(w.note(7, gib / 4, false), None, "and only one");
        w.note(9, 20 * gib, true);
        w.retain(&std::collections::HashSet::from([7]));
        assert_eq!(
            w.note(9, 20 * gib, true),
            Some(true),
            "a runner that went away is forgotten"
        );
    }

    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_sample_exposes_capacity_without_inventing_device_usage() {
        let (tx, rx) = watch::channel(Arc::new(GpuSnapshot::unavailable()));
        let worker = std::thread::spawn(move || run_metal(tx));
        for _ in 0..100 {
            if rx.borrow().available {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let value = serde_json::to_value(&*rx.borrow().clone()).unwrap();
        drop(rx);
        worker.join().unwrap();
        assert_eq!(value["available"], true);
        let gpu = &value["gpus"][0];
        assert!(gpu["metal"]["recommended_working_set"].as_u64().unwrap() > 0);
        for field in [
            "util_gpu",
            "util_mem",
            "mem_used",
            "mem_total",
            "temp_c",
            "power_w",
        ] {
            assert!(gpu[field].is_null(), "{field} is not a Metal sensor");
        }
    }

    #[test]
    fn old_runners_and_missing_timings_do_not_become_zero_measurements() {
        assert!(serde_json::from_value::<MetalRunnerMetrics>(serde_json::json!({})).is_err());
        let value: MetalRunnerMetrics = serde_json::from_value(serde_json::json!({
            "allocated_bytes": 1024, "completed_commands": 0,
            "gpu_seconds_total": 0.0, "last_command_ms": null
        }))
        .unwrap();
        assert_eq!(value.last_command_ms, None);
        assert_eq!(value.allocated_bytes, 1024);
    }
}

// ── §9 reconciliation: inside view vs outside view vs device ────────────────

/// One runner's memory story, both views joined.
#[derive(Debug, Clone, Serialize)]
pub struct RunnerVram {
    pub port: u16,
    pub pid: u32,
    /// The GPU (NVML index) this runner's memory sits on - an OS-level fact
    /// from the per-device process lists, not trusted config. None when
    /// attribution is unavailable (WDDM blind spot). A runner appearing on
    /// several devices reports the one holding the most bytes.
    pub gpu: Option<u32>,
    /// Outside view: NVML per-PID bytes (summed across devices).
    pub nvml_mem: Option<u64>,
    /// Inside view: the runner's allocator-ledger self-report (model_mem).
    /// None for encoder-only runners (no generative engine section) or when
    /// the stats capability is unreachable.
    pub self_mem: Option<u64>,
    /// nvml - self, when both exist. A positive residue up to ~1.5 GiB is the
    /// CUDA context + driver overhead the ledger deliberately doesn't count;
    /// beyond the threshold it's flagged (leak/fragmentation).
    pub drift: Option<i64>,
    pub anomaly: bool,
    /// The runner's live engine section (tok/s, phase, KV, memory split) as
    /// self-reported over the admin pipe - labeled view, runner-authored
    /// (schemaless here so the runner's stats schema can grow freely). The
    /// Studio's GPU dock reads its engine strip from this.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal: Option<MetalRunnerMetrics>,
}

/// The manager's alertable gauge (doc §9): sum(self-reports) ≈ NVML per-PID ≈
/// device used, continuously. Every paddock box runs its own leak detector.
#[derive(Debug, Clone, Serialize)]
pub struct Reconciliation {
    pub ts: u64,
    pub runners: Vec<RunnerVram>,
    /// Whether NVML could attribute per-PID bytes at all. Windows WDDM often
    /// hides other processes' GPU memory - then the outside view (and the
    /// drift check) is honestly absent, not zero.
    pub attribution: bool,
    /// NVML bytes attributed to paddock runner PIDs (what stopping them
    /// frees). None when attribution is unavailable.
    pub paddock_mem: Option<u64>,
    /// Device used minus paddock: foreign processes + driver/desktop
    /// overhead. None when attribution is unavailable (claiming the whole
    /// device is "other" while a 25 GB runner serves would be a lie).
    pub other_mem: Option<u64>,
    pub device_used: u64,
    pub device_total: u64,
    /// Any runner over its drift threshold.
    pub anomaly: bool,
    /// The emergency: the fleet's own ledgers commit more than the card has.
    /// Computable even under the WDDM per-PID blind spot (self-reports +
    /// device total are both known), and it means the OS is paging VRAM into
    /// system RAM - the freeze-the-whole-machine failure mode. The admission
    /// guard exists to make this unreachable; if it shows anyway (adopted
    /// runners, an overcommit bypass), the UI must scream.
    pub overcommit: Option<OverCommit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverCommit {
    /// Σ fleet ledgers (self-reports), bytes.
    pub committed: u64,
    pub device_total: u64,
}

/// Per-runner drift tolerance before flagging. First-cut heuristic: the CUDA
/// context, cudart/cuBLAS workspaces and (on WDDM) paging pool sit outside
/// the allocator ledger - measured ~0.3-1.2 GiB on this stack. Negative drift
/// (ledger claims more than the OS sees) gets a tighter bound: that shape is
/// always a bookkeeping bug.
const DRIFT_HIGH: i64 = 2 << 30; // 2 GiB over
const DRIFT_LOW: i64 = -(256 << 20); // 256 MiB under

/// Spawn the reconciler task: every few seconds, join the latest NVML sample
/// against the supervisor's runner list + each runner's admin self-report.
/// Publishing over a watch keeps readers non-blocking; None until the first
/// sample with NVML available.
/// A reconciler that never runs, for a box with no card to reconcile against.
/// Same shape as `Telemetry::disabled`: the receiver stays valid forever and
/// answers "nothing to report" without a task waking every five seconds to
/// find that out again.
pub fn no_reconciler() -> watch::Receiver<Arc<Option<Reconciliation>>> {
    let (tx, rx) = watch::channel(Arc::new(None));
    std::mem::forget(tx);
    rx
}

pub fn start_reconciler(
    gpu: Telemetry,
    supervisor: Arc<crate::supervisor::Supervisor>,
) -> watch::Receiver<Arc<Option<Reconciliation>>> {
    let (tx, rx) = watch::channel(Arc::new(None));
    tokio::spawn(async move {
        let mut warned = DriftWarnings::default();
        loop {
            let snap = gpu.latest();
            if snap.available {
                let recon = reconcile(&snap, &supervisor, &mut warned).await;
                if tx.send(Arc::new(Some(recon))).is_err() {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
    rx
}

/// What the drift warning last said per runner PID, so a runner outside
/// tolerance is reported when it leaves it, when the drift moves, and as a
/// periodic reminder - not every sample. Warned every two seconds, one
/// healthy runner's constant drift was 43,200 identical lines a day, which
/// buried whatever else the log had to say and read as an alarm nobody could
/// act on.
#[derive(Default)]
struct DriftWarnings(std::collections::HashMap<u32, (i64, std::time::Instant)>);

/// A drift change worth another line, and the reminder interval for one that
/// holds still.
const DRIFT_REWARN_DELTA: i64 = 1 << 30;
const DRIFT_REWARN_EVERY: Duration = Duration::from_secs(600);

impl DriftWarnings {
    /// Whether `pid`'s drift, flagged or not this sample, deserves a line:
    /// Some(true) warn, Some(false) the all-clear, None nothing new.
    fn note(&mut self, pid: u32, drift: i64, flagged: bool) -> Option<bool> {
        match (flagged, self.0.get(&pid).copied()) {
            (false, None) => None,
            (false, Some(_)) => {
                self.0.remove(&pid);
                Some(false)
            }
            (true, Some((last, at)))
                if (drift - last).abs() < DRIFT_REWARN_DELTA
                    && at.elapsed() < DRIFT_REWARN_EVERY =>
            {
                None
            }
            (true, _) => {
                self.0.insert(pid, (drift, std::time::Instant::now()));
                Some(true)
            }
        }
    }

    /// Forget runners that are gone.
    fn retain(&mut self, live: &std::collections::HashSet<u32>) {
        self.0.retain(|pid, _| live.contains(pid));
    }
}

async fn reconcile(
    snap: &GpuSnapshot,
    supervisor: &crate::supervisor::Supervisor,
    warned: &mut DriftWarnings,
) -> Reconciliation {
    use futures::{StreamExt, stream};
    // NVML per-PID, summed across devices (a runner is single-GPU today, but
    // the join shouldn't silently break when that changes). Alongside the sum,
    // remember which device holds the most of each pid's bytes - that is the
    // runner's GPU as an OS-level fact (feeds the per-GPU estimator).
    let mut by_pid: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    let mut dev_of: std::collections::HashMap<u32, (u32, u64)> = std::collections::HashMap::new();
    for g in &snap.gpus {
        for p in &g.procs {
            if let Some(m) = p.mem {
                *by_pid.entry(p.pid).or_insert(0) += m;
                let e = dev_of.entry(p.pid).or_insert((g.index, m));
                if m > e.1 {
                    *e = (g.index, m);
                }
            }
        }
    }

    let mut runners = Vec::new();
    let mut live_pids = std::collections::HashSet::new();
    let mut paddock_mem = 0u64;
    let mut anomaly = false;
    let mut samples = stream::iter(supervisor.list().await)
        .map(|view| async move {
            // Inside view over the admin pipe: the runner's engine self-report
            // (kept whole for the Studio; model_mem drives the drift math).
            let stats = if view.status == "unreachable" {
                None
            } else {
                tokio::time::timeout(
                    Duration::from_secs(2),
                    paddock_admin::client::AdminClient::new(view.port).stats(),
                )
                .await
                .ok()
                .and_then(Result::ok)
                // Don't attach a replacement process's counters to an old PID.
                .filter(|v| {
                    v.get("pid").and_then(serde_json::Value::as_u64) == Some(u64::from(view.pid))
                        && v.get("ts")
                            .and_then(serde_json::Value::as_u64)
                            .is_some_and(|ts| now_secs().abs_diff(ts) <= 10)
                })
            };
            (view, stats)
        })
        .buffer_unordered(8);
    while let Some((view, stats)) = samples.next().await {
        let nvml_mem = by_pid.get(&view.pid).copied();
        let engine = stats
            .as_ref()
            .and_then(|v| v.get("engine").filter(|e| !e.is_null()).cloned());
        let metal = stats
            .as_ref()
            .and_then(|v| v.get("metal"))
            .and_then(|m| serde_json::from_value::<MetalRunnerMetrics>(m.clone()).ok());
        let self_mem = engine.as_ref().and_then(|e| e.get("model_mem")?.as_u64());
        let drift = match (nvml_mem, self_mem) {
            (Some(n), Some(s)) => Some(n as i64 - s as i64),
            _ => None,
        };
        let flagged = drift.is_some_and(|d| !(DRIFT_LOW..=DRIFT_HIGH).contains(&d));
        live_pids.insert(view.pid);
        match warned.note(view.pid, drift.unwrap_or(0), flagged) {
            Some(true) => tracing::warn!(
                port = view.port,
                pid = view.pid,
                nvml = nvml_mem.unwrap_or(0),
                ledger = self_mem.unwrap_or(0),
                drift = drift.unwrap_or(0),
                "VRAM reconciliation drift outside tolerance (leak/fragmentation?) - \
                 repeated only if it moves by 1 GiB or in 10 minutes"
            ),
            Some(false) => tracing::info!(
                port = view.port,
                pid = view.pid,
                drift = drift.unwrap_or(0),
                "VRAM reconciliation drift back within tolerance"
            ),
            None => {}
        }
        anomaly |= flagged;
        paddock_mem += nvml_mem.unwrap_or(0);
        runners.push(RunnerVram {
            port: view.port,
            pid: view.pid,
            gpu: dev_of.get(&view.pid).map(|&(idx, _)| idx),
            nvml_mem,
            self_mem,
            drift,
            anomaly: flagged,
            engine,
            metal,
        });
    }

    let device_used: u64 = snap.gpus.iter().filter_map(|g| g.mem_used).sum();
    runners.sort_by_key(|r| (r.port, r.pid));
    let device_total: u64 = snap.gpus.iter().filter_map(|g| g.mem_total).sum();
    // Attribution holds when NVML reported bytes for at least one runner (or
    // there are no runners to attribute). All-None with live runners = the
    // WDDM blind spot - report absence, not zeros.
    let attribution = !snap.gpus.iter().any(|g| g.metal.is_some())
        && (runners.is_empty() || runners.iter().any(|r| r.nvml_mem.is_some()));
    warned.retain(&live_pids);
    // ledger sum vs the card: > total means WDDM is paging VRAM to system RAM
    let committed: u64 = runners.iter().filter_map(|r| r.self_mem).sum();
    let overcommit = (device_total > 0 && committed > device_total).then(|| {
        tracing::error!(
            committed, device_total,
            "fleet VRAM ledgers exceed the card - the OS is paging VRAM (machine-freeze risk); stop a model"
        );
        OverCommit { committed, device_total }
    });
    Reconciliation {
        ts: now_secs(),
        runners,
        attribution,
        paddock_mem: attribution.then_some(paddock_mem),
        other_mem: attribution.then(|| device_used.saturating_sub(paddock_mem)),
        device_used,
        device_total,
        anomaly,
        overcommit,
    }
}
