//! Stable text endpoint with a leased, restartable engine. Tokenization and
//! capabilities stay resident; all model/GPU ownership stays on engine threads.
use crate::residency::{Config, LoadPolicy, Pool, Snapshot};
use paddock_engine::{
    generator::VisionBudget,
    metrics::EngineMetrics,
    service::{Engine, EngineError, GenRequest, TokenEvent},
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::Ordering::Relaxed},
    time::{Duration, Instant, SystemTime},
};

#[derive(Clone)]
pub struct Handle(Inner);
#[derive(Clone)]
enum Inner {
    Loaded(Engine),
    Resident(Arc<Resident>),
}
struct Resident {
    pool: Pool<Engine>,
    metrics: Arc<EngineMetrics>,
    budget: u64,
    vision: Option<VisionBudget>,
    admission: Arc<tokio::sync::Semaphore>,
}
impl From<Engine> for Handle {
    fn from(value: Engine) -> Self {
        Self(Inner::Loaded(value))
    }
}
impl Handle {
    pub fn residency(&self) -> Option<Snapshot> {
        match &self.0 {
            Inner::Resident(r) => Some(r.pool.snapshot()),
            _ => None,
        }
    }
    pub fn residency_budget(&self) -> Option<u64> {
        match &self.0 {
            Inner::Resident(r) => Some(r.budget),
            _ => None,
        }
    }
    pub fn metrics(&self) -> Arc<EngineMetrics> {
        match &self.0 {
            Inner::Resident(r) => r.metrics.clone(),
            Inner::Loaded(e) => e.metrics(),
        }
    }
    pub fn canvas_width(&self) -> usize {
        match &self.0 {
            Inner::Resident(_) => 256,
            Inner::Loaded(e) => e.canvas_width(),
        }
    }
    /// The most denoising steps one structured read may take; a resident
    /// canvas keeps the same capability while unloaded.
    pub fn canvas_read_steps(&self) -> u32 {
        match &self.0 {
            Inner::Resident(_) => paddock_engine::service::READ_MAX_STEPS,
            Inner::Loaded(e) => e.canvas_read_steps(),
        }
    }
    /// True when structured reads may carry images.
    pub fn canvas_images(&self) -> bool {
        match &self.0 {
            Inner::Resident(r) => r.vision.is_some(),
            Inner::Loaded(e) => e.canvas_images(),
        }
    }
    pub fn vision_budget(&self) -> Option<VisionBudget> {
        match &self.0 {
            Inner::Resident(r) => r.vision,
            Inner::Loaded(e) => e.vision_budget(),
        }
    }
    pub async fn shutdown(&self, timeout: Duration) -> bool {
        match &self.0 {
            Inner::Loaded(e) => {
                let e = e.clone();
                tokio::task::spawn_blocking(move || e.shutdown(timeout))
                    .await
                    .unwrap_or(false)
            }
            Inner::Resident(r) => {
                r.admission.close();
                r.pool.shutdown(timeout).await
            }
        }
    }
    pub fn submit(&self, mut req: GenRequest) -> Result<(), String> {
        let Inner::Resident(r) = &self.0 else {
            let Inner::Loaded(e) = &self.0 else {
                unreachable!()
            };
            return e.submit(req);
        };
        // Bound queued requests INCLUDING their prompt/canvas allocations.
        let permit = r
            .admission
            .clone()
            .try_acquire_owned()
            .map_err(|_| "model_unavailable: residency queue is full or closing".to_owned())?;
        let r = r.clone();
        req.submitted.get_or_insert_with(Instant::now);
        let caller = req.events.clone();
        tokio::spawn(async move {
            let _permit = permit;
            if caller.is_closed() {
                return;
            }
            let lease = tokio::select! {
                biased;
                _ = caller.closed() => return,
                result = r.pool.acquire() => match result {
                    Ok(lease) => lease,
                    Err(error) => { let _ = caller.send(TokenEvent::Error(EngineError::overloaded(error))); return; }
                }
            };
            let (events, mut incoming) = tokio::sync::mpsc::unbounded_channel();
            req.events = events;
            if let Err(error) = lease.submit(req) {
                let _ = caller.send(TokenEvent::Error(EngineError::internal(error)));
                return;
            }
            loop {
                tokio::select! {
                    biased;
                    _ = caller.closed() => break,
                    event = incoming.recv() => match event {
                        Some(event) => { if caller.send(event).is_err() { break; } }
                        None => break,
                    }
                }
            }
            // Closing the engine reply channel cancels the request. A disposal
            // then waits for confirmed engine-thread/GPU release, not just Done.
            drop(incoming);
            drop(lease);
        });
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Options {
    pub policy: Config,
    pub config_path: Option<PathBuf>,
    signature: Signature,
    companion: Option<(PathBuf, Signature)>,
    vision: Option<VisionBudget>,
}
impl Options {
    pub fn new(policy: Config, path: &Path, config_path: Option<PathBuf>) -> Result<Self, String> {
        policy.validate()?;
        Ok(Self {
            policy,
            config_path,
            signature: signature(path)?,
            companion: None,
            vision: None,
        })
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn with_vision(
        mut self,
        companion: Option<&Path>,
        vision: Option<VisionBudget>,
    ) -> Result<Self, String> {
        self.companion = companion
            .map(|p| signature(p).map(|s| (p.to_owned(), s)))
            .transpose()?;
        self.vision = vision;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileStamp {
    path: PathBuf,
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    identity: (u64, u64),
}
type Signature = Vec<FileStamp>;
fn signature(path: &Path) -> Result<Signature, String> {
    let mut paths = if path.is_dir() {
        let mut paths = Vec::new();
        for (i, entry) in std::fs::read_dir(path)
            .map_err(|e| e.to_string())?
            .enumerate()
        {
            if i >= 1024 {
                return Err("too many checkpoint files".into());
            }
            let p = entry.map_err(|e| e.to_string())?.path();
            if p.is_file() {
                paths.push(p);
            }
        }
        paths
    } else {
        vec![path.to_owned()]
    };
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let path = p.canonicalize().map_err(|e| e.to_string())?;
            let meta = std::fs::metadata(&path).map_err(|e| e.to_string())?;
            Ok(FileStamp {
                path,
                len: meta.len(),
                modified: meta.modified().map_err(|e| e.to_string())?,
                #[cfg(unix)]
                identity: {
                    use std::os::unix::fs::MetadataExt;
                    (meta.dev(), meta.ino())
                },
            })
        })
        .collect()
}

pub(crate) fn configure(
    options: Options,
    path: PathBuf,
    device: String,
    gpu: usize,
    reservation: u64,
    metrics: Arc<EngineMetrics>,
    build: impl Fn() -> Result<Engine, String> + Send + Sync + 'static,
) -> Result<Handle, String> {
    let loaded_metrics = metrics.clone();
    let vision = options.vision;
    let build = move || {
        let _gate = crate::device_admission::load_lock(&device, gpu)?;
        let check = || {
            let companion_matches = match &options.companion {
                Some((p, saved)) => signature(p)? == *saved,
                None => true,
            };
            if signature(&path)? == options.signature && companion_matches {
                Ok(())
            } else {
                Err("model_changed: checkpoint changed; restart this endpoint".to_owned())
            }
        };
        check()?;
        let engine = build()?;
        if let Err(error) = check().and_then(|_| {
            if engine.canvas_width() == 256
                && engine.vision_budget() == vision
                && engine.canvas_read_steps() == paddock_engine::service::READ_MAX_STEPS
            {
                Ok(())
            } else {
                Err(
                    "model_changed: loaded capabilities differ from configured DiffusionGemma"
                        .into(),
                )
            }
        }) {
            dispose(engine, &loaded_metrics);
            return Err(error);
        }
        Ok(engine)
    };
    let initial = if options.policy.load == LoadPolicy::AtStartup {
        Some(build()?)
    } else {
        None
    };
    let cleared_metrics = metrics.clone();
    let pool = Pool::new(options.policy, initial, build, move |e| {
        dispose(e, &cleared_metrics)
    });
    if let Some(path) = options.config_path {
        pool.watch_policy(path);
    }
    Ok(Handle(Inner::Resident(Arc::new(Resident {
        pool,
        metrics,
        budget: reservation,
        vision,
        admission: Arc::new(tokio::sync::Semaphore::new(32)),
    }))))
}

fn dispose(engine: Engine, metrics: &EngineMetrics) {
    // Failure is fail-closed: Pool marks Failed and closes admission, rather
    // than loading a second copy over unconfirmed GPU ownership.
    assert!(
        engine.shutdown(Duration::from_secs(120)),
        "engine disposal did not confirm memory release"
    );
    metrics.weights_mem_bytes.store(0, Relaxed);
    metrics.kv_mem_bytes.store(0, Relaxed);
    metrics.model_mem_bytes.store(0, Relaxed);
    metrics.kv_used.store(0, Relaxed);
    metrics.kv_total.store(0, Relaxed);
    metrics.active_slots.store(0, Relaxed);
    metrics.phase.store(0, Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failing_handle(count: Arc<std::sync::atomic::AtomicUsize>) -> Handle {
        let pool = Pool::new(
            Config {
                load: LoadPolicy::OnDemand,
                ..Config::default()
            },
            None,
            move || {
                count.fetch_add(1, Relaxed);
                Err("insufficient_memory".into())
            },
            |_| {},
        );
        Handle(Inner::Resident(Arc::new(Resident {
            pool,
            metrics: Arc::new(EngineMetrics::default()),
            budget: 4096,
            vision: None,
            admission: Arc::new(tokio::sync::Semaphore::new(32)),
        })))
    }
    fn request(events: tokio::sync::mpsc::UnboundedSender<TokenEvent>) -> GenRequest {
        GenRequest {
            prompt: vec![1],
            max_tokens: 1,
            sampler: Default::default(),
            stop_tokens: vec![2],
            events,
            mm_chunks: None,
            constraint: None,
            logprobs: None,
            submitted: None,
            canvas_read: None,
            user_turn: false,
        }
    }
    #[tokio::test]
    async fn load_failure_is_a_request_error_and_does_not_change_endpoint_capabilities() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handle = failing_handle(count.clone());
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        handle.submit(request(tx)).unwrap();
        let event = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(event, TokenEvent::Error(ref e) if e.message == "insufficient_memory"));
        assert_eq!(count.load(Relaxed), 1);
        assert_eq!(handle.canvas_width(), 256);
        assert_eq!(handle.residency().unwrap().load_failures, 1);
        assert!(Arc::ptr_eq(&handle.metrics(), &handle.clone().metrics()));
    }
    #[tokio::test]
    async fn cancelled_request_never_triggers_a_cold_load() {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handle = failing_handle(count.clone());
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        drop(rx);
        handle.submit(request(tx)).unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(count.load(Relaxed), 0);
        assert!(handle.shutdown(Duration::from_secs(1)).await);
        let (tx, _) = tokio::sync::mpsc::unbounded_channel();
        assert!(handle.submit(request(tx)).is_err());
    }

    #[test]
    fn checkpoint_identity_detects_atomic_replacement_and_new_shards() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("config.json");
        std::fs::write(&p, "{}").unwrap();
        let a = signature(d.path()).unwrap();
        assert_eq!(a, signature(d.path()).unwrap());
        std::fs::write(d.path().join("new.json"), "{}").unwrap();
        // A real replacement is written after the file it replaces. Say so:
        // two writes microseconds apart can carry one NTFS timestamp, and on
        // Windows the stamp has no inode to tell the files apart, so without
        // this the test failed about two runs in five there.
        std::fs::File::options()
            .write(true)
            .open(d.path().join("new.json"))
            .unwrap()
            .set_modified(a[0].modified + Duration::from_secs(2))
            .unwrap();
        std::fs::rename(d.path().join("new.json"), &p).unwrap();
        assert_ne!(a, signature(d.path()).unwrap());
        let b = signature(d.path()).unwrap();
        std::fs::write(d.path().join("model.safetensors"), "weights").unwrap();
        assert_ne!(b, signature(d.path()).unwrap());
    }
}
