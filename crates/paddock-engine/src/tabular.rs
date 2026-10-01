//! Bounded table-inference service, separate from token generation. A single
//! owning thread serializes GPU submissions; abandoned queued jobs are skipped.
use paddock_models::kumo::recipe::{Cell, Fitted, RawTable};
use paddock_models::kumo::{KumoConfig, Table};
use std::sync::{
    Arc,
    atomic::Ordering::Relaxed,
    mpsc::{SyncSender, channel, sync_channel},
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

pub struct Input {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub categorical: Vec<bool>,
    pub query_rows: usize,
}
impl Input {
    pub fn table(&self) -> Table<'_> {
        Table {
            x: &self.x,
            y: &self.y,
            categorical: &self.categorical,
            query_rows: self.query_rows,
        }
    }
}
pub struct Output {
    pub values: Vec<f32>,
    pub gpu_seconds: f64,
    pub workspace_bytes: u64,
}
pub struct Info {
    pub config: KumoConfig,
    pub weight_bytes: u64,
}
pub trait TabularBackend {
    type Context;
    fn info(&self) -> Info;
    fn predict(&mut self, input: &Table<'_>) -> Result<Output, String>;
    fn cache_budget(&self) -> u64 {
        0
    }
    fn context_bytes(&self, _rows: usize, _columns: usize) -> u64 {
        0
    }
    fn fit(&mut self, _input: &Table<'_>) -> Result<(Self::Context, Output), String> {
        Err("context caching is unavailable".into())
    }
    fn query(&mut self, _ctx: &Self::Context, _x: &[f32], _rows: usize) -> Result<Output, String> {
        Err("context caching is unavailable".into())
    }
    /// An ensemble's members - tables of the same rows and columns - in as
    /// few passes as the backend can; the default runs them one at a time.
    fn predict_many(&mut self, inputs: &[Table<'_>]) -> Result<Vec<Output>, String> {
        inputs.iter().map(|t| self.predict(t)).collect()
    }
    /// [`Self::fit`] for an ensemble's members. A context may hold several
    /// members (in order); the default fits one a context.
    fn fit_many(
        &mut self,
        inputs: &[Table<'_>],
    ) -> Result<(Vec<Self::Context>, Vec<Output>), String> {
        let mut contexts = Vec::with_capacity(inputs.len());
        let mut outputs = Vec::with_capacity(inputs.len());
        for t in inputs {
            let (ctx, out) = self.fit(t)?;
            contexts.push(ctx);
            outputs.push(out);
        }
        Ok((contexts, outputs))
    }
    /// [`Self::query`] for every member of `contexts` (as `fit_many` returned
    /// them): `xs[m]` is member m's `rows` query rows.
    fn query_many(
        &mut self,
        contexts: &[Self::Context],
        xs: &[&[f32]],
        rows: usize,
    ) -> Result<Vec<Output>, String> {
        if contexts.len() != xs.len() {
            return Err("one context a member".into());
        }
        contexts
            .iter()
            .zip(xs)
            .map(|(ctx, x)| self.query(ctx, x, rows))
            .collect()
    }
    /// Device bytes the backend keeps between requests beyond its weights
    /// and fitted contexts (a resident pass workspace), for the memory ledger.
    fn workspace_bytes(&self) -> u64 {
        0
    }
    /// Called once the service has gone `IDLE_RELEASE` without a job: give
    /// back whatever only serves throughput.
    fn idle(&mut self) {}
}
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("tabular inference queue is full; retry shortly")]
    Busy,
    #[error("tabular inference worker is unavailable")]
    Unavailable,
    #[error("{0}")]
    Backend(String),
    #[error("fitted context does not exist or has expired")]
    NotFound,
}
enum Job {
    Prepared(Input, oneshot::Sender<Result<Output, String>>),
    Recipe(Operation, oneshot::Sender<Result<RecipeOutput, Error>>),
}
pub enum Operation {
    Predict {
        table: RawTable,
        query: Vec<Vec<Cell>>,
        estimators: usize,
        seed: u64,
    },
    Fit {
        id: String,
        table: RawTable,
        estimators: usize,
        seed: u64,
    },
    Query {
        id: String,
        query: Vec<Vec<Cell>>,
    },
    Release {
        id: String,
    },
}
pub struct RecipeOutput {
    pub output: Output,
    pub classes: Vec<Cell>,
    pub context_rows: usize,
    pub query_rows: usize,
    pub columns: usize,
    pub estimators: usize,
    pub seed: u64,
    pub context_id: Option<String>,
    pub cache_bytes: u64,
}
struct Session<C> {
    recipe: Fitted,
    contexts: Vec<C>,
    bytes: u64,
    last_used: Instant,
}
const CACHE_TTL: Duration = Duration::from_secs(600);
const MAX_CONTEXTS: usize = 4;
/// Quiet time after which a backend's throughput-only memory goes back.
const IDLE_RELEASE: Duration = Duration::from_secs(60);

fn expire<C>(sessions: &mut HashMap<String, Session<C>>) {
    sessions.retain(|_, s| s.last_used.elapsed() < CACHE_TTL);
}

fn run_recipe<B: TabularBackend>(
    backend: &mut B,
    sessions: &mut HashMap<String, Session<B::Context>>,
    op: Operation,
    reply: &oneshot::Sender<Result<RecipeOutput, Error>>,
) -> Result<RecipeOutput, Error> {
    let task = backend.info().config.task;
    let mut seconds = 0.;
    let mut workspace = 0;
    let empty = || Output {
        values: Vec::new(),
        gpu_seconds: 0.,
        workspace_bytes: 0,
    };
    match op {
        Operation::Release { id } => {
            sessions.remove(&id).ok_or(Error::NotFound)?;
            Ok(RecipeOutput {
                output: empty(),
                classes: Vec::new(),
                context_rows: 0,
                query_rows: 0,
                columns: 0,
                estimators: 0,
                seed: 0,
                context_id: Some(id),
                cache_bytes: 0,
            })
        }
        Operation::Query { id, query } => {
            let session = sessions.get_mut(&id).ok_or(Error::NotFound)?;
            let f = &session.recipe;
            f.validate_query(&query).map_err(Error::Invalid)?;
            session.last_used = Instant::now();
            if reply.is_closed() {
                return Err(Error::Unavailable);
            }
            let xs = f
                .members
                .iter()
                .map(|m| m.transform(&query))
                .collect::<Vec<_>>();
            let xs = xs.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let mut outputs = Vec::new();
            for out in backend
                .query_many(&session.contexts, &xs, query.len())
                .map_err(Error::Backend)?
            {
                seconds += out.gpu_seconds;
                workspace = workspace.max(out.workspace_bytes);
                outputs.push(out.values);
            }
            session.last_used = Instant::now();
            Ok(RecipeOutput {
                output: Output {
                    values: f
                        .reduce(&outputs, &task, query.len())
                        .map_err(Error::Backend)?,
                    gpu_seconds: seconds,
                    workspace_bytes: workspace,
                },
                classes: f.classes.clone(),
                context_rows: f.context_rows,
                query_rows: query.len(),
                columns: f.categorical.len(),
                estimators: f.members.len(),
                seed: f.seed,
                context_id: Some(id),
                cache_bytes: session.bytes,
            })
        }
        op => {
            let (table, query, estimators, seed, id) = match op {
                Operation::Predict {
                    table,
                    query,
                    estimators,
                    seed,
                } => (table, query, estimators, seed, None),
                Operation::Fit {
                    id,
                    table,
                    estimators,
                    seed,
                } => (table, Vec::new(), estimators, seed, Some(id)),
                _ => unreachable!(),
            };
            let mut f = Fitted::fit(&table, &task, estimators, seed).map_err(Error::Invalid)?;
            if id.is_none() {
                f.validate_query(&query).map_err(Error::Invalid)?;
            }
            let gpu_bytes = f
                .members
                .iter()
                .map(|m| backend.context_bytes(f.context_rows, m.categorical.len()))
                .sum::<u64>();
            let mut bytes = f.resident_bytes() + gpu_bytes;
            if id.is_some() {
                if backend.cache_budget() == 0 || bytes > backend.cache_budget() {
                    return Err(Error::Invalid("fitted ensemble exceeds the context-cache budget; reduce context, columns or estimators".into()));
                }
                // Never evict a context in use: this is the single owning thread.
                while sessions.len() >= MAX_CONTEXTS
                    || sessions.values().map(|s| s.bytes).sum::<u64>() + bytes
                        > backend.cache_budget()
                {
                    let oldest = sessions
                        .iter()
                        .min_by_key(|(_, s)| s.last_used)
                        .map(|(id, _)| id.clone())
                        .ok_or(Error::Unavailable)?;
                    sessions.remove(&oldest);
                }
            }
            if reply.is_closed() {
                return Err(Error::Unavailable);
            }
            // the members share rows and columns: one batch for the backend
            let xs = f
                .members
                .iter()
                .map(|member| {
                    let mut x = member.context.clone();
                    if id.is_none() {
                        x.extend(member.transform(&query));
                    }
                    x
                })
                .collect::<Vec<_>>();
            let tables = f
                .members
                .iter()
                .zip(&xs)
                .map(|(member, x)| Table {
                    x,
                    y: &member.y,
                    categorical: &member.categorical,
                    query_rows: query.len(),
                })
                .collect::<Vec<_>>();
            let (contexts, outs) = if id.is_some() {
                backend.fit_many(&tables).map_err(Error::Backend)?
            } else {
                (
                    Vec::new(),
                    backend.predict_many(&tables).map_err(Error::Backend)?,
                )
            };
            let mut outputs = Vec::new();
            for out in outs {
                seconds += out.gpu_seconds;
                workspace = workspace.max(out.workspace_bytes);
                outputs.push(out.values);
            }
            if id.is_some() {
                // Fitted transforms and actual neural KV are sufficient for
                // replay; don't retain duplicate training matrices/labels.
                for member in &mut f.members {
                    member.context = Vec::new();
                    member.y = Vec::new();
                }
                bytes = f.resident_bytes() + gpu_bytes;
            }
            let values = if id.is_some() {
                Vec::new()
            } else {
                f.reduce(&outputs, &task, query.len())
                    .map_err(Error::Backend)?
            };
            let result = RecipeOutput {
                output: Output {
                    values,
                    gpu_seconds: seconds,
                    workspace_bytes: workspace,
                },
                classes: f.classes.clone(),
                context_rows: f.context_rows,
                query_rows: query.len(),
                columns: f.categorical.len(),
                estimators: f.members.len(),
                seed: f.seed,
                context_id: id.clone(),
                cache_bytes: if id.is_some() { bytes } else { 0 },
            };
            if let Some(id) = id
                && !reply.is_closed()
            {
                sessions.insert(
                    id,
                    Session {
                        recipe: f,
                        contexts,
                        bytes,
                        last_used: Instant::now(),
                    },
                );
            }
            Ok(result)
        }
    }
}
#[derive(Clone)]
pub struct Tabular {
    tx: SyncSender<Job>,
    info: Arc<Info>,
    metrics: Arc<crate::metrics::EngineMetrics>,
    cache_budget: u64,
}
impl Tabular {
    pub fn spawn<F, B>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<B, String> + Send + 'static,
        B: TabularBackend + 'static,
    {
        let (tx, rx) = sync_channel::<Job>(8);
        let (ready, wait) = channel();
        let metrics = Arc::new(crate::metrics::EngineMetrics::default());
        let m = Arc::clone(&metrics);
        std::thread::Builder::new()
            .name("paddock-tabular".into())
            .spawn(move || {
                let mut backend = match build() {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let info = backend.info();
                m.weights_mem_bytes.store(info.weight_bytes, Relaxed);
                m.model_mem_bytes.store(info.weight_bytes, Relaxed);
                if ready.send(Ok((info, backend.cache_budget()))).is_err() {
                    return;
                }
                let mut sessions = HashMap::new();
                let mut last_job = Instant::now();
                loop {
                    expire(&mut sessions);
                    m.model_mem_bytes.store(
                        backend.info().weight_bytes
                            + backend.workspace_bytes()
                            + sessions.values().map(|s| s.bytes).sum::<u64>(),
                        Relaxed,
                    );
                    let job = match rx.recv_timeout(Duration::from_secs(1)) {
                        Ok(job) => job,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                            if last_job.elapsed() >= IDLE_RELEASE {
                                backend.idle();
                            }
                            continue;
                        }
                        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    };
                    last_job = Instant::now();
                    // A context can expire while recv_timeout waits. Never
                    // revive an expired handle just because its query woke us.
                    expire(&mut sessions);
                    match job {
                        Job::Prepared(input, reply) => {
                            if reply.is_closed() {
                                continue;
                            }
                            m.active_slots.store(1, Relaxed);
                            let out = backend.predict(&input.table());
                            let _ = reply.send(out);
                        }
                        Job::Recipe(op, reply) => {
                            if reply.is_closed() {
                                continue;
                            }
                            m.active_slots.store(1, Relaxed);
                            let out = run_recipe(&mut backend, &mut sessions, op, &reply);
                            let id = out
                                .as_ref()
                                .ok()
                                .filter(|r| r.query_rows == 0)
                                .and_then(|r| r.context_id.clone());
                            if reply.send(out).is_err()
                                && let Some(id) = id
                            {
                                sessions.remove(&id);
                            }
                        }
                    }
                    m.active_slots.store(0, Relaxed);
                    // predict returns only after GPU completion, including when
                    // the client has gone away. Never reuse live command memory.
                }
            })
            .map_err(|e| e.to_string())?;
        let (info, cache_budget) = wait
            .recv()
            .map_err(|_| "tabular worker failed during load")??;
        Ok(Self {
            tx,
            info: Arc::new(info),
            metrics,
            cache_budget,
        })
    }
    pub fn info(&self) -> &Info {
        &self.info
    }
    pub fn cache_budget_bytes(&self) -> u64 {
        self.cache_budget
    }
    pub fn metrics(&self) -> Arc<crate::metrics::EngineMetrics> {
        Arc::clone(&self.metrics)
    }
    pub async fn predict(&self, input: Input) -> Result<Output, Error> {
        input
            .table()
            .validate(&self.info.config.task)
            .map_err(Error::Invalid)?;
        let (tx, rx) = oneshot::channel();
        self.tx
            .try_send(Job::Prepared(input, tx))
            .map_err(|e| match e {
                std::sync::mpsc::TrySendError::Full(_) => Error::Busy,
                std::sync::mpsc::TrySendError::Disconnected(_) => Error::Unavailable,
            })?;
        rx.await
            .map_err(|_| Error::Unavailable)?
            .map_err(Error::Backend)
    }
    pub async fn recipe(&self, op: Operation) -> Result<RecipeOutput, Error> {
        let (tx, rx) = oneshot::channel();
        self.tx.try_send(Job::Recipe(op, tx)).map_err(|e| match e {
            std::sync::mpsc::TrySendError::Full(_) => Error::Busy,
            std::sync::mpsc::TrySendError::Disconnected(_) => Error::Unavailable,
        })?;
        rx.await.map_err(|_| Error::Unavailable)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake;
    struct Caching;
    impl TabularBackend for Caching {
        type Context = f32;
        fn info(&self) -> Info {
            Fake.info()
        }
        fn predict(&mut self, t: &Table<'_>) -> Result<Output, String> {
            Fake.predict(t)
        }
        fn cache_budget(&self) -> u64 {
            1 << 20
        }
        fn context_bytes(&self, _r: usize, _c: usize) -> u64 {
            4
        }
        fn fit(&mut self, t: &Table<'_>) -> Result<(f32, Output), String> {
            t.validate_context(&self.info().config.task)?;
            Ok((
                t.y[0],
                Output {
                    values: Vec::new(),
                    gpu_seconds: 0.,
                    workspace_bytes: 0,
                },
            ))
        }
        fn query(&mut self, ctx: &f32, _x: &[f32], rows: usize) -> Result<Output, String> {
            Ok(Output {
                values: vec![*ctx; rows * 999],
                gpu_seconds: 0.,
                workspace_bytes: 0,
            })
        }
    }
    fn raw(mean: f64) -> RawTable {
        RawTable {
            context: vec![vec![Cell::Number(1.)], vec![Cell::Number(2.)]],
            targets: vec![Cell::Number(mean), Cell::Number(mean)],
            categorical: vec![false],
        }
    }
    #[test]
    fn expiry_reclaims_only_idle_contexts() {
        let mut sessions = HashMap::new();
        for (id, age) in [
            ("expired", CACHE_TTL + Duration::from_secs(1)),
            ("live", Duration::ZERO),
        ] {
            sessions.insert(
                id.into(),
                Session {
                    recipe: Fitted::fit(&raw(0.), &paddock_models::kumo::Task::Regression, 1, 0)
                        .unwrap(),
                    contexts: vec![()],
                    bytes: 4,
                    last_used: Instant::now() - age,
                },
            );
        }
        expire(&mut sessions);
        assert!(!sessions.contains_key("expired"));
        assert!(sessions.contains_key("live"));
    }
    #[tokio::test]
    async fn context_isolation_lru_and_release() {
        let s = Tabular::spawn(|| Ok(Caching)).unwrap();
        for i in 0..MAX_CONTEXTS + 1 {
            let fit = s
                .recipe(Operation::Fit {
                    id: format!("ctx{i}"),
                    table: raw(i as f64),
                    estimators: 4,
                    seed: 7,
                })
                .await
                .unwrap();
            assert!(fit.cache_bytes > 0);
        }
        let query = |id| Operation::Query {
            id,
            query: vec![vec![Cell::Number(3.)]],
        };
        assert!(matches!(
            s.recipe(query("ctx0".into())).await,
            Err(Error::NotFound)
        ));
        for i in 1..MAX_CONTEXTS + 1 {
            let out = s.recipe(query(format!("ctx{i}"))).await.unwrap();
            assert_eq!(out.output.values, vec![i as f32; 999]);
            s.recipe(Operation::Release {
                id: format!("ctx{i}"),
            })
            .await
            .unwrap();
            assert!(matches!(
                s.recipe(query(format!("ctx{i}"))).await,
                Err(Error::NotFound)
            ));
        }
    }
    #[tokio::test]
    async fn raw_validation_does_not_create_a_context() {
        let s = Tabular::spawn(|| Ok(Caching)).unwrap();
        for estimators in [0, 17, usize::MAX] {
            assert!(matches!(
                s.recipe(Operation::Fit {
                    id: "invalid".into(),
                    table: raw(1.),
                    estimators,
                    seed: 0
                })
                .await,
                Err(Error::Invalid(_))
            ));
        }
        assert!(matches!(
            s.recipe(Operation::Query {
                id: "invalid".into(),
                query: vec![vec![Cell::Number(3.)]]
            })
            .await,
            Err(Error::NotFound)
        ));
    }
    impl TabularBackend for Fake {
        type Context = ();
        fn info(&self) -> Info {
            Info {
                config: KumoConfig {
                    task: paddock_models::kumo::Task::Regression,
                    size: "small".into(),
                    cell: 128,
                    embedding_layers: 4,
                    inducing: 128,
                    hidden: 512,
                    layers: 12,
                    heads: 8,
                    query_kv_heads: 8,
                },
                weight_bytes: 128,
            }
        }
        fn predict(&mut self, t: &Table<'_>) -> Result<Output, String> {
            Ok(Output {
                values: vec![t.y[0]; t.query_rows * 999],
                gpu_seconds: 0.,
                workspace_bytes: 128,
            })
        }
    }
    #[tokio::test]
    async fn validates_before_admission_and_reports_real_contract() {
        let service = Tabular::spawn(|| Ok(Fake)).unwrap();
        let input = || Input {
            x: vec![1., 2.],
            y: vec![3.],
            categorical: vec![false],
            query_rows: 1,
        };
        let mut invalid = input();
        invalid.query_rows = 0;
        assert!(matches!(
            service.predict(invalid).await,
            Err(Error::Invalid(_))
        ));
        assert_eq!(
            service.predict(input()).await.unwrap().values,
            vec![3.; 999]
        );
        assert_eq!(service.metrics().weights_mem_bytes.load(Relaxed), 128);
    }
    #[test]
    fn load_errors_are_returned() {
        assert!(
            Tabular::spawn(|| Err::<Fake, _>("bad weights".into()))
                .err()
                .unwrap()
                .contains("bad weights")
        );
    }

    #[test]
    fn bounded_queue_skips_canceled_jobs_and_releases_backend() {
        use std::sync::{atomic::AtomicUsize, mpsc::Receiver};
        use std::time::Duration;
        struct Blocking {
            started: SyncSender<()>,
            release: Receiver<()>,
            done: SyncSender<()>,
            calls: Arc<AtomicUsize>,
        }
        impl TabularBackend for Blocking {
            type Context = ();
            fn info(&self) -> Info {
                Fake.info()
            }
            fn predict(&mut self, t: &Table<'_>) -> Result<Output, String> {
                self.calls.fetch_add(1, Relaxed);
                self.started.send(()).unwrap();
                self.release.recv_timeout(Duration::from_secs(30)).unwrap();
                Fake.predict(t)
            }
        }
        impl Drop for Blocking {
            fn drop(&mut self) {
                let _ = self.done.send(());
            }
        }
        let (started, wait_start) = sync_channel(1);
        let (release, wait_release) = sync_channel(1);
        let (done, wait_done) = sync_channel(1);
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let service = Tabular::spawn(move || {
            Ok(Blocking {
                started,
                release: wait_release,
                done,
                calls: count,
            })
        })
        .unwrap();
        let input = || Input {
            x: vec![1., 2.],
            y: vec![3.],
            categorical: vec![false],
            query_rows: 1,
        };
        let (tx, rx) = oneshot::channel();
        assert!(service.tx.try_send(Job::Prepared(input(), tx)).is_ok());
        wait_start.recv_timeout(Duration::from_secs(30)).unwrap();
        drop(rx); // cancellation during active inference still completes
        for _ in 0..8 {
            let (tx, rx) = oneshot::channel();
            drop(rx);
            assert!(service.tx.try_send(Job::Prepared(input(), tx)).is_ok());
        }
        let (tx, _rx) = oneshot::channel();
        assert!(matches!(
            service.tx.try_send(Job::Prepared(input(), tx)),
            Err(std::sync::mpsc::TrySendError::Full(_))
        ));
        drop(service);
        release.send(()).unwrap();
        wait_done.recv_timeout(Duration::from_secs(30)).unwrap();
        assert_eq!(
            calls.load(Relaxed),
            1,
            "canceled queued requests must not run"
        );
    }
}
