//! Clef's decision service: a dedicated GPU thread owning the model, FIFO
//! request queue, oneshot replies - the Laya `decision` seam's shape for a
//! model whose unit of work is the REQUEST (one joint sequence: the state,
//! then every question with its options), not the question.
//!
//! Every pass packs whole requests in arrival order until the next one would
//! overflow the pass (rows, questions, options, requests). The model is
//! batch-invariant by construction (no GEMM splits K, the convolution, the
//! DeltaNet scan and every attention run per request, everything else is
//! row-local), so a request's logits never depend on what rode its pass -
//! the engine gate holds that bit-exact. A caller that went away takes its
//! request with it before its pass runs.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, SyncSender, channel, sync_channel};
use std::time::Instant;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

/// One question of a request, rows counted from the request's first row.
#[derive(Clone, Debug)]
pub struct ClefQuestion {
    /// the type embedding row (noul 0, choice 1, score 2)
    pub qtype: u32,
    /// the instruction's rows `[start, end)`
    pub span: (usize, usize),
    /// every option's text rows, encoding order
    pub options: Vec<(usize, usize)>,
}

/// One image of a request: its decoded pixels and where its tokens sit.
#[derive(Clone, Debug)]
pub struct ClefImage {
    /// interleaved RGB8, `height` rows of `width` pixels, as decoded
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
    /// the processor's resize target (`smart_resize`, multiples of 32)
    pub resized: (usize, usize),
    /// the request row of its first `<|image_pad|>` - its
    /// `(rh / 32) * (rw / 32)` tokens follow back to back
    pub row: usize,
}

impl ClefImage {
    /// The image's tokens: one per 2 x 2 window of 16-pixel patches.
    pub fn tokens(&self) -> usize {
        (self.resized.0 / 32) * (self.resized.1 / 32)
    }
}

/// One request of a pass, borrowed.
pub struct ClefRequest<'a> {
    pub ids: &'a [u32],
    pub questions: &'a [ClefQuestion],
    /// in sequence order, each on its own rows of `ids`
    pub images: &'a [ClefImage],
}

/// Per request, per question, per option: the head's logit.
pub type ClefLogits = Vec<Vec<Vec<f32>>>;

/// What the runner needs to describe the endpoint and check a request
/// without touching the GPU.
#[derive(Clone, Debug)]
pub struct ClefInfo {
    pub max_rows: usize,
    pub max_questions: usize,
    pub max_options: usize,
    pub max_requests: usize,
    pub vocab: usize,
    pub weight_bytes: u64,
    pub workspace_bytes: u64,
    /// whether the vision tower is loaded (requests may carry images)
    pub images: bool,
}

/// Constructed and owned on the decision thread.
pub trait ClefBackend {
    fn info(&self) -> ClefInfo;
    fn forward(&mut self, reqs: &[ClefRequest<'_>]) -> Result<ClefLogits, String>;
}

#[cfg(feature = "cuda")]
impl ClefBackend for crate::gpu_model::clef::GpuClef {
    fn info(&self) -> ClefInfo {
        use crate::gpu_model::clef::{MAX_OPTIONS, MAX_QUESTIONS, MAX_REQUESTS};
        ClefInfo {
            max_rows: self.max_rows(),
            max_questions: MAX_QUESTIONS,
            max_options: MAX_OPTIONS,
            max_requests: MAX_REQUESTS,
            vocab: self.cfg.vocab,
            weight_bytes: self.weight_bytes(),
            workspace_bytes: self.workspace_bytes(),
            images: self.has_vision(),
        }
    }

    fn forward(&mut self, reqs: &[ClefRequest<'_>]) -> Result<ClefLogits, String> {
        crate::gpu_model::clef::GpuClef::forward(self, reqs).map_err(|e| e.to_string())
    }
}

/// One request as the runner built it.
#[derive(Clone, Debug)]
pub struct ClefJob {
    pub ids: Vec<u32>,
    pub questions: Vec<ClefQuestion>,
    pub images: Vec<ClefImage>,
}

/// One request's answer.
#[derive(Clone, Debug, Default)]
pub struct ClefReply {
    /// per question, per option (encoding order): the raw logit
    pub logits: Vec<Vec<f32>>,
    /// the pass that carried it: its GPU wall and how many requests rode it
    pub gpu_ms: f64,
    pub pass_requests: usize,
}

/// Decoded RGB held by one request and by all queued/in-flight requests.
/// Image admission is independent of the token/queue-count limits: a tiny
/// resized image can still arrive as a 64 MP source photograph.
pub const CLEF_IMAGE_REQUEST_BYTES: usize = 256 * 1024 * 1024;
const IMAGE_QUEUE_BYTES: usize = 1024 * 1024 * 1024;

/// Owned by the blocking decoder, then the queue, then the executing batch.
/// Cancelling the HTTP future must not return these bytes while work owns them.
pub struct ClefImageReservation(OwnedSemaphorePermit);

impl ClefImageReservation {
    pub fn shrink_to(&mut self, bytes: usize) -> Result<(), String> {
        let excess = self
            .0
            .num_permits()
            .checked_sub(bytes)
            .ok_or("images exceed their decoded-byte reservation")?;
        drop(self.0.split(excess));
        Ok(())
    }
}

type Incoming = (
    ClefJob,
    oneshot::Sender<Result<ClefReply, String>>,
    Option<ClefImageReservation>,
);
const QUEUE_CAP: usize = 128;

/// Handle to the decision thread. Cloneable; requests are served FIFO.
#[derive(Clone)]
pub struct ClefDecider {
    tx: SyncSender<Incoming>,
    info: Arc<ClefInfo>,
    metrics: Arc<crate::metrics::EngineMetrics>,
    image_bytes: Arc<Semaphore>,
}

impl ClefDecider {
    /// Spawn the decision thread. `build` constructs the model on that
    /// thread (the CUDA context binds to it) and may fail; spawn blocks until
    /// it has and propagates the error.
    pub fn spawn<F, B>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<B, String> + Send + 'static,
        B: ClefBackend + 'static,
    {
        let (tx, rx) = sync_channel(QUEUE_CAP);
        let metrics = Arc::new(crate::metrics::EngineMetrics::default());
        let worker_metrics = Arc::clone(&metrics);
        let (ready_tx, ready_rx) = channel::<Result<ClefInfo, String>>();
        std::thread::Builder::new()
            .name("paddock-clef".into())
            .spawn(move || {
                let backend = match build() {
                    Ok(m) => m,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let info = backend.info();
                worker_metrics
                    .weights_mem_bytes
                    .store(info.weight_bytes, Relaxed);
                worker_metrics
                    .model_mem_bytes
                    .store(info.weight_bytes + info.workspace_bytes, Relaxed);
                let _ = ready_tx.send(Ok(info));
                serve(backend, rx, worker_metrics);
            })
            .map_err(|e| e.to_string())?;
        let info = ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self {
            tx,
            info: Arc::new(info),
            metrics,
            image_bytes: Arc::new(Semaphore::new(IMAGE_QUEUE_BYTES)),
        })
    }

    pub fn info(&self) -> &ClefInfo {
        &self.info
    }

    pub fn metrics(&self) -> Arc<crate::metrics::EngineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Non-waiting admission: never make an unbounded second queue of decoded
    /// images in suspended HTTP handlers. The runner reserves before decoding.
    pub fn reserve_image_bytes(&self, bytes: usize) -> Result<ClefImageReservation, String> {
        if bytes > CLEF_IMAGE_REQUEST_BYTES {
            return Err("images exceed the 256 MiB decoded-byte limit per request".into());
        }
        Arc::clone(&self.image_bytes)
            .try_acquire_many_owned(bytes as u32)
            .map(ClefImageReservation)
            .map_err(|_| "decision queue is full (image memory); retry shortly".into())
    }

    /// Answer one request; resolves when its pass has landed.
    pub async fn decide(&self, job: ClefJob) -> Result<ClefReply, String> {
        self.decide_reserved(job, None).await
    }

    pub async fn decide_reserved(
        &self,
        job: ClefJob,
        reservation: Option<ClefImageReservation>,
    ) -> Result<ClefReply, String> {
        let i = &self.info;
        let n = job.ids.len();
        let options: usize = job.questions.iter().map(|q| q.options.len()).sum();
        let span_ok = |(a, b): (usize, usize)| a < b && b <= n;
        // images in sequence order, each on its own rows inside the request
        let mut next_row = 0usize;
        let mut image_bytes = 0usize;
        for im in &job.images {
            let (rh, rw) = im.resized;
            if !i.images {
                return Err("this model was loaded without its vision tower".into());
            }
            if rh == 0
                || rw == 0
                || !rh.is_multiple_of(32)
                || !rw.is_multiple_of(32)
                || im.row < next_row
            {
                return Err("an image's place in the request is malformed".into());
            }
            if im.width == 0
                || im.height == 0
                || im
                    .width
                    .checked_mul(im.height)
                    .and_then(|n| n.checked_mul(3))
                    != Some(im.rgb.len())
            {
                return Err("an image's pixels do not match its size".into());
            }
            image_bytes = image_bytes
                .checked_add(im.rgb.len())
                .ok_or("image byte count overflow")?;
            next_row = (rh / 32)
                .checked_mul(rw / 32)
                .and_then(|n| im.row.checked_add(n))
                .ok_or("image token count overflow")?;
            if next_row > n {
                return Err("an image's tokens run past the request".into());
            }
        }
        if n == 0
            || n > i.max_rows
            || job.ids.iter().any(|&t| t as usize >= i.vocab)
            || job.questions.is_empty()
            || job.questions.len() > i.max_questions
            || options > i.max_options
            || job.questions.iter().any(|q| {
                q.qtype > 2
                    || !span_ok(q.span)
                    || q.options.is_empty()
                    || !q.options.iter().all(|&o| span_ok(o))
            })
        {
            return Err(format!(
                "a request of {n} tokens, {} questions and {options} options (the model takes \
                 1..={} tokens, up to {} questions and {} options)",
                job.questions.len(),
                i.max_rows,
                i.max_questions,
                i.max_options
            ));
        }
        let reservation = if let Some(mut r) = reservation {
            if !Arc::ptr_eq(r.0.semaphore(), &self.image_bytes) {
                return Err("image reservation belongs to another model".into());
            }
            r.shrink_to(image_bytes)?;
            Some(r)
        } else if image_bytes > 0 {
            Some(self.reserve_image_bytes(image_bytes)?)
        } else {
            None
        };
        let (tx, rx) = oneshot::channel();
        self.tx
            .try_send((job, tx, reservation))
            .map_err(|e| match e {
                std::sync::mpsc::TrySendError::Full(_) => {
                    "decision queue is full; retry shortly".to_string()
                }
                std::sync::mpsc::TrySendError::Disconnected(_) => {
                    "decision thread gone".to_string()
                }
            })?;
        rx.await
            .map_err(|_| "decision thread dropped the request".to_string())?
    }
}

fn serve(
    mut backend: impl ClefBackend,
    rx: Receiver<Incoming>,
    metrics: Arc<crate::metrics::EngineMetrics>,
) {
    let info = backend.info();
    let mut queue: VecDeque<Incoming> = VecDeque::new();
    loop {
        if queue.is_empty() {
            match rx.recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => return,
            }
        }
        while queue.len() < QUEUE_CAP {
            match rx.try_recv() {
                Ok(j) => queue.push_back(j),
                Err(_) => break,
            }
        }
        queue.retain(|(_, reply, _)| !reply.is_closed());
        if queue.is_empty() {
            continue;
        }
        // one pass: whole requests, FIFO, while they fit
        let (mut rows, mut nq, mut no, mut take) = (0usize, 0usize, 0usize, 0usize);
        for (job, _, _) in &queue {
            let (r, q, o) = (
                job.ids.len(),
                job.questions.len(),
                job.questions.iter().map(|q| q.options.len()).sum::<usize>(),
            );
            if take > 0
                && (rows + r > info.max_rows
                    || nq + q > info.max_questions
                    || no + o > info.max_options
                    || take + 1 > info.max_requests)
            {
                break;
            }
            rows += r;
            nq += q;
            no += o;
            take += 1;
        }
        let batch: Vec<Incoming> = queue.drain(..take).collect();
        let reqs: Vec<ClefRequest<'_>> = batch
            .iter()
            .map(|(j, _, _)| ClefRequest {
                ids: &j.ids,
                questions: &j.questions,
                images: &j.images,
            })
            .collect();
        let t0 = Instant::now();
        metrics.active_slots.store(take as u32, Relaxed);
        metrics.phase.store(crate::metrics::PHASE_PREFILL, Relaxed);
        let result = backend.forward(&reqs);
        metrics.phase.store(crate::metrics::PHASE_IDLE, Relaxed);
        metrics.active_slots.store(0, Relaxed);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(reqs);
        match result {
            Ok(out) => {
                metrics.prefill_tokens_total.fetch_add(rows as u64, Relaxed);
                for ((_, reply, _reservation), logits) in batch.into_iter().zip(out) {
                    let _ = reply.send(Ok(ClefReply {
                        logits,
                        gpu_ms: ms,
                        pass_requests: take,
                    }));
                }
            }
            Err(e) => {
                // a pass fails as a unit: every request in it says so
                for (_, reply, _reservation) in batch {
                    let _ = reply.send(Err(e.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Echoes each question's option count as its logits, records passes.
    struct Fake {
        passes: Arc<std::sync::Mutex<Vec<usize>>>,
    }

    impl ClefBackend for Fake {
        fn info(&self) -> ClefInfo {
            ClefInfo {
                max_rows: 10,
                max_questions: 8,
                max_options: 32,
                max_requests: 4,
                vocab: 100,
                weight_bytes: 0,
                workspace_bytes: 0,
                images: false,
            }
        }
        fn forward(&mut self, reqs: &[ClefRequest<'_>]) -> Result<ClefLogits, String> {
            self.passes.lock().unwrap().push(reqs.len());
            Ok(reqs
                .iter()
                .map(|r| {
                    r.questions
                        .iter()
                        .map(|q| vec![r.ids.len() as f32; q.options.len()])
                        .collect()
                })
                .collect())
        }
    }

    fn job(n: usize) -> ClefJob {
        ClefJob {
            ids: vec![1; n],
            questions: vec![ClefQuestion {
                qtype: 0,
                span: (0, 1),
                options: vec![(0, 1), (1, 2)],
            }],
            images: vec![],
        }
    }

    #[tokio::test]
    async fn requests_answer_and_refuse() {
        let passes = Arc::new(std::sync::Mutex::new(Vec::new()));
        let p = Arc::clone(&passes);
        let d = ClefDecider::spawn(move || Ok(Fake { passes: p })).unwrap();
        let r = d.decide(job(4)).await.unwrap();
        assert_eq!(r.logits, vec![vec![4.0, 4.0]]);
        // too long for a pass, a span past the rows, an id past the vocab
        assert!(d.decide(job(11)).await.is_err());
        let mut bad = job(4);
        bad.questions[0].options.push((3, 5));
        assert!(d.decide(bad).await.is_err());
        let mut bad = job(4);
        bad.ids[0] = 100;
        assert!(d.decide(bad).await.is_err());
        // concurrent requests ride shared passes, never past the row cap
        let (a, b, c) = tokio::join!(d.decide(job(4)), d.decide(job(5)), d.decide(job(6)));
        assert_eq!(a.unwrap().logits[0][0], 4.0);
        assert_eq!(b.unwrap().logits[0][0], 5.0);
        assert_eq!(c.unwrap().logits[0][0], 6.0);
        assert!(!passes.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn image_reservations_bound_bytes_and_release_on_validation_error() {
        let d = ClefDecider::spawn(|| {
            Ok(Fake {
                passes: Arc::default(),
            })
        })
        .unwrap();
        assert!(d.reserve_image_bytes(CLEF_IMAGE_REQUEST_BYTES + 1).is_err());
        let mut leases: Vec<_> = (0..4)
            .map(|_| d.reserve_image_bytes(CLEF_IMAGE_REQUEST_BYTES).unwrap())
            .collect();
        assert!(
            d.reserve_image_bytes(1)
                .err()
                .unwrap()
                .starts_with("decision queue is full")
        );
        leases[0].shrink_to(12).unwrap();
        assert_eq!(
            d.image_bytes.available_permits(),
            CLEF_IMAGE_REQUEST_BYTES - 12
        );
        assert!(leases[0].shrink_to(13).is_err());
        drop(leases);
        assert_eq!(d.image_bytes.available_permits(), IMAGE_QUEUE_BYTES);
        let reservation = d.reserve_image_bytes(12).unwrap();
        assert!(d.decide_reserved(job(11), Some(reservation)).await.is_err());
        assert_eq!(d.image_bytes.available_permits(), IMAGE_QUEUE_BYTES);
    }

    struct HeldPass {
        started: Option<oneshot::Sender<()>>,
        release: Receiver<()>,
    }
    impl ClefBackend for HeldPass {
        fn info(&self) -> ClefInfo {
            let mut info = Fake {
                passes: Arc::default(),
            }
            .info();
            info.images = true;
            info
        }
        fn forward(&mut self, _: &[ClefRequest<'_>]) -> Result<ClefLogits, String> {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            self.release
                .recv_timeout(std::time::Duration::from_secs(5))
                .map_err(|e| e.to_string())?;
            Ok(vec![vec![vec![1.0, 1.0]]])
        }
    }

    #[tokio::test]
    async fn cancelled_gpu_work_keeps_its_image_reservation_until_it_releases_pixels() {
        let (started, ready) = oneshot::channel();
        let (release, wait) = channel();
        let d = ClefDecider::spawn(move || {
            Ok(HeldPass {
                started: Some(started),
                release: wait,
            })
        })
        .unwrap();
        let mut image_job = job(4);
        image_job.images.push(ClefImage {
            rgb: vec![0; 12],
            width: 2,
            height: 2,
            resized: (32, 32),
            row: 0,
        });
        // Malformed internal callers cannot overflow arithmetic before validation.
        let mut malformed = image_job.clone();
        malformed.images[0].width = usize::MAX;
        assert!(d.decide(malformed).await.is_err());
        let mut malformed = image_job.clone();
        malformed.images[0].resized = (usize::MAX - 31, usize::MAX - 31);
        assert!(d.decide(malformed).await.is_err());
        let task_decider = d.clone();
        let lease = d.reserve_image_bytes(CLEF_IMAGE_REQUEST_BYTES).unwrap();
        let task =
            tokio::spawn(async move { task_decider.decide_reserved(image_job, Some(lease)).await });
        tokio::time::timeout(std::time::Duration::from_secs(5), ready)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(d.image_bytes.available_permits(), IMAGE_QUEUE_BYTES - 12);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(d.image_bytes.available_permits(), IMAGE_QUEUE_BYTES - 12);
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while d.image_bytes.available_permits() != IMAGE_QUEUE_BYTES {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
