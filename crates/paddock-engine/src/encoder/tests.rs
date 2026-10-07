//! Scheduler-only transport fixtures. No model inference or CPU oracle.
use super::*;
use std::sync::Mutex;

struct IdleTransport {
    ready: Arc<std::sync::atomic::AtomicBool>,
    released: Sender<()>,
    inflight: usize,
    cached: bool,
}
impl EncoderBackend for IdleTransport {
    type Pending = ();
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(7)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(if self.cached { 11 } else { 7 })
    }
    fn coalesce_row_budget(&self) -> usize {
        4
    }
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, _: &()) -> bool {
        self.ready.load(Relaxed)
    }
    fn embed_submit(&mut self, _: &[Vec<u32>], _: usize) -> Result<(), String> {
        self.inflight += 1;
        self.cached = true;
        Ok(())
    }
    fn embed_collect(&mut self, _: &()) -> Result<Vec<Vec<f32>>, String> {
        self.inflight -= 1;
        Ok(vec![vec![1.]])
    }
    fn idle_reclaim_after(&self) -> Option<std::time::Duration> {
        self.cached.then_some(std::time::Duration::from_millis(10))
    }
    fn reclaim_idle(&mut self) {
        assert_eq!(self.inflight, 0, "reclaim called before collection");
        self.cached = false;
        self.released.send(()).unwrap();
    }
    fn rerank_submit(&mut self, _: &[Vec<u32>], _: u32, _: u32, _: usize) -> Result<(), String> {
        Err("not supported by this transport".into())
    }
    fn rerank_collect(&mut self, _: &(), _: u32, _: u32) -> Result<Vec<f32>, String> {
        Err("not supported by this transport".into())
    }
}

#[test]
fn idle_reclamation_waits_for_collection_and_stops_waking_after_release() {
    let (released, rx) = channel();
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ready_copy = ready.clone();
    let metrics = Arc::new(EngineMetrics::default());
    let encoder = Encoder::spawn(
        move || {
            Ok(IdleTransport {
                ready: ready_copy,
                released,
                inflight: 0,
                cached: false,
            })
        },
        Some(metrics.clone()),
    )
    .unwrap();
    let (reply, response) = oneshot::channel();
    encoder
        .tx
        .send(EncodeJob::Embed {
            seqs: vec![vec![1]],
            media: Vec::new(),
            dimensions: None,
            reply,
        })
        .unwrap();
    assert!(matches!(
        rx.recv_timeout(std::time::Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    ready.store(true, Relaxed);
    assert_eq!(response.blocking_recv().unwrap().unwrap(), vec![vec![1.]]);
    rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    assert!(matches!(
        rx.recv_timeout(std::time::Duration::from_millis(50)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout)
    ));
    assert_eq!(metrics.model_mem_bytes.load(Relaxed), 7);
    // A new request must re-arm idle reclamation, with no stale wakeup loop.
    let (reply, response) = oneshot::channel();
    encoder
        .tx
        .send(EncodeJob::Embed {
            seqs: vec![vec![1]],
            media: Vec::new(),
            dimensions: None,
            reply,
        })
        .unwrap();
    response.blocking_recv().unwrap().unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
}

struct Transport {
    submitted: Arc<Mutex<Vec<usize>>>,
}
impl EncoderBackend for Transport {
    type Pending = Vec<Vec<u32>>;
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(7)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(11)
    }
    fn coalesce_row_budget(&self) -> usize {
        4
    }
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, _: &Self::Pending) -> bool {
        true
    }
    fn embed_submit(&mut self, s: &[Vec<u32>], _: usize) -> Result<Self::Pending, String> {
        let rows = s.iter().map(Vec::len).sum();
        assert!(rows <= 4, "scheduler overfilled backend row budget");
        self.submitted.lock().unwrap().push(rows);
        Ok(s.to_vec())
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> Result<Vec<Vec<f32>>, String> {
        Ok(p.iter().map(|s| vec![s[0] as f32]).collect())
    }
    fn rerank_submit(
        &mut self,
        s: &[Vec<u32>],
        _: u32,
        _: u32,
        l: usize,
    ) -> Result<Self::Pending, String> {
        self.embed_submit(s, l)
    }
    fn rerank_collect(&mut self, p: &Self::Pending, _: u32, _: u32) -> Result<Vec<f32>, String> {
        Ok(p.iter().map(|s| s[0] as f32).collect())
    }
}

#[tokio::test]
async fn bounded_merges_preserve_replies_and_skip_cancelled_jobs() {
    let submitted = Arc::new(Mutex::new(Vec::new()));
    let copy = submitted.clone();
    let metrics = Arc::new(EngineMetrics::default());
    let encoder = Encoder::spawn(
        move || Ok(Transport { submitted: copy }),
        Some(metrics.clone()),
    )
    .unwrap();
    assert!(!encoder.block_scale_calibration());
    let (tx, rx) = oneshot::channel();
    drop(rx);
    encoder
        .tx
        .send(EncodeJob::Embed {
            dimensions: None,
            media: Vec::new(),
            seqs: vec![vec![999; 99]],
            reply: tx,
        })
        .unwrap();
    let mut receivers = Vec::new();
    for i in 0..12 {
        let (reply, rx) = oneshot::channel();
        encoder
            .tx
            .send(EncodeJob::Embed {
                dimensions: None,
                media: Vec::new(),
                seqs: vec![vec![i; 3]],
                reply,
            })
            .unwrap();
        receivers.push(rx);
    }
    for (i, rx) in receivers.into_iter().enumerate() {
        assert_eq!(rx.await.unwrap().unwrap(), vec![vec![i as f32]]);
    }
    assert_eq!(
        encoder.rerank(vec![vec![20; 3]], 1, 2).await.unwrap(),
        vec![20.]
    );
    assert_eq!(metrics.weights_mem_bytes.load(Relaxed), 7);
    assert!(
        !encoder
            .apply_profile("cuda-only".into(), None)
            .await
            .unwrap()
    );
    assert!(encoder.calibrate(vec![], 0, vec![]).await.is_err());
    assert!(submitted.lock().unwrap().iter().all(|&n| n <= 4));
    assert!(
        encoder
            .embed_dimensions(vec![vec![1]], Some(128))
            .await
            .is_err()
    );
}

struct DimensionsTransport(Transport);
impl EncoderBackend for DimensionsTransport {
    type Pending = (Vec<Vec<u32>>, usize);
    fn weights_mem_bytes(&self) -> Option<u64> {
        None
    }
    fn device_mem_used(&self) -> Option<u64> {
        None
    }
    fn coalesce_row_budget(&self) -> usize {
        4
    }
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, _: &Self::Pending) -> bool {
        true
    }
    fn validate_dimensions(&self, dimensions: Option<usize>) -> Result<(), String> {
        match dimensions {
            None | Some(128 | 256 | 512 | 768) => Ok(()),
            _ => Err("unsupported dimensions".into()),
        }
    }
    fn embed_submit(&mut self, s: &[Vec<u32>], lane: usize) -> Result<Self::Pending, String> {
        self.embed_submit_dimensions(s, lane, None)
    }
    fn embed_submit_dimensions(
        &mut self,
        s: &[Vec<u32>],
        lane: usize,
        dimensions: Option<usize>,
    ) -> Result<Self::Pending, String> {
        self.validate_dimensions(dimensions)?;
        Ok((self.0.embed_submit(s, lane)?, dimensions.unwrap_or(768)))
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> Result<Vec<Vec<f32>>, String> {
        // A transport marker, not an embedding or host inference oracle.
        Ok(p.0.iter().map(|s| vec![s[0] as f32, p.1 as f32]).collect())
    }
    fn rerank_submit(
        &mut self,
        _: &[Vec<u32>],
        _: u32,
        _: u32,
        _: usize,
    ) -> Result<Self::Pending, String> {
        Err("not a reranker".into())
    }
    fn rerank_collect(&mut self, _: &Self::Pending, _: u32, _: u32) -> Result<Vec<f32>, String> {
        Err("not a reranker".into())
    }
}

#[tokio::test]
async fn mixed_dimensions_never_coalesce_into_the_wrong_output_shape() {
    let submitted = Arc::new(Mutex::new(Vec::new()));
    let copy = submitted.clone();
    let encoder = Encoder::spawn(
        move || Ok(DimensionsTransport(Transport { submitted: copy })),
        None,
    )
    .unwrap();
    let mut replies = Vec::new();
    for i in 0..24 {
        let dimensions = [None, Some(128), Some(128), Some(256), Some(512), Some(768)][i % 6];
        let (reply, rx) = oneshot::channel();
        encoder
            .tx
            .send(EncodeJob::Embed {
                seqs: vec![vec![i as u32]],
                media: Vec::new(),
                dimensions,
                reply,
            })
            .unwrap();
        replies.push((i, dimensions.unwrap_or(768), rx));
    }
    for (i, dim, rx) in replies {
        assert_eq!(rx.await.unwrap().unwrap(), vec![vec![i as f32, dim as f32]]);
    }
    assert!(
        encoder
            .embed_dimensions(vec![vec![1]], Some(129))
            .await
            .is_err()
    );
    assert!(
        encoder
            .embed_dimensions(vec![vec![1]], Some(0))
            .await
            .is_err()
    );
    assert!(submitted.lock().unwrap().iter().all(|&n| n <= 4));
}
