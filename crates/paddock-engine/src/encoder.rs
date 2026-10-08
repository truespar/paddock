//! Shared retrieval scheduler: a dedicated thread owns a native GPU backend.
//! Same-kind requests coalesce into ragged batches, with bounded enqueue-ahead
//! and per-request result slicing. Backend events and GPU heads stay behind the
//! executor contract, so HTTP handlers never depend on CUDA or Metal details.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Sender, channel};

use tokio::sync::oneshot;

mod backend;
pub mod embedding_gemma2;
use crate::metrics::EngineMetrics;
pub use backend::EncoderBackend;
#[cfg(test)]
mod tests;

/// What an encode job asks the model for.
pub enum EncodeJob {
    /// L2-normalized last-token embeddings, one per input sequence.
    /// `media[i]` holds sequence i's pictures and audio clips in the order
    /// their placeholder runs appear in its ids (empty for text); a backend
    /// without towers refuses any.
    Embed {
        seqs: Vec<Vec<u32>>,
        media: Vec<Vec<crate::service::MmChunk>>,
        dimensions: Option<usize>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>>,
    },
    /// Relevance scores (P(yes) at the last position), one per input sequence.
    Rerank {
        seqs: Vec<Vec<u32>>,
        yes: u32,
        no: u32,
        reply: oneshot::Sender<Result<Vec<f32>, String>>,
    },
    /// Load-time block-scale quality calibration on a tokenized retrieval
    /// task (docs then queries; `rel[q]` = each query's source doc). Picks
    /// the fastest per-model class profile that holds recall vs the Q8_0
    /// baseline; replies with the selected profile name.
    Calibrate {
        seqs: Vec<Vec<u32>>,
        n_docs: usize,
        rel: Vec<usize>,
        reply: oneshot::Sender<Result<&'static str, String>>,
    },
    /// Reranker calibration: same ladder, but the metric is the yes/no
    /// judge's ranking quality over `rel.len()` groups of `group` prompts.
    CalibrateRerank {
        seqs: Vec<Vec<u32>>,
        yes: u32,
        no: u32,
        group: usize,
        rel: Vec<usize>,
        reply: oneshot::Sender<Result<&'static str, String>>,
    },
    /// Apply a previously-calibrated profile by name (the disk-cache fast
    /// path). `smooth` optionally carries cached SmoothQuant s-vectors
    /// (imported first - smooth profiles need them). Replies false when it
    /// cannot apply - the caller then falls back to Calibrate.
    ApplyProfile {
        profile: String,
        smooth: Option<Vec<u8>>,
        reply: oneshot::Sender<Result<bool, String>>,
    },
    /// Serialize the calibrated SmoothQuant s-vectors for the caller's
    /// verdict cache (None until a calibration built them).
    ExportSmooth {
        reply: oneshot::Sender<Result<Option<Vec<u8>>, String>>,
    },
}

/// Handle to an encoder running on its own device-owning thread.
#[derive(Clone)]
pub struct Encoder {
    tx: Sender<EncodeJob>,
    block_scale_calibration: bool,
    /// (pictures, audio, video frames) - see `EncoderBackend::media_kinds`.
    media: (bool, bool, bool),
}

impl Encoder {
    /// Spawn the encoder thread. `build` constructs the model on that thread
    /// (CUDA context binding) and may fail; spawn blocks until build finishes
    /// and propagates the error.
    ///
    /// `metrics`, when given, gets the memory rows of `/api/stats` - an
    /// encoder published no engine block at all before, so the Studio's
    /// memory breakdown was blank. Only the memory counters are meaningful
    /// here: an encoder generates no tokens, so tok/s and phase stay at their
    /// (true) zero.
    pub fn spawn<F, B>(build: F, metrics: Option<Arc<EngineMetrics>>) -> Result<Self, String>
    where
        F: FnOnce() -> Result<B, String> + Send + 'static,
        B: EncoderBackend + 'static,
    {
        let (tx, rx) = channel::<EncodeJob>();
        let (ready_tx, ready_rx) = channel::<Result<(bool, (bool, bool, bool)), String>>();

        std::thread::Builder::new()
            .name("paddock-encoder".into())
            .spawn(move || {
                let mut model = match build() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok((m.block_scale_calibration(), m.media_kinds())));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // Re-read after every plane-mutating job, not just after load:
                // the startup calibration re-quantizes to a different class
                // and the weight total moves with it.
                let stamp = |m: &B| {
                    if let Some(mt) = metrics.as_deref() {
                        if let Some(b) = m.weights_mem_bytes() {
                            mt.weights_mem_bytes.store(b, Relaxed);
                        }
                        if let Some(b) = m.device_mem_used() {
                            mt.model_mem_bytes.store(b, Relaxed);
                        }
                    }
                };
                stamp(&model);
                // Cross-request coalescing + one-batch-deep pipelining.
                //
                // Coalescing: concurrent same-kind jobs queued behind the
                // running one merge into one ragged weight-amortized encode
                // (the tier-1 serving shape - N concurrent callers). Each
                // request still gets exactly its own slice of the results;
                // per-sequence math is batch-composition-independent. Merges
                // cap at a 64k-row chunk (~98% of the big-batch engine rate)
                // bounded by the model's scratch budget.
                //
                // Pipelining (the continuous-batching move for a prefill-only
                // encoder): a batch is SUBMITTED (all kernels enqueued, no
                // host sync) and the thread goes straight back to assembling
                // the next one from arrivals - the previous batch is
                // COLLECTED (event-gated side-stream readback + CPU head +
                // replies) only after the next submit, so the GPU never
                // idles on readback, rerank dots, reply serialization, or
                // client turnaround. Requests that arrive mid-batch join the
                // next chunk, one submit behind - the request-response
                // analog of joining at the next scheduler step.
                let coalesce_row_budget = model.coalesce_row_budget().min(65_536);
                let mut pending: std::collections::VecDeque<EncodeJob> =
                    std::collections::VecDeque::new();
                // Burst capture: after a merged batch's replies, its clients
                // all turn around at once; if the GPU is idle and only one
                // job is here, wait briefly for the burst (20 ms for a second
                // job, 5 ms trailing per arrival) instead of launching solo.
                // Single-client traffic never merged, so it never waits; with
                // a batch in flight the GPU is busy, so never wait either -
                // submit immediately and let the pipeline absorb stragglers.
                // The windows are the backend's call (`burst_windows`); the
                // default is the 20 / 5 ms above.
                let mut prev_merged = false;
                let windows = model.burst_windows();
                let burst = |merged: bool, got_any: bool| -> Option<std::time::Duration> {
                    let (first, trailing) = windows?;
                    if !merged {
                        None
                    } else if got_any {
                        Some(trailing)
                    } else {
                        Some(first)
                    }
                };

                // A submitted-but-uncollected batch: the pending GPU work
                // plus each merged request's sequence count and reply slot.
                enum Inflight<P> {
                    Embed(
                        P,
                        Vec<(usize, oneshot::Sender<Result<Vec<Vec<f32>>, String>>)>,
                    ),
                    Rerank(
                        P,
                        u32,
                        u32,
                        Vec<(usize, oneshot::Sender<Result<Vec<f32>, String>>)>,
                    ),
                }
                let collect = |model: &mut B, inf: Inflight<B::Pending>| match inf {
                    Inflight::Embed(p, parts) => match model.embed_collect(&p) {
                        Ok(mut vecs) => {
                            for (n, reply) in parts {
                                let rest = vecs.split_off(n);
                                let _ = reply.send(Ok(std::mem::replace(&mut vecs, rest)));
                            }
                        }
                        Err(e) => {
                            let e = e.to_string();
                            for (_, reply) in parts {
                                let _ = reply.send(Err(e.clone()));
                            }
                        }
                    },
                    Inflight::Rerank(p, yes, no, parts) => {
                        match model.rerank_collect(&p, yes, no) {
                            Ok(mut scores) => {
                                for (n, reply) in parts {
                                    let rest = scores.split_off(n);
                                    let _ = reply.send(Ok(std::mem::replace(&mut scores, rest)));
                                }
                            }
                            Err(e) => {
                                let e = e.to_string();
                                for (_, reply) in parts {
                                    let _ = reply.send(Err(e.clone()));
                                }
                            }
                        }
                    }
                };
                let ready = |model: &B, inf: &Inflight<B::Pending>| match inf {
                    Inflight::Embed(p, _) | Inflight::Rerank(p, _, _, _) => model.pool_ready(p),
                };

                // Dual-lane pipelining: batches alternate across two CUDA
                // streams so their kernels overlap on the GPU (wave-starved
                // small batches backfill each other's idle SMs). Per-lane
                // stream ordering makes enqueue-ahead safe, and each lane's
                // pool plane is double-buffered - so the cap is two
                // uncollected batches per lane, collected FIFO.
                let lanes = model.lanes();
                let trace = paddock_models::dev_var_os!("PADDOCK_TRACE_BATCH").is_some();
                let cap = model.inflight_capacity();
                assert!(lanes > 0 && cap >= lanes);
                let mut lane_seq = 0usize;
                let mut inflight: std::collections::VecDeque<Inflight<B::Pending>> =
                    std::collections::VecDeque::new();
                loop {
                    // Opportunistically flush finished batches so replies
                    // never wait on scheduling.
                    while inflight.front().is_some_and(|inf| ready(&model, inf)) {
                        let inf = inflight.pop_front().expect("front");
                        collect(&mut model, inf);
                        stamp(&model);
                    }
                    // Next job: block only when nothing is in flight. With
                    // batches in flight, poll - an arrival submits ahead;
                    // the oldest batch finishing collects it promptly.
                    let job = if let Some(j) = pending.pop_front() {
                        Some(j)
                    } else if inflight.is_empty() {
                        if let Some(timeout) = model.idle_reclaim_after() {
                            match rx.recv_timeout(timeout) {
                                Ok(j) => pending.push_back(j),
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    model.reclaim_idle();
                                    stamp(&model);
                                }
                                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                            continue;
                        }
                        match rx.recv() {
                            Ok(j) => Some(j),
                            Err(_) => break,
                        }
                    } else {
                        let mut got = None;
                        loop {
                            match rx.recv_timeout(std::time::Duration::from_micros(500)) {
                                Ok(j) => {
                                    got = Some(j);
                                    break;
                                }
                                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                    if ready(&model, inflight.front().expect("inflight")) {
                                        break; // reply now; nothing to submit ahead
                                    }
                                }
                                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        }
                        got
                    };
                    let Some(job) = job else {
                        if let Some(inf) = inflight.pop_front() {
                            collect(&mut model, inf);
                            stamp(&model);
                        }
                        continue;
                    };
                    match job {
                        EncodeJob::Embed {
                            seqs,
                            media,
                            dimensions,
                            reply,
                        } => {
                            if reply.is_closed() {
                                continue;
                            }
                            if let Err(e) = model
                                .validate_media(&seqs, &media)
                                .and_then(|()| model.validate_dimensions(dimensions))
                            {
                                let _ = reply.send(Err(e));
                                continue;
                            }
                            let mut parts = vec![(seqs, media, reply)];
                            let mut rows: usize = parts[0].0.iter().map(Vec::len).sum();
                            // Merge window: with a batch in flight, waiting
                            // for merge partners is free - the GPU is busy -
                            // so wait until the budget fills or the inflight
                            // completes (then feed the GPU immediately).
                            // Idle GPU: only the short burst windows.
                            while rows < coalesce_row_budget {
                                let got = match rx.try_recv() {
                                    Ok(j) => Some(j),
                                    // any batch still running: the GPU is
                                    // busy, so waiting for merge partners is
                                    // free - hold until the oldest finishes
                                    // (instant singleton submits leaked lone
                                    // requests onto lanes and starved the
                                    // burst split)
                                    Err(_) if !inflight.is_empty() => {
                                        if ready(&model, inflight.front().expect("front")) {
                                            None
                                        } else {
                                            match rx
                                                .recv_timeout(std::time::Duration::from_micros(500))
                                            {
                                                Ok(j) => Some(j),
                                                Err(_) => continue,
                                            }
                                        }
                                    }
                                    Err(_) => match burst(prev_merged, parts.len() > 1) {
                                        Some(t) => rx.recv_timeout(t).ok(),
                                        None => None,
                                    },
                                };
                                match got {
                                    Some(EncodeJob::Embed {
                                        seqs,
                                        media,
                                        dimensions: next_dimensions,
                                        reply,
                                    }) if next_dimensions == dimensions => {
                                        if reply.is_closed() {
                                            continue;
                                        }
                                        if let Err(e) = model.validate_media(&seqs, &media) {
                                            let _ = reply.send(Err(e));
                                            continue;
                                        }
                                        let extra = seqs.iter().map(Vec::len).sum::<usize>();
                                        if extra > coalesce_row_budget.saturating_sub(rows) {
                                            pending.push_back(EncodeJob::Embed {
                                                seqs,
                                                media,
                                                dimensions,
                                                reply,
                                            });
                                            break;
                                        }
                                        rows += seqs.iter().map(Vec::len).sum::<usize>();
                                        parts.push((seqs, media, reply));
                                    }
                                    Some(other) => {
                                        pending.push_back(other);
                                        break;
                                    }
                                    None => break,
                                }
                            }
                            prev_merged = parts.len() > 1;
                            // Burst split: when lanes are idle and one
                            // merge swallowed the whole client burst (the
                            // C=16,B=8 shape: all requests fit one batch),
                            // a single submit serializes the population on
                            // one lane. Split rows-balanced across the free
                            // lanes so the pieces overlap on the GPU.
                            let free = lanes.saturating_sub(inflight.len()).max(1);
                            let ways = free.min(parts.len());
                            let halves: Vec<Vec<(Vec<Vec<u32>>, _, _)>> = if ways >= 2 {
                                let total: usize = parts
                                    .iter()
                                    .map(|(s, _, _)| s.iter().map(Vec::len).sum::<usize>())
                                    .sum();
                                let mut out = Vec::with_capacity(ways);
                                let mut acc = 0usize;
                                let mut taken = 0usize;
                                let mut cur: Vec<(Vec<Vec<u32>>, _, _)> = Vec::new();
                                let n_parts = parts.len();
                                for (i, part) in parts.into_iter().enumerate() {
                                    acc += part.0.iter().map(Vec::len).sum::<usize>();
                                    cur.push(part);
                                    let want = (taken + 1) * total / ways;
                                    // keep at least one part per remaining way
                                    if acc >= want
                                        && taken + 1 < ways
                                        && n_parts - i > ways - taken - 1
                                    {
                                        out.push(std::mem::take(&mut cur));
                                        taken += 1;
                                    }
                                }
                                if !cur.is_empty() {
                                    out.push(cur);
                                }
                                out
                            } else {
                                vec![parts]
                            };
                            for parts in halves {
                                let mut all: Vec<Vec<u32>> = Vec::new();
                                let mut all_media = Vec::new();
                                let mut counts: Vec<(usize, _)> = Vec::with_capacity(parts.len());
                                for (s, m, r) in parts {
                                    counts.push((s.len(), r));
                                    all.extend(s);
                                    all_media.extend(m);
                                }
                                if inflight.len() >= cap {
                                    let inf = inflight.pop_front().expect("front");
                                    collect(&mut model, inf);
                                }
                                let lane = lane_seq % lanes;
                                lane_seq += 1;
                                match model.embed_submit_media(&all, &all_media, lane, dimensions) {
                                    Ok(p) => inflight.push_back(Inflight::Embed(p, counts)),
                                    Err(e) => {
                                        let e = e.to_string();
                                        for (_, reply) in counts {
                                            let _ = reply.send(Err(e.clone()));
                                        }
                                    }
                                }
                            }
                        }
                        EncodeJob::Rerank {
                            seqs,
                            yes,
                            no,
                            reply,
                        } => {
                            if reply.is_closed() {
                                continue;
                            }
                            if let Err(e) = model.validate(&seqs) {
                                let _ = reply.send(Err(e));
                                continue;
                            }
                            let mut parts = vec![(seqs, reply)];
                            let mut rows: usize = parts[0].0.iter().map(Vec::len).sum();
                            // same free-while-busy merge window as the embed arm
                            while rows < coalesce_row_budget {
                                let got = match rx.try_recv() {
                                    Ok(j) => Some(j),
                                    // any batch still running: the GPU is
                                    // busy, so waiting for merge partners is
                                    // free - hold until the oldest finishes
                                    // (instant singleton submits leaked lone
                                    // requests onto lanes and starved the
                                    // burst split)
                                    Err(_) if !inflight.is_empty() => {
                                        if ready(&model, inflight.front().expect("front")) {
                                            None
                                        } else {
                                            match rx
                                                .recv_timeout(std::time::Duration::from_micros(500))
                                            {
                                                Ok(j) => Some(j),
                                                Err(_) => continue,
                                            }
                                        }
                                    }
                                    Err(_) => match burst(prev_merged, parts.len() > 1) {
                                        Some(t) => rx.recv_timeout(t).ok(),
                                        None => None,
                                    },
                                };
                                match got {
                                    // only merge jobs scoring the same head pair
                                    Some(EncodeJob::Rerank {
                                        seqs,
                                        yes: y2,
                                        no: n2,
                                        reply,
                                    }) if (y2, n2) == (yes, no) => {
                                        if reply.is_closed() {
                                            continue;
                                        }
                                        if let Err(e) = model.validate(&seqs) {
                                            let _ = reply.send(Err(e));
                                            continue;
                                        }
                                        let extra = seqs.iter().map(Vec::len).sum::<usize>();
                                        if extra > coalesce_row_budget.saturating_sub(rows) {
                                            pending.push_back(EncodeJob::Rerank {
                                                seqs,
                                                yes,
                                                no,
                                                reply,
                                            });
                                            break;
                                        }
                                        rows += seqs.iter().map(Vec::len).sum::<usize>();
                                        parts.push((seqs, reply));
                                    }
                                    Some(other) => {
                                        pending.push_back(other);
                                        break;
                                    }
                                    None => break,
                                }
                            }
                            prev_merged = parts.len() > 1;
                            // burst split - see the embed arm
                            let free = lanes.saturating_sub(inflight.len()).max(1);
                            let ways = free.min(parts.len());
                            let halves: Vec<Vec<(Vec<Vec<u32>>, _)>> = if ways >= 2 {
                                let total: usize = parts
                                    .iter()
                                    .map(|(s, _)| s.iter().map(Vec::len).sum::<usize>())
                                    .sum();
                                let mut out = Vec::with_capacity(ways);
                                let mut acc = 0usize;
                                let mut taken = 0usize;
                                let mut cur: Vec<(Vec<Vec<u32>>, _)> = Vec::new();
                                let n_parts = parts.len();
                                for (i, part) in parts.into_iter().enumerate() {
                                    acc += part.0.iter().map(Vec::len).sum::<usize>();
                                    cur.push(part);
                                    let want = (taken + 1) * total / ways;
                                    if acc >= want
                                        && taken + 1 < ways
                                        && n_parts - i > ways - taken - 1
                                    {
                                        out.push(std::mem::take(&mut cur));
                                        taken += 1;
                                    }
                                }
                                if !cur.is_empty() {
                                    out.push(cur);
                                }
                                out
                            } else {
                                vec![parts]
                            };
                            for parts in halves {
                                let all: Vec<Vec<u32>> =
                                    parts.iter().flat_map(|(s, _)| s.iter().cloned()).collect();
                                let counts: Vec<(usize, _)> =
                                    parts.into_iter().map(|(s, r)| (s.len(), r)).collect();
                                if inflight.len() >= cap {
                                    let inf = inflight.pop_front().expect("front");
                                    collect(&mut model, inf);
                                }
                                let lane = lane_seq % lanes;
                                lane_seq += 1;
                                if trace {
                                    tracing::info!(
                                        "  [batch] lane {lane} docs {} parts {} inflight {}",
                                        all.len(),
                                        counts.len(),
                                        inflight.len()
                                    );
                                }
                                match model.rerank_submit(&all, yes, no, lane) {
                                    Ok(p) => {
                                        inflight.push_back(Inflight::Rerank(p, yes, no, counts))
                                    }
                                    Err(e) => {
                                        let e = e.to_string();
                                        for (_, reply) in counts {
                                            let _ = reply.send(Err(e.clone()));
                                        }
                                    }
                                }
                            }
                        }
                        EncodeJob::Calibrate {
                            seqs,
                            n_docs,
                            rel,
                            reply,
                        } => {
                            // calibration / profile ops mutate planes - the
                            // pipeline must drain first
                            while let Some(inf) = inflight.pop_front() {
                                collect(&mut model, inf);
                            }
                            let out = model
                                .calibrate_bs(&seqs, n_docs, &rel)
                                .map_err(|e| e.to_string());
                            stamp(&model); // planes changed -> weight total moved
                            let _ = reply.send(out);
                        }
                        EncodeJob::CalibrateRerank {
                            seqs,
                            yes,
                            no,
                            group,
                            rel,
                            reply,
                        } => {
                            // calibration / profile ops mutate planes - the
                            // pipeline must drain first
                            while let Some(inf) = inflight.pop_front() {
                                collect(&mut model, inf);
                            }
                            let out = model
                                .calibrate_bs_rerank(&seqs, yes, no, group, &rel)
                                .map_err(|e| e.to_string());
                            stamp(&model); // planes changed -> weight total moved
                            let _ = reply.send(out);
                        }
                        EncodeJob::ApplyProfile {
                            profile,
                            smooth,
                            reply,
                        } => {
                            // calibration / profile ops mutate planes - the
                            // pipeline must drain first
                            while let Some(inf) = inflight.pop_front() {
                                collect(&mut model, inf);
                            }
                            let out = (|| {
                                if let Some(bytes) = &smooth
                                    && !model.import_smooth(bytes).map_err(|e| e.to_string())?
                                {
                                    return Ok(false);
                                }
                                model.apply_bs_profile(&profile).map_err(|e| e.to_string())
                            })();
                            stamp(&model); // planes changed -> weight total moved
                            let _ = reply.send(out);
                        }
                        EncodeJob::ExportSmooth { reply } => {
                            // calibration / profile ops mutate planes - the
                            // pipeline must drain first
                            while let Some(inf) = inflight.pop_front() {
                                collect(&mut model, inf);
                            }
                            let out = model.export_smooth().map_err(|e| e.to_string());
                            let _ = reply.send(out);
                        }
                    }
                }
            })
            .map_err(|e| format!("failed to spawn encoder thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok((block_scale_calibration, media))) => Ok(Self {
                tx,
                block_scale_calibration,
                media,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("encoder thread died during startup".into()),
        }
    }

    /// Whether the backend embeds pictures (an attached picture tower).
    pub fn serves_images(&self) -> bool {
        self.media.0
    }

    /// Whether the backend embeds audio clips (an attached audio tower).
    pub fn serves_audio(&self) -> bool {
        self.media.1
    }

    /// A frame sequence is not a still-image album: it has its own processor.
    pub fn serves_video(&self) -> bool {
        self.media.2
    }

    /// Device-specific quality calibration is not a generic encoder feature.
    pub fn block_scale_calibration(&self) -> bool {
        self.block_scale_calibration
    }

    /// Embed a batch of tokenized sequences in one weight-amortized pass.
    pub async fn embed(&self, seqs: Vec<Vec<u32>>) -> Result<Vec<Vec<f32>>, String> {
        self.embed_dimensions(seqs, None).await
    }

    /// MRL truncation and re-normalization belong to the GPU backend. Jobs
    /// with different dimensions cannot be coalesced into one output plane.
    pub async fn embed_dimensions(
        &self,
        seqs: Vec<Vec<u32>>,
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let media = seqs.iter().map(|_| Vec::new()).collect();
        self.embed_media(seqs, media, dimensions).await
    }

    /// [`Self::embed_dimensions`] with pictures and audio clips: `media[i]`
    /// fills sequence i's placeholder runs in order (see
    /// [`EncodeJob::Embed`]). Only a backend with the towers accepts any.
    pub async fn embed_media(
        &self,
        seqs: Vec<Vec<u32>>,
        media: Vec<Vec<crate::service::MmChunk>>,
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        if media.len() != seqs.len() {
            return Err("one media list per sequence".into());
        }
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EncodeJob::Embed {
                seqs,
                media,
                dimensions,
                reply,
            })
            .map_err(|_| "encoder thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "encoder dropped the reply".to_owned())?
    }

    /// Run the load-time block-scale calibration (see [`EncodeJob::Calibrate`]).
    pub async fn calibrate(
        &self,
        seqs: Vec<Vec<u32>>,
        n_docs: usize,
        rel: Vec<usize>,
    ) -> Result<&'static str, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EncodeJob::Calibrate {
                seqs,
                n_docs,
                rel,
                reply,
            })
            .map_err(|_| "encoder thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "encoder dropped the reply".to_owned())?
    }

    /// Run the reranker-metric calibration (see [`EncodeJob::CalibrateRerank`]).
    pub async fn calibrate_rerank(
        &self,
        seqs: Vec<Vec<u32>>,
        yes: u32,
        no: u32,
        group: usize,
        rel: Vec<usize>,
    ) -> Result<&'static str, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EncodeJob::CalibrateRerank {
                seqs,
                yes,
                no,
                group,
                rel,
                reply,
            })
            .map_err(|_| "encoder thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "encoder dropped the reply".to_owned())?
    }

    /// Apply a cached block-scale profile (see [`EncodeJob::ApplyProfile`]).
    pub async fn apply_profile(
        &self,
        profile: String,
        smooth: Option<Vec<u8>>,
    ) -> Result<bool, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EncodeJob::ApplyProfile {
                profile,
                smooth,
                reply,
            })
            .map_err(|_| "encoder thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "encoder dropped the reply".to_owned())?
    }

    /// Export the calibrated SmoothQuant s-vectors (see
    /// [`EncodeJob::ExportSmooth`]).
    pub async fn export_smooth(&self) -> Result<Option<Vec<u8>>, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EncodeJob::ExportSmooth { reply })
            .map_err(|_| "encoder thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "encoder dropped the reply".to_owned())?
    }

    /// Score a batch of query-document sequences (yes/no relevance judge).
    pub async fn rerank(&self, seqs: Vec<Vec<u32>>, yes: u32, no: u32) -> Result<Vec<f32>, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EncodeJob::Rerank {
                seqs,
                yes,
                no,
                reply,
            })
            .map_err(|_| "encoder thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "encoder dropped the reply".to_owned())?
    }
}
