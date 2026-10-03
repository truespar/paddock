//! Bounded diarization frontend and arrival-order speaker state. Independent
//! from ASR, without VAD trimming, so silence and overlap retain their times.
//! Streaming algorithm adapted from MLX Audio; see packs/metal/diarization.NOTICE.md.
use crate::audio::dsp::{FftPlan, mel_power_frame_buffered};
use paddock_models::diarization::{
    CACHE, HIDDEN, HOP, MEL, Preset, SAMPLE_RATE, SPEAKERS, STACK, Segment, cache_selection,
    segments,
};
use std::sync::{
    Arc, Weak,
    atomic::Ordering::Relaxed,
    mpsc::{SyncSender, channel, sync_channel},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

pub const MAX_SAMPLES: usize = SAMPLE_RATE * 600;
pub struct AudioWindow<'a> {
    pub audio: &'a [f32],
    pub offset: usize,
    pub total: usize,
    pub start: usize,
    pub count: usize,
}
impl AudioWindow<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.audio.len() > SAMPLE_RATE * 64
            || self.total > MAX_SAMPLES
            || self.offset.checked_add(self.audio.len()) != Some(self.total)
            || self.count == 0
            || self.count > 3040
            || self.start > self.total / HOP
            || self.offset > (self.start * HOP).saturating_sub(257)
            || self.audio.iter().any(|v| !v.is_finite())
        {
            return Err("invalid diarization PCM window".into());
        }
        Ok(())
    }
}
pub trait Backend {
    fn pre_encode(&mut self, features: &[f32]) -> Result<Vec<f32>, String>;
    fn encode_audio(&mut self, fe: &Frontend, w: AudioWindow<'_>) -> Result<Vec<f32>, String> {
        w.validate()?;
        self.pre_encode(&fe.features(w.audio, w.offset, w.total, w.start, w.count))
    }
    fn predict(
        &mut self,
        embeddings: &[f32],
        valid: usize,
    ) -> Result<(Vec<[f32; SPEAKERS]>, f64), String>;
    fn frontend(&self) -> Result<Frontend, String>;
    fn silence(&self) -> &[f32];
    fn weight_bytes(&self) -> u64;
    fn workspace_bytes(&self) -> u64;
}
pub struct Frontend {
    window: Vec<f64>,
    fb: Vec<f64>,
    fft: FftPlan,
}
impl Frontend {
    pub fn new(window: &[f32], fb: &[f32]) -> Result<Self, String> {
        if window.len() != 400
            || fb.len() != MEL * 257
            || window.iter().chain(fb).any(|v| !v.is_finite())
        {
            return Err("invalid diarization filter buffers".into());
        }
        let mut padded = vec![0.; 512];
        for (i, &v) in window.iter().enumerate() {
            padded[56 + i] = v as f64;
        }
        Ok(Self {
            window: padded,
            fb: fb.iter().map(|&v| v as f64).collect(),
            fft: FftPlan::new(512),
        })
    }
    /// Frame-major; padding frames are zero features, not log(silence).
    pub fn features(
        &self,
        audio: &[f32],
        offset: usize,
        total: usize,
        start: usize,
        count: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.; count.div_ceil(STACK) * STACK * MEL];
        let mut frame = [0f64; 512];
        let mut energy = [0f64; MEL];
        let mut scratch = [0f64; 1281];
        let sample = |i: i64| -> f32 {
            if i < 0 || i as usize >= total {
                0.
            } else {
                audio[i as usize - offset]
            }
        };
        for j in 0..count {
            let id = start + j;
            if id >= total / HOP {
                continue;
            }
            for (k, v) in frame.iter_mut().enumerate() {
                let p = (id * HOP + k) as i64 - 256;
                *v = if p < 0 || p as usize >= total {
                    0.
                } else {
                    (sample(p) - 0.97f32 * sample(p - 1)) as f64
                };
            }
            if frame.iter().all(|&v| v == 0.) {
                // Exact silence, not VAD: retain every frame and its log guard.
                energy.fill(0.);
            } else {
                mel_power_frame_buffered(
                    &frame,
                    &self.window,
                    &self.fb,
                    &self.fft,
                    &mut scratch,
                    &mut energy,
                );
            }
            for (m, e) in energy.iter().enumerate() {
                out[j * MEL + m] = (*e + 2f64.powi(-24)).ln() as f32;
            }
        }
        out
    }
}
pub struct Stream {
    preset: Preset,
    cache: Vec<f32>,
    cache_probs: Vec<[f32; 8]>,
    fifo: Vec<f32>,
    compressed: bool,
    pub frames: usize,
    received: usize,
    offset: usize,
    audio: Vec<f32>,
    finished: bool,
    pub gpu_seconds: f64,
}
impl Stream {
    pub fn new(preset: Preset) -> Self {
        Self {
            preset,
            cache: Vec::new(),
            cache_probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
            frames: 0,
            received: 0,
            offset: 0,
            audio: Vec::new(),
            finished: false,
            gpu_seconds: 0.,
        }
    }
    pub fn retained_samples(&self) -> usize {
        self.audio.len()
    }
    pub fn retained_rows(&self) -> usize {
        (self.cache.len() + self.fifo.len()) / HIDDEN
    }
    // Arrival buffering does not need the device. Share this exact validation
    // with synchronous callers and Session's lookahead-only fast path.
    fn buffer(&mut self, pcm: &[f32]) -> Result<(), String> {
        if self.finished {
            return Err("diarization stream is finished".into());
        }
        if pcm.len() > SAMPLE_RATE * 32 || pcm.iter().any(|v| !v.is_finite()) {
            return Err("expected <=32 seconds of finite mono PCM".into());
        }
        if self
            .received
            .checked_add(pcm.len())
            .is_none_or(|n| n > MAX_SAMPLES)
        {
            return Err("diarization duration exceeds 600 seconds".into());
        }
        self.received += pcm.len();
        self.audio.extend_from_slice(pcm);
        Ok(())
    }
    fn ready(&self) -> bool {
        let g = self.preset.geometry();
        let needed = (self.frames + (g.chunk + g.right) * STACK - 1) * HOP + 256;
        self.received / HOP > self.frames && self.received >= needed
    }
    /// Feed <=32 seconds of finite mono 16-kHz PCM. State must be discarded
    /// after a backend error. Final drains lookahead once; repeated flush fails.
    pub fn feed<B: Backend>(
        &mut self,
        b: &mut B,
        fe: &Frontend,
        pcm: &[f32],
        final_chunk: bool,
    ) -> Result<Vec<[f32; 8]>, String> {
        self.buffer(pcm)?;
        self.process(b, fe, final_chunk)
    }
    /// Drain already validated/buffered arrivals. Only the owning device
    /// thread executes this for async sessions; final flush always goes there.
    fn process<B: Backend>(
        &mut self,
        b: &mut B,
        fe: &Frontend,
        final_chunk: bool,
    ) -> Result<Vec<[f32; 8]>, String> {
        let g = self.preset.geometry();
        let central = g.chunk * STACK;
        let right = g.right * STACK;
        let mut out = Vec::new();
        while self.received / HOP > self.frames {
            let available = self.received / HOP - self.frames;
            if !final_chunk && !self.ready() {
                break;
            }
            let n = central.min(available);
            let count = if final_chunk {
                (central + right).min((self.received / HOP + 1).div_ceil(16) * 16 - self.frames)
            } else {
                central + right
            };
            let chunk = b.encode_audio(
                fe,
                AudioWindow {
                    audio: &self.audio,
                    offset: self.offset,
                    total: self.received,
                    start: self.frames,
                    count,
                },
            )?;
            let cache_len = self.cache.len() / HIDDEN;
            let fifo_len = self.fifo.len() / HIDDEN;
            let start = cache_len + fifo_len;
            let mut combined = Vec::with_capacity(self.cache.len() + self.fifo.len() + chunk.len());
            combined.extend_from_slice(&self.cache);
            combined.extend_from_slice(&self.fifo);
            combined.extend_from_slice(&chunk);
            let (high, seconds) =
                b.predict(&combined, start + count.min(available).div_ceil(STACK))?;
            self.gpu_seconds += seconds;
            if high.len() != combined.len() / HIDDEN * STACK
                || high
                    .iter()
                    .flatten()
                    .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err("invalid diarization backend output".into());
            }
            let low: Vec<[f32; 8]> = high
                .as_chunks::<STACK>()
                .0
                .iter()
                .map(|rows| {
                    let mut p = [0.; 8];
                    for r in rows {
                        for (s, v) in p.iter_mut().enumerate() {
                            *v += r[s] / STACK as f32;
                        }
                    }
                    p
                })
                .collect();
            out.extend_from_slice(&high[start * STACK..start * STACK + n]);
            let nr = n.div_ceil(STACK);
            self.fifo.extend_from_slice(&chunk[..nr * HIDDEN]);
            let fifo_probs = &low[cache_len..start + nr];
            if self.fifo.len() / HIDDEN > g.fifo {
                let pop =
                    (self.fifo.len() / HIDDEN).min(g.update.max(self.fifo.len() / HIDDEN - g.fifo));
                self.cache.extend_from_slice(&self.fifo[..pop * HIDDEN]);
                if !self.compressed {
                    self.cache_probs = low[..cache_len].to_vec();
                }
                self.cache_probs.extend_from_slice(&fifo_probs[..pop]);
                self.fifo.drain(..pop * HIDDEN);
                if self.cache.len() / HIDDEN > CACHE {
                    let picks = cache_selection(&self.cache_probs)?;
                    let mut cache = Vec::with_capacity(CACHE * HIDDEN);
                    let mut probs = Vec::with_capacity(CACHE);
                    if b.silence().len() != HIDDEN {
                        return Err("invalid learned silence embedding".into());
                    }
                    for pick in picks {
                        match pick {
                            Some(i) => {
                                cache.extend_from_slice(&self.cache[i * HIDDEN..(i + 1) * HIDDEN]);
                                probs.push(self.cache_probs[i]);
                            }
                            None => {
                                cache.extend_from_slice(b.silence());
                                probs.push([0.; 8]);
                            }
                        }
                    }
                    self.cache = cache;
                    self.cache_probs = probs;
                    self.compressed = true;
                }
            }
            self.frames += n;
            let keep = (self.frames * HOP).saturating_sub(257);
            self.audio.drain(..keep - self.offset);
            self.offset = keep;
        }
        if final_chunk {
            self.finished = true;
            self.audio.clear();
        }
        Ok(out)
    }
}
pub struct Output {
    pub segments: Vec<Segment>,
    pub frames: usize,
    pub gpu_seconds: f64,
}
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("diarization queue is full; retry shortly")]
    Busy,
    #[error("diarization worker is unavailable")]
    Unavailable,
    #[error("{0}")]
    Backend(String),
}
struct Job {
    stream: Stream,
    final_chunk: bool,
    reply: oneshot::Sender<Result<(Stream, Vec<[f32; SPEAKERS]>), String>>,
    // Keep the reservation until even an abandoned in-flight GPU step ends.
    _permit: Arc<OwnedSemaphorePermit>,
}
pub struct Diarizer {
    tx: SyncSender<Job>,
    metrics: Arc<crate::metrics::EngineMetrics>,
    sessions: Arc<Semaphore>,
    worker_alive: Weak<()>,
}

/// Idle sockets own only bounded host state, never the worker or Metal device.
/// A step moves that state to the owning thread and returns it. Dropping an
/// in-flight future makes this session unusable rather than replaying audio.
pub struct Session {
    tx: SyncSender<Job>,
    stream: Option<Stream>,
    permit: Arc<OwnedSemaphorePermit>,
    worker_alive: Weak<()>,
}
pub struct Batch {
    pub probabilities: Vec<[f32; SPEAKERS]>,
    pub frames: usize,
    pub received_samples: usize,
    pub gpu_seconds: f64,
}
impl Session {
    pub async fn feed(&mut self, pcm: Vec<f32>, final_chunk: bool) -> Result<Batch, Error> {
        let state = self.stream.as_mut().ok_or(Error::Unavailable)?;
        if state.finished
            || pcm.len() > SAMPLE_RATE
            || pcm.iter().any(|v| !v.is_finite())
            || state
                .received
                .checked_add(pcm.len())
                .is_none_or(|n| n > MAX_SAMPLES || (final_chunk && n < HOP))
        {
            return Err(Error::Invalid("expected <=1 second of finite mono 16-kHz PCM per step, 0.01–600 seconds total, and one final flush".into()));
        }
        if self.worker_alive.strong_count() == 0 {
            return Err(Error::Unavailable);
        }
        state.buffer(&pcm).map_err(Error::Invalid)?;
        // No inference is ready until the preset's exact lookahead boundary.
        // Acknowledge these bounded arrivals without handing all session state
        // to the GPU worker or waiting behind another session's inference.
        if !final_chunk && !state.ready() {
            return Ok(Batch {
                probabilities: Vec::new(),
                frames: state.frames,
                received_samples: state.received,
                gpu_seconds: state.gpu_seconds,
            });
        }
        let stream = self.stream.take().ok_or(Error::Unavailable)?;
        let (reply, rx) = oneshot::channel();
        self.tx
            .try_send(Job {
                stream,
                final_chunk,
                reply,
                _permit: self.permit.clone(),
            })
            .map_err(|e| match e {
                std::sync::mpsc::TrySendError::Full(_) => Error::Busy,
                _ => Error::Unavailable,
            })?;
        let (stream, probabilities) = rx
            .await
            .map_err(|_| Error::Unavailable)?
            .map_err(Error::Backend)?;
        let batch = Batch {
            probabilities,
            frames: stream.frames,
            received_samples: stream.received,
            gpu_seconds: stream.gpu_seconds,
        };
        self.stream = Some(stream);
        Ok(batch)
    }
}
impl Diarizer {
    pub fn spawn<F, B>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<B, String> + Send + 'static,
        B: Backend + 'static,
    {
        let (tx, rx) = sync_channel::<Job>(2);
        let (ready, wait) = channel();
        let metrics = Arc::new(crate::metrics::EngineMetrics::default());
        let m = Arc::clone(&metrics);
        // Only the worker owns the strong reference. Even a lookahead-only
        // feed refuses a dead worker, including unwinding or failed startup.
        let alive = Arc::new(());
        let worker_alive = Arc::downgrade(&alive);
        std::thread::Builder::new()
            .name("paddock-diarization".into())
            .spawn(move || {
                let _alive = alive;
                let (mut b, fe) = match build().and_then(|b| b.frontend().map(|fe| (b, fe))) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                m.weights_mem_bytes.store(b.weight_bytes(), Relaxed);
                m.model_mem_bytes
                    .store(b.weight_bytes() + b.workspace_bytes(), Relaxed);
                if ready.send(Ok(())).is_err() {
                    return;
                }
                for mut job in rx {
                    if job.reply.is_closed() {
                        continue;
                    }
                    m.active_slots.store(1, Relaxed);
                    m.phase.store(crate::metrics::PHASE_PREFILL, Relaxed);
                    let result = job
                        .stream
                        .process(&mut b, &fe, job.final_chunk)
                        .map(|probs| (job.stream, probs));
                    m.active_slots.store(0, Relaxed);
                    m.phase.store(crate::metrics::PHASE_IDLE, Relaxed);
                    let _ = job.reply.send(result);
                }
            })
            .map_err(|e| e.to_string())?;
        wait.recv()
            .map_err(|_| "diarization worker failed at startup")??;
        Ok(Self {
            tx,
            metrics,
            sessions: Arc::new(Semaphore::new(2)),
            worker_alive,
        })
    }
    pub fn metrics(&self) -> Arc<crate::metrics::EngineMetrics> {
        Arc::clone(&self.metrics)
    }
    pub fn session(&self, preset: Preset) -> Result<Session, Error> {
        let permit = self
            .sessions
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Busy)?;
        Ok(Session {
            tx: self.tx.clone(),
            stream: Some(Stream::new(preset)),
            permit: Arc::new(permit),
            worker_alive: self.worker_alive.clone(),
        })
    }
    fn file_quantum(&self, preset: Preset) -> usize {
        if self.sessions.available_permits() == 0 {
            (preset.geometry().chunk * STACK * HOP).min(SAMPLE_RATE)
        } else {
            SAMPLE_RATE
        }
    }
    pub async fn diarize(
        &self,
        pcm: Vec<f32>,
        preset: Preset,
        threshold: f32,
    ) -> Result<Output, Error> {
        if pcm.len() < HOP
            || pcm.len() > MAX_SAMPLES
            || pcm.iter().any(|v| !v.is_finite())
            || !threshold.is_finite()
            || !(0.0..1.0).contains(&threshold)
        {
            return Err(Error::Invalid(
                "expected 0.01–600 seconds of finite mono 16-kHz PCM and threshold in [0,1)".into(),
            ));
        }
        let mut session = self.session(preset)?;
        let mut probs = Vec::new();
        // With a competing reservation, yield after at most one new window
        // (a full second is four ultra-low windows). Otherwise amortize host
        // scheduling over a second. Recheck each turn so new sessions get the
        // short quantum after any already-admitted work; final flush is one
        // bounded drain. Lookahead-only arrivals never occupy the worker.
        let mut offset = 0;
        while offset < pcm.len() {
            let end = (offset + self.file_quantum(preset)).min(pcm.len());
            probs.extend(
                session
                    .feed(pcm[offset..end].to_vec(), false)
                    .await?
                    .probabilities,
            );
            offset = end;
        }
        let last = session.feed(Vec::new(), true).await?;
        probs.extend(last.probabilities);
        Ok(Output {
            segments: segments(&probs, 0, threshold),
            frames: last.frames,
            gpu_seconds: last.gpu_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Fake;
    impl Backend for Fake {
        fn pre_encode(&mut self, f: &[f32]) -> Result<Vec<f32>, String> {
            Ok(vec![0.; f.len() / 1024 * 512])
        }
        fn predict(&mut self, x: &[f32], valid: usize) -> Result<(Vec<[f32; 8]>, f64), String> {
            let mut p = vec![[0.9, 0.8, 0.1, 0., 0., 0., 0., 0.]; x.len() / 512 * 8];
            p[valid * 8..].fill([0.; 8]);
            Ok((p, 0.))
        }
        fn frontend(&self) -> Result<Frontend, String> {
            Frontend::new(&[1.; 400], &vec![0.; 128 * 257])
        }
        fn silence(&self) -> &[f32] {
            &[0.; 512]
        }
        fn weight_bytes(&self) -> u64 {
            0
        }
        fn workspace_bytes(&self) -> u64 {
            0
        }
    }
    #[test]
    fn partition_invariance_flush_and_bounded_history() {
        let pcm = vec![0.; 16000 * 58 + 159];
        let fe = Fake.frontend().unwrap();
        for preset in [
            Preset::Offline,
            Preset::Low,
            Preset::VeryLow,
            Preset::UltraLow,
        ] {
            let mut previous = None;
            for chunk in [16001, 777] {
                let mut stream = Stream::new(preset);
                let mut all = Vec::new();
                for p in pcm.chunks(chunk) {
                    all.extend(stream.feed(&mut Fake, &fe, p, false).unwrap());
                    assert!(stream.retained_rows() <= 528);
                    assert!(stream.retained_samples() <= 16000 * 31);
                }
                all.extend(stream.feed(&mut Fake, &fe, &[], true).unwrap());
                assert_eq!(all.len(), 5800);
                assert_eq!(stream.frames, 5800);
                assert_eq!(stream.retained_samples(), 0);
                assert!(stream.compressed);
                assert_eq!(segments(&all, 0, 0.5).len(), 2);
                assert!(stream.feed(&mut Fake, &fe, &[], true).is_err());
                if let Some(p) = previous {
                    assert_eq!(all, p);
                }
                previous = Some(all);
            }
        }
    }
    #[test]
    fn invalid_input_does_not_mutate_state() {
        let mut s = Stream::new(Preset::Low);
        let fe = Fake.frontend().unwrap();
        assert!(s.feed(&mut Fake, &fe, &[f32::NAN], false).is_err());
        assert_eq!(s.received, 0);
        assert!(
            s.feed(&mut Fake, &fe, &vec![0.; 16000 * 32 + 1], false)
                .is_err()
        );
        assert_eq!(s.received, 0);
        assert!(Frontend::new(&[0.; 399], &[]).is_err());
        assert!(s.feed(&mut Fake, &fe, &[0.; 159], true).unwrap().is_empty());
    }
    #[tokio::test]
    async fn buffered_sessions_match_direct_processing_at_every_arrival() {
        let service = Diarizer::spawn(|| Ok(Fake)).unwrap();
        let fe = Fake.frontend().unwrap();
        let pcm = vec![0.; SAMPLE_RATE * 33 + 159];
        for preset in [
            Preset::Offline,
            Preset::Low,
            Preset::VeryLow,
            Preset::UltraLow,
        ] {
            // Includes small PCM messages and partial STFT frames; crosses
            // every preset's lookahead boundary, then flushes a short tail.
            for chunk in [159, 777, SAMPLE_RATE] {
                let mut direct = Stream::new(preset);
                let mut session = service.session(preset).unwrap();
                for audio in pcm.chunks(chunk) {
                    let expected = direct.feed(&mut Fake, &fe, audio, false).unwrap();
                    let actual = session.feed(audio.to_vec(), false).await.unwrap();
                    assert_eq!(actual.probabilities, expected);
                    assert_eq!(actual.frames, direct.frames);
                    assert_eq!(actual.received_samples, direct.received);
                    assert_eq!(actual.gpu_seconds, direct.gpu_seconds);
                }
                let expected = direct.feed(&mut Fake, &fe, &[], true).unwrap();
                let actual = session.feed(Vec::new(), true).await.unwrap();
                assert_eq!(actual.probabilities, expected);
                assert_eq!(actual.frames, pcm.len() / HOP);
                assert!(matches!(
                    session.feed(Vec::new(), false).await,
                    Err(Error::Invalid(_))
                ));
            }
        }
    }

    #[test]
    fn file_quantum_tracks_competing_reservations() {
        let service = Diarizer::spawn(|| Ok(Fake)).unwrap();
        let _file = service.session(Preset::Low).unwrap();
        for preset in [
            Preset::Offline,
            Preset::Low,
            Preset::VeryLow,
            Preset::UltraLow,
        ] {
            assert_eq!(service.file_quantum(preset), SAMPLE_RATE);
            let other = service.session(Preset::Low).unwrap();
            assert_eq!(
                service.file_quantum(preset),
                (preset.geometry().chunk * STACK * HOP).min(SAMPLE_RATE)
            );
            drop(other);
            assert_eq!(service.file_quantum(preset), SAMPLE_RATE);
        }
    }

    #[test]
    fn file_quanta_bound_regular_steps_without_changing_probabilities() {
        let pcm = vec![0.; SAMPLE_RATE * 33 + 159];
        let fe = Fake.frontend().unwrap();
        for preset in [
            Preset::Offline,
            Preset::Low,
            Preset::VeryLow,
            Preset::UltraLow,
        ] {
            let central = preset.geometry().chunk * STACK;
            let quantum = (central * HOP).min(SAMPLE_RATE);
            let mut expected = None;
            for chunk in [SAMPLE_RATE, quantum] {
                let mut stream = Stream::new(preset);
                let mut actual = Vec::new();
                for audio in pcm.chunks(chunk) {
                    let batch = stream.feed(&mut Fake, &fe, audio, false).unwrap();
                    if chunk == quantum {
                        assert!(
                            batch.len() <= central,
                            "regular file step ran more than one window"
                        );
                    }
                    actual.extend(batch);
                }
                actual.extend(stream.feed(&mut Fake, &fe, &[], true).unwrap());
                assert_eq!(actual.len(), pcm.len() / HOP);
                if let Some(e) = expected {
                    assert_eq!(actual, e);
                }
                expected = Some(actual);
            }
        }
    }

    #[tokio::test]
    async fn lookahead_does_not_accept_audio_after_worker_exit() {
        let (tx, rx) = sync_channel(2);
        drop(rx);
        let mut session = Session {
            tx,
            stream: Some(Stream::new(Preset::Offline)),
            permit: Arc::new(Arc::new(Semaphore::new(1)).try_acquire_owned().unwrap()),
            worker_alive: Weak::new(),
        };
        assert!(matches!(
            session.feed(vec![0.; 160], false).await,
            Err(Error::Unavailable)
        ));
        assert_eq!(session.stream.as_ref().unwrap().received, 0);
    }

    #[tokio::test]
    async fn independent_requests_reset_speaker_state() {
        let service = Diarizer::spawn(|| Ok(Fake)).unwrap();
        assert!(matches!(
            service.diarize(vec![0.; 160], Preset::Low, f32::NAN).await,
            Err(Error::Invalid(_))
        ));
        let (a, b) = tokio::join!(
            service.diarize(vec![0.; 16000], Preset::Low, 0.5),
            service.diarize(vec![0.; 8000], Preset::Offline, 0.5)
        );
        assert_eq!(a.unwrap().frames, 100);
        let b = b.unwrap();
        assert_eq!(b.frames, 50);
        assert_eq!(b.segments[0].end, 0.5);
    }

    #[tokio::test]
    async fn idle_session_does_not_own_worker_and_reservations_are_bounded() {
        let service = Diarizer::spawn(|| Ok(Fake)).unwrap();
        let a = service.session(Preset::Low).unwrap();
        let mut b = service.session(Preset::UltraLow).unwrap();
        assert!(matches!(service.session(Preset::Low), Err(Error::Busy)));
        assert!(matches!(
            b.feed(vec![f32::NAN], false).await,
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            b.feed(vec![0.; 16001], false).await,
            Err(Error::Invalid(_))
        ));
        let out = b.feed(vec![0.; 8000], true).await.unwrap();
        assert_eq!(out.frames, 50);
        assert_eq!(out.received_samples, 8000);
        assert!(b.feed(Vec::new(), true).await.is_err());
        drop(b);
        let out = service
            .diarize(vec![0.; 1600], Preset::Low, 0.5)
            .await
            .unwrap();
        assert_eq!(out.frames, 10);
        drop(a);
        assert_eq!(service.sessions.available_permits(), 2);
    }

    #[tokio::test]
    async fn cancelled_step_cannot_release_capacity_while_device_is_using_it() {
        use std::sync::atomic::AtomicBool;
        struct Slow {
            inner: Fake,
            entered: Arc<AtomicBool>,
            release: Arc<AtomicBool>,
        }
        impl Backend for Slow {
            fn pre_encode(&mut self, f: &[f32]) -> Result<Vec<f32>, String> {
                self.inner.pre_encode(f)
            }
            fn predict(&mut self, x: &[f32], valid: usize) -> Result<(Vec<[f32; 8]>, f64), String> {
                self.entered.store(true, Relaxed);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !self.release.load(Relaxed) {
                    if std::time::Instant::now() > deadline {
                        return Err("test gate timed out".into());
                    }
                    std::thread::yield_now();
                }
                self.inner.predict(x, valid)
            }
            fn frontend(&self) -> Result<Frontend, String> {
                self.inner.frontend()
            }
            fn silence(&self) -> &[f32] {
                self.inner.silence()
            }
            fn weight_bytes(&self) -> u64 {
                0
            }
            fn workspace_bytes(&self) -> u64 {
                0
            }
        }
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let (e, r) = (entered.clone(), release.clone());
        let service = Diarizer::spawn(move || {
            Ok(Slow {
                inner: Fake,
                entered: e,
                release: r,
            })
        })
        .unwrap();
        let mut idle = service.session(Preset::Low).unwrap();
        let mut active = service.session(Preset::UltraLow).unwrap();
        let job = tokio::spawn(async move { active.feed(vec![0.; 16000], true).await });
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !entered.load(Relaxed) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        // The other session is blocked inside predict. Lookahead-only input
        // must still be acknowledged without touching that worker or taking
        // another reservation. The original queued implementation times out.
        let lookahead = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            idle.feed(vec![0.; 640], false),
        )
        .await
        .unwrap()
        .unwrap();
        assert!(lookahead.probabilities.is_empty());
        assert_eq!(lookahead.received_samples, 640);
        assert_eq!(lookahead.frames, 0);
        job.abort();
        let _ = job.await;
        assert!(matches!(service.session(Preset::Low), Err(Error::Busy)));
        release.store(true, Relaxed);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while service.sessions.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(idle);
        assert_eq!(service.sessions.available_permits(), 2);
    }
}
