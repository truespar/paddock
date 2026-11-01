//! Native TTS request lifecycle and reproducible sampling. Backend execution
//! stays on its owning worker; bounded channels stop slow clients accumulating
//! audio, and disconnected clients cancel at the next bounded decoder batch.
use paddock_models::phonon::{FRAME_SAMPLES, MAX_FRAMES, MAX_INPUT_BYTES, SAMPLE_RATE};
use rand_chacha::ChaCha8Rng;
use rand_core::{Rng, SeedableRng};
use std::sync::mpsc::{SyncSender, sync_channel};

/// One worker plus a small bounded queue. HTTP admission counts requests until
/// the response body is consumed/dropped, including slow and queued clients.
pub const MAX_IN_FLIGHT: usize = 8;

#[derive(Clone, Copy)]
pub struct Options {
    pub seed: u64,
    pub temperature: f32,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0,
            temperature: 0.3,
        }
    }
}
impl Options {
    pub fn validate(self) -> Result<(), String> {
        if !self.temperature.is_finite() || !(0.0..=2.0).contains(&self.temperature) {
            return Err("temperature must be between 0 and 2".into());
        }
        Ok(())
    }
}
pub fn validate_text(text: &str) -> Result<(), String> {
    if text.trim().is_empty() || text.len() > MAX_INPUT_BYTES || text.contains('\0') {
        return Err(format!(
            "input must contain 1-{MAX_INPUT_BYTES} UTF-8 bytes of text, without NUL"
        ));
    }
    Ok(())
}
pub struct Noise {
    rng: ChaCha8Rng,
    scale: f64,
}
impl Noise {
    pub fn new(o: Options) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(o.seed),
            scale: f64::from(o.temperature).sqrt(),
        }
    }
    pub fn frame(&mut self) -> [f32; 32] {
        let mut out = [0.; 32];
        for p in out.as_chunks_mut::<2>().0 {
            let u = (f64::from(self.rng.next_u32()) + 0.5) / 4294967296.;
            let v = (f64::from(self.rng.next_u32()) + 0.5) / 4294967296.;
            let r = (-2. * u.ln()).sqrt() * self.scale;
            let a = std::f64::consts::TAU * v;
            p[0] = (r * a.cos()) as f32;
            p[1] = (r * a.sin()) as f32;
        }
        out
    }
}
pub trait Backend {
    fn generate(
        &mut self,
        text: &str,
        options: Options,
        emit: &mut dyn FnMut(Vec<f32>) -> bool,
    ) -> Result<(), String>;
}
struct Job {
    text: String,
    options: Options,
    output: tokio::sync::mpsc::Sender<Result<Output, String>>,
}
pub enum Output {
    Audio(Vec<f32>),
    Finished,
}
#[derive(Debug, thiserror::Error)]
pub enum StartError {
    #[error("{0}")]
    Invalid(String),
    #[error("speech worker is busy; retry shortly")]
    Busy,
    #[error("speech worker is unavailable")]
    Unavailable,
}
#[derive(Clone)]
pub struct Speech {
    jobs: SyncSender<Job>,
}
impl Speech {
    pub fn spawn<B: Backend + 'static>(
        load: impl FnOnce() -> Result<B, String> + Send + 'static,
    ) -> Result<Self, String> {
        // Match HTTP admission even if the worker has not yet received its
        // first job; otherwise a simultaneous burst can spuriously overflow.
        let (jobs, rx) = sync_channel::<Job>(MAX_IN_FLIGHT);
        let (ready, wait) = sync_channel(1);
        std::thread::Builder::new()
            .name("paddock-phonon".into())
            .spawn(move || {
                let mut backend = match load() {
                    Ok(b) => {
                        let _ = ready.send(Ok(()));
                        b
                    }
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    if job.output.is_closed() {
                        continue;
                    }
                    let result = backend.generate(&job.text, job.options, &mut |pcm| {
                        job.output.blocking_send(Ok(Output::Audio(pcm))).is_ok()
                    });
                    if let Err(e) = result {
                        let _ = job.output.blocking_send(Err(e));
                    } else {
                        let _ = job.output.blocking_send(Ok(Output::Finished));
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        wait.recv()
            .map_err(|_| "Phonon worker stopped during loading".to_string())??;
        Ok(Self { jobs })
    }
    pub fn start(
        &self,
        text: String,
        options: Options,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<Output, String>>, StartError> {
        validate_text(&text).map_err(StartError::Invalid)?;
        options.validate().map_err(StartError::Invalid)?;
        let (output, rx) = tokio::sync::mpsc::channel(2);
        self.jobs
            .try_send(Job {
                text,
                options,
                output,
            })
            .map_err(|e| match e {
                std::sync::mpsc::TrySendError::Full(_) => StartError::Busy,
                std::sync::mpsc::TrySendError::Disconnected(_) => StartError::Unavailable,
            })?;
        Ok(rx)
    }
}
pub fn pcm16(samples: &[f32]) -> Result<Vec<u8>, String> {
    if samples.iter().any(|v| !v.is_finite()) {
        return Err("Phonon produced non-finite audio".into());
    }
    Ok(samples
        .iter()
        .flat_map(|v| ((v.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes())
        .collect())
}
pub fn wav(pcm: &[u8]) -> Result<Vec<u8>, String> {
    if !pcm.len().is_multiple_of(2) || pcm.len() > MAX_FRAMES * FRAME_SAMPLES * 2 {
        return Err("invalid PCM length".into());
    }
    let mut out = Vec::with_capacity(44 + pcm.len());
    let size = pcm.len() as u32;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + size).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE as u32 * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&size.to_le_bytes());
    out.extend_from_slice(pcm);
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn queue_is_bounded_and_cancelled_jobs_never_enter_backend() {
        struct Fake(tokio::sync::mpsc::UnboundedSender<String>);
        impl Backend for Fake {
            fn generate(
                &mut self,
                text: &str,
                _: Options,
                emit: &mut dyn FnMut(Vec<f32>) -> bool,
            ) -> Result<(), String> {
                self.0.send(text.into()).unwrap();
                for _ in 0..100 {
                    if !emit(vec![0.; FRAME_SAMPLES]) {
                        break;
                    }
                }
                Ok(())
            }
        }
        let (entered, mut calls) = tokio::sync::mpsc::unbounded_channel();
        let service = Speech::spawn(move || Ok(Fake(entered))).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            let active = service.start("active".into(), Options::default()).unwrap();
            assert_eq!(calls.recv().await.as_deref(), Some("active"));
            // The unread active receiver blocks its worker on the bounded
            // audio channel. Queue slots therefore cannot race with dequeue.
            let mut queued = Vec::new();
            for _ in 0..MAX_IN_FLIGHT - 1 {
                queued.push(
                    service
                        .start("cancelled".into(), Options::default())
                        .unwrap(),
                );
            }
            let mut last = service.start("last".into(), Options::default()).unwrap();
            assert!(matches!(
                service.start("overflow".into(), Options::default()),
                Err(StartError::Busy)
            ));
            drop(queued);
            drop(active);
            assert_eq!(calls.recv().await.as_deref(), Some("last"));
            let mut frames = 0;
            while let Some(output) = last.recv().await {
                match output.unwrap() {
                    Output::Audio(_) => frames += 1,
                    Output::Finished => {
                        assert_eq!(frames, 100);
                        return;
                    }
                }
            }
            panic!("missing explicit completion marker");
        })
        .await
        .expect("queue must drain after cancellation");
        assert!(calls.try_recv().is_err());
    }

    #[tokio::test]
    async fn disconnected_stream_releases_worker_and_errors_are_not_success() {
        struct Fake;
        impl Backend for Fake {
            fn generate(
                &mut self,
                text: &str,
                _: Options,
                emit: &mut dyn FnMut(Vec<f32>) -> bool,
            ) -> Result<(), String> {
                for _ in 0..100 {
                    if !emit(vec![0.; FRAME_SAMPLES]) {
                        return Ok(());
                    }
                    if text == "fail" {
                        return Err("test synthesis failed".into());
                    }
                }
                Ok(())
            }
        }
        let service = Speech::spawn(|| Ok(Fake)).unwrap();
        let mut first = service.start("first".into(), Options::default()).unwrap();
        assert!(matches!(first.recv().await, Some(Ok(Output::Audio(_)))));
        drop(first);
        let mut next = service.start("fail".into(), Options::default()).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
            assert!(matches!(next.recv().await, Some(Ok(Output::Audio(_)))));
            assert!(matches!(next.recv().await,Some(Err(e)) if e=="test synthesis failed"));
            assert!(next.recv().await.is_none());
            let mut complete = service
                .start("complete".into(), Options::default())
                .unwrap();
            let mut n = 0;
            while let Some(v) = complete.recv().await {
                match v.unwrap() {
                    Output::Audio(_) => n += 1,
                    Output::Finished => {
                        assert_eq!(n, 100);
                        return;
                    }
                }
            }
            panic!("missing completion marker");
        })
        .await;
        assert!(result.is_ok(), "cancelled receiver must release the worker");
    }
    #[test]
    fn reproducible_finite_noise_and_zero_temperature() {
        let mut a = Noise::new(Options::default());
        let mut b = Noise::new(Options::default());
        for _ in 0..100 {
            let x = a.frame();
            assert_eq!(x, b.frame());
            assert!(x.iter().all(|x| x.is_finite()));
        }
        assert_eq!(
            Noise::new(Options {
                temperature: 0.,
                seed: 55
            })
            .frame(),
            [0.; 32]
        );
    }
    #[test]
    fn audio_container_and_validation() {
        let pcm = pcm16(&[-1., 0., 1.]).unwrap();
        let w = wav(&pcm).unwrap();
        assert_eq!(&w[0..4], b"RIFF");
        assert_eq!(w.len(), 50);
        assert!(pcm16(&[f32::NAN]).is_err());
        assert!(wav(&[0]).is_err());
        assert!(validate_text(" ").is_err());
        assert!(
            Options {
                temperature: f32::NAN,
                seed: 0
            }
            .validate()
            .is_err()
        );
    }
}
