//! Nemotron 3.5 ASR serving seam: one thread owns the model and every
//! session's device state; requests and live streams reach it as messages
//! and come back with the tokens their audio produced.
//!
//! The unit of work is a window (at most `MAX_WINDOW` encoder frames, 17.92
//! s of audio), never a request: a file is fed in window-sized quanta, so a
//! live session waits behind at most one window of somebody else's file, and
//! a session's arrivals that complete no chunk cost the worker nothing but
//! the copy. Tokens are append-only - an RNN-T emits each one once, at the
//! frame that produced it - so a stream's text only ever grows, and the
//! tokens a stream produced are the file's to the bit (the engine's windows
//! are invisible to the arithmetic).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, Weak};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, oneshot};

use crate::gpu_model::nemotron_asr::{GpuNemotronAsr, MAX_WINDOW, Session};
use paddock_models::nemotron_asr as arch;

/// Samples a file feeds the worker at a time: one full window's audio.
const FILE_QUANTUM: usize = MAX_WINDOW * arch::SUBSAMPLING * arch::HOP;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("the speech recognizer is at its session limit; retry shortly")]
    Busy,
    #[error("the speech recognizer is unavailable")]
    Unavailable,
    #[error("{0}")]
    Backend(String),
}

/// What a step produced: the tokens its audio completed (with their encoder
/// frames, 80 ms each) and where the stream stands.
#[derive(Clone, Debug, Default)]
pub struct Update {
    pub tokens: Vec<u32>,
    pub frames: Vec<u32>,
    pub samples: usize,
    pub encoded_frames: usize,
    pub finished: bool,
    pub gpu_seconds: f64,
}

impl Update {
    fn extend(&mut self, u: Update) {
        self.tokens.extend(u.tokens);
        self.frames.extend(u.frames);
        self.samples = u.samples;
        self.encoded_frames = u.encoded_frames;
        self.finished = u.finished;
        self.gpu_seconds += u.gpu_seconds;
    }
}

enum Msg {
    Open {
        id: u64,
        lookahead: usize,
        prompt: u32,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Feed {
        id: u64,
        pcm: Vec<f32>,
        finish: bool,
        reply: oneshot::Sender<Result<Update, String>>,
    },
    Close {
        id: u64,
    },
}

/// The checkpoint's facts a front end needs without a round trip.
#[derive(Clone)]
pub struct Facts {
    /// the prompt dictionary: (language key, prompt index)
    pub prompts: Vec<(String, u32)>,
    pub vocab: Arc<Vec<String>>,
    pub weight_bytes: u64,
    pub workspace_bytes: u64,
}

/// The handle to the worker.
pub struct Recognizer {
    tx: SyncSender<Msg>,
    facts: Facts,
    metrics: Arc<crate::metrics::EngineMetrics>,
    sessions: Arc<Semaphore>,
    next: AtomicU64,
    worker_alive: Weak<()>,
}

/// A live stream. Dropping it frees its device state.
pub struct RecognizerSession {
    id: u64,
    tx: SyncSender<Msg>,
    lookahead: usize,
    finished: bool,
    _permit: OwnedSemaphorePermit,
    worker_alive: Weak<()>,
}

impl Drop for RecognizerSession {
    fn drop(&mut self) {
        // a full queue only delays the release: the worker drops a session
        // whose handle is gone the next time it sees it
        let _ = self.tx.try_send(Msg::Close { id: self.id });
    }
}

impl RecognizerSession {
    pub fn lookahead(&self) -> usize {
        self.lookahead
    }

    /// Feed PCM (16 kHz mono, finite) and, with `finish`, close the
    /// recording. Returns the tokens it completed.
    pub async fn feed(&mut self, pcm: Vec<f32>, finish: bool) -> Result<Update, Error> {
        if self.finished {
            return Err(Error::Invalid("the stream is finished".into()));
        }
        if pcm.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("non-finite audio".into()));
        }
        if self.worker_alive.strong_count() == 0 {
            return Err(Error::Unavailable);
        }
        let (reply, rx) = oneshot::channel();
        self.tx
            .try_send(Msg::Feed {
                id: self.id,
                pcm,
                finish,
                reply,
            })
            .map_err(|e| match e {
                TrySendError::Full(_) => Error::Busy,
                TrySendError::Disconnected(_) => Error::Unavailable,
            })?;
        let u = rx
            .await
            .map_err(|_| Error::Unavailable)?
            .map_err(Error::Backend)?;
        self.finished = u.finished;
        Ok(u)
    }
}

impl Recognizer {
    /// Start the worker: `build` runs on it (the model is created where it
    /// lives), then one silent second warms every kernel's first launch.
    pub fn spawn<F>(build: F, max_sessions: usize) -> Result<Self, String>
    where
        F: FnOnce() -> Result<GpuNemotronAsr, String> + Send + 'static,
    {
        let (tx, rx) = sync_channel::<Msg>(4 * max_sessions.max(1));
        let (ready, wait) = channel::<Result<Facts, String>>();
        let metrics = Arc::new(crate::metrics::EngineMetrics::default());
        let m = Arc::clone(&metrics);
        let alive = Arc::new(());
        let worker_alive = Arc::downgrade(&alive);
        std::thread::Builder::new()
            .name("paddock-nemotron-asr".into())
            .spawn(move || {
                let _alive = alive;
                let mut model = match build() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready.send(Err(e));
                        return;
                    }
                };
                let warm = model
                    .prompt(arch::AUTO_PROMPT)
                    .ok_or_else(|| "no auto prompt".to_owned())
                    .and_then(|p| {
                        model
                            .transcribe(&vec![0.; arch::SAMPLE_RATE], arch::DEFAULT_LOOKAHEAD, p)
                            .map_err(|e| format!("Nemotron 3.5 ASR warm-up: {e}"))
                    });
                if let Err(e) = warm {
                    let _ = ready.send(Err(e));
                    return;
                }
                let facts = Facts {
                    prompts: model.prompts().to_vec(),
                    vocab: Arc::new(model.vocab().to_vec()),
                    weight_bytes: model.weight_bytes(),
                    workspace_bytes: model.workspace_bytes(),
                };
                m.weights_mem_bytes.store(facts.weight_bytes, Relaxed);
                m.model_mem_bytes
                    .store(facts.weight_bytes + facts.workspace_bytes, Relaxed);
                if ready.send(Ok(facts)).is_err() {
                    return;
                }
                // every stream's state, and the reply its last feed awaits
                struct Live {
                    s: Session,
                    reply: Option<(oneshot::Sender<Result<Update, String>>, std::time::Instant)>,
                }
                let mut live: HashMap<u64, Live> = HashMap::new();
                let mut turn = 0usize;
                loop {
                    // block only when no feed is waiting for its tokens
                    let waiting = live.values().any(|l| l.reply.is_some());
                    let mut msgs = Vec::new();
                    if !waiting {
                        match rx.recv() {
                            Ok(msg) => msgs.push(msg),
                            Err(_) => break,
                        }
                    }
                    while let Ok(msg) = rx.try_recv() {
                        msgs.push(msg);
                    }
                    for msg in msgs {
                        match msg {
                            Msg::Open {
                                id,
                                lookahead,
                                prompt,
                                reply,
                            } => {
                                let r = model.session(lookahead, prompt).map_err(|e| e.to_string());
                                let _ = reply.send(r.map(|s| {
                                    live.insert(id, Live { s, reply: None });
                                }));
                            }
                            Msg::Feed {
                                id,
                                pcm,
                                finish,
                                reply,
                            } => {
                                let Some(l) = live.get_mut(&id) else {
                                    let _ = reply.send(Err("no such stream".into()));
                                    continue;
                                };
                                if reply.is_closed() {
                                    // the caller gave up mid-flight: the stream
                                    // cannot be resumed without this audio
                                    live.remove(&id);
                                    continue;
                                }
                                let r = model.feed(&mut l.s, &pcm).map_err(|e| e.to_string());
                                if let Err(e) = r {
                                    let _ = reply.send(Err(e));
                                    live.remove(&id);
                                    continue;
                                }
                                if finish {
                                    model.close(&mut l.s);
                                }
                                l.reply = Some((reply, std::time::Instant::now()));
                            }
                            Msg::Close { id } => {
                                live.remove(&id);
                            }
                        }
                    }
                    m.active_slots.store(live.len() as u32, Relaxed);
                    // passes until no stream has a window ready: every stream
                    // with one rides each pass (its next window), the start
                    // rotating so no stream always waits behind the others
                    m.phase.store(crate::metrics::PHASE_PREFILL, Relaxed);
                    let mut failed: Option<String> = None;
                    loop {
                        let mut ready: Vec<(&u64, &mut Live)> = live
                            .iter_mut()
                            .filter(|(_, l)| GpuNemotronAsr::next_window(&l.s).is_some())
                            .collect();
                        if ready.is_empty() {
                            break;
                        }
                        ready.sort_by_key(|(id, _)| **id);
                        let k = turn % ready.len();
                        ready.rotate_left(k);
                        turn += 1;
                        let mut sessions: Vec<&mut Session> =
                            ready.into_iter().map(|(_, l)| &mut l.s).collect();
                        if let Err(e) = model.run(&mut sessions) {
                            failed = Some(e.to_string());
                            break;
                        }
                    }
                    m.phase.store(crate::metrics::PHASE_IDLE, Relaxed);
                    // answer every feed whose audio is consumed
                    let mut gone = Vec::new();
                    for (id, l) in live.iter_mut() {
                        if l.reply.is_none() {
                            continue;
                        }
                        if failed.is_none() && GpuNemotronAsr::next_window(&l.s).is_some() {
                            continue;
                        }
                        let Some((reply, t0)) = l.reply.take() else {
                            continue;
                        };
                        let out = match &failed {
                            Some(e) => Err(e.clone()),
                            None => Ok(Update {
                                tokens: std::mem::take(&mut l.s.tokens),
                                frames: std::mem::take(&mut l.s.frames),
                                samples: l.s.samples(),
                                encoded_frames: l.s.encoded_frames(),
                                finished: l.s.is_finished(),
                                gpu_seconds: t0.elapsed().as_secs_f64(),
                            }),
                        };
                        if out.is_err() || l.s.is_finished() {
                            gone.push(*id);
                        }
                        let _ = reply.send(out);
                    }
                    for id in gone {
                        live.remove(&id);
                    }
                    m.active_slots.store(live.len() as u32, Relaxed);
                }
            })
            .map_err(|e| e.to_string())?;
        let facts = wait
            .recv()
            .map_err(|_| "Nemotron 3.5 ASR worker failed at startup")??;
        Ok(Self {
            tx,
            facts,
            metrics,
            sessions: Arc::new(Semaphore::new(max_sessions.max(1))),
            next: AtomicU64::new(1),
            worker_alive,
        })
    }

    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    pub fn metrics(&self) -> Arc<crate::metrics::EngineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// The prompt index of a language key (`de-DE`, `de`, `auto`).
    pub fn prompt(&self, key: &str) -> Option<u32> {
        self.facts
            .prompts
            .iter()
            .find(|(k, _)| k == key)
            .map(|&(_, id)| id)
    }

    /// Open a live stream at `lookahead` (one of the published right
    /// contexts) under `prompt`.
    pub async fn session(&self, lookahead: usize, prompt: u32) -> Result<RecognizerSession, Error> {
        if !arch::LOOKAHEADS.contains(&lookahead) {
            return Err(Error::Invalid(format!(
                "lookahead {lookahead}: the model publishes {:?} (80 ms frames)",
                arch::LOOKAHEADS
            )));
        }
        if self.worker_alive.strong_count() == 0 {
            return Err(Error::Unavailable);
        }
        let permit = self
            .sessions
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Busy)?;
        let id = self.next.fetch_add(1, Relaxed);
        let (reply, rx) = oneshot::channel();
        self.tx
            .try_send(Msg::Open {
                id,
                lookahead,
                prompt,
                reply,
            })
            .map_err(|e| match e {
                TrySendError::Full(_) => Error::Busy,
                TrySendError::Disconnected(_) => Error::Unavailable,
            })?;
        rx.await
            .map_err(|_| Error::Unavailable)?
            .map_err(Error::Backend)?;
        Ok(RecognizerSession {
            id,
            tx: self.tx.clone(),
            lookahead,
            finished: false,
            _permit: permit,
            worker_alive: self.worker_alive.clone(),
        })
    }

    /// A whole recording, fed a window's audio at a time so live streams
    /// interleave with it.
    pub async fn transcribe(
        &self,
        pcm: &[f32],
        lookahead: usize,
        prompt: u32,
    ) -> Result<Update, Error> {
        let mut s = self.session(lookahead, prompt).await?;
        let mut out = Update::default();
        let mut parts = pcm.chunks(FILE_QUANTUM).peekable();
        if parts.peek().is_none() {
            out.extend(s.feed(Vec::new(), true).await?);
        }
        while let Some(part) = parts.next() {
            let last = parts.peek().is_none();
            out.extend(s.feed(part.to_vec(), last).await?);
        }
        Ok(out)
    }
}
