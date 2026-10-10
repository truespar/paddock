//! Payload checksums off the engine thread.
//!
//! BLAKE3 hashes ~1.5 GB/s on one core of this class of CPU (measured on the
//! GB10's Grace cores: 2.9 ms per 4 MiB run, 10.8 ms per 16 MiB blob shard),
//! and the RAM transport used to hash every finished store and load inline in
//! `poll` - on the service tick. A Qwen3.8 restore (400 MB of KV plus a
//! 157 MB DeltaNet state) spent ~0.4 s of its ~0.45 s there, and a store
//! window's worth of write-through could hold a decode tick for 100+ ms. A few
//! workers hash finished flights side by side instead; `poll` hands out their
//! completions as the digests come back.
//!
//! The bytes stay put meanwhile, by the transport's own ownership rules: a
//! store's extent is freed only after its completion, and a load's source
//! stays pinned by the catalog until its completion is consumed. Both
//! completions now simply wait for the digest.
//!
//! Loads go first: a restore has a request parked on it, a write-through
//! store has nobody waiting. One FIFO queued every restore behind the
//! mirror's backlog of digests.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread::JoinHandle;

use super::OpId;
use super::digest::Checksum;

/// Workers: enough to keep a restore near bus speed (~4x one core), few
/// enough to leave the serving process its cores.
const WORKERS: usize = 4;

struct Job {
    op: OpId,
    ptr: usize,
    len: usize,
}

/// The work queue: restore digests ahead of write-through ones.
#[derive(Default)]
struct Queue {
    loads: VecDeque<Job>,
    stores: VecDeque<Job>,
    closed: bool,
}

type Shared = Arc<(Mutex<Queue>, Condvar)>;

pub(super) struct HashPool {
    queue: Shared,
    rx: mpsc::Receiver<(OpId, Checksum)>,
    /// Empty when no worker could be started - then `submit` hashes inline,
    /// exactly as before the pool existed.
    workers: Vec<JoinHandle<()>>,
    /// Digests computed inline (no workers), handed out by the next collect.
    inline: Vec<(OpId, Checksum)>,
}

/// Next job for a worker, loads first; None once the queue is closed and
/// drained.
fn next(q: &Shared) -> Option<Job> {
    let (m, cv) = &**q;
    let mut g = m.lock().ok()?;
    loop {
        if let Some(j) = g.loads.pop_front().or_else(|| g.stores.pop_front()) {
            return Some(j);
        }
        if g.closed {
            return None;
        }
        g = cv.wait(g).ok()?;
    }
}

impl HashPool {
    pub(super) fn new() -> Self {
        let queue: Shared = Arc::default();
        let (done, rx) = mpsc::channel();
        let workers: Vec<JoinHandle<()>> = (0..WORKERS)
            .filter_map(|i| {
                let queue = queue.clone();
                let done = done.clone();
                std::thread::Builder::new()
                    .name(format!("kv-tier-hash-{i}"))
                    .spawn(move || {
                        while let Some(job) = next(&queue) {
                            // SAFETY: `submit`'s contract - the transport
                            // keeps these bytes alive and unmodified until it
                            // has collected this digest, or joined us.
                            let bytes = unsafe {
                                std::slice::from_raw_parts(job.ptr as *const u8, job.len)
                            };
                            if done.send((job.op, Checksum::of_payload(bytes))).is_err() {
                                return;
                            }
                        }
                    })
                    .ok()
            })
            .collect();
        if workers.is_empty() {
            tracing::warn!("KV tier: no checksum workers - hashing on the engine thread");
        }
        Self {
            queue,
            rx,
            workers,
            inline: Vec::new(),
        }
    }

    /// Hash `len` bytes at `ptr` for `op` - a restore's (`load`) ahead of
    /// any write-through's; the digest comes out of a later
    /// [`Self::collect`].
    ///
    /// # Safety
    /// The bytes must stay alive and unmodified until `collect` has returned
    /// `op`'s digest or the pool has been dropped.
    pub(super) unsafe fn submit(&mut self, op: OpId, ptr: *const u8, len: usize, load: bool) {
        let job = Job {
            op,
            ptr: ptr as usize,
            len,
        };
        if !self.workers.is_empty() {
            let (m, cv) = &*self.queue;
            if let Ok(mut q) = m.lock() {
                if load {
                    q.loads.push_back(job);
                } else {
                    q.stores.push_back(job);
                }
                cv.notify_one();
                return;
            }
        }
        // SAFETY: caller's contract, and we read the bytes right now.
        let bytes = unsafe { std::slice::from_raw_parts(job.ptr as *const u8, job.len) };
        self.inline.push((op, Checksum::of_payload(bytes)));
    }

    /// Digests finished since the last call. Never blocks.
    pub(super) fn collect(&mut self) -> Vec<(OpId, Checksum)> {
        let mut out = std::mem::take(&mut self.inline);
        while let Ok(d) = self.rx.try_recv() {
            out.push(d);
        }
        out
    }
}

impl Drop for HashPool {
    fn drop(&mut self) {
        // closing the queue lets the workers drain it and exit; joining
        // here keeps every read inside the owner's lifetime
        let (m, cv) = &*self.queue;
        if let Ok(mut q) = m.lock() {
            q.closed = true;
        }
        cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
