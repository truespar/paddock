//! Disk restores at depth, off the engine thread.
//!
//! A T2 load used to read its extent synchronously on the engine thread,
//! into one of the two staging-ring slots, and hash it twice there (the
//! store's verify, then the catalog's end-to-end check) - BLAKE3 runs
//! ~1.45 GB/s on a GB10 core, so ~6 ms of a 4 MiB extent was hashing. The
//! two slots freed only when a flight retired at the next pump, so a restore
//! moved two extents per scheduler tick: a 700 MB Qwen3.8-27B session took
//! 2.5-6 s off a 9 GB/s drive, against an estimate of 0.5 s, and every
//! other request's decode waited on the hashing.
//!
//! Here [`READERS`] threads each read whole extents - the depth the device
//! probe measured (4 MiB at queue depth 4+ is the device's peak), from
//! threads over positioned IO, the technique this tier ships everywhere
//! (see `io.rs`) - straight into pinned host memory the engine then copies
//! to the GPU from, and verify each against its commit record on the same
//! thread. The engine thread only submits and, on completion, launches the
//! transfer it would have launched anyway.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use super::OpId;
use super::digest::Checksum;
use super::io::{Backend, READERS};

/// One extent to read: `padded` bytes at `off` of `path` into `dst`, the
/// first `len` of them checked against `expect`.
pub(super) struct ReadJob {
    pub op: OpId,
    pub path: PathBuf,
    pub off: u64,
    pub dst: *mut u8,
    pub padded: usize,
    pub len: usize,
    pub expect: [u8; 32],
}

// SAFETY: `dst` is the job's exclusively until its completion is collected
// (see `T2Readers::submit`); nothing else touches the region meanwhile.
unsafe impl Send for ReadJob {}

pub(super) enum ReadDone {
    Ok(OpId),
    /// The bytes did not match their commit record - at-rest corruption.
    Corrupt(OpId),
    Failed(OpId, std::io::Error),
}

pub(super) struct T2Readers {
    tx: Option<mpsc::Sender<ReadJob>>,
    done: mpsc::Receiver<ReadDone>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl T2Readers {
    pub(super) fn new(io: Arc<Backend>) -> Self {
        let (tx, rx) = mpsc::channel::<ReadJob>();
        let (dtx, done) = mpsc::channel::<ReadDone>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..READERS)
            .map(|i| {
                let (rx, dtx, io) = (Arc::clone(&rx), dtx.clone(), Arc::clone(&io));
                std::thread::Builder::new()
                    .name(format!("kv-t2-read-{i}"))
                    .spawn(move || {
                        loop {
                            let job = {
                                let g = rx.lock().expect("t2 read queue");
                                match g.recv() {
                                    Ok(j) => j,
                                    Err(_) => break, // the transport is gone
                                }
                            };
                            // SAFETY: the submitter keeps `dst` valid and
                            // untouched for `padded` bytes until this reports
                            let buf =
                                unsafe { std::slice::from_raw_parts_mut(job.dst, job.padded) };
                            let r = match io.read_on(&job.path, buf, job.off, i) {
                                Ok(()) if Checksum::of_payload(&buf[..job.len]).0 == job.expect => {
                                    ReadDone::Ok(job.op)
                                }
                                Ok(()) => ReadDone::Corrupt(job.op),
                                Err(e) => ReadDone::Failed(job.op, e),
                            };
                            if dtx.send(r).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn kv-t2 reader")
            })
            .collect();
        Self {
            tx: Some(tx),
            done,
            workers,
        }
    }

    /// Queue a read. False when the pool is gone (the caller falls back).
    ///
    /// # Safety
    /// `job.dst` must stay valid, aligned for the device and untouched by
    /// anyone else for `job.padded` bytes until this op's [`ReadDone`] has
    /// been collected (dropping the pool waits for every queued read).
    pub(super) unsafe fn submit(&self, job: ReadJob) -> bool {
        self.tx.as_ref().is_some_and(|t| t.send(job).is_ok())
    }

    /// Every read finished since the last call.
    pub(super) fn collect(&self) -> Vec<ReadDone> {
        self.done.try_iter().collect()
    }
}

impl Drop for T2Readers {
    fn drop(&mut self) {
        self.tx = None; // closing the queue ends every worker once it drains
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::io::AlignedBuf;
    use super::super::nvme_store::NvmeStore;
    use super::*;

    fn wait(r: &T2Readers, n: usize) -> Vec<ReadDone> {
        let mut out = Vec::new();
        for _ in 0..5000 {
            out.extend(r.collect());
            if out.len() >= n {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        out
    }

    /// Several extents read at once, each verified on its reader; a byte
    /// flipped at rest comes back Corrupt, never as data.
    #[test]
    fn reads_land_verified_and_corruption_is_caught() {
        let d = std::env::temp_dir().join(format!("pkv-t2r-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        let payloads: Vec<Vec<u8>> = (0..6u8)
            .map(|n| {
                (0..(3 << 20) + 777)
                    .map(|i| n.wrapping_mul(31).wrapping_add(i as u8))
                    .collect()
            })
            .collect();
        for (n, p) in payloads.iter().enumerate() {
            s.store([n as u8; 32], 1, 1, p).unwrap();
        }
        let readers = T2Readers::new(s.reader_io());
        let mut bufs = Vec::new();
        for n in 0..6u8 {
            let plan = s.read_plan(&[n; 32]).unwrap();
            let mut b = AlignedBuf::new(plan.padded as usize, plan.align);
            let job = ReadJob {
                op: OpId(n as u64),
                path: plan.path,
                off: plan.off,
                dst: b.slice_mut().as_mut_ptr(),
                padded: plan.padded as usize,
                len: plan.len as usize,
                expect: plan.checksum,
            };
            // SAFETY: `b` lives (in `bufs`) until every read is collected
            assert!(unsafe { readers.submit(job) });
            bufs.push(b);
        }
        let done = wait(&readers, 6);
        assert_eq!(done.len(), 6);
        assert!(done.iter().all(|d| matches!(d, ReadDone::Ok(_))));
        for (n, b) in bufs.iter().enumerate() {
            assert_eq!(
                &b.slice()[..payloads[n].len()],
                &payloads[n][..],
                "extent {n}"
            );
        }

        // flip one byte of extent 2 at rest
        let plan = s.read_plan(&[2; 32]).unwrap();
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .open(&plan.path)
                .unwrap();
            f.seek(SeekFrom::Start(plan.off + 1000)).unwrap();
            f.write_all(&[payloads[2][1000] ^ 0xff]).unwrap();
            f.sync_all().unwrap();
        }
        let mut b = AlignedBuf::new(plan.padded as usize, plan.align);
        let job = ReadJob {
            op: OpId(99),
            path: plan.path,
            off: plan.off,
            dst: b.slice_mut().as_mut_ptr(),
            padded: plan.padded as usize,
            len: plan.len as usize,
            expect: plan.checksum,
        };
        // SAFETY: `b` outlives the collect below
        assert!(unsafe { readers.submit(job) });
        let done = wait(&readers, 1);
        assert!(matches!(done.as_slice(), [ReadDone::Corrupt(OpId(99))]));
        drop(readers);
        drop(s);
        let _ = std::fs::remove_dir_all(&d);
    }
}
