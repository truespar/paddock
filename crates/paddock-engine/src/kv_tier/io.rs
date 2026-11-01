//! Platform IO for the T2 segment files - unbuffered, positioned,
//! and issued at DEPTH, with the per-device geometry MEASURED at open.
//!
//! The device probes established the facts this module exists for:
//!
//! 1. **Queue depth is a 10x lever on reads.** 256 KiB random: 1.19 GB/s at
//!    qd1, 11.88 at qd16. 4 MiB: 6.99 -> 15.5. A synchronous QD1 restore
//!    throws away ~90% of the device, which is the difference between a
//!    restore that beats recompute and one that loses to it.
//! 2. **Write bandwidth is the scarce side AND it degrades with chunk size**
//!    through the RAID driver measured here: 2.51 GB/s at 256 KiB, 0.81 at 4 MiB,
//!    0.66 at 16 MiB. The best write chunk is a *device* fact, not a
//!    universal one - so we measure it instead of electing it globally
//!    (per-device probing at store open is mandatory product
//!    behavior; the same machine offers a 15 GB/s tier and a 0.25 GB/s trap
//!    one drive letter apart).
//! 3. The probe reached those numbers with **thread-emulated depth over
//!    positioned unbuffered IO**, each worker holding its own handle. That
//!    detail is load-bearing: a Windows synchronous handle serializes IO in
//!    the kernel, so sharing one handle (or a `try_clone` of it) is depth 1
//!    wearing a costume.
//!
//! We ship exactly the technique that was measured. That is also why there is
//! no io_uring dependency: at 1-4 MiB chunks the submission-syscall cost is
//! microseconds against milliseconds of device time, so a ring would buy
//! latency we could not measure, at the price of a Linux-only dep and a
//! second code path. The [`Backend`] seam is narrow enough that a ring (or a
//! GDS direct path) drops in behind it if a later probe on real NVMe says
//! otherwise.
//!
//! Everything here is elected or measured - there is no user-facing knob, per
//! the no-customer-side-tuning rule.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(windows)]
use std::os::windows::fs::{FileExt, OpenOptionsExt};

#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

/// `FILE_FLAG_NO_BUFFERING` - bypass the Windows cache manager entirely. The
/// KV tier is the cache; a page-cache copy of it is pure memory waste and it
/// hides the device's real behaviour from our own measurements.
#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

/// Read chunk floor (reads want >= 1 MB chunks). Below this the
/// per-request overhead starts to show against the transfer itself.
const READ_CHUNK_MIN: u64 = 1 << 20;
/// Read chunk ceiling. The 4 MiB rung is where random reads peak (15.5 GB/s
/// at qd4); bigger chunks buy nothing and cost queue slots.
const READ_CHUNK_MAX: u64 = 4 << 20;
/// Elected read depth: 4 MiB reads reach 15.48 GB/s at qd4 and 15.49 at
/// qd16 - qd4 captures the entire win, so we spend 3 worker threads, not 15.
const READ_DEPTH: usize = 4;

/// Write-chunk candidates the open-time probe ladders. 256 KiB and 4 MiB
/// bracket the measured spread (2.51 vs 0.81 GB/s on rcraid); a bare NVMe
/// device typically has the opposite slope, which is the point of measuring.
const WRITE_CHUNK_CANDIDATES: [u64; 2] = [256 << 10, 4 << 20];

/// Bytes moved per rung of the open-time probe. Big enough to leave the
/// first-write noise behind, small enough that the whole probe is ~20 ms.
const PROBE_BYTES: u64 = 4 << 20;
/// Rungs are sampled this many times and the best is kept. A probe competes
/// with whatever else the machine is doing, and the probes measured how
/// brutal that is (a busy volume drops reads to 3% retention) - the peak is
/// the device's property, the mean is the moment's.
const PROBE_REPS: usize = 3;

/// The catastrophic floor: below this the path is not storage we can cache
/// on at all (a failing disk, a network share, a throttled container mount).
/// Deliberately FAR under the "is this a good idea" line - whether a restore
/// beats recompute is the cost model's judgement, made per request against a
/// continuously measured rate, not a startup one-shot's. The probe numbers are
/// why: the same SSD probes at 15 GB/s idle and 0.39 GB/s with two writers on
/// it, so a one-shot sample near the decision boundary would refuse good
/// devices on a busy boot.
///
/// Not enforced in this crate's own unit tests. They open dozens of stores at
/// once on whatever the temp directory sits on, and that is the "two writers"
/// case above taken to its end: on a spinning array shared by 28 probes each
/// one reads 0.02 GB/s and the whole suite is refused. Those tests are about
/// the WAL, the segments and the geometry ladder, not about qualifying the
/// disk a developer happens to keep temp files on.
const MIN_VIABLE_READ_GBS: f64 = if cfg!(test) { 0.0 } else { 0.05 };
/// Below this we serve but say so - a usable-but-slow device changes the
/// cost model's answers, and the operator should hear it from us first.
const GOOD_READ_GBS: f64 = 1.0;

/// Alignment candidates for the discovery ladder. Never assume 4 KiB -
/// volumes reporting 512 B logical are common.
const ALIGN_CANDIDATES: [u64; 3] = [512, 4096, 65536];

pub fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

/// A heap buffer with an aligned interior view. Unbuffered IO constrains the
/// buffer ADDRESS as well as the offset and length, and Rust's allocator
/// makes no such promise, so we over-allocate and slice.
pub struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedBuf {
    pub fn new(len: usize, align: u64) -> Self {
        let a = align as usize;
        let mut raw = vec![0u8; len + a];
        let addr = raw.as_mut_ptr() as usize;
        let off = (a - (addr % a)) % a;
        Self { raw, off, len }
    }

    pub fn slice(&self) -> &[u8] {
        &self.raw[self.off..self.off + self.len]
    }

    pub fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

#[cfg(windows)]
fn pread(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    f.seek_read(buf, off)
}

#[cfg(unix)]
fn pread(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
    f.read_at(buf, off)
}

#[cfg(windows)]
fn pwrite(f: &File, buf: &[u8], off: u64) -> io::Result<usize> {
    f.seek_write(buf, off)
}

#[cfg(unix)]
fn pwrite(f: &File, buf: &[u8], off: u64) -> io::Result<usize> {
    f.write_at(buf, off)
}

/// Positioned read that fills `buf` or fails - short reads are retried at the
/// advanced offset, EOF is an error (the caller only ever asks for extents a
/// commit record says are there).
fn pread_exact(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        match pread(f, &mut buf[done..], off + done as u64) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short read: extent ends before its commit record says",
                ));
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

fn pwrite_all(f: &File, buf: &[u8], off: u64) -> io::Result<()> {
    let mut done = 0usize;
    while done < buf.len() {
        match pwrite(f, &buf[done..], off + done as u64) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "device accepted 0 bytes",
                ));
            }
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Open a segment file, unbuffered when the device allows it.
pub fn open_segment(path: &Path, unbuffered: bool) -> io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true).write(true).create(true);
    #[cfg(windows)]
    if unbuffered {
        o.custom_flags(FILE_FLAG_NO_BUFFERING);
    }
    #[cfg(target_os = "linux")]
    if unbuffered {
        o.custom_flags(libc::O_DIRECT);
    }
    let f = o.open(path)?;
    // macOS has no O_DIRECT; F_NOCACHE is the documented equivalent (skip
    // the unified buffer cache for this descriptor). It is advisory rather
    // than a hard alignment contract, which is why the alignment ladder
    // still runs - it just always succeeds at 512 there.
    #[cfg(target_os = "macos")]
    if unbuffered {
        // SAFETY: plain fcntl on a live descriptor; failure is non-fatal.
        unsafe {
            libc::fcntl(
                std::os::unix::io::AsRawFd::as_raw_fd(&f),
                libc::F_NOCACHE,
                1,
            );
        }
    }
    let _ = unbuffered;
    Ok(f)
}

/// A pool job: one chunk of one extent. The pointer rides across the thread
/// boundary because the caller BLOCKS until every chunk of its request has
/// reported - the borrow is scoped by that wait, not by the type system.
struct Job {
    /// The segment's handles; worker `i` uses `workers[i]`.
    file: Arc<Handles>,
    ptr: SendPtr,
    len: usize,
    off: u64,
    write: bool,
    done: mpsc::Sender<io::Result<()>>,
}

struct SendPtr(*mut u8);
// SAFETY: the target region is owned by the blocked caller for the whole
// lifetime of the job, and chunks never overlap (see `run_chunks`).
unsafe impl Send for SendPtr {}

/// Async disk readers (`t2_reads`): each owns a handle slot past the pool's.
pub const READERS: usize = 8;

/// Handle slots per segment: the calling thread's (slot 0, the only one
/// writes use), one per pool worker, one per async reader. On Windows each
/// is its own handle - a synchronous handle serializes IO in the kernel, so
/// sharing one is depth 1 (see the module note). Elsewhere positioned IO on
/// one descriptor runs in parallel (no shared file offset), so every slot
/// is the same descriptor and a big cache costs one per segment.
const SLOTS: usize = if cfg!(windows) {
    READ_DEPTH + READERS
} else {
    1
};

/// Every handle on one segment file, opened when a slot first needs it.
///
/// Owned by the [`Backend`], never by the workers. Workers used to cache a
/// handle per segment for the life of the process, and an open descriptor
/// keeps a deleted file's blocks allocated: every evicted segment that had
/// ever served a restore stayed on the disk until the runner exited,
/// invisible to the cache's own size and to `du`, while the volume filled.
struct Handles {
    path: PathBuf,
    unbuffered: bool,
    slots: [std::sync::OnceLock<File>; SLOTS],
}

impl Handles {
    fn file(&self, slot: usize) -> io::Result<&File> {
        // one shared descriptor where slots are not their own handles
        let cell = self.slots.get(slot).unwrap_or(&self.slots[0]);
        if let Some(f) = cell.get() {
            return Ok(f);
        }
        // two threads may race to open the slot; one handle wins, the other
        // closes - either way the slot ends up with a live handle
        let _ = cell.set(open_segment(&self.path, self.unbuffered)?);
        Ok(cell.get().expect("set above"))
    }
}

/// Worker threads that give the device its queue depth.
struct IoPool {
    tx: Option<mpsc::Sender<Job>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl IoPool {
    fn new(n: usize) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..n)
            .map(|i| {
                let rx = Arc::clone(&rx);
                std::thread::Builder::new()
                    .name(format!("kv-t2-io-{i}"))
                    .spawn(move || {
                        loop {
                            let job = {
                                let g = rx.lock().expect("io pool queue");
                                match g.recv() {
                                    Ok(j) => j,
                                    Err(_) => break, // sender dropped: shut down
                                }
                            };
                            let Job {
                                file,
                                ptr,
                                len,
                                off,
                                write,
                                done,
                            } = job;
                            // SAFETY: caller-owned region, exclusive for this
                            // chunk, valid until it reports done.
                            let buf = unsafe { std::slice::from_raw_parts_mut(ptr.0, len) };
                            let r = file.file(1 + i).and_then(|f| {
                                if write {
                                    pwrite_all(f, buf, off)
                                } else {
                                    pread_exact(f, buf, off)
                                }
                            });
                            // let go of the handles BEFORE reporting: once the
                            // caller has every report it may forget and delete
                            // the segment, and its space is free at once only
                            // if no worker still holds a descriptor
                            drop(file);
                            let _ = done.send(r);
                        }
                    })
                    .expect("spawn kv-t2 io worker")
            })
            .collect();
        Self {
            tx: Some(tx),
            workers,
        }
    }

    fn submit(&self, job: Job) -> bool {
        self.tx.as_ref().is_some_and(|t| t.send(job).is_ok())
    }
}

impl Drop for IoPool {
    fn drop(&mut self) {
        self.tx = None; // closing the channel ends every worker loop
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// One probe per VOLUME per process. `dir_for` gives every identity+scope
/// its own store directory, so a process can open a handful of them - all on
/// the same physical device, all with the same answer. Probing each would
/// cost real IO and, worse, make them contend with each other and measure
/// the contention rather than the device.
static PROBED: std::sync::LazyLock<Mutex<HashMap<String, DeviceClass>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// A stable per-volume key: the drive prefix on Windows, the device id on
/// unix. Both answer "same physical device?" without a mount-table walk.
fn volume_key(dir: &Path) -> String {
    #[cfg(windows)]
    {
        if let Some(std::path::Component::Prefix(p)) = dir.components().next() {
            return p.as_os_str().to_string_lossy().to_uppercase();
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(md) = std::fs::metadata(dir) {
            return format!("dev:{}", md.dev());
        }
    }
    dir.to_string_lossy().to_string()
}

/// What the open-time probe learned about this device.
#[derive(Debug, Clone, Copy)]
pub struct DeviceClass {
    pub align: u64,
    pub unbuffered: bool,
    pub read_gbs: f64,
    pub write_gbs: f64,
    pub write_chunk: u64,
}

/// The T2 IO backend: alignment and chunk geometry measured at open, reads
/// issued at depth, writes issued at the chunk size this device likes.
pub struct Backend {
    class: DeviceClass,
    pool: IoPool,
    /// Every open handle, per segment (see [`Handles`]).
    handles: Mutex<HashMap<PathBuf, Arc<Handles>>>,
}

impl Backend {
    /// Probe `dir`'s device and build a backend for it. `Err` means the
    /// device cannot serve a KV tier at all - the caller declines T2 loudly
    /// rather than serving restores that lose to recompute.
    pub fn open(dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let vkey = volume_key(dir);
        let cached = PROBED.lock().expect("probe cache").get(&vkey).copied();
        let class = match cached {
            Some(c) => c,
            None => {
                let c = probe_device(dir)?;
                PROBED.lock().expect("probe cache").insert(vkey, c);
                c
            }
        };
        tracing::info!(
            align = class.align,
            unbuffered = class.unbuffered,
            read_gbs = format!("{:.2}", class.read_gbs),
            write_gbs = format!("{:.2}", class.write_gbs),
            write_chunk_kib = class.write_chunk / 1024,
            depth = READ_DEPTH,
            "KV T2 device probed"
        );
        if class.read_gbs < GOOD_READ_GBS {
            tracing::warn!(
                read_gbs = format!("{:.2}", class.read_gbs),
                "KV T2 device is slow for a cache tier - restores will often \
                 lose the cost-model election to recompute, and the tier will \
                 mostly sit idle. Measured on YOUR device at open, not a \
                 default; the election itself is then re-measured continuously."
            );
        }
        Ok(Self {
            class,
            pool: IoPool::new(READ_DEPTH.saturating_sub(1)),
            handles: Mutex::new(HashMap::new()),
        })
    }

    pub fn class(&self) -> DeviceClass {
        self.class
    }

    pub fn align(&self) -> u64 {
        self.class.align
    }

    pub fn unbuffered(&self) -> bool {
        self.class.unbuffered
    }

    /// Read `buf.len()` bytes at `off`, split across the queue. `buf` must be
    /// alignment-sized and alignment-addressed when the backend is unbuffered
    /// (see [`AlignedBuf`]); `len` must be an aligned multiple.
    pub fn read_at(&self, path: &Path, buf: &mut [u8], off: u64) -> io::Result<()> {
        self.run_chunks(
            path,
            buf.as_mut_ptr(),
            buf.len(),
            off,
            false,
            self.read_chunk(buf.len()),
        )
    }

    /// Write `buf` at `off` in this device's preferred chunk size. Writes run
    /// at depth 1 by election: the probe measured the read-side depth ladder but not
    /// the write side, and guessing at depth on the scarce resource is how
    /// you turn a 2.5 GB/s device into a 0.4 GB/s one (fact 2's contention
    /// collapse). The write-QD ladder is queued for the next probe rev.
    pub fn write_at(&self, path: &Path, buf: &[u8], off: u64) -> io::Result<()> {
        let f = self.handles(path)?;
        let g = f.file(0)?;
        let chunk = self.class.write_chunk as usize;
        let mut done = 0usize;
        while done < buf.len() {
            let n = chunk.min(buf.len() - done);
            pwrite_all(g, &buf[done..done + n], off + done as u64)?;
            done += n;
        }
        Ok(())
    }

    pub fn sync(&self, path: &Path) -> io::Result<()> {
        self.handles(path)?.file(0)?.sync_data()
    }

    /// Read all of `buf` at `off` on async reader `reader`'s own handle slot
    /// (`t2_reads`): one positioned read, no pool - the readers ARE the depth.
    /// `buf` must be alignment-sized and -addressed when unbuffered.
    pub fn read_on(&self, path: &Path, buf: &mut [u8], off: u64, reader: usize) -> io::Result<()> {
        let f = self.handles(path)?;
        pread_exact(f.file(READ_DEPTH + reader)?, buf, off)
    }

    /// Close every handle on a segment (the file is about to be deleted, and
    /// its space only comes back once the last descriptor is gone). IO calls
    /// block until all their chunks report, so no job still holds these.
    pub fn forget(&self, path: &Path) {
        self.handles.lock().expect("segment handles").remove(path);
    }

    /// Chunk size for a request of `len`: at least 1 MiB, at most 4 MiB, and
    /// small enough that the request actually fills the queue. A 2 MiB extent
    /// runs as 2x1 MiB at depth 2 rather than one 2 MiB read at depth 1 -
    /// the ladder says the depth is worth more than the chunk size.
    fn read_chunk(&self, len: usize) -> u64 {
        let per_slot = align_up((len as u64).div_ceil(READ_DEPTH as u64), self.class.align);
        per_slot.clamp(READ_CHUNK_MIN, READ_CHUNK_MAX)
    }

    fn handles(&self, path: &Path) -> io::Result<Arc<Handles>> {
        let mut g = self.handles.lock().expect("segment handles");
        if let Some(f) = g.get(path) {
            return Ok(Arc::clone(f));
        }
        let f = Arc::new(Handles {
            path: path.to_path_buf(),
            unbuffered: self.class.unbuffered,
            slots: std::array::from_fn(|_| std::sync::OnceLock::new()),
        });
        // the first write creates the file: open the caller's slot now so an
        // unwritable path fails here, at the write that needs it
        f.file(0)?;
        g.insert(path.to_path_buf(), Arc::clone(&f));
        Ok(f)
    }

    /// Split [off, off+len) into chunks, run chunk 0 on this thread (the
    /// caller is a free queue slot) and the rest on the pool, then wait.
    fn run_chunks(
        &self,
        path: &Path,
        ptr: *mut u8,
        len: usize,
        off: u64,
        write: bool,
        chunk: u64,
    ) -> io::Result<()> {
        if len == 0 {
            return Ok(());
        }
        let file = self.handles(path)?;
        let (tx, rx) = mpsc::channel::<io::Result<()>>();
        let mut queued = 0usize;
        let mut pos = chunk.min(len as u64) as usize; // chunk 0 stays here
        while pos < len {
            let n = (chunk as usize).min(len - pos);
            // SAFETY: disjoint sub-range of the caller's region; the caller
            // blocks below until this job reports.
            let jp = SendPtr(unsafe { ptr.add(pos) });
            let job = Job {
                file: Arc::clone(&file),
                ptr: jp,
                len: n,
                off: off + pos as u64,
                write,
                done: tx.clone(),
            };
            if !self.pool.submit(job) {
                break; // pool gone: the tail runs inline below
            }
            queued += 1;
            pos += n;
        }
        // this thread's share
        let mine = file.file(0).and_then(|g| {
            let n = (chunk as usize).min(len);
            // SAFETY: chunk 0 of the caller's region, disjoint from queued.
            let b = unsafe { std::slice::from_raw_parts_mut(ptr, n) };
            if write {
                pwrite_all(g, b, off)
            } else {
                pread_exact(g, b, off)
            }
        });
        // anything the pool refused
        let mut tail = Ok(());
        if pos < len {
            // SAFETY: the un-queued tail, disjoint from every other chunk.
            let b = unsafe { std::slice::from_raw_parts_mut(ptr.add(pos), len - pos) };
            tail = file.file(0).and_then(|g| {
                if write {
                    pwrite_all(g, b, off + pos as u64)
                } else {
                    pread_exact(g, b, off + pos as u64)
                }
            });
        }
        drop(tx);
        let mut first_err = mine.err().or(tail.err());
        for _ in 0..queued {
            match rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => first_err = first_err.or(Some(e)),
                Err(_) => {
                    first_err = first_err.or(Some(io::Error::other("io worker vanished mid-chunk")))
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Discover alignment, then ladder the device so its geometry is a MEASURED
/// fact rather than a compiled-in guess.
fn probe_device(dir: &Path) -> io::Result<DeviceClass> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join("device.probe");
    let _ = std::fs::remove_file(&path);

    // 0. alignment ladder: the smallest granularity an unbuffered handle
    // accepts. If none works, the device (or filesystem) refuses direct IO -
    // fall back to buffered, which is correct, just slower.
    let mut align = 0u64;
    let mut unbuffered = false;
    if let Ok(f) = open_segment(&path, true) {
        for cand in ALIGN_CANDIDATES {
            let mut b = AlignedBuf::new(cand as usize, cand);
            if pwrite_all(&f, b.slice(), 0).is_ok() && pread_exact(&f, b.slice_mut(), 0).is_ok() {
                align = cand;
                unbuffered = true;
                break;
            }
        }
    }
    if !unbuffered {
        align = 4096;
        tracing::warn!(
            dir = %dir.display(),
            "unbuffered IO unavailable on this path - T2 falls back to buffered \
             IO (correct, but the page cache will hold a second copy of the \
             cache and the depth ladder flattens)"
        );
    }

    // 1. write ladder: pick the chunk this device is fastest at. rcraid gets
    // slower with size, bare NVMe usually faster - measuring is the only way
    // to be right on both.
    let f = open_segment(&path, unbuffered)?;
    let mut write_chunk = WRITE_CHUNK_CANDIDATES[0];
    let mut write_gbs = 0.0f64;
    for cand in WRITE_CHUNK_CANDIDATES {
        let chunk = align_up(cand, align);
        let buf = AlignedBuf::new(chunk as usize, align);
        let reps = PROBE_BYTES.div_ceil(chunk);
        let mut best = 0.0f64;
        for _ in 0..PROBE_REPS {
            let t0 = Instant::now();
            for i in 0..reps {
                pwrite_all(&f, buf.slice(), i * chunk)?;
            }
            f.sync_data()?;
            best = best.max((reps * chunk) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        if best > write_gbs {
            write_gbs = best;
            write_chunk = chunk;
        }
    }

    // 2. read rung, shaped like the real workload: a restore reads a handful
    // of multi-MiB extents at depth, so that is what we measure. Sequential
    // QD1 would flatter a rotational device exactly where it is worst, and
    // would under-report an SSD by the 10x the depth ladder is worth.
    let chunk = align_up(READ_CHUNK_MAX, align);
    let slots = (PROBE_BYTES / chunk).max(1);
    let mut read_gbs = 0.0f64;
    for rep in 0..PROBE_REPS {
        let t0 = Instant::now();
        std::thread::scope(|sc| -> io::Result<()> {
            let mut hs = Vec::new();
            for t in 0..READ_DEPTH {
                let path = &path;
                hs.push(sc.spawn(move || -> io::Result<()> {
                    let f = open_segment(path, unbuffered)?;
                    let mut b = AlignedBuf::new(chunk as usize, align);
                    // deterministic scatter: every slot read once per rung,
                    // start point rotated per worker and per rep
                    for i in 0..slots {
                        let slot = (i + (t as u64 + rep as u64 * 3) * 7) % slots;
                        pread_exact(&f, b.slice_mut(), slot * chunk)?;
                    }
                    Ok(())
                }));
            }
            for h in hs {
                h.join()
                    .map_err(|_| io::Error::other("probe worker panicked"))??;
            }
            Ok(())
        })?;
        let gbs = (slots * chunk * READ_DEPTH as u64) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        read_gbs = read_gbs.max(gbs);
    }

    drop(f);
    let _ = std::fs::remove_file(&path);

    if read_gbs < MIN_VIABLE_READ_GBS {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "device at {} reads {read_gbs:.2} GB/s - under the {MIN_VIABLE_READ_GBS:.2} \
                 GB/s floor, which is not storage a cache tier can live on (a failing disk, a \
                 network share, or a throttled mount). Point [kv_offload] nvme_path at local \
                 SSD storage, or leave nvme_gb unset and run the RAM tier alone.",
                dir.display()
            ),
        ));
    }
    Ok(DeviceClass {
        align,
        unbuffered,
        read_gbs,
        write_gbs,
        write_chunk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pdk-io-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn probe_discovers_a_workable_geometry() {
        let d = tdir("probe");
        let b = Backend::open(&d).expect("temp dir is a real device");
        let c = b.class();
        assert!(
            ALIGN_CANDIDATES.contains(&c.align),
            "align {} off the ladder",
            c.align
        );
        assert!(c.read_gbs > 0.0 && c.write_gbs > 0.0);
        assert!(
            WRITE_CHUNK_CANDIDATES
                .iter()
                .any(|&w| align_up(w, c.align) == c.write_chunk)
        );
        // the probe file never survives
        assert!(!d.join("device.probe").exists());
        std::fs::remove_dir_all(&d).ok();
    }

    /// The chunked+queued read must reassemble exactly what the write laid
    /// down - this is the property the whole restore path rests on.
    #[test]
    fn queued_reads_reassemble_the_extent() {
        let d = tdir("roundtrip");
        let b = Backend::open(&d).unwrap();
        let a = b.align();
        let p = d.join("seg.dat");
        for len_mib in [1u64, 3, 9] {
            let len = align_up(len_mib << 20, a) as usize;
            let mut src = AlignedBuf::new(len, a);
            for (i, v) in src.slice_mut().iter_mut().enumerate() {
                *v = (i.wrapping_mul(31) ^ len) as u8;
            }
            b.write_at(&p, src.slice(), 0).unwrap();
            b.sync(&p).unwrap();
            let mut dst = AlignedBuf::new(len, a);
            b.read_at(&p, dst.slice_mut(), 0).unwrap();
            assert_eq!(dst.slice(), src.slice(), "{len_mib} MiB round trip");
        }
        drop(b);
        std::fs::remove_dir_all(&d).ok();
    }

    /// Depth only pays if a request is actually split. A 9 MiB read must
    /// become >1 chunk; a 1 MiB read must stay whole (the floor).
    #[test]
    fn chunking_fills_the_queue_without_going_under_the_floor() {
        let d = tdir("chunks");
        let b = Backend::open(&d).unwrap();
        assert_eq!(
            b.read_chunk(1 << 20),
            READ_CHUNK_MIN,
            "1 MiB stays one chunk"
        );
        let c = b.read_chunk(9 << 20);
        assert!((READ_CHUNK_MIN..=READ_CHUNK_MAX).contains(&c));
        assert!(
            (9 << 20) / c as usize >= 2,
            "9 MiB must spread across the queue"
        );
        assert_eq!(
            b.read_chunk(64 << 20),
            READ_CHUNK_MAX,
            "big reads cap at the ceiling"
        );
        drop(b);
        std::fs::remove_dir_all(&d).ok();
    }

    /// Offsets are honoured per chunk - an off-by-one in the queue split
    /// would silently return neighbouring extents' bytes.
    #[test]
    fn reads_at_an_offset_land_on_the_right_bytes() {
        let d = tdir("offset");
        let b = Backend::open(&d).unwrap();
        let a = b.align();
        let p = d.join("seg.dat");
        let ext = align_up(6 << 20, a) as usize;
        let mut first = AlignedBuf::new(ext, a);
        first.slice_mut().fill(0xAA);
        let mut second = AlignedBuf::new(ext, a);
        second.slice_mut().fill(0x55);
        b.write_at(&p, first.slice(), 0).unwrap();
        b.write_at(&p, second.slice(), ext as u64).unwrap();
        b.sync(&p).unwrap();
        let mut got = AlignedBuf::new(ext, a);
        b.read_at(&p, got.slice_mut(), ext as u64).unwrap();
        assert!(got.slice().iter().all(|&v| v == 0x55));
        drop(b);
        std::fs::remove_dir_all(&d).ok();
    }
}
