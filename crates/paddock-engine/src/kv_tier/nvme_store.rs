//! T2 persistent store - the on-disk half of the tier, implementing
//!  exactly (deviations go back
//! into that spec first). Two invariants outrank everything else:
//!
//! 1. **Never misread.** Every byte handed back verifies against its commit
//!    record; every structure is versioned; recovery stops at the first torn
//!    record and discards everything after it.
//! 2. **Bounded-loss durability, absolute correctness.** Losing the last
//!    seconds of demotes at power loss is acceptable and REPORTED; a
//!    misread after any fault is not acceptable, ever.
//!
//! This module is the store CORE: layout, ordering, recovery, GC - plain
//! `std::fs` with explicit data syncs, driven synchronously. The elected
//! platform IO (unbuffered/overlapped Windows, io_uring + O_DIRECT Linux,
//! deep queue depths) arrives as the Phase-3.2 transport that CALLS this
//! core; buffered-with-fsync is a spec-sanctioned mode ("re-stages through
//! buffered IO with a loud warning"), and the commit ORDERING - data flush
//! before commit append before publish - is identical in both, which is
//! what the fault matrix actually tests.
//!
//! Fault injection: every ordering step consults a test-only kill hook, so
//! rows 1-10 and 15-21 of the preregistered matrix run mechanically as unit
//! tests (`kill -9 at step X` ≡ returning early at that step and reopening
//! the directory). Row 22 (real power loss) stays a scripted run on a
//! sacrificial machine.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::io::{AlignedBuf, Backend, align_up as io_align_up};

mod admit;
use admit::{VolumeRoom, WriteGate};

use super::Loc;
use super::digest::{CacheNamespace, PrivacyScope};

const MAGIC: u32 = 0x504b_5632; // "PKV2"
const FORMAT_VERSION: u16 = 1;
const KIND_SUPER: u16 = 1;
const KIND_COMMIT: u16 = 2;
const KIND_TOMBSTONE: u16 = 3;
const KIND_CKPT: u16 = 4;

const SUPER_SLOT: u64 = 4096;
/// Segment roll size. Extents are 2-16 MiB, so ~64-512 per segment.
const SEG_MAX: u64 = 1 << 30;
/// v1 stamped alignment (buffered mode needs none for correctness; the
/// Phase-3.2 direct-IO backend replaces this with the discovery ladder and
/// re-stamps).
const IO_ALIGN: u32 = 4096;

/// Test-only kill points - each names the state the "process died here"
/// fault leaves on disk (matrix rows 1-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kill {
    AfterDataWrite,        // row 1: data written (maybe cached), no flush
    AfterDataFlush,        // row 2: data durable, no commit record
    MidWalAppend,          // row 3: torn commit record
    AfterWalAppend,        // row 4: record written, not flushed
    MidCheckpoint,         // row 5: temp checkpoint half-written
    MidSuperblock,         // row 6: one slot mid-write
    MidCompaction,         // row 7: died between re-commits and the delete
    BeforeSegDelete,       // row 8: compaction committed, delete not yet run
    ShortDataWrite,        // row 9: device wrote only part of the payload
    DiskFullData,          // row 12: no space left while writing payload bytes
    DiskFullWal,           // row 13: no space left while appending the commit
    PermissionLost,        // row 14: the tree stopped being writable mid-run
    ShortWalAppend,        // row 23: the disk filled part-way through a commit append
    EvictBeforeTombstones, // row 24: evicted segment deleted, tombstones not yet out
}

impl Kill {
    /// The `io::Error` this fault presents as. Rows 12-14 are not our own
    /// invariants failing - they are the filesystem refusing, and the store
    /// has to behave identically whether the refusal is injected or real.
    fn as_io(self) -> Option<std::io::Error> {
        match self {
            Kill::DiskFullData | Kill::DiskFullWal | Kill::ShortWalAppend => Some(
                std::io::Error::new(std::io::ErrorKind::StorageFull, "no space left on device"),
            ),
            Kill::PermissionLost => Some(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "permission denied",
            )),
            _ => None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("store is locked by another process (lock file busy)")]
    Locked,
    #[error(
        "format version {found} is newer than this engine speaks ({FORMAT_VERSION}) - refusing (never partial-read)"
    )]
    VersionSkew { found: u16 },
    #[error("scope quota exhausted: {used} of {quota} bytes")]
    QuotaExhausted { used: u64, quota: u64 },
    /// The volume is down to its free-space reserve: `cap` is all the store
    /// may hold right now (`make_room` evicts toward it).
    #[error("volume free-space reserve reached: {used} held, room for {cap} bytes")]
    VolumeFull { used: u64, cap: u64 },
    /// Writes are paused after repeated device failures (see `admit`).
    #[error("disk writes paused after repeated failures (probing)")]
    Paused,
    /// A failed append could not be rolled back, so the manifest tail is
    /// unknown - any later record could land behind a torn one, and boot
    /// replay stops at the tear. Writes stop until a restart recovers it.
    #[error("manifest tail unknown after a failed append - writes stop until restart")]
    ManifestBroken,
    #[error("payload failed its commit checksum at read (at-rest corruption) - entry tombstoned")]
    Integrity,
    #[error("unknown key")]
    NotFound,
    #[error("killed at test hook {0:?}")]
    Killed(Kill),
}

impl StoreError {
    /// Refused before touching the device - quota, volume reserve or paused
    /// writes. Capacity, not a fault: callers keep the RAM copy and count
    /// nothing as an IO failure.
    pub fn is_capacity(&self) -> bool {
        matches!(
            self,
            Self::QuotaExhausted { .. } | Self::VolumeFull { .. } | Self::Paused
        )
    }
}

/// One committed extent's record - the WAL Commit payload and the
/// checkpoint's row format (identical bytes, one codec).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitRec {
    pub key: [u8; 32],
    pub generation: u64,
    pub schema_version: u16,
    pub segment: u32,
    pub offset: u64,
    pub len: u64,
    /// The tier's DOMAIN-SEPARATED payload hash (`Checksum::of_payload`,
    /// blake3 derive-key) - one hash domain across catalog, T1 and disk, so
    /// a preloaded entry's reference equals what any tier read reports.
    pub payload_checksum: [u8; 32],
}

const COMMIT_BODY: usize = 32 + 8 + 2 + 4 + 8 + 8 + 32; // 94
const REC_HEADER: usize = 8; // magic u32, version u16, kind u16
const REC_TAG: usize = 8; // blake3-short over header+body

impl CommitRec {
    fn encode_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.key);
        out.extend_from_slice(&self.generation.to_le_bytes());
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        out.extend_from_slice(&self.segment.to_le_bytes());
        out.extend_from_slice(&self.offset.to_le_bytes());
        out.extend_from_slice(&self.len.to_le_bytes());
        out.extend_from_slice(&self.payload_checksum);
    }

    fn decode_body(b: &[u8]) -> Self {
        let mut key = [0u8; 32];
        key.copy_from_slice(&b[0..32]);
        let mut payload_checksum = [0u8; 32];
        payload_checksum.copy_from_slice(&b[62..94]);
        Self {
            key,
            generation: u64::from_le_bytes(b[32..40].try_into().expect("8 bytes")),
            schema_version: u16::from_le_bytes(b[40..42].try_into().expect("2 bytes")),
            segment: u32::from_le_bytes(b[42..46].try_into().expect("4 bytes")),
            offset: u64::from_le_bytes(b[46..54].try_into().expect("8 bytes")),
            len: u64::from_le_bytes(b[54..62].try_into().expect("8 bytes")),
            payload_checksum,
        }
    }
}

fn tag8(bytes: &[u8]) -> [u8; 8] {
    let h = blake3::hash(bytes);
    h.as_bytes()[..8].try_into().expect("8 bytes")
}

fn record(kind: u16, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(REC_HEADER + body.len() + REC_TAG);
    out.extend_from_slice(&MAGIC.to_le_bytes());
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(body);
    let t = tag8(&out);
    out.extend_from_slice(&t);
    out
}

/// Parse one record at `b`; `Ok(Some((kind, body, total_len)))`, `Ok(None)`
/// for a clean end (all zero / empty), `Err(())` for a torn/invalid record
/// (replay stops and truncates there).
fn parse_record(b: &[u8]) -> Result<Option<(u16, &[u8], usize)>, ()> {
    if b.is_empty() || b.iter().take(REC_HEADER).all(|&x| x == 0) {
        return Ok(None);
    }
    if b.len() < REC_HEADER {
        return Err(());
    }
    if u32::from_le_bytes(b[0..4].try_into().expect("4 bytes")) != MAGIC {
        return Err(());
    }
    let version = u16::from_le_bytes(b[4..6].try_into().expect("2 bytes"));
    if version > FORMAT_VERSION {
        // an individual newer record inside an accepted store: treat as torn
        // (the superblock gate below refuses whole-store skew loudly)
        return Err(());
    }
    let kind = u16::from_le_bytes(b[6..8].try_into().expect("2 bytes"));
    let body_len = match kind {
        KIND_COMMIT => COMMIT_BODY,
        KIND_TOMBSTONE => 32 + 8,
        _ => return Err(()),
    };
    let total = REC_HEADER + body_len + REC_TAG;
    if b.len() < total {
        return Err(());
    }
    let expect: [u8; 8] = b[total - REC_TAG..total].try_into().expect("8 bytes");
    if tag8(&b[..total - REC_TAG]) != expect {
        return Err(());
    }
    Ok(Some((kind, &b[REC_HEADER..REC_HEADER + body_len], total)))
}

/// What recovery found - reporting, not just a boolean.
#[derive(Debug, Clone, Default)]
pub struct RecoveryReport {
    pub recovered_entries: u64,
    pub replayed_wal_records: u64,
    pub discarded_tail_records: u64,
    pub orphaned_bytes: u64,
    pub fresh_store: bool,
    /// Committed entries whose segment file is gone - an eviction that died
    /// after deleting the segment but before its tombstones were durable
    /// (row 24). Dropped, never served.
    pub lost_to_missing_segments: u64,
    /// Both superblock slots were invalid on a NON-empty directory - the
    /// cache was reset (loud alarm; a cache, not an archive).
    pub reset_after_corruption: bool,
    pub recovery_ms: u64,
}

/// One payload for [`NvmeStore::store_many`].
pub struct StoreItem<'a> {
    pub key: [u8; 32],
    pub generation: u64,
    pub schema_version: u16,
    pub payload: &'a [u8],
    /// The payload's `Checksum::of_payload` when the caller already has it
    /// (the tier hashed it off the engine thread when it entered T1), so the
    /// write does not hash it again. A wrong one cannot mislead: every read
    /// verifies the bytes against it, and a mismatch tombstones the entry.
    pub checksum: Option<[u8; 32]>,
}

/// Where a key's extent lives and what it must hash to (`read_plan`).
pub struct ReadPlan {
    pub path: PathBuf,
    pub off: u64,
    pub len: u64,
    /// `len` rounded up to the device's IO alignment - what a read moves.
    pub padded: u64,
    pub align: u64,
    pub checksum: [u8; 32],
}

pub struct StoreStats {
    pub live_entries: u64,
    pub live_bytes: u64,
    pub dead_bytes: u64,
    pub segments: u32,
    pub wal_bytes: u64,
    pub quota: u64,
}

/// Verified aligned read storage. Keep the I/O allocation as the immutable
/// payload rather than copying a hundreds-of-MiB checkpoint into a second Vec.
pub struct VerifiedRead {
    buffer: AlignedBuf,
    len: usize,
}
impl AsRef<[u8]> for VerifiedRead {
    fn as_ref(&self) -> &[u8] {
        &self.buffer.slice()[..self.len]
    }
}

pub struct NvmeStore {
    dir: PathBuf,
    _lock: File,
    wal: File,
    wal_len: u64,
    /// WAL offset covered by the checkpoint the superblock names.
    wal_committed_offset: u64,
    epoch: u64,
    /// Segment numbers we know about. The FILE handles live in the IO
    /// backend, which keeps one per worker so the device sees real queue
    /// depth.
    segments: HashSet<u32>,
    /// Platform IO for segment payloads: unbuffered, positioned, queued,
    /// with this device's geometry measured at open. WAL/superblock/
    /// checkpoint stay on ordinary buffered handles - they are small,
    /// ordering-critical, and flush-dominated, so direct IO
    /// would cost alignment padding on every record for no bandwidth.
    io: std::sync::Arc<Backend>,
    open_seg: u32,
    /// Append cursor in the open segment (aligned).
    cursor: u64,
    live: HashMap<[u8; 32], CommitRec>,
    dead: HashMap<u32, u64>,
    /// Payload bytes of `live` and of `dead`, kept as running sums: the
    /// quota is checked on every write, and summing 50K records per check
    /// was the old way.
    live_total: u64,
    dead_total: u64,
    quota: u64,
    /// Back-pressure from failed device writes.
    gate: WriteGate,
    /// The volume's free-space reserve - the store's second ceiling.
    room: VolumeRoom,
    /// Daily write budget, sized from the volume at open (`day_budget`).
    day_budget: u64,
    /// A failed append could not be rolled back (see `ManifestBroken`).
    wal_broken: bool,
    /// Unflushed WAL bytes (group commit); tombstones ride the next flush.
    wal_dirty: bool,
    pub integrity_failures: u64,
    /// Test-only kill hook.
    kill: Option<Kill>,
}

/// Make the cache tree owner-only. What is stored here is derived from
/// prompts, and it now OUTLIVES the process - activations are not plaintext,
/// but inversion research is far enough along that "it is only numbers" is
/// not a defence worth resting on. On unix the default umask commonly leaves
/// new directories world-readable; on Windows the data directory is already
/// user-scoped and the inherited ACL is the right one, so this is a no-op
/// there rather than a hand-rolled DACL.
///
/// Best effort by design: a filesystem that cannot express the mode (a
/// mounted share, FAT) must not stop the cache from working.
fn restrict_to_owner(dir: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for p in [dir, &dir.join("segments")] {
            if let Err(e) = std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700)) {
                tracing::debug!(path = %p.display(), err = %e, "KV cache: could not set owner-only mode");
            }
        }
    }
    #[cfg(not(unix))]
    let _ = dir;
}

fn seg_path(dir: &Path, seg: u32) -> PathBuf {
    dir.join("segments").join(format!("{seg:06}.seg"))
}

fn scope_hex(scope: &PrivacyScope) -> String {
    match scope {
        PrivacyScope::Shared => "shared".to_string(),
        PrivacyScope::PerUser(tag) => {
            let h = blake3::hash(tag);
            h.to_hex()[..32].to_string()
        }
    }
}

impl NvmeStore {
    /// Delete every cache namespace under `root` EXCEPT `keep`, and every one
    /// older than `max_age` regardless.
    ///
    /// A namespace is keyed by model identity, so changing a model's file,
    /// its context size, or its KV dtype strands the previous directory
    /// forever - correct (it can never be adopted again, which is the point
    /// of the identity) but unbounded. Nothing else in the system knows those
    /// directories exist, so the store reaps its own: on open, the runner
    /// tells it which namespace is live and how long an unused one may
    /// linger. Returns (directories removed, bytes reclaimed).
    ///
    /// Deliberately conservative: only paths under `root/kv-cache` that look
    /// like a namespace (32 hex characters) are ever considered, so pointing
    /// `nvme_path` at a populated directory can never delete a user's files.
    pub fn sweep_stale(root: &Path, keep: &Path, max_age: std::time::Duration) -> (usize, u64) {
        let base = root.join("kv-cache");
        let Ok(rd) = std::fs::read_dir(&base) else {
            return (0, 0);
        };
        let now = std::time::SystemTime::now();
        let (mut n, mut bytes) = (0usize, 0u64);
        for e in rd.flatten() {
            let p = e.path();
            let named_like_a_namespace = p
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()));
            if !named_like_a_namespace || !p.is_dir() {
                continue;
            }
            // `keep` is the live namespace's store dir (root/kv-cache/<id>/<scope>)
            if keep.starts_with(&p) {
                continue;
            }
            let idle = e
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| now.duration_since(t).ok())
                .unwrap_or_default();
            if idle < max_age {
                continue;
            }
            let sz = {
                fn walk(p: &Path) -> u64 {
                    let Ok(rd) = std::fs::read_dir(p) else {
                        return 0;
                    };
                    rd.flatten()
                        .map(|e| match e.file_type() {
                            Ok(t) if t.is_dir() => walk(&e.path()),
                            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
                        })
                        .sum()
                }
                walk(&p)
            };
            match std::fs::remove_dir_all(&p) {
                Ok(()) => {
                    n += 1;
                    bytes += sz;
                    tracing::info!(
                        namespace = %p.file_name().unwrap_or_default().to_string_lossy(),
                        idle_days = idle.as_secs() / 86_400,
                        reclaimed_mib = sz / (1 << 20),
                        "KV cache: retired a stale namespace (its model is gone or changed)"
                    );
                }
                // a live second runner holds its own lock file; leave it
                Err(e) => tracing::debug!(path = %p.display(), err = %e, "KV cache sweep skipped"),
            }
        }
        (n, bytes)
    }

    /// The store directory for a namespace under `data_dir`.
    pub fn dir_for(data_dir: &Path, ns: &CacheNamespace) -> PathBuf {
        let id_hex: String = ns.identity.0.iter().map(|b| format!("{b:02x}")).collect();
        data_dir
            .join("kv-cache")
            .join(&id_hex[..32])
            .join(scope_hex(&ns.scope))
    }

    /// Open (creating if absent) and RECOVER the store at `dir`. One opener
    /// at a time - a second process gets [`StoreError::Locked`], never a
    /// shared write.
    pub fn open(dir: &Path, quota: u64) -> Result<(Self, RecoveryReport), StoreError> {
        let t0 = std::time::Instant::now();
        std::fs::create_dir_all(dir.join("segments"))?;
        restrict_to_owner(dir);
        // probe the device before committing to it. The same
        // machine offers a 15 GB/s tier and a 0.25 GB/s trap one drive
        // letter apart, and a tier on the trap serves restores that lose to
        // recompute - an honest refusal beats a silent pessimisation.
        // row 5: a checkpoint interrupted before its rename leaves a temp
        // file; the superblock still names the old checkpoint, so the temp
        // is garbage - GC it
        // exclusive lock: Windows share_mode(0) is real OS exclusion; on
        // unix this is advisory create-or-open (the manager already ensures
        // one runner per store - this catches the accident loudly on the
        // platform where accidents happen most)
        let lock = {
            let mut o = OpenOptions::new();
            o.read(true).write(true).create(true);
            #[cfg(windows)]
            {
                use std::os::windows::fs::OpenOptionsExt;
                o.share_mode(0);
            }
            match o.open(dir.join("lock")) {
                Ok(f) => f,
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                    return Err(StoreError::Locked);
                }
                Err(e) if e.raw_os_error() == Some(32) => return Err(StoreError::Locked),
                Err(e) => return Err(e.into()),
            }
        };

        // macOS/Linux need actual process exclusion too. Acquire before probes,
        // recovery, or removing a writer's temporary checkpoint.
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: flock only consumes this live owned descriptor.
            if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                return Err(StoreError::Locked);
            }
        }
        let io = std::sync::Arc::new(Backend::open(&dir.join("segments"))?);
        let _ = std::fs::remove_file(dir.join("index.ckpt.tmp"));

        let mut report = RecoveryReport::default();

        // ---- superblock: two 4 KiB slots, higher valid epoch wins --------
        let meta_path = dir.join("store.meta");
        let mut meta = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&meta_path)?;
        let mut meta_bytes = vec![0u8; (SUPER_SLOT * 2) as usize];
        let n = meta.read(&mut meta_bytes)?;
        meta_bytes.resize((SUPER_SLOT * 2) as usize, 0);
        let parse_slot = |b: &[u8]| -> Option<(u64, u16, u64, u64)> {
            // {header, epoch u64, format u16, ckpt_len u64, wal_off u64,
            //  align u32, flags u32, tag8}
            if u32::from_le_bytes(b[0..4].try_into().expect("4 bytes")) != MAGIC {
                return None;
            }
            if u16::from_le_bytes(b[6..8].try_into().expect("2 bytes")) != KIND_SUPER {
                return None;
            }
            let body_end = REC_HEADER + 8 + 2 + 8 + 8 + 4 + 4;
            let expect: [u8; 8] = b[body_end..body_end + 8].try_into().ok()?;
            if tag8(&b[..body_end]) != expect {
                return None;
            }
            let epoch = u64::from_le_bytes(b[8..16].try_into().expect("8 bytes"));
            let format = u16::from_le_bytes(b[16..18].try_into().expect("2 bytes"));
            let ckpt_len = u64::from_le_bytes(b[18..26].try_into().expect("8 bytes"));
            let wal_off = u64::from_le_bytes(b[26..34].try_into().expect("8 bytes"));
            Some((epoch, format, ckpt_len, wal_off))
        };
        let a = parse_slot(&meta_bytes[..SUPER_SLOT as usize]);
        let b = parse_slot(&meta_bytes[SUPER_SLOT as usize..]);
        let best = match (a, b) {
            (Some(x), Some(y)) => Some(if x.0 >= y.0 { x } else { y }),
            (Some(x), None) => Some(x),
            (None, Some(y)) => Some(y),
            (None, None) => None,
        };
        if let Some((_, format, _, _)) = best
            && format > FORMAT_VERSION
        {
            return Err(StoreError::VersionSkew { found: format });
        }
        let had_data = n > 0 || dir.join("manifest.wal").exists();
        let (epoch, ckpt_len, mut wal_committed_offset) = match best {
            Some((e, _, cl, wo)) => (e, cl, wo),
            None => {
                if had_data {
                    report.reset_after_corruption = true;
                    tracing::error!(
                        dir = %dir.display(),
                        "KV store superblock unrecoverable - RESETTING the cache \
                         (a cache, not an archive); all prior entries are lost"
                    );
                    // segments/WAL/ckpt are garbage without a superblock
                    let _ = std::fs::remove_file(dir.join("manifest.wal"));
                    let _ = std::fs::remove_file(dir.join("index.ckpt"));
                    if let Ok(rd) = std::fs::read_dir(dir.join("segments")) {
                        for e in rd.flatten() {
                            let _ = std::fs::remove_file(e.path());
                        }
                    }
                } else {
                    report.fresh_store = true;
                }
                (0, 0, 0)
            }
        };

        // ---- checkpoint: the folded live set -----------------------------
        let mut live: HashMap<[u8; 32], CommitRec> = HashMap::new();
        if ckpt_len > 0 {
            let ok = (|| -> Option<()> {
                let bytes = std::fs::read(dir.join("index.ckpt")).ok()?;
                if bytes.len() as u64 != ckpt_len {
                    return None;
                }
                if bytes.len() < REC_HEADER + 8 + REC_TAG {
                    return None;
                }
                if u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes")) != MAGIC
                    || u16::from_le_bytes(bytes[6..8].try_into().expect("2 bytes")) != KIND_CKPT
                {
                    return None;
                }
                let tag_at = bytes.len() - REC_TAG;
                let expect: [u8; 8] = bytes[tag_at..].try_into().ok()?;
                if tag8(&bytes[..tag_at]) != expect {
                    return None;
                }
                let count = u64::from_le_bytes(
                    bytes[REC_HEADER..REC_HEADER + 8]
                        .try_into()
                        .expect("8 bytes"),
                );
                let mut at = REC_HEADER + 8;
                for _ in 0..count {
                    if at + COMMIT_BODY > tag_at {
                        return None;
                    }
                    let rec = CommitRec::decode_body(&bytes[at..at + COMMIT_BODY]);
                    live.insert(rec.key, rec);
                    at += COMMIT_BODY;
                }
                Some(())
            })()
            .is_some();
            if !ok {
                // spec: bad checkpoint falls back to replaying the whole WAL
                tracing::warn!(
                    "KV store checkpoint failed verification - replaying the full WAL instead"
                );
                live.clear();
                wal_committed_offset = 0;
            }
        }

        // ---- WAL replay from wal_committed_offset ------------------------
        let wal_path = dir.join("manifest.wal");
        let wal_bytes = std::fs::read(&wal_path).unwrap_or_default();
        let mut at = (wal_committed_offset as usize).min(wal_bytes.len());
        let mut wal_valid_end = at;
        loop {
            match parse_record(&wal_bytes[at..]) {
                Ok(None) => break,
                Ok(Some((kind, body, total))) => {
                    match kind {
                        KIND_COMMIT => {
                            let rec = CommitRec::decode_body(body);
                            live.insert(rec.key, rec);
                        }
                        KIND_TOMBSTONE => {
                            let mut key = [0u8; 32];
                            key.copy_from_slice(&body[0..32]);
                            live.remove(&key);
                        }
                        _ => unreachable!("parse_record filtered kinds"),
                    }
                    report.replayed_wal_records += 1;
                    at += total;
                    wal_valid_end = at;
                }
                Err(()) => {
                    // torn tail: everything before stands, everything after
                    // never happened - count what we discard
                    report.discarded_tail_records = 1; // at least the torn one
                    break;
                }
            }
        }
        // truncate the WAL to the valid prefix so the tear never re-parses
        let mut wal = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&wal_path)?;
        if wal_bytes.len() > wal_valid_end {
            wal.set_len(wal_valid_end as u64)?;
        }
        wal.seek(SeekFrom::Start(wal_valid_end as u64))?;

        // ---- allocator rebuild from the live set -------------------------
        // (in-flight-at-crash writes were uncommitted => their space is free
        // again; double allocation impossible because commit follows flush)
        let mut segments: HashSet<u32> = HashSet::new();
        let mut seg_files: Vec<(u32, std::path::PathBuf, u64)> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir.join("segments")) {
            for e in rd.flatten() {
                if let Some(n) = e
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.parse::<u32>().ok())
                {
                    let disk_len = e.metadata().map(|m| m.len()).unwrap_or(0);
                    seg_files.push((n, e.path(), disk_len));
                }
            }
        }
        // row 24: eviction deletes a segment BEFORE its tombstones go out (a
        // full disk has no room for them until the delete frees some), so a
        // crash in between leaves commits naming a file that is gone. Those
        // entries were evicted; serving them would only fail every read.
        let present: HashSet<u32> = seg_files.iter().map(|&(n, _, _)| n).collect();
        let orphan_commits: Vec<([u8; 32], u64)> = live
            .values()
            .filter(|r| !present.contains(&r.segment))
            .map(|r| (r.key, r.generation))
            .collect();
        for (key, _) in &orphan_commits {
            live.remove(key);
        }
        report.lost_to_missing_segments = orphan_commits.len() as u64;
        let mut seg_high: HashMap<u32, u64> = HashMap::new();
        let mut live_bytes_per_seg: HashMap<u32, u64> = HashMap::new();
        for rec in live.values() {
            let end = rec.offset + rec.len;
            let e = seg_high.entry(rec.segment).or_insert(0);
            if end > *e {
                *e = end;
            }
            *live_bytes_per_seg.entry(rec.segment).or_insert(0) += rec.len;
        }
        // the append head is the HIGHEST segment present - decide it before
        // the dead-segment sweep, or a fully-dead low-numbered segment hides
        // behind the running guess (fault row 8: delete retried at boot)
        let open_seg = seg_files
            .iter()
            .map(|&(n, _, _)| n)
            .max()
            .unwrap_or(1)
            .max(seg_high.keys().copied().max().unwrap_or(1));
        for (n, path, disk_len) in seg_files {
            segments.insert(n);
            let committed_end = seg_high.get(&n).copied().unwrap_or(0);
            if disk_len > committed_end {
                report.orphaned_bytes += disk_len - committed_end;
            }
            if committed_end == 0 && n != open_seg {
                // fully dead segment (compaction row 8: delete retried at
                // boot)
                let _ = std::fs::remove_file(&path);
                segments.remove(&n);
            }
        }
        let cursor = align_up(
            seg_high.get(&open_seg).copied().unwrap_or(0),
            IO_ALIGN as u64,
        );
        // reclaim orphan tails physically so cursor and file agree
        for (&segn, &high) in &seg_high {
            let p = seg_path(dir, segn);
            if let Ok(md) = std::fs::metadata(&p)
                && md.len() > high
            {
                let f = OpenOptions::new().write(true).open(&p)?;
                f.set_len(align_up(high, IO_ALIGN as u64))?;
            }
        }
        let dead: HashMap<u32, u64> = seg_high
            .iter()
            .map(|(&sn, &high)| {
                (
                    sn,
                    high - live_bytes_per_seg.get(&sn).copied().unwrap_or(0).min(high),
                )
            })
            .collect();

        report.recovered_entries = live.len() as u64;
        report.recovery_ms = t0.elapsed().as_millis() as u64;
        let live_total: u64 = live.values().map(|r| r.len).sum();
        let dead_total: u64 = dead.values().sum();
        // unit tests run on whatever volume holds the temp directory, so the
        // reserve is something they opt into with a fake (`set_space`)
        let mut space: admit::SpaceFn = if cfg!(test) {
            Box::new(|_| None)
        } else {
            Box::new(admit::volume_space)
        };
        let day_budget = admit::day_budget(space(dir).map(|(_, total)| total));
        let mut store = Self {
            dir: dir.to_path_buf(),
            io,
            _lock: lock,
            wal,
            wal_len: wal_valid_end as u64,
            wal_committed_offset,
            epoch,
            segments,
            open_seg,
            cursor,
            live,
            dead,
            live_total,
            dead_total,
            quota,
            gate: WriteGate::new(),
            room: VolumeRoom::new(space),
            day_budget,
            wal_broken: false,
            wal_dirty: false,
            integrity_failures: 0,
            kill: None,
        };
        // a fresh store stamps its first superblock so the next opener never
        // sees "no superblock + data" (which would read as corruption)
        if report.fresh_store {
            store.write_superblock(0, 0)?;
        }
        // and write the tombstones that eviction never got out, so a later
        // segment that reuses the number can never be named by them
        for (key, generation) in &orphan_commits {
            if store.append_tombstone(key, *generation).is_err() {
                break; // the next boot drops them again
            }
        }
        let _ = store.flush_group();
        Ok((store, report))
    }

    #[cfg(test)]
    pub fn set_kill(&mut self, k: Option<Kill>) {
        self.kill = k;
    }

    fn check_kill(&self, at: Kill) -> Result<(), StoreError> {
        if self.kill == Some(at) {
            return Err(StoreError::Killed(at));
        }
        Ok(())
    }

    /// This store's measured device geometry - the transport seeds the cost
    /// model's T2 bandwidth from it instead of guessing (an unseeded EWMA
    /// spends its first restores learning what open already measured).
    /// The namespace directory this store lives in.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn device(&self) -> super::io::DeviceClass {
        self.io.class()
    }

    /// Payload bytes the store holds on disk: live extents plus dead ones
    /// still sitting in their segments. The quota and the volume reserve are
    /// both checked against this - live bytes alone used to be the quota's
    /// measure, so a re-stored key's old extent was disk space nothing
    /// counted.
    fn footprint(&self) -> u64 {
        self.live_total + self.dead_total
    }

    /// The most the store may hold right now: its quota, or less when the
    /// volume is down to its reserve.
    fn limit(&mut self, now: std::time::Instant) -> u64 {
        let fp = self.footprint();
        self.quota.min(self.room.cap(&self.dir, now, fp))
    }

    /// The most the transports may write here in a UTC day - a quarter of
    /// the volume, at most 1 TiB (see `admit::day_budget`). Enforcing it is
    /// the writer's job: the store does not know when its day started.
    pub fn day_budget(&self) -> u64 {
        self.day_budget
    }

    /// Whether a write would be refused before the device right now -
    /// paused and not yet probe time, or the manifest broken. (At probe
    /// time this is false: the next write is the probe.)
    pub fn writes_blocked(&self) -> bool {
        self.wal_broken || self.gate.blocked(std::time::Instant::now())
    }

    /// Whether device writes are paused (repeated failures, probing).
    pub fn writes_paused(&self) -> bool {
        self.gate.paused() || self.wal_broken
    }

    #[cfg(test)]
    fn set_space(&mut self, probe: admit::SpaceFn) {
        self.room = VolumeRoom::new(probe);
    }

    /// Store one payload under `key` - the full ordering: reserve ->
    /// write -> data flush -> commit append -> WAL flush -> return. Completion
    /// implies durability (`durable-after-return`); the group-commit mode
    /// for background write-through arrives with the mirror-pass producer.
    ///
    /// Refusals that never touch the device come first: paused writes
    /// (`Paused` - before the others, so nobody evicts for a write that
    /// cannot run), then quota (`QuotaExhausted`) and the volume reserve
    /// (`VolumeFull`), both answered by `make_room`. A device error counts
    /// toward the pause.
    pub fn store(
        &mut self,
        key: [u8; 32],
        generation: u64,
        schema_version: u16,
        payload: &[u8],
    ) -> Result<Loc, StoreError> {
        let item = StoreItem {
            key,
            generation,
            schema_version,
            payload,
            checksum: None,
        };
        self.store_many(&[item]).map(|locs| locs[0])
    }

    /// Store several payloads as one group commit: every payload written,
    /// each segment they touched flushed once, then every commit record
    /// appended and the manifest flushed once - the same ordering as one
    /// [`Self::store`] (no commit before its data is durable), with two
    /// flushes per batch instead of two per extent. A chain written in one
    /// go (the tier's write-back at eviction) ran at ~0.4 GB/s on per-extent
    /// flushes. All or nothing: a failure publishes none of the batch.
    pub fn store_many(&mut self, items: &[StoreItem<'_>]) -> Result<Vec<Loc>, StoreError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if self.wal_broken {
            return Err(StoreError::ManifestBroken);
        }
        let now = std::time::Instant::now();
        if self.gate.blocked(now) {
            return Err(StoreError::Paused);
        }
        let len: u64 = items.iter().map(|i| i.payload.len() as u64).sum();
        let used = self.footprint();
        if used + len > self.quota {
            return Err(StoreError::QuotaExhausted {
                used,
                quota: self.quota,
            });
        }
        let cap = self.limit(now);
        if used + len > cap {
            return Err(StoreError::VolumeFull { used, cap });
        }
        if !self.gate.admit(now) {
            return Err(StoreError::Paused);
        }
        match self.store_io(items) {
            Ok(locs) => {
                self.gate.succeeded(now);
                self.room.set_short(false, used, cap);
                Ok(locs)
            }
            Err(StoreError::Io(e)) => {
                // the device says full while the sample said room: another
                // writer, or a quota statvfs cannot see - re-read before the
                // next write instead of trusting the sample's window
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::StorageFull | std::io::ErrorKind::QuotaExceeded
                ) {
                    self.room.invalidate();
                }
                self.gate.failed(now, &e);
                Err(StoreError::Io(e))
            }
            Err(e) => Err(e),
        }
    }

    fn store_io(&mut self, items: &[StoreItem<'_>]) -> Result<Vec<Loc>, StoreError> {
        // 1. allocate (in-memory only - nothing on disk references it yet)
        // and 2. write every payload. Unbuffered IO constrains length as well
        // as offset, so each tail is padded to the device alignment - the
        // commit record carries the true length, and offsets advance in
        // aligned strides, so padding is never read back as content.
        let a = self.io.align();
        let mut cursor = self.cursor;
        let mut placed: Vec<(u32, u64)> = Vec::with_capacity(items.len());
        let mut touched: Vec<u32> = Vec::new();
        for item in items {
            let n = item.payload.len() as u64;
            if cursor + n > SEG_MAX {
                self.open_seg += 1;
                cursor = 0;
            }
            let (seg, off) = (self.open_seg, cursor);
            // rows 12/14: the device refuses the payload write. Nothing is
            // committed, so the extent is simply never referenced - the
            // same shape as row 9's short write, reached by another error.
            if let Some(k) = self.kill
                && matches!(k, Kill::DiskFullData | Kill::PermissionLost)
                && let Some(e) = k.as_io()
            {
                return Err(StoreError::Io(e));
            }
            let path = seg_path(&self.dir, seg);
            self.segments.insert(seg);
            let padded = io_align_up(n, a) as usize;
            let mut buf = AlignedBuf::new(padded, a);
            buf.slice_mut()[..item.payload.len()].copy_from_slice(item.payload);
            if self.kill == Some(Kill::ShortDataWrite) {
                // row 9: the device accepted only part of the payload - the
                // commit is never issued, so the extent stays unreferenced
                let half = io_align_up(padded as u64 / 2, a).min(padded as u64) as usize;
                self.io.write_at(&path, &buf.slice()[..half], off)?;
                return Err(StoreError::Killed(Kill::ShortDataWrite));
            }
            self.io.write_at(&path, buf.slice(), off)?;
            placed.push((seg, off));
            if !touched.contains(&seg) {
                touched.push(seg);
            }
            cursor = align_up(off + n, IO_ALIGN as u64);
        }
        self.check_kill(Kill::AfterDataWrite)?;
        // 3. data flush before any commit record can exist - once per segment
        for &seg in &touched {
            self.io.sync(&seg_path(&self.dir, seg))?;
        }
        self.check_kill(Kill::AfterDataFlush)?;
        // 4. commit records, then one WAL flush
        let base = self.wal_len;
        let mut recs = Vec::with_capacity(items.len());
        for (item, &(segment, offset)) in items.iter().zip(&placed) {
            let rec = CommitRec {
                key: item.key,
                generation: item.generation,
                schema_version: item.schema_version,
                segment,
                offset,
                len: item.payload.len() as u64,
                payload_checksum: item
                    .checksum
                    .unwrap_or_else(|| super::digest::Checksum::of_payload(item.payload).0),
            };
            let mut body = Vec::with_capacity(COMMIT_BODY);
            rec.encode_body(&mut body);
            let bytes = record(KIND_COMMIT, &body);
            if self.kill == Some(Kill::MidWalAppend) {
                // torn append: half the record lands (row 3)
                let half = bytes.len() / 2;
                self.wal.write_all(&bytes[..half])?;
                self.wal.sync_data()?;
                return Err(StoreError::Killed(Kill::MidWalAppend));
            }
            // row 13: the payloads are durable by now (flushed above) but a
            // commit cannot be written. The extents stay unreferenced and
            // their space is reclaimed at the next boot as orphan tails -
            // existing entries are untouched, the row's stated requirement.
            if self.kill == Some(Kill::DiskFullWal)
                && let Some(e) = Kill::DiskFullWal.as_io()
            {
                self.wal_truncate_to(base);
                return Err(StoreError::Io(e));
            }
            if let Err(e) = self.wal_append(&bytes) {
                // none of the batch is published, so none of it may stay
                // in the manifest
                self.wal_truncate_to(base);
                return Err(e);
            }
            recs.push(rec);
        }
        self.check_kill(Kill::AfterWalAppend)?;
        if let Err(e) = self.wal.sync_data() {
            // not known durable, so not published - and the records must
            // not stay in the file either: these extents' space is reused
            // by the next write, which a surviving commit would then misname
            self.wal_truncate_to(base);
            return Err(StoreError::Io(e));
        }
        // 5. publish
        for rec in &recs {
            if let Some(old) = self.live.insert(rec.key, *rec) {
                *self.dead.entry(old.segment).or_insert(0) += old.len;
                self.live_total -= old.len;
                self.dead_total += old.len;
            }
            self.live_total += rec.len;
        }
        self.cursor = cursor;
        Ok(placed
            .into_iter()
            .map(|(seg, off)| loc_of(seg, off))
            .collect())
    }

    /// Append one whole record to the manifest, or none of it. A short
    /// append on a full disk used to leave a torn record behind while the
    /// process carried on: every later record landed after the tear, and
    /// boot replay stops AT the tear - so one ENOSPC silently discarded every
    /// commit and tombstone that followed it, at the next restart.
    fn wal_append(&mut self, bytes: &[u8]) -> Result<(), StoreError> {
        let res = if self.kill == Some(Kill::ShortWalAppend) {
            // row 23: part of the record fits, then the disk is full
            self.wal
                .write_all(&bytes[..bytes.len() / 2])
                .and_then(|()| Err(Kill::ShortWalAppend.as_io().expect("an io fault")))
        } else {
            self.wal.write_all(bytes)
        };
        match res {
            Ok(()) => {
                self.wal_len += bytes.len() as u64;
                Ok(())
            }
            Err(e) => Err(self.wal_rollback(e, 0)),
        }
    }

    /// Cut the manifest back to its last whole record after a failed append
    /// (`appended` = bytes of the failed record already counted in
    /// `wal_len`). If even that fails the tail is unknown and writes stop
    /// until a restart recovers it - better than appending behind a tear.
    fn wal_rollback(&mut self, e: std::io::Error, appended: u64) -> StoreError {
        self.wal_truncate_to(self.wal_len - appended);
        StoreError::Io(e)
    }

    /// Cut the manifest back to `len` bytes (the end of a whole record). If
    /// even that fails, the tail is unknown and writes stop until restart.
    fn wal_truncate_to(&mut self, len: u64) {
        self.wal_len = len;
        let healed = self
            .wal
            .set_len(len)
            .and_then(|()| self.wal.seek(SeekFrom::Start(len)));
        if let Err(h) = healed {
            self.wal_broken = true;
            tracing::error!(rollback = %h, "KV disk cache: manifest could not be rolled back after a failed append - disk writes stop until restart");
        }
    }

    /// Read + VERIFY a payload. A checksum mismatch tombstones the entry
    /// (never propagate a misread) and reports `Integrity`.
    pub fn read(&mut self, key: &[u8; 32]) -> Result<(u64, Vec<u8>), StoreError> {
        self.read_owned(key)
            .map(|(generation, bytes)| (generation, bytes.as_ref().to_vec()))
    }

    /// One allocation, no heap-to-heap payload copy. The checksum is verified
    /// before this owned storage can be published by any asynchronous caller.
    pub fn read_owned(&mut self, key: &[u8; 32]) -> Result<(u64, VerifiedRead), StoreError> {
        let rec = *self.live.get(key).ok_or(StoreError::NotFound)?;
        let a = self.io.align();
        let padded = io_align_up(rec.len, a) as usize;
        let mut ab = AlignedBuf::new(padded, a);
        self.io.read_at(
            &seg_path(&self.dir, rec.segment),
            ab.slice_mut(),
            rec.offset,
        )?;
        if super::digest::Checksum::of_payload(&ab.slice()[..rec.len as usize]).0
            != rec.payload_checksum
        {
            self.integrity_failures += 1;
            let _ = self.tombstone(key); // rides the next group flush
            tracing::error!(
                "KV store: extent failed its commit checksum (at-rest corruption) - tombstoned"
            );
            return Err(StoreError::Integrity);
        }
        Ok((
            rec.generation,
            VerifiedRead {
                buffer: ab,
                len: rec.len as usize,
            },
        ))
    }

    /// Read + VERIFY straight into a caller-owned buffer - the restore
    /// lane's path, where `dst` is the pinned staging slot itself, so a T2
    /// restore costs disk -> pinned -> GPU with no heap bounce in between.
    /// `dst` must be alignment-addressed and at least [`Self::padded_len`]
    /// long; callers that cannot promise that use [`Self::read`].
    ///
    /// SAFETY of the alignment contract is checked, not assumed: a
    /// misaligned or short buffer falls back to the allocating path rather
    /// than issuing an IO the device would reject.
    pub fn read_into(&mut self, key: &[u8; 32], dst: &mut [u8]) -> Result<(u64, u64), StoreError> {
        let rec = *self.live.get(key).ok_or(StoreError::NotFound)?;
        if dst.len() < rec.len as usize {
            return Err(StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "KV restore destination is shorter than the payload",
            )));
        }
        let a = self.io.align();
        let padded = io_align_up(rec.len, a) as usize;
        let aligned = (dst.as_ptr() as usize).is_multiple_of(a as usize);
        if !aligned || dst.len() < padded {
            let (generation, buf) = self.read(key)?;
            dst[..buf.len()].copy_from_slice(&buf);
            return Ok((generation, rec.len));
        }
        self.io.read_at(
            &seg_path(&self.dir, rec.segment),
            &mut dst[..padded],
            rec.offset,
        )?;
        if super::digest::Checksum::of_payload(&dst[..rec.len as usize]).0 != rec.payload_checksum {
            self.integrity_failures += 1;
            let _ = self.tombstone(key);
            tracing::error!(
                "KV store: extent failed its commit checksum (at-rest corruption) - tombstoned"
            );
            return Err(StoreError::Integrity);
        }
        Ok((rec.generation, rec.len))
    }

    /// Everything an off-thread reader needs to fetch and verify `key`
    /// (`t2_reads`): None when the key has no live extent.
    pub fn read_plan(&self, key: &[u8; 32]) -> Option<ReadPlan> {
        let r = self.live.get(key)?;
        let align = self.io.align();
        Some(ReadPlan {
            path: seg_path(&self.dir, r.segment),
            off: r.offset,
            len: r.len,
            padded: io_align_up(r.len, align),
            align,
            checksum: r.payload_checksum,
        })
    }

    /// The IO backend, for readers on other threads (`t2_reads`).
    pub fn reader_io(&self) -> std::sync::Arc<Backend> {
        std::sync::Arc::clone(&self.io)
    }

    /// An off-thread read found `key`'s bytes not matching their commit
    /// record: never serve it again (what `read_owned` does inline).
    pub fn mark_corrupt(&mut self, key: &[u8; 32]) {
        self.integrity_failures += 1;
        let _ = self.tombstone(key);
        tracing::error!(
            "KV store: extent failed its commit checksum (at-rest corruption) - tombstoned"
        );
    }

    /// Bytes a [`Self::read_into`] buffer must hold for `key` (the payload
    /// rounded up to this device's IO alignment).
    pub fn padded_len(&self, key: &[u8; 32]) -> Option<usize> {
        let rec = self.live.get(key)?;
        Some(io_align_up(rec.len, self.io.align()) as usize)
    }

    /// Everything the catalog needs to publish a durable copy as readable:
    /// (generation, location, payload length, commit checksum). None when
    /// this key has no live extent on disk.
    pub fn entry(&self, key: &[u8; 32]) -> Option<(u64, Loc, u64, [u8; 32])> {
        let r = self.live.get(key)?;
        Some((
            r.generation,
            loc_of(r.segment, r.offset),
            r.len,
            r.payload_checksum,
        ))
    }

    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.live.contains_key(key)
    }

    pub fn generation(&self, key: &[u8; 32]) -> Option<u64> {
        self.live.get(key).map(|r| r.generation)
    }

    /// The commit record's payload checksum - the catalog's integrity
    /// reference at preload (same blake3-of-payload the tier uses).
    pub fn commit_checksum(&self, key: &[u8; 32]) -> Option<[u8; 32]> {
        self.live.get(key).map(|r| r.payload_checksum)
    }

    /// Evict/TTL/Bad. Tombstones ride group flushes (loss at crash is
    /// harmless - the entry resurrects and is re-evicted).
    pub fn tombstone(&mut self, key: &[u8; 32]) -> Result<(), StoreError> {
        let Some(old) = self.live.remove(key) else {
            return Ok(());
        };
        *self.dead.entry(old.segment).or_insert(0) += old.len;
        self.live_total -= old.len;
        self.dead_total += old.len;
        self.append_tombstone(key, old.generation)
    }

    /// The tombstone RECORD alone - bookkeeping for an entry already gone
    /// from the live set.
    fn append_tombstone(&mut self, key: &[u8; 32], generation: u64) -> Result<(), StoreError> {
        let mut body = Vec::with_capacity(40);
        body.extend_from_slice(key);
        body.extend_from_slice(&generation.to_le_bytes());
        let bytes = record(KIND_TOMBSTONE, &body);
        self.wal_append(&bytes)?;
        self.wal_dirty = true;
        Ok(())
    }

    /// Flush pending group-commit records (tombstones).
    pub fn flush_group(&mut self) -> Result<(), StoreError> {
        if self.wal_dirty {
            self.wal.sync_data()?;
            self.wal_dirty = false;
        }
        Ok(())
    }

    fn write_superblock(&mut self, ckpt_len: u64, wal_off: u64) -> Result<(), StoreError> {
        self.epoch += 1;
        let mut body = Vec::new();
        body.extend_from_slice(&MAGIC.to_le_bytes());
        body.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        body.extend_from_slice(&KIND_SUPER.to_le_bytes());
        body.extend_from_slice(&self.epoch.to_le_bytes());
        body.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        body.extend_from_slice(&ckpt_len.to_le_bytes());
        body.extend_from_slice(&wal_off.to_le_bytes());
        body.extend_from_slice(&IO_ALIGN.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes()); // flags (encryption off)
        let t = tag8(&body);
        body.extend_from_slice(&t);
        body.resize(SUPER_SLOT as usize, 0);
        // write the other slot; a torn write always leaves one valid slot
        let slot = (self.epoch % 2) * SUPER_SLOT;
        let mut meta = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.dir.join("store.meta"))?;
        if self.kill == Some(Kill::MidSuperblock) {
            meta.seek(SeekFrom::Start(slot))?;
            meta.write_all(&body[..2048])?; // half a slot (row 6)
            meta.sync_data()?;
            return Err(StoreError::Killed(Kill::MidSuperblock));
        }
        meta.seek(SeekFrom::Start(slot))?;
        meta.write_all(&body)?;
        meta.sync_data()?;
        self.wal_committed_offset = wal_off;
        Ok(())
    }

    /// Fold the live set into a fresh checkpoint and logically truncate the
    /// WAL. Boot cost stays O(checkpoint + WAL tail), never a store scan.
    pub fn checkpoint(&mut self) -> Result<(), StoreError> {
        self.flush_group()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&KIND_CKPT.to_le_bytes());
        bytes.extend_from_slice(&(self.live.len() as u64).to_le_bytes());
        let mut recs: Vec<&CommitRec> = self.live.values().collect();
        recs.sort_by_key(|r| r.key);
        for r in recs {
            let mut b = Vec::with_capacity(COMMIT_BODY);
            r.encode_body(&mut b);
            bytes.extend_from_slice(&b);
        }
        let t = tag8(&bytes);
        bytes.extend_from_slice(&t);
        let tmp = self.dir.join("index.ckpt.tmp");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)?;
            if self.kill == Some(Kill::MidCheckpoint) {
                f.write_all(&bytes[..bytes.len() / 2])?;
                f.sync_data()?;
                return Err(StoreError::Killed(Kill::MidCheckpoint));
            }
            f.write_all(&bytes)?;
            f.sync_data()?;
        }
        std::fs::rename(&tmp, self.dir.join("index.ckpt"))?;
        let wal_off = self.wal_len;
        self.write_superblock(bytes.len() as u64, wal_off)?;
        Ok(())
    }

    /// Cache eviction (the store is a CACHE, not an archive): retire the
    /// OLDEST segments - log order, content neither re-stored nor compacted
    /// forward since it landed - until `bytes` fits under the store's limit:
    /// its quota, or less when the volume is down to its free-space reserve.
    /// Whole segments go, live and dead bytes alike (a segment of nothing
    /// but dead extents used to be invisible here). The open segment never
    /// evicts. A write bigger than everything the store may hold never fits,
    /// so the cache is not emptied for it - but the store still gives back
    /// whatever it holds over the limit. Returns the evicted keys so the
    /// caller can invalidate any references it holds - a dangling catalog
    /// entry costs a failed op AND a breaker count on every probe that
    /// elects it.
    pub fn make_room(&mut self, bytes: u64) -> Result<Vec<[u8; 32]>, StoreError> {
        let now = std::time::Instant::now();
        let limit = self.limit(now);
        let target = if bytes <= limit {
            limit - bytes
        } else {
            // logged on the edge here, not per refusal in `store`: a refusal
            // that eviction answers is the cache working, not running short
            self.room.set_short(true, self.footprint(), limit);
            limit
        };
        let mut evicted = Vec::new();
        if self.footprint() <= target {
            return Ok(evicted);
        }
        // the open segment never evicts, so when it alone holds more than
        // the target it has to roll first (a sub-segment quota used to fill
        // its first segment and never evict it). Only then: rolling on
        // every call would give each extent a segment file of its own.
        if self.segment_bytes(self.open_seg) > target {
            self.open_seg += 1;
            self.cursor = 0;
        }
        let reason = if limit < self.quota {
            "volume reserve"
        } else {
            "quota"
        };
        while self.footprint() > target {
            let Some(victim) = self
                .segments
                .iter()
                .copied()
                .filter(|&s| s != self.open_seg)
                .min()
            else {
                break; // everything sits in the open segment
            };
            let keys = self.retire_segment(victim)?;
            tracing::info!(
                segment = victim,
                entries = keys.len(),
                reason,
                "T2 cache eviction: oldest segment retired"
            );
            evicted.extend(keys);
        }
        if limit < self.quota {
            self.room.yielded(now);
        }
        Ok(evicted)
    }

    /// Live plus dead payload bytes in one segment.
    fn segment_bytes(&self, seg: u32) -> u64 {
        let live: u64 = self
            .live
            .values()
            .filter(|r| r.segment == seg)
            .map(|r| r.len)
            .sum();
        live + self.dead.get(&seg).copied().unwrap_or(0)
    }

    /// Evict one whole segment: drop its entries, delete the file, THEN
    /// write their tombstones. On a full disk the tombstones may not fit
    /// until the delete has freed space - the old order (tombstones first)
    /// could refuse to evict exactly when the space was gone. A crash between the delete and the
    /// tombstones is row 24: boot drops commits whose segment is gone.
    fn retire_segment(&mut self, victim: u32) -> Result<Vec<[u8; 32]>, StoreError> {
        let gone: Vec<([u8; 32], u64, u64)> = self
            .live
            .values()
            .filter(|r| r.segment == victim)
            .map(|r| (r.key, r.generation, r.len))
            .collect();
        for &(key, _, len) in &gone {
            self.live.remove(&key);
            self.live_total -= len;
        }
        self.dead_total -= self.dead.remove(&victim).unwrap_or(0);
        self.segments.remove(&victim);
        let p = seg_path(&self.dir, victim);
        self.io.forget(&p);
        let _ = std::fs::remove_file(p);
        self.check_kill(Kill::EvictBeforeTombstones)?;
        for &(key, generation, _) in &gone {
            if let Err(e) = self.append_tombstone(&key, generation) {
                // harmless: the boot rule drops what these would have
                tracing::debug!(err = %e, "T2 eviction: tombstones deferred to boot");
                break;
            }
        }
        if let Err(e) = self.flush_group() {
            tracing::debug!(err = %e, "T2 eviction: tombstone flush failed");
        }
        Ok(gone.into_iter().map(|(k, _, _)| k).collect())
    }

    /// Compact a segment whose dead ratio crossed 1/2: copy live extents to
    /// the open segment (each re-commits through the WAL - the same ordering
    /// as any write), then delete the old file. Defers to the caller's
    /// read-slack scheduling; bytes count against the endurance budget.
    pub fn maybe_compact(&mut self) -> Result<bool, StoreError> {
        let victim = self
            .dead
            .iter()
            .filter(|&(&sn, _)| sn != self.open_seg)
            .find(|&(&sn, &dead)| {
                let live: u64 = self
                    .live
                    .values()
                    .filter(|r| r.segment == sn)
                    .map(|r| r.len)
                    .sum();
                dead > live
            })
            .map(|(&sn, _)| sn);
        let Some(victim) = victim else {
            return Ok(false);
        };
        let movers: Vec<CommitRec> = self
            .live
            .values()
            .filter(|r| r.segment == victim)
            .copied()
            .collect();
        for (moved, rec) in movers.into_iter().enumerate() {
            if self.kill == Some(Kill::MidCompaction) && moved > 0 {
                // row 7: died with some extents re-committed and some not -
                // the old segment's commits still stand for the unmoved
                // ones, the moved ones read from their new home; recovery
                // sees a consistent live set either way
                return Err(StoreError::Killed(Kill::MidCompaction));
            }
            let (_g, payload) = self.read(&rec.key)?;
            self.store(rec.key, rec.generation, rec.schema_version, &payload)?;
        }
        self.check_kill(Kill::BeforeSegDelete)?; // row 8: delete retried at boot
        self.segments.remove(&victim);
        self.dead_total -= self.dead.remove(&victim).unwrap_or(0);
        let p = seg_path(&self.dir, victim);
        self.io.forget(&p);
        let _ = std::fs::remove_file(p);
        Ok(true)
    }

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            live_entries: self.live.len() as u64,
            live_bytes: self.live_total,
            dead_bytes: self.dead_total,
            segments: self.segments.len() as u32,
            wal_bytes: self.wal_len,
            quota: self.quota,
        }
    }

    /// Total bytes this store occupies on disk, segments and metadata alike -
    /// what a user means by "how big is the cache", as opposed to `live_bytes`
    /// which counts only payload still referenced.
    pub fn disk_bytes(&self) -> u64 {
        fn walk(p: &Path) -> u64 {
            let Ok(rd) = std::fs::read_dir(p) else {
                return 0;
            };
            rd.flatten()
                .map(|e| match e.file_type() {
                    Ok(t) if t.is_dir() => walk(&e.path()),
                    _ => e.metadata().map(|m| m.len()).unwrap_or(0),
                })
                .sum()
        }
        walk(&self.dir)
    }

    /// Iterate the live records - what the catalog preloads as
    /// `Ready(generation, loc)` at boot (restart persistence). The payload
    /// checksum rides along: it is the catalog's integrity reference.
    pub fn live_iter(&self) -> impl Iterator<Item = (&[u8; 32], u64, Loc, u64, [u8; 32])> {
        self.live.iter().map(|(k, r)| {
            (
                k,
                r.generation,
                loc_of(r.segment, r.offset),
                r.len,
                r.payload_checksum,
            )
        })
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

fn loc_of(seg: u32, off: u64) -> Loc {
    Loc(((seg as u64) << 40) | off)
}

/// Decode a store `Loc` back into (segment, offset).
pub fn loc_parts(l: Loc) -> (u32, u64) {
    ((l.0 >> 40) as u32, l.0 & ((1u64 << 40) - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tdir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pkv-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn k(n: u8) -> [u8; 32] {
        [n; 32]
    }

    fn payload(n: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| n.wrapping_add(i as u8)).collect()
    }

    #[test]
    fn roundtrip_and_restart_persistence() {
        let d = tdir("rt");
        {
            let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
            assert!(r.fresh_store);
            s.store(k(1), 7, 1, &payload(1, 100_000)).unwrap();
            s.store(k(2), 8, 1, &payload(2, 5_000)).unwrap();
            s.store(k(3), 9, 1, &payload(3, 250_000)).unwrap();
            assert_eq!(s.read(&k(2)).unwrap(), (8, payload(2, 5_000)));
        }
        // The restart-persistence property, at store level
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 3);
        assert!(!r.fresh_store && !r.reset_after_corruption);
        assert_eq!(s.read(&k(1)).unwrap(), (7, payload(1, 100_000)));
        assert_eq!(s.read(&k(3)).unwrap(), (9, payload(3, 250_000)));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn owned_read_excludes_padding_and_short_destinations_fail_without_writes() {
        let d = tdir("owned-read");
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        let data = payload(9, 50_003);
        s.store(k(9), 17, 1, &data).unwrap();
        let (generation, owned) = s.read_owned(&k(9)).unwrap();
        assert_eq!(generation, 17);
        assert_eq!(owned.as_ref(), data);
        let mut short = vec![121; 50_002];
        assert!(
            matches!(s.read_into(&k(9), &mut short), Err(StoreError::Io(e)) if e.kind()==std::io::ErrorKind::InvalidInput)
        );
        assert!(short.iter().all(|b| *b == 121));
        drop(s);
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(s.read_owned(&k(9)).unwrap().1.as_ref(), data);
        drop(s);
        drop(owned);
        std::fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn rows_1_2_uncommitted_data_is_orphaned_and_reclaimed() {
        for kill in [Kill::AfterDataWrite, Kill::AfterDataFlush] {
            let d = tdir(&format!("r12-{kill:?}"));
            {
                let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
                s.store(k(1), 1, 1, &payload(1, 8_000)).unwrap();
                s.set_kill(Some(kill));
                assert!(matches!(
                    s.store(k(2), 2, 1, &payload(2, 64_000)),
                    Err(StoreError::Killed(_))
                ));
            }
            let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
            assert_eq!(r.recovered_entries, 1, "{kill:?}: only the committed entry");
            assert!(
                !s.contains(&k(2)),
                "{kill:?}: uncommitted extent unreferenced"
            );
            assert!(r.orphaned_bytes > 0, "{kill:?}: orphan reported");
            // the orphan space is allocatable again and everything verifies
            s.store(k(3), 3, 1, &payload(3, 64_000)).unwrap();
            assert_eq!(s.read(&k(3)).unwrap().1, payload(3, 64_000));
            assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 8_000));
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    #[test]
    fn row_3_torn_wal_record_stops_replay_at_the_tear() {
        let d = tdir("r3");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 10_000)).unwrap();
            s.set_kill(Some(Kill::MidWalAppend));
            assert!(matches!(
                s.store(k(2), 2, 1, &payload(2, 10_000)),
                Err(StoreError::Killed(_))
            ));
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 1);
        assert!(r.discarded_tail_records >= 1, "the tear is counted");
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 10_000));
        // and the truncated WAL accepts new commits cleanly
        s.store(k(4), 4, 1, &payload(4, 1_000)).unwrap();
        drop(s);
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 2);
        assert_eq!(s.read(&k(4)).unwrap().1, payload(4, 1_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_4_unflushed_commit_is_consistent_either_way() {
        let d = tdir("r4");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.set_kill(Some(Kill::AfterWalAppend));
            let _ = s.store(k(1), 1, 1, &payload(1, 10_000));
        }
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        // the record may or may not have survived; both outcomes must be
        // consistent - if present, it must verify
        if s.contains(&k(1)) {
            assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 10_000));
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_5_kill_mid_checkpoint_keeps_the_old_view() {
        let d = tdir("r5");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 20_000)).unwrap();
            s.store(k(2), 2, 1, &payload(2, 20_000)).unwrap();
            s.set_kill(Some(Kill::MidCheckpoint));
            assert!(matches!(s.checkpoint(), Err(StoreError::Killed(_))));
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 2, "WAL replay covers everything");
        assert!(!d.join("index.ckpt.tmp").exists(), "temp checkpoint GC-ed");
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 20_000));
        assert_eq!(s.read(&k(2)).unwrap().1, payload(2, 20_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_6_kill_mid_superblock_boots_from_the_other_slot() {
        let d = tdir("r6");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 20_000)).unwrap();
            s.checkpoint().unwrap();
            s.store(k(2), 2, 1, &payload(2, 20_000)).unwrap();
            s.set_kill(Some(Kill::MidSuperblock));
            assert!(matches!(s.checkpoint(), Err(StoreError::Killed(_))));
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(
            r.recovered_entries, 2,
            "older slot + WAL tail cover everything"
        );
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 20_000));
        assert_eq!(s.read(&k(2)).unwrap().1, payload(2, 20_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_15_extent_corruption_tombstones_never_propagates() {
        let d = tdir("r15");
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        s.store(k(1), 1, 1, &payload(1, 50_000)).unwrap();
        // flip one byte of the at-rest extent
        let seg = seg_path(&d, 1);
        let mut bytes = std::fs::read(&seg).unwrap();
        bytes[100] ^= 0xff;
        std::fs::write(&seg, &bytes).unwrap();
        assert!(matches!(s.read(&k(1)), Err(StoreError::Integrity)));
        assert_eq!(s.integrity_failures, 1);
        assert!(!s.contains(&k(1)), "tombstoned, never served");
        s.flush_group().unwrap();
        drop(s);
        let (s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 0, "the poisoned entry stays gone");
        drop(s);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_16_wal_corruption_discards_the_tail_only() {
        let d = tdir("r16");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 10_000)).unwrap();
            s.store(k(2), 2, 1, &payload(2, 10_000)).unwrap();
        }
        // corrupt the last wal record tag
        let wal = d.join("manifest.wal");
        let mut bytes = std::fs::read(&wal).unwrap();
        let n = bytes.len();
        bytes[n - 1] ^= 0xff;
        std::fs::write(&wal, &bytes).unwrap();
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 1, "first stands, tail discarded");
        assert!(r.discarded_tail_records >= 1);
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 10_000));
        assert!(!s.contains(&k(2)));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_17_checkpoint_corruption_falls_back_to_full_replay() {
        let d = tdir("r17");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 10_000)).unwrap();
            s.checkpoint().unwrap();
        }
        let ck = d.join("index.ckpt");
        let mut bytes = std::fs::read(&ck).unwrap();
        bytes[10] ^= 0xff;
        std::fs::write(&ck, &bytes).unwrap();
        // the WAL was LOGICALLY truncated at checkpoint, so the fallback
        // replays it from 0 - the commit records are still physically there
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 1, "full WAL replay recovers");
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 10_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn rows_18_19_superblock_slots() {
        // one slot corrupt -> other wins
        let d = tdir("r18");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 10_000)).unwrap();
            s.checkpoint().unwrap();
        }
        {
            let meta = d.join("store.meta");
            let mut bytes = std::fs::read(&meta).unwrap();
            bytes[9] ^= 0xff; // corrupt slot A's epoch field
            std::fs::write(&meta, &bytes).unwrap();
        }
        {
            let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
            assert_eq!(r.recovered_entries, 1, "other slot + replay recover");
            assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 10_000));
        }
        // both slots corrupt -> reset + loud, store still OPENS (a cache)
        {
            let meta = d.join("store.meta");
            let mut bytes = std::fs::read(&meta).unwrap();
            for off in [1usize, 4097] {
                if off < bytes.len() {
                    bytes[off] ^= 0xff;
                }
            }
            std::fs::write(&meta, &bytes).unwrap();
        }
        let (s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert!(r.reset_after_corruption, "reset is reported, not silent");
        assert_eq!(r.recovered_entries, 0);
        drop(s);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_20_version_skew_refuses() {
        let d = tdir("r20");
        {
            let (_s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        }
        // hand-craft a valid slot with a NEWER format version
        let mut body = Vec::new();
        body.extend_from_slice(&MAGIC.to_le_bytes());
        body.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        body.extend_from_slice(&KIND_SUPER.to_le_bytes());
        body.extend_from_slice(&99u64.to_le_bytes()); // epoch (wins)
        body.extend_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&0u64.to_le_bytes());
        body.extend_from_slice(&IO_ALIGN.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        let t = tag8(&body);
        body.extend_from_slice(&t);
        body.resize(SUPER_SLOT as usize, 0);
        let meta = d.join("store.meta");
        let mut bytes = std::fs::read(&meta).unwrap();
        bytes.resize((SUPER_SLOT * 2) as usize, 0);
        bytes[..SUPER_SLOT as usize].copy_from_slice(&body);
        std::fs::write(&meta, &bytes).unwrap();
        assert!(matches!(
            NvmeStore::open(&d, 1 << 30),
            Err(StoreError::VersionSkew { .. })
        ));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[cfg(windows)]
    #[test]
    fn row_21_second_opener_is_refused() {
        let d = tdir("r21");
        let s1 = NvmeStore::open(&d, 1 << 30).unwrap();
        assert!(matches!(
            NvmeStore::open(&d, 1 << 30),
            Err(StoreError::Locked)
        ));
        drop(s1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn tombstone_compaction_and_segment_reclaim() {
        let d = tdir("gc");
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        s.store(k(1), 1, 1, &payload(1, 200_000)).unwrap();
        s.store(k(2), 2, 1, &payload(2, 50_000)).unwrap();
        // force the open segment forward so seg 1 becomes a victim candidate
        s.open_seg += 1;
        s.cursor = 0;
        s.tombstone(&k(1)).unwrap();
        s.flush_group().unwrap();
        assert!(s.maybe_compact().unwrap(), "dead-majority segment compacts");
        assert_eq!(
            s.read(&k(2)).unwrap().1,
            payload(2, 50_000),
            "live extent moved"
        );
        assert!(!seg_path(&d, 1).exists(), "old segment deleted");
        drop(s);
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 1);
        assert_eq!(s.read(&k(2)).unwrap().1, payload(2, 50_000));
        assert!(!s.contains(&k(1)));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_7_kill_mid_compaction_stays_consistent() {
        let d = tdir("r7");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 100_000)).unwrap();
            s.store(k(2), 2, 1, &payload(2, 100_000)).unwrap();
            s.store(k(3), 3, 1, &payload(3, 250_000)).unwrap();
            s.open_seg += 1;
            s.cursor = 0;
            s.tombstone(&k(3)).unwrap(); // majority dead in seg 1
            s.flush_group().unwrap();
            s.set_kill(Some(Kill::MidCompaction));
            assert!(matches!(s.maybe_compact(), Err(StoreError::Killed(_))));
        }
        // recovery: some extents moved, some not - every live key readable
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 2);
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 100_000));
        assert_eq!(s.read(&k(2)).unwrap().1, payload(2, 100_000));
        // and compaction can run to completion now
        while s.maybe_compact().unwrap() {}
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 100_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_8_kill_before_segment_delete_retries_at_boot() {
        let d = tdir("r8");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 100_000)).unwrap();
            s.store(k(2), 2, 1, &payload(2, 200_000)).unwrap();
            s.open_seg += 1;
            s.cursor = 0;
            s.tombstone(&k(2)).unwrap();
            s.flush_group().unwrap();
            s.set_kill(Some(Kill::BeforeSegDelete));
            assert!(matches!(s.maybe_compact(), Err(StoreError::Killed(_))));
            // the victim file still exists (delete never ran)
            assert!(seg_path(&d, 1).exists());
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 1);
        // the re-committed copy lives in the new segment; the fully-dead old
        // one was deleted at boot (row 8's retried delete)
        assert!(!seg_path(&d, 1).exists(), "dead segment deleted at boot");
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 100_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn row_9_short_data_write_never_commits() {
        let d = tdir("r9");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 10_000)).unwrap();
            s.set_kill(Some(Kill::ShortDataWrite));
            assert!(matches!(
                s.store(k(2), 2, 1, &payload(2, 50_000)),
                Err(StoreError::Killed(_))
            ));
            s.set_kill(None);
            // the failed op released nothing durable; a fresh store works
            s.store(k(3), 3, 1, &payload(3, 50_000)).unwrap();
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 2);
        assert!(!s.contains(&k(2)), "short-written extent never referenced");
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 10_000));
        assert_eq!(s.read(&k(3)).unwrap().1, payload(3, 50_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    // row 10 (short WAL write) is row 3's torn-record case - the replay
    // stops at the tear regardless of whether the shortfall came from a
    // torn append or a device short write; row_3 covers it. Row 11 (lost
    // completion) is the catalog race suite's drop_op class. Rows 12-14
    // (disk full / permission loss) surface as store() Io errors, which the
    // write gate counts - three in a row pause writes until a probe lands
    // (repeated_device_failures_pause_writes_until_a_probe_lands); a quota
    // analog is covered by quota_is_enforced_loudly. Row 23 is a short
    // commit append the process survives, row 24 an eviction that dies
    // before its tombstones.

    #[test]
    fn make_room_evicts_oldest_segments_durably() {
        let d = tdir("mr");
        let (mut s, _) = NvmeStore::open(&d, 600_000).unwrap();
        s.store(k(1), 1, 1, &payload(1, 200_000)).unwrap();
        s.open_seg += 1;
        s.cursor = 0;
        s.store(k(2), 2, 1, &payload(2, 200_000)).unwrap();
        // 400k live of 600k quota: admitting 300k must retire the oldest
        // segment (k1), not the newer one
        let ev = s.make_room(300_000).unwrap();
        assert_eq!(ev, vec![k(1)]);
        assert!(!s.contains(&k(1)));
        assert_eq!(
            s.read(&k(2)).unwrap().1,
            payload(2, 200_000),
            "newer content untouched"
        );
        s.store(k(3), 3, 1, &payload(3, 300_000)).unwrap();
        // nothing evictable outside the open segment -> loud quota, no loop
        s.open_seg += 1;
        s.cursor = 0;
        drop(s);
        let (mut s, r) = NvmeStore::open(&d, 600_000).unwrap();
        assert_eq!(r.recovered_entries, 2, "eviction survives restart");
        assert!(!s.contains(&k(1)));
        assert_eq!(s.read(&k(3)).unwrap().1, payload(3, 300_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The sweep deletes directories, so its restraint is the property worth
    /// testing: only namespace-shaped names, only under `kv-cache`, never the
    /// live one, and never anything younger than the TTL. A user who points
    /// `nvme_path` at a directory that already has files in it must get those
    /// files back untouched.
    #[test]
    fn the_sweep_only_ever_retires_its_own_stale_namespaces() {
        let root = tdir("sweep");
        let base = root.join("kv-cache");
        let live = base.join("aa".repeat(16)).join("shared");
        let stale = base.join("bb".repeat(16));
        let fresh = base.join("cc".repeat(16));
        let not_ours = base.join("my-important-directory");
        for d in [&live, &stale, &fresh, &not_ours] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("payload.bin"), vec![7u8; 4096]).unwrap();
        }
        // a file sitting directly in the cache root, not a directory at all
        std::fs::write(base.join("README.txt"), b"hands off").unwrap();

        // TTL of zero: everything eligible is stale, so only the EXCLUSIONS
        // can save a directory - which is exactly what we want to prove.
        let (n, bytes) = NvmeStore::sweep_stale(&root, &live, std::time::Duration::ZERO);

        assert!(live.exists(), "the live namespace must survive");
        assert!(
            not_ours.exists(),
            "a non-namespace name must never be touched"
        );
        assert!(
            base.join("README.txt").exists(),
            "a stray file must never be touched"
        );
        assert!(!stale.exists(), "a stale namespace should be retired");
        assert!(!fresh.exists(), "with a zero TTL, fresh is stale too");
        assert_eq!(n, 2, "exactly the two eligible namespaces");
        assert!(bytes >= 8192, "reclaimed bytes reported");

        // and a real TTL spares everything, because nothing here is old
        let (n2, _) = NvmeStore::sweep_stale(&root, &live, std::time::Duration::from_secs(3600));
        assert_eq!(n2, 0);
        std::fs::remove_dir_all(&root).ok();
    }

    /// `disk_bytes` answers "how big is this cache" including the metadata a
    /// payload-only count hides.
    #[test]
    fn disk_bytes_counts_the_whole_store() {
        let d = tdir("disk-bytes");
        let (mut st, _) = NvmeStore::open(&d, 1 << 20).unwrap();
        let empty = st.disk_bytes();
        st.store(k(1), 1, 1, &payload(1, 40_000)).unwrap();
        st.flush_group().unwrap();
        let after = st.disk_bytes();
        assert!(after > empty, "storing payload must grow the on-disk size");
        assert!(after >= 40_000, "the payload itself is on disk");
        drop(st);
        std::fs::remove_dir_all(&d).ok();
    }

    /// Rows 12 and 14: the filesystem refuses a payload write. The store
    /// must report it and commit nothing - a half-written extent that a
    /// commit record points at is the one outcome the protocol may never
    /// produce, whatever the device does.
    #[test]
    fn rows_12_14_a_refused_payload_write_never_commits() {
        for (row, kill) in [(12, Kill::DiskFullData), (14, Kill::PermissionLost)] {
            let d = tdir(&format!("row{row}"));
            let (mut st, _) = NvmeStore::open(&d, 1 << 20).unwrap();
            st.store(k(1), 1, 1, &payload(1, 8192)).unwrap();
            st.flush_group().unwrap();

            st.set_kill(Some(kill));
            let e = st.store(k(2), 1, 1, &payload(2, 8192)).unwrap_err();
            assert!(
                matches!(e, StoreError::Io(_)),
                "row {row}: expected an io error, got {e:?}"
            );
            st.set_kill(None);
            assert!(
                !st.contains(&k(2)),
                "row {row}: the refused write must not publish"
            );
            // the entry that was already there is untouched
            assert_eq!(st.read(&k(1)).unwrap().1, payload(1, 8192), "row {row}");
            drop(st);

            // and it survives the restart the same way
            let (mut st, rep) = NvmeStore::open(&d, 1 << 20).unwrap();
            assert!(st.contains(&k(1)), "row {row}: survivor recovered");
            assert!(
                !st.contains(&k(2)),
                "row {row}: the refused write stayed unpublished"
            );
            assert_eq!(rep.recovered_entries, 1, "row {row}");
            assert_eq!(st.read(&k(1)).unwrap().1, payload(1, 8192), "row {row}");
            drop(st);
            std::fs::remove_dir_all(&d).ok();
        }
    }

    /// Row 13: the payload lands durably but the COMMIT cannot be written.
    /// Existing entries stay readable, the new one does not exist, and the
    /// orphaned extent's space comes back at the next boot.
    #[test]
    fn row_13_a_refused_commit_leaves_an_orphan_and_nothing_else() {
        let d = tdir("row13");
        let (mut st, _) = NvmeStore::open(&d, 1 << 20).unwrap();
        st.store(k(1), 1, 1, &payload(1, 8192)).unwrap();
        st.flush_group().unwrap();

        st.set_kill(Some(Kill::DiskFullWal));
        let e = st.store(k(2), 1, 1, &payload(2, 8192)).unwrap_err();
        assert!(
            matches!(e, StoreError::Io(_)),
            "expected an io error, got {e:?}"
        );
        st.set_kill(None);
        assert!(!st.contains(&k(2)));
        drop(st);

        let (st, rep) = NvmeStore::open(&d, 1 << 20).unwrap();
        assert_eq!(
            rep.recovered_entries, 1,
            "only the committed entry comes back"
        );
        assert!(
            rep.orphaned_bytes > 0,
            "the uncommitted payload is reported as orphaned"
        );
        assert!(st.contains(&k(1)));
        assert!(!st.contains(&k(2)));
        drop(st);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn quota_is_enforced_loudly() {
        let d = tdir("q");
        let (mut s, _) = NvmeStore::open(&d, 100_000).unwrap();
        s.store(k(1), 1, 1, &payload(1, 90_000)).unwrap();
        assert!(matches!(
            s.store(k(2), 2, 1, &payload(2, 20_000)),
            Err(StoreError::QuotaExhausted { .. })
        ));
        // a tombstone frees no DISK space - the extent still sits in its
        // segment - so it frees no quota either; retiring the segment does
        s.tombstone(&k(1)).unwrap();
        assert!(matches!(
            s.store(k(2), 2, 1, &payload(2, 20_000)),
            Err(StoreError::QuotaExhausted { .. })
        ));
        assert!(
            s.make_room(20_000).unwrap().is_empty(),
            "nothing live to report"
        );
        assert_eq!(s.stats().dead_bytes, 0, "the dead extent's segment is gone");
        s.store(k(2), 2, 1, &payload(2, 20_000)).unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_re_stored_key_leaves_a_dead_extent_the_quota_counts() {
        let d = tdir("dead");
        let (mut s, _) = NvmeStore::open(&d, 120_000).unwrap();
        s.store(k(1), 1, 1, &payload(1, 50_000)).unwrap();
        s.store(k(1), 1, 1, &payload(1, 50_000)).unwrap();
        let st = s.stats();
        assert_eq!((st.live_bytes, st.dead_bytes), (50_000, 50_000));
        // live bytes alone (the old measure) would have let this in
        assert!(matches!(
            s.store(k(2), 2, 1, &payload(2, 30_000)),
            Err(StoreError::QuotaExhausted { used: 100_000, .. })
        ));
        assert_eq!(s.make_room(30_000).unwrap(), vec![k(1)]);
        s.store(k(2), 2, 1, &payload(2, 30_000)).unwrap();
        assert_eq!(s.stats().dead_bytes, 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    fn items<'a>(ps: &'a [Vec<u8>], first: u8) -> Vec<StoreItem<'a>> {
        ps.iter()
            .enumerate()
            .map(|(i, p)| StoreItem {
                key: k(first + i as u8),
                generation: 1,
                schema_version: 1,
                payload: p,
                checksum: Some(super::super::digest::Checksum::of_payload(p).0),
            })
            .collect()
    }

    /// A chain written as one group commit: two flushes for the batch, every
    /// extent published together, and all of it there after a restart.
    #[test]
    fn a_group_commit_lands_whole_and_survives_restart() {
        let d = tdir("gcommit");
        let ps: Vec<Vec<u8>> = (0..3u8).map(|n| payload(n, 50_000 + n as usize)).collect();
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            let locs = s.store_many(&items(&ps, 10)).unwrap();
            assert_eq!(locs.len(), 3);
            assert_eq!(s.stats().live_entries, 3);
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 3);
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(&s.read(&k(10 + i as u8)).unwrap().1, p);
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// All or nothing: a batch whose commit cannot be written publishes none
    /// of its extents, now or after a restart; what was there stays.
    #[test]
    fn a_group_commit_that_fails_publishes_nothing() {
        let d = tdir("gcfail");
        let ps: Vec<Vec<u8>> = (0..3u8).map(|n| payload(n, 20_000)).collect();
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 8192)).unwrap();
            s.set_kill(Some(Kill::DiskFullWal));
            assert!(matches!(
                s.store_many(&items(&ps, 20)),
                Err(StoreError::Io(_))
            ));
            s.set_kill(None);
            assert!((20..23).all(|n| !s.contains(&k(n))));
            // and the next batch lands where the failed one would have
            s.store_many(&items(&ps, 30)).unwrap();
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 4, "the survivor and the second batch");
        assert!((20..23).all(|n| !s.contains(&k(n))));
        assert_eq!(s.read(&k(31)).unwrap().1, ps[1]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The caller's checksum is trusted at write time, never at read time: a
    /// wrong one makes the entry unreadable (tombstoned), not misread.
    #[test]
    fn a_wrong_caller_checksum_can_only_fail_a_read() {
        let d = tdir("gcsum");
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        let p = payload(7, 30_000);
        let item = StoreItem {
            key: k(7),
            generation: 1,
            schema_version: 1,
            payload: &p,
            checksum: Some([0xab; 32]),
        };
        s.store_many(&[item]).unwrap();
        assert!(matches!(s.read(&k(7)), Err(StoreError::Integrity)));
        assert!(!s.contains(&k(7)));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Row 23: the disk fills part-way through a commit append and the
    /// process carries on. The torn half must not stay in the manifest -
    /// boot replay stops at a tear, and every later commit would be lost.
    #[test]
    fn row_23_a_short_commit_append_rolls_back_and_later_commits_survive() {
        let d = tdir("r23");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            s.store(k(1), 1, 1, &payload(1, 10_000)).unwrap();
            s.set_kill(Some(Kill::ShortWalAppend));
            assert!(matches!(
                s.store(k(2), 2, 1, &payload(2, 10_000)),
                Err(StoreError::Io(_))
            ));
            s.set_kill(None);
            assert!(!s.contains(&k(2)));
            s.store(k(3), 3, 1, &payload(3, 10_000)).unwrap();
            s.tombstone(&k(1)).unwrap();
            s.flush_group().unwrap();
        }
        let (mut s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.discarded_tail_records, 0, "no tear left behind");
        assert!(!s.contains(&k(1)), "the tombstone after the failure landed");
        assert!(!s.contains(&k(2)));
        assert_eq!(s.read(&k(3)).unwrap().1, payload(3, 10_000));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Row 24: eviction deletes the segment first (so a full disk has room
    /// for the tombstones) and dies before writing them. Boot must drop the
    /// commits that name the missing file - and tombstone them, so the next
    /// boot does not have to.
    #[test]
    fn row_24_an_eviction_that_dies_before_its_tombstones_is_dropped_at_boot() {
        let d = tdir("r24");
        {
            let (mut s, _) = NvmeStore::open(&d, 500_000).unwrap();
            s.store(k(1), 1, 1, &payload(1, 200_000)).unwrap();
            s.open_seg += 1;
            s.cursor = 0;
            s.store(k(2), 2, 1, &payload(2, 200_000)).unwrap();
            s.set_kill(Some(Kill::EvictBeforeTombstones));
            assert!(matches!(
                s.make_room(200_000),
                Err(StoreError::Killed(Kill::EvictBeforeTombstones))
            ));
            assert!(!seg_path(&d, 1).exists(), "the segment went first");
        }
        let (mut s, r) = NvmeStore::open(&d, 500_000).unwrap();
        assert_eq!(r.lost_to_missing_segments, 1);
        assert!(!s.contains(&k(1)), "never served from a file that is gone");
        assert_eq!(s.read(&k(2)).unwrap().1, payload(2, 200_000));
        drop(s);
        let (_s, r) = NvmeStore::open(&d, 500_000).unwrap();
        assert_eq!(r.lost_to_missing_segments, 0, "tombstoned at the last boot");
        assert_eq!(r.recovered_entries, 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The vLLM PR 61780 class: a disk that fails every write instantly must
    /// stop receiving them, and come back by itself once writes land again.
    #[test]
    fn repeated_device_failures_pause_writes_until_a_probe_lands() {
        let d = tdir("gate");
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        s.set_kill(Some(Kill::DiskFullData));
        for i in 0..3u8 {
            assert!(matches!(
                s.store(k(i), 1, 1, &payload(i, 8192)),
                Err(StoreError::Io(_))
            ));
        }
        assert!(s.writes_paused());
        // paused: refused before the device, and before quota/eviction work
        s.set_kill(None);
        assert!(matches!(
            s.store(k(9), 1, 1, &payload(9, 8192)),
            Err(StoreError::Paused)
        ));
        assert!(!s.contains(&k(9)));
        // the volume was "cleaned up": the first probe lands and resumes
        std::thread::sleep(std::time::Duration::from_millis(1050));
        s.store(k(9), 1, 1, &payload(9, 8192)).unwrap();
        assert!(!s.writes_paused());
        s.store(k(10), 1, 1, &payload(10, 8192)).unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The store never takes its volume's last free space: it holds what the
    /// volume can spare beyond its reserve, evicts toward that when other
    /// writers take space, and grows again when space comes back.
    #[test]
    fn the_volume_reserve_caps_the_store_and_eviction_gives_space_back() {
        use std::sync::{Arc, atomic::AtomicU64, atomic::Ordering::Relaxed};
        let d = tdir("room");
        let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
        // 1 MB volume: 100 KB reserve; 160 KB free -> room for 60 KB
        let avail = Arc::new(AtomicU64::new(160_000));
        let a = Arc::clone(&avail);
        s.set_space(Box::new(move |_| Some((a.load(Relaxed), 1_000_000))));
        s.store(k(1), 1, 1, &payload(1, 50_000)).unwrap();
        assert!(matches!(
            s.store(k(2), 2, 1, &payload(2, 20_000)),
            Err(StoreError::VolumeFull { cap: 60_000, .. })
        ));
        // FIFO under the volume's limit, exactly as under the quota
        assert_eq!(s.make_room(20_000).unwrap(), vec![k(1)]);
        s.store(k(2), 2, 1, &payload(2, 20_000)).unwrap();
        // another writer takes the rest of the volume: nothing fits, and the
        // store gives back what it holds over the new limit
        avail.store(0, Relaxed);
        s.room.invalidate();
        assert!(matches!(
            s.store(k(3), 3, 1, &payload(3, 10_000)),
            Err(StoreError::VolumeFull { .. })
        ));
        assert_eq!(s.make_room(10_000).unwrap(), vec![k(2)]);
        assert_eq!(s.stats().live_bytes, 0);
        // space comes back (the writer deleted its files): writes resume
        avail.store(500_000, Relaxed);
        s.room.invalidate();
        s.store(k(3), 3, 1, &payload(3, 10_000)).unwrap();
        let _ = std::fs::remove_dir_all(&d);
    }

    /// An evicted segment's space must come back when it is deleted. The IO
    /// workers used to keep a descriptor on every segment they had read from
    /// for the life of the process, and an open descriptor keeps a deleted
    /// file's blocks allocated - the cache's quota held while the volume
    /// filled anyway.
    #[cfg(target_os = "linux")]
    #[test]
    fn an_evicted_segment_that_served_restores_frees_its_space() {
        let d = tdir("fdleak");
        let (mut s, _) = NvmeStore::open(&d, 8 << 20).unwrap();
        // 3 MiB reads split into 1 MiB chunks, two of them on pool workers
        s.store(k(1), 1, 1, &payload(1, 3 << 20)).unwrap();
        s.open_seg += 1;
        s.cursor = 0;
        s.store(k(2), 2, 1, &payload(2, 3 << 20)).unwrap();
        assert_eq!(s.read(&k(1)).unwrap().1, payload(1, 3 << 20));
        assert_eq!(s.make_room(4 << 20).unwrap(), vec![k(1)]);
        assert!(!seg_path(&d, 1).exists());
        let victim = seg_path(&d, 1).to_string_lossy().to_string();
        let held: Vec<String> = std::fs::read_dir("/proc/self/fd")
            .unwrap()
            .flatten()
            .filter_map(|e| std::fs::read_link(e.path()).ok())
            .map(|t| t.to_string_lossy().to_string())
            .filter(|t| t.starts_with(&victim))
            .collect();
        assert!(held.is_empty(), "deleted segment still open: {held:?}");
        assert_eq!(s.read(&k(2)).unwrap().1, payload(2, 3 << 20));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn checkpoint_bounds_boot_cost() {
        let d = tdir("ck");
        {
            let (mut s, _) = NvmeStore::open(&d, 1 << 30).unwrap();
            for i in 0..20u8 {
                s.store(k(i), i as u64, 1, &payload(i, 4_000)).unwrap();
            }
            s.checkpoint().unwrap();
        }
        let (_s, r) = NvmeStore::open(&d, 1 << 30).unwrap();
        assert_eq!(r.recovered_entries, 20);
        assert_eq!(r.replayed_wal_records, 0, "boot = checkpoint + empty tail");
        let _ = std::fs::remove_dir_all(&d);
    }
}
