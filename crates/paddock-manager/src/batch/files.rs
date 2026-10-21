//! The bytes of a batch: uploads streamed to disk, items read back by offset,
//! results appended to files whose committed length the store records.
//!
//! Nothing here holds an input in memory. A 60 GB upload of base64 pictures
//! costs the manager one chunk at a time going in, and one item at a time
//! coming out.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use futures::{Stream, StreamExt};
use tokio::io::AsyncWriteExt;

/// Free space that must remain after a write: 4 GB, or 5% of the volume when
/// that is more. A batch that fills the disk takes the models dir and the
/// store down with it.
pub(crate) fn reserve(total: u64) -> u64 {
    (4u64 << 30).max(total / 20)
}

/// Refuse `incoming` more bytes on the volume holding `dir` when they would
/// leave less than [`reserve`]. Unknown space (a probe failure) is allowed:
/// the write itself is the final check.
pub(crate) fn headroom(dir: &Path, incoming: u64) -> Result<(), String> {
    let probe = std::iter::successors(Some(dir), |p| p.parent()).find(|p| p.exists());
    let Some(probe) = probe else { return Ok(()) };
    let (Ok(free), Ok(total)) = (fs4::available_space(probe), fs4::total_space(probe)) else {
        return Ok(());
    };
    let keep = reserve(total);
    if free.saturating_sub(incoming) < keep {
        return Err(format!(
            "not enough disk space: {} GB free, this needs {} GB and {} GB must stay free",
            free >> 30,
            incoming.div_ceil(1 << 30),
            keep >> 30
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) enum UploadError {
    /// The client's stream broke off.
    Body(String),
    /// Disk space or a write failed.
    Disk(String),
}

impl std::fmt::Display for UploadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Body(e) => write!(f, "upload interrupted: {e}"),
            Self::Disk(e) => f.write_str(e),
        }
    }
}

/// Stream `body` into `dest` through `dest.part`, renamed into place only once
/// complete, so a broken upload never leaves a file that looks whole. The
/// space check repeats every 256 MB: a Content-Length is a promise, the
/// volume is the truth. Returns the byte count.
pub(crate) async fn stream_to_file<S, B, E>(body: S, dest: &Path) -> Result<u64, UploadError>
where
    S: Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let part = part_path(dest);
    if let Some(dir) = dest.parent() {
        tokio::fs::create_dir_all(dir)
            .await
            .map_err(|e| UploadError::Disk(e.to_string()))?;
    }
    let result = write_part(body, &part).await;
    match result {
        Ok(n) => {
            tokio::fs::rename(&part, dest)
                .await
                .map_err(|e| UploadError::Disk(e.to_string()))?;
            Ok(n)
        }
        Err(e) => {
            let _ = tokio::fs::remove_file(&part).await;
            Err(e)
        }
    }
}

async fn write_part<S, B, E>(mut body: S, part: &Path) -> Result<u64, UploadError>
where
    S: Stream<Item = Result<B, E>> + Unpin,
    B: AsRef<[u8]>,
    E: std::fmt::Display,
{
    let disk = |e: std::io::Error| UploadError::Disk(e.to_string());
    let f = tokio::fs::File::create(part).await.map_err(disk)?;
    let mut w = tokio::io::BufWriter::with_capacity(1 << 20, f);
    let mut n: u64 = 0;
    let mut next_check: u64 = 0;
    while let Some(chunk) = body.next().await {
        let chunk = chunk.map_err(|e| UploadError::Body(e.to_string()))?;
        let bytes = chunk.as_ref();
        if n >= next_check {
            let dir = part.parent().unwrap_or(Path::new("."));
            headroom(dir, 256 << 20).map_err(UploadError::Disk)?;
            next_check = n + (256 << 20);
        }
        w.write_all(bytes).await.map_err(disk)?;
        n += bytes.len() as u64;
    }
    w.flush().await.map_err(disk)?;
    // durable before the rename publishes it: an input the store points at
    // must survive a power cut
    w.into_inner().sync_all().await.map_err(disk)?;
    Ok(n)
}

pub(crate) fn part_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// Read `len` bytes at `off` - one item's body.
pub(crate) fn read_at(path: &Path, off: u64, len: usize) -> std::io::Result<Vec<u8>> {
    let f = File::open(path)?;
    let mut buf = vec![0u8; len];
    read_exact_at(&f, &mut buf, off)?;
    Ok(buf)
}

#[cfg(unix)]
fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(f, buf, off)
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut buf: &mut [u8], mut off: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        match std::os::windows::fs::FileExt::seek_read(f, buf, off) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                off += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// An append-only result file. `len` is what has been written, `committed` is
/// what the store has recorded; opening truncates to the committed length,
/// which is the whole crash-recovery story for results (store/batches.rs).
pub(crate) struct ResultFile {
    file: File,
    pub(crate) len: u64,
    dirty: bool,
    #[cfg(test)]
    path: PathBuf,
}

/// Tests make the result writes under these directories fail, as a full disk
/// would (a per-directory switch: tests run side by side in one process).
#[cfg(test)]
pub(crate) static FAIL_WRITES_UNDER: std::sync::Mutex<Vec<PathBuf>> =
    std::sync::Mutex::new(Vec::new());

impl ResultFile {
    pub(crate) fn open(path: &Path, committed: u64) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;
        let on_disk = file.metadata()?.len();
        if on_disk < committed {
            // the store promises bytes the file does not have: refuse rather
            // than append after a hole
            return Err(std::io::Error::other(format!(
                "{} is {on_disk} bytes but {committed} are committed",
                path.display()
            )));
        }
        if on_disk > committed {
            file.set_len(committed)?;
            file.sync_data()?;
        }
        file.seek(SeekFrom::Start(committed))?;
        Ok(Self {
            file,
            len: committed,
            dirty: false,
            #[cfg(test)]
            path: path.to_owned(),
        })
    }

    /// Append one line (the caller supplies the trailing newline).
    pub(crate) fn append(&mut self, line: &[u8]) -> std::io::Result<()> {
        #[cfg(test)]
        if FAIL_WRITES_UNDER
            .lock()
            .unwrap()
            .iter()
            .any(|d| self.path.starts_with(d))
        {
            return Err(std::io::Error::other("No space left on device (injected)"));
        }
        if let Err(e) = self.file.write_all(line) {
            // a write that ran out of room part way left half a line: cut it
            // back so the next append starts on a line boundary again
            let _ = self.file.set_len(self.len);
            let _ = self.file.seek(SeekFrom::Start(self.len));
            return Err(e);
        }
        self.len += line.len() as u64;
        self.dirty = true;
        Ok(())
    }

    /// Make everything appended durable - before the store records it.
    pub(crate) fn sync(&mut self) -> std::io::Result<()> {
        if self.dirty {
            self.file.sync_data()?;
            self.dirty = false;
        }
        Ok(())
    }
}

/// A response body streaming the first `len` bytes of `path` - a result file
/// up to its committed length, an input file whole.
pub(crate) async fn body_of(path: &Path, len: u64) -> std::io::Result<axum::body::Body> {
    use tokio::io::AsyncReadExt;
    let f = tokio::fs::File::open(path).await?.take(len);
    let stream = futures::stream::unfold(Some(f), |state| async move {
        let mut f = state?;
        let mut buf = vec![0u8; 1 << 16];
        match f.read(&mut buf).await {
            Ok(0) => None,
            Ok(n) => {
                buf.truncate(n);
                Some((Ok(axum::body::Bytes::from(buf)), Some(f)))
            }
            // one error ends the stream (the client sees a cut-off body)
            Err(e) => Some((Err(e), None)),
        }
    });
    Ok(axum::body::Body::from_stream(stream))
}
