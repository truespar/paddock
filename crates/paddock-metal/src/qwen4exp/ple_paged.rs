//! File-backed affine PLE lookup. Only compressed rows cross the I/O boundary;
//! dequantization and every floating-point model operation remain on Metal.
//! GPU hashing independently checks every staged row's identity before use.
use super::{affine, mlx, ple};
use crate::{
    device::{Buffer, MetalDevice, MetalError, Result},
    weights::Weight,
};
use objc2_metal::MTLBuffer;
use paddock_models::safetensors::TensorFileReader;

const RECORD: usize = 104; // u32 identity + 80 code + 10 scale + 10 bias bytes
const CACHE_ROWS: usize = 65536;
const IO_WORKERS: usize = 8;
const PREFETCH_BATCH: usize = 256; // 16 tokens, all readers joined before checking GPU completion
pub(super) const LOOKAHEAD_ROWS: usize = 1024;
#[cfg(test)]
thread_local! {
    static SERIAL_IO_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
const MULT: [u64; 3] = [23703573157769, 20109073645365, 8052911324071];
const SIZES: [u32; 16] = [
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
    20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
];

pub(super) struct PagedTable {
    planes: Vec<[TensorFileReader; 3]>,
    // Direct-mapped and bounded: exact ID tags, no unbounded hash map growth or
    // float expansion. A collision only causes another positional read.
    cache: Vec<[u8; RECORD]>,
    lookahead: bool,
    prefetch_cost: std::time::Duration,
    pub(super) staging: Weight,
}

impl PagedTable {
    pub(super) fn bytes(rows: usize) -> u64 {
        (rows * 16 * RECORD) as u64
    }

    pub(super) fn new(d: &MetalDevice, source: &mlx::Source, rows: usize) -> Result<Self> {
        let mut planes = Vec::with_capacity(128);
        for shard in 0..128 {
            let base = format!("{}.ple_embedding.ngram_embedding.shards.{shard}", mlx::PLE);
            let reader = |suffix| {
                source
                    .map
                    .reader(&format!("{base}.{suffix}"))
                    .map_err(|e| MetalError::Model(e.to_string()))
            };
            planes.push([reader("weight")?, reader("scales")?, reader("biases")?]);
        }
        Ok(Self {
            planes,
            cache: vec![[255; RECORD]; CACHE_ROWS],
            lookahead: false,
            prefetch_cost: std::time::Duration::ZERO,
            staging: Weight {
                buffer: d.alloc(Self::bytes(rows) as usize)?,
                ty: affine::A4G32,
                k: 160,
                n: rows * 16,
            },
        })
    }

    /// Previous model work and prefix restores have completed. CPU reads only
    /// fenced integer token history; a new slot has the GPU reset's EOS history.
    /// No GPU submission occurs on a partial/failed file read.
    pub(super) fn stage(&mut self, plan: &ple::Plan, history: &Buffer) -> Result<()> {
        self.lookahead = false;
        let prefetched = std::mem::take(&mut self.prefetch_cost);
        let history = unsafe { history.read_u32(history.len() / 4) };
        let ids = hash_rows(&plan.tokens, &plan.meta, &history)?;
        if ids.len() > self.staging.n {
            return Err(MetalError::Model("PLE staging capacity exceeded".into()));
        }
        // SAFETY: model execution is single-owner and fenced before stage;
        // the staging allocation has no in-flight GPU users. Extent is checked.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                self.staging.buffer.raw.contents().as_ptr().cast::<u8>(),
                ids.len() * RECORD,
            )
        };
        let read = |id, record: &mut [u8; RECORD]| read_record(&self.planes, id, record);
        let started = std::time::Instant::now();
        #[cfg(test)]
        let serial = SERIAL_IO_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let serial = false;
        let result = if serial {
            stage_serial(&ids, &mut self.cache, dst, &read)
        } else {
            stage_rows(&ids, &mut self.cache, dst, &read)
        };
        self.lookahead =
            result.is_ok() && worth_prefetching(ids.len() / 16, started.elapsed(), prefetched);
        result
    }

    pub(super) fn should_prefetch(&self) -> bool {
        self.lookahead
    }

    /// Called after submission: only descriptors and the CPU compressed-row
    /// cache are accessed, never staging or GPU history. Work is bounded and
    /// completes before the model owner returns; cancellation cannot strand
    /// futures or publish results into a reused slot. Identity is the immutable
    /// checkpoint row ID, independently validated by the normal GPU gather.
    pub(super) fn prefetch(&mut self, ids: &[u32], gpu_done: &dyn Fn() -> bool) -> Result<()> {
        self.prefetch_cost = std::time::Duration::ZERO;
        if ids.len() > LOOKAHEAD_ROWS * 16 {
            return Err(MetalError::Model("PLE lookahead capacity exceeded".into()));
        }
        if ids.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let result = prefetch_rows(ids, &mut self.cache, gpu_done, &|id, record| {
            read_record(&self.planes, id, record)
        });
        if result.is_ok() {
            self.prefetch_cost = started.elapsed();
        }
        result
    }
}

// A cheap demand stage can be the result of useful read-ahead, not evidence
// that storage has become cheap. Consume the immediately preceding prefetch
// cost once; no permanent warm/cold flag survives workload or pressure changes.
fn worth_prefetching(
    rows: usize,
    demand: std::time::Duration,
    prefetched: std::time::Duration,
) -> bool {
    rows >= 128 && demand.max(prefetched) >= std::time::Duration::from_millis(8)
}

fn prefetch_rows(
    ids: &[u32],
    cache: &mut [[u8; RECORD]],
    gpu_done: &dyn Fn() -> bool,
    read: &(impl Fn(u32, &mut [u8; RECORD]) -> Result<()> + Sync),
) -> Result<()> {
    // Check completion between bounded batches, not after all future rows.
    // At most this one batch can extend past GPU completion; no detached I/O
    // remains. A single filesystem read has no hard wall-time bound.
    let mut records = [0u8; PREFETCH_BATCH * RECORD];
    for batch in ids.chunks(PREFETCH_BATCH) {
        if gpu_done() {
            break;
        }
        stage_rows(batch, cache, &mut records[..batch.len() * RECORD], read)?;
    }
    Ok(())
}

fn read_record(planes: &[[TensorFileReader; 3]], id: u32, record: &mut [u8; RECORD]) -> Result<()> {
    let shard = id as usize / mlx::SHARD_ROWS;
    let row = id as usize % mlx::SHARD_ROWS;
    let Some(planes) = planes.get(shard) else {
        return Err(MetalError::Model("PLE row outside checkpoint".into()));
    };
    for (plane, stride, range) in [(0, 80, 4..84), (1, 10, 84..94), (2, 10, 94..104)] {
        planes[plane]
            .read_at(row * stride, &mut record[range])
            .map_err(|e| MetalError::Model(format!("PLE row read failed: {e}")))?;
    }
    record[..4].copy_from_slice(&id.to_le_bytes());
    Ok(())
}

/// Only known prompt tokens may be looked up ahead. Never predict decode
/// tokens or inspect GPU history while a command is running.
pub(super) fn prompt_lookahead(tokens: &[u32], start: usize, count: usize) -> Result<Vec<u32>> {
    if count > LOOKAHEAD_ROWS || start > tokens.len() || count > tokens.len() - start {
        return Err(MetalError::Model("invalid PLE lookahead span".into()));
    }
    let history = [
        start.checked_sub(1).map_or(248044, |i| tokens[i]),
        start.checked_sub(2).map_or(248044, |i| tokens[i]),
    ];
    let meta = (0..count)
        .flat_map(|i| [0, (start + i) as u32, 0, count as u32])
        .collect::<Vec<_>>();
    hash_rows(&tokens[start..start + count], &meta, &history)
}

fn bucket(id: u32) -> usize {
    id.wrapping_mul(2654435761) as usize >> 16
}

fn tag(record: &[u8; RECORD]) -> u32 {
    u32::from_le_bytes(record[..4].try_into().expect("fixed record tag"))
}

/// A decode never creates workers or per-miss jobs. Invalidate the tag before
/// I/O; truncated planes cannot leave a valid tag on partially replaced data.
fn stage_serial(
    ids: &[u32],
    cache: &mut [[u8; RECORD]],
    dst: &mut [u8],
    read: &(impl Fn(u32, &mut [u8; RECORD]) -> Result<()> + Sync),
) -> Result<()> {
    for (&id, dst) in ids.iter().zip(dst.chunks_exact_mut(RECORD)) {
        let entry = &mut cache[bucket(id)];
        if tag(entry) != id {
            entry[..4].fill(255);
            read(id, entry)?;
        }
        dst.copy_from_slice(entry);
    }
    Ok(())
}

/// Deduplicate compressed-row misses and read at most eight independent
/// batches concurrently. Scope ownership drains every read (also on failure)
/// before borrowed descriptors/staging can disappear. No worker touches a GPU
/// buffer or publishes a cache tag. Host memory is bounded by the row capacity.
fn stage_rows(
    ids: &[u32],
    cache: &mut [[u8; RECORD]],
    dst: &mut [u8],
    read: &(impl Fn(u32, &mut [u8; RECORD]) -> Result<()> + Sync),
) -> Result<()> {
    if ids.len() > affine::MAX_ROWS * 16
        || dst.len() != ids.len() * RECORD
        || cache.len() != CACHE_ROWS
        || ids.contains(&u32::MAX)
    {
        return Err(MetalError::Model(
            "invalid PLE staging extent/identity".into(),
        ));
    }
    if ids.len() <= 8 * 16 {
        return stage_serial(ids, cache, dst, read);
    }
    let mut missing = Vec::new();
    for (row, (&id, dst)) in ids.iter().zip(dst.chunks_exact_mut(RECORD)).enumerate() {
        let entry = &cache[bucket(id)];
        if tag(entry) == id {
            dst.copy_from_slice(entry);
        } else {
            missing.push((id, row));
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    let mut jobs = Vec::new();
    for &(id, _) in &missing {
        if jobs.last().is_none_or(|&(prev, _)| prev != id) {
            jobs.push((id, [255u8; RECORD]));
        }
    }
    // Cached preads are faster without thread creation/syscall contention.
    // Probe bounded batches, not a one-time page-residency guess: a later
    // cold shard can still trigger parallel I/O. No persistent 'warm' bit can
    // become stale after memory pressure evicts filesystem pages.
    let mut read_count = 0;
    for batch in jobs.chunks_mut(32) {
        let started = std::time::Instant::now();
        for (id, record) in batch.iter_mut() {
            read(*id, record)?;
        }
        read_count += batch.len();
        if started.elapsed() >= std::time::Duration::from_micros(128) {
            break;
        }
    }
    let remaining = &mut jobs[read_count..];
    if remaining.len() < 128 {
        for (id, record) in remaining {
            read(*id, record)?;
        }
    } else {
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            let mut error = None;
            let chunk = remaining.len().div_ceil(IO_WORKERS);
            for rows in remaining.chunks_mut(chunk) {
                match std::thread::Builder::new()
                    .name("paddock-ple-read".into())
                    .spawn_scoped(scope, move || -> Result<()> {
                        for (id, record) in rows {
                            read(*id, record)?;
                        }
                        Ok(())
                    }) {
                    Ok(h) => handles.push(h),
                    Err(e) => {
                        error = Some(MetalError::Model(format!("PLE worker start failed: {e}")));
                        break;
                    }
                }
            }
            for handle in handles {
                let result = handle
                    .join()
                    .unwrap_or_else(|_| Err(MetalError::Model("PLE worker panicked".into())));
                if let Err(e) = result {
                    error.get_or_insert(e);
                }
            }
            match error {
                Some(e) => Err(e),
                None => Ok(()),
            }
        })?;
    }
    // Complete all file reads before publication. Duplicate IDs read once;
    // colliding direct-map buckets may evict each other but never alias data.
    if jobs.iter().any(|(id, record)| tag(record) != *id) {
        return Err(MetalError::Model(
            "PLE reader returned a mismatched identity".into(),
        ));
    }
    let mut index = 0;
    for (id, record) in jobs {
        while index < missing.len() && missing[index].0 == id {
            let row = missing[index].1;
            dst[row * RECORD..(row + 1) * RECORD].copy_from_slice(&record);
            index += 1;
        }
        cache[bucket(id)] = record;
    }
    Ok(())
}

// Integer addressing only. Keep the GPU hash as an independent production
// validator; no CPU dequantization, attention, projection or recurrent math.
fn hash_rows(tokens: &[u32], meta: &[u32], history: &[u32]) -> Result<Vec<u32>> {
    if meta.len() != tokens.len() * 4 || !history.len().is_multiple_of(2) {
        return Err(MetalError::Model("invalid PLE hash metadata".into()));
    }
    let mut out = Vec::with_capacity(tokens.len() * 16);
    for (row, m) in meta.chunks_exact(4).enumerate() {
        let [slot, pos, start, end] = [m[0] as usize, m[1] as usize, m[2] as usize, m[3] as usize];
        if slot >= history.len() / 2
            || start > row
            || end <= row
            || end > tokens.len()
            || pos < row - start
        {
            return Err(MetalError::Model("invalid PLE hash span".into()));
        }
        let old = if pos == row - start {
            [248044, 248044]
        } else {
            [history[slot * 2], history[slot * 2 + 1]]
        };
        let cur = tokens[row];
        let prev = if row > start { tokens[row - 1] } else { old[0] };
        let mut prev2 = if row >= start + 2 {
            tokens[row - 2]
        } else if row > start {
            old[0]
        } else {
            old[1]
        };
        if prev == 248044 {
            prev2 = 248044;
        }
        if [cur, prev, prev2].iter().any(|&v| v >= 248320) {
            return Err(MetalError::Model("invalid PLE history token".into()));
        }
        let pair = (cur as u64 * MULT[0]) ^ (prev as u64 * MULT[1]);
        let mut offset = 0;
        for (head, size) in SIZES.into_iter().enumerate() {
            let hash = if head < 8 {
                pair
            } else {
                pair ^ (prev2 as u64 * MULT[2])
            };
            out.push((hash % size as u64) as u32 + offset);
            offset += size;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fake_record(id: u32, record: &mut [u8; RECORD]) -> Result<()> {
        for (i, byte) in record.iter_mut().enumerate() {
            *byte = id.wrapping_mul(17).wrapping_add(i as u32) as u8;
        }
        record[..4].copy_from_slice(&id.to_le_bytes());
        Ok(())
    }

    #[test]
    fn lookahead_policy_tracks_cost_without_disabling_its_own_success() {
        use std::time::Duration as D;
        assert!(!worth_prefetching(127, D::from_secs(1), D::from_secs(1)));
        assert!(!worth_prefetching(128, D::from_micros(7999), D::ZERO));
        assert!(worth_prefetching(128, D::from_millis(8), D::ZERO));
        assert!(worth_prefetching(1024, D::ZERO, D::from_millis(8)));
        assert!(!worth_prefetching(
            1024,
            D::from_millis(1),
            D::from_millis(1)
        ));
    }

    #[test]
    fn read_ahead_stops_at_completion_and_demand_fills_the_remainder() {
        let ids = (0..1024).collect::<Vec<_>>();
        let mut cache = vec![[255u8; RECORD]; CACHE_ROWS];
        let calls = AtomicUsize::new(0);
        let read = |id, record: &mut [u8; RECORD]| {
            calls.fetch_add(1, Ordering::Relaxed);
            fake_record(id, record)
        };
        prefetch_rows(&ids, &mut cache, &|| true, &read).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        prefetch_rows(
            &ids,
            &mut cache,
            &|| calls.load(Ordering::Relaxed) >= PREFETCH_BATCH,
            &read,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), PREFETCH_BATCH);
        let mut out = vec![0; ids.len() * RECORD];
        stage_rows(&ids, &mut cache, &mut out, &read).unwrap();
        for (&id, actual) in ids.iter().zip(out.chunks_exact(RECORD)) {
            let mut expected = [0; RECORD];
            fake_record(id, &mut expected).unwrap();
            assert_eq!(actual, expected);
        }
        // A failed optional batch cannot label partial bytes as valid. Earlier
        // complete batches remain useful and all scoped workers have joined.
        cache.fill([255; RECORD]);
        assert!(
            prefetch_rows(&ids, &mut cache, &|| false, &|id, record| {
                if id == PREFETCH_BATCH as u32 + 39 {
                    return Err(MetalError::Model("injected read-ahead failure".into()));
                }
                fake_record(id, record)
            })
            .is_err()
        );
        for id in 0..PREFETCH_BATCH as u32 {
            assert_eq!(tag(&cache[bucket(id)]), id);
        }
        for &id in &ids[PREFETCH_BATCH..] {
            assert_ne!(tag(&cache[bucket(id)]), id);
        }
    }

    #[test]
    fn prompt_lookahead_preserves_eos_history_and_partial_spans() {
        let mut tokens = (0..1100).map(|i| 1000 + i as u32 * 17).collect::<Vec<_>>();
        for i in [0, 1, 15, 511, 1023] {
            tokens[i] = 248044;
        }
        let meta = (0..tokens.len())
            .flat_map(|i| [0, i as u32, 0, tokens.len() as u32])
            .collect::<Vec<_>>();
        let all = hash_rows(&tokens, &meta, &[248044, 248044]).unwrap();
        for start in [0, 1, 2, 3, 14, 15, 16, 17, 510, 511, 512, 513, 1024, 1100] {
            for count in [0, 1, 13, 127, LOOKAHEAD_ROWS] {
                let count = count.min(tokens.len() - start);
                assert_eq!(
                    prompt_lookahead(&tokens, start, count).unwrap(),
                    all[start * 16..(start + count) * 16]
                );
            }
        }
        assert!(prompt_lookahead(&tokens, usize::MAX, 1).is_err());
        assert!(prompt_lookahead(&tokens, tokens.len(), 1).is_err());
        assert!(prompt_lookahead(&tokens, 0, LOOKAHEAD_ROWS + 1).is_err());
    }

    #[test]
    #[ignore = "local checkpoint descriptors; verify compressed read-ahead never touches GPU staging"]
    fn paged_ple_lookahead_exact_bytes_and_staging_isolation() {
        let source = mlx::Source::open(std::path::Path::new(
            &std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap(),
        ))
        .unwrap();
        let d = MetalDevice::new(Some(16 << 20)).unwrap();
        let mut table = PagedTable::new(&d, &source, 512).unwrap();
        let tokens = (0..512).map(|i| 1000 + i).collect::<Vec<_>>();
        let rows = tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| (0, i, t))
            .collect::<Vec<_>>();
        let plan = ple::Plan::new(&rows, &[0], 512, 1024).unwrap();
        let history = upload(&d, &[248044, 248044]);
        let allocated = d.allocated_bytes();
        let count = table.staging.buffer.len() / 4;
        unsafe { table.staging.buffer.write_u32(&vec![0xdeadbeef; count]) };
        let ids = prompt_lookahead(&tokens, 0, tokens.len()).unwrap();
        table.prefetch(&ids, &|| false).unwrap();
        assert_eq!(
            unsafe { table.staging.buffer.read_u32(count) },
            vec![0xdeadbeef; count]
        );
        table.stage(&plan, &history).unwrap();
        let warmed = unsafe { table.staging.buffer.read_u32(count) };
        table.cache.fill([255; RECORD]);
        table.stage(&plan, &history).unwrap();
        assert_eq!(warmed, unsafe { table.staging.buffer.read_u32(count) });
        assert_eq!(allocated, d.allocated_bytes());
    }

    #[test]
    fn compressed_read_batches_deduplicate_and_preserve_order() {
        let ids = (0..2048).map(|i| (i * 17 % 257) as u32).collect::<Vec<_>>();
        let mut cache = vec![[255u8; RECORD]; CACHE_ROWS];
        let mut out = vec![0; ids.len() * RECORD];
        let calls = AtomicUsize::new(0);
        let read = |id, record: &mut [u8; RECORD]| {
            calls.fetch_add(1, Ordering::Relaxed);
            fake_record(id, record)
        };
        stage_rows(&ids, &mut cache, &mut out, &read).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 257);
        let mut reference_cache = vec![[255u8; RECORD]; CACHE_ROWS];
        let mut reference = vec![0; out.len()];
        stage_serial(&ids, &mut reference_cache, &mut reference, &fake_record).unwrap();
        assert!(out == reference);
        let calls_before = calls.load(Ordering::Relaxed);
        stage_rows(&ids, &mut cache, &mut out, &read).unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), calls_before);
        assert!(out == reference);
        let collision = (2..1_000_000).find(|&i| bucket(i) == bucket(1)).unwrap();
        let ids = (0..256)
            .map(|i| if i % 2 == 0 { 1 } else { collision })
            .collect::<Vec<_>>();
        out.resize(ids.len() * RECORD, 0);
        stage_rows(&ids, &mut cache, &mut out, &fake_record).unwrap();
        for (record, id) in out.chunks_exact(RECORD).zip(ids) {
            let mut expected = [0; RECORD];
            fake_record(id, &mut expected).unwrap();
            assert_eq!(record, expected);
        }
    }

    #[test]
    fn compressed_read_failure_drains_workers_without_cache_publication() {
        let ids = (0..256).collect::<Vec<_>>();
        for panic_worker in [false, true] {
            let mut cache = vec![[255u8; RECORD]; CACHE_ROWS];
            let mut out = vec![0; ids.len() * RECORD];
            let active = AtomicUsize::new(0);
            let read = |id, record: &mut [u8; RECORD]| {
                struct Guard<'a>(&'a AtomicUsize);
                impl Drop for Guard<'_> {
                    fn drop(&mut self) {
                        self.0.fetch_sub(1, Ordering::SeqCst);
                    }
                }
                active.fetch_add(1, Ordering::SeqCst);
                let _guard = Guard(&active);
                // Force a measured stall in the first probe, then fail on
                // a worker rather than before workers have been submitted.
                if id == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                if id == 39 {
                    assert!(!panic_worker, "injected worker failure");
                    return Err(MetalError::Model("injected short read".into()));
                }
                fake_record(id, record)
            };
            assert!(stage_rows(&ids, &mut cache, &mut out, &read).is_err());
            assert_eq!(active.load(Ordering::SeqCst), 0);
            assert!(cache.iter().all(|r| r == &[255; RECORD]));
        }
    }

    #[test]
    fn compressed_read_rejects_extent_and_bad_tags() {
        let mut cache = vec![[255u8; RECORD]; CACHE_ROWS];
        assert!(stage_rows(&[1], &mut cache, &mut [], &fake_record).is_err());
        assert!(stage_rows(&[u32::MAX], &mut cache, &mut [0; RECORD], &fake_record).is_err());
        let ids = (0..256).collect::<Vec<_>>();
        assert!(
            stage_rows(
                &ids,
                &mut cache,
                &mut vec![0; ids.len() * RECORD],
                &|_, _| Ok(())
            )
            .is_err()
        );
        assert!(cache.iter().all(|r| r == &[255; RECORD]));
    }

    #[test]
    #[ignore = "checkpoint I/O A/B; compressed bytes only, no resident table or model weights"]
    fn paged_ple_parallel_io_cost_and_exactness() {
        let source = mlx::Source::open(std::path::Path::new(
            &std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap(),
        ))
        .unwrap();
        let d = MetalDevice::new(Some(16 << 20)).unwrap();
        let mut paged = PagedTable::new(&d, &source, affine::MAX_ROWS).unwrap();
        drop(source);
        let history = upload(&d, &[248044, 248044]);
        for n in [1, 8, 128, 1024] {
            let mut times = [Vec::new(), Vec::new()];
            for round in 0..4 {
                let rows = (0..n)
                    .map(|i| (0, i, (1000 + i + round * 5000) as u32))
                    .collect::<Vec<_>>();
                let plan = ple::Plan::new(&rows, &[0], affine::MAX_ROWS, 8192).unwrap();
                let mut expected = None;
                for route in (0..2).map(|r| (r + round) % 2) {
                    paged.cache.fill([255; RECORD]);
                    SERIAL_IO_FOR_TEST.with(|v| v.set(route == 0));
                    let now = std::time::Instant::now();
                    let result = paged.stage(&plan, &history);
                    SERIAL_IO_FOR_TEST.with(|v| v.set(false));
                    result.unwrap();
                    times[route].push(now.elapsed().as_secs_f64());
                    let bytes = unsafe { paged.staging.buffer.read_u32(n * 16 * RECORD / 4) };
                    if let Some(expected) = &expected {
                        assert!(&bytes == expected);
                    } else {
                        expected = Some(bytes);
                    }
                }
            }
            eprintln!(
                "FLASH_PLE_IO {}",
                serde_json::json!({"rows":n,"seconds":times,"compressed_bytes_exact":true})
            );
        }
    }

    fn upload(d: &MetalDevice, v: &[u32]) -> Buffer {
        d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    }

    #[test]
    fn paged_ple_integer_addressing_matches_gpu_history_and_reset() {
        let d = MetalDevice::new(Some(8 << 20)).unwrap();
        for n in [1, 2, 3, 8, 129, 1024] {
            let tokens = (0..n)
                .map(|i| {
                    if i % 7 == 0 {
                        248044
                    } else {
                        (i * 9371 % 248320) as u32
                    }
                })
                .collect::<Vec<_>>();
            for start in [0, 1, 4097] {
                let meta = (0..n)
                    .flat_map(|i| [2, (start + i) as u32, 0, n as u32])
                    .collect::<Vec<_>>();
                let mut history = [17, 99, 23, 31, 248319, 0];
                let expected = hash_rows(&tokens, &meta, &history).unwrap();
                if start == 0 {
                    history[4..].fill(248044);
                }
                let t = upload(&d, &tokens);
                let m = upload(&d, &meta);
                let h = upload(&d, &history);
                let out = upload(&d, &vec![u32::MAX; n * 16 + 17]);
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "q4x_ple_hash",
                    &[&t, &m, &h, &out],
                    &[n as u32, 3],
                    [n, 1, 1],
                    32,
                );
                cmd.finish().unwrap();
                let actual = unsafe { out.read_u32(n * 16 + 17) };
                assert_eq!(&actual[..n * 16], expected);
                assert_eq!(&actual[n * 16..], &[u32::MAX; 17]);
            }
        }
        assert!(hash_rows(&[248320], &[0, 0, 0, 1], &[0, 0]).is_err());
        assert!(hash_rows(&[1], &[1, 0, 0, 1], &[0, 0]).is_err());
        assert!(hash_rows(&[1], &[0, 1, 0, 1], &[u32::MAX, 0]).is_err());
        assert!(hash_rows(&[1], &[0, 0, 1, 1], &[0, 0]).is_err());
        assert!(hash_rows(&[1], &[], &[0, 0]).is_err());
    }

    #[test]
    fn paged_ple_gpu_rejects_stale_row_id() {
        let d = MetalDevice::new(Some(8 << 20)).unwrap();
        let mut record = [0u8; RECORD];
        record[..4].copy_from_slice(&42u32.to_le_bytes());
        let w = d.upload(&record).unwrap();
        let ids = upload(&d, &[43]);
        let out = upload(&d, &[0; 160]);
        let cmd = d.begin().unwrap();
        cmd.dispatch("q4a_ple_staged", &[&w, &ids, &out], &[1], [1, 1, 1], 256);
        cmd.finish().unwrap();
        assert!(unsafe { out.read_f32(0, 160) }.iter().all(|v| v.is_nan()));
    }

    #[test]
    #[ignore = "32GB resident versus file-backed PLE; elected model and watchdog required"]
    fn paged_ple_compressed_rows_match_resident_gpu_gather() {
        let source = mlx::Source::open(std::path::Path::new(
            &std::env::var("PADDOCK_FLASH_NEXT_MLX_MODEL").unwrap(),
        ))
        .unwrap();
        let d = MetalDevice::new(Some(34 << 30)).unwrap();
        let resident = source.table(&d).unwrap();
        let mut paged = PagedTable::new(&d, &source, 1024).unwrap();
        drop(source); // Readers must not depend on the checkpoint's mmap lifetime.
        let before = d.allocated_bytes();
        let history = upload(&d, &[248044, 248044, 248319, 42, 17, 248044, 999, 777]);
        for n in [1usize, 8, 129, 1024] {
            let rows = (0..n)
                .map(|i| {
                    (
                        2,
                        37 + i,
                        if i % 13 == 0 {
                            248044
                        } else {
                            (i * 977 + 23) as u32 % 248320
                        },
                    )
                })
                .collect::<Vec<_>>();
            let plan = ple::Plan::new(&rows, &[0, 0, 37, 0], 1024, 4096).unwrap();
            let tokens = upload(&d, &plan.tokens);
            let meta = upload(&d, &plan.meta);
            let ids = d.alloc(n * 16 * 4).unwrap();
            let old = d.alloc(n * 2560 * 4).unwrap();
            let new = d.alloc(n * 2560 * 4).unwrap();
            for repeat in 0..2 {
                let start = std::time::Instant::now();
                paged.stage(&plan, &history).unwrap();
                let staging_ms = start.elapsed().as_secs_f64() * 1000.;
                let cmd = d.begin().unwrap();
                cmd.dispatch(
                    "q4x_ple_hash",
                    &[&tokens, &meta, &history, &ids],
                    &[n as u32, 4],
                    [n, 1, 1],
                    32,
                );
                cmd.dispatch(
                    "q4a_ple_gather",
                    &[&resident.buffer, &ids, &old],
                    &[(n * 16) as u32, mlx::SHARD_ROWS as u32, 128],
                    [(n * 2560).div_ceil(256), 1, 1],
                    256,
                );
                cmd.dispatch(
                    "q4a_ple_staged",
                    &[&paged.staging.buffer, &ids, &new],
                    &[(n * 16) as u32],
                    [(n * 2560).div_ceil(256), 1, 1],
                    256,
                );
                cmd.finish().unwrap();
                assert_eq!(unsafe { new.read_f32(0, n * 2560) }, unsafe {
                    old.read_f32(0, n * 2560)
                });
                eprintln!(
                    "FLASH_PLE_PAGED rows={n} repeat={repeat} stage_ms={staging_ms:.3} exact=true"
                );
            }
        }
        drop(history);
        assert_eq!(before, d.allocated_bytes());
    }
}
