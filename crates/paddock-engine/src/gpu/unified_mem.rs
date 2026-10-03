//! Memory a unified-memory die can really use, beyond what the OS reports.
//!
//! On an integrated die (DGX Spark GB10, Jetson) "free VRAM" is the OS's
//! MemAvailable (`GpuExecutor::device_mem_info` explains why). That figure has
//! one blind spot big enough to break sizing: the NVIDIA driver (open kernel
//! module >= 590, system-memory pools) keeps the pages an exiting CUDA process
//! frees in its own pools. /proc/meminfo books them as used under no category,
//! MemAvailable leaves them out, and the next CUDA allocation is served from
//! them first.
//!
//! Measured on a Spark on 2026-09-13, Qwen3.8-27B serving through the manager:
//! the first start planned KV against a 76 GiB grant; after one stop/start the
//! grant was 37 GiB, and `max_ctx 65536` refused to start ("needs 126.50 GiB
//! of KV, only 29.92 GiB fits"). No other process held memory - the container
//! held 1.35 GiB of anonymous memory and 31 GiB of page cache - yet
//! MemAvailable read 40 GiB and cuMemGetInfo 2.3 GiB, while a 96 GiB CUDA
//! allocation succeeded. Every restart shrinks the next grant by whatever the
//! last runner left in the pool.
//!
//! The load gate already asks the driver with a trial allocation before it
//! refuses weights. The sizers need an amount rather than a yes, so this
//! measures the reusable pool: allocate in chunks, write each one so its pages
//! really exist, and after every chunk read MemAvailable. A chunk the pool
//! served leaves MemAvailable where it was; a chunk the kernel served lowers it
//! by the chunk's size. Credit is what MemAvailable did not pay for, and the
//! walk stops at the first chunk the kernel mostly paid, so the probe evicts at
//! most about one chunk of page cache. Everything is freed straight after -
//! back into the same pool, where the load's own allocations then find it.
//!
//! Two bounds keep a wrong reading from over-committing the box. The credit
//! can never exceed the memory /proc/meminfo books under no category (minus
//! this process's own reserved pool and a slack for driver and firmware
//! allocations), and it is consumed as this process allocates: each byte our
//! pool gains since the probe is taken off the credit, because the driver
//! serves those allocations from its pool first. If the kernel served them
//! instead, MemAvailable has already dropped and the credit is taken off twice
//! - an undercount, which is today's behaviour, never an overcount.

/// /proc/meminfo, shared with the manager's telemetry so both read a
/// unified-memory die the same way.
pub(super) use paddock_models::meminfo::MemInfo;

/// Probe chunk. Bounds the page cache one probe can evict (the chunk the
/// kernel paid for, at most) and the time it takes (a 2 GiB write is quick
/// on any of these dies).
pub(super) const PROBE_CHUNK: u64 = 2 << 30;

/// Left out of the credit bound for allocations the driver and firmware hold
/// that are neither ours nor reusable: the CUDA context, GSP heaps, display.
pub(super) const DRIVER_SLACK: u64 = 2 << 30;

/// The reusable pool, proven once, then spent as this process allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RetainedCredit {
    /// Bytes the probe's allocations drew without lowering MemAvailable.
    pub proven: u64,
    /// This process's pool usage when the probe ran.
    pub ours_at_probe: u64,
}

impl RetainedCredit {
    /// Credit still standing: what we have allocated since the probe came out
    /// of the pool first.
    pub(super) fn remaining(&self, ours_now: u64) -> u64 {
        self.proven
            .saturating_sub(ours_now.saturating_sub(self.ours_at_probe))
    }
}

/// Measure how much of an allocation of up to `cap` bytes the driver's pool
/// serves. `available` reads MemAvailable, `take` allocates-and-writes one
/// chunk (None = refused), `give_back` frees one. Kept free of CUDA so the
/// walk is testable on any box; `GpuExecutor::retained_headroom` supplies the
/// real calls.
pub(super) fn measure_retained<H>(
    cap: u64,
    chunk: u64,
    mut available: impl FnMut() -> Option<u64>,
    mut take: impl FnMut(u64) -> Option<H>,
    mut give_back: impl FnMut(H),
) -> u64 {
    if cap == 0 || chunk == 0 {
        return 0;
    }
    let Some(mut before) = available() else {
        return 0;
    };
    let mut held = Vec::new();
    let (mut drawn, mut credit) = (0u64, 0u64);
    while drawn < cap {
        let size = chunk.min(cap - drawn);
        let Some(h) = take(size) else {
            break;
        };
        held.push(h);
        drawn += size;
        let Some(now) = available() else {
            break;
        };
        // what the kernel's own memory paid for this chunk
        let paid = before.saturating_sub(now);
        before = now;
        credit += size.saturating_sub(paid);
        if paid > size / 2 {
            // the pool has run dry: every further chunk would evict page cache
            break;
        }
    }
    for h in held {
        give_back(h);
    }
    credit.min(cap)
}

impl super::GpuExecutor {
    /// Driver-retained bytes this process may still count as free on a
    /// unified-memory die, on top of MemAvailable. Measured the first time a
    /// sizer finds MemAvailable binding, capped at `cap` (the most any sizer
    /// could take, so the probe never walks further than could change an
    /// answer), then spent as our pool grows. Must run on the engine thread,
    /// like every sizer that reads `vram_headroom`.
    pub(super) fn retained_headroom(&self, cap: u64) -> u64 {
        let ours = self.process_mem_used().unwrap_or(0);
        let mut slot = self
            .retained
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(c) = *slot {
            return c.remaining(ours);
        }
        let proven = match MemInfo::read() {
            Some(info) if !info.hugetlb_configured() => {
                let bound = info
                    .unaccounted()
                    .saturating_sub(self.pool_reserved_bytes().unwrap_or(0))
                    .saturating_sub(DRIVER_SLACK);
                let t0 = std::time::Instant::now();
                let proven = measure_retained(
                    cap.min(bound),
                    PROBE_CHUNK,
                    || MemInfo::read().map(|m| m.available),
                    |size| {
                        // SAFETY: a fresh allocation on this thread's current
                        // context, written only within its own length, and
                        // freed exactly once - here on a failed write, or by
                        // the give-back below.
                        unsafe {
                            let ptr = cudarc::driver::result::malloc_sync(size as usize).ok()?;
                            // written, so its pages exist: an allocation the
                            // driver only reserved would read as pool-served
                            if cudarc::driver::result::memset_d8_sync(ptr, 0, size as usize)
                                .is_err()
                            {
                                let _ = cudarc::driver::result::free_sync(ptr);
                                return None;
                            }
                            Some(ptr)
                        }
                    },
                    |ptr| {
                        // SAFETY: `ptr` came from the malloc_sync above and
                        // has not been freed.
                        if let Err(e) = unsafe { cudarc::driver::result::free_sync(ptr) } {
                            tracing::warn!(error = %e, "retained-pool probe: free failed");
                        }
                    },
                );
                let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
                tracing::info!(
                    reusable_gib = gib(proven),
                    available_gib = gib(info.available),
                    unaccounted_gib = gib(info.unaccounted()),
                    probe_ms = t0.elapsed().as_millis() as u64,
                    "unified-memory die: memory the driver kept from earlier CUDA processes \
                     measured as reusable - counted as free on top of MemAvailable"
                );
                proven
            }
            _ => 0,
        };
        *slot = Some(RetainedCredit {
            proven,
            ours_at_probe: ours,
        });
        proven
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A pool of `pool` bytes in front of a kernel with `kernel` bytes of
    /// MemAvailable: allocations drain the pool first, then the kernel.
    struct PooledBox {
        pool: u64,
        kernel: u64,
        refuse_after: u64,
        live: u64,
        freed: u64,
    }

    impl PooledBox {
        fn run(&mut self, cap: u64) -> u64 {
            let cell = std::cell::RefCell::new(self);
            measure_retained(
                cap,
                PROBE_CHUNK,
                || Some(cell.borrow().kernel),
                |size| {
                    let mut b = cell.borrow_mut();
                    if b.live + size > b.refuse_after {
                        return None;
                    }
                    let from_pool = size.min(b.pool);
                    b.pool -= from_pool;
                    let from_kernel = size - from_pool;
                    if from_kernel > b.kernel {
                        return None;
                    }
                    b.kernel -= from_kernel;
                    b.live += size;
                    Some(size)
                },
                |size| {
                    let mut b = cell.borrow_mut();
                    b.live -= size;
                    b.freed += size;
                    b.pool += size; // freed pages go back to the pool
                },
            )
        }
    }

    #[test]
    fn the_probe_credits_the_pool_and_stops_where_the_kernel_starts_paying() {
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        let credit = b.run(88 * GIB);
        assert_eq!(credit, 56 * GIB);
        // it walked one chunk into the kernel's memory and no further
        assert_eq!(b.kernel, 38 * GIB);
        assert_eq!(b.live, 0, "every chunk is freed");
        assert_eq!(b.freed, 58 * GIB);
    }

    #[test]
    fn the_probe_never_measures_past_its_cap_or_past_a_refusal() {
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(10 * GIB), 10 * GIB);
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: 20 * GIB,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(88 * GIB), 20 * GIB);
        assert_eq!(b.live, 0);
        // an odd cap still ends exactly on it
        let mut b = PooledBox {
            pool: 56 * GIB,
            kernel: 40 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(5 * GIB + 7), 5 * GIB + 7);
    }

    #[test]
    fn no_pool_means_no_credit_and_one_chunk_of_cache_at_most() {
        let mut b = PooledBox {
            pool: 0,
            kernel: 80 * GIB,
            refuse_after: u64::MAX,
            live: 0,
            freed: 0,
        };
        assert_eq!(b.run(60 * GIB), 0);
        assert_eq!(b.freed, PROBE_CHUNK);
    }

    #[test]
    fn credit_is_spent_by_what_this_process_allocates_afterwards() {
        let c = RetainedCredit {
            proven: 56 * GIB,
            ours_at_probe: GIB,
        };
        assert_eq!(c.remaining(GIB), 56 * GIB);
        // weights and planes land from the pool first
        assert_eq!(c.remaining(26 * GIB), 31 * GIB);
        assert_eq!(c.remaining(80 * GIB), 0);
        // our pool shrinking below the probe's reading adds nothing back
        assert_eq!(c.remaining(0), 56 * GIB);
    }
}
