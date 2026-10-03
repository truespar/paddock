//! /proc/meminfo as a unified-memory die's memory: on an integrated NVIDIA
//! part (DGX Spark GB10, Jetson) the GPU has no memory of its own, so what it
//! can allocate is the OS's - MemAvailable, or the free huge pages on a
//! hugetlb box - and the driver's own free figure, which counts the page
//! cache as used, undercounts it. One reading for the two places that need
//! it: the engine's load gate and pool probe (paddock-engine
//! `gpu::unified_mem`), and the manager's device telemetry, where NVML
//! reports no framebuffer for such a part.

/// /proc/meminfo, in bytes - only the fields the accounting reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemInfo {
    pub total: u64,
    pub free: u64,
    pub available: u64,
    buffers: u64,
    cached: u64,
    swap_cached: u64,
    anon: u64,
    slab: u64,
    kernel_stack: u64,
    page_tables: u64,
    sec_page_tables: u64,
    percpu: u64,
    hugetlb: u64,
    huge_total: u64,
    huge_free: u64,
    huge_size: u64,
}

impl MemInfo {
    /// None without MemTotal and MemAvailable - a kernel that old gives the
    /// caller no honest number to start from.
    pub fn parse(text: &str) -> Option<MemInfo> {
        let mut m = MemInfo::default();
        let (mut has_total, mut has_avail) = (false, false);
        // `Slab` is the sum of these two; kept as a fallback for a reading
        // that lacks the total line
        let (mut slab_total, mut s_reclaimable, mut s_unreclaim) = (None, 0u64, 0u64);
        for line in text.lines() {
            let Some((key, rest)) = line.split_once(':') else {
                continue;
            };
            let v: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let kb = v * 1024;
            match key {
                "MemTotal" => (m.total, has_total) = (kb, true),
                "MemFree" => m.free = kb,
                "MemAvailable" => (m.available, has_avail) = (kb, true),
                "Buffers" => m.buffers = kb,
                "Cached" => m.cached = kb,
                "SwapCached" => m.swap_cached = kb,
                "AnonPages" => m.anon = kb,
                "Slab" => slab_total = Some(kb),
                "SReclaimable" => s_reclaimable = kb,
                "SUnreclaim" => s_unreclaim = kb,
                "KernelStack" => m.kernel_stack = kb,
                "PageTables" => m.page_tables = kb,
                "SecPageTables" => m.sec_page_tables = kb,
                "Percpu" => m.percpu = kb,
                "Hugetlb" => m.hugetlb = kb,
                "HugePages_Total" => m.huge_total = v, // a count, not kB
                "HugePages_Free" => m.huge_free = v,
                "Hugepagesize" => m.huge_size = kb,
                _ => {}
            }
        }
        m.slab = slab_total.unwrap_or(s_reclaimable + s_unreclaim);
        (has_total && has_avail).then_some(m)
    }

    /// None off Linux: no integrated NVIDIA die exists anywhere else today.
    pub fn read() -> Option<MemInfo> {
        if !cfg!(target_os = "linux") {
            return None;
        }
        MemInfo::parse(&std::fs::read_to_string("/proc/meminfo").ok()?)
    }

    /// A box with hugetlb pages configured: CUDA can then only use those, so
    /// they are the whole answer (NVIDIA's DGX Spark snippet reads it the
    /// same way), and the driver-pool credit does not apply.
    pub fn hugetlb_configured(&self) -> bool {
        self.huge_total > 0 && self.huge_size > 0
    }

    /// What a CUDA allocation may take per the OS: MemAvailable, the reading
    /// NVIDIA publishes for the DGX Spark and the one llama.cpp, vLLM and
    /// SGLang all take - or the free huge pages on a hugetlb box.
    pub fn usable(&self) -> u64 {
        if self.hugetlb_configured() {
            self.huge_free * self.huge_size
        } else {
            self.available
        }
    }

    /// Bytes the kernel counts as used but files under none of its own
    /// categories: driver allocations, which on a unified-memory die means
    /// live CUDA memory plus the driver's retained pools. The upper bound on
    /// anything the probe may credit. `Cached` already includes shmem.
    pub fn unaccounted(&self) -> u64 {
        self.total
            .saturating_sub(self.free)
            .saturating_sub(self.buffers)
            .saturating_sub(self.cached)
            .saturating_sub(self.swap_cached)
            .saturating_sub(self.anon)
            .saturating_sub(self.slab)
            .saturating_sub(self.kernel_stack)
            .saturating_sub(self.page_tables)
            .saturating_sub(self.sec_page_tables)
            .saturating_sub(self.percpu)
            .saturating_sub(self.hugetlb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The lines read off the Spark's /proc/meminfo with no runner up, after
    /// the stop that led to the 37 GiB grant (2026-09-13). Only these were
    /// captured, which is also why `Slab` is missing here.
    const SPARK_AFTER_STOP: &str = "\
MemTotal:       127600816 kB
MemFree:         3139656 kB
MemAvailable:   40174336 kB
Buffers:           13252 kB
Cached:         37694160 kB
Unevictable:       24640 kB
Mlocked:           24640 kB
AnonPages:       1712244 kB
Shmem:              3028 kB
KReclaimable:     477572 kB
SUnreclaim:       520360 kB
NFS_Unstable:          0 kB
VmallocUsed:      101620 kB
Percpu:            27520 kB
CmaTotal:         131072 kB
CmaFree:           69320 kB
HugePages_Total:       0
Hugetlb:               0 kB
";

    #[test]
    fn the_spark_reading_books_the_driver_pool_under_no_category() {
        let m = MemInfo::parse(SPARK_AFTER_STOP).unwrap();
        assert_eq!(m.usable(), 40174336 * 1024);
        // ~80 GiB the kernel counts as used and cannot name - the retained
        // pool, since no CUDA process was running
        let u = m.unaccounted() as f64 / GIB as f64;
        assert!((79.0..82.0).contains(&u), "unaccounted {u:.1} GiB");
    }

    #[test]
    fn hugetlb_boxes_read_the_free_huge_pages() {
        let text = "MemTotal: 1000 kB\nMemAvailable: 900 kB\nHugePages_Total: 8\nHugePages_Free: 3\nHugepagesize: 1048576 kB\n";
        let m = MemInfo::parse(text).unwrap();
        assert!(m.hugetlb_configured());
        assert_eq!(m.usable(), 3 * GIB);
        assert_eq!(MemInfo::parse("MemTotal: 1000 kB\n"), None);
    }
}
