//! MoE expert offload, the VRAM side: a per-layer LRU cache of routed
//! experts over the host-mapped planes of `host_plane.rs`.
//!
//! The cache is three slot planes (gate/up/down) in the same repacked
//! k-quant layout as a resident plane, holding `slots` experts instead of
//! `n_expert`, plus the device-side bookkeeping the pack's resolve kernel
//! updates in place: `slot_of[n_expert]`, `expert_in[slots]`,
//! `last_use[slots]`, a tick counter. Per launch the engine runs
//! resolve (ids -> slots, miss jobs) then fill (miss bytes from the mirror
//! into their slots, PCIe-bound) and hands the unchanged MoE kernels the
//! slot planes and the remapped ids. All of it captures into the decode
//! graph; nothing here touches the host after load.
//!
//! Sizing: a slot is one expert's gate+up+down bytes (2.0 MB on the 35B-A3B
//! UD file). Measured on a 4032-token prompt, per-layer LRU hit rates run
//! 82% at 64 slots, 90% at 96, 94% at 128; enable_batch seats what the KV
//! plan leaves.

use cudarc::driver::{CudaSlice, DevicePtr};

use super::error::check;
use super::{GpuError, GpuExecutor, HostMappedKq, RepackedKQ};

/// Sentinel in `slot_of` / `expert_in`: not resident.
pub const MOE_CACHE_NONE: u32 = u32::MAX;

/// The `[moe_offload]` election, armed once by the runner before any model
/// loads (families read it at load and at enable_batch) - the same shape as
/// `kv_tier::pool_tier::set_tier_ram_bytes`. Budgets only: `vram_gb` caps
/// the slot cache; `None` lets the cache take what the KV plan leaves.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeOffloadCfg {
    pub enabled: bool,
    pub vram_bytes: Option<u64>,
}

static MOE_OFFLOAD: std::sync::OnceLock<MoeOffloadCfg> = std::sync::OnceLock::new();

pub fn set_moe_offload(cfg: MoeOffloadCfg) {
    let _ = MOE_OFFLOAD.set(cfg);
}

/// The armed config; `PADDOCK_MOE_HOST=1` is the development switch that
/// enables it without a config file (tests, bring-up).
pub fn moe_offload() -> MoeOffloadCfg {
    let mut c = MOE_OFFLOAD.get().copied().unwrap_or_default();
    if paddock_models::dev_var_os!("PADDOCK_MOE_HOST").is_some() {
        c.enabled = true;
    }
    c
}

/// `PADDOCK_MOE_CACHE_SLOTS=<n>`: development pin for the per-layer slot
/// count, overriding the auto size (the same instrument class as
/// PADDOCK_KV_POOL_BLOCKS). 0 = no cache (pure zero-copy).
pub fn moe_cache_slots_pin() -> Option<usize> {
    paddock_models::dev_var!("PADDOCK_MOE_CACHE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
}

pub struct ExpertCache {
    pub slots: usize,
    pub n_expert: usize,
    /// Rows the per-launch scratch can take (`idx_slot`, `jobs`).
    pub max_rows: usize,
    pub gate: RepackedKQ,
    pub up: RepackedKQ,
    pub down: RepackedKQ,
    slot_of: CudaSlice<u32>,
    expert_in: CudaSlice<u32>,
    last_use: CudaSlice<u32>,
    tick: CudaSlice<u32>,
    idx_slot: CudaSlice<u32>,
    jobs: CudaSlice<u32>,
    n_jobs: CudaSlice<u32>,
    /// `[rows resolved, misses]`, accumulated by every resolve.
    stats: CudaSlice<u32>,
    /// Expert-major prefill (waves): a prefill launch whose routed rows
    /// exceed `slots` is served in `n_waves` (= ceil(n_expert / slots))
    /// passes over the token-batched pair, each pass marking every
    /// out-of-wave pair ABSENT (`MOE_CACHE_NONE` in the routing, which the
    /// pair kernels skip). See `pd_moe_wave_plan` in the pack.
    pub n_waves: usize,
    /// `wave_of[n_expert]`: the wave each present expert landed in (NONE
    /// when absent from the launch), written by the plan.
    wave_of: CudaSlice<u32>,
    /// `wave_ids[n_waves * slots]`, `wave_cnt[n_waves]`: the plan's lists.
    wave_ids: CudaSlice<u32>,
    wave_cnt: CudaSlice<u32>,
    /// The wave's remapped routing (`max_rows`), written by the mask.
    idx_wave: CudaSlice<u32>,
    /// The wave's compacted in-wave pair indices (`max_rows`) + device
    /// count, and the tokens holding at least one (`max_rows`, over-sized)
    /// + count - what the LIST pair kernels stride over.
    pairs_w: CudaSlice<u32>,
    n_pairs_w: CudaSlice<u32>,
    rows_w: CudaSlice<u32>,
    n_rows_w: CudaSlice<u32>,
    /// Fill descriptors: mirror sources, slot destinations, bytes per expert -
    /// gate data, gate scales, up data, up scales, down data, down scales.
    src: [u64; 6],
    dst: [u64; 6],
    bytes: [u64; 6],
}

impl ExpertCache {
    /// Remapped routing (slot ids) written by the last resolve; the MoE
    /// kernels take it in place of the expert ids.
    pub fn idx_slot(&self) -> &CudaSlice<u32> {
        &self.idx_slot
    }

    /// The routing the wave mask wrote (the MoE pair takes it like
    /// `idx_slot` on the wave path).
    pub fn idx_wave(&self) -> &CudaSlice<u32> {
        &self.idx_wave
    }

    /// The wave's pair list + count and token list + count (see the mask).
    pub fn wave_lists(
        &self,
    ) -> (
        &CudaSlice<u32>,
        &CudaSlice<u32>,
        &CudaSlice<u32>,
        &CudaSlice<u32>,
    ) {
        (&self.pairs_w, &self.n_pairs_w, &self.rows_w, &self.n_rows_w)
    }

    /// `(rows resolved, misses)` since load - a sync + tiny readback, for
    /// logs and gates, never on the tick.
    pub fn stats(&self, exec: &GpuExecutor) -> Result<(u64, u64), GpuError> {
        let v = exec.to_host_u32(&self.stats)?;
        Ok((v[0] as u64, v[1] as u64))
    }

    /// VRAM the slot planes hold.
    pub fn vram_bytes(&self) -> u64 {
        (self.gate.data.len()
            + self.gate.scales.len()
            + self.up.data.len()
            + self.up.scales.len()
            + self.down.data.len()
            + self.down.scales.len()) as u64
    }

    /// Bytes one slot (one expert across the three planes) costs, from the
    /// mirrors' per-expert strides.
    pub fn slot_bytes(gate: &HostMappedKq, up: &HostMappedKq, down: &HostMappedKq) -> u64 {
        let per = |p: &RepackedKQ| ((p.data.len() + p.scales.len()) / p.dims[2]) as u64;
        per(gate) + per(up) + per(down)
    }
}

impl GpuExecutor {
    pub fn has_moe_cache(&self) -> bool {
        self.kernels.moe_cache_resolve.is_some() && self.kernels.moe_cache_fill.is_some()
    }

    /// The expert-major prefill path (slots 580-582) is in the pack.
    pub fn has_moe_wave(&self) -> bool {
        self.kernels.moe_wave_plan.is_some()
            && self.kernels.moe_cache_resolve_dev.is_some()
            && self.kernels.moe_wave_mask.is_some()
            && self.kernels.kquant_moe_gate_up_list.is_some()
            && self.kernels.kquant_moe_down_list.is_some()
    }

    /// Build a `slots`-expert cache over three host-mapped planes of one
    /// layer. Slot planes are allocated empty; the first ticks fill them.
    pub fn new_expert_cache(
        &self,
        gate: &HostMappedKq,
        up: &HostMappedKq,
        down: &HostMappedKq,
        slots: usize,
        max_rows: usize,
    ) -> Result<ExpertCache, GpuError> {
        let n_expert = gate.dims[2];
        if slots == 0 || slots > n_expert || up.dims[2] != n_expert || down.dims[2] != n_expert {
            return Err(GpuError::Driver(format!(
                "expert cache: {slots} slots over {n_expert} experts (up {}, down {})",
                up.dims[2], down.dims[2]
            )));
        }
        let mut src = [0u64; 6];
        let mut dst = [0u64; 6];
        let mut bytes = [0u64; 6];
        let mut planes = Vec::with_capacity(3);
        for (i, p) in [gate, up, down].into_iter().enumerate() {
            let per_data = p.data.len() / n_expert;
            let per_scales = p.scales.len() / n_expert;
            if per_data * n_expert != p.data.len() || per_scales * n_expert != p.scales.len() {
                return Err(GpuError::Driver(
                    "expert cache: plane bytes are not a whole number of experts".into(),
                ));
            }
            let data = self.alloc_u8(per_data * slots)?;
            let scales = self.alloc_u8(per_scales * slots)?;
            {
                let (sp, _g1) = p.data.device_ptr(&self.stream);
                let (ssp, _g2) = p.scales.device_ptr(&self.stream);
                let (dp, _g3) = data.device_ptr(&self.stream);
                let (dsp, _g4) = scales.device_ptr(&self.stream);
                src[2 * i] = sp;
                src[2 * i + 1] = ssp;
                dst[2 * i] = dp;
                dst[2 * i + 1] = dsp;
            }
            bytes[2 * i] = per_data as u64;
            bytes[2 * i + 1] = per_scales as u64;
            let mut dims = p.dims.clone();
            dims[2] = slots;
            planes.push(RepackedKQ {
                data,
                scales,
                dims,
                ty: p.ty,
            });
        }
        let n_waves = n_expert.div_ceil(slots);
        let down_p = planes.pop().expect("three planes pushed");
        let up_p = planes.pop().expect("three planes pushed");
        let gate_p = planes.pop().expect("three planes pushed");
        Ok(ExpertCache {
            slots,
            n_expert,
            max_rows,
            gate: gate_p,
            up: up_p,
            down: down_p,
            slot_of: self.to_device_u32(&vec![MOE_CACHE_NONE; n_expert])?,
            expert_in: self.to_device_u32(&vec![MOE_CACHE_NONE; slots])?,
            last_use: self.to_device_u32(&vec![0u32; slots])?,
            tick: self.to_device_u32(&[0u32])?,
            idx_slot: self.to_device_u32(&vec![0u32; max_rows])?,
            jobs: self.to_device_u32(&vec![0u32; 2 * max_rows])?,
            n_jobs: self.to_device_u32(&[0u32])?,
            stats: self.to_device_u32(&[0u32, 0u32])?,
            n_waves,
            wave_of: self.to_device_u32(&vec![MOE_CACHE_NONE; n_expert])?,
            wave_ids: self.to_device_u32(&vec![0u32; n_waves * slots])?,
            wave_cnt: self.to_device_u32(&vec![0u32; n_waves])?,
            idx_wave: self.to_device_u32(&vec![0u32; max_rows])?,
            pairs_w: self.to_device_u32(&vec![0u32; max_rows])?,
            n_pairs_w: self.to_device_u32(&[0u32])?,
            rows_w: self.to_device_u32(&vec![0u32; max_rows])?,
            n_rows_w: self.to_device_u32(&[0u32])?,
            src,
            dst,
            bytes,
        })
    }

    /// Resolve `rows` routed ids (`idx`) against the cache: writes
    /// `idx_slot`, updates the LRU state, records miss jobs. `rows` must not
    /// exceed the cache's slots (a tick never evicts what it reads).
    pub fn moe_cache_resolve(
        &self,
        c: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_cache_resolve
            .ok_or(GpuError::MissingOp("moe_cache_resolve"))?;
        if rows > c.slots || rows > c.max_rows {
            return Err(GpuError::Driver(format!(
                "expert cache resolve: {rows} rows over {} slots / {} scratch rows",
                c.slots, c.max_rows
            )));
        }
        let (ip, _g0) = idx.device_ptr(&self.stream);
        let (so, _g1) = c.slot_of.device_ptr(&self.stream);
        let (ei, _g2) = c.expert_in.device_ptr(&self.stream);
        let (lu, _g3) = c.last_use.device_ptr(&self.stream);
        let (tk, _g4) = c.tick.device_ptr(&self.stream);
        let (is, _g5) = c.idx_slot.device_ptr(&self.stream);
        let (jb, _g6) = c.jobs.device_ptr(&self.stream);
        let (nj, _g7) = c.n_jobs.device_ptr(&self.stream);
        let (st, _g8) = c.stats.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract; the state buffers are written by the
        // kernel in stream order and read by nothing else off-stream.
        check(unsafe {
            f(
                ip as *const _,
                rows as u32,
                c.slots as u32,
                so as *mut _,
                ei as *mut _,
                lu as *mut _,
                tk as *mut _,
                is as *mut _,
                jb as *mut _,
                nj as *mut _,
                st as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Expert-major prefill, step 1: plan the waves for `rows` routed ids.
    pub fn moe_wave_plan(
        &self,
        c: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_wave_plan
            .ok_or(GpuError::MissingOp("moe_wave_plan"))?;
        if rows > c.max_rows {
            return Err(GpuError::Driver(format!(
                "expert wave plan: {rows} rows over {} scratch rows",
                c.max_rows
            )));
        }
        let (ip, _g0) = idx.device_ptr(&self.stream);
        let (wo, _g1) = c.wave_of.device_ptr(&self.stream);
        let (wi, _g2) = c.wave_ids.device_ptr(&self.stream);
        let (wc, _g3) = c.wave_cnt.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 580); buffers sized at creation
        check(unsafe {
            f(
                ip as *const _,
                rows as u32,
                c.n_expert as u32,
                c.slots as u32,
                c.n_waves as u32,
                wo as *mut _,
                wi as *mut _,
                wc as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Expert-major prefill, step 2 (per wave): resolve the wave's ids
    /// through the LRU and record the miss jobs - `moe_cache_fill(c,
    /// c.slots)` copies them.
    pub fn moe_wave_resolve(&self, c: &ExpertCache, wave: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_cache_resolve_dev
            .ok_or(GpuError::MissingOp("moe_cache_resolve_dev"))?;
        let (wi, _g0) = c.wave_ids.device_ptr(&self.stream);
        let (wc, _g1) = c.wave_cnt.device_ptr(&self.stream);
        let (so, _g2) = c.slot_of.device_ptr(&self.stream);
        let (ei, _g3) = c.expert_in.device_ptr(&self.stream);
        let (lu, _g4) = c.last_use.device_ptr(&self.stream);
        let (tk, _g5) = c.tick.device_ptr(&self.stream);
        let (jb, _g6) = c.jobs.device_ptr(&self.stream);
        let (nj, _g7) = c.n_jobs.device_ptr(&self.stream);
        let (st, _g8) = c.stats.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 581); the id list and its count
        // are the plan's device outputs, offset to this wave
        check(unsafe {
            f(
                (wi as usize + wave * c.slots * 4) as *const _,
                (wc as usize + wave * 4) as *const _,
                c.slots as u32,
                so as *mut _,
                ei as *mut _,
                lu as *mut _,
                tk as *mut _,
                jb as *mut _,
                nj as *mut _,
                st as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Expert-major prefill, step 3 (per wave): the wave's routing into
    /// `idx_wave` - in-wave pairs to their slots, the rest marked absent.
    pub fn moe_wave_mask(
        &self,
        c: &ExpertCache,
        idx: &CudaSlice<u32>,
        rows: usize,
        n_active: usize,
        wave: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_wave_mask
            .ok_or(GpuError::MissingOp("moe_wave_mask"))?;
        let (ip, _g0) = idx.device_ptr(&self.stream);
        let (wo, _g1) = c.wave_of.device_ptr(&self.stream);
        let (so, _g2) = c.slot_of.device_ptr(&self.stream);
        let (iw, _g3) = c.idx_wave.device_ptr(&self.stream);
        let (pw, _g4) = c.pairs_w.device_ptr(&self.stream);
        let (npw, _g5) = c.n_pairs_w.device_ptr(&self.stream);
        let (rw, _g6) = c.rows_w.device_ptr(&self.stream);
        let (nrw, _g7) = c.n_rows_w.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract (slot 582); rows <= max_rows checked by the plan
        check(unsafe {
            f(
                ip as *const _,
                rows as u32,
                n_active as u32,
                wo as *const _,
                so as *const _,
                wave as u32,
                MOE_CACHE_NONE,
                iw as *mut _,
                pw as *mut _,
                npw as *mut _,
                rw as *mut _,
                nrw as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Copy the last resolve's misses (at most `rows` of them) from the host
    /// mirror into their slots.
    pub fn moe_cache_fill(&self, c: &ExpertCache, rows: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_cache_fill
            .ok_or(GpuError::MissingOp("moe_cache_fill"))?;
        let (jb, _g1) = c.jobs.device_ptr(&self.stream);
        let (nj, _g2) = c.n_jobs.device_ptr(&self.stream);
        // SAFETY: pack ABI v1 contract; src/dst/bytes are host arrays the
        // launcher copies by value before returning.
        check(unsafe {
            f(
                jb as *const _,
                nj as *const _,
                rows as u32,
                c.src.as_ptr() as *const _,
                c.dst.as_ptr() as *const _,
                c.bytes.as_ptr() as *const _,
                self.stream_ptr(),
            )
        })
    }
}
