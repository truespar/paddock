//! Granite prefix cache: a [`PagedRadix`] over the one full-attention block
//! pool. A hit ADOPTS blocks by refcount - nothing is copied.
//!
//! Granite is the easy case and it is worth saying why, because the families
//! that came first are all harder and their machinery does not apply:
//!
//! - **Every layer is full attention and pooled.** There is no recurrent state
//!   and no sliding window, so there is nothing to snapshot. qwen35 and laguna
//!   can only resume at a position whose non-poolable state was checkpointed,
//!   and throw away a perfectly good block match when none is reachable
//!   (`laguna/prefix.rs`: `let Some((pos, _)) = m.ckpt else { return Ok(0) }`).
//!   Granite resumes at every matched page, so its reuse is strictly better.
//!   No `attach_state`, no checkpoint pool, no `PagedMatch::ckpt`.
//! - **The two-boundary checkpoint rule does not apply.** That rule
//!   (`qwen35::ckpt_cuts`) is about where a STATE snapshot lands so the next
//!   turn can reach it. With no state there is no boundary to miss.
//! - **No copy-on-write.** Only whole 16-token blocks are cached, so a resume
//!   is block-aligned and the tail re-prefill starts at the cut - an adopted
//!   block is never written.
//!
//! ## Images
//!
//! Every image row of every prompt carries the same `<image>` placeholder id,
//! so a radix keyed on the row tokens would treat two different pictures as
//! the same prefix and serve one image's KV for another. That is the same
//! class as the bug qwen35 fixed at `prefix.rs:414` - "of two concurrent image
//! requests, the blue-image slot answered red" - and it is why multimodal
//! prompts were excluded from prefix caching engine-wide.
//!
//! The fix is to key the radix on a vector that is not the row tokens: text
//! rows contribute their token id, and each image row contributes a value
//! derived from the picture's CONTENT hash and the row's offset within it (see
//! [`image_key_row`]). Identical pictures produce identical keys and hit;
//! different pictures produce different keys and miss. The high bit is set on
//! every image-derived key so the two key spaces are provably disjoint from
//! real token ids - a text prompt can never adopt an image's blocks by
//! coincidence, whatever the hash does.
//!
//! This is what makes the document workload work: same page, many questions,
//! and every turn after the first resumes past the whole picture instead of
//! re-prefilling ~1.5-2.5k rows of it.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::BLOCK_TOKENS;
use crate::kv_tier::digest::{IdentityDigest, IdentityFields, PrivacyScope};
use crate::kv_tier::{CacheNamespace, Election, PlaneDesc, PoolTier, RamTransport};

use super::GpuGranite;

/// The granite tier instance type - full-attention T1 over the one block
/// pool (granite is the simple lane because every layer
/// is pooled full attention, so restored blocks alone are a usable resume -
/// no checkpoint requirement, unlike the SWA/hybrid families whose payloads
/// land in 1b.3).
pub(crate) type GraniteTier = PoolTier<RamTransport>;

pub(crate) use crate::kv_tier::pool_tier::tier_ram_bytes;

/// Don't resume prefixes shorter than this. The restore is only a refcount
/// bump plus one block-table upload, so the floor is about not churning the
/// radix on trivial prompts rather than about restore cost (laguna's 64 pays
/// for a 63 MB window blob; granite pays for neither).
const MIN_CACHE_PREFIX: usize = 32;

/// Blocks kept in reserve for radix retention when sizing the pool.
///
/// Sized explicitly rather than carried over from laguna's `+ 8` slack, per
/// the warning in `enable_batch_impl`: on granite-30b all 64 layers are
/// full-attention, so one block id costs 4 MiB here against laguna's ~1 MiB.
/// A blind slack of 8 slots' worth reserved 24 GiB when 8 GiB was addressable.
/// `PADDOCK_GRANITE_PREFIX_BLOCKS` overrides; 0 disables retention headroom
/// (the cache then lives entirely off blocks the pool would otherwise idle).
pub(crate) fn retention_blocks() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_GRANITE_PREFIX_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}

/// Free blocks the insert path evicts down to, so eviction happens off the
/// admission path. `PADDOCK_GRANITE_PX_MARGIN` overrides; 0 = evict only under
/// real pressure.
fn px_margin() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_GRANITE_PX_MARGIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

/// Engine-wide off switch, honoured by every family.
pub(crate) fn prefix_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
}

/// The image row key, shared with every other family that caches across
/// pictures - see [`crate::gpu_model::prefix_cache::image_key_row`] for why it
/// looks the way it does. Re-exported rather than redefined: three copies of a
/// hash that must agree is three chances to disagree.
pub(crate) use crate::gpu_model::prefix_cache::image_key_row;

impl GpuGranite {
    /// Match `keys` against the radix and adopt the matched blocks into
    /// `slot`'s table. Returns the resume position - a block-aligned token
    /// count already present in KV, so the caller starts its prefill there.
    ///
    /// `keys` is the RADIX key vector, which equals the row tokens for a
    /// text-only prompt and substitutes content-derived values at image rows.
    /// The caller must pass the same vector to [`Self::prefix_insert`] or the
    /// prompt will never hit itself.
    pub(crate) fn prefix_resume(
        &mut self,
        slot: usize,
        keys: &[u32],
    ) -> Result<usize, GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return Ok(0);
        };
        self.last_reused[slot] = 0;
        // TIER (D5 park/wake): the restore was consulted and parked at
        // admission (`tier_prefix_loading`); an elected restore has already
        // published by the time prefill runs. Pump for freshness so paths
        // that skip the consult still pick up published prefixes.
        if let Some(tier) = bs.tier.as_mut() {
            tier.pump_completions(radix, &mut bs.pool);
        }
        let blocks = radix.match_prefix(keys);
        let pos = blocks.len() * BLOCK_TOKENS;
        if pos < MIN_CACHE_PREFIX || pos >= keys.len() {
            return Ok(0);
        }
        bs.tables[slot].clear(&mut bs.pool);
        bs.tables[slot].share_prefix(&blocks, &mut bs.pool);
        let base = slot * bs.bps;
        for (j, &b) in bs.tables[slot].blocks().iter().enumerate() {
            bs.bt_host[base + j] = b;
        }
        self.exec
            .stream
            .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.last_reused[slot] = pos;
        Ok(pos)
    }

    /// Publish a finished prompt's blocks into the radix, then evict down to
    /// the free margin so the next admission does not pay for it.
    pub(crate) fn prefix_insert(&mut self, slot: usize, keys: &[u32]) {
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return;
        };
        let blocks = bs.tables[slot].blocks().to_vec();
        radix.insert(keys, &blocks, &mut bs.pool);
        let margin = crate::gpu_model::prefix_cache::evict_ahead_margin(
            px_margin(),
            bs.pool.capacity() as usize,
        );
        if let Some(tier) = bs.tier.as_mut() {
            // tier-aware evict-ahead: same LRU order, but a run whose
            // closing leaf goes is captured into T1 first (demote-on-evict)
            if margin > 0 && bs.pool.free_blocks() < margin {
                let after = exec.record_event().ok();
                let (_evicted, aux) = tier.pressure_demote(radix, &mut bs.pool, margin, after);
                // granite has no state pool - every claimed checkpoint just
                // recycles (hybrid families map these to blob spans, 1b.3)
                for a in aux {
                    radix.recycle_state(a.state_idx);
                }
            }
            // opportunistic completion drain (releases demote pins)
            tier.pump_completions(radix, &mut bs.pool);
            return;
        }
        while margin > 0 && bs.pool.free_blocks() < margin {
            let Some(radix) = bs.prefix.as_mut() else {
                break;
            };
            if radix.evict_lru(&mut bs.pool).is_none() {
                break;
            }
        }
    }

    /// Blocks the radix is holding that could be reclaimed - added to the
    /// pool's free count for admission accounting so the cache behaves as
    /// reclaimable capacity and not as a reservation. Without this a
    /// retention-heavy workload drives free to ~0 and the admission watermark
    /// serializes the server behind slot completions.
    pub(crate) fn prefix_evictable(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|bs| bs.prefix.as_ref().map(|r| r.evictable_blocks(&bs.pool)))
            .unwrap_or(0)
    }
}

impl GpuGranite {
    /// The tier's shutdown flush (see `Generator::tier_flush_at_exit`).
    pub(crate) fn tier_flush_impl(&mut self) {
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let (Some(tier), Some(radix)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return;
        };
        tier.flush_live(radix, &mut bs.pool, &mut || exec.record_event().ok(), None);
    }

    /// The per-tick tier pump (see `Generator::tier_pump`).
    pub(crate) fn tier_pump_impl(&mut self) {
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let (Some(tier), Some(radix)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return;
        };
        tier.pump_completions(radix, &mut bs.pool);
        tier.pump_flows(radix, &mut || exec.record_event().ok());
        // 2.3 write-through: pre-store retained chains in slack so later
        // eviction is free
        tier.mirror_slack(radix, &mut bs.pool, exec.record_event().ok(), 2, None);
    }

    /// The D5 admission consult (park/wake) - KV-only: probe + elect and,
    /// when elected, start the restore and PARK the request; publication
    /// alone is the win (granite resumes at every matched page, no
    /// checkpoint requirement).
    pub(crate) fn tier_consult_impl(&mut self, slot: usize, keys: &[u32]) -> bool {
        use crate::kv_tier::FlowStatus;
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return false;
        };
        let (Some(tier), Some(radix)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return false;
        };
        tier.pump_completions(radix, &mut bs.pool);
        {
            let exec2 = exec.clone();
            tier.pump_flows(radix, &mut || exec2.record_event().ok());
        }
        match tier.flow_status(slot, keys) {
            FlowStatus::Loading => return true,
            FlowStatus::Done { .. } => return false,
            FlowStatus::None => {}
        }
        let gpu_blocks = radix.match_prefix(keys).len();
        let Some(hit) = tier.probe(keys, gpu_blocks) else {
            return false;
        };
        let e = tier.elect(&hit, gpu_blocks);
        tracing::debug!(election = ?e, "granite tier election");
        let Election::Restore { est_us, .. } = e else {
            return false;
        };
        let after = exec.record_event().ok();
        match crate::kv_tier::RestoreFlow::begin(
            tier,
            &mut bs.pool,
            keys,
            &hit,
            None,
            est_us,
            after,
        ) {
            Some(flow) => {
                tier.park_flow(slot, flow);
                tracing::debug!(
                    slot,
                    boundary = hit.end_block,
                    "granite tier: restore parked (D5)"
                );
                true
            }
            None => false,
        }
    }
}

/// Build the T1 tier for granite's pool (called from `enable_batch_impl`
/// once the KV planes exist). Every failure is a loud decline - serving
/// continues untiered, never half-tiered.
pub(crate) fn build_tier(
    exec: &crate::gpu::GpuExecutor,
    hp: &super::Hparams,
    kv_dtype: crate::gpu::KvDtype,
    kv: &[super::batch::LayerKv],
    max_ctx: usize,
    ram_bytes: u64,
    content_id: ([u8; 32], [u8; 32]),
) -> Option<GraniteTier> {
    use cudarc::driver::DevicePtr;
    let kv_dim = hp.n_kv_heads * hp.head_dim;
    let stride = (16 * kv_dim * kv_dtype.bytes()) as u64;
    let mut planes = Vec::with_capacity(kv.len() * 2);
    for l in kv {
        for plane in [&l.k, &l.v] {
            let (p, _g) = plane.device_ptr(&exec.stream);
            planes.push(PlaneDesc {
                base: p,
                stride,
                bytes: stride,
            });
        }
    }
    // Identity: geometry + dtype + layout revision. The RAM tier is
    // per-process and dies with the runner, so weight-content identity is
    // not yet load-bearing - it becomes so (and gets the real tensor
    // hashes) with T2 persistence, where a stale cache could
    // cross runs.
    let architecture = format!(
        "granite v1 layers={} kv_dim={} dtype={:?} max_ctx={}",
        kv.len(),
        kv_dim,
        kv_dtype,
        max_ctx,
    );
    let ns = CacheNamespace {
        identity: IdentityDigest::compute(&IdentityFields {
            model_tensors: &content_id.0,
            adapter: b"",
            architecture: architecture.as_bytes(),
            cache_schema: b"pool-planes k/v interleaved v1",
            layout_abi: 1,
            tokenizer: &content_id.1,
        }),
        scope: PrivacyScope::Shared,
    };
    let transport = match crate::kv_tier::pool_tier::nvme_dir_for(&ns) {
        Some((dir, quota)) => RamTransport::with_t2(exec, ram_bytes, &dir, quota),
        None => RamTransport::new(exec, ram_bytes),
    };
    let transport = match transport {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(err = %e, "granite KV tier declined (transport)");
            return None;
        }
    };
    match PoolTier::new(&ns, planes, ram_bytes, transport) {
        Ok(mut t) => {
            t.preload_from_t2();
            Some(t)
        }
        Err(e) => {
            tracing::warn!(err = %e, "granite KV tier declined (geometry)");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MIN_CACHE_PREFIX, image_key_row};

    /// The property that makes a text prompt unable to adopt image blocks, and
    /// vice versa, regardless of what the content hash does.
    #[test]
    fn image_keys_never_collide_with_token_ids() {
        for h in [0u64, 1, 0xdead_beef, u64::MAX, 0x9e37_79b9_7f4a_7c15] {
            for j in [0usize, 1, 143, 2495] {
                let k = image_key_row(h, j);
                assert!(k & 0x8000_0000 != 0, "image key {k} lost its tag bit");
                // every real vocab this family ships is far below 2^31
                assert!(k > 1 << 30, "image key {k} lands in token-id range");
            }
        }
    }

    /// Different pictures must produce different key runs - this is the whole
    /// reason the cache can be let near an image at all.
    #[test]
    fn different_pictures_give_different_keys() {
        let a: Vec<u32> = (0..144)
            .map(|j| image_key_row(0x1111_2222_3333_4444, j))
            .collect();
        let b: Vec<u32> = (0..144)
            .map(|j| image_key_row(0x1111_2222_3333_4445, j))
            .collect();
        assert_ne!(a, b, "a one-bit content change must change the key run");
        // and not merely at one row: a radix matches 16-row pages, so a
        // difference confined to a late row would let an early page hit
        assert_ne!(a[0], b[0], "the FIRST page must already differ");
    }

    /// The same picture must key identically every time, or a document
    /// conversation never hits.
    #[test]
    fn the_same_picture_keys_identically() {
        let a: Vec<u32> = (0..144).map(|j| image_key_row(7, j)).collect();
        let b: Vec<u32> = (0..144).map(|j| image_key_row(7, j)).collect();
        assert_eq!(a, b);
    }

    /// Rows within one picture must not be interchangeable, so a partial match
    /// lines up position by position.
    #[test]
    fn rows_within_a_picture_are_ordered() {
        let rows: Vec<u32> = (0..64).map(|j| image_key_row(99, j)).collect();
        let mut uniq = rows.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), rows.len(), "rows of one image must be distinct");
    }

    #[test]
    fn the_floor_is_a_whole_number_of_pages() {
        assert_eq!(MIN_CACHE_PREFIX % crate::kv_pool::BLOCK_TOKENS, 0);
    }
}
