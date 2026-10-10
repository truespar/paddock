//! Aux blobs as first-class tier objects - the hybrid families' resume state
//! (DeltaNet/Mamba recurrent state, sliding-window rings) in RAM and on disk.
//!
//! A hybrid prefix restores only through consecutive runs from the chain head
//! AND the blob at the boundary, so this side of the tier keeps the two
//! together and finds a blob wherever it lives:
//!
//! - **Bindings.** Each blob is bound to the runs it resumes over
//!   (`blob_runs`, `run_owners`). T1 evicts runs no blob owns first, and an
//!   owned run only together with its blobs (see `evict_t1`).
//! - **Dominance.** A blob with a deeper blob over the same chain serves only
//!   prompts that leave the chain between the two - the next turn of a
//!   conversation resumes at the deeper one. Such blobs go first: a chain
//!   keeps its deepest state, the way vLLM and SGLang keep one recurrent
//!   state per request, while the shallower cuts we take for divergent
//!   tails survive whenever there is room. On a Qwen3.8-27B a blob is
//!   ~157 MB against 4 MiB per 128-token run, so this is most of T1.
//! - **Any tier.** The family declares its blob geometry once
//!   ([`PoolTier::declare_blob_pages`] / [`PoolTier::declare_blob_flat`]), so
//!   a probe finds a blob by its content keys on whichever tier holds every
//!   shard - T1, or T2 after a T1 eviction or a restart - not only through
//!   the in-memory T1 inventory. A T1 eviction publishes the durable copy
//!   (as runs always did), and a disk-sourced restore seats the blob back in
//!   T1 whole, or not at all: its shards never pose as runs.

use std::collections::HashMap;

use super::*;

/// A blob being seated in T1 by a disk-sourced restore: its shards come back
/// through the transport's read-fill outbox one by one.
pub(super) struct Fill {
    shards: usize,
    bytes: u64,
    runs: Vec<LogicalKey>,
    /// The restore's loads have all completed - every promotion it will
    /// produce is in the outbox by the next pump.
    resolved: bool,
}

impl<T: XferSink> PoolTier<T> {
    // -- geometry ----------------------------------------------------------

    /// Declare this model's blob: a checkpoint of `pages` pool pages (paged
    /// families). Idempotent; lets a probe find blobs the T1 inventory does
    /// not hold.
    pub fn declare_blob_pages(&mut self, pages: usize) {
        if pages > 0 {
            self.blob_geom = Some((
                pages as u64 * self.record_stride,
                pages.div_ceil(self.pages_per_shard()),
            ));
        }
    }

    /// Declare this model's blob: one flat checkpoint of `bytes` (families
    /// with a fixed checkpoint pool).
    pub fn declare_blob_flat(&mut self, bytes: u64) {
        if bytes > 0 {
            self.blob_geom = Some((bytes, bytes.div_ceil(Self::AUX_SHARD) as usize));
        }
    }

    // -- bindings ----------------------------------------------------------

    /// Keys of `path`'s complete runs through block `depth`, root-first -
    /// the runs a checkpoint at `depth` resumes over.
    pub(super) fn path_run_keys(
        &self,
        path: &[crate::paged_radix::LruPathEntry],
        depth: usize,
    ) -> Vec<LogicalKey> {
        let r = self.run_blocks;
        let end = depth.min(path.len()) / r * r;
        (0..end)
            .step_by(r)
            .map_while(|lo| {
                let run = &path[lo..lo + r];
                run.iter()
                    .all(|e| e.tkey.is_some())
                    .then_some(run[r - 1].tkey)
                    .flatten()
            })
            .collect()
    }

    /// The same keys straight from a prompt's tokens (a restore that has no
    /// radix path to read them off).
    fn token_run_keys(&self, tokens: &[u32], end_block: usize) -> Vec<LogicalKey> {
        let r = self.run_blocks;
        let mut key = self.ns_root;
        let mut out = Vec::with_capacity(end_block / r);
        for b in 0..end_block.min(tokens.len() / BLOCK_TOKENS) {
            key = key.child(&tokens[b * BLOCK_TOKENS..(b + 1) * BLOCK_TOKENS]);
            if (b + 1) % r == 0 {
                out.push(key);
            }
        }
        out
    }

    /// Bind boundary `blob` to the runs it resumes over (replacing any
    /// earlier binding) and stamp them now: a run is as recent as the
    /// newest boundary over it.
    pub(super) fn own_runs(&mut self, blob: LogicalKey, runs: Vec<LogicalKey>) {
        let now = self.tick();
        for k in &runs {
            *self.run_owners.entry(*k).or_insert(0) += 1;
            if let Some(m) = self.runs.get_mut(k) {
                m.last_used = now;
            }
        }
        if let Some(old) = self.blob_runs.insert(blob, runs) {
            self.unown(&old);
        }
    }

    /// Undo [`Self::own_runs`] - the blob retired or never stored.
    pub(super) fn disown_runs(&mut self, blob: LogicalKey) {
        if let Some(old) = self.blob_runs.remove(&blob) {
            self.unown(&old);
        }
    }

    fn unown(&mut self, runs: &[LogicalKey]) {
        for k in runs {
            if let Some(n) = self.run_owners.get_mut(k) {
                *n -= 1;
                if *n == 0 {
                    self.run_owners.remove(k);
                }
            }
        }
    }

    /// Bindings whose blob never stored - a claimed checkpoint the family
    /// recycled instead of demoting. Claims resolve within the call that
    /// made them, so any left at the next pressure pass are stale.
    pub(super) fn drop_stale_claims(&mut self) {
        let stale: Vec<LogicalKey> = self
            .blob_runs
            .keys()
            .filter(|k| !self.aux_meta.contains_key(k))
            .copied()
            .collect();
        for k in stale {
            self.disown_runs(k);
        }
    }

    // -- eviction order ----------------------------------------------------

    /// T1 blobs in eviction order: dominated ones first (a deeper blob is
    /// bound to this one's last run - its key - so it owns that run too),
    /// then the rest, least recently used first within each class.
    pub(super) fn blob_victims(&self) -> Vec<LogicalKey> {
        let mut v: Vec<(bool, u64, LogicalKey)> = self
            .aux_meta
            .iter()
            .map(|(k, m)| {
                let dominated = self.run_owners.get(k).is_some_and(|&n| n >= 2);
                (!dominated, m.last_used, *k)
            })
            .collect();
        v.sort_unstable_by_key(|&(leaf, used, _)| (leaf, used));
        v.into_iter().map(|(_, _, k)| k).collect()
    }

    // -- probe ---------------------------------------------------------------

    /// Deepest aux boundary at or below `max_block` for this prompt whose
    /// every shard is Ready on some tier - the position a hybrid family can
    /// actually resume at. T1 blobs come from the inventory; anything else
    /// (T2 after a T1 eviction or a restart) is found by its content keys
    /// with the declared geometry. Bumps a T1 boundary's LRU.
    pub fn probe_aux(&mut self, tokens: &[u32], max_block: usize) -> Option<AuxHit> {
        if self.tripped {
            return None;
        }
        let full = (tokens.len().saturating_sub(1) / BLOCK_TOKENS).min(max_block);
        let r = self.run_blocks;
        let mut key = self.ns_root;
        let mut keys_at = Vec::with_capacity(full);
        for b in 0..full {
            key = key.child(&tokens[b * BLOCK_TOKENS..(b + 1) * BLOCK_TOKENS]);
            keys_at.push(key);
        }
        for b in (1..=full).rev() {
            let k = keys_at[b - 1];
            let geom = match self.aux_meta.get(&k) {
                Some(m) => (m.bytes, m.shards),
                None => match self.blob_geom {
                    Some(g) if b % r == 0 => g,
                    _ => continue,
                },
            };
            let (bytes, shards) = geom;
            let mut nvme_bytes = 0u64;
            let mut all = true;
            for i in 0..shards {
                let sk = k.child_bytes("aux", &(i as u32).to_le_bytes());
                match self.ready_on(&sk) {
                    Some((Tier::Nvme, n)) => nvme_bytes += n,
                    Some(_) => {}
                    None => {
                        all = false;
                        break;
                    }
                }
            }
            if all {
                let now = self.tick();
                if let Some(m) = self.aux_meta.get_mut(&k) {
                    m.last_used = now;
                }
                return Some(AuxHit {
                    key: k,
                    end_block: b,
                    bytes,
                    shards,
                    nvme_bytes,
                });
            }
        }
        None
    }

    // -- retire ----------------------------------------------------------------

    /// Drop an aux boundary from T1: evict every shard entry and free its
    /// extents. A shard with a durable copy stays readable from T2 - the
    /// catalog takes the disk replica, exactly as a run's eviction does - so
    /// a blob written through to disk is still a restore, not a loss.
    pub(super) fn retire_aux(&mut self, key: LogicalKey) {
        self.disown_runs(key);
        let Some(m) = self.aux_meta.remove(&key) else {
            return;
        };
        let mut durable = 0;
        for i in 0..m.shards {
            let sk = key.child_bytes("aux", &(i as u32).to_le_bytes());
            let loc = self.catalog.ready_loc(&sk, Tier::Ram);
            if self.catalog.evict(&sk, Tier::Ram).is_ok()
                && let Some(l) = loc
            {
                self.transport.free_extent(l);
            }
            if let Some((t2loc, bytes, sum)) = self.transport.t2_entry(&sk.0)
                && (self.catalog.ready_bytes(&sk, Tier::Nvme).is_some()
                    || self.catalog.preload_ready(
                        sk,
                        Tier::Nvme,
                        t2loc,
                        super::super::digest::Checksum(sum),
                        bytes,
                    ))
            {
                durable += 1;
            }
        }
        if durable == m.shards {
            self.dec.promoted_to_disk += 1;
        }
    }

    // -- read-fill ---------------------------------------------------------

    /// A disk-sourced blob restore is starting: note its shards so their
    /// read-fill promotions seat the blob in T1 whole (bound to the runs
    /// under it, from `tokens`) instead of landing as loose entries.
    pub fn expect_aux_fill(&mut self, hit: &AuxHit, tokens: &[u32]) {
        if hit.nvme_bytes == 0 || self.aux_meta.contains_key(&hit.key) {
            return;
        }
        for i in 0..hit.shards {
            let sk = hit.key.child_bytes("aux", &(i as u32).to_le_bytes());
            self.fill_of.insert(sk, hit.key);
        }
        let runs = self.token_run_keys(tokens, hit.end_block);
        self.fills.insert(
            hit.key,
            Fill {
                shards: hit.shards,
                bytes: hit.bytes,
                runs,
                resolved: false,
            },
        );
    }

    /// A read-fill promotion for a blob shard: adopt it into the catalog as
    /// part of its blob's fill. False when `key` is not a shard of a fill (a
    /// run - the caller's business).
    pub(super) fn adopt_shard_promotion(
        &mut self,
        key: LogicalKey,
        loc: Loc,
        sum: [u8; 32],
        len: u64,
    ) -> bool {
        if !self.fill_of.contains_key(&key) {
            return false;
        }
        if !self.catalog.preload_ready(
            key,
            Tier::Ram,
            loc,
            super::super::digest::Checksum(sum),
            len,
        ) {
            self.transport.free_extent(loc);
        }
        true
    }

    /// The blob restore that `shard_key` belongs to has resolved.
    pub(super) fn fill_resolved(&mut self, shard_key: &LogicalKey) {
        if let Some(b) = self.fill_of.get(shard_key).copied()
            && let Some(f) = self.fills.get_mut(&b)
        {
            f.resolved = true;
        }
    }

    /// Settle resolved fills (after this pump's promotions were adopted): a
    /// blob with every shard back in T1 joins the inventory, bound to its
    /// runs; a partial one gives its T1 shards back - a blob is restorable
    /// whole or not at all, and an uninventoried T1 entry could never be
    /// evicted.
    pub(super) fn settle_fills(&mut self) {
        let done: Vec<LogicalKey> = self
            .fills
            .iter()
            .filter(|(_, f)| f.resolved)
            .map(|(k, _)| *k)
            .collect();
        for b in done {
            let Some(f) = self.fills.remove(&b) else {
                continue;
            };
            let shard_keys: Vec<LogicalKey> = (0..f.shards)
                .map(|i| b.child_bytes("aux", &(i as u32).to_le_bytes()))
                .collect();
            for sk in &shard_keys {
                self.fill_of.remove(sk);
            }
            let whole = shard_keys
                .iter()
                .all(|sk| self.catalog.ready_bytes(sk, Tier::Ram).is_some());
            if whole && !self.aux_meta.contains_key(&b) {
                let now = self.tick();
                self.aux_meta.insert(
                    b,
                    AuxMeta {
                        bytes: f.bytes,
                        shards: f.shards,
                        last_used: now,
                    },
                );
                self.own_runs(b, f.runs);
            } else if !whole && !self.aux_meta.contains_key(&b) {
                for sk in &shard_keys {
                    let loc = self.catalog.ready_loc(sk, Tier::Ram);
                    if self.catalog.evict(sk, Tier::Ram).is_ok()
                        && let Some(l) = loc
                    {
                        self.transport.free_extent(l);
                    }
                }
            }
        }
    }
}

/// The state of every fill, by boundary (`PoolTier::fills`).
pub(super) type Fills = HashMap<LogicalKey, Fill>;
