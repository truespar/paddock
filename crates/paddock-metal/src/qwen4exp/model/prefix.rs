//! Exact, process-local MLX prefix reuse. KV pages are immutable/refcounted;
//! every non-KV carried state is copied together in one fenced blit submission.
//! Retain a stable logical-boundary fallback plus a page-aligned prompt tail.
//! ALL arithmetic intervals of the retained prefix must match the new prompt.
//! Decode states are never substituted for prefill state.
use super::*;

fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_METAL_CACHE_TRACE").is_some())
}

pub(super) struct Entry {
    state: Buffer,
    history: Vec<u32>,
    table: BlockTable,
    touched: u64,
    contract: Vec<(usize, u16)>,
}

#[derive(Default)]
pub(super) struct PrefixCache {
    entries: Vec<Entry>,
    clock: u64,
    classes: Vec<u16>,
}

impl PrefixCache {
    pub(super) fn enabled(&self) -> bool {
        !self.entries.is_empty()
    }

    pub(super) fn reclaimable_blocks(&self, pool: &KvPool) -> usize {
        if !self.enabled() {
            return 0;
        }
        // Count each physical page once, even when backup checkpoints share
        // it. A page still owned by any live slot is not admission capacity.
        let mut refs = vec![0u32; pool.capacity() as usize];
        for entry in &self.entries {
            for &block in entry.table.blocks() {
                refs[block as usize] += 1;
            }
        }
        refs.iter()
            .enumerate()
            .filter(|&(b, &n)| n > 0 && n == pool.refcount(b as u32))
            .count()
    }
    pub(super) fn state_bytes(context: usize) -> usize {
        36 * deltanet::Cache::bytes(1)
            + 12 * (context.div_ceil(BLOCK_TOKENS) * 4 * 128 + 512) * 4
            + ple::State::cache_bytes(1)
    }

    pub(super) fn bytes(context: usize, entries: usize) -> u64 {
        (Self::state_bytes(context) * entries) as u64
    }

    pub(super) fn new(d: &MetalDevice, context: usize, count: usize) -> Result<Self> {
        let entries = (0..count)
            .map(|_| {
                Ok(Entry {
                    state: d.alloc(Self::state_bytes(context))?,
                    history: Vec::new(),
                    table: BlockTable::default(),
                    touched: 0,
                    contract: Vec::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            entries,
            clock: 0,
            classes: super::super::mlx::arithmetic_classes(),
        })
    }

    fn evict(&mut self, pool: &mut KvPool) -> bool {
        let Some(entry) = self
            .entries
            .iter_mut()
            .filter(|e| !e.history.is_empty())
            .min_by_key(|e| e.touched)
        else {
            return false;
        };
        entry.history.clear();
        entry.contract.clear();
        entry.table.clear(pool);
        true
    }

    #[cfg(test)]
    pub(super) fn clear(&mut self, pool: &mut KvPool) {
        while self.evict(pool) {}
    }
}

impl FlashNext {
    /// Copy only causally readable index rows plus ALL convolution/recurrent
    /// carries. Fixed offsets reserve the maximum index strip in each blob,
    /// but copying a short prefix does not traverse unused context storage.
    fn copy_prefix_state(
        &self,
        slot: usize,
        entry: usize,
        length: usize,
        restore: bool,
    ) -> Result<()> {
        transfer_state(
            &self.device,
            &self.prefix.entries[entry].state,
            self.layers.iter().map(|layer| match &layer.mixer {
                Mixer::Delta(_, cache) => StateLayer::Delta(cache),
                Mixer::Qsa(_, cache) => StateLayer::Qsa(cache),
            }),
            &self.scratch.ple,
            Transfer {
                slot,
                pages: self.pages,
                length,
                restore,
            },
        )
    }

    pub(super) fn restore_prefix(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        let matched = self
            .prefix
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                compatible(
                    &e.history,
                    tokens,
                    &self.slots[slot].plan,
                    &e.contract,
                    &self.prefix.classes,
                )
            })
            .max_by_key(|(_, e)| e.history.len())
            .map(|(i, e)| (i, e.history.len()));
        if trace() {
            // Log lengths/classes only, never prompt tokens or text. This
            // distinguishes absent state from an existing identical token
            // prefix rejected by the arithmetic contract.
            let longest = self
                .prefix
                .entries
                .iter()
                .filter(|e| {
                    !e.history.is_empty()
                        && e.history.len() < tokens.len()
                        && tokens.starts_with(&e.history)
                })
                .max_by_key(|e| e.history.len());
            let (candidate_tokens, stored_class, requested_class) =
                longest.map_or((0, 0, 0), |e| {
                    let logical = self.slots[slot].plan.at(e.history.len() - 1).0;
                    (
                        e.history.len(),
                        e.contract.last().map_or(0, |v| v.1),
                        self.prefix.classes[logical],
                    )
                });
            tracing::info!(
                slot,
                prompt_tokens = tokens.len(),
                reused_tokens = matched.map_or(0, |(_, n)| n),
                candidate_tokens,
                stored_class,
                requested_class,
                "Flash Next prefix restore plan"
            );
        }
        let Some((index, length)) = matched else {
            return Ok(0);
        };
        // A failed GPU copy is not a cache miss: the destination is uncertain.
        self.poisoned = true;
        self.copy_prefix_state(slot, index, length, true)?;
        let entry = &mut self.prefix.entries[index];
        self.slots[slot]
            .table
            .share_prefix(entry.table.blocks(), &mut self.pool);
        self.slots[slot].length = length;
        self.prefix.clock += 1;
        entry.touched = self.prefix.clock;
        self.poisoned = false;
        Ok(length)
    }

    pub(super) fn capture_prefix(&mut self, slot: usize, tokens: &[u32]) -> Result<()> {
        let length = self.slots[slot].length;
        let cuts = self.slots[slot].plan.cuts();
        if self.prefix.entries.is_empty()
            || length == 0
            || length >= tokens.len()
            || !cuts.contains(&length)
        {
            return Ok(());
        }
        let history = &tokens[..length];
        let contract = self.slots[slot].plan.contract(length, &self.prefix.classes);
        if self
            .prefix
            .entries
            .iter()
            .any(|e| e.history == history && e.contract == contract)
        {
            return Ok(());
        }
        let keep = if self.prefix.entries.len() >= self.slots.len() * 2 {
            cuts[0]
        } else {
            length
        };
        let Some(index) = replacement(
            self.prefix
                .entries
                .iter()
                .map(|e| (e.history.as_slice(), e.touched)),
            history,
            keep,
        ) else {
            return Ok(());
        };
        // Invalidate before overwriting; publish state and page references only
        // after successful completion. No partially captured entry is visible.
        self.prefix.entries[index].history.clear();
        self.prefix.entries[index].contract.clear();
        self.prefix.entries[index].table.clear(&mut self.pool);
        self.poisoned = true;
        self.copy_prefix_state(slot, index, length, false)?;
        let entry = &mut self.prefix.entries[index];
        entry.table.share_prefix(
            &self.slots[slot].table.blocks()[..length / BLOCK_TOKENS],
            &mut self.pool,
        );
        entry.history.extend_from_slice(history);
        entry.contract = contract;
        self.prefix.clock += 1;
        entry.touched = self.prefix.clock;
        self.poisoned = false;
        Ok(())
    }

    pub(super) fn reclaim_prefix_pages(&mut self, needed: usize) -> bool {
        while needed > self.pool.free_blocks() {
            if !self.prefix.evict(&mut self.pool) {
                return false;
            }
        }
        true
    }
}

enum StateLayer<'a> {
    Delta(&'a deltanet::Cache),
    Qsa(&'a qsa::Cache),
}

struct Transfer {
    slot: usize,
    pages: usize,
    length: usize,
    restore: bool,
}

fn transfer_state<'a>(
    device: &MetalDevice,
    blob: &'a Buffer,
    layers: impl Iterator<Item = StateLayer<'a>>,
    ple: &'a ple::State,
    transfer: Transfer,
) -> Result<()> {
    let Transfer {
        slot,
        pages,
        length,
        restore,
    } = transfer;
    let mut copies = Vec::with_capacity(98);
    let mut offset = 0;
    let mut region = |buffer: &'a Buffer, stride: usize, used: usize| {
        if restore {
            copies.push((blob, offset, buffer, slot * stride, used));
        } else {
            copies.push((buffer, slot * stride, blob, offset, used));
        }
        offset += stride;
    };
    for layer in layers {
        match layer {
            StateLayer::Delta(c) => {
                region(&c.state, deltanet::CELLS * 4, deltanet::CELLS * 4);
                region(&c.history, 3 * deltanet::CONV * 4, 3 * deltanet::CONV * 4);
            }
            StateLayer::Qsa(c) => {
                region(&c.pooled, pages * 4 * 128 * 4, length / 4 * 128 * 4);
                region(&c.ring, 512 * 4, 512 * 4);
            }
        }
    }
    region(&ple.history, 8, 8);
    region(&ple.ring, 9 * WIDE * 4, 9 * WIDE * 4);
    if offset != blob.len() {
        return Err(MetalError::Model(
            "Flash Next checkpoint layout mismatch".into(),
        ));
    }
    device.copy_regions(&copies)
}

fn compatible(
    history: &[u32],
    tokens: &[u32],
    plan: &prompt::Plan,
    contract: &[(usize, u16)],
    classes: &[u16],
) -> bool {
    !history.is_empty()
        && history.len().is_multiple_of(BLOCK_TOKENS)
        && history.len() < tokens.len()
        && tokens.starts_with(history)
        && !contract.is_empty()
        && plan.contract(history.len(), classes) == contract
}

#[cfg(test)]
pub(super) fn cuts(tokens: usize, chunk: usize) -> [usize; 2] {
    let last = tokens.saturating_sub(1) / chunk * chunk;
    // Leave 16..31 tokens for a rewritten assistant/tool header, rather than
    // retaining only a fragile near-terminal page. Short cold prompts avoid
    // an extra whole-graph submission solely for caching.
    let tail = tokens.saturating_sub(1 + BLOCK_TOKENS) / BLOCK_TOKENS * BLOCK_TOKENS;
    if tail > last && tokens > 256 {
        [last, tail]
    } else if last == 0 {
        [0, 0]
    } else {
        [last - BLOCK_TOKENS, last]
    }
}

#[cfg(test)]
pub(super) fn rows_until_cut(tokens: usize, offset: usize, chunk: usize) -> usize {
    cuts(tokens, chunk)
        .into_iter()
        .filter(|&n| n > offset)
        .min()
        .unwrap_or(tokens)
        - offset
}

fn replacement<'a>(
    entries: impl Iterator<Item = (&'a [u32], u64)>,
    history: &[u32],
    keep_from: usize,
) -> Option<usize> {
    entries
        .enumerate()
        .filter(|(_, (old, _))| {
            old.is_empty() || old.len() != keep_from || !history.starts_with(old)
        })
        .min_by_key(|(_, (old, touched))| {
            let ancestor = !old.is_empty() && history.starts_with(old);
            (
                if ancestor {
                    0
                } else if old.is_empty() {
                    1
                } else {
                    2
                },
                if ancestor { old.len() as u64 } else { *touched },
            )
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_carried_state_gpu_roundtrip_preserves_other_slots_and_unused_index_rows() {
        let d = MetalDevice::new(Some(512 << 20)).unwrap();
        let context = 64;
        let pages = context / BLOCK_TOKENS;
        let delta = (0..36)
            .map(|_| deltanet::Cache::new(&d, 2).unwrap())
            .collect::<Vec<_>>();
        let qsa = (0..12)
            .map(|_| qsa::Cache::new(&d, 2, pages).unwrap())
            .collect::<Vec<_>>();
        let ple = ple::State::new(&d, 1, 2, context).unwrap();
        let blob = d.alloc(PrefixCache::state_bytes(context)).unwrap();
        let layers = || {
            (0..48).map(|i| {
                if i % 4 == 3 {
                    StateLayer::Qsa(&qsa[i / 4])
                } else {
                    StateLayer::Delta(&delta[i - i / 4])
                }
            })
        };
        let length = 32;
        let mut fields = Vec::new();
        for c in &delta {
            fields.push((&c.state, deltanet::CELLS));
            fields.push((&c.history, 3 * deltanet::CONV));
        }
        for c in &qsa {
            fields.push((&c.pooled, length / 4 * 128));
            fields.push((&c.ring, 512));
        }
        fields.extend([(&ple.history, 2), (&ple.ring, 9 * WIDE)]);
        let word = |field: usize, i: usize| (field as u32 * 65537).wrapping_add(i as u32);
        for (field, &(buffer, _)) in fields.iter().enumerate() {
            let stride = buffer.len() / 8;
            let values = (0..stride)
                .map(|i| word(field, i))
                .chain(std::iter::repeat_n(0xdeadbeef, stride))
                .collect::<Vec<_>>();
            unsafe { buffer.write_u32(&values) };
        }
        transfer_state(
            &d,
            &blob,
            layers(),
            &ple,
            Transfer {
                slot: 0,
                pages,
                length,
                restore: false,
            },
        )
        .unwrap();
        for &(buffer, _) in &fields {
            unsafe { buffer.write_u32(&vec![0xabcd1234; buffer.len() / 8]) };
        }
        transfer_state(
            &d,
            &blob,
            layers(),
            &ple,
            Transfer {
                slot: 1,
                pages,
                length,
                restore: true,
            },
        )
        .unwrap();
        for (field, &(buffer, used)) in fields.iter().enumerate() {
            let words = unsafe { buffer.read_u32(buffer.len() / 4) };
            let stride = words.len() / 2;
            assert!(words[..stride].iter().all(|&v| v == 0xabcd1234));
            for (i, &got) in words[stride..].iter().enumerate() {
                assert_eq!(
                    got,
                    if i < used { word(field, i) } else { 0xdeadbeef },
                    "field={field} offset={i}"
                );
            }
        }
    }

    #[test]
    fn eviction_releases_only_cache_refs_not_an_active_shared_prefix() {
        let d = MetalDevice::new(Some(1 << 20)).unwrap();
        let mut pool = KvPool::with_blocks(4);
        let mut active = BlockTable::default();
        active.ensure(31, &mut pool).unwrap();
        let mut table = BlockTable::default();
        table.share_prefix(active.blocks(), &mut pool);
        let mut prefix = PrefixCache {
            entries: vec![Entry {
                state: d.alloc(4).unwrap(),
                history: vec![1; 32],
                table,
                touched: 1,
                contract: Vec::new(),
            }],
            clock: 1,
            classes: vec![],
        };
        assert_eq!(pool.free_blocks(), 2);
        assert!(prefix.evict(&mut pool));
        assert_eq!(pool.free_blocks(), 2);
        assert!(active.blocks().iter().all(|&b| pool.refcount(b) == 1));
        assert!(!prefix.evict(&mut pool));
        active.clear(&mut pool);
        assert_eq!(pool.free_blocks(), 4);
    }

    #[test]
    fn admission_reclaims_cache_only_pages_without_double_counting_shared_backups() {
        let d = MetalDevice::new(Some(1 << 20)).unwrap();
        let mut pool = KvPool::with_blocks(4);
        let mut active = BlockTable::default();
        active.ensure(47, &mut pool).unwrap();
        let mut entries = Vec::new();
        for blocks in [2, 3] {
            let mut table = BlockTable::default();
            table.share_prefix(&active.blocks()[..blocks], &mut pool);
            entries.push(Entry {
                state: d.alloc(4).unwrap(),
                history: vec![1; blocks * BLOCK_TOKENS],
                table,
                touched: blocks as u64,
                contract: Vec::new(),
            });
        }
        let mut prefix = PrefixCache {
            entries,
            clock: 3,
            classes: vec![],
        };
        assert_eq!(pool.free_blocks(), 1);
        assert_eq!(prefix.reclaimable_blocks(&pool), 0);
        active.clear(&mut pool);
        assert_eq!(pool.free_blocks(), 1, "telemetry still sees retained pages");
        assert_eq!(prefix.reclaimable_blocks(&pool), 3);
        active.share_prefix(&prefix.entries[0].table.blocks()[..1], &mut pool);
        assert_eq!(prefix.reclaimable_blocks(&pool), 2);
        assert!(prefix.evict(&mut pool));
        assert_eq!(prefix.reclaimable_blocks(&pool), 2);
        prefix.clear(&mut pool);
        assert_eq!(pool.free_blocks(), 3);
        assert_eq!(prefix.reclaimable_blocks(&pool), 0);
        active.clear(&mut pool);
        assert_eq!(pool.free_blocks(), 4);
    }

    #[test]
    fn prefix_matches_tokens_and_arithmetic_boundary_not_only_length() {
        let classes = super::super::super::mlx::arithmetic_classes();
        let compatible = |h: &[u32], tokens: &[u32], chunk: usize| {
            super::compatible(
                h,
                tokens,
                &prompt::Plan::grid(tokens.len(), chunk),
                &[(h.len(), classes[chunk])],
                &classes,
            )
        };
        let h = vec![7; 1024];
        assert!(compatible(&h, &[7; 1040], 1024));
        assert!(!compatible(&h, &h, 1024)); // terminal token uses decode arithmetic
        assert!(!compatible(&h, &[8; 1040], 1024));
        assert!(compatible(&h[..1008], &[7; 1040], 1024));
        assert!(compatible(&h[..1008], &[7; 1010], 1024)); // both single-K-part tiled contractions
        assert!(!compatible(&h[..1007], &[7; 1040], 1024));
        assert!(!compatible(&[], &[7; 1040], 1024));
        let mut edited = vec![7; 1040];
        edited[1023] = 9;
        assert!(!compatible(&h, &edited, 1024));
        assert!(compatible(&h[..1008], &edited, 1024));
        assert_eq!(cuts(1024, 1024), [0, 992]);
        assert_eq!(cuts(1025, 1024), [1008, 1024]);
        assert_eq!(rows_until_cut(1025, 1007, 1024), 1);
        assert_eq!(rows_until_cut(1025, 1008, 1024), 16);
        assert_eq!(cuts(1700, 1024), [1024, 1680]);
        assert!(super::compatible(
            &vec![7; 1680],
            &vec![7; 1800],
            &prompt::Plan::grid(1800, 1024),
            &prompt::Plan::grid(1700, 1024).contract(1680, &classes),
            &classes
        ));
        assert!(!super::compatible(
            &vec![7; 1680],
            &vec![7; 2020],
            &prompt::Plan::grid(2020, 1024),
            &prompt::Plan::grid(1700, 1024).contract(1680, &classes),
            &classes
        ));
        assert!(!super::compatible(
            &vec![7; 1680],
            &vec![7; 1800],
            &prompt::Plan::grid(1800, 1024),
            &[],
            &classes
        ));
        for tokens in 1..=4096 {
            let cut = cuts(tokens, 1024);
            assert!(cut[0] <= cut[1] && cut[1] < tokens);
            assert!(cut.iter().all(|c| c.is_multiple_of(BLOCK_TOKENS)));
            for offset in 0..tokens {
                let n = rows_until_cut(tokens, offset, 1024);
                assert!(n > 0 && offset + n <= tokens);
            }
        }
    }

    #[test]
    fn fast_branch_retires_ancestors_not_waiting_conversations() {
        for per_branch in [1, 2] {
            let mut entries = vec![(Vec::new(), 0); 4 * per_branch];
            let mut lengths = [0; 4];
            let mut clock = 0;
            for branch in [0, 1, 2, 3, 0, 0, 0, 2, 0, 3, 1, 1, 0, 2, 3] {
                lengths[branch] += 128;
                let h = vec![branch as u32; lengths[branch]];
                let keep = lengths[branch] - 128 * (per_branch - 1);
                let index =
                    replacement(entries.iter().map(|(v, t)| (v.as_slice(), *t)), &h, keep).unwrap();
                clock += 1;
                entries[index] = (h, clock);
                for (peer, &n) in lengths.iter().enumerate().filter(|(_, n)| **n > 0) {
                    assert!(entries.iter().any(|(h, _)| *h == vec![peer as u32; n]));
                }
            }
        }
    }

    #[test]
    fn advancing_tail_keeps_full_fallback_and_does_not_evict_peers() {
        let mut entries = vec![(Vec::new(), 0); 8];
        let mut last = [[0; 2]; 4];
        let mut clock = 0;
        for (branch, tokens) in [
            (0, 1700),
            (1, 1700),
            (2, 1700),
            (3, 1700),
            (0, 1740),
            (0, 1800),
            (0, 1950),
            (0, 2200),
            (0, 2700),
            (2, 2000),
        ] {
            let cut = cuts(tokens, 1024);
            for length in cut {
                let history = vec![branch as u32; length];
                if entries.iter().any(|(h, _)| *h == history) {
                    continue;
                }
                let index = replacement(
                    entries.iter().map(|(h, t)| (h.as_slice(), *t)),
                    &history,
                    cut[0],
                )
                .unwrap();
                clock += 1;
                entries[index] = (history, clock);
            }
            last[branch] = cut;
            for (peer, lengths) in last.iter().enumerate() {
                for &length in lengths.iter().filter(|&&n| n > 0) {
                    assert!(
                        entries.iter().any(|(h, _)| *h == vec![peer as u32; length]),
                        "lost peer={peer} prefix={length}"
                    );
                }
            }
        }
    }
}
