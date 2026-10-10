//! The hybrid restore path on the fake transport: a checkpoint's blob and
//! the KV it resumes over travel to T1 together, T1 lets go of whole chains
//! from the tail, a restored path survives until its checkpoint attaches,
//! and the ledger says why a held prefix was or was not used.

use super::super::restore_flow::{AuxPlan, FlowStatus, RestoreFlow};
use super::tests::{armed_radix, cached_chain, tier};
use super::*;
use crate::kv_pool::BlockTable;

/// Insert `tokens` (whole blocks + one tail token) as a tree-held chain.
fn chain(radix: &mut PagedRadix, pool: &mut KvPool, tokens: &[u32]) {
    let mut t = BlockTable::new();
    t.ensure(tokens.len() - 2, pool).unwrap();
    radix.insert(tokens, t.blocks(), pool);
    t.clear(pool);
}

/// Every live Qwen3.8 restore failed its attach: the restored blocks carry
/// no checkpoint until the blob round lands, so to the radix they are dead
/// KV - and the blob round's own page reservation (or any other request's
/// pressure) trimmed them first. The published path stays pinned until the
/// checkpoint attaches.
#[test]
fn a_restored_hybrid_path_survives_pressure_until_its_checkpoint_attaches() {
    let mut t = tier(256 << 20);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_paged(4, 6);
    let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
    radix
        .attach_state_with_pool(&tokens, 8 * BLOCK_TOKENS, &mut pool)
        .expect("checkpoint");
    t.press(&mut radix, &mut pool, 64, None, &mut || None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    assert_eq!(pool.free_blocks(), 64, "chain and checkpoint in T1");
    let hit = t.probe(&tokens, 0).expect("the KV is held");
    let aux = t.probe_aux(&tokens, hit.end_block).expect("and the blob");
    let plan = AuxPlan {
        hit: aux,
        state_base: 0,
        state_stride: 0,
    };
    let flow = RestoreFlow::begin(&mut t, &mut pool, &tokens, &hit, Some(plan), 1000.0, None)
        .expect("flow");
    t.park_flow(0, flow);
    // round one lands and publishes...
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    assert_eq!(radix.chain_depth(&tokens), 8);
    // ...and before the blob round, pressure trims dead KV
    radix.evict_dead_leaves(&mut pool, 64, 0);
    assert_eq!(radix.chain_depth(&tokens), 8, "the restored path is pinned");
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    assert_eq!(t.flow_status(0, &tokens), FlowStatus::Done { ok: true });
    assert!(radix.match_full(&tokens).ckpt.is_some(), "attached");
    // the pin went with the flow: everything frees again
    t.press(&mut radix, &mut pool, 64, None, &mut || None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    assert_eq!(pool.free_blocks(), 64, "no pin outlived its flow");
    t.catalog.check_invariants();
}

/// The blob round ran on its own guess (16 GB/s, ~20 ms here) instead of
/// the flow's priced budget: once checksums left the tick, every Qwen3.8
/// blob round outlived it and was abandoned - and abandoned flows counted
/// nowhere. The round inherits the flow's deadline; a give-up is counted.
#[test]
fn the_blob_round_inherits_the_flows_budget() {
    let mut t = tier(256 << 20);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_paged(4, 6);
    let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
    radix
        .attach_state_with_pool(&tokens, 8 * BLOCK_TOKENS, &mut pool)
        .expect("checkpoint");
    t.press(&mut radix, &mut pool, 64, None, &mut || None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    let start = |t: &mut PoolTier<_>, pool: &mut KvPool, est_us: f64| {
        let hit = t.probe(&tokens, 0).expect("held");
        let aux = t.probe_aux(&tokens, hit.end_block).expect("blob");
        let plan = AuxPlan {
            hit: aux,
            state_base: 0,
            state_stride: 0,
        };
        let flow =
            RestoreFlow::begin(t, pool, &tokens, &hit, Some(plan), est_us, None).expect("flow");
        t.park_flow(0, flow);
    };
    // a 1 s flow: the blocks land, the blob takes longer than any guess
    start(&mut t, &mut pool, 1e6);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    std::thread::sleep(std::time::Duration::from_millis(40));
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    assert_eq!(
        t.flow_status(0, &tokens),
        FlowStatus::Loading,
        "still in budget"
    );
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    assert_eq!(t.flow_status(0, &tokens), FlowStatus::Done { ok: true });
    // a flow past its whole budget gives up - and says so
    t.press(&mut radix, &mut pool, 64, None, &mut || None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    start(&mut t, &mut pool, 1.0);
    std::thread::sleep(std::time::Duration::from_millis(40));
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    assert_eq!(t.flow_status(0, &tokens), FlowStatus::Done { ok: false });
    assert_eq!(t.dec.abandoned, 1, "a give-up is counted");
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    t.pump_flows_with_pool(&mut radix, &mut pool, &mut || None);
    t.catalog.check_invariants();
}

/// The slack mirror shipped a live checkpoint's blob alone; the radix then
/// dropped the checkpoint and trimmed its path as dead KV on its own, and
/// the blob sat in T1 with nothing under it. The blob goes with its KV.
#[test]
fn a_mirrored_checkpoint_carries_its_kv_into_t1() {
    let mut t = tier(256 << 20);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_paged(4, 3);
    let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
    radix
        .attach_state_with_pool(&tokens, 8 * BLOCK_TOKENS, &mut pool)
        .expect("checkpoint");
    // no run budget: only what the checkpoint needs goes
    t.mirror_slack(&radix, &mut pool, None, 0, None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    // the blob's shard is not a run: a customer's tier counted every
    // mirrored shard into its run inventory (135,280 "runs" in 32 GiB)
    assert_eq!(t.stats().0, 2, "two KV runs; the shard is the blob's");
    // a live context takes the pool back without the tier: the
    // checkpoint's pages, then its path as dead KV
    assert!(radix.make_room(&mut pool, 64, 0));
    assert_eq!(radix.chain_depth(&tokens), 0);
    let hit = t.probe(&tokens, 0).expect("the KV went with the blob");
    assert_eq!(hit.end_block, 8);
    assert_eq!(t.probe_aux(&tokens, 8).map(|a| a.end_block), Some(8));
    t.catalog.check_invariants();
}

/// A blob already in T1 when its checkpoint is pressed out: the KV under it
/// still has to go (the path trims as dead the moment the checkpoint does).
#[test]
fn pressing_out_a_mirrored_checkpoint_still_stores_its_kv() {
    let mut t = tier(256 << 20);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_paged(4, 3);
    let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
    radix
        .attach_state_with_pool(&tokens, 8 * BLOCK_TOKENS, &mut pool)
        .expect("checkpoint");
    // the blob alone, as a mirror that lost the run half would have left it
    let (_, key, idx, _) = radix.state_attachments()[0];
    let span = AuxSpan::Pages(radix.state_pages(idx).to_vec());
    let key = key.expect("keyed");
    assert!(t.mirror_aux_span(key, 8, span, Some(&mut pool), None, Vec::new()));
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    assert!(t.probe(&tokens, 0).is_none(), "no KV yet");
    t.press(&mut radix, &mut pool, 64, None, &mut || None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    assert_eq!(t.probe(&tokens, 0).map(|h| h.end_block), Some(8));
    assert!(t.probe_aux(&tokens, 8).is_some());
    assert_eq!(pool.free_blocks(), 64);
    t.catalog.check_invariants();
}

/// T1 full: plain LRU took the chain head first (the oldest stamp) and
/// stranded every blob above it. Runs a blob owns leave only with their
/// blobs, the tail first; a head two chains share stays while either does.
#[test]
fn t1_lets_go_of_whole_chains_from_the_tail() {
    // runs 16 MiB, blobs 20 MiB: A (2 runs + blob) + B (1 more run + blob)
    // = 88 MiB of 104; C's 52 MiB needs exactly A's own 36 back
    let mut t = tier(104 << 20);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_capacity(4);
    let a: Vec<u32> = (0..8 * BLOCK_TOKENS as u32).chain([9]).collect();
    // B shares A's first run, then diverges
    let b: Vec<u32> = (0..4 * BLOCK_TOKENS as u32)
        .chain(50_000..50_000 + 4 * BLOCK_TOKENS as u32)
        .chain([9])
        .collect();
    let c: Vec<u32> = (90_000..90_000 + 8 * BLOCK_TOKENS as u32)
        .chain([9])
        .collect();
    fn store(
        t: &mut PoolTier<super::super::transport::FakeTransport>,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        tok: &[u32],
    ) {
        chain(radix, pool, tok);
        radix
            .attach_state(tok, 8 * BLOCK_TOKENS)
            .expect("checkpoint");
        let (_e, taken) = t.pressure_demote(radix, pool, 64, None);
        for x in taken {
            t.demote_aux(radix, x, 4096, 20 << 20, None);
        }
        t.transport.deliver_all();
        t.pump_completions(radix, pool);
    }
    store(&mut t, &mut radix, &mut pool, &a);
    store(&mut t, &mut radix, &mut pool, &b);
    assert!(t.probe_aux(&a, 8).is_some() && t.probe_aux(&b, 8).is_some());
    // B was just used: A is the stalest chain
    assert!(t.probe(&b, 0).is_some());
    store(&mut t, &mut radix, &mut pool, &c);
    assert!(t.probe_aux(&a, 8).is_none(), "A's blob went first");
    assert_eq!(
        t.probe(&a, 0).map(|h| h.end_block),
        Some(4),
        "then A's own tail - the shared head stays"
    );
    assert_eq!(t.probe(&b, 0).map(|h| h.end_block), Some(8), "B whole");
    assert!(t.probe_aux(&b, 8).is_some());
    assert_eq!(t.probe(&c, 0).map(|h| h.end_block), Some(8), "C whole");
    assert!(t.probe_aux(&c, 8).is_some());
    t.catalog.check_invariants();
}

/// The panel read "found 60, reused 0, rebuilt 0, delivered 0": hybrid hits
/// without a resumable boundary vanished, and the hybrid election bypassed
/// the counters. A held prefix with no state is a miss with its own reason;
/// an election is counted either way.
#[test]
fn a_hybrid_hit_without_its_state_is_a_no_state_miss() {
    let mut t = tier(256 << 20);
    t.cost.set_force_restore(true);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_capacity(2);
    let bare = cached_chain(&mut radix, &mut pool, 1, 8);
    t.pressure_demote(&mut radix, &mut pool, 64, None);
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    let hit = t.probe(&bare, 0).expect("the KV is held");
    assert!(t.elect_hybrid(&hit, None).is_none());
    assert_eq!((t.dec.lookups, t.dec.hits, t.dec.miss_no_state), (1, 0, 1));
    let whole = cached_chain(&mut radix, &mut pool, 2, 8);
    radix
        .attach_state(&whole, 8 * BLOCK_TOKENS)
        .expect("checkpoint");
    let (_e, taken) = t.pressure_demote(&mut radix, &mut pool, 64, None);
    for x in taken {
        t.demote_aux(&mut radix, x, 4096, 20 << 20, None);
    }
    t.transport.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    let hit = t.probe(&whole, 0).expect("held");
    let aux = t.probe_aux(&whole, hit.end_block);
    let (cut, _est) = t.elect_hybrid(&hit, aux.as_ref()).expect("restore elected");
    assert_eq!(cut.end_block, 8);
    assert_eq!((t.dec.hits, t.dec.elected_restore), (1, 1));
    t.catalog.check_invariants();
}

// -- dominance and the disk tier ---------------------------------------------

/// T1 full: a blob with a deeper blob over the same chain is the cheapest
/// thing to lose (only a prompt leaving the chain between the two needs
/// it) and goes before any chain's deepest state - plain LRU took the
/// stalest chain's only blob here.
#[test]
fn t1_drops_a_dominated_blob_before_any_chains_deepest() {
    // runs 16 MiB, blobs 20 MiB: B (2 runs + blob) 52, A (2 runs + blobs at
    // 4 and 8) 72 = 124 of 160; C's 52 needs one blob back
    let mut t = tier(160 << 20);
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix(&t);
    radix.set_state_capacity(4);
    let b: Vec<u32> = (70_000..70_000 + 8 * BLOCK_TOKENS as u32)
        .chain([9])
        .collect();
    let a: Vec<u32> = (0..8 * BLOCK_TOKENS as u32).chain([9]).collect();
    let c: Vec<u32> = (90_000..90_000 + 8 * BLOCK_TOKENS as u32)
        .chain([9])
        .collect();
    fn store(
        t: &mut PoolTier<super::super::transport::FakeTransport>,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        tok: &[u32],
        cuts: &[usize],
    ) {
        chain(radix, pool, tok);
        for &d in cuts {
            radix
                .attach_state(tok, d * BLOCK_TOKENS)
                .expect("checkpoint");
        }
        let (_e, taken) = t.pressure_demote(radix, pool, 64, None);
        for x in taken {
            t.demote_aux(radix, x, 4096, 20 << 20, None);
        }
        t.transport.deliver_all();
        t.pump_completions(radix, pool);
    }
    store(&mut t, &mut radix, &mut pool, &b, &[8]);
    store(&mut t, &mut radix, &mut pool, &a, &[4, 8]);
    assert_eq!(t.probe_aux(&a, 4).map(|h| h.end_block), Some(4));
    store(&mut t, &mut radix, &mut pool, &c, &[8]);
    assert!(t.probe_aux(&a, 4).is_none(), "the dominated cut went");
    assert_eq!(
        t.probe_aux(&a, 8).map(|h| h.end_block),
        Some(8),
        "A's deepest stays"
    );
    assert!(
        t.probe_aux(&b, 8).is_some(),
        "the stalest chain keeps its only state"
    );
    assert!(t.probe_aux(&c, 8).is_some());
    t.catalog.check_invariants();
}

/// The fake with a pretend disk: every store that lands is "written
/// through" (its key -> the same at-rest loc, which the fake keeps), and a
/// disk-sourced load read-fills a T1 copy through the promotion outbox.
#[derive(Default)]
struct DiskFake {
    /// Where the cost model's calibration lives (None: nowhere).
    home: Option<std::path::PathBuf>,
    inner: super::super::transport::FakeTransport,
    durable: HashMap<[u8; 32], (Loc, u64, [u8; 32])>,
    stores: HashMap<OpId, LogicalKey>,
    disk_loads: HashMap<OpId, (LogicalKey, Loc)>,
    fills: Vec<([u8; 32], Loc, [u8; 32], u64)>,
}

impl TierTransport for DiskFake {
    fn caps(&self) -> super::super::transport::TransportCaps {
        self.inner.caps()
    }
    fn submit(&mut self, job: super::super::transport::IoJob) -> Result<(), SubmitError> {
        use super::super::transport::IoJobKind;
        match &job.kind {
            IoJobKind::Store { .. } => {
                self.stores.insert(job.op, job.key);
            }
            IoJobKind::Load { loc, .. } if job.tier == Tier::Nvme => {
                self.disk_loads.insert(job.op, (job.key, *loc));
            }
            IoJobKind::Load { .. } => {}
        }
        self.inner.submit(job)
    }
    fn cancel(&mut self, op: OpId) {
        self.inner.cancel(op)
    }
    fn poll(&mut self) -> Vec<super::super::transport::IoCompletion> {
        use super::super::transport::IoOutcome;
        let out = self.inner.poll();
        for c in &out {
            match &c.outcome {
                IoOutcome::StoreDone {
                    loc,
                    bytes,
                    checksum,
                } => {
                    if let Some(k) = self.stores.remove(&c.op) {
                        self.durable.insert(k.0, (*loc, *bytes, checksum.0));
                    }
                }
                IoOutcome::LoadDone {
                    bytes, checksum, ..
                } => {
                    if let Some((k, loc)) = self.disk_loads.remove(&c.op) {
                        self.fills.push((k.0, loc, checksum.0, *bytes));
                    }
                }
                _ => {}
            }
        }
        out
    }
}

impl XferSink for DiskFake {
    fn t2_entry(&self, key: &[u8; 32]) -> Option<(Loc, u64, [u8; 32])> {
        self.durable.get(key).copied()
    }
    fn take_t2_promotions(&mut self) -> Vec<([u8; 32], Loc, [u8; 32], u64)> {
        std::mem::take(&mut self.fills)
    }
    fn expect_store(&mut self, _key: LogicalKey, _spec: XferSpec) -> Result<(), SubmitError> {
        Ok(())
    }
    fn expect_load(&mut self, _key: LogicalKey, _spec: XferSpec) -> Result<(), SubmitError> {
        Ok(())
    }
    // the bytes stay at their loc: the disk copy keeps reading them
    fn free_extent(&mut self, _loc: Loc) {}
    fn calibration_home(&self) -> Option<(std::path::PathBuf, String)> {
        Some((self.home.clone()?, "fake GPU sm_00 1sm".into()))
    }
}

fn disk_tier(ram: u64, disk: DiskFake) -> PoolTier<DiskFake> {
    use super::tests::{ns, planes};
    PoolTier::with_capacities(&ns(), planes(), ram, 1 << 30, disk).unwrap()
}

/// Paged checkpoint of 3 pages (12 MiB, one shard) on an 8-block chain:
/// press it to T1 (runs + blob), everything written through.
fn pressed_chain(
    t: &mut PoolTier<DiskFake>,
    radix: &mut PagedRadix,
    pool: &mut KvPool,
    seed: u32,
) -> Vec<u32> {
    let tokens = cached_chain(radix, pool, seed, 8);
    radix
        .attach_state_with_pool(&tokens, 8 * BLOCK_TOKENS, pool)
        .expect("checkpoint");
    t.press(radix, pool, 64, None, &mut || None);
    t.transport.inner.deliver_all();
    t.pump_completions(radix, pool);
    tokens
}

/// Run a parked restore of `tokens` to completion; true when the
/// checkpoint attached.
fn restore(
    t: &mut PoolTier<DiskFake>,
    radix: &mut PagedRadix,
    pool: &mut KvPool,
    tokens: &[u32],
) -> bool {
    let hit = t.probe(tokens, 0).expect("the KV is held");
    let aux = t.probe_aux(tokens, hit.end_block).expect("and the blob");
    let plan = AuxPlan {
        hit: aux,
        state_base: 0,
        state_stride: 0,
    };
    let flow = RestoreFlow::begin(t, pool, tokens, &hit, Some(plan), 1e6, None).expect("flow");
    t.park_flow(0, flow);
    for _ in 0..4 {
        t.transport.inner.deliver_all();
        t.pump_completions(radix, pool);
        t.pump_flows_with_pool(radix, pool, &mut || None);
    }
    t.flow_status(0, tokens) == FlowStatus::Done { ok: true }
        && radix.match_full(tokens).ckpt.is_some()
}

/// Evicted from T1, a written-through blob used to vanish from the tier
/// (its inventory entry went, nothing pointed at the disk copy) while its
/// runs stayed readable from disk - so "add nvme_gb" bought a hybrid
/// nothing. The blob falls back to its disk copy like a run.
#[test]
fn a_blob_evicted_from_ram_restores_from_disk() {
    // T1 holds one chain (2 x 16 MiB runs + a 12 MiB blob = 44 MiB)
    let mut t = disk_tier(64 << 20, DiskFake::default());
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix_of(&t);
    radix.set_state_paged(4, 3);
    t.declare_blob_pages(3);
    let a = pressed_chain(&mut t, &mut radix, &mut pool, 1);
    let _b = pressed_chain(&mut t, &mut radix, &mut pool, 2);
    let aux = t.probe_aux(&a, 8).expect("A's blob, now from disk");
    assert_eq!(aux.nvme_bytes, aux.bytes, "every shard on T2");
    assert!(t.dec.promoted_to_disk >= 1);
    assert!(
        restore(&mut t, &mut radix, &mut pool, &a),
        "restored off disk"
    );
    t.catalog.check_invariants();
}

/// A restart empties every in-memory index; the disk still holds the
/// chain. With the geometry declared, a probe finds the blob by its keys,
/// and the restore seats it back in T1 whole - shards in the blob
/// inventory, never in the run count.
#[test]
fn after_a_restart_a_blob_on_disk_restores_and_seats_back_whole() {
    let mut t = disk_tier(256 << 20, DiskFake::default());
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix_of(&t);
    radix.set_state_paged(4, 3);
    let a = pressed_chain(&mut t, &mut radix, &mut pool, 1);
    // restart: the disk survives, nothing else does
    let disk = std::mem::take(&mut t.transport);
    let entries: Vec<_> = disk.durable.iter().map(|(k, v)| (*k, *v)).collect();
    let mut t = disk_tier(256 << 20, disk);
    for (k, (loc, len, sum)) in entries {
        t.catalog.preload_ready(
            LogicalKey(k),
            Tier::Nvme,
            loc,
            super::super::digest::Checksum(sum),
            len,
        );
    }
    let mut pool = KvPool::with_blocks(64);
    let mut radix = armed_radix_of(&t);
    radix.set_state_paged(4, 3);
    assert!(
        t.probe_aux(&a, 8).is_none(),
        "undeclared: nothing to look for"
    );
    t.declare_blob_pages(3);
    assert!(
        restore(&mut t, &mut radix, &mut pool, &a),
        "restored off disk"
    );
    t.pump_completions(&mut radix, &mut pool); // settles the read-fill
    let blob = t.probe_aux(&a, 8).expect("blob").key;
    assert!(t.aux_meta.contains_key(&blob), "seated in T1 as a blob");
    assert_eq!(
        t.stats().0,
        2,
        "two runs read-filled; the shard is the blob's"
    );
    t.catalog.check_invariants();
}

fn armed_radix_of<T: XferSink>(t: &PoolTier<T>) -> PagedRadix {
    let mut r = PagedRadix::new();
    r.set_tier_root(t.tier_root());
    r
}

/// A restart's first hits all come off disk, before this process has
/// measured a prefill: the cost model starts from the rates the last run
/// measured on this device, saved beside the disk store.
#[test]
fn a_restart_prices_its_first_disk_hit_by_the_last_runs_calibration() {
    let dir = std::env::temp_dir().join(format!("pd-cal-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let disk = DiskFake {
        home: Some(dir.clone()),
        ..Default::default()
    };
    let mut t = disk_tier(64 << 20, disk);
    let mut pool = KvPool::with_blocks(8);
    let mut radix = armed_radix_of(&t);
    t.cost.observe_prefill(10_000, 8_000_000.0); // this run: 1.25 tok/ms
    t.pump_completions(&mut radix, &mut pool); // saved at the first chance
    let measured = t.cost.calibration().expect("measured");
    let disk = std::mem::take(&mut t.transport);
    let restarted = disk_tier(64 << 20, disk);
    let hit = HitShape {
        restore_bytes: u64::MAX >> 8,
        restore_tokens: 10_000,
        queued_bytes: 0,
        nvme_bytes: 0,
    };
    let est = match restarted.cost.elect(hit) {
        Election::Recompute { est_us, .. } => est_us,
        Election::Restore { .. } => unreachable!(),
    };
    assert!(
        (est - 10_000.0 / measured.prefill_tpus).abs() < 1.0,
        "priced by the saved rate, not the generic seed: {est}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
