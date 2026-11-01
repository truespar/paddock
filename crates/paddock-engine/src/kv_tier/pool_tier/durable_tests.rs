//! The durability plan on the fake disk: tips only, root-first, never past
//! a hole, a superseded tip stops at its runs, idle before written.

use std::time::{Duration, Instant};

use super::durable::{EVICT_WRITE_BURST, IDLE};
use super::hybrid_tests::{DiskFake, armed_radix_of, chain, disk_tier};
use super::tests::cached_chain;
use super::*;

/// Chain `tok` with checkpoints at the `cuts` (in blocks) and press it all
/// into T1: runs root-first, each checkpoint a 20 MiB flat blob (2 shards).
fn store(
    t: &mut PoolTier<DiskFake>,
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
    t.transport.inner.deliver_all();
    t.pump_completions(radix, pool);
}

fn tokens(seed: u32, blocks: usize) -> Vec<u32> {
    (0..(blocks * BLOCK_TOKENS) as u32)
        .map(|i| seed * 100_000 + i)
        .chain([9])
        .collect()
}

fn shard(blob: LogicalKey, i: u32) -> LogicalKey {
    blob.child_bytes("aux", &i.to_le_bytes())
}

fn setup() -> (PoolTier<DiskFake>, PagedRadix, KvPool) {
    let t = disk_tier(512 << 20, DiskFake::default());
    let mut radix = armed_radix_of(&t);
    radix.set_state_capacity(8);
    (t, radix, KvPool::with_blocks(64))
}

/// A chain with state at blocks 4 and 8: the deeper blob supersedes the
/// shallower one, so the disk gets both runs (root-first) and the blob at
/// 8 - never the one at 4.
#[test]
fn only_a_chains_tip_reaches_the_disk_root_first() {
    let (mut t, mut radix, mut pool) = setup();
    let a = tokens(1, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[4, 8]);
    let runs = t.token_run_keys(&a, 8);
    let (b4, b8) = (runs[0], runs[1]); // a blob's key is its last run's
    assert!(t.aux_meta.contains_key(&b4) && t.aux_meta.contains_key(&b8));
    t.persist_all();
    let log = &t.transport.log;
    assert_eq!(
        log,
        &[runs[0], runs[1], shard(b8, 0), shard(b8, 1)],
        "runs root-first, then the tip's state"
    );
    assert!(!t.transport.durable.contains_key(&shard(b4, 0).0));
}

/// A user turn's anchor keeps its disk copy beside the deeper reply path -
/// the next user message (or the same request sent again after a cut-off)
/// resumes there - until a later turn's anchor takes over on that chain.
#[test]
fn a_user_turn_anchor_is_kept_beside_its_reply_path() {
    for later_anchor in [false, true] {
        let (mut t, mut radix, mut pool) = setup();
        let a = tokens(5, 8);
        chain(&mut radix, &mut pool, &a);
        for d in [4, 8] {
            radix
                .attach_state(&a, d * BLOCK_TOKENS)
                .expect("checkpoint");
        }
        radix.mark_anchor(&a, 4 * BLOCK_TOKENS + 5);
        if later_anchor {
            radix.mark_anchor(&a, 8 * BLOCK_TOKENS + 5);
        }
        let (_e, taken) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        for x in taken {
            t.demote_aux(&mut radix, x, 4096, 20 << 20, None);
        }
        t.transport.inner.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let runs = t.token_run_keys(&a, 8);
        t.persist_all();
        let disk = &t.transport.durable;
        assert!(
            disk.contains_key(&shard(runs[1], 0).0),
            "the reply path's tip"
        );
        assert_eq!(
            disk.contains_key(&shard(runs[0], 0).0),
            !later_anchor,
            "the anchor, unless a deeper one took over"
        );
    }
}

/// State on its way into T1 - a demote's blob whose stores have not landed -
/// is found as landing (its stores started ahead of loads), so a consult can
/// wait for it instead of restoring the shallower state already there.
#[test]
fn state_still_being_stored_is_found_landing() {
    let (mut t, mut radix, mut pool) = setup();
    let a = tokens(6, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[4]);
    // the next turn's deeper checkpoint, demoted but not stored yet
    chain(&mut radix, &mut pool, &a);
    radix
        .attach_state(&a, 8 * BLOCK_TOKENS)
        .expect("checkpoint");
    let (_e, taken) = t.pressure_demote(&mut radix, &mut pool, 64, None);
    for x in taken {
        t.demote_aux(&mut radix, x, 4096, 20 << 20, None);
    }
    let hit = t.probe(&a, 0).expect("the runs are in T1");
    assert_eq!(hit.end_block, 8);
    assert_eq!(
        t.probe_aux(&a, 8).map(|x| x.end_block),
        Some(4),
        "8 not landed"
    );
    assert_eq!(
        t.blob_landing(&a, 4, 8),
        Some(4 * BLOCK_TOKENS),
        "8 is landing"
    );
    assert!(
        !t.transport.expedited.is_empty(),
        "its stores asked to go now"
    );
    assert_eq!(t.blob_landing(&a, 8, 8), None, "nothing deeper");
    t.transport.inner.deliver_all();
    t.pump_completions(&mut radix, &mut pool);
    assert_eq!(t.probe_aux(&a, 8).map(|x| x.end_block), Some(8), "landed");
    assert_eq!(t.blob_landing(&a, 8, 8), None);
}

/// Two sessions on one shared prompt: the shared state is where every new
/// session resumes, so it is kept beside both tips.
#[test]
fn state_where_chains_branch_is_kept() {
    let (mut t, mut radix, mut pool) = setup();
    let shared = tokens(2, 4);
    let a: Vec<u32> = shared[..4 * BLOCK_TOKENS]
        .iter()
        .copied()
        .chain(tokens(3, 4)[..4 * BLOCK_TOKENS].iter().copied())
        .chain([9])
        .collect();
    let b: Vec<u32> = shared[..4 * BLOCK_TOKENS]
        .iter()
        .copied()
        .chain(tokens(4, 4)[..4 * BLOCK_TOKENS].iter().copied())
        .chain([9])
        .collect();
    store(&mut t, &mut radix, &mut pool, &a, &[4, 8]);
    store(&mut t, &mut radix, &mut pool, &b, &[8]);
    let base = t.token_run_keys(&a, 8)[0];
    assert!(t.aux_meta.contains_key(&base), "the shared state is in T1");
    t.persist_all();
    let disk = &t.transport.durable;
    for tip in [t.token_run_keys(&a, 8)[1], t.token_run_keys(&b, 8)[1]] {
        assert!(disk.contains_key(&shard(tip, 0).0), "each session's tip");
    }
    assert!(disk.contains_key(&shard(base, 0).0), "and the branch point");
}

/// KV-only: a leaf's chain goes root-first - and not at all once a link
/// left T1 without a disk copy (everything past it is unreachable).
#[test]
fn a_chain_with_a_hole_is_not_written() {
    let (mut t, mut radix, mut pool) = setup();
    let a = tokens(5, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[]);
    let runs = t.token_run_keys(&a, 8);
    assert!(t.evict_run_t1(runs[0]), "the head leaves T1, no disk copy");
    t.persist_all();
    assert!(t.transport.log.is_empty(), "nothing behind a hole");

    let (mut t, mut radix, mut pool) = setup();
    store(&mut t, &mut radix, &mut pool, &a, &[]);
    t.persist_all();
    assert_eq!(t.transport.log, runs, "whole chain, head first");
}

/// The tip the plan started on is superseded before its state went out
/// (the session's next turn): its runs stay - they are the new tip's
/// prefix - and its state is never written.
#[test]
fn a_tip_superseded_mid_plan_stops_after_its_runs() {
    let (mut t, mut radix, mut pool) = setup();
    let a = tokens(6, 8);
    let runs = t.token_run_keys(&a, 8);
    store(
        &mut t,
        &mut radix,
        &mut pool,
        &a[..4 * BLOCK_TOKENS + 1],
        &[4],
    );
    t.transport.writes_left = Some(1);
    t.persist_all();
    assert_eq!(
        t.transport.log,
        [runs[0]],
        "the run, then the disk declined"
    );
    // the next turn: the same chain grows a deeper checkpoint
    store(&mut t, &mut radix, &mut pool, &a, &[8]);
    t.transport.writes_left = None;
    t.persist_all();
    let disk = &t.transport.durable;
    assert!(
        !disk.contains_key(&shard(runs[0], 0).0),
        "stale state never written"
    );
    assert!(disk.contains_key(&runs[1].0) && disk.contains_key(&shard(runs[1], 0).0));
    assert_eq!(
        t.transport.log.iter().filter(|k| **k == runs[0]).count(),
        1,
        "the shared run written once"
    );
}

/// A tip is written once it has been idle - not while its session is
/// still going (the next turn would supersede it), and not if it was
/// touched since.
#[test]
fn a_tip_is_written_once_idle() {
    let (mut t, mut radix, mut pool) = setup();
    let a = tokens(7, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[8]);
    let t0 = Instant::now();
    let pass = |t: &mut PoolTier<DiskFake>, at: Instant| {
        let mut n = 0;
        for i in 0..8 {
            let at = at + Duration::from_millis(i);
            n += t.persist_pass(at, false, at);
        }
        n
    };
    assert_eq!(pass(&mut t, t0), 0, "fresh: not yet");
    // touched after the first mark: still not idle at mark + IDLE
    t.tick();
    let tip = t.token_run_keys(&a, 8)[1];
    let now = t.lru;
    t.aux_meta.get_mut(&tip).expect("tip").last_used = now;
    assert_eq!(pass(&mut t, t0 + IDLE), 0, "touched since the mark");
    assert!(
        pass(&mut t, t0 + IDLE * 3) >= 4,
        "idle: both runs and the state"
    );
    assert!(t.transport.durable.contains_key(&shard(tip, 1).0));
}

// -- RAM eviction prefers what the disk also holds ---------------------------

/// Give `keys` (in T1) a disk copy, as the plan would.
fn mark_on_disk(t: &mut PoolTier<DiskFake>, keys: &[LogicalKey]) {
    for k in keys {
        let loc = t.catalog.ready_loc(k, Tier::Ram).expect("in T1");
        let bytes = t.catalog.ready_bytes(k, Tier::Ram).expect("bytes");
        let sum = t.catalog.ready_checksum(k, Tier::Ram).expect("checksum");
        assert_eq!(t.transport.persist(k, loc, bytes, sum), Persist::Written);
    }
}

/// Free one more run's worth of T1.
fn evict_one(t: &mut PoolTier<DiskFake>) {
    let need = t.catalog.ledger(Tier::Ram).free + 1;
    assert!(t.evict_t1(need, &[]));
}

/// KV-only: a chain the disk holds goes before an older one only RAM
/// holds - losing it costs disk speed, not the content - and stays
/// restorable from disk.
#[test]
fn ram_eviction_takes_what_the_disk_holds_first() {
    let (mut t, mut radix, mut pool) = setup();
    let (a, b) = (tokens(8, 8), tokens(9, 8));
    store(&mut t, &mut radix, &mut pool, &a, &[]);
    store(&mut t, &mut radix, &mut pool, &b, &[]);
    let (ra, rb) = (t.token_run_keys(&a, 8), t.token_run_keys(&b, 8));
    mark_on_disk(&mut t, &rb);
    evict_one(&mut t);
    for k in &ra {
        assert!(
            t.catalog.ready_bytes(k, Tier::Ram).is_some(),
            "A (older, RAM-only) stays"
        );
    }
    let gone: Vec<_> = rb
        .iter()
        .filter(|k| t.catalog.ready_bytes(k, Tier::Ram).is_none())
        .collect();
    assert_eq!(gone.len(), 1, "one of B's runs went");
    assert!(
        t.catalog.ready_bytes(gone[0], Tier::Nvme).is_some(),
        "readable from disk"
    );
    assert_eq!(
        t.probe(&b, 0).map(|h| h.end_block),
        Some(8),
        "B still restores whole"
    );
}

/// A chain with no disk copy leaves RAM from its tail: plain LRU took its
/// head (the oldest stamp - chains store root-first) and left the rest
/// unreachable.
#[test]
fn an_unwritten_chain_leaves_ram_from_its_tail() {
    let (mut t, mut radix, mut pool) = setup();
    t.transport.writes_left = Some(0); // the order alone, no write-back
    let a = tokens(10, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[]);
    let runs = t.token_run_keys(&a, 8);
    evict_one(&mut t);
    assert!(
        t.catalog.ready_bytes(&runs[0], Tier::Ram).is_some(),
        "the head stays"
    );
    assert!(
        t.catalog.ready_bytes(&runs[1], Tier::Ram).is_none(),
        "the tail went"
    );
    assert_eq!(
        t.probe(&a, 0).map(|h| h.end_block),
        Some(4),
        "the head still serves"
    );
}

/// Hybrid: T1 full, a blob the disk holds whole (and its runs) goes before
/// an older blob nothing else holds.
#[test]
fn a_blob_on_disk_leaves_ram_before_one_that_is_not() {
    // two 8-block chains with a blob at 8 (2 x 16 MiB runs + 20 MiB) fill
    // 104 of 120 MiB; a third needs 36 MiB more
    let mut t = disk_tier(120 << 20, DiskFake::default());
    let mut radix = armed_radix_of(&t);
    radix.set_state_capacity(8);
    let mut pool = KvPool::with_blocks(64);
    let (a, b, c) = (tokens(11, 8), tokens(12, 8), tokens(13, 8));
    store(&mut t, &mut radix, &mut pool, &a, &[8]);
    store(&mut t, &mut radix, &mut pool, &b, &[8]);
    let rb = t.token_run_keys(&b, 8);
    let blob_b = rb[1];
    mark_on_disk(&mut t, &[rb[0], rb[1], shard(blob_b, 0), shard(blob_b, 1)]);
    store(&mut t, &mut radix, &mut pool, &c, &[8]);
    let blob_a = t.token_run_keys(&a, 8)[1];
    assert!(
        t.aux_meta.contains_key(&blob_a),
        "A's state, RAM-only, stays"
    );
    assert!(
        !t.aux_meta.contains_key(&blob_b),
        "B's state, on disk, went first"
    );
    assert!(t.aux_meta.contains_key(&t.token_run_keys(&c, 8)[1]));
    t.catalog.check_invariants();
}

// -- write-back at eviction ---------------------------------------------------

/// T1 full: the stalest live tip goes - but its last copy reaches the disk
/// first, so the session restores from there instead of recomputing.
#[test]
fn eviction_writes_a_tip_that_exists_nowhere_else() {
    let mut t = disk_tier(120 << 20, DiskFake::default());
    let mut radix = armed_radix_of(&t);
    radix.set_state_capacity(8);
    let mut pool = KvPool::with_blocks(64);
    let (a, b, c) = (tokens(21, 8), tokens(22, 8), tokens(23, 8));
    store(&mut t, &mut radix, &mut pool, &a, &[8]);
    store(&mut t, &mut radix, &mut pool, &b, &[8]);
    assert!(
        t.transport.log.is_empty(),
        "nothing idle yet: nothing written"
    );
    store(&mut t, &mut radix, &mut pool, &c, &[8]);
    let ra = t.token_run_keys(&a, 8);
    assert!(!t.aux_meta.contains_key(&ra[1]), "A's state left T1");
    let disk = &t.transport.durable;
    for k in [ra[0], ra[1], shard(ra[1], 0), shard(ra[1], 1)] {
        assert!(disk.contains_key(&k.0), "A's chain and state written first");
    }
    assert_eq!(
        t.transport.log[..2],
        [ra[0], ra[1]],
        "root-first, like the plan"
    );
    t.catalog.check_invariants();
}

/// KV-only: the RAM-only leaf T1 lets go of is written (with the chain it
/// hangs off) before it goes - the prompt still restores whole.
#[test]
fn eviction_writes_a_ram_only_chain_before_its_tail_goes() {
    let (mut t, mut radix, mut pool) = setup();
    let a = tokens(24, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[]);
    let runs = t.token_run_keys(&a, 8);
    evict_one(&mut t);
    assert!(
        t.catalog.ready_bytes(&runs[1], Tier::Ram).is_none(),
        "the tail left T1"
    );
    assert_eq!(t.transport.log, runs, "the whole chain written, head first");
    assert_eq!(
        t.probe(&a, 0).map(|h| h.end_block),
        Some(8),
        "and restores whole"
    );
}

/// A disk that declines (budget, reserve, paused) changes nothing about
/// eviction: the content goes, as it did before.
#[test]
fn eviction_proceeds_when_the_disk_declines() {
    let (mut t, mut radix, mut pool) = setup();
    t.transport.writes_left = Some(0);
    let a = tokens(25, 8);
    store(&mut t, &mut radix, &mut pool, &a, &[]);
    let runs = t.token_run_keys(&a, 8);
    evict_one(&mut t);
    assert!(
        t.catalog.ready_bytes(&runs[1], Tier::Ram).is_none(),
        "evicted anyway"
    );
    assert!(t.transport.durable.is_empty());
}

/// A disk read in flight holds the background plan back (no duplex), but
/// not the write that keeps a tip T1 is dropping: waiting is not on offer
/// there, and skipping it lost the session.
#[test]
fn eviction_writes_a_tip_while_the_disk_reads() {
    let mut t = disk_tier(120 << 20, DiskFake::default());
    let mut radix = armed_radix_of(&t);
    radix.set_state_capacity(8);
    let mut pool = KvPool::with_blocks(64);
    let (a, b, c) = (tokens(26, 8), tokens(27, 8), tokens(28, 8));
    store(&mut t, &mut radix, &mut pool, &a, &[8]);
    store(&mut t, &mut radix, &mut pool, &b, &[8]);
    t.transport.reading = true;
    let at = Instant::now() + IDLE * 2;
    assert_eq!(t.persist_pass(at, true, at), 0, "the plan waits for reads");
    store(&mut t, &mut radix, &mut pool, &c, &[8]);
    let ra = t.token_run_keys(&a, 8);
    assert!(!t.aux_meta.contains_key(&ra[1]), "A's state left T1");
    for k in [ra[0], ra[1], shard(ra[1], 0), shard(ra[1], 1)] {
        assert!(
            t.transport.durable.contains_key(&k.0),
            "written on the way out, reads or not"
        );
    }
}

/// Eviction's writes are budgeted by a rate: a burst already spent drops
/// the tip (as a declining disk would), and the same debt paid back over
/// time writes it.
#[test]
fn eviction_writes_spend_a_budget_that_refills() {
    for (ago, kept) in [(Duration::ZERO, false), (Duration::from_secs(5), true)] {
        let mut t = disk_tier(120 << 20, DiskFake::default());
        let mut radix = armed_radix_of(&t);
        radix.set_state_capacity(8);
        let mut pool = KvPool::with_blocks(64);
        let (a, b, c) = (tokens(29, 8), tokens(30, 8), tokens(31, 8));
        store(&mut t, &mut radix, &mut pool, &a, &[8]);
        store(&mut t, &mut radix, &mut pool, &b, &[8]);
        t.owe_eviction_writes(EVICT_WRITE_BURST, ago);
        store(&mut t, &mut radix, &mut pool, &c, &[8]);
        let ra = t.token_run_keys(&a, 8);
        assert!(!t.aux_meta.contains_key(&ra[1]), "A's state left T1");
        assert_eq!(
            t.transport.durable.contains_key(&shard(ra[1], 0).0),
            kept,
            "debt {ago:?} old"
        );
    }
}

// -- the exit flush -------------------------------------------------------------

/// Shutdown: a checkpoint only the GPU holds - never mirrored, the newest
/// state of a session whose turn ended just before - reaches disk with the
/// runs it resumes over. The exit flush used to write only what T1 held.
#[test]
fn the_exit_flush_keeps_state_only_the_gpu_holds() {
    let mut disk = DiskFake::default();
    disk.auto_deliver = true;
    let mut t = disk_tier(512 << 20, disk);
    let mut radix = armed_radix_of(&t);
    radix.set_state_paged(4, 3);
    t.declare_blob_pages(3);
    let mut pool = KvPool::with_blocks(64);
    let tokens = cached_chain(&mut radix, &mut pool, 31, 8);
    radix
        .attach_state_with_pool(&tokens, 8 * BLOCK_TOKENS, &mut pool)
        .expect("checkpoint");
    assert!(
        t.transport.durable.is_empty(),
        "nothing in T1, nothing on disk"
    );
    t.flush_live(&mut radix, &mut pool, &mut || None, None);
    let runs = t.token_run_keys(&tokens, 8);
    let disk = &t.transport.durable;
    assert!(runs.iter().all(|k| disk.contains_key(&k.0)), "the chain");
    assert!(disk.contains_key(&shard(runs[1], 0).0), "and its state");
    assert_eq!(t.transport.log.first(), Some(&runs[0]), "head first");
}
