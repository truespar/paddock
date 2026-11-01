//! KV tier gates (GPU): gather/scatter parity on synthesized pools,
//! and the raw round-trip byte-equality gate through the real path -
//! TierCatalog + RamTransport + registered T1 slabs + the pack kernels.
//!
//! Gated like every GPU gate: skips loudly without a pack or device,
//! `PADDOCK_STRICT_GATES=1` turns skips into failures.

mod common;

use cudarc::driver::DevicePtr;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::kv_tier::catalog::{LoadResult, LoadStart};
use paddock_engine::kv_tier::digest::{IdentityDigest, IdentityFields, PrivacyScope};
use paddock_engine::kv_tier::{
    CacheNamespace, LogicalKey, PlaneDesc, RamTransport, TierCatalog, TierCatalogConfig,
    TierTransport, XferSpec,
};
use paddock_engine::kv_tier::{LoadDst, Tier, WaiterId};

/// Deterministic per-plane pool content: byte i of plane p.
fn pattern(p: usize, i: usize) -> u8 {
    (p.wrapping_mul(131) ^ i.wrapping_mul(7)).wrapping_add(13) as u8
}

/// Geometry shared by the gates: 6 planes (3 layers x K/V), 1 KiB per block
/// per plane, a 16-block pool. Small deliberately - the parity is about
/// LAYOUT, the bandwidth story is R1's.
const N_PLANES: usize = 6;
const PLANE_BLOCK_BYTES: usize = 1024;
const POOL_BLOCKS: usize = 16;

fn build_pool(exec: &GpuExecutor) -> Vec<cudarc::driver::CudaSlice<u8>> {
    (0..N_PLANES)
        .map(|p| {
            let host: Vec<u8> = (0..POOL_BLOCKS * PLANE_BLOCK_BYTES)
                .map(|i| pattern(p, i))
                .collect();
            exec.to_device_u8(&host).expect("pool plane upload")
        })
        .collect()
}

fn plane_descs(exec: &GpuExecutor, pool: &[cudarc::driver::CudaSlice<u8>]) -> Vec<PlaneDesc> {
    pool.iter()
        .map(|pl| {
            let (p, _g) = pl.device_ptr(&exec.stream);
            PlaneDesc {
                base: p,
                stride: PLANE_BLOCK_BYTES as u64,
                bytes: PLANE_BLOCK_BYTES as u64,
            }
        })
        .collect()
}

/// The host-side reference: what the packed extent must hold for `ids`.
fn expected_extent(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * N_PLANES * PLANE_BLOCK_BYTES);
    for &b in ids {
        for p in 0..N_PLANES {
            let start = b as usize * PLANE_BLOCK_BYTES;
            out.extend((0..PLANE_BLOCK_BYTES).map(|i| pattern(p, start + i)));
        }
    }
    out
}

#[test]
fn gather_scatter_parity_vs_host_reference() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let pool = build_pool(&exec);
    let descs = plane_descs(&exec, &pool);
    // scattered, unordered, with a repeat - the gather must follow ids, not
    // pool order
    let ids: Vec<u32> = vec![7, 2, 11, 0, 2];
    let record = (N_PLANES * PLANE_BLOCK_BYTES) as u64;

    // descriptor buffers exactly as the transport builds them
    let mut img = Vec::new();
    let mut off = 0u64;
    for d in &descs {
        img.extend_from_slice(&[d.base, d.stride, d.bytes, off]);
        off += d.bytes;
    }
    let d_img = {
        let mut b = exec.alloc_u64(img.len()).expect("desc alloc");
        exec.upload_u64(&img, &mut b).expect("desc upload");
        b
    };
    let d_ids = exec.to_device_u32(&ids).expect("ids upload");
    let mut extent = exec
        .alloc_u8(ids.len() * record as usize)
        .expect("extent alloc");

    exec.kv_gather_blocks(
        &d_img,
        &d_ids,
        &mut extent,
        record,
        PLANE_BLOCK_BYTES as u64,
        N_PLANES,
        ids.len(),
    )
    .expect("gather launch");
    let got = exec
        .to_host_range_u8(&extent, 0, ids.len() * record as usize)
        .expect("extent readback");
    assert_eq!(
        got,
        expected_extent(&ids),
        "gather layout != host reference"
    );

    // scatter the extent into a ZEROED second pool at different block ids;
    // scattered blocks must match the gathered content, untouched blocks
    // must stay zero
    let pool2: Vec<_> = (0..N_PLANES)
        .map(|_| {
            exec.to_device_u8(&vec![0u8; POOL_BLOCKS * PLANE_BLOCK_BYTES])
                .unwrap()
        })
        .collect();
    let descs2 = plane_descs(&exec, &pool2);
    let mut img2 = Vec::new();
    let mut off2 = 0u64;
    for d in &descs2 {
        img2.extend_from_slice(&[d.base, d.stride, d.bytes, off2]);
        off2 += d.bytes;
    }
    let d_img2 = {
        let mut b = exec.alloc_u64(img2.len()).expect("desc alloc");
        exec.upload_u64(&img2, &mut b).expect("desc upload");
        b
    };
    let dst_ids: Vec<u32> = vec![1, 14, 3, 9, 5];
    let d_dst = exec.to_device_u32(&dst_ids).expect("dst ids");
    exec.kv_scatter_blocks(
        &d_img2,
        &d_dst,
        &extent,
        record,
        PLANE_BLOCK_BYTES as u64,
        N_PLANES,
        dst_ids.len(),
    )
    .expect("scatter launch");
    for (p, plane) in pool2.iter().enumerate() {
        let host = exec
            .to_host_range_u8(plane, 0, POOL_BLOCKS * PLANE_BLOCK_BYTES)
            .expect("readback");
        for blk in 0..POOL_BLOCKS {
            let dst_slot = dst_ids.iter().position(|&d| d as usize == blk);
            let got = &host[blk * PLANE_BLOCK_BYTES..(blk + 1) * PLANE_BLOCK_BYTES];
            match dst_slot {
                Some(r) => {
                    let src_blk = ids[r] as usize;
                    let want: Vec<u8> = (0..PLANE_BLOCK_BYTES)
                        .map(|i| pattern(p, src_blk * PLANE_BLOCK_BYTES + i))
                        .collect();
                    assert_eq!(got, &want[..], "plane {p} dst block {blk} (record {r})");
                }
                None => {
                    assert!(
                        got.iter().all(|&b| b == 0),
                        "plane {p} block {blk} written outside the id list"
                    );
                }
            }
        }
    }
    eprintln!(
        "kv tier gather/scatter parity: {} records x {N_PLANES} planes exact",
        ids.len()
    );
}

fn test_namespace() -> CacheNamespace {
    CacheNamespace {
        identity: IdentityDigest::compute(&IdentityFields {
            model_tensors: b"gate-model",
            adapter: b"",
            architecture: b"synthetic-pool",
            cache_schema: b"6x1024",
            layout_abi: 1,
            tokenizer: b"none",
        }),
        scope: PrivacyScope::Shared,
    }
}

/// Poll until the transport yields completions (its work is event-driven on
/// its own lane; a bounded spin keeps the gate honest about hangs).
fn poll_until(t: &mut RamTransport, n: usize) -> Vec<paddock_engine::kv_tier::IoCompletion> {
    let mut out = Vec::new();
    for _ in 0..2000 {
        out.extend(t.poll());
        if out.len() >= n {
            return out;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    panic!(
        "transport produced {} of {n} completions within 2 s",
        out.len()
    );
}

#[test]
fn t1_slabs_past_the_first_build_off_thread_and_register() {
    // The budget past the self-test's slab builds on the supply thread
    // (host.rs `SlabSupply`) - never inside a serving tick, where one GiB
    // stalled the engine ~1 s. The thread binds the context itself, so its
    // slabs must come back page-locked exactly like the first.
    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let mut transport = RamTransport::new(&exec, 3 << 30).expect("transport");
    let first_pinned = transport.t1_slabs().2 == 1;
    let t0 = std::time::Instant::now();
    let (n, pinned) = loop {
        let (done, n, pinned) = transport.t1_slabs();
        if done {
            break (n, pinned);
        }
        assert!(t0.elapsed().as_secs() < 120, "slab supply never finished");
        std::thread::sleep(std::time::Duration::from_millis(5));
    };
    eprintln!(
        "kv tier T1: {n} slabs, {pinned} pinned, carved in {} ms",
        t0.elapsed().as_millis()
    );
    assert_eq!(n, 3, "the whole 3 GiB budget is carved");
    if first_pinned {
        assert_eq!(pinned, 3, "the supply's slabs register like the first");
    }
}

#[test]
fn catalog_ram_transport_roundtrip_is_byte_exact() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let mut transport = RamTransport::new(&exec, 64 << 20).expect("transport");
    eprintln!("kv tier T1 mode: {:?}", transport.host_mode());
    let mut catalog = TierCatalog::new(TierCatalogConfig {
        ram_capacity: 64 << 20,
        nvme_capacity: 0,
    });

    // ---- demote: 4 scattered blocks off a synthesized pool -----------------
    let pool = build_pool(&exec);
    let descs = plane_descs(&exec, &pool);
    let ids: Vec<u32> = vec![3, 8, 1, 12];
    let record = (N_PLANES * PLANE_BLOCK_BYTES) as u64;
    let sealed = record * ids.len() as u64;
    let key: LogicalKey = test_namespace().root().child(&[1, 2, 3]);

    // fence: the pool uploads ran on the serving stream; the transport's
    // lane must not gather before they land
    let seal_ev = exec.record_event().expect("seal event");
    transport
        .expect_store(
            key,
            XferSpec {
                planes: descs.clone(),
                block_ids: ids.clone(),
                after: Some(seal_ev),
            },
        )
        .expect("expect_store");
    catalog.reserve(key, Tier::Ram, sealed).unwrap();
    let (_op, job) = catalog.begin_store_adopt(key, Tier::Ram, sealed).unwrap();
    transport.submit(job).expect("submit store");
    for c in poll_until(&mut transport, 1) {
        assert!(catalog.on_completion(c).is_empty());
    }
    catalog.check_invariants();
    assert_eq!(
        catalog.ready_bytes(&key, Tier::Ram),
        Some(sealed),
        "demote published"
    );

    // ---- destroy the GPU copy (the eviction this restore exists for) ------
    drop(pool);

    // ---- restore into a fresh zeroed pool at different block ids ----------
    let pool2: Vec<_> = (0..N_PLANES)
        .map(|_| {
            exec.to_device_u8(&vec![0u8; POOL_BLOCKS * PLANE_BLOCK_BYTES])
                .unwrap()
        })
        .collect();
    let descs2 = plane_descs(&exec, &pool2);
    let dst_ids: Vec<u32> = vec![0, 5, 10, 15];
    let dst_ev = exec.record_event().expect("dst event");
    transport
        .expect_load(
            key,
            XferSpec {
                planes: descs2,
                block_ids: dst_ids.clone(),
                after: Some(dst_ev),
            },
        )
        .expect("expect_load");
    let start = catalog
        .begin_load(key, Tier::Ram, LoadDst::Gpu, WaiterId(7))
        .unwrap();
    let LoadStart::Started { op: _, job } = start else {
        panic!("fresh load must start")
    };
    transport.submit(job).expect("submit load");
    let mut wakes = Vec::new();
    for c in poll_until(&mut transport, 1) {
        wakes.extend(catalog.on_completion(c));
    }
    assert_eq!(wakes.len(), 1);
    assert_eq!(
        wakes[0].result,
        LoadResult::Ok,
        "restore integrity verified"
    );
    catalog.check_invariants();

    // scatter runs on the transport's lane; drain it before reading back
    transport.lane().synchronize().expect("lane drain");

    // ---- the gate: restored bytes == original pool bytes, exactly ---------
    for (p, plane) in pool2.iter().enumerate() {
        let host = exec
            .to_host_range_u8(plane, 0, POOL_BLOCKS * PLANE_BLOCK_BYTES)
            .expect("readback");
        for (r, &dst) in dst_ids.iter().enumerate() {
            let src_blk = ids[r] as usize;
            let got =
                &host[dst as usize * PLANE_BLOCK_BYTES..(dst as usize + 1) * PLANE_BLOCK_BYTES];
            let want: Vec<u8> = (0..PLANE_BLOCK_BYTES)
                .map(|i| pattern(p, src_blk * PLANE_BLOCK_BYTES + i))
                .collect();
            assert_eq!(
                got,
                &want[..],
                "plane {p} record {r}: restored bytes differ"
            );
        }
    }
    eprintln!(
        "kv tier round trip: {} blocks x {N_PLANES} planes demoted to T1 and \
         restored byte-exact ({:?})",
        ids.len(),
        transport.host_mode()
    );

    // ---- eviction: catalog releases the ledger bytes ----------------------
    let freed = catalog.evict(&key, Tier::Ram).expect("evict");
    assert_eq!(freed, sealed);
}

#[test]
fn cancelled_store_completes_failed_and_releases_everything() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let mut transport = RamTransport::new(&exec, 16 << 20).expect("transport");
    let mut catalog = TierCatalog::new(TierCatalogConfig {
        ram_capacity: 16 << 20,
        nvme_capacity: 0,
    });
    let pool = build_pool(&exec);
    let descs = plane_descs(&exec, &pool);
    let ids: Vec<u32> = vec![2, 4];
    let sealed = (N_PLANES * PLANE_BLOCK_BYTES) as u64 * ids.len() as u64;
    let key: LogicalKey = test_namespace().root().child(&[9]);
    transport
        .expect_store(
            key,
            XferSpec {
                planes: descs,
                block_ids: ids,
                after: None,
            },
        )
        .expect("expect_store");
    catalog.reserve(key, Tier::Ram, sealed).unwrap();
    let (op, job) = catalog.begin_store_adopt(key, Tier::Ram, sealed).unwrap();
    transport.submit(job).expect("submit");
    // cancel immediately - catalog first (its op dies now), then transport
    assert!(catalog.cancel_store(op));
    transport.cancel(op);
    let comps = poll_until(&mut transport, 1);
    for c in comps {
        catalog.on_completion(c); // stale by definition - op removed
    }
    assert_eq!(
        catalog.counters.stale_completions, 1,
        "post-cancel completion is stale"
    );
    catalog.check_invariants();
    assert_eq!(
        catalog.ledger(Tier::Ram).free,
        16 << 20,
        "everything released"
    );
    // the physical T1 extent must be gone too - the transport frees it on
    // the cancelled-store path
    assert_eq!(
        transport.t1_allocated(),
        0,
        "cancelled store leaked its T1 extent"
    );
}

/// A store a restore waits on goes ahead of the load-first rule: queued
/// behind a load in flight, `expedite` launches it anyway - and both land.
#[test]
fn an_expedited_store_launches_beside_a_load_in_flight() {
    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let mut transport = RamTransport::new(&exec, 64 << 20).expect("transport");
    let mut catalog = TierCatalog::new(TierCatalogConfig {
        ram_capacity: 64 << 20,
        nvme_capacity: 0,
    });
    let pool = build_pool(&exec);
    let descs = plane_descs(&exec, &pool);
    let sealed = (N_PLANES * PLANE_BLOCK_BYTES) as u64 * 2;
    let store = |transport: &mut RamTransport, catalog: &mut TierCatalog, key, ids: Vec<u32>| {
        transport
            .expect_store(
                key,
                XferSpec {
                    planes: descs.clone(),
                    block_ids: ids,
                    after: None,
                },
            )
            .expect("expect_store");
        catalog.reserve(key, Tier::Ram, sealed).unwrap();
        let (op, job) = catalog.begin_store_adopt(key, Tier::Ram, sealed).unwrap();
        transport.submit(job).expect("submit store");
        op
    };
    let a: LogicalKey = test_namespace().root().child(&[21]);
    let b: LogicalKey = test_namespace().root().child(&[22]);
    store(&mut transport, &mut catalog, a, vec![1, 2]);
    for c in poll_until(&mut transport, 1) {
        catalog.on_completion(c);
    }
    assert!(catalog.ready_bytes(&a, Tier::Ram).is_some());

    // a load of A in flight (not polled, so it stays on the lane's books)
    transport
        .expect_load(
            a,
            XferSpec {
                planes: descs.clone(),
                block_ids: vec![5, 6],
                after: None,
            },
        )
        .expect("expect_load");
    let LoadStart::Started { op: _, job } = catalog
        .begin_load(a, Tier::Ram, LoadDst::Gpu, WaiterId(1))
        .unwrap()
    else {
        panic!("fresh load must start")
    };
    transport.submit(job).expect("submit load");
    let op_b = store(&mut transport, &mut catalog, b, vec![3, 4]);
    assert_eq!(transport.stores_queued(), 1, "held behind the load");
    transport.expedite_inner(&[op_b]);
    assert_eq!(transport.stores_queued(), 0, "launched beside it");
    let mut wakes = Vec::new();
    for c in poll_until(&mut transport, 2) {
        wakes.extend(catalog.on_completion(c));
    }
    assert!(wakes.iter().all(|w| w.result == LoadResult::Ok));
    assert_eq!(catalog.ready_bytes(&b, Tier::Ram), Some(sealed), "B landed");
    catalog.check_invariants();
}

/// The full 1a.3 path on real hardware: a radix-cached chain demotes to T1
/// under eviction pressure (pins deferring the release), the GPU copy is
/// destroyed, and a later prompt probes + restores + PUBLISHES the prefix
/// back into the radix - where adoption serves it like any warm hit. Byte
/// equality against the original pool content closes the loop.
#[test]
fn pool_tier_demote_restore_publish_is_byte_exact() {
    use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
    use paddock_engine::kv_tier::PoolTier;
    use paddock_engine::paged_radix::PagedRadix;

    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let transport = RamTransport::new(&exec, 64 << 20).expect("transport");
    let mut dev_pool = build_pool(&exec);
    let descs = plane_descs(&exec, &dev_pool);
    let mut tier = PoolTier::new(&test_namespace(), descs, 64 << 20, transport).expect("tier");
    tier.cost.set_force_restore(true);
    let n = 8usize; // one full run at this geometry (record 6 KiB -> R = 8)
    assert_eq!(tier.run_blocks(), 8);
    let mut pool = KvPool::with_blocks(POOL_BLOCKS as u32);
    let mut radix = PagedRadix::new();
    radix.set_tier_root(tier.tier_root());

    // a slot prefills an 8-block chain (block ids 0..8 in order) + caches it
    let tokens: Vec<u32> = (0..n * BLOCK_TOKENS)
        .map(|i| 5000 + i as u32)
        .chain([9])
        .collect();
    let mut table = BlockTable::new();
    table.ensure(n * BLOCK_TOKENS - 1, &mut pool).unwrap();
    let chain_blocks: Vec<u32> = table.blocks().to_vec();
    radix.insert(&tokens, table.blocks(), &mut pool);
    table.clear(&mut pool);

    // demote under pressure: the whole chain evicts, the run stores async
    let seal = exec.record_event().expect("seal event");
    let (evicted, aux) = tier.pressure_demote(&mut radix, &mut pool, POOL_BLOCKS, Some(seal));
    assert_eq!(evicted, n);
    assert!(aux.is_empty());
    for _ in 0..2000 {
        tier.pump_completions(&mut radix, &mut pool);
        if tier.stats().2 == 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    assert_eq!(
        tier.stats(),
        (1, (N_PLANES * PLANE_BLOCK_BYTES * n) as u64, 0, 0)
    );
    assert_eq!(pool.free_blocks(), POOL_BLOCKS, "demote pins released");

    // destroy the GPU copy: zero every plane in PLACE, and prove it - a
    // vacuous destroy would make the byte gate below pass without T1
    let zeros = vec![0u8; POOL_BLOCKS * PLANE_BLOCK_BYTES];
    for plane in dev_pool.iter_mut() {
        exec.upload_u8(&zeros, plane).expect("zero plane");
    }
    let check = exec
        .to_host_range_u8(&dev_pool[0], 0, POOL_BLOCKS * PLANE_BLOCK_BYTES)
        .expect("verify destroy");
    assert!(
        check.iter().all(|&b| b == 0),
        "planes must actually be destroyed"
    );

    // skew the free list so restored block ids differ from the original
    // chain ids - dst == src would let an untouched pool masquerade as a
    // successful restore
    let skew = pool.alloc().unwrap();

    // probe + restore + publish
    let hit = tier.probe(&tokens, 0).expect("run restorable");
    assert_eq!((hit.start_block, hit.end_block), (0, n));
    let dst_ev = exec.record_event().expect("dst event");
    let ticket = tier
        .begin_restore(&hit, &tokens, &mut pool, Some(dst_ev))
        .expect("ticket");
    let mut wake = None;
    for _ in 0..2000 {
        tier.pump_completions(&mut radix, &mut pool);
        wake = tier.take_wake(ticket);
        if wake.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let wake = wake.expect("restore resolved");
    assert!(wake.ok && wake.end_block == n);
    tier.transport.lane().synchronize().expect("lane drain");
    pool.release(skew);

    // adoption path: the radix serves the restored prefix
    let restored = radix.match_prefix(&tokens);
    assert_eq!(restored.len(), n, "published prefix matches");
    assert_ne!(
        restored, chain_blocks,
        "the skew must have moved the destination ids"
    );

    // byte gate: restored block r must hold source block chain_blocks[r]'s
    // original pattern, in every plane
    for (p, plane) in dev_pool.iter().enumerate() {
        let host = exec
            .to_host_range_u8(plane, 0, POOL_BLOCKS * PLANE_BLOCK_BYTES)
            .expect("readback");
        for (r, &dst) in restored.iter().enumerate() {
            let src = chain_blocks[r] as usize;
            let got =
                &host[dst as usize * PLANE_BLOCK_BYTES..(dst as usize + 1) * PLANE_BLOCK_BYTES];
            let want: Vec<u8> = (0..PLANE_BLOCK_BYTES)
                .map(|i| pattern(p, src * PLANE_BLOCK_BYTES + i))
                .collect();
            assert_eq!(got, &want[..], "plane {p} restored record {r}");
        }
    }
    tier.catalog.check_invariants();
    eprintln!(
        "pool tier e2e: {n}-block chain demoted under pressure, GPU copy destroyed, \
         restored + republished byte-exact ({:?})",
        tier.transport.host_mode()
    );
}

/// The headline: a demoted prefix SURVIVES A RUNNER RESTART. The
/// tier + transport are torn down completely and rebuilt from the on-disk
/// store; the recovered entries preload into the catalog, the probe hits
/// from the Nvme tier, and the restore is byte-exact against the original
/// pool content. "Your session's cache survives a restart" - the moat
/// claim, as a test.
#[test]
fn t2_restart_persistence_survives_a_full_rebuild() {
    use paddock_engine::kv_pool::{BLOCK_TOKENS, BlockTable, KvPool};
    use paddock_engine::kv_tier::PoolTier;
    use paddock_engine::paged_radix::PagedRadix;

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
    let Some(exec) = common::gpu() else { return };
    if !exec.has_kv_tier_xfer() {
        common::missing("pack has no kv tier gather/scatter (slots 479/480) - rebuild it");
        return;
    }
    let t2dir = std::env::temp_dir().join(format!("pkv-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&t2dir);

    let n = 8usize;
    let tokens: Vec<u32> = (0..n * BLOCK_TOKENS)
        .map(|i| 7000 + i as u32)
        .chain([9])
        .collect();
    let chain_blocks: Vec<u32>;

    // ---- process one: demote; the exit flush writes the chain ------------
    {
        let transport =
            RamTransport::with_t2(&exec, 64 << 20, &t2dir, 1 << 30).expect("transport with T2");
        let dev_pool = build_pool(&exec);
        let descs = plane_descs(&exec, &dev_pool);
        let mut tier =
            PoolTier::with_capacities(&test_namespace(), descs, 64 << 20, 1 << 30, transport)
                .expect("tier");
        let mut pool = KvPool::with_blocks(POOL_BLOCKS as u32);
        let mut radix = PagedRadix::new();
        radix.set_tier_root(tier.tier_root());
        let mut table = BlockTable::new();
        table.ensure(n * BLOCK_TOKENS - 1, &mut pool).unwrap();
        chain_blocks = table.blocks().to_vec();
        radix.insert(&tokens, table.blocks(), &mut pool);
        table.clear(&mut pool);
        let seal = exec.record_event().expect("seal");
        tier.pressure_demote(&mut radix, &mut pool, POOL_BLOCKS, Some(seal));
        for _ in 0..2000 {
            tier.pump_completions(&mut radix, &mut pool);
            if tier.stats().2 == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(tier.stats().0 >= 1, "run resident in T1");
        // nothing rides the demote to disk: the durability plan writes a
        // chain's tip once it idles, or at exit - and nothing is queued for
        // disk without a plan
        assert_eq!(
            tier.pending_durable_writes(),
            0,
            "no disk write outside a plan"
        );
        // dev_pool, tier, transport all drop here - the "runner exit", whose
        // flush writes the chain root-first (process two recovering it is
        // the proof)
    }

    // ---- process two: rebuild everything from the directory --------------
    let transport =
        RamTransport::with_t2(&exec, 64 << 20, &t2dir, 1 << 30).expect("transport reopen");
    assert!(
        transport.t2().map(|t| t.stats().live_entries).unwrap_or(0) >= 1,
        "the store recovered the demoted run"
    );
    let dev_pool2 = build_pool(&exec); // fresh planes - the old GPU state is gone
    // overwrite them so the restored bytes can only come from disk
    let mut dev_pool2 = dev_pool2;
    let zeros = vec![0u8; POOL_BLOCKS * PLANE_BLOCK_BYTES];
    for plane in dev_pool2.iter_mut() {
        exec.upload_u8(&zeros, plane).expect("zero");
    }
    let descs2 = plane_descs(&exec, &dev_pool2);
    let mut tier =
        PoolTier::with_capacities(&test_namespace(), descs2, 64 << 20, 1 << 30, transport)
            .expect("tier reopen");
    let loaded = tier.preload_from_t2();
    assert!(loaded >= 1, "recovered entries visible to the catalog");
    tier.cost.set_force_restore(true);
    let mut pool = KvPool::with_blocks(POOL_BLOCKS as u32);
    let mut radix = PagedRadix::new();
    radix.set_tier_root(tier.tier_root());

    let hit = tier
        .probe(&tokens, 0)
        .expect("restart probe hits from the Nvme tier");
    assert_eq!((hit.start_block, hit.end_block), (0, n));
    let skew = pool.alloc().unwrap(); // dst ids must differ from source ids
    let dst_ev = exec.record_event().expect("ev");
    let ticket = tier
        .begin_restore(&hit, &tokens, &mut pool, Some(dst_ev))
        .expect("ticket");
    let mut wake = None;
    for _ in 0..2000 {
        tier.pump_completions(&mut radix, &mut pool);
        wake = tier.take_wake(ticket);
        if wake.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let wake = wake.expect("restore resolved");
    assert!(
        wake.ok && wake.end_block == n,
        "restored through the boundary: {wake:?}"
    );
    tier.transport.lane().synchronize().expect("drain");
    pool.release(skew);

    let restored = radix.match_prefix(&tokens);
    assert_eq!(restored.len(), n);
    assert_ne!(restored, chain_blocks, "skew moved the destination ids");
    for (p, plane) in dev_pool2.iter().enumerate() {
        let host = exec
            .to_host_range_u8(plane, 0, POOL_BLOCKS * PLANE_BLOCK_BYTES)
            .expect("read");
        for (r, &dst) in restored.iter().enumerate() {
            let src = chain_blocks[r] as usize;
            let got =
                &host[dst as usize * PLANE_BLOCK_BYTES..(dst as usize + 1) * PLANE_BLOCK_BYTES];
            let want: Vec<u8> = (0..PLANE_BLOCK_BYTES)
                .map(|i| pattern(p, src * PLANE_BLOCK_BYTES + i))
                .collect();
            assert_eq!(
                got,
                &want[..],
                "plane {p} record {r}: bytes must come FROM DISK"
            );
        }
    }
    tier.catalog.check_invariants();

    // ---- round three: the read-fill promotion serves the same bytes ------
    // The T2 loads seated T1 copies on their way through the pinned ring;
    // evict the published chain and restore again - this time sourced from
    // RAM - and the planes must land byte-exact once more.
    tier.pump_completions(&mut radix, &mut pool); // drain the promotion outbox
    assert!(
        tier.catalog
            .ledger(paddock_engine::kv_tier::Tier::Ram)
            .ready
            > 0,
        "read-fill adopted into the RAM tier"
    );
    while radix.evict_lru(&mut pool).is_some() {}
    for plane in dev_pool2.iter_mut() {
        exec.upload_u8(&zeros, plane).expect("re-zero");
    }
    let hit = tier.probe(&tokens, 0).expect("second probe hits (RAM now)");
    assert_eq!((hit.start_block, hit.end_block), (0, n));
    let dst_ev = exec.record_event().expect("ev");
    let ticket = tier
        .begin_restore(&hit, &tokens, &mut pool, Some(dst_ev))
        .expect("ticket2");
    let mut wake = None;
    for _ in 0..2000 {
        tier.pump_completions(&mut radix, &mut pool);
        wake = tier.take_wake(ticket);
        if wake.is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let wake = wake.expect("RAM restore resolved");
    assert!(wake.ok && wake.end_block == n, "RAM round: {wake:?}");
    tier.transport.lane().synchronize().expect("drain");
    let restored = radix.match_prefix(&tokens);
    assert_eq!(restored.len(), n);
    for (p, plane) in dev_pool2.iter().enumerate() {
        let host = exec
            .to_host_range_u8(plane, 0, POOL_BLOCKS * PLANE_BLOCK_BYTES)
            .expect("read");
        for (r, &dst) in restored.iter().enumerate() {
            let src = chain_blocks[r] as usize;
            let got =
                &host[dst as usize * PLANE_BLOCK_BYTES..(dst as usize + 1) * PLANE_BLOCK_BYTES];
            let want: Vec<u8> = (0..PLANE_BLOCK_BYTES)
                .map(|i| pattern(p, src * PLANE_BLOCK_BYTES + i))
                .collect();
            assert_eq!(got, &want[..], "plane {p} record {r}: promoted RAM bytes");
        }
    }
    tier.catalog.check_invariants();
    eprintln!(
        "T2 restart persistence: {n}-block chain demoted, tier fully rebuilt from disk, \
         restored byte-exact ({} entries preloaded), read-fill re-restore byte-exact from RAM",
        loaded
    );
    let _ = std::fs::remove_dir_all(&t2dir);
}
