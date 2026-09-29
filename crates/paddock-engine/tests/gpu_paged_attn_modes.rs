//! Phase A of paging Flash-Next's live KV: the paged address mode of every
//! kernel that still reads a dense slot-major strip (pack slots 688-696) -
//! the multi-slot prefill walk, the three decode arms, and the QSA index
//! store / scores / attention. Each paged launch must be BIT-IDENTICAL to its
//! dense twin over the same keys: the address is the only change.
//!
//! The tables are scrambled and interleaved (every slot's pages shuffled
//! through one pool, spare blocks between them): an identity or slot-ordered
//! table is blind to a kernel that adds the slot base twice or ignores the
//! table's slot stride. Rows past a slot's live keys are NaN in both layouts
//! and a slot's table entries past its live keys name a NaN block, so a
//! kernel that reads past the keys it may reach poisons its output (a masked
//! V row still multiplies a zero weight) and fails the finiteness check even
//! when both layouts agree.
//!
//! Synthetic data, no model: these run in seconds.

mod common;

use half::{bf16, f16};
use paddock_engine::gpu::qsa::QsaRoute;
use paddock_engine::gpu::{GpuExecutor, KvDtype};

/// The house deterministic input LCG.
fn det(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

/// A pool page holds 16 tokens (the engine's KV block).
const PAGE: usize = 16;

/// One scrambled paged layout of `slots` dense strips of `max_ctx` tokens.
struct Layout {
    slots: usize,
    max_ctx: usize,
    bps: usize,
    /// tokens each slot's launches may reach (its pages are mapped)
    live: Vec<usize>,
    /// [slots * bps] pool block of each (slot, page); pages past `live` name
    /// `poison`
    table: Vec<u32>,
    pool_blocks: usize,
    poison: u32,
}

impl Layout {
    fn new(max_ctx: usize, live: Vec<usize>, seed: u64) -> Self {
        let slots = live.len();
        let bps = max_ctx / PAGE;
        let pages: Vec<usize> = live.iter().map(|l| l.div_ceil(PAGE)).collect();
        let mapped: usize = pages.iter().sum();
        // spare blocks the table never names (the poison block among them)
        let pool_blocks = mapped + 5;
        let mut ids: Vec<u32> = (0..pool_blocks as u32).collect();
        let mut s = seed;
        for i in (1..ids.len()).rev() {
            ids.swap(i, lcg(&mut s) as usize % (i + 1));
        }
        let poison = ids[mapped + 2];
        let mut table = vec![poison; slots * bps];
        // page-major across slots: slot 0's page 0, slot 1's page 0, ... so
        // the slots' blocks interleave through the shuffled pool
        let mut next = 0;
        for i in 0..bps {
            for (sl, &np) in pages.iter().enumerate() {
                if i < np {
                    table[sl * bps + i] = ids[next];
                    next += 1;
                }
            }
        }
        assert_eq!(next, mapped);
        Self {
            slots,
            max_ctx,
            bps,
            live,
            table,
            pool_blocks,
            poison,
        }
    }

    /// A dense plane of `rpp` rows (of `row` elements) per 16-token page -
    /// 16 for KV, 16/cr for the QSA index - copied into pool order. Blocks
    /// no slot maps hold `fill`.
    fn scatter<T: Copy>(&self, dense: &[T], row: usize, rpp: usize, fill: T) -> Vec<T> {
        let mut pool = vec![fill; self.pool_blocks * rpp * row];
        let dense_page = |s: usize, i: usize| (s * self.bps + i) * rpp * row;
        for s in 0..self.slots {
            for i in 0..self.live[s].div_ceil(PAGE) {
                let b = self.table[s * self.bps + i] as usize;
                let src = &dense[dense_page(s, i)..dense_page(s, i) + rpp * row];
                pool[b * rpp * row..(b + 1) * rpp * row].copy_from_slice(src);
            }
        }
        pool
    }

    /// The inverse over the mapped pages: pool order back to the dense
    /// layout, unmapped pages `fill`.
    fn gather<T: Copy>(&self, pool: &[T], row: usize, rpp: usize, fill: T) -> Vec<T> {
        let mut dense = vec![fill; self.slots * self.bps * rpp * row];
        for s in 0..self.slots {
            for i in 0..self.live[s].div_ceil(PAGE) {
                let b = self.table[s * self.bps + i] as usize;
                let dst = (s * self.bps + i) * rpp * row;
                dense[dst..dst + rpp * row]
                    .copy_from_slice(&pool[b * rpp * row..(b + 1) * rpp * row]);
            }
        }
        dense
    }
}

/// KV element bytes for `dtype`: finite values in about [-4, 4].
fn kv_bytes(n: usize, dtype: KvDtype, seed: u64) -> Vec<u8> {
    match dtype {
        KvDtype::Fp16 => det(n, seed)
            .iter()
            .flat_map(|&v| f16::from_f32(v * 4.0).to_le_bytes())
            .collect(),
        KvDtype::Fp8E4m3 => {
            // sign | exponent field 0..=8 | mantissa: |v| <= 3.75, never NaN
            let mut s = seed;
            (0..n)
                .map(|_| {
                    let r = lcg(&mut s);
                    (((r & 1) << 7) | (((r >> 1) % 9) << 3) | ((r >> 5) & 7)) as u8
                })
                .collect()
        }
    }
}

/// One NaN KV element's bytes.
fn kv_nan(dtype: KvDtype) -> Vec<u8> {
    match dtype {
        KvDtype::Fp16 => f16::NAN.to_le_bytes().to_vec(),
        KvDtype::Fp8E4m3 => vec![0x7f],
    }
}

/// Dense K and V strips for `lay`: random over each slot's live tokens,
/// NaN past them. Returns (dense K, dense V, pool K, pool V) as bytes.
fn kv_planes(lay: &Layout, kv_dim: usize, dtype: KvDtype, seed: u64) -> [Vec<u8>; 4] {
    let eb = dtype.bytes();
    let row = kv_dim * eb;
    let nan: Vec<u8> = kv_nan(dtype).repeat(kv_dim);
    let strip = |which: u64| {
        let mut dense = kv_bytes(lay.slots * lay.max_ctx * kv_dim, dtype, seed + which);
        for s in 0..lay.slots {
            for t in lay.live[s]..lay.max_ctx {
                let o = (s * lay.max_ctx + t) * row;
                dense[o..o + row].copy_from_slice(&nan);
            }
        }
        dense
    };
    let (dk, dv) = (strip(0), strip(1));
    // the pool: whole rows of `row` bytes; the poison block and the spares NaN
    let pk = scatter_rows(lay, &dk, row, &nan);
    let pv = scatter_rows(lay, &dv, row, &nan);
    [dk, dv, pk, pv]
}

/// `Layout::scatter` over byte rows of any width (a NaN row as the fill).
fn scatter_rows(lay: &Layout, dense: &[u8], row: usize, nan_row: &[u8]) -> Vec<u8> {
    // scatter as u8 with a zero fill, then paint every unmapped block NaN
    let mut pool = lay.scatter(dense, row, PAGE, 0u8);
    let mut mapped = vec![false; lay.pool_blocks];
    for s in 0..lay.slots {
        for i in 0..lay.live[s].div_ceil(PAGE) {
            mapped[lay.table[s * lay.bps + i] as usize] = true;
        }
    }
    for (b, m) in mapped.iter().enumerate() {
        if !m {
            for r in 0..PAGE {
                let o = (b * PAGE + r) * row;
                pool[o..o + row].copy_from_slice(nan_row);
            }
        }
    }
    assert!(!mapped[lay.poison as usize]);
    pool
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Dense and paged outputs must be the same bits, and finite.
fn assert_same(dense: &[f32], paged: &[f32], what: &str) {
    assert_eq!(dense.len(), paged.len(), "{what}: length");
    if let Some(i) = dense.iter().position(|v| !v.is_finite()) {
        panic!(
            "{what}: dense output [{i}] = {} (a read past the live keys?)",
            dense[i]
        );
    }
    if let Some(i) = paged.iter().position(|v| !v.is_finite()) {
        panic!(
            "{what}: paged output [{i}] = {} (a read past the live keys?)",
            paged[i]
        );
    }
    let (d, p) = (bits(dense), bits(paged));
    if let Some(i) = d.iter().zip(&p).position(|(a, b)| a != b) {
        panic!(
            "{what}: paged differs from dense first at [{i}]: {} vs {} ({} of {} differ)",
            paged[i],
            dense[i],
            d.iter().zip(&p).filter(|(a, b)| a != b).count(),
            d.len()
        );
    }
}

fn exec_with_modes() -> Option<GpuExecutor> {
    let exec = common::gpu()?;
    if !exec.has_attn_paged_modes() || !exec.has_qsa_paged() {
        common::missing("pack has no paged address modes (slots 688-696; rebuild packs/cuda)");
        return None;
    }
    Some(exec)
}

/// The geometries: Flash-Next's (24 q / 2 kv heads of 256) and a 128-wide
/// head, whose prefill tiles take 32 keys and so span two pages.
const GEOMS: [(usize, usize, usize); 2] = [(24, 2, 256), (16, 2, 128)];
const DTYPES: [KvDtype; 2] = [KvDtype::Fp16, KvDtype::Fp8E4m3];

/// Four slots over 512-token strips: a fresh prompt, a resumed chunk, a lone
/// row, and a slot filled to its last token.
fn attn_layout() -> Layout {
    Layout::new(512, vec![341, 18, 71, 512], 0x5eed)
}

#[test]
fn prefill_batch_paged_matches_dense() {
    let Some(e) = exec_with_modes() else { return };
    let lay = attn_layout();
    // (slot, first position, rows): each run's tiles stay inside it
    let runs = [
        (2usize, 0usize, 71usize),
        (0, 300, 41),
        (1, 17, 1),
        (3, 500, 12),
    ];
    let (mut pos, mut slot, mut t_row0, mut t_slot) = (vec![], vec![], vec![], vec![]);
    for &(s, p0, n) in &runs {
        let r0 = pos.len() as u32;
        for i in 0..n {
            pos.push((p0 + i) as u32);
            slot.push(s as u32);
        }
        for t in 0..n.div_ceil(16) {
            t_row0.push(r0 + 16 * t as u32);
            t_slot.push(s as u32);
        }
    }
    let n = pos.len();
    let d_pos = e.to_device_u32(&pos).unwrap();
    let d_slot = e.to_device_u32(&slot).unwrap();
    let d_r0 = e.to_device_u32(&t_row0).unwrap();
    let d_ts = e.to_device_u32(&t_slot).unwrap();
    let d_bt = e.to_device_u32(&lay.table).unwrap();
    // + gemma4's global layer: at hd 512 the scalar tile sits 384 B under the
    // 99 KB opt-in cap, and it was refused on sm_121a while its window floors
    // were static shared (pd_apf_smem)
    for (nh, nkv, hd) in GEOMS.into_iter().chain([(16, 2, 512)]) {
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let d_q = e.to_device(&det(n * nh * hd, 11)).unwrap();
        let d_sinks = e.to_device(&det(nh, 12)).unwrap();
        for dtype in DTYPES {
            let [dk, dv, pk, pv] = kv_planes(&lay, kv_dim, dtype, 13);
            let (d_dk, d_dv) = (e.to_device_u8(&dk).unwrap(), e.to_device_u8(&dv).unwrap());
            let (d_pk, d_pv) = (e.to_device_u8(&pk).unwrap(), e.to_device_u8(&pv).unwrap());
            for swa in [0usize, 100] {
                let mut d_out = e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap();
                let mut p_out = e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap();
                e.attn_prefill_batch(
                    &d_q,
                    &d_dk,
                    &d_dv,
                    &d_sinks,
                    &mut d_out,
                    &d_pos,
                    &d_slot,
                    &d_r0,
                    &d_ts,
                    t_row0.len(),
                    nh,
                    nkv,
                    hd,
                    lay.max_ctx,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                e.attn_prefill_batch_paged(
                    &d_q,
                    &d_pk,
                    &d_pv,
                    &d_sinks,
                    &mut p_out,
                    &d_pos,
                    &d_slot,
                    &d_bt,
                    lay.bps,
                    &d_r0,
                    &d_ts,
                    t_row0.len(),
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                let what = format!("prefill_batch hd{hd} {dtype:?} swa{swa}");
                assert_same(
                    &e.to_host(&d_out).unwrap(),
                    &e.to_host(&p_out).unwrap(),
                    &what,
                );
                eprintln!("{what}: {n} rows bit-identical");
            }
        }
    }
}

#[test]
fn decode_arms_paged_match_dense() {
    let Some(e) = exec_with_modes() else { return };
    let lay = attn_layout();
    // one row per slot at its last live token, plus a verify-shaped pair
    // sharing slot 0 (rows of one slot at consecutive positions)
    let rows = [(2u32, 70u32), (0, 340), (3, 511), (1, 17), (0, 339)];
    let n = rows.len();
    let d_pos = e
        .to_device_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>())
        .unwrap();
    let d_slot = e
        .to_device_u32(&rows.iter().map(|r| r.0).collect::<Vec<_>>())
        .unwrap();
    let d_bt = e.to_device_u32(&lay.table).unwrap();
    for (nh, nkv, hd) in GEOMS {
        let kv_dim = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let d_q = e.to_device(&det(n * nh * hd, 21)).unwrap();
        let d_sinks = e.to_device(&det(nh, 22)).unwrap();
        for dtype in DTYPES {
            let [dk, dv, pk, pv] = kv_planes(&lay, kv_dim, dtype, 23);
            let (d_dk, d_dv) = (e.to_device_u8(&dk).unwrap(), e.to_device_u8(&dv).unwrap());
            let (d_pk, d_pv) = (e.to_device_u8(&pk).unwrap(), e.to_device_u8(&pv).unwrap());
            for swa in [0usize, 64] {
                let fresh = || e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap();
                let tag = |arm: &str| format!("{arm} hd{hd} {dtype:?} swa{swa}");
                // fmha (slot 537 vs 689)
                let (mut d_out, mut p_out) = (fresh(), fresh());
                e.attn_decode_fmha(
                    &d_q,
                    &d_dk,
                    &d_dv,
                    &d_sinks,
                    &mut d_out,
                    &d_pos,
                    Some(&d_slot),
                    nh,
                    nkv,
                    hd,
                    lay.max_ctx,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                e.attn_decode_fmha_paged(
                    &d_q,
                    &d_pk,
                    &d_pv,
                    &d_sinks,
                    &mut p_out,
                    &d_pos,
                    Some(&d_slot),
                    &d_bt,
                    lay.bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                assert_same(
                    &e.to_host(&d_out).unwrap(),
                    &e.to_host(&p_out).unwrap(),
                    &tag("fmha"),
                );
                // split fmha (545 vs 690)
                for split in [3usize, 7] {
                    let (mut d_out, mut p_out) = (fresh(), fresh());
                    let part_n = n * nh * split * (hd + 2);
                    let mut d_part = e.to_device(&vec![f32::NAN; part_n]).unwrap();
                    let mut p_part = e.to_device(&vec![f32::NAN; part_n]).unwrap();
                    e.attn_decode_fmha_sp(
                        &d_q,
                        &d_dk,
                        &d_dv,
                        &d_sinks,
                        &mut d_out,
                        &mut d_part,
                        &d_pos,
                        Some(&d_slot),
                        nh,
                        nkv,
                        hd,
                        lay.max_ctx,
                        kv_dim,
                        swa,
                        n,
                        split,
                        scale,
                        dtype,
                    )
                    .unwrap();
                    e.attn_decode_fmha_sp_paged(
                        &d_q,
                        &d_pk,
                        &d_pv,
                        &d_sinks,
                        &mut p_out,
                        &mut p_part,
                        &d_pos,
                        Some(&d_slot),
                        &d_bt,
                        lay.bps,
                        nh,
                        nkv,
                        hd,
                        kv_dim,
                        swa,
                        n,
                        split,
                        scale,
                        dtype,
                    )
                    .unwrap();
                    let what = tag(&format!("fmha_sp{split}"));
                    assert_same(
                        &e.to_host(&d_out).unwrap(),
                        &e.to_host(&p_out).unwrap(),
                        &what,
                    );
                }
                // parallel-score tile walk (536 vs 691)
                let (mut d_out, mut p_out) = (fresh(), fresh());
                e.attn_decode_batch_ps(
                    &d_q,
                    &d_dk,
                    &d_dv,
                    &d_sinks,
                    &mut d_out,
                    &d_pos,
                    Some(&d_slot),
                    nh,
                    nkv,
                    hd,
                    lay.max_ctx,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                e.attn_decode_batch_ps_paged(
                    &d_q,
                    &d_pk,
                    &d_pv,
                    &d_sinks,
                    &mut p_out,
                    &d_pos,
                    Some(&d_slot),
                    &d_bt,
                    lay.bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                assert_same(
                    &e.to_host(&d_out).unwrap(),
                    &e.to_host(&p_out).unwrap(),
                    &tag("batch_ps"),
                );
                // the serial tile walk's existing twin (the arm below ps),
                // which Flash-Next falls to once paged
                let (mut d_out, mut p_out) = (fresh(), fresh());
                e.attn_decode_batch(
                    &d_q,
                    &d_dk,
                    &d_dv,
                    &d_sinks,
                    &mut d_out,
                    &d_pos,
                    Some(&d_slot),
                    nh,
                    nkv,
                    hd,
                    lay.max_ctx,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                e.attn_decode_batch_paged(
                    &d_q,
                    &d_pk,
                    &d_pv,
                    &d_sinks,
                    &mut p_out,
                    &d_pos,
                    Some(&d_slot),
                    &d_bt,
                    lay.bps,
                    nh,
                    nkv,
                    hd,
                    kv_dim,
                    swa,
                    n,
                    scale,
                    dtype,
                )
                .unwrap();
                assert_same(
                    &e.to_host(&d_out).unwrap(),
                    &e.to_host(&p_out).unwrap(),
                    &tag("batch"),
                );
                eprintln!(
                    "{}: fmha, fmha_sp 3/7, batch_ps, batch bit-identical",
                    tag("decode")
                );
                // the control: the same launch with the slots' tables rotated
                // (every slot reads another's pages) must NOT match - the
                // table is what the paged mode addresses by
                if hd == 256 && dtype == KvDtype::Fp16 && swa == 0 {
                    let mut rot = lay.table.clone();
                    rot.rotate_left(lay.bps);
                    let d_rot = e.to_device_u32(&rot).unwrap();
                    let (mut d_out, mut p_out) = (fresh(), fresh());
                    e.attn_decode_fmha(
                        &d_q,
                        &d_dk,
                        &d_dv,
                        &d_sinks,
                        &mut d_out,
                        &d_pos,
                        Some(&d_slot),
                        nh,
                        nkv,
                        hd,
                        lay.max_ctx,
                        kv_dim,
                        swa,
                        n,
                        scale,
                        dtype,
                    )
                    .unwrap();
                    e.attn_decode_fmha_paged(
                        &d_q,
                        &d_pk,
                        &d_pv,
                        &d_sinks,
                        &mut p_out,
                        &d_pos,
                        Some(&d_slot),
                        &d_rot,
                        lay.bps,
                        nh,
                        nkv,
                        hd,
                        kv_dim,
                        swa,
                        n,
                        scale,
                        dtype,
                    )
                    .unwrap();
                    assert_ne!(
                        bits(&e.to_host(&d_out).unwrap()),
                        bits(&e.to_host(&p_out).unwrap()),
                        "a rotated table matched dense: the gate cannot see the table"
                    );
                }
            }
        }
    }
}

/// The QSA indexer's geometry: 4 heads of 128 over 4-token blocks; `k` is
/// small so 2048-token strips (512 blocks) exercise the top-k path.
const IDX_HD: usize = 128;
const CR: usize = 4;
const RPP: usize = PAGE / CR;

#[test]
fn qsa_index_paged_matches_dense() {
    let Some(e) = exec_with_modes() else { return };
    let lay = Layout::new(2048, vec![1501, 2048, 903, 64], 0xb10c);
    let cap = lay.max_ctx / CR;
    let d_bt = e.to_device_u32(&lay.table).unwrap();
    let nan = bf16::NAN;
    // the compressed caches: random rows over each slot's complete blocks,
    // NaN past them
    let mut dense_idx: Vec<bf16> = det(lay.slots * cap * IDX_HD, 31)
        .iter()
        .map(|&v| bf16::from_f32(v))
        .collect();
    for s in 0..lay.slots {
        for b in lay.live[s] / CR..cap {
            let o = (s * cap + b) * IDX_HD;
            dense_idx[o..o + IDX_HD].fill(nan);
        }
    }
    let pool_idx = lay.scatter(&dense_idx, IDX_HD, RPP, nan);

    // ---- store (663 vs 692): a run closing eight blocks, a lone closing
    // row, a row that closes none
    let rows = [(0u32, 1468u32), (2, 899), (3, 61)];
    let mut srows: Vec<(u32, u32)> = (0..32).map(|i| (0u32, 1468 + i)).collect();
    srows.extend_from_slice(&rows[1..]);
    let n = srows.len();
    let (ld, koff, ring_len) = (5 * IDX_HD, 4 * IDX_HD, 16usize);
    let d_raw = e.to_device(&det(n * ld, 32)).unwrap();
    // staged keys outside the cache's [-0.5, 0.5): every stored value shows
    let stage: Vec<f32> = det(n * IDX_HD, 33).iter().map(|v| v + 2.0).collect();
    let d_stage = e.to_device(&stage).unwrap();
    let d_pos = e
        .to_device_u32(&srows.iter().map(|r| r.1).collect::<Vec<_>>())
        .unwrap();
    let d_sl = e
        .to_device_u32(&srows.iter().map(|r| r.0).collect::<Vec<_>>())
        .unwrap();
    let ring0 = det(lay.slots * ring_len * IDX_HD, 34);
    let (mut d_ring, mut p_ring) = (e.to_device(&ring0).unwrap(), e.to_device(&ring0).unwrap());
    let mut d_idx = e.to_device_bf16(&dense_idx).unwrap();
    let mut p_idx = e.to_device_bf16(&pool_idx).unwrap();
    e.q4x_idx_store(
        &d_raw,
        &d_stage,
        &d_pos,
        &d_sl,
        &mut d_idx,
        &mut d_ring,
        n,
        IDX_HD,
        ld,
        koff,
        ring_len,
        CR,
        cap,
    )
    .unwrap();
    e.q4x_idx_store_paged(
        &d_raw,
        &d_stage,
        &d_pos,
        &d_sl,
        &mut p_idx,
        &mut p_ring,
        &d_bt,
        lay.bps,
        n,
        IDX_HD,
        ld,
        koff,
        ring_len,
        CR,
    )
    .unwrap();
    assert_eq!(
        bits(&e.to_host(&d_ring).unwrap()),
        bits(&e.to_host(&p_ring).unwrap()),
        "idx_store: rings differ"
    );
    let dense_after = e.to_host_bf16(&d_idx).unwrap();
    let paged_after = lay.gather(&e.to_host_bf16(&p_idx).unwrap(), IDX_HD, RPP, nan);
    // compare over every mapped page (unmapped pages are the gather's fill)
    for s in 0..lay.slots {
        let rows = lay.live[s].div_ceil(PAGE) * RPP;
        let o = s * cap * IDX_HD;
        let (a, b) = (
            &dense_after[o..o + rows * IDX_HD],
            &paged_after[o..o + rows * IDX_HD],
        );
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "idx_store: slot {s}'s cache differs"
        );
    }
    let wrote = dense_after
        .iter()
        .zip(&dense_idx)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        wrote,
        9 * IDX_HD,
        "idx_store wrote {wrote} values, want 9 blocks"
    );
    eprintln!("idx_store: 9 blocks + rings bit-identical");

    // ---- scores (664 / 670 vs 693 / 694): rows of every slot in one
    // launch, one selecting everything (nb <= k)
    let k = 64usize;
    let lrows = [
        (0u32, 1500u32),
        (1, 2047),
        (2, 902),
        (1, 1200),
        (3, 63),
        (0, 777),
    ];
    let n = lrows.len();
    let d_q = e.to_device(&det(n * 4 * IDX_HD, 35)).unwrap();
    let d_pos = e
        .to_device_u32(&lrows.iter().map(|r| r.1).collect::<Vec<_>>())
        .unwrap();
    let d_sl = e
        .to_device_u32(&lrows.iter().map(|r| r.0).collect::<Vec<_>>())
        .unwrap();
    let d_idx = e.to_device_bf16(&dense_idx).unwrap();
    let p_idx = e.to_device_bf16(&pool_idx).unwrap();
    let mut routes = vec![QsaRoute::Simt];
    if e.has_qsa_logits_mma() {
        routes.push(QsaRoute::Mma);
    }
    for route in routes {
        // unwritten scores keep the sentinel on both sides
        let mut d_sc = e.to_device(&vec![-7.0f32; n * cap]).unwrap();
        let mut p_sc = e.to_device(&vec![-7.0f32; n * cap]).unwrap();
        e.q4x_qsa_logits(
            route, &d_q, &d_idx, &d_pos, &d_sl, &mut d_sc, 0, n, 4, IDX_HD, cap, CR, k,
        )
        .unwrap();
        e.q4x_qsa_logits_paged(
            route, &d_q, &p_idx, &d_pos, &d_sl, &d_bt, lay.bps, &mut p_sc, 0, n, 4, IDX_HD, cap,
            CR, k,
        )
        .unwrap();
        let (a, b) = (e.to_host(&d_sc).unwrap(), e.to_host(&p_sc).unwrap());
        let what = format!("qsa_logits {route:?}");
        assert_same(&a, &b, &what);
        let scored = a.iter().filter(|&&v| v != -7.0).count();
        let want: usize = lrows
            .iter()
            .map(|r| (r.1 as usize + 1) / CR)
            .filter(|&nb| nb > k)
            .sum();
        assert_eq!(scored, want, "{what}: scored {scored}, want {want}");
        eprintln!("{what}: {scored} scores bit-identical");
    }
}

#[test]
fn qsa_attn_paged_matches_dense() {
    let Some(e) = exec_with_modes() else { return };
    let lay = Layout::new(2048, vec![1501, 2048, 903, 64], 0xa77e);
    let d_bt = e.to_device_u32(&lay.table).unwrap();
    let k = 64usize;
    // (slot, position, selected blocks): no tail, a 3-token tail, everything
    // (nb <= k), a 1-token tail
    let pick = |seed: u64, nb: usize| -> Vec<u32> {
        let mut ids: Vec<u32> = (0..nb as u32).collect();
        let mut s = seed;
        for i in (1..ids.len()).rev() {
            ids.swap(i, lcg(&mut s) as usize % (i + 1));
        }
        ids.truncate(k);
        ids.sort_unstable();
        ids
    };
    let rows: Vec<(u32, u32, Vec<u32>)> = vec![
        (0, 1499, pick(1, 375)),
        (1, 2046, pick(2, 511)),
        (3, 63, (0..16).collect()),
        (2, 900, pick(3, 225)),
        (1, 2047, pick(4, 512)),
    ];
    let n = rows.len();
    let mut sel = vec![0u32; n * k];
    let mut cnt = vec![0u32; n];
    for (r, (_, _, s)) in rows.iter().enumerate() {
        sel[r * k..r * k + s.len()].copy_from_slice(s);
        cnt[r] = s.len() as u32;
    }
    let d_pos = e
        .to_device_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>())
        .unwrap();
    let d_sl = e
        .to_device_u32(&rows.iter().map(|r| r.0).collect::<Vec<_>>())
        .unwrap();
    let d_sel = e.to_device_u32(&sel).unwrap();
    let d_cnt = e.to_device_u32(&cnt).unwrap();
    for (nh, nkv, hd) in GEOMS {
        let kv_dim = nkv * hd;
        let g = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let d_q = e.to_device(&det(n * nh * hd, 41)).unwrap();
        for dtype in DTYPES {
            let [dk, dv, pk, pv] = kv_planes(&lay, kv_dim, dtype, 43);
            let (d_dk, d_dv) = (e.to_device_u8(&dk).unwrap(), e.to_device_u8(&dv).unwrap());
            let (d_pk, d_pv) = (e.to_device_u8(&pk).unwrap(), e.to_device_u8(&pv).unwrap());
            let mut routes = vec![QsaRoute::Simt];
            if e.has_qsa_attn_mma() {
                routes.push(QsaRoute::Mma);
            }
            for route in routes {
                for splits in [1usize, 5] {
                    let part = |w: usize| {
                        e.to_device(&vec![f32::NAN; n * nkv * splits * g * w])
                            .unwrap()
                    };
                    let (mut d_po, mut d_pml) = (part(hd), part(2));
                    let (mut p_po, mut p_pml) = (part(hd), part(2));
                    e.q4x_qsa_attn(
                        route,
                        &d_q,
                        &d_dk,
                        &d_dv,
                        &d_pos,
                        &d_sl,
                        &d_sel,
                        &d_cnt,
                        &mut d_po,
                        &mut d_pml,
                        n,
                        nh,
                        nkv,
                        hd,
                        lay.max_ctx,
                        k,
                        CR,
                        splits,
                        scale,
                        dtype,
                    )
                    .unwrap();
                    e.q4x_qsa_attn_paged(
                        route, &d_q, &d_pk, &d_pv, &d_pos, &d_sl, &d_bt, lay.bps, &d_sel, &d_cnt,
                        &mut p_po, &mut p_pml, n, nh, nkv, hd, k, CR, splits, scale, dtype,
                    )
                    .unwrap();
                    let what = format!("qsa_attn {route:?} hd{hd} {dtype:?} splits{splits}");
                    // an empty split's partial is (-inf / -3e38, 0) on both sides
                    let (a, b) = (e.to_host(&d_pml).unwrap(), e.to_host(&p_pml).unwrap());
                    assert_eq!(bits(&a), bits(&b), "{what}: (m, l) partials differ");
                    let (mut d_out, mut p_out) = (
                        e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap(),
                        e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap(),
                    );
                    e.q4x_qsa_combine(&d_po, &d_pml, &mut d_out, n, nh, nkv, hd, splits)
                        .unwrap();
                    e.q4x_qsa_combine(&p_po, &p_pml, &mut p_out, n, nh, nkv, hd, splits)
                        .unwrap();
                    assert_same(
                        &e.to_host(&d_out).unwrap(),
                        &e.to_host(&p_out).unwrap(),
                        &what,
                    );
                    eprintln!("{what}: bit-identical");
                }
            }
        }
    }
}

/// The single-slot prefill Flash-Next takes today - `attn_prefill_f16` (P6i)
/// on f16 KV, the tiled `attn_prefill` otherwise - has existing paged twins;
/// at its 24 / 2 x 256 geometry they must be the same bits too (the paged
/// f16 entry's pf5 / pf7 arms take other group sizes and fall to P6i's twin
/// here). One slot's resumed chunk, every row the same slot as the
/// single-slot entries require (from pack 0.25 the f16 entry at this group
/// is the v4 arm, held to the f32 walk to the f16 class instead of a dense
/// twin). The slot is backed only to its live keys
/// (rows reaching 340): every row past them is NaN and every table entry
/// past its pages names the poison block, so a kernel that reads a stale
/// row into a product - P6i staged whole 64-key tiles and weighed the keys
/// past a row by zero, and 0 x NaN is NaN - fails here. Its two query
/// blocks end mid-page (keys to 332 and to 341), so both the straddling
/// sub-tile's zeroed rows and the skipped strips past it are exercised.
#[test]
fn single_slot_prefill_twins_match_dense() {
    let Some(e) = exec_with_modes() else { return };
    let lay = Layout::new(512, vec![341, 18, 71, 512], 0x5eed);
    let (nh, nkv, hd) = GEOMS[0];
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let n = 41usize;
    let d_pos = e.to_device_u32(&(300..341).collect::<Vec<u32>>()).unwrap();
    let d_slot = e.to_device_u32(&vec![0u32; n]).unwrap();
    let d_bt = e.to_device_u32(&lay.table).unwrap();
    let d_q = e.to_device(&det(n * nh * hd, 51)).unwrap();
    let d_sinks = e.to_device(&det(nh, 52)).unwrap();
    for dtype in DTYPES {
        let [dk, dv, pk, pv] = kv_planes(&lay, kv_dim, dtype, 53);
        let (d_dk, d_dv) = (e.to_device_u8(&dk).unwrap(), e.to_device_u8(&dv).unwrap());
        let (d_pk, d_pv) = (e.to_device_u8(&pk).unwrap(), e.to_device_u8(&pv).unwrap());
        let fresh = || e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap();
        let (mut d_out, mut p_out) = (fresh(), fresh());
        e.attn_prefill(
            &d_q,
            &d_dk,
            &d_dv,
            &d_sinks,
            &mut d_out,
            &d_pos,
            &d_slot,
            nh,
            nkv,
            hd,
            lay.max_ctx,
            kv_dim,
            0,
            n,
            scale,
            dtype,
        )
        .unwrap();
        e.attn_prefill_paged(
            &d_q, &d_pk, &d_pv, &d_sinks, &mut p_out, &d_pos, &d_slot, &d_bt, lay.bps, nh, nkv, hd,
            kv_dim, 0, n, scale, dtype,
        )
        .unwrap();
        let what = format!("attn_prefill {dtype:?}");
        assert_same(
            &e.to_host(&d_out).unwrap(),
            &e.to_host(&p_out).unwrap(),
            &what,
        );
        let tiled = e.to_host(&d_out).unwrap();
        // From pack 0.25 the paged tensor-core entry takes the v4 arm at
        // G = 12 (f16 and e4m3; O in f32) - no dense twin: it is held to the
        // f32 tiled walk above, to the f16 class, NaN past the live keys
        if e.pack_version() >= [0, 25, 0] {
            let mut p_out = fresh();
            e.attn_prefill_f16_paged(
                &d_q, &d_pk, &d_pv, &d_sinks, &mut p_out, &d_pos, &d_slot, &d_bt, lay.bps, nh, nkv,
                hd, kv_dim, 0, n, scale, dtype,
            )
            .unwrap();
            let v4 = e.to_host(&p_out).unwrap();
            let what = format!("attn_prefill_f16_paged (v4 G12) {dtype:?}");
            if let Some(i) = v4.iter().position(|v| !v.is_finite()) {
                panic!(
                    "{what}: output [{i}] = {} (a read past the live keys?)",
                    v4[i]
                );
            }
            let big = tiled.iter().fold(0f32, |m, x| m.max(x.abs()));
            let dev = v4
                .iter()
                .zip(&tiled)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            eprintln!("{what}: max |d| vs the f32 walk {dev:.3e} (max |ref| {big:.3e})");
            assert!(dev <= 4e-3 * big.max(1.0), "{what}: {dev} off the f32 walk");
        }
        // P6i is the f16-pool arm (its dense entry refuses e4m3, as the paged
        // one does); from 0.25 its paged twin is reached only at other groups
        if dtype == KvDtype::Fp16 && e.pack_version() < [0, 25, 0] {
            let (mut d_out, mut p_out) = (fresh(), fresh());
            e.attn_prefill_f16(
                &d_q,
                &d_dk,
                &d_dv,
                &d_sinks,
                &mut d_out,
                &d_pos,
                &d_slot,
                nh,
                nkv,
                hd,
                lay.max_ctx,
                kv_dim,
                0,
                n,
                scale,
                dtype,
            )
            .unwrap();
            e.attn_prefill_f16_paged(
                &d_q, &d_pk, &d_pv, &d_sinks, &mut p_out, &d_pos, &d_slot, &d_bt, lay.bps, nh, nkv,
                hd, kv_dim, 0, n, scale, dtype,
            )
            .unwrap();
            let what = "attn_prefill_f16 (P6i) Fp16";
            assert_same(
                &e.to_host(&d_out).unwrap(),
                &e.to_host(&p_out).unwrap(),
                what,
            );
        }
        eprintln!("single-slot prefill twins {dtype:?}: bit-identical");
    }
}

/// The batched-runs launch of the tensor-core prefill at Flash-Next's
/// geometry (24 / 2 x 256: the paged dispatcher's v4 arm at G = 12, from
/// pack 0.25). Four runs over four slots - a resumed chunk, a fresh prompt, a
/// verify-sized 4-row chunk and a slot's last rows - in ONE launch behind a
/// registered run table must write every row exactly as a launch of that
/// run alone does (`attn_prefill_f16_paged_at`, bit for bit), and stay
/// inside the f16 class of the f32 SIMT walk it replaces
/// (`attn_prefill_batch_paged`). Rows past each slot's live keys are NaN, so
/// a run that attended another run's slot, or a row read past its keys,
/// shows. f16 and e4m3 pools.
#[test]
fn prefill_runs_tensor_core_matches_single_runs() {
    let Some(e) = exec_with_modes() else { return };
    if e.pack_version() < [0, 25, 0]
        || !e.kernels_pf_runs_available()
        || !e.has_attn_prefill_f16_paged()
    {
        common::missing("a pack >= 0.25 with the batched-runs tensor-core prefill");
        return;
    }
    let lay = attn_layout();
    // (slot, first position, rows), contiguous in walk rows
    let runs = [
        (0usize, 300usize, 41usize),
        (2, 0, 71),
        (1, 14, 4),
        (3, 500, 12),
    ];
    let (mut pos, mut slot, mut offs, mut t_row0, mut t_slot) =
        (vec![], vec![], vec![0u32], vec![], vec![]);
    for &(s, p0, n) in &runs {
        let r0 = pos.len() as u32;
        for i in 0..n {
            pos.push((p0 + i) as u32);
            slot.push(s as u32);
        }
        for t in 0..n.div_ceil(16) {
            t_row0.push(r0 + 16 * t as u32);
            t_slot.push(s as u32);
        }
        offs.push(pos.len() as u32);
    }
    let n = pos.len();
    let maxn = runs.iter().map(|r| r.2).max().unwrap();
    let d_pos = e.to_device_u32(&pos).unwrap();
    let d_slot = e.to_device_u32(&slot).unwrap();
    let d_offs = e.to_device_u32(&offs).unwrap();
    let d_r0 = e.to_device_u32(&t_row0).unwrap();
    let d_ts = e.to_device_u32(&t_slot).unwrap();
    let d_bt = e.to_device_u32(&lay.table).unwrap();
    let (nh, nkv, hd) = GEOMS[0];
    let kv_dim = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let d_q = e.to_device(&det(n * nh * hd, 61)).unwrap();
    let d_sinks = e.to_device(&det(nh, 62)).unwrap();
    for dtype in DTYPES {
        let [_, _, pk, pv] = kv_planes(&lay, kv_dim, dtype, 63);
        let (d_pk, d_pv) = (e.to_device_u8(&pk).unwrap(), e.to_device_u8(&pv).unwrap());
        let fresh = || e.to_device(&vec![f32::NAN; n * nh * hd]).unwrap();
        let (mut a_out, mut b_out, mut c_out) = (fresh(), fresh(), fresh());
        // one launch, every run
        e.pf_runs_register(Some((&d_offs, runs.len() as u32, maxn as u32)))
            .unwrap();
        let walked = e.attn_prefill_f16_paged(
            &d_q, &d_pk, &d_pv, &d_sinks, &mut a_out, &d_pos, &d_slot, &d_bt, lay.bps, nh, nkv, hd,
            kv_dim, 0, n, scale, dtype,
        );
        e.pf_runs_register(None).unwrap();
        walked.unwrap();
        // each run alone, in place at its rows
        for (i, r) in runs.iter().enumerate() {
            e.attn_prefill_f16_paged_at(
                &d_q,
                &d_pk,
                &d_pv,
                &d_sinks,
                &mut b_out,
                &d_pos,
                &d_slot,
                offs[i] as usize,
                &d_bt,
                lay.bps,
                nh,
                nkv,
                hd,
                kv_dim,
                0,
                r.2,
                scale,
                dtype,
            )
            .unwrap();
        }
        let (a, b) = (e.to_host(&a_out).unwrap(), e.to_host(&b_out).unwrap());
        let what = format!("prefill runs (v4 G12) {dtype:?}");
        assert_same(&b, &a, &what);
        // the f32 SIMT walk it replaces: the same values to the f16 class
        e.attn_prefill_batch_paged(
            &d_q,
            &d_pk,
            &d_pv,
            &d_sinks,
            &mut c_out,
            &d_pos,
            &d_slot,
            &d_bt,
            lay.bps,
            &d_r0,
            &d_ts,
            t_row0.len(),
            nh,
            nkv,
            hd,
            kv_dim,
            0,
            n,
            scale,
            dtype,
        )
        .unwrap();
        let c = e.to_host(&c_out).unwrap();
        let big = c.iter().fold(0f32, |m, x| m.max(x.abs()));
        let dev = a
            .iter()
            .zip(&c)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        eprintln!(
            "{what}: {n} rows in one launch bit-identical to the runs alone; max |d| vs the f32 \
             walk {dev:.3e} (max |ref| {big:.3e})"
        );
        assert!(
            dev <= 4e-3 * big.max(1.0),
            "{what}: {dev} off the f32 walk (max |ref| {big})"
        );
    }
}
