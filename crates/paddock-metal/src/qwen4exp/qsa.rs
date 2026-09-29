//! Text QSA subgraph, not a Generator. Completed four-token indexer blocks
//! are immutable; the raw F32 carry has only four rows. Sparse attention
//! reads selected paged KV directly, never a context-sized dense mask.
use super::residual::{EPS, WIDTH, load_weight};
use super::{affine, mlx};
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::mapped::MappedGguf;

pub(super) const HEADS: usize = 24;
pub(super) const HD: usize = 256;
pub(super) const Q: usize = HEADS * HD;
pub(super) const KV: usize = 512;
pub(super) const BUDGET: usize = 512;
const SPLITS: usize = 4;

pub(super) struct Weights {
    #[cfg(test)]
    layer: usize,
    q: Weight,
    k: Weight,
    v: Weight,
    out: Weight,
    q_norm: Weight,
    k_norm: Weight,
    index_q: Weight,
    index_k: Option<Weight>,
    index_q_norm: Weight,
    index_k_norm: Weight,
}
impl Weights {
    pub(super) fn load_mlx(d: &MetalDevice, s: &mlx::Source, layer: usize) -> Result<Self> {
        if layer >= 48 || layer % 4 != 3 {
            return Err(MetalError::Model("not a Flash Next QSA layer".into()));
        }
        let root = format!("{}.layers.{layer}.self_attn", mlx::ROOT);
        let w = |suffix: &str| s.weight(d, &format!("{root}.{suffix}"));
        let index = format!("{root}.indexer.index_qk_proj");
        Ok(Self {
            #[cfg(test)]
            layer,
            q: w("q_proj")?,
            k: w("k_proj")?,
            v: w("v_proj")?,
            out: w("o_proj")?,
            q_norm: w("q_norm.weight")?,
            k_norm: w("k_norm.weight")?,
            index_q: s.weight(d, &index)?,
            index_k: None,
            index_q_norm: w("indexer.q_layernorm.weight")?,
            index_k_norm: w("indexer.k_layernorm.weight")?,
        })
    }
    pub(super) fn load(d: &MetalDevice, map: &MappedGguf, layer: usize) -> Result<Self> {
        if layer >= 48 || layer % 4 != 3 {
            return Err(MetalError::Model("not a Flash Next QSA layer".into()));
        }
        let w = |suffix: &str, dims: &[usize], ty| {
            let name = format!("blk.{layer}.{suffix}");
            let (info, bytes) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if info.raw_type != ty
                || (ty == 0
                    && !bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|b| f32::from_le_bytes(*b).is_finite()))
            {
                return Err(MetalError::Model(format!("invalid Flash Next QSA {name}")));
            }
            load_weight(d, map, &name, dims, ty)
        };
        Ok(Self {
            #[cfg(test)]
            layer,
            q: w("attn_q.weight", &[WIDTH, Q * 2], 14)?,
            k: w("attn_k.weight", &[WIDTH, KV], 14)?,
            v: w("attn_v.weight", &[WIDTH, KV], 14)?,
            out: w("attn_output.weight", &[Q, WIDTH], 14)?,
            q_norm: w("attn_q_norm.weight", &[HD], 0)?,
            k_norm: w("attn_k_norm.weight", &[HD], 0)?,
            index_q: w("indexer.q_proj.weight", &[WIDTH, 512], 30)?,
            index_k: Some(w("indexer.k_proj.weight", &[WIDTH, 128], 30)?),
            index_q_norm: w("indexer.q_norm.weight", &[128], 0)?,
            index_k_norm: w("indexer.k_norm.weight", &[128], 0)?,
        })
    }
}
fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    if affine::is_affine(w.ty) {
        return affine::project(cmd, w, x, y, rows);
    }
    if w.ty == 14 {
        crate::projection::project(cmd, &[(w, y)], x, rows);
    } else {
        let (name, c, r) = if rows <= 8 {
            ("linear", 4, 1)
        } else {
            ("q4s_bf16_mm", 32, 32)
        };
        cmd.dispatch(
            name,
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, rows as u32, 30, 1f32.to_bits()],
            [w.n.div_ceil(c), rows.div_ceil(r), 1],
            128,
        );
    }
}

#[derive(Debug)]
pub(super) struct Plan {
    meta: Vec<u32>,
    lengths: Vec<usize>,
    blocks: usize,
}
impl Plan {
    pub(super) fn new(
        rows: &[(usize, usize)],
        lengths: &[usize],
        capacity: usize,
        context: usize,
    ) -> Result<Self> {
        if rows.is_empty() || rows.len() > capacity {
            return Err(MetalError::Model("QSA empty/oversized batch".into()));
        }
        let mut p = Self {
            meta: vec![],
            lengths: lengths.to_vec(),
            blocks: 0,
        };
        let mut seen = vec![false; lengths.len()];
        let mut first = 0;
        while first < rows.len() {
            let slot = rows[first].0;
            if slot >= lengths.len() || seen[slot] {
                return Err(MetalError::Model("QSA invalid/repeated slot span".into()));
            }
            seen[slot] = true;
            let end = rows[first..]
                .iter()
                .position(|r| r.0 != slot)
                .map_or(rows.len(), |i| first + i);
            for &(s, pos) in &rows[first..end] {
                if pos != p.lengths[s] || pos >= context {
                    return Err(MetalError::Model("QSA invalid position/context".into()));
                }
                p.lengths[s] += 1;
                p.blocks = p.blocks.max((pos + 1) / 4);
                p.meta
                    .extend([s as u32, pos as u32, first as u32, end as u32]);
            }
            first = end;
        }
        Ok(p)
    }
}

/// Persistent QSA data only. KV is physically paged; pooled indexer keys
/// and the four-row carry are logically addressed per slot.
pub(super) struct Cache {
    pub(super) keys: Buffer,
    pub(super) values: Buffer,
    pub(super) pooled: Buffer,
    pub(super) ring: Buffer,
}
impl Cache {
    pub(super) fn bytes(slots: usize, pages: usize) -> usize {
        slots * pages * 16 * KV * 2 * 2 + slots * pages * 4 * 128 * 4 + slots * 512 * 4
    }
    pub(super) fn new(d: &MetalDevice, slots: usize, pages: usize) -> Result<Self> {
        Ok(Self {
            keys: d.alloc(slots * pages * 16 * KV * 2)?,
            values: d.alloc(slots * pages * 16 * KV * 2)?,
            pooled: d.alloc(slots * pages * 4 * 128 * 4)?,
            ring: d.alloc(slots * 512 * 4)?,
        })
    }
    // No stale KV can be read after a reset: lengths are zero and every
    // causally visible token is written before attention. Clear logical carry.
    pub(super) fn reset(&self, cmd: &Commands<'_>, slot: usize, pages: usize) {
        for (b, n) in [(&self.ring, 512), (&self.pooled, pages * 4 * 128)] {
            cmd.dispatch(
                "nemo_state_copy",
                &[b],
                &[n as u32, slot as u32, u32::MAX],
                [n.div_ceil(256), 1, 1],
                256,
            );
        }
    }
    #[cfg(test)]
    fn copy(&self, cmd: &Commands<'_>, pages: &Buffer, p: &[u32]) {
        cmd.dispatch(
            "q4s_copy",
            &[&self.keys, &self.values, &self.pooled, &self.ring, pages],
            p,
            [((p[7] as usize * KV).max(512)).div_ceil(256), 1, 1],
            256,
        );
    }
}
pub(super) struct Workspace {
    compact_capacity: usize,
    slots: usize,
    pages_per_slot: usize,
    blocks: usize,
    pub(super) pages: Buffer,
    meta: Buffer,
    qg: Buffer,
    k: Buffer,
    v: Buffer,
    pub(super) query: Buffer,
    pub(super) index_query: Buffer,
    pub(super) raw: Buffer,
    pub(super) scores: Buffer,
    pub(super) selected: Buffer,
    pub(super) counts: Buffer,
    pub(super) attn: Buffer,
    scratch: Buffer,
    parts: Buffer,
}
#[cfg(test)]
thread_local! {
    pub(super) static LOCAL_ATTENTION_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static PADDED_ATTENTION_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static GATHER_ATTENTION_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static VALUE_PITCH_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static DIRECT_RUNS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
impl Workspace {
    fn partition_units(capacity: usize, slots: usize, mlx: bool) -> usize {
        if mlx && capacity > slots * 8 {
            // One partition for every row; only <=8-row logical spans need
            // three more. Slot-local tails avoid prefix sums or a GPU->CPU
            // dependency and preserve the existing four-part decode order.
            capacity + slots * 8 * (SPLITS - 1)
        } else {
            capacity * SPLITS
        }
    }
    pub(super) fn mlx_bytes(capacity: usize, slots: usize, pages: usize) -> usize {
        Self::bytes(capacity, slots, pages)
            - (capacity * SPLITS - Self::partition_units(capacity, slots, true))
                * (2 * 8192 + HEADS * 258)
                * 4
    }
    pub(super) fn bytes(capacity: usize, slots: usize, pages: usize) -> usize {
        slots * pages * 4
            + capacity
                * 4
                * (4 + Q * 2
                    + KV * 2
                    + Q
                    + 512
                    + 128
                    + pages * 4
                    + BUDGET
                    + 1
                    + Q
                    + 2 * SPLITS * 8192
                    + HEADS * SPLITS * 258)
    }
    pub(super) fn new(
        d: &MetalDevice,
        capacity: usize,
        slots: usize,
        pages: usize,
    ) -> Result<Self> {
        Self::new_inner(d, capacity, slots, pages, false)
    }
    pub(super) fn new_mlx(
        d: &MetalDevice,
        capacity: usize,
        slots: usize,
        pages: usize,
    ) -> Result<Self> {
        Self::new_inner(d, capacity, slots, pages, true)
    }
    fn new_inner(
        d: &MetalDevice,
        capacity: usize,
        slots: usize,
        pages: usize,
        mlx: bool,
    ) -> Result<Self> {
        let blocks = pages * 4;
        let units = Self::partition_units(capacity, slots, mlx);
        // Diagnostic owners use this nonidentity mapping. The model replaces
        // it with its engine pool's block tables before each whole walk.
        let table: Vec<u32> = (0..slots * pages).rev().map(|i| i as u32).collect();
        Ok(Self {
            compact_capacity: if mlx && capacity > slots * 8 {
                capacity
            } else {
                0
            },
            slots,
            pages_per_slot: pages,
            blocks,
            pages: d.upload(
                &table
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            meta: d.alloc(capacity * 4 * 4)?,
            qg: d.alloc(capacity * Q * 2 * 4)?,
            k: d.alloc(capacity * KV * 4)?,
            v: d.alloc(capacity * KV * 4)?,
            query: d.alloc(capacity * Q * 4)?,
            index_query: d.alloc(capacity * 512 * 4)?,
            raw: d.alloc(capacity * 128 * 4)?,
            scores: d.alloc(capacity * blocks * 4)?,
            selected: d.alloc(capacity * BUDGET * 4)?,
            counts: d.alloc(capacity * 4)?,
            attn: d.alloc(capacity * Q * 4)?,
            scratch: d.alloc(units * 2 * 8192 * 4)?,
            parts: d.alloc(units * HEADS * 258 * 4)?,
        })
    }
    fn p(&self, rows: usize) -> [u32; 5] {
        [
            self.pages_per_slot as u32,
            self.blocks as u32,
            rows as u32,
            self.slots as u32,
            if rows <= 8 { 4 } else { 1 },
        ]
    }
    pub(super) fn stage(&self, plan: &Plan) {
        unsafe {
            self.meta.write_u32(&plan.meta);
        }
    }
    pub(super) fn encode(
        &self,
        cmd: &Commands<'_>,
        w: &Weights,
        cache: &Cache,
        plan: &Plan,
        x: &Buffer,
        out: &Buffer,
    ) {
        self.encode_inner(cmd, w, cache, plan, x, Some(out));
    }

    /// Populate exactly the same persistent state without evaluating dead
    /// queries or attention outputs. Only the terminal MLX layer may use this,
    /// and only when no caller (including a draft head) consumes its hidden rows.
    pub(super) fn encode_cache(
        &self,
        cmd: &Commands<'_>,
        w: &Weights,
        cache: &Cache,
        plan: &Plan,
        x: &Buffer,
        bad: &Buffer,
    ) {
        assert!(affine::is_affine(w.q.ty));
        self.encode_inner(cmd, w, cache, plan, x, None);
        let n = plan.meta.len() / 4;
        cmd.dispatch(
            "q4b_cache_finite",
            &[
                &cache.keys,
                &cache.values,
                &self.raw,
                &cache.pooled,
                &self.meta,
                &self.pages,
                bad,
            ],
            &self.p(n),
            [(n * KV).div_ceil(256), 1, 1],
            256,
        );
    }

    fn encode_inner(
        &self,
        cmd: &Commands<'_>,
        w: &Weights,
        cache: &Cache,
        plan: &Plan,
        x: &Buffer,
        out: Option<&Buffer>,
    ) {
        let n = plan.meta.len() / 4;
        let mlx = affine::is_affine(w.q.ty);
        let mut p = self.p(n);
        if mlx {
            // Storage/dispatch stride stays fixed. Each MLX sequence elects
            // its own split count on GPU; neighbours cannot change rounding.
            p[4] = SPLITS as u32;
        }
        for (needed, w, y) in [
            (out.is_some(), &w.q, &self.qg),
            (true, &w.k, &self.k),
            (true, &w.v, &self.v),
        ] {
            if needed {
                project(cmd, w, x, y, n);
            }
        }
        if let Some(index_k) = &w.index_k {
            if out.is_some() {
                project(cmd, &w.index_q, x, &self.index_query, n);
            }
            project(cmd, index_k, x, &self.raw, n);
        } else {
            // The checkpoint fuses Q/K index rows. Splitting its compressed
            // weights would change shape-dependent prefill reductions.
            // Reuse dead attention scratch for the fused projection result.
            project(cmd, &w.index_q, x, &self.scratch, n);
            cmd.dispatch(
                "q4b_index_split",
                &[&self.scratch, &self.index_query, &self.raw],
                &[n as u32],
                [(n * 640).div_ceil(256), 1, 1],
                256,
            );
        }
        for (needed, x, w, y, width, heads, stride) in [
            (
                out.is_some(),
                &self.qg,
                &w.q_norm,
                &self.query,
                256,
                24,
                512,
            ),
            (true, &self.k, &w.k_norm, &self.k, 256, 2, 256),
            (
                out.is_some(),
                &self.index_query,
                &w.index_q_norm,
                &self.index_query,
                128,
                4,
                128,
            ),
        ] {
            if !needed {
                continue;
            }
            cmd.dispatch(
                if mlx {
                    "q4b_norm_rope"
                } else {
                    "q4s_norm_rope"
                },
                &[x, &w.buffer, &self.meta, y],
                &[width, heads, stride, heads * stride, EPS.to_bits()],
                [heads as usize, n, 1],
                32,
            );
        }
        cmd.dispatch(
            if mlx { "q4b_store" } else { "q4s_store" },
            &[
                &self.k,
                &self.v,
                &self.meta,
                &self.pages,
                &cache.keys,
                &cache.values,
            ],
            &p,
            [(n * KV).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            if mlx { "q4b_pool" } else { "q4s_pool" },
            &[
                &self.raw,
                &cache.ring,
                &w.index_k_norm.buffer,
                &self.meta,
                &cache.pooled,
            ],
            &p,
            [n, 1, 1],
            32,
        );
        cmd.dispatch(
            "q4s_ring_commit",
            &[&self.raw, &cache.ring, &self.meta],
            &p,
            [(n * 128).div_ceil(256), 1, 1],
            256,
        );
        let Some(out) = out else { return };
        if plan.blocks > 512 {
            cmd.dispatch(
                "q4s_score",
                &[&self.index_query, &cache.pooled, &self.meta, &self.scores],
                &p,
                [plan.blocks.div_ceil(32), n, 1],
                128,
            );
        }
        cmd.dispatch(
            "q4s_select",
            &[&self.scores, &self.meta, &self.selected, &self.counts],
            &p,
            [n, 1, 1],
            256,
        );
        let mut attention_params = [
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            0,
            0,
            self.compact_capacity as u32,
        ];
        let logical_rows = if mlx { cmd.projection_rows() } else { None };
        if let Some(logical_rows) = logical_rows {
            // The mask is bounded host scheduling metadata, not GPU state.
            // Preserve serial QSA partitioning when a logical prefill chunk
            // is sliced down to a few physical rows by the shared scheduler.
            let mut mask = [0u32; 2];
            for &(start, _, count) in logical_rows {
                let slot = plan.meta[start * 4] as usize;
                if count <= 8 {
                    mask[slot / 32] |= 1 << (slot % 32);
                }
            }
            attention_params[5..7].copy_from_slice(&mask);
        } else if self.compact_capacity != 0 {
            for row in plan.meta.chunks_exact(4) {
                if row[3] - row[2] <= 8 {
                    attention_params[5 + row[0] as usize / 32] |= 1 << (row[0] % 32);
                }
            }
        }
        // Local BF16 staging is lossless for this BF16 cache. Keep F32
        // queries/probabilities/partials and the existing logical split mask.
        // Padding the two on-chip tensor pitches avoids the power-of-two
        // transposed layout. Arithmetic and logical partitioning are unchanged.
        let local_attention =
            mlx && self.compact_capacity != 0 && cmd.tensor_accelerated() && n <= 64;
        #[cfg(test)]
        let local_attention = local_attention && LOCAL_ATTENTION_FOR_TEST.with(|v| v.get());
        let padded_attention = mlx && self.compact_capacity != 0 && cmd.tensor_accelerated();
        #[cfg(test)]
        let padded_attention = padded_attention
            && PADDED_ATTENTION_FOR_TEST.with(|v| v.get())
            && LOCAL_ATTENTION_FOR_TEST.with(|v| v.get());
        // Share sparse/page addressing across each SIMD group and reuse it
        // for K/V. Wider loads retain the exact same padded BF16 tensors.
        let gather_attention = padded_attention;
        #[cfg(test)]
        let gather_attention = gather_attention && GATHER_ATTENTION_FOR_TEST.with(|v| v.get());
        let value_pitch = gather_attention;
        #[cfg(test)]
        let value_pitch = value_pitch && VALUE_PITCH_FOR_TEST.with(|v| v.get());
        #[cfg(test)]
        let direct = value_pitch && DIRECT_RUNS_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let direct = value_pitch;
        // The measured opportunity is dense prompt attention. Do not add
        // the run election to singleton decode or wholly sparse long tails.
        // A mixed call still checks physical contiguity separately per row.
        let direct = direct && n > 8 && plan.meta.chunks_exact(4).any(|m| m[1] < 2048);
        cmd.dispatch(
            if direct {
                "q4b_attention_direct_runs"
            } else if value_pitch {
                "q4b_attention_local_vpad1"
            } else if gather_attention {
                "q4b_attention_local_gather"
            } else if padded_attention {
                "q4b_attention_local_pad"
            } else if local_attention {
                "q4b_attention_local"
            } else if self.compact_capacity != 0 {
                assert!(mlx, "compact workspace is MLX-only");
                "q4b_attention_compact"
            } else if logical_rows.is_some() {
                "q4b_attention_contract"
            } else if mlx {
                "q4b_attention"
            } else {
                "q4s_attention"
            },
            &[
                &self.query,
                &cache.keys,
                &cache.values,
                &self.meta,
                &self.pages,
                &self.selected,
                &self.counts,
                &self.scratch,
                &self.parts,
            ],
            &attention_params,
            [2, n, p[4] as usize],
            128,
        );
        if self.compact_capacity != 0 {
            cmd.dispatch(
                "q4b_join_gate_compact",
                &[&self.parts, &self.qg, &self.attn, &self.meta],
                &attention_params,
                [n * HEADS, 1, 1],
                32,
            );
        } else {
            cmd.dispatch(
                if mlx {
                    "q4b_join_gate"
                } else {
                    "q4s_join_gate"
                },
                &[&self.parts, &self.qg, &self.attn],
                &p,
                [n * HEADS, 1, 1],
                32,
            );
        }
        project(cmd, &w.out, &self.attn, out, n);
    }
}

/// Test wrapper around the same shared workspace encoder as the full model.
#[cfg(test)]
pub(super) struct State {
    layer: usize,
    capacity: usize,
    context: usize,
    pub(super) lengths: Vec<usize>,
    poisoned: bool,
    pub(super) cache: Cache,
    pub(super) scratch: Workspace,
}
#[cfg(test)]
impl State {
    pub(super) fn bytes(capacity: usize, slots: usize, context: usize) -> Result<usize> {
        if !(1..=128).contains(&capacity)
            || !(1..=64).contains(&slots)
            || !(1..=262144).contains(&context)
        {
            return Err(MetalError::Model("QSA invalid rows/slots/context".into()));
        }
        let pages = context.div_ceil(16);
        Ok(Cache::bytes(slots, pages) + Workspace::bytes(capacity, slots, pages))
    }
    pub(super) fn new(
        d: &MetalDevice,
        w: &Weights,
        capacity: usize,
        slots: usize,
        context: usize,
    ) -> Result<Self> {
        Self::bytes(capacity, slots, context)?;
        let pages = context.div_ceil(16);
        let mut s = Self {
            layer: w.layer,
            capacity,
            context,
            lengths: vec![0; slots],
            poisoned: false,
            cache: Cache::new(d, slots, pages)?,
            scratch: Workspace::new(d, capacity, slots, pages)?,
        };
        for slot in 0..slots {
            s.reset(d, slot)?;
        }
        Ok(s)
    }
    fn healthy(&self) -> Result<()> {
        if self.poisoned {
            Err(MetalError::Model(
                "QSA state poisoned; reload required".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub(super) fn reset(&mut self, d: &MetalDevice, slot: usize) -> Result<()> {
        self.copy_impl(d, slot, slot, true)
    }
    pub(super) fn copy_slot(
        &mut self,
        d: &MetalDevice,
        source: usize,
        target: usize,
    ) -> Result<()> {
        self.copy_impl(d, source, target, false)
    }
    fn copy_impl(
        &mut self,
        d: &MetalDevice,
        source: usize,
        target: usize,
        reset: bool,
    ) -> Result<()> {
        self.healthy()?;
        if source >= self.lengths.len()
            || target >= self.lengths.len()
            || (!reset && source == target)
        {
            return Err(MetalError::Model("QSA invalid reset/copy slot".into()));
        }
        let len = if reset {
            self.scratch.pages_per_slot * 16
        } else {
            self.lengths[source]
        };
        let mut p = self.scratch.p(0).to_vec();
        p.extend([source as u32, target as u32, len as u32, reset as u32]);
        self.poisoned = true;
        let cmd = d.begin()?;
        self.cache.copy(&cmd, &self.scratch.pages, &p);
        cmd.submit()?.wait()?;
        self.lengths[target] = if reset { 0 } else { self.lengths[source] };
        self.poisoned = false;
        Ok(())
    }
    pub(super) fn run(
        &mut self,
        d: &MetalDevice,
        w: &Weights,
        rows: &[(usize, usize)],
        x: &Buffer,
        out: &Buffer,
    ) -> Result<()> {
        self.healthy()?;
        let plan = Plan::new(rows, &self.lengths, self.capacity, self.context)?;
        if self.layer != w.layer
            || x.len() < rows.len() * WIDTH * 4
            || out.len() < rows.len() * WIDTH * 4
        {
            return Err(MetalError::Model("QSA invalid layer/input/output".into()));
        }
        self.scratch.stage(&plan);
        self.poisoned = true;
        let cmd = d.begin()?;
        self.scratch.encode(&cmd, w, &self.cache, &plan, x, out);
        cmd.submit()?.wait()?;
        // This is GPU validation status, never host scoring/selection.
        if unsafe { self.scratch.counts.read_u32(rows.len()) }
            .iter()
            .any(|&n| n > 512)
        {
            return Err(MetalError::Model(
                "QSA nonfinite indexer score; state poisoned".into(),
            ));
        }
        self.lengths = plan.lengths;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_plan_rejects_bad_later_rows_and_partial_slot_reentry() {
        let lengths = [0, 2047, 0, 0];
        let p = Plan::new(&[(1, 2047), (1, 2048), (2, 0)], &lengths, 8, 4096).unwrap();
        assert_eq!(p.lengths, [0, 2049, 1, 0]);
        assert_eq!(p.blocks, 512);
        assert_eq!(p.meta, [1, 2047, 0, 2, 1, 2048, 0, 2, 2, 0, 2, 3]);
        for rows in [
            vec![],
            vec![(4, 0)],
            vec![(0, 0), (1, 2048)],
            vec![(0, 0), (1, 2047), (0, 1)],
            vec![(0, 0), (0, usize::MAX)],
        ] {
            assert!(Plan::new(&rows, &lengths, 128, 4096).is_err());
        }
        assert!(Plan::new(&[(0, 0), (0, 1)], &lengths, 1, 4096).is_err());
        assert!(Plan::new(&[(1, 2047)], &lengths, 1, 2047).is_err());
        for (r, s, c) in [
            (0, 1, 1),
            (129, 1, 1),
            (1, 0, 1),
            (1, 65, 1),
            (1, 1, 0),
            (1, 1, 262145),
        ] {
            assert!(State::bytes(r, s, c).is_err());
        }
    }
}
