//! Flash Next's GGUF DeltaNet: tiled value heads, sigmoid output gate, F32
//! state and chunk operands. Reuse our WY/TensorOps prefill and register
//! decode kernels; never widen heads or dequantize whole weight planes.
use super::residual::{self, EPS, WIDTH, load_weight};
use super::{affine, mlx};
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::mapped::MappedGguf;

pub(super) const KH: usize = 16;
pub(super) const VH: usize = 48;
pub(super) const VD: usize = VH * 128;
pub(super) const CONV: usize = 2 * KH * 128 + VD;
pub(super) const CELLS: usize = VH * 128 * 128;
const PREPARED: usize = VH * 17408;
#[cfg(test)]
thread_local! {
    pub(super) static BASELINE_RECURRENT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) struct Weights {
    #[cfg(test)]
    layer: usize,
    qkv: Weight,
    z: Weight,
    alpha: Weight,
    beta: Weight,
    conv: Weight,
    a: Weight,
    dt: Weight,
    norm: Weight,
    out: Weight,
}
impl Weights {
    pub(super) fn load_mlx(d: &MetalDevice, s: &mlx::Source, layer: usize) -> Result<Self> {
        if layer >= 48 || layer % 4 == 3 {
            return Err(MetalError::Model("not a Flash Next recurrent layer".into()));
        }
        let w = |suffix: &str| {
            s.weight(
                d,
                &format!("{}.layers.{layer}.linear_attn.{suffix}", mlx::ROOT),
            )
        };
        Ok(Self {
            #[cfg(test)]
            layer,
            qkv: w("in_proj_qkv")?,
            z: w("in_proj_z")?,
            alpha: w("in_proj_a")?,
            beta: w("in_proj_b")?,
            conv: w("conv1d.weight")?,
            a: w("A_log")?,
            dt: w("dt_bias")?,
            norm: w("norm.weight")?,
            out: w("out_proj")?,
        })
    }
    pub(super) fn load(d: &MetalDevice, map: &MappedGguf, layer: usize) -> Result<Self> {
        if layer >= 48 || layer % 4 == 3 {
            return Err(MetalError::Model("not a Flash Next recurrent layer".into()));
        }
        let w = |suffix: &str, dims: &[usize], types: &[u32]| {
            let name = format!("blk.{layer}.{suffix}");
            let (info, bytes) = map
                .tensor_bytes(&name)
                .map_err(|e| MetalError::Model(e.to_string()))?;
            if !types.contains(&info.raw_type)
                || (info.raw_type == 0
                    && !bytes.as_chunks::<4>().0.iter().all(|b| {
                        let x = f32::from_le_bytes(*b);
                        x.is_finite() && (suffix != "ssm_a" || x < 0.)
                    }))
            {
                return Err(MetalError::Model(format!(
                    "invalid Flash Next DeltaNet {name}"
                )));
            }
            load_weight(d, map, &name, dims, info.raw_type)
        };
        Ok(Self {
            #[cfg(test)]
            layer,
            qkv: w("attn_qkv.weight", &[WIDTH, CONV], &[8, 14])?,
            z: w("attn_gate.weight", &[WIDTH, VD], &[8, 14])?,
            alpha: w("ssm_alpha.weight", &[WIDTH, VH], &[0])?,
            beta: w("ssm_beta.weight", &[WIDTH, VH], &[0])?,
            conv: w("ssm_conv1d.weight", &[4, CONV], &[0])?,
            a: w("ssm_a", &[VH], &[0])?,
            dt: w("ssm_dt.bias", &[VH], &[0])?,
            norm: w("ssm_norm.weight", &[128], &[0])?,
            out: w("ssm_out.weight", &[VD, WIDTH], &[14])?,
        })
    }
}

fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, n: usize) {
    if w.ty == 14 {
        crate::projection::project(cmd, &[(w, y)], x, n);
    } else {
        residual::project(cmd, w, x, y, n);
    }
}

#[derive(Debug)]
pub(super) struct Plan {
    meta: Vec<u32>,
    bounds: Vec<u32>,
    spans: Vec<u32>,
    chunks: Vec<u32>,
    lengths: Vec<usize>,
    short: bool,
}
impl Plan {
    pub(super) fn new(
        rows: &[(usize, usize)],
        lengths: &[usize],
        capacity: usize,
        context: usize,
    ) -> Result<Self> {
        if rows.is_empty() || rows.len() > capacity {
            return Err(MetalError::Model(
                "Flash Next DeltaNet empty/oversized batch".into(),
            ));
        }
        let mut p = Self {
            meta: vec![],
            bounds: vec![],
            spans: vec![],
            chunks: vec![],
            lengths: lengths.to_vec(),
            short: false,
        };
        let mut seen = vec![false; lengths.len()];
        let mut first = 0;
        while first < rows.len() {
            let slot = rows[first].0;
            if slot >= lengths.len() || seen[slot] {
                return Err(MetalError::Model(
                    "Flash Next DeltaNet invalid/repeated slot span".into(),
                ));
            }
            seen[slot] = true;
            let end = rows[first..]
                .iter()
                .position(|r| r.0 != slot)
                .map_or(rows.len(), |i| first + i);
            for &(s, pos) in &rows[first..end] {
                if pos != p.lengths[s] || pos >= context {
                    return Err(MetalError::Model(
                        "Flash Next DeltaNet invalid position/context".into(),
                    ));
                }
                p.lengths[s] += 1;
                p.meta.extend([s as u32, pos as u32]);
                p.bounds.extend([first as u32, end as u32]);
            }
            p.spans.extend([
                first as u32,
                (end - first) as u32,
                slot as u32,
                (p.chunks.len() / 4) as u32,
            ]);
            if end - first < 16 {
                p.short = true;
            } else {
                for start in (first..end).step_by(32) {
                    p.chunks
                        .extend([start as u32, (end - start).min(32) as u32, slot as u32, 0]);
                }
            }
            first = end;
        }
        Ok(p)
    }
}

/// Only persistent recurrent state belongs to a layer. All 36 layers share
/// one Workspace; no layer owns a queue or publishes sequence lengths.
pub(super) struct Cache {
    pub(super) state: Buffer,
    pub(super) history: Buffer,
}
impl Cache {
    pub(super) fn bytes(slots: usize) -> usize {
        slots * (CELLS + 3 * CONV) * 4
    }
    pub(super) fn new(d: &MetalDevice, slots: usize) -> Result<Self> {
        Ok(Self {
            state: d.alloc(slots * CELLS * 4)?,
            history: d.alloc(slots * 3 * CONV * 4)?,
        })
    }
    pub(super) fn reset(&self, cmd: &Commands<'_>, slot: usize) {
        cmd.dispatch(
            "q4x_dn_reset",
            &[&self.state, &self.history],
            &[slot as u32],
            [CELLS.div_ceil(256), 1, 1],
            256,
        );
    }
    #[cfg(test)]
    pub(super) fn copy(&self, cmd: &Commands<'_>, source: usize, target: usize, slots: usize) {
        cmd.dispatch(
            "dn_checkpoint",
            &[&self.state, &self.history],
            &[
                source as u32,
                target as u32,
                slots as u32,
                CELLS as u32,
                (3 * CONV) as u32,
                1,
            ],
            [(CELLS + 3 * CONV).div_ceil(256), 1, 1],
            256,
        );
    }
}
pub(super) struct Workspace {
    slots: usize,
    meta: Buffer,
    bounds: Buffer,
    spans: Buffer,
    chunks: Buffer,
    checkpoint_rows: Buffer,
    checkpoint_spans: Buffer,
    pub(super) qkv: Buffer,
    pub(super) convolved: Buffer,
    z: Buffer,
    alpha: Buffer,
    beta: Buffer,
    pub(super) gates: Buffer,
    pub(super) attn: Buffer,
    prepared: Option<Buffer>,
}
impl Workspace {
    pub(super) fn bytes(capacity: usize, slots: usize) -> usize {
        let chunks = (capacity / 16).max(1);
        4 * (capacity * (5 + 2 * CONV + 2 * VD + 4 * VH) + slots * 8 + chunks * (4 + PREPARED))
    }
    pub(super) fn mlx_bytes(capacity: usize, slots: usize) -> usize {
        Self::bytes(capacity, slots) - (capacity / 16).max(1) * PREPARED * 4
    }
    pub(super) fn new(d: &MetalDevice, capacity: usize, slots: usize) -> Result<Self> {
        Self::new_inner(d, capacity, slots, false)
    }
    pub(super) fn new_mlx(d: &MetalDevice, capacity: usize, slots: usize) -> Result<Self> {
        Self::new_inner(d, capacity, slots, true)
    }
    fn new_inner(d: &MetalDevice, capacity: usize, slots: usize, mlx: bool) -> Result<Self> {
        let chunks = (capacity / 16).max(1);
        Ok(Self {
            slots,
            meta: d.alloc(capacity * 2 * 4)?,
            bounds: d.alloc(capacity * 2 * 4)?,
            spans: d.alloc(slots * 4 * 4)?,
            chunks: d.alloc(chunks * 4 * 4)?,
            checkpoint_rows: d.upload(&vec![0; capacity * 4])?,
            checkpoint_spans: d.upload(&vec![0; slots * 4 * 4])?,
            qkv: d.alloc(capacity * CONV * 4)?,
            convolved: d.alloc(capacity * CONV * 4)?,
            z: d.alloc(capacity * VD * 4)?,
            alpha: d.alloc(capacity * VH * 4)?,
            beta: d.alloc(capacity * VH * 4)?,
            gates: d.alloc(capacity * VH * 2 * 4)?,
            attn: d.alloc(capacity * VD * 4)?,
            // The MLX exact recurrent walk never uses GGUF's WY operands.
            // Do not reserve up to 204 MiB of unreachable workspace for it.
            prepared: if mlx {
                None
            } else {
                Some(d.alloc(chunks * PREPARED * 4)?)
            },
        })
    }
    /// Caller has waited for the previous whole model submission.
    pub(super) fn stage(&self, plan: &Plan) {
        unsafe {
            self.meta.write_u32(&plan.meta);
            self.bounds.write_u32(&plan.bounds);
            self.spans.write_u32(&plan.spans);
            self.chunks.write_u32(&plan.chunks);
        }
    }
    pub(super) fn encode(
        &self,
        cmd: &Commands<'_>,
        w: &Weights,
        cache: &Cache,
        plan: &Plan,
        x: &Buffer,
        output: &Buffer,
    ) {
        let n = plan.meta.len() / 2;
        if affine::is_affine(w.qkv.ty) {
            self.encode_mlx(cmd, w, cache, plan, x, output);
            return;
        }
        let spans = plan.spans.len() / 4;
        let chunks = plan.chunks.len() / 4;
        let p = [
            KH as u32,
            VH as u32,
            CONV as u32,
            n as u32,
            self.slots as u32,
            0,
            0,
        ];
        for (w, y) in [
            (&w.qkv, &self.qkv),
            (&w.z, &self.z),
            (&w.alpha, &self.alpha),
            (&w.beta, &self.beta),
        ] {
            project(cmd, w, x, y, n);
        }
        cmd.dispatch(
            "dn_conv",
            &[
                &self.qkv,
                &w.conv.buffer,
                &cache.history,
                &self.meta,
                &self.bounds,
                &self.convolved,
            ],
            &p,
            [(n * CONV).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch("dn_qk_norm", &[&self.convolved], &p, [KH * 2, n, 1], 32);
        cmd.dispatch(
            "dn_gates",
            &[
                &self.alpha,
                &self.beta,
                &w.a.buffer,
                &w.dt.buffer,
                &self.gates,
            ],
            &p,
            [(n * VH).div_ceil(256), 1, 1],
            256,
        );
        if plan.short {
            cmd.dispatch(
                "dn_recurrent",
                &[
                    &self.convolved,
                    &self.gates,
                    &cache.state,
                    &self.spans,
                    &self.meta,
                    &self.attn,
                    &self.checkpoint_rows,
                ],
                &p,
                [8, VH, spans],
                128,
            );
        }
        if chunks > 0 {
            let prepared = self.prepared.as_ref().expect("GGUF WY workspace");
            for kernel in ["dn_chunk_dots_strict", "dn_chunk_prepare_strict"] {
                cmd.dispatch(
                    kernel,
                    &[&self.convolved, &self.gates, &self.chunks, prepared],
                    &p,
                    [VH, chunks, 1],
                    128,
                );
            }
            cmd.dispatch(
                "dn_chunk_walk_strict",
                &[
                    prepared,
                    &self.gates,
                    &self.spans,
                    &self.chunks,
                    &self.meta,
                    &cache.state,
                    &self.attn,
                ],
                &p,
                [8, VH, spans],
                128,
            );
        }
        cmd.dispatch(
            "q4x_dn_gated_norm",
            &[&self.attn, &self.z, &w.norm.buffer],
            &[EPS.to_bits()],
            [VH, n, 1],
            32,
        );
        project(cmd, &w.out, &self.attn, output, n);
        cmd.dispatch(
            "dn_conv_commit",
            &[
                &self.qkv,
                &cache.history,
                &self.spans,
                &self.meta,
                &self.checkpoint_spans,
            ],
            &p,
            [CONV.div_ceil(256), spans, 1],
            256,
        );
    }

    fn encode_mlx(
        &self,
        cmd: &Commands<'_>,
        w: &Weights,
        cache: &Cache,
        plan: &Plan,
        x: &Buffer,
        output: &Buffer,
    ) {
        let n = plan.meta.len() / 2;
        let spans = plan.spans.len() / 4;
        let p = [
            KH as u32,
            VH as u32,
            CONV as u32,
            n as u32,
            self.slots as u32,
            0,
            EPS.to_bits(),
        ];
        affine::project_group(
            cmd,
            &[
                (&w.qkv, &self.qkv),
                (&w.z, &self.z),
                (&w.alpha, &self.alpha),
                (&w.beta, &self.beta),
            ],
            x,
            n,
        );
        cmd.dispatch(
            "mlx_dn_conv",
            &[
                &self.qkv,
                &w.conv.buffer,
                &cache.history,
                &self.meta,
                &self.bounds,
                &self.convolved,
            ],
            &p,
            [(n * CONV).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch("q4b_dn_qk_norm", &[&self.convolved], &p, [KH * 2, n, 1], 32);
        cmd.dispatch(
            "mlx_dn_gates",
            &[
                &self.alpha,
                &self.beta,
                &w.a.buffer,
                &w.dt.buffer,
                &self.gates,
            ],
            &p,
            [(n * VH).div_ceil(256), 1, 1],
            256,
        );
        // Same 16/48-head geometry and arithmetic as the qualified dense-27B
        // packed encoder. Reuse each query/key across eight value rows on M5
        // only; decode and older GPU families retain the original route.
        let packed =
            cmd.tensor_accelerated() && plan.spans.chunks_exact(4).any(|span| span[1] >= 32);
        #[cfg(test)]
        let packed = packed && !BASELINE_RECURRENT_FOR_TEST.with(|v| v.get());
        cmd.dispatch(
            if packed {
                "mlx_dn_recurrent_packed"
            } else {
                "mlx_dn_recurrent"
            },
            &[
                &self.convolved,
                &self.gates,
                &cache.state,
                &self.spans,
                &self.meta,
                &self.attn,
                &self.checkpoint_rows,
            ],
            &p,
            [if packed { 4 } else { 32 }, VH, spans],
            128,
        );
        cmd.dispatch(
            "q4b_dn_gated_norm",
            &[&self.attn, &self.z, &w.norm.buffer],
            &p,
            [VH, n, 1],
            32,
        );
        affine::project(cmd, &w.out, &self.attn, output, n);
        cmd.dispatch(
            "dn_conv_commit",
            &[
                &self.qkv,
                &cache.history,
                &self.spans,
                &self.meta,
                &self.checkpoint_spans,
            ],
            &p,
            [CONV.div_ceil(256), spans, 1],
            256,
        );
    }
}

/// Single-layer test owner wraps the same cache/workspace encoder as the
/// model. This wrapper never participates in serving.
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
        if !(1..=512).contains(&capacity)
            || !(1..=64).contains(&slots)
            || !(1..=262144).contains(&context)
        {
            return Err(MetalError::Model(
                "Flash Next DeltaNet invalid rows/slots/context".into(),
            ));
        }
        Ok(Cache::bytes(slots) + Workspace::bytes(capacity, slots))
    }
    pub(super) fn new(
        d: &MetalDevice,
        w: &Weights,
        capacity: usize,
        slots: usize,
        context: usize,
    ) -> Result<Self> {
        Self::bytes(capacity, slots, context)?;
        let mut s = Self {
            layer: w.layer,
            capacity,
            context,
            lengths: vec![0; slots],
            poisoned: false,
            cache: Cache::new(d, slots)?,
            scratch: Workspace::new(d, capacity, slots)?,
        };
        for slot in 0..slots {
            s.reset(d, slot)?;
        }
        Ok(s)
    }
    fn healthy(&self) -> Result<()> {
        if self.poisoned {
            Err(MetalError::Model(
                "Flash Next DeltaNet state poisoned; reload required".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub(super) fn reset(&mut self, d: &MetalDevice, slot: usize) -> Result<()> {
        self.healthy()?;
        if slot >= self.lengths.len() {
            return Err(MetalError::Model("invalid DeltaNet reset slot".into()));
        }
        self.poisoned = true;
        let cmd = d.begin()?;
        self.cache.reset(&cmd, slot);
        cmd.finish()?;
        self.lengths[slot] = 0;
        self.poisoned = false;
        Ok(())
    }
    /// Exact-boundary fork/restore of both caches. No partial-prefix reuse.
    /// Whole-model integration must include HC/PLE/QSA/KV ownership too.
    pub(super) fn copy_slot(
        &mut self,
        d: &MetalDevice,
        source: usize,
        target: usize,
    ) -> Result<()> {
        self.healthy()?;
        if source >= self.lengths.len() || target >= self.lengths.len() || source == target {
            return Err(MetalError::Model("invalid DeltaNet copy slots".into()));
        }
        self.poisoned = true;
        let cmd = d.begin()?;
        self.cache.copy(&cmd, source, target, self.lengths.len());
        cmd.submit()?.wait()?;
        self.lengths[target] = self.lengths[source];
        self.poisoned = false;
        Ok(())
    }
    pub(super) fn run(
        &mut self,
        d: &MetalDevice,
        w: &Weights,
        rows: &[(usize, usize)],
        x: &Buffer,
        output: &Buffer,
    ) -> Result<()> {
        self.healthy()?;
        let plan = Plan::new(rows, &self.lengths, self.capacity, self.context)?;
        if w.layer != self.layer
            || x.len() < rows.len() * WIDTH * 4
            || output.len() < rows.len() * WIDTH * 4
        {
            return Err(MetalError::Model(
                "invalid Flash Next DeltaNet layer/input/output".into(),
            ));
        }
        // No GPU-owned state changes until every row and buffer is validated.
        self.scratch.stage(&plan);
        self.poisoned = true;
        let cmd = d.begin()?;
        self.scratch.encode(&cmd, w, &self.cache, &plan, x, output);
        cmd.submit()?.wait()?;
        self.lengths = plan.lengths;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_checks_all_rows_and_bounds_chunk_workspace() {
        let lengths = [0, 9, 0, 17];
        let rows: Vec<_> = (0..33)
            .map(|i| (2, i))
            .chain([(1, 9)])
            .chain((17..33).map(|i| (3, i)))
            .collect();
        let p = Plan::new(&rows, &lengths, 64, 128).unwrap();
        assert_eq!(p.lengths, [0, 10, 33, 33]);
        assert_eq!(p.spans, [0, 33, 2, 0, 33, 1, 1, 2, 34, 16, 3, 2]);
        assert_eq!(p.chunks, [0, 32, 2, 0, 32, 1, 2, 0, 34, 16, 3, 0]);
        assert!(p.short);
        for n in 16usize..=512 {
            assert!(n.div_ceil(32) <= n / 16);
        }
        for rows in [
            vec![],
            vec![(4, 0)],
            vec![(1, 8)],
            vec![(0, 0), (0, 2)],
            vec![(0, 0), (1, 9), (0, 1)],
            vec![(0, 0), (1, usize::MAX)],
        ] {
            assert!(Plan::new(&rows, &lengths, 512, 128).is_err());
        }
        assert!(Plan::new(&[(0, 0), (0, 1)], &lengths, 1, 128).is_err());
        assert!(Plan::new(&[(1, 9)], &lengths, 8, 9).is_err());
        for (n, s, c) in [
            (0, 1, 1),
            (513, 1, 1),
            (1, 0, 1),
            (1, 65, 1),
            (1, 1, 0),
            (1, 1, 262145),
        ] {
            assert!(State::bytes(n, s, c).is_err());
        }
    }

    #[test]
    fn poisoned_owner_refuses_even_reset_or_cache_copy() {
        // No fabricated GPU error: exercise the fail-closed owner state
        // explicitly, before any method may submit GPU work.
        let d = MetalDevice::new(Some(32 << 20)).unwrap();
        let mut s = State {
            layer: 0,
            capacity: 1,
            context: 1,
            lengths: vec![0, 0],
            poisoned: true,
            cache: Cache::new(&d, 2).unwrap(),
            scratch: Workspace::new(&d, 1, 2).unwrap(),
        };
        assert!(s.reset(&d, 0).unwrap_err().to_string().contains("poisoned"));
        assert!(
            s.copy_slot(&d, 0, 1)
                .unwrap_err()
                .to_string()
                .contains("poisoned")
        );
        assert_eq!(s.lengths, [0, 0]);
    }
}
