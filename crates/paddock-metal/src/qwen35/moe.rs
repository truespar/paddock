//! Original GPU-only Qwen 35B-A3B MoE. The elected Q8 expert bytes stay
//! compressed; prefill gathers routed rows into bounded TensorOps tiles.
//! GPT-OSS shares stable integer compaction, not its biased/clipped FFN math.
use super::*;
use paddock_models::mapped::MappedGguf;

const EXPERTS: usize = 256;
const ACTIVE: usize = 8;
const WIDTH: usize = 2048;
const FF: usize = 512;

// Test-thread-local causal isolation, never a serving environment switch.
#[cfg(test)]
thread_local! {
    pub(super) static GROUPED_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static PRECISE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

pub(super) fn precise() -> bool {
    #[cfg(test)]
    return PRECISE_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    true
}

pub(super) fn validate(map: &MappedGguf, layers: usize) -> Result<()> {
    for (key, expected) in [
        ("expert_count", EXPERTS),
        ("expert_used_count", ACTIVE),
        ("expert_feed_forward_length", FF),
        ("expert_shared_feed_forward_length", FF),
    ] {
        if map.gguf().arch_field(key).and_then(|v| v.as_u64()) != Some(expected as u64) {
            return Err(MetalError::Model(format!(
                "Metal Qwen MoE requires {key}={expected}"
            )));
        }
    }
    // Reject an unsupported mixed-quant export before allocating/uploading
    // tens of gigabytes. Q4 expert support is a separate implementation gate.
    for i in 0..layers {
        for (name, dims, ty) in [
            ("ffn_gate_exps.weight", vec![WIDTH, FF, EXPERTS], 8),
            ("ffn_up_exps.weight", vec![WIDTH, FF, EXPERTS], 8),
            ("ffn_down_exps.weight", vec![FF, WIDTH, EXPERTS], 8),
            ("ffn_gate_inp.weight", vec![WIDTH, EXPERTS], 0),
            ("ffn_gate_inp_shexp.weight", vec![WIDTH], 0),
        ] {
            let name = format!("blk.{i}.{name}");
            let t = map
                .tensor_info(&name)
                .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
            if t.raw_type != ty || t.dims.iter().map(|&n| n as usize).collect::<Vec<_>>() != dims {
                return Err(MetalError::Model(format!(
                    "{name}: expected type {ty}, {dims:?}; Qwen Metal currently requires Q8 experts/F32 routers"
                )));
            }
        }
    }
    Ok(())
}

pub(super) struct Experts {
    router: Weight,
    shared_gate: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
}
impl Experts {
    pub(super) fn load(
        device: &MetalDevice,
        source: &checkpoint::Source,
        layer: usize,
    ) -> Result<Self> {
        let w =
            |name: &str, dims: &[usize]| source.load(device, &format!("blk.{layer}.{name}"), dims);
        Self::load_with(w)
    }
    pub(super) fn load_with(w: impl Fn(&str, &[usize]) -> Result<Weight>) -> Result<Self> {
        let result = Self {
            router: w("ffn_gate_inp.weight", &[WIDTH, EXPERTS])?,
            shared_gate: w("ffn_gate_inp_shexp.weight", &[WIDTH])?,
            gate: w("ffn_gate_exps.weight", &[WIDTH, FF, EXPERTS])?,
            up: w("ffn_up_exps.weight", &[WIDTH, FF, EXPERTS])?,
            down: w("ffn_down_exps.weight", &[FF, WIDTH, EXPERTS])?,
        };
        if [&result.gate, &result.up, &result.down]
            .iter()
            .any(|w| w.ty != 8)
            || [&result.router, &result.shared_gate]
                .iter()
                .any(|w| !matches!(w.ty, 0 | 30))
        {
            return Err(MetalError::Model(
                "Qwen MoE requires Q8 experts and F32/BF16 routers".into(),
            ));
        }
        Ok(result)
    }
}

pub(super) struct Workspace {
    rows: usize,
    logits: Buffer,
    shared_gate: Buffer,
    ids: Buffer,
    weights: Buffer,
    lists: Buffer,
    counts: Buffer,
    tiles: Buffer,
    gu: Buffer,
    out: Buffer,
}
impl Workspace {
    fn sizes(rows: usize) -> [usize; 9] {
        [
            rows * EXPERTS,
            rows,
            rows * ACTIVE,
            rows * ACTIVE,
            EXPERTS * rows * ACTIVE,
            EXPERTS,
            1 + 2 * ((rows * ACTIVE).div_ceil(16) + EXPERTS),
            rows * ACTIVE * FF * 2,
            rows * ACTIVE * WIDTH,
        ]
    }
    pub(super) fn bytes(rows: usize) -> u64 {
        Self::sizes(rows).into_iter().map(|n| n as u64 * 4).sum()
    }
    pub(super) fn new(device: &MetalDevice, rows: usize) -> Result<Self> {
        let mut sizes = Self::sizes(rows).into_iter();
        let mut a = || device.alloc(sizes.next().expect("nine MoE workspace planes") * 4);
        Ok(Self {
            rows,
            logits: a()?,
            shared_gate: a()?,
            ids: a()?,
            weights: a()?,
            lists: a()?,
            counts: a()?,
            tiles: a()?,
            gu: a()?,
            out: a()?,
        })
    }
    fn project(&self, cmd: &Commands<'_>, w: &Experts, input: &Buffer, rows: usize, grouped: bool) {
        assert!(rows <= self.rows);
        let tile = if rows * ACTIVE >= EXPERTS * 32 {
            32
        } else {
            16
        };
        if grouped {
            cmd.dispatch(
                "moe_align",
                &[&self.ids, &self.lists, &self.counts],
                &[(rows * ACTIVE) as u32],
                [EXPERTS, 1, 1],
                256,
            );
            cmd.dispatch(
                "moe_tiles",
                &[&self.counts, &self.tiles],
                &[EXPERTS as u32, tile as u32],
                [1, 1, 1],
                256,
            );
        }
        for down in [false, true] {
            let (k, n) = if down { (FF, WIDTH) } else { (WIDTH, FF) };
            let source = if down { &w.down.buffer } else { &w.gate.buffer };
            let input = if down { &self.gu } else { input };
            let out = if down { &self.out } else { &self.gu };
            let p = [k as u32, n as u32, rows as u32];
            if grouped {
                let kernel = match (down, tile, precise()) {
                    (false, 16, false) => "qmoe_gu_grouped16",
                    (false, _, false) => "qmoe_gu_grouped32",
                    (true, 16, false) => "qmoe_down_grouped16",
                    (true, _, false) => "qmoe_down_grouped32",
                    (false, 16, true) => "qmoe_gu_strict16",
                    (false, _, true) => "qmoe_gu_strict32",
                    (true, 16, true) => "qmoe_down_strict16",
                    (true, _, true) => "qmoe_down_strict32",
                };
                cmd.dispatch(
                    kernel,
                    &[
                        source,
                        &w.up.buffer,
                        input,
                        &self.lists,
                        &self.counts,
                        &self.tiles,
                        out,
                    ],
                    &p,
                    [
                        n.div_ceil(if precise() { 32 } else { 64 }) * if down { 1 } else { 2 },
                        (rows * ACTIVE).div_ceil(tile) + EXPERTS,
                        1,
                    ],
                    128,
                );
            } else if down {
                cmd.dispatch(
                    "qmoe_down_decode",
                    &[source, input, &self.ids, out],
                    &p,
                    [n.div_ceil(4), rows * ACTIVE, 1],
                    128,
                );
            } else {
                cmd.dispatch(
                    "qmoe_gu_decode",
                    &[source, &w.up.buffer, input, &self.ids, out],
                    &p,
                    [n.div_ceil(4), rows * ACTIVE, 1],
                    128,
                );
            }
            if !down {
                cmd.dispatch(
                    "qmoe_swiglu",
                    &[&self.gu],
                    &[FF as u32, (rows * ACTIVE) as u32],
                    [(rows * ACTIVE * FF).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }
}

impl Qwen35 {
    // MoE router boundaries amplify otherwise small row-count-dependent
    // F16 projection errors. Reuse our strict F32 TensorOps Q8 ladder for
    // the small dense backbone too; dense 9B/27B/MLX elections stay intact.
    pub(super) fn project(
        &self,
        cmd: &Commands<'_>,
        planes: &[(&Weight, &Buffer)],
        input: &Buffer,
        rows: usize,
        workspace: &Buffer,
    ) {
        if let Some(ternary) = &self.ternary {
            let head = planes.len() == 1 && std::ptr::eq(planes[0].0, self.head());
            return ternary.project(cmd, planes, input, rows, workspace, head);
        }
        if let Some(bonsai) = &self.bonsai {
            // The head consumes selected rows, not the plan's full row map.
            // Always preserve its strict, live-count-independent reduction.
            let head = planes.len() == 1 && std::ptr::eq(planes[0].0, self.head());
            return bonsai.project(
                cmd,
                planes,
                input,
                rows,
                if head { None } else { cmd.projection_rows() },
            );
        }
        if self.splash {
            return crate::splash::project(cmd, planes, input, rows, workspace);
        }
        if self.mlx
            && let Some(spans) = cmd.projection_rows()
        {
            let head = planes.len() == 1 && std::ptr::eq(planes[0].0, self.head());
            let head_span = [(0, rows, 1)];
            return crate::affine::project_stable(
                cmd,
                planes,
                input,
                rows,
                workspace,
                if head { &head_span } else { spans },
            );
        }
        #[cfg(test)]
        if self.mlx && rows <= 12 && projection::CANONICAL_MLX_FOR_TEST.with(|v| v.get()) {
            return crate::affine::project_verify(cmd, planes, input, rows, workspace, true);
        }
        if self.mlx && self.verifying {
            return crate::affine::project_verify(
                cmd,
                planes,
                input,
                rows,
                workspace,
                self.spec
                    .as_ref()
                    .expect("verification buffers allocated")
                    .live
                    == 1,
            );
        }
        if !self.geometry.moe() || !precise() {
            return projection::project(cmd, planes, input, rows, workspace);
        }
        for &(weight, output) in planes {
            if weight.ty == 8 && rows >= 2 {
                let (kernel, tile, cols) = if rows >= 16 {
                    let tile = if rows <= 16 {
                        16
                    } else if rows <= 32 {
                        32
                    } else {
                        64
                    };
                    (
                        match tile {
                            16 => "muse_q8_f32_16",
                            32 => "muse_q8_f32_32",
                            _ => "muse_q8_f32_64",
                        },
                        tile,
                        16,
                    )
                } else {
                    let tile = if rows <= 4 {
                        4
                    } else if rows <= 8 {
                        8
                    } else {
                        16
                    };
                    (
                        match tile {
                            4 => "linear_q8_r4",
                            8 => "linear_q8_r8",
                            _ => "linear_q8_r16",
                        },
                        tile,
                        4,
                    )
                };
                // Strict tensor route stages BK128; every elected Q8 input
                // dimension is divisible by 128, including the shared FFN.
                assert!(weight.k.is_multiple_of(128));
                cmd.dispatch(
                    kernel,
                    &[&weight.buffer, input, output],
                    &[
                        weight.k as u32,
                        weight.n as u32,
                        rows as u32,
                        weight.ty,
                        1f32.to_bits(),
                    ],
                    [weight.n.div_ceil(cols), rows.div_ceil(tile), 1],
                    128,
                );
            } else {
                projection::project(cmd, &[(weight, output)], input, rows, workspace);
            }
        }
    }

    // The shared FFN has already written delta; norm is still the identical
    // input used by every routed expert. Only the combined output joins x.
    pub(super) fn moe_ffn(&self, cmd: &Commands<'_>, w: &Experts, rows: usize) {
        let m = self.moe_scratch.as_ref().expect("MoE workspace loaded");
        let s = &self.scratch;
        // Routing is sensitive to near-ties. These two small projections
        // retain F32 input/accumulation even in prefill; generic TensorOps
        // projection would introduce an unnecessary F16 boundary. The base
        // router is F32 and nextn's BF16 values are exactly read as F32.
        for (weight, out) in [(&w.router, &m.logits), (&w.shared_gate, &m.shared_gate)] {
            cmd.dispatch(
                "linear",
                &[&weight.buffer, &s.norm, out],
                &[
                    weight.k as u32,
                    weight.n as u32,
                    rows as u32,
                    weight.ty,
                    1f32.to_bits(),
                ],
                [weight.n.div_ceil(4), rows, 1],
                128,
            );
        }
        cmd.dispatch(
            "qmoe_route",
            &[&m.logits, &m.ids, &m.weights],
            &[],
            [rows, 1, 1],
            32,
        );
        let grouped = rows >= 16;
        #[cfg(test)]
        let grouped = grouped && GROUPED_FOR_TEST.with(|value| value.get());
        m.project(cmd, w, &s.norm, rows, grouped);
        cmd.dispatch(
            "qmoe_fold",
            &[&m.out, &m.weights, &m.shared_gate, &s.delta],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
    }
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
