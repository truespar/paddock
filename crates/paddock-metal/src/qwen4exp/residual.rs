//! The four-stream residual graph, retaining the elected GGUF's folded gamma
//! and F32 activation contract across c=1, c=4 and prefill. Tiny projections
//! use register reuse; wider rows use bounded F32 TensorOps tiles.
use super::{affine, mlx};
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::mapped::MappedGguf;

pub(super) const WIDTH: usize = 2560;
pub(super) const WIDE: usize = WIDTH * 4;
const LOW: usize = 320;
pub(super) const EPS: f32 = 1e-6;
#[cfg(test)]
thread_local! {
    pub(super) static HC_VECTOR_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static HC_UP_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static HC_COMBINE_NORM_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

pub(super) fn load_weight(
    d: &MetalDevice,
    map: &MappedGguf,
    name: &str,
    dims: &[usize],
    ty: u32,
) -> Result<Weight> {
    if map.tensor_info(name).is_none_or(|t| t.raw_type != ty) {
        return Err(MetalError::Model(format!(
            "Flash Next {name}: expected type {ty}"
        )));
    }
    if crate::iquant::is_iq(ty) {
        Weight::load_iq(d, map, name, dims)
    } else {
        Weight::load(d, map, name, dims)
    }
}

pub(super) fn norm(
    cmd: &Commands<'_>,
    input: &Buffer,
    weight: &Weight,
    output: &Buffer,
    rows: usize,
) {
    cmd.dispatch(
        if weight.ty == mlx::FOLDED_NORM {
            "q4b_norm"
        } else {
            "q4x_norm"
        },
        &[input, &weight.buffer, output],
        &[WIDTH as u32, 4, EPS.to_bits()],
        [4, rows, 1],
        256,
    );
}

/// This projection API is intentionally narrower than Weight::linear: the
/// residual/PLE planes are Q8 or F32, and never accept a prepared F16 input.
pub(super) fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    if affine::is_affine(w.ty) {
        return affine::project(cmd, w, x, y, rows);
    }
    assert!(matches!(w.ty, 0 | 8) && w.k.is_multiple_of(32));
    let p = [w.k as u32, w.n as u32, rows as u32, w.ty, 1f32.to_bits()];
    if w.ty == 8 {
        let (name, n, r) = match rows {
            1 => ("linear_q8_r1", 4, 1),
            2..=4 => ("linear_q8_r4", 4, 4),
            5..=8 => ("linear_q8_r8", 4, 8),
            _ => ("q4x_q8_mm", 32, 32),
        };
        cmd.dispatch(
            name,
            &[&w.buffer, x, y],
            &p,
            [w.n.div_ceil(n), rows.div_ceil(r), 1],
            128,
        );
    } else {
        cmd.dispatch(
            "linear",
            &[&w.buffer, x, y],
            &p,
            [w.n.div_ceil(4), rows, 1],
            128,
        );
    }
}

pub(super) struct HyperConnection {
    norm: Weight,
    down: Weight,
    up: Weight,
    inject: Option<Weight>,
}
pub(super) struct Workspace {
    rows: usize,
    pub(super) norm: Buffer,
    pub(super) low: Buffer,
    pub(super) gate: Buffer,
    pub(super) inject: Buffer,
    pub(super) mixed: Buffer,
}
impl Workspace {
    pub(super) fn bytes(rows: usize) -> Result<usize> {
        if !(1..=affine::MAX_ROWS).contains(&rows) {
            return Err(MetalError::Model(
                "Flash Next residual rows must be 1..=2048".into(),
            ));
        }
        Ok(rows * (WIDE * 2 + LOW + 4 + WIDTH) * 4)
    }
    pub(super) fn new(d: &MetalDevice, rows: usize) -> Result<Self> {
        Self::bytes(rows)?;
        Ok(Self {
            rows,
            norm: d.alloc(rows * WIDE * 4)?,
            low: d.alloc(rows * LOW * 4)?,
            gate: d.alloc(rows * WIDE * 4)?,
            inject: d.alloc(rows * 4 * 4)?,
            mixed: d.alloc(rows * WIDTH * 4)?,
        })
    }
}
impl HyperConnection {
    pub(super) fn load_mlx(
        d: &MetalDevice,
        s: &mlx::Source,
        prefix: &str,
        inject: bool,
    ) -> Result<Self> {
        let w = |suffix: &str| s.weight(d, &format!("{prefix}.{suffix}"));
        Ok(Self {
            norm: w("hc_norm.weight")?,
            down: w("input_mix_weight_down")?,
            up: w("input_mix_weight_up")?,
            inject: if inject {
                Some(w("block_inject_weight")?)
            } else {
                None
            },
        })
    }
    pub(super) fn load(
        d: &MetalDevice,
        map: &MappedGguf,
        prefix: &str,
        inject: bool,
    ) -> Result<Self> {
        let w = |suffix: &str, dims: &[usize], ty| {
            load_weight(d, map, &format!("{prefix}_{suffix}.weight"), dims, ty)
        };
        Ok(Self {
            norm: w("norm", &[WIDE], 0)?,
            down: w("down", &[WIDE, LOW], 8)?,
            up: w("up", &[LOW, WIDE], 8)?,
            inject: if inject {
                Some(w("inject", &[WIDE, 4], 0)?)
            } else {
                None
            },
        })
    }

    pub(super) fn encode(&self, cmd: &Commands<'_>, h: &Buffer, s: &Workspace, rows: usize) {
        assert!(rows > 0 && rows <= s.rows && h.len() >= rows * WIDE * 4);
        norm(cmd, h, &self.norm, &s.norm, rows);
        self.encode_normalized(cmd, s, rows);
    }

    /// Consume the previous block's injection before this HC overwrites it.
    /// No deferred residual state escapes the call: both H and norm are written.
    pub(super) fn normalize_after(
        &self,
        cmd: &Commands<'_>,
        h: &Buffer,
        delta: &Buffer,
        s: &Workspace,
        rows: usize,
    ) -> bool {
        #[cfg(test)]
        let enabled = HC_COMBINE_NORM_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let enabled = true;
        if !enabled || !cmd.tensor_accelerated() || self.norm.ty != mlx::FOLDED_NORM {
            return false;
        }
        assert!(
            rows > 0
                && rows <= s.rows
                && h.len() >= rows * WIDE * 4
                && delta.len() >= rows * WIDTH * 4
        );
        cmd.dispatch(
            "q4b_hc_combine_norm",
            &[h, delta, &s.inject, &self.norm.buffer, &s.norm],
            &[WIDTH as u32, 4, EPS.to_bits()],
            [4, rows, 1],
            256,
        );
        true
    }

    pub(super) fn encode_normalized(&self, cmd: &Commands<'_>, s: &Workspace, rows: usize) {
        assert!(rows > 0 && rows <= s.rows);
        let mlx = affine::is_affine(self.down.ty);
        #[cfg(test)]
        let fused = HC_VECTOR_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let fused = true;
        let fused = fused
            && cmd.tensor_accelerated()
            && cmd.independent_rows()
            && rows <= 8
            && (self.down.ty, self.down.k, self.down.n) == (affine::A4G32, WIDE, LOW)
            && self
                .inject
                .as_ref()
                .is_some_and(|w| (w.ty, w.k, w.n) == (affine::A4G32, WIDE, 4));
        if fused {
            cmd.dispatch(
                "q4a_hc_down_vector",
                &[
                    &self.down.buffer,
                    &self.inject.as_ref().expect("checked injection").buffer,
                    &s.norm,
                    &s.low,
                    &s.inject,
                ],
                &[rows as u32],
                [42, rows, 1],
                64,
            );
        } else if rows > 8 && !mlx {
            // N=320 gives only 10..40 prefill tiles at our serving sizes.
            // Split the long reduction to occupy the GPU. The 32 partial
            // planes fit exactly in the existing gate buffer (320*32=WIDE),
            // which is dead until the later up projection. Dispatch barriers
            // fence reduction reads before that overwrite; no new allocation.
            const SPLITS: usize = 32;
            debug_assert_eq!(LOW * SPLITS, WIDE);
            cmd.dispatch(
                "q4x_hc_down_split",
                &[&self.down.buffer, &s.norm, &s.gate],
                &[rows as u32, SPLITS as u32],
                [LOW / 32, rows.div_ceil(32), SPLITS],
                128,
            );
            cmd.dispatch(
                "q4x_hc_down_reduce",
                &[&s.gate, &s.low],
                &[rows as u32, SPLITS as u32],
                [(rows * LOW).div_ceil(256), 1, 1],
                256,
            );
        } else {
            project(cmd, &self.down, &s.norm, &s.low, rows);
        }
        if !fused {
            cmd.dispatch(
                if mlx {
                    "q4b_scale_silu"
                } else {
                    "q4x_scale_silu"
                },
                &[&s.low],
                &[(rows * LOW) as u32],
                [(rows * LOW).div_ceil(256), 1, 1],
                256,
            );
        }
        #[cfg(test)]
        let fused_up = HC_UP_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let fused_up = true;
        let fused_up = fused_up
            && cmd.tensor_accelerated()
            && cmd.independent_rows()
            && rows <= 8
            && (self.up.ty, self.up.k, self.up.n) == (affine::A4G32, LOW, WIDE);
        if fused_up {
            cmd.dispatch(
                "q4a_hc_up_mix_vector",
                &[&self.up.buffer, &s.low, &s.norm, &s.gate, &s.mixed],
                &[rows as u32],
                [WIDTH / 4, rows, 1],
                128,
            );
        } else {
            project(cmd, &self.up, &s.low, &s.gate, rows);
            cmd.dispatch(
                if mlx { "q4b_hc_mix" } else { "q4x_hc_mix" },
                &[&s.norm, &s.gate, &s.mixed],
                &[WIDTH as u32, rows as u32],
                [(rows * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        if !fused && let Some(w) = &self.inject {
            if mlx {
                project(cmd, w, &s.norm, &s.inject, rows);
                cmd.dispatch(
                    "q4b_injection",
                    &[&s.inject],
                    &[(rows * 4) as u32],
                    [(rows * 4).div_ceil(256), 1, 1],
                    256,
                );
                return;
            }
            // Four tiny outputs cannot occupy the GPU with one SIMD each.
            // Use a fixed eight-SIMD reduction per output for every batch
            // shape, preserving F32 inputs and deterministic summation.
            // The shape/type is enforced by HyperConnection::load.
            cmd.dispatch(
                "q4x_inject_parallel",
                &[&w.buffer, &s.norm, &s.inject],
                &[rows as u32],
                [4, rows, 1],
                256,
            );
        }
    }

    pub(super) fn combine(
        &self,
        cmd: &Commands<'_>,
        h: &Buffer,
        delta: &Buffer,
        s: &Workspace,
        rows: usize,
    ) {
        assert!(
            self.inject.is_some()
                && rows > 0
                && rows <= s.rows
                && h.len() >= rows * WIDE * 4
                && delta.len() >= rows * WIDTH * 4
        );
        cmd.dispatch(
            if affine::is_affine(self.down.ty) {
                "q4b_hc_combine"
            } else {
                "q4x_hc_combine"
            },
            &[h, delta, &s.inject],
            &[WIDTH as u32, rows as u32],
            [(rows * WIDE).div_ceil(256), 1, 1],
            256,
        );
    }
}
