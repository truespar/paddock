//! The armed draft chain read back: its drafts, and - for the confidence
//! census - each draft's log p(top1) under the drafter.
//!
//! The census prices confidence-ranked verification before it is built:
//! every round records each slot's drafted depth, its accepted run and the
//! drafter's own confidence at every position, so calibration (does the
//! drafter's top-1 probability predict acceptance?) and any allocation of
//! verify rows can be replayed offline against outcomes the target really
//! produced. Off unless PADDOCK_SPEC_CONF is set; the pick is the same
//! either way (`argmax_top2_rows` keeps `argmax_rows`' tie rule and walk).

use cudarc::driver::CudaSlice;

use super::{GpuModelError, GpuQwen35};
use crate::gpu::{GpuError, GpuExecutor};

/// The census planes: one step's stats, every step's stats (i-major,
/// `[depth][rows][4]`, column 0 = log p(top1)), and the shape last launched.
pub(super) struct DraftConf {
    alt: CudaSlice<u32>,
    stats: CudaSlice<f32>,
    plane: CudaSlice<f32>,
    rows: usize,
    depth: usize,
    /// (rows, depth) of the chain last launched
    last: (usize, usize),
}

fn conf_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPEC_CONF").is_some())
}

/// One chain step's pick into `tok`: `argmax_rows`, or with the census armed
/// `argmax_top2_rows` and this step's stats copied into the plane.
#[allow(clippy::too_many_arguments)]
pub(super) fn pick_draft_step(
    exec: &GpuExecutor,
    conf: Option<&mut DraftConf>,
    logits: &CudaSlice<f32>,
    tok: &mut CudaSlice<u32>,
    step: usize,
    b: usize,
    vocab: usize,
) -> Result<(), GpuError> {
    match conf {
        Some(c) if step < c.depth && b <= c.rows => {
            exec.argmax_top2_rows(
                logits,
                tok,
                &mut c.alt,
                &mut c.stats,
                vocab as u32,
                b,
                vocab,
            )?;
            exec.copy_region(&c.stats, 0, &mut c.plane, step * b * 4, b * 4)
        }
        _ => exec.argmax_rows(logits, tok, b, vocab),
    }
}

impl GpuQwen35 {
    /// Seat the census planes for a chain over `b` rows (outside any
    /// capture: the draft graph bakes their addresses).
    pub(super) fn ensure_draft_conf(&mut self, b: usize) -> Result<(), GpuModelError> {
        if !conf_on() || !self.exec.has_argmax_top2_rows() {
            return Ok(());
        }
        let (rows, depth) = {
            let sb = self.spec_batch.as_ref().expect("spec batch");
            (sb.alloc_batch, sb.chain_depth)
        };
        let run = self.chain_steps();
        if !self
            .draft_conf
            .as_ref()
            .is_some_and(|c| c.rows >= rows && c.depth >= depth)
        {
            let e = &self.exec;
            self.draft_conf = Some(DraftConf {
                alt: e.alloc_u32(rows)?,
                stats: e.alloc(rows * 4)?,
                plane: e.alloc(depth * rows * 4)?,
                rows,
                depth,
                last: (0, 0),
            });
        }
        if let Some(c) = self.draft_conf.as_mut() {
            c.last = (b, run);
        }
        Ok(())
    }

    /// The last chain's confidences in its row order: log p(top1) per drafted
    /// position. None unless the census is armed.
    pub fn spec_draft_conf_mtp(&self) -> Result<Option<Vec<Vec<f32>>>, GpuModelError> {
        let Some(c) = self.draft_conf.as_ref() else {
            return Ok(None);
        };
        let (n, depth) = c.last;
        let flat = self.exec.to_host(&c.plane)?;
        Ok(Some(
            (0..n)
                .map(|r| (0..depth).map(|i| flat[(i * n + r) * 4]).collect())
                .collect(),
        ))
    }

    /// Read the armed chain's draft plane without disarming (drivers call
    /// this post-verify - the picks readback already synced the stream, so
    /// this is a cheap device->host copy of n_draft×n u32).
    pub(super) fn spec_chain_peek(&self) -> Result<Vec<Vec<u32>>, GpuModelError> {
        let (slots, k_use) = self.spec_chain.as_ref().expect("armed chain");
        let n = slots.len();
        let sb = self.spec_batch.as_ref().expect("spec batch");
        let flat = self.exec.to_host_u32(&sb.d_draft)?;
        Ok((0..n)
            .map(|b| (0..*k_use).map(|i| flat[i * n + b]).collect())
            .collect())
    }

    /// Async draft chain, phase 2: the armed chain's drafts in pendings
    /// order; disarms. `Ok(None)` when nothing is armed.
    pub fn spec_draft_fetch_mtp(&mut self) -> Result<Option<Vec<Vec<u32>>>, GpuModelError> {
        if self.spec_chain.is_none() {
            return Ok(None);
        }
        let out = self.spec_chain_peek()?;
        self.spec_chain = None;
        Ok(Some(out))
    }
}
