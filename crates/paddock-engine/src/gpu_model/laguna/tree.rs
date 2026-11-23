//! Ripple's DFlash2 rounds as a draft TREE: the selector lattice grown into
//! up to 32 nodes a slot (`dflash2_tree`), verified in one pass, the target's
//! own picks walking it - the deepest confirmed path is the round's output.
//! Measured on the four bench prompts, teacher-forced against the target's
//! greedy stream: 5.71 tokens a round on the chain, 7.37 on a 32-node tree.
//!
//! The round rides the service's ARMED protocol unchanged: `tree_begin` (its
//! `spec_draft_begin`) drafts and grows the trees; the verify gets the
//! service's chunks only for their slots, lengths and plans - node `n` of a
//! slot's tree takes the plan of chunk row `depth(n)`: exactly one node a
//! depth is ever consulted by the walk, so every sample a path uses is fresh
//! (the tree's sampled rounds stay the target's distribution). The strip
//! entry hands back each slot's tokens; the plans entry answers in chain
//! form (the path as both drafts and picks) for the service's own replay.
//!
//! Rows: node `j` of a slot whose committed end is `base` appends at cache
//! slot `base + j` and reads at logical position `base + depth(j)`; the
//! verify attention masks the tree's own keys to each row's ancestors (slot
//! 975). The commit moves the accepted path's K/V rows down to `base + k`
//! (slot 976) and its tapped residuals to the front of its band, then the
//! drafter appends their features at the true positions.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::sampler::DevicePlan;

use paddock_models::ripple::draft;

use super::batch::SPEC_ROWS;
use super::dflash2::MAX_DRAFT;
use super::dflash2_tree::{DraftTree, build_tree};
use super::{Flavor, GpuLaguna};

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    GpuError::Driver(e.to_string())
}

/// Nodes a slot's tree verifies, anchor included (the masks are one u32).
pub(super) const TREE_NODES: usize = 32;
/// Verify rows a tree round spends over all its slots: 32 nodes a slot at
/// one or two, 16 at four. GB10, four slots, tokens a second (greedy /
/// temperature 1.0): 16 nodes 48.1 / 52.8, 24 nodes 47.6 / 47.7, 32 nodes
/// 50.9 / 43.7 - past 64 rows the round's own cost outgrows what the extra
/// nodes accept.
const TREE_ROWS: usize = 64;
/// A node's fan-out: the selector row's top candidates.
const TREE_CHILDREN: usize = 16;
/// The proposal temperature over a lattice row (teacher-forced, tokens a
/// round: 7.37 at 1.0, 7.20 at 1.5).
const TREE_TEMP: f32 = 1.0;
/// A move-plan record: [slot, n, (src, dst) x n] for paths up to 15 deep.
const MOVE_STRIDE: usize = 2 + 2 * MAX_DRAFT;

/// What the armed round carries between `tree_begin` and the verify.
#[derive(Default)]
pub(crate) struct TreeState {
    /// per slot: its tree for this round (None = cold, pending-only)
    armed: Option<Vec<(usize, DraftTree)>>,
    /// the plans entry's chain-shaped drafts for the service's replay
    fetch: Option<Vec<Vec<u32>>>,
    /// compaction planes, built at the first round
    planes: Option<TreePlanes>,
}

struct TreePlanes {
    /// every layer's K / V pool base and table kind (0 full, 1 ring)
    kp: CudaSlice<u64>,
    vp: CudaSlice<u64>,
    kind: CudaSlice<u32>,
    plan: CudaSlice<u32>,
}

/// One slot's verified tree: the accepted path's tokens and the bonus.
pub(crate) struct TreeOut {
    pub path: Vec<u32>,
    pub bonus: u32,
}

impl GpuLaguna {
    /// Draft-tree rounds are on: Ripple with its drafter, the pack's lattice,
    /// masked rung and mover, and the hd128 tile the rung lives on.
    /// PADDOCK_RIPPLE_CHAIN=1 keeps the chain (the A/B arm).
    pub(crate) fn tree_on(&self) -> bool {
        self.hp.flavor == Flavor::Ripple
            && self.dflash2.as_ref().is_some_and(|d| d.state.is_some())
            && self.exec.has_dflash_lattice()
            && self.exec.has_spec_tree()
            && self.exec.has_hrow_gather()
            && self.hp.head_dim == 128
            && matches!(self.kv_dtype, KvDtype::Fp8E4m3 | KvDtype::Fp16)
            && paddock_models::dev_var_os!("PADDOCK_RIPPLE_CHAIN").is_none()
    }

    /// The armed round's first half: draft every warm slot's block, grow
    /// its tree off the lattice, keep the trees for the verify. Returns the
    /// drafted depth and a kept flag per pendings entry (the service's
    /// `spec_draft_begin` contract).
    pub(crate) fn tree_begin(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GpuModelError> {
        let depth = k.min(MAX_DRAFT);
        if !self.tree_on() || depth == 0 || pendings.is_empty() {
            return Ok(None);
        }
        let (feat, cap) = {
            let st = self
                .dflash2
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("state");
            (st.feat.clone(), st.draft_slots)
        };
        let mut reqs: Vec<(usize, usize, usize, u32)> = Vec::new();
        for (i, &(slot, tok)) in pendings.iter().enumerate() {
            let Some(&(_, e)) = feat.get(slot) else {
                continue;
            };
            let p = e as usize;
            if p == 0 || p + TREE_NODES > self.max_ctx || !self.d2_warm(slot, p) {
                continue;
            }
            match self.ensure_full_rows(&[slot as u32], &[(p + TREE_NODES - 1) as u32]) {
                Ok(()) => reqs.push((i, slot, p, tok)),
                Err(GpuModelError::PoolExhausted) => {}
                Err(e) => return Err(e),
            }
        }
        if reqs.is_empty() {
            return Ok(None);
        }
        // the round's rows: a tree a kept slot, one row every other
        // (PADDOCK_TREE_NODES / _CHILDREN: dev shapes for the A/B)
        let budget = paddock_models::dev_var!("PADDOCK_TREE_NODES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(TREE_NODES)
            .min(TREE_NODES);
        let children = paddock_models::dev_var!("PADDOCK_TREE_CHILDREN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(TREE_CHILDREN);
        let others = pendings.len() - reqs.len();
        let rows = TREE_ROWS.max(reqs.len() * 2).min(SPEC_ROWS);
        let nodes = budget.min(rows.saturating_sub(others) / reqs.len());
        if nodes < 2 {
            return Ok(None);
        }
        let mut kept = vec![false; pendings.len()];
        let mut armed = Vec::with_capacity(reqs.len());
        for chunk in reqs.chunks(cap) {
            self.d2_draft_pass(chunk)?;
            let lat = self.d2_lattice(chunk.len())?;
            for (b, &(i, slot, _, tok)) in chunk.iter().enumerate() {
                let (ids, scores) = &lat[b];
                let k = draft::SEL_TOP_K;
                let t = build_tree(ids, scores, tok, k, depth, nodes, children, TREE_TEMP);
                kept[i] = true;
                armed.push((slot, t));
            }
        }
        let ts = &mut self.dflash2.as_mut().expect("drafter").tree;
        ts.armed = Some(armed);
        ts.fetch = None;
        Ok(Some((depth, kept)))
    }

    /// The service's replay drafts for an armed round's plans entry.
    pub(crate) fn tree_fetch(&mut self) -> Option<Vec<Vec<u32>>> {
        let ts = &mut self.dflash2.as_mut()?.tree;
        ts.armed = None;
        ts.fetch.take()
    }

    /// Whether a round is armed (the plans entry answers for it).
    pub(crate) fn tree_armed(&self) -> bool {
        self.dflash2
            .as_ref()
            .is_some_and(|d| d.tree.armed.is_some())
    }

    /// The armed round's verify: `reqs` are the service's chunks (slot,
    /// committed end, [pending, placeholders]), `plans` one a chunk row (None
    /// = greedy). Returns each request's path and bonus; Ok(None) declines
    /// (the service falls back), the armed trees consumed either way.
    pub(crate) fn tree_round(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: Option<&[DevicePlan]>,
    ) -> Result<Option<Vec<TreeOut>>, GpuModelError> {
        let Some(armed) = self.dflash2.as_mut().and_then(|d| d.tree.armed.take()) else {
            return Ok(None);
        };
        let mut trees: Vec<DraftTree> = Vec::with_capacity(reqs.len());
        let (mut toks, mut pos, mut tpos, mut tmask, mut slots) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut runs = Vec::with_capacity(reqs.len());
        let mut row_plans = Vec::new();
        let mut plan_base = 0usize;
        for (slot, start, chunk) in reqs {
            let depth = chunk.len().saturating_sub(1);
            let tree = (depth > 0)
                .then(|| {
                    armed
                        .iter()
                        .find(|(s, _)| s == slot)
                        .map(|(_, t)| t.pruned(depth))
                })
                .flatten()
                .unwrap_or_else(|| DraftTree::anchor(chunk[0]));
            if *start + tree.tokens.len() > self.max_ctx {
                return Ok(None);
            }
            runs.push((toks.len(), tree.tokens.len()));
            for j in 0..tree.tokens.len() {
                toks.push(tree.tokens[j]);
                pos.push((*start + j) as u32);
                tpos.push((*start + tree.depth[j] as usize) as u32);
                tmask.push(tree.anc[j]);
                tmask.push(*start as u32);
                slots.push(*slot as u32);
                if let Some(p) = plans {
                    row_plans.push(p[plan_base + tree.depth[j] as usize]);
                }
            }
            plan_base += chunk.len();
            trees.push(tree);
        }
        let total = toks.len();
        if total > SPEC_ROWS || plans.is_some_and(|p| p.len() != plan_base) {
            return Ok(None);
        }
        let packed = match plans {
            Some(_) => match Self::pack_spec_par(&row_plans, self.device_trunc_supported()) {
                Some(p) => Some(p),
                None => return Ok(None),
            },
            None => None,
        };
        for ((slot, start, _), &(_, n)) in reqs.iter().zip(&runs) {
            match self.ensure_full_rows(&[*slot as u32], &[(*start + n - 1) as u32]) {
                Ok(()) => {}
                Err(GpuModelError::PoolExhausted) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
        self.tree_verify_rows((&toks, &pos, &slots), (&tpos, &tmask), &runs)?;
        let picks = self.spec_picks(packed.as_ref(), total)?;
        let mut outs = Vec::with_capacity(reqs.len());
        let mut paths = Vec::with_capacity(reqs.len());
        for (t, &(base, _)) in trees.iter().zip(&runs) {
            let path = t.accept(|n| picks[base + n]);
            let last = path.last().copied().unwrap_or(0);
            outs.push(TreeOut {
                path: path.iter().map(|&n| t.tokens[n]).collect(),
                bonus: picks[base + last],
            });
            paths.push(path);
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let shape: Vec<(usize, usize, usize)> = reqs
                .iter()
                .zip(&runs)
                .zip(&paths)
                .map(|(((_, _, c), &(_, n)), p)| (c.len(), n, p.len()))
                .collect();
            tracing::info!("[tree] (chunk, nodes, accepted) a slot: {shape:?}");
        }
        self.tree_commit(reqs, &runs, &paths)?;
        // the plans entry's replay: the path as drafts, then a token the
        // bonus cannot equal
        let fetch = reqs
            .iter()
            .zip(&outs)
            .map(|((_, _, c), o)| {
                let mut d = o.path.clone();
                d.resize(c.len().saturating_sub(1), o.bonus ^ 1);
                d
            })
            .collect();
        self.dflash2.as_mut().expect("drafter").tree.fetch = Some(fetch);
        Ok(Some(outs))
    }

    /// The armed round through the chain entries (the service's own replay:
    /// reasoning slots, greedy): picks in the chain layout - a request's path
    /// tokens, then its bonus, the rest of its chunk the bonus again - which
    /// the fetched drafts (the path, then a token the bonus is not) replay
    /// into exactly the path and the bonus.
    pub(crate) fn tree_chain_picks(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: Option<&[DevicePlan]>,
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        let Some(outs) = self.tree_round(reqs, plans)? else {
            return Ok(None);
        };
        let mut picks = Vec::with_capacity(reqs.iter().map(|r| r.2.len()).sum());
        for ((_, _, c), o) in reqs.iter().zip(outs) {
            let mut p = o.path;
            p.resize(c.len(), o.bonus);
            picks.extend(p);
        }
        Ok(Some(picks))
    }

    /// Land the accepted paths: their K/V rows at their true positions in
    /// every layer, their tapped residuals at the front of their runs, then
    /// the drafter's features for the committed rows.
    fn tree_commit(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        runs: &[(usize, usize)],
        paths: &[Vec<usize>],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        self.tree_planes()?;
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        let row_bytes = kv_dim * if self.kv_dtype == KvDtype::Fp16 { 2 } else { 1 };
        // the move plan: one group a slot whose path left its node order
        let mut plan = Vec::new();
        for ((slot, start, _), path) in reqs.iter().zip(paths) {
            if path.iter().enumerate().all(|(k, &n)| n == k + 1) {
                continue;
            }
            let mut g = vec![*slot as u32, path.len() as u32];
            for (k, &n) in path.iter().enumerate() {
                g.push((*start + n) as u32);
                g.push((*start + k + 1) as u32);
            }
            g.resize(MOVE_STRIDE, 0);
            plan.extend(g);
        }
        // the committed rows, slot by slot: the anchor, then the path, at
        // their true positions (the drafter packs them, d2_commit_rows)
        let mut rows = Vec::new();
        let mut spans = Vec::with_capacity(runs.len());
        for (((slot, start, _), &(base, _)), path) in reqs.iter().zip(runs).zip(paths) {
            spans.push((rows.len(), path.len() + 1));
            for (k, n) in std::iter::once(0).chain(path.iter().copied()).enumerate() {
                rows.push(((base + n) as u32, (*start + k) as u32, *slot as u32));
            }
        }
        if !plan.is_empty() {
            let bs = self.batch.as_ref().expect("batch enabled");
            let (d_bt, swa_bt, bps) = (&bs.d_bt, &bs.swa_bt, bs.bps);
            let n_layers = self.layers.len();
            let tp = self
                .dflash2
                .as_mut()
                .and_then(|d| d.tree.planes.as_mut())
                .expect("planes");
            let mut v = tp
                .plan
                .try_slice_mut(0..plan.len())
                .ok_or_else(|| GpuError::Driver("tree plan".into()))?;
            exec.stream.memcpy_htod(&plan, &mut v).map_err(drv)?;
            exec.kv_tree_move(
                (&tp.kp, &tp.vp, &tp.kind),
                n_layers,
                (d_bt, bps),
                (swa_bt, bps),
                &tp.plan,
                plan.len() / MOVE_STRIDE,
                MOVE_STRIDE,
                row_bytes,
            )?;
        }
        self.d2_commit_rows(&rows, &spans)
    }

    /// The compaction planes: the layers' pool bases (stable from
    /// enable_batch on), the move plan, the tap gather's map and landing.
    fn tree_planes(&mut self) -> Result<(), GpuModelError> {
        if self
            .dflash2
            .as_ref()
            .is_some_and(|d| d.tree.planes.is_some())
        {
            return Ok(());
        }
        use cudarc::driver::DevicePtr;
        let exec = self.exec.clone();
        let bs = self.batch.as_ref().expect("batch enabled");
        let (mut kp, mut vp, mut kind) = (Vec::new(), Vec::new(), Vec::new());
        for (l, layer) in self.layers.iter().enumerate() {
            let (k, _g1) = bs.kv[l].k.device_ptr(&exec.stream);
            let (v, _g2) = bs.kv[l].v.device_ptr(&exec.stream);
            kp.push(k);
            vp.push(v);
            kind.push(u32::from(layer.is_swa));
        }
        let n_slots = bs.n_slots;
        let planes = TreePlanes {
            kp: exec.stream.clone_htod(&kp).map_err(drv)?,
            vp: exec.stream.clone_htod(&vp).map_err(drv)?,
            kind: exec.to_device_u32(&kind)?,
            plan: exec.alloc_u32(n_slots * MOVE_STRIDE)?,
        };
        self.dflash2.as_mut().expect("drafter").tree.planes = Some(planes);
        Ok(())
    }
}
