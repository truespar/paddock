//! DFlash2 drafts as a TREE: the selector lattice (every edge of the block's
//! K x K transition matrices, slot 974) grown best-first into a fixed node
//! budget, so a verify round checks several continuations at once instead of
//! the greedy walk's single chain.
//!
//! Policy (the shape Ricursive's own DGX Spark bundle verifies): each edge
//! row is softmaxed at a proposal temperature, a node's children are its
//! row's top candidates, and the tree keeps the `nodes - 1` paths of highest
//! product probability. A child's product never exceeds its parent's, so the
//! best-first frontier pops parents before children: node order is a valid
//! layout (every path's node indices increase with depth), and the kept set
//! is closed under ancestors. The proposal only shapes the tree - accepting
//! a node is the target's call alone (its greedy pick, or its own sample at
//! that node), so the scores never touch the output distribution.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One slot's draft tree. Node 0 is the anchor (the committed pending
/// token); `parent[0] == 0`, `depth[0] == 0`.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parent: Vec<u16>,
    pub depth: Vec<u8>,
    /// bit j set = node j is the node itself or one of its ancestors (the
    /// verify's attention mask over the tree's own keys); budgets <= 32
    pub anc: Vec<u32>,
}

impl DraftTree {
    /// The one-node tree: a slot verified pending-only.
    pub fn anchor(token: u32) -> Self {
        Self {
            tokens: vec![token],
            parent: vec![0],
            depth: vec![0],
            anc: vec![1],
        }
    }

    /// The tree cut to `max_depth`: node order kept, so a kept node's
    /// ancestors (all shallower) are kept and come first.
    pub fn pruned(&self, max_depth: usize) -> Self {
        let mut t = Self::anchor(self.tokens[0]);
        let mut map = vec![0u16; self.tokens.len()];
        for j in 1..self.tokens.len() {
            if self.depth[j] as usize > max_depth {
                continue;
            }
            let n = t.tokens.len();
            let p = map[self.parent[j] as usize];
            map[j] = n as u16;
            t.tokens.push(self.tokens[j]);
            t.parent.push(p);
            t.depth.push(self.depth[j]);
            t.anc.push(t.anc[p as usize] | (1u32 << n));
        }
        t
    }

    /// The deepest path whose tokens the target confirms: `next(node)` is the
    /// target's token after `node` (its pick at that row); from the anchor,
    /// move to the child carrying it while one does. Returns the path's node
    /// indices past the anchor, in depth order.
    pub fn accept(&self, mut next: impl FnMut(usize) -> u32) -> Vec<usize> {
        let mut path = Vec::new();
        let mut at = 0usize;
        loop {
            let want = next(at);
            let child = (at + 1..self.tokens.len())
                .find(|&c| self.parent[c] as usize == at && self.tokens[c] == want);
            match child {
                Some(c) => {
                    path.push(c);
                    at = c;
                }
                None => return path,
            }
        }
    }
}

/// A frontier entry: the path product, then a tie order that keeps the pop
/// deterministic (shallower first, then the parent's earlier pop, then the
/// row's higher-scored candidate).
struct Frontier {
    prob: f64,
    depth: u8,
    parent: u16,
    rank: u8,
    cand: u8,
}

impl PartialEq for Frontier {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Frontier {}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Frontier {
    fn cmp(&self, o: &Self) -> Ordering {
        self.prob
            .total_cmp(&o.prob)
            .then(o.depth.cmp(&self.depth))
            .then(o.parent.cmp(&self.parent))
            .then(o.rank.cmp(&self.rank))
    }
}

/// The edge row a node's children come from: the anchor's at the first
/// position (every predecessor row is the anchor's there, row 0 is read),
/// else its candidate's row one position on.
fn edge_row(scores: &[f32], k: usize, depth: usize, cand: usize) -> &[f32] {
    let p = if depth == 0 { 0 } else { cand };
    &scores[(depth * k + p) * k..(depth * k + p + 1) * k]
}

/// Grow one block's tree. `ids` is `[positions, k]` (the candidates'
/// tokens), `scores` `[positions, k, k]` (predecessor row, candidate
/// column). `max_depth` caps the drafted depth (<= positions), `nodes` the
/// budget with the anchor (<= 32), `children` a node's fan-out, `temp` the
/// proposal temperature.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn build_tree(
    ids: &[u32],
    scores: &[f32],
    anchor: u32,
    k: usize,
    max_depth: usize,
    nodes: usize,
    children: usize,
    temp: f32,
) -> DraftTree {
    let positions = ids.len() / k.max(1);
    debug_assert_eq!(scores.len(), positions * k * k);
    let max_depth = max_depth.min(positions);
    let nodes = nodes.clamp(1, 32);
    let mut t = DraftTree {
        tokens: vec![anchor],
        parent: vec![0],
        depth: vec![0],
        anc: vec![1],
    };
    let mut heap = BinaryHeap::new();
    let expand =
        |heap: &mut BinaryHeap<Frontier>, node: usize, depth: usize, cand: usize, base: f64| {
            if depth >= max_depth {
                return;
            }
            let row = edge_row(scores, k, depth, cand);
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if !m.is_finite() {
                return;
            }
            let w: Vec<f64> = row
                .iter()
                .map(|&v| f64::from((v - m) / temp).exp())
                .collect();
            let z: f64 = w.iter().sum();
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            for (rank, &c) in order.iter().take(children).enumerate() {
                heap.push(Frontier {
                    prob: base * w[c] / z,
                    depth: (depth + 1) as u8,
                    parent: node as u16,
                    rank: rank as u8,
                    cand: c as u8,
                });
            }
        };
    expand(&mut heap, 0, 0, 0, 1.0);
    while t.tokens.len() < nodes {
        let Some(f) = heap.pop() else { break };
        let node = t.tokens.len();
        let (d, c) = (f.depth as usize, f.cand as usize);
        t.tokens.push(ids[(d - 1) * k + c]);
        t.parent.push(f.parent);
        t.depth.push(f.depth);
        t.anc.push(t.anc[f.parent as usize] | (1u32 << node));
        expand(&mut heap, node, d, c, f.prob);
    }
    t
}

impl super::GpuLaguna {
    /// The last draft pass's lattice over its first `n` blocks - the tree
    /// oracle's (and the tree round's) input.
    #[doc(hidden)]
    pub fn spec_last_lattice(
        &mut self,
        n: usize,
    ) -> Result<Vec<(Vec<u32>, Vec<f32>)>, crate::gpu_model::gpt_oss::GpuModelError> {
        self.d2_lattice(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-position, 3-candidate lattice where the chain's argmax path is
    /// confident at the first step and split at the second: the tree spends
    /// its budget on the split.
    fn lattice() -> (Vec<u32>, Vec<f32>) {
        let ids = vec![10, 11, 12, 20, 21, 22, 30, 31, 32];
        let mut s = vec![-10.0f32; 3 * 3 * 3];
        // position 0 (from the anchor): 10 clearly
        s[0] = 5.0;
        s[1] = 0.0;
        s[2] = -1.0;
        // position 1, from candidate 0 (token 10): 20 and 21 nearly tied
        s[9] = 2.0;
        s[10] = 1.9;
        s[11] = -3.0;
        // position 2, from candidates 0 / 1 (tokens 20 / 21)
        s[18] = 3.0;
        s[18 + 3 + 1] = 3.0;
        (ids, s)
    }

    #[test]
    fn best_first_layout() {
        let (ids, s) = lattice();
        let t = build_tree(&ids, &s, 7, 3, 3, 6, 2, 1.0);
        assert_eq!(t.tokens[0], 7);
        assert_eq!(t.tokens.len(), 6);
        // parents before children, ancestors in every mask
        for n in 1..t.tokens.len() {
            let p = t.parent[n] as usize;
            assert!(p < n, "node {n}'s parent {p} after it");
            assert_eq!(t.depth[n], t.depth[p] + 1);
            assert_eq!(t.anc[n], t.anc[p] | (1 << n));
        }
        // the confident first token, then both near-tied seconds
        assert_eq!(t.tokens[1], 10);
        let seconds: Vec<u32> = (1..6)
            .filter(|&n| t.parent[n] == 1)
            .map(|n| t.tokens[n])
            .collect();
        assert!(
            seconds.contains(&20) && seconds.contains(&21),
            "{seconds:?}"
        );
    }

    #[test]
    fn accept_walks_the_target() {
        let (ids, s) = lattice();
        let t = build_tree(&ids, &s, 7, 3, 3, 8, 2, 1.0);
        // the target says 10, then 21, then 31, then something unseen
        let want = |n: usize| match t.tokens[n] {
            7 => 10,
            10 => 21,
            21 => 31,
            _ => 99,
        };
        let path = t.accept(want);
        let toks: Vec<u32> = path.iter().map(|&n| t.tokens[n]).collect();
        assert_eq!(toks, vec![10, 21, 31]);
        // a chain is a tree with one child a node
        let c = build_tree(&ids, &s, 7, 3, 3, 4, 1, 1.0);
        let toks: Vec<u32> = c.tokens[1..].to_vec();
        assert_eq!(toks, vec![10, 20, 30]);
    }
}
