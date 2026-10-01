//! Qwen3.8-Flash-Next prefix cache - the qwen35 / nemotron design over this
//! family's paged live KV (pages.rs).
//!
//! A radix over 16-token KV pages that later prompts adopt by refcount, plus
//! recurrent-state checkpoints taken at a prompt's last two page boundaries
//! (`ckpt_cuts`) and at the reply's page boundaries, so the next turn - the
//! same history re-sent plus a reply and a new message - resumes at the
//! deepest checkpoint under its match and prefills only the divergent tail.
//! The radix holds the pages of the live pool (no copy: a publish retains the
//! slot's pages, a resume adopts them), and a checkpoint is pool pages too:
//! its flat record - per GDN layer the recurrence then the conv window, then
//! the PLE conv ring - spread over whole pages' slots in every attention
//! layer's K, V and index planes (`ckpt_pages::PageLayout`), copied to and
//! from the slot's state by one batched copy. The token stream the PLE n-gram
//! gather hashes is host state and is re-derived from the prompt.
//!
//! A continued conversation is a radix resume like any other: its pages are
//! the slot's own from its last turn (filed by the prompt's publish and the
//! reply checkpoint), the state a copy out of the checkpoint's pages. When
//! the pool needs room the radix gives back dead KV first, then the stalest
//! checkpoint, then LRU KV; the plan guarantees the live turns' checkpoints
//! and full context for every slot, and holds the rest while it can.
//!
//! In-walk checkpoints: a prefill walk writes the state at its cut rows from
//! inside itself (`CkptSink`) into a flat staging blob per cut - pages are
//! not contiguous - and the blob commits into the checkpoint's pages after
//! the walk (`commit_staged`), nemotron's staged cuts.
//!
//! The walk continues a sequence mid-way (`walk_span` with `from > 0`):
//! attention already takes per-row positions against the slot's cache, the
//! recurrence starts from the slot's state, and the two causal convs - which
//! left-pad with zeros at their base row - get their window rows re-staged in
//! front of the span's first rows (`resume_*` in forward.rs), which is the
//! whole-sequence conv bit for bit.

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::ckpt_pages::{Dir, PageLayout};
use crate::gpu::{GpuError, GpuExecutor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::kv_pool::BlockId;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};

use super::pages::KvPages;

/// Engine-wide off switch, honoured by every family.
pub(crate) fn prefix_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
}

/// Don't bother snapshotting checkpoints for prompts shorter than this.
pub(super) const MIN_SNAPSHOT_LEN: usize = 3 * BLOCK_TOKENS;
/// A resume must skip at least this much, or the restore costs more than the
/// rows it saves (two pages).
const MIN_RESUME: usize = 2 * BLOCK_TOKENS;
/// In-walk cuts a walk carries at most (a prompt's two), so staging blobs.
pub(super) const STAGED_CUTS: usize = 2;

/// The checkpoint boundaries for a prompt: its last two full page boundaries,
/// ascending ([0, 0] when the prompt is too short). Two, not one: a re-rendered
/// multi-turn history diverges inside the trailing generation header, and
/// whenever the prompt's final partial page is shorter than that header the
/// divergence crosses the last boundary - a checkpoint only there is
/// unreachable for the next turn (the qwen35 law).
pub(super) fn ckpt_cuts(t_len: usize) -> [usize; 2] {
    if t_len < MIN_SNAPSHOT_LEN {
        return [0, 0];
    }
    let b1 = (t_len - 1) / BLOCK_TOKENS * BLOCK_TOKENS;
    [b1.saturating_sub(BLOCK_TOKENS), b1]
}

/// One checkpoint's record, in f32 elements: per GDN layer, in layer order,
/// the recurrence then its conv window, then the PLE ring.
#[derive(Clone, Copy)]
pub(super) struct CkptGeometry {
    pub(super) st_elems: usize,
    pub(super) win_elems: usize,
    pub(super) ple_elems: usize,
    pub(super) ckpt_f32: usize,
}

impl CkptGeometry {
    pub(super) fn of(cfg: &Qwen4ExpConfig) -> Self {
        let n_gdn = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b, Qwen4ExpBlock::Gdn))
            .count();
        let st_elems = cfg.gdn_v_heads * cfg.gdn_k_dim * cfg.gdn_v_dim;
        let win_elems = (cfg.gdn_conv - 1) * cfg.gdn_qkv_rows();
        let ple_elems = if cfg.ple_layers.is_empty() {
            0
        } else {
            (cfg.ple_conv - 1) * super::forward::PLE_DILATION * cfg.hc_width()
        };
        Self {
            st_elems,
            win_elems,
            ple_elems,
            ckpt_f32: n_gdn * (st_elems + win_elems) + ple_elems,
        }
    }

    pub(super) fn bytes(&self) -> u64 {
        (self.ckpt_f32 * 4) as u64
    }
}

/// The staging blobs as a prefill walk writes in-walk checkpoints into them
/// - the checkpoint record's layout, `idx` a staging blob (`STAGED_CUTS`).
pub(super) struct CkptSink<'a> {
    pub(super) pool: &'a mut CudaSlice<f32>,
    pub(super) ckpt_f32: usize,
    pub(super) st_elems: usize,
    pub(super) win_elems: usize,
}

impl CkptSink<'_> {
    /// Element offset of GDN layer `gdn_ord`'s recurrence in blob `idx`.
    pub(super) fn state_off(&self, idx: u32, gdn_ord: usize) -> usize {
        idx as usize * self.ckpt_f32 + gdn_ord * (self.st_elems + self.win_elems)
    }
    /// Element offset of GDN layer `gdn_ord`'s conv window in blob `idx`.
    pub(super) fn win_off(&self, idx: u32, gdn_ord: usize) -> usize {
        self.state_off(idx, gdn_ord) + self.st_elems
    }
    /// Element offset of the PLE ring in blob `idx` (after all `n_gdn`).
    pub(super) fn ple_off(&self, idx: u32, n_gdn: usize) -> usize {
        idx as usize * self.ckpt_f32 + n_gdn * (self.st_elems + self.win_elems)
    }
}

/// FNV-1a over a prompt's ids: which prompt took an in-walk checkpoint.
fn prompt_hash(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Where the walk that took an in-walk checkpoint began: the prompt (its
/// length and hash) and the checkpoint it resumed from - `None` for a cold
/// walk - named by index AND attach generation, so a later re-send can tell
/// that very checkpoint from a newer one filed at the same boundary.
#[derive(Clone, Copy, PartialEq, Eq)]
struct WalkSrc {
    t_len: usize,
    hash: u64,
    from: Option<Origin>,
}

/// A resume point: boundary, checkpoint index, the generation it was attached
/// under.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Origin {
    pos: usize,
    idx: u32,
    stamp: u64,
}

/// The slot's carried state, which a checkpoint snapshots and restores.
pub(super) struct SlotState<'a> {
    pub(super) recur: &'a mut [Option<CudaSlice<f32>>],
    pub(super) gdn_win: &'a mut [Option<CudaSlice<f32>>],
    pub(super) ple_win: Option<&'a mut CudaSlice<f32>>,
}

pub(super) struct PrefixCache {
    geo: CkptGeometry,
    /// a checkpoint's record over a page's slots in every pool plane
    layout: PageLayout,
    /// `[STAGED_CUTS][ckpt_f32]`: in-walk cuts before they commit into pages
    stage: CudaSlice<f32>,
    /// batched-copy descriptors (src, dst, bytes) x max_descs
    descs: CudaSlice<u64>,
    max_descs: usize,
    last_reused: Vec<usize>,
    stats: bool,
    /// per checkpoint index: the walk that took the checkpoint inside itself,
    /// which an exact re-send of that prompt repeats (see `resume`). None for
    /// a checkpoint a walk boundary took (cut walks, the reply checkpoint).
    src: Vec<Option<WalkSrc>>,
    /// per checkpoint index: the generation of its current attachment. An
    /// index is recycled when its checkpoint is stolen or dropped, so (index,
    /// generation) is what names one checkpoint for as long as it lives.
    gens: Vec<u64>,
    next_gen: u64,
    /// per slot: where its current walk began (`resume`'s verdict, or a
    /// walk boundary `walk_starts` named), which the in-walk cuts it takes
    /// record as their origin
    walk_from: Vec<Option<Origin>>,
    /// per slot: the latest point its state sits at that an exact re-send
    /// can reach again - the resume point (None there = cold) or the last
    /// walk-boundary checkpoint filed for it - and where that is
    repeat_at: Vec<Option<(usize, Option<Origin>)>>,
}

impl PrefixCache {
    /// The cache for `slots` seats over the pool whose planes `layout` names;
    /// `n_ckpt` is the radix's checkpoint index space.
    pub(super) fn new(
        exec: &GpuExecutor,
        geo: CkptGeometry,
        layout: PageLayout,
        slots: usize,
        n_ckpt: u32,
    ) -> Result<Self, GpuModelError> {
        // every segment a checkpoint copy moves stays 16-aligned (the
        // batched copy's contract), which the record's parts must be
        for (what, elems) in [
            ("recurrence", geo.st_elems),
            ("conv window", geo.win_elems),
            ("PLE ring", geo.ple_elems),
        ] {
            if !(elems * 4).is_multiple_of(16) {
                return Err(GpuModelError::Unsupported(format!(
                    "qwen4exp prefix cache: a checkpoint's {what} is {} bytes, not a \
                     16-byte multiple",
                    elems * 4
                )));
            }
        }
        let pages = layout.pages_for(geo.bytes());
        // a triple per plane slot of every page, plus the partial slots each
        // record part opens and closes
        let parts = 2 * geo.ckpt_f32.max(1) / (geo.st_elems + geo.win_elems).max(1) + 2;
        let max_descs = pages * layout.planes().len() + 2 * parts + 8;
        let stage = exec.alloc(STAGED_CUTS * geo.ckpt_f32)?;
        let descs = exec.alloc_u64(3 * max_descs)?;
        tracing::info!(
            "qwen4exp prefix cache: zero-copy radix over the KV pool; a checkpoint is {} MB \
             over {pages} pool pages",
            geo.bytes() >> 20
        );
        Ok(Self {
            geo,
            layout,
            stage,
            descs,
            max_descs,
            last_reused: vec![0; slots],
            stats: paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some(),
            src: vec![None; n_ckpt as usize],
            gens: vec![0; n_ckpt as usize],
            next_gen: 0,
            walk_from: vec![None; slots],
            repeat_at: vec![None; slots],
        })
    }

    /// Stamp checkpoint `idx` with a fresh attach generation.
    fn stamp(&mut self, idx: u32) {
        self.next_gen += 1;
        if let Some(g) = self.gens.get_mut(idx as usize) {
            *g = self.next_gen;
        }
    }

    /// How many leading tokens the last prefill of `slot` took from the cache
    /// (the usage line's `cached_tokens`); cleared on read.
    pub(super) fn take_reused(&mut self, slot: usize) -> usize {
        self.last_reused.get_mut(slot).map_or(0, std::mem::take)
    }

    /// The resume point for `tokens` in `slot`: the deepest checkpoint under
    /// the radix match, the slot's table pointed at the matched pages (no
    /// copy) and its state restored from the checkpoint's pages. 0 = nothing
    /// touched.
    pub(super) fn resume(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        tokens: &[u32],
        pages: &mut KvPages,
        st: SlotState<'_>,
    ) -> Result<usize, GpuModelError> {
        let t_len = tokens.len();
        // cold unless a checkpoint below says otherwise
        self.walk_from[slot] = None;
        self.repeat_at[slot] = Some((0, None));
        let Some((radix, _)) = pages.radix_pool() else {
            return Ok(0);
        };
        let m = radix.match_full(tokens);
        if self.stats {
            // the whole resident checkpoint set, so a shallow resume can be
            // told apart from a divergent prompt (the match) and from a
            // stolen checkpoint (the set)
            let mut at: Vec<usize> = radix
                .state_attachments()
                .iter()
                .map(|&(d, _, _)| d * BLOCK_TOKENS)
                .collect();
            at.sort_unstable();
            tracing::info!(
                "qwen4exp-match: slot {slot} t_len {t_len} matched {} tok, ckpt {:?}, resident {at:?}",
                m.blocks.len() * BLOCK_TOKENS,
                m.ckpt.map(|c| c.0)
            );
        }
        let Some((mut pos, mut idx)) = m.ckpt else {
            return Ok(0);
        };
        // An exact re-send of the prompt that took this checkpoint inside its
        // own walk repeats that walk: resuming at the in-walk cut would replay
        // rows the first run computed inside one walk through a shorter one,
        // and the two agree only to the last ulp (the in-walk checkpoint
        // trade, chosen 2026-09-15: the re-send stays bit-identical to its
        // first run). So it starts where the first run started - cold when
        // that run was cold, and at the very checkpoint it resumed from when
        // that one still stands (same boundary, same index, same attach
        // generation, pages under it held by the path). Before 2026-09-30
        // every exact re-send went cold, which on a long conversation is the
        // whole context: a retried 177K-token turn re-prefilled 177K tokens
        // (201 s) instead of its own 11.7K-token tail.
        let src = self.src.get(idx as usize).copied().flatten();
        if let Some(src) = src.filter(|s| s.t_len == t_len && s.hash == prompt_hash(tokens)) {
            let again = src.from.filter(|o| {
                o.pos >= MIN_RESUME
                    && o.pos < t_len
                    && m.blocks.len() * BLOCK_TOKENS >= o.pos
                    && self.gens.get(o.idx as usize) == Some(&o.stamp)
            });
            match again.filter(|o| radix.resume_state_at(tokens, o.pos) == Some(o.idx)) {
                Some(o) => {
                    if self.stats {
                        tracing::info!(
                            "qwen4exp-resume: t_len {t_len} is an exact re-send - repeating its \
                             first walk from {}",
                            o.pos
                        );
                    }
                    (pos, idx) = (o.pos, o.idx);
                }
                None => {
                    if self.stats {
                        tracing::info!(
                            "qwen4exp-resume: t_len {t_len} is an exact re-send - cold (its first \
                             walk {})",
                            if src.from.is_some() {
                                "resumed from a checkpoint that is gone"
                            } else {
                                "was cold"
                            }
                        );
                    }
                    return Ok(0);
                }
            }
        }
        if pos < MIN_RESUME || pos >= t_len || m.blocks.len() * BLOCK_TOKENS < pos {
            if self.stats {
                tracing::info!(
                    "qwen4exp-resume: t_len {t_len} matched {} tok, ckpt {pos} - not resumable",
                    m.blocks.len() * BLOCK_TOKENS
                );
            }
            return Ok(0);
        }
        pages.adopt(slot, &m.blocks[..pos / BLOCK_TOKENS]);
        let ck: Vec<BlockId> = pages
            .radix
            .as_ref()
            .expect("radix present")
            .state_pages(idx)
            .to_vec();
        self.state_copy(exec, slot, &ck, st, Dir::FromPages)?;
        self.last_reused[slot] = pos;
        let origin = Origin {
            pos,
            idx,
            stamp: self.gens.get(idx as usize).copied().unwrap_or(0),
        };
        self.walk_from[slot] = Some(origin);
        self.repeat_at[slot] = Some((pos, Some(origin)));
        if self.stats {
            tracing::info!(
                "qwen4exp-resume: slot {slot} t_len {t_len} matched {} tok, resumed at {pos} \
                 (ckpt {idx})",
                m.blocks.len() * BLOCK_TOKENS
            );
        }
        Ok(pos)
    }

    /// After the walk reached `upto` tokens of `tokens` in `slot`: file every
    /// full page up to there under the radix (retaining the slot's pages -
    /// the path's existing nodes keep theirs) and, when asked, attach a state
    /// checkpoint at `upto` (a page boundary) and snapshot the slot's carried
    /// state into its pages. Returns the checkpoint's index when one was
    /// attached (the reply checkpoint tracks its own for the detach).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn publish(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        tokens: &[u32],
        upto: usize,
        snapshot: bool,
        pages: &mut KvPages,
        st: SlotState<'_>,
    ) -> Result<Option<u32>, GpuModelError> {
        let full = upto / BLOCK_TOKENS;
        if full == 0 {
            return Ok(None);
        }
        let blocks: Vec<BlockId> = pages.blocks(slot)[..full].to_vec();
        let Some((radix, pool)) = pages.radix_pool() else {
            return Ok(None);
        };
        radix.insert(&tokens[..full * BLOCK_TOKENS], &blocks, pool);
        if !(snapshot && upto.is_multiple_of(BLOCK_TOKENS)) {
            return Ok(None);
        }
        let Some(idx) = radix.attach_state_with_pool(tokens, upto, pool) else {
            return Ok(None);
        };
        let ck = radix.state_pages(idx).to_vec();
        self.state_copy(exec, slot, &ck, st, Dir::ToPages)?;
        if let Some(s) = self.src.get_mut(idx as usize) {
            *s = None;
        }
        self.stamp(idx);
        // the slot's state right now, filed at a walk boundary: a re-send
        // that resumes here walks on from the very same state
        if let Some(r) = self.repeat_at.get_mut(slot) {
            *r = Some((
                upto,
                Some(Origin {
                    pos: upto,
                    idx,
                    stamp: self.gens[idx as usize],
                }),
            ));
        }
        if self.stats {
            tracing::info!("qwen4exp-ckpt: slot {slot} cut {upto} idx {idx}");
        }
        Ok(Some(idx))
    }

    /// `slot`'s next walk starts at `pos` and takes checkpoint cuts inside
    /// itself: name where an exact re-send can start that walk again - the
    /// slot's resume point, or the checkpoint its last walk boundary filed -
    /// as the origin those cuts record. False when neither sits at `pos`: the
    /// caller files one there and asks again, and if that fails too the
    /// cuts record no origin and a re-send of the prompt goes cold, as when
    /// an origin is gone.
    ///
    /// Why the walk's own start and not the prompt's: a long prompt walks in
    /// pieces (`walk_rows`), and only the last carries the cuts. Named by the
    /// resume point, a re-send repeated every piece - the whole prompt when it
    /// had come in cold (a 30K-token re-send re-prefilled 30K tokens, 21.8 s).
    /// Named by a checkpoint at the last piece's start - a boundary the walk
    /// has anyway, so filing it costs a state copy, not a walk - it repeats
    /// that piece alone. And a prompt whose earlier rows rode ticks beside
    /// other slots' decode rows is only repeatable from a boundary after them.
    pub(super) fn walk_starts(&mut self, slot: usize, pos: usize) -> bool {
        match self.repeat_at.get(slot).copied().flatten() {
            Some((at, origin)) if at == pos => {
                self.walk_from[slot] = origin;
                true
            }
            _ => {
                self.walk_from[slot] = None;
                false
            }
        }
    }

    /// The staging blobs as a prefill walk writes in-walk checkpoints.
    pub(super) fn ckpt_sink(&mut self) -> CkptSink<'_> {
        CkptSink {
            pool: &mut self.stage,
            ckpt_f32: self.geo.ckpt_f32,
            st_elems: self.geo.st_elems,
            win_elems: self.geo.win_elems,
        }
    }

    /// A checkpoint index - its pages drawn from the pool - for a checkpoint
    /// the next walk writes from inside itself (commit its staging blob with
    /// [`Self::commit_staged`], attach it with [`Self::attach_reserved`] once
    /// the pages up to its cut are filed, or give it back with
    /// [`Self::recycle_ckpt`]).
    pub(super) fn reserve_ckpt(&mut self, pages: &mut KvPages) -> Option<u32> {
        let (radix, pool) = pages.radix_pool()?;
        radix.reserve_state_slot_with_pool(pool)
    }

    /// Commit staging blob `blob` (an in-walk cut the last walk wrote) into
    /// checkpoint `idx`'s pages.
    pub(super) fn commit_staged(
        &mut self,
        exec: &GpuExecutor,
        pages: &KvPages,
        blob: usize,
        idx: u32,
    ) -> Result<(), GpuModelError> {
        let ck = pages
            .radix
            .as_ref()
            .expect("radix present")
            .state_pages(idx);
        let bytes = self.geo.bytes();
        let mut descs = Vec::with_capacity(3 * self.max_descs);
        {
            let (sp, _g) = self.stage.device_ptr(&exec.stream);
            self.layout.push_copy(
                ck,
                0,
                sp + blob as u64 * bytes,
                bytes,
                Dir::ToPages,
                &mut descs,
            );
        }
        self.run_descs(exec, &descs)
    }

    /// Attach reserved checkpoint `idx` at `cut` of `tokens`, recording the
    /// prompt that took it; on a miss (the node is gone or already
    /// checkpointed) the index and its pages go back.
    pub(super) fn attach_reserved(
        &mut self,
        slot: usize,
        tokens: &[u32],
        cut: usize,
        idx: u32,
        pages: &mut KvPages,
    ) -> bool {
        let src = Some(WalkSrc {
            t_len: tokens.len(),
            hash: prompt_hash(tokens),
            from: self.walk_from.get(slot).copied().flatten(),
        });
        self.attach_staged(tokens, cut, idx, pages, src, "in-walk cut")
    }

    /// Attach reserved checkpoint `idx` - a reply checkpoint a speculative
    /// round rebuilt, already committed into its pages - at `cut` of the
    /// slot's sequence. No prompt took it inside its own walk, so a re-send
    /// resumes from it; on a miss the index and its pages go back.
    pub(super) fn attach_reply(
        &mut self,
        tokens: &[u32],
        cut: usize,
        idx: u32,
        pages: &mut KvPages,
    ) -> bool {
        self.attach_staged(tokens, cut, idx, pages, None, "in-round reply cut")
    }

    fn attach_staged(
        &mut self,
        tokens: &[u32],
        cut: usize,
        idx: u32,
        pages: &mut KvPages,
        src: Option<WalkSrc>,
        what: &str,
    ) -> bool {
        let Some((radix, pool)) = pages.radix_pool() else {
            return false;
        };
        if radix.attach_state_at(tokens, cut, idx) {
            if let Some(s) = self.src.get_mut(idx as usize) {
                *s = src;
            }
            self.stamp(idx);
            if self.stats {
                tracing::info!("qwen4exp-ckpt: {what} {cut} idx {idx}");
            }
            true
        } else {
            radix.recycle_state(idx);
            radix.reclaim(pool);
            false
        }
    }

    /// Give back a reserved index that was never attached, and its pages.
    pub(super) fn recycle_ckpt(&mut self, idx: u32, pages: &mut KvPages) {
        if let Some((radix, pool)) = pages.radix_pool() {
            radix.recycle_state(idx);
            radix.reclaim(pool);
        }
    }

    /// Drop the checkpoint at `cut` of `tokens` - but only if it is still
    /// `idx`: the radix may have evicted it and handed the index to another
    /// prompt since, and another slot walking the same sequence may have
    /// re-checkpointed the node, neither of which is ours to detach.
    pub(super) fn drop_ckpt(&mut self, tokens: &[u32], cut: usize, idx: u32, pages: &mut KvPages) {
        let Some((radix, pool)) = pages.radix_pool() else {
            return;
        };
        if radix.detach_state_if(tokens, cut, idx) {
            radix.recycle_state(idx);
            radix.reclaim(pool);
        }
    }

    /// The slot's carried state <-> checkpoint pages `ck`, one batched copy:
    /// each part of the record (per GDN layer the recurrence then the conv
    /// window, then the PLE ring) at its offset in the record.
    fn state_copy(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        ck: &[BlockId],
        st: SlotState<'_>,
        dir: Dir,
    ) -> Result<(), GpuModelError> {
        let (st_elems, win_elems, ple_elems) =
            (self.geo.st_elems, self.geo.win_elems, self.geo.ple_elems);
        let layout = &self.layout;
        let mut descs: Vec<u64> = Vec::with_capacity(3 * self.max_descs);
        let mut off = 0u64;
        let mut part = |descs: &mut Vec<u64>, base: u64, elems: usize| {
            let len = (elems * 4) as u64;
            layout.push_copy(ck, off, base + (slot * elems * 4) as u64, len, dir, descs);
            off += len;
        };
        for li in 0..st.recur.len() {
            let Some(r) = st.recur[li].as_ref() else {
                continue;
            };
            let w = st.gdn_win[li]
                .as_ref()
                .expect("gdn layer has a conv window");
            let (rp, _g1) = r.device_ptr(&exec.stream);
            let (wp, _g2) = w.device_ptr(&exec.stream);
            part(&mut descs, rp, st_elems);
            part(&mut descs, wp, win_elems);
        }
        if ple_elems > 0 {
            let w = st.ple_win.expect("model has a PLE ring");
            let (wp, _g3) = w.device_ptr(&exec.stream);
            part(&mut descs, wp, ple_elems);
        }
        self.run_descs(exec, &descs)
    }

    fn run_descs(&mut self, exec: &GpuExecutor, descs: &[u64]) -> Result<(), GpuModelError> {
        for chunk in descs.chunks(3 * self.max_descs) {
            let n = chunk.len() / 3;
            {
                let mut v = self.descs.slice_mut(0..chunk.len());
                exec.stream
                    .memcpy_htod(chunk, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            exec.batched_copy(&self.descs, n)?;
        }
        Ok(())
    }
}
