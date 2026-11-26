//! Inkling 2's prefix cache: a `PagedRadix` over the full visits' pool
//! blocks (zero copy - a hit adopts the cached blocks by refcount) plus
//! checkpoints of the state the radix cannot hold, so a prompt resumes at a
//! checkpointed block boundary instead of position 0.
//!
//! **A checkpoint** is everything a row at its boundary reads besides the
//! full visits' K / V: every visit's four short-conv carried rows (the last
//! three inputs of each, f32 - 3.8 MB), the 40 sliding visits' K / V rows of
//! the WINDOW before it (84 MB at fp8), and the DSpark draft's two local
//! layers' rows of the same window (bf16, 8 MB). The draft's full layer rides
//! the same pool blocks as the target's, and its feature ring needs nothing:
//! a draft round reads the last committed row's features, which the resumed
//! tail rewrites.
//!
//! **Cuts.** A prompt checkpoints at its last block boundary - the position
//! the next agentic turn resumes from: the template renders no generation
//! prompt, so a turn's prompt is the previous one plus the reply as re-sent
//! plus the new messages - and at the back-off boundary ~256 tokens behind it
//! (the same document, another question), plus the scheduler's divergence
//! hints. The conv rows are live state that every row advances, so a cut
//! inside a chunk is captured MID-PASS (slot 1014, before each conv
//! overwrites its state); the window rows are landed from the rings right
//! after the pass (a ring holds the window plus a whole chunk). Staged
//! checkpoints attach once the prompt's pages are in the radix.
//!
//! **Replies.** A slot whose prompt landed tracks every token it is fed and
//! checkpoints each block boundary it reaches exactly - a decode tick's row,
//! or a speculative round once its rollback settled - keeping one rolling
//! checkpoint (the previous goes back to the pool), so a next turn that
//! re-sends the reply as generated resumes near its end. When the reply's
//! first tool call opens, the live checkpoint is pinned: clients re-serialize
//! calls, so the reasoning before one is the part a next request shares.
//!
//! Picture prompts stay out: every picture row carries the same placeholder
//! id, so two pictures under the same text would key the same path.
//!
//! Sharing needs no copy-on-write: checkpoints sit on block boundaries and a
//! resume adopts only the blocks before its boundary, so no shared block is
//! ever written.

use cudarc::driver::{CudaSlice, DevicePtr};
use paddock_models::inkling2 as ik;

use super::GpuInkling2;
use super::KV;
use super::kv::{CONV_ATTN, Kv, full_visit};
use crate::gpu::{GpuError, GpuExecutor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::{BLOCK_TOKENS, backoff_cut, reply_ckpt_disabled};
use crate::paged_radix::PagedRadix;

/// Don't resume prefixes shorter than this (the restore beats the prefill).
pub(crate) const MIN_CACHE_PREFIX: usize = 64;
/// Don't checkpoint prompts shorter than this.
const MIN_SNAPSHOT_LEN: usize = 4 * BLOCK_TOKENS;
/// Checkpoints the pool holds a seated slot (a prompt lands up to two, a
/// reply one rolling plus its pins), between a floor and a cap: the pool is a
/// flat reserve (~96 MB a checkpoint), and past four sessions the radix
/// recycles the stalest instead of the reserve growing with the slots.
const CKPTS_PER_SLOT: usize = 4;
const CKPTS_MIN: usize = 8;
const CKPTS_MAX: usize = 16;

fn ckpts(slots: usize) -> usize {
    (CKPTS_PER_SLOT * slots).clamp(CKPTS_MIN, CKPTS_MAX)
}
/// Mid-pass captures a pass can carry.
const MAX_CAPTURES: usize = 256;
/// Window rows a checkpoint keeps before its boundary: a row reads the
/// WINDOW newest keys, its own included.
const WIN_ROWS: usize = ik::WINDOW;

/// f32s one visit's four conv states take: K and V (KV wide), the
/// attention and MLP branches (WIDTH), three rows each.
const VISIT_CONV: usize = 3 * (2 * KV + 2 * ik::WIDTH);
/// f32s a checkpoint's conv states take.
pub(crate) const CONV_FLOATS: usize = ik::VISITS * VISIT_CONV;

/// Where conv `c` of visit `vi` starts inside a checkpoint's conv states.
pub(crate) fn conv_off(vi: usize, c: usize) -> usize {
    let lead: usize = (0..c).map(conv_width).sum();
    vi * VISIT_CONV + 3 * lead
}

/// Channels conv `c` carries.
pub(crate) fn conv_width(c: usize) -> usize {
    if c < CONV_ATTN { KV } else { ik::WIDTH }
}

/// Bytes a K / V row takes in the store class `dtype`.
fn row_bytes(dtype: u32) -> usize {
    KV * match dtype {
        1 => 1,
        2 => 4,
        _ => 2,
    }
}

/// Window bytes one checkpoint takes: the sliding visits' K and V rows and,
/// with the draft, its two local layers' (bf16).
fn win_bytes(dtype: u32, draft: bool) -> usize {
    let sliding = (0..ik::VISITS).filter(|&v| !full_visit(v)).count();
    let d = if draft {
        2 * 2 * WIN_ROWS * row_bytes(3)
    } else {
        0
    };
    sliding * 2 * WIN_ROWS * row_bytes(dtype) + d
}

/// Device bytes the prefix cache claims at `slots` seated slots - the
/// reserve `enable_batch` plans before the KV pool (0 = off).
pub(crate) fn reserve_bytes(exec: &GpuExecutor, slots: usize, dtype: u32, draft: bool) -> u64 {
    if !enabled(exec) {
        return 0;
    }
    let n = ckpts(slots);
    (n * (CONV_FLOATS * 4 + win_bytes(dtype, draft)) + MAX_CAPTURES * 16) as u64
}

fn enabled(exec: &GpuExecutor) -> bool {
    exec.has_inkling2_conv_capture()
        && paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_none()
}

/// A replying slot's cache state: every token its sequence holds (the
/// prompt, then each one it was fed) and its rolling checkpoint.
pub(crate) struct Reply {
    seq: Vec<u32>,
    live: Option<(usize, u32)>,
}

pub(crate) struct Prefix {
    pub radix: PagedRadix,
    /// [n][CONV_FLOATS] f32 and [n][win] bytes
    conv: CudaSlice<f32>,
    win_buf: CudaSlice<u8>,
    win: usize,
    /// a pass's mid-pass captures: {r0, kk, slot, conv offset} each
    cap_items: CudaSlice<u32>,
    pub cap_n: usize,
    pub last_reused: Vec<usize>,
    /// per slot: the cuts its queued prompt checkpoints, the checkpoints
    /// captured so far (boundary, index) awaiting the prompt's insert
    pub cuts: Vec<Vec<usize>>,
    pub staged: Vec<Vec<(usize, u32)>>,
    pub reply: Vec<Option<Reply>>,
}

impl Prefix {
    /// The cache for `slots` slots over `kv` (None when off).
    pub(crate) fn build(
        exec: &GpuExecutor,
        slots: usize,
        dtype: u32,
        draft: bool,
    ) -> Result<Option<Self>, GpuModelError> {
        if !enabled(exec) {
            return Ok(None);
        }
        let n = ckpts(slots);
        let win = win_bytes(dtype, draft);
        let mut radix = PagedRadix::new();
        radix.set_state_capacity(n as u32);
        Ok(Some(Self {
            radix,
            conv: exec.alloc(n * CONV_FLOATS)?,
            win_buf: exec.alloc_u8(n * win)?,
            win,
            cap_items: exec.alloc_u32(4 * MAX_CAPTURES)?,
            cap_n: 0,
            last_reused: vec![0; slots],
            cuts: vec![Vec::new(); slots],
            staged: vec![Vec::new(); slots],
            reply: (0..slots).map(|_| None).collect(),
        }))
    }

    /// The cuts a prompt of `len` tokens checkpoints: its last block
    /// boundary, the back-off boundary behind it and the scheduler's hints
    /// (block-aligned, inside the prompt), ascending.
    pub(crate) fn cuts_for(len: usize, hints: &[usize]) -> Vec<usize> {
        if len < MIN_SNAPSHOT_LEN {
            return Vec::new();
        }
        let last = len / BLOCK_TOKENS * BLOCK_TOKENS;
        let mut cuts: Vec<usize> = hints
            .iter()
            .map(|&h| h / BLOCK_TOKENS * BLOCK_TOKENS)
            .filter(|&h| h >= MIN_CACHE_PREFIX && h < len)
            .collect();
        cuts.push(last);
        if let Some(b) = backoff_cut(last, BLOCK_TOKENS).filter(|&b| b >= MIN_CACHE_PREFIX) {
            cuts.push(b);
        }
        cuts.sort_unstable();
        cuts.dedup();
        cuts
    }

    /// End `slot`'s reply tracking (its checkpoints stay in the radix).
    pub(crate) fn reply_release(&mut self, slot: usize) {
        self.reply[slot] = None;
    }

    /// Hand back the checkpoints `slot`'s dropped prompt had staged.
    pub(crate) fn drop_staged(&mut self, slot: usize) {
        for (_, idx) in std::mem::take(&mut self.staged[slot]) {
            self.radix.recycle_state(idx);
        }
        self.cuts[slot].clear();
    }
}

/// Slot 1014 for conv `c` of visit `vi` over its input `x` (`k` partials,
/// row stride `xstride`) - every capture of the pass in flight, before the
/// conv rewrites `state`.
pub(crate) fn capture(
    e: &GpuExecutor,
    px: Option<&mut Prefix>,
    vi: usize,
    c: usize,
    x: (&CudaSlice<f32>, usize, usize),
    state: &CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    let Some(p) = px.filter(|p| p.cap_n > 0) else {
        return Ok(());
    };
    e.inkling2_conv_capture(
        x,
        conv_width(c),
        state,
        (&p.cap_items, p.cap_n),
        &mut p.conv,
        conv_off(vi, c),
    )?;
    Ok(())
}

/// Back `slot`'s rows up to `upto` with blocks, evicting from the prefix
/// cache when the pool runs short.
pub(crate) fn ensure_rows(
    e: &GpuExecutor,
    kv: &mut Kv,
    px: Option<&mut Prefix>,
    slot: usize,
    upto: usize,
) -> Result<(), GpuModelError> {
    match kv.ensure(e, slot, upto) {
        Err(GpuModelError::PoolExhausted) => {
            let Some(p) = px else {
                return Err(GpuModelError::PoolExhausted);
            };
            let need = upto.div_ceil(BLOCK_TOKENS) - kv.tables[slot].blocks().len();
            if !p.radix.make_room(&mut kv.pool, need, 0) {
                return Err(GpuModelError::PoolExhausted);
            }
            kv.ensure(e, slot, upto)
        }
        r => r,
    }
}

/// `(src, dst, bytes)` copies between `slot`'s rings and checkpoint `idx`'s
/// window for the rows before `pos` (`to_ckpt` picks the direction): each
/// sliding visit's K then V, then the draft's local layers', every plane's
/// rows wrapped at the ring.
fn window_copies(
    kv: &Kv,
    p: &Prefix,
    e: &GpuExecutor,
    slot: usize,
    pos: usize,
    idx: u32,
    to_ckpt: bool,
) -> Vec<u64> {
    let count = pos.min(WIN_ROWS);
    let first = pos - count;
    let (wp, _g) = p.win_buf.device_ptr(&e.stream);
    let mut at = wp + (idx as usize * p.win) as u64;
    let mut out = Vec::new();
    let mut plane = |base: u64, rb: usize, out: &mut Vec<u64>| {
        // [first, pos) at ring rows slot * ring + q % ring, in up to two runs
        let mut q = first;
        while q < pos {
            let r = q % kv.ring;
            let run = (kv.ring - r).min(pos - q);
            let ring_at = base + ((slot * kv.ring + r) * rb) as u64;
            let ckpt_at = at + ((q - first) * rb) as u64;
            let (src, dst) = if to_ckpt {
                (ring_at, ckpt_at)
            } else {
                (ckpt_at, ring_at)
            };
            out.extend([src, dst, (run * rb) as u64]);
            q += run;
        }
        // the layout reserves the whole window a plane
        at += (WIN_ROWS * rb) as u64;
    };
    let rb = row_bytes(kv.dtype);
    for (v, store) in kv.stores.iter().enumerate() {
        if full_visit(v) {
            continue;
        }
        for s in [&store.k, &store.v] {
            let (sp, _g2) = s.device_ptr(&e.stream);
            plane(sp, rb, &mut out);
        }
    }
    // the draft's local layers (0 and 2; 1 is paged)
    for store in kv.draft.iter().step_by(2) {
        for s in [&store.k, &store.v] {
            let (sp, _g2) = s.device_ptr(&e.stream);
            plane(sp, row_bytes(3), &mut out);
        }
    }
    out
}

/// `(src, dst, bytes)` copies between `slot`'s conv states and checkpoint
/// `idx`'s (`to_ckpt` picks the direction).
fn conv_copies(
    kv: &Kv,
    p: &Prefix,
    e: &GpuExecutor,
    slot: usize,
    idx: u32,
    to_ckpt: bool,
) -> Vec<u64> {
    let (cp, _g) = p.conv.device_ptr(&e.stream);
    let base = cp + (idx as usize * CONV_FLOATS * 4) as u64;
    let mut out = Vec::with_capacity(ik::VISITS * 4 * 3);
    for (vi, planes) in kv.conv.iter().enumerate() {
        for (c, pl) in planes.iter().enumerate() {
            let cw = conv_width(c);
            let (sp, _g2) = pl.device_ptr(&e.stream);
            let live = sp + (slot * 3 * cw * 4) as u64;
            let ck = base + (conv_off(vi, c) * 4) as u64;
            let (src, dst) = if to_ckpt { (live, ck) } else { (ck, live) };
            out.extend([src, dst, (3 * cw * 4) as u64]);
        }
    }
    out
}

fn run_copies(e: &GpuExecutor, descs: &[u64]) -> Result<(), GpuError> {
    if descs.is_empty() {
        return Ok(());
    }
    let d = e
        .stream
        .clone_htod(descs)
        .map_err(|err| GpuError::Driver(err.to_string()))?;
    e.batched_copy(&d, descs.len() / 3)
}

impl GpuInkling2 {
    /// Try to resume `tokens` on `slot` (its old rows already released):
    /// adopt the cached blocks up to the deepest checkpoint and restore the
    /// checkpoint's conv states and windows. The resume position (0 =
    /// cold - the caller zeroes the convs).
    pub(crate) fn prefix_resume(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<usize, GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let kv = &mut self.kv;
        let Some(p) = self.prefix.as_mut() else {
            return Ok(0);
        };
        p.last_reused[slot] = 0;
        let m = p.radix.match_full(tokens);
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!(
                "inkling2-prefix: slot {slot} len {} matched {} ckpt {:?}",
                tokens.len(),
                m.blocks.len() * BLOCK_TOKENS,
                m.ckpt.map(|(p, _)| p)
            );
        }
        let Some((pos, idx)) = m.ckpt else {
            return Ok(0);
        };
        if pos < MIN_CACHE_PREFIX || pos >= tokens.len() {
            return Ok(0);
        }
        let nb = pos / BLOCK_TOKENS;
        kv.tables[slot].clear(&mut kv.pool);
        kv.tables[slot].share_prefix(&m.blocks[..nb], &mut kv.pool);
        kv.upload_row(e, slot)?;
        let mut descs = conv_copies(kv, p, e, slot, idx, false);
        descs.extend(window_copies(kv, p, e, slot, pos, idx, false));
        run_copies(e, &descs)?;
        kv.lens[slot] = pos;
        p.last_reused[slot] = pos;
        Ok(pos)
    }

    /// Before a pass: reserve a checkpoint for every cut that a chunk in
    /// `chunks` (`(slot, first position, rows, first pass row)`) reaches, and
    /// arm the mid-pass captures. Returns the (slot, boundary, index) the
    /// pass lands, for [`Self::prefix_land`].
    pub(crate) fn prefix_arm(
        &mut self,
        chunks: &[(usize, usize, usize, usize)],
    ) -> Result<Vec<(usize, usize, u32)>, GpuModelError> {
        let exec = self.exec.clone();
        let Some(p) = self.prefix.as_mut() else {
            return Ok(Vec::new());
        };
        let mut items = Vec::new();
        let mut landing = Vec::new();
        for &(slot, done, rows, r0) in chunks {
            let end = done + rows;
            for &cut in &p.cuts[slot] {
                if cut <= done || cut > end || landing.len() >= MAX_CAPTURES {
                    continue;
                }
                let Some(idx) = p.radix.reserve_state_slot() else {
                    continue;
                };
                items.extend([
                    r0 as u32,
                    (cut - done) as u32,
                    slot as u32,
                    idx * CONV_FLOATS as u32,
                ]);
                landing.push((slot, cut, idx));
            }
        }
        p.cap_n = landing.len();
        if !items.is_empty() {
            exec.upload_u32(&items, &mut p.cap_items)?;
        }
        Ok(landing)
    }

    /// After the pass armed by [`Self::prefix_arm`]: land each captured
    /// checkpoint's windows from the rings (the pass wrote at most a chunk
    /// past its boundary; a ring holds the window plus a chunk) and stage
    /// it for its prompt's insert.
    pub(crate) fn prefix_land(
        &mut self,
        landing: &[(usize, usize, u32)],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let kv = &self.kv;
        let Some(p) = self.prefix.as_mut() else {
            return Ok(());
        };
        p.cap_n = 0;
        let mut descs = Vec::new();
        for &(slot, cut, idx) in landing {
            descs.extend(window_copies(kv, p, e, slot, cut, idx, true));
            p.staged[slot].push((cut, idx));
        }
        run_copies(e, &descs)?;
        Ok(())
    }

    /// `slot`'s prompt `tokens` landed: its pages into the radix, its staged
    /// checkpoints attached, and its reply tracked from here.
    pub(crate) fn prefix_insert(&mut self, slot: usize, tokens: &[u32]) {
        let kv = &mut self.kv;
        let Some(p) = self.prefix.as_mut() else {
            return;
        };
        p.cuts[slot].clear();
        let blocks = kv.tables[slot].blocks().to_vec();
        p.radix.insert_for(slot, tokens, &blocks, &mut kv.pool);
        for (pos, idx) in std::mem::take(&mut p.staged[slot]) {
            if !p.radix.attach_state_at(tokens, pos, idx) {
                p.radix.recycle_state(idx);
            }
        }
        p.reply[slot] = (!reply_ckpt_disabled()).then(|| Reply {
            seq: tokens.to_vec(),
            live: None,
        });
    }

    /// `ids` were fed to `slot` and it now holds `kv.lens[slot]` rows, its
    /// conv states exactly there (`exact`: a decode row, or a speculative
    /// run already settled): track them, and checkpoint a block boundary.
    pub(crate) fn reply_fed(
        &mut self,
        slot: usize,
        ids: &[u32],
        exact: bool,
    ) -> Result<(), GpuModelError> {
        let len = self.kv.lens[slot];
        let Some(p) = self.prefix.as_mut() else {
            return Ok(());
        };
        let Some(r) = p.reply[slot].as_mut() else {
            return Ok(());
        };
        r.seq.extend_from_slice(ids);
        if r.seq.len() != len {
            // a gap in what the slot was fed: stop tracking
            p.reply[slot] = None;
            return Ok(());
        }
        if exact && len.is_multiple_of(BLOCK_TOKENS) && len >= MIN_SNAPSHOT_LEN {
            self.reply_snapshot(slot)?;
        }
        Ok(())
    }

    /// A speculative run settled `slot` at `pos` rows (its conv states
    /// rolled back exactly there): drop the rejected tail of what it was
    /// fed, and checkpoint a block boundary.
    pub(crate) fn reply_settled(&mut self, slot: usize, pos: usize) -> Result<(), GpuModelError> {
        let Some(p) = self.prefix.as_mut() else {
            return Ok(());
        };
        let Some(r) = p.reply[slot].as_mut() else {
            return Ok(());
        };
        if r.seq.len() < pos {
            p.reply[slot] = None;
            return Ok(());
        }
        r.seq.truncate(pos);
        if pos.is_multiple_of(BLOCK_TOKENS) && pos >= MIN_SNAPSHOT_LEN {
            self.reply_snapshot(slot)?;
        }
        Ok(())
    }

    /// Checkpoint `slot` at the block boundary it holds: its pages so far
    /// into the radix, a checkpoint of its live state, the previous rolling
    /// one back to the pool.
    fn reply_snapshot(&mut self, slot: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let kv = &mut self.kv;
        let pos = kv.lens[slot];
        let Some(p) = self.prefix.as_mut() else {
            return Ok(());
        };
        let Some(r) = p.reply[slot].as_mut() else {
            return Ok(());
        };
        let seq = r.seq[..pos].to_vec();
        let blocks = kv.tables[slot].blocks()[..pos / BLOCK_TOKENS].to_vec();
        p.radix.insert_for(slot, &seq, &blocks, &mut kv.pool);
        let Some(idx) = p.radix.attach_state(&seq, pos) else {
            return Ok(());
        };
        let mut descs = conv_copies(kv, p, e, slot, idx, true);
        descs.extend(window_copies(kv, p, e, slot, pos, idx, true));
        run_copies(e, &descs)?;
        let r = p.reply[slot].as_mut().expect("checked above");
        if let Some((prev, pidx)) = r.live.replace((pos, idx))
            && p.radix.detach_state_if(&seq[..prev], prev, pidx)
        {
            p.radix.recycle_state(pidx);
        }
        Ok(())
    }

    /// The reply's first tool call opened: keep the live checkpoint (the
    /// reasoning before the call) - the next one no longer replaces it.
    pub(crate) fn reply_pin_impl(&mut self, slot: usize) {
        if let Some(r) = self
            .prefix
            .as_mut()
            .and_then(|p| p.reply.get_mut(slot))
            .and_then(Option::as_mut)
        {
            r.live = None;
        }
    }

    /// Gates only: `slot`'s conv states (every visit's four, checkpoint
    /// layout) and its window rows before its length (sliding visits' and the
    /// draft's local layers', checkpoint layout) - what a checkpoint of it
    /// would hold.
    #[doc(hidden)]
    pub fn slot_state(&self, slot: usize) -> Result<(Vec<f32>, Vec<u8>), GpuModelError> {
        let e = &*self.exec;
        let kv = &self.kv;
        let mut conv = Vec::with_capacity(CONV_FLOATS);
        for planes in &kv.conv {
            for (c, pl) in planes.iter().enumerate() {
                let cw = conv_width(c);
                let host = e.to_host(pl)?;
                conv.extend_from_slice(&host[slot * 3 * cw..(slot + 1) * 3 * cw]);
            }
        }
        let pos = kv.lens[slot];
        let first = pos - pos.min(WIN_ROWS);
        let mut win = Vec::new();
        let mut plane = |buf: &CudaSlice<u8>, rb: usize| -> Result<(), GpuModelError> {
            let host = e.to_host_u8_len(buf, buf.len())?;
            for q in first..pos {
                let r = slot * kv.ring + q % kv.ring;
                win.extend_from_slice(&host[r * rb..(r + 1) * rb]);
            }
            Ok(())
        };
        for (v, store) in kv.stores.iter().enumerate() {
            if !full_visit(v) {
                plane(&store.k, row_bytes(kv.dtype))?;
                plane(&store.v, row_bytes(kv.dtype))?;
            }
        }
        for store in kv.draft.iter().step_by(2) {
            plane(&store.k, row_bytes(3))?;
            plane(&store.v, row_bytes(3))?;
        }
        Ok((conv, win))
    }

    /// Blocks the pool could hand out: the free ones and the ones only the
    /// cache holds.
    pub(crate) fn pool_free_impl(&self) -> usize {
        let free = self.kv.pool.free_blocks();
        free + self
            .prefix
            .as_ref()
            .map_or(0, |p| p.radix.evictable_blocks(&self.kv.pool))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_cuts_at_its_last_boundary_and_the_back_off() {
        // under the floor: nothing
        assert!(Prefix::cuts_for(40, &[]).is_empty());
        let c = Prefix::cuts_for(1000, &[]);
        assert_eq!(*c.last().unwrap(), 992);
        assert!(c.iter().all(|p| p % BLOCK_TOKENS == 0 && *p < 1000));
        // hints floor to blocks and stay inside the prompt
        let c = Prefix::cuts_for(1000, &[130, 5000, 10]);
        assert!(c.contains(&128) && !c.iter().any(|p| !(MIN_CACHE_PREFIX..1000).contains(p)));
        // a prompt ending on a boundary checkpoints its end
        assert_eq!(*Prefix::cuts_for(1024, &[]).last().unwrap(), 1024);
    }

    #[test]
    fn the_conv_layout_tiles_a_checkpoint() {
        assert_eq!(conv_off(0, 0), 0);
        assert_eq!(conv_off(0, 1), 3 * KV);
        assert_eq!(conv_off(0, 2), 6 * KV);
        assert_eq!(conv_off(0, 3), 6 * KV + 3 * ik::WIDTH);
        assert_eq!(conv_off(1, 0), VISIT_CONV);
        assert_eq!(conv_off(ik::VISITS - 1, 3) + 3 * ik::WIDTH, CONV_FLOATS);
    }
}
