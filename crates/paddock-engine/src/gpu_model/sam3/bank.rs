//! SAM 3's memory bank: what one tracker state keeps of the frames it has
//! tracked, and which of it a frame's memory attention reads - Meta's
//! `_prepare_memory_conditioned_features` with memory selection on (the
//! video model builds its tracker with `apply_temporal_disambiguation`).
//!
//! A tracker state is the objects the detector found on one frame: Meta
//! batches them in one `inference_state`, and its bookkeeping is the
//! state's, not an object's. Every frame it has tracked holds, for each
//! object, the memory (`[5184][64]`, stored bf16 as Meta stores it), the
//! object pointer (`[256]`) and the two scores memory selection reads.
//!
//! What frame `f` reads - the same frames for every object of the state:
//! - conditioning frames (births and re-conditionings), at most 4: all of
//!   them while the state has had at most 4, in the order they came; past
//!   that the closest before `f`, the closest at or after it, then the
//!   rest by distance (`select_closest_cond_frames`). Temporal row 6.
//! - previous frames: Meta's `frame_filter` lists the frames before `f`
//!   whose `eff_iou_score` is over 0.01 - scanning back, never down to
//!   frame 0 (its range stops short), the latest 15 - and appends `f - 1`
//!   when it is not in the list. The last six entries are read, the
//!   `t_pos`-th of them (1 the oldest, 6 the latest) at temporal row
//!   `6 - t_pos`. An entry with nothing behind it (an `f - 1` that has
//!   since become a conditioning frame) is skipped.
//! - object pointers: the selected conditioning frames' that are not after
//!   `f`, at their distance in frames; then the list's, latest first, at
//!   their RANK (1, 2, ...), one short of the list's oldest entry (Meta's
//!   loop breaks there). Positions are over `min(num_frames, 16) - 1`.
//!
//! `eff_iou_score` is a frame's, for the whole state: Meta multiplies a
//! `[B, 1]` score by a `[B]` IoU and takes the mean of the `[B, B]`
//! product, which is the product of the two means, not the mean of the
//! products. An object's score is `2 sigmoid(logit) - 1` where the logit is
//! positive, else 0; its IoU the best of the three candidates'.
//!
//! What this does not do that Meta does is keep everything: Meta holds every
//! tracked frame's memory for the whole video (663 KB an object a frame).
//! This keeps what the selection can still read on a later frame, tracking
//! forward: the 4 newest conditioning frames, the memories of the 6 newest
//! selectable frames and of the latest frame, the pointers and scores of the
//! 16 newest and of the latest - about 12 memories an object at any time.
//! The one way that drops something Meta would still read is a frame that
//! was not selectable becoming selectable, which only the rescoring after an
//! object's removal can do, and only on a frame where every object of the
//! state but a barely-present one was gone.
//!
//! Forward tracking only, a camera or a stream from its start.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::{bf16, f16};

use super::GpuModelError;
use super::detector::sine_pos_table;
use super::load::Reader;
use super::memattn::MEMATTN_MAX_KEYS;
use crate::gpu::GpuExecutor;

const GRID: usize = 72;
const TOKENS: usize = GRID * GRID;
const D: usize = 256;
const MEM_DIM: usize = 64;
/// Meta's `num_maskmem`: the conditioning frames' temporal row is 6, the
/// previous frames' 0-5.
const NUM_MASKMEM: usize = 7;
/// `max_cond_frames_in_attn`.
const MAX_COND: usize = 4;
/// `max_obj_ptrs_in_encoder`, before `min(num_frames, .)`.
const MAX_PTRS: usize = 16;
/// The most pointers a frame reads: 4 conditioning frames' and 15 more.
pub const BANK_MAX_POINTERS: usize = MAX_COND + MAX_PTRS - 1;
/// `mf_threshold`: a frame is selectable over this `eff_iou_score`.
const MF_THRESHOLD: f32 = 0.01;
/// Selectable frames whose memories the bank keeps (the six previous-frame
/// reads) and whose pointers it keeps (the list's 15 and one to spare).
const KEEP_MEM: usize = NUM_MASKMEM - 1;
const KEEP_PTR: usize = MAX_PTRS;

/// One object on one tracked frame.
#[derive(Default)]
struct Slot {
    /// `[5184][64]` bf16, once the memory encoder has run on the frame
    mem: Option<CudaSlice<bf16>>,
    /// row of the bank's pointer pool
    ptr: Option<u32>,
    /// the object-score logit and the best candidate's IoU (selection)
    logit: f32,
    iou: f32,
}

struct Frame {
    idx: u32,
    objs: Vec<Slot>,
}

impl Frame {
    /// Meta's `cal_mem_score`, for the state.
    fn eff(&self) -> f32 {
        let n = self.objs.len().max(1) as f32;
        let (mut norm, mut iou) = (0f32, 0f32);
        for o in &self.objs {
            if o.logit > 0.0 {
                norm += 2.0 / (1.0 + (-o.logit).exp()) - 1.0;
            }
            iou += o.iou;
        }
        (norm / n) * (iou / n)
    }
}

/// Which rows frame [`Sam3BankPlan::frame`] reads, the same for every object
/// of the state: memory frames `(conditioning, frame, temporal row)`, then
/// pointers `(conditioning, frame, distance)`.
#[derive(Clone, Debug)]
pub struct Sam3BankPlan {
    pub frame: u32,
    pub memories: Vec<(bool, u32, usize)>,
    pub pointers: Vec<(bool, u32, u32)>,
    /// the pointers' positions are `distance / tmax`
    pub tmax: f32,
}

impl Sam3BankPlan {
    /// The memory attention's key count, and how many of them are memory
    /// tokens (rope on their keys; the pointer tokens after them have none).
    pub fn keys(&self) -> (usize, usize) {
        let nrope = self.memories.len() * TOKENS;
        (nrope + 4 * self.pointers.len(), nrope)
    }
}

/// One tracker state's bank (Meta's `output_dict` for a state's objects).
pub struct Sam3Bank {
    exec: Arc<GpuExecutor>,
    objs: usize,
    /// conditioning frames, in the order they came (Meta's dict order)
    cond: Vec<Frame>,
    /// how many conditioning frames the state has had (Meta keeps them all;
    /// past 4 its selection order changes)
    cond_seen: usize,
    /// the other tracked frames, ascending
    prev: Vec<Frame>,
    /// memory slots ready for reuse
    spare: Vec<CudaSlice<bf16>>,
    /// the pointers, `[rows][256]`, and the free rows
    pool: CudaSlice<f32>,
    pool_free: Vec<u32>,
}

impl Sam3Bank {
    /// An empty bank for a state of `objs` objects.
    pub fn new(exec: Arc<GpuExecutor>, objs: usize) -> Result<Self, GpuModelError> {
        if objs == 0 {
            return Err(GpuModelError::Unsupported(
                "sam3 bank: a tracker state needs an object".into(),
            ));
        }
        let rows = 24 * objs;
        let pool = exec.alloc(rows * D)?;
        Ok(Self {
            exec,
            objs,
            cond: Vec::new(),
            cond_seen: 0,
            prev: Vec::new(),
            spare: Vec::new(),
            pool,
            pool_free: (0..rows as u32).rev().collect(),
        })
    }

    /// Objects in the state.
    pub fn objects(&self) -> usize {
        self.objs
    }

    fn new_frame(&self, idx: u32) -> Frame {
        Frame {
            idx,
            objs: (0..self.objs).map(|_| Slot::default()).collect(),
        }
    }

    fn frame_mut(&mut self, idx: u32) -> Option<&mut Frame> {
        if let Some(i) = self.cond.iter().position(|c| c.idx == idx) {
            return Some(&mut self.cond[i]);
        }
        let i = self.prev.binary_search_by_key(&idx, |p| p.idx).ok()?;
        Some(&mut self.prev[i])
    }

    fn frame(&self, cond: bool, idx: u32) -> Option<&Frame> {
        if cond {
            self.cond.iter().find(|c| c.idx == idx)
        } else {
            let i = self.prev.binary_search_by_key(&idx, |p| p.idx).ok()?;
            Some(&self.prev[i])
        }
    }

    fn check_obj(&self, obj: usize) -> Result<(), GpuModelError> {
        if obj >= self.objs {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 bank: object {obj} of a {}-object state",
                self.objs
            )));
        }
        Ok(())
    }

    /// A pool row, the pool doubled when it is full.
    fn pool_row(&mut self) -> Result<u32, GpuModelError> {
        if let Some(r) = self.pool_free.pop() {
            return Ok(r);
        }
        let rows = self.pool.len() / D;
        let mut grown = self.exec.alloc(2 * rows * D)?;
        self.exec
            .copy_region(&self.pool, 0, &mut grown, 0, rows * D)?;
        self.pool = grown;
        self.pool_free
            .extend((rows as u32 + 1..2 * rows as u32).rev());
        Ok(rows as u32)
    }

    /// Copy `ptr` (`[256]` f32) into object `obj`'s pointer on frame `idx`.
    fn put_ptr(&mut self, idx: u32, obj: usize, ptr: &CudaSlice<f32>) -> Result<(), GpuModelError> {
        let have = self.frame_mut(idx).and_then(|f| f.objs[obj].ptr);
        let row = match have {
            Some(r) => r,
            None => self.pool_row()?,
        };
        self.exec
            .copy_region(ptr, 0, &mut self.pool, row as usize * D, D)?;
        if let Some(f) = self.frame_mut(idx) {
            f.objs[obj].ptr = Some(row);
        }
        Ok(())
    }

    /// A tracked frame's result for one object: its pointer (`[256]`), its
    /// object-score logit and its best candidate's IoU. The frame becomes a
    /// previous frame of the state; tracking runs forward, so it is the
    /// newest one.
    pub fn track(
        &mut self,
        idx: u32,
        obj: usize,
        ptr: &CudaSlice<f32>,
        logit: f32,
        iou: f32,
    ) -> Result<(), GpuModelError> {
        self.check_obj(obj)?;
        if self.cond.iter().any(|c| c.idx == idx) {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 bank: frame {idx} is a conditioning frame, not a tracked one"
            )));
        }
        if self.prev.binary_search_by_key(&idx, |p| p.idx).is_err() {
            if self.prev.last().is_some_and(|p| p.idx > idx) {
                return Err(GpuModelError::Unsupported(format!(
                    "sam3 bank: frame {idx} is behind the newest tracked one (forward only)"
                )));
            }
            let f = self.new_frame(idx);
            self.prev.push(f);
        }
        self.put_ptr(idx, obj, ptr)?;
        let f = self.frame_mut(idx).expect("inserted above");
        f.objs[obj].logit = logit;
        f.objs[obj].iou = iou;
        Ok(())
    }

    /// Frame `idx` conditions the state: a birth, or a re-conditioning from a
    /// detection (Meta's `add_new_mask`, every frame with a mask input a
    /// conditioning one). `obj`'s pointer becomes the mask-as-output pass's;
    /// the other objects keep the pointer they were tracked with. A tracked
    /// frame leaves the previous frames for good, as Meta pops it.
    pub fn condition(
        &mut self,
        idx: u32,
        obj: usize,
        ptr: &CudaSlice<f32>,
    ) -> Result<(), GpuModelError> {
        self.check_obj(obj)?;
        if !self.cond.iter().any(|c| c.idx == idx) {
            let f = match self.prev.binary_search_by_key(&idx, |p| p.idx) {
                Ok(i) => self.prev.remove(i),
                Err(_) => self.new_frame(idx),
            };
            self.cond.push(f);
            self.cond_seen += 1;
        }
        self.put_ptr(idx, obj, ptr)
    }

    /// Object `obj`'s memory on frame `idx` (conditioning or tracked): the
    /// encoder's f32 `[5184][64]` at `src[src_off..]`, stored bf16. A later
    /// call for the same frame replaces it, as Meta's memory update does.
    pub fn set_memory(
        &mut self,
        idx: u32,
        obj: usize,
        src: &CudaSlice<f32>,
        src_off: usize,
    ) -> Result<(), GpuModelError> {
        self.check_obj(obj)?;
        let n = TOKENS * MEM_DIM;
        let mut slot = match self.frame_mut(idx).and_then(|f| f.objs[obj].mem.take()) {
            Some(s) => s,
            None => match self.spare.pop() {
                Some(s) => s,
                None => self.exec.stream_alloc_bf16(n)?,
            },
        };
        self.exec.sam3_bank_store(src, src_off, &mut slot, n)?;
        let f = self.frame_mut(idx).ok_or_else(|| {
            GpuModelError::Unsupported(format!("sam3 bank: no frame {idx} to hold a memory"))
        })?;
        f.objs[obj].mem = Some(slot);
        Ok(())
    }

    fn release(&mut self, s: Slot) {
        if let Some(m) = s.mem {
            self.spare.push(m);
        }
        if let Some(r) = s.ptr {
            self.pool_free.push(r);
        }
    }

    /// Drop object `obj` from every frame (Meta's `remove_object`; the
    /// frames' scores are the remaining objects' from here on). Returns how
    /// many objects are left.
    pub fn remove_object(&mut self, obj: usize) -> Result<usize, GpuModelError> {
        self.check_obj(obj)?;
        let mut freed = Vec::new();
        for f in self.cond.iter_mut().chain(self.prev.iter_mut()) {
            freed.push(f.objs.remove(obj));
        }
        for s in freed {
            self.release(s);
        }
        self.objs -= 1;
        Ok(self.objs)
    }

    /// What frame `idx` reads (the state must have a conditioning frame, as
    /// Meta asserts). `num_frames` is the video's length when it is known; a
    /// stream's is not, and only a clip under 16 frames changes anything.
    pub fn plan(&self, idx: u32, num_frames: Option<u32>) -> Result<Sam3BankPlan, GpuModelError> {
        if self.cond.is_empty() {
            return Err(GpuModelError::Unsupported(
                "sam3 bank: a tracked frame needs a conditioning frame".into(),
            ));
        }
        let cond: Vec<u32> = self.cond.iter().map(|c| c.idx).collect();
        let prev: Vec<(u32, f32)> = self.prev.iter().map(|p| (p.idx, p.eff())).collect();
        Ok(select(&cond, self.cond_seen, &prev, idx, num_frames))
    }

    /// After frame `idx` is done (tracked, re-conditioned, its memories
    /// written): let go of what no later frame can read (module doc).
    pub fn prune(&mut self, idx: u32) {
        let mut freed = Vec::new();
        if self.cond.len() > MAX_COND {
            let mut by_age: Vec<u32> = self.cond.iter().map(|c| c.idx).collect();
            by_age.sort_unstable_by(|a, b| b.cmp(a));
            let oldest_kept = by_age[MAX_COND - 1];
            let mut i = 0;
            while i < self.cond.len() {
                if self.cond[i].idx < oldest_kept {
                    freed.extend(self.cond.remove(i).objs);
                } else {
                    i += 1;
                }
            }
        }
        // the selectable previous frames, newest first
        let mut rank = 0usize;
        let mut i = self.prev.len();
        while i > 0 {
            i -= 1;
            let f = &mut self.prev[i];
            let latest = f.idx == idx;
            let selectable = f.idx > 0 && f.eff() > MF_THRESHOLD;
            if selectable {
                rank += 1;
            }
            if latest || (selectable && rank <= KEEP_MEM) {
                continue;
            }
            if selectable && rank <= KEEP_PTR {
                for s in &mut f.objs {
                    if let Some(m) = s.mem.take() {
                        freed.push(Slot {
                            mem: Some(m),
                            ..Slot::default()
                        });
                    }
                }
                continue;
            }
            freed.extend(self.prev.remove(i).objs);
        }
        for s in freed {
            self.release(s);
        }
    }
}

/// Meta's selection for frame `idx` (module doc): `cond` the conditioning
/// frames in the order they came (`cond_seen` how many the state has had),
/// `prev` the tracked frames ascending with their `eff_iou_score`.
fn select(
    cond: &[u32],
    cond_seen: usize,
    prev: &[(u32, f32)],
    idx: u32,
    num_frames: Option<u32>,
) -> Sam3BankPlan {
    let max_ptrs = num_frames.map_or(MAX_PTRS, |n| (n as usize).clamp(1, MAX_PTRS));

    // select_closest_cond_frames
    let n = cond.len();
    let sel: Vec<usize> = if cond_seen <= MAX_COND {
        (0..n).collect()
    } else {
        let mut sel = Vec::with_capacity(MAX_COND);
        let before = (0..n).filter(|&i| cond[i] < idx).max_by_key(|&i| cond[i]);
        let after = (0..n).filter(|&i| cond[i] >= idx).min_by_key(|&i| cond[i]);
        sel.extend(before);
        sel.extend(after);
        let mut rest: Vec<usize> = (0..n).filter(|i| !sel.contains(i)).collect();
        // stable, as Python's sorted
        rest.sort_by_key(|&i| (cond[i] as i64 - idx as i64).abs());
        rest.truncate(MAX_COND.saturating_sub(sel.len()));
        sel.extend(rest);
        sel
    };

    // frame_filter: the tracked frames before idx, newest first, never
    // frame 0 (Meta's range stops at 1); the cap counts frames it looked at
    let mut valid: Vec<u32> = Vec::new();
    if idx > 0 {
        for &(g, eff) in prev.iter().rev() {
            if g >= idx {
                continue;
            }
            if g == 0 {
                break;
            }
            if eff > MF_THRESHOLD {
                valid.insert(0, g);
            }
            if valid.len() >= max_ptrs - 1 {
                break;
            }
        }
        if !valid.contains(&(idx - 1)) {
            valid.push(idx - 1);
        }
    }
    // a listed frame is a tracked one, or a conditioning frame the
    // selection left out
    let lookup = |g: u32| -> Option<bool> {
        if prev.binary_search_by_key(&g, |p| p.0).is_ok() {
            Some(false)
        } else if (0..n).any(|i| cond[i] == g && !sel.contains(&i)) {
            Some(true)
        } else {
            None
        }
    };

    let mut memories = Vec::with_capacity(MAX_COND + NUM_MASKMEM - 1);
    for &i in &sel {
        memories.push((true, cond[i], NUM_MASKMEM - 1));
    }
    for t_pos in 1..NUM_MASKMEM {
        let t_rel = NUM_MASKMEM - t_pos;
        if t_rel > valid.len() {
            continue;
        }
        let g = valid[valid.len() - t_rel];
        if let Some(c) = lookup(g) {
            memories.push((c, g, NUM_MASKMEM - t_pos - 1));
        }
    }
    let mut pointers = Vec::with_capacity(BANK_MAX_POINTERS);
    for &i in &sel {
        if cond[i] <= idx {
            pointers.push((true, cond[i], idx - cond[i]));
        }
    }
    for t_diff in 1..max_ptrs {
        if t_diff >= valid.len() {
            break;
        }
        let g = valid[valid.len() - t_diff];
        if let Some(c) = lookup(g) {
            pointers.push((c, g, t_diff as u32));
        }
    }
    Sam3BankPlan {
        frame: idx,
        memories,
        pointers,
        tmax: (max_ptrs - 1).max(1) as f32,
    }
}

/// The bank's landing: the temporal tables and the two `[keys][64]` f16
/// planes the memory attention reads, filled for one object at a time.
pub struct GpuSam3BankKv {
    exec: Arc<GpuExecutor>,
    /// the memory's sine table, `[5184][64]`
    pos: CudaSlice<f32>,
    /// `memory_temporal_positional_encoding`, `[7][64]`
    tpos: CudaSlice<f32>,
    /// `temporal_positional_encoding_projection_layer`, `[64][256]` + `[64]`
    proj_w: CudaSlice<f32>,
    proj_b: CudaSlice<f32>,
    kin: CudaSlice<f16>,
    v: CudaSlice<f16>,
    meta: CudaSlice<u32>,
}

impl GpuSam3BankKv {
    pub fn load_dir(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_sam3_bank() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's memory bank (slots 790-791) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let st = super::checkpoint::open(dir)?;
        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };
        let tpos = {
            let v = r.f32s(
                "tracker_model.memory_temporal_positional_encoding",
                &[NUM_MASKMEM, 1, 1, MEM_DIM],
            )?;
            r.dev(&v)?
        };
        let proj = "tracker_model.temporal_positional_encoding_projection_layer";
        let proj_w = {
            let v = r.f32s(&format!("{proj}.weight"), &[MEM_DIM, D])?;
            r.dev(&v)?
        };
        let proj_b = r.vec(&format!("{proj}.bias"), MEM_DIM)?;
        let pos = r.dev(&sine_pos_table(GRID, MEM_DIM, 10000.0))?;
        let kin = exec.alloc_f16(MEMATTN_MAX_KEYS * MEM_DIM)?;
        let v = exec.alloc_f16(MEMATTN_MAX_KEYS * MEM_DIM)?;
        let meta = exec.alloc_u32(2 * BANK_MAX_POINTERS)?;
        Ok(Self {
            exec,
            pos,
            tpos,
            proj_w,
            proj_b,
            kin,
            v,
            meta,
        })
    }

    /// Fill the planes with object `obj`'s bank for `plan`: returns the key
    /// count and how many of them are memory tokens, what
    /// [`super::GpuSam3MemAttn::run`] takes with [`Self::kin`] and
    /// [`Self::vmem`].
    pub fn fill(
        &mut self,
        bank: &Sam3Bank,
        plan: &Sam3BankPlan,
        obj: usize,
    ) -> Result<(usize, usize), GpuModelError> {
        bank.check_obj(obj)?;
        let (nk, nrope) = plan.keys();
        if nk > MEMATTN_MAX_KEYS || plan.pointers.len() > BANK_MAX_POINTERS {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 bank: {nk} keys, {} pointers (at most {MEMATTN_MAX_KEYS}, \
                 {BANK_MAX_POINTERS})",
                plan.pointers.len()
            )));
        }
        let missing = |what: &str, cond: bool, g: u32| {
            GpuModelError::Unsupported(format!(
                "sam3 bank: no {what} for object {obj} on {} frame {g}",
                if cond { "conditioning" } else { "tracked" }
            ))
        };
        let exec = self.exec.clone();
        let mut row = 0;
        for &(cond, g, trow) in &plan.memories {
            let mem = bank
                .frame(cond, g)
                .and_then(|f| f.objs[obj].mem.as_ref())
                .ok_or_else(|| missing("memory", cond, g))?;
            exec.sam3_bank_mem_rows(
                mem,
                &self.pos,
                &self.tpos,
                trow,
                &mut self.kin,
                &mut self.v,
                row,
                TOKENS,
            )?;
            row += TOKENS;
        }
        let mut rows = Vec::with_capacity(plan.pointers.len());
        let mut dist = Vec::with_capacity(plan.pointers.len());
        for &(cond, g, d) in &plan.pointers {
            let r = bank
                .frame(cond, g)
                .and_then(|f| f.objs[obj].ptr)
                .ok_or_else(|| missing("pointer", cond, g))?;
            rows.push(r);
            dist.push(d);
        }
        exec.sam3_bank_ptr_rows(
            &bank.pool,
            &rows,
            &dist,
            &mut self.meta,
            (&self.proj_w, &self.proj_b),
            &mut self.kin,
            &mut self.v,
            row,
            plan.tmax,
        )?;
        Ok((nk, nrope))
    }

    /// `f16(M + pos)`, `[keys][64]`, after [`Self::fill`].
    pub fn kin(&self) -> &CudaSlice<f16> {
        &self.kin
    }

    /// `f16(M)`, `[keys][64]`, after [`Self::fill`].
    pub fn vmem(&self) -> &CudaSlice<f16> {
        &self.v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(p: &Sam3BankPlan) -> (Vec<(bool, u32, usize)>, Vec<(bool, u32, u32)>) {
        (p.memories.clone(), p.pointers.clone())
    }

    /// Meta's reference clip on frames 1-3 and 17 (one state, frame 0 its
    /// birth, every tracked frame selectable, frame 16 re-conditioned): the
    /// key counts its memory attention was handed - 5188, 10372, 15560 and
    /// 36352 with 64 of them pointer tokens.
    #[test]
    fn selection_matches_metas_reference_clip() {
        let all: Vec<(u32, f32)> = (1..=15).map(|g| (g, 0.5)).collect();
        let at = |idx: u32, cond: &[u32]| {
            let prev: Vec<(u32, f32)> = all
                .iter()
                .copied()
                .filter(|&(g, _)| g < idx && !cond.contains(&g))
                .collect();
            select(cond, cond.len(), &prev, idx, Some(270))
        };
        let p = at(1, &[0]);
        assert_eq!(frames(&p), (vec![(true, 0, 6)], vec![(true, 0, 1)]));
        assert_eq!(p.keys(), (5188, 5184));
        // frame 1's pointer is never read on frame 2: the list is [1] and
        // the rank loop stops one short of its oldest entry
        let p = at(2, &[0]);
        assert_eq!(
            frames(&p),
            (vec![(true, 0, 6), (false, 1, 0)], vec![(true, 0, 2)])
        );
        assert_eq!(p.keys().0, 10372);
        let p = at(3, &[0]);
        assert_eq!(
            frames(&p),
            (
                vec![(true, 0, 6), (false, 1, 1), (false, 2, 0)],
                vec![(true, 0, 3), (false, 2, 1)]
            )
        );
        assert_eq!(p.keys().0, 15560);
        // frame 16 is a conditioning frame now: listed as f - 1, read as
        // neither a previous memory nor a ranked pointer
        let p = at(17, &[0, 16]);
        let mem: Vec<u32> = p.memories.iter().map(|m| m.1).collect();
        assert_eq!(mem, [0, 16, 11, 12, 13, 14, 15]);
        // t_pos 1-5 are frames 11-15; t_pos 6 was frame 16, skipped, so
        // the newest previous frame sits at temporal row 1
        let rows: Vec<usize> = p.memories.iter().map(|m| m.2).collect();
        assert_eq!(rows, [6, 6, 5, 4, 3, 2, 1]);
        let ptr: Vec<(u32, u32)> = p.pointers.iter().map(|q| (q.1, q.2)).collect();
        let mut want = vec![(0, 17), (16, 1)];
        want.extend((2..=15).map(|d| (17 - d, d)));
        assert_eq!(ptr, want);
        assert_eq!(p.keys(), (36352, 36288));
    }

    /// A frame the selection rejects is skipped, except as the frame just
    /// before; frame 0 is never listed; the list caps at 15 frames looked at.
    #[test]
    fn selection_skips_unselectable_frames() {
        // frames 1-30 tracked, 25 and 29 with the object gone
        let prev: Vec<(u32, f32)> = (1..=30)
            .map(|g| (g, if g == 25 || g == 29 { 0.0 } else { 0.5 }))
            .collect();
        let p = select(&[0], 1, &prev, 30, None);
        let mem: Vec<u32> = p.memories.iter().map(|m| m.1).collect();
        assert_eq!(mem, [0, 23, 24, 26, 27, 28, 29]);
        let ptr: Vec<u32> = p.pointers.iter().map(|q| q.1).collect();
        // the list: 15 selectable frames back to 13, then 29 appended; read
        // from 29 at rank 1 down to rank 15 - not frame 13
        let mut want = vec![
            0, 29, 28, 27, 26, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
        ];
        assert_eq!(ptr, want);
        // the frame before selectable: 15 frames, read one short
        let p = select(&[0], 1, &prev, 29, None);
        want = vec![0, 28, 27, 26, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14];
        assert_eq!(p.pointers.iter().map(|q| q.1).collect::<Vec<_>>(), want);
    }

    /// Past four conditioning frames: the newest four, newest first.
    #[test]
    fn selection_takes_the_closest_conditioning_frames() {
        let cond = [0, 16, 32, 48, 64];
        let prev: Vec<(u32, f32)> = (65..=69).map(|g| (g, 0.5)).collect();
        let p = select(&cond, 5, &prev, 70, None);
        let c: Vec<u32> = p.memories.iter().filter(|m| m.0).map(|m| m.1).collect();
        assert_eq!(c, [64, 48, 32, 16]);
        let d: Vec<u32> = p.pointers.iter().filter(|q| q.0).map(|q| q.2).collect();
        assert_eq!(d, [6, 22, 38, 54]);
        // pruned to the newest four, the order still follows Meta's count
        let p = select(&cond[1..], 5, &prev, 70, None);
        let c: Vec<u32> = p.memories.iter().filter(|m| m.0).map(|m| m.1).collect();
        assert_eq!(c, [64, 48, 32, 16]);
        // at four or fewer, the order they came in
        let p = select(&cond[1..], 4, &prev, 70, None);
        let c: Vec<u32> = p.memories.iter().filter(|m| m.0).map(|m| m.1).collect();
        assert_eq!(c, [16, 32, 48, 64]);
    }

    /// A short clip shrinks the pointer window and the position scale.
    #[test]
    fn selection_follows_a_short_clip() {
        let prev: Vec<(u32, f32)> = (1..=19).map(|g| (g, 0.5)).collect();
        let p = select(&[0], 1, &prev, 20, Some(11));
        assert_eq!(p.tmax, 10.0);
        // min(11, 16) - 1 = 10 frames listed (19 back to 10), read one short
        assert_eq!(p.pointers.len(), 1 + 9);
        assert_eq!(p.pointers.last(), Some(&(false, 11, 9)));
    }
}
