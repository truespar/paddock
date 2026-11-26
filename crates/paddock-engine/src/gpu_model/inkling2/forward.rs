//! The lane's forward: a call's rows - each slot's next positions, one run
//! a slot - through the 48 layer visits, the reference's control flow
//! (transformers' `Inkling2LoopTextModel.forward`) step for step. Every
//! kernel is row-wise or reads its row's (slot, position), so a prompt
//! chunk, a decode tick and a mix of both take the same path.

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;
use paddock_models::inkling2 as ik;

use super::draft;
use super::kv::{CONV_ATTN, CONV_K, CONV_MLP, CONV_V, full_visit};
use super::{DYN, GEMV_ROWS, GpuInkling2, KV, Layer, MAX_SPLITS, QRS, SPLIT_ROWS, TILE_ROWS};
use crate::gpu::{
    GpuError, GpuExecutor, INKLING2_KV_F32, Inkling2AttnOut, Inkling2KvLayout, QuantTensor,
};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Which rows of a call want logits.
#[derive(Clone, Copy, Debug)]
pub enum Logits {
    None,
    Last,
    All,
}

/// One run of a call's rows: the next `ids` of `slot`'s sequence.
pub(crate) struct Seg<'a> {
    pub slot: usize,
    pub ids: &'a [u32],
    pub want: Logits,
}

/// Rows of logits one head GEMM produces at once.
pub(crate) const HEAD_ROWS: usize = 256;
/// Keys a split should get at least (one of the kernel's 32-key tiles): a
/// decode row's few hundred keys still spread over the die.
const KEYS_PER_SPLIT: usize = 32;
/// Where a pass's wanted rows go.
pub(crate) enum PassOut<'a> {
    /// Every wanted row's logits to the host: `each(segment, row, logits)`.
    Logits(&'a mut dyn FnMut(usize, usize, &[f32])),
    /// Every wanted row sampled on the device by its plan (`params` rows x 4
    /// words, sample_rows' `{inv_t bits, u bits, mode, 0}`, flat across the
    /// passes of a call), the ids appended in row order.
    Picks(&'a [u32], &'a mut Vec<u32>),
}

/// A captured pass's host-side shape: everything that sets its launches'
/// grids and branches (positions, tiles, run tables and block tables live in
/// device buffers the stage uploads before each replay).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GraphKey {
    n: usize,
    runs: usize,
    max_run: usize,
    nd: usize,
    n_tiles: usize,
    splits: [usize; 4],
    stash: bool,
}

/// Decode ticks up to this many rows replay captured graphs.
const GRAPH_ROWS: usize = 64;

/// The hidden planes a visit reads: its input, the block states (pool
/// indices) and, in the loop's later passes, whether messages are in `msg`.
struct Visit<'a> {
    hin: usize,
    blocks: &'a [usize],
    msgs: bool,
}

/// What every visit of a call shares.
struct Pass {
    n: usize,
    runs: usize,
    max_run: usize,
    /// Rows the split attention takes from the front (all of them, or a
    /// mixed tick's decode rows) and the prompt-row tiles after them.
    nd: usize,
    n_tiles: usize,
    /// Key splits of the sliding and the full visits' split attention.
    split_swa: usize,
    split_full: usize,
    /// Key splits of the tiles (slot 1009) for a pass of few tiles and no
    /// split rows; 1 = unsplit.
    tsplit_swa: usize,
    tsplit_full: usize,
    /// Keep the bring-up trace (slot 0's prompt from position 0).
    trace: bool,
    /// A verify pass: stash every conv's carried rows and inputs (the
    /// rollback of a partly accepted run).
    stash: bool,
}

fn exhausted() -> GpuModelError {
    GpuModelError::Unsupported("Inkling 2: hidden pool exhausted".into())
}

impl GpuInkling2 {
    /// Run `ids` (slot 0's next positions) through the model in chunks;
    /// hand each wanted row's logits ([VOCAB_REAL], softcapped) to
    /// `each(row of this call, logits)`.
    pub fn forward_each(
        &mut self,
        ids: &[u32],
        want: Logits,
        mut each: impl FnMut(usize, &[f32]),
    ) -> Result<(), GpuModelError> {
        let chunk = self.scratch.rows;
        let n = ids.len();
        let mut i = 0;
        while i < n {
            let rows = chunk.min(n - i);
            let w = match want {
                Logits::All => Logits::All,
                Logits::Last if i + rows == n => Logits::Last,
                _ => Logits::None,
            };
            let seg = Seg {
                slot: 0,
                ids: &ids[i..i + rows],
                want: w,
            };
            self.forward_segs(&[seg], &mut |_, r, l| each(i + r, l))?;
            i += rows;
        }
        Ok(())
    }

    /// [`Self::forward_each`] for the last row's logits.
    pub fn forward_last(&mut self, ids: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let mut out = Vec::new();
        self.forward_each(ids, Logits::Last, |_, l| out = l.to_vec())?;
        Ok(out)
    }

    /// Key splits for `rows` rows reading up to `span` keys each: SPLIT_FILL
    /// CTAs an SM, never under KEYS_PER_SPLIT keys a split.
    fn splits(&self, rows: usize, span: usize) -> usize {
        if rows > SPLIT_ROWS {
            return 1;
        }
        let fill = (SPLIT_FILL * self.exec.sm_count()).div_ceil(rows * ik::KV_HEADS);
        fill.min(span.div_ceil(KEYS_PER_SPLIT))
            .clamp(1, split_cap(rows))
    }

    /// Key splits for a pass's tiles: few tiles of a pure tile pass (a
    /// verify pass's runs, a short prompt at a long context) leave most of
    /// the die idle with a CTA a (tile, K / V head) walking every key, so
    /// their keys split (slot 1009) until the grid holds SPLIT_FILL CTAs an SM;
    /// prompt chunks fill the die with tiles alone.
    fn tile_splits(&self, nd: usize, n: usize, n_tiles: usize, span: usize) -> usize {
        if nd > 0 || n > SPLIT_ROWS || n_tiles == 0 || !self.exec.has_inkling2_tile_split() {
            return 1;
        }
        let fill = (SPLIT_FILL * self.exec.sm_count()).div_ceil(n_tiles * ik::KV_HEADS);
        fill.min(span.div_ceil(4 * TILE_KEYS))
            .clamp(1, split_cap(n))
    }

    /// One pass over `segs` (distinct slots, rows in order, at most a
    /// chunk's worth); `each(segment, row in it, logits)` for the rows their
    /// `want` asks for. Every slot's blocks are backed before anything is
    /// stored, so PoolExhausted leaves every sequence as it was. A decode
    /// tick (a row a segment, every row's logits) replays a captured graph
    /// of its shape from the second time that shape runs.
    pub(crate) fn forward_segs(
        &mut self,
        segs: &[Seg<'_>],
        each: &mut dyn FnMut(usize, usize, &[f32]),
    ) -> Result<(), GpuModelError> {
        self.forward_segs_as(segs, false, each)
    }

    /// [`Self::forward_segs`]; with `stash` a verify pass (runs of at most
    /// 8 rows, every conv's carried rows and inputs kept for the rollback).
    pub(crate) fn forward_segs_as(
        &mut self,
        segs: &[Seg<'_>],
        stash: bool,
        each: &mut dyn FnMut(usize, usize, &[f32]),
    ) -> Result<(), GpuModelError> {
        self.forward_segs_out(segs, stash, &mut PassOut::Logits(each))
    }

    /// [`Self::forward_segs_as`] with the wanted rows going to `out`.
    pub(crate) fn forward_segs_out(
        &mut self,
        segs: &[Seg<'_>],
        stash: bool,
        out: &mut PassOut<'_>,
    ) -> Result<(), GpuModelError> {
        let Some(mut ps) = self.stage(segs)? else {
            return Ok(());
        };
        ps.stash = stash && !self.kv.stash.is_empty();
        let n = ps.n;
        // (row of the call, segment, row in it) for every wanted row
        let mut wanted = Vec::new();
        let mut base = 0;
        for (si, s) in segs.iter().enumerate() {
            let len = s.ids.len();
            match s.want {
                Logits::None => {}
                Logits::Last => wanted.push((base + len - 1, si, len - 1)),
                Logits::All => wanted.extend((0..len).map(|r| (base + r, si, r))),
            }
            base += len;
        }
        // decode ticks (a row a segment) and verify passes (runs, every
        // row's logits, the conv stash) replay a graph of their shape
        // (a pass with picture rows writes them after the gather, which no
        // captured graph does)
        let tick = !ps.trace
            && self.overlay.is_empty()
            && n <= GRAPH_ROWS
            && wanted.len() == n
            && (segs.len() == n || ps.stash);
        if tick {
            let key = GraphKey {
                n,
                runs: ps.runs,
                max_run: ps.max_run,
                nd: ps.nd,
                n_tiles: ps.n_tiles,
                splits: [ps.split_swa, ps.split_full, ps.tsplit_swa, ps.tsplit_full],
                stash: ps.stash,
            };
            if !self.graphs.contains_key(&key) && !self.graph_seen.insert(key) {
                let g = self.capture(&ps)?;
                self.graphs.insert(key, g);
            }
            match self.graphs.get(&key) {
                Some(g) => {
                    g.0.launch()
                        .map_err(|err| GpuError::Driver(format!("inkling2 tick graph: {err}")))?
                }
                None => self.tick_body(&ps)?,
            }
            for s in segs {
                self.kv.lens[s.slot] += s.ids.len();
            }
            self.draft_after_pass(segs, n)?;
            self.prefix_fed(segs, ps.stash)?;
            return self.emit(&wanted, out);
        }
        let (hidden, blocks) = self.walk(&ps)?;
        // the draft reads every row's final output (its last feature)
        if self.draft.is_some() || !wanted.is_empty() {
            self.final_norm(hidden, &blocks, n)?;
            self.tap_final(n)?;
        }
        for s in segs {
            self.kv.lens[s.slot] += s.ids.len();
        }
        self.draft_after_pass(segs, n)?;
        self.prefix_fed(segs, ps.stash)?;
        if wanted.is_empty() {
            return Ok(());
        }
        for group in wanted.chunks(HEAD_ROWS) {
            self.head_rows(group)?;
            self.emit(group, out)?;
        }
        Ok(())
    }

    /// A pass fed `segs`: every replying slot tracks its rows, and a decode
    /// row that closed a block checkpoints it (prefix.rs). A verify pass's
    /// rows wait for their rollback to settle.
    fn prefix_fed(&mut self, segs: &[Seg<'_>], stash: bool) -> Result<(), GpuModelError> {
        if self.prefix.is_none() {
            return Ok(());
        }
        for s in segs {
            self.reply_fed(s.slot, s.ids, !stash && s.ids.len() == 1)?;
        }
        Ok(())
    }

    /// The logits of `rows` (in `head_out`, in order) to `out`: to the host
    /// a row at a time, or sampled on the device by their plans (slot
    /// sample_rows) with only the ids read back.
    fn emit(
        &mut self,
        rows: &[(usize, usize, usize)],
        out: &mut PassOut<'_>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let (k, v) = (rows.len(), ik::VOCAB_REAL);
        let sc = &mut self.scratch;
        match out {
            PassOut::Logits(each) => {
                let host = e.to_host_len(&sc.head_out, k * v)?;
                for (j, &(_, si, r)) in rows.iter().enumerate() {
                    each(si, r, &host[j * v..(j + 1) * v]);
                }
            }
            PassOut::Picks(params, ids) => {
                let at = ids.len();
                let par = params.get(at * 4..(at + k) * 4).ok_or_else(|| {
                    GpuModelError::Unsupported("inkling2: fewer sampling plans than rows".into())
                })?;
                e.upload_u32(par, &mut sc.pick_par)?;
                e.sample_rows(&sc.head_out, &sc.pick_par, &mut sc.pick_ids, k, v)?;
                ids.extend(e.to_host_u32_len(&sc.pick_ids, k)?);
            }
        }
        Ok(())
    }

    /// Check `segs`, back their rows with blocks and upload the row tables;
    /// None for an empty call.
    fn stage(&mut self, segs: &[Seg<'_>]) -> Result<Option<Pass>, GpuModelError> {
        let n: usize = segs.iter().map(|s| s.ids.len()).sum();
        if n == 0 {
            return Ok(None);
        }
        if n > self.scratch.rows {
            return Err(GpuModelError::BatchTooLarge {
                got: n,
                max: self.scratch.rows,
            });
        }
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let (mut ids, mut pos, mut slots) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let (mut run_off, mut run_slot) = (vec![0u32], Vec::with_capacity(segs.len()));
        let mut max_run = 0usize;
        let mut seen = vec![false; self.kv.slots];
        for s in segs {
            if s.slot >= self.kv.slots || std::mem::replace(&mut seen[s.slot], true) {
                return Err(GpuModelError::Unsupported(format!(
                    "Inkling 2: slot {} twice or past the {} seated",
                    s.slot, self.kv.slots
                )));
            }
            let p0 = self.kv.lens[s.slot];
            let end = p0 + s.ids.len();
            if end > self.max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: end,
                    max: self.max_ctx,
                });
            }
            ids.extend_from_slice(s.ids);
            pos.extend((p0..end).map(|p| p as u32));
            slots.extend(std::iter::repeat_n(s.slot as u32, s.ids.len()));
            run_off.push(ids.len() as u32);
            run_slot.push(s.slot as u32);
            max_run = max_run.max(s.ids.len());
        }
        for s in segs {
            let end = self.kv.lens[s.slot] + s.ids.len();
            super::prefix::ensure_rows(e, &mut self.kv, self.prefix.as_mut(), s.slot, end)?;
        }
        // prompt runs take the tensor-core tiles (f16 / fp8 stores): every
        // row after the leading single-row segments (a mixed tick's decode
        // rows, which keep the split attention)
        let lead = segs.iter().take_while(|s| s.ids.len() == 1).count();
        let tiled = lead < segs.len() && self.kv.dtype != INKLING2_KV_F32 && e.has_inkling2_tile();
        let (nd, mut tiles) = (if tiled { lead } else { n }, Vec::new());
        let (mut span_d, mut span_t) = (0, 0);
        let mut row = 0;
        for (i, s) in segs.iter().enumerate() {
            let p0 = self.kv.lens[s.slot];
            let len = s.ids.len();
            if !tiled || i < lead {
                span_d = span_d.max(p0 + len);
            } else {
                span_t = span_t.max(p0 + len);
                for j in (0..len).step_by(TILE_ROWS) {
                    let k = TILE_ROWS.min(len - j);
                    tiles.extend([(row + j) as u32, k as u32, s.slot as u32, (p0 + j) as u32]);
                }
            }
            row += len;
        }
        let sc = &mut self.scratch;
        e.upload_u32(&ids, &mut sc.ids)?;
        e.upload_u32(&pos, &mut sc.pos)?;
        e.upload_u32(&slots, &mut sc.slots)?;
        e.upload_u32(&run_off, &mut sc.run_off)?;
        e.upload_u32(&run_slot, &mut sc.run_slot)?;
        if !tiles.is_empty() {
            e.upload_u32(&tiles, &mut sc.tiles)?;
        }
        Ok(Some(Pass {
            n,
            runs: segs.len(),
            max_run,
            nd,
            n_tiles: tiles.len() / 4,
            split_swa: self.splits(nd.max(1), span_d.min(ik::WINDOW)),
            split_full: self.splits(nd.max(1), span_d),
            tsplit_swa: self.tile_splits(nd, n, tiles.len() / 4, span_t.min(ik::WINDOW)),
            tsplit_full: self.tile_splits(nd, n, tiles.len() / 4, span_t),
            trace: self.trace_rows > 0
                && segs.len() == 1
                && segs[0].slot == 0
                && self.kv.lens[0] == 0,
            stash: false,
        }))
    }

    /// A decode tick's device work: the walk, the final norm and every
    /// row's logits into `head_out` - what a tick graph records.
    fn tick_body(&mut self, ps: &Pass) -> Result<(), GpuModelError> {
        let (hidden, blocks) = self.walk(ps)?;
        self.final_norm(hidden, &blocks, ps.n)?;
        self.tap_final(ps.n)?;
        let rows: Vec<(usize, usize, usize)> = (0..ps.n).map(|r| (r, r, 0)).collect();
        self.head_rows(&rows)
    }

    /// With the draft: visit `vi`'s input rows (pool plane `plane`) as their
    /// feature when `vi` is a tap.
    fn tap(&mut self, vi: usize, plane: usize, n: usize) -> Result<(), GpuModelError> {
        let Some(t) = draft::TAPS.iter().position(|&v| v == vi) else {
            return Ok(());
        };
        if self.draft.is_none() {
            return Ok(());
        }
        let w = ik::WIDTH;
        let sc = &mut self.scratch;
        self.exec.inkling2_copy2d(
            (&sc.pool[plane], 0, w),
            (&mut sc.feat, t * w, draft::NFEAT * w),
            w,
            n,
        )?;
        Ok(())
    }

    /// With the draft: the final normed rows (`mx`) as the last feature.
    fn tap_final(&mut self, n: usize) -> Result<(), GpuModelError> {
        if self.draft.is_none() {
            return Ok(());
        }
        let w = ik::WIDTH;
        let sc = &mut self.scratch;
        self.exec.inkling2_copy2d(
            (&sc.mx, 0, w),
            (&mut sc.feat, (draft::NFEAT - 1) * w, draft::NFEAT * w),
            w,
            n,
        )?;
        Ok(())
    }

    /// Record [`Self::tick_body`] for `ps`'s shape (nothing executes).
    fn capture(&mut self, ps: &Pass) -> Result<super::SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|err| GpuError::Driver(format!("inkling2 pre-capture sync: {err}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|err| GpuError::Driver(format!("inkling2 begin_capture: {err}")))?;
        let rec = self.tick_body(ps);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|err| GpuError::Driver(format!("inkling2 end_capture: {err}")));
        // a record failure surfaces only after the capture is cleanly ended
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("inkling2 capture produced no graph".into()))?;
        Ok(super::SendGraph(graph))
    }

    /// Embedding through the coda over the staged rows: the final hidden
    /// plane and the block states (pool indices).
    fn walk(&mut self, ps: &Pass) -> Result<(usize, Vec<usize>), GpuModelError> {
        let n = ps.n;
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let w = ik::WIDTH;
        {
            let sc = &mut self.scratch;
            e.embed_gather_bf16(&self.embed, &sc.ids, &mut sc.emb, w, n, 1.0)?;
            e.rmsnorm_batch(&sc.emb, &self.embed_norm, &mut sc.pool[0], w, ik::EPS, n)?;
            // a picture prompt's rows: the tower's over the placeholders'
            // (the reference scatters them past the embedding's norm)
            for s in &self.overlay {
                e.copy_region(&s.pic.rows, s.from * w, &mut sc.pool[0], s.row * w, s.n * w)?;
            }
        }
        // pool 0 holds the normed embedding: layer 0's input and the first
        // block state
        let mut free: Vec<usize> = (1..self.scratch.pool.len()).rev().collect();
        let mut hidden = 0usize;
        let mut blocks: Vec<usize> = Vec::new();
        // prelude
        for i in 0..ik::LOOP_START {
            let out = free.pop().ok_or_else(exhausted)?;
            self.tap(i, hidden, n)?;
            self.layer_visit(
                i,
                i,
                Visit {
                    hin: hidden,
                    blocks: &blocks,
                    msgs: false,
                },
                out,
                ps,
            )?;
            self.trace_visit(out, ps)?;
            if i.is_multiple_of(ik::ATTN_RES_BLOCK) {
                blocks.push(hidden);
            } else {
                free.push(hidden);
            }
            hidden = out;
        }
        // the core: every pass restarts from the prelude's output and block
        // states; passes 2.. read messages from the previous pass's last
        // block state and output
        let (pre_hidden, pre_blocks) = (hidden, blocks.clone());
        let core = ik::LOOP_END - ik::LOOP_START;
        let mut prev: Option<(usize, usize)> = None;
        for pass in 0..ik::PASSES {
            let mut hdn = pre_hidden;
            let mut blk = pre_blocks.clone();
            for l in 0..core {
                let i = ik::LOOP_START + l;
                if let Some((pb, ph)) = prev {
                    let sc = &mut self.scratch;
                    e.inkling2_attn_res(
                        &[&sc.pool[pb]],
                        (&sc.pool[ph], 0),
                        &self.msg_q[l],
                        4,
                        &mut sc.msg,
                        n,
                        w,
                        n * w,
                    )?;
                }
                let out = free.pop().ok_or_else(exhausted)?;
                let vis = Visit {
                    hin: hdn,
                    blocks: &blk,
                    msgs: prev.is_some(),
                };
                let vi = ik::LOOP_START + pass * core + l;
                self.tap(vi, hdn, n)?;
                self.layer_visit(i, vi, vis, out, ps)?;
                self.trace_visit(out, ps)?;
                if i.is_multiple_of(ik::ATTN_RES_BLOCK) {
                    blk.push(hdn);
                } else if hdn != pre_hidden {
                    free.push(hdn);
                }
                hdn = out;
            }
            if let Some((pb, ph)) = prev.take() {
                free.push(pb);
                free.push(ph);
            }
            // the pass's block states past the prelude's: the first is the
            // prelude output again, the last the next pass's message source
            for &b in &blk[pre_blocks.len()..blk.len() - 1] {
                if b != pre_hidden {
                    free.push(b);
                }
            }
            if pass + 1 < ik::PASSES {
                prev = Some((*blk.last().expect("a core block"), hdn));
            } else {
                hidden = hdn;
                blocks = blk;
            }
        }
        // coda
        for i in ik::LOOP_END..ik::LAYERS {
            let out = free.pop().ok_or_else(exhausted)?;
            let v = i + (ik::PASSES - 1) * core;
            self.tap(v, hidden, n)?;
            self.layer_visit(
                i,
                v,
                Visit {
                    hin: hidden,
                    blocks: &blocks,
                    msgs: false,
                },
                out,
                ps,
            )?;
            self.trace_visit(out, ps)?;
            if i.is_multiple_of(ik::ATTN_RES_BLOCK) {
                blocks.push(hidden);
            } else {
                free.push(hidden);
            }
            hidden = out;
        }
        Ok((hidden, blocks))
    }

    /// The final attention residual over the block states + the norm, every
    /// row into `mx`.
    fn final_norm(
        &mut self,
        hidden: usize,
        blocks: &[usize],
        n: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let w = ik::WIDTH;
        let sc = &mut self.scratch;
        let states: Vec<&CudaSlice<f32>> = blocks.iter().map(|&b| &sc.pool[b]).collect();
        e.inkling2_attn_res(
            &states,
            (&sc.pool[hidden], 0),
            &self.final_q,
            1,
            &mut sc.o,
            n,
            w,
            n * w,
        )?;
        e.rmsnorm_batch(&sc.o, &self.norm, &mut sc.mx, w, ik::EPS, n)?;
        Ok(())
    }

    /// Softcapped logits of `group`'s rows of `mx` (at most HEAD_ROWS) into
    /// `head_out`, in group order.
    fn head_rows(&mut self, group: &[(usize, usize, usize)]) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let (w, v, k) = (ik::WIDTH, ik::VOCAB_REAL, group.len());
        let sc = &mut self.scratch;
        // one copy when the group's rows are consecutive (a prompt's rows, a
        // decode tick's), else a row at a time
        if group[k - 1].0 - group[0].0 == k - 1 {
            e.copy_region(&sc.mx, group[0].0 * w, &mut sc.head_in, 0, k * w)?;
        } else {
            for (j, &(r, _, _)) in group.iter().enumerate() {
                e.copy_region(&sc.mx, r * w, &mut sc.head_in, j * w, w)?;
            }
        }
        e.bf16_gemm_rows(&self.unembed, 0, v, &sc.head_in, &mut sc.head_out, k)?;
        e.softcap(&mut sc.head_out, k * v, ik::LOGIT_SOFTCAP)?;
        Ok(())
    }

    /// Keep visit output `plane`'s first rows (the bring-up trace).
    fn trace_visit(&mut self, plane: usize, ps: &Pass) -> Result<(), GpuModelError> {
        if !ps.trace {
            return Ok(());
        }
        let rows = self.trace_rows.min(ps.n);
        let host = self.exec.to_host(&self.scratch.pool[plane])?;
        self.trace.push(host[..rows * ik::WIDTH].to_vec());
        Ok(())
    }

    /// One layer visit: `v.hin` through layer `li` at visit `vi` (its K / V
    /// stores and conv states) into pool plane `out`.
    fn layer_visit(
        &mut self,
        li: usize,
        vi: usize,
        v: Visit<'_>,
        out: usize,
        ps: &Pass,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let ly: &Layer = &self.layers[li];
        let kv = &mut self.kv;
        let sc = &mut self.scratch;
        // a pass that reaches a prefix-cache cut captures each conv's state
        // there before the conv moves it on (prefix.rs)
        let mut px = self.prefix.as_mut();
        let n = ps.n;
        let (w, heads, hd) = (ik::WIDTH, ik::HEADS, ik::HEAD_DIM);
        let nw = n * w;
        let boundary = li.is_multiple_of(ik::ATTN_RES_BLOCK);
        let ctrl = ly.ctrl.as_ref().filter(|_| v.msgs);
        let rank = ik::GATE_RANK;
        let trace = if ps.trace && vi == 0 {
            self.trace_rows.min(n)
        } else {
            0
        };
        debug_assert_eq!(ly.full, full_visit(vi));
        // the branch tails in one pass each (slot 1004), unless a verify pass
        // stashes the convs' inputs or a trace keeps them
        let fuse = !ps.stash && trace == 0 && e.has_inkling2_sconv_sum();
        let stages = &mut self.stages;
        // keep `width` columns at `off` of the first `trace` rows (stride apart)
        let mut snap = |name: &str,
                        buf: &CudaSlice<f32>,
                        off: usize,
                        stride: usize,
                        width: usize|
         -> Result<(), GpuModelError> {
            if trace == 0 {
                return Ok(());
            }
            let host = e.to_host(buf)?;
            let v: Vec<f32> = (0..trace)
                .flat_map(|r| host[off + r * stride..off + r * stride + width].to_vec())
                .collect();
            stages.push((name.to_string(), v));
            Ok(())
        };

        // ---- the attention's inputs: [q | k | v] rows in ares (one row set
        // on the sliding layers), each through its controller in the later
        // passes, then the input norm(s) into qin / kin / vin
        {
            let states: Vec<&CudaSlice<f32>> = v.blocks.iter().map(|&b| &sc.pool[b]).collect();
            match &ly.attn_q {
                Some(q) => {
                    let nq = if ly.full { 3 } else { 1 };
                    e.inkling2_attn_res(
                        &states,
                        (&sc.pool[v.hin], 0),
                        q,
                        nq,
                        &mut sc.ares,
                        n,
                        w,
                        nw,
                    )?;
                }
                None => e.copy_region(&sc.pool[v.hin], 0, &mut sc.ares, 0, nw)?,
            }
        }
        snap("embed", &sc.ares, 0, w, w)?;
        // sliding layers without messages feed one normed input to q, k, v
        let shared = !ly.full && ctrl.is_none();
        let kvw = if ly.full { KV + DYN } else { KV };
        // where the projections take the prefill pair, the norms write its
        // bf16 rows straight away (slot 1007: the narrowing to the bit)
        let narrow =
            trace == 0 && e.has_rmsnorm_batch_bf16() && pf_all(e, &[QRS, heads * hd, kvw], n);
        for m in 0..if shared { 1 } else { 3 } {
            let src = if ly.full { m * nw } else { 0 };
            let norm = &ly.in_norm[if ly.full { m } else { 0 }];
            let (dst, dst16) = match m {
                0 => (&mut sc.qin, &mut sc.xin16[0]),
                1 => (&mut sc.kin, &mut sc.xin16[1]),
                _ => (&mut sc.vin, &mut sc.xin16[2]),
            };
            match ctrl {
                Some(c) => {
                    let c = &c[m];
                    ctrl_down(
                        e,
                        &c.down,
                        (&sc.ares, src),
                        (&sc.msg, m * nw),
                        (&mut sc.cat, &mut sc.z),
                        n,
                    )?;
                    e.inkling2_msg_mix(
                        &sc.z,
                        &c.forget,
                        &c.gain,
                        (&sc.ares, src),
                        (&sc.msg, m * nw),
                        &mut sc.gtmp,
                        n,
                        w,
                        rank,
                    )?;
                    if narrow {
                        e.rmsnorm_batch_bf16_at(&sc.gtmp, 0, norm, dst16, w, ik::EPS, n)?;
                    } else {
                        e.rmsnorm_batch(&sc.gtmp, norm, dst, w, ik::EPS, n)?;
                    }
                }
                None if narrow => {
                    e.rmsnorm_batch_bf16_at(&sc.ares, src, norm, dst16, w, ik::EPS, n)?
                }
                None => e.rmsnorm_batch_at(&sc.ares, src, norm, dst, w, ik::EPS, n)?,
            }
        }
        let (kin, vin) = if shared {
            (&sc.qin, &sc.qin)
        } else {
            (&sc.kin, &sc.vin)
        };
        snap("in", &sc.qin, 0, w, w)?;

        // ---- projections and the per-head prep
        if narrow {
            let (ki, vj) = if shared { (0, 0) } else { (1, 2) };
            e.bf16_gemm_pf_rows(&ly.qrs, 0, QRS, &sc.xin16[0], &mut sc.qrs, n)?;
            e.bf16_gemm_pf_rows(&ly.gate, 0, heads * hd, &sc.xin16[0], &mut sc.gate, n)?;
            e.bf16_gemm_pf_rows(&ly.k, 0, kvw, &sc.xin16[ki], &mut sc.kp, n)?;
            e.bf16_gemm_pf_rows(&ly.v, 0, kvw, &sc.xin16[vj], &mut sc.vp, n)?;
        } else if shared {
            dense_rows(
                e,
                &mut [
                    (&ly.qrs, QRS, &mut sc.qrs),
                    (&ly.gate, heads * hd, &mut sc.gate),
                    (&ly.k, kvw, &mut sc.kp),
                    (&ly.v, kvw, &mut sc.vp),
                ],
                &sc.qin,
                &mut sc.x16,
                n,
            )?;
        } else {
            dense_rows(
                e,
                &mut [
                    (&ly.qrs, QRS, &mut sc.qrs),
                    (&ly.gate, heads * hd, &mut sc.gate),
                ],
                &sc.qin,
                &mut sc.x16,
                n,
            )?;
            dense_rows(e, &mut [(&ly.k, kvw, &mut sc.kp)], kin, &mut sc.x16, n)?;
            dense_rows(e, &mut [(&ly.v, kvw, &mut sc.vp)], vin, &mut sc.x16, n)?;
        }
        let log = if ly.full { Some(&sc.pos) } else { None };
        let lp = (ik::LOG_ALPHA, ik::LOG_FLOOR as f32);
        e.inkling2_head_prep(
            &mut sc.qrs,
            0,
            n,
            heads,
            hd,
            QRS,
            Some(&ly.q_scale),
            ik::QK_CLIP,
            false,
            log,
            lp,
        )?;
        e.inkling2_head_prep(
            &mut sc.qrs,
            heads * hd,
            n,
            heads,
            ik::D_REL,
            QRS,
            Some(&ly.rel_scale),
            ik::REL_CLIP,
            false,
            log,
            lp,
        )?;
        snap("q", &sc.qrs, 0, QRS, heads * hd)?;
        snap("rel", &sc.qrs, heads * hd, QRS, heads * ik::D_REL)?;
        snap("sink", &sc.qrs, heads * hd + heads * ik::D_REL, QRS, heads)?;
        snap("k_lin", &sc.kp, 0, kvw, KV)?;
        // K / V: conv into kc / vc, then the key's clip / the value's unit
        // RMS in place, then into this visit's store
        let runs = (&sc.run_off, &sc.run_slot, ps.runs);
        let dtype = kv.dtype;
        for (src, conv, prep, state, key) in [
            (&sc.kp, &ly.k_conv, &mut sc.kc, CONV_K, true),
            (&sc.vp, &ly.v_conv, &mut sc.vc, CONV_V, false),
        ] {
            let dy = ly.full.then_some((src, KV, kvw, ik::KV_HEADS));
            super::prefix::capture(
                e,
                px.as_deref_mut(),
                vi,
                state,
                (src, 1, kvw),
                &kv.conv[vi][state],
            )?;
            if ps.stash {
                e.inkling2_conv_stash(
                    (src, 0, kvw),
                    KV,
                    &kv.conv[vi][state],
                    runs,
                    &mut kv.stash[vi][state],
                )?;
            }
            e.inkling2_sconv(
                (src, 0, kvw),
                (prep, 0, KV),
                KV,
                conv,
                dy,
                &mut kv.conv[vi][state],
                runs,
                n,
                ps.max_run,
            )?;
            snap(if key { "k_conv" } else { "v_conv" }, prep, 0, KV, KV)?;
            let (clip, rms) = if key {
                (ik::QK_CLIP, false)
            } else {
                (0.0, true)
            };
            e.inkling2_head_prep(prep, 0, n, ik::KV_HEADS, hd, KV, None, clip, rms, None, lp)?;
            snap(if key { "k" } else { "v" }, prep, 0, KV, KV)?;
            let layout = if ly.full {
                Inkling2KvLayout::Paged(&kv.bt, kv.bps)
            } else {
                Inkling2KvLayout::Ring(kv.ring)
            };
            let store = &mut kv.stores[vi];
            e.inkling2_kv_store(
                (prep, 0, KV),
                (&sc.slots, &sc.pos),
                if key { &mut store.k } else { &mut store.v },
                layout,
                dtype,
                KV,
                n,
            )?;
        }
        let (window, splits) = if ly.full {
            (0, ps.split_full)
        } else {
            (ik::WINDOW, ps.split_swa)
        };
        let layout = if ly.full {
            Inkling2KvLayout::Paged(&kv.bt, kv.bps)
        } else {
            Inkling2KvLayout::Ring(kv.ring)
        };
        let part = (splits > 1).then_some((&mut sc.part_o, &mut sc.part_ml));
        let store = &kv.stores[vi];
        let scale = 1.0 / (hd as f32).sqrt();
        let (qx, rx, sx) = (
            (&sc.qrs, 0, QRS),
            (&sc.qrs, heads * hd, QRS),
            (&sc.qrs, heads * hd + heads * ik::D_REL, QRS),
        );
        if ps.nd > 0 {
            e.inkling2_attn_kv(
                qx,
                rx,
                sx,
                (&store.k, &store.v),
                layout,
                dtype,
                (&sc.slots, &sc.pos),
                &ly.rel_bank,
                &mut sc.att,
                part,
                ps.nd,
                (heads, ik::KV_HEADS),
                (window, ik::REL_EXTENT),
                splits,
                scale,
            )?;
        }
        // the tiles' rows come out gated (slot 1005) - and, where the o
        // projection takes the prefill pair, as its bf16 rows; the split rows
        // (the pass's first nd) take mul_sigmoid (and the narrowing)
        let tsplit = if ly.full {
            ps.tsplit_full
        } else {
            ps.tsplit_swa
        };
        let gated = ps.n_tiles > 0 && tsplit == 1 && e.has_inkling2_tile_g();
        let o16 = gated && trace == 0 && pf_all(e, &[w], n);
        if ps.n_tiles > 0 && tsplit > 1 {
            e.inkling2_attn_tile_split(
                qx,
                rx,
                sx,
                (&store.k, &store.v),
                layout,
                dtype,
                (&sc.tiles, ps.n_tiles),
                &ly.rel_bank_t,
                &mut sc.att,
                (&mut sc.part_o, &mut sc.part_ml),
                n,
                (heads, ik::KV_HEADS),
                (window, ik::REL_EXTENT),
                scale,
                tsplit,
            )?;
        } else if ps.n_tiles > 0 {
            let out = if o16 {
                Inkling2AttnOut::Gated16(&mut sc.x16, &sc.gate, heads * hd)
            } else if gated {
                Inkling2AttnOut::Gated(&mut sc.att, &sc.gate, heads * hd)
            } else {
                Inkling2AttnOut::Plain(&mut sc.att)
            };
            e.inkling2_attn_tile(
                qx,
                rx,
                sx,
                (&store.k, &store.v),
                layout,
                dtype,
                (&sc.tiles, ps.n_tiles),
                &ly.rel_bank_t,
                out,
                (heads, ik::KV_HEADS),
                (window, ik::REL_EXTENT),
                scale,
            )?;
        }
        let gate_rows = if gated { ps.nd } else { n };
        if gate_rows > 0 {
            e.mul_sigmoid(&mut sc.att, &sc.gate, gate_rows * heads * hd)?;
        }
        snap("att", &sc.att, 0, heads * hd, heads * hd)?;
        if o16 {
            if ps.nd > 0 {
                e.convert_f32_bf16(&sc.att, &mut sc.x16, ps.nd * heads * hd)?;
            }
            e.bf16_gemm_pf_rows(&ly.o, 0, w, &sc.x16, &mut sc.o, n)?;
        } else {
            dense_rows(e, &mut [(&ly.o, w, &mut sc.o)], &sc.att, &mut sc.x16, n)?;
        }

        // ---- the attention conv and the residual: a boundary layer restarts
        // it (its input became a block state), the others add their input
        super::prefix::capture(
            e,
            px.as_deref_mut(),
            vi,
            CONV_ATTN,
            (&sc.o, 1, w),
            &kv.conv[vi][CONV_ATTN],
        )?;
        if fuse {
            let (src, dst) = if boundary {
                (None, &mut sc.ac)
            } else {
                (Some(&sc.pool[v.hin]), &mut sc.res)
            };
            e.inkling2_sconv_sum(
                (&sc.o, 1),
                src,
                dst,
                w,
                &ly.attn_conv,
                &mut kv.conv[vi][CONV_ATTN],
                runs,
                n,
                ps.max_run,
            )?;
        } else {
            if ps.stash {
                e.inkling2_conv_stash(
                    (&sc.o, 0, w),
                    w,
                    &kv.conv[vi][CONV_ATTN],
                    runs,
                    &mut kv.stash[vi][CONV_ATTN],
                )?;
            }
            e.inkling2_sconv(
                (&sc.o, 0, w),
                (&mut sc.ac, 0, w),
                w,
                &ly.attn_conv,
                None,
                &mut kv.conv[vi][CONV_ATTN],
                runs,
                n,
                ps.max_run,
            )?;
            snap("attn", &sc.o, 0, w, w)?;
            snap("post_attn_conv", &sc.ac, 0, w, w)?;
            if !boundary {
                e.copy_region(&sc.pool[v.hin], 0, &mut sc.res, 0, nw)?;
                e.add(&mut sc.res, &sc.ac, nw)?;
            }
        }
        let residual = if boundary { &sc.ac } else { &sc.res };

        // ---- the MLP input
        {
            let mut idx: Vec<usize> = v.blocks.to_vec();
            if boundary {
                idx.push(v.hin);
            }
            let states: Vec<&CudaSlice<f32>> = idx.iter().map(|&b| &sc.pool[b]).collect();
            e.inkling2_attn_res(&states, (residual, 0), &ly.mlp_q, 1, &mut sc.min, n, w, nw)?;
        }
        match ctrl {
            Some(c) => {
                let c = &c[3];
                ctrl_down(
                    e,
                    &c.down,
                    (&sc.min, 0),
                    (&sc.msg, 3 * nw),
                    (&mut sc.cat, &mut sc.z),
                    n,
                )?;
                e.inkling2_msg_mix(
                    &sc.z,
                    &c.forget,
                    &c.gain,
                    (&sc.min, 0),
                    (&sc.msg, 3 * nw),
                    &mut sc.gtmp,
                    n,
                    w,
                    rank,
                )?;
                e.rmsnorm_batch(&sc.gtmp, &ly.mlp_norm, &mut sc.mx, w, ik::EPS, n)?;
            }
            None => e.rmsnorm_batch(&sc.min, &ly.mlp_norm, &mut sc.mx, w, ik::EPS, n)?,
        }

        snap("mlp_input", &sc.min, 0, w, w)?;
        snap("router_in", &sc.mx, 0, w, w)?;
        // ---- the routed experts (the router weight rides into the down
        // input's quantization, as the serving class does)
        // the router's 128 columns at verify widths: the multi-row GEMV
        // runs them on 8 CTAs (latency-bound), the tile K-splits a thin grid
        // (GB10, c1 decode with rounds: +1.6 tok/s with the draft's
        // tensor-core attention)
        if (2..=8).contains(&n) {
            e.bf16_gemm_tile(&ly.router, None, &sc.mx, &mut sc.logits, ik::EXPERTS, n)?;
        } else {
            e.bf16_gemm_rows(&ly.router, 0, ik::EXPERTS, &sc.mx, &mut sc.logits, n)?;
        }
        snap("router", &sc.logits, 0, ik::EXPERTS, ik::EXPERTS)?;
        e.moe_topk_sigmoid_batch(
            &sc.logits,
            &ly.router_bias,
            ik::ROUTE_SCALE,
            ik::EXPERTS,
            ik::TOP_K,
            &mut sc.idx,
            &mut sc.wts,
            n,
        )?;
        let ex = &ly.experts;
        let caps = (ik::GATE_SOFTCAP, ik::UP_SOFTCAP);
        e.inkling2_quant(&sc.mx, w, None, true, ex.fmt, &mut sc.xa, n, w)?;
        let pairs = n * ik::TOP_K;
        // decode widths: the GEMVs (a warp a column streams the routed
        // planes; measured ahead of the tensor-core arms there, whose tiles
        // leave the die mostly idle at a handful of pairs)
        if n <= GEMV_ROWS {
            e.inkling2_moe_gu(ex, &sc.xa, &sc.idx, ik::TOP_K, &mut sc.act, n, caps)?;
            e.inkling2_quant(
                &sc.act,
                ik::FF,
                Some(&sc.wts),
                true,
                ex.fmt,
                &mut sc.aa,
                pairs,
                ik::FF,
            )?;
            e.inkling2_moe_dn(ex, &sc.aa, &sc.idx, ik::TOP_K, &mut sc.y, n)?;
        } else {
            // the NVFP4 class takes 64-pair blocks (slots 1002-1003: an
            // expert's planes read half as often through a deeper ring,
            // measured ahead from 8 tokens up); MXFP8 32 (slots 991-992, or
            // 987-988 off the block-scale tensor cores)
            let bm = if ex.fmt == 0 && e.has_inkling2_ms() {
                64
            } else {
                32
            };
            e.inkling2_sort(&sc.idx, &mut sc.sorted, n, ik::TOP_K, ik::EXPERTS, bm)?;
            // on 64-pair blocks the act rows land as bf16(act x router
            // weight) - the quantizer's own input, so the same codes off half
            // the bytes (slot 1006)
            let ms16 = bm == 64 && e.has_inkling2_ms16();
            if ms16 {
                e.inkling2_moe_gu_ms16(
                    ex,
                    &sc.xa,
                    &sc.sorted,
                    ik::TOP_K,
                    &mut sc.act,
                    &sc.wts,
                    caps,
                )?;
                e.inkling2_quant16(&sc.act, ex.fmt, &mut sc.aa, pairs, ik::FF)?;
            } else {
                e.inkling2_moe_gu_grp(ex, &sc.xa, &sc.sorted, ik::TOP_K, &mut sc.act, caps)?;
                e.inkling2_quant(
                    &sc.act,
                    ik::FF,
                    Some(&sc.wts),
                    true,
                    ex.fmt,
                    &mut sc.aa,
                    pairs,
                    ik::FF,
                )?;
            }
            e.inkling2_moe_dn_grp(ex, &sc.aa, &sc.sorted, ik::TOP_K, &mut sc.part)?;
            if !fuse {
                e.zero_region(&mut sc.y, 0, nw)?;
                e.moe_slot_combine(&sc.part, &mut sc.y, w, ik::TOP_K, n)?;
            }
        }
        let residual = if boundary { &sc.ac } else { &sc.res };
        if fuse {
            // the experts' sum (the GEMVs' rows, or the 8 partials a row),
            // the MLP conv and the residual in one pass
            let input = if n <= GEMV_ROWS {
                (&sc.y, 1)
            } else {
                (&sc.part, ik::TOP_K)
            };
            super::prefix::capture(
                e,
                px.as_deref_mut(),
                vi,
                CONV_MLP,
                (input.0, input.1, w),
                &kv.conv[vi][CONV_MLP],
            )?;
            e.inkling2_sconv_sum(
                input,
                Some(residual),
                &mut sc.pool[out],
                w,
                &ly.mlp_conv,
                &mut kv.conv[vi][CONV_MLP],
                runs,
                n,
                ps.max_run,
            )?;
            return Ok(());
        }
        super::prefix::capture(e, px, vi, CONV_MLP, (&sc.y, 1, w), &kv.conv[vi][CONV_MLP])?;
        if ps.stash {
            e.inkling2_conv_stash(
                (&sc.y, 0, w),
                w,
                &kv.conv[vi][CONV_MLP],
                runs,
                &mut kv.stash[vi][CONV_MLP],
            )?;
        }
        e.inkling2_sconv(
            (&sc.y, 0, w),
            (&mut sc.yc, 0, w),
            w,
            &ly.mlp_conv,
            None,
            &mut kv.conv[vi][CONV_MLP],
            runs,
            n,
            ps.max_run,
        )?;
        snap("experts", &sc.y, 0, w, w)?;
        snap("post_mlp_conv", &sc.yc, 0, w, w)?;
        e.copy_region(residual, 0, &mut sc.pool[out], 0, nw)?;
        e.add(&mut sc.pool[out], &sc.yc, nw)?;
        Ok(())
    }
}

/// A tile's key step (slot 993's BN); a tile split takes four at least.
const TILE_KEYS: usize = 64;

/// CTAs an SM a split attention fills to: a decode row at a 22K context walked
/// ~1.9K keys a CTA at two (GB10: 13.5 ms of a 42 ms tick).
const SPLIT_FILL: usize = 8;

/// Key splits `rows` rows may take: the partial planes hold SPLIT_ROWS x
/// MAX_SPLITS (row, split) pairs, the combine folds up to 64.
fn split_cap(rows: usize) -> usize {
    (SPLIT_ROWS * MAX_SPLITS / rows.max(1)).min(64)
}

/// Planes at least this wide take the prefill pair when it is elected; the
/// router (128) and the controllers' downs (64) stay on the ladder, whose
/// K-split fills the die there (GB10, 2048 rows: router 81 vs 148 us).
const PF_OUT: usize = 1024;

/// `y_i = W_i x` for dense planes sharing the input `x` over `n` rows: the
/// bf16 prefill pair (slot 756) on rows narrowed once, where every plane is
/// wide enough and the executor elects the pair for it, else the ladder.
/// The same bf16 x bf16, f32-accumulate class either way, bit-identical to
/// the ladder's unsplit tile (GB10, 2048 rows: qrs 1294 -> 751 us, gate
/// 1255 -> 655, o 1191 -> 628, k 384 -> 320 with the narrowing).
fn dense_rows(
    e: &GpuExecutor,
    planes: &mut [(&QuantTensor, usize, &mut CudaSlice<f32>)],
    x: &CudaSlice<f32>,
    x16: &mut CudaSlice<half::bf16>,
    n: usize,
) -> Result<(), GpuModelError> {
    let outs: Vec<usize> = planes.iter().map(|p| p.1).collect();
    let pf = pf_all(e, &outs, n);
    if pf {
        e.convert_f32_bf16(x, x16, n * planes[0].0.dims[0])?;
    }
    for (w, out, y) in planes.iter_mut() {
        if pf {
            e.bf16_gemm_pf_rows(w, 0, *out, x16, y, n)?;
        } else {
            e.bf16_gemm_rows(w, 0, *out, x, y, n)?;
        }
    }
    Ok(())
}

/// Planes `outs` wide over `n` rows all take the prefill pair.
fn pf_all(e: &GpuExecutor, outs: &[usize], n: usize) -> bool {
    outs.iter()
        .all(|&out| out >= PF_OUT && e.bf16_pf_elect(out, n))
}

/// Rows from which a controller's cat and down projection run as one pass
/// (slot 1008); narrower waves keep slot 978 + the decode GEMVs.
const MSG_DOWN_ROWS: usize = 64;

/// A loop controller's z = down x cat(h, m) over `n` rows: one pass from
/// MSG_DOWN_ROWS (no cat plane), else slot 978's cat and the ladder.
fn ctrl_down(
    e: &GpuExecutor,
    down: &QuantTensor,
    h: (&CudaSlice<f32>, usize),
    m: (&CudaSlice<f32>, usize),
    (cat, z): (&mut CudaSlice<f32>, &mut CudaSlice<f32>),
    n: usize,
) -> Result<(), GpuModelError> {
    let (w, rank) = (ik::WIDTH, ik::GATE_RANK);
    if n >= MSG_DOWN_ROWS && !no_msg_down() && e.inkling2_msg_down(h, m, down, z, n, w, rank)? {
        return Ok(());
    }
    e.inkling2_msg_cat(h, m, cat, n, w)?;
    e.bf16_gemm_rows(down, 0, rank, cat, z, n)?;
    Ok(())
}

/// PADDOCK_INKLING2_NO_MSG_DOWN: the controllers keep slot 978 + the ladder.
fn no_msg_down() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_INKLING2_NO_MSG_DOWN").is_some())
}
