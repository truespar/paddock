//! Unlimited-OCR-2's SR-SWA on the batched lane (see [`super::Dash`]).
//!
//! Host side: the prompt's page layout (the page-2D rope coordinates of every
//! prompt row and the candidate block range of every prompt query), the
//! per-slot decode state machine the reference advances token by token, and the
//! page-count guard. Device side: one block summary per (slot, 32-key block)
//! and a 32-row staging block per slot, per layer, plus the row streams the
//! router kernels read each pass.
//!
//! Positions and layout, as the checkpoint's inference path builds them:
//! `cum[i]` counts `<PAGE>` markers up to i, `last_img` is the last image row
//! and `N = cum[last_img]`. Prompt row i sits in page bucket `cum[i]` (rows
//! after the last image fall back to bucket `cum[i] - N` = 0) at token
//! coordinate "rows of that bucket before it". A prompt query of page `k =
//! cum[i] - 1` ranks the blocks holding page k's image tokens; BOS and the
//! trailing instruction rank every block. Decode: a fed `<PAGE>` opens the next
//! bucket (its token count continuing that bucket's prompt rows), anything else
//! counts on; the response page `gen_markers - 1` ranks its image page's blocks.
//!
//! KV: a decode query reads its 128-key window and prompt blocks only, so the
//! slot keeps every row before `keep` (the prompt, through the block that
//! straddles its end) and a 128-row ring after it - exact under the trained
//! layout (text before the first generated `<PAGE>` could rank generated
//! blocks in the reference; here such a row ranks the prompt's).
//!
//! Prefix caching is off on this layout: a resumed prompt would need its block
//! summaries restored with its KV.

use cudarc::driver::CudaSlice;

use super::Dash;
use super::load::GpuDeepseekOcr;
use crate::gpu::{DASH_BLOCK, DASH_D, DASH_KCAP, GpuError, GpuExecutor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// The reference's per-page output cap (`max_page_tokens`): a page that runs
/// this long is closed by the guard (the next `<PAGE>`, or EOS on the last).
/// Public so the runner's echo states the value that runs.
pub const MAX_PAGE_TOKENS: u32 = 20_000;
/// The candidate range meaning "every remote block".
const ALL: [u32; 2] = [1, 0];

/// A prompt's page structure, built once at admission.
#[derive(Clone, Debug)]
pub(crate) struct DashLayout {
    pub prompt_len: usize,
    /// `<PAGE>` markers before the last image row; 0 = the degenerate layout
    /// (no image, or no marker: one bucket, coordinates = positions).
    pub n_pages: u32,
    /// Per prompt row: page-2D rope (tok, page parity).
    pub coords: Vec<[u32; 2]>,
    /// Per prompt row: candidate block range ([`ALL`] = every remote block).
    pub cand: Vec<[u32; 2]>,
    /// Prompt rows per page bucket (bucket 0: BOS / instruction, k + 1: page k).
    pub pf_cnt: Vec<u32>,
    /// Blocks holding each page's image tokens.
    pub page_blocks: Vec<[u32; 2]>,
    /// Decode state at the prompt's last row.
    pub end_bucket: u32,
    pub end_tok: u32,
    /// Rows kept for good once decode starts; the ring follows.
    pub keep: usize,
}

impl DashLayout {
    pub fn build(ids: &[u32], d: &Dash) -> Self {
        let s = ids.len();
        let bs = d.block_size;
        let pm = d.page_mod.max(1);
        let mut cum = vec![0u32; s];
        let mut c = 0u32;
        for (i, &t) in ids.iter().enumerate() {
            c += u32::from(t == d.page_token_id);
            cum[i] = c;
        }
        let last_img = ids.iter().rposition(|&t| t == d.image_token_id);
        let n = last_img.map_or(0, |li| cum[li]);
        if n == 0 {
            // no page marker before the last image (or no image): one bucket
            return DashLayout {
                prompt_len: s,
                n_pages: 0,
                coords: (0..s as u32).map(|i| [i, 0]).collect(),
                cand: vec![ALL; s],
                pf_cnt: vec![s as u32],
                page_blocks: Vec::new(),
                end_bucket: 0,
                end_tok: s.saturating_sub(1) as u32,
                keep: s,
            };
        }
        let last_img = last_img.expect("n > 0 implies an image");
        let bucket = |i: usize| -> u32 {
            if i <= last_img {
                cum[i]
            } else {
                cum[i].saturating_sub(n)
            }
        };
        let mut cnt = vec![0u32; n as usize + 1];
        let mut pf_cnt = vec![0u32; n as usize + 1];
        let mut coords = Vec::with_capacity(s);
        for i in 0..s {
            let b = bucket(i) as usize;
            coords.push([cnt[b], b as u32 % pm]);
            cnt[b] += 1;
            if i <= last_img {
                pf_cnt[b] += 1;
            }
        }
        let mut page_blocks = vec![[u32::MAX, 0u32]; n as usize];
        for (i, &t) in ids.iter().enumerate() {
            if t == d.image_token_id && cum[i] > 0 {
                let pg = &mut page_blocks[(cum[i] - 1) as usize];
                let blk = (i / bs) as u32;
                pg[0] = pg[0].min(blk);
                pg[1] = pg[1].max(blk);
            }
        }
        let cand = (0..s)
            .map(|i| {
                // the query's page, 0-based (the instruction after the last
                // image counts from the end: -1, unconstrained)
                let q = if i <= last_img {
                    cum[i] as i64 - 1
                } else {
                    cum[i] as i64 - 1 - n as i64
                };
                match usize::try_from(q).ok().and_then(|q| page_blocks.get(q)) {
                    Some(&[lo, hi]) if lo <= hi => [lo, hi],
                    _ => ALL,
                }
            })
            .collect();
        let keep = s.max((last_img / bs + 1) * bs);
        DashLayout {
            prompt_len: s,
            n_pages: n,
            coords,
            cand,
            pf_cnt,
            page_blocks,
            end_bucket: bucket(s - 1),
            end_tok: cnt[bucket(s - 1) as usize] - 1,
            keep,
        }
    }
}

/// One slot's decode state: the reference's rope state machine, its decode
/// page counter and the page-count guard, advanced once per fed token.
#[derive(Clone, Debug)]
pub(crate) struct DashSlot {
    pub lay: DashLayout,
    bucket: u32,
    tok: u32,
    /// `<PAGE>` tokens generated so far (the guard's and the response page's
    /// count).
    gen_markers: u32,
    /// Tokens since the last generated `<PAGE>` (from the start before one).
    page_len: u32,
}

impl DashSlot {
    pub fn new(lay: DashLayout) -> Self {
        DashSlot {
            bucket: lay.end_bucket,
            tok: lay.end_tok,
            gen_markers: 0,
            page_len: 0,
            lay,
        }
    }

    /// Feed one generated token at its decode step: the row's rope
    /// coordinates and candidate block range.
    fn feed(&mut self, t: u32, d: &Dash) -> ([u32; 2], [u32; 2]) {
        let top = self.lay.n_pages;
        if t == d.page_token_id {
            self.bucket = (self.bucket + 1).min(top);
            self.tok = self.lay.pf_cnt[self.bucket as usize];
            self.gen_markers += 1;
            self.page_len = 0;
        } else {
            self.tok += 1;
            self.page_len += 1;
        }
        let coords = [self.tok, self.bucket % d.page_mod.max(1)];
        // the prompt's kept blocks bound every decode candidate
        let kept_hi = ((self.lay.keep - 1) / d.block_size) as u32;
        let page = self.gen_markers as i64 - 1;
        let cand = match usize::try_from(page)
            .ok()
            .and_then(|p| self.lay.page_blocks.get(p))
        {
            Some(&[lo, hi]) if lo <= hi => [lo, hi],
            _ => [0, kept_hi],
        };
        (coords, cand)
    }

    /// The page-count guard for the token the row's logits pick next.
    pub fn guard_mode(&self) -> u32 {
        let n = self.lay.n_pages;
        if n == 0 {
            0
        } else if self.gen_markers > n {
            2
        } else if self.page_len >= MAX_PAGE_TOKENS {
            if self.gen_markers < n { 3 } else { 2 }
        } else if self.gen_markers < n {
            1
        } else {
            0
        }
    }
}

/// Device state of the router, per layer, plus the per-pass row streams.
pub(crate) struct DashDev {
    pub kbar_blocks: usize,
    /// [slots][kbar_blocks][128] f32 per layer.
    pub kbar: Vec<CudaSlice<f32>>,
    /// [slots][32][128] f32 per layer: the keys of a slot's unfinished block.
    pub staging: Vec<CudaSlice<f32>>,
    /// Page-2D rope token frequencies [63].
    pub inv_tok: CudaSlice<f32>,
    /// Router `[q_r | k_r]` rows [cap][256].
    pub rqk: CudaSlice<f32>,
    pub d_coords: CudaSlice<u32>,
    pub d_rinfo: CudaSlice<u32>,
    pub d_ainfo: CudaSlice<u32>,
    pub d_items: CudaSlice<u32>,
    pub n_items: usize,
    pub sel_blk: CudaSlice<i32>,
    pub sel_bias: CudaSlice<f32>,
    pub sel_n: CudaSlice<u32>,
    pub d_guard: CudaSlice<u32>,
    pub slots: Vec<Option<DashSlot>>,
}

impl DashDev {
    pub fn new(
        exec: &GpuExecutor,
        d: &Dash,
        n_layer: usize,
        slots: usize,
        cap: usize,
        max_ctx: usize,
    ) -> Result<Self, GpuError> {
        let kbar_blocks = max_ctx.div_ceil(d.block_size);
        let n_tok = DASH_D / 2 - 1;
        // the reference's `1 / tok_theta ** (j / n_tok)` in f32
        let inv: Vec<f32> = (0..n_tok)
            .map(|j| 1.0 / d.tok_theta.powf(j as f32 / n_tok as f32))
            .collect();
        let mut kbar = Vec::with_capacity(n_layer);
        let mut staging = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            kbar.push(exec.alloc(slots * kbar_blocks * DASH_D)?);
            staging.push(exec.alloc(slots * DASH_BLOCK * DASH_D)?);
        }
        Ok(DashDev {
            kbar_blocks,
            kbar,
            staging,
            inv_tok: exec.to_device(&inv)?,
            rqk: exec.alloc(cap * 2 * DASH_D)?,
            d_coords: exec.alloc_u32(cap * 2)?,
            d_rinfo: exec.alloc_u32(cap * 6)?,
            d_ainfo: exec.alloc_u32(cap * 4)?,
            d_items: exec.alloc_u32(cap * 6)?,
            n_items: 0,
            sel_blk: exec.alloc_i32(cap * DASH_KCAP)?,
            sel_bias: exec.alloc(cap * DASH_KCAP)?,
            sel_n: exec.alloc_u32(cap)?,
            d_guard: exec.alloc_u32(cap)?,
            slots: vec![None; slots],
        })
    }

    /// Device bytes, for the KV / state ledger.
    pub fn bytes(&self) -> u64 {
        let f = |n: usize| n as u64 * 4;
        self.kbar.iter().map(|k| f(k.len())).sum::<u64>()
            + self.staging.iter().map(|s| f(s.len())).sum::<u64>()
    }
}

impl GpuDeepseekOcr {
    /// The crop-tile cap a request plans under: the tower's own (v1's mmproj
    /// says 32), at most 16 on Unlimited-OCR-2 - its `infer` caps gundam at
    /// `max_crops = 16`, a decoder-side choice its byte-identical tower file
    /// cannot carry.
    pub(super) fn tile_cap(&self, tower: usize) -> usize {
        if self.hp.dash.is_some() {
            tower.min(16)
        } else {
            tower
        }
    }

    /// Record a prompt's page layout for `slot` (image rows as the image
    /// token id). Every prefill entry calls this before its first pass.
    pub(super) fn dash_admit(&mut self, slot: usize, ids: &[u32]) {
        let Some(d) = self.hp.dash else { return };
        let lay = DashLayout::build(ids, &d);
        if lay.n_pages == 0 {
            tracing::debug!(
                "unlimited-ocr-2: slot {slot} prompt has no <PAGE>-marked image - one-bucket layout \
                 (the checkpoint trains only on <PAGE><image> prompts)"
            );
        }
        if let Some(dd) = self.batch.as_mut().and_then(|b| b.dash.as_mut()) {
            dd.slots[slot] = Some(DashSlot::new(lay));
        }
    }

    /// Where the ring starts for `slot` once its prompt is in: `keep` on this
    /// lane, the prompt length otherwise.
    pub(super) fn ring_base(&self, slot: usize, n_rows: usize) -> usize {
        self.batch
            .as_ref()
            .and_then(|b| b.dash.as_ref())
            .and_then(|dd| dd.slots[slot].as_ref())
            .map_or(n_rows, |s| s.lay.keep)
    }

    /// Before a prompt's first head row: its slot's guard mode at row 0.
    pub(super) fn dash_head_guard(&mut self, slot: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let Some(dd) = self.batch.as_mut().and_then(|b| b.dash.as_mut()) else {
            return Ok(());
        };
        let mode = dd.slots[slot].as_ref().map_or(0, DashSlot::guard_mode);
        let mut v = dd
            .d_guard
            .try_slice_mut(0..1)
            .ok_or_else(|| GpuError::Driver("dash guard slice".into()))?;
        exec.stream
            .memcpy_htod(&[mode], &mut v)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(())
    }

    /// Build and upload one pass's router streams. A row whose position lies
    /// past its slot's prompt is a decode row: it advances the slot's state
    /// machine (exactly once - every pass uploads each fed token once) and
    /// gets a summary item even when it has nothing to summarize, so a
    /// captured decode graph's launch shapes never move.
    pub(super) fn dash_prepare(
        &mut self,
        toks: &[u32],
        pos: &[u32],
        slots: &[u32],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let Some(d) = self.hp.dash else { return Ok(()) };
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let ring = d.local_window as u32;
        let Some(dd) = bs.dash.as_mut() else {
            return Err(
                GpuError::Unsupported("unlimited-ocr-2 router state not enabled".into()).into(),
            );
        };
        let r = toks.len();
        let mut coords = Vec::with_capacity(2 * r);
        let mut rinfo = Vec::with_capacity(6 * r);
        let mut ainfo = Vec::with_capacity(4 * r);
        let mut guard = vec![0u32; r];
        let mut items: Vec<u32> = Vec::new();
        let bsz = d.block_size;
        let mut i = 0;
        while i < r {
            let (s, p) = (slots[i] as usize, pos[i] as usize);
            // A decode-band row at a prompt position is a HOLE: the scheduler's
            // dense step feeds (token 0, position 0) to every slot below its
            // high-water mark that is not prefilled yet - empty, encoding, or
            // still chunking its prompt. That is BOS at row 0, which recomputes
            // the prompt's own row 0 (it attends to itself only, no remote
            // block exists that early), so its KV write is harmless; what must
            // not happen is the router reading it as a prompt row - a summary
            // item would rebuild block 0's summary from that one key - or a
            // generated token stepping the slot's page machine.
            if i < dec && dd.slots[s].as_ref().is_none_or(|st| p < st.lay.prompt_len) {
                let c0 = dd.slots[s].as_ref().map_or([0, 0], |st| st.lay.coords[0]);
                coords.extend_from_slice(&c0);
                rinfo.extend_from_slice(&[s as u32, p as u32, ALL[0], ALL[1], 1, u32::MAX]);
                ainfo.extend_from_slice(&[s as u32, p as u32, u32::MAX, ring]);
                // n_chunk 0: the summary kernel's no-op item (fixed decode shapes)
                items.extend_from_slice(&[s as u32, 0, i as u32, 0, 0, 0]);
                i += 1;
                continue;
            }
            let st = dd.slots[s].as_mut().ok_or_else(|| {
                GpuError::Unsupported(format!("unlimited-ocr-2: slot {s} has no admitted prompt"))
            })?;
            if p >= st.lay.prompt_len {
                let (c, cand) = st.feed(toks[i], &d);
                coords.extend_from_slice(&c);
                let kept_hi = ((st.lay.keep - 1) / bsz) as u32;
                rinfo.extend_from_slice(&[
                    s as u32,
                    p as u32,
                    cand[0],
                    cand[1],
                    (p / bsz + 1) as u32,
                    kept_hi,
                ]);
                ainfo.extend_from_slice(&[s as u32, p as u32, st.lay.keep as u32, ring]);
                guard[i] = st.guard_mode();
                let blk = p / bsz;
                let live = p < st.lay.keep && blk < dd.kbar_blocks;
                items.extend_from_slice(&[
                    s as u32,
                    blk as u32,
                    i as u32,
                    u32::from(live),
                    if live { (p % bsz) as u32 } else { 0 },
                    u32::from(live && p % bsz != bsz - 1),
                ]);
                i += 1;
                continue;
            }
            // a run of this slot's prompt rows within one block
            let lay = &st.lay;
            let blk = p / bsz;
            let mut j = i;
            while j < r
                && slots[j] as usize == s
                && (pos[j] as usize) < lay.prompt_len
                && pos[j] as usize / bsz == blk
            {
                let q = pos[j] as usize;
                coords.extend_from_slice(&lay.coords[q]);
                rinfo.extend_from_slice(&[
                    s as u32,
                    q as u32,
                    lay.cand[q][0],
                    lay.cand[q][1],
                    lay.prompt_len.div_ceil(bsz) as u32,
                    u32::MAX,
                ]);
                ainfo.extend_from_slice(&[s as u32, q as u32, u32::MAX, ring]);
                j += 1;
            }
            let last = pos[j - 1] as usize;
            items.extend_from_slice(&[
                s as u32,
                blk as u32,
                i as u32,
                (j - i) as u32,
                (p % bsz) as u32,
                u32::from(last % bsz != bsz - 1),
            ]);
            i = j;
        }
        let up = |host: &[u32], dst: &mut CudaSlice<u32>| -> Result<(), GpuError> {
            if host.is_empty() {
                return Ok(());
            }
            let mut v = dst
                .try_slice_mut(0..host.len())
                .ok_or_else(|| GpuError::Driver("dash stream slice".into()))?;
            exec.stream
                .memcpy_htod(host, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))
        };
        up(&coords, &mut dd.d_coords)?;
        up(&rinfo, &mut dd.d_rinfo)?;
        up(&ainfo, &mut dd.d_ainfo)?;
        up(&items, &mut dd.d_items)?;
        up(&guard, &mut dd.d_guard)?;
        dd.n_items = items.len() / 6;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dash() -> Dash {
        Dash {
            block_size: 32,
            local_blocks: 4,
            local_window: 128,
            index_dim: 128,
            topk_blocks: 12,
            entmax_window: 16,
            sigma: 1.0,
            score_scale: 128.0 / 128f32.sqrt(),
            floor: 1e-4,
            image_only: true,
            image_token_id: 128_815,
            page_token_id: 128_827,
            eos_token_id: 1,
            per_page_image_tokens: 273,
            tok_theta: 10_000.0,
            page_theta: 256.0,
            page_mod: 2,
        }
    }

    /// `[0] + ([PAGE] + [IMG] * t) per page + 4 instruction ids`
    fn prompt(pages: &[usize]) -> Vec<u32> {
        let d = dash();
        let mut v = vec![0u32];
        for &t in pages {
            v.push(d.page_token_id);
            v.extend(std::iter::repeat_n(d.image_token_id, t));
        }
        v.extend([37460, 4366, 76466, 16]);
        v
    }

    #[test]
    fn layout_matches_the_reference_coordinates() {
        let d = dash();
        let ids = prompt(&[273, 273]);
        let l = DashLayout::build(&ids, &d);
        assert_eq!(l.prompt_len, 1 + 2 * 274 + 4);
        assert_eq!(l.n_pages, 2);
        // BOS bucket 0; page 0 = bucket 1 (marker tok 0, images 1..); page 1 =
        // bucket 2 (parity 0); the instruction back in bucket 0 at 1..4
        assert_eq!(l.coords[0], [0, 0]);
        assert_eq!(l.coords[1], [0, 1]);
        assert_eq!(l.coords[2], [1, 1]);
        assert_eq!(l.coords[275], [0, 0]);
        assert_eq!(l.coords[276], [1, 0]);
        assert_eq!(
            l.coords[549..553].to_vec(),
            vec![[1, 0], [2, 0], [3, 0], [4, 0]]
        );
        assert_eq!(l.pf_cnt, vec![1, 274, 274]);
        // page 0's image rows 2..=274 -> blocks 0..=8; page 1's 276..=548 -> 8..=17
        assert_eq!(l.page_blocks, vec![[0, 8], [8, 17]]);
        assert_eq!(l.cand[0], ALL); // BOS: unconstrained
        assert_eq!(l.cand[1], [0, 8]); // page 0's marker
        assert_eq!(l.cand[300], [8, 17]);
        assert_eq!(l.cand[550], ALL); // instruction
        // last image 548 -> block 17 ends at 576 > S = 553
        assert_eq!(l.keep, 576);
        assert_eq!((l.end_bucket, l.end_tok), (0, 4));
    }

    #[test]
    fn decode_state_follows_the_reference_machine() {
        let d = dash();
        let mut s = DashSlot::new(DashLayout::build(&prompt(&[273, 273]), &d));
        assert_eq!(s.guard_mode(), 1, "EOS banned before any page opens");
        // the first generated <PAGE> opens bucket 1 at its prompt count
        let (c, cand) = s.feed(d.page_token_id, &d);
        assert_eq!(c, [274, 1]);
        assert_eq!(cand, [0, 8]);
        let (c, _) = s.feed(10_212, &d);
        assert_eq!(c, [275, 1]);
        let (c, cand) = s.feed(d.page_token_id, &d);
        assert_eq!((c, cand), ([274, 0], [8, 17]));
        assert_eq!(s.guard_mode(), 0);
        // a third marker: past the prompt's pages -> EOS forced, bucket clamped
        let (c, _) = s.feed(d.page_token_id, &d);
        assert_eq!(c, [274, 0]);
        assert_eq!(s.guard_mode(), 2);
    }

    #[test]
    fn a_page_without_a_marker_is_one_bucket() {
        let d = dash();
        let mut ids = vec![0u32];
        ids.extend(std::iter::repeat_n(d.image_token_id, 273));
        ids.extend([1, 2, 3]);
        let l = DashLayout::build(&ids, &d);
        assert_eq!(l.n_pages, 0);
        assert_eq!(l.coords[10], [10, 0]);
        assert_eq!(DashSlot::new(l).guard_mode(), 0);
    }
}
