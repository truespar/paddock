//! Where a paged recurrent-state checkpoint's bytes live (issue #33).
//!
//! A checkpoint in the paged backing ([`crate::paged_radix::PagedRadix::set_state_paged`])
//! is spread over KV pool pages. One page contributes its slot in every plane
//! the layout lists - the K and V plane of each full-attention layer - and the
//! checkpoint's flat bytes fill those slots page by page, planes in order
//! inside each page. That is the record layout the KV tier's transport
//! already uses for KV runs (block-major, a record = the concatenated plane
//! slots), so a paged checkpoint stored in RAM is byte for byte the flat blob
//! plus padding at the tail, and the tier ships it as a page list with no
//! special case.
//!
//! Everything here is address arithmetic over `u64` device pointers; the
//! copies themselves are the pack's `batched_copy` (one CTA per descriptor,
//! 16-byte units), so the checkpoint moves exactly as the flat one did - only
//! the descriptor list is longer (~9.7k for a 27B checkpoint over fp8 KV).

use crate::kv_pool::BlockId;

/// Checkpoints one turn writes: the two prompt cuts (RC7 - a hybrid resumes
/// at the last two page boundaries) and the reply's. The live turns' working
/// set, and so the part of the checkpoint demand a plan must back on top of
/// full context: without it a slot at its full window could not keep the
/// checkpoint its own next turn resumes from. A prompt's back-off cut
/// (`prefix_cache::backoff_cut`) is not counted: an appending next turn never
/// resumes there, so it rides the evictable want.
pub const CKPTS_PER_TURN: u64 = 3;
/// Cap on that mandatory part - and the old flat floor. Until issue #33 each
/// hybrid family reserved at least this many as a fixed pool whatever the
/// width, which alone kept a 3090 from serving a 27B past ~21k at one slot; a
/// wide server now reserves no more than it did, a narrow one only its turns.
pub const CKPTS_FLOOR: u64 = 16;
/// Cap on the want: a 128-distinct-prefix working set at two cuts a prompt.
pub const CKPTS_MAX: u64 = 256;

/// The checkpoint pages a plan asks for, in pool blocks: (mandatory, wanted
/// beyond that). `want_per_slot` is the family's measured demand (qwen35 6,
/// nemotron 8 - an agentic cohort's turns arrive as a wave, so one wave's
/// worth is not enough). The want beyond the mandatory part rides the plan's
/// retention and is bought only while the grant affords it.
///
/// One definition per family's plan, width sizer and allocation, so they
/// cannot disagree - they once did, and only the allocation honoured the dev
/// override, which is how a 256-checkpoint A/B arm asked the allocator for
/// ~42.5 GiB against a 14.61 GiB plan.
pub fn page_demand(slots: usize, pages_per_ckpt: usize, want_per_slot: u64) -> (usize, usize) {
    if pages_per_ckpt == 0 {
        return (0, 0); // no recurrent state to checkpoint
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_KV_STATE_CKPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return (n * pages_per_ckpt, 0);
    }
    let must = (slots as u64 * CKPTS_PER_TURN).min(CKPTS_FLOOR);
    let want = (slots as u64 * want_per_slot).clamp(CKPTS_FLOOR, CKPTS_MAX);
    (
        must as usize * pages_per_ckpt,
        want.saturating_sub(must) as usize * pages_per_ckpt,
    )
}

/// Descriptor capacity (u64 words) for one checkpoint's page-split copy: a
/// triple per page slot it spans, plus the partial slots each of its
/// `segments` flat spans can open and close.
pub fn desc_cap(ckpt_bytes: u64, min_slot: u64, segments: usize) -> usize {
    3 * (ckpt_bytes.div_ceil(min_slot.max(16)) as usize + 2 * segments + 2)
}

/// Which way a copy runs relative to the pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    /// flat span -> checkpoint pages (a snapshot)
    ToPages,
    /// checkpoint pages -> flat span (a restore)
    FromPages,
}

/// The planes one page's record is made of, in record order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageLayout {
    /// (device base, slot bytes) per plane. A plane is `pool_blocks` slots
    /// back to back, so block `b`'s slot starts at `base + b * slot`.
    planes: Vec<(u64, u64)>,
    /// Running start of each plane's slot inside a record.
    starts: Vec<u64>,
    payload: u64,
}

impl PageLayout {
    /// `planes` as (base, slot bytes). Every slot must be a 16-byte multiple
    /// so each descriptor stays 16-aligned; a zero-byte plane is dropped.
    pub fn new(planes: impl IntoIterator<Item = (u64, u64)>) -> Self {
        let planes: Vec<(u64, u64)> = planes.into_iter().filter(|p| p.1 > 0).collect();
        debug_assert!(
            planes
                .iter()
                .all(|&(b, s)| b.is_multiple_of(16) && s.is_multiple_of(16)),
            "page slots must be 16-aligned"
        );
        let mut starts = Vec::with_capacity(planes.len());
        let mut payload = 0u64;
        for &(_, slot) in &planes {
            starts.push(payload);
            payload += slot;
        }
        Self {
            planes,
            starts,
            payload,
        }
    }

    /// Bytes one page holds for a checkpoint.
    pub fn payload(&self) -> u64 {
        self.payload
    }

    /// Pages a checkpoint of `bytes` needs.
    pub fn pages_for(&self, bytes: u64) -> usize {
        bytes.div_ceil(self.payload.max(1)) as usize
    }

    /// The planes as (base, slot bytes), for the tier's transfer spec.
    pub fn planes(&self) -> &[(u64, u64)] {
        &self.planes
    }

    /// Append `batched_copy` triples (src, dst, len) moving `len` bytes
    /// between the contiguous device span at `flat` and checkpoint bytes
    /// `[off, off + len)` held in `pages`. Split at every slot boundary, so
    /// each triple stays inside one plane slot of one page.
    pub fn push_copy(
        &self,
        pages: &[BlockId],
        off: u64,
        flat: u64,
        len: u64,
        dir: Dir,
        out: &mut Vec<u64>,
    ) {
        debug_assert!(
            off.is_multiple_of(16) && len.is_multiple_of(16) && flat.is_multiple_of(16),
            "checkpoint segments are 16-aligned"
        );
        debug_assert!(
            off + len <= pages.len() as u64 * self.payload,
            "copy runs past the checkpoint's pages"
        );
        let mut done = 0u64;
        while done < len {
            let pos = off + done;
            let page = (pos / self.payload) as usize;
            let within = pos % self.payload;
            // the plane whose slot holds `within` (few planes: a scan is fine)
            let p = self.starts.partition_point(|&s| s <= within) - 1;
            let (base, slot) = self.planes[p];
            let in_slot = within - self.starts[p];
            let chunk = (slot - in_slot).min(len - done);
            let dev = base + pages[page] as u64 * slot + in_slot;
            let (src, dst) = match dir {
                Dir::ToPages => (flat + done, dev),
                Dir::FromPages => (dev, flat + done),
            };
            out.extend_from_slice(&[src, dst, chunk]);
            done += chunk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two planes of 32-byte slots (a toy of one full-attn layer's K and V).
    fn layout() -> PageLayout {
        PageLayout::new([(0x1000, 32), (0x9000, 32)])
    }

    /// Resolve checkpoint byte `pos` the slow way: page, then plane, then offset.
    fn addr(l: &PageLayout, pages: &[BlockId], pos: u64) -> u64 {
        let page = pages[(pos / l.payload()) as usize] as u64;
        let mut w = pos % l.payload();
        for &(base, slot) in l.planes() {
            if w < slot {
                return base + page * slot + w;
            }
            w -= slot;
        }
        unreachable!()
    }

    #[test]
    fn demand_is_the_live_turns_then_the_want() {
        // one slot: its turn (3) is mandatory, the rest of the 16-floor want optional
        assert_eq!(page_demand(1, 10, 6), (30, 130));
        // eight slots: 24 turns capped at the old floor, want 48 (qwen35) / 64 (nemotron)
        assert_eq!(page_demand(8, 10, 6), (160, 320));
        assert_eq!(page_demand(8, 10, 8), (160, 480));
        // the want never exceeds its cap
        assert_eq!(page_demand(64, 1, 6), (16, 240));
        assert_eq!(page_demand(4, 0, 6), (0, 0));
    }

    #[test]
    fn payload_and_pages() {
        let l = layout();
        assert_eq!(l.payload(), 64);
        assert_eq!(l.pages_for(64), 1);
        assert_eq!(l.pages_for(65), 2);
        assert_eq!(l.pages_for(0), 0);
    }

    #[test]
    fn every_byte_lands_where_the_record_layout_says() {
        let l = layout();
        let pages = [5u32, 2, 9];
        // a segment starting mid-slot and crossing plane and page boundaries
        let (off, len, flat) = (16u64, 144u64, 0x7000_0000u64);
        let mut d = Vec::new();
        l.push_copy(&pages, off, flat, len, Dir::ToPages, &mut d);
        let mut covered = 0u64;
        for t in d.chunks(3) {
            let (src, dst, n) = (t[0], t[1], t[2]);
            assert!(n > 0 && n.is_multiple_of(16));
            for b in (0..n).step_by(16) {
                let pos = off + (src + b - flat);
                assert_eq!(dst + b, addr(&l, &pages, pos), "byte {pos}");
            }
            covered += n;
        }
        assert_eq!(covered, len, "the whole segment, once");
    }

    #[test]
    fn a_restore_is_the_snapshot_with_src_and_dst_swapped() {
        let l = layout();
        let pages = [3u32, 1];
        let (mut to, mut from) = (Vec::new(), Vec::new());
        l.push_copy(&pages, 0, 0x4000, 128, Dir::ToPages, &mut to);
        l.push_copy(&pages, 0, 0x4000, 128, Dir::FromPages, &mut from);
        assert_eq!(to.len(), from.len());
        for (a, b) in to.chunks(3).zip(from.chunks(3)) {
            assert_eq!((a[0], a[1], a[2]), (b[1], b[0], b[2]));
        }
    }

    #[test]
    fn descriptors_never_straddle_a_slot() {
        let l = PageLayout::new([(0, 48), (4096, 16), (8192, 64)]);
        let pages: Vec<BlockId> = (0..8).rev().collect();
        let mut d = Vec::new();
        l.push_copy(&pages, 32, 0x10_0000, l.payload() * 5, Dir::ToPages, &mut d);
        for t in d.chunks(3) {
            let (dst, n) = (t[1], t[2]);
            let (base, slot) = *l
                .planes()
                .iter()
                .rev()
                .find(|&&(b, _)| b <= dst)
                .expect("inside a plane");
            let in_slot = (dst - base) % slot;
            assert!(in_slot + n <= slot, "descriptor crosses a slot edge");
        }
    }
}
