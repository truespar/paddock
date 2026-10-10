//! Shared prefix-block primitives for the paged radix caches.
//!
//! The dense `RadixKvCache` that used to live here (a copy-in/copy-out block
//! store for the non-paged KV mode) is gone: the paged pool +
//! `PagedRadix` zero-copy cache is the only prefix cache now - pool mode is
//! the serving default for every family, the newer families (gemma4, laguna,
//! granite) never had a dense lane, and the dense fallback was a silent
//! second regime waiting to swallow a config. What remains is what all the
//! paged caches share: the page-size constant, the image-content radix keys,
//! the image-span checkpoint guard, and the evict-ahead margin clamp.

/// Tokens per cache page. Prompts share their common leading full pages; the last
/// partial page (< this) is never cached, and re-prefilled (cheap).
pub const BLOCK_TOKENS: usize = 16;

/// How far behind a prompt's trailing checkpoint pair the hybrids' BACK-OFF
/// checkpoint sits. The trailing pair catches a re-rendered history, whose
/// divergence sits in the generation header; nothing caught a prompt whose
/// TAIL was rewritten - the same document with another question, an edited
/// last instruction - and a recurrent state can only resume at a snapshot, so
/// every such prompt re-prefilled whole: a 31K ledger asked a new question
/// took 24 s on Qwen3.8-27B (GB10), where vLLM and SGLang resume at an
/// aligned block. One boundary ~256 tokens back covers a rewritten tail of
/// that size for one more staged snapshot a prompt.
pub const BACKOFF_TOKENS: usize = 256;

/// The back-off boundary for a prompt whose last checkpoint boundary is `b1`
/// on a `step` grid (`step` >= BLOCK_TOKENS; the KV tier's run span when
/// armed): `BACKOFF_TOKENS` behind it - at least two steps - rounded down to
/// the grid. None when that would not sit strictly below the trailing pair
/// (`b1 - step`, `b1`), or would itself sit under BACKOFF_TOKENS - a prompt
/// that short re-prefills cheaply, and a checkpoint that shallow is under
/// every family's resume floor anyway.
pub fn backoff_cut(b1: usize, step: usize) -> Option<usize> {
    if paddock_models::dev_var_os!("PADDOCK_NO_CKPT_BACKOFF").is_some() {
        return None;
    }
    let step = step.max(BLOCK_TOKENS);
    let c = b1.checked_sub(BACKOFF_TOKENS.max(2 * step))? / step * step;
    (c >= BACKOFF_TOKENS && c + step < b1).then_some(c)
}

/// Stage F (the reply checkpoint) off switch, shared by every family that
/// snapshots its recurrent state during decode (nemotron, qwen35): with it
/// set the next turn resumes at the previous PROMPT's last boundary and
/// re-prefills the whole reply.
pub fn reply_ckpt_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_REPLY_CKPT").is_some()
}

/// The radix key for row `j` of an image whose content hashes to `h`.
///
/// Every image row of every prompt carries the same `<image>` placeholder id, so
/// a radix keyed on the row tokens treats two different pictures as the same
/// prefix and serves one image's KV for the other - "of two concurrent image
/// requests, the blue-image slot answered red". That is why multimodal prompts
/// were excluded from prefix caching engine-wide. The fix is to key image rows
/// on the picture's CONTENT instead, and this is the one definition of how, so
/// the families cannot drift apart on it.
///
/// Two properties, both load-bearing:
///
/// 1. **Different pictures differ.** `h` is the content hash each family's image
///    cache already computes, so two pictures agree here exactly when they agree
///    byte for byte.
/// 2. **It can never equal a token id.** The high bit is set, and real ids are
///    bounded by the vocab (~100-260k), so the image and text key spaces do not
///    overlap. Without this a text prompt could - however improbably - hash into
///    an image's path and adopt KV for rows it never wrote.
///
/// Folding in `j` keeps a picture's run internally ordered, so a prompt that
/// reuses only PART of an image still matches position by position rather than
/// matching any row against any other.
pub fn image_key_row(h: u64, j: usize) -> u32 {
    let mixed = h
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add((j as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9));
    // fold to 31 bits, then set the high bit
    0x8000_0000 | ((mixed ^ (mixed >> 32)) as u32 & 0x7fff_ffff)
}

/// The evict-ahead free margin an insert may hold back, clamped to what this
/// pool can actually honour.
///
/// Every family evicts down to a free margin after inserting, so eviction
/// happens off the admission path and freed ids come back through the free
/// list's LIFO reuse. The margins were tuned as ABSOLUTE block counts against
/// big-context servers (gemma4 2048, laguna 1024, granite 256) - and an
/// absolute count is a silent trap on a small one: a runner with max_ctx 8192
/// and one slot has a pool of a few hundred blocks, so `free < margin` holds
/// even with the pool completely empty. The loop then evicts everything the
/// insert just published, every time, and the prefix cache is not degraded but
/// off, with nothing anywhere saying so. Measured on gemma4 at 8k/1 slot: a
/// repeated identical prompt logged `matched 0` on every request, and `matched
/// 80` once the margin was manually zeroed.
///
/// A quarter of the pool is the ceiling: enough free blocks that admissions
/// still draw from the LIFO list, with the other three quarters left for the
/// retention the cache exists to hold. Warn once when the configured value had
/// to be cut, because "your margin is bigger than your pool" is a
/// configuration mistake the user cannot otherwise see.
pub fn evict_ahead_margin(configured: usize, pool_capacity: usize) -> usize {
    if configured == 0 {
        return 0; // explicitly disabled: evict only under real pressure
    }
    let ceiling = pool_capacity / 4;
    if configured > ceiling {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            tracing::warn!(
                "prefix cache: evict-ahead margin {configured} blocks exceeds a quarter of the \
                 {pool_capacity}-block pool; using {ceiling}. An unclamped margin this large \
                 would evict every prefix as soon as it was cached."
            );
        });
    }
    configured.min(ceiling)
}

/// Walk a checkpoint cut back to a page boundary at or before the start of any
/// image span it lands strictly inside.
///
/// gemma4v gives an image's rows MUTUAL visibility (it decodes them
/// non-causally), so a resume landing strictly inside a picture would
/// re-prefill rows whose attention reaches keys the adopted blocks already
/// hold, written in a different order. qwen35 used to do the same (an
/// attention bound at the span's last row); its rows are raster-causal now,
/// but it keeps the guard - a picture spliced whole per pass is what keeps its
/// per-pass encoding and image-keyed radix simple.
///
/// Rather than reason about whether that is benign, no CHECKPOINT is ever
/// attached inside an image span - and since a resume position is exactly a
/// checkpoint position, mid-span resumes cannot occur at all. No guard is then
/// needed on the resume side or in the prefill loop, and there is no state
/// where "the cut moved" and "the resume moved" can disagree.
///
/// Walking BACK rather than forward matters: forward would place the cut past
/// rows the tail must then re-prefill, and the tail is what re-writes the
/// picture. Back lands before the picture, so a span is either entirely adopted
/// or entirely re-prefilled.
///
/// Spans arrive in row order, so one reverse pass settles it: each step moves
/// the cut before a picture, which can only expose an earlier picture, never a
/// later one. Pure, so the rule is testable without a GPU.
pub fn cut_outside_image_spans(mut cut: usize, img_spans: &[(usize, usize)]) -> usize {
    for &(a, b) in img_spans.iter().rev() {
        if cut > a && cut < b {
            cut = a / BLOCK_TOKENS * BLOCK_TOKENS;
        }
    }
    cut
}

/// Where a multimodal prefill of rows `[start, t_len)` ends each pass, and
/// whether that end is one of the `ckpt` cuts (snapshot + publish there).
///
/// Text prompts reach the engine in scheduler chunks; a prompt carrying a
/// picture used to be prefilled from its start to its first checkpoint cut in
/// ONE pass, and the checkpoint cuts sit at the prompt's last page
/// boundaries - so the pass was the whole prompt. The shared prefill scratch
/// is planned for `pass` rows, and every longer pass regrew it to the prompt's
/// length and kept it: measured on qwen3.8-27B, an 18K-row picture prompt
/// took the scratch from 6.4 to 13.4 GB for the rest of the runner's life, and
/// the regrowth briefly held both. So passes end at most `pass` rows apart as
/// well as at the checkpoint cuts.
///
/// A pass may never end strictly inside a picture - its rows attend to the
/// picture's last row, which must be in the same pass - so an end that lands
/// inside one walks back to the picture's start (not to a page boundary: this
/// is a pass cut, not a checkpoint, and a page-aligned walk could land before
/// the pass even began). A picture longer than `pass` cannot be split at all
/// and runs as its own pass; the serving lane caps one picture's tokens at the
/// planned pass so that stays unreachable, but the rule is total without it.
///
/// `ckpt` must hold cuts outside every picture (`cut_outside_image_spans`),
/// and `start` must not lie inside one (it is 0 or a checkpoint position).
pub fn mm_pass_ends(
    start: usize,
    t_len: usize,
    ckpt: &[usize],
    img_spans: &[(usize, usize)],
    pass: usize,
) -> Vec<(usize, bool)> {
    let pass = pass.max(1);
    let mut out = Vec::new();
    let mut a = start;
    while a < t_len {
        let next_ckpt = ckpt.iter().copied().filter(|&c| c > a && c < t_len).min();
        let hard = next_ckpt.unwrap_or(t_len);
        let end = if hard - a <= pass {
            hard
        } else {
            let mut t = a + pass;
            for &(s, e) in img_spans.iter().rev() {
                if t > s && t < e {
                    t = s;
                }
            }
            if t > a {
                t
            } else {
                // a picture starts at `a` and outruns one pass: it goes whole
                img_spans
                    .iter()
                    .find(|&&(s, e)| s <= a && a < e)
                    .map_or(hard, |&(_, e)| e.min(hard))
            }
        };
        out.push((end, next_ckpt == Some(end)));
        a = end;
    }
    out
}

#[cfg(test)]
mod span_tests {
    use super::{BLOCK_TOKENS, cut_outside_image_spans as cut_outside};

    /// The property the whole guard exists for: an image's rows attend to their
    /// span's last position, so a checkpoint inside a picture would let a later
    /// turn resume mid-picture and re-prefill rows whose attention bound points
    /// past the cut. No checkpoint inside a span, no such resume.
    #[test]
    fn a_cut_never_lands_inside_a_picture() {
        // one picture occupying rows [20, 300)
        let spans = [(20usize, 300usize)];
        for cut in (0..400).step_by(BLOCK_TOKENS) {
            let out = cut_outside(cut, &spans);
            assert!(
                out <= 20 || out >= 300,
                "cut {cut} landed at {out}, inside [20, 300)"
            );
            assert!(out <= cut, "the cut may only move BACK, {cut} -> {out}");
            assert_eq!(out % BLOCK_TOKENS, 0, "cut {out} left the page grid");
        }
    }

    /// A cut clear of every picture is untouched - the common document shape is
    /// image first then a long text tail, and that tail must stay resumable.
    #[test]
    fn a_cut_past_the_picture_is_left_alone() {
        let spans = [(20usize, 300usize)];
        assert_eq!(cut_outside(320, &spans), 320);
        assert_eq!(cut_outside(300, &spans), 300);
        // and one before it
        assert_eq!(cut_outside(16, &spans), 16);
    }

    /// Several pictures: stepping back out of one must not strand the cut
    /// inside an earlier one.
    #[test]
    fn stepping_back_out_of_one_picture_clears_the_earlier_ones() {
        // pictures at [16, 100) and [104, 200) with only 4 rows between them
        let spans = [(16usize, 100usize), (104usize, 200usize)];
        for cut in (0..240).step_by(BLOCK_TOKENS) {
            let out = cut_outside(cut, &spans);
            for &(a, b) in &spans {
                assert!(out <= a || out >= b, "cut {cut} -> {out} inside [{a}, {b})");
            }
        }
    }

    /// A text-only prompt has no spans, so nothing moves.
    #[test]
    fn without_pictures_the_cut_is_the_text_cut() {
        for cut in [0usize, 16, 64, 1024] {
            assert_eq!(cut_outside(cut, &[]), cut);
        }
    }

    /// qwen35's shape: a system preamble, then a big picture, then the question.
    /// Both of the two-boundary rule's cuts land in the text tail and must
    /// survive untouched, or a document conversation resumes nowhere.
    #[test]
    fn the_text_tail_after_a_picture_stays_resumable() {
        // 24 rows of preamble, a 1440-row picture, then a 400-row tail
        let spans = [(24usize, 1464usize)];
        for cut in [1856usize, 1840] {
            assert_eq!(
                cut_outside(cut, &spans),
                cut,
                "tail cut {cut} was walked back"
            );
        }
    }
}

#[cfg(test)]
mod margin_tests {
    use super::evict_ahead_margin;

    /// The bug this closes: a margin tuned for a big-context server is larger
    /// than a small server's entire pool, so the evict-ahead loop runs until
    /// the radix is empty on every insert and the cache is silently off.
    /// Measured on gemma4 at max_ctx 8192 with one slot before the clamp.
    #[test]
    fn a_margin_larger_than_the_pool_cannot_empty_it() {
        // gemma4's default against a small server's few hundred blocks
        let pool = 400;
        let m = evict_ahead_margin(2048, pool);
        assert!(
            m < pool,
            "margin {m} still covers the whole {pool}-block pool"
        );
        // and it leaves the majority of the pool for retention, which is the
        // point - a cache that can hold nothing is not a cache
        assert!(
            pool - m >= pool * 3 / 4,
            "only {} blocks left to cache with",
            pool - m
        );
    }

    /// On the servers these numbers were tuned for, nothing changes.
    #[test]
    fn a_margin_the_pool_can_afford_is_untouched() {
        assert_eq!(evict_ahead_margin(2048, 40_000), 2048);
        assert_eq!(evict_ahead_margin(1024, 8_192), 1024);
        assert_eq!(evict_ahead_margin(256, 4_096), 256);
        // exactly at the ceiling is still affordable
        assert_eq!(evict_ahead_margin(1024, 4_096), 1024);
    }

    /// 0 means "evict only under real pressure" and must stay exactly that -
    /// the clamp must not turn a deliberate opt-out into a quarter-pool margin.
    #[test]
    fn zero_stays_disabled() {
        assert_eq!(evict_ahead_margin(0, 40_000), 0);
        assert_eq!(evict_ahead_margin(0, 0), 0);
    }

    /// A pool too small to spare anything asks for no margin at all, rather
    /// than one block that evicts the only page there was room for.
    #[test]
    fn a_tiny_pool_gets_no_margin() {
        for cap in 0..4 {
            assert_eq!(evict_ahead_margin(2048, cap), 0, "capacity {cap}");
        }
    }
}

#[cfg(test)]
mod pass_tests {
    use super::mm_pass_ends;

    /// Walk the plan and check the invariants every caller relies on: passes
    /// tile [start, t_len) in order, none is longer than `pass` unless it is
    /// exactly one oversized picture, none ends inside a picture, and every
    /// checkpoint cut is a pass end flagged as such.
    fn check(start: usize, t_len: usize, ckpt: &[usize], spans: &[(usize, usize)], pass: usize) {
        let plan = mm_pass_ends(start, t_len, ckpt, spans, pass);
        let mut a = start;
        for &(end, is_ckpt) in &plan {
            assert!(end > a, "pass [{a}, {end}) is empty or backwards");
            let oversized_picture = spans.iter().any(|&(s, e)| s == a && e == end);
            assert!(
                end - a <= pass || oversized_picture,
                "pass [{a}, {end}) is {} rows over a {pass}-row plan",
                end - a
            );
            for &(s, e) in spans {
                assert!(
                    !(end > s && end < e),
                    "pass end {end} inside picture [{s}, {e})"
                );
            }
            assert_eq!(is_ckpt, ckpt.contains(&end), "ckpt flag wrong at {end}");
            a = end;
        }
        assert_eq!(a, t_len, "the plan stops short of the prompt");
        for &c in ckpt {
            if c > start && c < t_len {
                assert!(plan.iter().any(|&(e, k)| e == c && k), "ckpt cut {c} lost");
            }
        }
    }

    /// The measured case: ~14K text rows then a 4096-row picture then the
    /// question. One pass used to cover everything up to the checkpoint cuts.
    #[test]
    fn a_long_document_with_a_picture_runs_in_planned_passes() {
        let spans = [(14_000usize, 18_096usize)];
        let ckpt = [18_192usize, 18_208];
        check(0, 18_243, &ckpt, &spans, 8192);
        let plan = mm_pass_ends(0, 18_243, &ckpt, &spans, 8192);
        assert!(plan.iter().all(|&(e, _)| e <= 18_243));
        assert!(plan.len() >= 3, "{plan:?}");
    }

    /// An end landing inside a picture walks back to the picture's start, and
    /// the picture then rides whole in the next pass.
    #[test]
    fn a_pass_end_inside_a_picture_moves_to_its_start() {
        let spans = [(6000usize, 10_096usize)];
        let plan = mm_pass_ends(0, 12_000, &[], &spans, 8192);
        assert_eq!(plan[0], (6000, false));
        assert_eq!(plan[1], (12_000, false));
        check(0, 12_000, &[], &spans, 8192);
    }

    /// Pictures back to back: stepping out of one must not strand the end in
    /// the one before it.
    #[test]
    fn adjacent_pictures_split_between_them() {
        let spans = [
            (100usize, 4196usize),
            (4196usize, 8292usize),
            (8292usize, 12_388usize),
        ];
        check(0, 12_500, &[], &spans, 8192);
        check(0, 12_500, &[], &spans, 4096);
    }

    /// A picture longer than a pass cannot be split; it runs alone and the
    /// text around it still chunks.
    #[test]
    fn an_oversized_picture_runs_as_its_own_pass() {
        let spans = [(3000usize, 19_384usize)];
        let plan = mm_pass_ends(0, 30_000, &[], &spans, 8192);
        assert_eq!(plan[0], (3000, false));
        assert_eq!(plan[1], (19_384, false));
        check(0, 30_000, &[], &spans, 8192);
    }

    /// A resumed prompt starts at its checkpoint and a short tail is one pass;
    /// a text-only or short prompt is exactly the old cut walk.
    #[test]
    fn short_tails_and_short_prompts_are_unchanged() {
        assert_eq!(
            mm_pass_ends(4096, 4200, &[], &[(0, 4000)], 8192),
            vec![(4200, false)]
        );
        assert_eq!(
            mm_pass_ends(0, 5000, &[4960, 4976], &[(16, 4112)], 8192),
            vec![(4960, true), (4976, true), (5000, false)]
        );
        check(0, 5000, &[4960, 4976], &[(16, 4112)], 8192);
    }

    /// Exhaustive sweep over small shapes so an off-by-one at any boundary
    /// shows up here rather than as a wrong answer.
    #[test]
    fn invariants_hold_across_a_sweep() {
        for pass in [7usize, 16, 33] {
            for t_len in [1usize, 15, 64, 97] {
                for s in (0..t_len).step_by(9) {
                    for len in [1usize, 5, 20, 40] {
                        let e = (s + len).min(t_len);
                        if e <= s {
                            continue;
                        }
                        let spans = [(s, e)];
                        let ckpt: Vec<usize> = [t_len / 2, t_len.saturating_sub(3)]
                            .into_iter()
                            .map(|c| super::cut_outside_image_spans(c, &spans))
                            .filter(|&c| c > 0 && c < t_len && (c <= s || c >= e))
                            .collect();
                        check(0, t_len, &ckpt, &spans, pass);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod backoff_tests {
    use super::*;

    #[test]
    fn backoff_sits_strictly_below_the_trailing_pair() {
        // a ~31K prompt on the page grid: 256 tokens behind its last boundary
        assert_eq!(backoff_cut(31744, BLOCK_TOKENS), Some(31488));
        // a prompt under ~512 tokens keeps its two (cheap to re-prefill)
        assert_eq!(backoff_cut(256, BLOCK_TOKENS), None);
        assert_eq!(backoff_cut(496, BLOCK_TOKENS), None);
        assert_eq!(backoff_cut(512, BLOCK_TOKENS), Some(256));
        // a coarse tier grid keeps it at least two steps back, on the grid
        assert_eq!(backoff_cut(4096, 512), Some(3072));
        assert_eq!(backoff_cut(1024, 512), None);
    }
}
