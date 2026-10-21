//! Keep the current boundaries and bounded, best-effort user-turn anchors.
//! Global LRU alone lets a fast tool loop evict the other active conversations.
//! No new GPU allocations, per-client identity, or changes to cached arithmetic.
use super::{Checkpoint, multimodal};

pub(super) fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_METAL_CACHE_TRACE").is_some())
}

pub(super) fn superseded(
    checkpoint: &Checkpoint,
    history: &[u32],
    images: &[multimodal::ImageKey],
    keep_from: usize,
) -> bool {
    checkpoint.history.len() < keep_from && covers(checkpoint, history, images)
}

fn covers(checkpoint: &Checkpoint, history: &[u32], images: &[multimodal::ImageKey]) -> bool {
    !checkpoint.history.is_empty()
        && history.starts_with(&checkpoint.history)
        && multimodal::prefix_images_match(&checkpoint.images, images, checkpoint.history.len())
}

/// CUDA's bounded user-turn retention, adapted to Metal's smaller fixed cache.
/// Always leave a rolling entry: a two-entry cache keeps one anchor and one
/// trailing cut. Optional captures skip anchors; page pressure/cold restores
/// can still reclaim them, so they never prevent admission or allocate state.
pub(super) fn mark_anchor(
    cache: &mut [Checkpoint],
    history: &[u32],
    images: &[multimodal::ImageKey],
    upto: usize,
) -> Option<usize> {
    let cap = (cache.len() / 4).max(2).min(cache.len().saturating_sub(1));
    if cap == 0 {
        return None;
    }
    let index = cache
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.reserved && c.history.len() <= upto && covers(c, history, images))
        .max_by_key(|(_, c)| c.history.len())
        .map(|(i, _)| i)?;
    cache[index].anchor = true;
    if cache.iter().filter(|c| c.anchor).count() > cap
        && let Some(old) = cache
            .iter()
            .enumerate()
            .filter(|(i, c)| *i != index && c.anchor)
            .min_by_key(|(_, c)| c.touched)
            .map(|(i, _)| i)
    {
        cache[old].anchor = false;
    }
    Some(cache[index].history.len())
}

/// Prefer an older boundary on the exact same token/image branch. Preserve both
/// current trailing cuts: the shorter one survives chat-template header edits.
/// Empty entries come next, then ordinary LRU for unrelated/new conversations.
/// Skip an optional capture if only anchors or current backups remain.
/// Reserved GPU copy destinations are never eligible, including aborted restores.
pub(super) fn replacement(
    cache: &[Checkpoint],
    history: &[u32],
    images: &[multimodal::ImageKey],
    keep_from: usize,
    pending_cut: bool,
) -> Option<usize> {
    replacement_excluding(cache, history, images, keep_from, pending_cut, None)
}

/// Generated tails must not destroy ordinary prompt reuse. Chat user-turn
/// anchors already provide that protection. Raw completions have no anchor:
/// keep their latest prompt snapshot and rotate only the second same-branch
/// entry (or an empty entry). Never steal a peer's state for this optional work.
pub(super) fn continuation_replacement(
    cache: &[Checkpoint],
    history: &[u32],
    images: &[multimodal::ImageKey],
    prefill_end: usize,
) -> Option<usize> {
    let protected = if cache.iter().any(|c| c.anchor && covers(c, history, images)) {
        None
    } else {
        Some(
            cache
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    !c.reserved && c.history.len() <= prefill_end && covers(c, history, images)
                })
                .max_by_key(|(_, c)| c.history.len())?
                .0,
        )
    };
    replacement_excluding(cache, history, images, history.len(), false, protected)
}

fn replacement_excluding(
    cache: &[Checkpoint],
    history: &[u32],
    images: &[multimodal::ImageKey],
    keep_from: usize,
    pending_cut: bool,
    protected: Option<usize>,
) -> Option<usize> {
    // Under a full cache, an anchored conversation's second trailing cut is
    // optional. Do not evict another conversation just to keep three cuts of
    // this one: max_batch * 2 entries must still serve max_batch conversations.
    let has_anchor = cache.iter().any(|c| c.anchor && covers(c, history, images));
    let has_trailing = has_anchor
        && (pending_cut
            || cache.iter().any(|c| {
                !c.reserved
                    && !c.anchor
                    && c.history.len() >= keep_from
                    && covers(c, history, images)
            }));
    cache
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            protected != Some(*i)
                && (protected.is_none() || c.history.is_empty() || covers(c, history, images))
                && !c.reserved
                && !c.anchor
                && !(c.history.len() >= keep_from && covers(c, history, images))
                && (!has_trailing
                    || c.history.is_empty()
                    || superseded(c, history, images, keep_from))
        })
        .min_by_key(|(_, c)| {
            let old = superseded(c, history, images, keep_from);
            (
                if old {
                    0
                } else if c.history.is_empty() {
                    1
                } else {
                    2
                },
                if old {
                    c.history.len() as u64
                } else {
                    c.touched
                },
            )
        })
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Most policy tests publish each cut immediately. The reserved-plan test
    // below explicitly supplies a not-yet-published GPU cut instead.
    fn replacement(
        cache: &[Checkpoint],
        history: &[u32],
        images: &[multimodal::ImageKey],
        keep_from: usize,
    ) -> Option<usize> {
        super::replacement(cache, history, images, keep_from, false)
    }

    fn checkpoint(token: u32, n: usize, touched: u64) -> Checkpoint {
        Checkpoint {
            history: vec![token; n],
            touched,
            ..Default::default()
        }
    }

    #[test]
    fn generated_tails_keep_unanchored_prompts_and_do_not_steal_peers() {
        let mut cache = vec![checkpoint(1, 96, 1), checkpoint(1, 112, 2)];
        for cut in (128..1024).step_by(64) {
            let i = continuation_replacement(&cache, &vec![1; cut], &[], 113).unwrap();
            assert_eq!(
                i, 0,
                "keep the prompt snapshot even without a user-turn anchor"
            );
            cache[i] = checkpoint(1, cut, cut as u64);
            assert_eq!(cache[1].history.len(), 112);
        }
        cache[0].reserved = true;
        cache.push(checkpoint(2, 96, 0));
        assert_eq!(continuation_replacement(&cache, &[1; 1088], &[], 113), None);
        assert_eq!(continuation_replacement(&cache, &[3; 1088], &[], 113), None);
        // An existing conversation anchor keeps the earlier retention policy.
        cache[0].reserved = false;
        cache[1].anchor = true;
        assert_eq!(
            continuation_replacement(&cache, &[1; 1088], &[], 113),
            Some(0)
        );
    }

    #[test]
    fn four_unanchored_decode_loops_keep_prompt_and_tail_with_eight_entries() {
        let mut cache = (1..=4)
            .flat_map(|token| [checkpoint(token, 96, 1), checkpoint(token, 112, 2)])
            .collect::<Vec<_>>();
        let mut lengths = [128; 4];
        for branch in [0, 1, 2, 3, 0, 0, 2, 0, 3, 1, 1, 3] {
            let cut = lengths[branch];
            let history = vec![branch as u32 + 1; cut];
            let i = continuation_replacement(&cache, &history, &[], 113).unwrap();
            cache[i] = checkpoint(branch as u32 + 1, cut, cut as u64);
            lengths[branch] += 64;
            for peer in 1..=4 {
                assert!(cache.iter().any(|c| c.history == vec![peer; 112]));
                assert_eq!(cache.iter().filter(|c| c.history[0] == peer).count(), 2);
            }
        }
    }

    #[test]
    fn anchors_choose_committed_prefix_and_are_bounded_by_recency() {
        let mut cache: Vec<_> = (0..8).map(|_| Checkpoint::default()).collect();
        cache[0] = checkpoint(1, 64, 1);
        cache[1] = checkpoint(1, 80, 2);
        cache[2] = checkpoint(1, 96, 3);
        cache[2].reserved = true;
        assert_eq!(mark_anchor(&mut cache, &[1; 112], &[], 79), Some(64));
        assert_eq!(mark_anchor(&mut cache, &[1; 112], &[], 96), Some(80));
        cache[3] = checkpoint(2, 64, 4);
        assert_eq!(mark_anchor(&mut cache, &[2; 112], &[], 96), Some(64));
        assert!(!cache[0].anchor, "oldest anchor becomes ordinary again");
        assert!(cache[1].anchor && cache[3].anchor);
        assert!(!cache[2].anchor, "in-flight GPU state is not publishable");
        assert_eq!(mark_anchor(&mut cache, &[3; 112], &[], 96), None);
        assert_eq!(mark_anchor(&mut cache, &[1; 48], &[], 96), None);
        assert_eq!(cache.iter().filter(|c| c.anchor).count(), 2);
    }

    #[test]
    fn long_tool_loop_retires_ordinary_ancestors_before_user_turn_anchor() {
        let mut cache: Vec<_> = (0..8).map(|_| Checkpoint::default()).collect();
        cache[0] = checkpoint(1, 64, 1);
        assert_eq!(mark_anchor(&mut cache, &[1; 65], &[], 64), Some(64));
        for first in (80..1024).step_by(32) {
            for cut in [first, first + 16] {
                let i = replacement(&cache, &vec![1; cut], &[], first).unwrap();
                cache[i] = checkpoint(1, cut, cut as u64);
            }
            assert!(cache[0].anchor && cache[0].history.len() == 64);
            for cut in [first, first + 16] {
                assert!(cache.iter().any(|c| c.history.len() == cut));
            }
        }
    }

    #[test]
    fn two_entry_cache_keeps_an_anchor_and_one_rolling_cut_without_extra_state() {
        let mut cache = vec![checkpoint(1, 64, 1), checkpoint(1, 96, 2)];
        mark_anchor(&mut cache, &[1; 112], &[], 64);
        // Two-entry caches cannot hold an anchor plus both trailing cuts.
        // Skip the second cut; never steal the first cut or the anchor.
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), None);
        assert_eq!(replacement(&cache, &[1; 144], &[], 128), Some(1));
        // A new user turn replaces the anchor, leaving a rolling entry.
        assert_eq!(mark_anchor(&mut cache, &[1; 112], &[], 96), Some(96));
        assert!(!cache[0].anchor && cache[1].anchor);
        cache[0].reserved = true;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), None);
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), None);
        assert_eq!(
            mark_anchor(&mut [checkpoint(1, 64, 1)], &[1; 96], &[], 64),
            None
        );
    }

    #[test]
    fn reserved_first_cut_also_prevents_stealing_another_conversations_state() {
        let mut cache = vec![
            checkpoint(1, 64, 1),
            checkpoint(1, 80, 2),
            checkpoint(2, 64, 3),
            checkpoint(2, 80, 4),
        ];
        mark_anchor(&mut cache, &[1; 81], &[], 80);
        assert_eq!(replacement(&cache, &[1; 96], &[], 96), Some(0));
        cache[0].history.clear();
        cache[0].reserved = true;
        assert_eq!(super::replacement(&cache, &[1; 112], &[], 96, true), None);
        assert_eq!(cache[2].history, vec![2; 64]);
        assert_eq!(cache[3].history, vec![2; 80]);
    }

    #[test]
    fn retires_ancestors_before_another_conversation_or_current_backup() {
        let cache = vec![
            checkpoint(1, 64, 1),
            checkpoint(2, 64, 9),
            checkpoint(2, 80, 10),
            checkpoint(2, 96, 11),
        ];
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), Some(1));
        assert!(!superseded(&cache[3], &[2; 112], &[], 96));
        assert_eq!(replacement(&cache, &[3; 112], &[], 96), Some(0));
    }

    #[test]
    fn uses_empty_entries_but_never_reserved_destinations() {
        let mut cache = vec![checkpoint(1, 64, 1), Checkpoint::default()];
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), Some(1));
        cache[1].reserved = true;
        assert_eq!(replacement(&cache, &[2; 112], &[], 96), Some(0));
        cache[0].reserved = true;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), None);
        assert_eq!(replacement(&[], &[1; 112], &[], 96), None);
    }

    #[test]
    fn skips_optional_capture_instead_of_overwriting_its_current_backup() {
        let mut cache = vec![checkpoint(1, 96, 1), Checkpoint::default()];
        cache[1].reserved = true;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), None);
        cache[1].reserved = false;
        assert_eq!(replacement(&cache, &[1; 112], &[], 96), Some(1));
    }

    #[test]
    fn four_interleaved_tool_loops_keep_both_boundaries_with_eight_entries() {
        let mut cache: Vec<_> = (0..8).map(|_| Checkpoint::default()).collect();
        let mut clock = 0;
        let mut lengths = [64; 4];
        // A fast branch runs ahead while the other three wait on tools.
        for branch in [0, 1, 2, 3, 0, 0, 0, 2, 0, 3, 1, 1, 0, 2, 3] {
            let first = lengths[branch];
            for cut in [first, first + 16] {
                let history = vec![branch as u32; cut];
                let i = replacement(&cache, &history, &[], first).unwrap();
                clock += 1;
                cache[i] = checkpoint(branch as u32, cut, clock);
            }
            lengths[branch] += 64;
            for (peer, &next) in lengths.iter().enumerate().filter(|(_, n)| **n > 64) {
                for cut in [next - 64, next - 48] {
                    assert!(cache.iter().any(|c| c.history == vec![peer as u32; cut]));
                }
            }
        }
    }

    #[test]
    fn bounded_anchors_do_not_crowd_out_four_active_tool_loops() {
        let mut cache: Vec<_> = (0..8).map(|_| Checkpoint::default()).collect();
        let mut clock = 0;
        let mut lengths = [64; 4];
        for branch in [0, 1, 2, 3, 0, 0, 0, 2, 3, 0, 1, 2, 3, 1] {
            let first = lengths[branch];
            for cut in [first, first + 16] {
                let history = vec![branch as u32; cut];
                if let Some(i) = replacement(&cache, &history, &[], first) {
                    clock += 1;
                    cache[i] = checkpoint(branch as u32, cut, clock);
                }
            }
            if first == 64 {
                mark_anchor(&mut cache, &vec![branch as u32; 81], &[], 80);
            }
            lengths[branch] += 64;
            for (peer, &next) in lengths.iter().enumerate().filter(|(_, n)| **n > 64) {
                assert!(
                    cache
                        .iter()
                        .any(|c| { c.history.len() >= next - 64 && c.history[0] == peer as u32 }),
                    "branch {peer} lost all of its trailing cuts"
                );
            }
            assert!(cache.iter().filter(|c| c.anchor).count() <= 2);
        }
    }

    #[test]
    fn a_diverged_or_shortened_branch_is_not_an_ancestor() {
        let c = checkpoint(2, 64, 1);
        assert!(!superseded(&c, &[2; 48], &[], 96));
        let mut changed = vec![2; 112];
        changed[63] = 3;
        assert!(!superseded(&c, &changed, &[], 96));
        assert!(!superseded(&Checkpoint::default(), &[2; 112], &[], 96));
    }
}
