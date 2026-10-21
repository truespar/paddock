//! Stall accounting for scheduler branches that finish a tick with `continue`.
//! Marks close phases, not individual GPU operations. An unfinished phase may
//! include sampling/emission inside that branch; do not call its time CPU work.
use std::time::Duration;

pub(super) const NAMES: [&str; 6] = ["admit", "prefill", "mixed", "spec", "decode", "sample+emit"];

pub(super) fn spans(wall: Duration, marks: [Duration; 5]) -> ([Duration; 6], Option<&'static str>) {
    let mut spans = [Duration::ZERO; 6];
    let mut previous = Duration::ZERO;
    for (i, mark) in marks.into_iter().enumerate() {
        if mark.is_zero() {
            // None of the later endpoints ran. Charge the remainder to the
            // branch still executing, never to a fictitious host-sampling tail.
            spans[i] = wall.saturating_sub(previous);
            return (spans, Some(NAMES[i]));
        }
        let end = mark.max(previous).min(wall);
        spans[i] = end.saturating_sub(previous);
        previous = end;
    }
    spans[5] = wall.saturating_sub(previous);
    (spans, None)
}

/// Wall-clock microseconds, for trace lines.
pub(super) fn trace_us() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_early_exit_books_the_unfinished_branch_once() {
        let wall = Duration::from_millis(1000);
        for finished in 0..=5 {
            let marks = std::array::from_fn(|i| {
                if i < finished {
                    Duration::from_millis((i as u64 + 1) * 10)
                } else {
                    Duration::ZERO
                }
            });
            let (parts, early) = spans(wall, marks);
            assert_eq!(parts.iter().copied().sum::<Duration>(), wall);
            assert!(
                parts[..finished]
                    .iter()
                    .all(|p| *p == Duration::from_millis(10))
            );
            assert_eq!(
                parts[finished],
                wall - Duration::from_millis(finished as u64 * 10)
            );
            assert!(parts[finished + 1..].iter().all(Duration::is_zero));
            assert_eq!(early, (finished < 5).then_some(NAMES[finished]));
        }
    }

    #[test]
    fn dense_tail_is_only_measured_after_decode_really_closed() {
        let (parts, early) = spans(
            Duration::from_millis(80),
            [10, 20, 30, 40, 70].map(Duration::from_millis),
        );
        assert_eq!(parts.map(|p| p.as_millis()), [10, 10, 10, 10, 30, 10]);
        assert_eq!(early, None);
    }
}
