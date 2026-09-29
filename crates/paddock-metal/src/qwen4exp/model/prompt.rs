//! Token-derived logical chunks, independent of cache hits and physical grants.
//! Message cuts round DOWN to whole KV pages. The lookahead that selects such
//! a cut is not itself cached: reuse must also compare the entire arithmetic
//! history, not merely token equality or the last contraction class.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct Markers {
    opener: u32,
    assistant: Vec<u32>,
}

impl Markers {
    pub(super) fn load(path: &Path) -> Option<Self> {
        let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(path).ok()?;
        let opener = tokenizer.token_to_id("<|im_start|>")?;
        let assistant = tokenizer.encode("<|im_start|>assistant\n").ok()?;
        // Only the validated Flash-Next ChatML header. Unknown tokenizers keep
        // the existing grid; never infer boundaries from ordinary text IDs.
        if tokenizer.encode("<|im_start|>").ok()? != [opener]
            || assistant.first() != Some(&opener)
            || assistant.len() < 2
            || tokenizer.decode(&assistant, false).ok()? != "<|im_start|>assistant\n"
        {
            return None;
        }
        Some(Self { opener, assistant })
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct Plan {
    ends: Vec<usize>,
    cuts: [usize; 2],
    #[cfg(test)]
    chunk: usize,
}

impl Plan {
    pub(super) fn new(tokens: &[u32], chunk: usize, markers: Option<&Markers>) -> Self {
        let mut points = Vec::new();
        if let Some(markers) = markers {
            for (at, &token) in tokens.iter().enumerate() {
                if token != markers.opener {
                    continue;
                }
                // We retain the latest prompt boundaries, not a dedicated
                // system-prefix checkpoint. Cutting at the second message
                // would fragment cold work without retaining that state.
                if tokens[at..].starts_with(&markers.assistant) {
                    points.push(at / BLOCK_TOKENS * BLOCK_TOKENS);
                }
            }
        }
        Self::build(tokens.len(), chunk, &points)
    }

    #[cfg(test)]
    pub(super) fn grid(length: usize, chunk: usize) -> Self {
        Self::build(length, chunk, &[])
    }

    fn build(length: usize, chunk: usize, points: &[usize]) -> Self {
        assert!(length > 0 && chunk > 0 && chunk.is_multiple_of(BLOCK_TOKENS));
        let body = length - 1;
        let minimum = 256.min(chunk);
        let (mut start, mut next) = (0, 0);
        let mut ends = Vec::new();
        let mut last = 0;
        while start < body {
            while next < points.len() && points[next] < start + minimum {
                next += 1;
            }
            let mut end = (start + chunk).min(body);
            if let Some(&point) = points.get(next)
                && point < end
            {
                end = point;
            }
            last = start;
            ends.push(end);
            start = end;
        }
        // The final prompt token always uses singleton decode arithmetic.
        if body > 0
            && body.is_multiple_of(BLOCK_TOKENS)
            && ends.last().is_some_and(|&end| end == body)
            && body - last == chunk
        {
            last = body;
        }
        ends.push(length);
        let tail = length.saturating_sub(1 + BLOCK_TOKENS) / BLOCK_TOKENS * BLOCK_TOKENS;
        let cuts = if tail > last && length > 256 {
            [last, tail]
        } else if last == 0 {
            [0, 0]
        } else if points.binary_search(&last).is_ok() {
            // The message-aligned fallback is already before the mutable
            // assistant header. An extra page just before it adds a tiny
            // dispatch and a state copy without improving useful reuse.
            [last, last]
        } else {
            [last.saturating_sub(BLOCK_TOKENS), last]
        };
        Self {
            ends,
            cuts,
            #[cfg(test)]
            chunk,
        }
    }

    pub(super) fn at(&self, offset: usize) -> (usize, usize) {
        let i = self.ends.partition_point(|&end| end <= offset);
        let end = self.ends[i];
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        let logical = end - start;
        #[cfg(test)]
        let logical = if i + 1 < self.ends.len()
            && super::serving::CANONICAL_PREFILL_FOR_TEST.with(|v| v.get())
        {
            self.chunk
        } else {
            logical
        };
        (logical, end - offset)
    }

    pub(super) fn cuts(&self) -> [usize; 2] {
        self.cuts
    }

    pub(super) fn until_cut(&self, offset: usize) -> usize {
        self.cuts
            .iter()
            .copied()
            .filter(|&n| n > offset)
            .min()
            .unwrap_or_else(|| *self.ends.last().expect("nonempty prompt"))
            - offset
    }

    /// Coalesce equal arithmetic classes, including a page cut inside a logical
    /// chunk. Physical slices never enter this key. This preserves the previous
    /// safe reuse of different logical row counts with identical contractions.
    pub(super) fn contract(&self, length: usize, classes: &[u16]) -> Vec<(usize, u16)> {
        let mut result: Vec<(usize, u16)> = Vec::new();
        let mut start = 0;
        for &end in &self.ends {
            if start >= length {
                break;
            }
            let class = classes[self.at(start).0];
            let stop = end.min(length);
            if let Some(last) = result.last_mut()
                && last.1 == class
            {
                last.0 = stop;
            } else {
                result.push((stop, class));
            }
            start = end;
        }
        result
    }
}

impl FlashNext {
    pub(super) fn prompt_plan(&self, tokens: &[u32]) -> Plan {
        Plan::new(tokens, self.chunk, self.markers.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn markers() -> Markers {
        Markers {
            opener: 99,
            assistant: vec![99, 98, 97],
        }
    }
    fn conversation() -> Vec<u32> {
        let mut tokens = vec![5; 1700];
        tokens[0] = 99;
        tokens[670] = 99;
        tokens[1650..1653].copy_from_slice(&[99, 98, 97]);
        tokens
    }

    #[test]
    fn page_aligned_message_cuts_and_singleton_end() {
        let tokens = conversation();
        let plan = Plan::new(&tokens, 1024, Some(&markers()));
        assert_eq!(plan.ends, [1024, 1648, 1699, 1700]);
        assert_eq!(plan.cuts(), [1648, 1680]);
        assert_eq!(plan.at(1648), (51, 51));
        assert_eq!(plan.at(1661), (51, 38));
        assert_eq!(plan.at(1699), (1, 1));
        let short_header = Plan::new(&tokens[..1658], 1024, Some(&markers()));
        assert_eq!(short_header.ends, [1024, 1648, 1657, 1658]);
        assert_eq!(short_header.cuts(), [1648, 1648]);
        assert_eq!(short_header.until_cut(1024), 624);
    }

    #[test]
    fn appended_turn_preserves_message_prefix_but_not_incompatible_tail() {
        let tokens = conversation();
        let mut follow = tokens.clone();
        follow.extend([6; 320]);
        let a = Plan::new(&tokens, 1024, Some(&markers()));
        let b = Plan::new(&follow, 1024, Some(&markers()));
        let classes = super::super::super::mlx::arithmetic_classes();
        assert_eq!(a.contract(1648, &classes), b.contract(1648, &classes));
        assert_ne!(a.contract(1680, &classes), b.contract(1680, &classes));
        follow[1650] = 6;
        let changed = Plan::new(&follow, 1024, Some(&markers()));
        assert_ne!(a.contract(1648, &classes), changed.contract(1648, &classes));
        // Removing lookahead may be safe when every arithmetic class remains
        // identical. Do not discard that existing equivalence optimization.
        let mut equivalent = vec![5; 2070];
        equivalent[2002..2005].copy_from_slice(&[99, 98, 97]);
        let selected = Plan::new(&equivalent, 1024, Some(&markers()));
        equivalent[2002] = 6;
        let removed = Plan::new(&equivalent, 1024, Some(&markers()));
        assert_eq!(
            selected.contract(2000, &classes),
            removed.contract(2000, &classes)
        );
        // A shorter boundary really changes the class. Its selecting header
        // lies AFTER the retained page; token equality alone is insufficient.
        follow[1362..1365].copy_from_slice(&[99, 98, 97]);
        let selected = Plan::new(&follow, 1024, Some(&markers()));
        follow[1362] = 6;
        let removed = Plan::new(&follow, 1024, Some(&markers()));
        assert_ne!(
            selected.contract(1360, &classes),
            removed.contract(1360, &classes)
        );
    }

    #[test]
    fn grid_is_unchanged_and_grants_do_not_select_arithmetic() {
        for chunk in [128, 256, 512, 1024] {
            for length in 1..=4097 {
                let plan = Plan::grid(length, chunk);
                assert_eq!(plan.cuts(), super::super::prefix::cuts(length, chunk));
                for offset in [0, length / 2, length - 1] {
                    assert_eq!(
                        plan.at(offset),
                        super::super::serving::logical_chunk(length, offset, chunk)
                    );
                }
            }
        }
        let tokens = conversation();
        let plan = Plan::new(&tokens, 1024, Some(&markers()));
        for budget in [1, 3, 13, 32, 125, 511, 2048] {
            let mut at = 0;
            while at < tokens.len() {
                let (logical, remaining) = plan.at(at);
                let count = budget.min(remaining).min(plan.until_cut(at));
                assert!(count > 0 && count <= logical);
                for i in at..at + count {
                    assert_eq!(plan.at(i).0, logical);
                }
                at += count;
            }
        }
    }
}
