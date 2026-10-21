//! Model-free speculative-decoding drafters. The verify side lives on each
//! model (`forward_spec_batch` / the B=1 spec loop); this module is the
//! host-side proposal machinery shared by the serving scheduler, the models'
//! own greedy-spec loops, and the benches.

/// Shortest and longest suffix lengths the drafter matches on. The floor is
/// 3: a 2-gram fallback measured precision-NEGATIVE in G3 (it fired on
/// ", " glue and proposed 473 garbage drafts with 0 accepted = a 0.6x
/// slowdown). The ladder above 3 is pure precision: any occurrence of a
/// longer suffix contains the 3-suffix at the same end position, so the
/// match SET equals 3-gram-only - only the chosen continuation improves
/// (longest matching context wins, which disambiguates template-boundary
/// grams and survives more verify rows).
const GRAM_MIN: usize = 3;
const GRAM_MAX: usize = 8;

/// Model-free longest-suffix n-gram drafter (prompt-lookup decoding,
/// SuffixDecoding-lite): map every n-gram (n = 3..=8) of the context to the
/// position right after its most recent occurrence; at draft time try the
/// LONGEST suffix first and propose the tokens that followed its last prior
/// match. No match at any n -> no draft. Two positions are kept per gram
/// because the newest entry is always the current suffix itself (its
/// continuation does not exist yet). Keys are FNV-1a hashes (collisions are
/// safe: every draft is verified, a collision only costs acceptance).
/// Purely host-side: a push appends, and the first `draft` after pushes
/// indexes them (six map inserts a token) - so a slot that never drafts
/// (speculation off, or a batch the ladder holds at K = 0) pays a Vec push
/// per token, prompt seeding included; a lookup is up to six probes, ~1 us
/// against a >5 ms verify pass.
#[derive(Default)]
pub struct NgramDraft {
    hist: Vec<u32>,
    /// `hist[..indexed]` is in `grams`.
    indexed: usize,
    // per suffix length n (index n - GRAM_MIN): gram hash -> (most recent
    // end position, previous end position); "end position" = index just past
    // the gram, where its continuation starts. 0 is a safe "none" sentinel.
    grams: [std::collections::HashMap<u64, (u32, u32)>; GRAM_MAX - GRAM_MIN + 1],
}

fn gram_hash(gram: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in gram {
        h = (h ^ t as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl NgramDraft {
    pub fn push(&mut self, t: u32) {
        self.hist.push(t);
    }

    /// Index every gram ending past `indexed`, in push order.
    fn catch_up(&mut self) {
        while self.indexed < self.hist.len() {
            self.indexed += 1;
            let end = self.indexed;
            for g in GRAM_MIN..=GRAM_MAX.min(end) {
                let key = gram_hash(&self.hist[end - g..end]);
                let e = self.grams[g - GRAM_MIN].entry(key).or_insert((0, 0));
                *e = (end as u32, e.0);
            }
        }
    }

    /// Propose up to `k` tokens continuing the current suffix (may be fewer
    /// if the match runs into the end of history), or empty on no match at
    /// any suffix length. Longest matching suffix wins.
    pub fn draft(&mut self, k: usize) -> Vec<u32> {
        self.catch_up();
        let n = self.hist.len();
        for g in (GRAM_MIN..=GRAM_MAX.min(n)).rev() {
            let key = gram_hash(&self.hist[n - g..n]);
            let Some(&(recent, prev)) = self.grams[g - GRAM_MIN].get(&key) else {
                continue;
            };
            // the most recent entry is always the current suffix itself
            let p = if (recent as usize) < n { recent } else { prev } as usize;
            if p >= g && p < n {
                return self.hist[p..(p + k).min(n)].to_vec();
            }
        }
        Vec::new()
    }
}
