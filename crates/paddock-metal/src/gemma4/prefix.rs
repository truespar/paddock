use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static FAIL_PUBLISH_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A snapshot contains only the last ring's tokens, so a matching global KV
/// prefix is not sufficient. All window positions the next query can address
/// must still be in this snapshot. Leave a token to recompute selected logits.
fn reusable(cached: &[u32], tokens: &[u32], ring: usize, window: usize) -> usize {
    let common = cached
        .iter()
        .zip(tokens)
        .take_while(|(a, b)| a == b)
        .count()
        .min(tokens.len().saturating_sub(1));
    if common == 0
        || common.saturating_add(1).saturating_sub(window) < cached.len().saturating_sub(ring)
    {
        0
    } else {
        common
    }
}
impl Gemma4 {
    pub(super) fn evict(&mut self) -> bool {
        let Some(i) = self
            .cache
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.history.is_empty())
            .min_by_key(|(_, c)| c.touched)
            .map(|(i, _)| i)
        else {
            return false;
        };
        self.cache[i].table.clear(&mut self.pool);
        self.cache[i].history.clear();
        self.cache[i].images.clear();
        true
    }
    fn copy_ring(&self, from: usize, to: usize, history: usize) -> Result<()> {
        let cmd = self.device.begin()?;
        for l in self.layers.iter().filter(|l| l.sliding) {
            let stride = self.ring * l.kv_width() / 2;
            let words = history.min(self.ring) * l.kv_width() / 2;
            for b in [&l.keys, &l.values] {
                copy_words(&cmd, b, b, from * stride, to * stride, words);
            }
        }
        self.dflash_checkpoint(&cmd, from, to);
        cmd.finish()?;
        Ok(())
    }
    pub(super) fn prepare(&mut self, slot: usize, tokens: &[u32]) -> Result<usize> {
        self.prepare_mm(slot, tokens, None)
    }
    pub(super) fn prepare_mm(
        &mut self,
        slot: usize,
        tokens: &[u32],
        layout: Option<multimodal::Layout>,
    ) -> Result<usize> {
        self.require_committed()?;
        if self
            .prefill_phase
            .as_ref()
            .is_some_and(|p| p.contains(slot))
        {
            return Err(MetalError::Model(
                "Gemma slot has a suspended prefill".into(),
            ));
        }
        if slot >= self.slots.len()
            || tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= self.vocab)
        {
            return Err(MetalError::Model(
                "invalid Gemma prefill slot/tokens/context".into(),
            ));
        }
        self.slots[slot].table.clear(&mut self.pool);
        self.slots[slot].history.clear();
        self.slots[slot].reused = 0;
        self.slots[slot].mm = layout;
        if let Some(d) = &mut self.dflash {
            // A fresh admission changes the cohort's acceptance distribution.
            d.budget = Default::default();
        }
        if let Some(d) = &mut self.mtp {
            d.cursor[slot] = None;
        }
        let hit = self
            .cache
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let n = reusable(&c.history, tokens, self.ring, self.window);
                let n = multimodal::prefix_cut(
                    &c.images,
                    self.slots[slot]
                        .mm
                        .as_ref()
                        .map_or(&[], |m| m.keys.as_slice()),
                    n,
                );
                // Walking a cut back out of an image may invalidate ring
                // coverage even when the untrimmed common prefix was safe.
                let n = if n.saturating_add(1).saturating_sub(self.window)
                    < c.history.len().saturating_sub(self.ring)
                {
                    0
                } else {
                    n
                };
                (i, n)
            })
            .filter(|(_, n)| *n > 0)
            .max_by_key(|&(_, n)| n);
        let Some((i, n)) = hit else {
            return Ok(0);
        };
        self.copy_ring(self.slots.len() + i, slot, self.cache[i].history.len())?;
        let blocks = &self.cache[i].table.blocks()[..n.div_ceil(BLOCK_TOKENS)];
        self.slots[slot].table.share_prefix(blocks, &mut self.pool);
        self.slots[slot].history.extend_from_slice(&tokens[..n]);
        self.slots[slot].reused = n;
        self.clock += 1;
        self.cache[i].touched = self.clock;
        Ok(n)
    }
    pub(super) fn publish(&mut self, slot: usize) -> Result<()> {
        self.require_committed()?;
        if self.slots[slot].history.is_empty() {
            return Ok(());
        }
        // One bounded snapshot per slot. Publication is after GPU completion;
        // aborts never publish partial prompts as completed prefixes.
        // Invalidate the old owner BEFORE overwriting its ring. If submission
        // fails after a partial copy, stale metadata must not expose that
        // partially replaced ring to a later admission.
        self.cache[slot].history.clear();
        self.cache[slot].images.clear();
        self.cache[slot].table.clear(&mut self.pool);
        #[cfg(test)]
        if FAIL_PUBLISH_FOR_TEST.get() {
            return Err(MetalError::Model("injected prefix-copy failure".into()));
        }
        self.copy_ring(
            slot,
            self.slots.len() + slot,
            self.slots[slot].history.len(),
        )?;
        self.cache[slot]
            .table
            .share_prefix(self.slots[slot].table.blocks(), &mut self.pool);
        self.cache[slot]
            .history
            .clone_from(&self.slots[slot].history);
        self.clock += 1;
        self.cache[slot].touched = self.clock;
        self.cache[slot].images = self.slots[slot]
            .mm
            .as_ref()
            .map_or_else(Vec::new, |m| m.keys.clone());
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::reusable;
    #[test]
    fn prefix_reuse_requires_the_entire_window_not_only_global_pages() {
        let cached: Vec<u32> = (0..4096).collect();
        assert_eq!(reusable(&cached, &cached, 1536, 1024), 4095);
        assert_eq!(reusable(&cached, &cached[..3583], 1536, 1024), 0);
        assert_eq!(reusable(&cached, &cached[..3584], 1536, 1024), 3583);
        assert_eq!(reusable(&[], &cached, 1536, 1024), 0);
        assert_eq!(reusable(&cached, &[], 1536, 1024), 0);
        let mut branch = cached.clone();
        branch[4000] = 9000;
        assert_eq!(reusable(&cached, &branch, 1536, 1024), 4000);
    }
}
