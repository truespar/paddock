use super::*;
use paddock_engine::generator::{GenError, Generator};

#[cfg(test)]
thread_local! {
    pub(super) static POLICY_FOR_TEST: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    // Arithmetic experiment only. Do not expose a runner bypass for the
    // prefix compatibility guard: qualify the graph before changing it.
    pub(super) static CANONICAL_PREFILL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl Generator for FlashNext {
    fn reset(&mut self) {
        self.pending.clear();
        for i in 0..self.slots.len() {
            self.release(i);
        }
        // Never clear poison: reset/cancel is not recovery from GPU failure.
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn max_context(&self) -> usize {
        self.context
    }
    fn enable_batch(&mut self, max: usize) -> std::result::Result<usize, GenError> {
        self.healthy()?;
        Ok(max.min(self.slots.len()))
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.cache_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks())
    }
    fn pool_admission_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks() + self.prefix.reclaimable_blocks(&self.pool))
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.slots
            .get_mut(slot)
            .map_or(0, |s| std::mem::take(&mut s.reused))
    }
    fn forward(&mut self, token: u32) -> std::result::Result<Vec<f32>, GenError> {
        if self.pending.iter().any(|p| p.slot == 0) {
            return Err(GenError::Backend("Flash Next slot is prefilling".into()));
        }
        Ok(self.execute(&[(0, token, self.slots[0].length as u32)], &[0])?)
    }
    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(0, tokens)?)
    }
    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(slot, tokens)?)
    }
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        self.healthy()?;
        if tokens.len() != positions.len() || tokens.len() > self.slots.len() {
            return Err(GenError::Backend("Flash Next batch shape mismatch".into()));
        }
        let rows = tokens
            .iter()
            .zip(positions)
            .enumerate()
            .filter(|(_, (_, p))| **p != 0)
            .map(|(i, (&t, &p))| (i, t, p))
            .collect::<Vec<_>>();
        if rows
            .iter()
            .any(|r| self.pending.iter().any(|p| p.slot == r.0))
        {
            return Err(GenError::Backend(
                "Flash Next decode overlaps pending prefill".into(),
            ));
        }
        let mut out = vec![0.; tokens.len() * VOCAB];
        if !rows.is_empty() {
            let logits = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())?;
            for (i, r) in rows.iter().enumerate() {
                out[r.0 * VOCAB..(r.0 + 1) * VOCAB]
                    .copy_from_slice(&logits[i * VOCAB..(i + 1) * VOCAB]);
            }
        }
        Ok(out)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for i in 0..self.slots.len() {
            if !occupied.get(i).copied().unwrap_or(false)
                && !self.pending.iter().any(|p| p.slot == i)
            {
                self.release(i);
            }
        }
    }
    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    // the scheduler's tick pacer reads the FIFO queue from each offset
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.pending
            .iter()
            .map(|p| (p.slot, p.offset, p.tokens.len() - p.offset))
            .collect()
    }
    // the pass chunk less the decode rows sharing it
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        // The wider physical pass is cold-prefill only. A live decoder must
        // not inherit twice the previous maximum work between tokens.
        if decode_rows == 0 {
            self.capacity
        } else {
            self.chunk
        }
        .saturating_sub(decode_rows)
    }
    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> std::result::Result<(), GenError> {
        let reused = self.prepare(slot, &tokens)?;
        self.pending.push_back(Pending {
            slot,
            tokens,
            offset: reused,
        });
        Ok(())
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.pending.retain(|p| p.slot != slot);
        self.release(slot);
        true
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        self.healthy()?;
        if decodes.len() > self.slots.len()
            || decodes
                .iter()
                .any(|r| self.pending.iter().any(|p| p.slot == r.0))
        {
            return Err(GenError::Backend(
                "invalid Flash Next mixed decode slots".into(),
            ));
        }
        let mut rows = decodes.to_vec();
        let mut complete = Vec::new();
        let mlx = self.is_mlx();
        #[cfg(test)]
        let policy = POLICY_FOR_TEST.with(|v| v.get());
        #[cfg(not(test))]
        let policy = 0;
        let budget = if policy == 2 && !decodes.is_empty() {
            budget.min(crate::schedule::row_cap(decodes.len(), self.chunk) - decodes.len())
        } else {
            budget
        };
        let advances = crate::schedule::grants(
            &self
                .pending
                .iter()
                .map(|p| {
                    // Cap eligible work before apportionment. Otherwise a
                    // logical boundary discards most of a grant while a
                    // neighbouring prompt could have used the spare rows.
                    let mut remaining = if mlx {
                        self.slots[p.slot].plan.at(p.offset).1
                    } else {
                        p.tokens.len() - p.offset
                    };
                    if self.prefix.enabled() {
                        remaining = remaining.min(self.slots[p.slot].plan.until_cut(p.offset));
                    }
                    (remaining, p.tokens.len())
                })
                .collect::<Vec<_>>(),
            budget.min(self.prefill_tick_cap(decodes.len())),
            decodes.is_empty() && policy == 0,
        );
        let mut contracts = if mlx {
            vec![1; decodes.len()]
        } else {
            Vec::new()
        };
        for (p, &n) in self.pending.iter().zip(&advances) {
            if mlx && n > 0 {
                contracts.push(self.slots[p.slot].plan.at(p.offset).0);
            }
            for i in p.offset..p.offset + n {
                rows.push((p.slot, p.tokens[i], i as u32));
            }
            if n > 0 && p.offset + n == p.tokens.len() {
                complete.push((p.slot, rows.len() - 1, p.tokens.len()));
            }
        }
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let outputs = (0..decodes.len())
            .chain(complete.iter().map(|r| r.1))
            .collect::<Vec<_>>();
        let logits = if mlx {
            self.execute_contracts(&rows, &outputs, Some(&contracts))?
        } else {
            self.execute(&rows, &outputs)?
        };
        for (p, n) in self.pending.iter_mut().zip(advances) {
            p.offset += n;
        }
        let captures = self
            .pending
            .iter()
            .filter(|p| {
                self.prefix.enabled()
                    && p.offset > 0
                    && p.offset < p.tokens.len()
                    && self.slots[p.slot].plan.cuts().contains(&p.offset)
            })
            .map(|p| (p.slot, p.tokens.clone()))
            .collect::<Vec<_>>();
        for (slot, tokens) in captures {
            self.capture_prefix(slot, &tokens)?;
        }
        let done = complete
            .iter()
            .enumerate()
            .map(|(i, &(slot, _, n))| {
                let r = decodes.len() + i;
                (slot, logits[r * VOCAB..(r + 1) * VOCAB].to_vec(), n)
            })
            .collect();
        self.pending.retain(|p| p.offset < p.tokens.len());
        Ok((logits[..decodes.len() * VOCAB].to_vec(), done))
    }
}

/// (Arithmetic shape, rows before the next boundary). Prompt-only serial
/// execution and arbitrary bounded mixed grants must use the same graph.
#[cfg(test)]
pub(super) fn logical_chunk(tokens: usize, offset: usize, chunk: usize) -> (usize, usize) {
    assert!(tokens > 0 && offset < tokens);
    let body = tokens - 1;
    if offset == body {
        return (1, 1);
    }
    let start = offset / chunk * chunk;
    let length = (body - start).min(chunk);
    #[cfg(test)]
    let logical = if CANONICAL_PREFILL_FOR_TEST.with(|v| v.get()) {
        chunk
    } else {
        length
    };
    #[cfg(not(test))]
    let logical = length;
    (logical, start + length - offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_prompt_chunks_do_not_depend_on_grants() {
        for chunk in [CHUNK, MLX_CHUNK] {
            for tokens in [1, 8, 9, 13, 26, 128, 129, 130, 813, 2048, 4096] {
                for budget in [1, 3, 8, 13, 32, 63, 125, 128] {
                    let mut offset = 0;
                    while offset < tokens {
                        let (logical, remaining) = logical_chunk(tokens, offset, chunk);
                        let n = remaining.min(budget);
                        assert!(n > 0 && logical <= chunk && n <= logical);
                        assert!(offset + n <= tokens);
                        if offset < tokens - 1 {
                            assert!(offset + n < tokens);
                            assert_eq!(logical, (tokens - 1 - offset / chunk * chunk).min(chunk));
                        } else {
                            assert_eq!((logical, n), (1, 1));
                        }
                        offset += n;
                    }
                }
            }
        }
    }
}
