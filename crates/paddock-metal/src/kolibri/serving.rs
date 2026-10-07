use super::*;
use paddock_engine::generator::{GenError, Generator};
impl Generator for Kolibri {
    fn reset(&mut self) {
        self.pending.clear();
        for s in &mut self.slots {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.reused = 0;
        }
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn max_context(&self) -> usize {
        self.context
    }
    fn enable_batch(&mut self, max: usize) -> std::result::Result<usize, GenError> {
        Ok(max.min(self.slots.len()))
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.kv_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn forward(&mut self, token: u32) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.execute(&[(0, token, self.slots[0].history.len() as u32)], &[0])?)
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
        if tokens.len() != positions.len() || tokens.len() > self.slots.len() {
            return Err(GenError::Backend("Kolibri batch shape mismatch".into()));
        }
        let rows = tokens
            .iter()
            .zip(positions)
            .enumerate()
            .filter(|(_, (_, p))| **p != 0)
            .map(|(i, (&t, &p))| (i, t, p))
            .collect::<Vec<_>>();
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
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        std::mem::take(&mut self.slots[slot].reused)
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks() + self.radix.evictable_blocks(&self.pool))
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for i in 0..self.slots.len() {
            if !occupied.get(i).copied().unwrap_or(false)
                && !self.pending.iter().any(|p| p.slot == i)
            {
                if !self.slots[i].history.is_empty() {
                    self.publish(i);
                }
                self.slots[i].table.clear(&mut self.pool);
                self.slots[i].history.clear();
                self.slots[i].reused = 0;
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
    // the mixed grant: row_cap less the decode rows sharing it
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        crate::schedule::row_cap(decode_rows, CHUNK).saturating_sub(decode_rows)
    }
    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> std::result::Result<(), GenError> {
        if self.pending.iter().any(|p| p.slot == slot) {
            return Err(GenError::Backend("slot already prefilling".into()));
        }
        let reused = self.prepare(slot, &tokens)?;
        self.pending.push_back(Pending {
            slot,
            work: tokens.len() - reused,
            tokens,
            offset: reused,
        });
        Ok(())
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.pending.retain(|p| p.slot != slot);
        if let Some(s) = self.slots.get_mut(slot) {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.reused = 0;
        }
        true
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        if decodes.len() > self.slots.len()
            || decodes
                .iter()
                .any(|r| self.pending.iter().any(|p| p.slot == r.0))
        {
            return Err(GenError::Backend(
                "invalid Kolibri mixed decode slots".into(),
            ));
        }
        let mut rows = decodes.to_vec();
        let mut complete = Vec::new();
        let cap = crate::schedule::row_cap(decodes.len(), CHUNK);
        let advances = crate::schedule::grants(
            &self
                .pending
                .iter()
                .map(|p| (p.tokens.len() - p.offset, p.work))
                .collect::<Vec<_>>(),
            budget.min(cap.saturating_sub(rows.len())),
            decodes.is_empty(),
        );
        for (pending, &n) in self.pending.iter().zip(&advances) {
            for i in pending.offset..pending.offset + n {
                rows.push((pending.slot, pending.tokens[i], i as u32));
            }
            if n > 0 && pending.offset + n == pending.tokens.len() {
                complete.push((pending.slot, rows.len() - 1, pending.tokens.len()));
            }
        }
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let outputs = (0..decodes.len())
            .chain(complete.iter().map(|r| r.1))
            .collect::<Vec<_>>();
        let logits = self.execute(&rows, &outputs)?;
        for (p, n) in self.pending.iter_mut().zip(advances) {
            p.offset += n;
        }
        let done = complete
            .iter()
            .enumerate()
            .map(|(i, &(slot, _, n))| {
                let r = decodes.len() + i;
                (slot, logits[r * VOCAB..(r + 1) * VOCAB].to_vec(), n)
            })
            .collect();
        for &(slot, _, _) in &complete {
            self.publish(slot);
        }
        self.pending.retain(|p| p.offset < p.tokens.len());
        Ok((logits[..decodes.len() * VOCAB].to_vec(), done))
    }
}
