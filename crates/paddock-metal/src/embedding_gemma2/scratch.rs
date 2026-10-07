//! Grow on admitted work, reuse within a burst, release after actual idle.
//! The pending command owns its allocation generation; enqueue-ahead growth
//! and cancelled requests cannot free buffers still referenced by the GPU.
use super::*;
use std::rc::Rc;

impl Scratch {
    fn allocate(device: &MetalDevice, rows: usize) -> Result<Self> {
        let a = |n| device.alloc(rows * n * 4);
        Ok(Self {
            rows,
            x: a(WIDTH)?,
            norm: a(WIDTH)?,
            delta: a(WIDTH)?,
            q: a(2048)?,
            k: a(512)?,
            v: a(512)?,
            attn: a(2048)?,
            gate: a(FF)?,
            up: a(FF)?,
            ple: a(WIDTH * LAYERS)?,
            token_out: a(DIM)?,
            scores: device.alloc(128 * rows * 4 * 2)?,
        })
    }
}

impl EmbeddingGemma2 {
    pub(super) fn workspace(&mut self, rows: usize) -> Result<Rc<Scratch>> {
        if self.scratch.as_ref().is_none_or(|s| s.rows < rows) {
            // Clear our cache reference first: if no command retains the old
            // generation, its budget is available for the larger allocation.
            // An allocation failure leaves the model usable for smaller jobs.
            self.scratch = None;
            let rows = rows.div_ceil(64) * 64;
            self.scratch = Some(Rc::new(Scratch::allocate(&self.device, rows)?));
        }
        Ok(self.scratch.as_ref().expect("allocated workspace").clone())
    }
}
