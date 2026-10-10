//! SAM 3's CLIP concept tower. F32 embeddings/residuals, F16 contractions,
//! causal attention and erf-GELU, then the detector's 1024 -> 256 resizer.
//! The pooled CLIP projection is unused by SAM 3 and must not be loaded.
//! This is a component, not a mask-serving model or a parity claim.
use super::*;
use paddock_models::sam3::Sam3TextConfig;

/// An owned, bounded text-tower workspace. One prompt occupies exactly one
/// 32-row GEMM tile: batching changes grid size, not its reduction contract.
pub(super) struct TextWorkspace {
    pub(super) cap: usize,
    ids: Buffer,
    x: Buffer,
    norm: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    projection: Buffer,
    wide: Buffer,
    features: Buffer,
    bad: Buffer,
}

impl TextWorkspace {
    pub(super) fn required_bytes(cap: usize) -> Result<u64> {
        // Bound every row count and GPU index before allocating. No implicit
        // max(1), truncation or wraparound for malformed capacity requests.
        if !(1..=32).contains(&cap) {
            return Err(error(
                "SAM 3 text capacity must be between 1 and 32 prompts",
            ));
        }
        Ok((cap * 32 * (4 + 4 * 1024 + 2 * 9 * 1024 + 2 * 4096 + 4 * 256) + 4) as u64)
    }
    pub(super) fn new(d: &MetalDevice, cap: usize) -> Result<Self> {
        Self::required_bytes(cap)?;
        let rows = cap * 32;
        let half = |cols| d.alloc(rows * cols * 2);
        Ok(Self {
            cap,
            ids: d.alloc(rows * 4)?,
            x: d.alloc(rows * 1024 * 4)?,
            norm: half(1024)?,
            qkv: half(3072)?,
            q: half(1024)?,
            k: half(1024)?,
            v: half(1024)?,
            attention: half(1024)?,
            projection: half(1024)?,
            wide: half(4096)?,
            features: d.alloc(rows * 256 * 4)?,
            bad: d.alloc(4)?,
        })
    }
}

/// The official SAM 3 text tower and detector resizer, not the unused CLIP
/// pooled projection. Tokens use paddock_tokenizer::sam3's 32-id layout.
pub struct Sam3Text {
    pub(super) device: MetalDevice,
    pub(super) cfg: Sam3TextConfig,
    pub(super) token: Buffer,
    pub(super) position: Buffer,
    pub(super) blocks: Vec<Block>,
    pub(super) final_norm: Norm,
    pub(super) resizer: Conv,
    pub(super) ws: TextWorkspace,
    pub(super) weight_bytes: u64,
    pub(super) encoded_prompts: usize,
}

fn norm(c: &Commands<'_>, w: &TextWorkspace, bias: &Buffer, n: &Norm, rows: usize, add: bool) {
    c.dispatch(
        "sam3_norm",
        &[&w.x, &w.projection, bias, &n.w, &n.b, &w.norm],
        &[1024, u32::from(add)],
        [rows, 1, 1],
        256,
    );
}

impl Sam3Text {
    pub fn config(&self) -> &Sam3TextConfig {
        &self.cfg
    }
    pub fn max_batch(&self) -> usize {
        self.ws.cap
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        TextWorkspace::required_bytes(self.ws.cap).expect("validated text capacity")
    }
    pub fn resident_bytes_required(max_prompts: usize) -> Result<u64> {
        Ok(Self::planned_weight_bytes() + TextWorkspace::required_bytes(max_prompts)?)
    }

    /// Encode whole padded prompts. This does not truncate or tokenize; the
    /// shared SAM 3 tokenizer supplies ids and the valid lengths the detector
    /// will need. Attention computes padded rows, but causal valid rows never
    /// see future padding. No request-size GPU allocations or readback planes.
    pub fn encode(&mut self, ids: &[u32], prompts: usize) -> Result<()> {
        // Invalidate first, including bad arguments and a GPU failure. A
        // failed request must not expose features from an earlier concept.
        self.encoded_prompts = 0;
        if prompts == 0 || prompts > self.ws.cap {
            return Err(error(format!(
                "SAM 3 text batch {prompts} outside 1..={}",
                self.ws.cap
            )));
        }
        let rows = prompts * 32;
        if ids.len() != rows {
            return Err(error(format!(
                "SAM 3 text needs {rows} token ids, got {}",
                ids.len()
            )));
        }
        if let Some(id) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab) {
            return Err(error(format!(
                "SAM 3 text token {id} outside vocabulary {}",
                self.cfg.vocab
            )));
        }
        let w = &self.ws;
        // The model exclusively owns the queue; every previous encode fences
        // before returning, so both shared-buffer host writes are safe.
        unsafe {
            w.ids.write_u32(ids);
            w.bad.write_u32(&[0]);
        }
        let c = self.device.begin()?;
        point(
            &c,
            "sam3_text_embed",
            &[&self.token, &self.position, &w.ids, &w.x],
            &[rows as u32],
            rows * 1024,
        );
        norm(&c, w, &self.blocks[0].n1.b, &self.blocks[0].n1, rows, false);
        for (li, b) in self.blocks.iter().enumerate() {
            let mm = |m: &Conv, x: &Buffer, out: &Buffer, mode| {
                vision::mm(&c, &m.w, &m.b, x, out, rows, mode);
            };
            mm(&b.qkv, &w.norm, &w.qkv, 1);
            point(
                &c,
                "sam3_text_qkv",
                &[&w.qkv, &b.qkv.b, &w.q, &w.k, &w.v],
                &[rows as u32],
                rows * 1024,
            );
            c.dispatch(
                "sam3_text_attention",
                &[&w.q, &w.k, &w.v, &w.attention],
                &[prompts as u32],
                [prompts * 16, 1, 1],
                256,
            );
            mm(&b.out, &w.attention, &w.projection, 1);
            norm(&c, w, &b.out.b, &b.n2, rows, true);
            mm(&b.up, &w.norm, &w.wide, 5);
            mm(&b.down, &w.wide, &w.projection, 1);
            let next = self.blocks.get(li + 1).map_or(&self.final_norm, |b| &b.n1);
            norm(&c, w, &b.down.b, next, rows, true);
        }
        vision::mm(
            &c,
            &self.resizer.w,
            &self.resizer.b,
            &w.norm,
            &w.features,
            rows,
            4,
        );
        // Refuse nonfinite results rather than cache/serve a poisoned prompt.
        // Only one four-byte status word crosses back, not the feature plane.
        point(
            &c,
            "vis_finite",
            &[&w.features, &w.bad],
            &[(rows * 256) as u32],
            rows * 256,
        );
        c.finish()?;
        if unsafe { w.bad.read_u32(1)[0] } != 0 {
            return Err(error("SAM 3 text produced nonfinite features"));
        }
        self.encoded_prompts = prompts;
        Ok(())
    }

    /// Fenced diagnostic readback of exactly the last successful batch.
    /// No stale tail from a larger prior batch can be observed.
    pub fn read_features(&self) -> Result<Vec<f32>> {
        if self.encoded_prompts == 0 {
            return Err(error("SAM 3 text has no successfully encoded prompts"));
        }
        Ok(unsafe {
            self.ws
                .features
                .read_f32(0, self.encoded_prompts * 32 * 256)
        })
    }
}
