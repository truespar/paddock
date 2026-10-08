//! Vision splice: interleaved text/image prefill (the exclusive multimodal
//! path - the engine drains all slots first, so this runs single-sequence
//! in slot 0 from position 0).
//!
//! Semantics mirror llama.cpp mtmd for gemma4v exactly:
//! - the template's per-image `<|image|>` placeholder becomes
//!   `<|image>` (begin) + the encoder's soft tokens + `<image|>` (end)
//! - image rows enter the residual stream UNSCALED (ggml only multiplies
//!   token lookups by √n_embd - `ubatch.token ? sqrtf(n_embd) : 1.0`)
//! - image rows decode NON-CAUSALLY within their span
//!   (`mtmd_decode_use_non_causal` = true for gemma4v): our attention
//!   kernels take positions only as CAUSAL BOUNDS (rope is a separate
//!   pass), so image rows carry an attention-bound override = the span's
//!   last position while keeping their true positions for rope + KV writes.

use cudarc::driver::CudaSlice;

use crate::gpu::GpuError;
use crate::gpu_model::prefix_cache::{BLOCK_TOKENS, cut_outside_image_spans, image_key_row};
use crate::service::MmChunk;

use super::{Arch, GpuGemma4};

/// This module serves two families and they do not share a vision tower.
/// gemma-4's is a 27-layer SigLIP (RMS norms, GEGLU, NEOX rope, 3×3 avg-pool);
/// muse-glimmer's is a 50-layer Perception Encoder (LayerNorms with bias, an
/// erf-GELU MLP, NORM rope, 32×32 window attention, channel-outer pixel
/// shuffle). They share exactly one thing - the output shape, `[n, llm_embd]`
/// rows to splice - and that is where this enum joins them.
///
/// EXHAUSTIVE on PURPOSE - no `_` arm (see [`Arch`]). A third architecture in
/// this module must not compile until someone has opened its clip graph and
/// decided which tower it gets; falling through to gemma4's would produce
/// plausible features from the wrong encoder.
/// Boxed: the two towers differ ~2x in inline size and this enum lives in the
/// model struct, so the smaller arm would carry the larger one's footprint.
pub(crate) enum VisionTower {
    Gemma4(Box<super::vision::VisionModel>),
    Muse(Box<super::muse_vision::VisionModel>),
}

impl VisionTower {
    /// The projector's output width, which must equal the LLM's residual width.
    pub(crate) fn llm_embd(&self) -> usize {
        match self {
            VisionTower::Gemma4(v) => v.llm_embd(),
            VisionTower::Muse(v) => v.llm_embd(),
        }
    }

    pub(crate) fn budget(&self) -> crate::generator::VisionBudget {
        match self {
            VisionTower::Gemma4(v) => v.budget(),
            VisionTower::Muse(v) => v.budget(),
        }
    }

    /// Device bytes the tower's own weight planes hold.
    pub(crate) fn weight_bytes(&self) -> usize {
        match self {
            VisionTower::Gemma4(v) => v.weight_bytes(),
            VisionTower::Muse(v) => v.weight_bytes(),
        }
    }

    /// Soft tokens a `w`x`h` picture will encode to, without encoding it.
    pub(crate) fn tokens_for(&self, w: usize, h: usize) -> usize {
        match self {
            VisionTower::Gemma4(v) => v.tokens_for(w, h),
            VisionTower::Muse(v) => v.tokens_for(w, h),
        }
    }

    /// Preprocess + encode one RGB8 image -> (device rows, row count).
    /// Preprocessing is part of the tower, not of the caller: the two disagree
    /// on the resize filter (bilinear vs LANCZOS), on whether the image is
    /// letterboxed or stretched, and on the grid the patches come out in.
    fn encode_rgb(&self, rgb: &[u8], w: usize, h: usize) -> Result<EncodedImage, GpuError> {
        match self {
            VisionTower::Gemma4(v) => {
                let (resized, tw, th) = v.resize_rgb(rgb, w, h);
                let o = v.encode_resized(super::vision::Resized::Host(&resized), tw, th)?;
                Ok(EncodedImage {
                    embd: o.embd,
                    n_tokens: o.n_tokens,
                })
            }
            VisionTower::Muse(v) => {
                let (patches, gw, gh) = v.preprocess_rgb(rgb, w, h);
                let o = v.encode(&patches, gw, gh)?;
                Ok(EncodedImage {
                    embd: o.embd,
                    n_tokens: o.n_tokens,
                })
            }
        }
    }
}

/// One tower's output rows, whichever tower produced them - the only shape the
/// splice below cares about.
pub(crate) struct EncodedImage {
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

/// One prefill row of the interleaved stream.
enum Row {
    Token(u32),
    /// (image index, row within that image's encoded embeddings)
    Image(usize, usize),
}

/// FNV-1a over the raw image bytes (dims folded in by the caller).
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl GpuGemma4 {
    pub fn attach_vision(
        &mut self,
        map: &paddock_models::mapped::MappedGguf,
    ) -> Result<(), GpuError> {
        self.attach_vision_with(map, None)
    }

    /// Soft-token rows of the largest picture the attached tower emits.
    pub(crate) fn max_picture_rows(&self) -> usize {
        self.vision
            .as_ref()
            .map_or(0, |v| v.budget().max_tokens as usize)
    }

    /// How far one picture longer than an SWA sub-span `span` runs past it:
    /// the ring's picture allowance. The sub-span cutter never splits a
    /// picture, so anything longer than a span is a sub-span on its own -
    /// 0 for gemma4's 280-token pictures, a whole extra span for muse's 4096.
    pub(crate) fn picture_overshoot(&self, span: usize) -> usize {
        self.max_picture_rows().saturating_sub(span)
    }

    /// The narrowest prefill pass: one serving tick, and never less than one
    /// whole picture (its rows prefill together).
    pub(crate) fn pass_floor(&self) -> usize {
        super::forward::pf_rows_floor().max(self.max_picture_rows().next_multiple_of(128))
    }

    /// `attach_vision` plus the endpoint's resolved options. `max_image_tokens`
    /// is the per-image soft-token ceiling from `servers/<port>.toml`; None
    /// keeps the checkpoint's published budget. Only the gemma4 tower reads
    /// it - muse-glimmer sizes from its own grid - and the runner says so
    /// rather than letting the field look effective when it is not.
    pub fn attach_vision_with(
        &mut self,
        map: &paddock_models::mapped::MappedGguf,
        max_image_tokens: Option<usize>,
    ) -> Result<(), GpuError> {
        // one picture prefills in one pass, and no pass is wider than PF_ROWS
        let max_image_tokens = max_image_tokens.map(|t| t.min(super::forward::PF_ROWS));
        // The tower is elected by the TEXT model's arch, not by the mmproj's
        // projector string: they must agree, and the text side is what already
        // decided every other constant.
        let vm = match self.hp.arch {
            // DiffusionGemma's config carries Gemma 4's vision config verbatim
            // (27 layers, 1152 wide, 280 tokens); its processor is
            // Gemma4Processor. No mmproj ships for it yet, but the tower
            // class is the same one.
            Arch::Gemma4 | Arch::DiffusionGemma => VisionTower::Gemma4(Box::new(
                super::vision::VisionModel::load(self.exec.clone(), map, max_image_tokens)?,
            )),
            Arch::MuseGlimmer => VisionTower::Muse(Box::new(
                super::muse_vision::VisionModel::load(self.exec.clone(), map)?,
            )),
        };
        if vm.llm_embd() != self.hp.n_embd {
            return Err(GpuError::Driver(format!(
                "mmproj projects to {} but the model embd is {}",
                vm.llm_embd(),
                self.hp.n_embd
            )));
        }
        if self.img_beg_id.is_none() || self.img_end_id.is_none() {
            let (b, e) = super::image_markers(self.hp.arch);
            return Err(GpuError::Driver(format!(
                "vocab lacks the {b}/{e} markers - not a {} vision model",
                self.hp.arch.key()
            )));
        }
        // The tower is WEIGHTS, and it loads after the loader snapshotted the
        // weights line - so without this its ~1 GiB was reported as
        // `scratch_mem` (model_mem - weights - kv, a derived remainder), which
        // is how a gemma4 memory ledger came to show "7.33 GiB of scratch" and
        // read as impossible. Same correction the DFlash drafter makes.
        self.weights_bytes = Some(self.weights_bytes.unwrap_or(0) + vm.weight_bytes() as u64);
        self.vision = Some(vm);
        Ok(())
    }

    /// The prefix radix's identity of one picture (its image-row keys).
    /// The picture store keys on a 256-bit digest of the same inputs
    /// (`picture_store::picture_key`), so the two can only disagree on a
    /// 64-bit collision here.
    fn picture_hash(rgb: &[u8], w: usize, h: usize) -> u64 {
        hash_bytes(rgb)
            ^ (w as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ (h as u64).rotate_left(32)
    }

    /// Borrow the encoded picture, from the picture store when its bytes were
    /// encoded before and through the tower otherwise. The store is the only
    /// owner - a hit used to allocate and copy the rows, and a miss kept a
    /// second copy for the cache, whose 16 entries were bounded by count (a
    /// muse picture is 109 MB) and kept each picture's raw RGB for an exact
    /// compare.
    fn encode_picture(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<std::sync::Arc<EncodedImage>, GpuError> {
        let key = crate::gpu_model::picture_store::picture_key(rgb, w, h);
        if let Some(p) = self.pictures.get(&key) {
            return Ok(p);
        }
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| GpuError::Driver("no mmproj attached".into()))?;
        let out = vision.encode_rgb(rgb, w, h)?;
        let bytes = (out.embd.len() * std::mem::size_of::<f32>()) as u64;
        Ok(self.pictures.insert(key, out, bytes))
    }

    /// The plan's charges for image input, which allocates on demand and so
    /// was never in it: one tower pass at the largest picture the tower emits,
    /// MEASURED by a profile run at load (a blank max-size picture through
    /// the tower, its pool high-water read - ~0.2 GB on gemma4's 280-token
    /// grid, ~2 GB on muse's 4096), and the picture store's byte budget at the
    /// widest pass the ladder can elect.
    pub(crate) fn vision_reserves(&self) -> Result<Vec<crate::kv_plan::Reserve>, GpuError> {
        let Some(v) = self.vision.as_ref() else {
            return Ok(Vec::new());
        };
        let side = ((v.budget().max_pixels as f64).sqrt() as usize).max(1);
        let blank = vec![0u8; side * side * 3];
        let (_, tower) = self
            .exec
            .pool_peak_during(|| v.encode_rgb(&blank, side, side))?;
        let widest = super::forward::pf_rows(self.max_ctx).max(self.pass_floor());
        Ok(vec![
            crate::kv_plan::Reserve::new("vision tower pass", tower),
            crate::kv_plan::Reserve::new("picture store", self.picture_budget_bytes(widest)),
        ])
    }

    /// What the picture store may hold: two prefill passes of picture rows -
    /// the one in flight and one for reuse (a repeated conversation prefix is
    /// the radix KV cache's job, not this one's).
    pub(crate) fn picture_budget_bytes(&self, pass_rows: usize) -> u64 {
        (2 * pass_rows * self.hp.n_embd * std::mem::size_of::<f32>()) as u64
    }

    /// Exclusive multimodal prefill: encode every image, splice
    /// begin + soft tokens + end around each, prefill the whole stream from
    /// position 0 (slot 0), return the last row's logits and the ROW COUNT.
    ///
    /// The row count goes back to the caller rather than staying an internal
    /// cursor because the serial engine reports usage from it - text tokens
    /// alone under-count an image prompt by an order of magnitude.
    pub(crate) fn multimodal_prefill(
        &mut self,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuError> {
        let (logits, rows) = self.multimodal_prefill_slot(0, chunks)?;
        self.pos = rows;
        Ok((logits, rows))
    }

    /// The mm prefill against batch slot `slot` (S8 shape, qwen35 parity).
    /// Returns the last row's logits and the total row count - image rows
    /// included, so it differs from the prompt's token count and the service
    /// must use it as the slot's KV position.
    ///
    /// PREFIX CACHING APPLIES here, keyed on content rather than on
    /// row tokens: every image row carries the same placeholder, so a radix
    /// keyed on the row stream would serve one picture's KV for another. Text
    /// rows key as themselves and image rows key off the picture's content hash
    /// - see [`crate::gpu_model::prefix_cache::image_key_row`]. That is what
    ///   makes the document workload work: same page, many questions, and every
    ///   turn after the first resumes past the whole picture rather than
    ///   re-prefilling its soft tokens.
    ///
    /// gemma4v is the awkward one, and the reason granite went first. Its image
    /// rows decode NON-CAUSALLY: each attends to its span's last position, so a
    /// resume landing strictly inside a picture would re-prefill rows whose
    /// attention bound points at keys the adopted blocks already hold under a
    /// different write order. Rather than reason about whether that is benign,
    /// no CHECKPOINT is ever attached inside an image span - and since a resume
    /// position is exactly a checkpoint position, mid-span resumes cannot occur
    /// at all. See [`Self::mm_prefix_cut`].
    pub(crate) fn multimodal_prefill_slot(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuError> {
        if self.vision.is_none() {
            return Err(GpuError::Driver("no mmproj attached".into()));
        }
        assert!(
            slot < self.n_slots.max(1),
            "slot {slot} >= enabled {}",
            self.n_slots
        );
        let (beg, end) = (
            self.img_beg_id.expect("markers checked at vision attach"),
            self.img_end_id.expect("markers checked at vision attach"),
        );

        // pass 1: each picture's radix identity and soft-token count - from
        // the resize alone, so the whole layout is known before any picture
        // is encoded (each is encoded when the pass that splices it comes up)
        let mut sources: Vec<(&[u8], usize, usize)> = Vec::new();
        let mut n_tok: Vec<usize> = Vec::new();
        let mut hashes: Vec<u64> = Vec::new();
        {
            let vision = self
                .vision
                .as_ref()
                .ok_or_else(|| GpuError::Driver("no mmproj attached".into()))?;
            for ch in chunks {
                if let MmChunk::Image { rgb, w, h } = ch {
                    sources.push((rgb, *w, *h));
                    n_tok.push(vision.tokens_for(*w, *h));
                    hashes.push(Self::picture_hash(rgb, *w, *h));
                }
            }
        }
        // pass 2: the interleaved row stream, and the radix key vector beside
        // it - one key per row, image rows keyed on content (see the doc above)
        let mut rows: Vec<Row> = Vec::new();
        let mut keys: Vec<u32> = Vec::new();
        // soft-token row range of each picture, for the mid-span guard
        let mut img_spans: Vec<(usize, usize)> = Vec::new();
        let mut img_k = 0usize;
        for ch in chunks {
            match ch {
                MmChunk::Text(ids) => {
                    rows.extend(ids.iter().map(|&t| Row::Token(t)));
                    keys.extend_from_slice(ids);
                }
                MmChunk::Image { .. } => {
                    rows.push(Row::Token(beg));
                    keys.push(beg);
                    let first = rows.len();
                    let n = n_tok[img_k];
                    rows.extend((0..n).map(|r| Row::Image(img_k, r)));
                    keys.extend((0..n).map(|r| image_key_row(hashes[img_k], r)));
                    img_spans.push((first, first + n));
                    rows.push(Row::Token(end));
                    keys.push(end);
                    img_k += 1;
                }
                MmChunk::Audio { .. } => {
                    return Err(GpuError::Driver(
                        "gemma4 serves images, not audio - routing bug".into(),
                    ));
                }
                MmChunk::OcrCrop(_) => {
                    return Err(GpuError::Driver(
                        "OCR crop directive on gemma4 - routing bug".into(),
                    ));
                }
                MmChunk::VisionPixels { .. } => {
                    return Err(GpuError::Driver(
                        "pixel-budget directive on gemma4 - routing bug".into(),
                    ));
                }
            }
        }
        debug_assert_eq!(rows.len(), keys.len(), "one radix key per prefill row");
        if rows.is_empty() {
            return Err(GpuError::Driver("empty multimodal prompt".into()));
        }
        if rows.len() > self.max_ctx {
            return Err(GpuError::Driver(format!(
                "multimodal prompt is {} rows but max_ctx is {}",
                rows.len(),
                self.max_ctx
            )));
        }

        // image spans (for the non-causal attention-bound override):
        // span_end[i] = the absolute position of image i's last soft token
        let mut span_end = vec![0usize; n_tok.len()];
        for (pos, row) in rows.iter().enumerate() {
            if let Row::Image(i, _) = row {
                span_end[*i] = pos;
            }
        }

        // Same admission shape as the text path (batch.rs): clear, try to
        // resume, then grow the table to cover the whole prompt. `start` is a
        // block-aligned row count already resident in KV - 0 on a cold prompt.
        self.gpool_clear_slot(slot);
        let start = self.prefix_resume(slot, &keys)?;
        // `keys` are cache keys (image spans hash into them), not token ids,
        // so the DFlash coverage trim `prefix_resume` just ran compared two
        // different id spaces and its agreement means nothing here. Multimodal
        // resumes stay cold until the walk refills the ring. Rebuilding this
        // properly wants the mm path to record its own key mirror - the same
        // seam, just keyed the way the resume asks about it.
        self.dflash_clear_slot(slot);
        self.ensure_global_rows(&[slot as u32], &[(rows.len() - 1) as u32])?;
        let cut = self.mm_prefix_cut(rows.len(), start, &img_spans);
        let slot_fill = vec![slot as u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&slot_fill, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;

        let n_embd = self.hp.n_embd;
        // resume: rows [0, start) are already in KV, so the tail starts there
        // and `base` stays ABSOLUTE - positions, rope and the non-causal
        // attention bounds are all indexed off the full prompt, not the tail.
        // Passes end at most `pf_rows` apart and NEVER inside a picture: its
        // rows attend to its last row, so a pass that ended mid-picture had
        // its first rows reading KV rows the next pass had not written yet -
        // whatever the slot held before (found with two different prior
        // prompts: the straddling prefill's logits moved by up to 3.2). The
        // pass floor keeps one whole picture inside a pass.
        let passes = crate::gpu_model::prefix_cache::mm_pass_ends(
            start,
            rows.len(),
            &[],
            &img_spans,
            self.pf_rows,
        );
        let mut base = start;
        let mut last_len = 0usize;
        for (end, _) in passes {
            // the pictures this pass splices (a pass never cuts one), encoded
            // or borrowed from the store now and released when the pass ends -
            // a prompt's pictures are never all held at once
            let mut pictures: Vec<Option<std::sync::Arc<EncodedImage>>> = vec![None; n_tok.len()];
            for (k, &(s0, e0)) in img_spans.iter().enumerate() {
                if s0 >= base && e0 <= end {
                    let (rgb, w, h) = sources[k];
                    let p = self.encode_picture(rgb, w, h)?;
                    if p.n_tokens != n_tok[k] {
                        return Err(GpuError::Driver(format!(
                            "a {w}x{h} picture encoded to {} soft tokens where its resize \
                             planned {} - the layout would splice it wrong",
                            p.n_tokens, n_tok[k]
                        )));
                    }
                    pictures[k] = Some(p);
                }
            }
            let chunk = &rows[base..end];
            let r = chunk.len();
            let positions: Vec<u32> = (0..r).map(|i| (base + i) as u32).collect();
            // attention bounds: image rows see through their whole span
            let attn_pos: Vec<u32> = chunk
                .iter()
                .enumerate()
                .map(|(i, row)| match row {
                    Row::Token(_) => (base + i) as u32,
                    Row::Image(img, _) => span_end[*img].max(base + i) as u32,
                })
                .collect();
            {
                let sc = &mut self.scratch;
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_pos)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                self.exec
                    .stream
                    .memcpy_htod(&attn_pos, &mut sc.pf_attn_pos)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                // token rows stage in pf_tmp (zeroed -> image slots stay 0),
                // one √embd scale covers them, then image rows overwrite
                self.exec
                    .stream
                    .memset_zeros(&mut sc.pf_tmp)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            for (i, row) in chunk.iter().enumerate() {
                if let Row::Token(t) = row {
                    let sc = &mut self.scratch;
                    super::EmbdTable::of(&self.token_embd, &self.head).row(
                        &self.exec,
                        *t,
                        &mut sc.embd_id,
                        &mut sc.pf_row,
                        n_embd,
                    )?;
                    self.exec
                        .copy_region(&sc.pf_row, 0, &mut sc.pf_tmp, i * n_embd, n_embd)?;
                }
            }
            {
                let sc = &mut self.scratch;
                self.exec
                    .stream
                    .memset_zeros(&mut sc.pf_x)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                self.exec
                    .scale_add(&mut sc.pf_x, &sc.pf_tmp, self.hp.embd_scale(), r * n_embd)?;
            }
            for (i, row) in chunk.iter().enumerate() {
                if let Row::Image(img, ir) = row {
                    let src = &pictures[*img]
                        .as_ref()
                        .expect("a pass's pictures are encoded before it runs")
                        .embd;
                    let sc = &mut self.scratch;
                    self.exec
                        .copy_region(src, ir * n_embd, &mut sc.pf_x, i * n_embd, n_embd)?;
                }
            }
            // After the image splice, deliberately: the reference norms
            // `inpL` once, and mtmd has already substituted the projected
            // image rows into it by then - so image rows get normalized too.
            // (The sqrt scale above is text-only for the opposite reason: the
            // projector's output is not a raw embedding lookup.) No-op on
            // gemma4.
            {
                let sc = &mut self.scratch;
                super::GpuGemma4::embd_preamble(
                    &self.exec,
                    &self.hp,
                    self.embd_ones.as_ref(),
                    &mut sc.pf_x,
                    r,
                )?;
            }
            // image-aware SWA sub-spans: cut every `swa_span` rows, never inside
            // a picture - an end that lands in one walks back to its start, so
            // a sub-span is at most `swa_span` rows or one whole longer picture,
            // which is exactly what the ring was sized for (`swa_overshoot`).
            // Extending PAST the picture instead (the old cutter) grew a
            // sub-span by up to a whole picture, and anything over a 288-row
            // allowance - muse's 4096-token pictures, a raised gemma4 cap -
            // wrapped the ring onto keys the window still needed.
            let in_chunk: Vec<(usize, usize)> = img_spans
                .iter()
                .filter(|&&(s0, e0)| s0 >= base && e0 <= end)
                .map(|&(s0, e0)| (s0 - base, e0 - base))
                .collect();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            let mut o = 0usize;
            for (e, _) in
                crate::gpu_model::prefix_cache::mm_pass_ends(0, r, &[], &in_chunk, self.swa_span)
            {
                if self.paging.is_some() && e - o > self.swa_span + self.swa_overshoot {
                    return Err(GpuError::Driver(format!(
                        "a {}-row picture outgrows the SWA ring's {} + {} rows (the tower \
                         was attached after the ring was sized) - refusing rather than \
                         wrap the ring onto keys the window still reads",
                        e - o,
                        self.swa_span,
                        self.swa_overshoot
                    )));
                }
                spans.push((o, e - o));
                o = e;
            }
            self.prefill_layers(r, &[(0, r)], &spans, 0)?;
            base = end;
            last_len = r;
        }
        let logits = self.logits_from_pf_row(last_len - 1)?;
        self.prefix_insert(slot, &keys, cut)?;
        Ok((logits, rows.len()))
    }

    /// The checkpoint cut for a multimodal prompt: [`Self::prefix_cut`]'s
    /// answer, walked back to a page boundary that is not strictly inside an
    /// image span.
    ///
    /// [`cut_outside_image_spans`] is what keeps gemma4v's non-causal image
    /// rows safe, and qwen35 shares it - the rule and its tests live in
    /// `gpu_model/prefix_cache.rs` so the two families cannot drift on it.
    fn mm_prefix_cut(
        &self,
        n_rows: usize,
        start: usize,
        img_spans: &[(usize, usize)],
    ) -> Option<usize> {
        let cut = cut_outside_image_spans(self.prefix_cut(n_rows, start)?, img_spans);
        (cut > start && cut >= BLOCK_TOKENS).then_some(cut)
    }
}
