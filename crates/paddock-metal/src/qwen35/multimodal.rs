//! Image admission, exact-content encoder cache, interleaved M-RoPE and
//! embedding injection. KV row indices and rotary positions are different
//! after an image; keeping both explicit also makes speculative rollback safe.
use super::*;
use paddock_engine::{generator::MmAdmit, service::MmChunk};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, PartialEq, Eq)]
pub(super) struct ImageKey {
    digest: [u8; 32],
    hash: u64,
    rgb: Arc<[u8]>,
    w: usize,
    h: usize,
    offset: usize,
    nx: usize,
    ny: usize,
}
impl ImageKey {
    pub(super) fn end(&self) -> usize {
        self.offset + self.nx * self.ny
    }
    pub(super) fn starts_in(&self, start: usize, end: usize) -> bool {
        self.offset >= start && self.offset < end
    }
    pub(super) fn inside(&self, cut: usize) -> bool {
        cut > self.offset && cut < self.end()
    }
    pub(super) fn chain(
        &self,
        key: paddock_engine::kv_tier::digest::LogicalKey,
    ) -> paddock_engine::kv_tier::digest::LogicalKey {
        let mut identity = Vec::with_capacity(72);
        identity.extend_from_slice(&self.digest);
        for n in [self.w, self.h, self.offset, self.nx, self.ny] {
            identity.extend_from_slice(&(n as u64).to_le_bytes());
        }
        key.child_bytes("qwen-image-v1", &identity)
    }
    fn same_image(&self, other: &Self) -> bool {
        self.hash == other.hash && self.w == other.w && self.h == other.h && self.rgb == other.rgb
    }
}

pub(super) struct CachedImage {
    key: ImageKey,
    output: vision::Output,
    touched: u64,
}

pub(super) struct Layout {
    pub(super) ids: Vec<u32>,
    pub(super) keys: Vec<ImageKey>,
    positions: Vec<[u32; 4]>,
    limits: Vec<u32>,
    pub(super) images: Vec<vision::Output>,
    final_position: u32,
}
impl Layout {
    pub(super) fn inside_image(&self, cut: usize) -> bool {
        self.keys
            .iter()
            .any(|i| cut > i.offset && cut < i.offset + i.nx * i.ny)
    }
    pub(super) fn limit(&self, row: usize) -> u32 {
        self.limits.get(row).copied().unwrap_or(row as u32)
    }
}

pub(super) struct Encoding {
    requests: Vec<(usize, Layout)>,
    unique: Vec<ImageKey>,
    outputs: Vec<Option<vision::Output>>,
    missing: Vec<usize>,
    in_flight: Vec<usize>,
    job: Option<vision::Job>,
}

fn err(s: impl Into<String>) -> MetalError {
    MetalError::Model(s.into())
}

pub(super) fn prefix_images_match(old: &[ImageKey], new: &[ImageKey], cut: usize) -> bool {
    !old.iter().chain(new).any(|i| i.inside(cut))
        && old
            .iter()
            .filter(|i| i.end() <= cut)
            .eq(new.iter().filter(|i| i.end() <= cut))
}

impl Qwen35 {
    pub(super) fn cold_image_cohort_encoding(&self) -> bool {
        if self.pending.is_empty() || self.encoding.is_empty() {
            return false;
        }
        // A resumed prefix or a short text request must never wait behind
        // large images. Restrict this ordering to comparable cold image work
        // which can benefit from the existing proportional row planner.
        if self
            .pending
            .iter()
            .any(|p| self.slots[p.slot].mm.is_none() || p.work != p.tokens.len())
        {
            return false;
        }
        let sizes = self
            .pending
            .iter()
            .map(|p| p.work)
            .chain(
                self.encoding
                    .iter()
                    .flat_map(|e| e.requests.iter().map(|(_, m)| m.ids.len())),
            )
            .collect::<Vec<_>>();
        comparable_images(&sizes)
    }

    pub fn attach_vision(&mut self, path: &Path) -> Result<()> {
        if self.ternary.is_some() && path.is_dir() {
            return Err(err(
                "Bonsai PTQ1 requires its BF16 GGUF vision companion, not an MLX tower",
            ));
        }
        if self.mlx
            && !(self.splash && path.join("manifest.json").is_file())
            && !(self.bonsai.is_some() && path.join("hadamard.json").is_file())
            && !(self.head.is_none()
                && matches!(self.geometry, Geometry::DENSE_08B | Geometry::DENSE_4B)
                && path.is_dir())
        {
            return Err(MetalError::Model("native MLX vision tower ingestion is not yet qualified; a GGUF companion would change the checkpoint".into()));
        }
        self.require_committed()?;
        if self.vision.is_some()
            || self.cold.is_some()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || !self.encoding.is_empty()
        {
            return Err(err("attach vision once, before prefill"));
        }
        self.source_versions.extend(crate::offload::versions(path)?);
        let before = self.device.allocated_bytes();
        let v = vision::Vision::load(&self.device, path, self.width)?;
        self.weight_bytes += self.device.allocated_bytes() - before;
        self.vision = Some(v);
        Ok(())
    }

    pub fn image_cache_reuses(&self) -> u64 {
        self.image_cache_reused
    }

    pub(super) fn layout(&self, chunks: Vec<MmChunk>) -> Result<Layout> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| err("configure --mmproj to enable Qwen images"))?;
        let mut lay = Layout {
            ids: Vec::new(),
            keys: Vec::new(),
            positions: Vec::new(),
            limits: Vec::new(),
            images: Vec::new(),
            final_position: 0,
        };
        for chunk in chunks {
            match chunk {
                MmChunk::Text(ids) => {
                    if ids.iter().any(|&id| id as usize >= self.vocab)
                        || lay.ids.len().saturating_add(ids.len()) > self.context
                    {
                        return Err(err("invalid or over-context multimodal text"));
                    }
                    for id in ids {
                        lay.ids.push(id);
                        lay.positions.push([lay.final_position; 4]);
                        lay.limits.push(lay.ids.len() as u32 - 1);
                        lay.final_position += 1;
                    }
                }
                MmChunk::Image { rgb, w, h } => {
                    let (tw, th) = vision.dimensions(w, h)?;
                    if rgb.len() != w * h * 3 {
                        return Err(err("RGB byte count mismatch"));
                    }
                    let (nx, ny) = (tw / 32, th / 32);
                    let offset = lay.ids.len();
                    let n = nx * ny;
                    if offset.saturating_add(n) > self.context {
                        return Err(err(format!(
                            "multimodal prompt exceeds context {} (image has {n} rows)",
                            self.context
                        )));
                    }
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    rgb.hash(&mut hasher);
                    w.hash(&mut hasher);
                    h.hash(&mut hasher);
                    lay.keys.push(ImageKey {
                        digest: *blake3::hash(&rgb).as_bytes(),
                        hash: hasher.finish(),
                        rgb: rgb.into(),
                        w,
                        h,
                        offset,
                        nx,
                        ny,
                    });
                    for j in 0..n {
                        lay.ids.push(0);
                        lay.positions.push([
                            lay.final_position,
                            lay.final_position + (j / nx) as u32,
                            lay.final_position + (j % nx) as u32,
                            0,
                        ]);
                        // M-RoPE shares the temporal coordinate, not an
                        // attention domain: image rows remain raster-causal
                        // in the language backbone. Only the ViT is noncausal.
                        lay.limits.push((offset + j) as u32);
                    }
                    lay.final_position += nx.max(ny) as u32;
                }
                _ => {
                    return Err(err(
                        "Metal Qwen supports image/text chunks, not audio or OCR directives",
                    ));
                }
            }
        }
        if lay.keys.is_empty() {
            return Err(err("multimodal prompt has no image"));
        }
        Ok(lay)
    }

    pub(super) fn rope_position(&self, slot: usize, pos: usize) -> [u32; 4] {
        self.slots[slot].mm.as_ref().map_or([pos as u32; 4], |m| {
            m.positions
                .get(pos)
                .copied()
                .unwrap_or([m.final_position + pos.saturating_sub(m.ids.len()) as u32; 4])
        })
    }

    pub(super) fn inject_images(&self, cmd: &Commands<'_>, rows: &[(usize, u32, u32)]) {
        for (slot, s) in self.slots.iter().enumerate() {
            let Some(mm) = &s.mm else { continue };
            for (key, image) in mm.keys.iter().zip(&mm.images) {
                let Some(first) = rows.iter().position(|r| {
                    r.0 == slot
                        && r.2 as usize >= key.offset
                        && (r.2 as usize) < key.offset + key.nx * key.ny
                }) else {
                    continue;
                };
                let offset = rows[first].2 as usize - key.offset;
                let count = rows[first..]
                    .iter()
                    .take_while(|r| r.0 == slot && (r.2 as usize) < key.offset + key.nx * key.ny)
                    .count();
                cmd.dispatch(
                    if self.splash || (self.mlx && self.bonsai.is_none()) {
                        "splash_image_copy"
                    } else {
                        "spec_copy"
                    },
                    &[&image.embd, &self.scratch.x],
                    &[
                        (offset * self.width) as u32,
                        (first * self.width) as u32,
                        (count * self.width) as u32,
                    ],
                    [(count * self.width).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }

    pub(super) fn inject_mtp_images(&self, cmd: &Commands<'_>, rows: usize) {
        for (slot, state) in self.slots.iter().enumerate() {
            let Some(mm) = &state.mm else { continue };
            for (key, image) in mm.keys.iter().zip(&mm.images) {
                cmd.dispatch(
                    "vis_inject",
                    &[&image.embd, &self.scratch.meta, &self.scratch.x],
                    &[
                        self.width as u32,
                        rows as u32,
                        slot as u32,
                        key.offset as u32,
                        (key.nx * key.ny) as u32,
                    ],
                    [(rows * self.width).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }

    fn copy_image(&self, image: &vision::Output) -> Result<vision::Output> {
        let embd = self.device.alloc(image.embd.len())?;
        let cmd = self.device.begin()?;
        cmd.dispatch(
            "spec_copy",
            &[&image.embd, &embd],
            &[0, 0, (embd.len() / 4) as u32],
            [(embd.len() / 4).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish()?;
        Ok(vision::Output {
            embd,
            nx: image.nx,
            ny: image.ny,
        })
    }

    pub(super) fn admit_images(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let mut verdicts = Vec::new();
        let mut wave = Encoding {
            requests: Vec::new(),
            unique: Vec::new(),
            outputs: Vec::new(),
            missing: Vec::new(),
            in_flight: Vec::new(),
            job: None,
        };
        let mut patches = 0;
        for (slot, chunks) in items {
            if let Err(e) = self.require_committed() {
                verdicts.push((slot, MmAdmit::Failed(e.into())));
                continue;
            }
            if slot >= self.slots.len()
                || self.pending.iter().any(|p| p.slot == slot)
                || self
                    .encoding
                    .iter()
                    .any(|e| e.requests.iter().any(|r| r.0 == slot))
                || wave.requests.iter().any(|r| r.0 == slot)
            {
                verdicts.push((
                    slot,
                    MmAdmit::Failed(err(format!(
                        "invalid or already-admitted image slot {slot} (capacity {}, pending {:?}, encoding {:?}, wave {:?})",
                        self.slots.len(), self.pending.iter().map(|p| p.slot).collect::<Vec<_>>(),
                        self.encoding.iter().flat_map(|e| e.requests.iter().map(|r| r.0)).collect::<Vec<_>>(),
                        wave.requests.iter().map(|r| r.0).collect::<Vec<_>>()
                    )).into()),
                ));
                continue;
            }
            match self.layout(chunks) {
                Err(e) => verdicts.push((slot, MmAdmit::Failed(e.into()))),
                Ok(lay) => {
                    let added: usize = lay
                        .keys
                        .iter()
                        .filter(|k| !wave.unique.iter().any(|i| k.same_image(i)))
                        .map(|k| k.nx * k.ny * 4)
                        .sum();
                    if patches + added > 65_536 && !wave.requests.is_empty() {
                        self.encoding.push_back(wave);
                        wave = Encoding {
                            requests: Vec::new(),
                            unique: Vec::new(),
                            outputs: Vec::new(),
                            missing: Vec::new(),
                            in_flight: Vec::new(),
                            job: None,
                        };
                        patches = 0;
                    }
                    for key in &lay.keys {
                        if !wave.unique.iter().any(|i| key.same_image(i)) {
                            patches += key.nx * key.ny * 4;
                            wave.unique.push(key.clone());
                        }
                    }
                    wave.requests.push((slot, lay));
                    verdicts.push((slot, MmAdmit::Encoding));
                }
            }
        }
        if !wave.requests.is_empty() {
            self.encoding.push_back(wave);
        }
        verdicts
    }

    fn encode_wave(&mut self, wave: &mut Encoding) -> Result<bool> {
        // Patch setup, blocks and the merger share the same allowance. A
        // fresh job need not wait behind an unrelated decode just to enter
        // its first block, but setup time must not be charged twice or lost.
        let quantum = encoder_quantum(std::time::Duration::from_secs_f64(self.last_gpu_seconds));
        let started = std::time::Instant::now();
        if wave.outputs.is_empty() {
            for (i, key) in wave.unique.iter().enumerate() {
                if let Some(at) = self.image_cache.iter().position(|c| key.same_image(&c.key)) {
                    self.clock += 1;
                    self.image_cache[at].touched = self.clock;
                    wave.outputs
                        .push(Some(self.copy_image(&self.image_cache[at].output)?));
                    self.image_cache_reused += 1;
                } else {
                    wave.outputs.push(None);
                    wave.missing.push(i);
                }
            }
        }
        if wave.job.is_none() && !wave.missing.is_empty() {
            // One request may contain more than a wave's worth of images.
            // Bound scratch per encoder job, not the whole request: finish
            // successive sub-batches before publishing any prompt rows.
            let mut patches = 0;
            let n = wave
                .missing
                .iter()
                .take_while(|&&i| {
                    let key = &wave.unique[i];
                    patches += key.nx * key.ny * 4;
                    patches <= 65_536
                })
                .count();
            wave.in_flight = wave.missing.drain(..n).collect();
            let images = wave
                .in_flight
                .iter()
                .map(|&i| {
                    let k = &wave.unique[i];
                    (&*k.rgb, k.w, k.h)
                })
                .collect::<Vec<_>>();
            wave.job = Some(
                self.vision
                    .as_ref()
                    .expect("vision attached")
                    .start(&self.device, &images)?,
            );
            if started.elapsed() >= quantum {
                return Ok(false);
            }
        }
        if let Some(job) = &mut wave.job {
            // One block per scheduler tick badly over-fragments small images:
            // ~10 ms of ViT work waited behind 27 decode ticks at c4. Group
            // useful work within the shared wall quantum. Predict the
            // next block from the last completed one before admitting it;
            // one indivisible block may exceed the quantum on large images.
            // No timer, admission hold, or changed arithmetic is involved.
            let outputs = loop {
                if let Some(outputs) = self.vision.as_ref().expect("vision attached").step_budget(
                    &self.device,
                    job,
                    quantum.saturating_sub(started.elapsed()),
                )? {
                    break outputs;
                }
                if !encoder_budget_allows(started.elapsed(), job.block_cost, quantum) {
                    return Ok(false);
                }
            };
            tracing::info!(
                images = outputs.len(),
                gpu_ms = job.gpu_seconds * 1000.0,
                submits = job.submits,
                "Metal vision wave encoded"
            );
            for (&i, output) in wave.in_flight.iter().zip(outputs) {
                // Both byte bound and entry bound; never let repeated image
                // uploads consume the decoder's entire unified-memory grant.
                while !self.image_cache.is_empty()
                    && (self.image_cache.len() >= 16
                        || self
                            .image_cache
                            .iter()
                            .map(|c| c.output.embd.len() + c.key.rgb.len())
                            .sum::<usize>()
                            + output.embd.len()
                            + wave.unique[i].rgb.len()
                            > 256 << 20)
                {
                    let at = self
                        .image_cache
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, c)| c.touched)
                        .expect("nonempty")
                        .0;
                    self.image_cache.swap_remove(at);
                }
                if output.embd.len() + wave.unique[i].rgb.len() <= 256 << 20 {
                    self.clock += 1;
                    self.image_cache.push(CachedImage {
                        key: wave.unique[i].clone(),
                        output: self.copy_image(&output)?,
                        touched: self.clock,
                    });
                }
                wave.outputs[i] = Some(output);
            }
            wave.job = None;
            wave.in_flight.clear();
            if !wave.missing.is_empty() {
                return Ok(false);
            }
        }
        for (_, layout) in &mut wave.requests {
            for key in &layout.keys {
                let i = wave
                    .unique
                    .iter()
                    .position(|i| key.same_image(i))
                    .expect("image in wave");
                layout
                    .images
                    .push(self.copy_image(wave.outputs[i].as_ref().expect("encoded image"))?);
            }
        }
        Ok(true)
    }

    pub(super) fn step_images(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut wave) = self.encoding.pop_front() else {
            return Vec::new();
        };
        // Consult exact pixel/layout keys before invoking the tower. A restored
        // prefix already contains the image's contribution to every decoder
        // layer, recurrent state and companion. Its M-RoPE layout is rebuilt
        // from this request; no encoder output is needed for a text-only suffix.
        if self.cold.is_some() && wave.job.is_none() && wave.outputs.is_empty() {
            let mut parked = false;
            for (slot, layout) in &wave.requests {
                self.slots[*slot].cold_consulted = true;
                parked |= self.cold_loading_images(&layout.ids, &layout.keys);
            }
            if parked {
                self.encoding.push_front(wave);
                return Vec::new();
            }
            let mut verdicts = Vec::new();
            let mut remaining = Vec::new();
            for (slot, layout) in wave.requests.drain(..) {
                let covered = self.cache.iter().any(|c| {
                    !c.reserved
                        && c.history.len() < layout.ids.len()
                        && layout.ids.starts_with(&c.history)
                        && layout.keys.iter().all(|i| i.end() <= c.history.len())
                        && prefix_images_match(&c.images, &layout.keys, c.history.len())
                });
                if covered {
                    let tokens = layout.ids.clone();
                    match self.prepare_mm(slot, &tokens, Some(layout)) {
                        Ok(reused) => {
                            self.pending.push_back(Pending {
                                slot,
                                work: tokens.len() - reused,
                                tokens,
                                offset: reused,
                            });
                            verdicts.push((slot, MmAdmit::Queued));
                        }
                        Err(e) => verdicts.push((slot, MmAdmit::Failed(e.into()))),
                    }
                } else {
                    remaining.push((slot, layout));
                }
            }
            wave.requests = remaining;
            wave.unique.retain(|k| {
                wave.requests
                    .iter()
                    .any(|(_, l)| l.keys.iter().any(|i| k.same_image(i)))
            });
            if !verdicts.is_empty() || wave.requests.is_empty() {
                if !wave.requests.is_empty() {
                    self.encoding.push_front(wave);
                }
                return verdicts;
            }
        }
        match self.encode_wave(&mut wave) {
            Ok(false) => {
                self.encoding.push_front(wave);
                Vec::new()
            }
            Err(e) => wave
                .requests
                .into_iter()
                .map(|(slot, _)| {
                    let error = match &e {
                        MetalError::Memory(_) => GenError::OutOfMemory,
                        _ => GenError::Backend(e.to_string()),
                    };
                    (slot, MmAdmit::Failed(error))
                })
                .collect(),
            Ok(true) => wave
                .requests
                .into_iter()
                .map(|(slot, layout)| {
                    let tokens = layout.ids.clone();
                    let r = self.prepare_mm(slot, &tokens, Some(layout));
                    match r {
                        Ok(reused) => {
                            self.pending.push_back(Pending {
                                slot,
                                work: tokens.len() - reused,
                                tokens,
                                offset: reused,
                            });
                            (slot, MmAdmit::Queued)
                        }
                        Err(e) => (slot, MmAdmit::Failed(e.into())),
                    }
                })
                .collect(),
        }
    }

    pub(super) fn abort_images(&mut self, slot: usize) {
        for e in &mut self.encoding {
            e.requests.retain(|r| r.0 != slot);
        }
        self.encoding.retain(|e| !e.requests.is_empty());
    }

    pub(super) fn prefill_images(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize)> {
        // This compatibility entry point is exclusive. Draining another
        // caller's queued completion here would silently lose its logits.
        if !self.pending.is_empty() || !self.encoding.is_empty() {
            return Err(err(
                "synchronous multimodal prefill requires an empty admission queue",
            ));
        }
        for (_, result) in self.admit_images(vec![(slot, chunks.to_vec())]) {
            if let MmAdmit::Failed(e) = result {
                return Err(err(e.to_string()));
            }
        }
        while self
            .encoding
            .iter()
            .any(|e| e.requests.iter().any(|r| r.0 == slot))
        {
            for (_, result) in self.step_images() {
                if let MmAdmit::Failed(e) = result {
                    return Err(err(e.to_string()));
                }
            }
        }
        loop {
            let (_, done) = self
                .forward_mixed(&[], CHUNK)
                .map_err(|e| err(e.to_string()))?;
            if let Some((_, logits, n)) = done.into_iter().find(|r| r.0 == slot) {
                return Ok((logits, n));
            }
        }
    }
}

fn encoder_quantum(previous_forward: std::time::Duration) -> std::time::Duration {
    // Share the tick with measured backbone work. Leave 16 ms of the soft
    // 192 ms streaming quantum for sampling/HTTP and cost prediction error.
    // A cold/cheap backbone retains the full encoder grant. Stale or expensive
    // observations can only shrink it; step_budget still advances one block
    // even with zero allowance, so this is not a hard deadline or a wait.
    PREFILL_QUANTUM.min(std::time::Duration::from_millis(176).saturating_sub(previous_forward))
}

fn encoder_budget_allows(
    elapsed: std::time::Duration,
    last_block: std::time::Duration,
    quantum: std::time::Duration,
) -> bool {
    elapsed.saturating_add(last_block) < quantum
}

fn comparable_images(sizes: &[usize]) -> bool {
    let smallest = sizes.iter().copied().min().unwrap_or(0);
    let largest = sizes.iter().copied().max().unwrap_or(0);
    sizes.len() >= 2 && smallest >= 256 && largest <= smallest.saturating_mul(2)
}

#[cfg(test)]
mod budget_tests {
    use super::{encoder_budget_allows, encoder_quantum};
    use std::time::Duration;

    #[test]
    fn durable_image_identity_commits_pixels_geometry_and_position() {
        use super::{ImageKey, prefix_images_match};
        use paddock_engine::kv_tier::digest::LogicalKey;
        let a = ImageKey {
            digest: *blake3::hash(&[1, 2, 3]).as_bytes(),
            hash: 7,
            rgb: vec![1, 2, 3].into(),
            w: 1,
            h: 1,
            offset: 64,
            nx: 4,
            ny: 4,
        };
        let root = LogicalKey([0; 32]);
        let mut b = a.clone();
        b.digest = *blake3::hash(&[1, 2, 4]).as_bytes();
        b.rgb = vec![1, 2, 4].into();
        assert_ne!(
            a.chain(root),
            b.chain(root),
            "even a legacy 64-bit hash collision cannot alias"
        );
        let mut moved = a.clone();
        moved.offset += 16;
        assert_ne!(a.chain(root), moved.chain(root));
        let mut reshaped = a.clone();
        reshaped.nx = 2;
        reshaped.ny = 8;
        assert_ne!(a.chain(root), reshaped.chain(root));
        assert!(
            prefix_images_match(std::slice::from_ref(&a), &[b.clone()], 64),
            "different future images do not poison an earlier text prefix"
        );
        assert!(
            !prefix_images_match(std::slice::from_ref(&a), std::slice::from_ref(&a), 65),
            "never cut inside an image"
        );
        assert!(prefix_images_match(
            std::slice::from_ref(&a),
            std::slice::from_ref(&a),
            80
        ));
        assert!(!prefix_images_match(
            std::slice::from_ref(&a),
            std::slice::from_ref(&b),
            80
        ));
        let checkpoint = super::Checkpoint {
            history: vec![1; 80],
            images: vec![a.clone()],
            ..Default::default()
        };
        assert!(super::retention::superseded(
            &checkpoint,
            &[1; 112],
            &[a],
            96
        ));
        assert!(
            !super::retention::superseded(&checkpoint, &[1; 112], &[b], 96),
            "same placeholder tokens with different pixels are not a superseded branch"
        );
    }

    #[test]
    fn encoder_groups_cheap_blocks_but_yields_before_expensive_next_block() {
        let ms = Duration::from_millis;
        let budget = ms(64);
        assert!(encoder_budget_allows(ms(1), ms(1), budget));
        assert!(encoder_budget_allows(ms(8), ms(2), budget));
        assert!(encoder_budget_allows(ms(12), ms(6), budget));
        assert!(encoder_budget_allows(ms(20), ms(20), budget));
        assert!(!encoder_budget_allows(ms(50), ms(20), budget));
        assert!(!encoder_budget_allows(Duration::MAX, ms(1), budget));
        assert!(!encoder_budget_allows(ms(8), ms(2), ms(10)));
        assert!(!encoder_budget_allows(ms(0), ms(0), ms(0)));
    }

    #[test]
    fn encoder_allowance_accounts_for_the_preceding_backbone_pass() {
        let ms = Duration::from_millis;
        assert_eq!(encoder_quantum(ms(0)), ms(128));
        assert_eq!(encoder_quantum(ms(48)), ms(128));
        assert_eq!(encoder_quantum(ms(64)), ms(112));
        assert_eq!(encoder_quantum(ms(100)), ms(76));
        assert_eq!(encoder_quantum(ms(160)), ms(16));
        assert_eq!(encoder_quantum(ms(176)), ms(0));
        assert_eq!(encoder_quantum(Duration::MAX), ms(0));
    }

    #[test]
    fn cold_image_cohorts_exclude_short_or_asymmetric_work() {
        use super::comparable_images;
        assert!(comparable_images(&[680, 684, 678, 688]));
        assert!(!comparable_images(&[]));
        assert!(!comparable_images(&[680]));
        assert!(!comparable_images(&[128, 680]));
        assert!(!comparable_images(&[680, 2048]));
        assert!(comparable_images(&[1024, 2048]));
    }
}
