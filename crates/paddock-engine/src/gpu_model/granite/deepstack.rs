//! DeepStack injection: folding granite-vision's 8 streams into the LLM's
//! residual stream at 8 different depths.
//!
//! Design settled; the
//! five things that are silent when wrong, restated where they are implemented:
//!
//! 1. **Image slots are ZEROED**, not left holding the `<image>` placeholder
//!    embedding. Since the first thing written is stream 0, "zero then add" and
//!    "replace" are the same operation - `apply_embed` replaces.
//! 2. **`embedding_scale` (12.0) is for token embeddings only.** Vision rows
//!    enter unscaled. `apply_embed` therefore runs after the whole-buffer
//!    scale in `embed_rows`, overwriting it. llama.cpp gates the same way:
//!    `if (f_embedding_scale != 0 && (ubatch.token || n_deepstack_layers == 0))`
//!    - an embedding ubatch on a deepstack model skips the multiply entirely.
//! 3. **Injection is ADDITIVE**, not a scatter-replace, for streams 1..7.
//! 4. **It happens before the target layer runs**, i.e. before that layer's
//!    attn_norm reads `x`.
//! 5. **Stream 0 is not a layer injection** - it is the image's input
//!    embedding. `granite.deepstack_mapping` accordingly never names it, and
//!    layer 0's entry is -1. Upstream's `modeling.py` words this as an
//!    injection at layer 0 into zeroed slots; identical arithmetic.
//!
//! ## Addressing
//!
//! Spans are resolved from each row's `(slot, position)`, not from an offset
//! into the current chunk. A tick's row list mixes decode rows and prefill rows
//! from several slots, and a prompt can be cut at any row, so chunk-relative
//! addressing would be wrong the first time a chunk boundary landed inside an
//! image. Position-keyed lookup costs a few integer comparisons per row and is
//! correct for every cut.

use std::sync::Arc;

use crate::gpu::{GpuError, GpuExecutor};
use crate::gpu_model::granite::vision::MediaFeatures;

/// One media item placed in a slot's prompt: the rows `[pos, pos + tokens)`
/// of that slot carry its encoded features.
///
/// "Media" rather than "image" because granite-speech rides the same
/// machinery: an audio clip is one item whose features have a
/// single stream, so it takes the `apply_embed` replace and no layer
/// injections. Only one modality is ever live - the speech checkpoint has no
/// `deepstack_mapping` and the vision one has no audio tower - so the two
/// never share a registry at runtime, only the code.
pub(crate) struct PlacedMedia {
    /// First prompt position of the item's rows.
    pub pos: usize,
    /// Shared because a chunked prefill touches the same item across several
    /// ticks, and the streams are ~12 MB per image per tap.
    pub feats: Arc<MediaFeatures>,
}

impl PlacedMedia {
    fn contains(&self, p: usize) -> bool {
        p >= self.pos && p < self.pos + self.feats.tokens
    }
}

/// A contiguous run of rows in the CURRENT call that maps to a contiguous run
/// of one media item's stream rows.
pub(crate) struct InjectSpan {
    /// First row within this call's row buffer.
    pub dst_row: usize,
    /// First row within the item's streams - nonzero when a chunk boundary
    /// cut it, which is exactly the case chunk-relative addressing would get
    /// wrong.
    pub src_row: usize,
    pub n_rows: usize,
    pub feats: Arc<MediaFeatures>,
}

/// One encoded picture held for reuse: the 8 DeepStack streams plus enough to
/// prove a lookup is the same image (hash is the index, the bytes are the
/// proof).
pub(crate) struct GraniteImageCacheEntry {
    hash: u64,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
    feats: Arc<MediaFeatures>,
    /// device bytes across the streams - the budget unit
    bytes: usize,
    last_used: u64,
}

/// FNV-1a over the raw image bytes (dims folded in by the caller).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The cache index for one picture: content hash with the dimensions folded in.
/// An index only - every hit is confirmed against the exact bytes.
pub(crate) fn img_key(rgb: &[u8], w: usize, h: usize) -> u64 {
    fnv1a64(rgb) ^ (w as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (h as u64).rotate_left(32)
}

/// The radix content hash for one audio clip - the same role `img_key` plays
/// for a picture, over the sample bit patterns.
///
/// Not confirmed against the bytes afterwards (there is no audio cache to
/// confirm against), so this feeds the radix key alone, where a collision
/// would mean serving one clip's KV for another. Hence the whole clip, every
/// sample: a strided digest would be cheaper and would make two clips that
/// differ only between the strides indistinguishable. The cost is one FNV
/// pass over 4 bytes/sample - ~8 ms for a 10-minute clip against ~250 ms of
/// tower encode for the same clip.
pub(crate) fn audio_key(samples: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &s in samples {
        // to_bits, not the float: -0.0 and 0.0 are different samples to
        // hash under (they are the same audio, but never both appear in one
        // decode of the same file, so nothing hits less often for it)
        h ^= s.to_bits() as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^ (samples.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// VRAM the image cache may pin, `PADDOCK_IMG_CACHE_MB` (default 512 MiB).
/// Sized for a handful of ordinary pictures or a couple of max-grid ones; an
/// image bigger than the whole cap is served without being cached rather than
/// evicting everything for one entry.
pub(super) fn img_cache_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_IMG_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(512)
            * (1 << 20)
    })
}

/// Per-slot registry of images currently placed in that slot's prompt.
#[derive(Default)]
pub(crate) struct MediaRegistry {
    slots: Vec<Vec<PlacedMedia>>,
}

impl MediaRegistry {
    pub(crate) fn ensure_slots(&mut self, n: usize) {
        if self.slots.len() < n {
            self.slots.resize_with(n, Vec::new);
        }
    }

    /// Attach an image to `slot` starting at prompt position `pos`.
    pub(crate) fn place(&mut self, slot: usize, pos: usize, feats: Arc<MediaFeatures>) {
        self.ensure_slots(slot + 1);
        self.slots[slot].push(PlacedMedia { pos, feats });
    }

    /// Drop a slot's images - call when the sequence is reset or evicted, or
    /// the streams leak for the process's lifetime.
    pub(crate) fn clear_slot(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.clear();
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.is_empty())
    }

    /// Does this row resolve to an image row? Same test `plan` does but
    /// without the Arc clones, so the mm lane can check - independently of the
    /// row plan that built the stream - that the registry covers exactly the
    /// rows the plan reserved. That drift is the silent one: a registry that
    /// resolves nothing leaves the placeholder embedding in place and the model
    /// answers fluently about a picture it never saw.
    pub(crate) fn hits(&self, slot: u32, pos: usize) -> bool {
        self.slots
            .get(slot as usize)
            .is_some_and(|imgs| imgs.iter().any(|im| im.contains(pos)))
    }

    /// Resolve this tick's rows into injection spans.
    ///
    /// `rows` is the tick's `(slot, position, token)` list. Rows are walked in
    /// order and coalesced: consecutive rows of the same image whose positions
    /// advance by one become a single span. In the common case - one image, one
    /// slot, whole prompt in one chunk - that is exactly one span.
    pub(crate) fn plan(&self, rows: &[(u32, u32, u32)]) -> Vec<InjectSpan> {
        let hit = |slot: u32, p: usize| -> Option<(usize, usize)> {
            let imgs = self.slots.get(slot as usize)?;
            let idx = imgs.iter().position(|im| im.contains(p))?;
            Some((idx, p - imgs[idx].pos))
        };
        plan_spans(rows, hit)
            .into_iter()
            .map(|(dst_row, src_row, n_rows, slot, img)| InjectSpan {
                dst_row,
                src_row,
                n_rows,
                feats: Arc::clone(&self.slots[slot as usize][img].feats),
            })
            .collect()
    }
}

/// Collapse `n` images to the set that actually needs encoding.
///
/// `same(a, b)` reports whether images `a` and `b` are the same picture.
/// Returns the unique representatives (indices into the input, in first-seen
/// order) and, per input image, which representative serves it.
///
/// Two requests asking about the same new document in one wave is the case the
/// image cache cannot catch - nothing has been encoded yet - so it has to be
/// caught here or the tower runs over identical pixels twice. Kept free of
/// device buffers because the mapping back is where a wave would answer about
/// somebody else's picture.
pub(super) fn dedup_images(
    n: usize,
    same: impl Fn(usize, usize) -> bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut uniq: Vec<usize> = Vec::new();
    let mut of: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        match uniq.iter().position(|&u| same(u, i)) {
            Some(p) => of.push(p),
            None => {
                of.push(uniq.len());
                uniq.push(i);
            }
        }
    }
    (uniq, of)
}

/// The span arithmetic, free of device buffers so it can be tested directly.
///
/// `hit(slot, pos)` reports whether that row is an image row and, if so, which
/// image of the slot and how far into it. Returns
/// `(dst_row, src_row, n_rows, slot, image_index)` per coalesced run.
fn plan_spans(
    rows: &[(u32, u32, u32)],
    hit: impl Fn(u32, usize) -> Option<(usize, usize)>,
) -> Vec<(usize, usize, usize, u32, usize)> {
    let mut spans: Vec<(usize, usize, usize, u32, usize)> = Vec::new();
    for (i, &(slot, pos, _)) in rows.iter().enumerate() {
        let Some((img, src)) = hit(slot, pos as usize) else {
            continue;
        };
        // Extend the open span only if this row continues the same image of the
        // same slot, contiguously on both sides. Requiring contiguity in src as
        // well as dst is what stops two chunks of one image, or two slots'
        // images that happen to be adjacent in the row list, from merging into
        // one wrong span.
        match spans.last_mut() {
            Some(sp) if sp.0 + sp.2 == i && sp.1 + sp.2 == src && sp.3 == slot && sp.4 == img => {
                sp.2 += 1;
            }
            _ => spans.push((i, src, 1, slot, img)),
        }
    }
    spans
}

/// Write stream 0 over the image rows of the embedding buffer.
///
/// REPLACES rather than adds: the slots are defined to be zero before the first
/// injection, and whatever the placeholder token gathered (plus the ×12 scale
/// `embed_rows` just applied to the whole buffer) must not survive. This is the
/// step that keeps `embedding_scale` off the vision path.
pub(crate) fn apply_embed(
    exec: &GpuExecutor,
    x: &mut cudarc::driver::CudaSlice<f32>,
    spans: &[InjectSpan],
    embd: usize,
) -> Result<(), GpuError> {
    for sp in spans {
        let src = stream(sp, 0)?;
        exec.copy_region(
            src,
            sp.src_row * embd,
            x,
            sp.dst_row * embd,
            sp.n_rows * embd,
        )?;
    }
    Ok(())
}

/// Add stream `k` into the image rows, before layer `k`'s target runs.
pub(crate) fn apply_layer(
    exec: &GpuExecutor,
    x: &mut cudarc::driver::CudaSlice<f32>,
    spans: &[InjectSpan],
    embd: usize,
    k: usize,
) -> Result<(), GpuError> {
    for sp in spans {
        let src = stream(sp, k)?;
        exec.add_at(
            x,
            sp.dst_row * embd,
            src,
            sp.src_row * embd,
            sp.n_rows * embd,
        )?;
    }
    Ok(())
}

impl crate::gpu_model::granite::GpuGranite {
    /// Load the granite-vision mmproj alongside an already-loaded text model.
    ///
    /// Refuses the combination the two files can silently disagree on: the
    /// text model's `deepstack_mapping` names stream indices, and the mmproj
    /// decides how many streams exist. A mismatch would otherwise surface as a
    /// missing-stream error deep inside a prefill, or - worse, if the mmproj
    /// had more projectors than the map uses - as silently dropped taps.
    pub fn attach_vision(
        &mut self,
        mmproj: &paddock_models::mapped::MappedGguf,
    ) -> Result<(), GpuError> {
        let v = super::vision::VisionModel::load(Arc::clone(&self.exec), mmproj)?;
        let want = self.hp.deepstack.iter().copied().max().unwrap_or(-1);
        if !self.hp.has_deepstack() {
            return Err(GpuError::Driver(
                "this granite checkpoint has no deepstack_mapping - it is the text-only model, \
                 and an mmproj cannot be attached to it"
                    .into(),
            ));
        }
        // streams are 0..=want, so want+1 of them; stream 0 is the embedding
        if v.projs.len() != (want + 1) as usize {
            return Err(GpuError::Driver(format!(
                "granite-vision mmproj has {} projectors but the text model's deepstack_mapping \
                 references streams up to {want} ({} expected) - mismatched mmproj",
                v.projs.len(),
                want + 1
            )));
        }
        // The vocab must carry the placeholder the template renders, or the
        // prompt has nowhere to put the picture: the runner splits at this id
        // to build its chunks, and the mm lane fills the image's rows with it.
        // A vision mmproj on a vocab without it is a mismatched file pair.
        if self.img_pad_id.is_none() {
            return Err(GpuError::Driver(
                "granite-vision: the model's vocab has no `<image>` token, so there is no \
                 placeholder for the chat template to render or for the prompt to splice at \
                 - the mmproj and the text model are not a matching pair"
                    .into(),
            ));
        }
        self.vision = Some(v);
        Ok(())
    }

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// Per-layer vision-stream targets, as read from `granite.deepstack_mapping`
    /// (-1 = no injection at that layer).
    pub fn deepstack_map(&self) -> &[i32] {
        &self.hp.deepstack
    }

    /// How many `<image>` placeholder rows a picture of this size needs. Pure
    /// arithmetic - call it while BUILDING the prompt, before any pixels move.
    pub fn image_tokens(&self, w: usize, h: usize) -> Result<usize, GpuError> {
        self.vision
            .as_ref()
            .ok_or_else(|| GpuError::Driver("granite: no mmproj attached".into()))?
            .image_tokens(w, h)
    }

    /// Preprocess, encode and register one picture at `pos` in `slot`'s prompt.
    ///
    /// The returned row count must equal the placeholder run the prompt
    /// reserved. It is the same number [`Self::image_tokens`] gave, and both
    /// come out of the AnyRes plan - but the caller should compare them anyway,
    /// because a mismatch here is the failure mode that shifts every token
    /// after the image without erroring.
    pub fn place_image(
        &mut self,
        slot: usize,
        pos: usize,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<usize, GpuError> {
        let feats = self.encode_image_cached(rgb, w, h)?;
        Ok(self.place_feats(slot, pos, feats))
    }

    /// Register already-encoded features at `pos` in `slot`'s prompt - the
    /// hand-off from a batched wave encode, where the pixels were turned into
    /// streams before any slot's prefill started.
    pub(crate) fn place_feats(
        &mut self,
        slot: usize,
        pos: usize,
        feats: Arc<MediaFeatures>,
    ) -> usize {
        let n = feats.tokens;
        self.media.place(slot, pos, feats);
        n
    }

    /// Encode every image of a whole admission wave in one tower pass.
    ///
    /// Returns per-request, per-image features in chunk order, so the caller
    /// can hand each slot's prefill exactly the pictures its chunks name.
    ///
    /// Three filters run before anything reaches the GPU, in this order,
    /// because each one makes the next cheaper:
    ///
    /// 1. **Cache hits drop out** - a repeat picture is an `Arc` clone.
    /// 2. **The misses are deduplicated among themselves** - two requests in
    ///    one wave asking about the same new document encode it once. This is
    ///    the case the cache cannot catch (nothing has been encoded yet).
    /// 3. **What is left is concatenated into one pass.**
    ///
    /// Without this, N concurrent image requests each ran their own tower +
    /// 8 Q-Formers, strictly serially, before any of them prefilled - the vi8
    /// TTFT staircase. gemma4 still has that gap (notes it).
    pub(crate) fn encode_wave(
        &mut self,
        reqs: &[&[crate::service::MmChunk]],
    ) -> Result<Vec<Vec<Arc<MediaFeatures>>>, GpuError> {
        use crate::service::MmChunk;

        // Flatten to (request, image) so the mapping back is a total function
        // over one index space rather than a nested walk done twice.
        let flat: Vec<(usize, &[u8], usize, usize)> = reqs
            .iter()
            .enumerate()
            .flat_map(|(r, chunks)| {
                chunks.iter().filter_map(move |ch| match ch {
                    MmChunk::Image { rgb, w, h } => Some((r, rgb.as_slice(), *w, *h)),
                    // audio/directive chunks never route here; plan_rows
                    // rejects them before this walk, so skipping is
                    // unreachable-safe
                    MmChunk::Text(_)
                    | MmChunk::Audio { .. }
                    | MmChunk::OcrCrop(_)
                    | MmChunk::VisionPixels { .. } => None,
                })
            })
            .collect();
        let keys: Vec<u64> = flat
            .iter()
            .map(|&(_, rgb, w, h)| img_key(rgb, w, h))
            .collect();

        // 1. cache hits drop out - a repeat picture is an Arc clone
        let hits: Vec<Option<Arc<MediaFeatures>>> = flat
            .iter()
            .zip(&keys)
            .map(|(&(_, rgb, w, h), &k)| self.cache_lookup(k, rgb, w, h))
            .collect();

        // 2. what is left is deduplicated among itself
        let todo: Vec<usize> = (0..flat.len()).filter(|&i| hits[i].is_none()).collect();
        let (uniq, of) = dedup_images(todo.len(), |a, b| {
            let (ia, ib) = (todo[a], todo[b]);
            keys[ia] == keys[ib]
                && flat[ia].2 == flat[ib].2
                && flat[ia].3 == flat[ib].3
                && flat[ia].1 == flat[ib].1
        });

        // 3. One tower pass over every genuinely-new picture in the wave
        let encoded: Vec<Arc<MediaFeatures>> = if uniq.is_empty() {
            Vec::new()
        } else {
            let v = self
                .vision
                .as_ref()
                .ok_or_else(|| GpuError::Driver("granite: no mmproj attached".into()))?;
            let batch: Vec<(&[u8], usize, usize)> = uniq
                .iter()
                .map(|&u| {
                    let (_, rgb, w, h) = flat[todo[u]];
                    (rgb, w, h)
                })
                .collect();
            v.encode_images(&batch)?.into_iter().map(Arc::new).collect()
        };
        for (&u, feats) in uniq.iter().zip(&encoded) {
            let (_, rgb, w, h) = flat[todo[u]];
            self.cache_insert(keys[todo[u]], rgb, w, h, feats);
        }

        // 4. back to per-request order
        let mut out: Vec<Vec<Arc<MediaFeatures>>> = vec![Vec::new(); reqs.len()];
        let mut miss = 0usize;
        for (i, &(r, ..)) in flat.iter().enumerate() {
            let feats = match &hits[i] {
                Some(f) => Arc::clone(f),
                None => {
                    let f = Arc::clone(&encoded[of[miss]]);
                    miss += 1;
                    f
                }
            };
            out[r].push(feats);
        }
        Ok(out)
    }

    /// Encode one picture, or hand back the streams a previous request already
    /// produced for the exact same bytes.
    ///
    /// Worth 361 ms per repeat, measured with the 640x440 chart
    /// (image request vs a text-only prompt of comparable row count, min of 5,
    /// 1 output token so the number is prefill-dominated). Without it every
    /// turn of an image conversation re-runs the 27-block tower over up to 5
    /// tiles AND all 8 windowed Q-Formers to produce bytes it already had -
    /// on the agentic, prefix-heavy workload this project targets, that is the
    /// wrong thing to repeat. gemma4 and qwen35 have both cached their tower
    /// output since bring-up; granite was the family that didn't.
    ///
    /// A hit is a pure `Arc` clone, no device copy: `MediaFeatures` is
    /// read-only after encode (`apply_embed`/`apply_layer` only read
    /// `feats.streams`) and the registry already holds it behind an `Arc`, so
    /// the cache and the placement share one allocation. gemma4's cache
    /// predates that shape and does a dtod clone per hit.
    ///
    /// Keyed by hash AND verified against the exact bytes - a collision must
    /// never answer about a different picture.
    fn encode_image_cached(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Result<Arc<MediaFeatures>, GpuError> {
        let key = img_key(rgb, w, h);
        if let Some(feats) = self.cache_lookup(key, rgb, w, h) {
            return Ok(feats);
        }
        let v = self
            .vision
            .as_ref()
            .ok_or_else(|| GpuError::Driver("granite: no mmproj attached".into()))?;
        let feats = Arc::new(v.encode_image(rgb, w, h)?);
        self.cache_insert(key, rgb, w, h, &feats);
        Ok(feats)
    }

    /// Hand back a previously-encoded picture, or `None`. Keyed by hash AND
    /// verified against the exact bytes - a collision must never answer about a
    /// different picture.
    pub(super) fn cache_lookup(
        &mut self,
        key: u64,
        rgb: &[u8],
        w: usize,
        h: usize,
    ) -> Option<Arc<MediaFeatures>> {
        self.img_cache_clock += 1;
        let clock = self.img_cache_clock;
        let e = self
            .img_cache
            .iter_mut()
            .find(|e| e.hash == key && e.w == w && e.h == h && e.rgb == rgb)?;
        e.last_used = clock;
        let feats = Arc::clone(&e.feats);
        self.img_cache_reused += 1;
        Some(feats)
    }

    /// Admit a freshly-encoded picture, evicting LRU entries to stay inside the
    /// byte budget. An image bigger than the whole cap is served without being
    /// cached rather than evicting everything for one entry.
    pub(super) fn cache_insert(
        &mut self,
        key: u64,
        rgb: &[u8],
        w: usize,
        h: usize,
        feats: &Arc<MediaFeatures>,
    ) {
        // BYTE-budgeted, not entry-counted: granite's entry is the 8 DeepStack
        // streams, so it is ~45 MB for a 2x2-grid picture and ~190 MB for a
        // max-grid strip - two orders over gemma4's ~3 MB soft-token block,
        // where a fixed 16-entry cap would be several GB of VRAM.
        let bytes: usize = feats.streams.iter().map(|s| s.len() * 4).sum();
        let cap = img_cache_cap();
        if bytes > cap {
            return;
        }
        while self.img_cache_bytes + bytes > cap && !self.img_cache.is_empty() {
            let lru = self
                .img_cache
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .expect("non-empty");
            // Dropping the entry releases the cache's Arc. If a live request
            // still has this image placed, its Arc keeps the streams alive
            // until that slot is cleared - eviction bounds what the CACHE pins,
            // not what serving needs.
            self.img_cache_bytes -= self.img_cache[lru].bytes;
            self.img_cache.swap_remove(lru);
        }
        self.img_cache_bytes += bytes;
        self.img_cache.push(GraniteImageCacheEntry {
            hash: key,
            w,
            h,
            rgb: rgb.to_vec(),
            feats: Arc::clone(feats),
            bytes,
            last_used: self.img_cache_clock,
        });
    }

    /// Cache hits so far - the serving-side witness that repeat images are not
    /// re-encoded (telemetry + tests).
    pub fn image_cache_reuses(&self) -> u64 {
        self.img_cache_reused
    }

    /// Device bytes the image cache currently pins.
    pub fn image_cache_bytes(&self) -> usize {
        self.img_cache_bytes
    }

    /// Forget a slot's images - on sequence reset or eviction. Without this the
    /// streams (~12 MB per image per tap) stay alive as long as the process.
    pub fn clear_slot_images(&mut self, slot: usize) {
        self.media.clear_slot(slot);
    }
}

fn stream(sp: &InjectSpan, k: usize) -> Result<&cudarc::driver::CudaSlice<f32>, GpuError> {
    sp.feats.streams.get(k).ok_or_else(|| {
        GpuError::Driver(format!(
            "granite deepstack wants vision stream {k} but the projector produced only {} - \
             the mmproj's projector count and the text model's deepstack_mapping disagree",
            sp.feats.streams.len()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{dedup_images, plan_spans};

    /// `same` from a list of picture identities - the shape `encode_wave`
    /// builds out of (hash, w, h, bytes).
    fn by_id(ids: &[u32]) -> impl Fn(usize, usize) -> bool + '_ {
        move |a, b| ids[a] == ids[b]
    }

    #[test]
    fn distinct_pictures_all_get_encoded() {
        let (uniq, of) = dedup_images(4, by_id(&[7, 8, 9, 10]));
        assert_eq!(uniq, vec![0, 1, 2, 3]);
        assert_eq!(of, vec![0, 1, 2, 3]);
    }

    /// The case the image cache cannot catch: two requests in one wave asking
    /// about the same picture that has never been encoded. The tower must run
    /// over it once, and both requests must land on that one result.
    #[test]
    fn the_same_new_picture_twice_in_a_wave_encodes_once() {
        let (uniq, of) = dedup_images(4, by_id(&[7, 8, 7, 8]));
        assert_eq!(uniq, vec![0, 1], "two distinct pictures, two encodes");
        assert_eq!(of, vec![0, 1, 0, 1], "requests 2 and 3 reuse them");
    }

    /// Every image must resolve to a representative that is that image - the
    /// mapping is what makes a wave answer about the right picture.
    #[test]
    fn every_image_maps_to_a_representative_of_itself() {
        let ids = [3u32, 1, 3, 3, 2, 1];
        let (uniq, of) = dedup_images(ids.len(), by_id(&ids));
        assert_eq!(of.len(), ids.len(), "the mapping is total");
        for (i, &rep) in of.iter().enumerate() {
            assert_eq!(
                ids[uniq[rep]], ids[i],
                "image {i} mapped to a different picture"
            );
        }
        assert_eq!(uniq.len(), 3, "3 distinct pictures among 6");
    }

    #[test]
    fn an_empty_wave_needs_no_encode() {
        let (uniq, of) = dedup_images(0, |_, _| unreachable!("nothing to compare"));
        assert!(uniq.is_empty() && of.is_empty());
    }

    /// One image at position `pos` spanning `n` rows, in slot `s`.
    fn one_image(s: u32, pos: usize, n: usize) -> impl Fn(u32, usize) -> Option<(usize, usize)> {
        move |slot, p| (slot == s && p >= pos && p < pos + n).then(|| (0, p - pos))
    }

    #[test]
    fn whole_image_in_one_chunk_is_one_span() {
        // 3 text rows, then 144 image rows
        let rows: Vec<(u32, u32, u32)> = (0..147).map(|p| (0u32, p as u32, 0u32)).collect();
        let spans = plan_spans(&rows, one_image(0, 3, 144));
        assert_eq!(spans, vec![(3, 0, 144, 0, 0)]);
    }

    /// The case position-keyed addressing exists for: a chunk boundary lands
    /// inside the image, so the second chunk must start at src row 40, not 0.
    /// Chunk-relative addressing would replay the image's first rows here.
    #[test]
    fn chunk_cut_inside_an_image_resumes_at_the_right_source_row() {
        let first: Vec<(u32, u32, u32)> = (0..43).map(|p| (0u32, p as u32, 0u32)).collect();
        assert_eq!(
            plan_spans(&first, one_image(0, 3, 144)),
            vec![(3, 0, 40, 0, 0)]
        );

        // next chunk resumes at prompt position 43 => source row 40
        let second: Vec<(u32, u32, u32)> = (43..147).map(|p| (0u32, p as u32, 0u32)).collect();
        assert_eq!(
            plan_spans(&second, one_image(0, 3, 144)),
            vec![(0, 40, 104, 0, 0)]
        );
    }

    #[test]
    fn text_only_rows_produce_no_spans() {
        let rows: Vec<(u32, u32, u32)> = (0..64).map(|p| (0u32, p as u32, 0u32)).collect();
        assert!(plan_spans(&rows, one_image(0, 1000, 144)).is_empty());
    }

    /// A mixed tick: decode rows for other slots, then a slot's image rows.
    /// The decode rows must not be swept into the span, and the image rows must
    /// still resolve even though they do not start at row 0.
    #[test]
    fn decode_rows_ahead_of_image_rows_do_not_join_the_span() {
        let mut rows = vec![(1u32, 500u32, 0u32), (2u32, 900u32, 0u32)];
        rows.extend((10..20).map(|p| (0u32, p as u32, 0u32)));
        let spans = plan_spans(&rows, one_image(0, 10, 144));
        assert_eq!(spans, vec![(2, 0, 10, 0, 0)]);
    }

    /// Two slots whose image rows are adjacent in the row list must stay two
    /// spans even when dst and src happen to line up.
    #[test]
    fn two_slots_never_merge() {
        let hit = |slot: u32, p: usize| -> Option<(usize, usize)> {
            ((slot == 0 || slot == 1) && (4..8).contains(&p)).then(|| (0, p - 4))
        };
        let rows = vec![
            (0u32, 4u32, 0u32),
            (0u32, 5u32, 0u32),
            (1u32, 6u32, 0u32),
            (1u32, 7u32, 0u32),
        ];
        // rows 2..3 continue contiguously in dst AND src, but belong to slot 1
        assert_eq!(
            plan_spans(&rows, hit),
            vec![(0, 0, 2, 0, 0), (2, 2, 2, 1, 0)]
        );
    }
}
