//! The ENCODER BUDGET: a wave's vision encode, spread across scheduler ticks.
//!
//! Putting image prompts on the chunked prefill queue made the
//! QUEUEING half free - and left the ENCODE as the whole remaining stall. The
//! route witness said so directly:
//!
//! ```text
//! pd route: mm chunked-admit 1 reqs - encode 329.9ms, queue 0.2ms
//! ```
//!
//! 270-330 ms of one max-grid picture, during which no decode row moves and no
//! amount of PREFILL chunking helps, because none of it is prefill. This module
//! is the other half: vLLM V1's `max_num_encoder_input_tokens` design, where
//! the scheduler spends a bounded slice of encoder work per step instead of
//! running a request's encoder to completion inline.
//!
//! ## Why granite makes this tractable
//!
//! A tile never sees another tile - `vision_attn_at` runs one attention per
//! tile and `WindowIndices::build` replicates the window tables per tile with
//! shifted offsets - so encoding a SLICE of a tile stack produces exactly the
//! rows that slice would have contributed to the whole. See `VisionModel::
//! encode`, where the property is stated next to the code that depends on it.
//! A budgeted group is therefore a slice, not a second code path.
//!
//! ALGEBRAICALLY exactly, not bit-exactly, and the difference is worth stating
//! because it is easy to over-claim here. Every GEMM's M changes with the group
//! size, so cuBLAS picks different tiles and accumulates in a different order:
//! `PADDOCK_VISION_VERIFY=1` measures max |Δ| 0.04 against a peak |x| of 19.4
//! (~0.2%) between a 1-tile-group encode and the whole stack, scattered with no
//! structure. Same class as batched-vs-serial prefill, and it moved 1 greedy
//! answer in a 48-cell picture x question x group-size sweep. Corruption looks
//! nothing like this: it would be a contiguous BAND of rows wrong by O(|x|),
//! which is exactly what `wave_verify` exists to tell apart.
//!
//! ## The three questions this settles
//!
//! **Where partial work lives between ticks.** In [`WaveEncode`], owned by the
//! model, one per admission wave. It holds the wave's own chunks: the service
//! hands the images over at admission and gets a slot back only once it is
//! genuinely queued, so there is never a moment where two owners could disagree
//! about whose picture it is.
//!
//! **Tiles or rows.** TILES. A tile is the indivisible unit of the tower and of
//! the Q-Formers, and its cost is flat - a tile is a tile whatever picture it
//! came from - which a row budget would have to keep converting back. The
//! budget is deliberately not shared with the prefill row budget: encoder work
//! and decoder work cost differently per unit, and vLLM keeps its two budgets
//! separate for the same reason.
//!
//! **Interleave or run to completion.** Interleave within a wave, serialize
//! across waves. A wave's pictures concatenate into one tile stack (that is the
//! batching) and the stack is then sliced by the budget, so every group is
//! still a shared tower pass. A wave arriving while another is in flight queues
//! behind it rather than being folded in mid-stack - folding would mean
//! re-planning a stack that already has rows written into it.
//!
//! ## The fast path is unchanged
//!
//! When a wave's tiles fit inside one budget - the common case, since an
//! ordinary photo is 2-5 tiles - registration encodes the whole thing in its
//! first step and the slot is queued in the same call the scheduler made. That
//! is the pre-budget path exactly, including its timing.

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::GpuGranite;
use super::preprocess::AnyResPlan;
use super::vision::MediaFeatures;
use crate::generator::{GenError, MmAdmit};
use crate::gpu::GpuError;
use crate::service::MmChunk;

/// Tiles one budgeted group encodes (`PADDOCK_VISION_TILES`).
///
/// A value at or above a wave's tile count is the unbudgeted encode exactly,
/// which is what makes the A/B honest - and is also why the pin below exists
/// as its own switch rather than as "set this very high".
pub(crate) fn tile_budget() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_VISION_TILES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_TILE_BUDGET)
    })
}

/// Default group size, measured on the A6000 against granite-vision-4.1-4b
/// with the 3840x384 max-grid strip (11 tiles), no other load:
///
/// ```text
///   tiles/group   groups   max group   total encode
///        1          11       42.1 ms      813.6 ms
///        2           6       41.4          590.0
///        3           4       56.4          573.0     <- default
///        5           3       88.5          542.7
///       16           1      247.0          543.2     (= unbudgeted)
/// ```
///
/// The shape of that table is the whole argument. Group cost falls roughly
/// linearly down to ~3 and then stops falling - below it the per-pass fixed
/// costs (window tables, tap allocations, 8 separate Q-Former launches)
/// dominate and only the TOTAL moves, +50% at a group of 1 for no further tail
/// improvement. 3 sits at the knee: 4.4x off the worst hold for ~5% on an
/// encode that is not on anyone's critical path when the server is busy - and
/// when it is idle there is nothing to stall, so the 5% is invisible there too.
const DEFAULT_TILE_BUDGET: usize = 3;

/// A/B pin: restores the whole-wave encode this replaced, so the stall it
/// removes can be measured rather than asserted.
pub(crate) fn budget_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_VISION_NO_BUDGET").is_some()
}

/// Where the group starting at `cursor` ends, given `total` tiles in the stack.
///
/// Trivial, and pulled out anyway because the accumulation buffer's correctness
/// is exactly "successive calls to this partition `[0, total)` with no gap and
/// no overlap". A gap leaves a band of the buffer at its allocated zeros, which
/// is a picture with blank rows in the middle of it - served, plausible-looking
/// and completely silent. The test below is that property, not this arithmetic.
fn group_end(cursor: usize, total: usize, budget: usize) -> usize {
    total.min(cursor.saturating_add(budget))
}

/// One admission wave's encode, in flight across ticks.
///
/// The mapping vectors (`flat` .. `of`) are built once at registration and are
/// read-only after: they are what turns a flat, deduplicated tile stack back
/// into per-request, per-chunk features, and rebuilding them per group would be
/// both wasted work and a chance for the two orders to drift.
pub(crate) struct WaveEncode {
    /// The wave itself. Held rather than handed back, so there is exactly one
    /// owner of the pixels while they are mid-encode.
    items: Vec<(usize, Vec<MmChunk>)>,
    /// Slots whose request died while this wave was encoding - skipped at
    /// finalize rather than removed, because every mapping vector below indexes
    /// into `items` and compacting it would invalidate all of them.
    dead: Vec<usize>,
    /// One entry per image in the wave, in (request, chunk) order.
    flat: Vec<(usize, usize)>,
    /// Content key per flat image (image-cache index; hits are byte-verified).
    keys: Vec<u64>,
    /// Cache hits, resolved at registration - these never reach the tower.
    hits: Vec<Option<Arc<MediaFeatures>>>,
    /// Flat indices that missed the cache.
    todo: Vec<usize>,
    /// `todo` positions that are DISTINCT pictures - two requests in one wave
    /// asking about the same new document encode it once.
    uniq: Vec<usize>,
    /// `todo` position -> its position in `uniq`.
    of: Vec<usize>,
    /// Per unique picture: its AnyRes plan and its first TILE in the stack.
    /// Both are pure arithmetic, so both are known before a pixel moves.
    plans: Vec<AnyResPlan>,
    bases: Vec<usize>,
    /// Normalized tiles, filled LAZILY as the cursor reaches each picture - the
    /// host-side tiling (bicubic resize + normalize) is real work and belongs
    /// inside a budgeted step, not in a lump at registration.
    ///
    /// Granularity is one PICTURE, not one tile: `AnyResPlan::tiles` cuts a
    /// whole image in one go. So the group that first reaches a picture carries
    /// all of its tiling, which is why the first group of a wave measures
    /// larger than the rest (72 ms vs ~18 ms on the max-grid strip). Splitting
    /// that further means teaching the plan to cut one tile at a time; it is
    /// worth doing only if host tiling ever dominates a group, and today it
    /// does not.
    tiles: Vec<Vec<f32>>,
    /// Tiles the finished stack will hold, from `plans` alone.
    total_tiles: usize,
    /// Projected rows accumulated so far, per unique picture, one buffer per
    /// DeepStack stream - allocated when the picture's first tile is encoded
    /// and packed into its features (then freed) the moment its last tile is.
    ///
    /// It used to be one set sized for the WHOLE wave up front, and packing
    /// then made a newline copy and a packed copy of all of it at once: ~12
    /// GiB at the peak for 32 max-grid pictures, allocated before the first
    /// tile encoded. Per picture, the transient is one picture's worth.
    acc: Vec<Option<Vec<CudaSlice<f32>>>>,
    /// Per unique picture, its packed features once all its tiles are in.
    packed: Vec<Option<Arc<MediaFeatures>>>,
    /// Tiles encoded so far.
    cursor: usize,
}

impl WaveEncode {
    fn slots(&self) -> Vec<usize> {
        self.items.iter().map(|(k, _)| *k).collect()
    }

    fn done(&self) -> bool {
        self.cursor >= self.total_tiles
    }

    /// The picture at flat position `i`: bytes, width, height.
    fn pic(&self, i: usize) -> (&[u8], usize, usize) {
        image_at(&self.items, self.flat[i])
    }
}

/// The picture at one flat position. A free function rather than a closure so
/// the returned borrow elides to `items` cleanly - the whole wave is read
/// through this, and every caller needs it to coexist with a `&mut self`.
fn image_at(items: &[(usize, Vec<MmChunk>)], at: (usize, usize)) -> (&[u8], usize, usize) {
    match &items[at.0].1[at.1] {
        MmChunk::Image { rgb, w, h } => (rgb, *w, *h),
        MmChunk::Text(_)
        | MmChunk::Audio { .. }
        | MmChunk::OcrCrop(_)
        | MmChunk::VisionPixels { .. } => {
            unreachable!("flat indexes image chunks only")
        }
    }
}

/// Every slot of a wave gets the same systemic verdict. An encode failure is
/// alloc/driver/planning, never one request's fault, so half-serving the wave
/// would be reporting a lie to whichever half got through.
fn fail_all(slots: Vec<usize>, e: impl std::fmt::Display) -> Vec<(usize, MmAdmit)> {
    let msg = e.to_string();
    slots
        .into_iter()
        .map(|k| (k, MmAdmit::Failed(GenError::Backend(msg.clone()))))
        .collect()
}

impl GpuGranite {
    /// The plan's charges for image and audio input, which allocate on
    /// demand and so were never in it.
    ///
    /// The tower pass is MEASURED: one budget group of blank max-grid tiles
    /// through the tower, its pool high-water read at load (a profile run -
    /// eight Q-Former stacks deep, arithmetic would drift from the code). On
    /// top of it one picture's accumulation, newline copy and packed features
    /// (a picture packs the moment its last tile encodes), and the image
    /// cache's own byte cap.
    pub(crate) fn vision_reserves(&self) -> Result<Vec<crate::kv_plan::Reserve>, GpuError> {
        // the speech tower's scratch grows on demand with the longest clip,
        // and stops at its segment cap - the room left to grow is reserved
        let mut out = Vec::new();
        if let Some(a) = self.audio.as_ref() {
            out.push(crate::kv_plan::Reserve::new(
                "audio encoder growth",
                a.scratch_growth_bytes() as u64,
            ));
        }
        let Some(v) = self.vision.as_ref() else {
            return Ok(out);
        };
        let b = v.budget();
        let edge = b.max_edge.map_or(3840, |e| e as usize);
        let short = (b.max_pixels as usize / edge).max(1);
        let blank = vec![0u8; short * edge * 3];
        let (_, tiles, _) = v.tile_stack(&[(&blank, short, edge)])?;
        let group = tile_budget().clamp(1, tiles.len().max(1));
        let (_, tower) = self.exec.pool_peak_during(|| v.encode(&tiles[..group]))?;
        let per_tile = (v.tokens_per_tile() * v.proj_width() * std::mem::size_of::<f32>()) as u64;
        let picture = 3 * tiles.len() as u64 * per_tile * v.n_streams() as u64;
        out.push(crate::kv_plan::Reserve::new(
            "vision tower pass",
            tower + picture,
        ));
        out.push(crate::kv_plan::Reserve::new(
            "image cache",
            super::deepstack::img_cache_cap() as u64,
        ));
        Ok(out)
    }

    /// Register an admission wave and spend the first budget group on it.
    ///
    /// Returns one verdict per slot: `Queued` when the wave finished inside this
    /// call (the ordinary case - an everyday photo is 2-5 tiles), `Encoding`
    /// when it did not.
    pub(crate) fn wave_begin(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let slots: Vec<usize> = items.iter().map(|(k, _)| *k).collect();
        let wave = match self.wave_plan(items) {
            Ok(w) => w,
            Err(e) => return fail_all(slots, e),
        };
        let first = self.enc.is_empty();
        self.enc.push_back(wave);
        // Only the FRONT wave is in flight; one queued behind another waits.
        if first {
            let out = self.wave_step();
            if !out.is_empty() {
                return out;
            }
        }
        slots.into_iter().map(|k| (k, MmAdmit::Encoding)).collect()
    }

    /// Spend one budget group on the wave at the head of the queue.
    ///
    /// Returns per-slot results only when a wave completed in this call -
    /// otherwise an empty vec, meaning "still going". The scheduler calls this
    /// once per tick and moves the reported slots out of its encoding set.
    pub(crate) fn wave_step(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(front) = self.enc.front() else {
            return Vec::new();
        };
        // A wave of nothing but cache hits has no tiles at all, and finalizes
        // without the tower ever running.
        if !front.done() {
            let budget = if budget_disabled() {
                usize::MAX
            } else {
                tile_budget()
            };
            let end = group_end(front.cursor, front.total_tiles, budget);
            if let Err(e) = self.encode_group(end) {
                let wave = self.enc.pop_front().expect("front checked");
                return fail_all(wave.slots(), e);
            }
        }
        if !self.enc.front().expect("front checked").done() {
            return Vec::new();
        }
        let wave = self.enc.pop_front().expect("front checked");
        self.wave_finish(wave)
    }

    /// True while any wave is mid-encode - the scheduler's cue to keep calling
    /// [`Self::wave_step`].
    pub(crate) fn wave_pending(&self) -> bool {
        !self.enc.is_empty()
    }

    /// Forget slots whose request died. Their features are never packed and
    /// their prefill never queued, but the wave still runs to completion for
    /// everyone else in it - the tile stack is shared, so there is nothing to
    /// usefully cancel, and finishing still populates the image cache.
    pub(crate) fn wave_release(&mut self, occupied: &[bool]) {
        for wave in &mut self.enc {
            let gone: Vec<usize> = wave
                .items
                .iter()
                .map(|(k, _)| *k)
                .filter(|k| !occupied.get(*k).copied().unwrap_or(false))
                .filter(|k| !wave.dead.contains(k))
                .collect();
            wave.dead.extend(gone);
        }
    }

    /// Build the wave's mapping and its accumulation buffers. No pixels move:
    /// the tile stack is PLANNED (`AnyResPlan::n_tiles`) but not cut.
    fn wave_plan(&mut self, items: Vec<(usize, Vec<MmChunk>)>) -> Result<WaveEncode, GpuError> {
        // Flatten to (request, chunk) so the mapping back is a total function
        // over one index space rather than a nested walk done twice.
        let flat: Vec<(usize, usize)> = items
            .iter()
            .enumerate()
            .flat_map(|(r, (_, chunks))| {
                chunks
                    .iter()
                    .enumerate()
                    .filter(|(_, ch)| matches!(ch, MmChunk::Image { .. }))
                    .map(move |(ci, _)| (r, ci))
            })
            .collect();
        let keys: Vec<u64> = flat
            .iter()
            .map(|&at| {
                let (rgb, w, h) = image_at(&items, at);
                super::deepstack::img_key(rgb, w, h)
            })
            .collect();

        // 1. cache hits drop out - a repeat picture is an Arc clone
        let mut hits: Vec<Option<Arc<MediaFeatures>>> = Vec::with_capacity(flat.len());
        for (i, &at) in flat.iter().enumerate() {
            let (rgb, w, h) = image_at(&items, at);
            hits.push(self.cache_lookup(keys[i], rgb, w, h));
        }

        // 2. what is left is deduplicated among itself - two requests in one
        //    wave asking about the same new document encode it once, which the
        //    cache cannot catch because nothing has been encoded yet
        let todo: Vec<usize> = (0..flat.len()).filter(|&i| hits[i].is_none()).collect();
        let (uniq, of) = super::deepstack::dedup_images(todo.len(), |a, b| {
            let (ra, wa, ha) = image_at(&items, flat[todo[a]]);
            let (rb, wb, hb) = image_at(&items, flat[todo[b]]);
            keys[todo[a]] == keys[todo[b]] && (wa, ha) == (wb, hb) && ra == rb
        });

        // 3. plan the tile stack - arithmetic only, so the whole shape (and
        //    with it the accumulation size) is known before a pixel moves
        let v = self
            .vision
            .as_ref()
            .ok_or_else(|| GpuError::Driver("granite: no mmproj attached".into()))?;
        let mut plans = Vec::with_capacity(uniq.len());
        let mut bases = Vec::with_capacity(uniq.len());
        let mut total_tiles = 0usize;
        for &u in &uniq {
            let (_, w, h) = image_at(&items, flat[todo[u]]);
            let plan = v.plan(w, h).ok_or_else(|| {
                GpuError::Driver(format!("granite-vision: cannot plan a {w}x{h} image"))
            })?;
            bases.push(total_tiles);
            total_tiles += plan.n_tiles();
            plans.push(plan);
        }
        let acc = (0..uniq.len()).map(|_| None).collect();
        let packed = (0..uniq.len()).map(|_| None).collect();

        Ok(WaveEncode {
            items,
            dead: Vec::new(),
            flat,
            keys,
            hits,
            todo,
            uniq,
            of,
            plans,
            bases,
            tiles: Vec::new(),
            total_tiles,
            acc,
            packed,
            cursor: 0,
        })
    }

    /// Encode the front wave's tiles `[cursor, end)` and append them to its
    /// accumulation buffers.
    ///
    /// Host tiling happens here too, lazily, for exactly the pictures this group
    /// reaches: a bicubic resize of a large source is not free, and doing the
    /// whole wave's worth at registration would put the very lump back into the
    /// tick that the budget exists to break up.
    fn encode_group(&mut self, end: usize) -> Result<(), GpuError> {
        let v = self
            .vision
            .as_ref()
            .ok_or_else(|| GpuError::Driver("granite: no mmproj attached".into()))?;
        let wave = self.enc.front_mut().expect("caller checked");
        // Cut forward until the stack covers this group. `bases` is ascending
        // and starts at 0, so the picture to cut next is always the last one
        // whose base is at or before the current length.
        while wave.tiles.len() < end {
            let i = wave
                .bases
                .iter()
                .rposition(|&b| b <= wave.tiles.len())
                .expect("bases[0] == 0");
            debug_assert_eq!(
                wave.bases[i],
                wave.tiles.len(),
                "a cut starts at a picture boundary"
            );
            let (rgb, w, h) = image_at(&wave.items, wave.flat[wave.todo[wave.uniq[i]]]);
            let (_, cut, _) = v.tile_stack(&[(rgb, w, h)])?;
            wave.tiles.extend(cut);
        }
        let streams = v.encode(&wave.tiles[wave.cursor..end])?.streams;
        let per_row = v.tokens_per_tile() * v.proj_width();
        let verify = paddock_models::dev_var_os!("PADDOCK_VISION_VERIFY").is_some();
        // scatter the group's rows into each picture it touches, and pack every
        // picture whose last tile this group encoded
        for j in 0..wave.plans.len() {
            let (b0, b1) = (wave.bases[j], wave.bases[j] + wave.plans[j].n_tiles());
            let (t0, t1) = (b0.max(wave.cursor), b1.min(end));
            if t0 >= t1 {
                continue;
            }
            if wave.acc[j].is_none() {
                let rows = (b1 - b0) * per_row;
                wave.acc[j] = Some(
                    (0..streams.len())
                        .map(|_| self.exec.alloc(rows))
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            let acc = wave.acc[j].as_mut().expect("allocated above");
            for (dst, src) in acc.iter_mut().zip(&streams) {
                self.exec.copy_region(
                    src,
                    (t0 - wave.cursor) * per_row,
                    dst,
                    (t0 - b0) * per_row,
                    (t1 - t0) * per_row,
                )?;
            }
            if t1 == b1 {
                let rows = (b1 - b0) * v.tokens_per_tile();
                let feats = v
                    .pack_wave(acc, rows, std::slice::from_ref(&wave.plans[j]), &[0])?
                    .pop()
                    .expect("one plan, one picture");
                wave.packed[j] = Some(Arc::new(feats));
                if !verify {
                    wave.acc[j] = None;
                }
            }
        }
        // the host tiles are spent once encoded (verify re-encodes them all)
        if !verify {
            for t in &mut wave.tiles[wave.cursor..end] {
                *t = Vec::new();
            }
        }
        wave.cursor = end;
        Ok(())
    }

    /// `PADDOCK_VISION_VERIFY=1`: re-encode the finished stack in one pass and
    /// report how far the budgeted accumulation drifted from it.
    ///
    /// The budget's whole correctness claim is that a tile slice encodes to
    /// exactly the rows it would have contributed to the whole, so this is the
    /// claim stated as a measurement. Kept in the tree (behind the env var, and
    /// it doubles the work when on) because "the answer changed" is a terrible
    /// place to start debugging an encoder: it cannot tell a corrupted row from
    /// a GEMM that picked a different tile size at a different M.
    fn wave_verify(&self, wave: &WaveEncode) {
        let Some(v) = self.vision.as_ref() else {
            return;
        };
        let whole = match v.encode(&wave.tiles) {
            Ok(f) => f.streams,
            Err(e) => {
                tracing::warn!("vision-verify: whole-stack encode failed: {e}");
                return;
            }
        };
        // the per-picture accumulations laid end to end, as the whole-stack
        // encode lays them
        let per_tile = v.tokens_per_tile() * v.proj_width();
        let mut joined: Vec<Vec<f32>> = vec![Vec::new(); whole.len()];
        for acc in wave.acc.iter().flatten() {
            for (s, buf) in acc.iter().enumerate() {
                match self.exec.to_host(buf) {
                    Ok(h) => joined[s].extend(h),
                    Err(_) => return,
                }
            }
        }
        debug_assert!(
            joined
                .iter()
                .all(|j| j.len() == wave.total_tiles * per_tile)
        );
        for (i, (ha, b)) in joined.into_iter().zip(&whole).enumerate() {
            let hb = match self.exec.to_host(b) {
                Ok(y) => y,
                Err(_) => return,
            };
            let per_row = v.proj_width();
            let (mut worst, mut at) = (0f32, 0usize);
            for (j, (&x, &y)) in ha.iter().zip(&hb).enumerate() {
                let d = (x - y).abs();
                if d > worst {
                    worst = d;
                    at = j;
                }
            }
            let mag = hb.iter().fold(0f32, |m, &y| m.max(y.abs()));
            tracing::info!(
                "vision-verify: stream {i} max|Δ| {worst:.6} at row {} (peak |x| {mag:.3})",
                at / per_row
            );
        }
    }

    /// Pack a finished stack back into per-request features and queue every
    /// live slot's prompt onto the chunked prefill lane.
    fn wave_finish(&mut self, wave: WaveEncode) -> Vec<(usize, MmAdmit)> {
        if paddock_models::dev_var_os!("PADDOCK_VISION_VERIFY").is_some() {
            self.wave_verify(&wave);
        }
        // every picture was packed as its last tile encoded
        let encoded: Vec<Arc<MediaFeatures>> = match wave.packed.iter().cloned().collect() {
            Some(p) => p,
            None => {
                return fail_all(
                    wave.slots(),
                    GpuError::Driver("granite: a wave finished with a picture unpacked".into()),
                );
            }
        };
        for (j, &u) in wave.uniq.iter().enumerate() {
            let i = wave.todo[u];
            let (rgb, w, h) = wave.pic(i);
            self.cache_insert(wave.keys[i], rgb, w, h, &encoded[j]);
        }

        // Back to per-request order: a hit keeps its own Arc, a miss takes the
        // one its dedup class produced.
        let mut per_req: Vec<Vec<Arc<MediaFeatures>>> = vec![Vec::new(); wave.items.len()];
        let mut miss = 0usize;
        for (i, &(r, _)) in wave.flat.iter().enumerate() {
            let feats = match &wave.hits[i] {
                Some(f) => Arc::clone(f),
                None => {
                    let f = Arc::clone(&encoded[wave.of[miss]]);
                    miss += 1;
                    f
                }
            };
            per_req[r].push(feats);
        }

        let dead = wave.dead;
        let items = wave.items;
        items
            .into_iter()
            .zip(per_req)
            .map(|((k, chunks), feats)| {
                // A slot whose request died still reports, so the scheduler can
                // drop it from its encoding set; there is just nothing to queue.
                if dead.contains(&k) {
                    return (k, MmAdmit::Queued);
                }
                match self.multimodal_prefill_begin(k, &chunks, Some(&feats)) {
                    Ok(_rows) => (k, MmAdmit::Queued),
                    Err(e) => (k, MmAdmit::Failed(GenError::Backend(e.to_string()))),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::group_end;

    /// The accumulation invariant: successive groups tile `[0, total)` exactly
    /// once, at every budget and every stack size. A gap here is a band of the
    /// accumulation buffer left at its allocated zeros - an image with blank
    /// rows through the middle, served without a word of complaint.
    #[test]
    fn groups_partition_the_stack_at_every_budget() {
        for total in 0..40usize {
            for budget in 1..14usize {
                let (mut cursor, mut spans, mut steps) = (0usize, Vec::new(), 0);
                while cursor < total {
                    let end = group_end(cursor, total, budget);
                    assert!(
                        end > cursor,
                        "total {total} budget {budget}: no progress at {cursor}"
                    );
                    assert!(
                        end <= total,
                        "total {total} budget {budget}: ran past the stack"
                    );
                    spans.push((cursor, end));
                    cursor = end;
                    steps += 1;
                    assert!(
                        steps <= total + 1,
                        "total {total} budget {budget}: did not converge"
                    );
                }
                assert_eq!(
                    cursor, total,
                    "total {total} budget {budget}: stopped short"
                );
                // contiguous and gapless, which is what the buffer needs
                assert!(
                    spans.windows(2).all(|w| w[0].1 == w[1].0),
                    "total {total} budget {budget}: {spans:?} is not contiguous"
                );
                assert_eq!(spans.first().map(|s| s.0).unwrap_or(0), 0);
                assert_eq!(spans.iter().map(|s| s.1 - s.0).sum::<usize>(), total);
            }
        }
    }

    /// The unbudgeted pin is one group, whatever the stack - that is what makes
    /// the A/B a comparison of scheduling rather than of two different encodes.
    #[test]
    fn an_unbounded_budget_is_exactly_one_group() {
        for total in 1..40usize {
            assert_eq!(group_end(0, total, usize::MAX), total);
        }
    }
}
