//! Vision splice for the DeepSeek-OCR batch lane: DeepEncoder
//! output rows woven into the decoder's prefill row stream in the order the
//! reference's `masked_scatter_` fills them - which is the FEATURE concat
//! order `[local crops, global view, view_seperator]`, not the placeholder-id
//! order (see [`super::tiling::Block`] for the receipts; getting this
//! backwards keeps every token count right and quietly answers about the
//! wrong part of the page).
//!
//! Per image, one device plane `[image_tokens, n_embd]` is assembled once:
//! the encoded views land in a staging buffer (crops, then the global view,
//! then the two learned rows), and a single `pixel_shuffle_rows` gather walks
//! [`super::tiling::Layout::blocks`] to stitch the crop grid, drop an
//! `image_newline` at the end of every token row, and close with the
//! separator. Prefill chunks then splice with one contiguous dtod copy per
//! image run - the interleaved layout [text..][image..][text..] guarantees at
//! most a few runs per chunk.
//!
//! ## Prefix caching (the point of doing this carefully)
//!
//! Every image row carries the same `<image>` placeholder id, so a radix
//! keyed on row TOKENS would serve one picture's KV for another. Image rows
//! key on the picture's content hash instead
//! ([`crate::gpu_model::prefix_cache::image_key_row`], granite's scheme).
//! "Same page, many questions" then resumes past the whole picture - and the
//! tower runs only for the sources the recomputed tail actually reads: a
//! block-aligned resume that lands in the global view's last rows re-encodes
//! the 1 global view, not the up-to-32 crops ([`SpliceSource`] accounting).
//!
//! Unlike gemma4v there are no non-causal image spans here - the reference
//! runs plain causal attention over the whole sequence - so resumes may land
//! anywhere, no `cut_outside_image_spans` guard needed.
//!
//! Multi-page pixels follow the REFERENCE (each page `ImageOps.pad`ed to the
//! 1024² base view, aspect preserved) rather than the lossy squash-to-640²
//! fallback other implementations use. Tokens are identical either way, and
//! where two implementations disagree on pixels the checkpoint's own code
//! wins.

use cudarc::driver::CudaSlice;

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::gpu_model::prefix_cache::image_key_row;
use crate::service::MmChunk;

use super::batch::PfCuts;
use super::encode::MAX_VIEWS;
use super::load::GpuDeepseekOcr;
use super::tiling::{Block, Layout, Mode};

/// The family's `<image>` placeholder id (`image_token_id = 128815` in the
/// reference's prepare loop). Only ever a stand-in: every row carrying it has
/// its embedding overwritten from the image plane before the first layer.
pub const IMAGE_TOKEN_ID: u32 = 128_815;

/// One prefill row of the interleaved stream.
#[derive(Clone, Copy)]
pub(crate) enum Row {
    Token(u32),
    /// (image index, row within that image's spliced plane)
    Image(usize, usize),
}

/// FNV-1a-style over the raw image bytes with the dimensions folded in - the
/// same number keys the radix rows, so "same picture" has exactly one
/// definition. Four independent 64-bit lanes, 32 bytes per step: the
/// byte-serial version cost 3-6 ms of admission-wave host time per page (the
/// c8 idle probe's gap.adm segment). The key is process-internal
/// (in-memory radix only), so the exact function is free to change as long
/// as it stays deterministic within a serve.
fn content_hash(rgb: &[u8], w: usize, h: usize) -> u64 {
    const P: u64 = 0x100000001b3;
    let mut lane = [
        0xcbf29ce484222325u64,
        0x84222325cbf29ce4,
        0x9ce484222325cbf2,
        0x25cbf29ce4842223,
    ];
    let (words, rest) = rgb.as_chunks::<32>();
    for c in words {
        for (i, l) in lane.iter_mut().enumerate() {
            let word = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().expect("8-byte word"));
            *l = (*l ^ word).wrapping_mul(P);
        }
    }
    let mut hash = lane[0];
    for &l in &lane[1..] {
        hash = (hash ^ l).wrapping_mul(P);
    }
    for &b in rest {
        hash ^= b as u64;
        hash = hash.wrapping_mul(P);
    }
    hash ^ (w as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (h as u64).rotate_left(32)
}

/// Fold one image's block geometry into its radix key. Same picture, same
/// bytes - but a gundam encode and a base encode of it produce entirely
/// different KV rows, and their key streams share a prefix (row 0, 1, 2, ...)
/// unless the geometry disambiguates. Without this fold, a page cached from a
/// single-image (gundam) request could be resumed into a multi-page (base)
/// prompt - or, with the crop override, into a forced-base one - silently
/// splicing the wrong rows. Deliberately derived from the BLOCKS rather than
/// the mode: the gundam small-image bail emits exactly base's
/// `[Global, Separator]`, its rows really are identical, and sharing them
/// across modes is correct.
fn geom_code(blocks: &[Block]) -> u64 {
    let mut code = 0x51_7c_c1_b7_27_22_0a_95u64;
    for b in blocks {
        let v = match *b {
            Block::Local { rows, cols } => 1 | (rows as u64) << 8 | (cols as u64) << 32,
            Block::Global { side } => 2 | (side as u64) << 8,
            Block::Separator => 3,
        };
        code = (code.rotate_left(13) ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    code
}

/// One request's plan: the layout, the interleaved row stream with its radix
/// keys, and per-image geometry - everything derivable without touching the
/// GPU, so the wave can build it for every slot before any slot executes.
/// Per-image blocks are RANGES into `layout.blocks` (the layout owns them).
/// Fully OWNED (pixels resolve through [`images_of`]) so the chunked lane can
/// hold a plan across ticks next to the chunks it was made from.
pub(super) struct MmPlan {
    pub(super) layout: Layout,
    pub(super) img_blocks: Vec<std::ops::Range<usize>>,
    pub(super) img_tokens: Vec<usize>,
    pub(super) rows: Vec<Row>,
    pub(super) keys: Vec<u32>,
    pub(super) img_start: Vec<usize>,
    pub(super) n_rows: usize,
}

/// The (rgb, w, h) of every image chunk, in chunk order - the resolver the
/// plan's per-image vectors index into.
pub(super) fn images_of(chunks: &[MmChunk]) -> Vec<(&[u8], usize, usize)> {
    chunks
        .iter()
        .filter_map(|ch| match ch {
            MmChunk::Image { rgb, w, h } => Some((rgb.as_slice(), *w, *h)),
            _ => None,
        })
        .collect()
}

/// Plan one request: one Layout across all images (mode elected from the
/// COUNT unless overridden, a grid per image), per-image block runs, then the
/// interleaved row stream and its radix keys (content-derived for image rows
/// - the module doc's whole argument). An `OcrCrop` directive chunk is this
///   family's per-request override: `Base` forces the no-crop layout,
///   `Gundam` tiles every image (Unlimited-OCR-2's multi-page gundam; the
///   runner refuses it on v1, whose reference mandates non-crop for
///   multi-image).
pub(super) fn mm_plan(chunks: &[MmChunk], max_tiles: usize) -> Result<MmPlan, GpuModelError> {
    let mut sizes: Vec<(usize, usize)> = Vec::new();
    let mut crop = None;
    for ch in chunks {
        match ch {
            MmChunk::Image { w, h, .. } => {
                sizes.push((*w, *h));
            }
            MmChunk::Text(_) => {}
            MmChunk::OcrCrop(m) => crop = Some(*m),
            MmChunk::VisionPixels { .. } => {
                return Err(GpuError::Driver(
                    "pixel-budget directive on deepseek-ocr - its crop planner \
                     sizes images itself (routing bug)"
                        .into(),
                )
                .into());
            }
            MmChunk::Audio { .. } => {
                return Err(GpuError::Driver(
                    "deepseek-ocr serves images, not audio - routing bug".into(),
                )
                .into());
            }
        }
    }
    if sizes.is_empty() {
        return Err(GpuError::Driver("multimodal prefill with no image".into()).into());
    }
    let layout = Layout::plan(&sizes, max_tiles, crop);
    let img_blocks = layout.img_blocks.clone();
    debug_assert_eq!(img_blocks.len(), sizes.len());
    let img_tokens: Vec<usize> = img_blocks
        .iter()
        .map(|r| layout.blocks[r.clone()].iter().map(Block::tokens).sum())
        .collect();

    let mut rows: Vec<Row> = Vec::new();
    let mut keys: Vec<u32> = Vec::new();
    let mut img_start: Vec<usize> = Vec::new();
    let mut img_k = 0usize;
    for ch in chunks {
        match ch {
            MmChunk::Text(ids) => {
                rows.extend(ids.iter().map(|&t| Row::Token(t)));
                keys.extend_from_slice(ids);
            }
            MmChunk::Image { rgb, w, h } => {
                let hash = content_hash(rgb, *w, *h)
                    ^ geom_code(&layout.blocks[img_blocks[img_k].clone()]);
                img_start.push(rows.len());
                rows.extend((0..img_tokens[img_k]).map(|r| Row::Image(img_k, r)));
                keys.extend((0..img_tokens[img_k]).map(|r| image_key_row(hash, r)));
                img_k += 1;
            }
            MmChunk::OcrCrop(_) => {} // consumed in the sizing pass
            MmChunk::Audio { .. } | MmChunk::VisionPixels { .. } => {
                unreachable!("rejected above")
            }
        }
    }
    let n_rows = rows.len();
    Ok(MmPlan {
        layout,
        img_blocks,
        img_tokens,
        rows,
        keys,
        img_start,
        n_rows,
    })
}

/// Which tower sources rows `[need_from, ..)` of one image's plane read:
/// the block walk reads crops inside the Local block, the global view inside
/// Global (whose last row is the separator, hence the -1).
pub(super) fn prep_need(
    blocks: &[Block],
    grid: Option<super::tiling::Grid>,
    need_from: usize,
) -> (bool, bool) {
    let local: usize = blocks
        .iter()
        .filter(|b| matches!(b, Block::Local { .. }))
        .map(Block::tokens)
        .sum();
    let total: usize = blocks.iter().map(Block::tokens).sum();
    let tiles = grid.map_or(0, |g| g.tiles());
    (tiles > 0 && need_from < local, need_from < total - 1)
}

/// Everything a preprocess worker needs to build a [`PreppedImage`] without
/// touching the model. All Copy - snapshotted at plan time.
#[derive(Clone, Copy)]
pub(super) struct PrepSpec {
    pub(super) grid: Option<super::tiling::Grid>,
    pub(super) base_px: usize,
    pub(super) tile_px: usize,
    pub(super) fill: [u8; 3],
    pub(super) need_crops: bool,
    pub(super) need_global: bool,
}

/// Host-side preprocess product for one image: the PIL-parity resample/crop
/// outputs as RAW u8 interleaved-RGB view planes, detached from the model so
/// it can run on worker threads. Normalize + im2row + f16 moved onto the
/// device (`pd_ocr_patches_u8`): the f32 patch plane was 4x the
/// upload bytes and its host-side build + staged copy were the c8 idle
/// probe's gap.enc segment.
#[derive(Default)]
pub(super) struct PreppedImage {
    /// (first tile index, view count, u8 pixel plane) per crop chunk of
    /// [`MAX_VIEWS`].
    pub(super) crops: Vec<(usize, usize, Vec<u8>)>,
    /// The padded global view's u8 pixel plane.
    pub(super) global: Option<Vec<u8>>,
}

/// Drain the prep channel until job `j` has landed, then hand it over.
/// A closed channel (worker died) yields whatever already arrived - the
/// caller's inline fallback covers the rest.
fn wait_job(
    rx: &std::sync::mpsc::Receiver<(usize, PreppedImage)>,
    got: &mut [Option<PreppedImage>],
    j: usize,
) -> Option<PreppedImage> {
    while got[j].is_none() {
        match rx.recv() {
            Ok((i, p)) => got[i] = Some(p),
            Err(_) => break,
        }
    }
    got[j].take()
}

/// The pure host half of the tower run - see [`PreppedImage`].
pub(super) fn prep_image(rgb: &[u8], w: usize, h: usize, s: &PrepSpec) -> PreppedImage {
    let mut crops: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    if s.need_crops {
        let g = s.grid.expect("need_crops has a grid");
        // dynamic_preprocess: one resize of the original to the grid's pixel
        // envelope, then exact 640² crops in row-major tile order
        let resized =
            super::preprocess::resize_rgb8(rgb, w, h, g.cols * s.tile_px, g.rows * s.tile_px);
        for (c0, n) in (0..g.tiles())
            .collect::<Vec<_>>()
            .chunks(MAX_VIEWS)
            .map(|c| (c[0], c.len()))
        {
            let mut px = Vec::with_capacity(n * s.tile_px * s.tile_px * 3);
            for t in c0..c0 + n {
                px.extend(super::preprocess::crop_tile(
                    &resized,
                    g.cols * s.tile_px,
                    s.tile_px,
                    t % g.cols,
                    t / g.cols,
                ));
            }
            crops.push((c0, n, px));
        }
    }
    let global = s.need_global.then(|| {
        // the global view always comes from the ORIGINAL image, aspect
        // preserved on the mean-color square - never from the crop resize
        super::preprocess::pad_to_square(rgb, w, h, s.base_px, s.fill)
    });
    PreppedImage { crops, global }
}

impl GpuDeepseekOcr {
    /// Snapshot of everything a preprocess worker needs, from the attached
    /// tower and one layout's geometry. Need flags start false; callers flip
    /// exactly the one they mean (`PrepSpec { need_crops: true, ..spec }`).
    pub(super) fn prep_spec(&self, mode: Mode, grid: Option<super::tiling::Grid>) -> PrepSpec {
        let vis = self.vision.as_ref().expect("vision attached");
        let mean = vis.hp.image_mean;
        // ImageOps.pad fill = `tuple(int(x * 255) for x in mean)` - trunc, 127
        let fill = [
            (mean[0] * 255.0) as u8,
            (mean[1] * 255.0) as u8,
            (mean[2] * 255.0) as u8,
        ];
        let (base_px, tile_px) = mode.px();
        PrepSpec {
            grid,
            base_px,
            tile_px,
            fill,
            need_crops: false,
            need_global: false,
        }
    }

    /// The request-pricing budget, computed from the attached tower's tile
    /// budget (32 on Unlimited-OCR, 6 on the DeepSeek-OCR pair). The pixel
    /// bounds are the crop grid's envelope: past `max_pixels` the grid resize
    /// discards detail; there is no useful lower bound (small images take the
    /// global-only bail and get padded up).
    pub(crate) fn vision_budget_impl(&self) -> Option<crate::generator::VisionBudget> {
        let vis = self.vision.as_ref()?;
        let (tile, max_tiles) = (
            super::vision::TILE_PX as u64,
            self.tile_cap(vis.hp.max_tiles),
        );
        let max_tok = super::tiling::max_image_tokens(max_tiles) as u32;
        Some(crate::generator::VisionBudget {
            max_pixels: max_tiles as u64 * tile * tile,
            min_pixels: 1,
            max_edge: None,
            // one 640² tile is ~110 tokens (100 + 10 newline slots)
            pixels_per_token: tile * tile / 110,
            max_tokens: max_tok,
            // the global-only bail: one padded 1024 view + separator
            min_tokens: 273,
        })
    }

    /// Attach the DeepEncoder tower from the mmproj GGUF.
    pub fn attach_vision(
        &mut self,
        map: &paddock_models::mapped::MappedGguf,
    ) -> Result<(), GpuError> {
        let vm = super::vision::DeepEncoder::load(self.exec.clone(), map)?;
        if vm.hp.llm_embd != self.hp.n_embd {
            return Err(GpuError::Driver(format!(
                "ocr mmproj projects to {} but the decoder width is {}",
                vm.hp.llm_embd, self.hp.n_embd
            )));
        }
        // Splice staging at the tile budget's worst case (a full-grid gundam
        // image), so `assemble_image_plane` never allocates it mid-serve. The
        // staging is [crops][global][newline][separator] rows; the index is
        // one entry per plane row.
        let t_tile = vm.hp.tokens_per_side(super::vision::TILE_PX);
        let t_base = vm.hp.tokens_per_side(super::vision::BASE_PX);
        let src_max = (vm.hp.max_tiles * t_tile * t_tile + t_base * t_base + 2) * self.hp.n_embd;
        self.mm_src = Some(self.exec.alloc(src_max)?);
        self.mm_idx = Some(
            self.exec
                .alloc_u32(super::tiling::max_image_tokens(vm.hp.max_tiles))?,
        );
        // fold the tower into the model's weight total (paddleocr_vl
        // precedent, load.rs:184) - without this the will-it-fit surface
        // read ~755 MB of tower weights as free VRAM
        self.weights_bytes += vm.weight_bytes() as u64;
        self.vision = Some(vm);
        Ok(())
    }

    /// Multimodal prefill into `slot` - the wave of one. Serial-surface
    /// callers (warmup, the classic per-slot path) pay one chunk clone; the
    /// serving path arrives through [`Self::multimodal_prefill_wave`] with
    /// the whole admission wave and never lands here.
    pub fn multimodal_prefill_slot(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let mut r = self.multimodal_prefill_wave(vec![(slot, chunks.to_vec())]);
        r.pop().expect("wave of one").1
    }

    /// A whole multimodal admission wave, pipelined (door B):
    /// host preprocessing - the PIL-parity resamples and the im2row, ~100 ms
    /// a page - used to run inline per slot, so in a c4 wave the last request
    /// waited out three strangers' host work before its own GPU time started.
    /// Now: plan + PROBE the radix per slot (read-only; the real resume still
    /// happens per slot in wave order, so same-wave dedupe is exactly what it
    /// was), preprocess every needed image on worker threads, and run the
    /// serial GPU phase in wave order, consuming each slot's planes as they
    /// land (completion-order channel - the first slot never waits on the
    /// last slot's preprocess).
    pub(crate) fn multimodal_prefill_wave(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), GpuModelError>)> {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        let mut out: Vec<(usize, Result<(Vec<f32>, usize), GpuModelError>)> =
            Vec::with_capacity(items.len());
        let Some(vis) = self.vision.as_ref() else {
            return items
                .into_iter()
                .map(|(k, _)| {
                    (
                        k,
                        Err(GpuError::Driver("no ocr mmproj attached".into()).into()),
                    )
                })
                .collect();
        };
        let max_tiles = self.tile_cap(vis.hp.max_tiles);

        // ── phase 1, serial: plan every slot and probe the radix for its
        // expected resume, which decides what is worth preprocessing.
        let plans: Vec<(usize, Result<MmPlan, GpuModelError>)> = items
            .iter()
            .map(|(slot, chunks)| (*slot, mm_plan(chunks, max_tiles)))
            .collect();
        let imgs: Vec<Vec<(&[u8], usize, usize)>> =
            items.iter().map(|(_, chunks)| images_of(chunks)).collect();
        // One job per SOURCE (crops / global), not per image: the two read
        // independent inputs (crops the grid resize, global the original),
        // so a single request's global prep runs while its crops encode -
        // the overlap the old inline path got for free and a per-image job
        // would lose.
        let mut jobs: Vec<(&[u8], usize, usize, PrepSpec)> = Vec::new();
        let mut job_of: std::collections::HashMap<(usize, usize), (Option<usize>, Option<usize>)> =
            std::collections::HashMap::new();
        for (wi, (_, planned)) in plans.iter().enumerate() {
            let Ok(plan) = planned else { continue };
            let probe = self.prefix_probe(&plan.keys);
            for (k, &(rgb, w, h)) in imgs[wi].iter().enumerate() {
                let spec = self.prep_spec(plan.layout.mode, plan.layout.grids[k]);
                if plan.img_start[k] + plan.img_tokens[k] <= probe {
                    continue; // expected fully resumed: nothing to preprocess
                }
                let need_from = probe.saturating_sub(plan.img_start[k]);
                let blocks = &plan.layout.blocks[plan.img_blocks[k].clone()];
                let (need_crops, need_global) = prep_need(blocks, plan.layout.grids[k], need_from);
                let jc = need_crops.then(|| {
                    jobs.push((
                        rgb,
                        w,
                        h,
                        PrepSpec {
                            need_crops: true,
                            ..spec
                        },
                    ));
                    jobs.len() - 1
                });
                let jg = need_global.then(|| {
                    jobs.push((
                        rgb,
                        w,
                        h,
                        PrepSpec {
                            need_global: true,
                            ..spec
                        },
                    ));
                    jobs.len() - 1
                });
                job_of.insert((wi, k), (jc, jg));
            }
        }

        // ── phases 2+3, overlapped: workers preprocess (jobs are in wave
        // order, so early slots complete first); the GPU phase consumes each
        // slot's prepped images off the channel the moment they exist.
        let n_workers = jobs
            .len()
            .min(std::thread::available_parallelism().map_or(1, |n| n.get()));
        let trace = paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some();
        let t0 = trace.then(std::time::Instant::now);
        if trace {
            tracing::info!(
                "mm-wave: {} slots, {} prep jobs on {n_workers} workers",
                items.len(),
                jobs.len(),
            );
        }
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::channel::<(usize, PreppedImage)>();
            let jobs_ref = &jobs;
            let next_ref = &next;
            for _ in 0..n_workers {
                let tx = tx.clone();
                s.spawn(move || {
                    loop {
                        let i = next_ref.fetch_add(1, Relaxed);
                        if i >= jobs_ref.len() {
                            break;
                        }
                        let (rgb, w, h, spec) = &jobs_ref[i];
                        // a dead receiver just means the GPU phase failed early
                        let _ = tx.send((i, prep_image(rgb, *w, *h, spec)));
                    }
                });
            }
            drop(tx);

            let mut got: Vec<Option<PreppedImage>> = (0..jobs.len()).map(|_| None).collect();
            for (wi, (slot, planned)) in plans.into_iter().enumerate() {
                let plan = match planned {
                    Ok(p) => p,
                    Err(e) => {
                        out.push((slot, Err(e)));
                        continue;
                    }
                };
                let res = (|| {
                    self.admit(slot, plan.n_rows)?;
                    let start = self.prefill_resume_rows(slot, &plan.keys, plan.n_rows)?;
                    // towers, lazily: an image runs only the encodes whose
                    // rows the recomputed tail [start, n) actually reads
                    let mut planes: Vec<Option<CudaSlice<f32>>> = Vec::new();
                    for (k, &(rgb, w, h)) in imgs[wi].iter().enumerate() {
                        if plan.img_start[k] + plan.img_tokens[k] <= start {
                            planes.push(None); // fully resumed: no tower at all
                            continue;
                        }
                        let (jc, jg) = job_of.get(&(wi, k)).copied().unwrap_or((None, None));
                        let crops = jc
                            .and_then(|j| wait_job(&rx, &mut got, j))
                            .map(|p| p.crops)
                            .unwrap_or_default();
                        // deferred deliberately: assemble calls this after the
                        // crops encode launches, so the worker preps the
                        // global while the GPU chews the crops
                        let global = || {
                            jg.and_then(|j| wait_job(&rx, &mut got, j))
                                .and_then(|p| p.global)
                        };
                        let need_from = start.saturating_sub(plan.img_start[k]);
                        planes.push(Some(self.assemble_image_plane(
                            rgb,
                            w,
                            h,
                            crops,
                            global,
                            &plan.layout.blocks[plan.img_blocks[k].clone()],
                            plan.layout.grids[k],
                            plan.layout.mode,
                            need_from,
                        )?));
                    }
                    self.mm_walk(slot, &plan, &planes, start)
                })();
                if let Some(t0) = t0 {
                    tracing::info!(
                        "mm-wave: slot {slot} done at {:.0} ms ({})",
                        t0.elapsed().as_secs_f64() * 1e3,
                        if res.is_ok() { "ok" } else { "err" },
                    );
                }
                out.push((slot, res));
            }
        });
        out
    }

    /// The chunked pass, text path's shape with the splice added. All three
    /// position streams are the true position: the ring is not armed until
    /// the boundary mark below, so wpos == apos == pos throughout.
    fn mm_walk(
        &mut self,
        slot: usize,
        plan: &MmPlan,
        planes: &[Option<CudaSlice<f32>>],
        start: usize,
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        if self.hp.dash.is_some() {
            let ids: Vec<u32> = plan
                .rows
                .iter()
                .map(|row| match row {
                    Row::Token(t) => *t,
                    Row::Image(..) => IMAGE_TOKEN_ID,
                })
                .collect();
            self.dash_admit(slot, &ids);
        }
        let mut base = start;
        let mut last_len = 0usize;
        for chunk in plan.rows[start..].chunks(pf_rows()) {
            let r = chunk.len();
            let toks: Vec<u32> = chunk
                .iter()
                .map(|row| match row {
                    Row::Token(t) => *t,
                    Row::Image(..) => IMAGE_TOKEN_ID,
                })
                .collect();
            let pos: Vec<u32> = (base..base + r).map(|p| p as u32).collect();
            let slots = vec![slot as u32; r];
            self.ensure_rows(&slots, &pos)?;
            self.upload_rows(&toks, &pos, &pos, &pos, &slots, 0)?;
            self.embed_rows(r)?;
            // splice: overwrite image rows with plane rows, one dtod per
            // contiguous run (dst rows in sc.x and src rows in the plane
            // advance in lockstep by construction)
            let mut i = 0usize;
            while i < r {
                if let Row::Image(img, prow) = chunk[i] {
                    let mut len = 1usize;
                    while i + len < r
                        && matches!(chunk[i + len], Row::Image(i2, r2) if i2 == img && r2 == prow + len)
                    {
                        len += 1;
                    }
                    let plane = planes[img]
                        .as_ref()
                        .expect("plane exists for any image the tail touches");
                    let bs = self.batch.as_mut().expect("batch enabled");
                    exec.copy_region(plane, prow * embd, &mut bs.sc.x, i * embd, len * embd)?;
                    i += len;
                } else {
                    i += 1;
                }
            }
            // triage twin of rows_pass_body's PADDOCK_OCR_CHUNK_DUMP print
            let dump = paddock_models::dev_var_os!("PADDOCK_OCR_CHUNK_DUMP").is_some();
            if dump {
                let h = self.dump_hash_x(r)?;
                eprintln!("ocr walk-pass r={r} pos0={base} xin={h:016x}");
            }
            self.layer_walk(
                r,
                Some(&PfCuts {
                    dec: 0,
                    runs: vec![(0, r)],
                }),
            )?;
            if dump {
                let h = self.dump_hash_x(r)?;
                let b0 = self.dump_hash_blk0()?;
                eprintln!("ocr walk-pass r={r} pos0={base} xout={h:016x} blk0={b0:016x}");
            }
            base += r;
            last_len = r;
        }

        self.prefix_insert(slot, &plan.keys);
        let ring = self.ring_base(slot, plan.n_rows);
        self.batch.as_mut().expect("batch enabled").ring[slot].prefill_len = Some(ring);
        self.dash_head_guard(slot)?;
        let logits = self.head_row(last_len - 1)?;
        Ok((logits, plan.n_rows))
    }

    /// Encode one image's prepped planes and gather its spliced plane
    /// `[image_tokens, n_embd]` in block order. `need_from_row` is the first
    /// plane row the caller will read: sources that end before it (crops
    /// first, then the global view) skip their tower run entirely, leaving
    /// never-read rows behind (stale staging from an earlier image - harmless
    /// by the same argument that lets them skip) - the resumed-request
    /// economy.
    ///
    /// `crops`/`global` are the wave's off-thread host work; `global` is a
    /// DEFERRED fetch called only after the crops encode has launched, so
    /// the worker preps the global view while the GPU chews the crops (the
    /// overlap the old inline path had). Whatever the real resume needs
    /// beyond what arrives (the probe under-predicted, or eviction shrank
    /// the match between probe and resume) is built inline - the pipeline
    /// is an accelerator, never a correctness dependency.
    #[allow(clippy::too_many_arguments)]
    fn assemble_image_plane(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        crops: Vec<(usize, usize, Vec<u8>)>,
        global: impl FnOnce() -> Option<Vec<u8>>,
        blocks: &[Block],
        grid: Option<super::tiling::Grid>,
        mode: Mode,
        need_from_row: usize,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        // the skip decision from the real resume (see prep_need)
        let (need_crops, need_global) = prep_need(blocks, grid, need_from_row);
        let spec = self.prep_spec(mode, grid);
        let (base_px, tile_px) = mode.px();
        let crops = if need_crops && crops.is_empty() {
            prep_image(
                rgb,
                w,
                h,
                &PrepSpec {
                    need_crops: true,
                    ..spec
                },
            )
            .crops
        } else {
            crops
        };
        if need_crops {
            for (c0, chunk_tiles, pr) in &crops {
                self.stage_crops_chunk(pr, *c0, *chunk_tiles, tile_px)?;
            }
        }
        if need_global {
            // fetched here - after the crops encode launched - so a prep
            // worker's global runs under the crops' GPU time
            let pr = global()
                .or_else(|| {
                    prep_image(
                        rgb,
                        w,
                        h,
                        &PrepSpec {
                            need_global: true,
                            ..spec
                        },
                    )
                    .global
                })
                .expect("need_global spec yields a plane");
            self.stage_global(&pr, base_px, tile_px, grid)?;
        }
        self.finish_image_plane(blocks, grid, mode)
    }

    /// Encode one ≤[`MAX_VIEWS`]-view crops chunk and land it in the staging
    /// slab at its tile offset. One tower call - the chunked lane's
    /// encode-budget unit (~25-50 ms).
    ///
    /// Staging: `[crop rows][global rows][newline][separator]` in the
    /// persistent slab attach_vision sized at the tile budget's worst case
    /// (mid-serve allocs are the door rung C closed). The slab holds one
    /// image's views at a time; the chunked lane serializes entries, so rows
    /// staged across ticks stay whose they are until that image's gather.
    pub(super) fn stage_crops_chunk(
        &mut self,
        pr: &[u8],
        c0: usize,
        chunk_tiles: usize,
        tile_px: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let vis = self.vision.as_mut().expect("vision attached");
        let t_tile = vis.hp.tokens_per_side(tile_px);
        let enc = vis.encode(super::encode::PatchSrc::U8(pr), chunk_tiles, tile_px)?;
        let src = self
            .mm_src
            .as_mut()
            .expect("attach_vision allocated splice staging");
        assert!(
            (c0 + chunk_tiles) * t_tile * t_tile * embd <= src.len(),
            "splice staging: crop chunk at {c0}+{chunk_tiles} over the attach-time envelope",
        );
        exec.copy_region(
            &enc.embd,
            0,
            src,
            c0 * t_tile * t_tile * embd,
            chunk_tiles * t_tile * t_tile * embd,
        )?;
        self.mm_tower_views += chunk_tiles as u64;
        Ok(())
    }

    /// Encode the padded global view (one tower call) into the staging slab
    /// after the crop rows.
    pub(super) fn stage_global(
        &mut self,
        pr: &[u8],
        base_px: usize,
        tile_px: usize,
        grid: Option<super::tiling::Grid>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let vis = self.vision.as_mut().expect("vision attached");
        let t_tile = vis.hp.tokens_per_side(tile_px);
        let t_base = vis.hp.tokens_per_side(base_px);
        let crop_rows = grid.map_or(0, |g| g.tiles()) * t_tile * t_tile;
        let global_rows = t_base * t_base;
        let enc = vis.encode(super::encode::PatchSrc::U8(pr), 1, base_px)?;
        let src = self
            .mm_src
            .as_mut()
            .expect("attach_vision allocated splice staging");
        assert!(
            (crop_rows + global_rows) * embd <= src.len(),
            "splice staging: global view over the attach-time envelope",
        );
        exec.copy_region(&enc.embd, 0, src, crop_rows * embd, global_rows * embd)?;
        self.mm_tower_views += 1;
        Ok(())
    }

    /// Close one image's plane: the learned newline/separator rows plus the
    /// block-order gather over whatever the stage calls landed. Launch-only -
    /// the chunked lane runs it in the same tick as the image's last encode.
    pub(super) fn finish_image_plane(
        &mut self,
        blocks: &[Block],
        grid: Option<super::tiling::Grid>,
        mode: Mode,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let vis = self.vision.as_mut().expect("vision attached");
        let (base_px, tile_px) = mode.px();
        let t_tile = vis.hp.tokens_per_side(tile_px);
        let t_base = vis.hp.tokens_per_side(base_px);
        let crop_rows = grid.map_or(0, |g| g.tiles()) * t_tile * t_tile;
        let global_rows = t_base * t_base;
        let src = self
            .mm_src
            .as_mut()
            .expect("attach_vision allocated splice staging");
        assert!(
            (crop_rows + global_rows + 2) * embd <= src.len(),
            "splice staging: {} views over the attach-time envelope",
            crop_rows + global_rows,
        );
        exec.copy_region(
            &vis.image_newline,
            0,
            src,
            (crop_rows + global_rows) * embd,
            embd,
        )?;
        exec.copy_region(
            &vis.view_separator,
            0,
            src,
            (crop_rows + global_rows + 1) * embd,
            embd,
        )?;

        // ── the gather: [`Layout::blocks`] walked in splice order. The Local
        // block stitches the tile grid - token (R, C) of the stitched rows
        // comes from tile (R/t, C/t) at in-tile row (R%t)*t + C%t, matching
        // the reference's view(hc, wc, t, t).permute(0, 2, 1, 3, 4) exactly.
        let nl = (crop_rows + global_rows) as u32;
        let sep = nl + 1;
        let mut idx: Vec<u32> = Vec::new();
        for b in blocks {
            match *b {
                Block::Local { rows, cols } => {
                    let gcols = grid.expect("Local block has a grid").cols;
                    for row in 0..rows {
                        for col in 0..cols {
                            let v = (row / t_tile) * gcols + col / t_tile;
                            idx.push(
                                (v * t_tile * t_tile + (row % t_tile) * t_tile + col % t_tile)
                                    as u32,
                            );
                        }
                        idx.push(nl);
                    }
                }
                Block::Global { side } => {
                    debug_assert_eq!(side, t_base);
                    for row in 0..side {
                        for col in 0..side {
                            idx.push((crop_rows + row * side + col) as u32);
                        }
                        idx.push(nl);
                    }
                }
                Block::Separator => idx.push(sep),
            }
        }
        // the index rides its slab too; only `plane` is a real allocation -
        // it escapes to the caller and lives across the chunked prefill
        let d_idx = self
            .mm_idx
            .as_mut()
            .expect("attach_vision allocated splice staging");
        assert!(
            idx.len() <= d_idx.len(),
            "splice index: {} rows over the envelope",
            idx.len()
        );
        exec.upload_u32(&idx, d_idx)?;
        let mut plane = exec.alloc(idx.len() * embd)?;
        exec.pixel_shuffle_rows(src, d_idx, &mut plane, idx.len(), 1, embd)?;
        Ok(plane)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::service::OcrCropMode;

    /// Per-image blocks the way `mm_plan` takes them.
    fn per_img(layout: &Layout) -> Vec<&[Block]> {
        layout
            .img_blocks
            .iter()
            .map(|r| &layout.blocks[r.clone()])
            .collect()
    }

    /// The geometry fold that keeps the radix honest: the same picture
    /// encoded gundam (single-image) and base (multi-page or forced) yields
    /// different KV rows, so their radix keys must differ from row 0 - before
    /// this fold they shared a 273-key prefix and a cross-mode hit would have
    /// resumed corrupt rows.
    #[test]
    fn same_image_different_geometry_never_shares_keys() {
        let gundam = Layout::plan(&[(1240, 1754)], 32, None);
        let base = Layout::plan(&[(1240, 1754)], 32, Some(OcrCropMode::Base));
        assert_ne!(
            geom_code(per_img(&gundam)[0]),
            geom_code(per_img(&base)[0]),
            "gundam and base rows differ - their key streams must too"
        );
        // ...and a page inside a multi-image request folds like forced base
        let multi = Layout::plan(&[(1240, 1754), (800, 600)], 32, None);
        assert_eq!(geom_code(per_img(&multi)[0]), geom_code(per_img(&base)[0]));
    }

    /// The deliberate exception: the gundam small-image bail emits exactly
    /// base's `[Global, Separator]` and its rows are base rows, so sharing
    /// across the modes is correct and wanted (same page cached small stays
    /// warm when a multi-page request includes it).
    #[test]
    fn small_image_bail_shares_with_base() {
        let bail = Layout::plan(&[(640, 480)], 32, None);
        assert!(!bail.tiled(), "under one tile: the bail engaged");
        let base = Layout::plan(&[(640, 480)], 32, Some(OcrCropMode::Base));
        assert_eq!(geom_code(per_img(&bail)[0]), geom_code(per_img(&base)[0]));
        // different grids must never collide either
        let g23 = Layout::plan(&[(1240, 1754)], 32, None); // 2x3
        let g11w = Layout::plan(&[(4000, 1000)], 32, None); // wide grid
        assert_ne!(g23.grids, g11w.grids);
        assert_ne!(geom_code(per_img(&g23)[0]), geom_code(per_img(&g11w)[0]));
    }
}
