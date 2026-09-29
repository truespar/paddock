//! PaddleOCR-VL multimodal serving.
//!
//! Two lanes share the [`MmPlan`] machinery here:
//!
//! - the exclusive serial path: one image prompt at a time through the
//!   serial spine (`forward_multimodal_impl` - the no-paged-kv fallback and
//!   the parity reference);
//! - the batched slot lane: `multimodal_prefill_wave` prefills whole
//!   admission waves into paged-KV slots (host preprocess pipelined on worker
//!   threads, deepseek-ocr's door-B shape), and chunked.rs rides the same
//!   plan for the stall-free encoder-budget queue.
//!
//! ## Prefix caching keys (the grid fold is load-bearing)
//!
//! Every image row carries the same `<|IMAGE_PLACEHOLDER|>` id, so radix keys
//! for image rows come from the picture's CONTENT hash
//! ([`crate::gpu_model::prefix_cache::image_key_row`], granite's scheme) -
//! with the RESOLVED grid folded in. On this family the same bytes can
//! legitimately produce different rows: the per-request pixel budget
//! (`mm_processor_kwargs`) moves `smart_resize`'s target, and with it the
//! grid, the token count, every M-RoPE position, and every KV row. The grid
//! (ny, nx) pins all of that - target dims are exactly `(ny·28, nx·28)` and
//! the resize reads only original bytes + target dims - so hash ⊕ grid is
//! the complete row identity and two budgets that resolve to the same grid
//! correctly share cache.
//!
//! Parity stance on budgets: the family resizes itself from the original
//! request pixels (preprocess.rs is gated bit-exact against the HF
//! processor), so the published [`VisionBudget`] ceiling is the LARGEST
//! budget any prompt class may ask for - Spotting's 1 605 632 - and the
//! runner's protective downscale only ever fires above that. The caller's
//! actual budget arrives as an [`MmChunk::VisionPixels`] directive and
//! defaults to the checkpoint's own 112 896 / 1 003 520.

use cudarc::driver::CudaSlice;

use crate::generator::VisionBudget;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::gpu_model::prefix_cache::image_key_row;
use crate::service::MmChunk;

use super::IMAGE_TOKEN;
use super::forward::{MmGrid, Positions, build_positions};
use super::load::GpuPaddleOcrVl;
use super::preprocess::{FACTOR, PixelBudget};

/// Spotting is served with this per-request max (the client hardcodes it);
/// every other class stays at or below the checkpoint default.
const SPOTTING_MAX_PIXELS: u64 = 1_605_632;

/// One prefill row of the interleaved stream.
#[derive(Clone, Copy)]
pub(crate) enum Row {
    Token(u32),
    /// (image index, row within that image's projector plane)
    Image(usize, usize),
}

/// FNV-1a over the raw image bytes with the dimensions folded in - four
/// independent 64-bit lanes, 32 bytes per step (the byte-serial version cost
/// deepseek-ocr 3-6 ms of admission host time per page). The key
/// is process-internal (in-memory radix only), so the exact function is free
/// to change as long as it stays deterministic within a serve.
fn content_hash(rgb: &[u8], w: usize, h: usize) -> u64 {
    const P: u64 = 0x100000001b3;
    let mut lane = [
        0xcbf29ce484222325u64,
        0x84222325cbf29ce4,
        0x9ce484222325cbf2,
        0x25cbf29ce4842223,
    ];
    let (words, tail) = rgb.as_chunks::<32>();
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
    for &b in tail {
        hash ^= b as u64;
        hash = hash.wrapping_mul(P);
    }
    hash ^ (w as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (h as u64).rotate_left(32)
}

/// Fold the RESOLVED grid into an image's radix key - the module doc's whole
/// argument (same bytes, different budget => different rows).
fn grid_code(g: MmGrid) -> u64 {
    (0x51_7c_c1_b7_27_22_0a_95u64 ^ ((g.ny as u64) << 24 | g.nx as u64))
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// One request's plan: the interleaved row stream with radix keys, the
/// resolved grids and 3-axis positions - everything derivable without
/// touching the GPU (`smart_resize` is host math), so admission can plan a
/// whole wave, probe the radix, and decide prep work before any slot runs.
pub(crate) struct MmPlan {
    pub(super) rows: Vec<Row>,
    pub(super) keys: Vec<u32>,
    pub(super) grids: Vec<MmGrid>,
    pub(super) img_start: Vec<usize>,
    pub(super) img_tokens: Vec<usize>,
    /// `build_positions` over the full id stream - the M-RoPE truth the
    /// chunked walk slices per pass, and whose `next` fixes the slot delta.
    pub(super) pos: Positions,
    pub(super) n_rows: usize,
    pub(super) budget: PixelBudget,
}

/// The (rgb, w, h) of every image chunk, in chunk order.
pub(super) fn images_of(chunks: &[MmChunk]) -> Vec<(&[u8], usize, usize)> {
    chunks
        .iter()
        .filter_map(|ch| match ch {
            MmChunk::Image { rgb, w, h } => Some((rgb.as_slice(), *w, *h)),
            _ => None,
        })
        .collect()
}

/// Resolve the request's pixel budget: the directive rides at the front of
/// the chunk list (the runner injects it before any image), missing halves
/// fall to `default`. Scanned in full before any image math so it can never
/// apply to only a suffix.
///
/// Both halves are held to the ceiling this model publishes (the one
/// `vision_budget_impl` reports). The client's numbers used to be taken as
/// given: a `max_pixels` of a few hundred megapixels grew the tower's
/// workspace to match and kept it, bounded only by the prompt fitting
/// max_ctx - memory any caller could ask for. Above the ceiling a request
/// gets the ceiling, the same answer `detail: high` gets.
pub(super) fn budget_of(
    chunks: &[MmChunk],
    default: PixelBudget,
) -> Result<PixelBudget, GpuModelError> {
    let ceiling = (SPOTTING_MAX_PIXELS as usize).max(default.max_pixels);
    let mut budget = default;
    for c in chunks {
        if let MmChunk::VisionPixels {
            min_pixels,
            max_pixels,
        } = c
        {
            if let Some(m) = min_pixels {
                budget.min_pixels = (*m as usize).min(ceiling);
            }
            if let Some(m) = max_pixels {
                budget.max_pixels = (*m as usize).min(ceiling);
            }
        }
    }
    if budget.min_pixels > budget.max_pixels {
        return Err(GpuModelError::Unsupported(format!(
            "mm_processor_kwargs: min_pixels {} exceeds max_pixels {}",
            budget.min_pixels, budget.max_pixels
        )));
    }
    Ok(budget)
}

/// Plan one request: resolve the budget, size every image on the host
/// (`smart_resize` -> grid), build the interleaved row stream, its
/// content-derived radix keys, and the 3-axis position plan.
pub(super) fn mm_plan(
    chunks: &[MmChunk],
    default_budget: PixelBudget,
) -> Result<MmPlan, GpuModelError> {
    let budget = budget_of(chunks, default_budget)?;
    let mut rows: Vec<Row> = Vec::new();
    let mut keys: Vec<u32> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut grids: Vec<MmGrid> = Vec::new();
    let mut img_start: Vec<usize> = Vec::new();
    let mut img_tokens: Vec<usize> = Vec::new();
    for ch in chunks {
        match ch {
            MmChunk::Text(t) => {
                rows.extend(t.iter().map(|&tok| Row::Token(tok)));
                keys.extend_from_slice(t);
                ids.extend_from_slice(t);
            }
            MmChunk::Image { rgb, w, h } => {
                let (th, tw) = super::preprocess::smart_resize(*h, *w, FACTOR, budget)
                    .map_err(GpuModelError::Unsupported)?;
                let g = MmGrid {
                    ny: th / FACTOR,
                    nx: tw / FACTOR,
                };
                let n = g.ny * g.nx;
                let hash = content_hash(rgb, *w, *h) ^ grid_code(g);
                let k = grids.len();
                img_start.push(rows.len());
                img_tokens.push(n);
                rows.extend((0..n).map(|r| Row::Image(k, r)));
                keys.extend((0..n).map(|r| image_key_row(hash, r)));
                ids.extend(std::iter::repeat_n(IMAGE_TOKEN, n));
                grids.push(g);
            }
            MmChunk::VisionPixels { .. } => {} // consumed by budget_of
            MmChunk::Audio { .. } => {
                return Err(GpuModelError::Unsupported(
                    "paddleocr-vl serves images, not audio - routing bug".into(),
                ));
            }
            MmChunk::OcrCrop(_) => {
                return Err(GpuModelError::Unsupported(
                    "deepseek-ocr crop directive on paddleocr-vl - routing bug".into(),
                ));
            }
        }
    }
    if grids.is_empty() {
        return Err(GpuModelError::Unsupported(
            "multimodal prefill with no image".into(),
        ));
    }
    let pos = build_positions(&ids, IMAGE_TOKEN, &grids)?;
    let n_rows = rows.len();
    Ok(MmPlan {
        rows,
        keys,
        grids,
        img_start,
        img_tokens,
        pos,
        n_rows,
        budget,
    })
}

/// Slice the plan's 3-axis positions for rows [base, base+len) into the
/// axis-major [4, len] layout the mrope kernel reads (axis 3 has a zero
/// section and is never read; it mirrors t like the serial spine's upload).
pub(super) fn plan_mrope(pos: &Positions, base: usize, len: usize) -> Vec<u32> {
    let mut m = vec![0u32; 4 * len];
    m[..len].copy_from_slice(&pos.t[base..base + len]);
    m[len..2 * len].copy_from_slice(&pos.h[base..base + len]);
    m[2 * len..3 * len].copy_from_slice(&pos.w[base..base + len]);
    m[3 * len..].copy_from_slice(&pos.t[base..base + len]);
    m
}

/// Host preprocess product for one image: the bit-exact planar f32 plane
/// `encode` consumes - detached from the model so it runs on worker threads.
pub(super) struct Prepped {
    pub(super) planar: Vec<f32>,
    pub(super) w: usize,
    pub(super) h: usize,
}

pub(super) fn prep_image(
    rgb: &[u8],
    w: usize,
    h: usize,
    budget: PixelBudget,
) -> Result<Prepped, GpuModelError> {
    let (planar, tw, th) =
        super::preprocess::preprocess_rgb(rgb, w, h, budget).map_err(GpuModelError::Unsupported)?;
    Ok(Prepped {
        planar,
        w: tw,
        h: th,
    })
}

/// Drain the prep channel until job `j` has landed, then hand it over. A
/// closed channel (worker died) yields whatever arrived - the caller's
/// inline fallback covers the rest.
fn wait_job(
    rx: &std::sync::mpsc::Receiver<(usize, Prepped)>,
    got: &mut [Option<Prepped>],
    j: usize,
) -> Option<Prepped> {
    while got[j].is_none() {
        match rx.recv() {
            Ok((i, p)) => got[i] = Some(p),
            Err(_) => break,
        }
    }
    got[j].take()
}

impl GpuPaddleOcrVl {
    pub(crate) fn vision_budget_impl(&self) -> Option<VisionBudget> {
        let v = self.vision.as_ref()?;
        // one merged decoder token covers a 28×28 source patch
        let px_per_tok = (FACTOR * FACTOR) as u64;
        let max_pixels = SPOTTING_MAX_PIXELS.max(v.budget.max_pixels as u64);
        Some(VisionBudget {
            max_pixels,
            min_pixels: v.budget.min_pixels as u64,
            max_edge: None,
            pixels_per_token: px_per_tok,
            max_tokens: (max_pixels / px_per_tok) as u32,
            min_tokens: (v.budget.min_pixels as u64 / px_per_tok) as u32,
        })
    }

    /// The checkpoint-default pixel budget (from the attached mmproj header).
    pub(super) fn default_budget(&self) -> PixelBudget {
        self.vision
            .as_ref()
            .map_or(PixelBudget::DEFAULT, |v| v.budget)
    }

    /// Exclusive serial path: encode every image at the request's
    /// budget, splice the projector rows at image-token runs, prefill through
    /// the serial spine, and hand back the last row's logits plus the row
    /// count. Decode continues via `Generator::forward` at the M-RoPE delta
    /// positions the prefill left behind.
    pub(crate) fn forward_multimodal_impl(
        &mut self,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let budget = budget_of(chunks, self.default_budget())?;
        let mut ids: Vec<u32> = Vec::new();
        let mut grids: Vec<MmGrid> = Vec::new();
        let mut encoded = Vec::new();
        for c in chunks {
            match c {
                MmChunk::Text(t) => ids.extend_from_slice(t),
                MmChunk::Image { rgb, w, h } => {
                    let vis = self.vision.as_mut().ok_or_else(|| {
                        GpuModelError::Unsupported(
                            "image request but no vision tower attached (--mmproj)".into(),
                        )
                    })?;
                    let (planar, tw, th) = super::preprocess::preprocess_rgb(rgb, *w, *h, budget)
                        .map_err(GpuModelError::Unsupported)?;
                    let out = vis.encode(&planar, tw, th)?;
                    ids.extend(std::iter::repeat_n(IMAGE_TOKEN, out.ny * out.nx));
                    grids.push(MmGrid {
                        ny: out.ny,
                        nx: out.nx,
                    });
                    encoded.push(out);
                }
                MmChunk::VisionPixels { .. } => {} // consumed by budget_of
                MmChunk::Audio { .. } => {
                    return Err(GpuModelError::Unsupported(
                        "paddleocr-vl serves images, not audio - routing bug".into(),
                    ));
                }
                MmChunk::OcrCrop(_) => {
                    return Err(GpuModelError::Unsupported(
                        "deepseek-ocr crop directive on paddleocr-vl - routing bug".into(),
                    ));
                }
            }
        }
        if encoded.is_empty() {
            return Err(GpuModelError::Unsupported(
                "multimodal prefill with no image".into(),
            ));
        }

        // all images' projector rows, one plane, prompt order - the same
        // in-order splice the reference does with masked_scatter
        let embd = self.hp.n_embd;
        let total: usize = encoded.iter().map(|e| e.ny * e.nx).sum();
        let mut plane = self.exec.alloc(total * embd)?;
        let mut at = 0usize;
        for e in &encoded {
            let n = e.ny * e.nx * embd;
            self.exec.copy_region(&e.embd, 0, &mut plane, at, n)?;
            at += n;
        }

        let taps = self.prefill_taps(&ids, IMAGE_TOKEN, Some(&plane), &grids, &[])?;
        Ok((taps.last_logits, ids.len()))
    }

    // ── the batched slot lane ────────────────────────────────────────

    /// Multimodal prefill into `slot` - the wave of one (warmup, the classic
    /// per-slot path).
    pub(crate) fn multimodal_prefill_slot(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let mut r = self.multimodal_prefill_wave(vec![(slot, chunks.to_vec())]);
        r.pop().expect("wave of one").1
    }

    /// A whole multimodal admission wave, pipelined (deepseek-ocr's door-B
    /// shape): plan + PROBE the radix per slot (read-only), preprocess every
    /// needed image on worker threads, then run the serial GPU phase in wave
    /// order consuming each slot's planes as they land - the first slot never
    /// waits on the last slot's host work.
    pub(crate) fn multimodal_prefill_wave(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), GpuModelError>)> {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        let mut out: Vec<(usize, Result<(Vec<f32>, usize), GpuModelError>)> =
            Vec::with_capacity(items.len());
        if self.vision.is_none() {
            return items
                .into_iter()
                .map(|(k, _)| {
                    (
                        k,
                        Err(GpuModelError::Unsupported(
                            "image request but no vision tower attached (--mmproj)".into(),
                        )),
                    )
                })
                .collect();
        }
        let default_budget = self.default_budget();

        // phase 1, serial: plan every slot and probe the radix - decides
        // what is worth preprocessing.
        let plans: Vec<(usize, Result<MmPlan, GpuModelError>)> = items
            .iter()
            .map(|(slot, chunks)| (*slot, mm_plan(chunks, default_budget)))
            .collect();
        let imgs: Vec<Vec<(&[u8], usize, usize)>> =
            items.iter().map(|(_, chunks)| images_of(chunks)).collect();
        let mut jobs: Vec<(&[u8], usize, usize, PixelBudget)> = Vec::new();
        let mut job_of: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for (wi, (_, planned)) in plans.iter().enumerate() {
            let Ok(plan) = planned else { continue };
            let probe = self.prefix_probe(&plan.keys);
            for (k, &(rgb, w, h)) in imgs[wi].iter().enumerate() {
                if plan.img_start[k] + plan.img_tokens[k] <= probe {
                    continue; // expected fully resumed: nothing to preprocess
                }
                jobs.push((rgb, w, h, plan.budget));
                job_of.insert((wi, k), jobs.len() - 1);
            }
        }

        // phases 2+3, overlapped: workers preprocess (jobs are in wave
        // order); the GPU phase consumes each slot's planes off the channel.
        let n_workers = jobs
            .len()
            .min(std::thread::available_parallelism().map_or(1, |n| n.get()));
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            let (tx, rx) = std::sync::mpsc::channel::<(usize, Prepped)>();
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
                        let (rgb, w, h, budget) = jobs_ref[i];
                        // a failed prep just falls back to the inline path (which
                        // reports the real error); a dead receiver means the GPU
                        // phase failed early
                        if let Ok(p) = prep_image(rgb, w, h, budget) {
                            let _ = tx.send((i, p));
                        }
                    }
                });
            }
            drop(tx);

            let mut got: Vec<Option<Prepped>> = (0..jobs.len()).map(|_| None).collect();
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
                    // towers, lazily: an image encodes only if the recomputed
                    // tail [start, n) reads any of its rows
                    let mut planes: Vec<Option<CudaSlice<f32>>> = Vec::new();
                    for (k, &(rgb, w, h)) in imgs[wi].iter().enumerate() {
                        if plan.img_start[k] + plan.img_tokens[k] <= start {
                            planes.push(None); // fully resumed: no tower at all
                            continue;
                        }
                        let prepped = match job_of
                            .get(&(wi, k))
                            .and_then(|&j| wait_job(&rx, &mut got, j))
                        {
                            Some(p) => p,
                            // probe under-predicted or the worker died:
                            // inline prep - the pipeline accelerates, never
                            // gates
                            None => prep_image(rgb, w, h, plan.budget)?,
                        };
                        planes.push(Some(self.encode_plan_image(&plan, k, &prepped)?));
                    }
                    self.mm_walk(slot, &plan, &planes, start)
                })();
                out.push((slot, res));
            }
        });
        out
    }

    /// Encode one prepped image and verify the tower agreed with the plan's
    /// grid - the plan derived it from the same `smart_resize`, so a mismatch
    /// is a real bug (silent misalignment would splice the wrong rows), not a
    /// tolerable variation.
    pub(super) fn encode_plan_image(
        &mut self,
        plan: &MmPlan,
        k: usize,
        prepped: &Prepped,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        let vis = self.vision.as_mut().expect("vision attached");
        let out = vis.encode(&prepped.planar, prepped.w, prepped.h)?;
        let g = plan.grids[k];
        if out.ny != g.ny || out.nx != g.nx {
            return Err(GpuModelError::Unsupported(format!(
                "image {k}: tower grid {}x{} disagrees with the plan's {}x{}",
                out.ny, out.nx, g.ny, g.nx
            )));
        }
        Ok(out.embd)
    }

    /// The chunked prefill walk for one planned request (wave path): rows
    /// advance at `pf_rows` per pass, image rows splice from the per-image
    /// planes, M-RoPE slices come from the plan. Publishes the prefix and
    /// fixes the slot's M-RoPE delta at completion.
    fn mm_walk(
        &mut self,
        slot: usize,
        plan: &MmPlan,
        planes: &[Option<CudaSlice<f32>>],
        start: usize,
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let mut base = start;
        let mut last_len = 0usize;
        for chunk in plan.rows[start..].chunks(pf_rows()) {
            let r = chunk.len();
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, row)| {
                    let t = match row {
                        Row::Token(t) => *t,
                        Row::Image(..) => IMAGE_TOKEN,
                    };
                    (slot as u32, (base + j) as u32, t)
                })
                .collect();
            let toks: Vec<u32> = rows.iter().map(|x| x.2).collect();
            let pos: Vec<u32> = rows.iter().map(|x| x.1).collect();
            let slots = vec![slot as u32; r];
            let mrope = plan_mrope(&plan.pos, base, r);
            self.ensure_rows(&slots, &pos)?;
            self.upload_rows(&toks, &pos, &mrope, &slots)?;
            self.embed_rows(r)?;
            // splice: overwrite image rows with plane rows, one dtod per
            // contiguous run
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
            self.layer_walk(
                r,
                Some(&super::batch::PfCuts {
                    dec: 0,
                    runs: vec![(0, r)],
                }),
            )?;
            base += r;
            last_len = r;
        }

        self.prefix_insert(slot, &plan.keys);
        let bs = self.batch.as_mut().expect("batch enabled");
        bs.mrope_delta[slot] = plan.pos.next as i64 - plan.n_rows as i64;
        let logits = self.head_row(last_len - 1)?;
        Ok((logits, plan.n_rows))
    }
}

#[cfg(test)]
mod budget_tests {
    use super::{PixelBudget, SPOTTING_MAX_PIXELS, budget_of};
    use crate::service::MmChunk;

    fn directive(min: Option<u64>, max: Option<u64>) -> Vec<MmChunk> {
        vec![MmChunk::VisionPixels {
            min_pixels: min,
            max_pixels: max,
        }]
    }

    /// A client's pixel budget is held to the model's published ceiling -
    /// it used to be taken as given, and the tower's workspace grew to match.
    #[test]
    fn a_directive_cannot_raise_the_budget_past_the_published_ceiling() {
        let default = PixelBudget::DEFAULT;
        let ceiling = (SPOTTING_MAX_PIXELS as usize).max(default.max_pixels);
        let b = budget_of(&directive(None, Some(400_000_000)), default).expect("budget");
        assert_eq!(b.max_pixels, ceiling);
        let b =
            budget_of(&directive(Some(300_000_000), Some(400_000_000)), default).expect("budget");
        assert_eq!((b.min_pixels, b.max_pixels), (ceiling, ceiling));
        // and a directive inside the ceiling is honoured as asked
        let b = budget_of(&directive(Some(100_000), Some(800_000)), default).expect("budget");
        assert_eq!((b.min_pixels, b.max_pixels), (100_000, 800_000));
    }
}
