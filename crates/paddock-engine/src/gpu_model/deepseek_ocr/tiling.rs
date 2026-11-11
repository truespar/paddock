//! Tile election and image-token accounting for the DeepSeek-OCR family.
//!
//! Pure geometry - no CUDA - but it decides the prompt length, so it is the
//! thing most able to break parity silently: get a tile count or a newline slot
//! wrong and every downstream token is offset while the output still *looks*
//! like a document parse. It is therefore validated against a number measured
//! from a reference implementation rather than against our own reasoning:
//! `prompt_tokens = 907` for an A4 test page, which [`Layout`] reproduces
//! exactly.
//!
//! Mirrors `dynamic_preprocess` / `find_closest_aspect_ratio` in the
//! checkpoint's `modeling_unlimitedocr.py`.

use crate::service::OcrCropMode;

/// Vision patch size (SAM's conv stem), and the 4x token downsample the
/// projector applies on top. Both are processor_config constants, not config.json.
const PATCH: usize = 16;
const DOWNSAMPLE: usize = 4;

/// Side length in TOKENS of one rendered view of `px` pixels.
/// `ceil((px / PATCH) / DOWNSAMPLE)` - 1024 -> 16, 640 -> 10.
pub fn num_queries(px: usize) -> usize {
    (px.div_ceil(PATCH)).div_ceil(DOWNSAMPLE)
}

/// Which rendering mode a request gets. Derived from the request shape unless
/// the request overrides it: by default one image tiles ("gundam") and several
/// read base, the references' auto policy. Unlimited-OCR-2 also tiles several
/// pages on request (its reference reads every page with its own grid); v1's
/// runner refuses that, its reference mandating non-crop for multi-image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Every image: a whole-page view at `base_px` plus up to `max_tiles`
    /// crops rendered at `tile_px`, each image with its own grid.
    Gundam {
        base_px: usize,
        tile_px: usize,
        max_tiles: usize,
    },
    /// Multi-page (and PDF): one view per page at `base_px`, no cropping.
    Base { base_px: usize },
}

impl Mode {
    /// The family default: gundam for a single image, base for several.
    /// `max_tiles` is 32 on Unlimited-OCR, 16 on Unlimited-OCR-2 and 6 on the
    /// DeepSeek-OCR pair - a per-IMAGE budget.
    ///
    /// `crop` is the request's explicit override (the reference's `crop_mode`,
    /// SGLang's `images_config`): `Base` sizes even one image base, `Gundam`
    /// tiles even several.
    pub fn elect(n_images: usize, max_tiles: usize, crop: Option<OcrCropMode>) -> Self {
        let tiled = match crop {
            Some(OcrCropMode::Base) => false,
            Some(OcrCropMode::Gundam) => true,
            None => n_images == 1,
        };
        if tiled {
            Mode::Gundam {
                base_px: 1024,
                tile_px: 640,
                max_tiles,
            }
        } else {
            Mode::Base { base_px: 1024 }
        }
    }

    /// (base_px, tile_px) for preprocessing - base mode's "tile" is the base
    /// view, which is what makes the two modes share one preprocess path.
    pub fn px(self) -> (usize, usize) {
        match self {
            Mode::Gundam {
                base_px, tile_px, ..
            } => (base_px, tile_px),
            Mode::Base { base_px } => (base_px, base_px),
        }
    }
}

/// The chosen crop grid: `cols` tiles across, `rows` down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub cols: usize,
    pub rows: usize,
}

impl Grid {
    pub fn tiles(&self) -> usize {
        self.cols * self.rows
    }
}

/// Pick the crop grid whose aspect ratio is closest to the image's.
///
/// Faithful to `find_closest_aspect_ratio`, including its tie-break: on an
/// exact tie the later candidate wins only if the source image is big enough to
/// justify the extra tiles (`area > 0.5 * tile_px^2 * tiles`). Candidates are
/// visited in ascending tile count, so without that rule the cheapest grid
/// keeps a tie - which is usually the right answer and is why a 1240x1754 page
/// lands on 2x3 rather than the identical-ratio 4x6.
///
/// One deliberate divergence, stated because it is unprovable rather than
/// hidden: the reference builds its candidates in a `set` and sorts by tile
/// count only, so ties among EQUAL tile counts are resolved by Python's set
/// iteration order, which is unspecified. We iterate `(tiles, cols, rows)`
/// ascending, which is deterministic. Any image where that changes the answer
/// is one where the reference itself is not reproducible.
pub fn elect_grid(w: usize, h: usize, tile_px: usize, min_tiles: usize, max_tiles: usize) -> Grid {
    debug_assert!(w > 0 && h > 0 && tile_px > 0);
    let aspect = w as f64 / h as f64;
    let area = (w * h) as f64;

    let mut candidates: Vec<Grid> = Vec::new();
    for cols in 1..=max_tiles {
        for rows in 1..=max_tiles {
            let t = cols * rows;
            if t >= min_tiles && t <= max_tiles {
                candidates.push(Grid { cols, rows });
            }
        }
    }
    candidates.sort_by_key(|g| (g.tiles(), g.cols, g.rows));

    let mut best = Grid { cols: 1, rows: 1 };
    let mut best_diff = f64::INFINITY;
    for g in candidates {
        let diff = (aspect - g.cols as f64 / g.rows as f64).abs();
        if diff < best_diff {
            best_diff = diff;
            best = g;
        } else if diff == best_diff && area > 0.5 * (tile_px * tile_px) as f64 * g.tiles() as f64 {
            best = g;
        }
    }
    best
}

/// How many image tokens a request occupies, and the grids it chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub mode: Mode,
    /// Each image's crop grid, in image order: `None` in base mode and for an
    /// image that takes gundam's small-image bail.
    pub grids: Vec<Option<Grid>>,
    /// Total image-id tokens the `<image>` marker(s) expand to.
    pub image_tokens: usize,
    /// Views the vision tower must actually encode (global + crops, or pages).
    pub views: usize,
    /// The token blocks in SPLICE ORDER - see [`Block`]. `image_tokens` is
    /// their sum, and the graph should walk this rather than re-deriving an
    /// order from the counts.
    pub blocks: Vec<Block>,
    /// Each image's run of `blocks`, in image order.
    pub img_blocks: Vec<std::ops::Range<usize>>,
}

/// Token run for one self-contained view of side `nq` tokens: `nq` rows of
/// `nq` tokens, each row followed by an `image_newline` slot, then one
/// `view_seperator`. 1024px -> 16*(16+1) + 1 = 273.
fn view_tokens(nq: usize) -> usize {
    nq * (nq + 1) + 1
}

/// Order the embeddings are spliced in, which is not the order the reference's
/// placeholder-id loop suggests.
///
/// `modeling_unlimitedocr.py` lays the `<image>` ids out as global-then-local
/// (lines 913-917) but concatenates the FEATURES as
/// `[local, global, view_seperator]` (line 540). Both are the same length, and
/// every placeholder carries the same id, so `masked_scatter_` simply fills
/// them in feature order - the id loop's grouping is decorative. vLLM's
/// processor builds the same `[local_tiled, global, view_seperator]`
/// (`deepseek_ocr.py:563`), so the crops really do come first.
///
/// Worth stating loudly because it is invisible to any token-COUNT check: get
/// it backwards and the prompt length still matches, the model still answers,
/// and the answer is quietly about the wrong part of the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block {
    /// The stitched crop grid, `rows` token-rows each of `cols` tokens plus a
    /// trailing `image_newline`. Gundam mode only, and it comes first.
    Local { rows: usize, cols: usize },
    /// The whole-page view, `side` rows of `side` tokens each plus a trailing
    /// `image_newline`.
    Global { side: usize },
    /// A single `view_seperator`, once per image, at the end.
    Separator,
}

impl Block {
    pub fn tokens(&self) -> usize {
        match *self {
            Block::Local { rows, cols } => rows * (cols + 1),
            Block::Global { side } => side * (side + 1),
            Block::Separator => 1,
        }
    }
}

/// The largest token run one image can occupy under a tile budget - the
/// worst electable grid's local block plus the global view and separator.
/// Exact (walked over every legal grid), for the vision budget's
/// `max_tokens`; the extreme is a 1×N strip grid, not the squarest one.
pub fn max_image_tokens(max_tiles: usize) -> usize {
    let nq_t = num_queries(640);
    let nq_b = num_queries(1024);
    let mut worst = 0usize;
    for cols in 1..=max_tiles {
        for rows in 1..=max_tiles {
            if cols * rows >= 2 && cols * rows <= max_tiles {
                let local = Block::Local {
                    rows: rows * nq_t,
                    cols: cols * nq_t,
                }
                .tokens();
                worst = worst.max(local);
            }
        }
    }
    worst + view_tokens(nq_b)
}

impl Layout {
    /// Plan a request. `sizes` is one (width, height) per input image; `crop`
    /// is the per-request override (see [`Mode::elect`]).
    pub fn plan(sizes: &[(usize, usize)], max_tiles: usize, crop: Option<OcrCropMode>) -> Layout {
        let mode = Mode::elect(sizes.len(), max_tiles, crop);
        let mut blocks: Vec<Block> = Vec::new();
        let mut grids: Vec<Option<Grid>> = Vec::with_capacity(sizes.len());
        let mut img_blocks = Vec::with_capacity(sizes.len());
        for &(w, h) in sizes {
            let start = blocks.len();
            let grid = match mode {
                Mode::Base { base_px } => {
                    // Every page is an independent view: rows + newlines + its
                    // own separator. This is what the reference's no-crop
                    // branch emits, and it is why N adjacent <image> markers
                    // are token-identical to the reference's single marker
                    // (see the family template).
                    blocks.extend([
                        Block::Global {
                            side: num_queries(base_px),
                        },
                        Block::Separator,
                    ]);
                    None
                }
                // The small-image bail, from the reference's prepare loop
                // (`if image.size[0] <= 640 and image.size[1] <= 640:
                // crop_ratio = [1, 1]`, vLLM ports it verbatim): an image that
                // already fits one tile gets no crops - just the padded global
                // view and its separator, 273 tokens. Skipping this branch
                // would tile a thumbnail into 2+ upscaled crops the reference
                // never encodes, and every downstream token would be offset.
                Mode::Gundam {
                    base_px, tile_px, ..
                } if w <= tile_px && h <= tile_px => {
                    blocks.extend([
                        Block::Global {
                            side: num_queries(base_px),
                        },
                        Block::Separator,
                    ]);
                    None
                }
                Mode::Gundam {
                    base_px,
                    tile_px,
                    max_tiles,
                } => {
                    let grid = elect_grid(w, h, tile_px, 2, max_tiles);
                    let nq_tile = num_queries(tile_px);
                    // The crops are not independent views: they are stitched
                    // into one grid of (rows*nq) token-rows by (cols*nq)
                    // token-columns, and the newline lands at the end of each
                    // row of the STITCHED grid - not per tile. Treating each
                    // crop as its own view is the easy mistake here and
                    // inflates the count by (tiles-1) rows' worth of newlines.
                    //
                    // They also come first, and there is exactly one separator
                    // an image, at its very end - not one between the two
                    // views. See [`Block`]. Several tiled pages repeat this
                    // per page, each with its own grid (the reference's
                    // per-page loop).
                    blocks.extend([
                        Block::Local {
                            rows: grid.rows * nq_tile,
                            cols: grid.cols * nq_tile,
                        },
                        Block::Global {
                            side: num_queries(base_px),
                        },
                        Block::Separator,
                    ]);
                    Some(grid)
                }
            };
            grids.push(grid);
            img_blocks.push(start..blocks.len());
        }
        Layout {
            mode,
            image_tokens: blocks.iter().map(Block::tokens).sum(),
            views: grids.iter().map(|g| 1 + g.map_or(0, |g| g.tiles())).sum(),
            grids,
            blocks,
            img_blocks,
        }
    }

    /// Crop tiles across every image.
    pub fn tiles(&self) -> usize {
        self.grids.iter().flatten().map(Grid::tiles).sum()
    }

    /// Any image is tiled.
    pub fn tiled(&self) -> bool {
        self.grids.iter().any(Option::is_some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_TILES: usize = 32; // Unlimited-OCR; the DeepSeek-OCR pair uses 6

    #[test]
    fn token_side_matches_the_reference_constants() {
        assert_eq!(num_queries(1024), 16);
        assert_eq!(num_queries(640), 10);
    }

    /// A whole page in base mode is 273 tokens - 16 rows of 16, a newline each,
    /// plus one separator. Quoted throughout, so pin it.
    #[test]
    fn one_base_view_is_273_tokens() {
        assert_eq!(view_tokens(num_queries(1024)), 273);
    }

    /// The cross-check: an A4 test page (1240x1754, @150dpi) measures
    /// `prompt_tokens = 907` on a reference implementation for
    /// `<image>document parsing.`. The text around the marker is
    /// BOS + "document parsing." = 4 tokens, so the image run must be exactly
    /// 903. This is the assertion that says our geometry agrees with the
    /// reference rather than merely with itself.
    #[test]
    fn battery_page_reproduces_the_arbiters_907() {
        let plan = Layout::plan(&[(1240, 1754)], MAX_TILES, None);
        assert_eq!(
            plan.grids,
            vec![Some(Grid { cols: 2, rows: 3 })],
            "portrait A4 elects 2 wide x 3 tall"
        );
        assert_eq!(plan.image_tokens, 903);
        assert_eq!(plan.views, 7, "1 global + 6 crops");
        assert_eq!(plan.image_tokens + 4, 907, "the reference's prompt_tokens");

        // The order matters and no count check can see it: crops first, then
        // the whole page, then one separator.
        assert_eq!(
            plan.blocks,
            vec![
                Block::Local { rows: 30, cols: 20 },
                Block::Global { side: 16 },
                Block::Separator
            ]
        );
        assert_eq!(
            plan.blocks[0].tokens(),
            630,
            "30 rows of 20 tokens + a newline each"
        );
        assert_eq!(plan.blocks[1].tokens(), 272);
        assert_eq!(
            plan.blocks.iter().map(Block::tokens).sum::<usize>(),
            plan.image_tokens
        );
    }

    /// One separator per IMAGE, at the end of that image's run - so a 3-page
    /// request has three, not one.
    #[test]
    fn every_page_gets_its_own_separator() {
        let plan = Layout::plan(&[(1240, 1754), (1240, 1754), (800, 600)], MAX_TILES, None);
        assert_eq!(
            plan.blocks
                .iter()
                .filter(|b| **b == Block::Separator)
                .count(),
            3
        );
        assert_eq!(
            plan.blocks.first(),
            Some(&Block::Global { side: 16 }),
            "no crops in base mode"
        );
    }

    /// The tie the tie-break rule exists for: 2x3 and 4x6 have the identical
    /// ratio, and an A4 page is not big enough to earn the 24-tile grid.
    #[test]
    fn ties_keep_the_cheaper_grid_unless_the_page_is_large() {
        assert_eq!(
            elect_grid(1240, 1754, 640, 2, 32),
            Grid { cols: 2, rows: 3 }
        );
        // Same aspect, but ~8x the area: now the larger grid clears
        // area > 0.5 * 640^2 * 24 and wins the tie.
        let big = elect_grid(1240 * 3, 1754 * 3, 640, 2, 32);
        assert!(
            big.tiles() > 6,
            "a much larger page should earn more tiles, got {big:?}"
        );
        assert_eq!(big.cols as f64 / big.rows as f64, 2.0 / 3.0);
    }

    #[test]
    fn multi_page_is_base_mode_and_scales_per_page() {
        let pages = [(1240, 1754), (1240, 1754), (800, 600)];
        let plan = Layout::plan(&pages, MAX_TILES, None);
        assert_eq!(
            plan.mode,
            Mode::Base { base_px: 1024 },
            "multi-image must not crop"
        );
        assert!(!plan.tiled());
        assert_eq!(plan.views, 3);
        assert_eq!(plan.image_tokens, 3 * 273);
    }

    /// The worst-case token run is the 1×32 strip grid (320 token-rows of 11),
    /// not the squarest grid - the budget's max_tokens must price it.
    #[test]
    fn max_image_tokens_is_the_strip_grid() {
        assert_eq!(max_image_tokens(32), 320 * 11 + 273);
        // battery page's 2x3 grid stays well under it
        assert!(Layout::plan(&[(1240, 1754)], 32, None).image_tokens < max_image_tokens(32));
    }

    /// An image that fits one tile takes the reference's small-image bail:
    /// global view + separator only, no crop grid at all.
    #[test]
    fn small_single_image_gets_no_crops() {
        let plan = Layout::plan(&[(640, 640)], MAX_TILES, None);
        assert!(!plan.tiled());
        assert_eq!(plan.views, 1);
        assert_eq!(plan.image_tokens, 273);
        assert_eq!(
            plan.blocks,
            vec![Block::Global { side: 16 }, Block::Separator]
        );
        // one pixel over the bail on either axis and cropping engages
        assert!(Layout::plan(&[(641, 640)], MAX_TILES, None).tiled());
        assert!(Layout::plan(&[(640, 641)], MAX_TILES, None).tiled());
    }

    /// The crop override: forced base on a single image is exactly the
    /// multi-page layout of one page - one padded global view + separator,
    /// 273 tokens, no grid - which is also the reference's
    /// `crop_mode=False` single-image config.
    #[test]
    fn forced_base_single_image_is_one_page() {
        let forced = Layout::plan(&[(1240, 1754)], MAX_TILES, Some(OcrCropMode::Base));
        assert_eq!(forced.mode, Mode::Base { base_px: 1024 });
        assert!(!forced.tiled());
        assert_eq!(forced.views, 1);
        assert_eq!(forced.image_tokens, 273);
        assert_eq!(
            forced.blocks,
            vec![Block::Global { side: 16 }, Block::Separator]
        );
        // forced base on a multi-image request changes nothing - it already is
        let pages = [(1240, 1754), (800, 600)];
        assert_eq!(
            Layout::plan(&pages, MAX_TILES, Some(OcrCropMode::Base)),
            Layout::plan(&pages, MAX_TILES, None)
        );
    }

    /// The tile budget is the family's discriminator (32 vs 6) and must bind.
    #[test]
    fn tile_budget_binds() {
        let wide = (4000, 1000); // 4:1, wants a lot of columns
        let g32 = elect_grid(wide.0, wide.1, 640, 2, 32);
        let g6 = elect_grid(wide.0, wide.1, 640, 2, 6);
        assert!(g32.tiles() <= 32 && g6.tiles() <= 6);
        assert!(g6.tiles() <= g32.tiles());
    }

    /// Several pages tiled on request (Unlimited-OCR-2): every page its own
    /// grid and its own [Local, Global, Separator] run, a page that fits one
    /// tile taking the bail - the reference's per-page loop.
    #[test]
    fn forced_gundam_tiles_every_page_with_its_own_grid() {
        let pages = [(1448, 2048), (2048, 1448), (600, 580)];
        let plan = Layout::plan(&pages, 16, Some(OcrCropMode::Gundam));
        assert_eq!(
            plan.grids,
            vec![
                Some(Grid { cols: 2, rows: 3 }),
                // 4x3 (4:3) sits closer to 2048:1448 than 3x2 does - the
                // reference's dynamic_preprocess says the same
                Some(Grid { cols: 4, rows: 3 }),
                None
            ]
        );
        assert_eq!(plan.img_blocks, vec![0..3, 3..6, 6..8]);
        assert_eq!(
            plan.blocks,
            vec![
                Block::Local { rows: 30, cols: 20 },
                Block::Global { side: 16 },
                Block::Separator,
                Block::Local { rows: 30, cols: 40 },
                Block::Global { side: 16 },
                Block::Separator,
                Block::Global { side: 16 },
                Block::Separator,
            ]
        );
        // 273 + 10h(10w + 1) a tiled page, 273 for the bail
        assert_eq!(plan.image_tokens, (273 + 630) + (273 + 1230) + 273);
        assert_eq!((plan.views, plan.tiles()), (1 + 6 + 1 + 12 + 1, 18));
        // one page forced gundam is the single-image default restated
        assert_eq!(
            Layout::plan(&pages[..1], 16, Some(OcrCropMode::Gundam)),
            Layout::plan(&pages[..1], 16, None)
        );
    }
}
