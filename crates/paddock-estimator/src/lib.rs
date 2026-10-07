//! Will-it-fit estimation.
//!
//! Honesty is the product feature here (thesis point 3): predictions come with
//! their assumptions attached. The central rule is that **VRAM is two
//! different things and they must not be added together**:
//!
//! - `resident` - weights plus fixed per-slot state. This must fit, and it is
//!   what decides whether a model loads.
//! - `kv_pool` - the KV cache, a *shared paged pool* that takes whatever VRAM
//!   is left. It never decides whether a model fits; it decides how much
//!   context you get once it does.
//!
//! Conflating the two is not a rounding error, it inverts the answer. An
//! earlier version priced the dense worst case - every slot pinned at full
//! context at once - as a hard requirement, and concluded a 27B needed 107 GB
//! and a 0.6B embedding model 124 GB, while reporting "does not fit" for
//! envelopes the box in question serves every day. The engine allocates
//! `min(budget, 40% of free, dense-equivalent)` and runs 32 sequences out of
//! 16 GB.
//!
//! Two corollaries the type system now enforces: encoders
//! (`ModelKind::Encoder`) hold no cache between calls, so context and
//! concurrency cost them nothing; and fixed overhead is kind-dependent, since
//! an encoder captures no decode graph.
//!
//! Geometry comes from `paddock_models::probe`, which reads it off the GGUF
//! header (bounded prefix, never the weights), so answers are derived from the
//! actual file rather than a catalog guess. See `ModelShape::from_report`.
//!
//! What this crate deliberately does not do is re-derive the engine's exact
//! allocator behaviour - that lives in the model families and would rot in a
//! second copy.
//!
//! Math blueprint: FlexGen/MoE-Lightning closed-form byte counts (see
//! moe-*).

use serde::{Deserialize, Serialize};

/// The CUDA context, cuBLAS handles and driver-side allocations any process
/// pays once it touches the GPU at all, regardless of model or size.
pub const CUDA_CONTEXT: u64 = 512 << 20;

/// Decode-path graph capture, allocator headroom and fixed pools, on TOP of
/// the context. Mirrors the `graph_margin` the qwen35 sizer charges itself
/// before handing any VRAM to the KV pool.
///
/// This is generative-only. Charging it flat was making a 0.64 GB embedding
/// model quote 3.9 GB: an encoder runs one forward pass per call with no
/// captured decode graph and no persistent pools, so it pays the context and
/// its transient activations, nothing more.
pub const GRAPH_MARGIN: u64 = (3 << 30) - CUDA_CONTEXT;

/// Fixed overhead for a model of this kind.
fn fixed_overhead(kind: ModelKind) -> u64 {
    match kind {
        ModelKind::Encoder | ModelKind::Image => CUDA_CONTEXT,
        ModelKind::Generative => CUDA_CONTEXT + GRAPH_MARGIN,
    }
}

/// Prefill span the engine sizes its convolution scratch against, independent
/// of batch. Mirrors `unified_prefill_rows().max(8192)`.
const PREFILL_SPAN: u64 = 8192;

/// DeltaNet prefix checkpoints the engine keeps at minimum.
///
/// This constant was right and UNUSED - declared here, never added to
/// `resident` - on the reasoning that the pool "grows into spare VRAM and
/// shrinks again under pressure, so only the floor is a genuine cost". The
/// first half is true of the RUNNING pool and false of the LOAD GATE: the
/// engine reserves `n_ckpt_est = ((grant/5)/per_ckpt).clamp(16, 256)`
/// checkpoints before it hands a single byte to the KV pool, and nothing
/// shrinks it at plan time. Measured on Qwen3.8-27B: the engine
/// charged 2.34 GiB here - exactly 16 x per-checkpoint - while this page said
/// "Fits" and the server then refused to start.
const PREFIX_CKPT_FLOOR: u64 = 16;
/// ...and the ceiling, matching the engine's own clamp.
const PREFIX_CKPT_CAP: u64 = 256;
/// Checkpoint staging blobs a hybrid engine keeps beside its pool: one per
/// cut a prompt stages in a pass - the trailing pair and the back-off
/// boundary (qwen35 CKPT_STAGE_BLOBS, qwen4exp STAGED_CUTS; nemotron keeps
/// its two a slot - no back-off there). Two until the back-off cut.
const CKPT_STAGING_BLOBS: u64 = 3;
/// Checkpoints one turn writes (two prompt cuts and the reply's) - what a
/// family with paged checkpoints must back per slot, up to the floor above.
/// Mirrors qwen35's `STATE_CKPTS_PER_TURN`.
const PREFIX_CKPTS_PER_TURN: u64 = 3;
/// Divisor on the grant that the checkpoint pool may spend - the engine's
/// `STATE_CKPT_GRANT_DIV`. The pool is SELF-SIZED from what the endpoint was
/// given, not fixed at its floor, and modelling only the floor is how this
/// estimate came to under-charge a 27B by 2.5 GiB (measured: the engine
/// reserved 4.82 GiB where this said 2.34, because it had room for 33
/// checkpoints and the floor is 16). Under-charging is the dangerous
/// direction - it promises a start the runner then refuses.
const PREFIX_CKPT_GRANT_DIV: u64 = 5;

/// What the stream-ordered allocator holds beyond the sum of the planes.
///
/// A model is not one allocation: Qwen3.8-27B lands 1,228 of them, and the
/// pool rounds and pads every one. The engine measures the difference and says
/// so - "planes total 19.02 GB vs ledger-minus-ctx 20.38 GB -> allocator slack
/// 1.37 GB across 1228 allocations" - and that ledger, not the plane sum, is
/// what `vram_headroom()` subtracts from the budget. An estimate built from
/// exact tensor bytes is therefore right about the tensors and short by this
/// on the thing that actually decides whether a start fits.
///
/// One measurement (Qwen3.8-27B UD-Q4_K_XL: 1.28 GiB over 17.56
/// GiB of planes = 7.3%), rounded up to 8%. Over-stating is the safe direction
/// for a fit check for the same reason `SpecCost::default` over-states its
/// verify width: it refuses a start that would have been tight instead of
/// promising one that dies. Refine it by reading the engine's own audit line
/// on more models rather than by taste.
const ALLOCATOR_SLACK_SHARE: f64 = 0.08;

/// Live speculative slots the engine will carry state for at once. Mirrors
/// qwen35's `serve_spec_live_max()` default; the engine degrades spec live
/// before it narrows width, so this is a cap and not a multiplier on
/// concurrency.
const SPEC_LIVE_MAX: u64 = 8;

/// The engine will not hand more than this share of free VRAM to the KV pool,
/// as a backstop against any allocation its budget under-counts. It binds
/// before plain arithmetic does: on a 48 GB A6000 with 39.97 GB free it capped
/// qwen3.5-9B at 15.98 GB of KV, which is 14 full-context slots at 32768 - the
/// engine's own refusal message - where the arithmetic floor alone would have
/// allowed 23. Predicting the *server's* answer matters more than being
/// theoretically right: a page that promises an envelope startup then rejects
/// is the same lie in a new place. Mirrors the `free / 5 * 2` clamp in the
/// qwen35 pool sizer.
const KV_POOL_SHARE_CAP: f64 = 0.4;

/// Width of one cached KV element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDtype {
    /// Checkpoint-native F32 (e.g. Bonsai's F32 auxiliary tensors promote the
    /// entire Metal decoder). This is storage accounting, not a CUDA election.
    F32,
    /// Exact, and what a card without FP8 tensor cores serves whatever it was
    /// asked for - this crate has no hardware input of its own, so the CALLER
    /// resolves that (`paddock_models::gpu_support::fp8_kv`) and hands the
    /// effective width down. Getting it wrong here under-counts the KV pool by
    /// exactly 2x, which is the fit answer flipping.
    F16,
    /// E4M3 - halves the dominant term on FP8-capable cards.
    Fp8E4m3,
}

impl KvDtype {
    pub fn bytes(self) -> u64 {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
            KvDtype::Fp8E4m3 => 1,
        }
    }
}

/// One block that holds a KV cache.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KvLayer {
    pub k_dim: u64,
    pub v_dim: u64,
    /// Sliding window in tokens; `None` = full attention, which grows with
    /// context. Absent in a published block means full attention - a `0` would
    /// read as a zero-token window and price the block at nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<u64>,
}

/// Per-slot state carried by recurrent blocks - flat in context, linear in
/// concurrency (the mirror image of KV).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RecurrentShape {
    pub layers: u64,
    pub state_elems: u64,
    pub conv_elems: u64,
    pub conv_dim: u64,
    pub elem_bytes: u64,
    /// The engine keeps this family's prefix checkpoints in the KV pool's own
    /// pages (issue #33; the qwen35 and nemotron families). Then only the live turns'
    /// checkpoints are a load-time cost, and the rest is bought from what the
    /// grant has left after the context - never out of it. Families on a
    /// fixed checkpoint pool keep the floor-plus-pool pricing.
    #[serde(default)]
    pub paged_checkpoints: bool,
}

/// How paddock serves a model - which decides whether a KV cache exists at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// Autoregressive decode: holds a growing KV cache across steps.
    Generative,
    /// Embeddings and rerankers. One forward pass per input, nothing cached
    /// between calls - the server routes these through a separate encoder path
    /// (`serving::is_encoder_arch`). Pricing them with a decode cache is how a
    /// 0.6B embedding model came out "needing" 124 GB.
    Encoder,
    /// Image generation: a diffusion transformer with its text encoder and
    /// VAE beside it. One render per request and nothing kept between them,
    /// so it prices like an encoder - weights, companions, workspace, no KV
    /// pool, no logits - but it is its own kind because its "context" is a
    /// picture: the workspace is sized by the largest output the endpoint
    /// serves, not by a token window, and the Studio must not draw it a
    /// context slider.
    Image,
}

/// Static cross-attention K/V an encoder-decoder holds per slot (whisper).
///
/// Sized by the ENCODER WINDOW, not by the request: whisper's audio window is
/// a constant 30 s = 1500 frames whether the clip is four seconds or thirty,
/// so this is the same for every request and never shrinks. That makes it
/// behave exactly like recurrent state - fixed per sequence, independent of
/// context, paid per concurrent slot - and it is charged alongside it below.
///
/// It is the DOMINANT per-slot term where it exists: 234 MiB of a slot's 304
/// at f16 on whisper-large-v3, so a will-it-fit that ignored it would be
/// wrong by ~4x at any real concurrency.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CrossKv {
    pub layers: u64,
    pub frames: u64,
    pub k_dim: u64,
    pub v_dim: u64,
}

impl CrossKv {
    /// Bytes one slot holds, at the serving KV dtype. Whisper's cross planes
    /// ride the same `--kv-cache-dtype` election as its self-attention, so
    /// this must not be hard-coded to f16.
    pub fn bytes_per_slot(&self, kv: KvDtype) -> u64 {
        self.layers * self.frames * (self.k_dim + self.v_dim) * kv.bytes()
    }
}

/// Everything about a model that bears on its memory footprint.
#[derive(Debug, Clone)]
pub struct ModelShape {
    /// Resident weight bytes of the WEIGHTS file alone. Companions get their
    /// own terms (`tower_bytes` here, the drafter in `SpecCost`) so each one
    /// can be reported as its own line rather than disappearing into a lump.
    pub weight_bytes: u64,
    /// The mmproj tower - VISION or AUDIO - resident for the endpoint's whole
    /// life whenever one is wired: the engine loads it at startup, not on the
    /// first image or the first clip. That makes it unlike the drafter, which
    /// has a toggle; a model with a tower always pays it, so the estimate
    /// always has to charge it. 0 for a text-only composition.
    ///
    /// It is deliberately not folded into `weight_bytes`: the picker shows a
    /// per-artifact file size next to this number, and a "weights" figure that
    /// silently exceeded the artifact's own bytes would read as a bug.
    pub tower_bytes: u64,
    /// Persistent serving scratch the engine pins for this model beyond its
    /// weights and KV - the weights artifact's `workspace` field, measured per
    /// release. Zero for most models, where `GRAPH_MARGIN` already covers the
    /// odds and ends; gemma-4-26B-A4B's MoE expert scratch alone is 5.79 GiB
    /// at the default 32-slot width (engine `scratch_mem` self-report)
    /// - double the whole margin, so a fit check that skips it
    ///   says "fits" about a start that doesn't. Kept out of `weight_bytes` for
    ///   the same reporting honesty as the tower.
    pub workspace_bytes: u64,
    pub kind: ModelKind,
    pub kv_layers: Vec<KvLayer>,
    /// Additional context-sized KV capacity reserved by the backend (e.g.
    /// one prefix-cache sequence beyond the active Metal GPT-OSS slots).
    /// Zero preserves the original CUDA planner's pool policy. This is not
    /// additional concurrency and must not multiply logits or recurrent state.
    pub kv_reserve_sequences: u64,
    pub vocab: u64,
    pub recurrent: Option<RecurrentShape>,
    /// Static per-slot cross-attention cache, for encoder-decoders. See
    /// [`CrossKv`] - where it exists it is most of what a slot costs.
    pub cross_kv: Option<CrossKv>,
    /// The window the model was trained for, from the GGUF header. A hard
    /// ceiling on anything this crate reports: no amount of spare VRAM buys
    /// context the weights can't address.
    pub max_ctx: u64,
    /// Bytes of in-file MTP/nextn blocks, already counted inside
    /// `weight_bytes`. The engine loads them only when speculating, so they
    /// come back off when it is not - this is what makes the spec toggle move
    /// the number for a model whose drafter ships inside its weights file.
    pub nextn_bytes: u64,
}

/// The half of [`ModelShape`] that is a property of the WEIGHTS FILE, published
/// in the registry so will-it-fit runs identically before and after download
///
/// Why only half: `ModelShape` is assembled from three sources - the file's own
/// geometry, the composition the caller is pricing (which tower), and per-release
/// measurements (`workspace`). Only the first is intrinsic to the artifact, so
/// only the first can be published. `tower_bytes` still comes from whichever
/// vision artifact the caller wires, `workspace_bytes` from the artifact's
/// existing `workspace` field.
///
/// `weight_bytes` is the RESIDENT number, not the file size, and that
/// distinction is the whole point. The loader repacks quants on the way to the
/// GPU, so a Q4_K file is ~13.7% larger in VRAM than on disk  - and
/// both the estimator's `total_size` and the picker's `bytes * 1.05` were wrong
/// by exactly that. Q8_0 happens to repack 1:1 (`RepackedQ8` holds 32 data + 2
/// scale bytes per 32 elements, the same 34 the GGUF block does), which is why
/// the error hid: most of the catalog is Q8_0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedShape {
    /// Resident weight bytes on the GPU - post-repack, not the file size.
    pub weight_bytes: u64,
    pub kind: ModelKind,
    /// The KV blocks as runs, not one entry per block. A published shape is
    /// read by people as well as by the estimator, and "46 recurrent blocks and
    /// 6 that page a 256/256 cache" is both shorter and truer than 6 identical
    /// stanzas - a hybrid's whole point is that its blocks come in kinds.
    #[serde(default)]
    pub kv_layers: Vec<KvRun>,
    #[serde(default)]
    pub vocab: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrent: Option<RecurrentShape>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_kv: Option<CrossKv>,
    #[serde(default)]
    pub max_ctx: u64,
    #[serde(default)]
    pub nextn_bytes: u64,
    /// Whether `weight_bytes` was WITNESSED or computed. It ships with the
    /// numbers rather than being inferred, because the two are not equally
    /// strong and a reader deserves to know which they have.
    pub source: ShapeSource,
}

/// A run of CONSECUTIVE identical KV blocks - the published, collapsed form of
/// [`KvLayer`]. Expanded back out by [`PublishedShape::into_model_shape`], so
/// the estimator still prices block by block and nothing downstream has to know
/// the file said it once.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KvRun {
    pub k_dim: u64,
    pub v_dim: u64,
    /// See [`KvLayer::window`] - absent means full attention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<u64>,
    /// How many blocks in a row have exactly this shape.
    pub count: u64,
}

/// Where a [`PublishedShape`]'s `weight_bytes` came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeSource {
    /// The loader was run on real hardware and asked what it had allocated
    /// (`weights_mem_bytes`, the same counter `/api/stats` publishes). Exact by
    /// construction: it is not a model of the loader, it is the loader.
    Measured,
    /// Derived from the file header without loading - the fallback for an
    /// artifact no box we have can hold (gpt-oss-120b and laguna-s-2.1 are both
    /// larger than this A6000). Honest, but it cannot see what the loader does
    /// on the way to the GPU: repack growth, an f16 norm widened to f32, a tied
    /// lm_head repacked twice. Replace with a measurement the first
    /// time the artifact lands on a card that fits it.
    Probed,
}

impl PublishedShape {
    /// Build the publishable half from a probe report plus a `weight_bytes`
    /// the CALLER established - see [`ShapeSource`] for what establishing it
    /// means. Deliberately mirrors [`ModelShape::from_report`] field for field
    /// so the published block and a live probe cannot describe the same file
    /// differently.
    pub fn from_report(
        r: &paddock_models::probe::ModelReport,
        weight_bytes: u64,
        kind: ModelKind,
        source: ShapeSource,
    ) -> Self {
        let m = ModelShape::from_report(r, weight_bytes, kind);
        // Collapse CONSECUTIVE equals only. Merging across a gap would lose the
        // interleave order, and order is load-bearing for a hybrid: the runner
        // maps block index to cache kind, so "6 full then 46 recurrent" is a
        // different model from one that alternates.
        let mut kv_layers: Vec<KvRun> = Vec::new();
        for l in &m.kv_layers {
            match kv_layers.last_mut() {
                Some(run)
                    if run.k_dim == l.k_dim && run.v_dim == l.v_dim && run.window == l.window =>
                {
                    run.count += 1;
                }
                _ => kv_layers.push(KvRun {
                    k_dim: l.k_dim,
                    v_dim: l.v_dim,
                    window: l.window,
                    count: 1,
                }),
            }
        }
        Self {
            weight_bytes: m.weight_bytes,
            kind: m.kind,
            kv_layers,
            vocab: m.vocab,
            recurrent: m.recurrent,
            cross_kv: m.cross_kv,
            max_ctx: m.max_ctx,
            nextn_bytes: m.nextn_bytes,
            source,
        }
    }

    /// Complete the shape with the two terms only the CALLER knows: which tower
    /// this composition wires, and the artifact's measured serving workspace.
    pub fn into_model_shape(self, tower_bytes: u64, workspace_bytes: u64) -> ModelShape {
        ModelShape {
            weight_bytes: self.weight_bytes,
            tower_bytes,
            workspace_bytes,
            kind: self.kind,
            kv_reserve_sequences: 0,
            kv_layers: self
                .kv_layers
                .iter()
                .flat_map(|r| {
                    std::iter::repeat_n(
                        KvLayer {
                            k_dim: r.k_dim,
                            v_dim: r.v_dim,
                            window: r.window,
                        },
                        r.count as usize,
                    )
                })
                .collect(),
            vocab: self.vocab,
            recurrent: self.recurrent,
            cross_kv: self.cross_kv,
            max_ctx: self.max_ctx,
            nextn_bytes: self.nextn_bytes,
        }
    }
}

impl ModelShape {
    /// Build from a probe report. `weight_bytes` is a separate argument
    /// because the report describes one file, while the caller decides which
    /// companions (mmproj, MTP drafter) are actually being served.
    ///
    /// `tower_bytes` starts at 0 for the same reason and is set by the caller
    /// - the probe of a weights GGUF cannot see whether an mmproj sits beside
    ///   it, and only the caller knows which composition is being priced.
    pub fn from_report(
        r: &paddock_models::probe::ModelReport,
        weight_bytes: u64,
        kind: ModelKind,
    ) -> Self {
        Self {
            weight_bytes,
            tower_bytes: 0,
            workspace_bytes: 0,
            kind,
            kv_reserve_sequences: 0,
            kv_layers: r
                .kv_layers
                .iter()
                .map(|l| KvLayer {
                    k_dim: l.k_dim,
                    v_dim: l.v_dim,
                    window: l.window,
                })
                .collect(),
            vocab: r.token_count.unwrap_or(0),
            recurrent: r.recurrent.as_ref().map(|s| RecurrentShape {
                layers: s.layers,
                state_elems: s.state_elems,
                conv_elems: s.conv_elems,
                conv_dim: s.conv_dim,
                elem_bytes: s.elem_bytes,
                // the archs whose engine family keeps its checkpoints in
                // pool pages: GpuQwen35, GpuNemotron and Qwen4ExpGpu
                paged_checkpoints: matches!(
                    r.architecture.as_deref(),
                    Some(
                        "qwen35"
                            | "qwen35moe"
                            | "nemotron_h_moe"
                            | "nemotron_h"
                            | "nemotron-h"
                            | "qwen4exp"
                    )
                ),
            }),
            cross_kv: r.cross_kv.as_ref().map(|c| CrossKv {
                layers: c.layers,
                frames: c.frames,
                k_dim: c.k_dim,
                v_dim: c.v_dim,
            }),
            max_ctx: r.context_length.unwrap_or(0),
            nextn_bytes: r.nextn_bytes,
        }
    }

    /// KV bytes for one sequence at `ctx` tokens. Sliding-window blocks stop
    /// growing at their window, which is why a gemma4- or gpt-oss-shaped model
    /// costs far less at long context than its block count suggests.
    pub fn kv_per_sequence(&self, ctx: u64, kv: KvDtype) -> u64 {
        self.kv_layers
            .iter()
            .map(|l| (l.k_dim + l.v_dim) * kv.bytes() * l.window.map_or(ctx, |w| w.min(ctx)))
            .sum()
    }

    /// The longest context (up to the trained window) whose KV for ONE
    /// sequence fits in `bytes` - the inverse of `kv_per_sequence`. That cost
    /// is piecewise linear, flattening as each windowed layer reaches its
    /// window, so dividing by the one-token cost (which charges every windowed
    /// layer as if it kept growing) undercounts any windowed model: gemma4's
    /// full 262K pool bought "24752" that way. Monotone, so bisect.
    pub fn ctx_for_kv_bytes(&self, bytes: u64, kv: KvDtype) -> u64 {
        let top = self.max_ctx;
        if self.kv_per_sequence(top, kv) <= bytes {
            return top;
        }
        // invariant: lo fits, hi does not
        let (mut lo, mut hi) = (0, top);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.kv_per_sequence(mid, kv) <= bytes {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// One token's share of a full-window sequence: the slope a client may
    /// multiply by context x sessions. Exactly the one-token cost for an
    /// unwindowed model; for a windowed one, the average at the trained
    /// window, so the bounded layers stop inflating it. Rounded DOWN: Studio
    /// compares slope x window x sessions against the pool, which is sized to
    /// the exact cost, and rounding up made a pool that holds every session
    /// read as one short.
    pub fn kv_bytes_per_token(&self, kv: KvDtype) -> u64 {
        match self.max_ctx {
            0 => self.kv_per_sequence(1, kv),
            top => match self.kv_per_sequence(top, kv) {
                0 => 0,
                total => (total / top).max(1),
            },
        }
    }
}

/// What the user is asking the server to serve.
///
/// Context is deliberately not an input. It is a *derived capability*: every
/// model carries its own trained ceiling and the card backs some fraction of
/// it, so asking a user to pick a context from a fixed list is both arbitrary
/// and wrong at the edges - it offered 131072 to models capped at 32768 while
/// hiding qwen3.5-9B's real 262144. Concurrency is the genuine choice; context
/// is the answer.
#[derive(Debug, Clone, Copy)]
pub struct Envelope {
    /// Sequences served at the same time.
    pub concurrency: u64,
    pub kv_dtype: KvDtype,
    /// What speculative decode costs this endpoint in resident VRAM, when it is
    /// switched on. `None` = not speculating.
    ///
    /// It has to be part of the envelope, not an afterthought: turning
    /// speculation on loads a drafter and widens the verify round, and a
    /// will-it-fit that ignored both would quietly promise a context the card
    /// cannot actually back once the endpoint starts.
    pub spec: Option<SpecCost>,
    /// What prefix-cache offload costs this endpoint, when it is switched on.
    /// `None` = no tier.
    ///
    /// Same reasoning as `spec`: arming the tier takes device staging out of
    /// the KV pool before a single token is served, so an estimate that
    /// ignored it would promise a context the runner then quietly seats
    /// smaller. It also commits host RAM, which is not VRAM and must never be
    /// added to a VRAM total - it rides through to its own line.
    pub offload: Option<OffloadCost>,
}

/// The two resources an armed tier holds. Deliberately separate, because they
/// are different resources: one competes with the KV pool, the other with the
/// rest of the machine.
#[derive(Debug, Clone, Copy)]
pub struct OffloadCost {
    /// Device staging extents - VRAM, reserved for as long as the tier is
    /// armed. Comes from `paddock_models::kv_tier_geom` so the engine's
    /// reserve and this subtraction are literally the same number.
    pub device_staging_bytes: u64,
    /// The host-RAM CEILING the operator set (`[kv_offload] ram_gb`). Not
    /// VRAM, and not allocated up front - the tier grows into it in 1 GiB
    /// slabs - but it is a real commitment at steady state, so a fit surface
    /// that never mentions it is hiding the price of the feature.
    pub host_ram_bytes: u64,
}

impl OffloadCost {
    /// The cost of an armed tier at `ram_gb` of host budget.
    pub fn armed(host_ram_bytes: u64) -> Self {
        Self {
            device_staging_bytes: paddock_models::kv_tier_geom::device_staging_bytes(),
            host_ram_bytes,
        }
    }
}

/// The two resident terms speculation adds. Both are knowable up front, which
/// is why they belong here rather than in a fudge factor.
#[derive(Debug, Clone, Copy)]
pub struct SpecCost {
    /// Drafter weights held on device. Zero for in-file MTP (qwen3.5/3.6
    /// `nextn`): those tensors are inside the weights file and are already
    /// counted in `weight_bytes` - adding them again would charge twice.
    pub drafter_bytes: u64,
    /// Logits rows per slot in one verify round: 1 pending + K drafts. The
    /// batched logits plane is held for the whole round, so this multiplies it
    /// directly - gemma4 at c32 measured 268 MB (32 slots x 8 rows x 262144
    /// vocab x 4 B), which is the term this reproduces.
    pub verify_rows_per_slot: u64,
}

impl Default for SpecCost {
    fn default() -> Self {
        // 8 rows/slot is gemma4's measured verify width and the widest we
        // currently allocate for; using it for every family over-states the
        // narrower ones, and over-stating is the safe direction for a fit
        // check - it refuses a start that would have been tight, rather than
        // promising one that OOMs.
        Self {
            drafter_bytes: 0,
            verify_rows_per_slot: 8,
        }
    }
}

/// The card, as the engine sees it at load time.
#[derive(Debug, Clone, Copy)]
pub struct Device {
    /// Free VRAM - what the engine actually gets to size against. Measuring
    /// against *total* is the classic over-promise: a desktop session can hold
    /// several GB before paddock starts.
    pub free_bytes: u64,
    pub total_bytes: u64,
}

/// What a model costs, split into the part that must fit and the part that
/// simply uses whatever is left.
///
/// The split is the whole point. An earlier version of this crate priced the
/// DENSE worst case - every slot pinned at full context simultaneously - as a
/// hard requirement, and concluded that a 27B needs 107 GB and a 0.6B embedding
/// model 124 GB. Neither is true. The engine allocates a *shared paged pool*
/// sized to what's available (`min(budget, 40% of free, dense-equivalent)`) and
/// runs 32 sequences out of 16 GB. So the cache does not decide whether a model
/// fits; it decides how much context you get once it does.
#[derive(Debug, Clone, Serialize)]
pub struct Estimate {
    pub weights: u64,
    /// The mmproj tower (vision OR audio), when one is served. Its own line
    /// because it is weights, not overhead - reporting 1.1 GB of mmproj as
    /// "engine overhead" would be a true total with a false story.
    pub tower: u64,
    /// Persistent serving scratch declared by the weights artifact (MoE
    /// expert staging and the like), measured per release. Its own line for
    /// the same reason as the tower: folding gigabytes of pinned scratch into
    /// "overhead" would be a true total with a false story.
    pub workspace: u64,
    /// Per-slot state that is FLAT in context: recurrent/DeltaNet state, and
    /// an encoder-decoder's static cross-attention cache (whisper's is 234 MiB
    /// a slot at f16, most of what a slot costs).
    pub state: u64,
    /// Batched logits + block tables + span-sized convolution scratch, plus
    /// the prefix-cache tier's device staging when one is armed.
    pub overhead: u64,
    /// `overhead` and `fixed`, term by term.
    ///
    /// One number called "engine overhead" is unreadable at this scale: a
    /// 27B reports ~4.6 GiB of it with speculation off and ~13 with it on,
    /// and a reader with no split cannot tell a measured allocator tax from
    /// a per-slot reservation they could shrink by lowering concurrency
    /// ("how can a 17GB model have 11GB engine overhead -
    /// that just sounds weird"). Every field here is already computed; they
    /// were simply summed away before anyone could see them.
    pub overhead_parts: OverheadParts,
    /// Host RAM the prefix cache may hold - a CEILING, grown into lazily, and
    /// not part of any VRAM figure on this struct. Zero when no tier is
    /// armed. Its own field precisely so it can never be summed into a
    /// device total by accident.
    pub host_ram: u64,
    /// The floor that has to fit before anything can be served.
    pub resident: u64,
    /// What the KV pool would actually be given here - elastic, capped at what
    /// the model's own ceiling could use and at the engine's share cap. 0 for
    /// encoders, which cache nothing between calls.
    pub kv_pool: u64,
    /// The headline answer: the longest context this card can serve for this
    /// model, at this concurrency. Never exceeds the model's trained window.
    pub max_ctx: u64,
    /// The model's trained window, so a caller can say whether `max_ctx` is
    /// the model's limit or the card's.
    pub model_max_ctx: u64,
    /// Which ceiling bit.
    pub limited_by: LimitedBy,
    /// Per-token KV cost for one sequence, so a caller can plot the curve.
    pub kv_bytes_per_token: u64,
    pub fit: Fit,
}

/// `overhead` split into the terms that produce it, plus the fixed floor
/// that sits outside it. Bytes.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct OverheadParts {
    /// The allocator's rounding and padding across every resident plane - a
    /// MEASURED share, not a guess (see `ALLOCATOR_SLACK_SHARE`).
    pub allocator_slack: u64,
    /// Hybrid prefix-cache checkpoints: the floor a recurrent model needs
    /// before it can cache prefixes at all.
    pub prefix_checkpoints: u64,
    /// The batched logits plane, held for the whole decode step.
    pub logits: u64,
    /// One u32 per page per slot.
    pub block_tables: u64,
    /// Span-sized convolution scratch for recurrent families.
    pub conv_scratch: u64,
    /// What speculation reserves BEYOND its drafter weights: draft-row logits
    /// and, on recurrent models, per-slot state at draft depth. Zero when not
    /// speculating, and the largest single term when it is.
    pub spec_state: u64,
    /// Device staging held by an armed KV-offload tier.
    pub offload_staging: u64,
    /// The checkpoint pool above its floor. Not part of `overhead` - the
    /// engine spends it out of the grant, so it comes off the KV pool rather
    /// than the resident floor. Reported because a reader looking at a
    /// smaller-than-expected context deserves to see what took it.
    pub prefix_pool_extra: u64,
    /// CUDA context + graph margin: a FLAT floor every generative model pays,
    /// independent of its size. Outside `overhead` (it joins `resident`
    /// directly), reported here so a breakdown adds up to what the reader
    /// sees.
    pub fixed: u64,
}

/// What stopped the context going higher - the difference between "buy more
/// VRAM" and "this is all the model has".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitedBy {
    /// Running the model's full trained window; spare VRAM wouldn't add any.
    Model,
    /// The cache ran out first - fewer concurrent sessions would buy more.
    Vram,
    /// Encoders have no window to speak of.
    NotApplicable,
}

/// Does the model fit on the card at all? Judged on `resident` - the cache
/// sizes itself to what's left over.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Fit {
    /// On-device with room for a useful cache.
    Fits { headroom_bytes: u64 },
    /// Loads, but leaves so little that the cache is squeezed to near its
    /// minimum - serving will be short-context or heavily preempting.
    Tight { headroom_bytes: u64 },
    /// The weights and fixed state alone don't fit.
    DoesNotFit { short_by_bytes: u64 },
}

/// One u32 per 16-token page per slot. Sized at the model's ceiling - an upper
/// bound, and a couple of MB, which keeps `resident` free of any circular
/// dependency on the context it is used to derive.
fn block_tables(shape: &ModelShape, env: &Envelope) -> u64 {
    match shape.kind {
        ModelKind::Encoder | ModelKind::Image => 0,
        ModelKind::Generative => env.concurrency * shape.max_ctx.div_ceil(16) * 4,
    }
}

/// Batched logits, held for the decode step. An encoder emits embeddings, not
/// a distribution over the vocabulary - a reranker's yes/no scoring touches a
/// couple of rows per call and is transient, so it rides in the margin.
fn logits(shape: &ModelShape, env: &Envelope) -> u64 {
    match shape.kind {
        ModelKind::Encoder | ModelKind::Image => 0,
        // A speculative round scores 1 pending + K drafted tokens per slot in
        // one pass, so the plane is that many times wider and stays resident
        // for the round.
        ModelKind::Generative => {
            let rows = env.spec.map_or(1, |s| s.verify_rows_per_slot.max(1));
            env.concurrency * rows * shape.vocab * 4
        }
    }
}

pub fn estimate(shape: &ModelShape, env: &Envelope, dev: &Device) -> Estimate {
    // "encoder" below means the single-pass class - nothing held between
    // calls, no pool to size. An image lane is priced the same way.
    let encoder = shape.kind != ModelKind::Generative;
    let n = env.concurrency.max(1);
    let kv_sequences = n.saturating_add(shape.kv_reserve_sequences);
    // An encoder holds nothing between calls, so concurrency is the only knob
    // that touches it at all, and then only through transient activations.
    let (state, conv_scratch) = match (&shape.recurrent, encoder) {
        (Some(r), false) => (
            r.layers * (r.state_elems + r.conv_elems) * r.elem_bytes * n,
            // two span-sized buffers; sized by prefill span, not by concurrency
            2 * (PREFILL_SPAN + r.conv_elems / r.conv_dim.max(1)) * r.conv_dim * r.elem_bytes,
        ),
        _ => (0, 0),
    };
    // The encoder-decoder's static cross cache joins the same term: it is the
    // other thing that is fixed per sequence and multiplied by slots. It is
    // not part of `kv_pool` - the pool is what buys context, and no amount of
    // it buys a bigger audio window - so it belongs in the resident floor,
    // where a slot count that does not fit refuses instead of promising.
    let state = state
        + match (&shape.cross_kv, encoder) {
            (Some(c), false) => c.bytes_per_slot(env.kv_dtype) * n,
            _ => 0,
        };
    // What the engine reserves for prefix REUSE, before the KV pool exists.
    //
    // Two terms, both per-checkpoint multiples: the checkpoint pool itself at
    // its floor, and the CKPT_STAGING_BLOBS the engine keeps to move a snapshot
    // in and out. A checkpoint is one slot's worth of recurrent state, so it is
    // the same arithmetic as `state` above with the slot count replaced by the
    // pool depth. Together they were the largest single thing this estimate did
    // not know about (2.63 of the 3.71 GiB it was short on Qwen3.8-27B).
    let per_ckpt = match (&shape.recurrent, encoder) {
        (Some(r), false) => r.layers * (r.state_elems + r.conv_elems) * r.elem_bytes,
        _ => 0,
    };
    // Paged checkpoints (issue #33) put only the live turns' pages in front of
    // the context - three a slot, up to the old floor - and each occupies
    // whole pool pages, so it is priced in pages.
    let paged_ckpts = shape
        .recurrent
        .as_ref()
        .is_some_and(|r| r.paged_checkpoints);
    let ckpt_floor = if paged_ckpts {
        (PREFIX_CKPTS_PER_TURN * n).min(PREFIX_CKPT_FLOOR)
    } else {
        PREFIX_CKPT_FLOOR
    };
    let ckpt_bytes = if paged_ckpts {
        let page = shape.kv_per_sequence(16, env.kv_dtype).max(1);
        per_ckpt.div_ceil(page) * page
    } else {
        per_ckpt
    };
    let prefix_ckpt = ckpt_floor * ckpt_bytes + CKPT_STAGING_BLOBS * per_ckpt;
    // ...and the pool above its floor, which is a different kind of number.
    //
    // Measured: the engine self-sizes this pool from the grant -
    // `((grant / 5) / per_ckpt).clamp(16, 256)` - and on the 27B at a 24.71
    // GiB grant it took 33 checkpoints for 4.82 GiB where the floor alone is
    // 2.34. Charging only the floor under-stated the endpoint by 2.5 GiB.
    //
    // But the excess is not part of the resident floor, and putting it there
    // was wrong in a way the suite caught immediately: `resident` would then
    // grow with free VRAM, so a bigger card would make a model look heavier.
    // The engine spends this out of the GRANT, where it competes with the KV
    // pool - so that is where it is charged. Load-time requirement: the
    // floor. Division of what is left: this.
    //
    // Paged, there is no such pool: the wanted checkpoints beyond the floor
    // are bought only from what the context leaves, so they take nothing.
    let prefix_pool_extra = if paged_ckpts {
        0
    } else {
        let planes_now = shape.weight_bytes + shape.tower_bytes + shape.workspace_bytes;
        let grant = dev
            .free_bytes
            .saturating_sub(planes_now)
            .saturating_sub(fixed_overhead(shape.kind));
        let n = (grant / PREFIX_CKPT_GRANT_DIV)
            .checked_div(per_ckpt.max(1))
            .unwrap_or(0)
            .clamp(PREFIX_CKPT_FLOOR, PREFIX_CKPT_CAP);
        n.saturating_sub(PREFIX_CKPT_FLOOR) * per_ckpt
    };
    // Speculation's RUNTIME state, which is not its weights: allocated LAZILY
    // on the first spec round, so it is invisible at load and fatal afterwards
    // if the pool already took the room. Three terms, mirroring the engine's
    // own `spec_est` (qwen35/batch.rs) - the dense MTP K/V plane, the draft
    // logits rows, and (for a hybrid) the recurrent state the draft chain
    // carries. Only the first was here before; see (3).
    let spec_state = match (env.spec, encoder) {
        (Some(s), false) => {
            let live = n.min(SPEC_LIVE_MAX);
            let rows = s.verify_rows_per_slot.max(1);
            // (1) the dense MTP K/V plane at full context.
            let per_layer = shape
                .kv_layers
                .first()
                .map_or(0, |l| (l.k_dim + l.v_dim) * env.kv_dtype.bytes());
            let mtp_kv = live * shape.max_ctx * per_layer;
            // (2) the draft chain's own logits rows.
            let draft_logits = live * rows * shape.vocab * 4;
            // (3) a HYBRID's drafter carries recurrent + conv state for every
            // draft row, and this term was missing entirely - the estimate
            // priced a DeltaNet model's speculation as if only (1) existed.
            // It is the same per-slot arithmetic as `state` above with the
            // draft depth in place of the slot count, which is what makes it
            // big: depth multiplies the whole recurrent width, per live spec
            // slot. Measured on Qwen3.5-9B Q8 at 4096x32 - the engine reserved
            // 2.08 GiB for it and the grant paid for none, which is most of
            // the 2.47 GiB the runner was then short of a startable KV pool.
            let draft_state = shape.recurrent.as_ref().map_or(0, |r| {
                // conv_elems is (kernel - 1) columns wide, so this recovers it
                // without RecurrentShape having to carry the kernel itself.
                let conv_hist = r.conv_elems / r.conv_dim.max(1);
                live * r.layers
                    * (rows * r.state_elems + (conv_hist + rows) * r.conv_dim)
                    * r.elem_bytes
            });
            mtp_kv + draft_logits + draft_state
        }
        _ => 0,
    };
    // Every resident PLANE pays the allocator's rounding. It joins `overhead`
    // rather than becoming a line of its own because the breakdown is a
    // REPORTED surface - `resident_parts_sum_to_the_whole` exists to stop a
    // resident term being added without somewhere for a reader to see it, and
    // the Manager's fit bars are drawn from exactly these fields.
    let planes = shape.weight_bytes + shape.tower_bytes + shape.workspace_bytes;
    let alloc_slack = (planes as f64 * ALLOCATOR_SLACK_SHARE) as u64;
    let staging = env.offload.map_or(0, |o| o.device_staging_bytes);
    let overhead = conv_scratch
        + block_tables(shape, env)
        + logits(shape, env)
        + prefix_ckpt
        + spec_state
        + alloc_slack
        // an armed prefix-cache tier reserves its staging extents out of the
        // same VRAM the pool is sized from (the engine's "kv-tier staging"
        // reserve), so the estimate must charge for them or it will draw a
        // context the runner then seats smaller
        + staging;
    // Speculation moves the WEIGHTS term two ways, and both are needed or the
    // toggle appears to do nothing:
    //   - a sideloaded drafter is extra weights, resident for the endpoint's
    //     life and therefore unavailable to the KV pool;
    //   - an IN-FILE drafter (nextn) is already inside weight_bytes because
    //     that is the file's size, but the engine skips loading those blocks
    //     when not speculating - so not speculating hands them back.
    let drafter = env.spec.map_or(0, |s| s.drafter_bytes);
    let weights = if env.spec.is_some() {
        shape.weight_bytes
    } else {
        shape.weight_bytes.saturating_sub(shape.nextn_bytes)
    };
    // The vision tower is unconditional where it is wired at all - no toggle,
    // loaded at startup - so it joins the floor with no gate of its own. The
    // declared workspace rides the same rule: the engine pins it at load.
    let resident = weights
        + shape.tower_bytes
        + shape.workspace_bytes
        + state
        + overhead
        + drafter
        + fixed_overhead(shape.kind);

    let headroom = dev.free_bytes.saturating_sub(resident);
    // Cache big enough to run every slot at the model's full window; there is
    // nothing to spend beyond that.
    let want = if encoder {
        0
    } else {
        shape.kv_per_sequence(shape.max_ctx, env.kv_dtype) * kv_sequences
    };
    let kv_pool = want
        // the self-sized checkpoint pool is spent before the KV pool gets what
        // is left - the engine's own budget line reads
        // "grant - ... - prefix state pool - ... => N GiB for KV"
        .min(headroom.saturating_sub(prefix_pool_extra))
        .min((KV_POOL_SHARE_CAP * dev.free_bytes as f64) as u64);

    // The inversion: context is what the pool BUYS, not what the user asks for.
    let per_token = shape.kv_bytes_per_token(env.kv_dtype);
    let (max_ctx, limited_by) = if encoder || per_token == 0 {
        (0, LimitedBy::NotApplicable)
    } else {
        let by_vram = shape.ctx_for_kv_bytes(kv_pool / kv_sequences, env.kv_dtype);
        if by_vram >= shape.max_ctx {
            (shape.max_ctx, LimitedBy::Model)
        } else {
            // round down to a whole 16-token page, the engine's allocation unit
            (by_vram / 16 * 16, LimitedBy::Vram)
        }
    };

    let prefix_cache_floor = if encoder {
        0
    } else {
        // Deliberately the FLOOR, not the self-sized pool: this gates the
        // "tight" verdict, and the question it asks is whether the endpoint
        // can cache prefixes at all. The pool being able to grow past its
        // minimum is not what makes an endpoint workable.
        shape
            .recurrent
            .as_ref()
            .map_or(0, |_| ckpt_floor * ckpt_bytes)
    };

    // Fit is decided by `resident` alone. A model that loads but can't reach a
    // workable window is "tight", not "too big".
    let fit = if resident > dev.free_bytes {
        Fit::DoesNotFit {
            short_by_bytes: resident - dev.free_bytes,
        }
    } else if encoder {
        Fit::Fits {
            headroom_bytes: headroom,
        }
    } else if (max_ctx < MIN_USEFUL_CTX && limited_by == LimitedBy::Vram)
        || headroom < prefix_cache_floor
    {
        Fit::Tight {
            headroom_bytes: headroom,
        }
    } else {
        Fit::Fits {
            headroom_bytes: headroom,
        }
    };

    Estimate {
        weights: shape.weight_bytes,
        tower: shape.tower_bytes,
        workspace: shape.workspace_bytes,
        state,
        overhead,
        overhead_parts: OverheadParts {
            allocator_slack: alloc_slack,
            prefix_checkpoints: prefix_ckpt,
            logits: logits(shape, env),
            block_tables: block_tables(shape, env),
            conv_scratch,
            spec_state,
            offload_staging: staging,
            prefix_pool_extra,
            fixed: fixed_overhead(shape.kind),
        },
        host_ram: env.offload.map_or(0, |o| o.host_ram_bytes),
        resident,
        kv_pool,
        max_ctx,
        model_max_ctx: shape.max_ctx,
        limited_by,
        kv_bytes_per_token: per_token,
        fit,
    }
}

/// Below this window the model technically loads but isn't worth serving. The
/// engine floors its own pool at 128 blocks of 16 tokens per slot.
///
/// It is a verdict on the VRAM, never on the model: a short window is only
/// "tight" when the CARD is what cut it short. A model whose own trained
/// ceiling is below this is being served in full and fits - whisper's
/// transcript window is 448 tokens by construction (30 s of audio does not
/// produce more), and reporting that as a cramped endpoint would have been an
/// honest number wearing the wrong label. Guarded by `LimitedBy::Vram` at the
/// use site for exactly that reason.
const MIN_USEFUL_CTX: u64 = 2048;

/// The context/concurrency trade-off, for callers that want to show the curve
/// rather than a single point. Cheap: it is the same closed form per step.
///
/// Takes the caller's whole `Envelope` and varies only concurrency. It used to
/// take a bare `KvDtype` and rebuild the rest itself, which silently dropped
/// every other term - once speculation became one of them, the curve would have
/// promised context that the endpoint's own resident drafter had already spent.
/// The curve and the point estimate have to be the same arithmetic, so they
/// take the same input.
pub fn ctx_curve(
    shape: &ModelShape,
    dev: &Device,
    env: &Envelope,
    steps: &[u64],
) -> Vec<(u64, u64)> {
    steps
        .iter()
        .map(|&n| {
            let e = estimate(
                shape,
                &Envelope {
                    concurrency: n,
                    ..*env
                },
                dev,
            );
            (n, e.max_ctx)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// qwen3.5-9B Q8_0 exactly as the probe reads it off the GGUF: 8 of 32
    /// backbone blocks hold KV (Gated-DeltaNet hybrid), K and V 1024 wide, 24
    /// recurrent blocks, trained for a 262144-token window.
    fn qwen35_9b() -> ModelShape {
        ModelShape {
            weight_bytes: 9_786_061_152,
            kv_reserve_sequences: 0,
            tower_bytes: 0,
            workspace_bytes: 0,
            kind: ModelKind::Generative,
            kv_layers: vec![
                KvLayer {
                    k_dim: 1024,
                    v_dim: 1024,
                    window: None
                };
                8
            ],
            vocab: 248_320,
            recurrent: Some(RecurrentShape {
                layers: 24,
                state_elems: 524_288,
                conv_elems: 24_576,
                conv_dim: 8_192,
                elem_bytes: 4,
                paged_checkpoints: true,
            }),
            cross_kv: None,
            max_ctx: 262_144,
            nextn_bytes: 0,
        }
    }

    /// KB-Whisper large-v3 exactly as the probe reads it off our GGUF
    /// (verified against the file): 32 decoder blocks, K and V
    /// 1280 wide, trained for a 448-token transcript, and a static cross
    /// cache over the fixed 1500-frame audio window.
    fn whisper_large_v3() -> ModelShape {
        ModelShape {
            weight_bytes: 3_223_785_280,
            kv_reserve_sequences: 0,
            tower_bytes: 0,
            workspace_bytes: 0,
            kind: ModelKind::Generative,
            kv_layers: vec![
                KvLayer {
                    k_dim: 1280,
                    v_dim: 1280,
                    window: None
                };
                32
            ],
            vocab: 51_866,
            recurrent: None,
            cross_kv: Some(CrossKv {
                layers: 32,
                frames: 1500,
                k_dim: 1280,
                v_dim: 1280,
            }),
            max_ctx: 448,
            nextn_bytes: 0,
        }
    }

    /// The whole reason `CrossKv` exists: before it, a whisper file probed as
    /// zero KV layers and this estimate came out as "weights plus change",
    /// understating a 32-slot endpoint by about 9 GB.
    ///
    /// The bar is the MEASURED footprint of the real server at 32 slots
    /// (`nvidia-smi`): 13 780 MiB at f16 and 8 916 MiB at fp8, of
    /// which the decode pool is 9740 / 4870. An estimate is allowed to be
    /// conservative - over-stating refuses a start that would have been
    /// tight, which is the safe direction - but it must never come in under
    /// what the server actually takes.
    #[test]
    fn whisper_prices_its_cross_attention_cache_per_slot() {
        let shape = whisper_large_v3();
        let mib = |b: u64| b / (1 << 20);
        for (kv, measured_mib, pool_mib) in [
            (KvDtype::F16, 13_780u64, 9_740u64),
            (KvDtype::Fp8E4m3, 8_916, 4_870),
        ] {
            let env = Envelope {
                concurrency: 32,
                kv_dtype: kv,
                spec: None,
                offload: None,
            };
            // a card big enough that nothing is clamped - this test is about
            // what is CHARGED, not about what fits
            let dev = Device {
                free_bytes: 96 << 30,
                total_bytes: 96 << 30,
            };
            let e = estimate(&shape, &env, &dev);

            // the per-slot cross cache alone must reproduce the cross half of
            // the measured pool: 32 layers x 1500 frames x 2560 elems x bytes
            let cross_total = shape.cross_kv.unwrap().bytes_per_slot(kv) * 32;
            let self_total = shape.kv_per_sequence(448, kv) * 32;
            assert_eq!(
                mib(cross_total + self_total),
                pool_mib,
                "{kv:?}: cross+self per 32 slots must equal the engine's own pool report"
            );
            // and it must actually land in the estimate, not just be computable
            assert_eq!(
                e.state, cross_total,
                "{kv:?}: the cross cache belongs in `state`"
            );
            assert!(
                mib(e.resident + e.kv_pool) >= measured_mib,
                "{kv:?}: estimate {} MiB is UNDER the measured {measured_mib} MiB",
                mib(e.resident + e.kv_pool)
            );
        }
    }

    /// Context is not the lever for whisper - its transcript window is fixed
    /// at 448 by the weights, and no amount of VRAM buys more. What VRAM buys
    /// is CONCURRENCY, so the fit verdict has to move with slot count.
    #[test]
    fn whisper_scales_with_slots_not_context() {
        let shape = whisper_large_v3();
        let kv = KvDtype::Fp8E4m3;
        // 8 GiB: comfortably enough for one transcription, nowhere near
        // enough for 64 of them once each carries its own audio window
        let dev = Device {
            free_bytes: 8 << 30,
            total_bytes: 8 << 30,
        };
        let at = |n: u64| {
            estimate(
                &shape,
                &Envelope {
                    concurrency: n,
                    kv_dtype: kv,
                    spec: None,
                    offload: None,
                },
                &dev,
            )
        };
        let (one, many) = (at(1), at(64));
        assert_eq!(
            one.max_ctx, 448,
            "the trained window is the ceiling, always"
        );
        assert_eq!(
            one.limited_by,
            LimitedBy::Model,
            "VRAM never limits whisper's context"
        );
        // the resident floor grows by exactly the CROSS cache per extra slot
        // (the self-attention half lives in the pool, not the floor)
        let per_slot = shape.cross_kv.unwrap().bytes_per_slot(kv);
        assert!(
            many.resident - one.resident >= 63 * per_slot,
            "63 more slots must add at least 63 cross caches: {} vs {}",
            many.resident - one.resident,
            63 * per_slot
        );
        assert!(
            matches!(one.fit, Fit::Fits { .. }),
            "one transcription fits an 8 GB card"
        );
        assert!(
            matches!(many.fit, Fit::DoesNotFit { .. }),
            "64 audio windows plus 3 GB of weights cannot fit 8 GB - it must refuse, not promise"
        );
    }

    /// gemma-4-26B-A4B Q8_0 exactly as the probe reads it off the GGUF
    /// (verified against the file): 30 blocks - 25 sliding-window
    /// (1024) at 2048-wide K/V, 5 full-attention at 1024. The full-attention
    /// blocks are V-less in the FILE (no attn_v tensor; the probe falls back
    /// to v_dim = k_dim), but the engine allocates the V plane anyway
    /// (gemma4/batch.rs alloc_kv pushes k AND v unconditionally), so the
    /// fallback prices the engine's actual behaviour, not a guess.
    ///
    /// `workspace_bytes` is the engine's own `scratch_mem` self-report at the
    /// default 32-slot width (/api/stats): MoE expert staging the
    /// GRAPH_MARGIN cannot cover. It is 3.81 GiB at 1 slot, so the flat charge
    /// over-states narrow envelopes by ~2 GiB - the safe direction.
    fn gemma4_a4b() -> ModelShape {
        let mut kv_layers = vec![
            KvLayer {
                k_dim: 2048,
                v_dim: 2048,
                window: Some(1024)
            };
            25
        ];
        kv_layers.extend(vec![
            KvLayer {
                k_dim: 1024,
                v_dim: 1024,
                window: None
            };
            5
        ]);
        ModelShape {
            weight_bytes: 26_859_861_728,
            kv_reserve_sequences: 0,
            tower_bytes: 1_194_828_256,
            workspace_bytes: 6_070_491_580,
            kind: ModelKind::Generative,
            kv_layers,
            vocab: 262_144,
            recurrent: None,
            cross_kv: None,
            max_ctx: 262_144,
            nextn_bytes: 0,
        }
    }

    /// The A4B's estimate must never come in under what the server measurably
    /// takes (nvidia-smi): 33 755 MiB at
    /// max-batch 1 (mmproj attached, spec off) and 45 433 MiB at max-batch 32
    /// with the MTP drafter attached - both at the family-default fp8 KV.
    /// Before `workspace_bytes` existed the c1 answer was ~1.1 GiB under.
    #[test]
    fn a4b_moe_workspace_keeps_the_estimate_above_the_measured_serve() {
        let shape = gemma4_a4b();
        let mib = |b: u64| b / (1 << 20);
        // a card big enough that nothing is clamped - the test is about what
        // is CHARGED (the PRO 6000 the measurements come from)
        let dev = Device {
            free_bytes: 96 << 30,
            total_bytes: 96 << 30,
        };
        let c1 = estimate(
            &shape,
            &Envelope {
                concurrency: 1,
                kv_dtype: KvDtype::Fp8E4m3,
                spec: None,
                offload: None,
            },
            &dev,
        );
        assert!(
            mib(c1.resident + c1.kv_pool) >= 33_755,
            "c1 estimate {} MiB is UNDER the measured 33 755 MiB",
            mib(c1.resident + c1.kv_pool)
        );
        let c32 = estimate(
            &shape,
            &Envelope {
                concurrency: 32,
                kv_dtype: KvDtype::Fp8E4m3,
                spec: Some(SpecCost {
                    drafter_bytes: 461_766_816,
                    ..Default::default()
                }),
                offload: None,
            },
            &dev,
        );
        assert!(
            mib(c32.resident + c32.kv_pool) >= 45_433,
            "c32 estimate {} MiB is UNDER the measured 45 433 MiB",
            mib(c32.resident + c32.kv_pool)
        );
        // the workspace is charged as its own reported line, not smuggled
        // into weights (which must stay the artifact's file size)
        assert_eq!(c1.workspace, 6_070_491_580);
        assert_eq!(c1.weights, 26_859_861_728);
        // and the SWA split holds: past every window, only the 5 full-
        // attention blocks keep growing
        let at = |ctx| shape.kv_per_sequence(ctx, KvDtype::Fp8E4m3);
        assert_eq!(at(8192) - at(4096), 4096 * 5 * (1024 + 1024));
    }

    /// Qwen3-Embedding-0.6B: 28 dense blocks, all with a K projection - which
    /// under the old dense-worst-case model produced "needs 124 GB" for a
    /// 0.6 GB file.
    fn embedding_0_6b() -> ModelShape {
        ModelShape {
            weight_bytes: 639_150_592,
            kv_reserve_sequences: 0,
            tower_bytes: 0,
            workspace_bytes: 0,
            kind: ModelKind::Encoder,
            kv_layers: vec![
                KvLayer {
                    k_dim: 1024,
                    v_dim: 1024,
                    window: None
                };
                28
            ],
            vocab: 151_669,
            recurrent: None,
            cross_kv: None,
            max_ctx: 32_768,
            nextn_bytes: 0,
        }
    }

    /// The A6000 as the server reports it once the loaded model is counted as
    /// reclaimable: ~43 GB available to a model on a 48 GB card.
    fn a6000() -> Device {
        Device {
            free_bytes: 46_500_000_000,
            total_bytes: 51_539_607_552,
        }
    }

    fn at(shape: &ModelShape, n: u64) -> Estimate {
        estimate(
            shape,
            &Envelope {
                concurrency: n,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
            &a6000(),
        )
    }

    /// The bug that forced this rewrite: the context list stopped at 131072 and
    /// there was no way to ask for qwen3.5-9B's real 262144. Context is now
    /// derived, so a single session gets the model's whole trained window.
    #[test]
    fn a_single_session_reaches_the_models_full_window() {
        let e = at(&qwen35_9b(), 1);
        assert_eq!(e.max_ctx, 262_144);
        assert_eq!(e.limited_by, LimitedBy::Model);
    }

    /// Spare VRAM never buys context the weights cannot address.
    #[test]
    fn context_never_exceeds_the_trained_window() {
        let huge = Device {
            free_bytes: 900 << 30,
            total_bytes: 900 << 30,
        };
        let e = estimate(
            &qwen35_9b(),
            &Envelope {
                concurrency: 1,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
            &huge,
        );
        assert_eq!(e.max_ctx, 262_144);
        assert_eq!(e.limited_by, LimitedBy::Model);
    }

    /// Concurrency trades against context, monotonically - the curve the page
    /// draws. Doubling the sessions must never increase the window.
    #[test]
    fn context_falls_monotonically_as_concurrency_rises() {
        let shape = qwen35_9b();
        let curve = ctx_curve(
            &shape,
            &a6000(),
            &Envelope {
                concurrency: 1,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
            &[1, 2, 4, 8, 16, 32],
        );
        for w in curve.windows(2) {
            assert!(
                w[1].1 <= w[0].1,
                "context rose from {:?} to {:?}",
                w[0],
                w[1]
            );
        }
        // and at the wide end the card, not the model, is what binds
        let wide = at(&shape, 32);
        assert_eq!(wide.limited_by, LimitedBy::Vram);
        assert!(wide.max_ctx >= 8192, "max_ctx = {}", wide.max_ctx);
    }

    /// Whatever context is reported must actually be backed by the pool that
    /// was allocated - the arithmetic has to close.
    #[test]
    fn reported_context_is_backed_by_the_reported_pool() {
        let shape = qwen35_9b();
        for n in [1u64, 4, 32] {
            let e = at(&shape, n);
            let needed = shape.kv_per_sequence(e.max_ctx, KvDtype::F16) * n;
            assert!(
                needed <= e.kv_pool,
                "n={n}: needs {needed}, pool {}",
                e.kv_pool
            );
        }
    }

    /// A windowed model's context is the inverse of its real KV cost, not the
    /// pool over its one-token cost: gemma4's full 262K window fits a 5.6 GB
    /// pool, and the division used to report 24752 of it, limited by "vram",
    /// with gigabytes spare - so every start shrank onto that.
    #[test]
    fn windowed_models_reach_the_window_their_pool_holds() {
        let shape = gemma4_a4b();
        let e = at(&shape, 1);
        assert_eq!(e.max_ctx, 262_144);
        assert_eq!(e.limited_by, LimitedBy::Model);
        // and where the card does bind, the window is tight against the pool:
        // one page more would not fit
        for n in [8u64, 32] {
            let e = at(&shape, n);
            assert_eq!(e.limited_by, LimitedBy::Vram);
            let cost = |ctx| shape.kv_per_sequence(ctx, KvDtype::F16) * n;
            assert!(cost(e.max_ctx) <= e.kv_pool, "n={n}");
            assert!(cost(e.max_ctx + 16) > e.kv_pool, "n={n}");
        }
        // the slope a client multiplies by context: the full-window average
        // (~the 5 full layers' rate), not the one-token cost that charges all
        // 25 windowed layers on every token
        assert!(
            shape.kv_bytes_per_token(KvDtype::F16) < shape.kv_per_sequence(1, KvDtype::F16) / 10
        );
        assert_eq!(
            qwen35_9b().kv_bytes_per_token(KvDtype::F16),
            qwen35_9b().kv_per_sequence(1, KvDtype::F16)
        );
        // and a client's slope x window x sessions never exceeds the pool that
        // holds them all
        let e = at(&shape, 1);
        assert!(shape.kv_bytes_per_token(KvDtype::F16) * e.max_ctx <= e.kv_pool);
    }

    /// Context is a whole number of the engine's 16-token pages.
    #[test]
    fn context_lands_on_a_page_boundary() {
        for n in [1u64, 3, 7, 32] {
            assert_eq!(at(&qwen35_9b(), n).max_ctx % 16, 0);
        }
    }

    /// Encoders cache nothing between calls, so they have no window to report
    /// and their cost does not move with concurrency.
    #[test]
    fn encoders_have_no_window_and_no_envelope_cost() {
        let shape = embedding_0_6b();
        let (one, many) = (at(&shape, 1), at(&shape, 64));
        assert_eq!(one.kv_pool, 0);
        assert_eq!(one.limited_by, LimitedBy::NotApplicable);
        assert_eq!(one.resident, many.resident);
        // a 0.64 GB file lands near its weights, not in three figures. Read
        // through the reported parts rather than restating the formula: an
        // encoder's only overhead is the allocator slack its planes pay, and
        // hardcoding the sum here is what made this test fail the first time a
        // real resident term was added rather than catching a mistake.
        assert_eq!(one.resident, one.weights + one.overhead + CUDA_CONTEXT);
        assert!(one.overhead > 0, "planes pay allocator rounding");
        assert!(one.resident < 2 * one.weights, "still near its weights");
    }

    /// Halving the KV element width buys twice the context, nothing else.
    #[test]
    fn fp8_kv_doubles_the_window() {
        let shape = qwen35_9b();
        let dev = a6000();
        let f16 = estimate(
            &shape,
            &Envelope {
                concurrency: 32,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
            &dev,
        );
        let fp8 = estimate(
            &shape,
            &Envelope {
                concurrency: 32,
                kv_dtype: KvDtype::Fp8E4m3,
                spec: None,
                offload: None,
            },
            &dev,
        );
        // Doubled, to within the 16-token page both answers are rounded down
        // to - flooring twice at half the width can leave one page in hand.
        let doubled = f16.max_ctx * 2;
        assert!(
            fp8.max_ctx >= doubled && fp8.max_ctx <= doubled + 16,
            "f16 {} -> fp8 {}",
            f16.max_ctx,
            fp8.max_ctx
        );
        // The floor does not move with the KV dtype - except by page
        // rounding: paged checkpoints occupy whole pool pages, and a page is
        // half the bytes at fp8, so each checkpoint rounds up by less than
        // one f16 page either way.
        let page = shape.kv_per_sequence(16, KvDtype::F16);
        assert!(
            fp8.resident.abs_diff(f16.resident) <= PREFIX_CKPT_FLOOR * page,
            "f16 {} vs fp8 {}",
            f16.resident,
            fp8.resident
        );
    }

    /// Sliding-window blocks stop growing at their window - the property that
    /// makes gemma4 and gpt-oss cheap at long context.
    #[test]
    fn sliding_window_layers_stop_growing() {
        let shape = ModelShape {
            weight_bytes: 0,
            kv_reserve_sequences: 0,
            tower_bytes: 0,
            workspace_bytes: 0,
            kind: ModelKind::Generative,
            kv_layers: vec![
                KvLayer {
                    k_dim: 512,
                    v_dim: 512,
                    window: Some(128),
                },
                KvLayer {
                    k_dim: 512,
                    v_dim: 512,
                    window: None,
                },
            ],
            vocab: 0,
            recurrent: None,
            cross_kv: None,
            max_ctx: 131_072,
            nextn_bytes: 0,
        };
        let at = |ctx| shape.kv_per_sequence(ctx, KvDtype::F16);
        assert_eq!(at(8192) - at(4096), 4096 * (512 + 512) * 2);
    }

    #[test]
    fn reserved_kv_capacity_costs_memory_but_is_not_serving_concurrency() {
        let mut shape = qwen35_9b();
        shape.recurrent = None;
        shape.weight_bytes = 0;
        shape.max_ctx = 4096;
        let dev = Device {
            free_bytes: 64 << 30,
            total_bytes: 64 << 30,
        };
        for n in [1, 4] {
            let env = Envelope {
                concurrency: n,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            };
            shape.kv_reserve_sequences = 0;
            let ordinary = estimate(&shape, &env, &dev);
            shape.kv_reserve_sequences = 1;
            let reserved = estimate(&shape, &env, &dev);
            let per_sequence = shape.kv_per_sequence(4096, KvDtype::F16);
            assert_eq!(ordinary.kv_pool, n * per_sequence);
            assert_eq!(reserved.kv_pool, (n + 1) * per_sequence);
            assert_eq!(reserved.resident, ordinary.resident);
            assert_eq!(reserved.max_ctx, 4096);
        }
        // Under memory pressure, the inverse must pay for the reserved
        // sequence too, or it advertises a context that cannot be allocated.
        shape.max_ctx = 131072;
        let dev = Device {
            free_bytes: 5 << 30,
            total_bytes: 5 << 30,
        };
        let env = Envelope {
            concurrency: 1,
            kv_dtype: KvDtype::F16,
            spec: None,
            offload: None,
        };
        shape.kv_reserve_sequences = 0;
        let ordinary = estimate(&shape, &env, &dev);
        shape.kv_reserve_sequences = 1;
        let reserved = estimate(&shape, &env, &dev);
        assert_eq!(reserved.kv_pool, ordinary.kv_pool);
        assert!(reserved.max_ctx <= ordinary.max_ctx / 2);
        assert!(shape.kv_per_sequence(reserved.max_ctx, KvDtype::F16) * 2 <= reserved.kv_pool);
    }

    /// An in-file drafter (nextn) is inside `weight_bytes` because that is the
    /// FILE's size, but the engine only loads those blocks when speculating.
    /// So the spec toggle has to move `resident` even with no separate drafter
    /// to charge - otherwise the control appears to do nothing on exactly the
    /// models whose MTP ships in the weights (qwen3.5/3.6).
    #[test]
    fn in_file_mtp_is_given_back_when_not_speculating() {
        let mut shape = qwen35_9b();
        shape.nextn_bytes = 260_000_000;
        let dev = a6000();
        let off = estimate(
            &shape,
            &Envelope {
                concurrency: 4,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
            &dev,
        );
        let on = estimate(
            &shape,
            &Envelope {
                concurrency: 4,
                kv_dtype: KvDtype::F16,
                // in-file MTP: no separate drafter file to charge
                spec: Some(SpecCost {
                    drafter_bytes: 0,
                    ..Default::default()
                }),
                offload: None,
            },
            &dev,
        );
        assert!(
            on.resident > off.resident,
            "speculating must cost more than not"
        );
        // the nextn blocks, plus the wider verify logits plane
        assert!(
            on.resident - off.resident >= shape.nextn_bytes,
            "nextn bytes must be in the delta"
        );
    }

    /// A hybrid's draft chain carries RECURRENT state, not just an MTP K/V
    /// plane, and the estimate used to price only the plane.
    ///
    /// The engine reserves `n_lin * (k+1) * state_elems * 4` per live spec slot
    /// (qwen35/batch.rs `spec_est`) - the draft depth multiplies the whole
    /// recurrent width. On Qwen3.5-9B Q8 at 4096x32 that is 2.08 GiB the grant
    /// paid nothing for, and the endpoint then refused to start because the KV
    /// pool it was promised did not fit inside its own budget.
    ///
    /// Guards the shape of the term, not a constant: speculating on a
    /// recurrent model must cost more than speculating on the same model with
    /// its recurrent blocks removed.
    #[test]
    fn a_hybrids_draft_chain_carries_recurrent_state() {
        let dev = a6000();
        let env = |spec| Envelope {
            concurrency: 8,
            kv_dtype: KvDtype::F16,
            spec,
            offload: None,
        };
        let hybrid = qwen35_9b();
        assert!(
            hybrid.recurrent.is_some(),
            "fixture must be a hybrid for this to mean anything"
        );
        let mut dense = hybrid.clone();
        dense.recurrent = None;

        let cost = |shape: &ModelShape| {
            let on = estimate(shape, &env(Some(SpecCost::default())), &dev);
            let off = estimate(shape, &env(None), &dev);
            on.resident - off.resident
        };
        assert!(
            cost(&hybrid) > cost(&dense),
            "a hybrid's speculation must cost more than a dense model's: hybrid {} vs dense {}",
            cost(&hybrid),
            cost(&dense),
        );
    }

    /// A vision tower is resident weights and has to be charged. Unlike the
    /// drafter there is no toggle: the engine loads the mmproj at startup
    /// whenever one is wired, so an endpoint that serves images pays it from
    /// the first second. Leaving it out understated granite-vision-4.1-4b by
    /// 1.10 GB, which is the picker saying "fits" about a start that might not.
    #[test]
    fn a_vision_tower_is_charged_as_resident_weights() {
        let mut shape = qwen35_9b();
        let text_only = at(&shape, 4);
        shape.tower_bytes = 1_100_000_000;
        let with_tower = at(&shape, 4);
        // The tower is a resident PLANE, so it pays the allocator's rounding
        // like every other one - the endpoint holds more than the mmproj's own
        // bytes and the fit check has to know it.
        let slack = (1_100_000_000f64 * ALLOCATOR_SLACK_SHARE) as u64;
        assert_eq!(
            with_tower.resident - text_only.resident,
            1_100_000_000 + slack
        );
        // reported on its own line, not lumped into weights or overhead -
        // an mmproj shown as "engine overhead" is a true total, false story
        assert_eq!(with_tower.tower, 1_100_000_000);
        assert_eq!(with_tower.weights, text_only.weights);
        assert_eq!(with_tower.overhead - text_only.overhead, slack);
        // and it comes out of the cache, not out of nowhere
        assert!(with_tower.max_ctx <= text_only.max_ctx);
    }

    /// Issue #33: a family whose checkpoints live in pool pages is charged one
    /// turn of them per slot, up to the old floor, and loses no pool out of
    /// its context; a fixed-pool family keeps the 16-deep floor and its pool.
    #[test]
    fn paged_checkpoints_price_the_live_turns_not_a_fixed_pool() {
        let paged = qwen35_9b();
        let mut fixed = qwen35_9b();
        if let Some(r) = fixed.recurrent.as_mut() {
            r.paged_checkpoints = false;
        }
        let dev = Device {
            free_bytes: 24 << 30,
            total_bytes: 24 << 30,
        };
        let one = Envelope {
            concurrency: 1,
            kv_dtype: KvDtype::F16,
            spec: None,
            offload: None,
        };
        let (p, f) = (estimate(&paged, &one, &dev), estimate(&fixed, &one, &dev));
        let r = paged.recurrent.expect("hybrid");
        let per_ckpt = r.layers * (r.state_elems + r.conv_elems) * r.elem_bytes;
        assert!(
            f.overhead_parts.prefix_checkpoints - p.overhead_parts.prefix_checkpoints
                >= 12 * per_ckpt,
            "one slot pays 3 checkpoints, not 16"
        );
        assert_eq!(p.overhead_parts.prefix_pool_extra, 0);
        assert!(
            p.max_ctx >= f.max_ctx,
            "and the context gets the difference"
        );
        // at width 8 the turns (24) are capped at the old floor (16)
        let eight = Envelope {
            concurrency: 8,
            ..one
        };
        let (p8, f8) = (
            estimate(&paged, &eight, &dev),
            estimate(&fixed, &eight, &dev),
        );
        assert!(
            p8.overhead_parts.prefix_checkpoints
                <= f8.overhead_parts.prefix_checkpoints + 16 * per_ckpt / 8,
            "a wide paged server reserves no more than the fixed floor (up to page rounding)"
        );
    }

    /// The three terms the engine reserves before the KV pool and this crate
    /// used to ignore. Measured on Qwen3.8-27B: the engine charged
    /// itself 7.49 GiB, this estimate charged 3.78, and the difference is why
    /// the edit page said "Fits" about a 131072-token server that then refused
    /// to start. Each is asserted through the reported breakdown,
    /// so removing one fails here rather than in a user's face.
    #[test]
    fn the_reserves_the_engine_takes_before_kv_are_charged() {
        let shape = qwen35_9b();
        let dev = Device {
            free_bytes: 48 << 30,
            total_bytes: 48 << 30,
        };
        let plain = Envelope {
            concurrency: 1,
            kv_dtype: KvDtype::F16,
            spec: None,
            offload: None,
        };
        let speccy = Envelope {
            spec: Some(SpecCost::default()),
            ..plain
        };
        let a = estimate(&shape, &plain, &dev);
        let b = estimate(&shape, &speccy, &dev);

        let r = shape.recurrent.expect("qwen35 is a hybrid");
        let per_ckpt = r.layers * (r.state_elems + r.conv_elems) * r.elem_bytes;
        // one slot's turn of checkpoints (qwen35's are pool pages), plus the
        // two staging blobs
        assert!(
            a.overhead >= (PREFIX_CKPTS_PER_TURN + 2) * per_ckpt,
            "prefix checkpoints unpriced: overhead {} < {}",
            a.overhead,
            (PREFIX_CKPTS_PER_TURN + 2) * per_ckpt
        );
        // the allocator's rounding on every resident plane
        assert!(a.overhead >= (shape.weight_bytes as f64 * ALLOCATOR_SLACK_SHARE) as u64);
        // speculation's RUNTIME state, which is not its weights: turning spec
        // on has to cost more than the drafter's bytes (0 for in-file MTP)
        assert!(
            b.overhead > a.overhead,
            "spec runtime state unpriced: {} vs {}",
            b.overhead,
            a.overhead
        );
        // and every one of them comes out of the context on offer
        assert!(b.max_ctx <= a.max_ctx);
    }

    /// Only the resident floor can fail to fit; the cache shrinks to the card.
    #[test]
    fn only_the_resident_floor_can_fail_to_fit() {
        let e = estimate(
            &qwen35_9b(),
            &Envelope {
                concurrency: 1,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
            &Device {
                free_bytes: 6 << 30,
                total_bytes: 8 << 30,
            },
        );
        let Fit::DoesNotFit { short_by_bytes } = e.fit else {
            panic!("expected no-fit")
        };
        assert_eq!(short_by_bytes, e.resident - (6 << 30));
    }

    /// A model that loads but can't reach a workable window is "tight", not
    /// "too big" - a distinction the dense-worst-case model couldn't make.
    #[test]
    fn a_loadable_model_with_no_room_for_context_is_tight() {
        let shape = qwen35_9b();
        let env = Envelope {
            concurrency: 1,
            kv_dtype: KvDtype::F16,
            spec: None,
            offload: None,
        };
        // Derive the floor rather than reconstruct it - `resident` also carries
        // recurrent state and scratch, and hand-adding the parts got it wrong.
        // 40 MB over is ~1200 tokens at this model's 32 KB/token: it loads, and
        // there is nothing left to serve with.
        let floor = estimate(
            &shape,
            &env,
            &Device {
                free_bytes: u64::MAX / 2,
                total_bytes: u64::MAX / 2,
            },
        )
        .resident;
        let barely = floor + 40_000_000;
        let e = estimate(
            &shape,
            &env,
            &Device {
                free_bytes: barely,
                total_bytes: barely,
            },
        );
        assert!(matches!(e.fit, Fit::Tight { .. }), "{:?}", e.fit);
        assert!(e.max_ctx < MIN_USEFUL_CTX, "max_ctx = {}", e.max_ctx);
    }

    /// The parts of the resident floor still sum to the whole. Run with a
    /// vision tower too, so a new resident term can never be added without a
    /// reported line to go with it - the failure mode this guards is a total
    /// that grows while the breakdown silently stops adding up.
    /// Arming the tier costs the resident floor exactly the staging reserve
    /// the engine takes - no more (which would under-promise context) and no
    /// less (which would promise context the runner cannot seat).
    ///
    /// Whether that reaches the KV POOL depends on what the pool is limited
    /// by: it is `want.min(headroom).min(share_cap)`, so on a roomy card the
    /// staging comes out of unused headroom and the served context is
    /// unchanged. Both cases are checked, because "arming the cache shrinks
    /// my context" is true only in the second one and saying otherwise would
    /// be its own inaccuracy.
    #[test]
    fn arming_the_prefix_cache_costs_the_floor_exactly_the_staging_reserve() {
        let shape = qwen35_9b();
        let base = Envelope {
            concurrency: 4,
            kv_dtype: KvDtype::F16,
            spec: None,
            offload: None,
        };
        let armed = Envelope {
            offload: Some(OffloadCost::armed(24 << 30)),
            ..base
        };
        let staging = paddock_models::kv_tier_geom::device_staging_bytes();

        // roomy card: the floor moves, the pool does not
        let roomy = Device {
            free_bytes: 40 << 30,
            total_bytes: 48 << 30,
        };
        let (off, on) = (
            estimate(&shape, &base, &roomy),
            estimate(&shape, &armed, &roomy),
        );
        assert_eq!(
            on.overhead - off.overhead,
            staging,
            "staging belongs to overhead"
        );
        assert_eq!(
            on.resident - off.resident,
            staging,
            "and to the resident floor"
        );
        assert!(
            on.kv_pool <= off.kv_pool,
            "the pool never GROWS by arming a cache"
        );

        // headroom-limited card: now the pool is what pays, exactly. The
        // precondition is asserted rather than assumed - if the fixture's
        // shape changes this fails saying why instead of silently testing
        // the roomy case twice.
        // derived from the fixture rather than guessed: just enough free VRAM
        // to fit the floor plus a GiB, so headroom is the binding limit
        let tight = Device {
            free_bytes: off.resident + (1 << 30),
            total_bytes: 48 << 30,
        };
        let (off, on) = (
            estimate(&shape, &base, &tight),
            estimate(&shape, &armed, &tight),
        );
        let cap = (KV_POOL_SHARE_CAP * tight.free_bytes as f64) as u64;
        let headroom = tight.free_bytes - off.resident;
        assert!(
            off.kv_pool > 0 && headroom < cap,
            "fixture no longer headroom-limited (headroom {headroom}, cap {cap})"
        );
        assert_eq!(on.resident - off.resident, staging);
        assert_eq!(
            off.kv_pool - on.kv_pool,
            staging,
            "when headroom is the limit, the pool pays the staging and nothing else"
        );
    }

    /// Host RAM is a different resource. It must be REPORTED - a fit surface
    /// that never mentions the feature's real price is hiding it - and it must
    /// never move a single VRAM figure.
    #[test]
    fn host_ram_is_reported_and_never_counted_as_vram() {
        let shape = qwen35_9b();
        let dev = Device {
            free_bytes: 40 << 30,
            total_bytes: 48 << 30,
        };
        let base = Envelope {
            concurrency: 4,
            kv_dtype: KvDtype::F16,
            spec: None,
            offload: None,
        };
        let small = estimate(
            &shape,
            &Envelope {
                offload: Some(OffloadCost::armed(1 << 30)),
                ..base
            },
            &dev,
        );
        // deliberately larger than the whole card: if host RAM leaked into any
        // VRAM term, this would move it
        let huge = estimate(
            &shape,
            &Envelope {
                offload: Some(OffloadCost::armed(200 << 30)),
                ..base
            },
            &dev,
        );
        assert_eq!(small.host_ram, 1 << 30);
        assert_eq!(huge.host_ram, 200 << 30, "the commitment is reported");
        assert_eq!(
            huge.resident, small.resident,
            "host RAM must not touch the floor"
        );
        assert_eq!(huge.overhead, small.overhead, "nor the overhead line");
        assert_eq!(huge.kv_pool, small.kv_pool, "nor the pool");
        assert_eq!(huge.max_ctx, small.max_ctx, "nor the served context");
        // and with no tier there is nothing to report
        assert_eq!(estimate(&shape, &base, &dev).host_ram, 0);
    }

    #[test]
    fn resident_parts_sum_to_the_whole() {
        let mut shape = qwen35_9b();
        for _ in 0..2 {
            let e = at(&shape, 8);
            assert_eq!(
                e.weights
                    + e.tower
                    + e.workspace
                    + e.state
                    + e.overhead
                    + fixed_overhead(ModelKind::Generative),
                e.resident
            );
            shape.tower_bytes = 1_100_000_000;
            shape.workspace_bytes = 6_000_000_000;
        }
    }
}
