//! Whisper decoder - batched greedy transcription over decode SLOTS
//! (bring-up - the serving lane). Reference: transformers
//! `WhisperDecoder`; vLLM `models/whisper.py` is the serving cross-reference
//! and its prompt construction is what this matches exactly.
//!
//! Per decoder layer, in order:
//!   LN -> self-attention (causal, over the ≤448 tokens decoded so far)
//!     -> +res
//!   LN -> CROSS-attention over the encoder window -> +res
//!   LN -> fc1 -> GELU-erf -> fc2 -> +res
//! then a final LN and the `proj_out` head.
//!
//! Cross-attention is the mechanism no other served family has, and its
//! shape is what makes it cheap: K and V come from the ENCODER states (no LN,
//! no position, no causality) and are therefore computed once per window and
//! static for the whole decode. `encode_into` precomputes both planes per
//! layer straight into the slot's f16 cache; each decode step is then two
//! single-query attentions over f16 K/V - `whisper_dec_attn`, one kernel for
//! both, per-slot key lengths and a flash-decoding key split.
//!
//! Everything is BATCHED over ROWS. `x`, `q`, the logits - all compact
//! `[batch, ...]` in ACTIVE order; the K/V planes are `[cap, stride, d]`
//! addressed through a `slots` index vector, so a slot that finishes leaves
//! the active set without any cache moving. `pos`, the fed token and the slot
//! id are device-resident vectors uploaded per step, which is also what keeps
//! the step free of per-slot host branching.
//!
//! ROWS are not SLOTS, and the indirection through `slots` is what makes them
//! separable: serving runs one row per slot, but nothing in the tick requires
//! that, and the word-timing pass runs a chunk of rows on one slot at
//! consecutive positions - a prefill in all but name. `Scratch` is sized by
//! rows, the KV planes by slots.
//!
//! Prompting follows the checkpoint's decode contract, ids read from the
//! file rather than assumed: `<|startoftranscript|><|{lang}|><|transcribe|>
//! <|notimestamps|>`, then greedy tokens until `<|endoftext|>`. vLLM builds
//! the identical string (`whisper.py get_generation_prompt`). On suppression
//! the reference is openai/whisper, not vLLM: its decoder masks the prompt
//! block and `<|nospeech|>` before every softmax (`SuppressTokens`), and so
//! does this one, through the pick kernel's mask (`DecodeTokens::suppressed`).
//! Without it a window holding a sliver of audio comes back as a confident
//! `<|nospeech|>` token that the no-speech guard keeps (it is confident) and
//! the word-timing pass refuses. The HF generation config's wider
//! `suppress_tokens` list (punctuation and symbols) stays unapplied.
//!
//! KV precision: `kv_dtype`, f16 by default and fp8-e4m3 under
//! `--kv-cache-dtype fp8_e4m3`. f16 is the reference's own class (vLLM serves
//! whisper at `--dtype float16` and its KV cache follows the model dtype),
//! and it already halved what the cross planes cost - they were 491 MB per
//! window at f32.
//!
//! The fp8 arm exists because whisper's cross planes are unlike any other
//! family's KV: 1500 encoder frames per slot per layer, static for the whole
//! decode and never shorter than the full window, so a c32 step reads
//! 32 slots x 32 layers x 1500 x 1280 x 2 planes = 7.9 GB per TOKEN. That one
//! kernel measured 27% of all GPU time in a c32 battery, running at the
//! card's achievable read bandwidth - element width is the only lever on it.
//! Both caches are BYTE planes sized `elems * kv_dtype.bytes()`, the shape
//! every other family's cache already has.

use std::collections::{HashMap, HashSet};

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;
use half::f16;

use crate::audio::{MelFeatures, PAD_SAMPLES, whisper_features};
use crate::gpu::{GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::{GpuWhisper, LangProb, SendGraph};

/// Must match `PD_WD_MAX_SPLITS` in `packs/cuda/src/asr/whisper.cuh` - the
/// partial buffer is sized for the launcher's largest possible key split.
const MAX_SPLITS: usize = 32;

/// The decode slot pool: per-layer K/V planes plus the step scratch, allocated
/// at `prepare_batch`. Mid-serve `cudaMalloc` is a known serve-killer, and
/// these are the big buffers.
///
/// The KV planes are never re-allocated. The row scratch can be, but only by
/// `ensure_decode_rows`, only upward, and only off the serving path - see
/// `Scratch`.
pub struct DecodeBatch {
    cap: usize,
    /// per-layer cross-attention K/V, [cap, n_audio_ctx, d] at `kv`
    cross_k: Vec<CudaSlice<u8>>,
    cross_v: Vec<CudaSlice<u8>>,
    /// per-layer causal self-attention cache, [cap, n_text_ctx, d] at `kv`
    self_k: Vec<CudaSlice<u8>>,
    self_v: Vec<CudaSlice<u8>>,
    /// the element type both caches were ALLOCATED at - read from here, never
    /// from the model's current setting, so a mid-life `set_kv_dtype` cannot
    /// reinterpret a pool that was sized for the other width
    kv: KvDtype,
    /// everything one tick computes, sized by ROW count
    s: Scratch,
    /// one window's encoder states at f16, and the LAYER-BATCHED cross K/V
    /// GEMM landing, [n_audio_ctx, n_layer*d] f32: all 32 layers'
    /// cross K (then V) land from one GEMM each - reused for both planes
    enc_stage: CudaSlice<f16>,
    enc_kv_all: CudaSlice<f32>,
    /// device tables of the per-layer cross plane base pointers, for the
    /// one-launch batched store (built once here, so capture-safe)
    cross_k_ptrs: CudaSlice<u64>,
    cross_v_ptrs: CudaSlice<u64>,
    /// slot index vector for the admission pass - front b entries are the
    /// batched windows' target slots
    d_slots: CudaSlice<u32>,
    /// side stream the admission graph replays on, overlapping the decode
    /// tick (P38); eager/capture passes stay on the main stream
    enc_stream: std::sync::Arc<cudarc::driver::CudaStream>,
    /// completion of the last admission pass - synced before its runs join
    enc_done: Option<cudarc::driver::CudaEvent>,
    /// one logits row on the host path (language detection only)
    row: CudaSlice<f32>,
    bytes: u64,
    /// One captured decode tick per (active-row count, timestamp-rules on,
    /// mmaf declined). The step is a fixed chain of ~420 launches whose only
    /// per-step inputs are device vectors, so it replays as a graph - and it
    /// has to, because the launch train was ~4.6 ms of the 12.4 ms step at
    /// c32 (measured against the kernel total).
    ///
    /// The rules flag is part of the key because it adds a launch: a server
    /// nobody asks timestamps of then never records or replays it, and pays
    /// exactly what it paid before. Row counts are the second half - the
    /// only other thing that changes the launch SHAPE, since key lengths
    /// ride device memory. The mmaf flag is the overlap route:
    /// a tick whose admission encode is in flight on the side stream must
    /// not run the mmaf decode arm (P39 - mmaf × tc5p reads stale W-ring
    /// slabs, a below-PTX HW interaction), so those ticks replay a variant
    /// captured with the pack's slot-409 gate declining mmaf. Every other
    /// tick keeps mmaf's speed.
    graphs: HashMap<(usize, bool, bool), SendGraph>,
    /// Shapes that have run once eagerly. cuBLAS picks (and may allocate for)
    /// an algorithm on first sight of a shape, and an allocation during
    /// capture is a hard driver error - so every shape runs once before it is
    /// recorded.
    warmed: HashSet<(usize, bool, bool)>,
    /// Whether the last admission pass is still in flight on the encoder
    /// side stream - set by the transcriber before each tick, read by
    /// `step_replay` to pick the mmaf-off graph variant. Meaningless (and
    /// false) while overlap is gated off.
    enc_inflight: bool,
}

/// Everything one tick computes, indexed by ACTIVE ROW.
///
/// Rows are not slots, and separating them is what makes the teacher-forced
/// word-timing pass affordable: a slot owns ~245 MB of cross+self
/// KV planes, so widening the POOL to run N tokens at once is out of the
/// question - but those N tokens all live on one slot, and only this scratch
/// has to widen. `rows` is a high-water mark that grows independently of `cap`.
struct Scratch {
    /// how many rows every buffer below is sized for
    rows: usize,
    /// the residual stream, [rows, d]
    x: CudaSlice<f32>,
    /// pre-norm landing every GEMM eats, [rows, d]
    nrm: CudaSlice<f16>,
    /// post-GELU fc2 input, [rows, ffn]
    ffh: CudaSlice<f16>,
    /// merged q|k|v landing, [rows, 3d]
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    /// attention output - f16 because the out_proj GEMM is its only consumer
    attn: CudaSlice<f16>,
    /// out_proj / fc2 landing, [rows, d]
    proj: CudaSlice<f32>,
    ff: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    /// flash-decoding partials, [rows, heads, MAX_SPLITS, hd+2]
    part: CudaSlice<f32>,
    d_slots: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_tok: CudaSlice<u32>,
    d_next: CudaSlice<u32>,
    /// `[rows]` - the RUNNER-UP at each row's pick, written by the same
    /// reduction. `vocab` where there wasn't one.
    d_alt: CudaSlice<u32>,
    /// `[rows, 4]` - per active row,
    /// `{log p(top1), p(<|nospeech|>), log p(top2), H2}`, written by the same
    /// reduction that picks the token. Sixteen bytes a row is nothing next to
    /// `d_next`'s own readback, and it is what lets a transcript report
    /// confidence without a second pass over the logits.
    d_stats: CudaSlice<f32>,
    /// per-row `{flags, lowest allowed timestamp}` for the timestamp grammar
    /// - `[rows, 2]`, uploaded per step like `d_pos`
    d_rules: CudaSlice<u32>,
    /// the ids the pick never returns (`DecodeTokens::suppressed`), resident
    /// for the graph's lifetime like every other buffer the tick records
    d_suppress: CudaSlice<u32>,
}

/// What one batched decode step produced, per ACTIVE row (not per slot).
///
/// `logprob` is the chosen token's own log-probability - the confidence a
/// transcript reports. `nospeech` is `<|nospeech|>`'s probability at this
/// step, which only means anything at the very first one: OpenAI's
/// `no_speech_prob` is defined there, where the model is still deciding
/// whether the window contains speech at all.
///
/// `runner_up` is what the model NEARLY picked instead, as `(id, log p)`, and
/// `None` where the row had no second candidate. The gap between the two is
/// the margin - a step where the model was torn between two words reads very
/// differently from one where it was merely diffuse, and only the second is
/// usually still correct.
pub use crate::whisper::StepOut;

impl DecodeBatch {
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Every byte the pool holds - reported at prepare time, because "how
    /// much VRAM does N concurrent transcriptions cost" is a question the
    /// will-it-fit surface has to be able to answer.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Upload the rows' `(slot, token, position)` - the whole per-step host
    /// input the tick has. `step_batch` does this behind its own validation;
    /// the word-timing pass drives `step_body` directly and needs the same
    /// seam without reaching into the pool's buffers itself.
    pub(super) fn feed_rows(
        &mut self,
        exec: &crate::gpu::GpuExecutor,
        slots: &[u32],
        toks: &[u32],
        pos: &[u32],
    ) -> Result<(), GpuError> {
        exec.upload_u32(slots, &mut self.s.d_slots)?;
        exec.upload_u32(toks, &mut self.s.d_tok)?;
        exec.upload_u32(pos, &mut self.s.d_pos)
    }
}

impl GpuWhisper {
    /// The row-indexed step scratch for `rows` active rows. Pulled out because
    /// two callers need it: `prepare_batch` at the pool's slot count, and
    /// `ensure_decode_rows` when the word-timing pass wants a wider tick.
    fn alloc_scratch(
        exec: &crate::gpu::GpuExecutor,
        rows: usize,
        hp: &super::Hparams,
        heads: usize,
        hd: usize,
        suppress: &[u32],
    ) -> Result<Scratch, GpuModelError> {
        let (d, ffn) = (hp.d_model, hp.dec_ffn);
        Ok(Scratch {
            rows,
            x: exec.alloc(rows * d)?,
            nrm: exec.alloc_f16(rows * d)?,
            ffh: exec.alloc_f16(rows * ffn)?,
            qkv: exec.alloc(rows * 3 * d)?,
            q: exec.alloc(rows * d)?,
            attn: exec.alloc_f16(rows * d)?,
            proj: exec.alloc(rows * d)?,
            ff: exec.alloc(rows * ffn)?,
            logits: exec.alloc(rows * hp.n_vocab)?,
            part: exec.alloc(rows * heads * MAX_SPLITS * (hd + 2))?,
            d_slots: exec.to_device_u32(&vec![0u32; rows])?,
            d_pos: exec.to_device_u32(&vec![0u32; rows])?,
            d_tok: exec.to_device_u32(&vec![0u32; rows])?,
            d_next: exec.to_device_u32(&vec![0u32; rows])?,
            d_alt: exec.to_device_u32(&vec![0u32; rows])?,
            d_stats: exec.alloc(rows * 4)?,
            d_rules: exec.to_device_u32(&vec![0u32; rows * 2])?,
            d_suppress: exec.to_device_u32(suppress)?,
        })
    }

    /// Widen the tick to at least `n` active rows, reallocating the scratch if
    /// it is narrower. Never shrinks - a high-water mark, so a request that
    /// asked for word timing once does not pay the allocation again.
    ///
    /// Every captured graph is DROPPED here: a graph records the addresses it
    /// was captured with, and the buffers those nodes point at are about to be
    /// freed. Replaying one afterwards would write into memory that is no
    /// longer ours, which is the kind of failure that surfaces somewhere else
    /// entirely.
    pub(super) fn ensure_decode_rows(&mut self, n: usize) -> Result<(), GpuModelError> {
        let (heads, hd) = (self.hp.n_dec_heads, self.hp.head_dim);
        let exec = self.exec.clone();
        let st = self
            .decode
            .as_mut()
            .ok_or_else(|| GpuModelError::Unsupported("whisper: prepare_batch first".into()))?;
        if n <= st.s.rows {
            return Ok(());
        }
        st.graphs.clear();
        st.warmed.clear();
        // free the old scratch before asking for the new one, so the peak is
        // one set and not two
        st.s = Self::alloc_scratch(&exec, 1, &self.hp, heads, hd, &self.tokens.suppressed())?;
        st.s = Self::alloc_scratch(&exec, n, &self.hp, heads, hd, &self.tokens.suppressed())?;
        tracing::debug!(rows = n, "whisper decode scratch widened");
        Ok(())
    }

    /// Allocate the decode slot pool. `cap` is the concurrency ceiling: each
    /// slot owns a full set of cross-attention planes, which is what whisper
    /// costs on any engine (32 layers × 1500 frames × d_model × 2 planes).
    pub fn prepare_batch(&mut self, cap: usize) -> Result<(), GpuModelError> {
        let cap = cap.max(1);
        if self.decode.as_ref().is_some_and(|b| b.cap >= cap) {
            return Ok(());
        }
        // The admission graph's recorded nodes point straight at the pool's
        // cross-K/V planes, so a pool swap would leave it replaying into
        // freed memory. Drop it here and let the next admission re-record
        // against the new buffers.
        if let Some(sc) = self.enc.as_mut() {
            sc.forget_graph();
        }
        self.decode = None; // free the old pool before asking for a bigger one
        let exec = self.exec.clone();
        // No `ffn` in this tuple, unlike its twin in the decode path below: this is
        // the KV-pool reallocation, which sizes cross/self K and V and never touches
        // the feed-forward width. Copied wholesale from that twin and dead ever
        // since - surfaced by a fresh compile of the hardened flavour.
        let (d, ctx) = (self.hp.d_model, self.hp.n_text_ctx);
        let (heads, hd) = (self.hp.n_dec_heads, self.hp.head_dim);
        let n_enc = self.hp.n_audio_ctx;
        let n_layer = self.dec_layers.len();

        let mut cross_k = Vec::with_capacity(n_layer);
        let mut cross_v = Vec::with_capacity(n_layer);
        let mut self_k = Vec::with_capacity(n_layer);
        let mut self_v = Vec::with_capacity(n_layer);
        let kv = self.kv_dtype;
        let kvb = kv.bytes();
        for _ in 0..n_layer {
            cross_k.push(exec.alloc_u8(cap * n_enc * d * kvb)?);
            cross_v.push(exec.alloc_u8(cap * n_enc * d * kvb)?);
            self_k.push(exec.alloc_u8(cap * ctx * d * kvb)?);
            self_v.push(exec.alloc_u8(cap * ctx * d * kvb)?);
        }
        let cross_k_ptrs = exec.pointer_table(&cross_k)?;
        let cross_v_ptrs = exec.pointer_table(&cross_v)?;
        let batch = DecodeBatch {
            cap,
            kv,
            bytes: (2 * cap * n_enc * d + 2 * cap * ctx * d) as u64 * n_layer as u64 * kvb as u64,
            cross_k,
            cross_v,
            self_k,
            self_v,
            s: Self::alloc_scratch(&exec, cap, &self.hp, heads, hd, &self.tokens.suppressed())?,
            enc_stage: exec.alloc_f16(super::encoder::enc_batch_env() * n_enc * d)?,
            // the layer-batched landing scales with the admission batch cap
            //  - 246 MB per window at large-v3, the price of
            // running admission's 64 cross GEMMs as two
            enc_kv_all: exec.alloc(super::encoder::enc_batch_env() * n_enc * n_layer * d)?,
            cross_k_ptrs,
            cross_v_ptrs,
            d_slots: exec.to_device_u32(&[0u32; 8])?,
            enc_stream: exec.new_side_stream()?,
            enc_done: None,
            row: exec.alloc(self.hp.n_vocab)?,
            graphs: HashMap::new(),
            warmed: HashSet::new(),
            enc_inflight: false,
        };
        // side-slab bytes the kv_mib line never covered: the layer-batched
        // encoder-KV landing dominates (~492 MB at large-v3 bmax=2) and was
        // resident-but-unnamed
        let side_mib =
            (batch.enc_kv_all.len() * 4 + batch.enc_stage.len() * 2 + batch.row.len() * 4) as u64
                / (1 << 20);
        tracing::info!(
            slots = cap,
            kv_mib = batch.bytes / (1 << 20),
            enc_landing_mib = side_mib,
            kv = if kv == KvDtype::Fp8E4m3 {
                "fp8-e4m3"
            } else {
                "f16"
            },
            "whisper decode pool resident (cross + self K/V; enc_landing = \
             layer-batched encoder-KV + stage + row scratch)"
        );
        self.decode = Some(batch);
        // the encoder's working set is the other resident half - take it here
        // too, so the first request pays no allocation the hundredth wouldn't
        self.ensure_enc_scratch()
    }

    pub fn batch_cap(&self) -> usize {
        self.decode.as_ref().map_or(0, |b| b.cap)
    }

    /// Encode one 30 s window into `slot` and precompute every layer's
    /// cross-attention K/V from it. The mel is the CALLER's - the serving
    /// path computes it off this thread, which is the whole point of taking
    /// features rather than samples here.
    pub fn encode_into(&mut self, slot: usize, mel: &MelFeatures) -> Result<(), GpuModelError> {
        self.encode_into_batch(&[slot], &[mel])
    }

    /// Batched admission: encode up to `enc_batch_cap()` windows
    /// as one audio-major pass - window i lands in `slots[i]`. The 1-unit-
    /// fill encoder GEMMs (wo, fc2) re-fill at b>=2 and the per-GEMM fixed
    /// cost amortizes; the pass replays as one graph per batch width.
    pub fn encode_into_batch(
        &mut self,
        slots: &[usize],
        mels: &[&MelFeatures],
    ) -> Result<(), GpuModelError> {
        if slots.is_empty() || slots.len() != mels.len() {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: {} slots for {} windows",
                slots.len(),
                mels.len()
            )));
        }
        let cap = self.decode.as_ref().map_or(0, |st| st.cap);
        if slots.iter().any(|&s| s >= cap) {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: slot outside the {cap}-slot decode pool (prepare_batch first)"
            )));
        }
        // P38 ROOT CAUSE: consecutive admission passes in one
        // scheduler iteration (the start burst) raced - pass j's uploads on
        // the main stream rewrote the shared encoder scratch (mel, staging,
        // d_slots) while pass j-1's replay was still reading it on the side
        // stream. Serialize passes against each other here; steady state is
        // one pass per tick, so this sync is free when it matters.
        self.encode_sync()?;
        self.upload_windows(mels)?;
        let exec = self.exec.clone();
        let st = self.decode.as_mut().expect("checked above");
        let ids: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
        exec.upload_u32(&ids, &mut st.d_slots)?;
        // The replay overlaps the next decode tick from the side stream
        // (P38): the pass writes only its own slots' cross-K/V planes,
        // disjoint from every concurrently decoding slot. Eager/capture
        // passes stay on the main stream (capture is stream-bound). Either
        // way `enc_done` orders the admitted runs' merge one tick later.
        // discriminator (P38 audit): SAMESTREAM=1 runs the identical event/
        // holdback path with the replay on the MAIN stream - zero true
        // concurrency. Corrupt => scheduling logic; clean => kernel-level.
        let enc_stream =
            if paddock_models::dev_var!("PADDOCK_WHISPER_ENC_SAMESTREAM").is_ok_and(|v| v != "0") {
                exec.stream.clone()
            } else {
                st.enc_stream.clone()
            };
        let upl = exec.record_event()?;
        enc_stream
            .wait(&upl)
            .map_err(|e| GpuError::Driver(format!("whisper enc-stream wait: {e}")))?;
        // On by default since P40: the P38 corruption was the
        // decode band's mmaf arm racing the encoder's tc5p (P39 root cause -
        // stale W-ring slab reads, below the PTX contract), and overlap
        // ticks now replay a decode-graph variant captured with mmaf
        // declined (see `step_replay`), which the acceptance battery
        // measured WER-clean at multiplied request counts. `enc_overlap`
        // still refuses to overlap on a pack that cannot route.
        let replayed = if self.enc_overlap() {
            self.admit_replay(Some(&enc_stream))?
        } else {
            self.admit_replay(None)?
        };
        let done = if replayed {
            enc_stream
                .record_event(None)
                .map_err(|e| GpuError::Driver(format!("whisper enc event: {e}")))?
        } else {
            exec.record_event()?
        };
        self.decode.as_mut().expect("checked above").enc_done = Some(done);
        Ok(())
    }

    /// Whether the admission replay overlaps the decode tick from the side
    /// stream. On by default since the P40 acceptance battery (WER 2.09 with
    /// a token-identical error profile and zero bursts at 336-request c8 /
    /// 448-request c32 legs) - PADDOCK_WHISPER_ENC_OVERLAP=0 reverts.
    /// Requires the pack's
    /// slot-409 mmaf gate either way: overlap without the dual-graph route
    /// is the P38/P39 transcript-corruption config (mmaf × tc5p), so a pack
    /// that cannot route cannot overlap, however the env is set.
    pub fn enc_overlap(&self) -> bool {
        self.exec.has_f16_mmaf_gate()
            && paddock_models::dev_var!("PADDOCK_WHISPER_ENC_OVERLAP").map_or(true, |v| v != "0")
    }

    /// Tell the decode tick whether an admission replay is in flight on the
    /// side stream - the transcriber sets this before every step, and it is
    /// what routes the step onto the mmaf-off graph variant.
    pub fn set_enc_inflight(&mut self, on: bool) {
        if let Some(st) = self.decode.as_mut() {
            st.enc_inflight = on;
        }
    }

    /// Block until the last admission pass's GPU work is complete (P38) -
    /// called before its runs join the decode batch. A no-op when nothing
    /// is outstanding.
    pub fn encode_sync(&mut self) -> Result<(), GpuModelError> {
        if let Some(ev) = self.decode.as_mut().and_then(|st| st.enc_done.take()) {
            ev.synchronize()
                .map_err(|e| GpuError::Driver(format!("whisper encode sync: {e}")))?;
        }
        Ok(())
    }

    /// Encode + every layer's cross-attention K/V: the captured admission
    /// tick. Allocation-free and host-read-free by construction.
    pub(crate) fn admit_body(&mut self) -> Result<(), GpuModelError> {
        self.encode_body()?;
        let (n, d) = (self.hp.n_audio_ctx, self.hp.d_model);
        let exec = self.exec.clone();
        // the states never leave the device: the scratch's own landing is the
        // cross-K/V GEMM's input, so admission costs no copy of its own
        let enc = self.enc.as_ref().expect("encode_body allocated it");
        let b = enc.batch_staged();
        let states = enc.states();
        let st = self.decode.as_mut().expect("encode_into checked it");
        let n_layer = self.dec_layers.len();
        exec.convert_f32_f16(states, &mut st.enc_stage, b * n * d)?;
        // K and V come straight off the encoder states - no LN, no positions,
        // no mask. Every layer reads the same states, so the whole set runs
        // as one layer-batched GEMM per plane family plus one batched store
        // (64 GEMMs that each idled half the tc5p clusters, 806us,
        // -> 2 x 135). The batched pass widens both GEMMs to b*n rows and
        // the store fans rows out by audio. Whisper gives k_proj
        // no bias anywhere.
        exec.matvec_batch_f16(&self.cross_wk_all, &st.enc_stage, &mut st.enc_kv_all, b * n)?;
        exec.whisper_kv_store_slots(
            &st.enc_kv_all,
            None,
            &st.cross_k_ptrs,
            &st.d_slots,
            b * n,
            d,
            n_layer,
            n,
            st.kv,
            n,
        )?;
        exec.matvec_batch_f16(&self.cross_wv_all, &st.enc_stage, &mut st.enc_kv_all, b * n)?;
        exec.whisper_kv_store_slots(
            &st.enc_kv_all,
            Some(&self.cross_bv_all),
            &st.cross_v_ptrs,
            &st.d_slots,
            b * n,
            d,
            n_layer,
            n,
            st.kv,
            n,
        )?;
        Ok(())
    }

    /// One decode step for every active slot. `slots`/`tokens`/`pos` are
    /// parallel arrays in active order: slot id, the token to feed, and the
    /// position it lands at. Returns each row's greedy argmax and how sure the
    /// model was about it.
    ///
    /// `rules` is the timestamp grammar's per-row state - `[b, 2]` of
    /// `{flags, lowest allowed timestamp}` built by `ts_state` - or `None`
    /// when nothing in this batch asked for timestamps. Passing it costs one
    /// extra launch and a second captured graph; passing `None` leaves the
    /// plain-text lane byte for byte what it was.
    pub fn step_batch(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        pos: &[u32],
        rules: Option<&[u32]>,
    ) -> Result<StepOut, GpuModelError> {
        let b = slots.len();
        if b == 0 {
            return Ok(StepOut::default());
        }
        if tokens.len() != b || pos.len() != b {
            return Err(GpuModelError::Unsupported(
                "whisper: slots/tokens/pos must be the same length".into(),
            ));
        }
        if let Some(r) = rules
            && r.len() != b * 2
        {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: timestamp state is {} u32 for {b} rows (want {})",
                r.len(),
                b * 2
            )));
        }
        let exec = self.exec.clone();
        let (ctx, vocab) = (self.hp.n_text_ctx, self.hp.n_vocab);
        let st = self
            .decode
            .as_mut()
            .ok_or_else(|| GpuModelError::Unsupported("whisper: prepare_batch first".into()))?;
        if b > st.cap {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: {b} active rows over a {}-slot pool",
                st.cap
            )));
        }
        for (&s, &p) in slots.iter().zip(pos) {
            if s as usize >= st.cap {
                return Err(GpuModelError::Unsupported(format!(
                    "whisper: slot {s} out of pool"
                )));
            }
            if p as usize >= ctx {
                return Err(GpuModelError::Unsupported(format!(
                    "whisper: decoder position {p} reached the served context {ctx}"
                )));
            }
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: token {t} outside vocab {vocab}"
            )));
        }
        exec.upload_u32(slots, &mut st.s.d_slots)?;
        exec.upload_u32(tokens, &mut st.s.d_tok)?;
        exec.upload_u32(pos, &mut st.s.d_pos)?;
        if let Some(r) = rules {
            exec.upload_u32(r, &mut st.s.d_rules)?;
        }

        self.step_replay(b, rules.is_some())?;
        let pool = self.decode.as_ref().expect("pool");
        let next = self.exec.to_host_u32(&pool.s.d_next)?;
        let alt = self.exec.to_host_u32(&pool.s.d_alt)?;
        // `[log p(top1), p(<|nospeech|>), log p(top2), H2]` interleaved per row
        // - 16 bytes a slot, riding the same sync the token readback already
        // pays for. H2 (the row's Renyi-2 entropy) is written by the kernel and
        // not read here: it is second signal, and exposing it on
        // one lane would make the Studio's compare view asymmetric, since the
        // generative families' sampler only sees a top-k slice of the row.
        let stats = self.exec.to_host_len(&pool.s.d_stats, b * 4)?;
        Ok(StepOut {
            next: next[..b].to_vec(),
            logprob: stats.iter().step_by(4).copied().collect(),
            nospeech: stats.iter().skip(1).step_by(4).copied().collect(),
            runner_up: (0..b)
                // the kernel writes `vocab` for "there wasn't one" - the same
                // out-of-range convention the probe uses
                .map(|r| ((alt[r] as usize) < vocab).then(|| (alt[r], stats[r * 4 + 2])))
                .collect(),
        })
    }

    /// Replay the decode tick for `b` active rows, recording it first if this
    /// width has been warmed but not yet captured.
    ///
    /// The overlap route: a tick whose admission encode is still
    /// in flight on the side stream picks the `no_mmaf` graph variant - the
    /// pack's slot-409 gate declines the mmaf decode arm during that
    /// variant's warm and capture, because mmaf × the encoder's tc5p is the
    /// one kernel pairing that corrupts under true stream concurrency (P39).
    /// Only rows 5..=32 can elect mmaf at all, so narrower (and wider) ticks
    /// share one variant instead of capturing an identical twin.
    fn step_replay(&mut self, b: usize, rules: bool) -> Result<(), GpuModelError> {
        let pool = self.decode.as_ref().expect("prepare_batch ran");
        let no_mmaf = pool.enc_inflight && (5..=32).contains(&b) && self.exec.has_f16_mmaf_gate();
        let key = (b, rules, no_mmaf);
        if pool.graphs.contains_key(&key) {
            return pool.graphs[&key]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("whisper decode graph launch: {e}")).into());
        }
        let exec = self.exec.clone();
        if no_mmaf {
            exec.f16_mmaf_set(false);
        }
        let out = self.step_record(b, rules, key);
        if no_mmaf {
            exec.f16_mmaf_set(true);
        }
        out
    }

    /// The warm-then-capture half of `step_replay`, split out so the mmaf
    /// gate above wraps every launch path - including the eager warm run -
    /// and is restored even when recording fails.
    fn step_record(
        &mut self,
        b: usize,
        rules: bool,
        key: (usize, bool, bool),
    ) -> Result<(), GpuModelError> {
        let pool = self.decode.as_ref().expect("prepare_batch ran");
        if !pool.warmed.contains(&key) {
            self.step_body(b, rules, None)?;
            self.decode.as_mut().expect("pool").warmed.insert(key);
            return Ok(());
        }
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("whisper pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("whisper begin_capture: {e}")))?;
        let rec = self.step_body(b, rules, None);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("whisper end_capture: {e}")));
        rec?; // a record failure is only surfaceable after capture ends cleanly
        let graph = graph?
            .ok_or_else(|| GpuError::Driver("whisper decode capture produced no graph".into()))?;
        graph
            .launch()
            .map_err(|e| GpuError::Driver(format!("whisper decode graph launch: {e}")))?;
        self.decode
            .as_mut()
            .expect("pool")
            .graphs
            .insert(key, SendGraph(graph));
        Ok(())
    }

    /// The decode tick itself: fixed launch chain, every input already in
    /// device memory. Allocation-free by construction - capture forbids it.
    ///
    /// `dump` is the word-timing capture and is `None` for every
    /// served step; when it is set the tick additionally writes the alignment
    /// heads' cross-attention probabilities out. Nothing else about the step
    /// changes - the timing pass decodes exactly what serving decodes.
    pub(super) fn step_body(
        &mut self,
        b: usize,
        rules: bool,
        mut dump: Option<&mut super::timing::XattnDump>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (d, ctx, ffn) = (self.hp.d_model, self.hp.n_text_ctx, self.hp.dec_ffn);
        let (heads, hd) = (self.hp.n_dec_heads, self.hp.head_dim);
        let n_enc = self.hp.n_audio_ctx;
        let (eps, vocab) = (self.hp.eps, self.hp.n_vocab);
        let scale = 1.0 / (hd as f32).sqrt();
        let n_layer = self.dec_layers.len();
        let st = self.decode.as_mut().expect("prepare_batch ran");

        // embedding + LEARNED position, one gather-add per active row
        exec.whisper_embed_pos(
            &self.tok_embd.buf,
            &self.dec_pos,
            &st.s.d_tok,
            &st.s.d_pos,
            &mut st.s.x,
            d,
            b,
        )?;
        let l0 = &self.dec_layers[0].self_attn;
        exec.whisper_ln_f16(&st.s.x, &l0.ln.w, &l0.ln.b, &mut st.s.nrm, b, d, eps)?;

        for li in 0..n_layer {
            let layer = &self.dec_layers[li];
            // ---- causal self-attention ----
            let a = &layer.self_attn;
            exec.matvec_batch_f16(&a.wqkv, &st.s.nrm, &mut st.s.qkv, b)?;
            exec.whisper_qkv_split(
                &st.s.qkv,
                Some(&a.bq),
                Some(&a.bv),
                &mut st.s.q,
                &mut st.self_k[li],
                &mut st.self_v[li],
                &st.s.d_slots,
                &st.s.d_pos,
                d,
                ctx,
                b,
                st.kv,
            )?;
            // one query row against keys 0..=pos is the causal mask
            exec.whisper_dec_attn(
                &st.s.q,
                None,
                &st.self_k[li],
                &st.self_v[li],
                &st.s.d_slots,
                Some(&st.s.d_pos),
                &mut st.s.attn,
                Some(&mut st.s.part),
                ctx,
                0,
                1,
                heads,
                hd,
                b,
                scale,
                st.kv,
            )?;
            exec.matvec_batch_f16(&a.wo, &st.s.attn, &mut st.s.proj, b)?;
            let c = &layer.cross_attn;
            exec.whisper_res_ln_f16(
                &mut st.s.x,
                &st.s.proj,
                &a.bo,
                &c.ln.w,
                &c.ln.b,
                &mut st.s.nrm,
                b,
                d,
                eps,
            )?;

            // ---- cross-attention over the static encoder planes ----
            exec.matvec_batch_f16(&c.wq, &st.s.nrm, &mut st.s.q, b)?;
            // the word-timing read-out sits here, on the very query the
            // attention below is about to use - a re-derived query could drift
            // from the one the transcript came out of
            if let Some(dp) = dump.as_deref_mut() {
                dp.capture(
                    &exec,
                    li,
                    &st.s.q,
                    Some(&c.bq),
                    &st.cross_k[li],
                    &st.s.d_slots,
                    n_enc,
                    heads,
                    hd,
                    scale,
                    st.kv,
                )?;
            }
            exec.whisper_dec_attn(
                &st.s.q,
                Some(&c.bq),
                &st.cross_k[li],
                &st.cross_v[li],
                &st.s.d_slots,
                None,
                &mut st.s.attn,
                Some(&mut st.s.part),
                n_enc,
                n_enc,
                0,
                heads,
                hd,
                b,
                scale,
                st.kv,
            )?;
            exec.matvec_batch_f16(&c.wo, &st.s.attn, &mut st.s.proj, b)?;
            let m = &layer.mlp;
            exec.whisper_res_ln_f16(
                &mut st.s.x,
                &st.s.proj,
                &c.bo,
                &m.ln.w,
                &m.ln.b,
                &mut st.s.nrm,
                b,
                d,
                eps,
            )?;

            // ---- MLP; its residual seam also raises the next pre-norm
            // (the last layer's is the decoder's final norm) ----
            exec.matvec_batch_f16(&m.fc1_w, &st.s.nrm, &mut st.s.ff, b)?;
            exec.whisper_bias_gelu_f16(&st.s.ff, &m.fc1_b, &mut st.s.ffh, b, ffn)?;
            exec.matvec_batch_f16(&m.fc2_w, &st.s.ffh, &mut st.s.proj, b)?;
            let next = if li + 1 < n_layer {
                &self.dec_layers[li + 1].self_attn.ln
            } else {
                &self.dec_ln
            };
            exec.whisper_res_ln_f16(
                &mut st.s.x,
                &st.s.proj,
                &m.fc2_b,
                &next.w,
                &next.b,
                &mut st.s.nrm,
                b,
                d,
                eps,
            )?;
        }

        exec.matvec_batch_f16(&self.head, &st.s.nrm, &mut st.s.logits, b)?;
        if rules {
            // Whisper's own timestamp grammar, before the pick - the prompt
            // only offers the mode, this is what makes the model take it (see
            // the kernel's note: KB-Whisper's unconstrained argmax at the
            // first sampled position is `<|notimestamps|>` at p=0.794).
            exec.whisper_ts_rules(
                &mut st.s.logits,
                &st.s.d_rules,
                b,
                vocab,
                self.tokens.eot,
                self.tokens.no_timestamps,
                self.tokens.timestamp_begin,
                self.max_initial_ts,
            )?;
        }
        // The greedy pick, the runner-up and the confidence readouts all come
        // out of one log-sum-exp: the token is bit-identical to what
        // `argmax_rows` would have chosen (same tie rule), so this is not a
        // "confidence mode" that could move a transcript - it is the same pick
        // carrying four more numbers.
        exec.argmax_top2_rows(
            &st.s.logits,
            &mut st.s.d_next,
            &mut st.s.d_alt,
            &mut st.s.d_stats,
            self.tokens.nospeech,
            Some(&st.s.d_suppress),
            b,
            vocab,
        )?;
        Ok(())
    }

    /// One active row's full logits - the language-detection path only
    /// (whisper detects by argmax over the language tokens of the
    /// `<|startoftranscript|>` step, which no row-wise argmax can answer).
    pub fn logits_row(&mut self, b: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.n_vocab;
        let exec = self.exec.clone();
        let st = self
            .decode
            .as_mut()
            .ok_or_else(|| GpuModelError::Unsupported("whisper: prepare_batch first".into()))?;
        // `row` is resident scratch, not state - a mid-buffer host read needs
        // a contiguous source, and allocating one per call mid-serve is the
        // thing this pool exists to avoid
        exec.copy_region(&st.s.logits, b * vocab, &mut st.row, 0, vocab)?;
        Ok(exec.to_host_len(&st.row, vocab)?)
    }

    /// The bare language codes this checkpoint declares, in its own order -
    /// the exact set `language` may name and the only set detection can
    /// answer from. Read out of the file's `lang_to_id` map at load, never a
    /// baked table, which is what keeps a 99-language checkpoint from being
    /// scored against a 100-entry list (whisper.cpp's bug: its hardcoded
    /// table makes "Cantonese" resolve to `<|translate|>` on those models, so
    /// the translate logit enters the softmax as a language).
    pub fn languages(&self) -> Vec<String> {
        self.langs.iter().map(|(c, _)| c.clone()).collect()
    }

    /// The full language POSTERIOR for a window, from its
    /// `<|startoftranscript|>` logits - every language the checkpoint
    /// declares, sorted best first.
    ///
    /// This is whisper's own normalisation: every non-language token is
    /// masked to -inf and the softmax renormalised over what is left, so the
    /// probabilities sum to 1 over the CANDIDATES rather than over the vocab.
    /// `<|nospeech|>` is not a language token, so it is masked out with
    /// everything else - which is exactly why this can never abstain, and why
    /// silence comes back as a confident language (the VAD gate, not this
    /// function, is what answers that; language-identification §3.3).
    ///
    /// Keeping the whole distribution rather than the argmax is the change of
    /// and the reason is measured: whisper's argmax scores 83.9% on
    /// FLEURS while the oracle over its own 10-best scores 98.6%
    /// (arXiv 2409.18428). The right answer is nearly always in here; the
    /// unconstrained argmax over 99 candidates is what loses it.
    pub fn language_posterior(&self, logits: &[f32]) -> Result<Vec<LangProb>, GpuModelError> {
        super::posterior_over(&self.langs, logits).map_err(GpuModelError::Unsupported)
    }

    /// Resolve the language for a window from its `<|startoftranscript|>`
    /// logits: the top of the posterior above, which is the same token
    /// vLLM's `get_language_detection_prompt` path picks too. Kept because the
    /// probe examples and the serial `transcribe` want one answer, not a
    /// distribution - the serving path reads the posterior.
    pub fn detect_language(&self, logits: &[f32]) -> Result<(String, u32), GpuModelError> {
        let top = self
            .language_posterior(logits)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                GpuModelError::Unsupported("whisper: checkpoint has no language map".into())
            })?;
        Ok((top.code, top.id))
    }

    /// The two fixed prompt tokens that follow `<|startoftranscript|>` and
    /// the language token: `(transcribe, no_timestamps)`.
    ///
    /// `no_timestamps` is only fed when the caller did not ask for times -
    /// dropping it is exactly how whisper is told to emit its timestamp
    /// tokens, and it is also why timestamps are opt-in: the prompt change
    /// moves the decode, so a request that wants plain text keeps the prompt
    /// every WER gate was measured on.
    pub fn prompt_tail(&self) -> (u32, u32) {
        (self.tokens.transcribe, self.tokens.no_timestamps)
    }

    /// Transcribe a whole clip on one slot - the serial convenience the
    /// probe example and the load tests use. The serving path drives
    /// `encode_into` / `step_batch` directly so it can run slots together.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        lang: Option<&str>,
        max_tokens: usize,
    ) -> Result<(String, Vec<Vec<u32>>), GpuModelError> {
        self.prepare_batch(1)?;
        let mut lang_code = lang.unwrap_or_default().to_owned();
        let mut windows = Vec::new();
        let mut off = 0usize;
        while off < samples.len().max(1) {
            let end = (off + PAD_SAMPLES).min(samples.len());
            let mel = whisper_features(&samples[off..end]).map_err(GpuModelError::Unsupported)?;
            self.encode_into(0, &mel)?;
            // later windows inherit the first window's language: whisper
            // detects per window, and a mid-clip flip is a transcript bug,
            // not multilingual support
            let want = if lang_code.is_empty() {
                lang
            } else {
                Some(lang_code.as_str())
            };
            let (detected, ids) = self.transcribe_window(want, max_tokens)?;
            if lang_code.is_empty() {
                lang_code = detected;
            }
            windows.push(ids);
            off = end;
        }
        Ok((lang_code, windows))
    }

    /// Greedy-decode slot 0's already-encoded window. `lang` is a bare code
    /// ("sv"); `None` runs whisper's own detection.
    fn transcribe_window(
        &mut self,
        lang: Option<&str>,
        max_tokens: usize,
    ) -> Result<(String, Vec<u32>), GpuModelError> {
        let (sot, eot) = (self.tokens.sot, self.tokens.eot);
        let ctx = self.hp.n_text_ctx;
        let slots = [0u32];
        let mut pos = 0u32;
        let step = |me: &mut Self, tok: u32, p: &mut u32| -> Result<u32, GpuModelError> {
            // the serial convenience path is the plain-text lane only; the
            // serving scheduler is what drives the timestamp grammar
            let out = me.step_batch(&slots, &[tok], &[*p], None)?;
            *p += 1;
            Ok(out.next[0])
        };

        let mut argmax = step(self, sot, &mut pos)?;
        let (detected, lang_tok) = match lang {
            Some(code) => {
                let id = self.lang_token(code).ok_or_else(|| {
                    GpuModelError::Unsupported(format!(
                        "whisper: language {code:?} is not in this checkpoint's map"
                    ))
                })?;
                (code.to_owned(), id)
            }
            // the sot step's own logits already are the detector
            None => {
                let row = self.logits_row(0)?;
                self.detect_language(&row)?
            }
        };
        let (transcribe, no_ts) = self.prompt_tail();
        for tok in [lang_tok, transcribe, no_ts] {
            argmax = step(self, tok, &mut pos)?;
        }

        let mut out = Vec::new();
        for _ in 0..max_tokens {
            if argmax == eot {
                break;
            }
            out.push(argmax);
            // one step short of the served context means the next step would
            // have nowhere to write - stop cleanly instead of erroring
            if pos as usize + 1 >= ctx {
                break;
            }
            argmax = step(self, argmax, &mut pos)?;
        }
        Ok((detected, out))
    }

    /// Resolve a bare language code ("sv") to its token id, from the map the
    /// checkpoint itself declares.
    pub fn lang_token(&self, code: &str) -> Option<u32> {
        self.langs
            .iter()
            .find(|(c, _)| c == code)
            .map(|(_, id)| *id)
    }
}
