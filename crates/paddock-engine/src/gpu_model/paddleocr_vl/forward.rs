//! ERNIE-4.5-0.3B serial forward  - the parity spine the oracle
//! gate drives. Token-by-token with dense per-layer KV, granite's shape with
//! the granite scalars deleted and the plain rope swapped for sectioned
//! 3D M-RoPE:
//!
//! ```text
//! embd = tok_embd[t]                (or a projector row for image tokens)
//! layer: x += Wo·Attn(mrope(Q,K), V)     of RMSNorm(x)   scale 1/sqrt(128)
//!        x += down·SwiGLU(gate,up)       of RMSNorm(x)
//! head:  logits = output · RMSNorm(x)
//! ```
//!
//! Attention is STRICTLY causal in sequence order - the reference builds a
//! plain `create_causal_mask`, so image tokens attend raster-causally within
//! their own grid too (qwen35 now runs the same; it used to give a picture's
//! rows mutual visibility, which copied here would be a silent divergence).
//! KV rows sit at their sequence
//! index; the 3-axis M-RoPE positions ride separately per token.
//!
//! Because the spine feeds every prefill token through the same cached-decode
//! path a generated token takes, the oracle gate's 463-row multimodal probe
//! is a 463-step teacher-forced decode check - there is no separate prefill
//! graph to cross-validate yet. The batched lanes come with the serving surface.

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use paddock_kernels::reference::ops::YarnRope;

use super::load::GpuPaddleOcrVl;
use crate::gpu_model::gpt_oss::GpuModelError;

/// Merged (post 2×2) grid extents of one image, in DECODER tokens -
/// `VisionOutput { ny, nx }` hands exactly these.
#[derive(Debug, Clone, Copy)]
pub struct MmGrid {
    pub ny: usize,
    pub nx: usize,
}

/// The three M-RoPE axes for a token run, plus where the sequence continues.
pub struct Positions {
    pub t: Vec<u32>,
    pub h: Vec<u32>,
    pub w: Vec<u32>,
    /// max over all axes/tokens + 1 - decode steps count on from here on all
    /// three axes (the reference's rope_deltas rule, asserted against a full
    /// recompute in the oracle dump).
    pub next: u32,
}

/// Port of the reference's `get_rope_index` for image-bearing sequences
/// (images only - no video; t is always 1). Text tokens advance one position
/// on all three axes; an image block puts t at its base, h/w at base + grid
/// coordinate, and the sequence resumes at base + max(nx, ny). Consumes
/// grids in order; refuses when the image-token runs don't tile the grids.
pub fn build_positions(
    ids: &[u32],
    image_token: u32,
    grids: &[MmGrid],
) -> Result<Positions, GpuModelError> {
    let mut p = Positions {
        t: Vec::with_capacity(ids.len()),
        h: Vec::with_capacity(ids.len()),
        w: Vec::with_capacity(ids.len()),
        next: 0,
    };
    let bad = |msg: String| GpuModelError::Unsupported(format!("paddleocr-vl positions: {msg}"));
    let mut pos = 0u32;
    let mut img = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] != image_token {
            p.t.push(pos);
            p.h.push(pos);
            p.w.push(pos);
            pos += 1;
            i += 1;
            continue;
        }
        let g = grids
            .get(img)
            .copied()
            .ok_or_else(|| bad(format!("image tokens at {i} but only {img} grids given")))?;
        img += 1;
        let n = g.ny * g.nx;
        if ids[i..].len() < n || ids[i..i + n].iter().any(|&t| t != image_token) {
            return Err(bad(format!(
                "image {img} wants a run of {n} image tokens at {i} - the prompt has fewer"
            )));
        }
        let base = pos;
        for j in 0..n {
            p.t.push(base);
            p.h.push(base + (j / g.nx) as u32);
            p.w.push(base + (j % g.nx) as u32);
        }
        pos = base + g.nx.max(g.ny) as u32;
        i += n;
    }
    if img != grids.len() {
        return Err(bad(format!(
            "{} grids given, {img} image runs found",
            grids.len()
        )));
    }
    p.next = pos;
    Ok(p)
}

/// Per-sequence decode state: dense per-layer KV (18 × [max_ctx, 256]),
/// sequence position and the M-RoPE continuation counter - the two diverge
/// as soon as an image is in the prefix.
pub(crate) struct DecodeState {
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    /// next KV row = tokens consumed so far.
    pub pos: usize,
    /// next M-RoPE position (all three axes - text continuation).
    pub mrope_next: u32,
    pub d_token: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    /// [4, 1] axis-major (t, h, w, 0) for the current token.
    pub d_mrope: CudaSlice<u32>,
    /// constant [0] - slot 0.
    pub d_slots: CudaSlice<u32>,
}

/// Decode-step scratch, allocated once.
pub(crate) struct Scratch {
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    /// `alloc_no_sinks` (-1e30): ERNIE has no attention sinks, and a zeroed
    /// buffer would inject a phantom softmax term (granite's burn).
    pub d_sinks: CudaSlice<f32>,
    pub d_ffn_gate: CudaSlice<f32>,
    pub d_ffn_up: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
}

/// Where one token's input row comes from.
#[derive(Clone, Copy)]
enum RowSource<'a> {
    Token(u32),
    /// Row `row` of a device plane of projector outputs `[n, n_embd]`.
    Image {
        embd: &'a CudaSlice<f32>,
        row: usize,
    },
}

/// Full per-row taps for the oracle gate - mirrors the dump script's hooks:
/// post-splice input embeddings, chosen layer outputs (post both residuals),
/// the final-norm rows, and the last row's logits.
pub struct DecTaps {
    pub embd: Vec<f32>,
    pub layers: HashMap<usize, Vec<f32>>,
    pub norm: Vec<f32>,
    pub last_logits: Vec<f32>,
}

impl GpuPaddleOcrVl {
    fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() && self.scratch.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let (mut kv_k, mut kv_v) = (
            Vec::with_capacity(hp.n_layer),
            Vec::with_capacity(hp.n_layer),
        );
        for _ in 0..hp.n_layer {
            kv_k.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
            kv_v.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
        }
        self.decode = Some(DecodeState {
            kv_k,
            kv_v,
            pos: 0,
            mrope_next: 0,
            d_token: e.alloc_u32(1)?,
            d_pos: e.alloc_u32(1)?,
            d_mrope: e.alloc_u32(4)?,
            d_slots: e.alloc_u32(1)?, // zeroed -> slot 0
        });
        self.scratch = Some(Scratch {
            d_x: e.alloc(hp.n_embd)?,
            d_xn: e.alloc(hp.n_embd)?,
            d_q: e.alloc(hp.n_head * hp.head_dim)?,
            d_k: e.alloc(kv_dim)?,
            d_v: e.alloc(kv_dim)?,
            d_attn: e.alloc(hp.n_head * hp.head_dim)?,
            d_proj: e.alloc(hp.n_embd)?,
            d_sinks: e.alloc_no_sinks(hp.n_head)?,
            d_ffn_gate: e.alloc(hp.n_ff)?,
            d_ffn_up: e.alloc(hp.n_ff)?,
            d_logits: e.alloc(hp.n_vocab)?,
        });
        Ok(())
    }

    pub fn reset_decode(&mut self) {
        if let Some(ds) = self.decode.as_mut() {
            ds.pos = 0;
            ds.mrope_next = 0;
            // KV needs no clearing: every attention read is position-bounded.
        }
    }

    /// One token through the stack. `mrope` is (t, h, w); the KV row is the
    /// running sequence position. Leaves the final hidden state (pre
    /// output-norm) in `scratch.d_x`; logits are a separate call so tap-less
    /// prefill rows skip the head GEMV.
    fn step_row(&mut self, src: RowSource<'_>, mrope: [u32; 3]) -> Result<(), GpuModelError> {
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_heads, n_kv_heads, head_dim) =
            (hp.n_embd, hp.n_head, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv_heads * head_dim;
        let (eps, n_ff) = (hp.eps, hp.n_ff);
        let scale = 1.0 / (head_dim as f32).sqrt();
        // ext_factor 0 => plain rope through the yarn parameterization; the
        // sectioned axis walk happens in the mrope kernel itself.
        let yarn = YarnRope::new(
            hp.n_rot,
            hp.rope_base,
            1.0,
            hp.n_ctx_train,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();
        let sections = hp.sections;
        let n_rot = hp.n_rot;
        let kv_dtype = self.kv_dtype;
        let layers = std::mem::take(&mut self.layers);
        let out = (|| -> Result<(), GpuModelError> {
            let sc = self.scratch.as_mut().expect("scratch");
            let ds = self.decode.as_mut().expect("decode");
            if ds.pos >= self.max_ctx {
                return Err(GpuModelError::Unsupported(format!(
                    "context full: {} tokens at max_ctx {}",
                    ds.pos, self.max_ctx
                )));
            }
            let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            exec.stream
                .memcpy_htod(&[ds.pos as u32], &mut ds.d_pos)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(&[mrope[0], mrope[1], mrope[2], 0], &mut ds.d_mrope)
                .map_err(drv)?;

            match src {
                RowSource::Token(t) => {
                    exec.stream
                        .memcpy_htod(&[t], &mut ds.d_token)
                        .map_err(drv)?;
                    exec.embed_gather_plane(
                        &self.tok_embd,
                        &ds.d_token,
                        &mut sc.d_x,
                        embd,
                        1,
                        1.0,
                    )?;
                }
                RowSource::Image { embd: plane, row } => {
                    exec.copy_region(plane, row * embd, &mut sc.d_x, 0, embd)?;
                }
            }

            for (li, layer) in layers.iter().enumerate() {
                exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
                layer.wq.gemv_exact(&exec, &sc.d_xn, &mut sc.d_q)?;
                layer.wk.gemv_exact(&exec, &sc.d_xn, &mut sc.d_k)?;
                layer.wv.gemv_exact(&exec, &sc.d_xn, &mut sc.d_v)?;
                exec.mrope(
                    &mut sc.d_q,
                    &ds.d_mrope,
                    1,
                    n_heads,
                    head_dim,
                    n_rot,
                    yarn,
                    sections,
                )?;
                exec.mrope(
                    &mut sc.d_k,
                    &ds.d_mrope,
                    1,
                    n_kv_heads,
                    head_dim,
                    n_rot,
                    yarn,
                    sections,
                )?;
                exec.kv_append_batch(
                    &sc.d_k,
                    &mut ds.kv_k[li],
                    &ds.d_pos,
                    Some(&ds.d_slots),
                    kv_dim,
                    self.max_ctx,
                    1,
                    kv_dtype,
                )?;
                exec.kv_append_batch(
                    &sc.d_v,
                    &mut ds.kv_v[li],
                    &ds.d_pos,
                    Some(&ds.d_slots),
                    kv_dim,
                    self.max_ctx,
                    1,
                    kv_dtype,
                )?;
                // window 0 = full attention, strictly causal via d_pos
                exec.attn_decode_batch(
                    &sc.d_q,
                    &ds.kv_k[li],
                    &ds.kv_v[li],
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &ds.d_pos,
                    Some(&ds.d_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    self.max_ctx,
                    kv_dim,
                    0,
                    1,
                    scale,
                    kv_dtype,
                )?;
                layer.wo.gemv_exact(&exec, &sc.d_attn, &mut sc.d_proj)?;
                exec.add(&mut sc.d_x, &sc.d_proj, embd)?;

                exec.rmsnorm_batch(&sc.d_x, &layer.ffn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
                layer.gate.gemv_exact(&exec, &sc.d_xn, &mut sc.d_ffn_gate)?;
                layer.up.gemv_exact(&exec, &sc.d_xn, &mut sc.d_ffn_up)?;
                exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, n_ff)?;
                layer
                    .down
                    .gemv_exact(&exec, &sc.d_ffn_gate, &mut sc.d_proj)?;
                exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
            }
            ds.pos += 1;
            Ok(())
        })();
        self.layers = layers;
        out
    }

    /// Norm the current hidden row and run the untied head; returns logits.
    fn head(&mut self) -> Result<Vec<f32>, GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let sc = self.scratch.as_mut().expect("scratch");
        exec.rmsnorm_batch(
            &sc.d_x,
            &self.output_norm.buf,
            &mut sc.d_xn,
            hp.n_embd,
            hp.eps,
            1,
        )?;
        self.lm_head.gemv_exact(&exec, &sc.d_xn, &mut sc.d_logits)?;
        Ok(exec.to_host(&sc.d_logits)?)
    }

    /// Prefill `ids` from a reset state. Image-token rows read consecutive
    /// rows of `image_embd` (the projector output plane, `[n, n_embd]` on
    /// device - the splice the reference does with masked_scatter); `grids`
    /// are the merged extents in order. With `tap_layers` the walk also
    /// collects the oracle gate's full per-row taps (slow - host readback per
    /// row per tap); pass `&[]` to skip everything but the final logits.
    pub fn prefill_taps(
        &mut self,
        ids: &[u32],
        image_token: u32,
        image_embd: Option<&CudaSlice<f32>>,
        grids: &[MmGrid],
        tap_layers: &[usize],
    ) -> Result<DecTaps, GpuModelError> {
        let pos = build_positions(ids, image_token, grids)?;
        self.reset_decode();
        self.ensure_decode()?;
        let embd = self.hp.n_embd;
        let want_taps = !tap_layers.is_empty();
        let mut taps = DecTaps {
            embd: Vec::new(),
            layers: tap_layers.iter().map(|&l| (l, Vec::new())).collect(),
            norm: Vec::new(),
            last_logits: Vec::new(),
        };
        let mut img_row = 0usize;
        for (i, &tok) in ids.iter().enumerate() {
            let src = if tok == image_token {
                let plane = image_embd.ok_or_else(|| {
                    GpuModelError::Unsupported(
                        "paddleocr-vl: image tokens in the prompt but no image embeddings".into(),
                    )
                })?;
                let row = img_row;
                img_row += 1;
                RowSource::Image { embd: plane, row }
            } else {
                RowSource::Token(tok)
            };
            let mrope = [pos.t[i], pos.h[i], pos.w[i]];
            if want_taps {
                // capture the input row (post-splice inputs_embeds) before
                // the stack overwrites d_x
                self.fill_row_for_tap(src)?;
                let sc = self.scratch.as_ref().expect("scratch");
                taps.embd.extend(self.exec.to_host(&sc.d_x)?);
                self.step_row_tapped(src, mrope, &mut taps)?;
                let exec = self.exec.clone();
                let (eps, on) = (self.hp.eps, &self.output_norm.buf);
                let sc = self.scratch.as_mut().expect("scratch");
                exec.rmsnorm_batch(&sc.d_x, on, &mut sc.d_xn, embd, eps, 1)?;
                taps.norm.extend(exec.to_host(&sc.d_xn)?);
            } else {
                self.step_row(src, mrope)?;
            }
        }
        self.decode.as_mut().expect("decode").mrope_next = pos.next;
        taps.last_logits = self.head()?;
        Ok(taps)
    }

    /// The tap-instrumented twin of `step_row`. Kept adjacent and shaped
    /// identically deliberately: any edit to one must land in the other (the
    /// pair is small enough that a shared closure-heavy core would obscure
    /// more than it deduplicates).
    fn step_row_tapped(
        &mut self,
        src: RowSource<'_>,
        mrope: [u32; 3],
        taps: &mut DecTaps,
    ) -> Result<(), GpuModelError> {
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_heads, n_kv_heads, head_dim) =
            (hp.n_embd, hp.n_head, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv_heads * head_dim;
        let (eps, n_ff) = (hp.eps, hp.n_ff);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let yarn = YarnRope::new(
            hp.n_rot,
            hp.rope_base,
            1.0,
            hp.n_ctx_train,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();
        let sections = hp.sections;
        let n_rot = hp.n_rot;
        let kv_dtype = self.kv_dtype;
        let layers = std::mem::take(&mut self.layers);
        let out = (|| -> Result<(), GpuModelError> {
            let sc = self.scratch.as_mut().expect("scratch");
            let ds = self.decode.as_mut().expect("decode");
            if ds.pos >= self.max_ctx {
                return Err(GpuModelError::Unsupported("context full".into()));
            }
            let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
            exec.stream
                .memcpy_htod(&[ds.pos as u32], &mut ds.d_pos)
                .map_err(drv)?;
            exec.stream
                .memcpy_htod(&[mrope[0], mrope[1], mrope[2], 0], &mut ds.d_mrope)
                .map_err(drv)?;
            match src {
                RowSource::Token(t) => {
                    exec.stream
                        .memcpy_htod(&[t], &mut ds.d_token)
                        .map_err(drv)?;
                    exec.embed_gather_plane(
                        &self.tok_embd,
                        &ds.d_token,
                        &mut sc.d_x,
                        embd,
                        1,
                        1.0,
                    )?;
                }
                RowSource::Image { embd: plane, row } => {
                    exec.copy_region(plane, row * embd, &mut sc.d_x, 0, embd)?;
                }
            }
            for (li, layer) in layers.iter().enumerate() {
                exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
                layer.wq.gemv_exact(&exec, &sc.d_xn, &mut sc.d_q)?;
                layer.wk.gemv_exact(&exec, &sc.d_xn, &mut sc.d_k)?;
                layer.wv.gemv_exact(&exec, &sc.d_xn, &mut sc.d_v)?;
                exec.mrope(
                    &mut sc.d_q,
                    &ds.d_mrope,
                    1,
                    n_heads,
                    head_dim,
                    n_rot,
                    yarn,
                    sections,
                )?;
                exec.mrope(
                    &mut sc.d_k,
                    &ds.d_mrope,
                    1,
                    n_kv_heads,
                    head_dim,
                    n_rot,
                    yarn,
                    sections,
                )?;
                exec.kv_append_batch(
                    &sc.d_k,
                    &mut ds.kv_k[li],
                    &ds.d_pos,
                    Some(&ds.d_slots),
                    kv_dim,
                    self.max_ctx,
                    1,
                    kv_dtype,
                )?;
                exec.kv_append_batch(
                    &sc.d_v,
                    &mut ds.kv_v[li],
                    &ds.d_pos,
                    Some(&ds.d_slots),
                    kv_dim,
                    self.max_ctx,
                    1,
                    kv_dtype,
                )?;
                exec.attn_decode_batch(
                    &sc.d_q,
                    &ds.kv_k[li],
                    &ds.kv_v[li],
                    &sc.d_sinks,
                    &mut sc.d_attn,
                    &ds.d_pos,
                    Some(&ds.d_slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    self.max_ctx,
                    kv_dim,
                    0,
                    1,
                    scale,
                    kv_dtype,
                )?;
                layer.wo.gemv_exact(&exec, &sc.d_attn, &mut sc.d_proj)?;
                exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
                exec.rmsnorm_batch(&sc.d_x, &layer.ffn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
                layer.gate.gemv_exact(&exec, &sc.d_xn, &mut sc.d_ffn_gate)?;
                layer.up.gemv_exact(&exec, &sc.d_xn, &mut sc.d_ffn_up)?;
                exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, n_ff)?;
                layer
                    .down
                    .gemv_exact(&exec, &sc.d_ffn_gate, &mut sc.d_proj)?;
                exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
                if let Some(sink) = taps.layers.get_mut(&li) {
                    sink.extend(exec.to_host(&sc.d_x)?);
                }
            }
            ds.pos += 1;
            Ok(())
        })();
        self.layers = layers;
        out
    }

    /// Fill `d_x` with the row's input embedding only (the `embd` tap).
    fn fill_row_for_tap(&mut self, src: RowSource<'_>) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        match src {
            RowSource::Token(t) => {
                exec.stream
                    .memcpy_htod(&[t], &mut ds.d_token)
                    .map_err(drv)?;
                exec.embed_gather_plane(&self.tok_embd, &ds.d_token, &mut sc.d_x, embd, 1, 1.0)?;
            }
            RowSource::Image { embd: plane, row } => {
                exec.copy_region(plane, row * embd, &mut sc.d_x, 0, embd)?;
            }
        }
        Ok(())
    }

    /// Greedy continuation from a prefilled state: argmax(`last_logits`),
    /// then feed each pick back at the max+1 delta positions (all three axes
    /// - the reference's rope_deltas rule). Returns the chosen ids, stopping
    ///   after `eos`.
    pub fn greedy(
        &mut self,
        last_logits: &[f32],
        steps: usize,
        eos: u32,
    ) -> Result<Vec<u32>, GpuModelError> {
        if self.decode.is_none() {
            return Err(GpuModelError::Unsupported("greedy before prefill".into()));
        }
        let mut out = Vec::with_capacity(steps);
        let mut logits = last_logits.to_vec();
        for _ in 0..steps {
            let tok = argmax(&logits);
            out.push(tok);
            if tok == eos {
                break;
            }
            let mp = self.decode.as_ref().expect("decode").mrope_next;
            self.step_row(RowSource::Token(tok), [mp, mp, mp])?;
            self.decode.as_mut().expect("decode").mrope_next = mp + 1;
            logits = self.head()?;
        }
        Ok(out)
    }
}

impl GpuPaddleOcrVl {
    /// One serial token: sequence position from the KV walk, M-RoPE position
    /// from the text-continuation counter (they diverge once an image is in
    /// the prefix). Returns the full logits row.
    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_decode()?;
        let mp = self.decode.as_ref().expect("decode").mrope_next;
        self.step_row(RowSource::Token(token), [mp, mp, mp])?;
        self.decode.as_mut().expect("decode").mrope_next = mp + 1;
        self.head()
    }
}

fn gen_err(e: GpuModelError) -> crate::generator::GenError {
    match e {
        GpuModelError::PoolExhausted => crate::generator::GenError::PoolExhausted,
        other => crate::generator::GenError::Backend(other.to_string()),
    }
}

/// The serving contract: the batched paged-KV lane (batch.rs/chunked.rs
/// - enable_batch, slot prefills, mixed ticks, mm slots, encoder budget) with
///   the exclusive serial path kept intact as the no-paged-kv fallback and
///   the parity reference. The deepseek-ocr Generator arm, minus spec (no
///   drafter exists for this family - no spec legs ever).
impl crate::generator::Generator for GpuPaddleOcrVl {
    /// The live mempool counter every family publishes as its device ledger
    /// (`/api/stats` `model_mem`): without it the manager had nothing to
    /// reconcile this model's NVML figure against.
    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn reset(&mut self) {
        self.reset_decode();
        if let Some(bs) = self.batch.as_mut() {
            bs.slot0_pos = 0;
            // fresh sequence on the serial surface: slot 0 ropes at delta 0
            // until its next prefill fixes a new one
            bs.mrope_delta[0] = 0;
        }
        // KV needs no clearing: every attention read is position-bounded.
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, crate::generator::GenError> {
        // Batched engine: serial-surface callers (warmup, serial loop) decode
        // through slot 0 of the batch lane - its M-RoPE delta from the last
        // prefill keeps the positions honest past any image.
        if self.batch.is_some() {
            let pos = self.batch.as_ref().expect("batch").slot0_pos;
            if pos >= self.max_ctx {
                return Err(crate::generator::GenError::Backend(format!(
                    "context full: {pos} rows at max_ctx {}",
                    self.max_ctx
                )));
            }
            self.batch_step_slots(&[token], &[pos as u32], &[0])
                .map_err(gen_err)?;
            self.batch.as_mut().expect("batch").slot0_pos = pos + 1;
            return self.read_batch_logits(1).map_err(gen_err);
        }
        self.forward_token(token).map_err(gen_err)
    }

    fn vocab(&self) -> usize {
        self.hp.n_vocab
    }

    fn max_context(&self) -> usize {
        self.max_ctx
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        if let Some(bs) = self.batch.as_ref() {
            return Some(bs.kv_bytes);
        }
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        Some(if self.decode.is_some() {
            (2 * self.hp.n_layer * self.max_ctx * kv_dim * self.kv_dtype.bytes()) as u64
        } else {
            0
        })
    }

    // ── the batched serving lane (batch.rs) ─────────────────────────────────

    /// Real error on a pack without paged KV - the service's documented
    /// "genuinely can't" signal (its single-user election treats any Ok as
    /// "the batch state is built"), which routes serving onto the
    /// exclusive serial lane.
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, crate::generator::GenError> {
        self.enable_batch(max_batch).map_err(gen_err)
    }

    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        self.batch_step(tokens, positions).map_err(gen_err)?;
        self.read_batch_logits(tokens.len()).map_err(gen_err)
    }

    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        let logits = self.forward_prefill(slot, tokens).map_err(gen_err)?;
        if slot == 0 {
            self.batch.as_mut().expect("batch enabled").slot0_pos = tokens.len();
        }
        Ok(logits)
    }

    /// Single-stream bulk prefill. Batched: slot 0. Serial: the spine's
    /// whole-prompt pass (prefill_taps without taps).
    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        if self.batch.is_some() {
            return crate::generator::Generator::forward_prefill(self, 0, tokens);
        }
        let taps = self
            .prefill_taps(tokens, super::IMAGE_TOKEN, None, &[], &[])
            .map_err(gen_err)?;
        Ok(taps.last_logits)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots(occupied);
        // the chunked queue and the encode queue hold their own slot lists -
        // a dead encode entry still REPORTS at its turn, it just queues
        // nothing (the scheduler's encoding set only clears on reports)
        self.chunk_release(occupied);
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        self.pool_free_blocks()
    }

    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.take_prefill_reused(slot)
    }

    // ── stall-free chunked prefill + the encoder budget (chunked.rs) ────────

    fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some()
    }

    // the mixed tick's FIFO over the queue, row-exact from each cursor
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.cursor, c.rows.len() - c.cursor))
            .collect()
    }

    // plan_chunk's cap under the pass's row capacity (the decode rows share it)
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.batch.as_ref().map_or(0, |bs| {
            crate::gpu_model::granite::batch::pf_rows().min(bs.cap.saturating_sub(decode_rows))
        })
    }

    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), crate::generator::GenError> {
        self.prefill_begin_impl(slot, tokens).map_err(gen_err)
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), crate::generator::GenError> {
        self.forward_mixed_impl(decodes, budget).map_err(gen_err)
    }

    fn supports_device_sampling(&self) -> bool {
        self.supports_device_sampling_impl()
    }

    /// OCR requests are overwhelmingly greedy; when a no-repeat-ngram guard
    /// is armed the scheduler grants Device(Greedy) per tick only where the
    /// guard would ban nothing, and there is no decode-pipe lookahead here
    /// for that check to go stale on.
    fn device_greedy_ngram_ok(&self) -> bool {
        true
    }

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, crate::generator::GenError> {
        self.forward_batch_sampled_impl(tokens, positions, plans)
            .map_err(gen_err)
    }

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        crate::generator::GenError,
    > {
        self.forward_mixed_sampled_impl(decodes, budget, plans, fin_plans)
            .map_err(gen_err)
    }

    /// Image prompts ride the same stall-free queue text does; the tower
    /// encode spends one call (one image) per tick under the encoder budget.
    /// A/B pin: PADDOCK_MM_NO_CHUNK=1 restores the blocking wave path.
    fn supports_chunked_multimodal(&self) -> bool {
        self.vision.is_some()
            && self.batch.is_some()
            && paddock_models::dev_var_os!("PADDOCK_MM_NO_CHUNK").is_none()
    }

    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        self.prefill_begin_multimodal_impl(items)
    }

    fn encode_step(&mut self) -> Vec<(usize, crate::generator::MmAdmit)> {
        self.encode_step_impl()
    }

    fn encoding_pending(&self) -> bool {
        !self.enc.is_empty()
    }

    // ── vision (multimodal.rs) ──────────────────────────────────────────────

    /// Image requests ride ordinary batch slots - never the exclusive
    /// drain-the-server path - once the paged lane is up. On this family the
    /// prompt is mostly picture, so slot-riding plus the content-keyed radix
    /// is the whole serving story.
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some() && self.batch.is_some()
    }

    fn vision_budget(&self) -> Option<crate::generator::VisionBudget> {
        self.vision_budget_impl()
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), crate::generator::GenError> {
        let out = self
            .multimodal_prefill_slot(slot, chunks)
            .map_err(gen_err)?;
        if slot == 0 {
            self.batch.as_mut().expect("batch enabled").slot0_pos = out.1;
        }
        Ok(out)
    }

    /// The scheduler's whole admission wave in one call - every slot's host
    /// preprocessing runs on worker threads while the GPU executes the
    /// earlier slots.
    fn forward_prefill_multimodal_batch(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), crate::generator::GenError>)> {
        let out: Vec<_> = self
            .multimodal_prefill_wave(items)
            .into_iter()
            .map(|(k, r)| (k, r.map_err(gen_err)))
            .collect();
        for (k, r) in &out {
            if *k == 0
                && let Ok((_, n)) = r
            {
                self.batch.as_mut().expect("batch enabled").slot0_pos = *n;
            }
        }
        out
    }

    /// The exclusive-path entry. Batched: route through slot 0 of the batch
    /// lane. Serial (no paged KV): the spine path - reset + whole-prompt
    /// splice prefill, the parity reference.
    fn forward_multimodal(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, crate::generator::GenError> {
        if self.vision.is_none() {
            return Ok(None);
        }
        if self.batch.is_some() {
            let out = crate::generator::Generator::forward_prefill_multimodal(self, 0, chunks)?;
            return Ok(Some(out));
        }
        self.forward_multimodal_impl(chunks)
            .map(Some)
            .map_err(gen_err)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text-only degenerates to arange on all axes.
    #[test]
    fn text_positions_are_arange() {
        let p = build_positions(&[5, 9, 7, 7], 100295, &[]).unwrap();
        assert_eq!(p.t, [0, 1, 2, 3]);
        assert_eq!(p.h, [0, 1, 2, 3]);
        assert_eq!(p.w, [0, 1, 2, 3]);
        assert_eq!(p.next, 4);
    }

    /// The reference docstring's own worked example, adapted to one image:
    /// 3 text, a 2x3 (ny=2, nx=3) image, 2 text. Image base = 3; text resumes
    /// at 3 + max(2,3) = 6.
    #[test]
    fn image_positions_follow_the_reference_recipe() {
        const IMG: u32 = 100295;
        let ids = [1, 2, 3, IMG, IMG, IMG, IMG, IMG, IMG, 4, 5];
        let p = build_positions(&ids, IMG, &[MmGrid { ny: 2, nx: 3 }]).unwrap();
        assert_eq!(p.t, [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7]);
        assert_eq!(p.h, [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7]);
        assert_eq!(p.w, [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7]);
        assert_eq!(p.next, 8);
    }

    #[test]
    fn grid_mismatch_is_refused() {
        const IMG: u32 = 100295;
        // 5 image tokens for a 2x3 grid - must refuse, not misalign silently
        let ids = [1, IMG, IMG, IMG, IMG, IMG, 2];
        assert!(build_positions(&ids, IMG, &[MmGrid { ny: 2, nx: 3 }]).is_err());
        // an image run with no grid at all
        assert!(build_positions(&[IMG], IMG, &[]).is_err());
    }
}
