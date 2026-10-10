//! Qwen3.8 DFlash2: exact GGUF five-layer block drafter, noncausal windowed
//! MPP attention, grouped dynamic convolutions and rank-256 path selection.
//! Shared target embedding/head; window-bounded conditioning rings include
//! independent cached-prefix checkpoints. Rejected future rows are masked.
use super::*;
use objc2_metal::MTLBuffer;
use paddock_models::{gguf::Value, mapped::MappedGguf};

#[cfg(test)]
thread_local! {
    pub(super) static FULL_DRAFT_HEAD_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

const DH: usize = 32;
const DK: usize = 8;
const DIM: usize = 128;
const TAPS: [usize; 5] = [6, 20, 34, 48, 62];

#[cfg(test)]
#[path = "dflash_prefix_tests.rs"]
mod prefix_tests;

struct Conv {
    base: Weight,
    proj: Weight,
}
struct DraftLayer {
    norm: Weight,
    post: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    qn: Weight,
    kn: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    ac: Conv,
    fc: Conv,
    keys: Buffer,
    values: Buffer,
}
pub(super) struct Dflash {
    fc: Weight,
    enc_norm: Weight,
    out_norm: Weight,
    selector: Weight,
    pred: Weight,
    succ: Weight,
    layers: Vec<DraftLayer>,
    pages: Buffer,
    ring_tokens: usize,
    pub(super) taps: Buffer,
    conditioning: Buffer,
    z: Buffer,
    x: Buffer,
    norm: Buffer,
    delta: Buffer,
    conv: Buffer,
    coeff: Buffer,
    q: Buffer,
    qn: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    gate: Buffer,
    up: Buffer,
    gemm: Buffer,
    logits: Buffer,
    top: Buffer,
    top_parts: Buffer,
    selector_h: Buffer,
    out: Buffer,
}

impl Qwen35 {
    pub(super) fn dflash_cold_spans(&self, slot: usize) -> Vec<crate::offload::Span<'_>> {
        let Some(d) = &self.dflash else {
            return Vec::new();
        };
        let bytes = d.ring_tokens * DK * DIM * 2;
        d.layers
            .iter()
            .flat_map(|l| {
                [
                    (&l.keys, slot * bytes, bytes),
                    (&l.values, slot * bytes, bytes),
                ]
            })
            .collect()
    }
    pub(super) fn reserve_dflash_rows(&mut self, n: usize) -> Result<()> {
        if let Some(d) = &mut self.dflash {
            let a = |width| self.device.alloc(n * width * 4);
            let (taps, conditioning, z, k, v, gemm) = (
                a(self.width * 5)?,
                a(self.width)?,
                a(self.width)?,
                a(DK * DIM)?,
                a(DK * DIM)?,
                self.device.alloc(if self.splash {
                    crate::splash::workspace_bytes(n, self.width * 5, self.ff)
                } else {
                    n * self.width * 5 * 2
                })?,
            );
            d.taps = taps;
            d.conditioning = conditioning;
            d.z = z;
            d.k = k;
            d.v = v;
            d.gemm = gemm;
        }
        Ok(())
    }
    /// The registered upstream GGUF retains both v2 convolutions and selector.
    /// Reject incompatible geometry/version; never run these weights as v1.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<()> {
        if self.bonsai.is_some() || self.ternary.is_some() {
            return Err(MetalError::Model("Bonsai requires its own validated speculative companion; a base-Qwen DFlash checkpoint is not interchangeable".into()));
        }
        self.source_versions.extend(crate::offload::versions(path)?);
        if self.geometry != Geometry::DENSE_27B {
            return Err(MetalError::Model(
                "Metal DFlash2 currently requires the dense 27B target and its elected companion"
                    .into(),
            ));
        }
        self.require_committed()?;
        if self.dflash.is_some()
            || self.cold.is_some()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || self.cache.iter().any(|s| !s.history.is_empty())
        {
            return Err(MetalError::Model(
                "attach DFlash2 once, before prefill".into(),
            ));
        }
        let packed = if path.join("manifest.json").is_file() {
            if !self.splash {
                return Err(MetalError::Model(
                    "Splash draft requires its paired Splash target".into(),
                ));
            }
            Some(
                paddock_models::splash::Draft::open(path)
                    .map_err(|e| MetalError::Model(e.to_string()))?,
            )
        } else {
            None
        };
        let map = if packed.is_none() {
            Some(MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?)
        } else {
            None
        };
        if let Some(map) = &map {
            if map.gguf().architecture() != Some("dflash") {
                return Err(MetalError::Model("expected DFlash2 GGUF".into()));
            }
            for (key, want) in [
                ("embedding_length", self.width),
                ("feed_forward_length", self.ff),
                ("block_count", 5),
                ("attention.head_count", DH),
                ("attention.head_count_kv", DK),
                ("attention.key_length", DIM),
                ("attention.value_length", DIM),
                ("attention.sliding_window", 2048),
                ("block_size", 8),
                ("conv_kernel_size", 2),
                ("conv_group_size", 16),
                ("selector_rank", 256),
                ("selector_top_k", 16),
            ] {
                if map.gguf().arch_field(key).and_then(Value::as_u64) != Some(want as u64) {
                    return Err(MetalError::Model(format!("incompatible DFlash2 {key}")));
                }
            }
            let taps = match map.gguf().arch_field("target_layers") {
                Some(Value::Array(v)) => v.iter().map(Value::as_u64).collect::<Vec<_>>(),
                _ => Vec::new(),
            };
            if taps != TAPS.iter().map(|&n| Some(n as u64)).collect::<Vec<_>>()
                || !matches!(
                    map.gguf().arch_field("attention.causal"),
                    Some(Value::Bool(false))
                )
                || map
                    .gguf()
                    .arch_field("rope.freq_base")
                    .and_then(Value::as_f32)
                    != Some(10_000_000.)
                || map
                    .gguf()
                    .arch_field("attention.layer_norm_rms_epsilon")
                    .and_then(Value::as_f32)
                    != Some(self.eps)
                || map
                    .gguf()
                    .metadata
                    .get("tokenizer.ggml.mask_token_id")
                    .and_then(Value::as_u64)
                    != Some(248070)
            {
                return Err(MetalError::Model(
                    "incompatible DFlash2 taps/causality/rotary/mask".into(),
                ));
            }
        }
        let w = |n: &str, d: &[usize]| {
            if let Some(source) = &packed {
                let tensor = source
                    .tensor(n)
                    .map_err(|e| MetalError::Model(e.to_string()))?;
                checkpoint::splash_weight(&self.device, tensor, n, d)
            } else {
                Weight::load(
                    &self.device,
                    map.as_ref().expect("GGUF or packed source"),
                    n,
                    d,
                )
            }
        };
        let norm = |n: &str, d: &[usize]| -> Result<Weight> {
            let v = w(n, d)?;
            if v.ty != 0 {
                return Err(MetalError::Model(format!("DFlash2 {n} must be F32")));
            }
            Ok(v)
        };
        let before = self.device.allocated_bytes();
        // The window ends at the draft block's final row. Keep 64 spare rows:
        // prefix cuts trail a finishing prompt by at most 31, and a future
        // verification writes at most 8. Neither may overwrite a saved
        // boundary's still-live 2048-row window. Every ring is page-aligned.
        let ring_pages = self
            .page_stride
            .min((2048usize + 64).div_ceil(BLOCK_TOKENS));
        let ring_tokens = ring_pages * BLOCK_TOKENS;
        let kv = self.state_slots * ring_tokens * DK * DIM * 2;
        let mut layers = Vec::new();
        for i in 0..5 {
            let p = |n: &str| format!("blk.{i}.{n}");
            let cv = |n: &str| -> Result<Conv> {
                Ok(Conv {
                    base: norm(&p(&format!("{n}_base")), &[self.width, 2, 2])?,
                    proj: w(
                        &p(&format!("{n}_proj.weight")),
                        &[self.width, self.width / 4],
                    )?,
                })
            };
            layers.push(DraftLayer {
                norm: norm(&p("attn_norm.weight"), &[self.width])?,
                post: norm(&p("ffn_norm.weight"), &[self.width])?,
                q: w(&p("attn_q.weight"), &[self.width, DH * DIM])?,
                k: w(&p("attn_k.weight"), &[self.width, DK * DIM])?,
                v: w(&p("attn_v.weight"), &[self.width, DK * DIM])?,
                o: w(&p("attn_output.weight"), &[DH * DIM, self.width])?,
                qn: norm(&p("attn_q_norm.weight"), &[DIM])?,
                kn: norm(&p("attn_k_norm.weight"), &[DIM])?,
                gate: w(&p("ffn_gate.weight"), &[self.width, self.ff])?,
                up: w(&p("ffn_up.weight"), &[self.width, self.ff])?,
                down: w(&p("ffn_down.weight"), &[self.ff, self.width])?,
                ac: cv("attn_conv")?,
                fc: cv("ffn_conv")?,
                keys: self.device.alloc(kv)?,
                values: self.device.alloc(kv)?,
            });
        }
        let fc = w("fc.weight", &[self.width * 5, self.width])?;
        let enc_norm = norm("enc.output_norm.weight", &[self.width])?;
        let out_norm = norm("output_norm.weight", &[self.width])?;
        let selector = w("selector_hidden.weight", &[self.width, 256])?;
        let pred = w("selector_predecessor.weight", &[256, self.vocab])?;
        let succ = w("selector_successor.weight", &[256, self.vocab])?;
        if ![&pred, &succ]
            .iter()
            .all(|w| matches!(w.ty, 12 | 13 | 14 | 23) || (packed.is_some() && w.ty == 30))
        {
            return Err(MetalError::Model(
                "DFlash2 selector requires exact K-quant codebooks".into(),
            ));
        }
        let added = self.device.allocated_bytes() - before - kv as u64 * 10;
        self.ensure_verify()?;
        let rows = self
            .spec
            .as_ref()
            .expect("verification buffers allocated")
            .rows;
        let a = |n| self.device.alloc(self.row_capacity * n * 4);
        let pages = self.device.upload(
            &(0..self.slots.len())
                .flat_map(|s| {
                    (0..self.page_stride)
                        .flat_map(move |p| ((s * ring_pages + p % ring_pages) as u32).to_le_bytes())
                })
                .collect::<Vec<_>>(),
        )?;
        self.dflash = Some(Dflash {
            fc,
            enc_norm,
            out_norm,
            selector,
            pred,
            succ,
            layers,
            pages,
            ring_tokens,
            taps: a(self.width * 5)?,
            conditioning: a(self.width)?,
            z: a(self.width)?,
            x: a(self.width)?,
            norm: a(self.width)?,
            delta: a(self.width)?,
            conv: a(self.width)?,
            coeff: a(self.width / 4)?,
            q: a(DH * DIM)?,
            qn: a(DH * DIM)?,
            k: a(DK * DIM)?,
            v: a(DK * DIM)?,
            attn: a(DH * DIM)?,
            gate: a(self.ff)?,
            up: a(self.ff)?,
            gemm: self.device.alloc(if self.splash {
                crate::splash::workspace_bytes(self.row_capacity, self.width * 5, self.ff)
            } else {
                self.row_capacity * self.width * 5 * 2
            })?,
            logits: self.device.alloc(rows * self.vocab * 4)?,
            top: self.device.alloc(rows * 16 * 8)?,
            top_parts: self
                .device
                .alloc(rows * self.vocab.div_ceil(4096) * 16 * 8)?,
            selector_h: self.device.alloc(rows * 256 * 4)?,
            out: self.device.alloc(rows * 4)?,
        });
        self.weight_bytes += added;
        self.kv_bytes += kv as u64 * 10;
        tracing::info!(
            weight_bytes = added,
            kv_bytes = kv * 10,
            "native Metal Qwen DFlash2 attached; block=8, conv=2, selector rank=256/top16"
        );
        Ok(())
    }

    pub(super) fn dflash_checkpoint(&self, cmd: &Commands<'_>, from: usize, to: usize) {
        let Some(d) = &self.dflash else { return };
        let words = d.ring_tokens * DK * DIM / 2;
        for layer in &d.layers {
            for b in [&layer.keys, &layer.values] {
                cmd.dispatch(
                    "spec_copy_words",
                    &[b, b],
                    &[(from * words) as u32, (to * words) as u32, words as u32],
                    [words.div_ceil(256), 1, 1],
                    256,
                );
            }
        }
    }

    pub(super) fn dflash_tap(&self, cmd: &Commands<'_>, layer: usize, m: usize) {
        if let Some(d) = &self.dflash
            && let Some(tap) = TAPS.iter().position(|&n| n == layer)
        {
            cmd.dispatch(
                "df_tap",
                &[&self.scratch.x, &d.taps],
                &[self.width as u32, m as u32, tap as u32],
                [(m * self.width).div_ceil(256), 1, 1],
                256,
            );
        }
    }
    pub(super) fn dflash_append(&self, cmd: &Commands<'_>, m: usize) {
        let Some(d) = &self.dflash else { return };
        let s = &self.scratch;
        d.fc.linear(cmd, &d.taps, &d.conditioning, m, 1., &d.gemm);
        cmd.dispatch(
            if self.splash { "splash_df_rms" } else { "rms" },
            &[&d.conditioning, &d.enc_norm.buffer, &d.z],
            &[self.width as u32, 0, self.eps.to_bits()],
            [m, 1, 1],
            256,
        );
        for w in &d.layers {
            projections(cmd, &[(&w.k, &d.k), (&w.v, &d.v)], &d.z, m, &d.gemm);
            cmd.dispatch(
                if self.splash {
                    "splash_df_kstore"
                } else {
                    "df_kstore"
                },
                &[
                    &d.k,
                    &d.v,
                    &w.kn.buffer,
                    &s.meta,
                    &d.pages,
                    &w.keys,
                    &w.values,
                    &s.mrope,
                    &s.bounds,
                ],
                &[
                    DK as u32,
                    self.page_stride as u32,
                    self.eps.to_bits(),
                    10_000_000f32.to_bits(),
                    d.ring_tokens as u32,
                    0,
                ],
                [DK, m, 1],
                32,
            );
        }
    }

    pub(super) fn dflash_draft(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.require_committed()?;
        if self.dflash.is_none() || pendings.is_empty() || k == 0 {
            return Ok(None);
        }
        // MLX's verifier may take a shorter prefix, but the noncausal
        // drafter must still see its trained eight-row window.
        let block = if self.mlx {
            spec::BLOCK
        } else {
            (k + 1).min(spec::BLOCK)
        };
        let rows = pendings.len() * block;
        if rows
            > self
                .spec
                .as_ref()
                .expect("verification buffers allocated")
                .rows
        {
            return Ok(None);
        }
        let mut seen = vec![false; self.slots.len()];
        for &(slot, t) in pendings {
            if slot >= seen.len()
                || self.pending.iter().any(|p| p.slot == slot)
                || seen[slot]
                || t as usize >= self.vocab
                || self.slots[slot].history.is_empty()
                || self.slots[slot].history.len() + block > self.context
            {
                return Ok(None);
            }
            seen[slot] = true;
        }
        let s = &self.scratch;
        let d = self.dflash.as_ref().expect("DFlash2 attached");
        let mut tokens = Vec::new();
        let mut meta = Vec::new();
        let mut tiles = Vec::new();
        for (i, &(slot, t)) in pendings.iter().enumerate() {
            tiles.extend([(i * block) as u32, block as u32]);
            for j in 0..block {
                tokens.push(if j == 0 { t } else { 248070 });
                meta.extend([slot as u32, (self.slots[slot].history.len() + j) as u32]);
            }
        }
        unsafe {
            s.ids.write_u32(&tokens);
            s.meta.write_u32(&meta);
            s.attn_tiles.write_u32(&tiles);
            s.mrope.write_u32(
                &meta
                    .chunks_exact(2)
                    .flat_map(|r| self.rope_position(r[0] as usize, r[1] as usize))
                    .collect::<Vec<_>>(),
            );
        }
        let cmd = self.device.begin()?;
        cmd.dispatch(
            if self.mlx { "mlx_embed" } else { "embed" },
            &[&self.embedding.buffer, &s.ids, &d.x],
            &[
                self.width as u32,
                rows as u32,
                if self.mlx {
                    self.vocab as u32
                } else {
                    self.embedding.ty
                },
                1f32.to_bits(),
            ],
            [(rows * self.width).div_ceil(256), 1, 1],
            256,
        );
        let conv = |cv: &Conv, input: &Buffer, side: u32| {
            cmd.dispatch(
                "df_conv",
                &[input, &cv.base.buffer, &d.coeff, &d.conv],
                &[self.width as u32, rows as u32, block as u32, side],
                [(rows * self.width / 4).div_ceil(256), 1, 1],
                256,
            )
        };
        for w in &d.layers {
            cmd.dispatch(
                if self.splash { "splash_df_rms" } else { "rms" },
                &[&d.x, &w.norm.buffer, &d.norm],
                &[self.width as u32, 0, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            w.ac.proj.linear(&cmd, &d.norm, &d.coeff, rows, 1., &d.gemm);
            conv(&w.ac, &d.norm, 0);
            projections(
                &cmd,
                &[(&w.q, &d.q), (&w.k, &d.k), (&w.v, &d.v)],
                &d.conv,
                rows,
                &d.gemm,
            );
            cmd.dispatch(
                if self.splash {
                    "splash_df_qnorm"
                } else {
                    "df_qnorm"
                },
                &[&d.q, &w.qn.buffer, &s.mrope, &d.qn],
                &[
                    DH as u32,
                    self.page_stride as u32,
                    self.eps.to_bits(),
                    10_000_000f32.to_bits(),
                ],
                [DH, rows, 1],
                32,
            );
            cmd.dispatch(
                if self.splash {
                    "splash_df_kstore"
                } else {
                    "df_kstore"
                },
                &[
                    &d.k,
                    &d.v,
                    &w.kn.buffer,
                    &s.meta,
                    &d.pages,
                    &w.keys,
                    &w.values,
                    &s.mrope,
                    &s.bounds,
                ],
                &[
                    DK as u32,
                    self.page_stride as u32,
                    self.eps.to_bits(),
                    10_000_000f32.to_bits(),
                    d.ring_tokens as u32,
                    1,
                ],
                [DK, rows, 1],
                32,
            );
            cmd.dispatch(
                if self.splash {
                    "splash_df_query"
                } else {
                    "attention_query"
                },
                &[&d.qn, &d.gemm],
                &[(DH * DIM) as u32, 0, rows as u32],
                [((rows + 32) * DH * DIM).div_ceil(256), 1, 1],
                256,
            );
            if self.splash {
                assert_eq!(rows, pendings.len() * 8);
                assert!(
                    d.gemm.len()
                        >= (rows + 32) * DH * DIM * 2 + pendings.len() * DK * 8 * 32 * 130 * 4
                );
            }
            cmd.dispatch(
                if self.splash {
                    "splash_df_attention_grouped"
                } else {
                    "df_attention"
                },
                &[
                    &d.gemm,
                    &w.keys,
                    &w.values,
                    &s.meta,
                    &d.pages,
                    &d.attn,
                    &s.attn_tiles,
                ],
                &[
                    DH as u32,
                    DK as u32,
                    self.page_stride as u32,
                    (1f32 / (DIM as f32).sqrt()).to_bits(),
                    rows as u32,
                ],
                if self.splash {
                    [DK, pendings.len(), 8]
                } else {
                    [DH, pendings.len(), 1]
                },
                128,
            );
            if self.splash {
                cmd.dispatch(
                    "splash_df_attention_join",
                    &[&d.gemm, &d.attn],
                    &[rows as u32],
                    [rows * DH, 1, 1],
                    32,
                );
            }
            w.o.linear(&cmd, &d.attn, &d.delta, rows, 1., &d.gemm);
            conv(&w.ac, &d.delta, 1);
            cmd.dispatch(
                if self.splash {
                    "mlx_residual"
                } else {
                    "residual"
                },
                &[&d.x, &d.conv],
                &[(rows * self.width) as u32, 1f32.to_bits()],
                [(rows * self.width).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                if self.splash { "splash_df_rms" } else { "rms" },
                &[&d.x, &w.post.buffer, &d.norm],
                &[self.width as u32, 0, self.eps.to_bits()],
                [rows, 1, 1],
                256,
            );
            w.fc.proj.linear(&cmd, &d.norm, &d.coeff, rows, 1., &d.gemm);
            conv(&w.fc, &d.norm, 0);
            projections(
                &cmd,
                &[(&w.gate, &d.gate), (&w.up, &d.up)],
                &d.conv,
                rows,
                &d.gemm,
            );
            cmd.dispatch(
                "swiglu",
                &[&d.gate, &d.up],
                &[(rows * self.ff) as u32],
                [(rows * self.ff).div_ceil(256), 1, 1],
                256,
            );
            w.down.linear(&cmd, &d.gate, &d.delta, rows, 1., &d.gemm);
            conv(&w.fc, &d.delta, 1);
            cmd.dispatch(
                if self.splash {
                    "mlx_residual"
                } else {
                    "residual"
                },
                &[&d.x, &d.conv],
                &[(rows * self.width) as u32, 1f32.to_bits()],
                [(rows * self.width).div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.dispatch(
            if self.splash { "splash_df_rms" } else { "rms" },
            &[&d.x, &d.out_norm.buffer, &d.norm],
            &[self.width as u32, 0, self.eps.to_bits()],
            [rows, 1, 1],
            256,
        );
        // Noncausal hidden states still use all eight trained positions.
        // Only their output projection may be shortened: selection is a
        // left-to-right chain, so positions after k cannot affect its prefix.
        // Keep a full R4 tile: recompiling the legacy contraction at R1/R2/R3
        // can change rounding even with the same source expression. Retain
        // its unused seed row too. Concurrent drafting, Splash and older
        // hardware keep their independently qualified path.
        let short_head = self.mlx
            && !self.splash
            && self.device.tensor_accelerated()
            && pendings.len() == 1
            && k <= 3;
        #[cfg(test)]
        let short_head = short_head && !FULL_DRAFT_HEAD_FOR_TEST.with(|v| v.get());
        let head_rows = if short_head { 4 } else { rows };
        let selected = if short_head { k + 1 } else { block };
        // The drafter borrows the target's head, which may be MLX affine
        // rather than a GGUF tensor. Keep all draft-owned weights unchanged.
        if self.mlx {
            self.project(
                &cmd,
                &[(self.head(), &d.logits)],
                &d.norm,
                head_rows,
                &d.gemm,
            );
        } else {
            self.head()
                .linear(&cmd, &d.norm, &d.logits, rows, 1., &d.gemm);
        }
        cmd.dispatch(
            "df_top16",
            &[&d.logits, &d.top_parts],
            &[self.vocab as u32],
            [self.vocab.div_ceil(4096), head_rows, 1],
            256,
        );
        cmd.dispatch(
            "df_top16_merge",
            &[&d.top_parts, &d.top],
            &[(self.vocab.div_ceil(4096) * 16) as u32],
            [head_rows, 1, 1],
            256,
        );
        d.selector
            .linear(&cmd, &d.norm, &d.selector_h, rows, 1., &d.gemm);
        cmd.dispatch(
            if short_head {
                "df_select_prefix"
            } else {
                "df_select"
            },
            &[
                &d.top,
                &d.pred.buffer,
                &d.succ.buffer,
                &d.selector_h,
                &s.ids,
                &d.out,
            ],
            &[block as u32, d.pred.ty, d.succ.ty, selected as u32],
            [pendings.len(), 1, 1],
            256,
        );
        cmd.finish()?;
        // SAFETY: the requested prefix was written and the command completed.
        // Never expose stale suffix tokens from an earlier, longer proposal.
        let out = unsafe {
            std::slice::from_raw_parts(
                d.out.raw.contents().as_ptr().cast::<u32>(),
                if short_head { selected } else { rows },
            )
            .to_vec()
        };
        Ok(Some(
            out.chunks(if short_head { selected } else { block })
                .map(|r| r[1..].to_vec())
                .collect(),
        ))
    }
}
