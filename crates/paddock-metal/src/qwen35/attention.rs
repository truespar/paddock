use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static BASELINE_PREFILL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static CANONICAL_PREFILL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_GQA_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_LANGUAGE_NAX_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static FORCE_LANGUAGE_NAX_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn split_append(rows: usize, visible_kv: usize, workspace_bytes: usize, heads: usize) -> bool {
    (16..=128).contains(&rows) && visible_kv >= 256 && workspace_bytes >= rows * heads * 4 * 258 * 4
}

impl Qwen35 {
    pub(super) fn register_language_attention(&self) -> bool {
        let shape = self.geometry == Geometry::DENSE_08B;
        #[cfg(test)]
        let shape = shape
            || (self.geometry == Geometry::DENSE_4B
                && FORCE_LANGUAGE_NAX_FOR_TEST.with(|v| v.get()));
        let enabled = self.mlx
            && self.head.is_none()
            && self.vision.is_some()
            && self.device.tensor_accelerated()
            && shape;
        #[cfg(test)]
        let enabled = enabled && !BASELINE_LANGUAGE_NAX_FOR_TEST.with(|v| v.get());
        enabled
    }
    pub(super) fn full_attention(
        &self,
        cmd: &Commands<'_>,
        w: &FullAttention,
        m: usize,
        tiles: usize,
        decode_rows: usize,
        long_tiles: usize,
        (decode_length, append_length): (usize, usize),
    ) {
        if self.bonsai.is_some() {
            return self.bonsai_attention(cmd, w, m, tiles, decode_rows, decode_length);
        }
        let s = &self.scratch;
        let heads = self.geometry.heads;
        let kv_heads = self.geometry.kv_heads;
        self.project(
            cmd,
            &[(&w.q, &s.qraw), (&w.k, &s.k), (&w.v, &s.v)],
            &s.norm,
            m,
            &s.gemm,
        );
        let p = [
            heads as u32,
            kv_heads as u32,
            self.page_stride as u32,
            self.rope.to_bits(),
            self.eps.to_bits(),
            self.rotary as u32,
        ];
        cmd.dispatch(
            if self.mlx {
                "mlx_qnorm_rope"
            } else {
                "qwen_qnorm_rope"
            },
            &[&s.qraw, &w.q_norm.buffer, &s.mrope, &s.q],
            &p,
            [heads, m, 1],
            32,
        );
        cmd.dispatch(
            if self.mlx {
                "mlx_knorm_store"
            } else {
                "qwen_knorm_store"
            },
            &[
                &s.k,
                &s.v,
                &w.k_norm.buffer,
                &s.meta,
                &s.pages,
                &w.keys,
                &w.values,
                &s.mrope,
            ],
            &p,
            [kv_heads, m, 1],
            32,
        );
        if tiles > 0 {
            // Only elect the indexed traversal on the measured M5 dense-27B
            // geometry; other families and older GPUs retain their kernels.
            let indexed = cmd.tensor_accelerated() && self.geometry == Geometry::DENSE_27B;
            #[cfg(test)]
            let indexed = indexed && !BASELINE_PREFILL_FOR_TEST.with(|v| v.get());
            // Share six Q heads per KV page without changing the 16-token
            // softmax reduction. Reuse existing scratch; many ragged spans
            // may need more padded rows than it covers, so retain fallback.
            let grouped = self.stable_affine_contract()
                && indexed
                && append_length >= 256
                && !self.verifying
                && self.slots.iter().all(|slot| slot.mm.is_none())
                && s.gemm.len() >= tiles * 32 * heads * 256 * 2;
            #[cfg(test)]
            let grouped = grouped && !BASELINE_GQA_FOR_TEST.with(|v| v.get());
            let strict = self.geometry.moe() && moe::precise();
            // Reuse the decode-parts allocation only when it covers every
            // append row and four partitions. Join completes before decode
            // reuses this storage. Wide prefill retains its query-parallel route.
            // A sequence's softmax tree cannot depend on another request's
            // presence. Stable MLX prefill keeps the one-part contraction.
            let split = self.splash
                || (!self.stable_affine_contract()
                    && split_append(m, append_length, s.attn_parts.len(), heads));
            #[cfg(test)]
            let split = split && !(self.mlx && CANONICAL_PREFILL_FOR_TEST.with(|v| v.get()));
            // Prepared DeltaNet chunks are dead in a full-attention layer.
            // Their allocation covers (ceil(rows/32) + 3*slots) chunks of
            // value_heads*17408 elements: larger than four K/V tiles per
            // Q head in every elected geometry. Each group owns one tile;
            // the next DeltaNet preparation overwrites it after this pass.
            if !self.mlx || self.splash {
                assert!(
                    s.prepared.len()
                        >= tiles
                            * heads
                            * if split { 4 } else { 1 }
                            * if self.splash { 64 } else { 32 }
                            * 256
                            * if strict { 4 } else { 2 }
                );
            }
            if self.splash || grouped {
                assert!(s.gemm.len() >= tiles * 32 * heads * 256 * 2);
                cmd.dispatch(
                    if grouped {
                        "mlx_attention_query_grouped"
                    } else {
                        "splash_attention_query_grouped"
                    },
                    &[&s.q, &s.gemm, &s.attn_tiles],
                    &[tiles as u32],
                    [tiles * 32 * heads, 1, 1],
                    256,
                );
            } else if !strict {
                cmd.dispatch(
                    if self.mlx {
                        "mlx_attention_query"
                    } else {
                        "attention_query"
                    },
                    &[&s.q, &s.gemm],
                    &[(heads * 256) as u32, 0, m as u32],
                    [((m + 32) * heads * 256).div_ceil(256), 1, 1],
                    256,
                );
            }
            let buffers = [
                if strict { &s.q } else { &s.gemm },
                &w.keys,
                &w.values,
                &s.meta,
                &s.pages,
                if split { &s.attn_parts } else { &s.attn },
                &s.attn_tiles,
                &s.limits,
                &s.prepared,
            ];
            let kernel = if grouped {
                "mlx_attention_prefill_gqa"
            } else if self.splash {
                "splash_attention_prefill_grouped"
            } else if self.mlx && split {
                "mlx_attention_prefill_split"
            } else if self.mlx && append_length >= 256 {
                // Read each existing BF16 physical page directly. The
                // one-partition kernel retains the same 16-token causal
                // blocks and softmax order, without staging/transposing KV.
                if indexed {
                    "mlx_attention_prefill_indexed"
                } else {
                    "mlx_attention_prefill_direct"
                }
            } else if self.mlx {
                "mlx_attention_prefill"
            } else if strict && split {
                "qwen_attention_prefill_split_strict"
            } else if strict {
                "qwen_attention_prefill_strict"
            } else if split {
                "qwen_attention_prefill_split"
            } else {
                "qwen_attention_prefill"
            };
            let kernel = if self.register_language_attention() && !split {
                "mlx_attention_prefill_nax"
            } else {
                kernel
            };
            cmd.dispatch(
                kernel,
                &buffers[..if self.mlx && !self.splash { 8 } else { 9 }],
                &[
                    heads as u32,
                    kv_heads as u32,
                    self.page_stride as u32,
                    (1.0f32 / 16.0).to_bits(),
                ],
                if grouped {
                    [kv_heads, tiles * 4, 1]
                } else if self.splash {
                    [kv_heads, tiles * 4, 4]
                } else {
                    [heads, tiles, if split { 4 } else { 1 }]
                },
                if self.splash || grouped { 256 } else { 128 },
            );
            if split {
                cmd.dispatch(
                    "qwen_attention_prefill_join",
                    &[&s.attn_parts, &s.attn, &s.attn_tiles],
                    &[heads as u32],
                    [heads, tiles, 32],
                    32,
                );
            }
        }
        if decode_rows > 0 {
            // Same M5/native dense-27B qualification boundary as stable cold
            // prefill. Do not alter Splash, GGUF or earlier GPU elections.
            let stable = self.stable_affine_prefill();
            #[cfg(test)]
            let stable = stable && !projection::BASELINE_ATTENTION_FOR_TEST.with(|v| v.get());
            // MLX reductions belong to each sequence, not its current peers.
            // Keep the single-stream floor for decode and speculative rows;
            // the shader derives each row's split count from its own limit.
            let decode_live = if self.splash || stable {
                1
            } else if self.mlx && self.verifying {
                self.spec
                    .as_ref()
                    .expect("verification buffers allocated")
                    .live
            } else {
                decode_rows
            };
            #[cfg(test)]
            let decode_live = if self.mlx && projection::CANONICAL_MLX_FOR_TEST.with(|v| v.get()) {
                1
            } else {
                decode_live
            };
            let splits = decode_length
                .div_ceil(128)
                .max(16usize.div_ceil(decode_live))
                .clamp(1, MAX_SPLITS);
            cmd.dispatch(
                if stable && heads / kv_heads == 4 {
                    "mlx_attention_stable_gqa4"
                } else if stable {
                    "mlx_attention_stable"
                } else if self.mlx && self.verifying {
                    "mlx_attention_verify"
                } else if self.mlx && heads / kv_heads == 4 {
                    "mlx_attention_decode_gqa4"
                } else if self.mlx {
                    "mlx_attention_decode"
                } else {
                    self.geometry.decode_kernel()
                },
                &[
                    &s.q,
                    &w.keys,
                    &w.values,
                    &s.meta,
                    &s.pages,
                    &s.decode_rows,
                    &s.attn_parts,
                    &s.limits,
                ],
                &[
                    heads as u32,
                    kv_heads as u32,
                    self.page_stride as u32,
                    (1.0f32 / 16.0).to_bits(),
                    splits as u32,
                    16usize.div_ceil(decode_live) as u32,
                ],
                [kv_heads, decode_rows, splits],
                128,
            );
            cmd.dispatch(
                "qwen_attention_merge",
                &[&s.attn_parts, &s.attn, &s.decode_rows],
                &[heads as u32, splits as u32],
                [heads * decode_rows, 1, 1],
                32,
            );
        }
        if long_tiles > 0 {
            let parts = crate::splash::DECODE_PARTS;
            assert!(self.splash && heads == 24 && kv_heads == 4);
            assert!(s.gemm.len() >= long_tiles * 8 * heads * 256 * 2);
            assert!(s.prepared.len() >= long_tiles * kv_heads * parts * 64 * 256 * 2);
            assert!(s.attn_parts.len() >= long_tiles * 8 * heads * parts * 258 * 4);
            cmd.dispatch(
                "splash_attention_query_decode",
                &[&s.q, &s.gemm, &s.long_decode_tiles],
                &[long_tiles as u32],
                [long_tiles * 8 * heads, 1, 1],
                256,
            );
            cmd.dispatch(
                "splash_attention_decode_grouped",
                &[
                    &s.gemm,
                    &w.keys,
                    &w.values,
                    &s.meta,
                    &s.pages,
                    &s.attn_parts,
                    &s.long_decode_tiles,
                    &s.limits,
                    &s.prepared,
                ],
                &[
                    heads as u32,
                    kv_heads as u32,
                    self.page_stride as u32,
                    (1f32 / 16.).to_bits(),
                ],
                [kv_heads, long_tiles, parts],
                256,
            );
            cmd.dispatch(
                "splash_attention_decode_join",
                &[&s.attn_parts, &s.attn, &s.long_decode_tiles],
                &[heads as u32],
                [heads, long_tiles, 8],
                32,
            );
        }
        cmd.dispatch(
            if self.mlx {
                "mlx_attn_gate"
            } else {
                "qwen_attn_gate"
            },
            &[&s.attn, &s.qraw],
            &[(m * heads * 256) as u32],
            [(m * heads * 256).div_ceil(256), 1, 1],
            256,
        );
        self.project(cmd, &[(&w.o, &s.delta)], &s.attn, m, &s.gemm);
    }
}

#[cfg(test)]
mod tests {
    use super::split_append;

    #[test]
    fn append_partition_election_requires_its_own_context_and_workspace() {
        for heads in [16, 24] {
            let bytes = 64 * heads * 4 * 258 * 4;
            assert!(split_append(64, 1089, bytes, heads));
            assert!(!split_append(64, 63, bytes, heads));
            assert!(!split_append(64, 1089, bytes - 1, heads));
            assert!(!split_append(129, 1089, usize::MAX, heads));
            assert!(!split_append(15, 1089, usize::MAX, heads));
            assert!(split_append(128, 256, usize::MAX, heads));
            assert!(!split_append(128, 255, usize::MAX, heads));
        }
    }
}
