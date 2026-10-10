use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static BASELINE_RECURRENT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A validated, immutable row plan. GPU metadata is re-uploaded on resumption:
/// unrelated slots may have decoded or changed their page tables meanwhile.
pub(super) struct Plan {
    pub(super) rows: Vec<(usize, u32, u32)>,
    pub(super) selected: Vec<usize>,
    spans: Vec<u32>,
    chunks: Vec<u32>,
    bounds: Vec<u32>,
    tiles: Vec<u32>,
    decode_rows: Vec<u32>,
    long_decode_tiles: Vec<u32>,
    checkpoint_rows: Vec<u32>,
    checkpoint_spans: Vec<u32>,
    checkpoints: Vec<(usize, usize, Vec<u32>)>,
}

impl Qwen35 {
    /// M5 packed prompt contractions have their own qualified arithmetic.
    /// Cached suffixes remain prompt work, even if only one row is left;
    /// independently decoding requests never inherit that role from peers.
    pub(super) fn stable_bonsai_prefill(&self) -> bool {
        let enabled = self.bonsai.is_some() && self.device.tensor_accelerated();
        #[cfg(test)]
        let enabled = enabled && !bonsai::BASELINE_PREFILL.with(|v| v.get());
        enabled
    }
    pub(super) fn stable_affine_prefill(&self) -> bool {
        self.mlx
            && !self.splash
            && self.bonsai.is_none()
            && matches!(
                self.geometry,
                Geometry::DENSE_08B | Geometry::DENSE_4B | Geometry::DENSE_27B
            )
            && self.device.tensor_accelerated()
    }
    pub(super) fn stable_affine_contract(&self) -> bool {
        let stable = self.stable_affine_prefill();
        #[cfg(test)]
        let stable = stable
            && !projection::BASELINE_AFFINE_FOR_TEST.with(|v| v.get())
            && !projection::CANONICAL_MLX_FOR_TEST.with(|v| v.get());
        stable
    }
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        selected: &[usize],
    ) -> Result<Vec<f32>> {
        self.require_committed()?;
        objc2::rc::autoreleasepool(|_| {
            let plan = self.plan_execution(rows, selected)?;
            let result = self.execute_plan(&plan);
            self.unreserve_plan(&plan);
            if result.is_ok() {
                // Persist the final reusable boundary without needing eviction
                // pressure or a future request. The adjacent backup boundary
                // remains resident and is still spilled on eviction. Only
                // completed, unreserved snapshots enter the bounded writer.
                for &(index, slot, ref history) in &plan.checkpoints {
                    if self.slots[slot].cuts.last() == Some(&history.len()) {
                        self.spill_checkpoint(index);
                    }
                }
            }
            result
        })
    }
    pub(super) fn plan_execution(
        &mut self,
        rows: &[(usize, u32, u32)],
        selected: &[usize],
    ) -> Result<Plan> {
        let m = rows.len();
        if m == 0
            || m > self.row_capacity
            || selected.len()
                > if self.verifying {
                    self.spec
                        .as_ref()
                        .expect("verification buffers allocated")
                        .rows
                } else {
                    self.slots.len()
                }
            || selected.iter().any(|&r| r >= m)
        {
            return Err(MetalError::Model(
                "invalid Qwen execution row count/selection".into(),
            ));
        }
        let mut lengths: Vec<_> = self.slots.iter().map(|s| s.history.len()).collect();
        let mut seen = vec![false; self.slots.len()];
        let mut previous_slot = None;
        for &(slot, token, pos) in rows {
            if slot >= self.slots.len()
                || token as usize >= self.vocab
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(MetalError::Model(format!(
                    "invalid Qwen row slot={slot}, token={token}, position={pos}"
                )));
            }
            if previous_slot != Some(slot) {
                if seen[slot] {
                    return Err(MetalError::Model(
                        "Qwen rows revisit an already packed slot".into(),
                    ));
                }
                seen[slot] = true;
                previous_slot = Some(slot);
            }
            lengths[slot] += 1;
        }
        for &(slot, _, pos) in rows {
            while self.slots[slot]
                .table
                .ensure(pos as usize, &mut self.pool)
                .is_err()
            {
                if !self.evict_checkpoint() {
                    return Err(MetalError::Memory("Qwen KV pool exhausted".into()));
                }
            }
        }
        // Recurrent spans must be contiguous within this batch. Validate the
        // complete plan before mutating any GPU state or publishing history.
        let mut spans = Vec::new();
        let mut chunks = Vec::new();
        let mut bounds = vec![0u32; m * 2];
        let mut tiles = Vec::new();
        let mut decode_rows = Vec::new();
        let mut long_decode_tiles = Vec::new();
        let mut checkpoint_rows = vec![0u32; m];
        let mut checkpoint_spans = Vec::new();
        let mut checkpoints = Vec::new();
        let mut first = 0;
        while first < m {
            let slot = rows[first].0;
            let mut end = first + 1;
            while end < m && rows[end].0 == slot {
                end += 1;
            }
            let count = end - first;
            let mut cuts = [0u32; 4];
            let mut cut_index = 0;
            for r in first..end {
                let cut = rows[r].2 as usize + 1;
                if self.verifying || !self.slots[slot].cuts.contains(&cut) {
                    continue;
                }
                let mut history = self.slots[slot].history.clone();
                history.extend(rows[first..=r].iter().map(|r| r.1));
                if self.cache.iter().any(|c| {
                    c.history == history
                        && c.images == self.slots[slot].mm.as_ref().map_or(&[][..], |m| &m.keys)
                }) {
                    continue;
                }
                let images = self.slots[slot].mm.as_ref().map_or(&[][..], |m| &m.keys);
                let keep_from = self.slots[slot].cuts[0];
                let Some(index) = retention::replacement(&self.cache, &history, images, keep_from)
                else {
                    // Async cold restores can still own checkpoint state after
                    // a requesting slot is cancelled/reused. Capturing a prefix
                    // is optional; never steal their destination or panic.
                    continue;
                };
                if retention::trace() {
                    tracing::info!(
                        slot,
                        index,
                        previous = self.cache[index].history.len(),
                        tokens = history.len(),
                        keep_from,
                        superseded =
                            retention::superseded(&self.cache[index], &history, images, keep_from),
                        "metal-cache-capture"
                    );
                }
                self.spill_checkpoint(index);
                self.cache[index].reserved = true;
                self.cache[index].table.clear(&mut self.pool);
                self.cache[index].history.clear();
                let destination = (self.slots.len() + index + 1) as u32;
                checkpoint_rows[r] = destination;
                cuts[cut_index * 2] = r as u32;
                cuts[cut_index * 2 + 1] = destination;
                cut_index += 1;
                checkpoints.push((index, slot, history));
            }
            checkpoint_spans.extend(cuts);
            spans.extend([
                first as u32,
                count as u32,
                slot as u32,
                (chunks.len() / 4) as u32,
            ]);
            for r in first..end {
                bounds[r * 2] = first as u32;
                bounds[r * 2 + 1] = end as u32;
            }
            let prompt_attention =
                self.splash || self.bonsai.is_some() || self.stable_affine_contract();
            #[cfg(test)]
            let prompt_attention = prompt_attention
                || (self.mlx && projection::CANONICAL_PROMPT_PHASE_FOR_TEST.with(|v| v.get()));
            if count >= 16
                || (prompt_attention
                    && !self.verifying
                    && (rows[first].2 as usize) < self.slots[slot].prefill_end)
            {
                let mut r = first;
                while r < end {
                    let mut next = (r + 32).min(end);
                    if let Some(cut) = (r..next).find(|&i| checkpoint_rows[i] != 0) {
                        next = cut + 1;
                    }
                    chunks.extend([
                        r as u32,
                        (next - r) as u32,
                        slot as u32,
                        checkpoint_rows[next - 1],
                    ]);
                    r = next;
                }
                for r in (first..end).step_by(32) {
                    tiles.extend([r as u32, (end - r).min(32) as u32]);
                }
            } else {
                let long = if self.splash {
                    (first..end)
                        .find(|&r| rows[r].2 as usize + 1 >= crate::splash::LONG_DECODE)
                        .unwrap_or(end)
                } else {
                    end
                };
                decode_rows.extend((first..long).map(|r| r as u32));
                for start in (long..end).step_by(8) {
                    long_decode_tiles.extend([start as u32, (end - start).min(8) as u32]);
                }
            }
            first = end;
        }
        Ok(Plan {
            rows: rows.to_vec(),
            selected: selected.to_vec(),
            spans,
            chunks,
            bounds,
            tiles,
            decode_rows,
            long_decode_tiles,
            checkpoint_rows,
            checkpoint_spans,
            checkpoints,
        })
    }

    pub(super) fn unreserve_plan(&mut self, plan: &Plan) {
        for &(index, _, _) in &plan.checkpoints {
            self.cache[index].reserved = false;
        }
    }

    /// Complete one bounded causal batch, including decode riders, through
    /// every layer. Only completed batches can publish state/checkpoints.
    fn execute_plan(&mut self, plan: &Plan) -> Result<Vec<f32>> {
        let g = self.geometry;
        let Plan {
            rows,
            selected,
            spans,
            chunks,
            bounds,
            tiles,
            decode_rows,
            long_decode_tiles,
            checkpoint_rows,
            checkpoint_spans,
            checkpoints,
        } = plan;
        let m = rows.len();
        let mut pages = vec![0u32; self.slots.len() * self.page_stride];
        for (i, slot) in self.slots.iter().enumerate() {
            pages[i * self.page_stride..i * self.page_stride + slot.table.blocks().len()]
                .copy_from_slice(slot.table.blocks());
        }
        let s = &self.scratch;
        // SAFETY: one engine thread owns this graph; previous submit completed.
        unsafe {
            s.ids
                .write_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &rows
                    .iter()
                    .flat_map(|r| [r.0 as u32, r.2])
                    .collect::<Vec<_>>(),
            );
            s.outputs
                .write_u32(&selected.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.pages.write_u32(&pages);
            s.spans.write_u32(spans);
            s.chunks.write_u32(chunks);
            s.bounds.write_u32(bounds);
            s.attn_tiles.write_u32(tiles);
            s.decode_rows.write_u32(decode_rows);
            s.long_decode_tiles.write_u32(long_decode_tiles);
            s.checkpoint_rows.write_u32(checkpoint_rows);
            s.checkpoint_spans.write_u32(checkpoint_spans);
            s.mrope.write_u32(
                &rows
                    .iter()
                    .flat_map(|&(slot, _, pos)| self.rope_position(slot, pos as usize))
                    .collect::<Vec<_>>(),
            );
            s.limits.write_u32(
                &rows
                    .iter()
                    .map(|&(slot, _, pos)| {
                        self.slots[slot]
                            .mm
                            .as_ref()
                            .map_or(pos, |m| m.limit(pos as usize))
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let mut packed_spans: Vec<(usize, usize, crate::splash::Phase)> = Vec::new();
        let mut affine_spans: Vec<(usize, usize, usize)> = Vec::new();
        if self.stable_affine_contract() || self.stable_bonsai_prefill() || self.ternary.is_some() {
            for (row, &(slot, _, pos)) in rows.iter().enumerate() {
                let logical = if !self.verifying && (pos as usize) < self.slots[slot].prefill_end {
                    CHUNK
                } else {
                    1
                };
                #[cfg(test)]
                let logical = if crate::affine::REFERENCE_CHUNKS_FOR_TEST.with(|v| v.get())
                    && logical == CHUNK
                {
                    let remaining = self.slots[slot]
                        .prefill_end
                        .saturating_sub(1)
                        .saturating_sub(pos as usize / 2048 * 2048);
                    if pos as usize + 1 == self.slots[slot].prefill_end {
                        1
                    } else {
                        remaining.clamp(1, 2048)
                    }
                } else {
                    logical
                };
                if let Some(last) = affine_spans.last_mut().filter(|s| s.2 == logical) {
                    last.1 += 1;
                } else {
                    affine_spans.push((row, 1, logical));
                }
            }
        }
        if self.splash {
            for (row, &(slot, _, pos)) in rows.iter().enumerate() {
                let prefill = !self.verifying && (pos as usize) < self.slots[slot].prefill_end;
                let phase = crate::splash::Phase::for_row(prefill, pos as usize);
                if let Some(last) = packed_spans.last_mut().filter(|s| s.2 == phase) {
                    last.1 += 1;
                } else {
                    packed_spans.push((row, 1, phase));
                }
            }
        }
        // Only the qualified text prefill geometry changes intermediate storage.
        // Images, verification and other model families retain their old routes.
        let compact_text_ffn = self.mlx
            && !self.splash
            && self.bonsai.is_none()
            && !self.verifying
            && g == Geometry::DENSE_27B
            && crate::affine::compact_prefill_rows(m)
            && affine_spans.iter().all(|s| s.2 == CHUNK)
            && rows
                .iter()
                .all(|(slot, _, _)| self.slots[*slot].mm.is_none());
        // Request arrival may slice the first cold wave to 32 rows. Its BF16
        // projection partials must not change solely because peers arrived a
        // tick later. Pin pure text prefill to the qualified 512-row contract.
        // Retain the former pure-prefill election for other eligible routes
        // and test baselines. Phase-local spans below also cover tiny tails,
        // decode riders and verification on the stable dense-27B graph.
        let stable_prefill = self.stable_affine_prefill()
            && !self.verifying
            && (13..=CHUNK).contains(&m)
            && rows.iter().all(|&(slot, _, pos)| {
                self.slots[slot].mm.is_none() && (pos as usize) < self.slots[slot].prefill_end
            });
        #[cfg(test)]
        let stable_prefill =
            stable_prefill && !crate::affine::BASELINE_COLD_CONTRACT_FOR_TEST.with(|v| v.get());
        let cmd = self.device.begin()?;
        let cmd = if self.splash {
            cmd.with_packed_spans(&packed_spans)
        } else if !affine_spans.is_empty() {
            cmd.with_projection_rows(&affine_spans)
                .with_affine_prefill_rows(CHUNK)
        } else if stable_prefill {
            cmd.with_affine_prefill_rows(CHUNK)
        } else {
            cmd
        };
        // Native RMS uses four contiguous values per lane, preserving the
        // reference's reduction tree as well as its BF16 pre-weight boundary.
        let norm_threads = if self.mlx {
            (self.width.div_ceil(128) * 32).min(1024)
        } else {
            256
        };
        {
            if let Some(bonsai) = &self.bonsai {
                bonsai.embed(&cmd, &self.embedding, &s.ids, &s.x, m);
            } else if let Some(ternary) = &self.ternary {
                ternary.embed(&cmd, &self.embedding, &s.ids, &s.x, m);
            } else {
                cmd.dispatch(
                    if self.mlx { "mlx_embed" } else { "embed" },
                    &[&self.embedding.buffer, &s.ids, &s.x],
                    &[
                        self.width as u32,
                        m as u32,
                        if self.mlx {
                            self.vocab as u32
                        } else {
                            self.embedding.ty
                        },
                        1.0f32.to_bits(),
                    ],
                    [(m * self.width).div_ceil(256), 1, 1],
                    256,
                );
            }
            // Tower rows already inhabit the model residual basis. In
            // particular, inject AFTER PTQ1's inverse embedding FWHT; only
            // the later linear inputs receive Bonsai's forward rotations.
            self.inject_images(&cmd, rows);
            cmd.dispatch(
                if self.bonsai.is_some() {
                    "bonsai_rms"
                } else if self.mlx {
                    "mlx_rms"
                } else {
                    "rms"
                },
                &[&s.x, &self.layers[0].norm.buffer, &s.norm],
                &[
                    self.width as u32,
                    self.layers[0].norm.ty,
                    self.eps.to_bits(),
                ],
                [m, 1, 1],
                norm_threads,
            );
        }
        let residual_norm = |norm: &Weight| {
            cmd.dispatch(
                if self.bonsai.is_some() {
                    "bonsai_residual_rms"
                } else if self.mlx {
                    "mlx_residual_rms"
                } else {
                    "residual_rms"
                },
                &[&s.x, &s.delta, &norm.buffer, &s.norm],
                &[
                    self.width as u32,
                    norm.ty,
                    self.eps.to_bits(),
                    1.0f32.to_bits(),
                ],
                [m, 1, 1],
                norm_threads,
            )
        };
        // Append occupancy depends on the append's visible KV domain, not
        // the (often much shorter) text decode riders. Keep their elections
        // separate: using decode length here leaves long image scans serial.
        let attention_lengths = (
            decode_rows
                .iter()
                .map(|&r| rows[r as usize].2 as usize + 1)
                .max()
                .unwrap_or(1),
            tiles
                .chunks_exact(2)
                .map(|tile| {
                    let (slot, _, pos) = rows[(tile[0] + tile[1] - 1) as usize];
                    self.slots[slot]
                        .mm
                        .as_ref()
                        .map_or(pos, |m| m.limit(pos as usize)) as usize
                        + 1
                })
                .max()
                .unwrap_or(1),
        );
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            self.dflash_tap(&cmd, i, m);
            match &layer.mixer {
                Mixer::Linear(w) => {
                    let p = [
                        KEY_HEADS as u32,
                        g.value_heads as u32,
                        g.conv() as u32,
                        m as u32,
                        self.state_slots as u32,
                        w.index as u32,
                        0,
                    ];
                    if self.mlx {
                        self.project(
                            &cmd,
                            &[
                                (&w.qkv, &s.qkv),
                                (&w.z, &s.z),
                                (&w.alpha, &s.alpha),
                                (&w.beta, &s.beta),
                            ],
                            &s.norm,
                            m,
                            &s.gemm,
                        );
                    } else {
                        self.project(&cmd, &[(&w.qkv, &s.qkv), (&w.z, &s.z)], &s.norm, m, &s.gemm);
                        self.project(
                            &cmd,
                            &[(&w.alpha, &s.alpha), (&w.beta, &s.beta)],
                            &s.norm,
                            m,
                            &s.gemm,
                        );
                    }
                    cmd.dispatch(
                        if self.mlx && self.bonsai.is_none() {
                            "mlx_dn_conv"
                        } else {
                            "dn_conv"
                        },
                        &[
                            &s.qkv,
                            &w.conv.buffer,
                            &self.conv,
                            &s.meta,
                            &s.bounds,
                            &s.convolved,
                        ],
                        &p,
                        [(m * g.conv()).div_ceil(256), 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        if self.bonsai.is_some() {
                            "bonsai_dn_qk_norm"
                        } else if self.mlx {
                            "mlx_dn_qk_norm"
                        } else {
                            "dn_qk_norm"
                        },
                        &[&s.convolved],
                        &p,
                        [KEY_HEADS * 2, m, 1],
                        32,
                    );
                    cmd.dispatch(
                        if self.mlx && self.bonsai.is_none() {
                            "mlx_dn_gates"
                        } else {
                            "dn_gates"
                        },
                        &[&s.alpha, &s.beta, &w.a.buffer, &w.dt.buffer, &s.gates],
                        &p,
                        [(m * g.value_heads).div_ceil(256), 1, 1],
                        256,
                    );
                    if let Some(v) = self.spec.as_ref().filter(|_| self.verifying) {
                        cmd.dispatch(
                            "spec_copy",
                            &[&s.qkv, &v.conv_input],
                            &[
                                0,
                                (w.index * v.rows * g.conv()) as u32,
                                (m * g.conv()) as u32,
                            ],
                            [(m * g.conv()).div_ceil(256), 1, 1],
                            256,
                        );
                        let mut vp = p;
                        vp[6] = v.rows as u32;
                        cmd.dispatch(
                            if self.mlx {
                                "mlx_dn_verify"
                            } else {
                                "dn_verify"
                            },
                            &[
                                &s.convolved,
                                &s.gates,
                                &self.state,
                                &s.spans,
                                &s.meta,
                                &s.attn,
                                &v.updates,
                            ],
                            &vp,
                            [
                                if self.mlx { 32 } else { 8 },
                                g.value_heads,
                                spans.len() / 4,
                            ],
                            128,
                        );
                    } else {
                        cmd.dispatch(
                            "dn_conv_commit",
                            &[&s.qkv, &self.conv, &s.spans, &s.meta, &s.checkpoint_spans],
                            &p,
                            [g.conv().div_ceil(256), spans.len() / 4, 1],
                            256,
                        );
                        if self.mlx || !decode_rows.is_empty() {
                            // M5 prefill reuses inputs across value rows. The
                            // dense-27B packed route owns eight rows per SIMD,
                            // retaining each float4 FMA chain and its reduction
                            // tree. Other geometries keep quad reuse; short
                            // decode and verification keep their old elections.
                            let reuse = self.bonsai.is_some()
                                || (self.mlx
                                    && !self.splash
                                    && cmd.tensor_accelerated()
                                    && spans.chunks_exact(4).any(|span| span[1] >= 32));
                            let packed = reuse
                                && self.bonsai.is_none()
                                && self.geometry == Geometry::DENSE_27B;
                            #[cfg(test)]
                            let packed = packed && !BASELINE_RECURRENT_FOR_TEST.with(|v| v.get());
                            cmd.dispatch(
                                if self.bonsai.is_some() {
                                    "bonsai_dn_recurrent"
                                } else if packed {
                                    "mlx_dn_recurrent_packed"
                                } else if reuse {
                                    "mlx_dn_recurrent_quad"
                                } else if self.mlx {
                                    "mlx_dn_recurrent"
                                } else {
                                    "dn_recurrent"
                                },
                                &[
                                    &s.convolved,
                                    &s.gates,
                                    &self.state,
                                    &s.spans,
                                    &s.meta,
                                    &s.attn,
                                    &s.checkpoint_rows,
                                ],
                                &p,
                                [
                                    if packed {
                                        4
                                    } else if self.mlx && !reuse {
                                        32
                                    } else {
                                        8
                                    },
                                    g.value_heads,
                                    spans.len() / 4,
                                ],
                                128,
                            );
                        }
                        if !self.mlx && !chunks.is_empty() {
                            #[cfg(test)]
                            if self.diagnostic_serial_prefill {
                                // GPU-only causal isolation: change just the
                                // recurrence, keeping projections/attention,
                                // prompt chunks and all other math identical.
                                let mut serial = p;
                                serial[6] = 1;
                                cmd.dispatch(
                                    "dn_recurrent",
                                    &[
                                        &s.convolved,
                                        &s.gates,
                                        &self.state,
                                        &s.spans,
                                        &s.meta,
                                        &s.attn,
                                        &s.checkpoint_rows,
                                    ],
                                    &serial,
                                    [8, g.value_heads, spans.len() / 4],
                                    128,
                                );
                            }
                            #[cfg(test)]
                            let chunked = !self.diagnostic_serial_prefill;
                            #[cfg(not(test))]
                            let chunked = true;
                            if chunked {
                                let strict = g.moe() && moe::precise();
                                cmd.dispatch(
                                    if strict {
                                        "dn_chunk_dots_strict"
                                    } else {
                                        "dn_chunk_dots"
                                    },
                                    &[&s.convolved, &s.gates, &s.chunks, &s.prepared],
                                    &p,
                                    [g.value_heads, chunks.len() / 4, 1],
                                    128,
                                );
                                cmd.dispatch(
                                    if strict {
                                        "dn_chunk_prepare_strict"
                                    } else {
                                        "dn_chunk_prepare"
                                    },
                                    &[&s.convolved, &s.gates, &s.chunks, &s.prepared],
                                    &p,
                                    [g.value_heads, chunks.len() / 4, 1],
                                    128,
                                );
                                cmd.dispatch(
                                    if strict {
                                        "dn_chunk_walk_strict"
                                    } else {
                                        "dn_chunk_walk"
                                    },
                                    &[
                                        &s.prepared,
                                        &s.gates,
                                        &s.spans,
                                        &s.chunks,
                                        &s.meta,
                                        &self.state,
                                        &s.attn,
                                    ],
                                    &p,
                                    [8, g.value_heads, spans.len() / 4],
                                    128,
                                );
                            }
                        }
                    }
                    let mut norm_p = p;
                    norm_p[6] = self.eps.to_bits();
                    cmd.dispatch(
                        if self.bonsai.is_some() {
                            "bonsai_dn_gated_norm"
                        } else if self.mlx {
                            "mlx_dn_gated_norm"
                        } else {
                            "dn_gated_norm"
                        },
                        &[&s.attn, &s.z, &w.norm.buffer],
                        &norm_p,
                        [g.value_heads, m, 1],
                        32,
                    );
                    self.project(&cmd, &[(&w.out, &s.delta)], &s.attn, m, &s.gemm);
                }
                Mixer::Full(w) => {
                    self.full_attention(
                        &cmd,
                        w,
                        m,
                        tiles.len() / 2,
                        decode_rows.len(),
                        long_decode_tiles.len() / 2,
                        attention_lengths,
                    );
                }
            }
            residual_norm(&layer.post_norm);
            let compact_ffn = compact_text_ffn
                && crate::affine::prefill_ffn(
                    &cmd,
                    [&layer.gate, &layer.up, &layer.down],
                    &s.norm,
                    [&s.gate, &s.up, &s.delta],
                    m,
                    &s.gemm,
                );
            if !compact_ffn {
                if self.splash {
                    crate::splash::gate_up(
                        &cmd,
                        &layer.gate,
                        &layer.up,
                        &s.norm,
                        &s.gate,
                        m,
                        &s.gemm,
                    );
                } else {
                    self.project(
                        &cmd,
                        &[(&layer.gate, &s.gate), (&layer.up, &s.up)],
                        &s.norm,
                        m,
                        &s.gemm,
                    );
                    cmd.dispatch(
                        if self.mlx && self.bonsai.is_none() {
                            "mlx_swiglu"
                        } else {
                            "swiglu"
                        },
                        &[&s.gate, &s.up],
                        &[(m * self.ff) as u32],
                        [(m * self.ff).div_ceil(256), 1, 1],
                        256,
                    );
                }
                self.project(&cmd, &[(&layer.down, &s.delta)], &s.gate, m, &s.gemm);
            }
            if let Some(experts) = &layer.moe {
                self.moe_ffn(&cmd, experts, m);
            }
            if let Some(next) = self.layers.get(i + 1) {
                residual_norm(&next.norm);
            } else {
                cmd.dispatch(
                    if self.mlx && self.bonsai.is_none() {
                        "mlx_residual"
                    } else {
                        "residual"
                    },
                    &[&s.x, &s.delta],
                    &[(m * self.width) as u32, 1.0f32.to_bits()],
                    [(m * self.width).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        if !selected.is_empty() {
            // Conditioning uses private scratch; the target's final hidden
            // remains intact for MTP and selected-token logits.
            cmd.dispatch(
                if self.bonsai.is_some() {
                    "bonsai_rms_selected"
                } else if self.mlx {
                    "mlx_rms_selected"
                } else {
                    "rms_selected"
                },
                &[&s.x, &self.output_norm.buffer, &s.outputs, &s.norm],
                &[self.width as u32, self.output_norm.ty, self.eps.to_bits()],
                [selected.len(), 1, 1],
                norm_threads,
            );
            self.project(
                &cmd,
                &[(
                    self.head(),
                    if self.verifying {
                        &self
                            .spec
                            .as_ref()
                            .expect("verification buffers allocated")
                            .logits
                    } else {
                        &s.logits
                    },
                )],
                &s.norm,
                selected.len(),
                &s.gemm,
            );
        }
        if self.verifying {
            cmd.dispatch(
                "spec_copy",
                &[
                    &s.x,
                    &self
                        .spec
                        .as_ref()
                        .expect("verification buffers allocated")
                        .hidden,
                ],
                &[0, 0, (m * self.width) as u32],
                [(m * self.width).div_ceil(256), 1, 1],
                256,
            );
        } else {
            self.mtp_catchup(
                &cmd,
                &s.x,
                m,
                tiles.len() / 2,
                decode_rows.len(),
                rows.iter().map(|r| r.2 as usize + 1).max().unwrap_or(1),
            );
        }
        self.dflash_append(&cmd, m);
        for &(index, slot, _) in checkpoints {
            self.dflash_checkpoint(&cmd, slot, self.slots.len() + index);
        }
        if self.greedy_output {
            let v = self.spec.as_ref().expect("verification buffers allocated");
            cmd.dispatch(
                "spec_argmax",
                &[if self.verifying { &v.logits } else { &s.logits }, &v.picks],
                &[self.vocab as u32],
                [selected.len(), 1, 1],
                256,
            );
        }
        self.last_gpu_seconds = cmd.finish()?;
        if self.greedy_output && self.verifying {
            return Ok(Vec::new());
        }
        // SAFETY: GPU writes completed successfully; history cannot precede it.
        let logits = if self.greedy_output {
            Vec::new()
        } else {
            unsafe {
                if self.verifying {
                    &self
                        .spec
                        .as_ref()
                        .expect("verification buffers allocated")
                        .logits
                } else {
                    &s.logits
                }
                .read_f32(0, selected.len() * self.vocab)
            }
        };
        if self.verifying {
            return Ok(logits);
        }
        for &(slot, token, _) in rows {
            self.slots[slot].history.push(token);
        }
        for &(index, slot, ref history) in checkpoints {
            self.cache[index].table.share_prefix(
                &self.slots[slot].table.blocks()[..history.len() / BLOCK_TOKENS],
                &mut self.pool,
            );
            self.cache[index].history.clone_from(history);
            self.cache[index].images = self.slots[slot]
                .mm
                .as_ref()
                .map_or_else(Vec::new, |m| m.keys.clone());
            self.clock += 1;
            self.cache[index].touched = self.clock;
        }
        Ok(logits)
    }
}
