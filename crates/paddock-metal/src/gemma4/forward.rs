use super::*;

#[cfg(test)]
type BatchTrace = Vec<(String, Vec<f32>)>;

#[cfg(test)]
thread_local! {
    pub(super) static NAX_PREFILL_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(super) static BASELINE_ROUNDED_MERGE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BASELINE_DECODE_AHEAD_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static FIXED_DECODE_SPLITS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static SINGLE_DECODE_PROJECTION_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static BATCH_TRACE_FOR_TEST: std::cell::RefCell<Option<BatchTrace>> = const { std::cell::RefCell::new(None) };
}

// Do not trade padded arithmetic for another weight sweep. Retain the shared
// ladder for narrow rows, full chunks and every other model. The 129..192 and
// 257..288 tails use the same number of weight tiles but 25% fewer matrix rows.
pub(super) fn prefill96(rows: usize) -> bool {
    (129..=CHUNK).contains(&rows) && rows.div_ceil(96) == rows.div_ceil(128)
}

pub(super) fn prefill96_projection(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    m: usize,
    workspace: &Buffer,
) {
    let k = planes[0].0.k;
    assert!(
        planes
            .iter()
            .all(|(w, _)| w.k == k && matches!(w.ty, 12 | 13 | 14 | 23))
    );
    cmd.dispatch(
        "linear_input_padded",
        &[input, workspace],
        &[k as u32, 0, m as u32],
        [
            (k.div_ceil(128) * 128 * m.div_ceil(128) * 128).div_ceil(256),
            1,
            1,
        ],
        256,
    );
    if planes.len() == 1 {
        let (w, out) = planes[0];
        cmd.dispatch(
            "linear_ktile96",
            &[&w.buffer, workspace, out],
            &[k as u32, w.n as u32, m as u32, w.ty, 1f32.to_bits()],
            [w.n.div_ceil(32), m.div_ceil(96), 1],
            128,
        );
    } else {
        assert!(matches!(planes.len(), 2 | 3));
        let third = planes.get(2).unwrap_or(&planes[1]);
        cmd.dispatch(
            "linear_multi_ktile96",
            &[
                &planes[0].0.buffer,
                &planes[1].0.buffer,
                &third.0.buffer,
                workspace,
                planes[0].1,
                planes[1].1,
                third.1,
            ],
            &[
                k as u32,
                planes[0].0.n as u32,
                planes[1].0.n as u32,
                if planes.len() == 3 {
                    third.0.n as u32
                } else {
                    0
                },
                m as u32,
                planes[0].0.ty,
                planes[1].0.ty,
                third.0.ty,
            ],
            [
                planes.iter().map(|(w, _)| w.n.div_ceil(32)).sum(),
                m.div_ceil(96),
                1,
            ],
            128,
        );
    }
}

pub(super) use crate::weights::paired_projections as pair_projection;
impl Gemma4 {
    fn project(&self, cmd: &Commands<'_>, planes: &[(&Weight, &Buffer)], input: &Buffer, m: usize) {
        if self.diffusion.is_some() {
            for &(w, out) in planes {
                diffusion::project(cmd, w, input, out, m, self.mlx);
            }
            return;
        }
        if self.mlx {
            #[cfg(test)]
            if (2..=4).contains(&m) && SINGLE_DECODE_PROJECTION_FOR_TEST.get() {
                if planes[0].0.k == 5376 {
                    let (a, ao) = planes[0];
                    let (b, bo) = *planes.get(1).unwrap_or(&planes[0]);
                    let (c, co) = *planes.get(2).unwrap_or(&planes[0]);
                    cmd.dispatch(
                        "mlx_affine_bias8",
                        &[input, &self.scratch.gemm],
                        &[a.k as u32, m as u32],
                        [(m * a.k / 8).div_ceil(256), 1, 1],
                        256,
                    );
                    cmd.dispatch(
                        [
                            "mlx_affine_canonical2",
                            "mlx_affine_canonical3",
                            "mlx_affine_canonical4",
                        ][m - 2],
                        &[
                            &a.buffer,
                            &b.buffer,
                            &c.buffer,
                            input,
                            ao,
                            bo,
                            co,
                            &self.scratch.gemm,
                        ],
                        &[
                            a.k as u32,
                            a.n as u32,
                            if planes.len() > 1 { b.n as u32 } else { 0 },
                            if planes.len() > 2 { c.n as u32 } else { 0 },
                            m as u32,
                        ],
                        [planes.iter().map(|(w, _)| w.n.div_ceil(16)).sum(), 1, 1],
                        128,
                    );
                    return;
                }
                crate::affine::project_verify(cmd, planes, input, m, &self.scratch.gemm, true);
                return;
            }
            crate::affine::project(cmd, planes, input, m, &self.scratch.gemm);
            return;
        }
        if self.moe_scratch.is_some() {
            moe::project(cmd, planes, input, m, &self.scratch.gemm);
            return;
        }
        if self.muse && !self.verifying && m >= 1024 && planes.iter().all(|(w, _)| w.ty == 8) {
            muse::image_projection(cmd, planes, input, m, &self.scratch.gemm);
            return;
        }
        if !self.verifying
            && prefill96(m)
            && planes
                .iter()
                .all(|(w, _)| matches!(w.ty, 12 | 13 | 14 | 23))
        {
            prefill96_projection(cmd, planes, input, m, &self.scratch.gemm);
            return;
        }
        if (3..=4).contains(&m) && planes.iter().all(|(w, _)| matches!(w.ty, 12..=14)) {
            pair_projection(cmd, planes, input, m);
            return;
        }
        if !self.verifying
            || (m <= 4
                && planes
                    .iter()
                    .all(|(w, _)| matches!(w.ty, 12 | 13 | 14 | 23)))
        {
            // K-quant narrow decode already retains F32 operands. Its fused
            // independent projections are also valid during verification;
            // Q8's half-activation decode election is deliberately excluded.
            if planes.len() == 1 {
                planes[0]
                    .0
                    .linear(cmd, input, planes[0].1, m, 1., &self.scratch.gemm);
            } else {
                projections(cmd, planes, input, m, &self.scratch.gemm);
            }
            return;
        }
        // Verification must stay close to the single-token target graph:
        // F16 operands accumulated a >0.1 logit error on a ring-wrap probe.
        // Retain F32 activations/dequantized values. Register reuse wins
        // narrow rows; strict-F32 MPP removes the measured wide-rung cost.
        for &(w, out) in planes {
            let p = [w.k as u32, w.n as u32, m as u32, w.ty, 1f32.to_bits()];
            let (kernel, columns, rows) = if matches!(w.ty, 12 | 13 | 14 | 23) {
                if m > 8 {
                    cmd.dispatch(
                        "gemma_staged_f32_32",
                        &[&w.buffer, input, out],
                        &p,
                        [w.n.div_ceil(16), m.div_ceil(32), 1],
                        128,
                    );
                    continue;
                }
                match m {
                    1 => ("linear_kquant1", 4, 1),
                    2 => ("linear_kquant2", 16, 2),
                    3 => ("linear_kquant3", 16, 3),
                    4 => ("linear_kquant4", 16, 4),
                    5 => ("gemma_full5", 16, 5),
                    6 => ("gemma_full6", 16, 6),
                    7 => ("gemma_full7", 16, 7),
                    8 => ("gemma_full8", 16, 8),
                    _ => ("gemma_verify16", 16, 16),
                }
            } else if w.ty == 8 {
                if self.muse && m > 8 {
                    // Same F32 operands, bounded Q8 staging. The register
                    // rung below scaled poorly at Muse's K=6656/19968:
                    // measured 7-20x projection losses at 16-64 verify rows.
                    // Keep the 202k-token head's distinct shape election.
                    let (kernel, bm) = if w.n == self.vocab && m <= 16 {
                        ("muse_q8_f32_16", 16)
                    } else if w.n == self.vocab && m >= 64 {
                        ("muse_q8_f32_64", 64)
                    } else {
                        ("muse_q8_f32_32", 32)
                    };
                    cmd.dispatch(
                        kernel,
                        &[&w.buffer, input, out],
                        &p,
                        [w.n.div_ceil(16), m.div_ceil(bm), 1],
                        128,
                    );
                    continue;
                }
                if m == 1 {
                    ("linear_q8_r1", 4, 1)
                } else if m > 8 {
                    ("linear_q8_r16", 4, 16)
                } else {
                    ("linear_q8_r8", 4, 8)
                }
            } else {
                ("linear", 4, 1)
            };
            cmd.dispatch(
                kernel,
                &[&w.buffer, input, out],
                &p,
                [w.n.div_ceil(columns), m.div_ceil(rows), 1],
                128,
            );
        }
    }
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
    ) -> Result<Vec<f32>> {
        self.require_committed()?;
        if self
            .prefill_phase
            .as_ref()
            .is_some_and(|p| rows.iter().any(|r| p.contains(r.0)))
        {
            return Err(MetalError::Model(
                "Gemma slot has a suspended prefill".into(),
            ));
        }
        objc2::rc::autoreleasepool(|_| {
            self.execute_inner(rows, outputs, 0..self.layers.len(), None)
        })
    }
    pub(super) fn execute_slice(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
        layers: std::ops::Range<usize>,
        carry: &sliced_prefill::Carry,
    ) -> Result<Vec<f32>> {
        self.require_committed()?;
        if self.verifying || layers.is_empty() || layers.end > self.layers.len() {
            return Err(MetalError::Model(
                "invalid Gemma prefill layer range".into(),
            ));
        }
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, outputs, layers, Some(carry)))
    }
    fn execute_inner(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
        layers: std::ops::Range<usize>,
        carry: Option<&sliced_prefill::Carry>,
    ) -> Result<Vec<f32>> {
        let final_slice = layers.end == self.layers.len();
        let canvas = self.diffusion.as_ref().is_some_and(|d| !d.pass.is_empty());
        let m = rows.len();
        if m == 0
            || m > self.scratch.rows
            || outputs.len()
                > if canvas {
                    CHUNK
                } else if self.verifying {
                    self.spec.as_ref().expect("verification allocated").rows
                } else {
                    self.slots.len()
                }
            || outputs.iter().any(|&r| r >= m)
        {
            return Err(MetalError::Model(
                "invalid Gemma execution/output rows".into(),
            ));
        }
        let mut lengths: Vec<_> = self.slots.iter().map(|s| s.history.len()).collect();
        for &(slot, tok, pos) in rows {
            if slot >= lengths.len()
                || tok as usize >= self.vocab
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(MetalError::Model(format!(
                    "invalid Gemma row slot={slot}, token={tok}, position={pos}"
                )));
            }
            lengths[slot] += 1;
        }
        let mut copies = Vec::new();
        let limits = rows
            .iter()
            .map(|r| {
                if canvas {
                    let pass = self
                        .diffusion
                        .as_ref()
                        .expect("canvas lane")
                        .pass
                        .iter()
                        .find(|p| p.slot == r.0)
                        .expect("validated canvas slot");
                    return (pass.base + pass.width - 1) as u32;
                }
                self.slots[r.0].mm.as_ref().map_or(r.2 as usize, |m| {
                    if self.muse {
                        r.2 as usize
                    } else {
                        m.limit(r.2 as usize)
                    }
                }) as u32
            })
            .collect::<Vec<_>>();
        let image_batch = rows.iter().zip(&limits).any(|(r, &limit)| limit > r.2);
        for (r, &limit) in rows.iter().zip(&limits) {
            if limit as usize >= lengths[r.0] {
                return Err(MetalError::Model(
                    "Gemma prefill split a bidirectional image span".into(),
                ));
            }
        }
        for &(slot, _, pos) in rows {
            loop {
                if self.slots[slot]
                    .table
                    .ensure(pos as usize, &mut self.pool)
                    .is_ok()
                {
                    break;
                }
                if !self.evict() {
                    for &(source, _) in &copies {
                        self.pool.release(source);
                    }
                    return Err(MetalError::Memory("Gemma global KV exhausted".into()));
                }
            }
            loop {
                match self.slots[slot].table.cow_at(pos as usize, &mut self.pool) {
                    Ok(Some(pair)) => {
                        // A later allocation may evict the checkpoint that
                        // owned the source. Keep its bytes alive until the GPU
                        // copy completes, including across subsequent COWs.
                        self.pool.retain(pair.0);
                        copies.push(pair);
                        break;
                    }
                    Ok(None) => break,
                    Err(_) => {
                        if !self.evict() {
                            for &(source, _) in &copies {
                                self.pool.release(source);
                            }
                            return Err(MetalError::Memory(
                                "Gemma KV copy-on-write exhausted".into(),
                            ));
                        }
                    }
                }
            }
        }
        let mut pages = vec![0; self.slots.len() * self.page_stride];
        for (i, slot) in self.slots.iter().enumerate() {
            let b = slot.table.blocks();
            pages[i * self.page_stride..i * self.page_stride + b.len()].copy_from_slice(b);
        }
        let mut tiles = Vec::new();
        let mut decode = Vec::new();
        // A diffusion canvas has one shared retained prefix window, not a
        // different sliding floor for each query. This is independent of the
        // checkpoint's weight format; only MLX's arithmetic rounds QK/P to BF16.
        let diffusion_attention = self.diffusion.is_some() && (self.mlx || canvas);
        let mut first = 0;
        while first < m {
            let mut end = first + 1;
            while end < m && rows[end].0 == rows[first].0 {
                end += 1;
            }
            // A multi-token span reuses one K/V tile for every causal query,
            // including narrow MTP verification. One 32-query launch per
            // individual draft row wastes nearly the entire matrix tile.
            if end - first > 1 || diffusion_attention {
                let tile = if diffusion_attention { 16 } else { 32 };
                for at in (first..end).step_by(tile) {
                    tiles.extend([at as u32, (end - at).min(tile) as u32]);
                }
            } else {
                decode.extend((first..end).map(|r| r as u32));
            }
            first = end;
        }
        // The split workspace is provisioned per output slot. Small prefill
        // segments can contain more than one row per slot, so tile those too
        // rather than indexing a decode-only allocation out of bounds.
        if self.verifying {
            tiles.clear();
            decode = (0..m as u32).collect();
        } else if decode.len() > self.slots.len() {
            for row in decode.drain(..) {
                tiles.extend([row, 1]);
            }
        }
        let length = decode
            .iter()
            .map(|&r| rows[r as usize].2 as usize + 1)
            .max()
            .unwrap_or(1);
        let s = &self.scratch;
        let norm_threads = if self.mlx {
            self.width.div_ceil(128).min(32) * 32
        } else {
            256
        };
        let logits = if canvas {
            &self.diffusion.as_ref().expect("canvas lane").logits
        } else if self.verifying {
            &self.spec.as_ref().expect("verification allocated").logits
        } else {
            &s.logits
        };
        unsafe {
            s.ids
                .write_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &rows
                    .iter()
                    .flat_map(|r| [r.0 as u32, r.2])
                    .collect::<Vec<_>>(),
            );
            s.pages.write_u32(&pages);
            s.limits.write_u32(&limits);
            s.tiles.write_u32(&tiles);
            s.decode_rows.write_u32(&decode);
            s.outputs
                .write_u32(&outputs.iter().map(|&r| r as u32).collect::<Vec<_>>());
        }
        self.dflash_rows(rows);
        let cmd = match self.device.begin() {
            Ok(cmd) => cmd,
            Err(e) => {
                for &(source, _) in &copies {
                    self.pool.release(source);
                }
                return Err(e);
            }
        };
        #[cfg(test)]
        let mut diagnostic = Vec::<(String, Buffer, usize)>::new();
        macro_rules! capture {
            ($name:expr, $buffer:expr, $count:expr) => {{
                #[cfg(test)]
                if (self.diffusion.is_some() && std::env::var_os("PADDOCK_DG_TRACE").is_some())
                    || BATCH_TRACE_FOR_TEST.with(|v| v.borrow().is_some())
                {
                    let count = $count;
                    let buffer = self.device.alloc(count * 4)?;
                    copy_words(&cmd, $buffer, &buffer, 0, 0, count);
                    diagnostic.push(($name.to_string(), buffer, count));
                }
            }};
        }
        for layer in self.layers.iter().filter(|l| !l.sliding) {
            let words = BLOCK_TOKENS * layer.kv_width() / 2;
            for &(from, to) in &copies {
                for buffer in [&layer.keys, &layer.values] {
                    copy_words(
                        &cmd,
                        buffer,
                        buffer,
                        from as usize * words,
                        to as usize * words,
                        words,
                    );
                }
            }
        }
        if layers.start > 0 {
            let carry = carry.expect("resumed prefill has GPU state");
            copy_words(&cmd, &carry.x, &s.x, 0, 0, m * self.width);
            copy_words(&cmd, &carry.norm, &s.norm, 0, 0, m * self.width);
            if let (Some(d), Some(taps)) = (&self.dflash, &carry.taps) {
                copy_words(&cmd, taps, &d.taps, 0, 0, m * self.width * 5);
            }
        } else {
            cmd.dispatch(
                if self.diffusion.is_some() {
                    "dg_embed"
                } else if self.mlx {
                    "gmlx_embed"
                } else {
                    "embed"
                },
                &[&self.embedding.buffer, &s.ids, &s.x],
                &[
                    self.width as u32,
                    m as u32,
                    if self.diffusion.is_some() || self.mlx {
                        self.vocab as u32
                    } else {
                        self.embedding.ty
                    },
                    if self.diffusion.is_some() {
                        self.embedding.ty
                    } else {
                        (if self.muse {
                            1.
                        } else {
                            (self.width as f32).sqrt()
                        })
                        .to_bits()
                    },
                    u32::from(self.mlx),
                ],
                [(m * self.width).div_ceil(256), 1, 1],
                256,
            );
            if let Some(lane) = &self.diffusion
                && canvas
            {
                lane.preamble(&cmd, &self.embedding, &s.x, self.eps, self.mlx);
            }
            capture!("embedding", &s.x, m * self.width);
            self.inject_images(&cmd, rows);
            if self.muse {
                if self.mlx {
                    cmd.dispatch(
                        "gmlx_norm",
                        &[&s.x, &self.output_norm.buffer, &s.x],
                        &[self.width as u32, 2, self.eps.to_bits()],
                        [m, 1, 1],
                        norm_threads,
                    );
                } else {
                    cmd.dispatch(
                        "muse_embedding_norm",
                        &[&s.x],
                        &[self.width as u32, self.eps.to_bits()],
                        [m, 1, 1],
                        256,
                    );
                }
            }
            cmd.dispatch(
                if self.mlx { "gmlx_norm" } else { "rms" },
                &[&s.x, &self.layers[0].norm.buffer, &s.norm],
                &[
                    self.width as u32,
                    if self.mlx {
                        u32::from(self.muse)
                    } else {
                        self.layers[0].norm.ty
                    },
                    self.eps.to_bits(),
                ],
                [m, 1, 1],
                norm_threads,
            );
        }
        let sandwich = |post: &Weight, next: &Weight, scale: f32| {
            if self.mlx {
                cmd.dispatch(
                    "gmlx_sandwich",
                    &[&s.x, &s.delta, &post.buffer, &next.buffer, &s.norm],
                    &[
                        self.width as u32,
                        u32::from(self.muse),
                        self.eps.to_bits(),
                        if self.muse {
                            1e-8f32.to_bits()
                        } else {
                            self.eps.to_bits()
                        },
                        scale.to_bits(),
                    ],
                    [m, 1, 1],
                    norm_threads,
                );
                return;
            }
            cmd.dispatch(
                if self.muse {
                    "muse_sandwich"
                } else {
                    "gemma_sandwich"
                },
                &[&s.x, &s.delta, &post.buffer, &next.buffer, &s.norm],
                &[
                    self.width as u32,
                    post.ty,
                    next.ty,
                    self.eps.to_bits(),
                    scale.to_bits(),
                ],
                [m, 1, 1],
                256,
            )
        };
        for (i, l) in self
            .layers
            .iter()
            .enumerate()
            .take(layers.end)
            .skip(layers.start)
        {
            self.dflash_tap(&cmd, i, m);
            let hd = l.hd();
            let rounded_merge = self.mlx
                && !self.muse
                && l.attn_gate.is_none()
                && self.device.tensor_accelerated()
                && decode.len() == m
                && matches!(hd, 256 | 512);
            #[cfg(test)]
            let rounded_merge = rounded_merge && !BASELINE_ROUNDED_MERGE_FOR_TEST.get();
            let kh = l.kh();
            let heads = l.heads;
            let planes = if let Some(v) = &l.v {
                vec![(&l.q, &s.q), (&l.k, &s.k), (v, &s.v)]
            } else {
                vec![(&l.q, &s.q), (&l.k, &s.k)]
            };
            if i == 0 {
                capture!("stage.norm", &s.norm, m * self.width);
            }
            self.project(&cmd, &planes, &s.norm, m);
            if i == 0 {
                capture!("stage.qraw", &s.q, m * heads * hd);
                capture!("stage.kraw", &s.k, m * kh * hd);
            }
            let rope = self.rope[usize::from(!l.sliding)];
            cmd.dispatch(
                if self.mlx {
                    if self.muse {
                        "mmlx_qnorm"
                    } else {
                        "gmlx_qnorm"
                    }
                } else if self.muse {
                    "muse_qnorm"
                } else {
                    "gemma_qnorm"
                },
                &[&s.q, &l.q_norm.buffer, &s.meta, &self.factors.buffer],
                &[
                    heads as u32,
                    hd as u32,
                    l.q_norm.ty,
                    self.eps.to_bits(),
                    rope.to_bits(),
                    u32::from(!l.sliding),
                ],
                [heads, m, 1],
                32,
            );
            let mut p = vec![
                heads as u32,
                kh as u32,
                self.page_stride as u32,
                if l.sliding { self.window as u32 } else { 0 },
                self.ring as u32,
            ];
            p.extend([hd as u32, l.k_norm.ty, self.eps.to_bits(), rope.to_bits()]);
            cmd.dispatch(
                if self.mlx {
                    if self.muse {
                        "mmlx_store"
                    } else {
                        "gmlx_store"
                    }
                } else if self.muse {
                    "muse_kv_store"
                } else {
                    "gemma_kv_store"
                },
                &[
                    &s.k,
                    if l.v.is_some() { &s.v } else { &s.k },
                    &l.k_norm.buffer,
                    &s.meta,
                    &s.pages,
                    &self.factors.buffer,
                    &l.keys,
                    &l.values,
                ],
                &p,
                [kh, m, 1],
                32,
            );
            p.truncate(5);
            if i == 0 {
                capture!("stage.qrot", &s.q, m * heads * hd);
            }
            if diffusion_attention {
                let mut params = p.clone();
                params.extend([u32::from(canvas), u32::from(self.mlx)]);
                params.extend(self.slots.iter().map(|s| s.history.len() as u32));
                cmd.dispatch(
                    match (self.mlx, hd) {
                        (true, 256) => "dg_attention256",
                        (true, _) => "dg_attention512",
                        (false, 256) => "dg_gguf_attention256",
                        (false, _) => "dg_gguf_attention512",
                    },
                    &[
                        &s.q, &l.keys, &l.values, &s.meta, &s.pages, &s.attn, &s.tiles, &s.limits,
                    ],
                    &params,
                    [heads, tiles.len() / 2, 1],
                    128,
                );
            } else if !tiles.is_empty() {
                let strict = self.mlx || self.moe_scratch.is_some();
                // Qualified dense 31B text geometry only. Keep vision,
                // diffusion, MoE, speculative serving and Apple7–9 on
                // their existing arithmetic contracts. No runtime opt-in.
                let nax = self.mlx
                    && !self.muse
                    && self.diffusion.is_none()
                    && self.moe_scratch.is_none()
                    && self.spec.is_none()
                    && !self.verifying
                    && self.width == 5376
                    && self.layers.len() == 60
                    && rows
                        .iter()
                        .all(|&(slot, _, _)| self.slots[slot].mm.is_none())
                    && !image_batch
                    && matches!(hd, 256 | 512)
                    && cmd.tensor_accelerated();
                #[cfg(test)]
                let nax = nax && NAX_PREFILL_FOR_TEST.get().unwrap_or(true);
                if nax {
                    assert!(s.gemm.len() >= m * heads * hd * 2);
                    cmd.dispatch(
                        "mlx_input_compact",
                        &[&s.q, &s.gemm],
                        &[(heads * hd) as u32, m as u32],
                        [(m * heads * hd).div_ceil(256), 1, 1],
                        256,
                    );
                }
                if !strict {
                    cmd.dispatch(
                        "attention_query",
                        &[&s.q, &s.gemm],
                        &[(heads * hd) as u32, 0, m as u32],
                        [((m + 32) * heads * hd).div_ceil(256), 1, 1],
                        256,
                    );
                }
                let mut buffers = vec![
                    if strict && !nax { &s.q } else { &s.gemm },
                    &l.keys,
                    &l.values,
                    &s.meta,
                    &s.pages,
                    &s.attn,
                    &s.tiles,
                ];
                if image_batch {
                    buffers.push(&s.limits);
                }
                cmd.dispatch(
                    if nax {
                        if hd == 256 {
                            "gmlx_nax_prefill256"
                        } else {
                            "gmlx_nax_prefill512"
                        }
                    } else if self.mlx {
                        match (image_batch, hd) {
                            (true, 256) => "gmlx_image_prefill256",
                            (true, 512) => "gmlx_image_prefill512",
                            (_, 128) => "gmlx_prefill128",
                            (_, 256) => "gmlx_prefill256",
                            _ => "gmlx_prefill512",
                        }
                    } else if self.muse {
                        "muse_prefill"
                    } else if strict {
                        match (image_batch, l.sliding) {
                            (true, true) => "gmoe_image_prefill256",
                            (true, false) => "gmoe_image_prefill512",
                            (false, true) => "gmoe_prefill256",
                            (false, false) => "gmoe_prefill512",
                        }
                    } else {
                        match (image_batch, l.sliding) {
                            (true, true) => "gemma_image_prefill256",
                            (true, false) => "gemma_image_prefill512",
                            (false, true) => "gemma_prefill256",
                            (false, false) => "gemma_prefill512",
                        }
                    },
                    &buffers,
                    &p,
                    [heads, tiles.len() / 2, if nax && hd == 512 { 2 } else { 1 }],
                    128,
                );
            }
            if !decode.is_empty() {
                let visible = if l.sliding {
                    length.min(self.window)
                } else {
                    length
                };
                let splits = if self.moe_scratch.is_some() {
                    // Near-tied routing magnifies different split/merge
                    // reductions. A fixed domain also covers verify blocks
                    // crossing a 128-token boundary and mixed-length slots;
                    // each GPU row still bounds its own visible positions.
                    SPLITS
                } else {
                    visible
                        .div_ceil(128)
                        .max(16usize.div_ceil(decode.len()))
                        .clamp(1, SPLITS)
                };
                #[cfg(test)]
                let splits = if FIXED_DECODE_SPLITS_FOR_TEST.get() {
                    visible.div_ceil(128).clamp(16, SPLITS)
                } else {
                    splits
                };
                p.push(splits as u32);
                let decode_ahead = self.mlx
                    && !self.muse
                    && self.width == 5376
                    && self.moe_scratch.is_none()
                    && self.diffusion.is_none()
                    && self.device.tensor_accelerated()
                    && decode.len() == 4
                    && visible >= 1024;
                #[cfg(test)]
                let decode_ahead = decode_ahead && !BASELINE_DECODE_AHEAD_FOR_TEST.get();
                cmd.dispatch(
                    if self.mlx {
                        match hd {
                            256 if decode_ahead => "gmlx_value2_256",
                            128 => "gmlx_decode128",
                            256 => "gmlx_decode256",
                            _ => "gmlx_decode512",
                        }
                    } else if self.muse {
                        // Uninstrumented M5 election: cooperative matrix KV
                        // reuse wins the long c=4 cell; its setup loses narrow
                        // contexts. GPU validation timings invert this result.
                        if decode.len() == 4 && visible >= 4096 {
                            "muse_decode"
                        } else {
                            "muse_decode_vector"
                        }
                    } else if l.sliding {
                        "gemma_decode256"
                    } else {
                        "gemma_decode512"
                    },
                    &[
                        &s.q,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.decode_rows,
                        &s.parts,
                    ],
                    &p,
                    [kh, decode.len(), splits],
                    128,
                );
                cmd.dispatch(
                    if self.muse {
                        "muse_merge"
                    } else if rounded_merge {
                        "gmlx_merge_rounded"
                    } else {
                        "gemma_merge_shared"
                    },
                    &[&s.parts, &s.attn, &s.decode_rows],
                    &[heads as u32, splits as u32, hd as u32],
                    [heads * decode.len(), 1, 1],
                    32,
                );
            }
            if let Some(gate) = &l.attn_gate {
                self.project(&cmd, &[(gate, &s.attn_gate)], &s.norm, m);
                cmd.dispatch(
                    if self.mlx { "gmlx_gate" } else { "muse_gate" },
                    &[&s.attn, &s.attn_gate],
                    &[(m * heads * hd) as u32],
                    [(m * heads * hd).div_ceil(256), 1, 1],
                    256,
                );
            }
            if self.mlx && l.attn_gate.is_none() && !rounded_merge {
                cmd.dispatch(
                    "gmlx_round",
                    &[&s.attn],
                    &[(m * heads * hd) as u32],
                    [(m * heads * hd).div_ceil(256), 1, 1],
                    256,
                );
            }
            self.project(&cmd, &[(&l.o, &s.delta)], &s.attn, m);
            if i == 0 {
                capture!("stage.attn", &s.attn, m * heads * hd);
            }
            if i == 0 {
                capture!("stage.attn_delta", &s.delta, m * self.width);
            }
            sandwich(&l.post_attn, &l.ffn_norm, 1.);
            if i == 0 {
                capture!("stage.ffn_norm", &s.norm, m * self.width);
            }
            self.project(&cmd, &[(&l.gate, &s.gate), (&l.up, &s.up)], &s.norm, m);
            if i == 0 {
                capture!("stage.gate", &s.gate, m * self.ff);
                capture!("stage.up", &s.up, m * self.ff);
            }
            cmd.dispatch(
                if self.mlx {
                    if self.muse {
                        "mlx_swiglu"
                    } else {
                        "gmlx_geglu"
                    }
                } else if self.muse {
                    "muse_swiglu"
                } else {
                    "gemma_geglu"
                },
                &[&s.gate, &s.up],
                &[(m * self.ff) as u32],
                [(m * self.ff).div_ceil(256), 1, 1],
                256,
            );
            self.project(&cmd, &[(&l.down, &s.delta)], &s.gate, m);
            if i == 0 {
                capture!("stage.shared", &s.delta, m * self.width);
            }
            if let Some(experts) = &l.moe {
                self.moe_scratch
                    .as_ref()
                    .expect("MoE workspace")
                    .execute(&cmd, experts, &s.x, &s.delta, m, self.eps);
                #[cfg(test)]
                if i == 0 {
                    for (name, buffer, count) in self.moe_scratch.as_ref().unwrap().diagnostic(m) {
                        capture!(name, buffer, count);
                    }
                }
            }
            sandwich(
                &l.post_ffn,
                self.layers
                    .get(i + 1)
                    .map_or(&self.output_norm, |l| &l.norm),
                l.scale,
            );
            capture!(i, &s.x, m * self.width);
        }
        if !final_slice {
            // Preserve both values: recomputing the next RMS after a yield
            // would introduce a different arithmetic path at layer cuts.
            let carry = carry.expect("partial prefill has GPU state");
            copy_words(&cmd, &s.x, &carry.x, 0, 0, m * self.width);
            copy_words(&cmd, &s.norm, &carry.norm, 0, 0, m * self.width);
            if let (Some(d), Some(taps)) = (&self.dflash, &carry.taps) {
                copy_words(&cmd, &d.taps, taps, 0, 0, m * self.width * 5);
            }
        }
        if final_slice {
            self.dflash_append(&cmd, m);
        }
        if final_slice && !outputs.is_empty() {
            cmd.dispatch(
                if self.mlx {
                    "mlx_rms_selected"
                } else {
                    "rms_selected"
                },
                &[&s.x, &self.output_norm.buffer, &s.outputs, &s.norm],
                &[self.width as u32, self.output_norm.ty, self.eps.to_bits()],
                [outputs.len(), 1, 1],
                norm_threads,
            );
            if self.verifying {
                copy_words(
                    &cmd,
                    &s.norm,
                    &self.spec.as_ref().expect("verification allocated").hidden,
                    0,
                    0,
                    outputs.len() * self.width,
                );
            } else if let Some(d) = &self.mtp {
                for (i, &row) in outputs.iter().enumerate() {
                    copy_words(
                        &cmd,
                        &s.norm,
                        &d.pending,
                        i * self.width,
                        rows[row].0 * self.width,
                        self.width,
                    );
                }
            }
            self.project(
                &cmd,
                &[(self.output.as_ref().unwrap_or(&self.embedding), logits)],
                &s.norm,
                outputs.len(),
            );
            cmd.dispatch(
                if self.diffusion.is_some() {
                    "gemma_softcap"
                } else if self.mlx {
                    "gmlx_softcap"
                } else if self.muse {
                    "muse_softcap"
                } else {
                    "gemma_softcap"
                },
                &[logits],
                &[
                    (outputs.len() * self.vocab) as u32,
                    self.softcap.to_bits(),
                    self.logit_scale.to_bits(),
                ],
                [(outputs.len() * self.vocab).div_ceil(256), 1, 1],
                256,
            );
            if self.greedy_verify {
                cmd.dispatch(
                    "spec_argmax",
                    &[
                        logits,
                        &self.spec.as_ref().expect("verification allocated").picks,
                    ],
                    &[self.vocab as u32],
                    [outputs.len(), 1, 1],
                    256,
                );
            }
        }
        let completed = cmd.finish();
        for &(source, _) in &copies {
            self.pool.release(source);
        }
        self.last_gpu_seconds = completed?;
        #[cfg(test)]
        BATCH_TRACE_FOR_TEST.with(|v| {
            if let Some(trace) = v.borrow_mut().as_mut() {
                for (name, buffer, count) in &diagnostic {
                    trace.push((name.clone(), unsafe { buffer.read_f32(0, *count / m) }));
                }
            }
        });
        #[cfg(test)]
        if let Some(root) = std::env::var_os("PADDOCK_DG_TRACE") {
            let root = std::path::Path::new(&root);
            std::fs::create_dir_all(root).map_err(|e| MetalError::Model(e.to_string()))?;
            for (name, buffer, count) in diagnostic {
                let bytes = unsafe { buffer.read_f32(0, count) }
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect::<Vec<_>>();
                std::fs::write(
                    root.join(format!(
                        "{}.{name}.f32",
                        if canvas { "canvas" } else { "encoder" }
                    )),
                    bytes,
                )
                .map_err(|e| MetalError::Model(e.to_string()))?;
            }
        }
        if !final_slice {
            // Uncommitted KV is private to the pending slot. Neither its
            // history nor MTP cursor/prefix snapshot becomes visible yet.
            return Ok(Vec::new());
        }
        if !self.verifying && !canvas {
            for &(slot, tok, _) in rows {
                self.slots[slot].history.push(tok);
                if let Some(d) = &mut self.mtp {
                    d.cursor[slot] = None;
                }
            }
            if let Some(d) = &mut self.mtp {
                for &row in outputs {
                    let (slot, _, pos) = rows[row];
                    d.cursor[slot] = Some(pos as usize + 1);
                }
            }
        }
        Ok(if self.greedy_verify || canvas {
            Vec::new()
        } else {
            unsafe { logits.read_f32(0, outputs.len() * self.vocab) }
        })
    }
}
