//! In-file or sideloaded nextn. Shared target embeddings/head and page table;
//! one extra paged attention layer. The chain stays on GPU between steps.
use super::*;
use objc2_metal::MTLBuffer;
use paddock_models::mapped::MappedGguf;

pub(super) struct Mtp {
    eh: Weight,
    en: Weight,
    hn: Weight,
    head_norm: Weight,
    layer: Layer,
    pub pending: Buffer,
    h: Buffer,
    concat: Buffer,
    drafted: Buffer,
}

impl Qwen35 {
    pub(super) fn mtp_cold_spans(
        &self,
        slot: usize,
        blocks: &[u32],
    ) -> Vec<crate::offload::Span<'_>> {
        let Some(d) = &self.mtp else {
            return Vec::new();
        };
        let mut spans = vec![(&d.pending, slot * self.width * 4, self.width * 4)];
        if let Mixer::Full(a) = &d.layer.mixer {
            let bytes = BLOCK_TOKENS * self.geometry.kv_heads * 256 * 2;
            for buffer in [&a.keys, &a.values] {
                spans.extend(crate::offload::paged_spans(buffer, blocks, bytes));
            }
        }
        spans
    }
    pub(super) fn reserve_mtp_rows(&mut self, n: usize) -> Result<()> {
        if let Some(d) = &mut self.mtp {
            let h = self.device.alloc(n * self.width * 4)?;
            let concat = self.device.alloc(n * self.width * 8)?;
            d.h = h;
            d.concat = concat;
        }
        Ok(())
    }
    /// Explicit opt-in, keeping spec-off loading/memory/forward unchanged.
    /// Must attach before prefill so draft KV and target state share a cursor.
    pub fn attach_mtp(&mut self, path: &Path) -> Result<()> {
        if self.ternary.is_some() {
            return Err(MetalError::Model(
                "Bonsai PTQ1 has no compatible MTP checkpoint".into(),
            ));
        }
        self.source_versions.extend(crate::offload::versions(path)?);
        let g = self.geometry;
        if self.mlx {
            return Err(MetalError::Model("the native MLX checkpoint has no MTP weights; GGUF MTP cannot be attached to this arithmetic path".into()));
        }
        self.require_committed()?;
        if self.mtp.is_some()
            || self.cold.is_some()
            || self.slots.iter().any(|s| !s.history.is_empty())
            || self.cache.iter().any(|s| !s.history.is_empty())
        {
            return Err(MetalError::Model("attach MTP once, before prefill".into()));
        }
        let map = MappedGguf::open(path).map_err(|e| MetalError::Model(e.to_string()))?;
        if map.gguf().architecture() != Some(if g.moe() { "qwen35moe" } else { "qwen35" })
            || map
                .gguf()
                .arch_field("nextn_predict_layers")
                .and_then(|v| v.as_u64())
                != Some(1)
        {
            return Err(MetalError::Model(
                "MTP requires one qwen35 nextn block".into(),
            ));
        }
        if g.moe() {
            moe::validate(&map, 0)?;
        }
        let count = map
            .gguf()
            .arch_field("block_count")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| MetalError::Model("MTP block count missing".into()))?;
        let index = count
            .checked_sub(1)
            .ok_or_else(|| MetalError::Model("invalid MTP block count".into()))?;
        let w = |name: &str, dims: &[usize]| {
            Weight::load(&self.device, &map, &format!("blk.{index}.{name}"), dims)
        };
        let f32w = |name: &str, dims: &[usize]| -> Result<Weight> {
            let t = w(name, dims)?;
            if t.ty != 0 {
                return Err(MetalError::Model(format!("MTP {name} must be F32")));
            }
            Ok(t)
        };
        let kv = self.pool.capacity() as usize * BLOCK_TOKENS * g.kv_heads * 256 * 2;
        let before = self.device.allocated_bytes();
        let layer = Layer {
            norm: w("attn_norm.weight", &[self.width])?,
            post_norm: w("post_attention_norm.weight", &[self.width])?,
            mixer: Mixer::Full(FullAttention {
                q: w("attn_q.weight", &[self.width, g.heads * 512])?,
                k: w("attn_k.weight", &[self.width, g.kv_heads * 256])?,
                v: w("attn_v.weight", &[self.width, g.kv_heads * 256])?,
                o: w("attn_output.weight", &[g.heads * 256, self.width])?,
                q_norm: f32w("attn_q_norm.weight", &[256])?,
                k_norm: f32w("attn_k_norm.weight", &[256])?,
                keys: self.device.alloc(kv)?,
                values: self.device.alloc(kv)?,
            }),
            gate: w(
                if g.moe() {
                    "ffn_gate_shexp.weight"
                } else {
                    "ffn_gate.weight"
                },
                &[self.width, self.ff],
            )?,
            up: w(
                if g.moe() {
                    "ffn_up_shexp.weight"
                } else {
                    "ffn_up.weight"
                },
                &[self.width, self.ff],
            )?,
            down: w(
                if g.moe() {
                    "ffn_down_shexp.weight"
                } else {
                    "ffn_down.weight"
                },
                &[self.ff, self.width],
            )?,
            // The elected nextn block has BF16 (not F32) routers. Preserve
            // those source bytes through the existing native projection path.
            moe: if g.moe() {
                Some(moe::Experts::load_with(w)?)
            } else {
                None
            },
        };
        let eh = w("nextn.eh_proj.weight", &[self.width * 2, self.width])?;
        let en = w("nextn.enorm.weight", &[self.width])?;
        let hn = w("nextn.hnorm.weight", &[self.width])?;
        let head_norm = w("nextn.shared_head_norm.weight", &[self.width])?;
        let added = self.device.allocated_bytes() - before - kv as u64 * 2;
        self.ensure_verify()?;
        self.mtp = Some(Mtp {
            eh,
            en,
            hn,
            head_norm,
            layer,
            pending: self.device.alloc(self.state_slots * self.width * 4)?,
            h: self.device.alloc(self.row_capacity * self.width * 4)?,
            concat: self.device.alloc(self.row_capacity * self.width * 2 * 4)?,
            drafted: self.device.alloc(self.slots.len() * spec::BLOCK * 4)?,
        });
        self.weight_bytes += added;
        self.kv_bytes += kv as u64 * 2;
        tracing::info!(
            weight_bytes = added,
            kv_bytes = kv * 2,
            "native Metal Qwen MTP attached; exact in-file weights, GPU-resident chain"
        );
        Ok(())
    }

    fn mtp_block(&self, cmd: &Commands<'_>, m: usize, tiles: usize, decodes: usize, length: usize) {
        let d = self.mtp.as_ref().expect("MTP attached");
        let s = &self.scratch;
        let w = &d.layer;
        cmd.dispatch(
            "embed",
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[
                self.width as u32,
                m as u32,
                self.embedding.ty,
                1f32.to_bits(),
            ],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
        if tiles > 0 {
            self.inject_mtp_images(cmd, m);
        }
        cmd.dispatch(
            "rms",
            &[&s.x, &d.en.buffer, &s.norm],
            &[self.width as u32, d.en.ty, self.eps.to_bits()],
            [m, 1, 1],
            256,
        );
        cmd.dispatch(
            "rms",
            &[&s.delta, &d.hn.buffer, &s.x],
            &[self.width as u32, d.hn.ty, self.eps.to_bits()],
            [m, 1, 1],
            256,
        );
        cmd.dispatch(
            "mtp_concat",
            &[&s.norm, &s.x, &d.concat],
            &[self.width as u32, m as u32],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
        if self.geometry.moe() {
            self.project(cmd, &[(&d.eh, &s.x)], &d.concat, m, &s.gemm);
        } else {
            d.eh.linear(cmd, &d.concat, &s.x, m, 1., &s.gemm);
        }
        cmd.dispatch(
            "rms",
            &[&s.x, &w.norm.buffer, &s.norm],
            &[self.width as u32, w.norm.ty, self.eps.to_bits()],
            [m, 1, 1],
            256,
        );
        let Mixer::Full(attn) = &w.mixer else {
            unreachable!()
        };
        // Catchup already supplies the longest position across all rows;
        // standalone drafting has no append tiles. This upper bound covers
        // both domains without underestimating an image's visible KV range.
        self.full_attention(cmd, attn, m, tiles, decodes, 0, (length, length));
        cmd.dispatch(
            "residual_rms",
            &[&s.x, &s.delta, &w.post_norm.buffer, &s.norm],
            &[
                self.width as u32,
                w.post_norm.ty,
                self.eps.to_bits(),
                1f32.to_bits(),
            ],
            [m, 1, 1],
            256,
        );
        let planes = &[(&w.gate, &s.gate), (&w.up, &s.up)];
        if self.geometry.moe() {
            self.project(cmd, planes, &s.norm, m, &s.gemm);
        } else {
            projections(cmd, planes, &s.norm, m, &s.gemm);
        }
        cmd.dispatch(
            "swiglu",
            &[&s.gate, &s.up],
            &[(m * self.ff) as u32],
            [(m * self.ff).div_ceil(256), 1, 1],
            256,
        );
        if self.geometry.moe() {
            self.project(cmd, &[(&w.down, &s.delta)], &s.gate, m, &s.gemm);
        } else {
            w.down.linear(cmd, &s.gate, &s.delta, m, 1., &s.gemm);
        }
        if let Some(experts) = &w.moe {
            self.moe_ffn(cmd, experts, m);
        }
        cmd.dispatch(
            "residual",
            &[&s.x, &s.delta],
            &[(m * self.width) as u32, 1f32.to_bits()],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
    }

    pub(super) fn mtp_catchup(
        &self,
        cmd: &Commands<'_>,
        source: &Buffer,
        m: usize,
        tiles: usize,
        decodes: usize,
        length: usize,
    ) {
        let Some(d) = self.mtp.as_ref() else { return };
        let s = &self.scratch;
        cmd.dispatch(
            "spec_copy",
            &[source, &d.h],
            &[0, 0, (m * self.width) as u32],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "mtp_hidden",
            &[&d.h, &d.pending, &s.meta, &s.bounds, &s.delta],
            &[self.width as u32, m as u32, 0],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
        self.mtp_block(cmd, m, tiles, decodes, length);
        cmd.dispatch(
            "mtp_publish",
            &[&d.h, &d.pending, &s.meta, &s.bounds, &s.checkpoint_rows],
            &[self.width as u32, m as u32],
            [(m * self.width).div_ceil(256), 1, 1],
            256,
        );
    }

    pub(super) fn mtp_draft(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>> {
        self.require_committed()?;
        if self.mtp.is_none() || pendings.is_empty() || k == 0 {
            return Ok(None);
        }
        let n = pendings.len();
        let k = k.min(spec::BLOCK - 1);
        let mut seen = vec![false; self.slots.len()];
        for &(slot, t) in pendings {
            if slot >= seen.len()
                || self.pending.iter().any(|p| p.slot == slot)
                || seen[slot]
                || t as usize >= self.vocab
                || self.slots[slot].history.is_empty()
                || self.slots[slot].history.len() + k >= self.context
            {
                return Ok(None);
            }
            seen[slot] = true;
        }
        for &(slot, _) in pendings {
            let horizon = self.slots[slot].history.len() + k;
            while self.slots[slot]
                .table
                .ensure(horizon, &mut self.pool)
                .is_err()
            {
                if !self.evict_checkpoint() {
                    return Err(MetalError::Memory("MTP KV pool exhausted".into()));
                }
            }
        }
        let mut pages = vec![0; self.slots.len() * self.page_stride];
        for (i, slot) in self.slots.iter().enumerate() {
            pages[i * self.page_stride..i * self.page_stride + slot.table.blocks().len()]
                .copy_from_slice(slot.table.blocks());
        }
        let s = &self.scratch;
        let d = self.mtp.as_ref().expect("MTP attached");
        unsafe {
            s.ids
                .write_u32(&pendings.iter().map(|r| r.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &pendings
                    .iter()
                    .flat_map(|r| [r.0 as u32, self.slots[r.0].history.len() as u32])
                    .collect::<Vec<_>>(),
            );
            s.decode_rows.write_u32(&(0..n as u32).collect::<Vec<_>>());
            s.mrope.write_u32(
                &pendings
                    .iter()
                    .flat_map(|r| self.rope_position(r.0, self.slots[r.0].history.len()))
                    .collect::<Vec<_>>(),
            );
            s.limits.write_u32(
                &pendings
                    .iter()
                    .map(|r| self.slots[r.0].history.len() as u32)
                    .collect::<Vec<_>>(),
            );
            s.pages.write_u32(&pages);
        }
        let length = pendings
            .iter()
            .map(|r| self.slots[r.0].history.len() + k)
            .max()
            .expect("nonempty draft cohort");
        let cmd = self.device.begin()?;
        cmd.dispatch(
            "mtp_hidden",
            &[&d.h, &d.pending, &s.meta, &s.bounds, &s.delta],
            &[self.width as u32, n as u32, 1],
            [(n * self.width).div_ceil(256), 1, 1],
            256,
        );
        for step in 0..k {
            self.mtp_block(&cmd, n, 0, n, length);
            cmd.dispatch(
                "rms",
                &[&s.x, &d.head_norm.buffer, &s.delta],
                &[self.width as u32, d.head_norm.ty, self.eps.to_bits()],
                [n, 1, 1],
                256,
            );
            if self.geometry.moe() {
                self.project(&cmd, &[(self.head(), &s.logits)], &s.delta, n, &s.gemm);
            } else {
                self.head()
                    .linear(&cmd, &s.delta, &s.logits, n, 1., &s.gemm);
            }
            cmd.dispatch(
                "spec_argmax",
                &[&s.logits, &s.ids],
                &[self.vocab as u32],
                [n, 1, 1],
                256,
            );
            cmd.dispatch(
                "mtp_advance",
                &[&s.ids, &s.meta, &d.drafted, &s.mrope, &s.limits],
                &[n as u32, step as u32, k as u32],
                [n.div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.finish()?;
        // SAFETY: whole chain complete; one compact readback, no per-step sync.
        let ids = unsafe {
            std::slice::from_raw_parts(d.drafted.raw.contents().as_ptr().cast::<u32>(), n * k)
                .to_vec()
        };
        Ok(Some(ids.chunks(k).map(|r| r.to_vec()).collect()))
    }
}
