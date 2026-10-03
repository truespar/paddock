use super::*;
#[cfg(test)]
thread_local! {
    pub(super) static STRICT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static LINEAR_WALK_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static UNBLOCKED_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
impl Clef {
    pub(super) fn backbone(
        &self,
        cmd: &Commands<'_>,
        p: &plan::Plan,
        images: &[(usize, usize, Buffer)],
    ) {
        let c = &self.config;
        let s = &self.ws;
        let m = p.ids.len();
        let meta = &s.metadata;
        // BF16 tensor projections consume a two-plane activation split. The
        // packed Q8/affine kernels accumulate in F32 already: keep their input
        // exact, and avoid both the extra dispatches and needless rounding.
        let split = self.layers[0].gate_up.weight.kind == 0;
        #[cfg(test)]
        let split = split && !STRICT_FOR_TEST.with(|v| v.get());
        // The convolution/K scratch is dead at every projection boundary.
        // Its bytes hold both BF16 activation parts without another allocation.
        let prepare = |x: &Buffer, width: usize| {
            if split {
                point(
                    cmd,
                    "clef_prepare",
                    &[x, &s.conv],
                    &[(m * width) as u32],
                    m * width,
                );
            }
        };
        let project = |l: &Linear, x: &Buffer, y: &Buffer, epi: u32| {
            if split {
                l.parts(cmd, &s.conv, y, m, epi);
            } else {
                l.run(cmd, x, y, m, epi);
            }
        };
        self.embed.gather(cmd, &meta.ids, None, &s.x, c.hidden, m);
        for (row, count, image) in images {
            point(
                cmd,
                "spec_copy",
                &[image, &s.x],
                &[0, (row * c.hidden) as u32, (count * c.hidden) as u32],
                count * c.hidden,
            );
        }
        #[cfg(test)]
        if let Some(trace) = &self.trace {
            point(
                cmd,
                "spec_copy",
                &[&s.x, trace],
                &[0, 0, (m * c.hidden) as u32],
                m * c.hidden,
            );
        }
        // Layer indices are used only by the test-only GPU trace capture.
        #[cfg_attr(not(test), allow(clippy::unused_enumerate_index))]
        for (_index, layer) in self.layers.iter().enumerate() {
            layer.norm.run(cmd, &s.x, &s.norm, m);
            #[cfg(test)]
            if _index == 0
                && let Some(trace) = &self.trace
            {
                point(
                    cmd,
                    "spec_copy",
                    &[&s.norm, trace],
                    &[
                        0,
                        ((self.layers.len() + 1) * m * c.hidden) as u32,
                        (m * c.hidden) as u32,
                    ],
                    m * c.hidden,
                );
            }
            prepare(&s.norm, c.hidden);
            match &layer.mixer {
                Mixer::Delta(l) => {
                    project(&l.qkv, &s.norm, &s.wide, 0);
                    project(&l.z, &s.norm, &s.z, 0);
                    project(&l.ab, &s.norm, &s.ab, 0);
                    #[cfg(test)]
                    if _index == 0
                        && let Some(trace) = &self.trace
                    {
                        point(
                            cmd,
                            "spec_copy",
                            &[&s.wide, trace],
                            &[
                                0,
                                ((self.layers.len() + 2) * m * c.hidden) as u32,
                                (m * c.gdn_qkv_rows()) as u32,
                            ],
                            m * c.gdn_qkv_rows(),
                        );
                    }
                    point(
                        cmd,
                        "clef_conv",
                        &[&s.wide, &l.conv, &meta.bounds, &s.conv],
                        &[c.gdn_qkv_rows() as u32, m as u32],
                        m * c.gdn_qkv_rows(),
                    );
                    let dp = [
                        c.gdn_k_heads as u32,
                        c.gdn_v_heads as u32,
                        c.gdn_qkv_rows() as u32,
                        m as u32,
                        0,
                        0,
                        c.eps.to_bits(),
                    ];
                    #[cfg(test)]
                    if _index == 0
                        && let Some(trace) = &self.trace
                    {
                        point(
                            cmd,
                            "spec_copy",
                            &[&s.conv, trace],
                            &[
                                0,
                                ((self.layers.len() + 4) * m * c.hidden) as u32,
                                (m * c.gdn_qkv_rows()) as u32,
                            ],
                            m * c.gdn_qkv_rows(),
                        );
                    }
                    cmd.dispatch(
                        if self.rms_qk {
                            "clef_mlx_qk_norm"
                        } else {
                            "dn_qk_norm"
                        },
                        &[&s.conv],
                        &dp,
                        [2 * c.gdn_k_heads, m, 1],
                        32,
                    );
                    point(
                        cmd,
                        "clef_gates",
                        &[&s.ab, &l.a, &l.dt, &s.gates],
                        &[c.gdn_v_heads as u32, m as u32],
                        m * c.gdn_v_heads,
                    );
                    cmd.dispatch(
                        "clef_recurrent",
                        &[&s.conv, &s.gates, &meta.runs, &s.core],
                        &[
                            c.gdn_v_heads as u32,
                            c.gdn_k_heads as u32,
                            u32::from(self.tiled_heads),
                        ],
                        [8, c.gdn_v_heads, p.runs.len() / 2],
                        128,
                    );
                    cmd.dispatch(
                        "dn_gated_norm",
                        &[&s.core, &s.z, &l.norm],
                        &dp,
                        [c.gdn_v_heads, m, 1],
                        32,
                    );
                    prepare(&s.core, c.gdn_v_width());
                    project(&l.out, &s.core, &s.x, 1);
                }
                Mixer::Attention(l) => {
                    project(&l.q, &s.norm, &s.wide, 0);
                    project(&l.k, &s.norm, &s.k, 0);
                    project(&l.v, &s.norm, &s.v, 0);
                    cmd.dispatch(
                        "clef_rope",
                        &[&s.wide, &l.qnorm, &self.rope, &meta.positions, &s.q],
                        &[
                            c.n_heads as u32,
                            c.head_dim as u32,
                            c.n_rot as u32,
                            c.attn_q_rows() as u32,
                            1,
                            c.eps.to_bits(),
                            c.mrope_sections[1],
                            c.mrope_sections[2],
                        ],
                        [c.n_heads, m, 1],
                        32,
                    );
                    // Distinct output; no rotary lane races with its partner.
                    cmd.dispatch(
                        "clef_rope",
                        &[&s.k, &l.knorm, &self.rope, &meta.positions, &s.conv],
                        &[
                            c.n_kv_heads as u32,
                            c.head_dim as u32,
                            c.n_rot as u32,
                            c.kv_width() as u32,
                            0,
                            c.eps.to_bits(),
                            c.mrope_sections[1],
                            c.mrope_sections[2],
                        ],
                        [c.n_kv_heads, m, 1],
                        32,
                    );
                    cmd.dispatch(
                        "clef_causal",
                        &[&s.q, &s.conv, &s.v, &s.core, &meta.causal],
                        &[
                            c.n_heads as u32,
                            c.n_kv_heads as u32,
                            c.q_width() as u32,
                            c.kv_width() as u32,
                            c.kv_width() as u32,
                            0,
                            0,
                            0,
                        ],
                        [c.n_heads, p.causal.len() / 4, 1],
                        128,
                    );
                    point(
                        cmd,
                        "clef_attn_gate",
                        &[&s.core, &s.wide],
                        &[c.q_width() as u32, m as u32, c.head_dim as u32],
                        m * c.q_width(),
                    );
                    prepare(&s.core, c.q_width());
                    project(&l.out, &s.core, &s.x, 1);
                }
            }
            layer.post.run(cmd, &s.x, &s.norm, m);
            if split {
                prepare(&s.norm, c.hidden);
                layer.gate_up.parts(cmd, &s.conv, &s.ffn, m, 3);
                layer.down.parts(cmd, &s.ffn, &s.x, m, 1);
            } else {
                layer.gate_up.run(cmd, &s.norm, &s.ffn, m, 3);
                layer.down.run(cmd, &s.ffn, &s.x, m, 1);
            }
            #[cfg(test)]
            if let Some(trace) = &self.trace {
                point(
                    cmd,
                    "spec_copy",
                    &[&s.x, trace],
                    &[
                        0,
                        ((_index + 1) * m * c.hidden) as u32,
                        (m * c.hidden) as u32,
                    ],
                    m * c.hidden,
                );
            }
        }
        self.norm.run(cmd, &s.x, &s.norm, m);
    }
}
