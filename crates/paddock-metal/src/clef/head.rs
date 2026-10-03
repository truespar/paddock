use super::*;
struct Mha {
    q: Linear,
    kv: Linear,
    out: Linear,
}
struct Evidence {
    query: Norm,
    memory: Norm,
    attn: Mha,
    norm: Norm,
    up: Linear,
    down: Linear,
}
struct Decoder {
    n1: Norm,
    n2: Norm,
    n3: Norm,
    qkv: Linear,
    out: Linear,
    cross: Mha,
    up: Linear,
    down: Linear,
}
pub(super) struct Head {
    hidden: Norm,
    memory: Linear,
    question: Linear,
    option_question: Linear,
    global: Linear,
    option_context: Linear,
    option_lexical: Linear,
    types: Buffer,
    evidence: Vec<Evidence>,
    summary: Norm,
    layers: Vec<Decoder>,
    field_norm: Norm,
    option_norm: Norm,
    scorer: Linear,
    score_weight: Buffer,
    score_bias: f32,
    prior: f32,
    joint: f32,
    gate: f32,
}
impl Head {
    pub(super) fn load_gguf(d: &MetalDevice, s: &gguf::Source<'_>, c: &ClefConfig) -> Result<Self> {
        let (w, ff) = (c.head.width, c.head.feedforward);
        let blk = |i: usize, n: &str| format!("dec.blk.{i}.{n}");
        let norm = |name: &str| s.norm(d, name, w, 1e-5, true);
        let lin = |names: &[&str], k, n, bias| s.linear(d, names, k, n, bias);
        let mha = |i| -> Result<Mha> {
            Ok(Mha {
                q: lin(&[&blk(i, "cross_attn_q")], w, w, true)?,
                kv: lin(
                    &[&blk(i, "cross_attn_k"), &blk(i, "cross_attn_v")],
                    w,
                    w,
                    true,
                )?,
                out: lin(&[&blk(i, "cross_attn_o")], w, w, true)?,
            })
        };
        let evidence = (0..c.head.routing_layers)
            .map(|i| {
                Ok(Evidence {
                    query: norm(&blk(i, "cross_attn_norm"))?,
                    memory: norm(&blk(i, "cross_attn_norm_kv"))?,
                    attn: mha(i)?,
                    norm: norm(&blk(i, "ffn_norm"))?,
                    up: lin(&[&blk(i, "ffn_up")], w, ff, true)?,
                    down: lin(&[&blk(i, "ffn_down")], ff, w, true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let layers = (c.head.routing_layers..c.head.routing_layers + c.head.layers)
            .map(|i| {
                Ok(Decoder {
                    n1: norm(&blk(i, "attn_norm"))?,
                    n2: norm(&blk(i, "cross_attn_norm"))?,
                    n3: norm(&blk(i, "ffn_norm"))?,
                    qkv: lin(
                        &[&blk(i, "attn_q"), &blk(i, "attn_k"), &blk(i, "attn_v")],
                        w,
                        w,
                        true,
                    )?,
                    out: lin(&[&blk(i, "attn_o")], w, w, true)?,
                    cross: mha(i)?,
                    up: lin(&[&blk(i, "ffn_up")], w, ff, true)?,
                    down: lin(&[&blk(i, "ffn_down")], ff, w, true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let proj = |n: &str| lin(&[&format!("decision.proj_{n}")], c.hidden, w, false);
        let scales = s.values("decision.scales", &[3])?;
        Ok(Self {
            hidden: s.norm(d, "decision.hidden_norm", c.hidden, 1e-5, true)?,
            memory: proj("memory")?,
            question: proj("question")?,
            option_question: proj("option_question")?,
            global: proj("global")?,
            option_context: proj("option_context")?,
            option_lexical: proj("option_lexical")?,
            types: upload(d, &s.values("token_types.weight", &[w, 3])?)?,
            evidence,
            summary: norm("decision.option_summary_norm")?,
            layers,
            field_norm: norm("decision.field_norm")?,
            option_norm: norm("decision.option_norm")?,
            scorer: lin(&["decision.scorer"], 4 * w, w, true)?,
            score_weight: upload(d, &s.values("decision.scorer_out.weight", &[w, 1])?)?,
            score_bias: s.values("decision.scorer_out.bias", &[1])?[0],
            prior: scales[0],
            joint: scales[1],
            gate: scales[2],
        })
    }
    pub fn load(d: &MetalDevice, s: &load::Source, c: &ClefConfig) -> Result<Self> {
        let w = c.head.width;
        let ff = c.head.feedforward;
        let norm = |n: &str| s.norm(d, n, w, 1e-5, true);
        let mha = |n: &str| -> Result<Mha> {
            Ok(Mha {
                q: s.rows(d, &format!("{n}.in_proj"), 0..w, 3 * w, w, true)?,
                kv: s.rows(d, &format!("{n}.in_proj"), w..3 * w, 3 * w, w, true)?,
                out: s.linear(d, &format!("{n}.out_proj"), w, w, true)?,
            })
        };
        let evidence = (0..c.head.routing_layers)
            .map(|i| {
                let p = format!("evidence_layers.{i}");
                Ok(Evidence {
                    query: norm(&format!("{p}.query_norm"))?,
                    memory: norm(&format!("{p}.memory_norm"))?,
                    attn: mha(&format!("{p}.attention"))?,
                    norm: norm(&format!("{p}.feedforward_norm"))?,
                    up: s.linear(d, &format!("{p}.feedforward.0"), ff, w, true)?,
                    down: s.linear(d, &format!("{p}.feedforward.3"), w, ff, true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let layers = (0..c.head.layers)
            .map(|i| {
                let p = format!("layers.{i}");
                Ok(Decoder {
                    n1: norm(&format!("{p}.norm1"))?,
                    n2: norm(&format!("{p}.norm2"))?,
                    n3: norm(&format!("{p}.norm3"))?,
                    qkv: s.rows(
                        d,
                        &format!("{p}.self_attn.in_proj"),
                        0..3 * w,
                        3 * w,
                        w,
                        true,
                    )?,
                    out: s.linear(d, &format!("{p}.self_attn.out_proj"), w, w, true)?,
                    cross: mha(&format!("{p}.multihead_attn"))?,
                    up: s.linear(d, &format!("{p}.linear1"), ff, w, true)?,
                    down: s.linear(d, &format!("{p}.linear2"), w, ff, true)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let projection = |n: &str| s.linear(d, n, w, c.hidden, false);
        let scalar = |n: &str| -> Result<f32> { Ok(s.values(n, &[])?[0]) };
        Ok(Self {
            hidden: s.norm(d, "hidden_norm", c.hidden, 1e-5, true)?,
            memory: projection("memory_projection")?,
            question: projection("question_projection")?,
            option_question: projection("option_question_projection")?,
            global: projection("global_projection")?,
            option_context: projection("option_context_projection")?,
            option_lexical: projection("option_lexical_projection")?,
            types: upload(d, &s.values("type_embedding.weight", &[3, w])?)?,
            evidence,
            summary: norm("option_summary_norm")?,
            layers,
            field_norm: norm("field_norm")?,
            option_norm: norm("option_norm")?,
            scorer: s.linear(d, "residual_scorer.0", w, 4 * w, true)?,
            score_weight: upload(d, &s.values("residual_scorer.3.weight", &[1, w])?)?,
            score_bias: s.values("residual_scorer.3.bias", &[1])?[0],
            prior: scalar("prior_logit_scale")?.min(100f32.ln()).exp(),
            joint: scalar("joint_logit_scale")?.min(100f32.ln()).exp(),
            gate: 1. / (1. + (-scalar("residual_gate")?).exp()),
        })
    }
}

impl Clef {
    pub(super) fn head_forward(&self, c: &Commands<'_>, p: &plan::Plan) {
        let h = &self.head;
        let s = &self.ws;
        let hw = &s.head;
        let meta = &s.metadata;
        let (rows, nq, no, nr) = (p.ids.len(), p.types.len(), p.qof.len(), p.runs.len() / 2);
        let (dd, w, heads) = (
            self.config.hidden,
            self.config.head.width,
            self.config.head.heads,
        );
        let mean = |x: &Buffer, spans: &Buffer, y: &Buffer, n| {
            point(
                c,
                "clef_mean",
                &[x, spans, y],
                &[dd as u32, n as u32],
                n * dd,
            )
        };
        let gather = |y: &Buffer, x: &Buffer, indices: &Buffer, n| {
            point(
                c,
                "clef_gather_add",
                &[y, x, indices],
                &[w as u32, n as u32],
                n * w,
            )
        };
        // Strided packed K/V and Q/K/V are views, not reshapes/copies. Tiles
        // carry request-local source ranges, including decoder self-attention.
        let attention = |q: &Buffer,
                         k: &Buffer,
                         v: &Buffer,
                         out: &Buffer,
                         tiles: &Buffer,
                         n: usize,
                         qs: usize,
                         ks: usize,
                         vs: usize,
                         qo: usize,
                         ko: usize,
                         vo: usize| {
            c.dispatch(
                "clef_attention",
                &[q, k, v, out, tiles],
                &[
                    heads as u32,
                    heads as u32,
                    qs as u32,
                    ks as u32,
                    vs as u32,
                    qo as u32,
                    ko as u32,
                    vo as u32,
                ],
                [heads, n / 4, 1],
                128,
            );
        };
        h.hidden.run(c, &s.norm, &s.x, rows);
        h.memory.run(c, &s.x, &s.k, rows, 0);
        mean(&s.x, &meta.questions, &hw.qvec, nq);
        mean(&s.x, &meta.globals, &hw.glob, nr);
        mean(&s.x, &meta.options, &hw.ctx, no);
        self.lexical
            .gather(c, &meta.ids, Some(&meta.options), &hw.lex, dd, no);
        h.option_context.run(c, &hw.ctx, &hw.oq, no, 0);
        h.option_lexical.run(c, &hw.lex, &hw.oq, no, 1);
        h.option_question.run(c, &hw.qvec, &hw.qo, nq, 0);
        gather(&hw.oq, &hw.qo, &meta.qof, no);
        for ev in &h.evidence {
            ev.query.run(c, &hw.oq, &hw.tq, no);
            ev.memory.run(c, &s.k, &s.v, rows);
            ev.attn.q.run(c, &hw.tq, &hw.att, no, 0);
            ev.attn.kv.run(c, &s.v, &s.wide, rows, 0);
            attention(
                &hw.att,
                &s.wide,
                &s.wide,
                &hw.tq,
                &meta.option_tiles,
                p.option_tiles.len(),
                w,
                2 * w,
                2 * w,
                0,
                0,
                w,
            );
            ev.attn.out.run(c, &hw.tq, &hw.oq, no, 1);
            ev.norm.run(c, &hw.oq, &hw.tq, no);
            ev.up.run(c, &hw.tq, &hw.wide, no, 2);
            ev.down.run(c, &hw.wide, &hw.oq, no, 1);
        }
        h.question.run(c, &hw.qvec, &hw.fields, nq, 0);
        c.dispatch(
            "clef_route",
            &[&hw.oq, &hw.fields, &meta.qopts, &hw.summ, &hw.logits],
            &[w as u32],
            [nq, 1, 1],
            256,
        );
        h.summary.run(c, &hw.summ, &hw.tq, nq);
        point(
            c,
            "residual",
            &[&hw.fields, &hw.tq],
            &[(nq * w) as u32, 1f32.to_bits()],
            nq * w,
        );
        h.global.run(c, &hw.glob, &hw.gproj, nr, 0);
        gather(&hw.fields, &hw.gproj, &meta.rof, nq);
        gather(&hw.fields, &h.types, &meta.types, nq);
        for l in &h.layers {
            l.n1.run(c, &hw.fields, &hw.tq, nq);
            l.qkv.run(c, &hw.tq, &hw.qkv, nq, 0);
            attention(
                &hw.qkv,
                &hw.qkv,
                &hw.qkv,
                &hw.att,
                &meta.self_tiles,
                p.self_tiles.len(),
                3 * w,
                3 * w,
                3 * w,
                0,
                w,
                2 * w,
            );
            l.out.run(c, &hw.att, &hw.fields, nq, 1);
            l.n2.run(c, &hw.fields, &hw.tq, nq);
            l.cross.q.run(c, &hw.tq, &hw.att, nq, 0);
            l.cross.kv.run(c, &s.k, &s.wide, rows, 0);
            attention(
                &hw.att,
                &s.wide,
                &s.wide,
                &hw.tq,
                &meta.field_tiles,
                p.field_tiles.len(),
                w,
                2 * w,
                2 * w,
                0,
                0,
                w,
            );
            l.cross.out.run(c, &hw.tq, &hw.fields, nq, 1);
            l.n3.run(c, &hw.fields, &hw.tq, nq);
            l.up.run(c, &hw.tq, &hw.wide, nq, 2);
            l.down.run(c, &hw.wide, &hw.fields, nq, 1);
        }
        h.field_norm.run(c, &hw.fields, &hw.summ, nq);
        h.option_norm.run(c, &hw.oq, &hw.onorm, no);
        point(
            c,
            "clef_features",
            &[&hw.summ, &hw.onorm, &meta.qof, &hw.wide],
            &[w as u32, no as u32],
            no * w,
        );
        h.scorer.run(c, &hw.wide, &hw.hid, no, 2);
        c.dispatch(
            "clef_score",
            &[
                &hw.lex,
                &hw.qvec,
                &hw.glob,
                &meta.qof,
                &meta.rof,
                &hw.summ,
                &hw.onorm,
                &hw.hid,
                &h.score_weight,
                &hw.logits,
            ],
            &[
                dd as u32,
                w as u32,
                h.prior.to_bits(),
                h.joint.to_bits(),
                h.gate.to_bits(),
                h.score_bias.to_bits(),
            ],
            [no, 1, 1],
            256,
        );
    }
}
