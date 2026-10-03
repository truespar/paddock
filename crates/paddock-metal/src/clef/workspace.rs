use super::*;
pub(super) struct Metadata {
    pub ids: Buffer,
    pub runs: Buffer,
    pub bounds: Buffer,
    pub questions: Buffer,
    pub options: Buffer,
    pub globals: Buffer,
    pub qof: Buffer,
    pub rof: Buffer,
    pub types: Buffer,
    pub qopts: Buffer,
    pub causal: Buffer,
    pub option_tiles: Buffer,
    pub field_tiles: Buffer,
    pub self_tiles: Buffer,
    pub positions: Buffer,
}
impl Metadata {
    pub fn sizes() -> [usize; 15] {
        [
            MAX_ROWS,
            2 * MAX_REQUESTS,
            2 * MAX_ROWS,
            2 * MAX_QUESTIONS,
            2 * MAX_OPTIONS,
            2 * MAX_REQUESTS,
            MAX_OPTIONS,
            MAX_QUESTIONS,
            MAX_QUESTIONS,
            2 * MAX_QUESTIONS,
            4 * (MAX_ROWS.div_ceil(16) + MAX_REQUESTS),
            4 * (MAX_OPTIONS.div_ceil(16) + MAX_REQUESTS),
            4 * (MAX_QUESTIONS.div_ceil(16) + MAX_REQUESTS),
            4 * (MAX_QUESTIONS.div_ceil(16) + MAX_REQUESTS),
            3 * MAX_ROWS,
        ]
    }
    fn new(d: &MetalDevice) -> Result<Self> {
        let [
            ids,
            runs,
            bounds,
            questions,
            options,
            globals,
            qof,
            rof,
            types,
            qopts,
            causal,
            option_tiles,
            field_tiles,
            self_tiles,
            positions,
        ] = Self::sizes().map(|n| d.alloc(n * 4));
        Ok(Self {
            ids: ids?,
            runs: runs?,
            bounds: bounds?,
            questions: questions?,
            options: options?,
            globals: globals?,
            qof: qof?,
            rof: rof?,
            types: types?,
            qopts: qopts?,
            causal: causal?,
            option_tiles: option_tiles?,
            field_tiles: field_tiles?,
            self_tiles: self_tiles?,
            positions: positions?,
        })
    }
    pub fn write(&self, p: &plan::Plan) {
        // SAFETY: one execution thread, previous pass fully fenced; Plan bounds
        // all planes before this point. No buffers escape this model.
        for (b, v) in [
            (&self.ids, &p.ids),
            (&self.runs, &p.runs),
            (&self.bounds, &p.bounds),
            (&self.questions, &p.questions),
            (&self.options, &p.options),
            (&self.globals, &p.globals),
            (&self.qof, &p.qof),
            (&self.rof, &p.rof),
            (&self.types, &p.types),
            (&self.qopts, &p.qopts),
            (&self.causal, &p.causal),
            (&self.option_tiles, &p.option_tiles),
            (&self.field_tiles, &p.field_tiles),
            (&self.self_tiles, &p.self_tiles),
            (&self.positions, &p.positions),
        ] {
            unsafe {
                b.write_u32(v);
            }
        }
    }
}
pub(super) struct HeadWs {
    pub qvec: Buffer,
    pub glob: Buffer,
    pub ctx: Buffer,
    pub lex: Buffer,
    pub wide: Buffer,
    pub oq: Buffer,
    pub tq: Buffer,
    pub att: Buffer,
    pub qo: Buffer,
    pub fields: Buffer,
    pub summ: Buffer,
    pub gproj: Buffer,
    pub qkv: Buffer,
    pub onorm: Buffer,
    pub hid: Buffer,
    pub logits: Buffer,
}
impl HeadWs {
    fn sizes(c: &ClefConfig) -> [usize; 16] {
        let (q, o, r, w, d) = (
            MAX_QUESTIONS,
            MAX_OPTIONS,
            MAX_REQUESTS,
            c.head.width,
            c.hidden,
        );
        [
            q * d,
            r * d,
            o * d,
            o * d,
            o * c.head.feedforward.max(4 * w),
            o * w,
            o * w,
            o * w,
            q * w,
            q * w,
            q * w,
            r * w,
            q * 3 * w,
            o * w,
            o * w,
            o,
        ]
    }
    fn new(d: &MetalDevice, c: &ClefConfig) -> Result<Self> {
        let [
            qvec,
            glob,
            ctx,
            lex,
            wide,
            oq,
            tq,
            att,
            qo,
            fields,
            summ,
            gproj,
            qkv,
            onorm,
            hid,
            logits,
        ] = Self::sizes(c).map(|n| d.alloc(n * 4));
        Ok(Self {
            qvec: qvec?,
            glob: glob?,
            ctx: ctx?,
            lex: lex?,
            wide: wide?,
            oq: oq?,
            tq: tq?,
            att: att?,
            qo: qo?,
            fields: fields?,
            summ: summ?,
            gproj: gproj?,
            qkv: qkv?,
            onorm: onorm?,
            hid: hid?,
            logits: logits?,
        })
    }
}
pub(super) struct Workspace {
    pub metadata: Metadata,
    pub head: HeadWs,
    pub x: Buffer,
    pub norm: Buffer,
    pub wide: Buffer,
    pub q: Buffer,
    pub k: Buffer,
    pub v: Buffer,
    pub z: Buffer,
    pub core: Buffer,
    pub ffn: Buffer,
    pub ab: Buffer,
    pub gates: Buffer,
    pub conv: Buffer,
}
impl Workspace {
    fn sizes(c: &ClefConfig) -> [usize; 12] {
        let r = MAX_ROWS;
        [
            r * c.hidden,
            r * c.hidden,
            r * c.gdn_qkv_rows().max(c.attn_q_rows()).max(2 * c.head.width),
            r * c.q_width(),
            r * c.kv_width().max(c.head.width),
            r * c.kv_width().max(c.head.width),
            r * c.gdn_v_width(),
            r * c.q_width().max(c.gdn_v_width()),
            r * c.ffn,
            r * 2 * c.gdn_v_heads,
            r * 2 * c.gdn_v_heads,
            r * c.gdn_qkv_rows(),
        ]
    }
    pub fn bytes(c: &ClefConfig) -> u64 {
        ((Self::sizes(c).iter().sum::<usize>()
            + HeadWs::sizes(c).iter().sum::<usize>()
            + Metadata::sizes().iter().sum::<usize>())
            * 4) as u64
    }
    pub fn new(d: &MetalDevice, c: &ClefConfig) -> Result<Self> {
        let [x, norm, wide, q, k, v, z, core, ffn, ab, gates, conv] =
            Self::sizes(c).map(|n| d.alloc(n * 4));
        Ok(Self {
            metadata: Metadata::new(d)?,
            head: HeadWs::new(d, c)?,
            x: x?,
            norm: norm?,
            wide: wide?,
            q: q?,
            k: k?,
            v: v?,
            z: z?,
            core: core?,
            ffn: ffn?,
            ab: ab?,
            gates: gates?,
            conv: conv?,
        })
    }
}
