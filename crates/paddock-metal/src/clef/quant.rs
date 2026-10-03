//! Resident checkpoint planes: never expand a quantized model into BF16.
//! Dequantization is tile-local on the GPU; the same exact-weight arithmetic
//! serves backbone, schema head and sparse embedding/lexical gathers.
use super::*;

pub(super) struct Plane {
    pub data: Buffer,
    // 0 BF16, 1 GGUF Q8_0, 2 MLX affine-8/group-64 with BF16 side planes.
    pub kind: u32,
    pub affine: Option<(Buffer, Buffer)>,
}
impl From<Buffer> for Plane {
    fn from(data: Buffer) -> Self {
        Self {
            data,
            kind: 0,
            affine: None,
        }
    }
}
impl Plane {
    pub fn buffers(&self) -> [&Buffer; 3] {
        let (scale, bias) = self
            .affine
            .as_ref()
            .map_or((&self.data, &self.data), |(s, b)| (s, b));
        [&self.data, scale, bias]
    }
    pub fn gather(
        &self,
        c: &Commands<'_>,
        ids: &Buffer,
        spans: Option<&Buffer>,
        y: &Buffer,
        width: usize,
        rows: usize,
    ) {
        let [w, s, b] = self.buffers();
        point(
            c,
            "clef_quant_gather",
            &[w, s, b, ids, spans.unwrap_or(ids), y],
            &[
                width as u32,
                rows as u32,
                self.kind,
                u32::from(spans.is_some()),
            ],
            width * rows,
        );
    }
}
impl Linear {
    pub(super) fn quantized(
        &self,
        c: &Commands<'_>,
        x: &Buffer,
        y: &Buffer,
        rows: usize,
        epi: u32,
        parts: bool,
    ) {
        let [w, s, b] = self.weight.buffers();
        c.dispatch(
            "clef_mm_quant",
            &[w, s, b, x, y, self.bias.as_ref().unwrap_or(x)],
            &[
                self.k as u32,
                self.n as u32,
                rows as u32,
                self.weight.kind,
                u32::from(parts),
                u32::from(self.bias.is_some()),
                epi,
            ],
            [self.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
    }
}
