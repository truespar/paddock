//! Fixed image-encoder residency, separate from bounded request-size staging.
//! Only dimensions are evaluated here; no checkpoint/model math on the CPU.
use super::*;

impl Sam3Vision {
    pub(super) fn planned_weight_bytes() -> u64 {
        let (d, f, ff) = (1024u64, 256u64, 4736u64);
        // F16 matrices; two F32 norms and all five projection biases.
        let block = 2 * (4 * d * d + 2 * d * ff) + 4 * (9 * d + ff);
        // Three transposed convolutions, three 1x1 and three 3x3 per neck.
        let neck = 2 * (d * (d / 2) * 4 + (d / 2) * (d / 4) * 4 + d * (d / 2) * 4)
            + 2 * ((d / 4 + d / 2 + d) * f + 3 * 9 * f * f)
            + 4 * (d / 2 + d / 4 + d / 2 + 6 * f);
        let patch = 592 * d * 2;
        let pos_norm = (576 * d + 2 * d) * 4;
        let skips = (f * (f / 8 + f / 4)) * 2 + (f / 8 + f / 4) * 4;
        let rope = (576 + 5184) * 64 * 4;
        let tiles = 2 * 162 * 4 * 4;
        patch + pos_norm + 32 * block + 2 * neck + skips + rope + tiles
    }
    /// Component residency only. Complete SAM 3 detector/tracker serving
    /// must add its own heads, text and video-memory reservations to this.
    pub fn resident_bytes_required() -> u64 {
        Self::planned_weight_bytes() + Workspace::required_bytes()
    }
}

impl Workspace {
    pub(super) fn required_bytes() -> u64 {
        let row = 5184u64 * 1024;
        let levels = (288 * 288 + 144 * 144 + 72 * 72) * 256;
        // Seven half row planes: norm, projection, Q/K/V, attention, raster.
        let half = 5184 * 592 + 7 * row + 5184 * 4736 + 144 * 144 * 512 + 288 * 288 * 256;
        let f32 = row + 3 * row + 144 * 144 * 1024 + 2 * levels + 288 * 288 * 32 + 144 * 144 * 64;
        1008 * 1008 * 3 + half * 2 + f32 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sam3_workspace_reservation_is_exact_and_released() {
        let d = MetalDevice::new(Some(Workspace::required_bytes())).unwrap();
        let before = d.allocated_bytes();
        let ws = Workspace::new(&d).unwrap();
        assert_eq!(d.allocated_bytes() - before, Workspace::required_bytes());
        assert!(
            d.alloc(1).is_err(),
            "exact reservation accepted one extra byte"
        );
        drop(ws);
        assert_eq!(d.allocated_bytes(), before);
    }
}
