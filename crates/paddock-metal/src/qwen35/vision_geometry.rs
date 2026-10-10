//! Elected towers, with independent language/tower shape matching. A shared
//! projector name is not permission to execute an arbitrary ViT geometry.
use super::{Result, error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct TowerGeometry {
    pub width: usize,
    pub ff: usize,
    pub layers: usize,
    pub heads: usize,
}

impl TowerGeometry {
    pub fn for_decoder(width: usize) -> Result<Self> {
        let (width, ff, layers, heads) = match width {
            1024 => (768, 3072, 12, 12),
            2560 => (1024, 4096, 24, 16),
            2048 | 4096 | 5120 => (1152, 4304, 27, 16),
            _ => return Err(error("unsupported Qwen vision/language geometry")),
        };
        Ok(Self {
            width,
            ff,
            layers,
            heads,
        })
    }

    pub fn padded_width(self) -> usize {
        self.heads * (self.width / self.heads).next_multiple_of(16)
    }

    pub fn qkv_kernel(self) -> &'static str {
        match (self.width, self.heads) {
            (768, 12) => "vis_qkv64_12",
            (1024, 16) => "vis_qkv64_16",
            _ => "vis_qkv",
        }
    }

    pub fn attention_kernel(self) -> &'static str {
        match (self.width, self.heads) {
            (768, 12) => "vis_attention64_12",
            (1024, 16) => "vis_attention64_16",
            _ => "vis_attention",
        }
    }

    pub fn pillow_resize(self) -> bool {
        // Elected with the LightOn same-checkpoint OCR parity suite. Preserve
        // the older tower's preprocessing until its own new-reference gate.
        self.width != 1152
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elected_towers_do_not_pad_small_models_to_the_27b_tower() {
        for (decoder, width, ff, layers, heads, padded) in [
            (1024, 768, 3072, 12, 12, 768),
            (2560, 1024, 4096, 24, 16, 1024),
            (5120, 1152, 4304, 27, 16, 1280),
        ] {
            let g = TowerGeometry::for_decoder(decoder).unwrap();
            assert_eq!(
                (g.width, g.ff, g.layers, g.heads, g.padded_width()),
                (width, ff, layers, heads, padded)
            );
        }
        assert!(TowerGeometry::for_decoder(1280).is_err());
    }
}
