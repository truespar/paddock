//! Elected hybrid shapes, not an open-ended family-name allowlist.
//! Kernel head widths/rotary/interval remain independently checked by load.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Geometry {
    pub width: usize,
    pub ff: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub value_heads: usize,
}

impl Geometry {
    pub const DENSE_08B: Self = Self {
        width: 1024,
        ff: 3584,
        layers: 24,
        heads: 8,
        kv_heads: 2,
        value_heads: 16,
    };
    pub const DENSE_4B: Self = Self {
        width: 2560,
        ff: 9216,
        layers: 32,
        heads: 16,
        kv_heads: 4,
        value_heads: 32,
    };
    pub const DENSE_9B: Self = Self {
        width: 4096,
        ff: 12288,
        layers: 32,
        heads: 16,
        kv_heads: 4,
        value_heads: 32,
    };
    pub const DENSE_27B: Self = Self {
        width: 5120,
        ff: 17408,
        layers: 64,
        heads: 24,
        kv_heads: 4,
        value_heads: 48,
    };

    pub const MOE_35B: Self = Self {
        width: 2048,
        ff: 512,
        layers: 40,
        heads: 16,
        kv_heads: 2,
        value_heads: 32,
    };

    pub fn moe(self) -> bool {
        self == Self::MOE_35B
    }

    pub fn prepared_element_bytes(self) -> usize {
        if self.moe() { 4 } else { 2 }
    }

    // FFN width does not bound the attention/DeltaNet inputs of a tiny-expert
    // model. This shared conversion workspace must cover their widest plane.
    pub fn projection_width(self) -> usize {
        self.ff
            .max(self.conv())
            .max(self.width * 2)
            .max(self.heads * 512)
    }

    pub fn validate(self) -> Result<Self> {
        if [
            Self::DENSE_08B,
            Self::DENSE_4B,
            Self::DENSE_9B,
            Self::DENSE_27B,
            Self::MOE_35B,
        ]
        .contains(&self)
        {
            Ok(self)
        } else {
            Err(MetalError::Model(format!(
                "Metal Qwen requires elected dense 0.8B/4B/9B/27B or MoE 35B geometry, got {self:?}"
            )))
        }
    }

    pub fn linear_layers(self) -> usize {
        self.layers - self.full_layers()
    }
    pub fn full_layers(self) -> usize {
        self.layers / 4
    }
    pub fn conv(self) -> usize {
        (KEY_HEADS * 2 + self.value_heads) * 128
    }
    pub fn state(self) -> usize {
        self.value_heads * 128 * 128
    }
    pub fn decode_kernel(self) -> &'static str {
        match self.heads / self.kv_heads {
            4 => "qwen_attention_decode_gqa4",
            6 => "qwen_attention_decode",
            8 => "qwen_attention_decode_gqa8",
            _ => unreachable!("validated Qwen GQA ratio"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_shapes_bound_shared_scratch_and_state_without_27b_padding() {
        for (g, linear, state, conv) in [
            (Geometry::DENSE_08B, 18, 262144, 6144),
            (Geometry::DENSE_4B, 24, 524288, 8192),
            (Geometry::DENSE_9B, 24, 524288, 8192),
            (Geometry::DENSE_27B, 48, 786432, 10240),
            (Geometry::MOE_35B, 30, 524288, 8192),
        ] {
            assert_eq!(g.validate().unwrap(), g);
            assert_eq!(g.linear_layers(), linear);
            assert_eq!(g.state(), state);
            assert_eq!(g.conv(), conv);
            // Attention and DeltaNet share output storage. Prepared DeltaNet
            // chunks cover four independent BK32 attention partitions/head.
            assert_eq!(g.heads * 256, g.value_heads * 128);
            assert!(g.value_heads * 17408 >= g.heads * 4 * 32 * 256);
            assert!(CHUNK * g.projection_width() >= (CHUNK + 32) * g.heads * 256);
            assert!(g.projection_width() >= g.conv().max(g.width * 2));
        }
        for invalid in [
            Geometry {
                heads: 24,
                ..Geometry::DENSE_9B
            },
            Geometry {
                value_heads: 48,
                ..Geometry::DENSE_9B
            },
            Geometry {
                layers: 33,
                ..Geometry::DENSE_9B
            },
            Geometry {
                width: 8192,
                ..Geometry::DENSE_27B
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }
}
