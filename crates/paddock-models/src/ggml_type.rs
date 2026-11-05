//! GGML tensor types as they appear in GGUF files.
//!
//! IDs verified against ggml-org/llama.cpp master `ggml.h`
//! (GGML_TYPE_COUNT = 42; MXFP4 = 39, NVFP4 = 40). Unknown IDs are carried as
//! data, not errors - a newer file must not brick the parser, but anything we
//! can't size gets flagged instead of guessed.
//!
//! Two ids sit far outside that range on purpose: PrismML's ternary packings
//! (142, 143), private to their llama.cpp fork and numbered clear of upstream's
//! counter so the two can never collide. Layouts verified against the fork's
//! `ggml-common.h` (branch `prism`).

/// A tensor's storage type. `Unknown` keeps forward-compat: we can list and
/// route around tensors we can't execute yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
    /// PrismML: one trit in a 2-bit slot, one f16 scale per 128 weights.
    Pq2_0,
    /// PrismML: five trits a byte (base 3), one f16 scale per 128 weights.
    Ptq1_0,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_raw(raw: u32) -> Self {
        use GgmlType::*;
        match raw {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2K,
            11 => Q3K,
            12 => Q4K,
            13 => Q5K,
            14 => Q6K,
            15 => Q8K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => Bf16,
            34 => Tq1_0,
            35 => Tq2_0,
            39 => Mxfp4,
            40 => Nvfp4,
            41 => Q1_0,
            142 => Pq2_0,
            143 => Ptq1_0,
            other => Unknown(other),
        }
    }

    /// (elements per block, bytes per block) - the pair that turns dims into
    /// byte sizes. None = we haven't verified the layout yet; callers must treat
    /// that as "cannot size", never assume.
    /// The GGUF `ggml_type` id (the inverse of [`GgmlType::from_raw`]).
    pub fn raw(&self) -> u32 {
        use GgmlType::*;
        match *self {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q4_1 => 3,
            Q5_0 => 6,
            Q5_1 => 7,
            Q8_0 => 8,
            Q8_1 => 9,
            Q2K => 10,
            Q3K => 11,
            Q4K => 12,
            Q5K => 13,
            Q6K => 14,
            Q8K => 15,
            Iq2Xxs => 16,
            Iq2Xs => 17,
            Iq3Xxs => 18,
            Iq1S => 19,
            Iq4Nl => 20,
            Iq3S => 21,
            Iq2S => 22,
            Iq4Xs => 23,
            I8 => 24,
            I16 => 25,
            I32 => 26,
            I64 => 27,
            F64 => 28,
            Iq1M => 29,
            Bf16 => 30,
            Tq1_0 => 34,
            Tq2_0 => 35,
            Mxfp4 => 39,
            Nvfp4 => 40,
            Q1_0 => 41,
            Pq2_0 => 142,
            Ptq1_0 => 143,
            Unknown(other) => other,
        }
    }

    pub fn block_layout(&self) -> Option<(usize, usize)> {
        use GgmlType::*;
        Some(match self {
            F32 => (1, 4),
            F16 | Bf16 => (1, 2),
            F64 | I64 => (1, 8),
            I32 => (1, 4),
            I16 => (1, 2),
            I8 => (1, 1),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Q2K => (256, 84),
            Q3K => (256, 110),
            Q4K => (256, 144),
            Q5K => (256, 176),
            Q6K => (256, 210),
            Q8K => (256, 292),
            Iq4Nl => (32, 18),
            Iq4Xs => (256, 136),
            // the i-quant family: codebook-indexed 8-weight groups (ggml-common.h)
            Iq2Xxs => (256, 66),
            Iq2Xs => (256, 74),
            Iq2S => (256, 82),
            Iq3Xxs => (256, 98),
            Iq3S => (256, 110),
            Iq1S => (256, 50),
            Iq1M => (256, 56),
            // 32 elems: 1-byte shared E8M0 scale + 16 bytes of packed FP4
            Mxfp4 => (32, 17),
            // 64 elems: 4 UE4M3 scales (one per 16) + 32 bytes of packed FP4
            // (gguf_nvfp4 repacks it to the row layout)
            Nvfp4 => (64, 36),
            // 128 elems: f16 d + 32 bytes of 2-bit slots (code - 1 = the trit)
            Pq2_0 => (128, 34),
            // 128 elems: qs[24] (5 trits a byte) + qh[2] (4 trits a byte) + f16 d
            Ptq1_0 => (128, 28),
            // iq1/iq2/iq3/tq/q1_0 layouts not yet verified against ggml -
            // fill in when a model in our matrix actually needs them
            _ => return None,
        })
    }

    /// Byte size of `n_elements` stored in this type, if the layout is known
    /// and n divides cleanly into blocks.
    pub fn byte_size(&self, n_elements: u64) -> Option<u64> {
        let (block_elems, block_bytes) = self.block_layout()?;
        let be = block_elems as u64;
        if !n_elements.is_multiple_of(be) {
            return None; // partial blocks don't exist in valid files
        }
        Some(n_elements / be * block_bytes as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_sizing_matches_gpt_oss_expectations() {
        // one 32-elem block = 17 bytes; a 2880-wide row = 90 blocks = 1530 bytes
        assert_eq!(GgmlType::Mxfp4.byte_size(2880), Some(1530));
    }

    #[test]
    fn prism_ternary_sizing_matches_the_bonsai_27b_file() {
        assert_eq!(GgmlType::from_raw(143), GgmlType::Ptq1_0);
        assert_eq!(GgmlType::Ptq1_0.raw(), 143);
        assert_eq!(GgmlType::from_raw(142), GgmlType::Pq2_0);
        // ffn_down of Ternary-Bonsai-2-27B: 17408 x 5120 weights at 28 B / 128
        assert_eq!(GgmlType::Ptq1_0.byte_size(17408 * 5120), Some(19_496_960));
        assert_eq!(GgmlType::Pq2_0.byte_size(17408 * 5120), Some(23_674_880));
        // 1.75 and 2.125 bits a weight, scale included
        assert_eq!(GgmlType::Ptq1_0.byte_size(128), Some(28));
        assert_eq!(GgmlType::Pq2_0.byte_size(128), Some(34));
    }

    #[test]
    fn unknown_types_are_carried_not_dropped() {
        assert_eq!(GgmlType::from_raw(99), GgmlType::Unknown(99));
        assert_eq!(GgmlType::Unknown(99).byte_size(64), None);
    }

    #[test]
    fn partial_blocks_are_rejected() {
        assert_eq!(GgmlType::Q4K.byte_size(100), None); // 100 % 256 != 0
    }
}
