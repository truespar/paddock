//! GGUF NVFP4 (ggml type 40) -> the house NVFP4 row layout.
//!
//! ggml's block (ggml-common.h `block_nvfp4`) covers 64 elements: four UE4M3
//! scale bytes, one per 16-element sub-block, then 32 bytes of E2M1 codes.
//! Inside sub-block `s`, byte `j` (0..8) holds element `j` in its low nibble
//! and element `j + 8` in its high nibble. ggml decodes a code as
//! `kvalues_mxfp4[q] * ue4m3(d)` with doubled table values and a halved
//! scale, which is exactly `e2m1(q) * e4m3(d)` - the modelopt value - so the
//! scale bytes carry over unchanged and only the nibbles move: the row layout
//! (`[out, in/2]` codes, `[out, in/16]` scales) packs adjacent elements
//! `2i` / `2i + 1` into byte `i`. The per-tensor F32 factor ships as a
//! separate `<name>.scale` tensor and is not part of the block.
//!
//! Two scale bytes have no meaning in the format and are handled here rather
//! than in a kernel: `0x7F` (E4M3's NaN) decodes as 0 in ggml, so it becomes
//! the zero byte; a set sign bit is refused, since an unsigned scale with a
//! sign is a broken file, not a value.

/// Elements per ggml NVFP4 block, and its bytes.
pub const BLOCK_ELEMS: usize = 64;
pub const BLOCK_BYTES: usize = 36;

/// What a repack found besides the bytes it moved.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RepackStats {
    /// `0x7F` scale bytes rewritten to zero.
    pub nan_scales: usize,
}

/// Repack `rows` rows of `in_dim` elements from ggml NVFP4 blocks into
/// `(codes [rows, in_dim/2], scales [rows, in_dim/16])`.
pub fn repack_rows(
    src: &[u8],
    rows: usize,
    in_dim: usize,
) -> Result<(Vec<u8>, Vec<u8>, RepackStats), String> {
    if !in_dim.is_multiple_of(BLOCK_ELEMS) {
        return Err(format!(
            "nvfp4 row of {in_dim} elements is not a whole number of 64-element blocks"
        ));
    }
    let blocks = rows * (in_dim / BLOCK_ELEMS);
    if src.len() != blocks * BLOCK_BYTES {
        return Err(format!(
            "nvfp4 tensor holds {} bytes, [{rows}, {in_dim}] needs {}",
            src.len(),
            blocks * BLOCK_BYTES
        ));
    }
    let mut codes = vec![0u8; rows * in_dim / 2];
    let mut scales = vec![0u8; rows * in_dim / 16];
    let mut stats = RepackStats::default();
    for (b, blk) in src.as_chunks::<BLOCK_BYTES>().0.iter().enumerate() {
        let (d, qs) = blk.split_at(4);
        for (s, &byte) in d.iter().enumerate() {
            if byte & 0x80 != 0 {
                return Err(format!(
                    "nvfp4 block {b}: scale byte {byte:#04x} has its sign bit set"
                ));
            }
            scales[b * 4 + s] = if byte == 0x7f {
                stats.nan_scales += 1;
                0
            } else {
                byte
            };
            let sub = &qs[s * 8..s * 8 + 8];
            let nib = |m: usize| {
                if m < 8 {
                    sub[m] & 0x0f
                } else {
                    sub[m - 8] >> 4
                }
            };
            let out = &mut codes[b * 32 + s * 8..b * 32 + s * 8 + 8];
            for (i, o) in out.iter_mut().enumerate() {
                *o = nib(2 * i) | (nib(2 * i + 1) << 4);
            }
        }
    }
    Ok((codes, scales, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelopt::{e2m1_to_f32, e4m3_to_f32};

    /// The ggml decode, written from the block definition: element `j` of
    /// sub-block `s` sits in the low nibble of byte `j`, element `j + 8` in
    /// the high nibble.
    fn ggml_value(blk: &[u8], e: usize) -> f32 {
        let (s, m) = (e / 16, e % 16);
        let byte = blk[4 + s * 8 + (m % 8)];
        let code = if m < 8 { byte & 0x0f } else { byte >> 4 };
        let d = if blk[s] == 0x7f {
            0.0
        } else {
            e4m3_to_f32(blk[s])
        };
        e2m1_to_f32(code) * d
    }

    fn row_value(codes: &[u8], scales: &[u8], in_dim: usize, r: usize, e: usize) -> f32 {
        let byte = codes[r * in_dim / 2 + e / 2];
        let code = if e.is_multiple_of(2) {
            byte & 0x0f
        } else {
            byte >> 4
        };
        e2m1_to_f32(code) * e4m3_to_f32(scales[r * in_dim / 16 + e / 16])
    }

    #[test]
    fn every_element_keeps_its_value() {
        let (rows, in_dim) = (3usize, 128usize);
        let mut src = Vec::new();
        let mut seed = 0x2545_f491u32;
        for _ in 0..rows * in_dim / BLOCK_ELEMS {
            let mut blk = [0u8; BLOCK_BYTES];
            for (i, b) in blk.iter_mut().enumerate() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // scales stay unsigned; the codes take every nibble
                *b = if i < 4 {
                    (seed >> 24) as u8 & 0x7e
                } else {
                    (seed >> 24) as u8
                };
            }
            src.extend_from_slice(&blk);
        }
        // one NaN scale, which must read as zero on both sides
        src[1] = 0x7f;
        let (codes, scales, stats) = repack_rows(&src, rows, in_dim).expect("repack");
        assert_eq!(stats.nan_scales, 1);
        for r in 0..rows {
            for e in 0..in_dim {
                let blk = &src[(r * in_dim + e) / BLOCK_ELEMS * BLOCK_BYTES..][..BLOCK_BYTES];
                let want = ggml_value(blk, e % BLOCK_ELEMS);
                let got = row_value(&codes, &scales, in_dim, r, e);
                assert_eq!(got.to_bits(), want.to_bits(), "row {r} element {e}");
            }
        }
    }

    #[test]
    fn a_signed_scale_or_a_short_tensor_is_refused() {
        let mut blk = [0u8; BLOCK_BYTES];
        blk[2] = 0x81;
        assert!(repack_rows(&blk, 1, 64).is_err());
        assert!(repack_rows(&[0u8; 35], 1, 64).is_err());
        assert!(repack_rows(&[0u8; 36], 1, 32).is_err());
    }
}
