//! The accurate integer IDCT (libjpeg's `jpeg_idct_islow`, jidctint.c):
//! dequantize, a column pass into a 2^PASS1_BITS-scaled workspace, a row pass,
//! each output through the range-limit table - its wraparound included, so a
//! wild coefficient maps where libjpeg maps it. The butterflies run in 64-bit
//! (`JLONG` is a C long, 64-bit on the LP64 builds the reference ships), the
//! workspace is an int as libjpeg's is. The zero-AC shortcuts libjpeg takes
//! give the same bits as the full butterflies, so they are not repeated
//! here.

use crate::Component;

const CONST_BITS: i64 = 13;
const PASS1_BITS: i64 = 2;
const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

/// DESCALE: round and shift (an arithmetic shift, as libjpeg's RIGHT_SHIFT),
/// then libjpeg's `(int)` cast.
#[inline]
fn descale(x: i64, n: i64) -> i32 {
    ((x + (1 << (n - 1))) >> n) as i32
}

/// The post-IDCT range limit for a centered value `x`: libjpeg indexes its
/// table with `x & 1023`, which reads as a 10-bit signed value - -128..127
/// shift up to 0..255, 128..511 clamp to 255, -512..-129 to 0.
#[inline]
fn range_limit(x: i32) -> u8 {
    let i = x & 1023;
    match i {
        0..=127 => (i + 128) as u8,
        128..=511 => 255,
        512..=895 => 0,
        _ => (i - 896) as u8,
    }
}

/// The even and odd butterflies of one 8-point pass over `v` (dequantized
/// coefficients or the workspace); returns the eight sums before descaling,
/// in output order 0..7.
#[inline]
fn butterfly(v: [i64; 8]) -> [i64; 8] {
    // even part: the rotator is sqrt(2) * c(-6)
    let (z2, z3) = (v[2], v[6]);
    let z1 = (z2 + z3) * FIX_0_541196100;
    let tmp2 = z1 + z3 * -FIX_1_847759065;
    let tmp3 = z1 + z2 * FIX_0_765366865;
    let tmp0 = (v[0] + v[4]) << CONST_BITS;
    let tmp1 = (v[0] - v[4]) << CONST_BITS;
    let (tmp10, tmp13) = (tmp0 + tmp3, tmp0 - tmp3);
    let (tmp11, tmp12) = (tmp1 + tmp2, tmp1 - tmp2);
    // odd part: i0..i3 are y7, y5, y3, y1
    let (mut t0, mut t1, mut t2, mut t3) = (v[7], v[5], v[3], v[1]);
    let z1 = t0 + t3;
    let z2 = t1 + t2;
    let z3 = t0 + t2;
    let z4 = t1 + t3;
    let z5 = (z3 + z4) * FIX_1_175875602;
    t0 *= FIX_0_298631336;
    t1 *= FIX_2_053119869;
    t2 *= FIX_3_072711026;
    t3 *= FIX_1_501321110;
    let z1 = z1 * -FIX_0_899976223;
    let z2 = z2 * -FIX_2_562915447;
    let z3 = z3 * -FIX_1_961570560 + z5;
    let z4 = z4 * -FIX_0_390180644 + z5;
    t0 += z1 + z3;
    t1 += z2 + z4;
    t2 += z2 + z3;
    t3 += z1 + z4;
    [
        tmp10 + t3,
        tmp11 + t2,
        tmp12 + t1,
        tmp13 + t0,
        tmp13 - t0,
        tmp12 - t1,
        tmp11 - t2,
        tmp10 - t3,
    ]
}

/// One block: `coef` (natural order) against `q` (natural order) into 8 rows
/// of `out` at `stride`.
fn block(coef: &[i16; 64], q: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for col in 0..8 {
        // DEQUANTIZE: (JCOEF) * (ISLOW_MULT_TYPE, a short) in int
        let v: [i64; 8] = std::array::from_fn(|r| {
            i64::from(i32::from(coef[r * 8 + col]) * i32::from(q[r * 8 + col] as i16))
        });
        let o = butterfly(v);
        for r in 0..8 {
            ws[r * 8 + col] = descale(o[r], CONST_BITS - PASS1_BITS);
        }
    }
    for r in 0..8 {
        let v: [i64; 8] = std::array::from_fn(|c| i64::from(ws[r * 8 + c]));
        let o = butterfly(v);
        let row = &mut out[r * stride..r * stride + 8];
        for c in 0..8 {
            row[c] = range_limit(descale(o[c], CONST_BITS + PASS1_BITS + 3));
        }
    }
}

/// A component's sample plane: every block holding real samples through
/// the IDCT. Returns the plane and its row stride (`bw * 8`); its rows cover
/// `bh * 8`.
pub(crate) fn component_plane(c: &Component, q: &[u16; 64]) -> (Vec<u8>, usize) {
    let stride = c.bw * 8;
    let mut plane = vec![0u8; stride * c.bh * 8];
    for by in 0..c.bh {
        for bx in 0..c.bw {
            let off = by * 8 * stride + bx * 8;
            block(&c.coefs[by * c.stride + bx], q, &mut plane[off..], stride);
        }
    }
    (plane, stride)
}
