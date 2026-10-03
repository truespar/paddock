//! Entropy decoding: libjpeg's canonical Huffman tables (jdhuff.c
//! `jpeg_make_d_derived_tbl`), its bit reader (byte stuffing, markers, the
//! zero bits it feeds past a premature marker, restart resynchronisation) and
//! the coefficient decoders of sequential (jdhuff.c `decode_mcu`) and
//! progressive scans (jdphuff.c: DC first / refine, AC first / refine).
//! Coefficients are stored as libjpeg stores them, `JCOEF` (i16) casts and
//! all.

use crate::{Error, NATURAL};

/// One decoding table: per code length, the largest code and the offset into
/// the values, as libjpeg derives them.
pub(crate) struct HuffTable {
    maxcode: [i32; 18],
    valoffset: [i32; 17],
    vals: Vec<u8>,
}

impl HuffTable {
    /// `bits[l]` codes of length l (1..=16) over `vals`, canonical order.
    pub(crate) fn new(bits: &[u8; 17], vals: &[u8], dc: bool) -> Result<Self, Error> {
        let mut huffsize = Vec::with_capacity(vals.len());
        for (l, &n) in bits.iter().enumerate().skip(1) {
            huffsize.extend(std::iter::repeat_n(l as u8, usize::from(n)));
        }
        // the codes, each length's continuing the last's shifted left
        let mut huffcode = Vec::with_capacity(huffsize.len());
        let (mut code, mut si, mut p) = (0u32, 1u8, 0usize);
        while p < huffsize.len() {
            while p < huffsize.len() && huffsize[p] == si {
                huffcode.push(code);
                code += 1;
                p += 1;
            }
            // a length's codes must fit its bits (jdhuff's JERR_BAD_HUFF_TABLE)
            if code >= 1u32 << si {
                return Err(Error::Format(
                    "a Huffman table over-subscribes its lengths".into(),
                ));
            }
            code <<= 1;
            si += 1;
        }
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 17];
        let mut p = 0usize;
        for l in 1..=16 {
            let n = usize::from(bits[l]);
            if n > 0 {
                valoffset[l] = p as i32 - huffcode[p] as i32;
                p += n;
                maxcode[l] = huffcode[p - 1] as i32;
            }
        }
        maxcode[17] = 0x000F_FFFF; // the decoder's sentinel
        if dc && vals.iter().any(|&v| v > 15) {
            return Err(Error::Format("a DC Huffman value above 15".into()));
        }
        Ok(Self {
            maxcode,
            valoffset,
            vals: vals.to_vec(),
        })
    }
}

/// The entropy-coded bytes of one scan, read as libjpeg's bit reader reads
/// them.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits: u32,
    /// a marker stopped the data: past it every bit reads 0
    marker: bool,
    /// bits were wanted past that marker (libjpeg's insufficient_data): the
    /// MCUs after the current one are left as they are
    pub(crate) insufficient: bool,
    /// whether the MCU being decoded began with the data already exhausted
    skip: bool,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            buf: 0,
            bits: 0,
            marker: false,
            insufficient: false,
            skip: false,
        }
    }

    fn fill(&mut self, need: u32) {
        while self.bits < need {
            let mut byte = 0u8;
            let mut real = false;
            if !self.marker && self.pos < self.data.len() {
                let c = self.data[self.pos];
                if c != 0xFF {
                    byte = c;
                    real = true;
                    self.pos += 1;
                } else {
                    // FF 00 is a stuffed FF; FF FF.. fill bytes run into
                    // whatever follows; anything else is a marker
                    let mut q = self.pos + 1;
                    while q < self.data.len() && self.data[q] == 0xFF {
                        q += 1;
                    }
                    if q < self.data.len() && self.data[q] == 0 {
                        byte = 0xFF;
                        real = true;
                        self.pos = q + 1;
                    } else {
                        self.marker = true;
                    }
                }
            }
            if !real {
                // past the data: zeros, and the data is now insufficient -
                // libjpeg sets that flag only when the bits are actually needed
                self.insufficient = true;
            }
            self.buf = (self.buf << 8) | u64::from(byte);
            self.bits += 8;
        }
    }

    fn get_bits(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        self.fill(n);
        self.bits -= n;
        ((self.buf >> self.bits) & ((1u64 << n) - 1)) as i32
    }

    fn get_bit(&mut self) -> i32 {
        self.get_bits(1)
    }

    /// jdhuff's slow-path HUFF_DECODE: a bit at a time against each length's
    /// largest code; a code past 16 bits decodes as 0 (libjpeg warns).
    fn decode(&mut self, t: &HuffTable) -> i32 {
        let mut code = self.get_bit();
        let mut l = 1usize;
        while code > t.maxcode[l] {
            code = (code << 1) | self.get_bit();
            l += 1;
            if l > 16 {
                return 0;
            }
        }
        let i = (code + t.valoffset[l]) as usize;
        t.vals.get(i).map_or(0, |&v| i32::from(v))
    }

    /// A restart boundary: the bits left are dropped, the RSTn marker is
    /// consumed (resynchronising past anything else, as jpeg_resync_to_restart
    /// does for a stream that stays in order), and the data is sufficient again.
    pub(crate) fn restart(&mut self) {
        self.bits = 0;
        self.buf = 0;
        // find the marker: where the reader stopped, or the next one
        let mut p = self.pos;
        loop {
            while p < self.data.len() && self.data[p] != 0xFF {
                p += 1;
            }
            let mut q = p;
            while q < self.data.len() && self.data[q] == 0xFF {
                q += 1;
            }
            if q >= self.data.len() {
                self.pos = self.data.len();
                break;
            }
            let m = self.data[q];
            if (0xD0..=0xD7).contains(&m) {
                self.pos = q + 1;
                break;
            }
            if m == 0 {
                p = q + 1;
                continue;
            }
            // some other marker: the scan's end - leave it for the caller
            self.pos = p;
            self.marker = true;
            self.insufficient = false;
            return;
        }
        self.marker = false;
        self.insufficient = false;
    }

    /// Mark the start of an MCU: one begun with the data exhausted is left
    /// untouched (libjpeg's `if (!insufficient_data)` around decode_mcu).
    pub(crate) fn mcu_begin(&mut self) {
        self.skip = self.insufficient;
    }

    /// Where the marker that ends this scan begins.
    pub(crate) fn scan_end(&self) -> usize {
        let mut p = self.pos;
        loop {
            while p < self.data.len() && self.data[p] != 0xFF {
                p += 1;
            }
            let mut q = p;
            while q < self.data.len() && self.data[q] == 0xFF {
                q += 1;
            }
            if q >= self.data.len() {
                return self.data.len();
            }
            let m = self.data[q];
            if m == 0 || (0xD0..=0xD7).contains(&m) {
                p = q + 1;
                continue;
            }
            return p;
        }
    }
}

/// HUFF_EXTEND: the s-bit magnitude category's value.
fn extend(r: i32, s: i32) -> i32 {
    if r < (1 << (s - 1)) {
        r + ((-1) << s) + 1
    } else {
        r
    }
}

/// A scan's band and successive-approximation bits.
pub(crate) struct Scan {
    pub progressive: bool,
    pub ss: usize,
    pub se: usize,
    pub ah: u32,
    pub al: u32,
}

/// Decode one block of the current MCU into `blk` (natural order).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_block(
    br: &mut BitReader<'_>,
    blk: &mut [i16; 64],
    scan: &Scan,
    dc: Option<&HuffTable>,
    ac: Option<&HuffTable>,
    pred: &mut i32,
    eobrun: &mut u32,
) {
    if br.skip {
        return;
    }
    if !scan.progressive {
        // jdhuff decode_mcu: the DC difference, then the AC run/size pairs
        let (Some(dc), Some(ac)) = (dc, ac) else {
            return;
        };
        let s = br.decode(dc);
        let mut v = 0;
        if s != 0 {
            let r = br.get_bits(s as u32);
            v = extend(r, s);
        }
        *pred += v;
        blk[0] = *pred as i16;
        let mut k = 1usize;
        while k < 64 {
            let rs = br.decode(ac);
            let (r, s) = ((rs >> 4) as usize, rs & 15);
            if s != 0 {
                k += r;
                let bits = br.get_bits(s as u32);
                blk[NATURAL[k]] = extend(bits, s) as i16;
            } else {
                if r != 15 {
                    break;
                }
                k += 15;
            }
            k += 1;
        }
        return;
    }
    let (p1, m1) = (1i32 << scan.al, (-1i32) << scan.al);
    if scan.ss == 0 {
        if scan.ah == 0 {
            // DC first: the difference, shifted up by Al
            let Some(dc) = dc else {
                return;
            };
            let s = br.decode(dc);
            let mut v = 0;
            if s != 0 {
                let r = br.get_bits(s as u32);
                v = extend(r, s);
            }
            *pred += v;
            blk[0] = (*pred << scan.al) as i16;
        } else if br.get_bit() != 0 {
            // DC refine: one more bit
            blk[0] |= p1 as i16;
        }
        return;
    }
    let Some(ac) = ac else {
        return;
    };
    if scan.ah == 0 {
        // AC first
        if *eobrun > 0 {
            *eobrun -= 1;
            return;
        }
        let mut k = scan.ss;
        while k <= scan.se {
            let rs = br.decode(ac);
            let (r, s) = (rs >> 4, rs & 15);
            if s != 0 {
                k += r as usize;
                let bits = br.get_bits(s as u32);
                blk[NATURAL[k]] = (extend(bits, s) << scan.al) as i16;
            } else if r == 15 {
                k += 15;
            } else {
                *eobrun = 1 << r;
                if r != 0 {
                    *eobrun += br.get_bits(r as u32) as u32;
                }
                *eobrun -= 1;
                break;
            }
            k += 1;
        }
        return;
    }
    // AC refine (jdphuff decode_mcu_AC_refine)
    let refine = |br: &mut BitReader<'_>, c: &mut i16| {
        if br.get_bit() != 0 && (i32::from(*c) & p1) == 0 {
            *c = if *c >= 0 {
                (i32::from(*c) + p1) as i16
            } else {
                (i32::from(*c) + m1) as i16
            };
        }
    };
    let mut k = scan.ss;
    if *eobrun == 0 {
        while k <= scan.se {
            let rs = br.decode(ac);
            let (mut r, mut s) = (rs >> 4, rs & 15);
            if s != 0 {
                s = if br.get_bit() != 0 { p1 } else { m1 };
            } else if r != 15 {
                *eobrun = 1 << r;
                if r != 0 {
                    *eobrun += br.get_bits(r as u32) as u32;
                }
                break;
            }
            loop {
                let c = &mut blk[NATURAL[k]];
                if *c != 0 {
                    refine(br, c);
                } else {
                    r -= 1;
                    if r < 0 {
                        break;
                    }
                }
                k += 1;
                if k > scan.se {
                    break;
                }
            }
            if s != 0 {
                blk[NATURAL[k]] = s as i16;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= scan.se {
            let c = &mut blk[NATURAL[k]];
            if *c != 0 {
                refine(br, c);
            }
            k += 1;
        }
        *eobrun -= 1;
    }
}
