//! A JPEG decoder whose output is libjpeg-turbo's, bit for bit.
//!
//! Model references decode their pictures through libjpeg-turbo - torchvision's
//! `decode_image` and Pillow both link it - and JPEG decoding is not exact by
//! specification: the standard bounds the IDCT's error, it does not fix its
//! arithmetic, and chroma upsampling and the color transform are the decoder's
//! own. A different decoder hands a model pixels a few levels off on a tenth of
//! the samples, and a model's answers move with them (on Clef, as far as the
//! vendor's whole BF16 precision class moves them). This crate decodes the way
//! libjpeg-turbo's default decompression does (`jpeg_read_header` +
//! `jpeg_start_decompress`, RGB out - what both of those call):
//!
//!   - baseline and extended sequential (8-bit) and progressive Huffman scans,
//!     restart intervals;
//!   - the accurate integer IDCT (`JDCT_ISLOW`, jidctint.c) and its range-limit
//!     table, wraparound included;
//!   - the upsampler libjpeg elects per component (jdsample.c): fancy (the
//!     triangle filter with its alternating rounding biases) for 2h1v, 1h2v
//!     and 2h2v, plain replication otherwise; context rows duplicated at the
//!     image's top and bottom edges as its main controller does (jdmainct.c);
//!   - its color conversion (jdcolor.c): YCbCr through its 16-bit fixed-point
//!     tables, grayscale replicated, RGB copied, YCCK to CMYK; CMYK to RGB as
//!     torchvision converts it;
//!   - the color space guess from JFIF / Adobe markers and component ids
//!     (jdapimin.c).
//!
//! Not read (an error, so a caller can fall back to another decoder): lossless
//! and arithmetic-coded files, precisions other than 8 bits, a height deferred
//! to a DNL marker. One documented difference remains: libjpeg smooths blocks
//! of a progressive file whose low AC coefficients never finish refining
//! (`do_block_smoothing`); this decoder leaves them as coded.
//!
//! Written from libjpeg-turbo's sources as a reference (IJG / BSD-style
//! licences); the code is this crate's own.

mod huffman;
mod idct;
mod output;

use huffman::{BitReader, HuffTable};

/// A decoded picture: interleaved RGB8, `height` rows of `width` pixels.
#[derive(Debug)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    /// the first APP1 Exif payload (after `Exif\0\0`), for orientation
    pub exif: Option<Vec<u8>>,
}

#[derive(Debug)]
pub enum Error {
    /// not a JPEG, or a malformed one
    Format(String),
    /// a JPEG this decoder does not read (see the crate docs)
    Unsupported(String),
    /// past the caller's pixel budget
    TooLarge { width: usize, height: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Format(m) => write!(f, "malformed JPEG: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported JPEG: {m}"),
            Error::TooLarge { width, height } => {
                write!(f, "a {width} x {height} JPEG is past the pixel budget")
            }
        }
    }
}

impl std::error::Error for Error {}

fn bad<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Format(m.into()))
}

/// Whether `data` starts like a JPEG (SOI, then a marker).
pub fn sniff(data: &[u8]) -> bool {
    data.len() >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF
}

/// The zigzag position -> natural (row-major) position, with libjpeg's 16
/// trailing guards: a corrupt run past 63 lands on 63, never out of the block.
pub(crate) const NATURAL: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ColorSpace {
    Gray,
    YCbCr,
    Rgb,
    Cmyk,
    Ycck,
}

pub(crate) struct Component {
    pub id: u8,
    pub h: usize,
    pub v: usize,
    pub tq: usize,
    /// ceil(width * h / max_h), ceil(height * v / max_v)
    pub dw: usize,
    pub dh: usize,
    /// blocks holding real samples
    pub bw: usize,
    pub bh: usize,
    /// the coefficient plane's blocks a row (MCU padded)
    pub stride: usize,
    pub coefs: Vec<[i16; 64]>,
}

struct Frame {
    progressive: bool,
    width: usize,
    height: usize,
    comps: Vec<Component>,
    max_h: usize,
    max_v: usize,
    mcux: usize,
    mcuy: usize,
}

/// Decode `data` to RGB8 as libjpeg-turbo does, refusing a picture of more
/// than `max_pixels` pixels before allocating for it.
pub fn decode_rgb(data: &[u8], max_pixels: u64) -> Result<Image, Error> {
    if !sniff(data) {
        return bad("no SOI marker");
    }
    let mut pos = 2usize;
    let mut qt = [[0u16; 64]; 4];
    let mut dc_tables: [Option<HuffTable>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffTable>; 4] = [None, None, None, None];
    let mut restart = 0usize;
    let mut frame: Option<Frame> = None;
    let (mut jfif, mut adobe) = (false, None::<u8>);
    let mut exif = None;
    loop {
        let marker = next_marker(data, &mut pos)?;
        match marker {
            0xD8 => return bad("a second SOI"),
            0xD9 => break,
            0xD0..=0xD7 | 0x01 => continue,
            _ => {}
        }
        if pos + 2 > data.len() {
            return bad("truncated segment");
        }
        let len = usize::from(u16::from_be_bytes([data[pos], data[pos + 1]]));
        if len < 2 || pos + len > data.len() {
            return bad("segment length past the data");
        }
        let seg = &data[pos + 2..pos + len];
        pos += len;
        match marker {
            0xE0 => {
                // jdmarker examine_app0: a JFIF identifier in a long enough APP0
                if seg.len() >= 14 && seg.starts_with(b"JFIF\0") {
                    jfif = true;
                }
            }
            0xE1 => {
                if exif.is_none() && seg.starts_with(b"Exif\0\0") {
                    exif = Some(seg[6..].to_vec());
                }
            }
            0xEE => {
                // examine_app14: "Adobe", version, flags0, flags1, transform
                if seg.len() >= 12 && seg.starts_with(b"Adobe") {
                    adobe = Some(seg[11]);
                }
            }
            0xDB => read_dqt(seg, &mut qt)?,
            0xC4 => read_dht(seg, &mut dc_tables, &mut ac_tables)?,
            0xDD => {
                if seg.len() < 2 {
                    return bad("DRI");
                }
                restart = usize::from(u16::from_be_bytes([seg[0], seg[1]]));
            }
            0xC0..=0xC2 => {
                if frame.is_some() {
                    return bad("a second frame header");
                }
                frame = Some(read_sof(seg, marker == 0xC2, max_pixels)?);
            }
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => {
                return Err(Error::Unsupported(format!(
                    "frame type SOF{} (lossless, hierarchical or arithmetic coding)",
                    marker - 0xC0
                )));
            }
            0xDA => {
                let f = frame
                    .as_mut()
                    .ok_or(Error::Format("a scan before the frame".into()))?;
                pos = read_scan(data, pos, seg, f, &dc_tables, &ac_tables, restart)?;
            }
            0xDC => return Err(Error::Unsupported("a DNL marker".into())),
            _ => {} // COM, other APPn: nothing the pixels depend on
        }
    }
    let f = frame.ok_or(Error::Format("no frame".into()))?;
    let cs = color_space(&f.comps, jfif, adobe)?;
    let planes: Vec<(Vec<u8>, usize)> = f
        .comps
        .iter()
        .map(|c| idct::component_plane(c, &qt[c.tq]))
        .collect();
    let rgb = output::to_rgb(&f.comps, &planes, f.width, f.height, f.max_h, f.max_v, cs);
    Ok(Image {
        width: f.width,
        height: f.height,
        rgb,
        exif,
    })
}

/// jdapimin default_decompress_parms: the color space libjpeg guesses.
fn color_space(comps: &[Component], jfif: bool, adobe: Option<u8>) -> Result<ColorSpace, Error> {
    Ok(match comps.len() {
        1 => ColorSpace::Gray,
        3 => {
            if jfif {
                ColorSpace::YCbCr
            } else if let Some(t) = adobe {
                if t == 0 {
                    ColorSpace::Rgb
                } else {
                    ColorSpace::YCbCr
                }
            } else {
                let ids = (comps[0].id, comps[1].id, comps[2].id);
                if ids == (82, 71, 66) {
                    ColorSpace::Rgb
                } else {
                    ColorSpace::YCbCr
                }
            }
        }
        4 => match adobe {
            Some(0) => ColorSpace::Cmyk,
            Some(_) => ColorSpace::Ycck,
            None => ColorSpace::Cmyk,
        },
        n => {
            return Err(Error::Unsupported(format!(
                "{n} color components (1, 3 or 4 are read)"
            )));
        }
    })
}

/// The next marker code at or after `pos` (fill bytes skipped); `pos` ends
/// past it.
fn next_marker(data: &[u8], pos: &mut usize) -> Result<u8, Error> {
    // libjpeg's next_marker skips any garbage before the 0xFF, with a warning
    while *pos < data.len() && data[*pos] != 0xFF {
        *pos += 1;
    }
    while *pos < data.len() && data[*pos] == 0xFF {
        *pos += 1;
    }
    if *pos >= data.len() {
        return bad("no EOI");
    }
    let m = data[*pos];
    *pos += 1;
    if m == 0 {
        return next_marker(data, pos);
    }
    Ok(m)
}

fn read_dqt(mut seg: &[u8], qt: &mut [[u16; 64]; 4]) -> Result<(), Error> {
    while !seg.is_empty() {
        let (pq, tq) = (seg[0] >> 4, usize::from(seg[0] & 15));
        if tq > 3 || pq > 1 {
            return bad("DQT table id / precision");
        }
        let n = if pq == 0 { 64 } else { 128 };
        if seg.len() < 1 + n {
            return bad("DQT short");
        }
        for i in 0..64 {
            let v = if pq == 0 {
                u16::from(seg[1 + i])
            } else {
                u16::from_be_bytes([seg[1 + 2 * i], seg[2 + 2 * i]])
            };
            qt[tq][NATURAL[i]] = v;
        }
        seg = &seg[1 + n..];
    }
    Ok(())
}

fn read_dht(
    mut seg: &[u8],
    dc: &mut [Option<HuffTable>; 4],
    ac: &mut [Option<HuffTable>; 4],
) -> Result<(), Error> {
    while !seg.is_empty() {
        if seg.len() < 17 {
            return bad("DHT short");
        }
        let (tc, th) = (seg[0] >> 4, usize::from(seg[0] & 15));
        if tc > 1 || th > 3 {
            return bad("DHT table class / id");
        }
        let mut bits = [0u8; 17];
        bits[1..].copy_from_slice(&seg[1..17]);
        let count: usize = bits[1..].iter().map(|&b| usize::from(b)).sum();
        if count > 256 || seg.len() < 17 + count {
            return bad("DHT counts");
        }
        let vals = &seg[17..17 + count];
        let t = HuffTable::new(&bits, vals, tc == 0)?;
        if tc == 0 {
            dc[th] = Some(t);
        } else {
            ac[th] = Some(t);
        }
        seg = &seg[17 + count..];
    }
    Ok(())
}

fn read_sof(seg: &[u8], progressive: bool, max_pixels: u64) -> Result<Frame, Error> {
    if seg.len() < 6 {
        return bad("SOF short");
    }
    if seg[0] != 8 {
        return Err(Error::Unsupported(format!("{}-bit samples", seg[0])));
    }
    let height = usize::from(u16::from_be_bytes([seg[1], seg[2]]));
    let width = usize::from(u16::from_be_bytes([seg[3], seg[4]]));
    let n = usize::from(seg[5]);
    if height == 0 {
        return Err(Error::Unsupported(
            "a height deferred to a DNL marker".into(),
        ));
    }
    if width == 0 || n == 0 || seg.len() < 6 + 3 * n {
        return bad("SOF geometry");
    }
    if (width as u64) * (height as u64) > max_pixels {
        return Err(Error::TooLarge { width, height });
    }
    let mut comps = Vec::with_capacity(n);
    for i in 0..n {
        let c = &seg[6 + 3 * i..9 + 3 * i];
        let (h, v) = (usize::from(c[1] >> 4), usize::from(c[1] & 15));
        if !(1..=4).contains(&h) || !(1..=4).contains(&v) || c[2] > 3 {
            return bad("SOF sampling factors / quant table");
        }
        comps.push(Component {
            id: c[0],
            h,
            v,
            tq: usize::from(c[2]),
            dw: 0,
            dh: 0,
            bw: 0,
            bh: 0,
            stride: 0,
            coefs: Vec::new(),
        });
    }
    let max_h = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let max_v = comps.iter().map(|c| c.v).max().unwrap_or(1);
    if comps.iter().any(|c| max_h % c.h != 0 || max_v % c.v != 0) {
        // jdsample's JERR_FRACT_SAMPLE_NOTIMPL
        return Err(Error::Unsupported("fractional sampling factors".into()));
    }
    let mcux = width.div_ceil(8 * max_h);
    let mcuy = height.div_ceil(8 * max_v);
    for c in &mut comps {
        c.dw = (width * c.h).div_ceil(max_h);
        c.dh = (height * c.v).div_ceil(max_v);
        c.bw = (width * c.h).div_ceil(max_h * 8);
        c.bh = (height * c.v).div_ceil(max_v * 8);
        c.stride = mcux * c.h;
        c.coefs = vec![[0i16; 64]; c.stride * mcuy * c.v];
    }
    Ok(Frame {
        progressive,
        width,
        height,
        comps,
        max_h,
        max_v,
        mcux,
        mcuy,
    })
}

/// One scan: its header `seg`, its entropy-coded data from `pos`. Returns the
/// position of the marker that ends it.
fn read_scan(
    data: &[u8],
    pos: usize,
    seg: &[u8],
    f: &mut Frame,
    dc: &[Option<HuffTable>; 4],
    ac: &[Option<HuffTable>; 4],
    restart: usize,
) -> Result<usize, Error> {
    let ns = usize::from(*seg.first().ok_or(Error::Format("SOS short".into()))?);
    if ns == 0 || ns > 4 || seg.len() < 1 + 2 * ns + 3 {
        return bad("SOS header");
    }
    let mut members = Vec::with_capacity(ns);
    for i in 0..ns {
        let id = seg[1 + 2 * i];
        let t = seg[2 + 2 * i];
        let ci = f
            .comps
            .iter()
            .position(|c| c.id == id)
            .ok_or(Error::Format(format!("SOS names component {id}")))?;
        members.push((ci, usize::from(t >> 4), usize::from(t & 15)));
    }
    if ns > 1
        && members
            .iter()
            .map(|&(ci, _, _)| f.comps[ci].h * f.comps[ci].v)
            .sum::<usize>()
            > 10
    {
        // the spec's (and jdinput's) bound on an interleaved MCU
        return bad("sampling factors too large for an interleaved scan");
    }
    let (ss, se) = (usize::from(seg[1 + 2 * ns]), usize::from(seg[2 + 2 * ns]));
    let (ah, al) = (seg[3 + 2 * ns] >> 4, seg[3 + 2 * ns] & 15);
    let scan = huffman::Scan {
        progressive: f.progressive,
        ss,
        se,
        ah: u32::from(ah),
        al: u32::from(al),
    };
    if f.progressive {
        // jdinput / jdphuff start_pass_phuff_decoder's parameter checks
        let ok = if ss == 0 {
            se == 0
        } else {
            se >= ss && se <= 63 && ns == 1
        };
        if !ok || al > 13 || (ah != 0 && al != ah - 1) {
            return bad("progressive scan parameters");
        }
    }
    // a sequential scan decodes the whole block whatever Ss / Se / Ah / Al
    // say (libjpeg warns and does the same)
    for &(_, td, ta) in &members {
        let need_dc = !f.progressive || ss == 0;
        let need_ac = !f.progressive || ss > 0;
        if (need_dc && dc[td].is_none() && !(f.progressive && ah != 0))
            || (need_ac && ac[ta].is_none())
        {
            return bad("a scan's Huffman table is not defined");
        }
    }
    let mut br = BitReader::new(data, pos);
    let mut preds = [0i32; 4];
    let mut eobrun = 0u32;
    let mut todo = restart;
    let mut first = true;
    // per MCU: a restart boundary when its interval is spent (the bits
    // left dropped, the predictions and the EOB run reset), then the MCU
    let mut step = |br: &mut BitReader<'_>, preds: &mut [i32; 4], eobrun: &mut u32| {
        if restart > 0 {
            if todo == 0 && !first {
                br.restart();
                *preds = [0; 4];
                *eobrun = 0;
                todo = restart;
            }
            todo -= 1;
        }
        first = false;
        br.mcu_begin();
    };
    if ns == 1 {
        // a non-interleaved scan: one block an MCU over the component's own
        // blocks, not the MCU-padded plane
        let (ci, td, ta) = members[0];
        let c = &mut f.comps[ci];
        for by in 0..c.bh {
            for bx in 0..c.bw {
                step(&mut br, &mut preds, &mut eobrun);
                let blk = &mut c.coefs[by * c.stride + bx];
                huffman::decode_block(
                    &mut br,
                    blk,
                    &scan,
                    dc[td].as_ref(),
                    ac[ta].as_ref(),
                    &mut preds[0],
                    &mut eobrun,
                );
            }
        }
    } else {
        for my in 0..f.mcuy {
            for mx in 0..f.mcux {
                step(&mut br, &mut preds, &mut eobrun);
                for (k, &(ci, td, ta)) in members.iter().enumerate() {
                    let c = &mut f.comps[ci];
                    for y in 0..c.v {
                        for x in 0..c.h {
                            let idx = (my * c.v + y) * c.stride + mx * c.h + x;
                            huffman::decode_block(
                                &mut br,
                                &mut c.coefs[idx],
                                &scan,
                                dc[td].as_ref(),
                                ac[ta].as_ref(),
                                &mut preds[k],
                                &mut eobrun,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(br.scan_end())
}
