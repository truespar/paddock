//! AVIF decode, linked in. No native library, no pack, no build step, nothing
//! beside the executable.
//!
//! Why this is AVIF only, and will be until we write an HEVC decoder. HEIC and
//! AVIF share a container and could not be less alike underneath: AVIF is AV1,
//! which is royalty-free and has a BSD-2 decoder in pure Rust; HEIC is
//! HEVC/H.265, and every production HEVC decoder in existence - libde265,
//! FFmpeg's hevcdec, openHEVC - is (L)GPL. Embedding one in a closed binary
//! obliges us to hand every user relinkable object code, which is a bigger
//! concession than the feature is worth. It was ruled that native
//! code is linked in like pdfium or it is not shipped, so `Codec::Heic`
//! returns `NoDecoder` and says why. The in-house intra-only HEVC decoder is
//! what closes that, and it will replace this file's container half too.
//!
//! What does the work:
//!   - `avif-parse` finds the primary item and hands over its AV1 bitstream.
//!   - `rav1d` decodes it - memorysafety.org's Rust port of dav1d, BSD-2,
//!     built with `default-features = false` so no assembler is involved.
//!   - The YUV -> RGB conversion below is ours, because it has to read the
//!     sequence header to be right and neither crate does it.

use std::io::Cursor;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::picture::Dav1dPicture;

use crate::{Codec, Error, Rendition};

// rav1d exports dav1d's C API with `#[no_mangle]`, and offers no curated Rust
// API of its own - `pub mod src`/`pub mod include` expose the internals rather
// than a supported surface. So the entry points are declared here and resolved
// against the symbols rav1d links into this binary. Signatures transcribed from
// rav1d's own `src/lib.rs`; the types come from the crate so they cannot drift.
//
// `improper_ctypes` fires on `Dav1dContext`, which is rav1d's `RawArc<T>`: a
// `NonNull<T>` plus a `PhantomData<T>`. rustc warns because PhantomData has no
// C representation, but it also has no SIZE - the struct is one pointer, and
// this is the same declaration rav1d compiles on the other side of the call.
// Allowed here rather than silenced globally, so a genuinely unsafe type in a
// future signature still gets caught.
#[allow(improper_ctypes)]
unsafe extern "C" {
    fn dav1d_default_settings(s: NonNull<Dav1dSettings>);
    fn dav1d_open(
        c_out: Option<NonNull<Option<Dav1dContext>>>,
        s: Option<NonNull<Dav1dSettings>>,
    ) -> i32;
    fn dav1d_data_create(buf: Option<NonNull<Dav1dData>>, sz: usize) -> *mut u8;
    fn dav1d_data_unref(buf: Option<NonNull<Dav1dData>>);
    fn dav1d_send_data(c: Option<Dav1dContext>, r#in: Option<NonNull<Dav1dData>>) -> i32;
    fn dav1d_get_picture(c: Option<Dav1dContext>, out: Option<NonNull<Dav1dPicture>>) -> i32;
    fn dav1d_picture_unref(p: Option<NonNull<Dav1dPicture>>);
    fn dav1d_close(c_out: Option<NonNull<Option<Dav1dContext>>>);
}

/// dav1d pixel layouts, which rav1d types as a bare `c_uint`.
const I400: u32 = 0;
const I420: u32 = 1;
const I422: u32 = 2;
const I444: u32 = 3;

/// Matrix coefficients, from the AV1 spec's `matrix_coefficients` table.
const MC_IDENTITY: u32 = 0;
const MC_BT709: u32 = 1;
const MC_UNSPECIFIED: u32 = 2;
const MC_BT2020_NCL: u32 = 9;

/// Can this build decode that codec? AVIF always; HEIC never, and that is a
/// property of the format's licensing rather than of the install.
pub fn available_for(codec: Codec) -> bool {
    matches!(codec, Codec::Avif)
}

/// What decodes AVIF here, for the startup banner and `/api/server`. Constant
/// rather than probed: it is linked in, so there is nothing to discover.
pub fn describe() -> Option<String> {
    Some("rav1d (built in)".to_string())
}

pub fn decode(bytes: &[u8], codec: Codec, max_pixels: Option<u32>) -> Result<Rendition, Error> {
    if codec != Codec::Avif {
        return Err(Error::NoDecoder { codec });
    }
    let av = parse_container(bytes)?;
    // The ALPHA item, if there is one, is deliberately dropped: a Rendition is
    // RGB8 because both consumers want that - a JPEG rendition has no alpha to
    // put it in, and a vision tower takes three channels. Transparency
    // therefore renders as whatever the colour plane holds underneath, which is
    // the same thing every "flatten to JPEG" path does. Verified against a real
    // alpha AVIF: colour is unaffected.
    let mut r = decode_av1(&av.primary_item, max_pixels)?;
    // avif-parse does not read `irot`/`imir` - the fields do not exist on its
    // types - so a rotated photo would come out sideways with nothing to say
    // so. Recovered by our own scan of the same bytes; see `orientation`.
    apply_orientation(&mut r, orientation(bytes));
    Ok(r)
}

/// Read the container, and do not let it take the process down.
///
/// avif-parse PANICS on a truncated file rather than returning its error: its
/// `check_parser_state` is a `debug_assert_eq!` followed by the `Err` it should
/// have returned, so a debug build aborts where a release build recovers.
/// Found by the exhaustive-prefix test in tests/decode.rs, which is exactly
/// what that test is for.
///
/// Two reasons this is caught rather than shrugged at. A parser fed
/// attacker-controlled box sizes is the last place to accept "it only panics
/// in debug" - the debug build is what every test and every developer runs.
/// And the difference between debug and release behaviour on the same input is
/// its own hazard: a bug reproduced in development would vanish in the field.
///
/// Goes away with the in-house container parser; until then the boundary is
/// here, narrow and named.
fn parse_container(bytes: &[u8]) -> Result<avif_parse::AvifData, Error> {
    let parsed = std::panic::catch_unwind(|| avif_parse::read_avif(&mut Cursor::new(bytes)));
    match parsed {
        Ok(Ok(av)) => Ok(av),
        Ok(Err(e)) => Err(Error::Decode(format!(
            "this AVIF's container did not parse: {e}"
        ))),
        Err(_) => Err(Error::Decode(
            "this AVIF's container is malformed or truncated".into(),
        )),
    }
}

/// Decode one AV1 still image to RGB8.
fn decode_av1(obu: &[u8], max_pixels: Option<u32>) -> Result<Rendition, Error> {
    if obu.is_empty() {
        return Err(Error::Decode("this AVIF holds no image data".into()));
    }
    // SAFETY for the whole block: every pointer is either checked or produced
    // by the call above it, and the context and picture are released on every
    // exit path.
    unsafe {
        let mut settings: Dav1dSettings = MaybeUninit::zeroed().assume_init();
        dav1d_default_settings(NonNull::from(&mut settings));
        // A still image is one frame. dav1d's default frame delay pipelines
        // decoding to hide latency, and with a single frame in flight
        // `dav1d_get_picture` then returns EAGAIN forever - measured, not
        // guessed: the first version of this returned -11 and nothing else.
        settings.max_frame_delay = 1;
        settings.n_threads = 1;
        if let Some(max_pixels) = max_pixels {
            settings.frame_size_limit = max_pixels;
        }

        let mut ctx: Option<Dav1dContext> = None;
        if dav1d_open(
            Some(NonNull::from(&mut ctx)),
            Some(NonNull::from(&mut settings)),
        ) < 0
        {
            return Err(Error::Decode("could not start the AV1 decoder".into()));
        }
        let finish = |r: Result<Rendition, Error>| {
            let mut c = ctx;
            dav1d_close(Some(NonNull::from(&mut c)));
            r
        };

        let mut data: Dav1dData = MaybeUninit::zeroed().assume_init();
        let buf = dav1d_data_create(Some(NonNull::from(&mut data)), obu.len());
        if buf.is_null() {
            return finish(Err(Error::Decode(
                "could not allocate for the AV1 bitstream".into(),
            )));
        }
        std::ptr::copy_nonoverlapping(obu.as_ptr(), buf, obu.len());

        let sent = dav1d_send_data(ctx, Some(NonNull::from(&mut data)));
        // send_data empties this on success, but leaves ownership with us on
        // refusal (including the frame-size gate). Unref safely handles both.
        dav1d_data_unref(Some(NonNull::from(&mut data)));
        if sent < 0 {
            return finish(Err(Error::Decode(format!(
                "the AV1 bitstream was rejected ({sent})"
            ))));
        }

        let mut pic: Dav1dPicture = MaybeUninit::zeroed().assume_init();
        let got = dav1d_get_picture(ctx, Some(NonNull::from(&mut pic)));
        if got < 0 {
            return finish(Err(Error::Decode(format!(
                "no image came out of the AV1 decoder ({got}) - the file is probably truncated"
            ))));
        }
        let out = to_rgb(&pic);
        dav1d_picture_unref(Some(NonNull::from(&mut pic)));
        finish(out)
    }
}

/// YUV -> RGB8, reading the picture's own colour metadata.
///
/// # Safety
/// `pic` must be a live picture from `dav1d_get_picture`.
unsafe fn to_rgb(pic: &Dav1dPicture) -> Result<Rendition, Error> {
    let (w, h) = (pic.p.w, pic.p.h);
    if w <= 0 || h <= 0 {
        return Err(Error::Decode(format!(
            "the decoder reported a {w}x{h} image"
        )));
    }
    let (w, h) = (w as usize, h as usize);
    let total = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Decode("image dimensions overflow".into()))?;

    let layout = pic.p.layout;
    let bpc = pic.p.bpc as u32;
    // Chroma subsampling shifts, straight off the layout.
    let (sx, sy) = match layout {
        I420 => (1u32, 1u32),
        I422 => (1, 0),
        I444 | I400 => (0, 0),
        other => return Err(Error::Decode(format!("unsupported pixel layout {other}"))),
    };

    // Colour handling comes from the sequence header when there is one. A file
    // that omits it is treated as limited-range BT.601, which is what libavif
    // falls back to and therefore what the encoders in the wild assume.
    // SAFETY: caller's contract - seq_hdr is set by the decoder on success.
    let (mtrx, full_range) = unsafe {
        match pic.seq_hdr {
            Some(s) => {
                let s = s.as_ref();
                (s.mtrx, s.color_range != 0)
            }
            None => (MC_UNSPECIFIED, false),
        }
    };

    let plane = |i: usize| -> Option<(*const u8, usize)> {
        let p = pic.data[i]?.as_ptr() as *const u8;
        // stride[0] covers luma, stride[1] both chroma planes
        let s = pic.stride[if i == 0 { 0 } else { 1 }];
        if s <= 0 { None } else { Some((p, s as usize)) }
    };
    let (yp, ys) = plane(0).ok_or_else(|| Error::Decode("no luma plane".into()))?;
    let chroma = (layout != I400).then(|| (plane(1), plane(2)));

    // One reader for both bit depths. 10- and 12-bit AVIF are ordinary for HDR
    // photos, and their planes are u16 with the stride still counted in BYTES.
    let shift = bpc.saturating_sub(8);
    let read = |base: *const u8, stride: usize, x: usize, y: usize| -> i32 {
        // SAFETY: callers clamp x/y inside the plane; stride is the row pitch.
        unsafe {
            if bpc <= 8 {
                *base.add(y * stride + x) as i32
            } else {
                (base.add(y * stride + x * 2).cast::<u16>().read_unaligned() >> shift) as i32
            }
        }
    };

    // Limited ("studio") range packs luma into 16..235 and chroma into 16..240
    // at 8 bits; full range uses the whole 0..255. Getting this backwards is
    // the classic washed-out-or-crushed photo, so it is read, never assumed.
    let (y_off, y_scale, c_scale) = if full_range {
        (0.0f32, 1.0f32, 1.0f32)
    } else {
        (16.0, 255.0 / 219.0, 255.0 / 224.0)
    };
    // Luma coefficients per matrix. BT.601 is the fallback for "unspecified".
    let (kr, kb) = match mtrx {
        MC_BT709 => (0.2126f32, 0.0722f32),
        MC_BT2020_NCL => (0.2627, 0.0593),
        _ => (0.299, 0.114), // BT.601 / 170M / 470BG, and unspecified
    };
    let kg = 1.0 - kr - kb;

    let mut rgb = vec![0u8; total];
    for y in 0..h {
        for x in 0..w {
            let luma = (read(yp, ys, x, y) as f32 - y_off) * y_scale;
            let (r, g, b);
            match chroma {
                Some((Some((up, us)), Some((vp, vs)))) => {
                    let (cx, cy) = (x >> sx, y >> sy);
                    let cb = (read(up, us, cx, cy) as f32 - 128.0) * c_scale;
                    let cr = (read(vp, vs, cx, cy) as f32 - 128.0) * c_scale;
                    if mtrx == MC_IDENTITY {
                        // MC_IDENTITY is GBR stored in the YUV planes, not a
                        // colour transform. Converting it would be wrong twice.
                        g = luma;
                        b = cb + 128.0;
                        r = cr + 128.0;
                    } else {
                        r = luma + 2.0 * (1.0 - kr) * cr;
                        b = luma + 2.0 * (1.0 - kb) * cb;
                        g = luma - (2.0 * (kr * (1.0 - kr) * cr + kb * (1.0 - kb) * cb)) / kg;
                    }
                }
                // Monochrome, or a chroma plane the decoder did not give us:
                // grey is the honest rendering, not a colour guess.
                _ => {
                    r = luma;
                    g = luma;
                    b = luma;
                }
            }
            let i = (y * w + x) * 3;
            rgb[i] = r.clamp(0.0, 255.0) as u8;
            rgb[i + 1] = g.clamp(0.0, 255.0) as u8;
            rgb[i + 2] = b.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(Rendition {
        width: w as u32,
        height: h as u32,
        rgb,
    })
}

/// What the container says about rotation and mirroring.
///
/// `irot` is anticlockwise in 90° steps; `imir` mirrors about a vertical
/// (`axis == 1`) or horizontal (`axis == 0`) axis. Both live in the item
/// property container `ipco`, and both are ignored by avif-parse.
///
/// Deliberately a FLAT SCAN of the `ipco` box rather than a full item-property
/// walk: a still AVIF has one image, so every `irot`/`imir` in the file applies
/// to it, and resolving `ipma` associations would be parsing the container a
/// second time to learn nothing. The proper walk arrives with the HEIF parser
/// that HEVC needs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Orientation {
    /// anticlockwise quarter-turns, 0..=3
    rot: u8,
    mirror_vertical_axis: bool,
    mirror_horizontal_axis: bool,
}

fn orientation(bytes: &[u8]) -> Orientation {
    let mut o = Orientation::default();
    let Some(ipco) = find_box(bytes, b"ipco", 8) else {
        return o;
    };
    // ipco's children are plain boxes; only the two we care about are read.
    let mut i = 0usize;
    while i + 8 <= ipco.len() {
        let size = u32::from_be_bytes([ipco[i], ipco[i + 1], ipco[i + 2], ipco[i + 3]]) as usize;
        let kind = &ipco[i + 4..i + 8];
        // A zero or nonsense size would loop forever; stop rather than guess.
        if size < 8 || i + size > ipco.len() {
            break;
        }
        let body = &ipco[i + 8..i + size];
        match kind {
            b"irot" if !body.is_empty() => o.rot = body[0] & 0x03,
            b"imir" if !body.is_empty() => {
                if body[0] & 0x01 == 0 {
                    o.mirror_horizontal_axis = true;
                } else {
                    o.mirror_vertical_axis = true;
                }
            }
            _ => {}
        }
        i += size;
    }
    o
}

/// First box of `kind` anywhere in `bytes`, returned as its body. `depth`
/// bounds how far we recurse into container boxes.
fn find_box<'a>(bytes: &'a [u8], kind: &[u8; 4], depth: u32) -> Option<&'a [u8]> {
    if depth == 0 {
        return None;
    }
    let mut i = 0usize;
    while i + 8 <= bytes.len() {
        let size =
            u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize;
        let this = &bytes[i + 4..i + 8];
        // size 0 means "to end of file"; size 1 means a 64-bit size follows.
        // Neither appears around `ipco` in a still image, and treating them as
        // terminators is safer than mis-stepping through the file.
        let size = if size == 0 { bytes.len() - i } else { size };
        if size < 8 || i + size > bytes.len() {
            return None;
        }
        let body = &bytes[i + 8..i + size];
        if this == kind {
            return Some(body);
        }
        // Only descend through the containers on the path to ipco; walking
        // into media data would be slow and could match by accident.
        if matches!(this, b"meta" | b"iprp") {
            // `meta` is a FullBox: 4 bytes of version+flags before its children.
            let inner = if this == b"meta" && body.len() > 4 {
                &body[4..]
            } else {
                body
            };
            if let Some(found) = find_box(inner, kind, depth - 1) {
                return Some(found);
            }
        }
        i += size;
    }
    None
}

fn apply_orientation(r: &mut Rendition, o: Orientation) {
    if o == Orientation::default() {
        return;
    }
    // Mirror first, then rotate - the order the HEIF spec applies them in
    // (ISO/IEC 23008-12 §6.5.10: `imir` precedes `irot`).
    if o.mirror_vertical_axis {
        flip_horizontally(r);
    }
    if o.mirror_horizontal_axis {
        flip_vertically(r);
    }
    for _ in 0..o.rot {
        rotate_ccw(r);
    }
}

fn flip_horizontally(r: &mut Rendition) {
    let (w, h) = (r.width as usize, r.height as usize);
    for y in 0..h {
        let row = y * w * 3;
        for x in 0..w / 2 {
            let (a, b) = (row + x * 3, row + (w - 1 - x) * 3);
            for c in 0..3 {
                r.rgb.swap(a + c, b + c);
            }
        }
    }
}

fn flip_vertically(r: &mut Rendition) {
    let (w, h) = (r.width as usize, r.height as usize);
    let stride = w * 3;
    for y in 0..h / 2 {
        let (a, b) = (y * stride, (h - 1 - y) * stride);
        for i in 0..stride {
            r.rgb.swap(a + i, b + i);
        }
    }
}

/// One quarter-turn anticlockwise. Allocates: rotation changes the row pitch,
/// so it cannot be done in place without a permutation walk that would be
/// slower and far harder to read.
fn rotate_ccw(r: &mut Rendition) {
    let (w, h) = (r.width as usize, r.height as usize);
    let mut out = vec![0u8; r.rgb.len()];
    // dst is h wide and w tall; dst(x', y') = src(w-1-y', x')
    for y in 0..w {
        for x in 0..h {
            let src = (x * w + (w - 1 - y)) * 3;
            let dst = (y * h + x) * 3;
            out[dst..dst + 3].copy_from_slice(&r.rgb[src..src + 3]);
        }
    }
    r.rgb = out;
    r.width = h as u32;
    r.height = w as u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `meta > iprp > ipco > irot/imir` nest, which is where a real
    /// file keeps them.
    fn with_props(props: &[(&[u8; 4], u8)]) -> Vec<u8> {
        let mut ipco = Vec::new();
        for (kind, val) in props {
            ipco.extend_from_slice(&9u32.to_be_bytes());
            ipco.extend_from_slice(*kind);
            ipco.push(*val);
        }
        let boxed = |kind: &[u8; 4], body: &[u8]| {
            let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
            v.extend_from_slice(kind);
            v.extend_from_slice(body);
            v
        };
        let iprp = boxed(b"iprp", &boxed(b"ipco", &ipco));
        // meta is a FullBox: version+flags before the children
        let mut meta_body = vec![0u8; 4];
        meta_body.extend_from_slice(&iprp);
        let mut file = boxed(b"ftyp", b"avif\0\0\0\0");
        file.extend_from_slice(&boxed(b"meta", &meta_body));
        file
    }

    #[test]
    fn rotation_and_mirroring_are_read_out_of_the_container() {
        assert_eq!(orientation(&with_props(&[(b"irot", 1)])).rot, 1);
        assert_eq!(orientation(&with_props(&[(b"irot", 3)])).rot, 3);
        // only the low two bits are the angle
        assert_eq!(orientation(&with_props(&[(b"irot", 0xFC)])).rot, 0);
        let m = orientation(&with_props(&[(b"imir", 1)]));
        assert!(m.mirror_vertical_axis && !m.mirror_horizontal_axis);
        let m = orientation(&with_props(&[(b"imir", 0)]));
        assert!(m.mirror_horizontal_axis && !m.mirror_vertical_axis);
    }

    #[test]
    fn a_file_with_no_properties_is_upright() {
        assert_eq!(
            orientation(b"\0\0\0\x10ftypavif\0\0\0\0avif"),
            Orientation::default()
        );
        // and junk must not hang the scan or panic
        assert_eq!(orientation(&[0u8; 64]), Orientation::default());
        assert_eq!(orientation(&[0xFF; 64]), Orientation::default());
    }

    /// A 2x1 image: left pixel red, right pixel blue.
    fn two_px() -> Rendition {
        Rendition {
            width: 2,
            height: 1,
            rgb: vec![255, 0, 0, 0, 0, 255],
        }
    }

    #[test]
    fn a_quarter_turn_swaps_the_axes_and_moves_the_right_pixel_to_the_top() {
        let mut r = two_px();
        rotate_ccw(&mut r);
        assert_eq!((r.width, r.height), (1, 2));
        // anticlockwise: the right pixel ends up on TOP
        assert_eq!(&r.rgb[0..3], &[0, 0, 255]);
        assert_eq!(&r.rgb[3..6], &[255, 0, 0]);
    }

    #[test]
    fn four_quarter_turns_are_the_identity() {
        let mut r = two_px();
        for _ in 0..4 {
            rotate_ccw(&mut r);
        }
        assert_eq!((r.width, r.height), (2, 1));
        assert_eq!(r.rgb, two_px().rgb);
    }

    #[test]
    fn mirroring_about_the_vertical_axis_swaps_left_and_right() {
        let mut r = two_px();
        apply_orientation(
            &mut r,
            Orientation {
                mirror_vertical_axis: true,
                ..Default::default()
            },
        );
        assert_eq!(&r.rgb[0..3], &[0, 0, 255]);
        assert_eq!(&r.rgb[3..6], &[255, 0, 0]);
    }

    #[test]
    fn heic_is_refused_by_this_backend_on_purpose() {
        // Not "the pack is missing" - there is no pack. HEVC has no decoder we
        // are allowed to embed, and that is a fact about the format.
        assert!(!available_for(Codec::Heic));
        assert!(available_for(Codec::Avif));
        assert!(matches!(
            decode(b"whatever", Codec::Heic, None),
            Err(Error::NoDecoder { codec: Codec::Heic })
        ));
    }
}
