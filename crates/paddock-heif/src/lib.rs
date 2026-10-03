//! HEIF-family images: what the container says, and (with the decoder built)
//! the pixels inside it.
//!
//! Two formats share one container and could not be less alike
//! legally:
//!
//! - **AVIF** is AV1 in HEIF. Royalty-free, and every current browser decodes
//!   it, so the Studio shows one without asking us for anything.
//! - **HEIC** is HEVC/H.265 in HEIF - what an iPhone writes by default. No
//!   browser but Safari decodes it, because HEVC sits in a patent pool. We
//!   bundle a decoder the way pdfium is bundled rather than leave the format
//!   refused.
//!
//! The SPLIT this FILE DRAWS. Sniffing is always compiled: a `ftyp` box is
//! four bytes of brand, and knowing "this is HEIC" is what lets every refusal
//! name the format instead of calling the file binary junk. Decoding is a
//! separate question, and the answer differs by codec.
//!
//! AVIF DECODES; HEIC does not, and that is a licensing fact rather than a
//! missing install. AV1 has a BSD-2 decoder in pure Rust (rav1d), so it links
//! straight in - no library beside the executable, no pack, no build step.
//! HEVC has none: libde265, FFmpeg's hevcdec and openHEVC are all (L)GPL, and
//! embedding one in a closed binary obliges us to hand every user relinkable
//! object code. It was ruled that native code is linked in like
//! pdfium or it is not shipped, so HEIC is refused - honestly, by name, and
//! permanently until we write an intra-only HEVC decoder of our own.
//!
//! So `Error::NoDecoder` here does not mean "you are missing something you
//! could install". It means this program cannot read that codec, and a caller
//! should say so rather than suggest a fix that does not exist.
//!
//! What this CRATE does not do: touch the stored file. A rendition is derived
//! for looking at; metadata is read from the original bytes by
//! `paddock_filemeta`, which already parses HEIF. That separation is the EXIF
//! lesson written down - a re-encode is a viewing copy and never the record.

/// What the container's `ftyp` brand says is inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// HEVC/H.265 - iPhone's default. Patent-pooled; needs libde265.
    Heic,
    /// AV1 - royalty-free, and browsers decode it themselves.
    Avif,
}

impl Codec {
    /// The media type to hand a client. Distinct from what the browser SENT:
    /// Windows leaves `File.type` empty for `.HEIC` often enough that the
    /// extension was the only clue the Studio had.
    pub fn mime(self) -> &'static str {
        match self {
            Codec::Heic => "image/heic",
            Codec::Avif => "image/avif",
        }
    }
    /// How to say it to a person.
    pub fn label(self) -> &'static str {
        match self {
            Codec::Heic => "HEIC",
            Codec::Avif => "AVIF",
        }
    }
    /// Can a current browser show this without us decoding it? AVIF yes
    /// (Chrome 85+, Firefox 93+, Safari 16+); HEIC only on Safari, which is
    /// not a bet a local app gets to make.
    pub fn browser_decodes(self) -> bool {
        matches!(self, Codec::Avif)
    }
}

/// Decoded pixels: interleaved RGB8, `width * height * 3` bytes.
#[derive(Debug)]
pub struct Rendition {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The bytes are not a HEIF-family file at all - the caller sniffed wrong
    /// or the file is truncated.
    #[error("not a HEIF-family image")]
    NotHeif,
    /// Sniffed fine, but we cannot decode that codec - today only HEIC, and
    /// for the licensing reason in the module header. Carries the codec so the
    /// refusal can name the format.
    ///
    /// The message deliberately does not say "not installed" or "not in this
    /// build": there is nothing the reader can add to make it work, and
    /// implying otherwise sends them looking.
    #[error("{} images use HEVC, which this program cannot decode", .codec.label())]
    NoDecoder { codec: Codec },
    /// libheif said no. Its message rides along verbatim.
    #[error("{0}")]
    Decode(String),
}

/// Is this a HEIF-family image, and which codec? Reads the ISO-BMFF `ftyp`
/// box: a 4-byte big-endian size, the literal `ftyp`, then a 4-byte major
/// brand followed by compatible brands. We scan the compatible list too,
/// because a real iPhone file's major brand is `heic` but a Google Photos
/// export may lead with `mif1` and only mention `heic` further down.
///
/// Deliberately tolerant about size and deliberately strict about position:
/// `ftyp` must be the first box, which is what the spec requires and what
/// keeps this from matching a brand string buried in arbitrary bytes.
pub fn sniff(bytes: &[u8]) -> Option<Codec> {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return None;
    }
    // The declared box size bounds the brand list; clamp it to what we have so
    // a lying header cannot walk off the end.
    let declared = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let end = declared.clamp(12, bytes.len());
    // major brand at 8..12, then compatible brands in 4-byte slots from 16
    let mut found_heif_container = false;
    let mut i = 8;
    while i + 4 <= end {
        match &bytes[i..i + 4] {
            // AVIF first: an avif file also lists `mif1`, and the codec is
            // what we are after, not the container family.
            b"avif" | b"avis" => return Some(Codec::Avif),
            b"heic" | b"heix" | b"hevc" | b"hevx" | b"heim" | b"heis" => return Some(Codec::Heic),
            // container brands that say HEIF without naming a codec
            b"mif1" | b"msf1" | b"miaf" => found_heif_container = true,
            _ => {}
        }
        // skip the 4-byte minor VERSION that follows the major brand
        i += if i == 8 { 8 } else { 4 };
    }
    // A bare `mif1` with no codec brand is a HEIF container of unknown
    // contents. Call it HEIC: that is overwhelmingly what such files are, and
    // guessing wrong costs an honest "cannot decode" rather than corruption.
    found_heif_container.then_some(Codec::Heic)
}

/// Decode to RGB8. `Err(NoDecoder)` when the native library is not in this
/// build - a caller turns that into an honest refusal naming the format, not
/// a broken image tile.
pub fn decode(bytes: &[u8]) -> Result<Rendition, Error> {
    let codec = sniff(bytes).ok_or(Error::NotHeif)?;
    backend::decode(bytes, codec, None)
}

/// Bound the AV1 frame before its pixel planes are allocated. A check on the
/// returned RGB is too late for compressed images with enormous dimensions.
pub fn decode_limited(bytes: &[u8], max_pixels: u32) -> Result<Rendition, Error> {
    if max_pixels == 0 {
        return Err(Error::Decode("the image pixel budget is exhausted".into()));
    }
    let codec = sniff(bytes).ok_or(Error::NotHeif)?;
    backend::decode(bytes, codec, Some(max_pixels))
}

/// Can we decode this codec at all? AVIF yes, HEIC no - see the module header.
///
/// Per-codec rather than one flag, because the two answers are permanently
/// different and a single `have_decoder()` would have to lie about one of them.
pub fn can_decode(codec: Codec) -> bool {
    backend::available_for(codec)
}

/// What decodes AVIF here, for the startup banner and `/api/server`.
pub fn decoder_version() -> Option<String> {
    backend::describe()
}

mod backend;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal `ftyp` box: size, 'ftyp', major brand, minor version,
    /// then compatible brands. Enough for the sniffer, which is all it reads.
    fn ftyp(major: &[u8; 4], compat: &[&[u8; 4]]) -> Vec<u8> {
        let size = 16 + 4 * compat.len();
        let mut v = (size as u32).to_be_bytes().to_vec();
        v.extend_from_slice(b"ftyp");
        v.extend_from_slice(major);
        v.extend_from_slice(&0u32.to_be_bytes());
        for c in compat {
            v.extend_from_slice(*c);
        }
        v
    }

    #[test]
    fn an_iphone_photo_is_heic() {
        assert_eq!(
            sniff(&ftyp(b"heic", &[b"mif1", b"MiHE"])),
            Some(Codec::Heic)
        );
    }

    #[test]
    fn avif_wins_over_the_container_brand() {
        // a real avif lists mif1 as well; the CODEC is the answer we want
        assert_eq!(
            sniff(&ftyp(b"avif", &[b"mif1", b"miaf"])),
            Some(Codec::Avif)
        );
        // and when the container brand leads, the codec brand still decides
        assert_eq!(sniff(&ftyp(b"mif1", &[b"avif"])), Some(Codec::Avif));
    }

    #[test]
    fn a_codec_less_heif_container_is_assumed_heic() {
        // guessing wrong here costs an honest refusal, never a wrong decode
        assert_eq!(sniff(&ftyp(b"mif1", &[b"miaf"])), Some(Codec::Heic));
    }

    #[test]
    fn ftyp_must_lead_and_the_file_must_be_long_enough() {
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\n\0\0\0\r"), None);
        assert_eq!(sniff(b"too short"), None);
        // the brand string buried in ordinary bytes must not match: `ftyp` is
        // required at offset 4 and nowhere else
        let mut junk = vec![0u8; 64];
        junk[20..24].copy_from_slice(b"ftyp");
        junk[24..28].copy_from_slice(b"heic");
        assert_eq!(sniff(&junk), None);
    }

    #[test]
    fn a_lying_box_size_cannot_walk_off_the_end() {
        let mut v = ftyp(b"heic", &[]);
        v[0..4].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(sniff(&v), Some(Codec::Heic));
    }

    #[test]
    fn the_browser_split_is_the_licensing_split() {
        // AVIF is royalty-free and every browser has it; HEIC is the one we
        // have to decode ourselves, and the reason this crate exists
        assert!(Codec::Avif.browser_decodes());
        assert!(!Codec::Heic.browser_decodes());
    }

    /// A HEIC is refused by codec, before anything tries to read pixels - and
    /// the refusal names the format and the reason. It must never read as
    /// "something is missing here", because nothing is.
    #[test]
    fn a_heic_is_refused_by_name_and_says_why() {
        let e = decode(&ftyp(b"heic", &[])).expect_err("HEVC is not decodable here");
        assert!(
            matches!(e, Error::NoDecoder { codec: Codec::Heic }),
            "got {e:?}"
        );
        let msg = e.to_string();
        assert!(msg.contains("HEIC") && msg.contains("HEVC"), "{msg}");
        assert!(
            !msg.contains("install") && !msg.contains("build"),
            "must not imply a fix that does not exist: {msg}"
        );
        assert!(!can_decode(Codec::Heic));
        assert!(can_decode(Codec::Avif));
    }

    /// An AVIF gets as far as the decoder, so a header with no image in it
    /// comes back as a decode failure - a different answer from the one above,
    /// and the distinction is the whole point of splitting them.
    #[test]
    fn an_empty_avif_fails_at_the_decoder_not_at_the_gate() {
        let e = decode(&ftyp(b"avif", &[])).expect_err("a header alone is not an image");
        assert!(matches!(e, Error::Decode(_)), "got {e:?}");
        // Non-HEIF bytes are a different answer again.
        assert!(matches!(
            decode(b"\x89PNG\r\n\x1a\n\0\0\0\r"),
            Err(Error::NotHeif)
        ));
    }
}
