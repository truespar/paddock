//! Does the decoder actually decode? The unit tests in lib.rs are all sniffing
//! and container scanning; everything below the `ftyp` box is rav1d's business
//! and only a real file exercises it.
//!
//! These assert UNCONDITIONALLY, unlike the version that preceded them. When
//! AVIF decode was a native library loaded at runtime, every case here had to
//! accept an honest "no decoder" as a pass - which meant a machine without the
//! pack showed green tests that had decoded nothing. rav1d is linked in, so
//! there is no such state and no branch to hide in.
//!
//! Fixtures are two 32x32 test patterns from libheif's repository; see
//! tests/data/README.md.

use paddock_heif::{Codec, Error};

const AVIF32: &[u8] = include_bytes!("data/avif32.heif");
const HEVC32: &[u8] = include_bytes!("data/hevc32.heif");

#[test]
fn avif_pixel_budget_is_enforced_by_the_decoder() {
    let full = paddock_heif::decode(AVIF32).unwrap();
    let bounded = paddock_heif::decode_limited(AVIF32, 1024).unwrap();
    assert_eq!(bounded.rgb, full.rgb);
    assert!(paddock_heif::decode_limited(AVIF32, 1023).is_err());
    assert!(paddock_heif::decode_limited(AVIF32, 0).is_err());
    // A rejected decode must not poison the next image or leak a context.
    assert_eq!(
        paddock_heif::decode_limited(AVIF32, 1024).unwrap().rgb,
        full.rgb
    );
}

#[test]
fn a_real_avif_decodes_to_pixels() {
    assert_eq!(paddock_heif::sniff(AVIF32), Some(Codec::Avif));
    let r = paddock_heif::decode(AVIF32).expect("AVIF must decode");

    assert_eq!((r.width, r.height), (32, 32));
    // Tightly packed RGB8. rav1d hands back planar YUV with a stride wider
    // than the image, so this is the conversion's contract, not a formality.
    assert_eq!(r.rgb.len(), 32 * 32 * 3);
    // A correctly-sized buffer of zeros would satisfy everything above. The
    // fixture is a colour test pattern.
    assert!(r.rgb.iter().any(|&b| b != 0), "decoded to all black");
    // ...and it is not flat grey either, which is what a broken chroma path
    // produces: luma right, both chroma planes dropped.
    assert!(
        r.rgb
            .as_chunks::<3>()
            .0
            .iter()
            .any(|p| p[0] != p[1] || p[1] != p[2]),
        "every pixel is grey - the chroma planes are not reaching the output"
    );
}

/// HEIC is HEVC, and no HEVC decoder can be embedded in a closed binary
/// without publishing relinkable object code. So this is a
/// permanent, deliberate refusal - not a missing install - and the message has
/// to say so without implying a fix.
#[test]
fn a_real_heic_is_refused_by_name() {
    assert_eq!(paddock_heif::sniff(HEVC32), Some(Codec::Heic));
    let e = paddock_heif::decode(HEVC32).expect_err("HEVC is not decodable here");
    assert!(
        matches!(e, Error::NoDecoder { codec: Codec::Heic }),
        "got {e:?}"
    );
    let msg = e.to_string();
    assert!(msg.contains("HEIC"), "{msg}");
    assert!(!msg.to_lowercase().contains("install"), "{msg}");
}

#[test]
fn the_two_codecs_answer_differently_about_what_is_possible() {
    assert!(paddock_heif::can_decode(Codec::Avif));
    assert!(!paddock_heif::can_decode(Codec::Heic));
    assert!(paddock_heif::decoder_version().is_some_and(|v| v.contains("rav1d")));
}

/// Truncation is the common real failure - a half-finished upload. It must be
/// an error about this file, not a panic and not a zero-byte image.
#[test]
fn a_truncated_avif_is_an_error_not_a_crash() {
    let half = &AVIF32[..AVIF32.len() / 2];
    // Still sniffs: the ftyp box survives, which is why identification reads
    // only the first twelve bytes.
    assert_eq!(paddock_heif::sniff(half), Some(Codec::Avif));
    assert!(matches!(paddock_heif::decode(half), Err(Error::Decode(_))));
}

/// Every prefix of a real file, one byte at a time. Container parsing walks
/// attacker-controlled box sizes, so "does not panic on a short read" is worth
/// asserting exhaustively rather than at one arbitrary cut.
#[test]
fn no_prefix_of_a_real_file_panics() {
    for n in 0..AVIF32.len() {
        let _ = paddock_heif::decode(&AVIF32[..n]);
    }
    for n in 0..HEVC32.len() {
        let _ = paddock_heif::decode(&HEVC32[..n]);
    }
}

#[test]
fn a_png_is_not_a_heif_at_all() {
    assert!(matches!(
        paddock_heif::decode(b"\x89PNG\r\n\x1a\n\0\0\0\r"),
        Err(Error::NotHeif)
    ));
}
