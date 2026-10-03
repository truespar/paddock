//! libjpeg-turbo 3.2.0's own decodes (through torchvision's decode_image,
//! mode RGB), pinned as FNV-1a hashes of the RGB bytes, for a handful of the
//! parity corpus's files (the whole corpus is checked offline against the
//! same decodes): progressive 4:2:0 at an odd size, baseline 4:2:2, the 4:4:0 and
//! 4:1:1 upsamplers, YCCK, Adobe RGB, progressive grayscale, a progressive
//! scan with restart intervals, Adobe CMYK.

fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
        (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

const CASES: &[(&str, usize, usize, u64)] = &[
    ("enc61x45_411.jpg", 61, 45, 0x7c89811bc1f0a177),
    ("enc61x45_ycck.jpg", 61, 45, 0xcb863f5a2d2884ca),
    ("enc9x7_440.jpg", 9, 7, 0x013571ffada73b55),
    ("enc9x7_adobe_rgb.jpg", 9, 7, 0xc3fa280a0a6ee764),
    ("pil17x13_s2_prog.jpg", 17, 13, 0xe46b139abb56edf9),
    ("pil33x31_s1_base.jpg", 33, 31, 0x3b0ae2287fab0306),
    ("pil_cmyk.jpg", 250, 190, 0xbbc280dc56eeeec3),
    ("pil_gray_prog.jpg", 250, 190, 0x29f04415adb1f121),
    ("pil_rst_blocks_prog.jpg", 250, 190, 0x905f22944a96d7e1),
];

#[test]
fn decodes_as_libjpeg_turbo() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    for &(name, w, h, want) in CASES {
        let data = std::fs::read(dir.join(name)).expect("test file");
        let im = paddock_jpeg::decode_rgb(&data, 1 << 24).expect("decodes");
        assert_eq!((im.width, im.height), (w, h), "{name}: size");
        assert_eq!(
            fnv1a(&im.rgb),
            want,
            "{name}: pixels differ from libjpeg-turbo's"
        );
    }
}

#[test]
fn refuses_what_it_does_not_read() {
    assert!(!paddock_jpeg::sniff(b"\x89PNG\r\n"));
    assert!(paddock_jpeg::decode_rgb(b"\xff\xd8\xff", 1 << 24).is_err());
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let data = std::fs::read(dir.join("pil_cmyk.jpg")).expect("test file");
    // past the pixel budget: refused before allocating
    assert!(matches!(
        paddock_jpeg::decode_rgb(&data, 1000),
        Err(paddock_jpeg::Error::TooLarge {
            width: 250,
            height: 190
        })
    ));
}
