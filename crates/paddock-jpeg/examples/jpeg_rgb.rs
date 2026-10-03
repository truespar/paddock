//! Decode each JPEG named on the command line to `<file>.rgb` (raw RGB8) and
//! print its size - the parity check's half on this side (the reference's
//! half is libjpeg-turbo through torchvision).
fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        match paddock_jpeg::decode_rgb(&data, 1 << 30) {
            Ok(im) => {
                std::fs::write(format!("{path}.rgb"), &im.rgb).expect("write");
                println!("{path} {} {}", im.width, im.height);
            }
            Err(e) => println!("{path} ERR {e}"),
        }
    }
}
