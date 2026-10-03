//! Pictures for a lane held to a Python reference's decoder (Clef): JPEG as
//! libjpeg-turbo decodes it, everything else as the endpoint always has,
//! upright by EXIF, never resized - and the `data:` URI reading every
//! surface shares.

use image::ImageDecoder as _;

/// The bytes of a base64 `data:` URI.
pub(crate) fn data_url_bytes(url: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(
            "only data: image URIs are supported (the server does not fetch remote URLs); \
             inline the image as base64"
                .into(),
        );
    };
    let (meta, b64) = rest.split_once(',').ok_or("malformed data: URI")?;
    if !meta.ends_with(";base64") {
        return Err("data: image URI must be base64-encoded".into());
    }
    base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("image base64: {e}"))
}

/// The most pixels a reference-decoded picture may have (64 MP, any
/// format): a JPEG's decode holds its coefficients and RGB at once (~10 bytes
/// a pixel), and the engine stages the decoded bytes in a resident plane
/// (Clef: ~89 MP of room) rather than allocating per picture.
pub(crate) const REFERENCE_MAX_PIXELS: u64 = 64_000_000;

/// Decode a `data:` image URI for a lane held to a Python reference's
/// decoder (Clef): a JPEG is decoded as libjpeg-turbo decodes it
/// (`paddock_jpeg`, bit for bit) - torchvision and Pillow both link
/// libjpeg-turbo, and another decoder's rounding (the `image` crate's is off
/// a few levels on a tenth of the samples) moves such a model's answers as
/// far as its vendor's whole precision class does. Every other format, and a
/// JPEG `paddock_jpeg` does not read (lossless, arithmetic, 12-bit), decodes
/// with the shared image codecs - PNG, GIF and WebP already decode to the
/// reference's bytes. Upright by EXIF orientation either way; never resized.
#[cfg(test)]
pub(crate) fn decode_image_url_reference(url: &str) -> Result<(Vec<u8>, usize, usize), String> {
    decode_image_url_reference_limited(url, REFERENCE_MAX_PIXELS)
}

/// The caller may impose a smaller aggregate remaining budget. Check all
/// formats before allocating their pixel planes, not just JPEG.
pub(crate) fn decode_image_url_reference_limited(
    url: &str,
    max_pixels: u64,
) -> Result<(Vec<u8>, usize, usize), String> {
    let max_pixels = max_pixels.min(REFERENCE_MAX_PIXELS);
    if max_pixels == 0 {
        return Err("the decoded image budget is exhausted".into());
    }
    let bytes = data_url_bytes(url)?;
    if paddock_jpeg::sniff(&bytes) {
        match paddock_jpeg::decode_rgb(&bytes, max_pixels) {
            Ok(im) => {
                let rgb = image::RgbImage::from_raw(im.width as u32, im.height as u32, im.rgb)
                    .ok_or_else(|| "image decode: decoded plane is the wrong size".to_string())?;
                let mut img = image::DynamicImage::ImageRgb8(rgb);
                if let Some(o) = im
                    .exif
                    .as_deref()
                    .and_then(image::metadata::Orientation::from_exif_chunk)
                {
                    img.apply_orientation(o);
                }
                let rgb = img.into_rgb8();
                let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                return Ok((rgb.into_raw(), w, h));
            }
            Err(paddock_jpeg::Error::Unsupported(_)) => {}
            Err(paddock_jpeg::Error::TooLarge { width, height }) => {
                return Err(too_large(width, height, max_pixels));
            }
            Err(e) => return Err(format!("image decode: {e}")),
        }
    }
    let rgb = if paddock_heif::sniff(&bytes).is_some() {
        let im = paddock_heif::decode_limited(&bytes, max_pixels as u32)
            .map_err(|e| format!("image decode: {e}"))?;
        // The HEIF decoder already applied the container's orientation.
        image::RgbImage::from_raw(im.width, im.height, im.rgb)
            .ok_or("image decode: decoded plane is the wrong size")?
    } else {
        let mut reader = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(|e| format!("image decode: {e}"))?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(max_pixels as u32);
        limits.max_image_height = Some(max_pixels as u32);
        // Includes 16-bit RGBA source planes and orientation/RGB copies.
        limits.max_alloc = Some(max_pixels * 16);
        reader.limits(limits);
        let mut decoder = reader
            .into_decoder()
            .map_err(|e| format!("image decode: {e}"))?;
        let (w, h) = decoder.dimensions();
        if u64::from(w) * u64::from(h) > max_pixels {
            return Err(too_large(w as usize, h as usize, max_pixels));
        }
        let orientation = decoder
            .orientation()
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        let mut im =
            image::DynamicImage::from_decoder(decoder).map_err(|e| format!("image decode: {e}"))?;
        im.apply_orientation(orientation);
        im.into_rgb8()
    };
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok((rgb.into_raw(), w, h))
}

fn too_large(w: usize, h: usize, max_pixels: u64) -> String {
    format!(
        "a {w} x {h} picture exceeds the {max_pixels} remaining pixel budget - downscale it before sending"
    )
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn non_jpeg_pixel_budget_is_checked_and_valid_pixels_unchanged() {
        for format in [
            image::ImageFormat::Png,
            image::ImageFormat::Bmp,
            image::ImageFormat::Tiff,
        ] {
            let im = image::RgbImage::from_pixel(32, 48, image::Rgb([7, 83, 155]));
            let mut bytes = std::io::Cursor::new(Vec::new());
            im.write_to(&mut bytes, format).unwrap();
            let uri = format!(
                "data:image/unknown;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(bytes.into_inner())
            );
            let (rgb, w, h) = decode_image_url_reference_limited(&uri, 32 * 48).unwrap();
            assert_eq!((w, h), (32, 48));
            assert_eq!(rgb, im.into_raw());
            assert!(decode_image_url_reference_limited(&uri, 32 * 48 - 1).is_err());
            assert!(decode_image_url_reference_limited(&uri, 0).is_err());
        }
    }
}
