//! In-process macOS PDF rasterization. CoreGraphics is an OS framework, not
//! WebKit, a subprocess, or a downloadable sidecar. Each call owns a document
//! and bitmap context; no AppKit/main-thread work. Called by pdf's existing
//! spawn_blocking paths. RGB output has a white background and top-left rows.
use crate::pdf::{PageSel, PdfConfig, PdfError, PdfPage, RenderedPdf};
use std::ffi::c_void;
use std::ptr::{NonNull, null_mut};

type Ref = *mut c_void;
#[repr(C)]
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Size {
    w: f64,
    h: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    origin: Point,
    size: Size,
}
#[repr(C)]
struct Transform {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    tx: f64,
    ty: f64,
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGDataProviderCreateWithData(
        info: Ref,
        data: *const c_void,
        size: usize,
        release: Option<unsafe extern "C" fn(Ref, *const c_void, usize)>,
    ) -> Ref;
    fn CGDataProviderRelease(p: Ref);
    fn CGPDFDocumentCreateWithProvider(p: Ref) -> Ref;
    fn CGPDFDocumentRelease(p: Ref);
    fn CGPDFDocumentIsEncrypted(p: Ref) -> bool;
    fn CGPDFDocumentIsUnlocked(p: Ref) -> bool;
    fn CGPDFDocumentGetNumberOfPages(p: Ref) -> usize;
    fn CGPDFDocumentGetPage(p: Ref, page: usize) -> Ref;
    fn CGPDFPageGetBoxRect(p: Ref, box_type: i32) -> Rect;
    fn CGPDFPageGetRotationAngle(p: Ref) -> i32;
    fn CGPDFPageGetDrawingTransform(
        p: Ref,
        box_type: i32,
        rect: Rect,
        rotate: i32,
        preserve: bool,
    ) -> Transform;
    fn CGColorSpaceCreateDeviceRGB() -> Ref;
    fn CGColorSpaceRelease(p: Ref);
    fn CGBitmapContextCreate(
        data: Ref,
        w: usize,
        h: usize,
        bits: usize,
        stride: usize,
        space: Ref,
        info: u32,
    ) -> Ref;
    fn CGContextRelease(p: Ref);
    fn CGContextSetRGBFillColor(p: Ref, r: f64, g: f64, b: f64, a: f64);
    fn CGContextFillRect(p: Ref, rect: Rect);
    fn CGContextConcatCTM(p: Ref, transform: Transform);
    fn CGContextDrawPDFPage(p: Ref, page: Ref);
}

/// Every Create follows the corresponding Release, including error paths.
struct Owned(NonNull<c_void>, unsafe extern "C" fn(Ref));
impl Owned {
    fn new(p: Ref, release: unsafe extern "C" fn(Ref)) -> Option<Self> {
        NonNull::new(p).map(|p| Self(p, release))
    }
    fn ptr(&self) -> Ref {
        self.0.as_ptr()
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { (self.1)(self.ptr()) }
    }
}

struct Document<'a> {
    doc: Owned,
    _provider: Owned,
    _bytes: &'a [u8],
}
impl<'a> Document<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self, PdfError> {
        // Provider borrows bytes; document and provider both die before bytes.
        unsafe {
            let provider = Owned::new(
                CGDataProviderCreateWithData(null_mut(), bytes.as_ptr().cast(), bytes.len(), None),
                CGDataProviderRelease,
            )
            .ok_or_else(|| PdfError::Load("Could not create PDF data provider".into()))?;
            let doc = Owned::new(
                CGPDFDocumentCreateWithProvider(provider.ptr()),
                CGPDFDocumentRelease,
            )
            .ok_or_else(|| PdfError::Load("Invalid PDF document".into()))?;
            if CGPDFDocumentIsEncrypted(doc.ptr()) && !CGPDFDocumentIsUnlocked(doc.ptr()) {
                return Err(PdfError::Load(
                    "Password-protected PDF must be unlocked before attachment".into(),
                ));
            }
            Ok(Self {
                doc,
                _provider: provider,
                _bytes: bytes,
            })
        }
    }
    fn count(&self) -> usize {
        unsafe { CGPDFDocumentGetNumberOfPages(self.doc.ptr()) }
    }
    fn page(&self, index: usize, edge: Option<u32>, dpi: f64) -> Result<PdfPage, PdfError> {
        unsafe {
            let page = CGPDFDocumentGetPage(self.doc.ptr(), index + 1);
            if page.is_null() {
                return Err(PdfError::Render(index, "Page not found".into()));
            }
            let rect = CGPDFPageGetBoxRect(page, 1); // CropBox, respecting offsets.
            let (mut pw, mut ph) = (rect.size.w, rect.size.h);
            if CGPDFPageGetRotationAngle(page).rem_euclid(180) == 90 {
                std::mem::swap(&mut pw, &mut ph);
            }
            if !pw.is_finite() || !ph.is_finite() || pw <= 0. || ph <= 0. {
                return Err(PdfError::Render(index, "Invalid page dimensions".into()));
            }
            let dpi = edge.map(|e| f64::from(e) / pw.max(ph) * 72.).unwrap_or(dpi);
            if !dpi.is_finite() || dpi <= 0. {
                return Err(PdfError::Render(index, "Invalid resolution".into()));
            }
            let dpi = dpi.min(300.);
            let (wf, hf) = ((pw * dpi / 72.).max(1.), (ph * dpi / 72.).max(1.));
            // Explicit bounds before allocating, including forensic DPI callers.
            if wf > 8192. || hf > 8192. || wf * hf > 16_777_216. {
                return Err(PdfError::Render(
                    index,
                    "Page exceeds the 16-megapixel raster limit".into(),
                ));
            }
            let (w, h) = (wf as usize, hf as usize);
            let mut rgba = vec![255u8; w * h * 4];
            let space = Owned::new(CGColorSpaceCreateDeviceRGB(), CGColorSpaceRelease)
                .ok_or_else(|| PdfError::Render(index, "RGB color space unavailable".into()))?;
            // big-endian component order + noneSkipLast = R,G,B,X bytes.
            let context = Owned::new(
                CGBitmapContextCreate(
                    rgba.as_mut_ptr().cast(),
                    w,
                    h,
                    8,
                    w * 4,
                    space.ptr(),
                    (4 << 12) | 5,
                ),
                CGContextRelease,
            )
            .ok_or_else(|| PdfError::Render(index, "Bitmap allocation failed".into()))?;
            let dest = Rect {
                origin: Point { x: 0., y: 0. },
                size: Size {
                    w: w as f64,
                    h: h as f64,
                },
            };
            CGContextSetRGBFillColor(context.ptr(), 1., 1., 1., 1.);
            CGContextFillRect(context.ptr(), dest);
            // Bitmap rows already use image (top-left) ordering. A UIKit-style
            // context flip would invert the raw RGB result.
            CGContextConcatCTM(
                context.ptr(),
                CGPDFPageGetDrawingTransform(page, 1, dest, 0, true),
            );
            CGContextDrawPDFPage(context.ptr(), page);
            drop(context); // flush and release before reading borrowed storage
            let mut rgb = Vec::with_capacity(w * h * 3);
            for pixel in rgba.chunks_exact(4) {
                rgb.extend_from_slice(&pixel[..3]);
            }
            Ok(PdfPage { rgb, w, h })
        }
    }
}

pub(crate) fn render(bytes: &[u8], cfg: &PdfConfig, sel: PageSel) -> Result<RenderedPdf, PdfError> {
    let doc = Document::new(bytes)?;
    let total = doc.count();
    if total == 0 {
        return Err(PdfError::Empty);
    }
    let (start, want_end) = sel.resolve(total, "the PDF").map_err(PdfError::Pages)?;
    let end = want_end.min(start.saturating_add(cfg.max_pages.max(1) - 1));
    let mut pages = Vec::new();
    let mut allocated = 0usize;
    for index in start - 1..end {
        let page = doc.page(index, Some(cfg.long_edge), 300.)?;
        allocated = allocated.saturating_add(page.rgb.len());
        if allocated > 256 * 1024 * 1024 {
            return Err(PdfError::Render(
                index,
                "Selected pages exceed the 256 MiB raster budget; select fewer pages".into(),
            ));
        }
        pages.push(page);
    }
    Ok(RenderedPdf {
        truncated: pages.len() < total,
        pages,
        total_pages: total,
        first_page: start,
        ceiling_clipped: end < want_end,
    })
}

pub(crate) fn render_page_rgb(bytes: &[u8], page: u32, dpi: f32) -> Option<(Vec<u8>, u32, u32)> {
    let p = Document::new(bytes)
        .ok()?
        .page(page as usize, None, dpi.into())
        .ok()?;
    Some((p.rgb, p.w as u32, p.h as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let drawing = "1 0 0 rg 0 50 100 50 re f\n0 0 1 rg 0 0 100 50 re f\n";
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>".to_owned(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>".to_owned(),
            format!("<< /Length {} >>\nstream\n{}endstream", drawing.len(), drawing),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /CropBox [0 0 100 50] /Rotate 90 /Resources << >> /Contents 4 0 R >>".to_owned(),
        ];
        let mut pdf = "%PDF-1.4\n".to_owned();
        let mut offsets = vec![0];
        for (i, o) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf += &format!("{} 0 obj\n{o}\nendobj\n", i + 1);
        }
        let xref = pdf.len();
        pdf += &format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len());
        for n in offsets.iter().skip(1) {
            pdf += &format!("{n:010} 00000 n \n");
        }
        pdf += &format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            offsets.len()
        );
        pdf.into_bytes()
    }
    #[test]
    fn native_rgb_orientation_crop_rotation_and_page_ceiling() {
        let bytes = fixture();
        let config = PdfConfig {
            max_pages: 1,
            long_edge: 100,
            max_dpi: 300.0,
        };
        let p = render(&bytes, &config, PageSel::All).unwrap();
        assert_eq!((p.total_pages, p.first_page, p.pages.len()), (2, 1, 1));
        assert!(p.ceiling_clipped && p.truncated);
        let p = &p.pages[0];
        assert_eq!((p.w, p.h, p.rgb.len()), (100, 100, 30000));
        assert_eq!(
            &p.rgb[(10 * 100 + 50) * 3..(10 * 100 + 50) * 3 + 3],
            &[255, 0, 0],
            "top rows must be the red top of the PDF"
        );
        assert_eq!(
            &p.rgb[(90 * 100 + 50) * 3..(90 * 100 + 50) * 3 + 3],
            &[0, 0, 255]
        );
        let p = render(&bytes, &config, PageSel::Range(2, 2)).unwrap();
        assert_eq!((p.first_page, p.pages[0].w, p.pages[0].h), (2, 50, 100));
        assert!(!p.ceiling_clipped);
        assert!(render(&bytes, &config, PageSel::Range(3, 3)).is_err());
        assert!(render(b"not a PDF", &config, PageSel::All).is_err());
        assert!(render_page_rgb(&bytes, 0, f32::NAN).is_none());
    }
    #[test]
    fn independent_documents_render_on_background_threads() {
        let bytes = fixture();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| render_page_rgb(&bytes, 0, 72.).unwrap().0))
                .collect();
            let outputs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            assert!(outputs.iter().all(|v| v == &outputs[0]));
        });
    }
}
