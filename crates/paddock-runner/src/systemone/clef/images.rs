//! Admission precedes blocking decode. The byte lease follows RGB ownership
//! into the decision queue; dropping an HTTP future cannot free it early.

use paddock_engine::clef_decision::{
    CLEF_IMAGE_REQUEST_BYTES, ClefDecider, ClefImage, ClefImageReservation,
};
use tokio::sync::Semaphore;

use super::{FACTOR, Fail, MAX_IMAGES, bad, decision_failure};

// Bound decoder scratch independently of retained RGB. Two active decoders
// keep large-photo bursts off Tokio's unbounded blocking queue. Ordinary c=4
// requests still queue for GPU work after their short decode phase.
static DECODERS: Semaphore = Semaphore::const_new(2);

pub(super) async fn read_images(
    decider: &ClefDecider,
    urls: Vec<String>,
    (lo, hi): (u64, u64),
) -> Result<(Vec<ClefImage>, Option<ClefImageReservation>), Fail> {
    if urls.len() > MAX_IMAGES {
        return Err(bad(format!(
            "images: {} given, at most {MAX_IMAGES}",
            urls.len()
        )));
    }
    if urls.is_empty() {
        return Ok((Vec::new(), None));
    }
    let mut reservation = decider
        .reserve_image_bytes(CLEF_IMAGE_REQUEST_BYTES)
        .map_err(decision_failure)?;
    // Waiting holds only bounded request bytes + a byte lease (at most four
    // full decode reservations), never already-decoded images. Cancellation
    // releases both before spawn; afterwards the blocking closure owns them.
    let decoder = DECODERS
        .acquire()
        .await
        .map_err(|_| decision_failure("image decoder closed".into()))?;
    tokio::task::spawn_blocking(move || {
        let _decoder = decoder;
        let mut used = 0usize;
        let mut images = Vec::with_capacity(urls.len());
        for (i, url) in urls.iter().enumerate() {
            let remaining = ((CLEF_IMAGE_REQUEST_BYTES - used) / 3) as u64;
            let (rgb, w, h) =
                crate::reference_image::decode_image_url_reference_limited(url, remaining)
                    .map_err(|e| format!("images[{i}]: {e}"))?;
            used += rgb.len();
            let (rh, rw) = paddock_models::clef::smart_resize(h as u64, w as u64, FACTOR, lo, hi)
                .map_err(|e| format!("images[{i}]: {e}"))?;
            images.push(ClefImage {
                rgb,
                width: w,
                height: h,
                resized: (rh as usize, rw as usize),
                row: 0,
            });
        }
        reservation.shrink_to(used)?;
        Ok::<_, String>((images, Some(reservation)))
    })
    .await
    .map_err(|e| bad(format!("images: the decode task failed: {e}")))?
    .map_err(bad)
}
