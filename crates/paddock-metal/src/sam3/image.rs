//! Bounded, reusable input staging. Only shape/filter planning is on the host;
//! pixel filtering and normalization execute on Metal. Pillow needs F64 shape
//! arithmetic (not available in MSL), like the existing Clef processor.
use super::*;
use objc2_metal::MTLBuffer;

const MAX_STAGING: usize = 192 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sam3InputKind {
    Picture,
    VideoFrame,
}

struct Axis {
    ranges: Buffer,
    weights: Buffer,
    taps: usize,
}
struct Staging {
    key: (usize, usize, usize, Sam3InputKind),
    src: Buffer,
    mid: Buffer,
    axes: [Axis; 2],
}
#[derive(Default)]
pub(super) struct Input {
    staging: Option<Staging>,
}

fn taps(input: usize, output: usize, kind: Sam3InputKind) -> Result<usize> {
    if input == 0 || output == 0 || input > u32::MAX as usize || output > u32::MAX as usize {
        return Err(error("SAM 3 invalid resize dimensions"));
    }
    let n = 2 * (input as f64 / output as f64).max(1.).ceil() as usize + 1;
    let limit = if kind == Sam3InputKind::Picture {
        96
    } else {
        64
    };
    if n > limit {
        return Err(error(format!(
            "SAM 3 resize {input} -> {output} exceeds {limit}-tap input limit"
        )));
    }
    Ok(n)
}

fn axis(d: &MetalDevice, input: usize, output: usize, kind: Sam3InputKind) -> Result<Axis> {
    let taps = taps(input, output, kind)?;
    if kind == Sam3InputKind::Picture {
        let ranges = d.alloc(output * 8)?;
        let weights = d.alloc(output * taps * 4)?;
        let scale = input as f32 / output as f32;
        // CUDA's span has a double reciprocal, then one F32 rounding.
        let inv = (1.0f64 / f64::from(scale.max(1.))) as f32;
        let c = d.begin()?;
        point(
            &c,
            "sam3_resize_coeff",
            &[&ranges, &weights],
            &[
                input as u32,
                output as u32,
                taps as u32,
                scale.to_bits(),
                inv.to_bits(),
            ],
            output,
        );
        c.finish()?;
        return Ok(Axis {
            ranges,
            weights,
            taps,
        });
    }
    let scale = f64::from(input as f32) / output as f64;
    let support = scale.max(1.);
    let mut ranges = Vec::<u32>::with_capacity(output * 2);
    let mut weights = vec![0i32; output * taps];
    for i in 0..output {
        let center = (i as f64 + 0.5) * scale;
        let lo = ((center - support + 0.5) as i64).max(0) as usize;
        let hi = ((center + support + 0.5) as usize).min(input);
        let count = hi - lo;
        ranges.extend([lo as u32, count as u32]);
        let mut row = [0.; 64];
        for (j, w) in row[..count].iter_mut().enumerate() {
            let x = (((j + lo) as f64 - center + 0.5) * (1. / support)).abs();
            *w = if x < 1. { 1. - x } else { 0. };
        }
        let total: f64 = row[..count].iter().sum();
        for (j, w) in row[..count].iter().enumerate() {
            weights[i * taps + j] = (0.5 + w / total * f64::from(1 << 22)) as i32;
        }
    }
    Ok(Axis {
        ranges: d.upload(
            &ranges
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        weights: d.upload(
            &weights
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        taps,
    })
}

impl Input {
    // Changing geometry evicts the old plan; no unbounded per-shape cache.
    // A failed resize never authorizes use of old image features.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn land(
        &mut self,
        d: &MetalDevice,
        rgb: &[u8],
        width: usize,
        height: usize,
        side: usize,
        kind: Sam3InputKind,
        dst: &Buffer,
    ) -> Result<()> {
        let expected = width.checked_mul(height).and_then(|n| n.checked_mul(3));
        let intermediate = height.checked_mul(side).and_then(|n| {
            n.checked_mul(if kind == Sam3InputKind::Picture {
                12
            } else {
                3
            })
        });
        if expected != Some(rgb.len())
            || rgb.len() > MAX_STAGING
            || intermediate.is_none_or(|n| n > MAX_STAGING)
            || side.checked_mul(side).and_then(|n| n.checked_mul(3)) != Some(dst.len())
        {
            return Err(error(
                "SAM 3 invalid RGB geometry or resize staging exceeds 192 MiB",
            ));
        }
        // Validate both axes before admitting any allocation or GPU work.
        taps(width, side, kind)?;
        taps(height, side, kind)?;
        let key = (width, height, side, kind);
        if self.staging.as_ref().is_none_or(|s| s.key != key) {
            self.staging = None;
            self.staging = Some(Staging {
                key,
                src: d.alloc(rgb.len())?,
                mid: d.alloc(intermediate.expect("validated intermediate size"))?,
                axes: [axis(d, width, side, kind)?, axis(d, height, side, kind)?],
            });
        }
        let s = self.staging.as_ref().expect("admitted staging");
        // Previous invocation fenced both input and retained destination.
        unsafe {
            std::ptr::copy_nonoverlapping(
                rgb.as_ptr(),
                s.src.raw.contents().as_ptr().cast(),
                rgb.len(),
            );
        }
        let video = kind == Sam3InputKind::VideoFrame;
        if width == side && height == side {
            d.copy_regions(&[(&s.src, 0, dst, 0, rgb.len())])?;
            return Ok(());
        }
        let c = d.begin()?;
        let kernel = if video {
            "sam3_resize_video"
        } else {
            "sam3_resize_image"
        };
        let skip_h = video && width == side;
        let skip_v = video && height == side;
        if !skip_h {
            let a = &s.axes[0];
            point(
                &c,
                kernel,
                &[
                    &s.src,
                    if skip_v { dst } else { &s.mid },
                    &a.ranges,
                    &a.weights,
                ],
                &[width as u32, side as u32, height as u32, 1, a.taps as u32],
                height * side * 3,
            );
        }
        if !skip_v {
            let a = &s.axes[1];
            point(
                &c,
                kernel,
                &[
                    if skip_h { &s.src } else { &s.mid },
                    dst,
                    &a.ranges,
                    &a.weights,
                ],
                &[height as u32, side as u32, side as u32, 0, a.taps as u32],
                side * side * 3,
            );
        }
        c.finish()?;
        Ok(())
    }
    pub(super) fn bytes(&self) -> u64 {
        self.staging.as_ref().map_or(0, |s| {
            (s.src.len()
                + s.mid.len()
                + s.axes
                    .iter()
                    .map(|a| a.ranges.len() + a.weights.len())
                    .sum::<usize>()) as u64
        })
    }
}
