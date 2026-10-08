//! Shape-only filter planning uses F64 (Metal has no native doubles). Pixel
//! convolution, clipping, normalization and patches all stay on the GPU.
//! The plan follows torchvision uint8 AA bicubic, shared contract with CUDA.
use super::*;
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1. {
        ((1.5 * x - 2.5) * x) * x + 1.
    } else if x < 2. {
        ((-0.5 * x + 2.5) * x - 4.) * x + 2.
    } else {
        0.
    }
}
fn axis(
    d: &MetalDevice,
    input: usize,
    output: usize,
    pillow: bool,
) -> Result<(Buffer, Buffer, u32, usize)> {
    let scale = input as f64 / output as f64;
    let support = 2. * scale.max(1.);
    let taps = 2 * support.ceil() as usize + 1;
    let mut ranges = Vec::<u32>::new();
    let mut weights = vec![0.; output * taps];
    let mut max = 0f64;
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let count = end.saturating_sub(start).min(taps);
        ranges.extend([start as u32, count as u32]);
        let row = &mut weights[i * taps..i * taps + count];
        for (j, w) in row.iter_mut().enumerate() {
            *w = cubic(((j + start) as f64 - center + 0.5) * (1. / scale.max(1.)));
        }
        let sum: f64 = row.iter().sum();
        for w in row {
            if sum != 0. {
                *w /= sum;
            }
            max = max.max(*w);
        }
    }
    let mut precision = 0;
    while precision < 22 && ((0.5 + max * f64::from(1u32 << (precision + 1))) as i32) < 32768 {
        precision += 1;
    }
    // mlx-vlm uses Pillow's 22-bit signed coefficients; the official HF
    // processor uses torchvision's adaptive int16 precision. Preserve both
    // checkpoint contracts rather than silently changing the input pixels.
    if pillow {
        precision = 22;
    }
    // Upload signed coefficients, not the planning doubles.
    let weights = weights
        .iter()
        .flat_map(|w| ((*w * f64::from(1u32 << precision)).round() as i32).to_le_bytes())
        .collect::<Vec<_>>();
    Ok((
        d.upload(
            &ranges
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?,
        d.upload(&weights)?,
        precision,
        taps,
    ))
}
struct Axis {
    key: (usize, usize, bool),
    ranges: Buffer,
    weights: Buffer,
    precision: u32,
    taps: usize,
}
/// One plan per axis, replaced on geometry changes. Repeated webcam frames
/// reuse both coefficient uploads without a growing shape cache.
#[derive(Default)]
pub(crate) struct Cache {
    axes: [Option<Axis>; 2],
}

impl Cache {
    pub(crate) fn is_empty(&self) -> bool {
        self.axes.iter().all(Option::is_none)
    }
    pub(super) fn pixels(
        &mut self,
        d: &MetalDevice,
        im: &ClefImage,
        pillow: bool,
    ) -> Result<Buffer> {
        self.pixels_raw(d, &im.rgb, im.width, im.height, im.resized, pillow)
    }

    pub(crate) fn pixels_raw(
        &mut self,
        d: &MetalDevice,
        rgb: &[u8],
        width: usize,
        height: usize,
        resized: (usize, usize),
        pillow: bool,
    ) -> Result<Buffer> {
        let (h, w) = resized;
        if width == 0
            || height == 0
            || w == 0
            || h == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
        {
            return Err(error("invalid RGB resize geometry"));
        }
        if height
            .checked_mul(w)
            .and_then(|n| n.checked_mul(3))
            .is_none_or(|n| n > 192 << 20)
        {
            return Err(error("image resize staging exceeds 192 MiB"));
        }
        let mut source = d.upload(rgb)?;
        for (index, (input, output, lines, horizontal)) in
            [(width, w, height, true), (height, h, w, false)]
                .into_iter()
                .enumerate()
        {
            if input == output {
                continue;
            }
            let key = (input, output, pillow);
            if self.axes[index].as_ref().is_none_or(|p| p.key != key) {
                // Drop the previous geometry before admitting the replacement.
                self.axes[index] = None;
                let (ranges, weights, precision, taps) = axis(d, input, output, pillow)?;
                self.axes[index] = Some(Axis {
                    key,
                    ranges,
                    weights,
                    precision,
                    taps,
                });
            }
            let plan = self.axes[index].as_ref().expect("prepared axis");
            let target = d.alloc(output * lines * 3)?;
            let cmd = d.begin()?;
            point(
                &cmd,
                "clef_vis_resize",
                &[&source, &target, &plan.ranges, &plan.weights],
                &[
                    input as u32,
                    output as u32,
                    lines as u32,
                    u32::from(horizontal),
                    plan.precision,
                    plan.taps as u32,
                ],
                output * lines * 3,
            );
            cmd.finish()?;
            source = target;
        }
        Ok(source)
    }
}
#[cfg(test)]
pub(in crate::clef) fn pixels(d: &MetalDevice, im: &ClefImage, pillow: bool) -> Result<Buffer> {
    Cache::default().pixels(d, im, pillow)
}
