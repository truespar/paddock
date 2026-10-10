//! Checkpoint -> device planes. `model.safetensors` is all F32, in the
//! Transformers `PPDocLayoutV3ForObjectDetection` layout. Every conv arrives
//! with its BatchNorm (eval statistics, eps 1e-5) folded in at load:
//! `w' = w * g / sqrt(var + eps)`, `b' = beta - mean * g / sqrt(var + eps)`,
//! in f32 before the one round to f16 - so the fold is never a second
//! rounding. Conv weights `[out][in][ky][kx]` are re-laid `[out][ky][kx][in]`
//! to match the im2row's tap order, the K dimension zero-padded to a multiple
//! of 8.

use cudarc::driver::CudaSlice;
use paddock_models::safetensors::{StDtype, TensorSource};

use super::{ConvBn, DwConv, GpuModelError};
use crate::gpu::{GpuExecutor, HalfTensor};

const BN_EPS: f32 = 1e-5;

/// How a conv + BatchNorm pair is named in the file.
#[derive(Clone, Copy)]
pub(super) enum Names {
    /// `<p>.convolution.weight` / `<p>.normalization.*` (HGNetV2, the mask head)
    ConvolutionNormalization,
    /// `<p>.conv.weight` / `<p>.norm.*` (the encoder's ConvNormLayer)
    ConvNorm,
    /// `<p>.0.weight` / `<p>.1.*` (the input projections' Sequential)
    Seq,
}

pub(super) struct Reader<'a> {
    pub(super) st: &'a dyn TensorSource,
    pub(super) exec: &'a GpuExecutor,
    pub(super) bytes: u64,
}

impl Reader<'_> {
    /// A tensor's f32 values, its shape checked exactly.
    pub(super) fn f32s(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, GpuModelError> {
        let (t, b) = self
            .st
            .tensor(name)
            .ok_or_else(|| GpuModelError::MissingMeta(format!("pp-doclayout-v3 tensor {name}")))?;
        if t.dtype != StDtype::F32 || t.shape != shape {
            return Err(GpuModelError::Unsupported(format!(
                "pp-doclayout-v3 {name}: {:?} {:?} (want F32 {shape:?})",
                t.dtype, t.shape
            )));
        }
        Ok(b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    pub(super) fn dev(&mut self, host: &[f32]) -> Result<CudaSlice<f32>, GpuModelError> {
        self.bytes += (host.len() * 4) as u64;
        Ok(self.exec.to_device(host)?)
    }

    /// The folded per-channel scale and bias of `<bn>`.
    fn bn(&self, bn: &str, c: usize) -> Result<(Vec<f32>, Vec<f32>), GpuModelError> {
        let g = self.f32s(&format!("{bn}.weight"), &[c])?;
        let beta = self.f32s(&format!("{bn}.bias"), &[c])?;
        let mean = self.f32s(&format!("{bn}.running_mean"), &[c])?;
        let var = self.f32s(&format!("{bn}.running_var"), &[c])?;
        let scale: Vec<f32> = (0..c).map(|i| g[i] / (var[i] + BN_EPS).sqrt()).collect();
        let bias = (0..c).map(|i| beta[i] - mean[i] * scale[i]).collect();
        Ok((scale, bias))
    }

    fn names(prefix: &str, names: Names) -> (String, String) {
        match names {
            Names::ConvolutionNormalization => (
                format!("{prefix}.convolution.weight"),
                format!("{prefix}.normalization"),
            ),
            Names::ConvNorm => (format!("{prefix}.conv.weight"), format!("{prefix}.norm")),
            Names::Seq => (format!("{prefix}.0.weight"), format!("{prefix}.1")),
        }
    }

    /// A dense k x k conv + its BatchNorm, folded, as a GEMM plane.
    pub(super) fn conv_bn(
        &mut self,
        prefix: &str,
        names: Names,
        (cin, cout, k, stride): (usize, usize, usize, usize),
    ) -> Result<ConvBn, GpuModelError> {
        let (wn, bn) = Self::names(prefix, names);
        let w = self.f32s(&wn, &[cout, cin, k, k])?;
        let (scale, bias) = self.bn(&bn, cout)?;
        let taps = k * k * cin;
        let kpad = taps.next_multiple_of(8);
        let mut plane = vec![0f32; cout * kpad];
        for o in 0..cout {
            for c in 0..cin {
                for ky in 0..k {
                    for kx in 0..k {
                        let v = w[((o * cin + c) * k + ky) * k + kx] * scale[o];
                        plane[o * kpad + (ky * k + kx) * cin + c] = v;
                    }
                }
            }
        }
        let buf = self.exec.to_device_f16(&plane, &wn)?;
        self.bytes += (plane.len() * 2) as u64;
        Ok(ConvBn {
            w: HalfTensor {
                buf,
                dims: vec![kpad, cout],
            },
            b: self.dev(&bias)?,
            k,
            stride,
            cin,
            cout,
            kpad,
        })
    }

    /// RepVGG's 3 x 3 + 1 x 1 branches (each with its BatchNorm) as the one
    /// 3 x 3 conv they sum to: `W = W3 s1 + center(W1 s2)`, `b = b1 + b2` -
    /// the form the official exported graph ships.
    pub(super) fn repvgg(&mut self, prefix: &str, c: usize) -> Result<ConvBn, GpuModelError> {
        let w3 = self.f32s(&format!("{prefix}.conv1.conv.weight"), &[c, c, 3, 3])?;
        let w1 = self.f32s(&format!("{prefix}.conv2.conv.weight"), &[c, c, 1, 1])?;
        let (s1, b1) = self.bn(&format!("{prefix}.conv1.norm"), c)?;
        let (s2, b2) = self.bn(&format!("{prefix}.conv2.norm"), c)?;
        let kpad = 9 * c;
        let mut plane = vec![0f32; c * kpad];
        for o in 0..c {
            for i in 0..c {
                for t in 0..9 {
                    let mut v = w3[(o * c + i) * 9 + t] * s1[o];
                    if t == 4 {
                        v += w1[o * c + i] * s2[o];
                    }
                    plane[o * kpad + t * c + i] = v;
                }
            }
        }
        let bias: Vec<f32> = (0..c).map(|o| b1[o] + b2[o]).collect();
        let what = format!("{prefix} (repvgg)");
        let buf = self.exec.to_device_f16(&plane, &what)?;
        self.bytes += (plane.len() * 2) as u64;
        Ok(ConvBn {
            w: HalfTensor {
                buf,
                dims: vec![kpad, c],
            },
            b: self.dev(&bias)?,
            k: 3,
            stride: 1,
            cin: c,
            cout: c,
            kpad,
        })
    }

    /// A plain conv with its own bias (no BatchNorm), as a GEMM plane.
    pub(super) fn conv_bias(
        &mut self,
        prefix: &str,
        (cin, cout, k): (usize, usize, usize),
    ) -> Result<ConvBn, GpuModelError> {
        let wn = format!("{prefix}.weight");
        let w = self.f32s(&wn, &[cout, cin, k, k])?;
        let bias = self.f32s(&format!("{prefix}.bias"), &[cout])?;
        let kpad = (k * k * cin).next_multiple_of(8);
        let mut plane = vec![0f32; cout * kpad];
        for o in 0..cout {
            for c in 0..cin {
                for t in 0..k * k {
                    plane[o * kpad + t * cin + c] = w[(o * cin + c) * k * k + t];
                }
            }
        }
        let buf = self.exec.to_device_f16(&plane, &wn)?;
        self.bytes += (plane.len() * 2) as u64;
        Ok(ConvBn {
            w: HalfTensor {
                buf,
                dims: vec![kpad, cout],
            },
            b: self.dev(&bias)?,
            k,
            stride: 1,
            cin,
            cout,
            kpad,
        })
    }

    /// `nn.Linear` `[out, in]` (already the GEMM layout) and its bias, both
    /// times `scale` (1/sqrt(d) folded into a query projection).
    pub(super) fn linear(
        &mut self,
        prefix: &str,
        (din, dout): (usize, usize),
        scale: f32,
    ) -> Result<(HalfTensor, CudaSlice<f32>), GpuModelError> {
        let wn = format!("{prefix}.weight");
        let mut w = self.f32s(&wn, &[dout, din])?;
        let mut b = self.f32s(&format!("{prefix}.bias"), &[dout])?;
        if scale != 1.0 {
            w.iter_mut().for_each(|v| *v *= scale);
            b.iter_mut().for_each(|v| *v *= scale);
        }
        let buf = self.exec.to_device_f16(&w, &wn)?;
        self.bytes += (w.len() * 2) as u64;
        Ok((
            HalfTensor {
                buf,
                dims: vec![din, dout],
            },
            self.dev(&b)?,
        ))
    }

    /// A Linear whose input width is zero-padded to `kin` (the GEMM's K
    /// must be a multiple of 8: the 4-wide box input of `query_pos_head`).
    pub(super) fn linear_padded(
        &mut self,
        prefix: &str,
        (din, dout): (usize, usize),
        kin: usize,
    ) -> Result<(HalfTensor, CudaSlice<f32>), GpuModelError> {
        let wn = format!("{prefix}.weight");
        let w = self.f32s(&wn, &[dout, din])?;
        let b = self.f32s(&format!("{prefix}.bias"), &[dout])?;
        let mut plane = vec![0f32; dout * kin];
        for o in 0..dout {
            plane[o * kin..o * kin + din].copy_from_slice(&w[o * din..(o + 1) * din]);
        }
        let buf = self.exec.to_device_f16(&plane, &wn)?;
        self.bytes += (plane.len() * 2) as u64;
        Ok((
            HalfTensor {
                buf,
                dims: vec![kin, dout],
            },
            self.dev(&b)?,
        ))
    }

    /// The global pointer's `dense` (`[2 * hs, d]`) split into its query rows
    /// (the first `hs`, times 1/sqrt(hs)) and its key rows.
    pub(super) fn pointer(
        &mut self,
        prefix: &str,
        d: usize,
        hs: usize,
    ) -> Result<((HalfTensor, CudaSlice<f32>), (HalfTensor, CudaSlice<f32>)), GpuModelError> {
        let wn = format!("{prefix}.weight");
        let w = self.f32s(&wn, &[2 * hs, d])?;
        let b = self.f32s(&format!("{prefix}.bias"), &[2 * hs])?;
        let scale = 1.0 / (hs as f32).sqrt();
        let mut half = |rows: std::ops::Range<usize>, s: f32| {
            let wp: Vec<f32> = w[rows.start * d..rows.end * d]
                .iter()
                .map(|v| v * s)
                .collect();
            let bp: Vec<f32> = b[rows].iter().map(|v| v * s).collect();
            let buf = self.exec.to_device_f16(&wp, &wn)?;
            self.bytes += (wp.len() * 2) as u64;
            Ok::<_, GpuModelError>((
                HalfTensor {
                    buf,
                    dims: vec![d, hs],
                },
                self.dev(&bp)?,
            ))
        };
        let q = half(0..hs, scale)?;
        let k = half(hs..2 * hs, 1.0)?;
        Ok((q, k))
    }

    /// LayerNorm weight + bias.
    pub(super) fn norm(
        &mut self,
        prefix: &str,
        d: usize,
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>), GpuModelError> {
        let w = self.f32s(&format!("{prefix}.weight"), &[d])?;
        let b = self.f32s(&format!("{prefix}.bias"), &[d])?;
        Ok((self.dev(&w)?, self.dev(&b)?))
    }

    /// A depthwise k x k conv + its BatchNorm, folded, f32.
    pub(super) fn dw_bn(
        &mut self,
        prefix: &str,
        (c, k, stride): (usize, usize, usize),
    ) -> Result<DwConv, GpuModelError> {
        let (wn, bn) = Self::names(prefix, Names::ConvolutionNormalization);
        let w = self.f32s(&wn, &[c, 1, k, k])?;
        let (scale, bias) = self.bn(&bn, c)?;
        let folded: Vec<f32> = (0..c * k * k).map(|i| w[i] * scale[i / (k * k)]).collect();
        Ok(DwConv {
            w: self.dev(&folded)?,
            b: self.dev(&bias)?,
            k,
            stride,
            c,
        })
    }
}
