//! Same-checkpoint GPU captures isolate attention from quantized projections.
use super::*;

#[test]
#[ignore = "external MLX GPU SDPA captures; PADDOCK_EG2_ATTENTION"]
fn embeddinggemma2_attention_reference() {
    let root = std::env::var("PADDOCK_EG2_ATTENTION").unwrap();
    let device = MetalDevice::new(None).unwrap();
    for hd in [256usize, 512] {
        for name in ["ragged", "window", "long"] {
            let f = paddock_models::safetensors::SafetensorsFile::open(
                &Path::new(&root).join(format!("attention{hd}-{name}.safetensors")),
            )
            .unwrap();
            let lengths: Vec<u32> = f
                .bytes("lengths")
                .unwrap()
                .1
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let rows = lengths.iter().sum::<u32>() as usize;
            let upload = |v: &[u32]| {
                device
                    .upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
                    .unwrap()
            };
            let mut meta = Vec::new();
            let mut ends = Vec::new();
            let mut tiles = Vec::new();
            let mut start = 0;
            for &length in &lengths {
                for pos in 0..length {
                    meta.extend([start, pos]);
                    ends.push(length - 1);
                }
                for pos in (0..length).step_by(32) {
                    tiles.extend([start + pos, (length - pos).min(32)]);
                }
                start += length;
            }
            let nt = tiles.len() / 2;
            let (meta, ends, tiles) = (upload(&meta), upload(&ends), upload(&tiles));
            let q = device.upload(f.bytes("q").unwrap().1).unwrap();
            let k = device.upload(f.bytes("k").unwrap().1).unwrap();
            let v = device.upload(f.bytes("v").unwrap().1).unwrap();
            let y = device.alloc(rows * 4 * hd * 4).unwrap();
            let scores = device.alloc(128 * rows.div_ceil(64) * 64 * 4 * 2).unwrap();
            let c = device.begin().unwrap();
            if hd == 512 || lengths.iter().all(|&n| n < 1024) {
                let mut start = 0;
                for &length in &lengths {
                    for offset in (0..length).step_by(128) {
                        let count = (length - offset).min(128);
                        let p = [start, length, offset, count];
                        c.dispatch(
                            if hd == 512 {
                                "eg2_global_qk"
                            } else {
                                "eg2_local_qk"
                            },
                            &[&q, &k, &scores],
                            &p,
                            [length.div_ceil(64) as usize, count.div_ceil(32) as usize, 4],
                            128,
                        );
                        c.dispatch(
                            if length > 4096 {
                                "eg2_global_softmax"
                            } else {
                                "eg2_block_softmax"
                            },
                            &[&scores],
                            &p,
                            [count as usize, 4, 1],
                            if length > 4096 {
                                256
                            } else {
                                length.div_ceil(128) as usize * 32
                            },
                        );
                        c.dispatch(
                            if hd == 512 {
                                "eg2_global_pv"
                            } else {
                                "eg2_local_pv"
                            },
                            &[&scores, &v, &y],
                            &p,
                            [hd / 64, count.div_ceil(32) as usize, 4],
                            128,
                        );
                    }
                    start += length;
                }
            } else {
                c.dispatch(
                    "eg2_mlx_attention256",
                    &[&q, &k, &v, &meta, &ends, &y, &tiles],
                    &[4, 2, 0, 512, 0, 0, 1],
                    [4, nt, 1],
                    128,
                );
                c.dispatch(
                    "gmlx_round",
                    &[&y],
                    &[(rows * 4 * hd) as u32],
                    [(rows * 4 * hd).div_ceil(256), 1, 1],
                    256,
                );
            }
            c.finish().unwrap();
            let actual = unsafe { y.read_f32(0, rows * 4 * hd) };
            let expected: Vec<f32> = f
                .bytes("y")
                .unwrap()
                .1
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(actual.len(), expected.len());
            let mut aa = 0.;
            let mut bb = 0.;
            let mut ab = 0.;
            let mut max = 0f32;
            let mut different = 0;
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(a.is_finite());
                aa += f64::from(a).powi(2);
                bb += f64::from(b).powi(2);
                ab += f64::from(a) * f64::from(b);
                max = max.max((a - b).abs());
                different += usize::from(a != b);
            }
            let cosine = ab / (aa * bb).sqrt();
            eprintln!(
                "attention{hd} {name}: cosine={cosine:.10} max={max} unequal={different}/{}",
                actual.len()
            );
            if name == "ragged" {
                assert_eq!(different, 0, "short SDPA changed: {hd}");
            }
            assert!(cosine > 0.99999999, "isolated SDPA drift: {hd} {name}");
        }
    }
}
