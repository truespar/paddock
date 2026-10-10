use super::*;
use objc2_metal::MTLBuffer;

fn halves(b: &Buffer, n: usize) -> Vec<f32> {
    // Test callers have completed the producing command buffer.
    unsafe {
        std::slice::from_raw_parts(b.raw.contents().as_ptr().cast::<half::f16>(), n)
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MMPROJ and PADDOCK_METAL_LIGHTON_SEAMS from metal-lighton-vision.py"]
fn lighton_mlx_tower_seams_diagnostic() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").unwrap();
    let captures = std::env::var_os("PADDOCK_METAL_LIGHTON_SEAMS").unwrap();
    let cfg = paddock_models::mlx::QwenConfig::read(Path::new(&path)).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&d, Path::new(&path), cfg.width).unwrap();
    assert!(vision.mlx_bf16);
    let check = |name: &str, buffer: &Buffer, n: usize| {
        let raw = std::fs::read(Path::new(&captures).join(format!("{name}.f32"))).unwrap();
        assert_eq!(raw.len(), n * 4);
        let expected: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let actual = unsafe { buffer.read_f32(0, n) };
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| {
                assert!(a.is_finite());
                (a - b).abs()
            })
            .fold(0f32, f32::max);
        let sq: f64 = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|v| f64::from(*v).powi(2)).sum();
        let exact = actual
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        eprintln!(
            "LIGHTON_SEAM {name} max={max} relative_l2={} exact={exact}/{n}",
            (sq / norm.max(1e-30)).sqrt()
        );
    };
    let rgb: Vec<_> = (0..256 * 256 * 3)
        .map(|i| ((i * 13 + i / (256 * 3) * 7) % 256) as u8)
        .collect();
    let mut job = vision.start(&d, &[(&rgb, 256, 256)]).unwrap();
    let pixels = d.alloc(job.rows * 1536 * 4).unwrap();
    let bad = d.upload(&[0u8; 4]).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "vis_cast",
        &[&job.stage, &pixels, &bad],
        &[(job.rows * 1536) as u32, 30, 0],
        [(job.rows * 1536).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    check("pixels", &pixels, job.rows * 1536);
    let zero_pos = d
        .upload(&vec![0; 2304 * vision.geometry.width * 4])
        .unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "qmlx_vis_patch_mm",
        &[&vision.patch0.buffer, &job.stage, &job.attn],
        &[vision.geometry.width as u32, job.rows as u32],
        [vision.geometry.width.div_ceil(64), job.rows.div_ceil(64), 1],
        128,
    );
    cmd.dispatch(
        "qmlx_vis_position",
        &[&job.attn, &job.attn, &vision.bias.buffer, &zero_pos],
        &[16, 16, 0, vision.geometry.width as u32, 48],
        [(job.rows * vision.geometry.width).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    check("patch", &job.attn, job.rows * vision.geometry.width);
    check("position", &job.x, job.rows * vision.geometry.width);
    let b = &vision.blocks[0];
    let cmd = d.begin().unwrap();
    vision.ln(&cmd, &job.x, &b.ln1, &b.ln1b, &job.stage, job.rows);
    vision.project(&cmd, &b.qkv, &job.stage, &job.qkv, &b.qkvb, job.rows, 1);
    cmd.finish().unwrap();
    check("norm", &job.stage, job.rows * vision.geometry.width);
    check("qkv", &job.qkv, job.rows * vision.geometry.width * 3);
    // Isolate arithmetic from upstream drift: replay each reference seam
    // through the real GPU kernel, never substitute it during serving.
    let input = |name: &str, bf16: bool| {
        let raw = std::fs::read(Path::new(&captures).join(format!("{name}.f32"))).unwrap();
        if bf16 {
            d.upload(
                &raw.chunks_exact(4)
                    .flat_map(|b| {
                        half::bf16::from_f32(f32::from_le_bytes(b.try_into().unwrap()))
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        } else {
            d.upload(&raw).unwrap()
        }
    };
    let norm = input("norm", false);
    let position = input("position", false);
    let cmd = d.begin().unwrap();
    vision.ln(&cmd, &position, &b.ln1, &b.ln1b, &job.stage, job.rows);
    cmd.finish().unwrap();
    check("norm", &job.stage, job.rows * vision.geometry.width);
    let cmd = d.begin().unwrap();
    vision.project(&cmd, &b.qkv, &norm, &job.qkv, &b.qkvb, job.rows, 1);
    cmd.finish().unwrap();
    check("qkv", &job.qkv, job.rows * vision.geometry.width * 3);
    let (q, k, v) = (
        input("query", true),
        input("key", true),
        input("value", true),
    );
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        if vision.geometry.heads == 12 {
            "qmlx_vis_attention_12"
        } else {
            "qmlx_vis_attention_16"
        },
        &[&q, &k, &v, &job.attn, &job.tiles],
        &[31],
        [vision.geometry.heads, job.tile_count, 1],
        128,
    );
    cmd.finish().unwrap();
    check("attention", &job.attn, job.rows * vision.geometry.width);
    let norm2 = input("norm2", false);
    let reference_qkv = input("qkv", false);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        if vision.geometry.heads == 12 {
            "qmlx_vis_qkv_12"
        } else {
            "qmlx_vis_qkv_16"
        },
        &[&reference_qkv, &job.xy, &job.q, &job.k, &job.v],
        &[job.rows as u32],
        [
            ((job.rows + 64) * vision.geometry.width).div_ceil(256),
            1,
            1,
        ],
        256,
    );
    cmd.finish().unwrap();
    for (name, buffer) in [("query", &job.q), ("key", &job.k), ("value", &job.v)] {
        let expanded = d.alloc(job.rows * vision.geometry.width * 4).unwrap();
        let bad = d.upload(&[0u8; 4]).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "vis_cast",
            &[buffer, &expanded, &bad],
            &[(job.rows * vision.geometry.width) as u32, 30, 0],
            [(job.rows * vision.geometry.width).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        check(name, &expanded, job.rows * vision.geometry.width);
    }
    let attn = input("attention", false);
    let residual = input("position", false);
    let cmd = d.begin().unwrap();
    vision.project(&cmd, &b.out, &attn, &residual, &b.outb, job.rows, 2);
    cmd.finish().unwrap();
    check("residual", &residual, job.rows * vision.geometry.width);
    let reference_residual = input("residual", false);
    let cmd = d.begin().unwrap();
    vision.ln(
        &cmd,
        &reference_residual,
        &b.ln2,
        &b.ln2b,
        &job.attn,
        job.rows,
    );
    cmd.finish().unwrap();
    check("norm2", &job.attn, job.rows * vision.geometry.width);
    let cmd = d.begin().unwrap();
    vision.project(&cmd, &b.up, &norm2, &job.stage, &b.upb, job.rows, 1);
    cmd.finish().unwrap();
    check("up", &job.stage, job.rows * vision.geometry.ff);
    let cmd = d.begin().unwrap();
    vision.project(&cmd, &b.up, &norm2, &job.stage, &b.upb, job.rows, 3);
    cmd.finish().unwrap();
    check("gelu", &job.stage, job.rows * vision.geometry.ff);
    let reference_gelu = input("gelu", false);
    let cmd = d.begin().unwrap();
    vision.project(
        &cmd,
        &b.down,
        &reference_gelu,
        &reference_residual,
        &b.downb,
        job.rows,
        2,
    );
    cmd.finish().unwrap();
    check(
        "block-0",
        &reference_residual,
        job.rows * vision.geometry.width,
    );
    for layer in 0..vision.blocks.len() {
        assert!(vision.step(&d, &mut job).unwrap().is_none());
        if layer == 0 || layer == vision.blocks.len() - 1 {
            check(
                &format!("block-{layer}"),
                &job.x,
                job.rows * vision.geometry.width,
            );
        }
    }
    let output = vision.step(&d, &mut job).unwrap().unwrap();
    check("merger", &output[0].embd, job.rows / 4 * cfg.width);
}

#[test]
fn mlx_patch_pixels_match_processor_rounding_and_temporal_layout() {
    let d = MetalDevice::new(Some(16 << 20)).unwrap();
    let rgb: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
    let source = d.upload(&rgb).unwrap();
    // A nonzero ragged offset, both temporal planes, all byte values, and
    // sentinels on both sides of the exact four-row destination.
    let out = d.upload(&vec![0xff; 6 * 1536 * 2]).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "qmlx_vis_patches",
        &[&source, &out],
        &[32, 32, 32, 32, 1],
        [4 * 1536 / 256, 1, 1],
        256,
    );
    cmd.finish().unwrap();
    let actual: Vec<_> = unsafe {
        std::slice::from_raw_parts(out.raw.contents().as_ptr().cast::<half::bf16>(), 6 * 1536)
    }
    .iter()
    .map(|v| v.to_f32())
    .collect();
    assert!(
        actual[..1536]
            .iter()
            .chain(&actual[5 * 1536..])
            .all(|v| v.is_nan())
    );
    for row in 0..4 {
        for j in 0..1536 {
            let p = j % 768;
            let x = row % 2 * 16 + p / 3 % 16;
            let y = row / 2 * 16 + p / 48;
            let rescaled = f32::from(rgb[(y * 32 + x) * 3 + p % 3]) * (1. / 255.);
            let expected = half::bf16::from_f32((rescaled - 0.5) * 2.).to_f32();
            assert_eq!(
                actual[(row + 1) * 1536 + j],
                expected,
                "row={row} element={j}"
            );
        }
    }
}

#[test]
#[ignore = "isolated GPU cost; run alone, never beside another GPU measurement"]
fn mlx_vision_attention_cost() {
    let rows = std::env::var("PADDOCK_METAL_ATTENTION_ROWS")
        .map_or(11520, |s| s.parse::<usize>().unwrap());
    let heads =
        std::env::var("PADDOCK_METAL_ATTENTION_HEADS").map_or(12, |s| s.parse::<usize>().unwrap());
    assert!((1..=16384).contains(&rows) && matches!(heads, 12 | 16));
    let d = MetalDevice::new(None).unwrap();
    let width = heads * 64;
    let plane = |salt| {
        d.upload(
            &(0..rows * width)
                .flat_map(|i| {
                    half::bf16::from_f32(((i * salt % 127) as f32 - 63.) / 32.).to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (q, k, v) = (plane(3), plane(7), plane(11));
    let out = d.alloc(rows * width * 4).unwrap();
    let mut previous: Option<Vec<f32>> = None;
    for (query_tile, bounded) in [(32usize, true), (64, false), (64, true), (32, false)] {
        let kernel = match (heads, bounded) {
            (12, true) => "qmlx_vis_attention_bounded_12",
            (16, true) => "qmlx_vis_attention_bounded_16",
            (12, false) => "qmlx_vis_attention_12",
            (16, false) => "qmlx_vis_attention_16",
            _ => unreachable!(),
        };
        let kernel = if std::env::var_os("PADDOCK_METAL_ATTENTION_CANDIDATE").is_some() {
            match (heads, bounded) {
                (12, true) => "qmlx_vis_attention_candidate_bounded_12",
                (16, true) => "qmlx_vis_attention_candidate_bounded_16",
                (12, false) => "qmlx_vis_attention_candidate_12",
                (16, false) => "qmlx_vis_attention_candidate_16",
                _ => unreachable!(),
            }
        } else {
            kernel
        };
        let tiles: Vec<_> = (0..rows)
            .step_by(query_tile)
            .flat_map(|row| {
                [
                    row as u32,
                    (rows - row).min(query_tile) as u32,
                    0,
                    rows as u32,
                ]
            })
            .flat_map(u32::to_le_bytes)
            .collect();
        let tiles = d.upload(&tiles).unwrap();
        let (mut wall_ms, mut gpu_ms) = (Vec::new(), Vec::new());
        for rep in 0..9 {
            let start = std::time::Instant::now();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                kernel,
                &[&q, &k, &v, &out, &tiles],
                &[31],
                [heads, rows.div_ceil(query_tile), 1],
                query_tile * 2,
            );
            let gpu = cmd.finish().unwrap();
            if rep >= 2 {
                wall_ms.push(start.elapsed().as_secs_f64() * 1000.);
                gpu_ms.push(gpu * 1000.);
            }
        }
        let values = unsafe { out.read_f32(0, rows * width) };
        assert!(values.iter().all(|v| v.is_finite()));
        if let Some(ref p) = previous {
            assert_eq!(
                values
                    .iter()
                    .zip(p.iter())
                    .position(|(a, b)| a.to_bits() != b.to_bits()),
                None,
                "tile scheduling changed arithmetic"
            );
        }
        previous = Some(values);
        eprintln!(
            "ATTENTION_COST {}",
            serde_json::json!({
            "kernel":kernel,"rows":rows,"heads":heads,"query_tile":query_tile,"bounded":bounded,"wall_ms":wall_ms,"gpu_ms":gpu_ms
            })
        );
    }
}

#[test]
fn mlx_vision_attention_bounds_and_gpu_oracle() {
    check_mlx_vision_attention_bounds(false);
}

#[test]
fn mlx_vision_attention_candidate_bounds_and_gpu_oracle() {
    check_mlx_vision_attention_bounds(true);
}

fn check_mlx_vision_attention_bounds(candidate: bool) {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    for (heads, query_tile) in [(12usize, 32), (12, 64), (16, 32), (16, 64)] {
        let sizes = [1usize, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 128, 132];
        let rows: usize = sizes.iter().sum();
        let width = heads * 64;
        let plane = |salt| {
            let values: Vec<_> = (0..rows * width)
                .map(|i| half::bf16::from_f32(((i * salt % 127) as f32 - 63.) / 32.).to_f32())
                .collect();
            (
                d.upload(
                    &values
                        .iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
                d.upload(
                    &values
                        .iter()
                        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            )
        };
        let ((q, qh), (k, kh), (v, vh)) = (plane(3), plane(7), plane(11));
        let (mut tiles, mut bounds, mut first) = (Vec::<u32>::new(), Vec::<u32>::new(), 0);
        for n in sizes {
            for row in (0..n).step_by(query_tile) {
                tiles.extend([
                    (first + row) as u32,
                    (n - row).min(query_tile) as u32,
                    first as u32,
                    n as u32,
                ]);
            }
            for _ in 0..n {
                bounds.extend([first as u32, (first + n) as u32]);
            }
            first += n;
        }
        let upload = |v: &[u32]| {
            d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let (tiles_buffer, bounds) = (upload(&tiles), upload(&bounds));
        let out = d
            .upload(
                &vec![f32::NAN; (rows + 32) * width]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let reference = d.alloc(rows * width * 4).unwrap();
        let bounded = d.alloc(rows * width * 4).unwrap();
        let cmd = d.begin().unwrap();
        let suffix = if candidate { "_candidate" } else { "" };
        cmd.dispatch(
            &format!("qmlx_vis_attention{suffix}_{heads}"),
            &[&q, &k, &v, &out, &tiles_buffer],
            &[31],
            [heads, tiles.len() / 4, 1],
            query_tile * 2,
        );
        cmd.dispatch(
            &format!("qmlx_vis_attention{suffix}_bounded_{heads}"),
            &[&q, &k, &v, &bounded, &tiles_buffer],
            &[31],
            [heads, tiles.len() / 4, 1],
            query_tile * 2,
        );
        cmd.dispatch(
            "vis_attention_check",
            &[&qh, &kh, &vh, &reference, &bounds],
            &[64, heads as u32, 64],
            [heads, rows, 1],
            32,
        );
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, (rows + 32) * width) };
        let bounded = unsafe { bounded.read_f32(0, rows * width) };
        assert_eq!(
            actual
                .iter()
                .zip(&bounded)
                .position(|(a, b)| a.to_bits() != b.to_bits()),
            None,
            "aligned specialization changed arithmetic"
        );
        let reference = unsafe { reference.read_f32(0, rows * width) };
        for (i, (&a, &r)) in actual.iter().zip(&reference).enumerate() {
            assert!(
                a.is_finite() && (a - r).abs() < 0.016,
                "heads={heads} query_tile={query_tile} element={i} actual={a} oracle={r}"
            );
            assert_eq!(
                a,
                half::bf16::from_f32(a).to_f32(),
                "missing output BF16 boundary"
            );
        }
        assert!(
            actual[rows * width..].iter().all(|v| v.is_nan()),
            "tail guard overwritten"
        );
    }
}

#[test]
fn native_bf16_projection_preserves_range_and_identity() {
    // An identity contraction, not a host matrix oracle. These values expose
    // accidental F16 reinterpretation, underflow, overflow and double rounding.
    let device = MetalDevice::new(Some(16 << 20)).unwrap();
    let values: [f32; 8] = [1.001, 1e-10, 131072.0, -131072.0, 0.0, -1e-10, 3.5, -7.0];
    let input = (0..32 * 8).map(|i| values[i % 8]).collect::<Vec<_>>();
    let x = device
        .upload(
            &input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let w = Weight {
        buffer: device
            .upload(
                &(0..64 * 32)
                    .flat_map(|i| {
                        half::bf16::from_f32(if i % 32 == i / 32 % 32 { 1.0 } else { 0.0 })
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ty: 30,
        k: 32,
        n: 64,
    };
    let b = Weight {
        buffer: device.upload(&vec![0; 64 * 4]).unwrap(),
        ty: 0,
        k: 64,
        n: 1,
    };
    let out = device.alloc(8 * 64 * 4).unwrap();
    let cmd = device.begin().unwrap();
    Vision::mm::<false, false>(&cmd, &w, &x, &out, &b, 8, 1);
    cmd.finish().unwrap();
    let actual = unsafe { out.read_f32(0, 8 * 64) };
    for (i, &v) in actual.iter().enumerate() {
        assert_eq!(v, input[i / 64 * 32 + i % 32], "element {i}");
    }
    // The accelerated vision route has a distinct, explicitly bounded
    // multiplication contract. It must still retain exponent range and
    // finite output. This does not relax the strict identity assertion above.
    let cmd = device.begin().unwrap();
    Vision::mm::<true, false>(&cmd, &w, &x, &out, &b, 8, 1);
    cmd.finish().unwrap();
    for (i, v) in unsafe { out.read_f32(0, 8 * 64) }.into_iter().enumerate() {
        let expected = input[i / 64 * 32 + i % 32];
        assert!(v.is_finite());
        assert!(
            (v - expected).abs() <= expected.abs() / 1024.0,
            "accelerated identity {i}: {v} / {expected}"
        );
    }
}

#[test]
fn fused_gelu_preserves_large_outliers_and_finite_output() {
    let d = MetalDevice::new(Some(16 << 20)).unwrap();
    let values = [-100.0, -34.0, -12.0, 0.0, 12.0, 16.0, 34.0, 100.0];
    let x = d
        .upload(
            &values
                .iter()
                .flat_map(|&v| {
                    (0..32).flat_map(move |i| {
                        half::f16::from_f32(if i == 0 { v } else { 0.0 }).to_le_bytes()
                    })
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let w = Weight {
        buffer: d
            .upload(
                &(0..64 * 32)
                    .flat_map(|i| {
                        half::f16::from_f32(if i % 32 == 0 { 1.0 } else { 0.0 }).to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ty: 1,
        k: 32,
        n: 64,
    };
    let b = Weight {
        buffer: d.upload(&vec![0; 64 * 4]).unwrap(),
        ty: 0,
        k: 64,
        n: 1,
    };
    let y = d.alloc(values.len() * 64 * 2).unwrap();
    let cmd = d.begin().unwrap();
    Vision::mm::<false, true>(&cmd, &w, &x, &y, &b, values.len(), 3);
    cmd.finish().unwrap();
    let actual = halves(&y, values.len() * 64);
    for (i, &v) in values.iter().enumerate() {
        assert!(
            actual[i * 64..(i + 1) * 64]
                .iter()
                .all(|&a| a == v.max(0.0)),
            "outlier {v}"
        );
    }
}

#[test]
fn half_weights_consume_float_activations_and_preserve_float_epilogues() {
    let device = MetalDevice::new(Some(16 << 20)).unwrap();
    // Native FP16 towers still consume F32 layernorm/attention buffers. Include
    // values outside FP16's range so a stray cast is caught, not merely a bad
    // byte reinterpretation. Exercise both matrix tile sizes and epilogues.
    for rows in [7usize, 129] {
        let input: Vec<f32> = (0..rows * 32)
            .map(|i| [1.001, 131072., -131072., 1e-10][i % 4])
            .collect();
        let x = device
            .upload_parts(&[&input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()])
            .unwrap();
        let w = Weight {
            buffer: device
                .upload_parts(&[&(0..64 * 32)
                    .flat_map(|i| {
                        half::f16::from_f32(if i % 32 == (i / 32) % 32 { 1. } else { 0. })
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>()])
                .unwrap(),
            ty: 1,
            k: 32,
            n: 64,
        };
        let bias = Weight {
            buffer: device.upload_parts(&[&vec![0; 64 * 4]]).unwrap(),
            ty: 0,
            k: 64,
            n: 1,
        };
        let output = device.alloc(rows * 64 * 4).unwrap();
        for epilogue in [1, 3] {
            let cmd = device.begin().unwrap();
            Vision::mm::<true, false>(&cmd, &w, &x, &output, &bias, rows, epilogue);
            cmd.finish().unwrap();
            // SAFETY: the completed command initialized every output element.
            for (i, actual) in unsafe { output.read_f32(0, rows * 64) }
                .into_iter()
                .enumerate()
            {
                let value = input[i / 64 * 32 + i % 32];
                let expected = if epilogue == 1 {
                    value
                } else {
                    value
                        * 0.5
                        * (1. + (0.7978846 * value * (1. + 0.044715 * value * value)).tanh())
                };
                assert!(
                    (actual - expected).abs() <= 1e-5 * expected.abs().max(1e-10),
                    "{rows}/{epilogue}/{i}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn noncausal_tensor_attention_matches_gpu_oracle_and_isolates_ragged_images() {
    for decoder in [1024, 2560, 5120] {
        ragged_attention_matches_oracle(TowerGeometry::for_decoder(decoder).unwrap());
    }
}

fn ragged_attention_matches_oracle(g: TowerGeometry) {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let e = g.width;
    let padded = g.padded_width();
    let hd = e / g.heads;
    let pad = padded / g.heads;
    // Three KV tiles in the middle image exercise repeated online rescaling
    // and accumulation; both image boundaries remain deliberately ragged.
    let sizes = [37usize, 132, 13];
    let rows: usize = sizes.iter().sum();
    let plane = |seed: usize| {
        d.upload(
            &(0..(rows + 64) * padded)
                .flat_map(|i| {
                    let v = if i % pad < hd && i / padded < rows {
                        ((i * seed % 31) as f32 - 15.0) / 64.0
                    } else {
                        0.0
                    };
                    half::f16::from_f32(v).to_le_bytes()
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    let (q, k, v) = (plane(3), plane(7), plane(11));
    let mut tiles = Vec::new();
    let mut bounds = Vec::new();
    let mut first = 0;
    for n in sizes {
        for row in (0..n).step_by(32) {
            tiles.extend([
                (first + row) as u32,
                (n - row).min(32) as u32,
                first as u32,
                n as u32,
            ]);
        }
        for _ in 0..n {
            bounds.extend([first as u32, (first + n) as u32]);
        }
        first += n;
    }
    let upload = |v: &[u32]| {
        d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap()
    };
    let t = upload(&tiles);
    let b = upload(&bounds);
    let out = d
        .upload(
            &(0..(rows + 32) * e)
                .flat_map(|_| half::f16::NAN.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let reference = d.alloc(rows * e * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        g.attention_kernel(),
        &[&q, &k, &v, &out, &t],
        &[1],
        [g.heads, tiles.len() / 4, 1],
        64,
    );
    cmd.dispatch(
        "vis_attention_check",
        &[&q, &k, &v, &reference, &b],
        &[hd as u32, g.heads as u32, pad as u32],
        [g.heads, rows, 1],
        32,
    );
    cmd.finish().unwrap();
    let actual = halves(&out, rows * e);
    let expected = unsafe { reference.read_f32(0, rows * e) };
    assert!(actual.iter().all(|v| v.is_finite()));
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 0.0001, "GPU attention error {max}");
    let full = d.alloc(rows * e * 4).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        g.attention_kernel(),
        &[&q, &k, &v, &full, &t],
        &[0],
        [g.heads, tiles.len() / 4, 1],
        64,
    );
    cmd.finish().unwrap();
    let actual = unsafe { full.read_f32(0, rows * e) };
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max < 0.000002,
        "F32 probability/output GPU attention error {max}"
    );
    assert!(
        halves(&out, (rows + 32) * e)[rows * e..]
            .iter()
            .all(|v| v.is_nan()),
        "tail guard"
    );
}

#[test]
#[ignore = "requires PADDOCK_METAL_MMPROJ and an M5"]
fn tower_finite_and_ragged_batch_isolation() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("mmproj path");
    let device = MetalDevice::new(None).unwrap();
    let width = if Path::new(&path).is_dir() {
        paddock_models::mlx::QwenConfig::read(Path::new(&path))
            .unwrap()
            .width
    } else {
        let map = MappedGguf::open(Path::new(&path)).unwrap();
        map.tensor_info("mm.2.weight").unwrap().dims[1] as usize
    };
    let geometry = TowerGeometry::for_decoder(width).unwrap();
    let vision = Vision::load(&device, Path::new(&path), width).unwrap();
    if vision.mlx_bf16 {
        assert_eq!(vision.dimensions(144, 512).unwrap(), (128, 512));
        assert_eq!(vision.dimensions(176, 512).unwrap(), (192, 512));
        assert!(vision.dimensions(0, 512).is_err());
        assert!(vision.dimensions(1, 201).is_err());
    }
    let red: Vec<u8> = (0..256 * 256).flat_map(|_| [255, 0, 0]).collect();
    let pattern: Vec<u8> = (0..288 * 288 * 3).map(|i| (i * 13 % 251) as u8).collect();
    let encode = |images: &[(&[u8], usize, usize)]| {
        let mut job = vision.start(&device, images).unwrap();
        loop {
            let output = vision.step(&device, &mut job).unwrap();
            assert!(
                unsafe { job.x.read_f32(0, job.rows * geometry.width) }
                    .iter()
                    .all(|v| v.is_finite()),
                "block {}",
                job.layer
            );
            if let Some(output) = output {
                return output;
            }
        }
    };
    let a = encode(&[(&red, 256, 256)]);
    let b = encode(&[(&pattern, 288, 288)]);
    // Submission grouping must not change any vision embedding bit. Include
    // ragged row counts and reverse image order below as separate coverage.
    let mut grouped = vision.start(&device, &[(&pattern, 288, 288)]).unwrap();
    let grouped_out = loop {
        if let Some(out) = vision
            .step_budget(&device, &mut grouped, PREFILL_QUANTUM)
            .unwrap()
        {
            break out;
        }
    };
    assert!(
        grouped.submits < geometry.layers + 2,
        "blocks should share command submissions"
    );
    let mut fused = vision.start(&device, &[(&pattern, 288, 288)]).unwrap();
    let fused_out = vision
        .step_blocks(&device, &mut fused, geometry.layers, true)
        .unwrap()
        .unwrap();
    assert_eq!(fused.layer, geometry.layers);
    assert_eq!(
        fused.submits, 2,
        "setup plus blocks/merger; no publication-only yield"
    );
    assert_eq!(
        unsafe { fused_out[0].embd.read_f32(0, 81 * width) },
        unsafe { b[0].embd.read_f32(0, 81 * width) },
        "fused and independently submitted tower/merger must be byte-identical"
    );
    assert_eq!(
        unsafe { grouped_out[0].embd.read_f32(0, 81 * width) },
        unsafe { b[0].embd.read_f32(0, 81 * width) }
    );
    for inputs in [
        vec![(&red[..], 256, 256), (&pattern[..], 288, 288)],
        vec![(&pattern[..], 288, 288), (&red[..], 256, 256)],
    ] {
        let batched = encode(&inputs);
        for (i, o) in batched.iter().enumerate() {
            let reference = if inputs[i].1 == 256 { &a[0] } else { &b[0] };
            let actual = unsafe { o.embd.read_f32(0, o.nx * o.ny * width) };
            let expected = unsafe { reference.embd.read_f32(0, actual.len()) };
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("ragged tower image {i}: max error {max}");
            assert!(max < 0.02, "ragged tower image contamination {max}");
        }
    }
}

/// Diagnostic transport only: all model arithmetic uses the real GPU path.
/// Invoked by the greedy-parity harness (--vision-tensor-debug), whose newest
/// prebuilt GPU reference prints the same first/last-three tensor samples.
#[test]
#[ignore = "requires canonical mmproj and an M5; tensor diagnostic via greedy-parity.py"]
fn vision_white_tensor_capture() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("mmproj path");
    let device = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&device, Path::new(&path), 5120).unwrap();
    assert_eq!(vision.mean, [0.5; 3]);
    assert_eq!(vision.std, [0.5; 3]);
    let white = vec![255u8; 256 * 256 * 3];
    let mut job = vision.start(&device, &[(&white, 256, 256)]).unwrap();
    let mut captures = serde_json::Map::new();
    let edge = |n: usize| (0..n).filter(move |&i| i < 3 || i >= n.saturating_sub(3));
    let mut capture =
        |name: String, buffer: &Buffer, dims: [usize; 3], strides: [usize; 3], offset: usize| {
            let len = offset
                + (dims[2] - 1) * strides[2]
                + (dims[1] - 1) * strides[1]
                + (dims[0] - 1) * strides[0]
                + 1;
            let values = unsafe { buffer.read_f32(0, len) };
            assert!(values.iter().all(|v| v.is_finite()));
            let mut samples = Vec::new();
            for k in edge(dims[2]) {
                for j in edge(dims[1]) {
                    for i in edge(dims[0]) {
                        samples.push(
                            values[offset + i * strides[0] + j * strides[1] + k * strides[2]],
                        );
                    }
                }
            }
            captures.insert(
                name,
                serde_json::json!({"dims": [dims[0],dims[1],dims[2],1], "samples": samples}),
            );
        };
    capture("inp_pos_emb".into(), &job.x, [E, job.rows, 1], [1, E, 0], 0);
    for layer in 0..27 {
        let b = &vision.blocks[layer];
        // Observe the real normalization separately. step repeats it with the
        // same input; this observation cannot affect the residual stream.
        let cmd = device.begin().unwrap();
        vision.ln(&cmd, &job.x, &b.ln1, &b.ln1b, &job.stage, job.rows);
        cmd.finish().unwrap();
        capture(
            format!("ln1-{layer}"),
            &job.stage,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
        if layer == 0 {
            let cmd = device.begin().unwrap();
            cmd.dispatch(
                "vis_bmm64",
                &[&b.qkv.buffer, &job.stage, &job.qkv, &b.qkvb.buffer],
                &[E as u32, (E * 3) as u32, job.rows as u32, 1],
                [(E * 3).div_ceil(64), job.rows.div_ceil(64), 1],
                128,
            );
            cmd.finish().unwrap();
            capture(
                "Qcur-strict-0".into(),
                &job.qkv,
                [72, 16, job.rows],
                [1, 72, E * 3],
                0,
            );
        }
        // Observe the attention boundary before the production step reuses
        // its scratch for LN2. All arithmetic is the native GPU path; this
        // repeated observation leaves the residual stream untouched.
        let cmd = device.begin().unwrap();
        Vision::mm::<true, false>(&cmd, &b.qkv, &job.stage, &job.qkv, &b.qkvb, job.rows, 1);
        cmd.dispatch(
            "vis_qkv",
            &[&job.qkv, &job.xy, &job.q, &job.k, &job.v],
            &[job.rows as u32],
            [((job.rows + 64) * 1280).div_ceil(256), 1, 1],
            256,
        );
        cmd.dispatch(
            "vis_attention",
            &[&job.q, &job.k, &job.v, &job.attn, &job.tiles],
            &[0],
            [16, job.tile_count, 1],
            64,
        );
        cmd.finish().unwrap();
        capture(
            format!("kqv_out-{layer}"),
            &job.attn,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
        assert!(vision.step(&device, &mut job).unwrap().is_none());
        capture(
            format!("ffn_inp_normed-{layer}"),
            &job.attn,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
        for (name, offset) in [("Qcur", 0), ("Kcur", E), ("Vcur", E * 2)] {
            capture(
                format!("{name}-{layer}"),
                &job.qkv,
                [72, 16, job.rows],
                [1, 72, E * 3],
                offset,
            );
        }
        capture(
            format!("layer_out-{layer}"),
            &job.x,
            [E, job.rows, 1],
            [1, E, 0],
            0,
        );
    }
    let outputs = vision.step(&device, &mut job).unwrap().unwrap();
    if let Some(path) = std::env::var_os("PADDOCK_METAL_EMBEDDING_CAPTURE") {
        use std::io::Write;
        // Serialize completed GPU output, never a host inference result.
        // create_new prevents an accidental overwrite of an earlier capture.
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&64i32.to_le_bytes()).unwrap();
        file.write_all(&5120i32.to_le_bytes()).unwrap();
        let values = unsafe { outputs[0].embd.read_f32(0, 64 * 5120) };
        file.write_all(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    capture(
        "merger_output".into(),
        &outputs[0].embd,
        [5120, 64, 1],
        [1, 5120, 0],
        0,
    );
    eprintln!("VISION_TENSORS {}", serde_json::Value::Object(captures));
}

/// Encoder-only cost curve; deliberately separate from HTTP TTFT and from
/// tensor-capture instrumentation. Every repetition starts a cold GPU job.
#[test]
#[ignore = "requires canonical mmproj and an M5; GPU cost diagnostic"]
fn vision_encoder_cost_curve() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").expect("mmproj path");
    let device = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&device, Path::new(&path), 5120).unwrap();
    for side in [256, 768, 1024] {
        let rgb: Vec<u8> = (0..side * side * 3).map(|i| (i * 13 % 251) as u8).collect();
        for repeat in 0..3 {
            let start = std::time::Instant::now();
            let mut job = vision.start(&device, &[(&rgb, side, side)]).unwrap();
            while vision
                .step_blocks(&device, &mut job, 27, true)
                .unwrap()
                .is_none()
            {}
            eprintln!(
                "VISION_COST {}",
                serde_json::json!({"side":side,"repeat":repeat,
                "gpu_ms":job.gpu_seconds*1000.0,"wall_ms":start.elapsed().as_secs_f64()*1000.0})
            );
        }
    }
}
