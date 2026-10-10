//! Reference-input replay isolates local arithmetic from accumulated tower drift.
use super::*;

fn fingerprint(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .fold(0xcbf29ce484222325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        })
}

#[test]
fn mlx_vision_norm_matches_reference_variance_contract() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    // Fingerprints are pinned to the M5 reference, not a claim about M1–M4.
    if !d.tensor_accelerated() {
        return;
    }
    for (n, expected) in [(768, 0xdec360a9e1bc6924), (1024, 0x4f68250e69b77e49)] {
        let bf = |x| half::bf16::from_f32(x).to_f32();
        let x: Vec<_> = (0..513 * n)
            .map(|i| bf(((i * 73 % 4093) as f32 - 2046.) / 256.))
            .collect();
        let w: Vec<_> = (0..n)
            .map(|i| bf(((i * 37 % 127) as f32 - 63.) / 64.))
            .collect();
        let b: Vec<_> = (0..n)
            .map(|i| bf(((i * 11 % 31) as f32 - 15.) / 128.))
            .collect();
        let upload = |v: &[f32]| {
            d.upload(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
                .unwrap()
        };
        let (x, w, b) = (upload(&x), upload(&w), upload(&b));
        let out = d.upload(&vec![0xff; (513 * n + 256) * 4]).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "qmlx_vis_norm",
            &[&x, &w, &b, &out],
            &[n as u32, 1e-6f32.to_bits()],
            [513, 1, 1],
            n.div_ceil(256) * 32,
        );
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, 513 * n + 256) };
        assert_eq!(fingerprint(&actual[..513 * n]), expected, "width={n}");
        assert!(actual[513 * n..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn mlx_patch_projection_matches_conv3d_reduction_and_bounds() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    if !d.tensor_accelerated() {
        return;
    }
    // Expected values are 64-bit FNV-1a fingerprints of MLX v0.32.3's output on
    // the same synthetic inputs, over every F32-expanded byte; no checkpoint fixtures.
    for (rows, expected) in [(33, 0xd8182c00681d9a89), (65, 0x7c9529e77d04cd7f)] {
        let x: Vec<_> = (0..rows * 1536)
            .flat_map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 128.).to_le_bytes())
            .collect();
        let w: Vec<_> = (0..96 * 1536)
            .flat_map(|i| {
                half::bf16::from_f32(((i * 17 % 65521) as f32 - 32760.) / 32768.).to_le_bytes()
            })
            .collect();
        let x = d.upload(&x).unwrap();
        let w = d.upload(&w).unwrap();
        let out = d.upload(&vec![0xff; (rows * 96 + 256) * 4]).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "qmlx_vis_patch_mm",
            &[&w, &x, &out],
            &[96, rows as u32],
            [2, rows.div_ceil(64), 1],
            128,
        );
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, rows * 96 + 256) };
        let rounded: Vec<_> = actual[..rows * 96]
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect();
        assert_eq!(fingerprint(&rounded), expected, "rows={rows}");
        assert!(actual[rows * 96..].iter().all(|v| v.is_nan()));
    }
}

#[test]
fn mlx_vision_position_preserves_linspace_operation_order() {
    let d = MetalDevice::new(Some(32 << 20)).unwrap();
    let bf = |x| half::bf16::from_f32(x).to_f32();
    let e = 32;
    let pos: Vec<_> = (0..48 * 48 * e)
        .map(|i| bf(((i * 37 % 1013) as f32 - 506.) / 256.))
        .collect();
    let pos_gpu = d
        .upload(&pos.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    let bias = d.upload(&vec![0; e * 4]).unwrap();
    for (pw, ph) in [(90, 128), (128, 90), (16, 16), (2, 34)] {
        let rows = pw * ph;
        let out = d.upload(&vec![0; rows * e * 4]).unwrap();
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            "qmlx_vis_position",
            &[&out, &out, &bias, &pos_gpu],
            &[pw as u32, ph as u32, 0, e as u32, 48],
            [(rows * e).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        let actual = unsafe { out.read_f32(0, rows * e) };
        for row in 0..rows {
            let y = (row / 4 / (pw / 2)) * 2 + (row % 4) / 2;
            let x = (row / 4 % (pw / 2)) * 2 + row % 2;
            let sy = (y as f32 / (ph - 1) as f32) * 47.;
            let sx = (x as f32 / (pw - 1) as f32) * 47.;
            let (y0, x0) = (sy as usize, sx as usize);
            let (y1, x1) = ((y0 + 1).min(47), (x0 + 1).min(47));
            let (dy, dx) = (sy - y0 as f32, sx - x0 as f32);
            for col in 0..e {
                let a = bf(pos[(y0 * 48 + x0) * e + col] * bf((1. - dy) * (1. - dx)));
                let b = bf(pos[(y0 * 48 + x1) * e + col] * bf((1. - dy) * dx));
                let c = bf(pos[(y1 * 48 + x0) * e + col] * bf(dy * (1. - dx)));
                let dd = bf(pos[(y1 * 48 + x1) * e + col] * bf(dy * dx));
                assert_eq!(
                    actual[row * e + col],
                    bf(bf(bf(a + b) + c) + dd),
                    "grid={pw}x{ph} row={row} col={col}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires --all-blocks/--blocks capture from metal-lighton-vision.py"]
fn lighton_mlx_block_contracts() {
    let path = std::env::var_os("PADDOCK_METAL_MMPROJ").unwrap();
    let captures = std::env::var_os("PADDOCK_METAL_LIGHTON_SEAMS").unwrap();
    let captures = Path::new(&captures);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(captures.join("manifest.json")).unwrap()).unwrap();
    let cfg = paddock_models::mlx::QwenConfig::read(Path::new(&path)).unwrap();
    let d = MetalDevice::new(None).unwrap();
    let vision = Vision::load(&d, Path::new(&path), cfg.width).unwrap();
    assert!(vision.mlx_bf16);
    let read = |name: &str| std::fs::read(captures.join(format!("{name}.f32"))).unwrap();
    let input = |name: &str, bf16: bool| {
        let raw = read(name);
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
    let check = |name: &str, buffer: &Buffer| {
        let raw = read(name);
        let expected: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let actual = unsafe { buffer.read_f32(0, expected.len()) };
        let mut differences = 0;
        let mut max = 0f32;
        let mut squared = 0f64;
        let mut norm = 0f64;
        for (a, b) in actual.iter().zip(&expected) {
            assert!(a.is_finite() && b.is_finite());
            differences += usize::from(a.to_bits() != b.to_bits());
            max = max.max((a - b).abs());
            squared += f64::from(a - b).powi(2);
            norm += f64::from(*b).powi(2);
        }
        eprintln!(
            "LIGHTON_CONTRACT {name} different={differences}/{} max={max} relative_l2={}",
            expected.len(),
            (squared / norm.max(1e-30)).sqrt()
        );
        differences
    };
    let rgb = std::fs::read(captures.join("image.rgb")).unwrap();
    let w = manifest["image_size"][0].as_u64().unwrap() as usize;
    let h = manifest["image_size"][1].as_u64().unwrap() as usize;
    let job = vision.start(&d, &[(&rgb, w, h)]).unwrap();
    let rows = job.rows;
    assert_eq!(
        rows,
        manifest["grid"][0]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .product::<usize>()
    );
    let mut differences = check("position", &job.x);
    let pixels = d.alloc(rows * 1536 * 4).unwrap();
    let bad = d.upload(&[0u8; 4]).unwrap();
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "vis_cast",
        &[&job.stage, &pixels, &bad],
        &[(rows * 1536) as u32, 30, 0],
        [(rows * 1536).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    differences += check("pixels", &pixels);
    let zero_pos = d
        .upload(&vec![0; 2304 * vision.geometry.width * 4])
        .unwrap();
    let zero_bias = d.upload(&vec![0; vision.geometry.width * 4]).unwrap();
    let pw = manifest["grid"][0][2].as_u64().unwrap() as u32;
    let ph = manifest["grid"][0][1].as_u64().unwrap() as u32;
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "qmlx_vis_patch_mm",
        &[&vision.patch0.buffer, &job.stage, &job.attn],
        &[vision.geometry.width as u32, rows as u32],
        [vision.geometry.width.div_ceil(64), rows.div_ceil(64), 1],
        128,
    );
    cmd.dispatch(
        "qmlx_vis_position",
        &[&job.attn, &job.attn, &vision.bias.buffer, &zero_pos],
        &[pw, ph, 0, vision.geometry.width as u32, 48],
        [(rows * vision.geometry.width).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    differences += check("patch", &job.attn);
    let patch = input("patch", false);
    let cmd = d.begin().unwrap();
    cmd.dispatch(
        "qmlx_vis_position",
        &[&patch, &patch, &zero_bias, &vision.pos.buffer],
        &[pw, ph, 0, vision.geometry.width as u32, 48],
        [(rows * vision.geometry.width).div_ceil(256), 1, 1],
        256,
    );
    cmd.finish().unwrap();
    differences += check("position", &patch);
    let candidate = std::env::var_os("PADDOCK_METAL_ATTENTION_CANDIDATE").is_some();
    let mut walk = vision.start(&d, &[(&rgb, w, h)]).unwrap();
    let mut walk_differences = 0;
    for i in 0..vision.blocks.len() {
        assert!(vision.step(&d, &mut walk).unwrap().is_none());
        let name = format!("block-{i}");
        if !manifest["seams"][&name].is_null() {
            eprintln!("LIGHTON_WALK layer={i}");
            walk_differences += check(&name, &walk.x);
        }
    }
    let output = vision.step(&d, &mut walk).unwrap().unwrap();
    eprintln!("LIGHTON_WALK merger");
    walk_differences += check("merger", &output[0].embd);
    for (i, b) in vision.blocks.iter().enumerate() {
        let prefix = format!("layer-{i}-");
        if manifest["seams"][format!("{prefix}input")].is_null() {
            continue;
        }
        let name = |s: &str| format!("{prefix}{s}");
        let run_norm = |source: &str, target: &str, weight: &Weight, bias: &Weight| {
            let x = input(&name(source), false);
            let cmd = d.begin().unwrap();
            vision.ln(&cmd, &x, weight, bias, &job.stage, rows);
            cmd.finish().unwrap();
            check(&name(target), &job.stage)
        };
        differences += run_norm("input", "norm", &b.ln1, &b.ln1b);
        differences += run_norm("residual", "norm2", &b.ln2, &b.ln2b);
        let qkv = input(&name("qkv"), false);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            if vision.geometry.heads == 12 {
                "qmlx_vis_qkv_12"
            } else {
                "qmlx_vis_qkv_16"
            },
            &[&qkv, &job.xy, &job.q, &job.k, &job.v],
            &[rows as u32],
            [((rows + 64) * vision.geometry.width).div_ceil(256), 1, 1],
            256,
        );
        cmd.finish().unwrap();
        for (target, buffer) in [("query", &job.q), ("key", &job.k), ("value", &job.v)] {
            let expanded = d.alloc(rows * vision.geometry.width * 4).unwrap();
            let bad = d.upload(&[0u8; 4]).unwrap();
            let cmd = d.begin().unwrap();
            cmd.dispatch(
                "vis_cast",
                &[buffer, &expanded, &bad],
                &[(rows * vision.geometry.width) as u32, 30, 0],
                [(rows * vision.geometry.width).div_ceil(256), 1, 1],
                256,
            );
            cmd.finish().unwrap();
            differences += check(&name(target), &expanded);
        }
        for (source, target, weight, bias, epilogue) in [
            ("norm", "qkv", &b.qkv, &b.qkvb, 1),
            ("norm2", "up", &b.up, &b.upb, 1),
            ("norm2", "gelu", &b.up, &b.upb, 3),
            ("gelu", "down", &b.down, &b.downb, 1),
        ] {
            let x = input(&name(source), false);
            let out = d.alloc(rows * weight.n * 4).unwrap();
            let cmd = d.begin().unwrap();
            vision.project(&cmd, weight, &x, &out, bias, rows, epilogue);
            cmd.finish().unwrap();
            differences += check(&name(target), &out);
        }
        let x = input(&name("attention"), false);
        let out = input(&name("input"), false);
        let cmd = d.begin().unwrap();
        vision.project(&cmd, &b.out, &x, &out, &b.outb, rows, 2);
        cmd.finish().unwrap();
        differences += check(&name("residual"), &out);
        let x = input(&name("gelu"), false);
        let out = input(&name("residual"), false);
        let cmd = d.begin().unwrap();
        vision.project(&cmd, &b.down, &x, &out, &b.downb, rows, 2);
        cmd.finish().unwrap();
        differences += check(&format!("block-{i}"), &out);
        let q = input(&name("query"), true);
        let k = input(&name("key"), true);
        let v = input(&name("value"), true);
        let cmd = d.begin().unwrap();
        cmd.dispatch(
            match (candidate, vision.geometry.heads) {
                (true, 12) => "qmlx_vis_attention_candidate_12",
                (true, _) => "qmlx_vis_attention_candidate_16",
                (false, 12) => "qmlx_vis_attention_12",
                (false, _) => "qmlx_vis_attention_16",
            },
            &[&q, &k, &v, &job.attn, &job.tiles],
            &[31],
            [vision.geometry.heads, job.tile_count, 1],
            128,
        );
        cmd.finish().unwrap();
        differences += check(&name("attention"), &job.attn);
    }
    let x = input(&format!("block-{}", vision.blocks.len() - 1), false);
    let cmd = d.begin().unwrap();
    vision.ln(&cmd, &x, &vision.post, &vision.postb, &job.attn, rows);
    cmd.finish().unwrap();
    differences += check("merger-norm", &job.attn);
    for (source, target, weight, bias, epilogue) in [
        ("merger-norm", "merger-up", &vision.mm0, &vision.mm0b, 1),
        ("merger-norm", "merger-gelu", &vision.mm0, &vision.mm0b, 4),
        ("merger-gelu", "merger", &vision.mm2, &vision.mm2b, 1),
    ] {
        let x = input(source, false);
        let out = d.alloc(rows / 4 * weight.n * 4).unwrap();
        let cmd = d.begin().unwrap();
        vision.project(&cmd, weight, &x, &out, bias, rows / 4, epilogue);
        cmd.finish().unwrap();
        differences += check(target, &out);
    }
    eprintln!(
        "LIGHTON_CONTRACT total_differences={differences} walk_differences={walk_differences} candidate_attention={candidate}"
    );
    if std::env::var_os("PADDOCK_METAL_REQUIRE_PARITY").is_some() {
        assert_eq!(differences, 0, "reference-input contracts differ");
        assert_eq!(walk_differences, 0, "complete tower differs");
    }
}
