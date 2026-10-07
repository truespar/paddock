//! Matched-weight GPU microdiagnostic. Baseline is the production stable
//! vector contraction, not a CPU oracle. Matrix differences are reported,
//! never silently accepted as parity. Timings are warm-cache, not serving.
use crate::device::{Buffer, Commands, MetalDevice};
use crate::{affine, weights::Weight};
use paddock_models::safetensors::ShardedSafetensors;
use std::path::Path;

fn dispatch(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, m: usize, route: usize) {
    let r = m.div_ceil(m.div_ceil(5));
    match route {
        0 => cmd.dispatch(
            &format!("mlx_affine_stable{r}"),
            &[&w.buffer, &w.buffer, &w.buffer, x, y, y, y],
            &[w.k as u32, w.n as u32, 0, 0, m as u32],
            [w.n.div_ceil(8), m.div_ceil(r), 1],
            64,
        ),
        1..=3 => {
            let (r, c) = if route == 1 { (r, 2) } else { (8, route - 1) };
            cmd.dispatch(
                &format!("mlx_few_r{r}c{c}"),
                &[&w.buffer, x, y],
                &[w.k as u32, w.n as u32, m as u32],
                [w.n.div_ceil(8 * c), m.div_ceil(r), 1],
                64,
            );
        }
        6..=7 => {
            let lanes = if route == 6 { 4 } else { 2 };
            cmd.dispatch(
                &format!("mlx_few_fold{r}l{lanes}"),
                &[&w.buffer, x, y],
                &[w.k as u32, w.n as u32, m as u32],
                [w.n.div_ceil(64 / lanes), m.div_ceil(r), 1],
                64,
            );
        }
        8 => cmd.dispatch(
            &format!("mlx_few_packed{r}"),
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, m as u32],
            [w.n.div_ceil(8), m.div_ceil(r), 1],
            64,
        ),
        _ => cmd.dispatch(
            if route == 4 {
                "mlx_few_mma"
            } else {
                "mlx_few_register"
            },
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, m as u32],
            [w.n.div_ceil(8), m.div_ceil(8), 1],
            if route == 4 { 128 } else { 256 },
        ),
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; exact occupancy/shape experiment"]
fn affine_few_shape_probe() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let source = ShardedSafetensors::open_dir(Path::new(&path)).unwrap();
    let d = MetalDevice::new(None).unwrap();
    for (name, k, n) in [
        ("model.layers.0.mlp.gate_proj", 5120usize, 17408usize),
        ("model.layers.0.mlp.down_proj", 17408, 5120),
        ("model.layers.0.linear_attn.out_proj", 6144, 5120),
        ("model.layers.0.linear_attn.in_proj_qkv", 5120, 10240),
        ("lm_head", 5120, 248320),
    ] {
        let w = affine::load(&d, &source, &format!("language_model.{name}.weight"), k, n).unwrap();
        let x = d
            .upload(
                &(0..16 * k)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 37 % 1999) as f32 - 999.) / 997.)
                            .to_bits()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let y = d.alloc((16 * n + 32) * 4).unwrap();
        for m in [2usize, 3, 4, 5, 8, 12] {
            let r = m.div_ceil(m.div_ceil(5));
            let routes = [(8, 0), (4, 0), (16, 0), (32, 0), (8, k), (8, 1), (8, 8)];
            let run = |cmd: &Commands<'_>, route: usize| {
                let (c, fixed) = routes[route];
                if route == 0 {
                    dispatch(cmd, &w, &x, &y, m, if r == 2 { 0 } else { 8 });
                } else if route >= 5 {
                    let rows = if route == 6 { 8 } else { r };
                    cmd.dispatch(
                        &if route == 6 {
                            "mlx_few_packed8".into()
                        } else {
                            format!("mlx_few_shared{r}")
                        },
                        &[&w.buffer, &x, &y],
                        &[k as u32, n as u32, m as u32],
                        [n.div_ceil(8), m.div_ceil(rows), 1],
                        64,
                    );
                } else {
                    cmd.dispatch(
                        &format!("mlx_few_shape_r{r}c{c}k{fixed}"),
                        &[&w.buffer, &x, &y],
                        &[k as u32, n as u32, m as u32],
                        [n.div_ceil(c), m.div_ceil(r), 1],
                        c * 8,
                    );
                }
            };
            let mut expected = Vec::new();
            for route in 0..routes.len() {
                unsafe {
                    y.write_u32(&vec![u32::MAX; m * n + 32]);
                }
                let cmd = d.begin().unwrap();
                run(&cmd, route);
                cmd.finish().unwrap();
                let got = unsafe { y.read_u32(m * n + 32) };
                assert!(got[..m * n].iter().all(|v| f32::from_bits(*v).is_finite()));
                assert!(got[m * n..].iter().all(|v| *v == u32::MAX));
                if route == 0 {
                    expected = got;
                } else {
                    let differences = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
                    assert_eq!(differences, 0, "{name} m={m} route={route}");
                }
            }
            let mut times = vec![Vec::new(); routes.len()];
            for round in 0..9 {
                for shift in 0..routes.len() {
                    let route = (shift + round) % routes.len();
                    let cmd = d.begin().unwrap();
                    for _ in 0..8 {
                        run(&cmd, route);
                    }
                    let us = cmd.finish().unwrap() * 1e6 / 8.;
                    if round >= 2 {
                        times[route].push(us);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "FEW_SHAPE {}",
                serde_json::json!({"weight":name,"k":k,"n":n,"rows":m,
                "routes":routes,"median_us":times.iter().map(|t|t[3]).collect::<Vec<_>>(),"samples_us":times})
            );
        }
    }
}

#[test]
#[ignore = "requires PADDOCK_METAL_MLX_MODEL; paired real-weight GPU timings"]
fn affine_few_row_probe() {
    let path = std::env::var("PADDOCK_METAL_MLX_MODEL").unwrap();
    let source = ShardedSafetensors::open_dir(Path::new(&path)).unwrap();
    let d = MetalDevice::new(None).unwrap();
    for (name, k, n) in [
        ("model.layers.0.mlp.gate_proj", 5120usize, 17408usize),
        ("model.layers.0.mlp.down_proj", 17408, 5120),
        ("model.layers.0.linear_attn.out_proj", 6144, 5120),
        ("model.layers.0.linear_attn.in_proj_a", 5120, 48),
        ("lm_head", 5120, 248320),
    ] {
        let name = format!("language_model.{name}.weight");
        let w = affine::load(&d, &source, &name, k, n).unwrap();
        let values: Vec<u8> = (0..16 * k)
            .flat_map(|i| {
                half::bf16::from_f32(
                    (((i as u32).wrapping_mul(2654435761) >> 16) % 1999) as f32 / 997. - 1.,
                )
                .to_bits()
                .to_le_bytes()
            })
            .collect();
        let x = d.upload(&values).unwrap();
        let y = d.alloc((16 * n + 32) * 4).unwrap();
        for m in [2usize, 3, 4, 5, 6, 7, 8, 12, 16] {
            let mut times: Vec<Vec<f64>> = vec![Vec::new(); 9];
            let mut differences = vec![0; 9];
            let mut reference = Vec::new();
            for (route, difference) in differences.iter_mut().enumerate() {
                unsafe {
                    y.write_u32(&vec![u32::MAX; m * n + 32]);
                }
                let cmd = d.begin().unwrap();
                dispatch(&cmd, &w, &x, &y, m, route);
                cmd.finish().unwrap();
                let got = unsafe { y.read_f32(0, m * n + 32) };
                assert!(got[..m * n].iter().all(|v| v.is_finite()));
                assert!(got[m * n..].iter().all(|v| v.to_bits() == u32::MAX));
                if route == 0 {
                    reference = got[..m * n].to_vec();
                }
                *difference = got[..m * n]
                    .iter()
                    .zip(&reference)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                if !(4..6).contains(&route) {
                    assert_eq!(*difference, 0, "{name} m={m} route={route}");
                }
            }
            // Rotate order every round; discard two complete warm-up rounds.
            for round in 0..9 {
                for i in 0..9 {
                    let route = (i + round) % 9;
                    let cmd = d.begin().unwrap();
                    for _ in 0..4 {
                        dispatch(&cmd, &w, &x, &y, m, route);
                    }
                    let us = cmd.finish().unwrap() * 1e6 / 4.;
                    if round >= 2 {
                        times[route].push(us);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "FEW_ROWS {}",
                serde_json::json!({"weight":name,"k":k,"n":n,"rows":m,
                "routes":["baseline","paired","wide8","paired8","mma","register","fold4","fold2","packed"],"different":differences,
                "elements":m*n,"median_us":times.iter().map(|t| t[3]).collect::<Vec<_>>(),"samples_us":times})
            );
        }
    }
}

#[test]
fn packed_affine_fused_rows_preserve_bits_offsets_and_guards() {
    let d = MetalDevice::new(None).unwrap();
    for k in [5120usize, 6144, 17408] {
        let weights: Vec<_> = [1024usize, 1041, 1536]
            .into_iter()
            .enumerate()
            .map(|(plane, n)| {
                let mut data: Vec<u8> = (0..n * k / 8)
                    .flat_map(|i| {
                        ((i + plane * 917) as u32)
                            .wrapping_mul(2654435761)
                            .to_le_bytes()
                    })
                    .collect();
                for bias in [false, true] {
                    data.extend((0..n * k / 64).flat_map(|i| {
                        let value = ((i * 31 % 251) as f32 - 125.) / 9973.;
                        half::bf16::from_f32(if bias { value * 7.25 } else { value })
                            .to_bits()
                            .to_le_bytes()
                    }));
                }
                Weight {
                    buffer: d.upload(&data).unwrap(),
                    ty: affine::AFFINE4,
                    k,
                    n,
                }
            })
            .collect();
        let rows = 32;
        let x = d
            .upload(
                &(0..rows * k)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 37 % 1999) as f32 - 999.) / 113.)
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let outputs: Vec<_> = weights
            .iter()
            .map(|w| d.alloc((rows * w.n + 32) * 4).unwrap())
            .collect();
        let workspace = d
            .alloc(affine::workspace_bytes(k, 1536, rows) + 128)
            .unwrap();
        for count in [1usize, 2, 3, 4, 5, 6, 7, 8, 12, 16, 31] {
            for planes in [1usize, 2, 3] {
                // Leading row stays on R1, exercising all per-plane offsets.
                let spans = [(0, 1, 1), (1, count, 1)];
                let mut expected = Vec::new();
                for candidate in [false, true] {
                    affine::BASELINE_PACKED_FOR_TEST.with(|v| v.set(!candidate));
                    for y in &outputs {
                        unsafe {
                            y.write_u32(&vec![u32::MAX; y.len() / 4]);
                        }
                    }
                    let cmd = d.begin().unwrap();
                    affine::project_stable(
                        &cmd,
                        &weights
                            .iter()
                            .zip(&outputs)
                            .take(planes)
                            .collect::<Vec<_>>(),
                        &x,
                        count + 1,
                        &workspace,
                        &spans,
                    );
                    cmd.finish().unwrap();
                    let got: Vec<Vec<u32>> = outputs
                        .iter()
                        .map(|y| unsafe { y.read_u32(y.len() / 4) })
                        .collect();
                    for i in 0..planes {
                        assert!(
                            got[i][..(count + 1) * weights[i].n]
                                .iter()
                                .all(|b| f32::from_bits(*b).is_finite())
                        );
                        assert!(
                            got[i][(count + 1) * weights[i].n..]
                                .iter()
                                .all(|b| *b == u32::MAX)
                        );
                    }
                    if candidate {
                        assert_eq!(got, expected, "k={k}, count={count}, planes={planes}");
                    } else {
                        expected = got;
                    }
                }
            }
        }
    }
    affine::BASELINE_PACKED_FOR_TEST.with(|v| v.set(false));
}
