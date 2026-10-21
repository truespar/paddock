use super::*;

fn previous(logits: &mut [f32], legal: &mut dyn FnMut(u32) -> bool) -> Option<u32> {
    loop {
        let best = argmax(logits);
        if logits[best as usize] == f32::NEG_INFINITY {
            return None;
        }
        if legal(best) {
            return Some(best);
        }
        logits[best as usize] = f32::NEG_INFINITY;
    }
}

#[test]
fn constrained_heap_preserves_picks_visits_and_masks() {
    let mut sampler = Sampler::new(SamplingParams::default());
    for length in [1, 2, 3, 4, 5, 6, 17, 65, 257] {
        for variant in 0..8 {
            let row: Vec<_> = (0..length)
                .map(|i| match (i + variant) % 11 {
                    0 => f32::NEG_INFINITY,
                    1 => f32::INFINITY,
                    2 => f32::NAN,
                    3 => -0.0,
                    4 => 0.0,
                    _ => ((i * 37 + variant * 13) % 7) as f32 - 3.0,
                })
                .collect();
            // Every possible legal ID, then no legal token. Reuse one sampler
            // across changing vocabulary sizes to exercise retained scratch.
            for target in 0..=length {
                let mut a = row.clone();
                let mut b = row.clone();
                let mut visits_a = Vec::new();
                let mut visits_b = Vec::new();
                let expected = previous(&mut a, &mut |id| {
                    visits_a.push(id);
                    id as usize == target
                });
                let actual = sampler.sample_constrained(&mut b, &[], &mut |id| {
                    visits_b.push(id);
                    id as usize == target
                });
                assert_eq!(actual, expected);
                assert_eq!(visits_b, visits_a);
                assert_eq!(
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn constrained_heap_preserves_stochastic_fallback_rng() {
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 1,
        seed: 148,
        ..Default::default()
    };
    let mut sampler = Sampler::new(params.clone());
    let mut control = Sampler::new(params);
    for length in [17, 257, 33] {
        let mut row: Vec<_> = (0..length).map(|i| -(i as f32)).collect();
        let mut expected = row.clone();
        let token = sampler.sample_constrained(&mut row, &[], &mut |id| id == length - 1);
        let _ = control.next_uniform();
        assert_eq!(token, previous(&mut expected, &mut |id| id == length - 1));
        assert_eq!(row, expected);
        assert_eq!(sampler.next_uniform(), control.next_uniform());
    }
}

#[test]
fn constrained_heap_leaves_common_path_allocation_free() {
    let mut sampler = Sampler::new(SamplingParams::default());
    let mut row = [0.0, 2.0, 3.0, 1.0];
    assert_eq!(
        sampler.sample_constrained(&mut row, &[], &mut |_| true),
        Some(2)
    );
    assert_eq!(sampler.scratch.capacity(), 0);
    assert_eq!(sampler.highest_legal(&mut [], &mut |_| false), None);
}

#[test]
#[ignore = "CPU-only constrained-sampling cost diagnostic; run apart from GPU serving benchmarks"]
fn constrained_heap_cost() {
    for vocabulary in [248_320, 262_144] {
        // These are Bonsai/Gemma vocabulary widths, not captured model logits.
        // A legal token below 8191 rejected candidates models a hard grammar
        // boundary without paying the old quadratic scan over the entire tail.
        let row: Vec<_> = (0..vocabulary)
            .map(|i| {
                if i < 8192 {
                    -(i as f32)
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        for round in 0..4 {
            let mut sampler = Sampler::new(SamplingParams::default());
            let mut times = [0.0; 2];
            for at in [round % 2, 1 - round % 2] {
                let mut logits = row.clone();
                let start = std::time::Instant::now();
                let token = if at == 0 {
                    previous(&mut logits, &mut |id| id == 8191)
                } else {
                    sampler.sample_constrained(&mut logits, &[], &mut |id| id == 8191)
                };
                times[at] = start.elapsed().as_secs_f64();
                assert_eq!(token, Some(8191));
            }
            eprintln!(
                "CONSTRAINED_COST vocab={vocabulary} round={round} baseline_s={} heap_s={}",
                times[0], times[1]
            );
        }
    }
}
