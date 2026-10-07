//! Same-ID backend API rung. It includes native scheduling, host sampling,
//! and token readback, but not HTTP/tokenization. Never headline as serving.
use super::*;
use paddock_engine::generator::Generator;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

fn sample(logits: &[f32]) -> u32 {
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|x| x.is_finite()));
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("vocab nonempty")
        .0 as u32
}

#[test]
#[ignore = "requires PADDOCK_KOLIBRI_MLX, PADDOCK_KOLIBRI_CASES, PADDOCK_KOLIBRI_REPORT"]
fn performance_rung() {
    assert!(
        std::env::var_os("PADDOCK_METAL_PROFILE").is_none(),
        "counter instrumentation invalidates timings"
    );
    assert!(
        std::env::var_os("PADDOCK_KOLIBRI_TRACE").is_none(),
        "snapshot fences invalidate timings"
    );
    let root = std::env::var("PADDOCK_KOLIBRI_MLX").unwrap();
    let cases_path = std::env::var("PADDOCK_KOLIBRI_CASES").unwrap();
    let out_path = std::env::var("PADDOCK_KOLIBRI_REPORT").unwrap();
    assert!(
        !Path::new(&out_path).exists(),
        "never overwrite a previous rung"
    );
    let cases: Value = serde_json::from_slice(&std::fs::read(&cases_path).unwrap()).unwrap();
    let cap = cases["max_tokens"].as_u64().unwrap() as usize;
    assert!(cap > 0);
    let context = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|c| c["prompts"].as_array().unwrap())
        .map(|p| p.as_array().unwrap().len() + cap)
        .max()
        .unwrap()
        .max(2048);
    let rounds = std::env::var("PADDOCK_KOLIBRI_ROUNDS")
        .ok()
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(3);
    let tokenizer = paddock_tokenizer::GgufTokenizer::from_hf_dir(Path::new(&root)).unwrap();
    let start = Instant::now();
    let mut model = Kolibri::load(Path::new(&root), context, 4, None).unwrap();
    let mut report = json!({"completed":false,"scope":"backend API, not HTTP; cold KV every batch; greedy fixed length; no EOS stop; no speculation",
        "load_s":start.elapsed().as_secs_f64(),"cases_file":cases_path,"model":root,
        "weight_bytes":model.weight_bytes,"kv_bytes":model.kv_bytes,"allocated_bytes":model.device.allocated_bytes(),
        "memory_scope":"Metal allocation ledger, not peak process footprint","context":context,"batches":[]});
    let save =
        |v: &Value| std::fs::write(&out_path, serde_json::to_vec_pretty(v).unwrap()).unwrap();
    save(&report);
    for round in 0..=rounds {
        for case in cases["cases"].as_array().unwrap() {
            model.reset();
            while model.radix.evict_lru(&mut model.pool).is_some() {}
            assert_eq!(model.radix.cached_blocks(), 0);
            let prompts: Vec<Vec<u32>> = serde_json::from_value(case["prompts"].clone()).unwrap();
            let mut outputs = vec![Vec::<u32>::new(); prompts.len()];
            let mut times = vec![Vec::<f64>::new(); prompts.len()];
            let mut gpu_s = 0.;
            let start = Instant::now();
            for (slot, ids) in prompts.iter().enumerate() {
                model.prefill_begin(slot, ids.clone()).unwrap();
            }
            while outputs.iter().any(|ids| ids.len() < cap) {
                let decode = outputs
                    .iter()
                    .enumerate()
                    .filter(|(_, ids)| !ids.is_empty() && ids.len() < cap)
                    .map(|(slot, ids)| {
                        (
                            slot,
                            *ids.last().unwrap(),
                            (prompts[slot].len() + ids.len() - 1) as u32,
                        )
                    })
                    .collect::<Vec<_>>();
                let (logits, done) = model.forward_mixed(&decode, CHUNK).unwrap();
                gpu_s += model.last_gpu_seconds;
                for (row, &(slot, _, _)) in decode.iter().enumerate() {
                    outputs[slot].push(sample(&logits[row * VOCAB..(row + 1) * VOCAB]));
                    times[slot].push(start.elapsed().as_secs_f64());
                }
                for (slot, logits, _) in done {
                    outputs[slot].push(sample(&logits));
                    times[slot].push(start.elapsed().as_secs_f64());
                }
            }
            let wall = start.elapsed().as_secs_f64();
            let result = json!({"case":case["name"],"round":round,"warmup":round==0,"wall_s":wall,
                "output_tps":(cap*prompts.len()) as f64/wall,"gpu_s":gpu_s,
                "outputs":outputs.iter().enumerate().map(|(i,ids)|json!({"ids":ids,"times_s":times[i],"text":tokenizer.decode(ids,false).unwrap()})).collect::<Vec<_>>()});
            eprintln!(
                "KOLIBRI_RUNG {}",
                json!({"case":case["name"],"round":round,"wall_s":wall,"gpu_s":gpu_s,"output_tps":(cap*prompts.len()) as f64/wall})
            );
            report["batches"].as_array_mut().unwrap().push(result);
            save(&report);
        }
    }
    report["completed"] = json!(true);
    save(&report);
}
