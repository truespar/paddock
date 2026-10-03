//! Same-weight Clef benchmark, using the independent oracle's encoded inputs.
//! cargo run -p paddock-metal --release --example clef_bench -- MODEL CASE_JSON [BATCH] [REPEATS]
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use paddock_engine::clef_decision::{ClefBackend, ClefQuestion, ClefRequest};
    use std::{path::Path, time::Instant};
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        return Err("expected MODEL CASE_JSON [BATCH] [REPEATS]".into());
    }
    let batch: usize = args.get(3).map_or(Ok(1), |x| x.parse())?;
    let repeats: usize = args.get(4).map_or(Ok(5), |x| x.parse())?;
    if !(1..=256).contains(&batch) || repeats == 0 {
        return Err("invalid batch/repeats".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let ids = value["input_ids"]
        .as_array()
        .ok_or("input_ids")?
        .iter()
        .map(|x| x.as_u64().expect("oracle token ID") as u32)
        .collect::<Vec<_>>();
    let questions = value["questions"]
        .as_array()
        .ok_or("questions")?
        .iter()
        .map(|v| {
            let span = |s: &serde_json::Value| {
                (
                    s[0].as_u64().expect("oracle span start") as usize,
                    s[1].as_u64().expect("oracle span end") as usize,
                )
            };
            ClefQuestion {
                qtype: v["type"].as_u64().expect("oracle question type") as u32,
                span: span(&v["span"]),
                options: v["option_spans"]
                    .as_array()
                    .expect("oracle option spans")
                    .iter()
                    .map(span)
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let mut model = paddock_metal::Clef::load(Path::new(&args[1]), None)?;
    let reqs = (0..batch)
        .map(|_| ClefRequest {
            ids: &ids,
            questions: &questions,
            images: &[],
        })
        .collect::<Vec<_>>();
    let expected = model.forward(&reqs)?;
    let mut times = Vec::new();
    let mut thermal = Vec::new();
    for _ in 0..repeats {
        let before = objc2_foundation::NSProcessInfo::processInfo()
            .thermalState()
            .0;
        let start = Instant::now();
        let result = model.forward(&reqs)?;
        times.push(start.elapsed().as_secs_f64() * 1000.);
        thermal.push((
            before,
            objc2_foundation::NSProcessInfo::processInfo()
                .thermalState()
                .0,
        ));
        assert_eq!(result, expected);
    }
    println!(
        "{}",
        serde_json::json!({"case":value["id"],"rows_each":ids.len(),"batch":batch,"ms":times,"thermal_before_after":thermal,
        "weight_bytes":model.info().weight_bytes,"workspace_bytes":model.info().workspace_bytes,"logits":expected})
    );
    Ok(())
}
#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Clef Metal qualification requires macOS");
    std::process::exit(1);
}
