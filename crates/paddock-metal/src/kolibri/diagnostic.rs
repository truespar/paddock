//! Test-only synchronized GPU snapshots. Never used for timings or inference.
use super::*;

pub(super) fn snapshot<'a>(
    cmd: Commands<'a>,
    device: &'a MetalDevice,
    label: &str,
    buffer: &Buffer,
    offset: usize,
    count: usize,
) -> Result<Commands<'a>> {
    let Ok(dir) = std::env::var("PADDOCK_KOLIBRI_TRACE") else {
        return Ok(cmd);
    };
    let gpu_s = cmd.finish()?;
    std::fs::write(
        std::path::Path::new(&dir).join(format!("{label}.gpu.json")),
        gpu_s.to_string(),
    )
    .map_err(|e| MetalError::Model(e.to_string()))?;
    // SAFETY: GPU writes completed above; caller supplies a valid buffer span.
    let values = unsafe { buffer.read_f32(offset, count) };
    std::fs::write(
        std::path::Path::new(&dir).join(format!("{label}.json")),
        serde_json::to_vec(&values).expect("finite GPU values"),
    )
    .map_err(|e| MetalError::Model(e.to_string()))?;
    device.begin()
}

#[test]
#[ignore = "requires PADDOCK_KOLIBRI_MLX and PADDOCK_KOLIBRI_CASES; optional TRACE/Metal counters"]
fn checkpoint_trace() {
    let root = std::env::var("PADDOCK_KOLIBRI_MLX").unwrap();
    if let Ok(dir) = std::env::var("PADDOCK_KOLIBRI_TRACE") {
        std::fs::create_dir(&dir).unwrap();
    }
    let mut model = Kolibri::load(std::path::Path::new(&root), 2048, 1, None).unwrap();
    // The same IDs as the external reference smoke, bypassing tokenizer policy.
    let cases: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("PADDOCK_KOLIBRI_CASES").unwrap()).unwrap(),
    )
    .unwrap();
    let tokens: Vec<u32> = serde_json::from_value(cases["cases"][0]["prompts"][0].clone()).unwrap();
    let rows = tokens
        .iter()
        .enumerate()
        .map(|(i, &t)| (0, t, i as u32))
        .collect::<Vec<_>>();
    model.execute(&rows, &[rows.len() - 1]).unwrap();
}
