//! The confidence census's writer (dev): one JSON line per speculative round,
//! with its width, verify rows and wall time, and for every slot the chain
//! drafted for its depth, accepted run and the drafter's log p(top1) at each
//! position (see the qwen35 `spec_conf.rs` docs for what it is for).
//! PADDOCK_SPEC_CONF_LOG=<path>, with PADDOCK_SPEC_CONF=1 arming the planes.

use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use crate::generator::{Generator, SpecAccepted};

fn sink() -> Option<&'static Mutex<std::fs::File>> {
    static F: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    F.get_or_init(|| {
        let path = paddock_models::dev_var_os!("PADDOCK_SPEC_CONF_LOG")?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(Mutex::new(file))
    })
    .as_ref()
}

/// Book a strip round: `warm[i]` marks the live entries the chain drafted
/// for (its rows, in live order), `reqs[i].2` their chunks, `strip[i]` the
/// accepted runs (a+1).
pub(super) fn strip_round(
    generator: &dyn Generator,
    warm: &[bool],
    reqs: &[(usize, usize, Vec<u32>)],
    strip: &[SpecAccepted],
    t0: std::time::Instant,
) {
    if sink().is_none() {
        return;
    }
    let accepted: Vec<usize> = (0..reqs.len())
        .map(|i| strip.get(i).map_or(0, |s| s.accepted.saturating_sub(1)))
        .collect();
    write(generator, warm, reqs, &accepted, t0);
}

/// Book a plans round from its flat picks: a slot's accepted run is its
/// drafts matching the picks before them (the replay's own rule).
pub(super) fn plans_round(
    generator: &dyn Generator,
    warm: &[bool],
    reqs: &[(usize, usize, Vec<u32>)],
    picks: &[u32],
    t0: std::time::Instant,
) {
    if sink().is_none() {
        return;
    }
    let mut base = 0usize;
    let accepted: Vec<usize> = reqs
        .iter()
        .map(|(_, _, chunk)| {
            let mut a = 0usize;
            while a + 1 < chunk.len() && picks.get(base + a) == Some(&chunk[a + 1]) {
                a += 1;
            }
            base += chunk.len();
            a
        })
        .collect();
    write(generator, warm, reqs, &accepted, t0);
}

fn write(
    generator: &dyn Generator,
    warm: &[bool],
    reqs: &[(usize, usize, Vec<u32>)],
    accepted: &[usize],
    t0: std::time::Instant,
) {
    let Some(file) = sink() else {
        return;
    };
    let us = t0.elapsed().as_micros();
    let conf = generator.spec_draft_conf().unwrap_or_default();
    let rows: usize = reqs.iter().map(|r| r.2.len()).sum();
    let mut line = format!(
        "{{\"live\":{},\"rows\":{rows},\"us\":{us},\"s\":[",
        reqs.len()
    );
    let mut row = 0usize;
    for (i, (_, _, chunk)) in reqs.iter().enumerate() {
        if !warm.get(i).copied().unwrap_or(false) {
            continue;
        }
        let c = conf.get(row).map(Vec::as_slice).unwrap_or_default();
        row += 1;
        let cs: Vec<String> = c.iter().map(|v| format!("{v:.4}")).collect();
        if !line.ends_with('[') {
            line.push(',');
        }
        line.push_str(&format!(
            "[{},{},[{}]]",
            chunk.len() - 1,
            accepted[i],
            cs.join(",")
        ));
    }
    line.push_str("]}\n");
    if let Ok(mut f) = file.lock() {
        let _ = f.write_all(line.as_bytes());
    }
}
