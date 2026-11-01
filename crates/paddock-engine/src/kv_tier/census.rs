//! The tier census (dev): one JSON line per tier event, so admission
//! policies for the disk tier can be priced offline against what a real
//! workload did - which bytes went to disk, which of them ever came back,
//! and when each left RAM. Keys are content keys, so a second run on the same
//! store joins against the first (restart persistence).
//!
//! PADDOCK_KV_CENSUS=<path>; nothing is written (or formatted) otherwise.
//! Events: `s` stored in RAM, `w` written to disk, `wd` skipped (already on
//! disk), `wx` disk write declined or failed, `wb` over the day's budget,
//! `r1` / `r2` restore read from RAM / disk, `e` left RAM (`d` = 1 when a disk
//! copy kept it), `x2` evicted from disk, `boot` a disk store opened (`b` =
//! entries recovered).

use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

fn sink() -> Option<&'static Mutex<std::fs::File>> {
    static F: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    F.get_or_init(|| {
        let path = paddock_models::dev_var_os!("PADDOCK_KV_CENSUS")?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()?;
        Some(Mutex::new(file))
    })
    .as_ref()
}

/// Record one event (`disk` only on `e`).
pub(crate) fn ev(kind: &str, key: &[u8; 32], bytes: u64, disk: Option<bool>) {
    let Some(f) = sink() else {
        return;
    };
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let k: String = key[..8].iter().map(|b| format!("{b:02x}")).collect();
    let d = match disk {
        Some(d) => format!(",\"d\":{}", u8::from(d)),
        None => String::new(),
    };
    let line = format!("{{\"t\":{ms},\"e\":\"{kind}\",\"k\":\"{k}\",\"b\":{bytes}{d}}}\n");
    if let Ok(mut f) = f.lock() {
        let _ = f.write_all(line.as_bytes());
    }
}
