//! Every PADDOCK_* name the engine WRITES (its tuned-default elections,
//! `envset::set_env` and the load-time tables) must be read back through a
//! LIVE reader - `std::env`, `envset::env_on`, `hardening::env_on` - never
//! through the `dev_var!` / `dev_var_os!` / `dev_on!` macros, which compile to
//! "unset" under `--features hardened`. A dev-macro reader of an elected name
//! means every shipped binary silently drops the election it was measured
//! with (paddock_models::hardening, "Engine->pack transport"). Ten such
//! readers were found on 2026-10-04 - among them the GB10 decode-pipe floor
//! and the qwen35 compact q/k prefill pair - so this scan holds the line.
//!
//! The scan is textual on purpose: it must see the same literals the macros
//! do. A name is "written" when it is the first element of a
//! `("PADDOCK_..", ..)` tuple or a `set_env("PADDOCK_..")` argument in a file
//! that calls `set_env`, or listed in a `*SET_KILL` table.

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).expect("read src dir") {
        let p = e.expect("dir entry").path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// The PADDOCK_* name starting right after `at` (skipping whitespace and one
/// opening quote), if any.
fn name_after(s: &str, at: usize) -> Option<&str> {
    let rest = s[at..].trim_start();
    let rest = rest.strip_prefix('"')?;
    if !rest.starts_with("PADDOCK_") {
        return None;
    }
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn line_of(s: &str, at: usize) -> usize {
    s[..at].matches('\n').count() + 1
}

#[test]
fn engine_elected_env_names_have_live_readers() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    let texts: Vec<(PathBuf, String)> = files
        .into_iter()
        .map(|p| {
            let t = std::fs::read_to_string(&p).expect("read source");
            (p, t)
        })
        .collect();

    let mut written = std::collections::BTreeSet::new();
    for (_, s) in &texts {
        if !s.contains("set_env") {
            continue;
        }
        for (i, _) in s.match_indices("set_env(") {
            if let Some(n) = name_after(s, i + "set_env(".len()) {
                written.insert(n.to_string());
            }
        }
        for (i, _) in s.match_indices("(\"PADDOCK_") {
            if let Some(n) = name_after(s, i + 1) {
                let after = s[i + 2 + n.len() + 1..].trim_start();
                if after.starts_with(',') {
                    written.insert(n.to_string());
                }
            }
        }
        for (i, _) in s.match_indices("SET_KILL") {
            let tail = &s[i..];
            if let (Some(a), Some(b)) = (tail.find("&["), tail.find(']'))
                && a < b
            {
                for part in tail[a..b].split('"').skip(1).step_by(2) {
                    if part.starts_with("PADDOCK_") {
                        written.insert(part.to_string());
                    }
                }
            }
        }
    }
    assert!(
        written.len() > 40,
        "the scan found only {} written names - it has stopped seeing the tables",
        written.len()
    );

    let mut bad = Vec::new();
    for (p, s) in &texts {
        for mac in ["dev_var!(", "dev_var_os!(", "dev_on!("] {
            for (i, _) in s.match_indices(mac) {
                if let Some(n) = name_after(s, i + mac.len())
                    && written.contains(n)
                {
                    let rel = p.strip_prefix(&src).unwrap_or(p);
                    bad.push(format!("{} at {}:{}", n, rel.display(), line_of(s, i)));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "engine-elected env names read through a dev macro (dropped in shipped builds):\n  {}",
        bad.join("\n  ")
    );
}
