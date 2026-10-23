//! Persisted step-time calibration: what the scheduler's time models learned
//! about this machine, carried across restarts.
//!
//! Two learners start every process from nothing. The mixed-tick pacers
//! (`pacing.rs`) fit a cost model from the ticks they time, and until the fit
//! establishes, a fresh serve's first long ingest runs bootstrap ticks to
//! excite it. The speculation controller (`spec_policy.rs`) measures a
//! latency cell per (width, draft length) and must anchor every width it
//! meets before it trusts its argmax - the warm-up its own docs name as why
//! the closed loop trails the hand-tuned ladder on short runs. Offline-
//! profiled systems start warm: DeepSeek's DeepInfer profiles decode step
//! times at startup and caches the curve on disk, keyed by model and
//! parallel layout, for an hour. We keep learning online and persist what
//! was learned instead, so a restart of the same configuration starts from
//! the last run's observations - and every one of them keeps being replaced
//! by live measurement exactly as before (the pacers' per-cell recency, the
//! controller's latency EWMA and evidence decay).
//!
//! The file is keyed by everything a step time depends on - runner build,
//! GPU, model and drafter files, kernel pack, batch width, context, KV format
//! (the runner assembles it, see `Home`). The key is stored in full and
//! compared on load, so another configuration, an older format or a torn
//! write all read as "nothing saved". Saved when the scheduler goes idle (at
//! most every 30 s), every 5 minutes under load, and on an idle shutdown;
//! written beside the target and renamed over it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::pacing::{PacerSnapshot, Pacers};
use crate::spec_policy::{SpecController, SpecSnapshot};

const FORMAT: &str = "paddock-step-calibration";
const VERSION: u64 = 1;
/// Idle saves at most this often.
const IDLE_EVERY: Duration = Duration::from_secs(30);
/// Under continuous load, a save this often regardless.
const BUSY_EVERY: Duration = Duration::from_secs(300);

/// Where a model's calibration persists, and the configuration it describes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Home {
    pub path: PathBuf,
    /// Every input a step time depends on, one `name value` per line.
    pub key: String,
}

impl Home {
    /// `dir/steps-<digest of key>.txt`.
    pub fn new(dir: &Path, key: String) -> Self {
        let digest = blake3::hash(key.as_bytes()).to_hex();
        Self {
            path: dir.join(format!("steps-{}.txt", &digest.as_str()[..16])),
            key,
        }
    }
}

/// Both learners' state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// one per tick kind, in `Pacers::KINDS` order
    pub pacers: Vec<PacerSnapshot>,
    pub spec: Option<SpecSnapshot>,
}

impl Snapshot {
    /// Line format: a header, the key (one `key` line per key line), then
    /// one `pacer` block per tick kind and an optional `spec` block, closed
    /// by `end` - a torn write has no `end`. Floats print in Rust's shortest
    /// round-trip form, so a save and a load are exact.
    pub fn encode(&self, key: &str) -> String {
        use std::fmt::Write as _;
        let mut o = format!("{FORMAT} {VERSION}\n");
        for line in key.lines() {
            let _ = writeln!(o, "key {line}");
        }
        for p in &self.pacers {
            let knee = p.knee.map_or("- -".to_owned(), |(f, b)| format!("{f} {b}"));
            let _ = writeln!(o, "pacer {} {knee}", p.obs.len());
            for &(r, a, w, d) in &p.obs {
                let _ = writeln!(o, "obs {r} {a} {w} {d}");
            }
        }
        if let Some(sp) = &self.spec {
            let _ = writeln!(o, "spec {} {}", sp.lat.len(), sp.cov.len());
            for &(b, k, t, n) in &sp.lat {
                let _ = writeln!(o, "lat {b} {k} {t} {n}");
            }
            for &(e, n) in &sp.cov {
                let _ = writeln!(o, "cov {e} {n}");
            }
            let join = |v: &[f64]| v.iter().map(f64::to_string).collect::<Vec<_>>().join(" ");
            let _ = writeln!(o, "tri {}", join(&sp.tri));
            let _ = writeln!(o, "hit {}", join(&sp.hit));
            let _ = writeln!(o, "acc {} {}", sp.acc_tokens, sp.acc_stops);
        }
        o.push_str("end\n");
        o
    }

    /// Parse `text` if it is this format and was saved for `key`. Anything
    /// else - another configuration, another version, a torn write, a value
    /// out of range - is None: starting cold is the safe fallback.
    pub fn decode(text: &str, key: &str) -> Option<Self> {
        let mut lines = text.lines().peekable();
        if lines.next()? != format!("{FORMAT} {VERSION}") {
            return None;
        }
        let mut saved_key = Vec::new();
        while let Some(rest) = lines.peek().and_then(|l| l.strip_prefix("key ")) {
            saved_key.push(rest);
            lines.next();
        }
        if saved_key.join("\n") != key.lines().collect::<Vec<_>>().join("\n") {
            return None;
        }
        // `tag` and the line's whitespace-separated fields after it
        fn fields<'a>(line: &'a str, tag: &str) -> Option<Vec<&'a str>> {
            let rest = line.strip_prefix(tag)?;
            if !rest.is_empty() && !rest.starts_with(' ') {
                return None;
            }
            Some(rest.split_whitespace().collect())
        }
        let num = |f: &str| f.parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0);
        let int = |f: &str| f.parse::<usize>().ok();
        let mut out = Snapshot::default();
        loop {
            let line = lines.next()?;
            if line == "end" {
                return lines.next().is_none().then_some(out);
            }
            if let Some(f) = fields(line, "pacer") {
                let [n, kf, kb] = f[..] else { return None };
                let knee = match (kf, kb) {
                    ("-", "-") => None,
                    _ => Some((int(kf)?, int(kb)?)),
                };
                let mut obs = Vec::new();
                for _ in 0..int(n)? {
                    let f = fields(lines.next()?, "obs")?;
                    let [r, a, w, d] = f[..] else { return None };
                    obs.push((num(r)?, num(a)?, num(w)?, int(d)?));
                }
                out.pacers.push(PacerSnapshot { obs, knee });
                continue;
            }
            // the only other block; anything else is not this format
            let f = fields(line, "spec")?;
            {
                let [n_lat, n_cov] = f[..] else { return None };
                let mut lat = Vec::new();
                for _ in 0..int(n_lat)? {
                    let f = fields(lines.next()?, "lat")?;
                    let [b, k, t, n] = f[..] else { return None };
                    lat.push((int(b)?, int(k)?, num(t)?, n.parse::<u32>().ok()?));
                }
                let mut cov = Vec::new();
                for _ in 0..int(n_cov)? {
                    let f = fields(lines.next()?, "cov")?;
                    let [e, n] = f[..] else { return None };
                    cov.push((num(e)?, num(n)?));
                }
                let floats = |line: &str, tag: &str| -> Option<Vec<f64>> {
                    fields(line, tag)?.into_iter().map(num).collect()
                };
                let tri = floats(lines.next()?, "tri")?;
                let hit = floats(lines.next()?, "hit")?;
                let acc = floats(lines.next()?, "acc")?;
                let [acc_tokens, acc_stops] = acc[..] else {
                    return None;
                };
                out.spec = Some(SpecSnapshot {
                    lat,
                    cov,
                    tri,
                    hit,
                    acc_tokens,
                    acc_stops,
                });
            }
        }
    }
}

/// Write `text` beside `path`, then rename it over: a reader sees the old
/// file or the new one, never half of either. The temporary name carries the
/// pid, so two runners of one configuration cannot interleave a write.
fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("txt.tmp{}", std::process::id()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// The scheduler's side: loads on open, saves on the cadence in the module
/// docs. Every failure is a log line - a calibration that cannot be read or
/// written leaves the learners exactly as they are without one.
pub(crate) struct Saver {
    home: Option<Home>,
    last: Instant,
    /// the learners' observation marks at the last save
    saved: u64,
    writer: Option<std::thread::JoinHandle<()>>,
}

impl Saver {
    /// Seed both learners from `home`'s file, if it holds this configuration.
    pub(crate) fn open(home: Option<Home>, pacers: &mut Pacers, spec: &mut SpecController) -> Self {
        let mut saved = 0;
        if let Some(h) = &home {
            match std::fs::read_to_string(&h.path) {
                Ok(text) => match Snapshot::decode(&text, &h.key) {
                    Some(s) => {
                        pacers.seed(&s.pacers);
                        if let Some(sp) = &s.spec {
                            spec.seed(sp);
                        }
                        let ticks: usize = s.pacers.iter().map(|p| p.obs.len()).sum();
                        tracing::info!(
                            path = %h.path.display(),
                            pacer_ticks = ticks,
                            spec_cells = s.spec.as_ref().map_or(0, |sp| sp.lat.len()),
                            "scheduler calibration: starting from the last run's step times"
                        );
                        saved = pacers.marks() + spec.marks();
                    }
                    None => tracing::info!(
                        path = %h.path.display(),
                        "scheduler calibration: saved file is for another configuration; starting cold"
                    ),
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => tracing::debug!(%e, "scheduler calibration not read; starting cold"),
            }
        }
        Self {
            home,
            last: Instant::now(),
            saved,
            writer: None,
        }
    }

    /// Called when the scheduler goes idle (`idle`) and on busy passes:
    /// saves on a background thread when something was learned and the
    /// cadence allows it.
    pub(crate) fn tick(&mut self, idle: bool, pacers: &Pacers, spec: &SpecController) {
        let every = if idle { IDLE_EVERY } else { BUSY_EVERY };
        if self.home.is_none() || self.last.elapsed() < every {
            return;
        }
        let marks = pacers.marks() + spec.marks();
        if marks == self.saved {
            return;
        }
        if self.writer.as_ref().is_some_and(|w| !w.is_finished()) {
            return; // the last write is still going; the next tick retries
        }
        let Some(home) = self.home.clone() else {
            return;
        };
        let text = snapshot(pacers, spec).encode(&home.key);
        self.saved = marks;
        self.last = Instant::now();
        self.writer = Some(std::thread::spawn(move || {
            if let Err(e) = write_atomic(&home.path, &text) {
                tracing::debug!(%e, "scheduler calibration not saved");
            }
        }));
    }

    /// The scheduler is leaving: save now, on this thread, if anything new
    /// was learned (a background writer could be cut off by the exit).
    pub(crate) fn finish(&mut self, pacers: &Pacers, spec: &SpecController) {
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        let Some(home) = &self.home else {
            return;
        };
        let marks = pacers.marks() + spec.marks();
        if marks == self.saved {
            return;
        }
        match write_atomic(&home.path, &snapshot(pacers, spec).encode(&home.key)) {
            Ok(()) => self.saved = marks,
            Err(e) => tracing::debug!(%e, "scheduler calibration not saved"),
        }
    }
}

fn snapshot(pacers: &Pacers, spec: &SpecController) -> Snapshot {
    Snapshot {
        pacers: pacers.snapshot(),
        spec: spec.snapshot(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Snapshot {
        Snapshot {
            pacers: vec![
                PacerSnapshot {
                    obs: vec![(1026.0, 0.52, 760.5, 0), (1026.0, 180.3, 3100.25, 176_000)],
                    knee: Some((1024, 512)),
                },
                PacerSnapshot::default(),
                PacerSnapshot::default(),
            ],
            spec: Some(SpecSnapshot {
                lat: vec![(0, 0, 0.0251, 9), (0, 7, 0.0412, 6)],
                cov: vec![(1.0, 0.0); 8],
                tri: vec![0.0, 40.5, 30.25],
                hit: vec![0.0, 31.0, 18.5],
                acc_tokens: 120.0,
                acc_stops: 33.5,
            }),
        }
    }

    #[test]
    fn round_trips_for_its_own_key_only() {
        let s = sample();
        let text = s.encode("runner 0.1.18\ndevice GB10");
        assert_eq!(
            Snapshot::decode(&text, "runner 0.1.18\ndevice GB10"),
            Some(s)
        );
        assert_eq!(Snapshot::decode(&text, "runner 0.1.19\ndevice GB10"), None);
        // a torn write, another format, an out-of-range value
        let key = "runner 0.1.18\ndevice GB10";
        for cut in [text.len() / 3, text.len() / 2, text.len() - 4] {
            assert_eq!(Snapshot::decode(&text[..cut], key), None, "cut at {cut}");
        }
        let other = text.replace(FORMAT, "something-else");
        assert_eq!(Snapshot::decode(&other, key), None);
        let negative = text.replace("760.5", "-760.5");
        assert_eq!(Snapshot::decode(&negative, key), None);
        assert_eq!(Snapshot::decode(&format!("{text}obs 1 1 1 1\n"), key), None);
    }

    #[test]
    fn the_file_name_follows_the_key() {
        let dir = Path::new("/x");
        let a = Home::new(dir, "a".into());
        assert_eq!(a, Home::new(dir, "a".into()));
        assert_ne!(a.path, Home::new(dir, "b".into()).path);
    }

    /// A restart of the same configuration starts where the last run
    /// stopped: the pacer's fit is established before its first tick, and
    /// the controller skips the anchor rounds a cold width pays.
    #[test]
    fn a_saved_run_seeds_a_fresh_one() {
        let dir = std::env::temp_dir().join(format!("paddock-steps-{}", std::process::id()));
        let home = Home::new(&dir, "key".into());
        let mut pacers = Pacers::new();
        let mut spec = SpecController::new(crate::spec_policy::SpecPolicy::Auto);
        crate::pacing::warm_for_tests(&mut pacers);
        crate::spec_policy::warm_for_tests(&mut spec);
        assert!(pacers.mixed_model().is_some());
        let mut cold = SpecController::new(crate::spec_policy::SpecPolicy::Auto);
        let mut saver = Saver::open(Some(home.clone()), &mut Pacers::new(), &mut cold);
        saver.finish(&pacers, &spec);
        assert!(home.path.is_file());

        let mut fresh = Pacers::new();
        let mut fresh_spec = SpecController::new(crate::spec_policy::SpecPolicy::Auto);
        assert!(fresh.mixed_model().is_none());
        let _ = Saver::open(Some(home.clone()), &mut fresh, &mut fresh_spec);
        assert!(
            crate::pacing::tests::same_fit(fresh.mixed_model(), pacers.mixed_model()),
            "same ticks, same fit"
        );
        assert_eq!(fresh_spec.snapshot(), spec.snapshot());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
