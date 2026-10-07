//! The text a runner that died during startup leaves the person who started
//! it: the reason first, read out of the tail of its log.

/// The runner's last log line is the actual reason ("device \"cuda\" needs a
/// kernel_pack path in config") - lead with it; the full tail follows for the
/// detail page. Never show Debug formatting (`Some(1)`) or ANSI color codes
/// to a person: the old text put the reason after a newline, and the fleet
/// row's single-line cell showed "exit Some(1) - log tail:" with nothing else.
/// Strip a tracing line's machinery so the SENTENCE is what a person meets.
///
/// A runner log line arrives as
///   `2026-08-17T16:36:22.840396Z ERROR paddock_runner::startup: server error
///    error=engine startup: qwen35 cannot serve max_ctx 131072 ...`
/// and the first ~90 characters of that are a timestamp, a level, a module
/// path and two layers of `error=` wrapping. Handed to a toast - which clamps
/// - the reader sees the timestamp and none of the answer. Seen in practice:
///   the whole 600-character explanation was reaching the browser correctly and
///   still read as "no error message, no nothing", because the part that fits was
///   all preamble.
///
/// Deliberately conservative: every step is optional, so a line that does not
/// look like this (a panic, a linker message, a bare string) passes through
/// untouched rather than being mangled by a guess.
fn human_line(line: &str) -> String {
    let mut s = line.trim();
    // ISO-8601 stamp, then the level word.
    if let Some((first, rest)) = s.split_once(char::is_whitespace)
        && first.len() >= 20
        && first.starts_with(|c: char| c.is_ascii_digit())
        && first.ends_with('Z')
    {
        s = rest.trim_start();
    }
    for lvl in ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"] {
        if let Some(rest) = s.strip_prefix(lvl) {
            s = rest.trim_start();
            break;
        }
    }
    // `some::module::path: ` - a target, only when it really looks like one.
    if let Some((head, rest)) = s.split_once(": ")
        && head.contains("::")
        && !head.contains(' ')
    {
        s = rest.trim_start();
    }
    // The runner's own wrappers: `server error error=` then `engine startup:`.
    // Both name the layer that caught it, not what went wrong.
    for cut in ["server error error=", "error="] {
        if let Some(i) = s.find(cut) {
            s = s[i + cut.len()..].trim_start();
            break;
        }
    }
    if let Some(rest) = s.strip_prefix("engine startup:") {
        s = rest.trim_start();
    }
    s.to_owned()
}

pub(super) fn died_on_startup_text(code: &Option<i32>, tail: &str) -> String {
    let clean = strip_ansi(tail);
    let last = clean
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(human_line)
        .unwrap_or_default();
    let code_s = code
        .map(|c| format!(" (exit code {c})"))
        .unwrap_or_default();
    if last.is_empty() {
        format!("the model server exited during startup{code_s} and left no log")
    } else {
        // The reason leads. The exit code is bookkeeping and goes after it; the
        // full tail stays for the detail view, below a blank line so a UI that
        // shows only the first paragraph still shows the whole answer.
        format!("{last}\n\n(the model server exited during startup{code_s})\n\nlog tail:\n{clean}")
    }
}

/// Tiny ESC-sequence skipper - the runner logs colored output, and a color
/// code inside an error message is noise squared.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line the browser actually met. What a toast can show
    /// is the first ~100 characters, so those characters have to be the answer.
    #[test]
    fn a_start_failure_leads_with_the_reason_not_a_timestamp() {
        let tail = "2026-08-17T16:36:20.802698Z  INFO paddock_runner::serving: kv cache: f16\n\
             2026-08-17T16:36:22.840396Z ERROR paddock_runner::startup: server error \
             error=engine startup: qwen35 cannot serve max_ctx 131072 x max_batch 1: needs \
             8.00 GiB of KV (8192 blocks), 3.75 GiB fits (3844 blocks, 61504 tokens shared). \
             Fixes: lower max_ctx to <=61504, or raise vram_budget";
        let text = died_on_startup_text(&Some(1), tail);
        let first = text.lines().next().unwrap();
        assert!(
            first.starts_with("qwen35 cannot serve max_ctx 131072"),
            "{first}"
        );
        // none of the machinery survives into the part a person reads
        for noise in [
            "2026-08-17T",
            "ERROR",
            "paddock_runner::startup",
            "error=",
            "engine startup:",
        ] {
            assert!(!first.contains(noise), "{noise:?} leaked into: {first}");
        }
        // and the actionable half is still in that same first line
        assert!(first.contains("Fixes: lower max_ctx"), "{first}");
        // the exit code and the whole tail remain, for the detail view
        assert!(text.contains("exit code 1"));
        assert!(text.contains("log tail:"));
        assert!(text.contains("kv cache: f16"));
    }

    /// A line that is not a tracing line must survive untouched - the stripper
    /// guesses, so it has to fail safe.
    #[test]
    fn human_line_leaves_unrecognised_shapes_alone() {
        for raw in [
            "thread 'main' panicked at src/main.rs:12:5: assertion failed",
            "LINK : fatal error LNK1181: cannot open input file 'pd-cuda.lib'",
            "CUDA error 2: out of memory",
            "",
        ] {
            assert_eq!(human_line(raw), raw.trim(), "mangled: {raw}");
        }
    }

    #[test]
    fn a_startup_death_with_no_log_still_says_something() {
        let text = died_on_startup_text(&Some(101), "");
        assert!(text.contains("exited during startup"), "{text}");
        assert!(text.contains("101"), "{text}");
    }
}
