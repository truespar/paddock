//! Nemotron 3.5 ASR throughput and transcripts: every 16-kHz WAV of a
//! directory (or one file) through the CUDA lane, one JSON line a clip
//! (`file`, `seconds` of audio, `ms` of wall, `text`) and a summary line.
//!
//!   nemotron_asr_bench <model.gguf> <dir|file.wav> [--language en-US]
//!       [--lookahead 3] [--stream] [--repeat N] [--limit N] [--concurrency N]
//!
//! `--stream` pushes each clip one chunk of audio at a time (a live
//! session's arrival order; same tokens as the file) and reports the
//! per-chunk latency too. `--concurrency N` keeps N clips in flight at once,
//! every stream riding the same passes (a file fed whole, a stream a chunk a
//! tick), as fast as the GPU goes. The pack comes from `PADDOCK_PACK`.

#![allow(clippy::unwrap_used)] // a probe binary: a bad argument or file should panic loudly

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use paddock_engine::audio::wav;
use paddock_engine::gpu::GpuExecutor;
use paddock_engine::gpu_model::nemotron_asr::GpuNemotronAsr;
use paddock_models::nemotron_asr::{self as arch, HOP};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let [model, input, ..] = args.as_slice() else {
        eprintln!(
            "usage: nemotron_asr_bench <model.gguf> <dir|file.wav> [--language L] [--lookahead R] [--stream] [--repeat N] [--limit N]"
        );
        std::process::exit(2);
    };
    let language = flag("--language").unwrap_or_else(|| "en-US".into());
    let lookahead: usize =
        flag("--lookahead").map_or(arch::DEFAULT_LOOKAHEAD, |v| v.parse().unwrap());
    let repeat: usize = flag("--repeat").map_or(1, |v| v.parse().unwrap());
    let limit: usize = flag("--limit").map_or(usize::MAX, |v| v.parse().unwrap());
    let stream = has("--stream");
    let mut files: Vec<PathBuf> = if Path::new(input).is_dir() {
        std::fs::read_dir(input)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "wav"))
            .collect()
    } else {
        vec![PathBuf::from(input)]
    };
    files.sort();
    files.truncate(limit);
    let pack = std::env::var_os("PADDOCK_PACK").map(PathBuf::from);
    let exec = GpuExecutor::with_pack(0, pack.as_deref()).expect("open the GPU");
    let t = Instant::now();
    let mut m = GpuNemotronAsr::load(Arc::new(exec), Path::new(model)).expect("load");
    let load_s = t.elapsed().as_secs_f64();
    let prompt = m
        .prompt(&language)
        .unwrap_or_else(|| panic!("no prompt for {language}"));
    let clips: Vec<(String, Vec<f32>)> = files
        .iter()
        .map(|p| {
            let a = wav::decode_wav(&std::fs::read(p).unwrap()).unwrap();
            assert_eq!(a.sample_rate as usize, arch::SAMPLE_RATE, "{}", p.display());
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                a.samples,
            )
        })
        .collect();
    // warm-up: every kernel's first launch pays its module load
    m.transcribe(&vec![0.; arch::SAMPLE_RATE], lookahead, prompt)
        .unwrap();
    let chunk = arch::chunk_mel_frames(lookahead, false) * HOP;
    let concurrency: usize = flag("--concurrency").map_or(1, |v| v.parse().unwrap());
    let (mut audio, mut wall) = (0f64, 0f64);
    let mut lat: Vec<f64> = Vec::new();
    let mut clip_ms: Vec<f64> = Vec::new();
    if concurrency > 1 {
        // N clips in flight through shared passes
        let order: Vec<usize> = (0..repeat).flat_map(|_| 0..clips.len()).collect();
        let mut next = 0usize;
        let mut active: Vec<(
            usize,
            paddock_engine::gpu_model::nemotron_asr::Session,
            usize,
            Instant,
        )> = Vec::new();
        let t0 = Instant::now();
        let mut texts: Vec<Option<String>> = vec![None; clips.len()];
        while next < order.len() || !active.is_empty() {
            while active.len() < concurrency && next < order.len() {
                let s = m.session(lookahead, prompt).unwrap();
                active.push((order[next], s, 0, Instant::now()));
                next += 1;
            }
            for (ci, s, at, _) in active.iter_mut() {
                let pcm = &clips[*ci].1;
                let end = if stream {
                    (*at + chunk).min(pcm.len())
                } else {
                    pcm.len()
                };
                if *at < end {
                    m.feed(s, &pcm[*at..end]).unwrap();
                    *at = end;
                }
                if *at == pcm.len() {
                    m.close(s);
                }
            }
            let c = Instant::now();
            let mut refs: Vec<_> = active.iter_mut().map(|(_, s, _, _)| s).collect();
            while m.run(&mut refs).unwrap() > 0 {}
            if stream {
                lat.push(c.elapsed().as_secs_f64());
            }
            let mut i = 0;
            while i < active.len() {
                if active[i].1.is_finished() {
                    let (ci, s, _, start) = active.swap_remove(i);
                    clip_ms.push(start.elapsed().as_secs_f64() * 1e3);
                    audio += clips[ci].1.len() as f64 / arch::SAMPLE_RATE as f64;
                    if texts[ci].is_none() {
                        texts[ci] = Some(arch::detokenize(m.vocab(), &s.tokens, false));
                    }
                } else {
                    i += 1;
                }
            }
        }
        wall = t0.elapsed().as_secs_f64();
        for (ci, t) in texts.iter().enumerate() {
            if let Some(t) = t {
                println!("{}", serde_json::json!({"file": clips[ci].0, "text": t}));
            }
        }
    } else {
        for rep in 0..repeat {
            for (name, pcm) in &clips {
                let t = Instant::now();
                let s = if stream {
                    let mut s = m.session(lookahead, prompt).unwrap();
                    for part in pcm.chunks(chunk) {
                        let c = Instant::now();
                        m.push(&mut s, part).unwrap();
                        lat.push(c.elapsed().as_secs_f64());
                    }
                    m.finish(&mut s).unwrap();
                    s
                } else {
                    m.transcribe(pcm, lookahead, prompt).unwrap()
                };
                let ms = t.elapsed().as_secs_f64() * 1e3;
                let seconds = pcm.len() as f64 / arch::SAMPLE_RATE as f64;
                audio += seconds;
                wall += ms / 1e3;
                clip_ms.push(ms);
                if rep == 0 {
                    let text = arch::detokenize(m.vocab(), &s.tokens, false);
                    println!(
                        "{}",
                        serde_json::json!({"file": name, "seconds": seconds, "ms": ms, "text": text})
                    );
                }
            }
        }
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v.get(((v.len() as f64 - 1.) * p) as usize).copied()
    };
    println!(
        "{}",
        serde_json::json!({
            "summary": true,
            "clips": clips.len() * repeat,
            "concurrency": concurrency,
            "audio_seconds": audio,
            "wall_seconds": wall,
            "rtfx": audio / wall,
            "load_seconds": load_s,
            "lookahead": lookahead,
            "stream": stream,
            "tick_ms_p50": pct(&mut lat, 0.5).map(|v| v * 1e3),
            "tick_ms_p95": pct(&mut lat, 0.95).map(|v| v * 1e3),
            "clip_ms_p50": pct(&mut clip_ms, 0.5),
            "clip_ms_p95": pct(&mut clip_ms, 0.95),
        })
    );
}
