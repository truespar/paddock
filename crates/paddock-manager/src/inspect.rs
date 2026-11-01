//! `paddock inspect` - the honest model card.

use std::path::Path;

use paddock_estimator::{ModelKind, PublishedShape, ShapeSource};
use paddock_models::probe::{ModelReport, probe_path};

/// What `--shape` was asked to emit. The geometry is read from the file either
/// way; this says where `weight_bytes` comes from.
pub enum Shape {
    /// Resident bytes handed in from a real load - `weights_mem_bytes` off a
    /// runner that actually served this file. See [`ShapeSource::Measured`].
    Measured(u64),
    /// No load was possible; fall back to the file's own byte count and SAY so
    /// in `source`. Wrong by the repack delta, which is why it is
    /// the fallback and not the default.
    Probed,
}

pub fn run(path: &Path, json: bool, shape: Option<Shape>, encoder: bool) -> std::process::ExitCode {
    let report = match probe_path(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("inspect failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
    };

    if let Some(want) = shape {
        let (bytes, source) = match want {
            Shape::Measured(b) => (b, ShapeSource::Measured),
            Shape::Probed => (report.file_size, ShapeSource::Probed),
        };
        let kind = if encoder {
            ModelKind::Encoder
        } else {
            ModelKind::Generative
        };
        print_shape(&PublishedShape::from_report(&report, bytes, kind, source));
        return std::process::ExitCode::SUCCESS;
    }

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => {
                eprintln!("serialization failed: {e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    } else {
        print_card(&report);
    }
    std::process::ExitCode::SUCCESS
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn fmt_opt(v: Option<u64>) -> String {
    v.map_or("-".into(), |v| v.to_string())
}

fn print_card(r: &ModelReport) {
    // split families: sizes/tensors above are family totals, say so
    let shards = if r.shards > 1 {
        format!(", {} files", r.shards)
    } else {
        String::new()
    };
    println!(
        "{}  ({:.2} GiB{shards}, GGUF v{})",
        r.path.display(),
        gib(r.file_size),
        r.gguf_version
    );
    println!(
        "architecture   {}",
        r.architecture.as_deref().unwrap_or("(undeclared)")
    );
    println!("context        {}", fmt_opt(r.context_length));
    println!(
        "geometry       blocks {} | embd {} | heads {}/{} kv | sliding window {}",
        fmt_opt(r.block_count),
        fmt_opt(r.embedding_length),
        fmt_opt(r.head_count),
        fmt_opt(r.head_count_kv),
        fmt_opt(r.sliding_window),
    );
    // The encoder-decoder's static cross cache is the dominant per-slot cost
    // where it exists, and it is invisible in every other line on this card:
    // it does not grow with context and does not shrink for a short clip, so
    // neither `context` nor the block count hints at it. Whisper holds a whole
    // 30 s audio window per concurrent transcription.
    if let Some(c) = &r.cross_kv {
        let per_slot = c.layers * c.frames * (c.k_dim + c.v_dim);
        println!(
            "cross-attention {} blocks x {} encoder frames = {:.0} MiB per concurrent \
             request at f16 ({:.0} at fp8), flat in context",
            c.layers,
            c.frames,
            (per_slot * 2) as f64 / (1u64 << 20) as f64,
            per_slot as f64 / (1u64 << 20) as f64,
        );
    }
    if r.expert_count.is_some() {
        println!(
            "experts        {} ({} active)",
            fmt_opt(r.expert_count),
            fmt_opt(r.expert_used_count)
        );
    }
    println!(
        "tokenizer      {} ({} tokens) | chat template: {}",
        r.tokenizer_model.as_deref().unwrap_or("-"),
        fmt_opt(r.token_count),
        if r.has_chat_template { "yes" } else { "no" },
    );
    println!("tensors        {}", r.tensor_count);
    println!("quant mix");
    for b in &r.quant_mix {
        match b.bytes {
            Some(bytes) => println!(
                "  {:<8} {:>4} tensors  {:>8.2} GiB",
                b.type_name,
                b.tensors,
                gib(bytes)
            ),
            // unverified layout: say so instead of printing a made-up size
            None => println!(
                "  {:<8} {:>4} tensors  (size unknown - unverified type layout)",
                b.type_name, b.tensors
            ),
        }
    }
}

/// Emit the `[model.artifact.shape]` block for models.toml.
///
/// Hand-written rather than `toml::to_string`, for two reasons a serializer
/// cannot give us: the indentation has to match the surrounding file (which
/// nests sub-tables two spaces per level), and an array of tables has to come
/// after every scalar of its parent or TOML re-parents the scalars that follow
/// it. The shapes generator splices this output verbatim.
fn print_shape(s: &PublishedShape) {
    let kind = match s.kind {
        ModelKind::Generative => "generative",
        ModelKind::Encoder => "encoder",
        ModelKind::Image => "image",
        ModelKind::Speech => "speech",
    };
    let source = match s.source {
        ShapeSource::Measured => "measured",
        ShapeSource::Probed => "probed",
    };
    println!("  [model.artifact.shape]");
    println!("  weight_bytes = {}", s.weight_bytes);
    println!("  kind = \"{kind}\"");
    println!("  vocab = {}", s.vocab);
    println!("  max_ctx = {}", s.max_ctx);
    println!("  nextn_bytes = {}", s.nextn_bytes);
    println!("  source = \"{source}\"");
    print_kv_runs(s);
    if let Some(r) = &s.recurrent {
        println!();
        println!("    [model.artifact.shape.recurrent]");
        println!("    layers = {}", r.layers);
        println!("    state_elems = {}", r.state_elems);
        println!("    conv_elems = {}", r.conv_elems);
        println!("    conv_dim = {}", r.conv_dim);
        println!("    elem_bytes = {}", r.elem_bytes);
    }
    if let Some(c) = &s.cross_kv {
        println!();
        println!("    [model.artifact.shape.cross_kv]");
        println!("    layers = {}", c.layers);
        println!("    frames = {}", c.frames);
        println!("    k_dim = {}", c.k_dim);
        println!("    v_dim = {}", c.v_dim);
    }
}

/// The KV runs, printed as a plain key so they land before the sub-tables - a
/// TOML scalar after `[model.artifact.shape.recurrent]` would belong to the
/// recurrent table, not to the shape.
fn print_kv_runs(s: &PublishedShape) {
    // Inline tables on one line each: the runs are the short, readable part of
    // the block and a stanza per run would bury them under their own headers.
    // A window is priced separately from a full block, so the runs stay
    // distinct rather than being summed into a layer count - that distinction
    // is what makes gemma4 and gpt-oss cheap at long context.
    if !s.kv_layers.is_empty() {
        println!("  kv_layers = [");
        for r in &s.kv_layers {
            // Full attention is the ABSENT key, not a sentinel: TOML has no
            // null, and `window = 0` would deserialize as Some(0) - a window of
            // zero tokens, which prices the block at nothing.
            let w = r
                .window
                .map_or(String::new(), |w| format!(", window = {w}"));
            println!(
                "    {{ k_dim = {}, v_dim = {}{w}, count = {} }},",
                r.k_dim, r.v_dim, r.count
            );
        }
        println!("  ]");
    }
}
