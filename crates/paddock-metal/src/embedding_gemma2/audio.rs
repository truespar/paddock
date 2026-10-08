//! Native 12-layer Conformer. The exact processor frontend and every model
//! operation run on Metal; only immutable DSP coefficient tables are planned
//! on the host. Clipped linears share F16 inputs and tensor contractions.
use super::vision::{point, weight};
use super::*;
use crate::device::Commands;
use paddock_engine::encoder::embedding_gemma2::audio_frames;
use paddock_models::{gguf::Value, mapped::MappedGguf};
const E: usize = 1024;
const F: usize = 4096;
struct Linear {
    w: Weight,
    input: [f32; 2],
    output: [f32; 2],
}
struct Ffn {
    pre: Weight,
    up: Linear,
    down: Linear,
    post: Weight,
}
struct Block {
    f1: Ffn,
    f2: Ffn,
    pre: Weight,
    qkv: Weight,
    input: [f32; 2],
    limits: [u32; 8],
    out: Linear,
    post: Weight,
    rel: Buffer,
    pds: Weight,
    conv_pre: Weight,
    pw1: Linear,
    dw: Weight,
    conv_norm: Weight,
    pw2: Linear,
    close: Weight,
}
pub(super) struct Audio {
    mlx: bool,
    window: Buffer,
    bank: Buffer,
    twiddle: Buffer,
    conv: [(Weight, Weight); 2],
    input: Weight,
    blocks: Vec<Block>,
    output: Weight,
    bias: Weight,
    projection: Weight,
    #[cfg(test)]
    pub(super) llama_floor: bool,
}
fn upload(d: &MetalDevice, v: &[f32]) -> Result<Buffer> {
    d.upload(&v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
}
fn mm(c: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    if w.ty == 0x108 {
        c.dispatch(
            "eg2_project_a8_f32",
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, rows as u32],
            [w.n.div_ceil(64), rows.div_ceil(32), 1],
            128,
        );
        return;
    }
    c.dispatch(
        if w.ty == 30 { "vis_bmm64" } else { "vis_mm64" },
        &[&w.buffer, x, y, &w.buffer],
        &[w.k as u32, w.n as u32, rows as u32, 0],
        [w.n.div_ceil(64), rows.div_ceil(64), 1],
        128,
    );
}
/// Flags: residual, post RMS, closing RMS, next RMS, next input. One launch
/// owns the complete row; no global atomics or batch-dependent reductions.
fn seam(
    c: &Commands<'_>,
    x: &Buffer,
    y: &Buffer,
    post: &Weight,
    close: &Weight,
    next: &Weight,
    stage: &Buffer,
    rows: usize,
    flags: u32,
    bounds: [f32; 2],
    scale: f32,
    input: [f32; 2],
) {
    c.dispatch(
        "eg2a_seam",
        &[x, y, &post.buffer, &close.buffer, &next.buffer, stage],
        &[
            flags,
            bounds[0].to_bits(),
            bounds[1].to_bits(),
            scale.to_bits(),
            input[0].to_bits(),
            input[1].to_bits(),
        ],
        [rows, 1, 1],
        256,
    );
}
impl Audio {
    #[cfg(test)]
    pub(super) fn log_mel(&self, d: &MetalDevice, samples: &[f32]) -> Result<Vec<f32>> {
        let frames = audio_frames(samples.len()).map_err(error)?;
        let pcm = upload(d, samples)?;
        let mel = d.alloc(frames * 128 * 4)?;
        let c = d.begin()?;
        c.dispatch(
            "eg2a_mel",
            &[&pcm, &self.window, &self.bank, &self.twiddle, &mel],
            &[samples.len() as u32, 0],
            [frames, 1, 1],
            256,
        );
        c.finish()?;
        Ok(unsafe { mel.read_f32(0, frames * 128) })
    }
    pub(super) fn load(d: &MetalDevice, m: &MappedGguf) -> Result<Self> {
        let u = |k| m.gguf().metadata.get(k).and_then(Value::as_u64);
        if [
            u("clip.audio.block_count"),
            u("clip.audio.embedding_length"),
            u("clip.audio.attention.head_count"),
            u("clip.audio.num_mel_bins"),
            u("clip.audio.projection_dim"),
        ] != [Some(12), Some(1024), Some(8), Some(128), Some(512)]
        {
            return Err(error(
                "EmbeddingGemma 2 requires its 12-layer 1024-wide gemma4a companion",
            ));
        }
        Self::from_weights(d, false, |name, shape, ty| weight(d, m, name, shape, ty))
    }
    pub(super) fn from_weights(
        d: &MetalDevice,
        mlx: bool,
        w: impl Fn(&str, &[usize], u32) -> Result<Weight>,
    ) -> Result<Self> {
        let scalar = |name: String| -> Result<f32> {
            let v = w(&name, &[1], 0)?;
            Ok(unsafe { v.buffer.read_f32(0, 1)[0] })
        };
        let linear = |name: String, k, n| -> Result<Linear> {
            let input = [
                scalar(format!("{name}.input_min"))?,
                scalar(format!("{name}.input_max"))?,
            ];
            let output = [
                scalar(format!("{name}.output_min"))?,
                scalar(format!("{name}.output_max"))?,
            ];
            if input[0] > input[1]
                || output[0] > output[1]
                || input.iter().any(|v| v.abs() > 65504.)
            {
                return Err(error(format!("{name}: invalid clipped-linear bounds")));
            }
            Ok(Linear {
                w: w(&format!("{name}.weight"), &[k, n], 1)?,
                input,
                output,
            })
        };
        // Relative positions are fixed shape metadata, evaluated on the GPU.
        let positions = d.alloc(13 * E * 4)?;
        let c = d.begin()?;
        point(&c, "eg2a_positions", &[&positions], &[mlx as u32], 13 * E);
        c.finish()?;
        let mut blocks = Vec::new();
        for i in 0..12 {
            let key = |s: &str| format!("a.blk.{i}.{s}");
            let vec = |s: &str| w(&key(s), &[E], 0);
            let ffn = |s: &str| -> Result<Ffn> {
                Ok(Ffn {
                    pre: vec(&format!("ffn_norm{s}.weight"))?,
                    post: vec(&format!("ffn_post_norm{s}.weight"))?,
                    up: linear(key(&format!("ffn_up{s}")), E, F)?,
                    down: linear(key(&format!("ffn_down{s}")), F, E)?,
                })
            };
            let (q, k, v) = (
                linear(key("attn_q"), E, E)?,
                linear(key("attn_k"), E, E)?,
                linear(key("attn_v"), E, E)?,
            );
            if q.input != k.input || q.input != v.input {
                return Err(error("audio QKV input clamps differ"));
            }
            let qkv = Weight {
                buffer: d.alloc(3 * E * E * 2)?,
                ty: if mlx { 30 } else { 1 },
                k: E,
                n: 3 * E,
            };
            let relw = w(&key("attn_k_rel.weight"), &[E, E], if mlx { 30 } else { 0 })?;
            let rel = d.alloc(13 * E * 4)?;
            let c = d.begin()?;
            for (i, l) in [&q, &k, &v].into_iter().enumerate() {
                point(
                    &c,
                    "spec_copy",
                    &[&l.w.buffer, &qkv.buffer],
                    &[0, (i * E * E / 2) as u32, (E * E / 2) as u32],
                    E * E / 2,
                );
            }
            if mlx {
                c.dispatch(
                    "gmlx_vmm64",
                    &[&relw.buffer, &positions, &rel, &relw.buffer],
                    &[E as u32, E as u32, 13, 0],
                    [E.div_ceil(64), 1, 1],
                    128,
                );
            } else {
                super::vision::mm(&c, &relw, &positions, &rel, 13);
            }
            c.finish()?;
            let out = linear(key("attn_out"), E, E)?;
            let limits = [
                q.output[0],
                q.output[1],
                k.output[0],
                k.output[1],
                v.output[0],
                v.output[1],
                out.input[0],
                out.input[1],
            ]
            .map(f32::to_bits);
            blocks.push(Block {
                f1: ffn("")?,
                f2: ffn("_1")?,
                pre: vec("attn_pre_norm.weight")?,
                qkv,
                input: q.input,
                limits,
                out,
                post: vec("attn_post_norm.weight")?,
                rel,
                pds: w(&key("per_dim_scale.weight"), &[128], 0)?,
                conv_pre: vec("conv_norm.weight")?,
                pw1: linear(key("conv_pw1"), E, 2 * E)?,
                dw: w(&key("conv_dw.weight"), &[5, E], 0)?,
                conv_norm: vec("norm_conv.weight")?,
                pw2: linear(key("conv_pw2"), E, E)?,
                close: vec("ln2.weight")?,
            });
        }
        use paddock_engine::audio::dsp::{MelScale, hann_periodic, mel_filterbank};
        let window = hann_periodic(320)
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<_>>();
        let bank = mel_filterbank(128, 512, 16000., 0., 8000., MelScale::Htk, false)
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<_>>();
        let twiddle = (0..256)
            .flat_map(|i| {
                let (s, c) = (-2. * std::f64::consts::PI * i as f64 / 512.).sin_cos();
                [
                    c as f32,
                    s as f32,
                    (c - f64::from(c as f32)) as f32,
                    (s - f64::from(s as f32)) as f32,
                ]
            })
            .collect::<Vec<_>>();
        Ok(Self {
            mlx,
            window: upload(d, &window)?,
            bank: upload(d, &bank)?,
            twiddle: upload(d, &twiddle)?,
            blocks,
            conv: [
                (
                    w("a.conv1d.0.weight", &[3, 3, 1, 128], 0)?,
                    w("a.conv1d.0.norm.weight", &[128], 0)?,
                ),
                (
                    w("a.conv1d.1.weight", &[3, 3, 128, 32], 0)?,
                    w("a.conv1d.1.norm.weight", &[32], 0)?,
                ),
            ],
            input: w("a.input_projection.weight", &[E, E], 0)?,
            output: w("a.pre_encode.out.weight", &[E, 1536], 1)?,
            bias: w("a.pre_encode.out.bias", &[1536], 0)?,
            projection: w("mm.a.input_projection.weight", &[1536, WIDTH], 1)?,
            #[cfg(test)]
            llama_floor: false,
        })
    }
    /// Pack projection rows without padding audio or joining causal domains.
    /// One bounded command owns all clips; only the small frontend scratch is
    /// reused serially. The expensive Conformer projections read weights once
    /// per packed tile, not once per request.
    pub(super) fn encode_batch(&self, d: &MetalDevice, clips: &[&[f32]]) -> Result<Vec<Buffer>> {
        if clips.is_empty() {
            return Err(error("empty audio batch"));
        }
        let frames_per_clip: Vec<_> = clips
            .iter()
            .map(|s| audio_frames(s.len()).map_err(error))
            .collect::<Result<_>>()?;
        let rows: usize = frames_per_clip.iter().map(|f| f.div_ceil(4)).sum();
        if rows > 4096 {
            return Err(error("audio projection group exceeds 4096 rows"));
        }
        let frames = *frames_per_clip.iter().max().expect("nonempty clips");
        let t1 = frames.div_ceil(2);
        let pcms = clips
            .iter()
            .map(|s| upload(d, s))
            .collect::<Result<Vec<_>>>()?;
        let mut first = 0usize;
        let starts: Vec<_> = frames_per_clip
            .iter()
            .flat_map(|f| {
                let start = first;
                first += f.div_ceil(4);
                std::iter::repeat_n(start as u32, f.div_ceil(4))
            })
            .flat_map(u32::to_le_bytes)
            .collect();
        let starts = d.upload(&starts)?;
        let outputs = frames_per_clip
            .iter()
            .map(|f| d.alloc(f.div_ceil(4) * WIDTH * 4))
            .collect::<Result<Vec<_>>>()?;
        let mel = d.alloc(frames * 128 * 4)?;
        let c0 = d.alloc(t1 * 64 * 128 * 4)?;
        let c1 = d.alloc(frames.div_ceil(4) * E * 4)?;
        let x = d.alloc(rows * E * 4)?;
        let operand_bytes = if self.mlx { 4 } else { 2 };
        let s = d.alloc(rows * F * operand_bytes)?;
        let h = d.alloc(rows * F * operand_bytes)?;
        let wide = d.alloc(rows * F * 4)?;
        let y = d.alloc(rows * E * 4)?;
        let o = d.alloc(rows * 1536 * 4)?;
        let output = d.alloc(rows * WIDTH * 4)?;
        let c = d.begin()?;
        let floor = false;
        #[cfg(test)]
        let floor = floor || self.llama_floor;
        let mut first = 0usize;
        for ((samples, pcm), frames) in clips.iter().zip(&pcms).zip(&frames_per_clip) {
            let t1 = frames.div_ceil(2);
            let clip_rows = frames.div_ceil(4);
            c.dispatch(
                "eg2a_mel",
                &[pcm, &self.window, &self.bank, &self.twiddle, &mel],
                &[samples.len() as u32, floor as u32],
                [*frames, 1, 1],
                256,
            );
            for (i, src, dst, ti, fi, ci, co) in [
                (0, &mel, &c0, *frames, 128, 1, 128),
                (1, &c0, &c1, t1, 64, 128, 32),
            ] {
                c.dispatch(
                    "eg2a_subsample",
                    &[src, &self.conv[i].0.buffer, &self.conv[i].1.buffer, dst],
                    &[ti as u32, fi, ci, co],
                    [fi.div_ceil(2) as usize, ti.div_ceil(2), 1],
                    co as usize,
                );
            }
            c.dispatch_at(
                "gv_patch_project",
                &[&self.input.buffer, &c1, &x, &self.input.buffer],
                &[0, 0, first * E * 4, 0],
                &[E as u32, E as u32, clip_rows as u32, 0],
                [E.div_ceil(64), clip_rows.div_ceil(64), 1],
                128,
            );
            first += clip_rows;
        }
        let seam = |c: &Commands<'_>,
                    x: &Buffer,
                    y: &Buffer,
                    post: &Weight,
                    close: &Weight,
                    next: &Weight,
                    stage: &Buffer,
                    rows,
                    flags: u32,
                    bounds,
                    scale,
                    input| {
            seam(
                c,
                x,
                y,
                post,
                close,
                next,
                stage,
                rows,
                flags | if self.mlx { 32 } else { 0 },
                bounds,
                scale,
                input,
            );
        };
        let first = &self.blocks[0].f1;
        seam(
            &c,
            &x,
            &y,
            &first.post,
            &first.pre,
            &first.pre,
            &s,
            rows,
            24,
            [0., 0.],
            0.,
            first.up.input,
        );
        let ff = |f: &Ffn| {
            mm(&c, &f.up.w, &s, &wide, rows);
            point(
                &c,
                "eg2a_act",
                &[&wide, &h],
                &[
                    (rows * F) as u32,
                    f.up.output[0].to_bits(),
                    f.up.output[1].to_bits(),
                    f.down.input[0].to_bits(),
                    f.down.input[1].to_bits(),
                    self.mlx as u32,
                ],
                rows * F,
            );
            mm(&c, &f.down.w, &h, &y, rows);
        };
        for (i, b) in self.blocks.iter().enumerate() {
            ff(&b.f1);
            seam(
                &c,
                &x,
                &y,
                &b.f1.post,
                &b.close,
                &b.pre,
                &s,
                rows,
                27,
                b.f1.down.output,
                0.5,
                b.input,
            );
            mm(&c, &b.qkv, &s, &wide, rows);
            c.dispatch(
                "eg2a_attention",
                &[&wide, &b.rel, &b.pds.buffer, &h, &starts],
                &[b.limits.as_slice(), &[self.mlx as u32]].concat(),
                [rows, 1, 1],
                256,
            );
            mm(&c, &b.out.w, &h, &y, rows);
            seam(
                &c,
                &x,
                &y,
                &b.post,
                &b.close,
                &b.conv_pre,
                &s,
                rows,
                27,
                b.out.output,
                1.,
                b.pw1.input,
            );
            mm(&c, &b.pw1.w, &s, &wide, rows);
            c.dispatch(
                "eg2a_conv",
                &[&wide, &b.dw.buffer, &b.conv_norm.buffer, &h, &starts],
                &[
                    b.pw1.output[0].to_bits(),
                    b.pw1.output[1].to_bits(),
                    b.pw2.input[0].to_bits(),
                    b.pw2.input[1].to_bits(),
                    self.mlx as u32,
                ],
                [rows, 1, 1],
                256,
            );
            mm(&c, &b.pw2.w, &h, &y, rows);
            seam(
                &c,
                &x,
                &y,
                &b.post,
                &b.close,
                &b.f2.pre,
                &s,
                rows,
                25,
                b.pw2.output,
                1.,
                b.f2.up.input,
            );
            ff(&b.f2);
            let next = self.blocks.get(i + 1).map(|b| &b.f1);
            seam(
                &c,
                &x,
                &y,
                &b.f2.post,
                &b.close,
                next.map_or(&b.pre, |f| &f.pre),
                &s,
                rows,
                if next.is_some() { 31 } else { 23 },
                b.f2.down.output,
                0.5,
                next.map_or([f32::NEG_INFINITY, f32::INFINITY], |f| f.up.input),
            );
        }
        mm(&c, &self.output, &s, &o, rows);
        c.dispatch(
            "eg2a_out",
            &[&o, &self.bias.buffer, &h],
            &[self.mlx as u32],
            [rows, 1, 1],
            256,
        );
        mm(&c, &self.projection, &h, &output, rows);
        let mut first = 0usize;
        for (frames, dest) in frames_per_clip.iter().zip(&outputs) {
            let n = frames.div_ceil(4) * WIDTH;
            point(
                &c,
                "spec_copy",
                &[&output, dest],
                &[first as u32, 0, n as u32],
                n,
            );
            first += n;
        }
        c.finish()?;
        Ok(outputs)
    }
}
