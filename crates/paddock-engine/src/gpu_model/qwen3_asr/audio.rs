//! Qwen3-ASR audio tower (`clip` GGUF, `projector_type == "qwen3a"`).
//! Reference: transformers 5.14.1 `modeling_qwen3_asr.py` (the
//! upstream graph; llama.cpp b10327 `models/qwen3a.cpp` studied as the
//! second reference). Dataflow per clip:
//!
//!   normalized log-mel [n_frames, 128] (host, `crate::audio`)
//!   -> per 100-frame chunk (last chunk zero-padded to 100 - upstream runs
//!     the convs on the padded chunk and TRUNCATES the token outputs, and
//!     pad-derived GELU(bias) values legitimately leak into the last valid
//!     outputs through odd-width stages, so exact-length conv would diverge)
//!   -> 3× conv2d(k3, s2, p1) + GELU-erf over (mel 128->64->32->16, time
//!     100->50->25->13), channels 1->480->480->480
//!   -> conv_out [7680->1024] (no bias) + sinusoidal positions reset 0..12
//!     per chunk (the mmproj's baked `a.position_embd` table)
//!   -> keep the first `audio_token_count(n_frames)` token rows (valid rows
//!     are a prefix: only the clip's last chunk is short)
//!   -> 24 × pre-LN blocks [LN1 -> QKV(+bias) -> bidirectional attention over
//!     104-token windows (8 chunks; windows never attend across) ->
//!     out(+bias) -> +res -> LN2 -> up(+bias) -> GELU-erf -> down(+bias) -> +res]
//!   -> post-LN -> mlp1(+bias) -> GELU-erf -> mlp2(+bias)
//!   -> [n_tokens, 2048] LLM-space audio embeddings.
//!
//! Chassis follows qwen35/vision.rs: f32 activations over f16 weight planes,
//! one f16 staging buffer, `vision_attn_at` for the unmasked window
//! attention (hd 64 rides the mma path). The conv stem is implicit-GEMM:
//! stage 1's im2col is built on host with the mel frames, stages 2-3 gather
//! on device via `gather_rows_avg(k=1)` driven by a host-built index table
//! with a dedicated always-zero row as the out-of-range source (`alloc` is
//! alloc_zeros, so the pad row costs nothing) - no new pack kernel. Weight
//! k-orders are permuted at load so every gather emits contiguous channel
//! runs: conv2/conv3 (c,ky,kx)->(ky,kx,c); conv3 emits rows in (w3,h3) order
//! and conv_out's k is permuted (c,h3)->(h3,c), which makes conv_out's input
//! a contiguous 16-row view of conv3's output (the vision-merger trick).
//!
//! Long clips run the conv stem in bounded chunk groups (scratch stays a few
//! hundred MB regardless of clip length); the 24 encoder layers then run
//! over the full token stream in one pass (20 min of audio is only ~15.6k
//! tokens). Encoder windows are 104-token groups, which align with 8-chunk
//! groups because every chunk except the clip's last yields exactly 13
//! tokens - group boundaries can never straddle a conv group (16 chunks).
//!
//! ## Scratch is RESIDENT, not per-encode
//!
//! Every plane below is allocated once in [`TowerScratch`] and reused. The
//! bring-up version allocated them inside `encode` - 21 `alloc_zeros` +
//! 17 memsets per call, group-capacity sized (~350 MB of alloc/free/zero
//! churn regardless of clip length). At c32 that path runs ~79 times a
//! SECOND on the engine thread, one clip per admission, and profiling put the
//! whole tower at ~30% of the wall for 18% of the useful GPU work: 542
//! launches per encode, its episodes 2.1 ms busy inside a 3.7 ms span.
//!
//! Resident is the same peak VRAM the transient path already reached - the
//! difference is that it is now visible to the KV-pool sizing at enable time
//! rather than appearing under it at the first encode.
//!
//! ## The cost here is the LAUNCH TRAIN
//!
//! An earlier note claimed the transient allocs were also what pinned the
//! encode to the tick stream, and acted on it: the whole encode moved
//! to a forked lane with two events per encode. Throughput REGRESSED,
//! and - the number that settles it - `encode`'s host wall
//! did not move: 2.9 ms/enc single-stream, 2.9 ms/enc on a fresh empty side
//! stream. That time was never drain against a saturated queue; it is the
//! host cost of ISSUING ~540 launches, which no amount of stream separation
//! can move off this thread. The witness below splits it (`PADDOCK_TICK_STATS=1`):
//! ~150 `cublasGemmEx` at 7.7 us each = 1.16 ms, ~390 of our own launches at
//! 4.4 us each = 1.73 ms, host im2col 0.19 ms.
//!
//! So the open door is launch COUNT, not stream or allocation: per layer this
//! issues 6 GEMMs and 16 elementwise launches (layernorm, 3 f32->f16 converts,
//! 6 bias_adds, attention, gelu, 2 residual adds). Fusing them - f16-out
//! layernorm, merged QKV, bias+gelu, bias+residual, f16 attention epilogue -
//! is worth ~1 ms of every encode and ~80 ms of every wall second.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::audio::{CHUNK_FRAMES, MelFeatures, N_MEL, audio_token_count};
use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::vision::host_f32;

/// Conv-stem geometry: (mel, time) per stage under k3/s2/p1.
const H: [usize; 4] = [N_MEL, 64, 32, 16];
const W: [usize; 4] = [CHUNK_FRAMES, 50, 25, 13];
/// Chunks per conv-stem pass - bounds the im2col scratch (~350 MB at 16)
/// while keeping the GEMMs wide. Must stay a multiple of 8 so attention
/// windows (8 chunks) never straddle a group.
const CONV_GROUP: usize = 16;
/// Tokens per encoder attention window (8 chunks x 13).
const WINDOW_TOKENS: usize = 104;

struct ABlock {
    ln1_w: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    wq: HalfTensor,
    bq: CudaSlice<f32>,
    wk: HalfTensor,
    bk: CudaSlice<f32>,
    wv: HalfTensor,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2_w: CudaSlice<f32>,
    ln2_b: CudaSlice<f32>,
    up_w: HalfTensor,
    up_b: CudaSlice<f32>,
    down_w: HalfTensor,
    down_b: CudaSlice<f32>,
}

/// The encoded clip: audio-token embeddings ready for LLM injection.
pub struct AudioOutput {
    /// [n_tokens, llm_embd] device-resident audio embeddings.
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

/// Token rows one encoder SEGMENT covers (~5.3 min of audio): the clip runs
/// through the conv stem, the 24 layers and the projector a segment at a
/// time, so every token-domain plane is sized for this once, at load.
///
/// Those planes used to grow to the longest clip seen and stay there, the
/// whole clip in one pass, bounded only by the prompt fitting max_ctx - and
/// each growth built the new planes while the old were still held. Segments
/// are exact, not an approximation: they are a whole number of conv groups
/// (20 x 16 chunks), so no conv group straddles one, and a multiple of the
/// 104-token attention window, so no window does - attention never crosses a
/// window and every other op is per row. A clip that fits one segment runs
/// exactly the passes it always did.
const SEGMENT_TOKENS: usize = 20 * CONV_GROUP * 13;
const _: () = assert!(SEGMENT_TOKENS.is_multiple_of(WINDOW_TOKENS));

/// Encode witness (`PADDOCK_TICK_STATS=1`): the admission tower is ~24% of
/// the c32 ASR wall and sits outside every tick bucket, so the tick table
/// cannot see it. The three-way split - host im2col / cuBLAS / our own
/// launches - is what ANSWERED "stop doing host work on the engine thread"
/// vs "issue fewer launches" vs "get off the tick stream": the last was
/// tried and falsified (see the module note), and this witness is what the
/// fusion rung will be scored on. Aggregated
/// over a 5 s window - this path runs ~79 times a second.
mod witness {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static N: AtomicU64 = AtomicU64::new(0);
    pub static ROWS: AtomicU64 = AtomicU64::new(0);
    /// host-side im2col loop
    pub static IM2COL_NS: AtomicU64 = AtomicU64::new(0);
    /// everything else up to the last launch returning (the launch train)
    pub static ISSUE_NS: AtomicU64 = AtomicU64::new(0);
    /// the cuBLAS half of the launch train (150 gemm_ex calls per encode)
    pub static GEMM_NS: AtomicU64 = AtomicU64::new(0);
    /// wall from entry to the point the caller can use the span
    pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);

    pub fn on() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_TICK_STATS").is_some())
    }

    pub fn record(rows: usize, im2col: u64, gemm: u64, total: u64) {
        N.fetch_add(1, Relaxed);
        ROWS.fetch_add(rows as u64, Relaxed);
        IM2COL_NS.fetch_add(im2col, Relaxed);
        GEMM_NS.fetch_add(gemm, Relaxed);
        ISSUE_NS.fetch_add(total.saturating_sub(im2col), Relaxed);
        TOTAL_NS.fetch_add(total, Relaxed);
        static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
        let mut last = LAST.lock().expect("witness clock");
        let due = last.is_none_or(|t| t.elapsed().as_secs() >= 5);
        if !due {
            return;
        }
        *last = Some(std::time::Instant::now());
        let n = N.swap(0, Relaxed).max(1);
        let (r, i, g, s, t) = (
            ROWS.swap(0, Relaxed),
            IM2COL_NS.swap(0, Relaxed),
            GEMM_NS.swap(0, Relaxed),
            ISSUE_NS.swap(0, Relaxed),
            TOTAL_NS.swap(0, Relaxed),
        );
        tracing::info!(
            "towerstats: {n} encodes, {:.0} rows avg - im2col {:.2} ms/enc, gemm {:.2} ms/enc, other-launch {:.2} ms/enc, total {:.2} ms/enc ({:.0} ms/s)",
            r as f64 / n as f64,
            i as f64 / n as f64 / 1e6,
            g as f64 / n as f64 / 1e6,
            (s - g.min(s)) as f64 / n as f64 / 1e6,
            t as f64 / n as f64 / 1e6,
            t as f64 / 5.0 / 1e6,
        );
    }
}

/// Every device plane `encode` touches, allocated once (see the module note).
///
/// The conv-stem planes are sized at the FIXED [`CONV_GROUP`] capacity even
/// though a short clip's group is `min(CONV_GROUP, chunks)`: the index tables
/// are built at capacity too, and a shorter group is exactly a prefix of them
/// (rows are chunk-major, and the always-zero pad row sits at the capacity
/// offset). That prefix property is what lets one set of tables and planes
/// serve every clip length - it is the same property the group loop already
/// relied on for its tail group.
pub(crate) struct TowerScratch {
    // conv stem, capacity-sized and never resized
    idx2: CudaSlice<u32>,
    idx3: CudaSlice<u32>,
    /// stage-1 im2col plane, uploaded per group (host-built from the mel)
    h1d: CudaSlice<f32>,
    h1: Vec<f32>,
    a1: CudaSlice<f32>,
    a2: CudaSlice<f32>,
    a3: CudaSlice<f32>,
    g2: CudaSlice<f32>,
    g3: CudaSlice<f32>,
    gtok: CudaSlice<f32>,
    // token domain, grown on demand
    rows_cap: usize,
    chunks_cap: usize,
    /// f16 staging for every GEMM input - the conv stem's widest gather plane
    /// dominates until a clip passes ~10k tokens (~13 min), so this is
    /// effectively fixed too.
    s16: CudaSlice<f16>,
    tok: CudaSlice<f32>,
    /// sinusoidal positions, PREBUILT: row j takes table row `j % W[3]`, so
    /// one resident plane serves every clip instead of a per-encode host
    /// build plus upload.
    pos: CudaSlice<f32>,
    x: CudaSlice<f32>,
    n: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    at: CudaSlice<f32>,
    up: CudaSlice<f32>,
    m: CudaSlice<f32>,
    /// one segment's projector output, copied into the clip's span
    seg_out: CudaSlice<f32>,
}

/// Host-side stopwatch around one launch. The tower's cost is the LAUNCH
/// TRAIN, not the GPU: the port moved every kernel below onto its own
/// CUDA stream and this number did not move (2.9 ms/enc either way), so the
/// split that matters is which API calls the time is actually in - a cuBLAS
/// `gemm_ex` is a heuristic lookup plus a launch, several times the cost of
/// one of our own kernel launches, and the tower issues ~150 of them.
#[inline]
fn timed<T>(acc: &mut u64, on: bool, f: impl FnOnce() -> T) -> T {
    if !on {
        return f();
    }
    let t = std::time::Instant::now();
    let r = f();
    *acc += t.elapsed().as_nanos() as u64;
    r
}

impl TowerScratch {
    fn new(
        exec: &GpuExecutor,
        ch: usize,
        embd: usize,
        ffn: usize,
        out_dim: usize,
        pos_table: &[f32],
    ) -> Result<Self, GpuModelError> {
        let g = CONV_GROUP;
        let (rows1, rows2, rows3) = (g * H[1] * W[1], g * H[2] * W[2], g * H[3] * W[3]);
        // the +1 row on a1/a2 is the gather's always-zero pad source; it is
        // zeroed by this one alloc and no kernel ever writes it again
        let me = Self {
            idx2: exec.to_device_u32(&AudioTower::im2col_idx(g, 2, false, rows1))?,
            idx3: exec.to_device_u32(&AudioTower::im2col_idx(g, 3, true, rows2))?,
            h1d: exec.alloc(rows1 * 9)?,
            h1: vec![0f32; rows1 * 9],
            a1: exec.alloc((rows1 + 1) * ch)?,
            a2: exec.alloc((rows2 + 1) * ch)?,
            a3: exec.alloc(rows3 * ch)?,
            g2: exec.alloc(rows2 * 9 * ch)?,
            g3: exec.alloc(rows3 * 9 * ch)?,
            gtok: exec.alloc(g * W[3] * embd)?,
            rows_cap: 0,
            chunks_cap: 0,
            s16: exec.alloc_f16(rows2 * 9 * ch)?,
            tok: exec.alloc(0)?,
            pos: exec.alloc(0)?,
            x: exec.alloc(0)?,
            n: exec.alloc(0)?,
            q: exec.alloc(0)?,
            k: exec.alloc(0)?,
            v: exec.alloc(0)?,
            at: exec.alloc(0)?,
            up: exec.alloc(0)?,
            m: exec.alloc(0)?,
            seg_out: exec.alloc(SEGMENT_TOKENS * out_dim)?,
        };
        let mut me = me;
        me.grow(
            exec,
            SEGMENT_TOKENS,
            SEGMENT_TOKENS / W[3],
            embd,
            ffn,
            ch,
            pos_table,
        )?;
        Ok(me)
    }

    #[allow(clippy::too_many_arguments)]
    fn grow(
        &mut self,
        exec: &GpuExecutor,
        rows: usize,
        chunks: usize,
        embd: usize,
        ffn: usize,
        ch: usize,
        pos_table: &[f32],
    ) -> Result<(), GpuModelError> {
        self.tok = exec.alloc(chunks * W[3] * embd)?;
        // one resident copy of the periodic position plane
        let mut plane = vec![0f32; rows * embd];
        for j in 0..rows {
            let p = j % W[3];
            plane[j * embd..(j + 1) * embd].copy_from_slice(&pos_table[p * embd..(p + 1) * embd]);
        }
        self.pos = exec.to_device(&plane)?;
        for buf in [
            &mut self.x,
            &mut self.n,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.at,
            &mut self.m,
        ] {
            *buf = exec.alloc(rows * embd)?;
        }
        self.up = exec.alloc(rows * ffn)?;
        let want = (rows * ffn.max(embd)).max(CONV_GROUP * H[2] * W[2] * 9 * ch);
        if want > self.s16.len() {
            self.s16 = exec.alloc_f16(want)?;
        }
        self.rows_cap = rows;
        self.chunks_cap = chunks;
        Ok(())
    }

    /// Resident bytes - the VRAM report and the pool sizing both need this to
    /// be an honest number rather than a surprise at the first encode.
    pub(crate) fn bytes(&self) -> usize {
        4 * (self.h1d.len()
            + self.a1.len()
            + self.a2.len()
            + self.a3.len()
            + self.g2.len()
            + self.g3.len()
            + self.gtok.len()
            + self.tok.len()
            + self.pos.len()
            + self.x.len()
            + self.n.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.at.len()
            + self.up.len()
            + self.m.len()
            + self.seg_out.len())
            + 2 * self.s16.len()
            + 4 * (self.idx2.len() + self.idx3.len())
    }
}

pub struct AudioTower {
    exec: Arc<GpuExecutor>,
    embd: usize,
    n_heads: usize,
    head_dim: usize,
    ffn: usize,
    /// conv channel width (480 from the file).
    ch: usize,
    eps: f32,
    /// LLM embedding width the projector emits (2048).
    pub out_dim: usize,
    conv1_w: HalfTensor,
    conv1_b: CudaSlice<f32>,
    conv2_w: HalfTensor,
    conv2_b: CudaSlice<f32>,
    conv3_w: HalfTensor,
    conv3_b: CudaSlice<f32>,
    conv_out_w: HalfTensor,
    blocks: Vec<ABlock>,
    post_ln_w: CudaSlice<f32>,
    post_ln_b: CudaSlice<f32>,
    mm1_w: HalfTensor,
    mm1_b: CudaSlice<f32>,
    mm2_w: HalfTensor,
    mm2_b: CudaSlice<f32>,
    /// Resident activation planes - see the module note on why `encode` no
    /// longer allocates.
    sc: TowerScratch,
}

impl AudioTower {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuModelError> {
        let meta = &map.gguf().metadata;
        let proj = meta
            .get("clip.audio.projector_type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if proj != "qwen3a" {
            return Err(GpuModelError::MissingMeta(format!(
                "mmproj is not a Qwen3-ASR audio tower (projector_type '{proj}', want 'qwen3a')"
            )));
        }
        let u = |k: &str| {
            meta.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let n_layers = u("clip.audio.block_count")? as usize;
        let embd = u("clip.audio.embedding_length")? as usize;
        let n_heads = u("clip.audio.attention.head_count")? as usize;
        let ffn = u("clip.audio.feed_forward_length")? as usize;
        let out_dim = u("clip.audio.projection_dim")? as usize;
        let n_mel = u("clip.audio.num_mel_bins")? as usize;
        assert_eq!(n_mel, N_MEL, "mel geometry mismatch");
        let eps = meta
            .get("clip.audio.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-5);

        let e = exec.clone();
        let dt = |name: &str| -> Result<HalfTensor, GpuModelError> { Ok(e.upload_f16(map, name)?) };
        let e2 = exec.clone();
        let vec1 = |name: &str| -> Result<CudaSlice<f32>, GpuModelError> {
            let (host, _) = host_f32(map, name)?;
            Ok(e2.to_device(&host)?)
        };

        // conv1: file k-order per out-channel is [in=1][ky][kx] - already the
        // (ky,kx) order the host stage-1 im2col emits.
        let mut conv1_w = dt("a.conv2d.1.weight")?;
        let ch = conv1_w.dims[3];
        conv1_w.dims = vec![9, ch];
        // conv2/conv3: permute k (c,ky,kx) -> (ky,kx,c) to match the gather.
        let e3 = exec.clone();
        let permute_conv = move |name: &str| -> Result<HalfTensor, GpuModelError> {
            let (w, dims) = host_f32(map, name)?;
            let (cin, cout) = (dims[2], dims[3]);
            let k = 9 * cin;
            let mut out = vec![0f32; k * cout];
            for o in 0..cout {
                for c in 0..cin {
                    for kyx in 0..9 {
                        out[o * k + kyx * cin + c] = w[o * k + c * 9 + kyx];
                    }
                }
            }
            Ok(HalfTensor {
                buf: e3.to_device_f16(&out, name)?,
                dims: vec![k, cout],
            })
        };
        let conv2_w = permute_conv("a.conv2d.2.weight")?;
        let conv3_w = permute_conv("a.conv2d.3.weight")?;
        // conv_out: permute k (c, h3) -> (h3, c) so its input is a contiguous
        // view of conv3's (w3, h3)-ordered rows.
        let conv_out_w = {
            let (w, dims) = host_f32(map, "a.conv_out.weight")?;
            let (k, cout) = (dims[0], dims[1]);
            let h3 = k / ch;
            let mut out = vec![0f32; k * cout];
            for o in 0..cout {
                for c in 0..ch {
                    for f in 0..h3 {
                        out[o * k + f * ch + c] = w[o * k + c * h3 + f];
                    }
                }
            }
            HalfTensor {
                buf: exec.to_device_f16(&out, "a.conv_out.weight")?,
                dims: vec![k, cout],
            }
        };

        let (pos, pos_dims) = host_f32(map, "a.position_embd.weight")?;
        assert_eq!(pos_dims[0], embd, "position table width");

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let t = |s: &str| format!("a.blk.{i}.{s}");
            blocks.push(ABlock {
                ln1_w: vec1(&t("ln1.weight"))?,
                ln1_b: vec1(&t("ln1.bias"))?,
                wq: dt(&t("attn_q.weight"))?,
                bq: vec1(&t("attn_q.bias"))?,
                wk: dt(&t("attn_k.weight"))?,
                bk: vec1(&t("attn_k.bias"))?,
                wv: dt(&t("attn_v.weight"))?,
                bv: vec1(&t("attn_v.bias"))?,
                wo: dt(&t("attn_out.weight"))?,
                bo: vec1(&t("attn_out.bias"))?,
                ln2_w: vec1(&t("ln2.weight"))?,
                ln2_b: vec1(&t("ln2.bias"))?,
                up_w: dt(&t("ffn_up.weight"))?,
                up_b: vec1(&t("ffn_up.bias"))?,
                down_w: dt(&t("ffn_down.weight"))?,
                down_b: vec1(&t("ffn_down.bias"))?,
            });
        }

        let sc = TowerScratch::new(&exec, ch, embd, ffn, out_dim, &pos)?;
        let me = Self {
            exec,
            embd,
            n_heads,
            head_dim: embd / n_heads,
            ffn,
            ch,
            eps,
            out_dim,
            conv1_w,
            conv1_b: vec1("a.conv2d.1.bias")?,
            conv2_w,
            conv2_b: vec1("a.conv2d.2.bias")?,
            conv3_w,
            conv3_b: vec1("a.conv2d.3.bias")?,
            conv_out_w,
            blocks,
            post_ln_w: vec1("a.post_ln.weight")?,
            post_ln_b: vec1("a.post_ln.bias")?,
            mm1_w: dt("mm.a.mlp.1.weight")?,
            mm1_b: vec1("mm.a.mlp.1.bias")?,
            mm2_w: dt("mm.a.mlp.2.weight")?,
            mm2_b: vec1("mm.a.mlp.2.bias")?,
            sc,
        };
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            scratch_mib = me.sc.bytes() / (1 << 20),
            layers = n_layers,
            "qwen3-asr audio tower resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Build the tower from the aligner's HF safetensors  - the
    /// same graph the GGUF path serves, sourced from the upstream tensor
    /// names. Two deltas from `load`: dims come from `config.json` instead of
    /// clip metadata, and the sinusoidal position table is SYNTHESIZED (the
    /// GGUF converter bakes `a.position_embd`; the HF checkpoint computes it
    /// - whisper's log-timescale sinusoids over `a_max_pos` rows, and only
    ///   rows 0..13 are ever read since positions reset per chunk).
    pub(crate) fn load_st(
        exec: Arc<GpuExecutor>,
        st: &paddock_models::safetensors::ShardedSafetensors,
        cfg: &paddock_models::safetensors::AlignerConfig,
    ) -> Result<Self, GpuModelError> {
        use super::aligner::{bf16_to_f16, bf16_to_f32, st_bf16};
        assert_eq!(cfg.a_mels, N_MEL, "mel geometry mismatch");
        let (embd, ffn, ch) = (cfg.a_dmodel, cfg.a_ffn, cfg.a_ch);
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        let e = exec.clone();
        // HF Linear [out, in] row-major is byte-identical to the GGUF layout
        // the GEMMs read (dims [in, out], out-major rows of k) - direct
        // widen, no permute. bf16 -> f16 is exact (mantissa widens); range
        // clips are tallied and refused below like the decoder side.
        let mut bad = 0usize;
        // the tally rides as a param so `plane` holds no long-lived borrow
        let plane = |name: &str,
                     out: usize,
                     inn: usize,
                     bad: &mut usize|
         -> Result<HalfTensor, GpuModelError> {
            let (v, nb) = bf16_to_f16(st_bf16(st, name, &[out, inn])?);
            *bad += nb;
            Ok(HalfTensor {
                buf: e
                    .stream
                    .clone_htod(&v)
                    .map_err(drv)
                    .map_err(GpuModelError::from)?,
                dims: vec![inn, out],
            })
        };
        let e2 = exec.clone();
        let vecf = |name: &str, n: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            let v = bf16_to_f32(st_bf16(st, name, &[n])?);
            e2.stream
                .clone_htod(&v)
                .map_err(drv)
                .map_err(GpuModelError::from)
        };

        // conv1 [ch, 1, 3, 3]: per-out k is already the (ky, kx) order the
        // host stage-1 im2col emits.
        let conv1_w = {
            let (v, nb) = bf16_to_f16(st_bf16(
                st,
                "model.audio_tower.conv2d1.weight",
                &[ch, 1, 3, 3],
            )?);
            bad += nb;
            HalfTensor {
                buf: exec.stream.clone_htod(&v).map_err(drv)?,
                dims: vec![9, ch],
            }
        };
        // conv2/conv3 [ch, ch, 3, 3]: permute k (c, ky, kx) -> (ky, kx, c) to
        // match the device gather (same move as the GGUF loader).
        let e3 = exec.clone();
        let permute_conv = |name: &str, bad: &mut usize| -> Result<HalfTensor, GpuModelError> {
            let w = bf16_to_f32(st_bf16(st, name, &[ch, ch, 3, 3])?);
            let k = 9 * ch;
            let mut out = vec![0f32; k * ch];
            for o in 0..ch {
                for c in 0..ch {
                    for kyx in 0..9 {
                        let v = w[o * k + c * 9 + kyx];
                        if !v.is_finite() {
                            *bad += 1;
                        }
                        out[o * k + kyx * ch + c] = v;
                    }
                }
            }
            Ok(HalfTensor {
                buf: e3.to_device_f16(&out, name)?,
                dims: vec![k, ch],
            })
        };
        let conv2_w = permute_conv("model.audio_tower.conv2d2.weight", &mut bad)?;
        let conv3_w = permute_conv("model.audio_tower.conv2d3.weight", &mut bad)?;
        // conv_out [embd, 16*ch]: permute k (c, h3) -> (h3, c) so the input is
        // a contiguous view of conv3's (w3, h3)-ordered rows.
        let conv_out_w = {
            let k = H[3] * ch;
            let w = bf16_to_f32(st_bf16(
                st,
                "model.audio_tower.conv_out.weight",
                &[embd, k],
            )?);
            let h3 = H[3];
            let mut out = vec![0f32; k * embd];
            for o in 0..embd {
                for c in 0..ch {
                    for f in 0..h3 {
                        out[o * k + f * ch + c] = w[o * k + c * h3 + f];
                    }
                }
            }
            HalfTensor {
                buf: exec.to_device_f16(&out, "model.audio_tower.conv_out.weight")?,
                dims: vec![k, embd],
            }
        };

        // whisper sinusoids: row p = [sin(p·inv) | cos(p·inv)],
        // inv[i] = exp(-ln(10000)/(d/2-1) · i) - modeling_qwen3_asr.py's
        // SinusoidsPositionEmbedding, f32 like the reference.
        let half_d = embd / 2;
        let log_inc = (10000f32).ln() / (half_d - 1) as f32;
        let mut pos = vec![0f32; cfg.a_max_pos * embd];
        for p in 0..cfg.a_max_pos {
            for i in 0..half_d {
                let t = p as f32 * (-log_inc * i as f32).exp();
                pos[p * embd + i] = t.sin();
                pos[p * embd + half_d + i] = t.cos();
            }
        }

        let mut blocks = Vec::with_capacity(cfg.a_layers);
        for i in 0..cfg.a_layers {
            let t = |s: &str| format!("model.audio_tower.layers.{i}.{s}");
            blocks.push(ABlock {
                ln1_w: vecf(&t("self_attn_layer_norm.weight"), embd)?,
                ln1_b: vecf(&t("self_attn_layer_norm.bias"), embd)?,
                wq: plane(&t("self_attn.q_proj.weight"), embd, embd, &mut bad)?,
                bq: vecf(&t("self_attn.q_proj.bias"), embd)?,
                wk: plane(&t("self_attn.k_proj.weight"), embd, embd, &mut bad)?,
                bk: vecf(&t("self_attn.k_proj.bias"), embd)?,
                wv: plane(&t("self_attn.v_proj.weight"), embd, embd, &mut bad)?,
                bv: vecf(&t("self_attn.v_proj.bias"), embd)?,
                wo: plane(&t("self_attn.out_proj.weight"), embd, embd, &mut bad)?,
                bo: vecf(&t("self_attn.out_proj.bias"), embd)?,
                ln2_w: vecf(&t("final_layer_norm.weight"), embd)?,
                ln2_b: vecf(&t("final_layer_norm.bias"), embd)?,
                up_w: plane(&t("fc1.weight"), ffn, embd, &mut bad)?,
                up_b: vecf(&t("fc1.bias"), ffn)?,
                down_w: plane(&t("fc2.weight"), embd, ffn, &mut bad)?,
                down_b: vecf(&t("fc2.bias"), embd)?,
            });
        }

        let mm1_w = plane(
            "model.multi_modal_projector.linear_1.weight",
            embd,
            embd,
            &mut bad,
        )?;
        let mm2_w = plane(
            "model.multi_modal_projector.linear_2.weight",
            cfg.a_out_dim,
            embd,
            &mut bad,
        )?;
        if bad > 0 {
            return Err(GpuModelError::Unsupported(format!(
                "aligner tower: {bad} non-finite/out-of-range weight values"
            )));
        }

        let sc = TowerScratch::new(&exec, ch, embd, ffn, cfg.a_out_dim, &pos)?;
        let me = Self {
            exec,
            embd,
            n_heads: cfg.a_heads,
            head_dim: embd / cfg.a_heads,
            ffn,
            ch,
            // nn.LayerNorm default - the HF audio_config carries no eps
            eps: 1e-5,
            out_dim: cfg.a_out_dim,
            conv1_w,
            conv1_b: vecf("model.audio_tower.conv2d1.bias", ch)?,
            conv2_w,
            conv2_b: vecf("model.audio_tower.conv2d2.bias", ch)?,
            conv3_w,
            conv3_b: vecf("model.audio_tower.conv2d3.bias", ch)?,
            conv_out_w,
            blocks,
            post_ln_w: vecf("model.audio_tower.ln_post.weight", embd)?,
            post_ln_b: vecf("model.audio_tower.ln_post.bias", embd)?,
            mm1_w,
            mm1_b: vecf("model.multi_modal_projector.linear_1.bias", embd)?,
            mm2_w,
            mm2_b: vecf("model.multi_modal_projector.linear_2.bias", cfg.a_out_dim)?,
            sc,
        };
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            layers = cfg.a_layers,
            "aligner audio tower resident at f16 (safetensors source)"
        );
        Ok(me)
    }

    pub fn weight_bytes(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.wq.bytes()
                    + b.wk.bytes()
                    + b.wv.bytes()
                    + b.wo.bytes()
                    + b.up_w.bytes()
                    + b.down_w.bytes()
            })
            .sum();
        blk + self.conv1_w.bytes()
            + self.conv2_w.bytes()
            + self.conv3_w.bytes()
            + self.conv_out_w.bytes()
            + self.mm1_w.bytes()
            + self.mm2_w.bytes()
    }

    /// Host im2col index table for a k3/s2/p1 stage over `chunks` chunks:
    /// entry per (out_row, ky, kx) = source ROW in the [chunks*h_in*w_in]
    /// activation plane, or `zero_row` when out of range. `w_major` emits
    /// out rows in (w2, h2) order (the conv3 -> conv_out seam).
    fn im2col_idx(chunks: usize, stage: usize, w_major: bool, zero_row: usize) -> Vec<u32> {
        let (h_in, w_in) = (H[stage - 1], W[stage - 1]);
        let (h_out, w_out) = (H[stage], W[stage]);
        let mut idx = Vec::with_capacity(chunks * h_out * w_out * 9);
        for ch in 0..chunks {
            let n = h_out * w_out;
            for rr in 0..n {
                let (h2, w2) = if w_major {
                    (rr % h_out, rr / h_out)
                } else {
                    (rr / w_out, rr % w_out)
                };
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let hi = 2 * h2 + ky;
                        let wi = 2 * w2 + kx;
                        // p1: source coordinate is (hi-1, wi-1)
                        idx.push(if hi >= 1 && hi <= h_in && wi >= 1 && wi <= w_in {
                            ((ch * h_in + hi - 1) * w_in + wi - 1) as u32
                        } else {
                            zero_row as u32
                        });
                    }
                }
            }
        }
        idx
    }

    /// Encode one clip's mel features into LLM-space audio embeddings.
    ///
    /// `&mut self` only because the activation planes are resident state now
    ///  - the math is unchanged and every buffer is fully written
    ///    before it is read, so back-to-back encodes cannot see each other.
    pub fn encode(&mut self, mel: &MelFeatures) -> Result<AudioOutput, GpuModelError> {
        let exec = self.exec.clone();
        let n_frames = mel.n_frames;
        assert!(n_frames > 0);
        let chunks = n_frames.div_ceil(CHUNK_FRAMES);
        let n_tokens = audio_token_count(n_frames);
        let wit = witness::on();
        let t_enter = std::time::Instant::now();
        let (mut im2col_ns, mut gemm_ns) = (0u64, 0u64);
        // the one allocation left per encode: the span outlives the call (the
        // slot's registry holds it until its prefill rows are consumed)
        let mut d_out = exec.alloc(n_tokens * self.out_dim)?;
        // one SEGMENT at a time (see SEGMENT_TOKENS): its valid tokens are a
        // prefix of its chunks' rows - only the clip's last chunk is short
        let seg_chunks = SEGMENT_TOKENS / W[3];
        let mut c0 = 0usize;
        while c0 < chunks {
            let sc_chunks = seg_chunks.min(chunks - c0);
            let t0 = c0 * W[3];
            let rows = (sc_chunks * W[3]).min(n_tokens - t0);
            self.encode_segment(mel, c0, sc_chunks, rows, wit, &mut im2col_ns, &mut gemm_ns)?;
            exec.copy_region(
                &self.sc.seg_out,
                0,
                &mut d_out,
                t0 * self.out_dim,
                rows * self.out_dim,
            )?;
            c0 += sc_chunks;
        }

        // Debug tap: dump the projector output for oracle comparison against
        // the upstream transformers encoder (our ASR oracle tool).
        // Appends one [n_tokens, 2048] f32 block per encode call.
        if let Ok(path) = paddock_models::dev_var!("PADDOCK_ASR_DUMP_EMBD") {
            let host = exec.to_host_len(&d_out, n_tokens * self.out_dim)?;
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| crate::gpu::GpuError::Driver(format!("embd dump {path}: {e}")))?;
            let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bytes)
                .map_err(|e| crate::gpu::GpuError::Driver(format!("embd dump {path}: {e}")))?;
            tracing::info!(n_tokens, path, "dumped audio tower embeddings");
        }

        if wit {
            witness::record(
                n_tokens,
                im2col_ns,
                gemm_ns,
                t_enter.elapsed().as_nanos() as u64,
            );
        }
        Ok(AudioOutput {
            embd: d_out,
            n_tokens,
        })
    }

    /// Conv stem, 24 encoder layers and projector over chunks
    /// `[c0, c0 + sc_chunks)` - `rows` valid tokens - into `sc.seg_out`.
    #[allow(clippy::too_many_arguments)]
    fn encode_segment(
        &mut self,
        mel: &MelFeatures,
        c0: usize,
        sc_chunks: usize,
        rows: usize,
        wit: bool,
        im2col_ns: &mut u64,
        gemm_ns: &mut u64,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (e, ch) = (self.embd, self.ch);
        // ---- conv stem, in groups of up to CONV_GROUP chunks ----
        // Every plane and index table below is the resident CAPACITY one and
        // a short group reads a prefix of it.
        let g = CONV_GROUP.min(sc_chunks);
        let sc = &mut self.sc;
        let mut done = 0usize;
        while done < sc_chunks {
            let gc = g.min(sc_chunks - done);
            let (r1, r2, r3) = (gc * H[1] * W[1], gc * H[2] * W[2], gc * H[3] * W[3]);
            // stage 1 im2col on host straight from the mel frames. `data` is
            // chunk-aligned and carries the upstream pad content past
            // `n_frames` (silence log-mel / feature zeros - Not unconditional
            // zeros; see MelFeatures), so every staged frame reads real data.
            let mel_frames = mel.data.len() / N_MEL;
            let t_im = std::time::Instant::now();
            for c in 0..gc {
                for h1 in 0..H[1] {
                    for w1 in 0..W[1] {
                        let row = (c * H[1] + h1) * W[1] + w1;
                        for ky in 0..3usize {
                            for kx in 0..3usize {
                                let (hi, wi) = (2 * h1 + ky, 2 * w1 + kx);
                                let v = if hi >= 1 && hi <= H[0] && wi >= 1 && wi <= W[0] {
                                    let frame = (c0 + done + c) * CHUNK_FRAMES + wi - 1;
                                    debug_assert!(frame < mel_frames, "mel not chunk-aligned");
                                    mel.data[frame * N_MEL + (hi - 1)]
                                } else {
                                    0.0
                                };
                                sc.h1[row * 9 + ky * 3 + kx] = v;
                            }
                        }
                    }
                }
            }
            if wit {
                *im2col_ns += t_im.elapsed().as_nanos() as u64;
            }
            exec.upload_f32(&sc.h1[..r1 * 9], &mut sc.h1d)?;
            exec.convert_f32_f16(&sc.h1d, &mut sc.s16, r1 * 9)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&self.conv1_w, &sc.s16, &mut sc.a1, r1)
            })?;
            exec.bias_add(&mut sc.a1, &self.conv1_b, r1, ch)?;
            exec.gelu_erf(&mut sc.a1, r1 * ch)?;

            exec.gather_rows_avg(&sc.a1, &sc.idx2, &mut sc.g2, r2 * 9, 1, ch)?;
            exec.convert_f32_f16(&sc.g2, &mut sc.s16, r2 * 9 * ch)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&self.conv2_w, &sc.s16, &mut sc.a2, r2)
            })?;
            exec.bias_add(&mut sc.a2, &self.conv2_b, r2, ch)?;
            exec.gelu_erf(&mut sc.a2, r2 * ch)?;

            exec.gather_rows_avg(&sc.a2, &sc.idx3, &mut sc.g3, r3 * 9, 1, ch)?;
            exec.convert_f32_f16(&sc.g3, &mut sc.s16, r3 * 9 * ch)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&self.conv3_w, &sc.s16, &mut sc.a3, r3)
            })?;
            exec.bias_add(&mut sc.a3, &self.conv3_b, r3, ch)?;
            exec.gelu_erf(&mut sc.a3, r3 * ch)?;

            // conv_out reads conv3's (w3, h3)-ordered rows as [gc*13, 16*ch]
            let tr = gc * W[3];
            exec.convert_f32_f16(&sc.a3, &mut sc.s16, tr * H[3] * ch)?;
            // token rows for this group land at their offset in the segment
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&self.conv_out_w, &sc.s16, &mut sc.gtok, tr)
            })?;
            exec.copy_region(&sc.gtok, 0, &mut sc.tok, done * W[3] * e, tr * e)?;
            done += gc;
        }

        // ---- sinusoidal positions, reset 0..12 per chunk; a segment starts on
        // a chunk boundary, so j % 13 is the within-chunk index here too (the
        // plane is resident and prebuilt - see TowerScratch::grow) ----
        exec.copy_region(&sc.tok, 0, &mut sc.x, 0, rows * e)?;
        exec.add(&mut sc.x, &sc.pos, rows * e)?;

        // ---- 24 encoder layers over the segment's token stream ----
        let ffn = self.ffn;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        for blk in &self.blocks {
            exec.layernorm(&sc.x, &blk.ln1_w, &blk.ln1_b, &mut sc.n, rows, e, self.eps)?;
            exec.convert_f32_f16(&sc.n, &mut sc.s16, rows * e)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&blk.wq, &sc.s16, &mut sc.q, rows)
            })?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&blk.wk, &sc.s16, &mut sc.k, rows)
            })?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&blk.wv, &sc.s16, &mut sc.v, rows)
            })?;
            exec.bias_add(&mut sc.q, &blk.bq, rows, e)?;
            exec.bias_add(&mut sc.k, &blk.bk, rows, e)?;
            exec.bias_add(&mut sc.v, &blk.bv, rows, e)?;
            let mut off = 0usize;
            while off < rows {
                let n_w = WINDOW_TOKENS.min(rows - off);
                exec.vision_attn_at(
                    &sc.q,
                    &sc.k,
                    &sc.v,
                    &mut sc.at,
                    off,
                    n_w,
                    self.n_heads,
                    self.head_dim,
                    scale,
                )?;
                off += n_w;
            }
            exec.convert_f32_f16(&sc.at, &mut sc.s16, rows * e)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&blk.wo, &sc.s16, &mut sc.n, rows)
            })?;
            exec.bias_add(&mut sc.n, &blk.bo, rows, e)?;
            exec.add(&mut sc.x, &sc.n, rows * e)?;

            exec.layernorm(&sc.x, &blk.ln2_w, &blk.ln2_b, &mut sc.n, rows, e, self.eps)?;
            exec.convert_f32_f16(&sc.n, &mut sc.s16, rows * e)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&blk.up_w, &sc.s16, &mut sc.up, rows)
            })?;
            exec.bias_add(&mut sc.up, &blk.up_b, rows, ffn)?;
            exec.gelu_erf(&mut sc.up, rows * ffn)?;
            exec.convert_f32_f16(&sc.up, &mut sc.s16, rows * ffn)?;
            timed(gemm_ns, wit, || {
                exec.matvec_batch_f16(&blk.down_w, &sc.s16, &mut sc.n, rows)
            })?;
            exec.bias_add(&mut sc.n, &blk.down_b, rows, e)?;
            exec.add(&mut sc.x, &sc.n, rows * e)?;
        }

        // ---- post-LN + projector ----
        exec.layernorm(
            &sc.x,
            &self.post_ln_w,
            &self.post_ln_b,
            &mut sc.n,
            rows,
            e,
            self.eps,
        )?;
        exec.convert_f32_f16(&sc.n, &mut sc.s16, rows * e)?;
        timed(gemm_ns, wit, || {
            exec.matvec_batch_f16(&self.mm1_w, &sc.s16, &mut sc.m, rows)
        })?;
        exec.bias_add(&mut sc.m, &self.mm1_b, rows, e)?;
        exec.gelu_erf(&mut sc.m, rows * e)?;
        exec.convert_f32_f16(&sc.m, &mut sc.s16, rows * e)?;
        timed(gemm_ns, wit, || {
            exec.matvec_batch_f16(&self.mm2_w, &sc.s16, &mut sc.seg_out, rows)
        })?;
        exec.bias_add(&mut sc.seg_out, &self.mm2_b, rows, self.out_dim)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The index-table construction is the whole conv correctness story on
    /// the gather side - pin its edges off-GPU.
    #[test]
    fn im2col_index_edges() {
        // stage 2, 1 chunk: out (0,0) reads in (-1..2, -1..2) => 4 pad + rows
        let idx = AudioTower::im2col_idx(1, 2, false, 99999);
        assert_eq!(idx.len(), 32 * 25 * 9);
        // out (0,0), k (0,0) is pad; k (1,1) is in-row (0,0); k (2,2) is (1,1)
        assert_eq!(idx[0], 99999);
        assert_eq!(idx[4], 0);
        assert_eq!(idx[8], 51); // k (2,2) reads in (1,1) = row 1*50 + 1
        // last out (31,24): center reads (62,48)
        let last = (32 * 25 - 1) * 9;
        assert_eq!(idx[last + 4], (62 * 50 + 48) as u32);
        // bottom-right corner k (2,2) -> (63,49) valid (64x50 input)
        assert_eq!(idx[last + 8], (63 * 50 + 49) as u32);
    }

    #[test]
    fn w_major_row_order_feeds_conv_out_contiguously() {
        // stage 3, 1 chunk, w_major: row rr = w3*16 + h3. Row 0 is (h=0,w=0),
        // row 1 is (h=1,w=0) - 16 consecutive rows share one w3 = one token.
        let idx = AudioTower::im2col_idx(1, 3, true, 7777);
        assert_eq!(idx.len(), 16 * 13 * 9);
        // rows 0 and 1 (h3 0 and 1 at w3=0) read inputs 2 apart in h
        let c00 = idx[4]; // (h=0,w=0) center = in (0? ) - p1: (2*0+1-1, 2*0+1-1) = (0,0)
        let c10 = idx[9 + 4]; // (h=1,w=0) center = (2,0)
        assert_eq!(c00, 0);
        assert_eq!(c10, (2 * 25) as u32);
    }
}
