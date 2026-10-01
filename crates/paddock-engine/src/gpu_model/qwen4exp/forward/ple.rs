//! Flash-Next PLE n-gram staging: the table rows a walk's tokens hash to,
//! widened into `d_emb` before the walk runs.
//!
//! The ids are a pure function of the token stream, so the host hashes them
//! (integer arithmetic, no table memory). The rows come off one of two lanes:
//!
//! * DEVICE table (a discrete card that can hold it, slot 532): the whole
//!   table uploaded once and gathered on the device.
//! * HOST mapping (a unified-memory die, where a device copy would hold the
//!   table twice, or a card that cannot hold it): the table stays a file
//!   mapping, and beside a loaded model only part of it is page-cache
//!   resident (GB10 kept 2-47% of it however it was warmed). The host does
//!   what only the host can - hint every page the walk needs at once, so the
//!   misses go to the disk together instead of one at a time, then copy each
//!   row's bytes as the table stores them into one compact plane, split over
//!   threads - and the device decodes (slot 697).
//!
//! Until 2026-09-30 the host lane was one serial loop that also decoded: per
//! row it formatted the shard's tensor name, looked it up, widened the row to
//! f32 on the CPU and uploaded 42 MB per 4096-row walk; and its n-gram hash
//! re-scanned the stream from the start for every position, which is
//! quadratic in the prompt.
//!
//! Probed on GB10 before choosing this shape (the pack's
//! ple_rows_gb10_bench, 2026-09-30): the GPU can read the mapping itself
//! (ATS), 11.6 ms for a walk's 65K rows when their pages are mapped - but a
//! row whose page is not faults per page from the device, 17-20 s for 65K
//! cold rows even after hinting. This lane pays 37 ms warm / 57 ms cold on
//! 8 threads. The rows are routinely cold here, so the host keeps the
//! page-in. Served, a 16.6K-token prompt's cold TTFT went 11.95 -> 11.67 s
//! (32K x 2); the quadratic hash grew with the prompt, so the long prompts
//! gain the most.

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::{PleSource, Qwen4ExpGpu, Scratch};
use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen4exp::PleW;
use paddock_kernels::reference::qwen4exp as rq;
use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MapAccess;
use paddock_models::qwen4exp::Qwen4ExpConfig;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

/// Slot 697's row formats (qwen4exp.cuh).
const FMT_E4M3: u32 = 0;
const FMT_NVFP4: u32 = 1;

/// The host lane's view of one PLE layer's table: where each row sits in
/// the mapping and how many bytes it is, resolved once at load so a walk
/// formats no tensor names.
pub(super) struct PleHostTable {
    /// slot 697's row format
    fmt: u32,
    /// the GGUF table's type - the host decode for a pack without slot 697
    gguf_ty: Option<GgmlType>,
    width: usize,
    /// bytes of one row in its plane, and of its NVFP4 group scales in theirs
    row: usize,
    scale_row: usize,
    /// rows per plane; the planes in row-id order, each with its NVFP4
    /// group-scale plane (the safetensors table is 128 shards, the GGUF one
    /// tensor)
    per_plane: usize,
    planes: Vec<(String, Option<String>)>,
}

impl PleHostTable {
    /// Bytes one staged row takes: the row, then its group scales.
    pub(super) fn row_bytes(&self) -> usize {
        self.row + self.scale_row
    }

    /// Rows the table holds - the bound every hashed id is checked against.
    pub(super) fn rows(&self) -> usize {
        self.per_plane * self.planes.len()
    }

    /// PLE layer `li`'s table in a safetensors checkpoint: `ngram_split`
    /// shards of e4m3 rows (the FP8 export) or of NVFP4 rows, each with a
    /// plane of e4m3 group scales beside it.
    pub(super) fn of_st(
        st: &ShardedSafetensors,
        c: &Qwen4ExpConfig,
        li: usize,
    ) -> Result<Self, GpuModelError> {
        let width = c.ple_embed / c.ple_heads();
        let emb = format!("model.language_model.layers.{li}.ple.ple_embedding.ngram_embedding");
        let bad = |what: String| GpuModelError::Unsupported(format!("PLE table: {what}"));
        let mut planes = Vec::with_capacity(c.ngram_split);
        let mut layout: Option<(bool, usize)> = None;
        for sh in 0..c.ngram_split {
            let name = format!("{emb}.shard_{sh}.weight");
            let (t, _) = st
                .bytes(&name)
                .ok_or_else(|| bad(format!("{name} missing")))?;
            let nvfp4 = match t.dtype {
                StDtype::F8E4m3 => false,
                StDtype::U8 => true,
                other => return Err(bad(format!("{name}: dtype {other:?}, want F8E4m3 or U8"))),
            };
            let want = if nvfp4 { width / 2 } else { width };
            if t.shape.len() != 2 || t.shape[1] != want {
                return Err(bad(format!(
                    "{name}: shape {:?}, want [rows, {want}]",
                    t.shape
                )));
            }
            // one layout and one row count for every shard: a row id splits
            // into (shard, row) by a division, so a re-sharded checkpoint
            // must refuse here rather than read the wrong row
            let rows = t.shape[0];
            match layout {
                None => layout = Some((nvfp4, rows)),
                Some(l) if l == (nvfp4, rows) => {}
                Some(_) => return Err(bad(format!("{name}: shards differ in layout or rows"))),
            }
            let scales = if nvfp4 {
                let s = format!("{name}_scale");
                let (ts, _) = st.bytes(&s).ok_or_else(|| bad(format!("{s} missing")))?;
                if ts.dtype != StDtype::F8E4m3 || ts.shape != [rows, width / 16] {
                    return Err(bad(format!(
                        "{s}: {:?} {:?}, want F8E4m3 [{rows}, {}]",
                        ts.dtype,
                        ts.shape,
                        width / 16
                    )));
                }
                Some(s)
            } else {
                None
            };
            planes.push((name, scales));
        }
        let (nvfp4, per_plane) = layout.ok_or_else(|| bad("no shards".into()))?;
        Ok(Self {
            fmt: if nvfp4 { FMT_NVFP4 } else { FMT_E4M3 },
            gguf_ty: None,
            width,
            row: if nvfp4 { width / 2 } else { width },
            scale_row: if nvfp4 { width / 16 } else { 0 },
            per_plane,
            planes,
        })
    }

    /// The GGUF lane's table: one tensor of `rows` rows of `row_bytes`.
    pub(super) fn of_gguf(
        name: &str,
        ty: GgmlType,
        row_bytes: usize,
        rows: usize,
        width: usize,
    ) -> Result<Self, GpuModelError> {
        let fmt = match ty {
            GgmlType::Q8_0 => 2,
            GgmlType::Q4_0 => 3,
            GgmlType::Iq4Nl => 4,
            GgmlType::F16 => 5,
            GgmlType::Bf16 => 6,
            GgmlType::F32 => 7,
            other => {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: PLE table type {other:?} has no row decoder"
                )));
            }
        };
        Ok(Self {
            fmt,
            gguf_ty: Some(ty),
            width,
            row: row_bytes,
            scale_row: 0,
            per_plane: rows,
            planes: vec![(name.to_owned(), None)],
        })
    }

    /// Copy the rows `ids` name into `out` (`ids.len() * row_bytes()`, in
    /// the ids' order), every page under them hinted first. Split over
    /// threads for a prompt walk; a decode tick's few rows stay on the
    /// caller's. Returns the thread count (the phase line prints it).
    pub(super) fn gather(
        &self,
        src: &PleSource,
        ids: &[u32],
        out: &mut [u8],
    ) -> Result<usize, GpuModelError> {
        let rb = self.row_bytes();
        if out.len() != ids.len() * rb {
            return Err(GpuModelError::Unsupported(format!(
                "PLE gather: {} bytes of staging for {} rows of {rb}",
                out.len(),
                ids.len()
            )));
        }
        // checked before any read: a bad id would index past a plane
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.rows()) {
            return Err(GpuModelError::Unsupported(format!(
                "PLE row {id} outside the {}-row table",
                self.rows()
            )));
        }
        let mut planes = Vec::with_capacity(self.planes.len());
        for (rows, scales) in &self.planes {
            let r = plane_bytes(src, rows)?;
            let s = match scales {
                Some(s) => plane_bytes(src, s)?,
                None => &[][..],
            };
            if r.len() < self.per_plane * self.row || s.len() < self.per_plane * self.scale_row {
                return Err(GpuModelError::Unsupported(format!(
                    "{rows}: plane shorter than {} rows",
                    self.per_plane
                )));
            }
            planes.push((r, s));
        }
        let threads = gather_threads(ids.len());
        if threads == 1 {
            self.gather_part(src, &planes, ids, out);
            return Ok(1);
        }
        let per = ids.len().div_ceil(threads);
        std::thread::scope(|s| {
            for (ids, out) in ids.chunks(per).zip(out.chunks_mut(per * rb)) {
                let planes = &planes;
                s.spawn(move || self.gather_part(src, planes, ids, out));
            }
        });
        Ok(threads)
    }

    /// One thread's rows: hint them all (one call per plane touched), then
    /// copy them. The hint queues every miss's page-in at once, so the copy
    /// waits for the slowest page rather than for their sum.
    fn gather_part(&self, src: &PleSource, planes: &[(&[u8], &[u8])], ids: &[u32], out: &mut [u8]) {
        let (row, srow, rb) = (self.row, self.scale_row, self.row_bytes());
        let mut hint: Vec<Vec<(usize, usize)>> = vec![Vec::new(); planes.len()];
        let mut hint_s: Vec<Vec<(usize, usize)>> = vec![Vec::new(); planes.len()];
        for &id in ids {
            let (p, r) = (id as usize / self.per_plane, id as usize % self.per_plane);
            hint[p].push((r * row, row));
            if srow > 0 {
                hint_s[p].push((r * srow, srow));
            }
        }
        for (p, (rows, scales)) in self.planes.iter().enumerate() {
            if !hint[p].is_empty() {
                advise(src, rows, &hint[p]);
            }
            if let Some(s) = scales
                && !hint_s[p].is_empty()
            {
                advise(src, s, &hint_s[p]);
            }
        }
        for (&id, dst) in ids.iter().zip(out.chunks_exact_mut(rb)) {
            let (p, r) = (id as usize / self.per_plane, id as usize % self.per_plane);
            let (rows, scales) = planes[p];
            dst[..row].copy_from_slice(&rows[r * row..(r + 1) * row]);
            dst[row..].copy_from_slice(&scales[r * srow..(r + 1) * srow]);
        }
    }

    /// The staged rows decoded on the host - for a pack without slot 697.
    /// The same arithmetic as the device decode, so the same bits.
    fn decode_host(&self, raw: &[u8], scale: f32) -> Vec<f32> {
        let (rb, w) = (self.row_bytes(), self.width);
        let mut out = vec![0f32; raw.len() / rb * w];
        for (row, o) in raw.chunks_exact(rb).zip(out.chunks_exact_mut(w)) {
            match self.gguf_ty {
                Some(ty) => ple_row_dequant(ty, row, o),
                None if self.fmt == FMT_E4M3 => {
                    for (v, &b) in o.iter_mut().zip(row) {
                        *v = rq::e4m3_to_f32(b) * scale;
                    }
                }
                None => {
                    let view = paddock_models::modelopt::Nvfp4View {
                        packed: &row[..self.row],
                        scales: &row[self.row..],
                        scale2: scale,
                        n: 1,
                        k: w,
                    };
                    o.copy_from_slice(&view.dequant_row_f32(0));
                }
            }
        }
        out
    }
}

/// A plane's bytes in the mapping.
fn plane_bytes<'a>(src: &'a PleSource, name: &str) -> Result<&'a [u8], GpuModelError> {
    match src {
        PleSource::St(st) => st
            .bytes(name)
            .map(|(_, b)| b)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing"))),
        PleSource::Gguf { map, .. } => map
            .tensor_bytes(name)
            .map(|(_, b)| b)
            .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}"))),
    }
}

/// Queue the page-in of `ranges` of plane `name`. Best effort: a refused
/// hint changes when the bytes arrive, never what they are.
fn advise(src: &PleSource, name: &str, ranges: &[(usize, usize)]) {
    match src {
        PleSource::St(st) => {
            let _ = st.advise_tensor(name, MapAccess::WillNeed, ranges);
        }
        PleSource::Gguf { map, .. } => {
            let _ = map.advise_tensor(name, MapAccess::WillNeed, ranges);
        }
    }
}

/// Threads for a gather of `rows` rows: one below a prompt walk's size (a
/// decode tick's 16 rows a slot cost less than a spawn), else up to 8 - a
/// 4096-row walk is 65K rows, and the page-ins behind the hints are what
/// the threads wait on.
fn gather_threads(rows: usize) -> usize {
    static CPUS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let cpus = *CPUS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    if rows < 2048 {
        1
    } else {
        cpus.clamp(1, 8).min(rows / 1024)
    }
}

impl Qwen4ExpGpu {
    /// Decode the host lane's staged PLE rows on the host instead of on the
    /// device (a gate instrument: the device decode must give the same bits).
    /// Returns whether any layer reads its table on the host lane.
    pub fn set_ple_host_decode(&mut self, on: bool) -> bool {
        self.ple_host_decode = on;
        self.ple_host.iter().any(Option::is_some)
    }

    /// Stage a walk's PLE rows into `d_emb`. `runs` are (slot, stream index
    /// of its first row, rows) in walk-row order; each hashes its own
    /// stream, which must already hold the rows (the stream carries the
    /// 2-token EOS priming on its front, so a prompt row `i` is index
    /// `i + 2`).
    pub(super) fn stage_ple(
        &mut self,
        runs: &[(usize, usize, usize)],
    ) -> Result<(), GpuModelError> {
        let heads = self.cfg.ple_heads();
        let total: usize = runs.iter().map(|r| r.2).sum();
        for li in 0..self.cfg.n_layer {
            let Some(ple) = self.layers[li].ple.as_ref() else {
                continue;
            };
            let t0 = std::time::Instant::now();
            let mut ids = Vec::with_capacity(total * heads);
            for &(sl, first, n) in runs {
                ids.extend(ple_row_ids(&self.cfg, ple, &self.stream[sl], first, n)?);
            }
            if let Some(tab) = ple.table.as_ref() {
                stage_ple_device(&self.exec, &self.cfg, ple, tab, &ids, &mut self.sc)?;
                continue;
            }
            let host = self.ple_host[li].as_ref().ok_or_else(|| {
                GpuModelError::Unsupported(format!("layer {li}: PLE table on neither lane"))
            })?;
            let t_ids = t0.elapsed();
            let rows = ids.len();
            let rb = host.row_bytes();
            self.ple_stage.clear();
            self.ple_stage.resize(rows * rb, 0);
            let threads = host.gather(&self.st, &ids, &mut self.ple_stage)?;
            let t_gather = t0.elapsed();
            if self.exec.has_q4x_ple_rows() && !self.ple_host_decode {
                self.exec.upload_u8(&self.ple_stage, &mut self.ple_raw)?;
                self.exec.q4x_ple_rows(
                    &self.ple_raw,
                    &mut self.sc.d_emb,
                    host.fmt,
                    ple.table_scale,
                    rows,
                    host.width,
                    rb,
                )?;
            } else {
                let emb = host.decode_host(&self.ple_stage, ple.table_scale);
                self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
            }
            if super::phase_ms_on() && total > self.slots {
                eprintln!(
                    "[q4x-phase] ple-stage n={total} rows {rows}: hash {:.2} gather {:.2} \
                     ({threads} threads) upload {:.2} ms",
                    t_ids.as_secs_f64() * 1e3,
                    (t_gather - t_ids).as_secs_f64() * 1e3,
                    (t0.elapsed() - t_gather).as_secs_f64() * 1e3,
                );
            }
        }
        Ok(())
    }
}

/// Bytes one `width`-wide row occupies in the host row decoder's types.
/// `None` for a type it does not decode.
pub(in crate::gpu_model::qwen4exp) fn ple_row_bytes(ty: GgmlType, width: usize) -> Option<usize> {
    let blocks32 = |bytes: usize| width.is_multiple_of(32).then_some(width / 32 * bytes);
    match ty {
        GgmlType::F32 => Some(width * 4),
        GgmlType::F16 | GgmlType::Bf16 => Some(width * 2),
        GgmlType::Q8_0 => blocks32(34),
        GgmlType::Q4_0 | GgmlType::Iq4Nl => blocks32(18),
        _ => None,
    }
}

/// The IQ4_NL codebook (ggml-common.h, MIT).
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Decode one table row (`row.len() == ple_row_bytes(ty, out.len())`).
fn ple_row_dequant(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = half::f16::from_le_bytes(*c).to_f32();
            }
        }
        GgmlType::Bf16 => {
            for (o, c) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = f32::from_bits((u16::from_le_bytes(*c) as u32) << 16);
            }
        }
        GgmlType::Q8_0 => {
            for (blk, o) in row
                .as_chunks::<34>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
            {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..32 {
                    o[j] = (blk[2 + j] as i8) as f32 * d;
                }
            }
        }
        GgmlType::Q4_0 => {
            for (blk, o) in row
                .as_chunks::<18>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
            {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..16 {
                    let q = blk[2 + j];
                    o[j] = ((q & 0xf) as i32 - 8) as f32 * d;
                    o[j + 16] = ((q >> 4) as i32 - 8) as f32 * d;
                }
            }
        }
        GgmlType::Iq4Nl => {
            for (blk, o) in row
                .as_chunks::<18>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<32>().0.iter_mut())
            {
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for j in 0..16 {
                    let q = blk[2 + j];
                    o[j] = KVALUES_IQ4NL[(q & 0xf) as usize] as f32 * d;
                    o[j + 16] = KVALUES_IQ4NL[(q >> 4) as usize] as f32 * d;
                }
            }
        }
        _ => unreachable!("ple_row_bytes gated the type"),
    }
}

/// The n-gram row ids for `n` consecutive positions of one request's token
/// stream, starting at stream index `first` - `[n, ple_heads]`, GLOBAL ids
/// (each head's table offset already folded in).
///
/// Pure integer arithmetic on the token ids: it touches no table memory, which
/// is exactly why the hash stays on the host while the rows are read off the
/// table. `rq::ple_window`'s previous-EOS scan is O(i) per position (the host
/// lane called it per row until 2026-09-30: quadratic in the prompt), so the
/// running cursor here is what keeps a long prefill linear.
fn ple_row_ids(
    c: &Qwen4ExpConfig,
    ple: &PleW,
    stream: &[i64],
    first: usize,
    n: usize,
) -> Result<Vec<u32>, GpuModelError> {
    let hpn = c.heads_per_ngram;
    let heads = c.ple_heads();
    let eos = c.bos_id as i64;
    if first + n > stream.len() {
        return Err(GpuModelError::Unsupported(format!(
            "ple ids: {n} rows at {first} but the stream holds {}",
            stream.len()
        )));
    }
    // the last EOS before the first row, looked for from the back: a
    // decode tick hashes one row at the end of the whole stream
    let mut prev_eos: i64 = stream[..first]
        .iter()
        .rposition(|&t| t == eos)
        .map_or(-1, |j| j as i64);
    let mut out = vec![0u32; n * heads];
    for tk in 0..n {
        let i = first + tk;
        let pos_in_seg = i as i64 - prev_eos - 1;
        // rq::ple_window: a token within `shift` of its segment start reads
        // EOS instead of the real previous token
        let mut w = [stream[i], eos, eos];
        for (shift, slot) in [(1usize, 1usize), (2, 2)] {
            if i >= shift && pos_in_seg >= shift as i64 {
                w[slot] = stream[i - shift];
            }
        }
        for ngram in 2..=c.ngram_size {
            let mut mixed = w[0].wrapping_mul(ple.multipliers[0]);
            for (wk, m) in w.iter().zip(&ple.multipliers).take(ngram).skip(1) {
                mixed ^= wk.wrapping_mul(*m);
            }
            let start = (ngram - 2) * hpn;
            for hh in 0..hpn {
                let rid =
                    mixed.rem_euclid(ple.head_vocab[start + hh]) + ple.head_offset[start + hh];
                // a bad id would read anywhere in a 51.2 GB buffer, so it is
                // checked rather than trusted
                if rid < 0 || rid as usize >= ple.table_rows.max(1) {
                    return Err(GpuModelError::Unsupported(format!(
                        "ple row id {rid} outside the {}-row table",
                        ple.table_rows
                    )));
                }
                out[tk * heads + start + hh] = rid as u32;
            }
        }
        if stream[i] == eos {
            prev_eos = i as i64;
        }
    }
    Ok(out)
}

/// Stage `n` PLE rows into `sc.d_emb` off the device table (slot 532).
fn stage_ple_device(
    exec: &Arc<GpuExecutor>,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    table: &CudaSlice<u8>,
    ids: &[u32],
    sc: &mut Scratch,
) -> Result<(), GpuModelError> {
    let heads = c.ple_heads();
    let width = c.ple_embed / heads;
    exec.upload_u32(ids, &mut sc.d_ple_ids)?;
    exec.q4x_ple_gather(
        table,
        &sc.d_ple_ids,
        &mut sc.d_emb,
        ple.table_scale,
        ids.len() / heads,
        heads,
        width,
    )?;
    Ok(())
}

/// Fault the host-lane PLE table into the page cache at LOAD, not during the
/// first prompts.
///
/// The gather reads 16 rows a token out of a table that is 26.8 GiB on an
/// MX-quantized export, so a cold mapping pays those as disk seeks on the
/// critical path - a TTFT problem, not a throughput one, and the repo has met
/// it before (`ple-table-page-cache-trap`, which the GGUF lane's baselines
/// note handles by warming the shards by hand before every lane). Measured on
/// Mia's export before this, 3 reps of one serve: 1486.8 -> 1312.5 -> 1168.1
/// ms p50 latency, i.e. still warming on the third rep, with aiperf's
/// end-to-end throughput climbing 24.35 -> 27.26 -> 28.24 underneath it.
///
/// Costs a one-off sequential read at load, which is the cheap way to buy it:
/// the load is already disk-bound and the table is contiguous per shard. On
/// an integrated die this is the whole table's residency plan - the mapping
/// IS device memory there, so there is no second copy to make.
pub(super) fn warm_ple_table(st: &ShardedSafetensors, c: &Qwen4ExpConfig, li: usize) {
    let emb = format!("model.language_model.layers.{li}.ple.ple_embedding");
    let t0 = std::time::Instant::now();
    let mut bytes = 0usize;
    for sh in 0..c.ngram_split {
        for suffix in ["weight", "weight_scale"] {
            let name = format!("{emb}.ngram_embedding.shard_{sh}.{suffix}");
            // a shard that has no scale plane is the FP8 table, not an error
            if let Ok(n) = st.warm_tensor(&name) {
                bytes += n;
                // what the cache cannot keep faults back one row at a time:
                // its page only, never the readahead window around it
                let _ = st.advise_tensor(&name, MapAccess::Random, &[(0, n)]);
            }
        }
    }
    if bytes > 0 {
        let s = t0.elapsed().as_secs_f64();
        eprintln!(
            "[q4x-ple] warmed {:.1} GiB of n-gram table in {s:.1}s ({:.0} MB/s)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            bytes as f64 / 1e6 / s.max(1e-9),
        );
    }
}

/// Whether to make the 51.2 GB n-gram table device-resident. Refuses when
/// the card cannot hold it on top of everything already loaded, so a smaller
/// board still runs (slowly) rather than failing to load; `PADDOCK_Q4X_PLE_HOST`
/// forces the host lane for A/Bs.
pub(super) fn ple_device_table(exec: &Arc<GpuExecutor>, c: &Qwen4ExpConfig) -> bool {
    if std::env::var("PADDOCK_Q4X_PLE_HOST").is_ok_and(|v| v != "0") {
        eprintln!("[q4x-ple] HOST lane (PADDOCK_Q4X_PLE_HOST)");
        return false;
    }
    if !exec.has_q4x_ple_gather() {
        eprintln!("[q4x-ple] HOST lane: pack has no q4x_ple_gather (slot 532)");
        return false;
    }
    // UNIFIED-MEMORY DIE: there is nothing to move. The table is already in
    // DRAM as a file mapping, and on GB10/Jetson a "device" allocation is the
    // same physical memory - so copying it in does not shorten a single read,
    // it just holds 51.2 GB twice. The device table is a DISCRETE-card
    // optimization: there the copy turns a PCIe round trip per gather into a
    // local read, which is what bought the 891-48697 ms prefill ticks back.
    //
    // Taking it here is how NVIDIA's official NVFP4 checkpoint killed the box
    // (2026-09-19): 79 GB of weights plus a 51.2 GB second copy of a table
    // that was already resident, on a 121 GiB board. The GGUF lane has always
    // host-mapped this table on this hardware and serves 29-34 tok/s doing it.
    if exec.is_integrated() {
        eprintln!(
            "[q4x-ple] HOST lane: unified-memory die - the mapping IS device memory, \
             a device copy would hold the table twice"
        );
        return false;
    }
    let want = (c.ngram_vocab_base as usize) * c.ple_heads() * (c.ple_embed / c.ple_heads());
    let Ok((free, _)) = cudarc::driver::result::mem_get_info() else {
        return true; // no honest number - let the allocation decide
    };
    // 4 GiB of slack: the table is the last big claim and the scratch planes
    // below it still have to fit
    const SLACK: usize = 4 << 30;
    if want + SLACK > free {
        eprintln!(
            "[q4x-ple] HOST lane: table needs {:.1} GiB, {:.1} GiB free",
            want as f64 / (1u64 << 30) as f64,
            free as f64 / (1u64 << 30) as f64,
        );
        return false;
    }
    true
}
