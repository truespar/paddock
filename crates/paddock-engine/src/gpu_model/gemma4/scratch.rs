//! The prefill/decode scratch planes, and the dimensions they are sized from.
//!
//! Split out of `load.rs` (which had crossed the 2,500-line ceiling) when the
//! scratch stopped being a load-time constant. It is allocated twice now: once
//! by the loader, and again by `enable_batch` whenever the KV plan cannot seat
//! the configured server beside the chunk width the loader picked. That is why
//! the sizing inputs live in [`ScratchDims`] rather than being read off the
//! layer list - `enable_batch` holds `&mut self` and must be able to rebuild
//! without re-borrowing the weights.
//!
//! Every `pf_*` / `moe_*` plane is `[pf_rows, dim]`, so the whole set is linear
//! in `pf_rows` (0.62 MiB/row on gemma-4-31B Q8_0, measured). The handful of
//! planes that are not row-scaled - `pf_skfix`, `pf_fin`, `pf_xq`/`pf_xs` - are
//! the ~120 MB floor a stub build still pays.

use cudarc::driver::CudaSlice;

use super::Scratch;
use crate::gpu::{GpuError, GpuExecutor};

/// Batched-prefill chunk rows. 2048 (llama's ubatch class) puts the mmq
/// GEMMs above the pipe rung's 1024-row floor; widest scratch row set is
/// ~1 GB - noise next to the 33 GB of weights.
// 8192: SGLang's B200 heuristic runs 16384-token prefill chunks -
// at our old 2048 cap every full-weight pass amortized over 8x fewer
// tokens and the BM=128 expert-sort PAD fraction sat at ~2x (255 blocks
// for 128 experts). The tick stays 2048 by DEFAULT (decode cadence);
// PADDOCK_G4_TICK_ROWS raises it per deployment up to this cap.
pub(crate) const PF_ROWS: usize = 8192;

/// Rows the prefill scratch is actually allocated for on this server - and
/// therefore the size every prefill lane chunks at. The two must never
/// diverge: the scratch planes are `[rows, dim]`, so a chunk wider than the
/// allocation is an out-of-bounds write, not a slow path.
///
/// PF_ROWS above is a CEILING, not a size. It was raised 2048 -> 8192 for
/// B200 muse waves, and allocating the full 8192 unconditionally costs every
/// server ~5.0 GiB on gemma-4-31B - even a 4096-context serve, which can
/// never present more than 4096 rows in one chunk. That unusable half of the
/// scratch is enough to push the batch pool to zero slots and drop the model
/// onto the serial engine.
///
/// Two things bound a chunk, and the wider wins:
///  - a single-sequence bulk prefill, at most `max_ctx` rows;
///  - one serving tick - the mixed prefill budget plus every live slot's
///    spec-verify rows. [`pf_rows_floor`] covers the budget plus 1024 rows of
///    verify, i.e. any width up to ~113 slots.
///
/// This is where a server STARTS. `enable_batch` steps it down the halving
/// ladder when the KV plan cannot seat the configured context beside it, so
/// `GpuGemma4::pf_rows` - not this - is the width every lane must read.
pub(crate) fn pf_rows(max_ctx: usize) -> usize {
    PF_ROWS.min(max_ctx.next_multiple_of(128).max(pf_rows_floor()))
}

/// The narrowest chunk this server may run: one serving tick - the mixed
/// prefill budget plus 1024 rows of spec verify.
///
/// Below it a tick overruns the planes, which is an out-of-bounds write and
/// not a slow path, so this is where `batch::set_pf_rows`'s ladder stops.
pub(crate) fn pf_rows_floor() -> usize {
    (super::batch::mixed_tick_rows() + 1024).next_multiple_of(128)
}

/// Everything the scratch planes are sized from, resolved once at load.
///
/// The per-layer maxima are precomputed deliberately: they are the only thing
/// the builder used the layer list for, and holding a borrow of it would stop
/// `enable_batch` rebuilding the scratch through `&mut self`.
pub(crate) struct ScratchDims {
    pub n_embd: usize,
    pub n_head: usize,
    /// global-layer head dim - the unit-weight vector for the weightless V
    /// norm is one head wide
    pub hd_global: usize,
    pub n_vocab: usize,
    pub n_ff: usize,
    /// widest per-row attention output: `n_head * head_dim_global`
    pub max_q: usize,
    /// widest per-row K or V span across the layers
    pub max_kv: usize,
    /// widest fused qkv-concat row (q + 2kv, or q + kv on V-less globals) -
    /// the all-band concat arm writes this, which a max_q-only plane overflows
    pub max_qkv: usize,
    /// muse-glimmer ships a sigmoid attention output gate; gemma4 does not, and
    /// pays a 1-element stub instead of `pf_rows * max_q`
    pub gated: bool,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_exp: usize,
    pub f8_on: bool,
    pub f8row: bool,
    pub f8w_pf: bool,
    pub f8a: bool,
    pub f8t_dec: bool,
    /// any dense plane (attention, shared FFN, head, the diffusion MLP) is
    /// k-quant: the W4A8 staging planes are real
    pub kq_dense: bool,
    /// the routed experts are k-quant seats: their sums + grouped CSR planes
    pub kq_moe: bool,
    /// the dense FFN's live Q8 gate/up planes land straight into the down
    /// GEMM's input at prefill (pack slot 865): size `pf_yq_ff` for it
    pub q8_gu: bool,
    /// Raw (non-pool) accumulator for the uniq-routing diagnostic, armed once
    /// by the loader. Carried rather than re-armed because `g4_moe_uniq_arm`
    /// leaks its buffer by design (process lifetime) - arming it per rebuild
    /// would leak one per rung of the chunk ladder.
    pub moe_uniq_dev: u64,
}

/// Allocate the scratch set for `pf_rows`-row prefill chunks.
///
/// `pf_rows` is the chunk width every prefill lane then splits at - the two
/// must never diverge, because the planes are `[pf_rows, dim]` and a wider
/// chunk is an out-of-bounds write, not a slow path. `GpuGemma4::pf_rows` is
/// the live value; see `batch::set_pf_rows`.
pub(crate) fn build(
    exec: &GpuExecutor,
    d: &ScratchDims,
    pf_rows: usize,
) -> Result<Scratch, GpuError> {
    let (n_embd, n_vocab, n_ff) = (d.n_embd, d.n_vocab, d.n_ff);
    let (max_q, max_kv) = (d.max_q, d.max_kv);
    let (n_expert, n_expert_used, ff_exp) = (d.n_expert, d.n_expert_used, d.ff_exp);
    let (f8_on, f8row, f8w_pf, f8a, f8t_dec) = (d.f8_on, d.f8row, d.f8w_pf, d.f8a, d.f8t_dec);
    let gate_q = if d.gated { max_q } else { 1 };

    let name =
        |what: &str, e: &dyn std::fmt::Display| GpuError::Driver(format!("scratch.{what}: {e}"));
    let alloc = |n: usize| -> Result<CudaSlice<f32>, GpuError> {
        exec.stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| name("f32 plane", &e))
    };

    // sorted-MoE layout bounds (moe_align PAD padding): worst case over the
    // block tiles - BM=32 has the most BLOCKS, BM=128 (the tc5 f8 lane) the
    // most padded ROWS.
    let moe_pairs = pf_rows * n_expert_used.max(1);
    let mb32 = (moe_pairs + n_expert * 31).div_ceil(32);
    let mb64 = (moe_pairs + n_expert * 63).div_ceil(64);
    let mb128 = (moe_pairs + n_expert * 127).div_ceil(128);
    let (moe_srows, moe_blocks) = if n_expert != 0 {
        (
            (mb32 * 32).max(mb64 * 64).max(mb128 * 128),
            mb32.max(mb64).max(mb128),
        )
    } else {
        (0, 0)
    };
    // tc5 f8 expert-lane planes are sized on the BM=128 superset
    let moe_f8_rows = if n_expert != 0 { mb128 * 128 } else { 0 };
    let ff_pad = ff_exp.next_multiple_of(128).max(1);
    // k-quant seats: the grouped CSR at the 8-row group has the most blocks
    let kq_mb8 = (moe_pairs + n_expert * 7).div_ceil(8).max(1);
    // k-quant dense staging: the widest quantize input (n_embd, n_ff, wo's)
    let kq_in = n_ff.max(n_embd).max(max_q);

    Ok(Scratch {
        moe_verify: false,
        x: alloc(n_embd)?,
        normed: alloc(n_embd)?,
        q: alloc(max_q)?,
        k: alloc(max_kv)?,
        v: alloc(max_kv)?,
        kn: alloc(max_kv)?,
        vn: alloc(max_kv)?,
        qn: alloc(max_q)?,
        attn: alloc(max_q)?,
        agate: alloc(gate_q)?,
        proj: alloc(n_embd)?,
        gate: alloc(n_ff)?,
        up: alloc(n_ff)?,
        logits: alloc(n_vocab)?,
        stream_tmp: alloc(n_embd)?,
        pos: exec.alloc_u32(1).map_err(|e| name("pos", &e))?,
        ones: {
            // unit weight for the UNWEIGHTED V rmsnorm - one head wide
            let host = vec![1.0f32; d.hd_global];
            exec.stream
                .clone_htod(&host)
                .map_err(|e| name("ones", &e))?
        },
        // this family has no attention sinks; the plane is the -inf identity
        neg_inf_sinks: exec
            .alloc_no_sinks(d.n_head)
            .map_err(|e| name("sinks", &e))?,
        pf_x: alloc(pf_rows * n_embd)?,
        pf_tmp: alloc(pf_rows * n_embd)?,
        pf_normed: alloc(pf_rows * n_embd)?,
        // sized for the widest per-row output: separate q (max_q) OR the fused
        // qkv-concat row - the all-band concat arm writes r x concat at prefill
        // chunks, which overflows a max_q-only buffer on global layers
        pf_q: alloc(pf_rows * max_q.max(d.max_qkv))?,
        pf_qn: alloc(pf_rows * max_q)?,
        pf_k: alloc(pf_rows * max_kv)?,
        pf_v: alloc(pf_rows * max_kv)?,
        pf_kn: alloc(pf_rows * max_kv)?,
        pf_vn: alloc(pf_rows * max_kv)?,
        pf_attn: alloc(pf_rows * max_q)?,
        pf_agate: alloc(if gate_q > 1 { pf_rows * max_q } else { 1 })?,
        pf_proj: alloc(pf_rows * n_embd)?,
        pf_gate: alloc(pf_rows * 2 * n_ff)?, // fused gate|up rows
        pf_up: alloc(pf_rows * n_ff)?,
        pf_row: alloc(n_embd)?,
        pf_pos: exec.alloc_u32(pf_rows).map_err(|e| name("pf_pos", &e))?,
        pf_fin: alloc(64 * n_vocab)?,
        pf_toks: exec.alloc_u32(pf_rows).map_err(|e| name("pf_toks", &e))?,
        pf_runs: exec.alloc_u32(65).map_err(|e| name("pf_runs", &e))?,
        pf_attn_pos: exec
            .alloc_u32(pf_rows)
            .map_err(|e| name("pf_attn_pos", &e))?,
        pf_slots: {
            let zeros = vec![0u32; pf_rows];
            exec.stream
                .clone_htod(&zeros)
                .map_err(|e| name("pf_slots", &e))?
        },
        // widest mmq/mma quantize INPUT across the whole walk: ffn_down (n_ff),
        // the pre-norm rows (n_embd), and the wo input (n_head*hd - 8192 on
        // hd-512 layers). On the 31B the fat n_ff covered all of these by
        // accident; the A4B's shared ff is only 2112 < n_embd 2816, and sizing
        // by n_ff alone was an OOB write from the first attn quantize.
        pf_yq: exec
            .alloc_u8(
                n_ff.max(n_embd).max(max_q).div_ceil(128) * pf_rows.next_multiple_of(128) * 144,
            )
            .map_err(|e| name("pf_yq", &e))?,
        // the down GEMM's mmq input when gate|up land it directly (pf_yq is
        // that launch's input, so it cannot take the output)
        pf_yq_ff: exec
            .alloc_u8(if d.q8_gu {
                n_ff.div_ceil(128) * pf_rows.next_multiple_of(128) * 144
            } else {
                32
            })
            .map_err(|e| name("pf_yq_ff", &e))?,
        // skfix also holds the ks K-split partial planes: the M-col rung
        // (wide-spec verify, <=192 rows) peaks at nz*out*rows = 2*21504*192 f32
        // (gate/up) - 9M covers every dense shape
        pf_skfix: alloc(12 * 1024 * 1024)?,
        // 192 rows: the mma_ks/M-col quantize class serves the wide spec verify
        // (was 64 - the mma_ks BN cap before the M-col rung). Same widest-input
        // rule as pf_yq (wo input outgrows n_ff on the A4B).
        pf_xq: exec
            .alloc_i8(192 * n_ff.max(n_embd).max(max_q))
            .map_err(|e| name("pf_xq", &e))?,
        pf_xs: alloc(192 * n_ff.max(n_embd).max(max_q) / 32)?,
        // f8a included: its attn arms quantize r*n_embd / r*(n_head*hd) into
        // these planes - an f8-only predicate left 32-byte stubs under a live
        // F8A build whenever f8w/f8row were off (OOB writes).
        pf_e4q: exec
            .alloc_i8(if f8_on || f8row || f8w_pf || f8a {
                pf_rows * n_ff.max(n_embd).max(max_q)
            } else {
                32
            })
            .map_err(|e| name("pf_e4q", &e))?,
        pf_e4s: exec
            .alloc_u8(if f8_on || f8w_pf || f8a {
                pf_rows * n_ff.max(n_embd).max(max_q) / 32
            } else {
                32
            })
            .map_err(|e| name("pf_e4s", &e))?,
        // f8t decode arms row-quant up to 64 rows too - the 1-float stub was a
        // live OOB write surviving on allocation padding
        pf_e4rs: alloc(
            if f8row || f8t_dec || paddock_models::dev_var_os!("PADDOCK_G4_PC").is_some() {
                pf_rows
            } else {
                1
            },
        )?,
        // ones xrs for the fin-e4s static-store route - filled once here,
        // read-only forever after (pf_e4rs is per-tick volatile)
        pf_fae4rs: {
            let n = if f8row || f8t_dec { pf_rows } else { 1 };
            let mut b = alloc(n)?;
            exec.stream
                .memcpy_htod(&vec![1.0f32; n], &mut b)
                .map_err(|e| name("pf_fae4rs", &e))?;
            b
        },
        // fused-gu landing planes (not pf_e4q: the fused GEMM reads pf_e4q via
        // TMA while storing - same-buffer would race)
        pf_ffq: exec
            .alloc_i8(if f8_on || f8a { pf_rows * n_ff } else { 32 })
            .map_err(|e| name("pf_ffq", &e))?,
        pf_ffs: exec
            .alloc_u8(if f8_on || f8a {
                pf_rows * n_ff / 32
            } else {
                32
            })
            .map_err(|e| name("pf_ffs", &e))?,
        // k-quant staging: the widest quantize input of the walk (the same
        // rule pf_yq / pf_xq follow), sized only when a plane needs it
        kq_xq: exec
            .alloc_i8(if d.kq_dense { kq_in } else { 1 })
            .map_err(|e| name("kq_xq", &e))?,
        kq_xs: alloc(if d.kq_dense { kq_in / 32 } else { 1 })?,
        kq_ssums: alloc(if d.kq_dense { kq_in / 16 } else { 1 })?,
        pf_ssums: alloc(if d.kq_dense { 192 * kq_in / 16 } else { 1 })?,
        pf_xsums: alloc(if d.kq_dense {
            kq_in.div_ceil(128) * pf_rows.next_multiple_of(128) * 4
        } else {
            1
        })?,
        embd_id: exec.alloc_u32(1).map_err(|e| name("embd_id", &e))?,
        moe_ssums: alloc(if d.kq_moe {
            (pf_rows * n_embd).max(pf_rows * n_expert_used * ff_exp) / 16
        } else {
            1
        })?,
        kq_srow: exec
            .alloc_u32(if d.kq_moe { kq_mb8 * 8 } else { 1 })
            .map_err(|e| name("kq_srow", &e))?,
        kq_sslot: exec
            .alloc_u32(if d.kq_moe { kq_mb8 * 8 } else { 1 })
            .map_err(|e| name("kq_sslot", &e))?,
        kq_bexp: exec
            .alloc_u32(if d.kq_moe { kq_mb8 } else { 1 })
            .map_err(|e| name("kq_bexp", &e))?,
        kq_sfq: exec
            .alloc_i8(if d.kq_moe { mb32 * 32 * ff_exp } else { 1 })
            .map_err(|e| name("kq_sfq", &e))?,
        kq_sfs: alloc(if d.kq_moe { mb32 * 32 * ff_exp / 32 } else { 1 })?,
        moe_emap: exec
            .alloc_u32(if d.kq_moe { 2 * n_expert } else { 1 })
            .map_err(|e| name("moe_emap", &e))?,
        // hybrid-MoE lane (26B-A4B): pf_rows-sized like the pf planes; 1-elem
        // stubs on dense models.
        moe_xn: alloc(if n_expert != 0 { pf_rows * n_embd } else { 1 })?,
        moe_out: alloc(if n_expert != 0 { pf_rows * n_embd } else { 1 })?,
        moe_logits: alloc(if n_expert != 0 { pf_rows * n_expert } else { 1 })?,
        moe_idx: exec
            .alloc_u32(if n_expert != 0 {
                pf_rows * n_expert_used
            } else {
                1
            })
            .map_err(|e| name("moe_idx", &e))?,
        moe_w: alloc(if n_expert != 0 {
            pf_rows * n_expert_used
        } else {
            1
        })?,
        moe_xq: exec
            .alloc_i8(if n_expert != 0 { pf_rows * n_embd } else { 1 })
            .map_err(|e| name("moe_xq", &e))?,
        moe_xs: alloc(if n_expert != 0 {
            pf_rows * n_embd / 32
        } else {
            1
        })?,
        // flat-scale e4m3 expert lane - same shape as moe_xq/moe_xs, different
        // encoding. Allocated unconditionally so the arm is a pure env flip.
        moe_x8q: exec
            .alloc_u8(if n_expert != 0 { pf_rows * n_embd } else { 1 })
            .map_err(|e| name("moe_x8q", &e))?,
        moe_x8s: alloc(if n_expert != 0 {
            pf_rows * n_embd / 32
        } else {
            1
        })?,
        moe_fused: alloc(if n_expert != 0 {
            pf_rows * n_expert_used * ff_exp
        } else {
            1
        })?,
        // fq/fs serve both expert classes: token-batched rows (= pairs) and the
        // sorted layout's PAD-padded rows (BM=64 superset)
        moe_fq: exec
            .alloc_i8(if n_expert != 0 { moe_srows * ff_exp } else { 1 })
            .map_err(|e| name("moe_fq", &e))?,
        moe_fs: alloc(if n_expert != 0 {
            moe_srows * ff_exp / 32
        } else {
            1
        })?,
        moe_zbias: alloc(n_expert.max(1))?, // alloc_zeros - stays zero
        moe_srow: exec
            .alloc_u32(moe_srows.max(1))
            .map_err(|e| name("moe_srow", &e))?,
        moe_sslot: exec
            .alloc_u32(moe_srows.max(1))
            .map_err(|e| name("moe_sslot", &e))?,
        moe_bexp: exec
            .alloc_u32(moe_blocks.max(1))
            .map_err(|e| name("moe_bexp", &e))?,
        moe_srow2: exec
            .alloc_u32(if n_expert != 0 { mb32 * 32 } else { 1 })
            .map_err(|e| name("moe_srow2", &e))?,
        moe_sslot2: exec
            .alloc_u32(if n_expert != 0 { mb32 * 32 } else { 1 })
            .map_err(|e| name("moe_sslot2", &e))?,
        moe_bexp2: exec
            .alloc_u32(if n_expert != 0 { mb32 } else { 1 })
            .map_err(|e| name("moe_bexp2", &e))?,
        moe_pairmap: alloc(if n_expert != 0 {
            pf_rows * n_expert_used
        } else {
            1
        })?,
        moe_part: alloc(if n_expert != 0 {
            pf_rows * n_expert_used * n_embd
        } else {
            1
        })?,
        moe_e4q: exec
            .alloc_i8(if n_expert != 0 { pf_rows * n_embd } else { 1 })
            .map_err(|e| name("moe_e4q", &e))?,
        moe_e4s: exec
            .alloc_u8(if n_expert != 0 {
                pf_rows * n_embd / 32
            } else {
                1
            })
            .map_err(|e| name("moe_e4s", &e))?,
        moe_xg: exec
            .alloc_u8((moe_f8_rows * n_embd).max(1))
            .map_err(|e| name("moe_xg", &e))?,
        moe_sg: exec
            .alloc_u8((moe_f8_rows * n_embd / 32).max(1))
            .map_err(|e| name("moe_sg", &e))?,
        moe_gu: alloc((moe_f8_rows * 2 * ff_exp).max(1))?,
        // alloc_zeros: the K-tail [ff_exp, ff_pad) is a STANDING zero region
        // (only geglu2_pad writes here, and only [0, ff_exp))
        moe_fq8: exec
            .alloc_u8((moe_f8_rows * ff_pad).max(1))
            .map_err(|e| name("moe_fq8", &e))?,
        moe_fs8: exec
            .alloc_u8((moe_f8_rows * ff_pad / 32).max(1))
            .map_err(|e| name("moe_fs8", &e))?,
        moe_uniq_dev: d.moe_uniq_dev,
    })
}
