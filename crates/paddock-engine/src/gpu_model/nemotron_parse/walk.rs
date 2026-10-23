//! The batch-invariant layer walk: a row's bits depend on its own request
//! alone - not on how many other rows share its pass, which requests they
//! belong to, how its prompt was cut into chunks, or whether it is a decode
//! tick's row or a verify round's.
//!
//! Two row classes, each one set of kernels at every width:
//! - **decode-class rows** (a decode tick, a verify round, the leading decode
//!   band of a mixed tick): the linears and the head on `dense_w16` f16 (a
//!   weight row is the mma's M, a token its N, every CTA walks the full K in
//!   one fixed order), attention on the rows partial under slot 744's pow2
//!   split law - a row's splits, tiles and fold follow its own key count, and
//!   its group-mates (a verify chunk's rows read their slot's KV once) never
//!   move its bits. 16 splits at every width, so a captured graph bakes them.
//! - **prefill rows** (picture prefix and prompt): the linears on the unsplit
//!   mma rings (slot 894 and the f16 GELU landing; every ring walks one
//!   output's K in the same order at any row count), attention on the WMMA
//!   prefill tile per run (key tiles at absolute positions, a row's masked
//!   tail tiles add exact zeros, softmax state per row).
//!
//! A token is always produced by the same class - the first one from the
//! prompt's last row, every later one from a decode-class row - so a page
//! reads byte-identically alone, at c32, chunked any way, with or without
//! speculation. Everything else on the walk (norms, biases, GELU, residual
//! adds, the KV append) is row-wise.

use crate::gpu::{GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::nemotron::batch::{ROWS_GROUP, rows_groups_law, w16_split_size};
use paddock_models::nemotron_parse::DECODER_LN_EPS;

use super::load::GpuNemotronParse;

/// Splits of the decode-class attention: the pow2 law's budget AND the
/// launched count. z = max(256, next_pow2(ceil(n / 16))) keys a split, so a
/// page's ~3.5K-row context spreads over 14 splits of 256 (8 kv heads x 14 =
/// 112 CTAs for one slot - the die's 48 SMs stay busy) and 16384 keys stay
/// within 16 splits of 1024.
pub(super) const INV_SPLITS: usize = 16;

/// The pow2 law word of [`INV_SPLITS`] (0 = take `n_splits` as the budget).
const INV_LAW: usize = 0;

/// The decode band's attention groups for rows with `slots` / `pos`:
/// consecutive same-slot rows at consecutive positions form a run (a verify
/// chunk), split where the law's split size changes and at 8 rows. Flat
/// `[first row, rows]` pairs for `d_groups`.
pub(super) fn band_groups(slots: &[u32], pos: &[u32]) -> Vec<u32> {
    let mut runs: Vec<(usize, usize, u32)> = Vec::new();
    for (i, (&s, &p)) in slots.iter().zip(pos).enumerate() {
        match runs.last_mut() {
            Some((off, n, p0)) if slots[*off] == s && *p0 + *n as u32 == p => *n += 1,
            _ => runs.push((i, 1, p)),
        }
    }
    rows_groups_law(runs, |n| w16_split_size(INV_LAW, INV_SPLITS, n), ROWS_GROUP)
}

impl GpuNemotronParse {
    /// Upload the decode band's groups (rows `0..slots.len()`); their count
    /// is what the next walk (and a graph captured over it) attends with.
    pub(super) fn upload_band_groups(
        &mut self,
        slots: &[u32],
        pos: &[u32],
    ) -> Result<usize, GpuModelError> {
        if slots.is_empty() {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.n_groups = 0;
            return Ok(0);
        }
        let g = band_groups(slots, pos);
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut v = bs
            .sc
            .d_groups
            .try_slice_mut(0..g.len())
            .ok_or_else(|| GpuError::Driver("band groups slice".into()))?;
        self.exec
            .stream
            .memcpy_htod(&g, &mut v)
            .map_err(crate::gpu::from_driver)?;
        bs.n_groups = g.len() / 2;
        Ok(bs.n_groups)
    }

    /// One walk over `r` staged rows: the leading `dec` decode-class rows
    /// (their groups already uploaded), then the prefill `runs` (same-slot
    /// `(first row, rows)` spans of rows `dec..r`).
    pub(super) fn layer_walk_inv(
        &mut self,
        r: usize,
        dec: usize,
        runs: &[(usize, usize)],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let cfg = &self.cfg;
        let (dm, nh, n_kv, hd, ffn) = (
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.head_dim(),
            cfg.ffn,
        );
        let kv_dim = n_kv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let kv_dtype = self.kv_dtype;
        let max_ctx = self.max_ctx;
        let p = r - dec;
        // the hd 64 WMMA prefill tile wants f16 caches (the family pins f16 KV)
        let wmma_pf = matches!(kv_dtype, KvDtype::Fp16) && exec.has_attn_prefill_f16_paged();
        let bs = self.batch.as_mut().expect("batch enabled");
        let (bps, n_groups) = (bs.bps, bs.n_groups);
        for (li, layer) in self.layers.iter().enumerate() {
            let sc = &mut bs.sc;
            exec.layernorm_f16(
                &sc.x,
                &layer.ln1.w,
                &layer.ln1.b,
                &mut sc.xn16,
                r,
                dm,
                DECODER_LN_EPS,
            )?;
            // q / k / v: the band on dense_w16, the prefill rows on the rings
            for (w, b, y, out) in [
                (&layer.wq, &layer.bq, &mut sc.q, dm),
                (&layer.wk, &layer.bk, &mut sc.k, kv_dim),
                (&layer.wv, &layer.bv, &mut sc.v, kv_dim),
            ] {
                if dec > 0 {
                    exec.dense_w16_f16(w, &sc.xn16, y, out, dec)?;
                }
                exec.f16_gemm_ring_rows(w, &sc.xn16, dec, y, dec, p, None)?;
                exec.bias_add(y, b, r, out)?;
            }
            let kvs = &mut bs.kv[li];
            for (src, pool) in [(&sc.k, &mut kvs.k), (&sc.v, &mut kvs.v)] {
                exec.kv_append_batch_paged(
                    src,
                    pool,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &bs.d_bt,
                    bps,
                    kv_dim,
                    r,
                    kv_dtype,
                )?;
            }
            if dec > 0 {
                exec.attn_rows_partial_fixed(
                    &sc.q,
                    &kvs.k,
                    &kvs.v,
                    &mut sc.attn_o,
                    &mut sc.attn_ml,
                    &sc.d_pos,
                    &sc.d_slots,
                    &sc.d_groups,
                    n_groups,
                    Some((&bs.d_bt, bps)),
                    max_ctx,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    dec,
                    INV_SPLITS,
                    0,
                    scale,
                    kv_dtype,
                    INV_LAW,
                    None,
                )?;
                exec.attn_combine_batch(
                    &sc.attn_o,
                    &sc.attn_ml,
                    &sc.sinks,
                    &mut sc.attn,
                    nh,
                    hd,
                    INV_SPLITS,
                    dec,
                )?;
            }
            for &(off, len) in runs {
                if wmma_pf {
                    exec.attn_prefill_f16_paged_at(
                        &sc.q,
                        &kvs.k,
                        &kvs.v,
                        &sc.sinks,
                        &mut sc.attn,
                        &sc.d_pos,
                        &sc.d_slots,
                        off,
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        kv_dim,
                        0,
                        len,
                        scale,
                        kv_dtype,
                    )?;
                } else {
                    exec.attn_decode_batch_rows_paged(
                        &sc.q,
                        &kvs.k,
                        &kvs.v,
                        &sc.sinks,
                        &mut sc.attn,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        kv_dim,
                        0,
                        off,
                        len,
                        scale,
                        kv_dtype,
                    )?;
                }
            }
            exec.convert_f32_f16(&sc.attn, &mut sc.attn16, r * dm)?;
            // o onto the residual: the band through scr + the row-wise add,
            // the prefill rows with the bias + residual in the ring's epilogue
            if dec > 0 {
                exec.dense_w16_f16(&layer.wo, &sc.attn16, &mut sc.scr, dm, dec)?;
                exec.add_bias_res(&mut sc.x, &sc.scr, &layer.bo, dec, dm)?;
            }
            exec.f16_gemm_ring_rows(
                &layer.wo,
                &sc.attn16,
                dec,
                &mut sc.x,
                dec,
                p,
                Some(&layer.bo),
            )?;
            exec.layernorm_f16(
                &sc.x,
                &layer.ln2.w,
                &layer.ln2.b,
                &mut sc.xn16,
                r,
                dm,
                DECODER_LN_EPS,
            )?;
            if dec > 0 {
                exec.dense_w16_f16(&layer.fc1, &sc.xn16, &mut sc.scr, ffn, dec)?;
                exec.gelu_erf_bias_f16(&sc.scr, &layer.fc1_b, &mut sc.ff16, dec, ffn)?;
            }
            exec.f16_gemm_h_gelu_rows(
                &layer.fc1,
                &sc.xn16,
                dec,
                &mut sc.ff16,
                dec,
                &layer.fc1_b,
                p,
            )?;
            if dec > 0 {
                exec.dense_w16_f16(&layer.fc2, &sc.ff16, &mut sc.scr, dm, dec)?;
                exec.add_bias_res(&mut sc.x, &sc.scr, &layer.fc2_b, dec, dm)?;
            }
            exec.f16_gemm_ring_rows(
                &layer.fc2,
                &sc.ff16,
                dec,
                &mut sc.x,
                dec,
                p,
                Some(&layer.fc2_b),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_chunks_group_per_slot_and_ticks_row_by_row() {
        // a tick: every row its own slot
        assert_eq!(
            band_groups(&[3, 0, 7], &[10, 20, 30]),
            vec![0, 1, 1, 1, 2, 1]
        );
        // a verify round: [pending, draft] per slot
        assert_eq!(
            band_groups(&[3, 3, 0, 0], &[10, 11, 20, 21]),
            vec![0, 2, 2, 2]
        );
        // a chunk across the pow2 edge (4096 keys -> 4097) splits there
        assert_eq!(band_groups(&[1, 1], &[4095, 4096]), vec![0, 1, 1, 1]);
    }
}
