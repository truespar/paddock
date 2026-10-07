//! Native MLX affine storage: packed U32 codes followed by BF16 scales/biases.
//! Concatenation is byte-only; no weights are requantized or expanded on the host.
use crate::device::{Buffer, Commands, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

// Internal tag, outside the GGUF type namespace. Only the MLX loader creates it.
pub(crate) const AFFINE4: u32 = 0x100;

#[derive(Clone, Copy)]
enum Arithmetic {
    Adaptive,
    Llama(usize),
    Verify(bool),
    StablePrefill,
    StableDecode,
}

#[cfg(test)]
thread_local! {
    pub(crate) static BASELINE_PACKED_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    // Numerical isolation only: fix the reference prefill contraction width.
    // Not a serving election or a performance-qualified implementation.
    pub(crate) static CANONICAL_PREFILL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static SEPARATE_INPUTS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static BASELINE_STORE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static BASELINE_FFN_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static BASELINE_ADMISSION_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static BASELINE_TAIL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static BASELINE_COLD_CONTRACT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static BASELINE_STAGING_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(crate) static STAGING_MASK_FOR_TEST: std::cell::Cell<u8> = const { std::cell::Cell::new(31) };
    pub(crate) static BASELINE_RAGGED_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn compact_prefill_rows(rows: usize) -> bool {
    // The 32-row initial admission can leave a 480-row tail. All of these
    // shapes fit the existing 512-row padded input, without more allocation.
    let eligible = (480..=512).contains(&rows);
    #[cfg(test)]
    let eligible = eligible && (rows == 512 || !BASELINE_RAGGED_FOR_TEST.with(|v| v.get()));
    eligible
}

fn prefill_tile(rows: usize, k: usize, narrowest: usize, accelerated: bool) -> usize {
    // An initial admission slice leaves a nearly-full final wave (often 480
    // rows). Its 128-row-padded input already covers two complete 256-row
    // tiles; masking the final output is cheaper than rereading every weight
    // for eight 64-row tiles. Do not extend to shapes needing extra padding.
    let dense_tail = (480..512).contains(&rows) && matches!(k, 5120 | 6144 | 17408);
    #[cfg(test)]
    let dense_tail = dense_tail && !BASELINE_TAIL_FOR_TEST.with(|v| v.get());
    if rows <= 32 {
        32
    } else if accelerated
        && (rows.is_multiple_of(256) || dense_tail)
        && k >= 4096
        && (rows >= 512 || k >= 8192 || dense_tail)
        && narrowest >= 1024
    {
        256
    } else {
        64
    }
}

fn verify_columns(k: usize, widest: usize, rows: usize, accelerated: bool) -> usize {
    if !accelerated || rows == 1 || !matches!(k, 5120 | 17408) {
        16
    } else if widest <= 64 {
        4
    } else if rows <= 3 || ((5..=6).contains(&rows) && k <= 8192) {
        8
    } else {
        16
    }
}

// BF16 partial contractions are part of the pinned reference's arithmetic,
// not a tunable performance knob. Partitions contain whole affine groups.
fn partitions(k: usize, n: usize, m: usize) -> usize {
    #[cfg(test)]
    let m = if CANONICAL_PREFILL_FOR_TEST.with(|v| v.get()) {
        512
    } else {
        m
    };
    let mut parts = (512 / (n.div_ceil(32) * m.div_ceil(32))).min(k / 64).max(1);
    while parts > 1 && !k.is_multiple_of(parts * 64) {
        parts -= 1;
    }
    parts
}

/// Exact scratch envelope, including split-contraction tails. Callers reserve
/// this before admitting the model, never grow it inside a serving dispatch.
pub(crate) fn workspace_bytes(k: usize, n: usize, capacity: usize) -> usize {
    (1..=capacity)
        .map(|m| {
            if m < 13 {
                return m * k * 2;
            }
            let prefix = k.div_ceil(128) * 128 * m.div_ceil(128) * 128;
            let parts = partitions(k, n, m);
            (prefix + if parts > 1 { parts * m * n } else { 0 }) * 2
        })
        .max()
        .unwrap_or(0)
}

pub(crate) fn load(
    d: &MetalDevice,
    source: &ShardedSafetensors,
    name: &str,
    k: usize,
    n: usize,
) -> Result<Weight> {
    if !k.is_multiple_of(64) || k == 0 || n == 0 {
        return Err(MetalError::Model(format!(
            "{name}: invalid affine dimensions"
        )));
    }
    let base = name
        .strip_suffix(".weight")
        .ok_or_else(|| MetalError::Model(format!("{name}: expected weight name")))?;
    let mut parts = Vec::new();
    for (key, dtype, shape) in [
        (name.to_owned(), StDtype::U32, [n, k / 8]),
        (format!("{base}.scales"), StDtype::Bf16, [n, k / 64]),
        (format!("{base}.biases"), StDtype::Bf16, [n, k / 64]),
    ] {
        let (info, bytes) = source
            .bytes(&key)
            .ok_or_else(|| MetalError::Model(format!("missing {key}")))?;
        if info.dtype != dtype || info.shape != shape {
            return Err(MetalError::Model(format!(
                "{key}: expected {dtype:?} {shape:?}, got {:?} {:?}",
                info.dtype, info.shape
            )));
        }
        parts.push(bytes);
    }
    let buffer = d.upload_parts(&parts)?;
    Ok(Weight {
        buffer,
        ty: AFFINE4,
        k,
        n,
    })
}

pub(crate) fn project(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    project_inner(
        cmd,
        planes,
        input,
        rows,
        workspace,
        (Arithmetic::Adaptive, 0),
    );
}

/// Plain MLX Llama's small projections retain vector arithmetic through 32
/// rows on M5. Keep this graph-local: hybrid Qwen has its own qualified tree.
pub(crate) fn project_llama(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
) {
    let Some(spans) = cmd.projection_rows() else {
        return project_inner(
            cmd,
            planes,
            input,
            rows,
            workspace,
            (Arithmetic::Llama(rows), 0),
        );
    };
    let limit = llama_vector_limit(cmd.tensor_accelerated(), planes);
    let key = |logical: usize| {
        (
            logical == 1,
            logical >= limit,
            if logical >= limit {
                logical.div_ceil(32)
            } else {
                0
            },
        )
    };
    let mut i = 0;
    let mut end = 0;
    while i < spans.len() {
        let (first, mut count, logical) = spans[i];
        assert!(first == end && count > 0 && logical > 0 && first + count <= rows);
        i += 1;
        // Join compatible request spans without changing their numerical
        // contract or rereading weights once per request. Split only at a
        // different contraction tree or the existing scratch grant's limit.
        while let Some(&(next, extra, other)) = spans.get(i) {
            assert!(next == first + count && extra > 0 && next + extra <= rows);
            if key(logical) != key(other)
                || planes.iter().any(|(w, _)| {
                    let m = count + extra;
                    let parts = if logical >= limit {
                        partitions(w.k, w.n, logical)
                    } else {
                        1
                    };
                    let needed = if logical >= limit {
                        (w.k.div_ceil(128) * 128 * m.div_ceil(128) * 128
                            + if parts > 1 { parts * m * w.n } else { 0 })
                            * 2
                    } else {
                        m * w.k * 2
                    };
                    needed > workspace.len()
                })
            {
                break;
            }
            count += extra;
            i += 1;
        }
        project_inner(
            cmd,
            planes,
            input,
            count,
            workspace,
            (Arithmetic::Llama(logical), first),
        );
        end = first + count;
    }
    assert_eq!(end, rows);
}

fn llama_vector_limit(accelerated: bool, planes: &[(&Weight, &Buffer)]) -> usize {
    if accelerated && planes.iter().all(|(w, _)| w.k <= 2048 && w.n <= 2048) {
        33
    } else {
        13
    }
}

/// Phase belongs to the sequence, not the number of physical rows. Every
/// selected vocabulary row is a vector contraction; body prompt rows keep
/// their tensor contraction even in a one-token cache suffix.
pub(crate) fn project_stable(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
    spans: &[(usize, usize, usize)],
) {
    let mut end = 0;
    for &(first, count, logical) in spans {
        assert!(first == end && count > 0 && first + count <= rows && matches!(logical, 1 | 512));
        let mode = if logical == 1 {
            Arithmetic::StableDecode
        } else {
            Arithmetic::StablePrefill
        };
        project_inner(cmd, planes, input, count, workspace, (mode, first));
        end += count;
    }
    assert_eq!(end, rows);
}

/// Verification adds candidate rows, not independent decode requests. Keep
/// the ordinary decode arithmetic (including the single-stream BF16 bias
/// reduction), instead of electing a different matmul from the wider shape.
pub(crate) fn project_verify(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
    single: bool,
) {
    project_inner(
        cmd,
        planes,
        input,
        rows,
        workspace,
        (Arithmetic::Verify(single), 0),
    );
}

/// Keep already-rounded gate/up activations in BF16 and produce the down
/// projection's input directly. Same matrix and activation rounding boundaries;
/// no persistent expansion, new allocation or decode/verification election.
/// Ragged near-full tiles zero-fill the down-projection input padding.
pub(crate) fn prefill_ffn(
    cmd: &Commands<'_>,
    weights: [&Weight; 3],
    input: &Buffer,
    outputs: [&Buffer; 3],
    rows: usize,
    workspace: &Buffer,
) -> bool {
    let [gate, up, down] = weights;
    if !cmd.tensor_accelerated()
        || !compact_prefill_rows(rows)
        || weights.iter().any(|w| w.ty != AFFINE4)
        || (gate.k, gate.n, up.k, up.n, down.k, down.n) != (5120, 17408, 5120, 17408, 17408, 5120)
    {
        return false;
    }
    #[cfg(test)]
    if BASELINE_FFN_FOR_TEST.with(|v| v.get()) {
        return false;
    }
    let [g, u, out] = outputs;
    let padded_rows = rows.next_multiple_of(128);
    assert!(workspace.len() >= padded_rows * down.k * 2);
    project_group(
        cmd,
        &[(gate, g), (up, u)],
        input,
        rows,
        workspace,
        (Arithmetic::Adaptive, 0, false, true),
    );
    cmd.dispatch(
        "mlx_swiglu_compact",
        &[g, u, workspace],
        &[(rows * down.k) as u32, (padded_rows * down.k) as u32],
        [(padded_rows * down.k).div_ceil(256), 1, 1],
        256,
    );
    project_group(
        cmd,
        &[(down, out)],
        input,
        rows,
        workspace,
        (Arithmetic::Adaptive, 0, true, false),
    );
    true
}

fn project_inner(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
    (arithmetic, first): (Arithmetic, usize),
) {
    assert!((1..=4).contains(&planes.len()));
    let k = planes[0].0.k;
    assert!(planes.iter().all(|(w, _)| w.ty == AFFINE4 && w.k == k));
    assert!(input.len() >= (first + rows) * k * 4);
    assert!(
        planes
            .iter()
            .all(|(w, out)| out.len() >= (first + rows) * w.n * 4)
    );
    // DeltaNet's wide QKV/Z pair and narrow alpha/beta pair consume the same
    // activation. Keep their individual tile/reduction elections, but prepare
    // the padded/compact input (or verifier bias sums) only once. Split-K
    // intermediates live after the input prefix, so cannot invalidate it.
    let width = if planes.len() == 4 { 2 } else { planes.len() };
    for (i, group) in planes.chunks(width).enumerate() {
        let prepared = i != 0;
        #[cfg(test)]
        let prepared = prepared && !SEPARATE_INPUTS_FOR_TEST.with(|v| v.get());
        project_group(
            cmd,
            group,
            input,
            rows,
            workspace,
            (arithmetic, first, prepared, false),
        );
    }
}

fn project_group(
    cmd: &Commands<'_>,
    planes: &[(&Weight, &Buffer)],
    input: &Buffer,
    rows: usize,
    workspace: &Buffer,
    (arithmetic, first, prepared, compact): (Arithmetic, usize, bool, bool),
) {
    assert!((1..=3).contains(&planes.len()) && rows > 0);
    let k = planes[0].0.k;
    assert!(planes.iter().all(|(w, _)| w.ty == AFFINE4 && w.k == k));
    let decode = match arithmetic {
        Arithmetic::Verify(single) => Some(single),
        Arithmetic::Llama(logical) => Some(logical == 1),
        _ => None,
    };
    let logical_rows = if let Arithmetic::Llama(logical) = arithmetic {
        logical
    } else {
        rows
    };
    let vector_limit = if matches!(arithmetic, Arithmetic::Llama(_)) {
        llama_vector_limit(cmd.tensor_accelerated(), planes)
    } else {
        13
    };
    let prefill = matches!(arithmetic, Arithmetic::StablePrefill)
        || (matches!(arithmetic, Arithmetic::Adaptive | Arithmetic::Llama(_))
            && logical_rows >= vector_limit);
    let fast_contract = matches!(arithmetic, Arithmetic::StableDecode);
    let contraction_rows = if matches!(arithmetic, Arithmetic::StablePrefill) {
        512
    } else {
        cmd.affine_prefill_rows().unwrap_or(logical_rows)
    };
    let single = decode.unwrap_or(rows == 1) && k.is_multiple_of(512) && !fast_contract;
    // Minimize weight re-reads, then balance the last vector tile: six rows
    // use 3+3, not 5+1. The reference uses this same arithmetic grouping.
    let vector_rows = rows.div_ceil(rows.div_ceil(5));
    let verify_rows = if rows <= 4 {
        rows
    } else if rows <= 6 && cmd.tensor_accelerated() {
        3
    } else {
        4
    };
    let verify_columns = verify_columns(
        k,
        planes
            .iter()
            .map(|(w, _)| w.n)
            .max()
            .expect("projection validated one to three planes"),
        rows,
        cmd.tensor_accelerated(),
    );
    // Broad complete tiles and qualified nearly-full tails amortize the
    // packed-weight loader. Other ragged, narrow and short batches keep the
    // small tile. This changes storage/reuse, not the K reduction tree.
    let wide_staging = prefill
        && cmd.tensor_accelerated()
        && cmd.affine_prefill_rows() == Some(512)
        && compact_prefill_rows(rows)
        && matches!(k, 5120 | 6144 | 17408)
        && planes.iter().all(|(w, _)| w.n >= 1024);
    #[cfg(test)]
    let wide_staging = wide_staging && !BASELINE_STAGING_FOR_TEST.with(|v| v.get());
    #[cfg(test)]
    let wide_staging = wide_staging
        && STAGING_MASK_FOR_TEST.with(|v| {
            let bit = match (k, planes[0].0.n) {
                (5120, 17408) => 1,
                (17408, _) => 2,
                (5120, 12288) => 4,
                (5120, 10240) => 8,
                (6144, _) => 16,
                // The default diagnostic mask must retain the serving
                // election even for unrelated synthetic output widths.
                _ => 31,
            };
            v.get() & bit != 0
        });
    let tile = if wide_staging {
        128
    } else {
        prefill_tile(
            rows,
            k,
            planes
                .iter()
                .map(|(w, _)| w.n)
                .min()
                .expect("projection validated one to three planes"),
            cmd.tensor_accelerated(),
        )
    };
    assert!(
        !compact
            || (prefill
                && (tile == 256 || wide_staging)
                && compact_prefill_rows(rows)
                && planes
                    .iter()
                    .all(|(w, _)| w.n.is_multiple_of(32) && partitions(k, w.n, rows) == 1))
    );
    let (x, columns, row_groups) = if prefill {
        let elements = k.div_ceil(128) * 128 * rows.div_ceil(128) * 128;
        assert!(workspace.len() >= elements * 2);
        if !prepared {
            cmd.dispatch_at(
                "mlx_input",
                &[input, workspace],
                &[first * k * 4, 0],
                &[k as u32, rows as u32],
                [elements.div_ceil(256), 1, 1],
                256,
            );
        }
        (workspace, 32, rows.div_ceil(tile))
    } else if single {
        (input, verify_columns, rows.div_ceil(verify_rows))
    } else if rows > 1 || fast_contract {
        assert!(workspace.len() >= rows * k * 2);
        if !prepared {
            cmd.dispatch_at(
                "mlx_input_compact",
                &[input, workspace],
                &[first * k * 4, 0],
                &[k as u32, rows as u32],
                [(rows * k).div_ceil(256), 1, 1],
                256,
            );
        }
        (workspace, 8, rows.div_ceil(vector_rows))
    } else {
        (
            input,
            if single {
                16
            } else if vector_rows > 1 {
                8
            } else {
                32
            },
            rows.div_ceil(vector_rows),
        )
    };
    // Only measured M5 dense-27B wide projections use the rounded bulk store.
    // The tile shape, K contraction and split-K selection remain unchanged.
    let bulk_store = cmd.tensor_accelerated() && matches!(k, 5120 | 17408);
    // The short first-admission wave keeps its 32-row tile and exact BF16
    // partial tree. Pad weight staging and reuse each scale/bias across more
    // packed codes, as on the wide route; do not widen the admission grant.
    let tiny_prompt = matches!(arithmetic, Arithmetic::StablePrefill) && rows < 13;
    #[cfg(test)]
    let tiny_prompt = tiny_prompt && !BASELINE_RAGGED_FOR_TEST.with(|v| v.get());
    let small_admission = cmd.tensor_accelerated()
        && matches!(k, 5120 | 17408)
        && ((13..=32).contains(&rows) || tiny_prompt);
    #[cfg(test)]
    let small_admission = small_admission && !BASELINE_ADMISSION_FOR_TEST.with(|v| v.get());
    #[cfg(test)]
    let bulk_store = bulk_store && !BASELINE_STORE_FOR_TEST.with(|v| v.get());
    let kernel = if prefill {
        match tile {
            128 if compact => "mlx_affine_prefill_compact128",
            128 if k == 6144 => "mlx_affine_prefill_deep128",
            128 => "mlx_affine_prefill_wide128",
            32 if small_admission => "mlx_affine_prefill_load32_m32",
            32 => "mlx_affine_tile32",
            256 if compact => "mlx_affine_prefill_compact256",
            256 if bulk_store => "mlx_affine_prefill_store256",
            256 => "mlx_affine_prefill_rows256",
            _ if !cmd.tensor_accelerated() => "mlx_affine_prefill64",
            _ => "mlx_affine_prefill_load32",
        }
    } else if single {
        match (verify_rows, verify_columns) {
            (1, _) => "mlx_affine_single",
            (2, 4) => "mlx_affine_verify_narrow2",
            (3, 4) => "mlx_affine_verify_narrow3",
            (_, 4) => "mlx_affine_verify_narrow4",
            (2, 8) => "mlx_affine_verify_half2",
            (3, 8) => "mlx_affine_verify_half3",
            (2, _) => "mlx_affine_verify_single2",
            (3, _) => "mlx_affine_verify_single3",
            _ => "mlx_affine_verify_single4",
        }
    } else if fast_contract {
        [
            "mlx_affine_stable1",
            "mlx_affine_stable2",
            "mlx_affine_stable3",
            "mlx_affine_stable4",
            "mlx_affine_stable5",
        ][vector_rows - 1]
    } else {
        [
            "mlx_affine1",
            "mlx_affine_compact2",
            "mlx_affine_compact3",
            "mlx_affine_compact4",
            "mlx_affine_compact5",
        ][vector_rows - 1]
    };
    // Read one complete affine group per lane, preserving the fixed F32
    // contraction. R2 loses to the old loader; narrow gates also stay there.
    // Election is restricted to the measured M5 dense-27B geometries.
    let packed = fast_contract
        && rows >= 3
        && cmd.tensor_accelerated()
        && matches!(k, 5120 | 6144 | 17408)
        && planes.iter().all(|(w, _)| w.n >= 1024)
        && (3..=5).contains(&vector_rows);
    #[cfg(test)]
    let packed = packed && !BASELINE_PACKED_FOR_TEST.with(|v| v.get());
    let kernel = if packed {
        match vector_rows {
            3 => "mlx_affine_packed3",
            4 => "mlx_affine_packed4",
            5 => "mlx_affine_packed5",
            _ => kernel,
        }
    } else {
        kernel
    };
    if prefill
        && planes
            .iter()
            .any(|(w, _)| partitions(k, w.n, contraction_rows) > 1)
    {
        let prefix = k.div_ceil(128) * 128 * rows.div_ceil(128) * 128;
        for &(w, out) in planes {
            let parts = partitions(k, w.n, contraction_rows);
            if parts == 1 {
                cmd.dispatch_at(
                    kernel,
                    &[&w.buffer, &w.buffer, &w.buffer, x, out, out, out],
                    &[
                        0,
                        0,
                        0,
                        0,
                        first * w.n * 4,
                        first * w.n * 4,
                        first * w.n * 4,
                    ],
                    &[k as u32, w.n as u32, 0, 0, rows as u32],
                    [w.n.div_ceil(32), row_groups, 1],
                    128,
                );
            } else {
                assert!(workspace.len() >= (prefix + parts * rows * w.n) * 2);
                let p = [
                    k as u32,
                    w.n as u32,
                    rows as u32,
                    parts as u32,
                    prefix as u32,
                ];
                cmd.dispatch(
                    if small_admission {
                        "mlx_affine_parts_padded"
                    } else {
                        "mlx_affine_parts"
                    },
                    &[&w.buffer, workspace],
                    &p,
                    [w.n.div_ceil(32), rows.div_ceil(32), parts],
                    128,
                );
                cmd.dispatch_at(
                    "mlx_affine_join",
                    &[workspace, out],
                    &[0, first * w.n * 4],
                    &p,
                    [
                        (rows * w.n * if parts >= 32 { 32 } else { 1 }).div_ceil(256),
                        1,
                        1,
                    ],
                    256,
                );
            }
        }
        return;
    }
    // Independent domains remain fused even with very narrow alpha/beta planes.
    let second = planes.get(1).unwrap_or(&planes[0]);
    let third = planes.get(2).unwrap_or(second);
    // This input-only reduction is shared by every column and fused plane.
    // Reuse existing scratch; never allocate per-token GPU or host storage.
    let buffers = [
        &planes[0].0.buffer,
        &second.0.buffer,
        &third.0.buffer,
        x,
        planes[0].1,
        second.1,
        third.1,
        workspace,
    ];
    if single && rows > 1 && !prepared {
        assert!(workspace.len() >= rows * k / 16 * 4);
        cmd.dispatch_at(
            "mlx_affine_bias",
            &[input, workspace],
            &[first * k * 4, 0],
            &[k as u32, rows as u32],
            [(rows * k / 16).div_ceil(256), 1, 1],
            256,
        );
    }
    let offsets = [
        0,
        0,
        0,
        if std::ptr::eq(x, input) {
            first * k * 4
        } else {
            0
        },
        first * planes[0].0.n * 4,
        first * second.0.n * 4,
        first * third.0.n * 4,
        0,
    ];
    cmd.dispatch_at(
        kernel,
        &buffers[..if single && rows > 1 { 8 } else { 7 }],
        &offsets[..if single && rows > 1 { 8 } else { 7 }],
        &[
            k as u32,
            planes[0].0.n as u32,
            if planes.len() >= 2 {
                second.0.n as u32
            } else {
                0
            },
            if planes.len() == 3 {
                third.0.n as u32
            } else {
                0
            },
            rows as u32,
        ],
        [
            planes.iter().map(|(w, _)| w.n.div_ceil(columns)).sum(),
            row_groups,
            1,
        ],
        if prefill || single || (rows == 1 && !fast_contract) {
            128
        } else {
            64
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifier_columns_keep_large_working_sets_on_qualified_routes() {
        assert_eq!(verify_columns(5120, 48, 2, true), 4);
        assert_eq!(verify_columns(17408, 5120, 2, true), 8);
        assert_eq!(verify_columns(17408, 5120, 3, true), 8);
        assert_eq!(verify_columns(17408, 5120, 6, true), 16);
        assert_eq!(verify_columns(5120, 17408, 6, true), 8);
        assert_eq!(verify_columns(5120, 17408, 4, true), 16);
        assert_eq!(verify_columns(5120, 48, 1, true), 16);
        assert_eq!(verify_columns(5120, 48, 6, false), 16);
        assert_eq!(verify_columns(2048, 48, 2, true), 16);
    }

    #[test]
    fn wide_prefill_requires_acceleration_and_qualified_padded_tiles() {
        assert_eq!(prefill_tile(32, 5120, 17408, true), 32);
        for rows in [33, 65, 129, 255, 257, 384, 385, 479, 513] {
            assert_eq!(prefill_tile(rows, 17408, 5120, true), 64);
        }
        for rows in [480, 496, 511] {
            for k in [5120, 6144, 17408] {
                assert_eq!(prefill_tile(rows, k, 1024, true), 256);
                assert_eq!(prefill_tile(rows, k, 48, true), 64);
                assert_eq!(prefill_tile(rows, k, 1024, false), 64);
            }
            assert_eq!(prefill_tile(rows, 4096, 1024, true), 64);
        }
        assert_eq!(prefill_tile(256, 5120, 17408, true), 64);
        assert_eq!(prefill_tile(256, 17408, 5120, true), 256);
        assert_eq!(prefill_tile(512, 5120, 17408, true), 256);
        assert_eq!(prefill_tile(512, 5120, 48, true), 64);
        assert_eq!(prefill_tile(512, 512, 17408, true), 64);
        assert_eq!(prefill_tile(512, 5120, 17408, false), 64);
    }

    #[test]
    fn speculative_rows_preserve_decode_projection_arithmetic() {
        let device = MetalDevice::new(Some(128 << 20)).unwrap();
        for k in [512, 1024, 5120, 17408] {
            for n in [48, 257] {
                // Tiny planes and ragged final columns.
                let mut bytes: Vec<u8> = (0..k * n / 8)
                    .flat_map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes())
                    .collect();
                for bias in [false, true] {
                    bytes.extend((0..k * n / 64).flat_map(|i| {
                        let bits = (i as u32).wrapping_mul(1664525).wrapping_add(1013904223);
                        // Vary both exponent and mantissa. Tiny integer/power-of-two
                        // fixtures can hide reassociation until a full model runs.
                        let v = if bias {
                            -((bits % 499 + 1) as f32) / 113.
                        } else {
                            (bits % 997 + 1) as f32 / 9973.
                        };
                        half::bf16::from_f32(v).to_bits().to_le_bytes()
                    }));
                }
                let weight = Weight {
                    buffer: device.upload(&bytes).unwrap(),
                    ty: AFFINE4,
                    k,
                    n,
                };
                for rows in [2, 3, 4, 5, 6, 7, 8, 15, 32] {
                    let values: Vec<f32> = (0..k * rows)
                        .map(|i| {
                            half::bf16::from_f32(((i * 37) % 1999) as f32 / 113. - 8.).to_f32()
                        })
                        .collect();
                    let input = device
                        .upload(
                            &values
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                    let output = device
                        .upload(
                            &(0..rows * n + 32)
                                .flat_map(|_| f32::NAN.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                    let workspace = device.alloc(workspace_bytes(k, n, rows.max(4))).unwrap();
                    for single in [true, false] {
                        unsafe {
                            output.write_u32(&vec![u32::MAX; rows * n + 32]);
                        }
                        let cmd = device.begin().unwrap();
                        project_verify(
                            &cmd,
                            &[(&weight, &output)],
                            &input,
                            rows,
                            &workspace,
                            single,
                        );
                        cmd.finish().unwrap();
                        let got = unsafe { output.read_f32(0, rows * n) };
                        assert!(
                            unsafe { output.read_f32(rows * n, 32) }
                                .iter()
                                .all(|v| v.is_nan())
                        );
                        for r in 0..rows {
                            let copies = if single { 1 } else { 4 };
                            let reference_input = device
                                .upload(
                                    &(0..copies)
                                        .flat_map(|_| {
                                            values[r * k..(r + 1) * k]
                                                .iter()
                                                .flat_map(|v| v.to_le_bytes())
                                        })
                                        .collect::<Vec<_>>(),
                                )
                                .unwrap();
                            let reference = device.alloc(copies * n * 4).unwrap();
                            let cmd = device.begin().unwrap();
                            project(
                                &cmd,
                                &[(&weight, &reference)],
                                &reference_input,
                                copies,
                                &workspace,
                            );
                            cmd.finish().unwrap();
                            let expected = unsafe { reference.read_f32(0, n) };
                            let mismatch = got[r * n..(r + 1) * n]
                                .iter()
                                .zip(&expected)
                                .enumerate()
                                .find(|(_, (actual, expected))| actual != expected);
                            assert!(
                                mismatch.is_none(),
                                "K={k}, rows={rows}, row={r}, single={single}, first mismatch {mismatch:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
