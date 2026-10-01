//! Flash Next's affine storage. This is deliberately separate from the
//! qualified dense-Qwen group-64 kernels: changing that arithmetic would
//! invalidate an unrelated model's parity gate. No host dequantization.
use crate::{
    device::{Buffer, Commands, MetalDevice, MetalError, Result},
    weights::Weight,
};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

pub(super) const A4G32: u32 = 0x101;
pub(super) const A8G64: u32 = 0x102;
#[cfg(test)]
thread_local! {
    pub(super) static SEPARATE_SPANS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static SHAPE_ONLY_SPANS_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static INLINE_INPUT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static PLAIN_DENSE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static PLAIN_SPLIT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static PADDED_TILES_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static TILE_REUSE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static STAGED_ROUTER_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static SHARED_INPUT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static JOINED_INPUT_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static JOINED_SLAB_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static JOINED_ROW_GROUP_FOR_TEST: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    pub(super) static PACKED_WIDE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static ROW_REUSE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Reuse unpacked weights without starving small projections of independent
/// threadgroups. Physical rows select reuse, never the contraction arithmetic:
/// the caller must first elect the wide contract from the logical prompt.
pub(super) fn wide_row_group(rows: usize, columns: usize) -> usize {
    let groups = columns.div_ceil(8);
    if (rows / 4) * groups >= 128 {
        4
    } else if (rows / 2) * groups >= 128 {
        2
    } else {
        1
    }
}

/// Valid only within an explicit group of projections of the same immutable
/// input. The arena's prefix contains BF16 rows; slab weights and staged
/// split-K partials live strictly after it. A fallback that can overwrite the
/// prefix invalidates this record. No buffer-address cache survives a call.
#[derive(Default)]
struct InputReuse {
    enabled: bool,
    staged: std::cell::Cell<Option<[u32; 3]>>,
}

impl InputReuse {
    fn stage(&self, cmd: &Commands<'_>, x: &Buffer, scratch: &Buffer, p: &[u32]) {
        let key = [p[0], p[2], p[6]]; // K, physical rows, source-row offset
        if self.enabled && self.staged.replace(Some(key)) == Some(key) {
            return;
        }
        cmd.dispatch(
            "q4a_input",
            &[x, scratch],
            p,
            [
                (p[2] as usize)
                    .next_multiple_of(32)
                    .saturating_mul(p[0] as usize)
                    .div_ceil(256),
                1,
                1,
            ],
            256,
        );
    }
}

/// Share preparation, never arithmetic: every weight still elects its own
/// coalesced spans, split-K count, quantization format and original kernel.
pub(super) fn project_group(
    cmd: &Commands<'_>,
    weights: &[(&Weight, &Buffer)],
    x: &Buffer,
    rows: usize,
) {
    #[cfg(test)]
    let enabled = SHARED_INPUT_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    let enabled = false; // exact, but no repeatable full-model performance win yet
    let input = InputReuse {
        enabled,
        ..Default::default()
    };
    let joined = project_input_pair(cmd, weights, x, rows);
    for &(w, y) in &weights[if joined { 2 } else { 0 }..] {
        project_reusing(cmd, w, x, y, rows, &input);
    }
}

/// Join QKV/Z launch grids, not their arithmetic. Keep the two packed weight
/// planes and output strides; a workgroup still owns exactly one original
/// output tile. Singleton recurrent gates deliberately remain vector kernels.
fn project_input_pair(
    cmd: &Commands<'_>,
    weights: &[(&Weight, &Buffer)],
    x: &Buffer,
    rows: usize,
) -> bool {
    #[cfg(test)]
    let enabled = JOINED_INPUT_FOR_TEST.with(|v| v.get())
        && !SEPARATE_SPANS_FOR_TEST.with(|v| v.get())
        && !INLINE_INPUT_FOR_TEST.with(|v| v.get())
        && !PLAIN_DENSE_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    let enabled = true;
    #[cfg(test)]
    let slab_enabled = JOINED_SLAB_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    let slab_enabled = false; // isolated win did not translate into a c=4 win
    if !enabled || rows < 64 || (!slab_enabled && rows >= 1536) {
        return false;
    }
    let Some(&(a, ay)) = weights.first() else {
        return false;
    };
    let Some(&(b, by)) = weights.get(1) else {
        return false;
    };
    let Some(scratch) = cmd.projection_workspace() else {
        return false;
    };
    if !cmd.tensor_accelerated()
        || cmd.independent_rows()
        || !padded_tiles()
        || !tile_reuse()
        || (a.ty, b.ty, a.k, b.k, a.n, b.n) != (A4G32, A4G32, 2560, 2560, 10240, 6144)
        || scratch.len() < 32 * a.k * 2
        || (cmd.projection_rows().is_none() && rows > MAX_LOGICAL_ROWS)
    {
        return false;
    }
    let default = [(0, rows, rows)];
    let spans = cmd.projection_rows().unwrap_or(&default);
    let plan = |w: &Weight| coalesced_spans(w.k, w.n, w.ty, Some(scratch.len()), spans);
    // Split-K can differ for the two N dimensions. Never coalesce by the
    // concatenated width, or perturb either projection's original row groups.
    if !plan(a).eq(plan(b)) {
        return false;
    }
    assert!(x.len() >= rows * a.k * 4 && ay.len() >= rows * a.n * 4 && by.len() >= rows * b.n * 4);
    for (start, count, logical) in plan(a) {
        if count < 64
            || [a, b]
                .iter()
                .any(|w| contraction(w.k, w.n, w.ty, logical) != (2, 1))
        {
            project_span_reusing(
                cmd,
                a,
                x,
                ay,
                (count, start, logical),
                &InputReuse::default(),
            );
            project_span_reusing(
                cmd,
                b,
                x,
                by,
                (count, start, logical),
                &InputReuse::default(),
            );
            continue;
        }
        #[cfg(test)]
        let slab = slab_enabled
            && slab_shape(a.k, a.n, count)
            && (scratch.len() / (a.k * 2)).saturating_sub(256) >= 32;
        #[cfg(not(test))]
        let slab = false;
        let capacity =
            ((scratch.len() / (a.k * 2)).saturating_sub(if slab { 256 } else { 0 }) / 32) * 32;
        for offset in (0..count).step_by(capacity) {
            let len = capacity.min(count - offset);
            let mut p = [0u32; 17];
            #[cfg(test)]
            {
                p[16] = JOINED_ROW_GROUP_FOR_TEST.with(|v| v.get());
            }
            #[cfg(not(test))]
            {
                p[16] = 0;
            }
            if p[16] == 0 {
                p[16] = if slab { 8 } else { 1 };
            }
            assert!([1, 4, 8, 16].contains(&p[16]));
            p[..7].copy_from_slice(&[
                a.k as u32,
                a.n as u32,
                len as u32,
                4,
                32,
                1,
                (start + offset) as u32,
            ]);
            p.copy_within(..7, 7);
            p[8] = b.n as u32;
            InputReuse::default().stage(cmd, x, scratch, &p);
            if slab {
                let input_bytes = len.next_multiple_of(32) * a.k * 2;
                // Every slab and the QKV/Z boundary align to full 64-column
                // tiles, so no tensor tile changes its shape or K reduction.
                let columns = (scratch.len() - input_bytes) / (a.k * 2) / 64 * 64;
                for first in (0..a.n + b.n).step_by(columns) {
                    let width = columns.min(a.n + b.n - first);
                    p[14] = first as u32;
                    p[15] = width as u32;
                    cmd.dispatch_at(
                        "q4a_weight_pair_slab",
                        &[&a.buffer, &b.buffer, scratch],
                        &[0, 0, input_bytes],
                        &p,
                        [(width * a.k / 32).div_ceil(256), 1, 1],
                        256,
                    );
                    cmd.dispatch_at(
                        "q4a_mm4_pair_slab",
                        &[scratch, scratch, ay, by],
                        &[input_bytes, 0, 0, 0],
                        &p,
                        [width.div_ceil(64), len.div_ceil(64), 1],
                        128,
                    );
                }
            } else {
                let wide = len >= 512;
                cmd.dispatch(
                    if wide {
                        "q4a_mm4_pair_wide"
                    } else {
                        "q4a_mm4_pair"
                    },
                    &[&a.buffer, &b.buffer, scratch, ay, by],
                    &p,
                    [
                        (a.n + b.n).div_ceil(if wide { 64 } else { 32 }),
                        len.div_ceil(32),
                        1,
                    ],
                    128,
                );
            }
        }
    }
    true
}

fn tile_reuse() -> bool {
    #[cfg(test)]
    return TILE_REUSE_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    true
}

/// Reuse only measured single-part M5 affine-4 shapes. Split-K and decode
/// keep their existing arithmetic. Small matrices retain on-chip unpacking.
fn slab_shape(k: usize, n: usize, rows: usize) -> bool {
    matches!((k, n), (2560, 6144 | 10240 | 12288)) && rows >= 1536
        || (k, n) == (320, 10240) && rows >= 512
        || (k, n) == (2560, 640) && rows >= 1024
}

// Reserve at least 256 weight columns alongside the padded input, then bound
// both physical row slices and column slabs by the actual arena. No global
// weight expansion, extra allocation, or change to logical contraction.
#[cfg(test)]
pub(super) fn project_slab(
    cmd: &Commands<'_>,
    w: &Buffer,
    x: &Buffer,
    scratch: &Buffer,
    y: &Buffer,
    p: &[u32; 7],
) -> bool {
    project_slab_reusing(cmd, w, x, scratch, y, p, &InputReuse::default())
}

fn project_slab_reusing(
    cmd: &Commands<'_>,
    w: &Buffer,
    x: &Buffer,
    scratch: &Buffer,
    y: &Buffer,
    p: &[u32; 7],
    input: &InputReuse,
) -> bool {
    let (k, n, rows) = (p[0] as usize, p[1] as usize, p[2] as usize);
    assert!(k.is_multiple_of(64) && k > 0 && n > 0 && rows > 0);
    assert_eq!((p[3], p[4], p[5]), (4, 32, 1));
    let minimum_columns = 256.min(n.next_multiple_of(32));
    let capacity = (scratch.len() / (k * 2)).saturating_sub(minimum_columns) / 32 * 32;
    if capacity < 32 {
        return false;
    }
    for offset in (0..rows).step_by(capacity) {
        let count = capacity.min(rows - offset);
        let input_bytes = count.next_multiple_of(32) * k * 2;
        let columns = ((scratch.len() - input_bytes) / (k * 2) / 32) * 32;
        let mut q = [0u32; 9];
        q[..7].copy_from_slice(p);
        q[2] = count as u32;
        q[6] += offset as u32;
        input.stage(cmd, x, scratch, &q);
        for first in (0..n).step_by(columns) {
            let width = columns.min((n - first).next_multiple_of(32));
            q[7] = first as u32;
            q[8] = width as u32;
            cmd.dispatch_at(
                "q4a_weight_slab",
                &[w, scratch],
                &[0, input_bytes],
                &q,
                [(width * k / 32).div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch_at(
                "q4a_mm4_slab",
                &[scratch, scratch, y],
                &[input_bytes, 0, 0],
                &q,
                [width.div_ceil(64), count.div_ceil(64), 1],
                128,
            );
        }
    }
    true
}

// Padding changes only the on-chip weight pitch, not the logical tensor,
// accumulation or split/join contract. The previous pitch remains a test
// comparator; production elects this only in the staged M5 matrix paths.
fn padded_tiles() -> bool {
    #[cfg(test)]
    return PADDED_TILES_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    true
}
// Split-K partials and bounded BF16 inputs share one reusable allocation
// with single-part dense and routed-expert input staging. Wide dense
// projections slice physical rows to fit, retaining their logical contraction;
// top-10 expert-down K=640 needs 6400 values per model row. These uses never
// overlap. No persistent expanded weights or post-load GPU allocation.
// Physical dispatch capacity is independent from a prompt's immutable
// arithmetic shape. Wider shared passes must not re-elect split-K or masks.
pub(super) const MAX_LOGICAL_ROWS: usize = 1024;
pub(super) const MAX_ROWS: usize = 2048;
#[cfg(test)]
pub(super) const WORKSPACE_BYTES: usize = MAX_ROWS * 6400 * 2;
pub(super) const fn workspace_bytes(capacity: usize) -> usize {
    capacity * 6400 * 2
}
pub(super) fn is_affine(ty: u32) -> bool {
    matches!(ty, A4G32 | A8G64)
}

pub(super) fn format(ty: u32) -> (usize, usize) {
    match ty {
        A4G32 => (4, 32),
        A8G64 => (8, 64),
        _ => panic!("not a Flash Next affine tensor"),
    }
}

/// Complete row-count-dependent dense contraction: vector, wide vector or
/// tiled split-K. Prefix compatibility consumes the very same election as
/// dispatch, not a separately maintained list of numerical breakpoints.
pub(super) fn contraction(k: usize, n: usize, ty: u32, rows: usize) -> (u8, usize) {
    if rows == 1 || n <= 48 {
        return (0, 1);
    }
    if rows < 13 {
        return (1, 1);
    }
    let align = format(ty).1.max(32);
    let mut parts = (512 / (n.div_ceil(32) * rows.div_ceil(32)))
        .min(k / align)
        .max(1);
    while !k.is_multiple_of(parts * align) {
        parts -= 1;
    }
    (2, parts)
}

/// Validate all three planes before allocating. Shape is logical row-major,
/// including the leading expert dimension, not the packed U32 shape.
pub(super) fn parts<'a>(
    source: &'a ShardedSafetensors,
    base: &str,
    shape: &[usize],
    ty: u32,
) -> Result<Vec<&'a [u8]>> {
    let (bits, group) = format(ty);
    if shape.len() < 2
        || shape.contains(&0)
        || !shape.last().is_some_and(|k| k.is_multiple_of(group))
    {
        return Err(MetalError::Model(format!(
            "{base}: invalid affine shape {shape:?}"
        )));
    }
    let mut result = Vec::with_capacity(3);
    for (suffix, dtype, divisor) in [
        ("weight", StDtype::U32, 32 / bits),
        ("scales", StDtype::Bf16, group),
        ("biases", StDtype::Bf16, group),
    ] {
        let name = format!("{base}.{suffix}");
        let mut expected = shape.to_vec();
        *expected
            .last_mut()
            .expect("affine shape has at least two validated dimensions") /= divisor;
        let (info, bytes) = source
            .bytes(&name)
            .ok_or_else(|| MetalError::Model(format!("missing {name}")))?;
        if info.dtype != dtype || info.shape != expected {
            return Err(MetalError::Model(format!(
                "{name}: expected {dtype:?} {expected:?}, got {:?} {:?}",
                info.dtype, info.shape
            )));
        }
        result.push(bytes);
    }
    Ok(result)
}

pub(super) fn load(
    d: &MetalDevice,
    source: &ShardedSafetensors,
    base: &str,
    shape: &[usize],
    ty: u32,
) -> Result<Weight> {
    let parts = parts(source, base, shape, ty)?;
    let size = parts.iter().map(|p| p.len()).sum();
    let buffer = d.upload_with(size, |out| {
        let mut offset = 0;
        for (part, suffix) in parts.iter().zip(["weight", "scales", "biases"]) {
            source
                .read_into(
                    &format!("{base}.{suffix}"),
                    &mut out[offset..offset + part.len()],
                )
                .map_err(|e| MetalError::Model(e.to_string()))?;
            offset += part.len();
        }
        Ok(())
    })?;
    Ok(Weight {
        buffer,
        ty,
        k: *shape
            .last()
            .expect("parts validated the affine shape before upload"),
        n: shape[..shape.len() - 1].iter().product(),
    })
}

/// Bounded tiled prefill and direct compressed-vector decode. The narrow
/// HC injection / recurrent gates use a row-invariant vector contraction,
/// as required by the upstream model's explicit singleton projection.
pub(super) fn project(cmd: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    project_reusing(cmd, w, x, y, rows, &InputReuse::default());
}

fn project_reusing(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    y: &Buffer,
    rows: usize,
    input: &InputReuse,
) {
    if !cmd.independent_rows()
        && let Some(spans) = cmd.projection_rows()
    {
        #[cfg(test)]
        if SEPARATE_SPANS_FOR_TEST.with(|v| v.get()) {
            for &(start, count, logical) in spans {
                project_span_reusing(cmd, w, x, y, (count, start, logical), input);
            }
            return;
        }
        let mut end = 0;
        for (start, count, logical_count) in coalesced_spans(
            w.k,
            w.n,
            w.ty,
            cmd.projection_workspace().map(Buffer::len),
            spans,
        ) {
            assert!(start == end && count > 0 && start + count <= rows);
            assert!(count <= MAX_ROWS && logical_count <= MAX_LOGICAL_ROWS);
            project_span_reusing(cmd, w, x, y, (count, start, logical_count), input);
            end += count;
        }
        assert_eq!(end, rows);
        return;
    }
    project_span_reusing(cmd, w, x, y, (rows, 0, rows), input);
}

/// Dense projections are row-local. Adjacent physical slices may share a
/// dispatch only when this weight's arithmetic contract is identical. Logical
/// lengths need not be equal: many lengths elect the same vector or split-K
/// contraction. Physical rows are bounded separately by capacity and the
/// actual split-K workspace. Do not re-elect arithmetic using the merged row
/// count: that would change vector/tile or partition order. This is per projection;
/// recurrent/attention/expert metadata and prefix compatibility are untouched.
fn coalesced_spans(
    k: usize,
    n: usize,
    ty: u32,
    workspace: Option<usize>,
    spans: &[(usize, usize, usize)],
) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
    let mut next = 0;
    std::iter::from_fn(move || {
        let &(start, mut count, mut logical) = spans.get(next)?;
        assert!(count > 0 && count <= logical && logical <= MAX_LOGICAL_ROWS);
        let signature = contraction(k, n, ty, logical);
        let capacity = if signature.0 == 2 && signature.1 > 1 {
            workspace.map_or(MAX_ROWS, |bytes| {
                MAX_ROWS.min(bytes / (n * signature.1 * 2))
            })
        } else {
            MAX_ROWS
        };
        next += 1;
        while let Some(&(at, len, contract)) = spans.get(next) {
            assert!(len > 0 && len <= contract && contract <= MAX_LOGICAL_ROWS);
            let compatible = contract == logical || signature == contraction(k, n, ty, contract);
            #[cfg(test)]
            let compatible =
                compatible && (!SHAPE_ONLY_SPANS_FOR_TEST.with(|v| v.get()) || contract == logical);
            #[cfg(test)]
            let capacity = if SHAPE_ONLY_SPANS_FOR_TEST.with(|v| v.get()) {
                logical
            } else {
                capacity
            };
            if at != start + count || !compatible || count + len > capacity {
                break;
            }
            count += len;
            logical = logical.max(contract);
            next += 1;
        }
        Some((start, count, logical))
    })
}

#[cfg(test)]
pub(super) fn project_span(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    y: &Buffer,
    rows: usize,
    start: usize,
    logical_rows: usize,
) {
    project_span_reusing(
        cmd,
        w,
        x,
        y,
        (rows, start, logical_rows),
        &InputReuse::default(),
    );
}

fn project_span_reusing(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    y: &Buffer,
    span: (usize, usize, usize),
    input: &InputReuse,
) {
    let (rows, start, logical_rows) = span;
    let (bits, group) = format(w.ty);
    assert!(rows > 0 && x.len() >= (start + rows) * w.k * 4 && y.len() >= (start + rows) * w.n * 4);
    let (kind, parts) = contraction(
        w.k,
        w.n,
        w.ty,
        if cmd.independent_rows() {
            1
        } else {
            logical_rows
        },
    );
    let tile = kind == 2;
    let wide = kind == 1;
    let packed_wide = cmd.tensor_accelerated();
    #[cfg(test)]
    let packed_wide = packed_wide && PACKED_WIDE_FOR_TEST.with(|v| v.get());
    #[cfg(test)]
    let row_reuse = ROW_REUSE_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    let row_reuse = true;
    let wide_rows = if wide && packed_wide && row_reuse {
        wide_row_group(rows, w.n)
    } else {
        1
    };
    // Test-binary-only arithmetic bisect. No runner setting or production
    // branch: isolate compiler specialization from the prefill graph.
    #[cfg(test)]
    let audit = *{
        static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        MODE.get_or_init(
            || match std::env::var("PADDOCK_FLASH_NEXT_MLX_MV_AUDIT").as_deref() {
                Ok("generic") => 1,
                Ok("entries") => 2,
                Err(_) => 0,
                _ => panic!("invalid test-only affine audit mode"),
            },
        )
    };
    #[cfg(not(test))]
    let audit = 0;
    if audit == 1 && !tile && !wide {
        cmd.dispatch(
            "q4a_mv",
            &[&w.buffer, x, y],
            &[
                w.k as u32,
                w.n as u32,
                rows as u32,
                bits as u32,
                group as u32,
                1,
                start as u32,
            ],
            [w.n.div_ceil(16), rows, 1],
            128,
        );
        return;
    }
    let narrow = audit != 2 && !tile && !wide && bits == 4 && w.n == 4;
    let half_group = !tile
        && !wide
        && audit != 2
        && bits == 4
        && w.k >= 8192
        && w.k.is_multiple_of(512)
        && w.n >= 8
        && w.n <= 512
        && w.n.is_multiple_of(8);
    #[cfg(test)]
    let staged_router = STAGED_ROUTER_FOR_TEST.with(|v| v.get());
    #[cfg(not(test))]
    let staged_router = true;
    let staged_split =
        tile && parts > 1 && (bits == 4 || staged_router) && rows >= 64 && cmd.tensor_accelerated();
    #[cfg(test)]
    let staged_split = staged_split && !PLAIN_SPLIT_FOR_TEST.with(|v| v.get());
    if staged_split && let Some(scratch) = cmd.projection_workspace() {
        // Reserve padded input and partials together. Bound physical slicing
        // by the actual arena, never by an assumed allocation. The logical
        // partition count stays fixed even at a short final physical slice.
        let capacity = (scratch.len() / (2 * (w.k + parts * w.n)) / 32) * 32;
        if capacity >= 32 {
            for offset in (0..rows).step_by(capacity) {
                let count = capacity.min(rows - offset);
                let padded = count.next_multiple_of(32);
                let partial_offset = padded * w.k * 2;
                assert!(partial_offset + count * w.n * parts * 2 <= scratch.len());
                let p = [
                    w.k as u32,
                    w.n as u32,
                    count as u32,
                    bits as u32,
                    group as u32,
                    parts as u32,
                    (start + offset) as u32,
                ];
                input.stage(cmd, x, scratch, &p);
                let span = w.k / parts;
                let kernel = if bits == 8 {
                    assert!(span.is_multiple_of(64));
                    match (span.is_multiple_of(128), padded_tiles()) {
                        (true, true) => "q4a_mm8_split_device128_pad8",
                        (false, true) => "q4a_mm8_split_device64_pad8",
                        (true, false) => "q4a_mm8_split_device128",
                        (false, false) => "q4a_mm8_split_device64",
                    }
                } else if padded_tiles() && span.is_multiple_of(128) {
                    "q4a_mm4_split_device128_pad8"
                } else if padded_tiles() && span.is_multiple_of(64) {
                    "q4a_mm4_split_device64_pad8"
                } else if padded_tiles() {
                    "q4a_mm4_split_device32_pad8"
                } else if span.is_multiple_of(128) {
                    "q4a_mm4_split_device128_group32"
                } else if span.is_multiple_of(64) {
                    "q4a_mm4_split_device64_group32"
                } else {
                    "q4a_mm4_split_device32_group32"
                };
                cmd.dispatch_at(
                    kernel,
                    &[&w.buffer, scratch, scratch],
                    &[0, 0, partial_offset],
                    &p,
                    [w.n.div_ceil(32), count.div_ceil(32), parts],
                    128,
                );
                cmd.dispatch_at(
                    "q4a_mm_join",
                    &[scratch, y],
                    &[partial_offset, 0],
                    &p,
                    [(count * w.n).div_ceil(256), 1, 1],
                    256,
                );
            }
            return;
        }
    }
    let hc_down = bits == 4 && w.k == 10240 && w.n == 320;
    #[cfg(test)]
    let hc_down = hc_down && !PLAIN_SPLIT_FOR_TEST.with(|v| v.get());
    let staged_input = tile
        && parts == 1
        && cmd.tensor_accelerated()
        && rows >= 64
        && ((w.n >= 512 && w.k <= 6144) || hc_down)
        && w.k.is_multiple_of(64);
    #[cfg(test)]
    let staged_input = staged_input && !INLINE_INPUT_FOR_TEST.with(|v| v.get());
    if staged_input
        && let Some(scratch) = cmd.projection_workspace()
        && scratch.len() / (w.k * 2) >= 32
    {
        if bits == 4
            && tile_reuse()
            && slab_shape(w.k, w.n, rows)
            && project_slab_reusing(
                cmd,
                &w.buffer,
                x,
                scratch,
                y,
                &[w.k as u32, w.n as u32, rows as u32, 4, 32, 1, start as u32],
                input,
            )
        {
            return;
        }
        let capacity = (scratch.len() / (w.k * 2) / 32) * 32;
        for offset in (0..rows).step_by(capacity) {
            let count = capacity.min(rows - offset);
            let p = [
                w.k as u32,
                w.n as u32,
                count as u32,
                bits as u32,
                group as u32,
                1,
                (start + offset) as u32,
            ];
            input.stage(cmd, x, scratch, &p);
            #[cfg(test)]
            let grouped = !PLAIN_DENSE_FOR_TEST.with(|v| v.get());
            #[cfg(not(test))]
            let grouped = true;
            let wide = bits == 4
                && grouped
                && padded_tiles()
                && tile_reuse()
                && count >= 512
                && matches!((w.k, w.n), (2560, 6144 | 10240 | 12288) | (6144, 2560));
            cmd.dispatch(
                match (bits, w.k.is_multiple_of(128), grouped) {
                    (4, true, true) if wide => "q4a_mm4_device128_wide",
                    (4, true, true) if padded_tiles() => "q4a_mm4_device128_pad8",
                    (4, false, true) if padded_tiles() => "q4a_mm4_device64_pad8",
                    (4, true, true) => "q4a_mm4_device128_group32",
                    (4, false, true) => "q4a_mm4_device64_group32",
                    (4, true, false) => "q4a_mm4_device128",
                    (4, false, false) => "q4a_mm4_device64",
                    (8, true, _) => "q4a_mm8_device128",
                    (8, false, _) => "q4a_mm8_device64",
                    _ => unreachable!(),
                },
                &[&w.buffer, scratch, y],
                &p,
                [
                    w.n.div_ceil(if wide { 64 } else { 32 }),
                    count.div_ceil(32),
                    1,
                ],
                128,
            );
        }
        return;
    }
    input.staged.set(None);
    if tile
        && parts > 1
        && let Some(scratch) = cmd.projection_workspace()
    {
        assert!(scratch.len() >= rows * w.n * parts * 2);
        let p = [
            w.k as u32,
            w.n as u32,
            rows as u32,
            bits as u32,
            group as u32,
            parts as u32,
            start as u32,
        ];
        cmd.dispatch(
            "q4a_mm_split",
            &[&w.buffer, x, scratch],
            &p,
            [w.n.div_ceil(32), rows.div_ceil(32), parts],
            128,
        );
        cmd.dispatch(
            "q4a_mm_join",
            &[scratch, y],
            &p,
            [(rows * w.n).div_ceil(256), 1, 1],
            256,
        );
        return;
    }
    cmd.dispatch(
        if tile && parts == 1 {
            if bits == 4 {
                "q4a_mm4_packed"
            } else {
                "q4a_mm8_packed"
            }
        } else if tile {
            "q4a_mm"
        } else if wide {
            match (packed_wide, bits, wide_rows) {
                (true, 4, 4) => "q4a_wide4_rows4",
                (true, 8, 4) => "q4a_wide8_rows4",
                (true, 4, 2) => "q4a_wide4_rows2",
                (true, 8, 2) => "q4a_wide8_rows2",
                (true, 4, _) => "q4a_wide4_packed",
                (true, 8, _) => "q4a_wide8_packed",
                _ => "q4a_wide",
            }
        } else if narrow {
            "q4a_mv4_narrow"
        } else if half_group {
            "q4a_mv4_fast2"
        } else {
            let fast = w.k.is_multiple_of((32 / bits) * 64) && w.n >= 8 && w.n.is_multiple_of(8);
            match (bits, fast) {
                (4, true) => "q4a_mv4_fast",
                (4, false) => "q4a_mv4",
                (8, true) => "q4a_mv8_fast",
                (8, false) => "q4a_mv8",
                _ => unreachable!(),
            }
        },
        &[&w.buffer, x, y],
        &[
            w.k as u32,
            w.n as u32,
            rows as u32,
            bits as u32,
            group as u32,
            parts as u32,
            start as u32,
        ],
        [
            w.n.div_ceil(if tile {
                32
            } else if wide {
                8
            } else if narrow {
                4
            } else if half_group {
                8
            } else {
                16
            }),
            rows.div_ceil(if tile { 32 } else { wide_rows }),
            1,
        ],
        if wide || half_group { 64 } else { 128 },
    );
}

pub(super) fn gather(cmd: &Commands<'_>, w: &Weight, ids: &Buffer, y: &Buffer, rows: usize) {
    let (bits, group) = format(w.ty);
    assert!(ids.len() >= rows * 4 && y.len() >= rows * w.k * 4);
    cmd.dispatch(
        "q4a_gather",
        &[&w.buffer, ids, y],
        &[
            w.k as u32,
            w.n as u32,
            rows as u32,
            bits as u32,
            group as u32,
        ],
        [(rows * w.k).div_ceil(256), 1, 1],
        256,
    );
}

/// Expert entries stay GPU-resident. `input_per_entry` distinguishes the
/// repeated input rows of gate/up from the routed activation rows of down.
#[cfg(test)]
pub(super) fn experts(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    ids: &Buffer,
    y: &Buffer,
    entries: usize,
    input_per_entry: bool,
) {
    experts_ordered(cmd, w, x, ids, y, entries, input_per_entry, None);
}

pub(super) fn experts_ordered(
    cmd: &Commands<'_>,
    w: &Weight,
    x: &Buffer,
    ids: &Buffer,
    y: &Buffer,
    entries: usize,
    input_per_entry: bool,
    order: Option<&Buffer>,
) {
    assert_eq!(w.ty, A4G32);
    assert!(w.n.is_multiple_of(512));
    let n = w.n / 512;
    assert!(entries > 0 && entries <= MAX_ROWS * 10 && entries.is_multiple_of(10));
    assert!(ids.len() >= entries * 4 && y.len() >= entries * n * 4);
    assert!(
        x.len()
            >= (if input_per_entry {
                entries
            } else {
                entries / 10
            }) * w.k
                * 4
    );
    let buffers = [&w.buffer, x, ids, y, order.unwrap_or(ids)];
    if let Some(order) = order {
        assert!(order.len() >= entries * 4);
    }
    // Same arithmetic, shared compressed loads. Keep sparse/decode and the
    // narrower down contraction on their established vector programs.
    let paired = order.is_some() && entries >= 640 && w.k == 2560 && n == 640;
    cmd.dispatch(
        if paired {
            "q4a_expert4_fast_pair"
        } else {
            match (
                w.k.is_multiple_of(512) && n >= 8 && n.is_multiple_of(8),
                order.is_some(),
            ) {
                (true, true) => "q4a_expert4_fast_ordered",
                (false, true) => "q4a_expert4_ordered",
                (true, false) => "q4a_expert4_fast",
                (false, false) => "q4a_expert4",
            }
        },
        &buffers,
        &[
            w.k as u32,
            n as u32,
            entries as u32,
            u32::from(input_per_entry),
            512,
        ],
        if paired {
            [n.div_ceil(16) * 4, entries.div_ceil(8), 1]
        } else if order.is_some() {
            [n.div_ceil(16) * 8, entries.div_ceil(8), 1]
        } else {
            [n.div_ceil(16), entries, 1]
        },
        128,
    );
}

#[cfg(test)]
mod span_tests {
    use super::{
        A4G32, A8G64, MAX_LOGICAL_ROWS, MAX_ROWS, WORKSPACE_BYTES, coalesced_spans, contraction,
    };

    #[test]
    fn dense_coalescing_preserves_contract_and_scratch_bound() {
        let spans = [
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 1024),
            (3, 255, 1024),
            (258, 256, 1024),
            (514, 512, 1024),
            (1026, 128, 1024),
            (1154, 127, 128),
            (1281, 1, 128),
            (1282, 1, 1),
        ];
        assert_eq!(
            coalesced_spans(10240, 320, A4G32, Some(WORKSPACE_BYTES), &spans).collect::<Vec<_>>(),
            [(0, 2, 1), (2, 1152, 1024), (1154, 128, 128), (1282, 1, 1),]
        );
        assert_eq!(coalesced_spans(10240, 320, A4G32, None, &[]).count(), 0);
        assert_eq!(
            coalesced_spans(10240, 320, A4G32, None, &[(0, 1, 4), (2, 1, 4)]).count(),
            2
        );
    }

    #[test]
    fn unequal_lengths_merge_only_with_the_same_weight_contraction() {
        let spans = [
            (0, 1, 1),
            (1, 17, 700),
            (18, 15, 750),
            (33, 31, 800),
            (64, 64, 1024),
            (128, 31, 900),
            (159, 1, 1),
        ];
        assert_eq!(
            coalesced_spans(10240, 320, A4G32, Some(WORKSPACE_BYTES), &spans).collect::<Vec<_>>(),
            [(0, 1, 1), (1, 63, 800), (64, 95, 1024), (159, 1, 1)]
        );
        assert_eq!(
            coalesced_spans(2560, 6144, A4G32, Some(WORKSPACE_BYTES), &spans).collect::<Vec<_>>(),
            [(0, 1, 1), (1, 158, 1024), (159, 1, 1)]
        );
        for (k, n, ty) in [
            (10240, 320, A4G32),
            (2560, 6144, A4G32),
            (2560, 512, A8G64),
            (10240, 4, A4G32),
            (320, 10240, A4G32),
        ] {
            for a in 1..=MAX_LOGICAL_ROWS {
                for b in 1..=MAX_LOGICAL_ROWS {
                    let rows = [(0, 1, a), (1, 1, b)];
                    let merged =
                        coalesced_spans(k, n, ty, Some(WORKSPACE_BYTES), &rows).collect::<Vec<_>>();
                    assert_eq!(merged.iter().map(|s| s.1).sum::<usize>(), 2);
                    for &(start, count, logical) in &merged {
                        assert!(count <= MAX_ROWS && logical <= MAX_ROWS);
                        let (kind, parts) = contraction(k, n, ty, logical);
                        if kind == 2 && parts > 1 {
                            assert!(count * n * parts * 2 <= WORKSPACE_BYTES);
                        }
                        for s in &rows[start..start + count] {
                            assert_eq!(contraction(k, n, ty, logical), contraction(k, n, ty, s.2));
                        }
                    }
                }
            }
        }
        let rows = [(0, 64, 64), (64, 64, 64)];
        assert_eq!(
            coalesced_spans(10240, 320, A4G32, Some(1 << 20), &rows).count(),
            2
        );
        assert_eq!(
            coalesced_spans(10240, 320, A4G32, Some(WORKSPACE_BYTES), &rows).collect::<Vec<_>>(),
            [(0, 128, 64)]
        );
    }
}
