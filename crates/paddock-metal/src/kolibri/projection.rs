use super::*;

pub(super) fn project(cmd: &Commands<'_>, w: &Matrix, x: &Buffer, y: &Buffer, rows: usize) {
    assert_eq!(w.bits, 8);
    let p = [w.k as u32, w.n as u32, rows as u32, 8, 64, 1, 0];
    if rows >= 16 {
        cmd.dispatch(
            "q4a_mm8_packed",
            &[&w.buffer, x, y],
            &p,
            [w.n.div_ceil(32), rows.div_ceil(32), 1],
            128,
        );
    } else {
        // One independent contraction per logical row. Request arrival or a
        // different decode batch size must not change the reduction tree.
        cmd.dispatch(
            "q4a_mv",
            &[&w.buffer, x, y],
            &p,
            [w.n.div_ceil(16), rows, 1],
            128,
        );
    }
}

pub(super) fn norm(
    cmd: &Commands<'_>,
    x: &Buffer,
    w: &Buffer,
    y: &Buffer,
    width: usize,
    rows: usize,
) {
    cmd.dispatch(
        "mlx_rms",
        &[x, w, y],
        &[width as u32, 0, 1e-6f32.to_bits()],
        [rows, 1, 1],
        width.div_ceil(128) * 32,
    );
}

pub(super) fn experts(cmd: &Commands<'_>, l: &Layer, s: &Scratch, rows: usize) {
    cmd.dispatch(
        "linear",
        &[&l.router, &s.norm, &s.router],
        &[WIDTH as u32, EXPERTS as u32, rows as u32, 0, 1f32.to_bits()],
        [EXPERTS.div_ceil(4), rows, 1],
        128,
    );
    cmd.dispatch(
        "kolibri_route",
        &[&s.router, &l.bias, &s.picks, &s.probabilities],
        &[],
        [rows, 1, 1],
        32,
    );
    let grouped = rows >= 16;
    if grouped {
        cmd.dispatch(
            "moe_align",
            &[&s.picks, &s.lists, &s.counts],
            &[(rows * ACTIVE) as u32],
            [EXPERTS, 1, 1],
            256,
        );
        cmd.dispatch("kolibri_tiles", &[&s.counts, &s.tiles], &[], [1, 1, 1], 512);
    }
    for (w, input, output, per_entry) in [
        (&l.gate, &s.norm, &s.gate, false),
        (&l.up, &s.norm, &s.up, false),
    ] {
        expert_project(cmd, w, input, output, s, rows, per_entry, grouped);
    }
    cmd.dispatch(
        "mlx_swiglu",
        &[&s.gate, &s.up],
        &[(rows * ACTIVE * FF) as u32],
        [(rows * ACTIVE * FF).div_ceil(256), 1, 1],
        256,
    );
    expert_project(cmd, &l.down, &s.gate, &s.expert_out, s, rows, true, grouped);
    cmd.dispatch(
        "kolibri_fold",
        &[&s.expert_out, &s.probabilities, &s.delta],
        &[(rows * WIDTH) as u32],
        [(rows * WIDTH).div_ceil(256), 1, 1],
        256,
    );
}

#[allow(clippy::too_many_arguments)]
fn expert_project(
    cmd: &Commands<'_>,
    w: &Matrix,
    x: &Buffer,
    y: &Buffer,
    s: &Scratch,
    rows: usize,
    per_entry: bool,
    grouped: bool,
) {
    let n = w.n / EXPERTS;
    let p = [w.k as u32, n as u32, rows as u32, u32::from(per_entry)];
    if grouped {
        cmd.dispatch(
            "kolibri_grouped",
            &[&w.buffer, x, &s.lists, &s.counts, &s.tiles, y],
            &p,
            [n.div_ceil(32), (rows * ACTIVE).div_ceil(16) + EXPERTS, 1],
            128,
        );
    } else {
        cmd.dispatch(
            "kolibri_expert_mv",
            &[&w.buffer, x, &s.picks, y],
            &p,
            [n.div_ceil(16), rows * ACTIVE, 1],
            128,
        );
    }
}
