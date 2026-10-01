//! qwen4exp loader oracle (the GPU side of the loader-oracle pattern): upload one GDN
//! layer, one attention layer and their MoE planes, then read every plane
//! BACK and compare against the checkpoint bytes / an exact host widen.
//! Byte-equality is the whole gate - the charter is bf16 parity, so any
//! difference is a loader bug, never tolerance.
//!
//! Sized to run beside a live serve: two layers ~= 1.5 GB (the 51 GB PLE
//! table is deliberately not loaded here; its projections are).

mod common;

use paddock_engine::gpu_model::qwen4exp::{MixerW, load_layer, load_ple_projections};
use paddock_models::modelopt::nvfp4_view;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// The exact f32 value of a checkpoint plane the engine widens on the host:
/// bf16 as shipped, or MXFP8 - OCP e4m3 (bias 7, 0x7F/0xFF NaN, exponent 0
/// subnormal) times its block's ue8m0 scale (2^(b-127), 0xFF NaN), one scale
/// per 32 values. Written from the spec here rather than borrowed from the
/// loader, so the gate is an oracle and not the loader checking itself.
fn exact_f32(st: &ShardedSafetensors, name: &str) -> Vec<f32> {
    let (t, bytes) = st.bytes(name).unwrap_or_else(|| panic!("{name}: missing"));
    match t.dtype {
        StDtype::Bf16 => bf16_to_f32(bytes),
        StDtype::F8E4m3 => {
            let (_, sb) = st
                .bytes(&format!("{name}_scale"))
                .unwrap_or_else(|| panic!("{name}_scale: missing"));
            let e4m3 = |b: u8| -> f32 {
                let (sign, exp, man) = (b >> 7, (b >> 3) & 0x0F, b & 0x07);
                let mag = if exp == 0x0F && man == 7 {
                    f32::NAN
                } else if exp == 0 {
                    man as f32 / 8.0 * 2f32.powi(-6)
                } else {
                    (1.0 + man as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
                };
                if sign == 1 { -mag } else { mag }
            };
            let ue8m0 = |b: u8| {
                if b == 0xFF {
                    f32::NAN
                } else {
                    2f32.powi(b as i32 - 127)
                }
            };
            bytes
                .iter()
                .enumerate()
                .map(|(i, &b)| e4m3(b) * ue8m0(sb[i / 32]))
                .collect()
        }
        other => panic!("{name}: no exact widen for {other:?}"),
    }
}

#[test]
fn qwen4exp_layer_planes_round_trip() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let c = Qwen4ExpConfig::read(&dir).expect("config");
    let st = ShardedSafetensors::open_dir(&dir).expect("shards");
    assert_eq!(c.blocks[0], Qwen4ExpBlock::Gdn);
    assert_eq!(c.blocks[3], Qwen4ExpBlock::Attention);

    for li in [0usize, 3] {
        let layer = load_layer(&exec, &st, &c, li).expect("layer loads");
        let p = format!("model.language_model.layers.{li}");

        // one representative dense plane per mixer kind: device bytes must be
        // identical to the checkpoint's, in whichever class the file ships -
        // bf16 (NVIDIA's export) or MXFP8, whose e4m3 payload and ue8m0
        // `weight_scale` both upload unconverted (the distilled MX export)
        let (name, plane) = match &layer.mixer {
            MixerW::Gdn(g) => (format!("{p}.linear_attn.in_proj_qkv.weight"), &g.qkv),
            MixerW::Attn(a) => (format!("{p}.self_attn.q_proj.weight"), &a.q),
        };
        let (_, want) = st.bytes(&name).expect("checkpoint bytes");
        if let Some(raw) = plane.raw_bf16() {
            let got: Vec<u8> = exec.to_host_range_u8(raw, 0, want.len()).expect("dtoh");
            assert!(got == want, "{name}: device bytes differ from checkpoint");
        } else if let Some((data, scale)) = plane.raw_mxf8() {
            let got: Vec<u8> = exec.to_host_range_u8(data, 0, want.len()).expect("dtoh");
            assert!(
                got == want,
                "{name}: device payload differs from checkpoint"
            );
            let scale_name = format!("{name}_scale");
            let (_, want_s) = st.bytes(&scale_name).expect("checkpoint scale bytes");
            let got_s: Vec<u8> = exec.to_host_range_u8(scale, 0, want_s.len()).expect("dtoh");
            assert!(
                got_s == want_s,
                "{scale_name}: device bytes differ from checkpoint"
            );
        } else {
            panic!(
                "{name}: class {} holds a re-encoding - no byte oracle applies",
                plane.class()
            );
        }

        // the LAUNCH FOLD is a byte concatenation and nothing more: the hc
        // plane's first `lowrank` rows must be the checkpoint's down plane and
        // its tail rows the inject plane, both unchanged. A fold that quietly
        // reordered or re-encoded either half would still produce plausible
        // logits, so it is gated here rather than inferred from a forward.
        let hcw = &layer.attn_hc;
        assert_eq!(
            hcw.inject_rows, c.hc_count,
            "attn hc did not fold its inject"
        );
        let hw = c.hc_width();
        let fused = hcw.down.raw_bf16().expect("folded plane is bf16");
        let (_, down_want) = st
            .bytes(&format!(
                "{p}.attn_hyper_connection.input_mix_weight_down.weight"
            ))
            .expect("down bytes");
        let (_, inj_want) = st
            .bytes(&format!(
                "{p}.attn_hyper_connection.block_inject_weight.weight"
            ))
            .expect("inject bytes");
        let head: Vec<u8> = exec
            .to_host_range_u8(fused, 0, down_want.len())
            .expect("dtoh");
        assert!(
            head == down_want,
            "folded hc plane's head is not the down plane"
        );
        let tail: Vec<u8> = exec
            .to_host_range_u8(fused, c.hc_lowrank * hw * 2, inj_want.len())
            .expect("dtoh");
        assert!(
            tail == inj_want,
            "folded hc plane's tail is not the inject plane"
        );

        // the GDN a||b fold, same claim
        if let MixerW::Gdn(g) = &layer.mixer {
            // exact in either class the file ships (bf16, or MXFP8 widened)
            let a_want = exact_f32(&st, &format!("{p}.linear_attn.in_proj_a.weight"));
            let b_want = exact_f32(&st, &format!("{p}.linear_attn.in_proj_b.weight"));
            let ab: Vec<f32> = exec.to_host(&g.ab.buf).expect("dtoh");
            let hv = c.gdn_v_heads * c.hidden;
            assert_eq!(ab.len(), 2 * hv, "a||b plane width");
            assert_eq!(ab[..hv], a_want[..], "a half of the fold");
            assert_eq!(ab[hv..], b_want[..], "b half of the fold");
        }

        // the router||shared-gate fold: last row is the shared expert's gate
        let router: Vec<f32> = exec.to_host(&layer.moe.router.buf).expect("dtoh");
        let sg_want = exact_f32(&st, &format!("{p}.mlp.shared_expert_gate.weight"));
        assert_eq!(
            router.len(),
            (c.n_expert + 1) * c.hidden,
            "router plane width"
        );
        assert_eq!(
            router[c.n_expert * c.hidden..],
            sg_want[..],
            "router plane's last row is not the shared-expert gate"
        );

        // hyper-connection norm: exact f32 widen
        let nb = exact_f32(&st, &format!("{p}.attn_hyper_connection.hc_norm.weight"));
        let got_n: Vec<f32> = exec.to_host(&layer.attn_hc.norm.buf).expect("dtoh");
        assert_eq!(got_n, nb, "{p} hc_norm widen not exact");

        // MoE gate plane: concatenated nibbles equal per-expert checkpoint
        // views at both ends of the expert range; scale2 array matches
        let paddock_engine::gpu_model::qwen4exp::ExpertSeats::Nvf4 { gate: plane, .. } =
            &layer.moe.seats
        else {
            panic!("{p}: the safetensors lane seats NVFP4 experts");
        };
        assert_eq!(
            (plane.n_expert, plane.ff, plane.in_dim),
            (c.n_expert, c.moe_ff, c.hidden)
        );
        let data: Vec<u8> = exec
            .to_host_range_u8(&plane.data, 0, c.n_expert * c.moe_ff * c.hidden / 2)
            .expect("dtoh");
        let s2: Vec<f32> = exec.to_host(&plane.scale2).expect("dtoh");
        let stride = c.moe_ff * c.hidden / 2;
        for e in [0usize, 255, c.n_expert - 1] {
            let v = nvfp4_view(&st, &format!("{p}.mlp.experts.{e}.gate_proj")).unwrap();
            assert!(
                data[e * stride..(e + 1) * stride] == *v.packed,
                "L{li} expert {e} gate nibbles differ"
            );
            assert_eq!(s2[e], v.scale2, "L{li} expert {e} scale2");
        }
    }
    eprintln!("qwen4exp layer oracle: layers 0 (GDN) + 3 (attn) byte-exact");
}

#[test]
fn qwen4exp_ple_projections_load() {
    let Some(dir) = common::model_dir("QWEN4EXP_DIR", &["Qwen3.8-Flash-Next-NVFP4"]) else {
        return;
    };
    let Some(exec) = common::gpu_arc() else {
        return;
    };
    let c = Qwen4ExpConfig::read(&dir).expect("config");
    let st = ShardedSafetensors::open_dir(&dir).expect("shards");
    // projections + hash buffers only - pin the audited I64 values; the
    // 51 GB table upload is exercised by the (heavy) full-load gate later.
    let ple = load_ple_projections(&exec, &st, &c, c.ple_layers[0]).expect("ple");
    assert_eq!(
        ple.multipliers,
        vec![23703573157769, 20109073645365, 8052911324071]
    );
    assert_eq!(ple.head_vocab.len(), 16);
    assert_eq!(ple.head_offset[0], 0);
    assert!(ple.table_scale.is_finite() && ple.table_scale > 0.0);
}
