//! Qwen3.5/3.6 weight load, residency planes, VRAM sizing.

use super::*;
use crate::gpu::{DeviceTensor, GpuError, GpuExecutor, KvDtype, QuantW, RepackedMxfp4};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::ggml_type::GgmlType;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;
use std::sync::Arc;

/// e4m3 (OCP FP8) byte -> f32: bias 7, 3-bit mantissa, denormals at e=0,
/// S.1111.111 is NaN (e4m3 has no infinities).
fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = (b & 7) as f32;
    if e == 0 {
        s * m * (-9f32).exp2()
    } else if e == 15 && (b & 7) == 7 {
        f32::NAN
    } else {
        s * (1.0 + m / 8.0) * ((e as i32 - 7) as f32).exp2()
    }
}

/// One routed-expert seat of block `i`: the k-quant plane repacked into
/// VRAM, or under `[moe_offload]` into device-mapped host memory
/// (gpu/host_plane.rs) for the slot cache to serve. `None` when the tensor
/// is not k-quant (the caller's Q8 path) or the pack lacks the k-quant MoE
/// pair (PADDOCK_NO_MOE_KQ=1 restores the old load error for A/B).
fn kq_expert_seat(
    exec: &GpuExecutor,
    map: &MappedGguf,
    i: usize,
    name: &str,
) -> Result<Option<ExpW>, GpuError> {
    if paddock_models::dev_var_os!("PADDOCK_NO_MOE_KQ").is_some() || !exec.has_kquant_moe() {
        return Ok(None);
    }
    let full = format!("blk.{i}.{name}");
    Ok(if crate::gpu::moe_offload().enabled {
        exec.try_repack_kquant_host_mapped(map, &full)?
            .map(ExpW::KqHost)
    } else {
        exec.try_repack_kquant(map, &full)?.map(ExpW::Kq)
    })
}

impl GpuQwen35 {
    /// Load a Qwen3.5/3.6 GGUF onto `exec`'s device. Reads the `qwen35.*`
    /// metadata, classifies each block as DeltaNet or full-attn from
    /// `full_attention_interval`, and uploads every tensor (Q8_0 kept resident,
    /// norms/A/dt as f32). MTP block (`nextn.*`) is ignored for base inference.
    /// `sm100` = this is a cc 10 die (B200 class). Two entries of the stack
    /// below were elected on sm_120/sm_86 and measured NEGATIVE here, so they
    /// are withheld on that die only -- the flags themselves stay, and an
    /// explicit `PADDOCK_<flag>=1` still forces them anywhere.
    fn apply_default_stack(sm100: bool) {
        // Withheld on cc 10, each with the measurement that says so
        // (qwen3.6-27b, uniform all-cold, one scenario per server):
        //
        //   PADDOCK_F8W8_TMA -- pd_f8_gemm_w8_o16 has no tcgen05 route and its
        //     TMA gate is `cma >= 9`, so on B200 the prefill always took the
        //     sm_90-era kernel: 1.58 ms/launch vs 0.049 for the tcgen05
        //     sibling. Cold TTFT 416.8 -> 40.7 ms, prefill span 383 -> 30.9.
        //
        //   PADDOCK_UNIFIED -- the fused prefill+decode tick. Its own note in
        //     service.rs records it regressing wide batch on some dies; on
        //     B200 it still does, and at c8 the mechanism is TTFT (roughly
        //     halved when it is off).
        //
        // Both remain DEFAULT-ON everywhere else; neither has been re-measured
        // on sm_89/9.0/12.x by this change.
        //
        // The two need different treatment, and getting it wrong cost a full
        // measurement round: PADDOCK_UNIFIED is read as "is the flag set?", so
        // withholding it is enough. PADDOCK_F8W8_TMA is not -- `f8lin_enabled`
        // tests only whether the KILL is absent, so merely declining to set the
        // flag leaves the tile-linear prefill lane on and the slow
        // pd_f8_gemm_lin_ktk route with it (measured: default TTFT back to
        // 432 ms and c1 down to 91.69). The kill must be set ACTIVELY.
        // PADDOCK_UNIFIED is not withheld: it is a concurrency TRADE on this
        // die, not a loss. Measured both ways, default build otherwise, it
        // wins single-stream decisively and loses the wider widths.
        // At c1 the fused tick is what puts the single prompt on the fast
        // unified prefill route; withholding it sends c1 down
        // prefill_slot_chunk and TTFT goes 40 -> 432 ms. Turning it off
        // process-wide therefore trades a ~29% single-stream loss for the
        // wide-batch wins.
        // The real fix is to choose per tick on queue depth, not per process
        // (filed) -- so nothing is withheld here.
        const SM100_WITHHOLD: &[&str] = &[];
        const SM100_SET_KILL: &[&str] = &["PADDOCK_NO_F8W8_TMA"];
        // (flag, kill switch) pairs -> "1"; value vars -> tuned default
        const FLAGS: &[(&str, &str)] = &[
            ("PADDOCK_QWEN35_MOE_FP4", "PADDOCK_NO_MOE_FP4"),
            ("PADDOCK_CHUNKED_PREFILL", "PADDOCK_NO_CHUNKED_PREFILL"),
            ("PADDOCK_QWEN35_CHUNK_BATCH", "PADDOCK_NO_CHUNK_BATCH"),
            ("PADDOCK_QWEN35_W8", "PADDOCK_NO_W8"),
            ("PADDOCK_F8W8_TMA", "PADDOCK_NO_F8W8_TMA"),
            ("PADDOCK_QWEN35_SPEC", "PADDOCK_NO_SPEC"),
            ("PADDOCK_ROUTER_GEMM", "PADDOCK_NO_ROUTER_GEMM"),
            ("PADDOCK_UNIFIED", "PADDOCK_NO_UNIFIED"),
            ("PADDOCK_ATTN_PF_V2", "PADDOCK_NO_ATTN_PF_V2"),
            ("PADDOCK_AB_F32", "PADDOCK_NO_AB_F32"),
            ("PADDOCK_MOE_PART_BF16", "PADDOCK_NO_MOE_PART_BF16"),
            // native-fp8 decode lane: DEFAULT on after the gate passed both
            // axes - PPL a wash, decode throughput up. Self-disables via the
            // guards:
            // sm_89+ pack capability, dense-Q8 tensors only, VRAM headroom
            // (falls back to the exact-Q8 chain when planes can't build).
            ("PADDOCK_F8_DECODE", "PADDOCK_NO_F8_DECODE"),
            // batched cold prefill (the wave-refill fix): the pipe's
            // wave-batched admissions call forward_prefill_batch with the whole
            // completion wave - the batched pass reads weights once for the
            // cohort. Its own gates keep tiny cohorts serial (>=2 items,
            // >=2048 total rows - the dc4-class regression guard).
            ("PADDOCK_QWEN35_BATCH_PREFILL", "PADDOCK_NO_BATCH_PREFILL"),
        ];
        const VALUES: &[(&str, &str)] = &[
            ("PADDOCK_QWEN35_MOE_FP4_MIN", "1"),
            ("PADDOCK_QMOE_SORTED_MIN", "1"),
            // span re-election, once the per-span fixed cost came down:
            // 2048 -> 1024. A ladder over 512/1024/1536/2048 on long prompts
            // at width has 1024 winning throughput AND cutting TTFT sharply;
            // every other cell is neutral (the cap never binds under
            // 1024-row prompts), and 512 still over-fragments. ITL is
            // span-invariant everywhere - what is left of the wide-batch ITL
            // gap is per-pass fixed cost, not span waits.
            ("PADDOCK_UNIFIED_PREFILL_ROWS", "1024"),
            // f16 DN recurrent state: halves the decode state band (192 MB
            // r+w/launch at b=32, the probed in-tick roof) and buys real
            // wide-batch throughput at held TTFT. Quality: +0.09% PPL
            // (prefix=1) / -0.07% (prefix=1024, chunked) against a +0.4-0.7
            // DN band - f16's 10 mantissa bits pass decisively where bf16's 7
            // failed at +1.47% (falsified twice; stays probe-only). Note the
            // class is not aggressive: vLLM's mamba_ssm_cache_dtype="auto"
            // ships bf16 states, a worse rounding class. The rs walk + VL
            // ride the ST template; =0 pins exact f32.
            ("PADDOCK_DN_STATE_F16", "1"),
            // tf32 chunked-scan: PREC=1 measures faster than 3xTF32, and
            // tf32's 10-bit mantissa is still FINER than the bf16 that fla
            // kernels run for the same products.
            ("PADDOCK_DNC_MMA", "1"),
            ("PADDOCK_DNC_MMA_G", "32"),
            // stage1 mma rework: 3xTF32 dots + hierarchical
            // explicit (I+M)^-1 + mma dw/du, 2 CTAs/SM - stage1 143.7->88.0us.
            // Gated: oracle band, PPL_PREFIX=512 19.770->19.731, greedy,
            // suite, full sweep. Explicit =0 disables (kill switch).
            ("PADDOCK_DNC_S1MMA", "1"),
            // stage1 PREC election: the walk's own
            // tf32 class extended to stage1's dot + dw/du chains (was
            // hardcoded 3xTF32). Gated: oracle band bounded (state 1.1e-2 ->
            // 2.7e-2 vs the sequential oracle at H=48), chunked pipeline
            // 740.5->701.1us (-5.3% - stage1 is serial-phase-bound, not
            // MMA-bound, so 3x->1x buys far less than the FLOP ratio), PPL
            // +0.45% = inside the accepted-class chaos band (scalar-stage1
            // control +0.37%, full-scalar control +0.73% on the same
            // corpus), suite 5/5, serve A/B neutral,
            // greedy: short exact vs llama.cpp, long chunked prompts fork
            // once with the same single-seam shape HEAD already has (HEAD
            // forks at 666, prec1 at 434 on the probe prompt). =3 reverts.
            ("PADDOCK_DNC_S1PREC", "1"),
            // register-state bf16-operand walk: the classic v2 walk is the
            // proven optimum of the f32/tf32 chunk FORMAT (8 schedules +
            // operand-bytes falsified), so what was left on the table was
            // INSTRUCTION ECONOMY - 6 cvt/mma + scalar frag loads where a
            // bf16 ldmatrix does the same work. walk_rs keeps f32 state
            // (bf16 state stays falsified) and moves the mma operands to
            // bf16: proto -61% kernel / -33.5% full stage1+walk route at the
            // 2048-row serve span. Gated: proto band (out rmsrel 2.4e-3,
            // the same operand class the reference kernels use),
            // PPL-distance, greedy fork-shape, suite, serve A/B. =0 kills
            // to the classic pair.
            ("PADDOCK_DNC_RS", "1"),
            // stage1 bf16-operand rebuild: the walk's operand-class
            // transformation applied to stage1 itself (ldmatrix + m16n8k16
            // bf16 dots/dw/du, f32 acc, T-build scalar f32 verbatim, q/k
            // staged once, 3 CTAs/SM). Kernel 192.5->113.2us at the 2048-row
            // span; serve TTFT down a few ms. Gated: proto band (dw rms
            // 1.2e-3 / coef 9.2e-3, the standard bf16 product class;
            // qb/kb/cg/gsh BIT-identical; vl==per-span BIT-exact), PPL
            // slightly better than the control, serve-side greedy, serve
            // A/B. =0 kills to stage1_v2.
            ("PADDOCK_DNC_S1RS", "1"),
            // skinny-out f32 K-split rung: the decay/ba plane GEMM refills
            // the wave via grid-z K-splits, exact-f32 FMA per window +
            // deterministic combine. Gated: PPL chaos-band probe (a
            // known-benign perturbation moves pf1024 +/-0.5-0.65%, and the ks
            // delta is in-band), greedy exact, spot bests at every width.
            // Engages only at batch >= 1024.
            ("PADDOCK_F32NT_KS", "1"),
            // ba-plane tf32 rung: warp-mma twin of the SIMT f32 kernel
            // (row-major cp.async tiles - the family's wall was the k-major
            // transpose scatter, 12 STS/thread/K-step), single tf32 ("p1",
            // the S1PREC class; its 10-bit inputs are 8x finer than a bf16
            // nvjet kernel on this exact plane), and ALLOCATION-FREE -
            // nt_ks's per-call cudaMallocAsync was the TTFT p90 straggler
            // class. Probe 260 -> 92 us at the wave; a paired serve A/B cuts
            // TTFT sharply with ITL flat and a tight p90. Gated: f64 probe
            // bands, serve
            // greedy (ctl-vs-ctl stable, single-seam forks), long-prefix
            // PPL, ABBA. "1" = 3xTF32 arm, =0 kills to nt_ks.
            ("PADDOCK_BA_TF32", "p1"),
            // QKC compact-bf16 q/k pair: on all-vl prefill ticks the conv
            // emits Hg-compact bf16 q/k and the vl chunked GDN entry reads
            // them - BIT-IDENTICAL values (the consumer rounded the same f32
            // itself before), 12x fewer q/k bytes. Proto gate word-exact on
            // all seven outputs; a paired serve A/B shaves a couple of ms off
            // TTFT and its p90, ITL flat. Engine
            // latch (dn_qkc) mirrors the rs-route envs, so killing any of
            // RS/S1MMA/S1RS also reverts this. =0 kills alone.
            ("PADDOCK_DNC_QKC", "1"),
        ];
        // set_env, not set_var: several of these gate PACK launcher arms via
        // C getenv, which on Windows never sees a bare set_var (the CRT-copy
        // trap - see envset.rs; a DNC_RS/S1MMA divergence 801'd every
        // >=128-row prefill span on the A6000)
        if sm100 {
            for kill in SM100_SET_KILL {
                // an explicit opt-in still wins: only elect the kill when the
                // operator has not asked for the flag by name
                let flag = kill.replacen("PADDOCK_NO_", "PADDOCK_", 1);
                if std::env::var_os(&flag).is_none() && std::env::var_os(kill).is_none() {
                    crate::envset::set_env(kill, "1");
                }
            }
        }
        for (var, kill) in FLAGS {
            if sm100 && SM100_WITHHOLD.contains(var) {
                continue;
            }
            if std::env::var_os(var).is_none() && std::env::var_os(kill).is_none() {
                crate::envset::set_env(var, "1");
            }
        }
        for (var, val) in VALUES {
            if std::env::var_os(var).is_none() {
                crate::envset::set_env(var, val);
            }
        }
        // tc5q admission at 192 tiles on sm_100. The pack's 256-tile
        // threshold encodes "~2 items per persistent CTA" but is a constant
        // where it should be a function of nsm: the 9B's gu plane (192 tiles)
        // missed the route and rode tc5p_m2 at 3.3 TB/s where tc5q clocks
        // 4.37 on this die. Measured across the full 9B ladder: single-stream
        // decode gains, every other width flat-to-up, coherence intact.
        // 27b unaffected (its sub-256 planes are all <=97 tiles,
        // where the persistent grid would idle a third of the die - do not
        // lower further without a new ladder). Explicit env always wins.
        if sm100 && std::env::var_os("PADDOCK_TC5Q_MINTILES").is_none() {
            crate::envset::set_env("PADDOCK_TC5Q_MINTILES", "192");
        }
        // tc5r@128 decode: with the f8t lane present, b in 65..=128 decode
        // graphs keep the f8t class (f8t_gemm routes 65..256 via tc5r - One
        // weight pass) instead of splitting into 2x64 halves. At b=128 the
        // whole-tick form is a large throughput and ITL win over the split.
        // Dies without the f8t lane keep 64: a
        // b>64 graph there records the 154us mma arm per layer (the
        // graph-arm law) - exactly what the split exists to prevent.
        // Explicit env always wins.
        if sm100 && std::env::var_os("PADDOCK_F8T_DEC_BMAX").is_none() {
            crate::envset::set_env("PADDOCK_F8T_DEC_BMAX", "128");
        }
    }

    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        Self::load_with(exec, map, max_ctx, None)
    }

    /// `load` plus explicit options the caller's config layer resolved -
    /// `fp8_native_dir` is an official-FP8 safetensors snapshot to source the
    /// f8 FFN planes from (the runner's `fp8_native` config field / env / flag;
    /// the engine itself never reads the environment for product config).
    pub fn load_with(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
        fp8_native_dir: Option<&std::path::Path>,
    ) -> Result<Self, GpuModelError> {
        // MoE expert offload (PADDOCK_MOE_HOST=1): the k-quant routed-expert
        // planes land in device-mapped host memory (gpu/host_plane.rs), so
        // they are not VRAM the gate should charge. The bytes it does not
        // charge are exactly the tensors the loader will mirror: k-quant
        // `*_exps` planes of the backbone; Q8 experts and the nextn/MTP
        // experts stay resident and stay charged.
        let moe_host = crate::gpu::moe_offload().enabled
            && paddock_models::dev_var_os!("PADDOCK_NO_MOE_KQ").is_none()
            && exec.has_kquant_moe();
        let host_bytes: u64 = if moe_host {
            map.tensor_infos()
                .filter(|t| {
                    t.name.starts_with("blk.")
                        && t.name.ends_with("_exps.weight")
                        && crate::gpu::kq_params(t.ggml_type).is_some()
                })
                .filter_map(|t| t.byte_size())
                .sum()
        } else {
            0
        };
        if host_bytes > 0 {
            tracing::info!(
                host_gib = host_bytes as f64 / (1u64 << 30) as f64,
                "qwen35 MoE expert offload: routed-expert planes will be host-mapped (PADDOCK_MOE_HOST)"
            );
        }
        // The slot cache is sized AFTER the KV plan (enable_batch), from
        // what the plan leaves - it is not charged here.
        exec.vram_load_gate(map.total_len().saturating_sub(host_bytes), "qwen3.5/3.6")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine: drop cudarc's cross-stream event tracking (pure
        // overhead here, and it blocks CUDA-graph capture). Must precede all allocs.
        exec.disable_event_tracking();
        // Small dies default the prefill-GEMM cp.async gate down to 128: the
        // sync 128x128 mmq underfills 84 SMs in the 65..1024-row band and the
        // pipe/hi rungs measured ~15% faster prefill there (A6000, 27b, 725
        // tokens: 721-760 -> 610-648 ms). Explicit env always wins; >=128-SM
        // dies keep the measured 1024 gate. Same Q8_0 numeric class either way.
        if exec.sm_count() < 128 && std::env::var_os("PADDOCK_MMQ_HI_MIN").is_none() {
            crate::envset::set_env("PADDOCK_MMQ_HI_MIN", "128");
        }
        // Small dies also warm the draft head for LONGER prompts: the 2048
        // cap ("+30% TTFT, decode too short to repay") is B200/gemma4
        // wide-batch data. A6000 single-user measures the opposite trade
        // (27b, 4.7k prompt): TTFT up ~15% but long-context decode more than
        // doubles - spec runs at depth, breakeven ~20 generated tokens.
        // But the warm rides the serial prefill lane, so its cost is not
        // "+15%" at depth: it forfeits the fast batched prefill entirely. On
        // a 200K prompt, warm=max_ctx is ~7x the end-to-end time of the same
        // run with the warm skipped - a TTFT cliff to speed a decode phase a
        // long-prompt request rarely has. 16384 keeps the whole measured-win
        // band (4.7K-class agentic prompts) with margin and caps the cliff at
        // a few seconds. Explicit env always wins.
        if exec.sm_count() < 128 && std::env::var_os("PADDOCK_QWEN35_SPEC_WARM_MAX").is_none() {
            crate::envset::set_env("PADDOCK_QWEN35_SPEC_WARM_MAX", "16384");
        }
        // Big dies take the 4096-row unified tick, re-A/B'd on the NVFP4
        // lane once krs + mimalloc + batched zero_slot + f8d head had shrunk
        // per-tick fixed costs again: over a 1024..8192 ladder at width,
        // throughput is no longer tick-size-flat and 4096 wins it while
        // keeping TTFT good. The older 2048 pick chose the TTFT optimum back
        // when throughput was flat; this supersedes it on the same
        // fixed-costs-changed reasoning that produced it. Short-prompt cells
        // are invariant (the cap never binds under 1024-row prompts).
        // Explicit env wins.
        if exec.sm_count() >= 128 && std::env::var_os("PADDOCK_UNIFIED_PREFILL_ROWS").is_none() {
            crate::envset::set_env("PADDOCK_UNIFIED_PREFILL_ROWS", "4096");
        }
        // The PROVEN qwen35 performance stack is the DEFAULT: every lever
        // here passed its gate (bit-exact, 2e-5 oracle, or PPL) and has
        // served with it. Installed as process env because the
        // pack launchers latch getenv - a paddock process serves one model, so
        // this scopes the defaults to qwen35 exactly (gpt-oss never runs this).
        // An explicit env value always wins (we only fill unset vars), and each
        // flag has a PADDOCK_NO_* kill switch. Deliberately not defaulted:
        // PADDOCK_DN_STATE_BF16 (real depth-compounding PPL drift - the
        // max-throughput mode stays an explicit opt-in; and on the A6000 it
        // is also slower, because the halved-state-bytes win is a big-die
        // DRAM-wall effect that does not hold at 84 SMs, so there is no
        // reason to opt in on small dies).
        // has_f8t_gemm() is the cc-10 marker: the pack NULLs f8t_gemm and
        // f8_repack_tiles off cc 10 (exact major+minor).
        Self::apply_default_stack(exec.has_f8t_gemm());

        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);

        // block_count includes any trailing nextn/MTP blocks (27B: 64 backbone + 1
        // MTP). The backbone loop must exclude them; the MTP block loads separately.
        let n_blocks_all = u("block_count")? as usize;
        let n_nextn = map
            .gguf()
            .arch_field("nextn_predict_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let n_layers = n_blocks_all - n_nextn;
        let embd = u("embedding_length")? as usize;
        let n_heads = u("attention.head_count")? as usize;
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        let head_dim = u("attention.key_length")? as usize;
        // Dense models carry feed_forward_length; the MoE variant (qwen35moe,
        // e.g. Qwen3.6-35B-A3B) carries the expert_* quartet instead. For MoE,
        // `ff` becomes the widest per-token FFN scratch row (expert ff and the
        // shared expert ff - 512/512 on the 35B).
        let moe = if map.gguf().arch_field("expert_count").is_some() {
            Some(MoeDims {
                n_expert: u("expert_count")? as usize,
                n_active: u("expert_used_count")? as usize,
                moe_ff: u("expert_feed_forward_length")? as usize,
                shexp_ff: u("expert_shared_feed_forward_length")? as usize,
            })
        } else {
            None
        };
        let ff = match &moe {
            Some(m) => m.moe_ff.max(m.shexp_ff),
            None => u("feed_forward_length")? as usize,
        };
        let rms_eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let full_attn_interval = u("full_attention_interval")? as usize;

        // DeltaNet geometry lives in the overloaded ssm.* namespace.
        let state_size = u("ssm.state_size")? as usize; // per-head k/v dim
        let n_k_heads = u("ssm.group_count")? as usize; // #key heads
        let n_v_heads = u("ssm.time_step_rank")? as usize; // #value heads (not a dt rank)
        let conv_k = u("ssm.conv_kernel")? as usize;
        let key_dim = state_size * n_k_heads;
        let value_dim = state_size * n_v_heads;
        let conv_dim = 2 * key_dim + value_dim;

        // Partial M-RoPE.
        let n_rot = u("rope.dimension_count")? as usize;
        let sections = read_sections(map)?;
        let rope_base = f("rope.freq_base").unwrap_or(1e7);
        let ctx_train = u("context_length").unwrap_or(max_ctx as u64) as usize;
        // ext_factor 0 => plain (no YaRN) rope; multimodal sectioning happens in
        // the mrope kernel from `sections`, not here.
        let yarn_params =
            YarnRope::new(n_rot, rope_base, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0).kernel_params();

        // Per-component VRAM ledger - snapshot free VRAM between load phases so the
        // startup log shows what each part of the model actually costs on the device
        // (the "show what's on GPU" product principle). Non-capturing closures, so no
        // borrow entanglement; deltas are GB-scale so desktop VRAM jitter is noise.
        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let gb = |used: u64| used as f64 / 1e9;
        let v_start = vfree();

        // token_embd stays RESIDENT in its file quant (input lookup only -
        // gathered rows are dequantized on the fly, exactly like llama). UD
        // k-quant files ship it Q4_K; plain exports Q8_0.
        let te_ty = map
            .tensor_info("token_embd.weight")
            .map(|t| t.ggml_type)
            .ok_or_else(|| GpuModelError::MissingMeta("token_embd.weight".into()))?;
        let (tok_embd, vocab) = if crate::gpu::kq_params(te_ty).is_some() {
            let t = exec.repack_kquant(map, "token_embd.weight")?;
            let vocab = t.dims[1];
            (TokEmbd::Kq(t), vocab)
        } else {
            let t = exec.upload_raw(map, "token_embd.weight")?;
            if t.ty != GgmlType::Q8_0 {
                return Err(GpuModelError::Unsupported(format!(
                    "token_embd.weight quant {:?} has no resident gather path",
                    t.ty
                )));
            }
            let vocab = t.dims[1];
            (TokEmbd::Q8(t), vocab)
        };
        let v_embd = vfree();
        // Report the exact resident byte size; the phase delta also absorbs
        // first-use CUDA context + kernel-module load, split out so neither is
        // misread.
        let embd_bytes = tok_embd.resident_bytes() as u64;
        tracing::info!(
            "qwen35 VRAM  input embeddings token_embd ({})   {:>7.2} GB",
            tok_embd.label(),
            gb(embd_bytes)
        );
        tracing::info!(
            "qwen35 VRAM  + first-use CUDA ctx / kernel modules {:>7.2} GB",
            gb(v_start.saturating_sub(v_embd).saturating_sub(embd_bytes))
        );

        // fp8-native ingestion (opt-in): an HF snapshot dir sources the f8 FFN
        // planes from the bf16 safetensors checkpoint - one quantization
        // (bf16 -> e4m3) instead of two (bf16 -> Q8_0 -> e4m3). Same f8w
        // format, so the whole landed f8 stack serves it. The dir arrives as
        // an explicit load option (config field `fp8_native`), never via env.
        //
        // A REQUESTED fp8-native dir that cannot be opened is an ERROR, not a
        // fallback. It used to warn and quietly serve Q8-derived planes, and
        // that silence cost real work: an HF snapshot dir holding only
        // tokenizer/config and no safetensors would serve the Q8-derived
        // class while every label said `fp8native`. A weight-class mismatch
        // that announces itself only in a WARN nobody greps is exactly the
        // silent failure the product principles forbid, and it turns any
        // like-for-like comparison into a coin flip. Ask for the class, get
        // the class or an error.
        let fp8_native = match fp8_native_dir {
            None => None,
            Some(d) => match paddock_models::safetensors::ShardedSafetensors::open_dir(d) {
                Ok(st) => {
                    tracing::info!(
                        "fp8-native ingestion: {} tensors from {}",
                        st.names().count(),
                        d.display()
                    );
                    Some(st)
                }
                Err(e) => {
                    return Err(GpuModelError::Unsupported(format!(
                        "fp8-native ingestion was requested but {} cannot be read \
                         ({e}) - refusing to serve Q8-derived planes under an \
                         fp8-native label; point --fp8-native at a checkpoint \
                         dir holding the .safetensors shards, or drop the flag \
                         to serve the GGUF class deliberately",
                        d.display()
                    )));
                }
            },
        };
        let st_bytes = |gguf_name: &str| -> Option<std::borrow::Cow<'_, [u8]>> {
            use paddock_models::safetensors::StDtype;
            let st = fp8_native.as_ref()?;
            let hf = paddock_models::safetensors::qwen35_hf_name(gguf_name)?;
            let (t, b) = st.bytes(&hf)?;
            match t.dtype {
                StDtype::Bf16 => Some(std::borrow::Cow::Borrowed(b)),
                // official-FP8 checkpoints (Qwen *-FP8): e4m3 weights with a
                // BF16 `weight_scale_inv` grid, one scale per 128x128 block.
                // Dequant to bf16 host-side (exact: e4m3 x bf16 is f32-
                // representable); the existing bf16 -> f8w converters then
                // REQUANTIZE to the per-32 e8m0 serving format. This is a
                // decode + re-encode, not a byte passthrough: each weight is
                // re-rounded once onto a per-32 pow2-scaled e4m3 grid. The
                // hop error is small next to the source's coarse block
                // scales (PPL 19.06/19.19 vs 18.77 for our own bf16-derived
                // planes - the official checkpoint measures worse either
                // way), but exact-value serving of these checkpoints would
                // need a bf16-block-scale GEMM fold this engine doesn't have.
                StDtype::F8E4m3 => {
                    if t.shape.len() != 2 {
                        return None;
                    }
                    let (rows, cols) = (t.shape[0], t.shape[1]);
                    // llm-compressor channel-strategy planes (the NVFP4
                    // export's fp8 islands): `<m>.weight_scale` BF16 [rows, 1]
                    // per-output-row. Dequant is the same e4m3 x bf16 walk as
                    // the block grid below with a degenerate 1-wide column
                    // grid, so reuse that path by synthesizing its geometry.
                    let (sb, scols) = if let Some((ts, sb)) = st.bytes(&format!("{hf}_scale_inv")) {
                        if ts.dtype != StDtype::Bf16 || ts.shape.len() != 2 {
                            return None;
                        }
                        let (srows, scols) = (ts.shape[0], ts.shape[1]);
                        if srows != rows.div_ceil(128) || scols != cols.div_ceil(128) {
                            return None;
                        }
                        (sb, scols)
                    } else {
                        let base = hf.strip_suffix(".weight")?;
                        let (ts, sb) = st.bytes(&format!("{base}.weight_scale"))?;
                        if ts.dtype != StDtype::Bf16 || ts.shape.iter().product::<usize>() != rows {
                            return None;
                        }
                        (sb, 0usize) // scols=0 marks the channel layout
                    };
                    let mut out = vec![0u8; rows * cols * 2];
                    let nthreads = std::thread::available_parallelism()
                        .map(|n| n.get().min(16))
                        .unwrap_or(8);
                    let band = rows.div_ceil(nthreads);
                    std::thread::scope(|sc| {
                        for (ti, chunk) in out.chunks_mut(band * cols * 2).enumerate() {
                            let r0 = ti * band;
                            sc.spawn(move || {
                                for (rr, orow) in chunk.chunks_mut(cols * 2).enumerate() {
                                    let r = r0 + rr;
                                    let wrow = &b[r * cols..(r + 1) * cols];
                                    // scols=0: channel layout, one scale per row
                                    let srow = if scols == 0 {
                                        &sb[r * 2..]
                                    } else {
                                        &sb[(r / 128) * scols * 2..]
                                    };
                                    for c in 0..cols {
                                        let si = if scols == 0 { 0 } else { (c / 128) * 2 };
                                        let sc16 = u16::from_le_bytes([srow[si], srow[si + 1]]);
                                        let scale = f32::from_bits((sc16 as u32) << 16);
                                        let v = e4m3_to_f32(wrow[c]) * scale;
                                        let bits = v.to_bits();
                                        // f32 -> bf16 round-to-nearest-even
                                        let bf =
                                            ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16;
                                        orow[c * 2..c * 2 + 2].copy_from_slice(&bf.to_le_bytes());
                                    }
                                }
                            });
                        }
                    });
                    Some(std::borrow::Cow::Owned(out))
                }
                _ => None,
            }
        };
        // Raw fp8-channel plane out of the same snapshot: the e4m3 bytes as
        // STORED plus the per-output-row `weight_scale` (BF16 [rows, 1]) as
        // f32 - the f8row class's own layout, no dequant, no requant. None
        // for bf16 tensors, block-scale (`_scale_inv` grid) checkpoints, and
        // anything whose scale vector does not match its row count.
        let st_f8row = |gguf_name: &str| -> Option<(&[u8], Vec<f32>, usize, usize)> {
            use paddock_models::safetensors::StDtype;
            let st = fp8_native.as_ref()?;
            let hf = paddock_models::safetensors::qwen35_hf_name(gguf_name)?;
            let (t, b) = st.bytes(&hf)?;
            if t.dtype != StDtype::F8E4m3 || t.shape.len() != 2 {
                return None;
            }
            let (rows, cols) = (t.shape[0], t.shape[1]);
            if b.len() != rows * cols {
                return None;
            }
            let base = hf.strip_suffix(".weight")?;
            let (ts, sb) = st.bytes(&format!("{base}.weight_scale"))?;
            if ts.dtype != StDtype::Bf16
                || ts.shape.iter().product::<usize>() != rows
                || sb.len() != rows * 2
            {
                return None;
            }
            let scales: Vec<f32> = sb
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect();
            Some((b, scales, rows, cols))
        };
        let mut layers = Vec::with_capacity(n_layers);
        // FFN REPLACE precondition, hoisted above the backbone loop so the
        // Q8_0 source can be dropped in the same iteration that builds its
        // e4m3 twin rather than in a pass afterwards (see `replace_q8` - a
        // batch pass leaves every hole pinned between live twins and the
        // driver never gets the memory back).
        //
        // `kq_resident` is only finalized after the loop, at its natural site,
        // so this uses `kq_early`: a CONSERVATIVE pre-image built from file
        // facts alone. Every input `kq_resident` reads is decided by the
        // checkpoint's tensor types - k-quant seats anywhere, a k-quant token
        // embedding, or non-Q8_0 ssm_alpha/ssm_beta (which force `alpha_w`
        // None and with it the serial spine). It can only be more pessimistic
        // than the real value, never less; the loop's verdict is checked
        // against the real `kq_resident` after the loop and a disagreement is
        // a hard load error, not a debug_assert (release strips those, and
        // this seam has already shipped one silent corruption that way).
        let kq_early = matches!(&tok_embd, TokEmbd::Kq(_))
            || map
                .tensor_infos()
                .any(|t| crate::gpu::kq_params(t.ggml_type).is_some())
            || map.tensor_infos().any(|t| {
                (t.name.ends_with("ssm_alpha.weight") || t.name.ends_with("ssm_beta.weight"))
                    && t.ggml_type != GgmlType::Q8_0
            });
        let ffn_replace = !kq_early
            && exec.has_f8lin_gemv()
            && paddock_models::dev_var_os!("PADDOCK_NO_LIN_GEMV").is_none()
            && f8_ffn_min() <= 2
            && f8_ffn_pf_min() == 0;
        let (mut ffn_freed, mut ffn_replaced) = (0u64, 0usize);
        // Bytes the REPLACE never allocated in the first place (the preferred
        // half - see `stub_plane`), as opposed to `ffn_freed`, which is bytes
        // that were allocated and then dropped.
        let mut ffn_skipped_bytes = 0u64;
        // Same shape for the dense PROJECTIONS: the e4m3 twin is built per
        // layer inside the loop and the Q8_0 source dropped in the same
        // iteration. The whole-model pass this replaced left the mixer sources
        // live across the entire backbone load, interleaved with the f8-ffn
        // twins, and freeing them afterwards cost 4.7 GB of pool holes.
        let w8_lane =
            std::env::var_os("PADDOCK_QWEN35_W8").is_some() && exec.has_f8_gemm_w8() && !kq_early;
        let proj_replace = w8_lane && w8_min_batch() == 0 && f8_dec_min() <= 1;
        // ...and the projections can go one better than dropping the source:
        // skip the upload entirely, exactly as the FFN half does, by building
        // the e4m3 twins from file transients. Two lanes still read the
        // RESIDENT Q8_0 mixer planes and veto that:
        //   f8t mixer projections (`parts_of` walks the layer's QuantW seats)
        //   an fp8_native/NVFP4 snapshot (bf16 bytes via cat_native)
        // FUSE_DN does not veto - it reads the file itself (repack_q8_concat2).
        let proj_skip_upload = proj_replace && fp8_native.is_none() && !f8t_attn_enabled(&exec);
        let mut bs_w8: Vec<LayerW8> = Vec::with_capacity(n_layers);
        let (mut proj_freed, mut proj_replaced) = (0u64, 0usize);
        let mut proj_skipped_bytes = 0u64;
        // fused gate|up planes, indexed by layer (None for MoE / non-Q8 / VRAM-tight)
        let mut bs_gu_planes: Vec<Option<RepackedQ8>> = Vec::with_capacity(n_layers);
        // fused DN in_qkv|gate_w planes, indexed by layer (None for full-attn)
        let mut bs_dn_planes: Vec<Option<RepackedQ8>> = Vec::with_capacity(n_layers);
        // f8 FFN planes (native-fp8 decode lane), indexed by layer
        let mut bs_f8ffn_planes: Vec<Option<[(RepackedMxfp4, usize, usize); 2]>> =
            Vec::with_capacity(n_layers);
        // byte-passthrough decode planes (PADDOCK_FP8_BS debug knob + an
        // fp8_native official-FP8 snapshot): raw e4m3 bytes + f32 block scales
        let fp8_bs = paddock_models::dev_var_os!("PADDOCK_FP8_BS").is_some();
        let mut bs_f8ffn_bs_planes: Vec<Option<[(RepackedMxfp4, usize, usize); 2]>> =
            Vec::with_capacity(n_layers);
        let mut bs_f8t_ffn_planes: Vec<Option<[crate::gpu::F8TilePlane; 2]>> =
            Vec::with_capacity(n_layers);
        let mut bs_f8row_ffn_planes: Vec<Option<F8RowFfn>> = Vec::with_capacity(n_layers);
        let mut bs_f8t_attn_planes: Vec<Option<[crate::gpu::F8TilePlane; 2]>> =
            Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            // Matmul weights load quantized-resident with per-TENSOR dispatch:
            // Q8_0 repacks into the aligned data + f16-scale streams the
            // vectorized decode GEMV reads; the k-quant family (UD/XL files)
            // repacks into its own streams. Memory-neutral at rest (the
            // interleaved upload is freed either way).
            let qt = |name: &str| exec.load_quantw(map, &format!("blk.{i}.{name}"));
            // Q8_0-only seats (MoE experts - the stage-3 k-quant arm).
            let qt8 = |name: &str| exec.repack_q8(map, &format!("blk.{i}.{name}"));

            let is_full = (i + 1) % full_attn_interval == 0;
            // Projection e4m3 twins, built before the Q8_0 seats exist - same
            // move as the FFN half below, and for the same reason: a plane
            // that is never allocated leaves no hole for the mempool to
            // strand. The conversion runs on per-layer file transients through
            // the same core the resident path uses (`build_w8_from`), each
            // transient dropped as soon as its plane exists.
            //
            // None here (k-quant seats, a missing tensor, or a lane that vetoed
            // skipping) simply falls back to uploading the resident planes and
            // replacing them afterwards - the pre-existing behaviour.
            let w8_early = proj_skip_upload
                .then(|| build_w8_from_file(&exec, map, i, is_full))
                .flatten();
            // Observed on the planes that were actually built, never predicted.
            let proj_covered = w8_early.is_some();
            let seatp = |n: &str| -> Result<QuantW, GpuModelError> {
                if !proj_covered {
                    return Ok(qt(n)?);
                }
                let t = map
                    .tensor_info(&format!("blk.{i}.{n}"))
                    .ok_or_else(|| GpuModelError::MissingMeta(format!("blk.{i}.{n}")))?;
                let dims: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
                super::stub_plane(&exec, dims)
            };
            // `mut`: where the twins were not built from the file, the Q8_0
            // sources are uploaded and dropped in place further down.
            let mut mixer = if is_full {
                Mixer::Full(FullAttnWeights {
                    wq: seatp("attn_q.weight")?,
                    wk: seatp("attn_k.weight")?,
                    wv: seatp("attn_v.weight")?,
                    q_norm: dt("attn_q_norm.weight")?,
                    k_norm: dt("attn_k_norm.weight")?,
                    wo: seatp("attn_output.weight")?,
                })
            } else {
                // alpha/beta: the fused decode kernel + x2 prefill pair are
                // Q8_0-class. Non-Q8 exports (UD ships F16 here) skip the
                // repacked pair and make the ab_f32 plane MANDATORY - tiny
                // ([embd, 2*n_v_heads] f32) and exact (host widen / exact
                // dequant), consumed by matvec_f32 + delta_gate_ab.
                let ab_q8 = map
                    .tensor_info(&format!("blk.{i}.ssm_alpha.weight"))
                    .is_some_and(|t| t.ggml_type == GgmlType::Q8_0)
                    && map
                        .tensor_info(&format!("blk.{i}.ssm_beta.weight"))
                        .is_some_and(|t| t.ggml_type == GgmlType::Q8_0);
                let want_ab_plane = !ab_q8
                    || (std::env::var_os("PADDOCK_AB_F32").is_some() && exec.has_delta_gate_ab());
                if !ab_q8 && !exec.has_delta_gate_ab() {
                    return Err(GpuModelError::Unsupported(
                        "non-Q8_0 ssm_alpha/ssm_beta need the delta_gate_ab kernel (older pack)"
                            .into(),
                    ));
                }
                Mixer::Linear(DeltaNetWeights {
                    in_qkv: seatp("attn_qkv.weight")?,
                    conv_w: dt("ssm_conv1d.weight")?,
                    alpha_w: if ab_q8 {
                        Some(qt8("ssm_alpha.weight")?)
                    } else {
                        None
                    },
                    beta_w: if ab_q8 {
                        Some(qt8("ssm_beta.weight")?)
                    } else {
                        None
                    },
                    ab_f32: if want_ab_plane {
                        let a = dt("ssm_alpha.weight")?;
                        let b = dt("ssm_beta.weight")?;
                        let na = a.buf.len();
                        let mut buf = exec.alloc(na + b.buf.len())?;
                        exec.copy_region(&a.buf, 0, &mut buf, 0, na)?;
                        exec.copy_region(&b.buf, 0, &mut buf, na, b.buf.len())?;
                        Some(DeviceTensor {
                            buf,
                            dims: vec![a.dims[0], 2 * a.dims[1]],
                        })
                    } else {
                        None
                    },
                    ssm_a: dt("ssm_a")?,
                    dt_bias: dt("ssm_dt.bias")?,
                    ssm_norm: dt("ssm_norm.weight")?,
                    gate_w: seatp("attn_gate.weight")?,
                    out_w: seatp("ssm_out.weight")?,
                })
            };

            // FFN e4m3 twin, built before the Q8_0 planes exist.
            //
            // The Q8-derived arm sources its bytes from `map` directly
            // (repack_q8_concat2 / repack_q8), so the RESIDENT Q8_0 planes were
            // never an input to it. That means a REPLACE-elected FFN can skip
            // the upload entirely rather than allocate 18 GB and free it again
            // - and a plane that is never allocated leaves no hole for the
            // mempool to strand. That is the whole difference between the
            // 0.71 GB an allocation-only load fragments and the 5.9 GB an
            // allocate-then-free load pays; freeing in-loop instead of in a
            // batch only moves the holes around (measured 5.91 / 4.74 / 5.84
            // for batch / half-in-loop / both-in-loop).
            // See.
            //
            // Plain-GGUF lanes only. With an fp8_native/NVFP4 snapshot attached
            // the FFN may take the checkpoint-exact Nvf4Dense arm or the
            // native-bf16 f8w source, and both are decided from `ffn` itself
            // further down - those keep the original ordering untouched.
            let f8ffn_early = if fp8_native.is_none()
                && moe.is_none()
                && std::env::var_os("PADDOCK_F8_DECODE").is_some()
                && exec.has_f8d_gemm_mma_ks()
                && exec.vram_headroom().is_some_and(|h| h > 24 << 30)
            {
                (|| -> Option<[(RepackedMxfp4, usize, usize); 2]> {
                    let q8_dims = |n: &str| -> Option<(usize, usize)> {
                        let t = map.tensor_info(&format!("blk.{i}.{n}"))?;
                        (t.ggml_type == GgmlType::Q8_0)
                            .then(|| (t.dims[0] as usize, t.dims[1] as usize))
                    };
                    let (gi, go) = q8_dims("ffn_gate.weight")?;
                    let (ui, uo) = q8_dims("ffn_up.weight")?;
                    let (di, dout) = q8_dims("ffn_down.weight")?;
                    if (ui, uo) != (gi, go) {
                        return None;
                    }
                    // FUSED gate|up (memory-neutral: same bytes as the two
                    // separate planes). Decode runs it as one 2ff GEMM;
                    // prefill row-slices at 0/ff, byte-identical to separate.
                    // Each Q8 staging pair is dropped before the next is
                    // taken, so the transient is one plane deep, not a model.
                    let gu_q8 = exec
                        .repack_q8_concat2(
                            map,
                            &format!("blk.{i}.ffn_gate.weight"),
                            &format!("blk.{i}.ffn_up.weight"),
                        )
                        .ok()?;
                    let gu = exec.q8_0_to_f8w(&gu_q8).ok()?;
                    drop(gu_q8);
                    let dn_q8 = exec
                        .repack_q8(map, &format!("blk.{i}.ffn_down.weight"))
                        .ok()?;
                    let dn = exec.q8_0_to_f8w(&dn_q8).ok()?;
                    drop(dn_q8);
                    // tile-linear conversion happens here rather than at the
                    // shared site below, because `is_lin()` is what decides
                    // whether the Q8_0 source may be skipped - the test has to
                    // run on the finished plane, not on a prediction of it.
                    let cv = |t: (RepackedMxfp4, usize, usize)| -> Option<(RepackedMxfp4, usize, usize)> {
                        if f8lin_enabled(&exec) && t.1.is_multiple_of(128) && t.2.is_multiple_of(16) {
                            let (w, i, o) = t;
                            Some((exec.f8w_repack_lin(w, i, o).ok()?, i, o))
                        } else {
                            Some(t)
                        }
                    };
                    Some([cv((gu, gi, 2 * go))?, cv((dn, di, dout))?])
                })()
            } else {
                None
            };
            let ffn_lin_done = f8ffn_early.is_some();
            // Observed on the plane that was actually built, never predicted:
            // the twin exists AND carries the layout the b=1 GEMV reads.
            let ffn_covered = ffn_replace
                && f8ffn_early
                    .as_ref()
                    .is_some_and(|p| p[0].0.is_lin() && p[1].0.is_lin());
            // `mut`: where the twin did not cover, the Q8_0 sources are still
            // uploaded and get dropped in place further down this iteration.
            let mut ffn = if moe.is_some() {
                // k-quant expert seats (the stage-3 arm): the file's
                // Q4_K/Q6_K expert tensors stay k-quant-resident (~0.55x the
                // Q8 expert bytes). This is the only load path for k-quant
                // MoE files - repack_q8 refuses non-Q8_0 tensors, so before
                // this arm UD MoE files failed at blk.0.ffn_gate_exps -
                // hence auto-on when the pack has the pair
                // (PADDOCK_NO_MOE_KQ=1 restores the old load error for A/B).
                // Serving classes: token-batched pair at decode, the sorted
                // kq mma pair (single-dtype; UD files pair gate/up types in
                // practice) past the sorted boundary - moe_ffn dispatches
                // per seat. gate and up must agree in residency (one fused
                // kernel call); down is independent. Q8_0 tensors keep qt8 +
                // the Q8 sorted family.
                // Under [moe_offload] the k-quant seats land in host-mapped
                // memory instead (gpu/host_plane.rs); Q8 seats stay resident.
                let kq_exp = |name: &str| kq_expert_seat(&exec, map, i, name);
                let (gate_exps, up_exps) = match (
                    kq_exp("ffn_gate_exps.weight")?,
                    kq_exp("ffn_up_exps.weight")?,
                ) {
                    (Some(g), Some(u)) => (g, u),
                    _ => (
                        ExpW::Q8(qt8("ffn_gate_exps.weight")?),
                        ExpW::Q8(qt8("ffn_up_exps.weight")?),
                    ),
                };
                let down_exps = match kq_exp("ffn_down_exps.weight")? {
                    Some(d) => d,
                    None => ExpW::Q8(qt8("ffn_down_exps.weight")?),
                };
                Ffn::Moe(MoeFfnWeights {
                    router_w: dt("ffn_gate_inp.weight")?,
                    gate_exps,
                    up_exps,
                    down_exps,
                    shexp_gate_inp: dt("ffn_gate_inp_shexp.weight")?,
                    shexp_gate: qt("ffn_gate_shexp.weight")?,
                    shexp_up: qt("ffn_up_shexp.weight")?,
                    shexp_down: qt("ffn_down_shexp.weight")?,
                    gate_exps_fp4: None,
                    up_exps_fp4: None,
                    down_exps_fp4: None,
                    moe_zero_bias: None,
                    cache: None,
                })
            } else if let Some(nv) = (|| -> Option<Ffn> {
                // Checkpoint-exact NVFP4 dense FFN (the qwen3.8
                // lane): when the fp8_native snapshot is an llm-compressor
                // NVFP4 export, its MLP planes ship as weight_packed triples.
                // Upload them byte-for-byte (W4A16 serving, zero requant) and
                // skip the entire Q8-derived FFN stack for this layer -
                // planes and aux lanes both - which is also what makes the
                // lane fit (~150 MB of fp4 replaces ~284 MB of Q8_0/layer).
                // Layers the recipe kept at fp8 (channel scales, e.g. the
                // last 8) have no weight_packed tensor and fall through to
                // the Dense path, whose f8w lane sources them via st_bytes.
                let st = fp8_native.as_ref()?;
                if !exec.has_nvf4_ckpt()
                    || paddock_models::dev_var_os!("PADDOCK_NO_NVF4_FFN").is_some()
                {
                    // Say it, once, and say WHY. Falling back to the Q8-derived
                    // planes is correct off sm_120a, but it is also a silent
                    // downgrade of a build the user picked and downloaded ~22 GB
                    // for: same answers as plain Q8_0, same memory as plain
                    // Q8_0, no sign anything was ignored. The Studio greys the
                    // choice out ahead of time; this is the backstop for a
                    // hand-written config file that asks anyway.
                    if i == 0 && !exec.nvf4_ckpt_arch() {
                        let (maj, min) = exec.compute_capability();
                        tracing::warn!(
                            "NVFP4 checkpoint planes ignored: this GPU is sm_{maj}{min} and the \
                             W4A16 nvf4 kernels need sm_89 or newer (the e2m1 nibble decode \
                             converts through e4m3). Serving the Q8_0 base instead - same \
                             answers and same memory as picking Full quality."
                        );
                    }
                    return None;
                }
                let plane = |gguf: &str| -> Option<crate::gpu::Nvf4Plane> {
                    let hf = paddock_models::safetensors::qwen35_hf_name(gguf)?;
                    let prefix = hf.strip_suffix(".weight")?.to_string();
                    let v = paddock_models::modelopt::nvfp4_view(st, &prefix).ok()?;
                    exec.nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
                        .ok()
                };
                let gate = plane(&format!("blk.{i}.ffn_gate.weight"))?;
                let up = plane(&format!("blk.{i}.ffn_up.weight"))?;
                let down = plane(&format!("blk.{i}.ffn_down.weight"))?;
                if i == 0 {
                    tracing::info!(
                        "qwen35 nvf4 FFN lane: checkpoint-exact W4A16 planes (first layer {i})"
                    );
                }
                Some(Ffn::Nvf4Dense { gate, up, down })
            })() {
                bs_gu_planes.push(None);
                nv
            } else {
                // fusion program: a merged gate|up plane serves the b>=8
                // decode band as one die-filling ks GEMM (544 tiles -> nz=1:
                // no partial planes, no combine) - the same K-split fusion
                // economics as vLLM's merged gate_up_proj, worth ~4.5 ms per
                // step. Q8-only; duplicates the two planes for now
                // (the per-tensor originals still serve prefill/serial), so
                // it's gated on generous free VRAM. PADDOCK_NO_FUSE_GU kills.
                // (concat2 validates Q8_0 itself - non-Q8 tensors fall to None)
                // OPT-IN for now: the planes DUPLICATE gate/up (~12 GB on the
                // 27B) and the measured decode win after the shape-aware nz
                // retune is only ~1% - the duplication isn't worth defaulting
                // until the memory-neutral
                // conversion (drop per-tensor planes, all paths fused) lands.
                let gu = if paddock_models::dev_var_os!("PADDOCK_FUSE_GU").is_some()
                    && exec.has_swiglu_fused()
                    // headroom, not raw free: duplicated planes must fit the
                    // configured vram_budget, not another runner's grant
                    && exec.vram_headroom().is_some_and(|h| h > 46 << 30)
                {
                    exec.repack_q8_concat2(
                        map,
                        &format!("blk.{i}.ffn_gate.weight"),
                        &format!("blk.{i}.ffn_up.weight"),
                    )
                    .ok()
                } else {
                    None
                };
                bs_gu_planes.push(gu);
                // The REPLACE, expressed as an upload that never happens. The
                // stub carries the true dims, so every `.dims()` consumer and
                // `stub_guard`'s "this site has no e4m3 arm" refusal still work
                // exactly as they did when the plane was allocated and freed.
                let mut seat = |n: &str| -> Result<QuantW, GpuModelError> {
                    if !ffn_covered {
                        return Ok(qt(n)?);
                    }
                    let t = map
                        .tensor_info(&format!("blk.{i}.{n}"))
                        .ok_or_else(|| GpuModelError::MissingMeta(format!("blk.{i}.{n}")))?;
                    let dims: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
                    ffn_skipped_bytes += t.byte_size().unwrap_or(0);
                    ffn_replaced += 1;
                    super::stub_plane(&exec, dims)
                };
                Ffn::Dense {
                    gate: seat("ffn_gate.weight")?,
                    up: seat("ffn_up.weight")?,
                    down: seat("ffn_down.weight")?,
                }
            };
            // DN in_proj fusion: merged in_qkv|gate_w plane -
            // vLLM's exact 16384-out DN merge (their grid (1,256,1)). One
            // 256-tile ks GEMM per linear layer at decode widths instead of
            // two + combines. OPT-IN like FUSE_GU (duplicates ~4.3 GB on the
            // 27B until the memory-neutral conversion).
            let dn = if matches!(&mixer, Mixer::Linear(_))
                && paddock_models::dev_var_os!("PADDOCK_FUSE_DN").is_some()
                && exec.has_row_slice()
                && exec.vram_headroom().is_some_and(|h| h > 42 << 30)
            {
                exec.repack_q8_concat2(
                    map,
                    &format!("blk.{i}.attn_qkv.weight"),
                    &format!("blk.{i}.attn_gate.weight"),
                )
                .ok()
            } else {
                None
            };
            bs_dn_planes.push(dn);
            // native-fp8 decode lane: e4m3 FFN planes converted from the Q8
            // originals (the 70%-of-bytes mats LayerW8 never covered). OPT-IN
            // (PADDOCK_F8_DECODE): e4m3 W8A8 is a labeled precision class AND
            // the planes duplicate ~17 GB on the 27B until fp8-native
            // checkpoints load directly. VRAM-guarded per layer.
            // Why-not diagnostics (layer 0 only). The f8 FFN decode lane is a
            // four-condition opt-in and a silent None here is indistinguishable
            // from "built" in the VRAM audit - on the B200 bring-up box the FFN
            // then serves Q8_0 int8, which is the one class this die de-rates
            // (1148 TOPS vs ~7.5P e4m3), so a quietly-unbuilt plane is a large
            // and invisible perf cliff. Say which condition failed.
            if i == 0 {
                let env = std::env::var_os("PADDOCK_F8_DECODE").is_some();
                let kern = exec.has_f8d_gemm_mma_ks();
                let head = exec.vram_headroom();
                tracing::info!(
                    env_set = env,
                    has_f8d_kernel = kern,
                    vram_headroom_gb = head.map(|h| h >> 30),
                    headroom_ok = head.is_some_and(|h| h > 24 << 30),
                    ffn_kind = match &ffn {
                        Ffn::Dense { .. } => "dense",
                        Ffn::Nvf4Dense { .. } => "nvf4-dense",
                        Ffn::Moe(_) => "moe",
                    },
                    "qwen35 f8-ffn decode lane: gate check"
                );
            }
            // Checkpoint-exact fp8 dense FFN (the f8row class, see F8RowFfn):
            // a Dense layer whose three MLP tensors the snapshot stores as
            // fp8-channel (llm-compressor "strategy: channel" - the NVFP4
            // export's islands, layers 56-63 on Qwen3.8-27B) is served from
            // the file's own e4m3 bytes + per-row f32 scales. It replaces
            // the lin twin for that layer entirely (same 1 B/param resident,
            // no second copy), and the Q8_0 seats are dropped below on the
            // same coverage argument the lin twin used: every width has an
            // arm. Requires the Q8-only file shape (`!kq_early`): the serial
            // spine's k-quant readers are the one path this lane never arms.
            let f8row_ffn: Option<F8RowFfn> = if f8row_ffn_enabled(&exec)
                && !kq_early
                && f8ffn_early.is_none()
                && fp8_native.is_some()
            {
                match &ffn {
                    Ffn::Dense { gate, up, down } => (|| -> Option<F8RowFfn> {
                        let (gb, gs, gr, gc) = st_f8row(&format!("blk.{i}.ffn_gate.weight"))?;
                        let (ub, us, ur, uc) = st_f8row(&format!("blk.{i}.ffn_up.weight"))?;
                        let (db, ds, dr, dc) = st_f8row(&format!("blk.{i}.ffn_down.weight"))?;
                        // [ff, embd] x2 and [embd, ff]; the seats' own dims
                        // ([in, out]) are the reference every consumer reads
                        let (gd, ud, dd) = (gate.dims(), up.dims(), down.dims());
                        if (gr, gc) != (gd[1], gd[0])
                            || (ur, uc) != (ud[1], ud[0])
                            || (dr, dc) != (dd[1], dd[0])
                            || (ur, uc) != (gr, gc)
                            || (dr, dc) != (gc, gr)
                            || gc % 32 != 0
                            || gr % 32 != 0
                        {
                            tracing::warn!(
                                layer = i, gate = ?(gr, gc), up = ?(ur, uc), down = ?(dr, dc),
                                "qwen35 f8row FFN: checkpoint plane geometry does not match the seats; lin twin instead"
                            );
                            return None;
                        }
                        let gate = exec.fp8_ckpt_to_f8row_rows(gb, &gs, gc, gr).ok()?;
                        let up = exec.fp8_ckpt_to_f8row_rows(ub, &us, uc, ur).ok()?;
                        let down = exec.fp8_ckpt_to_f8row_rows(db, &ds, dc, dr).ok()?;
                        Some(F8RowFfn {
                            gate,
                            up,
                            down,
                            embd: gc,
                            ff: gr,
                        })
                    })(),
                    _ => None,
                }
            } else {
                None
            };
            let f8ffn = if f8ffn_early.is_some() {
                // built above, before the Q8_0 seats - already lin-converted
                f8ffn_early
            } else if f8row_ffn.is_none()
                && std::env::var_os("PADDOCK_F8_DECODE").is_some()
                && exec.has_f8d_gemm_mma_ks()
                && exec.vram_headroom().is_some_and(|h| h > 24 << 30)
            {
                match &ffn {
                    Ffn::Dense { gate, up, down } => (|| -> Option<_> {
                        if i == 0 {
                            tracing::info!(
                                gate_is_kq = gate.kq().is_some(),
                                up_is_kq = up.kq().is_some(),
                                down_is_kq = down.kq().is_some(),
                                st_gate = st_bytes(&format!("blk.{i}.ffn_gate.weight")).is_some(),
                                st_up = st_bytes(&format!("blk.{i}.ffn_up.weight")).is_some(),
                                st_down = st_bytes(&format!("blk.{i}.ffn_down.weight")).is_some(),
                                "qwen35 f8-ffn decode lane: source check"
                            );
                        }
                        // FUSED gate|up f8 plane (memory-neutral: same bytes
                        // as the two separate planes; the +2 GB Q8 concat is
                        // a load-time transient). Decode runs it as one 2ff
                        // GEMM (the b<=16 launch-economics win: it halves a
                        // ~640-kernel tick); prefill row-slices the same
                        // plane at offsets 0/ff, byte-identical to separate.
                        let g = gate.kq().is_none().then(|| gate.q8())?;
                        let d = down.kq().is_none().then(|| down.q8())?;
                        up.kq().is_none().then_some(())?;
                        // native-bf16 source when available (no Q8 hop)
                        if let (Some(gb), Some(ub), Some(db)) = (
                            st_bytes(&format!("blk.{i}.ffn_gate.weight")),
                            st_bytes(&format!("blk.{i}.ffn_up.weight")),
                            st_bytes(&format!("blk.{i}.ffn_down.weight")),
                        ) {
                            // PADDOCK_F8_ROWSCALE: the scale-free per-row
                            // stream (1.0 B/param, -3% decode bytes; decode
                            // rides f8r, prefill FFN falls back to Q8 mmq -
                            // an A/B mode for the decode-dominant configs,
                            // labeled precision class like the rest)
                            if paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_some()
                                && exec.has_f8r()
                            {
                                return Some([
                                    (
                                        exec.bf16_to_f8r_concat2(&gb, &ub, g.dims[0]).ok()?,
                                        g.dims[0],
                                        2 * g.dims[1],
                                    ),
                                    (
                                        exec.bf16_to_f8r(&db, d.dims[0], d.dims[1]).ok()?,
                                        d.dims[0],
                                        d.dims[1],
                                    ),
                                ]);
                            }
                            return Some([
                                (
                                    exec.bf16_to_f8w_concat2(&gb, &ub).ok()?,
                                    g.dims[0],
                                    2 * g.dims[1],
                                ),
                                (exec.bf16_to_f8w(&db).ok()?, d.dims[0], d.dims[1]),
                            ]);
                        }
                        let gu_q8 = exec
                            .repack_q8_concat2(
                                map,
                                &format!("blk.{i}.ffn_gate.weight"),
                                &format!("blk.{i}.ffn_up.weight"),
                            )
                            .ok()?;
                        let gu = exec.q8_0_to_f8w(&gu_q8).ok()?;
                        drop(gu_q8);
                        Some([
                            (gu, g.dims[0], 2 * g.dims[1]),
                            (exec.q8_0_to_f8w(d).ok()?, d.dims[0], d.dims[1]),
                        ])
                    })(),
                    // NVFP4 checkpoint layers build the same f8w prefill
                    // planes off the checkpoint's own values. Without them
                    // the wide prefill pass falls to `nvf4_ffn`, whose W4A16
                    // tcp kernel is a DECODE-band arm: a solo 2048-token
                    // prefill spends 863.8 of 950.4 ms
                    // (90.9%) in `pd_nvf4_gemm_tcp_kernel` at 5.1 ms per
                    // GEMM. Same lane with a fast FFN prefill arm: 131.0 ms
                    // (7.3x), and the 16x512 batched prefill 4027 -> 566 ms.
                    // Gated on real headroom like the f8t build above.
                    // Not built when the W4A4 family's wide arm serves the
                    // band (`nvf4_wide_w4a4`): the twin measures slower than
                    // f4t at every prefill width and costs ~15 GB.
                    Ffn::Nvf4Dense { .. } if nvf4_wide_w4a4(&exec) => None,
                    Ffn::Nvf4Dense { gate, up, down } => {
                        (|| -> Option<[(RepackedMxfp4, usize, usize); 2]> {
                            // one f32 dequant buffer at a time (4 B/param) plus
                            // the ~1.03 B/param that stays resident, per plane
                            let need = (gate.out_dim * gate.in_dim * 5) as u64;
                            if !exec.vram_headroom().is_some_and(|h| h > need + (24 << 30)) {
                                return None;
                            }
                            let (gi, go) = (gate.in_dim, gate.out_dim);
                            if up.in_dim != gi || up.out_dim != go {
                                return None;
                            }
                            Some([
                                (exec.nvf4_to_f8w_concat2(gate, up).ok()?, gi, 2 * go),
                                (exec.nvf4_to_f8w(down).ok()?, down.in_dim, down.out_dim),
                            ])
                        })()
                    }
                    Ffn::Moe(_) => None,
                }
            } else {
                None
            };
            if i == 0 {
                tracing::info!(
                    built = f8ffn.is_some(),
                    f8row = f8row_ffn.is_some(),
                    nvf4_src = matches!(&ffn, Ffn::Nvf4Dense { .. }),
                    nvf4_wide_w4a4 = nvf4_wide_w4a4(&exec),
                    "qwen35 f8-ffn decode lane: result"
                );
            }

            // Checkpoint-native NVFP4 gate|up plane (slots 462-465). The
            // decode tick reads the MLP at 0.5 B/param instead of the f8t
            // planes' 1.0 - the MLP is 17.1 of a 25.6 GB tick, so this is
            // most of the decode byte budget, and the checkpoint already
            // ships these 56 layers as those nibbles. Costs only the
            // fused nibble copy plus a repacked scale vector; the f8t planes
            let f8t_ffn = if f8t_ffn_enabled(&exec) {
                match &ffn {
                    Ffn::Dense { gate, up, down } => {
                        (|| -> Option<[crate::gpu::F8TilePlane; 2]> {
                            let g = gate.kq().is_none().then(|| gate.q8())?;
                            let u = up.kq().is_none().then(|| up.q8())?;
                            let d = down.kq().is_none().then(|| down.q8())?;
                            let (gi, go) = (g.dims[0], g.dims[1]);
                            // 128-multiple dims are the repacker's contract
                            if gi % 128 != 0 || go % 128 != 0 {
                                return None;
                            }
                            let gt = exec
                                .f8_repack_tiles(exec.q8_0_to_f8row(g).ok()?, gi, go)
                                .ok()?;
                            let ut = exec
                                .f8_repack_tiles(exec.q8_0_to_f8row(u).ok()?, gi, go)
                                .ok()?;
                            let n = gt.tiles.len();
                            let mut tiles = exec.alloc_u8(n + ut.tiles.len()).ok()?;
                            let mut scale: cudarc::driver::CudaSlice<f32> =
                                exec.stream.alloc_zeros(go * 2).ok()?;
                            let mut v = tiles.try_slice_mut(0..n)?;
                            exec.stream.memcpy_dtod(&gt.tiles, &mut v).ok()?;
                            let mut v = tiles.try_slice_mut(n..n + ut.tiles.len())?;
                            exec.stream.memcpy_dtod(&ut.tiles, &mut v).ok()?;
                            let mut v = scale.try_slice_mut(0..go)?;
                            exec.stream.memcpy_dtod(&gt.scale, &mut v).ok()?;
                            let mut v = scale.try_slice_mut(go..2 * go)?;
                            exec.stream.memcpy_dtod(&ut.scale, &mut v).ok()?;
                            let (di, dn) = (d.dims[0], d.dims[1]);
                            if di % 128 != 0 || dn % 128 != 0 {
                                return None;
                            }
                            let dt = exec
                                .f8_repack_tiles(exec.q8_0_to_f8row(d).ok()?, di, dn)
                                .ok()?;
                            let pgu = crate::gpu::F8TilePlane {
                                tiles,
                                scale,
                                flat: None,
                                flat_minb: 0,
                                flat_gui: false,
                                scale_il: None,
                            };
                            // gluq silu twin DEFAULT on (gates: rscale bit-equal,
                            // mean parity, exact probe; coherence symmetric with
                            // the control; 3-leg interleaved wide A/B with no
                            // overlap on either throughput or ITL):
                            let dt = dt;
                            Some([pgu, dt])
                        })()
                    }
                    // NVFP4 checkpoint layers build the same tile planes off
                    // the checkpoint's own values (nvf4_to_f8row). The enum
                    // comment above says none of the Dense aux lanes build for
                    // an Nvf4Dense layer "which is also what makes the lane
                    // fit" - that memory argument is right on a 96 GB card and
                    // wrong here: the B200 lane loads at 46.6 GB with 159 GB
                    // of headroom, and paying ~1 byte/param buys ~2.7x the
                    // wide-batch throughput. Gated on real headroom so the
                    // small-card configuration keeps the fp4 residency.
                    Ffn::Nvf4Dense { gate, up, down } => {
                        (|| -> Option<[crate::gpu::F8TilePlane; 2]> {
                            // one f32 dequant buffer at a time (4 B/param) plus the
                            // 1 B/param that stays resident, per plane
                            let need = (gate.out_dim * gate.in_dim * 5) as u64;
                            if !exec.vram_headroom().is_some_and(|h| h > need + (24 << 30)) {
                                return None;
                            }
                            let (gi, go) = (gate.in_dim, gate.out_dim);
                            if gi % 128 != 0 || go % 128 != 0 || up.in_dim != gi || up.out_dim != go
                            {
                                return None;
                            }
                            let gt = exec
                                .f8_repack_tiles(exec.nvf4_to_f8row(gate).ok()?, gi, go)
                                .ok()?;
                            let ut = exec
                                .f8_repack_tiles(exec.nvf4_to_f8row(up).ok()?, gi, go)
                                .ok()?;
                            let n = gt.tiles.len();
                            let mut tiles = exec.alloc_u8(n + ut.tiles.len()).ok()?;
                            let mut scale: cudarc::driver::CudaSlice<f32> =
                                exec.stream.alloc_zeros(go * 2).ok()?;
                            let mut v = tiles.try_slice_mut(0..n)?;
                            exec.stream.memcpy_dtod(&gt.tiles, &mut v).ok()?;
                            let mut v = tiles.try_slice_mut(n..n + ut.tiles.len())?;
                            exec.stream.memcpy_dtod(&ut.tiles, &mut v).ok()?;
                            let mut v = scale.try_slice_mut(0..go)?;
                            exec.stream.memcpy_dtod(&gt.scale, &mut v).ok()?;
                            let mut v = scale.try_slice_mut(go..2 * go)?;
                            exec.stream.memcpy_dtod(&ut.scale, &mut v).ok()?;
                            let (di, dn) = (down.in_dim, down.out_dim);
                            if di % 128 != 0 || dn % 128 != 0 {
                                return None;
                            }
                            let dt = exec
                                .f8_repack_tiles(exec.nvf4_to_f8row(down).ok()?, di, dn)
                                .ok()?;
                            let pgu = crate::gpu::F8TilePlane {
                                tiles,
                                scale,
                                flat: None,
                                flat_minb: 0,
                                flat_gui: false,
                                scale_il: None,
                            };
                            let dt = dt;
                            Some([pgu, dt])
                        })()
                    }
                    Ffn::Moe(_) => None,
                }
            } else {
                None
            };
            if i == 0 {
                tracing::info!(
                    enabled = f8t_ffn_enabled(&exec),
                    built = f8t_ffn.is_some(),
                    nvf4_src = matches!(&ffn, Ffn::Nvf4Dense { .. }),
                    "qwen35 tcgen05 (f8t) FFN decode lane"
                );
            }
            bs_f8t_ffn_planes.push(f8t_ffn);
            // Mixer-projection half of the same lane. Both layer kinds reduce
            // to the same shape - one fused input projection, one output
            // projection - so they share one builder and one dispatch contract.
            let f8t_attn = if f8t_attn_enabled(&exec) {
                // Concatenate N Q8_0 tensors that share an in_dim into one tile
                // plane. Byte-exact because each tensor's tile stream is
                // plane-relative; scales concatenate row-wise alongside.
                let concat =
                    |mut parts: Vec<crate::gpu::F8TilePlane>| -> Option<crate::gpu::F8TilePlane> {
                        if parts.len() == 1 {
                            return parts.pop();
                        }
                        let (nt, ns) = parts
                            .iter()
                            .fold((0, 0), |(t, s), p| (t + p.tiles.len(), s + p.scale.len()));
                        let mut tiles = exec.alloc_u8(nt).ok()?;
                        let mut scale: cudarc::driver::CudaSlice<f32> =
                            exec.stream.alloc_zeros(ns).ok()?;
                        let (mut dpos, mut spos) = (0usize, 0usize);
                        for p in &parts {
                            let mut v = tiles.try_slice_mut(dpos..dpos + p.tiles.len())?;
                            exec.stream.memcpy_dtod(&p.tiles, &mut v).ok()?;
                            dpos += p.tiles.len();
                            let mut v = scale.try_slice_mut(spos..spos + p.scale.len())?;
                            exec.stream.memcpy_dtod(&p.scale, &mut v).ok()?;
                            spos += p.scale.len();
                        }
                        Some(crate::gpu::F8TilePlane {
                            tiles,
                            scale,
                            flat: None,
                            flat_minb: 0,
                            flat_gui: false,
                            scale_il: None,
                        })
                    };
                let parts_of = |ws: &[&QuantW]| -> Option<Vec<crate::gpu::F8TilePlane>> {
                    let inp = ws[0].dims()[0];
                    let mut parts = Vec::with_capacity(ws.len());
                    for w in ws {
                        let q = w.kq().is_none().then(|| w.q8())?;
                        let (i, o) = (q.dims[0], q.dims[1]);
                        // 128-multiple dims are the repacker's contract, and
                        // tile-aligned out offsets fall out of the same check
                        if i != inp || i % 128 != 0 || o % 128 != 0 {
                            return None;
                        }
                        parts.push(
                            exec.f8_repack_tiles(exec.q8_0_to_f8row(q).ok()?, i, o)
                                .ok()?,
                        );
                    }
                    Some(parts)
                };
                let fuse =
                    |ws: &[&QuantW]| -> Option<crate::gpu::F8TilePlane> { concat(parts_of(ws)?) };
                // alpha||beta as one zero-padded 128-row tile block. They are
                // [embd, n_v_heads] = [5120, 48] each: too narrow for the
                // repacker's 128-multiple contract on their own, but together
                // with 32 zero rows they make exactly one row-tile, so they
                // ride the in_qkv|gate_w stream instead of costing two
                // separate gemv launches. Unfused, that pair profiles at 96
                // kernels/step, grid 48, 4.91 us each -- ~0.47 ms/step, and
                // it is pure overhead: engines that fold these projections
                // into their GEMMs have no equivalent kernels at all.
                // Padding rows are zero DATA and zero SCALE, and nothing ever
                // row-slices them back out.
                let fuse_ab = |aw: &RepackedQ8,
                               bw: &RepackedQ8,
                               inp: usize|
                 -> Option<crate::gpu::F8TilePlane> {
                    let (na, nb) = (aw.dims[1], bw.dims[1]);
                    if aw.dims[0] != inp || bw.dims[0] != inp || na + nb > 128 {
                        return None;
                    }
                    let ra = exec.q8_0_to_f8row(aw).ok()?;
                    let rb = exec.q8_0_to_f8row(bw).ok()?;
                    let mut data = exec.alloc_u8(inp * 128).ok()?; // zeroed
                    let mut scale: cudarc::driver::CudaSlice<f32> =
                        exec.stream.alloc_zeros(128).ok()?;
                    {
                        let mut v = data.try_slice_mut(0..inp * na)?;
                        exec.stream.memcpy_dtod(&ra.data, &mut v).ok()?;
                        let mut v = data.try_slice_mut(inp * na..inp * (na + nb))?;
                        exec.stream.memcpy_dtod(&rb.data, &mut v).ok()?;
                        let mut v = scale.try_slice_mut(0..na)?;
                        exec.stream.memcpy_dtod(&ra.scale, &mut v).ok()?;
                        let mut v = scale.try_slice_mut(na..na + nb)?;
                        exec.stream.memcpy_dtod(&rb.scale, &mut v).ok()?;
                    }
                    exec.f8_repack_tiles(crate::gpu::F8RowPlane { data, scale }, inp, 128)
                        .ok()
                };
                match &mixer {
                    Mixer::Full(w) => (|| {
                        // wq is already [embd, 2*q_dim] (query || out-gate);
                        // the fusion appends k and v to the same stream.
                        let inp = fuse(&[&w.wq, &w.wk, &w.wv])?;
                        let out = fuse(&[&w.wo])?;
                        Some([inp, out])
                    })(),
                    Mixer::Linear(w) => (|| {
                        let mut parts = parts_of(&[&w.in_qkv, &w.gate_w])?;
                        // Best-effort: a layer whose alpha/beta are not Q8_0
                        // (UD k-quant exports ship F16 and take the ab_f32
                        // route) simply keeps the two-gemv form. The decode
                        // side detects the fold from the plane's out_dim, so
                        // the two cases can coexist across layers.
                        if let (Some(aw), Some(bw)) = (w.alpha_w.as_ref(), w.beta_w.as_ref())
                            && let Some(ab) = fuse_ab(aw, bw, w.in_qkv.dims()[0])
                        {
                            parts.push(ab);
                        }
                        let inp = concat(parts)?;
                        let out = fuse(&[&w.out_w])?;
                        Some([inp, out])
                    })(),
                }
            } else {
                None
            };
            if i == 0 || i == 1 {
                tracing::info!(
                    layer = i,
                    kind = match &mixer {
                        Mixer::Full(_) => "full",
                        Mixer::Linear(_) => "deltanet",
                    },
                    enabled = f8t_attn_enabled(&exec),
                    built = f8t_attn.is_some(),
                    "qwen35 tcgen05 (f8t) mixer-projection decode lane"
                );
            }
            bs_f8t_attn_planes.push(f8t_attn);
            // tile-linear conversion for the FFN planes (same lane as the
            // W8 projections in build_w8_planes; marker-scale dispatch)
            let f8ffn = if ffn_lin_done {
                // the early build already converted; converting twice would
                // repack an already-boxed plane
                f8ffn
            } else if f8lin_enabled(&exec) {
                match f8ffn {
                    Some([a, b]) => {
                        let cv = |t: (RepackedMxfp4, usize, usize)| -> Result<_, GpuModelError> {
                            if t.1.is_multiple_of(128) && t.2.is_multiple_of(16) {
                                let (w, i, o) = t;
                                Ok((exec.f8w_repack_lin(w, i, o)?, i, o))
                            } else {
                                Ok(t)
                            }
                        };
                        Some([cv(a)?, cv(b)?])
                    }
                    None => None,
                }
            } else {
                f8ffn
            };
            bs_f8ffn_planes.push(f8ffn);
            if f8row_ffn.is_some() {
                tracing::info!(
                    layer = i,
                    "qwen35 f8row FFN: checkpoint-exact fp8 planes built (gate/up/down, per-row scales)"
                );
            }
            bs_f8row_ffn_planes.push(f8row_ffn);
            // FFN REPLACE, in the same iteration that built the twin: the Q8_0
            // gate/up/down go now, so this layer's freed bytes are what the
            // next layer's upload allocates into and the pool never holds both
            // formats. `covered` is the twin this loop just produced and the
            // layout the b=1 GEMV actually reads - not a recollection of it.
            // The f8row twin covers by construction (every width arms).
            let f8row_cov = bs_f8row_ffn_planes.last().is_some_and(|o| o.is_some());
            if ffn_replace || f8row_cov {
                let covered = f8row_cov
                    || bs_f8ffn_planes
                        .last()
                        .and_then(|o| o.as_ref())
                        .is_some_and(|p| p[0].0.is_lin() && p[1].0.is_lin());
                if covered && let Ffn::Dense { gate, up, down } = &mut ffn {
                    for w in [gate, up, down] {
                        let n = replace_q8(&exec, w)?;
                        ffn_freed += n;
                        ffn_replaced += usize::from(n > 0);
                    }
                }
            }
            // PROJECTION lane: build this layer's e4m3 twins and drop its Q8_0
            // sources, now that every mixer-derived plane (bs_dn, f8t_attn)
            // has been taken. Same iteration, same reason as the FFN half.
            if let Some(w8) = w8_early {
                // built from file transients before the seats existed; the
                // "replace" already happened as an upload that never ran
                let seats: &[&str] = if is_full {
                    &[
                        "attn_q.weight",
                        "attn_k.weight",
                        "attn_v.weight",
                        "attn_output.weight",
                    ]
                } else {
                    &["attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"]
                };
                for n in seats {
                    if let Some(t) = map.tensor_info(&format!("blk.{i}.{n}")) {
                        proj_skipped_bytes += t.byte_size().unwrap_or(0);
                        proj_replaced += 1;
                    }
                }
                bs_w8.push(w8);
            } else if w8_lane {
                let (w8, freed, n) =
                    build_w8_mixer(&exec, &mut mixer, i, proj_replace, &|i, name| {
                        // DN value-head ordering (found in the
                        // garbage-at-r>=512 hunt): llama.cpp's GGUF converter
                        // REORDERS the value heads of in_proj_z (attn_gate
                        // rows) and out_proj (ssm_out input cols) - row-matching
                        // blk.0 against the official FP8 snapshot mapped GGUF
                        // gate head h -> HF head pi(h) (1->4, 2->6, 3->10,
                        // 15->23) while q/k/v/o/in_qkv are identity. Every
                        // consumer expects GGUF order, so raw HF bytes for
                        // these two scramble all 48 DN layers once the W8
                        // planes engage (>= w8_min_batch rows). Until an
                        // ingestion permute lands, source them from the exact
                        // Q8-derived path (same W8A8 plane class; native
                        // sourcing keeps the rest).
                        if name == "attn_gate.weight" || name == "ssm_out.weight" {
                            return None;
                        }
                        st_bytes(&format!("blk.{i}.{name}")).map(|c| c.into_owned())
                    })?;
                bs_w8.push(w8);
                proj_freed += freed;
                proj_replaced += n;
            }
            // byte-passthrough (bs) FFN decode planes: the checkpoint's raw
            // e4m3 bytes serve unmodified (marker-8 planes; decode-only).
            let f8bs = if fp8_bs {
                (|| {
                    use paddock_models::safetensors::StDtype;
                    let st = fp8_native.as_ref()?;
                    let raw = |g: &str| -> Option<(Vec<u8>, Vec<f32>, usize, usize)> {
                        let hf = paddock_models::safetensors::qwen35_hf_name(g)?;
                        let (t, b) = st.bytes(&hf)?;
                        if t.dtype != StDtype::F8E4m3 || t.shape.len() != 2 {
                            return None;
                        }
                        let (rows, cols) = (t.shape[0], t.shape[1]);
                        let (ts, sb) = st.bytes(&format!("{hf}_scale_inv"))?;
                        if ts.dtype != StDtype::Bf16
                            || ts.shape != [rows.div_ceil(128), cols.div_ceil(128)]
                        {
                            return None;
                        }
                        let scales: Vec<f32> = sb
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                            .collect();
                        Some((b.to_vec(), scales, rows, cols))
                    };
                    let (mut fb, mut fs, gr, gc) = raw(&format!("blk.{i}.ffn_gate.weight"))?;
                    let (ub, us, ur, uc) = raw(&format!("blk.{i}.ffn_up.weight"))?;
                    if uc != gc {
                        return None;
                    }
                    // fused gate|up: row-wise concat of bytes and scale rows
                    fb.extend_from_slice(&ub);
                    fs.extend_from_slice(&us);
                    let (db, ds, dr, dc) = raw(&format!("blk.{i}.ffn_down.weight"))?;
                    let d_gu = exec.stream.clone_htod(&fb).ok()?;
                    let gu = exec.f8w_build_lin_bs(d_gu, &fs, gc, gr + ur).ok()?;
                    let d_dn = exec.stream.clone_htod(&db).ok()?;
                    let dn = exec.f8w_build_lin_bs(d_dn, &ds, dc, dr).ok()?;
                    tracing::info!("fp8-bs planes layer {i}: gu {gc}x{} dn {dc}x{dr}", gr + ur);
                    Some([(gu, gc, gr + ur), (dn, dc, dr)])
                })()
            } else {
                None
            };
            bs_f8ffn_bs_planes.push(f8bs);
            layers.push(Qwen35Layer {
                attn_norm: dt("attn_norm.weight")?,
                post_norm: dt("post_attention_norm.weight")?,
                ffn,
                mixer,
            });
            // MoE iterations don't reach the Dense arm's push - keep indices aligned
            if bs_gu_planes.len() < layers.len() {
                bs_gu_planes.push(None);
            }
        }
        // K-quant residency scan: routing (serial-only in stage 1) + the
        // per-pass dequant-scratch size. Head/embedding excluded from the max
        // (never dequanted whole - GEMV/gather only).
        let mut kq_max_elems = 0usize;
        let mut kq_resident = matches!(&tok_embd, TokEmbd::Kq(_));
        fn note_kq(w: &QuantW, resident: &mut bool, max_elems: &mut usize) {
            if let Some(k) = w.kq() {
                *resident = true;
                *max_elems = (*max_elems).max(k.dims.iter().product());
            }
        }
        for l in &layers {
            match &l.mixer {
                Mixer::Full(w) => {
                    for q in [&w.wq, &w.wk, &w.wv, &w.wo] {
                        note_kq(q, &mut kq_resident, &mut kq_max_elems);
                    }
                }
                Mixer::Linear(w) => {
                    for q in [&w.in_qkv, &w.gate_w, &w.out_w] {
                        note_kq(q, &mut kq_resident, &mut kq_max_elems);
                    }
                    // non-Q8 alpha/beta force the serial spine too (batched
                    // decode's fused gate kernel is Q8_0-class)
                    kq_resident |= w.alpha_w.is_none();
                }
            }
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    for q in [gate, up, down] {
                        note_kq(q, &mut kq_resident, &mut kq_max_elems);
                    }
                }
                // nvf4 planes are never k-quant-resident
                Ffn::Nvf4Dense { .. } => {}
                // k-quant EXPERT seats mark residency too (they need the
                // d_ssums scratch for the mu term) but never enter
                // kq_max_elems - experts are never dequanted whole.
                Ffn::Moe(m) => {
                    kq_resident |= m.gate_exps.q8().is_none()
                        || m.up_exps.q8().is_none()
                        || m.down_exps.q8().is_none();
                }
            }
        }
        let v_back = vfree();
        tracing::info!(
            "qwen35 VRAM  backbone weights (attn/deltanet/ffn) {:>7.2} GB",
            gb(v_embd.saturating_sub(v_back))
        );

        // MTP draft block (27B and up): the nextn block stored immediately after the
        // backbone (dense OR MoE FFN - both handled below). It is consumed only by the
        // speculative-decode draft path, so skip loading it entirely when spec is off
        // (PADDOCK_NO_SPEC): that reclaims the whole draft block's VRAM - ~1 GB+ on the
        // MoE variant (a full attn + MoE-FFN layer) - at zero cost, since the draft
        // path is disabled anyway. Default (spec on) loads it for the ~8% decode gain.
        // K-quant models load it too: the spec matmuls ride
        // the W4A8 rungs (mmq's k-quant arm - dp4a + K-split mma), and the
        // nextn block's own weights go through the same per-tensor QuantW
        // dispatch below. MoE-FFN nextn blocks still need Q8 expert seats.
        let spec_wanted = std::env::var_os("PADDOCK_NO_SPEC").is_none();
        let mtp = if n_nextn > 0 && spec_wanted {
            assert_eq!(n_nextn, 1, "only a single nextn/MTP block is supported");
            let i = n_layers; // blk.<n_layers>.*
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let qt = |name: &str| exec.load_quantw(map, &format!("blk.{i}.{name}"));
            let qt8 = |name: &str| exec.repack_q8(map, &format!("blk.{i}.{name}"));
            Some(MtpWeights {
                eh_proj: qt("nextn.eh_proj.weight")?,
                enorm: dt("nextn.enorm.weight")?,
                hnorm: dt("nextn.hnorm.weight")?,
                head_norm: dt("nextn.shared_head_norm.weight")?,
                attn_norm: dt("attn_norm.weight")?,
                post_norm: dt("post_attention_norm.weight")?,
                attn: FullAttnWeights {
                    wq: qt("attn_q.weight")?,
                    wk: qt("attn_k.weight")?,
                    wv: qt("attn_v.weight")?,
                    q_norm: dt("attn_q_norm.weight")?,
                    k_norm: dt("attn_k_norm.weight")?,
                    wo: qt("attn_output.weight")?,
                },
                ffn: if moe.is_some() {
                    // nextn/MTP MoE: same per-tensor seat rule as the
                    // backbone (k-quant files ship k-quant nextn experts too;
                    // the spec paths pin sorted_ok=false, so kq seats ride
                    // the token-batched pair there anyway).
                    let kq_exp = |name: &str| kq_expert_seat(&exec, map, i, name);
                    let (gate_exps, up_exps) = match (
                        kq_exp("ffn_gate_exps.weight")?,
                        kq_exp("ffn_up_exps.weight")?,
                    ) {
                        (Some(g), Some(u)) => (g, u),
                        _ => (
                            ExpW::Q8(qt8("ffn_gate_exps.weight")?),
                            ExpW::Q8(qt8("ffn_up_exps.weight")?),
                        ),
                    };
                    let down_exps = match kq_exp("ffn_down_exps.weight")? {
                        Some(d) => d,
                        None => ExpW::Q8(qt8("ffn_down_exps.weight")?),
                    };
                    Ffn::Moe(MoeFfnWeights {
                        router_w: dt("ffn_gate_inp.weight")?,
                        gate_exps,
                        up_exps,
                        down_exps,
                        shexp_gate_inp: dt("ffn_gate_inp_shexp.weight")?,
                        shexp_gate: qt("ffn_gate_shexp.weight")?,
                        shexp_up: qt("ffn_up_shexp.weight")?,
                        shexp_down: qt("ffn_down_shexp.weight")?,
                        gate_exps_fp4: None,
                        up_exps_fp4: None,
                        down_exps_fp4: None,
                        moe_zero_bias: None,
                        cache: None,
                    })
                } else {
                    Ffn::Dense {
                        gate: qt("ffn_gate.weight")?,
                        up: qt("ffn_up.weight")?,
                        down: qt("ffn_down.weight")?,
                    }
                },
            })
        } else {
            None
        };

        // A no-op sink per query head - -inf, the exact softmax identity.
        let sinks = exec.alloc_no_sinks(n_heads)?;

        let out_norm = exec.upload(map, "output_norm.weight")?;
        // Untied output projection, with the tied-embedding fallback (some
        // exports omit `output.weight` and reuse `token_embd`). Per-tensor
        // dispatch: UD files ship the head Q6_K.
        let mut output = if map.tensor_info("output.weight").is_some() {
            exec.load_quantw(map, "output.weight")?
        } else {
            exec.load_quantw(map, "token_embd.weight")?
        };
        let kq_resident = kq_resident || output.kq().is_some();
        // f8 lm_head (PADDOCK_F8_LMHEAD): Q8 head -> f8w -> tile-linear, the
        // same conversion pipeline as the FFN planes. The mt_dp4a head GEMM
        // measured 870 GB/s (access-pattern bound); the lin stream runs the
        // f8 lane's ~1.4 TB/s class. ~0.8 GB dup; labeled precision class.
        // Default-on wherever the f8t lane is (i.e. sm_100), per that lane's
        // PPL gate: the head contributes +0.22% of the stack's +1.09%. Still
        // opt-in-able by env on other dies; PADDOCK_NO_F8_LMHEAD kills it.
        // Checkpoint-sourced arm: the NVFP4/fp8 exports SHIP an
        // fp8 lm_head (channel-strategy weight_scale island), which is the
        // class it is meant to serve in -- the Q8_0 GGUF head we ran instead
        // cost 528 us x 2785 calls, ~3.8% of a wide leg's GPU. When the
        // fp8_native
        // snapshot carries the head, build the f8 plane from its bytes and
        // default the f8 head on: that is the checkpoint's own serving
        // class. Same kill switch as the Q8-derived arm.
        //
        // The dequant runs in ROW BANDS into a device staging buffer rather
        // than through st_bytes: pdfium's static lib brings Chromium's
        // PartitionAlloc in as the PROCESS allocator, and any host
        // allocation over its ~2 GiB direct-map cap hits its int3
        // IMMEDIATE_CRASH (SIGTRAP, no message - found when the head's whole
        // bf16 image, 2.54 GB, trapped the loader). cudaMalloc
        // is not shimmed, so the full-size buffer lives on device and hosts
        // only ever see ~680 MB bands. The real fix is a pdfium rebuild
        // with the allocator shim off; until then nothing host-side may
        // allocate >= 2 GiB in one piece.
        // ELECTED for the Q8_0 GGUF lane too, wherever the f8d kernel
        // exists: the fp8-native and f8t lanes already shipped the f8d head
        // as the b >= 8 class, and the Q8_0 lane paying the int8 vocab GEMM
        // instead measured as 9.6% of a c8 tick (pd_q8_0_gemm_mt_dp4a,
        // ~1.1 ms). The Q8_0->e4m3 requant A/B wins that band with the other
        // widths flat and spec-verify acceptance unchanged
        // - the class change is labeled, b=1 parity surfaces stay
        // Q8_0-exact via the existing row gates. Costs one duplicate f8
        // plane (vocab x embd bytes, 1.27 GB at 248320x5120).
        let out_f8 = if paddock_models::dev_var_os!("PADDOCK_NO_F8_LMHEAD").is_none()
            && (paddock_models::dev_var_os!("PADDOCK_F8_LMHEAD").is_some()
                || f8t_ffn_enabled(&exec)
                || fp8_native.is_some()
                || exec.has_f8d_gemm_mma_ks())
        {
            match &output {
                QuantW::Q8(q8) => (|| {
                    let d = &q8.dims;
                    let (i, o) = (d[0], d[1]);
                    if i % 128 != 0 || o % 16 != 0 {
                        return None;
                    }
                    let ck = (|| -> Option<RepackedMxfp4> {
                        use paddock_models::safetensors::StDtype;
                        let st = fp8_native.as_ref()?;
                        let (t, wb) = st.bytes("lm_head.weight")?;
                        if t.dtype != StDtype::F8E4m3 || t.shape != [o, i] {
                            return None;
                        }
                        let (ts, sb) = st.bytes("lm_head.weight_scale")?;
                        if ts.dtype != StDtype::Bf16 || ts.shape.iter().product::<usize>() != o {
                            return None;
                        }
                        let n_bytes = o * i * 2;
                        let mut staged = exec.alloc_u8(n_bytes).ok()?;
                        let band = 65536usize.min(o);
                        let mut host = vec![0u8; band * i * 2];
                        let mut r0 = 0usize;
                        while r0 < o {
                            let rows = band.min(o - r0);
                            let chunk = &mut host[..rows * i * 2];
                            let nthreads = std::thread::available_parallelism()
                                .map(|n| n.get().min(16))
                                .unwrap_or(8);
                            let tband = rows.div_ceil(nthreads);
                            std::thread::scope(|scp| {
                                for (ti, hchunk) in chunk.chunks_mut(tband * i * 2).enumerate() {
                                    let rb = r0 + ti * tband;
                                    scp.spawn(move || {
                                        for (rr, orow) in hchunk.chunks_mut(i * 2).enumerate() {
                                            let r = rb + rr;
                                            let wrow = &wb[r * i..(r + 1) * i];
                                            let sc16 =
                                                u16::from_le_bytes([sb[r * 2], sb[r * 2 + 1]]);
                                            let scale = f32::from_bits((sc16 as u32) << 16);
                                            for c in 0..i {
                                                let v = e4m3_to_f32(wrow[c]) * scale;
                                                let bits = v.to_bits();
                                                let bf = ((bits + 0x7FFF + ((bits >> 16) & 1))
                                                    >> 16)
                                                    as u16;
                                                orow[c * 2..c * 2 + 2]
                                                    .copy_from_slice(&bf.to_le_bytes());
                                            }
                                        }
                                    });
                                }
                            });
                            exec.upload_u8_at(chunk, &mut staged, r0 * i * 2).ok()?;
                            r0 += rows;
                        }
                        let w = exec.bf16_to_f8w_dev(&staged, n_bytes).ok()?;
                        tracing::info!("qwen35: f8 lm_head from checkpoint fp8 plane ({i}x{o})");
                        Some(w)
                    })();
                    let w = match ck {
                        Some(w) => w,
                        None if paddock_models::dev_var_os!("PADDOCK_F8_LMHEAD")
                            .is_some()
                            || f8t_ffn_enabled(&exec)
                            // Q8_0-lane election (see the gate comment above)
                            || exec.has_f8d_gemm_mma_ks() =>
                        {
                            let w = exec.q8_0_to_f8w(q8).ok()?;
                            tracing::info!("qwen35: f8 lm_head plane built ({i}x{o})");
                            w
                        }
                        None => return None,
                    };
                    // Tile-linear only where the lin lane is on: every consumer
                    // routes an `is_lin()` plane to f8_gemm_lin_kt, and the pack
                    // nulls that entry when its TMA encoder is unavailable, so a
                    // lin head with the lane off is a MissingOp on the first
                    // prefill. Row-major here keeps it on f8d_gemm_mma_ks.
                    let lin = if f8lin_enabled(&exec) {
                        exec.f8w_repack_lin(w, i, o).ok()?
                    } else {
                        w
                    };
                    Some((lin, i, o))
                })(),
                _ => None,
            }
        } else {
            None
        };

        // f8t TILE plane for the head. vocab 248320 = 1940 row-tiles, far above
        // the wmma route's 256-tile gate, so the head lands on the same
        // warp-level mma.sync path that carries the FFN gate_up -- where the
        // f8d lin head measured 0.505 ms/step, ~6% of the whole decode tick.
        // Costs ~1.3 GB of VRAM for the duplicate plane; only built where the
        // f8t lane runs (sm_100). Kill: PADDOCK_NO_F8T_LMHEAD.
        let out_f8t = if f8t_ffn_enabled(&exec)
            && paddock_models::dev_var_os!("PADDOCK_NO_F8T_LMHEAD").is_none()
        {
            match &output {
                QuantW::Q8(q8) => (|| {
                    let (i, o) = (q8.dims[0], q8.dims[1]);
                    if i % 128 != 0 || o % 128 != 0 {
                        return None;
                    }
                    let p = exec
                        .f8_repack_tiles(exec.q8_0_to_f8row(q8).ok()?, i, o)
                        .ok()?;
                    tracing::info!("qwen35: f8t lm_head tile plane built ({i}x{o})");
                    Some((p, i, o))
                })(),
                _ => None,
            }
        } else {
            None
        };

        // lm_head REPLACE. Not a reclaim: the Q8_0 head goes the moment its
        // e4m3 twin exists, at the point of conversion, so there is no window
        // with two full residencies and no separate reclaim block carrying
        // its own coverage argument. This is the shape vLLM and SGLang have
        // -- `process_weights_after_loading` replaces the parameter and the
        // source refcount drops there; nothing keeps a second format for a
        // small-batch path.
        //
        // The precondition is the same election every consumer calls
        // (`head_f8`, which tests `rows >= f8_head_min()`), so it cannot drift
        // from them the way the FFN reclaim's `<= 1` drifted from its
        // consumers' `>`. f8_head_min() must cover a single row, because a
        // one-row head call is what prefill, vision, the draft chain and the
        // finishing-span epilogue all issue.
        //
        // f8t (sm_100) keeps its own twin: that plane is built from `output`
        // above and the f8t sites read it, not the Q8 head, but the tile lane
        // is only elected at b <= 64 so the Q8 head is still its fallback.
        // Nothing is dropped where f8t is live.
        if let (Some(_), true, None) = (out_f8.as_ref(), f8_head_min() <= 1, out_f8t.as_ref())
            && let QuantW::Q8(q) = &mut output
        {
            let freed = (q.data.len() + q.scale.len()) as u64;
            *q = crate::gpu::RepackedQ8 {
                data: exec.alloc_u8(32)?,
                scale: exec.alloc_u8(32)?,
                dims: q.dims.clone(),
            };
            tracing::info!(
                "qwen35 VRAM  lm_head REPLACE: Q8_0 head dropped, {:.2} GB returned \
                     (every row count serves the e4m3 head)",
                gb(freed)
            );
        }

        // b1: fp8 W8A8 planes for the dense projections. Opt-in and quality-gated
        // (lossy vs Q8_0) - off by default. Built here while `layers`/`exec` are
        // still owned locally, before they move into `Self`. K-quant models skip
        // the aux-plane builders entirely (Q8_0-source converters, batched-only
        // consumers).
        // PROJECTION REPLACE (the 7.4 GB). Same shape as the
        // lm_head REPLACE above and for the same reason: the drop is part of
        // the conversion, so there is no separate coverage argument to drift
        // from the consumers. The precondition tests the same comparisons the
        // consumers spell -- `r > w8_min_batch()` on the prefill arms and
        // `b >= f8_dec_min()` on the decode arms -- so a floor that does not
        // cover a band cannot silently leave a reader behind.
        //
        // The drop itself lives inside build_w8_planes, per layer, right after
        // that layer's e4m3 planes exist (the pool-fragmentation
        // audit): a batch pass at the end of load leaves every freed hole
        // sandwiched between live replacement planes and the driver never gets
        // the memory back. See `replace_q8`.
        //
        // Consumer audit, enumerated by GREP not recollection (the FFN
        // reclaim shipped twice because its audit was written from memory):
        //   prefix.rs   10 sites, `r > w8_min`                         OK at 0
        //   batch.rs    10 sites prefill `r > w8_min` + decode
        //               `b >= f8_dec_min()`                            OK at 0/1
        //   multimodal  7 sites -- had no w8 arm at all; wired since,
        //               incl. keep_xn, since the e4m3 arm quantizes from xn
        //   spec.rs     verify walk: 2 arms were on a bare `r >= 8`, so the
        //               c1 verify chunk (k+1 ~ 3-5 rows) read Q8 -- now on
        //               `r > w8_min_batch()`; 7 fallbacks stub_guard'd
        //   spec.rs     forward_chunk + forward.rs serial: reachable only from
        //               generate_greedy_spec / generate_greedy / forward_one,
        //               which no service or runner path calls (tests and
        //               examples only) -- 14 sites refuse loudly instead
        //   f8t_gemm    sites read .dims() only; a stub keeps dims
        // Every remaining Q8 projection arm calls stub_guard first.
        // nv4 and f8t are not in this test deliberately: both take priority over
        // w8 only above their own floors, and with w8_min == 0 the w8 arm holds
        // every band below them, so their Q8 fallbacks are unreachable rather
        // than merely unlikely. What the test does need is a w8 twin for the
        // layer being dropped, and build_w8_planes tests exactly the slots it
        // just filled -- the build and the drop cannot disagree.
        // Built and REPLACED per layer inside the backbone loop above; what is
        // left here is the same kq check the FFN half gets, plus the log.
        if proj_replaced > 0 && kq_resident {
            return Err(GpuModelError::Unsupported(format!(
                "qwen35: the projection REPLACE dropped {proj_replaced} Q8_0 planes under \
                 kq_early=false, but the loaded model is k-quant resident -- LOADER BUG. \
                 Re-run with PADDOCK_QWEN35_W8_MIN=64 to keep the Q8_0 planes resident \
                 while it is fixed."
            )));
        }
        if !bs_w8.is_empty() {
            tracing::info!(
                "qwen35: fp8 W8A8 dense-proj planes built for {} layers (PADDOCK_QWEN35_W8, min batch {})",
                bs_w8.len(),
                w8_min_batch()
            );
        }
        if proj_replaced > 0 {
            tracing::info!(
                "qwen35 VRAM  projection REPLACE: {} Q8_0 projection planes replaced -- {:.2} GB \
                 never allocated + {:.2} GB dropped at the point of conversion (every band \
                 serves e4m3)",
                proj_replaced,
                gb(proj_skipped_bytes),
                gb(proj_freed)
            );
        }

        // nvf4 (W4A4) planes for the dense projections - the fp4×fp4 2×-MMA lever
        // for the 26%-of-prefill proj GEMM. Opt-in + perplexity-gated (lossy).
        let bs_nv4 = if paddock_models::dev_var_os!("PADDOCK_QWEN35_PROJ_NV4").is_some()
            && exec.has_mxfp4_gemm_nv4()
            && !kq_resident
        {
            let planes = build_nv4_planes(&exec, &layers)?;
            tracing::info!(
                "qwen35: nvf4 W4A4 dense-proj planes built for {} layers (PADDOCK_QWEN35_PROJ_NV4, min batch {})",
                planes.len(),
                proj_nv4_min_batch()
            );
            planes
        } else {
            Vec::new()
        };

        // b2: fp4 (W4A8) planes for the routed MoE experts. Opt-in + quality-gated
        // (lossy vs Q8_0), off by default. Built here while `layers`/`exec` are
        // owned locally. One shared all-zeros bias plane (qwen MoE has no bias)
        // is Arc-shared across every MoE layer.
        if std::env::var_os("PADDOCK_QWEN35_MOE_FP4").is_some()
            && exec
                .kernels()
                .map(|k| k.mxfp4_moe_gate_up_bs.is_some())
                .unwrap_or(false)
            && exec
                .kernels()
                .map(|k| k.mxfp4_moe_down_bs.is_some())
                .unwrap_or(false)
            && exec
                .kernels()
                .map(|k| k.mxfp4_gu_interleave.is_some())
                .unwrap_or(false)
            && exec
                .kernels()
                .map(|k| k.q8_0_to_mxfp4.is_some())
                .unwrap_or(false)
            && let Some(md) = moe
        {
            let bias_len = md.n_expert * md.moe_ff.max(embd);
            let mut zb = exec.alloc(bias_len)?;
            exec.stream
                .memset_zeros(&mut zb)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            let zb = Arc::new(zb);
            let mut n = 0usize;
            for l in &mut layers {
                if let Ffn::Moe(m) = &mut l.ffn {
                    // The bs/dp4a MoE kernels read gate+up as one interleaved
                    // plane ([gate 64 | up 64] per row, KC=128) through the
                    // gate_data pointer; up_data is a 16 B dummy they never
                    // deref (gpt_oss.rs:670). Scales stay per-plane. down is
                    // a single plane (no interleave). fp4 planes requant
                    // from Q8 seats - k-quant seats skip (they'd double-
                    // requant; the kq opt-in targets decode, not the fp4
                    // prefill lever).
                    let (Some(g8), Some(u8_), Some(d8)) =
                        (m.gate_exps.q8(), m.up_exps.q8(), m.down_exps.q8())
                    else {
                        continue;
                    };
                    let g = exec.q8_0_to_mxfp4(g8)?;
                    let u = exec.q8_0_to_mxfp4(u8_)?;
                    let gu = exec.gu_interleave(&g, &u, embd / 32, md.n_expert * md.moe_ff)?;
                    m.gate_exps_fp4 = Some(RepackedMxfp4 {
                        data: gu,
                        scale: g.scale,
                    });
                    m.up_exps_fp4 = Some(RepackedMxfp4 {
                        data: exec.alloc_u8(16)?,
                        scale: u.scale,
                    });
                    m.down_exps_fp4 = Some(exec.q8_0_to_mxfp4(d8)?);
                    m.moe_zero_bias = Some(zb.clone());
                    n += 1;
                }
            }
            tracing::info!(
                "qwen35: fp4 W4A8 MoE expert planes built for {n} layers \
                     (PADDOCK_QWEN35_MOE_FP4, min batch {})",
                moe_fp4_min_batch()
            );
        }

        // Q8 ORIGINAL RECLAIM (non-KV-overhead R2.4). The dense FFN carried two
        // full residencies - Q8_0 and e4m3 - because no band could read the
        // e4m3 lin boxes below the width GEMM's floor. Every band can now:
        // b=1 decode on pd_f8lin_gemv (batched AND serial paths), b>=2 decode
        // and all prefill on the f8 GEMM once their floors are lowered. When
        // that is true the Q8 planes are dead weight, so they go - 32-byte
        // stubs keep `dims` for the shape lookups that outlive the data
        // (gemma4's f8r reclaim, same shape).
        //
        // This is deliberately not a switch of its own: it fires exactly when
        // every consumer is covered, and stays off otherwise. A stub whose
        // reader still exists is silent corruption, so the conditions below
        // are the whole safety argument - widen them only with a consumer
        // audit (grep every `Ffn::Dense` arm across batch/forward/prefix).
        // Consumer audit (the first one was wrong on all three
        // counts -- a one-token prompt served '!!!!!!!!' on the default build
        // until this was fixed):
        //   decode  b == 1      -> lin GEMV arm, needs `covered` below   OK
        //   decode  b >= 2      -> `b >= f8_ffn_min()`, min 2            OK
        //   prefill r >= 1      -> `r > f8_ffn_pf_min()`, so the gate must
        //                          be 0, not 1 -- `<= 1` left r == 1 on the
        //                          stubs. This is why the test is `== 0`.
        //   vision  prefill     -> multimodal.rs was on w8_min (64); now on
        //                          f8_ffn_pf_min like the other prefill arms
        //   forward.rs prefill()-> no f8 arm at all (examples/tests/ppl);
        //                          guarded loudly in forward.rs::prefill
        //   spec.rs forward_chunk -> had no f8 arm at all. This AUDIT did not
        //                          LIST spec.rs, and the fix above was then
        //                          verified on the nospec variant only, so
        //                          nothing caught it: on the default build
        //                          every `--spec auto` request died with
        //                          CUDA_ERROR_ILLEGAL_ADDRESS and the spec
        //                          lane ran at a quarter of its speed.
        //                          Now on `r > f8_ffn_pf_min()`.
        //   spec.rs record_spec_verify -> had an f8 arm, gated on a bare
        //                          `r >= 8`, so r < 8 read the stubs anyway.
        //                          Now on the same prefill floor.
        //   spec.rs mtp_block_pass{,_b} -> walk `self.mtp`, not `layers` --
        //                          drafter planes are never stubbed. Safe, and
        //                          checked rather than assumed.
        // Every remaining Q8 fallback arm that a stubbed plane can reach now
        // calls `stub_guard` first: this audit has been wrong twice, so the
        // next miss must refuse loudly instead of serving corruption.
        // The structural fix is one resident plane per tensor --
        // The REPLACE itself already happened, per layer, inside the backbone
        // loop. What is left here is the CHECK: `kq_early` was a conservative
        // pre-image of `kq_resident`, and if it was ever wrong in the unsafe
        // direction we have dropped planes the k-quant spine still reads.
        // Refuse the load rather than serve from freed memory.
        if ffn_replaced > 0 && kq_resident {
            return Err(GpuModelError::Unsupported(format!(
                "qwen35: the FFN REPLACE dropped {ffn_replaced} Q8_0 planes under \
                 kq_early=false, but the loaded model is k-quant resident. The \
                 pre-image missed a k-quant seat the file's tensor types did not \
                 name, and the kq spine reads Q8_0-class planes -- this is a \
                 LOADER BUG, not a config error. Re-run with \
                 PADDOCK_QWEN35_F8_FFN_PF_MIN=2 to keep the Q8_0 planes resident \
                 while it is fixed."
            )));
        }
        if ffn_replaced > 0 {
            tracing::info!(
                "qwen35 VRAM  q8-original REPLACE: {} dense FFN planes replaced -- {:.2} GB never \
                 allocated + {:.2} GB dropped at the point of conversion (every band serves e4m3)",
                ffn_replaced,
                gb(ffn_skipped_bytes),
                gb(ffn_freed)
            );
        }

        // Q8 HEAD RECLAIM - Not done, deliberately. The e4m3 head (1.31 GB)
        // and the Q8_0 head (1.35 GB) are still both resident, and the floor
        // unification above only moves the GATED sites. The head is read from
        // eight more places that call `gemv_any(&self.output, ..)` directly:
        // multimodal.rs:534, forward.rs:607/842, spec.rs:173/433,
        // dflash.rs:1602 and the guarded batch.rs fallbacks. Stubbing the plane
        // under them is an illegal access - measured, not hypothesised: the
        // first attempt booted fine, reported "1.35 GB returned", and then died
        // on the first request with CUDA_ERROR_ILLEGAL_ADDRESS from
        // prefix.rs's head site (now routed to f8, which is why that one is
        // fixed above). The remaining sites live on the spec, drafter and
        // vision paths, which a text smoke run never exercises, so they need
        // their own pass with each path actually driven - 1.35 GB does not
        // justify a latent illegal access in the spec lane.

        // return load staging (repack raw uploads) from the mempool to the OS
        // before the resident snapshot - pool-held frees read as "used" and
        // would both inflate this ledger and starve the sizers below
        exec.trim_mem_pool();
        let v_head = vfree();
        // The PUBLISHED resident-weight line. The vfree() ledger below is a
        // per-phase diagnostic and counts the CUDA context/modules/cuBLAS as
        // model bytes; the pool's own used counter does not, and reproduces
        // exactly between loads. Nothing below here allocates.
        let weights_bytes = exec.settled_mem_used();
        // Sampled right after settled_mem_used' trim, so the pair is one
        // consistent view: `used` is what is live, `reserved` is what the pool
        // holds from the driver. Their difference is the fragmentation the
        // trim could not return.
        let pool_reserved = exec.pool_reserved_bytes();
        tracing::info!(
            "qwen35 VRAM  output head + norms + aux planes     {:>7.2} GB",
            gb(v_back.saturating_sub(v_head))
        );
        tracing::info!(
            "qwen35 VRAM  = model resident total               {:>7.2} GB  (device free {:.2} GB)",
            gb(v_start.saturating_sub(v_head)),
            gb(v_head)
        );
        {
            // exact-bytes audit vs the free-VRAM ledger above. Two named
            // groups: the BASE planes (the checkpoint's own tensors) and the
            // DERIVED decode/batch-lane planes (f8-ffn, W8 projections, f8
            // head, fused concats - default-ON elections living outside the
            // base struct fields). The derived group used to go unsummed,
            // and its ~25 GiB on the 27B printed as
            // "allocator slack", which sent a whole VRAM investigation
            // toward heap granularity.
            // What remains as slack after both groups is genuine allocator
            // rounding - the 7-8% class the estimator models.
            let mut bb = AuSum::default();
            for l in &layers {
                bb.dt(&l.attn_norm);
                bb.dt(&l.post_norm);
                match &l.mixer {
                    Mixer::Full(w) => bb.attn(w),
                    Mixer::Linear(w) => bb.dn(w),
                }
                bb.ffn(&l.ffn);
            }
            let mut mt = AuSum::default();
            if let Some(m) = &mtp {
                mt.qw(&m.eh_proj);
                mt.dt(&m.enorm);
                mt.dt(&m.hnorm);
                mt.dt(&m.head_norm);
                mt.dt(&m.attn_norm);
                mt.dt(&m.post_norm);
                mt.attn(&m.attn);
                mt.ffn(&m.ffn);
            }
            let mut hd = AuSum::default();
            hd.qw(&output);
            hd.dt(&out_norm);
            hd.bytes += (sinks.len() * 4) as u64;
            hd.allocs += 1;
            // derived lanes, by group
            let mut dv_f8ffn = AuSum::default();
            for e in bs_f8ffn_planes
                .iter()
                .chain(bs_f8ffn_bs_planes.iter())
                .flatten()
            {
                dv_f8ffn.fp4(&e[0].0);
                dv_f8ffn.fp4(&e[1].0);
            }
            for p in bs_f8row_ffn_planes.iter().flatten() {
                for pl in [&p.gate, &p.up, &p.down] {
                    dv_f8ffn.bytes += (pl.data.len() + pl.scale.len() * 4) as u64;
                    dv_f8ffn.allocs += 2;
                }
            }
            let mut dv_proj = AuSum::default();
            for w in bs_w8.iter().chain(bs_nv4.iter()) {
                for p in [&w.wq, &w.wk, &w.wv, &w.wo, &w.in_qkv, &w.gate_w, &w.out_w]
                    .into_iter()
                    .flatten()
                {
                    dv_proj.fp4(p);
                }
            }
            for e in bs_f8t_attn_planes.iter().flatten() {
                for t in e {
                    dv_proj.bytes += (t.tiles.len()
                        + t.scale.len() * 4
                        + t.flat.as_ref().map_or(0, |f| f.len())
                        + t.scale_il.as_ref().map_or(0, |s| s.len() * 4))
                        as u64;
                    dv_proj.allocs += 2;
                }
            }
            let mut dv_fused = AuSum::default();
            for p in bs_gu_planes.iter().chain(bs_dn_planes.iter()).flatten() {
                dv_fused.q8(p);
            }
            let mut dv_head = AuSum::default();
            if let Some((p, _, _)) = &out_f8 {
                dv_head.fp4(p);
            }
            if let Some((t, _, _)) = &out_f8t {
                dv_head.bytes +=
                    (t.tiles.len() + t.scale.len() * 4 + t.flat.as_ref().map_or(0, |f| f.len()))
                        as u64;
                dv_head.allocs += 2;
            }
            let ctx = v_start.saturating_sub(v_embd).saturating_sub(embd_bytes);
            let planes = embd_bytes + bb.bytes + mt.bytes + hd.bytes;
            let derived = dv_f8ffn.bytes + dv_proj.bytes + dv_fused.bytes + dv_head.bytes;
            let ledger = v_start.saturating_sub(v_head);
            let n_allocs = bb.allocs
                + mt.allocs
                + hd.allocs
                + 1
                + dv_f8ffn.allocs
                + dv_proj.allocs
                + dv_fused.allocs
                + dv_head.allocs;
            tracing::info!(
                "qwen35 VRAM audit  exact planes: backbone {:.2} GB/{} allocs · \
                 mtp {:.2}/{} · head+norms {:.2}/{} · embd {:.2}",
                gb(bb.bytes),
                bb.allocs,
                gb(mt.bytes),
                mt.allocs,
                gb(hd.bytes),
                hd.allocs,
                gb(embd_bytes)
            );
            tracing::info!(
                "qwen35 VRAM audit  derived lanes: f8-ffn {:.2} GB/{} allocs · \
                 proj w8/nv4/f8t {:.2}/{} · fused gu|dn {:.2}/{} · f8 head {:.2}/{}",
                gb(dv_f8ffn.bytes),
                dv_f8ffn.allocs,
                gb(dv_proj.bytes),
                dv_proj.allocs,
                gb(dv_fused.bytes),
                dv_fused.allocs,
                gb(dv_head.bytes),
                dv_head.allocs
            );
            tracing::info!(
                "qwen35 VRAM audit  base {:.2} + derived {:.2} = {:.2} GB vs \
                 ledger-minus-ctx {:.2} GB -> allocator slack {:.2} GB across {} allocations",
                gb(planes),
                gb(derived),
                gb(planes + derived),
                gb(ledger.saturating_sub(ctx)),
                gb(ledger.saturating_sub(ctx).saturating_sub(planes + derived)),
                n_allocs
            );
            // The line above compares audited bytes against the free-VRAM
            // delta, and that delta is three things at once. Split them, or
            // "slack" keeps absorbing whatever nobody named - which is how
            // ~25 GiB of duplicate planes hid inside it for as long as they
            // did.
            //
            //   pool used      - live allocations. audited-vs-used is the only
            //                    honest test of whether the audit is COMPLETE:
            //                    a gap here is planes nobody counted.
            //   pool RESERVED  - what the pool holds from the driver. minus
            //                    used = fragmentation trim_to(0) could not
            //                    return (a retained block with one live
            //                    allocation in it stays).
            //   ledger-RESERVED- everything that is not a pool allocation:
            //                    CUDA context, lazily-loaded kernel modules,
            //                    cuBLAS workspaces. Irreducible for us.
            if let (Some(used), Some(reserved)) = (weights_bytes, pool_reserved) {
                let audited = planes + derived;
                tracing::info!(
                    "qwen35 VRAM audit  residency split: pool live {:.2} GB (audited {:.2} -> \
                     UNAUDITED {:.2}) · pool retained-not-live {:.2} GB · \
                     ctx/modules/cuBLAS {:.2} GB · ledger {:.2} GB",
                    gb(used),
                    gb(audited),
                    gb(used.saturating_sub(audited)),
                    gb(reserved.saturating_sub(used)),
                    gb(ledger.saturating_sub(reserved)),
                    gb(ledger)
                );
            }
        }
        if kq_resident {
            // "show which quant" product principle + honest serving-mode label
            tracing::info!(
                "qwen35: k-quant weights resident (Q4_K/Q5_K/Q6_K/IQ4_XS mix) - \
                 W4A8 serving on every width: dp4a GEMV single-stream decode \
                 (PADDOCK_KQ_EXACT_GEMV=1 pins the exact-f32 oracle GEMV), int8 \
                 dp4a/K-split-MMA batch ladders, decode pipe on, spec/MTP on \
                 the same rungs (needs the file to ship the nextn block - \
                 unsloth UD exports strip it); MoE expert seats k-quant-resident \
                 too (token-batched decode + sorted kq-mma prefill classes)"
            );
        }

        Ok(Self {
            exec,
            n_layers,
            embd,
            n_heads,
            n_kv_heads,
            head_dim,
            ff,
            moe,
            state_size,
            n_k_heads,
            n_v_heads,
            conv_k,
            value_dim,
            conv_dim,
            rms_eps,
            n_rot,
            sections,
            yarn_params,
            vocab,
            max_ctx,
            tok_embd,
            layers,
            bs_w8,
            bs_nv4,
            bs_gu: bs_gu_planes,
            bs_dn: bs_dn_planes,
            bs_f8ffn: bs_f8ffn_planes,
            bs_f8ffn_bs: bs_f8ffn_bs_planes,
            bs_f8t_ffn: bs_f8t_ffn_planes,
            bs_f8row_ffn: bs_f8row_ffn_planes,
            bs_f8t_attn: bs_f8t_attn_planes,
            out_f8,
            out_f8t,
            out_norm,
            output,
            kq_resident,
            kq_max_elems,
            mtp,
            weights_bytes,
            content_id: (
                crate::kv_tier::fingerprint::weights(map),
                crate::kv_tier::fingerprint::tokenizer(map),
            ),
            dflash: None,
            dflash_pending_append: None,
            spec_round_dflash: false,
            spec_rs_draws: None,
            spec_round_rs: false,
            sinks,
            kv_dtype: KvDtype::Fp16,
            overlap_exec: None,
            lane_swapped: false,
            unified_inflight: None,
            scratch: None,
            decode_arena: None,
            spec: None,
            batch: None,
            spec_batch: None,
            spec_pending: None,
            spec_chain: None,
            spec_warm_wanted: true,
            spec_ring_probed: false,
            spec_live_vram_cap: None,
            prefill_chunk_rows: super::chunk_tick_rows(),
            history: Vec::new(),
            decode: None,
            vision: None,
            last_reused: Vec::new(),
            chunked: Vec::new(),
            image_cache: Vec::new(),
            image_cache_clock: 0,
            image_cache_reused: 0,
            pipe: None,
        })
    }

    /// Measured device bytes this model holds (weights + KV/state pools) -
    /// see `GpuExecutor::process_mem_used`.
    pub fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    /// Resident weight bytes measured at load - see the `weights_bytes` field.
    pub fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_bytes
    }

    /// Context-state bytes for the memory-breakdown API: full-attn KV planes
    /// (batch or serial, whichever is live) + the MTP drafter's KV + the
    /// DeltaNet recurrent/conv state - the hybrid's linear-attention layers
    /// hold context in those f32 states instead of KV, so an honest
    /// "what does context cost" number must include them.
    pub fn kv_mem_bytes(&self) -> Option<u64> {
        let opt = |v: &[Option<cudarc::driver::CudaSlice<u8>>]| -> u64 {
            v.iter().flatten().map(|c| c.len() as u64).sum()
        };
        let optf = |v: &[Option<cudarc::driver::CudaSlice<f32>>]| -> u64 {
            v.iter().flatten().map(|c| (c.len() * 4) as u64).sum()
        };
        let mut total = 0u64;
        let mut any = false;
        if let Some(b) = self.batch.as_ref() {
            total += opt(&b.kv_k) + opt(&b.kv_v) + optf(&b.recur) + optf(&b.conv_win);
            any = true;
        } else if let Some(d) = self.decode.as_ref() {
            total += opt(&d.kv_k) + opt(&d.kv_v);
            total += d.mtp_kv_k.as_ref().map_or(0, |c| c.len() as u64);
            total += d.mtp_kv_v.as_ref().map_or(0, |c| c.len() as u64);
            any = true;
        }
        any.then_some(total)
    }

    /// Width-by-VRAM: clamp the requested continuous-batching width so the
    /// batched state enable_batch is about to allocate - plus the LAZY serving
    /// spec state that follows when PADDOCK_QWEN35_SPEC is set - fits the VRAM
    /// actually free after weights. The KV pool auto-sizer already shrinks the
    /// KV BUDGET to fit, but its per-slot floor (128 blocks/slot) and the
    /// non-KV per-slot state scale with width, and width 32 + a 29 GB Q8-27B +
    /// spec measured a hard OOM on 48 GB that dropped the server to the
    /// serial engine. Honest will-it-fit: width is the knob that gives.
    /// Estimation mirrors the pool sizer's `other` accounting (keep in sync);
    /// an under-estimate is caught by the service's halving retry. Clamping is
    /// loud, never silent; PADDOCK_NO_WIDTH_CLAMP=1 bypasses.
    pub(super) fn width_by_vram(&self, requested: usize) -> (usize, Option<usize>) {
        if requested <= 1 || paddock_models::dev_var_os!("PADDOCK_NO_WIDTH_CLAMP").is_some() {
            return (requested, None);
        }
        self.exec.trim_mem_pool(); // pool-held frees must not read as used
        // budget-aware headroom: None = driver gave no number (bypass, as
        // before); a real 0 now flows through and clamps to width 1 - under a
        // configured vram_budget "the card looks free" no longer means "ours"
        let Some(free) = self.exec.vram_headroom() else {
            return (requested, None); // no measurement - let allocation (and the retry) decide
        };
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let state_elems = self.n_v_heads * self.state_size * self.state_size;
        let win_elems = (self.conv_k - 1) * self.conv_dim;
        let n_lin = self.n_linear_layers() as u64;
        let n_full = (self.n_layers - self.n_linear_layers()) as u64;
        let bps = self.max_ctx.div_ceil(BLOCK_TOKENS) as u64;
        let per_block = (BLOCK_TOKENS * kv_dim * kv_bytes) as u64 * 2 * n_full;
        let state_win = (state_elems + win_elems) as u64;
        // width-independent: conv staging, the demand-sized prefix-state pool
        // (STATE_CKPTS_PER_SLOT per requested slot, clamped - the plan may
        // shrink it further into what the grant has left), ckpt staging, the
        // wave buffers for a full chunk, and the graph/headroom residual. The
        // profiled serving scratch is already in the ledger by the time `free`
        // is read (enable_batch_sized allocates it first).
        let ckpt_pool = (requested as u64 * super::batch::STATE_CKPTS_PER_SLOT)
            .clamp(super::batch::STATE_CKPTS_MIN, super::batch::STATE_CKPTS_MAX)
            * n_lin
            * state_win
            * 4;
        let chunk = super::chunk_tick_rows();
        let fixed: u64 = 2
            * (self.conv_k as u64 - 1 + unified_prefill_rows().max(8192) as u64)
            * self.conv_dim as u64
            * 4
            + ckpt_pool
            + 2 * n_lin * state_win * 4
            + self.pf_bufs_bytes(chunk, chunk, requested)
            + super::batch::graph_pools_headroom_bytes();
        // per slot: recurrent+conv state, the pool's per-slot block floor,
        // a batched logits row, and the block-table row
        let per_slot: u64 =
            n_lin * state_win * 4 + bps.min(128) * per_block + self.vocab as u64 * 4 + bps * 4;
        // lazy serving-spec state (ensure_serve_spec): dense MTP KV at max_ctx
        // per live slot dominates; live is capped by both the env knob and mb.
        // The 35B-Q8-on-48GB case: the live-8 reservation is
        // WIDTH-INDEPENDENT and clamped a 32-wide serve down to 6 - so the
        // sizer now degrades SPEC live before width: spec at live 2 still
        // covers the c1/c2 sessions where it pays, while width carries the
        // serving aggregate. Explicit PADDOCK_QWEN35_SPEC_LIVE_MAX always wins.
        let spec_cost = |mb: usize, live_cap: usize| -> u64 {
            if !self.serve_spec_on() || live_cap == 0 {
                return 0;
            }
            let live = live_cap.clamp(1, mb) as u64;
            let k1 = (self.serve_spec_k() + 1) as u64;
            // Verify-state term mirrors batch.rs spec_est (same condition as
            // the enable_spec_batch alloc): legacy snapshots are the full
            // state per draft position (~1.1 GB/live-row at 27B - this term
            // alone clamped the 32-row spec serve to 16); the snapshot-free
            // path (dflash) stashes only k_hat/v/g/beta, ~state_size
            // times smaller.
            let verify_state = if self.spec_snapshot_verify() {
                n_lin * k1 * state_elems as u64 * 4
            } else {
                n_lin
                    * k1
                    * 2
                    * ((state_elems / self.state_size) as u64 + self.n_v_heads as u64)
                    * 4
            };
            live * (2 * self.max_ctx as u64 * kv_dim as u64 * kv_bytes as u64
                + k1 * self.vocab as u64 * 4
                + verify_state
                + n_lin * (self.conv_k as u64 - 1 + k1) * self.conv_dim as u64 * 4)
        };
        let width_at = |live_cap: usize| -> usize {
            let mut mb = requested;
            while mb > 1 {
                let est = fixed + spec_cost(mb, live_cap) + mb as u64 * per_slot;
                if est <= free {
                    break;
                }
                mb -= 1;
            }
            mb
        };
        let live_pref = self.serve_spec_live_max();
        let mut mb = width_at(live_pref);
        let mut live = live_pref;
        if mb < requested
            && self.serve_spec_on()
            && paddock_models::dev_var_os!("PADDOCK_QWEN35_SPEC_LIVE_MAX").is_none()
        {
            let mut l = live_pref;
            while l > 1 {
                l /= 2;
                let w = width_at(l);
                if w > mb {
                    mb = w;
                    live = l;
                }
                if w >= requested {
                    break;
                }
            }
        }
        if live < live_pref {
            tracing::info!(
                "qwen35 width-by-VRAM: serving-spec live cap {live_pref} -> {live} \
                 (frees the width-independent draft state for batch width; \
                 PADDOCK_QWEN35_SPEC_LIVE_MAX pins it)."
            );
        }
        if mb < requested {
            tracing::info!(
                "qwen35 width-by-VRAM: max_batch {requested} -> {mb} ({:.1} GB free after \
                 weights; ~{:.2} GB/slot batched state, {:.1} GB width-independent{}). \
                 PADDOCK_NO_WIDTH_CLAMP=1 forces the requested width.",
                free as f64 / 1e9,
                per_slot as f64 / 1e9,
                (fixed + spec_cost(mb, live)) as f64 / 1e9,
                if self.serve_spec_on() {
                    " incl. serving-spec estimate"
                } else {
                    ""
                },
            );
        }
        (mb, if live < live_pref { Some(live) } else { None })
    }
}
