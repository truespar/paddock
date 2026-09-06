//! KV admission planning - the one arbiter for "how much KV may this server
//! hold on this card".
//!
//! Every family used to answer that itself, and it grew into ten
//! hand-rolled solvers of the same arithmetic: seven copies of a flat 1 GiB
//! `VRAM_HEADROOM`, two of a 3 GiB graph margin, a separate 1.5 GiB one in
//! qwen35's width election, three different answers to "what do we do when it
//! does not fit" (refuse / narrow / take it anyway), and - the reason this
//! module exists - two arms that never consulted the grant at all.
//!
//! That last one shipped: a Qwen3.8-27B server configured `vram_budget =
//! 30720` (30 GiB), `max_batch = 1`, `max_ctx = 131072` logged its budget
//! correctly and then put ~41 GB on the card. `max_batch <= 1` skipped
//! qwen35's pool sizer, and the dense fallback beside it reserved a flat
//! `max_batch × max_ctx × kv_dim × kv_bytes` per layer with no reference to
//! `vram_headroom()`. The budget was not exceeded by a bad estimate; it was
//! never read.
//!
//! So the fix is not another check in that arm. It is that **there is no arm
//! that skips the arbiter**: a family states its geometry, this module returns
//! a plan or a refusal, and dense allocation stops being a separate concept -
//! it is simply a plan whose block count reached the addressable ceiling.
//!
//! # The model
//!
//! ```text
//!   grant  =  vram_headroom()            what this runner may still take:
//!                                        its budget minus its own ledger,
//!                                        clamped to what the device has free
//!
//!   grant - Σ reserves - slots × per_slot_bytes  =  bytes for the KV pool
//!
//!   pool_blocks = that / block_bytes, clamped to
//!       ceiling  slots × blocks_per_slot + retention   (nothing above is
//!                                                       addressable - a slot's
//!                                                       block table holds
//!                                                       exactly bps entries)
//!       floor    every slot able to hold one prefill chunk, or admission
//!                deadlocks on its own first chunk
//! ```
//!
//! Reserves are *named* because a refusal has to say which term ate the card.
//! "no silent failures" is not only about erroring - a server that quietly
//! seats 6 of the 32 slots you asked for has failed silently too, so a plan
//! that came out narrower than the ask reports itself at WARN with every term
//! spelled out.
//!
//! # What this module deliberately does not decide
//!
//! Whether a slot without room for a full `max_ctx` is worth seating is a
//! *policy* question and the families genuinely disagree, so it is [`WhenShort`]
//! rather than a constant here. Paged serving's whole point is that N sequences
//! share a budget and rarely reach the window at once - vLLM sizes KV from a
//! utilization fraction and treats `max_model_len` as a ceiling, admitting and
//! preempting against the pool. qwen35/gpt-oss instead refuse, because a pool
//! that silently backed 23 of 32 slots produced 152 s TTFT tails with nothing
//! on screen to attribute them to. Both readings are defensible; is
//! where the default gets decided. Keeping it one field on one struct is what
//! makes that a decision rather than an archaeology exercise.

use crate::kv_pool::BLOCK_TOKENS;

/// One named charge against the grant.
///
/// The name is printed in the plan log and in any refusal, so write it for the
/// operator reading a startup failure - "prefill scratch", not "scratch_est".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reserve {
    pub what: &'static str,
    pub bytes: u64,
}

impl Reserve {
    pub fn new(what: &'static str, bytes: u64) -> Self {
        Self { what, bytes }
    }
}
/// The operator's graph/scratch reserve override (`graph_scratch_mib`).
///
/// gpt-oss still charges a fixed 3 GiB "graph/prefill scratch" by default and
/// this overrides it. qwen35 profiles its prefill scratch at load and computes
/// its wave buffers, so there the override applies only to the small residual
/// it still names (`graph pools + headroom`). The fixed default was sized for
/// 16-48 GB cards and starves 8 GB cards of both KV and - since
/// `[moe_offload]` - the slot cache the plan's leftovers seat.
static GRAPH_SCRATCH_MIB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
/// Arm the override once before any model loads (the same shape as
/// `pool_tier::set_tier_ram_bytes` / `gpu::set_moe_offload`). `None` = keep
/// each family's default.
pub fn set_graph_scratch_mib(mib: u64) {
    let _ = GRAPH_SCRATCH_MIB.set(mib);
}
/// The armed override, if any, in MiB.
pub fn graph_scratch_override_mib() -> Option<u64> {
    GRAPH_SCRATCH_MIB.get().copied()
}
/// The armed override, or the 3 GiB default (gpt-oss's `graph/prefill
/// scratch` reserve).
pub fn graph_scratch_reserve_bytes() -> u64 {
    graph_scratch_override_mib().map_or(3 * (1 << 30), |m| m << 20)
}

/// What to do when the grant cannot back a full `max_ctx` for every slot asked
/// for. See the module note - this is the open policy question, not a detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhenShort {
    /// Seat fewer slots and say so. The pool still gets whatever is left, so
    /// the seated slots each keep full-context backing. gemma4/laguna's
    /// long-standing behaviour, and granite/nemotron's (which ask for a floor
    /// rather than full context, so they narrow only in the sense of refusing
    /// below it).
    Narrow,
    /// Keep the asked-for width or refuse the whole config. qwen35/gpt-oss:
    /// their block tables are sized for full context per slot, and a pool that
    /// cannot back it queues the difference invisibly.
    Refuse,
}

/// A family's KV geometry. Everything is bytes or blocks - no family-specific
/// concepts reach this module, which is what lets one solver serve all of them.
#[derive(Debug, Clone)]
pub struct Demand {
    /// Family name, for logs and refusals.
    pub family: &'static str,
    /// Context window this server was configured for, for messages.
    pub max_ctx: usize,
    /// Slots asked for (`--max-batch`).
    pub slots: usize,
    /// Blocks one slot needs to hold `max_ctx` - `max_ctx.div_ceil(BLOCK_TOKENS)`.
    pub blocks_per_slot: usize,
    /// Bytes one pool block costs across every layer that draws from the pool.
    /// One block id addresses all of them, so this is the whole-model cost.
    pub block_bytes: u64,
    /// Bytes one slot costs that the pool cannot share: SWA rings, recurrent
    /// state, conv windows, its logits row, its block table. Scales with the
    /// seated slot count, which is why narrowing buys anything.
    pub per_slot_bytes: u64,
    /// Blocks the radix tree may hold above the addressable ceiling (nodes keep
    /// blocks after their sequence ends). 0 when the prefix cache is off.
    pub retention_blocks: usize,
    /// Blocks each slot must be able to hold before admission can make
    /// progress - one prefill chunk, or a sequence deadlocks on its own first
    /// chunk.
    pub floor_blocks_per_slot: usize,
    /// Absolute block floor regardless of width.
    pub floor_blocks_min: usize,
    /// Fixed charges that do not scale with slots.
    pub reserves: Vec<Reserve>,
    /// See [`WhenShort`].
    pub when_short: WhenShort,
    /// Cap the pool at this fraction of the grant.
    ///
    /// A hedge for families whose `reserves` are hand-enumerated rather than
    /// derived, so an omission costs a smaller pool instead of an OOM into the
    /// serial fallback. Measured on 27B-Q4: honestly-enumerated reserves still
    /// budgeted an 11.8 GB pool, the lazily-allocated spec state then pushed
    /// past free, and c1 collapsed 74 -> 31 t/s. Families that enumerate
    /// completely pass `None`, and the goal is for every family to - qwen35
    /// got there 2026-09-06 (profiled scratch, computed wave buffers, a
    /// measured residual, and a ledger audit after allocation); gpt-oss still
    /// hedges.
    pub hedge_fraction: Option<f64>,
}

impl Default for Demand {
    fn default() -> Self {
        Self {
            family: "model",
            max_ctx: 0,
            slots: 1,
            blocks_per_slot: 0,
            block_bytes: 0,
            per_slot_bytes: 0,
            retention_blocks: 0,
            floor_blocks_per_slot: 0,
            floor_blocks_min: 0,
            reserves: Vec::new(),
            when_short: WhenShort::Narrow,
            hedge_fraction: None,
        }
    }
}

/// What the arbiter decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// Slots to actually seat. `<= Demand::slots`; the difference is reported.
    pub slots: usize,
    /// Blocks to allocate in the shared pool.
    pub pool_blocks: usize,
    /// Bytes those blocks cost.
    pub pool_bytes: u64,
    /// Bytes the seated slots' non-poolable state costs.
    pub slot_bytes: u64,
    /// Tokens the pool holds across all sequences at once. The number
    /// asks to surface: with paged KV this, not `max_ctx`, is what bounds
    /// concurrent work.
    pub token_capacity: usize,
    /// The pool cannot back `max_ctx` for every seated slot simultaneously, so
    /// long sequences share the budget and admission may queue or preempt.
    /// Never true under [`WhenShort::Refuse`] - those refuse instead.
    pub shared: bool,
}

/// A config the engine can prove infeasible, with the arithmetic that proves it.
#[derive(Debug, Clone)]
pub struct WontFit {
    pub message: String,
}

impl std::fmt::Display for WontFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// Upper-case the first letter, so a fix list reads as a sentence when it
/// leads one ("lower max_ctx..." -> "Lower max_ctx...").
fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

impl Demand {
    /// Blocks addressable at this width. A slot's block table holds exactly
    /// `blocks_per_slot` entries, so anything above this is VRAM taken off the
    /// desktop that no sequence can ever reach.
    fn ceiling(&self, slots: usize) -> u64 {
        slots as u64 * self.blocks_per_slot as u64 + self.retention_blocks as u64
    }

    /// Blocks needed before admission can make progress at this width, never
    /// more than are addressable.
    fn floor(&self, slots: usize) -> u64 {
        (slots as u64 * self.floor_blocks_per_slot as u64)
            .max(self.floor_blocks_min as u64)
            .min(self.ceiling(slots))
    }

    /// Blocks needed for every seated slot to hold `max_ctx` at once.
    fn full_context(&self, slots: usize) -> u64 {
        slots as u64 * self.blocks_per_slot as u64
    }

    fn fixed(&self) -> u64 {
        self.reserves.iter().map(|r| r.bytes).sum()
    }

    /// Blocks the grant affords at this width, before the floor is consulted.
    /// `None` when the fixed + per-slot charges already exceed the grant.
    fn affordable(&self, grant: u64, slots: usize) -> Option<u64> {
        let charged = self
            .fixed()
            .saturating_add(slots as u64 * self.per_slot_bytes);
        let kv = grant.checked_sub(charged)?;
        let per_block = self.block_bytes.max(1);
        let mut blocks = kv / per_block;
        if let Some(f) = self.hedge_fraction {
            blocks = blocks.min((grant as f64 * f) as u64 / per_block);
        }
        Some(blocks.min(self.ceiling(slots)))
    }

    /// Solve. This is the only way a family may size KV.
    ///
    /// `grant` is [`crate::gpu::Exec::vram_headroom`] - the budget minus this
    /// runner's own ledger, clamped to device free. Callers must treat a
    /// missing reading as an error rather than as permission: "the driver did
    /// not answer" is not "take what you like", and reading it that way is
    /// exactly how 30 GiB became 41.
    pub fn plan(&self, grant: u64) -> Result<Plan, WontFit> {
        // Under Refuse the width is not ours to move - the family wants the
        // config it was given or an actionable failure, not a quiet demotion.
        let narrowest = match self.when_short {
            WhenShort::Narrow => 1,
            WhenShort::Refuse => self.slots,
        };
        for slots in (narrowest..=self.slots).rev() {
            let Some(blocks) = self.affordable(grant, slots) else {
                continue;
            };
            if blocks < self.floor(slots) {
                continue;
            }
            let full = self.full_context(slots);
            if self.when_short == WhenShort::Refuse && blocks < full {
                continue;
            }
            return Ok(Plan {
                slots,
                pool_blocks: blocks as usize,
                pool_bytes: blocks * self.block_bytes,
                slot_bytes: slots as u64 * self.per_slot_bytes,
                token_capacity: blocks as usize * BLOCK_TOKENS,
                shared: blocks < full,
            });
        }
        Err(self.wont_fit(grant))
    }

    /// Plan, or - when nothing fits - the smallest runnable shape instead of a
    /// refusal: one slot at the progress floor.
    ///
    /// For the caller that has already freed the cache it would fall back to.
    /// gemma4 hands its load-time serial planes back before measuring (free
    /// then measure needs no correction term and cannot drift from what the
    /// allocator really did), so by the time it plans there is nothing to
    /// return to - an `Err` there would leave the model with no KV at all, and
    /// the service's width backstop would decode against nothing. It takes this
    /// instead, the allocation fails honestly, and its restore path rebuilds
    /// the 1-slot serial shape.
    ///
    /// Still a BOUNDED shape - one slot, the floor - never the
    /// `max_batch × max_ctx` reservation this module exists to delete. Reach
    /// for [`Demand::plan`] everywhere else: a refusal the operator can read is
    /// worth more than a server that starts and then cannot serve.
    pub fn plan_or_minimum(&self, grant: u64) -> Plan {
        self.plan(grant).unwrap_or_else(|e| {
            tracing::warn!(
                family = self.family,
                "{e} - trying one slot and letting the allocator have the last word"
            );
            let blocks = self.floor(1);
            Plan {
                slots: 1,
                pool_blocks: blocks as usize,
                pool_bytes: blocks * self.block_bytes,
                slot_bytes: self.per_slot_bytes,
                token_capacity: blocks as usize * BLOCK_TOKENS,
                shared: blocks < self.full_context(1),
            }
        })
    }

    /// The refusal, with every term that produced it and the fixes that would
    /// clear it.
    fn wont_fit(&self, grant: u64) -> WontFit {
        let asked = self.slots;
        let got = self.affordable(grant, asked).unwrap_or(0);
        let need = match self.when_short {
            WhenShort::Refuse => self.full_context(asked),
            WhenShort::Narrow => self.floor(asked),
        };
        let per_block = self.block_bytes.max(1);

        let mut terms = vec![format!("grant {:.2} GiB", gib(grant))];
        for r in &self.reserves {
            terms.push(format!("{} {:.2}", r.what, gib(r.bytes)));
        }
        if self.per_slot_bytes > 0 {
            terms.push(format!(
                "{asked} x per-slot state {:.2}",
                gib(asked as u64 * self.per_slot_bytes)
            ));
        }

        // What would work. Both are honest divisions of the same budget, so
        // offer whichever ones are actually reachable rather than a generic
        // "lower something".
        let mut fixes = Vec::new();
        let fit_slots = (1..=asked).rev().find(|&n| {
            self.affordable(grant, n)
                .is_some_and(|b| b >= self.floor(n))
        });
        let fit_ctx = (got as usize).checked_div(asked).unwrap_or(0) * BLOCK_TOKENS;
        if fit_ctx >= BLOCK_TOKENS && fit_ctx < self.max_ctx {
            fixes.push(format!("lower max_ctx to <={fit_ctx}"));
        }
        match fit_slots {
            Some(n) if n < asked => fixes.push(format!("lower max_batch to <={n}")),
            _ => {}
        }
        fixes.push("raise vram_budget, or free VRAM on this card".into());

        // The dev switch does not join `fixes`: those LEAD the message and are
        // what a person reads in a toast or an inline row, and `PADDOCK_*` is
        // compiled out of shipped builds and sealed out of the environment at
        // startup - advice the reader usually cannot take, occupying the one
        // sentence guaranteed to be seen (including it pushed the headline to
        // 272 characters). It rides after the ledger for
        // whoever is holding a dev build. Offered only when the pool could
        // actually run the width: oversubscription trades a full-context
        // guarantee for queueing, which is useless if we cannot even seat the
        // progress floor.
        let dev_hint = if !paddock_models::hardening::HARDENED
            && self.when_short == WhenShort::Refuse
            && got >= self.floor(asked)
        {
            " PADDOCK_KV_OVERSUBSCRIBE=1 serves a shared paged KV budget instead \
             (requests beyond it queue or preempt)."
        } else {
            ""
        };

        // One line, and the ANSWER first.
        //
        // This message is read in three places with three different budgets: a
        // toast (a couple of lines), the inline row on the endpoint page, and
        // the log. It cannot contain a blank line - the runner writes it as a
        // single log record, and the manager recovers the reason by taking the
        // last non-empty LINE of the tail, so a paragraph break would hand the
        // UI the arithmetic and drop the sentence. So: what failed and what to
        // do about it come first and end on a sentence boundary, the ledger
        // follows for whoever is diagnosing, and the caller truncates.
        // (The first version led with the ledger and the toast was way too
        // long - correct, and unreadable, which is the same bug the timestamp
        // preamble had.)
        WontFit {
            message: format!(
                "{} cannot serve max_ctx {} x max_batch {asked}: needs {:.2} GiB of KV, \
                 only {:.2} GiB fits. {}. \
                 Budget: {} => {:.2} GiB for KV at {:.2} MiB/block \
                 ({got} of {need} blocks, {} tokens shared across all sequences).{}",
                self.family,
                self.max_ctx,
                gib(need * per_block),
                gib(got * per_block),
                capitalize(&fixes.join(", or ")),
                terms.join(" - "),
                gib(got * per_block),
                per_block as f64 / (1u64 << 20) as f64,
                got as usize * BLOCK_TOKENS,
                dev_hint,
            ),
        }
    }
}

impl Plan {
    /// Say what was decided and where the grant went.
    ///
    /// At INFO when the ask was met whole; at WARN when it was not, because a
    /// server quietly narrower than its config is the failure mode this whole
    /// module exists to stop being invisible.
    pub fn report(&self, d: &Demand, grant: u64) {
        let terms = d
            .reserves
            .iter()
            .map(|r| format!("{} {:.2}", r.what, gib(r.bytes)))
            .collect::<Vec<_>>()
            .join(", ");
        if self.slots < d.slots || self.shared {
            tracing::warn!(
                family = d.family,
                asked_slots = d.slots,
                seated_slots = self.slots,
                max_ctx = d.max_ctx,
                grant_gib = gib(grant),
                pool_gib = gib(self.pool_bytes),
                slot_state_gib = gib(self.slot_bytes),
                token_capacity = self.token_capacity,
                full_context_tokens = d.blocks_per_slot * BLOCK_TOKENS * self.slots,
                reserves = %terms,
                "KV plan is narrower than the configuration asked for"
            );
        } else {
            tracing::info!(
                family = d.family,
                slots = self.slots,
                max_ctx = d.max_ctx,
                grant_gib = gib(grant),
                pool_gib = gib(self.pool_bytes),
                slot_state_gib = gib(self.slot_bytes),
                pool_blocks = self.pool_blocks,
                token_capacity = self.token_capacity,
                reserves = %terms,
                "KV plan"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A family shaped like granite: one pool, no per-slot state.
    fn pooled() -> Demand {
        Demand {
            family: "test",
            max_ctx: 8192,
            slots: 4,
            blocks_per_slot: 512,
            block_bytes: 4 << 20, // 4 MiB/block-set, granite-30b's real figure
            floor_blocks_per_slot: 64,
            floor_blocks_min: 256,
            reserves: vec![Reserve::new("slack", GIB), Reserve::new("scratch", GIB)],
            ..Default::default()
        }
    }

    #[test]
    fn the_grant_bounds_the_pool() {
        // 10 GiB grant, 2 GiB reserved => 8 GiB of KV => 2048 blocks, and the
        // ceiling (4 x 512) is exactly 2048, so it lands on the ceiling.
        let d = pooled();
        let p = d.plan(10 * GIB).unwrap();
        assert_eq!(p.pool_blocks, 2048);
        assert_eq!(p.pool_bytes, 8 * GIB);
        assert!(!p.shared);
    }

    #[test]
    fn nothing_above_the_addressable_ceiling_is_taken() {
        // A card with room to spare must not grow the pool past what a slot's
        // block table can even address - that VRAM could never be reached.
        let d = pooled();
        let p = d.plan(200 * GIB).unwrap();
        assert_eq!(p.pool_blocks, 2048, "capped at slots x blocks_per_slot");
        assert_eq!(p.token_capacity, 2048 * BLOCK_TOKENS);
    }

    /// The regression that produced this module. A single-slot server with a
    /// 131072 window and a 30 GiB budget must be bounded by the budget, not by
    /// `max_batch x max_ctx`.
    #[test]
    fn a_single_slot_server_is_bounded_by_its_budget() {
        let d = Demand {
            family: "qwen35",
            max_ctx: 131072,
            slots: 1,
            blocks_per_slot: 131072 / BLOCK_TOKENS,
            block_bytes: 2 << 20,
            floor_blocks_per_slot: 128,
            reserves: vec![Reserve::new("graph/scratch", 3 * GIB)],
            ..Default::default()
        };
        // Model resident: the grant is what is left of a 30 GiB budget.
        let grant = 9 * GIB;
        let p = d.plan(grant).unwrap();
        assert!(
            p.pool_bytes <= grant,
            "pool {} exceeded the grant {grant}",
            p.pool_bytes
        );
        // Dense would have reserved the full window regardless: 8192 blocks x
        // 2 MiB = 16 GiB against a 9 GiB grant. That is the 41-GB-on-a-30-GiB-
        // budget bug, in one assertion.
        assert!(p.pool_blocks < d.blocks_per_slot);
        assert!(p.shared, "a shared budget, honestly reported as one");
    }

    /// The live case, with the numbers the runner actually measured:
    /// Qwen3.8-27B UD-Q4_K_XL, `vram_budget = 30720`, max_batch 1,
    /// max_ctx 131072, spec on. Before the planner this configuration reported
    /// its 30.0 GiB budget and then put ~41 GB on the card.
    #[test]
    fn the_qwen38_27b_overrun_refuses_instead_of_allocating() {
        let mib = |n: f64| (n * (1u64 << 20) as f64) as u64;
        let d = Demand {
            family: "qwen35",
            max_ctx: 131072,
            slots: 1,
            blocks_per_slot: 131072 / BLOCK_TOKENS, // 8192
            block_bytes: 1 << 20,                   // 1.00 MiB/block, measured
            per_slot_bytes: mib(150.0),
            floor_blocks_per_slot: 128,
            reserves: vec![
                Reserve::new("conv staging", mib(645.0)),
                Reserve::new("prefix state pool", mib(2396.0)),
                Reserve::new("checkpoint staging", mib(297.0)),
                Reserve::new("draft state (spec)", mib(1106.0)),
                Reserve::new("prefill wave buffers", mib(1160.0)),
                Reserve::new("graph pools + headroom", mib(768.0)),
            ],
            when_short: WhenShort::Refuse,
            // qwen35 no longer hedges: the terms above are the enumeration.
            hedge_fraction: None,
            ..Default::default()
        };
        // What was left of the 30 GiB budget once the weights AND the profiled
        // serving scratch were resident (the scratch precedes the grant now).
        let grant = mib(11510.0);
        let e = d.plan(grant).unwrap_err();
        // It must refuse rather than reserve the full 8 GiB window...
        assert!(
            e.message.contains("cannot serve max_ctx 131072"),
            "{}",
            e.message
        );
        // ...say what would fit, in tokens...
        assert!(e.message.contains("tokens shared"), "{}", e.message);
        // ...and name a fix the operator can act on. Case-insensitive: the fix
        // list leads a sentence now, so it is capitalised.
        let lower = e.message.to_lowercase();
        assert!(lower.contains("lower max_ctx"), "{}", e.message);
        assert!(lower.contains("vram_budget"), "{}", e.message);
        // The ANSWER and the FIX come before the ledger, and end on a sentence
        // boundary a toast can cut at - a reader with two lines has to get the
        // whole point (the first version led with the arithmetic and was far
        // too long for both the toast and the inline row).
        let head = e
            .message
            .split_once(". Budget:")
            .expect("ledger follows the answer")
            .0;
        assert!(head.len() < 200, "headline is {} chars: {head}", head.len());
        assert!(
            head.to_lowercase().contains("lower max_ctx"),
            "no fix in the headline: {head}"
        );

        // The same server at a context the budget can actually back starts, and
        // starts inside the grant - the whole point.
        let ok = Demand {
            max_ctx: 49152,
            blocks_per_slot: 49152 / BLOCK_TOKENS,
            ..d.clone()
        };
        let p = ok.plan(grant).expect("48k must fit where 128k does not");
        assert!(p.pool_bytes + p.slot_bytes + ok.fixed() <= grant);
        assert!(!p.shared, "a full-context plan, not an oversubscribed one");
    }

    #[test]
    fn narrowing_seats_fewer_slots_rather_than_failing() {
        let d = Demand {
            per_slot_bytes: 2 * GIB, // a fat per-slot ring, gemma4-shaped
            ..pooled()
        };
        // 4 slots would need 8 GiB of ring alone on top of 2 GiB of reserves.
        let p = d.plan(9 * GIB).unwrap();
        assert!(p.slots < 4, "seated {} of 4", p.slots);
        assert!(p.slot_bytes + p.pool_bytes <= 9 * GIB);
    }

    #[test]
    fn refuse_does_not_quietly_demote_the_width() {
        let d = Demand {
            when_short: WhenShort::Refuse,
            ..pooled()
        };
        // Enough for 1 slot's full context, nowhere near 4.
        let e = d.plan(3 * GIB).unwrap_err();
        assert!(e.message.contains("max_batch"), "{}", e.message);
        // and it must not have silently served a narrower server instead
        assert!(d.plan(3 * GIB).is_err());
    }

    #[test]
    fn a_refusal_names_the_term_that_ate_the_card() {
        let d = Demand {
            when_short: WhenShort::Refuse,
            reserves: vec![
                Reserve::new("prefill scratch", 6 * GIB),
                Reserve::new("prefix checkpoints", GIB),
            ],
            ..pooled()
        };
        let e = d.plan(8 * GIB).unwrap_err();
        assert!(e.message.contains("prefill scratch 6.00"), "{}", e.message);
        assert!(
            e.message.contains("prefix checkpoints 1.00"),
            "{}",
            e.message
        );
        assert!(e.message.contains("grant 8.00"), "{}", e.message);
    }

    #[test]
    fn a_grant_that_cannot_meet_the_progress_floor_refuses() {
        let d = pooled();
        // 2 GiB of reserves and nothing left over.
        let e = d.plan(2 * GIB).unwrap_err();
        assert!(e.message.contains("cannot serve"), "{}", e.message);
        // Never a plan for zero blocks: admission would deadlock on its own
        // first chunk and the server would look hung rather than refused.
        assert!(d.plan(2 * GIB).is_err());
    }

    #[test]
    fn the_floor_is_a_requirement_not_a_raise() {
        // qwen35 used to apply its floor as `.max(slots * 128)` after every
        // budget clamp, so the floor could allocate straight past the grant -
        // the same defect class as the dense arm, one order smaller.
        let d = Demand {
            floor_blocks_per_slot: 4096,
            floor_blocks_min: 0,
            ..pooled()
        };
        let grant = 3 * GIB; // 1 GiB of KV after reserves = 256 blocks
        if let Ok(p) = d.plan(grant) {
            assert!(
                p.pool_bytes + p.slot_bytes <= grant,
                "floor allocated {} past a {grant} grant",
                p.pool_bytes
            )
        }
    }

    #[test]
    fn retention_is_addressable_and_the_floor_never_exceeds_it() {
        let d = Demand {
            retention_blocks: 512,
            ..pooled()
        };
        let p = d.plan(200 * GIB).unwrap();
        assert_eq!(p.pool_blocks, 2048 + 512);
        // and a floor larger than the ceiling cannot make a config unservable
        let tight = Demand {
            floor_blocks_min: 1 << 20,
            ..pooled()
        };
        assert!(tight.plan(200 * GIB).is_ok());
    }

    #[test]
    fn the_hedge_caps_the_damage_of_an_omitted_reserve() {
        let d = Demand {
            hedge_fraction: Some(0.4),
            reserves: vec![],
            ..pooled()
        };
        let p = d.plan(20 * GIB).unwrap();
        assert!(
            p.pool_bytes <= 8 * GIB,
            "hedge should cap at 40% of the grant, got {}",
            gib(p.pool_bytes)
        );
    }

    #[test]
    fn every_plan_fits_inside_its_grant() {
        // The invariant the whole module exists for, over a spread of shapes.
        for slots in [1usize, 2, 8, 32] {
            for ctx in [4096usize, 32768, 131072] {
                for grant_gib in [1u64, 4, 9, 24, 48] {
                    let d = Demand {
                        family: "sweep",
                        max_ctx: ctx,
                        slots,
                        blocks_per_slot: ctx / BLOCK_TOKENS,
                        block_bytes: 1 << 20,
                        per_slot_bytes: 128 << 20,
                        floor_blocks_per_slot: 64,
                        floor_blocks_min: 256,
                        reserves: vec![Reserve::new("slack", GIB)],
                        ..Default::default()
                    };
                    let grant = grant_gib * GIB;
                    if let Ok(p) = d.plan(grant) {
                        assert!(
                            p.pool_bytes + p.slot_bytes + d.fixed() <= grant,
                            "slots {slots} ctx {ctx} grant {grant_gib} GiB: \
                             pool {:.2} + slots {:.2} + reserves {:.2} > grant",
                            gib(p.pool_bytes),
                            gib(p.slot_bytes),
                            gib(d.fixed()),
                        );
                        assert!(p.slots >= 1 && p.slots <= slots);
                    }
                }
            }
        }
    }
}
