//! What reasoning control the served checkpoint actually implements -
//! MEASURED from its own chat template at load, not assumed from its family.
//!
//! Why this is not a table. Qwen3.8-27B grades
//! reasoning effort at low/medium/xhigh; Qwen3.6-27B and Qwen3.5-9B have only
//! an on/off `enable_thinking`. All three report `general.architecture =
//! qwen35` and all three parse as `Dialect::QwenXml`, so neither of the two
//! identities the runner keys on can tell them apart. Anything hand-written -
//! an arch row, a dialect arm - is guaranteed to be wrong for one of them, and
//! it was: every 3.8 request collapsed to `enable_thinking: true` and the
//! ladder was unreachable.
//!
//! The template is the contract, so the template is what we ask. Templates come
//! in two shapes and the probe distinguishes them by rendering a sentinel value
//! no real ladder contains:
//!
//! - **Validating** (Qwen3.8): the template enumerates what it accepts and
//!   `raise_exception`s on anything else. The sentinel raises, so the accepted
//!   set is knowable exactly - render each candidate level and keep the ones
//!   that survive. This is the authority: it is the shipped artifact's own
//!   declaration, and it cannot drift from the file we are serving.
//! - **Interpolating** (gpt-oss, Muse Glimmer): the value goes straight into
//!   the prompt text (`"Reasoning: " + reasoning_effort`), so any string
//!   renders and probing would happily report a seven-rung ladder for a model
//!   trained on three. There the real vocabulary lives in the model card, which
//!   is what `Dialect::effort_kwarg` already holds with its citation - so that
//!   table stays, scoped to exactly the case where measurement cannot answer.
//!
//! The DEFAULT rung is always measured, both shapes alike: render with the
//! variable unset and see which rung's render it equals. That is how the
//! checkpoint's published default (Qwen3.8 xhigh, gpt-oss medium, Muse high)
//! reaches the wire instead of a house value - the same principle as the
//! elected sampling profiles in `paddock_models::sampling`.

use crate::parsers::Dialect;

/// The two template variables any family we serve grades effort with. Probed
/// by name because a template reading neither is the common case (Qwen3.6,
/// gemma4, laguna, granite) and must come back with no ladder at all.
const KWARGS: [&str; 2] = ["reasoning_effort", "reasoning_strength"];

/// Candidate rungs, lowest first. This is the OpenAI `reasoning_effort`
/// vocabulary minus `none`, which is the request to stop reasoning rather than
/// a level to reason at - see `ReasoningCaps::off`.
const CANDIDATES: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

/// A value no published ladder contains. A template that raises on it is
/// telling us it validates its input, which is what makes the measured
/// accepted-set trustworthy.
const SENTINEL: &str = "paddock-probe-not-a-real-level";

/// Where a ladder's rung names came from. Surfaced so a reader can tell a
/// measured fact from a cited one without going and looking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LadderSource {
    /// The template validates its input and we read the accepted set off it.
    Template,
    /// The template interpolates freely; the rungs are the model card's, via
    /// `Dialect::effort_kwarg`.
    Card,
    /// The template interpolates freely and we have no citation for this
    /// family - the OpenAI wire's own three levels, which is what a client
    /// asking for "effort" means and what such a template will render
    /// verbatim. Honest fallback, not a measurement.
    Wire,
}

/// One served model's reasoning surface.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningCaps {
    /// The template variable that grades effort, or `None` when this template
    /// grades nothing.
    pub kwarg: Option<&'static str>,
    /// The rungs it distinguishes, lowest first, in the template's own
    /// spelling (Qwen3.8 says `xhigh`, not `high`). Empty when `kwarg` is
    /// `None`.
    pub levels: Vec<String>,
    /// The rung the template picks when the variable is unset - the
    /// checkpoint's published default. `None` only if no rung reproduces the
    /// unset render, which would mean the template's default is unreachable
    /// through the variable.
    pub default_level: Option<String>,
    /// Every effort name the variable takes as it is, aliases included -
    /// Qwen3.8 takes `high` and renders it as `xhigh`, Inkling 2 takes all
    /// seven wire names (its five rungs, `xhigh` as `max`'s alias, and `none`
    /// as an effort of 0.0 - which its card says still reasons, so it is the
    /// bottom of the dial, not an off switch). A request naming one of these
    /// goes to the template verbatim; only a name it refuses is ranked onto
    /// `levels`. A ladder that interpolates freely accepts exactly its
    /// `levels`.
    pub accepted: Vec<String>,
    /// `enable_thinking` measurably changes what this template renders, so
    /// reasoning can be turned off. Measured, which is the point: the old
    /// dialect list had already been wrong twice (gemma4 and laguna both read
    /// the flag while `has_thinking_toggle` said they did not).
    pub off: bool,
    /// This template keeps a prior turn's thinking in the prompt when asked to,
    /// and drops it when asked not to - `preserve_thinking`, a kwarg Qwen3.6
    /// and Qwen3.8 read and nothing else we serve does.
    ///
    /// Measured, and it has to be: the kwarg appears in the template as a plain
    /// name, so a substring check would say yes for a template that mentions it
    /// in a branch it never takes. The probe renders a two-turn history both
    /// ways and asks whether the model would actually be shown something
    /// different.
    ///
    /// False does not mean prior thinking is dropped - it means the template
    /// offers no say in the matter, so there is no control to draw.
    pub preserve: bool,
    pub source: LadderSource,
}

impl ReasoningCaps {
    /// Nothing to control: no ladder and no off switch.
    pub fn none() -> Self {
        ReasoningCaps {
            kwarg: None,
            levels: Vec::new(),
            default_level: None,
            accepted: Vec::new(),
            off: false,
            preserve: false,
            source: LadderSource::Template,
        }
    }

    /// Does this model answer to a reasoning control at all? `false` is what
    /// makes `reasoning_effort` an honest 400 rather than a silent no-op.
    pub fn reasons(&self) -> bool {
        self.kwarg.is_some() || self.off
    }

    /// The control shape a UI should draw: a graded picker, an on/off switch,
    /// or nothing. A model with both (Qwen3.8) reports `effort` - its off
    /// position is the bottom item of the one picker, not a second control.
    pub fn style(&self) -> &'static str {
        if self.kwarg.is_some() {
            "effort"
        } else if self.off {
            "toggle"
        } else {
            "none"
        }
    }

    /// Rank a requested level onto the rungs this template really has.
    /// `rank` is `chat::reasoning_effort_rank`'s output over the seven-value
    /// wire vocabulary; asking a 3-rung family for `max` lands on its top.
    pub fn clamp(&self, rank: usize) -> Option<&str> {
        self.levels
            .get(rank.min(self.levels.len().saturating_sub(1)))
            .map(String::as_str)
    }

    /// The value a request's `effort` sends: the name itself where the
    /// template takes it, else its rank clamped onto the rungs. Name first
    /// because the rank has four steps and a ladder can have more - Inkling
    /// 2's five rungs put every ranked request one rung low (`high` rendered
    /// as `medium`).
    pub fn rung_for(&self, effort: &str, rank: usize) -> Option<&str> {
        match self.accepted.iter().find(|a| *a == effort) {
            Some(a) => Some(a.as_str()),
            None => self.clamp(rank),
        }
    }
}

/// Render a one-turn probe conversation. Deliberately minimal: no tools, no
/// system message, so the only thing that can move the output is the kwarg
/// under test.
fn probe_render(template: &str, kwargs: serde_json::Value) -> Result<String, String> {
    let msgs = [serde_json::json!({"role": "user", "content": "probe"})];
    crate::chat_template::render(template, &msgs, None, Some(&kwargs))
}

/// Does this template let a caller decide whether a prior turn's thinking stays
/// in the prompt?
///
/// Needs a history rather than one turn - `preserve_thinking` only reaches a
/// completed assistant message, so the single-turn probe above cannot see it -
/// and needs that message to CARRY reasoning, or a template that honours the
/// kwarg still renders the same nothing both ways and reports false.
///
/// A functional test, deliberately: the kwarg is an ordinary name in the
/// template text, so grepping for it would answer yes for a template that
/// mentions it in a branch it never takes. This asks the only question that
/// matters - would the model be shown something different.
fn probes_preserve(template: &str) -> bool {
    let msgs = [
        serde_json::json!({"role": "user", "content": "probe"}),
        serde_json::json!({
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "paddock-probe-prior-thinking",
        }),
        serde_json::json!({"role": "user", "content": "again"}),
    ];
    let render = |keep: bool| {
        crate::chat_template::render(
            template,
            &msgs,
            None,
            Some(&serde_json::json!({"enable_thinking": true, "preserve_thinking": keep})),
        )
    };
    match (render(true), render(false)) {
        (Ok(a), Ok(b)) => a != b,
        _ => false,
    }
}

/// Measure `template`'s reasoning surface. `dialect` is consulted for one
/// thing - the cited rung names of a template that interpolates instead of
/// validating (see the module docs).
///
/// Costs a dozen renders of a two-message conversation, each of which reparses
/// the template. That is a few milliseconds against a model load measured in
/// seconds, and it buys a capability that cannot be stale.
pub fn probe(template: &str, dialect: Dialect) -> ReasoningCaps {
    // Can thinking be turned off? The functional test, not a family list: does
    // flipping the flag change what the model is shown. Both renders must
    // succeed - a template that errors either way is not offering a control.
    let think_on = probe_render(template, serde_json::json!({"enable_thinking": true}));
    let think_off = probe_render(template, serde_json::json!({"enable_thinking": false}));
    let off = match (&think_on, &think_off) {
        (Ok(a), Ok(b)) => a != b,
        _ => false,
    };

    // Independent of the ladder and of the off switch - a template can grade
    // nothing and still decide what to do with yesterday's thinking - so it is
    // measured up here and rides every return path below.
    let preserve = probes_preserve(template);

    // Every effort probe below holds thinking on, because that is the mode a
    // ladder applies in (Qwen3.8 skips its whole effort block when thinking is
    // off) and it is the runner's serving default.
    let Ok(base) = think_on else {
        return ReasoningCaps {
            off,
            preserve,
            ..ReasoningCaps::none()
        };
    };
    let render = |kw: &str, v: &str| {
        probe_render(
            template,
            serde_json::json!({"enable_thinking": true, kw: v}),
        )
    };

    // Which variable does this template read? A template reads `kw` if setting
    // it moves the render or makes it fail; one that ignores the name renders
    // the base output no matter what we put there.
    let mut found: Option<(&'static str, bool)> = None;
    for kw in KWARGS {
        let sentinel = render(kw, SENTINEL);
        let low = render(kw, "low");
        let validating = sentinel.is_err();
        let reads = validating
            || sentinel.as_deref().ok() != Some(base.as_str())
            || low.as_deref().ok() != Some(base.as_str());
        if reads {
            found = Some((kw, validating));
            break;
        }
    }
    let Some((kwarg, validating)) = found else {
        return ReasoningCaps {
            off,
            preserve,
            ..ReasoningCaps::none()
        };
    };

    // The rungs. A validating template has already told us its accepted set -
    // read it. An interpolating one accepts everything, so measurement would
    // invent levels; take the cited ones instead.
    let (levels, mut accepted, source) = if validating {
        let (levels, accepted) = measure_levels(&render, kwarg);
        (levels, accepted, LadderSource::Template)
    } else {
        let (levels, source): (Vec<String>, _) = match dialect.effort_kwarg() {
            // the citation has to be for this variable, or it is a different
            // family's ladder wearing the same name
            Some((cited, ladder)) if cited == kwarg => (
                ladder.iter().map(|s| (*s).to_owned()).collect(),
                LadderSource::Card,
            ),
            _ => (
                ["low", "medium", "high"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                LadderSource::Wire,
            ),
        };
        // an interpolating template renders any name; the ones it grades
        // are its ladder
        (levels.clone(), levels, source)
    };
    if levels.is_empty() {
        return ReasoningCaps {
            off,
            preserve,
            ..ReasoningCaps::none()
        };
    }

    // Which rung is the template's own default? `base` was rendered with the
    // variable unset, so whichever rung reproduces it is the published
    // default - measured identically for both template shapes.
    let default_level = levels
        .iter()
        .find(|l| render(kwarg, l).as_deref().ok() == Some(base.as_str()))
        .cloned();

    // A validating template that takes the spec's `none` itself (Inkling 2:
    // an effort of 0.0) gets it verbatim. Not a rung and not `off`: where the
    // flag turns thinking off, `none` still goes there (chat's merge checks
    // `off` first), and where it does not, nothing says the variable's
    // bottom stops the reasoning - Inkling's card says it does not.
    if validating && render(kwarg, "none").is_ok() {
        accepted.push("none".to_owned());
    }

    ReasoningCaps {
        kwarg: Some(kwarg),
        levels,
        default_level,
        accepted,
        off,
        preserve,
        source,
    }
}

/// Read a validating template's accepted set: render every candidate, drop the
/// ones it refuses, and collapse aliases. Returns the rungs and every name
/// the template took (aliases included).
///
/// Two candidates that render IDENTICALLY are one rung wearing two names -
/// Qwen3.8 rewrites `high` to `xhigh` before it does anything with it. The
/// survivor is the higher-ranked name, which is the template's own canonical
/// spelling in the one case we can check (`xhigh` is what its error message
/// lists; `high` is only the alias it accepts).
fn measure_levels(
    render: &impl Fn(&str, &str) -> Result<String, String>,
    kwarg: &str,
) -> (Vec<String>, Vec<String>) {
    let mut rungs: Vec<(String, String)> = Vec::new(); // (rendered, name)
    let mut accepted = Vec::new();
    for cand in CANDIDATES {
        let Ok(out) = render(kwarg, cand) else {
            continue;
        };
        accepted.push(cand.to_owned());
        match rungs.iter_mut().find(|(seen, _)| *seen == out) {
            // CANDIDATES ascends, so a later match is the higher-ranked name
            Some(slot) => slot.1 = cand.to_owned(),
            None => rungs.push((out, cand.to_owned())),
        }
    }
    (rungs.into_iter().map(|(_, name)| name).collect(), accepted)
}

/// The reasoning-stop policy for a request whose render opened a think block
/// (`paddock_engine::sampler::ReasoningStop`: the model's own stops do not end
/// a reply inside its reasoning). The dialect's single-token close leaves the
/// block, so does a tool call's opener where calls sit inside it, and a run
/// of stop draws forces the thinking budget's own exit recipe. None where the
/// render left thinking closed, where reasoning is channel-structured
/// (Harmony, muse: a multi-token exit interleaved with stop semantics), or
/// where the tokenizer has no single close token.
pub(crate) fn stop_policy(
    model: &crate::serving::ServingModel,
    thinking_open: bool,
) -> Option<paddock_engine::sampler::ReasoningStop> {
    if !thinking_open {
        return None;
    }
    let (close, call) = match model.dialect {
        Dialect::QwenXml | Dialect::Laguna => ("</think>", Some("<tool_call>")),
        Dialect::GemmaChannel => (crate::parsers::G_CLOSE, None),
        _ => return None,
    };
    let close = model.tokenizer.token_to_id(close)?;
    let mut leave = vec![close];
    leave.extend(call.and_then(|c| model.tokenizer.token_to_id(c)));
    let mut exit = model
        .tokenizer
        .encode(crate::chat::THINK_EXIT_PHRASE)
        .ok()?;
    exit.push(close);
    Some(paddock_engine::sampler::ReasoningStop { leave, exit })
}

/// The thinking governor for a compaction's summarization pass: that pass
/// renders with the conversation's own thinking config (so its prefix rides
/// the radix cache of the conversation being compacted), and an open think
/// block is closed after one token by the budget's exit sequence, so the
/// output budget goes to the summary. None when thinking is not open, or on
/// a family whose reasoning the budget cannot close (it then thinks first).
pub(crate) fn summary_think_close(
    model: &crate::serving::ServingModel,
    thinking_open: bool,
) -> Option<crate::chat::ThinkBudget> {
    thinking_open
        .then(|| crate::chat::think_budget(model, 1, true, "the summarization pass").ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Byte-exact templates from the GGUFs we serve. The whole point of
    // the probe is that it reads the SHIPPED file, so the gate has to as well.
    const QWEN38: &str = include_str!("../tests/fixtures/qwen38_chat_template.jinja");
    const QWEN36: &str = include_str!("../tests/fixtures/qwen36_chat_template.jinja");
    const QWEN35: &str = include_str!("../tests/fixtures/qwen35_chat_template.jinja");
    const GPTOSS: &str = include_str!("../tests/fixtures/gptoss_chat_template.jinja");
    const MUSE: &str = include_str!("../tests/fixtures/muse_chat_template.jinja");
    const GEMMA4: &str = include_str!("../tests/fixtures/gemma4_chat_template.jinja");
    const LAGUNA: &str = include_str!("../tests/fixtures/laguna_chat_template.jinja");
    const GRANITE: &str = include_str!("../tests/fixtures/granite_chat_template.jinja");
    const INKLING2: &str = include_str!("../tests/fixtures/inkling2_chat_template.jinja");

    #[test]
    fn qwen38_grades_low_medium_xhigh_and_defaults_to_xhigh() {
        let c = probe(QWEN38, Dialect::QwenXml);
        assert_eq!(c.kwarg, Some("reasoning_effort"));
        assert_eq!(c.levels, ["low", "medium", "xhigh"]);
        // the checkpoint's published default, not a house value
        assert_eq!(c.default_level.as_deref(), Some("xhigh"));
        assert!(c.off, "3.8 still takes enable_thinking");
        // read off the file, so it cannot drift from what we serve
        assert_eq!(c.source, LadderSource::Template);
        assert_eq!(c.style(), "effort");
    }

    #[test]
    fn qwen38_collapses_high_onto_xhigh_instead_of_listing_both() {
        // the template rewrites 'high' to 'xhigh' before validating, so they
        // are one rung - advertising both would promise a distinction the
        // model does not make
        let c = probe(QWEN38, Dialect::QwenXml);
        assert!(!c.levels.contains(&"high".to_owned()));
        assert_eq!(c.levels.len(), 3);
    }

    #[test]
    fn inkling2_takes_every_wire_name_and_its_own_none() {
        let c = probe(INKLING2, Dialect::Inkling);
        assert_eq!(c.kwarg, Some("reasoning_effort"));
        // effort 0.1 / 0.2 / 0.7 / 0.9 / 0.99, xhigh being max's alias
        assert_eq!(c.levels, ["minimal", "low", "medium", "high", "max"]);
        // the template's unset effort is 0.9
        assert_eq!(c.default_level.as_deref(), Some("high"));
        // no enable_thinking, and its card: the lowest effort "does not fully
        // disable reasoning" - so no off switch to draw
        assert!(!c.off);
        assert_eq!(c.style(), "effort");
        // five rungs are more than the rank's four steps: every name must
        // reach the template as itself, or `high` renders as `medium`
        for (name, level) in [
            ("none", "0"),
            ("minimal", "0.1"),
            ("low", "0.2"),
            ("medium", "0.7"),
            ("high", "0.9"),
            ("xhigh", "0.99"),
            ("max", "0.99"),
        ] {
            let kw = crate::chat::merge_reasoning_effort(&c, name, None)
                .expect("valid")
                .expect("kwargs");
            assert_eq!(kw["reasoning_effort"], name, "{name}");
            assert!(kw.get("enable_thinking").is_none(), "{name}");
            let out = probe_render(INKLING2, kw).expect("renders");
            assert!(
                out.contains(&format!("Thinking effort level: {level}<")),
                "{name}: {out}"
            );
        }
    }

    #[test]
    fn the_older_qwens_have_a_switch_and_no_ladder() {
        // 3.5 and 3.6 report the same arch and the same dialect as 3.8, which
        // is exactly why this cannot be a table
        for (name, tpl) in [("3.5", QWEN35), ("3.6", QWEN36)] {
            let c = probe(tpl, Dialect::QwenXml);
            assert_eq!(c.kwarg, None, "qwen{name} grades nothing");
            assert!(c.levels.is_empty(), "qwen{name}");
            assert!(c.off, "qwen{name} takes enable_thinking");
            assert_eq!(c.style(), "toggle", "qwen{name}");
        }
    }

    #[test]
    fn an_interpolating_template_takes_its_rungs_from_the_card() {
        // gpt-oss splices the value into "Reasoning: <x>" - every string
        // renders, so measurement would invent four extra rungs
        let c = probe(GPTOSS, Dialect::Harmony);
        assert_eq!(c.kwarg, Some("reasoning_effort"));
        assert_eq!(c.levels, ["low", "medium", "high"]);
        assert_eq!(c.source, LadderSource::Card);
        // the DEFAULT is still measured, from the template's own guard
        assert_eq!(c.default_level.as_deref(), Some("medium"));
        assert!(!c.off, "harmony has no off position");

        let m = probe(MUSE, Dialect::MuseChannel);
        assert_eq!(m.kwarg, Some("reasoning_strength"));
        assert_eq!(m.levels, ["low", "medium", "high", "xhigh"]);
        assert_eq!(m.default_level.as_deref(), Some("high"));
        assert!(
            !m.off,
            "muse renders its reasoning preamble unconditionally"
        );
    }

    #[test]
    fn the_toggle_families_are_recognised_by_render_not_by_name() {
        // gemma4 and laguna were both missing from the old hand-written list
        // for a while; a measured probe cannot make that mistake
        for (name, tpl, dialect) in [
            ("gemma4", GEMMA4, Dialect::GemmaChannel),
            ("laguna", LAGUNA, Dialect::Laguna),
        ] {
            let c = probe(tpl, dialect);
            assert!(c.off, "{name} reads enable_thinking");
            assert_eq!(c.kwarg, None, "{name} grades nothing");
            assert_eq!(c.style(), "toggle", "{name}");
        }
    }

    #[test]
    fn a_model_that_cannot_reason_says_so() {
        let c = probe(GRANITE, Dialect::JsonToolCall);
        assert!(!c.reasons());
        assert_eq!(c.style(), "none");
        assert!(c.clamp(0).is_none());
    }

    #[test]
    fn clamping_lands_on_the_rungs_the_template_really_has() {
        let c = probe(QWEN38, Dialect::QwenXml);
        // ranks are chat::reasoning_effort_rank's: none/minimal/low=0,
        // medium=1, high=2, xhigh/max=3
        assert_eq!(c.clamp(0), Some("low"));
        assert_eq!(c.clamp(1), Some("medium"));
        assert_eq!(c.clamp(2), Some("xhigh"));
        assert_eq!(c.clamp(3), Some("xhigh"), "max lands on the top rung");
    }

    #[test]
    fn the_measured_default_is_what_an_unset_request_renders() {
        // the property the whole default measurement exists for: serving a
        // model with no effort set must produce exactly the prompt its authors
        // published, byte for byte
        for (name, tpl, dialect) in [
            ("qwen3.8", QWEN38, Dialect::QwenXml),
            ("gpt-oss", GPTOSS, Dialect::Harmony),
            ("muse", MUSE, Dialect::MuseChannel),
        ] {
            let c = probe(tpl, dialect);
            let kw = c.kwarg.expect(name);
            let dflt = c.default_level.as_deref().expect(name);
            let unset =
                probe_render(tpl, serde_json::json!({"enable_thinking": true})).expect(name);
            let explicit =
                probe_render(tpl, serde_json::json!({"enable_thinking": true, kw: dflt}))
                    .expect(name);
            assert_eq!(unset, explicit, "{name} default is {dflt}");
        }
    }

    #[test]
    fn a_template_with_no_reasoning_at_all_is_cheap_to_ask() {
        // no panics, no false positives on a template that never heard of any
        // of these names
        let c = probe(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}",
            Dialect::Plain,
        );
        assert_eq!(c, ReasoningCaps::none());
    }

    /// Which families let a caller keep a prior turn's thinking. Pinned per
    /// template rather than asserted in general, because this is the fact the
    /// Studio draws a control from - a family flipping either way is a UI
    /// change and should have to say so here first.
    #[test]
    fn only_the_qwen_templates_grade_preserve_thinking() {
        for (name, tpl, want) in [
            ("qwen38", QWEN38, true),
            ("qwen36", QWEN36, true),
            ("qwen35", QWEN35, false),
            ("gptoss", GPTOSS, false),
            ("muse", MUSE, false),
            ("gemma4", GEMMA4, false),
            ("laguna", LAGUNA, false),
            ("granite", GRANITE, false),
        ] {
            assert_eq!(
                probe(tpl, Dialect::Plain).preserve,
                want,
                "{name}: preserve_thinking support measured wrong"
            );
        }
    }

    /// The probe must be measuring the BRANCH, not the mention. A template that
    /// names the kwarg but never lets it change the render has no control to
    /// offer, and saying it does would draw a switch that does nothing.
    #[test]
    fn naming_preserve_thinking_without_honouring_it_is_not_support() {
        let mentions_only = "{% if preserve_thinking is defined %}{% endif %}\
            {% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
        assert!(!probe(mentions_only, Dialect::Plain).preserve);
    }

    /// And it must not report support for a template that merely renders
    /// reasoning_content unconditionally - that keeps thinking, but gives the
    /// caller no say, which is a different answer.
    #[test]
    fn unconditional_thinking_is_not_a_control_either() {
        let always = "{% for m in messages %}{{ m.role }}: \
            {{ m.reasoning_content }}{{ m.content }}\n{% endfor %}";
        assert!(!probe(always, Dialect::Plain).preserve);
    }
}
