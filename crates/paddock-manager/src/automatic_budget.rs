//! An admitted reservation is not a user ceiling. Keep that distinction in
//! the file without adding runner-schema fields: old runners ignore comments.
//! The marker includes its value, so a hand-edited number becomes explicit.
use toml_edit::DocumentMut;

fn marked(doc: &DocumentMut) -> Option<u64> {
    let value = doc.get("vram_budget")?.as_value()?;
    let number = u64::try_from(value.as_integer()?).ok()?;
    let suffix = value.decor().suffix()?.as_str()?.trim();
    (suffix == format!("# paddock:auto-budget={number}")).then_some(number)
}

pub(crate) fn is_automatic(text: &str) -> bool {
    text.parse::<DocumentMut>()
        .ok()
        .and_then(|d| marked(&d))
        .is_some()
}

/// Only absent or manager-marked budgets may be repriced. An unmarked legacy
/// grant is conservatively explicit; never guess away an operator's ceiling.
pub(crate) fn pin(text: &str, grant: u64) -> String {
    let Ok(mut doc) = text.parse::<DocumentMut>() else {
        return text.into();
    };
    if doc.get("vram_budget").is_some() && marked(&doc).is_none() {
        return text.into();
    }
    let Ok(grant) = i64::try_from(grant) else {
        return text.into();
    };
    let mut value = toml_edit::Value::from(grant);
    value
        .decor_mut()
        .set_suffix(format!(" # paddock:auto-budget={grant}"));
    doc["vram_budget"] = toml_edit::Item::Value(value);
    doc.to_string()
}

/// Preserve provenance across a serialization which cannot retain comments,
/// but only if the numeric reservation itself survived unchanged.
pub(crate) fn preserve(source: &str, output: String) -> String {
    let Some(grant) = source.parse::<DocumentMut>().ok().and_then(|d| marked(&d)) else {
        return output;
    };
    mark(&output, Some(grant))
}

pub(crate) fn mark(text: &str, grant: Option<u64>) -> String {
    let Some(grant) = grant else {
        return text.into();
    };
    let Ok(mut doc) = text.parse::<DocumentMut>() else {
        return text.into();
    };
    let Some(value) = doc
        .get_mut("vram_budget")
        .and_then(toml_edit::Item::as_value_mut)
    else {
        return text.into();
    };
    if value.as_integer().and_then(|n| u64::try_from(n).ok()) != Some(grant) {
        return text.into();
    }
    value
        .decor_mut()
        .set_suffix(format!(" # paddock:auto-budget={grant}"));
    doc.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_grants_reprice_but_manual_and_legacy_limits_do_not() {
        let source = "# keep\nmodel='fixture'\n[custom]\nkeep=true\n";
        let small = pin(source, 1024);
        let large = pin(&small, 2048);
        assert!(is_automatic(&small) && is_automatic(&large));
        assert!(large.contains("# keep") && large.contains("keep=true"));
        assert_eq!(marked(&large.parse().unwrap()), Some(2048));
        let manual = small.replace("vram_budget = 1024", "vram_budget = 1536");
        assert!(!is_automatic(&manual));
        assert_eq!(pin(&manual, 4096), manual);
        let legacy = "vram_budget=1024\n";
        assert!(!is_automatic(legacy));
        assert_eq!(pin(legacy, 4096), legacy);
        assert!(!is_automatic(
            "vram_budget=1024 # paddock:auto-budget=2048\n"
        ));
        assert!(is_automatic(&preserve(&small, legacy.into())));
        assert_eq!(
            preserve(&small, "vram_budget=2048\n".into()),
            "vram_budget=2048\n"
        );
    }
}
