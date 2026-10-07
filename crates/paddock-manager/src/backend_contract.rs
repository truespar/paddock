//! Control-plane checks shared by web/native previews, saves and starts.
use std::path::Path;

use crate::registry::{CatalogArtifact, CatalogModel, Registry};

pub fn metal_kv_offload(model: &CatalogModel, weights: &CatalogArtifact) -> bool {
    weights.runtime.supports_backend("metal")
        && model.id != "qwen3.8-flash-next"
        && weights.capabilities(model).iter().any(|c| c == "chat")
        && matches!(
            model.family.as_deref(),
            Some(
                "qwen3.5"
                    | "qwen3.6"
                    | "qwen3.8"
                    | "bonsai"
                    | "granite"
                    | "minicpm"
                    | "gpt-oss"
                    | "laguna"
            )
        )
}

pub fn path_kv_offload(path: &Path) -> bool {
    if path.is_dir() {
        return paddock_models::mlx::QwenConfig::read(path).is_ok()
            || paddock_models::mlx::MiniCpmConfig::read(path).is_ok()
            || paddock_models::bonsai::BonsaiConfig::read(path).is_ok();
    }
    paddock_models::probe::probe_path(path).is_ok_and(|p| {
        matches!(
            p.architecture.as_deref(),
            Some("qwen35" | "qwen35moe" | "granite" | "llama" | "gpt-oss" | "laguna")
        )
    })
}

pub fn validate(registry: &Registry, backend: &str, doc: &toml::Value) -> Result<(), String> {
    if backend != "metal" {
        return Ok(());
    }
    if doc
        .get("device")
        .and_then(toml::Value::as_str)
        .is_some_and(|v| v != backend)
    {
        return Err(format!(
            "This manager serves {backend}; the configuration selects a different backend."
        ));
    }
    let path = Path::new(doc.get("model").and_then(toml::Value::as_str).unwrap_or(""));
    let selected = registry
        .identify_weights(path)
        .and_then(|(model, artifact)| {
            let model = registry.catalog_of(&model)?;
            Some((model, model.artifact(&artifact)?))
        });
    let kv = doc
        .get("kv_cache_dtype")
        .and_then(toml::Value::as_str)
        .unwrap_or("auto");
    let required = selected.and_then(|(_, a)| a.runtime.kv_cache_dtype.as_deref());
    let native_f32 =
        required == Some("f32") || paddock_models::bonsai::BonsaiConfig::read(path).is_ok();
    let directory = path.is_dir() || selected.is_some_and(|(_, a)| a.runtime.checkpoint_dir);
    if !(matches!(kv, "auto" | "f16") || native_f32 && kv == "f32") || directory && kv == "f16" {
        return Err("Choose checkpoint-native KV precision for Metal: auto for MLX, auto/f16 for GGUF, or f32 for Bonsai MLX.".into());
    }
    for key in ["kernel_pack", "fp8_native", "max_image_tokens"] {
        if doc.get(key).is_some() {
            return Err(format!(
                "{key} is not supported by this Metal runner. Remove it before saving or starting."
            ));
        }
    }
    if doc
        .get("moe_offload")
        .and_then(|v| v.get("enabled"))
        .and_then(toml::Value::as_bool)
        == Some(true)
    {
        return Err("Metal MoE expert offload is not implemented.".into());
    }
    if let Some(offload) = doc.get("kv_offload")
        && offload.get("enabled").and_then(toml::Value::as_bool) == Some(true)
    {
        if !selected.map_or_else(|| path_kv_offload(path), |(m, a)| metal_kv_offload(m, a)) {
            return Err("This model's Metal backend does not support KV offloading. Disable it before saving or starting.".into());
        }
        let number = |key| {
            offload
                .get(key)
                .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|n| n as f64)))
                .unwrap_or(0.0)
        };
        let (ram, disk) = (number("ram_gb"), number("nvme_gb"));
        if !ram.is_finite() || ram <= 0.0 || !disk.is_finite() || disk < 0.0 {
            return Err("Metal KV offloading needs a positive RAM transfer budget and a nonnegative disk budget.".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn installed_and_not_yet_downloaded_exports_have_the_same_contract() {
        let reg = Registry::new(std::env::temp_dir().join("contract-models")).with_backend("metal");
        for (id, artifact, supported) in [
            ("bonsai-2-27b", "mlx-2bit", true),
            ("bonsai-2-27b", "ptq1", true),
            ("qwen3.8-27b", "mlx-4bit", true),
            ("minicpm5-2b", "q8", true),
            ("minicpm5-2b", "q4", true),
            ("minicpm5-2b", "mlx-4bit", true),
            ("gemma-4-31b", "mlx-4bit", false),
            ("muse-glimmer-30b", "mlx-4bit", false),
            ("qwen3.8-flash-next", "mlx-4bit", false),
            ("kolibri-1", "mlx-mixed-4-8bit", false),
        ] {
            let model = reg.catalog_of(id).unwrap();
            let a = model.artifact(artifact).unwrap();
            assert_eq!(metal_kv_offload(model, a), supported);
            let path = a.entry_path(reg.models_dir()).unwrap();
            let mut doc: toml::Value = toml::from_str("device = 'metal'\nkv_cache_dtype = 'auto'\n[kv_offload]\nenabled = true\nram_gb = 8\nnvme_gb = 0").unwrap();
            doc.as_table_mut().unwrap().insert(
                "model".into(),
                toml::Value::String(path.to_string_lossy().into()),
            );
            assert_eq!(validate(&reg, "metal", &doc).is_ok(), supported, "{id}");
            doc["kv_offload"]["enabled"] = toml::Value::Boolean(false);
            assert!(validate(&reg, "metal", &doc).is_ok());
            doc["kv_cache_dtype"] = toml::Value::String("fp8_e4m3".into());
            assert!(validate(&reg, "metal", &doc).is_err());
        }
    }
    #[test]
    fn raw_toml_cannot_bypass_backend_controls() {
        let reg = Registry::new("models".into()).with_backend("metal");
        for text in [
            "device = 'cuda'",
            "max_image_tokens = 256",
            "kernel_pack = '/a/cuda.pack'",
            "fp8_native = '/a/fp8'",
            "[moe_offload]\nenabled = true",
        ] {
            let doc: toml::Value = toml::from_str(text).unwrap();
            assert!(validate(&reg, "metal", &doc).is_err(), "{text}");
        }
    }
}
