//! One config parser/projection shared by web, native and runner supervision.
use super::*;

impl Supervisor {
    /// Reconstruct a SpawnSpec from an endpoint's config FILE alone - the file
    /// is the truth for everything the endpoint serves with, catalog identity
    /// included: `[catalog]` names the model, `model` names the
    /// bytes, and `Registry::identity_for` reconciles the two. A file without
    /// the block falls back to its weights path (a filesystem path is a valid
    /// spawn model) and an election can still layer identity on top, which is
    /// what every config written before the `[catalog]` block existed does.
    pub fn spec_from_config_file(&self, path: &Path) -> Result<SpawnSpec, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|err| format!("config file {} unreadable: {err}", path.display()))?;
        self.spec_from_config_text(&raw)
            .map_err(|err| format!("config file {}: {err}", path.display()))
    }

    /// The same projection, over config TEXT that need not be on disk yet.
    ///
    /// This is what `/api/servers/project` serves, and it exists so the Studio's
    /// Simple tab has no rule of its own. The editor works on an unsaved buffer,
    /// so it cannot read the saved file - and the previous answer, re-deriving
    /// the model identity in the browser, is what produced two separate
    /// disagreements in one day (`/api/servers` vs `heal_spec_identity`, then
    /// the browser vs `identify_weights`). A mirror kept in sync by hand is a
    /// defect with a delay fuse; one implementation, reachable over HTTP, has no
    /// second copy to drift.
    pub fn spec_from_config_text(&self, raw: &str) -> Result<SpawnSpec, String> {
        // A leading BOM is tolerated by Rust's toml and refused by the Studio's
        // parser, which cost a session. Strip it here so the one parser
        // both halves now share cannot disagree about it either.
        let raw = raw.strip_prefix('\u{feff}').unwrap_or(raw);
        let v: toml::Value =
            toml::from_str(raw).map_err(|err| format!("does not parse as TOML: {err}"))?;
        crate::native_endpoints::validate_residency(&v)?;
        let get_usize = |k: &str| {
            v.get(k)
                .and_then(toml::Value::as_integer)
                .map(|n| n as usize)
        };
        let get_str = |k: &str| v.get(k).and_then(toml::Value::as_str).map(String::from);
        let mcp_servers = v
            .get("mcp_servers")
            .and_then(|x| serde_json::to_value(x).ok())
            .and_then(|x| match x {
                serde_json::Value::Array(a) => Some(a),
                _ => None,
            })
            .unwrap_or_default();
        // The `[forensics]` block round-trips as a whole: enabled + any hand-set
        // auto/tool/device. A block that does not deserialize (a stray key) is
        // simply dropped from the projection rather than failing the whole
        // parse - the same forgiving stance the rest of this reader takes.
        let forensics = v
            .get("forensics")
            .cloned()
            .and_then(|x| x.try_into::<ForensicsSpec>().ok());
        let kv_offload = v
            .get("kv_offload")
            .cloned()
            .and_then(|x| x.try_into::<KvOffloadSpec>().ok());
        // Catalog identity: what the file DECLARES, reconciled against the
        // weights it points at. `identity_for` handles a file with no block -
        // which is every config written before that block existed - by recognizing the
        // weights, so nothing on disk needs migrating.
        let weights = get_str("model").unwrap_or_default();
        let declared = v.get("catalog").and_then(toml::Value::as_table);
        let ident = self.registry.identity_for(
            declared.and_then(|c| {
                Some((
                    c.get("model").and_then(toml::Value::as_str)?,
                    c.get("artifact").and_then(toml::Value::as_str),
                ))
            }),
            Path::new(&weights),
        );
        Ok(SpawnSpec {
            residency: v
                .get("residency")
                .cloned()
                .map(|v| v.try_into())
                .transpose()
                .map_err(|_| "Invalid model residency policy.")?,
            kv_offload,
            // identity when we have one, the weights path otherwise - a
            // filesystem path is a valid spawn model
            model: ident.as_ref().map_or(weights, |(id, _)| id.clone()),
            artifact: ident.and_then(|(_, a)| a),
            drafter: declared
                .and_then(|c| c.get("drafter"))
                .and_then(toml::Value::as_str)
                .map(str::to_owned),
            // a start re-serves what is on disk; never re-downloads a
            // deleted model behind the operator's back
            pull: false,
            fp8_native: get_str("fp8_native").is_some(),
            // the config file speaks for itself: a file with mmproj serves
            // vision. Preserve both explicit answers for optional bundled
            // towers, which have no mmproj key to carry an enabled state.
            vision: v.get("vision").and_then(toml::Value::as_bool),
            // the audio tower is opt-in, so both answers are kept: a file
            // with `audio_mmproj` serves audio, one with `audio = false`
            // does not, and a re-render must not fall back to the default
            audio: if v.get("audio_mmproj").is_some() {
                Some(true)
            } else {
                v.get("audio").and_then(toml::Value::as_bool)
            },
            host: get_str("host")
                .map(|host| host.parse().map_err(|_| "invalid endpoint bind address"))
                .transpose()?,
            port: v
                .get("port")
                .and_then(toml::Value::as_integer)
                .map(|n| n as u16),
            max_ctx: get_usize("max_ctx"),
            max_batch: get_usize("max_batch"),
            api_key: get_str("api_key"),
            runner_version: None,
            gpu: get_str("gpu"),
            persist: true,
            pinned: false,
            web_search_provider: get_str("web_search_provider"),
            web_search_api_key: get_str("web_search_api_key"),
            mcp_servers,
            kv_cache_dtype: get_str("kv_cache_dtype"),
            spec_policy: get_str("spec"),
            vram_budget: v
                .get("vram_budget")
                .and_then(toml::Value::as_integer)
                .filter(|_| !crate::automatic_budget::is_automatic(raw))
                .map(|n| n.max(0) as u64),
            automatic_budget: None,
            evict: Vec::new(),
            forensics,
        })
    }

    /// The Start/Edit page's Simple tab, projected from config TEXT.
    ///
    /// Everything here is READ from the buffer; nothing is invented and nothing
    /// is written. It exists so the browser holds no reconciliation rule of its
    /// own - see `spec_from_config_text` for what that cost when it did.
    pub fn project_config_text(&self, raw: &str) -> Result<ConfigProjection, String> {
        let spec = self.spec_from_config_text(raw)?;
        // `mmproj` presence is the optional vision switch, and SpawnSpec deliberately
        // carries `vision: None` for a file ("the file speaks for itself" - a
        // spawn must not re-decide it), so read it here rather than bend that.
        let trimmed = raw.strip_prefix('\u{feff}').unwrap_or(raw);
        let v: toml::Value = toml::from_str(trimmed).map_err(|e| e.to_string())?;
        let runtime = self
            .registry
            .catalog_of(&spec.model)
            .and_then(|m| spec.artifact.as_deref().and_then(|id| m.artifact(id)))
            .map(|a| &a.runtime);
        let towers = runtime.and_then(|r| r.optional_towers.as_ref());
        let embedded_vision = runtime.is_some_and(|r| r.embedded_vision)
            || towers.and_then(|t| t.vision.as_ref()).is_some_and(|t| {
                v.get("vision")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(t.default)
            });
        let bundled_audio = towers.and_then(|t| t.audio.as_ref()).is_some_and(|t| {
            v.get("audio")
                .and_then(toml::Value::as_bool)
                .unwrap_or(t.default)
        });
        Ok(ConfigProjection {
            residency: spec.residency,
            residency_supported: crate::native_endpoints::residency_supported(&v),
            weights: v
                .get("model")
                .and_then(toml::Value::as_str)
                .map(String::from),
            vision: (v.get("mmproj").is_some() || embedded_vision)
                && v.get("vision").and_then(toml::Value::as_bool) != Some(false),
            audio: (v.get("audio_mmproj").is_some() || bundled_audio)
                && v.get("audio").and_then(toml::Value::as_bool) != Some(false),
            fp8_native: spec.fp8_native,
            model: spec.model,
            artifact: spec.artifact,
            drafter: spec.drafter,
            max_ctx: spec.max_ctx,
            max_batch: spec.max_batch,
            gpu: spec.gpu,
            kv_cache_dtype: spec.kv_cache_dtype,
            spec: spec.spec_policy,
            api_key: spec.api_key,
            vram_budget: spec.vram_budget,
            web_search_provider: spec.web_search_provider,
            web_search_api_key: spec.web_search_api_key,
            mcp_servers: spec.mcp_servers,
            forensics: spec.forensics,
            kv_offload: spec.kv_offload,
        })
    }
}
