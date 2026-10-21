//! MCP servers for a Responses request: which tools a call may use and
//! whether it needs approval, the `mcp` tool entry resolved to a server
//! (inline url or the launch registry), and the gathered tool catalog the
//! agent loop runs with. A child of responses.rs.

use super::*;

/// Whether a specific tool needs approval, honoring the OpenAI `require_approval`
/// grammar: `"never"` / `"always"`, or `{always:{tool_names?}, never:{tool_names?}}`
/// (a bare `always` with no `tool_names` gates every tool). Falls back to the
/// registered server's stored `requireApproval` flag when the request omits it.
fn tool_needs_approval(req_val: Option<&Value>, stored_default: bool, tool_name: &str) -> bool {
    match req_val {
        Some(Value::String(s)) => s == "always",
        Some(Value::Object(o)) => match o.get("always") {
            Some(always) => match always.get("tool_names").and_then(Value::as_array) {
                Some(arr) if !arr.is_empty() => arr.iter().any(|n| n.as_str() == Some(tool_name)),
                _ => true, // `always` with no tool_names -> gate all
            },
            None => false, // only `never` present -> auto-run
        },
        _ => stored_default,
    }
}

/// Whether a tool passes the OpenAI `allowed_tools` filter: an array of names,
/// or `{tool_names?, read_only?}` (read_only matches the tool's `readOnlyHint`).
fn tool_allowed(allowed: Option<&Value>, tool: &paddock_mcp::McpTool) -> bool {
    match allowed {
        None | Some(Value::Null) => true,
        Some(Value::Array(arr)) => arr.iter().any(|n| n.as_str() == Some(&tool.name)),
        Some(Value::Object(o)) => {
            let name_ok = match o.get("tool_names").and_then(Value::as_array) {
                Some(arr) => arr.iter().any(|n| n.as_str() == Some(&tool.name)),
                None => true,
            };
            let ro_ok = match o.get("read_only").and_then(Value::as_bool) {
                Some(true) => {
                    tool.annotations
                        .as_ref()
                        .and_then(|a| a.get("readOnlyHint"))
                        .and_then(Value::as_bool)
                        == Some(true)
                }
                _ => true,
            };
            name_ok && ro_ok
        }
        _ => true,
    }
}

/// A JSON object of strings -> HashMap (headers, env), non-strings skipped.
pub(crate) fn str_map(v: Option<&Value>) -> std::collections::HashMap<String, String> {
    v.and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// Resolve an `mcp` tool entry to a server config: inline `server_url` wins;
/// a bare `server_label` falls back to the launch registry (this endpoint's
/// own config file - HTTP url+headers, or a stdio command+args+env).
/// Unknown labels are an honest error, never a silent no-tool run.
fn resolve_mcp_server(
    state: &AppState,
    t: &Value,
) -> Result<(paddock_mcp::ServerConfig, bool), String> {
    use paddock_mcp::{ServerConfig, Transport};
    if t.get("connector_id").is_some() {
        return Err(
            "mcp `connector_id` (OpenAI-hosted connectors) is not supported; \
                    use `server_url` or a registered `server_label`"
                .to_string(),
        );
    }
    let label = t
        .get("server_label")
        .and_then(Value::as_str)
        .ok_or("an mcp tool needs a server_label")?;
    if let Some(url) = t.get("server_url").and_then(Value::as_str) {
        let mut headers: std::collections::HashMap<String, String> = t
            .get("headers")
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        // OpenAI `authorization` is the OAuth bearer token for the server.
        if let Some(tok) = t.get("authorization").and_then(Value::as_str) {
            headers
                .entry("Authorization".to_string())
                .or_insert_with(|| format!("Bearer {tok}"));
        }
        let cfg = ServerConfig {
            id: format!("inline:{label}"),
            label: label.to_string(),
            transport: Transport::Http {
                url: url.to_string(),
                headers,
            },
        };
        // OpenAI's require_approval knob: "always" gates every call behind an
        // approval item. An inline server is an arbitrary third-party URL the
        // request just handed us - honor the caller's ask for a leash.
        let approval = t.get("require_approval").and_then(Value::as_str) == Some("always");
        return Ok((cfg, approval));
    }
    // No server_url: resolve the label against this runner's launch registry
    // (the config file's mcp_servers - a live view, so a tool the manager
    // just materialized resolves without a restart; standalone runners keep
    // their startup entries).
    let registry_entry = state
        .live
        .snapshot()
        .mcp_servers
        .iter()
        .find(|e| e.get("server_label").and_then(Value::as_str) == Some(label))
        .cloned();
    if let Some(entry) = registry_entry {
        let approval = entry.get("require_approval").and_then(Value::as_str) == Some("always");
        if let Some(url) = entry.get("server_url").and_then(Value::as_str) {
            let headers = str_map(entry.get("headers"));
            let cfg = ServerConfig {
                id: format!("registry:{label}"),
                label: label.to_string(),
                transport: Transport::Http {
                    url: url.to_string(),
                    headers,
                },
            };
            return Ok((cfg, approval));
        }
        // stdio: a local process speaking MCP - the npx/uvx class of server
        if let Some(command) = entry.get("command").and_then(Value::as_str) {
            let args = entry
                .get("args")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let cfg = ServerConfig {
                id: format!("registry:{label}"),
                label: label.to_string(),
                transport: Transport::Stdio {
                    command: command.to_string(),
                    args,
                    env: str_map(entry.get("env")),
                },
            };
            return Ok((cfg, approval));
        }
    }
    Err(format!(
        "mcp tool {label:?} has no server_url and is not in this runner's \
         registry - pass server_url inline, or register the server in the \
         manager (it hands the registry to every model it starts)"
    ))
}

/// Discover the tools of every `mcp` tool in the request (lazy-connecting the
/// pool), namespaced by server label so two servers can't collide, applying
/// `allowed_tools` filters and computing per-tool approval.
pub(super) async fn gather_mcp(
    state: &AppState,
    req: &ResponsesRequest,
) -> Result<Gathered, String> {
    use crate::tool_search::{self, CatalogTool};
    let mut g = Gathered {
        tools: Vec::new(),
        routing: HashMap::new(),
        needs_approval: HashSet::new(),
        listings: Vec::new(),
        catalog: Vec::new(),
        web_search: None,
        forensics: None,
        current_time: None,
        instructions: Vec::new(),
    };
    let Some(tools) = req.tools.as_ref() else {
        return Ok(g);
    };
    // Every tool's full function schema, kept per SERVER: disclosure is decided
    // one server at a time (see below), so the defs cannot be flattened until
    // that decision is made.
    let mut per_server: Vec<(String, Vec<Value>)> = Vec::new();
    for t in tools {
        let ty = t.get("type").and_then(Value::as_str);
        // `web_search` (incl. the OpenAI `web_search_preview` / dated aliases)
        // is served by the user's configured provider - a request asking for it
        // on an unconfigured server gets a clear 400, not a silent no-tool run.
        if ty.is_some_and(|s| s.starts_with("web_search")) {
            // Inline provider spec (paddock extension fields): the MANAGER
            // expands its stored Settings into the tool object when it relays
            // Studio chats - the runner stays stateless and one box-level key
            // serves every model (incl. every compare lane). Falls back to
            // this runner's own launch config for direct callers.
            let inline = t
                .get("paddock_provider")
                .and_then(Value::as_str)
                .and_then(crate::websearch::Provider::parse)
                .zip(t.get("paddock_api_key").and_then(Value::as_str))
                .map(|(provider, key)| crate::websearch::SearchConfig {
                    provider,
                    api_key: key.to_string(),
                });
            let Some(cfg) = inline.or_else(|| state.live.snapshot().web_search.clone()) else {
                return Err("web search is not set up on this runner; launch it \
                            with --web-search-provider/--web-search-api-key (or \
                            configure a provider in the manager's Studio Settings)"
                    .into());
            };
            // The OpenAI tool's knobs: search_context_size -> the depth dial
            // (result count, how hard the provider works, how much page text
            // comes back), filters.allowed_domains, and the approximate
            // user_location in full.
            let mut opts = crate::websearch::SearchOpts::default()
                .context_size(t.get("search_context_size").and_then(Value::as_str));
            if let Some(doms) = t
                .get("filters")
                .and_then(|f| f.get("allowed_domains"))
                .and_then(Value::as_array)
            {
                opts.allowed_domains = doms
                    .iter()
                    .filter_map(|d| d.as_str().map(str::to_string))
                    .collect();
            }
            opts.location = crate::websearch::Location::from_json(t.get("user_location"));
            g.web_search = Some(crate::websearch::WebSpec { cfg, opts });
            continue;
        }
        // Forensic tool (paddock extension): served only when the runner has
        // `[forensics] tool = true`. A request asking for it on a runner that
        // did not enable it gets a clear 400, never a silent no-tool run.
        if ty == Some("forensics") {
            let Some(rt) = state.forensics.clone().filter(|r| r.tool) else {
                return Err("forensic analysis is not enabled on this runner; set \
                            [forensics] enabled = true and tool = true in its config"
                    .into());
            };
            g.forensics = Some(rt);
            continue;
        }
        // Clock tool (paddock extension): always served - a clock needs no
        // provider or enablement. The declaration's `timezone` is validated
        // here so a junk zone is a 400 at request time, not a wrong answer.
        if ty == Some("current_time") {
            g.current_time = Some(crate::clock::parse_spec(t)?);
            continue;
        }
        if ty != Some("mcp") {
            continue;
        }
        let (cfg, stored_default) = resolve_mcp_server(state, t)?;
        let req_approval = t.get("require_approval");
        // allowed_tools: the request's own filter wins; else the registry
        // entry's configured filter (per-model config) applies.
        let allowed_owned = t.get("allowed_tools").cloned().or_else(|| {
            state
                .live
                .snapshot()
                .mcp_servers
                .iter()
                .find(|e| e.get("server_label").and_then(Value::as_str) == Some(cfg.label.as_str()))
                .and_then(|e| e.get("allowed_tools"))
                .cloned()
        });
        let allowed = allowed_owned.as_ref();
        let mtools = state
            .mcp
            .list_tools(&cfg)
            .await
            .map_err(|e| format!("mcp server {:?}: {e}", cfg.label))?;
        let mut listing = Vec::new();
        let mut server_defs: Vec<Value> = Vec::new();
        for mt in mtools {
            if !tool_allowed(allowed, &mt) {
                continue;
            }
            let ns = format!("{}__{}", cfg.label, mt.name);
            if tool_needs_approval(req_approval, stored_default, &mt.name) {
                g.needs_approval.insert(ns.clone());
            }
            server_defs.push(json!({
                "type": "function",
                "function": { "name": ns, "description": mt.description, "parameters": mt.input_schema },
            }));
            g.catalog.push(CatalogTool {
                name: ns.clone(),
                description: mt.description.clone().unwrap_or_default(),
                input_schema: mt.input_schema.clone(),
            });
            g.routing.insert(ns, (cfg.clone(), mt.name.clone()));
            listing.push(mt);
        }
        if let Some(instr) = state.mcp.instructions(&cfg).await {
            // State the names we actually declared. A server writes its
            // instructions in terms of its own tool names, but this path
            // namespaces them as `<label>__<tool>` - so guidance saying "call
            // artifact_create" sent the model looking for a tool that does not
            // exist under that name, and it burned a round on mcp_search_tools
            // instead. The server cannot know the prefix; we
            // do, so we say it.
            let declared: Vec<&str> = listing.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();
            let names: Vec<String> = declared
                .iter()
                .map(|n| format!("{}__{n}", cfg.label))
                .collect();
            g.instructions.push(if names.is_empty() {
                instr
            } else {
                format!(
                    "{instr}

Call this server's tools by these exact names: {}.",
                    names.join(", ")
                )
            });
        }
        g.listings.push(Listing {
            server_label: cfg.label.clone(),
            tools: listing,
        });
        // One entry per server (a label repeated across two `mcp` tool entries
        // is still one server, and namespacing already assumes that).
        match per_server.iter_mut().find(|(l, _)| l == &cfg.label) {
            Some((_, defs)) => defs.extend(server_defs),
            None => per_server.push((cfg.label.clone(), server_defs)),
        }
    }

    // Disclosure, decided per SERVER: spend the budget smallest-first and hide
    // only the servers that are actually big. This used to be one global switch
    // - over the threshold, every schema vanished at once - which meant a
    // 5-tool server behind a 40-tool one lost its schemas for reasons that had
    // nothing to do with it, and its calls then travelled through
    // `mcp_call_tool` where the argument grammar cannot reach them. See
    // tool_search::disclose_servers for why that is a correctness bug.
    let weights: Vec<tool_search::ServerWeight> = per_server
        .iter()
        .map(|(label, defs)| tool_search::ServerWeight {
            label: label.clone(),
            tools: defs.len(),
            chars: defs.iter().map(|d| d.to_string().chars().count()).sum(),
        })
        .collect();
    let shown = tool_search::disclose_servers(&weights, state.max_ctx);
    let mut direct_tools: Vec<Value> = Vec::new();
    let mut hidden_labels: Vec<String> = Vec::new();
    let mut hidden_tools = 0usize;
    for (label, defs) in per_server {
        if shown.contains(&label) {
            direct_tools.extend(defs);
        } else {
            hidden_tools += defs.len();
            hidden_labels.push(label);
        }
    }
    // The pair is declared either way, so the model is told either way -
    // declaring a capability and not mentioning it is how it ends up never
    // used. Which text depends on what actually happened to the schemas.
    if !g.catalog.is_empty() {
        g.instructions.push(if hidden_labels.is_empty() {
            tool_search::SEARCH_AVAILABLE_INSTRUCTIONS.to_string()
        } else if direct_tools.is_empty() {
            tool_search::SEARCH_MODE_INSTRUCTIONS.to_string()
        } else {
            tool_search::partial_mode_instructions(&hidden_labels, hidden_tools)
        });
    }
    // The search pair is always declared, in every mode. Disclosure used to be
    // an either/or: under the threshold the model got every schema and no way
    // to search, over it every schema vanished at once. Both halves bite - a
    // tool the model overlooks was unrecoverable below the line, and crossing
    // the line cost a discovery round on every request. Now the schemas are
    // what degrades, server by server; searchability never does.
    g.tools = direct_tools;
    // ...but only when there is a catalog to search. A web-search-only request
    // used to get the pair too, and two meta-tools that can only ever return
    // nothing are an invitation to waste a round.
    if !g.catalog.is_empty() {
        g.tools.push(tool_search::search_tool_def());
        g.tools.push(tool_search::call_tool_def());
    }
    // web_search is one small schema - always disclosed directly, never behind
    // the catalog-search meta-tools.
    if g.web_search.is_some() {
        g.tools.push(crate::websearch::tool_def());
    }
    // Forensics is one small schema - disclosed directly like web_search.
    if g.forensics.is_some() {
        g.tools.push(crate::forensics::tool_def());
    }
    // The clock is the smallest schema of all.
    if g.current_time.is_some() {
        g.tools.push(crate::clock::tool_def());
    }
    Ok(g)
}
