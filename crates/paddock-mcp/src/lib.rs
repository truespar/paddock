//! MCP (Model Context Protocol) client for Paddock - a thin, paddock-facing
//! wrapper over the official `rmcp` SDK plus a lazy connection pool.
//!
//! Registered servers are inert until a tool is actually needed; `McpManager`
//! connects on first use and keeps one live client per server, shared across
//! every request/conversation (this is the "reuse" the API exposes).
//!
//! P0 wires the **stdio** (child-process) transport; Streamable HTTP + OAuth
//! land next behind the same `ServerConfig`/`McpClient` surface.

use std::collections::HashMap;
use std::sync::Arc;

pub mod oauth;

use rmcp::ServiceExt;
use rmcp::model::CallToolRequestParams;
use rmcp::transport::TokioChildProcess;
use tokio::process::Command;
use tokio::sync::Mutex;

pub type Result<T> = std::result::Result<T, McpError>;

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("mcp transport error: {0}")]
    Transport(String),
    #[error("mcp service error: {0}")]
    Service(String),
    #[error("mcp oauth error: {0}")]
    Oauth(String),
    #[error("mcp transport not supported yet: {0}")]
    Unsupported(&'static str),
}

/// How to reach an MCP server.
#[derive(Clone, Debug)]
pub enum Transport {
    /// Spawn a local process speaking MCP over stdio.
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// Connect to a remote server over Streamable HTTP (wired next).
    Http {
        url: String,
        headers: HashMap<String, String>,
    },
}

/// Env names a stdio server's config may not set: the dynamic loader reads
/// these before the tool's own code runs, so they would turn any allowed
/// command into a different program. A tool needs none of them to work.
fn steers_loader(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.starts_with("LD_") || upper.starts_with("DYLD_")
}

/// A registered MCP server. `id` keys the connection pool; `label` namespaces
/// its tools so two servers can't collide.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub id: String,
    pub label: String,
    pub transport: Transport,
}

/// A tool discovered from an MCP server (`tools/list`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct McpTool {
    pub name: String,
    pub description: Option<String>,
    /// JSON Schema for the tool's arguments (fed to the model's tool set).
    pub input_schema: serde_json::Value,
    /// MCP tool annotations (`readOnlyHint`, `title`, ...) as JSON, or null.
    /// Surfaced in `mcp_list_tools` and used for `allowed_tools` read-only filters.
    pub annotations: Option<serde_json::Value>,
}

/// The result of a `tools/call`, flattened to JSON so callers never touch rmcp
/// types directly.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolResult {
    pub is_error: bool,
    /// The MCP content blocks (text/image/resource) as JSON.
    pub content: serde_json::Value,
}

/// One live connection to an MCP server.
pub struct McpClient {
    service: rmcp::service::RunningService<rmcp::RoleClient, ()>,
    label: String,
}

impl McpClient {
    /// Connect + run the MCP `initialize` handshake.
    pub async fn connect(cfg: &ServerConfig) -> Result<Self> {
        let service = match &cfg.transport {
            Transport::Stdio { command, args, env } => {
                if let Some(name) = env.keys().find(|k| steers_loader(k)) {
                    return Err(McpError::Transport(format!(
                        "mcp stdio server {:?}: env {name} is refused - variables that \
                         steer the dynamic loader never pass to a tool process",
                        cfg.label
                    )));
                }
                let mut cmd = Command::new(command);
                cmd.args(args);
                for (k, v) in env {
                    cmd.env(k, v);
                }
                let transport =
                    TokioChildProcess::new(cmd).map_err(|e| McpError::Transport(e.to_string()))?;
                ().serve(transport)
                    .await
                    .map_err(|e| McpError::Service(e.to_string()))?
            }
            Transport::Http { url, headers } => {
                use rmcp::transport::StreamableHttpClientTransport;
                use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;

                // Custom headers (e.g. Authorization) ride the config; TLS/https
                // comes from reqwest's rustls (rmcp owns the client).
                let mut custom = std::collections::HashMap::new();
                for (k, v) in headers {
                    match (
                        http::HeaderName::from_bytes(k.as_bytes()),
                        http::HeaderValue::from_str(v),
                    ) {
                        (Ok(name), Ok(val)) => {
                            custom.insert(name, val);
                        }
                        _ => tracing::warn!(header = %k, "skipping invalid mcp http header"),
                    }
                }
                // #[non_exhaustive] as of rmcp 3.x, so no struct literal - the
                // constructor takes the uri and the rest stays at its defaults.
                let mut config = StreamableHttpClientTransportConfig::with_uri(url.clone());
                config.custom_headers = custom;
                config.max_sse_event_size = 4 * 1024 * 1024;
                // Credential-bearing MCP requests must never follow redirects
                // to an unreviewed destination, including custom API-key headers.
                let client = reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(8))
                    .redirect(reqwest::redirect::Policy::none())
                    .build()
                    .map_err(|_| McpError::Transport("HTTP client unavailable".into()))?;
                let transport = StreamableHttpClientTransport::with_client(client, config);
                ().serve(transport)
                    .await
                    .map_err(|e| McpError::Service(e.to_string()))?
            }
        };
        tracing::info!(server = %cfg.label, "mcp connected");
        Ok(Self {
            service,
            label: cfg.label.clone(),
        })
    }

    /// Discover the server's tools.
    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        let mut tools = Vec::new();
        let mut cursor = None;
        let mut seen = std::collections::HashSet::new();
        let mut total_bytes = 0usize;
        for page in 0..64 {
            let mut params = rmcp::model::PaginatedRequestParams::default();
            params.cursor = cursor;
            let response = self
                .service
                .list_tools(Some(params))
                .await
                .map_err(|e| McpError::Service(e.to_string()))?;
            total_bytes = total_bytes.saturating_add(
                serde_json::to_vec(&response)
                    .map_err(|_| McpError::Service("Invalid tool listing".into()))?
                    .len(),
            );
            if tools.len() + response.tools.len() > 4096 || total_bytes > 16 * 1024 * 1024 {
                return Err(McpError::Service(
                    "Tool listing exceeds the 4096-tool / 16 MiB limit".into(),
                ));
            }
            tools.extend(response.tools);
            cursor = response.next_cursor;
            let Some(next) = &cursor else {
                break;
            };
            if page == 63 || !seen.insert(next.clone()) {
                return Err(McpError::Service(
                    "Tool listing pagination did not terminate".into(),
                ));
            }
        }
        Ok(tools
            .into_iter()
            .map(|t| McpTool {
                name: t.name.to_string(),
                description: t.description.map(|d| d.to_string()),
                input_schema: serde_json::Value::Object((*t.input_schema).clone()),
                annotations: t
                    .annotations
                    .as_ref()
                    .map(|a| serde_json::to_value(a).unwrap_or(serde_json::Value::Null)),
            })
            .collect())
    }

    /// Execute a tool call.
    pub async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<ToolResult> {
        let arguments = match args {
            serde_json::Value::Object(m) => Some(m),
            serde_json::Value::Null => None,
            other => Some(serde_json::Map::from_iter([("input".to_string(), other)])),
        };
        let out = self
            .service
            // rmcp 3.x: #[non_exhaustive] plus the MCP 2026-07-28 additions
            // (`input_responses` / `request_state` for multi-round-trip tool
            // calls, SEP-2322) replacing the old `task` slot. We do not drive
            // MRTR retries yet, so both stay unset.
            .call_tool({
                let p = CallToolRequestParams::new(name.to_string());
                match arguments {
                    Some(a) => p.with_arguments(a),
                    None => p,
                }
            })
            .await
            .map_err(|e| McpError::Service(e.to_string()))?;
        Ok(ToolResult {
            is_error: out.is_error.unwrap_or(false),
            content: serde_json::to_value(&out.content).unwrap_or(serde_json::Value::Null),
        })
    }

    /// The server's own `instructions` from the initialize handshake.
    ///
    /// The MCP spec has servers describe how to use them here, and hosts are
    /// meant to put it in front of the model - it is system-prompt material,
    /// not decoration. We used to collect it and drop it on the floor, which is
    /// why a tool whose description alone said "make an artifact" got ignored:
    /// the server's actual guidance never reached the prompt.
    pub fn instructions(&self) -> Option<String> {
        self.service
            .peer_info()
            .and_then(|i| i.instructions.clone())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Cache key for a pooled connection: the id plus everything baked into the
/// transport at connect time.
///
/// `cfg.id` alone is not enough and the difference is a data leak. Headers are
/// captured when the transport is built, so two requests sharing a label but
/// carrying different headers used to share one connection - the first
/// request's headers won, for the life of the pool entry. Found live: every
/// inline `mcp` tool is `inline:<label>`, so five chats
/// writing artifacts (identified by an `x-paddock-conversation` header) all
/// landed in whichever chat connected first. Any per-request credential on an
/// inline connector had the same problem.
fn pool_key(cfg: &ServerConfig) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match &cfg.transport {
        Transport::Http { url, headers } => {
            "http".hash(&mut h);
            url.hash(&mut h);
            // A HashMap iterates in arbitrary order; sort or the same headers
            // hash differently per process run and the pool never hits.
            let mut kv: Vec<_> = headers.iter().collect();
            kv.sort();
            kv.hash(&mut h);
        }
        Transport::Stdio { command, args, env } => {
            "stdio".hash(&mut h);
            command.hash(&mut h);
            args.hash(&mut h);
            let mut kv: Vec<_> = env.iter().collect();
            kv.sort();
            kv.hash(&mut h);
        }
    }
    format!("{}#{:x}", cfg.id, h.finish())
}

/// Does a pool key belong to this server id? One id can hold several entries
/// (one per distinct header/env set), so a disconnect has to sweep the family.
fn key_belongs_to(key: &str, id: &str) -> bool {
    key == id || (key.starts_with(id) && key[id.len()..].starts_with('#'))
}

/// Lazy, pooled connection manager: one live `McpClient` per (server id +
/// transport identity), shared across all requests. Connections are created on
/// first use and reused.
#[derive(Default)]
pub struct McpManager {
    clients: Mutex<HashMap<String, Arc<McpClient>>>,
}

impl McpManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the pooled client for `cfg`, connecting lazily on first use.
    pub async fn client(&self, cfg: &ServerConfig) -> Result<Arc<McpClient>> {
        let key = pool_key(cfg);
        if let Some(c) = self.clients.lock().await.get(&key) {
            return Ok(c.clone());
        }
        // Connect outside the lock so a slow handshake doesn't block the pool.
        let client = Arc::new(McpClient::connect(cfg).await?);
        let mut map = self.clients.lock().await;
        // Another task may have connected while we did; keep the first winner.
        Ok(map.entry(key).or_insert(client).clone())
    }

    pub async fn list_tools(&self, cfg: &ServerConfig) -> Result<Vec<McpTool>> {
        self.client(cfg).await?.list_tools().await
    }

    /// The server's handshake `instructions`, if it sent any. Cheap - the
    /// client is already pooled and the value came in on initialize.
    pub async fn instructions(&self, cfg: &ServerConfig) -> Option<String> {
        self.client(cfg).await.ok()?.instructions()
    }

    pub async fn call_tool(
        &self,
        cfg: &ServerConfig,
        name: &str,
        args: serde_json::Value,
    ) -> Result<ToolResult> {
        self.client(cfg).await?.call_tool(name, args).await
    }

    /// Drop a server's connection (reaps a stdio child once no request holds it).
    /// Drop every pooled connection for a server id. One id can now hold
    /// several entries (one per distinct header/env set), so this removes the
    /// whole family rather than a single key - a caller asking to disconnect
    /// "github" means all of it, not the variant it happens to hold.
    pub async fn disconnect(&self, id: &str) {
        self.clients
            .lock()
            .await
            .retain(|k, _| !key_belongs_to(k, id));
    }

    /// Drop every connection (server shutdown).
    pub async fn shutdown(&self) {
        self.clients.lock().await.clear();
    }
}
pub mod clock;
pub mod loop_budget;
pub mod tool_search;

#[cfg(test)]
mod tests {
    use super::*;

    fn http(id: &str, url: &str, headers: &[(&str, &str)]) -> ServerConfig {
        ServerConfig {
            id: id.into(),
            label: "l".into(),
            transport: Transport::Http {
                url: url.into(),
                headers: headers
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            },
        }
    }

    /// The leak this guards: same label, different per-request header. These
    /// must not share a pooled connection, or the first caller's header serves
    /// every later one.
    #[test]
    fn differing_headers_do_not_share_a_connection() {
        let a = http(
            "inline:artifacts",
            "http://x/mcp",
            &[("x-paddock-conversation", "chat-a")],
        );
        let b = http(
            "inline:artifacts",
            "http://x/mcp",
            &[("x-paddock-conversation", "chat-b")],
        );
        assert_ne!(pool_key(&a), pool_key(&b));
    }

    #[test]
    fn identical_configs_do_share_one() {
        let a = http(
            "inline:artifacts",
            "http://x/mcp",
            &[("h", "1"), ("g", "2")],
        );
        // Same pairs, inserted the other way round: a HashMap may iterate them
        // in either order, and an order-sensitive hash would miss the pool.
        let b = http(
            "inline:artifacts",
            "http://x/mcp",
            &[("g", "2"), ("h", "1")],
        );
        assert_eq!(pool_key(&a), pool_key(&b));
    }

    #[test]
    fn a_different_url_is_a_different_connection() {
        let a = http("inline:s", "http://x/mcp", &[]);
        let b = http("inline:s", "http://y/mcp", &[]);
        assert_ne!(pool_key(&a), pool_key(&b));
    }

    #[tokio::test]
    async fn a_stdio_server_may_not_steer_the_loader() {
        for name in [
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "ld_audit",
            "DYLD_INSERT_LIBRARIES",
        ] {
            let cfg = ServerConfig {
                id: "registry:s".into(),
                label: "s".into(),
                transport: Transport::Stdio {
                    command: "true".into(),
                    args: Vec::new(),
                    env: [(name.to_string(), "x".to_string())].into(),
                },
            };
            let err = McpClient::connect(&cfg).await.err().expect("refused");
            assert!(err.to_string().contains("is refused"), "{name}: {err}");
        }
        // an ordinary tool variable is not caught by it
        assert!(!steers_loader("API_TOKEN") && !steers_loader("OLD_PATH"));
    }

    #[test]
    fn disconnect_sweeps_every_variant_of_an_id() {
        let a = pool_key(&http("inline:s", "http://x/mcp", &[("c", "a")]));
        let b = pool_key(&http("inline:s", "http://x/mcp", &[("c", "b")]));
        assert!(key_belongs_to(&a, "inline:s") && key_belongs_to(&b, "inline:s"));
        // A id that merely shares a prefix must not be swept with it.
        let other = pool_key(&http("inline:svc", "http://x/mcp", &[]));
        assert!(
            !key_belongs_to(&other, "inline:s"),
            "prefix collision: {other}"
        );
    }
}
