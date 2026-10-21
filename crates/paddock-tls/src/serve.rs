//! One port, both schemes.
//!
//! The obvious design - https on the configured port, http on another - breaks
//! every bookmark, every `--endpoint` flag and every health check the moment
//! TLS is switched on, and leaves a user who typed `http://` looking at either
//! a dead port or a screenful of binary. So instead the listener reads the
//! **first byte** of each connection before deciding anything. A TLS
//! ClientHello always begins with `0x16` (the TLS record type for `handshake`);
//! an HTTP request always begins with a method's first letter. One byte
//! separates them with no ambiguity.
//!
//! Given that, the routing rule writes itself:
//!
//! | first byte | peer | what happens |
//! |---|---|---|
//! | `0x16` | anyone | TLS, then the app |
//! | else | loopback | plain HTTP, exactly as before TLS existed |
//! | else | remote | `301` to the https URL |
//!
//! The loopback row is load-bearing. The CLI reaches the manager over
//! `http://127.0.0.1`, `healthz` is polled the same way, and the Studio hands
//! runners a plain `http://127.0.0.1:<port>/api/mcp/artifacts` callback URL for
//! its own MCP server. All of that keeps working untouched, which is what
//! makes turning TLS on by default safe rather than a migration.
//!
//! `ROOT_PATH` is the one exception to the redirect: the root certificate has
//! to be fetchable *before* the client trusts anything, so redirecting it into
//! a warning page would be a circular argument.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::ConnectInfo;
use hyper::body::Incoming;
use hyper_util::rt::TokioIo;
use tokio::net::{TcpListener, TcpStream};
use tower_service::Service;

/// Fetchable over plain http from anywhere, redirect or no redirect.
pub const ROOT_PATH: &str = "/tls/root.crt";

/// The TLS record type for `handshake`, and the first byte of every
/// ClientHello since SSL 3.0.
const TLS_HANDSHAKE: u8 = 0x16;

/// How long a connection may sit having sent nothing. Without this an idle
/// opener - a port scanner, a half-open proxy probe - parks a task forever.
const SNIFF_TIMEOUT: Duration = Duration::from_secs(15);

/// Serve `app` on `listener`, terminating TLS for clients that ask for it.
///
/// `tls` is optional so a box whose identity could not be established still
/// serves - degraded to cleartext, having said so - rather than not at all.
pub async fn serve(
    listener: TcpListener,
    app: Router,
    tls: Option<Arc<rustls::ServerConfig>>,
    port: u16,
) -> std::io::Result<()> {
    let acceptor = tls.map(tokio_rustls::TlsAcceptor::from);
    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(pair) => pair,
            // A single failed accept is not a reason to stop serving; the
            // usual cause is a client that vanished between SYN and accept.
            Err(e) => {
                tracing::debug!(%e, "accept failed");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let app = app.clone();
        let acceptor = acceptor.clone();
        tokio::spawn(async move {
            if let Err(e) = handle(stream, peer, app, acceptor, port).await {
                tracing::trace!(%peer, %e, "connection ended");
            }
        });
    }
}

async fn handle(
    stream: TcpStream,
    peer: SocketAddr,
    app: Router,
    acceptor: Option<tokio_rustls::TlsAcceptor>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut first = [0u8; 1];
    let n = match tokio::time::timeout(SNIFF_TIMEOUT, stream.peek(&mut first)).await {
        Ok(Ok(n)) => n,
        // Timed out or failed before saying anything - nothing to serve.
        _ => return Ok(()),
    };
    let spoke_tls = n == 1 && first[0] == TLS_HANDSHAKE;

    match (spoke_tls, acceptor) {
        (true, Some(acceptor)) => {
            let tls = acceptor.accept(stream).await?;
            http1(TokioIo::new(tls), app_service(app, peer, true)).await
        }
        // A TLS handshake with nothing to answer it. Close: there is no error
        // we could write that a TLS client would be able to read, and feeding
        // a ClientHello to an HTTP parser only turns silence into noise.
        (true, None) => Ok(()),
        // Cleartext from the box itself - or from anyone at all when there is
        // no identity to offer. Serve it exactly as before TLS existed.
        (false, None) => http1(TokioIo::new(stream), app_service(app, peer, false)).await,
        (false, Some(_)) if peer.ip().is_loopback() => {
            http1(TokioIo::new(stream), app_service(app, peer, false)).await
        }
        // Cleartext from the network, with https available: send them there.
        (false, Some(_)) => http1(TokioIo::new(stream), redirect_service(app, peer, port)).await,
    }
}

async fn http1<I, S>(io: I, svc: S) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    I: hyper::rt::Read + hyper::rt::Write + Unpin + Send + 'static,
    S: hyper::service::Service<
            hyper::Request<Incoming>,
            Response = hyper::Response<axum::body::Body>,
            Error = std::convert::Infallible,
        > + Send,
    S::Future: Send,
{
    // **`with_upgrades` is not optional.** `/api/gpu/stream` and the realtime
    // transcription relay are WebSockets, and hyper only hands the upgraded
    // connection back to the handler when the connection was served this way.
    // Without it both sockets negotiate a 101 and then hang forever.
    hyper::server::conn::http1::Builder::new()
        .serve_connection(io, svc)
        .with_upgrades()
        .await
        .map_err(Into::into)
}

/// Whether a request arrived over TLS, for the handlers that must hand back an
/// absolute URL to this server (a batch's `results_url`): one port serves
/// both schemes, so nothing else in the request says which it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Secure(pub bool);

/// The app, with the peer address attached.
///
/// Re-inserting `ConnectInfo` by hand is what `axum::serve(..)
/// .into_make_service_with_connect_info()` would have done. It is not a
/// nicety: the manager's auth middleware exempts loopback peers, so a request
/// arriving without this looks remote, and the box's own Studio starts being
/// asked for an API key it was never shown.
fn app_service(
    app: Router,
    peer: SocketAddr,
    secure: bool,
) -> impl hyper::service::Service<
    hyper::Request<Incoming>,
    Response = hyper::Response<axum::body::Body>,
    Error = std::convert::Infallible,
    Future: Send,
> + Send {
    hyper::service::service_fn(move |mut req: hyper::Request<Incoming>| {
        req.extensions_mut().insert(ConnectInfo(peer));
        req.extensions_mut().insert(Secure(secure));
        app.clone().call(req)
    })
}

/// Answers a cleartext request from the network with a permanent redirect to
/// the same path over https - except for the root certificate, which has to
/// stay reachable before any trust exists.
fn redirect_service(
    app: Router,
    peer: SocketAddr,
    port: u16,
) -> impl hyper::service::Service<
    hyper::Request<Incoming>,
    Response = hyper::Response<axum::body::Body>,
    Error = std::convert::Infallible,
    Future: Send,
> + Send {
    use axum::http::{HeaderValue, StatusCode, header};

    hyper::service::service_fn(move |mut req: hyper::Request<Incoming>| {
        let app = app.clone();
        async move {
            if req.uri().path() == ROOT_PATH {
                req.extensions_mut().insert(ConnectInfo(peer));
                return app.clone().call(req).await;
            }
            // Send them back to the authority they ASKED for, not the one we
            // happen to be bound to. They typed a name that resolves here, and
            // substituting an address of our own would hand them a certificate
            // for a name they never asked about. Keeping the client's port
            // matters for the same reason: behind a port-forward the outside
            // port is not `port`, and rewriting it would redirect to a door
            // that is not open. Our own port fills in only when Host carried
            // none (a client that used the default 80).
            //
            // `rsplit_once` is correct for `[::1]:11555` - the bracket form is
            // the only way a Host header may carry a v6 literal.
            let authority = req
                .headers()
                .get(header::HOST)
                .and_then(|v| v.to_str().ok())
                .filter(|h| !h.is_empty())
                .map(|h| {
                    if h.rsplit_once(':')
                        .is_some_and(|(_, p)| p.parse::<u16>().is_ok())
                    {
                        h.to_owned()
                    } else {
                        format!("{h}:{port}")
                    }
                })
                .unwrap_or_else(|| format!("{}:{port}", req.uri().host().unwrap_or("localhost")));
            let tail = req
                .uri()
                .path_and_query()
                .map(|p| p.as_str())
                .unwrap_or("/");
            let target = format!("https://{authority}{tail}");

            let mut res = axum::http::Response::new(axum::body::Body::from(format!(
                "This address is served over https.\n{target}\n"
            )));
            *res.status_mut() = StatusCode::MOVED_PERMANENTLY;
            if let Ok(v) = HeaderValue::from_str(&target) {
                res.headers_mut().insert(header::LOCATION, v);
            }
            res.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/plain; charset=utf-8"),
            );
            // Deliberately no Strict-Transport-Security. HSTS is ignored on an
            // IP origin anyway, and on a hostname it would pin the browser to
            // https for this host - outliving any later decision to
            // reconfigure, and unresettable from our side.
            Ok(res)
        }
    })
}
