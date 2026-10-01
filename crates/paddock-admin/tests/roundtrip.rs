//! End-to-end over the real local transport (named pipe on Windows, UDS on
//! Unix): server + client + enumeration. This is the same pair the runner and
//! manager use, so a green run here proves the wire, the security attributes,
//! and the busy-retry path all hold on this OS.

use axum::routing::{get, post};
use axum::{Json, Router};
use paddock_admin::client::AdminClient;
use paddock_admin::types::{DrainState, Identify, WIRE_VERSION};

fn test_router(port: u16) -> Router {
    Router::new()
        .route(
            "/v1/identify",
            get(move || async move {
                Json(Identify {
                    wire: WIRE_VERSION,
                    role: "runner".into(),
                    version: "0.0.0-test".into(),
                    pid: std::process::id(),
                    port,
                    model: Some("test-model".into()),
                    embedder: None,
                    asr: None,
                    aligner: None,
                    image: None,
                    reader: None,
                    tabular: None,
                    started_at_unix: 0,
                    instance_id: "itest-instance".into(),
                    capabilities: vec!["stats".into()],
                    // the field is Option on the wire for runner/manager skew,
                    // and this fixture is a bare runner that reports no
                    // speculation - None is the honest value, not a placeholder
                    spec: None,
                })
            }),
        )
        .route(
            "/v1/drain",
            post(|| async {
                Json(DrainState {
                    draining: true,
                    in_flight: 0,
                    drained: true,
                    timed_out: false,
                })
            }),
        )
}

#[tokio::test]
async fn identify_and_drain_roundtrip_over_local_transport() {
    // A port unlikely to collide with anything real; only the endpoint NAME is
    // derived from it - no TCP is bound anywhere in this test.
    let port: u16 = 45911;
    tokio::spawn(paddock_admin::server::serve(port, test_router(port)));
    // Give the endpoint a moment to exist.
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    let client = AdminClient::new(port);
    let id = client.identify().await.expect("identify over the pipe");
    assert_eq!(id.wire, WIRE_VERSION);
    assert_eq!(id.role, "runner");
    assert_eq!(id.port, port);
    assert_eq!(id.model.as_deref(), Some("test-model"));
    assert_eq!(id.instance_id, "itest-instance");
    assert!(id.capabilities.iter().any(|c| c == "stats"));

    let drain = client.drain(Some(1000)).await.expect("drain call");
    assert!(drain.draining && drain.drained);

    // The reconciliation input (§6.1): this endpoint is discoverable.
    assert!(
        paddock_admin::enumerate().contains(&port),
        "enumerate() must list the live endpoint"
    );

    // A second sequential connection works (fresh instance stood up after the
    // first connect - the busy-retry path).
    let again = client.identify().await.expect("second connection");
    assert_eq!(again.pid, std::process::id());
}

#[tokio::test]
async fn connect_to_absent_endpoint_is_a_clean_error() {
    let client = AdminClient::new(45912); // nothing serves this
    let err = client.identify().await.expect_err("must fail");
    assert!(
        matches!(
            err,
            paddock_admin::client::AdminError::Connect { port: 45912, .. }
        ),
        "got {err:?}"
    );
}
