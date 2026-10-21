//! Completion webhooks (a paddock extension: both vendors are poll-first, and
//! OpenAI's own webhooks are account-level). A batch created with
//! `paddock.webhook_url` gets one POST when it reaches a terminal state, in
//! OpenAI's event shape:
//! `{object: "event", id, type: "batch.completed" | "batch.failed" |
//! "batch.expired" | "batch.cancelled", created_at, data: {id}}`.
//!
//! Delivery state lives in the batch row, so a manager restart resumes the
//! retries. Retries back off from 5 s to an hour and stop 24 h after the
//! batch ended. The event id is derived from the batch id and so is stable
//! across retries: a receiver can de-duplicate on it.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;

use super::{Batches, now_ms};

/// Webhook states in the batch row.
const PENDING: i64 = 1;
const DELIVERED: i64 = 2;
const ABANDONED: i64 = 3;

const GIVE_UP_MS: i64 = 24 * 3_600_000;

pub(crate) async fn run(b: Arc<Batches>) {
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap_or_default();
    loop {
        tokio::time::sleep(b.tuning.tick.max(Duration::from_millis(200)) * 5).await;
        // housekeeping rides the same clock: pending uploads past their day
        super::uploads::sweep(&b);
        let rows = match b.db.batch_open() {
            Ok(r) => r,
            Err(_) => continue,
        };
        let now = now_ms();
        for r in rows {
            if r.webhook_state != PENDING || !r.is_terminal() || r.webhook_next_ms > now {
                continue;
            }
            let Some(url) = r.options["webhook_url"].as_str() else {
                let _ =
                    b.db.batch_webhook_set(&r.id, ABANDONED, r.webhook_attempts, 0);
                continue;
            };
            let ended = [r.completed_ms, r.cancelled_ms, r.expired_ms, r.failed_ms]
                .into_iter()
                .flatten()
                .next()
                .unwrap_or(now);
            let hex = r.id.rsplit('_').next().unwrap_or(&r.id);
            let event = json!({
                "object": "event",
                "id": format!("evt_{hex}"),
                "type": format!("batch.{}", r.status),
                "created_at": ended / 1000,
                "data": {"id": r.id},
            });
            let ok = http
                .post(url)
                .json(&event)
                .send()
                .await
                .is_ok_and(|res| res.status().is_success());
            let attempts = r.webhook_attempts + 1;
            if ok {
                let _ = b.db.batch_webhook_set(&r.id, DELIVERED, attempts, 0);
                tracing::info!(batch = %r.id, "batch: webhook delivered");
            } else if now - ended > GIVE_UP_MS {
                let _ = b.db.batch_webhook_set(&r.id, ABANDONED, attempts, 0);
                tracing::warn!(batch = %r.id, attempts, "batch: webhook abandoned after 24 h");
            } else {
                let delay = (5_000i64 << attempts.min(10)).min(3_600_000);
                let _ =
                    b.db.batch_webhook_set(&r.id, PENDING, attempts, now + delay);
            }
        }
    }
}
